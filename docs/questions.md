# Clarification Questions: Boreal / Aether Integration & Next Optimization Wave

These questions target the highest-leverage design decisions for turning `marray-mask` from a standalone Mojo morsel VM into a drop-in orchestration and compute replacement under `boreal-io`, `boreal-postproc`, `boreal-stat`, `aether`, `polaris`, and `mogc-api`.

**Answers below** are point-in-time conclusions from current boreal-io / postproc / stat / aether / mogc-api code and [`docs/boreal/`](boreal/README.md) contracts. No operational filesystem paths.

---

## 1. `boreal-io` (`cmc_io_fst_hdr` / `cmc_io_fst_data`) & `RecordCatalog` Boundary

1. **Direct Mojo import vs. C-ABI `ChunkDecoder` boundary:**
   Can `marray-mask` import `cmc_io_fst_data` (`unpack_many_f32_into` / `unpack_*_into`) directly as a Mojo package, or are `boreal-io` and `marray-mask` deployed as separate shared libraries that should connect via a C-ABI function-pointer table?

   **Answer:** Treat them as **separate shared libraries** in production. Prefer a stable **C-ABI `ChunkDecoder` / unpack-into table** (conda-pinned boreal-io `.so` + marray `.so`). Same-repo Mojo `-I src/mojo` import of `cmc_io_fst` is fine for monorepo experiments only—not the deploy contract. `cmc_io_fst_data` is already a Python extension module wrapping those codecs; marray should not depend on importing that Python module from inside the morsel VM.

2. **Thread safety & GIL independence of `cmc_io_fst_data`:**
   `02-fst-decode-pipeline.md` notes a process-local `np.memmap` LRU in Python. Is the underlying Mojo unpacker pure `(src_bytes_ptr, lng, datyp, nbits, dst_ptr)`—100% safe to invoke concurrently from Mojo `max.algorithm.parallelize` worker threads—or does any datyp (`6`, `130`, `133`, `134`) fall back to Python or hold shared mutable state?

   **Answer:** The Mojo batch path (`unpack_*_from_raw` / `unpack_jobs_f32_parallel` in `cmc_io_fst` unpack_batch) is **read-only raw bytes → distinct dest buffers** and parallelizes over jobs—safe when destinations do not overlap. The **`np.memmap` LRU is Python-side** (`boreal.io.xarray.decode`), not inside the codec. Product Mojo covers `0,1,2,4,5,130,133,134`. **`datyp=6` can still hit Python/librmn-style packers**—do not assume every datyp is GIL-free or lock-free under the Python extension entry points.

3. **Native `@@` / `!@` bitmap output format (`datyp=2, nbits=1`):**
   Today, does `cmc_io_fst_data` unpack companion `@@` bitmap records into `float32` (`0.0`/`1.0`), `uint8`, or raw 1-bit words?
   * *Why it matters:* If `cmc_io_fst_data` can expose 1-bit packed `UInt64` words directly, `morsel.mojo` converts `fstd2nc` polarity (`keep > 0, mask == 0`) in **1 CPU instruction per 64 cells (`~word`)**—32× less memory traffic than unpacking `@@` into `float32` and running `np.where`.

   **Answer:** Today **`float32` planes**. `datyp=2` goes through `compact_integer_unpack_f32` into a float destination; write/mask helpers also build **float32 0/1 bitmaps**. Callers do **not** receive packed `UInt64` words. marray should convert float/`uint8` bitmaps into `MSTATE_BITS` on ingest (or add a future packed-bit unpack entry in boreal-io).

4. **Realistic `datyp=134` (~76% catalog share) decode cost & sub-slab behavior:**
   Is `armn_uncompress` (`datyp=134`) strictly whole-record (must decode the full `(ni, nj)` plane before regional slicing `[i0:i1, j0:j1]`), and what is its typical decode time per `(ni, nj)` plane at GDPS `U`/`Z`/`O` scales?
   * *Why it matters:* Knowing the unpack-to-math ratio lets us tune morsel/slab prefetching and add a realistic decode-cost model to `bench_boreal_realistic.py`.

   **Answer:** **Whole-record.** `unpack_134_from_raw` uses `nelm = ni * nj * nk`; regional `[i0:i1, j0:j1]` means decode the full plane then slice. Do not model sub-rectangle decompress inside the compressor. For cost, use boreal-io published warm **column_compute** peers (order **~10² ms** for multi-level GDPS press column open/compute in `docs/formats/fst/benchmarks.md`) and measure **per-plane** in `bench_boreal_realistic` rather than inventing µs—prefetch whole planes.

5. **Cross-cube variable pairing across `list[xr.Dataset]`:**
   In operational `open_fst` calls, how frequently do downstream plugins pair variables that landed in *different* Datasets in the returned list (e.g., 2D surface `P0` or `TM` paired with 3D hybrid `TT`/`HU`, or ragged `split` time blocks)?

   **Answer:** **Common.** `split_dataset_groups` keys include **`vctype`** (plus grid/`ni`/`nj`/`grtyp` and ragged `ip1_set`/`time_set`)—surface/`P0` vs hybrid 3D routinely become **separate Datasets**. Plugins already accept `list[Dataset]` (`from_sources`, `pick_cube_for_mode`). `RecordCatalog` must support **cross-group joins** by nomvar/grid/time, not only single-Dataset algebra.

---

## 2. `boreal-postproc` & `boreal-stat` Compute Contracts

6. **Vertical column-scan memory layout (`CAPE`, `LCL`, parcel, wet-bulb):**
   For vertical plugins in `cmc_postproc_kernels`, do the kernels currently iterate strided over `(n_z, nj, ni)` C-order cubes (jumping `nj * ni` floats per level), or do they transpose to `(nj, ni, n_z)` first?
   * *Why it matters:* Streaming **64-column horizontal tiles across all `n_z` levels** (`n_z × 64 × 4 B ≈ 10–20 KB`, 100% L1d-resident) vectorizes vertical physics across 64 columns in AVX2/AVX-512 with zero full-cube transpose.

   **Answer:** Mojo kernels (`lift_columns`, `vertical_scan_columns`) operate on **flattened columns** (`n_pts` tasks, vertical extent `nk`, column index `i`) after Python peels/transposes cubes to trailing `(y, x)` and packs columns—**not** a raw C `(n_z, nj, ni)` strided walk inside Mojo. marray should offer a **column-tile vertical schedule** matching `(column, level)` access, not assume cube-order morsels alone.

7. **`boreal-stat` `float64` precision requirement:**
   `08-peel-kernel-rewrap.md` notes `boreal-stat` peels to `values_f64` for `ke`, `zonavg`, `globavg`, `stdev`, and pair `RMSE`. Are `Float32` morsel intermediates with `Float64` reduction accumulators (current `morsel.mojo` behavior) within acceptance tolerances, or do specific `r.diag` / pair-metric parity gates require full `Float64` morsel registers?

   **Answer:** Today **`values_f64` always** (`np.ascontiguousarray(..., dtype=float64)`). Keep **f64 reduction accumulators**. f32 morsel math is fine for many elementwise postproc paths if parity atol matches, but **r.diag / pair RMSE gates should assume an f64 payload path** until measured otherwise.

8. **Multi-output / multi-reduction DAG frequency:**
   How common are single-pass multi-sink queries in `boreal-stat` and `boreal-postproc` (e.g., computing `globavg + zonavg + timavg` together on `ke`, or emitting `WindModulus + WindDirection` together from `UU, VV`)?
   * *Why it matters:* Multi-sink fusion cuts Bench 4 (`ke` `globavg` + `zonavg`) in half (`84.9 ms → ~42 ms`) by evaluating `ke` once instead of twice.

   **Answer:** The **public API is mostly one sink per call** (`WindModulus` vs combined `WindModulusAndDirection`; `ke` then separate `globavg` / `zonavg`). Real pipelines often compose multiple sinks—**multi-sink DAG fusion is high value** even when Spooki classes are single-output.

9. **Weighted reduction targets beyond `globavg`:**
   Do `boreal-stat` `zonavg` / `meriavg` / `timavg` also need weighted axis reductions (`sum(X * w) / sum(w)` per `(time, vertical)` or per `(vertical, y)`), or is `areacell` weighting primarily used for 2D horizontal/global reductions?

   **Answer:** **`areacell` / `areacell_yy` are primarily for `globavg`.** `zonavg` is an unweighted mean over longitude; `meriavg` accepts optional weights; `timavg` is unweighted. Prioritize weighted **horizontal/global** reductions first.

---

## 3. `aether`, `polaris` & `mogc-api` Zero-Copy Handoff

10. **`aether` CSR `apply_weights` buffer contract:**
    What exact input buffer interface does `aether`'s Mojo CSR apply prefer (C-contiguous `float32` NumPy array, raw pointer/Span, DLPack, or MAX `Buffer`), and does it batch multiple 2D fields `(n_fields, nj * ni)` in a single SpMV call?

    **Answer:** Production path is `apply_weights_csr(flat, indptr, indices, data, extrap_value)` with **`flat` shape `(n_fields, n_src)`**, C-contiguous NumPy; Mojo **promotes apply to float64** and batches large `n_fields` via `field_chunk`. Prefer contiguous buffers / future Span for marray handoff—**DLPack is not required today**.

11. **`aether` missing-value handling:**
    Does `aether`'s CSR SpMV expect `NaN` in-band for masked source cells (and re-normalize weights over non-NaN neighbors), or could it eventually consume a `MSTATE_BITS` 64-bit validity bitmask directly?

    **Answer:** Today SpMV uses **`extrap_value` for empty CSR rows**; there is **no bitmask API** and no NaN-renormalize of neighbors. In-band **NaN in sources will poison** weighted sums unless pre-zeroed/masked. Handoff as **pre-masked finite f32/f64 planes** for now; bitmask-aware SpMV would be a new aether feature.

12. **`polaris` / `mogc-api` concurrency & p95 latency profile:**
    In `mogc-api` (`GetMap`) and `polaris`, what percentage of p95 request time is currently spent in `open_fst` / peel vs. `aether` remap vs. `mojoviz` LUT/PNG, and how many concurrent HTTP worker threads hit the same open archive simultaneously?
    * *Why it matters:* Under multi-request concurrency, eliminating Python GIL acquisition during `open_fst` + slice + derived-field compute compounds the single-request `8×–37×` speedup even further.

    **Answer:** mogc documents **cold `open_fst` ~0.3–1 s** (dominates first hit) vs **warm GetMap ~17 ms** (FIFO PNG/field cache). Serve path is typically **one uvicorn worker**; warmup / point-series use a **thread pool (≤8)** sharing an in-process cache. GIL-free open/slice helps **parallel warmup and cold multi-lead** more than single-worker warm PNG hits. Exact p95 stage splits belong in `X-MOGC-Profile` / `/metrics/profile` on a live deploy—not hard-coded here.

---

## 4. Packaging, Duck-Array UX & Streaming Sinks

13. **Phase A Duck Array (`xarray.BackendEntrypoint` + `__array_namespace__`) priority:**
    Should the next code milestone focus on **M1 (`MmapRecordCatalog` + `ChunkDecoder` wired to `cmc_io_fst_data`)** or **M4 (Phase A Xarray duck array wrapping `marray_xr` inside real `xr.Dataset`)** so `aether` and `boreal-postproc` can test it immediately?

    **Answer:** **M1 first** (`RecordCatalog` + `ChunkDecoder` → boreal unpack)—kills HLG / decode boundary. Ship **M4 (Phase A duck array) immediately after** so aether/postproc can consume without rewriting plugins. Do not start with M4 alone.

14. **Streaming write-out priority (`write_fst` vs. `mozarr`):**
    For large derived archives where full `.compute()` OOMs, which streaming sink is the higher immediate priority: chunked FST write-out (`fst_mask_fill=1e30`) or streaming Zarr V3 encode (`mozarr`)?

    **Answer:** **Chunked `write_fst`** first (attrs stamp + `fst_mask_fill=1e30`) for the boreal round-trip / postproc path. **mozarr** next for cloud / PB archives.

15. **Coordinate types for `.sel()` parity:**
    Which coordinate lookup patterns appear most in `boreal` consumers: `isel` by integer index, `sel` by `datetime64[ns]` validity time / forecast lead hours, or `sel` by decoded `vertical` pressure/hybrid level (`ip1`)?

    **Answer:** Serving apps (**polaris / mogc**) lean **`isel` / lead-path filters + nearest vertical**. Science plugins lean **decoded `vertical` values**. CF **`datetime64` time** exists on cubes but is secondary to lead/index selection on the hot serve path. Implement all three; optimize **`isel` + vertical nearest** first.
