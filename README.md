# `marray-mask` — General Mojo 1.2 Masked Lazy Array & Xarray/Dask Engine

A high-performance **Mojo 1.2** masked array, whole-record `ChunkDecoder` catalog, and lazy bytecode execution stack designed to replace Python's `NumPy.ma + Dask + Xarray` chunk-orchestration bottleneck on large ($10\text{ GB} \to 1\text{ PB+}$) scientific/NWP datasets across the **boreal** (`boreal-io`, `boreal-postproc`, `boreal-stat`, `polaris`, `mogc-api`) and **aether** ecosystem.

📖 **Deep-Dive Documentation:**
* **[`docs/ARCHITECTURE_AND_ROADMAP.md`](docs/ARCHITECTURE_AND_ROADMAP.md)** — Full architectural design, C-ABI `ChunkDecoder` & M4 duck-array contracts, roofline math, B1–B8 Boreal/Aether benchmarks, Mojo 1.2 subtleties, and remaining roadmap.
* **[`docs/boreal/`](docs/boreal/README.md)** & **[`docs/questions.md`](docs/questions.md)** — Cross-package contracts and answered engineering decisions for `boreal-io`, `boreal-postproc`, `boreal-stat`, `aether`, `polaris`, and `mogc-api`.

---

## Highlights & Realistic Boreal / Aether Benchmarks (Suites B1–B8)

Verified with `assert_allclose` (including `< 1e-12` relative IEEE-754 `float64` parity on `boreal-stat` `r.diag` / `RMSE`) on Ryzen 7 5800X (8C/16T) across C-contiguous `(time, vertical, y, x)` cubes, `fstd2nc`-polarity `float32` `0/1` `@@` bitmaps (`keep > 0, mask == 0`), whole-record `datyp=134` compressed planes, and all 7 Tier-1 RPN `grtyp` families (`U, O, Z, G, N, L, M`):

| Realistic Boreal / Aether Suite ([`bench_boreal_realistic.py`](benchmarks/bench_boreal_realistic.py)) | `record` HLG | `column` HLG (16T) | Eager Peel / Transpose | `marray_xr` (Mojo 1.2) | Speedup |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **B1:** `open_fst` Assembly (80 leads × 40 lvls × 6 vars = 19,200 recs) | `36.22 ms` | `1.79 ms` | — | **`0.017 ms`** (`0.53 ms` M4 `xr.Dataset`) | **`2,144×` vs `record`** (`106×` vs `column`) |
| **B2:** `polaris` Thin 2D Peel (`isel` + nearest `vertical` → `(nj,ni)` buf) | `7.82 ms` | `32.12 ms` | — | **`0.772 ms`** (`2.38 ms` M4 `xr.Dataset`) | **`10.1×` vs `record`** (**`41.6×` vs `column`**) |
| **B3:** `boreal-postproc` `Humidex` (`float32` `@@`) + `stream_write_fst` (`1e30`) | — | `428.58 ms` | `689.65 ms` | **`76.48 ms`** (`89.0 ms` streaming sink) | **`5.6×` vs `column`** (**`9.0×` vs Eager**) |
| **B4:** `boreal-stat` `r.diag` (`ke` `globavg` + `zonavg`, single-pass `compute_many`) | — | `457.80 ms` | `1,195.35 ms` (`values_f64`) | **`61.63 ms`** (`119.6 ms` exact `float64`) | **`7.4×` vs `column`** (**`10.0×` exact `f64`**) |
| **B5:** Tier-1 `grtyp` Sweep (`U, O, Z, G, N, L, M`, 99% catalog share) | `142–258 ms` | `131–281 ms` | `171–344 ms` | **`9.62–18.98 ms`** | **`12.8×–15.8×` vs `column` HLG** |
| **B6:** M1 C-ABI `ChunkDecoder` (`datyp=134` + `datyp=2` `@@`, cross-`vctype` + regional) | — | `523.49 ms` (`79.0 ms` reg) | — | **`193.08 ms`** (**`23.79 ms`** regional) | **`2.7×–3.3×` faster + `~21 MB` RAM cap** |
| **B7:** `boreal-postproc` Vertical Column Scan (`TILE_COLS=64` vs 4D transpose) | — | — | `1,088.62 ms` (`419 MB` copy) | **`116.14 ms`** | **`9.4×` faster** (zero 4D transpose) |
| **B8:** `aether` Batched `(4, 327240)` Finite Prep + `mogc-api` 8-Thread Warmup | — | `20.21 ms` / `132.1 ms` | — | **`0.547 ms`** / **`6.68 ms`** | **`36.9×` `aether` prep** / **`19.8×` 8T warmup** |

*(On raw quantized `uint16` full-cube benchmarks in [`bench_mojo_vs_dask_xarray.py`](benchmarks/bench_mojo_vs_dask_xarray.py), single-pass decode + L1 morsel fusion reaches **`49.89 ms` (`28.9 GB/s` physical DRAM read)** vs `5,025 ms` in Dask [`100.7×`] and `7,863 ms` in `numpy.ma` [`157.6×`].)*

---

## Quick Start (Linux / macOS / WSL2)

```bash
git clone git@github.com:itsgifnotjiff/marray-mask.git
cd marray-mask

pixi install
pixi run test            # 6 Mojo std.testing.TestSuite unit tests (incl. catalog & vertical tiles)
pixi run synth-bench     # Standalone Mojo Tier-1 grtyp (U/O/Z/L/G/N/M) fill+decode demo
pixi run bench-boreal    # Realistic Boreal/Aether B1-B8 suite (M1 catalog, M4 duck array, aether, stat)
pixi run bench-vs-dask   # General 4-way Mojo VM vs NumPy.ma vs Xarray vs Dask benchmark
```

## Example Usage: M1 `RecordCatalog`, M4 `xr.Dataset` Duck Array & `aether` Handoff

```python
import numpy as np
import marray_xr as mxr

# 1. Open packed FST archive into `list[mxr.Dataset]` split by `vctype` (hybrid + surface)
ds_list = mxr.open_fst_catalog(archive)  # zero payload unpack at open time!
tt = mxr.from_sources(ds_list, "TT")
tm = mxr.from_sources(ds_list, "TM")     # 3D surface auto-broadcasts against 4D hybrid!

# 2. M4 Drop-in `xarray.Dataset` (`MarrayDuckArray`) for unmodified downstream plugins:
xr_datasets = mxr.open_fst_xarray(archive)
uv_2d = np.hypot(xr_datasets[0]["WindU"].isel(time=3), xr_datasets[0]["WindV"].isel(time=3))

# 3. Single-pass multi-sink fusion (`globavg` + `zonavg` in ONE scan) & `aether` finite flat buffer:
ke = 0.5 * ((ds_list[0].find_by_nomvar("UU") * 0.514444) ** 2 + (ds_list[0].find_by_nomvar("VV") * 0.514444) ** 2)
glob_ke, zon_ke = mxr.compute_many(
    ke.lazy_weighted_mean(ds_list[0].cell_measures["areacell"]),
    ke.lazy_mean(("time", "x")),
)
flat_for_aether = mxr.to_aether_flat([tt.isel(time=0, vertical=0), tm.isel(time=0)], masked_fill=0.0)
```

## Repository Structure

* [`docs/ARCHITECTURE_AND_ROADMAP.md`](docs/ARCHITECTURE_AND_ROADMAP.md) — Architecture spec, C-ABI contracts, optimizations, Mojo 1.2 lessons, and roadmap
* [`src/marray/morsel.mojo`](src/marray/morsel.mojo) — 64-byte aligned `MorselReg` (`Float32`) & `MorselReg64` (`Float64`), `float32` `@@` bitmaps, 4-way unrolled SIMD kernels
* [`src/marray/catalog.mojo`](src/marray/catalog.mojo) — M1 C-ABI `ChunkDecoder`, `is_datyp_parallel_safe` (`datyp=6` guard), `PlaneSlotCache`, and reference `datyp=134`/`datyp=2` codecs
* [`src/marray/vertical.mojo`](src/marray/vertical.mojo) — L1-resident `TILE_COLS = 64` `(column, level)` vertical physics scheduler (`cmc_postproc_kernels` contract)
* [`src/marray/vm.mojo`](src/marray/vm.mojo) — 24-opcode bytecode VM, 3-pass multi-sink optimizer, spatial-partitioned axis reductions, and `Float64` parity engine
* [`src/marray_mojo.mojo`](src/marray_mojo.mojo) — `PyInit_marray_mojo` (`PythonModuleBuilder`) + zero-GIL C-ABI shared library exports
* [`python/marray_xr/`](python/marray_xr/__init__.py) — `Dataset`/`DataArray`, M1 [`catalog.py`](python/marray_xr/catalog.py), M4 [`duck.py`](python/marray_xr/duck.py) (`MarrayDuckArray`), `compute_many`, `to_aether_flat`, `stream_write_fst`
* [`.agents/skills/`](.agents/skills/) — Vendored official Modular Mojo & MAX skills (`modular/skills`)
