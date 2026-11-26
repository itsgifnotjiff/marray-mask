# `marray-mask`: Technical Architecture, Optimizations, Subtleties & Roadmap

> **Goal:** Replace the `Python + NumPy.ma + Dask + Xarray` chunk-orchestration stack for large-scale ($10\text{ GB} \to 1\text{ PB+}$), multi-variable, Fortran-layered scientific datasets with a **zero-allocation, L1-cache morselized, bitmask-propagating Mojo 1.2 execution engine** and lightweight labeled `Dataset` / `DataArray` semantics.

---

## Table of Contents

1. [Motivation & Quantitative Bottleneck Analysis](#1-motivation--quantitative-bottleneck-analysis)
2. [Core Innovations & Head-to-Head Benchmark Analysis](#2-core-innovations--head-to-head-benchmark-analysis)
3. [System Architecture (4-Layer Stack)](#3-system-architecture-4-layer-stack)
4. [Compiler & Hardware Sympathy Optimizations](#4-compiler--hardware-sympathy-optimizations)
5. [Mojo 1.2 Language Subtleties & Engineering Lessons](#5-mojo-12-language-subtleties--engineering-lessons)
6. [Current Limitations & Honest Technical Debt Audit](#6-current-limitations--honest-technical-debt-audit)
7. [Strategic Opportunities (I/O, Duck Arrays, GPU)](#7-strategic-opportunities-io-duck-arrays-gpu)
8. [Roadmap to a Full Production Xarray + Dask Replacement](#8-roadmap-to-a-full-production-xarray--dask-replacement)
9. [Developer Continuation Guide (Picking Up on a New Machine)](#9-developer-continuation-guide-picking-up-on-a-new-machine)

---

## 1. Motivation & Quantitative Bottleneck Analysis

Even when heavy binary decompression and array decoding are written in Mojo and exposed zero-copy to Python as `numpy.ndarray` buffers, keeping **Dask + Xarray + NumPy** as the chunk orchestrator hits four fundamental walls:

### 1.1 Amdahl's Law in Explicit Chunk-Level Task Graphs (`Dask`)
Dask compiles array expressions into an explicit Python dictionary of tasks with complexity $O(\text{ops} \times \text{chunks})$:
* **Per-task Python overhead:** Every Dask task incurs $\sim 0.3\text{–}1.5\text{ ms}$ of single-threaded Python graph construction, key hashing, scheduler state transitions, and GIL/thread-pool handoffs. When a Mojo SIMD kernel processes a chunk in $5\text{–}25\text{ }\mu\text{s}$, **$>98\%$ of wall-clock time is spent inside Python scheduling**.
* **Graph-construction OOM at Petabyte scale:** A $1\text{ PB}$ archive stored in $16\text{ MB}$ layers contains $\sim 6.25 \times 10^7$ chunks. Wrapping custom per-layer binary decoders via `dask.array.map_blocks` materializes explicit Python dictionary entries—Measured at **$2,566\text{ ms}$ for just $2,000$ slabs**, scaling linearly to **minutes and tens of gigabytes of coordinator RAM** long before reading a single byte of data.

### 1.2 Cache-Hierarchy Destruction via Full-Chunk Materialization
In NumPy, Xarray, and Dask, every operator (`hypot`, `*`, `+`, `/`, `where`) materializes a full intermediate array buffer before the next operator runs:
* Consider an `(800 × 401)` `float32` horizontal slab ($1.28\text{ MB}$) or a full `(800 × 401 × 20 × 25)` forecast cube ($641.6\text{ MB}$ per variable).
* Across a 10-operator expression on 16 threads, full-slab buffers consume $>120\text{ MB}$—blowing past per-core L1d ($32\text{ KB}$), per-core L2 ($512\text{ KB}$–$1\text{ MB}$), and shared L3 ($32\text{ MB}$). Every intermediate tensor spills to main DRAM ($\sim 30\text{–}50\text{ GB/s}$ on DDR4/DDR5) instead of staying in L1d ($\sim 500\text{–}1,000+\text{ GB/s}$ aggregate across cores).

### 1.3 Missing-Data Representation Tax (`numpy.ma` vs. `NaN` vs. `UInt64` Bitmasks)
Scientific binary formats (ECCC RPN `fstd`, WMO GRIB2, NetCDF/HDF5 CF-conventions) encode missingness heterogeneously: sentinel integers (`0xFFFF`, `-32768`) prior to `scale_factor`/`add_offset` unpacking, 1-bit land/sea or validity bitmaps (`@@`), or companion `uint8` quality-control flags.

| Property | `numpy.ma.MaskedArray` | Xarray / NumPy `NaN` | `marray-mask` 3-State `UInt64` Bitmask |
| :--- | :--- | :--- | :--- |
| **Mask memory overhead** | $8\text{ bits/elem}$ (`+25%` on `f32`, `+50%` on `i16`) | $0\text{ bits}$ (in-band float payload) | **$1\text{ bit/elem}$ (`+3.1%` on `f32`)**, **$0\text{ bits}$ when `MSTATE_VALID`** |
| **Preserves integer dtypes?** | Yes | **No** (forces cast to `float32`/`float64`) | **Yes** (out-of-band bitmask) |
| **All-valid layer fast path** | Scans/allocates `bool` masks unless `nomask` | Still runs `isnan` checks in `nanmean`/`nanmin` | **`MSTATE_VALID`: zero mask memory, unmasked SIMD** |
| **All-masked layer short-circuit** | Computes payload then masks | Computes `NaN` across entire array | **`MSTATE_MASKED`: $O(1)$ skip of math & reductions** |
| **Mask propagation cost** | Byte-by-byte boolean ops + heavy Python wrapper | Implicit IEEE 754 `NaN` propagation | **1 CPU `or` instruction per 64 elements (`UInt64 \| UInt64`)** |
| **Reduction behavior** | Allocates masked copies / single-threaded ufuncs | Branchy/slow `np.nanmean` / `np.nanstd` passes | **64-cell `0` (all-valid) & `~0` (all-masked) word skipping + `pop_count`** |

### 1.4 Fortran Column-Major (`(ni, nj)`) Layout Fidelity
Operational NWP and Fortran-heritage models (such as ECCC's GEM/GDPS) store horizontal slabs column-major (`flat_idx = i + j * ni`, where longitude `i` has stride `1`). Python stacks default to C-order (`row-major`), often triggering hidden `np.ascontiguousarray` transpose copies or strided gather loads. `marray-mask` treats stride-1 innermost axes natively without transposing.

---

## 2. Core Innovations & Head-to-Head Benchmark Analysis

### 2.1 Four Core Architectural Innovations
1. **Parametric $O(1)$-Memory Grid Catalogs:** Datasets represent chunked N-D domains parametrically via 1D coordinate axes and slice bounds rather than enumerating chunk tasks. Opening a **6,000-cycle GDPS $0.15^\circ$ archive** ($48.6\text{ million}$ slabs, **$560.34\text{ TB}$** logical `float32`) takes **$349\text{ }\mu\text{s}$** and $\sim 30\text{ KB}$ of RAM; `.sel()` binary-searches ascending/descending coordinates and prunes $9,720,000 \to 322$ slabs in **$315\text{ }\mu\text{s}$**.
2. **L1-Cache Morselized Bytecode VM (`MORSEL_LEN = 2048`):** Instead of evaluating DAG nodes slab-by-slab, worker threads stream **2,048-element ($8\text{ KB}$) 64-byte-aligned morsels** through a compiled register bytecode program. All intermediate registers ($3\text{–}5 \times 8.25\text{ KB} \approx 33\text{ KB}$/thread) remain resident in L1d/L2 cache regardless of whether the dataset is $10\text{ MB}$ or $1\text{ PB}$.
3. **3-Pass General Bytecode Optimizer:** Dynamically folds CF `scale_factor`/`add_offset` arithmetic into raw source loads (`BC_LOAD`), fuses `FMA` / `POW2_SUM` / `HYPOT`, and performs linear-scan liveness register coloring so arbitrary user Python/Mojo formulas achieve hand-fused kernel speeds without domain hardcoding.
4. **Encoding-Normalized 3-State Bitmask Algebra:** Decodes sentinel integers, IEEE `NaN`s, `numpy.ma` boolean masks, 1-bit bitmaps, and `uint8` QC flags in a **single memory pass** into `(MSTATE_VALID | MSTATE_MASKED | MSTATE_BITS)`, automatically promoting clean layers (e.g., upper-troposphere `500 hPa` temperature above terrain) to `MSTATE_VALID`.

### 2.2 Benchmark Progression & Roofline Verification
Hardware: **AMD Ryzen 7 5800X (8 cores / 16 threads, Zen 3 AVX2, 32 KB L1d/core, 512 KB L2/core, 32 MB L3, Dual-Channel DDR4)** running on WSL2 Ubuntu (`NumPy 2.4.3`, `Xarray 2026.2.0`, `Dask 2026.3.0`, `Mojo 1.2.0 -O3`). All engines read the **exact same pre-allocated $1.44\text{ GB}$ Fortran-ordered memory cube** (`800 × 401 × 20 × 25`, zero disk I/O) and were verified with `np.testing.assert_allclose`:

| Workload | Xarray + Dask (16T) | Eager Xarray (`NaN`) | NumPy (`numpy.ma`) | Mojo v0.1 (Slab Arena) | Mojo v0.2 (General L1 Morsel VM) | v0.2 Speedup vs. Dask |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dataset / Graph Setup** (2,000 raw layers) | `2,566.48 ms` | `< 0.20 ms` | — | `< 0.20 ms` | **`< 0.20 ms`** | **`> 12,000×`** |
| **W1:** Regional `.sel()` + 10-op Moist Jet + 2D `(lon, lat)` Mean Map | `2,185.40 ms` | `30.39 ms` | `30.74 ms` | `1.84 ms` | **`1.80 ms`** | **`1,216.9×`** (`17×` vs Xarray/NumPy) |
| **W2:** Full Global Cube (All 2,000 slabs, $5.13\text{ GB}$ `float32`) | `5,025.17 ms` | `4,040.41 ms` | `7,863.24 ms` | `782.08 ms` | **`49.89 ms`** | **`100.7×`** (`81×` vs Xarray, `158×` vs `ma`) |
| **W3:** Global Eddy Kinetic Energy `0.5*(U^2+V^2)` reduced over `('lon','time')` → `(lat, level)` | `2,149.99 ms` | `2,249.43 ms` | — | — | **`179.18 ms`** | **`12.0×`** (`12.6×` vs Xarray) |
| **W4:** 3D→4D Stride-0 Broadcast `TM(lon,lat,time) - TT(lon,lat,lvl,time)` (`mean` + `std`) | `829.69 ms` | — | — | — | **`10.32 ms`** | **`80.4×`** |

#### Why Mojo v0.2 Beat Mojo v0.1 by `15.7×` (`782.08 ms → 49.89 ms`) on Workload 2
In Workload 2, the engine reads $4 \times 160.4\text{M}$ `uint16` values + $1 \times 160.4\text{M}$ `uint8` QC flags = **$1.4436\text{ GB}$ of physical RAM** ($5.13\text{ GB}$ of logical `float32` tensors):
$$\text{Sustained Physical Read Bandwidth} = \frac{1.4436\text{ GB}}{0.04989\text{ s}} = \mathbf{28.94\text{ GB/s}} \quad (\text{Effective `float32` Throughput} = \mathbf{102.9\text{ GB/s}})$$
On dual-channel DDR4-3200/3600 ($\sim 35\text{–}42\text{ GB/s}$ theoretical peak, $\sim 30\text{–}34\text{ GB/s}$ achievable multi-core read), **`49.89 ms` is literally at the hardware DRAM bandwidth roofline**. Three changes unlocked this jump while simultaneously removing all benchmark-specific code:
1. **L1d Morselization (`123 MB → 0.5 MB` total working set):** v0.1 allocated six `(800 × 401)` (`1.28 MB`) slab buffers per thread ($123\text{ MB}$ across 16 threads), spilling every intermediate op to DRAM. v0.2 uses four `2,048`-element (`8 KB`) morsel registers per thread ($32\text{ KB}$/thread), keeping all intermediate reads/writes inside L1d/L2 cache.
2. **Single-Pass Decode + Bitmask Construction:** v0.1 scanned raw `uint16` slabs once to unpack floats and a second time whenever `any_sentinel` fired, doubling DRAM reads on low-level slabs. v0.2 packs 64-bit mask words in registers during the single load pass.
3. **General CF Scale/Offset Folding:** Folding `* 0.514444` and `+ 273.15` into `SourceBinding` eliminated two full vector passes.

### 2.3 Realistic Boreal & Aether Stack Benchmark (`benchmarks/bench_boreal_realistic.py`)

Using the cross-package contracts in [`docs/boreal/`](boreal/README.md) and the Tier-1 operational `grtyp` synthesizer in [`src/marray/synth.mojo`](../src/marray/synth.mojo), [`benchmarks/bench_boreal_realistic.py`](../benchmarks/bench_boreal_realistic.py) tests `marray_xr` against the **exact production patterns** of `boreal-io`, `boreal-postproc`, `boreal-stat`, `polaris`, and `aether`:
* **Cube layout:** C-contiguous `(time, vertical, y, x)` (`(n_t, n_z, nj, ni)`) over Fortran-linear `(ni, nj)` plane bytes (`x` stride-1 innermost, zero transpose).
* **Missingness:** Companion FST `@@` bitmaps with `fstd2nc` polarity (`keep where bitmap > 0, mask where bitmap == 0`), producing `NaN` in-memory (`DATASET_FILL_VALUE`) or `1e30` (`fst_mask_fill`) directly on write-out.
* **Baselines (zero disk I/O; raw `float32` slabs & `@@` bitmaps resident in RAM, isolating pure orchestration + bitmap-apply + peel/rewrap + math):**
  1. `open_fst(chunks="record")` — Dask `HighLevelGraph` (`tokenize` + `_lazy_record_chunk` per `(t, z)` record + `np.where(bmap > 0, slab, np.nan)`).
  2. `open_fst(chunks="column")` — Dask `HighLevelGraph` (`_lazy_column_chunk` batching all `n_z` levels per `time` step).
  3. `open_fst(chunks=False)` — Eager `np.where` bitmap apply + `.values` peel → NumPy kernel → `xr.DataArray` rewrap (`fst_*` attrs).

#### Suite Results (`grtyp=U` Yin–Yang Supergrid `16 × 10 × 606 × 540` + Suites B1–B8)

Following the engineering decisions clarified in [`docs/questions.md`](questions.md) (Q1–Q15), `marray-mask` now implements **M1 (`RecordCatalog` + C-ABI `ChunkDecoder` + `PlaneSlotCache`)**, **M4 (`MarrayDuckArray` inside genuine `xr.Dataset`)**, **single-pass multi-sink fusion (`compute_many`)**, **exact `Float64` morsel parity (`precision='float64'`, `rmse_f64`)**, **`TILE_COLS = 64` `(column, level)` vertical scans**, **batched finite `aether` CSR handoff (`to_aether_flat`)**, and **chunked streaming `stream_write_fst`**:

| Realistic Boreal / Aether Suite (`bench_boreal_realistic.py`) | `chunks='record'` HLG | `chunks='column'` HLG (16T) | Eager Peel / Transpose | `marray_xr` (Mojo 1.2) | Speedup |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **B1:** `open_fst` Assembly (80 leads × 40 lvls × 6 vars = 19,200 recs) | `36.22 ms` | `1.79 ms` | — | **`0.017 ms`** (`0.533 ms` M4 `xr.Dataset`) | **`2,144×` vs `record`** (`106×` vs `column`) |
| **B2:** `polaris` Thin 2D Peel (`isel` + nearest `vertical` → `(nj, ni)` buf) | `7.816 ms` | `32.119 ms` | — | **`0.772 ms`** (`2.384 ms` M4 `xr.Dataset`) | **`10.1×` vs `record`** (**`41.6×` vs `column`**) |
| **B3:** `boreal-postproc` `Humidex` (`float32` `@@`) + `stream_write_fst` (`1e30`) | — | `428.58 ms` | `689.65 ms` | **`76.48 ms`** (`89.00 ms` streaming sink) | **`5.6×` vs `column`** (**`9.0×` vs Eager**) |
| **B4:** `boreal-stat` `r.diag` (`ke` `globavg` + `zonavg`, single-pass `compute_many`) | — | `457.80 ms` | `1,195.35 ms` (`values_f64`) | **`61.63 ms`** (`119.58 ms` exact `float64`) | **`7.4×` vs `column`** (**`10.0×` exact `f64`**, `rel_err=2.0e-14`) |
| **B6:** M1 C-ABI `ChunkDecoder` (`datyp=134` + `datyp=2` `@@`, cross-`vctype` + regional) | — | `523.49 ms` (`78.97 ms` reg) | — | **`193.08 ms`** (**`23.79 ms`** regional) | **`2.7×–3.3×` faster** + **`~21 MB` RAM cap** |
| **B7:** `boreal-postproc` Vertical Column Scan (`TILE_COLS=64` vs 4D transpose) | — | — | `1,088.62 ms` (`419 MB` copy) | **`116.14 ms`** | **`9.4×` faster** (zero 4D transpose) |
| **B8:** `aether` Batched `(4, 327240)` Finite Prep + `mogc-api` 8-Thread 16-Lead Warmup | — | `20.206 ms` / `132.10 ms` | — | **`0.547 ms`** / **`6.68 ms`** | **`36.9×` `aether` prep** / **`19.8×` 8T warmup** |

> [!TIP]
> **Resolving the `column` vs. `record` Dilemma (`04-lazy-layouts.md`):** Today in `boreal-io`, `chunks="column"` keeps HLG construction manageable (`1.79 ms`) at the cost of `41×` slower 2D peels in `polaris` (`32.12 ms`, over-reading the whole vertical column), whereas `chunks="record"` speeds up 2D peels (`7.82 ms`) at the cost of `20×` larger HLG dictionaries (`36.22 ms` at 19k records, seconds/GBs at suite scale). `marray-mask`'s parametric catalog gives **both** a `17 µs` open **and** a `0.772 ms` single-record 2D peel from the exact same `Dataset`.

#### Bench 5 — Tier-1 Operational `grtyp` Catalog Sweep (~99% of Operational Header Rows)
Workload: 5-variable moist wind-thermal diagnostic (`hypot(UU, VV) * 0.514444 * HU + (TM_sfc - TT_4d)`) across `SynthGridSpec.large`:

| `grtyp` (Catalog Share) | Geometry & Mask Hook | Cube Shape `(t, z, nj, ni)` | Combined Mask % | `record` HLG | `column` HLG (16T) | Eager Xarray | `marray_xr` | vs. `column` HLG |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`U`** (~53%) | Yin–Yang supergrid + panel seam | `(8, 10, 606, 540)` | `16.09%` | `297.89 ms` | `313.29 ms` | `393.31 ms` | **`20.43 ms`** | **`15.3×`** |
| **`O`** (~23%) | Regional NW-Atlantic heavy coastal `@@` | `(8, 10, 480, 660)` | `61.78%` | `287.76 ms` | `304.45 ms` | `387.34 ms` | **`18.14 ms`** | **`16.8×`** |
| **`Z`** (~15%) | Stretched tic-tac + rotated pole | `(8, 10, 303, 600)` | `28.12%` | `170.20 ms` | `150.35 ms` | `220.74 ms` | **`11.29 ms`** | **`13.3×`** |
| **`G`** (~3%) | Gaussian latitudes | `(8, 10, 288, 576)` | `16.24%` | `171.73 ms` | `146.93 ms` | `195.38 ms` | **`10.65 ms`** | **`13.8×`** |
| **`N`** (~3%) | Polar stereo + outside unit-disk mask | `(8, 10, 480, 480)` | `41.57%` | `202.70 ms` | `212.87 ms` | `308.57 ms` | **`16.64 ms`** | **`12.8×`** |
| **`L`** (~2%) | Regular geographic lat-lon | `(8, 10, 303, 600)` | `15.44%` | `219.65 ms` | `166.72 ms` | `227.24 ms` | **`13.75 ms`** | **`12.1×`** |
| **`M`** (<1%) | Unstructured mesh node list (`nj` thin) | `(8, 10, 4, 80000)` | `18.24%` | `237.51 ms` | `281.88 ms` | `354.67 ms` | **`23.63 ms`** | **`11.9×`** |

Notice **`grtyp=O` (`61.78%` masked)** achieved the **highest speedup (`16.8×`)**: 64-element `ALL_ONES` word skipping in `morsel.mojo` skips continental land blocks completely during reductions, whereas NumPy/Xarray/Dask compute and scan every masked `NaN` cell.

---

## 3. System Architecture (4-Layer Stack)

```mermaid
flowchart TD
    subgraph Frontend["Layer 4: Semantic Labeled Frontends"]
        PY["Python Drop-In API (`python/marray_xr`)\nDataset / DataArray / Variable / __array_ufunc__ / .sel()"]
        MJ_XR["Pure-Mojo API (`src/marray/xarray.mojo`)\nDataset / DataArray / CoordAxis / Variant Attrs"]
    end

    subgraph Bridge["Interop & ABI Boundary (`src/marray_mojo.mojo` & `numpy_bridge.mojo`)"]
        PYINIT["Native Python Extension (`PyInit_marray_mojo`)\nPythonModuleBuilder"]
        CABI["Zero-GIL C-ABI Bytecode Entry (`marray_exec_general_plan`)\nCPython-version agnostic (3.11–3.14+)"]
        NPB["Mojo std.python Bridge (`from_numpy_slab` / `to_numpy_masked`)"]
    end

    subgraph Optimizer["Layer 2 & 3: General Bytecode Compiler & Morsel Scheduler (`src/marray/vm.mojo`)"]
        CSE["Pass 1: Common Subexpression Elimination (CSE)"]
        FUSE["Pass 2: CF Scale/Offset Load Folding + FMA / POW2_SUM / HYPOT Peephole"]
        REG["Pass 3: Dead-Node Pruning + Linear-Scan Liveness Register Coloring"]
        EXEC["Parallel Morsel Pull Engine (`max.algorithm.parallelize`)\nPer-source contiguity coalescing + Stride-0 cross-rank broadcasting"]
    end

    subgraph Kernels["Layer 1: 64B-Aligned SIMD & Bitmask Kernels (`src/marray/morsel.mojo` & `slab.mojo`)"]
        MREG["`MorselReg` (2,048 f32 = 8 KB + 32 x UInt64 mask = 256 B)\n64-byte cache-line aligned `Allocation`"]
        DEC["Single-Pass Decoders (`UInt16` sentinel, `Float32` NaN, `UInt8` QC/ma mask)"]
        SIMD["4-Way Unrolled `simd_width_of` Math + 64-Element Word-Skipping Reductions"]
    end

    PY --> CABI
    PY -.optional.-> PYINIT
    MJ_XR --> NPB
    PYINIT --> CSE
    CABI --> CSE
    CSE --> FUSE --> REG --> EXEC --> MREG
    EXEC --> DEC --> SIMD
```

### Repository Module Map

| File | Role |
| :--- | :--- |
| [`src/marray/morsel.mojo`](../src/marray/morsel.mojo) | 64-byte aligned `MorselReg` (`Float32`) & `MorselReg64` (`Float64`), single-pass span decoders (incl. native `MSRC_FST_BITMAP_F32` `float32` `0/1` `@@` planes), 4-way unrolled SIMD math, and 64-element bitmask word-skipping `ReduceAcc` kernels. |
| [`src/marray/catalog.mojo`](../src/marray/catalog.mojo) | M1 separate-`.so` C-ABI `ChunkDecoder` (`ChunkDecoderCABI`), `is_datyp_parallel_safe` (`datyp=6` guard), whole-record `PlaneSlotCache` (1 decode/plane on regional slices), and reference `datyp=134` / `datyp=2` codecs. |
| [`src/marray/vertical.mojo`](../src/marray/vertical.mojo) | L1d-resident `TILE_COLS = 64` `TileTensor` `(column, level)` vertical physics scheduler matching `cmc_postproc_kernels` (`lift_columns` / `vertical_scan_columns`) with zero 4D cube transpose. |
| [`src/marray/tensor_layout_ops.mojo`](../src/marray/tensor_layout_ops.mojo) | Additive `TileTensor` + `TensorLayout` (`col_major` Fortran & `row_major` Boreal) 2D stencils, static `vectorize[W]` CPU kernels, and persistent `CompiledMaxSession` (`max.gpu` `DeviceContext` + pre-staged `aether` CSR weights on VRAM). |
| [`src/marray/vm.mojo`](../src/marray/vm.mojo) | 24-opcode general bytecode ISA, 3-pass multi-sink compiler (`compile_and_optimize_multi_dag`), `execute_multi_sink_plan`, spatial-partitioned high-rank axis reductions, and `execute_bytecode_plan_f64`. |
| [`src/marray_mojo.mojo`](../src/marray_mojo.mojo) | Shared-library entrypoint exporting `PyInit_marray_mojo` (`PythonModuleBuilder`) plus zero-GIL C-ABI entrypoints (`marray_exec_multi_sink_plan`, `marray_exec_general_plan_f64`, `marray_exec_vertical_column_scan`, `marray_builtin_fst_chunk_decoder`). |
| [`python/marray_xr/`](../python/marray_xr/__init__.py) | `Dataset`/`DataArray`/`Variable`, [`catalog.py`](../python/marray_xr/catalog.py) (`open_fst_catalog`, cross-`vctype` `from_sources`), [`duck.py`](../python/marray_xr/duck.py) (M4 `MarrayDuckArray`, `open_fst_xarray`), `compute_many`, `to_aether_flat`, `rmse_f64`, and `stream_write_fst`. |
| [`src/marray/slab.mojo`](../src/marray/slab.mojo) | Standalone pure-Mojo 2D Fortran-contiguous `MaskedSlab` container, 1-bit bitmap (`@@`) decoder, and synthetic ECCC GDPS layer generator. |
| [`src/marray/lazy.mojo`](../src/marray/lazy.mojo) | Standalone pure-Mojo `GDPSGridSpec`, `DomainWindow`, `LazyGraph`, and slab-level bounded `WorkerArena` executor. |
| [`src/marray/xarray.mojo`](../src/marray/xarray.mojo) | Standalone pure-Mojo `Dataset`, `DataArray`, `CoordAxis` ($O(\log N)$ ascending/descending binary search), and `Variant` metadata `attrs`. |
| [`src/marray/numpy_bridge.mojo`](../src/marray/numpy_bridge.mojo) | Direct Mojo-side `std.python` zero-copy pointer access (`zero_copy_ptr_f32`) and `numpy.ma` ↔ `MaskedSlab` conversion. |
| [`tests/test_marray_mask.mojo`](../tests/test_marray_mask.mojo) | `std.testing.TestSuite` unit tests covering Fortran indexing, 3-way mask propagation, tile-straddling tail bounds (`n=65`), and bytecode fusion/register coloring. |
| [`benchmarks/bench_mojo_vs_dask_xarray.py`](../benchmarks/bench_mojo_vs_dask_xarray.py) | 4-workload head-to-head verification and timing suite against `numpy.ma`, eager `xarray`, and `xarray + dask`. |

---

## 4. Compiler & Hardware Sympathy Optimizations

### 4.1 64-Byte Cache-Line Aligned Vector Loads (`VEC_ALIGN`)
By default in Mojo/LLVM, `ptr.unsafe_load[width=W](off)` assumes scalar alignment (`4 bytes` for `Float32`), which can emit unaligned loads or split wide vectors across cache-line boundaries. `MorselReg` allocates storage via:
```mojo
comptime W: Int = simd_width_of[DType.float32]()
comptime VEC_ALIGN: Int = align_of[SIMD[DType.float32, W]]()
comptime CACHE_ALIGN: Alignment = Alignment.of[SIMD[DType.float32, 16]]()  # 64B
```
All internal morsel loops step by `STEP_W = W * 4` and issue `unsafe_load[width=W, alignment=VEC_ALIGN]` / `unsafe_store[alignment=VEC_ALIGN]`, guaranteeing aligned `vmovaps` / `vfma231ps` codegen on AVX2 and AVX-512.

### 4.2 Breaking Loop-Carried FMA Dependency Chains in Reductions
On modern out-of-order CPUs (e.g., Zen 3/4/5, Golden Cove, Apple M-series), `Float64` vector addition/FMA has a 3–4 cycle latency with 0.5-cycle reciprocal throughput. A single accumulator `sum_v += d0` stalls the pipeline waiting on the previous iteration's writeback. `morsel_accumulate_stats` maintains independent `s0, s1` and `mn0, mn1` / `mx0, mx1` vector accumulators using `std.math.min` / `max` (emitting branchless `vminps` / `vmaxps`), doubling reduction IPC.

### 4.3 64-Element Bitmask Word Skipping
In real geophysical grids, masked points cluster spatially (continents in ocean grids, mountain ranges on low-altitude pressure surfaces, radar range rings). Rather than unpacking mask bits for every SIMD vector:
* `word == ~UInt64(0)` (`0xFFFFFFFFFFFFFFFF`): **all 64 cells are masked** → skipped in a single integer branch (`continue`).
* `word == UInt64(0)`: **all 64 cells are valid** → executes 8 unmasked SIMD loads + `valid_cnt += 64` with zero mask-unpacking overhead.
* Mixed boundary words (coastlines, mountain edges): uses hardware `pop_count(word)` for exact valid counts and unpacks byte-level `SIMD[DType.bool, 8]` predicates only on non-zero bytes.

### 4.4 Per-Source Contiguity Coalescing + Stride-0 Broadcasting
Before launching threads, `execute_bytecode_plan` inspects each `SourceBinding`'s physical strides against the active `.sel()` window (`_is_source_coalesced`):
* If a variable's slice spans full inner dimensions (`0..f0`, `0..f1`, ...) with contiguous Fortran strides (`1, f0, f0*f1, ...`), that source streams flat 2,048-element blocks directly via pointer offset (`coalesced = True`).
* Simultaneously, lower-rank variables in the *same* expression (such as 3D `TM(lon, lat, time)` combined with 4D `TT(lon, lat, level, time)`) carry `stride = 0` along missing axes (`level`) and gather seamlessly without materializing a broadcasted 4D copy in RAM.

---

## 5. Mojo 1.2 Language Subtleties & Engineering Lessons

When extending this codebase, keep these exact Mojo 1.2 behaviors in mind (also codified in [`.agents/skills/mojo-syntax`](../.agents/skills/mojo-syntax/SKILL.md)):

1. **Origin Exclusivity & Borrow-Disjoint Scratch Swapping:**
   Passing `ws.regs[dst]` (`mut`) alongside `ws.regs[r0]` (`imm`) into `morsel_add(dst, a, b)` triggers a compile-time aliasing error because both borrow `origin_of(ws.regs)`. Do **not** paper over this with `MutAnyOrigin` (which disables exclusivity checks and extends lifetimes across the scope). Instead, `MorselWorkspace` holds a separate field `ws.scratch: MorselReg` with a disjoint origin:
   ```mojo
   morsel_add(ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], count)
   swap(ws.regs[inst.dst_reg], ws.scratch)  # O(1) pointer swap, zero element copies!
   ```
2. **`match` Bare-Identifier Binding Trap (`case == CONST:` vs. `case CONST:`):**
   In Mojo pattern matching, a bare identifier `case BC_ADD:` **does not compare against the `comptime` constant `BC_ADD`**—it introduces a new variable binding named `BC_ADD` that matches *every* value and makes all subsequent arms unreachable! Always write `case == BC_ADD:` when matching `Int` constants (or conform enum structs to `EnumLike` and match `.CASE`).
3. **`divmod(x, d)` Returns `(quotient, remainder)`:**
   Just like Python's `q, r = divmod(x, d)`, the **first** tuple element is `x // d` and the **second** is `x % d`. Unpacking `var idx, rem = divmod(cur, d0_len)` reverses them, causing `idx >= d0_len`, `span <= 0`, and an infinite strided gather loop. Always unpack `var rem, idx = divmod(cur, d0_len)`.
4. **Linear `Allocation[T]` Ownership vs. `Pointer` Helpers:**
   `self._data_alloc.unsafe_ptr()` carries origin `origin_of(self._data_alloc._alloc)`. Rather than writing out nested internal origin paths on accessor methods, declare return types via `type_of(self._data_alloc.unsafe_ptr())` with `mut self` (for mutable pointers) and `self` (for immutable pointers).
5. **Dual `PyInit_` + `abi("C")` Shared Library Pattern:**
   * Conda's `c-compiler` package ships an older `GLIBC 2.17` sysroot that fails linking standalone Mojo executables (`undefined reference to dladdr@GLIBC_2.34`), whereas `mojo build -O3 --emit shared-lib` links cleanly on any Linux/macOS environment.
   * Exporting **both** `PyInit_marray_mojo` (`PythonModuleBuilder`) and `marray_exec_general_plan` (`abi("C")`) in [`src/marray_mojo.mojo`](../src/marray_mojo.mojo) allows native `import marray_mojo` in matching Python environments while letting `ctypes` call the zero-GIL C-ABI across *any* Python version (including Python 3.13/3.14+ free-threading builds) with zero CPython ABI coupling.
6. **ASAP Destruction When Laundering Addresses via `Int(list.unsafe_ptr())`:**
   Because Mojo uses **ASAP (As-Soon-As-Possible) value destruction** rather than lexical-scope end destruction, converting a Mojo `List` or `Allocation` pointer to a raw integer (`Int(buf.unsafe_ptr())`) for a C-ABI / `ImmUntrackedOrigin` descriptor hides provenance from the borrow checker. If `buf` is not referenced later in the function, Mojo frees `buf` *immediately* after `Int(buf.unsafe_ptr())`—before the C-ABI / morsel reader executes! Always pin Mojo-owned buffers past the final untracked consumer call with `_ = buf` (or keep them referenced in Python across `ctypes` calls via a `keep_alive` list).
7. **Zero-Overhead Separate-`.so` C-ABI `ChunkDecoder` Function Pointers:**
   To link `boreal-io`'s `cmc_io_fst` `.so` with `marray_mojo.so` without Python GIL callbacks (`docs/questions.md` Q1), declare `comptime ChunkDecoderCABI = def(...) thin abi("C") -> Int32`, verify `comptime assert size_of[ChunkDecoderCABI]() == size_of[Int]()`, and cast the symbol address (`var slot = fn_addr; var fn_ptr = Pointer(to=slot).unsafe_bitcast[ChunkDecoderCABI]()[]`). LLVM compiles this to a direct register `call` instruction with zero wrapper overhead.

---

## 6. Current Limitations & Honest Technical Debt Audit

1. **Fixed 4D `divmod` Unpack in Non-Coalesced Loader:**
   `_load_morsel_from_source` and the non-specialized branch of `execute_bytecode_plan` currently unpack 4 dimensions (`d0..d3`), while the ABI reserves `MAX_DIMS = 6`. Supporting arbitrary $N$-dimensional tensors ($N \le 8+$) cleanly requires replacing per-morsel `divmod` chains with a rank-generic **coordinate odometer** (`coord: InlineArray[Int, MAX_DIMS]` incremented along axis 0 by `span` with carry propagation).
2. **Axis-Reduction Partial Buffer Memory Scaling (`n_workers * out_cells`):**
   Currently, when reducing along a subset of axes (`want_axis_out`), each worker thread accumulates into a private `out_cells` buffer (`n_workers * out_cells * 12 bytes`) to avoid atomic contention.
   * For 2D maps `(800, 401)` across 16 threads, `16 * 320,800 * 12 B = 61 MB`—completely fine.
   * However, if a user reduces *only* `time` (`25 -> 1`) while keeping 3D `(2400, 1201, 80)` (`out_cells = 230.6M`), `16 * out_cells` would allocate `~44 GB`!
   * **Fix (Milestone M2):** When `out_cells > 262_144`, partition `parallelize` across **disjoint tiles of the kept output coordinates** instead of flat input morsels. Each worker then accumulates into a single L2-sized output tile (`1×` output memory, zero cross-worker merge pass!).
3. **Predicate Masking vs. General Boolean & 3-Arg `where(cond, x, y)`:**
   `DataArray.where()` currently optimizes threshold comparisons (`da.where(da >= c)`), whereas general Xarray code frequently writes cross-variable masks (`ds["TT"].where((ds["HU"] > 5.0) & (ds["lat"] > 45.0), other=0.0)`). Adding explicit boolean/bitmask registers (`BC_CMP_LT/LE/GT/GE/EQ/NE`, `BC_BIT_AND/OR/NOT`, and ternary `BC_WHERE(cond_mask_reg, x_reg, y_reg)`) generalizes this completely.
4. **Dtype Coverage & `datetime64` / `cftime` Coordinates:**
   * Fast-path SIMD span loaders currently cover `uint16` and `float32` (casting other NumPy dtypes on input). Native single-pass loaders are still needed for `int8`, `uint8`, `int16`, `int32`, `uint32`, `int64`, and `float64` (plus `float64` compute pipelines when scientific precision requires 64-bit mantissas, e.g., geopotential height anomalies or hydrostatic integration).
   * `CoordAxis` currently indexes `float64` coordinates; Xarray users rely heavily on ISO-8601 string slicing (`ds.sel(time=slice("2026-11-01", "2026-11-05"))`) translated to `int64` epoch nanoseconds.
5. **In-Memory Pointers vs. Out-of-Core Streaming File Readers & Sinks:**
   The current VM reads strided memory addresses (`data_addr`). For datasets larger than RAM ($100\text{ GB} \to 1\text{ PB}$), `BC_LOAD` must pull chunks on demand from a **File/Granule Catalog + Pluggable Mojo Binary Decoder** (calling OS `pread` / `mmap` + decompression per touched `(file_id, byte_offset, byte_len)` chunk and evicting compressed buffers immediately after morsel consumption), plus a streaming chunked sink (`to_zarr` / `to_binary`).
6. **Cross-Element / Cross-Chunk Stencils, Regridding & GroupBy:**
   All current VM ops are point-wise + axis reductions. Horizontal finite differences ($\partial T/\partial x$, vorticity, divergence), rolling windows, bilinear/conservative grid interpolation, and `groupby("time.month")` require halo-exchange or key-grouped accumulators.

---

## 7. Strategic Opportunities (I/O, Duck Arrays, GPU)

### 7.1 Plugging In `boreal-io` (`cmc_io_fst_hdr` + `cmc_io_fst_data`) Directly
> **See also:** [`docs/boreal/`](boreal/README.md) — how boreal-io / aether / postproc / stat use xarray, dask, and FST decode today (contracts for `RecordCatalog`, masks, and peel→kernel→rewrap).

`boreal-io` already cracks XDF/RSF headers (`cmc_io_fst_hdr`) and unpacks `datyp = 0, 1, 2, 4, 5, 130, 133, 134` in Mojo (`cmc_io_fst_data`: `unpack_many_f32_into`). Connecting them directly to `vm.mojo` removes Python from the I/O and bitmap-apply loop completely:
1. Feed `cmc_io_fst_hdr` / `fst-classifier-mojo` header rows (`path, addr, lng, datyp, nbits, ni, nj, nk, mask_addr, mask_lng`) directly into a flat `MmapRecordCatalog` (`byte_offset ≈ (addr - 1) * 8`).
2. Partition work across Mojo worker threads so each thread unpacks touched records + companion `@@` / `!@` bitmaps (`keep > 0, mask == 0`) into a reusable slab slot, immediately converting `@@` into `MSTATE_VALID | MSTATE_MASKED | MSTATE_BITS` (`MSRC_FST_BITMAP_U8` / `MSRC_FST_BITMAP_WORDS`) without a second `np.where` pass.
3. Add **Multi-Sink Reduction Fusion** in `execute_bytecode_plan` so `boreal-stat` pipelines requesting multiple reductions on the same derived expression (e.g., `globavg` + `zonavg` on `ke` in Bench 4) evaluate `ke` once instead of twice—cutting Bench 4's `84.88 ms` in half to `~42 ms` (`>12×` vs. `column` HLG).
4. Add a **2–4 thread I/O prefetch ring buffer** (or `io_uring` on Linux via C FFI) + zero-copy DLPack / MAX Buffer handoff into **aether**'s Mojo CSR `apply_weights` and **mojoviz** LUT painters—**requiring zero Mojo language `async/await`**.

### 7.2 Drop-In Xarray Integration via Python Array API (`__array_namespace__`)
You do **not** have to rewrite Xarray's plotting, NetCDF/Zarr export, or `repr` HTML widgets to kill Dask!
* Xarray natively wraps any duck array implementing `__array_ufunc__`, `__array_function__`, or the **Python Array API standard (`__array_namespace__`)** (the exact mechanism `cubed` and `pint` use inside Xarray).
* Wrapping `marray_xr.DataArray` as an Xarray duck-array backend (`xr.open_dataset("run.fstd", engine="marray_mojo")`) lets scientists use **100% standard `xarray.Dataset`** (including `hvplot`, `matplotlib`, `cartopy`, CF metadata, and `datetime64` indexes!) while `marray_mojo` intercepts lazy math, slices, masks, and `.compute()`—completely bypassing Dask!

### 7.3 Empirical Proof: `TileTensor` / `TensorLayout`, Static Compilation & GPU (`bench_tensorlayout_and_gpu.py`)

[`benchmarks/bench_tensorlayout_and_gpu.py`](../benchmarks/bench_tensorlayout_and_gpu.py) benchmarks [`src/marray/tensor_layout_ops.mojo`](../src/marray/tensor_layout_ops.mojo) (`TileTensor` `col_major` Fortran & `row_major` Boreal + `std.algorithm.vectorize[W]` + persistent `CompiledMaxSession` on `NVIDIA GeForce RTX 5080` vs. 16-thread Ryzen 7 5800X CPU, all `assert_allclose` verified):

| Proof Workload (`grtyp=U` `16 × 10 × 606 × 540`) | Dask `column` HLG | 16T CPU (`marray`) | GPU End-to-End (incl. PCIe) | GPU Warm In-VRAM (`TileTensor`) | Architectural Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Proof 1:** 2D Vorticity Stencil (`dv/dx - du/dy` + 5-pt `@@` mask dilation) | `597.64 ms` | **`62.77 ms`** (`row_major`) / **`62.73 ms`** (`col_major` F) | **`47.96 ms`** (`12.5×` vs Dask) | **`1.759 ms`** (**`35.7×` vs CPU**, **`340×` vs Dask**) | `TileTensor` handles Fortran `(ni,nj,ns)` & Boreal `(ns,nj,ni)` at identical speed with zero transpose and 1-pass stencil fusion. |
| **Proof 2:** Morsel Bytecode VM vs. Static AOT Kernel (`Humidex + WindModulus`) | `954.81 ms` | **`117.82 ms`** (VM, `5 µs` JIT) vs `129.70 ms` (flat AOT) | `190.70 ms` *(1.26 GB raw PCIe)* | **`1.639 ms`** (**`79.1×` vs CPU**, **`583×` vs Dask**) | **Zero interpreter penalty on CPU**: 2048-elem morsel dispatch + 64-bit word skipping beats flat AOT. Uncompressed 1-pass PCIe transfers lose; multi-op VRAM residency wins `79×`. |
| **Proof 3A:** 4D Coalesced Vertical `CAPE` Parcel Lift | `1,088.62 ms` *(Py peel+T)* | **`143.60 ms`** (`TILE_COLS=64`) | **`54.85 ms`** (**`2.6×` vs CPU**) | **`0.614 ms`** (**`233.7×` vs CPU**, **`1,772×` vs Py**) | Coalesced `x = global_idx.x` 4D `TileTensor` threads read across `z` with zero transpose, beating CPU even with cold PCIe transfers. |
| **Proof 3B:** Compressed `datyp=134` + `@@` → Unpack → `UV` → `aether` CSR `(4, 65536)` | `> 25 ms` | **`26.170 ms`** | **`0.699 ms`** (**`37.5×` vs CPU**) | — | **Killer `mogc-api` / `polaris` pattern**: upload `2 MB` compressed FST records to persistent session (`294 ms` one-time init), decompress + mask + `aether` CSR SpMV in VRAM, return compact tile in **`0.70 ms`**. |

> [!IMPORTANT]
> **`DevicePassable` Rule for `TileTensor` GPU Kernels (`Mojo 1.2` + `MAX 26.7`):** Target-width `Int` and `UInt` intentionally do **not** conform to `DevicePassable` in `ctx.enqueue_function`. Always construct GPU `TileTensor` dynamic shapes with fixed-width `Int32` tuples (`row_major((Int32(nj), Int32(ni)))` / `col_major((Int32(ni), Int32(nj)))`) and type scalar kernel bounds as `Int32` / `Int64`.

---

## 8. Roadmap to a Full Production Xarray + Dask Replacement

Boreal-facing milestone checklist and phased UX (duck array → pure Mojo Dataset): [`docs/boreal/10-mojo-replacement-checklist.md`](boreal/10-mojo-replacement-checklist.md).

### Milestone M1 — Pluggable Out-of-Core Binary File Catalog & Decoder Trait *(Core Engine Implemented)*
* **Implemented in v0.3 (`src/marray/catalog.mojo`, `python/marray_xr/catalog.py`):**
  * Separate-`.so` `ChunkDecoderCABI` function-pointer interface + `is_datyp_parallel_safe` (`datyp=6` serial safety guard).
  * Per-worker `PlaneSlotCache` whole-record plane decoder (`datyp=134` + `datyp=2` `float32` `0/1` `@@` bitmaps), regional `[j0:j1, i0:i1]` single-decode slicing, cross-`vctype` `list[Dataset]` joins (`from_sources`), and chunked `stream_write_fst` (`fst_mask_fill=1e30`).
* **Next Step (On Work Machine with `boreal-io`):** Wire `cmc_io_fst_hdr` header arrays + `cmc_io_fst` raw unpacker symbol address + OS `pread`/`mmap` file-descriptor pool directly into `CatalogVarSpec`, and optionally add a packed-`UInt64` 1-bit `@@` unpack entry in `boreal-io` (`32×` less bitmap traffic than `float32` `0/1` planes).

### Milestone M2 — Rank-Generic Coordinate Odometer & Output-Tiled Axis Reductions *(Partially Implemented)*
* **Implemented in v0.3 (`src/marray/vm.mojo`):**
  * Spatial-partitioned axis reductions when inner `(d0, d1)` are kept and `out_cells > 65,536` (`0` cross-worker buffer replication), single-pass multi-sink fusion (`execute_multi_sink_plan` / `compute_many`), and full `Float64` morsel parity path (`execute_bytecode_plan_f64` / `rmse_f64`).
* **Remaining:** Replace 4D `divmod` unpacking with a rank-$N$ ($1 \le N \le 8$) coordinate odometer and add native `int8`/`int16`/`int32`/`int64` span loaders.

### Milestone M3 — Full Boolean Mask Algebra & Ternary `where(cond, x, y)`
* **Deliverables:**
  * Add `BC_CMP_{EQ,NE,LT,LE,GT,GE}` producing bitmask registers, `BC_MASK_{AND,OR,XOR,NOT}`, and 3-operand `BC_WHERE(cond, true_val, false_val)`.
  * Support `DataArray` boolean operators (`&`, `|`, `~`) and cross-variable `xr.where(cond, x, y)`.

### Milestone M4 — Xarray `BackendEntrypoint` & `__array_namespace__` Duck Array *(Phase A Implemented)*
* **Implemented in v0.3 (`python/marray_xr/duck.py`):**
  * `MarrayDuckArray` (`__array_ufunc__`, `__array_function__`, `__getitem__`, `__array__`) + `open_fst_xarray` returning genuine `list[xr.Dataset]` objects without Dask.
* **Next Step:** Register `[project.entry-points."xarray.backends"]` (`xr.open_dataset(..., engine="marray_fst")`) and complete full Python Array API (`__array_namespace__`) coverage (`concat`, `broadcast_arrays`, `where`).

### Milestone M5 — Spatial Stencils, Halo Exchange & Coordinate Alignment
* **Deliverables:**
  * 2D Fortran slab stencil kernels (`diff_x`, `diff_y`, spherical $\nabla$, vorticity $\zeta$, divergence, `rolling` windows) with mask dilation at invalid neighbors.
  * Automatic inner/outer coordinate alignment (`align`) and `groupby` / temporal `resample` binned accumulators.

### Milestone M6 — Multi-Node Embarrassingly Parallel Partitioning & GPU Backend
* **Deliverables:**
  * Stateless shard execution (`shard_id`, `num_shards`) for zero-coordinator SLURM / MPI / Kubernetes petabyte batch runs + optional C-FFI `aws-c-s3` / `object_store` HTTP range reader for cloud Zarr/GRIB archives.
  * Optional `max.gpu` `TileTensor` bytecode backend for compute-bound multi-member ensemble kernels.

---

## 9. Developer Continuation Guide (Picking Up on a New Machine)

### Prerequisites
* Linux (`x86_64` or `aarch64`), macOS (`arm64`), or Windows with **WSL2 Ubuntu**.
* [Pixi](https://pixi.sh) (`curl -fsSL https://pixi.sh/install.sh | bash`).

### Clone, Build, Test & Benchmark
```bash
git clone git@github.com:itsgifnotjiff/marray-mask.git
cd marray-mask

# 1. Install Mojo 1.x, MAX, Python, NumPy, Xarray, and Dask via Pixi:
pixi install

# 2. Run the Mojo std.testing.TestSuite unit tests:
pixi run test

# 3. Verify bidirectional zero-copy Mojo <-> NumPy/Xarray interop:
pixi run interop

# 4. Run the standalone pure-Mojo ECCC GDPS petabyte catalog & slab demo:
pixi run demo

# 5. Build `build/marray_mojo.so` and run both benchmark suites:
pixi run bench-boreal    # Realistic Boreal/Aether suite (record/column HLG, peel-rewrap, U..M)
pixi run bench-vs-dask   # General 4-way Mojo VM vs NumPy.ma vs Xarray vs Dask benchmark
```

### Included Agent Skills
This repository vendors Modular's official agent skills under [`.agents/skills/`](../.agents/skills/):
* `mojo-syntax` (including [`references/idiomatic-mojo.md`](../.agents/skills/mojo-syntax/references/idiomatic-mojo.md))
* `mojo-python-interop`
* `mojo-gpu-fundamentals`
* `new-modular-project`

Any AI coding agent supporting the Agent Skills Standard (Antigravity, Claude Code, Cursor, etc.) will automatically load these rules when editing Mojo code in this repo.
