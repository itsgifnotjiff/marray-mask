# `marray-mask` — General Mojo 1.2 Masked Lazy Array & Xarray/Dask Engine

A high-performance **Mojo 1.2** masked array and lazy bytecode execution stack designed to replace Python's `NumPy.ma + Dask + Xarray` chunk-orchestration bottleneck on large ($10\text{ GB} \to 1\text{ PB+}$), Fortran-contiguous scientific datasets (modeled on ECCC's **GDPS** NWP archive).

📖 **Deep-Dive Documentation:**
* **[`docs/ARCHITECTURE_AND_ROADMAP.md`](docs/ARCHITECTURE_AND_ROADMAP.md)** — Full architectural design, roofline math, bytecode compiler internals, realistic Boreal/Aether benchmarks, Mojo 1.2 subtleties, limitations audit, and M1–M6 roadmap.
* **[`docs/boreal/`](docs/boreal/README.md)** — Cross-package contracts for `boreal-io` (`open_fst`, `@@` bitmaps, `column`/`record` HLG), `boreal-postproc`, `boreal-stat`, `aether`, `polaris`, and `mogc-api`.

---

## Highlights & Realistic Boreal Stack Benchmarks

Tested against realistic `boreal-io` `HighLevelGraph` (`chunks="record"` and `chunks="column"`), `chunks=False` Eager Peel→NumPy→Rewrap, and Eager Xarray across all 7 Tier-1 RPN `grtyp` geometries (`U` Yin–Yang 53%, `O` Coastal 23%, `Z` Rotated 15%, `G` Gaussian 3%, `N` Polar Stereo 3%, `L` Lat-Lon 2%, `M` Mesh <1%) using C-contiguous `(time, vertical, y, x)` cubes and `fstd2nc`-polarity `@@` bitmaps (`keep > 0, mask == 0`, verified `assert_allclose` on Ryzen 7 5800X 16T):

| Realistic Boreal Suite (`benchmarks/bench_boreal_realistic.py`) | `record` HLG | `column` HLG (16T) | Eager Peel/Rewrap | `marray_xr` (Mojo 1.2 VM) | Speedup |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **B1:** `open_fst` Graph Assembly (80 leads × 40 lvls × 6 vars = 19,200 records) | `32.76 ms` | `1.62 ms` | — | **`0.012 ms`** | **`2,760×` vs `record`** (`137×` vs `column`) |
| **B2:** `polaris` Thin 2D Peel (`grtyp=U` `606×540`, `WindModulus` → `aether` `(nj,ni)` buf) | `7.024 ms` | `31.841 ms` | — | **`0.858 ms`** | **`8.2×` vs `record`** (**`37.1×` vs `column`**) |
| **B3:** `boreal-postproc` Spooki `Humidex` (`exp`) + Direct `1e30` (`fst_mask_fill`) Write Buf | — | `525.57 ms` | `635.69 ms` | **`60.43 ms`** | **`8.7×` vs `column`** (**`10.5×` vs Eager**) |
| **B4:** `boreal-stat` `r.diag` (`ke` + Yin–Yang `areacell` Weighted `globavg` + `zonavg`) | — | `537.69 ms` | `549.57 ms` | **`84.88 ms`** | **`6.3×` vs `column`** (`6.5×` vs Eager) |
| **B5:** Tier-1 `grtyp` Sweep (`U, O, Z, G, N, L, M` at `SynthGridSpec.large`) | `170–298 ms` | `147–313 ms` | `195–393 ms` | **`10.7–23.6 ms`** | **`11.9×–16.8×` vs `column` HLG** |

*(On raw quantized `uint16` full-cube benchmarks in [`bench_mojo_vs_dask_xarray.py`](benchmarks/bench_mojo_vs_dask_xarray.py), single-pass decode + L1 morsel fusion reaches **`49.89 ms` (`28.9 GB/s` physical DRAM read)** vs `5,025 ms` in Dask [`100.7×`] and `7,863 ms` in `numpy.ma` [`157.6×`].)*

---

## Quick Start (Linux / macOS / WSL2)

```bash
git clone git@github.com:itsgifnotjiff/marray-mask.git
cd marray-mask

pixi install
pixi run test            # Mojo std.testing.TestSuite unit tests (incl. Tier-1 synth grids)
pixi run synth-bench     # Standalone Mojo Tier-1 grtyp (U/O/Z/L/G/N/M) fill+decode demo
pixi run bench-boreal    # Realistic Boreal/Aether suite (record/column HLG, peel-rewrap, U..M)
pixi run bench-vs-dask   # General 4-way Mojo VM vs NumPy.ma vs Xarray vs Dask benchmark
```

## Example Usage (`python/marray_xr`)

```python
import numpy as np
import marray_xr as mxr

ds = mxr.Dataset(
    {
        "TT": mxr.Variable(("lon", "lat", "level", "time"), tt_u16, scale=0.01, offset=-120.0, fill_value=0xFFFF),
        "UU": mxr.Variable(("lon", "lat", "level", "time"), uu_u16, scale=0.01, offset=-120.0, fill_value=0xFFFF),
        "VV": mxr.Variable(("lon", "lat", "level", "time"), vv_u16, scale=0.01, offset=-120.0, fill_value=0xFFFF),
        "HU": mxr.Variable(("lon", "lat", "level", "time"), hu_u16, scale=0.001, qc_flags=hu_qf, qc_max_valid=1),
        "TM": mxr.Variable(("lon", "lat", "time"), tm_f32, mask=land_mask),  # 3D auto-broadcasts into 4D!
    },
    coords={"lon": lons, "lat": lats, "level": levels, "time": times},
)

sub = ds.sel(lon=slice(280.0, 312.0), lat=slice(42.0, 58.0), level=slice(1000.0, 500.0), time=slice(6.0, 60.0))
wspd = np.hypot(sub["UU"], sub["VV"]) * 0.514444
ratio = (wspd * sub["HU"]) / (sub["TT"] + 273.15)
jet = ratio.where(ratio >= 0.35)

map_2d = jet.mean(dim=("level", "time"))  # Returns labeled xarray.DataArray
```

## Repository Structure

* [`docs/ARCHITECTURE_AND_ROADMAP.md`](docs/ARCHITECTURE_AND_ROADMAP.md) — Comprehensive technical spec, optimizations, Mojo 1.2 subtleties, limitations, and M1–M6 roadmap
* [`src/marray/morsel.mojo`](src/marray/morsel.mojo) — 64-byte aligned 2,048-element L1 `MorselReg`, single-pass decoders & 4-way unrolled SIMD kernels
* [`src/marray/vm.mojo`](src/marray/vm.mojo) — General 24-opcode bytecode VM, 3-pass optimizer (CSE + CF-scale/FMA/HYPOT fusion + register coloring) & N-D executor
* [`src/marray_mojo.mojo`](src/marray_mojo.mojo) — Native `PyInit_marray_mojo` (`PythonModuleBuilder`) + zero-GIL C-ABI shared library entrypoint
* [`python/marray_xr/`](python/marray_xr/__init__.py) — General Python Xarray/Dask replacement frontend
* [`src/marray/slab.mojo`](src/marray/slab.mojo), [`lazy.mojo`](src/marray/lazy.mojo), [`xarray.mojo`](src/marray/xarray.mojo), [`numpy_bridge.mojo`](src/marray/numpy_bridge.mojo) — Standalone pure-Mojo stack & NumPy zero-copy bridge
* [`.agents/skills/`](.agents/skills/) — Vendored official Modular Mojo & MAX skills (`modular/skills`)
