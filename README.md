# `marray-mask` — General Mojo 1.2 Masked Lazy Array & Xarray/Dask Engine

A high-performance **Mojo 1.2** masked array and lazy bytecode execution stack designed to replace Python's `NumPy.ma + Dask + Xarray` chunk-orchestration bottleneck on large ($10\text{ GB} \to 1\text{ PB+}$), Fortran-contiguous scientific datasets (modeled on ECCC's **GDPS** NWP archive).

📖 **Deep-Dive Documentation:** See **[`docs/ARCHITECTURE_AND_ROADMAP.md`](docs/ARCHITECTURE_AND_ROADMAP.md)** for the full architectural design, roofline math, bytecode compiler internals, Mojo 1.2 subtleties, limitations audit, and M1–M6 production roadmap.

---

## Highlights & Verified Benchmark Numbers

Tested on identical pre-allocated Fortran-contiguous memory buffers (`800 × 401 × 20 × 25` ECCC GDPS cube, $1.44\text{ GB}$ raw / $5.13\text{ GB}$ logical `float32`, AMD Ryzen 7 5800X 16 threads, `NumPy 2.4.3`, `Xarray 2026.2.0`, `Dask 2026.3.0`, `Mojo 1.2.0 -O3`, verified `assert_allclose`):

| Workload | Xarray + Dask (16T) | Eager Xarray (`NaN`) | NumPy (`numpy.ma`) | `marray_xr` (Mojo 1.2 VM) | Speedup vs. Dask |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dataset / Graph Setup** (2,000 raw layers) | `2,566.48 ms` | `< 0.20 ms` | — | **`< 0.20 ms`** | **`> 12,000×`** |
| **W1:** Regional `.sel()` + Moist Jet + 2D `(lon, lat)` Mean Map | `2,185.40 ms` | `30.39 ms` | `30.74 ms` | **`1.80 ms`** | **`1,216.9×`** (`17×` vs Xarray/NumPy) |
| **W2:** Full Global Cube (All 2,000 slabs, $5.13\text{ GB}$ `float32`) | `5,025.17 ms` | `4,040.41 ms` | `7,863.24 ms` | **`49.89 ms`** | **`100.7×`** (`158×` vs `numpy.ma`) |
| **W3:** Eddy Kinetic Energy `0.5*(U^2+V^2)` → `(lat, level)` Zonal Profile | `2,149.99 ms` | `2,249.43 ms` | — | **`179.18 ms`** | **`12.0×`** |
| **W4:** 3D→4D Stride-0 Broadcast `TM - TT` (`mean` & `std`) | `829.69 ms` | — | — | **`10.32 ms`** | **`80.4×`** |

---

## Quick Start (Linux / macOS / WSL2)

```bash
git clone git@github.com:itsgifnotjiff/marray-mask.git
cd marray-mask

pixi install
pixi run test            # Mojo std.testing.TestSuite unit tests
pixi run interop         # Bidirectional zero-copy Mojo <-> NumPy.ma / Xarray check
pixi run demo            # Pure-Mojo 560 TB virtual GDPS catalog & slab benchmark
pixi run bench-vs-dask   # Build build/marray_mojo.so & run 4-workload Dask/Xarray comparison
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
