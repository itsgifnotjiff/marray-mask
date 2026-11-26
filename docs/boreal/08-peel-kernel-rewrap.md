# Peel → kernel → rewrap (postproc & stat)

Across **boreal-postproc**, **boreal-stat**, and app hot paths, compute is **not** expressed as rich Dask graphs. The dominant pattern is:

```text
xarray DataArray/Dataset
  → np.asarray(da.values) / .data   # peel (forces compute if dask-backed)
  → Mojo .so or NumPy kernel
  → xr.DataArray(..., dims, coords, attrs)  # rewrap + stamp
```

Do **not** prioritize `map_blocks` / `Client` / `persist` parity for these packages. Prioritize zero-copy peel, fused kernels, masked reductions, and attr-preserving rewrap (or a Dataset API that never peeled).

## boreal-postproc

**Lifecycle:** `NamedPlugin(ds, **opts).compute() -> xr.Dataset` (also `from_sources(list[Dataset])`).

**Resolve fields:** name → `attrs["fst_nomvar"|"nomvar"]` → OpenData id.

**Open habit:** examples and CLI use `open_fst(..., chunks=False)` so plugins see numpy-backed cubes.

**Hot paths (representative):**

| Class | Examples |
|-------|----------|
| Pointwise | Humidex, WindModulus, dewpoint/depression, mixing ratio, density, scale/add |
| Vertical | column scans, CAPE/LCL/parcel composites, wet-bulb / potential temps |
| Ensemble | by-point mean/std/percentile/threshold fractions |
| Spatial | InterpolationHorizontalGrid (aether), GridCut/Merge, stenfilt, weighted averages |
| Time | interval difference / min-max across forecast slices |

Mojo kernels (`cmc_postproc_kernels`) cover many arith/hypot/atan2/thermo/vorticity paths with NumPy fallbacks under `backend=auto` size thresholds. Attr stamp via boreal-io `update_metadata` ([06](06-cf-attrs-and-consumer-contract.md)).

## boreal-stat

**APIs:** `ke`, `zonavg`, `meriavg`, `globavg`, `timavg`/`timmax`/`timmin`, `stdev`, pair metrics, spectral helpers, …

**Pattern:** `find_by_nomvar` → `values_f64` → Mojo (`cmcstat_kernels` / pair metrics) or NumPy → `rewrap` preserving dims/coords/attrs. `_map_cubes` over `open_fst` multi-cube lists.

**Weighted globals:** prefer `areacell` / `areacell_yy` from `open_fst(..., cell_measures=True)`; Gaussian weights via `numpy.polynomial.legendre.leggauss` where needed.

No in-library Dask client—kernels expect contiguous f64 (or documented dtypes).

## Ops the morsel VM should cover first

Aligned with marray-mask W1–W4 and boreal consumers:

1. Elementwise arith + fused `hypot` / `pow2_sum` (wind, KE).
2. Masked reductions (nanmean-style) with all-valid / all-masked short circuits.
3. Axis reductions over `time` / `vertical` / horizontal with weighted means.
4. Broadcast surface−3D (e.g. `TM(lon,lat,time) - TT(lon,lat,level,time)`).
5. Column-parallel vertical scans/indices (thermo).
6. Attr-preserving Dataset assign without NaN-align accidents.

Horizontal CSR remapping stays **aether** ([07](07-aether-xarray-usage.md)); call it with buffers, don’t rebuild weights inside the VM.

## What not to prioritize from these repos

- Distributed dask schedulers
- Generic `map_blocks` wrappers around plugins
- Reimplementing Spooki plugin orchestration graphs in Mojo before the array engine is solid

Plugin **dispatch** can remain Python; the payload path should not.
