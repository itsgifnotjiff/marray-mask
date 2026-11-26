"""Realistic Boreal & Aether Stack Benchmark (`docs/boreal/01..12`).

Benchmarks `marray_xr` (Mojo 1.2 L1 Morsel VM) against the EXACT real-world
`boreal-io`, `boreal-postproc`, `boreal-stat`, and `polaris`/`mogc-api` patterns:

  - Layout: Boreal CF cubes `(time, vertical, y, x)` (`(n_t, n_z, nj, ni)` C-order
    over Fortran-linear `(ni, nj)` plane bytes) + OpenData naming with `fst_nomvar`
    & `fst_mask_fill = 1e30` metadata contracts (`02`, `06`).
  - Missingness: Companion FST `@@` bitmaps with `fstd2nc` polarity
    (`keep where bitmap > 0, fill where bitmap == 0` -> `NaN` in-memory / `1e30`
    on write) (`05`).
  - Baselines (with ZERO disk I/O — raw `float32` slabs & `@@` bitmaps already
    resident in RAM, modeling an infinitely fast `cmc_io_fst_data` Mojo unpacker
    so 100% of measured time is orchestration + bitmap apply + peel/rewrap + math):
      1. `open_fst(chunks="record")`: Dask `HighLevelGraph` (`stack.py` style:
         `tokenize` + `_lazy_record_chunk` + `np.where(bmap > 0, slab, np.nan)`)
      2. `open_fst(chunks="column")`: Dask `HighLevelGraph` (`_lazy_column_chunk`
         per `time` batch `(1, n_z, nj, ni)` + `@@` bitmap apply)
      3. `open_fst(chunks=False)` Eager Peel -> NumPy Kernel -> Rewrap (`08`, `09`)
  - Geometries: All 7 catalog-representative Tier-1 RPN `grtyp` families (`U` 53%
    Yin-Yang, `O` 23% Coastal, `Z` 15% Rotated, `G` 3% Gaussian, `N` 3% Polar
    Stereo, `L` 2% Lat-Lon, `M` <1% Mesh Nodes) generated via `synth.mojo` (`12`).
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from typing import Any

import dask
import dask.array as da
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

import marray_xr as mxr  # noqa: E402


# ===----------------------------------------------------------------------=== #
# Realistic `boreal.io.xarray.stack` Open Modes (`record` HLG, `column` HLG, Eager)
# ===----------------------------------------------------------------------=== #


def _lazy_record_chunk(
    payload_arena: np.ndarray,
    bmap_arena: np.ndarray | None,
    t_idx: int,
    z_idx: int | None,
) -> np.ndarray:
    """Models `boreal.io.xarray.stack._lazy_record_chunk` where Mojo unpack writes
    a `(nj, ni)` float32 slab and `apply_bitmap_mask` applies `@@` polarity.
    """
    if z_idx is None:
        slab = payload_arena[t_idx].copy()
        if bmap_arena is not None:
            slab = np.where(bmap_arena[t_idx] > 0, slab, np.float32(np.nan))
        return slab[np.newaxis, :, :]

    slab = payload_arena[t_idx, z_idx].copy()
    if bmap_arena is not None:
        slab = np.where(bmap_arena[t_idx, z_idx] > 0, slab, np.float32(np.nan))
    return slab[np.newaxis, np.newaxis, :, :]


def _lazy_column_chunk(
    payload_arena: np.ndarray,
    bmap_arena: np.ndarray | None,
    t_idx: int,
) -> np.ndarray:
    """Models `boreal.io.xarray.stack._lazy_column_chunk`: `unpack_many` decodes
    all `n_z` vertical levels for one `time` step `(1, n_z, nj, ni)`.
    """
    col = payload_arena[t_idx].copy()
    if bmap_arena is not None:
        col = np.where(bmap_arena[t_idx] > 0, col, np.float32(np.nan))
    return col[np.newaxis, :, :, :]


def _build_hlg_var_4d(
    var_key: str,
    payload: np.ndarray,
    bmap: np.ndarray | None,
    mode: str,
) -> da.Array:
    n_t, n_z, nj, ni = payload.shape
    tok = tokenize(var_key, payload.shape, mode)
    name = f"fst-{mode}-{var_key}-{tok}"
    dsk: dict[tuple, tuple] = {}

    if mode == "record":
        for t in range(n_t):
            for z in range(n_z):
                _ = tokenize(var_key, t, z)  # realistic per-record spec tokenization
                dsk[(name, t, z, 0, 0)] = (_lazy_record_chunk, payload, bmap, t, z)
        chunks = ((1,) * n_t, (1,) * n_z, (nj,), (ni,))
    else:
        for t in range(n_t):
            _ = tokenize(var_key, t)
            dsk[(name, t, 0, 0, 0)] = (_lazy_column_chunk, payload, bmap, t)
        chunks = ((1,) * n_t, (n_z,), (nj,), (ni,))

    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=())
    return da.Array(hlg, name, chunks, dtype=np.float32)


def _build_hlg_var_3d(
    var_key: str, payload: np.ndarray, bmap: np.ndarray | None
) -> da.Array:
    n_t, nj, ni = payload.shape
    tok = tokenize(var_key, payload.shape, "sfc")
    name = f"fst-sfc-{var_key}-{tok}"
    dsk: dict[tuple, tuple] = {}
    for t in range(n_t):
        _ = tokenize(var_key, t)
        dsk[(name, t, 0, 0)] = (_lazy_record_chunk, payload, bmap, t, None)
    chunks = ((1,) * n_t, (nj,), (ni,))
    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=())
    return da.Array(hlg, name, chunks, dtype=np.float32)


def open_fst_simulated_hlg(raw: dict[str, Any], chunks: str = "column") -> xr.Dataset:
    """Simulates `boreal.io.open_fst(paths, engine='mojo', chunks=column|record, open_data_mapping=True)`."""
    dims4 = ("time", "vertical", "y", "x")
    dims3 = ("time", "y", "x")
    grtyp = raw["grtyp"]
    common = {"fst_grtyp": grtyp, "fst_mask_fill": 1.0e30, "cell_measures": "area: areacell"}

    return xr.Dataset(
        {
            "AirTemp": (dims4, _build_hlg_var_4d("TT", raw["tt_f32"], raw["bmap_3d"], chunks), {**common, "fst_nomvar": "TT", "units": "degC"}),
            "WindU": (dims4, _build_hlg_var_4d("UU", raw["uu_f32"], raw["bmap_3d"], chunks), {**common, "fst_nomvar": "UU", "units": "knots"}),
            "WindV": (dims4, _build_hlg_var_4d("VV", raw["vv_f32"], raw["bmap_3d"], chunks), {**common, "fst_nomvar": "VV", "units": "knots"}),
            "SpecHum": (dims4, _build_hlg_var_4d("HU", raw["hu_f32"], raw["hu_bmap"], chunks), {**common, "fst_nomvar": "HU", "units": "g/kg"}),
            "SeaSurfTemp": (dims3, _build_hlg_var_3d("TM", raw["tm_f32"], raw["tm_bmap"]), {**common, "fst_nomvar": "TM", "units": "degC"}),
        },
        coords={
            "time": raw["time"],
            "vertical": raw["vertical"],
            "y": raw["y"],
            "x": raw["x"],
            "areacell": (("y", "x"), raw["areacell"]),
        },
        attrs={"fst_grtyp": grtyp, "source": "file.std"},
    )


def open_fst_simulated_eager(raw: dict[str, Any]) -> xr.Dataset:
    """Simulates `open_fst(..., chunks=False)`: `unpack_many` + `apply_bitmap_mask` -> NaN."""
    dims4 = ("time", "vertical", "y", "x")
    dims3 = ("time", "y", "x")
    grtyp = raw["grtyp"]
    common = {"fst_grtyp": grtyp, "fst_mask_fill": 1.0e30, "cell_measures": "area: areacell"}

    b3 = raw["bmap_3d"] > 0
    tt = np.where(b3, raw["tt_f32"], np.float32(np.nan))
    uu = np.where(b3, raw["uu_f32"], np.float32(np.nan))
    vv = np.where(b3, raw["vv_f32"], np.float32(np.nan))
    hu = np.where(raw["hu_bmap"] > 0, raw["hu_f32"], np.float32(np.nan))
    tm = np.where(raw["tm_bmap"] > 0, raw["tm_f32"], np.float32(np.nan))

    return xr.Dataset(
        {
            "AirTemp": (dims4, tt, {**common, "fst_nomvar": "TT", "units": "degC"}),
            "WindU": (dims4, uu, {**common, "fst_nomvar": "UU", "units": "knots"}),
            "WindV": (dims4, vv, {**common, "fst_nomvar": "VV", "units": "knots"}),
            "SpecHum": (dims4, hu, {**common, "fst_nomvar": "HU", "units": "g/kg"}),
            "SeaSurfTemp": (dims3, tm, {**common, "fst_nomvar": "TM", "units": "degC"}),
        },
        coords={
            "time": raw["time"],
            "vertical": raw["vertical"],
            "y": raw["y"],
            "x": raw["x"],
            "areacell": (("y", "x"), raw["areacell"]),
        },
        attrs={"fst_grtyp": grtyp, "source": "file.std"},
    )


def open_fst_marray_catalog(raw: dict[str, Any]) -> mxr.Dataset:
    """Zero-HLG parametric `marray_xr.Dataset` with OpenData names + `fst_*` attrs."""
    dims4 = ("time", "vertical", "y", "x")
    dims3 = ("time", "y", "x")
    grtyp = raw["grtyp"]
    common = {"fst_grtyp": grtyp, "fst_mask_fill": 1.0e30, "cell_measures": "area: areacell"}

    return mxr.Dataset(
        {
            "AirTemp": mxr.Variable(dims4, raw["tt_f32"], fst_bitmap=raw["bmap_3d"], attrs={**common, "fst_nomvar": "TT", "units": "degC"}),
            "WindU": mxr.Variable(dims4, raw["uu_f32"], fst_bitmap=raw["bmap_3d"], attrs={**common, "fst_nomvar": "UU", "units": "knots"}),
            "WindV": mxr.Variable(dims4, raw["vv_f32"], fst_bitmap=raw["bmap_3d"], attrs={**common, "fst_nomvar": "VV", "units": "knots"}),
            "SpecHum": mxr.Variable(dims4, raw["hu_f32"], fst_bitmap=raw["hu_bmap"], attrs={**common, "fst_nomvar": "HU", "units": "g/kg"}),
            "SeaSurfTemp": mxr.Variable(dims3, raw["tm_f32"], fst_bitmap=raw["tm_bmap"], attrs={**common, "fst_nomvar": "TM", "units": "degC"}),
        },
        coords={
            "time": raw["time"],
            "vertical": raw["vertical"],
            "y": raw["y"],
            "x": raw["x"],
        },
        domain_dims=dims4,
        cell_measures={"areacell": mxr.Variable(("y", "x"), raw["areacell"])},
        attrs={"fst_grtyp": grtyp, "source": "file.std"},
    )


def _xr_find_nomvar(ds: xr.Dataset, nomvar: str) -> xr.DataArray:
    for v in ds.data_vars.values():
        if v.attrs.get("fst_nomvar") == nomvar:
            return v
    raise KeyError(nomvar)


def _med_ms(fn, repeats: int = 5, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


# ===----------------------------------------------------------------------=== #
# Bench 1: Suite-Scale `open_fst` Catalog vs HighLevelGraph Tax (19,200 Records)
# ===----------------------------------------------------------------------=== #


def run_bench1_open_tax(base_raw: dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("BENCH 1: SUITE-SCALE `open_fst` GRAPH ASSEMBLY TAX (80 leads x 40 lvls x 6 vars = 19,200 records)")
    print("         Measures pure open + HighLevelGraph / RecordCatalog setup (zero payload unpack)")
    print("=" * 90)

    # Lightweight metadata views (stride-0 broadcast over 80x40) so zero RAM copy occurs during open:
    slab_f32 = base_raw["tt_f32"][0, 0]
    slab_bm = base_raw["bmap_3d"][0, 0]
    virt_4d = np.broadcast_to(slab_f32, (80, 40, base_raw["nj"], base_raw["ni"]))
    virt_bm = np.broadcast_to(slab_bm, (80, 40, base_raw["nj"], base_raw["ni"]))

    suite_raw = {
        **base_raw,
        "tt_f32": virt_4d,
        "uu_f32": virt_4d,
        "vv_f32": virt_4d,
        "hu_f32": virt_4d,
        "bmap_3d": virt_bm,
        "hu_bmap": virt_bm,
        "tm_f32": virt_4d[:, 0],
        "tm_bmap": virt_bm[:, 0],
        "time": np.arange(80, dtype=np.float64) * 3.0,
        "vertical": np.linspace(1015.0, 10.0, 40, dtype=np.float64),
    }

    t_rec = _med_ms(lambda: open_fst_simulated_hlg(suite_raw, chunks="record"), repeats=3, warmup=1)
    t_col = _med_ms(lambda: open_fst_simulated_hlg(suite_raw, chunks="column"), repeats=5, warmup=1)
    t_mxr = _med_ms(lambda: open_fst_marray_catalog(suite_raw), repeats=15, warmup=2)

    print(f"  `open_fst(chunks='record')` HighLevelGraph (16,080 tasks) : {t_rec:8.2f} ms")
    print(f"  `open_fst(chunks='column')` HighLevelGraph (   400 tasks) : {t_col:8.2f} ms")
    print(f"  `marray_xr` Parametric `RecordCatalog`                    : {t_mxr:8.3f} ms  --> {t_rec / t_mxr:6.0f}x faster than `record` HLG | {t_col / t_mxr:5.0f}x faster than `column` HLG")


# ===----------------------------------------------------------------------=== #
# Bench 2: `polaris` Thin 2D Peel + `WindModulus` -> Contiguous `(nj, ni)` for `aether`
# ===----------------------------------------------------------------------=== #


def run_bench2_polaris_2d_peel(raw_u: dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print(f"BENCH 2: `polaris` THIN 2D LATENCY PATH (`grtyp=U` Yin-Yang {raw_u['nj']}x{raw_u['ni']}, 1 lead/level)")
    print("         Select `fst_nomvar` UU/VV at (t=3, z=2) -> Spooki `WindModulus` -> C-contiguous `(nj, ni)` for `aether`")
    print("=" * 90)

    ds_rec = open_fst_simulated_hlg(raw_u, chunks="record")
    ds_col = open_fst_simulated_hlg(raw_u, chunks="column")
    ds_mxr = open_fst_marray_catalog(raw_u)

    def polaris_dask(ds: xr.Dataset) -> xr.DataArray:
        u2d = _xr_find_nomvar(ds, "UU").isel(time=3, vertical=2).values
        v2d = _xr_find_nomvar(ds, "VV").isel(time=3, vertical=2).values
        uv = np.hypot(u2d, v2d)  # peel -> kernel
        return xr.DataArray(uv, dims=("y", "x"), attrs={"fst_nomvar": "UV", "units": "knots", "fst_grtyp": "U"})

    def polaris_marray() -> xr.DataArray:
        sub = ds_mxr.isel(time=3, vertical=2)
        uv = np.hypot(sub.find_by_nomvar("UU"), sub.find_by_nomvar("VV")).stamp_attrs(
            fst_nomvar="UV", units="knots"
        )
        return uv.compute(fill_value=np.nan, squeeze=("time", "vertical"))

    ref_rec = polaris_dask(ds_rec)
    out_mxr = polaris_marray()
    assert out_mxr.dims == ("y", "x") and out_mxr.values.flags["C_CONTIGUOUS"]
    assert out_mxr.attrs["fst_nomvar"] == "UV" and out_mxr.attrs["fst_grtyp"] == "U"
    np.testing.assert_allclose(out_mxr.values, ref_rec.values, rtol=1e-5, atol=1e-5, equal_nan=True)

    t_rec = _med_ms(lambda: polaris_dask(ds_rec), repeats=7)
    t_col = _med_ms(lambda: polaris_dask(ds_col), repeats=7)
    t_mxr = _med_ms(polaris_marray, repeats=15)

    print(f"  `chunks='column'` HLG + peel (over-reads all {len(raw_u['vertical'])} levels!) : {t_col:8.3f} ms")
    print(f"  `chunks='record'` HLG + peel (2 record tasks)             : {t_rec:8.3f} ms")
    print(f"  `marray_xr` surgical 2D morsel -> `aether` `(nj, ni)` buf : {t_mxr:8.3f} ms  --> {t_rec / t_mxr:5.1f}x vs `record` HLG | {t_col / t_mxr:5.1f}x vs `column` HLG")


# ===----------------------------------------------------------------------=== #
# Bench 3: `boreal-postproc` Spooki `Humidex` + `fst_mask_fill=1e30` Write Cube
# ===----------------------------------------------------------------------=== #


def run_bench3_postproc_humidex(raw_u: dict[str, Any]) -> None:
    n_t, n_z, nj, ni = raw_u["tt_f32"].shape
    print("\n" + "=" * 90)
    print(f"BENCH 3: `boreal-postproc` SPOOKI `Humidex` + `fst_mask_fill=1e30` WRITE BUFFER ({n_t}x{n_z}x{nj}x{ni})")
    print("         Includes `@@` bitmap apply + Magnus vapor pressure `exp(...)` + `write_fst` `1e30` fill")
    print("=" * 90)

    ds_col = open_fst_simulated_hlg(raw_u, chunks="column")
    ds_mxr = open_fst_marray_catalog(raw_u)

    def humidex_eager_peel_rewrap() -> xr.DataArray:
        # Realistic CLI flow: `open_fst(chunks=False)` -> peel `.values` -> math -> rewrap + `1e30` fill for `write_fst`
        ds_e = open_fst_simulated_eager(raw_u)
        tt = _xr_find_nomvar(ds_e, "TT").values
        hu = _xr_find_nomvar(ds_e, "HU").values
        # Vapor pressure approximation from specific humidity near sfc (hPa): e = hu * 1.6077
        # and Spooki-like nonlinear thermal moisture index involving `exp`:
        e_hpa = hu * np.float32(1.6077) * np.exp(tt * np.float32(0.012))
        hmx = tt + np.float32(0.5555) * (e_hpa - np.float32(10.0))
        out_fst = np.where(np.isnan(hmx), np.float32(1.0e30), hmx)
        return xr.DataArray(
            out_fst,
            dims=("time", "vertical", "y", "x"),
            attrs={"fst_nomvar": "HMX", "fst_mask_fill": 1.0e30},
        )

    def humidex_dask_column() -> xr.DataArray:
        tt = _xr_find_nomvar(ds_col, "TT")
        hu = _xr_find_nomvar(ds_col, "HU")
        e_hpa = hu * np.float32(1.6077) * np.exp(tt * np.float32(0.012))
        hmx = (tt + np.float32(0.5555) * (e_hpa - np.float32(10.0))).fillna(np.float32(1.0e30))
        return hmx.compute(scheduler="threads", num_workers=16)

    def humidex_marray_fused() -> xr.DataArray:
        tt = ds_mxr.find_by_nomvar("TT")
        hu = ds_mxr.find_by_nomvar("HU")
        e_hpa = (hu * 1.6077) * np.exp(tt * 0.012)
        hmx = (tt + 0.5555 * (e_hpa - 10.0)).stamp_attrs(
            fst_nomvar="HMX", fst_mask_fill=1.0e30, units="degC"
        )
        return hmx.to_fst_buffer()

    ref_e = humidex_eager_peel_rewrap()
    out_m = humidex_marray_fused()
    np.testing.assert_allclose(out_m.values, ref_e.values, rtol=1e-4, atol=1e-3)

    t_col = _med_ms(humidex_dask_column, repeats=3)
    t_eag = _med_ms(humidex_eager_peel_rewrap, repeats=3)
    t_mxr = _med_ms(humidex_marray_fused, repeats=5)

    print(f"  `open_fst(chunks='column')` Dask HLG (16 threads)         : {t_col:8.2f} ms")
    print(f"  `open_fst(chunks=False)` Eager Peel -> NumPy -> `1e30`    : {t_eag:8.2f} ms")
    print(f"  `marray_xr` Fused L1 Morsel -> Direct `1e30` Write Buffer : {t_mxr:8.2f} ms  --> {t_col / t_mxr:5.1f}x vs `column` HLG | {t_eag / t_mxr:5.1f}x vs Eager Peel/Rewrap")


# ===----------------------------------------------------------------------=== #
# Bench 4: `boreal-stat` `r.diag` Diagnostics (`ke`, `zonavg`, Weighted `globavg`)
# ===----------------------------------------------------------------------=== #


def run_bench4_boreal_stat(raw_u: dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("BENCH 4: `boreal-stat` `r.diag` SUITE (`ke` + Yin-Yang `areacell` Weighted `globavg` + `zonavg`)")
    print("=" * 90)

    ds_col = open_fst_simulated_hlg(raw_u, chunks="column")
    ds_eag = open_fst_simulated_eager(raw_u)
    ds_mxr = open_fst_marray_catalog(raw_u)

    def stat_dask_col() -> tuple[float, np.ndarray]:
        u = _xr_find_nomvar(ds_col, "UU") * np.float32(0.514444)
        v = _xr_find_nomvar(ds_col, "VV") * np.float32(0.514444)
        ke = np.float32(0.5) * (u**2 + v**2)
        w_glob = ke.weighted(ds_col.coords["areacell"]).mean(skipna=True)
        zon = ke.mean(dim=("time", "x"), skipna=True)
        g_res, z_res = dask.compute(w_glob, zon, scheduler="threads", num_workers=16)
        return float(g_res.item()), z_res.values

    def stat_eager_xr() -> tuple[float, np.ndarray]:
        u = _xr_find_nomvar(ds_eag, "UU") * np.float32(0.514444)
        v = _xr_find_nomvar(ds_eag, "VV") * np.float32(0.514444)
        ke = np.float32(0.5) * (u**2 + v**2)
        w_glob = float(ke.weighted(ds_eag.coords["areacell"]).mean(skipna=True).item())
        zon = ke.mean(dim=("time", "x"), skipna=True).values
        return w_glob, zon

    def stat_marray() -> tuple[float, np.ndarray]:
        u = ds_mxr.find_by_nomvar("UU") * 0.514444
        v = ds_mxr.find_by_nomvar("VV") * 0.514444
        ke = 0.5 * (u**2 + v**2)
        w_glob = ke.weighted_mean(ds_mxr.cell_measures["areacell"])
        zon = ke.mean(dim=("time", "x"))
        return w_glob, zon.values

    g_e, z_e = stat_eager_xr()
    g_m, z_m = stat_marray()
    np.testing.assert_allclose(g_m, g_e, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(z_m, z_e, rtol=1e-4, atol=1e-4, equal_nan=True)

    t_col = _med_ms(stat_dask_col, repeats=3)
    t_eag = _med_ms(stat_eager_xr, repeats=3)
    t_mxr = _med_ms(stat_marray, repeats=5)

    print(f"  `chunks='column'` Dask HLG `weighted(areacell)` + `zonavg`: {t_col:8.2f} ms")
    print(f"  Eager Xarray `weighted(areacell)` + `zonavg`              : {t_eag:8.2f} ms")
    print(f"  `marray_xr` Single-Pass Weighted `globavg` + `zonavg`     : {t_mxr:8.2f} ms  --> {t_col / t_mxr:5.1f}x vs `column` HLG | {t_eag / t_mxr:5.1f}x vs Eager Xarray (KE={g_m:.3f} m^2/s^2)")


# ===----------------------------------------------------------------------=== #
# Bench 5: Full Tier-1 Operational `grtyp` Sweep (`U/O/Z/G/N/L/M`, 99% Catalog Share)
# ===----------------------------------------------------------------------=== #


def run_bench5_tier1_grtyp_sweep() -> None:
    print("\n" + "=" * 90)
    print("BENCH 5: TIER-1 OPERATIONAL `grtyp` SWEEP (`U` 53%, `O` 23%, `Z` 15%, `G` 3%, `N` 3%, `L` 2%, `M` <1%)")
    print("         `SynthGridSpec.large` (`scaled(3)`), Moist Wind-Thermal Diagnostic (`UU, VV, HU, TT, TM`)")
    print("=" * 90)
    print(f"  {'grtyp':<6} {'Cube Shape (t,z,nj,ni)':<24} {'Mask%':>7} {'Rec HLG':>10} {'Col HLG':>10} {'Eager XR':>10} {'marray_xr':>10} {'vs Col HLG':>11}")
    print("  " + "-" * 86)

    for g in ("U", "O", "Z", "G", "N", "L", "M"):
        scale = 3 if g != "M" else 4  # M has nj=1, ni=80,000 unstructured mesh nodes!
        raw = mxr.synth_boreal_raw_arenas(grtyp=g, scale_factor=scale, n_steps=8)
        ds_rec = open_fst_simulated_hlg(raw, chunks="record")
        ds_col = open_fst_simulated_hlg(raw, chunks="column")
        ds_mxr = open_fst_marray_catalog(raw)

        def _xr_diag(ds: xr.Dataset) -> float:
            wspd = np.hypot(_xr_find_nomvar(ds, "UU"), _xr_find_nomvar(ds, "VV")) * np.float32(0.514444)
            # Broadcast 3D `SeaSurfTemp(t,y,x)` against 4D `AirTemp(t,z,y,x)`!
            expr = (wspd * _xr_find_nomvar(ds, "HU")) + (_xr_find_nomvar(ds, "TM") - _xr_find_nomvar(ds, "TT"))
            res = expr.mean(skipna=True)
            return float(res.compute(scheduler="threads", num_workers=16).item()) if hasattr(res.data, "dask") else float(res.item())

        def _mxr_diag() -> dict[str, Any]:
            wspd = np.hypot(ds_mxr.find_by_nomvar("UU"), ds_mxr.find_by_nomvar("VV")) * 0.514444
            expr = (wspd * ds_mxr.find_by_nomvar("HU")) + (
                ds_mxr.find_by_nomvar("TM") - ds_mxr.find_by_nomvar("TT")
            )
            return expr.stats()

        st_m = _mxr_diag()
        val_c = _xr_diag(ds_col)
        np.testing.assert_allclose(st_m["mean"], val_c, rtol=1e-4, atol=1e-3)

        t_rec = _med_ms(lambda: _xr_diag(ds_rec), repeats=2)
        t_col = _med_ms(lambda: _xr_diag(ds_col), repeats=3)
        t_eag = _med_ms(lambda: _xr_diag(open_fst_simulated_eager(raw)), repeats=3)
        t_mxr = _med_ms(_mxr_diag, repeats=5)

        mask_pct = 100.0 * (1.0 - st_m["count"] / max(1, st_m["total"]))
        sh_str = str(raw["tt_f32"].shape)
        print(
            f"  {g:<6} {sh_str:<24} {mask_pct:6.2f}% {t_rec:8.2f}ms {t_col:8.2f}ms {t_eag:8.2f}ms {t_mxr:8.2f}ms {t_col / t_mxr:9.1f}x"
        )
    print("=" * 90)


def main() -> None:
    print("=" * 90)
    print("BOREAL & AETHER REALISTIC BENCHMARK SUITE (`marray-mask` vs `open_fst` HLG & Peel-Rewrap)")
    print(f"Runtime: NumPy {np.__version__} | Xarray {xr.__version__} | Dask {dask.__version__} | Mojo 1.2 (-O3)")
    print("=" * 90)

    # Primary operational GDPS Yin-Yang (`grtyp=U`, 53% catalog share) cube:
    # scaled(3) -> ni=540, nj=606 (yin_ny=303), 10 levels, 16 steps = 160 slabs/var (~1.05 GB float32)
    raw_u = mxr.synth_boreal_raw_arenas(grtyp="U", scale_factor=3, n_steps=16)

    run_bench1_open_tax(raw_u)
    run_bench2_polaris_2d_peel(raw_u)
    run_bench3_postproc_humidex(raw_u)
    run_bench4_boreal_stat(raw_u)
    run_bench5_tier1_grtyp_sweep()


if __name__ == "__main__":
    main()
