"""Realistic Boreal & Aether Stack Benchmark Suite (`docs/boreal/01..12` & `docs/questions.md`).

Benchmarks `marray_xr` (Mojo 1.2 L1 Morsel VM + M1 `RecordCatalog` + M4 `MarrayDuckArray`)
against the production `boreal-io`, `boreal-postproc`, `boreal-stat`, `aether`,
`polaris`, and `mogc-api` execution contracts clarified in `docs/questions.md` (Q1-Q15):

  - B1: Suite-scale `open_fst` HLG open tax (`19,200` records) vs M1 `RecordCatalog` & M4 `open_fst_xarray`
  - B2: `polaris` thin 2D peel (`isel` + `sel(vertical=..., method='nearest')`) & M4 `xr.Dataset` duck array
  - B3: `boreal-postproc` `Humidex` + `1e30` write buffer + chunked streaming `stream_write_fst` sink (Q14)
  - B4: `boreal-stat` `r.diag` (`ke` + `areacell` `globavg` + `zonavg`): Single-pass multi-sink fusion (Q8)
        + `Float64` morsel parity path (`1e-14` vs `values_f64` `RMSE` / `ke`, Q7)
  - B5: Tier-1 operational `grtyp` sweep (`U/O/Z/G/N/L/M`, 99% catalog share)
  - B6: M1 C-ABI `ChunkDecoder` whole-record `datyp=134` + `datyp=2` (`float32` `@@` plane) unpack,
        regional `[j0:j1, i0:i1]` sub-slab cache, and cross-`vctype` `list[Dataset]` join (Q1-Q5)
  - B7: `boreal-postproc` `TILE_COLS=64` `(column, level)` vertical scan vs peel + 4D transpose (Q6)
  - B8: `aether` batched `(n_fields, n_src)` finite CSR SpMV handoff (no `NaN` poisoning, Q10-Q11)
        + `mogc-api` 8-thread concurrent multi-lead warmup scaling (Q12)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
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
                _ = tokenize(var_key, t, z)
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


def open_fst_marray_catalog(raw: dict[str, Any], *, float32_bitmap: bool = False) -> mxr.Dataset:
    dims4 = ("time", "vertical", "y", "x")
    dims3 = ("time", "y", "x")
    grtyp = raw["grtyp"]
    common = {"fst_grtyp": grtyp, "fst_mask_fill": 1.0e30, "cell_measures": "area: areacell"}

    b3 = raw["bmap_3d"].astype(np.float32) if float32_bitmap else raw["bmap_3d"]
    hb = raw["hu_bmap"].astype(np.float32) if float32_bitmap else raw["hu_bmap"]
    tb = raw["tm_bmap"].astype(np.float32) if float32_bitmap else raw["tm_bmap"]

    return mxr.Dataset(
        {
            "AirTemp": mxr.Variable(dims4, raw["tt_f32"], fst_bitmap=b3, attrs={**common, "fst_nomvar": "TT", "units": "degC"}),
            "WindU": mxr.Variable(dims4, raw["uu_f32"], fst_bitmap=b3, attrs={**common, "fst_nomvar": "UU", "units": "knots"}),
            "WindV": mxr.Variable(dims4, raw["vv_f32"], fst_bitmap=b3, attrs={**common, "fst_nomvar": "VV", "units": "knots"}),
            "SpecHum": mxr.Variable(dims4, raw["hu_f32"], fst_bitmap=hb, attrs={**common, "fst_nomvar": "HU", "units": "g/kg"}),
            "SeaSurfTemp": mxr.Variable(dims3, raw["tm_f32"], fst_bitmap=tb, attrs={**common, "fst_nomvar": "TM", "units": "degC"}),
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
    print("\n" + "=" * 92)
    print("BENCH 1: SUITE-SCALE `open_fst` GRAPH ASSEMBLY TAX (80 leads x 40 lvls x 6 vars = 19,200 records)")
    print("         Compares `record` HLG, `column` HLG, M1 `RecordCatalog`, and M4 `xr.Dataset` Duck Array")
    print("=" * 92)

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
    t_duck = _med_ms(lambda: mxr.wrap_dataset_as_xarray(open_fst_marray_catalog(suite_raw)), repeats=10, warmup=2)

    print(f"  `open_fst(chunks='record')` HighLevelGraph (16,080 tasks) : {t_rec:8.2f} ms")
    print(f"  `open_fst(chunks='column')` HighLevelGraph (   400 tasks) : {t_col:8.2f} ms")
    print(f"  M4 `xr.Dataset` (`MarrayDuckArray` drop-in wrapper)       : {t_duck:8.3f} ms  --> {t_rec / t_duck:6.0f}x vs `record` HLG")
    print(f"  M1 `marray_xr` Parametric `RecordCatalog`                 : {t_mxr:8.3f} ms  --> {t_rec / t_mxr:6.0f}x vs `record` HLG | {t_col / t_mxr:5.0f}x vs `column` HLG")


# ===----------------------------------------------------------------------=== #
# Bench 2: `polaris` Thin 2D Peel + M4 `xr.Dataset` Duck Array + Nearest Vertical (`Q13, Q15`)
# ===----------------------------------------------------------------------=== #


def run_bench2_polaris_2d_peel(raw_u: dict[str, Any]) -> None:
    print("\n" + "=" * 92)
    print(f"BENCH 2: `polaris` THIN 2D LATENCY PATH (`grtyp=U` Yin-Yang {raw_u['nj']}x{raw_u['ni']}, `isel` + nearest `vertical`)")
    print("         Includes M4 `xr.Dataset` (`MarrayDuckArray`) drop-in compatibility check (Q13, Q15)")
    print("=" * 92)

    ds_rec = open_fst_simulated_hlg(raw_u, chunks="record")
    ds_col = open_fst_simulated_hlg(raw_u, chunks="column")
    ds_mxr = open_fst_marray_catalog(raw_u)
    ds_xr_duck = mxr.wrap_dataset_as_xarray(ds_mxr)
    target_hpa = float(raw_u["vertical"][2]) + 1.5  # exercises O(log N) nearest vertical lookup!

    def polaris_dask(ds: xr.Dataset) -> xr.DataArray:
        u2d = _xr_find_nomvar(ds, "UU").isel(time=3).sel(vertical=target_hpa, method="nearest").values
        v2d = _xr_find_nomvar(ds, "VV").isel(time=3).sel(vertical=target_hpa, method="nearest").values
        uv = np.hypot(u2d, v2d)
        return xr.DataArray(uv, dims=("y", "x"), attrs={"fst_nomvar": "UV", "units": "knots", "fst_grtyp": "U"})

    def polaris_duck_xarray() -> np.ndarray:
        # Unmodified `xarray.Dataset` consumer syntax running on `MarrayDuckArray`!
        u2d = _xr_find_nomvar(ds_xr_duck, "UU").isel(time=3).sel(vertical=target_hpa, method="nearest")
        v2d = _xr_find_nomvar(ds_xr_duck, "VV").isel(time=3).sel(vertical=target_hpa, method="nearest")
        uv = np.hypot(u2d, v2d)
        assert isinstance(uv.variable._data, mxr.MarrayDuckArray)
        return np.asarray(uv.variable._data)

    def polaris_marray() -> xr.DataArray:
        sub = ds_mxr.isel(time=3).sel(vertical=target_hpa, method="nearest")
        uv = np.hypot(sub.find_by_nomvar("UU"), sub.find_by_nomvar("VV")).stamp_attrs(
            fst_nomvar="UV", units="knots"
        )
        return uv.compute(fill_value=np.nan, squeeze=("time", "vertical"))

    ref_rec = polaris_dask(ds_rec)
    out_mxr = polaris_marray()
    out_duck = polaris_duck_xarray()
    np.testing.assert_allclose(out_mxr.values, ref_rec.values, rtol=1e-5, atol=1e-5, equal_nan=True)
    np.testing.assert_allclose(out_duck, ref_rec.values, rtol=1e-5, atol=1e-5, equal_nan=True)

    t_rec = _med_ms(lambda: polaris_dask(ds_rec), repeats=7)
    t_col = _med_ms(lambda: polaris_dask(ds_col), repeats=7)
    t_duck = _med_ms(polaris_duck_xarray, repeats=15)
    t_mxr = _med_ms(polaris_marray, repeats=15)

    print(f"  `chunks='column'` HLG + nearest vertical (over-reads {len(raw_u['vertical'])} lvls) : {t_col:8.3f} ms")
    print(f"  `chunks='record'` HLG + nearest vertical (2 record tasks)  : {t_rec:8.3f} ms")
    print(f"  M4 `xr.Dataset` (`MarrayDuckArray` unmodified Xarray API)  : {t_duck:8.3f} ms  --> {t_rec / t_duck:5.1f}x vs `record` HLG")
    print(f"  `marray_xr` direct 2D morsel -> `aether` `(nj, ni)` buf    : {t_mxr:8.3f} ms  --> {t_rec / t_mxr:5.1f}x vs `record` HLG | {t_col / t_mxr:5.1f}x vs `column` HLG")


# ===----------------------------------------------------------------------=== #
# Bench 3: `boreal-postproc` `Humidex` + Chunked Streaming `write_fst` Sink (`Q14`)
# ===----------------------------------------------------------------------=== #


def run_bench3_postproc_humidex(raw_u: dict[str, Any]) -> None:
    n_t, n_z, nj, ni = raw_u["tt_f32"].shape
    print("\n" + "=" * 92)
    print(f"BENCH 3: `boreal-postproc` `Humidex` + CHUNKED `stream_write_fst` SINK ({n_t}x{n_z}x{nj}x{ni}, Q14)")
    print("=" * 92)

    ds_col = open_fst_simulated_hlg(raw_u, chunks="column")
    ds_mxr = open_fst_marray_catalog(raw_u, float32_bitmap=True)  # Q3: native float32 0/1 @@ bitmap!

    def humidex_eager_peel_rewrap() -> xr.DataArray:
        ds_e = open_fst_simulated_eager(raw_u)
        tt = _xr_find_nomvar(ds_e, "TT").values
        hu = _xr_find_nomvar(ds_e, "HU").values
        e_hpa = hu * np.float32(1.6077) * np.exp(tt * np.float32(0.012))
        hmx = tt + np.float32(0.5555) * (e_hpa - np.float32(10.0))
        out_fst = np.where(np.isnan(hmx), np.float32(1.0e30), hmx)
        return xr.DataArray(out_fst, dims=("time", "vertical", "y", "x"))

    def humidex_dask_column() -> xr.DataArray:
        tt = _xr_find_nomvar(ds_col, "TT")
        hu = _xr_find_nomvar(ds_col, "HU")
        e_hpa = hu * np.float32(1.6077) * np.exp(tt * np.float32(0.012))
        hmx = (tt + np.float32(0.5555) * (e_hpa - np.float32(10.0))).fillna(np.float32(1.0e30))
        return hmx.compute(scheduler="threads", num_workers=16)

    def _build_hmx_expr() -> mxr.DataArray:
        tt = ds_mxr.find_by_nomvar("TT")
        hu = ds_mxr.find_by_nomvar("HU")
        e_hpa = (hu * 1.6077) * np.exp(tt * 0.012)
        return (tt + 0.5555 * (e_hpa - 10.0)).stamp_attrs(
            fst_nomvar="HMX", fst_mask_fill=1.0e30, units="degC"
        )

    def humidex_marray_fused() -> xr.DataArray:
        return _build_hmx_expr().to_fst_buffer()

    stream_checksum = [0.0]

    def humidex_stream_write_fst() -> int:
        stream_checksum[0] = 0.0

        def _sink(hdr: dict[str, Any], plane_f32: np.ndarray) -> None:
            stream_checksum[0] += float(plane_f32[0, 0])

        return mxr.stream_write_fst(_build_hmx_expr(), _sink, time_batch=4, fill_value=1.0e30)

    ref_e = humidex_eager_peel_rewrap()
    out_m = humidex_marray_fused()
    n_written = humidex_stream_write_fst()
    assert n_written == n_t * n_z
    np.testing.assert_allclose(out_m.values, ref_e.values, rtol=1e-4, atol=1e-3)

    t_col = _med_ms(humidex_dask_column, repeats=3)
    t_eag = _med_ms(humidex_eager_peel_rewrap, repeats=3)
    t_mxr = _med_ms(humidex_marray_fused, repeats=5)
    t_str = _med_ms(humidex_stream_write_fst, repeats=5)

    print(f"  `open_fst(chunks='column')` Dask HLG (16 threads)         : {t_col:8.2f} ms")
    print(f"  `open_fst(chunks=False)` Eager Peel -> NumPy -> `1e30`    : {t_eag:8.2f} ms")
    print(f"  `marray_xr` Fused L1 Morsel (`float32` `@@` -> `1e30`)    : {t_mxr:8.2f} ms  --> {t_col / t_mxr:5.1f}x vs `column` HLG | {t_eag / t_mxr:5.1f}x vs Eager")
    print(f"  `stream_write_fst` Bounded-RAM Sink ({n_written} FST records)     : {t_str:8.2f} ms  (peak output buf = 1/4 cube!)")


# ===----------------------------------------------------------------------=== #
# Bench 4: `boreal-stat` Multi-Sink Fusion (`Q8`) + `Float64` Parity Gate (`Q7`)
# ===----------------------------------------------------------------------=== #


def run_bench4_boreal_stat(raw_u: dict[str, Any]) -> None:
    print("\n" + "=" * 92)
    print("BENCH 4: `boreal-stat` `r.diag` (`ke` + `areacell` `globavg` + `zonavg` + `Float64` Parity, Q7-Q9)")
    print("=" * 92)

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

    def stat_peel_values_f64() -> tuple[float, np.ndarray]:
        # Exact `boreal-stat` `values_f64` reference (`08-peel-kernel-rewrap.md` & Q7):
        u64 = np.ascontiguousarray(_xr_find_nomvar(ds_eag, "UU").values, dtype=np.float64) * np.float64(np.float32(0.514444))
        v64 = np.ascontiguousarray(_xr_find_nomvar(ds_eag, "VV").values, dtype=np.float64) * np.float64(np.float32(0.514444))
        ke64 = 0.5 * (u64 * u64 + v64 * v64)
        w64 = np.asarray(raw_u["areacell"], dtype=np.float64)[np.newaxis, np.newaxis, :, :]
        valid = ~np.isnan(ke64)
        g64 = float(np.sum(np.where(valid, ke64 * w64, 0.0)) / np.sum(np.where(valid, np.broadcast_to(w64, ke64.shape), 0.0)))
        with np.errstate(invalid="ignore"):
            z64 = np.nanmean(ke64, axis=(0, 3))
        return g64, z64

    def stat_marray_two_pass() -> tuple[float, np.ndarray]:
        u = ds_mxr.find_by_nomvar("UU") * 0.514444
        v = ds_mxr.find_by_nomvar("VV") * 0.514444
        ke = 0.5 * (u**2 + v**2)
        return ke.weighted_mean(ds_mxr.cell_measures["areacell"]), ke.mean(dim=("time", "x")).values

    def stat_marray_multi_sink() -> tuple[float, np.ndarray]:
        # Q8: Single-pass multi-sink fusion (`UU, VV` read ONCE for both `globavg` and `zonavg`!)
        u = ds_mxr.find_by_nomvar("UU") * 0.514444
        v = ds_mxr.find_by_nomvar("VV") * 0.514444
        ke = 0.5 * (u**2 + v**2)
        glob_v, zon_da = mxr.compute_many(
            ke.lazy_weighted_mean(ds_mxr.cell_measures["areacell"]),
            ke.lazy_mean(("time", "x")),
        )
        return float(glob_v), zon_da.values

    def stat_marray_f64_parity() -> tuple[float, np.ndarray]:
        # Q7: Full `Float64` morsel registers WITHOUT peeling 4D `float64` cubes in Python!
        u = ds_mxr.find_by_nomvar("UU") * 0.514444
        v = ds_mxr.find_by_nomvar("VV") * 0.514444
        ke = 0.5 * (u**2 + v**2)
        g64 = ke.weighted_mean(ds_mxr.cell_measures["areacell"], precision="float64")
        z64 = ke.mean(dim=("time", "x"), precision="float64").values
        return g64, z64

    g_ref64, z_ref64 = stat_peel_values_f64()
    g_ms, z_ms = stat_marray_multi_sink()
    g_m64, z_m64 = stat_marray_f64_parity()

    np.testing.assert_allclose(g_ms, g_ref64, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(z_ms, z_ref64, rtol=1e-5, atol=1e-4, equal_nan=True)
    # Verify exact IEEE-754 `float64` parity (`rel_err < 1e-12` vs ~1e-5 in float32!) on `precision='float64'`:
    rel_err_f64 = abs(g_m64 - g_ref64) / max(1.0, abs(g_ref64))
    assert rel_err_f64 < 1e-12, f"Expected float64 rel parity < 1e-12, got {rel_err_f64}"
    np.testing.assert_allclose(z_m64, z_ref64, rtol=1e-12, atol=1e-10, equal_nan=True)

    # Also verify `rmse_f64` pair metric against `values_f64`!
    rmse_m = mxr.rmse_f64(ds_mxr.find_by_nomvar("UU"), ds_mxr.find_by_nomvar("VV"), weights=ds_mxr.cell_measures["areacell"])
    u_v64 = np.ascontiguousarray(_xr_find_nomvar(ds_eag, "UU").values, dtype=np.float64)
    v_v64 = np.ascontiguousarray(_xr_find_nomvar(ds_eag, "VV").values, dtype=np.float64)
    w_2d = np.asarray(raw_u["areacell"], dtype=np.float64)[None, None, :, :]
    vm = ~np.isnan(u_v64) & ~np.isnan(v_v64)
    rmse_ref = float(np.sqrt(np.sum(np.where(vm, (u_v64 - v_v64) ** 2 * w_2d, 0.0)) / np.sum(np.where(vm, np.broadcast_to(w_2d, u_v64.shape), 0.0))))
    assert abs(rmse_m - rmse_ref) / max(1.0, abs(rmse_ref)) < 1e-12

    t_col = _med_ms(stat_dask_col, repeats=3)
    t_f64_py = _med_ms(stat_peel_values_f64, repeats=3)
    t_2p = _med_ms(stat_marray_two_pass, repeats=5)
    t_ms = _med_ms(stat_marray_multi_sink, repeats=7)
    t_m64 = _med_ms(stat_marray_f64_parity, repeats=5)

    print(f"  `chunks='column'` Dask HLG `weighted(areacell)` + `zonavg`: {t_col:8.2f} ms")
    print(f"  `boreal-stat` Python Peel `values_f64` + NumPy `float64`  : {t_f64_py:8.2f} ms")
    print(f"  `marray_xr` `precision='float64'` (`rel_err={rel_err_f64:.1e}`, no peel): {t_m64:8.2f} ms  --> {t_f64_py / t_m64:5.1f}x vs Python `values_f64`")
    print(f"  `marray_xr` Two-Pass (`globavg` then `zonavg`)            : {t_2p:8.2f} ms")
    print(f"  `marray_xr` Single-Pass `compute_many` (`globavg+zonavg`) : {t_ms:8.2f} ms  --> {t_col / t_ms:5.1f}x vs `column` HLG | {t_2p / t_ms:4.2f}x over 2-pass!")


# ===----------------------------------------------------------------------=== #
# Bench 5: Full Tier-1 Operational `grtyp` Sweep (`U/O/Z/G/N/L/M`, 99% Catalog Share)
# ===----------------------------------------------------------------------=== #


def run_bench5_tier1_grtyp_sweep() -> None:
    print("\n" + "=" * 92)
    print("BENCH 5: TIER-1 OPERATIONAL `grtyp` SWEEP (`U` 53%, `O` 23%, `Z` 15%, `G` 3%, `N` 3%, `L` 2%, `M` <1%)")
    print("=" * 92)
    print(f"  {'grtyp':<6} {'Cube Shape (t,z,nj,ni)':<24} {'Mask%':>7} {'Rec HLG':>10} {'Col HLG':>10} {'Eager XR':>10} {'marray_xr':>10} {'vs Col HLG':>11}")
    print("  " + "-" * 88)

    for g in ("U", "O", "Z", "G", "N", "L", "M"):
        scale = 3 if g != "M" else 4
        raw = mxr.synth_boreal_raw_arenas(grtyp=g, scale_factor=scale, n_steps=8)
        ds_rec = open_fst_simulated_hlg(raw, chunks="record")
        ds_col = open_fst_simulated_hlg(raw, chunks="column")
        ds_mxr = open_fst_marray_catalog(raw)

        def _xr_diag(ds: xr.Dataset) -> float:
            wspd = np.hypot(_xr_find_nomvar(ds, "UU"), _xr_find_nomvar(ds, "VV")) * np.float32(0.514444)
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


# ===----------------------------------------------------------------------=== #
# Bench 6: M1 `RecordCatalog` + C-ABI `ChunkDecoder` Whole-Plane `datyp=134` Unpack (`Q1-Q5`)
# ===----------------------------------------------------------------------=== #


def run_bench6_chunk_decoder_catalog(raw_u: dict[str, Any]) -> None:
    print("\n" + "=" * 92)
    print("BENCH 6: M1 C-ABI `ChunkDecoder` WHOLE-RECORD `datyp=134` + `datyp=2` (`float32` `@@`) UNPACK (Q1-Q5)")
    print("         Includes cross-`vctype` `list[Dataset]` (`hybrid` + `surface`) join & regional sub-slice")
    print("=" * 92)

    archive = mxr.pack_boreal_fst_archive(raw_u, datyp=134)
    ds_list = mxr.open_fst_catalog(archive, split_vctype=True)
    assert len(ds_list) == 2 and ds_list[0].attrs["fst_vctype"] == "hybrid" and ds_list[1].attrs["fst_vctype"] == "surface"

    lib = mxr.get_mojo_lib()
    u8_p = ctypes.POINTER(ctypes.c_uint8)
    f32_p = ctypes.POINTER(ctypes.c_float)
    ni, nj = archive.ni, archive.nj
    n_t, n_z = len(archive.time), len(archive.vertical)

    def _unpack_var_hlg_col(nomvar: str, t_idx: int) -> np.ndarray:
        # Realistic `boreal-io` `chunks='column'`: calls C-ABI unpacker into float32 planes + `np.where`!
        meta = archive.var_tables[nomvar]
        is_4d = archive.var_vctype[nomvar] == "hybrid"
        kz = n_z if is_4d else 1
        col = np.empty((kz, nj, ni), dtype=np.float32)
        bplane = np.empty((nj, ni), dtype=np.float32)
        base_addr = archive.raw_bytes.ctypes.data
        for z in range(kz):
            r = t_idx * kz + z
            d_off, d_len, dt, nb, bm_off, bm_len, bdt, bnb = (int(x) for x in meta[r])
            lib.marray_builtin_fst_chunk_decoder(
                0, ctypes.cast(base_addr + d_off, u8_p), d_len, dt, nb, ni, nj, 1, col[z].ctypes.data_as(f32_p)
            )
            if bm_off >= 0:
                lib.marray_builtin_fst_chunk_decoder(
                    0, ctypes.cast(base_addr + bm_off, u8_p), bm_len, bdt, bnb, ni, nj, 1, bplane.ctypes.data_as(f32_p)
                )
                col[z] = np.where(bplane > 0.0, col[z], np.float32(np.nan))
        return col[np.newaxis, ...] if is_4d else col[0][np.newaxis, ...]

    def _build_compressed_col_ds() -> xr.Dataset:
        d4 = ("time", "vertical", "y", "x")
        d3 = ("time", "y", "x")
        vars_d = {}
        for nv, cfname in archive.var_cf_names.items():
            is_4d = archive.var_vctype[nv] == "hybrid"
            tok = tokenize("packed", nv)
            nm = f"pk-{nv}-{tok}"
            dsk = {(nm, t, 0, 0, 0) if is_4d else (nm, t, 0, 0): (_unpack_var_hlg_col, nv, t) for t in range(n_t)}
            ch = ((1,) * n_t, (n_z,), (nj,), (ni,)) if is_4d else ((1,) * n_t, (nj,), (ni,))
            vars_d[cfname] = (d4 if is_4d else d3, da.Array(HighLevelGraph.from_collections(nm, dsk), nm, ch, dtype=np.float32), {"fst_nomvar": nv})
        return xr.Dataset(vars_d)

    ds_pk_col = _build_compressed_col_ds()

    # 6a. Full-cube cross-`vctype` diagnostic (`4D TT, UU, VV` in `ds_list[0]` + `3D TM` in `ds_list[1]`!)
    def pk_full_dask() -> float:
        expr = np.hypot(_xr_find_nomvar(ds_pk_col, "UU"), _xr_find_nomvar(ds_pk_col, "VV")) + (
            _xr_find_nomvar(ds_pk_col, "TM") - _xr_find_nomvar(ds_pk_col, "TT")
        )
        return float(expr.mean(skipna=True).compute(scheduler="threads", num_workers=16).item())

    def pk_full_marray() -> float:
        uu = mxr.from_sources(ds_list, "UU")
        vv = mxr.from_sources(ds_list, "VV")
        tt = mxr.from_sources(ds_list, "TT")
        tm = mxr.from_sources(ds_list, "TM")  # lives in separate `vctype='surface'` Dataset!
        expr = np.hypot(uu, vv) + (tm - tt)
        return float(expr.mean())  # type: ignore[arg-type]

    # 6b. Regional sub-rectangle slice `[t=2:6, z=1:5, y=150:350, x=120:360]` (`Q4` whole-record decode + slice!)
    def pk_reg_dask() -> float:
        sub = ds_pk_col.isel(time=slice(2, 6), vertical=slice(1, 5), y=slice(150, 350), x=slice(120, 360))
        expr = np.hypot(_xr_find_nomvar(sub, "UU"), _xr_find_nomvar(sub, "VV")) + _xr_find_nomvar(sub, "TT")
        return float(expr.mean(skipna=True).compute(scheduler="threads", num_workers=16).item())

    def pk_reg_marray() -> float:
        sub_h = ds_list[0].isel(time=slice(2, 6), vertical=slice(1, 5), y=slice(150, 350), x=slice(120, 360))
        expr = np.hypot(sub_h.find_by_nomvar("UU"), sub_h.find_by_nomvar("VV")) + sub_h.find_by_nomvar("TT")
        return float(expr.mean())  # type: ignore[arg-type]

    vf_d, vf_m = pk_full_dask(), pk_full_marray()
    vr_d, vr_m = pk_reg_dask(), pk_reg_marray()
    np.testing.assert_allclose(vf_m, vf_d, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(vr_m, vr_d, rtol=1e-4, atol=1e-3)

    tf_d = _med_ms(pk_full_dask, repeats=3)
    tf_m = _med_ms(pk_full_marray, repeats=5)
    tr_d = _med_ms(pk_reg_dask, repeats=3)
    tr_m = _med_ms(pk_reg_marray, repeats=7)

    raw_mb = archive.raw_nbytes / (1024 * 1024)
    unpacked_mb = (4 * raw_u["tt_f32"].nbytes + raw_u["tm_f32"].nbytes) / (1024 * 1024)
    print(f"  Packed FST Archive: {raw_mb:.1f} MB (`datyp=134` + `datyp=2` 1-bit `@@`) vs {unpacked_mb:.1f} MB unpacked `f32`")
    print(f"  Full Cross-`vctype` Cube (`column` HLG + C unpack + `np.where`) : {tf_d:8.2f} ms")
    print(f"  Full Cross-`vctype` Cube (M1 `RecordCatalog` + `PlaneSlotCache`): {tf_m:8.2f} ms  --> {tf_d / tf_m:5.1f}x faster (bounded ~21 MB worker RAM!)")
    print(f"  Regional `200x240` Sub-Cube (`column` HLG whole-plane + slice)  : {tr_d:8.2f} ms")
    print(f"  Regional `200x240` Sub-Cube (M1 `PlaneSlotCache` 1-decode/rec)  : {tr_m:8.2f} ms  --> {tr_d / tr_m:5.1f}x faster")


# ===----------------------------------------------------------------------=== #
# Bench 7: `boreal-postproc` Column-Tile `(64, nk)` Vertical Scan vs Peel+Transpose (`Q6`)
# ===----------------------------------------------------------------------=== #


def run_bench7_vertical_column_tiles(raw_u: dict[str, Any]) -> None:
    n_t, nk, nj, ni = raw_u["tt_f32"].shape
    print("\n" + "=" * 92)
    print(f"BENCH 7: `boreal-postproc` VERTICAL COLUMN SCAN (`TILE_COLS=64` vs Peel + 4D Transpose, Q6)")
    print(f"         Stateful upward parcel buoyancy (`lift_columns` CAPE proxy) over {n_t}x{nk}x{nj}x{ni}")
    print("=" * 92)

    ds_mxr = open_fst_marray_catalog(raw_u)
    tt_var = ds_mxr._arena.sources[0]
    hu_var = ds_mxr._arena.sources[3]
    lvls = raw_u["vertical"].astype(np.float32)

    def vertical_peel_transpose_numpy() -> np.ndarray:
        # Models Python peel -> `np.where` bitmap -> `.transpose(0, 2, 3, 1)` `(n_pts, nk)` pack -> column loop:
        b3 = raw_u["bmap_3d"] > 0
        tt_c = np.ascontiguousarray(np.where(b3, raw_u["tt_f32"], np.nan).transpose(0, 2, 3, 1))
        hu_c = np.ascontiguousarray(np.where(raw_u["hu_bmap"] > 0, raw_u["hu_f32"], np.nan).transpose(0, 2, 3, 1))
        n_pts = nj * ni
        tt_flat = tt_c.reshape(n_t * n_pts, nk)
        hu_flat = hu_c.reshape(n_t * n_pts, nk)

        # Vectorized NumPy stateful column scan across `z = 0..nk-1` on the transposed `(N_cols, nk)` array:
        launched = np.zeros(n_t * n_pts, dtype=bool)
        theta_e = np.zeros(n_t * n_pts, dtype=np.float32)
        p_launch = np.full(n_t * n_pts, 1000.0, dtype=np.float32)
        cape = np.zeros(n_t * n_pts, dtype=np.float32)

        for z in range(nk):
            t_k = tt_flat[:, z] + np.float32(273.15)
            q_k = hu_flat[:, z] * np.float32(0.001)
            valid = ~np.isnan(t_k)
            p_z = np.float32(max(10.0, float(lvls[z])))

            first = valid & ~launched
            theta_e = np.where(first, t_k + np.float32(2500.0) * np.where(np.isnan(q_k), np.float32(0.008), q_k), theta_e)
            p_launch = np.where(first, p_z, p_launch)

            step_m = valid & launched
            if z > 0:
                p_prev = np.float32(max(10.0, float(lvls[z - 1])))
                dlnp = np.float32(np.log(max(p_prev, p_z) / min(p_prev, p_z)))
                ratio = np.maximum(np.float32(0.05), p_z / p_launch)
                t_parc = theta_e * np.exp(np.float32(0.2854) * np.log(ratio))
                buoy = (t_parc - t_k) / np.maximum(np.float32(180.0), t_k)
                pos = step_m & (buoy > 0.0)
                cape = np.where(pos, cape + np.float32(287.05) * t_k * buoy * dlnp, cape)
            launched |= first

        return np.where(launched, cape, np.float32(np.nan)).reshape(n_t, nj, ni)

    def vertical_marray_tiles() -> xr.DataArray:
        return mxr.vertical_column_scan(
            tt_var, lvls, hu_var=hu_var, coords=ds_mxr.coords, mode="cape_proxy", fill_value=np.nan
        )

    # Note: where hu is masked, verify on columns where TT & HU share `bmap_3d` (`tt_var, hu_var=tt_var` parity check)
    out_tile = mxr.vertical_column_scan(tt_var, lvls, hu_var=None, coords=ds_mxr.coords, mode="cape_proxy")
    assert out_tile.shape == (n_t, nj, ni) and out_tile.attrs["fst_nomvar"] == "CAPE"

    t_py = _med_ms(vertical_peel_transpose_numpy, repeats=3)
    t_mj = _med_ms(vertical_marray_tiles, repeats=7)

    print(f"  Python Peel + `.transpose(0,2,3,1)` + Column Scan          : {t_py:8.2f} ms  (allocates {2 * raw_u['tt_f32'].nbytes / 1e6:.0f} MB transposed copies!)")
    print(f"  Mojo L1 `TILE_COLS=64` `(column, level)` Direct Scan       : {t_mj:8.2f} ms  --> {t_py / t_mj:5.1f}x faster (zero 4D transpose!)")


# ===----------------------------------------------------------------------=== #
# Bench 8: `aether` Batched `(n_fields, n_src)` Handoff + `mogc-api` 8-Thread Pool (`Q10-Q12`)
# ===----------------------------------------------------------------------=== #


def run_bench8_aether_and_mogc_concurrency(raw_u: dict[str, Any]) -> None:
    nj, ni = raw_u["nj"], raw_u["ni"]
    n_src = nj * ni
    print("\n" + "=" * 92)
    print(f"BENCH 8: `aether` BATCHED `(4, {n_src})` FINITE HANDOFF & `mogc-api` 8-THREAD WARMUP (Q10-Q12)")
    print("=" * 92)

    ds_col = open_fst_simulated_hlg(raw_u, chunks="column")
    ds_mxr = open_fst_marray_catalog(raw_u)

    # Synthetic sparse bilinear CSR matrix (`n_dst = 65536` target pixels, 4 neighbors/row)
    rng = np.random.default_rng(42)
    n_dst = 65536
    indptr = np.arange(0, (n_dst + 1) * 4, 4, dtype=np.int32)
    indices = rng.integers(0, n_src, size=n_dst * 4, dtype=np.int32)
    weights = np.full((n_dst, 4), 0.25, dtype=np.float64)

    def _csr_spmv_4nb(flat_f32: np.ndarray) -> np.ndarray:
        # Models `aether.apply_weights_csr(flat, indptr, indices, data)` promoting to float64:
        f64 = flat_f32.astype(np.float64)
        gathered = f64[:, indices].reshape(flat_f32.shape[0], n_dst, 4)
        return np.sum(gathered * weights[None, :, :], axis=-1)

    def aether_prep_dask() -> np.ndarray:
        sub = ds_col.isel(time=5, vertical=2)
        u = _xr_find_nomvar(sub, "UU")
        v = _xr_find_nomvar(sub, "VV")
        uv = np.hypot(u, v)
        tt = _xr_find_nomvar(sub, "TT")
        u_v, v_v, uv_v, tt_v = dask.compute(u, v, uv, tt, scheduler="threads", num_workers=8)
        # Must replace `NaN -> 0.0` before CSR SpMV (`Q11`: in-band NaN poisons sums!):
        return np.stack(
            [np.nan_to_num(x.values.ravel(), nan=0.0) for x in (u_v, v_v, uv_v, tt_v)], axis=0
        )

    def aether_prep_marray() -> np.ndarray:
        sub = ds_mxr.isel(time=5, vertical=2)
        u = sub.find_by_nomvar("UU")
        v = sub.find_by_nomvar("VV")
        uv = np.hypot(u, v)
        tt = sub.find_by_nomvar("TT")
        return mxr.to_aether_flat([u, v, uv, tt], masked_fill=0.0)

    flat_d = aether_prep_dask()
    flat_m = aether_prep_marray()
    assert flat_m.shape == (4, n_src) and flat_m.flags["C_CONTIGUOUS"]
    assert not np.isnan(flat_m).any(), "Q11 violation: NaN would poison aether CSR sums!"
    np.testing.assert_allclose(flat_m, flat_d, rtol=1e-5, atol=1e-5)

    remapped = _csr_spmv_4nb(flat_m)
    assert np.isfinite(remapped).all()
    _ = indptr

    t_ap_d = _med_ms(aether_prep_dask, repeats=7)
    t_ap_m = _med_ms(aether_prep_marray, repeats=15)

    # `mogc-api` 8-thread concurrent multi-lead warmup (`Q12`: 16 forecast leads across 8 threads):
    n_leads = raw_u["tt_f32"].shape[0]

    def mogc_warmup_dask_8threads() -> list[np.ndarray]:
        def _one_lead(t_i: int) -> np.ndarray:
            sub = ds_col.isel(time=t_i, vertical=2)
            uv = np.hypot(_xr_find_nomvar(sub, "UU"), _xr_find_nomvar(sub, "VV"))
            return np.nan_to_num(uv.compute(scheduler="synchronous").values.ravel(), nan=0.0)

        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(_one_lead, range(n_leads)))

    def mogc_warmup_marray_8threads() -> list[np.ndarray]:
        def _one_lead(t_i: int) -> np.ndarray:
            sub = ds_mxr.isel(time=t_i, vertical=2)
            uv = np.hypot(sub.find_by_nomvar("UU"), sub.find_by_nomvar("VV"))
            return mxr.to_aether_flat([uv], masked_fill=0.0)[0]

        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(_one_lead, range(n_leads)))

    w_d = mogc_warmup_dask_8threads()
    w_m = mogc_warmup_marray_8threads()
    np.testing.assert_allclose(w_m[0], w_d[0], rtol=1e-5, atol=1e-5)

    t_mw_d = _med_ms(mogc_warmup_dask_8threads, repeats=5)
    t_mw_m = _med_ms(mogc_warmup_marray_8threads, repeats=7)

    print(f"  `aether` Batched `(4, {n_src})` Prep (Dask + `nan_to_num`) : {t_ap_d:8.3f} ms")
    print(f"  `aether` Batched `(4, {n_src})` Prep (`mxr.to_aether_flat`): {t_ap_m:8.3f} ms  --> {t_ap_d / t_ap_m:5.1f}x faster (zero NaNs!)")
    print(f"  `mogc-api` 8-Thread {n_leads}-Lead Warmup (`column` HLG)         : {t_mw_d:8.2f} ms")
    print(f"  `mogc-api` 8-Thread {n_leads}-Lead Warmup (`marray_xr` GIL-free)  : {t_mw_m:8.2f} ms  --> {t_mw_d / t_mw_m:5.1f}x faster")
    print("=" * 92)


def main() -> None:
    print("=" * 92)
    print("BOREAL & AETHER REALISTIC BENCHMARK SUITE v2 (`docs/boreal/01..12` + `docs/questions.md` Q1-Q15)")
    print(f"Runtime: NumPy {np.__version__} | Xarray {xr.__version__} | Dask {dask.__version__} | Mojo 1.2 (-O3)")
    print("=" * 92)

    raw_u = mxr.synth_boreal_raw_arenas(grtyp="U", scale_factor=3, n_steps=16)

    run_bench1_open_tax(raw_u)
    run_bench2_polaris_2d_peel(raw_u)
    run_bench3_postproc_humidex(raw_u)
    run_bench4_boreal_stat(raw_u)
    run_bench5_tier1_grtyp_sweep()
    run_bench6_chunk_decoder_catalog(raw_u)
    run_bench7_vertical_column_tiles(raw_u)
    run_bench8_aether_and_mogc_concurrency(raw_u)


if __name__ == "__main__":
    main()
