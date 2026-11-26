"""Real-FST hybrid benchmark: boreal-io decode + marray-mask B2/B4/B6 workloads.

Phases
------
1. **Decode** — ``boreal.io.open_fst`` on live GDPS ``pres`` leads (``datyp=134``).
2. **Compute** — marray_xr vs simulated column-HLG / eager xarray (R-B2, R-B4).
3. **Catalog** — re-pack decoded arenas → M1 ``open_fst_catalog`` (R-B6 style).

Env
---
- ``MARRAY_FST_PRES_ROOT`` — GDPS ``prog/pres`` directory
- ``MARRAY_CMC_IO`` — boreal-io / cmc-io checkout root
- ``MARRAY_ALLOW_LOGIN=1`` — allow heavy runs on login/vis (PBS script gates this)

Usage
-----
::

    pixi run bench-fst-real -- --smoke
    pixi run bench-fst-real -- --leads 4 --mode both
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import dask
import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import fst_real_load as fstload  # noqa: E402
import marray_xr as mxr  # noqa: E402

# Reuse simulated open_fst helpers from the synthetic suite.
from bench_boreal_realistic import (  # noqa: E402
    _med_ms,
    _xr_find_nomvar,
    open_fst_marray_catalog,
    open_fst_simulated_eager,
    open_fst_simulated_hlg,
)


def _print_header(title: str) -> None:
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def run_decode_report(raw: dict[str, Any], timing: fstload.LoadTiming) -> None:
    _print_header("PHASE 0: REAL FST DECODE (boreal-io / cmc_io_fst_data)")
    n_t, n_z, nj, ni = raw["tt_f32"].shape
    unpacked_gb = (
        raw["tt_f32"].nbytes
        + raw["uu_f32"].nbytes
        + raw["vv_f32"].nbytes
        + raw["hu_f32"].nbytes
        + raw["tm_f32"].nbytes
    ) / (1024**3)
    print(f"  leads      : {len(raw['source_paths'])}  {raw['source_paths']}")
    print(f"  grtyp/shape: {raw['grtyp']}  TT={raw['tt_f32'].shape}  (t,z,nj,ni)")
    print(f"  has real TM: {raw.get('has_tm')}  (else fabricated from TT[:,:,0] for pack API)")
    print(f"  unpacked   : {unpacked_gb:.2f} GiB float32 arenas")
    print(f"  open_fst   : {timing.decode_ms:10.1f} ms")
    print(f"  stack/norm : {timing.stack_ms:10.1f} ms")
    print(f"  total load : {timing.total_ms:10.1f} ms")


def run_rb2_polaris_peel(raw: dict[str, Any], *, repeats: int) -> None:
    _print_header("R-B2: polaris thin 2D peel (WindModulus) on REAL arenas")
    ds_rec = open_fst_simulated_hlg(raw, chunks="record")
    ds_col = open_fst_simulated_hlg(raw, chunks="column")
    ds_mxr = open_fst_marray_catalog(raw)
    target = float(raw["vertical"][min(3, len(raw["vertical"]) - 1)])
    t_idx = min(0, raw["tt_f32"].shape[0] - 1)

    def peel_rec() -> np.ndarray:
        u = _xr_find_nomvar(ds_rec, "UU").isel(time=t_idx).sel(vertical=target, method="nearest")
        v = _xr_find_nomvar(ds_rec, "VV").isel(time=t_idx).sel(vertical=target, method="nearest")
        out = np.hypot(u, v) * np.float32(0.514444)
        return out.compute(scheduler="threads", num_workers=8).values

    def peel_col() -> np.ndarray:
        u = _xr_find_nomvar(ds_col, "UU").isel(time=t_idx).sel(vertical=target, method="nearest")
        v = _xr_find_nomvar(ds_col, "VV").isel(time=t_idx).sel(vertical=target, method="nearest")
        out = np.hypot(u, v) * np.float32(0.514444)
        return out.compute(scheduler="threads", num_workers=8).values

    def peel_mxr() -> np.ndarray:
        sub = ds_mxr.isel(time=t_idx).sel(vertical=target, method="nearest")
        uv = np.hypot(sub.find_by_nomvar("UU"), sub.find_by_nomvar("VV")) * 0.514444
        out = uv.compute(fill_value=np.nan, squeeze=("time", "vertical"))
        return np.asarray(out.values)

    ref = peel_rec()
    got = peel_mxr()
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-3, equal_nan=True)
    t_rec = _med_ms(peel_rec, repeats=max(2, repeats // 2))
    t_col = _med_ms(peel_col, repeats=repeats)
    t_mxr = _med_ms(peel_mxr, repeats=repeats + 2)
    print(f"  record HLG : {t_rec:8.2f} ms")
    print(f"  column HLG : {t_col:8.2f} ms")
    print(f"  marray_xr  : {t_mxr:8.2f} ms  --> {t_col / t_mxr:5.1f}x vs column | {t_rec / t_mxr:5.1f}x vs record")


def run_rb4_ke_stat(raw: dict[str, Any], *, repeats: int) -> None:
    _print_header("R-B4: boreal-stat ke + areacell weighted globavg + zonavg (REAL)")
    ds_col = open_fst_simulated_hlg(raw, chunks="column")
    ds_eag = open_fst_simulated_eager(raw)
    ds_mxr = open_fst_marray_catalog(raw)

    def stat_dask() -> tuple[float, np.ndarray]:
        u = _xr_find_nomvar(ds_col, "UU") * np.float32(0.514444)
        v = _xr_find_nomvar(ds_col, "VV") * np.float32(0.514444)
        ke = np.float32(0.5) * (u**2 + v**2)
        w_glob = ke.weighted(ds_col.coords["areacell"]).mean(skipna=True)
        zon = ke.mean(dim=("time", "x"), skipna=True)
        g_res, z_res = dask.compute(w_glob, zon, scheduler="threads", num_workers=8)
        return float(g_res.item()), z_res.values

    def stat_eager() -> tuple[float, np.ndarray]:
        u = _xr_find_nomvar(ds_eag, "UU") * np.float32(0.514444)
        v = _xr_find_nomvar(ds_eag, "VV") * np.float32(0.514444)
        ke = np.float32(0.5) * (u**2 + v**2)
        w_glob = float(ke.weighted(ds_eag.coords["areacell"]).mean(skipna=True).item())
        zon = ke.mean(dim=("time", "x"), skipna=True).values
        return w_glob, zon

    def stat_mxr() -> tuple[float, np.ndarray]:
        u = ds_mxr.find_by_nomvar("UU") * 0.514444
        v = ds_mxr.find_by_nomvar("VV") * 0.514444
        ke = 0.5 * (u**2 + v**2)
        glob_v, zon_da = mxr.compute_many(
            ke.lazy_weighted_mean(ds_mxr.cell_measures["areacell"]),
            ke.lazy_mean(("time", "x")),
        )
        return float(glob_v), np.asarray(zon_da.values)

    g_e, z_e = stat_eager()
    g_m, z_m = stat_mxr()
    np.testing.assert_allclose(g_m, g_e, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(z_m, z_e, rtol=1e-3, atol=1e-3, equal_nan=True)

    t_col = _med_ms(stat_dask, repeats=max(2, repeats // 2))
    t_eag = _med_ms(stat_eager, repeats=repeats)
    t_mxr = _med_ms(stat_mxr, repeats=repeats + 2)
    print(f"  column HLG : {t_col:8.2f} ms")
    print(f"  eager xr   : {t_eag:8.2f} ms")
    print(f"  marray_xr  : {t_mxr:8.2f} ms  --> {t_col / t_mxr:5.1f}x vs column | {t_eag / t_mxr:5.1f}x vs eager  (KE={g_m:.4f})")


def run_rb6_catalog(raw: dict[str, Any], *, repeats: int) -> None:
    _print_header(
        "R-B6: M1 open_fst_catalog on re-packed REAL arenas "
        "(builtin datyp=134 codec; cmc_io cost is Phase 0)"
    )
    t0 = time.perf_counter()
    archive = fstload.pack_decoded_archive(raw, datyp=134)
    pack_ms = (time.perf_counter() - t0) * 1000.0
    ds_list = mxr.open_fst_catalog(archive, split_vctype=True)
    print(f"  pack_ms    : {pack_ms:8.2f} ms  archive={archive.raw_nbytes / (1024**2):.1f} MB")
    print(f"  n_datasets : {len(ds_list)}  vctypes={[d.attrs.get('fst_vctype') for d in ds_list]}")

    n_t, n_z = len(archive.time), len(archive.vertical)
    # Regional window scaled to grid size (keep ~200×240 class when possible).
    nj, ni = archive.nj, archive.ni
    y0, y1 = max(0, nj // 5), min(nj, nj // 5 + min(200, nj // 2))
    x0, x1 = max(0, ni // 5), min(ni, ni // 5 + min(240, ni // 2))
    t1 = min(n_t, max(1, n_t))
    z1 = min(n_z, max(1, min(4, n_z)))

    def full_mxr() -> float:
        uu = mxr.from_sources(ds_list, "UU")
        vv = mxr.from_sources(ds_list, "VV")
        tt = mxr.from_sources(ds_list, "TT")
        expr = np.hypot(uu, vv) + tt
        if raw.get("has_tm") and any(d.attrs.get("fst_vctype") == "surface" for d in ds_list):
            tm = mxr.from_sources(ds_list, "TM")
            expr = np.hypot(uu, vv) + (tm - tt)
        return float(expr.mean())  # type: ignore[arg-type]

    def reg_mxr() -> float:
        hyb = next(d for d in ds_list if d.attrs.get("fst_vctype") in ("hybrid", "merged") or "AirTemp" in d.data_vars)
        sub = hyb.isel(time=slice(0, t1), vertical=slice(0, z1), y=slice(y0, y1), x=slice(x0, x1))
        expr = np.hypot(sub.find_by_nomvar("UU"), sub.find_by_nomvar("VV")) + sub.find_by_nomvar("TT")
        return float(expr.mean())  # type: ignore[arg-type]

    # Parity vs eager arenas on a cheap scalar.
    ds_eag = open_fst_simulated_eager(raw)
    u = _xr_find_nomvar(ds_eag, "UU")
    v = _xr_find_nomvar(ds_eag, "VV")
    tt = _xr_find_nomvar(ds_eag, "TT")
    ref = float((np.hypot(u, v) + tt).mean(skipna=True).item())
    got = full_mxr()
    np.testing.assert_allclose(got, ref, rtol=5e-3, atol=5e-2)

    tf = _med_ms(full_mxr, repeats=repeats)
    tr = _med_ms(reg_mxr, repeats=repeats + 2)
    print(f"  full cube  : {tf:8.2f} ms  (M1 RecordCatalog + PlaneSlotCache)")
    print(f"  regional   : {tr:8.2f} ms  window y=[{y0}:{y1}] x=[{x0}:{x1}] z=[:{z1}]")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=None, help="GDPS prog/pres directory")
    p.add_argument("--leads", type=int, default=1, help="number of lead files to open")
    p.add_argument("--lead-name", type=str, default=None, help="exact lead basename")
    p.add_argument("--max-levels", type=int, default=None, help="cap vertical levels after decode")
    p.add_argument(
        "--mode",
        choices=("arenas", "catalog", "both"),
        default="both",
        help="arenas=R-B2/R-B4, catalog=R-B6, both=all",
    )
    p.add_argument("--smoke", action="store_true", help="1 lead, few levels, fewer repeats")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--cmc-io", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repeats = args.repeats
    max_levels = args.max_levels
    n_leads = args.leads
    if args.smoke:
        n_leads = 1
        repeats = min(repeats, 3)
        if max_levels is None:
            max_levels = 8

    print("=" * 92)
    print("REAL-FST HYBRID BENCHMARK (marray-mask × boreal-io)")
    print(f"NumPy {np.__version__} | Xarray {xr.__version__} | Dask {dask.__version__}")
    print("=" * 92)

    cmc = fstload.ensure_boreal_on_path(args.cmc_io)
    print(f"  cmc-io     : {cmc}")
    leads = fstload.select_leads(args.root, n_leads=n_leads, name=args.lead_name)
    print(f"  pres root  : {leads[0].parent}")
    print(f"  leads      : {[p.name for p in leads]}")
    if max_levels:
        print(f"  max_levels : {max_levels}")

    raw, timing = fstload.load_raw_arenas(
        leads, max_levels=max_levels, cmc_root=cmc
    )
    run_decode_report(raw, timing)

    if args.mode in ("arenas", "both"):
        run_rb2_polaris_peel(raw, repeats=repeats)
        run_rb4_ke_stat(raw, repeats=repeats)
    if args.mode in ("catalog", "both"):
        run_rb6_catalog(raw, repeats=repeats)

    print("\n" + "=" * 92)
    print("DONE")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
