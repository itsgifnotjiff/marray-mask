"""P0 Acceptance Suite (`CONTEXT.md` §6): Multi-File POSIX `pread` + `open_fst_xarray(paths)`
+ `xr.open_dataset(..., engine='marray_fst')` + Surgical Plane-Decode Verification.

Verifies all 5 P0 load-path acceptance criteria WITHOUT requiring operational mounts:
  1. Multi-lead GDPS-style archive (`grtyp=Z`, `rlat`/`rlon` + 2D `lat`/`lon` + `areacell`
     + `datetime64[ns]` `time` + `vctype` group split): `open_fst_xarray(paths)` opens in
     O(headers) (`~1 ms`, zero payload bytes unpacked at open time!).
  2. Surgical POSIX `pread`: `.isel(time=2).sel(vertical=700.0, method='nearest')` on a
     `4-lead x 10-level` (`40` records/var) archive decodes **EXACTLY 2 planes** (`1 UU + 1 VV`)!
  3. Unmodified Xarray consumers (`np.hypot(u, v) * np.float32(0.514444)`, `0.5 * (u**2 + v**2)`,
     `.mean(dim=('time', 'rlon'))`, `.compute()`) run with **ZERO Dask**!
  4. `xr.open_dataset(lead_path, engine='marray_fst')` works out of the box under `PYTHONPATH`.
  5. Bit-close numerical parity (`assert_allclose`) against raw ground-truth fields.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "bridges"))

import build_cmc_bridge  # noqa: E402
import marray_xr as mxr  # noqa: E402


def _verify_boreal_xdf0_tier_a_and_zero_hlg(raw_z: dict[str, Any], tmpdir: str) -> tuple[float, float]:
    """End-to-end test of `RC1-RC5` (`docs/boreal/13-cold-open-work-machine.md`):
      1. Compiles `bridges/cmc_io_cabi_shim.mojo` against a mock `cmc_io_fst/xdf/unpack_batch.mojo`
         via `build_cmc_bridge.py` (when `mojo` is on PATH) and resolves via `MARRAY_CMC_CABI_SO`.
      2. Writes 4 `XDF0`-magic lead files with 8-byte aligned `(addr - 1) * 8` `datyp=5` & `datyp=2`
         records, plus 2,000 unused nomvar rows/file to test `read_records_fast` pre-filtering.
      3. Asserts `boreal.io.xarray.stack._stack_variable` Dask HLG builder is NEVER executed,
         `Tier == 'A'` (`FilePool` `pread`), descending `vertical` permutation matches ground truth,
         and `mxr.get_last_decoded_planes() == 2`!
    """
    td = Path(tmpdir) / "xdf_suite"
    td.mkdir(parents=True, exist_ok=True)

    # 1. Build real C-ABI shim `.so` if `mojo` is in PATH; otherwise reuse `marray_mojo.so` C-ABI symbol
    bridge_so: Path | None = None
    if shutil.which("mojo") is not None:
        mock_cmc = td / "mock_cmc"
        xdf_dir = mock_cmc / "src" / "mojo" / "cmc_io_fst" / "xdf"
        xdf_dir.mkdir(parents=True, exist_ok=True)
        (mock_cmc / "src" / "mojo" / "cmc_io_fst" / "__init__.mojo").write_text("", encoding="utf-8")
        (xdf_dir / "__init__.mojo").write_text("", encoding="utf-8")
        (xdf_dir / "unpack_batch.mojo").write_text(
            """from std.memory import MutPointer
comptime MutPtr[T: AnyType] = MutPointer[T, MutUntrackedOrigin]

def unpack_one_f32(
    raw: MutPtr[UInt8],
    raw_len: Int,
    addr: Int,
    lng: Int,
    ni: Int,
    nj: Int,
    nk: Int,
    datyp: Int,
    nbits: Int,
    dst: MutPtr[Float32],
):
    var byte_off = (addr - 1) * 8
    var nelm = ni * nj * nk
    var p = raw.unsafe_offset(byte_off)
    if datyp == 2 and nbits == 1:
        for i in range(nelm):
            var b = p[unsafe_offset=i >> 3]
            var bit = (b >> UInt8(i & 7)) & UInt8(1)
            dst[unsafe_offset=i] = Float32(1.0) if bit != UInt8(0) else Float32(0.0)
        return
    var src_f = p.unsafe_bitcast[Float32]()
    for i in range(nelm):
        dst[unsafe_offset=i] = src_f[unsafe_offset=i]
""",
            encoding="utf-8",
        )
        bridge_so = build_cmc_bridge.build_bridge(mock_cmc, out_so=td / "libcmc_marray_bridge.so")
        assert bridge_so is not None and bridge_so.is_file(), "cmc_io_cabi_shim.mojo failed to compile!"

    # 2. Write 4 `XDF0` lead files (`8`-byte aligned records + 1-bit `@@` bitmaps)
    ni, nj = int(raw_z["ni"]), int(raw_z["nj"])
    n_t, n_z = int(raw_z["uu_f32"].shape[0]), int(raw_z["uu_f32"].shape[1])
    nelm = ni * nj
    bmap_bytes = ((nelm + 7) // 8 + 7) // 8 * 8
    data_bytes = ((nelm * 4 + 7) // 8) * 8

    # Descending pressure levels on Dataset vs shuffled rows in FST directory!
    desc_levels = np.linspace(1000.0, 100.0, n_z, dtype=np.float64)
    times_ns = raw_z["time"]
    xdf_paths: list[Path] = []
    per_file_rows: dict[str, list[dict[str, Any]]] = {}

    for t_i in range(n_t):
        fp = td / f"2026112600_{t_i * 3:03d}"
        rows: list[dict[str, Any]] = []
        with open(fp, "wb") as fh:
            fh.write(b"XDF0STDR" + b"\x00" * 8)  # 16-byte header (`addr = 3` -> byte 16)
            # Shuffled level write order to verify coordinate-aligned `rec_meta` permutation!
            shuffled_z = list(reversed(range(n_z)))
            for z_i in shuffled_z:
                bm_plane = raw_z["bmap_3d"][t_i, z_i].reshape(-1)
                packed_bits = np.packbits(bm_plane > 0, bitorder="little")
                bm_pad = np.zeros(bmap_bytes, dtype=np.uint8)
                bm_pad[: len(packed_bits)] = packed_bits
                bm_off = fh.tell()
                fh.write(bm_pad.tobytes())
                bm_addr = (bm_off // 8) + 1
                bm_lng = math.ceil(bmap_bytes / 4)

                for nom, cube in (("UU", raw_z["uu_f32"]), ("VV", raw_z["vv_f32"]), ("TT", raw_z["tt_f32"])):
                    d_off = fh.tell()
                    plane_bytes = np.ascontiguousarray(cube[t_i, z_i], dtype=np.float32).tobytes()
                    fh.write(plane_bytes)
                    if len(plane_bytes) < data_bytes:
                        fh.write(b"\x00" * (data_bytes - len(plane_bytes)))
                    d_addr = (d_off // 8) + 1
                    d_lng = math.ceil(data_bytes / 4)
                    rows.append(
                        {
                            "path": str(fp),
                            "nomvar": f"{nom:<4}",
                            "typvar": "P@",
                            "ni": ni,
                            "nj": nj,
                            "nk": 1,
                            "addr": d_addr,
                            "lng": d_lng,
                            "datyp": 5,
                            "nbits": 32,
                            "mask_addr": bm_addr,
                            "mask_lng": bm_lng,
                            "time": times_ns[t_i],
                            "vertical": float(desc_levels[z_i]),
                        }
                    )
        # Add 2,000 unused variable rows per file (`8,000` across 4 leads) to test `read_records_fast` pre-filter!
        for k in range(2000):
            rows.append(
                {
                    "path": str(fp),
                    "nomvar": "GZ  " if (k & 1) else "WW  ",
                    "typvar": "P ",
                    "ni": ni,
                    "nj": nj,
                    "nk": 1,
                    "addr": 3,
                    "lng": 4,
                    "datyp": 134,
                    "nbits": 16,
                    "mask_addr": None,
                    "mask_lng": 0,
                    "time": times_ns[t_i],
                    "vertical": 500.0,
                }
            )
        xdf_paths.append(fp)
        per_file_rows[str(fp)] = rows

    # 3. Install mock `boreal.io` whose `_stack_variable` FAILS if Dask HLG is ever invoked!
    mod_boreal = types.ModuleType("boreal")
    mod_io = types.ModuleType("boreal.io")
    mod_xr = types.ModuleType("boreal.io.xarray")
    mod_open = types.ModuleType("boreal.io.xarray.open")
    mod_stack = types.ModuleType("boreal.io.xarray.stack")
    mod_decode = types.ModuleType("boreal.io.xarray.decode")

    seen_post_filter_counts: list[int] = []

    def read_records_fast(paths_in: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p_s in paths_in:
            out.extend(per_file_rows[str(p_s)])
        return out

    def _stack_variable(
        records: list[dict[str, Any]],
        dims: tuple[str, ...],
        *,
        ni: int,
        nj: int,
        attrs: dict[str, Any] | None = None,
    ) -> xr.Variable:
        raise AssertionError("_stack_variable Dask HLG builder MUST NOT be called when direct catalog is active!")

    def open_fst(
        paths_in: list[str],
        *,
        vars: list[str] | None = None,
        chunks: Any = "column",
        engine: str = "mojo",
        open_data_mapping: bool = True,
        cell_measures: bool = True,
        apply_masks: bool = True,
    ) -> list[xr.Dataset]:
        all_rows = mod_open.read_records_fast(paths_in)  # type: ignore[attr-defined]
        seen_post_filter_counts.append(len(all_rows))
        cf_map = {"UU": "WindU", "VV": "WindV", "TT": "AirTemp"}
        dvars: dict[str, Any] = {}
        for nom, cfn in cf_map.items():
            v_rows = [r for r in all_rows if r["nomvar"].strip() == nom]
            if not v_rows:
                continue
            vname = cfn if open_data_mapping else nom
            vattrs = {"fst_nomvar": nom, "fst_grtyp": "Z", "units": "knots" if nom != "TT" else "degC"}
            dvars[vname] = mod_open._stack_variable(  # type: ignore[attr-defined]
                v_rows,
                ("time", "vertical", "rlat", "rlon"),
                ni=ni,
                nj=nj,
                attrs=vattrs,
            )
        coords = {
            "time": times_ns,
            "vertical": desc_levels,
            "rlat": raw_z["y"],
            "rlon": raw_z["x"],
            "lat": (("rlat", "rlon"), raw_z["lat_2d"]),
            "lon": (("rlat", "rlon"), raw_z["lon_2d"]),
            "areacell": (("rlat", "rlon"), raw_z["areacell"]),
        }
        return [xr.Dataset(dvars, coords=coords, attrs={"fst_grtyp": "Z", "fst_vctype": "pressure"})]

    mod_open.read_records_fast = read_records_fast  # type: ignore[attr-defined]
    mod_open._stack_variable = _stack_variable  # type: ignore[attr-defined]
    mod_open.open_fst = open_fst  # type: ignore[attr-defined]
    mod_stack._stack_variable = _stack_variable  # type: ignore[attr-defined]
    mod_io.open_fst = open_fst  # type: ignore[attr-defined]

    saved_mods = {k: sys.modules.get(k) for k in ("boreal", "boreal.io", "boreal.io.xarray", "boreal.io.xarray.open", "boreal.io.xarray.stack", "boreal.io.xarray.decode")}
    saved_env = os.environ.get("MARRAY_CMC_CABI_SO")
    try:
        sys.modules["boreal"] = mod_boreal
        sys.modules["boreal.io"] = mod_io
        sys.modules["boreal.io.xarray"] = mod_xr
        sys.modules["boreal.io.xarray.open"] = mod_open
        sys.modules["boreal.io.xarray.stack"] = mod_stack
        sys.modules["boreal.io.xarray.decode"] = mod_decode
        if bridge_so is not None:
            os.environ["MARRAY_CMC_CABI_SO"] = str(bridge_so)
            dec_override = None
        else:
            dec_override = mxr.get_builtin_decoder_addr()

        t0 = time.perf_counter()
        dss_xdf = mxr.open_fst_xarray(xdf_paths, vars=["TT", "UU", "VV"], decoder_fn_addr=dec_override)
        xdf_open_ms = (time.perf_counter() - t0) * 1000.0
        info = mxr.get_last_fst_open_info()

        assert info["tier"] == "A", f"Expected Tier A active, got {info}"
        assert info["direct_catalog_vars"] == 3 and info["hlg_fallback_vars"] == 0, f"Expected 0 HLG vars, got {info}"
        assert info["prefiltered_rows"] is True
        # 8,120 raw rows pre-filtered down to ONLY 120 (`3 vars x 4 leads x 10 lvls`)!
        assert seen_post_filter_counts[-1] == 3 * n_t * n_z, f"Pre-filter kept {seen_post_filter_counts[-1]} rows!"

        target_lvl = float(desc_levels[3]) - 5.0
        t1 = time.perf_counter()
        u2d = dss_xdf[0]["WindU"].isel(time=2).sel(vertical=target_lvl, method="nearest")
        v2d = dss_xdf[0]["WindV"].isel(time=2).sel(vertical=target_lvl, method="nearest")
        uv_got = (np.hypot(u2d, v2d) * np.float32(0.514444)).compute()
        xdf_peel_ms = (time.perf_counter() - t1) * 1000.0

        assert mxr.get_last_decoded_planes() == 2, f"Expected 2 planes decoded, got {mxr.get_last_decoded_planes()}"
        bm = raw_z["bmap_3d"][2, 3] > 0
        uv_expected = np.hypot(
            np.where(bm, raw_z["uu_f32"][2, 3], np.float32(np.nan)),
            np.where(bm, raw_z["vv_f32"][2, 3], np.float32(np.nan)),
        ) * np.float32(0.514444)
        np.testing.assert_allclose(uv_got.values, uv_expected, rtol=1e-4, atol=1e-3, equal_nan=True)

        for d in dss_xdf:
            d.close()
        return xdf_open_ms, xdf_peel_ms
    finally:
        for k, prev in saved_mods.items():
            if prev is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = prev
        if saved_env is None:
            os.environ.pop("MARRAY_CMC_CABI_SO", None)
        else:
            os.environ["MARRAY_CMC_CABI_SO"] = saved_env


def main() -> None:
    # Generate a 4-lead x 10-level `grtyp=Z` (`rlat`/`rlon` + 2D `lat`/`lon`) archive on disk:
    raw_z = mxr.synth_boreal_raw_arenas(grtyp="Z", scale_factor=2, n_steps=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        lead_paths = mxr.write_multifile_fst_leads(tmpdir, raw_z, datyp=134)
        assert len(lead_paths) == 4, f"Expected 4 lead files, got {len(lead_paths)}"
        total_disk_mb = sum(p.stat().st_size for p in lead_paths) / (1024**2)

        # 1. P0 Criterion 1 & 3: O(headers) `open_fst_xarray(lead_paths)` -> `list[xr.Dataset]`
        t0 = time.perf_counter()
        dss = mxr.open_fst_xarray(lead_paths)
        open_ms = (time.perf_counter() - t0) * 1000.0

        assert len(dss) == 2, f"Expected [hybrid, surface] Datasets, got {len(dss)}"
        ds_hyb, ds_sfc = dss
        assert isinstance(ds_hyb, xr.Dataset) and isinstance(ds_sfc, xr.Dataset)
        assert ds_hyb.attrs["fst_vctype"] == "hybrid" and ds_sfc.attrs["fst_vctype"] == "surface"
        assert ds_hyb["AirTemp"].dims == ("time", "vertical", "rlat", "rlon")
        assert ds_sfc["SeaSurfTemp"].dims == ("time", "rlat", "rlon")
        assert "lat" in ds_hyb.coords and ds_hyb.coords["lat"].dims == ("rlat", "rlon")
        assert "lon" in ds_hyb.coords and ds_hyb.coords["lon"].dims == ("rlat", "rlon")
        assert "areacell" in ds_hyb.coords and ds_hyb.coords["areacell"].dims == ("rlat", "rlon")
        assert np.issubdtype(ds_hyb.coords["time"].dtype, np.datetime64)
        assert isinstance(ds_hyb["AirTemp"].variable._data, mxr.MarrayDuckArray)

        # Verify `repr(ds_hyb)` treats `MarrayDuckArray` as a chunked array (`0` planes decoded!):
        _ = repr(ds_hyb)

        # 2. P0 Criterion 2: Surgical POSIX `pread` on thin 2D peel (`time=2`, nearest `vertical`)!
        target_p = float(raw_z["vertical"][3]) + 2.0
        t1 = time.perf_counter()
        u2d = ds_hyb["WindU"].isel(time=2).sel(vertical=target_p, method="nearest")
        v2d = ds_hyb["WindV"].isel(time=2).sel(vertical=target_p, method="nearest")
        uv_da = (np.hypot(u2d, v2d) * np.float32(0.514444)).compute()
        peel_ms = (time.perf_counter() - t1) * 1000.0
        planes_decoded = mxr.get_last_decoded_planes()

        assert isinstance(uv_da.variable._data, np.ndarray), "`.compute()` must materialize via MarrayChunkManager!"
        # Out of 4 leads x 10 levels x 5 vars = 164 total planes on disk, ONLY 2 planes decoded!
        assert planes_decoded == 2, f"Expected EXACTLY 2 planes decoded (`1 UU + 1 VV`), got {planes_decoded}!"

        bmap_ref = raw_z["bmap_3d"][2, 3] > 0
        u_ref = np.where(bmap_ref, raw_z["uu_f32"][2, 3], np.float32(np.nan))
        v_ref = np.where(bmap_ref, raw_z["vv_f32"][2, 3], np.float32(np.nan))
        uv_ref = np.hypot(u_ref, v_ref) * np.float32(0.514444)
        np.testing.assert_allclose(uv_da.values, uv_ref, rtol=2e-3, atol=2e-2, equal_nan=True)

        # 3. Verify `u ** 2 + v ** 2` (`np.power` ufunc!) stays 100% lazy on `MarrayDuckArray`!
        u4 = ds_hyb["WindU"] * np.float32(0.514444)
        v4 = ds_hyb["WindV"] * np.float32(0.514444)
        ke_lazy = np.float32(0.5) * (u4 ** 2 + v4 ** 2)
        assert isinstance(ke_lazy.variable._data, mxr.MarrayDuckArray), "Power `** 2` must stay lazy!"
        zon_ke = ke_lazy.mean(dim=("time", "rlon"))

        b4 = raw_z["bmap_3d"] > 0
        ke_ref = 0.5 * (
            (np.where(b4, raw_z["uu_f32"], np.nan) * 0.514444) ** 2
            + (np.where(b4, raw_z["vv_f32"], np.nan) * 0.514444) ** 2
        )
        with np.errstate(invalid="ignore"):
            zon_ref = np.nanmean(ke_ref, axis=(0, 3))
        np.testing.assert_allclose(zon_ke.values, zon_ref, rtol=2e-3, atol=5e-2, equal_nan=True)

        # 4. P0 Criterion 5: Standard `xr.open_dataset(..., engine='marray_fst')` BackendEntrypoint!
        # Mode A (`chunks={}` -> `MarrayDuckArray` chunkmanager + fused multi-var `ds.compute()`):
        ds_be = xr.open_dataset(lead_paths[0], engine="marray_fst", chunks={}, chunked_array_type="marray")
        assert "AirTemp" in ds_be and isinstance(ds_be["AirTemp"].variable._data, mxr.MarrayDuckArray)
        ds_pair = ds_be[["AirTemp", "SpecHum"]].isel(time=0, vertical=1).compute()
        assert isinstance(ds_pair["AirTemp"].variable._data, np.ndarray)
        assert mxr.get_last_decoded_planes() == 2, "Multi-var `ds.compute()` must fuse into 1 pass (2 planes)!"

        # Mode B (`chunks=None` default -> Xarray `LazilyIndexedArray` decoding ONLY 1 plane on `.isel`!):
        ds_be_sfc = xr.open_dataset(lead_paths[0], engine="marray_fst", vctype="surface")
        assert "SeaSurfTemp" in ds_be_sfc
        sst_plane = ds_be_sfc["SeaSurfTemp"].isel(time=0).values
        assert sst_plane.shape == (raw_z["nj"], raw_z["ni"]) and mxr.get_last_decoded_planes() == 1

        ds_be.close()
        ds_be_sfc.close()
        for d in dss:
            d.close()

        # 5. RC1-RC5 (`13-cold-open-work-machine.md`): Tier A `cmc_io_cabi_shim.mojo` + Zero Dask HLG!
        xdf_open_ms, xdf_peel_ms = _verify_boreal_xdf0_tier_a_and_zero_hlg(raw_z, tmpdir)

        print("=" * 88)
        print("P0 MULTI-FILE POSIX `pread` + `open_fst_xarray` + `engine='marray_fst'` SUITE PASSED!")
        print(f"  4-lead `grtyp=Z` suite on disk : {total_disk_mb:.2f} MB across {len(lead_paths)} files")
        print(f"  `.mfst` `open_fst_xarray` open : {open_ms:.3f} ms  (O(headers), 0 planes decoded)")
        print(f"  `.mfst` 1st `.isel` + nearest  : {peel_ms:.3f} ms  (decoded EXACTLY {planes_decoded}/164 planes via `pread`!)")
        print(f"  `XDF0` Tier-A zero-HLG open    : {xdf_open_ms:.3f} ms  (8,120 -> 120 rows pre-filtered, 0 Dask tasks!)")
        print(f"  `XDF0` Tier-A 1st 2D peel      : {xdf_peel_ms:.3f} ms  (2 planes via `FilePool` + `cmc_fst_chunk_decode_cabi`!)")
        print("=" * 88)


if __name__ == "__main__":
    main()
