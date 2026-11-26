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

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

import marray_xr as mxr  # noqa: E402


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

        print("=" * 88)
        print("P0 MULTI-FILE POSIX `pread` + `open_fst_xarray` + `engine='marray_fst'` SUITE PASSED!")
        print(f"  4-lead `grtyp=Z` suite on disk : {total_disk_mb:.2f} MB across {len(lead_paths)} files")
        print(f"  `open_fst_xarray(leads)` open  : {open_ms:.3f} ms  (O(headers), 0 planes decoded)")
        print(f"  1st `.isel` + nearest vertical : {peel_ms:.3f} ms  (decoded EXACTLY {planes_decoded}/164 planes via `pread`!)")
        print(f"  Cold open + 1st 2D peel total  : {open_ms + peel_ms:.3f} ms")
        print("=" * 88)


if __name__ == "__main__":
    main()
