"""M1 `RecordCatalog`, Multi-File POSIX `FilePool` & Archive Builder (`CONTEXT.md` P0).

Provides:
  - `FilePool`: RAII owner around `CatalogFilePool` (`marray_file_pool_create` /
    `marray_file_pool_destroy`) managing read-only POSIX file descriptors for
    multi-lead FST suites (`pread` on demand, zero Python `np.memmap` LRU).
  - `PackedFSTArchive` & `pack_boreal_fst_archive`: in-RAM archive fixture (`B6`, `R-B6`).
  - `write_multifile_fst_leads`: writes multi-lead on-disk `.mfst` files (`1` file per
    forecast lead, `rlat`/`rlon` + 2D `lat`/`lon` + `areacell` + `datetime64` `time`)
    for local multi-file POSIX `pread` & `xr.open_dataset(..., engine='marray_fst')` CI.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import marray_xr as mxr


REC_META_COLS = 10


class FilePool:
    """RAII owner of a C/Mojo `CatalogFilePool` (`fds` for multi-file POSIX `pread`)."""

    def __init__(self, paths: Sequence[str | Path]) -> None:
        self._lib = mxr.get_mojo_lib()
        self.paths = [str(Path(p)) for p in paths]
        encoded = [p.encode("utf-8") + b"\x00" for p in self.paths]
        offsets = np.zeros(max(1, len(encoded)), dtype=np.int64)
        cur = 0
        for i, chunk in enumerate(encoded):
            offsets[i] = cur
            cur += len(chunk)
        blob = np.frombuffer(b"".join(encoded) if encoded else b"\x00", dtype=np.uint8).copy()

        u8_p = ctypes.POINTER(ctypes.c_uint8)
        i64_p = ctypes.POINTER(ctypes.c_int64)
        self.handle = int(
            self._lib.marray_file_pool_create(
                blob.ctypes.data_as(u8_p),
                offsets.ctypes.data_as(i64_p),
                len(self.paths),
            )
        )

    def close(self) -> None:
        if getattr(self, "handle", 0) != 0:
            self._lib.marray_file_pool_destroy(self.handle)
            self.handle = 0

    def __del__(self) -> None:
        self.close()


@dataclass
class PackedFSTArchive:
    """In-memory / mmap-like packed FST byte archive (`datyp=134` + `datyp=2` `@@` bitmaps)."""

    grtyp: str
    ni: int
    nj: int
    raw_bytes: np.ndarray                 # contiguous `uint8` raw archive buffer
    var_tables: dict[str, np.ndarray]     # `nomvar -> (n_records, 10)` int64 offset/datyp table
    var_vctype: dict[str, str]            # `nomvar -> 'hybrid' | 'surface'`
    var_cf_names: dict[str, str]
    var_units: dict[str, str]
    time: np.ndarray
    vertical: np.ndarray
    y: np.ndarray
    x: np.ndarray
    areacell: np.ndarray

    @property
    def raw_nbytes(self) -> int:
        return int(self.raw_bytes.nbytes)


def pack_boreal_fst_archive(
    raw: dict[str, Any], *, datyp: int = 134
) -> PackedFSTArchive:
    """Pack a `synth_boreal_raw_arenas` fixture into whole-record `datyp=134` compressed
    planes + 1-bit `datyp=2` `@@` companion records in parallel via Mojo (`marray_pack_fst_var`).
    """
    lib = mxr.get_mojo_lib()
    ni, nj = int(raw["ni"]), int(raw["nj"])
    n_t, n_z = int(raw["tt_f32"].shape[0]), int(raw["tt_f32"].shape[1])
    nelm = ni * nj

    data_rec_b = math.ceil(nelm / 64) * 136 if datyp == 134 else nelm * 4
    bmap_rec_b = math.ceil(nelm / 8)

    specs = [
        ("TT", "AirTemp", "degC", "hybrid", raw["tt_f32"], raw["bmap_3d"], n_t * n_z),
        ("UU", "WindU", "knots", "hybrid", raw["uu_f32"], raw["bmap_3d"], n_t * n_z),
        ("VV", "WindV", "knots", "hybrid", raw["vv_f32"], raw["bmap_3d"], n_t * n_z),
        ("HU", "SpecHum", "g/kg", "hybrid", raw["hu_f32"], raw["hu_bmap"], n_t * n_z),
        ("TM", "SeaSurfTemp", "degC", "surface", raw["tm_f32"], raw["tm_bmap"], n_t),
    ]

    total_bytes = sum(n_rec * (data_rec_b + bmap_rec_b) for *_, n_rec in specs)
    archive_buf = np.empty(max(8, total_bytes), dtype=np.uint8)

    f32_p = ctypes.POINTER(ctypes.c_float)
    u8_p = ctypes.POINTER(ctypes.c_uint8)
    i64_p = ctypes.POINTER(ctypes.c_int64)

    var_tables: dict[str, np.ndarray] = {}
    var_vctype: dict[str, str] = {}
    var_cf: dict[str, str] = {}
    var_units: dict[str, str] = {}
    cur_off = 0

    for nomvar, cf_name, units, vctype, arr_f32, bmap_u8, n_rec in specs:
        meta = np.empty((n_rec, REC_META_COLS), dtype=np.int64)
        written = int(
            lib.marray_pack_fst_var(
                arr_f32.ctypes.data_as(f32_p),
                bmap_u8.ctypes.data_as(u8_p),
                1,
                n_rec,
                ni,
                nj,
                datyp,
                cur_off,
                archive_buf.ctypes.data_as(u8_p),
                meta.ctypes.data_as(i64_p),
            )
        )
        cur_off += written
        var_tables[nomvar] = meta
        var_vctype[nomvar] = vctype
        var_cf[nomvar] = cf_name
        var_units[nomvar] = units

    return PackedFSTArchive(
        grtyp=raw["grtyp"],
        ni=ni,
        nj=nj,
        raw_bytes=archive_buf,
        var_tables=var_tables,
        var_vctype=var_vctype,
        var_cf_names=var_cf,
        var_units=var_units,
        time=raw["time"],
        vertical=raw["vertical"],
        y=raw["y"],
        x=raw["x"],
        areacell=raw["areacell"],
    )


def write_multifile_fst_leads(
    out_dir: str | Path,
    raw: dict[str, Any],
    *,
    datyp: int = 134,
    base_date_iso: str = "2026-11-26T00:00:00",
) -> list[Path]:
    """Write a multi-lead on-disk `.mfst` suite (`1` file per forecast lead `YYYYMMDDHH_HHH.mfst`)
    with a small JSON directory header + raw `datyp=134` / `datyp=2` `@@` payload blocks!
    Enables testing multi-file OS `pread` + `open_fst_xarray(paths)` + `engine='marray_fst'`
    anywhere without committing binary fixtures or requiring operational mounts.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    n_t = int(raw["tt_f32"].shape[0])
    grtyp = str(raw["grtyp"])
    y_dim, x_dim = ("rlat", "rlon") if grtyp == "Z" else (("lat", "lon") if grtyp in ("L", "G") else ("y", "x"))

    base_dt = np.datetime64(base_date_iso, "ns")
    lead_paths: list[Path] = []

    for t_i in range(n_t):
        lead_hr = int(round(float(raw["time"][t_i])))
        single_raw = {
            **raw,
            "tt_f32": np.ascontiguousarray(raw["tt_f32"][t_i : t_i + 1]),
            "uu_f32": np.ascontiguousarray(raw["uu_f32"][t_i : t_i + 1]),
            "vv_f32": np.ascontiguousarray(raw["vv_f32"][t_i : t_i + 1]),
            "hu_f32": np.ascontiguousarray(raw["hu_f32"][t_i : t_i + 1]),
            "bmap_3d": np.ascontiguousarray(raw["bmap_3d"][t_i : t_i + 1]),
            "hu_bmap": np.ascontiguousarray(raw["hu_bmap"][t_i : t_i + 1]),
            "tm_f32": np.ascontiguousarray(raw["tm_f32"][t_i : t_i + 1]),
            "tm_bmap": np.ascontiguousarray(raw["tm_bmap"][t_i : t_i + 1]),
            "time": np.asarray([raw["time"][t_i]], dtype=np.float64),
        }
        pk = pack_boreal_fst_archive(single_raw, datyp=datyp)
        valid_dt = base_dt + np.timedelta64(lead_hr, "h")

        # File layout: `8B magic ("MFST0001") + 8B json_len_le + json_utf8 (padded to 64B) + aux_coords + payload`
        coord_blobs = [
            np.ascontiguousarray(raw["y"], dtype=np.float64).tobytes(),
            np.ascontiguousarray(raw["x"], dtype=np.float64).tobytes(),
            np.ascontiguousarray(raw["vertical"], dtype=np.float64).tobytes(),
            np.ascontiguousarray(raw["areacell"], dtype=np.float32).tobytes(),
            np.ascontiguousarray(raw["lat_2d"], dtype=np.float32).tobytes(),
            np.ascontiguousarray(raw["lon_2d"], dtype=np.float32).tobytes(),
        ]
        coord_lens = [len(b) for b in coord_blobs]

        hdr_obj = {
            "grtyp": grtyp,
            "ni": pk.ni,
            "nj": pk.nj,
            "y_dim": y_dim,
            "x_dim": x_dim,
            "valid_time_ns": int(valid_dt.view(np.int64)),
            "lead_hr": lead_hr,
            "coord_byte_lens": coord_lens,
            "vars": {
                nv: {
                    "cf_name": pk.var_cf_names[nv],
                    "units": pk.var_units[nv],
                    "vctype": pk.var_vctype[nv],
                    "records": pk.var_tables[nv].tolist(),
                }
                for nv in pk.var_tables
            },
        }
        hdr_bytes = json.dumps(hdr_obj).encode("utf-8")
        prefix_len = 16 + len(hdr_bytes)
        pad_len = (64 - (prefix_len % 64)) % 64
        coord_start = prefix_len + pad_len
        payload_start = coord_start + sum(coord_lens)

        fname = out_root / f"2026112600_{lead_hr:03d}.mfst"
        with open(fname, "wb") as fh:
            fh.write(b"MFST0001")
            fh.write(int(len(hdr_bytes)).to_bytes(8, "little"))
            fh.write(hdr_bytes)
            if pad_len:
                fh.write(b"\x00" * pad_len)
            for cb in coord_blobs:
                fh.write(cb)
            fh.write(pk.raw_bytes.tobytes())
        lead_paths.append(fname)

    return lead_paths


def open_fst_catalog(
    archive: PackedFSTArchive,
    *,
    decoder_fn_addr: int | None = None,
    decoder_ctx_addr: int = 0,
    split_vctype: bool = True,
) -> list[mxr.Dataset]:
    """Open an in-memory `PackedFSTArchive` into `list[mxr.Dataset]` split by `vctype` (`Q5`)."""
    dec_addr = (
        mxr.get_builtin_decoder_addr() if decoder_fn_addr is None else int(decoder_fn_addr)
    )
    base_addr = int(archive.raw_bytes.ctypes.data)
    n_t = len(archive.time)
    n_z = len(archive.vertical)
    common = {
        "fst_grtyp": archive.grtyp,
        "fst_mask_fill": mxr.FST_MASK_FILL,
        "cell_measures": "area: areacell",
    }

    hyb_vars: dict[str, mxr.Variable] = {}
    sfc_vars: dict[str, mxr.Variable] = {}

    for nomvar, meta in archive.var_tables.items():
        vctype = archive.var_vctype[nomvar]
        cf_name = archive.var_cf_names[nomvar]
        attrs = {
            **common,
            "fst_nomvar": nomvar,
            "fst_vctype": vctype,
            "units": archive.var_units[nomvar],
        }
        if vctype == "hybrid":
            cspec = mxr.CatalogVarSpec(
                raw_owner=archive.raw_bytes,
                raw_base_addr=base_addr,
                rec_meta=meta,
                plane_ni=archive.ni,
                plane_nj=archive.nj,
                rec_s_z=1,
                rec_s_t=n_z,
                decoder_fn_addr=dec_addr,
                decoder_ctx_addr=decoder_ctx_addr,
                logical_shape=(n_t, n_z, archive.nj, archive.ni),
            )
            hyb_vars[cf_name] = mxr.Variable(
                ("time", "vertical", "y", "x"), catalog_spec=cspec, attrs=attrs
            )
        else:
            cspec = mxr.CatalogVarSpec(
                raw_owner=archive.raw_bytes,
                raw_base_addr=base_addr,
                rec_meta=meta,
                plane_ni=archive.ni,
                plane_nj=archive.nj,
                rec_s_z=0,
                rec_s_t=1,
                decoder_fn_addr=dec_addr,
                decoder_ctx_addr=decoder_ctx_addr,
                logical_shape=(n_t, archive.nj, archive.ni),
            )
            sfc_vars[cf_name] = mxr.Variable(
                ("time", "y", "x"), catalog_spec=cspec, attrs=attrs
            )

    areacell_var = mxr.Variable(("y", "x"), archive.areacell)

    if not split_vctype:
        merged = {**hyb_vars, **sfc_vars}
        return [
            mxr.Dataset(
                merged,
                coords={
                    "time": archive.time,
                    "vertical": archive.vertical,
                    "y": archive.y,
                    "x": archive.x,
                },
                domain_dims=("time", "vertical", "y", "x"),
                cell_measures={"areacell": areacell_var},
                attrs={"fst_grtyp": archive.grtyp, "fst_vctype": "merged", "source": "file.std"},
            )
        ]

    out_list: list[mxr.Dataset] = []
    if hyb_vars:
        out_list.append(
            mxr.Dataset(
                hyb_vars,
                coords={
                    "time": archive.time,
                    "vertical": archive.vertical,
                    "y": archive.y,
                    "x": archive.x,
                },
                domain_dims=("time", "vertical", "y", "x"),
                cell_measures={"areacell": areacell_var},
                attrs={"fst_grtyp": archive.grtyp, "fst_vctype": "hybrid", "source": "file.std"},
            )
        )
    if sfc_vars:
        out_list.append(
            mxr.Dataset(
                sfc_vars,
                coords={
                    "time": archive.time,
                    "y": archive.y,
                    "x": archive.x,
                },
                domain_dims=("time", "y", "x"),
                cell_measures={"areacell": areacell_var},
                attrs={"fst_grtyp": archive.grtyp, "fst_vctype": "surface", "source": "file.std"},
            )
        )
    return out_list
