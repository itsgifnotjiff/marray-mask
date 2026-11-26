"""M1 `RecordCatalog` & C-ABI `ChunkDecoder` Archive Builder (`docs/questions.md` Q1-Q5).

Models the exact production `boreal-io` (`cmc_io_fst_hdr` + `cmc_io_fst_data`) contract:
  - Separate-`.so` C-ABI `ChunkDecoder` function pointer (`decoder_fn_addr`).
  - Whole-record compressed `datyp=134` payload records + `datyp=2, nbits=1` `@@`
    companion bitmap records (unpacked into `float32` `0/1` planes on demand).
  - Splits archives by `(grtyp, ni, nj, vctype)` into `list[Dataset]` (`ds_hybrid`
    4D `(time, vertical, y, x)` + `ds_surface` 3D `(time, y, x)`), supporting zero-copy
    cross-Dataset expressions via `mxr.from_sources(...)`!
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

import marray_xr as mxr


@dataclass
class PackedFSTArchive:
    """In-memory / mmap-like packed FST byte archive (`datyp=134` + `datyp=2` `@@` bitmaps)."""

    grtyp: str
    ni: int
    nj: int
    raw_bytes: np.ndarray                 # contiguous `uint8` raw archive buffer
    var_tables: dict[str, np.ndarray]     # `nomvar -> (n_records, 8)` int64 offset/datyp table
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
        meta = np.empty((n_rec, 8), dtype=np.int64)
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


def open_fst_catalog(
    archive: PackedFSTArchive,
    *,
    decoder_fn_addr: int | None = None,
    decoder_ctx_addr: int = 0,
    split_vctype: bool = True,
) -> list[mxr.Dataset]:
    """Open a packed FST archive into `list[mxr.Dataset]` split by `vctype` (`Q5`)
    with ZERO payload decompression at open time and C-ABI `ChunkDecoder` binding (`Q1`).
    """
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
                rec_s_z=0,  # stride-0 broadcast across `vertical` when paired with 4D cubes!
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
