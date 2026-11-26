"""Live Multi-File FST / `.mfst` Header -> `CatalogVarSpec` + `ChunkDecoder` Connector (`CONTEXT.md` P0).

Supports two multi-file on-disk formats seamlessly:
  1. **Operational RPN FST (`.std` / `YYYYMMDDHH_HHH`, XDF0 & RSF0)** via `boreal-io`
     (`MARRAY_CMC_IO` or installed `boreal.io`):
       - **Header / CF Assembly (`O(headers)`, zero payload unpack):** Intercepts
         `boreal.io.xarray.stack._stack_variable` (with fallback to lazy record-spec
         extraction) so `boreal.io`'s Mojo `cmc_io_fst_hdr` scan, `ip1` vertical decode,
         `group5005`, `attach_mask_addresses`, `rlat`/`rlon` + 2D `lat`/`lon`, `areacell`,
         and `split_dataset_groups` run untouched while skipping Dask `HighLevelGraph`
         task creation!
       - **Live `ChunkDecoder` (Zero Re-Encoding):**
         * **Tier A (16-Thread Zero-GIL C-ABI):** Uses `cmc_fst_chunk_decode_cabi` from
           `$MARRAY_CMC_DECODER_SO` / `build/libcmc_marray_bridge.so` / `cmc_io_fst_data*.so`
           + `FilePool` POSIX `pread` when exported (`bridges/cmc_io_cabi_shim.mojo`).
         * **Tier B (Zero-Build Live `boreal-io` Unpacker Trampoline):** Wraps
           `boreal.io`'s native `cmc_io_fst_data` single-record unpacker in a C-ABI
           `ctypes.CFUNCTYPE` (`CAT_FLAG_ADDR_MODE | CAT_FLAG_FORCE_SERIAL`) so XDF0
           and RSF0 files decode surgically on demand on Day 1 without recompiling `cmc-io`!
  2. **Standalone Multi-Lead `.mfst` Files (`write_multifile_fst_leads`)**:
     Reads only the small JSON header prefix (`O(headers)`), opens all lead files in
     `FilePool` (`pread`), and decodes touched `datyp=134` + `datyp=2` `@@` planes via
     `marray_builtin_fst_chunk_decoder` with zero `boreal-io` dependency!
"""

from __future__ import annotations

import ctypes
import glob
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import xarray as xr

import marray_xr as mxr
from .catalog import FilePool, REC_META_COLS


CAT_FLAG_FORCE_SERIAL = 1
CAT_FLAG_ADDR_MODE = 2

_C_DECODER_FN = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int64,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
)


def _expand_input_paths(source: str | Path | Sequence[str | Path]) -> list[Path]:
    raw_items = [source] if isinstance(source, (str, Path)) else list(source)
    out: list[Path] = []
    for item in raw_items:
        s = str(item)
        p = Path(s)
        if p.is_dir():
            mfst = sorted(p.glob("*.mfst"))
            if mfst:
                out.extend(mfst)
            else:
                out.extend(sorted(x for x in p.iterdir() if x.is_file()))
        elif any(ch in s for ch in "*?[]"):
            out.extend(Path(m) for m in sorted(glob.glob(s)))
        else:
            out.append(p)
    if not out:
        raise FileNotFoundError(f"No FST input files matched: {source!r}")
    return out


def _is_mfst_file(path: Path) -> bool:
    if path.suffix == ".mfst":
        return True
    try:
        with open(path, "rb") as fh:
            return fh.read(8) == b"MFST0001"
    except OSError:
        return False


# ===----------------------------------------------------------------------=== #
# Path 1: Standalone Multi-File `.mfst` POSIX `pread` Reader
# ===----------------------------------------------------------------------=== #


def _open_mfst_paths_catalog(
    paths: list[Path],
    *,
    vars_filter: set[str] | None,
    decoder_fn_addr: int | None,
    split_vctype: bool,
    open_data_mapping: bool,
    cell_measures: bool,
) -> list[mxr.Dataset]:
    pool = FilePool(paths)
    dec_addr = (
        mxr.get_builtin_decoder_addr() if decoder_fn_addr is None else int(decoder_fn_addr)
    )

    lead_headers: list[dict[str, Any]] = []
    payload_offsets: list[int] = []
    first_coords: dict[str, np.ndarray] = {}

    for f_idx, p in enumerate(paths):
        with open(p, "rb") as fh:
            magic = fh.read(8)
            if magic != b"MFST0001":
                pool.close()
                raise ValueError(f"Invalid .mfst header in {p}")
            hdr_len = int.from_bytes(fh.read(8), "little")
            hdr = json.loads(fh.read(hdr_len).decode("utf-8"))
            prefix_len = 16 + hdr_len
            pad_len = (64 - (prefix_len % 64)) % 64
            c_lens = hdr["coord_byte_lens"]
            if f_idx == 0:
                if pad_len:
                    fh.seek(pad_len, os.SEEK_CUR)
                ni, nj = int(hdr["ni"]), int(hdr["nj"])
                first_coords["y"] = np.frombuffer(fh.read(c_lens[0]), dtype=np.float64).copy()
                first_coords["x"] = np.frombuffer(fh.read(c_lens[1]), dtype=np.float64).copy()
                first_coords["vertical"] = np.frombuffer(fh.read(c_lens[2]), dtype=np.float64).copy()
                first_coords["areacell"] = np.frombuffer(fh.read(c_lens[3]), dtype=np.float32).reshape(nj, ni).copy()
                first_coords["lat_2d"] = np.frombuffer(fh.read(c_lens[4]), dtype=np.float32).reshape(nj, ni).copy()
                first_coords["lon_2d"] = np.frombuffer(fh.read(c_lens[5]), dtype=np.float32).reshape(nj, ni).copy()
            payload_offsets.append(prefix_len + pad_len + sum(c_lens))
            lead_headers.append(hdr)

    h0 = lead_headers[0]
    grtyp = str(h0["grtyp"])
    ni, nj = int(h0["ni"]), int(h0["nj"])
    y_dim, x_dim = str(h0["y_dim"]), str(h0["x_dim"])
    n_t = len(lead_headers)
    n_z = len(first_coords["vertical"])
    times_ns = np.array([h["valid_time_ns"] for h in lead_headers], dtype="datetime64[ns]")

    common_attrs = {
        "fst_grtyp": grtyp,
        "fst_mask_fill": mxr.FST_MASK_FILL,
    }
    if cell_measures:
        common_attrs["cell_measures"] = "area: areacell"

    hyb_vars: dict[str, mxr.Variable] = {}
    sfc_vars: dict[str, mxr.Variable] = {}

    for nomvar, vinfo0 in h0["vars"].items():
        cf_name = str(vinfo0["cf_name"])
        if vars_filter and nomvar not in vars_filter and cf_name not in vars_filter:
            continue
        vname = cf_name if open_data_mapping else nomvar
        vctype = str(vinfo0["vctype"])
        is_hyb = vctype == "hybrid"
        recs_per_t = n_z if is_hyb else 1
        meta = np.empty((n_t * recs_per_t, REC_META_COLS), dtype=np.int64)

        for t_i, h in enumerate(lead_headers):
            p_base = payload_offsets[t_i]
            src_rows = np.asarray(h["vars"][nomvar]["records"], dtype=np.int64)
            for r_i in range(recs_per_t):
                dst_r = t_i * recs_per_t + r_i
                row = src_rows[r_i].copy()
                row[0] = t_i                    # file_id in `FilePool`!
                row[1] = p_base + int(row[1])   # absolute byte offset in file `t_i`!
                if int(row[6]) >= 0:
                    row[5] = t_i
                    row[6] = p_base + int(row[6])
                meta[dst_r] = row

        vattrs = {
            **common_attrs,
            "fst_nomvar": nomvar,
            "fst_vctype": vctype,
            "units": str(vinfo0["units"]),
        }
        cspec = mxr.CatalogVarSpec(
            raw_owner=pool,
            raw_base_addr=0,
            rec_meta=meta,
            plane_ni=ni,
            plane_nj=nj,
            rec_s_z=1 if is_hyb else 0,
            rec_s_t=n_z if is_hyb else 1,
            decoder_fn_addr=dec_addr,
            decoder_ctx_addr=0,
            file_pool_handle=pool.handle,
            flags=0,
            logical_shape=(n_t, n_z, nj, ni) if is_hyb else (n_t, nj, ni),
        )
        if is_hyb:
            hyb_vars[vname] = mxr.Variable(
                ("time", "vertical", y_dim, x_dim), catalog_spec=cspec, attrs=vattrs
            )
        else:
            sfc_vars[vname] = mxr.Variable(
                ("time", y_dim, x_dim), catalog_spec=cspec, attrs=vattrs
            )

    extra_2d: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
    if y_dim not in ("lat",) and x_dim not in ("lon",):
        extra_2d["lat"] = ((y_dim, x_dim), first_coords["lat_2d"])
        extra_2d["lon"] = ((y_dim, x_dim), first_coords["lon_2d"])

    measures = (
        {"areacell": mxr.Variable((y_dim, x_dim), first_coords["areacell"])}
        if cell_measures
        else {}
    )

    if not split_vctype:
        merged = {**hyb_vars, **sfc_vars}
        return [
            mxr.Dataset(
                merged,
                coords={
                    "time": times_ns,
                    "vertical": first_coords["vertical"],
                    y_dim: first_coords["y"],
                    x_dim: first_coords["x"],
                },
                domain_dims=("time", "vertical", y_dim, x_dim),
                cell_measures=measures,
                extra_coords_2d=extra_2d,
                close_fn=pool.close,
                attrs={"fst_grtyp": grtyp, "fst_vctype": "merged", "source": "file.std"},
            )
        ]

    out_ds: list[mxr.Dataset] = []
    if hyb_vars:
        out_ds.append(
            mxr.Dataset(
                hyb_vars,
                coords={
                    "time": times_ns,
                    "vertical": first_coords["vertical"],
                    y_dim: first_coords["y"],
                    x_dim: first_coords["x"],
                },
                domain_dims=("time", "vertical", y_dim, x_dim),
                cell_measures=measures,
                extra_coords_2d=extra_2d,
                close_fn=pool.close,
                attrs={"fst_grtyp": grtyp, "fst_vctype": "hybrid", "source": "file.std"},
            )
        )
    if sfc_vars:
        out_ds.append(
            mxr.Dataset(
                sfc_vars,
                coords={
                    "time": times_ns,
                    y_dim: first_coords["y"],
                    x_dim: first_coords["x"],
                },
                domain_dims=("time", y_dim, x_dim),
                cell_measures=measures,
                extra_coords_2d=extra_2d,
                close_fn=pool.close if not out_ds else None,
                attrs={"fst_grtyp": grtyp, "fst_vctype": "surface", "source": "file.std"},
            )
        )
    return out_ds


# ===----------------------------------------------------------------------=== #
# Path 2: Live Operational RPN FST (`.std`) Connector via `boreal-io` (`MARRAY_CMC_IO`)
# ===----------------------------------------------------------------------=== #


def _ensure_boreal_importable(cmc_root: str | Path | None = None) -> Path | None:
    cands: list[Path] = []
    if cmc_root is not None:
        cands.append(Path(cmc_root).expanduser())
    env_c = os.environ.get("MARRAY_CMC_IO", "").strip()
    if env_c:
        cands.append(Path(env_c).expanduser())
    repo_root = Path(__file__).resolve().parents[2]
    for rel in ("../cmc-io", "../boreal-io"):
        cands.append((repo_root / rel).resolve())

    for cand in cands:
        if (cand / "src" / "boreal" / "io").is_dir() or (cand / "boreal" / "io").is_dir():
            for p in (cand, cand / "src"):
                s = str(p)
                if s not in sys.path:
                    sys.path.insert(0, s)
            return cand
    return None


def _resolve_native_cmc_cabi_symbol(cmc_dir: Path | None) -> tuple[int, Any]:
    """Check `$MARRAY_CMC_DECODER_SO`, `build/libcmc_marray_bridge.so`, and `cmc_io_fst_data*.so`
    for a native zero-GIL `ChunkDecoderCABI` export (`cmc_fst_chunk_decode_cabi`).
    """
    repo_root = Path(__file__).resolve().parents[2]
    so_cands: list[Path] = []
    env_so = os.environ.get("MARRAY_CMC_DECODER_SO", "").strip()
    if env_so:
        so_cands.append(Path(env_so).expanduser())
    so_cands.append(repo_root / "build" / "libcmc_marray_bridge.so")
    if cmc_dir is not None:
        so_cands.extend(sorted(cmc_dir.glob("**/cmc_io_fst_data*.so")))

    for so_p in so_cands:
        if not so_p.is_file():
            continue
        try:
            cdll = ctypes.CDLL(str(so_p))
            for sym in ("cmc_fst_chunk_decode_cabi", "marray_fst_chunk_decoder"):
                fn = getattr(cdll, sym, None)
                if fn is not None:
                    addr = int(ctypes.cast(fn, ctypes.c_void_p).value or 0)
                    if addr != 0:
                        return addr, cdll
        except OSError:
            continue
    return 0, None


def _make_boreal_addr_trampoline(file_paths: list[str]) -> tuple[int, Any]:
    """Create a live `ChunkDecoderCABI` C trampoline wrapping `boreal.io`'s native
    `cmc_io_fst_data` single-record unpacker (`CAT_FLAG_ADDR_MODE | CAT_FLAG_FORCE_SERIAL`).
    Works on BOTH XDF0 and RSF0 files immediately with zero `cmc-io` recompilation!
    """
    from boreal.io.xarray import decode as bdecode  # type: ignore[import-not-found]

    unpack_single = getattr(bdecode, "unpack_record_from_path", None)
    unpack_many = getattr(bdecode, "unpack_records_into_path", None)

    def _c_cb(
        ctx_and_addr: int,
        _raw_ptr: Any,
        rpn_lng: int,
        datyp: int,
        nbits: int,
        ni: int,
        nj: int,
        packed_nk: int,
        dst_f32: Any,
    ) -> int:
        try:
            file_id = int(packed_nk) >> 8
            nk = int(packed_nk) & 0xFF
            path = file_paths[file_id]
            rpn_addr = int(ctx_and_addr)
            out_view = np.ctypeslib.as_array(dst_f32, shape=(int(nj), int(ni)))
            if unpack_single is not None:
                # boreal API: keyword-only after path; layout="C" → (nj, ni)
                plane = unpack_single(
                    path,
                    addr=rpn_addr,
                    lng=int(rpn_lng),
                    ni=int(ni),
                    nj=int(nj),
                    nk=max(1, nk),
                    datyp=int(datyp),
                    nbits=int(nbits),
                    layout="C",
                )
                arr = np.asarray(plane, dtype=np.float32)
                if arr.shape == (int(ni), int(nj)):
                    arr = arr.T
                out_view[:] = arr.reshape(int(nj), int(ni))
                return 0
            if unpack_many is not None:
                # specs=(addr,lng,ni,nj,nk,datyp,nbits); starts=flat element offsets
                unpack_many(
                    path,
                    [(rpn_addr, int(rpn_lng), int(ni), int(nj), max(1, nk), int(datyp), int(nbits))],
                    out_view.reshape(-1),
                    [0],
                )
                return 0
            return -1
        except Exception:
            return -2

    cb_obj = _C_DECODER_FN(_c_cb)
    fn_addr = int(ctypes.cast(cb_obj, ctypes.c_void_p).value or 0)
    return fn_addr, cb_obj


def _extract_record_specs_from_dask_da(
    da_xr: xr.DataArray, file_index: dict[str, int], use_addr_mode: bool
) -> tuple[np.ndarray, int, int] | None:
    """Extract `(path, addr, lng, datyp, nbits, mask_addr, mask_lng)` specs from a
    `boreal.io.open_fst(..., chunks='record')` lazy variable WITHOUT computing Dask!
    """
    dask_arr = da_xr.variable._data
    hlg = getattr(dask_arr, "__dask_graph__", lambda: None)()
    if hlg is None:
        return None

    dims = tuple(str(d) for d in da_xr.dims)
    has_t = "time" in dims
    has_z = "vertical" in dims
    n_t = int(da_xr.sizes["time"]) if has_t else 1
    n_z = int(da_xr.sizes["vertical"]) if has_z else 1
    n_rec = n_t * n_z
    meta = np.zeros((n_rec, REC_META_COLS), dtype=np.int64)
    meta[:, 0] = -1
    meta[:, 5] = -1
    meta[:, 6] = -1

    # Inspect task tuples inside the record-mode HighLevelGraph layer
    name = dask_arr.name
    for key, task in hlg.items():
        if not (isinstance(key, tuple) and key and key[0] == name):
            continue
        t_i = int(key[1]) if has_t and len(key) > 1 else 0
        z_i = int(key[2]) if (has_t and has_z and len(key) > 2) else (int(key[1]) if (not has_t and has_z and len(key) > 1) else 0)
        r_idx = t_i * n_z + z_i
        if r_idx >= n_rec or not isinstance(task, tuple):
            continue

        # Locate spec dict/tuple containing `(path, addr, lng, ...)` in task args:
        flat_args = list(task[1:])
        while len(flat_args) == 1 and isinstance(flat_args[0], (tuple, list)):
            flat_args = list(flat_args[0])

        path_s = next((a for a in flat_args if isinstance(a, (str, Path)) and Path(str(a)).exists()), None)
        ints = [int(a) for a in flat_args if isinstance(a, (int, np.integer))]
        if path_s is None and isinstance(flat_args[0] if flat_args else None, dict):
            spec_d = flat_args[0]
            path_s = spec_d.get("path")
            addr = int(spec_d.get("addr", spec_d.get("swa", 0)))
            lng = int(spec_d.get("lng", 0))
            dt = int(spec_d.get("datyp", 134))
            nb = int(spec_d.get("nbits", 16))
            m_addr = int(spec_d.get("mask_addr", -1) or -1)
            m_lng = int(spec_d.get("mask_lng", 0) or 0)
        elif path_s is not None and len(ints) >= 2:
            addr, lng = ints[0], ints[1]
            dt = ints[5] if len(ints) >= 6 else 134
            nb = ints[6] if len(ints) >= 7 else 16
            m_addr = ints[7] if len(ints) >= 8 else -1
            m_lng = ints[8] if len(ints) >= 9 else 0
        else:
            continue

        p_str = str(Path(str(path_s)))
        if p_str not in file_index:
            file_index[p_str] = len(file_index)
        fid = file_index[p_str]

        meta[r_idx, 0] = fid
        if use_addr_mode:
            meta[r_idx, 1] = addr
            meta[r_idx, 2] = lng
        else:
            meta[r_idx, 1] = max(0, (addr - 1) * 8)
            meta[r_idx, 2] = lng * 4
        meta[r_idx, 3] = dt
        meta[r_idx, 4] = nb
        if m_addr > 0 and m_lng > 0:
            meta[r_idx, 5] = fid
            meta[r_idx, 6] = m_addr if use_addr_mode else max(0, (m_addr - 1) * 8)
            meta[r_idx, 7] = m_lng if use_addr_mode else m_lng * 4
            meta[r_idx, 8] = 2
            meta[r_idx, 9] = 1

    rec_s_z = 1 if has_z else 0
    rec_s_t = n_z if has_t else 0
    return meta, rec_s_z, rec_s_t


def _open_real_fst_via_boreal(
    paths: list[Path],
    *,
    vars_filter: Sequence[str] | None,
    decoder_fn_addr: int | None,
    open_data_mapping: bool,
    cell_measures: bool,
    cmc_root: str | Path | None,
) -> list[xr.Dataset]:
    cmc_dir = _ensure_boreal_importable(cmc_root)
    from boreal.io import open_fst  # type: ignore[import-not-found]

    str_paths = [str(p) for p in paths]
    native_addr, native_owner = (
        (int(decoder_fn_addr), None)
        if decoder_fn_addr is not None
        else _resolve_native_cmc_cabi_symbol(cmc_dir)
    )
    use_addr_mode = native_addr == 0

    # Lazy metadata/header scan via boreal-io (`engine="mojo"`, zero payload unpack!)
    lazy_dss = open_fst(
        str_paths,
        vars=list(vars_filter) if vars_filter else None,
        chunks="record",
        engine="mojo",
        open_data_mapping=open_data_mapping,
        cell_measures=cell_measures,
        apply_masks=True,
    )

    file_index: dict[str, int] = {p: i for i, p in enumerate(str_paths)}
    group_specs: list[tuple[xr.Dataset, dict[str, tuple[np.ndarray, int, int, tuple[str, ...], dict[str, Any]]]]] = []

    for b_ds in lazy_dss:
        v_specs: dict[str, tuple[np.ndarray, int, int, tuple[str, ...], dict[str, Any]]] = {}
        for vname, da_xr in b_ds.data_vars.items():
            if vname == "areacell" or da_xr.ndim < 2:
                continue
            ext = _extract_record_specs_from_dask_da(da_xr, file_index, use_addr_mode)
            if ext is not None:
                meta, r_sz, r_st = ext
                v_specs[str(vname)] = (
                    meta,
                    r_sz,
                    r_st,
                    tuple(str(d) for d in da_xr.dims),
                    dict(da_xr.attrs),
                )
        group_specs.append((b_ds, v_specs))

    ordered_files = [p for p, _ in sorted(file_index.items(), key=lambda kv: kv[1])]
    if use_addr_mode:
        dec_addr, dec_owner = _make_boreal_addr_trampoline(ordered_files)
        pool = None
        flags = CAT_FLAG_ADDR_MODE | CAT_FLAG_FORCE_SERIAL
    else:
        pool = FilePool(ordered_files)
        dec_addr = native_addr
        dec_owner = (pool, native_owner)
        flags = 0

    out_xr_list: list[xr.Dataset] = []
    for b_ds, v_specs in group_specs:
        if not v_specs:
            out_xr_list.append(b_ds)
            continue
        lead_dims = max((sp[3] for sp in v_specs.values()), key=len)
        ni = int(b_ds.sizes[lead_dims[-1]])
        nj = int(b_ds.sizes[lead_dims[-2]])

        mx_vars: dict[str, mxr.Variable] = {}
        for vname, (meta, r_sz, r_st, vdims, vattrs) in v_specs.items():
            cspec = mxr.CatalogVarSpec(
                raw_owner=dec_owner,
                raw_base_addr=0,
                rec_meta=meta,
                plane_ni=ni,
                plane_nj=nj,
                rec_s_z=r_sz,
                rec_s_t=r_st,
                decoder_fn_addr=dec_addr,
                decoder_ctx_addr=0,
                file_pool_handle=pool.handle if pool is not None else 0,
                flags=flags,
                logical_shape=tuple(int(b_ds.sizes[d]) for d in vdims),
            )
            mx_vars[vname] = mxr.Variable(vdims, catalog_spec=cspec, attrs=vattrs)

        coords_1d = {
            d: np.asarray(b_ds.coords[d].values) if d in b_ds.coords else np.arange(b_ds.sizes[d], dtype=np.float64)
            for d in lead_dims
        }
        mx_ds = mxr.Dataset(
            mx_vars,
            coords=coords_1d,
            domain_dims=lead_dims,
            close_fn=pool.close if pool is not None else None,
            attrs=dict(b_ds.attrs),
        )

        # Preserve 100% of `boreal.io`'s exact 1D/2D coords (`lat`/`lon`, `areacell`, CRS, `hyb_a`/`b`)
        # while swapping data variables from Dask HLG -> `MarrayDuckArray`!
        new_data_vars: dict[str, Any] = {}
        for vname in b_ds.data_vars:
            if vname in mx_vars:
                mda = mx_ds[vname]
                vdims = mx_vars[vname].dims
                new_data_vars[vname] = (
                    vdims,
                    mxr.MarrayDuckArray(mda, vdims),
                    dict(b_ds[vname].attrs),
                )
            else:
                new_data_vars[vname] = b_ds[vname]

        xr_ds = xr.Dataset(new_data_vars, coords=b_ds.coords, attrs=dict(b_ds.attrs))
        if pool is not None:
            xr_ds.set_close(pool.close)
        out_xr_list.append(xr_ds)

    return out_xr_list


def open_fst_paths_catalog(
    paths: str | Path | Sequence[str | Path],
    *,
    vars: Sequence[str] | None = None,
    decoder_fn_addr: int | None = None,
    split_vctype: bool = True,
    open_data_mapping: bool = True,
    cell_measures: bool = True,
) -> list[mxr.Dataset]:
    expanded = _expand_input_paths(paths)
    vars_set = set(vars) if vars else None
    if _is_mfst_file(expanded[0]):
        return _open_mfst_paths_catalog(
            expanded,
            vars_filter=vars_set,
            decoder_fn_addr=decoder_fn_addr,
            split_vctype=split_vctype,
            open_data_mapping=open_data_mapping,
            cell_measures=cell_measures,
        )
    raise ValueError("For raw `.std` files, call `open_fst_xarray(paths)` to preserve full CF coords.")


def open_fst_paths_xarray(
    paths: str | Path | Sequence[str | Path],
    *,
    vars: Sequence[str] | None = None,
    decoder_fn_addr: int | None = None,
    split_vctype: bool = True,
    open_data_mapping: bool = True,
    cell_measures: bool = True,
    cmc_root: str | Path | None = None,
) -> list[xr.Dataset]:
    expanded = _expand_input_paths(paths)
    if _is_mfst_file(expanded[0]):
        cats = _open_mfst_paths_catalog(
            expanded,
            vars_filter=set(vars) if vars else None,
            decoder_fn_addr=decoder_fn_addr,
            split_vctype=split_vctype,
            open_data_mapping=open_data_mapping,
            cell_measures=cell_measures,
        )
        return [mxr.wrap_dataset_as_xarray(c) for c in cats]

    return _open_real_fst_via_boreal(
        expanded,
        vars_filter=vars,
        decoder_fn_addr=decoder_fn_addr,
        open_data_mapping=open_data_mapping,
        cell_measures=cell_measures,
        cmc_root=cmc_root,
    )
