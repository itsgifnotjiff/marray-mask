"""Live Multi-File FST / `.mfst` Header -> `CatalogVarSpec` + `ChunkDecoder` Connector (`CONTEXT.md` P0).

Supports two multi-file on-disk formats seamlessly:
  1. **Operational RPN FST (`.std` / `YYYYMMDDHH_HHH`, XDF0 & RSF0)** via `boreal-io`
     (`MARRAY_CMC_IO` or installed `boreal.io`):
       - **Direct Header -> Catalog Assembly (Zero Dask HLG):**
         * Parallelizes `read_records_fast` across multi-lead files (`ThreadPoolExecutor`)
           and pre-filters raw header rows to requested `nomvar`s + RPN coordinate/vgrid/mask
           companions (`>>`, `^^`, `^>`, `!!`, `HY`, `P0`, `PT`, `@@`, `!@`) BEFORE Python
           DataFrame IP1/date/mask enrichment.
         * Intercepts `boreal.io.xarray.stack._stack_variable` (and `open._stack_variable`)
           to build `(n_rec, 10)` `int64` `rec_meta` + `CatalogVarSpec` + `MarrayDuckArray`
           directly from group record columns—skipping `dask.base.tokenize`, `HighLevelGraph`
           construction, HLG task scraping, and double `xr.Dataset` rebuilds!
         * Retains a fast `chunks='column'` zero-`stat()` fallback for exotic 5D/3D-record cubes.
       - **Live `ChunkDecoder` (Zero Re-Encoding):**
         * **Tier A (16-Thread Zero-GIL C-ABI, XDF0):** Resolves `cmc_fst_chunk_decode_cabi`
           in O(1) (cached, zero recursive `glob`) from `$MARRAY_CMC_DECODER_SO`,
           `$MARRAY_CMC_CABI_SO`, `build/libcmc_marray_bridge.so`, or `cmc_io_fst_data.so`
           + `FilePool` POSIX `pread`.
         * **Tier B (Live `boreal-io` Unpacker Trampoline, RSF0 / Fallback):** Wraps
           `boreal.io.xarray.decode` keyword-only `unpack_record_from_path(..., layout='C')`
           in a `ctypes.CFUNCTYPE` (`CAT_FLAG_ADDR_MODE | CAT_FLAG_FORCE_SERIAL`).
  2. **Standalone Multi-Lead `.mfst` Files (`write_multifile_fst_leads`)**:
     Reads only the small JSON header prefix (`O(headers)`), opens all lead files in
     `FilePool` (`pread`), and decodes touched `datyp=134` + `datyp=2` `@@` planes via
     `marray_builtin_fst_chunk_decoder` with zero `boreal-io` dependency!
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import ctypes
import functools
import glob
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import threading
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

import marray_xr as mxr
from .catalog import FilePool, REC_META_COLS


CAT_FLAG_FORCE_SERIAL = 1
CAT_FLAG_ADDR_MODE = 2

_OPEN_HOOK_LOCK = threading.RLock()
_READ_TLS = threading.local()

_LAST_FST_OPEN_INFO: dict[str, Any] = {
    "tier": "none",
    "direct_catalog_vars": 0,
    "hlg_fallback_vars": 0,
    "prefiltered_rows": False,
}

# RPN coordinate, vertical descriptor, and bitmap companion nomvars that MUST always be kept
_RPN_META_NOMVARS = frozenset(
    {
        ">>",
        "^^",
        "^>",
        "!!",
        "!!SF",
        "HY",
        "P0",
        "PT",
        "E1",
        "@@",
        "!@",
        "##",
    }
)

# Common OpenData CF -> 4-char RPN nomvar inverse map for safe early header filtering
_INV_OPENDATA_MAP: dict[str, str] = {
    "AirTemp": "TT",
    "WindU": "UU",
    "WindV": "VV",
    "SpecHum": "HU",
    "SeaSurfTemp": "TM",
    "GeopotentialHeight": "GZ",
    "SurfPres": "P0",
    "SeaLevelPres": "PN",
    "DewPointDepression": "ES",
    "DewPoint": "TD",
    "RelHum": "HR",
    "VertMotion": "WW",
    "Precip": "PR",
    "CloudFrac": "NT",
}

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


def get_last_fst_open_info() -> dict[str, Any]:
    """Return diagnostic telemetry from the most recent `open_fst_xarray(paths)` call."""
    return dict(_LAST_FST_OPEN_INFO)


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


def _is_all_xdf0(paths: Sequence[Path]) -> bool:
    """Verify every file starts with classic RPN `XDF0` magic (required for Tier-A `xdf.unpack_batch`)."""
    for p in paths:
        try:
            with open(p, "rb") as fh:
                if fh.read(4) != b"XDF0":
                    return False
        except OSError:
            return False
    return True


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
                row[0] = t_i
                row[1] = p_base + int(row[1])
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

    _LAST_FST_OPEN_INFO.update(
        {
            "tier": "mfst",
            "direct_catalog_vars": len(hyb_vars) + len(sfc_vars),
            "hlg_fallback_vars": 0,
            "prefiltered_rows": False,
        }
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
# Path 2: Live Operational RPN FST (`.std`) Connector via `boreal-io`
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


@functools.lru_cache(maxsize=16)
def _cached_load_cmc_cabi_so(so_path_str: str, mtime_ns: int) -> tuple[int, Any]:
    _ = mtime_ns
    try:
        cdll = ctypes.CDLL(so_path_str)
        for sym in ("cmc_fst_chunk_decode_cabi", "marray_fst_chunk_decoder"):
            fn = getattr(cdll, sym, None)
            if fn is not None:
                addr = int(ctypes.cast(fn, ctypes.c_void_p).value or 0)
                if addr != 0:
                    return addr, cdll
    except OSError:
        pass
    return 0, None


def _resolve_native_cmc_cabi_symbol(cmc_dir: Path | None) -> tuple[int, Any]:
    """O(1) resolution of Tier-A `ChunkDecoderCABI` `.so` (zero recursive `glob("**")`!).

    Checks `$MARRAY_CMC_DECODER_SO`, `$MARRAY_CMC_CABI_SO` (alias),
    `build/libcmc_marray_bridge.so`, `build/libcmc_fst_cabi.so`, and direct
    `<cmc_dir>/src/cmc_io_fst_data.so`.
    """
    repo_root = Path(__file__).resolve().parents[2]
    so_cands: list[Path] = []
    for env_key in ("MARRAY_CMC_DECODER_SO", "MARRAY_CMC_CABI_SO"):
        val = os.environ.get(env_key, "").strip()
        if val:
            so_cands.append(Path(val).expanduser())
    so_cands.append(repo_root / "build" / "libcmc_marray_bridge.so")
    so_cands.append(repo_root / "build" / "libcmc_fst_cabi.so")
    if cmc_dir is not None:
        so_cands.append(cmc_dir / "build" / "libcmc_marray_bridge.so")
        so_cands.append(cmc_dir / "src" / "cmc_io_fst_data.so")
        so_cands.append(cmc_dir / "cmc_io_fst_data.so")

    for so_p in so_cands:
        try:
            st = so_p.stat()
        except OSError:
            continue
        addr, cdll = _cached_load_cmc_cabi_so(str(so_p), st.st_mtime_ns)
        if addr != 0:
            return addr, cdll
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


def _build_keep_nomvar_set(vars_filter: Sequence[str] | None) -> frozenset[str] | None:
    """Resolve `vars_filter` to stripped + 4-char padded RPN `nomvar`s + required meta companions."""
    if not vars_filter:
        return None
    keep: set[str] = set()
    for item in vars_filter:
        s = str(item).strip()
        if not s:
            continue
        if s in _INV_OPENDATA_MAP:
            nom = _INV_OPENDATA_MAP[s]
        elif len(s) <= 4 and s.upper() == s:
            nom = s
        else:
            # Unknown CF alias -> skip early pre-filter so boreal's naming stage resolves it
            return None
        keep.add(nom)
        keep.add(f"{nom:<4}")
    for m in _RPN_META_NOMVARS:
        keep.add(m)
        keep.add(f"{m:<4}")
    return frozenset(keep)


def _filter_raw_header_rows(rows: Any, keep_nomvars: frozenset[str] | None) -> Any:
    if keep_nomvars is None or rows is None:
        return rows
    # pandas.DataFrame fast vectorized path
    if hasattr(rows, "columns") and "nomvar" in rows.columns:
        nom_col = rows["nomvar"]
        mask = nom_col.isin(keep_nomvars)
        if not bool(mask.all()):
            # Also check `.str.strip()` in case of unusual padding
            stripped_mask = nom_col.astype(str).str.strip().isin(keep_nomvars)
            if "typvar" in rows.columns:
                t_mask = rows["typvar"].astype(str).str.strip().isin({"@@", "!@"})
                stripped_mask = stripped_mask | t_mask
            return rows.loc[mask | stripped_mask].reset_index(drop=True)
        return rows
    # list[dict] path
    if isinstance(rows, list) and rows and isinstance(rows[0], Mapping) and "nomvar" in rows[0]:
        return [
            r
            for r in rows
            if str(r.get("nomvar", "")).strip() in keep_nomvars
            or str(r.get("typvar", "")).strip() in ("@@", "!@")
        ]
    return rows


def _concat_header_parts(parts: list[Any]) -> Any:
    if not parts:
        return parts
    p0 = parts[0]
    if hasattr(p0, "columns"):
        import pandas as pd  # type: ignore[import-not-found]

        return pd.concat(parts, ignore_index=True)
    if isinstance(p0, list):
        out: list[Any] = []
        for part in parts:
            out.extend(part)
        return out
    return parts[0]


@functools.lru_cache(maxsize=8)
def _analyze_stack_fn(fn: Any) -> tuple[inspect.Signature, str]:
    """Inspect `_stack_variable` signature & AST return shape once per process."""
    sig = inspect.signature(fn)
    ret_ann = str(sig.return_annotation)
    if "DataArray" in ret_ann:
        return sig, "dataarray"
    if "Variable" in ret_ann:
        return sig, "variable"
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        mod = ast.parse(src)
        fdef = next((n for n in mod.body if isinstance(n, ast.FunctionDef)), None)
        if fdef is not None:
            ret_kinds: set[str] = set()
            for node in ast.walk(fdef):
                if isinstance(node, ast.Return) and node.value is not None:
                    val = node.value
                    if isinstance(val, ast.Tuple):
                        ret_kinds.add(f"tuple_{len(val.elts)}")
                    elif isinstance(val, ast.Call):
                        unparsed = ast.unparse(val.func)
                        if unparsed.endswith("DataArray"):
                            ret_kinds.add("dataarray")
                        elif unparsed.endswith("Variable"):
                            ret_kinds.add("variable")
            if len(ret_kinds) == 1:
                return sig, next(iter(ret_kinds))
    except Exception:
        pass
    return sig, "array"


def _normalize_time_key(val: Any) -> int | float | str:
    if isinstance(val, np.datetime64) or hasattr(val, "asm8"):
        arr = np.asarray(val, dtype="datetime64[ns]")
        return int(arr.view(np.int64))
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return float(val)
    return str(val)


def _match_axis_indices(
    col_values: Sequence[Any] | np.ndarray, axis_coord: Sequence[Any] | np.ndarray, is_float_axis: bool
) -> np.ndarray | None:
    n_ax = len(axis_coord)
    n_rows = len(col_values)
    if n_ax <= 1:
        return np.zeros(n_rows, dtype=np.int64)

    if is_float_axis:
        try:
            ax_f = np.asarray(axis_coord, dtype=np.float64)
            col_f = np.asarray(col_values, dtype=np.float64)
            exact = {float(v): i for i, v in enumerate(ax_f)}
            out = np.empty(n_rows, dtype=np.int64)
            for i, cv in enumerate(col_f):
                fv = float(cv)
                idx = exact.get(fv, -1)
                if idx < 0:
                    diffs = np.abs(ax_f - fv)
                    best = int(np.argmin(diffs))
                    if diffs[best] <= max(1e-5, 1e-5 * abs(float(ax_f[best]))):
                        idx = best
                    else:
                        return None
                out[i] = idx
            return out
        except Exception:
            return None

    lut = {_normalize_time_key(v): i for i, v in enumerate(axis_coord)}
    out = np.empty(n_rows, dtype=np.int64)
    for i, cv in enumerate(col_values):
        idx = lut.get(_normalize_time_key(cv), -1)
        if idx < 0:
            return None
        out[i] = idx
    return out


def _resolve_fid(path_val: Any, file_index: dict[str, int], ordered_files: list[str]) -> int:
    s = str(path_val)
    fid = file_index.get(s, -1)
    if fid >= 0:
        return fid
    rs = os.path.realpath(s)
    fid = file_index.get(rs, -1)
    if fid >= 0:
        return fid
    fid = len(ordered_files)
    ordered_files.append(s)
    file_index[s] = fid
    file_index[rs] = fid
    return fid


def _fill_meta_row(
    meta: np.ndarray,
    r_idx: int,
    fid: int,
    addr: int,
    lng: int,
    dt: int,
    nb: int,
    m_addr: int,
    m_lng: int,
    use_addr_mode: bool,
) -> None:
    meta[r_idx, 0] = fid
    if use_addr_mode:
        meta[r_idx, 1] = addr
        meta[r_idx, 2] = lng
    else:
        meta[r_idx, 1] = max(0, (addr - 1) * 8)
        meta[r_idx, 2] = lng * 8
    meta[r_idx, 3] = dt
    meta[r_idx, 4] = nb
    if m_addr > 0 and m_lng > 0:
        meta[r_idx, 5] = fid
        meta[r_idx, 6] = m_addr if use_addr_mode else max(0, (m_addr - 1) * 8)
        meta[r_idx, 7] = m_lng if use_addr_mode else m_lng * 8
        meta[r_idx, 8] = 2
        meta[r_idx, 9] = 1


def _safe_opt_int(val: Any, default: int = -1) -> int:
    if val is None:
        return default
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _try_build_duck_from_stack_call(
    orig_stack_var: Any,
    s_args: tuple[Any, ...],
    s_kwargs: dict[str, Any],
    *,
    file_index: dict[str, int],
    ordered_files: list[str],
    use_addr_mode: bool,
    dec_addr: int,
    dec_owner: Any,
    pool_handle: int,
    flags: int,
) -> Any | None:
    """Build `CatalogVarSpec` + `MarrayDuckArray` directly from `_stack_variable` args!

    Returns `None` to fall back cleanly if an exotic cube rank (`member`, `nk > 1`,
    `subgrid != 'all'`, or pre-materialized `d`) is encountered.
    """
    sig, ret_kind = _analyze_stack_fn(orig_stack_var)
    try:
        ba = sig.bind_partial(*s_args, **s_kwargs)
        ba.apply_defaults()
    except TypeError:
        return None
    args_d = ba.arguments

    subgrid = args_d.get("subgrid")
    if subgrid not in (None, "all", ""):
        return None

    # 1. Locate records DataFrame or list[dict] (directly or as `.df`/`.records`/`.rows` attr)
    rows_obj: Any = None
    is_df = False
    owner_obj: Any = None
    cands = list(args_d.values())
    for val in list(cands):
        for attr_n in ("df", "records", "rows", "table"):
            if hasattr(val, attr_n):
                cands.append(getattr(val, attr_n))
                if owner_obj is None:
                    owner_obj = val

    for val in cands:
        if hasattr(val, "columns") and ("addr" in val.columns or "swa" in val.columns) and "lng" in val.columns:
            rows_obj = val
            is_df = True
            break
        if (
            isinstance(val, (list, tuple))
            and val
            and isinstance(val[0], Mapping)
            and ("addr" in val[0] or "swa" in val[0])
            and "lng" in val[0]
        ):
            rows_obj = val
            break
    if rows_obj is None or len(rows_obj) == 0:
        return None

    # Skip if payload column `d` is already materialized
    if is_df:
        if "d" in rows_obj.columns and rows_obj["d"].notna().any():
            return None
        if "nk" in rows_obj.columns and int(rows_obj["nk"].max()) > 1:
            return None
    else:
        if any(r.get("d") is not None for r in rows_obj):
            return None
        if any(int(r.get("nk", 1) or 1) > 1 for r in rows_obj):
            return None

    # 2. Extract target `dims`, `(ni, nj)`, `t_coord`, and `z_coord` (with column-unique fallback!)
    dims_arg = args_d.get("dims") or (getattr(owner_obj, "dims", None) if owner_obj is not None else None)
    coords_arg = args_d.get("coords") or (getattr(owner_obj, "coords", None) if owner_obj is not None else None)
    t_coord: Any = None
    z_coord: Any = None

    for k, v in args_d.items():
        if v is None or isinstance(v, (str, bytes, Mapping)) or hasattr(v, "columns") or not hasattr(v, "__len__"):
            continue
        if len(v) > 0 and isinstance(v[0], Mapping):
            continue
        kl = k.lower()
        if t_coord is None and any(tok in kl for tok in ("time", "date", "valid")):
            t_coord = v
        elif z_coord is None and any(tok in kl for tok in ("lev", "vert", "ip1")):
            z_coord = v

    if isinstance(coords_arg, Mapping):
        if t_coord is None and "time" in coords_arg:
            cv = coords_arg["time"]
            t_coord = getattr(cv, "values", cv)
        if z_coord is None and "vertical" in coords_arg:
            cv = coords_arg["vertical"]
            z_coord = getattr(cv, "values", cv)

    # Identify time & vertical columns in `rows_obj`
    t_col_vals: Any = None
    for tcol in ("time", "valid_time", "datev", "validity", "npas"):
        if is_df and tcol in rows_obj.columns:
            t_col_vals = rows_obj[tcol].to_numpy()
            break
        if not is_df and tcol in rows_obj[0]:
            t_col_vals = [r[tcol] for r in rows_obj]
            break

    z_col_vals: Any = None
    for zcol in ("vertical", "level", "ip1_val", "ip1_decoded", "ip1"):
        if is_df and zcol in rows_obj.columns:
            z_col_vals = rows_obj[zcol].to_numpy()
            break
        if not is_df and zcol in rows_obj[0]:
            z_col_vals = [r[zcol] for r in rows_obj]
            break

    if t_coord is None and t_col_vals is not None:
        seen_t: dict[Any, Any] = {}
        for tv in t_col_vals:
            seen_t.setdefault(_normalize_time_key(tv), tv)
        t_coord = [seen_t[k] for k in sorted(seen_t.keys())]

    if z_coord is None and z_col_vals is not None:
        try:
            u_z = np.unique(np.asarray(z_col_vals, dtype=np.float64))
            if len(u_z) > 1 or (dims_arg is not None and "vertical" in dims_arg):
                z_coord = u_z
        except Exception:
            pass

    if dims_arg is not None and isinstance(dims_arg, (tuple, list)):
        vdims = tuple(str(d) for d in dims_arg)
    else:
        sp_dims = args_d.get("spatial_dims") or args_d.get("yx_dims")
        if sp_dims and len(sp_dims) == 2:
            y_dim, x_dim = str(sp_dims[0]), str(sp_dims[1])
        else:
            y_dim = str(args_d.get("y_dim") or "y")
            x_dim = str(args_d.get("x_dim") or "x")
        d_list: list[str] = []
        if t_coord is not None:
            d_list.append("time")
        if z_coord is not None and len(z_coord) > 1:
            d_list.append("vertical")
        d_list.extend([y_dim, x_dim])
        vdims = tuple(d_list)

    if "member" in vdims or len(vdims) < 2 or len(vdims) > 4:
        return None
    for k in ("members", "member_vals", "member_index"):
        mv = args_d.get(k)
        if mv is not None and hasattr(mv, "__len__") and len(mv) > 1:
            return None

    ni = int(args_d.get("ni") or (rows_obj["ni"].iloc[0] if is_df else rows_obj[0]["ni"]))
    nj = int(args_d.get("nj") or (rows_obj["nj"].iloc[0] if is_df else rows_obj[0]["nj"]))

    has_t = "time" in vdims
    has_z = "vertical" in vdims
    if has_t and t_coord is None:
        return None
    if has_z and z_coord is None:
        return None

    n_t = len(t_coord) if has_t and t_coord is not None else 1
    n_z = len(z_coord) if has_z and z_coord is not None else 1

    # 3. Map each record row to `(t_i, z_i)`
    n_rows = len(rows_obj)
    if has_t and n_t > 1:
        if t_col_vals is None:
            return None
        t_indices = _match_axis_indices(t_col_vals, t_coord, False)
        if t_indices is None:
            return None
    else:
        t_indices = np.zeros(n_rows, dtype=np.int64)

    if has_z and n_z > 1:
        if z_col_vals is None:
            return None
        z_indices = _match_axis_indices(z_col_vals, z_coord, True)
        if z_indices is None:
            return None
    else:
        z_indices = np.zeros(n_rows, dtype=np.int64)

    # 4. Populate `(n_t * n_z, 10)` `rec_meta` table directly (zero Dask!)
    meta = np.zeros((n_t * n_z, REC_META_COLS), dtype=np.int64)
    meta[:, 0] = -1
    meta[:, 5] = -1
    meta[:, 6] = -1

    if is_df:
        addr_key = "addr" if "addr" in rows_obj.columns else "swa"
        paths_arr = rows_obj["path"].to_numpy()
        addrs_arr = rows_obj[addr_key].to_numpy(dtype=np.int64)
        lngs_arr = rows_obj["lng"].to_numpy(dtype=np.int64)
        dts_arr = rows_obj["datyp"].to_numpy(dtype=np.int64) if "datyp" in rows_obj.columns else np.full(n_rows, 134, dtype=np.int64)
        nbs_arr = rows_obj["nbits"].to_numpy(dtype=np.int64) if "nbits" in rows_obj.columns else np.full(n_rows, 16, dtype=np.int64)
        ma_arr = rows_obj["mask_addr"].to_numpy() if "mask_addr" in rows_obj.columns else None
        ml_arr = rows_obj["mask_lng"].to_numpy() if "mask_lng" in rows_obj.columns else None
        for i in range(n_rows):
            r_idx = int(t_indices[i]) * n_z + int(z_indices[i])
            fid = _resolve_fid(paths_arr[i], file_index, ordered_files)
            ma = _safe_opt_int(ma_arr[i], -1) if ma_arr is not None else -1
            ml = _safe_opt_int(ml_arr[i], 0) if ml_arr is not None else 0
            _fill_meta_row(
                meta, r_idx, fid, int(addrs_arr[i]), int(lngs_arr[i]), int(dts_arr[i]), int(nbs_arr[i]), ma, ml, use_addr_mode
            )
    else:
        for i, r in enumerate(rows_obj):
            r_idx = int(t_indices[i]) * n_z + int(z_indices[i])
            fid = _resolve_fid(r["path"], file_index, ordered_files)
            addr = int(r.get("addr", r.get("swa", 0)))
            lng = int(r["lng"])
            dt = int(r.get("datyp", 134))
            nb = int(r.get("nbits", 16))
            ma = _safe_opt_int(r.get("mask_addr"), -1)
            ml = _safe_opt_int(r.get("mask_lng"), 0)
            _fill_meta_row(meta, r_idx, fid, addr, lng, dt, nb, ma, ml, use_addr_mode)

    shape_map: dict[str, int] = {vdims[-2]: nj, vdims[-1]: ni}
    if has_t:
        shape_map["time"] = n_t
    if has_z:
        shape_map["vertical"] = n_z
    lshape = tuple(shape_map[d] for d in vdims)

    vattrs = dict(args_d.get("attrs") or {})
    cspec = mxr.CatalogVarSpec(
        raw_owner=dec_owner,
        raw_base_addr=0,
        rec_meta=meta,
        plane_ni=ni,
        plane_nj=nj,
        rec_s_z=1 if has_z else 0,
        rec_s_t=n_z if has_t else 0,
        decoder_fn_addr=dec_addr,
        decoder_ctx_addr=0,
        file_pool_handle=pool_handle,
        flags=flags,
        logical_shape=lshape,
    )
    coords_1d: dict[str, np.ndarray] = {
        vdims[-2]: np.arange(nj, dtype=np.float64),
        vdims[-1]: np.arange(ni, dtype=np.float64),
    }
    if has_t and t_coord is not None:
        tc = np.asarray(t_coord)
        coords_1d["time"] = tc if np.issubdtype(tc.dtype, np.datetime64) else tc.astype(np.float64, copy=False)
    if has_z and z_coord is not None:
        coords_1d["vertical"] = np.asarray(z_coord, dtype=np.float64)

    vname = str(args_d.get("name") or args_d.get("var_name") or "v")
    mx_var = mxr.Variable(vdims, catalog_spec=cspec, attrs=vattrs)
    mx_ds = mxr.Dataset({vname: mx_var}, coords=coords_1d, domain_dims=vdims)
    duck_arr = mxr.MarrayDuckArray(mx_ds[vname], vdims)
    duck_arr._prov_t_coord = list(t_coord) if (has_t and t_coord is not None) else None  # type: ignore[attr-defined]
    duck_arr._prov_z_coord = np.asarray(z_coord, dtype=np.float64) if (has_z and z_coord is not None) else None  # type: ignore[attr-defined]

    if ret_kind == "variable":
        return xr.Variable(vdims, duck_arr, attrs=vattrs)
    if ret_kind == "dataarray":
        return xr.DataArray(duck_arr, dims=vdims, attrs=vattrs, name=vname)
    if ret_kind == "tuple_2":
        return (vdims, duck_arr)
    if ret_kind == "tuple_3":
        return (vdims, duck_arr, vattrs)
    return duck_arr


def _extract_record_specs_from_dask_da(
    da_xr: xr.DataArray,
    file_index: dict[str, int],
    ordered_files: list[str],
    use_addr_mode: bool,
) -> tuple[np.ndarray, int, int] | None:
    """Fallback extractor supporting BOTH `chunks='column'` (few tasks!) and `chunks='record'`,
    using O(1) `file_index` lookups with ZERO `Path.exists()` filesystem `stat()` syscalls!
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

    def _is_path_arg(x: Any) -> bool:
        if isinstance(x, Path):
            return True
        if isinstance(x, str):
            return x in file_index or "/" in x or "\\" in x or x.endswith((".std", ".fst"))
        return False

    name = dask_arr.name
    matched_any = False
    for key, task in hlg.items():
        if not (isinstance(key, tuple) and key and key[0] == name and isinstance(task, tuple)):
            continue
        t_i = int(key[1]) if has_t and len(key) > 1 else 0
        z_i = (
            int(key[2])
            if (has_t and has_z and len(key) > 2)
            else (int(key[1]) if (not has_t and has_z and len(key) > 1) else 0)
        )
        flat_args = list(task[1:])
        while len(flat_args) == 1 and isinstance(flat_args[0], (tuple, list)):
            flat_args = list(flat_args[0])

        path_s = next((a for a in flat_args if _is_path_arg(a)), None)

        # Column-mode task: carries a list/tuple of per-level spec tuples/dicts!
        seq_specs = next(
            (
                a
                for a in flat_args
                if isinstance(a, (list, tuple))
                and a
                and isinstance(a[0], (tuple, list, Mapping))
            ),
            None,
        )
        if seq_specs is not None and (path_s is not None or isinstance(seq_specs[0], Mapping)):
            for k_z, sp in enumerate(seq_specs):
                r_idx = t_i * n_z + (k_z if len(seq_specs) > 1 else z_i)
                if r_idx >= n_rec:
                    continue
                if isinstance(sp, Mapping):
                    ps = sp.get("path", path_s)
                    if ps is None:
                        continue
                    fid = _resolve_fid(ps, file_index, ordered_files)
                    addr = int(sp.get("addr", sp.get("swa", 0)))
                    lng = int(sp.get("lng", 0))
                    dt = int(sp.get("datyp", 134))
                    nb = int(sp.get("nbits", 16))
                    ma = _safe_opt_int(sp.get("mask_addr"), -1)
                    ml = _safe_opt_int(sp.get("mask_lng"), 0)
                else:
                    if path_s is None or len(sp) < 2:
                        continue
                    fid = _resolve_fid(path_s, file_index, ordered_files)
                    addr, lng = int(sp[0]), int(sp[1])
                    dt = int(sp[5]) if len(sp) >= 6 else 134
                    nb = int(sp[6]) if len(sp) >= 7 else 16
                    ma = _safe_opt_int(sp[7], -1) if len(sp) >= 8 else -1
                    ml = _safe_opt_int(sp[8], 0) if len(sp) >= 9 else 0
                if addr > 0 and lng > 0:
                    _fill_meta_row(meta, r_idx, fid, addr, lng, dt, nb, ma, ml, use_addr_mode)
                    matched_any = True
            continue

        # Record-mode single-record task:
        r_idx = t_i * n_z + z_i
        if r_idx >= n_rec:
            continue
        ints = [int(a) for a in flat_args if isinstance(a, (int, np.integer))]
        if path_s is None and flat_args and isinstance(flat_args[0], Mapping):
            spec_d = flat_args[0]
            path_s = spec_d.get("path")
            if path_s is None:
                continue
            fid = _resolve_fid(path_s, file_index, ordered_files)
            addr = int(spec_d.get("addr", spec_d.get("swa", 0)))
            lng = int(spec_d.get("lng", 0))
            dt = int(spec_d.get("datyp", 134))
            nb = int(spec_d.get("nbits", 16))
            ma = _safe_opt_int(spec_d.get("mask_addr"), -1)
            ml = _safe_opt_int(spec_d.get("mask_lng"), 0)
        elif path_s is not None and len(ints) >= 2:
            fid = _resolve_fid(path_s, file_index, ordered_files)
            addr, lng = ints[0], ints[1]
            dt = ints[5] if len(ints) >= 6 else 134
            nb = ints[6] if len(ints) >= 7 else 16
            ma = ints[7] if len(ints) >= 8 else -1
            ml = ints[8] if len(ints) >= 9 else 0
        else:
            continue

        if addr > 0 and lng > 0:
            _fill_meta_row(meta, r_idx, fid, addr, lng, dt, nb, ma, ml, use_addr_mode)
            matched_any = True

    if not matched_any:
        return None
    return meta, (1 if has_z else 0), (n_z if has_t else 0)


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
    import boreal.io.xarray.open as bopen  # type: ignore[import-not-found]
    import boreal.io.xarray.stack as bstack  # type: ignore[import-not-found]

    str_paths = [str(p) for p in paths]
    ordered_files = list(str_paths)
    file_index: dict[str, int] = {}
    for i, sp in enumerate(ordered_files):
        file_index[sp] = i
        file_index[os.path.realpath(sp)] = i

    native_addr, native_owner = (
        (int(decoder_fn_addr), None)
        if decoder_fn_addr is not None
        else _resolve_native_cmc_cabi_symbol(cmc_dir)
    )
    use_tier_a = native_addr != 0 and _is_all_xdf0(paths)
    use_addr_mode = not use_tier_a

    # Allocate decoder / FilePool BEFORE open_fst so `_stack_variable` builds final MarrayDuckArrays!
    if use_tier_a:
        pool: FilePool | None = FilePool(ordered_files)
        dec_addr = native_addr
        dec_owner: Any = (pool, native_owner)
        pool_handle = pool.handle
        flags = 0
    else:
        pool = None
        dec_addr, dec_owner = _make_boreal_addr_trampoline(ordered_files)
        pool_handle = 0
        flags = CAT_FLAG_ADDR_MODE | CAT_FLAG_FORCE_SERIAL

    keep_nomvars = _build_keep_nomvar_set(vars_filter)
    direct_count = 0
    fallback_count = 0
    did_prefilter = False

    orig_stack_bstack = getattr(bstack, "_stack_variable", None)
    orig_stack_bopen = getattr(bopen, "_stack_variable", None)
    orig_read_bopen = getattr(bopen, "read_records_fast", None)

    def _wrapped_read_records(first_arg: Any, *r_args: Any, **r_kwargs: Any) -> Any:
        nonlocal did_prefilter
        assert orig_read_bopen is not None
        if getattr(_READ_TLS, "active", False):
            return orig_read_bopen(first_arg, *r_args, **r_kwargs)
        _READ_TLS.active = True
        try:
            if (
                isinstance(first_arg, (list, tuple))
                and len(first_arg) > 1
                and isinstance(first_arg[0], (str, os.PathLike))
            ):
                n_w = min(8, len(first_arg))
                with ThreadPoolExecutor(max_workers=n_w) as ex:
                    futs = [
                        ex.submit(
                            lambda p: _filter_raw_header_rows(
                                orig_read_bopen([p], *r_args, **r_kwargs), keep_nomvars
                            ),
                            p_item,
                        )
                        for p_item in first_arg
                    ]
                    parts = [f.result() for f in futs]
                if keep_nomvars is not None:
                    did_prefilter = True
                return _concat_header_parts(parts)

            raw_res = orig_read_bopen(first_arg, *r_args, **r_kwargs)
            if keep_nomvars is not None:
                did_prefilter = True
            return _filter_raw_header_rows(raw_res, keep_nomvars)
        finally:
            _READ_TLS.active = False

    target_orig_stack = orig_stack_bstack or orig_stack_bopen

    def _wrapped_stack_variable(*s_args: Any, **s_kwargs: Any) -> Any:
        nonlocal direct_count
        if target_orig_stack is not None:
            built = _try_build_duck_from_stack_call(
                target_orig_stack,
                s_args,
                s_kwargs,
                file_index=file_index,
                ordered_files=ordered_files,
                use_addr_mode=use_addr_mode,
                dec_addr=dec_addr,
                dec_owner=dec_owner,
                pool_handle=pool_handle,
                flags=flags,
            )
            if built is not None:
                direct_count += 1
                return built
            return target_orig_stack(*s_args, **s_kwargs)
        raise RuntimeError("_stack_variable missing")

    with _OPEN_HOOK_LOCK:
        if orig_read_bopen is not None:
            setattr(bopen, "read_records_fast", _wrapped_read_records)
        if orig_stack_bstack is not None:
            setattr(bstack, "_stack_variable", _wrapped_stack_variable)
        if orig_stack_bopen is not None:
            setattr(bopen, "_stack_variable", _wrapped_stack_variable)
        try:
            # Pass `chunks="column"` so even if an exotic 5D variable falls back,
            # boreal builds `n_t` column tasks instead of `n_t * n_z` record tasks!
            dss = open_fst(
                str_paths,
                vars=list(vars_filter) if vars_filter else None,
                chunks="column",
                engine="mojo",
                open_data_mapping=open_data_mapping,
                cell_measures=cell_measures,
                apply_masks=True,
            )
        finally:
            if orig_read_bopen is not None:
                setattr(bopen, "read_records_fast", orig_read_bopen)
            if orig_stack_bstack is not None:
                setattr(bstack, "_stack_variable", orig_stack_bstack)
            if orig_stack_bopen is not None:
                setattr(bopen, "_stack_variable", orig_stack_bopen)

    # If `FilePool` encountered any canonicalized extra path during open, refresh pool once
    if use_tier_a and pool is not None and len(ordered_files) > len(pool.paths):
        old_pool = pool
        pool = FilePool(ordered_files)
        pool_handle = pool.handle
        dec_owner = (pool, native_owner)
        for ds in dss:
            for da in ds.data_vars.values():
                d_data = da.variable._data
                if isinstance(d_data, mxr.MarrayDuckArray):
                    for src_v in d_data._inner._arena.sources:
                        if src_v.catalog_spec is not None:
                            src_v.catalog_spec.file_pool_handle = pool_handle
                            src_v.catalog_spec.raw_owner = dec_owner
        old_pool.close()

    # Sync final CF coords/attrs onto direct `MarrayDuckArray`s, and convert any HLG fallback vars
    out_xr_list: list[xr.Dataset] = []
    for b_ds in dss:
        needs_fallback_swap: dict[str, tuple[np.ndarray, int, int, tuple[str, ...], dict[str, Any]]] = {}
        for vname, da_xr in b_ds.data_vars.items():
            if vname == "areacell" or da_xr.ndim < 2:
                continue
            vdata = da_xr.variable._data
            if isinstance(vdata, mxr.MarrayDuckArray):
                # Sync Dataset's decoded dims, 1D coords, axis permutation & final stamped attrs!
                real_dims = tuple(str(d) for d in da_xr.dims)
                vdata._active_dims = real_dims
                vdata._inner._arena.domain_dims = real_dims
                vdata._inner._arena.exec_dims = tuple(reversed(real_dims))
                vdata._inner._slices = {
                    d: (0, int(da_xr.sizes[d])) for d in real_dims
                }
                vdata._inner._arena.full_shape_by_dim = {
                    d: int(da_xr.sizes[d]) for d in real_dims
                }
                vdata._inner.attrs.update(da_xr.attrs)

                # Verify / align `rec_meta` row order if `b_ds` sorted `time` or `vertical` differently!
                prov_t = getattr(vdata, "_prov_t_coord", None)
                prov_z = getattr(vdata, "_prov_z_coord", None)
                cs_list = [s.catalog_spec for s in vdata._inner._arena.sources if s.catalog_spec is not None]
                if cs_list:
                    cs0 = cs_list[0]
                    nt_ds = int(da_xr.sizes["time"]) if "time" in real_dims else 1
                    nz_ds = int(da_xr.sizes["vertical"]) if "vertical" in real_dims else 1
                    t_perm = np.arange(nt_ds, dtype=np.int64)
                    z_perm = np.arange(nz_ds, dtype=np.int64)
                    if prov_t is not None and "time" in b_ds.coords and nt_ds > 1 and len(prov_t) == nt_ds:
                        tp = _match_axis_indices(b_ds.coords["time"].values, prov_t, False)
                        if tp is not None:
                            t_perm = tp
                    if prov_z is not None and "vertical" in b_ds.coords and nz_ds > 1 and len(prov_z) == nz_ds:
                        zp = _match_axis_indices(b_ds.coords["vertical"].values, prov_z, True)
                        if zp is not None:
                            z_perm = zp
                    if not (np.array_equal(t_perm, np.arange(nt_ds)) and np.array_equal(z_perm, np.arange(nz_ds))):
                        grid_idx = (t_perm[:, None] * nz_ds + z_perm[None, :]).reshape(-1)
                        cs0.rec_meta = np.ascontiguousarray(cs0.rec_meta[grid_idx])

                new_coords_1d: dict[str, np.ndarray] = {}
                for dn in real_dims:
                    if dn in b_ds.coords:
                        cv = np.asarray(b_ds.coords[dn].values)
                        new_coords_1d[dn] = (
                            cv if np.issubdtype(cv.dtype, np.datetime64) else cv.astype(np.float64, copy=False)
                        )
                    else:
                        new_coords_1d[dn] = np.arange(int(da_xr.sizes[dn]), dtype=np.float64)
                vdata._inner.coords = new_coords_1d
            else:
                ext = _extract_record_specs_from_dask_da(
                    da_xr, file_index, ordered_files, use_addr_mode
                )
                if ext is not None:
                    fallback_count += 1
                    meta, r_sz, r_st = ext
                    needs_fallback_swap[str(vname)] = (
                        meta,
                        r_sz,
                        r_st,
                        tuple(str(d) for d in da_xr.dims),
                        dict(da_xr.attrs),
                    )

        if not needs_fallback_swap:
            if pool is not None and not out_xr_list:
                b_ds.set_close(pool.close)
            out_xr_list.append(b_ds)
            continue

        lead_dims = max((sp[3] for sp in needs_fallback_swap.values()), key=len)
        ni = int(b_ds.sizes[lead_dims[-1]])
        nj = int(b_ds.sizes[lead_dims[-2]])
        mx_vars: dict[str, mxr.Variable] = {}
        for vname, (meta, r_sz, r_st, vdims, vattrs) in needs_fallback_swap.items():
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
        new_data_vars: dict[str, Any] = {}
        for vname in b_ds.data_vars:
            if vname in mx_vars:
                vdims = mx_vars[vname].dims
                new_data_vars[vname] = (
                    vdims,
                    mxr.MarrayDuckArray(mx_ds[vname], vdims),
                    dict(b_ds[vname].attrs),
                )
            else:
                new_data_vars[vname] = b_ds[vname]

        xr_ds = xr.Dataset(new_data_vars, coords=b_ds.coords, attrs=dict(b_ds.attrs))
        if pool is not None and not out_xr_list:
            xr_ds.set_close(pool.close)
        out_xr_list.append(xr_ds)

    _LAST_FST_OPEN_INFO.update(
        {
            "tier": "A" if use_tier_a else "B",
            "direct_catalog_vars": direct_count,
            "hlg_fallback_vars": fallback_count,
            "prefiltered_rows": did_prefilter,
        }
    )
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
