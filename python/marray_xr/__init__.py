"""General Python Xarray + Dask Replacement Backed by Mojo (`marray_xr`).

Implements all Boreal & Aether production contracts (`docs/boreal/01..12` & `docs/questions.md`):
  - **M1 `RecordCatalog` + C-ABI `ChunkDecoder`** (`catalog.py`): Separate-`.so` C-ABI
    whole-record plane decoder (`datyp=134` block float + `datyp=2, nbits=1` `@@` ->
    `float32` `0/1` bitmap planes), per-worker `PlaneSlotCache` (zero duplicate plane
    decompressions on regional `[j0:j1, i0:i1]` slices, bounded RAM), `datyp=6` serial
    safety guard, and cross-`vctype` `list[Dataset]` alignment (`from_sources`).
  - **M4 Phase A Xarray Duck Array** (`duck.py`): `MarrayDuckArray` inside genuine
    `xarray.Dataset` / `xarray.DataArray` (`open_fst_xarray`), bypassing Dask HLG
    while keeping standard Xarray `.isel()`, `.sel(method='nearest')`, ufuncs, & reductions.
  - **Multi-Sink Single-Pass Fusion** (`compute_many`, `to_aether_flat`): Evaluates
    `globavg + zonavg` or batched `aether` `(n_fields, n_src)` finite buffers (`masked_fill=0.0`,
    preventing `NaN` poisoning in `apply_weights_csr`) in ONE pass.
  - **`Float64` Morsel Parity Mode** (`precision='float64'`, `rmse_f64`): Exact IEEE-754
    `float64` parity for `boreal-stat` `values_f64` (`r.diag`, `ke`, `stdev`, `RMSE`).
  - **Column-Tile `(64, nk)` Vertical Scan** (`vertical_column_scan`) & **Chunked
    Streaming `write_fst` Sink** (`stream_write_fst` with `fst_mask_fill=1e30`).
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import xarray as xr


DTYPE_F32 = 0
DTYPE_F64 = 1
DTYPE_U16 = 2

MSRC_NONE = 0
MSRC_SENTINEL_INT = 1
MSRC_NAN = 2
MSRC_BYTE_GT = 3
MSRC_BYTE_NE = 4
MSRC_FST_BITMAP_U8 = 6    # fstd2nc / boreal-io @@ uint8 polarity: keep > 0, mask == 0
MSRC_FST_BITMAP_F32 = 8   # boreal-io `compact_integer_unpack_f32` (`datyp=2`) float32 0/1 plane

BC_LOAD = 1
BC_CONST = 2
BC_ADD = 3
BC_SUB = 4
BC_MUL = 5
BC_DIV = 6
BC_HYPOT = 7
BC_POW2_SUM = 8
BC_MIN = 9
BC_MAX = 10
BC_AFFINE = 11
BC_FMA = 12
BC_NEG = 13
BC_ABS = 14
BC_SQRT = 15
BC_EXP = 16
BC_LOG = 17
BC_SIN = 18
BC_COS = 19
BC_MASK_LT = 20
BC_MASK_LE = 21
BC_MASK_GT = 22
BC_MASK_GE = 23
BC_FILLNA = 24

AGG_MEAN = 0
AGG_SUM = 1
AGG_MIN = 2
AGG_MAX = 3
AGG_STD = 4
AGG_COUNT = 5

VCOL_HYPSOMETRIC_INTEGRAL = 0
VCOL_CAPE_PARCEL_PROXY = 1
VCOL_COLUMN_MAX = 2

FST_MASK_FILL = 1.0e30

GRTYP_CODES = {
    "U": 0,
    "O": 1,
    "Z": 2,
    "L": 3,
    "G": 4,
    "N": 5,
    "M": 6,
}

_UFUNC_MAP = {
    np.add: BC_ADD,
    np.subtract: BC_SUB,
    np.multiply: BC_MUL,
    np.true_divide: BC_DIV,
    np.divide: BC_DIV,
    np.hypot: BC_HYPOT,
    np.minimum: BC_MIN,
    np.maximum: BC_MAX,
    np.negative: BC_NEG,
    np.absolute: BC_ABS,
    np.fabs: BC_ABS,
    np.sqrt: BC_SQRT,
    np.exp: BC_EXP,
    np.log: BC_LOG,
    np.sin: BC_SIN,
    np.cos: BC_COS,
}

_LIB_CACHE: ctypes.CDLL | None = None
_LAST_DECODED_PLANES: int = 0


def get_last_decoded_planes() -> int:
    """Return exact number of `RecordCatalog` planes decoded during the most recent query."""
    return _LAST_DECODED_PLANES


def _resolve_so_path() -> Path:
    env_override = os.environ.get("MARRAY_MOJO_SO")
    if env_override:
        return Path(env_override)
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "build" / "marray_mojo.so",
        repo_root / "build" / "marray_mojo.dylib",
        Path.home() / ".cache" / "marray_mojo.so",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def get_mojo_lib() -> ctypes.CDLL:
    global _LIB_CACHE
    if _LIB_CACHE is not None:
        return _LIB_CACHE

    lib = ctypes.CDLL(str(_resolve_so_path()))
    i64_p = ctypes.POINTER(ctypes.c_int64)
    i32_p = ctypes.POINTER(ctypes.c_int32)
    f32_p = ctypes.POINTER(ctypes.c_float)
    f64_p = ctypes.POINTER(ctypes.c_double)
    u8_p = ctypes.POINTER(ctypes.c_uint8)

    lib.marray_init_runtime.argtypes = []
    lib.marray_init_runtime.restype = ctypes.c_int32

    lib.marray_builtin_fst_chunk_decoder.argtypes = [
        ctypes.c_int64,
        u8_p,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        f32_p,
    ]
    lib.marray_builtin_fst_chunk_decoder.restype = ctypes.c_int32

    lib.marray_pack_fst_var.argtypes = [
        f32_p,
        u8_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int64,
        u8_p,
        i64_p,
    ]
    lib.marray_pack_fst_var.restype = ctypes.c_int64

    lib.marray_file_pool_create.argtypes = [u8_p, i64_p, ctypes.c_int32]
    lib.marray_file_pool_create.restype = ctypes.c_int64

    lib.marray_file_pool_destroy.argtypes = [ctypes.c_int64]
    lib.marray_file_pool_destroy.restype = ctypes.c_int32

    lib.marray_exec_general_plan.argtypes = [
        i64_p,
        f32_p,
        ctypes.c_int64,
        i32_p,
        i32_p,
        i32_p,
        i32_p,
        f32_p,
        f32_p,
        ctypes.c_int64,
        ctypes.c_int64,
        i64_p,
        i64_p,
        i64_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        f64_p,
        f32_p,
    ]
    lib.marray_exec_general_plan.restype = ctypes.c_int32

    lib.marray_exec_multi_sink_plan.argtypes = [
        i64_p,
        f32_p,
        ctypes.c_int64,
        i64_p,
        f32_p,
        ctypes.c_int64,
        i32_p,
        i32_p,
        i32_p,
        i32_p,
        f32_p,
        f32_p,
        ctypes.c_int64,
        ctypes.c_int64,
        i64_p,
        i64_p,
        i64_p,
        ctypes.c_int64,
        i32_p,
        i32_p,
        i32_p,
        i32_p,
        f32_p,
        i64_p,
        f64_p,
    ]
    lib.marray_exec_multi_sink_plan.restype = ctypes.c_int32

    lib.marray_exec_general_plan_f64.argtypes = [
        i64_p,
        f32_p,
        ctypes.c_int64,
        i32_p,
        i32_p,
        i32_p,
        i32_p,
        f32_p,
        f32_p,
        ctypes.c_int64,
        ctypes.c_int64,
        i64_p,
        i64_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
        f64_p,
        f64_p,
    ]
    lib.marray_exec_general_plan_f64.restype = ctypes.c_int32

    lib.marray_exec_vertical_column_scan.argtypes = [
        f32_p,
        f32_p,
        ctypes.c_int32,
        u8_p,
        ctypes.c_int32,
        f32_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        f32_p,
    ]
    lib.marray_exec_vertical_column_scan.restype = ctypes.c_int32

    lib.marray_synth_grid_dims.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        i64_p,
    ]
    lib.marray_synth_grid_dims.restype = ctypes.c_int32

    lib.marray_synth_boreal_cube.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        f32_p,
        f32_p,
        f32_p,
        f32_p,
        u8_p,
        u8_p,
        f32_p,
        u8_p,
        f64_p,
        f64_p,
        f64_p,
        f64_p,
        f32_p,
        f32_p,
        f32_p,
    ]
    lib.marray_synth_boreal_cube.restype = ctypes.c_int32

    lib.marray_init_runtime()
    _LIB_CACHE = lib
    return lib


def get_builtin_decoder_addr() -> int:
    """Return the C-ABI function pointer address of `marray_builtin_fst_chunk_decoder`."""
    lib = get_mojo_lib()
    return int(ctypes.cast(lib.marray_builtin_fst_chunk_decoder, ctypes.c_void_p).value or 0)


def synth_boreal_raw_arenas(
    grtyp: str = "U", scale_factor: int = 1, n_steps: int = -1
) -> dict[str, Any]:
    """Generate catalog-representative Tier-1 FST raw record arenas (`U/O/Z/L/G/N/M`)."""
    lib = get_mojo_lib()
    gcode = GRTYP_CODES[grtyp]
    dims_out = np.zeros(5, dtype=np.int64)
    i64_p = ctypes.POINTER(ctypes.c_int64)
    f32_p = ctypes.POINTER(ctypes.c_float)
    f64_p = ctypes.POINTER(ctypes.c_double)
    u8_p = ctypes.POINTER(ctypes.c_uint8)

    lib.marray_synth_grid_dims(gcode, scale_factor, n_steps, dims_out.ctypes.data_as(i64_p))
    ni, nj, n_lvls, n_t, yin_ny = (int(x) for x in dims_out)

    shape4 = (n_t, n_lvls, nj, ni)
    shape3 = (n_t, nj, ni)
    shape2 = (nj, ni)

    tt_f32 = np.empty(shape4, dtype=np.float32, order="C")
    uu_f32 = np.empty(shape4, dtype=np.float32, order="C")
    vv_f32 = np.empty(shape4, dtype=np.float32, order="C")
    hu_f32 = np.empty(shape4, dtype=np.float32, order="C")
    bmap_3d = np.empty(shape4, dtype=np.uint8, order="C")
    hu_bmap = np.empty(shape4, dtype=np.uint8, order="C")
    tm_f32 = np.empty(shape3, dtype=np.float32, order="C")
    tm_bmap = np.empty(shape3, dtype=np.uint8, order="C")

    x_1d = np.empty(ni, dtype=np.float64)
    y_1d = np.empty(nj, dtype=np.float64)
    levels = np.empty(n_lvls, dtype=np.float64)
    times = np.empty(n_t, dtype=np.float64)
    lon_2d = np.empty(shape2, dtype=np.float32, order="C")
    lat_2d = np.empty(shape2, dtype=np.float32, order="C")
    areacell = np.empty(shape2, dtype=np.float32, order="C")

    lib.marray_synth_boreal_cube(
        gcode,
        scale_factor,
        n_steps,
        tt_f32.ctypes.data_as(f32_p),
        uu_f32.ctypes.data_as(f32_p),
        vv_f32.ctypes.data_as(f32_p),
        hu_f32.ctypes.data_as(f32_p),
        bmap_3d.ctypes.data_as(u8_p),
        hu_bmap.ctypes.data_as(u8_p),
        tm_f32.ctypes.data_as(f32_p),
        tm_bmap.ctypes.data_as(u8_p),
        x_1d.ctypes.data_as(f64_p),
        y_1d.ctypes.data_as(f64_p),
        levels.ctypes.data_as(f64_p),
        times.ctypes.data_as(f64_p),
        lon_2d.ctypes.data_as(f32_p),
        lat_2d.ctypes.data_as(f32_p),
        areacell.ctypes.data_as(f32_p),
    )

    return {
        "grtyp": grtyp,
        "ni": ni,
        "nj": nj,
        "yin_ny": yin_ny,
        "tt_f32": tt_f32,
        "uu_f32": uu_f32,
        "vv_f32": vv_f32,
        "hu_f32": hu_f32,
        "bmap_3d": bmap_3d,
        "hu_bmap": hu_bmap,
        "tm_f32": tm_f32,
        "tm_bmap": tm_bmap,
        "x": x_1d,
        "y": y_1d,
        "vertical": levels,
        "time": times,
        "lon_2d": lon_2d,
        "lat_2d": lat_2d,
        "areacell": areacell,
    }


@dataclass
class CatalogVarSpec:
    """Whole-record plane catalog binding (`Q1-Q4` & `CONTEXT.md` P0: in-RAM or multi-file `pread`)."""

    raw_owner: Any                # keeps `mmap` / `np.ndarray` / `FilePool` / `CFUNCTYPE` alive!
    raw_base_addr: int
    rec_meta: np.ndarray          # contiguous `(n_records, 10)` int64 array
    plane_ni: int
    plane_nj: int
    rec_s_z: int                  # `1` for 4D `(t, z, y, x)`, `0` for 3D `(t, y, x)` surface!
    rec_s_t: int                  # `n_z` for 4D, `1` for 3D surface, `0` for static 2D!
    decoder_fn_addr: int = 0
    decoder_ctx_addr: int = 0
    file_pool_handle: int = 0     # `CatalogFilePool*` handle for multi-file POSIX `pread`
    flags: int = 0                # `CAT_FLAG_FORCE_SERIAL (1)`, `CAT_FLAG_ADDR_MODE (2)`, `(prefix<<8)`
    logical_shape: tuple[int, ...] = ()


class Variable:
    """General N-D variable supporting strided RAM arrays AND `RecordCatalog` compressed planes."""

    def __init__(
        self,
        dims: Sequence[str],
        data: np.ndarray | None = None,
        *,
        scale: float = 1.0,
        offset: float = 0.0,
        fill_value: int | float | None = None,
        fst_bitmap: np.ndarray | None = None,
        qc_flags: np.ndarray | None = None,
        qc_max_valid: int = 1,
        mask: np.ndarray | None = None,
        nan_masked: bool = False,
        catalog_spec: CatalogVarSpec | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        self.dims = tuple(dims)
        self.attrs = dict(attrs or {})
        self.scale = float(scale)
        self.offset = float(offset)
        self.aux_buf: np.ndarray | None = None
        self.catalog_spec = catalog_spec

        if catalog_spec is not None:
            self.data = np.empty(0, dtype=np.float32)
            self.dtype_code = DTYPE_F32
            self.mask_kind = MSRC_FST_BITMAP_F32
            self.mask_param = 0
            self.attrs.setdefault("fst_mask_fill", FST_MASK_FILL)
            return

        assert data is not None, "Either `data` or `catalog_spec` must be provided"
        if np.ma.isMaskedArray(data):
            raw_m = np.ma.getmask(data)
            if raw_m is not np.ma.nomask and mask is None:
                mask = np.asarray(np.ma.getmaskarray(data), dtype=np.uint8)
            data = np.ma.getdata(data)

        self.data = data
        if data.dtype == np.uint16:
            self.dtype_code = DTYPE_U16
        elif data.dtype == np.float64:
            self.dtype_code = DTYPE_F64
        elif data.dtype == np.float32:
            self.dtype_code = DTYPE_F32
        else:
            self.data = np.asarray(data, dtype=np.float32)
            self.dtype_code = DTYPE_F32

        if fst_bitmap is not None:
            if fst_bitmap.dtype == np.float32:
                # Native `boreal-io` `compact_integer_unpack_f32` (`datyp=2`) float32 0/1 plane!
                self.aux_buf = (
                    fst_bitmap
                    if fst_bitmap.flags["C_CONTIGUOUS"] or fst_bitmap.flags["F_CONTIGUOUS"]
                    else np.ascontiguousarray(fst_bitmap)
                )
                self.mask_kind = MSRC_FST_BITMAP_F32
            else:
                self.aux_buf = (
                    fst_bitmap
                    if fst_bitmap.dtype == np.uint8
                    else fst_bitmap.astype(np.uint8, copy=False)
                )
                self.mask_kind = MSRC_FST_BITMAP_U8
            self.mask_param = 0
            self.attrs.setdefault("fst_mask_fill", FST_MASK_FILL)
        elif qc_flags is not None:
            self.aux_buf = qc_flags
            self.mask_kind = MSRC_BYTE_GT
            self.mask_param = int(qc_max_valid)
        elif mask is not None:
            self.aux_buf = (
                mask if mask.dtype == np.uint8 else mask.astype(np.uint8, copy=False)
            )
            self.mask_kind = MSRC_BYTE_NE
            self.mask_param = 0
        elif fill_value is not None and self.dtype_code == DTYPE_U16:
            self.mask_kind = MSRC_SENTINEL_INT
            self.mask_param = int(fill_value)
        elif nan_masked or (fill_value is not None and np.isnan(fill_value)):
            self.mask_kind = MSRC_NAN
            self.mask_param = 0
        else:
            self.mask_kind = MSRC_NONE
            self.mask_param = 0


@dataclass(frozen=True)
class _MaskCond:
    target: "DataArray"
    mask_opcode: int
    threshold: float


@dataclass(frozen=True)
class SinkSpec:
    """Specification for single-pass multi-sink execution (`compute_many`)."""

    da: "DataArray"
    reduce_dims: tuple[str, ...] | None  # `None` = scalar, `()` = materialize, `('time',)` = axis
    agg_mode: int                        # `-1` = materialize, `0..5` = AGG_MEAN..AGG_COUNT
    weight_var: Variable | None = None
    fill_value: float = np.nan
    squeeze_dims: tuple[str, ...] = ()


class _ExprArena:
    """Shared CSE-deduplicated expression graph for a dataset domain."""

    def __init__(
        self,
        domain_dims: tuple[str, ...],
        exec_dims: tuple[str, ...],
        full_shape_by_dim: dict[str, int],
        c_order_domain: bool,
    ) -> None:
        self.domain_dims = domain_dims
        self.exec_dims = exec_dims
        self.full_shape_by_dim = full_shape_by_dim
        self.c_order_domain = c_order_domain
        self.sources: list[Variable] = []
        self.src_load_node: dict[int, int] = {}
        self.nodes: list[tuple[int, int, int, int, float, float]] = []
        self.cse_map: dict[tuple[int, int, int, int, float, float], int] = {}

    def register_source(self, var: Variable) -> int:
        vid = id(var)
        if vid in self.src_load_node:
            return self.src_load_node[vid]
        sid = len(self.sources)
        self.sources.append(var)
        nid = self.emit(BC_LOAD, sid, -1, -1, 0.0, 0.0)
        self.src_load_node[vid] = nid
        return nid

    def import_subdag_from(self, other_arena: "_ExprArena", other_nid: int) -> int:
        """Import a `DataArray` sub-DAG from another `Dataset` group (`Q5`: cross-`vctype`
        surface `P0`/`TM` + 3D hybrid `TT`/`HU` pairing with zero data copy!).
        """
        memo: dict[int, int] = {}

        def _walk(nid: int) -> int:
            if nid in memo:
                return memo[nid]
            op, a, b, c, p0, p1 = other_arena.nodes[nid]
            if op == BC_LOAD:
                res = self.register_source(other_arena.sources[a])
            else:
                na = _walk(a) if a >= 0 else -1
                nb = _walk(b) if b >= 0 else -1
                nc = _walk(c) if c >= 0 else -1
                res = self.emit(op, na, nb, nc, p0, p1)
            memo[nid] = res
            return res

        return _walk(other_nid)

    def emit(
        self,
        op: int,
        a: int = -1,
        b: int = -1,
        c: int = -1,
        p0: float = 0.0,
        p1: float = 0.0,
    ) -> int:
        key = (op, a, b, c, float(p0), float(p1))
        if key in self.cse_map:
            return self.cse_map[key]
        nid = len(self.nodes)
        self.nodes.append(key)
        self.cse_map[key] = nid
        return nid

    def clone_subdag_for_roots(
        self,
        root_nids: Sequence[int],
        extra_weight_vars: Sequence[Variable | None] = (),
    ) -> tuple[list[Variable], np.ndarray, np.ndarray, list[int], list[int]]:
        max_nid = max(root_nids) if root_nids else -1
        needed = [False] * (max_nid + 1)
        for rn in root_nids:
            if 0 <= rn <= max_nid:
                needed[rn] = True
        for i in range(max_nid, -1, -1):
            if not needed[i]:
                continue
            op, a, b, c, _, _ = self.nodes[i]
            if op != BC_LOAD and a >= 0:
                needed[a] = True
            if b >= 0:
                needed[b] = True
            if c >= 0:
                needed[c] = True

        remap: dict[int, int] = {}
        used_sources: list[Variable] = []
        used_src_by_id: dict[int, int] = {}
        compact: list[tuple[int, int, int, int, float, float]] = []

        for i in range(max_nid + 1):
            if not needed[i]:
                continue
            op, a, b, c, p0, p1 = self.nodes[i]
            if op == BC_LOAD:
                sv = self.sources[a]
                vid = id(sv)
                if vid not in used_src_by_id:
                    used_src_by_id[vid] = len(used_sources)
                    used_sources.append(sv)
                na = used_src_by_id[vid]
            else:
                na = remap[a] if a >= 0 else -1
            nb = remap[b] if b >= 0 else -1
            nc = remap[c] if c >= 0 else -1
            remap[i] = len(compact)
            compact.append((op, na, nb, nc, p0, p1))

        wt_sids: list[int] = []
        for wv in extra_weight_vars:
            if wv is None:
                wt_sids.append(-1)
            else:
                vid = id(wv)
                if vid not in used_src_by_id:
                    used_src_by_id[vid] = len(used_sources)
                    used_sources.append(wv)
                wt_sids.append(used_src_by_id[vid])

        remapped_roots = [remap[rn] for rn in root_nids]
        m = len(compact)
        i32_buf = np.empty((4, m), dtype=np.int32)
        f32_buf = np.empty((2, m), dtype=np.float32)
        for k, (op, na, nb, nc, p0, p1) in enumerate(compact):
            i32_buf[0, k] = op
            i32_buf[1, k] = na
            i32_buf[2, k] = nb
            i32_buf[3, k] = nc
            f32_buf[0, k] = p0
            f32_buf[1, k] = p1
        return used_sources, i32_buf, f32_buf, remapped_roots, wt_sids

    def clone_subdag_for_root(
        self, root_nid: int, extra_weight_var: Variable | None = None
    ) -> tuple[list[Variable], np.ndarray, np.ndarray, int]:
        used, i32_b, f32_b, _, wts = self.clone_subdag_for_roots(
            [root_nid], [extra_weight_var]
        )
        return used, i32_b, f32_b, wts[0]


def _coord_as_numeric(coord: np.ndarray) -> np.ndarray:
    if np.issubdtype(coord.dtype, np.datetime64) or np.issubdtype(coord.dtype, np.timedelta64):
        return coord.astype("datetime64[ns]").view(np.int64)
    return coord


def _val_as_numeric(coord: np.ndarray, val: Any) -> float | int:
    if np.issubdtype(coord.dtype, np.datetime64):
        return int(np.datetime64(val, "ns").view(np.int64))
    if np.issubdtype(coord.dtype, np.timedelta64):
        return int(np.timedelta64(val, "ns").view(np.int64))
    return float(val)


def _search_coord_nearest(coord: np.ndarray, val: Any) -> int:
    """O(log N) nearest lookup on monotonic ascending OR descending (`vertical` `ip1`) axes (`Q15`)."""
    n = len(coord)
    if n <= 1:
        return 0
    num_c = _coord_as_numeric(coord)
    q = _val_as_numeric(coord, val)
    if num_c[-1] >= num_c[0]:
        pos = int(np.searchsorted(num_c, q, side="left"))
        if pos <= 0:
            return 0
        if pos >= n:
            return n - 1
        return pos - 1 if abs(q - num_c[pos - 1]) <= abs(num_c[pos] - q) else pos

    neg_c = -num_c
    pos = int(np.searchsorted(neg_c, -q, side="left"))
    if pos <= 0:
        return 0
    if pos >= n:
        return n - 1
    return pos - 1 if abs(num_c[pos - 1] - q) <= abs(q - num_c[pos]) else pos


def _search_coord_slice(coord: np.ndarray, sl: slice) -> tuple[int, int]:
    n = len(coord)
    if n == 0:
        return 0, 0
    if sl.start is None and sl.stop is None:
        return 0, n
    num_c = _coord_as_numeric(coord)
    asc = n < 2 or num_c[-1] >= num_c[0]
    a, b = sl.start, sl.stop
    if asc:
        lo = _val_as_numeric(coord, a) if a is not None else -np.inf
        hi = _val_as_numeric(coord, b) if b is not None else np.inf
        if lo > hi:
            lo, hi = hi, lo
        return int(np.searchsorted(num_c, lo, side="left")), int(
            np.searchsorted(num_c, hi, side="right")
        )
    neg_c = -num_c
    v0 = _val_as_numeric(coord, a) if a is not None else np.inf
    v1 = _val_as_numeric(coord, b) if b is not None else -np.inf
    hi, lo = max(v0, v1), min(v0, v1)
    return int(np.searchsorted(neg_c, -hi, side="left")), int(
        np.searchsorted(neg_c, -lo, side="right")
    )


def _pack_sources_and_catalogs(
    used_sources: list[Variable],
    exec_dims: tuple[str, ...],
    with_catalog_col: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any]]:
    ns = len(used_sources)
    n_cols = 12 if with_catalog_col else 11
    src_i64 = np.zeros((ns, n_cols), dtype=np.int64)
    src_f32 = np.zeros((ns, 2), dtype=np.float32)

    cat_specs: list[CatalogVarSpec] = []
    keep_alive: list[Any] = []

    for i, var in enumerate(used_sources):
        if var.catalog_spec is not None:
            assert with_catalog_col, "Catalog-backed Variable requires multi-sink / catalog executor"
            cs = var.catalog_spec
            cid = len(cat_specs)
            cat_specs.append(cs)
            keep_alive.append(cs.raw_owner)
            keep_alive.append(cs.rec_meta)
            src_i64[i, 11] = cid
            src_f32[i, 0] = var.scale
            src_f32[i, 1] = var.offset
            continue

        item_sz = var.data.dtype.itemsize
        src_i64[i, 0] = var.data.ctypes.data
        src_i64[i, 1] = var.aux_buf.ctypes.data if var.aux_buf is not None else 0
        src_i64[i, 2] = var.dtype_code
        src_i64[i, 3] = var.mask_kind
        src_i64[i, 4] = var.mask_param
        if with_catalog_col:
            src_i64[i, 11] = -1
        src_f32[i, 0] = var.scale
        src_f32[i, 1] = var.offset

        for d_idx, d_name in enumerate(exec_dims):
            if d_name in var.dims:
                v_ax = var.dims.index(d_name)
                elem_stride = (
                    0
                    if var.data.shape[v_ax] <= 1
                    else var.data.strides[v_ax] // item_sz
                )
                src_i64[i, 5 + d_idx] = elem_stride
            else:
                src_i64[i, 5 + d_idx] = 0

    nc = len(cat_specs)
    cat_i64 = np.zeros((max(1, nc), 11), dtype=np.int64)
    cat_f32 = np.zeros((max(1, nc), 2), dtype=np.float32)
    for c_i, cs in enumerate(cat_specs):
        cat_i64[c_i, 0] = cs.raw_base_addr
        cat_i64[c_i, 1] = cs.rec_meta.ctypes.data
        cat_i64[c_i, 2] = cs.rec_meta.shape[0]
        cat_i64[c_i, 3] = cs.plane_ni
        cat_i64[c_i, 4] = cs.plane_nj
        cat_i64[c_i, 5] = cs.rec_s_z
        cat_i64[c_i, 6] = cs.rec_s_t
        cat_i64[c_i, 7] = cs.decoder_fn_addr
        cat_i64[c_i, 8] = cs.decoder_ctx_addr
        cat_i64[c_i, 9] = cs.file_pool_handle
        cat_i64[c_i, 10] = cs.flags
        cat_f32[c_i, 0] = 1.0
        cat_f32[c_i, 1] = 0.0

    return src_i64, src_f32, cat_i64, cat_f32[: max(1, nc)], keep_alive


class DataArray:
    """Lazy labeled masked array preserving Boreal `fst_*` metadata & axis order."""

    __array_priority__ = 1000

    def __init__(
        self,
        arena: _ExprArena,
        node_id: int,
        coords: dict[str, np.ndarray],
        slices_by_dim: dict[str, tuple[int, int]],
        name: str = "expr",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self._arena = arena
        self._node_id = node_id
        self.coords = coords
        self._slices = slices_by_dim
        self.name = name
        self.attrs = dict(attrs or {})

    @property
    def dims(self) -> tuple[str, ...]:
        return self._arena.domain_dims

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(max(0, self._slices[d][1] - self._slices[d][0]) for d in self.dims)

    @property
    def has_catalog_source(self) -> bool:
        return any(s.catalog_spec is not None for s in self._arena.sources)

    def stamp_attrs(self, **new_attrs: Any) -> "DataArray":
        merged = dict(self.attrs)
        merged.update(new_attrs)
        return DataArray(
            self._arena, self._node_id, self.coords, dict(self._slices), self.name, merged
        )

    def _wrap(self, nid: int, name: str = "expr", arena: _ExprArena | None = None, coords: dict[str, np.ndarray] | None = None, slices: dict[str, tuple[int, int]] | None = None) -> "DataArray":
        carried = {
            k: v
            for k, v in self.attrs.items()
            if k.startswith("fst_") and k != "fst_nomvar"
        }
        return DataArray(
            arena or self._arena,
            nid,
            coords if coords is not None else self.coords,
            dict(slices if slices is not None else self._slices),
            name=name,
            attrs=carried,
        )

    def _align_pair(self, other: Any) -> tuple[_ExprArena, int, int, dict[str, np.ndarray], dict[str, tuple[int, int]]]:
        """Support BOTH intra-Dataset ops AND cross-`vctype` `list[Dataset]` pairing (`Q5`)."""
        if not isinstance(other, DataArray):
            return (
                self._arena,
                self._node_id,
                self._arena.emit(BC_CONST, p0=float(other)),
                self.coords,
                self._slices,
            )
        if other._arena is self._arena:
            return self._arena, self._node_id, other._node_id, self.coords, self._slices

        # Cross-Dataset alignment (e.g. 4D hybrid `TT(time, vertical, y, x)` + 3D surface `P0(time, y, x)`)
        if len(other.dims) > len(self.dims):
            nid_self = other._arena.import_subdag_from(self._arena, self._node_id)
            return other._arena, nid_self, other._node_id, other.coords, other._slices
        nid_other = self._arena.import_subdag_from(other._arena, other._node_id)
        return self._arena, self._node_id, nid_other, self.coords, self._slices

    def __add__(self, other: Any) -> "DataArray":
        ar, l, r, co, sl = self._align_pair(other)
        return self._wrap(ar.emit(BC_ADD, l, r), arena=ar, coords=co, slices=sl)

    def __radd__(self, other: Any) -> "DataArray":
        ar, r, l, co, sl = self._align_pair(other)
        return self._wrap(ar.emit(BC_ADD, l, r), arena=ar, coords=co, slices=sl)

    def __sub__(self, other: Any) -> "DataArray":
        ar, l, r, co, sl = self._align_pair(other)
        return self._wrap(ar.emit(BC_SUB, l, r), arena=ar, coords=co, slices=sl)

    def __rsub__(self, other: Any) -> "DataArray":
        ar, r, l, co, sl = self._align_pair(other)
        return self._wrap(ar.emit(BC_SUB, l, r), arena=ar, coords=co, slices=sl)

    def __mul__(self, other: Any) -> "DataArray":
        ar, l, r, co, sl = self._align_pair(other)
        return self._wrap(ar.emit(BC_MUL, l, r), arena=ar, coords=co, slices=sl)

    def __rmul__(self, other: Any) -> "DataArray":
        ar, r, l, co, sl = self._align_pair(other)
        return self._wrap(ar.emit(BC_MUL, l, r), arena=ar, coords=co, slices=sl)

    def __truediv__(self, other: Any) -> "DataArray":
        ar, l, r, co, sl = self._align_pair(other)
        return self._wrap(ar.emit(BC_DIV, l, r), arena=ar, coords=co, slices=sl)

    def __rtruediv__(self, other: Any) -> "DataArray":
        ar, r, l, co, sl = self._align_pair(other)
        return self._wrap(ar.emit(BC_DIV, l, r), arena=ar, coords=co, slices=sl)

    def __neg__(self) -> "DataArray":
        return self._wrap(self._arena.emit(BC_NEG, self._node_id))

    def __abs__(self) -> "DataArray":
        return self._wrap(self._arena.emit(BC_ABS, self._node_id))

    def __pow__(self, power: float) -> "DataArray":
        if power == 2 or power == 2.0:
            return self._wrap(self._arena.emit(BC_MUL, self._node_id, self._node_id))
        if power == 0.5:
            return self._wrap(self._arena.emit(BC_SQRT, self._node_id))
        lg = self._arena.emit(BC_LOG, self._node_id)
        sc = self._arena.emit(BC_AFFINE, lg, p0=float(power), p1=0.0)
        return self._wrap(self._arena.emit(BC_EXP, sc))

    def __ge__(self, thr: float) -> _MaskCond:  # type: ignore[override]
        return _MaskCond(self, BC_MASK_LT, float(thr))

    def __gt__(self, thr: float) -> _MaskCond:  # type: ignore[override]
        return _MaskCond(self, BC_MASK_LE, float(thr))

    def __le__(self, thr: float) -> _MaskCond:  # type: ignore[override]
        return _MaskCond(self, BC_MASK_GT, float(thr))

    def __lt__(self, thr: float) -> _MaskCond:  # type: ignore[override]
        return _MaskCond(self, BC_MASK_GE, float(thr))

    def where(self, cond: _MaskCond) -> "DataArray":
        if not isinstance(cond, _MaskCond) or cond.target._node_id != self._node_id:
            raise ValueError("Currently supports `da.where(da >= threshold)` predicates")
        return self._wrap(
            self._arena.emit(cond.mask_opcode, self._node_id, p0=cond.threshold)
        )

    def fillna(self, value: float) -> "DataArray":
        return self._wrap(self._arena.emit(BC_FILLNA, self._node_id, p0=float(value)))

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method != "__call__":
            return NotImplemented
        if ufunc is np.square and len(inputs) == 1:
            return self ** 2.0
        if ufunc is np.power and len(inputs) == 2 and isinstance(inputs[0], DataArray) and not isinstance(inputs[1], DataArray):
            return inputs[0] ** float(inputs[1])
        if ufunc not in _UFUNC_MAP:
            return NotImplemented
        opcode = _UFUNC_MAP[ufunc]
        if len(inputs) == 1:
            return self._wrap(self._arena.emit(opcode, self._node_id))
        if len(inputs) == 2:
            a_in, b_in = inputs
            if isinstance(a_in, DataArray):
                ar, l, r, co, sl = a_in._align_pair(b_in)
                return a_in._wrap(ar.emit(opcode, l, r), arena=ar, coords=co, slices=sl)
            assert isinstance(b_in, DataArray)
            ar, r, l, co, sl = b_in._align_pair(a_in)
            return b_in._wrap(ar.emit(opcode, l, r), arena=ar, coords=co, slices=sl)
        return NotImplemented

    def sel(self, method: str | None = "nearest", **indexers: Any) -> "DataArray":
        ns = dict(self._slices)
        for d_name, sel_val in indexers.items():
            if d_name == "method":
                continue
            c_arr = self.coords[d_name]
            cur_a, cur_b = ns[d_name]
            if isinstance(sel_val, slice):
                s0, s1 = _search_coord_slice(c_arr, sel_val)
            else:
                idx = _search_coord_nearest(c_arr, sel_val)
                s0, s1 = idx, idx + 1
            ns[d_name] = (max(cur_a, s0), min(cur_b, s1))
        return DataArray(self._arena, self._node_id, self.coords, ns, self.name, self.attrs)

    def isel(self, **indexers: Any) -> "DataArray":
        ns = dict(self._slices)
        for d_name, sel_val in indexers.items():
            cur_a, cur_b = ns[d_name]
            span = cur_b - cur_a
            if isinstance(sel_val, slice):
                start, stop, _ = sel_val.indices(span)
                ns[d_name] = (cur_a + start, cur_a + stop)
            else:
                idx = int(sel_val)
                if idx < 0:
                    idx += span
                ns[d_name] = (cur_a + idx, cur_a + idx + 1)
        return DataArray(self._arena, self._node_id, self.coords, ns, self.name, self.attrs)

    # Lazy sink builders for `mxr.compute_many(...)` (`Q8` multi-sink fusion):
    def lazy_weighted_mean(self, weights: Variable) -> SinkSpec:
        return SinkSpec(self, None, AGG_MEAN, weight_var=weights)

    def lazy_mean(self, dim: str | Sequence[str] | None = None) -> SinkSpec:
        dims_t = None if dim is None else ((dim,) if isinstance(dim, str) else tuple(dim))
        return SinkSpec(self, dims_t, AGG_MEAN)

    def lazy_std(self, dim: str | Sequence[str] | None = None) -> SinkSpec:
        dims_t = None if dim is None else ((dim,) if isinstance(dim, str) else tuple(dim))
        return SinkSpec(self, dims_t, AGG_STD)

    def lazy_materialize(self, *, fill_value: float = np.nan, squeeze: Sequence[str] = ()) -> SinkSpec:
        return SinkSpec(self, (), -1, fill_value=fill_value, squeeze_dims=tuple(squeeze))

    def _execute(
        self,
        reduce_dims: Sequence[str] | None,
        agg_mode: int,
        need_global_std: bool = False,
        weight_var: Variable | None = None,
        fill_value: float = np.nan,
        squeeze_dims: Sequence[str] = (),
        precision: str = "float32",
    ) -> tuple[np.ndarray, xr.DataArray | None]:
        # Route `RecordCatalog`-backed expressions transparently through `compute_many`!
        if self.has_catalog_source and precision != "float64":
            rdims = None if reduce_dims is None else tuple(reduce_dims)
            res = compute_many(
                SinkSpec(
                    self,
                    rdims,
                    agg_mode,
                    weight_var=weight_var,
                    fill_value=fill_value,
                    squeeze_dims=tuple(squeeze_dims),
                ),
                _return_raw_stats=(reduce_dims is None and agg_mode >= 0),
            )[0]
            if reduce_dims is None and agg_mode >= 0:
                return res, None  # type: ignore[return-value]
            return np.zeros(9, dtype=np.float64), res  # type: ignore[return-value]

        lib = get_mojo_lib()
        used_sources, dag_i32, dag_f32, wt_sid = self._arena.clone_subdag_for_root(
            self._node_id, extra_weight_var=weight_var
        )
        exec_dims = self._arena.exec_dims
        ndim = len(exec_dims)
        ns = len(used_sources)
        src_i64, src_f32, _, _, _ = _pack_sources_and_catalogs(
            used_sources, exec_dims, with_catalog_col=False
        )

        full_ext = np.ones(6, dtype=np.int64)
        sl_starts = np.zeros(6, dtype=np.int64)
        sl_stops = np.ones(6, dtype=np.int64)
        for d_idx, d_name in enumerate(exec_dims):
            full_ext[d_idx] = self._arena.full_shape_by_dim[d_name]
            sl_starts[d_idx] = self._slices[d_name][0]
            sl_stops[d_idx] = self._slices[d_name][1]

        out_stats = np.zeros(9, dtype=np.float64)
        use_f64 = precision == "float64"
        out_dtype = np.float64 if use_f64 else np.float32

        if reduce_dims is None and agg_mode >= 0:
            red_mask = -1
            out_arr = np.empty(1, dtype=out_dtype)
            out_dims: tuple[str, ...] = ()
            out_shape: tuple[int, ...] = ()
        else:
            red_set = set(reduce_dims or ())
            red_mask = 0
            kept_exec_dims: list[str] = []
            kept_exec_shape: list[int] = []
            for d_idx, d_name in enumerate(exec_dims):
                if d_name in red_set:
                    red_mask |= 1 << d_idx
                else:
                    kept_exec_dims.append(d_name)
                    a, b = self._slices[d_name]
                    kept_exec_shape.append(max(0, b - a))

            if self._arena.c_order_domain:
                out_dims = tuple(reversed(kept_exec_dims))
                out_shape = tuple(reversed(kept_exec_shape))
                out_arr = np.empty(out_shape if out_shape else (1,), dtype=out_dtype, order="C")
            else:
                out_dims = tuple(kept_exec_dims)
                out_shape = tuple(kept_exec_shape)
                out_arr = np.empty(out_shape if out_shape else (1,), dtype=out_dtype, order="F")

        i64_p = ctypes.POINTER(ctypes.c_int64)
        i32_p = ctypes.POINTER(ctypes.c_int32)
        f32_p = ctypes.POINTER(ctypes.c_float)
        f64_p = ctypes.POINTER(ctypes.c_double)

        if use_f64:
            lib.marray_exec_general_plan_f64(
                src_i64.ctypes.data_as(i64_p),
                src_f32.ctypes.data_as(f32_p),
                ns,
                dag_i32[0].ctypes.data_as(i32_p),
                dag_i32[1].ctypes.data_as(i32_p),
                dag_i32[2].ctypes.data_as(i32_p),
                dag_i32[3].ctypes.data_as(i32_p),
                dag_f32[0].ctypes.data_as(f32_p),
                dag_f32[1].ctypes.data_as(f32_p),
                dag_i32.shape[1],
                ndim,
                sl_starts.ctypes.data_as(i64_p),
                sl_stops.ctypes.data_as(i64_p),
                red_mask,
                agg_mode,
                1 if need_global_std else 0,
                wt_sid,
                float(fill_value),
                out_stats.ctypes.data_as(f64_p),
                out_arr.ctypes.data_as(f64_p),
            )
        else:
            lib.marray_exec_general_plan(
                src_i64.ctypes.data_as(i64_p),
                src_f32.ctypes.data_as(f32_p),
                ns,
                dag_i32[0].ctypes.data_as(i32_p),
                dag_i32[1].ctypes.data_as(i32_p),
                dag_i32[2].ctypes.data_as(i32_p),
                dag_i32[3].ctypes.data_as(i32_p),
                dag_f32[0].ctypes.data_as(f32_p),
                dag_f32[1].ctypes.data_as(f32_p),
                dag_i32.shape[1],
                ndim,
                full_ext.ctypes.data_as(i64_p),
                sl_starts.ctypes.data_as(i64_p),
                sl_stops.ctypes.data_as(i64_p),
                red_mask,
                agg_mode,
                1 if need_global_std else 0,
                wt_sid,
                float(fill_value),
                out_stats.ctypes.data_as(f64_p),
                out_arr.ctypes.data_as(f32_p),
            )

        if reduce_dims is None and agg_mode >= 0:
            return out_stats, None

        sq_set = set(squeeze_dims)
        final_dims = []
        final_shape = []
        kept_coords = {}
        for d_name, sz in zip(out_dims, out_shape):
            if d_name in sq_set and sz == 1:
                continue
            final_dims.append(d_name)
            final_shape.append(sz)
            if d_name in self.coords:
                a, b = self._slices[d_name]
                kept_coords[d_name] = self.coords[d_name][a:b]

        if len(final_shape) != len(out_shape):
            out_arr = out_arr.reshape(tuple(final_shape))

        xr_da = xr.DataArray(
            out_arr, dims=tuple(final_dims), coords=kept_coords, name=self.name, attrs=dict(self.attrs)
        )
        return out_stats, xr_da

    def stats(self, *, precision: str = "float32") -> dict[str, float | int]:
        st, _ = self._execute(
            reduce_dims=None, agg_mode=AGG_MEAN, need_global_std=True, precision=precision
        )
        return {
            "mean": float(st[0]),
            "std": float(st[1]),
            "min": float(st[2]),
            "max": float(st[3]),
            "sum": float(st[4]),
            "count": int(st[5]),
            "total": int(st[6]),
            "opt_insts": int(st[7]),
            "opt_regs": int(st[8]),
        }

    def weighted_mean(self, weights: Variable, *, precision: str = "float32") -> float:
        st, _ = self._execute(
            reduce_dims=None, agg_mode=AGG_MEAN, weight_var=weights, precision=precision
        )
        return float(st[0])

    def _reduce_dispatch(
        self,
        dim: str | Sequence[str] | None,
        agg_mode: int,
        stat_idx: int,
        precision: str = "float32",
    ) -> float | int | xr.DataArray:
        if dim is None:
            st, _ = self._execute(
                reduce_dims=None,
                agg_mode=agg_mode,
                need_global_std=(agg_mode == AGG_STD),
                precision=precision,
            )
            return int(st[stat_idx]) if agg_mode == AGG_COUNT else float(st[stat_idx])
        dims_seq = (dim,) if isinstance(dim, str) else tuple(dim)
        _, xr_da = self._execute(reduce_dims=dims_seq, agg_mode=agg_mode, precision=precision)
        assert xr_da is not None
        return xr_da

    def mean(self, dim: str | Sequence[str] | None = None, *, precision: str = "float32") -> float | xr.DataArray:
        return self._reduce_dispatch(dim, AGG_MEAN, 0, precision)  # type: ignore[return-value]

    def std(self, dim: str | Sequence[str] | None = None, *, precision: str = "float32") -> float | xr.DataArray:
        return self._reduce_dispatch(dim, AGG_STD, 1, precision)  # type: ignore[return-value]

    def min(self, dim: str | Sequence[str] | None = None) -> float | xr.DataArray:
        return self._reduce_dispatch(dim, AGG_MIN, 2)  # type: ignore[return-value]

    def max(self, dim: str | Sequence[str] | None = None) -> float | xr.DataArray:
        return self._reduce_dispatch(dim, AGG_MAX, 3)  # type: ignore[return-value]

    def sum(self, dim: str | Sequence[str] | None = None, *, precision: str = "float32") -> float | xr.DataArray:
        return self._reduce_dispatch(dim, AGG_SUM, 4, precision)  # type: ignore[return-value]

    def count(self, dim: str | Sequence[str] | None = None) -> int | xr.DataArray:
        return self._reduce_dispatch(dim, AGG_COUNT, 5)  # type: ignore[return-value]

    def compute(
        self,
        *,
        fill_value: float = np.nan,
        squeeze: Sequence[str] = (),
        precision: str = "float32",
    ) -> xr.DataArray:
        _, xr_da = self._execute(
            reduce_dims=(),
            agg_mode=-1,
            fill_value=fill_value,
            squeeze_dims=squeeze,
            precision=precision,
        )
        assert xr_da is not None
        return xr_da

    def to_fst_buffer(self) -> xr.DataArray:
        fill = float(self.attrs.get("fst_mask_fill", FST_MASK_FILL))
        return self.compute(fill_value=fill)

    def stats_and_reduce(
        self, dim: str | Sequence[str], agg: str = "mean"
    ) -> tuple[dict[str, float | int], xr.DataArray]:
        agg_lut = {
            "mean": AGG_MEAN,
            "sum": AGG_SUM,
            "min": AGG_MIN,
            "max": AGG_MAX,
            "std": AGG_STD,
            "count": AGG_COUNT,
        }
        dims_seq = (dim,) if isinstance(dim, str) else tuple(dim)
        st, xr_da = self._execute(reduce_dims=dims_seq, agg_mode=agg_lut[agg])
        assert xr_da is not None
        info = {
            "mean": float(st[0]),
            "min": float(st[2]),
            "max": float(st[3]),
            "sum": float(st[4]),
            "count": int(st[5]),
            "total": int(st[6]),
            "opt_insts": int(st[7]),
            "opt_regs": int(st[8]),
        }
        return info, xr_da


def compute_many(
    *items: DataArray | SinkSpec, _return_raw_stats: bool = False
) -> list[Any]:
    """Execute multiple lazy reductions / materializations in ONE single-pass Mojo scan (`Q8`).

    Example (`boreal-stat` `ke` -> `globavg` + `zonavg` in one pass):
        `glob_ke, zon_ke = mxr.compute_many(ke.lazy_weighted_mean(areacell), ke.lazy_mean(('time', 'x')))`
    """
    if not items:
        return []
    specs: list[SinkSpec] = [
        it if isinstance(it, SinkSpec) else SinkSpec(it, (), -1) for it in items
    ]
    lead = specs[0].da
    arena = lead._arena
    exec_dims = arena.exec_dims
    ndim = len(exec_dims)

    root_nids: list[int] = []
    wt_vars: list[Variable | None] = []
    for sp in specs:
        if sp.da._arena is arena:
            root_nids.append(sp.da._node_id)
        else:
            root_nids.append(arena.import_subdag_from(sp.da._arena, sp.da._node_id))
        wt_vars.append(sp.weight_var)

    used_sources, dag_i32, dag_f32, remapped_roots, wt_sids = arena.clone_subdag_for_roots(
        root_nids, wt_vars
    )
    src_i64, src_f32, cat_i64, cat_f32, _keep = _pack_sources_and_catalogs(
        used_sources, exec_dims, with_catalog_col=True
    )
    n_cats = sum(1 for v in used_sources if v.catalog_spec is not None)

    full_ext = np.ones(6, dtype=np.int64)
    sl_starts = np.zeros(6, dtype=np.int64)
    sl_stops = np.ones(6, dtype=np.int64)
    for d_idx, d_name in enumerate(exec_dims):
        full_ext[d_idx] = arena.full_shape_by_dim[d_name]
        sl_starts[d_idx] = lead._slices[d_name][0]
        sl_stops[d_idx] = lead._slices[d_name][1]

    k_sinks = len(specs)
    sink_roots_i32 = np.array(remapped_roots, dtype=np.int32)
    sink_rmasks_i32 = np.zeros(k_sinks, dtype=np.int32)
    sink_aggs_i32 = np.zeros(k_sinks, dtype=np.int32)
    sink_wts_i32 = np.array(wt_sids, dtype=np.int32)
    sink_fills_f32 = np.zeros(k_sinks, dtype=np.float32)
    sink_addrs_i64 = np.zeros(k_sinks, dtype=np.int64)

    out_arrays: list[tuple[np.ndarray, tuple[str, ...], tuple[int, ...]]] = []
    dummy = np.empty(1, dtype=np.float32)

    for s_i, sp in enumerate(specs):
        sink_aggs_i32[s_i] = sp.agg_mode
        sink_fills_f32[s_i] = float(sp.fill_value)
        if sp.reduce_dims is None and sp.agg_mode >= 0:
            sink_rmasks_i32[s_i] = -1
            sink_addrs_i64[s_i] = dummy.ctypes.data
            out_arrays.append((dummy, (), ()))
        else:
            red_set = set(sp.reduce_dims or ())
            rm = 0
            kept_ed: list[str] = []
            kept_es: list[int] = []
            for d_idx, d_name in enumerate(exec_dims):
                if d_name in red_set:
                    rm |= 1 << d_idx
                else:
                    kept_ed.append(d_name)
                    a, b = lead._slices[d_name]
                    kept_es.append(max(0, b - a))
            sink_rmasks_i32[s_i] = rm
            if arena.c_order_domain:
                od = tuple(reversed(kept_ed))
                osh = tuple(reversed(kept_es))
                arr = np.empty(osh if osh else (1,), dtype=np.float32, order="C")
            else:
                od = tuple(kept_ed)
                osh = tuple(kept_es)
                arr = np.empty(osh if osh else (1,), dtype=np.float32, order="F")
            sink_addrs_i64[s_i] = arr.ctypes.data
            out_arrays.append((arr, od, osh))

    out_stats = np.zeros((k_sinks, 9), dtype=np.float64)
    lib = get_mojo_lib()
    i64_p = ctypes.POINTER(ctypes.c_int64)
    i32_p = ctypes.POINTER(ctypes.c_int32)
    f32_p = ctypes.POINTER(ctypes.c_float)
    f64_p = ctypes.POINTER(ctypes.c_double)

    lib.marray_exec_multi_sink_plan(
        src_i64.ctypes.data_as(i64_p),
        src_f32.ctypes.data_as(f32_p),
        len(used_sources),
        cat_i64.ctypes.data_as(i64_p),
        cat_f32.ctypes.data_as(f32_p),
        n_cats,
        dag_i32[0].ctypes.data_as(i32_p),
        dag_i32[1].ctypes.data_as(i32_p),
        dag_i32[2].ctypes.data_as(i32_p),
        dag_i32[3].ctypes.data_as(i32_p),
        dag_f32[0].ctypes.data_as(f32_p),
        dag_f32[1].ctypes.data_as(f32_p),
        dag_i32.shape[1],
        ndim,
        full_ext.ctypes.data_as(i64_p),
        sl_starts.ctypes.data_as(i64_p),
        sl_stops.ctypes.data_as(i64_p),
        k_sinks,
        sink_roots_i32.ctypes.data_as(i32_p),
        sink_rmasks_i32.ctypes.data_as(i32_p),
        sink_aggs_i32.ctypes.data_as(i32_p),
        sink_wts_i32.ctypes.data_as(i32_p),
        sink_fills_f32.ctypes.data_as(f32_p),
        sink_addrs_i64.ctypes.data_as(i64_p),
        out_stats.ctypes.data_as(f64_p),
    )

    global _LAST_DECODED_PLANES
    if n_cats > 0 and k_sinks > 0:
        _LAST_DECODED_PLANES = int(out_stats[0, 8])

    results: list[Any] = []
    stat_slot = {AGG_MEAN: 0, AGG_STD: 1, AGG_MIN: 2, AGG_MAX: 3, AGG_SUM: 4, AGG_COUNT: 5}
    for s_i, sp in enumerate(specs):
        if sp.reduce_dims is None and sp.agg_mode >= 0:
            if _return_raw_stats:
                results.append(out_stats[s_i].copy())
            else:
                val = out_stats[s_i, stat_slot[sp.agg_mode]]
                results.append(int(val) if sp.agg_mode == AGG_COUNT else float(val))
        else:
            arr, od, osh = out_arrays[s_i]
            sq_set = set(sp.squeeze_dims)
            f_dims = []
            f_sh = []
            k_coords = {}
            for dn, sz in zip(od, osh):
                if dn in sq_set and sz == 1:
                    continue
                f_dims.append(dn)
                f_sh.append(sz)
                if dn in sp.da.coords:
                    a, b = lead._slices[dn]
                    k_coords[dn] = sp.da.coords[dn][a:b]
            if len(f_sh) != len(osh):
                arr = arr.reshape(tuple(f_sh))
            results.append(
                xr.DataArray(
                    arr,
                    dims=tuple(f_dims),
                    coords=k_coords,
                    name=sp.da.name,
                    attrs=dict(sp.da.attrs),
                )
            )
    return results


def to_aether_flat(
    fields: Sequence[DataArray],
    *,
    masked_fill: float = 0.0,
) -> np.ndarray:
    """Materialize `n_fields` 2D slices/expressions in ONE fused Mojo call directly into
    `aether.apply_weights_csr`'s `(n_fields, n_src)` C-contiguous contract (`Q10`, `Q11`),
    substituting finite `masked_fill` (`0.0` default, NEVER `NaN`!) on masked source cells.
    """
    if not fields:
        return np.empty((0, 0), dtype=np.float32)
    lead = fields[0]
    arena = lead._arena
    exec_dims = arena.exec_dims
    ndim = len(exec_dims)

    n_src = 1
    for d_name in exec_dims:
        a, b = lead._slices[d_name]
        n_src *= max(0, b - a)

    n_fields = len(fields)
    flat = np.empty((n_fields, n_src), dtype=np.float32, order="C")

    root_nids = [
        f._node_id if f._arena is arena else arena.import_subdag_from(f._arena, f._node_id)
        for f in fields
    ]
    used_sources, dag_i32, dag_f32, remapped_roots, _ = arena.clone_subdag_for_roots(
        root_nids, [None] * n_fields
    )
    src_i64, src_f32, cat_i64, cat_f32, _keep = _pack_sources_and_catalogs(
        used_sources, exec_dims, with_catalog_col=True
    )
    n_cats = sum(1 for v in used_sources if v.catalog_spec is not None)

    full_ext = np.ones(6, dtype=np.int64)
    sl_starts = np.zeros(6, dtype=np.int64)
    sl_stops = np.ones(6, dtype=np.int64)
    for d_idx, d_name in enumerate(exec_dims):
        full_ext[d_idx] = arena.full_shape_by_dim[d_name]
        sl_starts[d_idx] = lead._slices[d_name][0]
        sl_stops[d_idx] = lead._slices[d_name][1]

    sink_roots = np.array(remapped_roots, dtype=np.int32)
    sink_rmasks = np.zeros(n_fields, dtype=np.int32)
    sink_aggs = np.full(n_fields, -1, dtype=np.int32)
    sink_wts = np.full(n_fields, -1, dtype=np.int32)
    sink_fills = np.full(n_fields, float(masked_fill), dtype=np.float32)
    sink_addrs = np.array([flat[f].ctypes.data for f in range(n_fields)], dtype=np.int64)
    out_stats = np.zeros((n_fields, 9), dtype=np.float64)

    lib = get_mojo_lib()
    i64_p = ctypes.POINTER(ctypes.c_int64)
    i32_p = ctypes.POINTER(ctypes.c_int32)
    f32_p = ctypes.POINTER(ctypes.c_float)
    f64_p = ctypes.POINTER(ctypes.c_double)

    lib.marray_exec_multi_sink_plan(
        src_i64.ctypes.data_as(i64_p),
        src_f32.ctypes.data_as(f32_p),
        len(used_sources),
        cat_i64.ctypes.data_as(i64_p),
        cat_f32.ctypes.data_as(f32_p),
        n_cats,
        dag_i32[0].ctypes.data_as(i32_p),
        dag_i32[1].ctypes.data_as(i32_p),
        dag_i32[2].ctypes.data_as(i32_p),
        dag_i32[3].ctypes.data_as(i32_p),
        dag_f32[0].ctypes.data_as(f32_p),
        dag_f32[1].ctypes.data_as(f32_p),
        dag_i32.shape[1],
        ndim,
        full_ext.ctypes.data_as(i64_p),
        sl_starts.ctypes.data_as(i64_p),
        sl_stops.ctypes.data_as(i64_p),
        n_fields,
        sink_roots.ctypes.data_as(i32_p),
        sink_rmasks.ctypes.data_as(i32_p),
        sink_aggs.ctypes.data_as(i32_p),
        sink_wts.ctypes.data_as(i32_p),
        sink_fills.ctypes.data_as(f32_p),
        sink_addrs.ctypes.data_as(i64_p),
        out_stats.ctypes.data_as(f64_p),
    )
    return flat


def rmse_f64(pred: DataArray, obs: DataArray, *, weights: Variable | None = None) -> float:
    """Exact `Float64` pair `RMSE` (`boreal-stat` `values_f64` parity gate, `Q7`)."""
    sq_err = (pred - obs) ** 2
    if weights is not None:
        mse = sq_err.weighted_mean(weights, precision="float64")
    else:
        mse = float(sq_err.mean(precision="float64"))  # type: ignore[arg-type]
    return float(np.sqrt(mse))


def vertical_column_scan(
    tt_var: Variable,
    levels: np.ndarray,
    *,
    hu_var: Variable | None = None,
    coords: Mapping[str, np.ndarray] | None = None,
    mode: str = "cape_proxy",
    ext_kernel_fn_addr: int = 0,
    ext_kernel_ctx_addr: int = 0,
    fill_value: float = np.nan,
    nomvar: str = "CAPE",
) -> xr.DataArray:
    """Execute `TILE_COLS = 64` L1d-resident `(column, level)` vertical physics (`Q6`)
    without peeling or transposing the 4D `(n_t, nk, nj, ni)` cube in Python.
    """
    mode_map = {
        "hypsometric": VCOL_HYPSOMETRIC_INTEGRAL,
        "cape_proxy": VCOL_CAPE_PARCEL_PROXY,
        "column_max": VCOL_COLUMN_MAX,
    }
    lib = get_mojo_lib()
    n_t, nk, nj, ni = tt_var.data.shape
    lvls_f32 = np.ascontiguousarray(levels, dtype=np.float32)
    out = np.empty((n_t, nj, ni), dtype=np.float32, order="C")

    f32_p = ctypes.POINTER(ctypes.c_float)
    u8_p = ctypes.POINTER(ctypes.c_uint8)
    bmap_kind = 0
    bmap_ptr = tt_var.data.ctypes.data_as(u8_p)
    if tt_var.aux_buf is not None:
        bmap_ptr = tt_var.aux_buf.ctypes.data_as(u8_p)
        bmap_kind = 2 if tt_var.mask_kind == MSRC_FST_BITMAP_F32 else 1

    hu_ptr = (
        hu_var.data.ctypes.data_as(f32_p)
        if hu_var is not None
        else tt_var.data.ctypes.data_as(f32_p)
    )

    lib.marray_exec_vertical_column_scan(
        tt_var.data.ctypes.data_as(f32_p),
        hu_ptr,
        1 if hu_var is not None else 0,
        bmap_ptr,
        bmap_kind,
        lvls_f32.ctypes.data_as(f32_p),
        n_t,
        nk,
        nj,
        ni,
        mode_map[mode],
        int(ext_kernel_fn_addr),
        int(ext_kernel_ctx_addr),
        float(fill_value),
        out.ctypes.data_as(f32_p),
    )

    out_coords = {}
    if coords is not None:
        for k in ("time", "y", "x"):
            if k in coords:
                out_coords[k] = coords[k]
    attrs = {
        k: v for k, v in tt_var.attrs.items() if k.startswith("fst_") and k != "fst_nomvar"
    }
    attrs["fst_nomvar"] = nomvar
    attrs.setdefault("fst_mask_fill", FST_MASK_FILL)
    return xr.DataArray(out, dims=("time", "y", "x"), coords=out_coords, name=nomvar, attrs=attrs)


def stream_write_fst(
    fields: DataArray | Mapping[str, DataArray],
    record_sink: Callable[[dict[str, Any], np.ndarray], None],
    *,
    time_batch: int = 1,
    fill_value: float = FST_MASK_FILL,
) -> int:
    """Chunked streaming `write_fst` sink (`Q14`): evaluates PB-scale lazy graphs in
    bounded `time_batch` windows with direct SIMD `1e30` (`fst_mask_fill`) substitution
    and stamped `fst_*` record headers—never allocating a full 4D output cube!
    """
    field_map = {fields.name: fields} if isinstance(fields, DataArray) else dict(fields)
    if not field_map:
        return 0
    lead = next(iter(field_map.values()))
    has_time = "time" in lead.dims
    t_total = lead.shape[lead.dims.index("time")] if has_time else 1
    records_written = 0

    for t0 in range(0, t_total, max(1, time_batch)):
        t1 = min(t_total, t0 + max(1, time_batch))
        sliced_specs = [
            (
                da.isel(time=slice(t0, t1)) if has_time else da
            ).lazy_materialize(fill_value=fill_value)
            for da in field_map.values()
        ]
        batch_cubes = compute_many(*sliced_specs)

        for (fname, orig_da), xr_chunk in zip(field_map.items(), batch_cubes):
            nomvar = orig_da.attrs.get("fst_nomvar", fname[:4].upper())
            grtyp = orig_da.attrs.get("fst_grtyp", "U")
            arr = xr_chunk.values
            t_coords = xr_chunk.coords.get("time", np.arange(t0, t1))
            has_vert = "vertical" in xr_chunk.dims
            z_coords = xr_chunk.coords.get("vertical", np.array([0.0]))

            for bt in range(arr.shape[0] if has_time else 1):
                if has_vert:
                    for bz in range(arr.shape[1]):
                        plane = arr[bt, bz] if has_time else arr[bz]
                        hdr = {
                            "nomvar": nomvar,
                            "grtyp": grtyp,
                            "ip1": float(z_coords[bz]),
                            "time": t_coords[bt] if has_time else 0,
                            "ni": int(plane.shape[-1]),
                            "nj": int(plane.shape[-2]),
                            "fst_mask_fill": float(fill_value),
                        }
                        record_sink(hdr, plane)
                        records_written += 1
                else:
                    plane = arr[bt] if has_time else arr
                    hdr = {
                        "nomvar": nomvar,
                        "grtyp": grtyp,
                        "ip1": 0.0,
                        "time": t_coords[bt] if has_time else 0,
                        "ni": int(plane.shape[-1]),
                        "nj": int(plane.shape[-2]),
                        "fst_mask_fill": float(fill_value),
                    }
                    record_sink(hdr, plane)
                    records_written += 1

    return records_written


class Dataset:
    """General labeled N-D Dataset implementing Boreal CF + `fst_*` contracts."""

    def __init__(
        self,
        data_vars: Mapping[str, Variable],
        coords: Mapping[str, np.ndarray],
        *,
        domain_dims: Sequence[str] | None = None,
        cell_measures: Mapping[str, Variable] | None = None,
        extra_coords_2d: Mapping[str, tuple[tuple[str, str], np.ndarray]] | None = None,
        close_fn: Callable[[], None] | None = None,
        attrs: Mapping[str, Any] | None = None,
        _arena: _ExprArena | None = None,
        _var_nodes: dict[str, int] | None = None,
        _var_attrs: dict[str, dict[str, Any]] | None = None,
        _slices: dict[str, tuple[int, int]] | None = None,
        _measures: dict[str, Variable] | None = None,
    ) -> None:
        self.coords = {
            k: (v if np.issubdtype(np.asarray(v).dtype, np.datetime64) else np.asarray(v, dtype=np.float64))
            for k, v in coords.items()
        }
        self.extra_coords_2d = dict(extra_coords_2d or {})
        self._close_fn = close_fn
        self.attrs = dict(attrs or {})

        if _arena is not None and _var_nodes is not None and _slices is not None:
            self._arena = _arena
            self._var_nodes = _var_nodes
            self._var_attrs = _var_attrs or {}
            self._slices = _slices
            self.cell_measures = _measures or {}
            return

        lead_var = max(data_vars.values(), key=lambda v: len(v.dims))
        if domain_dims is None:
            ordered: list[str] = []
            for v in sorted(data_vars.values(), key=lambda x: -len(x.dims)):
                for d in v.dims:
                    if d not in ordered:
                        ordered.append(d)
            dom_dims = tuple(ordered)
        else:
            dom_dims = tuple(domain_dims)

        if lead_var.catalog_spec is not None:
            c_order = True
            exec_dims = tuple(reversed(dom_dims))
        elif len(lead_var.dims) >= 2 and lead_var.data.strides[-1] < lead_var.data.strides[0]:
            c_order = True
            exec_dims = tuple(reversed(dom_dims))
        else:
            c_order = False
            exec_dims = dom_dims

        full_shape_by_dim = {d: len(self.coords[d]) for d in dom_dims}
        self._arena = _ExprArena(dom_dims, exec_dims, full_shape_by_dim, c_order)
        self._var_nodes = {
            name: self._arena.register_source(var) for name, var in data_vars.items()
        }
        self._var_attrs = {name: dict(var.attrs) for name, var in data_vars.items()}
        self._slices = {d: (0, full_shape_by_dim[d]) for d in dom_dims}
        self.cell_measures = dict(cell_measures or {})

    @property
    def dims(self) -> dict[str, int]:
        return {
            d: max(0, self._slices[d][1] - self._slices[d][0])
            for d in self._arena.domain_dims
        }

    def __getitem__(self, name: str) -> DataArray:
        return DataArray(
            self._arena,
            self._var_nodes[name],
            self.coords,
            dict(self._slices),
            name=name,
            attrs=self._var_attrs.get(name),
        )

    def find_by_nomvar(self, nomvar: str) -> DataArray:
        if nomvar in self._var_nodes:
            return self[nomvar]
        for vname, vattrs in self._var_attrs.items():
            if vattrs.get("fst_nomvar") == nomvar or vattrs.get("nomvar") == nomvar:
                return self[vname]
        raise KeyError(f"No variable matching nomvar={nomvar!r}")

    def sel(self, method: str | None = "nearest", **indexers: Any) -> "Dataset":
        probe = DataArray(self._arena, 0, self.coords, dict(self._slices))
        sliced = probe.sel(method=method, **indexers)
        return Dataset(
            {},
            self.coords,
            attrs=self.attrs,
            _arena=self._arena,
            _var_nodes=self._var_nodes,
            _var_attrs=self._var_attrs,
            _slices=sliced._slices,
            _measures=self.cell_measures,
        )

    def isel(self, **indexers: Any) -> "Dataset":
        probe = DataArray(self._arena, 0, self.coords, dict(self._slices))
        sliced = probe.isel(**indexers)
        return Dataset(
            {},
            self.coords,
            attrs=self.attrs,
            _arena=self._arena,
            _var_nodes=self._var_nodes,
            _var_attrs=self._var_attrs,
            _slices=sliced._slices,
            _measures=self.cell_measures,
        )


def from_sources(sources: Sequence[Dataset], nomvar: str) -> DataArray:
    """Resolve `nomvar` across `list[Dataset]` (`Q5`: `split_dataset_groups` by `vctype`)."""
    for ds in sources:
        try:
            return ds.find_by_nomvar(nomvar)
        except KeyError:
            continue
    raise KeyError(f"nomvar={nomvar!r} not found in any of {len(sources)} Datasets")


from .catalog import (  # noqa: E402
    FilePool,
    PackedFSTArchive,
    open_fst_catalog,
    pack_boreal_fst_archive,
    write_multifile_fst_leads,
)
from .duck import (  # noqa: E402
    MarrayDuckArray,
    open_fst_xarray,
    wrap_dataset_as_xarray,
)
from .fst_open import (  # noqa: E402
    open_fst_paths_catalog,
    open_fst_paths_xarray,
)
from .max_compiled import (  # noqa: E402
    CompiledMaxSession,
    compiled_humidex_wind_cpu,
    has_gpu_accelerator,
    vorticity_stencil_cpu,
)
from .xarray_backend import (  # noqa: E402
    MarrayFstBackendEntrypoint,
    register_xarray_backend,
)

register_xarray_backend()

__all__ = [
    "CatalogVarSpec",
    "CompiledMaxSession",
    "DataArray",
    "Dataset",
    "FST_MASK_FILL",
    "FilePool",
    "MarrayDuckArray",
    "MarrayFstBackendEntrypoint",
    "PackedFSTArchive",
    "SinkSpec",
    "Variable",
    "compiled_humidex_wind_cpu",
    "compute_many",
    "from_sources",
    "get_builtin_decoder_addr",
    "get_last_decoded_planes",
    "get_mojo_lib",
    "has_gpu_accelerator",
    "open_fst_catalog",
    "open_fst_paths_catalog",
    "open_fst_paths_xarray",
    "open_fst_xarray",
    "pack_boreal_fst_archive",
    "register_xarray_backend",
    "rmse_f64",
    "stream_write_fst",
    "synth_boreal_raw_arenas",
    "to_aether_flat",
    "vertical_column_scan",
    "vorticity_stencil_cpu",
    "wrap_dataset_as_xarray",
    "write_multifile_fst_leads",
]
