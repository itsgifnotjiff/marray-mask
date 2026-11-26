"""General Python Xarray + Dask Drop-In Replacement Backed by Mojo (`marray_xr`).

Zero domain or benchmark hardcoding:
  - Arbitrary named dimensions & monotonic ascending/descending coordinates
  - Arbitrary N-D strided variables + automatic stride-0 cross-rank broadcasting
  - Pluggable CF-conventions `scale_factor` / `add_offset` + heterogeneous masks
    (`fill_value` integer sentinels, IEEE `NaN`s, `numpy.ma` boolean masks,
    companion `uint8` quality flags)
  - Full lazy expression DAG with NumPy `__array_ufunc__` (`np.hypot`, `np.sqrt`,
    `np.exp`, `np.log`, `np.sin`, `np.cos`, `np.minimum`, `np.maximum`, `+`, `-`,
    `*`, `/`, `** 2`, `abs`, `.where(...)`, `.fillna(...)`)
  - Executes via Mojo's general 3-pass bytecode optimizer (CSE + algebraic FMA /
    CF-scale folding + linear-scan register coloring) & L1-cache morsel VM.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr


# Must match `src/marray/morsel.mojo` & `src/marray/vm.mojo`:
DTYPE_F32 = 0
DTYPE_U16 = 2

MSRC_NONE = 0
MSRC_SENTINEL_INT = 1
MSRC_NAN = 2
MSRC_BYTE_GT = 3
MSRC_BYTE_NE = 4

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


def _get_lib() -> ctypes.CDLL:
    global _LIB_CACHE
    if _LIB_CACHE is not None:
        return _LIB_CACHE

    so_path = _resolve_so_path()
    lib = ctypes.CDLL(str(so_path))
    i64_p = ctypes.POINTER(ctypes.c_int64)
    i32_p = ctypes.POINTER(ctypes.c_int32)
    f32_p = ctypes.POINTER(ctypes.c_float)
    f64_p = ctypes.POINTER(ctypes.c_double)
    u8_p = ctypes.POINTER(ctypes.c_uint8)

    lib.marray_init_runtime.argtypes = []
    lib.marray_init_runtime.restype = ctypes.c_int32
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
        f64_p,
        f32_p,
        u8_p,
    ]
    lib.marray_exec_general_plan.restype = ctypes.c_int32
    lib.marray_init_runtime()
    _LIB_CACHE = lib
    return lib


class Variable:
    """General N-D array variable with optional CF scale/offset and mask encoding."""

    def __init__(
        self,
        dims: Sequence[str],
        data: np.ndarray,
        *,
        scale: float = 1.0,
        offset: float = 0.0,
        fill_value: int | float | None = None,
        qc_flags: np.ndarray | None = None,
        qc_max_valid: int = 1,
        mask: np.ndarray | None = None,
        nan_masked: bool = False,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        self.dims = tuple(dims)
        self.attrs = dict(attrs or {})

        if np.ma.isMaskedArray(data):
            raw_m = np.ma.getmask(data)
            if raw_m is not np.ma.nomask and mask is None:
                mask = np.asarray(np.ma.getmaskarray(data), dtype=np.uint8)
            data = np.ma.getdata(data)

        self.data = data
        self.scale = float(scale)
        self.offset = float(offset)
        self.aux_buf: np.ndarray | None = None

        if data.dtype == np.uint16:
            self.dtype_code = DTYPE_U16
        elif data.dtype == np.float32:
            self.dtype_code = DTYPE_F32
        else:
            self.data = np.asarray(data, dtype=np.float32)
            self.dtype_code = DTYPE_F32

        if qc_flags is not None:
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
    """Lazy predicate produced by comparisons like `da >= 0.35` for `da.where(...)`."""

    target: "DataArray"
    mask_opcode: int  # opcode marking which lanes to MASK OUT (invert of keep cond!)
    threshold: float


class _ExprArena:
    """Shared CSE-deduplicated expression graph for a dataset domain."""

    def __init__(self, domain_dims: tuple[str, ...], full_shape: tuple[int, ...]) -> None:
        self.domain_dims = domain_dims
        self.full_shape = full_shape
        self.sources: list[Variable] = []
        self.src_index: dict[int, int] = {}
        self.nodes: list[tuple[int, int, int, int, float, float]] = []
        self.cse_map: dict[tuple[int, int, int, int, float, float], int] = {}

    def register_source(self, var: Variable) -> int:
        vid = id(var)
        if vid in self.src_index:
            return self.src_index[vid]
        sid = len(self.sources)
        self.sources.append(var)
        self.src_index[vid] = sid
        return self.emit(BC_LOAD, sid, -1, -1, 0.0, 0.0)

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

    def clone_subdag_for_root(self, root_nid: int) -> tuple[list[Variable], np.ndarray, np.ndarray]:
        """Extract compact reachable topological node arrays ending at `root_nid`."""
        needed = [False] * (root_nid + 1)
        needed[root_nid] = True
        for i in range(root_nid, -1, -1):
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
        used_src_map: dict[int, int] = {}
        compact: list[tuple[int, int, int, int, float, float]] = []

        for i in range(root_nid + 1):
            if not needed[i]:
                continue
            op, a, b, c, p0, p1 = self.nodes[i]
            if op == BC_LOAD:
                if a not in used_src_map:
                    used_src_map[a] = len(used_sources)
                    used_sources.append(self.sources[a])
                na = used_src_map[a]
            else:
                na = remap[a] if a >= 0 else -1
            nb = remap[b] if b >= 0 else -1
            nc = remap[c] if c >= 0 else -1
            remap[i] = len(compact)
            compact.append((op, na, nb, nc, p0, p1))

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
        return used_sources, i32_buf, f32_buf


def _search_coord_slice(coord: np.ndarray, sl: slice) -> tuple[int, int]:
    n = len(coord)
    if n == 0:
        return 0, 0
    if sl.start is None and sl.stop is None:
        return 0, n
    asc = n < 2 or coord[-1] >= coord[0]
    a = sl.start
    b = sl.stop
    if asc:
        lo = float(a) if a is not None else -np.inf
        hi = float(b) if b is not None else np.inf
        if lo > hi:
            lo, hi = hi, lo
        i0 = int(np.searchsorted(coord, lo, side="left"))
        i1 = int(np.searchsorted(coord, hi, side="right"))
        return i0, i1
    else:
        # Descending axis (e.g. atmospheric pressure hPa: 1015 -> 50)
        neg_c = -coord
        v0 = float(a) if a is not None else np.inf
        v1 = float(b) if b is not None else -np.inf
        hi = max(v0, v1)
        lo = min(v0, v1)
        i0 = int(np.searchsorted(neg_c, -hi, side="left"))
        i1 = int(np.searchsorted(neg_c, -lo, side="right"))
        return i0, i1


class DataArray:
    """Lazy labeled masked array supporting Xarray/NumPy ufuncs, `.sel()`, `.where()`, and reductions."""

    __array_priority__ = 1000

    def __init__(
        self,
        arena: _ExprArena,
        node_id: int,
        coords: dict[str, np.ndarray],
        slices: tuple[tuple[int, int], ...],
        name: str = "expr",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self._arena = arena
        self._node_id = node_id
        self.coords = coords
        self._slices = slices
        self.name = name
        self.attrs = dict(attrs or {})

    @property
    def dims(self) -> tuple[str, ...]:
        return self._arena.domain_dims

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(max(0, b - a) for a, b in self._slices)

    def _wrap(self, nid: int, name: str = "expr") -> "DataArray":
        return DataArray(self._arena, nid, self.coords, self._slices, name=name)

    def _coerce_other(self, other: Any) -> int:
        if isinstance(other, DataArray):
            if other._arena is not self._arena:
                raise ValueError("DataArrays must belong to the same Dataset arena")
            return other._node_id
        return self._arena.emit(BC_CONST, p0=float(other))

    # --- Standard Arithmetic & Comparisons ---
    def __add__(self, other: Any) -> "DataArray":
        return self._wrap(self._arena.emit(BC_ADD, self._node_id, self._coerce_other(other)))

    def __radd__(self, other: Any) -> "DataArray":
        return self._wrap(self._arena.emit(BC_ADD, self._coerce_other(other), self._node_id))

    def __sub__(self, other: Any) -> "DataArray":
        return self._wrap(self._arena.emit(BC_SUB, self._node_id, self._coerce_other(other)))

    def __rsub__(self, other: Any) -> "DataArray":
        return self._wrap(self._arena.emit(BC_SUB, self._coerce_other(other), self._node_id))

    def __mul__(self, other: Any) -> "DataArray":
        return self._wrap(self._arena.emit(BC_MUL, self._node_id, self._coerce_other(other)))

    def __rmul__(self, other: Any) -> "DataArray":
        return self._wrap(self._arena.emit(BC_MUL, self._coerce_other(other), self._node_id))

    def __truediv__(self, other: Any) -> "DataArray":
        return self._wrap(self._arena.emit(BC_DIV, self._node_id, self._coerce_other(other)))

    def __rtruediv__(self, other: Any) -> "DataArray":
        return self._wrap(self._arena.emit(BC_DIV, self._coerce_other(other), self._node_id))

    def __neg__(self) -> "DataArray":
        return self._wrap(self._arena.emit(BC_NEG, self._node_id))

    def __abs__(self) -> "DataArray":
        return self._wrap(self._arena.emit(BC_ABS, self._node_id))

    def __pow__(self, power: float) -> "DataArray":
        if power == 2 or power == 2.0:
            return self._wrap(self._arena.emit(BC_MUL, self._node_id, self._node_id))
        if power == 0.5:
            return self._wrap(self._arena.emit(BC_SQRT, self._node_id))
        # General `x**p = exp(p * log(x))`
        lg = self._arena.emit(BC_LOG, self._node_id)
        sc = self._arena.emit(BC_AFFINE, lg, p0=float(power), p1=0.0)
        return self._wrap(self._arena.emit(BC_EXP, sc))

    # Lazy comparison predicates for `da.where(da >= threshold)`:
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
        if method != "__call__" or ufunc not in _UFUNC_MAP:
            return NotImplemented
        opcode = _UFUNC_MAP[ufunc]
        if len(inputs) == 1:
            return self._wrap(self._arena.emit(opcode, self._coerce_other(inputs[0])))
        if len(inputs) == 2:
            return self._wrap(
                self._arena.emit(
                    opcode, self._coerce_other(inputs[0]), self._coerce_other(inputs[1])
                )
            )
        return NotImplemented

    # --- Coordinate Selection (`.sel()` / `.isel()`) ---
    def sel(self, **indexers: Any) -> "DataArray":
        new_slices = list(self._slices)
        for dim_name, sel_val in indexers.items():
            ax = self.dims.index(dim_name)
            c_arr = self.coords[dim_name]
            cur_a, cur_b = new_slices[ax]
            if isinstance(sel_val, slice):
                s0, s1 = _search_coord_slice(c_arr, sel_val)
            else:
                idx = int(np.argmin(np.abs(c_arr - float(sel_val))))
                s0, s1 = idx, idx + 1
            new_slices[ax] = (max(cur_a, s0), min(cur_b, s1))
        return DataArray(self._arena, self._node_id, self.coords, tuple(new_slices), self.name, self.attrs)

    def isel(self, **indexers: Any) -> "DataArray":
        new_slices = list(self._slices)
        for dim_name, sel_val in indexers.items():
            ax = self.dims.index(dim_name)
            cur_a, cur_b = new_slices[ax]
            span = cur_b - cur_a
            if isinstance(sel_val, slice):
                start, stop, _ = sel_val.indices(span)
                new_slices[ax] = (cur_a + start, cur_a + stop)
            else:
                idx = int(sel_val)
                new_slices[ax] = (cur_a + idx, cur_a + idx + 1)
        return DataArray(self._arena, self._node_id, self.coords, tuple(new_slices), self.name, self.attrs)

    # --- Execution & Reductions ---
    def _execute(
        self,
        reduce_dims: Sequence[str] | None,
        agg_mode: int,
        need_global_std: bool = False,
    ) -> tuple[np.ndarray, xr.DataArray | None]:
        lib = _get_lib()
        used_sources, dag_i32, dag_f32 = self._arena.clone_subdag_for_root(self._node_id)
        ndim = len(self.dims)
        ns = len(used_sources)

        src_i64 = np.zeros((ns, 11), dtype=np.int64)
        src_f32 = np.zeros((ns, 2), dtype=np.float32)

        for i, var in enumerate(used_sources):
            item_sz = var.data.dtype.itemsize
            src_i64[i, 0] = var.data.ctypes.data
            src_i64[i, 1] = var.aux_buf.ctypes.data if var.aux_buf is not None else 0
            src_i64[i, 2] = var.dtype_code
            src_i64[i, 3] = var.mask_kind
            src_i64[i, 4] = var.mask_param
            src_f32[i, 0] = var.scale
            src_f32[i, 1] = var.offset

            # Map variable's physical element strides onto dataset domain dims
            # (missing dims get stride 0 -> automatic zero-copy broadcasting!)
            for d_idx, d_name in enumerate(self.dims):
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

        full_ext = np.ones(6, dtype=np.int64)
        sl_starts = np.zeros(6, dtype=np.int64)
        sl_stops = np.ones(6, dtype=np.int64)
        for d_idx in range(ndim):
            full_ext[d_idx] = self._arena.full_shape[d_idx]
            sl_starts[d_idx] = self._slices[d_idx][0]
            sl_stops[d_idx] = self._slices[d_idx][1]

        out_stats = np.zeros(9, dtype=np.float64)

        if reduce_dims is None and agg_mode >= 0:
            # Global scalar reduction only
            red_mask = -1
            out_arr = np.empty(1, dtype=np.float32)
            out_msk = np.empty(1, dtype=np.uint8)
            kept_dims: list[str] = []
            kept_shape: list[int] = []
        else:
            red_set = set(reduce_dims or ())
            red_mask = 0
            kept_dims = []
            kept_shape = []
            for d_idx, d_name in enumerate(self.dims):
                if d_name in red_set:
                    red_mask |= 1 << d_idx
                else:
                    kept_dims.append(d_name)
                    kept_shape.append(max(0, self._slices[d_idx][1] - self._slices[d_idx][0]))
            out_arr = np.empty(tuple(kept_shape) if kept_shape else (1,), dtype=np.float32, order="F")
            out_msk = np.empty(tuple(kept_shape) if kept_shape else (1,), dtype=np.uint8, order="F")

        i64_p = ctypes.POINTER(ctypes.c_int64)
        i32_p = ctypes.POINTER(ctypes.c_int32)
        f32_p = ctypes.POINTER(ctypes.c_float)
        f64_p = ctypes.POINTER(ctypes.c_double)
        u8_p = ctypes.POINTER(ctypes.c_uint8)

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
            out_stats.ctypes.data_as(f64_p),
            out_arr.ctypes.data_as(f32_p),
            out_msk.ctypes.data_as(u8_p),
        )

        if reduce_dims is None and agg_mode >= 0:
            return out_stats, None

        # Build labeled xarray.DataArray with NaN for masked cells (plus `.ma` attribute)
        kept_coords = {}
        for d_name in kept_dims:
            ax = self.dims.index(d_name)
            a, b = self._slices[ax]
            kept_coords[d_name] = self.coords[d_name][a:b]

        filled = out_arr.copy(order="F")
        filled[out_msk != 0] = np.nan
        xr_da = xr.DataArray(filled, dims=tuple(kept_dims), coords=kept_coords, name=self.name)
        return out_stats, xr_da

    def stats(self) -> dict[str, float | int]:
        st, _ = self._execute(reduce_dims=None, agg_mode=AGG_MEAN, need_global_std=True)
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

    def _reduce_dispatch(
        self, dim: str | Sequence[str] | None, agg_mode: int, stat_idx: int
    ) -> float | int | xr.DataArray:
        if dim is None:
            st, _ = self._execute(
                reduce_dims=None,
                agg_mode=agg_mode,
                need_global_std=(agg_mode == AGG_STD),
            )
            return int(st[stat_idx]) if agg_mode == AGG_COUNT else float(st[stat_idx])
        dims_seq = (dim,) if isinstance(dim, str) else tuple(dim)
        _, xr_da = self._execute(reduce_dims=dims_seq, agg_mode=agg_mode)
        assert xr_da is not None
        return xr_da

    def mean(self, dim: str | Sequence[str] | None = None) -> float | xr.DataArray:
        return self._reduce_dispatch(dim, AGG_MEAN, 0)  # type: ignore[return-value]

    def std(self, dim: str | Sequence[str] | None = None) -> float | xr.DataArray:
        return self._reduce_dispatch(dim, AGG_STD, 1)  # type: ignore[return-value]

    def min(self, dim: str | Sequence[str] | None = None) -> float | xr.DataArray:
        return self._reduce_dispatch(dim, AGG_MIN, 2)  # type: ignore[return-value]

    def max(self, dim: str | Sequence[str] | None = None) -> float | xr.DataArray:
        return self._reduce_dispatch(dim, AGG_MAX, 3)  # type: ignore[return-value]

    def sum(self, dim: str | Sequence[str] | None = None) -> float | xr.DataArray:
        return self._reduce_dispatch(dim, AGG_SUM, 4)  # type: ignore[return-value]

    def count(self, dim: str | Sequence[str] | None = None) -> int | xr.DataArray:
        return self._reduce_dispatch(dim, AGG_COUNT, 5)  # type: ignore[return-value]

    def compute(self) -> xr.DataArray:
        _, xr_da = self._execute(reduce_dims=(), agg_mode=-1)
        assert xr_da is not None
        return xr_da

    def stats_and_reduce(
        self, dim: str | Sequence[str], agg: str = "mean"
    ) -> tuple[dict[str, float | int], xr.DataArray]:
        """Single-pass execution returning both global stats and an axis-reduced map!"""
        agg_lut = {
            "mean": AGG_MEAN,
            "sum": AGG_SUM,
            "min": AGG_MIN,
            "max": AGG_MAX,
            "std": AGG_STD,
            "count": AGG_COUNT,
        }
        dims_seq = (dim,) if isinstance(dim, str) else tuple(dim)
        st, xr_da = self._execute(
            reduce_dims=dims_seq,
            agg_mode=agg_lut[agg],
            need_global_std=False,
        )
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


class Dataset:
    """General labeled N-D Dataset replacing Xarray + Dask with Mojo execution."""

    def __init__(
        self,
        data_vars: Mapping[str, Variable],
        coords: Mapping[str, np.ndarray],
        *,
        domain_dims: Sequence[str] | None = None,
        attrs: Mapping[str, Any] | None = None,
        _arena: _ExprArena | None = None,
        _var_nodes: dict[str, int] | None = None,
        _slices: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        self.coords = {k: np.asarray(v, dtype=np.float64) for k, v in coords.items()}
        self.attrs = dict(attrs or {})

        if _arena is not None and _var_nodes is not None and _slices is not None:
            self._arena = _arena
            self._var_nodes = _var_nodes
            self._slices = _slices
            return

        if domain_dims is None:
            # Infer canonical domain dims in order of appearance, largest-rank var first
            ordered: list[str] = []
            for v in sorted(data_vars.values(), key=lambda x: -len(x.dims)):
                for d in v.dims:
                    if d not in ordered:
                        ordered.append(d)
            self._dims = tuple(ordered)
        else:
            self._dims = tuple(domain_dims)

        full_shape = tuple(len(self.coords[d]) for d in self._dims)
        self._arena = _ExprArena(self._dims, full_shape)
        self._var_nodes = {
            name: self._arena.register_source(var) for name, var in data_vars.items()
        }
        self._slices = tuple((0, n) for n in full_shape)

    @property
    def dims(self) -> dict[str, int]:
        return {
            d: max(0, self._slices[i][1] - self._slices[i][0])
            for i, d in enumerate(self._arena.domain_dims)
        }

    def __getitem__(self, name: str) -> DataArray:
        return DataArray(
            self._arena, self._var_nodes[name], self.coords, self._slices, name=name
        )

    def sel(self, **indexers: Any) -> "Dataset":
        probe = DataArray(self._arena, 0, self.coords, self._slices)
        sliced = probe.sel(**indexers)
        return Dataset(
            {},
            self.coords,
            attrs=self.attrs,
            _arena=self._arena,
            _var_nodes=self._var_nodes,
            _slices=sliced._slices,
        )

    def isel(self, **indexers: Any) -> "Dataset":
        probe = DataArray(self._arena, 0, self.coords, self._slices)
        sliced = probe.isel(**indexers)
        return Dataset(
            {},
            self.coords,
            attrs=self.attrs,
            _arena=self._arena,
            _var_nodes=self._var_nodes,
            _slices=sliced._slices,
        )
