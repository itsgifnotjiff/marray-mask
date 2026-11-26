"""M4 Phase A Xarray Duck-Array Backend (`docs/questions.md` Q13, `10-mojo-migration-targets.md`).

Wraps lazy `mxr.DataArray` nodes inside genuine `xarray.Dataset` / `xarray.DataArray`
objects via `MarrayDuckArray` (`__array_ufunc__`, `__array_function__`, `__getitem__`,
`__array__`). Existing `boreal-postproc`, `boreal-stat`, `polaris`, and `aether`
plugins can slice (`.isel()`, `.sel(method='nearest')`), compose NumPy ufuncs
(`np.hypot`, `np.exp`, arithmetic), reduce (`.mean(dim=...)`), or peel (`.values` /
`np.asarray(...)`) WITHOUT constructing a Dask `HighLevelGraph`!
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

import marray_xr as mxr


class MarrayDuckArray:
    """Lazy Mojo-backed duck array living inside `xarray.Variable._data`."""

    __array_priority__ = 1000

    def __init__(
        self,
        inner: mxr.DataArray,
        active_dims: tuple[str, ...] | None = None,
    ) -> None:
        self._inner = inner
        self._active_dims = active_dims if active_dims is not None else inner.dims

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(np.float32)

    @property
    def shape(self) -> tuple[int, ...]:
        s_map = dict(zip(self._inner.dims, self._inner.shape))
        return tuple(s_map[d] for d in self._active_dims)

    @property
    def ndim(self) -> int:
        return len(self._active_dims)

    @property
    def size(self) -> int:
        sz = 1
        for n in self.shape:
            sz *= n
        return sz

    def _squeezed_dims(self) -> tuple[str, ...]:
        act = set(self._active_dims)
        return tuple(d for d in self._inner.dims if d not in act)

    def compute(self, *, fill_value: float = np.nan) -> np.ndarray:
        xr_da = self._inner.compute(
            fill_value=fill_value, squeeze=self._squeezed_dims()
        )
        return xr_da.values

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        arr = self.compute(fill_value=np.nan)
        if dtype is not None and arr.dtype != dtype:
            return arr.astype(dtype)
        return arr

    def __getitem__(self, key: Any) -> "MarrayDuckArray":
        # Xarray passes `ExplicitlyIndexed` wrappers (`BasicIndexer((slice, int, ...))`)!
        raw_key = getattr(key, "tuple", key)
        if not isinstance(raw_key, tuple):
            raw_key = (raw_key,)

        # Expand `Ellipsis` if present
        if Ellipsis in raw_key:
            e_pos = raw_key.index(Ellipsis)
            n_fill = self.ndim - (len(raw_key) - 1)
            raw_key = raw_key[:e_pos] + (slice(None),) * max(0, n_fill) + raw_key[e_pos + 1 :]

        isel_kw: dict[str, Any] = {}
        new_active: list[str] = []
        ax_i = 0
        for item in raw_key:
            if item is None:
                continue
            d_name = self._active_dims[ax_i]
            ax_i += 1
            if isinstance(item, (int, np.integer)):
                isel_kw[d_name] = int(item)
            elif isinstance(item, slice):
                if item != slice(None):
                    isel_kw[d_name] = item
                new_active.append(d_name)
            else:
                # Fallback for arbitrary 1D fancy indexers
                arr = self.compute()
                return arr[raw_key]  # type: ignore[return-value]

        while ax_i < len(self._active_dims):
            new_active.append(self._active_dims[ax_i])
            ax_i += 1

        sliced_inner = self._inner.isel(**isel_kw) if isel_kw else self._inner
        return MarrayDuckArray(sliced_inner, tuple(new_active))

    def _unwrap_arg(self, arg: Any) -> Any:
        return arg._inner if isinstance(arg, MarrayDuckArray) else arg

    def __add__(self, other: Any) -> "MarrayDuckArray":
        return MarrayDuckArray(self._inner + self._unwrap_arg(other), self._active_dims)

    def __radd__(self, other: Any) -> "MarrayDuckArray":
        return MarrayDuckArray(self._unwrap_arg(other) + self._inner, self._active_dims)

    def __sub__(self, other: Any) -> "MarrayDuckArray":
        return MarrayDuckArray(self._inner - self._unwrap_arg(other), self._active_dims)

    def __rsub__(self, other: Any) -> "MarrayDuckArray":
        return MarrayDuckArray(self._unwrap_arg(other) - self._inner, self._active_dims)

    def __mul__(self, other: Any) -> "MarrayDuckArray":
        return MarrayDuckArray(self._inner * self._unwrap_arg(other), self._active_dims)

    def __rmul__(self, other: Any) -> "MarrayDuckArray":
        return MarrayDuckArray(self._unwrap_arg(other) * self._inner, self._active_dims)

    def __truediv__(self, other: Any) -> "MarrayDuckArray":
        return MarrayDuckArray(self._inner / self._unwrap_arg(other), self._active_dims)

    def __rtruediv__(self, other: Any) -> "MarrayDuckArray":
        return MarrayDuckArray(self._unwrap_arg(other) / self._inner, self._active_dims)

    def __neg__(self) -> "MarrayDuckArray":
        return MarrayDuckArray(-self._inner, self._active_dims)

    def __pow__(self, power: float) -> "MarrayDuckArray":
        return MarrayDuckArray(self._inner ** power, self._active_dims)

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        unwrapped = [self._unwrap_arg(x) for x in inputs]
        active = self._active_dims
        for x in inputs:
            if isinstance(x, MarrayDuckArray) and len(x._active_dims) > len(active):
                active = x._active_dims
        res = self._inner.__array_ufunc__(ufunc, method, *unwrapped, **kwargs)
        if res is NotImplemented:
            # Graceful fallback to materialized NumPy ufunc for esoteric ufuncs
            np_in = [np.asarray(x) if isinstance(x, MarrayDuckArray) else x for x in inputs]
            return getattr(ufunc, method)(*np_in, **kwargs)
        return MarrayDuckArray(res, active)

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        red_map = {
            np.mean: "mean",
            np.nanmean: "mean",
            np.sum: "sum",
            np.nansum: "sum",
            np.std: "std",
            np.nanstd: "std",
            np.min: "min",
            np.nanmin: "min",
            np.max: "max",
            np.nanmax: "max",
        }
        if func in red_map and args and isinstance(args[0], MarrayDuckArray):
            target: MarrayDuckArray = args[0]
            axis = kwargs.get("axis", args[1] if len(args) > 1 else None)
            mname = red_map[func]
            if axis is None:
                val = getattr(target._inner, mname)()
                return np.asarray(val, dtype=np.float32)
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            red_dims = tuple(target._active_dims[a] for a in axes) + target._squeezed_dims()
            xr_out = getattr(target._inner, mname)(dim=red_dims)
            return xr_out.values

        np_args = [np.asarray(a) if isinstance(a, MarrayDuckArray) else a for a in args]
        return func(*np_args, **kwargs)


def wrap_dataset_as_xarray(ds: mxr.Dataset) -> xr.Dataset:
    """Wrap an `mxr.Dataset` into a standard `xarray.Dataset` backed by `MarrayDuckArray`."""
    data_vars: dict[str, tuple[tuple[str, ...], MarrayDuckArray, dict[str, Any]]] = {}
    for vname in ds._var_nodes:
        mda = ds[vname]
        # Respect variable's native rank (e.g. 3D surface `(time, y, x)` inside 4D dataset!)
        src_var = None
        op, a, *_ = ds._arena.nodes[mda._node_id]
        if op == mxr.BC_LOAD:
            src_var = ds._arena.sources[a]
        v_dims = src_var.dims if src_var is not None else mda.dims
        data_vars[vname] = (v_dims, MarrayDuckArray(mda, v_dims), dict(mda.attrs))

    coords: dict[str, Any] = {k: v for k, v in ds.coords.items()}
    if "areacell" in ds.cell_measures:
        coords["areacell"] = (("y", "x"), ds.cell_measures["areacell"].data)

    return xr.Dataset(data_vars, coords=coords, attrs=dict(ds.attrs))


def open_fst_xarray(
    archive: mxr.PackedFSTArchive,
    *,
    decoder_fn_addr: int | None = None,
    split_vctype: bool = True,
) -> list[xr.Dataset]:
    """M4 Phase A `open_fst` replacement returning genuine `list[xr.Dataset]` objects
    backed by `MarrayDuckArray` + M1 `RecordCatalog` (`ChunkDecoder` whole-plane decode)!
    """
    cats = mxr.open_fst_catalog(
        archive, decoder_fn_addr=decoder_fn_addr, split_vctype=split_vctype
    )
    return [wrap_dataset_as_xarray(c) for c in cats]
