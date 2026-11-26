"""M4 Phase A Xarray Duck-Array Backend (`CONTEXT.md` P0 & `docs/questions.md` Q13).

Wraps lazy `mxr.DataArray` nodes inside genuine `xarray.Dataset` / `xarray.DataArray`
objects via `MarrayDuckArray` (`__array_ufunc__`, `__array_function__`, `__getitem__`,
`__array__`). Preserves Boreal CF horizontal dimension names (`rlat`/`rlon`, `lat`/`lon`,
or `y`/`x`), 2D geographic `lat`/`lon` coordinates, `areacell`, `datetime64[ns]` `time`,
and `fst_*` metadata with ZERO Dask imports!
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

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

    @property
    def chunks(self) -> tuple[tuple[int, ...], ...]:
        """Record-plane chunk topology (`(1, ..., 1, nj, ni)`) for Xarray repr & ChunkManager."""
        sh = self.shape
        nd = len(sh)
        return tuple((1,) * n if i < nd - 2 else (n,) for i, n in enumerate(sh))

    def _squeezed_dims(self) -> tuple[str, ...]:
        act = set(self._active_dims)
        return tuple(d for d in self._inner.dims if d not in act)

    def compute(self, *, fill_value: float = np.nan, **kwargs: Any) -> np.ndarray:
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
        raw_key = getattr(key, "tuple", key)
        if not isinstance(raw_key, tuple):
            raw_key = (raw_key,)

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
    """Wrap an `mxr.Dataset` into a standard `xarray.Dataset` backed by `MarrayDuckArray`,
    preserving `rlat`/`rlon` or `y`/`x` spatial dims, 2D `lat`/`lon`, and `areacell`!
    """
    data_vars: dict[str, tuple[tuple[str, ...], MarrayDuckArray, dict[str, Any]]] = {}
    for vname in ds._var_nodes:
        mda = ds[vname]
        src_var = None
        op, a, *_ = ds._arena.nodes[mda._node_id]
        if op == mxr.BC_LOAD:
            src_var = ds._arena.sources[a]
        v_dims = src_var.dims if src_var is not None else mda.dims
        data_vars[vname] = (v_dims, MarrayDuckArray(mda, v_dims), dict(mda.attrs))

    coords: dict[str, Any] = {k: v for k, v in ds.coords.items()}
    for cname, (cdims, carr) in getattr(ds, "extra_coords_2d", {}).items():
        coords[cname] = (cdims, carr)
    if "areacell" in ds.cell_measures:
        ac_var = ds.cell_measures["areacell"]
        ac_dims = ac_var.dims if len(ac_var.dims) == 2 else ds._arena.domain_dims[-2:]
        coords["areacell"] = (ac_dims, ac_var.data)

    xr_ds = xr.Dataset(data_vars, coords=coords, attrs=dict(ds.attrs))
    if getattr(ds, "_close_fn", None) is not None:
        xr_ds.set_close(ds._close_fn)
    return xr_ds


def open_fst_xarray(
    source: mxr.PackedFSTArchive | str | Path | Sequence[str | Path],
    *,
    vars: Sequence[str] | None = None,
    decoder_fn_addr: int | None = None,
    split_vctype: bool = True,
    open_data_mapping: bool = True,
    cell_measures: bool = True,
    cmc_root: str | Path | None = None,
) -> list[xr.Dataset]:
    """M4 Phase A `open_fst` replacement (`CONTEXT.md` P0):
      - Accepts EITHER on-disk FST / `.mfst` file path(s) (`open_fst_xarray(paths)`)
        OR an in-memory `PackedFSTArchive`.
      - Returns `list[xr.Dataset]` backed by `MarrayDuckArray` + `RecordCatalog`
        with `O(headers)` open time, surgical per-plane decode on `.isel()`/`.sel()`,
        CF coords (`rlat`/`rlon`, 2D `lat`/`lon`, `areacell`, `datetime64` `time`),
        and ZERO Dask `HighLevelGraph` construction!
    """
    if isinstance(source, mxr.PackedFSTArchive):
        cats = mxr.open_fst_catalog(
            source, decoder_fn_addr=decoder_fn_addr, split_vctype=split_vctype
        )
        return [wrap_dataset_as_xarray(c) for c in cats]

    return mxr.open_fst_paths_xarray(
        source,
        vars=vars,
        decoder_fn_addr=decoder_fn_addr,
        split_vctype=split_vctype,
        open_data_mapping=open_data_mapping,
        cell_measures=cell_measures,
        cmc_root=cmc_root,
    )
