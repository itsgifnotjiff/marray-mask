"""Xarray `BackendEntrypoint` (`engine="marray_fst"`, `CONTEXT.md` P0 #5).

Allows unmodified Xarray callers to run:
    `ds = xr.open_dataset(path_or_paths, engine="marray_fst")`

Also auto-registers `"marray_fst"` into `xarray.backends.plugins` when `marray_xr`
is imported, so `engine="marray_fst"` works immediately on HPC nodes (`ppp8`) where
`python/` is added via `PYTHONPATH` without requiring `pip install -e .`!
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import xarray as xr
from xarray.backends import BackendArray, BackendEntrypoint
from xarray.backends import plugins as xr_plugins
from xarray.core import indexing as xr_indexing
from xarray.namedarray import parallelcompat as xr_parallel

import marray_xr as mxr
from marray_xr.duck import MarrayDuckArray


class MarrayBackendArray(BackendArray):
    """Xarray `BackendArray` adapter around `MarrayDuckArray` for `xr.open_dataset(..., engine='marray_fst')`.

    - With `chunks=None`: supports Xarray `IndexingSupport.BASIC` lazy indexing (decodes ONLY
      touched planes on `.isel(...)` + `.values`).
    - With `chunks={}` (`chunked_array_type='marray'`, or default when Dask is absent):
      `MarrayChunkManager.from_array` unwraps `._duck` directly into `MarrayDuckArray` in O(1)!
    """

    def __init__(self, duck: MarrayDuckArray) -> None:
        self._duck = duck
        self.shape = duck.shape
        self.dtype = duck.dtype

    def __getitem__(self, key: xr_indexing.ExplicitIndexer) -> np.ndarray:
        return xr_indexing.explicit_indexing_adapter(
            key,
            self.shape,
            xr_indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple[Any, ...]) -> np.ndarray:
        sliced = self._duck[key]
        return sliced.compute() if isinstance(sliced, MarrayDuckArray) else np.asarray(sliced)


class MarrayChunkManager(xr_parallel.ChunkManagerEntrypoint[MarrayDuckArray]):
    """Zero-Dask Xarray `ChunkManagerEntrypoint` for `MarrayDuckArray`.

    Hooks `da.compute()`, `da.load()`, `ds.compute()` (single-pass multi-sink `compute_many`!),
    `da.chunks`, and lazy `repr(ds)` (0 planes decoded on `print(ds)`) without Dask!
    """

    array_cls: type[MarrayDuckArray]
    available: bool = True

    def __init__(self) -> None:
        self.array_cls = MarrayDuckArray

    def is_chunked_array(self, data: Any) -> bool:
        return isinstance(data, MarrayDuckArray)

    def chunks(self, data: MarrayDuckArray) -> tuple[tuple[int, ...], ...]:
        return data.chunks

    def normalize_chunks(
        self,
        chunks: Any,
        shape: tuple[int, ...] | None = None,
        limit: int | None = None,
        dtype: Any = None,
        previous_chunks: tuple[tuple[int, ...], ...] | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        if shape is None:
            return ()
        if (
            isinstance(chunks, tuple)
            and len(chunks) == len(shape)
            and all(isinstance(c, tuple) for c in chunks)
        ):
            return chunks
        out: list[tuple[int, ...]] = []
        for i, n in enumerate(shape):
            csz = n if i >= len(shape) - 2 else 1
            if isinstance(chunks, int) and chunks > 0:
                csz = min(n, chunks)
            elif (
                isinstance(chunks, tuple)
                and i < len(chunks)
                and isinstance(chunks[i], int)
                and chunks[i] > 0
            ):
                csz = min(n, chunks[i])
            if n <= 0:
                out.append((0,))
            elif csz <= 0 or csz >= n:
                out.append((n,))
            else:
                q, r = divmod(n, csz)
                out.append((csz,) * q + ((r,) if r else ()))
        return tuple(out)

    def from_array(self, data: Any, chunks: Any, **kwargs: Any) -> MarrayDuckArray:
        cur = data
        while True:
            if isinstance(cur, MarrayDuckArray):
                return cur
            if hasattr(cur, "_duck"):
                return cur._duck
            if hasattr(cur, "array"):
                cur = cur.array
            elif hasattr(cur, "_array"):
                cur = cur._array
            else:
                break
        arr = np.asarray(data, dtype=np.float32)
        dims = tuple(f"dim_{i}" for i in range(arr.ndim))
        coords = {d: np.arange(s, dtype=np.float64) for d, s in zip(dims, arr.shape)}
        ds = mxr.Dataset({"v": mxr.Variable(dims, arr)}, coords, domain_dims=dims)
        return MarrayDuckArray(ds["v"], dims)

    def compute(self, *data: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Materialize 1 or more `MarrayDuckArray`s—fusing multi-variable `ds.compute()` in ONE Mojo pass!"""
        duck_indices = [i for i, d in enumerate(data) if isinstance(d, MarrayDuckArray)]
        if not duck_indices:
            return tuple(np.asarray(d) if hasattr(d, "__array__") else d for d in data)
        if len(duck_indices) == 1:
            idx = duck_indices[0]
            out = list(data)
            out[idx] = data[idx].compute()
            return tuple(out)

        # Group by compatible `(exec_dims, shape)` so `ds.compute()` runs in ONE single-pass Mojo scan!
        out = list(data)
        groups: dict[tuple[tuple[str, ...], tuple[int, ...]], list[int]] = {}
        for idx in duck_indices:
            d: MarrayDuckArray = data[idx]
            key = (d._inner.dims, d._inner.shape)
            groups.setdefault(key, []).append(idx)

        total_decs = 0
        for grp in groups.values():
            if len(grp) == 1:
                d0: MarrayDuckArray = data[grp[0]]
                had_cat = d0._inner.has_catalog_source
                out[grp[0]] = d0.compute()
                if had_cat:
                    total_decs += mxr.get_last_decoded_planes()
            else:
                had_cat = any(data[idx]._inner.has_catalog_source for idx in grp)
                sinks = [
                    mxr.SinkSpec(
                        data[idx]._inner,
                        (),
                        -1,
                        fill_value=np.nan,
                        squeeze_dims=data[idx]._squeezed_dims(),
                    )
                    for idx in grp
                ]
                xr_outs = mxr.compute_many(*sinks)
                if had_cat:
                    total_decs += mxr.get_last_decoded_planes()
                for idx, xr_da in zip(grp, xr_outs):
                    out[idx] = xr_da.values
        if total_decs > 0:
            mxr._LAST_DECODED_PLANES = total_decs
        return tuple(out)

    def persist(self, *data: Any, **kwargs: Any) -> tuple[Any, ...]:
        return self.compute(*data, **kwargs)

    def apply_gufunc(
        self,
        func: Callable[..., Any],
        signature: str,
        *args: Any,
        axes: Sequence[tuple[int, ...]] | None = None,
        keepdims: bool = False,
        output_dtypes: Any = None,
        vectorize: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        try:
            res = func(*args, **kwargs)
            if isinstance(res, MarrayDuckArray):
                return res
        except Exception:
            pass
        mat_args = [a.compute() if isinstance(a, MarrayDuckArray) else a for a in args]
        fn = np.vectorize(func, signature=signature) if vectorize else func
        return fn(*mat_args, **kwargs)


class MarrayFstBackendEntrypoint(BackendEntrypoint):
    """Xarray backend returning `MarrayDuckArray`-backed lazy FST `Dataset`s."""

    description = "Mojo marray-mask lazy FST / .mfst backend (zero Dask HLG)"
    url = "https://github.com/itsgifnotjiff/marray-mask"
    supports_groups = True

    open_dataset_parameters = (
        "filename_or_obj",
        "drop_variables",
        "cache",
        "chunks",
        "mask_and_scale",
        "decode_times",
        "decode_coords",
        "vars",
        "group_index",
        "vctype",
        "split_vctype",
        "open_data_mapping",
        "cell_measures",
        "cmc_root",
    )

    def guess_can_open(self, filename_or_obj: Any) -> bool:
        if not isinstance(filename_or_obj, (str, os.PathLike)):
            return False
        ext = Path(filename_or_obj).suffix.lower()
        return ext in (".mfst", ".fst", ".std")

    def open_dataset(  # type: ignore[override]
        self,
        filename_or_obj: str | os.PathLike[Any] | Sequence[str | os.PathLike[Any]],
        *,
        drop_variables: str | Iterable[str] | None = None,
        cache: bool | None = None,
        chunks: Any = None,
        mask_and_scale: bool | None = True,
        decode_times: bool | None = True,
        decode_coords: bool | str | None = True,
        vars: Sequence[str] | None = None,
        group_index: int = 0,
        vctype: str | None = None,
        split_vctype: bool = True,
        open_data_mapping: bool = True,
        cell_measures: bool = True,
        cmc_root: str | os.PathLike[Any] | None = None,
    ) -> xr.Dataset:
        ds_list = mxr.open_fst_xarray(
            filename_or_obj,  # type: ignore[arg-type]
            vars=vars,
            split_vctype=split_vctype,
            open_data_mapping=open_data_mapping,
            cell_measures=cell_measures,
            cmc_root=cmc_root,
        )
        if not ds_list:
            return xr.Dataset()

        chosen: xr.Dataset | None = None
        if vctype is not None:
            for cand in ds_list:
                if cand.attrs.get("fst_vctype") == vctype:
                    chosen = cand
                    break
            if chosen is None:
                raise KeyError(f"No Dataset group with fst_vctype={vctype!r}")
        else:
            chosen = ds_list[min(int(group_index), len(ds_list) - 1)]

        if drop_variables:
            drops = [drop_variables] if isinstance(drop_variables, str) else list(drop_variables)
            keep_drops = [d for d in drops if d in chosen]
            if keep_drops:
                chosen = chosen.drop_vars(keep_drops)

        # Wrap data_vars in `MarrayBackendArray` for Xarray's `_protect_dataset_variables_inplace`:
        # - `chunks=None`: lazy `ExplicitlyIndexed` plane slicing
        # - `chunks={}`: unwrapped back to `MarrayDuckArray` in O(1) by `MarrayChunkManager.from_array`
        for vname, var in chosen.data_vars.items():
            if isinstance(var.variable._data, MarrayDuckArray):
                duck = var.variable._data
                var.variable._data = xr_indexing.LazilyIndexedArray(MarrayBackendArray(duck))
                nd = duck.ndim
                var.encoding["preferred_chunks"] = {
                    d: (1 if i < nd - 2 else s) for i, (d, s) in enumerate(zip(var.dims, duck.shape))
                }
        return chosen


def register_xarray_backend() -> None:
    """Register `engine='marray_fst'` AND `MarrayChunkManager` dynamically in Xarray's plugin caches."""
    try:
        engines = xr_plugins.list_engines()
        if "marray_fst" not in engines:
            engines["marray_fst"] = MarrayFstBackendEntrypoint()  # type: ignore[index]
    except Exception:
        pass
    try:
        cms = xr_parallel.list_chunkmanagers()
        if "marray" not in cms:
            cms["marray"] = MarrayChunkManager()
    except Exception:
        pass
