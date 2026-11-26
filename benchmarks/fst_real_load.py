"""Load live GDPS FST leads into marray-mask bench arenas / PackedFSTArchive.

Resolves boreal-io (`cmc-io`) via ``MARRAY_CMC_IO`` (required unless a relative
``../cmc-io`` / ``../boreal-io`` sibling exists). Opens ``pres`` files with
``boreal.io.open_fst``. Suite roots come from ``MARRAY_FST_PRES_ROOT`` or
``--root`` — never committed as absolute site paths.

See ``docs/boreal/12-synthetic-benchmark-data.md``.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

LEAD_RE = re.compile(r"^\d{10}_\d{3}$")

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _pres_root_from_env() -> Path | None:
    raw = os.environ.get("MARRAY_FST_PRES_ROOT", "").strip()
    return Path(raw).expanduser() if raw else None

DEFAULT_NOMVARS = ("TT", "UU", "VV", "HU")
CF_BY_NOMVAR = {
    "TT": "AirTemp",
    "UU": "WindU",
    "VV": "WindV",
    "HU": "SpecHum",
    "TM": "SeaSurfTemp",
}


@dataclass
class LoadTiming:
    scan_ms: float
    decode_ms: float
    stack_ms: float

    @property
    def total_ms(self) -> float:
        return self.scan_ms + self.decode_ms + self.stack_ms


def _cmc_io_candidates() -> list[Path]:
    """Env first, then relative sibling checkouts only (no absolute site/home roots)."""
    out: list[Path] = []
    env = os.environ.get("MARRAY_CMC_IO", "").strip()
    if env:
        out.append(Path(env).expanduser())
    for rel in ("../cmc-io", "../boreal-io"):
        out.append((_REPO_ROOT / rel).resolve())
    return out


def resolve_cmc_io_root() -> Path:
    for cand in _cmc_io_candidates():
        root = cand.resolve()
        if (root / "src" / "boreal" / "io").is_dir() or (root / "boreal" / "io").is_dir():
            return root
    raise FileNotFoundError(
        "boreal-io / cmc-io checkout not found. Set MARRAY_CMC_IO to the repo root "
        "(needs src/cmc_io_fst_data.so and boreal.io), or place a ../cmc-io sibling."
    )


def ensure_boreal_on_path(cmc_root: Path | None = None) -> Path:
    root = resolve_cmc_io_root() if cmc_root is None else Path(cmc_root)
    for p in (root, root / "src"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def list_pres_leads(root: Path | str | None = None) -> list[Path]:
    """List ``YYYYMMDDHH_HHH`` FST files under a GDPS ``prog/pres`` directory."""
    if root is not None:
        d = Path(root)
    else:
        env_root = _pres_root_from_env()
        if env_root is None:
            raise FileNotFoundError(
                "pres root not set. Pass root=... or export MARRAY_FST_PRES_ROOT "
                "to a GDPS-style prog/pres directory."
            )
        d = env_root
    if not d.is_dir():
        raise FileNotFoundError(f"pres root not found: {d}")
    leads = sorted(p for p in d.iterdir() if p.is_file() and LEAD_RE.match(p.name))
    if not leads:
        raise FileNotFoundError(f"no YYYYMMDDHH_HHH leads under {d}")
    return leads


def _find_by_nomvar(ds: Any, nomvar: str) -> Any:
    for name, var in ds.data_vars.items():
        if str(var.attrs.get("fst_nomvar", name)).strip() == nomvar:
            return var
    if nomvar in ds:
        return ds[nomvar]
    raise KeyError(nomvar)


def _as_c_yx(arr: np.ndarray, ni: int, nj: int) -> np.ndarray:
    """Normalize a horizontal plane to C-order ``(nj, ni)`` (Boreal y, x)."""
    a = np.asarray(arr)
    if a.shape == (nj, ni):
        return np.ascontiguousarray(a, dtype=np.float32)
    if a.shape == (ni, nj):
        return np.ascontiguousarray(a.T, dtype=np.float32)
    if a.size == ni * nj:
        # Prefer Fortran field layout (ni, nj) from FST unpack.
        return np.ascontiguousarray(np.reshape(a, (ni, nj), order="F").T, dtype=np.float32)
    raise ValueError(f"cannot reshape plane {a.shape} to (nj={nj}, ni={ni})")


def _spatial_dim_names(dims: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Return (y_name, x_name) for CF / boreal spatial dims."""
    y_aliases = ("y", "rlat", "lat", "j", "n")
    x_aliases = ("x", "rlon", "lon", "i", "e")
    y_name = next((d for d in dims if d in y_aliases), None)
    x_name = next((d for d in dims if d in x_aliases), None)
    return y_name, x_name


def _cube_to_tzyx(da: Any, ni: int, nj: int, n_z: int | None) -> np.ndarray:
    """Stack an xarray DataArray into C-order ``(time, vertical, y, x)`` or ``(time, y, x)``."""
    vals = np.asarray(da.values, dtype=np.float32)
    dims = tuple(str(d) for d in da.dims)
    # Drop leading size-1 member / ragged dims.
    while vals.ndim > 4 and vals.shape[0] == 1:
        vals = vals[0]
        dims = dims[1:]

    y_name, x_name = _spatial_dim_names(dims)

    # Fast path: already (time, vertical, rlat/y, rlon/x) C-order from open_fst.
    if vals.ndim == 4 and y_name and x_name and dims[-2] == y_name and dims[-1] == x_name:
        if vals.shape[-2] == nj and vals.shape[-1] == ni:
            n_lev = vals.shape[1] if n_z is None else min(vals.shape[1], n_z)
            return np.ascontiguousarray(vals[:, :n_lev, :, :])
        if vals.shape[-2] == ni and vals.shape[-1] == nj:
            # swapped horizontal
            n_lev = vals.shape[1] if n_z is None else min(vals.shape[1], n_z)
            return np.ascontiguousarray(np.swapaxes(vals[:, :n_lev], -1, -2))

    if vals.ndim == 2:
        return _as_c_yx(vals, ni, nj)[np.newaxis, ...]

    if vals.ndim == 3:
        out = np.empty((vals.shape[0], nj, ni), dtype=np.float32)
        for i in range(vals.shape[0]):
            out[i] = _as_c_yx(vals[i], ni, nj)
        return out

    if vals.ndim == 4:
        n_t, n_lev = vals.shape[0], vals.shape[1]
        if n_z is not None:
            n_lev = min(n_lev, n_z)
        out = np.empty((n_t, n_lev, nj, ni), dtype=np.float32)
        for t in range(n_t):
            for z in range(n_lev):
                out[t, z] = _as_c_yx(vals[t, z], ni, nj)
        return out

    raise ValueError(f"unsupported field rank {vals.ndim} dims={dims} shape={vals.shape}")

def _bitmap_from_values(arr: np.ndarray) -> np.ndarray:
    """``uint8`` keep-mask: 1 where finite, 0 where NaN/inf (fstd2nc keep>0 polarity)."""
    return np.isfinite(arr).astype(np.uint8, copy=False)


def _coord_1d(ds: Any, *names: str, length: int, fallback: np.ndarray) -> np.ndarray:
    for n in names:
        if n in ds.coords:
            c = np.asarray(ds.coords[n].values).reshape(-1)
            if c.size >= length:
                return c[:length].astype(np.float64, copy=False)
    return fallback


def load_raw_arenas(
    leads: Sequence[Path | str],
    *,
    nomvars: Sequence[str] = DEFAULT_NOMVARS,
    max_levels: int | None = None,
    cmc_root: Path | str | None = None,
) -> tuple[dict[str, Any], LoadTiming]:
    """Eager-decode ``leads`` via ``boreal.io.open_fst`` into synth-compatible arenas."""
    ensure_boreal_on_path(Path(cmc_root) if cmc_root else None)
    from boreal.io import open_fst  # noqa: WPS410 — soft dep after path inject

    paths = [str(Path(p)) for p in leads]
    if not paths:
        raise ValueError("no leads")

    t_scan0 = time.perf_counter()
    # Header/index work is folded into open_fst; we still split wall phases around it.
    t_dec0 = time.perf_counter()
    scan_ms = (t_dec0 - t_scan0) * 1000.0  # reserved; open_fst does scan+decode together

    ds_list = open_fst(
        paths,
        vars=list(nomvars),
        chunks=False,
        engine="mojo",
        open_data_mapping=True,
        cell_measures=True,
        apply_masks=True,
    )
    decode_ms = (time.perf_counter() - t_dec0) * 1000.0

    t_stack0 = time.perf_counter()
    # Prefer the Dataset with the deepest vertical axis that carries TT (pres → 50 lvls).
    def _vert_rank(d: Any) -> int:
        try:
            tt0 = _find_by_nomvar(d, "TT")
        except KeyError:
            return -1
        return int(tt0.sizes.get("vertical", tt0.shape[1] if tt0.ndim >= 4 else 1))

    primary = max(ds_list, key=_vert_rank)
    tt = _find_by_nomvar(primary, "TT")
    dims = tuple(str(d) for d in tt.dims)
    y_name, x_name = _spatial_dim_names(dims)
    if y_name and x_name:
        nj = int(tt.sizes[y_name])
        ni = int(tt.sizes[x_name])
    else:
        nj = int(tt.shape[-2])
        ni = int(tt.shape[-1])

    grtyp = str(
        tt.attrs.get("grtyp")
        or tt.attrs.get("fst_grtyp")
        or primary.attrs.get("fst_grtyp")
        or "Z"
    )
    tt_f32 = _cube_to_tzyx(tt, ni, nj, max_levels)
    if tt_f32.ndim == 3:
        tt_f32 = tt_f32[:, np.newaxis, :, :]
    n_t, n_z = tt_f32.shape[0], tt_f32.shape[1]

    def load4(nom: str) -> np.ndarray:
        da = _find_by_nomvar(primary, nom)
        cube = _cube_to_tzyx(da, ni, nj, max_levels)
        if cube.ndim == 3:
            cube = cube[:, np.newaxis, :, :]
        if cube.shape[0] != n_t or cube.shape[1] != n_z:
            raise ValueError(f"{nom} shape {cube.shape} != TT {(n_t, n_z, nj, ni)}")
        return cube

    uu_f32 = load4("UU") if "UU" in nomvars else np.zeros_like(tt_f32)
    vv_f32 = load4("VV") if "VV" in nomvars else np.zeros_like(tt_f32)
    hu_f32 = load4("HU") if "HU" in nomvars else np.zeros_like(tt_f32)

    bmap_3d = _bitmap_from_values(tt_f32)
    hu_bmap = _bitmap_from_values(hu_f32)

    # Surface TM is often absent from GDPS pres — fabricate from level-0 TT for pack API.
    has_tm = False
    try:
        tm_da = _find_by_nomvar(primary, "TM")
        tm_f32 = _cube_to_tzyx(tm_da, ni, nj, None)
        if tm_f32.ndim == 4:
            tm_f32 = tm_f32[:, 0]
        has_tm = True
    except KeyError:
        tm_f32 = tt_f32[:, 0].copy()
    tm_bmap = _bitmap_from_values(tm_f32)

    vertical = _coord_1d(
        primary,
        "vertical",
        "level",
        "pres",
        "pressure",
        length=n_z,
        fallback=np.linspace(1000.0, 10.0, n_z, dtype=np.float64),
    )
    times = _coord_1d(
        primary,
        "time",
        "forecast_period",
        length=n_t,
        fallback=np.arange(n_t, dtype=np.float64),
    )
    # time coord may be datetime64 — convert to float hours from first stamp.
    if np.issubdtype(times.dtype, np.datetime64):
        t0 = times[0]
        times = ((times - t0) / np.timedelta64(1, "h")).astype(np.float64)
    elif np.issubdtype(times.dtype, np.timedelta64):
        times = (times / np.timedelta64(1, "h")).astype(np.float64)

    y = _coord_1d(
        primary, "rlat", "y", "lat", length=nj, fallback=np.arange(nj, dtype=np.float64)
    )
    x = _coord_1d(
        primary, "rlon", "x", "lon", length=ni, fallback=np.arange(ni, dtype=np.float64)
    )

    if "areacell" in primary.data_vars:
        ac = np.asarray(primary["areacell"].values)
        areacell = np.ascontiguousarray(ac.reshape(nj, ni), dtype=np.float32) if ac.size == nj * ni else _as_c_yx(ac, ni, nj)
    elif "areacell" in primary.coords:
        areacell = _as_c_yx(primary.coords["areacell"].values, ni, nj)
    else:
        areacell = np.ones((nj, ni), dtype=np.float32)

    lon_2d = np.empty((nj, ni), dtype=np.float32)
    lat_2d = np.empty((nj, ni), dtype=np.float32)
    if "lon" in primary.coords and getattr(primary.coords["lon"], "ndim", 0) == 2:
        lon_2d[:] = np.ascontiguousarray(primary.coords["lon"].values, dtype=np.float32)
        lat_2d[:] = np.ascontiguousarray(primary.coords["lat"].values, dtype=np.float32)
    else:
        lon_2d[:] = x[np.newaxis, :].astype(np.float32)
        lat_2d[:] = y[:, np.newaxis].astype(np.float32)
    raw = {
        "grtyp": grtyp,
        "ni": ni,
        "nj": nj,
        "yin_ny": nj // 2,
        "tt_f32": tt_f32,
        "uu_f32": uu_f32,
        "vv_f32": vv_f32,
        "hu_f32": hu_f32,
        "bmap_3d": bmap_3d,
        "hu_bmap": hu_bmap,
        "tm_f32": tm_f32,
        "tm_bmap": tm_bmap,
        "x": x.astype(np.float64, copy=False),
        "y": y.astype(np.float64, copy=False),
        "vertical": vertical.astype(np.float64, copy=False)[:n_z],
        "time": times.astype(np.float64, copy=False)[:n_t],
        "lon_2d": lon_2d,
        "lat_2d": lat_2d,
        "areacell": areacell,
        "has_tm": has_tm,
        "source_paths": paths,
        "n_ds_open_fst": len(ds_list),
    }
    stack_ms = (time.perf_counter() - t_stack0) * 1000.0
    # Attribute open_fst wall to decode; scan is not separately exposed.
    timing = LoadTiming(scan_ms=scan_ms, decode_ms=decode_ms, stack_ms=stack_ms)
    return raw, timing


def pack_decoded_archive(raw: dict[str, Any], *, datyp: int = 134) -> Any:
    """Re-pack decoded arenas into ``PackedFSTArchive`` for M1 ``open_fst_catalog``.

    Uses the **builtin** reference ``datyp=134`` codec (not live ``cmc_io`` bytes).
    Real ``cmc_io`` unpack cost belongs in ``LoadTiming.decode_ms``.
    """
    import marray_xr as mxr

    return mxr.pack_boreal_fst_archive(raw, datyp=datyp)


def select_leads(
    root: Path | str | None = None,
    *,
    n_leads: int = 1,
    name: str | None = None,
) -> list[Path]:
    leads = list_pres_leads(root)
    if name is not None:
        hit = [p for p in leads if p.name == name]
        if not hit:
            where = root or _pres_root_from_env() or "<unset MARRAY_FST_PRES_ROOT>"
            raise FileNotFoundError(f"lead {name!r} not under {where}")
        return hit
    if n_leads <= 0:
        raise ValueError("n_leads must be positive")
    return leads[:n_leads]
