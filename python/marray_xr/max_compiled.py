"""Additive `TensorLayout` (`col_major` Fortran & `row_major` Boreal) and Compiled
MAX / GPU (`DeviceContext`) Session Wrapper (`python/marray_xr/max_compiled.py`).

Models `InferenceSession.load(graph)`: compiles/allocates device `TileTensor` arenas
and pre-stages static `aether` sparse CSR weights ONCE at session creation, then
executes zero-interpreter multi-op pipelines on multi-core CPU or GPU.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

import marray_xr as mxr


_BOUND_MAX_ABI = False


def _ensure_max_abi() -> ctypes.CDLL:
    global _BOUND_MAX_ABI
    lib = mxr.get_mojo_lib()
    if _BOUND_MAX_ABI:
        return lib

    i32_p = ctypes.POINTER(ctypes.c_int32)
    f32_p = ctypes.POINTER(ctypes.c_float)
    f64_p = ctypes.POINTER(ctypes.c_double)
    u8_p = ctypes.POINTER(ctypes.c_uint8)

    lib.marray_has_gpu_accelerator.argtypes = []
    lib.marray_has_gpu_accelerator.restype = ctypes.c_int32

    lib.marray_exec_vorticity_stencil_cpu.argtypes = [
        f32_p, f32_p, u8_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        f32_p,
    ]
    lib.marray_exec_vorticity_stencil_cpu.restype = ctypes.c_int32

    lib.marray_exec_compiled_humidex_wind_cpu.argtypes = [
        f32_p, f32_p, f32_p, f32_p, u8_p,
        ctypes.c_int64, ctypes.c_float,
        f32_p, f32_p,
    ]
    lib.marray_exec_compiled_humidex_wind_cpu.restype = ctypes.c_int32

    lib.marray_max_session_create.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32,
        f32_p, i32_p, i32_p, f64_p,
    ]
    lib.marray_max_session_create.restype = ctypes.c_int64

    lib.marray_max_session_destroy.argtypes = [ctypes.c_int64]
    lib.marray_max_session_destroy.restype = ctypes.c_int32

    lib.marray_max_session_stage_cube.argtypes = [
        ctypes.c_int64, f32_p, f32_p, f32_p, f32_p, u8_p
    ]
    lib.marray_max_session_stage_cube.restype = ctypes.c_int32

    lib.marray_max_session_vorticity_gpu.argtypes = [
        ctypes.c_int64, ctypes.c_int32,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int32, f32_p,
    ]
    lib.marray_max_session_vorticity_gpu.restype = ctypes.c_int32

    lib.marray_max_session_humidex_wind_gpu.argtypes = [
        ctypes.c_int64, ctypes.c_int32, ctypes.c_int32,
        f32_p, f32_p, f32_p, f32_p, u8_p,
        ctypes.c_float, f32_p, f32_p,
    ]
    lib.marray_max_session_humidex_wind_gpu.restype = ctypes.c_int32

    lib.marray_max_session_vertical_cape_gpu.argtypes = [
        ctypes.c_int64, ctypes.c_int32, ctypes.c_int32,
        f32_p, f32_p, u8_p,
        ctypes.c_float, f32_p,
    ]
    lib.marray_max_session_vertical_cape_gpu.restype = ctypes.c_int32

    lib.marray_max_session_fst_decode_aether_gpu.argtypes = [
        ctypes.c_int64, u8_p, u8_p, u8_p, u8_p, ctypes.c_double, f64_p
    ]
    lib.marray_max_session_fst_decode_aether_gpu.restype = ctypes.c_int32

    _BOUND_MAX_ABI = True
    return lib


def has_gpu_accelerator() -> bool:
    return bool(_ensure_max_abi().marray_has_gpu_accelerator() != 0)


def vorticity_stencil_cpu(
    u: np.ndarray,
    v: np.ndarray,
    bmap: np.ndarray,
    *,
    dx: float = 15000.0,
    dy: float = 15000.0,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Execute layout-polymorphic `TileTensor` 5-point vorticity stencil on CPU.
    Automatically dispatches `col_major((ni, nj, ns))` for Fortran arrays and
    `row_major((ns, nj, ni))` for Boreal C-order arrays with ZERO transpose!
    """
    lib = _ensure_max_abi()
    is_f = bool(u.flags["F_CONTIGUOUS"] and not u.flags["C_CONTIGUOUS"])
    if is_f:
        ni, nj = int(u.shape[0]), int(u.shape[1])
        ns = int(np.prod(u.shape[2:])) if u.ndim > 2 else 1
        out = np.empty_like(u, dtype=np.float32, order="F")
    else:
        nj, ni = int(u.shape[-2]), int(u.shape[-1])
        ns = int(np.prod(u.shape[:-2])) if u.ndim > 2 else 1
        out = np.empty_like(u, dtype=np.float32, order="C")

    f32_p = ctypes.POINTER(ctypes.c_float)
    u8_p = ctypes.POINTER(ctypes.c_uint8)
    lib.marray_exec_vorticity_stencil_cpu(
        u.ctypes.data_as(f32_p),
        v.ctypes.data_as(f32_p),
        bmap.ctypes.data_as(u8_p),
        ns,
        nj,
        ni,
        1 if is_f else 0,
        float(0.5 / dx),
        float(0.5 / dy),
        float(fill_value),
        out.ctypes.data_as(f32_p),
    )
    return out


def compiled_humidex_wind_cpu(
    tt: np.ndarray,
    hu: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    bmap: np.ndarray,
    *,
    fill_value: float = mxr.FST_MASK_FILL,
) -> tuple[np.ndarray, np.ndarray]:
    """Static compiled CPU `Humidex + WindModulus` (`parallelize` + `vectorize[W]`, zero VM)."""
    lib = _ensure_max_abi()
    out_hmx = np.empty_like(tt, dtype=np.float32)
    out_uv = np.empty_like(tt, dtype=np.float32)
    f32_p = ctypes.POINTER(ctypes.c_float)
    u8_p = ctypes.POINTER(ctypes.c_uint8)
    lib.marray_exec_compiled_humidex_wind_cpu(
        tt.ctypes.data_as(f32_p),
        hu.ctypes.data_as(f32_p),
        uu.ctypes.data_as(f32_p),
        vv.ctypes.data_as(f32_p),
        bmap.ctypes.data_as(u8_p),
        int(tt.size),
        float(fill_value),
        out_hmx.ctypes.data_as(f32_p),
        out_uv.ctypes.data_as(f32_p),
    )
    return out_hmx, out_uv


class CompiledMaxSession:
    """Persistent GPU/MAX `TileTensor` session (`DeviceContext` + pre-staged CSR weights)."""

    def __init__(
        self,
        shape_4d: tuple[int, int, int, int],
        levels: np.ndarray,
        *,
        csr_indptr: np.ndarray | None = None,
        csr_indices: np.ndarray | None = None,
        csr_weights: np.ndarray | None = None,
    ) -> None:
        self._lib = _ensure_max_abi()
        self.n_t, self.nk, self.nj, self.ni = (int(x) for x in shape_4d)
        lvls_f32 = np.ascontiguousarray(levels, dtype=np.float32)

        if csr_indptr is not None and csr_indices is not None and csr_weights is not None:
            indptr = np.ascontiguousarray(csr_indptr, dtype=np.int32)
            indices = np.ascontiguousarray(csr_indices, dtype=np.int32)
            weights = np.ascontiguousarray(csr_weights, dtype=np.float64)
            self.n_dst = int(len(indptr) - 1)
            nnz = int(len(indices))
        else:
            indptr = np.zeros(2, dtype=np.int32)
            indices = np.zeros(1, dtype=np.int32)
            weights = np.zeros(1, dtype=np.float64)
            self.n_dst = 1
            nnz = 1

        f32_p = ctypes.POINTER(ctypes.c_float)
        i32_p = ctypes.POINTER(ctypes.c_int32)
        f64_p = ctypes.POINTER(ctypes.c_double)

        self._handle = int(
            self._lib.marray_max_session_create(
                self.n_t,
                self.nk,
                self.nj,
                self.ni,
                self.n_dst,
                nnz,
                lvls_f32.ctypes.data_as(f32_p),
                indptr.ctypes.data_as(i32_p),
                indices.ctypes.data_as(i32_p),
                weights.ctypes.data_as(f64_p),
            )
        )
        if self._handle == 0:
            raise RuntimeError("Failed to initialize CompiledMaxSession GPU DeviceContext")

    def close(self) -> None:
        if getattr(self, "_handle", 0) != 0:
            self._lib.marray_max_session_destroy(self._handle)
            self._handle = 0

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "CompiledMaxSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def stage_cube(
        self, tt: np.ndarray, hu: np.ndarray, uu: np.ndarray, vv: np.ndarray, bmap: np.ndarray
    ) -> None:
        f32_p = ctypes.POINTER(ctypes.c_float)
        u8_p = ctypes.POINTER(ctypes.c_uint8)
        rc = self._lib.marray_max_session_stage_cube(
            self._handle,
            tt.ctypes.data_as(f32_p),
            hu.ctypes.data_as(f32_p),
            uu.ctypes.data_as(f32_p),
            vv.ctypes.data_as(f32_p),
            bmap.ctypes.data_as(u8_p),
        )
        if rc != 0:
            raise RuntimeError(f"stage_cube failed with status {rc}")

    def vorticity_gpu(
        self,
        *,
        fortran_order: bool = False,
        dx: float = 15000.0,
        dy: float = 15000.0,
        fill_value: float = np.nan,
        copy_back: bool = True,
    ) -> np.ndarray:
        shape = (self.ni, self.nj, self.n_t * self.nk) if fortran_order else (self.n_t, self.nk, self.nj, self.ni)
        out = np.empty(shape if copy_back else (1,), dtype=np.float32, order="F" if fortran_order else "C")
        f32_p = ctypes.POINTER(ctypes.c_float)
        rc = self._lib.marray_max_session_vorticity_gpu(
            self._handle,
            1 if fortran_order else 0,
            float(0.5 / dx),
            float(0.5 / dy),
            float(fill_value),
            1 if copy_back else 0,
            out.ctypes.data_as(f32_p),
        )
        if rc != 0:
            raise RuntimeError(f"vorticity_gpu failed with status {rc}")
        return out

    def humidex_wind_gpu(
        self,
        tt: np.ndarray,
        hu: np.ndarray,
        uu: np.ndarray,
        vv: np.ndarray,
        bmap: np.ndarray,
        *,
        upload_inputs: bool = True,
        copy_outputs: bool = True,
        fill_value: float = mxr.FST_MASK_FILL,
    ) -> tuple[np.ndarray, np.ndarray]:
        sh = tt.shape if copy_outputs else (1,)
        out_hmx = np.empty(sh, dtype=np.float32)
        out_uv = np.empty(sh, dtype=np.float32)
        f32_p = ctypes.POINTER(ctypes.c_float)
        u8_p = ctypes.POINTER(ctypes.c_uint8)
        rc = self._lib.marray_max_session_humidex_wind_gpu(
            self._handle,
            1 if upload_inputs else 0,
            1 if copy_outputs else 0,
            tt.ctypes.data_as(f32_p),
            hu.ctypes.data_as(f32_p),
            uu.ctypes.data_as(f32_p),
            vv.ctypes.data_as(f32_p),
            bmap.ctypes.data_as(u8_p),
            float(fill_value),
            out_hmx.ctypes.data_as(f32_p),
            out_uv.ctypes.data_as(f32_p),
        )
        if rc != 0:
            raise RuntimeError(f"humidex_wind_gpu failed with status {rc}")
        return out_hmx, out_uv

    def vertical_cape_gpu(
        self,
        tt: np.ndarray,
        hu: np.ndarray,
        bmap: np.ndarray,
        *,
        upload_inputs: bool = True,
        copy_output: bool = True,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        sh = (self.n_t, self.nj, self.ni) if copy_output else (1,)
        out = np.empty(sh, dtype=np.float32)
        f32_p = ctypes.POINTER(ctypes.c_float)
        u8_p = ctypes.POINTER(ctypes.c_uint8)
        rc = self._lib.marray_max_session_vertical_cape_gpu(
            self._handle,
            1 if upload_inputs else 0,
            1 if copy_output else 0,
            tt.ctypes.data_as(f32_p),
            hu.ctypes.data_as(f32_p),
            bmap.ctypes.data_as(u8_p),
            float(fill_value),
            out.ctypes.data_as(f32_p),
        )
        if rc != 0:
            raise RuntimeError(f"vertical_cape_gpu failed with status {rc}")
        return out

    def fst_decode_and_aether_csr_gpu(
        self,
        raw_u_bytes: np.ndarray,
        raw_v_bytes: np.ndarray,
        raw_t_bytes: np.ndarray,
        raw_bm_bytes: np.ndarray,
        *,
        extrap_value: float = -999.0,
    ) -> np.ndarray:
        out = np.empty((4, self.n_dst), dtype=np.float64, order="C")
        u8_p = ctypes.POINTER(ctypes.c_uint8)
        f64_p = ctypes.POINTER(ctypes.c_double)
        rc = self._lib.marray_max_session_fst_decode_aether_gpu(
            self._handle,
            raw_u_bytes.ctypes.data_as(u8_p),
            raw_v_bytes.ctypes.data_as(u8_p),
            raw_t_bytes.ctypes.data_as(u8_p),
            raw_bm_bytes.ctypes.data_as(u8_p),
            float(extrap_value),
            out.ctypes.data_as(f64_p),
        )
        if rc != 0:
            raise RuntimeError(f"fst_decode_and_aether_csr_gpu failed with status {rc}")
        return out
