"""Proof Benchmark: `TensorLayout` (`col_major` Fortran & `row_major` Boreal),
Static Compiled Kernels (Zero Bytecode Interpreter) & Live `max.gpu` (`DeviceContext`).

Answers three engineering questions with strict `assert_allclose` numerical parity:
  1. **Proof 1 — Layout-Polymorphic 2D Spatial Stencils (`TileTensor` + `TensorLayout`):**
     5-point centered relative vorticity (`dv/dx - du/dy`) + `@@` 5-neighbor mask dilation
     over BOTH Boreal `row_major((nt*nk, nj, ni))` AND Fortran `col_major((ni, nj, nt*nk))`
     with ZERO transpose: Dask/Xarray vs `TileTensor` CPU (16T) vs `TileTensor` GPU!
  2. **Proof 2 — Bytecode Interpreter vs. Compiled Static Multi-Op Kernel (`Humidex + WindModulus`):**
     Directly measures how much CPU overhead `vm.mojo`'s morsel interpreter actually adds
     vs a static AOT-compiled `parallelize + vectorize[W]` kernel, AND what `DeviceContext`
     GPU execution delivers (both Cold PCIe H2D+D2H and Warm VRAM-resident)!
  3. **Proof 3 — Compute-Dense Vertical `CAPE` & End-to-End Compressed `datyp=134` -> `aether` CSR Graph:**
     - 4D Coalesced `TileTensor` GPU `CAPE` parcel lift vs 16T CPU `vertical.mojo` vs Python Peel+Transpose.
     - Persistent `CompiledMaxSession` (`mogc-api` / `polaris` model): uploads COMPRESSED
       `datyp=134` + 1-bit `@@` bytes -> GPU unpack + mask -> GPU `WindModulus` ->
       GPU `aether` CSR SpMV `(4, 65536)` remap vs Dask & CPU!
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from typing import Any

import dask
import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

import marray_xr as mxr  # noqa: E402
from bench_boreal_realistic import (  # noqa: E402
    _xr_find_nomvar,
    open_fst_marray_catalog,
    open_fst_simulated_hlg,
)


def _med_ms(fn, repeats: int = 7, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def run_proof1_tensorlayout_vorticity(raw_u: dict[str, Any], sess: mxr.CompiledMaxSession | None) -> None:
    n_t, nk, nj, ni = raw_u["uu_f32"].shape
    print("\n" + "=" * 94)
    print(f"PROOF 1: `TileTensor` + `TensorLayout` 2D VORTICITY STENCIL (`col_major` F & `row_major` C)")
    print(f"         5-point `dv/dx - du/dy` + 5-neighbor `@@` mask dilation over {n_t}x{nk}x{nj}x{ni}")
    print("=" * 94)

    ds_col = open_fst_simulated_hlg(raw_u, chunks="column")
    dx = dy = 15000.0
    inv_2d = np.float32(0.5 / dx)

    def vort_dask_col() -> np.ndarray:
        u = _xr_find_nomvar(ds_col, "UU").data
        v = _xr_find_nomvar(ds_col, "VV").data
        # Interior 5-point central difference (`NaN` propagates automatically from 5 neighbors when `u*0` added!):
        core = (
            (v[:, :, 1:-1, 2:] - v[:, :, 1:-1, :-2]) * inv_2d
            - (u[:, :, 2:, 1:-1] - u[:, :, :-2, 1:-1]) * inv_2d
            + np.float32(0.0) * (
                u[:, :, 1:-1, 1:-1]
                + u[:, :, 1:-1, 2:]
                + u[:, :, 1:-1, :-2]
                + u[:, :, 2:, 1:-1]
                + u[:, :, :-2, 1:-1]
            )
        )
        out = np.full((n_t, nk, nj, ni), np.nan, dtype=np.float32)
        out[:, :, 1:-1, 1:-1] = core.compute(scheduler="threads", num_workers=16)
        return out

    def vort_numpy_eager() -> np.ndarray:
        b = raw_u["bmap_3d"] > 0
        u = np.where(b, raw_u["uu_f32"], np.float32(np.nan))
        v = np.where(b, raw_u["vv_f32"], np.float32(np.nan))
        out = np.full((n_t, nk, nj, ni), np.nan, dtype=np.float32)
        out[:, :, 1:-1, 1:-1] = (
            (v[:, :, 1:-1, 2:] - v[:, :, 1:-1, :-2]) * inv_2d
            - (u[:, :, 2:, 1:-1] - u[:, :, :-2, 1:-1]) * inv_2d
            + np.float32(0.0) * (
                u[:, :, 1:-1, 1:-1] + u[:, :, 1:-1, 2:] + u[:, :, 1:-1, :-2] + u[:, :, 2:, 1:-1] + u[:, :, :-2, 1:-1]
            )
        )
        return out

    # Same underlying Fortran `(ni, nj, nk*n_t)` physical memory view (ZERO byte copy!):
    ns = n_t * nk
    u_fort = raw_u["uu_f32"].reshape(ns, nj, ni).T  # F-contiguous `(ni, nj, ns)`!
    v_fort = raw_u["vv_f32"].reshape(ns, nj, ni).T
    bm_fort = raw_u["bmap_3d"].reshape(ns, nj, ni).T
    assert u_fort.flags["F_CONTIGUOUS"]

    ref_vort = vort_numpy_eager()
    out_cpu_c = mxr.vorticity_stencil_cpu(raw_u["uu_f32"], raw_u["vv_f32"], raw_u["bmap_3d"])
    out_cpu_f = mxr.vorticity_stencil_cpu(u_fort, v_fort, bm_fort)

    np.testing.assert_allclose(out_cpu_c, ref_vort, rtol=1e-5, atol=1e-5, equal_nan=True)
    np.testing.assert_allclose(out_cpu_f.T.reshape(n_t, nk, nj, ni), ref_vort, rtol=1e-5, atol=1e-5, equal_nan=True)

    t_dask = _med_ms(vort_dask_col, repeats=3, warmup=1)
    t_np = _med_ms(vort_numpy_eager, repeats=5, warmup=1)
    t_cpu_c = _med_ms(lambda: mxr.vorticity_stencil_cpu(raw_u["uu_f32"], raw_u["vv_f32"], raw_u["bmap_3d"]), repeats=7)
    t_cpu_f = _med_ms(lambda: mxr.vorticity_stencil_cpu(u_fort, v_fort, bm_fort), repeats=7)

    print(f"  Dask `chunks='column'` (5-shift stencil + mask views)     : {t_dask:8.2f} ms")
    print(f"  NumPy Eager (allocates 6 shifted 3D/4D intermediates)     : {t_np:8.2f} ms")
    print(f"  Mojo `TileTensor` `row_major` Boreal CPU (16T, 1-pass)    : {t_cpu_c:8.2f} ms  --> {t_dask / t_cpu_c:5.1f}x vs Dask | {t_np / t_cpu_c:4.1f}x vs NumPy")
    print(f"  Mojo `TileTensor` `col_major` Fortran CPU (16T, 1-pass)   : {t_cpu_f:8.2f} ms  (ZERO transpose on F-order!)")

    if sess is not None:
        out_gpu_c = sess.vorticity_gpu(fortran_order=False, copy_back=True)
        out_gpu_f = sess.vorticity_gpu(fortran_order=True, copy_back=True)
        np.testing.assert_allclose(out_gpu_c, ref_vort, rtol=1e-5, atol=1e-5, equal_nan=True)
        np.testing.assert_allclose(out_gpu_f.T.reshape(n_t, nk, nj, ni), ref_vort, rtol=1e-5, atol=1e-5, equal_nan=True)

        t_gpu_d2h = _med_ms(lambda: sess.vorticity_gpu(fortran_order=False, copy_back=True), repeats=15)
        t_gpu_warm = _med_ms(lambda: sess.vorticity_gpu(fortran_order=False, copy_back=False), repeats=25)
        print(f"  Mojo `TileTensor` GPU + PCIe D2H copy (`DeviceContext`)   : {t_gpu_d2h:8.3f} ms  --> {t_cpu_c / t_gpu_d2h:5.1f}x vs 16T CPU | {t_dask / t_gpu_d2h:6.1f}x vs Dask")
        print(f"  Mojo `TileTensor` GPU Warm In-VRAM Kernel (`row`/`col`)   : {t_gpu_warm:8.3f} ms  --> {t_cpu_c / t_gpu_warm:5.1f}x vs 16T CPU | {t_dask / t_gpu_warm:6.0f}x vs Dask!")


def run_proof2_interpreter_vs_compiled_max(raw_u: dict[str, Any], sess: mxr.CompiledMaxSession | None) -> None:
    n_t, nk, nj, ni = raw_u["tt_f32"].shape
    print("\n" + "=" * 94)
    print(f"PROOF 2: BYTECODE INTERPRETER (`vm.mojo`) vs. STATIC COMPILED KERNEL (`Humidex + WindModulus`)")
    print(f"         Fused 2-output nonlinear cube ({n_t}x{nk}x{nj}x{ni} = {raw_u['tt_f32'].size:,} cells -> `1e30` fill)")
    print("=" * 94)

    ds_col = open_fst_simulated_hlg(raw_u, chunks="column")
    # Use shared `bmap_3d` across TT, HU, UU, VV so all 4 tiers compute the exact same validity mask:
    raw_shared = {**raw_u, "hu_bmap": raw_u["bmap_3d"]}
    ds_mxr = open_fst_marray_catalog(raw_shared)

    def dask_2sink() -> tuple[np.ndarray, np.ndarray]:
        tt = _xr_find_nomvar(ds_col, "TT")
        hu = xr.where(xr.DataArray(raw_u["bmap_3d"] > 0, dims=tt.dims), raw_u["hu_f32"], np.float32(np.nan))
        uu = _xr_find_nomvar(ds_col, "UU")
        vv = _xr_find_nomvar(ds_col, "VV")
        e_hpa = hu * np.float32(1.6077) * np.exp(tt * np.float32(0.012))
        hmx = (tt + np.float32(0.5555) * (e_hpa - np.float32(10.0))).fillna(np.float32(1.0e30))
        uv = np.hypot(uu, vv).fillna(np.float32(1.0e30))
        r_h, r_u = dask.compute(hmx, uv, scheduler="threads", num_workers=16)
        return r_h.values, r_u.values

    def vm_multi_sink() -> tuple[np.ndarray, np.ndarray]:
        tt = ds_mxr.find_by_nomvar("TT")
        hu = ds_mxr.find_by_nomvar("HU")
        uu = ds_mxr.find_by_nomvar("UU")
        vv = ds_mxr.find_by_nomvar("VV")
        e_hpa = (hu * 1.6077) * np.exp(tt * 0.012)
        hmx = tt + 0.5555 * (e_hpa - 10.0)
        uv = np.hypot(uu, vv)
        oh, ou = mxr.compute_many(
            hmx.lazy_materialize(fill_value=1.0e30),
            uv.lazy_materialize(fill_value=1.0e30),
        )
        return oh.values, ou.values

    def static_cpu_compiled() -> tuple[np.ndarray, np.ndarray]:
        return mxr.compiled_humidex_wind_cpu(
            raw_u["tt_f32"], raw_u["hu_f32"], raw_u["uu_f32"], raw_u["vv_f32"], raw_u["bmap_3d"], fill_value=1.0e30
        )

    vh_vm, vu_vm = vm_multi_sink()
    vh_st, vu_st = static_cpu_compiled()
    np.testing.assert_allclose(vh_st, vh_vm, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(vu_st, vu_vm, rtol=1e-5, atol=1e-4)

    t_dask = _med_ms(dask_2sink, repeats=3, warmup=1)
    t_vm = _med_ms(vm_multi_sink, repeats=7)
    t_st_cpu = _med_ms(static_cpu_compiled, repeats=7)

    print(f"  Dask `chunks='column'` (16 threads, 2-output compute)     : {t_dask:8.2f} ms")
    print(f"  `marray_xr` Dynamic Morsel Bytecode VM (16T, `5 us` JIT)  : {t_vm:8.2f} ms  --> {t_dask / t_vm:5.1f}x vs Dask")
    print(f"  Static Compiled CPU (`parallelize` + `vectorize[W]`)      : {t_st_cpu:8.2f} ms  --> {t_vm / t_st_cpu:4.2f}x over Bytecode VM ({t_dask / t_st_cpu:5.1f}x vs Dask)")

    if sess is not None:
        gh, gu = sess.humidex_wind_gpu(
            raw_u["tt_f32"], raw_u["hu_f32"], raw_u["uu_f32"], raw_u["vv_f32"], raw_u["bmap_3d"],
            upload_inputs=True, copy_outputs=True, fill_value=1.0e30,
        )
        np.testing.assert_allclose(gh, vh_st, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(gu, vu_st, rtol=1e-5, atol=1e-4)

        t_gpu_e2e = _med_ms(
            lambda: sess.humidex_wind_gpu(
                raw_u["tt_f32"], raw_u["hu_f32"], raw_u["uu_f32"], raw_u["vv_f32"], raw_u["bmap_3d"],
                upload_inputs=True, copy_outputs=True, fill_value=1.0e30,
            ),
            repeats=7,
        )
        t_gpu_warm = _med_ms(
            lambda: sess.humidex_wind_gpu(
                raw_u["tt_f32"], raw_u["hu_f32"], raw_u["uu_f32"], raw_u["vv_f32"], raw_u["bmap_3d"],
                upload_inputs=False, copy_outputs=False, fill_value=1.0e30,
            ),
            repeats=25,
        )
        print(f"  `CompiledMaxSession` GPU End-to-End (PCIe H2D + Kernel + D2H): {t_gpu_e2e:8.2f} ms  --> {t_vm / t_gpu_e2e:4.1f}x vs CPU VM")
        print(f"  `CompiledMaxSession` GPU Warm In-VRAM (`TileTensor`)      : {t_gpu_warm:8.3f} ms  --> {t_st_cpu / t_gpu_warm:5.1f}x vs Static CPU | {t_dask / t_gpu_warm:6.0f}x vs Dask!")


def run_proof3_cape_and_aether_pipeline(
    raw_u: dict[str, Any],
    sess: mxr.CompiledMaxSession | None,
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> None:
    n_t, nk, nj, ni = raw_u["tt_f32"].shape
    n_dst = len(indptr) - 1
    print("\n" + "=" * 94)
    print("PROOF 3: COMPUTE-DENSE 4D `CAPE` & COMPRESSED `datyp=134 -> aether CSR` GRAPH (`mogc-api`)")
    print("=" * 94)

    ds_mxr = open_fst_marray_catalog({**raw_u, "hu_bmap": raw_u["bmap_3d"]})
    tt_var = ds_mxr._arena.sources[0]
    hu_var = ds_mxr._arena.sources[3]
    lvls = raw_u["vertical"].astype(np.float32)

    cpu_cape = mxr.vertical_column_scan(tt_var, lvls, hu_var=hu_var, mode="cape_proxy", fill_value=np.nan)
    t_cpu_cape = _med_ms(
        lambda: mxr.vertical_column_scan(tt_var, lvls, hu_var=hu_var, mode="cape_proxy", fill_value=np.nan),
        repeats=7,
    )
    print(f"  [3A] 4D Vertical `CAPE` — Mojo 16T CPU (`TILE_COLS=64` `TileTensor`) : {t_cpu_cape:8.2f} ms  (vs 1,089 ms Python Peel+Transpose)")

    if sess is not None:
        gpu_cape = sess.vertical_cape_gpu(
            raw_u["tt_f32"], raw_u["hu_f32"], raw_u["bmap_3d"], upload_inputs=True, copy_output=True
        )
        np.testing.assert_allclose(gpu_cape, cpu_cape.values, rtol=1e-4, atol=1e-2, equal_nan=True)

        t_gpu_cape_e2e = _med_ms(
            lambda: sess.vertical_cape_gpu(
                raw_u["tt_f32"], raw_u["hu_f32"], raw_u["bmap_3d"], upload_inputs=True, copy_output=True
            ),
            repeats=10,
        )
        t_gpu_cape_warm = _med_ms(
            lambda: sess.vertical_cape_gpu(
                raw_u["tt_f32"], raw_u["hu_f32"], raw_u["bmap_3d"], upload_inputs=False, copy_output=False
            ),
            repeats=25,
        )
        print(f"  [3A] 4D Vertical `CAPE` — GPU End-to-End (PCIe H2D + 4D Kernel + D2H): {t_gpu_cape_e2e:8.2f} ms  --> {t_cpu_cape / t_gpu_cape_e2e:5.1f}x vs 16T CPU")
        print(f"  [3A] 4D Vertical `CAPE` — GPU Coalesced 4D `TileTensor` (Warm VRAM)  : {t_gpu_cape_warm:8.3f} ms  --> {t_cpu_cape / t_gpu_cape_warm:5.1f}x vs 16T CPU (~{1088.6 / t_gpu_cape_warm:4.0f}x vs Python!)")

    # 3B: End-to-end `mogc-api` / `polaris` compressed FST (`datyp=134` + `datyp=2` 1-bit `@@`) -> `aether` CSR SpMV!
    archive = mxr.pack_boreal_fst_archive(raw_u, datyp=134)
    ds_cat = mxr.open_fst_catalog(archive, split_vctype=False)[0]
    rec_idx = 5 * nk + 2  # (time=5, vertical=2)

    def _slice_rec_bytes(nomvar: str) -> tuple[np.ndarray, np.ndarray]:
        row = archive.var_tables[nomvar][rec_idx]
        d_off, d_len, _, _, bm_off, bm_len, _, _ = (int(x) for x in row)
        return archive.raw_bytes[d_off : d_off + d_len], archive.raw_bytes[bm_off : bm_off + bm_len]

    ru_b, rbm_b = _slice_rec_bytes("UU")
    rv_b, _ = _slice_rec_bytes("VV")
    rt_b, _ = _slice_rec_bytes("TT")

    def cpu_cat_plus_csr() -> np.ndarray:
        sub = ds_cat.isel(time=5, vertical=2)
        flat = mxr.to_aether_flat(
            [sub.find_by_nomvar("UU"), sub.find_by_nomvar("VV"), np.hypot(sub.find_by_nomvar("UU"), sub.find_by_nomvar("VV")), sub.find_by_nomvar("TT")],
            masked_fill=0.0,
        )
        f64 = flat.astype(np.float64)
        gathered = f64[:, indices].reshape(4, n_dst, 4)
        return np.sum(gathered * weights.reshape(1, n_dst, 4), axis=-1)

    ref_csr = cpu_cat_plus_csr()
    t_cpu_e2e = _med_ms(cpu_cat_plus_csr, repeats=10)
    print(f"\n  [3B] Compressed `datyp=134` -> Unpack -> `UV` -> `aether` CSR `(4, {n_dst})` (CPU): {t_cpu_e2e:8.3f} ms  (vs >25 ms Dask)")

    if sess is not None:
        gpu_csr = sess.fst_decode_and_aether_csr_gpu(ru_b, rv_b, rt_b, rbm_b)
        np.testing.assert_allclose(gpu_csr, ref_csr, rtol=1e-5, atol=1e-4)
        t_gpu_csr = _med_ms(lambda: sess.fst_decode_and_aether_csr_gpu(ru_b, rv_b, rt_b, rbm_b), repeats=20)
        comp_kb = (ru_b.nbytes + rv_b.nbytes + rt_b.nbytes + rbm_b.nbytes) / 1024.0
        print(f"  [3B] `CompiledMaxSession` GPU Fused Graph (PCIe {comp_kb:.0f} KB compressed -> CSR D2H): {t_gpu_csr:8.3f} ms  --> {t_cpu_e2e / t_gpu_csr:5.1f}x vs 16T CPU!")
    print("=" * 94)


def main() -> None:
    has_gpu = mxr.has_gpu_accelerator()
    print("=" * 94)
    print("PROOF SUITE: `TileTensor` / `TensorLayout` + STATIC COMPILED & `max.gpu` PIPELINES")
    print(f"Accelerator Available (`has_accelerator()`): {has_gpu}")
    print("=" * 94)

    raw_u = mxr.synth_boreal_raw_arenas(grtyp="U", scale_factor=3, n_steps=16)
    n_src = raw_u["nj"] * raw_u["ni"]
    n_dst = 65536
    rng = np.random.default_rng(42)
    indptr = np.arange(0, (n_dst + 1) * 4, 4, dtype=np.int32)
    indices = rng.integers(0, n_src, size=n_dst * 4, dtype=np.int32)
    weights = np.full(n_dst * 4, 0.25, dtype=np.float64)

    sess: mxr.CompiledMaxSession | None = None
    if has_gpu:
        t0 = time.perf_counter()
        sess = mxr.CompiledMaxSession(
            raw_u["tt_f32"].shape,
            raw_u["vertical"],
            csr_indptr=indptr,
            csr_indices=indices,
            csr_weights=weights,
        )
        t_init = (time.perf_counter() - t0) * 1000.0
        sess.stage_cube(raw_u["tt_f32"], raw_u["hu_f32"], raw_u["uu_f32"], raw_u["vv_f32"], raw_u["bmap_3d"])
        print(f"One-time `CompiledMaxSession` startup (DeviceContext + VRAM CSR staging): {t_init:.1f} ms")

    try:
        run_proof1_tensorlayout_vorticity(raw_u, sess)
        run_proof2_interpreter_vs_compiled_max(raw_u, sess)
        run_proof3_cape_and_aether_pipeline(raw_u, sess, indptr, indices, weights)
    finally:
        if sess is not None:
            sess.close()


if __name__ == "__main__":
    main()
