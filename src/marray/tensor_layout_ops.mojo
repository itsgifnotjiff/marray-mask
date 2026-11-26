"""Additive `TileTensor` / `TensorLayout` (`col_major` Fortran & `row_major` Boreal)
and MAX-Ready Compiled Static CPU + GPU (`max.gpu` `DeviceContext`) Pipeline Kernels.

Designed to:
  1. Never slow down or alter the `5 us`-startup `vm.mojo` L1 morsel bytecode VM.
  2. Use `TileTensor` + `TensorLayout` (`col_major` for Fortran `(ni, nj, ns)` and
     `row_major` for Boreal CF `(ns, nj, ni)`) + `std.algorithm.vectorize[W]` +
     `max.algorithm.parallelize` so ONE layout-polymorphic kernel handles both
     Fortran and Boreal memory orders without transposes or raw pointer offset math.
  3. Provide MAX-Graph-ready static compiled pipelines and a persistent
     `CompiledMaxSession` (`DeviceContext` + pre-staged `aether` sparse CSR weights
     on VRAM) executing seamlessly on multi-core CPU AND NVIDIA/AMD/Apple GPUs.
"""

from std.algorithm import vectorize
from std.math import ceildiv, exp, fma, log, max, min, sqrt
from std.memory import ImmPointer, Layout, MutPointer, alloc
from std.runtime import parallelism_level
from std.sys import has_accelerator, simd_width_of
from max.algorithm import parallelize
from max.gpu import block_idx, global_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import (
    ImmTileTensor,
    MutTileTensor,
    TensorLayout,
    TileTensor,
    col_major,
    row_major,
)


comptime W_F32: Int = simd_width_of[DType.float32]()


# ===----------------------------------------------------------------------=== #
# 1. Layout-Polymorphic 2D Vorticity Stencil (`col_major` Fortran & `row_major` Boreal)
# ===----------------------------------------------------------------------=== #


@inline(.always)
def _vorticity_5pt_fortran[
    LT: TensorLayout
](
    u: ImmTileTensor[DType.float32, LT, _],
    v: ImmTileTensor[DType.float32, LT, _],
    bmap: ImmTileTensor[DType.uint8, LT, _],
    i: Int,
    j: Int,
    s: Int,
    ni: Int,
    nj: Int,
    inv_2dx: Float32,
    inv_2dy: Float32,
    fill_val: Float32,
) -> Float32:
    """Fortran `col_major((ni, nj, ns))` 5-point centered vorticity with `@@` mask dilation."""
    comptime assert u.flat_rank == 3 and v.flat_rank == 3 and bmap.flat_rank == 3
    if i <= 0 or i >= ni - 1 or j <= 0 or j >= nj - 1:
        return fill_val
    if (
        rebind[UInt8](bmap[i, j, s]) == UInt8(0)
        or rebind[UInt8](bmap[i - 1, j, s]) == UInt8(0)
        or rebind[UInt8](bmap[i + 1, j, s]) == UInt8(0)
        or rebind[UInt8](bmap[i, j - 1, s]) == UInt8(0)
        or rebind[UInt8](bmap[i, j + 1, s]) == UInt8(0)
    ):
        return fill_val
    var dv_dx = (rebind[Float32](v[i + 1, j, s]) - rebind[Float32](v[i - 1, j, s])) * inv_2dx
    var du_dy = (rebind[Float32](u[i, j + 1, s]) - rebind[Float32](u[i, j - 1, s])) * inv_2dy
    return dv_dx - du_dy


@inline(.always)
def _vorticity_5pt_boreal[
    LT: TensorLayout
](
    u: ImmTileTensor[DType.float32, LT, _],
    v: ImmTileTensor[DType.float32, LT, _],
    bmap: ImmTileTensor[DType.uint8, LT, _],
    s: Int,
    j: Int,
    i: Int,
    nj: Int,
    ni: Int,
    inv_2dx: Float32,
    inv_2dy: Float32,
    fill_val: Float32,
) -> Float32:
    """Boreal CF `row_major((ns, nj, ni))` 5-point centered vorticity with `@@` mask dilation."""
    comptime assert u.flat_rank == 3 and v.flat_rank == 3 and bmap.flat_rank == 3
    if i <= 0 or i >= ni - 1 or j <= 0 or j >= nj - 1:
        return fill_val
    if (
        rebind[UInt8](bmap[s, j, i]) == UInt8(0)
        or rebind[UInt8](bmap[s, j, i - 1]) == UInt8(0)
        or rebind[UInt8](bmap[s, j, i + 1]) == UInt8(0)
        or rebind[UInt8](bmap[s, j - 1, i]) == UInt8(0)
        or rebind[UInt8](bmap[s, j + 1, i]) == UInt8(0)
    ):
        return fill_val
    var dv_dx = (rebind[Float32](v[s, j, i + 1]) - rebind[Float32](v[s, j, i - 1])) * inv_2dx
    var du_dy = (rebind[Float32](u[s, j + 1, i]) - rebind[Float32](u[s, j - 1, i])) * inv_2dy
    return dv_dx - du_dy


def exec_vorticity_stencil_cpu(
    u_ptr: ImmPointer[Float32, _],
    v_ptr: ImmPointer[Float32, _],
    bmap_ptr: ImmPointer[UInt8, _],
    n_slabs: Int,
    nj: Int,
    ni: Int,
    fortran_order: Bool,
    inv_2dx: Float32,
    inv_2dy: Float32,
    fill_val: Float32,
    out_ptr: MutPointer[Float32, _],
):
    """Multi-core CPU `TileTensor` 2D stencil supporting BOTH `col_major` (Fortran)
    and `row_major` (Boreal CF) with zero transpose and 1-pass `@@` mask dilation.
    """
    var hw = max(1, parallelism_level())
    var nw = min(hw, max(1, n_slabs * nj))

    if fortran_order:
        var lay_f = col_major((Int32(ni), Int32(nj), Int32(n_slabs)))
        var tu = TileTensor(u_ptr, lay_f)
        var tv = TileTensor(v_ptr, lay_f)
        var tbm = TileTensor(bmap_ptr, lay_f)
        var tout = TileTensor(out_ptr, lay_f)

        def f_worker(wid: Int) {imm}:
            comptime assert tu.flat_rank == 3 and tout.flat_rank == 3
            var rows_tot = n_slabs * nj
            var r0 = (wid * rows_tot) // nw
            var r1 = ((wid + 1) * rows_tot) // nw
            for r in range(r0, r1):
                var s, j = divmod(r, nj)
                for i in range(ni):
                    var val = _vorticity_5pt_fortran(
                        tu, tv, tbm, i, j, s, ni, nj, inv_2dx, inv_2dy, fill_val
                    )
                    tout[i, j, s] = rebind[tout.ElementType](val)

        parallelize(f_worker, nw)
    else:
        var lay_c = row_major((Int32(n_slabs), Int32(nj), Int32(ni)))
        var tu = TileTensor(u_ptr, lay_c)
        var tv = TileTensor(v_ptr, lay_c)
        var tbm = TileTensor(bmap_ptr, lay_c)
        var tout = TileTensor(out_ptr, lay_c)

        def c_worker(wid: Int) {imm}:
            comptime assert tu.flat_rank == 3 and tout.flat_rank == 3
            var rows_tot = n_slabs * nj
            var r0 = (wid * rows_tot) // nw
            var r1 = ((wid + 1) * rows_tot) // nw
            for r in range(r0, r1):
                var s, j = divmod(r, nj)
                for i in range(ni):
                    var val = _vorticity_5pt_boreal(
                        tu, tv, tbm, s, j, i, nj, ni, inv_2dx, inv_2dy, fill_val
                    )
                    tout[s, j, i] = rebind[tout.ElementType](val)

        parallelize(c_worker, nw)


# ===----------------------------------------------------------------------=== #
# 2. Static Compiled CPU Kernel (`parallelize` + `vectorize[W]`, Zero Interpreter)
# ===----------------------------------------------------------------------=== #


def exec_compiled_humidex_wind_cpu(
    tt_ptr: ImmPointer[Float32, _],
    hu_ptr: ImmPointer[Float32, _],
    uu_ptr: ImmPointer[Float32, _],
    vv_ptr: ImmPointer[Float32, _],
    bmap_ptr: ImmPointer[UInt8, _],
    total_elems: Int,
    fill_val: Float32,
    out_hmx: MutPointer[Float32, _],
    out_uv: MutPointer[Float32, _],
):
    """AOT-compiled fused `Humidex + WindModulus` CPU kernel using `parallelize` +
    `std.algorithm.vectorize[W_F32]` with ZERO bytecode interpreter dispatch.
    Isolates exact interpreter overhead vs `vm.mojo` `compute_many`!
    """
    var hw = max(1, parallelism_level())
    var n_chunks = max(1, ceildiv(total_elems, 16384))
    var nw = min(hw, n_chunks)

    def cpu_worker(wid: Int) {imm}:
        var c0 = (wid * n_chunks) // nw
        var c1 = ((wid + 1) * n_chunks) // nw
        var start = c0 * 16384
        var stop = min(total_elems, c1 * 16384)
        var span = stop - start
        if span <= 0:
            return

        def _vec_body[width: Int](idx: Int) {
            imm start,
            imm tt_ptr,
            imm hu_ptr,
            imm uu_ptr,
            imm vv_ptr,
            imm bmap_ptr,
            imm fill_val,
            imm out_hmx,
            imm out_uv,
        }:
            var off = start + idx
            var bm = bmap_ptr.unsafe_load[width=width](off)
            var valid = bm.gt(SIMD[.uint8, width](0))
            var tt = tt_ptr.unsafe_load[width=width](off)
            var hu = hu_ptr.unsafe_load[width=width](off)
            var uu = uu_ptr.unsafe_load[width=width](off)
            var vv = vv_ptr.unsafe_load[width=width](off)

            var e_hpa = (hu * SIMD[.float32, width](1.6077)) * exp(
                tt * SIMD[.float32, width](0.012)
            )
            var hmx = fma(
                SIMD[.float32, width](0.5555),
                e_hpa - SIMD[.float32, width](10.0),
                tt,
            )
            var uv = sqrt(fma(uu, uu, vv * vv))
            var fv = SIMD[.float32, width](fill_val)

            out_hmx.unsafe_store(off, valid.select(hmx, fv))
            out_uv.unsafe_store(off, valid.select(uv, fv))

        vectorize[W_F32](span, _vec_body)

    parallelize(cpu_worker, nw)


# ===----------------------------------------------------------------------=== #
# 3. GPU `TileTensor` Kernels (`max.gpu` Portable Across NVIDIA / AMD / Apple)
# ===----------------------------------------------------------------------=== #


def _gpu_vorticity_boreal_kernel[
    LT: TensorLayout
](
    u: ImmTileTensor[DType.float32, LT, ImmUntrackedOrigin],
    v: ImmTileTensor[DType.float32, LT, ImmUntrackedOrigin],
    bmap: ImmTileTensor[DType.uint8, LT, ImmUntrackedOrigin],
    out_t: MutTileTensor[DType.float32, LT, MutUntrackedOrigin],
    n_slabs: Int32,
    nj: Int32,
    ni: Int32,
    inv_2dx: Float32,
    inv_2dy: Float32,
    fill_val: Float32,
):
    comptime assert u.flat_rank == 3 and v.flat_rank == 3 and bmap.flat_rank == 3 and out_t.flat_rank == 3
    var i = global_idx.x
    var sj = global_idx.y
    var nj_i = Int(nj)
    var ni_i = Int(ni)
    var ns_i = Int(n_slabs)
    if i >= ni_i or sj >= ns_i * nj_i:
        return
    var s, j = divmod(sj, nj_i)
    var val = _vorticity_5pt_boreal(
        u, v, bmap, s, j, i, nj_i, ni_i, inv_2dx, inv_2dy, fill_val
    )
    out_t[s, j, i] = rebind[out_t.ElementType](val)


def _gpu_vorticity_fortran_kernel[
    LT: TensorLayout
](
    u: ImmTileTensor[DType.float32, LT, ImmUntrackedOrigin],
    v: ImmTileTensor[DType.float32, LT, ImmUntrackedOrigin],
    bmap: ImmTileTensor[DType.uint8, LT, ImmUntrackedOrigin],
    out_t: MutTileTensor[DType.float32, LT, MutUntrackedOrigin],
    n_slabs: Int32,
    nj: Int32,
    ni: Int32,
    inv_2dx: Float32,
    inv_2dy: Float32,
    fill_val: Float32,
):
    comptime assert u.flat_rank == 3 and v.flat_rank == 3 and bmap.flat_rank == 3 and out_t.flat_rank == 3
    var i = global_idx.x
    var sj = global_idx.y
    var nj_i = Int(nj)
    var ni_i = Int(ni)
    var ns_i = Int(n_slabs)
    if i >= ni_i or sj >= ns_i * nj_i:
        return
    var s, j = divmod(sj, nj_i)
    var val = _vorticity_5pt_fortran(
        u, v, bmap, i, j, s, ni_i, nj_i, inv_2dx, inv_2dy, fill_val
    )
    out_t[i, j, s] = rebind[out_t.ElementType](val)


def _gpu_humidex_wind_kernel[
    LT: TensorLayout
](
    tt: ImmTileTensor[DType.float32, LT, ImmUntrackedOrigin],
    hu: ImmTileTensor[DType.float32, LT, ImmUntrackedOrigin],
    uu: ImmTileTensor[DType.float32, LT, ImmUntrackedOrigin],
    vv: ImmTileTensor[DType.float32, LT, ImmUntrackedOrigin],
    bmap: ImmTileTensor[DType.uint8, LT, ImmUntrackedOrigin],
    out_hmx: MutTileTensor[DType.float32, LT, MutUntrackedOrigin],
    out_uv: MutTileTensor[DType.float32, LT, MutUntrackedOrigin],
    total_elems: Int32,
    fill_val: Float32,
):
    comptime assert (
        tt.flat_rank == 1
        and hu.flat_rank == 1
        and uu.flat_rank == 1
        and vv.flat_rank == 1
        and bmap.flat_rank == 1
        and out_hmx.flat_rank == 1
        and out_uv.flat_rank == 1
    )
    var tid = global_idx.x
    if tid >= Int(total_elems):
        return
    if rebind[UInt8](bmap[tid]) == UInt8(0):
        out_hmx[tid] = rebind[out_hmx.ElementType](fill_val)
        out_uv[tid] = rebind[out_uv.ElementType](fill_val)
        return
    var t_v = rebind[Float32](tt[tid])
    var h_v = rebind[Float32](hu[tid])
    var u_v = rebind[Float32](uu[tid])
    var v_v = rebind[Float32](vv[tid])
    var e_hpa = (h_v * Float32(1.6077)) * exp(t_v * Float32(0.012))
    var hmx = fma(Float32(0.5555), e_hpa - Float32(10.0), t_v)
    var uv = sqrt(fma(u_v, u_v, v_v * v_v))
    out_hmx[tid] = rebind[out_hmx.ElementType](hmx)
    out_uv[tid] = rebind[out_uv.ElementType](uv)


def _gpu_vertical_cape_4d_kernel[
    LT4: TensorLayout, LT3: TensorLayout, LT1: TensorLayout
](
    tt: ImmTileTensor[DType.float32, LT4, ImmUntrackedOrigin],
    hu: ImmTileTensor[DType.float32, LT4, ImmUntrackedOrigin],
    bmap: ImmTileTensor[DType.uint8, LT4, ImmUntrackedOrigin],
    levels: ImmTileTensor[DType.float32, LT1, ImmUntrackedOrigin],
    out_cape: MutTileTensor[DType.float32, LT3, MutUntrackedOrigin],
    n_t: Int32,
    nk: Int32,
    nj: Int32,
    ni: Int32,
    fill_val: Float32,
):
    """Coalesced 4D `TileTensor` `(t, z, y, x)` GPU column-scan kernel!
    Adjacent threads in a warp own adjacent `x` columns -> 100% coalesced 128B GDDR
    loads across every `z = 0..nk-1` level with ZERO 4D cube transpose!
    """
    comptime assert (
        tt.flat_rank == 4
        and hu.flat_rank == 4
        and bmap.flat_rank == 4
        and levels.flat_rank == 1
        and out_cape.flat_rank == 3
    )
    var x = global_idx.x
    var ty = global_idx.y
    var ni_i = Int(ni)
    var nj_i = Int(nj)
    var nt_i = Int(n_t)
    var nk_i = Int(nk)
    if x >= ni_i or ty >= nt_i * nj_i:
        return
    var t_idx, y = divmod(ty, nj_i)

    var launched = False
    var theta_e_p = Float32(0.0)
    var p_launch = Float32(1000.0)
    var cape = Float32(0.0)

    for z in range(nk_i):
        if rebind[UInt8](bmap[t_idx, z, y, x]) == UInt8(0):
            continue
        var t_env_k = rebind[Float32](tt[t_idx, z, y, x]) + Float32(273.15)
        var q_env = rebind[Float32](hu[t_idx, z, y, x]) * Float32(0.001)
        var p_z = max(Float32(10.0), rebind[Float32](levels[z]))
        if not launched:
            launched = True
            p_launch = p_z
            theta_e_p = t_env_k + Float32(2500.0) * q_env
        else:
            var p_prev = max(Float32(10.0), rebind[Float32](levels[z - 1]))
            var dlnp = log(max(p_prev, p_z) / min(p_prev, p_z))
            var ratio = max(Float32(0.05), p_z / p_launch)
            var t_parcel = theta_e_p * exp(Float32(0.2854) * log(ratio))
            var buoy = (t_parcel - t_env_k) / max(Float32(180.0), t_env_k)
            if buoy > Float32(0.0):
                cape = fma(Float32(287.05) * t_env_k * buoy, dlnp, cape)

    out_cape[t_idx, y, x] = rebind[out_cape.ElementType](
        cape if launched else fill_val
    )


def _gpu_unpack134_and_prep_aether_kernel[
    LT2: TensorLayout
](
    raw_u: ImmPointer[UInt8, ImmUntrackedOrigin],
    raw_v: ImmPointer[UInt8, ImmUntrackedOrigin],
    raw_t: ImmPointer[UInt8, ImmUntrackedOrigin],
    raw_bm: ImmPointer[UInt8, ImmUntrackedOrigin],
    flat_src: MutTileTensor[DType.float32, LT2, MutUntrackedOrigin],  # (4, n_src)
    n_src: Int32,
):
    """Fused GPU Custom-Op Stage 1: Decompresses `datyp=134` 64-cell blocks + 1-bit
    `datyp=2` `@@` bitmap for `UU, VV, TT`, computes `UV = hypot(UU, VV)`, and writes
    finite pre-zeroed `(4, n_src)` `TileTensor` (`Q11`: zero `NaN`s in `aether`!).
    """
    comptime assert flat_src.flat_rank == 2
    var k = global_idx.x
    if k >= Int(n_src):
        return
    var b_byte = raw_bm[unsafe_offset=k >> 3]
    var is_valid = ((b_byte >> UInt8(k & 7)) & UInt8(1)) != UInt8(0)
    if not is_valid:
        flat_src[0, k] = rebind[flat_src.ElementType](Float32(0.0))
        flat_src[1, k] = rebind[flat_src.ElementType](Float32(0.0))
        flat_src[2, k] = rebind[flat_src.ElementType](Float32(0.0))
        flat_src[3, k] = rebind[flat_src.ElementType](Float32(0.0))
        return

    var blk = k >> 6
    var lane = k & 63
    var blk_off = blk * 136

    var hu_hdr = raw_u.unsafe_offset(blk_off).unsafe_bitcast[Float32]()
    var u_q = raw_u.unsafe_offset(blk_off + 8).unsafe_bitcast[UInt16]()[unsafe_offset=lane]
    var u_val = fma(Float32(u_q), hu_hdr[unsafe_offset=1], hu_hdr[unsafe_offset=0])

    var hv_hdr = raw_v.unsafe_offset(blk_off).unsafe_bitcast[Float32]()
    var v_q = raw_v.unsafe_offset(blk_off + 8).unsafe_bitcast[UInt16]()[unsafe_offset=lane]
    var v_val = fma(Float32(v_q), hv_hdr[unsafe_offset=1], hv_hdr[unsafe_offset=0])

    var ht_hdr = raw_t.unsafe_offset(blk_off).unsafe_bitcast[Float32]()
    var t_q = raw_t.unsafe_offset(blk_off + 8).unsafe_bitcast[UInt16]()[unsafe_offset=lane]
    var t_val = fma(Float32(t_q), ht_hdr[unsafe_offset=1], ht_hdr[unsafe_offset=0])

    var uv_val = sqrt(fma(u_val, u_val, v_val * v_val))
    flat_src[0, k] = rebind[flat_src.ElementType](u_val)
    flat_src[1, k] = rebind[flat_src.ElementType](v_val)
    flat_src[2, k] = rebind[flat_src.ElementType](uv_val)
    flat_src[3, k] = rebind[flat_src.ElementType](t_val)


def _gpu_aether_csr_spmv_kernel[
    LTSrc: TensorLayout, LTDst: TensorLayout
](
    flat_src: ImmTileTensor[DType.float32, LTSrc, ImmUntrackedOrigin],  # (n_fields, n_src)
    indptr: ImmPointer[Int32, ImmUntrackedOrigin],
    indices: ImmPointer[Int32, ImmUntrackedOrigin],
    weights: ImmPointer[Float64, ImmUntrackedOrigin],
    out_dst: MutTileTensor[DType.float64, LTDst, MutUntrackedOrigin],   # (n_fields, n_dst)
    n_fields: Int32,
    n_dst: Int32,
    extrap_val: Float64,
):
    """Fused GPU Custom-Op Stage 2: Exact `aether.apply_weights_csr` (`float64`
    accumulator + `extrap_value` on empty rows) over batched `(n_fields, n_dst)`!
    """
    comptime assert flat_src.flat_rank == 2 and out_dst.flat_rank == 2
    var row = global_idx.x
    if row >= Int(n_dst):
        return
    var p0 = Int(indptr[unsafe_offset=row])
    var p1 = Int(indptr[unsafe_offset=row + 1])
    var nf = Int(n_fields)

    if p0 >= p1:
        for f in range(nf):
            out_dst[f, row] = rebind[out_dst.ElementType](extrap_val)
        return

    for f in range(nf):
        var acc = Float64(0.0)
        for p in range(p0, p1):
            var col = Int(indices[unsafe_offset=p])
            var w = weights[unsafe_offset=p]
            acc = fma(Float64(rebind[Float32](flat_src[f, col])), w, acc)
        out_dst[f, row] = rebind[out_dst.ElementType](acc)


# ===----------------------------------------------------------------------=== #
# 4. Persistent `CompiledMaxSession` (Models `InferenceSession.load(graph)`)
# ===----------------------------------------------------------------------=== #


struct CompiledMaxSession(Movable):
    """Persistent GPU/MAX execution session holding `DeviceContext`, pre-allocated
    device `TileTensor` buffers, and pre-staged `aether` sparse CSR weights on VRAM!
    Eliminates per-call context/malloc/weight-upload overhead (`mogc-api` / `polaris`).
    """

    var ctx: DeviceContext
    var max_elems: Int
    var n_t: Int
    var nk: Int
    var nj: Int
    var ni: Int
    var n_dst: Int
    var nnz: Int

    var d_tt: DeviceBuffer[DType.float32]
    var d_hu: DeviceBuffer[DType.float32]
    var d_uu: DeviceBuffer[DType.float32]
    var d_vv: DeviceBuffer[DType.float32]
    var d_bmap: DeviceBuffer[DType.uint8]
    var d_lvls: DeviceBuffer[DType.float32]
    var d_out_a: DeviceBuffer[DType.float32]
    var d_out_b: DeviceBuffer[DType.float32]

    # Pre-staged `aether` CSR buffers & compressed FST ingest buffers:
    var d_raw_u: DeviceBuffer[DType.uint8]
    var d_raw_v: DeviceBuffer[DType.uint8]
    var d_raw_t: DeviceBuffer[DType.uint8]
    var d_raw_bm: DeviceBuffer[DType.uint8]
    var d_flat_src: DeviceBuffer[DType.float32]
    var d_indptr: DeviceBuffer[DType.int32]
    var d_indices: DeviceBuffer[DType.int32]
    var d_weights: DeviceBuffer[DType.float64]
    var d_csr_out: DeviceBuffer[DType.float64]

    def __init__(
        out self,
        n_t: Int,
        nk: Int,
        nj: Int,
        ni: Int,
        n_dst: Int,
        nnz: Int,
        h_levels: ImmPointer[Float32, ImmUntrackedOrigin],
        h_indptr: ImmPointer[Int32, ImmUntrackedOrigin],
        h_indices: ImmPointer[Int32, ImmUntrackedOrigin],
        h_weights: ImmPointer[Float64, ImmUntrackedOrigin],
    ) raises:
        self.ctx = DeviceContext()
        self.n_t = n_t
        self.nk = nk
        self.nj = nj
        self.ni = ni
        self.n_dst = max(1, n_dst)
        self.nnz = max(1, nnz)
        var nelm_4d = max(1, n_t * nk * nj * ni)
        var nelm_2d = max(1, nj * ni)
        self.max_elems = nelm_4d

        self.d_tt = self.ctx.enqueue_create_buffer[DType.float32](nelm_4d)
        self.d_hu = self.ctx.enqueue_create_buffer[DType.float32](nelm_4d)
        self.d_uu = self.ctx.enqueue_create_buffer[DType.float32](nelm_4d)
        self.d_vv = self.ctx.enqueue_create_buffer[DType.float32](nelm_4d)
        self.d_bmap = self.ctx.enqueue_create_buffer[DType.uint8](nelm_4d)
        self.d_lvls = self.ctx.enqueue_create_buffer[DType.float32](max(1, nk))
        self.d_out_a = self.ctx.enqueue_create_buffer[DType.float32](nelm_4d)
        self.d_out_b = self.ctx.enqueue_create_buffer[DType.float32](nelm_4d)

        var rec_bytes = ceildiv(nelm_2d, 64) * 136
        var bm_bytes = ceildiv(nelm_2d, 8)
        self.d_raw_u = self.ctx.enqueue_create_buffer[DType.uint8](rec_bytes)
        self.d_raw_v = self.ctx.enqueue_create_buffer[DType.uint8](rec_bytes)
        self.d_raw_t = self.ctx.enqueue_create_buffer[DType.uint8](rec_bytes)
        self.d_raw_bm = self.ctx.enqueue_create_buffer[DType.uint8](bm_bytes)
        self.d_flat_src = self.ctx.enqueue_create_buffer[DType.float32](4 * nelm_2d)

        self.d_indptr = self.ctx.enqueue_create_buffer[DType.int32](self.n_dst + 1)
        self.d_indices = self.ctx.enqueue_create_buffer[DType.int32](self.nnz)
        self.d_weights = self.ctx.enqueue_create_buffer[DType.float64](self.nnz)
        self.d_csr_out = self.ctx.enqueue_create_buffer[DType.float64](4 * self.n_dst)

        if nk > 0:
            self.ctx.enqueue_copy(self.d_lvls, h_levels)
        if n_dst > 0 and nnz > 0:
            self.ctx.enqueue_copy(self.d_indptr, h_indptr)
            self.ctx.enqueue_copy(self.d_indices, h_indices)
            self.ctx.enqueue_copy(self.d_weights, h_weights)
        self.ctx.synchronize()

    def __init__(out self, *, deinit move: Self):
        self.ctx = move.ctx^
        self.max_elems = move.max_elems
        self.n_t = move.n_t
        self.nk = move.nk
        self.nj = move.nj
        self.ni = move.ni
        self.n_dst = move.n_dst
        self.nnz = move.nnz
        self.d_tt = move.d_tt^
        self.d_hu = move.d_hu^
        self.d_uu = move.d_uu^
        self.d_vv = move.d_vv^
        self.d_bmap = move.d_bmap^
        self.d_lvls = move.d_lvls^
        self.d_out_a = move.d_out_a^
        self.d_out_b = move.d_out_b^
        self.d_raw_u = move.d_raw_u^
        self.d_raw_v = move.d_raw_v^
        self.d_raw_t = move.d_raw_t^
        self.d_raw_bm = move.d_raw_bm^
        self.d_flat_src = move.d_flat_src^
        self.d_indptr = move.d_indptr^
        self.d_indices = move.d_indices^
        self.d_weights = move.d_weights^
        self.d_csr_out = move.d_csr_out^

    def stage_cube_inputs(
        mut self,
        h_tt: ImmPointer[Float32, ImmUntrackedOrigin],
        h_hu: ImmPointer[Float32, ImmUntrackedOrigin],
        h_uu: ImmPointer[Float32, ImmUntrackedOrigin],
        h_vv: ImmPointer[Float32, ImmUntrackedOrigin],
        h_bmap: ImmPointer[UInt8, ImmUntrackedOrigin],
    ) raises:
        self.ctx.enqueue_copy(self.d_tt, h_tt)
        self.ctx.enqueue_copy(self.d_hu, h_hu)
        self.ctx.enqueue_copy(self.d_uu, h_uu)
        self.ctx.enqueue_copy(self.d_vv, h_vv)
        self.ctx.enqueue_copy(self.d_bmap, h_bmap)
        self.ctx.synchronize()

    def exec_vorticity_gpu(
        mut self,
        fortran_order: Bool,
        inv_2dx: Float32,
        inv_2dy: Float32,
        fill_val: Float32,
        copy_back: Bool,
        h_out: MutPointer[Float32, MutUntrackedOrigin],
    ) raises:
        var ns = self.n_t * self.nk
        var gx = ceildiv(self.ni, 32)
        var gy = ceildiv(ns * self.nj, 8)
        if fortran_order:
            var lay_f = col_major((Int32(self.ni), Int32(self.nj), Int32(ns)))
            var tu = TileTensor(self.d_uu, lay_f)
            var tv = TileTensor(self.d_vv, lay_f)
            var tbm = TileTensor(self.d_bmap, lay_f)
            var tout = TileTensor(self.d_out_a, lay_f)
            comptime k_f = _gpu_vorticity_fortran_kernel[type_of(lay_f)]
            self.ctx.enqueue_function[k_f](
                tu.as_imm(),
                tv.as_imm(),
                tbm.as_imm(),
                tout,
                Int32(ns),
                Int32(self.nj),
                Int32(self.ni),
                inv_2dx,
                inv_2dy,
                fill_val,
                grid_dim=(gx, gy),
                block_dim=(32, 8),
            )
        else:
            var lay_c = row_major((Int32(ns), Int32(self.nj), Int32(self.ni)))
            var tu = TileTensor(self.d_uu, lay_c)
            var tv = TileTensor(self.d_vv, lay_c)
            var tbm = TileTensor(self.d_bmap, lay_c)
            var tout = TileTensor(self.d_out_a, lay_c)
            comptime k_c = _gpu_vorticity_boreal_kernel[type_of(lay_c)]
            self.ctx.enqueue_function[k_c](
                tu.as_imm(),
                tv.as_imm(),
                tbm.as_imm(),
                tout,
                Int32(ns),
                Int32(self.nj),
                Int32(self.ni),
                inv_2dx,
                inv_2dy,
                fill_val,
                grid_dim=(gx, gy),
                block_dim=(32, 8),
            )
        if copy_back:
            self.ctx.enqueue_copy(h_out, self.d_out_a)
        self.ctx.synchronize()

    def exec_humidex_wind_gpu(
        mut self,
        upload_inputs: Bool,
        copy_outputs: Bool,
        h_tt: ImmPointer[Float32, ImmUntrackedOrigin],
        h_hu: ImmPointer[Float32, ImmUntrackedOrigin],
        h_uu: ImmPointer[Float32, ImmUntrackedOrigin],
        h_vv: ImmPointer[Float32, ImmUntrackedOrigin],
        h_bmap: ImmPointer[UInt8, ImmUntrackedOrigin],
        fill_val: Float32,
        h_out_hmx: MutPointer[Float32, MutUntrackedOrigin],
        h_out_uv: MutPointer[Float32, MutUntrackedOrigin],
    ) raises:
        if upload_inputs:
            self.ctx.enqueue_copy(self.d_tt, h_tt)
            self.ctx.enqueue_copy(self.d_hu, h_hu)
            self.ctx.enqueue_copy(self.d_uu, h_uu)
            self.ctx.enqueue_copy(self.d_vv, h_vv)
            self.ctx.enqueue_copy(self.d_bmap, h_bmap)

        var lay1 = row_major((Int32(self.max_elems),))
        var t_tt = TileTensor(self.d_tt, lay1)
        var t_hu = TileTensor(self.d_hu, lay1)
        var t_uu = TileTensor(self.d_uu, lay1)
        var t_vv = TileTensor(self.d_vv, lay1)
        var t_bm = TileTensor(self.d_bmap, lay1)
        var t_oh = TileTensor(self.d_out_a, lay1)
        var t_ou = TileTensor(self.d_out_b, lay1)
        comptime kern = _gpu_humidex_wind_kernel[type_of(lay1)]
        self.ctx.enqueue_function[kern](
            t_tt.as_imm(),
            t_hu.as_imm(),
            t_uu.as_imm(),
            t_vv.as_imm(),
            t_bm.as_imm(),
            t_oh,
            t_ou,
            Int32(self.max_elems),
            fill_val,
            grid_dim=ceildiv(self.max_elems, 256),
            block_dim=256,
        )
        if copy_outputs:
            self.ctx.enqueue_copy(h_out_hmx, self.d_out_a)
            self.ctx.enqueue_copy(h_out_uv, self.d_out_b)
        self.ctx.synchronize()

    def exec_vertical_cape_gpu(
        mut self,
        upload_inputs: Bool,
        copy_output: Bool,
        h_tt: ImmPointer[Float32, ImmUntrackedOrigin],
        h_hu: ImmPointer[Float32, ImmUntrackedOrigin],
        h_bmap: ImmPointer[UInt8, ImmUntrackedOrigin],
        fill_val: Float32,
        h_out_cape: MutPointer[Float32, MutUntrackedOrigin],
    ) raises:
        if upload_inputs:
            self.ctx.enqueue_copy(self.d_tt, h_tt)
            self.ctx.enqueue_copy(self.d_hu, h_hu)
            self.ctx.enqueue_copy(self.d_bmap, h_bmap)

        var lay4 = row_major(
            (Int32(self.n_t), Int32(self.nk), Int32(self.nj), Int32(self.ni))
        )
        var lay3 = row_major((Int32(self.n_t), Int32(self.nj), Int32(self.ni)))
        var lay1 = row_major((Int32(self.nk),))
        var t_tt = TileTensor(self.d_tt, lay4)
        var t_hu = TileTensor(self.d_hu, lay4)
        var t_bm = TileTensor(self.d_bmap, lay4)
        var t_lv = TileTensor(self.d_lvls, lay1)
        var d_cape_sub = DeviceBuffer[DType.float32](
            self.ctx, self.d_out_a.unsafe_ptr(), self.n_t * self.nj * self.ni, owning=False
        )
        var t_out = TileTensor(d_cape_sub, lay3)
        comptime kern = _gpu_vertical_cape_4d_kernel[
            type_of(lay4), type_of(lay3), type_of(lay1)
        ]
        self.ctx.enqueue_function[kern](
            t_tt.as_imm(),
            t_hu.as_imm(),
            t_bm.as_imm(),
            t_lv.as_imm(),
            t_out,
            Int32(self.n_t),
            Int32(self.nk),
            Int32(self.nj),
            Int32(self.ni),
            fill_val,
            grid_dim=(ceildiv(self.ni, 32), ceildiv(self.n_t * self.nj, 8)),
            block_dim=(32, 8),
        )
        if copy_output:
            self.ctx.enqueue_copy(h_out_cape, d_cape_sub)
        self.ctx.synchronize()

    def exec_fst_decode_and_aether_csr_gpu(
        mut self,
        h_raw_u: ImmPointer[UInt8, ImmUntrackedOrigin],
        h_raw_v: ImmPointer[UInt8, ImmUntrackedOrigin],
        h_raw_t: ImmPointer[UInt8, ImmUntrackedOrigin],
        h_raw_bm: ImmPointer[UInt8, ImmUntrackedOrigin],
        extrap_val: Float64,
        h_csr_out_f64: MutPointer[Float64, MutUntrackedOrigin],
    ) raises:
        """Compiled End-to-End `mogc-api` / `polaris` Graph:
        Uploads COMPRESSED `datyp=134` + 1-bit `@@` record bytes -> GPU unpack & mask ->
        GPU `WindModulus` -> GPU `aether` CSR SpMV -> downloads compact `(4, n_dst)` tile!
        """
        var n_src = self.nj * self.ni
        self.ctx.enqueue_copy(self.d_raw_u, h_raw_u)
        self.ctx.enqueue_copy(self.d_raw_v, h_raw_v)
        self.ctx.enqueue_copy(self.d_raw_t, h_raw_t)
        self.ctx.enqueue_copy(self.d_raw_bm, h_raw_bm)

        var lay_src = row_major((Int32(4), Int32(n_src)))
        var lay_dst = row_major((Int32(4), Int32(self.n_dst)))
        var t_flat = TileTensor(self.d_flat_src, lay_src)
        var t_out = TileTensor(self.d_csr_out, lay_dst)

        comptime k_unpack = _gpu_unpack134_and_prep_aether_kernel[type_of(lay_src)]
        self.ctx.enqueue_function[k_unpack](
            ImmPointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(self.d_raw_u.unsafe_ptr())),
            ImmPointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(self.d_raw_v.unsafe_ptr())),
            ImmPointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(self.d_raw_t.unsafe_ptr())),
            ImmPointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(self.d_raw_bm.unsafe_ptr())),
            t_flat,
            Int32(n_src),
            grid_dim=ceildiv(n_src, 256),
            block_dim=256,
        )

        comptime k_spmv = _gpu_aether_csr_spmv_kernel[type_of(lay_src), type_of(lay_dst)]
        self.ctx.enqueue_function[k_spmv](
            t_flat.as_imm(),
            ImmPointer[Int32, ImmUntrackedOrigin](unsafe_from_address=Int(self.d_indptr.unsafe_ptr())),
            ImmPointer[Int32, ImmUntrackedOrigin](unsafe_from_address=Int(self.d_indices.unsafe_ptr())),
            ImmPointer[Float64, ImmUntrackedOrigin](unsafe_from_address=Int(self.d_weights.unsafe_ptr())),
            t_out,
            Int32(4),
            Int32(self.n_dst),
            extrap_val,
            grid_dim=ceildiv(self.n_dst, 256),
            block_dim=256,
        )

        self.ctx.enqueue_copy(h_csr_out_f64, self.d_csr_out)
        self.ctx.synchronize()
