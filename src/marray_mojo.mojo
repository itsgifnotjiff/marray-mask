"""General Mojo Python Extension & C-ABI Bytecode / Catalog Module (`marray_mojo`).

Exports:
  1. `PyInit_marray_mojo` (`PythonModuleBuilder`)
  2. `marray_exec_general_plan` (`abi("C")`) — general single-root bytecode VM
  3. `marray_exec_multi_sink_plan` (`abi("C")`) — M1 `RecordCatalog` + multi-sink
     single-pass fused VM (`globavg + zonavg`, batched `aether` `(n_fields, n_src)`,
     whole-record `ChunkDecoder` plane cache, `datyp=6` serial safety guard)
  4. `marray_exec_general_plan_f64` (`abi("C")`) — `Float64` morsel VM for
     `boreal-stat` `values_f64` (`r.diag`, `RMSE`, `ke`) parity gates
  5. `marray_exec_vertical_column_scan` (`abi("C")`) — L1-resident `TILE_COLS=64`
     `(column, level)` vertical physics schedule (`cmc_postproc_kernels`)
  6. `marray_builtin_fst_chunk_decoder` & `marray_pack_fst_var` (`abi("C")`) —
     stable C-ABI `ChunkDecoder` reference codec (`datyp=134` block float +
     `datyp=2, nbits=1` `@@` -> `float32` `0/1` bitmap planes)
  7. `marray_synth_boreal_cube` (`abi("C")`) — parallel Tier-1 `grtyp` fixture generator
"""

from std.math import ceildiv, cos, max, min
from std.memory import ImmPointer, MutPointer
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.runtime import initialize_runtime, parallelism_level
from std.sys import simd_width_of
from max.algorithm import parallelize

from marray.catalog import (
    CatalogSourceDesc,
    REC_META_STRIDE,
    builtin_fst_chunk_decode,
    pack_bmap_plane_datyp2_1bit,
    pack_f32_plane_datyp134,
)
from marray.slab import (
    VAR_HU,
    VAR_TM,
    VAR_TT,
    VAR_UU,
    VAR_VV,
    MaskedSlab,
    RawLayerScratch,
)
from marray.synth import (
    GRTYP_U,
    SynthGridSpec,
    build_axes,
    fill_and_decode_slab,
)
from marray.vertical import execute_vertical_column_scan
from marray.vm import (
    SourceBinding,
    compile_and_optimize_dag,
    compile_and_optimize_multi_dag,
    execute_bytecode_plan,
    execute_bytecode_plan_f64,
    execute_multi_sink_plan,
)


@export("marray_init_runtime")
def marray_init_runtime() abi("C") -> Int32:
    initialize_runtime()
    return Int32(parallelism_level())


@export("marray_builtin_fst_chunk_decoder")
def marray_builtin_fst_chunk_decoder(
    ctx_addr: Int64,
    raw_ptr: ImmPointer[UInt8, ImmUntrackedOrigin],
    raw_byte_len: Int64,
    datyp: Int32,
    nbits: Int32,
    ni: Int32,
    nj: Int32,
    nk: Int32,
    dst_f32: MutPointer[Float32, MutUntrackedOrigin],
) abi("C") -> Int32:
    return builtin_fst_chunk_decode(
        ctx_addr, raw_ptr, raw_byte_len, datyp, nbits, ni, nj, nk, dst_f32
    )


@export("marray_pack_fst_var")
def marray_pack_fst_var(
    src_f32: ImmPointer[Float32, ImmUntrackedOrigin],
    bmap_u8: ImmPointer[UInt8, ImmUntrackedOrigin],
    has_bmap: Int32,
    n_records: Int32,
    ni: Int32,
    nj: Int32,
    datyp: Int32,
    base_byte_offset: Int64,
    dst_archive_bytes: MutPointer[UInt8, MutUntrackedOrigin],
    out_rec_meta_i64: MutPointer[Int64, MutUntrackedOrigin],  # `n_records * 8` Int64s
) abi("C") -> Int64:
    """Pack `n_records` `(nj, ni)` slabs in parallel into whole-record `datyp=134`
    blocks + optional `datyp=2, nbits=1` `@@` companion records!
    Returns total bytes written starting at `base_byte_offset`.
    """
    var n_rec = Int(n_records)
    var nelm = Int(ni) * Int(nj)
    var data_rec_bytes = (
        ceildiv(nelm, 64) * 136 if Int(datyp) == 134 else nelm * 4
    )
    var bmap_rec_bytes = ceildiv(nelm, 8) if has_bmap != 0 else 0
    var stride_bytes = data_rec_bytes + bmap_rec_bytes

    var hw = max(1, parallelism_level())
    var nw = min(hw, max(1, n_rec))

    def pack_worker(wid: Int) {imm}:
        var r0 = (wid * n_rec) // nw
        var r1 = ((wid + 1) * n_rec) // nw
        for r in range(r0, r1):
            var rec_off = Int(base_byte_offset) + r * stride_bytes
            var p_src = src_f32.unsafe_offset(r * nelm)
            var p_dst = dst_archive_bytes.unsafe_offset(rec_off)
            if Int(datyp) == 134:
                _ = pack_f32_plane_datyp134(p_src, nelm, p_dst)
            else:
                var dst_f = p_dst.unsafe_bitcast[Float32]()
                for k in range(nelm):
                    dst_f[unsafe_offset=k] = p_src[unsafe_offset=k]

            var m_base = r * REC_META_STRIDE
            out_rec_meta_i64[unsafe_offset=m_base + 0] = Int64(rec_off)
            out_rec_meta_i64[unsafe_offset=m_base + 1] = Int64(data_rec_bytes)
            out_rec_meta_i64[unsafe_offset=m_base + 2] = Int64(datyp)
            out_rec_meta_i64[unsafe_offset=m_base + 3] = Int64(
                16 if Int(datyp) == 134 else 32
            )

            if has_bmap != 0:
                var bm_off = rec_off + data_rec_bytes
                _ = pack_bmap_plane_datyp2_1bit(
                    bmap_u8.unsafe_offset(r * nelm),
                    nelm,
                    dst_archive_bytes.unsafe_offset(bm_off),
                )
                out_rec_meta_i64[unsafe_offset=m_base + 4] = Int64(bm_off)
                out_rec_meta_i64[unsafe_offset=m_base + 5] = Int64(bmap_rec_bytes)
                out_rec_meta_i64[unsafe_offset=m_base + 6] = Int64(2)
                out_rec_meta_i64[unsafe_offset=m_base + 7] = Int64(1)
            else:
                out_rec_meta_i64[unsafe_offset=m_base + 4] = Int64(-1)
                out_rec_meta_i64[unsafe_offset=m_base + 5] = Int64(0)
                out_rec_meta_i64[unsafe_offset=m_base + 6] = Int64(0)
                out_rec_meta_i64[unsafe_offset=m_base + 7] = Int64(0)

    parallelize(pack_worker, nw)
    return Int64(n_rec * stride_bytes)


@export("marray_exec_general_plan")
def marray_exec_general_plan(
    src_i64: ImmPointer[Int64, ImmUntrackedOrigin],
    src_f32: ImmPointer[Float32, ImmUntrackedOrigin],
    n_sources: Int64,
    node_ops: ImmPointer[Int32, ImmUntrackedOrigin],
    node_a: ImmPointer[Int32, ImmUntrackedOrigin],
    node_b: ImmPointer[Int32, ImmUntrackedOrigin],
    node_c: ImmPointer[Int32, ImmUntrackedOrigin],
    node_p0: ImmPointer[Float32, ImmUntrackedOrigin],
    node_p1: ImmPointer[Float32, ImmUntrackedOrigin],
    n_nodes: Int64,
    ndim: Int64,
    full_extents: ImmPointer[Int64, ImmUntrackedOrigin],
    slice_starts: ImmPointer[Int64, ImmUntrackedOrigin],
    slice_stops: ImmPointer[Int64, ImmUntrackedOrigin],
    reduce_axes_mask: Int32,
    agg_mode: Int32,
    needs_global_std: Int32,
    weight_src_id: Int32,
    out_fill_f32: Float32,
    out_stats_f64: MutPointer[Float64, MutUntrackedOrigin],
    out_arr_f32: MutPointer[Float32, MutUntrackedOrigin],
) abi("C") -> Int32:
    var ns = Int(n_sources)
    var sources = List[SourceBinding](capacity=ns)
    for i in range(ns):
        var b = i * 11
        sources.append(
            SourceBinding(
                Int(src_i64[unsafe_offset=b + 0]),
                Int(src_i64[unsafe_offset=b + 1]),
                Int(src_i64[unsafe_offset=b + 2]),
                Int(src_i64[unsafe_offset=b + 3]),
                Int(src_i64[unsafe_offset=b + 4]),
                src_f32[unsafe_offset=i * 2 + 0],
                src_f32[unsafe_offset=i * 2 + 1],
                Int(src_i64[unsafe_offset=b + 5]),
                Int(src_i64[unsafe_offset=b + 6]),
                Int(src_i64[unsafe_offset=b + 7]),
                Int(src_i64[unsafe_offset=b + 8]),
                Int(src_i64[unsafe_offset=b + 9]),
                Int(src_i64[unsafe_offset=b + 10]),
                -1,
            )
        )

    var plan = compile_and_optimize_dag(
        node_ops,
        node_a,
        node_b,
        node_c,
        node_p0,
        node_p1,
        Int(n_nodes),
        sources,
        needs_sum_sq=(needs_global_std != 0),
    )

    execute_bytecode_plan(
        plan,
        sources,
        Int(ndim),
        full_extents,
        slice_starts,
        slice_stops,
        Int(reduce_axes_mask),
        Int(agg_mode),
        Int(weight_src_id),
        out_fill_f32,
        out_stats_f64,
        out_arr_f32,
    )
    out_stats_f64[unsafe_offset=7] = Float64(len(plan.insts))
    out_stats_f64[unsafe_offset=8] = Float64(plan.num_regs)
    return Int32(0)


@export("marray_exec_multi_sink_plan")
def marray_exec_multi_sink_plan(
    src_i64: ImmPointer[Int64, ImmUntrackedOrigin],   # `n_sources * 12` (col 11 = catalog_id)
    src_f32: ImmPointer[Float32, ImmUntrackedOrigin],
    n_sources: Int64,
    cat_i64: ImmPointer[Int64, ImmUntrackedOrigin],   # `n_cats * 9` Int64s
    cat_f32: ImmPointer[Float32, ImmUntrackedOrigin], # `n_cats * 2` Float32s (scale, offset)
    n_cats: Int64,
    node_ops: ImmPointer[Int32, ImmUntrackedOrigin],
    node_a: ImmPointer[Int32, ImmUntrackedOrigin],
    node_b: ImmPointer[Int32, ImmUntrackedOrigin],
    node_c: ImmPointer[Int32, ImmUntrackedOrigin],
    node_p0: ImmPointer[Float32, ImmUntrackedOrigin],
    node_p1: ImmPointer[Float32, ImmUntrackedOrigin],
    n_nodes: Int64,
    ndim: Int64,
    full_extents: ImmPointer[Int64, ImmUntrackedOrigin],
    slice_starts: ImmPointer[Int64, ImmUntrackedOrigin],
    slice_stops: ImmPointer[Int64, ImmUntrackedOrigin],
    n_sinks: Int64,
    sink_root_nodes: ImmPointer[Int32, ImmUntrackedOrigin],
    sink_red_masks: ImmPointer[Int32, ImmUntrackedOrigin],
    sink_agg_modes: ImmPointer[Int32, ImmUntrackedOrigin],
    sink_wt_srcs: ImmPointer[Int32, ImmUntrackedOrigin],
    sink_fills: ImmPointer[Float32, ImmUntrackedOrigin],
    sink_out_addrs: ImmPointer[Int64, ImmUntrackedOrigin],
    out_sink_stats_f64: MutPointer[Float64, MutUntrackedOrigin],
) abi("C") -> Int32:
    var ns = Int(n_sources)
    var sources = List[SourceBinding](capacity=ns)
    for i in range(ns):
        var b = i * 12
        sources.append(
            SourceBinding(
                Int(src_i64[unsafe_offset=b + 0]),
                Int(src_i64[unsafe_offset=b + 1]),
                Int(src_i64[unsafe_offset=b + 2]),
                Int(src_i64[unsafe_offset=b + 3]),
                Int(src_i64[unsafe_offset=b + 4]),
                src_f32[unsafe_offset=i * 2 + 0],
                src_f32[unsafe_offset=i * 2 + 1],
                Int(src_i64[unsafe_offset=b + 5]),
                Int(src_i64[unsafe_offset=b + 6]),
                Int(src_i64[unsafe_offset=b + 7]),
                Int(src_i64[unsafe_offset=b + 8]),
                Int(src_i64[unsafe_offset=b + 9]),
                Int(src_i64[unsafe_offset=b + 10]),
                Int(src_i64[unsafe_offset=b + 11]),
            )
        )

    var nc = Int(n_cats)
    var cat_descs = List[CatalogSourceDesc](capacity=nc)
    for c in range(nc):
        var cb = c * 9
        cat_descs.append(
            CatalogSourceDesc(
                Int(cat_i64[unsafe_offset=cb + 0]),
                Int(cat_i64[unsafe_offset=cb + 1]),
                Int(cat_i64[unsafe_offset=cb + 2]),
                Int(cat_i64[unsafe_offset=cb + 3]),
                Int(cat_i64[unsafe_offset=cb + 4]),
                Int(cat_i64[unsafe_offset=cb + 5]),
                Int(cat_i64[unsafe_offset=cb + 6]),
                Int(cat_i64[unsafe_offset=cb + 7]),
                Int(cat_i64[unsafe_offset=cb + 8]),
                cat_f32[unsafe_offset=c * 2 + 0],
                cat_f32[unsafe_offset=c * 2 + 1],
            )
        )

    var nk_sinks = Int(n_sinks)
    var roots = List[Int](capacity=nk_sinks)
    var want_sq = False
    for s in range(nk_sinks):
        roots.append(Int(sink_root_nodes[unsafe_offset=s]))
        if Int(sink_agg_modes[unsafe_offset=s]) == 4:
            want_sq = True

    var plan = compile_and_optimize_multi_dag(
        node_ops,
        node_a,
        node_b,
        node_c,
        node_p0,
        node_p1,
        Int(n_nodes),
        roots,
        sources,
        cat_descs,
        needs_sum_sq=want_sq,
    )

    execute_multi_sink_plan(
        plan,
        sources,
        cat_descs,
        Int(ndim),
        full_extents,
        slice_starts,
        slice_stops,
        nk_sinks,
        sink_red_masks,
        sink_agg_modes,
        sink_wt_srcs,
        sink_fills,
        sink_out_addrs,
        out_sink_stats_f64,
    )
    if nk_sinks > 0:
        out_sink_stats_f64[unsafe_offset=7] = Float64(len(plan.insts))
    return Int32(0)


@export("marray_exec_general_plan_f64")
def marray_exec_general_plan_f64(
    src_i64: ImmPointer[Int64, ImmUntrackedOrigin],
    src_f32: ImmPointer[Float32, ImmUntrackedOrigin],
    n_sources: Int64,
    node_ops: ImmPointer[Int32, ImmUntrackedOrigin],
    node_a: ImmPointer[Int32, ImmUntrackedOrigin],
    node_b: ImmPointer[Int32, ImmUntrackedOrigin],
    node_c: ImmPointer[Int32, ImmUntrackedOrigin],
    node_p0: ImmPointer[Float32, ImmUntrackedOrigin],
    node_p1: ImmPointer[Float32, ImmUntrackedOrigin],
    n_nodes: Int64,
    ndim: Int64,
    slice_starts: ImmPointer[Int64, ImmUntrackedOrigin],
    slice_stops: ImmPointer[Int64, ImmUntrackedOrigin],
    reduce_axes_mask: Int32,
    agg_mode: Int32,
    needs_global_std: Int32,
    weight_src_id: Int32,
    out_fill_f64: Float64,
    out_stats_f64: MutPointer[Float64, MutUntrackedOrigin],
    out_arr_f64: MutPointer[Float64, MutUntrackedOrigin],
) abi("C") -> Int32:
    var ns = Int(n_sources)
    var sources = List[SourceBinding](capacity=ns)
    for i in range(ns):
        var b = i * 11
        sources.append(
            SourceBinding(
                Int(src_i64[unsafe_offset=b + 0]),
                Int(src_i64[unsafe_offset=b + 1]),
                Int(src_i64[unsafe_offset=b + 2]),
                Int(src_i64[unsafe_offset=b + 3]),
                Int(src_i64[unsafe_offset=b + 4]),
                src_f32[unsafe_offset=i * 2 + 0],
                src_f32[unsafe_offset=i * 2 + 1],
                Int(src_i64[unsafe_offset=b + 5]),
                Int(src_i64[unsafe_offset=b + 6]),
                Int(src_i64[unsafe_offset=b + 7]),
                Int(src_i64[unsafe_offset=b + 8]),
                Int(src_i64[unsafe_offset=b + 9]),
                Int(src_i64[unsafe_offset=b + 10]),
                -1,
            )
        )
    var plan = compile_and_optimize_dag(
        node_ops,
        node_a,
        node_b,
        node_c,
        node_p0,
        node_p1,
        Int(n_nodes),
        sources,
        needs_sum_sq=(needs_global_std != 0),
    )
    execute_bytecode_plan_f64(
        plan,
        sources,
        Int(ndim),
        slice_starts,
        slice_stops,
        Int(reduce_axes_mask),
        Int(agg_mode),
        Int(weight_src_id),
        out_fill_f64,
        out_stats_f64,
        out_arr_f64,
    )
    out_stats_f64[unsafe_offset=7] = Float64(len(plan.insts))
    out_stats_f64[unsafe_offset=8] = Float64(plan.num_regs)
    return Int32(0)


@export("marray_exec_vertical_column_scan")
def marray_exec_vertical_column_scan(
    tt_f32: ImmPointer[Float32, ImmUntrackedOrigin],
    hu_f32: ImmPointer[Float32, ImmUntrackedOrigin],
    has_hu: Int32,
    bmap_ptr: ImmPointer[UInt8, ImmUntrackedOrigin],
    bmap_kind: Int32,
    levels_f32: ImmPointer[Float32, ImmUntrackedOrigin],
    n_t: Int32,
    nk: Int32,
    nj: Int32,
    ni: Int32,
    kernel_mode: Int32,
    ext_kernel_fn_addr: Int64,
    ext_kernel_ctx_addr: Int64,
    out_fill_f32: Float32,
    out_2d_per_t: MutPointer[Float32, MutUntrackedOrigin],
) abi("C") -> Int32:
    execute_vertical_column_scan(
        tt_f32,
        hu_f32,
        has_hu != 0,
        bmap_ptr,
        Int(bmap_kind),
        levels_f32,
        Int(n_t),
        Int(nk),
        Int(nj),
        Int(ni),
        Int(kernel_mode),
        Int(ext_kernel_fn_addr),
        Int(ext_kernel_ctx_addr),
        out_fill_f32,
        out_2d_per_t,
    )
    return Int32(0)


@export("marray_synth_grid_dims")
def marray_synth_grid_dims(
    grtyp: Int32,
    scale_factor: Int32,
    n_steps_override: Int32,
    out_dims_i64: MutPointer[Int64, MutUntrackedOrigin],
) abi("C") -> Int32:
    var spec = SynthGridSpec.bench(Int(grtyp)).scaled(Int(scale_factor))
    if n_steps_override > 0:
        spec.n_steps = Int(n_steps_override)
    out_dims_i64[unsafe_offset=0] = Int64(spec.ni)
    out_dims_i64[unsafe_offset=1] = Int64(spec.nj)
    out_dims_i64[unsafe_offset=2] = Int64(spec.n_levels())
    out_dims_i64[unsafe_offset=3] = Int64(spec.n_steps)
    out_dims_i64[unsafe_offset=4] = Int64(spec.yin_ny)
    return Int32(0)


@export("marray_synth_boreal_cube")
def marray_synth_boreal_cube(
    grtyp: Int32,
    scale_factor: Int32,
    n_steps_override: Int32,
    tt_f32: MutPointer[Float32, MutUntrackedOrigin],
    uu_f32: MutPointer[Float32, MutUntrackedOrigin],
    vv_f32: MutPointer[Float32, MutUntrackedOrigin],
    hu_f32: MutPointer[Float32, MutUntrackedOrigin],
    bmap_3d_u8: MutPointer[UInt8, MutUntrackedOrigin],
    hu_bmap_u8: MutPointer[UInt8, MutUntrackedOrigin],
    tm_f32: MutPointer[Float32, MutUntrackedOrigin],
    tm_bmap_u8: MutPointer[UInt8, MutUntrackedOrigin],
    x_1d_f64: MutPointer[Float64, MutUntrackedOrigin],
    y_1d_f64: MutPointer[Float64, MutUntrackedOrigin],
    levels_f64: MutPointer[Float64, MutUntrackedOrigin],
    times_f64: MutPointer[Float64, MutUntrackedOrigin],
    lon_2d_f32: MutPointer[Float32, MutUntrackedOrigin],
    lat_2d_f32: MutPointer[Float32, MutUntrackedOrigin],
    areacell_f32: MutPointer[Float32, MutUntrackedOrigin],
) abi("C") -> Int32:
    var spec = SynthGridSpec.bench(Int(grtyp)).scaled(Int(scale_factor))
    if n_steps_override > 0:
        spec.n_steps = Int(n_steps_override)
    var axes = build_axes(spec)

    var ni = spec.ni
    var nj = spec.nj
    var n_lvls = spec.n_levels()
    var n_steps = spec.n_steps
    var n_cells = ni * nj
    var n_slabs = n_steps * n_lvls

    for i in range(ni):
        x_1d_f64[unsafe_offset=i] = axes.x_1d[i]
    for j in range(nj):
        y_1d_f64[unsafe_offset=j] = axes.y_1d[j]
    for z in range(n_lvls):
        levels_f64[unsafe_offset=z] = spec.levels_hpa[z]
    for t in range(n_steps):
        times_f64[unsafe_offset=t] = Float64(t) * spec.step_dt_hr

    for j in range(nj):
        var seam = (
            spec.grtyp == GRTYP_U
            and spec.yin_ny > 0
            and j >= spec.yin_ny - 2
            and j < spec.yin_ny + 2
        )
        var row_off = j * ni
        for i in range(ni):
            var k = row_off + i
            var lon_v = axes.lon_f[k]
            var lat_v = axes.lat_f[k]
            lon_2d_f32[unsafe_offset=k] = lon_v
            lat_2d_f32[unsafe_offset=k] = lat_v
            if seam:
                areacell_f32[unsafe_offset=k] = 0.0
            else:
                var w_cos = max(
                    Float32(0.02), cos(lat_v * Float32(0.0174532925))
                )
                areacell_f32[unsafe_offset=k] = w_cos

    var hw = max(1, parallelism_level())
    var n_workers = min(hw, n_slabs)

    def fill_worker(wid: Int) {imm}:
        var s_begin = (wid * n_slabs) // n_workers
        var s_end = ((wid + 1) * n_slabs) // n_workers
        var scratch = RawLayerScratch()
        var dst = MaskedSlab(ni, nj)

        for s_idx in range(s_begin, s_end):
            var z_i = s_idx % n_lvls
            var t_i = s_idx // n_lvls
            var step_hr = Float32(Float64(t_i) * spec.step_dt_hr)
            var lvl_hpa = Float32(spec.levels_hpa[z_i])
            var base = (t_i * n_lvls + z_i) * n_cells

            fill_and_decode_slab(
                spec, axes, VAR_TT, step_hr, lvl_hpa, 0, ni, 0, nj, scratch, dst
            )
            for k in range(n_cells):
                tt_f32[unsafe_offset=base + k] = dst.data[k]
                bmap_3d_u8[unsafe_offset=base + k] = (
                    UInt8(0) if dst.is_masked_flat(k) else UInt8(1)
                )

            fill_and_decode_slab(
                spec, axes, VAR_UU, step_hr, lvl_hpa, 0, ni, 0, nj, scratch, dst
            )
            for k in range(n_cells):
                uu_f32[unsafe_offset=base + k] = dst.data[k]

            fill_and_decode_slab(
                spec, axes, VAR_VV, step_hr, lvl_hpa, 0, ni, 0, nj, scratch, dst
            )
            for k in range(n_cells):
                vv_f32[unsafe_offset=base + k] = dst.data[k]

            fill_and_decode_slab(
                spec, axes, VAR_HU, step_hr, lvl_hpa, 0, ni, 0, nj, scratch, dst
            )
            for k in range(n_cells):
                hu_f32[unsafe_offset=base + k] = dst.data[k]
                hu_bmap_u8[unsafe_offset=base + k] = (
                    UInt8(0) if dst.is_masked_flat(k) else UInt8(1)
                )

            if z_i == 0:
                var base_2d = t_i * n_cells
                fill_and_decode_slab(
                    spec, axes, VAR_TM, step_hr, 1013.25, 0, ni, 0, nj, scratch, dst
                )
                for k in range(n_cells):
                    tm_f32[unsafe_offset=base_2d + k] = dst.data[k]
                    tm_bmap_u8[unsafe_offset=base_2d + k] = (
                        UInt8(0) if dst.is_masked_flat(k) else UInt8(1)
                    )

    parallelize(fill_worker, n_workers)
    return Int32(0)


def py_engine_info() raises -> PythonObject:
    initialize_runtime()
    return Python.dict(
        simd_f32_width=PythonObject(simd_width_of[DType.float32]()),
        simd_f64_width=PythonObject(simd_width_of[DType.float64]()),
        threads=PythonObject(parallelism_level()),
        morsel_elems=PythonObject(2048),
    )


@export
def PyInit_marray_mojo() abi("C") -> PythonObject:
    try:
        initialize_runtime()
        var m = PythonModuleBuilder("marray_mojo")
        m.def_function[py_engine_info]("engine_info")
        return m.finalize()
    except e:
        abort(t"failed to initialize marray_mojo: {e}")
