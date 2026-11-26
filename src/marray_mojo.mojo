"""General Mojo Python Extension & C-ABI Bytecode Module (`marray_mojo`).

Exports:
  1. `PyInit_marray_mojo` (`PythonModuleBuilder`)
  2. `marray_exec_general_plan` (`abi("C")`) — general bytecode DAG + weighted
     reductions + direct `NaN` / `1e30` (`fst_mask_fill`) SIMD materialization
  3. `marray_synth_boreal_cube` (`abi("C")`) — parallel C-speed fixture generator
     wrapping `src/marray/synth.mojo` (`GRTYP_U/O/Z/L/G/N/M` Tier-1 FST catalog
     grids with `fstd2nc` `@@` bitmap polarity `1=keep, 0=mask` & `areacell`).
"""

from std.math import cos, max, min
from std.memory import ImmPointer, MutPointer
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.runtime import initialize_runtime, parallelism_level
from std.sys import simd_width_of
from max.algorithm import parallelize

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
from marray.vm import (
    SourceBinding,
    compile_and_optimize_dag,
    execute_bytecode_plan,
)


@export("marray_init_runtime")
def marray_init_runtime() abi("C") -> Int32:
    initialize_runtime()
    return Int32(parallelism_level())


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


@export("marray_synth_grid_dims")
def marray_synth_grid_dims(
    grtyp: Int32,
    scale_factor: Int32,
    n_steps_override: Int32,
    out_dims_i64: MutPointer[Int64, MutUntrackedOrigin],  # [ni, nj, n_levels, n_steps, yin_ny]
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
    # Boreal C-order `(n_steps, n_levels, nj, ni)` raw unpacked float32 slabs +
    # fstd2nc-polarity `@@` uint8 bitmaps (`1 = valid, 0 = masked`):
    tt_f32: MutPointer[Float32, MutUntrackedOrigin],
    uu_f32: MutPointer[Float32, MutUntrackedOrigin],
    vv_f32: MutPointer[Float32, MutUntrackedOrigin],
    hu_f32: MutPointer[Float32, MutUntrackedOrigin],
    bmap_3d_u8: MutPointer[UInt8, MutUntrackedOrigin],
    hu_bmap_u8: MutPointer[UInt8, MutUntrackedOrigin],
    # 3D surface `(n_steps, nj, ni)` SST + land/sea `@@` bitmap (`1 = ocean, 0 = land`):
    tm_f32: MutPointer[Float32, MutUntrackedOrigin],
    tm_bmap_u8: MutPointer[UInt8, MutUntrackedOrigin],
    # Coordinates & cell_measures (`areacell` with Yin-Yang overlap weight=0):
    x_1d_f64: MutPointer[Float64, MutUntrackedOrigin],
    y_1d_f64: MutPointer[Float64, MutUntrackedOrigin],
    levels_f64: MutPointer[Float64, MutUntrackedOrigin],
    times_f64: MutPointer[Float64, MutUntrackedOrigin],
    lon_2d_f32: MutPointer[Float32, MutUntrackedOrigin],
    lat_2d_f32: MutPointer[Float32, MutUntrackedOrigin],
    areacell_f32: MutPointer[Float32, MutUntrackedOrigin],
) abi("C") -> Int32:
    """Populate realistic Boreal `(time, vertical, y, x)` C-contiguous arenas
    using `src/marray/synth.mojo` across CPU cores.
    """
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

    # 2D lon, lat, and `areacell` (cosine-latitude cell area; 0 in Yin-Yang seam)
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

            # TT, UU, VV share orographic/stereo/seam validity bitmap (`@@`)
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
