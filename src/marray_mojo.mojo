"""General Mojo Python Extension & C-ABI Bytecode Module (`marray_mojo`).

Follows `mojo-python-interop` and `mojo-syntax` skills:
  1. Exports `PyInit_marray_mojo` via `PythonModuleBuilder` so Python can
     `import marray_mojo` directly as a native extension module.
  2. Exports `marray_exec_general_plan` (`abi("C")`) accepting arbitrary
     bytecode DAGs and N-D strided variable bindings with zero GIL coupling
     and zero domain/benchmark-specific logic.
"""

from std.memory import ImmPointer, MutPointer
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.runtime import initialize_runtime, parallelism_level
from std.sys import simd_width_of

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
    # Packed source descriptors: `n_sources` rows of:
    #   int64[11]: [data_addr, aux_addr, dtype_code, mask_kind, mask_param, s0, s1, s2, s3, s4, s5]
    #   float32[2]: [scale, offset]
    src_i64: ImmPointer[Int64, ImmUntrackedOrigin],
    src_f32: ImmPointer[Float32, ImmUntrackedOrigin],
    n_sources: Int64,
    # Packed DAG nodes (`n_nodes`):
    node_ops: ImmPointer[Int32, ImmUntrackedOrigin],
    node_a: ImmPointer[Int32, ImmUntrackedOrigin],
    node_b: ImmPointer[Int32, ImmUntrackedOrigin],
    node_c: ImmPointer[Int32, ImmUntrackedOrigin],
    node_p0: ImmPointer[Float32, ImmUntrackedOrigin],
    node_p1: ImmPointer[Float32, ImmUntrackedOrigin],
    n_nodes: Int64,
    # N-D iteration domain & slice window (`ndim <= 6`):
    ndim: Int64,
    full_extents: ImmPointer[Int64, ImmUntrackedOrigin],
    slice_starts: ImmPointer[Int64, ImmUntrackedOrigin],
    slice_stops: ImmPointer[Int64, ImmUntrackedOrigin],
    # Reduction specification:
    reduce_axes_mask: Int32,  # -1 = global stats only; bitmask >= 0 = reduce marked axes
    agg_mode: Int32,          # 0=mean, 1=sum, 2=min, 3=max, 4=std, 5=count, -1=materialize
    needs_global_std: Int32,
    # Output buffers:
    out_stats_f64: MutPointer[Float64, MutUntrackedOrigin],  # [mean, std, min, max, sum, valid_cnt, total_cnt, opt_insts, opt_regs]
    out_arr_f32: MutPointer[Float32, MutUntrackedOrigin],
    out_mask_u8: MutPointer[UInt8, MutUntrackedOrigin],
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
        out_stats_f64,
        out_arr_f32,
        out_mask_u8,
    )
    out_stats_f64[unsafe_offset=7] = Float64(len(plan.insts))
    out_stats_f64[unsafe_offset=8] = Float64(plan.num_regs)
    return Int32(0)


# ===----------------------------------------------------------------------=== #
# Native Python Extension Entrypoint (`mojo-python-interop` skill pattern)
# ===----------------------------------------------------------------------=== #


def py_engine_info() raises -> PythonObject:
    initialize_runtime()
    return Python.dict(
        simd_f32_width=PythonObject(simd_width_of[DType.float32]()),
        threads=PythonObject(parallelism_level()),
        morsel_elems=PythonObject(2048),
    )


def py_exec_from_numpy_buffers(
    src_i64_arr: PythonObject,
    src_f32_arr: PythonObject,
    dag_i32_arr: PythonObject,
    dag_f32_arr: PythonObject,
    domain_i64_arr: PythonObject,
    out_tuple: PythonObject,
) raises -> PythonObject:
    """`PythonModuleBuilder` trampoline accepting contiguous NumPy plan buffers."""
    var p_src_i64 = ImmPointer[Int64, ImmUntrackedOrigin](
        unsafe_from_address=Int(py=src_i64_arr.ctypes.data)
    )
    var p_src_f32 = ImmPointer[Float32, ImmUntrackedOrigin](
        unsafe_from_address=Int(py=src_f32_arr.ctypes.data)
    )
    var n_sources = Int64(Int(py=src_i64_arr.shape[0]))

    var n_nodes = Int(py=dag_i32_arr.shape[1])
    var p_dag_i32 = ImmPointer[Int32, ImmUntrackedOrigin](
        unsafe_from_address=Int(py=dag_i32_arr.ctypes.data)
    )
    var p_dag_f32 = ImmPointer[Float32, ImmUntrackedOrigin](
        unsafe_from_address=Int(py=dag_f32_arr.ctypes.data)
    )

    var p_dom = ImmPointer[Int64, ImmUntrackedOrigin](
        unsafe_from_address=Int(py=domain_i64_arr.ctypes.data)
    )
    var ndim = p_dom[unsafe_offset=0]
    var red_mask = Int32(p_dom[unsafe_offset=1])
    var agg_mode = Int32(p_dom[unsafe_offset=2])
    var need_std = Int32(p_dom[unsafe_offset=3])
    var full_ext = p_dom.unsafe_offset(4)
    var sl_starts = p_dom.unsafe_offset(10)
    var sl_stops = p_dom.unsafe_offset(16)

    var p_stats = MutPointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_tuple[0].ctypes.data)
    )
    var p_arr = MutPointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_tuple[1].ctypes.data)
    )
    var p_msk = MutPointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_tuple[2].ctypes.data)
    )

    _ = marray_exec_general_plan(
        p_src_i64,
        p_src_f32,
        n_sources,
        p_dag_i32,
        p_dag_i32.unsafe_offset(n_nodes),
        p_dag_i32.unsafe_offset(n_nodes * 2),
        p_dag_i32.unsafe_offset(n_nodes * 3),
        p_dag_f32,
        p_dag_f32.unsafe_offset(n_nodes),
        Int64(n_nodes),
        ndim,
        full_ext,
        sl_starts,
        sl_stops,
        red_mask,
        agg_mode,
        need_std,
        p_stats,
        p_arr,
        p_msk,
    )
    return out_tuple[0]


@export
def PyInit_marray_mojo() abi("C") -> PythonObject:
    try:
        initialize_runtime()
        var m = PythonModuleBuilder("marray_mojo")
        m.def_function[py_engine_info]("engine_info")
        m.def_function[py_exec_from_numpy_buffers]("exec_plan")
        return m.finalize()
    except e:
        abort(t"failed to initialize marray_mojo: {e}")
