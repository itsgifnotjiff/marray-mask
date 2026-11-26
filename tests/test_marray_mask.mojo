"""Unit tests for `marray-mask` using `std.testing.TestSuite` (`mojo-syntax` skill)."""

from std.testing import TestSuite, assert_almost_equal, assert_equal, assert_true

from marray import (
    MASK_ALL_VALID,
    MASK_BITMASK,
    Dataset,
    GDPSGridSpec,
    MaskedSlab,
    decode_float32_bitmap_into,
    decode_quant16_sentinel_into,
    decode_uint16_qflag_into,
    hypot,
    slab_reduce_stats,
)
from marray.morsel import (
    MSRC_SENTINEL_INT,
    MSTATE_BITS,
    MSTATE_VALID,
    MorselReg,
    ReduceAcc,
    load_span_u16,
    morsel_accumulate_stats,
    morsel_hypot,
)
from marray.vm import (
    BC_ADD,
    BC_AFFINE,
    BC_HYPOT,
    BC_LOAD,
    BC_MUL,
    BC_SQRT,
    SourceBinding,
    compile_and_optimize_dag,
)
from std.memory import ImmPointer


def test_fortran_layout_and_subslab_slicing() raises:
    var s = MaskedSlab(5, 4)
    for j in range(4):
        for i in range(5):
            s.set(i, j, Float32(i + j * 10))
    assert_equal(s.data[1], Float32(1.0))
    assert_equal(s.data[5], Float32(10.0))

    s.mask_cell(2, 1)
    assert_equal(s.mask_state, MASK_BITMASK)
    assert_equal(s.valid_count(), 19)

    var sub_masked = s.slice_fortran(1, 4, 1, 3)
    assert_true(sub_masked.is_masked(1, 0))
    assert_almost_equal(Float64(sub_masked.get(2, 1)), 23.0, atol=1e-5)

    var sub_clean = s.slice_fortran(0, 2, 2, 4)
    assert_equal(sub_clean.mask_state, MASK_ALL_VALID)


def test_three_binary_encodings_and_propagation() raises:
    var ni = 20
    var nj = 5  # 100 elements (prime-like boundary across 2 UInt64 mask words!)
    var raw_u16 = List[UInt16](capacity=104)
    raw_u16.resize(104, UInt16(100))
    raw_u16[7] = UInt16(0xFFFF)
    raw_u16[70] = UInt16(0xFFFF)
    var s_sent = MaskedSlab()
    decode_quant16_sentinel_into(s_sent, raw_u16, ni, nj, 0.5, -10.0)
    assert_equal(s_sent.valid_count(), 98)

    var raw_f32 = List[Float32](capacity=104)
    raw_f32.resize(104, Float32(10.0))
    var raw_bm = List[UInt64](capacity=2)
    raw_bm.resize(2, UInt64(0))
    raw_bm[70 >> 6] |= UInt64(1) << UInt64(70 & 63)
    raw_bm[85 >> 6] |= UInt64(1) << UInt64(85 & 63)
    var s_bm = MaskedSlab()
    decode_float32_bitmap_into(s_bm, raw_f32, raw_bm, ni, nj)

    var raw_qf = List[UInt8](capacity=104)
    raw_qf.resize(104, UInt8(0))
    raw_qf[12] = UInt8(3)
    var raw_hu = List[UInt16](capacity=104)
    raw_hu.resize(104, UInt16(5))
    var s_qc = MaskedSlab()
    decode_uint16_qflag_into(s_qc, raw_hu, raw_qf, ni, nj, 1.0, 0.0)

    var combined = (s_sent - s_bm) * s_qc
    var st = slab_reduce_stats(combined)
    assert_equal(st.valid_count, 96)
    assert_almost_equal(st.mean(), 150.0, atol=1e-4)


def test_morsel_single_pass_decode_and_optimizer_fusion() raises:
    # Tile-straddling length 65 (`idiomatic-mojo` §13: test 63/64/65 tail logic!)
    var n = 65
    var u16_in = List[UInt16](capacity=72)
    u16_in.resize(72, UInt16(200))
    u16_in[0] = UInt16(0xFFFF)
    u16_in[63] = UInt16(0xFFFF)
    u16_in[64] = UInt16(0xFFFF)
    var dummy_u8 = List[UInt8](capacity=72)
    dummy_u8.resize(72, UInt8(0))

    var reg_a = MorselReg()
    var or_bits: UInt64 = 0
    load_span_u16(
        ImmPointer[UInt16, origin_of(u16_in)](u16_in.unsafe_ptr()),
        ImmPointer[UInt8, origin_of(dummy_u8)](dummy_u8.unsafe_ptr()),
        0,
        n,
        0.5,
        0.0,
        MSRC_SENTINEL_INT,
        0xFFFF,
        reg_a,
        or_bits,
    )
    reg_a.finalize_words(n, or_bits)
    assert_equal(reg_a.state, MSTATE_BITS)

    var reg_h = MorselReg()
    morsel_hypot(reg_h, reg_a, reg_a, n)
    var acc = ReduceAcc()
    morsel_accumulate_stats[True](reg_h, n, acc)
    assert_equal(acc.valid_count, 62)  # 65 - 3 sentinels = 62!
    assert_almost_equal(acc.mean(), 100.0 * 1.41421356, atol=1e-3)

    # Verify algebraic optimizer fuses `sqrt(u*u + v*v)` -> `BC_HYPOT` AND folds
    # `AFFINE(LOAD(0), 2.0, 10.0)` into a single `BC_LOAD`!
    var srcs: List[SourceBinding] = [
        SourceBinding(0, 0, 2, 0, 0, 0.01, -120.0, 1, 10, 100, 1000, 0, 0),
        SourceBinding(0, 0, 2, 0, 0, 0.01, -120.0, 1, 10, 100, 1000, 0, 0),
    ]
    var ops: List[Int32] = [
        Int32(BC_LOAD),
        Int32(BC_LOAD),
        Int32(BC_MUL),
        Int32(BC_MUL),
        Int32(BC_ADD),
        Int32(BC_SQRT),
        Int32(BC_AFFINE),
    ]
    var na: List[Int32] = [0, 1, 0, 1, 2, 4, 5]
    var nb: List[Int32] = [-1, -1, 0, 1, 3, -1, -1]
    var nc: List[Int32] = [-1, -1, -1, -1, -1, -1, -1]
    var p0: List[Float32] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.514444]
    var p1: List[Float32] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    var plan = compile_and_optimize_dag(
        ImmPointer[Int32, origin_of(ops)](ops.unsafe_ptr()),
        ImmPointer[Int32, origin_of(na)](na.unsafe_ptr()),
        ImmPointer[Int32, origin_of(nb)](nb.unsafe_ptr()),
        ImmPointer[Int32, origin_of(nc)](nc.unsafe_ptr()),
        ImmPointer[Float32, origin_of(p0)](p0.unsafe_ptr()),
        ImmPointer[Float32, origin_of(p1)](p1.unsafe_ptr()),
        len(ops),
        srcs,
    )
    # 7 naive nodes (`u, v, u*u, v*v, add, sqrt, scale`) collapse to 4 insts
    # (`LOAD u, LOAD v, HYPOT(u,v), AFFINE`) in just 3 non-aliasing L1 registers!
    assert_equal(len(plan.insts), 4)
    assert_equal(plan.insts[2].op, BC_HYPOT)
    assert_equal(plan.num_regs, 3)


def test_dataset_sel_descending_levels() raises:
    var grid = GDPSGridSpec.standard_gdps_15km(n_runs=10)
    var ds = Dataset.open_virtual_gdps(grid)
    var sub = ds.sel(
        time_min=6.0,
        time_max=24.0,
        level_min=500.0,
        level_max=1000.0,
        lon_min=280.0,
        lon_max=305.0,
        lat_min=45.0,
        lat_max=55.0,
        max_runs=1,
    )
    assert_equal(sub.window.n_steps(), 7)
    assert_equal(sub.window.n_levels(), 14)
    var wspd = hypot(sub["UU"], sub["VV"]).scale_offset(0.514444, 0.0, "m/s")
    var rep = wspd.compute_stats()
    assert_equal(rep.active_slabs, 98)
    assert_true(rep.stats.valid_count < rep.stats.total_count)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
