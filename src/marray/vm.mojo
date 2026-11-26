"""General N-D Bytecode Virtual Machine, Algebraic Graph Optimizer, Multi-Sink
Fused Executor, `RecordCatalog` Streamer, and `Float64` Parity Engine (`vm.mojo`).

Upgraded with all Boreal / Aether production contracts (`docs/questions.md` Q1-Q15):
  - `RecordCatalog` whole-record plane streaming (`CatalogSourceDesc` + per-worker
    `PlaneSlotCache`, `datyp=6` serial safety guard, `float32` `@@` bitmap planes).
  - Multi-sink / multi-reduction single-pass DAG execution (`compile_and_optimize_multi_dag`
    + `execute_multi_sink_plan`: evaluates `globavg + zonavg + timavg` or batched
    `aether` `(n_fields, n_src)` finite buffers in ONE scan over the inputs).
  - Spatial-partitioned high-rank axis reductions (eliminates `n_workers * out_cells`
    buffer replication on `timavg` / large spatial maps).
  - Exact `Float64` morsel register path (`execute_bytecode_plan_f64`) for `boreal-stat`
    `values_f64` (`r.diag`, `ke`, `stdev`, pair `RMSE`) parity gates without Python
    4D `float64` cube copies.
"""

from std.math import ceildiv, max, min, sqrt
from std.memory import ImmPointer, MutPointer
from std.runtime import parallelism_level
from max.algorithm import parallelize

from .catalog import (
    CatalogSourceDesc,
    PlaneSlotCache,
    load_morsel_from_catalog,
)
from .morsel import (
    DTYPE_F32,
    DTYPE_F64,
    DTYPE_U16,
    MORSEL_LEN,
    MSRC_FST_BITMAP_F32,
    MSRC_NONE,
    MSTATE_MASKED,
    MSTATE_VALID,
    MorselReg,
    MorselReg64,
    ReduceAcc,
    load_span_f32,
    load_span_into_reg64,
    load_span_u16,
    morsel64_accumulate_stats,
    morsel64_accumulate_weighted,
    morsel64_affine,
    morsel64_binary,
    morsel64_fma,
    morsel64_store_filled,
    morsel64_unary,
    morsel_accumulate_stats,
    morsel_accumulate_weighted,
    morsel_add,
    morsel_affine,
    morsel_div,
    morsel_fillna,
    morsel_fma,
    morsel_hypot,
    morsel_mask_cmp,
    morsel_max,
    morsel_min,
    morsel_mul,
    morsel_pow2_sum,
    morsel_store_filled,
    morsel_sub,
    morsel_unary_math,
)


comptime MAX_DIMS: Int = 6

# General Bytecode VM Opcodes:
comptime BC_LOAD: Int = 1
comptime BC_CONST: Int = 2
comptime BC_ADD: Int = 3
comptime BC_SUB: Int = 4
comptime BC_MUL: Int = 5
comptime BC_DIV: Int = 6
comptime BC_HYPOT: Int = 7
comptime BC_POW2_SUM: Int = 8
comptime BC_MIN: Int = 9
comptime BC_MAX: Int = 10
comptime BC_AFFINE: Int = 11
comptime BC_FMA: Int = 12
comptime BC_NEG: Int = 13
comptime BC_ABS: Int = 14
comptime BC_SQRT: Int = 15
comptime BC_EXP: Int = 16
comptime BC_LOG: Int = 17
comptime BC_SIN: Int = 18
comptime BC_COS: Int = 19
comptime BC_MASK_LT: Int = 20
comptime BC_MASK_LE: Int = 21
comptime BC_MASK_GT: Int = 22
comptime BC_MASK_GE: Int = 23
comptime BC_FILLNA: Int = 24

# Axis reduction aggregation modes:
comptime AGG_MEAN: Int = 0
comptime AGG_SUM: Int = 1
comptime AGG_MIN: Int = 2
comptime AGG_MAX: Int = 3
comptime AGG_STD: Int = 4
comptime AGG_COUNT: Int = 5


@fieldwise_init
struct BytecodeInst(Copyable, Movable):
    var op: Int
    var dst_reg: Int
    var r0: Int
    var r1: Int
    var r2: Int
    var src_id: Int
    var p0: Float32
    var p1: Float32


struct SourceBinding(Copyable, Movable):
    """Describes an N-D input buffer OR a `RecordCatalog` whole-record plane source."""

    var data_addr: Int
    var aux_addr: Int
    var dtype_code: Int
    var mask_kind: Int
    var mask_param: Int
    var scale: Float32
    var offset: Float32
    var s0: Int
    var s1: Int
    var s2: Int
    var s3: Int
    var s4: Int
    var s5: Int
    var catalog_id: Int  # >= 0 when backed by `CatalogSourceDesc` whole-record decoder!

    def __init__(
        out self,
        data_addr: Int,
        aux_addr: Int,
        dtype_code: Int,
        mask_kind: Int,
        mask_param: Int,
        scale: Float32,
        offset: Float32,
        s0: Int,
        s1: Int,
        s2: Int,
        s3: Int,
        s4: Int,
        s5: Int,
        catalog_id: Int = -1,
    ):
        self.data_addr = data_addr
        self.aux_addr = aux_addr
        self.dtype_code = dtype_code
        self.mask_kind = mask_kind
        self.mask_param = mask_param
        self.scale = scale
        self.offset = offset
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self.s5 = s5
        self.catalog_id = catalog_id

    @inline(.always)
    def stride(self, d: Int) -> Int:
        match d:
            case 0:
                return self.s0
            case 1:
                return self.s1
            case 2:
                return self.s2
            case 3:
                return self.s3
            case 4:
                return self.s4
            case _:
                return self.s5


@inline(.always)
def _is_source_coalesced(
    sb: SourceBinding,
    f0: Int,
    f1: Int,
    f2: Int,
    full_d0: Bool,
    full_d1: Bool,
    full_d2: Bool,
    d1_len: Int,
    d2_len: Int,
    d3_len: Int,
) -> Bool:
    if sb.catalog_id >= 0 or sb.dtype_code == DTYPE_F64:
        return False
    if not full_d0 or sb.s0 != 1:
        return False
    if d1_len > 1 and (not full_d1 or sb.s1 != f0):
        return False
    if d2_len > 1 and (not full_d2 or sb.s2 != f0 * f1):
        return False
    if d3_len > 1 and sb.s3 != f0 * f1 * f2:
        return False
    return True


struct CompiledPlan(Copyable, Movable):
    var insts: List[BytecodeInst]
    var root_reg: Int
    var sink_regs: List[Int]
    var num_regs: Int
    var needs_sum_sq: Bool

    def __init__(out self):
        self.insts = List[BytecodeInst]()
        self.root_reg = 0
        self.sink_regs = List[Int]()
        self.num_regs = 1
        self.needs_sum_sq = False


def compile_and_optimize_multi_dag(
    raw_ops: ImmPointer[Int32, _],
    raw_a: ImmPointer[Int32, _],
    raw_b: ImmPointer[Int32, _],
    raw_c: ImmPointer[Int32, _],
    raw_p0: ImmPointer[Float32, _],
    raw_p1: ImmPointer[Float32, _],
    n_nodes: Int,
    root_nodes: List[Int],
    mut sources: List[SourceBinding],
    mut cat_descs: List[CatalogSourceDesc],
    needs_sum_sq: Bool = False,
) -> CompiledPlan:
    """General 3-pass multi-sink DAG compiler (`1..K` sink roots sharing CSE nodes)."""
    var ops = List[Int](capacity=n_nodes)
    var a_idx = List[Int](capacity=n_nodes)
    var b_idx = List[Int](capacity=n_nodes)
    var c_idx = List[Int](capacity=n_nodes)
    var p0 = List[Float32](capacity=n_nodes)
    var p1 = List[Float32](capacity=n_nodes)
    var ref_cnt = List[Int](capacity=n_nodes)

    for i in range(n_nodes):
        ops.append(Int(raw_ops[unsafe_offset=i]))
        a_idx.append(Int(raw_a[unsafe_offset=i]))
        b_idx.append(Int(raw_b[unsafe_offset=i]))
        c_idx.append(Int(raw_c[unsafe_offset=i]))
        p0.append(raw_p0[unsafe_offset=i])
        p1.append(raw_p1[unsafe_offset=i])
        ref_cnt.append(0)

    # Normalize scalar constants into `BC_AFFINE`
    for i in range(n_nodes):
        var op = ops[i]
        var lhs = a_idx[i]
        var rhs = b_idx[i]
        if op == BC_ADD and rhs >= 0 and ops[rhs] == BC_CONST:
            ops[i] = BC_AFFINE
            b_idx[i] = -1
            p0[i] = 1.0
            p1[i] = p0[rhs]
        elif op == BC_ADD and lhs >= 0 and ops[lhs] == BC_CONST:
            ops[i] = BC_AFFINE
            a_idx[i] = rhs
            b_idx[i] = -1
            p0[i] = 1.0
            p1[i] = p0[lhs]
        elif op == BC_SUB and rhs >= 0 and ops[rhs] == BC_CONST:
            ops[i] = BC_AFFINE
            b_idx[i] = -1
            p0[i] = 1.0
            p1[i] = -p0[rhs]
        elif op == BC_MUL and rhs >= 0 and ops[rhs] == BC_CONST:
            ops[i] = BC_AFFINE
            b_idx[i] = -1
            p0[i] = p0[rhs]
            p1[i] = 0.0
        elif op == BC_MUL and lhs >= 0 and ops[lhs] == BC_CONST:
            ops[i] = BC_AFFINE
            a_idx[i] = rhs
            b_idx[i] = -1
            p0[i] = p0[lhs]
            p1[i] = 0.0
        elif op == BC_DIV and rhs >= 0 and ops[rhs] == BC_CONST:
            var denom = p0[rhs]
            if denom < -1.0e-12 or denom > 1.0e-12:
                ops[i] = BC_AFFINE
                b_idx[i] = -1
                p0[i] = Float32(1.0) / denom
                p1[i] = 0.0

    for i in range(n_nodes):
        if a_idx[i] >= 0 and ops[i] != BC_LOAD:
            ref_cnt[a_idx[i]] += 1
        if b_idx[i] >= 0:
            ref_cnt[b_idx[i]] += 1
        if c_idx[i] >= 0:
            ref_cnt[c_idx[i]] += 1
    for r in root_nodes:
        if r >= 0 and r < n_nodes:
            ref_cnt[r] += 1

    # Peephole algebraic fusion pass
    for i in range(n_nodes):
        var op = ops[i]
        var lhs = a_idx[i]
        var rhs = b_idx[i]

        if op == BC_AFFINE and lhs >= 0 and ops[lhs] == BC_AFFINE:
            var s0 = p0[lhs]
            var o0 = p1[lhs]
            var s1 = p0[i]
            var o1 = p1[i]
            a_idx[i] = a_idx[lhs]
            p0[i] = s0 * s1
            p1[i] = o0 * s1 + o1
            lhs = a_idx[i]

        if op == BC_AFFINE and lhs >= 0 and ops[lhs] == BC_LOAD:
            var orig_sid = a_idx[lhs]
            var sb = sources[orig_sid].copy()
            sb.scale = sb.scale * p0[i]
            sb.offset = sb.offset * p0[i] + p1[i]
            if sb.catalog_id >= 0:
                var cd = cat_descs[sb.catalog_id].copy()
                cd.scale = cd.scale * p0[i]
                cd.offset = cd.offset * p0[i] + p1[i]
                var new_cid = len(cat_descs)
                cat_descs.append(cd^)
                sb.catalog_id = new_cid
            var new_sid = len(sources)
            sources.append(sb^)
            ops[i] = BC_LOAD
            a_idx[i] = new_sid
            b_idx[i] = -1

        elif op == BC_ADD and lhs >= 0 and rhs >= 0:
            if (
                ops[lhs] == BC_MUL
                and ops[rhs] == BC_MUL
                and a_idx[lhs] == b_idx[lhs]
                and a_idx[rhs] == b_idx[rhs]
            ):
                ops[i] = BC_POW2_SUM
                a_idx[i] = a_idx[lhs]
                b_idx[i] = a_idx[rhs]
            elif ops[lhs] == BC_MUL and ref_cnt[lhs] == 1:
                ops[i] = BC_FMA
                c_idx[i] = rhs
                b_idx[i] = b_idx[lhs]
                a_idx[i] = a_idx[lhs]
            elif ops[rhs] == BC_MUL and ref_cnt[rhs] == 1:
                ops[i] = BC_FMA
                c_idx[i] = lhs
                a_idx[i] = a_idx[rhs]
                b_idx[i] = b_idx[rhs]

        elif op == BC_SQRT and lhs >= 0 and ops[lhs] == BC_POW2_SUM:
            ops[i] = BC_HYPOT
            a_idx[i] = a_idx[lhs]
            b_idx[i] = b_idx[lhs]

    # Mark live nodes backward from ALL sink roots
    var live = List[Bool](capacity=n_nodes)
    live.resize(n_nodes, False)
    for r in root_nodes:
        if r >= 0 and r < n_nodes:
            live[r] = True
    for rev in range(n_nodes):
        var i = n_nodes - 1 - rev
        if not live[i]:
            continue
        if ops[i] != BC_LOAD and a_idx[i] >= 0:
            live[a_idx[i]] = True
        if b_idx[i] >= 0:
            live[b_idx[i]] = True
        if c_idx[i] >= 0:
            live[c_idx[i]] = True

    var last_use = List[Int](capacity=n_nodes)
    last_use.resize(n_nodes, -1)
    for i in range(n_nodes):
        if not live[i]:
            continue
        if ops[i] != BC_LOAD and a_idx[i] >= 0:
            last_use[a_idx[i]] = i
        if b_idx[i] >= 0:
            last_use[b_idx[i]] = i
        if c_idx[i] >= 0:
            last_use[c_idx[i]] = i
    # Keep all sink root registers pinned until after bytecode execution!
    for r in root_nodes:
        if r >= 0 and r < n_nodes:
            last_use[r] = n_nodes

    var node_reg = List[Int](capacity=n_nodes)
    node_reg.resize(n_nodes, -1)
    var free_regs = List[Int]()
    var next_reg = 0
    var plan = CompiledPlan()
    plan.needs_sum_sq = needs_sum_sq

    for i in range(n_nodes):
        if not live[i]:
            continue
        var r_in0 = node_reg[a_idx[i]] if (ops[i] != BC_LOAD and a_idx[i] >= 0) else -1
        var r_in1 = node_reg[b_idx[i]] if b_idx[i] >= 0 else -1
        var r_in2 = node_reg[c_idx[i]] if c_idx[i] >= 0 else -1

        var chosen = -1
        for f_i in range(len(free_regs)):
            var cand = free_regs[f_i]
            if cand != r_in0 and cand != r_in1 and cand != r_in2:
                chosen = cand
                _ = free_regs.pop(f_i)
                break
        if chosen < 0:
            chosen = next_reg
            next_reg += 1
        node_reg[i] = chosen

        var sid = a_idx[i] if ops[i] == BC_LOAD else -1
        plan.insts.append(
            BytecodeInst(ops[i], chosen, r_in0, r_in1, r_in2, sid, p0[i], p1[i])
        )

        if ops[i] != BC_LOAD and a_idx[i] >= 0 and last_use[a_idx[i]] == i:
            free_regs.append(r_in0)
        if b_idx[i] >= 0 and b_idx[i] != a_idx[i] and last_use[b_idx[i]] == i:
            free_regs.append(r_in1)
        if (
            c_idx[i] >= 0
            and c_idx[i] != a_idx[i]
            and c_idx[i] != b_idx[i]
            and last_use[c_idx[i]] == i
        ):
            free_regs.append(r_in2)

    for r in root_nodes:
        var reg_r = node_reg[r] if (r >= 0 and r < n_nodes) else 0
        plan.sink_regs.append(reg_r)
    plan.root_reg = plan.sink_regs[0] if len(plan.sink_regs) > 0 else 0
    plan.num_regs = max(1, next_reg)
    return plan^


def compile_and_optimize_dag(
    raw_ops: ImmPointer[Int32, _],
    raw_a: ImmPointer[Int32, _],
    raw_b: ImmPointer[Int32, _],
    raw_c: ImmPointer[Int32, _],
    raw_p0: ImmPointer[Float32, _],
    raw_p1: ImmPointer[Float32, _],
    n_nodes: Int,
    mut sources: List[SourceBinding],
    needs_sum_sq: Bool = False,
) -> CompiledPlan:
    var roots = List[Int]()
    if n_nodes > 0:
        roots.append(n_nodes - 1)
    var empty_cats = List[CatalogSourceDesc]()
    return compile_and_optimize_multi_dag(
        raw_ops,
        raw_a,
        raw_b,
        raw_c,
        raw_p0,
        raw_p1,
        n_nodes,
        roots,
        sources,
        empty_cats,
        needs_sum_sq,
    )


struct MorselWorkspace(Movable):
    """Thread-local L1-cache register file + 1 disjoint scratch + 1 weight reg + catalog plane slots."""

    var regs: List[MorselReg]
    var scratch: MorselReg
    var wt_reg: MorselReg
    var cat_slots: List[PlaneSlotCache]

    def __init__(
        out self,
        num_regs: Int,
        cat_descs_ptr: ImmPointer[CatalogSourceDesc, _],
        n_cats: Int,
    ):
        self.regs = List[MorselReg](capacity=num_regs)
        for _ in range(num_regs):
            self.regs.append(MorselReg())
        self.scratch = MorselReg()
        self.wt_reg = MorselReg()
        self.cat_slots = List[PlaneSlotCache](capacity=n_cats)
        for c in range(n_cats):
            ref cd = cat_descs_ptr[unsafe_offset=c]
            self.cat_slots.append(PlaneSlotCache(cd.plane_ni * cd.plane_nj))

    def __init__(out self, *, deinit move: Self):
        self.regs = move.regs^
        self.scratch = move.scratch^
        self.wt_reg = move.wt_reg^
        self.cat_slots = move.cat_slots^


@inline(.always)
def _load_morsel_from_source(
    src: SourceBinding,
    cat_descs_ptr: ImmPointer[CatalogSourceDesc, _],
    ndim: Int,
    d0_start: Int,
    d0_len: Int,
    d1_start: Int,
    d1_len: Int,
    d2_start: Int,
    d2_len: Int,
    d3_start: Int,
    d3_len: Int,
    flat_offset: Int,
    count: Int,
    coalesced_all: Bool,
    mut ws_cats: List[PlaneSlotCache],
    mut dst: MorselReg,
):
    if src.catalog_id >= 0:
        load_morsel_from_catalog(
            cat_descs_ptr[unsafe_offset=src.catalog_id],
            ws_cats[src.catalog_id],
            d0_start,
            d0_len,
            d1_start,
            d1_len,
            d2_start,
            d2_len,
            d3_start,
            flat_offset,
            count,
            dst,
        )
        return

    var mp = dst.mask_ptr()
    var n_words = ceildiv(count, 64)
    if src.mask_kind != MSRC_NONE:
        for w in range(n_words):
            mp[unsafe_offset=w] = UInt64(0)
    var or_bits: UInt64 = 0
    var aux_b = 4 if src.mask_kind == MSRC_FST_BITMAP_F32 else 1

    # Fast path 1: Full-span contiguous coalesced 1D stream!
    if coalesced_all and src.s0 == 1:
        var base_elem = (
            d0_start * src.s0
            + d1_start * src.s1
            + d2_start * src.s2
            + d3_start * src.s3
            + flat_offset
        )
        var p_aux = ImmPointer[UInt8, ImmUntrackedOrigin](
            unsafe_from_address=src.aux_addr + base_elem * aux_b
            if src.aux_addr != 0
            else src.data_addr
        )
        if src.dtype_code == DTYPE_U16:
            var p_u16 = ImmPointer[UInt16, ImmUntrackedOrigin](
                unsafe_from_address=src.data_addr + base_elem * 2
            )
            load_span_u16(
                p_u16,
                p_aux,
                0,
                count,
                src.scale,
                src.offset,
                src.mask_kind,
                src.mask_param,
                dst,
                or_bits,
            )
        else:
            var p_f32 = ImmPointer[Float32, ImmUntrackedOrigin](
                unsafe_from_address=src.data_addr + base_elem * 4
            )
            load_span_f32(
                p_f32,
                p_aux,
                0,
                count,
                src.scale,
                src.offset,
                src.mask_kind,
                src.mask_param,
                dst,
                or_bits,
            )
        dst.finalize_words(count, or_bits)
        return

    var written = 0
    while written < count:
        var cur = flat_offset + written
        var rem0, i_rel = divmod(cur, d0_len)
        var rem1, j_rel = divmod(rem0, d1_len)
        var t_rel, z_rel = divmod(rem1, d2_len)

        var span = min(d0_len - i_rel, count - written)
        var elem_off = (
            (d0_start + i_rel) * src.s0
            + (d1_start + j_rel) * src.s1
            + (d2_start + z_rel) * src.s2
            + (d3_start + t_rel) * src.s3
        )

        if src.s0 == 1 and src.dtype_code != DTYPE_F64:
            var p_aux = ImmPointer[UInt8, ImmUntrackedOrigin](
                unsafe_from_address=src.aux_addr + elem_off * aux_b
                if src.aux_addr != 0
                else src.data_addr
            )
            if src.dtype_code == DTYPE_U16:
                var p_u16 = ImmPointer[UInt16, ImmUntrackedOrigin](
                    unsafe_from_address=src.data_addr + elem_off * 2
                )
                load_span_u16(
                    p_u16,
                    p_aux,
                    written,
                    span,
                    src.scale,
                    src.offset,
                    src.mask_kind,
                    src.mask_param,
                    dst,
                    or_bits,
                )
            else:
                var p_f32 = ImmPointer[Float32, ImmUntrackedOrigin](
                    unsafe_from_address=src.data_addr + elem_off * 4
                )
                load_span_f32(
                    p_f32,
                    p_aux,
                    written,
                    span,
                    src.scale,
                    src.offset,
                    src.mask_kind,
                    src.mask_param,
                    dst,
                    or_bits,
                )
        elif src.s0 == 1 and src.dtype_code == DTYPE_F64:
            var dp = dst.data_ptr()
            var p_f64 = ImmPointer[Float64, ImmUntrackedOrigin](
                unsafe_from_address=src.data_addr + elem_off * 8
            )
            for s_i in range(span):
                var d_idx = written + s_i
                var rv = Float32(p_f64[unsafe_offset=s_i]) * src.scale + src.offset
                var bad = False
                if src.aux_addr != 0 and src.mask_kind != MSRC_NONE:
                    if src.mask_kind == MSRC_FST_BITMAP_F32:
                        var bf = ImmPointer[Float32, ImmUntrackedOrigin](
                            unsafe_from_address=src.aux_addr + (elem_off + s_i) * 4
                        )[unsafe_offset=0]
                        bad = not (bf > Float32(0.0))
                    elif src.mask_kind == 6:
                        bad = (
                            ImmPointer[UInt8, ImmUntrackedOrigin](
                                unsafe_from_address=src.aux_addr + elem_off + s_i
                            )[unsafe_offset=0]
                            == UInt8(0)
                        )
                if bad:
                    dp[unsafe_offset=d_idx] = 0.0
                    mp[unsafe_offset=d_idx >> 6] |= UInt64(1) << UInt64(d_idx & 63)
                    or_bits |= UInt64(1)
                else:
                    dp[unsafe_offset=d_idx] = rv
        elif src.s0 == 0:
            var dp = dst.data_ptr()
            var bval: Float32
            var bmasked = False
            if src.dtype_code == DTYPE_U16:
                var p_u16 = ImmPointer[UInt16, ImmUntrackedOrigin](
                    unsafe_from_address=src.data_addr + elem_off * 2
                )
                var raw_q = p_u16[unsafe_offset=0]
                if src.mask_kind == 1 and Int(raw_q) == src.mask_param:
                    bval = 0.0
                    bmasked = True
                else:
                    bval = Float32(raw_q) * src.scale + src.offset
            elif src.dtype_code == DTYPE_F64:
                var p_f64 = ImmPointer[Float64, ImmUntrackedOrigin](
                    unsafe_from_address=src.data_addr + elem_off * 8
                )
                bval = Float32(p_f64[unsafe_offset=0]) * src.scale + src.offset
            else:
                var p_f32 = ImmPointer[Float32, ImmUntrackedOrigin](
                    unsafe_from_address=src.data_addr + elem_off * 4
                )
                bval = p_f32[unsafe_offset=0] * src.scale + src.offset
                if src.aux_addr != 0 and src.mask_kind != MSRC_NONE:
                    if src.mask_kind == MSRC_FST_BITMAP_F32:
                        var bf = ImmPointer[Float32, ImmUntrackedOrigin](
                            unsafe_from_address=src.aux_addr + elem_off * 4
                        )[unsafe_offset=0]
                        bmasked = not (bf > Float32(0.0))
                    elif src.mask_kind == 6:
                        bmasked = (
                            ImmPointer[UInt8, ImmUntrackedOrigin](
                                unsafe_from_address=src.aux_addr + elem_off
                            )[unsafe_offset=0]
                            == UInt8(0)
                        )
                    else:
                        bmasked = (
                            Int(
                                ImmPointer[UInt8, ImmUntrackedOrigin](
                                    unsafe_from_address=src.aux_addr + elem_off
                                )[unsafe_offset=0]
                            )
                            != src.mask_param
                        )
            for s_i in range(span):
                var d_idx = written + s_i
                dp[unsafe_offset=d_idx] = bval
                if bmasked:
                    mp[unsafe_offset=d_idx >> 6] |= UInt64(1) << UInt64(d_idx & 63)
                    or_bits |= UInt64(1)
        else:
            var dp = dst.data_ptr()
            var p_f32 = ImmPointer[Float32, ImmUntrackedOrigin](
                unsafe_from_address=src.data_addr
            )
            for s_i in range(span):
                var idx_s = elem_off + s_i * src.s0
                dp[unsafe_offset=written + s_i] = (
                    p_f32[unsafe_offset=idx_s] * src.scale + src.offset
                )

        written += span

    dst.finalize_words(count, or_bits)


@inline(.always)
def eval_morsel_bytecode(
    insts_ptr: ImmPointer[BytecodeInst, _],
    n_insts: Int,
    srcs_ptr: ImmPointer[SourceBinding, _],
    cat_descs_ptr: ImmPointer[CatalogSourceDesc, _],
    ndim: Int,
    d0_start: Int,
    d0_len: Int,
    d1_start: Int,
    d1_len: Int,
    d2_start: Int,
    d2_len: Int,
    d3_start: Int,
    d3_len: Int,
    flat_offset: Int,
    count: Int,
    coal_ptr: ImmPointer[Bool, _],
    mut ws: MorselWorkspace,
):
    for pc in range(n_insts):
        ref inst = insts_ptr[unsafe_offset=pc]
        match inst.op:
            case == BC_LOAD:
                _load_morsel_from_source(
                    srcs_ptr[unsafe_offset=inst.src_id],
                    cat_descs_ptr,
                    ndim,
                    d0_start,
                    d0_len,
                    d1_start,
                    d1_len,
                    d2_start,
                    d2_len,
                    d3_start,
                    d3_len,
                    flat_offset,
                    count,
                    coal_ptr[unsafe_offset=inst.src_id],
                    ws.cat_slots,
                    ws.regs[inst.dst_reg],
                )
            case == BC_CONST:
                ws.regs[inst.dst_reg].state = MSTATE_VALID
                var dp = ws.regs[inst.dst_reg].data_ptr()
                for k in range(count):
                    dp[unsafe_offset=k] = inst.p0
            case == BC_ADD:
                morsel_add(ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_SUB:
                morsel_sub(ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_MUL:
                morsel_mul(ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_DIV:
                morsel_div(ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_HYPOT:
                morsel_hypot(ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_POW2_SUM:
                morsel_pow2_sum(ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_MIN:
                morsel_min(ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_MAX:
                morsel_max(ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_AFFINE:
                morsel_affine(
                    ws.scratch, ws.regs[inst.r0], inst.p0, inst.p1, count
                )
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_FMA:
                morsel_fma(
                    ws.scratch,
                    ws.regs[inst.r0],
                    ws.regs[inst.r1],
                    ws.regs[inst.r2],
                    count,
                )
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_NEG:
                morsel_unary_math[0](ws.scratch, ws.regs[inst.r0], count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_ABS:
                morsel_unary_math[1](ws.scratch, ws.regs[inst.r0], count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_SQRT:
                morsel_unary_math[2](ws.scratch, ws.regs[inst.r0], count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_EXP:
                morsel_unary_math[3](ws.scratch, ws.regs[inst.r0], count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_LOG:
                morsel_unary_math[4](ws.scratch, ws.regs[inst.r0], count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_SIN:
                morsel_unary_math[5](ws.scratch, ws.regs[inst.r0], count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_COS:
                morsel_unary_math[6](ws.scratch, ws.regs[inst.r0], count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_MASK_LT:
                morsel_mask_cmp[0](ws.scratch, ws.regs[inst.r0], inst.p0, count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_MASK_LE:
                morsel_mask_cmp[1](ws.scratch, ws.regs[inst.r0], inst.p0, count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_MASK_GT:
                morsel_mask_cmp[2](ws.scratch, ws.regs[inst.r0], inst.p0, count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_MASK_GE:
                morsel_mask_cmp[3](ws.scratch, ws.regs[inst.r0], inst.p0, count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_FILLNA:
                morsel_fillna(ws.scratch, ws.regs[inst.r0], inst.p0, count)
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case _:
                pass


def execute_bytecode_plan(
    plan: CompiledPlan,
    sources: List[SourceBinding],
    ndim: Int,
    full_extents: ImmPointer[Int64, _],
    slice_starts: ImmPointer[Int64, _],
    slice_stops: ImmPointer[Int64, _],
    reduce_axes_mask: Int,
    agg_mode: Int,
    weight_src_id: Int,
    out_fill_f32: Float32,
    out_stats_f64: MutPointer[Float64, _],
    out_arr_f32: MutPointer[Float32, _],
):
    var empty_cats = List[CatalogSourceDesc]()
    execute_bytecode_plan_with_catalog(
        plan,
        sources,
        empty_cats,
        ndim,
        full_extents,
        slice_starts,
        slice_stops,
        reduce_axes_mask,
        agg_mode,
        weight_src_id,
        out_fill_f32,
        out_stats_f64,
        out_arr_f32,
    )


def execute_bytecode_plan_with_catalog(
    plan: CompiledPlan,
    sources: List[SourceBinding],
    cat_descs: List[CatalogSourceDesc],
    ndim: Int,
    full_extents: ImmPointer[Int64, _],
    slice_starts: ImmPointer[Int64, _],
    slice_stops: ImmPointer[Int64, _],
    reduce_axes_mask: Int,
    agg_mode: Int,
    weight_src_id: Int,
    out_fill_f32: Float32,
    out_stats_f64: MutPointer[Float64, _],
    out_arr_f32: MutPointer[Float32, _],
):
    """Execute compiled bytecode DAG over strided arrays OR `RecordCatalog` planes."""
    var d0_s = Int(slice_starts[unsafe_offset=0]) if ndim > 0 else 0
    var d0_e = Int(slice_stops[unsafe_offset=0]) if ndim > 0 else 1
    var d1_s = Int(slice_starts[unsafe_offset=1]) if ndim > 1 else 0
    var d1_e = Int(slice_stops[unsafe_offset=1]) if ndim > 1 else 1
    var d2_s = Int(slice_starts[unsafe_offset=2]) if ndim > 2 else 0
    var d2_e = Int(slice_stops[unsafe_offset=2]) if ndim > 2 else 1
    var d3_s = Int(slice_starts[unsafe_offset=3]) if ndim > 3 else 0
    var d3_e = Int(slice_stops[unsafe_offset=3]) if ndim > 3 else 1

    var d0_len = max(0, d0_e - d0_s)
    var d1_len = max(0, d1_e - d1_s)
    var d2_len = max(0, d2_e - d2_s)
    var d3_len = max(0, d3_e - d3_s)
    var total_elems = d0_len * d1_len * d2_len * d3_len
    if total_elems == 0:
        for k in range(7):
            out_stats_f64[unsafe_offset=k] = 0.0
        return

    var f0 = Int(full_extents[unsafe_offset=0]) if ndim > 0 else 1
    var f1 = Int(full_extents[unsafe_offset=1]) if ndim > 1 else 1
    var f2 = Int(full_extents[unsafe_offset=2]) if ndim > 2 else 1
    var full_d0 = d0_s == 0 and d0_e == f0
    var full_d1 = d1_s == 0 and d1_e == f1
    var full_d2 = d2_s == 0 and d2_e == f2

    var src_coal = List[Bool](capacity=len(sources))
    for s_i in range(len(sources)):
        src_coal.append(
            _is_source_coalesced(
                sources[s_i],
                f0,
                f1,
                f2,
                full_d0,
                full_d1,
                full_d2,
                d1_len,
                d2_len,
                d3_len,
            )
        )
    var coal_ptr = ImmPointer[Bool, origin_of(src_coal)](src_coal.unsafe_ptr())

    var par_safe = True
    var n_cats = len(cat_descs)
    for c_i in range(n_cats):
        if not cat_descs[c_i].is_parallel_safe():
            par_safe = False
            break

    var want_axis_out = reduce_axes_mask >= 0
    var out_s0 = 0
    var out_s1 = 0
    var out_s2 = 0
    var out_s3 = 0
    var out_cells = 1
    if want_axis_out:
        if (reduce_axes_mask & 1) == 0:
            out_s0 = out_cells
            out_cells *= d0_len
        if ndim > 1 and (reduce_axes_mask & 2) == 0:
            out_s1 = out_cells
            out_cells *= d1_len
        if ndim > 2 and (reduce_axes_mask & 4) == 0:
            out_s2 = out_cells
            out_cells *= d2_len
        if ndim > 3 and (reduce_axes_mask & 8) == 0:
            out_s3 = out_cells
            out_cells *= d3_len

    var materialize_direct = agg_mode < 0

    # Spatial-partitioned high-rank reduction (`timavg` / huge 2D mean, `out_cells > 65536`
    # when inner `(d0, d1)` are kept AND no whole-record catalog decoder requires single-pass):
    if (
        want_axis_out
        and not materialize_direct
        and out_cells > 65536
        and (reduce_axes_mask & 1) == 0
        and (reduce_axes_mask & 2) == 0
        and n_cats == 0
        and par_safe
    ):
        var hw_sp = max(1, parallelism_level())
        var nw_sp = min(hw_sp, d1_len)
        var insts_sp = ImmPointer[BytecodeInst, origin_of(plan.insts)](
            plan.insts.unsafe_ptr()
        )
        var srcs_sp = ImmPointer[SourceBinding, origin_of(sources)](
            sources.unsafe_ptr()
        )
        var cats_sp = ImmPointer[CatalogSourceDesc, origin_of(cat_descs)](
            cat_descs.unsafe_ptr()
        )
        var n_insts_sp = len(plan.insts)
        var num_regs_sp = plan.num_regs
        var root_sp = plan.root_reg
        var kept_z = d2_len if (ndim > 2 and (reduce_axes_mask & 4) == 0) else 1

        var sp_gsum = List[Float64](capacity=nw_sp)
        sp_gsum.resize(nw_sp, 0.0)
        var sp_gsq = List[Float64](capacity=nw_sp)
        sp_gsq.resize(nw_sp, 0.0)
        var sp_gmin = List[Float32](capacity=nw_sp)
        sp_gmin.resize(nw_sp, Float32(1.0e30))
        var sp_gmax = List[Float32](capacity=nw_sp)
        sp_gmax.resize(nw_sp, Float32(-1.0e30))
        var sp_gval = List[Int](capacity=nw_sp)
        sp_gval.resize(nw_sp, 0)
        var sp_gtot = List[Int](capacity=nw_sp)
        sp_gtot.resize(nw_sp, 0)

        var sp_gsum_p = sp_gsum.unsafe_ptr()
        var sp_gsq_p = sp_gsq.unsafe_ptr()
        var sp_gmin_p = sp_gmin.unsafe_ptr()
        var sp_gmax_p = sp_gmax.unsafe_ptr()
        var sp_gval_p = sp_gval.unsafe_ptr()
        var sp_gtot_p = sp_gtot.unsafe_ptr()

        var no_coal = List[Bool](capacity=len(sources))
        no_coal.resize(len(sources), False)
        var no_coal_p = ImmPointer[Bool, origin_of(no_coal)](no_coal.unsafe_ptr())

        def sp_worker(wid: Int) {imm}:
            var j_beg = (wid * d1_len) // nw_sp
            var j_end = ((wid + 1) * d1_len) // nw_sp
            if j_beg >= j_end:
                return
            var sub_d1 = j_end - j_beg
            var sub_out = d0_len * sub_d1 * kept_z
            var loc_sum = List[Float64](capacity=sub_out)
            loc_sum.resize(sub_out, 0.0)
            var loc_sq = List[Float64](capacity=sub_out if agg_mode == AGG_STD else 1)
            loc_sq.resize(sub_out if agg_mode == AGG_STD else 1, 0.0)
            var loc_ext = List[Float32](capacity=sub_out)
            loc_ext.resize(
                sub_out,
                Float32(1.0e30) if agg_mode == AGG_MIN else Float32(-1.0e30),
            )
            var loc_cnt = List[Int32](capacity=sub_out)
            loc_cnt.resize(sub_out, 0)

            var ws = MorselWorkspace(num_regs_sp, cats_sp, 0)
            var acc = ReduceAcc()
            var sub_elems = d0_len * sub_d1 * d2_len * d3_len
            var sub_morsels = ceildiv(sub_elems, MORSEL_LEN)

            for mid in range(sub_morsels):
                var f_off = mid * MORSEL_LEN
                var cnt = min(MORSEL_LEN, sub_elems - f_off)
                eval_morsel_bytecode(
                    insts_sp,
                    n_insts_sp,
                    srcs_sp,
                    cats_sp,
                    ndim,
                    d0_s,
                    d0_len,
                    d1_s + j_beg,
                    sub_d1,
                    d2_s,
                    d2_len,
                    d3_s,
                    d3_len,
                    f_off,
                    cnt,
                    no_coal_p,
                    ws,
                )
                ref root = ws.regs[root_sp]
                morsel_accumulate_stats[False](root, cnt, acc)
                if root.state != MSTATE_MASKED:
                    var dp = root.data_imm()
                    for k in range(cnt):
                        if not root.is_masked(k):
                            var cur = f_off + k
                            var r0, c0 = divmod(cur, d0_len)
                            var r1, c1 = divmod(r0, sub_d1)
                            var _, c2 = divmod(r1, d2_len)
                            var z_k = c2 if kept_z > 1 else 0
                            var o_loc = c0 + c1 * d0_len + z_k * (d0_len * sub_d1)
                            var v = dp[unsafe_offset=k]
                            loc_sum[o_loc] += Float64(v)
                            loc_cnt[o_loc] += Int32(1)
                            if agg_mode == AGG_STD:
                                loc_sq[o_loc] += Float64(v) * Float64(v)
                            elif agg_mode == AGG_MIN or agg_mode == AGG_MAX:
                                loc_ext[o_loc] = (
                                    min(loc_ext[o_loc], v)
                                    if agg_mode == AGG_MIN
                                    else max(loc_ext[o_loc], v)
                                )

            # Write disjoint spatial slice straight to `out_arr_f32` (zero cross-worker merge!)
            for zk in range(kept_z):
                for cj in range(sub_d1):
                    var g_j = j_beg + cj
                    for ci in range(d0_len):
                        var o_loc = ci + cj * d0_len + zk * (d0_len * sub_d1)
                        var g_idx = ci + g_j * d0_len + zk * (d0_len * d1_len)
                        var c_tot = loc_cnt[o_loc]
                        if c_tot == 0 and agg_mode != AGG_COUNT:
                            out_arr_f32[unsafe_offset=g_idx] = out_fill_f32
                        else:
                            match agg_mode:
                                case == AGG_MEAN:
                                    out_arr_f32[unsafe_offset=g_idx] = Float32(
                                        loc_sum[o_loc] / Float64(c_tot)
                                    )
                                case == AGG_SUM:
                                    out_arr_f32[unsafe_offset=g_idx] = Float32(
                                        loc_sum[o_loc]
                                    )
                                case == AGG_MIN | == AGG_MAX:
                                    out_arr_f32[unsafe_offset=g_idx] = loc_ext[o_loc]
                                case == AGG_STD:
                                    var m = loc_sum[o_loc] / Float64(c_tot)
                                    var vr = max(
                                        0.0, loc_sq[o_loc] / Float64(c_tot) - m * m
                                    )
                                    out_arr_f32[unsafe_offset=g_idx] = Float32(sqrt(vr))
                                case _:
                                    out_arr_f32[unsafe_offset=g_idx] = Float32(c_tot)

            sp_gsum_p[unsafe_offset=wid] = acc.sum
            sp_gsq_p[unsafe_offset=wid] = acc.sum_sq
            sp_gmin_p[unsafe_offset=wid] = acc.min_val
            sp_gmax_p[unsafe_offset=wid] = acc.max_val
            sp_gval_p[unsafe_offset=wid] = acc.valid_count
            sp_gtot_p[unsafe_offset=wid] = acc.total_count

        parallelize(sp_worker, nw_sp)
        var fin_sp = ReduceAcc()
        for w in range(nw_sp):
            fin_sp.merge(
                ReduceAcc(
                    sp_gsum[w],
                    sp_gsq[w],
                    sp_gmin[w],
                    sp_gmax[w],
                    sp_gval[w],
                    sp_gtot[w],
                )
            )
        out_stats_f64[unsafe_offset=0] = fin_sp.mean()
        out_stats_f64[unsafe_offset=1] = fin_sp.std()
        out_stats_f64[unsafe_offset=2] = Float64(fin_sp.min_val)
        out_stats_f64[unsafe_offset=3] = Float64(fin_sp.max_val)
        out_stats_f64[unsafe_offset=4] = fin_sp.sum
        out_stats_f64[unsafe_offset=5] = Float64(fin_sp.valid_count)
        out_stats_f64[unsafe_offset=6] = Float64(fin_sp.total_count)
        return

    var n_morsels = ceildiv(total_elems, MORSEL_LEN)
    var hw = max(1, parallelism_level()) if par_safe else 1
    var n_workers = min(hw, n_morsels)

    var insts_ptr = ImmPointer[BytecodeInst, origin_of(plan.insts)](
        plan.insts.unsafe_ptr()
    )
    var srcs_ptr = ImmPointer[SourceBinding, origin_of(sources)](
        sources.unsafe_ptr()
    )
    var cats_ptr = ImmPointer[CatalogSourceDesc, origin_of(cat_descs)](
        cat_descs.unsafe_ptr()
    )
    var n_insts = len(plan.insts)
    var num_regs = plan.num_regs
    var root_reg = plan.root_reg
    var want_sq = plan.needs_sum_sq or agg_mode == AGG_STD

    var alloc_out_per_worker = n_workers * out_cells if (
        want_axis_out and not materialize_direct
    ) else 1

    var w_sum = List[Float64](capacity=alloc_out_per_worker)
    w_sum.resize(alloc_out_per_worker, 0.0)
    var w_sq = List[Float64](capacity=alloc_out_per_worker if agg_mode == AGG_STD else 1)
    w_sq.resize(alloc_out_per_worker if agg_mode == AGG_STD else 1, 0.0)
    var w_ext = List[Float32](capacity=alloc_out_per_worker)
    w_ext.resize(
        alloc_out_per_worker,
        Float32(1.0e30) if agg_mode == AGG_MIN else Float32(-1.0e30),
    )
    var w_cnt = List[Int32](capacity=alloc_out_per_worker)
    w_cnt.resize(alloc_out_per_worker, 0)

    var p_gsum = List[Float64](capacity=n_workers)
    p_gsum.resize(n_workers, 0.0)
    var p_gsq = List[Float64](capacity=n_workers)
    p_gsq.resize(n_workers, 0.0)
    var p_gmin = List[Float32](capacity=n_workers)
    p_gmin.resize(n_workers, Float32(1.0e30))
    var p_gmax = List[Float32](capacity=n_workers)
    p_gmax.resize(n_workers, Float32(-1.0e30))
    var p_gval = List[Int](capacity=n_workers)
    p_gval.resize(n_workers, 0)
    var p_gtot = List[Int](capacity=n_workers)
    p_gtot.resize(n_workers, 0)
    var p_gwsum = List[Float64](capacity=n_workers)
    p_gwsum.resize(n_workers, 0.0)

    var gsum_p = p_gsum.unsafe_ptr()
    var gsq_p = p_gsq.unsafe_ptr()
    var gmin_p = p_gmin.unsafe_ptr()
    var gmax_p = p_gmax.unsafe_ptr()
    var gval_p = p_gval.unsafe_ptr()
    var gtot_p = p_gtot.unsafe_ptr()
    var gwsum_p = p_gwsum.unsafe_ptr()

    var ws_p = w_sum.unsafe_ptr()
    var wsq_p = w_sq.unsafe_ptr()
    var wext_p = w_ext.unsafe_ptr()
    var wc_p = w_cnt.unsafe_ptr()
    var use_weights = weight_src_id >= 0

    def worker(wid: Int) {imm}:
        var m_begin = (wid * n_morsels) // n_workers
        var m_end = ((wid + 1) * n_morsels) // n_workers
        if m_begin >= m_end:
            return

        var ws = MorselWorkspace(num_regs, cats_ptr, n_cats)
        var local_acc = ReduceAcc()
        var local_wx: Float64 = 0.0
        var local_ww: Float64 = 0.0
        var local_wcnt = 0
        var base_w = wid * out_cells

        for mid in range(m_begin, m_end):
            var flat_off = mid * MORSEL_LEN
            var cnt = min(MORSEL_LEN, total_elems - flat_off)

            eval_morsel_bytecode(
                insts_ptr,
                n_insts,
                srcs_ptr,
                cats_ptr,
                ndim,
                d0_s,
                d0_len,
                d1_s,
                d1_len,
                d2_s,
                d2_len,
                d3_s,
                d3_len,
                flat_off,
                cnt,
                coal_ptr,
                ws,
            )

            ref root = ws.regs[root_reg]
            if use_weights:
                _load_morsel_from_source(
                    srcs_ptr[unsafe_offset=weight_src_id],
                    cats_ptr,
                    ndim,
                    d0_s,
                    d0_len,
                    d1_s,
                    d1_len,
                    d2_s,
                    d2_len,
                    d3_s,
                    d3_len,
                    flat_off,
                    cnt,
                    coal_ptr[unsafe_offset=weight_src_id],
                    ws.cat_slots,
                    ws.wt_reg,
                )
                morsel_accumulate_weighted(
                    root, ws.wt_reg, cnt, local_wx, local_ww, local_wcnt
                )
                local_acc.total_count += cnt
            elif want_sq:
                morsel_accumulate_stats[True](root, cnt, local_acc)
            else:
                morsel_accumulate_stats[False](root, cnt, local_acc)

            if materialize_direct and want_axis_out:
                morsel_store_filled(
                    root, out_arr_f32, flat_off, cnt, out_fill_f32
                )
            elif want_axis_out and root.state != MSTATE_MASKED:
                var dp = root.data_imm()
                var slab_2d = d0_len * d1_len
                if reduce_axes_mask == 12 and out_s0 == 1 and out_s1 == d0_len:
                    var rem_in_slab = flat_off % slab_2d
                    if root.state == MSTATE_VALID:
                        for k in range(cnt):
                            var o_idx = base_w + rem_in_slab
                            var v = dp[unsafe_offset=k]
                            ws_p[unsafe_offset=o_idx] += Float64(v)
                            wc_p[unsafe_offset=o_idx] += Int32(1)
                            if agg_mode == AGG_STD:
                                wsq_p[unsafe_offset=o_idx] += Float64(v) * Float64(v)
                            elif agg_mode == AGG_MIN or agg_mode == AGG_MAX:
                                var cur_e = wext_p[unsafe_offset=o_idx]
                                wext_p[unsafe_offset=o_idx] = min(cur_e, v) if agg_mode == AGG_MIN else max(cur_e, v)
                            rem_in_slab += 1
                            if rem_in_slab == slab_2d:
                                rem_in_slab = 0
                    else:
                        for k in range(cnt):
                            if not root.is_masked(k):
                                var o_idx = base_w + rem_in_slab
                                var v = dp[unsafe_offset=k]
                                ws_p[unsafe_offset=o_idx] += Float64(v)
                                wc_p[unsafe_offset=o_idx] += Int32(1)
                                if agg_mode == AGG_STD:
                                    wsq_p[unsafe_offset=o_idx] += Float64(v) * Float64(v)
                                elif agg_mode == AGG_MIN or agg_mode == AGG_MAX:
                                    var cur_e = wext_p[unsafe_offset=o_idx]
                                    wext_p[unsafe_offset=o_idx] = min(cur_e, v) if agg_mode == AGG_MIN else max(cur_e, v)
                            rem_in_slab += 1
                            if rem_in_slab == slab_2d:
                                rem_in_slab = 0
                else:
                    for k in range(cnt):
                        if not root.is_masked(k):
                            var cur = flat_off + k
                            var r0, c0 = divmod(cur, d0_len)
                            var r1, c1 = divmod(r0, d1_len)
                            var c3, c2 = divmod(r1, d2_len)
                            var o_idx = (
                                base_w
                                + c0 * out_s0
                                + c1 * out_s1
                                + c2 * out_s2
                                + c3 * out_s3
                            )
                            var v = dp[unsafe_offset=k]
                            ws_p[unsafe_offset=o_idx] += Float64(v)
                            wc_p[unsafe_offset=o_idx] += Int32(1)
                            if agg_mode == AGG_STD:
                                wsq_p[unsafe_offset=o_idx] += Float64(v) * Float64(v)
                            elif agg_mode == AGG_MIN or agg_mode == AGG_MAX:
                                var cur_e = wext_p[unsafe_offset=o_idx]
                                wext_p[unsafe_offset=o_idx] = min(cur_e, v) if agg_mode == AGG_MIN else max(cur_e, v)

        if use_weights:
            gsum_p[unsafe_offset=wid] = local_wx
            gwsum_p[unsafe_offset=wid] = local_ww
            gval_p[unsafe_offset=wid] = local_wcnt
            gtot_p[unsafe_offset=wid] = local_acc.total_count
        else:
            gsum_p[unsafe_offset=wid] = local_acc.sum
            gsq_p[unsafe_offset=wid] = local_acc.sum_sq
            gmin_p[unsafe_offset=wid] = local_acc.min_val
            gmax_p[unsafe_offset=wid] = local_acc.max_val
            gval_p[unsafe_offset=wid] = local_acc.valid_count
            gtot_p[unsafe_offset=wid] = local_acc.total_count

    parallelize(worker, n_workers)

    if use_weights:
        var tot_wx = 0.0
        var tot_ww = 0.0
        var tot_vc = 0
        var tot_tc = 0
        for w in range(n_workers):
            tot_wx += p_gsum[w]
            tot_ww += p_gwsum[w]
            tot_vc += p_gval[w]
            tot_tc += p_gtot[w]
        out_stats_f64[unsafe_offset=0] = (
            (tot_wx / tot_ww) if tot_ww > 0.0 else (0.0 / 0.0)
        )
        out_stats_f64[unsafe_offset=1] = 0.0
        out_stats_f64[unsafe_offset=2] = 0.0
        out_stats_f64[unsafe_offset=3] = 0.0
        out_stats_f64[unsafe_offset=4] = tot_wx
        out_stats_f64[unsafe_offset=5] = Float64(tot_vc)
        out_stats_f64[unsafe_offset=6] = Float64(tot_tc)
    else:
        var final_acc = ReduceAcc()
        for w in range(n_workers):
            final_acc.merge(
                ReduceAcc(
                    p_gsum[w],
                    p_gsq[w],
                    p_gmin[w],
                    p_gmax[w],
                    p_gval[w],
                    p_gtot[w],
                )
            )
        out_stats_f64[unsafe_offset=0] = final_acc.mean()
        out_stats_f64[unsafe_offset=1] = final_acc.std()
        out_stats_f64[unsafe_offset=2] = Float64(final_acc.min_val)
        out_stats_f64[unsafe_offset=3] = Float64(final_acc.max_val)
        out_stats_f64[unsafe_offset=4] = final_acc.sum
        out_stats_f64[unsafe_offset=5] = Float64(final_acc.valid_count)
        out_stats_f64[unsafe_offset=6] = Float64(final_acc.total_count)

    if want_axis_out and not materialize_direct:
        for cell in range(out_cells):
            var s_tot = 0.0
            var sq_tot = 0.0
            var c_tot: Int32 = 0
            var ext_v = Float32(1.0e30) if agg_mode == AGG_MIN else Float32(-1.0e30)
            for w in range(n_workers):
                var off = w * out_cells + cell
                var cw = w_cnt[off]
                if cw > 0:
                    s_tot += w_sum[off]
                    c_tot += cw
                    if agg_mode == AGG_STD:
                        sq_tot += w_sq[off]
                    elif agg_mode == AGG_MIN:
                        ext_v = min(ext_v, w_ext[off])
                    elif agg_mode == AGG_MAX:
                        ext_v = max(ext_v, w_ext[off])

            if c_tot == 0 and agg_mode != AGG_COUNT:
                out_arr_f32[unsafe_offset=cell] = out_fill_f32
            else:
                match agg_mode:
                    case == AGG_MEAN:
                        out_arr_f32[unsafe_offset=cell] = Float32(
                            s_tot / Float64(c_tot)
                        )
                    case == AGG_SUM:
                        out_arr_f32[unsafe_offset=cell] = Float32(s_tot)
                    case == AGG_MIN | == AGG_MAX:
                        out_arr_f32[unsafe_offset=cell] = ext_v
                    case == AGG_STD:
                        var m = s_tot / Float64(c_tot)
                        var v = max(0.0, sq_tot / Float64(c_tot) - m * m)
                        out_arr_f32[unsafe_offset=cell] = Float32(sqrt(v))
                    case _:
                        out_arr_f32[unsafe_offset=cell] = Float32(c_tot)


# ===----------------------------------------------------------------------=== #
# Single-Pass Multi-Sink DAG Executor (`Q8`, `Q10`, `Q11`: `compute_many` & `aether`)
# ===----------------------------------------------------------------------=== #


def execute_multi_sink_plan(
    plan: CompiledPlan,
    sources: List[SourceBinding],
    cat_descs: List[CatalogSourceDesc],
    ndim: Int,
    full_extents: ImmPointer[Int64, _],
    slice_starts: ImmPointer[Int64, _],
    slice_stops: ImmPointer[Int64, _],
    n_sinks: Int,
    sink_red_masks: ImmPointer[Int32, _],   # -1 = scalar, 0 = materialize, >0 = axis reduce
    sink_agg_modes: ImmPointer[Int32, _],   # -1 = materialize, 0..5 = AGG_MEAN..AGG_COUNT
    sink_wt_srcs: ImmPointer[Int32, _],     # >= 0 for weighted global mean (`areacell`)
    sink_fills: ImmPointer[Float32, _],     # `NaN`, `1e30` (`write_fst`), or `0.0` (`aether`!)
    sink_out_addrs: ImmPointer[Int64, _],   # output `Float32*` address per sink
    out_sink_stats_f64: MutPointer[Float64, _],  # `n_sinks * 8` Float64 stats table
):
    """Evaluate `n_sinks` expressions/reductions in ONE fused pass over all inputs."""
    var d0_s = Int(slice_starts[unsafe_offset=0]) if ndim > 0 else 0
    var d0_e = Int(slice_stops[unsafe_offset=0]) if ndim > 0 else 1
    var d1_s = Int(slice_starts[unsafe_offset=1]) if ndim > 1 else 0
    var d1_e = Int(slice_stops[unsafe_offset=1]) if ndim > 1 else 1
    var d2_s = Int(slice_starts[unsafe_offset=2]) if ndim > 2 else 0
    var d2_e = Int(slice_stops[unsafe_offset=2]) if ndim > 2 else 1
    var d3_s = Int(slice_starts[unsafe_offset=3]) if ndim > 3 else 0
    var d3_e = Int(slice_stops[unsafe_offset=3]) if ndim > 3 else 1

    var d0_len = max(0, d0_e - d0_s)
    var d1_len = max(0, d1_e - d1_s)
    var d2_len = max(0, d2_e - d2_s)
    var d3_len = max(0, d3_e - d3_s)
    var total_elems = d0_len * d1_len * d2_len * d3_len
    if total_elems == 0 or n_sinks <= 0:
        return

    var f0 = Int(full_extents[unsafe_offset=0]) if ndim > 0 else 1
    var f1 = Int(full_extents[unsafe_offset=1]) if ndim > 1 else 1
    var f2 = Int(full_extents[unsafe_offset=2]) if ndim > 2 else 1
    var full_d0 = d0_s == 0 and d0_e == f0
    var full_d1 = d1_s == 0 and d1_e == f1
    var full_d2 = d2_s == 0 and d2_e == f2

    var src_coal = List[Bool](capacity=len(sources))
    for s_i in range(len(sources)):
        src_coal.append(
            _is_source_coalesced(
                sources[s_i],
                f0,
                f1,
                f2,
                full_d0,
                full_d1,
                full_d2,
                d1_len,
                d2_len,
                d3_len,
            )
        )
    var coal_ptr = ImmPointer[Bool, origin_of(src_coal)](src_coal.unsafe_ptr())

    var par_safe = True
    var n_cats = len(cat_descs)
    for c_i in range(n_cats):
        if not cat_descs[c_i].is_parallel_safe():
            par_safe = False
            break

    var n_morsels = ceildiv(total_elems, MORSEL_LEN)
    var hw = max(1, parallelism_level()) if par_safe else 1
    var n_workers = min(hw, n_morsels)

    # Precompute per-sink axis-reduction geometry & packed workspace offsets
    var s_out_s0 = List[Int](capacity=n_sinks)
    var s_out_s1 = List[Int](capacity=n_sinks)
    var s_out_s2 = List[Int](capacity=n_sinks)
    var s_out_s3 = List[Int](capacity=n_sinks)
    var s_out_cells = List[Int](capacity=n_sinks)
    var s_buf_base = List[Int](capacity=n_sinks)
    var total_axis_cells = 0

    for s in range(n_sinks):
        var rmask = Int(sink_red_masks[unsafe_offset=s])
        var agg = Int(sink_agg_modes[unsafe_offset=s])
        var os0 = 0
        var os1 = 0
        var os2 = 0
        var os3 = 0
        var oc = 0
        if rmask >= 0 and agg >= 0:
            oc = 1
            if (rmask & 1) == 0:
                os0 = oc
                oc *= d0_len
            if ndim > 1 and (rmask & 2) == 0:
                os1 = oc
                oc *= d1_len
            if ndim > 2 and (rmask & 4) == 0:
                os2 = oc
                oc *= d2_len
            if ndim > 3 and (rmask & 8) == 0:
                os3 = oc
                oc *= d3_len
        s_out_s0.append(os0)
        s_out_s1.append(os1)
        s_out_s2.append(os2)
        s_out_s3.append(os3)
        s_out_cells.append(oc)
        s_buf_base.append(total_axis_cells)
        total_axis_cells += oc

    var axis_alloc = max(1, n_workers * total_axis_cells)
    var w_sum = List[Float64](capacity=axis_alloc)
    w_sum.resize(axis_alloc, 0.0)
    var w_sq = List[Float64](capacity=axis_alloc)
    w_sq.resize(axis_alloc, 0.0)
    var w_cnt = List[Int32](capacity=axis_alloc)
    w_cnt.resize(axis_alloc, 0)

    var g_slots = n_workers * n_sinks
    var p_gsum = List[Float64](capacity=g_slots)
    p_gsum.resize(g_slots, 0.0)
    var p_gsq = List[Float64](capacity=g_slots)
    p_gsq.resize(g_slots, 0.0)
    var p_gwsum = List[Float64](capacity=g_slots)
    p_gwsum.resize(g_slots, 0.0)
    var p_gmin = List[Float32](capacity=g_slots)
    p_gmin.resize(g_slots, Float32(1.0e30))
    var p_gmax = List[Float32](capacity=g_slots)
    p_gmax.resize(g_slots, Float32(-1.0e30))
    var p_gval = List[Int](capacity=g_slots)
    p_gval.resize(g_slots, 0)
    var p_gtot = List[Int](capacity=g_slots)
    p_gtot.resize(g_slots, 0)

    var insts_ptr = ImmPointer[BytecodeInst, origin_of(plan.insts)](
        plan.insts.unsafe_ptr()
    )
    var sink_regs_ptr = ImmPointer[Int, origin_of(plan.sink_regs)](
        plan.sink_regs.unsafe_ptr()
    )
    var srcs_ptr = ImmPointer[SourceBinding, origin_of(sources)](
        sources.unsafe_ptr()
    )
    var cats_ptr = ImmPointer[CatalogSourceDesc, origin_of(cat_descs)](
        cat_descs.unsafe_ptr()
    )
    var os0_p = ImmPointer[Int, origin_of(s_out_s0)](s_out_s0.unsafe_ptr())
    var os1_p = ImmPointer[Int, origin_of(s_out_s1)](s_out_s1.unsafe_ptr())
    var os2_p = ImmPointer[Int, origin_of(s_out_s2)](s_out_s2.unsafe_ptr())
    var os3_p = ImmPointer[Int, origin_of(s_out_s3)](s_out_s3.unsafe_ptr())
    var sb_p = ImmPointer[Int, origin_of(s_buf_base)](s_buf_base.unsafe_ptr())

    var ws_p = w_sum.unsafe_ptr()
    var wsq_p = w_sq.unsafe_ptr()
    var wc_p = w_cnt.unsafe_ptr()
    var gsum_p = p_gsum.unsafe_ptr()
    var gsq_p = p_gsq.unsafe_ptr()
    var gwsum_p = p_gwsum.unsafe_ptr()
    var gmin_p = p_gmin.unsafe_ptr()
    var gmax_p = p_gmax.unsafe_ptr()
    var gval_p = p_gval.unsafe_ptr()
    var gtot_p = p_gtot.unsafe_ptr()

    var n_insts = len(plan.insts)
    var num_regs = plan.num_regs

    def ms_worker(wid: Int) {imm}:
        var m_begin = (wid * n_morsels) // n_workers
        var m_end = ((wid + 1) * n_morsels) // n_workers
        if m_begin >= m_end:
            return
        var ws = MorselWorkspace(num_regs, cats_ptr, n_cats)
        var loc_accs = List[ReduceAcc](capacity=n_sinks)
        var loc_wx = List[Float64](capacity=n_sinks)
        var loc_ww = List[Float64](capacity=n_sinks)
        var loc_wc = List[Int](capacity=n_sinks)
        for _ in range(n_sinks):
            loc_accs.append(ReduceAcc())
            loc_wx.append(0.0)
            loc_ww.append(0.0)
            loc_wc.append(0)

        for mid in range(m_begin, m_end):
            var flat_off = mid * MORSEL_LEN
            var cnt = min(MORSEL_LEN, total_elems - flat_off)
            var last_loaded_wt = -1

            eval_morsel_bytecode(
                insts_ptr,
                n_insts,
                srcs_ptr,
                cats_ptr,
                ndim,
                d0_s,
                d0_len,
                d1_s,
                d1_len,
                d2_s,
                d2_len,
                d3_s,
                d3_len,
                flat_off,
                cnt,
                coal_ptr,
                ws,
            )

            for s in range(n_sinks):
                var s_reg = sink_regs_ptr[unsafe_offset=s]
                ref root = ws.regs[s_reg]
                var rmask = Int(sink_red_masks[unsafe_offset=s])
                var agg = Int(sink_agg_modes[unsafe_offset=s])
                var wt_id = Int(sink_wt_srcs[unsafe_offset=s])
                var fill_v = sink_fills[unsafe_offset=s]

                if agg < 0:
                    # Direct materialization (`aether` `(n_fields, n_src)` or `write_fst` `1e30`)
                    var out_p = MutPointer[Float32, MutUntrackedOrigin](
                        unsafe_from_address=Int(sink_out_addrs[unsafe_offset=s])
                    )
                    morsel_store_filled(root, out_p, flat_off, cnt, fill_v)
                elif rmask < 0:
                    # Global scalar reduction (weighted `globavg` or unweighted stats)
                    if wt_id >= 0:
                        if last_loaded_wt != wt_id:
                            _load_morsel_from_source(
                                srcs_ptr[unsafe_offset=wt_id],
                                cats_ptr,
                                ndim,
                                d0_s,
                                d0_len,
                                d1_s,
                                d1_len,
                                d2_s,
                                d2_len,
                                d3_s,
                                d3_len,
                                flat_off,
                                cnt,
                                coal_ptr[unsafe_offset=wt_id],
                                ws.cat_slots,
                                ws.wt_reg,
                            )
                            last_loaded_wt = wt_id
                        morsel_accumulate_weighted(
                            root,
                            ws.wt_reg,
                            cnt,
                            loc_wx[s],
                            loc_ww[s],
                            loc_wc[s],
                        )
                        loc_accs[s].total_count += cnt
                    elif agg == AGG_STD:
                        morsel_accumulate_stats[True](root, cnt, loc_accs[s])
                    else:
                        morsel_accumulate_stats[False](root, cnt, loc_accs[s])
                else:
                    # Axis reduction (`zonavg`, `meriavg`, etc.)
                    if root.state != MSTATE_MASKED:
                        var dp = root.data_imm()
                        var w_base = wid * total_axis_cells + sb_p[unsafe_offset=s]
                        var os0 = os0_p[unsafe_offset=s]
                        var os1 = os1_p[unsafe_offset=s]
                        var os2 = os2_p[unsafe_offset=s]
                        var os3 = os3_p[unsafe_offset=s]
                        for k in range(cnt):
                            if not root.is_masked(k):
                                var cur = flat_off + k
                                var r0, c0 = divmod(cur, d0_len)
                                var r1, c1 = divmod(r0, d1_len)
                                var c3, c2 = divmod(r1, d2_len)
                                var o_idx = (
                                    w_base
                                    + c0 * os0
                                    + c1 * os1
                                    + c2 * os2
                                    + c3 * os3
                                )
                                var v = dp[unsafe_offset=k]
                                ws_p[unsafe_offset=o_idx] += Float64(v)
                                wc_p[unsafe_offset=o_idx] += Int32(1)
                                if agg == AGG_STD:
                                    wsq_p[unsafe_offset=o_idx] += Float64(v) * Float64(v)

        for s in range(n_sinks):
            var g_idx = wid * n_sinks + s
            if Int(sink_wt_srcs[unsafe_offset=s]) >= 0:
                gsum_p[unsafe_offset=g_idx] = loc_wx[s]
                gwsum_p[unsafe_offset=g_idx] = loc_ww[s]
                gval_p[unsafe_offset=g_idx] = loc_wc[s]
                gtot_p[unsafe_offset=g_idx] = loc_accs[s].total_count
            else:
                gsum_p[unsafe_offset=g_idx] = loc_accs[s].sum
                gsq_p[unsafe_offset=g_idx] = loc_accs[s].sum_sq
                gmin_p[unsafe_offset=g_idx] = loc_accs[s].min_val
                gmax_p[unsafe_offset=g_idx] = loc_accs[s].max_val
                gval_p[unsafe_offset=g_idx] = loc_accs[s].valid_count
                gtot_p[unsafe_offset=g_idx] = loc_accs[s].total_count

    parallelize(ms_worker, n_workers)

    # Finalize all `n_sinks` scalar stats & axis reduction outputs
    for s in range(n_sinks):
        var rmask = Int(sink_red_masks[unsafe_offset=s])
        var agg = Int(sink_agg_modes[unsafe_offset=s])
        var wt_id = Int(sink_wt_srcs[unsafe_offset=s])
        var fill_v = sink_fills[unsafe_offset=s]
        var st_base = s * 8

        if wt_id >= 0:
            var twx = 0.0
            var tww = 0.0
            var tvc = 0
            var ttc = 0
            for w in range(n_workers):
                var gi = w * n_sinks + s
                twx += p_gsum[gi]
                tww += p_gwsum[gi]
                tvc += p_gval[gi]
                ttc += p_gtot[gi]
            out_sink_stats_f64[unsafe_offset=st_base + 0] = (
                (twx / tww) if tww > 0.0 else (0.0 / 0.0)
            )
            out_sink_stats_f64[unsafe_offset=st_base + 4] = twx
            out_sink_stats_f64[unsafe_offset=st_base + 5] = Float64(tvc)
            out_sink_stats_f64[unsafe_offset=st_base + 6] = Float64(ttc)
        else:
            var fa = ReduceAcc()
            for w in range(n_workers):
                var gi = w * n_sinks + s
                fa.merge(
                    ReduceAcc(
                        p_gsum[gi],
                        p_gsq[gi],
                        p_gmin[gi],
                        p_gmax[gi],
                        p_gval[gi],
                        p_gtot[gi],
                    )
                )
            out_sink_stats_f64[unsafe_offset=st_base + 0] = fa.mean()
            out_sink_stats_f64[unsafe_offset=st_base + 1] = fa.std()
            out_sink_stats_f64[unsafe_offset=st_base + 2] = Float64(fa.min_val)
            out_sink_stats_f64[unsafe_offset=st_base + 3] = Float64(fa.max_val)
            out_sink_stats_f64[unsafe_offset=st_base + 4] = fa.sum
            out_sink_stats_f64[unsafe_offset=st_base + 5] = Float64(fa.valid_count)
            out_sink_stats_f64[unsafe_offset=st_base + 6] = Float64(fa.total_count)

        if rmask >= 0 and agg >= 0:
            var oc = s_out_cells[s]
            var s_off = s_buf_base[s]
            var out_p = MutPointer[Float32, MutUntrackedOrigin](
                unsafe_from_address=Int(sink_out_addrs[unsafe_offset=s])
            )
            for cell in range(oc):
                var s_tot = 0.0
                var sq_tot = 0.0
                var c_tot: Int32 = 0
                for w in range(n_workers):
                    var idx_w = w * total_axis_cells + s_off + cell
                    var cw = w_cnt[idx_w]
                    if cw > 0:
                        s_tot += w_sum[idx_w]
                        c_tot += cw
                        if agg == AGG_STD:
                            sq_tot += w_sq[idx_w]
                if c_tot == 0 and agg != AGG_COUNT:
                    out_p[unsafe_offset=cell] = fill_v
                elif agg == AGG_MEAN:
                    out_p[unsafe_offset=cell] = Float32(s_tot / Float64(c_tot))
                elif agg == AGG_SUM:
                    out_p[unsafe_offset=cell] = Float32(s_tot)
                elif agg == AGG_STD:
                    var m = s_tot / Float64(c_tot)
                    out_p[unsafe_offset=cell] = Float32(
                        sqrt(max(0.0, sq_tot / Float64(c_tot) - m * m))
                    )
                else:
                    out_p[unsafe_offset=cell] = Float32(c_tot)


# ===----------------------------------------------------------------------=== #
# `Float64` Morsel Execution Engine (`Q7`: `boreal-stat` `values_f64` Parity)
# ===----------------------------------------------------------------------=== #


struct MorselWorkspace64(Movable):
    var regs: List[MorselReg64]
    var scratch: MorselReg64
    var wt_reg: MorselReg64

    def __init__(out self, num_regs: Int):
        self.regs = List[MorselReg64](capacity=num_regs)
        for _ in range(num_regs):
            self.regs.append(MorselReg64())
        self.scratch = MorselReg64()
        self.wt_reg = MorselReg64()

    def __init__(out self, *, deinit move: Self):
        self.regs = move.regs^
        self.scratch = move.scratch^
        self.wt_reg = move.wt_reg^


@inline(.always)
def _load_morsel64(
    src: SourceBinding,
    d0_start: Int,
    d0_len: Int,
    d1_start: Int,
    d1_len: Int,
    d2_start: Int,
    d2_len: Int,
    d3_start: Int,
    flat_offset: Int,
    count: Int,
    mut dst: MorselReg64,
):
    var mp = dst.mask_ptr()
    var n_words = ceildiv(count, 64)
    if src.mask_kind != MSRC_NONE:
        for w in range(n_words):
            mp[unsafe_offset=w] = UInt64(0)
    var or_bits: UInt64 = 0
    var item_b = 8 if src.dtype_code == DTYPE_F64 else (2 if src.dtype_code == DTYPE_U16 else 4)
    var aux_b = 4 if src.mask_kind == MSRC_FST_BITMAP_F32 else 1

    var written = 0
    while written < count:
        var cur = flat_offset + written
        var rem0, i_rel = divmod(cur, d0_len)
        var rem1, j_rel = divmod(rem0, d1_len)
        var t_rel, z_rel = divmod(rem1, d2_len)
        var span = min(d0_len - i_rel, count - written)
        var elem_off = (
            (d0_start + i_rel) * src.s0
            + (d1_start + j_rel) * src.s1
            + (d2_start + z_rel) * src.s2
            + (d3_start + t_rel) * src.s3
        )
        if src.s0 == 1:
            var p_src = ImmPointer[UInt8, ImmUntrackedOrigin](
                unsafe_from_address=src.data_addr + elem_off * item_b
            )
            var p_aux = ImmPointer[UInt8, ImmUntrackedOrigin](
                unsafe_from_address=src.aux_addr + elem_off * aux_b
                if src.aux_addr != 0
                else src.data_addr
            )
            load_span_into_reg64(
                p_src,
                p_aux,
                src.dtype_code,
                written,
                span,
                Float64(src.scale),
                Float64(src.offset),
                src.mask_kind,
                src.mask_param,
                dst,
                or_bits,
            )
        elif src.s0 == 0:
            var p_src = ImmPointer[UInt8, ImmUntrackedOrigin](
                unsafe_from_address=src.data_addr + elem_off * item_b
            )
            var p_aux = ImmPointer[UInt8, ImmUntrackedOrigin](
                unsafe_from_address=src.aux_addr + elem_off * aux_b
                if src.aux_addr != 0
                else src.data_addr
            )
            load_span_into_reg64(
                p_src,
                p_aux,
                src.dtype_code,
                written,
                1,
                Float64(src.scale),
                Float64(src.offset),
                src.mask_kind,
                src.mask_param,
                dst,
                or_bits,
            )
            var bval = dst.data_imm()[unsafe_offset=written]
            var bmsk = (
                ((mp[unsafe_offset=written >> 6] >> UInt64(written & 63)) & UInt64(1))
                != UInt64(0)
                if src.mask_kind != MSRC_NONE
                else False
            )
            var dp = dst.data_ptr()
            for s_i in range(1, span):
                var d_i = written + s_i
                dp[unsafe_offset=d_i] = bval
                if bmsk:
                    mp[unsafe_offset=d_i >> 6] |= UInt64(1) << UInt64(d_i & 63)
        written += span

    dst.finalize_words(count, or_bits)


def execute_bytecode_plan_f64(
    plan: CompiledPlan,
    sources: List[SourceBinding],
    ndim: Int,
    slice_starts: ImmPointer[Int64, _],
    slice_stops: ImmPointer[Int64, _],
    reduce_axes_mask: Int,
    agg_mode: Int,
    weight_src_id: Int,
    out_fill_f64: Float64,
    out_stats_f64: MutPointer[Float64, _],
    out_arr_f64: MutPointer[Float64, _],
):
    """Execute bytecode DAG in full `Float64` morsel registers (`boreal-stat` `r.diag` parity)."""
    var d0_s = Int(slice_starts[unsafe_offset=0]) if ndim > 0 else 0
    var d0_e = Int(slice_stops[unsafe_offset=0]) if ndim > 0 else 1
    var d1_s = Int(slice_starts[unsafe_offset=1]) if ndim > 1 else 0
    var d1_e = Int(slice_stops[unsafe_offset=1]) if ndim > 1 else 1
    var d2_s = Int(slice_starts[unsafe_offset=2]) if ndim > 2 else 0
    var d2_e = Int(slice_stops[unsafe_offset=2]) if ndim > 2 else 1
    var d3_s = Int(slice_starts[unsafe_offset=3]) if ndim > 3 else 0
    var d3_e = Int(slice_stops[unsafe_offset=3]) if ndim > 3 else 1

    var d0_len = max(0, d0_e - d0_s)
    var d1_len = max(0, d1_e - d1_s)
    var d2_len = max(0, d2_e - d2_s)
    var d3_len = max(0, d3_e - d3_s)
    var total_elems = d0_len * d1_len * d2_len * d3_len
    if total_elems == 0:
        return

    var want_axis_out = reduce_axes_mask >= 0
    var out_s0 = 0
    var out_s1 = 0
    var out_s2 = 0
    var out_s3 = 0
    var out_cells = 1
    if want_axis_out:
        if (reduce_axes_mask & 1) == 0:
            out_s0 = out_cells
            out_cells *= d0_len
        if ndim > 1 and (reduce_axes_mask & 2) == 0:
            out_s1 = out_cells
            out_cells *= d1_len
        if ndim > 2 and (reduce_axes_mask & 4) == 0:
            out_s2 = out_cells
            out_cells *= d2_len
        if ndim > 3 and (reduce_axes_mask & 8) == 0:
            out_s3 = out_cells
            out_cells *= d3_len

    var materialize_direct = agg_mode < 0
    var n_morsels = ceildiv(total_elems, MORSEL_LEN)
    var hw = max(1, parallelism_level())
    var n_workers = min(hw, n_morsels)

    var alloc_axis = n_workers * out_cells if (want_axis_out and not materialize_direct) else 1
    var w_sum = List[Float64](capacity=alloc_axis)
    w_sum.resize(alloc_axis, 0.0)
    var w_sq = List[Float64](capacity=alloc_axis if agg_mode == AGG_STD else 1)
    w_sq.resize(alloc_axis if agg_mode == AGG_STD else 1, 0.0)
    var w_cnt = List[Int32](capacity=alloc_axis)
    w_cnt.resize(alloc_axis, 0)

    var p_gsum = List[Float64](capacity=n_workers)
    p_gsum.resize(n_workers, 0.0)
    var p_gsq = List[Float64](capacity=n_workers)
    p_gsq.resize(n_workers, 0.0)
    var p_gwsum = List[Float64](capacity=n_workers)
    p_gwsum.resize(n_workers, 0.0)
    var p_gmin = List[Float32](capacity=n_workers)
    p_gmin.resize(n_workers, Float32(1.0e30))
    var p_gmax = List[Float32](capacity=n_workers)
    p_gmax.resize(n_workers, Float32(-1.0e30))
    var p_gval = List[Int](capacity=n_workers)
    p_gval.resize(n_workers, 0)
    var p_gtot = List[Int](capacity=n_workers)
    p_gtot.resize(n_workers, 0)

    var insts_ptr = ImmPointer[BytecodeInst, origin_of(plan.insts)](plan.insts.unsafe_ptr())
    var srcs_ptr = ImmPointer[SourceBinding, origin_of(sources)](sources.unsafe_ptr())
    var n_insts = len(plan.insts)
    var num_regs = plan.num_regs
    var root_reg = plan.root_reg
    var want_sq = plan.needs_sum_sq or agg_mode == AGG_STD
    var use_weights = weight_src_id >= 0

    var ws_p = w_sum.unsafe_ptr()
    var wsq_p = w_sq.unsafe_ptr()
    var wc_p = w_cnt.unsafe_ptr()
    var gsum_p = p_gsum.unsafe_ptr()
    var gsq_p = p_gsq.unsafe_ptr()
    var gwsum_p = p_gwsum.unsafe_ptr()
    var gmin_p = p_gmin.unsafe_ptr()
    var gmax_p = p_gmax.unsafe_ptr()
    var gval_p = p_gval.unsafe_ptr()
    var gtot_p = p_gtot.unsafe_ptr()

    def f64_worker(wid: Int) {imm}:
        var m_begin = (wid * n_morsels) // n_workers
        var m_end = ((wid + 1) * n_morsels) // n_workers
        if m_begin >= m_end:
            return
        var ws = MorselWorkspace64(num_regs)
        var acc = ReduceAcc()
        var l_wx = 0.0
        var l_ww = 0.0
        var l_wc = 0
        var base_w = wid * out_cells

        for mid in range(m_begin, m_end):
            var flat_off = mid * MORSEL_LEN
            var cnt = min(MORSEL_LEN, total_elems - flat_off)

            for pc in range(n_insts):
                ref inst = insts_ptr[unsafe_offset=pc]
                match inst.op:
                    case == BC_LOAD:
                        _load_morsel64(
                            srcs_ptr[unsafe_offset=inst.src_id],
                            d0_s,
                            d0_len,
                            d1_s,
                            d1_len,
                            d2_s,
                            d2_len,
                            d3_s,
                            flat_off,
                            cnt,
                            ws.regs[inst.dst_reg],
                        )
                    case == BC_CONST:
                        ws.regs[inst.dst_reg].state = MSTATE_VALID
                        var dp = ws.regs[inst.dst_reg].data_ptr()
                        var cv = Float64(inst.p0)
                        for k in range(cnt):
                            dp[unsafe_offset=k] = cv
                    case == BC_ADD:
                        morsel64_binary[0](ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], cnt)
                        swap(ws.regs[inst.dst_reg], ws.scratch)
                    case == BC_SUB:
                        morsel64_binary[1](ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], cnt)
                        swap(ws.regs[inst.dst_reg], ws.scratch)
                    case == BC_MUL:
                        morsel64_binary[2](ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], cnt)
                        swap(ws.regs[inst.dst_reg], ws.scratch)
                    case == BC_DIV:
                        morsel64_binary[3](ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], cnt)
                        swap(ws.regs[inst.dst_reg], ws.scratch)
                    case == BC_HYPOT:
                        morsel64_binary[4](ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], cnt)
                        swap(ws.regs[inst.dst_reg], ws.scratch)
                    case == BC_POW2_SUM:
                        morsel64_binary[5](ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], cnt)
                        swap(ws.regs[inst.dst_reg], ws.scratch)
                    case == BC_MIN:
                        morsel64_binary[6](ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], cnt)
                        swap(ws.regs[inst.dst_reg], ws.scratch)
                    case == BC_MAX:
                        morsel64_binary[7](ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], cnt)
                        swap(ws.regs[inst.dst_reg], ws.scratch)
                    case == BC_AFFINE:
                        morsel64_affine(
                            ws.scratch,
                            ws.regs[inst.r0],
                            Float64(inst.p0),
                            Float64(inst.p1),
                            cnt,
                        )
                        swap(ws.regs[inst.dst_reg], ws.scratch)
                    case == BC_FMA:
                        morsel64_fma(
                            ws.scratch,
                            ws.regs[inst.r0],
                            ws.regs[inst.r1],
                            ws.regs[inst.r2],
                            cnt,
                        )
                        swap(ws.regs[inst.dst_reg], ws.scratch)
                    case == BC_NEG:
                        morsel64_unary[0](ws.scratch, ws.regs[inst.r0], cnt)
                        swap(ws.regs[inst.dst_reg], ws.scratch)
                    case == BC_ABS:
                        morsel64_unary[1](ws.scratch, ws.regs[inst.r0], cnt)
                        swap(ws.regs[inst.dst_reg], ws.scratch)
                    case == BC_SQRT:
                        morsel64_unary[2](ws.scratch, ws.regs[inst.r0], cnt)
                        swap(ws.regs[inst.dst_reg], ws.scratch)
                    case == BC_EXP:
                        morsel64_unary[3](ws.scratch, ws.regs[inst.r0], cnt)
                        swap(ws.regs[inst.dst_reg], ws.scratch)
                    case == BC_LOG:
                        morsel64_unary[4](ws.scratch, ws.regs[inst.r0], cnt)
                        swap(ws.regs[inst.dst_reg], ws.scratch)
                    case == BC_SIN:
                        morsel64_unary[5](ws.scratch, ws.regs[inst.r0], cnt)
                        swap(ws.regs[inst.dst_reg], ws.scratch)
                    case == BC_COS:
                        morsel64_unary[6](ws.scratch, ws.regs[inst.r0], cnt)
                        swap(ws.regs[inst.dst_reg], ws.scratch)
                    case _:
                        pass

            ref root = ws.regs[root_reg]
            if use_weights:
                _load_morsel64(
                    srcs_ptr[unsafe_offset=weight_src_id],
                    d0_s,
                    d0_len,
                    d1_s,
                    d1_len,
                    d2_s,
                    d2_len,
                    d3_s,
                    flat_off,
                    cnt,
                    ws.wt_reg,
                )
                morsel64_accumulate_weighted(root, ws.wt_reg, cnt, l_wx, l_ww, l_wc)
                acc.total_count += cnt
            else:
                morsel64_accumulate_stats(root, cnt, want_sq, acc)

            if materialize_direct and want_axis_out:
                morsel64_store_filled(root, out_arr_f64, flat_off, cnt, out_fill_f64)
            elif want_axis_out and root.state != MSTATE_MASKED:
                var dp = root.data_imm()
                for k in range(cnt):
                    if not root.is_masked(k):
                        var cur = flat_off + k
                        var r0, c0 = divmod(cur, d0_len)
                        var r1, c1 = divmod(r0, d1_len)
                        var c3, c2 = divmod(r1, d2_len)
                        var o_idx = (
                            base_w + c0 * out_s0 + c1 * out_s1 + c2 * out_s2 + c3 * out_s3
                        )
                        var v = dp[unsafe_offset=k]
                        ws_p[unsafe_offset=o_idx] += v
                        wc_p[unsafe_offset=o_idx] += Int32(1)
                        if agg_mode == AGG_STD:
                            wsq_p[unsafe_offset=o_idx] += v * v

        if use_weights:
            gsum_p[unsafe_offset=wid] = l_wx
            gwsum_p[unsafe_offset=wid] = l_ww
            gval_p[unsafe_offset=wid] = l_wc
            gtot_p[unsafe_offset=wid] = acc.total_count
        else:
            gsum_p[unsafe_offset=wid] = acc.sum
            gsq_p[unsafe_offset=wid] = acc.sum_sq
            gmin_p[unsafe_offset=wid] = acc.min_val
            gmax_p[unsafe_offset=wid] = acc.max_val
            gval_p[unsafe_offset=wid] = acc.valid_count
            gtot_p[unsafe_offset=wid] = acc.total_count

    parallelize(f64_worker, n_workers)

    if use_weights:
        var twx = 0.0
        var tww = 0.0
        var tvc = 0
        var ttc = 0
        for w in range(n_workers):
            twx += p_gsum[w]
            tww += p_gwsum[w]
            tvc += p_gval[w]
            ttc += p_gtot[w]
        out_stats_f64[unsafe_offset=0] = (twx / tww) if tww > 0.0 else (0.0 / 0.0)
        out_stats_f64[unsafe_offset=4] = twx
        out_stats_f64[unsafe_offset=5] = Float64(tvc)
        out_stats_f64[unsafe_offset=6] = Float64(ttc)
    else:
        var fa = ReduceAcc()
        for w in range(n_workers):
            fa.merge(
                ReduceAcc(
                    p_gsum[w],
                    p_gsq[w],
                    p_gmin[w],
                    p_gmax[w],
                    p_gval[w],
                    p_gtot[w],
                )
            )
        out_stats_f64[unsafe_offset=0] = fa.mean()
        out_stats_f64[unsafe_offset=1] = fa.std()
        out_stats_f64[unsafe_offset=2] = Float64(fa.min_val)
        out_stats_f64[unsafe_offset=3] = Float64(fa.max_val)
        out_stats_f64[unsafe_offset=4] = fa.sum
        out_stats_f64[unsafe_offset=5] = Float64(fa.valid_count)
        out_stats_f64[unsafe_offset=6] = Float64(fa.total_count)

    if want_axis_out and not materialize_direct:
        for cell in range(out_cells):
            var s_tot = 0.0
            var sq_tot = 0.0
            var c_tot: Int32 = 0
            for w in range(n_workers):
                var off = w * out_cells + cell
                var cw = w_cnt[off]
                if cw > 0:
                    s_tot += w_sum[off]
                    c_tot += cw
                    if agg_mode == AGG_STD:
                        sq_tot += w_sq[off]
            if c_tot == 0 and agg_mode != AGG_COUNT:
                out_arr_f64[unsafe_offset=cell] = out_fill_f64
            elif agg_mode == AGG_MEAN:
                out_arr_f64[unsafe_offset=cell] = s_tot / Float64(c_tot)
            elif agg_mode == AGG_SUM:
                out_arr_f64[unsafe_offset=cell] = s_tot
            elif agg_mode == AGG_STD:
                var m = s_tot / Float64(c_tot)
                out_arr_f64[unsafe_offset=cell] = sqrt(
                    max(0.0, sq_tot / Float64(c_tot) - m * m)
                )
            else:
                out_arr_f64[unsafe_offset=cell] = Float64(c_tot)
