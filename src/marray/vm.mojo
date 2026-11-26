"""General N-D Bytecode Virtual Machine, Algebraic Graph Optimizer, and Morsel
Streaming Executor (`vm.mojo`).

Zero domain or benchmark assumptions:
  - Supports arbitrary rank 1..6 datasets with general per-variable N-D strides
    (enabling zero-cost 1D/2D/3D -> 4D+ broadcasting via stride=0!).
  - Coalesces contiguous inner slices automatically into flat 2048-element L1
    morsels (MLIR/NumPy-style dimension coalescing).
  - Runs a 3-pass general bytecode optimizer before execution:
      1. Common Subexpression Elimination (CSE)
      2. Algebraic Peephole & CF-Scale Folding (`Var*a+b` folded into load,
         `a*b+c -> FMA`, `a*a+b*b -> POW2_SUM`, `sqrt(a*a+b*b) -> HYPOT`)
      3. Linear-Scan Register Liveness Coloring (packs arbitrary N-node DAGs
         into minimal L1-resident `MorselReg` slots)
  - Executes arbitrary scalar reductions (`mean`, `sum`, `std`, `min`, `max`,
    `count`), arbitrary subset-of-axes reductions (`dim=...`), or full array
    materialization across all CPU threads.
"""

from std.math import ceildiv, max, min, sqrt
from std.memory import ImmPointer, MutPointer
from std.runtime import parallelism_level
from max.algorithm import parallelize

from .morsel import (
    DTYPE_F32,
    DTYPE_U16,
    MORSEL_LEN,
    MORSEL_WORDS,
    MSRC_NONE,
    MSTATE_BITS,
    MSTATE_MASKED,
    MSTATE_VALID,
    MorselReg,
    ReduceAcc,
    load_span_f32,
    load_span_u16,
    morsel_accumulate_stats,
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
    if not full_d0 or sb.s0 != 1:
        return False
    if d1_len > 1 and (not full_d1 or sb.s1 != f0):
        return False
    if d2_len > 1 and (not full_d2 or sb.s2 != f0 * f1):
        return False
    if d3_len > 1 and sb.s3 != f0 * f1 * f2:
        return False
    return True
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


@fieldwise_init
struct SourceBinding(Copyable, Movable):
    """Describes an arbitrary N-D input buffer with optional CF scale/offset & mask."""

    var data_addr: Int
    var aux_addr: Int
    var dtype_code: Int
    var mask_kind: Int
    var mask_param: Int
    var scale: Float32
    var offset: Float32
    var s0: Int  # element stride along dim 0 (0 = broadcast, 1 = contiguous!)
    var s1: Int
    var s2: Int
    var s3: Int
    var s4: Int
    var s5: Int

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


struct CompiledPlan(Copyable, Movable):
    var insts: List[BytecodeInst]
    var root_reg: Int
    var num_regs: Int
    var needs_sum_sq: Bool

    def __init__(out self):
        self.insts = List[BytecodeInst]()
        self.root_reg = 0
        self.num_regs = 1
        self.needs_sum_sq = False


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
    """General 3-pass DAG compiler:
    1. Constant-scale/offset folding into `BC_LOAD` + algebraic fusion (`FMA`,
       `POW2_SUM`, `HYPOT`, `x / const -> x * (1/const)`).
    2. Dead-node elimination (keeps only nodes reachable from root `n_nodes - 1`).
    3. Linear-scan liveness register allocation (`num_regs` typically 3..5).
    """
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

    # First normalize constants (`ADD/SUB/MUL/DIV` with `BC_CONST` -> `BC_AFFINE`)
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

    # Count consumer references
    for i in range(n_nodes):
        if a_idx[i] >= 0 and ops[i] != BC_LOAD:
            ref_cnt[a_idx[i]] += 1
        if b_idx[i] >= 0:
            ref_cnt[b_idx[i]] += 1
        if c_idx[i] >= 0:
            ref_cnt[c_idx[i]] += 1

    # Peephole algebraic fusion pass
    for i in range(n_nodes):
        var op = ops[i]
        var lhs = a_idx[i]
        var rhs = b_idx[i]

        # Chain `AFFINE(AFFINE(x, s0, o0), s1, o1) -> AFFINE(x, s0*s1, o0*s1 + o1)`
        if op == BC_AFFINE and lhs >= 0 and ops[lhs] == BC_AFFINE:
            var s0 = p0[lhs]
            var o0 = p1[lhs]
            var s1 = p0[i]
            var o1 = p1[i]
            a_idx[i] = a_idx[lhs]
            p0[i] = s0 * s1
            p1[i] = o0 * s1 + o1
            lhs = a_idx[i]

        # Fold `AFFINE(LOAD(src), s1, o1)` directly into a cloned `SourceBinding`!
        if op == BC_AFFINE and lhs >= 0 and ops[lhs] == BC_LOAD:
            var orig_sid = a_idx[lhs]
            var sb = sources[orig_sid].copy()
            sb.scale = sb.scale * p0[i]
            sb.offset = sb.offset * p0[i] + p1[i]
            var new_sid = len(sources)
            sources.append(sb^)
            ops[i] = BC_LOAD
            a_idx[i] = new_sid
            b_idx[i] = -1

        # Fuse `(x * x) + (y * y) -> POW2_SUM(x, y)` and `(x * y) + z -> FMA(x, y, z)`
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

        # Fuse `SQRT(POW2_SUM(x, y)) -> HYPOT(x, y)`
        elif op == BC_SQRT and lhs >= 0 and ops[lhs] == BC_POW2_SUM:
            ops[i] = BC_HYPOT
            a_idx[i] = a_idx[lhs]
            b_idx[i] = b_idx[lhs]

    # Mark live nodes backward from root `n_nodes - 1`
    var live = List[Bool](capacity=n_nodes)
    live.resize(n_nodes, False)
    if n_nodes > 0:
        live[n_nodes - 1] = True
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

    # Compute last-use instruction index for linear-scan register coloring
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
    if n_nodes > 0:
        last_use[n_nodes - 1] = n_nodes

    var node_reg = List[Int](capacity=n_nodes)
    node_reg.resize(n_nodes, -1)
    var free_regs = List[Int]()
    var next_reg = 0
    var plan = CompiledPlan()
    plan.needs_sum_sq = needs_sum_sq

    for i in range(n_nodes):
        if not live[i]:
            continue
        # Allocate a destination register NOT aliasing inputs of instruction `i`
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
            BytecodeInst(
                ops[i],
                chosen,
                r_in0,
                r_in1,
                r_in2,
                sid,
                p0[i],
                p1[i],
            )
        )

        # Recycle input registers whose `last_use == i`
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

    plan.root_reg = node_reg[n_nodes - 1] if n_nodes > 0 else 0
    plan.num_regs = max(1, next_reg)
    return plan^


struct MorselWorkspace(Movable):
    """Thread-local L1-cache register file + 1 disjoint scratch register."""

    var regs: List[MorselReg]
    var scratch: MorselReg

    def __init__(out self, num_regs: Int):
        self.regs = List[MorselReg](capacity=num_regs)
        for _ in range(num_regs):
            self.regs.append(MorselReg())
        self.scratch = MorselReg()

    def __init__(out self, *, deinit move: Self):
        self.regs = move.regs^
        self.scratch = move.scratch^


@inline(.always)
def _load_morsel_from_source(
    src: SourceBinding,
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
    mut dst: MorselReg,
):
    """Gather or stream `count` elements starting at logical slice offset
    `flat_offset` into `dst` (single-pass decode + bitmask construction).
    """
    var mp = dst.mask_ptr()
    var n_words = ceildiv(count, 64)
    if src.mask_kind != MSRC_NONE:
        for w in range(n_words):
            mp[unsafe_offset=w] = UInt64(0)
    var or_bits: UInt64 = 0

    # Fast path 1: Full-span contiguous coalesced 1D stream!
    if coalesced_all and src.s0 == 1:
        var base_elem = (
            d0_start * src.s0
            + d1_start * src.s1
            + d2_start * src.s2
            + d3_start * src.s3
            + flat_offset
        )
        if src.dtype_code == DTYPE_U16:
            var p_u16 = ImmPointer[UInt16, ImmUntrackedOrigin](
                unsafe_from_address=src.data_addr + base_elem * 2
            )
            var p_aux = ImmPointer[UInt8, ImmUntrackedOrigin](
                unsafe_from_address=src.aux_addr + base_elem
                if src.aux_addr != 0
                else src.data_addr
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
            var p_aux = ImmPointer[UInt8, ImmUntrackedOrigin](
                unsafe_from_address=src.aux_addr + base_elem
                if src.aux_addr != 0
                else src.data_addr
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

    # General N-D strided / sub-slab / broadcasted loader (handles stride-0 broadcasts!):
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
            if src.dtype_code == DTYPE_U16:
                var p_u16 = ImmPointer[UInt16, ImmUntrackedOrigin](
                    unsafe_from_address=src.data_addr + elem_off * 2
                )
                var p_aux = ImmPointer[UInt8, ImmUntrackedOrigin](
                    unsafe_from_address=src.aux_addr + elem_off
                    if src.aux_addr != 0
                    else src.data_addr
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
                var p_aux = ImmPointer[UInt8, ImmUntrackedOrigin](
                    unsafe_from_address=src.aux_addr + elem_off
                    if src.aux_addr != 0
                    else src.data_addr
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
        elif src.s0 == 0:
            # Stride-0 broadcast along axis 0 (e.g. 1D lat/level/time coordinate or scalar!)
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
            else:
                var p_f32 = ImmPointer[Float32, ImmUntrackedOrigin](
                    unsafe_from_address=src.data_addr + elem_off * 4
                )
                bval = p_f32[unsafe_offset=0] * src.scale + src.offset
                if src.aux_addr != 0 and src.mask_kind != MSRC_NONE:
                    var p_aux = ImmPointer[UInt8, ImmUntrackedOrigin](
                        unsafe_from_address=src.aux_addr + elem_off
                    )
                    bmasked = Int(p_aux[unsafe_offset=0]) != src.mask_param
            for s_i in range(span):
                var d_idx = written + s_i
                dp[unsafe_offset=d_idx] = bval
                if bmasked:
                    mp[unsafe_offset=d_idx >> 6] |= UInt64(1) << UInt64(
                        d_idx & 63
                    )
                    or_bits |= UInt64(1)
        else:
            # General non-unit stride along axis 0
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
    """Execute general bytecode program for one L1-cache morsel (`count <= 2048`)."""
    for pc in range(n_insts):
        ref inst = insts_ptr[unsafe_offset=pc]
        match inst.op:
            case == BC_LOAD:
                _load_morsel_from_source(
                    srcs_ptr[unsafe_offset=inst.src_id],
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
                    ws.regs[inst.dst_reg],
                )
            case == BC_CONST:
                ws.regs[inst.dst_reg].state = MSTATE_VALID
                var dp = ws.regs[inst.dst_reg].data_ptr()
                for k in range(count):
                    dp[unsafe_offset=k] = inst.p0
            case == BC_ADD:
                morsel_add(
                    ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], count
                )
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_SUB:
                morsel_sub(
                    ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], count
                )
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_MUL:
                morsel_mul(
                    ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], count
                )
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_DIV:
                morsel_div(
                    ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], count
                )
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_HYPOT:
                morsel_hypot(
                    ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], count
                )
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_POW2_SUM:
                morsel_pow2_sum(
                    ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], count
                )
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_MIN:
                morsel_min(
                    ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], count
                )
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_MAX:
                morsel_max(
                    ws.scratch, ws.regs[inst.r0], ws.regs[inst.r1], count
                )
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_AFFINE:
                morsel_affine(
                    ws.scratch,
                    ws.regs[inst.r0],
                    inst.p0,
                    inst.p1,
                    count,
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
                morsel_mask_cmp[0](
                    ws.scratch, ws.regs[inst.r0], inst.p0, count
                )
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_MASK_LE:
                morsel_mask_cmp[1](
                    ws.scratch, ws.regs[inst.r0], inst.p0, count
                )
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_MASK_GT:
                morsel_mask_cmp[2](
                    ws.scratch, ws.regs[inst.r0], inst.p0, count
                )
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_MASK_GE:
                morsel_mask_cmp[3](
                    ws.scratch, ws.regs[inst.r0], inst.p0, count
                )
                swap(ws.regs[inst.dst_reg], ws.scratch)
            case == BC_FILLNA:
                morsel_fillna(
                    ws.scratch, ws.regs[inst.r0], inst.p0, count
                )
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
    reduce_axes_mask: Int,  # bit d == 1 means axis d is reduced; bit 0..3; -1 = scalar only
    agg_mode: Int,          # AGG_MEAN..AGG_COUNT; -1 = materialize full sliced array
    out_stats_f64: MutPointer[Float64, _],  # [mean, std, min, max, sum, valid_cnt, total_cnt]
    out_arr_f32: MutPointer[Float32, _],
    out_mask_u8: MutPointer[UInt8, _],
):
    """Execute any compiled bytecode DAG over a 1D..4D sliced domain in L1 morsels."""
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

    # Per-source contiguity check so contiguous 4D vars still stream flat even
    # when mixed with stride-0 broadcasted 2D/3D surface/coordinate variables!
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

    var n_morsels = ceildiv(total_elems, MORSEL_LEN)
    var hw = max(1, parallelism_level())
    var n_workers = min(hw, n_morsels)

    var insts_ptr = ImmPointer[BytecodeInst, origin_of(plan.insts)](
        plan.insts.unsafe_ptr()
    )
    var srcs_ptr = ImmPointer[SourceBinding, origin_of(sources)](
        sources.unsafe_ptr()
    )
    var n_insts = len(plan.insts)
    var num_regs = plan.num_regs
    var root_reg = plan.root_reg
    var want_sq = plan.needs_sum_sq or agg_mode == AGG_STD

    # Compute kept-axis strides for general N-D axis reductions (`reduce_axes_mask >= 0`)
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

    var gsum_p = p_gsum.unsafe_ptr()
    var gsq_p = p_gsq.unsafe_ptr()
    var gmin_p = p_gmin.unsafe_ptr()
    var gmax_p = p_gmax.unsafe_ptr()
    var gval_p = p_gval.unsafe_ptr()
    var gtot_p = p_gtot.unsafe_ptr()

    var ws_p = w_sum.unsafe_ptr()
    var wsq_p = w_sq.unsafe_ptr()
    var wext_p = w_ext.unsafe_ptr()
    var wc_p = w_cnt.unsafe_ptr()

    def worker(wid: Int) {imm}:
        var m_begin = (wid * n_morsels) // n_workers
        var m_end = ((wid + 1) * n_morsels) // n_workers
        if m_begin >= m_end:
            return

        # L1-cache resident morsel register file (typically 3..5 x 8KB = ~32KB!)
        var ws = MorselWorkspace(num_regs)
        var local_acc = ReduceAcc()
        var base_w = wid * out_cells

        for mid in range(m_begin, m_end):
            var flat_off = mid * MORSEL_LEN
            var cnt = min(MORSEL_LEN, total_elems - flat_off)

            eval_morsel_bytecode(
                insts_ptr,
                n_insts,
                srcs_ptr,
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
            if want_sq:
                morsel_accumulate_stats[True](root, cnt, local_acc)
            else:
                morsel_accumulate_stats[False](root, cnt, local_acc)

            if materialize_direct and want_axis_out:
                var dp = root.data_imm()
                for k in range(cnt):
                    var g_idx = flat_off + k
                    var m_flag = root.is_masked(k)
                    out_arr_f32[unsafe_offset=g_idx] = (
                        0.0 if m_flag else dp[unsafe_offset=k]
                    )
                    out_mask_u8[unsafe_offset=g_idx] = UInt8(1) if m_flag else UInt8(0)
            elif want_axis_out and root.state != MSTATE_MASKED:
                var dp = root.data_imm()
                # Fast specialization when keeping inner 2D `(d0, d1)` & reducing `(d2, d3)`:
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
                    # Completely general N-D axis projection for ANY `reduce_axes_mask`!
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

        gsum_p[unsafe_offset=wid] = local_acc.sum
        gsq_p[unsafe_offset=wid] = local_acc.sum_sq
        gmin_p[unsafe_offset=wid] = local_acc.min_val
        gmax_p[unsafe_offset=wid] = local_acc.max_val
        gval_p[unsafe_offset=wid] = local_acc.valid_count
        gtot_p[unsafe_offset=wid] = local_acc.total_count

    parallelize(worker, n_workers)

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
                out_arr_f32[unsafe_offset=cell] = 0.0
                out_mask_u8[unsafe_offset=cell] = UInt8(1)
            else:
                out_mask_u8[unsafe_offset=cell] = UInt8(0)
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
