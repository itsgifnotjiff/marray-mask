"""General 64-byte aligned L1-cache morsel registers and SIMD bitmask kernels.

Follows Modular `mojo-syntax` and `idiomatic-mojo` guidelines:
  - Hardware-queried `simd_width_of` + `align_of` + 64-byte `Alignment` `Allocation`
  - Unbound `_` origins on `MutPointer` / `ImmPointer` (no `AnyOrigin`)
  - 4-way `comptime for` unrolled vector loops to saturate dual AVX2/AVX-512 pipes
  - Single-pass raw source decoding + 64-bit `UInt64` bitmask construction
"""

from std.bit import pop_count
from std.math import (
    abs,
    align_up,
    ceildiv,
    cos,
    exp,
    fma,
    isnan,
    log,
    max,
    min,
    sin,
    sqrt,
)
from std.memory import (
    Alignment,
    Allocation,
    ImmPointer,
    Layout,
    MutPointer,
    alloc,
    dealloc,
)
from std.sys import align_of, simd_width_of


comptime W: Int = simd_width_of[DType.float32]()
comptime VEC_ALIGN: Int = align_of[SIMD[DType.float32, W]]()
comptime CACHE_ALIGN: Alignment = Alignment.of[SIMD[DType.float32, 16]]()

# 2048 float32s = 8 KB data + 256 B bitmask -> fits multiple regs in 32 KB L1d!
comptime MORSEL_LEN: Int = 2048
comptime MORSEL_WORDS: Int = MORSEL_LEN // 64
comptime UNROLL: Int = 4
comptime STEP_W: Int = W * UNROLL

comptime MSTATE_VALID: Int = 0
comptime MSTATE_MASKED: Int = 1
comptime MSTATE_BITS: Int = 2
comptime ALL_ONES: UInt64 = ~UInt64(0)

# General input dtype tags:
comptime DTYPE_F32: Int = 0
comptime DTYPE_F64: Int = 1
comptime DTYPE_U16: Int = 2
comptime DTYPE_I16: Int = 3
comptime DTYPE_I32: Int = 4
comptime DTYPE_U8: Int = 5

# General input mask encodings (CF-conventions / RPN / GRIB / NumPy ma):
comptime MSRC_NONE: Int = 0
comptime MSRC_SENTINEL_INT: Int = 1       # raw integer == sentinel_int
comptime MSRC_NAN: Int = 2                # IEEE NaN in float32/float64 source
comptime MSRC_BYTE_GT: Int = 3            # companion uint8 array > threshold_int
comptime MSRC_BYTE_NE: Int = 4            # companion uint8 array != threshold_int (e.g. numpy.ma bool!=0)
comptime MSRC_BITPACKED: Int = 5          # companion packed uint64 bitmask (1 = masked)
comptime MSRC_FST_BITMAP_U8: Int = 6      # fstd2nc / boreal-io @@ polarity: keep > 0, mask == 0
comptime MSRC_FST_BITMAP_WORDS: Int = 7   # packed UInt64 @@ bitmap: bit 1 = valid, bit 0 = masked (~word)
comptime MSRC_FST_BITMAP_F32: Int = 8     # boreal-io datyp=2 float32 @@ plane: > 0.0 valid, <= 0.0 masked

comptime W64: Int = simd_width_of[DType.float64]()
comptime VEC_ALIGN64: Int = align_of[SIMD[DType.float64, W64]]()


@inline(.always)
def _trailing_mask(n: Int) -> UInt64:
    var rem = n & 63
    if rem == 0:
        return ALL_ONES
    return (UInt64(1) << UInt64(rem)) - UInt64(1)


@inline(.always)
def _pack_bool8(m: SIMD[DType.bool, 8]) -> UInt64:
    comptime weights = SIMD[DType.uint8, 8](1, 2, 4, 8, 16, 32, 64, 128)
    return UInt64(m.select(weights, SIMD[DType.uint8, 8](0)).reduce_add())


@inline(.always)
def _unpack_bool8(byte_val: UInt8) -> SIMD[DType.bool, 8]:
    comptime lanes = SIMD[DType.uint8, 8](1, 2, 4, 8, 16, 32, 64, 128)
    return (SIMD[DType.uint8, 8](byte_val) & lanes).ne(0)


struct MorselReg(Movable):
    """64-byte cache-line aligned 2048-element L1 register (`Allocation`-owned)."""

    var _data_alloc: Allocation[Float32, alignment=CACHE_ALIGN]
    var _mask_alloc: Allocation[UInt64, alignment=CACHE_ALIGN]
    var state: Int

    def __init__(out self):
        self._data_alloc = alloc(
            Layout[Float32, alignment=CACHE_ALIGN](count=MORSEL_LEN)
        )
        self._mask_alloc = alloc(
            Layout[UInt64, alignment=CACHE_ALIGN](count=MORSEL_WORDS)
        )
        self.state = MSTATE_VALID
        var dp = self._data_alloc.unsafe_ptr()
        for i in range(0, MORSEL_LEN, W):
            dp.unsafe_store[alignment=VEC_ALIGN](i, SIMD[.float32, W](0.0))
        var mp = self._mask_alloc.unsafe_ptr()
        for w in range(MORSEL_WORDS):
            mp[unsafe_offset=w] = UInt64(0)

    def __init__(out self, *, deinit move: Self):
        self._data_alloc = move._data_alloc^
        self._mask_alloc = move._mask_alloc^
        self.state = move.state

    def __deinit__(deinit self):
        dealloc(self._data_alloc^)
        dealloc(self._mask_alloc^)

    @inline(.always)
    def data_ptr(mut self) -> type_of(self._data_alloc.unsafe_ptr()):
        return self._data_alloc.unsafe_ptr()

    @inline(.always)
    def data_imm(self) -> type_of(self._data_alloc.unsafe_ptr()):
        return self._data_alloc.unsafe_ptr()

    @inline(.always)
    def mask_ptr(mut self) -> type_of(self._mask_alloc.unsafe_ptr()):
        return self._mask_alloc.unsafe_ptr()

    @inline(.always)
    def mask_imm(self) -> type_of(self._mask_alloc.unsafe_ptr()):
        return self._mask_alloc.unsafe_ptr()

    @inline(.always)
    def is_masked(ref self, idx: Int) -> Bool:
        if self.state == MSTATE_VALID:
            return False
        if self.state == MSTATE_MASKED:
            return True
        var w = idx >> 6
        var b = idx & 63
        var word = self._mask_alloc.unsafe_ptr()[unsafe_offset=w]
        return ((word >> UInt64(b)) & UInt64(1)) != UInt64(0)

    def finalize_words(mut self, count: Int, or_acc: UInt64):
        if or_acc == UInt64(0) or count == 0:
            self.state = MSTATE_VALID
            return
        var n_words = ceildiv(count, 64)
        var tail = _trailing_mask(count)
        var mp = self._mask_alloc.unsafe_ptr()
        mp[unsafe_offset=n_words - 1] &= tail
        var any_m = False
        var all_m = True
        for w in range(n_words - 1):
            var word = mp[unsafe_offset=w]
            if word != UInt64(0):
                any_m = True
            if word != ALL_ONES:
                all_m = False
        var lw = mp[unsafe_offset=n_words - 1]
        if lw != UInt64(0):
            any_m = True
        if lw != tail:
            all_m = False
        if not any_m:
            self.state = MSTATE_VALID
        elif all_m:
            self.state = MSTATE_MASKED
        else:
            self.state = MSTATE_BITS


# ===----------------------------------------------------------------------=== #
# Single-Pass Contiguous Span Decoders Into `MorselReg`
# ===----------------------------------------------------------------------=== #


@inline(.always)
def load_span_u16(
    src_u16: ImmPointer[UInt16, _],
    aux_u8: ImmPointer[UInt8, _],
    dst_off: Int,
    span_len: Int,
    scale: Float32,
    offset: Float32,
    mask_kind: Int,
    mask_param: Int,
    mut reg: MorselReg,
    mut or_bits: UInt64,
):
    """Single-pass decode of `span_len` UInt16 elements into `reg` starting at `dst_off`."""
    var dp = reg.data_ptr()
    var mp = reg.mask_ptr()
    var sv = SIMD[.float32, 8](scale)
    var ov = SIMD[.float32, 8](offset)
    var zero_f = SIMD[.float32, 8](0.0)
    var sent_v = SIMD[.uint16, 8](UInt16(mask_param))
    var qmax_v = SIMD[.uint8, 8](UInt8(mask_param))

    var k = 0
    var vec_end = (span_len // 8) * 8
    if mask_kind == MSRC_NONE:
        while k < vec_end:
            var q = src_u16.unsafe_load[width=8](k)
            dp.unsafe_store(dst_off + k, fma(q.cast[.float32](), sv, ov))
            k += 8
        while k < span_len:
            dp[unsafe_offset=dst_off + k] = fma(
                Float32(src_u16[unsafe_offset=k]), scale, offset
            )
            k += 1
        return

    if mask_kind == MSRC_SENTINEL_INT:
        while k < vec_end:
            var q = src_u16.unsafe_load[width=8](k)
            var is_m = q.eq(sent_v)
            var val = fma(q.cast[.float32](), sv, ov)
            dp.unsafe_store(dst_off + k, is_m.select(zero_f, val))
            var bits8 = _pack_bool8(is_m)
            if bits8 != UInt64(0):
                or_bits |= bits8
                var d_idx = dst_off + k
                var w = d_idx >> 6
                var b = d_idx & 63
                mp[unsafe_offset=w] |= bits8 << UInt64(b)
                if b > 56:
                    mp[unsafe_offset=w + 1] |= bits8 >> UInt64(64 - b)
            k += 8
        while k < span_len:
            var qs = src_u16[unsafe_offset=k]
            var d_idx = dst_off + k
            if Int(qs) == mask_param:
                dp[unsafe_offset=d_idx] = 0.0
                var bit = UInt64(1) << UInt64(d_idx & 63)
                mp[unsafe_offset=d_idx >> 6] |= bit
                or_bits |= UInt64(1)
            else:
                dp[unsafe_offset=d_idx] = fma(Float32(qs), scale, offset)
            k += 1
        return

    # MSRC_BYTE_GT, MSRC_BYTE_NE, or MSRC_FST_BITMAP_U8 (`== 0` masked):
    var is_gt = mask_kind == MSRC_BYTE_GT
    var is_fst_zero = mask_kind == MSRC_FST_BITMAP_U8
    var zero_u8 = SIMD[.uint8, 8](0)
    while k < vec_end:
        var q = src_u16.unsafe_load[width=8](k)
        var qf = aux_u8.unsafe_load[width=8](k)
        var is_m = qf.eq(zero_u8) if is_fst_zero else (
            qf.gt(qmax_v) if is_gt else qf.ne(qmax_v)
        )
        var val = fma(q.cast[.float32](), sv, ov)
        dp.unsafe_store(dst_off + k, is_m.select(zero_f, val))
        var bits8 = _pack_bool8(is_m)
        if bits8 != UInt64(0):
            or_bits |= bits8
            var d_idx = dst_off + k
            var w = d_idx >> 6
            var b = d_idx & 63
            mp[unsafe_offset=w] |= bits8 << UInt64(b)
            if b > 56:
                mp[unsafe_offset=w + 1] |= bits8 >> UInt64(64 - b)
        k += 8
    while k < span_len:
        var d_idx = dst_off + k
        var qfs = Int(aux_u8[unsafe_offset=k])
        var bad = (qfs == 0) if is_fst_zero else (
            (qfs > mask_param) if is_gt else (qfs != mask_param)
        )
        if bad:
            dp[unsafe_offset=d_idx] = 0.0
            mp[unsafe_offset=d_idx >> 6] |= UInt64(1) << UInt64(d_idx & 63)
            or_bits |= UInt64(1)
        else:
            dp[unsafe_offset=d_idx] = fma(
                Float32(src_u16[unsafe_offset=k]), scale, offset
            )
        k += 1


@inline(.always)
def load_span_f32(
    src_f32: ImmPointer[Float32, _],
    aux_u8: ImmPointer[UInt8, _],
    dst_off: Int,
    span_len: Int,
    scale: Float32,
    offset: Float32,
    mask_kind: Int,
    mask_param: Int,
    mut reg: MorselReg,
    mut or_bits: UInt64,
):
    """Single-pass decode of `span_len` Float32 elements (unscaled or CF-scaled,
    supporting dense, NaN-masked, or companion byte/bitmask arrays).
    """
    var dp = reg.data_ptr()
    var mp = reg.mask_ptr()
    var is_identity = scale == Float32(1.0) and offset == Float32(0.0)
    var sv = SIMD[.float32, 8](scale)
    var ov = SIMD[.float32, 8](offset)
    var zero_f = SIMD[.float32, 8](0.0)
    var q_v = SIMD[.uint8, 8](UInt8(mask_param))

    var k = 0
    var vec_end = (span_len // 8) * 8

    if mask_kind == MSRC_NONE:
        if is_identity:
            while k < vec_end:
                dp.unsafe_store(dst_off + k, src_f32.unsafe_load[width=8](k))
                k += 8
            while k < span_len:
                dp[unsafe_offset=dst_off + k] = src_f32[unsafe_offset=k]
                k += 1
        else:
            while k < vec_end:
                dp.unsafe_store(
                    dst_off + k, fma(src_f32.unsafe_load[width=8](k), sv, ov)
                )
                k += 8
            while k < span_len:
                dp[unsafe_offset=dst_off + k] = fma(
                    src_f32[unsafe_offset=k], scale, offset
                )
                k += 1
        return

    if mask_kind == MSRC_NAN:
        while k < vec_end:
            var raw = src_f32.unsafe_load[width=8](k)
            var is_m = isnan(raw)
            var val = raw if is_identity else fma(raw, sv, ov)
            dp.unsafe_store(dst_off + k, is_m.select(zero_f, val))
            var bits8 = _pack_bool8(is_m)
            if bits8 != UInt64(0):
                or_bits |= bits8
                var d_idx = dst_off + k
                var w = d_idx >> 6
                var b = d_idx & 63
                mp[unsafe_offset=w] |= bits8 << UInt64(b)
                if b > 56:
                    mp[unsafe_offset=w + 1] |= bits8 >> UInt64(64 - b)
            k += 8
        while k < span_len:
            var rv = src_f32[unsafe_offset=k]
            var d_idx = dst_off + k
            if isnan(rv):
                dp[unsafe_offset=d_idx] = 0.0
                mp[unsafe_offset=d_idx >> 6] |= UInt64(1) << UInt64(d_idx & 63)
                or_bits |= UInt64(1)
            else:
                dp[unsafe_offset=d_idx] = rv if is_identity else fma(
                    rv, scale, offset
                )
            k += 1
        return

    if mask_kind == MSRC_FST_BITMAP_F32:
        # Native boreal-io `compact_integer_unpack_f32` (`datyp=2, nbits=1`) float32 0/1 plane:
        # keep > 0.0, mask <= 0.0 or NaN (fstd2nc polarity)
        var aux_f32 = aux_u8.unsafe_bitcast[Float32]()
        while k < vec_end:
            var raw = src_f32.unsafe_load[width=8](k)
            var bf = aux_f32.unsafe_load[width=8](k)
            var is_valid = bf.gt(zero_f)
            var is_m = ~is_valid
            var val = raw if is_identity else fma(raw, sv, ov)
            dp.unsafe_store(dst_off + k, is_valid.select(val, zero_f))
            var bits8 = _pack_bool8(is_m)
            if bits8 != UInt64(0):
                or_bits |= bits8
                var d_idx = dst_off + k
                var w = d_idx >> 6
                var b = d_idx & 63
                mp[unsafe_offset=w] |= bits8 << UInt64(b)
                if b > 56:
                    mp[unsafe_offset=w + 1] |= bits8 >> UInt64(64 - b)
            k += 8
        while k < span_len:
            var rv = src_f32[unsafe_offset=k]
            var bfv = aux_f32[unsafe_offset=k]
            var d_idx = dst_off + k
            if not (bfv > Float32(0.0)):
                dp[unsafe_offset=d_idx] = 0.0
                mp[unsafe_offset=d_idx >> 6] |= UInt64(1) << UInt64(d_idx & 63)
                or_bits |= UInt64(1)
            else:
                dp[unsafe_offset=d_idx] = rv if is_identity else fma(
                    rv, scale, offset
                )
            k += 1
        return

    # Companion byte mask (`MSRC_BYTE_GT`, `MSRC_BYTE_NE`, or `MSRC_FST_BITMAP_U8` where 0=masked)
    var is_gt = mask_kind == MSRC_BYTE_GT
    var is_fst_zero = mask_kind == MSRC_FST_BITMAP_U8
    var zero_u8 = SIMD[.uint8, 8](0)
    while k < vec_end:
        var raw = src_f32.unsafe_load[width=8](k)
        var qf = aux_u8.unsafe_load[width=8](k)
        var is_m = qf.eq(zero_u8) if is_fst_zero else (
            qf.gt(q_v) if is_gt else qf.ne(q_v)
        )
        var val = raw if is_identity else fma(raw, sv, ov)
        dp.unsafe_store(dst_off + k, is_m.select(zero_f, val))
        var bits8 = _pack_bool8(is_m)
        if bits8 != UInt64(0):
            or_bits |= bits8
            var d_idx = dst_off + k
            var w = d_idx >> 6
            var b = d_idx & 63
            mp[unsafe_offset=w] |= bits8 << UInt64(b)
            if b > 56:
                mp[unsafe_offset=w + 1] |= bits8 >> UInt64(64 - b)
        k += 8
    while k < span_len:
        var rv = src_f32[unsafe_offset=k]
        var d_idx = dst_off + k
        var qfs = Int(aux_u8[unsafe_offset=k])
        var bad = (qfs == 0) if is_fst_zero else (
            (qfs > mask_param) if is_gt else (qfs != mask_param)
        )
        if bad:
            dp[unsafe_offset=d_idx] = 0.0
            mp[unsafe_offset=d_idx >> 6] |= UInt64(1) << UInt64(d_idx & 63)
            or_bits |= UInt64(1)
        else:
            dp[unsafe_offset=d_idx] = rv if is_identity else fma(
                rv, scale, offset
            )
        k += 1


# ===----------------------------------------------------------------------=== #
# 64-Byte Aligned, 4-Way Unrolled Morsel Kernels & Mask Propagation
# ===----------------------------------------------------------------------=== #


@inline(.always)
def _prop_unary_mask(mut dst: MorselReg, a: MorselReg, count: Int):
    dst.state = a.state
    if a.state == MSTATE_BITS:
        var n_words = ceildiv(count, 64)
        var dp = dst.mask_ptr()
        var ap = a.mask_imm()
        for w in range(n_words):
            dp[unsafe_offset=w] = ap[unsafe_offset=w]


@inline(.always)
def _prop_binary_mask(
    mut dst: MorselReg, a: MorselReg, b: MorselReg, count: Int
):
    if a.state == MSTATE_MASKED or b.state == MSTATE_MASKED:
        dst.state = MSTATE_MASKED
        return
    if a.state == MSTATE_VALID and b.state == MSTATE_VALID:
        dst.state = MSTATE_VALID
        return
    var n_words = ceildiv(count, 64)
    var dp = dst.mask_ptr()
    if a.state == MSTATE_BITS and b.state == MSTATE_BITS:
        var ap = a.mask_imm()
        var bp = b.mask_imm()
        var or_w: UInt64 = 0
        for w in range(n_words):
            var u = ap[unsafe_offset=w] | bp[unsafe_offset=w]
            dp[unsafe_offset=w] = u
            or_w |= u
        dst.state = MSTATE_BITS
        dst.finalize_words(count, or_w)
    elif a.state == MSTATE_BITS:
        var ap = a.mask_imm()
        for w in range(n_words):
            dp[unsafe_offset=w] = ap[unsafe_offset=w]
        dst.state = MSTATE_BITS
    else:
        var bp = b.mask_imm()
        for w in range(n_words):
            dp[unsafe_offset=w] = bp[unsafe_offset=w]
        dst.state = MSTATE_BITS


@inline(.always)
def morsel_add(mut dst: MorselReg, a: MorselReg, b: MorselReg, count: Int):
    _prop_binary_mask(dst, a, b, count)
    if dst.state == MSTATE_MASKED:
        return
    var n_vec = align_up(count, STEP_W)
    var dp = dst.data_ptr()
    var ap = a.data_imm()
    var bp = b.data_imm()
    for base in range(0, n_vec, STEP_W):
        comptime for u in range(UNROLL):
            var off = base + u * W
            dp.unsafe_store[alignment=VEC_ALIGN](
                off,
                ap.unsafe_load[width=W, alignment=VEC_ALIGN](off)
                + bp.unsafe_load[width=W, alignment=VEC_ALIGN](off),
            )


@inline(.always)
def morsel_sub(mut dst: MorselReg, a: MorselReg, b: MorselReg, count: Int):
    _prop_binary_mask(dst, a, b, count)
    if dst.state == MSTATE_MASKED:
        return
    var n_vec = align_up(count, STEP_W)
    var dp = dst.data_ptr()
    var ap = a.data_imm()
    var bp = b.data_imm()
    for base in range(0, n_vec, STEP_W):
        comptime for u in range(UNROLL):
            var off = base + u * W
            dp.unsafe_store[alignment=VEC_ALIGN](
                off,
                ap.unsafe_load[width=W, alignment=VEC_ALIGN](off)
                - bp.unsafe_load[width=W, alignment=VEC_ALIGN](off),
            )


@inline(.always)
def morsel_mul(mut dst: MorselReg, a: MorselReg, b: MorselReg, count: Int):
    _prop_binary_mask(dst, a, b, count)
    if dst.state == MSTATE_MASKED:
        return
    var n_vec = align_up(count, STEP_W)
    var dp = dst.data_ptr()
    var ap = a.data_imm()
    var bp = b.data_imm()
    for base in range(0, n_vec, STEP_W):
        comptime for u in range(UNROLL):
            var off = base + u * W
            dp.unsafe_store[alignment=VEC_ALIGN](
                off,
                ap.unsafe_load[width=W, alignment=VEC_ALIGN](off)
                * bp.unsafe_load[width=W, alignment=VEC_ALIGN](off),
            )


@inline(.always)
def morsel_fma(
    mut dst: MorselReg, a: MorselReg, b: MorselReg, c: MorselReg, count: Int
):
    _prop_binary_mask(dst, a, b, count)
    if dst.state == MSTATE_MASKED or c.state == MSTATE_MASKED:
        dst.state = MSTATE_MASKED
        return
    if c.state == MSTATE_BITS:
        var n_words = ceildiv(count, 64)
        var dp_m = dst.mask_ptr()
        var cp_m = c.mask_imm()
        if dst.state == MSTATE_VALID:
            for w in range(n_words):
                dp_m[unsafe_offset=w] = cp_m[unsafe_offset=w]
            dst.state = MSTATE_BITS
        else:
            for w in range(n_words):
                dp_m[unsafe_offset=w] |= cp_m[unsafe_offset=w]

    var n_vec = align_up(count, STEP_W)
    var dp = dst.data_ptr()
    var ap = a.data_imm()
    var bp = b.data_imm()
    var cp = c.data_imm()
    for base in range(0, n_vec, STEP_W):
        comptime for u in range(UNROLL):
            var off = base + u * W
            dp.unsafe_store[alignment=VEC_ALIGN](
                off,
                fma(
                    ap.unsafe_load[width=W, alignment=VEC_ALIGN](off),
                    bp.unsafe_load[width=W, alignment=VEC_ALIGN](off),
                    cp.unsafe_load[width=W, alignment=VEC_ALIGN](off),
                ),
            )


@inline(.always)
def morsel_div(mut dst: MorselReg, a: MorselReg, b: MorselReg, count: Int):
    _prop_binary_mask(dst, a, b, count)
    if dst.state == MSTATE_MASKED:
        return
    var n_vec = align_up(count, STEP_W)
    var dp = dst.data_ptr()
    var ap = a.data_imm()
    var bp = b.data_imm()
    var eps = SIMD[.float32, W](1.0e-12)
    var ones = SIMD[.float32, W](1.0)
    var zeros = SIMD[.float32, W](0.0)
    var any_zero = False

    for base in range(0, n_vec, STEP_W):
        comptime for u in range(UNROLL):
            var off = base + u * W
            var av = ap.unsafe_load[width=W, alignment=VEC_ALIGN](off)
            var bv = bp.unsafe_load[width=W, alignment=VEC_ALIGN](off)
            var bad = abs(bv).lt(eps)
            dp.unsafe_store[alignment=VEC_ALIGN](
                off, bad.select(zeros, av / bad.select(ones, bv))
            )
            if bad.reduce_or():
                any_zero = True

    if any_zero:
        var n_words = ceildiv(count, 64)
        var mp = dst.mask_ptr()
        if dst.state == MSTATE_VALID:
            for w in range(n_words):
                mp[unsafe_offset=w] = UInt64(0)
        var or_acc: UInt64 = 0
        for w in range(n_words):
            var base = w * 64
            var wbits = mp[unsafe_offset=w]
            comptime for g in range(8):
                var off = base + g * 8
                if off < count:
                    var bv8 = bp.unsafe_load[width=8](off)
                    wbits |= _pack_bool8(abs(bv8).lt(SIMD[.float32, 8](1.0e-12))) << UInt64(g * 8)
            mp[unsafe_offset=w] = wbits
            or_acc |= wbits
        dst.state = MSTATE_BITS
        dst.finalize_words(count, or_acc)


@inline(.always)
def morsel_hypot(mut dst: MorselReg, a: MorselReg, b: MorselReg, count: Int):
    _prop_binary_mask(dst, a, b, count)
    if dst.state == MSTATE_MASKED:
        return
    var n_vec = align_up(count, STEP_W)
    var dp = dst.data_ptr()
    var ap = a.data_imm()
    var bp = b.data_imm()
    for base in range(0, n_vec, STEP_W):
        comptime for u in range(UNROLL):
            var off = base + u * W
            var av = ap.unsafe_load[width=W, alignment=VEC_ALIGN](off)
            var bv = bp.unsafe_load[width=W, alignment=VEC_ALIGN](off)
            dp.unsafe_store[alignment=VEC_ALIGN](
                off, sqrt(fma(av, av, bv * bv))
            )


@inline(.always)
def morsel_pow2_sum(
    mut dst: MorselReg, a: MorselReg, b: MorselReg, count: Int
):
    _prop_binary_mask(dst, a, b, count)
    if dst.state == MSTATE_MASKED:
        return
    var n_vec = align_up(count, STEP_W)
    var dp = dst.data_ptr()
    var ap = a.data_imm()
    var bp = b.data_imm()
    for base in range(0, n_vec, STEP_W):
        comptime for u in range(UNROLL):
            var off = base + u * W
            var av = ap.unsafe_load[width=W, alignment=VEC_ALIGN](off)
            var bv = bp.unsafe_load[width=W, alignment=VEC_ALIGN](off)
            dp.unsafe_store[alignment=VEC_ALIGN](off, fma(av, av, bv * bv))


@inline(.always)
def morsel_min(mut dst: MorselReg, a: MorselReg, b: MorselReg, count: Int):
    _prop_binary_mask(dst, a, b, count)
    if dst.state == MSTATE_MASKED:
        return
    var n_vec = align_up(count, STEP_W)
    var dp = dst.data_ptr()
    var ap = a.data_imm()
    var bp = b.data_imm()
    for base in range(0, n_vec, STEP_W):
        comptime for u in range(UNROLL):
            var off = base + u * W
            dp.unsafe_store[alignment=VEC_ALIGN](
                off,
                min(
                    ap.unsafe_load[width=W, alignment=VEC_ALIGN](off),
                    bp.unsafe_load[width=W, alignment=VEC_ALIGN](off),
                ),
            )


@inline(.always)
def morsel_max(mut dst: MorselReg, a: MorselReg, b: MorselReg, count: Int):
    _prop_binary_mask(dst, a, b, count)
    if dst.state == MSTATE_MASKED:
        return
    var n_vec = align_up(count, STEP_W)
    var dp = dst.data_ptr()
    var ap = a.data_imm()
    var bp = b.data_imm()
    for base in range(0, n_vec, STEP_W):
        comptime for u in range(UNROLL):
            var off = base + u * W
            dp.unsafe_store[alignment=VEC_ALIGN](
                off,
                max(
                    ap.unsafe_load[width=W, alignment=VEC_ALIGN](off),
                    bp.unsafe_load[width=W, alignment=VEC_ALIGN](off),
                ),
            )


@inline(.always)
def morsel_affine(
    mut dst: MorselReg,
    a: MorselReg,
    scale: Float32,
    offset: Float32,
    count: Int,
):
    _prop_unary_mask(dst, a, count)
    if dst.state == MSTATE_MASKED:
        return
    var n_vec = align_up(count, STEP_W)
    var dp = dst.data_ptr()
    var ap = a.data_imm()
    var sv = SIMD[.float32, W](scale)
    var ov = SIMD[.float32, W](offset)
    for base in range(0, n_vec, STEP_W):
        comptime for u in range(UNROLL):
            var off = base + u * W
            dp.unsafe_store[alignment=VEC_ALIGN](
                off, fma(ap.unsafe_load[width=W, alignment=VEC_ALIGN](off), sv, ov)
            )


@inline(.always)
def morsel_unary_math[
    op_tag: Int
](mut dst: MorselReg, a: MorselReg, count: Int):
    """Compile-time specialized unary math (`0=neg, 1=abs, 2=sqrt, 3=exp, 4=log, 5=sin, 6=cos`)."""
    _prop_unary_mask(dst, a, count)
    if dst.state == MSTATE_MASKED:
        return
    var n_vec = align_up(count, W)
    var dp = dst.data_ptr()
    var ap = a.data_imm()
    var zeros = SIMD[.float32, W](0.0)
    var ones = SIMD[.float32, W](1.0)
    var domain_violation = False

    for off in range(0, n_vec, W):
        var av = ap.unsafe_load[width=W, alignment=VEC_ALIGN](off)
        comptime match op_tag:
            case 0:
                dp.unsafe_store[alignment=VEC_ALIGN](off, -av)
            case 1:
                dp.unsafe_store[alignment=VEC_ALIGN](off, abs(av))
            case 2:
                var neg = av.lt(zeros)
                dp.unsafe_store[alignment=VEC_ALIGN](
                    off, sqrt(neg.select(zeros, av))
                )
                if neg.reduce_or():
                    domain_violation = True
            case 3:
                dp.unsafe_store[alignment=VEC_ALIGN](off, exp(av))
            case 4:
                var non_pos = av.le(zeros)
                dp.unsafe_store[alignment=VEC_ALIGN](
                    off, non_pos.select(zeros, log(non_pos.select(ones, av)))
                )
                if non_pos.reduce_or():
                    domain_violation = True
            case 5:
                dp.unsafe_store[alignment=VEC_ALIGN](off, sin(av))
            case 6:
                dp.unsafe_store[alignment=VEC_ALIGN](off, cos(av))
            case _:
                dp.unsafe_store[alignment=VEC_ALIGN](off, av)

    comptime if op_tag == 2 or op_tag == 4:
        if domain_violation:
            var n_words = ceildiv(count, 64)
            var mp = dst.mask_ptr()
            if dst.state == MSTATE_VALID:
                for w in range(n_words):
                    mp[unsafe_offset=w] = UInt64(0)
            var or_acc: UInt64 = 0
            for w in range(n_words):
                var base = w * 64
                var wbits = mp[unsafe_offset=w]
                comptime for g in range(8):
                    var off = base + g * 8
                    if off < count:
                        var v8 = ap.unsafe_load[width=8](off)
                        var bad = v8.lt(SIMD[.float32, 8](0.0)) if op_tag == 2 else v8.le(SIMD[.float32, 8](0.0))
                        wbits |= _pack_bool8(bad) << UInt64(g * 8)
                mp[unsafe_offset=w] = wbits
                or_acc |= wbits
            dst.state = MSTATE_BITS
            dst.finalize_words(count, or_acc)


@inline(.always)
def morsel_mask_cmp[
    cmp_tag: Int
](mut dst: MorselReg, a: MorselReg, threshold: Float32, count: Int):
    """Mask elements satisfying predicate (`0: <, 1: <=, 2: >, 3: >=`)."""
    if a.state == MSTATE_MASKED:
        dst.state = MSTATE_MASKED
        return
    var n_vec = align_up(count, STEP_W)
    var dp = dst.data_ptr()
    var ap = a.data_imm()
    for base in range(0, n_vec, STEP_W):
        comptime for u in range(UNROLL):
            var off = base + u * W
            dp.unsafe_store[alignment=VEC_ALIGN](
                off, ap.unsafe_load[width=W, alignment=VEC_ALIGN](off)
            )

    var n_words = ceildiv(count, 64)
    var mp = dst.mask_ptr()
    var has_prior = a.state == MSTATE_BITS
    var ap_m = a.mask_imm()
    var tv = SIMD[.float32, 8](threshold)
    var or_acc: UInt64 = 0

    for w in range(n_words):
        var base = w * 64
        var wbits: UInt64 = ap_m[unsafe_offset=w] if has_prior else UInt64(0)
        comptime for g in range(8):
            var off = base + g * 8
            if off < count:
                var v8 = ap.unsafe_load[width=8](off)
                comptime match cmp_tag:
                    case 0:
                        wbits |= _pack_bool8(v8.lt(tv)) << UInt64(g * 8)
                    case 1:
                        wbits |= _pack_bool8(v8.le(tv)) << UInt64(g * 8)
                    case 2:
                        wbits |= _pack_bool8(v8.gt(tv)) << UInt64(g * 8)
                    case _:
                        wbits |= _pack_bool8(v8.ge(tv)) << UInt64(g * 8)
        mp[unsafe_offset=w] = wbits
        or_acc |= wbits

    dst.state = MSTATE_BITS
    dst.finalize_words(count, or_acc)


@inline(.always)
def morsel_fillna(
    mut dst: MorselReg, a: MorselReg, fill_val: Float32, count: Int
):
    dst.state = MSTATE_VALID
    var dp = dst.data_ptr()
    var ap = a.data_imm()
    var fv = SIMD[.float32, 8](fill_val)

    if a.state == MSTATE_VALID:
        var n_vec = align_up(count, STEP_W)
        for base in range(0, n_vec, STEP_W):
            comptime for u in range(UNROLL):
                var off = base + u * W
                dp.unsafe_store[alignment=VEC_ALIGN](
                    off, ap.unsafe_load[width=W, alignment=VEC_ALIGN](off)
                )
        return

    if a.state == MSTATE_MASKED:
        var n_vec = align_up(count, W)
        var fv_w = SIMD[.float32, W](fill_val)
        for off in range(0, n_vec, W):
            dp.unsafe_store[alignment=VEC_ALIGN](off, fv_w)
        return

    var n_words = ceildiv(count, 64)
    var mp = a.mask_imm()
    for w in range(n_words):
        var word = mp[unsafe_offset=w]
        var base = w * 64
        if word == UInt64(0):
            comptime for g in range(8):
                var off = base + g * 8
                dp.unsafe_store(off, ap.unsafe_load[width=8](off))
        elif word == ALL_ONES:
            comptime for g in range(8):
                var off = base + g * 8
                dp.unsafe_store(off, fv)
        else:
            comptime for g in range(8):
                var off = base + g * 8
                var m8 = _unpack_bool8(UInt8((word >> UInt64(g * 8)) & UInt64(0xFF)))
                dp.unsafe_store(off, m8.select(fv, ap.unsafe_load[width=8](off)))


# ===----------------------------------------------------------------------=== #
# 4-Way Unrolled Morsel Reductions (Sum, SumSq, Min, Max, ValidCount)
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct ReduceAcc(Copyable, Movable):
    var sum: Float64
    var sum_sq: Float64
    var min_val: Float32
    var max_val: Float32
    var valid_count: Int
    var total_count: Int

    def __init__(out self):
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_val = Float32(1.0e30)
        self.max_val = Float32(-1.0e30)
        self.valid_count = 0
        self.total_count = 0

    def merge(mut self, other: Self):
        self.sum += other.sum
        self.sum_sq += other.sum_sq
        if other.valid_count > 0:
            self.min_val = min(self.min_val, other.min_val)
            self.max_val = max(self.max_val, other.max_val)
        self.valid_count += other.valid_count
        self.total_count += other.total_count

    def mean(self) -> Float64:
        if self.valid_count == 0:
            return 0.0 / 0.0
        return self.sum / Float64(self.valid_count)

    def variance(self) -> Float64:
        if self.valid_count == 0:
            return 0.0 / 0.0
        var m = self.mean()
        var v = (self.sum_sq / Float64(self.valid_count)) - m * m
        return max(0.0, v)

    def std(self) -> Float64:
        return sqrt(self.variance())


@inline(.always)
def morsel_accumulate_stats[
    need_sum_sq: Bool = False
](reg: MorselReg, count: Int, mut acc: ReduceAcc):
    """4-way unrolled AVX2/AVX-512 reduction with 64-element bitmask word skipping."""
    acc.total_count += count
    if count == 0 or reg.state == MSTATE_MASKED:
        return

    var dp = reg.data_imm()
    var pinf = SIMD[.float32, 8](1.0e30)
    var ninf = SIMD[.float32, 8](-1.0e30)
    var zeros = SIMD[.float32, 8](0.0)

    if reg.state == MSTATE_VALID:
        var s0 = SIMD[.float64, 8](0.0)
        var s1 = SIMD[.float64, 8](0.0)
        var sq0 = SIMD[.float64, 8](0.0)
        var sq1 = SIMD[.float64, 8](0.0)
        var mn0 = pinf
        var mn1 = pinf
        var mx0 = ninf
        var mx1 = ninf

        var pair_end = (count // 16) * 16
        var idx = 0
        while idx < pair_end:
            var v0 = dp.unsafe_load[width=8](idx)
            var v1 = dp.unsafe_load[width=8](idx + 8)
            var d0 = v0.cast[.float64]()
            var d1 = v1.cast[.float64]()
            s0 += d0
            s1 += d1
            comptime if need_sum_sq:
                sq0 = fma(d0, d0, sq0)
                sq1 = fma(d1, d1, sq1)
            mn0 = min(mn0, v0)
            mn1 = min(mn1, v1)
            mx0 = max(mx0, v0)
            mx1 = max(mx1, v1)
            idx += 16

        var s_tot = (s0 + s1).reduce_add()
        var sq_tot = (sq0 + sq1).reduce_add()
        var mn_tot = min(mn0, mn1).reduce_min()
        var mx_tot = max(mx0, mx1).reduce_max()

        while idx < count:
            var sv = dp[unsafe_offset=idx]
            var dv = Float64(sv)
            s_tot += dv
            comptime if need_sum_sq:
                sq_tot = fma(dv, dv, sq_tot)
            mn_tot = min(mn_tot, sv)
            mx_tot = max(mx_tot, sv)
            idx += 1

        acc.sum += s_tot
        acc.sum_sq += sq_tot
        acc.min_val = min(acc.min_val, mn_tot)
        acc.max_val = max(acc.max_val, mx_tot)
        acc.valid_count += count
        return

    # MSTATE_BITS: 64-element word-skipping SIMD reduction
    var s0 = SIMD[.float64, 8](0.0)
    var s1 = SIMD[.float64, 8](0.0)
    var sq0 = SIMD[.float64, 8](0.0)
    var mn0 = pinf
    var mx0 = ninf
    var valid_cnt = 0

    var n_words = ceildiv(count, 64)
    var mp = reg.mask_imm()
    for w in range(n_words):
        var word = mp[unsafe_offset=w]
        if w == n_words - 1 and (count & 63) != 0:
            word |= ~_trailing_mask(count)

        if word == ALL_ONES:
            continue
        elif word == UInt64(0):
            var base = w * 64
            comptime for g in range(0, 8, 2):
                var v0 = dp.unsafe_load[width=8](base + g * 8)
                var v1 = dp.unsafe_load[width=8](base + (g + 1) * 8)
                var d0 = v0.cast[.float64]()
                var d1 = v1.cast[.float64]()
                s0 += d0
                s1 += d1
                comptime if need_sum_sq:
                    sq0 = fma(d0, d0, sq0)
                    sq0 = fma(d1, d1, sq0)
                mn0 = min(mn0, min(v0, v1))
                mx0 = max(mx0, max(v0, v1))
            valid_cnt += 64
        else:
            var base = w * 64
            valid_cnt += 64 - Int(pop_count(word))
            comptime for g in range(8):
                var bm = UInt8((word >> UInt64(g * 8)) & UInt64(0xFF))
                if bm != UInt8(0xFF):
                    var v = dp.unsafe_load[width=8](base + g * 8)
                    if bm == UInt8(0):
                        var dv = v.cast[.float64]()
                        s0 += dv
                        comptime if need_sum_sq:
                            sq0 = fma(dv, dv, sq0)
                        mn0 = min(mn0, v)
                        mx0 = max(mx0, v)
                    else:
                        var mv = _unpack_bool8(bm)
                        var vs = mv.select(zeros, v).cast[.float64]()
                        s0 += vs
                        comptime if need_sum_sq:
                            sq0 = fma(vs, vs, sq0)
                        mn0 = min(mn0, mv.select(pinf, v))
                        mx0 = max(mx0, mv.select(ninf, v))

    if valid_cnt > 0:
        acc.sum += (s0 + s1).reduce_add()
        acc.sum_sq += sq0.reduce_add()
        acc.min_val = min(acc.min_val, mn0.reduce_min())
        acc.max_val = max(acc.max_val, mx0.reduce_max())
        acc.valid_count += valid_cnt


@inline(.always)
def morsel_accumulate_weighted(
    val_reg: MorselReg,
    wt_reg: MorselReg,
    count: Int,
    mut sum_wx: Float64,
    mut sum_w: Float64,
    mut valid_cnt: Int,
):
    """Single-pass area-weighted accumulation (`boreal-stat` `globavg` / `areacell_yy`).

    Sums `val * w` and `w` simultaneously ONLY over lanes unmasked in both
    `val_reg` and `wt_reg` (avoiding Xarray's 3-pass `da.weighted(w).mean()`!).
    """
    if count == 0 or val_reg.state == MSTATE_MASKED or wt_reg.state == MSTATE_MASKED:
        return

    var vp = val_reg.data_imm()
    var wp = wt_reg.data_imm()
    var zeros = SIMD[.float64, 8](0.0)

    if val_reg.state == MSTATE_VALID and wt_reg.state == MSTATE_VALID:
        var wx0 = SIMD[.float64, 8](0.0)
        var wx1 = SIMD[.float64, 8](0.0)
        var w0 = SIMD[.float64, 8](0.0)
        var w1 = SIMD[.float64, 8](0.0)
        var pair_end = (count // 16) * 16
        var idx = 0
        while idx < pair_end:
            var v_a = vp.unsafe_load[width=8](idx).cast[.float64]()
            var w_a = wp.unsafe_load[width=8](idx).cast[.float64]()
            var v_b = vp.unsafe_load[width=8](idx + 8).cast[.float64]()
            var w_b = wp.unsafe_load[width=8](idx + 8).cast[.float64]()
            wx0 = fma(v_a, w_a, wx0)
            wx1 = fma(v_b, w_b, wx1)
            w0 += w_a
            w1 += w_b
            idx += 16
        var swx = (wx0 + wx1).reduce_add()
        var sw = (w0 + w1).reduce_add()
        while idx < count:
            var sv = Float64(vp[unsafe_offset=idx])
            var sw_i = Float64(wp[unsafe_offset=idx])
            swx = fma(sv, sw_i, swx)
            sw += sw_i
            idx += 1
        sum_wx += swx
        sum_w += sw
        valid_cnt += count
        return

    var wx0 = SIMD[.float64, 8](0.0)
    var w0 = SIMD[.float64, 8](0.0)
    var n_words = ceildiv(count, 64)
    var vmp = val_reg.mask_imm()
    var wmp = wt_reg.mask_imm()
    var has_vm = val_reg.state == MSTATE_BITS
    var has_wm = wt_reg.state == MSTATE_BITS
    var v_cnt = 0

    for w in range(n_words):
        var word: UInt64 = 0
        if has_vm:
            word |= vmp[unsafe_offset=w]
        if has_wm:
            word |= wmp[unsafe_offset=w]
        if w == n_words - 1 and (count & 63) != 0:
            word |= ~_trailing_mask(count)

        if word == ALL_ONES:
            continue
        elif word == UInt64(0):
            var base = w * 64
            comptime for g in range(8):
                var vv = vp.unsafe_load[width=8](base + g * 8).cast[.float64]()
                var wv = wp.unsafe_load[width=8](base + g * 8).cast[.float64]()
                wx0 = fma(vv, wv, wx0)
                w0 += wv
            v_cnt += 64
        else:
            var base = w * 64
            v_cnt += 64 - Int(pop_count(word))
            comptime for g in range(8):
                var bm = UInt8((word >> UInt64(g * 8)) & UInt64(0xFF))
                if bm != UInt8(0xFF):
                    var vv = vp.unsafe_load[width=8](base + g * 8).cast[.float64]()
                    var wv = wp.unsafe_load[width=8](base + g * 8).cast[.float64]()
                    if bm == UInt8(0):
                        wx0 = fma(vv, wv, wx0)
                        w0 += wv
                    else:
                        var mv = _unpack_bool8(bm)
                        var w_eff = mv.select(zeros, wv)
                        wx0 = fma(mv.select(zeros, vv), w_eff, wx0)
                        w0 += w_eff

    sum_wx += wx0.reduce_add()
    sum_w += w0.reduce_add()
    valid_cnt += v_cnt


@inline(.always)
def morsel_store_filled(
    reg: MorselReg,
    dst_f32: MutPointer[Float32, _],
    dst_off: Int,
    count: Int,
    fill_value: Float32,
):
    """SIMD store of `reg` into output array with `fill_value` (`NaN` or `1e30`)
    substituted on masked lanes in a single pass (zero Python mask post-pass).
    """
    var fv = SIMD[.float32, 8](fill_value)
    var sp = reg.data_imm()

    if reg.state == MSTATE_MASKED:
        var vec_end = (count // 8) * 8
        var k = 0
        while k < vec_end:
            dst_f32.unsafe_store(dst_off + k, fv)
            k += 8
        while k < count:
            dst_f32[unsafe_offset=dst_off + k] = fill_value
            k += 1
        return

    if reg.state == MSTATE_VALID:
        var vec_end = (count // 8) * 8
        var k = 0
        while k < vec_end:
            dst_f32.unsafe_store(dst_off + k, sp.unsafe_load[width=8](k))
            k += 8
        while k < count:
            dst_f32[unsafe_offset=dst_off + k] = sp[unsafe_offset=k]
            k += 1
        return

    var mp = reg.mask_imm()
    var n_words = ceildiv(count, 64)
    for w in range(n_words):
        var word = mp[unsafe_offset=w]
        var base = w * 64
        comptime for g in range(8):
            var off = base + g * 8
            if off + 8 <= count:
                var v = sp.unsafe_load[width=8](off)
                var bm = UInt8((word >> UInt64(g * 8)) & UInt64(0xFF))
                if bm == UInt8(0):
                    dst_f32.unsafe_store(dst_off + off, v)
                elif bm == UInt8(0xFF):
                    dst_f32.unsafe_store(dst_off + off, fv)
                else:
                    dst_f32.unsafe_store(
                        dst_off + off, _unpack_bool8(bm).select(fv, v)
                    )
            elif off < count:
                for lane in range(count - off):
                    var idx = off + lane
                    dst_f32[unsafe_offset=dst_off + idx] = (
                        fill_value if reg.is_masked(idx) else sp[unsafe_offset=idx]
                    )


# ===----------------------------------------------------------------------=== #
# `Float64` Morsel Register Path (`boreal-stat` `values_f64` / `r.diag` Parity)
# ===----------------------------------------------------------------------=== #


struct MorselReg64(Movable):
    """64-byte aligned 2048-element `Float64` L1/L2 register (`16 KB` data + `256 B` mask)
    providing exact IEEE-754 `float64` parity for `boreal-stat` (`r.diag`, `RMSE`, `ke`).
    """

    var _data_alloc: Allocation[Float64, alignment=CACHE_ALIGN]
    var _mask_alloc: Allocation[UInt64, alignment=CACHE_ALIGN]
    var state: Int

    def __init__(out self):
        self._data_alloc = alloc(
            Layout[Float64, alignment=CACHE_ALIGN](count=MORSEL_LEN)
        )
        self._mask_alloc = alloc(
            Layout[UInt64, alignment=CACHE_ALIGN](count=MORSEL_WORDS)
        )
        self.state = MSTATE_VALID
        var dp = self._data_alloc.unsafe_ptr()
        for i in range(0, MORSEL_LEN, W64):
            dp.unsafe_store[alignment=VEC_ALIGN64](i, SIMD[.float64, W64](0.0))
        var mp = self._mask_alloc.unsafe_ptr()
        for w in range(MORSEL_WORDS):
            mp[unsafe_offset=w] = UInt64(0)

    def __init__(out self, *, deinit move: Self):
        self._data_alloc = move._data_alloc^
        self._mask_alloc = move._mask_alloc^
        self.state = move.state

    def __deinit__(deinit self):
        dealloc(self._data_alloc^)
        dealloc(self._mask_alloc^)

    @inline(.always)
    def data_ptr(mut self) -> type_of(self._data_alloc.unsafe_ptr()):
        return self._data_alloc.unsafe_ptr()

    @inline(.always)
    def data_imm(self) -> type_of(self._data_alloc.unsafe_ptr()):
        return self._data_alloc.unsafe_ptr()

    @inline(.always)
    def mask_ptr(mut self) -> type_of(self._mask_alloc.unsafe_ptr()):
        return self._mask_alloc.unsafe_ptr()

    @inline(.always)
    def mask_imm(self) -> type_of(self._mask_alloc.unsafe_ptr()):
        return self._mask_alloc.unsafe_ptr()

    @inline(.always)
    def is_masked(ref self, idx: Int) -> Bool:
        if self.state == MSTATE_VALID:
            return False
        if self.state == MSTATE_MASKED:
            return True
        var w = idx >> 6
        var b = idx & 63
        var word = self._mask_alloc.unsafe_ptr()[unsafe_offset=w]
        return ((word >> UInt64(b)) & UInt64(1)) != UInt64(0)

    def finalize_words(mut self, count: Int, or_acc: UInt64):
        if or_acc == UInt64(0) or count == 0:
            self.state = MSTATE_VALID
            return
        var n_words = ceildiv(count, 64)
        var tail = _trailing_mask(count)
        var mp = self._mask_alloc.unsafe_ptr()
        mp[unsafe_offset=n_words - 1] &= tail
        var any_m = False
        var all_m = True
        for w in range(n_words - 1):
            var word = mp[unsafe_offset=w]
            if word != UInt64(0):
                any_m = True
            if word != ALL_ONES:
                all_m = False
        var lw = mp[unsafe_offset=n_words - 1]
        if lw != UInt64(0):
            any_m = True
        if lw != tail:
            all_m = False
        if not any_m:
            self.state = MSTATE_VALID
        elif all_m:
            self.state = MSTATE_MASKED
        else:
            self.state = MSTATE_BITS


@inline(.always)
def load_span_into_reg64(
    src_bytes: ImmPointer[UInt8, _],
    aux_bytes: ImmPointer[UInt8, _],
    dtype_code: Int,
    dst_off: Int,
    span_len: Int,
    scale: Float64,
    offset: Float64,
    mask_kind: Int,
    mask_param: Int,
    mut reg: MorselReg64,
    mut or_bits: UInt64,
):
    """Promote `Float32`, `Float64`, or `UInt16` spans directly into `MorselReg64`
    while applying `@@` (`uint8` or `float32`) / `NaN` / sentinel masks in one pass.
    """
    var dp = reg.data_ptr()
    var mp = reg.mask_ptr()
    var is_id = scale == 1.0 and offset == 0.0

    for k in range(span_len):
        var raw_v: Float64
        if dtype_code == DTYPE_F64:
            raw_v = src_bytes.unsafe_bitcast[Float64]()[unsafe_offset=k]
        elif dtype_code == DTYPE_U16:
            raw_v = Float64(src_bytes.unsafe_bitcast[UInt16]()[unsafe_offset=k])
        else:
            raw_v = Float64(src_bytes.unsafe_bitcast[Float32]()[unsafe_offset=k])

        var bad = False
        if mask_kind == MSRC_NAN:
            bad = isnan(raw_v)
        elif mask_kind == MSRC_SENTINEL_INT:
            bad = Int(raw_v) == mask_param
        elif mask_kind == MSRC_FST_BITMAP_U8:
            bad = aux_bytes[unsafe_offset=k] == UInt8(0)
        elif mask_kind == MSRC_FST_BITMAP_F32:
            var bf = aux_bytes.unsafe_bitcast[Float32]()[unsafe_offset=k]
            bad = not (bf > Float32(0.0))
        elif mask_kind == MSRC_BYTE_GT:
            bad = Int(aux_bytes[unsafe_offset=k]) > mask_param
        elif mask_kind == MSRC_BYTE_NE:
            bad = Int(aux_bytes[unsafe_offset=k]) != mask_param

        var d_idx = dst_off + k
        if bad:
            dp[unsafe_offset=d_idx] = 0.0
            mp[unsafe_offset=d_idx >> 6] |= UInt64(1) << UInt64(d_idx & 63)
            or_bits |= UInt64(1)
        else:
            dp[unsafe_offset=d_idx] = raw_v if is_id else fma(raw_v, scale, offset)


@inline(.always)
def _prop_unary_mask64(mut dst: MorselReg64, a: MorselReg64, count: Int):
    dst.state = a.state
    if a.state == MSTATE_BITS:
        var n_words = ceildiv(count, 64)
        var dp = dst.mask_ptr()
        var ap = a.mask_imm()
        for w in range(n_words):
            dp[unsafe_offset=w] = ap[unsafe_offset=w]


@inline(.always)
def _prop_binary_mask64(
    mut dst: MorselReg64, a: MorselReg64, b: MorselReg64, count: Int
):
    if a.state == MSTATE_MASKED or b.state == MSTATE_MASKED:
        dst.state = MSTATE_MASKED
        return
    if a.state == MSTATE_VALID and b.state == MSTATE_VALID:
        dst.state = MSTATE_VALID
        return
    var n_words = ceildiv(count, 64)
    var dp = dst.mask_ptr()
    if a.state == MSTATE_BITS and b.state == MSTATE_BITS:
        var ap = a.mask_imm()
        var bp = b.mask_imm()
        var or_w: UInt64 = 0
        for w in range(n_words):
            var u = ap[unsafe_offset=w] | bp[unsafe_offset=w]
            dp[unsafe_offset=w] = u
            or_w |= u
        dst.state = MSTATE_BITS
        dst.finalize_words(count, or_w)
    elif a.state == MSTATE_BITS:
        var ap = a.mask_imm()
        for w in range(n_words):
            dp[unsafe_offset=w] = ap[unsafe_offset=w]
        dst.state = MSTATE_BITS
    else:
        var bp = b.mask_imm()
        for w in range(n_words):
            dp[unsafe_offset=w] = bp[unsafe_offset=w]
        dst.state = MSTATE_BITS


@inline(.always)
def morsel64_binary[
    op_tag: Int
](mut dst: MorselReg64, a: MorselReg64, b: MorselReg64, count: Int):
    """SIMD `Float64` binary ops (`0:add, 1:sub, 2:mul, 3:div, 4:hypot, 5:pow2_sum, 6:min, 7:max`)."""
    _prop_binary_mask64(dst, a, b, count)
    if dst.state == MSTATE_MASKED:
        return
    var n_vec = align_up(count, W64)
    var dp = dst.data_ptr()
    var ap = a.data_imm()
    var bp = b.data_imm()
    var eps = SIMD[.float64, W64](1.0e-15)
    var ones = SIMD[.float64, W64](1.0)
    var zeros = SIMD[.float64, W64](0.0)
    var any_zero = False

    for off in range(0, n_vec, W64):
        var av = ap.unsafe_load[width=W64, alignment=VEC_ALIGN64](off)
        var bv = bp.unsafe_load[width=W64, alignment=VEC_ALIGN64](off)
        comptime match op_tag:
            case 0:
                dp.unsafe_store[alignment=VEC_ALIGN64](off, av + bv)
            case 1:
                dp.unsafe_store[alignment=VEC_ALIGN64](off, av - bv)
            case 2:
                dp.unsafe_store[alignment=VEC_ALIGN64](off, av * bv)
            case 3:
                var bad = abs(bv).lt(eps)
                dp.unsafe_store[alignment=VEC_ALIGN64](
                    off, bad.select(zeros, av / bad.select(ones, bv))
                )
                if bad.reduce_or():
                    any_zero = True
            case 4:
                dp.unsafe_store[alignment=VEC_ALIGN64](
                    off, sqrt(fma(av, av, bv * bv))
                )
            case 5:
                dp.unsafe_store[alignment=VEC_ALIGN64](off, fma(av, av, bv * bv))
            case 6:
                dp.unsafe_store[alignment=VEC_ALIGN64](off, min(av, bv))
            case _:
                dp.unsafe_store[alignment=VEC_ALIGN64](off, max(av, bv))

    comptime if op_tag == 3:
        if any_zero:
            var n_words = ceildiv(count, 64)
            var mp = dst.mask_ptr()
            if dst.state == MSTATE_VALID:
                for w in range(n_words):
                    mp[unsafe_offset=w] = UInt64(0)
            var or_acc: UInt64 = mp[unsafe_offset=0]
            for k in range(count):
                if abs(bp[unsafe_offset=k]) < 1.0e-15:
                    mp[unsafe_offset=k >> 6] |= UInt64(1) << UInt64(k & 63)
                    or_acc |= UInt64(1)
            dst.state = MSTATE_BITS
            dst.finalize_words(count, or_acc)


@inline(.always)
def morsel64_fma(
    mut dst: MorselReg64,
    a: MorselReg64,
    b: MorselReg64,
    c: MorselReg64,
    count: Int,
):
    _prop_binary_mask64(dst, a, b, count)
    if dst.state == MSTATE_MASKED or c.state == MSTATE_MASKED:
        dst.state = MSTATE_MASKED
        return
    if c.state == MSTATE_BITS:
        var n_words = ceildiv(count, 64)
        var dp_m = dst.mask_ptr()
        var cp_m = c.mask_imm()
        if dst.state == MSTATE_VALID:
            for w in range(n_words):
                dp_m[unsafe_offset=w] = cp_m[unsafe_offset=w]
            dst.state = MSTATE_BITS
        else:
            for w in range(n_words):
                dp_m[unsafe_offset=w] |= cp_m[unsafe_offset=w]
    var n_vec = align_up(count, W64)
    var dp = dst.data_ptr()
    var ap = a.data_imm()
    var bp = b.data_imm()
    var cp = c.data_imm()
    for off in range(0, n_vec, W64):
        dp.unsafe_store[alignment=VEC_ALIGN64](
            off,
            fma(
                ap.unsafe_load[width=W64, alignment=VEC_ALIGN64](off),
                bp.unsafe_load[width=W64, alignment=VEC_ALIGN64](off),
                cp.unsafe_load[width=W64, alignment=VEC_ALIGN64](off),
            ),
        )


@inline(.always)
def morsel64_affine(
    mut dst: MorselReg64,
    a: MorselReg64,
    scale: Float64,
    offset: Float64,
    count: Int,
):
    _prop_unary_mask64(dst, a, count)
    if dst.state == MSTATE_MASKED:
        return
    var n_vec = align_up(count, W64)
    var dp = dst.data_ptr()
    var ap = a.data_imm()
    var sv = SIMD[.float64, W64](scale)
    var ov = SIMD[.float64, W64](offset)
    for off in range(0, n_vec, W64):
        dp.unsafe_store[alignment=VEC_ALIGN64](
            off,
            fma(ap.unsafe_load[width=W64, alignment=VEC_ALIGN64](off), sv, ov),
        )


@inline(.always)
def morsel64_unary[
    op_tag: Int
](mut dst: MorselReg64, a: MorselReg64, count: Int):
    _prop_unary_mask64(dst, a, count)
    if dst.state == MSTATE_MASKED:
        return
    var n_vec = align_up(count, W64)
    var dp = dst.data_ptr()
    var ap = a.data_imm()
    var zeros = SIMD[.float64, W64](0.0)
    var ones = SIMD[.float64, W64](1.0)
    var bad_dom = False
    for off in range(0, n_vec, W64):
        var av = ap.unsafe_load[width=W64, alignment=VEC_ALIGN64](off)
        comptime match op_tag:
            case 0:
                dp.unsafe_store[alignment=VEC_ALIGN64](off, -av)
            case 1:
                dp.unsafe_store[alignment=VEC_ALIGN64](off, abs(av))
            case 2:
                var neg = av.lt(zeros)
                dp.unsafe_store[alignment=VEC_ALIGN64](
                    off, sqrt(neg.select(zeros, av))
                )
                if neg.reduce_or():
                    bad_dom = True
            case 3:
                dp.unsafe_store[alignment=VEC_ALIGN64](off, exp(av))
            case 4:
                var np_m = av.le(zeros)
                dp.unsafe_store[alignment=VEC_ALIGN64](
                    off, np_m.select(zeros, log(np_m.select(ones, av)))
                )
                if np_m.reduce_or():
                    bad_dom = True
            case 5:
                dp.unsafe_store[alignment=VEC_ALIGN64](off, sin(av))
            case _:
                dp.unsafe_store[alignment=VEC_ALIGN64](off, cos(av))

    comptime if op_tag == 2 or op_tag == 4:
        if bad_dom:
            var n_words = ceildiv(count, 64)
            var mp = dst.mask_ptr()
            if dst.state == MSTATE_VALID:
                for w in range(n_words):
                    mp[unsafe_offset=w] = UInt64(0)
            var or_acc: UInt64 = 0
            for k in range(count):
                var v = ap[unsafe_offset=k]
                var bad = (v < 0.0) if op_tag == 2 else (v <= 0.0)
                if bad:
                    mp[unsafe_offset=k >> 6] |= UInt64(1) << UInt64(k & 63)
                    or_acc |= UInt64(1)
            dst.state = MSTATE_BITS
            dst.finalize_words(count, or_acc)


@inline(.always)
def morsel64_accumulate_stats(
    reg: MorselReg64,
    count: Int,
    need_sum_sq: Bool,
    mut acc: ReduceAcc,
):
    acc.total_count += count
    if count == 0 or reg.state == MSTATE_MASKED:
        return
    var dp = reg.data_imm()
    if reg.state == MSTATE_VALID:
        var s_tot = 0.0
        var sq_tot = 0.0
        var mn_tot = Float32(1.0e30)
        var mx_tot = Float32(-1.0e30)
        for k in range(count):
            var v = dp[unsafe_offset=k]
            s_tot += v
            if need_sum_sq:
                sq_tot = fma(v, v, sq_tot)
            var vf = Float32(v)
            mn_tot = min(mn_tot, vf)
            mx_tot = max(mx_tot, vf)
        acc.sum += s_tot
        acc.sum_sq += sq_tot
        acc.min_val = min(acc.min_val, mn_tot)
        acc.max_val = max(acc.max_val, mx_tot)
        acc.valid_count += count
        return

    var mp = reg.mask_imm()
    var n_words = ceildiv(count, 64)
    var s_tot = 0.0
    var sq_tot = 0.0
    var mn_tot = Float32(1.0e30)
    var mx_tot = Float32(-1.0e30)
    var v_cnt = 0
    for w in range(n_words):
        var word = mp[unsafe_offset=w]
        if w == n_words - 1 and (count & 63) != 0:
            word |= ~_trailing_mask(count)
        if word == ALL_ONES:
            continue
        var base = w * 64
        var lim = min(64, count - base)
        if word == UInt64(0):
            for l in range(lim):
                var v = dp[unsafe_offset=base + l]
                s_tot += v
                if need_sum_sq:
                    sq_tot = fma(v, v, sq_tot)
                var vf = Float32(v)
                mn_tot = min(mn_tot, vf)
                mx_tot = max(mx_tot, vf)
            v_cnt += lim
        else:
            for l in range(lim):
                if ((word >> UInt64(l)) & UInt64(1)) == UInt64(0):
                    var v = dp[unsafe_offset=base + l]
                    s_tot += v
                    if need_sum_sq:
                        sq_tot = fma(v, v, sq_tot)
                    var vf = Float32(v)
                    mn_tot = min(mn_tot, vf)
                    mx_tot = max(mx_tot, vf)
                    v_cnt += 1
    if v_cnt > 0:
        acc.sum += s_tot
        acc.sum_sq += sq_tot
        acc.min_val = min(acc.min_val, mn_tot)
        acc.max_val = max(acc.max_val, mx_tot)
        acc.valid_count += v_cnt


@inline(.always)
def morsel64_accumulate_weighted(
    val_reg: MorselReg64,
    wt_reg: MorselReg64,
    count: Int,
    mut sum_wx: Float64,
    mut sum_w: Float64,
    mut valid_cnt: Int,
):
    if count == 0 or val_reg.state == MSTATE_MASKED or wt_reg.state == MSTATE_MASKED:
        return
    var vp = val_reg.data_imm()
    var wp = wt_reg.data_imm()
    for k in range(count):
        if not val_reg.is_masked(k) and not wt_reg.is_masked(k):
            var vv = vp[unsafe_offset=k]
            var wv = wp[unsafe_offset=k]
            sum_wx = fma(vv, wv, sum_wx)
            sum_w += wv
            valid_cnt += 1


@inline(.always)
def morsel64_store_filled(
    reg: MorselReg64,
    dst_f64: MutPointer[Float64, _],
    dst_off: Int,
    count: Int,
    fill_value: Float64,
):
    var sp = reg.data_imm()
    if reg.state == MSTATE_MASKED:
        for k in range(count):
            dst_f64[unsafe_offset=dst_off + k] = fill_value
        return
    if reg.state == MSTATE_VALID:
        for k in range(count):
            dst_f64[unsafe_offset=dst_off + k] = sp[unsafe_offset=k]
        return
    for k in range(count):
        dst_f64[unsafe_offset=dst_off + k] = (
            fill_value if reg.is_masked(k) else sp[unsafe_offset=k]
        )
