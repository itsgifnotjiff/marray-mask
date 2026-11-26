"""Layer 1: Fortran-contiguous MaskedSlab, hardware UInt64 bitmasks, SIMD kernels,
and encoding-adaptive in-memory ECCC GDPS binary decoders.

Physical layout:
  - 2D horizontal NWP slabs `(ni, nj)` are stored column-major (Fortran order):
    `flat_idx = i + j * ni`, so axis 0 (`ni`, longitude) is stride-1 contiguous.
  - Missing/invalid masks from heterogeneous binary encodings (16-bit sentinel
    values, 1-bit GRIB2/RPN `@@` bitmaps, 8-bit quality flags) normalize at
    decode time into a 3-state mask representation:
      * MASK_ALL_VALID  (0): zero mask buffer overhead, unmasked SIMD fast path
      * MASK_ALL_MASKED (1): O(1) short-circuit skip across downstream ops
      * MASK_BITMASK    (2): packed UInt64 words (1 = masked, 0 = valid), 8x
                             smaller than NumPy `ma` byte masks, with 64-element
                             word-level skip for all-valid (0) and all-masked (~0)
                             blocks.
"""

from std.collections import List
from std.bit import pop_count
from std.math import sqrt, sin, cos


comptime MASK_ALL_VALID: Int = 0
comptime MASK_ALL_MASKED: Int = 1
comptime MASK_BITMASK: Int = 2

# Binary layer encodings mimicking ECCC GDPS (RPN fstd / GRIB2):
comptime ENC_QUANT16_SENTINEL: Int = 1  # UInt16 + scale/offset + 0xFFFF terrain sentinel
comptime ENC_FLOAT32_BITMAP: Int = 2    # Float32 + 1-bit land/sea bitmap (`@@`)
comptime ENC_UINT16_QFLAG: Int = 3      # UInt16 + scale/offset + UInt8 QC flag (>=2 masked)

# GDPS variable IDs:
comptime VAR_TT: Int = 0  # Air temperature (degC)          -> ENC_QUANT16_SENTINEL
comptime VAR_UU: Int = 1  # Zonal wind component (knots)    -> ENC_QUANT16_SENTINEL
comptime VAR_VV: Int = 2  # Meridional wind component (kt)  -> ENC_QUANT16_SENTINEL
comptime VAR_HU: Int = 3  # Specific humidity (g/kg)        -> ENC_UINT16_QFLAG
comptime VAR_TM: Int = 4  # Sea surface temperature (degC)  -> ENC_FLOAT32_BITMAP

comptime SIMD_W: Int = 8
comptime ALL_ONES_U64: UInt64 = ~UInt64(0)


@always_inline
def _padded_f32_len(n: Int) -> Int:
    var rem = n % SIMD_W
    if rem == 0:
        return n if n > 0 else SIMD_W
    return n + (SIMD_W - rem)


@always_inline
def _num_mask_words(n: Int) -> Int:
    return (n + 63) // 64


@always_inline
def _trailing_valid_bitmask(n: Int) -> UInt64:
    var rem = n & 63
    if rem == 0:
        return ALL_ONES_U64
    return (UInt64(1) << UInt64(rem)) - UInt64(1)


@always_inline
def _byte_to_bool8(byte_val: UInt8) -> SIMD[DType.bool, 8]:
    comptime bit_lanes = SIMD[DType.uint8, 8](1, 2, 4, 8, 16, 32, 64, 128)
    var splat = SIMD[DType.uint8, 8](byte_val)
    return (splat & bit_lanes).ne(SIMD[DType.uint8, 8](0))


@always_inline
def _bool8_to_byte(mask_vec: SIMD[DType.bool, 8]) -> UInt64:
    var weights = mask_vec.select(
        SIMD[DType.uint8, 8](1, 2, 4, 8, 16, 32, 64, 128),
        SIMD[DType.uint8, 8](0),
    )
    return UInt64(weights.reduce_add())


struct SlabStats(Copyable, Movable):
    var sum: Float64
    var min_val: Float32
    var max_val: Float32
    var valid_count: Int
    var total_count: Int

    def __init__(out self):
        self.sum = Float64(0.0)
        self.min_val = Float32(1.0e30)
        self.max_val = Float32(-1.0e30)
        self.valid_count = 0
        self.total_count = 0

    def mean(self) -> Float64:
        if self.valid_count == 0:
            return Float64(0.0) / Float64(0.0)
        return self.sum / Float64(self.valid_count)

    def masked_fraction(self) -> Float64:
        if self.total_count == 0:
            return Float64(0.0)
        return Float64(self.total_count - self.valid_count) / Float64(
            self.total_count
        )

    def merge(mut self, other: SlabStats):
        self.sum += other.sum
        if other.valid_count > 0:
            if other.min_val < self.min_val:
                self.min_val = other.min_val
            if other.max_val > self.max_val:
                self.max_val = other.max_val
        self.valid_count += other.valid_count
        self.total_count += other.total_count


struct MaskedSlab(Copyable, Movable):
    """2D Fortran-contiguous `(ni, nj)` data buffer + 3-state packed bitmask."""

    var ni: Int
    var nj: Int
    var data: List[Float32]
    var mask_words: List[UInt64]
    var mask_state: Int

    def __init__(out self, ni: Int = 0, nj: Int = 0, fill: Float32 = 0.0):
        self.ni = ni
        self.nj = nj
        var cap = _padded_f32_len(ni * nj)
        self.data = List[Float32](capacity=cap)
        self.data.resize(cap, fill)
        self.mask_words = List[UInt64]()
        self.mask_state = MASK_ALL_VALID

    @always_inline
    def size(self) -> Int:
        return self.ni * self.nj

    @always_inline
    def f_index(self, i: Int, j: Int) -> Int:
        """Column-major (Fortran) linear index: `i` (axis 0) has stride 1."""
        return i + j * self.ni

    def ensure_shape(mut self, ni: Int, nj: Int):
        """Reuse existing heap allocation across chunks when capacity suffices."""
        self.ni = ni
        self.nj = nj
        var needed = _padded_f32_len(ni * nj)
        if len(self.data) < needed:
            self.data.resize(needed, Float32(0.0))

    def ensure_bitmask_zeroed(mut self):
        var n_words = _num_mask_words(self.ni * self.nj)
        if len(self.mask_words) < n_words:
            self.mask_words.resize(n_words, UInt64(0))
        var m_ptr = self.mask_words.unsafe_ptr()
        for w in range(n_words):
            m_ptr.unsafe_store(w, UInt64(0))
        self.mask_state = MASK_BITMASK

    def ensure_bitmask_uninit(mut self):
        var n_words = _num_mask_words(self.ni * self.nj)
        if len(self.mask_words) < n_words:
            self.mask_words.resize(n_words, UInt64(0))
        self.mask_state = MASK_BITMASK

    def finalize_mask_state(mut self):
        """Demote BITMASK -> ALL_VALID or ALL_MASKED if uniform."""
        if self.mask_state != MASK_BITMASK:
            return
        var n = self.ni * self.nj
        if n == 0:
            self.mask_state = MASK_ALL_VALID
            return
        var n_words = _num_mask_words(n)
        var tail_mask = _trailing_valid_bitmask(n)
        self.mask_words[n_words - 1] = self.mask_words[n_words - 1] & tail_mask

        var any_masked = False
        var all_masked = True
        for w in range(n_words - 1):
            var word = self.mask_words[w]
            if word != UInt64(0):
                any_masked = True
            if word != ALL_ONES_U64:
                all_masked = False
        var last_w = self.mask_words[n_words - 1]
        if last_w != UInt64(0):
            any_masked = True
        if last_w != tail_mask:
            all_masked = False

        if not any_masked:
            self.mask_state = MASK_ALL_VALID
        elif all_masked:
            self.mask_state = MASK_ALL_MASKED

    @always_inline
    def get(self, i: Int, j: Int) -> Float32:
        return self.data[i + j * self.ni]

    @always_inline
    def set(mut self, i: Int, j: Int, val: Float32):
        self.data[i + j * self.ni] = val

    @always_inline
    def is_masked_flat(self, idx: Int) -> Bool:
        if self.mask_state == MASK_ALL_VALID:
            return False
        if self.mask_state == MASK_ALL_MASKED:
            return True
        var w = idx >> 6
        var b = idx & 63
        return ((self.mask_words[w] >> UInt64(b)) & UInt64(1)) != UInt64(0)

    @always_inline
    def is_masked(self, i: Int, j: Int) -> Bool:
        return self.is_masked_flat(i + j * self.ni)

    def mask_cell(mut self, i: Int, j: Int):
        var idx = i + j * self.ni
        if self.mask_state == MASK_ALL_MASKED:
            return
        if self.mask_state == MASK_ALL_VALID:
            self.ensure_bitmask_zeroed()
        var w = idx >> 6
        var b = idx & 63
        self.mask_words[w] = self.mask_words[w] | (UInt64(1) << UInt64(b))

    def valid_count(self) -> Int:
        var n = self.ni * self.nj
        if self.mask_state == MASK_ALL_VALID:
            return n
        if self.mask_state == MASK_ALL_MASKED or n == 0:
            return 0
        var n_words = _num_mask_words(n)
        var masked_total = 0
        for w in range(n_words - 1):
            masked_total += Int(pop_count(self.mask_words[w]))
        var last_word = self.mask_words[n_words - 1] & _trailing_valid_bitmask(n)
        masked_total += Int(pop_count(last_word))
        return n - masked_total

    def slice_fortran(self, i0: Int, i1: Int, j0: Int, j1: Int) -> MaskedSlab:
        """Extract sub-slab `[i0:i1, j0:j1]` keeping contiguous Fortran columns."""
        var sub_ni = i1 - i0
        var sub_nj = j1 - j0
        var out = MaskedSlab(sub_ni, sub_nj)
        if sub_ni <= 0 or sub_nj <= 0:
            return out^
        if self.mask_state == MASK_ALL_MASKED:
            out.mask_state = MASK_ALL_MASKED
            return out^

        var src_ptr = self.data.unsafe_ptr()
        var dst_ptr = out.data.unsafe_ptr()
        for jj in range(sub_nj):
            var src_col = (j0 + jj) * self.ni + i0
            var dst_col = jj * sub_ni
            for ii in range(sub_ni):
                dst_ptr.unsafe_store(dst_col + ii, src_ptr.unsafe_load(src_col + ii))

        if self.mask_state == MASK_BITMASK:
            out.ensure_bitmask_zeroed()
            for jj in range(sub_nj):
                var src_col = (j0 + jj) * self.ni + i0
                var dst_col = jj * sub_ni
                for ii in range(sub_ni):
                    if self.is_masked_flat(src_col + ii):
                        var d_idx = dst_col + ii
                        out.mask_words[d_idx >> 6] |= UInt64(1) << UInt64(
                            d_idx & 63
                        )
            out.finalize_mask_state()
        return out^

    def __add__(self, other: Self) -> Self:
        var out = MaskedSlab(self.ni, self.nj)
        slab_add_into(out, self, other)
        return out^

    def __sub__(self, other: Self) -> Self:
        var out = MaskedSlab(self.ni, self.nj)
        slab_sub_into(out, self, other)
        return out^

    def __mul__(self, other: Self) -> Self:
        var out = MaskedSlab(self.ni, self.nj)
        slab_mul_into(out, self, other)
        return out^

    def __truediv__(self, other: Self) -> Self:
        var out = MaskedSlab(self.ni, self.nj)
        slab_div_into(out, self, other)
        return out^


# ===----------------------------------------------------------------------=== #
# Seamless Mask Propagation Helpers & Vectorized Kernels
# ===----------------------------------------------------------------------=== #


def _propagate_binary_mask(mut dst: MaskedSlab, a: MaskedSlab, b: MaskedSlab):
    var n = dst.size()
    if a.mask_state == MASK_ALL_MASKED or b.mask_state == MASK_ALL_MASKED:
        dst.mask_state = MASK_ALL_MASKED
        return
    if a.mask_state == MASK_ALL_VALID and b.mask_state == MASK_ALL_VALID:
        dst.mask_state = MASK_ALL_VALID
        return

    var n_words = _num_mask_words(n)
    dst.ensure_bitmask_uninit()
    var d_ptr = dst.mask_words.unsafe_ptr()
    if a.mask_state == MASK_BITMASK and b.mask_state == MASK_BITMASK:
        var a_ptr = a.mask_words.unsafe_ptr()
        var b_ptr = b.mask_words.unsafe_ptr()
        for w in range(n_words):
            d_ptr.unsafe_store(w, a_ptr.unsafe_load(w) | b_ptr.unsafe_load(w))
    elif a.mask_state == MASK_BITMASK:
        var a_ptr = a.mask_words.unsafe_ptr()
        for w in range(n_words):
            d_ptr.unsafe_store(w, a_ptr.unsafe_load(w))
    else:
        var b_ptr = b.mask_words.unsafe_ptr()
        for w in range(n_words):
            d_ptr.unsafe_store(w, b_ptr.unsafe_load(w))


def _copy_unary_mask(mut dst: MaskedSlab, a: MaskedSlab):
    if a.mask_state == MASK_ALL_MASKED:
        dst.mask_state = MASK_ALL_MASKED
        return
    if a.mask_state == MASK_ALL_VALID:
        dst.mask_state = MASK_ALL_VALID
        return
    var n_words = _num_mask_words(dst.size())
    dst.ensure_bitmask_uninit()
    var d_ptr = dst.mask_words.unsafe_ptr()
    var a_ptr = a.mask_words.unsafe_ptr()
    for w in range(n_words):
        d_ptr.unsafe_store(w, a_ptr.unsafe_load(w))


def slab_add_into(mut dst: MaskedSlab, a: MaskedSlab, b: MaskedSlab):
    dst.ensure_shape(a.ni, a.nj)
    _propagate_binary_mask(dst, a, b)
    if dst.mask_state == MASK_ALL_MASKED:
        return
    var n = _padded_f32_len(dst.size())
    var dp = dst.data.unsafe_ptr()
    var ap = a.data.unsafe_ptr()
    var bp = b.data.unsafe_ptr()
    var idx = 0
    while idx < n:
        dp.unsafe_store(
            idx, ap.unsafe_load[width=SIMD_W](idx) + bp.unsafe_load[width=SIMD_W](idx)
        )
        idx += SIMD_W


def slab_sub_into(mut dst: MaskedSlab, a: MaskedSlab, b: MaskedSlab):
    dst.ensure_shape(a.ni, a.nj)
    _propagate_binary_mask(dst, a, b)
    if dst.mask_state == MASK_ALL_MASKED:
        return
    var n = _padded_f32_len(dst.size())
    var dp = dst.data.unsafe_ptr()
    var ap = a.data.unsafe_ptr()
    var bp = b.data.unsafe_ptr()
    var idx = 0
    while idx < n:
        dp.unsafe_store(
            idx, ap.unsafe_load[width=SIMD_W](idx) - bp.unsafe_load[width=SIMD_W](idx)
        )
        idx += SIMD_W


def slab_mul_into(mut dst: MaskedSlab, a: MaskedSlab, b: MaskedSlab):
    dst.ensure_shape(a.ni, a.nj)
    _propagate_binary_mask(dst, a, b)
    if dst.mask_state == MASK_ALL_MASKED:
        return
    var n = _padded_f32_len(dst.size())
    var dp = dst.data.unsafe_ptr()
    var ap = a.data.unsafe_ptr()
    var bp = b.data.unsafe_ptr()
    var idx = 0
    while idx < n:
        dp.unsafe_store(
            idx, ap.unsafe_load[width=SIMD_W](idx) * bp.unsafe_load[width=SIMD_W](idx)
        )
        idx += SIMD_W


def slab_div_into(mut dst: MaskedSlab, a: MaskedSlab, b: MaskedSlab):
    """Vectorized division that also auto-masks lanes where `|b| < 1e-12`."""
    dst.ensure_shape(a.ni, a.nj)
    _propagate_binary_mask(dst, a, b)
    if dst.mask_state == MASK_ALL_MASKED:
        return

    var total = dst.size()
    var n_pad = _padded_f32_len(total)
    var dp = dst.data.unsafe_ptr()
    var ap = a.data.unsafe_ptr()
    var bp = b.data.unsafe_ptr()
    var eps = SIMD[DType.float32, SIMD_W](1.0e-12)
    var ones = SIMD[DType.float32, SIMD_W](1.0)
    var zeros = SIMD[DType.float32, SIMD_W](0.0)
    var any_zero_div = False

    var idx = 0
    while idx < n_pad:
        var av = ap.unsafe_load[width=SIMD_W](idx)
        var bv = bp.unsafe_load[width=SIMD_W](idx)
        var abs_b = bv.lt(zeros).select(-bv, bv)
        var bad = abs_b.lt(eps)
        var safe_b = bad.select(ones, bv)
        dp.unsafe_store(idx, bad.select(zeros, av / safe_b))
        if _bool8_to_byte(bad) != UInt64(0):
            any_zero_div = True
        idx += SIMD_W

    if any_zero_div:
        if dst.mask_state == MASK_ALL_VALID:
            dst.ensure_bitmask_zeroed()
        for k in range(total):
            var bv_s = bp.unsafe_load(k)
            if bv_s > Float32(-1.0e-12) and bv_s < Float32(1.0e-12):
                dst.mask_words[k >> 6] |= UInt64(1) << UInt64(k & 63)
        dst.finalize_mask_state()


def slab_hypot_into(mut dst: MaskedSlab, u: MaskedSlab, v: MaskedSlab):
    """Fused horizontal wind speed `sqrt(u^2 + v^2)` in one memory pass."""
    dst.ensure_shape(u.ni, u.nj)
    _propagate_binary_mask(dst, u, v)
    if dst.mask_state == MASK_ALL_MASKED:
        return
    var n = _padded_f32_len(dst.size())
    var dp = dst.data.unsafe_ptr()
    var up = u.data.unsafe_ptr()
    var vp = v.data.unsafe_ptr()
    var idx = 0
    while idx < n:
        var uv = up.unsafe_load[width=SIMD_W](idx)
        var vv = vp.unsafe_load[width=SIMD_W](idx)
        dp.unsafe_store(idx, sqrt(uv * uv + vv * vv))
        idx += SIMD_W


def slab_sqrt_into(mut dst: MaskedSlab, a: MaskedSlab):
    """Vectorized `sqrt(a)` that auto-masks negative domain violations (`a < 0`)."""
    dst.ensure_shape(a.ni, a.nj)
    _copy_unary_mask(dst, a)
    if dst.mask_state == MASK_ALL_MASKED:
        return
    var total = dst.size()
    var n_pad = _padded_f32_len(total)
    var dp = dst.data.unsafe_ptr()
    var ap = a.data.unsafe_ptr()
    var zeros = SIMD[DType.float32, SIMD_W](0.0)
    var any_neg = False

    var idx = 0
    while idx < n_pad:
        var av = ap.unsafe_load[width=SIMD_W](idx)
        var neg = av.lt(zeros)
        dp.unsafe_store(idx, sqrt(neg.select(zeros, av)))
        if _bool8_to_byte(neg) != UInt64(0):
            any_neg = True
        idx += SIMD_W

    if any_neg:
        if dst.mask_state == MASK_ALL_VALID:
            dst.ensure_bitmask_zeroed()
        for k in range(total):
            if ap.unsafe_load(k) < Float32(0.0):
                dst.mask_words[k >> 6] |= UInt64(1) << UInt64(k & 63)
        dst.finalize_mask_state()


def slab_affine_into(
    mut dst: MaskedSlab, a: MaskedSlab, scale: Float32, offset: Float32
):
    dst.ensure_shape(a.ni, a.nj)
    _copy_unary_mask(dst, a)
    if dst.mask_state == MASK_ALL_MASKED:
        return
    var n = _padded_f32_len(dst.size())
    var dp = dst.data.unsafe_ptr()
    var ap = a.data.unsafe_ptr()
    var sv = SIMD[DType.float32, SIMD_W](scale)
    var ov = SIMD[DType.float32, SIMD_W](offset)
    var idx = 0
    while idx < n:
        dp.unsafe_store(idx, ap.unsafe_load[width=SIMD_W](idx) * sv + ov)
        idx += SIMD_W


def slab_mask_lt_into(mut dst: MaskedSlab, a: MaskedSlab, threshold: Float32):
    """Mask elements where `a < threshold`, preserving existing mask bits."""
    dst.ensure_shape(a.ni, a.nj)
    if a.mask_state == MASK_ALL_MASKED:
        dst.mask_state = MASK_ALL_MASKED
        return
    var total = dst.size()
    var n_pad = _padded_f32_len(total)
    var dp = dst.data.unsafe_ptr()
    var ap = a.data.unsafe_ptr()
    for idx in range(0, n_pad, SIMD_W):
        dp.unsafe_store(idx, ap.unsafe_load[width=SIMD_W](idx))

    dst.ensure_bitmask_zeroed()
    var n_words = _num_mask_words(total)
    if a.mask_state == MASK_BITMASK:
        for w in range(n_words):
            dst.mask_words[w] = a.mask_words[w]

    var tv = SIMD[DType.float32, SIMD_W](threshold)
    for w in range(n_words):
        var base = w * 64
        var new_bits: UInt64 = 0
        for g in range(8):
            var off = base + g * 8
            if off < total:
                var cmp = ap.unsafe_load[width=SIMD_W](off).lt(tv)
                new_bits |= _bool8_to_byte(cmp) << UInt64(g * 8)
        dst.mask_words[w] |= new_bits
    dst.finalize_mask_state()


def slab_reduce_stats(s: MaskedSlab) -> SlabStats:
    """SIMD reduction over a Fortran slab with 64-element bitmask word skipping."""
    var stats = SlabStats()
    var n = s.size()
    stats.total_count = n
    if n == 0 or s.mask_state == MASK_ALL_MASKED:
        return stats^

    var ptr = s.data.unsafe_ptr()
    var pos_inf = Float32(1.0e30)
    var neg_inf = Float32(-1.0e30)
    var zeros = SIMD[DType.float32, SIMD_W](0.0)
    var pinf_v = SIMD[DType.float32, SIMD_W](pos_inf)
    var ninf_v = SIMD[DType.float32, SIMD_W](neg_inf)

    if s.mask_state == MASK_ALL_VALID:
        var sum_v = SIMD[DType.float64, SIMD_W](0.0)
        var min_v = pinf_v
        var max_v = ninf_v
        var vec_end = (n // SIMD_W) * SIMD_W
        var idx = 0
        while idx < vec_end:
            var v = ptr.unsafe_load[width=SIMD_W](idx)
            sum_v += v.cast[DType.float64]()
            min_v = v.lt(min_v).select(v, min_v)
            max_v = v.gt(max_v).select(v, max_v)
            idx += SIMD_W

        stats.sum = sum_v.reduce_add()
        stats.min_val = min_v.reduce_min()
        stats.max_val = max_v.reduce_max()
        while idx < n:
            var sv = ptr.unsafe_load(idx)
            stats.sum += Float64(sv)
            if sv < stats.min_val:
                stats.min_val = sv
            if sv > stats.max_val:
                stats.max_val = sv
            idx += 1
        stats.valid_count = n
        return stats^

    # MASK_BITMASK path with 64-point block skipping:
    var sum_v = SIMD[DType.float64, SIMD_W](0.0)
    var min_v = pinf_v
    var max_v = ninf_v
    var valid_cnt = 0
    var n_words = _num_mask_words(n)
    var m_ptr = s.mask_words.unsafe_ptr()

    for w in range(n_words):
        var word = m_ptr.unsafe_load(w)
        if w == n_words - 1 and (n & 63) != 0:
            # Mark out-of-bounds padding lanes in final word as masked (1)
            word |= ~_trailing_valid_bitmask(n)

        if word == ALL_ONES_U64:
            # All 64 grid cells masked (e.g. continent interior for SST) -> skip!
            continue
        elif word == UInt64(0):
            # All 64 grid cells valid -> full-speed unmasked SIMD!
            var base = w * 64
            for g in range(8):
                var v = ptr.unsafe_load[width=SIMD_W](base + g * 8)
                sum_v += v.cast[DType.float64]()
                min_v = v.lt(min_v).select(v, min_v)
                max_v = v.gt(max_v).select(v, max_v)
            valid_cnt += 64
        else:
            # Mixed coastal/orographic boundary word -> lane-predicated SIMD select
            var base = w * 64
            valid_cnt += 64 - Int(pop_count(word))
            for g in range(8):
                var byte_m = UInt8((word >> UInt64(g * 8)) & UInt64(0xFF))
                if byte_m == UInt8(0xFF):
                    continue
                var v = ptr.unsafe_load[width=SIMD_W](base + g * 8)
                if byte_m == UInt8(0):
                    sum_v += v.cast[DType.float64]()
                    min_v = v.lt(min_v).select(v, min_v)
                    max_v = v.gt(max_v).select(v, max_v)
                else:
                    var m_vec = _byte_to_bool8(byte_m)
                    var v_sum = m_vec.select(zeros, v)
                    var v_min = m_vec.select(pinf_v, v)
                    var v_max = m_vec.select(ninf_v, v)
                    sum_v += v_sum.cast[DType.float64]()
                    min_v = v_min.lt(min_v).select(v_min, min_v)
                    max_v = v_max.gt(max_v).select(v_max, max_v)

    stats.valid_count = valid_cnt
    if valid_cnt > 0:
        stats.sum = sum_v.reduce_add()
        stats.min_val = min_v.reduce_min()
        stats.max_val = max_v.reduce_max()
    return stats^


# ===----------------------------------------------------------------------=== #
# Thread-Local Raw Binary Layer Scratch + Encoding-Adaptive GDPS Decoders
# ===----------------------------------------------------------------------=== #


struct RawLayerScratch(Copyable, Movable):
    """Reusable thread-local buffer holding raw encoded bytes before decoding."""

    var u16_buf: List[UInt16]
    var f32_buf: List[Float32]
    var u8_buf: List[UInt8]
    var bitmap_buf: List[UInt64]

    def __init__(out self):
        self.u16_buf = List[UInt16]()
        self.f32_buf = List[Float32]()
        self.u8_buf = List[UInt8]()
        self.bitmap_buf = List[UInt64]()

    def ensure_capacity(mut self, n: Int):
        var n_pad = _padded_f32_len(n)
        var n_words = _num_mask_words(n)
        if len(self.u16_buf) < n_pad:
            self.u16_buf.resize(n_pad, UInt16(0))
        if len(self.f32_buf) < n_pad:
            self.f32_buf.resize(n_pad, Float32(0.0))
        if len(self.u8_buf) < n_pad:
            self.u8_buf.resize(n_pad, UInt8(0))
        if len(self.bitmap_buf) < n_words:
            self.bitmap_buf.resize(n_words, UInt64(0))


def decode_quant16_sentinel_into(
    mut dst: MaskedSlab,
    raw_u16: List[UInt16],
    ni: Int,
    nj: Int,
    scale: Float32,
    offset: Float32,
    sentinel: UInt16 = UInt16(0xFFFF),
):
    """Decode RPN/GRIB2 16-bit quantized layer (`0xFFFF` -> bitmask, else `raw*scale+offset`)."""
    dst.ensure_shape(ni, nj)
    var total = ni * nj
    var n_pad = _padded_f32_len(total)
    var rp = raw_u16.unsafe_ptr()
    var dp = dst.data.unsafe_ptr()
    var sv = SIMD[DType.float32, SIMD_W](scale)
    var ov = SIMD[DType.float32, SIMD_W](offset)
    var sent_v = SIMD[DType.uint16, SIMD_W](sentinel)
    var zero_f = SIMD[DType.float32, SIMD_W](0.0)

    var any_sentinel = False
    for idx in range(0, n_pad, SIMD_W):
        var q = rp.unsafe_load[width=SIMD_W](idx)
        var is_sent = q.eq(sent_v)
        var val = q.cast[DType.float32]() * sv + ov
        dp.unsafe_store(idx, is_sent.select(zero_f, val))
        if _bool8_to_byte(is_sent) != UInt64(0):
            any_sentinel = True

    if not any_sentinel:
        dst.mask_state = MASK_ALL_VALID
        return

    dst.ensure_bitmask_zeroed()
    var n_words = _num_mask_words(total)
    for w in range(n_words):
        var base = w * 64
        var word_bits: UInt64 = 0
        for g in range(8):
            var off = base + g * 8
            if off < total:
                var q = rp.unsafe_load[width=SIMD_W](off)
                word_bits |= _bool8_to_byte(q.eq(sent_v)) << UInt64(g * 8)
        dst.mask_words[w] = word_bits
    dst.finalize_mask_state()


def decode_float32_bitmap_into(
    mut dst: MaskedSlab,
    raw_f32: List[Float32],
    raw_bitmap: List[UInt64],
    ni: Int,
    nj: Int,
):
    """Decode Fortran Float32 layer paired with GRIB2/RPN `@@` 1-bit land/sea bitmap."""
    dst.ensure_shape(ni, nj)
    var total = ni * nj
    var n_pad = _padded_f32_len(total)
    var rp = raw_f32.unsafe_ptr()
    var dp = dst.data.unsafe_ptr()
    for idx in range(0, n_pad, SIMD_W):
        dp.unsafe_store(idx, rp.unsafe_load[width=SIMD_W](idx))

    var n_words = _num_mask_words(total)
    dst.ensure_bitmask_uninit()
    var mp = dst.mask_words.unsafe_ptr()
    var bp = raw_bitmap.unsafe_ptr()
    for w in range(n_words):
        mp.unsafe_store(w, bp.unsafe_load(w))
    dst.finalize_mask_state()


def decode_uint16_qflag_into(
    mut dst: MaskedSlab,
    raw_u16: List[UInt16],
    raw_qflag: List[UInt8],
    ni: Int,
    nj: Int,
    scale: Float32,
    offset: Float32,
    max_valid_qflag: UInt8 = UInt8(1),
):
    """Decode quantized UInt16 layer paired with UInt8 quality flag (`qflag > 1` -> masked)."""
    dst.ensure_shape(ni, nj)
    var total = ni * nj
    var n_pad = _padded_f32_len(total)
    var rp = raw_u16.unsafe_ptr()
    var qp = raw_qflag.unsafe_ptr()
    var dp = dst.data.unsafe_ptr()
    var sv = SIMD[DType.float32, SIMD_W](scale)
    var ov = SIMD[DType.float32, SIMD_W](offset)
    var max_qv = SIMD[DType.uint8, SIMD_W](max_valid_qflag)

    var any_bad_qc = False
    for idx in range(0, n_pad, SIMD_W):
        var q = rp.unsafe_load[width=SIMD_W](idx)
        var qf = qp.unsafe_load[width=SIMD_W](idx)
        dp.unsafe_store(idx, q.cast[DType.float32]() * sv + ov)
        if _bool8_to_byte(qf.gt(max_qv)) != UInt64(0):
            any_bad_qc = True

    if not any_bad_qc:
        dst.mask_state = MASK_ALL_VALID
        return

    dst.ensure_bitmask_zeroed()
    var n_words = _num_mask_words(total)
    for w in range(n_words):
        var base = w * 64
        var word_bits: UInt64 = 0
        for g in range(8):
            var off = base + g * 8
            if off < total:
                var qf = qp.unsafe_load[width=SIMD_W](off)
                word_bits |= _bool8_to_byte(qf.gt(max_qv)) << UInt64(g * 8)
        dst.mask_words[w] = word_bits
    dst.finalize_mask_state()


# ===----------------------------------------------------------------------=== #
# In-Memory Dummy ECCC GDPS Layer Generator (Zero Disk I/O)
# ===----------------------------------------------------------------------=== #


@always_inline
def _pseudo_terrain_hpa(lon_deg: Float32, lat_deg: Float32) -> Float32:
    """Synthetic surface pressure (hPa) mimicking Rockies, Laurentians, Andes & Himalayas."""
    var p_sfc = Float32(1013.25)
    # Western Cordillera / Rockies (lon 235..255, lat 35..62) -> ~760 hPa peaks
    if (
        lon_deg >= Float32(235.0)
        and lon_deg <= Float32(255.0)
        and lat_deg >= Float32(35.0)
        and lat_deg <= Float32(62.0)
    ):
        p_sfc = Float32(780.0)
    # Laurentians / Labrador highlands (lon 285..298, lat 47..56) -> ~935 hPa
    elif (
        lon_deg >= Float32(285.0)
        and lon_deg <= Float32(298.0)
        and lat_deg >= Float32(47.0)
        and lat_deg <= Float32(56.0)
    ):
        p_sfc = Float32(935.0)
    # Tibetan Plateau / Himalayas (lon 75..100, lat 27..38) -> ~580 hPa
    elif (
        lon_deg >= Float32(75.0)
        and lon_deg <= Float32(100.0)
        and lat_deg >= Float32(27.0)
        and lat_deg <= Float32(38.0)
    ):
        p_sfc = Float32(580.0)
    return p_sfc


@always_inline
def _is_land_mask(lon_deg: Float32, lat_deg: Float32) -> Bool:
    """Synthetic North America + Eurasia landmass mask for SST (`TM`) bitmap."""
    # North American continent (lon 235..302, lat 25..72), leaving Atlantic (>302) as ocean
    if (
        lon_deg >= Float32(235.0)
        and lon_deg <= Float32(296.0)
        and lat_deg >= Float32(25.0)
        and lat_deg <= Float32(72.0)
    ):
        return True
    # Eurasia block
    if (
        lon_deg >= Float32(10.0)
        and lon_deg <= Float32(135.0)
        and lat_deg >= Float32(20.0)
        and lat_deg <= Float32(70.0)
    ):
        return True
    return False


def synthesize_and_decode_gdps_subslab(
    var_id: Int,
    step_hr: Float32,
    level_hpa: Float32,
    full_ni: Int,
    full_nj: Int,
    i0: Int,
    i1: Int,
    j0: Int,
    j1: Int,
    mut scratch: RawLayerScratch,
    mut dst: MaskedSlab,
):
    """Synthesize in-memory encoded GDPS sub-slab `[i0:i1, j0:j1]` in Fortran order
    and immediately run the variable's encoding-specific decoder into `dst`.
    """
    var sub_ni = i1 - i0
    var sub_nj = j1 - j0
    var total = sub_ni * sub_nj
    scratch.ensure_capacity(total)

    var dlon = Float32(360.0) / Float32(full_ni)
    var dlat = Float32(180.0) / Float32(full_nj - 1 if full_nj > 1 else 1)
    var phase = step_hr * Float32(0.035)

    if (
        var_id == VAR_TT
        or var_id == VAR_UU
        or var_id == VAR_VV
    ):
        # ENC_QUANT16_SENTINEL: 16-bit packed with 0xFFFF underground terrain mask
        var scale = Float32(0.01)
        var offset = Float32(-120.0)
        var u16_p = scratch.u16_buf.unsafe_ptr()
        for jj in range(sub_nj):
            var lat = Float32(-90.0) + Float32(j0 + jj) * dlat
            var lat_rad = lat * Float32(0.0174532925)
            var col_off = jj * sub_ni
            var meridional = cos(lat_rad)
            var jet = sin(lat_rad * Float32(2.0))
            for ii in range(sub_ni):
                var lon = Float32(i0 + ii) * dlon
                if level_hpa > _pseudo_terrain_hpa(lon, lat):
                    # Underground isobaric level -> 0xFFFF missing sentinel!
                    u16_p.unsafe_store(col_off + ii, UInt16(0xFFFF))
                else:
                    var wave = Float32(
                        ((i0 + ii) * 7 + (j0 + jj) * 13) & 63
                    ) * Float32(0.08)
                    var phys: Float32
                    if var_id == VAR_TT:
                        # Realistic lapse rate from +22C near 1000 hPa to -58C aloft
                        var lapse = (level_hpa - Float32(500.0)) * Float32(0.075)
                        phys = (
                            Float32(-18.0)
                            + Float32(32.0) * meridional
                            + lapse
                            + wave
                            - phase
                        )
                    elif var_id == VAR_UU:
                        # Westerly jet peaking aloft (knots)
                        var alt_boost = (
                            Float32(1015.0) - level_hpa
                        ) * Float32(0.055)
                        phys = Float32(12.0) + Float32(38.0) * jet + alt_boost + wave
                    else:
                        # Synoptic Rossby wave v-wind (knots)
                        phys = Float32(-8.0) + Float32(18.0) * jet + wave + phase
                    var q_int = Int((phys - offset) / scale + Float32(0.5))
                    if q_int < 0:
                        q_int = 0
                    elif q_int > 65534:
                        q_int = 65534
                    u16_p.unsafe_store(col_off + ii, UInt16(q_int))

        decode_quant16_sentinel_into(
            dst, scratch.u16_buf, sub_ni, sub_nj, scale, offset
        )

    elif var_id == VAR_TM:
        # ENC_FLOAT32_BITMAP: Sea Surface Temperature (degC) + 1-bit land/sea bitmap
        var n_words = _num_mask_words(total)
        var f32_p = scratch.f32_buf.unsafe_ptr()
        var bm_p = scratch.bitmap_buf.unsafe_ptr()
        for w in range(n_words):
            bm_p.unsafe_store(w, UInt64(0))

        for jj in range(sub_nj):
            var lat = Float32(-90.0) + Float32(j0 + jj) * dlat
            var abs_lat = -lat if lat < Float32(0.0) else lat
            var base_sst = Float32(28.5) - abs_lat * Float32(0.36)
            if base_sst < Float32(-1.8):
                base_sst = Float32(-1.8)
            var col_off = jj * sub_ni
            for ii in range(sub_ni):
                var lon = Float32(i0 + ii) * dlon
                var flat = col_off + ii
                f32_p.unsafe_store(
                    flat,
                    base_sst + Float32(((i0 + ii) & 15)) * Float32(0.06),
                )
                if _is_land_mask(lon, lat):
                    var w = flat >> 6
                    var b = flat & 63
                    bm_p.unsafe_store(
                        w, bm_p.unsafe_load(w) | (UInt64(1) << UInt64(b))
                    )

        decode_float32_bitmap_into(
            dst, scratch.f32_buf, scratch.bitmap_buf, sub_ni, sub_nj
        )

    else:
        # VAR_HU -> ENC_UINT16_QFLAG: Specific Humidity (g/kg) + UInt8 QC flag
        var scale = Float32(0.001)
        var offset = Float32(0.0)
        var u16_p = scratch.u16_buf.unsafe_ptr()
        var qf_p = scratch.u8_buf.unsafe_ptr()
        for jj in range(sub_nj):
            var lat = Float32(-90.0) + Float32(j0 + jj) * dlat
            var abs_lat = -lat if lat < Float32(0.0) else lat
            var base_hu = (Float32(18.0) - abs_lat * Float32(0.18)) * (
                level_hpa / Float32(1000.0)
            )
            if base_hu < Float32(0.2):
                base_hu = Float32(0.2)
            var col_off = jj * sub_ni
            for ii in range(sub_ni):
                var flat = col_off + ii
                var hu_val = base_hu + Float32(((i0 + ii) & 7)) * Float32(0.12)
                var q_int = Int((hu_val - offset) / scale + Float32(0.5))
                u16_p.unsafe_store(flat, UInt16(q_int))
                # Deterministic ~3% sensor QC flag=2 patches
                var hash_qc = ((i0 + ii) * 31 + (j0 + jj) * 17 + Int(step_hr)) & 63
                qf_p.unsafe_store(flat, UInt8(2) if hash_qc == 0 else UInt8(0))

        decode_uint16_qflag_into(
            dst, scratch.u16_buf, scratch.u8_buf, sub_ni, sub_nj, scale, offset
        )
