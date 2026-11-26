"""M1 `RecordCatalog` & C-ABI `ChunkDecoder` Whole-Record Plane Decode Engine.

Implements the production boundaries clarified in `docs/questions.md` (Q1-Q5):
  1. Separate-`.so` C-ABI `ChunkDecoder` function pointer (`ChunkDecoderCABI`):
     connects `marray` to `boreal-io` (`cmc_io_fst` `unpack_*_from_raw`) without
     importing Python extension modules inside the morsel VM.
  2. `datyp` thread-safety contract (`is_datyp_parallel_safe`): `0,1,2,4,5,130,133,134`
     execute in parallel across Mojo workers; `datyp=6` forces serial / pre-decoded safety.
  3. Native `float32` `@@` bitmap planes (`datyp=2, nbits=1` -> `0.0` / `1.0` `Float32`
     planes, matching `compact_integer_unpack_f32` today) converted into `MSTATE_BITS`
     in a single SIMD pass with zero Python `np.where`.
  4. Whole-record plane decode (`nelm = ni * nj * nk`) into per-worker L2/L3-resident
     `PlaneSlotCache` (`cached_rec_idx`), so regional `[j0:j1, i0:i1]` slices and
     multi-morsel scans decompress each `(t, z)` plane ONCE while bounding RAM to
     `n_workers * plane_bytes` instead of full `5+ GB` cube allocations.
"""

from std.math import ceildiv, clamp, fma, max, min
from std.memory import (
    Allocation,
    ImmPointer,
    Layout,
    MutPointer,
    alloc,
    dealloc,
)
from std.sys import size_of

from .morsel import CACHE_ALIGN, MSRC_FST_BITMAP_F32, MSRC_NONE, MorselReg, load_span_f32


# Stable C-ABI `ChunkDecoder` contract between `boreal-io` `.so` and `marray` `.so`:
#   (ctx_addr, raw_ptr, raw_byte_len, datyp, nbits, ni, nj, nk, dst_f32) -> status (0 = OK)
comptime ChunkDecoderCABI = def(
    Int64,
    ImmPointer[UInt8, ImmUntrackedOrigin],
    Int64,
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    MutPointer[Float32, MutUntrackedOrigin],
) thin abi("C") -> Int32

comptime REC_META_STRIDE: Int = 8  # [off, len, datyp, nbits, bmap_off, bmap_len, bmap_datyp, bmap_nbits]


@inline(.always)
def is_datyp_parallel_safe(datyp: Int) -> Bool:
    """`boreal-io` Mojo codecs cover `0,1,2,4,5,130,133,134` thread-safely when
    destinations do not overlap (`docs/questions.md` Q2). `datyp == 6` may invoke
    legacy/Python packers and must NOT run concurrently inside `parallelize`.
    """
    return (
        datyp == 0
        or datyp == 1
        or datyp == 2
        or datyp == 4
        or datyp == 5
        or datyp == 130
        or datyp == 133
        or datyp == 134
    )


@inline(.always)
def invoke_c_chunk_decoder(
    fn_addr: Int,
    ctx_addr: Int,
    raw_ptr: ImmPointer[UInt8, ImmUntrackedOrigin],
    raw_byte_len: Int,
    datyp: Int,
    nbits: Int,
    ni: Int,
    nj: Int,
    nk: Int,
    dst_f32: MutPointer[Float32, MutUntrackedOrigin],
) -> Int32:
    """Invoke external C-ABI `ChunkDecoder` function pointer or built-in codec."""
    if fn_addr == 0:
        return builtin_fst_chunk_decode(
            Int64(ctx_addr),
            raw_ptr,
            Int64(raw_byte_len),
            Int32(datyp),
            Int32(nbits),
            Int32(ni),
            Int32(nj),
            Int32(nk),
            dst_f32,
        )
    comptime assert size_of[ChunkDecoderCABI]() == size_of[Int](), "thin abi(C) fn must match pointer width"
    var addr_slot = fn_addr
    var fn_ptr = Pointer(to=addr_slot).unsafe_bitcast[ChunkDecoderCABI]()[]
    return fn_ptr(
        Int64(ctx_addr),
        raw_ptr,
        Int64(raw_byte_len),
        Int32(datyp),
        Int32(nbits),
        Int32(ni),
        Int32(nj),
        Int32(nk),
        dst_f32,
    )


def builtin_fst_chunk_decode(
    ctx_addr: Int64,
    raw_ptr: ImmPointer[UInt8, ImmUntrackedOrigin],
    raw_byte_len: Int64,
    datyp: Int32,
    nbits: Int32,
    ni: Int32,
    nj: Int32,
    nk: Int32,
    dst_f32: MutPointer[Float32, MutUntrackedOrigin],
) -> Int32:
    """Reference C-ABI whole-record codec (`nelm = ni * nj * nk`) modeling `cmc_io_fst`:
      - `datyp` 0, 1, 5: IEEE `float32` raw plane copy
      - `datyp` 2, `nbits == 1`: RPN `@@` 1-bit packed bitmap -> `float32` `0.0` / `1.0`
        plane (matching `compact_integer_unpack_f32` in `docs/questions.md` Q3!)
      - `datyp` 2 or 4, `nbits == 16`: `[min_f32, scale_f32]` + packed `UInt16` plane
      - `datyp` 134 / 130 / 133: Whole-record 64-cell block-floating-point `UInt16`
        quantizer (8-byte `[min_f32, scale_f32]` block header + 64 `UInt16` mantissas
        per block -> ~1.9x compression, whole-plane SIMD dequantization).
    """
    var nelm = Int(ni) * Int(nj) * max(1, Int(nk))
    if nelm <= 0:
        return Int32(0)

    var dt = Int(datyp)
    var nb = Int(nbits)

    if dt == 0 or dt == 1 or dt == 5:
        var src_f = raw_ptr.unsafe_bitcast[Float32]()
        var vec_end = (nelm // 8) * 8
        var k = 0
        while k < vec_end:
            dst_f32.unsafe_store(k, src_f.unsafe_load[width=8](k))
            k += 8
        while k < nelm:
            dst_f32[unsafe_offset=k] = src_f[unsafe_offset=k]
            k += 1
        return Int32(0)

    if dt == 2 and nb == 1:
        # Unpack 1-bit `@@` bitmap into `float32` `0.0` / `1.0` plane (`compact_integer_unpack_f32`)
        var ones = SIMD[.float32, 8](1.0)
        var zeros = SIMD[.float32, 8](0.0)
        comptime lanes = SIMD[DType.uint8, 8](1, 2, 4, 8, 16, 32, 64, 128)
        var n_full_bytes = nelm // 8
        for b_i in range(n_full_bytes):
            var byte_v = raw_ptr[unsafe_offset=b_i]
            var m8 = (SIMD[DType.uint8, 8](byte_v) & lanes).ne(0)
            dst_f32.unsafe_store(b_i * 8, m8.select(ones, zeros))
        var rem_start = n_full_bytes * 8
        if rem_start < nelm:
            var tail_b = raw_ptr[unsafe_offset=n_full_bytes]
            for bit in range(nelm - rem_start):
                var is_one = ((tail_b >> UInt8(bit)) & UInt8(1)) != UInt8(0)
                dst_f32[unsafe_offset=rem_start + bit] = (
                    Float32(1.0) if is_one else Float32(0.0)
                )
        return Int32(0)

    if (dt == 2 or dt == 4) and nb == 16:
        var hdr_f = raw_ptr.unsafe_bitcast[Float32]()
        var min_v = hdr_f[unsafe_offset=0]
        var scale_v = hdr_f[unsafe_offset=1]
        var q_ptr = raw_ptr.unsafe_offset(8).unsafe_bitcast[UInt16]()
        var sv = SIMD[.float32, 8](scale_v)
        var mv = SIMD[.float32, 8](min_v)
        var vec_end = (nelm // 8) * 8
        var k = 0
        while k < vec_end:
            var q = q_ptr.unsafe_load[width=8](k)
            dst_f32.unsafe_store(k, fma(q.cast[.float32](), sv, mv))
            k += 8
        while k < nelm:
            dst_f32[unsafe_offset=k] = fma(
                Float32(q_ptr[unsafe_offset=k]), scale_v, min_v
            )
            k += 1
        return Int32(0)

    # `datyp=134` (or `130`/`133`): 64-sample block-floating-point whole-record unpack!
    # Layout per 64-element block (`136` bytes): `[min_f32 (4B), scale_f32 (4B), q[64] UInt16 (128B)]`
    var n_blocks = ceildiv(nelm, 64)
    for blk in range(n_blocks):
        var blk_ptr = raw_ptr.unsafe_offset(blk * 136)
        var hdr = blk_ptr.unsafe_bitcast[Float32]()
        var b_min = hdr[unsafe_offset=0]
        var b_scale = hdr[unsafe_offset=1]
        var q_p = blk_ptr.unsafe_offset(8).unsafe_bitcast[UInt16]()
        var out_base = blk * 64
        var rem = nelm - out_base
        var sv = SIMD[.float32, 8](b_scale)
        var mv = SIMD[.float32, 8](b_min)
        if rem >= 64:
            comptime for g in range(8):
                var q8 = q_p.unsafe_load[width=8](g * 8)
                dst_f32.unsafe_store(
                    out_base + g * 8, fma(q8.cast[.float32](), sv, mv)
                )
        else:
            for lane in range(rem):
                dst_f32[unsafe_offset=out_base + lane] = fma(
                    Float32(q_p[unsafe_offset=lane]), b_scale, b_min
                )
    return Int32(0)


def pack_f32_plane_datyp134(
    src_f32: ImmPointer[Float32, _],
    nelm: Int,
    dst_bytes: MutPointer[UInt8, _],
) -> Int:
    """Pack a `Float32` `(ni, nj)` plane into 64-element `datyp=134` blocks (`136` B/block)."""
    var n_blocks = ceildiv(nelm, 64)
    for blk in range(n_blocks):
        var base = blk * 64
        var cnt = min(64, nelm - base)
        var b_min = src_f32[unsafe_offset=base]
        var b_max = b_min
        for i in range(1, cnt):
            var v = src_f32[unsafe_offset=base + i]
            b_min = min(b_min, v)
            b_max = max(b_max, v)
        var span = b_max - b_min
        var scale = span / Float32(65535.0) if span > Float32(1.0e-20) else Float32(0.0)
        var inv_s = Float32(65535.0) / span if span > Float32(1.0e-20) else Float32(0.0)

        var blk_out = dst_bytes.unsafe_offset(blk * 136)
        var hdr_out = blk_out.unsafe_bitcast[Float32]()
        hdr_out[unsafe_offset=0] = b_min
        hdr_out[unsafe_offset=1] = scale
        var q_out = blk_out.unsafe_offset(8).unsafe_bitcast[UInt16]()
        for i in range(cnt):
            var qf = clamp(
                (src_f32[unsafe_offset=base + i] - b_min) * inv_s + Float32(0.5),
                Float32(0.0),
                Float32(65535.0),
            )
            q_out[unsafe_offset=i] = UInt16(Int(qf))
        for i in range(cnt, 64):
            q_out[unsafe_offset=i] = UInt16(0)
    return n_blocks * 136


def pack_bmap_plane_datyp2_1bit(
    src_u8: ImmPointer[UInt8, _],
    nelm: Int,
    dst_bytes: MutPointer[UInt8, _],
) -> Int:
    """Pack a `0/1` `@@` validity plane into `datyp=2, nbits=1` packed bytes."""
    var n_bytes = ceildiv(nelm, 8)
    for b_i in range(n_bytes):
        var base = b_i * 8
        var lim = min(8, nelm - base)
        var acc: UInt8 = 0
        for bit in range(lim):
            if src_u8[unsafe_offset=base + bit] > UInt8(0):
                acc |= UInt8(1) << UInt8(bit)
        dst_bytes[unsafe_offset=b_i] = acc
    return n_bytes


@fieldwise_init
struct CatalogSourceDesc(Copyable, Movable):
    """Metadata descriptor for a `RecordCatalog` variable backed by whole-record FST planes."""

    var raw_base_addr: Int
    var rec_meta_addr: Int  # `n_records * 8` Int64 table
    var n_records: Int
    var plane_ni: Int
    var plane_nj: Int
    var rec_s_z: Int        # record index stride along `d2` (`vertical`), `0` for 2D/3D surface!
    var rec_s_t: Int        # record index stride along `d3` (`time`), `0` for static fields!
    var decoder_fn_addr: Int
    var decoder_ctx_addr: Int
    var scale: Float32
    var offset: Float32

    def is_parallel_safe(self) -> Bool:
        var meta = ImmPointer[Int64, ImmUntrackedOrigin](
            unsafe_from_address=self.rec_meta_addr
        )
        for r in range(self.n_records):
            var b = r * REC_META_STRIDE
            var dt = Int(meta[unsafe_offset=b + 2])
            if not is_datyp_parallel_safe(dt):
                return False
            var bm_off = Int(meta[unsafe_offset=b + 4])
            if bm_off >= 0:
                var bdt = Int(meta[unsafe_offset=b + 6])
                if not is_datyp_parallel_safe(bdt):
                    return False
        return True


struct PlaneSlotCache(Movable):
    """Per-worker 1-plane L2/L3 slot cache (`ni * nj` `Float32` payload + `@@` `Float32` bitmap).

    Guarantees whole-record `datyp=134` planes are decompressed ONCE per `(t, z)` slab
    even across 160+ L1 morsels or regional `[j0:j1, i0:i1]` sub-domain slices!
    """

    var plane_elems: Int
    var cached_rec_idx: Int
    var cached_has_bmap: Bool
    var _data_alloc: Allocation[Float32, alignment=CACHE_ALIGN]
    var _bmap_alloc: Allocation[Float32, alignment=CACHE_ALIGN]

    def __init__(out self, plane_elems: Int):
        var n_alloc = max(8, plane_elems)
        self.plane_elems = plane_elems
        self.cached_rec_idx = -1
        self.cached_has_bmap = False
        self._data_alloc = alloc(
            Layout[Float32, alignment=CACHE_ALIGN](count=n_alloc)
        )
        self._bmap_alloc = alloc(
            Layout[Float32, alignment=CACHE_ALIGN](count=n_alloc)
        )

    def __init__(out self, *, deinit move: Self):
        self.plane_elems = move.plane_elems
        self.cached_rec_idx = move.cached_rec_idx
        self.cached_has_bmap = move.cached_has_bmap
        self._data_alloc = move._data_alloc^
        self._bmap_alloc = move._bmap_alloc^

    def __deinit__(deinit self):
        dealloc(self._data_alloc^)
        dealloc(self._bmap_alloc^)

    @inline(.always)
    def ensure_decoded(mut self, desc: CatalogSourceDesc, rec_idx: Int):
        if self.cached_rec_idx == rec_idx:
            return
        var meta = ImmPointer[Int64, ImmUntrackedOrigin](
            unsafe_from_address=desc.rec_meta_addr + rec_idx * REC_META_STRIDE * 8
        )
        var d_off = Int(meta[unsafe_offset=0])
        var d_len = Int(meta[unsafe_offset=1])
        var dt = Int(meta[unsafe_offset=2])
        var nb = Int(meta[unsafe_offset=3])
        var bm_off = Int(meta[unsafe_offset=4])
        var bm_len = Int(meta[unsafe_offset=5])
        var b_dt = Int(meta[unsafe_offset=6])
        var b_nb = Int(meta[unsafe_offset=7])

        var raw_p = ImmPointer[UInt8, ImmUntrackedOrigin](
            unsafe_from_address=desc.raw_base_addr + d_off
        )
        var dst_d = MutPointer[Float32, MutUntrackedOrigin](
            unsafe_from_address=Int(self._data_alloc.unsafe_ptr())
        )
        _ = invoke_c_chunk_decoder(
            desc.decoder_fn_addr,
            desc.decoder_ctx_addr,
            raw_p,
            d_len,
            dt,
            nb,
            desc.plane_ni,
            desc.plane_nj,
            1,
            dst_d,
        )

        if bm_off >= 0 and bm_len > 0:
            var raw_bm = ImmPointer[UInt8, ImmUntrackedOrigin](
                unsafe_from_address=desc.raw_base_addr + bm_off
            )
            var dst_b = MutPointer[Float32, MutUntrackedOrigin](
                unsafe_from_address=Int(self._bmap_alloc.unsafe_ptr())
            )
            _ = invoke_c_chunk_decoder(
                desc.decoder_fn_addr,
                desc.decoder_ctx_addr,
                raw_bm,
                bm_len,
                b_dt,
                b_nb,
                desc.plane_ni,
                desc.plane_nj,
                1,
                dst_b,
            )
            self.cached_has_bmap = True
        else:
            self.cached_has_bmap = False

        self.cached_rec_idx = rec_idx


@inline(.always)
def load_morsel_from_catalog(
    desc: CatalogSourceDesc,
    mut slot: PlaneSlotCache,
    d0_start: Int,
    d0_len: Int,
    d1_start: Int,
    d1_len: Int,
    d2_start: Int,
    d2_len: Int,
    d3_start: Int,
    flat_offset: Int,
    count: Int,
    mut dst: MorselReg,
):
    """Stream `count` elements from whole-record decoded `(ni, nj)` planes into `dst`
    (handles both full planes AND regional `[j0:j1, i0:i1]` slices + stride-0 `vctype`
    surface->3D broadcasts!).
    """
    var mp = dst.mask_ptr()
    var n_words = ceildiv(count, 64)
    for w in range(n_words):
        mp[unsafe_offset=w] = UInt64(0)
    var or_bits: UInt64 = 0

    var full_plane = d0_start == 0 and d0_len == desc.plane_ni and d1_start == 0 and d1_len == desc.plane_nj
    var plane_sz = desc.plane_ni * desc.plane_nj

    var written = 0
    while written < count:
        var cur = flat_offset + written
        var rem0, i_rel = divmod(cur, d0_len)
        var rem1, j_rel = divmod(rem0, d1_len)
        var t_rel, z_rel = divmod(rem1, d2_len)

        var z_abs = d2_start + z_rel
        var t_abs = d3_start + t_rel
        var rec_idx = t_abs * desc.rec_s_t + z_abs * desc.rec_s_z
        slot.ensure_decoded(desc, rec_idx)

        var span: Int
        var plane_off: Int
        if full_plane:
            plane_off = j_rel * desc.plane_ni + i_rel
            span = min(plane_sz - plane_off, count - written)
        else:
            plane_off = (d1_start + j_rel) * desc.plane_ni + (d0_start + i_rel)
            span = min(d0_len - i_rel, count - written)

        var p_data = slot._data_alloc.unsafe_ptr().unsafe_offset(plane_off)
        var p_bmap_u8 = (
            slot._bmap_alloc.unsafe_ptr()
            .unsafe_offset(plane_off)
            .unsafe_bitcast[UInt8]()
        )
        var mkind = MSRC_FST_BITMAP_F32 if slot.cached_has_bmap else MSRC_NONE

        load_span_f32(
            p_data,
            p_bmap_u8,
            written,
            span,
            desc.scale,
            desc.offset,
            mkind,
            0,
            dst,
            or_bits,
        )
        written += span

    dst.finalize_words(count, or_bits)
