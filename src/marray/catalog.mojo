"""M1 `RecordCatalog`, Multi-File POSIX `pread` Pool & C-ABI `ChunkDecoder` Engine.

Implements the production contracts across `docs/questions.md` (Q1-Q5) and `CONTEXT.md` (P0):
  1. **Multi-File POSIX `pread` Pool (`CatalogFilePool`):** Opens `1..N` forecast lead
     files (`O_RDONLY`) once in C/Mojo heap (no Python `np.memmap` LRU) and performs
     thread-safe `pread` of ONLY touched `(t, z)` record byte spans into worker-local
     scratch buffers, optionally prepending 80 zero bytes (`FstRecordCodec` convention).
  2. **Separate-`.so` C-ABI `ChunkDecoder` (`ChunkDecoderCABI`):** Connects `marray`
     to `boreal-io` (`cmc_io_fst` `unpack_*_from_raw` or path/addr bridge) without
     constructing Dask `HighLevelGraph`s or unpacking untouched records.
  3. **RPN Missingness Fidelity:** Supports companion `@@` / `!@` `float32` `0/1`
     bitmap planes (`datyp=2, nbits=1`, keep `> 0`), alt-mask max-value sentinels
     (`datyp + 64`, `64 <= datyp < 128`), and `datyp=6` / GIL serial safety guards.
  4. **Per-Worker `PlaneSlotCache`:** Guarantees whole-record `datyp=134` planes are
     decoded ONCE per `(t, z)` slab across regional `[j0:j1, i0:i1]` slices while
     capping RAM at `n_workers * plane_bytes`.
"""

from std.ffi import external_call
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

# 10 Int64s per record:
#   [0] file_id (-1 = in-RAM `raw_base_addr + d_off`, >= 0 = `CatalogFilePool` fd index)
#   [1] d_off   (byte offset for `pread`/RAM, or RPN `addr` in `CAT_FLAG_ADDR_MODE`)
#   [2] d_len   (byte length for `pread`/RAM, or RPN `lng` in `CAT_FLAG_ADDR_MODE`)
#   [3] datyp   (RPN datyp; `64 <= datyp < 128` carries alt-mask `+64` max-sentinel bit)
#   [4] nbits
#   [5] bmap_file_id (-1 = in-RAM or none)
#   [6] bm_off  (-1 = no companion `@@` / `!@` bitmap)
#   [7] bm_len
#   [8] b_dt
#   [9] b_nb
comptime REC_META_STRIDE: Int = 10

comptime CAT_FLAG_FORCE_SERIAL: Int = 1   # bit 0: serialize worker decode (e.g. GIL callback / datyp=6)
comptime CAT_FLAG_ADDR_MODE: Int = 2      # bit 1: pass `(file_id, addr, lng)` directly to decoder without `pread`


struct CatalogFilePool(Movable):
    """Persistent read-only POSIX file-descriptor table for multi-file FST suites."""

    var fds: List[Int32]

    def __init__(
        out self,
        paths_blob: ImmPointer[UInt8, ImmUntrackedOrigin],
        path_offsets: ImmPointer[Int64, ImmUntrackedOrigin],
        n_files: Int,
    ):
        self.fds = List[Int32](capacity=max(1, n_files))
        for i in range(n_files):
            var off = Int(path_offsets[unsafe_offset=i])
            var c_path = paths_blob.unsafe_offset(off)
            # POSIX `O_RDONLY == 0` on Linux and macOS
            var fd = external_call["open", Int32](c_path, Int32(0))
            self.fds.append(fd)

    def __init__(out self, *, deinit move: Self):
        self.fds = move.fds^

    def close_all(mut self):
        for i in range(len(self.fds)):
            var fd = self.fds[i]
            if fd >= 0:
                _ = external_call["close", Int32](fd)
                self.fds[i] = Int32(-1)

    def __deinit__(deinit self):
        for i in range(len(self.fds)):
            var fd = self.fds[i]
            if fd >= 0:
                _ = external_call["close", Int32](fd)


@inline(.always)
def _pread_exact(
    fd: Int32,
    dst: MutPointer[UInt8, _],
    n_bytes: Int,
    file_offset: Int,
) -> Int:
    """Thread-safe POSIX `pread` loop handling short reads cleanly."""
    var done = 0
    while done < n_bytes:
        var got = external_call["pread", Int](
            fd,
            dst.unsafe_offset(done),
            n_bytes - done,
            Int64(file_offset + done),
        )
        if got <= 0:
            break
        done += got
    return done


@inline(.always)
def is_datyp_parallel_safe(datyp: Int) -> Bool:
    """Strip alt-mask bit (`+64` when `< 128`) and check Mojo parallel codec support (`Q2`)."""
    var base_dt = (datyp - 64) if (datyp >= 64 and datyp < 128) else datyp
    return (
        base_dt == 0
        or base_dt == 1
        or base_dt == 2
        or base_dt == 4
        or base_dt == 5
        or base_dt == 130
        or base_dt == 133
        or base_dt == 134
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
    """Reference C-ABI whole-record codec (`nelm = ni * nj * nk`):
      - `datyp` 0, 1, 5: IEEE `float32` raw plane copy
      - `datyp` 2, `nbits == 1`: RPN `@@` 1-bit packed bitmap -> `float32` `0.0` / `1.0`
      - `datyp` 2 or 4, `nbits == 16`: `[min_f32, scale_f32]` + packed `UInt16` plane
      - `datyp` 134 / 130 / 133: Whole-record 64-cell block-floating-point `UInt16`
    """
    var nelm = Int(ni) * Int(nj) * max(1, Int(nk))
    if nelm <= 0:
        return Int32(0)

    var raw_dt = Int(datyp)
    var dt = (raw_dt - 64) if (raw_dt >= 64 and raw_dt < 128) else raw_dt
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


struct CatalogSourceDesc(Copyable, Movable):
    """Metadata descriptor for a `RecordCatalog` variable backed by in-RAM or multi-file `pread` planes."""

    var raw_base_addr: Int
    var rec_meta_addr: Int      # `n_records * 10` Int64 table
    var n_records: Int
    var plane_ni: Int
    var plane_nj: Int
    var rec_s_z: Int            # `0` for 2D/3D surface broadcast across `vertical`!
    var rec_s_t: Int            # `0` for static fields!
    var decoder_fn_addr: Int
    var decoder_ctx_addr: Int
    var file_pool_addr: Int     # `CatalogFilePool*` when reading multi-file `.std` via `pread`
    var flags: Int              # bit 0 = force_serial, bit 1 = addr_mode, bits 8..15 = prefix_zeros
    var scale: Float32
    var offset: Float32

    def __init__(
        out self,
        raw_base_addr: Int,
        rec_meta_addr: Int,
        n_records: Int,
        plane_ni: Int,
        plane_nj: Int,
        rec_s_z: Int,
        rec_s_t: Int,
        decoder_fn_addr: Int,
        decoder_ctx_addr: Int,
        scale: Float32,
        offset: Float32,
        file_pool_addr: Int = 0,
        flags: Int = 0,
    ):
        self.raw_base_addr = raw_base_addr
        self.rec_meta_addr = rec_meta_addr
        self.n_records = n_records
        self.plane_ni = plane_ni
        self.plane_nj = plane_nj
        self.rec_s_z = rec_s_z
        self.rec_s_t = rec_s_t
        self.decoder_fn_addr = decoder_fn_addr
        self.decoder_ctx_addr = decoder_ctx_addr
        self.file_pool_addr = file_pool_addr
        self.flags = flags
        self.scale = scale
        self.offset = offset

    def __init__(out self, *, copy: Self):
        self.raw_base_addr = copy.raw_base_addr
        self.rec_meta_addr = copy.rec_meta_addr
        self.n_records = copy.n_records
        self.plane_ni = copy.plane_ni
        self.plane_nj = copy.plane_nj
        self.rec_s_z = copy.rec_s_z
        self.rec_s_t = copy.rec_s_t
        self.decoder_fn_addr = copy.decoder_fn_addr
        self.decoder_ctx_addr = copy.decoder_ctx_addr
        self.file_pool_addr = copy.file_pool_addr
        self.flags = copy.flags
        self.scale = copy.scale
        self.offset = copy.offset

    def __init__(out self, *, deinit move: Self):
        self.raw_base_addr = move.raw_base_addr
        self.rec_meta_addr = move.rec_meta_addr
        self.n_records = move.n_records
        self.plane_ni = move.plane_ni
        self.plane_nj = move.plane_nj
        self.rec_s_z = move.rec_s_z
        self.rec_s_t = move.rec_s_t
        self.decoder_fn_addr = move.decoder_fn_addr
        self.decoder_ctx_addr = move.decoder_ctx_addr
        self.file_pool_addr = move.file_pool_addr
        self.flags = move.flags
        self.scale = move.scale
        self.offset = move.offset

    def is_parallel_safe(self) -> Bool:
        if (self.flags & CAT_FLAG_FORCE_SERIAL) != 0:
            return False
        var meta = ImmPointer[Int64, ImmUntrackedOrigin](
            unsafe_from_address=self.rec_meta_addr
        )
        for r in range(self.n_records):
            var b = r * REC_META_STRIDE
            var dt = Int(meta[unsafe_offset=b + 3])
            if not is_datyp_parallel_safe(dt):
                return False
            var bm_off = Int(meta[unsafe_offset=b + 6])
            if bm_off >= 0:
                var bdt = Int(meta[unsafe_offset=b + 8])
                if not is_datyp_parallel_safe(bdt):
                    return False
        return True


struct PlaneSlotCache(Movable):
    """Per-worker 1-plane L2/L3 slot cache + reusable POSIX `pread` compressed scratch."""

    var plane_elems: Int
    var cached_rec_idx: Int
    var cached_has_bmap: Bool
    var decode_calls: Int
    var _data_alloc: Allocation[Float32, alignment=CACHE_ALIGN]
    var _bmap_alloc: Allocation[Float32, alignment=CACHE_ALIGN]
    var _raw_scratch: List[UInt8]

    def __init__(out self, plane_elems: Int):
        var n_alloc = max(8, plane_elems)
        self.plane_elems = plane_elems
        self.cached_rec_idx = -1
        self.cached_has_bmap = False
        self.decode_calls = 0
        self._data_alloc = alloc(
            Layout[Float32, alignment=CACHE_ALIGN](count=n_alloc)
        )
        self._bmap_alloc = alloc(
            Layout[Float32, alignment=CACHE_ALIGN](count=n_alloc)
        )
        self._raw_scratch = List[UInt8]()

    def __init__(out self, *, deinit move: Self):
        self.plane_elems = move.plane_elems
        self.cached_rec_idx = move.cached_rec_idx
        self.cached_has_bmap = move.cached_has_bmap
        self.decode_calls = move.decode_calls
        self._data_alloc = move._data_alloc^
        self._bmap_alloc = move._bmap_alloc^
        self._raw_scratch = move._raw_scratch^

    def __deinit__(deinit self):
        dealloc(self._data_alloc^)
        dealloc(self._bmap_alloc^)

    def _decode_one_record(
        mut self,
        desc: CatalogSourceDesc,
        file_id: Int,
        rec_off: Int,
        rec_len: Int,
        datyp: Int,
        nbits: Int,
        dst_f32: MutPointer[Float32, MutUntrackedOrigin],
    ):
        var addr_mode = (desc.flags & CAT_FLAG_ADDR_MODE) != 0
        if addr_mode:
            # Delegate `(file_id, rpn_addr, rpn_lng)` directly to external `boreal-io` bridge!
            # Encodes `file_id` in the upper 16 bits of `nk` (`(file_id << 8) | 1`) so XDF & RSF
            # multi-file suites work zero-copy through `ChunkDecoderCABI`!
            var packed_nk = (file_id << 8) | 1
            var dummy_p = ImmPointer[UInt8, ImmUntrackedOrigin](
                unsafe_from_address=Int(dst_f32)
            )
            _ = invoke_c_chunk_decoder(
                desc.decoder_fn_addr,
                desc.decoder_ctx_addr + rec_off,  # passes `addr` via `ctx_addr` when `ctx_addr==0` or table
                dummy_p,
                rec_len,
                datyp,
                nbits,
                desc.plane_ni,
                desc.plane_nj,
                packed_nk,
                dst_f32,
            )
            return

        if file_id >= 0 and desc.file_pool_addr != 0:
            var prefix_z = (desc.flags >> 8) & 0xFF
            var need_sz = prefix_z + rec_len
            if len(self._raw_scratch) < need_sz:
                self._raw_scratch.resize(need_sz, UInt8(0))
            for pz in range(prefix_z):
                self._raw_scratch[pz] = UInt8(0)
            var pool_p = ImmPointer[CatalogFilePool, ImmUntrackedOrigin](
                unsafe_from_address=desc.file_pool_addr
            )
            var fd = pool_p[].fds[file_id]
            _ = _pread_exact(
                fd,
                self._raw_scratch.unsafe_ptr().unsafe_offset(prefix_z),
                rec_len,
                rec_off,
            )
            var raw_p = ImmPointer[UInt8, ImmUntrackedOrigin](
                unsafe_from_address=Int(self._raw_scratch.unsafe_ptr())
            )
            _ = invoke_c_chunk_decoder(
                desc.decoder_fn_addr,
                desc.decoder_ctx_addr,
                raw_p,
                need_sz,
                datyp,
                nbits,
                desc.plane_ni,
                desc.plane_nj,
                1,
                dst_f32,
            )
            _ = self._raw_scratch
            return

        # In-RAM / mmap zero-copy fast path (`file_id < 0`):
        var raw_p = ImmPointer[UInt8, ImmUntrackedOrigin](
            unsafe_from_address=desc.raw_base_addr + rec_off
        )
        _ = invoke_c_chunk_decoder(
            desc.decoder_fn_addr,
            desc.decoder_ctx_addr,
            raw_p,
            rec_len,
            datyp,
            nbits,
            desc.plane_ni,
            desc.plane_nj,
            1,
            dst_f32,
        )

    @inline(.always)
    def ensure_decoded(mut self, desc: CatalogSourceDesc, rec_idx: Int):
        if self.cached_rec_idx == rec_idx:
            return
        var meta = ImmPointer[Int64, ImmUntrackedOrigin](
            unsafe_from_address=desc.rec_meta_addr + rec_idx * REC_META_STRIDE * 8
        )
        var f_id = Int(meta[unsafe_offset=0])
        var d_off = Int(meta[unsafe_offset=1])
        var d_len = Int(meta[unsafe_offset=2])
        var dt = Int(meta[unsafe_offset=3])
        var nb = Int(meta[unsafe_offset=4])
        var bf_id = Int(meta[unsafe_offset=5])
        var bm_off = Int(meta[unsafe_offset=6])
        var bm_len = Int(meta[unsafe_offset=7])
        var b_dt = Int(meta[unsafe_offset=8])
        var b_nb = Int(meta[unsafe_offset=9])

        # Missing slot in ragged `pad` mode (`d_len <= 0`): mark whole plane masked (`0.0` bitmap)
        var dst_d = MutPointer[Float32, MutUntrackedOrigin](
            unsafe_from_address=Int(self._data_alloc.unsafe_ptr())
        )
        var dst_b = MutPointer[Float32, MutUntrackedOrigin](
            unsafe_from_address=Int(self._bmap_alloc.unsafe_ptr())
        )
        if d_len <= 0:
            for k in range(self.plane_elems):
                dst_d[unsafe_offset=k] = Float32(0.0)
                dst_b[unsafe_offset=k] = Float32(0.0)
            self.cached_has_bmap = True
            self.cached_rec_idx = rec_idx
            return

        self._decode_one_record(desc, f_id, d_off, d_len, dt, nb, dst_d)
        self.decode_calls += 1

        var has_bm = False
        if bm_off >= 0 and bm_len > 0:
            self._decode_one_record(desc, bf_id, bm_off, bm_len, b_dt, b_nb, dst_b)
            has_bm = True

        # Alt-mask (`datyp + 64`, `05-masking-missing-data.md`): max unpacked value marks missing!
        if dt >= 64 and dt < 128 and self.plane_elems > 0:
            var max_v = dst_d[unsafe_offset=0]
            for k in range(1, self.plane_elems):
                max_v = max(max_v, dst_d[unsafe_offset=k])
            if not has_bm:
                for k in range(self.plane_elems):
                    dst_b[unsafe_offset=k] = (
                        Float32(0.0)
                        if dst_d[unsafe_offset=k] >= max_v
                        else Float32(1.0)
                    )
                has_bm = True
            else:
                for k in range(self.plane_elems):
                    if dst_d[unsafe_offset=k] >= max_v:
                        dst_b[unsafe_offset=k] = Float32(0.0)

        self.cached_has_bmap = has_bm
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


@inline(.always)
def load_morsel_from_predecoded_catalog(
    desc: CatalogSourceDesc,
    pre_slots_ptr: ImmPointer[PlaneSlotCache, _],
    cat_lookup_ptr: ImmPointer[Int, _],
    d0_start: Int,
    d0_len: Int,
    d1_start: Int,
    d1_len: Int,
    d2_len: Int,
    flat_offset: Int,
    count: Int,
    mut dst: MorselReg,
):
    """Zero-allocation read-only morsel loader over shared pre-decoded `(z, t)` planes.
    Eliminates redundant `16x` per-worker `pread` + `datyp=134` whole-plane decodes on thin
    2D peels (`polaris` / `mogc-api` `R-P0`) and unlocks 16-thread SIMD compute even when
    the upstream chunk decoder itself is serial (`CAT_FLAG_FORCE_SERIAL` / `datyp=6`).
    """
    var mp = dst.mask_ptr()
    var n_words = ceildiv(count, 64)
    for w in range(n_words):
        mp[unsafe_offset=w] = UInt64(0)
    var or_bits: UInt64 = 0

    var full_plane = (
        d0_start == 0
        and d0_len == desc.plane_ni
        and d1_start == 0
        and d1_len == desc.plane_nj
    )
    var plane_sz = desc.plane_ni * desc.plane_nj

    var written = 0
    while written < count:
        var cur = flat_offset + written
        var rem0, i_rel = divmod(cur, d0_len)
        var rem1, j_rel = divmod(rem0, d1_len)
        var t_rel, z_rel = divmod(rem1, d2_len)

        var slot_idx = cat_lookup_ptr[unsafe_offset=t_rel * d2_len + z_rel]
        ref slot = pre_slots_ptr[unsafe_offset=slot_idx]

        var span: Int
        var plane_off: Int
        if full_plane:
            plane_off = j_rel * desc.plane_ni + i_rel
            span = min(plane_sz - plane_off, count - written)
        else:
            plane_off = (d1_start + j_rel) * desc.plane_ni + (d0_start + i_rel)
            span = min(d0_len - i_rel, count - written)

        var p_data = ImmPointer[Float32, ImmUntrackedOrigin](
            unsafe_from_address=Int(slot._data_alloc.unsafe_ptr()) + plane_off * 4
        )
        var p_bmap_u8 = ImmPointer[UInt8, ImmUntrackedOrigin](
            unsafe_from_address=Int(slot._bmap_alloc.unsafe_ptr()) + plane_off * 4
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
