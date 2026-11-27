"""Separate-`.so` C-ABI `ChunkDecoder` Bridge for `boreal-io` (`cmc_io_fst.xdf.unpack_batch`).

Per `CONTEXT.md` P0 #2, `docs/questions.md` Q1, and `docs/boreal/13-cold-open-work-machine.md` §4 RC4:
  Wraps `cmc_io_fst.xdf.unpack_batch.unpack_one_f32` behind `ChunkDecoderCABI`
  (`@export("cmc_fst_chunk_decode_cabi")`) so `marray-mask` worker threads
  (`max.algorithm.parallelize` + `CatalogFilePool` POSIX `pread`) call production RPN
  `datyp` unpackers (`0, 1, 2, 4, 5, 130, 133, 134`) across `.so` boundaries with ZERO
  Python GIL acquisition.

Build automatically via `pixi run build-ext` (when `MARRAY_CMC_IO` is set), or manually:
  pixi run bash -c 'mojo build -O3 --emit shared-lib \
    -I "${MARRAY_CMC_IO}/src/mojo" \
    bridges/cmc_io_cabi_shim.mojo \
    -o build/libcmc_marray_bridge.so'
  export MARRAY_CMC_DECODER_SO="$PWD/build/libcmc_marray_bridge.so"
"""

from std.math import max
from std.memory import ImmPointer, MutPointer

from cmc_io_fst.xdf.unpack_batch import MutPtr, unpack_one_f32


@export("cmc_fst_chunk_decode_cabi")
def cmc_fst_chunk_decode_cabi(
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
    """Stable `ChunkDecoderCABI` entrypoint (`0` = success).

    `CatalogFilePool` `pread`s `raw_byte_len = rpn_lng * 8` bytes starting at file byte
    offset `(rpn_addr - 1) * 8` into `raw_ptr`. Within `raw_ptr`, byte offset `0` is
    RPN origin-1 word address `1` (`(1 - 1) * 8 == 0`), and `Int(raw_byte_len) >> 3`
    recovers the exact header `lng` while providing `rpn_lng * 8` readable bytes.
    Alt-mask (`64 <= datyp < 128`) is stripped here and applied by `PlaneSlotCache`.
    """
    var raw_dt = Int(datyp)
    var dt = (raw_dt - 64) if (raw_dt >= 64 and raw_dt < 128) else raw_dt
    var raw_p = MutPtr[UInt8](unsafe_from_address=Int(raw_ptr))
    var dst_p = MutPtr[Float32](unsafe_from_address=Int(dst_f32))
    var rpn_lng = max(1, Int(raw_byte_len) >> 3)
    var local_addr = 1

    unpack_one_f32(
        raw_p,
        Int(raw_byte_len),
        local_addr,
        rpn_lng,
        Int(ni),
        Int(nj),
        max(1, Int(nk)),
        dt,
        Int(nbits),
        dst_p,
    )
    return Int32(0)
