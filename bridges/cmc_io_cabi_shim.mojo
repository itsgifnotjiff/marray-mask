"""Separate-`.so` C-ABI `ChunkDecoder` Bridge Template for `boreal-io` (`cmc-io`).

Per `CONTEXT.md` P0 #2 and `docs/questions.md` Q1:
  Compile this thin C-ABI shim against `$MARRAY_CMC_IO/src/mojo` (`cmc_io_fst`) so
  `marray-mask` worker threads (`max.algorithm.parallelize`) call production RPN
  `datyp` unpackers (`0, 1, 2, 4, 5, 130, 133, 134`) directly across `.so` boundaries
  with ZERO Python GIL acquisition:

Build on work machine:
  mojo build -O3 --emit shared-lib \
    -I "${MARRAY_CMC_IO}/src/mojo" \
    bridges/cmc_io_cabi_shim.mojo \
    -o build/libcmc_marray_bridge.so

`python/marray_xr/fst_open.py` auto-discovers `build/libcmc_marray_bridge.so`
(or `$MARRAY_CMC_DECODER_SO`) and switches `CatalogSourceDesc` from single-thread
addr-callback mode (`Tier B`) to 16-thread zero-GIL POSIX `pread` + C-ABI mode (`Tier A`).
"""

from std.memory import ImmPointer, MutPointer, Span

# Adjust submodule path if `cmc_io_fst` reorganizes internal unpack_batch exports:
from cmc_io_fst.unpack_batch import (
    unpack_0_from_raw,
    unpack_1_from_raw,
    unpack_2_from_raw,
    unpack_4_from_raw,
    unpack_5_from_raw,
    unpack_130_from_raw,
    unpack_133_from_raw,
    unpack_134_from_raw,
)


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
    """Stable `ChunkDecoderCABI` entrypoint (`0` = success)."""
    var raw_dt = Int(datyp)
    var dt = (raw_dt - 64) if (raw_dt >= 64 and raw_dt < 128) else raw_dt
    var nelm = Int(ni) * Int(nj) * Int(nk)
    var raw_span = Span[UInt8, ImmUntrackedOrigin](
        ptr=raw_ptr, length=Int(raw_byte_len)
    )
    var dst_span = Span[Float32, MutUntrackedOrigin](
        ptr=dst_f32, length=nelm
    )

    match dt:
        case 134:
            unpack_134_from_raw(raw_span, Int(ni), Int(nj), Int(nk), Int(nbits), dst_span)
        case 2:
            unpack_2_from_raw(raw_span, Int(ni), Int(nj), Int(nk), Int(nbits), dst_span)
        case 4:
            unpack_4_from_raw(raw_span, Int(ni), Int(nj), Int(nk), Int(nbits), dst_span)
        case 1:
            unpack_1_from_raw(raw_span, Int(ni), Int(nj), Int(nk), Int(nbits), dst_span)
        case 5:
            unpack_5_from_raw(raw_span, Int(ni), Int(nj), Int(nk), Int(nbits), dst_span)
        case 130:
            unpack_130_from_raw(raw_span, Int(ni), Int(nj), Int(nk), Int(nbits), dst_span)
        case 133:
            unpack_133_from_raw(raw_span, Int(ni), Int(nj), Int(nk), Int(nbits), dst_span)
        case 0:
            unpack_0_from_raw(raw_span, Int(ni), Int(nj), Int(nk), Int(nbits), dst_span)
        case _:
            return Int32(-1)
    return Int32(0)
