"""Zero-copy & masked interop between Mojo `MaskedSlab` and Python `numpy` / `numpy.ma`.

Supports:
  1. `zero_copy_ptr_f32(py_arr)`: Direct `Pointer[Float32, MutAnyOrigin]` into an
     `F_CONTIGUOUS` NumPy `float32` array (`arr.ctypes.data`) for zero-copy reads
     and in-place SIMD mutation from Mojo.
  2. `from_numpy_slab(py_obj)`: Imports a Fortran-ordered `numpy.ndarray` or
     `numpy.ma.MaskedArray` into a `MaskedSlab`, packing any 1-byte NumPy boolean
     mask into our 8x-smaller `UInt64` hardware bitmask.
  3. `to_numpy_masked(slab)`: Exports a `MaskedSlab` back to a Fortran-ordered
     `numpy.ma.MaskedArray` ready to wrap directly in `xarray.DataArray`.
"""

from std.memory import Pointer
from std.os import abort
from std.python import Python, PythonObject

from .slab import (
    MASK_ALL_VALID,
    MASK_ALL_MASKED,
    MASK_BITMASK,
    SIMD_W,
    MaskedSlab,
    _bool8_to_byte,
    _num_mask_words,
)


def zero_copy_ptr_f32(py_arr: PythonObject) raises -> Pointer[Float32, MutAnyOrigin]:
    """Return a zero-copy mutable Mojo pointer to a Fortran-contiguous float32 ndarray.

    Caller must keep `py_arr` alive for the duration of pointer use.
    """
    if not Bool(py_arr.flags["F_CONTIGUOUS"]):
        abort(String("zero_copy_ptr_f32 requires an F_CONTIGUOUS numpy.ndarray"))
    var addr = Int(py=py_arr.ctypes.data)
    return Pointer[Float32, MutAnyOrigin](unsafe_from_address=addr)


def mutate_numpy_inplace_affine(
    py_arr: PythonObject, scale: Float32, offset: Float32
) raises:
    """In-place zero-copy SIMD transformation (`arr = arr * scale + offset`) on a
    Python Fortran `float32` NumPy array without allocating or copying buffers.
    """
    var ptr = zero_copy_ptr_f32(py_arr)
    var n = Int(py=py_arr.size)
    var sv = SIMD[DType.float32, SIMD_W](scale)
    var ov = SIMD[DType.float32, SIMD_W](offset)
    var vec_end = (n // SIMD_W) * SIMD_W
    var idx = 0
    while idx < vec_end:
        ptr.unsafe_store(idx, ptr.unsafe_load[width=SIMD_W](idx) * sv + ov)
        idx += SIMD_W
    while idx < n:
        ptr.unsafe_store(idx, ptr.unsafe_load(idx) * scale + offset)
        idx += 1


def from_numpy_slab(py_obj: PythonObject) raises -> MaskedSlab:
    """Import a 2D `numpy.ndarray` or `numpy.ma.MaskedArray` (`float32`, Fortran order)
    into a `MaskedSlab`, normalizing `numpy.ma` boolean byte masks into `UInt64` bitmasks.
    """
    var np = Python.import_module("numpy")
    var is_ma = Bool(np.ma.isMaskedArray(py_obj))
    var data_arr = np.ma.getdata(py_obj) if is_ma else py_obj
    var f_data = np.asfortranarray(data_arr, dtype=np.float32)

    var shape = f_data.shape
    if len(shape) != 2:
        abort(String("from_numpy_slab expects a 2D array (ni, nj)"))
    var ni = Int(py=shape[0])
    var nj = Int(py=shape[1])
    var total = ni * nj

    var out = MaskedSlab(ni, nj)
    var src_p = Pointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(py=f_data.ctypes.data)
    )
    var dst_p = out.data.unsafe_ptr()
    var vec_end = (total // SIMD_W) * SIMD_W
    var idx = 0
    while idx < vec_end:
        dst_p.unsafe_store(idx, src_p.unsafe_load[width=SIMD_W](idx))
        idx += SIMD_W
    while idx < total:
        dst_p.unsafe_store(idx, src_p.unsafe_load(idx))
        idx += 1

    if is_ma and Bool(py_obj.mask is not np.ma.nomask):
        var f_mask = np.asfortranarray(np.ma.getmaskarray(py_obj), dtype=np.uint8)
        var mp = Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(py=f_mask.ctypes.data)
        )
        out.ensure_bitmask_zeroed()
        var n_words = _num_mask_words(total)
        var zero_u8 = SIMD[DType.uint8, SIMD_W](0)
        for w in range(n_words):
            var base = w * 64
            var word_bits: UInt64 = 0
            for g in range(8):
                var off = base + g * 8
                if off + SIMD_W <= total:
                    var mv = mp.unsafe_load[width=SIMD_W](off).ne(zero_u8)
                    word_bits |= _bool8_to_byte(mv) << UInt64(g * 8)
                elif off < total:
                    for lane in range(total - off):
                        if mp.unsafe_load(off + lane) != UInt8(0):
                            word_bits |= UInt64(1) << UInt64(g * 8 + lane)
            out.mask_words[w] = word_bits
        out.finalize_mask_state()

    return out^


def to_numpy_masked(slab: MaskedSlab) raises -> PythonObject:
    """Export `MaskedSlab` to a Fortran-contiguous `numpy.ma.MaskedArray`."""
    var np = Python.import_module("numpy")
    var shape_tup = Python.tuple(slab.ni, slab.nj)
    var out_data = np.empty(shape_tup, dtype=np.float32, order="F")
    var total = slab.size()

    var src_p = slab.data.unsafe_ptr()
    var dst_p = Pointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(py=out_data.ctypes.data)
    )
    var vec_end = (total // SIMD_W) * SIMD_W
    var idx = 0
    while idx < vec_end:
        dst_p.unsafe_store(idx, src_p.unsafe_load[width=SIMD_W](idx))
        idx += SIMD_W
    while idx < total:
        dst_p.unsafe_store(idx, src_p.unsafe_load(idx))
        idx += 1

    if slab.mask_state == MASK_ALL_VALID:
        return np.ma.MaskedArray(out_data, mask=False)
    if slab.mask_state == MASK_ALL_MASKED:
        return np.ma.MaskedArray(out_data, mask=True)

    var out_mask = np.zeros(shape_tup, dtype=np.uint8, order="F")
    var mp = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=Int(py=out_mask.ctypes.data)
    )
    for k in range(total):
        if slab.is_masked_flat(k):
            mp.unsafe_store(k, UInt8(1))
    return np.ma.MaskedArray(out_data, mask=out_mask.astype(np.bool_))
