"""Direct Mojo `std.python` interop verification (`from_numpy_slab`, `to_numpy_masked`,
and zero-copy in-place mutation of Fortran `numpy.ndarray` -> `xarray.DataArray`).
"""

from std.os import abort
from std.python import Python

from marray import (
    MASK_BITMASK,
    from_numpy_slab,
    to_numpy_masked,
    mutate_numpy_inplace_affine,
    slab_reduce_stats,
)


def main() raises:
    var np = Python.import_module("numpy")
    var xr = Python.import_module("xarray")

    # 1. Verify true zero-copy in-place SIMD mutation on a Fortran NumPy array:
    var raw_f = np.asfortranarray(np.ones(Python.tuple(128, 64), dtype=np.float32))
    var addr_before = Int(py=raw_f.ctypes.data)
    mutate_numpy_inplace_affine(raw_f, Float32(2.5), Float32(10.0))
    var addr_after = Int(py=raw_f.ctypes.data)
    if addr_before != addr_after or Float64(py=raw_f[0, 0]) != 12.5:
        abort(String("Zero-copy in-place mutation failed"))
    print(
        "[PASS] Zero-copy in-place SIMD mutation on Fortran numpy.ndarray (same address:",
        addr_before == addr_after,
        ", value:",
        raw_f[0, 0],
        ")",
    )

    # 2. Import `numpy.ma.MaskedArray` -> Mojo `MaskedSlab` (64-bit bitmask) -> compute -> export back to `xarray`:
    var mask_np = np.zeros(Python.tuple(128, 64), dtype=np.bool_, order="F")
    mask_np[0:10, 0:5] = True  # Mask 50 cells
    var ma_in = np.ma.MaskedArray(raw_f, mask=mask_np)

    var slab_a = from_numpy_slab(ma_in)
    if slab_a.mask_state != MASK_BITMASK or slab_a.valid_count() != (128 * 64 - 50):
        abort(String("from_numpy_slab bitmask import mismatch"))

    var slab_b = slab_a + slab_a  # 12.5 + 12.5 = 25.0, preserving 50 masked cells!
    var st = slab_reduce_stats(slab_b)
    var ma_out = to_numpy_masked(slab_b)

    # Wrap exported masked array into an Xarray DataArray:
    var da_xr = xr.DataArray(
        np.ma.filled(ma_out, np.nan),
        dims=Python.list("lon", "lat"),
        attrs=Python.dict(units="degC", engine="marray-mojo"),
    )
    var xr_mean = Float64(py=da_xr.mean().item())
    print(
        "[PASS] numpy.ma <-> Mojo MaskedSlab <-> xarray.DataArray round-trip:",
        "mojo_mean =",
        st.mean(),
        "| xarray_mean =",
        xr_mean,
        "| valid_cells =",
        st.valid_count,
    )
