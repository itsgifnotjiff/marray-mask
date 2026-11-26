"""Q6 Vertical Column-Tile Schedule (`(column_tile=64, nk)`) for `boreal-postproc`.

Per `docs/questions.md` Q6:
  `cmc_postproc_kernels` (`lift_columns`, `vertical_scan_columns` for `CAPE`, `LCL`,
  parcel lift, wet-bulb, hypsometric integration) operate on flattened columns
  (`(column, level)`, length `nk` contiguous per column) after Python peels and
  transposes the full 4D cube `(n_t, nk, nj, ni) -> (n_t, nj, ni, nk)`.

This module eliminates the full-cube Python peel + 4D transpose by streaming
`TILE_COLS = 64` horizontal columns across all `nk` levels into an L1d-resident
`(64, nk)` micro-tile (`64 * 40 * 4 B = 10 KB`!), applying `@@` bitmap validity,
invoking either an external C-ABI `cmc_postproc_kernels` column callback or a
built-in vertical scan, and storing directly to `(n_t, nj, ni)` (`NaN` or `1e30`).
"""

from std.math import ceildiv, exp, fma, log, max, min
from std.memory import (
    Allocation,
    ImmPointer,
    Layout,
    MutPointer,
    alloc,
    dealloc,
)
from std.runtime import parallelism_level
from std.sys import size_of
from max.algorithm import parallelize

from .morsel import CACHE_ALIGN


comptime TILE_COLS: Int = 64

comptime VCOL_HYPSOMETRIC_INTEGRAL: Int = 0
comptime VCOL_CAPE_PARCEL_PROXY: Int = 1
comptime VCOL_COLUMN_MAX: Int = 2

# External C-ABI column-tile kernel contract matching `cmc_postproc_kernels`:
#   (ctx_addr, tt_col_major_f32 [n_cols, nk], hu_col_major_f32 [n_cols, nk],
#    valid_u8 [n_cols, nk], levels_f32 [nk], n_cols, nk, out_f32 [n_cols]) -> Int32
comptime VerticalColumnKernelCABI = def(
    Int64,
    ImmPointer[Float32, ImmUntrackedOrigin],
    ImmPointer[Float32, ImmUntrackedOrigin],
    ImmPointer[UInt8, ImmUntrackedOrigin],
    ImmPointer[Float32, ImmUntrackedOrigin],
    Int32,
    Int32,
    MutPointer[Float32, MutUntrackedOrigin],
) thin abi("C") -> Int32


struct ColumnTileScratch(Movable):
    """L1d-resident `(TILE_COLS=64, nk)` contiguous `(column, level)` buffers."""

    var _tt_alloc: Allocation[Float32, alignment=CACHE_ALIGN]
    var _hu_alloc: Allocation[Float32, alignment=CACHE_ALIGN]
    var _vm_alloc: Allocation[UInt8, alignment=CACHE_ALIGN]
    var _out_alloc: Allocation[Float32, alignment=CACHE_ALIGN]

    def __init__(out self, nk: Int):
        var cap = max(64, TILE_COLS * max(1, nk))
        self._tt_alloc = alloc(Layout[Float32, alignment=CACHE_ALIGN](count=cap))
        self._hu_alloc = alloc(Layout[Float32, alignment=CACHE_ALIGN](count=cap))
        self._vm_alloc = alloc(Layout[UInt8, alignment=CACHE_ALIGN](count=cap))
        self._out_alloc = alloc(
            Layout[Float32, alignment=CACHE_ALIGN](count=TILE_COLS)
        )

    def __init__(out self, *, deinit move: Self):
        self._tt_alloc = move._tt_alloc^
        self._hu_alloc = move._hu_alloc^
        self._vm_alloc = move._vm_alloc^
        self._out_alloc = move._out_alloc^

    def __deinit__(deinit self):
        dealloc(self._tt_alloc^)
        dealloc(self._hu_alloc^)
        dealloc(self._vm_alloc^)
        dealloc(self._out_alloc^)


@inline(.always)
def _eval_builtin_column_kernel(
    mode: Int,
    tt_cols: ImmPointer[Float32, _],
    hu_cols: ImmPointer[Float32, _],
    has_hu: Bool,
    valid_u8: ImmPointer[UInt8, _],
    levels_f32: ImmPointer[Float32, _],
    n_cols: Int,
    nk: Int,
    out_fill: Float32,
    out_cols: MutPointer[Float32, _],
):
    """Run stateful vertical column scan over `n_cols` packed `(column, level)` slices."""
    for c in range(n_cols):
        var c_base = c * nk
        if mode == VCOL_COLUMN_MAX:
            var best = Float32(-1.0e30)
            var any_v = False
            for z in range(nk):
                if valid_u8[unsafe_offset=c_base + z] != UInt8(0):
                    best = max(best, tt_cols[unsafe_offset=c_base + z])
                    any_v = True
            out_cols[unsafe_offset=c] = best if any_v else out_fill
        elif mode == VCOL_HYPSOMETRIC_INTEGRAL:
            # Geopotential thickness `R_d / g0 * integral(T_v d(ln p))` across valid column layers:
            var dz_sum = Float32(0.0)
            var valid_layers = 0
            for z in range(nk - 1):
                var v0 = valid_u8[unsafe_offset=c_base + z] != UInt8(0)
                var v1 = valid_u8[unsafe_offset=c_base + z + 1] != UInt8(0)
                if v0 and v1:
                    var t0 = tt_cols[unsafe_offset=c_base + z] + Float32(273.15)
                    var t1 = tt_cols[unsafe_offset=c_base + z + 1] + Float32(273.15)
                    if has_hu:
                        var q0 = hu_cols[unsafe_offset=c_base + z] * Float32(0.001)
                        var q1 = hu_cols[unsafe_offset=c_base + z + 1] * Float32(0.001)
                        t0 = fma(t0, Float32(0.608) * q0, t0)
                        t1 = fma(t1, Float32(0.608) * q1, t1)
                    var p0 = max(Float32(1.0), levels_f32[unsafe_offset=z])
                    var p1 = max(Float32(1.0), levels_f32[unsafe_offset=z + 1])
                    var dlnp = log(max(p0, p1) / min(p0, p1))
                    dz_sum = fma(Float32(29.27126) * Float32(0.5) * (t0 + t1), dlnp, dz_sum)
                    valid_layers += 1
            out_cols[unsafe_offset=c] = dz_sum if valid_layers > 0 else out_fill
        else:
            # `VCOL_CAPE_PARCEL_PROXY` (`lift_columns` stateful upward parcel buoyancy scan):
            # Launches surface/lowest unmasked parcel `(T_sfc, q_sfc)` and integrates positive
            # virtual temperature buoyancy along moist-adiabatic lapse across `z = 0..nk-1`!
            var launched = False
            var theta_e_p = Float32(0.0)
            var p_launch = Float32(1000.0)
            var cape = Float32(0.0)
            for z in range(nk):
                if valid_u8[unsafe_offset=c_base + z] == UInt8(0):
                    continue
                var t_env_k = tt_cols[unsafe_offset=c_base + z] + Float32(273.15)
                var q_env = (
                    hu_cols[unsafe_offset=c_base + z] * Float32(0.001)
                    if has_hu
                    else Float32(0.008)
                )
                var p_z = max(Float32(10.0), levels_f32[unsafe_offset=z])
                if not launched:
                    launched = True
                    p_launch = p_z
                    # Equivalent potential temperature proxy at launch level:
                    theta_e_p = t_env_k + Float32(2500.0) * q_env
                else:
                    var p_prev = max(Float32(10.0), levels_f32[unsafe_offset=z - 1])
                    var dlnp = log(max(p_prev, p_z) / min(p_prev, p_z))
                    # Moist parcel cooling rate vs environment:
                    var ratio = p_z / p_launch
                    var t_parcel = theta_e_p * exp(Float32(0.2854) * log(max(Float32(0.05), ratio)))
                    var buoy = (t_parcel - t_env_k) / max(Float32(180.0), t_env_k)
                    if buoy > Float32(0.0):
                        cape = fma(Float32(287.05) * t_env_k * buoy, dlnp, cape)
            out_cols[unsafe_offset=c] = cape if launched else out_fill


def execute_vertical_column_scan(
    tt_f32: ImmPointer[Float32, _],
    hu_f32: ImmPointer[Float32, _],
    has_hu: Bool,
    bmap_ptr: ImmPointer[UInt8, _],
    bmap_kind: Int,  # 0 = none, 1 = uint8 (keep>0), 2 = float32 (keep>0)
    levels_f32: ImmPointer[Float32, _],
    n_t: Int,
    nk: Int,
    nj: Int,
    ni: Int,
    kernel_mode: Int,
    ext_kernel_fn_addr: Int,
    ext_kernel_ctx_addr: Int,
    out_fill_f32: Float32,
    out_2d_per_t: MutPointer[Float32, _],
):
    """Execute vertical column physics over `(n_t, nk, nj, ni)` using L1-resident
    `TILE_COLS = 64` `(column, level)` tiles without any full-cube transpose.
    """
    var n_pts = nj * ni
    if n_t <= 0 or nk <= 0 or n_pts <= 0:
        return
    var tiles_per_t = ceildiv(n_pts, TILE_COLS)
    var total_tiles = n_t * tiles_per_t
    var hw = max(1, parallelism_level())
    var n_workers = min(hw, total_tiles)

    def worker(wid: Int) {imm}:
        var t_begin = (wid * total_tiles) // n_workers
        var t_end = ((wid + 1) * total_tiles) // n_workers
        if t_begin >= t_end:
            return
        var scratch = ColumnTileScratch(nk)
        var tt_tile = scratch._tt_alloc.unsafe_ptr()
        var hu_tile = scratch._hu_alloc.unsafe_ptr()
        var vm_tile = scratch._vm_alloc.unsafe_ptr()
        var out_tile = scratch._out_alloc.unsafe_ptr()
        var bmap_f32 = bmap_ptr.unsafe_bitcast[Float32]()

        for tid in range(t_begin, t_end):
            var t_idx, col_blk = divmod(tid, tiles_per_t)
            var col_start = col_blk * TILE_COLS
            var n_cols = min(TILE_COLS, n_pts - col_start)
            var cube_t_base = t_idx * nk * n_pts

            # Gather `n_cols` across `z = 0..nk-1` and pack in-register into `(column, level)`!
            for z in range(nk):
                var plane_off = cube_t_base + z * n_pts + col_start
                for c in range(n_cols):
                    var dst_idx = c * nk + z
                    tt_tile[unsafe_offset=dst_idx] = tt_f32[unsafe_offset=plane_off + c]
                    if has_hu:
                        hu_tile[unsafe_offset=dst_idx] = hu_f32[unsafe_offset=plane_off + c]
                    if bmap_kind == 0:
                        vm_tile[unsafe_offset=dst_idx] = UInt8(1)
                    elif bmap_kind == 1:
                        vm_tile[unsafe_offset=dst_idx] = (
                            UInt8(1)
                            if bmap_ptr[unsafe_offset=plane_off + c] > UInt8(0)
                            else UInt8(0)
                        )
                    else:
                        vm_tile[unsafe_offset=dst_idx] = (
                            UInt8(1)
                            if bmap_f32[unsafe_offset=plane_off + c] > Float32(0.0)
                            else UInt8(0)
                        )

            if ext_kernel_fn_addr != 0:
                comptime assert size_of[VerticalColumnKernelCABI]() == size_of[Int](), "fn ptr width"
                var slot_a = ext_kernel_fn_addr
                var ext_fn = Pointer(to=slot_a).unsafe_bitcast[VerticalColumnKernelCABI]()[]
                _ = ext_fn(
                    Int64(ext_kernel_ctx_addr),
                    ImmPointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(tt_tile)),
                    ImmPointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(hu_tile)),
                    ImmPointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(vm_tile)),
                    ImmPointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(levels_f32)),
                    Int32(n_cols),
                    Int32(nk),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out_tile)),
                )
            else:
                _eval_builtin_column_kernel(
                    kernel_mode,
                    tt_tile,
                    hu_tile,
                    has_hu,
                    vm_tile,
                    levels_f32,
                    n_cols,
                    nk,
                    out_fill_f32,
                    out_tile,
                )

            var out_base = t_idx * n_pts + col_start
            for c in range(n_cols):
                out_2d_per_t[unsafe_offset=out_base + c] = out_tile[unsafe_offset=c]

    parallelize(worker, n_workers)
