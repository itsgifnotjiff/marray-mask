"""Catalog-representative in-memory FST-like grids (U/O/Z/L/G/N/M) for benches.

Geometry + missingness encodings only. True datyp=134 unpack is out of scope.
"""

from std.collections import List
from std.math import sqrt, sin, cos, asin

from .slab import (
    VAR_TT,
    VAR_UU,
    VAR_VV,
    VAR_HU,
    VAR_TM,
    MaskedSlab,
    RawLayerScratch,
    SlabStats,
    decode_quant16_sentinel_into,
    decode_float32_bitmap_into,
    decode_uint16_qflag_into,
    slab_reduce_stats,
)


comptime GRTYP_U: Int = 0
comptime GRTYP_O: Int = 1
comptime GRTYP_Z: Int = 2
comptime GRTYP_L: Int = 3
comptime GRTYP_G: Int = 4
comptime GRTYP_N: Int = 5
comptime GRTYP_M: Int = 6

comptime _PI: Float32 = 3.14159265
comptime _DEG2RAD: Float32 = 0.0174532925


@always_inline
def _grtyp_name(g: Int) -> String:
    if g == GRTYP_U:
        return String("U")
    if g == GRTYP_O:
        return String("O")
    if g == GRTYP_Z:
        return String("Z")
    if g == GRTYP_L:
        return String("L")
    if g == GRTYP_G:
        return String("G")
    if g == GRTYP_N:
        return String("N")
    if g == GRTYP_M:
        return String("M")
    return String("?")


@always_inline
def _num_mask_words(n: Int) -> Int:
    return (n + 63) // 64


@always_inline
def _pseudo_terrain_hpa(lon_deg: Float32, lat_deg: Float32) -> Float32:
    var p_sfc = Float32(1013.25)
    if (
        lon_deg >= Float32(235.0)
        and lon_deg <= Float32(255.0)
        and lat_deg >= Float32(35.0)
        and lat_deg <= Float32(62.0)
    ):
        p_sfc = Float32(780.0)
    elif (
        lon_deg >= Float32(285.0)
        and lon_deg <= Float32(298.0)
        and lat_deg >= Float32(47.0)
        and lat_deg <= Float32(56.0)
    ):
        p_sfc = Float32(935.0)
    elif (
        lon_deg >= Float32(75.0)
        and lon_deg <= Float32(100.0)
        and lat_deg >= Float32(27.0)
        and lat_deg <= Float32(38.0)
    ):
        p_sfc = Float32(580.0)
    return p_sfc


@always_inline
def _is_land_mask(lon_deg: Float32, lat_deg: Float32, coastal_bias: Bool = False) -> Bool:
    if coastal_bias:
        # Heavier NW-Atlantic / Gulf St. Lawrence–style coastal land for O grids.
        if (
            lon_deg >= Float32(280.0)
            and lon_deg <= Float32(310.0)
            and lat_deg >= Float32(40.0)
            and lat_deg <= Float32(60.0)
        ):
            var h = ((Int(lon_deg * Float32(10.0)) * 17) ^ (Int(lat_deg * Float32(10.0)) * 31)) & 3
            return h != 0
        if (
            lon_deg >= Float32(235.0)
            and lon_deg <= Float32(296.0)
            and lat_deg >= Float32(25.0)
            and lat_deg <= Float32(72.0)
        ):
            return True
        return False
    if (
        lon_deg >= Float32(235.0)
        and lon_deg <= Float32(296.0)
        and lat_deg >= Float32(25.0)
        and lat_deg <= Float32(72.0)
    ):
        return True
    if (
        lon_deg >= Float32(10.0)
        and lon_deg <= Float32(135.0)
        and lat_deg >= Float32(20.0)
        and lat_deg <= Float32(70.0)
    ):
        return True
    return False


def _default_levels() -> List[Float64]:
    var lvls = List[Float64]()
    var p_list = [
        1000.0,
        950.0,
        925.0,
        850.0,
        700.0,
        600.0,
        500.0,
        400.0,
        300.0,
        250.0,
    ]
    for k in range(len(p_list)):
        lvls.append(p_list[k])
    return lvls^


def _tiny_levels() -> List[Float64]:
    var lvls = List[Float64]()
    lvls.append(1000.0)
    lvls.append(850.0)
    lvls.append(500.0)
    return lvls^


struct SynthGridSpec(Copyable, Movable):
    """Parametric horizontal + vertical + time extents for one synthetic grtyp."""

    var grtyp: Int
    var ni: Int
    var nj: Int
    var yin_ny: Int
    var n_steps: Int
    var step_dt_hr: Float64
    var levels_hpa: List[Float64]

    def __init__(
        out self,
        grtyp: Int,
        ni: Int,
        nj: Int,
        n_steps: Int,
        step_dt_hr: Float64,
        levels_hpa: List[Float64],
        yin_ny: Int = 0,
    ):
        self.grtyp = grtyp
        self.ni = ni
        self.nj = nj
        self.yin_ny = yin_ny
        self.n_steps = n_steps
        self.step_dt_hr = step_dt_hr
        self.levels_hpa = levels_hpa.copy()

    def n_levels(self) -> Int:
        return len(self.levels_hpa)

    def n_cells(self) -> Int:
        return self.ni * self.nj

    def total_slabs(self) -> Int:
        return self.n_steps * self.n_levels()

    def name(self) -> String:
        return _grtyp_name(self.grtyp)

    def scaled(self, factor: Int) -> SynthGridSpec:
        var f = factor if factor > 0 else 1
        var yin = 0
        if self.grtyp == GRTYP_U:
            yin = (self.yin_ny * f) if self.yin_ny > 0 else (self.nj * f) // 2
        return SynthGridSpec(
            self.grtyp,
            self.ni * f,
            self.nj * f,
            self.n_steps,
            self.step_dt_hr,
            self.levels_hpa.copy(),
            yin_ny=yin,
        )

    @staticmethod
    def tiny(grtyp: Int) -> SynthGridSpec:
        var lv = _tiny_levels()
        if grtyp == GRTYP_U:
            return SynthGridSpec(GRTYP_U, 32, 34, 2, 3.0, lv, yin_ny=17)
        if grtyp == GRTYP_M:
            return SynthGridSpec(GRTYP_M, 64, 1, 2, 3.0, lv)
        if grtyp == GRTYP_N:
            return SynthGridSpec(GRTYP_N, 24, 24, 2, 3.0, lv)
        if grtyp == GRTYP_O:
            return SynthGridSpec(GRTYP_O, 40, 28, 2, 3.0, lv)
        return SynthGridSpec(grtyp, 32, 17, 2, 3.0, lv)

    @staticmethod
    def bench(grtyp: Int) -> SynthGridSpec:
        var lv = _default_levels()
        if grtyp == GRTYP_U:
            return SynthGridSpec(GRTYP_U, 180, 202, 8, 3.0, lv, yin_ny=101)
        if grtyp == GRTYP_M:
            return SynthGridSpec(GRTYP_M, 20000, 1, 8, 3.0, lv)
        if grtyp == GRTYP_N:
            return SynthGridSpec(GRTYP_N, 160, 160, 8, 3.0, lv)
        if grtyp == GRTYP_O:
            return SynthGridSpec(GRTYP_O, 220, 160, 8, 3.0, lv)
        if grtyp == GRTYP_G:
            return SynthGridSpec(GRTYP_G, 192, 96, 8, 3.0, lv)
        return SynthGridSpec(grtyp, 200, 101, 8, 3.0, lv)

    @staticmethod
    def large(grtyp: Int) -> SynthGridSpec:
        return SynthGridSpec.bench(grtyp).scaled(3)


struct SynthAxes(Copyable, Movable):
    """1-D axes + Fortran-flat 2-D lon/lat (`flat = i + j * ni`)."""

    var ni: Int
    var nj: Int
    var yin_ny: Int
    var x_1d: List[Float64]
    var y_1d: List[Float64]
    var lon_f: List[Float32]
    var lat_f: List[Float32]

    def __init__(out self):
        self.ni = 0
        self.nj = 0
        self.yin_ny = 0
        self.x_1d = List[Float64]()
        self.y_1d = List[Float64]()
        self.lon_f = List[Float32]()
        self.lat_f = List[Float32]()

    @always_inline
    def lon_at(self, i: Int, j: Int) -> Float32:
        return self.lon_f[i + j * self.ni]

    @always_inline
    def lat_at(self, i: Int, j: Int) -> Float32:
        return self.lat_f[i + j * self.ni]

    @always_inline
    def outside_stereo_disk(self, i: Int, j: Int) -> Bool:
        var x = Float32(self.x_1d[i])
        var y = Float32(self.y_1d[j])
        return (x * x + y * y) > Float32(1.0)


def _resize_lonlat(mut axes: SynthAxes, ni: Int, nj: Int):
    axes.ni = ni
    axes.nj = nj
    var n = ni * nj
    if len(axes.lon_f) < n:
        axes.lon_f.resize(n, Float32(0.0))
    if len(axes.lat_f) < n:
        axes.lat_f.resize(n, Float32(0.0))
    if len(axes.x_1d) < ni:
        axes.x_1d.resize(ni, Float64(0.0))
    if len(axes.y_1d) < nj:
        axes.y_1d.resize(nj, Float64(0.0))


def _fill_regular_lonlat(
    mut axes: SynthAxes,
    lon0: Float32,
    lon1: Float32,
    lat0: Float32,
    lat1: Float32,
):
    var ni = axes.ni
    var nj = axes.nj
    var dlon = (lon1 - lon0) / Float32(ni if ni > 0 else 1)
    var dlat = (lat1 - lat0) / Float32(nj - 1 if nj > 1 else 1)
    for i in range(ni):
        axes.x_1d[i] = Float64(lon0 + Float32(i) * dlon)
    for j in range(nj):
        axes.y_1d[j] = Float64(lat0 + Float32(j) * dlat)
    for j in range(nj):
        var lat = lat0 + Float32(j) * dlat
        var col = j * ni
        for i in range(ni):
            axes.lon_f[col + i] = lon0 + Float32(i) * dlon
            axes.lat_f[col + i] = lat


def _fill_gaussian_lonlat(mut axes: SynthAxes):
    """Approximate Gaussian latitudes via asin-spaced roots (bench-quality, not Legendre)."""
    var ni = axes.ni
    var nj = axes.nj
    var dlon = Float32(360.0) / Float32(ni if ni > 0 else 1)
    for i in range(ni):
        axes.x_1d[i] = Float64(Float32(i) * dlon)
    for j in range(nj):
        # Map j in [0,nj) -> mu in (-1,1) then lat = asin(mu)
        var mu = Float32(-1.0) + (Float32(2.0) * Float32(j) + Float32(1.0)) / Float32(
            nj if nj > 0 else 1
        )
        if mu < Float32(-0.999):
            mu = Float32(-0.999)
        if mu > Float32(0.999):
            mu = Float32(0.999)
        var lat = asin(mu) / _DEG2RAD
        axes.y_1d[j] = Float64(lat)
        var col = j * ni
        for i in range(ni):
            axes.lon_f[col + i] = Float32(i) * dlon
            axes.lat_f[col + i] = lat


def _fill_rotated_z(mut axes: SynthAxes):
    """Irregular tic-tac axes + synthetic rotated-pole geographic lon/lat."""
    var ni = axes.ni
    var nj = axes.nj
    for i in range(ni):
        # Mild stretch toward domain edges
        var t = Float64(i) / Float64(ni - 1 if ni > 1 else 1)
        axes.x_1d[i] = -Float64(180.0) + Float64(360.0) * (t * t * (Float64(3.0) - Float64(2.0) * t))
    for j in range(nj):
        var t = Float64(j) / Float64(nj - 1 if nj > 1 else 1)
        axes.y_1d[j] = -Float64(90.0) + Float64(180.0) * (Float64(0.15) + Float64(0.7) * t)
    var pole_lat = Float32(56.0)
    var pole_lon = Float32(250.0)
    for j in range(nj):
        var rlat = Float32(axes.y_1d[j])
        var col = j * ni
        for i in range(ni):
            var rlon = Float32(axes.x_1d[i])
            # Lightweight rotated-E style warp (not bit-identical georef)
            var lon = pole_lon + rlon * Float32(0.85) + Float32(8.0) * sin(
                rlat * _DEG2RAD
            )
            var lat = pole_lat * Float32(0.35) + rlat * Float32(0.75)
            if lon < Float32(0.0):
                lon += Float32(360.0)
            if lon >= Float32(360.0):
                lon -= Float32(360.0)
            if lat < Float32(-90.0):
                lat = Float32(-90.0)
            if lat > Float32(90.0):
                lat = Float32(90.0)
            axes.lon_f[col + i] = lon
            axes.lat_f[col + i] = lat


def _fill_yin_yang(mut axes: SynthAxes, yin_ny: Int):
    axes.yin_ny = yin_ny
    var ni = axes.ni
    var nj = axes.nj
    for i in range(ni):
        axes.x_1d[i] = Float64(i)
    for j in range(nj):
        axes.y_1d[j] = Float64(j)
    var dlon = Float32(360.0) / Float32(ni if ni > 0 else 1)
    var dlat_yin = Float32(180.0) / Float32(yin_ny - 1 if yin_ny > 1 else 1)
    for j in range(nj):
        var panel = 0 if j < yin_ny else 1
        var j_loc = j if panel == 0 else j - yin_ny
        var lat: Float32
        var lon_shift: Float32
        if panel == 0:
            lat = Float32(-90.0) + Float32(j_loc) * dlat_yin
            lon_shift = Float32(0.0)
        else:
            lat = Float32(-90.0) + Float32(j_loc) * dlat_yin
            lon_shift = Float32(180.0)
        var col = j * ni
        for i in range(ni):
            var lon = Float32(i) * dlon + lon_shift
            if lon >= Float32(360.0):
                lon -= Float32(360.0)
            axes.lon_f[col + i] = lon
            axes.lat_f[col + i] = lat


def _fill_polar_stereo(mut axes: SynthAxes):
    var ni = axes.ni
    var nj = axes.nj
    for i in range(ni):
        axes.x_1d[i] = -Float64(1.0) + Float64(2.0) * Float64(i) / Float64(
            ni - 1 if ni > 1 else 1
        )
    for j in range(nj):
        axes.y_1d[j] = -Float64(1.0) + Float64(2.0) * Float64(j) / Float64(
            nj - 1 if nj > 1 else 1
        )
    for j in range(nj):
        var y = Float32(axes.y_1d[j])
        var col = j * ni
        for i in range(ni):
            var x = Float32(axes.x_1d[i])
            var r2 = x * x + y * y
            var lat = Float32(90.0) - Float32(90.0) * r2
            if lat < Float32(0.0):
                lat = Float32(0.0)
            var lon = Float32(0.0)
            if r2 > Float32(1.0e-12):
                lon = (asin(y / sqrt(r2 + Float32(1.0e-12))) / _DEG2RAD) + Float32(180.0)
            axes.lon_f[col + i] = lon
            axes.lat_f[col + i] = lat


def _fill_mesh_nodes(mut axes: SynthAxes):
    """nj=1 node list with irregular lon/lat packing."""
    var ni = axes.ni
    axes.y_1d[0] = Float64(0.0)
    for i in range(ni):
        axes.x_1d[i] = Float64(i)
        var t = Float32(i) / Float32(ni if ni > 0 else 1)
        var lon = Float32(220.0) + Float32(90.0) * t + Float32(3.0) * sin(
            Float32(i) * Float32(0.17)
        )
        var lat = Float32(35.0) + Float32(30.0) * t + Float32(2.0) * cos(
            Float32(i) * Float32(0.11)
        )
        axes.lon_f[i] = lon
        axes.lat_f[i] = lat


def build_axes(spec: SynthGridSpec) -> SynthAxes:
    var axes = SynthAxes()
    _resize_lonlat(axes, spec.ni, spec.nj)
    axes.yin_ny = spec.yin_ny
    if spec.grtyp == GRTYP_L:
        _fill_regular_lonlat(axes, Float32(0.0), Float32(360.0), Float32(-90.0), Float32(90.0))
    elif spec.grtyp == GRTYP_G:
        _fill_gaussian_lonlat(axes)
    elif spec.grtyp == GRTYP_Z:
        _fill_rotated_z(axes)
    elif spec.grtyp == GRTYP_U:
        var yin = spec.yin_ny if spec.yin_ny > 0 else spec.nj // 2
        _fill_yin_yang(axes, yin)
    elif spec.grtyp == GRTYP_N:
        _fill_polar_stereo(axes)
    elif spec.grtyp == GRTYP_O:
        _fill_regular_lonlat(
            axes, Float32(270.0), Float32(320.0), Float32(35.0), Float32(65.0)
        )
    elif spec.grtyp == GRTYP_M:
        _fill_mesh_nodes(axes)
    else:
        _fill_regular_lonlat(axes, Float32(0.0), Float32(360.0), Float32(-90.0), Float32(90.0))
    return axes^


@always_inline
def _yy_overlap_masked(spec: SynthGridSpec, j: Int) -> Bool:
    if spec.grtyp != GRTYP_U or spec.yin_ny <= 0:
        return False
    # Thin seam near panel boundary
    var seam = 2
    if j >= spec.yin_ny - seam and j < spec.yin_ny + seam:
        return True
    return False


def fill_and_decode_slab(
    spec: SynthGridSpec,
    axes: SynthAxes,
    var_id: Int,
    step_hr: Float32,
    level_hpa: Float32,
    i0: Int,
    i1: Int,
    j0: Int,
    j1: Int,
    mut scratch: RawLayerScratch,
    mut dst: MaskedSlab,
):
    """Synthesize encoded sub-slab using axes lon/lat, then decode into `dst`."""
    var sub_ni = i1 - i0
    var sub_nj = j1 - j0
    var total = sub_ni * sub_nj
    scratch.ensure_capacity(total)
    var coastal = spec.grtyp == GRTYP_O
    var phase = step_hr * Float32(0.035)

    if var_id == VAR_TT or var_id == VAR_UU or var_id == VAR_VV:
        var scale = Float32(0.01)
        var offset = Float32(-120.0)
        var u16_p = scratch.u16_buf.unsafe_ptr()
        for jj in range(sub_nj):
            var j = j0 + jj
            var col_off = jj * sub_ni
            for ii in range(sub_ni):
                var i = i0 + ii
                var lon = axes.lon_at(i, j)
                var lat = axes.lat_at(i, j)
                var force_mask = False
                if spec.grtyp == GRTYP_N and axes.outside_stereo_disk(i, j):
                    force_mask = True
                if _yy_overlap_masked(spec, j):
                    force_mask = True
                if force_mask or level_hpa > _pseudo_terrain_hpa(lon, lat):
                    u16_p.unsafe_store(col_off + ii, UInt16(0xFFFF))
                else:
                    var lat_rad = lat * _DEG2RAD
                    var meridional = cos(lat_rad)
                    var jet = sin(lat_rad * Float32(2.0))
                    var wave = Float32((i * 7 + j * 13) & 63) * Float32(0.08)
                    var phys: Float32
                    if var_id == VAR_TT:
                        var lapse = (level_hpa - Float32(500.0)) * Float32(0.075)
                        phys = (
                            Float32(-18.0)
                            + Float32(32.0) * meridional
                            + lapse
                            + wave
                            - phase
                        )
                    elif var_id == VAR_UU:
                        var alt_boost = (Float32(1015.0) - level_hpa) * Float32(0.055)
                        phys = Float32(12.0) + Float32(38.0) * jet + alt_boost + wave
                    else:
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
        var n_words = _num_mask_words(total)
        var f32_p = scratch.f32_buf.unsafe_ptr()
        var bm_p = scratch.bitmap_buf.unsafe_ptr()
        for w in range(n_words):
            bm_p.unsafe_store(w, UInt64(0))
        for jj in range(sub_nj):
            var j = j0 + jj
            var col_off = jj * sub_ni
            for ii in range(sub_ni):
                var i = i0 + ii
                var lon = axes.lon_at(i, j)
                var lat = axes.lat_at(i, j)
                var flat = col_off + ii
                var abs_lat = -lat if lat < Float32(0.0) else lat
                var base_sst = Float32(28.5) - abs_lat * Float32(0.36)
                if base_sst < Float32(-1.8):
                    base_sst = Float32(-1.8)
                f32_p.unsafe_store(
                    flat, base_sst + Float32((i & 15)) * Float32(0.06)
                )
                var land = _is_land_mask(lon, lat, coastal)
                if spec.grtyp == GRTYP_N and axes.outside_stereo_disk(i, j):
                    land = True
                if _yy_overlap_masked(spec, j):
                    land = True
                if land:
                    var w = flat >> 6
                    var b = flat & 63
                    bm_p.unsafe_store(
                        w, bm_p.unsafe_load(w) | (UInt64(1) << UInt64(b))
                    )
        decode_float32_bitmap_into(
            dst, scratch.f32_buf, scratch.bitmap_buf, sub_ni, sub_nj
        )

    else:
        # VAR_HU (+ default) -> quant16 + QC
        var scale = Float32(0.001)
        var offset = Float32(0.0)
        var u16_p = scratch.u16_buf.unsafe_ptr()
        var qf_p = scratch.u8_buf.unsafe_ptr()
        for jj in range(sub_nj):
            var j = j0 + jj
            var col_off = jj * sub_ni
            for ii in range(sub_ni):
                var i = i0 + ii
                var lon = axes.lon_at(i, j)
                var lat = axes.lat_at(i, j)
                var flat = col_off + ii
                var abs_lat = -lat if lat < Float32(0.0) else lat
                var base_hu = (Float32(18.0) - abs_lat * Float32(0.18)) * (
                    level_hpa / Float32(1000.0)
                )
                if base_hu < Float32(0.2):
                    base_hu = Float32(0.2)
                var hu_val = base_hu + Float32((i & 7)) * Float32(0.12)
                var q_int = Int((hu_val - offset) / scale + Float32(0.5))
                u16_p.unsafe_store(flat, UInt16(q_int))
                var hash_qc = (i * 31 + j * 17 + Int(step_hr)) & 63
                var qc = UInt8(2) if hash_qc == 0 else UInt8(0)
                if spec.grtyp == GRTYP_N and axes.outside_stereo_disk(i, j):
                    qc = UInt8(2)
                if _yy_overlap_masked(spec, j):
                    qc = UInt8(2)
                # Silence unused lon warning path for HU (lon used in land-ish QC salt)
                if coastal and _is_land_mask(lon, lat, True) and ((i + j) & 15) == 0:
                    qc = UInt8(2)
                qf_p.unsafe_store(flat, qc)
        decode_uint16_qflag_into(
            dst, scratch.u16_buf, scratch.u8_buf, sub_ni, sub_nj, scale, offset
        )


def materialize_var_cube(
    spec: SynthGridSpec,
    axes: SynthAxes,
    var_id: Int,
    mut scratch: RawLayerScratch,
    mut out_f32: List[Float32],
    mut out_mask_words: List[UInt64],
    mut out_states: List[Int],
) -> Int:
    """Decode all steps×levels into packed arenas. Returns total cells written per slab * n_slabs.

    Layout: slab `s` occupies `out_f32[s * n_pad ..]` with matching mask words / state.
    """
    var ni = spec.ni
    var nj = spec.nj
    var n_cells = ni * nj
    var n_pad = n_cells if (n_cells % 8) == 0 else n_cells + (8 - (n_cells % 8))
    var n_words = _num_mask_words(n_cells)
    var n_slabs = spec.total_slabs()
    var need_f = n_slabs * n_pad
    var need_w = n_slabs * n_words
    if len(out_f32) < need_f:
        out_f32.resize(need_f, Float32(0.0))
    if len(out_mask_words) < need_w:
        out_mask_words.resize(need_w, UInt64(0))
    if len(out_states) < n_slabs:
        out_states.resize(n_slabs, 0)

    var dst = MaskedSlab()
    var s = 0
    for t in range(spec.n_steps):
        var step_hr = Float32(Float64(t) * spec.step_dt_hr)
        for z in range(spec.n_levels()):
            var level = Float32(spec.levels_hpa[z])
            fill_and_decode_slab(
                spec, axes, var_id, step_hr, level, 0, ni, 0, nj, scratch, dst
            )
            var base_f = s * n_pad
            var base_w = s * n_words
            var dp = dst.data.unsafe_ptr()
            var op = out_f32.unsafe_ptr()
            for k in range(n_pad):
                op.unsafe_store(base_f + k, dp.unsafe_load(k))
            out_states[s] = dst.mask_state
            if len(dst.mask_words) > 0:
                var mp = dst.mask_words.unsafe_ptr()
                var ow = out_mask_words.unsafe_ptr()
                for w in range(n_words):
                    ow.unsafe_store(base_w + w, mp.unsafe_load(w))
            s += 1
    return n_slabs * n_cells


def slab_mask_fraction(dst: MaskedSlab) -> Float64:
    var st = slab_reduce_stats(dst)
    return st.masked_fraction()
