"""Layer 4: Lightweight Xarray-Style Semantic Glue (`CoordAxis`, `DataArray`, `Dataset`).

Provides the two features actually needed for NWP & scientific workflows:
  1. Labeled coordinate axes (`time`, `level`, `lon`, `lat`) supporting O(log N)
     monotonic ascending & descending binary-search selection (`.sel()`, `.isel()`)
     pushed directly down into the parametric lazy chunk window.
  2. Typed metadata containers (`attrs: Dict[String, AttrValue]`) on both
     variables (`DataArray`) and datasets (`Dataset`), plus seamless arithmetic
     operator overloading (`+`, `-`, `*`, `/`, `hypot`, `mask_lt`) that stages
     lazy DAGs without touching data until a terminal reduction/materialize call.
"""

from std.collections import List, Dict
from std.utils import Variant

from .slab import (
    VAR_TT,
    VAR_UU,
    VAR_VV,
    VAR_HU,
    VAR_TM,
    MaskedSlab,
)
from .lazy import (
    OP_SCAN_VAR,
    OP_CONST,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_HYPOT,
    OP_SQRT,
    OP_AFFINE,
    OP_MASK_LT,
    GDPSGridSpec,
    DomainWindow,
    ExprNode,
    LazyGraph,
    ExecutionReport,
    execute_fused_reduce,
    execute_spatial_mean_map,
    execute_single_slab,
)


comptime AttrValue = Variant[String, Int, Float64]


struct IndexRange(Copyable, Movable):
    var start: Int
    var stop: Int

    def __init__(out self, start: Int, stop: Int):
        self.start = start
        self.stop = stop


struct CoordAxis(Copyable, Movable):
    """1D monotonic coordinate vector supporting ascending (`lon`, `lat`, `time`)
    and descending (`level`: 1015 -> 50 hPa) O(log N) binary search.
    """

    var name: String
    var units: String
    var values: List[Float64]
    var ascending: Bool

    def __init__(out self, name: String, units: String, values: List[Float64]):
        self.name = name
        self.units = units
        self.values = values.copy()
        if len(values) >= 2:
            self.ascending = values[len(values) - 1] >= values[0]
        else:
            self.ascending = True

    def size(self) -> Int:
        return len(self.values)

    def sel_range(self, bound_a: Float64, bound_b: Float64) -> IndexRange:
        """Select inclusive coordinate interval `[min(a,b), max(a,b)]` in O(log N)."""
        var n = len(self.values)
        if n == 0:
            return IndexRange(0, 0)
        var lo_val = bound_a if bound_a <= bound_b else bound_b
        var hi_val = bound_b if bound_b >= bound_a else bound_a

        if self.ascending:
            # Lower bound: first idx with values[idx] >= lo_val
            var l = 0
            var r = n
            while l < r:
                var mid = (l + r) // 2
                if self.values[mid] < lo_val:
                    l = mid + 1
                else:
                    r = mid
            var idx_start = l

            # Upper bound: first idx with values[idx] > hi_val
            l = 0
            r = n
            while l < r:
                var mid = (l + r) // 2
                if self.values[mid] <= hi_val:
                    l = mid + 1
                else:
                    r = mid
            var idx_stop = l
            return IndexRange(idx_start, idx_stop)
        else:
            # Descending axis (e.g. pressure hPa: 1015.0 down to 50.0):
            # First idx where values[idx] <= hi_val
            var l = 0
            var r = n
            while l < r:
                var mid = (l + r) // 2
                if self.values[mid] > hi_val:
                    l = mid + 1
                else:
                    r = mid
            var idx_start = l

            # First idx where values[idx] < lo_val
            l = 0
            r = n
            while l < r:
                var mid = (l + r) // 2
                if self.values[mid] >= lo_val:
                    l = mid + 1
                else:
                    r = mid
            var idx_stop = l
            return IndexRange(idx_start, idx_stop)

    def sel_nearest(self, target: Float64) -> IndexRange:
        """Select single nearest coordinate index `[best, best + 1)`."""
        var n = len(self.values)
        if n == 0:
            return IndexRange(0, 0)
        var best_i = 0
        var best_diff = self.values[0] - target
        if best_diff < 0.0:
            best_diff = -best_diff
        for i in range(1, n):
            var diff = self.values[i] - target
            if diff < 0.0:
                diff = -diff
            if diff < best_diff:
                best_diff = diff
                best_i = i
        return IndexRange(best_i, best_i + 1)


def _attr_to_str(val: AttrValue) -> String:
    if val.isa[String]():
        return String('"') + val[String] + String('"')
    elif val.isa[Int]():
        return String(val[Int])
    else:
        return String(val[Float64])


struct DataArray(Copyable, Movable):
    """Xarray-like labeled lazy masked array backed by `LazyGraph`."""

    var name: String
    var dims: List[String]
    var attrs: Dict[String, AttrValue]
    var grid: GDPSGridSpec
    var window: DomainWindow
    var graph: LazyGraph
    var root_id: Int

    def __init__(
        out self,
        name: String,
        dims: List[String],
        attrs: Dict[String, AttrValue],
        grid: GDPSGridSpec,
        window: DomainWindow,
        graph: LazyGraph,
        root_id: Int,
    ):
        self.name = name
        self.dims = dims.copy()
        self.attrs = attrs.copy()
        self.grid = grid.copy()
        self.window = window.copy()
        self.graph = graph.copy()
        self.root_id = root_id

    @staticmethod
    def from_gdps_var(
        name: String,
        var_id: Int,
        units: String,
        long_name: String,
        encoding_desc: String,
        is_surface_2d: Bool,
        grid: GDPSGridSpec,
    ) -> DataArray:
        var dims = List[String]()
        dims.append(String("run"))
        dims.append(String("time"))
        dims.append(String("level"))
        dims.append(String("lon"))
        dims.append(String("lat"))

        var attrs = Dict[String, AttrValue]()
        attrs["units"] = AttrValue(units)
        attrs["long_name"] = AttrValue(long_name)
        attrs["encoding"] = AttrValue(encoding_desc)
        attrs["storage_order"] = AttrValue(String("Fortran (ni, nj)"))

        var g = LazyGraph()
        var rid = g.add_node(
            ExprNode(OP_SCAN_VAR, var_id=var_id, is_surface_2d=is_surface_2d)
        )
        return DataArray(
            name, dims, attrs, grid, grid.full_window(), g, rid
        )

    def rename(self, new_name: String, units: String = "") -> DataArray:
        var out = self.copy()
        out.name = new_name
        if units.byte_length() > 0:
            out.attrs["units"] = AttrValue(units)
        return out^

    def _binary_op(self, other: DataArray, op_code: Int, op_sym: String) -> DataArray:
        var merged = self.graph.copy()
        var rhs_id = merged.merge_subgraph(other.graph, other.root_id)
        var new_root = merged.add_node(ExprNode(op_code, lhs=self.root_id, rhs=rhs_id))
        var new_attrs = Dict[String, AttrValue]()
        new_attrs["expression"] = AttrValue(
            String("(") + self.name + String(" ") + op_sym + String(" ") + other.name + String(")")
        )
        new_attrs["storage_order"] = AttrValue(String("Fortran (ni, nj)"))
        return DataArray(
            self.name + "_" + other.name,
            self.dims,
            new_attrs,
            self.grid,
            self.window,
            merged,
            new_root,
        )

    def __add__(self, other: Self) -> Self:
        return self._binary_op(other, OP_ADD, "+")

    def __sub__(self, other: Self) -> Self:
        return self._binary_op(other, OP_SUB, "-")

    def __mul__(self, other: Self) -> Self:
        return self._binary_op(other, OP_MUL, "*")

    def __truediv__(self, other: Self) -> Self:
        return self._binary_op(other, OP_DIV, "/")

    def scale_offset(
        self, scale: Float32, offset: Float32 = 0.0, new_units: String = ""
    ) -> DataArray:
        var g = self.graph.copy()
        var rid = g.add_node(
            ExprNode(OP_AFFINE, lhs=self.root_id, p0=scale, p1=offset)
        )
        var out = DataArray(
            self.name, self.dims, self.attrs, self.grid, self.window, g, rid
        )
        if new_units.byte_length() > 0:
            out.attrs["units"] = AttrValue(new_units)
        return out^

    def sqrt(self) -> DataArray:
        var g = self.graph.copy()
        var rid = g.add_node(ExprNode(OP_SQRT, lhs=self.root_id))
        return DataArray(
            "sqrt_" + self.name,
            self.dims,
            self.attrs,
            self.grid,
            self.window,
            g,
            rid,
        )

    def mask_lt(self, threshold: Float32) -> DataArray:
        """Declarative masking (`da.where(da >= threshold)` in Xarray)."""
        var g = self.graph.copy()
        var rid = g.add_node(
            ExprNode(OP_MASK_LT, lhs=self.root_id, p0=threshold)
        )
        return DataArray(
            self.name, self.dims, self.attrs, self.grid, self.window, g, rid
        )

    def with_window(self, win: DomainWindow) -> DataArray:
        var out = self.copy()
        out.window = win.copy()
        return out^

    def sel(
        self,
        time_min: Float64 = -1.0e30,
        time_max: Float64 = 1.0e30,
        level_min: Float64 = -1.0e30,
        level_max: Float64 = 1.0e30,
        lon_min: Float64 = -1.0e30,
        lon_max: Float64 = 1.0e30,
        lat_min: Float64 = -1.0e30,
        lat_max: Float64 = 1.0e30,
        max_runs: Int = -1,
    ) -> DataArray:
        var t_ax = CoordAxis("time", "hours", self.grid.time_hours)
        var z_ax = CoordAxis("level", "hPa", self.grid.levels_hpa)
        var x_ax = CoordAxis("lon", "degrees_east", self.grid.lons_deg)
        var y_ax = CoordAxis("lat", "degrees_north", self.grid.lats_deg)

        var tr = t_ax.sel_range(time_min, time_max)
        var zr = z_ax.sel_range(level_min, level_max)
        var xr = x_ax.sel_range(lon_min, lon_max)
        var yr = y_ax.sel_range(lat_min, lat_max)

        var r1 = self.window.r1
        if max_runs > 0 and self.window.r0 + max_runs < r1:
            r1 = self.window.r0 + max_runs

        var nw = DomainWindow(
            self.window.r0,
            r1,
            tr.start if tr.start > self.window.t0 else self.window.t0,
            tr.stop if tr.stop < self.window.t1 else self.window.t1,
            zr.start if zr.start > self.window.z0 else self.window.z0,
            zr.stop if zr.stop < self.window.z1 else self.window.z1,
            xr.start if xr.start > self.window.i0 else self.window.i0,
            xr.stop if xr.stop < self.window.i1 else self.window.i1,
            yr.start if yr.start > self.window.j0 else self.window.j0,
            yr.stop if yr.stop < self.window.j1 else self.window.j1,
        )
        return self.with_window(nw)

    def sel_level(self, target_hpa: Float64) -> DataArray:
        var z_ax = CoordAxis("level", "hPa", self.grid.levels_hpa)
        var zr = z_ax.sel_nearest(target_hpa)
        var nw = self.window.copy()
        nw.z0 = zr.start
        nw.z1 = zr.stop
        return self.with_window(nw)

    def compute_stats(self) -> ExecutionReport:
        return execute_fused_reduce(
            self.grid, self.window, self.graph, self.root_id
        )

    def compute_spatial_mean(self) -> MaskedSlab:
        return execute_spatial_mean_map(
            self.grid, self.window, self.graph, self.root_id
        )

    def compute_first_slab(self) -> MaskedSlab:
        return execute_single_slab(
            self.grid, self.window, self.graph, self.root_id
        )


def hypot(u: DataArray, v: DataArray, name: String = "wspd") -> DataArray:
    """Fused lazy horizontal wind speed `sqrt(u^2 + v^2)`."""
    var merged = u.graph.copy()
    var rhs_id = merged.merge_subgraph(v.graph, v.root_id)
    var rid = merged.add_node(ExprNode(OP_HYPOT, lhs=u.root_id, rhs=rhs_id))
    var attrs = u.attrs.copy()
    attrs["long_name"] = AttrValue(String("Horizontal wind speed"))
    return DataArray(name, u.dims, attrs, u.grid, u.window, merged, rid)


struct Dataset(Copyable, Movable):
    """Xarray-like multi-variable NWP Dataset sharing coordinates, window, and metadata."""

    var grid: GDPSGridSpec
    var window: DomainWindow
    var attrs: Dict[String, AttrValue]
    var var_names: List[String]
    var vars: Dict[String, DataArray]

    def __init__(out self, grid: GDPSGridSpec):
        self.grid = grid.copy()
        self.window = grid.full_window()
        self.attrs = Dict[String, AttrValue]()
        self.var_names = List[String]()
        self.vars = Dict[String, DataArray]()

    @staticmethod
    def open_virtual_gdps(grid: GDPSGridSpec) -> Dataset:
        """Construct an ECCC GDPS multi-variable Dataset in <1 ms with zero disk reads."""
        var ds = Dataset(grid)
        ds.attrs["title"] = AttrValue(
            String("ECCC Global Deterministic Prediction System (GDPS)")
        )
        ds.attrs["institution"] = AttrValue(
            String("Environment and Climate Change Canada (MSC / CMC)")
        )
        ds.attrs["model"] = AttrValue(String("GEM-NEMO-CICE Global 15km"))
        ds.attrs["layer_layout"] = AttrValue(
            String("Fortran column-major (ni, nj) contiguous slabs")
        )

        ds.assign(
            "TT",
            DataArray.from_gdps_var(
                "TT",
                VAR_TT,
                "degC",
                "Air temperature",
                "ENC_QUANT16_SENTINEL (0xFFFF terrain mask)",
                False,
                grid,
            ),
        )
        ds.assign(
            "UU",
            DataArray.from_gdps_var(
                "UU",
                VAR_UU,
                "knots",
                "U-component of wind",
                "ENC_QUANT16_SENTINEL (0xFFFF terrain mask)",
                False,
                grid,
            ),
        )
        ds.assign(
            "VV",
            DataArray.from_gdps_var(
                "VV",
                VAR_VV,
                "knots",
                "V-component of wind",
                "ENC_QUANT16_SENTINEL (0xFFFF terrain mask)",
                False,
                grid,
            ),
        )
        ds.assign(
            "HU",
            DataArray.from_gdps_var(
                "HU",
                VAR_HU,
                "g/kg",
                "Specific humidity",
                "ENC_UINT16_QFLAG (UInt8 QC flag > 1 mask)",
                False,
                grid,
            ),
        )
        ds.assign(
            "TM",
            DataArray.from_gdps_var(
                "TM",
                VAR_TM,
                "degC",
                "Sea surface temperature",
                "ENC_FLOAT32_BITMAP (1-bit land/sea bitmap @@)",
                True,
                grid,
            ),
        )
        return ds^

    def assign(mut self, name: String, da: DataArray):
        var exists = False
        for i in range(len(self.var_names)):
            if self.var_names[i] == name:
                exists = True
                break
        if not exists:
            self.var_names.append(name)
        var aligned = da.with_window(self.window)
        aligned.name = name
        self.vars[name] = aligned^

    def __getitem__(self, name: String) raises -> DataArray:
        return self.vars[name].with_window(self.window)

    def sel(
        self,
        time_min: Float64 = -1.0e30,
        time_max: Float64 = 1.0e30,
        level_min: Float64 = -1.0e30,
        level_max: Float64 = 1.0e30,
        lon_min: Float64 = -1.0e30,
        lon_max: Float64 = 1.0e30,
        lat_min: Float64 = -1.0e30,
        lat_max: Float64 = 1.0e30,
        max_runs: Int = -1,
    ) raises -> Dataset:
        """Push coordinate range selection down across every variable in the Dataset."""
        var out = self.copy()
        var probe = DataArray.from_gdps_var(
            "_probe", VAR_TT, "", "", "", False, self.grid
        ).with_window(self.window)
        var sliced_probe = probe.sel(
            time_min=time_min,
            time_max=time_max,
            level_min=level_min,
            level_max=level_max,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            max_runs=max_runs,
        )
        out.window = sliced_probe.window.copy()
        for i in range(len(out.var_names)):
            var vn = out.var_names[i]
            out.vars[vn] = out.vars[vn].with_window(out.window)
        return out^

    def logical_f32_bytes(self) -> Int:
        return len(self.var_names) * self.window.active_cells() * 4

    def summary(self) raises -> String:
        var w = self.window.copy()
        var total_bytes = Float64(self.logical_f32_bytes())
        var size_str: String
        if total_bytes >= 1.0e15:
            size_str = String(total_bytes / 1.0e15) + String(" PB")
        elif total_bytes >= 1.0e12:
            size_str = String(total_bytes / 1.0e12) + String(" TB")
        elif total_bytes >= 1.0e9:
            size_str = String(total_bytes / 1.0e9) + String(" GB")
        else:
            size_str = String(total_bytes / 1.0e6) + String(" MB")

        var s = String("<marray.Dataset>  [Virtual Footprint: ") + size_str + String("]\n")
        s += (
            String("Dimensions:  (run: ")
            + String(w.n_runs())
            + String(", time: ")
            + String(w.n_steps())
            + String(", level: ")
            + String(w.n_levels())
            + String(", lon: ")
            + String(w.sub_ni())
            + String(", lat: ")
            + String(w.sub_nj())
            + String(")\n")
        )
        if w.n_steps() > 0 and w.n_levels() > 0 and w.sub_ni() > 0 and w.sub_nj() > 0:
            s += (
                String("Coordinates:\n")
                + String("  * time     (time)   float64  ")
                + String(self.grid.time_hours[w.t0])
                + String("h .. ")
                + String(self.grid.time_hours[w.t1 - 1])
                + String("h\n")
                + String("  * level    (level)  float64  ")
                + String(self.grid.levels_hpa[w.z0])
                + String(" .. ")
                + String(self.grid.levels_hpa[w.z1 - 1])
                + String(" hPa (descending)\n")
                + String("  * lon      (lon)    float64  ")
                + String(self.grid.lons_deg[w.i0])
                + String(" .. ")
                + String(self.grid.lons_deg[w.i1 - 1])
                + String(" deg_E\n")
                + String("  * lat      (lat)    float64  ")
                + String(self.grid.lats_deg[w.j0])
                + String(" .. ")
                + String(self.grid.lats_deg[w.j1 - 1])
                + String(" deg_N\n")
            )
        s += (
            String("Active Slabs per 3D Var: ")
            + String(w.active_slabs())
            + String(" (of ")
            + String(self.grid.total_slabs_per_3d_var())
            + String(" catalog slabs)\n")
        )
        s += String("Data variables:\n")
        for i in range(len(self.var_names)):
            var vn = self.var_names[i]
            var da = self.vars[vn].copy()
            var units_s = String("")
            if "units" in da.attrs:
                units_s = da.attrs["units"][String]
            var enc_s = String("derived DAG (") + String(len(da.graph.nodes)) + String(" ops)")
            if "encoding" in da.attrs:
                enc_s = da.attrs["encoding"][String]
            s += (
                String("    ")
                + vn
                + String("  (run, time, level, lon, lat) float32 [")
                + units_s
                + String("]  ")
                + enc_s
                + String("\n")
            )
        s += String("Attributes:\n")
        if "title" in self.attrs:
            s += String("    title:        ") + _attr_to_str(self.attrs["title"]) + String("\n")
        if "institution" in self.attrs:
            s += String("    institution:  ") + _attr_to_str(self.attrs["institution"]) + String("\n")
        if "layer_layout" in self.attrs:
            s += String("    layer_layout: ") + _attr_to_str(self.attrs["layer_layout"]) + String("\n")
        return s
