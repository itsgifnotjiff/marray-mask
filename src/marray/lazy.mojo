"""Layers 2 & 3: Parametric GDPS Grid Catalog, Closed-IR Lazy Expression DAG,
and Bounded-Memory Multi-Threaded Fused Pull Executor.

Key architectural properties:
  1. O(1)-memory parametric chunk grid: opening a 1+ Petabyte multi-cycle GDPS
     archive (~100M Fortran layers) allocates only 1D coordinate metadata (~30 KB),
     never an explicit Python/Dask dictionary of chunk tasks.
  2. Closed-IR expression arena (`List[ExprNode]`): O(ops) nodes total.
  3. Bounded-memory pull execution: each CPU worker thread pre-allocates ONE
     `WorkerArena` (reusable raw decode scratch + per-node `MaskedSlab` registers)
     and streams assigned `(run, time, level)` slabs in L2/L3 cache with zero
     heap allocations inside the chunk loop.
"""

from std.collections import List
from std.memory import Pointer
from std.time import perf_counter_ns
from std.runtime import parallelism_level
from max.algorithm import parallelize

from .slab import (
    MASK_ALL_VALID,
    MASK_ALL_MASKED,
    MASK_BITMASK,
    SlabStats,
    MaskedSlab,
    RawLayerScratch,
    slab_add_into,
    slab_sub_into,
    slab_mul_into,
    slab_div_into,
    slab_hypot_into,
    slab_sqrt_into,
    slab_affine_into,
    slab_mask_lt_into,
    slab_reduce_stats,
    synthesize_and_decode_gdps_subslab,
    _num_mask_words,
    _padded_f32_len,
)


# Expression IR opcodes:
comptime OP_SCAN_VAR: Int = 1
comptime OP_CONST: Int = 2
comptime OP_ADD: Int = 3
comptime OP_SUB: Int = 4
comptime OP_MUL: Int = 5
comptime OP_DIV: Int = 6
comptime OP_HYPOT: Int = 7
comptime OP_SQRT: Int = 8
comptime OP_AFFINE: Int = 9
comptime OP_MASK_LT: Int = 10


struct GDPSGridSpec(Copyable, Movable):
    """Parametric ECCC GDPS grid geometry (`run x time x level x lon(ni) x lat(nj)`)."""

    var n_runs: Int
    var time_hours: List[Float64]
    var levels_hpa: List[Float64]
    var lons_deg: List[Float64]
    var lats_deg: List[Float64]

    def __init__(
        out self,
        n_runs: Int,
        n_steps: Int,
        step_dt_hr: Float64,
        levels_hpa: List[Float64],
        ni: Int,
        nj: Int,
    ):
        self.n_runs = n_runs
        self.time_hours = List[Float64](capacity=n_steps)
        for s in range(n_steps):
            self.time_hours.append(Float64(s) * step_dt_hr)
        self.levels_hpa = levels_hpa.copy()

        self.lons_deg = List[Float64](capacity=ni)
        var dlon = Float64(360.0) / Float64(ni)
        for i in range(ni):
            self.lons_deg.append(Float64(i) * dlon)

        self.lats_deg = List[Float64](capacity=nj)
        var dlat = Float64(180.0) / Float64(nj - 1 if nj > 1 else 1)
        for j in range(nj):
            self.lats_deg.append(Float64(-90.0) + Float64(j) * dlat)

    @staticmethod
    def standard_gdps_15km(n_runs: Int = 1) -> GDPSGridSpec:
        """Official ECCC GDPS 0.15 deg global lat-lon grid (2400 x 1201, 81 steps, 20 isobaric levels)."""
        var lvls = List[Float64]()
        var p_list = [
            1015.0, 1000.0, 975.0, 950.0, 925.0,
            900.0, 875.0, 850.0, 800.0, 750.0,
            700.0, 650.0, 600.0, 550.0, 500.0,
            400.0, 300.0, 250.0, 150.0, 50.0,
        ]
        for k in range(len(p_list)):
            lvls.append(p_list[k])
        return GDPSGridSpec(n_runs, 81, 3.0, lvls, 2400, 1201)

    @staticmethod
    def benchmark_gdps_grid(
        n_runs: Int = 1, n_steps: Int = 24, ni: Int = 600, nj: Int = 301
    ) -> GDPSGridSpec:
        var lvls = List[Float64]()
        var p_list = [
            1000.0, 950.0, 925.0, 850.0, 700.0,
            600.0, 500.0, 400.0, 300.0, 250.0,
        ]
        for k in range(len(p_list)):
            lvls.append(p_list[k])
        return GDPSGridSpec(n_runs, n_steps, 3.0, lvls, ni, nj)

    def n_steps(self) -> Int:
        return len(self.time_hours)

    def n_levels(self) -> Int:
        return len(self.levels_hpa)

    def ni(self) -> Int:
        return len(self.lons_deg)

    def nj(self) -> Int:
        return len(self.lats_deg)

    def total_slabs_per_3d_var(self) -> Int:
        return self.n_runs * self.n_steps() * self.n_levels()

    def full_window(self) -> DomainWindow:
        return DomainWindow(
            0,
            self.n_runs,
            0,
            self.n_steps(),
            0,
            self.n_levels(),
            0,
            self.ni(),
            0,
            self.nj(),
        )


struct DomainWindow(Copyable, Movable):
    """Half-open index bounds pushed down to the parametric chunk reader."""

    var r0: Int
    var r1: Int
    var t0: Int
    var t1: Int
    var z0: Int
    var z1: Int
    var i0: Int
    var i1: Int
    var j0: Int
    var j1: Int

    def __init__(
        out self,
        r0: Int,
        r1: Int,
        t0: Int,
        t1: Int,
        z0: Int,
        z1: Int,
        i0: Int,
        i1: Int,
        j0: Int,
        j1: Int,
    ):
        self.r0 = r0
        self.r1 = r1
        self.t0 = t0
        self.t1 = t1
        self.z0 = z0
        self.z1 = z1
        self.i0 = i0
        self.i1 = i1
        self.j0 = j0
        self.j1 = j1

    def n_runs(self) -> Int:
        return self.r1 - self.r0 if self.r1 > self.r0 else 0

    def n_steps(self) -> Int:
        return self.t1 - self.t0 if self.t1 > self.t0 else 0

    def n_levels(self) -> Int:
        return self.z1 - self.z0 if self.z1 > self.z0 else 0

    def sub_ni(self) -> Int:
        return self.i1 - self.i0 if self.i1 > self.i0 else 0

    def sub_nj(self) -> Int:
        return self.j1 - self.j0 if self.j1 > self.j0 else 0

    def active_slabs(self) -> Int:
        return self.n_runs() * self.n_steps() * self.n_levels()

    def active_cells(self) -> Int:
        return self.active_slabs() * self.sub_ni() * self.sub_nj()


struct ExprNode(Copyable, Movable):
    var op: Int
    var lhs: Int
    var rhs: Int
    var var_id: Int
    var is_surface_2d: Bool
    var p0: Float32
    var p1: Float32

    def __init__(
        out self,
        op: Int,
        lhs: Int = -1,
        rhs: Int = -1,
        var_id: Int = -1,
        is_surface_2d: Bool = False,
        p0: Float32 = 0.0,
        p1: Float32 = 0.0,
    ):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs
        self.var_id = var_id
        self.is_surface_2d = is_surface_2d
        self.p0 = p0
        self.p1 = p1


struct LazyGraph(Copyable, Movable):
    """Flat arena of symbolic array operations (typically 1..30 nodes even at PB scale)."""

    var nodes: List[ExprNode]

    def __init__(out self):
        self.nodes = List[ExprNode]()

    def add_node(mut self, node: ExprNode) -> Int:
        var nid = len(self.nodes)
        self.nodes.append(node.copy())
        return nid

    def merge_subgraph(mut self, other: LazyGraph, other_root: Int) -> Int:
        """Clone reachable nodes from `other` into `self` and return remapped root ID."""
        var remap = List[Int](capacity=len(other.nodes))
        for idx in range(len(other.nodes)):
            var n = other.nodes[idx].copy()
            if n.lhs >= 0:
                n.lhs = remap[n.lhs]
            if n.rhs >= 0:
                n.rhs = remap[n.rhs]
            var new_id = self.add_node(n)
            remap.append(new_id)
        return remap[other_root]

    def count_scan_ops(self) -> Int:
        var c = 0
        for i in range(len(self.nodes)):
            if self.nodes[i].op == OP_SCAN_VAR:
                c += 1
        return c


struct WorkerArena(Copyable, Movable):
    """Pre-allocated per-thread execution workspace (zero per-chunk heap churn)."""

    var raw_scratch: RawLayerScratch
    var regs: List[MaskedSlab]
    var out_tmp: MaskedSlab

    def __init__(out self, num_regs: Int, sub_ni: Int, sub_nj: Int):
        self.raw_scratch = RawLayerScratch()
        self.raw_scratch.ensure_capacity(sub_ni * sub_nj)
        self.regs = List[MaskedSlab](capacity=num_regs)
        for _ in range(num_regs):
            self.regs.append(MaskedSlab(sub_ni, sub_nj))
        self.out_tmp = MaskedSlab(sub_ni, sub_nj)

    def approx_bytes(self) -> Int:
        var n_cells = self.out_tmp.size()
        var per_reg = _padded_f32_len(n_cells) * 4 + _num_mask_words(n_cells) * 8
        var raw_bytes = _padded_f32_len(n_cells) * (2 + 4 + 1) + _num_mask_words(
            n_cells
        ) * 8
        return (len(self.regs) + 1) * per_reg + raw_bytes


def eval_dag_zero_alloc(
    nodes_ptr: Pointer[ExprNode, _],
    root_id: Int,
    step_hr: Float32,
    level_hpa: Float32,
    full_ni: Int,
    full_nj: Int,
    i0: Int,
    i1: Int,
    j0: Int,
    j1: Int,
    mut arena: WorkerArena,
):
    """Zero-allocation DAG evaluation over pre-allocated `arena.regs`.

    Uses `arena.out_tmp` as a borrow-disjoint destination register and O(1)
    pointer `swap` into `arena.regs[nid]`, satisfying Mojo's strict origin
    aliasing rules with zero heap allocations or element copies.
    """
    var sub_ni = i1 - i0
    var sub_nj = j1 - j0

    for nid in range(root_id + 1):
        ref node = nodes_ptr[unsafe_offset=nid]
        if node.op == OP_SCAN_VAR:
            var eff_lvl = Float32(1013.25) if node.is_surface_2d else level_hpa
            synthesize_and_decode_gdps_subslab(
                node.var_id,
                step_hr,
                eff_lvl,
                full_ni,
                full_nj,
                i0,
                i1,
                j0,
                j1,
                arena.raw_scratch,
                arena.regs[nid],
            )
        elif node.op == OP_CONST:
            arena.regs[nid].ensure_shape(sub_ni, sub_nj)
            arena.regs[nid].mask_state = MASK_ALL_VALID
            var dp = arena.regs[nid].data.unsafe_ptr()
            var n_pad = _padded_f32_len(sub_ni * sub_nj)
            for k in range(n_pad):
                dp.unsafe_store(k, node.p0)
        elif node.op == OP_ADD:
            slab_add_into(arena.out_tmp, arena.regs[node.lhs], arena.regs[node.rhs])
            swap(arena.regs[nid], arena.out_tmp)
        elif node.op == OP_SUB:
            slab_sub_into(arena.out_tmp, arena.regs[node.lhs], arena.regs[node.rhs])
            swap(arena.regs[nid], arena.out_tmp)
        elif node.op == OP_MUL:
            slab_mul_into(arena.out_tmp, arena.regs[node.lhs], arena.regs[node.rhs])
            swap(arena.regs[nid], arena.out_tmp)
        elif node.op == OP_DIV:
            slab_div_into(arena.out_tmp, arena.regs[node.lhs], arena.regs[node.rhs])
            swap(arena.regs[nid], arena.out_tmp)
        elif node.op == OP_HYPOT:
            slab_hypot_into(arena.out_tmp, arena.regs[node.lhs], arena.regs[node.rhs])
            swap(arena.regs[nid], arena.out_tmp)
        elif node.op == OP_SQRT:
            slab_sqrt_into(arena.out_tmp, arena.regs[node.lhs])
            swap(arena.regs[nid], arena.out_tmp)
        elif node.op == OP_AFFINE:
            slab_affine_into(
                arena.out_tmp, arena.regs[node.lhs], node.p0, node.p1
            )
            swap(arena.regs[nid], arena.out_tmp)
        elif node.op == OP_MASK_LT:
            slab_mask_lt_into(arena.out_tmp, arena.regs[node.lhs], node.p0)
            swap(arena.regs[nid], arena.out_tmp)


struct ExecutionReport(Copyable, Movable):
    var stats: SlabStats
    var elapsed_ms: Float64
    var active_slabs: Int
    var total_catalog_slabs: Int
    var num_workers: Int
    var peak_arena_bytes: Int
    var logical_array_bytes_processed: Int

    def __init__(out self):
        self.stats = SlabStats()
        self.elapsed_ms = 0.0
        self.active_slabs = 0
        self.total_catalog_slabs = 0
        self.num_workers = 1
        self.peak_arena_bytes = 0
        self.logical_array_bytes_processed = 0

    def throughput_gb_s(self) -> Float64:
        if self.elapsed_ms <= 0.0:
            return 0.0
        var secs = self.elapsed_ms * 1.0e-3
        return (Float64(self.logical_array_bytes_processed) / 1.0e9) / secs

    def us_per_slab(self) -> Float64:
        if self.active_slabs == 0:
            return 0.0
        return (self.elapsed_ms * 1000.0) / Float64(self.active_slabs)


def execute_fused_reduce(
    grid: GDPSGridSpec,
    win: DomainWindow,
    graph: LazyGraph,
    root_id: Int,
) -> ExecutionReport:
    """Multi-threaded bounded-memory pull execution across all active `(run, time, level)` slabs."""
    var t_start = perf_counter_ns()
    var report = ExecutionReport()
    report.total_catalog_slabs = grid.total_slabs_per_3d_var()
    var n_slabs = win.active_slabs()
    report.active_slabs = n_slabs
    if n_slabs == 0 or win.sub_ni() == 0 or win.sub_nj() == 0:
        return report^

    var hw_threads = parallelism_level()
    if hw_threads < 1:
        hw_threads = 1
    var n_workers = hw_threads if n_slabs >= hw_threads else n_slabs
    report.num_workers = n_workers

    var sample_arena = WorkerArena(root_id + 1, win.sub_ni(), win.sub_nj())
    report.peak_arena_bytes = sample_arena.approx_bytes() * n_workers
    # Each DAG node touches 1 Float32 slab (4B/cell) logically:
    report.logical_array_bytes_processed = (
        n_slabs * (root_id + 1) * win.sub_ni() * win.sub_nj() * 4
    )

    var partial_sums = List[Float64](capacity=n_workers)
    partial_sums.resize(n_workers, Float64(0.0))
    var partial_mins = List[Float32](capacity=n_workers)
    partial_mins.resize(n_workers, Float32(1.0e30))
    var partial_maxs = List[Float32](capacity=n_workers)
    partial_maxs.resize(n_workers, Float32(-1.0e30))
    var partial_valid = List[Int](capacity=n_workers)
    partial_valid.resize(n_workers, 0)
    var partial_total = List[Int](capacity=n_workers)
    partial_total.resize(n_workers, 0)

    var p_sum = partial_sums.unsafe_ptr()
    var p_min = partial_mins.unsafe_ptr()
    var p_max = partial_maxs.unsafe_ptr()
    var p_val = partial_valid.unsafe_ptr()
    var p_tot = partial_total.unsafe_ptr()
    var nodes_ptr = graph.nodes.unsafe_ptr()
    var times_ptr = grid.time_hours.unsafe_ptr()
    var lvls_ptr = grid.levels_hpa.unsafe_ptr()

    var n_steps = win.n_steps()
    var n_lvls = win.n_levels()
    var t0 = win.t0
    var z0 = win.z0
    var i0 = win.i0
    var i1 = win.i1
    var j0 = win.j0
    var j1 = win.j1
    var full_ni = grid.ni()
    var full_nj = grid.nj()
    var n_regs = root_id + 1

    def worker(wid: Int) {imm}:
        var s_begin = (wid * n_slabs) // n_workers
        var s_end = ((wid + 1) * n_slabs) // n_workers
        if s_begin >= s_end:
            return

        # Allocate ONE bounded workspace for this CPU core:
        var arena = WorkerArena(n_regs, i1 - i0, j1 - j0)
        var local_stats = SlabStats()
        var regs_ptr = arena.regs.unsafe_ptr()

        for linear_idx in range(s_begin, s_end):
            var z_rel = linear_idx % n_lvls
            var rem = linear_idx // n_lvls
            var t_rel = rem % n_steps
            var r_rel = rem // n_steps

            var step_hr = Float32(
                times_ptr.unsafe_load(t0 + t_rel) + Float64(r_rel) * 0.25
            )
            var level_hpa = Float32(lvls_ptr.unsafe_load(z0 + z_rel))

            eval_dag_zero_alloc(
                nodes_ptr,
                root_id,
                step_hr,
                level_hpa,
                full_ni,
                full_nj,
                i0,
                i1,
                j0,
                j1,
                arena,
            )
            var s_stats = slab_reduce_stats(regs_ptr[unsafe_offset=root_id])
            local_stats.merge(s_stats)

        p_sum.unsafe_store(wid, local_stats.sum)
        p_min.unsafe_store(wid, local_stats.min_val)
        p_max.unsafe_store(wid, local_stats.max_val)
        p_val.unsafe_store(wid, local_stats.valid_count)
        p_tot.unsafe_store(wid, local_stats.total_count)

    parallelize(worker, n_workers)

    for w in range(n_workers):
        var ws = SlabStats()
        ws.sum = partial_sums[w]
        ws.min_val = partial_mins[w]
        ws.max_val = partial_maxs[w]
        ws.valid_count = partial_valid[w]
        ws.total_count = partial_total[w]
        report.stats.merge(ws)

    var t_end = perf_counter_ns()
    report.elapsed_ms = Float64(t_end - t_start) * 1.0e-6
    return report^


def execute_spatial_mean_map(
    grid: GDPSGridSpec,
    win: DomainWindow,
    graph: LazyGraph,
    root_id: Int,
) -> MaskedSlab:
    """Stream all `(run, time, level)` slabs in parallel and reduce along `(run, time, level)`
    into a 2D Fortran-ordered `(sub_ni, sub_nj)` `MaskedSlab`. Grid cells masked across
    all levels/times remain masked in the resulting 2D map!
    """
    var sub_ni = win.sub_ni()
    var sub_nj = win.sub_nj()
    var out = MaskedSlab(sub_ni, sub_nj)
    var n_slabs = win.active_slabs()
    var n_cells = sub_ni * sub_nj
    if n_slabs == 0 or n_cells == 0:
        out.mask_state = MASK_ALL_MASKED
        return out^

    var hw_threads = parallelism_level()
    if hw_threads < 1:
        hw_threads = 1
    var n_workers = hw_threads if n_slabs >= hw_threads else n_slabs

    var worker_sums = List[Float64](capacity=n_workers * n_cells)
    worker_sums.resize(n_workers * n_cells, Float64(0.0))
    var worker_cnts = List[Int32](capacity=n_workers * n_cells)
    worker_cnts.resize(n_workers * n_cells, Int32(0))

    var ws_ptr = worker_sums.unsafe_ptr()
    var wc_ptr = worker_cnts.unsafe_ptr()
    var nodes_ptr = graph.nodes.unsafe_ptr()
    var times_ptr = grid.time_hours.unsafe_ptr()
    var lvls_ptr = grid.levels_hpa.unsafe_ptr()

    var n_steps = win.n_steps()
    var n_lvls = win.n_levels()
    var t0 = win.t0
    var z0 = win.z0
    var i0 = win.i0
    var i1 = win.i1
    var j0 = win.j0
    var j1 = win.j1
    var full_ni = grid.ni()
    var full_nj = grid.nj()
    var n_regs = root_id + 1

    def worker(wid: Int) {imm}:
        var s_begin = (wid * n_slabs) // n_workers
        var s_end = ((wid + 1) * n_slabs) // n_workers
        if s_begin >= s_end:
            return

        var arena = WorkerArena(n_regs, sub_ni, sub_nj)
        var regs_ptr = arena.regs.unsafe_ptr()
        var base_off = wid * n_cells

        for linear_idx in range(s_begin, s_end):
            var z_rel = linear_idx % n_lvls
            var rem = linear_idx // n_lvls
            var t_rel = rem % n_steps
            var r_rel = rem // n_steps

            var step_hr = Float32(
                times_ptr.unsafe_load(t0 + t_rel) + Float64(r_rel) * 0.25
            )
            var level_hpa = Float32(lvls_ptr.unsafe_load(z0 + z_rel))

            eval_dag_zero_alloc(
                nodes_ptr,
                root_id,
                step_hr,
                level_hpa,
                full_ni,
                full_nj,
                i0,
                i1,
                j0,
                j1,
                arena,
            )
            ref root_slab = regs_ptr[unsafe_offset=root_id]
            if root_slab.mask_state == MASK_ALL_MASKED:
                continue
            var dp = root_slab.data.unsafe_ptr()
            if root_slab.mask_state == MASK_ALL_VALID:
                for k in range(n_cells):
                    ws_ptr.unsafe_store(
                        base_off + k,
                        ws_ptr.unsafe_load(base_off + k)
                        + Float64(dp.unsafe_load(k)),
                    )
                    wc_ptr.unsafe_store(
                        base_off + k, wc_ptr.unsafe_load(base_off + k) + Int32(1)
                    )
            else:
                for k in range(n_cells):
                    if not root_slab.is_masked_flat(k):
                        ws_ptr.unsafe_store(
                            base_off + k,
                            ws_ptr.unsafe_load(base_off + k)
                            + Float64(dp.unsafe_load(k)),
                        )
                        wc_ptr.unsafe_store(
                            base_off + k,
                            wc_ptr.unsafe_load(base_off + k) + Int32(1),
                        )

    parallelize(worker, n_workers)

    out.ensure_bitmask_zeroed()
    var out_dp = out.data.unsafe_ptr()
    for k in range(n_cells):
        var total_sum = Float64(0.0)
        var total_cnt = Int32(0)
        for w in range(n_workers):
            total_sum += worker_sums[w * n_cells + k]
            total_cnt += worker_cnts[w * n_cells + k]
        if total_cnt > Int32(0):
            out_dp.unsafe_store(k, Float32(total_sum / Float64(total_cnt)))
        else:
            out_dp.unsafe_store(k, Float32(0.0))
            out.mask_words[k >> 6] |= UInt64(1) << UInt64(k & 63)

    out.finalize_mask_state()
    return out^


def execute_single_slab(
    grid: GDPSGridSpec,
    win: DomainWindow,
    graph: LazyGraph,
    root_id: Int,
) -> MaskedSlab:
    """Materialize the first `(run, time, level)` Fortran sub-slab of `win`."""
    var sub_ni = win.sub_ni()
    var sub_nj = win.sub_nj()
    if win.active_slabs() == 0 or sub_ni == 0 or sub_nj == 0:
        var empty = MaskedSlab(sub_ni, sub_nj)
        empty.mask_state = MASK_ALL_MASKED
        return empty^

    var arena = WorkerArena(root_id + 1, sub_ni, sub_nj)
    var step_hr = Float32(
        grid.time_hours[win.t0] + Float64(win.r0) * 0.25
    )
    var level_hpa = Float32(grid.levels_hpa[win.z0])
    eval_dag_zero_alloc(
        graph.nodes.unsafe_ptr(),
        root_id,
        step_hr,
        level_hpa,
        grid.ni(),
        grid.nj(),
        win.i0,
        win.i1,
        win.j0,
        win.j1,
        arena,
    )
    return arena.regs[root_id].copy()


def execute_unfused_materializing_baseline(
    grid: GDPSGridSpec,
    win: DomainWindow,
    graph: LazyGraph,
    root_id: Int,
) -> ExecutionReport:
    """Baseline mimicking chunk-by-chunk orchestrators that allocate & materialize
    every intermediate chunk array per DAG node before reducing.
    """
    var t_start = perf_counter_ns()
    var report = ExecutionReport()
    report.total_catalog_slabs = grid.total_slabs_per_3d_var()
    var n_slabs = win.active_slabs()
    report.active_slabs = n_slabs
    var sub_ni = win.sub_ni()
    var sub_nj = win.sub_nj()
    report.logical_array_bytes_processed = (
        n_slabs * (root_id + 1) * sub_ni * sub_nj * 4
    )

    var hw_threads = parallelism_level()
    if hw_threads < 1:
        hw_threads = 1
    var n_workers = hw_threads if n_slabs >= hw_threads else n_slabs
    report.num_workers = n_workers

    # Unfused orchestrator materializes intermediate chunk arrays per op:
    var n_nodes = root_id + 1
    var per_slab_bytes = (
        _padded_f32_len(sub_ni * sub_nj) * 4 + _num_mask_words(sub_ni * sub_nj) * 8
    )
    report.peak_arena_bytes = n_slabs * n_nodes * per_slab_bytes

    var partial_sums = List[Float64](capacity=n_workers)
    partial_sums.resize(n_workers, Float64(0.0))
    var partial_valid = List[Int](capacity=n_workers)
    partial_valid.resize(n_workers, 0)
    var partial_total = List[Int](capacity=n_workers)
    partial_total.resize(n_workers, 0)

    var p_sum = partial_sums.unsafe_ptr()
    var p_val = partial_valid.unsafe_ptr()
    var p_tot = partial_total.unsafe_ptr()
    var nodes_ptr = graph.nodes.unsafe_ptr()
    var times_ptr = grid.time_hours.unsafe_ptr()
    var lvls_ptr = grid.levels_hpa.unsafe_ptr()

    var n_steps = win.n_steps()
    var n_lvls = win.n_levels()
    var t0 = win.t0
    var z0 = win.z0
    var i0 = win.i0
    var i1 = win.i1
    var j0 = win.j0
    var j1 = win.j1
    var full_ni = grid.ni()
    var full_nj = grid.nj()

    def worker(wid: Int) {imm}:
        var s_begin = (wid * n_slabs) // n_workers
        var s_end = ((wid + 1) * n_slabs) // n_workers
        var local_stats = SlabStats()

        for linear_idx in range(s_begin, s_end):
            var z_rel = linear_idx % n_lvls
            var rem = linear_idx // n_lvls
            var t_rel = rem % n_steps
            var r_rel = rem // n_steps
            var step_hr = Float32(
                times_ptr.unsafe_load(t0 + t_rel) + Float64(r_rel) * 0.25
            )
            var level_hpa = Float32(lvls_ptr.unsafe_load(z0 + z_rel))

            # Simulates unfused chunk-level DAG: fresh heap allocations per op per chunk!
            var fresh_arena = WorkerArena(n_nodes, sub_ni, sub_nj)
            eval_dag_zero_alloc(
                nodes_ptr,
                root_id,
                step_hr,
                level_hpa,
                full_ni,
                full_nj,
                i0,
                i1,
                j0,
                j1,
                fresh_arena,
            )
            var s_stats = slab_reduce_stats(fresh_arena.regs[root_id])
            local_stats.merge(s_stats)

        p_sum.unsafe_store(wid, local_stats.sum)
        p_val.unsafe_store(wid, local_stats.valid_count)
        p_tot.unsafe_store(wid, local_stats.total_count)

    parallelize(worker, n_workers)

    for w in range(n_workers):
        report.stats.sum += partial_sums[w]
        report.stats.valid_count += partial_valid[w]
        report.stats.total_count += partial_total[w]

    var t_end = perf_counter_ns()
    report.elapsed_ms = Float64(t_end - t_start) * 1.0e-6
    return report^
