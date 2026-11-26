"""End-to-End Pure-Mojo ECCC GDPS Prototype (`marray-mask`).

Demonstrates the full 4-layer stack in pure Mojo on in-memory synthetic ECCC
GDPS NWP data (no disk I/O):
  1. Petabyte-scale virtual catalog instantiation & O(log N) coordinate slice
     pushdown (`.sel()`) without Dask's O(chunks) Python task-graph explosion.
  2. Seamless propagation of 3 heterogeneous NWP binary mask encodings:
       * `TT`, `UU`, `VV`: 16-bit quantized + `0xFFFF` underground terrain sentinels
       * `TM` (SST):       Fortran `Float32` + 1-bit GRIB2/RPN `@@` land/sea bitmap
       * `HU` (Humidity):  16-bit quantized + `UInt8` sensor QC flags
  3. Multi-threaded bounded-memory pull execution vs. unfused chunk-materializing
     baseline across thousands of Fortran-contiguous layers.
"""

from std.time import perf_counter_ns
from marray import (
    MASK_ALL_VALID,
    MASK_BITMASK,
    GDPSGridSpec,
    Dataset,
    hypot,
    slab_reduce_stats,
    execute_unfused_materializing_baseline,
)


def _fmt_pct(frac: Float64) -> String:
    var p100 = Int(frac * 10000.0 + 0.5)
    var whole = p100 // 100
    var dec = p100 % 100
    var dec_s = String(dec) if dec >= 10 else String("0") + String(dec)
    return String(whole) + String(".") + dec_s + String("%")


def _fmt_ms(ms: Float64) -> String:
    var x100 = Int(ms * 100.0 + 0.5)
    var whole = x100 // 100
    var dec = x100 % 100
    var dec_s = String(dec) if dec >= 10 else String("0") + String(dec)
    return String(whole) + String(".") + dec_s + String(" ms")


def _fmt_f2(val: Float64) -> String:
    var neg = val < 0.0
    var av = -val if neg else val
    var x100 = Int(av * 100.0 + 0.5)
    var whole = x100 // 100
    var dec = x100 % 100
    var dec_s = String(dec) if dec >= 10 else String("0") + String(dec)
    var prefix = String("-") if neg else String("")
    return prefix + String(whole) + String(".") + dec_s


def demo_petabyte_catalog_and_slice_pushdown() raises:
    print("=" * 78)
    print("1. PETABYTE CATALOG & O(log N) COORDINATE SLICE PUSHDOWN (ZERO DASK GRAPH)")
    print("=" * 78)

    var t0 = perf_counter_ns()
    # 6,000 GDPS cycles x 81 forecast hours x 20 levels x 2400 x 1201 x 5 vars
    # = 48.6 million slabs per 3D var (~1.12 PB logical float32 footprint!)
    var pb_grid = GDPSGridSpec.standard_gdps_15km(n_runs=6000)
    var ds_pb = Dataset.open_virtual_gdps(pb_grid)
    var open_us = Float64(perf_counter_ns() - t0) * 1.0e-3

    print("Opened 6,000-cycle ECCC GDPS 0.15 deg archive in", _fmt_f2(open_us), "us:")
    print(ds_pb.summary())

    # Push down an Eastern Canada / NW Atlantic regional & vertical window:
    var t1 = perf_counter_ns()
    var ds_reg = ds_pb.sel(
        time_min=6.0,
        time_max=72.0,
        level_min=500.0,
        level_max=1000.0,
        lon_min=280.0,     # 80W (Great Lakes / Ontario)
        lon_max=312.0,     # 48W (Newfoundland / Labrador Sea)
        lat_min=42.0,      # 42N (Nova Scotia / Southern Ontario)
        lat_max=58.0,      # 58N (Northern Quebec / Labrador)
        max_runs=1,        # Latest 00Z cycle
    )
    var sel_us = Float64(perf_counter_ns() - t1) * 1.0e-3

    print(
        "Applied .sel(time=6..72h, level=1000..500hPa, lon=280..312E, lat=42..58N) in",
        _fmt_f2(sel_us),
        "us:",
    )
    print(
        "  -> Pruned slabs per 3D var:",
        pb_grid.total_slabs_per_3d_var(),
        "-->",
        ds_reg.window.active_slabs(),
        "slabs",
    )
    print(
        "  -> Pruned Fortran sub-slab: (2400, 1201) --> (",
        ds_reg.window.sub_ni(),
        ",",
        ds_reg.window.sub_nj(),
        ")",
    )


def demo_heterogeneous_mask_propagation() raises:
    print("\n" + "=" * 78)
    print("2. SEAMLESS MULTI-ENCODING MASK PROPAGATION (FORTRAN SLABS)")
    print("=" * 78)

    var grid = GDPSGridSpec.standard_gdps_15km(n_runs=1).copy()
    var ds = Dataset.open_virtual_gdps(grid).sel(
        time_min=12.0,
        time_max=48.0,
        lon_min=230.0,     # Spanning Rockies + Continent + NW Atlantic (230E..320E)
        lon_max=320.0,
        lat_min=35.0,
        lat_max=62.0,
    )

    # A) Compare 1000 hPa (intersects Rockies & Laurentians -> BITMASK) vs
    #    500 hPa (clear of terrain across N. America -> auto-promotes to ALL_VALID!)
    var tt_1000 = ds["TT"].sel_level(1000.0).compute_first_slab()
    var tt_500 = ds["TT"].sel_level(500.0).compute_first_slab()
    var state_1000 = String("MASK_BITMASK") if tt_1000.mask_state == MASK_BITMASK else String("MASK_ALL_VALID")
    var state_500 = String("MASK_ALL_VALID (zero-cost fast path!)") if tt_500.mask_state == MASK_ALL_VALID else String("MASK_BITMASK")

    print(
        "[A] Orographic Sentinel Normalization (`TT` - ENC_QUANT16_SENTINEL):"
    )
    print(
        "    1000 hPa slab state:",
        state_1000,
        "| valid cells:",
        tt_1000.valid_count(),
        "/",
        tt_1000.size(),
        "(Rockies & Laurentians masked)",
    )
    print(
        "     500 hPa slab state:",
        state_500,
        "| valid cells:",
        tt_500.valid_count(),
        "/",
        tt_500.size(),
    )

    # B) Cross-encoding expression: Marine Air-Sea Instability `TM_sst - TT_950hPa`
    #    Fuses 1-bit land/sea bitmap (`TM`) + 16-bit terrain sentinel (`TT`)!
    var air_sea_delta = (ds["TM"] - ds["TT"]).sel_level(950.0)
    var delta_rep = air_sea_delta.compute_stats()
    print(
        "\n[B] Cross-Encoding Fusion (`TM_sst - TT_950hPa` across N. America + Atlantic):"
    )
    print(
        "    Land bitmap (`TM`) | Terrain sentinel (`TT`) -> masked fraction:",
        _fmt_pct(delta_rep.stats.masked_fraction()),
    )
    print(
        "    Valid Atlantic ocean air-sea delta mean:",
        _fmt_f2(delta_rep.stats.mean()),
        "degC (min:",
        _fmt_f2(Float64(delta_rep.stats.min_val)),
        ", max:",
        _fmt_f2(Float64(delta_rep.stats.max_val)),
        ") in",
        _fmt_ms(delta_rep.elapsed_ms),
    )

    # C) 7-Node Multi-Variable Synoptic Diagnostic + Declarative Threshold Mask:
    #    `moist_jet = ((hypot(UU, VV) * 0.514444) * HU / (TT + 273.15)).mask_lt(0.35)`
    #    Combines: UU/VV/TT (sentinel) + HU (QC byte flag) + division domain + threshold!
    var sub_jet = ds.sel(level_min=500.0, level_max=1000.0)
    var wspd_ms = hypot(sub_jet["UU"], sub_jet["VV"], "wspd_ms").scale_offset(
        0.514444, 0.0, "m/s"
    )
    var tt_kelvin = sub_jet["TT"].scale_offset(1.0, 273.15, "K")
    var moist_jet = ((wspd_ms * sub_jet["HU"]) / tt_kelvin).mask_lt(0.35).rename(
        "moist_jet_idx", "m*g/(s*kg*K)"
    )

    # Register derived lazy variable back onto the Dataset (just like Xarray!)
    sub_jet.assign("moist_jet_idx", moist_jet)
    var jet_rep = sub_jet["moist_jet_idx"].compute_stats()
    var mean_2d = sub_jet["moist_jet_idx"].compute_spatial_mean()
    var map_stats = slab_reduce_stats(mean_2d)

    print(
        "\n[C] 7-Op Multi-Encoding Moist Jet DAG (`UU`, `VV`, `HU`, `TT` + `.mask_lt(0.35)`):"
    )
    print(
        "    DAG nodes:",
        len(moist_jet.graph.nodes),
        "| Active slabs:",
        jet_rep.active_slabs,
        "| Workers:",
        jet_rep.num_workers,
    )
    print(
        "    3D Reduced Mean:",
        _fmt_f2(jet_rep.stats.mean()),
        "| Masked fraction (Terrain | QC | Calm <0.35):",
        _fmt_pct(jet_rep.stats.masked_fraction()),
    )
    print(
        "    2D Column/Time Mean Map shape: (",
        mean_2d.ni,
        ",",
        mean_2d.nj,
        ") Fortran-contiguous | Valid cells:",
        map_stats.valid_count,
        "/",
        mean_2d.size(),
    )


def demo_throughput_benchmark() raises:
    print("\n" + "=" * 78)
    print("3. MULTI-CORE THROUGHPUT & MEMORY ARENA BENCHMARK (2,400 GDPS SLABS)")
    print("=" * 78)

    # 10 cycles x 24 forecast steps x 10 pressure levels = 2,400 slabs
    # Each slab is 600 x 301 Fortran Float32 (~722 KB/slab), 7-node DAG -> ~12.1 GB logical array traffic!
    var bench_grid = GDPSGridSpec.benchmark_gdps_grid(
        n_runs=10, n_steps=24, ni=600, nj=301
    )
    var ds = Dataset.open_virtual_gdps(bench_grid)

    var wspd_ms = hypot(ds["UU"], ds["VV"]).scale_offset(0.514444, 0.0, "m/s")
    var tt_k = ds["TT"].scale_offset(1.0, 273.15, "K")
    var diagnostic = ((wspd_ms * ds["HU"]) / tt_k).mask_lt(0.25)

    # Warm-up pass on 1 run (240 slabs):
    _ = diagnostic.sel(max_runs=1).compute_stats()

    # 1. Unfused chunk-materializing baseline (allocates per-chunk intermediate slabs):
    var unfused = execute_unfused_materializing_baseline(
        diagnostic.grid, diagnostic.window, diagnostic.graph, diagnostic.root_id
    )

    # 2. Fused bounded-arena Mojo pull engine (reuses 1 L2/L3 workspace per CPU thread):
    var fused = diagnostic.compute_stats()

    print(
        "Workload: 2,400 slabs x (600 x 301) Fortran grid x 8 DAG regs =",
        _fmt_f2(Float64(fused.logical_array_bytes_processed) / 1.0e9),
        "GB logical tensor traffic",
    )
    print("CPU Hardware Workers:", fused.num_workers)
    print("-" * 78)
    print(
        "Unfused Chunk-Allocating Baseline :",
        _fmt_ms(unfused.elapsed_ms),
        "|",
        _fmt_f2(unfused.us_per_slab()),
        "us/slab | Unbounded intermediate footprint:",
        _fmt_f2(Float64(unfused.peak_arena_bytes) / 1.0e9),
        "GB",
    )
    print(
        "Fused Bounded-Arena Mojo Executor :",
        _fmt_ms(fused.elapsed_ms),
        "|",
        _fmt_f2(fused.us_per_slab()),
        "us/slab | Total fixed scratch arena:       ",
        _fmt_f2(Float64(fused.peak_arena_bytes) / 1.0e6),
        "MB",
    )
    print(
        "Effective Fused Throughput        :",
        _fmt_f2(fused.throughput_gb_s()),
        "GB/s (includes raw binary synthesis + 3-encoding decode + mask SIMD!)",
    )
    print(
        "Verification (exact match)        : unfused mean =",
        _fmt_f2(unfused.stats.mean()),
        "vs fused mean =",
        _fmt_f2(fused.stats.mean()),
    )
    print("=" * 78)


def main() raises:
    demo_petabyte_catalog_and_slice_pushdown()
    demo_heterogeneous_mask_propagation()
    demo_throughput_benchmark()
