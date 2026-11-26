"""Time fill+decode for tier-1 synthetic grids (U/O/Z/L/G/N/M)."""

from std.collections import List
from std.time import perf_counter_ns
from marray import (
    GRTYP_U,
    GRTYP_O,
    GRTYP_Z,
    GRTYP_L,
    GRTYP_G,
    GRTYP_N,
    GRTYP_M,
    VAR_TT,
    VAR_TM,
    SynthGridSpec,
    MaskedSlab,
    RawLayerScratch,
    build_axes,
    fill_and_decode_slab,
    slab_mask_fraction,
)


def _bench_one(grtyp: Int):
    var spec = SynthGridSpec.bench(grtyp)
    var axes = build_axes(spec)
    var scratch = RawLayerScratch()
    var dst = MaskedSlab()
    var n_slabs = spec.total_slabs()
    var n_cells = spec.n_cells()

    var t0 = perf_counter_ns()
    var last_frac = Float64(0.0)
    for t in range(spec.n_steps):
        var step_hr = Float32(Float64(t) * spec.step_dt_hr)
        for z in range(spec.n_levels()):
            var level = Float32(spec.levels_hpa[z])
            var vid = VAR_TM if (z % 3) == 2 else VAR_TT
            fill_and_decode_slab(
                spec,
                axes,
                vid,
                step_hr,
                level,
                0,
                spec.ni,
                0,
                spec.nj,
                scratch,
                dst,
            )
            last_frac = slab_mask_fraction(dst)
    var t1 = perf_counter_ns()
    var ms = Float64(t1 - t0) / Float64(1_000_000.0)
    var bytes_out = Float64(n_slabs * n_cells) * Float64(4.0)
    var gbps = (bytes_out / Float64(1.0e9)) / (ms / Float64(1000.0)) if ms > Float64(
        0.0
    ) else Float64(0.0)
    print(
        "grtyp=",
        spec.name(),
        " ni=",
        spec.ni,
        " nj=",
        spec.nj,
        " slabs=",
        n_slabs,
        " ms=",
        ms,
        " GB/s~",
        gbps,
        " last_mask_frac=",
        last_frac,
    )


def main():
    print("synth_bench_demo: tier-1 grids at SynthGridSpec.bench")
    var types = List[Int]()
    types.append(GRTYP_U)
    types.append(GRTYP_O)
    types.append(GRTYP_Z)
    types.append(GRTYP_L)
    types.append(GRTYP_G)
    types.append(GRTYP_N)
    types.append(GRTYP_M)
    for k in range(len(types)):
        _bench_one(types[k])
