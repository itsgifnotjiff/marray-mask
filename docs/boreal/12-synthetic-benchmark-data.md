# Synthetic benchmark grids (U/O/Z/L/G/N/M)

In-memory synthetic cubes for **marray-mask** compute / morsel-VM benches. They mirror catalog-representative RPN `grtyp` geometries and missingness encodings **without** suite paths or on-disk FST I/O.

Implementation: [`src/marray/synth.mojo`](../../src/marray/synth.mojo). Demo: `pixi run synth-bench`.

## Why synthetic

- Scale `(ni, nj, levels, steps)` without opening operational archives.
- Control mask rates (terrain, land/sea, QC, polar disk, yin–yang overlap).
- Exercise Fortran slab layout + bitmask decode paths already in [`slab.mojo`](../../src/marray/slab.mojo).

## Catalog-informed coverage (headers only)

Shares below are approximate unique-header frequencies from an operational FST header catalog (~2.8M rows, `nk=1`). Use them to choose **representative** benches, not full type coverage.

| `grtyp` | ~share | Tier | Synth role |
|---------|-------:|------|------------|
| **U** | 53% | 1 | Yin–Yang supergrid |
| **O** | 23% | 1 | Regional / coastal-biased mesh-like lat-lon |
| **Z** | 15% | 1 | Tic-tac + rotated lon/lat |
| **G** | 3% | 1 | Gaussian latitudes |
| **N** | 3% | 1 | Northern polar stereo |
| **L** | 2% | 1 | Regular geographic lat/lon |
| **M** | ≪1% | 1 | Node-list mesh (edge case) |
| A/B/E/F/X/Y | rare | 2 | Deferred |

**U+Z+N+M alone ≈ 71%** and misses **O**. Tier-1 **U/O/Z/L/G/N/M ≈ 99%** of those header rows.

Path-pattern YAMLs describe suite/lead naming, not grid packing—do not use them as the geometry checklist.

## Scope split

1. **Compute / morsel VM (this module)** — geometry + missingness on logical slabs (`ENC_*` in slab).
2. **FST unpack (later)** — true `datyp=134` (dominant ~76% of catalog packings), 133, 1, 5 belong next to boreal-io `unpack_*`. v1 does **not** claim 134 codec fidelity.

| Catalog packing | ~share | Synth analogue |
|-----------------|-------:|----------------|
| `datyp=134`, 16-bit | ~76% | Not in v1 (unpack follow-on) |
| `datyp=2`, 1-bit `@@` | ~16% | `ENC_FLOAT32_BITMAP` |
| `datyp=1` compact | ~3% | `ENC_QUANT16_SENTINEL` |
| `datyp=5` IEEE | ~1% | float32 payload path |
| typvar `P@` / `@@` | large | Bitmap polarity via bitmap encoder |

## Geometry recipes (realistic enough, not georef oracles)

| Grid | Dims / axes | Mask hooks |
|------|-------------|------------|
| **L** | Regular lon ∈ [0,360), lat ∈ [-90,90] | Terrain sentinel; land/sea bitmap |
| **G** | Regular lon; approximate Gaussian lat (sin-spaced) | Same + polar-adjacent care |
| **Z** | Irregular 1-D `x`/`y` (tic-tac style); 2-D lon/lat from synthetic rotated pole | Terrain / land via lon/lat |
| **U** | `yin_ny = nj/2`; two panels stacked on `j` | Optional overlap band; panel-local lon/lat |
| **N** | Projected `x`,`y` ∈ [-1,1]; geographic via stereo | Outside unit disk masked |
| **O** | Regional lon/lat box (e.g. NW Atlantic–ish) | Heavier coastal land mask |
| **M** | `nj=1`, `ni=n_nodes` irregular nodes | Sparse land/QC on nodes |

Vertical: hybrid- and pressure-like level lists (`ip1_kind` diversity approximated by mixed hPa lists). Time: uniform lead hours.

## Presets

| Name | Intent | Typical scale |
|------|--------|----------------|
| `tiny` | Unit tests | ~32×17, few levels/steps |
| `bench` | Default wall-clock | ~200×101 (or U stacked), 10 levels, 8 steps |
| `large` | Bandwidth / scale | ~600×301 class; `scaled(factor)` multiplies `ni`/`nj` |

Exact factory sizes live in `SynthGridSpec`—adjust freely; keep Fortran `(ni,nj)` layout.

## API sketch

```text
spec = SynthGridSpec.bench(GRTYP_U)   # or O/Z/L/G/N/M
axes = build_axes(spec)
scratch = RawLayerScratch()
dst = MaskedSlab()
fill_and_decode_slab(spec, axes, VAR_TT, step_hr, level_hpa, 0, ni, 0, nj, scratch, dst)
# or:
n_cells = materialize_var_cube(spec, axes, VAR_UU, scratch, out_f32, out_mask_words, out_states)
```

Metrics: wall ms, effective GB/s of decoded f32, `masked_fraction` from `slab_reduce_stats`.

## Non-goals

- Full 13-`grtyp` coverage; bit-identical librmn / ezscint remaps.
- On-disk FST write (boreal-io).
- Embedding operational filesystem roots in fixtures or docs.
