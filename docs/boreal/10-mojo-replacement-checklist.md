# Mojo replacement checklist (map to roadmap)

Engine milestones: [`../ARCHITECTURE_AND_ROADMAP.md`](../ARCHITECTURE_AND_ROADMAP.md) §8. This page maps them onto **boreal / aether** deliverables.

## Two phases of UX

| Phase | Surface | Keeps working |
|-------|---------|----------------|
| **A — Duck array** | `__array_namespace__` / xarray BackendEntrypoint under real `xarray.Dataset` | aether `Regridder`, postproc plugins, hvplot, CF HTML repr |
| **B — Pure Mojo Dataset** | `marray` Dataset / DataArray for PB batch without dask imports | Suite-scale open→sel→reduce→sink |

Ship A first so polaris/mogc/postproc keep calling xarray APIs while kernels bypass dask.

## Milestone mapping

### M1 — RecordCatalog + ChunkDecoder

| Boreal concept | marray deliverable |
|----------------|--------------------|
| Mojo header rows (`path`, `addr`, `lng`, `datyp`, `nbits`, `ni/nj/nk`, keys) | `MmapRecordCatalog` / flat struct array |
| `unpack_*_into` / `unpack_many_*` | `trait ChunkDecoder` |
| `mask_addr` / `mask_lng` + polarity | Decode-time bitmask construction |
| fst-classifier header Parquet | Optional feedstock (no payloads) |
| mozarr / `write_fst` materialize windows | Streaming `sink_to_binary` / Zarr without full `.compute()` |

**Acceptance:** multi-file archive > tens of GB opens and streams under a hard RAM cap; catalog open is O(headers), not O(chunks)×Python graph.

### M2 — Rank-generic reductions

Match boreal-stat zonavg/meriavg/timavg/globavg shapes and weighted `areacell` means without `n_workers * out_cells` accumulator blowups.

### M3 — Boolean / `where` / mask algebra

Parity with postproc mask-replace and compound filters (`(TT < 0) & (PR > 1)` style) using bitmask registers—not `numpy.ma`.

### M4 — xarray BackendEntrypoint + duck array

- `xr.open_dataset(..., engine="marray")` or wrap cubes from a boreal-compatible open.
- `.sel` / `.isel`, attrs, datetime indexes work; **no dask import** required for compute.
- Preserve [06](06-cf-attrs-and-consumer-contract.md) `fst_*` contract.

### M5 — Stencils / align / groupby

- Finite differences / vorticity-class ops used in postproc (where not delegated to aether).
- Coordinate align for plugin pairing; temporal resample/groupby later.
- **aether CSR remapping stays external** until IR explicitly absorbs it.

### M6 — Multi-node + GPU

Shard embarrassingly parallel leads/members (polaris/mogc style caches become shard workers); optional GPU for ensemble-heavy thermo.

## Concrete acceptance sketches

1. **Open:** Equivalent of `open_fst(paths, engine="mojo")` catalog for a multi-lead GDPS-style archive in ≪1 s wall and ≪ RAM of a dask HLG for record chunks.
2. **W1-style:** regional `.sel` + moist-jet expression + 2-D mean with bitmask missingness (roadmap W1).
3. **Postproc-shaped:** Humidex/WindModulus-equivalent fused plan without peel/rewrap copies.
4. **Remap chain:** hand Buffer/duck array to aether CSR (or future stencil) for H then V without per-field Python materialize.
5. **Write:** stream subset to FST/Zarr without building a full eager cube.

## Priority order (boreal-informed)

1. Catalog + decode + bitmasks (kills open HLG + ma/NaN tax).
2. Peel-free elementwise + reductions (kills postproc/stat copies).
3. Duck array under xarray (keeps aether/apps).
4. Buffer handoff to aether / paint.
5. Stencils / groupby / multi-node.
