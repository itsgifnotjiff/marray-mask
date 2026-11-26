# End-to-end pipelines (path-free)

Operational file locations are omitted on purpose. These pipelines show where xarray/dask still sit between already-Mojo stages.

## polaris (globe / API)

```mermaid
sequenceDiagram
  participant App as polaris
  participant BIO as boreal-io
  participant PP as boreal-postproc
  participant AE as aether
  participant Viz as colorize
  App->>BIO: open_fst chunks=record engine=mojo
  BIO-->>App: list of Datasets cached per lead
  App->>App: select fst_nomvar; squeeze 2D
  opt derived layers
    App->>PP: Humidex / WindModulus.compute
    PP-->>App: stamped Dataset
  end
  App->>AE: Regridder bilinear to regular lat/lon
  AE-->>App: remapped fields as float32 grids
  App->>Viz: LUT / RGBA packets
```

**Notes:**

- Lazy knob is **`chunks="record"`** (thin FST records), not a Dask cluster.
- Derived humidex/wind are computed on demand from `TT`/`TD` or `UU`/`VV`, not always stored in the file.
- After remap, caches often hold NumPy grids—not live xarray—for packet assembly.
- Mojo already covers unpack + CSR apply; remaining cost is open/graph, peel, and Python request orchestration.

## mogc-api (GetMap / Coverage)

```mermaid
sequenceDiagram
  participant API as mogc-api
  participant BIO as boreal-io
  participant AE as aether
  participant Paint as mojoviz / LUT
  API->>BIO: warmup_mojo
  API->>BIO: open_fst chunks=False engine=mojo
  BIO-->>API: Datasets
  API->>API: level isel; cache 2D float32 + attrs
  API->>AE: BBOX sample / CSR apply / native resize
  AE-->>API: sampled field
  API->>Paint: boreal-colormaps LUT → PNG or CoverageJSON
```

**Notes:**

- Hot path is **anti-Dask**: `chunks=False`, often `apply_masks=False` / `cell_measures=False` for latency.
- Concurrency is thread-pool prefetch + byte budget caches, not dask.distributed.
- Profile stages typically break out `open_fst`, regrid/native resize, PNG, cache hit/miss.
- Zero-copy bridges (NumPy ↔ MAX Buffer / DLPack) exist on the paint side—model for marray handoff.

## Shared lesson

| Stage | Already fast (Mojo) | Still expensive (Python orchestration) |
|-------|---------------------|----------------------------------------|
| FST unpack | `cmc_io_fst_data` | Header→Dataset assembly; optional HLG |
| Remap apply | aether CSR | Per-field Dataset rebuild; weight session setup |
| Thermo derive | postproc kernels | Peel/rewrap + attr stamp |
| Paint | mojoviz / LUT | Building slim Datasets per request |

marray-mask wins by collapsing **open + expression + reduction** (and eventually buffer handoff into aether/paint) without forcing full-slab intermediates between stages.
