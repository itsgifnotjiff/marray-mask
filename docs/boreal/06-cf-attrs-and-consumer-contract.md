# CF attrs and consumer contract

Consumers (aether, boreal-postproc, boreal-stat, polaris, mogc-api) treat boreal-io cubes as the stable surface. A Mojo labeled frontend must preserve these semantics even when the backing array is not dask/numpy.

## Return shape

- `open_fst(...)` → **`list[xr.Dataset]`** (possibly empty; one entry per group).
- Helpers in boreal-stat (`as_dataset_list`, `unwrap_singleton`) normalize list / Dataset / DataArray inputs.
- Do not assume all nomvars of a forecast lead live in one Dataset (ragged split, multi-grid files).

## Dimensions and coordinates

| Concept | Typical names |
|---------|----------------|
| Forecast / validity | `time` |
| Vertical | `vertical` (+ level attributes from IP decode) |
| Ensemble | `member` when present |
| Horizontal | `y`/`x`, or `lat`/`lon`, or `rlat`/`rlon` (+ 2-D geographic) |
| Cell area | `areacell` / `areacell_yy` when `cell_measures` / yin-yang mask enabled |

Trailing spatial layout is **`(…, nj, ni)`** interpreted as `(…, y, x)` (or lat/lon analogues)—not Fortran `(ni, nj)` axis order in the xarray dims list.

## Variable identity

| Attr / mechanism | Role |
|------------------|------|
| `fst_nomvar` | Canonical FST code (`TT`, `UU`, …); always present even after OpenData rename |
| Dataset / DataArray **name** | OpenData id when `open_data_mapping=True` (default), else raw nomvar |
| `long_name`, `standard_name`, `units` | CF / dictionary overlay when available |
| `nomvar` | Sometimes mirrored; prefer `fst_nomvar` for lookups |

boreal-stat `find_by_nomvar` / `physical_data_vars` skip CF auxiliaries (e.g. grid_mapping scalars).

## Grid and packing attrs (representative)

Consumers fingerprint and stamp using attrs such as:

- `fst_grtyp`, `fst_ig1` … `fst_ig4`
- `fst_ip1`, `fst_ip2`, `fst_ip3` (and decoded level/interval helpers)
- `fst_etiket` / `fst_label`, `fst_typvar`, `fst_run`
- packing / header JSON-safe dumps where needed for round-trip
- `fst_mask_fill` (write fill; see [05](05-masking-missing-data.md))
- vertical: `fst_vcode`, hybrid `hyb_a`/`hyb_b`/`hyb_c`, formula_terms
- Dataset-level: `fst_meta_records`, `fst_toctoc`, optional source path attrs for repairing `>>`/`^^`

Exact key sets evolve; treat **`fst_*` namespace + CF grid_mapping** as required surface for remappers and writers.

## Postproc stamp / write contract

boreal-postproc resolves fields by name → `fst_nomvar` → OpenData id, then after kernels:

- `stamp_attrs` / `boreal.io.update_metadata`
- preserve units and PDS-align keys (forecast hour, grid, vertical)
- emit `old_*` provenance where boreal-io write paths expect it
- `write_fst` / `zap` / pressure helpers (`add_px_to_dataset`) / `adopt_fst_grid` after remaps

A Mojo Dataset API that drops these attrs will break round-trip and Spooki-facing tools even if numeric payloads match.

## aether expectations

- Prefer CF lon/lat; if missing, recover from RPN grid attrs / positional `>>`/`^^` (gallery and grid helpers patch when open_fst left them as data_vars).
- Grid fingerprint for weight cache: `grtyp` + `ig1..ig4` + sizes.
- Vertical remapper understands boreal-io vcode / HY / ETA quirks (`fst_vcode`, pressure units).

## Layout mismatches to remember

- Pair metrics / some oracles may see `(ni, nj)` vs `(y, x)` swaps—transpose when comparing.
- Mojo unpack F-linear planes vs C cube strides: same bytes, different axis labels ([02](02-fst-decode-pipeline.md)).
- Forcing C-order copies before every kernel is a major tax; duck arrays should advertise Fortran-friendly innermost stride when native.
