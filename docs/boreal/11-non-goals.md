# Non-goals (for this replacement track)

Clear boundaries keep marray-mask focused on **orchestration + masked array compute**, not on re-owning the entire boreal product surface.

## Stays in Python (or sibling packages)

- xarray HTML `repr`, hvplot / matplotlib / cartopy frontends (Phase A duck array is enough).
- polaris / mogc-api HTTP, deck.gl, OGC routing, colormap UX (`boreal-colormaps`).
- aether **weight builders**, Parquet weight I/O, and gallery CLIs.
- Spooki **plugin class dispatch** and CLI argument parsing (payloads may be Mojo).
- boreal-io **write compatibility** oracles (`ci_fstcomp`, fstpy parity gates) as product tests—not reimplemented here first.

## Must leave the GIL / Python scheduler

- FST decode **scheduling** at suite scale (replace HLG task dicts).
- Morsel math, bitmask propagation, reductions.
- Prefetch + unpack overlap with compute.
- Eventually: multi-field expression graphs that today peel every variable.

## Explicitly out of scope (unless milestones say otherwise)

- **Kerchunk** compatibility (boreal-io uses VirtualiZarr + `FstRecordCodec`, not kerchunk).
- Replacing **aether** CSR remapping inside the morsel VM before M5+ explicit design.
- Full **BURP** xarray DSG export (deferred even in boreal-io).
- Bit-identical ULPs with every librmn datyp-1 cell (boreal-io already documents sparse ULP drift).
- Shipping operational suite path configuration or embedding site roots in this repo’s docs/code.

## Documentation hygiene

- Do not paste absolute host paths, home directories, or operational filesystem roots into examples.
- Prefer package contracts and placeholders (`file.std`, `open_fst(paths, ...)`).
