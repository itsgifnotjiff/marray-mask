# Masking and missing data

Module: `boreal.io.xarray.masks` (pairing helpers also under `boreal.io.mask`).

FST missingness is **heterogeneous**: companion bitmaps, packing sentinels, and RPN numeric fill. boreal-io normalizes **in-memory** cubes toward xarray-friendly NaN while preserving **on-disk** write identity.

## Bitmap pairing (`@@` / `!@`)

- Data records whose `typvar` ends with `@` (and are not themselves bitmaps) pair with companion bitmap records (`@@`, `!@`).
- Pairing keys typically include path, nomvar, etiket, datev, ip1, ip2, ip3.
- `attach_mask_addresses` adds `mask_addr` / `mask_lng` (and fill metadata) onto masked data rows; bitmap rows remain in the DataFrame until field selection drops them for CF cubes.

### Polarity (fstd2nc-compatible)

```text
keep where bitmap > 0
fill where bitmap == 0
```

Implemented as `numpy.where(mask > 0, field, fill_value)` after unpack (`apply_bitmap_mask`).

## Dual fill convention

| Context | Value | Constant / attr |
|---------|-------|-----------------|
| In-memory Dataset (default) | NaN | `DATASET_FILL_VALUE` |
| Classic RPN / fstd2nc numeric identity | `1e30` | `DEFAULT_FILL_VALUE` / `FST_MASK_FILL` |
| Attr for writers | `1e30` | `attrs["fst_mask_fill"]` |
| xarray encoding | in-memory fill | `missing_value` / `encoding["_FillValue"]` |

Default `open_fst(..., apply_masks=True, fill_value=NaN)` so `.isnull()` / `to_masked_array()` work. Pass `fill_value=1e30` when bit-identical numeric compare to older tools is required. Bitmap-linked variables keep `fst_mask_fill=1e30` for `write_fst`.

## Alt-mask (`datyp + 64`)

When the alt-mask convention is set, the **maximum value** in the unpacked field marks missing (`apply_alt_mask`). Can combine with a companion bitmap.

## Interaction with lazy modes

- Column/record dask tasks apply masks **inside** the unpack task (mask addr/lng carried in the unpack spec tuple).
- `chunks="virtual"` **defaults `apply_masks=False`** and **rejects `apply_masks=True`**: one ManifestArray chunk is a single byte range; a masked field would need the data range **and** the bitmap range.

## Mapping to marray-mask bitmasks

marray-mask encodes missingness as a 3-state `UInt64` word stream (`MSTATE_VALID` | `MSTATE_MASKED` | `MSTATE_BITS`) instead of `numpy.ma` or in-band NaN.

| FST / boreal-io source | Suggested decode |
|------------------------|------------------|
| Bitmap `== 0` cells | `MSTATE_MASKED` (or bit plane) during/after unpack |
| Alt-mask max sentinel | Mark masked in the same pass as unpack |
| In-memory NaN cubes already opened | `isnan` → bitmask on ingest |
| Clean upper-air slabs (no bitmap) | Promote to `MSTATE_VALID` (zero mask storage) |
| Write-out to FST | Materialize NaN or `1e30` per `fst_mask_fill` contract |

Ideal hot path: **single memory pass** from packed bytes → float payload + bitmask words (roadmap §2.1 / §2.2), folding bitmap apply into decode rather than a second full-slab `np.where`.
