# Migrating a 0.14 store to 0.15

There is no in-place upgrade. The 0.14 layout — `atlas.json` plus a directory
per array name — and the 0.15 container share no bytes, and 0.15 cannot read the
old format. Opening a 0.14 store with 0.15 says so:

```text
not an atlas collection: found 'atlas.json' instead of 'data.atlas': this is an
atlas 0.14 store, whose format this build cannot read (rewrite it with atlas 0.15)
```

Migration means reading with 0.14 and writing with 0.15. The two packages cannot
coexist in one environment, so use two.

## Recipe

```bash
python -m venv .venv-old && .venv-old/bin/pip install "atlas-python==0.14.*"
python -m venv .venv-new && .venv-new/bin/pip install "atlas-python>=0.15"
```

**Step 1 — export with 0.14.** Read each dataset back as xarray and write it to
NetCDF:

```python
# run with .venv-old/bin/python
import pathlib
import atlas

src = atlas.Atlas.open("/data/old_store")
out = pathlib.Path("/data/intermediate")
out.mkdir(exist_ok=True)

for name in src.list_datasets():
    src.open_as_xarray_dataset(name).to_netcdf(out / f"{name}.nc")
    print("exported", name)
```

**Step 2 — import with 0.15.** One collection from the lot:

```python
# run with .venv-new/bin/python
import pathlib
import atlas
import xarray as xr

files = sorted(pathlib.Path("/data/intermediate").glob("*.nc"))

with atlas.AtlasWriter.create("/data/new_collection") as w:
    for nc in files:
        w.add_xarray_dataset(xr.open_dataset(nc), name=nc.stem)
        print("imported", nc.stem)

print(atlas.Atlas.open("/data/new_collection").list_datasets())
```

The intermediate NetCDF files exist only because the two atlas versions cannot
be imported together. Delete them afterwards.

## What to expect on the other side

**Deleted datasets are gone for good.** 0.14's `list_datasets` already hides
tombstoned datasets, so they are simply not exported. This is the one chance to
drop them for real — the new collection reclaims their space.

**Ordinals change.** 0.15 assigns them in write order. If you stored 0.14
ordinals anywhere, re-derive them from the new collection.

**Chunk shapes carry through NetCDF imperfectly.** A dataset that was chunked in
0.14 comes back from `to_netcdf` without dask chunking unless you re-chunk it.
Pass `chunks=` to `add_xarray_dataset` to set the on-disk chunking explicitly.

**RFC 3339 string attributes stay strings.** 0.14 stored timestamp attributes as
RFC 3339 strings and turned any string that parsed as one back into a timestamp.
0.15 has a real timestamp type and does not guess. An attribute that *should* be
a timestamp needs to be set as one:

```python
ds.set_attribute("created", 1_700_000_000_000_000_000, dtype="timestamp_nanoseconds")
```

**Type mismatches are no longer reported.** 0.14 warned or raised when two
datasets declared the same array name with unmergeable types. 0.15 stores each
dataset's types as declared and has no merged schema, so nothing complains. If
you relied on that warning as a data-quality check, do the check yourself
against the per-dataset schemas.

**Nothing replaces the pruning index.** 0.14's `pruning_index` and
`column_summaries` built min/max/null statistics across datasets for scan
pruning. 0.15 has no statistics. Attribute-based filtering still works and is
now free — attributes are in the footer, so filtering a collection by
`ds.get_attribute("site")` reads nothing beyond the open.

## Sanity check

After importing, compare what you expect against what landed:

```python
old_names = set(...)          # from the 0.14 export log
new = atlas.Atlas.open("/data/new_collection")
assert set(new.list_datasets()) == old_names
for name in new.list_datasets():
    view = new.dataset(name)
    print(name, view.list_arrays())
```

Verifying array *values* needs the Rust API, since Python no longer reads data.
`tests/cross_fixture.rs` shows the shape of that check.
