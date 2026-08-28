//! Reading a collection.
//!
//! Opening reads the container's tail and, if present, the deletion mask.
//! Nothing else. Every metadata question — which datasets exist, what arrays
//! they hold, what type and shape each array has, what attributes are attached
//! — is answered from that one read, with no further I/O.
//!
//! Array data is fetched only when [`DatasetView::read_array`] asks for it. The
//! first such call on a dataset opens its segment (two small range reads); the
//! call itself then fetches only the chunks that overlap the requested region.
//! A dataset you never read costs nothing.
//!
//! There is no write path here. A collection cannot be modified after it is
//! written; the one exception is [`Atlas::delete_dataset`], which rewrites the
//! mask sidecar and never touches the container.

use std::collections::BTreeSet;
use std::sync::Arc;

use array_format::{ArrayElement, ArrayFile, DeltaCache, FileConfig, FillValue, NoCompression};
use indexmap::IndexMap;
use object_store::path::Path as OsPath;
use object_store::{ObjectStore, ObjectStoreExt};
use parking_lot::RwLock;
use tracing::debug;

use crate::config::{DEFAULT_CACHE_CAPACITY, DEFAULT_IO_CACHE_CAPACITY};
use crate::format::footer::{CollectionFooter, DatasetEntry};
use crate::format::segment_store::SegmentStore;
use crate::format::{self, DATA_FILE, LEGACY_META_FILE, MASK_FILE, child, mask};
use crate::schema::{ArraySchema, Attr, DatasetSchema};
use crate::{Error, Result};

/// An open collection.
///
/// Cheap to clone in the sense that matters: [`dataset`](Self::dataset) hands
/// out views without I/O, and every view shares one block cache.
pub struct Atlas {
    store: Arc<dyn ObjectStore>,
    prefix: OsPath,
    data_path: OsPath,
    footer: Arc<CollectionFooter>,
    /// Ordinals hidden by the deletion mask. Behind a lock because
    /// `delete_dataset` updates it in place.
    deleted: Arc<RwLock<BTreeSet<u32>>>,
    /// One lazily-opened `ArrayFile` per dataset ordinal.
    segments: Arc<Vec<tokio::sync::OnceCell<Arc<ArrayFile>>>>,
    cache: Arc<DeltaCache>,
}

impl Atlas {
    /// Opens the collection under `prefix` in `store`.
    pub async fn open(store: Arc<dyn ObjectStore>, prefix: OsPath) -> Result<Self> {
        let data_path = child(&prefix, DATA_FILE);
        let size = match store.head(&data_path).await {
            Ok(meta) => meta.size,
            Err(object_store::Error::NotFound { .. }) => {
                return Err(Self::missing_container_error(&store, &prefix).await);
            }
            Err(e) => return Err(Error::ObjectStore(e)),
        };

        let footer = Self::read_footer(&store, &data_path, size).await?;
        let deleted = Self::read_mask(&store, &prefix, footer.datasets.len()).await?;
        debug!(
            datasets = footer.datasets.len(),
            deleted = deleted.len(),
            "opened collection"
        );

        let segments = (0..footer.datasets.len())
            .map(|_| tokio::sync::OnceCell::new())
            .collect();
        Ok(Self {
            store,
            prefix,
            data_path,
            footer: Arc::new(footer),
            deleted: Arc::new(RwLock::new(deleted)),
            segments: Arc::new(segments),
            cache: Arc::new(DeltaCache::new(
                DEFAULT_CACHE_CAPACITY,
                DEFAULT_IO_CACHE_CAPACITY,
            )),
        })
    }

    /// Opens a collection stored in a local directory.
    pub async fn open_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let store = object_store::local::LocalFileSystem::new_with_prefix(path.as_ref())?;
        Self::open(Arc::new(store), OsPath::default()).await
    }

    /// Reads the trailer and footer. One request covers both when the footer
    /// fits in the tail probe, which it does for most collections.
    async fn read_footer(
        store: &Arc<dyn ObjectStore>,
        data_path: &OsPath,
        size: u64,
    ) -> Result<CollectionFooter> {
        if size < format::HEADER_SIZE + format::TRAILER_SIZE {
            return Err(Error::NotAnAtlasCollection {
                hint: format!("file is {size} bytes, too short to hold a header and a trailer"),
            });
        }
        let probe = format::TAIL_PROBE_SIZE.min(size);
        let tail = store.get_range(data_path, (size - probe)..size).await?;
        let footer_size = format::decode_trailer(&tail)?;

        // The trailer already carries the magic and the version, so it settles
        // whether this file is ours. The leading magic is there for `file` and
        // friends. Check it when the probe happened to cover it, and skip the
        // extra round trip when it did not.
        if size <= probe {
            format::check_header(&tail[(probe - size) as usize..])?;
        }

        let footer_end = size - format::TRAILER_SIZE;
        let footer_start = footer_end.checked_sub(footer_size).ok_or_else(|| {
            Error::CorruptCollection(format!(
                "footer claims {footer_size} bytes but the file holds only {footer_end} before the trailer"
            ))
        })?;
        if footer_start < format::HEADER_SIZE {
            return Err(Error::CorruptCollection(format!(
                "footer of {footer_size} bytes would start inside the container header"
            )));
        }

        let bytes = if footer_size + format::TRAILER_SIZE <= probe {
            let from = (probe - format::TRAILER_SIZE - footer_size) as usize;
            let to = (probe - format::TRAILER_SIZE) as usize;
            tail.slice(from..to)
        } else {
            store.get_range(data_path, footer_start..footer_end).await?
        };
        CollectionFooter::decode(&bytes)
    }

    /// Reads the deletion mask. Absent means nothing is deleted.
    async fn read_mask(
        store: &Arc<dyn ObjectStore>,
        prefix: &OsPath,
        dataset_count: usize,
    ) -> Result<BTreeSet<u32>> {
        let path = child(prefix, MASK_FILE);
        match store.get(&path).await {
            Ok(r) => mask::decode(&r.bytes().await?, dataset_count),
            Err(object_store::Error::NotFound { .. }) => Ok(BTreeSet::new()),
            Err(e) => Err(Error::ObjectStore(e)),
        }
    }

    /// Builds the error for a prefix with no `data.atlas`, naming the 0.14
    /// layout when that is what is actually there.
    async fn missing_container_error(store: &Arc<dyn ObjectStore>, prefix: &OsPath) -> Error {
        let legacy = child(prefix, LEGACY_META_FILE);
        if store.head(&legacy).await.is_ok() {
            return Error::NotAnAtlasCollection {
                hint: format!(
                    "found '{LEGACY_META_FILE}' instead of '{DATA_FILE}': this is an atlas 0.14 \
                     store, whose format this build cannot read (rewrite it with atlas 0.15)"
                ),
            };
        }
        Error::NotAnAtlasCollection {
            hint: format!("no '{DATA_FILE}' under this prefix"),
        }
    }

    /// Names of the live datasets, in write order. Deleted datasets are
    /// omitted.
    pub fn list_datasets(&self) -> Vec<String> {
        let deleted = self.deleted.read();
        self.footer
            .datasets
            .iter()
            .enumerate()
            .filter(|(i, _)| !deleted.contains(&(*i as u32)))
            .map(|(_, d)| d.name.clone())
            .collect()
    }

    /// Whether a live dataset of this name exists.
    pub fn dataset_exists(&self, name: &str) -> bool {
        self.ordinal_of(name).is_some()
    }

    /// How many datasets are live.
    pub fn dataset_count(&self) -> usize {
        self.footer.datasets.len() - self.deleted.read().len()
    }

    /// Every distinct array name across the live datasets, sorted.
    pub fn list_arrays(&self) -> Vec<String> {
        let deleted = self.deleted.read();
        let mut names: BTreeSet<&str> = BTreeSet::new();
        for (i, ds) in self.footer.datasets.iter().enumerate() {
            if deleted.contains(&(i as u32)) {
                continue;
            }
            for name in self.footer.schema_of(ds).arrays.keys() {
                names.insert(name.as_str());
            }
        }
        names.into_iter().map(str::to_string).collect()
    }

    /// When the collection was written, in milliseconds since the Unix epoch.
    pub fn created_unix_ms(&self) -> i64 {
        self.footer.created_unix_ms
    }

    /// A view of one dataset. No I/O: the view answers metadata from the
    /// footer, and opens the segment only if data is read.
    pub fn dataset(&self, name: &str) -> Result<DatasetView> {
        let ordinal = self
            .ordinal_of(name)
            .ok_or_else(|| Error::DatasetNotFound(name.to_string()))?;
        Ok(DatasetView {
            store: Arc::clone(&self.store),
            data_path: self.data_path.clone(),
            footer: Arc::clone(&self.footer),
            segments: Arc::clone(&self.segments),
            cache: Arc::clone(&self.cache),
            ordinal,
        })
    }

    /// Hides a dataset by adding it to the deletion mask.
    ///
    /// The container is untouched: the dataset's bytes stay where they are, and
    /// the ordinals of the other datasets do not move. Only a full rewrite
    /// reclaims the space.
    ///
    /// Concurrent deletions are last-writer-wins. Serialize them if that
    /// matters.
    pub async fn delete_dataset(&self, name: &str) -> Result<()> {
        let ordinal = self
            .ordinal_of(name)
            .ok_or_else(|| Error::DatasetNotFound(name.to_string()))?;
        // Re-read rather than trust the in-memory set, so a deletion made
        // elsewhere since this handle opened is preserved.
        let mut deleted =
            Self::read_mask(&self.store, &self.prefix, self.footer.datasets.len()).await?;
        deleted.insert(ordinal);
        let path = child(&self.prefix, MASK_FILE);
        self.store.put(&path, mask::encode(&deleted).into()).await?;
        debug!(dataset = name, ordinal, "dataset deleted via mask");
        *self.deleted.write() = deleted;
        Ok(())
    }

    fn ordinal_of(&self, name: &str) -> Option<u32> {
        let deleted = self.deleted.read();
        self.footer
            .datasets
            .iter()
            .position(|d| d.name == name)
            .map(|i| i as u32)
            .filter(|i| !deleted.contains(i))
    }
}

impl std::fmt::Debug for Atlas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Atlas")
            .field("prefix", &self.prefix)
            .field("datasets", &self.dataset_count())
            .finish_non_exhaustive()
    }
}

/// A read-only view of one dataset.
///
/// Metadata comes from the collection footer and costs nothing. Only
/// [`read_array`](Self::read_array) touches the container.
pub struct DatasetView {
    store: Arc<dyn ObjectStore>,
    data_path: OsPath,
    footer: Arc<CollectionFooter>,
    segments: Arc<Vec<tokio::sync::OnceCell<Arc<ArrayFile>>>>,
    cache: Arc<DeltaCache>,
    ordinal: u32,
}

impl DatasetView {
    fn entry(&self) -> &DatasetEntry {
        &self.footer.datasets[self.ordinal as usize]
    }

    /// The dataset's name.
    pub fn name(&self) -> &str {
        &self.entry().name
    }

    /// The dataset's position in the collection. Stable for the life of the
    /// container, and what the deletion mask refers to.
    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// Where this dataset's segment sits in `data.atlas`.
    ///
    /// The range holds a complete `array-format` file. Copy those bytes out and
    /// they open on their own, with no atlas involved.
    pub fn segment_range(&self) -> std::ops::Range<u64> {
        let entry = self.entry();
        entry.seg_offset..(entry.seg_offset + entry.seg_len)
    }

    /// The dataset's schema: every array it declares, in definition order.
    pub fn schema(&self) -> &DatasetSchema {
        self.footer.schema_of(self.entry())
    }

    /// Array names, in definition order.
    pub fn list_arrays(&self) -> Vec<String> {
        self.schema().arrays.keys().cloned().collect()
    }

    /// The schema of one array, or `None` if this dataset does not declare it.
    pub fn array_meta(&self, array: &str) -> Option<&ArraySchema> {
        self.schema().arrays.get(array)
    }

    /// The fill value of one array: what a read returns for elements that were
    /// never written.
    pub fn array_fill_value(&self, array: &str) -> Option<FillValue> {
        self.array_meta(array)?.fill_value.clone().map(Into::into)
    }

    /// Dataset-level attributes, in the order they were set.
    pub fn attributes(&self) -> IndexMap<String, Attr> {
        self.footer.attrs_to_map(&self.entry().global_attrs)
    }

    /// One dataset-level attribute.
    pub fn get_attribute(&self, key: &str) -> Option<Attr> {
        self.attributes().shift_remove(key)
    }

    /// The attributes of one array, in the order they were set. Empty for an
    /// array with none, and for a name this dataset does not declare.
    pub fn array_attributes(&self, array: &str) -> IndexMap<String, Attr> {
        let Some(position) = self.schema().arrays.get_index_of(array) else {
            return IndexMap::new();
        };
        self.entry()
            .array_attrs
            .iter()
            .find(|(pos, _)| *pos as usize == position)
            .map(|(_, attrs)| self.footer.attrs_to_map(attrs))
            .unwrap_or_default()
    }

    /// One attribute of one array.
    pub fn get_array_attribute(&self, array: &str, key: &str) -> Option<Attr> {
        self.array_attributes(array).shift_remove(key)
    }

    /// Reads a region of `array`.
    ///
    /// Pass empty `start` and `shape` to read the whole array. Only the chunks
    /// that overlap the region are fetched; elements that were never written
    /// come from the fill value. `T` must match the array's declared type.
    pub async fn read_array<T: ArrayElement>(
        &self,
        array: &str,
        start: Vec<usize>,
        shape: Vec<usize>,
    ) -> Result<ndarray::ArcArray<T, ndarray::IxDyn>> {
        let schema = self
            .array_meta(array)
            .ok_or_else(|| Error::ArrayNotFound(array.to_string()))?;
        if schema.dtype != T::DTYPE {
            return Err(Error::CorruptCollection(format!(
                "array '{array}' of dataset '{}' is {:?}, not {:?}",
                self.name(),
                schema.dtype,
                T::DTYPE
            )));
        }
        let expected_shape = schema.shape.clone();
        let file = self.segment().await?;
        // The segment addresses the chunks; the footer describes the array.
        // They are written together, so a disagreement means a damaged file.
        let stored = file.get_array(array)?;
        let stored_shape: Vec<usize> = stored.layout.shape.iter().map(|&s| s as usize).collect();
        if stored_shape != expected_shape {
            return Err(Error::CorruptCollection(format!(
                "array '{array}' of dataset '{}' is {expected_shape:?} in the footer but \
                 {stored_shape:?} in its segment",
                self.name()
            )));
        }
        Ok(file.read_array::<T>(array, start, shape).await?)
    }

    /// Opens this dataset's segment, once per collection handle.
    async fn segment(&self) -> Result<&Arc<ArrayFile>> {
        self.segments[self.ordinal as usize]
            .get_or_try_init(|| async {
                let entry = self.entry();
                debug!(
                    dataset = %entry.name,
                    seg_offset = entry.seg_offset,
                    seg_len = entry.seg_len,
                    "opening segment"
                );
                let segment = SegmentStore::new(
                    Arc::clone(&self.store),
                    self.data_path.clone(),
                    self.ordinal,
                    entry.seg_offset,
                    entry.seg_len,
                );
                let path = segment.path();
                // Blocks record their own codec, so the reader never needs the
                // one the writer used.
                let file = ArrayFile::open(
                    Arc::new(segment) as Arc<dyn ObjectStore>,
                    path,
                    FileConfig {
                        codec: NoCompression,
                        block_target_size: crate::config::DEFAULT_BLOCK_TARGET_SIZE,
                        // Ignored: `cache` below overrides both budgets.
                        cache_capacity: DEFAULT_CACHE_CAPACITY as usize,
                        io_cache_capacity: DEFAULT_IO_CACHE_CAPACITY as usize,
                        cache: Some(Arc::clone(&self.cache)),
                    },
                )
                .await?;
                Ok::<_, Error>(Arc::new(file))
            })
            .await
    }
}

impl std::fmt::Debug for DatasetView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DatasetView")
            .field("name", &self.name())
            .field("ordinal", &self.ordinal)
            .field("arrays", &self.schema().arrays.len())
            .finish_non_exhaustive()
    }
}
