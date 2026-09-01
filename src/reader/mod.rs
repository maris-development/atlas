//! How a collection is read.
//!
//! An open reads the container's tail, and the deletion mask if one exists.
//! Nothing else. That one read answers every metadata question. Which datasets
//! exist, what arrays they hold, the type and shape of each array, and every
//! attribute. None of it costs more I/O.
//!
//! Atlas fetches array data only when [`DatasetView::read_array`] asks for it.
//! The first such call on a dataset opens its segment, in two small range
//! reads. The call then fetches only the chunks that overlap the region. A
//! dataset nobody reads costs nothing.
//!
//! There is no write path here. A collection cannot change after a write.
//! [`Atlas::delete_dataset`] is the one exception. It rewrites the mask
//! sidecar, and never touches the container.

use std::collections::BTreeSet;
use std::sync::Arc;

use array_format::{
    ArrayElement, ArrayFile, ArrayStats, DType, DeltaCache, FileConfig, FillValue, NoCompression,
};
use indexmap::IndexMap;
use object_store::path::Path as OsPath;
use object_store::{ObjectStore, ObjectStoreExt};
use parking_lot::RwLock;
use tracing::debug;

use crate::config::{DEFAULT_CACHE_CAPACITY, DEFAULT_IO_CACHE_CAPACITY};
use crate::format::footer::{ArrayStatsS, CollectionFooter, DatasetEntry};
use crate::format::segment_store::SegmentStore;
use crate::format::{self, DATA_FILE, MASK_FILE, child, mask};
use crate::schema::{ArraySchema, Attr, DatasetSchema};
use crate::{Error, Result};

/// An open collection.
///
/// Cheap to clone in the way that matters. [`dataset`](Self::dataset) hands
/// out a view with no I/O, and every view shares one block cache.
pub struct Atlas {
    store: Arc<dyn ObjectStore>,
    prefix: OsPath,
    data_path: OsPath,
    footer: Arc<CollectionFooter>,
    /// Ordinals the deletion mask hides. A lock guards the set, because
    /// `delete_dataset` updates it in place.
    deleted: Arc<RwLock<BTreeSet<u32>>>,
    /// One `ArrayFile` per dataset ordinal. Each one opens on demand.
    segments: Arc<Vec<tokio::sync::OnceCell<Arc<ArrayFile>>>>,
    cache: Arc<DeltaCache>,
    /// Size of `data.atlas`. The head request that opened it reports this.
    container_bytes: u64,
}

impl Atlas {
    /// Opens the collection under `prefix` in `store`.
    pub async fn open(store: Arc<dyn ObjectStore>, prefix: OsPath) -> Result<Self> {
        let data_path = child(&prefix, DATA_FILE);
        let size = match store.head(&data_path).await {
            Ok(meta) => meta.size,
            Err(object_store::Error::NotFound { .. }) => {
                return Err(Error::NotAnAtlasCollection {
                    hint: format!("no '{DATA_FILE}' under this prefix"),
                });
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
            container_bytes: size,
        })
    }

    /// Opens a collection stored in a local directory.
    pub async fn open_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let store = object_store::local::LocalFileSystem::new_with_prefix(path.as_ref())?;
        Self::open(Arc::new(store), OsPath::default()).await
    }

    /// Reads the trailer and the footer. One request covers both when the
    /// footer fits in the tail probe. Most collections fit.
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

        // The trailer holds the magic and the version, so it settles whether
        // this file is ours. The leading magic serves `file` and its like.
        // Check it only when the probe already covered it.
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

    /// Reads the deletion mask. An absent mask means nothing is deleted.
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

    /// Names of the live datasets, in write order. Deleted datasets stay out.
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

    /// One set of statistics for an array, over every live dataset. It gives
    /// the minimum, the maximum, how many elements equal the fill value, and
    /// how many elements there are.
    ///
    /// This comes from the footer, so it costs nothing. A deleted dataset
    /// stays out. So does a dataset that declares the name with another dtype,
    /// because two dtypes do not compare. Returns `None` when no live dataset
    /// holds statistics for the array.
    ///
    /// Use [`array_stats_by_dataset`](Self::array_stats_by_dataset) for the
    /// same numbers split per dataset, and [`DatasetView::array_stats`] for
    /// one dataset on its own.
    pub fn array_stats(&self, array: &str) -> Option<ArrayStats> {
        let deleted = self.deleted.read();
        let mut dtype: Option<&DType> = None;
        let mut merged: Option<ArrayStatsS> = None;
        for (ordinal, ds) in self.footer.datasets.iter().enumerate() {
            if deleted.contains(&(ordinal as u32)) {
                continue;
            }
            let Some((position, _, meta)) = self.footer.schema_of(ds).arrays.get_full(array) else {
                continue;
            };
            match dtype {
                None => dtype = Some(&meta.dtype),
                Some(first) if *first != meta.dtype => continue,
                Some(_) => {}
            }
            let Some((_, stats)) = ds
                .array_stats
                .iter()
                .find(|(pos, _)| *pos as usize == position)
            else {
                continue;
            };
            match &mut merged {
                Some(acc) => acc.merge(stats),
                None => merged = Some(stats.clone()),
            }
        }
        merged.map(|s| s.to_array_stats(array))
    }

    /// What each live dataset recorded about one array, in write order.
    ///
    /// One entry per live dataset that holds statistics for `array`. The
    /// deletion mask applies, so a hidden dataset never appears. A dataset
    /// that does not declare the array does not appear either. The list is
    /// empty when no live dataset holds the array.
    ///
    /// This comes from the footer, so it costs nothing.
    ///
    /// Unlike [`array_stats`](Self::array_stats), this keeps a dataset that
    /// declares the name with another dtype. Nothing merges here, so two
    /// dtypes never have to compare.
    pub fn array_stats_by_dataset(&self, array: &str) -> Vec<(String, ArrayStats)> {
        let deleted = self.deleted.read();
        let mut out = Vec::new();
        for (ordinal, ds) in self.footer.datasets.iter().enumerate() {
            if deleted.contains(&(ordinal as u32)) {
                continue;
            }
            let Some(position) = self.footer.schema_of(ds).arrays.get_index_of(array) else {
                continue;
            };
            let Some((_, stats)) = ds
                .array_stats
                .iter()
                .find(|(pos, _)| *pos as usize == position)
            else {
                continue;
            };
            out.push((ds.name.clone(), stats.to_array_stats(array)));
        }
        out
    }

    /// When the collection was written, in milliseconds since the Unix epoch.
    pub fn created_unix_ms(&self) -> i64 {
        self.footer.created_unix_ms
    }

    /// The container format version of this collection.
    pub fn format_version(&self) -> u32 {
        self.footer.version
    }

    /// The block codec the writer used. For information only. Every block
    /// records its own codec, so a read never consults this.
    pub fn codec(&self) -> crate::Codec {
        self.footer.codec
    }

    /// Size of `data.atlas` in bytes.
    pub fn container_bytes(&self) -> u64 {
        self.container_bytes
    }

    /// Datasets in the container, with those the mask hides.
    pub fn total_datasets(&self) -> usize {
        self.footer.datasets.len()
    }

    /// How many distinct schemas the datasets share between them. This falls
    /// below [`total_datasets`](Self::total_datasets) when two datasets hold
    /// the same arrays. That is the common case.
    pub fn interned_schemas(&self) -> usize {
        self.footer.schema_pool.len()
    }

    /// A view of one dataset. This costs no I/O. The view answers metadata
    /// from the footer. It opens the segment only for a data read.
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
    /// The container does not change. The dataset's bytes stay where they
    /// are, and no other ordinal moves. Only a full rewrite reclaims the
    /// space.
    ///
    /// Concurrent deletes are last-writer-wins. Serialize them if that
    /// matters.
    ///
    /// Use [`delete_datasets`](Self::delete_datasets) for more than one name.
    /// It costs the same two requests, whatever the count.
    pub async fn delete_dataset(&self, name: &str) -> Result<()> {
        self.delete_datasets([name]).await?;
        Ok(())
    }

    /// Hides many datasets in one pass. Returns how many the mask gained.
    ///
    /// The cost is two requests, whatever the number of names: one read of the
    /// mask, and one write of it. Delete ten thousand datasets for the price
    /// of one.
    ///
    /// A repeated name counts once. Every name must be live, so an absent or
    /// already deleted one returns [`Error::DatasetNotFound`] and writes
    /// nothing. Check the names against
    /// [`list_datasets`](Self::list_datasets) first to skip that.
    ///
    /// The container does not change, exactly as for one name.
    pub async fn delete_datasets<I, S>(&self, names: I) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let wanted: BTreeSet<String> = names.into_iter().map(|n| n.as_ref().to_string()).collect();
        if wanted.is_empty() {
            return Ok(0);
        }

        // One pass over the footer, so a large batch costs a scan and no more.
        // A name at a time would be one scan each.
        let mut ordinals: BTreeSet<u32> = BTreeSet::new();
        let mut found: BTreeSet<&str> = BTreeSet::new();
        {
            let deleted = self.deleted.read();
            for (ordinal, ds) in self.footer.datasets.iter().enumerate() {
                if wanted.contains(ds.name.as_str()) && !deleted.contains(&(ordinal as u32)) {
                    ordinals.insert(ordinal as u32);
                    found.insert(ds.name.as_str());
                }
            }
        }
        if found.len() != wanted.len() {
            let missing = wanted
                .iter()
                .find(|n| !found.contains(n.as_str()))
                .expect("a count mismatch leaves at least one name unfound");
            return Err(Error::DatasetNotFound(missing.clone()));
        }

        // Re-read the mask instead of the in-memory set. That keeps a delete
        // another handle made since this one opened.
        let mut deleted =
            Self::read_mask(&self.store, &self.prefix, self.footer.datasets.len()).await?;
        deleted.extend(&ordinals);
        let path = child(&self.prefix, MASK_FILE);
        self.store.put(&path, mask::encode(&deleted).into()).await?;
        debug!(count = ordinals.len(), "datasets deleted via mask");
        *self.deleted.write() = deleted;
        Ok(ordinals.len())
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
/// Metadata comes from the collection footer, and costs nothing. Only
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

    /// The dataset's position in the collection. It is stable for the life of
    /// the container. The deletion mask names it.
    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// Where this dataset's segment sits in `data.atlas`.
    ///
    /// The range holds a complete `array-format` file. Copy those bytes out,
    /// and they open on their own. Atlas plays no part.
    pub fn segment_range(&self) -> std::ops::Range<u64> {
        let entry = self.entry();
        entry.seg_offset..(entry.seg_offset + entry.seg_len)
    }

    /// The dataset's schema. Every array it declares, in definition order.
    pub fn schema(&self) -> &DatasetSchema {
        self.footer.schema_of(self.entry())
    }

    /// Array names, in definition order.
    pub fn list_arrays(&self) -> Vec<String> {
        self.schema().arrays.keys().cloned().collect()
    }

    /// The schema of one array. `None` if this dataset does not declare it.
    pub fn array_meta(&self, array: &str) -> Option<&ArraySchema> {
        self.schema().arrays.get(array)
    }

    /// The fill value of one array. A read returns it for every element
    /// nobody wrote.
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

    /// The attributes of one array, in the order somebody set them. Empty for
    /// an array with none, and for a name this dataset does not declare.
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

    /// What the write recorded about one array. The minimum, the maximum, how
    /// many elements equal the fill value, and how many there are.
    ///
    /// This comes from the footer, so it costs nothing. Returns `None` for an
    /// array this dataset does not declare, and for one nobody wrote.
    pub fn array_stats(&self, array: &str) -> Option<ArrayStats> {
        let position = self.schema().arrays.get_index_of(array)?;
        self.entry()
            .array_stats
            .iter()
            .find(|(pos, _)| *pos as usize == position)
            .map(|(_, stats)| stats.to_array_stats(array))
    }

    /// Reads a region of `array`.
    ///
    /// Pass an empty `start` and `shape` to read the whole array. Atlas
    /// fetches only the chunks the region overlaps. Every element nobody wrote
    /// comes from the fill value. `T` must match the array's declared type.
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
        // The segment addresses the chunks. The footer describes the array.
        // One write produces both, so a mismatch means a damaged file.
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

    /// Opens this dataset's segment. This runs once per collection handle.
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
                // A block records its own codec, so the reader never needs
                // the one the writer used.
                let file = ArrayFile::open(
                    Arc::new(segment) as Arc<dyn ObjectStore>,
                    path,
                    FileConfig {
                        codec: NoCompression,
                        block_target_size: crate::config::DEFAULT_BLOCK_TARGET_SIZE,
                        // Unused. `cache` below overrides both budgets.
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
