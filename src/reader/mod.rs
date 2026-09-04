//! How a collection is read.
//!
//! An open reads the container's tail, and the deletion mask if one exists.
//! Nothing else. That one read answers which datasets exist, what arrays and
//! attribute keys they declare, and the type of each. None of it costs more
//! I/O.
//!
//! The footer therefore describes the collection, and nothing more. It decodes
//! no value. Shape, chunking, attribute values, and statistics all live in the
//! segment that holds the data, and each carries its own type tag.
//! [`DatasetView::array_layout`] opens that segment to answer.
//!
//! A segment holds one **variable** across the whole collection, keyed by
//! dataset name. So the first read of `temperature` opens one file, in two
//! small range reads, and every other dataset's `temperature` then comes from
//! the same handle. The call fetches only the blocks the region overlaps, and
//! a block holds a run of neighbouring datasets. A variable nobody reads costs
//! nothing.
//!
//! There is no write path here. A collection cannot change after a write.
//! [`Atlas::delete_dataset`] is the one exception. It rewrites the mask
//! sidecar, and never touches the container.

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use array_format::{ArrayElement, ArrayFile, ArrayStats, BlockCache, ReadConfig, StatValue};
use indexmap::IndexMap;
use object_store::path::Path as OsPath;
use object_store::{ObjectStore, ObjectStoreExt};
use parking_lot::RwLock;
use tracing::debug;

use crate::config::{DEFAULT_CACHE_CAPACITY, DEFAULT_IO_CACHE_CAPACITY};
use crate::format::footer::CollectionFooter;
use crate::format::segment_store::SegmentStore;
use crate::format::{self, DATA_FILE, DATASET_ATTRS_VARIABLE, MASK_FILE, child, mask};
use crate::schema::{ArrayLayout, ArrayMeta, Attr, SchemaView};
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
    /// One `ArrayFile` per variable, in footer order. Each opens on demand,
    /// and every dataset that declares the array shares it.
    segments: Arc<Vec<tokio::sync::OnceCell<Arc<ArrayFile>>>>,
    cache: Arc<BlockCache>,
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

        let segments = (0..footer.variables.len())
            .map(|_| tokio::sync::OnceCell::new())
            .collect();
        Ok(Self {
            store,
            prefix,
            data_path,
            footer: Arc::new(footer),
            deleted: Arc::new(RwLock::new(deleted)),
            segments: Arc::new(segments),
            cache: Arc::new(BlockCache::new(
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
            .keys()
            .enumerate()
            .filter(|(i, _)| !deleted.contains(&(*i as u32)))
            .map(|(_, name)| name.to_string())
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
        // A schema names its arrays by pool index, so the walk collects
        // integers. Only the distinct ones resolve to a string.
        let mut ids: BTreeSet<u32> = BTreeSet::new();
        for (i, schema) in self.footer.datasets.values().enumerate() {
            if deleted.contains(&(i as u32)) {
                continue;
            }
            for &(name, _) in &self.footer.schema_of(*schema).arrays {
                ids.insert(name);
            }
        }
        let mut names: Vec<&str> = ids
            .into_iter()
            .filter_map(|id| self.footer.string(id))
            .collect();
        names.sort_unstable();
        names.into_iter().map(str::to_string).collect()
    }

    /// One set of statistics for an array, over every live dataset. It gives
    /// the minimum, the maximum, how many elements equal the fill value, and
    /// how many elements there are.
    ///
    /// The bytes come from that variable's segment, so the first call opens
    /// it. Every later call on the same handle is free. A deleted dataset
    /// stays out. Returns `None` when no live dataset holds the array.
    ///
    /// Use [`array_stats_by_dataset`](Self::array_stats_by_dataset) for the
    /// same numbers split per dataset, and [`DatasetView::array_stats`] for
    /// one dataset on its own.
    pub async fn array_stats(&self, array: &str) -> Result<Option<ArrayStats>> {
        let mut merged: Option<ArrayStats> = None;
        for stats in self.stats_of(array).await? {
            match &mut merged {
                Some(acc) => merge_stats(acc, &stats),
                None => {
                    let mut first = stats;
                    first.name = array.to_string();
                    merged = Some(first);
                }
            }
        }
        Ok(merged)
    }

    /// What each live dataset recorded about one array, in write order.
    ///
    /// Every entry names its dataset in [`ArrayStats::name`], so a row
    /// identifies itself. Join it to [`list_datasets`](Self::list_datasets)
    /// by that name. A dataset that does not declare the array has no entry,
    /// and neither has one the deletion mask hides.
    ///
    /// The list is empty when no live dataset holds the array.
    pub async fn array_stats_by_dataset(&self, array: &str) -> Result<Vec<ArrayStats>> {
        self.stats_of(array).await
    }

    /// The statistics of `array`, per live dataset.
    async fn stats_of(&self, array: &str) -> Result<Vec<ArrayStats>> {
        let Some(file) = self.try_segment(array).await? else {
            return Ok(Vec::new());
        };
        let hidden = self.deleted_names();
        Ok(file
            .stats()
            .filter(|stats| !hidden.contains(stats.name.as_str()))
            .cloned()
            .collect())
    }

    /// One attribute, over every live dataset that carries it.
    ///
    /// `array` names the array the attribute annotates. `None` reads the
    /// dataset-level attribute instead, out of the reserved `_datasets`
    /// segment.
    pub async fn attributes_by_dataset(
        &self,
        array: Option<&str>,
        attr: &str,
    ) -> Result<IndexMap<String, Attr>> {
        let variable = array.unwrap_or(DATASET_ATTRS_VARIABLE);
        let Some(segment) = self.try_segment(variable).await? else {
            return Ok(IndexMap::new());
        };
        // `list_datasets` walks the footer in write order, so the map keeps it.
        let mut out = IndexMap::new();
        for dataset in self.list_datasets() {
            if let Some(value) = read_attribute(segment, attr, &dataset) {
                out.insert(dataset, value);
            }
        }
        Ok(out)
    }

    /// Names of the datasets the deletion mask hides.
    fn deleted_names(&self) -> HashSet<&str> {
        let deleted = self.deleted.read();
        deleted
            .iter()
            .filter_map(|ordinal| self.footer.datasets.get_index(*ordinal as usize))
            .map(|(name, _)| name.as_str())
            .collect()
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

    /// Opens the segment that holds `array`. This runs once per collection
    /// handle, however many datasets ask for the same variable.
    async fn segment(&self, array: &str) -> Result<&Arc<ArrayFile>> {
        open_segment(
            &self.store,
            &self.data_path,
            &self.footer,
            &self.segments,
            &self.cache,
            array,
        )
        .await
    }

    /// The same, but `None` for a variable the collection does not hold. A
    /// collection-wide call answers about a name nobody declared, instead of
    /// failing on it.
    async fn try_segment(&self, array: &str) -> Result<Option<&Arc<ArrayFile>>> {
        match self.segment(array).await {
            Ok(file) => Ok(Some(file)),
            Err(Error::ArrayNotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// A view of one dataset. This costs no I/O. The view answers names,
    /// types, and statistics from the footer. It opens a segment for a data
    /// read, for a layout, and for an attribute value.
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
            for (ordinal, name) in self.footer.datasets.keys().enumerate() {
                if wanted.contains(name.as_str()) && !deleted.contains(&(ordinal as u32)) {
                    ordinals.insert(ordinal as u32);
                    found.insert(name.as_str());
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
        // The footer keys its datasets by name, so this is one hash lookup
        // whether the collection holds ten datasets or a million.
        self.footer
            .datasets
            .get_index_of(name)
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
    /// One `ArrayFile` per variable, shared with every other view.
    segments: Arc<Vec<tokio::sync::OnceCell<Arc<ArrayFile>>>>,
    cache: Arc<BlockCache>,
    ordinal: u32,
}

impl DatasetView {
    /// Which interned schema this dataset declares.
    fn schema_index(&self) -> u32 {
        let (_, schema) = self
            .footer
            .datasets
            .get_index(self.ordinal as usize)
            .expect("the view holds a live ordinal");
        *schema
    }

    /// The dataset's name. Every variable segment stores this dataset's
    /// array under it.
    pub fn name(&self) -> &str {
        self.footer
            .datasets
            .get_index(self.ordinal as usize)
            .map(|(name, _)| name.as_str())
            .expect("the view holds a live ordinal")
    }

    /// The dataset's position in the collection. It is stable for the life of
    /// the container. The deletion mask names it.
    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// What the dataset declares: every array name and element type, in
    /// definition order, and the same for its attributes.
    ///
    /// The view borrows from the footer and copies no name. Call
    /// [`SchemaView::to_owned_schema`] for an owned copy.
    pub fn schema(&self) -> SchemaView<'_> {
        self.footer.schema_view(self.schema_index())
    }

    /// Array names, in definition order.
    pub fn list_arrays(&self) -> Vec<String> {
        self.schema().names().map(str::to_string).collect()
    }

    /// The name and element type of one array. `None` if this dataset does
    /// not declare it.
    ///
    /// Shape and chunking are not here, because the footer does not hold
    /// them. Call [`array_layout`](Self::array_layout) for those.
    pub fn array_meta(&self, array: &str) -> Option<ArrayMeta<'_>> {
        self.schema().get(array)
    }

    /// The shape, chunking, dimension names, and fill value of one array.
    ///
    /// This opens the array's segment, unlike every other metadata call. The
    /// segment records the layout already, so the footer does not repeat it.
    ///
    /// A segment covers one variable across the whole collection, so the open
    /// happens once however many datasets ask. To sweep every dataset's
    /// layout therefore costs one open per variable.
    pub async fn array_layout(&self, array: &str) -> Result<ArrayLayout> {
        if self.schema().index_of(array).is_none() {
            return Err(Error::ArrayNotFound(array.to_string()));
        }
        let file = self.segment(array).await?;
        let info = file
            .array(self.name())
            .ok_or_else(|| Error::ArrayNotFound(array.to_string()))?;
        Ok(ArrayLayout::from_stored(info))
    }

    /// Dataset-level attributes, in the order they were set.
    ///
    /// The values live in the reserved `_datasets` segment, so this reads it.
    /// A dataset that declares no attribute costs no I/O, because the schema
    /// already says so.
    pub async fn attributes(&self) -> Result<IndexMap<String, Attr>> {
        let keys: Vec<&str> = self.schema().attribute_names().collect();
        if keys.is_empty() {
            return Ok(IndexMap::new());
        }
        let file = self.segment(DATASET_ATTRS_VARIABLE).await?;
        Ok(Self::resolve(file, self.name(), keys))
    }

    /// One dataset-level attribute.
    pub async fn get_attribute(&self, key: &str) -> Result<Option<Attr>> {
        let file = self.segment(DATASET_ATTRS_VARIABLE).await?;
        Ok(read_attribute(file, key, self.name()))
    }

    /// Reads `keys` off the entry named `entry` in `file`.
    fn resolve(file: &ArrayFile, entry: &str, keys: Vec<&str>) -> IndexMap<String, Attr> {
        keys.into_iter()
            .filter_map(|key| Some((key.to_string(), read_attribute(file, key, entry)?)))
            .collect()
    }

    /// The attributes of one array, in the order somebody set them. Empty for
    /// an array with none, and for a name this dataset does not declare.
    ///
    /// The values live on the array's own entry in that variable's segment, so
    /// this reads it. An array with no attribute costs no I/O.
    pub async fn array_attributes(&self, array: &str) -> Result<IndexMap<String, Attr>> {
        let Some(meta) = self.array_meta(array) else {
            return Ok(IndexMap::new());
        };
        let keys: Vec<&str> = meta.attribute_names();
        if keys.is_empty() {
            return Ok(IndexMap::new());
        }
        let file = self.segment(array).await?;
        Ok(Self::resolve(file, self.name(), keys))
    }

    /// One attribute of one array.
    pub async fn get_array_attribute(&self, array: &str, key: &str) -> Result<Option<Attr>> {
        let file = self.segment(array).await?;
        Ok(read_attribute(file, key, self.name()))
    }

    /// What the write recorded about one array. The minimum, the maximum, how
    /// many elements equal the fill value, and how many there are.
    ///
    /// This comes from the footer, so it costs nothing. Returns `None` for an
    /// array this dataset does not declare, and for one nobody wrote.
    pub async fn array_stats(&self, array: &str) -> Result<Option<ArrayStats>> {
        if self.schema().index_of(array).is_none() {
            return Ok(None);
        }
        let file = self.segment(array).await?;
        Ok(file.array_stats(self.name()).map(|stats| {
            let mut owned = stats.clone();
            owned.name = array.to_string();
            owned
        }))
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
        let meta = self
            .array_meta(array)
            .ok_or_else(|| Error::ArrayNotFound(array.to_string()))?;
        if *meta.dtype() != T::DTYPE {
            return Err(Error::CorruptCollection(format!(
                "array '{array}' of dataset '{}' is {:?}, not {:?}",
                self.name(),
                meta.dtype(),
                T::DTYPE
            )));
        }
        let file = self.segment(array).await?;
        // The segment keys on the dataset name, because it holds this array
        // for every dataset in the collection.
        Ok(file.read_array::<T>(self.name(), start, shape).await?)
    }

    /// Opens the segment that holds `array`. This runs once per collection
    /// handle, however many datasets ask for the same variable.
    async fn segment(&self, array: &str) -> Result<&Arc<ArrayFile>> {
        open_segment(
            &self.store,
            &self.data_path,
            &self.footer,
            &self.segments,
            &self.cache,
            array,
        )
        .await
    }
}

/// One attribute off one entry of a segment, or `None`.
///
/// A segment keys its entries by dataset name, so `dataset` names the entry.
/// An entry the segment does not hold reads as `None`, exactly as a key it
/// does not carry: neither is an error, because a dataset need not declare
/// every array of the collection.
fn read_attribute(segment: &ArrayFile, attr: &str, dataset: &str) -> Option<Attr> {
    segment
        .get_attribute(dataset, attr)
        .map(Attr::from_stored)
}

/// Folds one dataset's statistics for an array into the running total.
///
/// The counts add up. Each bound takes the wider of the two. A bound that is
/// absent yields to a bound that is present.
fn merge_stats(into: &mut ArrayStats, other: &ArrayStats) {
    into.null_count = into.null_count.saturating_add(other.null_count);
    into.row_count = into.row_count.saturating_add(other.row_count);
    into.min = wider(into.min.take(), other.min.clone(), Bound::Min);
    into.max = wider(into.max.take(), other.max.clone(), Bound::Max);
}

/// Which end of the range [`wider`] keeps.
#[derive(Clone, Copy)]
enum Bound {
    Min,
    Max,
}

/// The smaller or the larger of two bounds. `None` means a dataset recorded no
/// bound, so the other one stands.
fn wider(a: Option<StatValue>, b: Option<StatValue>, bound: Bound) -> Option<StatValue> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(a), Some(b)) => {
            let keep_a = match bound {
                Bound::Min => is_le(&a, &b),
                Bound::Max => !is_le(&a, &b),
            };
            Some(if keep_a { a } else { b })
        }
    }
}

/// Orders two bounds of one variant. Two variants mean two dtypes, and the
/// caller excludes those before it merges.
fn is_le(a: &StatValue, b: &StatValue) -> bool {
    match (a, b) {
        (StatValue::Int(a), StatValue::Int(b)) => a <= b,
        (StatValue::UInt(a), StatValue::UInt(b)) => a <= b,
        (StatValue::Float(a), StatValue::Float(b)) => a.total_cmp(b).is_le(),
        (StatValue::Bytes(a), StatValue::Bytes(b)) => a <= b,
        (StatValue::TimestampNs(a), StatValue::TimestampNs(b)) => a <= b,
        _ => false,
    }
}

/// Opens the segment that holds `array`, once per collection handle.
///
/// `Atlas` and `DatasetView` both need it. `Atlas` reads the reserved
/// attribute segment, and a view reads a variable's own.
async fn open_segment<'a>(
    store: &Arc<dyn ObjectStore>,
    data_path: &OsPath,
    footer: &CollectionFooter,
    segments: &'a [tokio::sync::OnceCell<Arc<ArrayFile>>],
    cache: &Arc<BlockCache>,
    array: &str,
) -> Result<&'a Arc<ArrayFile>> {
    let index = footer
        .string_id(array)
        .and_then(|id| footer.variable_index(id))
        .ok_or_else(|| Error::ArrayNotFound(array.to_string()))?;
    segments[index]
        .get_or_try_init(|| async {
            let entry = &footer.variables[index];
            debug!(
                variable = array,
                seg_offset = entry.seg_offset,
                seg_len = entry.seg_len,
                "opening segment"
            );
            let segment = SegmentStore::new(
                Arc::clone(store),
                data_path.clone(),
                index as u32,
                entry.seg_offset,
                entry.seg_len,
            );
            let path = segment.path();
            // Every segment shares the collection's cache, so the per-file
            // budgets do not apply.
            let file = ArrayFile::open(
                Arc::new(segment) as Arc<dyn ObjectStore>,
                path,
                ReadConfig {
                    cache: Some(Arc::clone(cache)),
                    ..ReadConfig::default()
                },
            )
            .await?;
            Ok::<_, Error>(Arc::new(file))
        })
        .await
}

impl std::fmt::Debug for DatasetView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DatasetView")
            .field("name", &self.name())
            .field("ordinal", &self.ordinal)
            .field("arrays", &self.schema().len())
            .finish_non_exhaustive()
    }
}
