//! End-to-end tests for the single-file immutable format.
//!
//! The lifecycle under test is short by design: build a collection, finish it,
//! open it, read from it, delete a dataset, reopen. There is no mutation path
//! to exercise because the format has none.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use atlas::{Atlas, AtlasWriter, Attr, Codec, Error, FillValue, WriterConfig};
use ndarray::{Array1, Array2, ArrayD};
use object_store::path::Path as OsPath;
use object_store::{ObjectStore, ObjectStoreExt};

// ── helpers ──────────────────────────────────────────────────────────

/// Builds a collection with three datasets covering the cases that matter:
/// several dtypes, a chunked array, an array that is defined but never
/// written, fill values, and attributes at both levels.
async fn build_fixture(path: &std::path::Path) {
    let w = AtlasWriter::create_path(path, WriterConfig::default())
        .await
        .unwrap();

    {
        let mut ds = w.add_dataset("jan_2024").await.unwrap();
        ds.define_array::<f32>(
            "temperature",
            vec!["lat".into(), "lon".into()],
            vec![4, 8],
            Some(vec![2, 4]),
            Some(FillValue::Float(f64::NAN)),
        )
        .await
        .unwrap();
        // Four chunks, written as one slab that spans all of them.
        let data =
            ArrayD::from_shape_fn(ndarray::IxDyn(&[4, 8]), |i| (i[0] * 8 + i[1]) as f32).into_dyn();
        ds.write_array("temperature", vec![0, 0], data.view())
            .await
            .unwrap();

        ds.define_array::<i64>("counts", vec!["lat".into()], vec![4], None, None)
            .await
            .unwrap();
        let counts = Array1::from_vec(vec![10i64, 20, 30, 40]).into_dyn();
        ds.write_array("counts", vec![0], counts.view())
            .await
            .unwrap();

        ds.set_attribute("month", Attr::Int64(1));
        ds.set_attribute("source", Attr::String("buoy".into()));
        ds.set_array_attribute("temperature", "units", Attr::String("celsius".into()))
            .unwrap();
        ds.finish().await.unwrap();
    }

    {
        // Same array shapes as jan_2024, so the schema interns to one entry.
        let mut ds = w.add_dataset("feb_2024").await.unwrap();
        ds.define_array::<f32>(
            "temperature",
            vec!["lat".into(), "lon".into()],
            vec![4, 8],
            Some(vec![2, 4]),
            Some(FillValue::Float(f64::NAN)),
        )
        .await
        .unwrap();
        let data = Array2::<f32>::from_elem([4, 8], -1.5).into_dyn();
        ds.write_array("temperature", vec![0, 0], data.view())
            .await
            .unwrap();
        ds.define_array::<i64>("counts", vec!["lat".into()], vec![4], None, None)
            .await
            .unwrap();
        // counts is declared but never written: it must read back as fill.
        ds.set_attribute("month", Attr::Int64(2));
        ds.finish().await.unwrap();
    }

    {
        // A different schema, and the dtypes the xarray layer leans on.
        let mut ds = w.add_dataset("stations").await.unwrap();
        ds.define_array::<String>("name", vec!["station".into()], vec![3], None, None)
            .await
            .unwrap();
        let names = Array1::from_vec(vec![
            "alpha".to_string(),
            "beta".to_string(),
            "gamma".to_string(),
        ])
        .into_dyn();
        ds.write_array("name", vec![0], names.view()).await.unwrap();

        ds.define_array::<array_format::TimestampNs>(
            "observed",
            vec!["station".into()],
            vec![3],
            None,
            Some(FillValue::TimestampNs(i64::MIN)),
        )
        .await
        .unwrap();
        let times = Array1::from_vec(vec![
            array_format::TimestampNs(1_700_000_000_000_000_000),
            array_format::TimestampNs(1_700_000_001_000_000_000),
            array_format::TimestampNs(1_700_000_002_000_000_000),
        ])
        .into_dyn();
        ds.write_array("observed", vec![0], times.view())
            .await
            .unwrap();
        ds.set_attribute(
            "installed",
            Attr::TimestampNanoseconds(1_600_000_000_000_000_000),
        );
        ds.finish().await.unwrap();
    }

    w.finish().await.unwrap();
}

/// Builds `datasets` datasets of one `len`-element array each, uncompressed so
/// the container is genuinely large. Used where a fixture has to exceed the
/// reader's tail probe.
async fn build_bulky_fixture(path: &std::path::Path, datasets: usize, len: usize) {
    let w = AtlasWriter::create_path(
        path,
        WriterConfig {
            codec: Codec::Uncompressed,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    for d in 0..datasets {
        let mut ds = w.add_dataset(&format!("ds{d}")).await.unwrap();
        ds.define_array::<f64>("x", vec!["i".into()], vec![len], Some(vec![len / 4]), None)
            .await
            .unwrap();
        let data = Array1::from_shape_fn(len, |i| (d * len + i) as f64).into_dyn();
        ds.write_array("x", vec![0], data.view()).await.unwrap();
        ds.set_attribute("index", Attr::Int64(d as i64));
        ds.finish().await.unwrap();
    }
    w.finish().await.unwrap();
}

/// An `ObjectStore` that counts requests, so laziness can be asserted rather
/// than assumed.
#[derive(Debug)]
struct CountingStore {
    inner: Arc<dyn ObjectStore>,
    gets: AtomicUsize,
    bytes: AtomicUsize,
}

impl CountingStore {
    fn new(inner: Arc<dyn ObjectStore>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            gets: AtomicUsize::new(0),
            bytes: AtomicUsize::new(0),
        })
    }
    fn gets(&self) -> usize {
        self.gets.load(Ordering::SeqCst)
    }
    fn bytes(&self) -> usize {
        self.bytes.load(Ordering::SeqCst)
    }
    fn reset(&self) {
        self.gets.store(0, Ordering::SeqCst);
        self.bytes.store(0, Ordering::SeqCst);
    }
}

impl std::fmt::Display for CountingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CountingStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for CountingStore {
    async fn put_opts(
        &self,
        location: &OsPath,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }
    async fn put_multipart_opts(
        &self,
        location: &OsPath,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }
    async fn get_opts(
        &self,
        location: &OsPath,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        // A head request transfers no body, so it counts as neither.
        let head = options.head;
        if !head {
            self.gets.fetch_add(1, Ordering::SeqCst);
        }
        let r = self.inner.get_opts(location, options).await?;
        if !head {
            self.bytes
                .fetch_add((r.range.end - r.range.start) as usize, Ordering::SeqCst);
        }
        Ok(r)
    }
    fn list(
        &self,
        prefix: Option<&OsPath>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }
    async fn list_with_delimiter(
        &self,
        prefix: Option<&OsPath>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }
    async fn copy_opts(
        &self,
        from: &OsPath,
        to: &OsPath,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<'static, object_store::Result<OsPath>>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<OsPath>> {
        self.inner.delete_stream(locations)
    }
}

// ── lifecycle ────────────────────────────────────────────────────────

#[tokio::test]
async fn a_finished_collection_reopens_with_all_its_metadata() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;

    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    assert_eq!(
        atlas.list_datasets(),
        vec!["jan_2024", "feb_2024", "stations"]
    );
    assert_eq!(atlas.dataset_count(), 3);
    // Sorted union across datasets.
    assert_eq!(
        atlas.list_arrays(),
        vec!["counts", "name", "observed", "temperature"]
    );

    let jan = atlas.dataset("jan_2024").unwrap();
    assert_eq!(jan.ordinal(), 0);
    assert_eq!(jan.list_arrays(), vec!["temperature", "counts"]);

    let meta = jan.array_meta("temperature").unwrap();
    assert_eq!(meta.shape, vec![4, 8]);
    assert_eq!(meta.chunk_shape, vec![2, 4]);
    assert_eq!(meta.dimension_names, vec!["lat", "lon"]);
    assert_eq!(meta.dtype, atlas::DType::Float32);
    assert!(matches!(
        jan.array_fill_value("temperature"),
        Some(FillValue::Float(f)) if f.is_nan()
    ));

    assert_eq!(jan.get_attribute("month"), Some(Attr::Int64(1)));
    assert_eq!(
        jan.get_attribute("source"),
        Some(Attr::String("buoy".into()))
    );
    assert_eq!(jan.attributes().len(), 2);
    assert_eq!(
        jan.get_array_attribute("temperature", "units"),
        Some(Attr::String("celsius".into()))
    );
    // counts has no attributes of its own.
    assert!(jan.array_attributes("counts").is_empty());
}

#[tokio::test]
async fn arrays_read_back_the_values_that_were_written() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();

    let jan = atlas.dataset("jan_2024").unwrap();
    let temp = jan
        .read_array::<f32>("temperature", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(temp.shape(), &[4, 8]);
    assert_eq!(temp[[0, 0]], 0.0);
    assert_eq!(temp[[3, 7]], 31.0);

    let counts = jan
        .read_array::<i64>("counts", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(counts.as_slice().unwrap(), &[10, 20, 30, 40]);

    let stations = atlas.dataset("stations").unwrap();
    let names = stations
        .read_array::<String>("name", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(names[[1]], "beta");
    let observed = stations
        .read_array::<array_format::TimestampNs>("observed", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(observed[[2]].0, 1_700_000_002_000_000_000);
}

#[tokio::test]
async fn a_partial_read_spanning_chunks_returns_the_right_window() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    let jan = atlas.dataset("jan_2024").unwrap();

    // Chunks are 2x4, so 1..3 x 3..5 straddles all four of them.
    let window = jan
        .read_array::<f32>("temperature", vec![1, 3], vec![2, 2])
        .await
        .unwrap();
    assert_eq!(window.shape(), &[2, 2]);
    assert_eq!(window[[0, 0]], 11.0);
    assert_eq!(window[[0, 1]], 12.0);
    assert_eq!(window[[1, 0]], 19.0);
    assert_eq!(window[[1, 1]], 20.0);
}

#[tokio::test]
async fn an_array_that_was_never_written_reads_as_its_fill_value() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();

    let feb = atlas.dataset("feb_2024").unwrap();
    // Declared, never written, and no explicit fill: zero for an integer.
    let counts = feb
        .read_array::<i64>("counts", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(counts.shape(), &[4]);
    assert_eq!(counts.as_slice().unwrap(), &[0, 0, 0, 0]);
}

#[tokio::test]
async fn identical_schemas_are_stored_once() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();

    // jan and feb declare the same arrays and differ only in attributes, which
    // do not live in the schema. They must share one interned entry.
    let jan = atlas.dataset("jan_2024").unwrap();
    let feb = atlas.dataset("feb_2024").unwrap();
    assert_eq!(jan.schema(), feb.schema());
    assert_ne!(jan.attributes(), feb.attributes());
}

#[tokio::test]
async fn timestamps_and_date_shaped_strings_stay_distinct() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let w = AtlasWriter::create_path(tmp.path(), WriterConfig::default())
            .await
            .unwrap();
        let mut ds = w.add_dataset("d").await.unwrap();
        ds.set_attribute(
            "when",
            Attr::TimestampNanoseconds(1_700_000_000_000_000_000),
        );
        ds.set_attribute("looks_like", Attr::String("2023-11-14T22:13:20Z".into()));
        ds.finish().await.unwrap();
        w.finish().await.unwrap();
    }
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    let ds = atlas.dataset("d").unwrap();
    assert_eq!(
        ds.get_attribute("when"),
        Some(Attr::TimestampNanoseconds(1_700_000_000_000_000_000))
    );
    assert_eq!(
        ds.get_attribute("looks_like"),
        Some(Attr::String("2023-11-14T22:13:20Z".into()))
    );
}

// ── deletion ─────────────────────────────────────────────────────────

#[tokio::test]
async fn deleting_a_dataset_hides_it_here_and_after_a_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let container_before = std::fs::metadata(tmp.path().join("data.atlas"))
        .unwrap()
        .len();

    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    atlas.delete_dataset("feb_2024").await.unwrap();

    // Visible immediately on the handle that deleted it.
    assert_eq!(atlas.list_datasets(), vec!["jan_2024", "stations"]);
    assert!(!atlas.dataset_exists("feb_2024"));
    assert!(matches!(
        atlas.dataset("feb_2024"),
        Err(Error::DatasetNotFound(_))
    ));

    // And after a reopen.
    let reopened = Atlas::open_path(tmp.path()).await.unwrap();
    assert_eq!(reopened.list_datasets(), vec!["jan_2024", "stations"]);
    // The deleted dataset's arrays are gone from the union too.
    assert_eq!(
        reopened.list_arrays(),
        vec!["counts", "name", "observed", "temperature"]
    );

    // The container itself never changed.
    let container_after = std::fs::metadata(tmp.path().join("data.atlas"))
        .unwrap()
        .len();
    assert_eq!(container_before, container_after);
    assert!(tmp.path().join("deleted.mask").exists());
}

#[tokio::test]
async fn ordinals_do_not_shift_when_a_dataset_is_deleted() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    assert_eq!(atlas.dataset("stations").unwrap().ordinal(), 2);

    atlas.delete_dataset("jan_2024").await.unwrap();
    let reopened = Atlas::open_path(tmp.path()).await.unwrap();
    // Still 2. Nothing was renumbered, so a stored ordinal stays valid.
    assert_eq!(reopened.dataset("stations").unwrap().ordinal(), 2);
    assert!(
        reopened
            .dataset("feb_2024")
            .unwrap()
            .read_array::<f32>("temperature", vec![], vec![])
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn deletions_accumulate_in_one_mask() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;

    // Separate handles, so the second must merge with what the first wrote
    // rather than overwrite it.
    Atlas::open_path(tmp.path())
        .await
        .unwrap()
        .delete_dataset("jan_2024")
        .await
        .unwrap();
    Atlas::open_path(tmp.path())
        .await
        .unwrap()
        .delete_dataset("stations")
        .await
        .unwrap();

    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    assert_eq!(atlas.list_datasets(), vec!["feb_2024"]);
}

#[tokio::test]
async fn deleting_twice_reports_the_dataset_as_missing() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    atlas.delete_dataset("jan_2024").await.unwrap();
    assert!(matches!(
        atlas.delete_dataset("jan_2024").await,
        Err(Error::DatasetNotFound(_))
    ));
}

#[tokio::test]
async fn a_mask_naming_an_unknown_dataset_is_ignored_not_fatal() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    // A mask left over from a larger collection: magic, version, count, then
    // ordinals 0 and 99.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"ATLM");
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&99u32.to_le_bytes());
    std::fs::write(tmp.path().join("deleted.mask"), bytes).unwrap();

    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    assert_eq!(atlas.list_datasets(), vec!["feb_2024", "stations"]);
}

#[tokio::test]
async fn a_foreign_file_at_the_mask_path_is_an_error() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    std::fs::write(tmp.path().join("deleted.mask"), b"definitely not a mask").unwrap();
    assert!(matches!(
        Atlas::open_path(tmp.path()).await,
        Err(Error::CorruptMask(_))
    ));
}

// ── laziness ─────────────────────────────────────────────────────────

#[tokio::test]
async fn opening_reads_only_the_tail_of_the_container() {
    let tmp = tempfile::tempdir().unwrap();
    // Comfortably larger than the 64 KiB tail probe, so "read the tail" and
    // "read the file" are different outcomes.
    build_bulky_fixture(tmp.path(), 8, 4096).await;
    let container_len = std::fs::metadata(tmp.path().join("data.atlas"))
        .unwrap()
        .len();
    assert!(
        container_len > 256 * 1024,
        "fixture is too small to prove anything"
    );

    let inner: Arc<dyn ObjectStore> =
        Arc::new(object_store::local::LocalFileSystem::new_with_prefix(tmp.path()).unwrap());
    let counting = CountingStore::new(inner);

    let atlas = Atlas::open(counting.clone(), OsPath::default())
        .await
        .unwrap();
    // One tail read covering trailer and footer, plus a miss on the absent
    // mask.
    assert!(
        counting.gets() <= 2,
        "opening issued {} reads, expected at most 2",
        counting.gets()
    );
    assert!(
        counting.bytes() <= 64 * 1024,
        "opening read {} bytes of a {container_len}-byte container",
        counting.bytes(),
    );

    // Metadata questions cost nothing more.
    counting.reset();
    let _ = atlas.list_datasets();
    let _ = atlas.list_arrays();
    let ds = atlas.dataset("ds0").unwrap();
    let _ = ds.list_arrays();
    let _ = ds.array_meta("x");
    let _ = ds.attributes();
    let _ = ds.array_attributes("x");
    assert_eq!(
        counting.gets(),
        0,
        "metadata access should not touch the store"
    );
}

#[tokio::test]
async fn reading_one_dataset_touches_only_its_own_segment() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;

    let inner: Arc<dyn ObjectStore> =
        Arc::new(object_store::local::LocalFileSystem::new_with_prefix(tmp.path()).unwrap());
    let counting = CountingStore::new(inner);
    let atlas = Atlas::open(counting.clone(), OsPath::default())
        .await
        .unwrap();

    // Ranges the reader is allowed to touch: the tail, and jan's segment.
    let bare = Atlas::open_path(tmp.path()).await.unwrap();
    let jan_ordinal = bare.dataset("jan_2024").unwrap().ordinal();
    assert_eq!(jan_ordinal, 0);

    counting.reset();
    let ds = atlas.dataset("jan_2024").unwrap();
    let counts = ds
        .read_array::<i64>("counts", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(counts.as_slice().unwrap(), &[10, 20, 30, 40]);
    let first = counting.gets();
    assert!(first > 0, "a data read must fetch something");

    // The segment is already open, so a second array from the same dataset
    // costs strictly fewer reads than the first.
    counting.reset();
    let _ = ds
        .read_array::<f32>("temperature", vec![], vec![])
        .await
        .unwrap();
    assert!(
        counting.gets() < first,
        "reopening cost {} reads, first read cost {first}",
        counting.gets()
    );
}

// ── writer behaviour ─────────────────────────────────────────────────

#[tokio::test]
async fn a_collection_with_no_datasets_is_valid() {
    let tmp = tempfile::tempdir().unwrap();
    AtlasWriter::create_path(tmp.path(), WriterConfig::default())
        .await
        .unwrap()
        .finish()
        .await
        .unwrap();

    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    assert!(atlas.list_datasets().is_empty());
    assert!(atlas.list_arrays().is_empty());
    assert!(matches!(
        atlas.dataset("nope"),
        Err(Error::DatasetNotFound(_))
    ));
}

#[tokio::test]
async fn a_dataset_with_no_arrays_still_round_trips() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let w = AtlasWriter::create_path(tmp.path(), WriterConfig::default())
            .await
            .unwrap();
        let mut ds = w.add_dataset("empty").await.unwrap();
        ds.set_attribute("note", Attr::String("no arrays here".into()));
        ds.finish().await.unwrap();
        w.finish().await.unwrap();
    }
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    let ds = atlas.dataset("empty").unwrap();
    assert!(ds.list_arrays().is_empty());
    assert_eq!(
        ds.get_attribute("note"),
        Some(Attr::String("no arrays here".into()))
    );
}

#[tokio::test]
async fn a_dataset_dropped_without_finish_never_enters_the_container() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let w = AtlasWriter::create_path(tmp.path(), WriterConfig::default())
            .await
            .unwrap();
        {
            let mut ds = w.add_dataset("abandoned").await.unwrap();
            ds.define_array::<f32>("x", vec!["i".into()], vec![4], None, None)
                .await
                .unwrap();
            let data = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).into_dyn();
            ds.write_array("x", vec![0], data.view()).await.unwrap();
            // No finish: dropped here.
        }
        let mut ds = w.add_dataset("kept").await.unwrap();
        ds.define_array::<f32>("x", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
        ds.finish().await.unwrap();
        w.finish().await.unwrap();
    }
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    assert_eq!(atlas.list_datasets(), vec!["kept"]);
}

#[tokio::test]
async fn a_writer_dropped_without_finish_leaves_nothing_readable() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let w = AtlasWriter::create_path(tmp.path(), WriterConfig::default())
            .await
            .unwrap();
        let mut ds = w.add_dataset("d").await.unwrap();
        ds.define_array::<f32>("x", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
        ds.finish().await.unwrap();
        // No w.finish(): no trailer is ever written.
    }
    assert!(Atlas::open_path(tmp.path()).await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn datasets_staged_concurrently_land_intact() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let w = Arc::new(
            AtlasWriter::create_path(tmp.path(), WriterConfig::default())
                .await
                .unwrap(),
        );
        // Stage all four at once. They finish in whatever order they finish;
        // the writer's lock is what keeps their segments from interleaving.
        let mut tasks = Vec::new();
        for d in 0..4usize {
            let w = Arc::clone(&w);
            tasks.push(tokio::spawn(async move {
                let mut ds = w.add_dataset(&format!("ds{d}")).await.unwrap();
                ds.define_array::<i32>("x", vec!["i".into()], vec![256], Some(vec![64]), None)
                    .await
                    .unwrap();
                let data = Array1::from_shape_fn(256, |i| (d * 1000 + i) as i32).into_dyn();
                ds.write_array("x", vec![0], data.view()).await.unwrap();
                ds.finish().await.unwrap();
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
        Arc::try_unwrap(w)
            .unwrap_or_else(|_| panic!("all tasks finished"))
            .finish()
            .await
            .unwrap();
    }

    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    assert_eq!(atlas.dataset_count(), 4);
    for d in 0..4usize {
        let ds = atlas.dataset(&format!("ds{d}")).unwrap();
        let got = ds.read_array::<i32>("x", vec![], vec![]).await.unwrap();
        assert_eq!(
            got[[0]],
            (d * 1000) as i32,
            "ds{d} got another dataset's data"
        );
        assert_eq!(got[[255]], (d * 1000 + 255) as i32);
    }
    // Segments still tile the container without gaps or overlap.
    let mut ranges: Vec<_> = (0..4)
        .map(|d| atlas.dataset(&format!("ds{d}")).unwrap().segment_range())
        .collect();
    ranges.sort_by_key(|r| r.start);
    for pair in ranges.windows(2) {
        assert_eq!(pair[0].end, pair[1].start, "segments must be contiguous");
    }
}

#[tokio::test]
async fn duplicate_names_are_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let w = AtlasWriter::create_path(tmp.path(), WriterConfig::default())
        .await
        .unwrap();
    w.add_dataset("d").await.unwrap().finish().await.unwrap();
    assert!(matches!(
        w.add_dataset("d").await,
        Err(Error::DatasetAlreadyExists(_))
    ));

    let mut ds = w.add_dataset("e").await.unwrap();
    ds.define_array::<f32>("x", vec!["i".into()], vec![2], None, None)
        .await
        .unwrap();
    assert!(matches!(
        ds.define_array::<f32>("x", vec!["i".into()], vec![2], None, None)
            .await,
        Err(Error::ArrayAlreadyExists(_))
    ));
}

#[tokio::test]
async fn invalid_names_are_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let w = AtlasWriter::create_path(tmp.path(), WriterConfig::default())
        .await
        .unwrap();
    for bad in ["", "_hidden", "a/b", "..", "."] {
        assert!(
            matches!(w.add_dataset(bad).await, Err(Error::InvalidName(_))),
            "expected '{bad}' to be refused"
        );
    }
    let mut ds = w.add_dataset("ok").await.unwrap();
    assert!(matches!(
        ds.define_array::<f32>("_x", vec!["i".into()], vec![2], None, None)
            .await,
        Err(Error::InvalidName(_))
    ));
}

#[tokio::test]
async fn writing_an_undefined_array_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let w = AtlasWriter::create_path(tmp.path(), WriterConfig::default())
        .await
        .unwrap();
    let mut ds = w.add_dataset("d").await.unwrap();
    let data = Array1::from_vec(vec![1.0f32]).into_dyn();
    assert!(matches!(
        ds.write_array("nope", vec![0], data.view()).await,
        Err(Error::ArrayNotFound(_))
    ));
    assert!(matches!(
        ds.set_array_attribute("nope", "k", Attr::Int64(1)),
        Err(Error::ArrayNotFound(_))
    ));
}

#[tokio::test]
async fn every_codec_produces_a_readable_collection() {
    for codec in [Codec::Zstd, Codec::Lz4, Codec::Uncompressed] {
        let tmp = tempfile::tempdir().unwrap();
        {
            let w = AtlasWriter::create_path(
                tmp.path(),
                WriterConfig {
                    codec,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            let mut ds = w.add_dataset("d").await.unwrap();
            ds.define_array::<f64>("x", vec!["i".into()], vec![4], Some(vec![2]), None)
                .await
                .unwrap();
            let data = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]).into_dyn();
            ds.write_array("x", vec![0], data.view()).await.unwrap();
            ds.finish().await.unwrap();
            w.finish().await.unwrap();
        }
        // The reader is told nothing about the codec: blocks describe
        // themselves.
        let atlas = Atlas::open_path(tmp.path()).await.unwrap();
        let got = atlas
            .dataset("d")
            .unwrap()
            .read_array::<f64>("x", vec![], vec![])
            .await
            .unwrap();
        assert_eq!(
            got.as_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0],
            "codec {codec:?}"
        );
    }
}

#[tokio::test]
async fn a_dataset_larger_than_one_block_round_trips() {
    let tmp = tempfile::tempdir().unwrap();
    // A small block target forces several blocks, exercising the streaming
    // copy of a staged segment into the container.
    let rows = 64;
    let cols = 256;
    {
        let w = AtlasWriter::create_path(
            tmp.path(),
            WriterConfig {
                codec: Codec::Uncompressed,
                block_target_size: 16 * 1024,
            },
        )
        .await
        .unwrap();
        let mut ds = w.add_dataset("big").await.unwrap();
        ds.define_array::<f64>(
            "x",
            vec!["r".into(), "c".into()],
            vec![rows, cols],
            Some(vec![8, 64]),
            None,
        )
        .await
        .unwrap();
        let data = ArrayD::from_shape_fn(ndarray::IxDyn(&[rows, cols]), |i| {
            (i[0] * cols + i[1]) as f64
        });
        ds.write_array("x", vec![0, 0], data.view()).await.unwrap();
        ds.finish().await.unwrap();
        w.finish().await.unwrap();
    }
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    let ds = atlas.dataset("big").unwrap();
    let got = ds.read_array::<f64>("x", vec![], vec![]).await.unwrap();
    assert_eq!(got[[0, 0]], 0.0);
    assert_eq!(
        got[[rows - 1, cols - 1]],
        ((rows - 1) * cols + cols - 1) as f64
    );
    // And a window from the middle, which must not fetch the whole array.
    let window = ds
        .read_array::<f64>("x", vec![30, 100], vec![2, 2])
        .await
        .unwrap();
    assert_eq!(window[[0, 0]], (30 * cols + 100) as f64);
}

#[tokio::test]
async fn many_slabs_into_one_array_assemble_correctly() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let w = AtlasWriter::create_path(tmp.path(), WriterConfig::default())
            .await
            .unwrap();
        let mut ds = w.add_dataset("d").await.unwrap();
        ds.define_array::<i32>("x", vec!["i".into()], vec![8], Some(vec![3]), None)
            .await
            .unwrap();
        // Slabs that are neither chunk-aligned nor the same size.
        for (start, values) in [
            (0usize, vec![0i32, 1]),
            (2, vec![2, 3, 4, 5]),
            (6, vec![6, 7]),
        ] {
            let block = Array1::from_vec(values).into_dyn();
            ds.write_array("x", vec![start], block.view())
                .await
                .unwrap();
        }
        ds.finish().await.unwrap();
        w.finish().await.unwrap();
    }
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    let got = atlas
        .dataset("d")
        .unwrap()
        .read_array::<i32>("x", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(got.as_slice().unwrap(), &[0, 1, 2, 3, 4, 5, 6, 7]);
}

// ── object store backends ────────────────────────────────────────────

#[tokio::test]
async fn a_collection_round_trips_on_an_in_memory_store_under_a_prefix() {
    let store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
    let prefix = OsPath::from("collections/2024");
    {
        let w = AtlasWriter::create(Arc::clone(&store), prefix.clone(), WriterConfig::default())
            .await
            .unwrap();
        let mut ds = w.add_dataset("d").await.unwrap();
        ds.define_array::<f32>("x", vec!["i".into()], vec![3], None, None)
            .await
            .unwrap();
        let data = Array1::from_vec(vec![1.0f32, 2.0, 3.0]).into_dyn();
        ds.write_array("x", vec![0], data.view()).await.unwrap();
        ds.finish().await.unwrap();
        w.finish().await.unwrap();
    }

    // The objects landed under the prefix, not at the root.
    assert!(
        store
            .head(&OsPath::from("collections/2024/data.atlas"))
            .await
            .is_ok()
    );

    let atlas = Atlas::open(Arc::clone(&store), prefix.clone())
        .await
        .unwrap();
    let got = atlas
        .dataset("d")
        .unwrap()
        .read_array::<f32>("x", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(got.as_slice().unwrap(), &[1.0, 2.0, 3.0]);

    atlas.delete_dataset("d").await.unwrap();
    assert!(
        store
            .head(&OsPath::from("collections/2024/deleted.mask"))
            .await
            .is_ok()
    );
    assert!(
        Atlas::open(store, prefix)
            .await
            .unwrap()
            .list_datasets()
            .is_empty()
    );
}

// ── rejecting things that are not collections ────────────────────────

#[tokio::test]
async fn an_empty_directory_is_not_a_collection() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(matches!(
        Atlas::open_path(tmp.path()).await,
        Err(Error::NotAnAtlasCollection { .. })
    ));
}

#[tokio::test]
async fn an_atlas_014_store_says_so() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("atlas.json"), br#"{"version":3}"#).unwrap();
    match Atlas::open_path(tmp.path()).await {
        Err(Error::NotAnAtlasCollection { hint }) => {
            assert!(hint.contains("0.14"), "unhelpful hint: {hint}");
        }
        other => panic!("expected a hinted rejection, got {other:?}"),
    }
}

#[tokio::test]
async fn a_foreign_file_named_data_atlas_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("data.atlas"), vec![0u8; 4096]).unwrap();
    assert!(matches!(
        Atlas::open_path(tmp.path()).await,
        Err(Error::NotAnAtlasCollection { .. })
    ));
}

#[tokio::test]
async fn a_truncated_container_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let path = tmp.path().join("data.atlas");
    let full = std::fs::read(&path).unwrap();

    // Losing the trailer loses the magic.
    std::fs::write(&path, &full[..full.len() - 4]).unwrap();
    assert!(matches!(
        Atlas::open_path(tmp.path()).await,
        Err(Error::NotAnAtlasCollection { .. })
    ));

    // A trailer that survives while the footer it points at does not.
    let mut cut = full[..full.len() / 2].to_vec();
    cut.extend_from_slice(&full[full.len() - 16..]);
    std::fs::write(&path, &cut).unwrap();
    assert!(Atlas::open_path(tmp.path()).await.is_err());
}

#[tokio::test]
async fn a_container_from_a_future_version_is_rejected_clearly() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let path = tmp.path().join("data.atlas");
    let mut bytes = std::fs::read(&path).unwrap();
    let len = bytes.len();
    // The version sits between the footer size and the trailing magic.
    bytes[len - 8..len - 4].copy_from_slice(&99u32.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    assert!(matches!(
        Atlas::open_path(tmp.path()).await,
        Err(Error::UnsupportedVersion {
            found: 99,
            expected: 1
        })
    ));
}

// ── framing ──────────────────────────────────────────────────────────

#[tokio::test]
async fn the_container_carries_the_documented_framing() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let bytes = std::fs::read(tmp.path().join("data.atlas")).unwrap();

    // Header: magic then version.
    assert_eq!(&bytes[0..4], b"ATLS");
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 1);

    // Trailer: footer size, version, magic.
    let len = bytes.len();
    assert_eq!(&bytes[len - 4..], b"ATLS");
    assert_eq!(
        u32::from_le_bytes(bytes[len - 8..len - 4].try_into().unwrap()),
        1
    );
    let footer_size = u64::from_le_bytes(bytes[len - 16..len - 8].try_into().unwrap()) as usize;
    assert!(footer_size > 0 && footer_size < len - 24);

    // Segments are packed back to back, starting right after the header, and
    // the first one ends where the second begins.
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    let first = atlas.dataset("jan_2024").unwrap().segment_range();
    let second = atlas.dataset("feb_2024").unwrap().segment_range();
    assert_eq!(first.start, 8);
    assert_eq!(first.end, second.start);
    // array-format is footer-addressed, so each segment ends in its own magic.
    assert_eq!(&bytes[first.end as usize - 4..first.end as usize], b"ARRF");
    // The last segment ends where the container footer starts.
    let last = atlas.dataset("stations").unwrap().segment_range();
    assert_eq!(last.end as usize, len - 16 - footer_size);
}

#[tokio::test]
async fn a_segment_cut_out_of_the_container_opens_on_its_own() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let bytes = std::fs::read(tmp.path().join("data.atlas")).unwrap();

    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    let jan = atlas.dataset("jan_2024").unwrap();
    let expected = jan
        .read_array::<i64>("counts", vec![], vec![])
        .await
        .unwrap();

    // Carve the segment out the way `dd` would, and hand it to array-format
    // with no atlas involved.
    let range = jan.segment_range();
    let carved = &bytes[range.start as usize..range.end as usize];

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("data.af"), carved).unwrap();
    let store: Arc<dyn ObjectStore> =
        Arc::new(object_store::local::LocalFileSystem::new_with_prefix(dir.path()).unwrap());
    let file = array_format::ArrayFile::open(
        store,
        OsPath::from("data.af"),
        array_format::FileConfig::new(array_format::NoCompression),
    )
    .await
    .unwrap();
    let direct = file
        .read_array::<i64>("counts", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(direct.as_slice().unwrap(), expected.as_slice().unwrap());
}

// ── type safety ──────────────────────────────────────────────────────

#[tokio::test]
async fn reading_an_array_at_the_wrong_type_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    let jan = atlas.dataset("jan_2024").unwrap();

    // temperature is f32.
    assert!(
        jan.read_array::<f64>("temperature", vec![], vec![])
            .await
            .is_err()
    );
    assert!(matches!(
        jan.read_array::<f32>("missing", vec![], vec![]).await,
        Err(Error::ArrayNotFound(_))
    ));
}
