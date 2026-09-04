//! End-to-end tests for the single-file immutable format.
//!
//! The lifecycle under test is short by design. Build a collection, finish
//! it, open it, read from it, delete a dataset, reopen. There is no mutation
//! path to test, because the format has none.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use atlas::{Atlas, AtlasWriter, Attr, Codec, Error, FillValue, StatValue, WriterConfig};
use ndarray::{Array1, Array2, ArrayD};
use object_store::path::Path as OsPath;
use object_store::{ObjectStore, ObjectStoreExt};

// ── helpers ──────────────────────────────────────────────────────────

/// Builds a collection of three datasets. They cover the cases that matter.
/// Several dtypes, a chunked array, an array nobody writes, fill values, and
/// attributes at both levels.
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
        // Four chunks. One slab spans all of them.
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
        // The same arrays as jan_2024, but one attribute fewer. The schema
        // names the attribute keys, so this is a second schema.
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
        // counts is declared and never written. It must read back as fill.
        ds.set_attribute("month", Attr::Int64(2));
        ds.finish().await.unwrap();
    }

    {
        // A different schema, and the dtypes the xarray layer needs.
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
        // An attribute has no timestamp type, so the epoch nanoseconds go in
        // as an integer and a second key names the unit.
        ds.set_attribute("installed", Attr::Int64(1_600_000_000_000_000_000));
        ds.set_attribute("installed_units", Attr::String("ns since epoch".into()));
        ds.finish().await.unwrap();
    }

    w.finish().await.unwrap();
}

/// Builds `datasets` datasets. Each holds one array of `len` elements, with no
/// compression, so the container is truly large. Use it when a fixture must
/// exceed the reader's tail probe.
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

/// An `ObjectStore` that counts requests. It turns an assumption about lazy
/// reads into an assertion.
#[derive(Debug)]
struct CountingStore {
    inner: Arc<dyn ObjectStore>,
    gets: AtomicUsize,
    puts: AtomicUsize,
    bytes: AtomicUsize,
}

impl CountingStore {
    fn new(inner: Arc<dyn ObjectStore>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            gets: AtomicUsize::new(0),
            puts: AtomicUsize::new(0),
            bytes: AtomicUsize::new(0),
        })
    }
    fn gets(&self) -> usize {
        self.gets.load(Ordering::SeqCst)
    }
    fn puts(&self) -> usize {
        self.puts.load(Ordering::SeqCst)
    }
    fn bytes(&self) -> usize {
        self.bytes.load(Ordering::SeqCst)
    }
    fn reset(&self) {
        self.gets.store(0, Ordering::SeqCst);
        self.puts.store(0, Ordering::SeqCst);
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
        self.puts.fetch_add(1, Ordering::SeqCst);
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
        // A head request moves no body, so it counts as neither.
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
    assert_eq!(*meta.dtype(), atlas::DType::Float32);
    let layout = jan.array_layout("temperature").await.unwrap();
    assert_eq!(layout.shape(), vec![4, 8]);
    assert_eq!(layout.chunk_shape(), vec![2, 4]);
    assert_eq!(layout.dimension_names(), vec!["lat", "lon"]);
    assert!(matches!(
        layout.fill_value(),
        Some(FillValue::Float(f)) if f.is_nan()
    ));

    assert_eq!(
        jan.get_attribute("month").await.unwrap(),
        Some(Attr::Int64(1))
    );
    assert_eq!(
        jan.get_attribute("source").await.unwrap(),
        Some(Attr::String("buoy".into()))
    );
    assert_eq!(jan.attributes().await.unwrap().len(), 2);
    assert_eq!(
        jan.get_array_attribute("temperature", "units")
            .await
            .unwrap(),
        Some(Attr::String("celsius".into()))
    );
    // counts has no attributes of its own.
    assert!(jan.array_attributes("counts").await.unwrap().is_empty());
}

#[tokio::test]
async fn collection_stats_fold_every_dataset_that_holds_the_array() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();

    // jan_2024 holds 0.0..=31.0 and feb_2024 holds -1.5, over 32 elements each.
    let temperature = atlas.array_stats("temperature").await.unwrap().unwrap();
    assert_eq!(temperature.name, "temperature");
    assert_eq!(temperature.min, Some(StatValue::Float(-1.5)));
    assert_eq!(temperature.max, Some(StatValue::Float(31.0)));
    assert_eq!(temperature.row_count, 64);
    assert_eq!(temperature.null_count, 0);

    // feb_2024 declares counts and never writes it. It therefore adds
    // elements and nulls, but no bounds. The bounds come from jan_2024 alone.
    let counts = atlas.array_stats("counts").await.unwrap().unwrap();
    assert_eq!(counts.min, Some(StatValue::Int(10)));
    assert_eq!(counts.max, Some(StatValue::Int(40)));
    assert_eq!(counts.row_count, 8);
    assert_eq!(counts.null_count, 4);

    // One dataset holds name, so the result matches that dataset alone.
    let stations = atlas.dataset("stations").unwrap();
    assert_eq!(
        atlas.array_stats("name").await.unwrap(),
        stations.array_stats("name").await.unwrap()
    );

    // No dataset declares this name.
    assert!(atlas.array_stats("missing").await.unwrap().is_none());
}

#[tokio::test]
async fn one_attribute_over_the_collection_keys_on_the_dataset_name() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();

    let months = atlas.attributes_by_dataset(None, "month").await.unwrap();
    // stations carries no month, so it has no entry at all.
    assert_eq!(months.len(), 2);
    assert_eq!(months["jan_2024"], Attr::Int64(1));
    assert_eq!(months["feb_2024"], Attr::Int64(2));
    assert!(!months.contains_key("stations"));

    // The map holds the datasets in write order, as `list_datasets` does.
    assert_eq!(
        months.keys().collect::<Vec<_>>(),
        vec!["jan_2024", "feb_2024"]
    );

    // Name for name, it equals what each dataset reports on its own.
    for name in atlas.list_datasets() {
        let own = atlas
            .dataset(&name)
            .unwrap()
            .get_attribute("month")
            .await
            .unwrap();
        assert_eq!(months.get(&name), own.as_ref(), "{name}");
    }

    // A key no dataset carries gives an empty map.
    assert!(
        atlas
            .attributes_by_dataset(None, "nope")
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn one_array_attribute_over_the_collection_keys_on_the_dataset_name() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();

    // jan_2024 is the only dataset that annotates temperature.
    let units = atlas
        .attributes_by_dataset(Some("temperature"), "units")
        .await
        .unwrap();
    assert_eq!(units.len(), 1);
    assert_eq!(units["jan_2024"], Attr::String("celsius".into()));

    // A key the array does not carry, and an array no dataset declares.
    assert!(
        atlas
            .attributes_by_dataset(Some("temperature"), "nope")
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        atlas
            .attributes_by_dataset(Some("missing"), "units")
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn the_mask_hides_a_dataset_from_the_attribute_map() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    assert_eq!(
        atlas
            .attributes_by_dataset(None, "month")
            .await
            .unwrap()
            .len(),
        2
    );

    atlas.delete_dataset("jan_2024").await.unwrap();
    let months = atlas.attributes_by_dataset(None, "month").await.unwrap();

    assert_eq!(atlas.list_datasets(), vec!["feb_2024", "stations"]);
    assert_eq!(months.len(), 1);
    assert_eq!(months.keys().collect::<Vec<_>>(), vec!["feb_2024"]);
    assert_eq!(months["feb_2024"], Attr::Int64(2));
    assert!(!months.contains_key("jan_2024"));
}

#[tokio::test]
async fn attributes_and_stats_join_into_one_table() {
    // The shape a Parquet row group index has: one row per dataset, with the
    // attributes beside the bounds. Every column keys on the dataset name, so
    // the join is a lookup and never an alignment by position. Every column
    // also keeps write order, so a walk of one gives the rows in order.
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();

    let months = atlas.attributes_by_dataset(None, "month").await.unwrap();
    let sources = atlas.attributes_by_dataset(None, "source").await.unwrap();
    let temperature: std::collections::HashMap<String, atlas::ArrayStats> = atlas
        .array_stats_by_dataset("temperature")
        .await
        .unwrap()
        .into_iter()
        .map(|stats| (stats.name.clone(), stats))
        .collect();

    let table: Vec<_> = atlas
        .list_datasets()
        .into_iter()
        .map(|name| {
            let bounds = temperature
                .get(&name)
                .map(|s| (s.min.clone(), s.max.clone()));
            let row = (
                months.get(&name).cloned(),
                sources.get(&name).cloned(),
                bounds,
            );
            (name, row)
        })
        .collect();

    assert_eq!(table.len(), 3);
    assert_eq!(table[0].0, "jan_2024");
    assert_eq!(table[0].1.0, Some(Attr::Int64(1)));
    assert_eq!(table[0].1.1, Some(Attr::String("buoy".into())));
    assert_eq!(
        table[0].1.2,
        Some((Some(StatValue::Float(0.0)), Some(StatValue::Float(31.0))))
    );
    // stations holds neither the attributes nor the array.
    assert_eq!(table[2].0, "stations");
    assert_eq!(table[2].1.0, None);
    assert_eq!(table[2].1.2, None);
}

#[tokio::test]
async fn per_dataset_stats_name_their_dataset_and_keep_write_order() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();

    let per = atlas.array_stats_by_dataset("temperature").await.unwrap();
    // Every row names its own dataset, so a row identifies itself.
    let names: Vec<&str> = per.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["jan_2024", "feb_2024"]);
    assert_eq!(per[0].max, Some(StatValue::Float(31.0)));
    assert_eq!(per[1].max, Some(StatValue::Float(-1.5)));

    // Each row holds the numbers that dataset reports on its own. Only the
    // name differs: a view names the array, and a row names the dataset.
    for stats in &per {
        let own = atlas
            .dataset(&stats.name)
            .unwrap()
            .array_stats("temperature")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(own.name, "temperature");
        assert_eq!((&own.min, &own.max), (&stats.min, &stats.max));
        assert_eq!(own.row_count, stats.row_count);
        assert_eq!(own.null_count, stats.null_count);
    }

    // stations is the only dataset that declares name.
    let per_name = atlas.array_stats_by_dataset("name").await.unwrap();
    assert_eq!(per_name.len(), 1);
    assert_eq!(per_name[0].name, "stations");

    // A name no dataset declares gives an empty list.
    assert!(
        atlas
            .array_stats_by_dataset("missing")
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn the_mask_hides_a_dataset_from_the_per_dataset_stats() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    assert_eq!(
        atlas
            .array_stats_by_dataset("temperature")
            .await
            .unwrap()
            .len(),
        2
    );

    atlas.delete_dataset("jan_2024").await.unwrap();
    let per = atlas.array_stats_by_dataset("temperature").await.unwrap();
    assert_eq!(per.len(), 1);
    assert_eq!(per[0].name, "feb_2024");

    // The mask on disk hides it for a fresh handle too.
    let reopened = Atlas::open_path(tmp.path()).await.unwrap();
    assert_eq!(
        reopened
            .array_stats_by_dataset("temperature")
            .await
            .unwrap()
            .len(),
        1
    );

    // Hide the last holder, and nothing remains to report.
    atlas.delete_dataset("feb_2024").await.unwrap();
    assert!(
        atlas
            .array_stats_by_dataset("temperature")
            .await
            .unwrap()
            .is_empty()
    );
    assert!(atlas.array_stats("temperature").await.unwrap().is_none());
}

#[tokio::test]
async fn a_deleted_dataset_drops_out_of_the_collection_stats() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    atlas.delete_dataset("feb_2024").await.unwrap();

    // Only jan_2024 remains to contribute.
    let temperature = atlas.array_stats("temperature").await.unwrap().unwrap();
    assert_eq!(temperature.min, Some(StatValue::Float(0.0)));
    assert_eq!(temperature.max, Some(StatValue::Float(31.0)));
    assert_eq!(temperature.row_count, 32);

    let jan = atlas.dataset("jan_2024").unwrap();
    assert_eq!(
        atlas.array_stats("counts").await.unwrap(),
        jan.array_stats("counts").await.unwrap()
    );

    // Delete the only dataset that holds an array, and nothing remains.
    atlas.delete_dataset("stations").await.unwrap();
    assert!(atlas.array_stats("name").await.unwrap().is_none());
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

    // Chunks are 2x4, so 1..3 x 3..5 covers part of all four.
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
    // Declared, never written, and with no explicit fill. An integer gives
    // zero.
    let counts = feb
        .read_array::<i64>("counts", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(counts.shape(), &[4]);
    assert_eq!(counts.as_slice().unwrap(), &[0, 0, 0, 0]);
}

#[tokio::test]
async fn the_schema_names_the_attribute_keys_too() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();

    // jan and feb declare the same arrays. jan carries `source` and feb does
    // not, and the schema names the keys, so the two do not share an entry.
    let jan = atlas.dataset("jan_2024").unwrap();
    let feb = atlas.dataset("feb_2024").unwrap();
    assert_eq!(
        jan.schema().names().collect::<Vec<_>>(),
        feb.schema().names().collect::<Vec<_>>()
    );
    assert_ne!(jan.schema(), feb.schema());
    assert_ne!(
        jan.attributes().await.unwrap(),
        feb.attributes().await.unwrap()
    );
    // The values differ, and both resolve against their own schema.
    assert_eq!(
        jan.get_attribute("month").await.unwrap(),
        Some(Attr::Int64(1))
    );
    assert_eq!(
        feb.get_attribute("month").await.unwrap(),
        Some(Attr::Int64(2))
    );
    assert_eq!(feb.get_attribute("source").await.unwrap(), None);
}

#[tokio::test]
async fn the_schema_view_reads_every_array_from_the_footer() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    let jan = atlas.dataset("jan_2024").unwrap();
    let schema = jan.schema();

    assert_eq!(schema.len(), 2);
    assert!(!schema.is_empty());
    assert_eq!(
        schema.names().collect::<Vec<_>>(),
        vec!["temperature", "counts"]
    );
    assert_eq!(schema.index_of("counts"), Some(1));
    assert_eq!(schema.index_of("missing"), None);

    // By name and by position give the same array.
    let by_name = schema.get("temperature").unwrap();
    let by_index = schema.get_index(0).unwrap();
    assert_eq!(by_name, by_index);
    assert_eq!(by_name.name(), "temperature");
    assert_eq!(*by_name.dtype(), atlas::DType::Float32);
    assert_eq!(by_name.position(), 0);
    assert!(schema.get_index(2).is_none());

    // Attribute keys and their types come from the footer too.
    assert_eq!(
        schema.attribute_names().collect::<Vec<_>>(),
        vec!["month", "source"]
    );
    assert_eq!(schema.attribute_dtype("month"), Some(&atlas::DType::Int64));
    assert_eq!(schema.attribute_dtype("missing"), None);
    assert_eq!(by_name.attribute_names(), vec!["units"]);
    assert_eq!(
        by_name.attribute_dtype("units"),
        Some(&atlas::DType::String)
    );

    // The owned copy holds the same names and types.
    let owned = schema.to_owned_schema();
    assert_eq!(
        owned.arrays.keys().collect::<Vec<_>>(),
        vec!["temperature", "counts"]
    );
    assert_eq!(owned.arrays["temperature"], atlas::DType::Float32);
    assert_eq!(owned.attributes["month"], atlas::DType::Int64);
    assert_eq!(
        owned.array_attributes["temperature"]["units"],
        atlas::DType::String
    );

    // Shape and chunking come from the segment, not the footer.
    let layout = jan.array_layout("temperature").await.unwrap();
    assert_eq!(layout.shape(), vec![4, 8]);
    assert_eq!(layout.chunk_shape(), vec![2, 4]);
    assert_eq!(layout.dimension_names(), vec!["lat", "lon"]);
    assert_eq!(layout.element_count(), 32);
    // A name this dataset does not declare has no layout to read.
    assert!(matches!(
        jan.array_layout("missing").await,
        Err(Error::ArrayNotFound(_))
    ));
}

#[tokio::test]
async fn datasets_that_differ_only_in_shape_share_one_schema() {
    // The shape is what varies across a directory of files, and it is not in
    // the schema. The segment records it. A thousand files of one convention
    // therefore intern to one entry, whatever their lengths.
    let tmp = tempfile::tempdir().unwrap();
    {
        let w = AtlasWriter::create_path(tmp.path(), WriterConfig::default())
            .await
            .unwrap();
        for (name, len) in [("short", 2usize), ("long", 3usize)] {
            let mut ds = w.add_dataset(name).await.unwrap();
            ds.define_array::<f32>("temperature", vec!["time".into()], vec![len], None, None)
                .await
                .unwrap();
            let data = Array1::from_vec(vec![1.5f32; len]).into_dyn();
            ds.write_array("temperature", vec![0], data.view())
                .await
                .unwrap();
            ds.set_attribute("site", Attr::String(name.into()));
            ds.finish().await.unwrap();
        }
        w.finish().await.unwrap();
    }
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();

    assert_eq!(atlas.total_datasets(), 2);
    assert_eq!(atlas.interned_schemas(), 1);
    assert_eq!(atlas.list_arrays(), vec!["temperature"]);

    let short = atlas.dataset("short").unwrap();
    let long = atlas.dataset("long").unwrap();
    assert_eq!(short.schema(), long.schema());
    // The attribute values still differ. Only the keys are in the schema.
    assert_eq!(
        short.get_attribute("site").await.unwrap(),
        Some(Attr::String("short".into()))
    );
    assert_eq!(
        long.get_attribute("site").await.unwrap(),
        Some(Attr::String("long".into()))
    );

    // The lengths differ, and both come from the one segment.
    let short_layout = short.array_layout("temperature").await.unwrap();
    let long_layout = long.array_layout("temperature").await.unwrap();
    assert_eq!(short_layout.shape(), vec![2]);
    assert_eq!(long_layout.shape(), vec![3]);
    assert_eq!(
        short_layout.dimension_names(),
        long_layout.dimension_names()
    );
    assert_eq!(
        short
            .read_array::<f32>("temperature", vec![], vec![])
            .await
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        long.read_array::<f32>("temperature", vec![], vec![])
            .await
            .unwrap()
            .len(),
        3
    );
}

#[tokio::test]
async fn one_array_resolves_to_its_own_position_in_each_schema() {
    // Two datasets declare the same arrays in opposite order. The footer
    // records a position, not a name, so a collection-wide call must map the
    // name once per schema.
    let tmp = tempfile::tempdir().unwrap();
    {
        let w = AtlasWriter::create_path(tmp.path(), WriterConfig::default())
            .await
            .unwrap();
        for (dataset, order, base) in [
            ("a", ["temperature", "salinity"], 1.0f32),
            ("b", ["salinity", "temperature"], 10.0f32),
        ] {
            let mut ds = w.add_dataset(dataset).await.unwrap();
            for (i, array) in order.iter().enumerate() {
                ds.define_array::<f32>(array, vec!["time".into()], vec![2], None, None)
                    .await
                    .unwrap();
                let offset = base + (i * 100) as f32;
                let data = Array1::from_vec(vec![offset, offset + 1.0]).into_dyn();
                ds.write_array(array, vec![0], data.view()).await.unwrap();
            }
            ds.finish().await.unwrap();
        }
        w.finish().await.unwrap();
    }
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();

    // The two orders make two schemas, and temperature sits elsewhere in each.
    assert_eq!(atlas.interned_schemas(), 2);
    let a = atlas.dataset("a").unwrap();
    let b = atlas.dataset("b").unwrap();
    assert_eq!(a.schema().index_of("temperature"), Some(0));
    assert_eq!(b.schema().index_of("temperature"), Some(1));

    // a holds 1.0..=2.0 and b holds 110.0..=111.0.
    let per = atlas.array_stats_by_dataset("temperature").await.unwrap();
    assert_eq!(per.len(), 2);
    assert_eq!(per[0].name, "a");
    assert_eq!(per[0].min, Some(StatValue::Float(1.0)));
    assert_eq!(per[1].name, "b");
    assert_eq!(per[1].min, Some(StatValue::Float(110.0)));

    let merged = atlas.array_stats("temperature").await.unwrap().unwrap();
    assert_eq!(merged.min, Some(StatValue::Float(1.0)));
    assert_eq!(merged.max, Some(StatValue::Float(111.0)));
    assert_eq!(merged.row_count, 4);
}

#[tokio::test]
async fn an_attribute_reads_back_the_type_it_stores() {
    // A read rebuilds the value from its stored tag alone, and never from the
    // schema. Every type therefore returns itself. There is no timestamp
    // attribute to lose, and a string that looks like a date is still a
    // string.
    let tmp = tempfile::tempdir().unwrap();
    {
        let w = AtlasWriter::create_path(tmp.path(), WriterConfig::default())
            .await
            .unwrap();
        let mut ds = w.add_dataset("d").await.unwrap();
        ds.set_attribute("when", Attr::Int64(1_700_000_000_000_000_000));
        ds.set_attribute("count", Attr::Int32(7));
        ds.set_attribute("looks_like", Attr::String("2023-11-14T22:13:20Z".into()));
        ds.finish().await.unwrap();
        w.finish().await.unwrap();
    }
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    let ds = atlas.dataset("d").unwrap();
    assert_eq!(
        ds.get_attribute("when").await.unwrap(),
        Some(Attr::Int64(1_700_000_000_000_000_000))
    );
    // A narrower integer keeps its width, because the stored form carries it.
    assert_eq!(
        ds.get_attribute("count").await.unwrap(),
        Some(Attr::Int32(7))
    );
    assert_eq!(
        ds.get_attribute("looks_like").await.unwrap(),
        Some(Attr::String("2023-11-14T22:13:20Z".into()))
    );

    // The schema records the same type the value carries, because there is
    // now nothing an attribute can declare that its value cannot hold.
    assert_eq!(
        ds.schema().attribute_dtype("when"),
        Some(&atlas::DType::Int64)
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

    // Visible at once on the handle that deleted it.
    assert_eq!(atlas.list_datasets(), vec!["jan_2024", "stations"]);
    assert!(!atlas.dataset_exists("feb_2024"));
    assert!(matches!(
        atlas.dataset("feb_2024"),
        Err(Error::DatasetNotFound(_))
    ));

    // And after a reopen.
    let reopened = Atlas::open_path(tmp.path()).await.unwrap();
    assert_eq!(reopened.list_datasets(), vec!["jan_2024", "stations"]);
    // The arrays of the deleted dataset leave the union too.
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
    // Still 2. Nothing renumbers, so a stored ordinal stays valid.
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
async fn deleting_many_datasets_costs_one_mask_write() {
    let tmp = tempfile::tempdir().unwrap();
    build_bulky_fixture(tmp.path(), 40, 64).await;

    let store = CountingStore::new(Arc::new(
        object_store::local::LocalFileSystem::new_with_prefix(tmp.path()).unwrap(),
    ));
    let atlas = Atlas::open(
        Arc::clone(&store) as Arc<dyn ObjectStore>,
        OsPath::default(),
    )
    .await
    .unwrap();
    store.reset();

    let names: Vec<String> = (0..30).map(|d| format!("ds{d}")).collect();
    assert_eq!(atlas.delete_datasets(&names).await.unwrap(), 30);

    // One read of the mask, and one write of it. Thirty names cost what one
    // name costs, which is the point of the call.
    assert_eq!(store.gets(), 1, "one mask read");
    assert_eq!(store.puts(), 1, "one mask write");

    let reopened = Atlas::open_path(tmp.path()).await.unwrap();
    assert_eq!(reopened.dataset_count(), 10);
    assert!(!reopened.dataset_exists("ds0"));
    assert!(reopened.dataset_exists("ds30"));
}

#[tokio::test]
async fn a_batch_delete_with_an_unknown_name_writes_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();

    assert!(matches!(
        atlas.delete_datasets(["jan_2024", "nope"]).await,
        Err(Error::DatasetNotFound(_))
    ));

    // The whole batch stands or falls together, so jan_2024 survives.
    assert_eq!(atlas.dataset_count(), 3);
    assert!(!tmp.path().join("deleted.mask").exists());
}

#[tokio::test]
async fn a_repeated_name_in_a_batch_counts_once() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();

    let hidden = atlas
        .delete_datasets(["jan_2024", "jan_2024", "stations"])
        .await
        .unwrap();
    assert_eq!(hidden, 2);
    assert_eq!(atlas.list_datasets(), vec!["feb_2024"]);
}

#[tokio::test]
async fn an_empty_batch_delete_does_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();

    assert_eq!(
        atlas.delete_datasets(Vec::<String>::new()).await.unwrap(),
        0
    );
    assert_eq!(atlas.dataset_count(), 3);
    assert!(!tmp.path().join("deleted.mask").exists());
}

#[tokio::test]
async fn deletions_accumulate_in_one_mask() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;

    // Two separate handles. The second must merge with what the first wrote,
    // and must not overwrite it.
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
    // A mask from a larger collection. Magic, version, count, then ordinals 0
    // and 99.
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
    // Well past the 64 KiB tail probe. "Read the tail" and "read the file"
    // then give different results.
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
    // One tail read covers the trailer and the footer. One more miss on the
    // absent mask.
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

    // The footer holds names and types. Those cost nothing more.
    counting.reset();
    let _ = atlas.list_datasets();
    let _ = atlas.list_arrays();
    let _ = atlas.dataset_count();
    let _ = atlas.interned_schemas();
    let ds = atlas.dataset("ds0").unwrap();
    let _ = ds.name();
    let _ = ds.ordinal();
    let _ = ds.list_arrays();
    let _ = ds.schema().attribute_names().count();
    let _ = ds.array_meta("x").unwrap().dtype();
    assert_eq!(counting.gets(), 0, "the footer must answer these alone");

    // Nothing else is in the footer. A statistic, an attribute value, and a
    // layout all live on the array they belong to, so each reads a segment.
    // `build_bulky_fixture` gives each dataset an `index` attribute, which
    // sits in the reserved `_datasets` segment.
    counting.reset();
    let stats = ds.array_stats("x").await.unwrap().unwrap();
    assert_eq!(stats.name, "x");
    assert!(stats.row_count > 0, "the segment recorded a row count");
    let opened = counting.gets();
    assert!(opened > 0, "a statistic comes from a segment");

    // That segment is open now, so everything else it holds is free.
    counting.reset();
    let _ = atlas.array_stats("x").await.unwrap();
    let _ = atlas.array_stats_by_dataset("x").await.unwrap();
    let _ = ds.array_layout("x").await.unwrap();
    let other = atlas.dataset("ds1").unwrap();
    let _ = other.array_stats("x").await.unwrap();
    assert_eq!(counting.gets(), 0, "the handle is cached");

    // The attribute values are in another segment, so that one opens once too.
    counting.reset();
    assert_eq!(ds.attributes().await.unwrap().len(), 1);
    let attrs_opened = counting.gets();
    assert!(attrs_opened > 0, "an attribute value comes from a segment");
    counting.reset();
    assert_eq!(other.attributes().await.unwrap().len(), 1);
    assert!(
        counting.gets() < attrs_opened,
        "the second dataset cost {} reads, the first {attrs_opened}",
        counting.gets()
    );

    // An array with no attribute reads nothing, because the schema says so.
    counting.reset();
    assert!(ds.array_attributes("x").await.unwrap().is_empty());
    assert_eq!(counting.gets(), 0, "an empty key list needs no segment");
}

#[tokio::test]
async fn one_variable_segment_opens_once_for_every_dataset() {
    let tmp = tempfile::tempdir().unwrap();
    build_fixture(tmp.path()).await;

    let inner: Arc<dyn ObjectStore> =
        Arc::new(object_store::local::LocalFileSystem::new_with_prefix(tmp.path()).unwrap());
    let counting = CountingStore::new(inner);
    let atlas = Atlas::open(counting.clone(), OsPath::default())
        .await
        .unwrap();

    counting.reset();
    let jan = atlas.dataset("jan_2024").unwrap();
    let first_read = jan
        .read_array::<f32>("temperature", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(first_read.shape(), &[4, 8]);
    let first = counting.gets();
    assert!(first > 0, "a data read must fetch something");

    // One segment holds `temperature` for every dataset. It is open now, so
    // the next dataset's temperature costs fewer reads than the first did.
    counting.reset();
    let feb = atlas.dataset("feb_2024").unwrap();
    let _ = feb
        .read_array::<f32>("temperature", vec![], vec![])
        .await
        .unwrap();
    assert!(
        counting.gets() < first,
        "the second dataset cost {} reads, the first {first}",
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
        ds.get_attribute("note").await.unwrap(),
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
            // No finish. The writer drops here.
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
        // No w.finish(), so no trailer ever lands.
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
        // Stage all four at once. They finish in any order. The writer's lock
        // keeps their segments apart.
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
    // Every dataset declares `x`, so one segment holds all four.
    assert_eq!(atlas.list_arrays(), vec!["x"]);
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
async fn ordinals_follow_add_dataset_order_not_finish_order() {
    let tmp = tempfile::tempdir().unwrap();
    let w = AtlasWriter::create_path(tmp.path(), WriterConfig::default())
        .await
        .unwrap();

    // Open three in order, then finish them backwards. Parallel staging does
    // the same thing, just without the deliberate reversal.
    let mut open = Vec::new();
    for name in ["a", "b", "c"] {
        let mut ds = w.add_dataset(name).await.unwrap();
        ds.define_array::<i64>("x", vec!["i".into()], vec![1], None, None)
            .await
            .unwrap();
        let data = Array1::from_vec(vec![name.as_bytes()[0] as i64]).into_dyn();
        ds.write_array("x", vec![0], data.view()).await.unwrap();
        open.push(ds);
    }
    for ds in open.into_iter().rev() {
        ds.finish().await.unwrap();
    }
    w.finish().await.unwrap();

    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    // Call order, not finish order.
    assert_eq!(atlas.list_datasets(), vec!["a", "b", "c"]);
    assert_eq!(atlas.dataset("a").unwrap().ordinal(), 0);
    assert_eq!(atlas.dataset("c").unwrap().ordinal(), 2);

    // Every segment still reads back, wherever its bytes landed.
    for name in ["a", "b", "c"] {
        let got = atlas
            .dataset(name)
            .unwrap()
            .read_array::<i64>("x", vec![], vec![])
            .await
            .unwrap();
        assert_eq!(got[[0]], name.as_bytes()[0] as i64);
    }
}

#[tokio::test]
async fn a_name_stays_reserved_after_an_aborted_dataset() {
    let tmp = tempfile::tempdir().unwrap();
    let w = AtlasWriter::create_path(tmp.path(), WriterConfig::default())
        .await
        .unwrap();

    // This dataset never enters the container.
    let ds = w.add_dataset("d").await.unwrap();
    drop(ds);

    // Its name is still spoken for. The reservation starts at add_dataset,
    // not at finish.
    assert!(matches!(
        w.add_dataset("d").await,
        Err(Error::DatasetAlreadyExists(_))
    ));

    w.add_dataset("e").await.unwrap().finish().await.unwrap();
    w.finish().await.unwrap();

    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    assert_eq!(atlas.list_datasets(), vec!["e"]);
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
        // Nothing tells the reader the codec. Each block describes itself.
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
    // A small block target forces several blocks. That tests the streamed
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
    // Now a window from the middle. It must not fetch the whole array.
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
        // Slabs with no chunk alignment and no common size.
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

    // The objects land under the prefix, not at the root.
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

    // A lost trailer loses the magic with it.
    std::fs::write(&path, &full[..full.len() - 4]).unwrap();
    assert!(matches!(
        Atlas::open_path(tmp.path()).await,
        Err(Error::NotAnAtlasCollection { .. })
    ));

    // The trailer survives. The footer it points at does not.
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
            expected: 7
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
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 7);

    // Trailer: footer size, version, magic.
    let len = bytes.len();
    assert_eq!(&bytes[len - 4..], b"ATLS");
    assert_eq!(
        u32::from_le_bytes(bytes[len - 8..len - 4].try_into().unwrap()),
        7
    );
    let footer_size = u64::from_le_bytes(bytes[len - 16..len - 8].try_into().unwrap()) as usize;
    assert!(footer_size > 0 && footer_size < len - 24);

    // One segment per distinct array name, packed back to back from just
    // after the header, and ending where the container footer starts.
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    assert_eq!(
        atlas.list_arrays(),
        vec!["counts", "name", "observed", "temperature"]
    );
    let segment_bytes = len - 8 - 16 - footer_size;
    assert!(segment_bytes > 0);
    // Each variable contributes one array-format file, and that file ends in
    // its own magic. The bytes just before the container footer are therefore
    // the last segment's trailer.
    let last = len - 16 - footer_size;
    assert_eq!(&bytes[last - 4..last], b"ARRF");
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

