use std::sync::Arc;

use atlas::{Atlas, Attr, DType, StatValue, StoreConfig};
use ndarray::ArrayD;
use object_store::{local::LocalFileSystem, path::Path};

fn make_store(tmp: &tempfile::TempDir) -> (Arc<dyn object_store::ObjectStore>, Path) {
    let store: Arc<dyn object_store::ObjectStore> = Arc::new(LocalFileSystem::new());
    let prefix = Path::from_absolute_path(tmp.path()).unwrap();
    (store, prefix)
}

#[tokio::test]
async fn create_write_read_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default()).await.unwrap();

    let data = ArrayD::<f32>::from_elem(vec![4, 4], 42.0_f32);

    {
        let mut ds = atlas.create_dataset("ds_jan").await.unwrap();
        ds.define_array::<f32>(
            "temperature",
            vec!["lat".into(), "lon".into()],
            vec![4, 4],
            None,
            None,
        )
        .await
        .unwrap();
        ds.write_array("temperature", vec![0, 0], data.view()).await.unwrap();
        ds.set_attribute("month", Attr::Int64(1));
    }
    atlas.flush().await.unwrap();

    let atlas2 = Atlas::open(store, prefix).await.unwrap();
    assert!(atlas2.dataset_exists("ds_jan"));

    let ds2 = atlas2.open_dataset("ds_jan").await.unwrap();
    let result = ds2
        .read_array::<f32>("temperature", vec![], vec![])
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result, data.into_shared());
    assert_eq!(ds2.get_attribute("month"), Some(Attr::Int64(1)));
}

#[tokio::test]
async fn two_datasets_share_array_file() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default()).await.unwrap();

    let data_jan = ArrayD::<f32>::from_elem(vec![2, 2], 1.0_f32);
    let data_feb = ArrayD::<f32>::from_elem(vec![2, 2], 2.0_f32);

    {
        let mut ds = atlas.create_dataset("ds_jan").await.unwrap();
        ds.define_array::<f32>("temp", vec!["x".into(), "y".into()], vec![2, 2], None, None)
            .await
            .unwrap();
        ds.write_array("temp", vec![0, 0], data_jan.view()).await.unwrap();
    }
    {
        let mut ds = atlas.create_dataset("ds_feb").await.unwrap();
        ds.define_array::<f32>("temp", vec!["x".into(), "y".into()], vec![2, 2], None, None)
            .await
            .unwrap();
        ds.write_array("temp", vec![0, 0], data_feb.view()).await.unwrap();
    }
    atlas.flush().await.unwrap();

    let atlas2 = Atlas::open(store, prefix).await.unwrap();
    let mut datasets = atlas2.list_datasets();
    datasets.sort();
    assert_eq!(datasets, vec!["ds_feb".to_string(), "ds_jan".to_string()]);

    let ds_jan = atlas2.open_dataset("ds_jan").await.unwrap();
    let ds_feb = atlas2.open_dataset("ds_feb").await.unwrap();

    let jan = ds_jan.read_array::<f32>("temp", vec![], vec![]).await.unwrap().unwrap();
    let feb = ds_feb.read_array::<f32>("temp", vec![], vec![]).await.unwrap().unwrap();

    assert_eq!(jan, data_jan.into_shared());
    assert_eq!(feb, data_feb.into_shared());
}

#[tokio::test]
async fn list_datasets_and_arrays() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default()).await.unwrap();

    for name in &["a", "b", "c"] {
        let mut ds = atlas.create_dataset(name).await.unwrap();
        ds.define_array::<f32>("x", vec!["i".into()], vec![3], None, None)
            .await
            .unwrap();
    }
    atlas.flush().await.unwrap();

    let atlas2 = Atlas::open(store, prefix).await.unwrap();
    let mut names = atlas2.list_datasets();
    names.sort();
    assert_eq!(names, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    assert_eq!(atlas2.list_arrays(), vec!["x".to_string()]);
}

#[tokio::test]
async fn delete_dataset() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default()).await.unwrap();

    {
        let mut ds = atlas.create_dataset("to_delete").await.unwrap();
        ds.define_array::<f32>("arr", vec!["i".into()], vec![4], None, None)
            .await
            .unwrap();
    }
    atlas.flush().await.unwrap();

    assert!(atlas.dataset_exists("to_delete"));
    atlas.delete_dataset("to_delete").await.unwrap();
    atlas.flush().await.unwrap();
    assert!(!atlas.dataset_exists("to_delete"));

    let atlas2 = Atlas::open(store, prefix).await.unwrap();
    assert!(!atlas2.dataset_exists("to_delete"));
}

#[tokio::test]
async fn attributes_survive_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default()).await.unwrap();

    {
        let mut ds = atlas.create_dataset("meta_test").await.unwrap();
        ds.define_array::<f32>("v", vec!["t".into()], vec![2], None, None)
            .await
            .unwrap();
        ds.set_attribute("sensor", Attr::String("ABC".into()));
        ds.set_attribute("year", Attr::Int64(2023));
    }
    atlas.flush().await.unwrap();

    let atlas2 = Atlas::open(store, prefix).await.unwrap();
    let ds2 = atlas2.open_dataset("meta_test").await.unwrap();
    assert_eq!(ds2.get_attribute("sensor"), Some(Attr::String("ABC".into())));
    assert_eq!(ds2.get_attribute("year"), Some(Attr::Int64(2023)));
}

#[tokio::test]
async fn reject_invalid_names() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store, prefix, StoreConfig::default()).await.unwrap();

    assert!(atlas.create_dataset("").await.is_err());
    assert!(atlas.create_dataset("..").await.is_err());
    assert!(atlas.create_dataset("a/b").await.is_err());
    assert!(atlas.create_dataset("_hidden").await.is_err());
}

#[tokio::test]
async fn meta_survives_flush_and_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default()).await.unwrap();

    {
        let mut ds = atlas.create_dataset("meta_test").await.unwrap();
        ds.define_array::<f32>(
            "temp",
            vec!["lat".into(), "lon".into()],
            vec![4, 8],
            Some(vec![2, 4]),
            None,
        )
        .await
        .unwrap();
        ds.define_array::<i64>("time", vec!["t".into()], vec![100], None, None)
            .await
            .unwrap();
        ds.set_attribute("year", Attr::Int64(2024));
        ds.set_attribute("active", Attr::Bool(true));
    }
    atlas.flush().await.unwrap();

    let atlas2 = Atlas::open(store, prefix).await.unwrap();
    let ds2 = atlas2.open_dataset("meta_test").await.unwrap();
    let meta = ds2.meta();

    let temp_schema = meta.arrays.get("temp").expect("temp array schema missing");
    assert_eq!(temp_schema.dtype, DType::Float32);
    assert_eq!(temp_schema.shape, vec![4, 8]);
    assert_eq!(temp_schema.chunk_shape, vec![2, 4]);
    assert_eq!(temp_schema.dimension_names, vec!["lat", "lon"]);

    let time_schema = meta.arrays.get("time").expect("time array schema missing");
    assert_eq!(time_schema.dtype, DType::Int64);
    assert_eq!(time_schema.shape, vec![100]);
    assert_eq!(time_schema.chunk_shape, vec![100]);

    assert_eq!(meta.attributes.get("year"), Some(&Attr::Int64(2024)));
    assert_eq!(meta.attributes.get("active"), Some(&Attr::Bool(true)));
}

#[tokio::test]
async fn atlas_no_implicit_flush() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default()).await.unwrap();

    {
        let mut ds = atlas.create_dataset("a").await.unwrap();
        ds.define_array::<f32>("arr", vec!["x".into()], vec![2], None, None).await.unwrap();
        ds.write_array("arr", vec![0], ndarray::arr1(&[1.0f32, 2.0]).into_dyn().view())
            .await
            .unwrap();
        // No flush — drop atlas-side handle but DO NOT call atlas.flush.
    }

    // A fresh Atlas opened on the same store sees nothing — atlas.json was never written.
    let atlas2 = Atlas::open(store, prefix).await.unwrap();
    assert!(atlas2.list_datasets().is_empty());
}

#[tokio::test]
async fn atlas_flush_persists_meta_and_data() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default()).await.unwrap();

    {
        let mut ds_a = atlas.create_dataset("a").await.unwrap();
        ds_a.define_array::<f32>("temp", vec!["x".into()], vec![4], None, None).await.unwrap();
        ds_a.write_array("temp", vec![0], ndarray::arr1(&[1.0f32, 2.0, 3.0, 4.0]).into_dyn().view())
            .await
            .unwrap();
    }
    {
        let mut ds_b = atlas.create_dataset("b").await.unwrap();
        ds_b.define_array::<f32>("temp", vec!["x".into()], vec![4], None, None).await.unwrap();
        ds_b.write_array("temp", vec![0], ndarray::arr1(&[5.0f32, 6.0, 7.0, 8.0]).into_dyn().view())
            .await
            .unwrap();
    }
    atlas.flush().await.unwrap();

    let atlas2 = Atlas::open(store, prefix).await.unwrap();
    let mut names = atlas2.list_datasets();
    names.sort();
    assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
    let a = atlas2.open_dataset("a").await.unwrap();
    let b = atlas2.open_dataset("b").await.unwrap();
    let a_data = a.read_array::<f32>("temp", vec![], vec![]).await.unwrap().unwrap();
    let b_data = b.read_array::<f32>("temp", vec![], vec![]).await.unwrap().unwrap();
    assert_eq!(a_data.as_slice().unwrap(), &[1.0f32, 2.0, 3.0, 4.0]);
    assert_eq!(b_data.as_slice().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]);
}

#[tokio::test]
async fn define_array_zero_dimensional_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default()).await.unwrap();

    {
        let mut ds = atlas.create_dataset("scalars").await.unwrap();
        ds.define_array::<f32>("answer", vec![], vec![], None, None).await.unwrap();
        let data = ndarray::Array0::from_elem((), 42.0_f32).into_dyn();
        ds.write_array("answer", vec![], data.view()).await.unwrap();
    }
    atlas.flush().await.unwrap();

    let atlas2 = Atlas::open(store, prefix).await.unwrap();
    let ds2 = atlas2.open_dataset("scalars").await.unwrap();
    let schema = ds2.meta();
    let answer = schema.arrays.get("answer").unwrap();
    assert_eq!(answer.shape, Vec::<usize>::new());
    assert_eq!(answer.dimension_names, Vec::<String>::new());

    let read = ds2.read_array::<f32>("answer", vec![], vec![]).await.unwrap().unwrap();
    assert_eq!(read.ndim(), 0);
    assert_eq!(read[ndarray::IxDyn(&[])], 42.0);
}

#[tokio::test]
async fn timestamp_ns_array_and_attr_survive_flush_and_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default()).await.unwrap();

    {
        let mut ds = atlas.create_dataset("ts").await.unwrap();
        ds.define_array::<atlas::TimestampNs>(
            "event_time",
            vec!["t".into()],
            vec![3],
            None,
            None,
        )
        .await
        .unwrap();

        let data = ndarray::arr1(&[
            atlas::TimestampNs(1_700_000_000_000_000_000),
            atlas::TimestampNs(1_700_000_000_000_000_001),
            atlas::TimestampNs(1_700_000_000_000_000_002),
        ])
        .into_dyn();
        ds.write_array("event_time", vec![0], data.view()).await.unwrap();
        ds.set_attribute("created_at", Attr::TimestampNanoseconds(1_700_000_000_000_000_000));
    }
    atlas.flush().await.unwrap();

    let atlas2 = Atlas::open(store, prefix).await.unwrap();
    let ds2 = atlas2.open_dataset("ts").await.unwrap();

    let meta = ds2.meta();
    let schema = meta.arrays.get("event_time").unwrap();
    assert_eq!(schema.dtype, DType::TimestampNs);

    let read = ds2
        .read_array::<atlas::TimestampNs>("event_time", vec![], vec![])
        .await
        .unwrap()
        .unwrap();
    assert_eq!(read[0].0, 1_700_000_000_000_000_000);
    assert_eq!(read[2].0, 1_700_000_000_000_000_002);

    assert_eq!(
        meta.attributes.get("created_at"),
        Some(&Attr::TimestampNanoseconds(1_700_000_000_000_000_000)),
    );
}

#[tokio::test]
async fn array_stats_after_flush() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store, prefix, StoreConfig::default()).await.unwrap();

    let mut ds = atlas.create_dataset("stats_test").await.unwrap();

    ds.define_array::<f32>("temp", vec!["x".into()], vec![4], None, None)
        .await
        .unwrap();
    let data = ndarray::arr1(&[10.0_f32, 20.0, 5.0, 15.0]).into_dyn();
    ds.write_array("temp", vec![0], data.view()).await.unwrap();

    ds.define_array::<i64>("time", vec!["t".into()], vec![3], None, None)
        .await
        .unwrap();
    let times = ndarray::arr1(&[100_i64, 200, 300]).into_dyn();
    ds.write_array("time", vec![0], times.view()).await.unwrap();

    // Stats are None before flush
    assert!(ds.array_stats("temp").await.is_none());

    // Drop the borrow so we can call atlas.flush.
    drop(ds);
    atlas.flush().await.unwrap();

    // Reopen to verify stats persisted.
    let ds_reopened = atlas.open_dataset("stats_test").await.unwrap();
    let temp_stats = ds_reopened.array_stats("temp").await.unwrap();
    assert_eq!(temp_stats.row_count, 4);
    assert_eq!(temp_stats.null_count, 0);
    assert_eq!(temp_stats.min, Some(StatValue::Float(5.0)));
    assert_eq!(temp_stats.max, Some(StatValue::Float(20.0)));

    let time_stats = ds_reopened.array_stats("time").await.unwrap();
    assert_eq!(time_stats.row_count, 3);
    assert_eq!(time_stats.min, Some(StatValue::Int(100)));
    assert_eq!(time_stats.max, Some(StatValue::Int(300)));
}

#[tokio::test]
async fn array_stats_survive_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default()).await.unwrap();

    {
        let mut ds = atlas.create_dataset("ds").await.unwrap();
        ds.define_array::<f64>("values", vec!["i".into()], vec![5], None, None)
            .await
            .unwrap();
        let data = ndarray::arr1(&[3.0_f64, 1.0, 4.0, 1.5, 9.0]).into_dyn();
        ds.write_array("values", vec![0], data.view()).await.unwrap();
    }
    atlas.flush().await.unwrap();

    let atlas2 = Atlas::open(store, prefix).await.unwrap();
    let ds2 = atlas2.open_dataset("ds").await.unwrap();
    let stats = ds2.array_stats("values").await.unwrap();
    assert_eq!(stats.row_count, 5);
    assert_eq!(stats.min, Some(StatValue::Float(1.0)));
    assert_eq!(stats.max, Some(StatValue::Float(9.0)));
}

#[tokio::test]
async fn array_stats_unknown_array_returns_none() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store, prefix, StoreConfig::default()).await.unwrap();
    let ds = atlas.create_dataset("ds").await.unwrap();
    assert!(ds.array_stats("ghost").await.is_none());
}
