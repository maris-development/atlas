use std::sync::Arc;

use array_store::{ArraySchema, ArrayStore, Attr, DatasetMeta, DType};
use ndarray::ArrayD;
use object_store::{local::LocalFileSystem, path::Path};

fn make_store(tmp: &tempfile::TempDir) -> (Arc<dyn object_store::ObjectStore>, Path) {
    let store: Arc<dyn object_store::ObjectStore> = Arc::new(LocalFileSystem::new());
    let prefix = Path::from_absolute_path(tmp.path()).unwrap();
    (store, prefix)
}

#[tokio::test(flavor = "current_thread")]
async fn create_write_read_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut array_store = ArrayStore::create(store.clone(), prefix.clone()).await.unwrap();

    let data = ArrayD::<f32>::from_elem(vec![4, 4], 42.0_f32);

    {
        let mut ds = array_store.create_dataset("ds_jan").await.unwrap();
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
        ds.set_attribute("month", Attr::Int32(1));
        ds.flush().await.unwrap();
    }

    let store2 = ArrayStore::open(store, prefix).await.unwrap();
    assert!(store2.dataset_exists("ds_jan"));

    let ds2 = store2.open_dataset("ds_jan").await.unwrap();
    let result = ds2
        .read_array::<f32>("temperature", vec![], vec![])
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result, data.into_shared());
    assert_eq!(ds2.get_attribute("month"), Some(&Attr::Int32(1)));
}

#[tokio::test(flavor = "current_thread")]
async fn two_datasets_share_array_file() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut array_store = ArrayStore::create(store.clone(), prefix.clone()).await.unwrap();

    let data_jan = ArrayD::<f32>::from_elem(vec![2, 2], 1.0_f32);
    let data_feb = ArrayD::<f32>::from_elem(vec![2, 2], 2.0_f32);

    {
        let mut ds = array_store.create_dataset("ds_jan").await.unwrap();
        ds.define_array::<f32>("temp", vec!["x".into(), "y".into()], vec![2, 2], None, None)
            .await
            .unwrap();
        ds.write_array("temp", vec![0, 0], data_jan.view()).await.unwrap();
        ds.flush().await.unwrap();
    }

    {
        let mut ds = array_store.create_dataset("ds_feb").await.unwrap();
        ds.define_array::<f32>("temp", vec!["x".into(), "y".into()], vec![2, 2], None, None)
            .await
            .unwrap();
        ds.write_array("temp", vec![0, 0], data_feb.view()).await.unwrap();
        ds.flush().await.unwrap();
    }

    let store2 = ArrayStore::open(store, prefix).await.unwrap();
    let mut datasets = store2.list_datasets();
    datasets.sort();
    assert_eq!(datasets, vec!["ds_feb", "ds_jan"]);

    let ds_jan = store2.open_dataset("ds_jan").await.unwrap();
    let ds_feb = store2.open_dataset("ds_feb").await.unwrap();

    let jan = ds_jan.read_array::<f32>("temp", vec![], vec![]).await.unwrap().unwrap();
    let feb = ds_feb.read_array::<f32>("temp", vec![], vec![]).await.unwrap().unwrap();

    assert_eq!(jan, data_jan.into_shared());
    assert_eq!(feb, data_feb.into_shared());
}

#[tokio::test(flavor = "current_thread")]
async fn list_datasets_and_arrays() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut array_store = ArrayStore::create(store.clone(), prefix.clone()).await.unwrap();

    for name in &["a", "b", "c"] {
        let mut ds = array_store.create_dataset(name).await.unwrap();
        ds.define_array::<f32>("x", vec!["i".into()], vec![3], None, None)
            .await
            .unwrap();
        ds.flush().await.unwrap();
    }

    let store2 = ArrayStore::open(store, prefix).await.unwrap();
    let mut names = store2.list_datasets();
    names.sort();
    assert_eq!(names, vec!["a", "b", "c"]);
    assert_eq!(store2.list_arrays(), vec!["x"]);
}

#[tokio::test(flavor = "current_thread")]
async fn delete_dataset() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut array_store = ArrayStore::create(store.clone(), prefix.clone()).await.unwrap();

    {
        let mut ds = array_store.create_dataset("to_delete").await.unwrap();
        ds.define_array::<f32>("arr", vec!["i".into()], vec![4], None, None)
            .await
            .unwrap();
        ds.flush().await.unwrap();
    }

    assert!(array_store.dataset_exists("to_delete"));
    array_store.delete_dataset("to_delete").await.unwrap();
    assert!(!array_store.dataset_exists("to_delete"));

    let store2 = ArrayStore::open(store, prefix).await.unwrap();
    assert!(!store2.dataset_exists("to_delete"));
}

#[tokio::test(flavor = "current_thread")]
async fn attributes_survive_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut array_store = ArrayStore::create(store.clone(), prefix.clone()).await.unwrap();

    {
        let mut ds = array_store.create_dataset("meta_test").await.unwrap();
        ds.define_array::<f32>("v", vec!["t".into()], vec![2], None, None)
            .await
            .unwrap();
        ds.set_attribute("sensor", Attr::String("ABC".into()));
        ds.set_attribute("year", Attr::UInt32(2023));
        ds.flush().await.unwrap();
    }

    let store2 = ArrayStore::open(store, prefix).await.unwrap();
    let ds2 = store2.open_dataset("meta_test").await.unwrap();
    assert_eq!(ds2.get_attribute("sensor"), Some(&Attr::String("ABC".into())));
    assert_eq!(ds2.get_attribute("year"), Some(&Attr::UInt32(2023)));
}

#[tokio::test(flavor = "current_thread")]
async fn reject_invalid_names() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut array_store = ArrayStore::create(store, prefix).await.unwrap();

    assert!(array_store.create_dataset("").await.is_err());
    assert!(array_store.create_dataset("..").await.is_err());
    assert!(array_store.create_dataset("a/b").await.is_err());
    assert!(array_store.create_dataset("_hidden").await.is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn meta_survives_flush_and_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut array_store = ArrayStore::create(store.clone(), prefix.clone()).await.unwrap();

    {
        let mut ds = array_store.create_dataset("meta_test").await.unwrap();
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
        ds.set_attribute("year", Attr::UInt32(2024));
        ds.set_attribute("active", Attr::Bool(true));
        ds.flush().await.unwrap();
    }

    let store2 = ArrayStore::open(store, prefix).await.unwrap();
    let ds2 = store2.open_dataset("meta_test").await.unwrap();
    let meta: &DatasetMeta = ds2.meta();

    let temp_schema: &ArraySchema = meta.arrays.get("temp").expect("temp array schema missing");
    assert_eq!(temp_schema.dtype, DType::Float32);
    assert_eq!(temp_schema.shape, vec![4, 8]);
    assert_eq!(temp_schema.chunk_shape, vec![2, 4]);
    assert_eq!(temp_schema.dimension_names, vec!["lat", "lon"]);

    let time_schema: &ArraySchema = meta.arrays.get("time").expect("time array schema missing");
    assert_eq!(time_schema.dtype, DType::Int64);
    assert_eq!(time_schema.shape, vec![100]);
    assert_eq!(time_schema.chunk_shape, vec![100]);

    assert_eq!(meta.attributes.get("year"), Some(&Attr::UInt32(2024)));
    assert_eq!(meta.attributes.get("active"), Some(&Attr::Bool(true)));
}
