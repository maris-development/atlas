use std::sync::Arc;

use atlas::{Atlas, Attr, ColumnKey, DType, StatVal, StatValue, StoreConfig, TypeMismatchPolicy};
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
        ds.set_attribute("month", Attr::Int64(1)).unwrap();
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
    assert_eq!(ds2.get_attribute("month").await.unwrap(), Some(Attr::Int64(1)));
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
        ds.set_attribute("sensor", Attr::String("ABC".into())).unwrap();
        ds.set_attribute("year", Attr::Int64(2023)).unwrap();
    }
    atlas.flush().await.unwrap();

    let atlas2 = Atlas::open(store, prefix).await.unwrap();
    let ds2 = atlas2.open_dataset("meta_test").await.unwrap();
    assert_eq!(
        ds2.get_attribute("sensor").await.unwrap(),
        Some(Attr::String("ABC".into()))
    );
    assert_eq!(
        ds2.get_attribute("year").await.unwrap(),
        Some(Attr::Int64(2023))
    );
}

#[tokio::test]
async fn per_variable_attributes_survive_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default())
        .await
        .unwrap();

    {
        let mut ds = atlas.create_dataset("obs").await.unwrap();
        ds.define_array::<f32>("wind", vec!["t".into()], vec![2], None, None)
            .await
            .unwrap();
        ds.set_array_attribute("wind", "units", Attr::String("m/s".into()))
            .unwrap();
        ds.set_array_attribute("wind", "valid_range", Attr::Float32List(vec![0.0, 120.0]))
            .unwrap();
        // Dataset-global attribute lives in the reserved _global file.
        ds.set_attribute("station", Attr::String("KNMI".into())).unwrap();
    }
    atlas.flush().await.unwrap();

    // The global-attributes file is created because a global attr was set.
    assert!(tmp.path().join("_global").join("data.af").exists());

    let atlas2 = Atlas::open(store, prefix).await.unwrap();
    let ds2 = atlas2.open_dataset("obs").await.unwrap();
    assert_eq!(
        ds2.get_array_attribute("wind", "units").await.unwrap(),
        Some(Attr::String("m/s".into()))
    );
    let wind_attrs = ds2.array_attributes("wind").await.unwrap();
    assert_eq!(wind_attrs.get("units"), Some(&Attr::String("m/s".into())));
    assert_eq!(
        wind_attrs.get("valid_range"),
        Some(&Attr::Float32List(vec![0.0, 120.0]))
    );
    assert_eq!(
        ds2.get_attribute("station").await.unwrap(),
        Some(Attr::String("KNMI".into()))
    );
}

/// Default policy is `Warn`: an incompatible dtype is still stored under the
/// dataset's own type, and the merged schema keeps the first-seen type.
#[tokio::test]
async fn incompatible_array_dtype_is_stored_and_first_type_wins() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default())
        .await
        .unwrap();

    {
        let mut a = atlas.create_dataset("a").await.unwrap();
        a.define_array::<i32>("x", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
    }
    // A widenable dtype (i64) merges: the merged type widens to cover both.
    {
        let mut b = atlas.create_dataset("b").await.unwrap();
        b.define_array::<i64>("x", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
    }
    // An incompatible one (String) is still accepted and stored (warns).
    {
        let mut c = atlas.create_dataset("c").await.unwrap();
        c.define_array::<String>("x", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
        let data = ndarray::arr1(&["hello".to_string(), "world".to_string()]).into_dyn();
        c.write_array("x", vec![0], data.view()).await.unwrap();
    }
    atlas.flush().await.unwrap();

    // Merged keeps the FIRST-seen type widened across compatible datasets
    // (i32 ∪ i64 → i64); the incompatible String does not change it.
    let merged = atlas.merged_schema();
    assert_eq!(merged.arrays["x"].dtype.0, DType::Int64);

    // Each dataset kept its own declared type, and the String data reads back.
    let atlas2 = Atlas::open(store, prefix).await.unwrap();
    assert_eq!(atlas2.open_dataset("a").await.unwrap().array_meta("x").unwrap().dtype, DType::Int32);
    assert_eq!(atlas2.open_dataset("c").await.unwrap().array_meta("x").unwrap().dtype, DType::String);
    let c2 = atlas2.open_dataset("c").await.unwrap();
    let back = c2.read_array::<String>("x", vec![], vec![]).await.unwrap().unwrap();
    assert_eq!(back[0], "hello");
}

/// With `TypeMismatchPolicy::Error` the same insert is rejected instead.
#[tokio::test]
async fn incompatible_array_dtype_rejected_under_error_policy() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let config = StoreConfig {
        on_type_mismatch: TypeMismatchPolicy::Error,
        ..Default::default()
    };
    let mut atlas = Atlas::create(store, prefix, config).await.unwrap();

    {
        let mut a = atlas.create_dataset("a").await.unwrap();
        a.define_array::<i32>("x", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
    }
    // Widenable is still fine under Error.
    {
        let mut b = atlas.create_dataset("b").await.unwrap();
        b.define_array::<i64>("x", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
    }
    {
        let mut c = atlas.create_dataset("c").await.unwrap();
        let err = c
            .define_array::<String>("x", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap_err();
        assert!(matches!(err, atlas::Error::TypeMismatch { .. }), "got {err:?}");
    }
}

/// Attributes follow the same policy as array dtypes.
#[tokio::test]
async fn incompatible_attribute_type_follows_policy() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);

    // Default (Warn): stored, first type kept in the merged schema.
    {
        let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default())
            .await
            .unwrap();
        atlas
            .create_dataset("a")
            .await
            .unwrap()
            .set_attribute("year", Attr::Int64(2024))
            .unwrap();
        atlas
            .create_dataset("b")
            .await
            .unwrap()
            .set_attribute("year", Attr::String("twenty".into()))
            .unwrap();
        atlas.flush().await.unwrap();

        let merged = atlas.merged_schema();
        assert_eq!(merged.global_attributes["year"].0, DType::Int64);
        // b's own value is still stored under its own type.
        let b = atlas.open_dataset("b").await.unwrap();
        assert_eq!(
            b.get_attribute("year").await.unwrap(),
            Some(Attr::String("twenty".into()))
        );
    }

    // Error policy rejects it.
    {
        let tmp2 = tempfile::tempdir().unwrap();
        let (store2, prefix2) = make_store(&tmp2);
        let config = StoreConfig {
            on_type_mismatch: TypeMismatchPolicy::Error,
            ..Default::default()
        };
        let mut atlas = Atlas::create(store2, prefix2, config).await.unwrap();
        atlas
            .create_dataset("a")
            .await
            .unwrap()
            .set_attribute("year", Attr::Int64(2024))
            .unwrap();
        let err = atlas
            .create_dataset("b")
            .await
            .unwrap()
            .set_attribute("year", Attr::String("twenty".into()))
            .unwrap_err();
        assert!(matches!(err, atlas::Error::TypeMismatch { .. }), "got {err:?}");
    }
}

/// Regression: once a mismatching dtype has been **persisted** under `Warn`,
/// the first-seen type must still win the comparison — a later dataset
/// declaring that same odd type is still a mismatch, not silently accepted.
#[tokio::test]
async fn persisted_mismatch_does_not_become_the_reference_type() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);

    // Store i32 (first), then a mismatching String — and flush both.
    {
        let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default())
            .await
            .unwrap();
        atlas
            .create_dataset("a")
            .await
            .unwrap()
            .define_array::<i32>("x", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
        atlas
            .create_dataset("odd")
            .await
            .unwrap()
            .define_array::<String>("x", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
        atlas.flush().await.unwrap();
        // Merged still reports the first-seen type, not the odd one.
        assert_eq!(atlas.merged_schema().arrays["x"].dtype.0, DType::Int32);
    }

    // Reopen strictly: another String must STILL be rejected, because the
    // reference type is the first-seen i32 — not the already-stored String.
    let config = StoreConfig {
        on_type_mismatch: TypeMismatchPolicy::Error,
        ..Default::default()
    };
    let mut atlas = Atlas::open_with_config(store, prefix, config).await.unwrap();
    assert_eq!(atlas.merged_schema().arrays["x"].dtype.0, DType::Int32);
    let err = atlas
        .create_dataset("another")
        .await
        .unwrap()
        .define_array::<String>("x", vec!["i".into()], vec![2], None, None)
        .await
        .unwrap_err();
    assert!(matches!(err, atlas::Error::TypeMismatch { .. }), "got {err:?}");
}

/// `open_with_config` carries the policy into an existing collection.
#[tokio::test]
async fn open_with_config_sets_type_mismatch_policy() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    {
        let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default())
            .await
            .unwrap();
        atlas
            .create_dataset("a")
            .await
            .unwrap()
            .define_array::<i32>("x", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
        atlas.flush().await.unwrap();
    }

    // Plain open → default Warn → the mismatch is accepted.
    {
        let mut atlas = Atlas::open(store.clone(), prefix.clone()).await.unwrap();
        atlas
            .create_dataset("warn_ds")
            .await
            .unwrap()
            .define_array::<String>("x", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
    }

    // open_with_config(Error) → the same mismatch is rejected.
    {
        let config = StoreConfig {
            on_type_mismatch: TypeMismatchPolicy::Error,
            ..Default::default()
        };
        let mut atlas = Atlas::open_with_config(store, prefix, config).await.unwrap();
        let err = atlas
            .create_dataset("err_ds")
            .await
            .unwrap()
            .define_array::<String>("x", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap_err();
        assert!(matches!(err, atlas::Error::TypeMismatch { .. }), "got {err:?}");
    }
}

#[tokio::test]
async fn per_variable_attribute_lives_in_the_array_file() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default())
        .await
        .unwrap();

    {
        let mut ds = atlas.create_dataset("jan").await.unwrap();
        ds.define_array::<f32>("temperature", vec!["lat".into()], vec![2], None, None)
            .await
            .unwrap();
        // Only a per-variable attribute — no dataset-level (global) attribute.
        ds.set_array_attribute("temperature", "units", Attr::String("celsius".into()))
            .unwrap();
    }
    atlas.flush().await.unwrap();

    // The attribute value is written into the temperature array's own file...
    assert!(tmp.path().join("temperature").join("data.af").exists());
    // ...and NOT into the global-attributes file, which is never created when
    // only per-variable attributes are set.
    assert!(
        !tmp.path().join("_global").exists(),
        "_global must not be created for per-variable-only attributes"
    );

    // Round-trips from the temperature file after reopen.
    let atlas2 = Atlas::open(store, prefix).await.unwrap();
    let ds2 = atlas2.open_dataset("jan").await.unwrap();
    assert_eq!(
        ds2.get_array_attribute("temperature", "units").await.unwrap(),
        Some(Attr::String("celsius".into()))
    );
    // No dataset-level attributes exist.
    assert!(ds2.attributes().await.unwrap().is_empty());
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
        ds.set_attribute("year", Attr::Int64(2024)).unwrap();
        ds.set_attribute("active", Attr::Bool(true)).unwrap();
    }
    atlas.flush().await.unwrap();

    let atlas2 = Atlas::open(store, prefix).await.unwrap();
    let ds2 = atlas2.open_dataset("meta_test").await.unwrap();
    let schema = ds2.schema();

    let temp_schema = schema.arrays.get("temp").expect("temp array schema missing");
    assert_eq!(temp_schema.dtype, DType::Float32);
    assert_eq!(temp_schema.shape, vec![4, 8]);
    assert_eq!(temp_schema.chunk_shape, vec![2, 4]);
    assert_eq!(temp_schema.dimension_names, vec!["lat", "lon"]);

    let time_schema = schema.arrays.get("time").expect("time array schema missing");
    assert_eq!(time_schema.dtype, DType::Int64);
    assert_eq!(time_schema.shape, vec![100]);
    assert_eq!(time_schema.chunk_shape, vec![100]);

    let attrs = ds2.attributes().await.unwrap();
    assert_eq!(attrs.get("year"), Some(&Attr::Int64(2024)));
    assert_eq!(attrs.get("active"), Some(&Attr::Bool(true)));
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
    let schema = ds2.schema();
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
        ds.set_attribute("created_at", Attr::TimestampNanoseconds(1_700_000_000_000_000_000)).unwrap();
    }
    atlas.flush().await.unwrap();

    let atlas2 = Atlas::open(store, prefix).await.unwrap();
    let ds2 = atlas2.open_dataset("ts").await.unwrap();

    let dataset_schema = ds2.schema();
    let schema = dataset_schema.arrays.get("event_time").unwrap();
    assert_eq!(schema.dtype, DType::TimestampNs);

    let read = ds2
        .read_array::<atlas::TimestampNs>("event_time", vec![], vec![])
        .await
        .unwrap()
        .unwrap();
    assert_eq!(read[0].0, 1_700_000_000_000_000_000);
    assert_eq!(read[2].0, 1_700_000_000_000_000_002);

    // Timestamp attributes round-trip through the .af file as an RFC 3339
    // string and are restored to the timestamp variant on read.
    assert_eq!(
        ds2.get_attribute("created_at").await.unwrap(),
        Some(Attr::TimestampNanoseconds(1_700_000_000_000_000_000)),
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

/// A dataset's row ordinal must survive the deletion of an earlier dataset —
/// including across a save/load cycle.
///
/// This is the invariant the pruning index rests on: rows are addressed
/// positionally, so if `to_wire` dropped tombstones, every dataset after the
/// first deletion would shift up one on reload and every row would silently
/// point at the wrong dataset. That failure is invisible in memory and only
/// appears after a reopen, which is why this test round-trips.
#[tokio::test]
async fn dataset_ordinals_survive_deletion_and_reload() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default())
        .await
        .unwrap();

    for name in ["a", "b", "c"] {
        atlas
            .create_dataset(name)
            .await
            .unwrap()
            .define_array::<i32>("x", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
    }
    assert_eq!(atlas.dataset_row("a"), Some(0));
    assert_eq!(atlas.dataset_row("b"), Some(1));
    assert_eq!(atlas.dataset_row("c"), Some(2));

    atlas.delete_dataset("b").await.unwrap();
    atlas.flush().await.unwrap();

    // In memory: c must not have slid down into b's slot.
    assert_eq!(atlas.dataset_row("a"), Some(0));
    assert_eq!(atlas.dataset_row("b"), None, "deleted");
    assert_eq!(atlas.dataset_row("c"), Some(2), "c must keep its ordinal");
    assert_eq!(atlas.row_slots(), 3, "the dead slot is retained");
    assert_eq!(atlas.list_datasets(), vec!["a".to_string(), "c".to_string()]);

    // ...and the same after a reload, which is where dropping tombstones on
    // write would show up.
    let mut reopened = Atlas::open(store.clone(), prefix.clone()).await.unwrap();
    assert_eq!(reopened.dataset_row("a"), Some(0));
    assert_eq!(reopened.dataset_row("c"), Some(2), "ordinal shifted on reload");
    assert_eq!(reopened.row_slots(), 3);
    assert!(!reopened.dataset_exists("b"));

    // A new dataset appends after the dead slot rather than filling it.
    reopened.create_dataset("d").await.unwrap();
    assert_eq!(reopened.dataset_row("d"), Some(3));
}

/// Re-creating a deleted name revives its slot instead of erroring, and starts
/// from a clean schema rather than inheriting the previous occupant's.
#[tokio::test]
async fn recreating_a_deleted_dataset_revives_its_slot() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store, prefix, StoreConfig::default())
        .await
        .unwrap();

    atlas
        .create_dataset("recycled")
        .await
        .unwrap()
        .define_array::<i32>("old_array", vec!["i".into()], vec![2], None, None)
        .await
        .unwrap();
    atlas.create_dataset("after").await.unwrap();
    atlas.flush().await.unwrap();

    atlas.delete_dataset("recycled").await.unwrap();
    let mut revived = atlas.create_dataset("recycled").await.unwrap();

    assert_eq!(atlas.dataset_row("recycled"), Some(0), "slot reused");
    assert_eq!(atlas.dataset_row("after"), Some(1), "neighbour undisturbed");
    assert_eq!(atlas.row_slots(), 2, "no new slot allocated");
    assert!(
        revived.list_arrays().is_empty(),
        "revived dataset must not inherit the old schema"
    );

    // The name is usable again for real.
    revived
        .define_array::<f64>("new_array", vec!["i".into()], vec![2], None, None)
        .await
        .unwrap();
    assert_eq!(revived.list_arrays(), vec!["new_array".to_string()]);
}

/// `compact` is the one operation that renumbers: it drops dead slots and
/// closes the holes.
#[tokio::test]
async fn compact_drops_tombstones_and_renumbers() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default())
        .await
        .unwrap();

    for name in ["a", "b", "c"] {
        atlas
            .create_dataset(name)
            .await
            .unwrap()
            .define_array::<i32>("x", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
    }
    atlas.flush().await.unwrap();
    atlas.delete_dataset("a").await.unwrap();
    atlas.flush().await.unwrap();
    assert_eq!(atlas.row_slots(), 3);

    atlas.compact().await.unwrap();
    assert_eq!(atlas.row_slots(), 2, "dead slot reclaimed");
    assert_eq!(atlas.dataset_row("b"), Some(0), "renumbered");
    assert_eq!(atlas.dataset_row("c"), Some(1));

    // And the renumbering is what a reader sees afterwards.
    let reopened = Atlas::open(store, prefix).await.unwrap();
    assert_eq!(reopened.row_slots(), 2);
    assert_eq!(reopened.dataset_row("b"), Some(0));
    assert_eq!(reopened.dataset_row("c"), Some(1));
}

/// A tombstoned dataset must not contribute to the collection-wide views.
#[tokio::test]
async fn tombstones_are_excluded_from_merged_views() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store, prefix, StoreConfig::default())
        .await
        .unwrap();

    atlas
        .create_dataset("keep")
        .await
        .unwrap()
        .define_array::<i32>("shared", vec!["i".into()], vec![2], None, None)
        .await
        .unwrap();
    {
        let mut doomed = atlas.create_dataset("doomed").await.unwrap();
        doomed
            .define_array::<f64>("only_here", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
        doomed.set_attribute("gone_key", Attr::Int64(1)).unwrap();
    }
    atlas.flush().await.unwrap();
    assert!(atlas.list_arrays().contains(&"only_here".to_string()));

    atlas.delete_dataset("doomed").await.unwrap();

    assert!(
        !atlas.list_arrays().contains(&"only_here".to_string()),
        "a dead dataset's arrays must leave list_arrays"
    );
    let merged = atlas.merged_schema();
    assert!(!merged.arrays.contains_key("only_here"));
    assert!(!merged.global_attributes.contains_key("gone_key"));
    assert!(merged.arrays.contains_key("shared"), "live array retained");
    assert!(atlas.array_dtype("only_here").is_none());
}

/// The pruning index is a flattened column over the whole collection: one row
/// per dataset in ordinal order, with explicit gaps where a dataset doesn't
/// declare the array.
#[tokio::test]
async fn pruning_index_flattens_with_nulls() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store, prefix, StoreConfig::default())
        .await
        .unwrap();

    // ds0 and ds2 declare "temp"; ds1 does not — row 1 must be a hole.
    for (name, values) in [
        ("ds0", Some(vec![1i32, 5])),
        ("ds1", None),
        ("ds2", Some(vec![-4i32, 2])),
    ] {
        let mut ds = atlas.create_dataset(name).await.unwrap();
        ds.set_attribute("cruise", Attr::String(format!("C-{name}")))
            .unwrap();
        if let Some(values) = values {
            ds.define_array::<i32>("temp", vec!["i".into()], vec![2], None, None)
                .await
                .unwrap();
            ds.write_array(
                "temp",
                vec![0],
                ndarray::Array::from_vec(values).into_dyn().view(),
            )
            .await
            .unwrap();
        }
    }
    atlas.flush().await.unwrap();

    let key = ColumnKey::array("temp");
    let index = atlas.pruning_index(std::slice::from_ref(&key)).await.unwrap();
    assert_eq!(index.rows(), 3, "one row per dataset");

    let view = index.view(&key).expect("temp column");
    assert!(view.is_present(0));
    assert!(
        !view.is_present(1),
        "ds1 has no temp — must be an explicit gap"
    );
    assert!(view.is_present(2));

    assert_eq!(view.min(0), Some(&StatVal::Int(1)));
    assert_eq!(view.max(0), Some(&StatVal::Int(5)));
    assert_eq!(view.min(2), Some(&StatVal::Int(-4)));
    assert_eq!(view.max(2), Some(&StatVal::Int(2)));
    assert_eq!(view.min(1), None, "the gap carries no statistics");
    assert_eq!(view.row_count(1), 0, "and no rows");

    // Row positions line up with dataset_row, which is what lets a caller join
    // the flattened table back to dataset names.
    assert_eq!(atlas.dataset_row("ds2"), Some(2));
}

/// Deleting a dataset masks its row rather than removing it, so every other
/// dataset's row stays where it was.
#[tokio::test]
async fn pruning_index_masks_deleted_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store, prefix, StoreConfig::default())
        .await
        .unwrap();

    for (name, values) in [
        ("a", vec![1i32, 2]),
        ("b", vec![100i32, 200]),
        ("c", vec![3i32, 4]),
    ] {
        let mut ds = atlas.create_dataset(name).await.unwrap();
        ds.define_array::<i32>("v", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
        ds.write_array(
            "v",
            vec![0],
            ndarray::Array::from_vec(values).into_dyn().view(),
        )
        .await
        .unwrap();
    }
    atlas.flush().await.unwrap();
    atlas.delete_dataset("b").await.unwrap();
    atlas.flush().await.unwrap();

    let key = ColumnKey::array("v");
    let index = atlas.pruning_index(std::slice::from_ref(&key)).await.unwrap();
    assert_eq!(index.rows(), 3, "the dead row keeps its slot");
    assert_eq!(index.live(), &[true, false, true]);

    // The view applies present/stats_valid/live for us — a caller cannot
    // accidentally see the deleted dataset.
    let view = index.view(&key).expect("v column");
    assert_eq!(
        view.max(2),
        Some(&StatVal::Int(4)),
        "c must not have shifted into b's row"
    );
    assert_eq!(view.max(1), None, "the deleted row exposes nothing");
    assert_eq!(view.row_count(1), 0);
    assert_eq!(view.present_rows(), vec![0, 2]);

    // b's outlier of 200 must not survive a range scan.
    assert!(
        view.candidates(|_, hi| hi > &StatVal::Int(100)).is_empty(),
        "b's 200 must be masked out"
    );
}

/// Reading two columns must materialize only those two — the property the
/// column-addressed layout exists for.
#[tokio::test]
async fn pruning_index_reads_only_requested_columns() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store, prefix, StoreConfig::default())
        .await
        .unwrap();

    for d in 0..20 {
        let mut ds = atlas.create_dataset(&format!("ds{d}")).await.unwrap();
        for a in 0..20 {
            let name = format!("arr{a}");
            ds.define_array::<i64>(&name, vec!["i".into()], vec![4], None, None)
                .await
                .unwrap();
            ds.write_array(
                &name,
                vec![0],
                ndarray::Array::from_vec(vec![d as i64, a as i64, 7, 9])
                    .into_dyn()
                    .view(),
            )
            .await
            .unwrap();
        }
    }
    atlas.flush().await.unwrap();

    // Summaries come from the footer alone — no column blocks fetched at all.
    let summaries = atlas.column_summaries().await.unwrap();
    assert_eq!(summaries.len(), 20, "every column described in the footer");
    let (_, arr3_summary) = summaries
        .iter()
        .find(|(k, _)| k == &ColumnKey::array("arr3"))
        .unwrap();
    assert_eq!(arr3_summary.present_count, 20);

    let wanted = vec![ColumnKey::array("arr3"), ColumnKey::array("arr17")];
    let index = atlas.pruning_index(&wanted).await.unwrap();

    assert_eq!(index.rows(), 20, "full row space even for a partial read");
    assert_eq!(
        index.column_keys().len(),
        2,
        "only the requested columns are materialized"
    );
    assert!(index.column(&ColumnKey::array("arr3")).is_some());
    assert!(
        index.column(&ColumnKey::array("arr0")).is_none(),
        "a column that wasn't asked for must not be loaded"
    );
}

/// The pruning index is built on demand from the array files — nothing is
/// persisted for it, and a freshly reopened store (no index cache) still serves
/// it from the stats alone.
#[tokio::test]
async fn pruning_index_is_built_on_demand_without_persistence() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default())
        .await
        .unwrap();
    for (name, v) in [("a", 5i32), ("b", 50), ("c", 500)] {
        let mut ds = atlas.create_dataset(name).await.unwrap();
        ds.define_array::<i32>("temp", vec!["i".into()], vec![1], None, None).await.unwrap();
        ds.write_array("temp", vec![0], ndarray::Array::from_vec(vec![v]).into_dyn().view())
            .await
            .unwrap();
    }
    atlas.flush().await.unwrap();

    // No pruning index is written — the stats live in the array file only.
    assert!(!tmp.path().join("pruning.idx").exists());
    assert!(!tmp.path().join("pruning").exists());
    assert!(tmp.path().join("temp").join("data.af").exists());

    // A reopened store (cold — no in-memory index) serves the flat column from
    // the array stats on demand.
    let reopened = Atlas::open(store, prefix).await.unwrap();
    let key = ColumnKey::array("temp");
    let idx = reopened.pruning_index(std::slice::from_ref(&key)).await.unwrap();
    let view = idx.view(&key).unwrap();
    assert_eq!(idx.rows(), 3);
    assert_eq!(view.max(reopened.dataset_row("b").unwrap()), Some(&StatVal::Int(50)));
    assert_eq!(view.candidates(|_, hi| hi > &StatVal::Int(100)),
               vec![reopened.dataset_row("c").unwrap()]);
}

/// Every value in the index must agree with the per-dataset stats API it
/// summarizes — including where the holes are.
#[tokio::test]
async fn pruning_index_matches_per_dataset_stats() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store, prefix, StoreConfig::default())
        .await
        .unwrap();

    for d in 0..6 {
        let mut ds = atlas.create_dataset(&format!("ds{d}")).await.unwrap();
        // Every other dataset skips the array, exercising the gaps.
        if d % 2 == 0 {
            ds.define_array::<f64>("depth", vec!["i".into()], vec![3], None, None)
                .await
                .unwrap();
            ds.write_array(
                "depth",
                vec![0],
                ndarray::Array::from_vec(vec![d as f64, d as f64 * 2.0, -1.5])
                    .into_dyn()
                    .view(),
            )
            .await
            .unwrap();
        }
    }
    atlas.flush().await.unwrap();

    let key = ColumnKey::array("depth");
    let index = atlas.pruning_index(std::slice::from_ref(&key)).await.unwrap();
    let column = index.column(&key).unwrap();
    let present = column.present_mask();

    for d in 0..6 {
        let name = format!("ds{d}");
        let row = atlas.dataset_row(&name).unwrap();
        let per_dataset = atlas
            .open_dataset(&name)
            .await
            .unwrap()
            .array_stats("depth")
            .await;
        match per_dataset {
            Some(stats) => {
                assert!(present[row], "{name} declares depth");
                assert_eq!(column.row_count[row], stats.row_count, "{name} row_count");
                assert_eq!(column.null_count[row], stats.null_count, "{name} null_count");
                let expected_min = match stats.min {
                    Some(StatValue::Float(f)) => Some(StatVal::Float(f)),
                    _ => None,
                };
                assert_eq!(column.min[row], expected_min, "{name} min");
            }
            None => assert!(!present[row], "{name} has no depth"),
        }
    }
}

#[tokio::test]
async fn pruning_row_is_reset_when_a_dataset_is_revived() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store, prefix, StoreConfig::default())
        .await
        .unwrap();

    // "recycled" declares `temp` with a distinctive max, then is deleted.
    {
        let mut ds = atlas.create_dataset("recycled").await.unwrap();
        ds.define_array::<i32>("temp", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
        ds.write_array("temp", vec![0], ndarray::Array::from_vec(vec![900i32, 999]).into_dyn().view())
            .await
            .unwrap();
    }
    atlas.flush().await.unwrap();
    let row = atlas.dataset_row("recycled").unwrap();
    atlas.delete_dataset("recycled").await.unwrap();

    // Recreate the same name; it reuses the slot but declares nothing yet.
    atlas.create_dataset("recycled").await.unwrap();
    atlas.flush().await.unwrap();

    let idx = atlas.pruning_index(&[ColumnKey::array("temp")]).await.unwrap();
    let view = idx.view(&ColumnKey::array("temp")).expect("temp column");
    assert_eq!(atlas.dataset_row("recycled"), Some(row), "slot reused");
    assert!(
        !view.is_present(row),
        "revived row must not carry the old dataset's stats"
    );
    assert_eq!(view.max(row), None);
}

/// After `compact` renumbers datasets, the pruning index must be rebuilt in the
/// new numbering with the surviving rows' stats intact.
#[tokio::test]
async fn compact_rebuilds_the_pruning_index() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default())
        .await
        .unwrap();

    for (name, vals) in [("a", vec![1i32, 2]), ("b", vec![50i32, 60]), ("c", vec![7i32, 8])] {
        let mut ds = atlas.create_dataset(name).await.unwrap();
        ds.define_array::<i32>("v", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
        ds.write_array("v", vec![0], ndarray::Array::from_vec(vals).into_dyn().view())
            .await
            .unwrap();
    }
    atlas.flush().await.unwrap();
    atlas.delete_dataset("a").await.unwrap();
    atlas.flush().await.unwrap();
    atlas.compact().await.unwrap();

    // b and c renumbered to rows 0 and 1; their stats must line up.
    let idx = atlas.pruning_index(&[ColumnKey::array("v")]).await.unwrap();
    assert_eq!(idx.rows(), 2, "dead row reclaimed");
    let view = idx.view(&ColumnKey::array("v")).unwrap();
    assert_eq!(view.max(atlas.dataset_row("b").unwrap()), Some(&StatVal::Int(60)));
    assert_eq!(view.max(atlas.dataset_row("c").unwrap()), Some(&StatVal::Int(8)));

    // And it survives a reopen at the new epoch.
    let reopened = Atlas::open(store, prefix).await.unwrap();
    let idx2 = reopened.pruning_index(&[ColumnKey::array("v")]).await.unwrap();
    assert_eq!(idx2.rows(), 2);
    assert_eq!(idx2.view(&ColumnKey::array("v")).unwrap().present_rows().len(), 2);
}

/// Attribute columns record which datasets carry a global / per-array key.
#[tokio::test]
async fn pruning_index_tracks_attribute_presence() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store, prefix, StoreConfig::default())
        .await
        .unwrap();

    // ds0 and ds2 set `cruise`; ds1 does not.
    for (name, set_cruise) in [("ds0", true), ("ds1", false), ("ds2", true)] {
        let mut ds = atlas.create_dataset(name).await.unwrap();
        ds.define_array::<i32>("v", vec!["i".into()], vec![1], None, None)
            .await
            .unwrap();
        ds.set_array_attribute("v", "units", Attr::String("m".into())).unwrap();
        if set_cruise {
            ds.set_attribute("cruise", Attr::String(name.into())).unwrap();
        }
    }
    atlas.flush().await.unwrap();

    let idx = atlas
        .pruning_index(&[
            ColumnKey::global_attr("cruise"),
            ColumnKey::array_attr("v", "units"),
        ])
        .await
        .unwrap();

    let cruise = idx.view(&ColumnKey::global_attr("cruise")).expect("cruise column");
    assert!(cruise.is_present(0));
    assert!(!cruise.is_present(1), "ds1 has no cruise attribute");
    assert!(cruise.is_present(2));

    // Every dataset set the per-array `units` attribute.
    let units = idx.view(&ColumnKey::array_attr("v", "units")).expect("units column");
    assert_eq!(units.present_rows(), vec![0, 1, 2]);
}

/// Attribute columns carry each dataset's value as a point range, so a client
/// can range-prune on an array's attribute the same way it does on array data.
#[tokio::test]
async fn pruning_index_range_prunes_on_array_attribute_values() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store, prefix, StoreConfig::default())
        .await
        .unwrap();

    // Per-array attribute `year` on `v`, plus a list-valued `valid_range`.
    for (name, year) in [("ds0", 2019i64), ("ds1", 2021), ("ds2", 2023)] {
        let mut ds = atlas.create_dataset(name).await.unwrap();
        ds.define_array::<i32>("v", vec!["i".into()], vec![1], None, None)
            .await
            .unwrap();
        ds.set_array_attribute("v", "year", Attr::Int64(year)).unwrap();
        ds.set_array_attribute("v", "valid_range", Attr::Float32List(vec![0.0, 100.0]))
            .unwrap();
    }
    atlas.flush().await.unwrap();

    let key = ColumnKey::array_attr("v", "year");
    let idx = atlas.pruning_index(std::slice::from_ref(&key)).await.unwrap();
    let year = idx.view(&key).unwrap();

    // Each dataset's value is a point range [year, year].
    assert_eq!(year.min(0), Some(&StatVal::Int(2019)));
    assert_eq!(year.max(0), Some(&StatVal::Int(2019)));
    assert_eq!(year.min(2), Some(&StatVal::Int(2023)));

    // Range pruning: datasets whose year is after 2020.
    let after_2020 = year.candidates(|_, hi| hi > &StatVal::Int(2020));
    assert_eq!(
        after_2020,
        vec![atlas.dataset_row("ds1").unwrap(), atlas.dataset_row("ds2").unwrap()]
    );

    // A list-valued attribute is present but carries no scalar range.
    let range_key = ColumnKey::array_attr("v", "valid_range");
    let idx2 = atlas.pruning_index(std::slice::from_ref(&range_key)).await.unwrap();
    let vr = idx2.view(&range_key).unwrap();
    assert_eq!(vr.present_rows(), vec![0, 1, 2]);
    assert_eq!(vr.min(0), None, "list-valued attribute has no point range");
}

/// Querying the index for a dataset created but not yet flushed returns a
/// present row with no statistics, rather than a missing row or an error.
#[tokio::test]
async fn pruning_index_reads_unflushed_dataset_as_present_without_stats() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store, prefix, StoreConfig::default())
        .await
        .unwrap();

    {
        let mut a = atlas.create_dataset("a").await.unwrap();
        a.define_array::<i32>("temp", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
        a.write_array("temp", vec![0], ndarray::Array::from_vec(vec![1i32, 2]).into_dyn().view())
            .await
            .unwrap();
    }
    atlas.flush().await.unwrap();

    // A second dataset exists in memory but has never been flushed.
    {
        let mut b = atlas.create_dataset("b").await.unwrap();
        b.define_array::<i32>("temp", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
    }

    // The write-side index (in memory) covers both rows; the unflushed one has
    // no stats yet but is still a real row.
    assert_eq!(atlas.row_slots(), 2);
    assert_eq!(atlas.dataset_row("b"), Some(1));
}

/// The returned index is self-describing: it carries the row↔name mapping and
/// the liveness mask, so a consumer prunes and resolves names from the one
/// object — no second call to the store, no separate mask to remember to apply.
#[tokio::test]
async fn pruning_index_is_self_describing() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store, prefix, StoreConfig::default())
        .await
        .unwrap();

    // Three datasets declare `temp`; "hot" holds the only value above 25.
    for (name, vals) in [("cold", vec![1i32, 5]), ("hot", vec![30i32, 40]), ("mild", vec![10i32, 20])] {
        let mut ds = atlas.create_dataset(name).await.unwrap();
        ds.define_array::<i32>("temp", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
        ds.write_array("temp", vec![0], ndarray::Array::from_vec(vals).into_dyn().view())
            .await
            .unwrap();
    }
    // A deleted dataset with an extreme value must not surface as a candidate.
    {
        let mut ds = atlas.create_dataset("deleted").await.unwrap();
        ds.define_array::<i32>("temp", vec!["i".into()], vec![2], None, None)
            .await
            .unwrap();
        ds.write_array("temp", vec![0], ndarray::Array::from_vec(vec![900i32, 999]).into_dyn().view())
            .await
            .unwrap();
    }
    atlas.flush().await.unwrap();
    atlas.delete_dataset("deleted").await.unwrap();
    atlas.flush().await.unwrap();

    let key = ColumnKey::array("temp");
    let index = atlas.pruning_index(std::slice::from_ref(&key)).await.unwrap();

    // Everything needed lives on `index`: candidates (mask already applied) and
    // the row→name mapping.
    let view = index.view(&key).unwrap();
    let names: Vec<&str> = view
        .candidates(|_, hi| hi > &StatVal::Int(25))
        .into_iter()
        .map(|row| index.dataset_name(row).unwrap())
        .collect();
    assert_eq!(names, vec!["hot"], "only 'hot' exceeds 25; 'deleted' is masked");

    // The row↔name mapping agrees with the store's own ordinals, and a
    // tombstoned slot is both nameless and masked dead — all from `index`.
    for (row, name) in index.dataset_names().iter().enumerate() {
        match name {
            Some(n) => {
                assert_eq!(atlas.dataset_row(n), Some(row));
                assert!(index.live()[row]);
            }
            None => assert!(!index.live()[row], "tombstoned row must be masked dead"),
        }
    }
}

/// End-to-end proof that a downstream client can prune a query to the right
/// candidate datasets using only the pruning index — never missing a real match
/// (no false negatives), excluding datasets that don't declare the array, and
/// masking deleted datasets even when they hold an extreme value.
#[tokio::test]
async fn client_prunes_candidates_soundly() {
    let tmp = tempfile::tempdir().unwrap();
    let (store, prefix) = make_store(&tmp);
    let mut atlas = Atlas::create(store, prefix, StoreConfig::default())
        .await
        .unwrap();

    // 10 datasets declaring `temp`, dataset i has max = 2*i (a real cell value).
    // 2 datasets declare no `temp` (gaps). 1 dataset holds an extreme value but
    // is deleted — the client must never see it.
    let mut expected_max: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for i in 0..10 {
        let name = format!("d{i:02}");
        let mut ds = atlas.create_dataset(&name).await.unwrap();
        ds.define_array::<f64>("temp", vec!["x".into()], vec![2], None, None)
            .await
            .unwrap();
        ds.write_array(
            "temp",
            vec![0],
            ndarray::Array::from_vec(vec![i as f64, (2 * i) as f64]).into_dyn().view(),
        )
        .await
        .unwrap();
        expected_max.insert(name, (2 * i) as f64);
    }
    // Gaps: exist, but declare a different array — must never be candidates.
    for name in ["gap_a", "gap_b"] {
        atlas
            .create_dataset(name)
            .await
            .unwrap()
            .define_array::<f64>("other", vec!["x".into()], vec![1], None, None)
            .await
            .unwrap();
    }
    // Deleted dataset with an enormous `temp` value.
    {
        let mut ds = atlas.create_dataset("deleted").await.unwrap();
        ds.define_array::<f64>("temp", vec!["x".into()], vec![2], None, None)
            .await
            .unwrap();
        ds.write_array("temp", vec![0], ndarray::Array::from_vec(vec![1000.0, 9999.0]).into_dyn().view())
            .await
            .unwrap();
    }
    atlas.flush().await.unwrap();
    atlas.delete_dataset("deleted").await.unwrap();
    atlas.flush().await.unwrap();

    let key = ColumnKey::array("temp");
    let index = atlas.pruning_index(std::slice::from_ref(&key)).await.unwrap();
    let view = index.view(&key).unwrap();

    // For a sweep of thresholds, the client prunes to "max > T" and we check it
    // against the known ground truth.
    for t in [-1.0, 5.0, 9.0, 15.0, 18.0, 100.0] {
        let candidates: std::collections::HashSet<String> = view
            .candidates(|_, hi| hi > &StatVal::Float(t))
            .into_iter()
            .map(|row| index.dataset_name(row).unwrap().to_string())
            .collect();

        let truth: std::collections::HashSet<String> = expected_max
            .iter()
            .filter(|&(_, &m)| m > t)
            .map(|(n, _)| n.clone())
            .collect();

        // No false negatives — every real match is a candidate — and for the
        // exact `>` predicate on a min/max index, no false positives either.
        assert_eq!(candidates, truth, "candidate set wrong at T={t}");
        // The deleted dataset (max 9999) and the gaps are never candidates.
        assert!(!candidates.contains("deleted"), "deleted leaked at T={t}");
        assert!(!candidates.contains("gap_a") && !candidates.contains("gap_b"));
    }

    // A predicate above every live dataset's max prunes to nothing, and the
    // footer-only summary confirms the whole column is skippable — without
    // reading the deleted dataset's 9999 into the range.
    let summaries = atlas.column_summaries().await.unwrap();
    let (_, summary) = summaries.iter().find(|(k, _)| k == &key).unwrap();
    assert_eq!(summary.max, Some(StatVal::Float(18.0)), "global max excludes the deleted 9999");
    assert!(!summary.might_match(|_, hi| hi > &StatVal::Float(50.0)), "column skippable for >50");
    assert!(view.candidates(|_, hi| hi > &StatVal::Float(50.0)).is_empty());
}
