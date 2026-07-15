use std::sync::Arc;

use atlas::{Atlas, Attr, DType, StatValue, StoreConfig, TypeMismatchPolicy};
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
