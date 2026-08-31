//! Reads a collection that Python wrote, and checks the values.
//!
//! Python can write array data but not read it back, so a pytest run cannot on
//! its own prove that the bytes the xarray layer wrote are the bytes it meant.
//! This test closes that loop: `atlas-python/tests/make_fixture.py` writes
//! `tests/fixtures/from_python/`, which is committed, and the Rust reader
//! asserts every value here.
//!
//! The fixture is built the way `atlas create` builds any collection: from a
//! directory of NetCDF files. So this also covers the round trip through
//! NetCDF, which is the only ingest route Python offers.
//!
//! Regenerate after an intentional change to the write path:
//!
//! ```text
//! python atlas-python/tests/make_fixture.py
//! ```
//!
//! The fixture's contents are defined by `build_dataset()` in that script. Keep
//! the two in sync.

use std::path::{Path, PathBuf};

use atlas::{Atlas, Attr, DType, FillValue, StatValue, TimestampNs};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/from_python")
}

#[tokio::test]
async fn the_python_written_fixture_holds_the_values_python_wrote() {
    let atlas = Atlas::open_path(fixture_dir()).await.unwrap();
    assert_eq!(atlas.list_datasets(), vec!["grid", "grid_copy"]);

    let grid = atlas.dataset("grid").unwrap();
    // Coordinates first, then data variables, in xarray's order.
    assert_eq!(
        grid.list_arrays(),
        vec!["lat", "lon", "temperature", "counts", "label", "observed"]
    );

    // float32, chunked 2x3 by the explicit `chunks=` argument.
    let temperature = grid
        .read_array::<f32>("temperature", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(temperature.shape(), &[4, 6]);
    for row in 0..4 {
        for col in 0..6 {
            assert_eq!(temperature[[row, col]], (row * 6 + col) as f32);
        }
    }
    let meta = grid.array_meta("temperature").unwrap();
    assert_eq!(meta.chunk_shape, vec![2, 3]);
    assert_eq!(meta.dimension_names, vec!["lat", "lon"]);

    // A window straddling all four chunks.
    let window = grid
        .read_array::<f32>("temperature", vec![1, 2], vec![2, 2])
        .await
        .unwrap();
    assert_eq!(window[[0, 0]], 8.0);
    assert_eq!(window[[1, 1]], 15.0);

    // int64 with no fill.
    let counts = grid
        .read_array::<i64>("counts", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(counts.as_slice().unwrap(), &[10, 20, 30, 40]);
    assert_eq!(grid.array_fill_value("counts"), None);

    // Object strings became variable-length atlas strings.
    let label = grid
        .read_array::<String>("label", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(label[[0]], "alpha");
    assert_eq!(label[[3]], "delta");

    // datetime64[ns] became TimestampNs, still in nanoseconds since the epoch.
    let observed = grid
        .read_array::<TimestampNs>("observed", vec![], vec![])
        .await
        .unwrap();
    let day = 86_400_000_000_000i64;
    let jan_1_2024 = 1_704_067_200_000_000_000i64;
    assert_eq!(observed[[0]].0, jan_1_2024);
    assert_eq!(observed[[3]].0, jan_1_2024 + 3 * day);

    // Coordinates are ordinary arrays.
    let lat = grid.read_array::<f64>("lat", vec![], vec![]).await.unwrap();
    assert_eq!(lat.as_slice().unwrap(), &[10.0, 20.0, 30.0, 40.0]);
}

#[tokio::test]
async fn the_python_written_fixture_carries_its_metadata() {
    let atlas = Atlas::open_path(fixture_dir()).await.unwrap();
    let grid = atlas.dataset("grid").unwrap();

    assert_eq!(
        grid.array_meta("temperature").unwrap().dtype,
        DType::Float32
    );
    assert_eq!(grid.array_meta("counts").unwrap().dtype, DType::Int64);
    assert_eq!(grid.array_meta("label").unwrap().dtype, DType::String);
    assert_eq!(
        grid.array_meta("observed").unwrap().dtype,
        DType::TimestampNs
    );

    // Dataset attributes, including the coordinate marker the xarray layer
    // writes and the JSON-encoded list.
    assert_eq!(grid.get_attribute("month"), Some(Attr::Int64(1)));
    assert_eq!(
        grid.get_attribute("station"),
        Some(Attr::String("KNMI".into()))
    );
    assert_eq!(
        grid.get_attribute("_pyatlas_coords"),
        Some(Attr::String("[\"lat\", \"lon\"]".into()))
    );
    assert_eq!(
        grid.get_attribute("bounds"),
        Some(Attr::String("json:[1.0, 2.0]".into()))
    );

    // Per-variable attributes landed on their own arrays.
    assert_eq!(
        grid.get_array_attribute("temperature", "units"),
        Some(Attr::String("celsius".into()))
    );
    assert_eq!(
        grid.get_array_attribute("temperature", "long_name"),
        Some(Attr::String("surface temperature".into()))
    );
    assert!(grid.array_attributes("counts").is_empty());

    // Floats default to a NaN fill; datetimes to NaT.
    assert!(matches!(
        grid.array_fill_value("temperature"),
        Some(FillValue::Float(f)) if f.is_nan()
    ));
    assert_eq!(
        grid.array_fill_value("observed"),
        Some(FillValue::TimestampNs(i64::MIN))
    );
    assert_eq!(
        grid.array_fill_value("label"),
        Some(FillValue::String(String::new()))
    );
}

#[tokio::test]
async fn the_two_python_datasets_share_one_interned_schema() {
    let atlas = Atlas::open_path(fixture_dir()).await.unwrap();
    let grid = atlas.dataset("grid").unwrap();
    let copy = atlas.dataset("grid_copy").unwrap();

    // Both were written from identical files under one `chunks=` setting, so
    // they declare the same arrays and the footer stores that schema once.
    assert_eq!(grid.list_arrays(), copy.list_arrays());
    assert_eq!(grid.schema(), copy.schema());
    assert_eq!(grid.attributes(), copy.attributes());

    // The copy's data is the same, and it lives in its own segment.
    assert_ne!(grid.segment_range(), copy.segment_range());
    let copy_counts = copy
        .read_array::<i64>("counts", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(copy_counts.as_slice().unwrap(), &[10, 20, 30, 40]);
}

#[tokio::test]
async fn the_python_written_fixture_carries_statistics() {
    let atlas = Atlas::open_path(fixture_dir()).await.unwrap();
    let grid = atlas.dataset("grid").unwrap();

    // Computed while the dataset was staged, then stored in the footer.
    let temperature = grid.array_stats("temperature").unwrap();
    assert_eq!(temperature.row_count, 24);
    assert_eq!(temperature.null_count, 0);
    assert_eq!(temperature.min, Some(StatValue::Float(0.0)));
    assert_eq!(temperature.max, Some(StatValue::Float(23.0)));

    let counts = grid.array_stats("counts").unwrap();
    assert_eq!(counts.min, Some(StatValue::Int(10)));
    assert_eq!(counts.max, Some(StatValue::Int(40)));

    // Strings compare lexicographically, as raw bytes.
    let label = grid.array_stats("label").unwrap();
    assert_eq!(label.min, Some(StatValue::Bytes(b"alpha".to_vec())));
    assert_eq!(label.max, Some(StatValue::Bytes(b"gamma".to_vec())));

    // Timestamps keep their own statistic type.
    let observed = grid.array_stats("observed").unwrap();
    let jan_1_2024 = 1_704_067_200_000_000_000i64;
    assert_eq!(observed.min, Some(StatValue::TimestampNs(jan_1_2024)));
    assert_eq!(
        observed.max,
        Some(StatValue::TimestampNs(jan_1_2024 + 3 * 86_400_000_000_000))
    );
}
