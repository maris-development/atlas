//! Reads a collection that Python wrote, and checks the values.
//!
//! Python writes array data, and cannot read it back. A pytest run alone
//! therefore cannot prove the xarray layer wrote the bytes it meant. This test
//! closes that loop. `atlas-python/tests/make_fixture.py` writes the committed
//! `tests/fixtures/from_python/`, and the Rust reader asserts every value
//! here.
//!
//! The fixture builds the way `atlas create` builds any collection, from a
//! directory of NetCDF files. This test therefore covers the round trip
//! through NetCDF too, which is the only ingest route Python offers.
//!
//! Regenerate after a deliberate change to the write path:
//!
//! ```text
//! python atlas-python/tests/make_fixture.py
//! ```
//!
//! `build_dataset()` in that script defines the fixture. Keep the two in step.

use std::path::{Path, PathBuf};

use atlas::{Atlas, Attr, DType, FillValue, StatValue, TimestampNs};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/from_python")
}

#[tokio::test]
async fn the_python_written_fixture_holds_the_values_python_wrote() {
    let atlas = Atlas::open_path(fixture_dir()).await.unwrap();
    assert_eq!(atlas.list_datasets(), vec!["grid.nc", "grid_copy.nc"]);

    let grid = atlas.dataset("grid.nc").unwrap();
    // Coordinates first, then data variables, in the order xarray gives.
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
    let layout = grid.array_layout("temperature").await.unwrap();
    assert_eq!(layout.chunk_shape(), vec![2, 3]);
    assert_eq!(layout.dimension_names(), vec!["lat", "lon"]);

    // A window across all four chunks.
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
    assert_eq!(
        grid.array_layout("counts").await.unwrap().fill_value(),
        None
    );

    // Object strings become variable-length atlas strings.
    let label = grid
        .read_array::<String>("label", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(label[[0]], "alpha");
    assert_eq!(label[[3]], "delta");

    // datetime64[ns] becomes TimestampNs, still in nanoseconds from the
    // epoch.
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
    let grid = atlas.dataset("grid.nc").unwrap();

    assert_eq!(
        *grid.array_meta("temperature").unwrap().dtype(),
        DType::Float32
    );
    assert_eq!(*grid.array_meta("counts").unwrap().dtype(), DType::Int64);
    assert_eq!(*grid.array_meta("label").unwrap().dtype(), DType::String);
    assert_eq!(
        *grid.array_meta("observed").unwrap().dtype(),
        DType::TimestampNs
    );

    // Dataset attributes, with the coordinate marker the xarray layer writes
    // and the JSON-encoded list.
    assert_eq!(
        grid.get_attribute("month").await.unwrap(),
        Some(Attr::Int64(1))
    );
    assert_eq!(
        grid.get_attribute("station").await.unwrap(),
        Some(Attr::String("KNMI".into()))
    );
    assert_eq!(
        grid.get_attribute("_pyatlas_coords").await.unwrap(),
        Some(Attr::String("[\"lat\", \"lon\"]".into()))
    );
    assert_eq!(
        grid.get_attribute("bounds").await.unwrap(),
        Some(Attr::String("json:[1.0, 2.0]".into()))
    );

    // Each per-variable attribute lands on its own array.
    assert_eq!(
        grid.get_array_attribute("temperature", "units")
            .await
            .unwrap(),
        Some(Attr::String("celsius".into()))
    );
    assert_eq!(
        grid.get_array_attribute("temperature", "long_name")
            .await
            .unwrap(),
        Some(Attr::String("surface temperature".into()))
    );
    assert!(grid.array_attributes("counts").await.unwrap().is_empty());

    // A float defaults to a NaN fill. A datetime defaults to NaT.
    assert!(matches!(
        grid.array_layout("temperature").await.unwrap().fill_value(),
        Some(FillValue::Float(f)) if f.is_nan()
    ));
    assert_eq!(
        grid.array_layout("observed").await.unwrap().fill_value(),
        Some(&FillValue::TimestampNs(i64::MIN))
    );
    assert_eq!(
        grid.array_layout("label").await.unwrap().fill_value(),
        Some(&FillValue::String(String::new()))
    );
}

#[tokio::test]
async fn the_two_python_datasets_share_one_interned_schema() {
    let atlas = Atlas::open_path(fixture_dir()).await.unwrap();
    let grid = atlas.dataset("grid.nc").unwrap();
    let copy = atlas.dataset("grid_copy.nc").unwrap();

    // Two equal files under one `chunks=` setting produce both datasets. They
    // declare the same arrays, so the footer stores that schema once.
    assert_eq!(grid.list_arrays(), copy.list_arrays());
    assert_eq!(grid.schema(), copy.schema());
    assert_eq!(
        grid.attributes().await.unwrap(),
        copy.attributes().await.unwrap()
    );

    // Both datasets read their counts out of the one `counts` segment, each
    // under its own name.
    let copy_counts = copy
        .read_array::<i64>("counts", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(copy_counts.as_slice().unwrap(), &[10, 20, 30, 40]);
}

#[tokio::test]
async fn the_python_written_fixture_carries_statistics() {
    let atlas = Atlas::open_path(fixture_dir()).await.unwrap();
    let grid = atlas.dataset("grid.nc").unwrap();

    // The staging step computes these. The footer then stores them.
    let temperature = grid.array_stats("temperature").await.unwrap().unwrap();
    assert_eq!(temperature.row_count, 24);
    assert_eq!(temperature.null_count, 0);
    assert_eq!(temperature.min, Some(StatValue::Float(0.0)));
    assert_eq!(temperature.max, Some(StatValue::Float(23.0)));

    let counts = grid.array_stats("counts").await.unwrap().unwrap();
    assert_eq!(counts.min, Some(StatValue::Int(10)));
    assert_eq!(counts.max, Some(StatValue::Int(40)));

    // A string compares lexicographically, as raw bytes.
    let label = grid.array_stats("label").await.unwrap().unwrap();
    assert_eq!(label.min, Some(StatValue::Bytes(b"alpha".to_vec())));
    assert_eq!(label.max, Some(StatValue::Bytes(b"gamma".to_vec())));

    // A timestamp keeps its own statistic type.
    let observed = grid.array_stats("observed").await.unwrap().unwrap();
    let jan_1_2024 = 1_704_067_200_000_000_000i64;
    assert_eq!(observed.min, Some(StatValue::TimestampNs(jan_1_2024)));
    assert_eq!(
        observed.max,
        Some(StatValue::TimestampNs(jan_1_2024 + 3 * 86_400_000_000_000))
    );
}
