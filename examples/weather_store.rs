//! A realistic collection on an object store, read lazily.
//!
//! Monthly weather grids and a station table. They go to an in-memory
//! `ObjectStore` under a prefix. Replace `InMemory` with `AmazonS3`, and
//! nothing else changes. The reader issues range reads only.
//!
//! The example is about what it does *not* fetch. An open reads the footer.
//! The dataset list, the schemas, and the attributes all come from that. A
//! window out of one grid fetches the chunks it overlaps, and no more.
//!
//! Run with: `cargo run --example weather_store`

use std::sync::Arc;

use atlas::{Atlas, AtlasWriter, Attr, FillValue, TimestampNs, WriterConfig};
use ndarray::{Array1, ArrayD, IxDyn};
use object_store::path::Path as OsPath;
use object_store::{ObjectStore, ObjectStoreExt, memory::InMemory};

const LAT: usize = 32;
const LON: usize = 64;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let prefix = OsPath::from("weather/2024");

    // ── Write ────────────────────────────────────────────────────────

    println!("=== Writing ===");
    let w =
        AtlasWriter::create(Arc::clone(&store), prefix.clone(), WriterConfig::default()).await?;

    for (month, name) in [(1u8, "jan"), (2, "feb"), (3, "mar")] {
        let mut ds = w.add_dataset(name).await?;

        ds.define_array::<f32>(
            "temperature",
            vec!["lat".into(), "lon".into()],
            vec![LAT, LON],
            Some(vec![8, 16]), // 4 x 4 = 16 chunks
            Some(FillValue::Float(f64::NAN)),
        )
        .await?;
        let grid = ArrayD::from_shape_fn(IxDyn(&[LAT, LON]), |i| {
            (month as f32) + (i[0] as f32) * 0.1 + (i[1] as f32) * 0.01
        });
        ds.write_array("temperature", vec![0, 0], grid.view())
            .await?;

        ds.define_array::<f32>(
            "precipitation",
            vec!["lat".into(), "lon".into()],
            vec![LAT, LON],
            Some(vec![8, 16]),
            Some(FillValue::Float(0.0)),
        )
        .await?;
        // The write covers the northern band only. The rest reads back as the
        // fill.
        let band = ArrayD::from_shape_fn(IxDyn(&[8, LON]), |i| (i[1] % 7) as f32);
        ds.write_array("precipitation", vec![0, 0], band.view())
            .await?;

        ds.set_attribute("month", Attr::Int64(month as i64));
        ds.set_attribute(
            "title",
            Attr::String(format!("2024-{month:02} monthly mean")),
        );
        ds.set_array_attribute("temperature", "units", Attr::String("celsius".into()))?;
        ds.set_array_attribute("precipitation", "units", Attr::String("mm".into()))?;
        ds.finish().await?;
        println!("  wrote '{name}'");
    }

    // A dataset with a different schema. This is the station table.
    {
        let mut ds = w.add_dataset("stations").await?;
        ds.define_array::<String>("name", vec!["station".into()], vec![3], None, None)
            .await?;
        let names = Array1::from_vec(vec![
            "Vlissingen".to_string(),
            "Den Helder".to_string(),
            "Terschelling".to_string(),
        ])
        .into_dyn();
        ds.write_array("name", vec![0], names.view()).await?;

        ds.define_array::<TimestampNs>(
            "installed",
            vec!["station".into()],
            vec![3],
            None,
            Some(FillValue::TimestampNs(i64::MIN)),
        )
        .await?;
        let installed = Array1::from_vec(vec![
            TimestampNs(1_000_000_000_000_000_000),
            TimestampNs(1_100_000_000_000_000_000),
            TimestampNs(1_200_000_000_000_000_000),
        ])
        .into_dyn();
        ds.write_array("installed", vec![0], installed.view())
            .await?;
        ds.finish().await?;
        println!("  wrote 'stations'");
    }

    w.finish().await?;

    let size = store.head(&prefix.clone().join("data.atlas")).await?.size;
    println!("  {} bytes at {prefix}/data.atlas\n", size);

    // ── Read ─────────────────────────────────────────────────────────

    println!("=== Reading ===");
    let atlas = Atlas::open(Arc::clone(&store), prefix.clone()).await?;
    println!("  datasets: {:?}", atlas.list_datasets());
    println!("  arrays:   {:?}", atlas.list_arrays());

    // Names and types for every dataset, and still no array byte.
    for name in atlas.list_datasets() {
        let ds = atlas.dataset(&name)?;
        let arrays: Vec<String> = ds
            .schema()
            .iter()
            .map(|a| format!("{}:{:?}", a.name(), a.dtype()))
            .collect();
        println!("  {name}: {}", arrays.join(", "));
    }
    println!();

    // A 2x2 window out of a 32x64 grid. This fetches one chunk, not 16.
    let jan = atlas.dataset("jan")?;
    let window = jan
        .read_array::<f32>("temperature", vec![10, 20], vec![2, 2])
        .await?;
    println!(
        "  jan/temperature[10..12, 20..22] = {:?}",
        window.as_slice().unwrap()
    );

    // A region nobody wrote comes back as the fill value. It stores no bytes.
    let dry = jan
        .read_array::<f32>("precipitation", vec![20, 0], vec![1, 4])
        .await?;
    println!(
        "  jan/precipitation[20, 0..4] = {:?} (never written)",
        dry.as_slice().unwrap()
    );

    let stations = atlas.dataset("stations")?;
    let names = stations
        .read_array::<String>("name", vec![], vec![])
        .await?;
    println!("  stations: {:?}", names.as_slice().unwrap());

    // ── Delete ───────────────────────────────────────────────────────

    println!("\n=== Deleting 'feb' ===");
    atlas.delete_dataset("feb").await?;
    let after = Atlas::open(store, prefix).await?;
    println!("  datasets now: {:?}", after.list_datasets());

    Ok(())
}
