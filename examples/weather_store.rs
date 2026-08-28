//! A realistic collection on an object store, read lazily.
//!
//! Monthly weather grids plus a station table, written to an in-memory
//! `ObjectStore` under a prefix. Swap `InMemory` for `AmazonS3` and nothing
//! else changes: the reader only ever issues range reads.
//!
//! The point of the example is what is *not* fetched. Opening reads the footer.
//! Listing, schemas, and attributes come from that. A window out of one grid
//! fetches the chunks it overlaps and nothing else.
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
    let mut w =
        AtlasWriter::create(Arc::clone(&store), prefix.clone(), WriterConfig::default()).await?;

    for (month, name) in [(1u8, "jan"), (2, "feb"), (3, "mar")] {
        let mut ds = w.add_dataset(name)?;

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
        ds.write_array("temperature", vec![0, 0], grid.view()).await?;

        ds.define_array::<f32>(
            "precipitation",
            vec!["lat".into(), "lon".into()],
            vec![LAT, LON],
            Some(vec![8, 16]),
            Some(FillValue::Float(0.0)),
        )
        .await?;
        // Only the northern band is written; the rest reads back as the fill.
        let band = ArrayD::from_shape_fn(IxDyn(&[8, LON]), |i| (i[1] % 7) as f32);
        ds.write_array("precipitation", vec![0, 0], band.view())
            .await?;

        ds.set_attribute("month", Attr::Int64(month as i64));
        ds.set_attribute("title", Attr::String(format!("2024-{month:02} monthly mean")));
        ds.set_array_attribute("temperature", "units", Attr::String("celsius".into()))?;
        ds.set_array_attribute("precipitation", "units", Attr::String("mm".into()))?;
        ds.finish().await?;
        println!("  wrote '{name}'");
    }

    // A dataset with a different schema: the station table.
    {
        let mut ds = w.add_dataset("stations")?;
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
        ds.write_array("installed", vec![0], installed.view()).await?;
        ds.finish().await?;
        println!("  wrote 'stations'");
    }

    w.finish().await?;

    let size = store
        .head(&prefix.clone().join("data.atlas"))
        .await?
        .size;
    println!("  {} bytes at {prefix}/data.atlas\n", size);

    // ── Read ─────────────────────────────────────────────────────────

    println!("=== Reading ===");
    let atlas = Atlas::open(Arc::clone(&store), prefix.clone()).await?;
    println!("  datasets: {:?}", atlas.list_datasets());
    println!("  arrays:   {:?}", atlas.list_arrays());

    // Metadata for every dataset, still without fetching a single array byte.
    for name in atlas.list_datasets() {
        let ds = atlas.dataset(&name)?;
        let shapes: Vec<String> = ds
            .schema()
            .arrays
            .iter()
            .map(|(array, schema)| format!("{array}{:?}", schema.shape))
            .collect();
        println!("  {name}: {}", shapes.join(", "));
    }
    println!();

    // A 2x2 window out of a 32x64 grid. One chunk is fetched, not 16.
    let jan = atlas.dataset("jan")?;
    let window = jan
        .read_array::<f32>("temperature", vec![10, 20], vec![2, 2])
        .await?;
    println!("  jan/temperature[10..12, 20..22] = {:?}", window.as_slice().unwrap());

    // Unwritten regions come back as the fill value, with no stored bytes.
    let dry = jan
        .read_array::<f32>("precipitation", vec![20, 0], vec![1, 4])
        .await?;
    println!("  jan/precipitation[20, 0..4] = {:?} (never written)", dry.as_slice().unwrap());

    let stations = atlas.dataset("stations")?;
    let names = stations.read_array::<String>("name", vec![], vec![]).await?;
    println!("  stations: {:?}", names.as_slice().unwrap());

    // ── Delete ───────────────────────────────────────────────────────

    println!("\n=== Deleting 'feb' ===");
    atlas.delete_dataset("feb").await?;
    let after = Atlas::open(store, prefix).await?;
    println!("  datasets now: {:?}", after.list_datasets());

    Ok(())
}
