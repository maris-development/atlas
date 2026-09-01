//! Many small datasets in one file.
//!
//! A fleet of sensors, with one dataset each. A directory-per-dataset layout
//! gives hundreds of small objects. This gives one `data.atlas`. To list the
//! fleet costs one range read, whatever the number of sensors.
//!
//! The example also shows what interning buys. Every sensor declares the same
//! two arrays. The footer therefore stores that schema once, and each dataset
//! points at it.
//!
//! Run with: `cargo run --example sensor_fleet`

use atlas::{Atlas, AtlasWriter, Attr, Codec, WriterConfig};
use ndarray::Array1;

const N_SENSORS: usize = 24;
const N_READINGS: usize = 24; // one per hour

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;

    // LZ4 decompresses faster than Zstd, and gives larger files. That suits a
    // read-heavy collection. Each block records its codec, so nothing tells
    // the reader which one the write used.
    let w = AtlasWriter::create_path(
        tmp.path(),
        WriterConfig {
            codec: Codec::Lz4,
            ..Default::default()
        },
    )
    .await?;

    println!("=== Writing {N_SENSORS} sensors ===");
    for sensor in 0..N_SENSORS {
        let mut ds = w.add_dataset(&format!("sensor-{sensor:03}")).await?;

        ds.define_array::<f64>(
            "readings",
            vec!["hour".into()],
            vec![N_READINGS],
            None,
            None,
        )
        .await?;
        // A rough daily curve, with one offset per sensor.
        let readings = Array1::from_shape_fn(N_READINGS, |h| {
            20.0 + (h as f64 / 4.0).sin() * 5.0 + sensor as f64 * 0.1
        })
        .into_dyn();
        ds.write_array("readings", vec![0], readings.view()).await?;

        ds.define_array::<i64>(
            "timestamps",
            vec!["hour".into()],
            vec![N_READINGS],
            None,
            None,
        )
        .await?;
        let base = 1_700_000_000i64;
        let timestamps =
            Array1::from_shape_fn(N_READINGS, |h| base + (h as i64) * 3_600).into_dyn();
        ds.write_array("timestamps", vec![0], timestamps.view())
            .await?;

        ds.set_attribute("sensor_id", Attr::Int64(sensor as i64));
        ds.set_attribute(
            "site",
            Attr::String(if sensor % 2 == 0 { "north" } else { "south" }.into()),
        );
        ds.set_array_attribute("readings", "units", Attr::String("celsius".into()))?;
        ds.finish().await?;
    }
    w.finish().await?;

    let size = std::fs::metadata(tmp.path().join("data.atlas"))?.len();
    println!("  one file, {size} bytes, {N_SENSORS} datasets\n");

    // ── Scanning the fleet without reading any data ──────────────────

    let atlas = Atlas::open_path(tmp.path()).await?;
    println!("=== Metadata scan (no array bytes fetched) ===");
    println!("  datasets: {}", atlas.dataset_count());
    println!("  distinct arrays: {:?}", atlas.list_arrays());

    // Attributes come from the footer. To filter the fleet by site is
    // therefore free.
    let mut north = Vec::new();
    for name in atlas.list_datasets() {
        let ds = atlas.dataset(&name)?;
        if ds.get_attribute("site") == Some(Attr::String("north".into())) {
            north.push(name);
        }
    }
    println!("  {} sensors at the north site", north.len());

    // Every sensor shares one interned schema.
    let first = atlas.dataset("sensor-000")?;
    let last = atlas.dataset(&format!("sensor-{:03}", N_SENSORS - 1))?;
    println!(
        "  schemas identical across the fleet: {}\n",
        first.schema() == last.schema()
    );

    // ── Reading one sensor ───────────────────────────────────────────

    println!("=== Reading one sensor ===");
    let ds = atlas.dataset("sensor-007")?;
    let readings = ds.read_array::<f64>("readings", vec![], vec![]).await?;
    let peak = readings.iter().cloned().fold(f64::MIN, f64::max);
    println!("  sensor-007 peak reading: {peak:.2}");

    // Only the last six hours. That is a sub-region of the array.
    let tail = ds
        .read_array::<f64>("readings", vec![N_READINGS - 6], vec![6])
        .await?;
    println!("  last six hours: {:?}", tail.as_slice().unwrap());

    Ok(())
}
