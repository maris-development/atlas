//! Demonstrates how dozens of datasets can share the same physical array files.
//!
//! A fleet of sensors each record `readings` (f64) and `timestamps` (i64).
//! Every dataset writes to the same two physical files — `readings/data.af` and
//! `timestamps/data.af` — keyed by dataset name inside each file. The store
//! holds only 2 physical arrays regardless of how many sensors exist.
//!
//! This example uses `Codec::Lz4` — a good choice for read-heavy analytics
//! workloads because LZ4 decompresses faster than Zstd at the cost of slightly
//! larger files. The codec is persisted in `atlas.json`, so `open` picks
//! it up automatically without any extra argument.
//!
//! After writing, we scan stats across all sensors to find the one with the
//! highest peak reading without loading any raw data.

use std::sync::Arc;

use atlas::{Atlas, Attr, Codec, StatValue, StoreConfig};
use ndarray::Array1;
use object_store::{local::LocalFileSystem, path::Path};

const N_SENSORS: usize = 8;
const N_READINGS: usize = 24; // one per hour

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let store: Arc<dyn object_store::ObjectStore> = Arc::new(LocalFileSystem::new());
    let prefix = Path::from_absolute_path(tmp.path())?;

    // LZ4 decompresses faster than Zstd — well suited for reading many dataset
    // entries in a tight stats-scan loop. The codec is stored in atlas.json
    // so Atlas::open() restores it without any extra argument.
    let mut s = Atlas::create(
        store.clone(),
        prefix.clone(),
        StoreConfig {
            codec: Codec::Lz4,
            ..Default::default()
        },
    )
    .await?;

    // ── Write one dataset per sensor ──────────────────────────────────────────

    let base_ts: i64 = 1_700_000_000;

    for i in 0..N_SENSORS {
        let name = format!("sensor_{i:02}");
        let mut ds = s.create_dataset(&name).await?;

        // readings: f64, one reading per hour
        ds.define_array::<f64>(
            "readings",
            vec!["hour".into()],
            vec![N_READINGS],
            None,
            None,
        )
        .await?;

        // Give each sensor a distinct baseline so stats differ across the fleet
        let baseline = 20.0 + i as f64 * 3.0;
        let values: Vec<f64> = (0..N_READINGS)
            .map(|h| baseline + (h as f64 * 0.5).sin() * 5.0)
            .collect();
        let arr = Array1::from_vec(values).into_dyn();
        ds.write_array("readings", vec![0], arr.view()).await?;

        // timestamps: i64, shared physical file with all other sensors
        ds.define_array::<i64>(
            "timestamps",
            vec!["hour".into()],
            vec![N_READINGS],
            None,
            None,
        )
        .await?;

        let ts = Array1::from_iter((0..N_READINGS as i64).map(|h| base_ts + h * 3600)).into_dyn();
        ds.write_array("timestamps", vec![0], ts.view()).await?;

        ds.set_attribute("sensor_id", Attr::Int64(i as i64));
        ds.set_attribute("unit", Attr::String("°C".into()));
    }
    s.flush().await?;

    // ── Two physical files, N_SENSORS logical datasets ────────────────────────

    let datasets = s.list_datasets().len();
    let arrays = s.list_arrays();
    println!("Logical datasets : {datasets}");
    println!("Physical arrays  : {:?}", arrays);
    println!("  → {N_SENSORS} datasets share {} files", arrays.len());

    // ── Reopen and scan stats without loading raw data ────────────────────────

    println!("\n─── Per-sensor stats (from stats file, no raw I/O) ────────────");

    // Codec is read from atlas.json — no StoreConfig needed on reopen.
    let s2 = Atlas::open(store, prefix).await?;

    let mut peak_sensor = String::new();
    let mut peak_max = f64::NEG_INFINITY;
    let mut peak_min = f64::INFINITY;

    let mut dataset_names: Vec<String> = s2.list_datasets().iter().map(|s| s.to_string()).collect();
    dataset_names.sort();

    for name in &dataset_names {
        let ds = s2.open_dataset(name).await?;
        let stats = ds.array_stats("readings").await.unwrap();

        let (lo, hi) = match (&stats.min, &stats.max) {
            (Some(StatValue::Float(lo)), Some(StatValue::Float(hi))) => (*lo, *hi),
            _ => continue,
        };

        println!(
            "  {name}: min={lo:.2}  max={hi:.2}  rows={}",
            stats.row_count
        );

        if hi > peak_max {
            peak_max = hi;
            peak_min = lo;
            peak_sensor = name.clone();
        }
    }

    println!("\nHighest peak: {peak_sensor}  max={peak_max:.2}  min={peak_min:.2}");

    // ── Verify the timestamps array is the same across all sensors ────────────

    println!("\n─── Timestamp consistency check ───────────────────────────────");

    let ds0 = s2.open_dataset("sensor_00").await?;
    let ts0 = ds0
        .read_array::<i64>("timestamps", vec![], vec![])
        .await?
        .unwrap();

    let ds7 = s2.open_dataset("sensor_07").await?;
    let ts7 = ds7
        .read_array::<i64>("timestamps", vec![], vec![])
        .await?
        .unwrap();

    assert_eq!(ts0, ts7, "timestamps are identical across sensors");
    println!("sensor_00 and sensor_07 share identical timestamps ✓");
    println!("first={} last={}", ts0[[0]], ts0[[N_READINGS - 1]]);

    Ok(())
}
