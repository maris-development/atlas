//! Demonstrates creating a weather data store with multiple datasets that share physical
//! array files. Shows chunked and full writes, partial reads, attributes, reopening, and
//! how the codec is persisted in `atlas.json` so `open` needs no codec argument.

use std::sync::Arc;

use atlas::{Atlas, Attr, DType, StatValue, StoreConfig};
use ndarray::{Array1, Array2};
use object_store::{local::LocalFileSystem, path::Path};

// Grid: 8 latitudes × 16 longitudes, stored in 4×8 chunks
const NLAT: usize = 8;
const NLON: usize = 16;
const CHUNK_LAT: usize = 4;
const CHUNK_LON: usize = 8;

// Time series length
const NTIME: usize = 24;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let store: Arc<dyn object_store::ObjectStore> = Arc::new(LocalFileSystem::new());
    let prefix = Path::from_absolute_path(tmp.path())?;

    // ── Create store ──────────────────────────────────────────────────────────
    //
    // StoreConfig sets the codec for this store. It is written to atlas.json
    // so that Atlas::open() picks it up automatically — no need to pass the
    // codec again on reopen.

    let mut s = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default()).await?;
    println!("Created store at {}", tmp.path().display());

    // ── Dataset 1: January 2024 ───────────────────────────────────────────────
    //
    // Arrays:
    //   temperature  f32[8, 16]  chunked [4, 8]   – written two chunks at a time
    //   pressure     f64[8, 16]  chunk = full      – written in one shot
    //   time         i64[24]     chunk = full      – 1-D, hourly timestamps

    {
        let mut ds = s.create_dataset("jan_2024").await?;

        // --- temperature: define with explicit chunk shape ---
        ds.define_array::<f32>(
            "temperature",
            vec!["lat".into(), "lon".into()],
            vec![NLAT, NLON],
            Some(vec![CHUNK_LAT, CHUNK_LON]),
            None,
        )
        .await?;

        // Write chunk by chunk. Each chunk is a sub-region of the full grid.
        for lat in 0..(NLAT / CHUNK_LAT) {
            for lon in 0..(NLON / CHUNK_LON) {
                let start = vec![lat * CHUNK_LAT, lon * CHUNK_LON];
                // Realistic-ish temperature values: warmer at low latitudes
                let base = 20.0 - (lat as f32) * 3.0 + (lon as f32) * 0.5;
                let chunk = Array2::<f32>::from_elem([CHUNK_LAT, CHUNK_LON], base).into_dyn();
                ds.write_array("temperature", start, chunk.view()).await?;
            }
        }
        println!("jan_2024: wrote temperature in {} chunks", (NLAT / CHUNK_LAT) * (NLON / CHUNK_LON));

        // --- pressure: no chunk shape → chunk equals full shape ---
        ds.define_array::<f64>(
            "pressure",
            vec!["lat".into(), "lon".into()],
            vec![NLAT, NLON],
            None,
            None,
        )
        .await?;

        // Write entire array in one call
        let pressure = Array2::<f64>::from_elem([NLAT, NLON], 1013.25).into_dyn();
        ds.write_array("pressure", vec![0, 0], pressure.view()).await?;
        println!("jan_2024: wrote pressure as single full-array write");

        // --- time: 1-D hourly Unix timestamps ---
        ds.define_array::<i64>(
            "time",
            vec!["hour".into()],
            vec![NTIME],
            None,
            None,
        )
        .await?;

        let base_ts: i64 = 1_704_067_200; // 2024-01-01 00:00 UTC
        let time = Array1::from_iter((0..NTIME as i64).map(|h| base_ts + h * 3600)).into_dyn();
        ds.write_array("time", vec![0], time.view()).await?;

        // --- per-dataset attributes ---
        ds.set_attribute("month", Attr::UInt32(1));
        ds.set_attribute("year", Attr::UInt32(2024));
        ds.set_attribute("station", Attr::String("KNMI".into()));
        ds.set_attribute("has_qc", Attr::Bool(true));

        ds.flush().await?;
        println!("jan_2024: flushed");
    }

    // ── Dataset 2: February 2024 ──────────────────────────────────────────────
    //
    // Arrays:
    //   temperature  f32[8, 16]  chunked [4, 8]   – shares physical file with jan_2024
    //   humidity     f32[8, 16]  chunk = full      – unique to this dataset

    {
        let mut ds = s.create_dataset("feb_2024").await?;

        // temperature re-uses the same physical file as jan_2024 (keyed by dataset name inside)
        ds.define_array::<f32>(
            "temperature",
            vec!["lat".into(), "lon".into()],
            vec![NLAT, NLON],
            Some(vec![CHUNK_LAT, CHUNK_LON]),
            None,
        )
        .await?;

        // Write the full grid in one call (still within the chunked array)
        let feb_temp = Array2::<f32>::from_elem([NLAT, NLON], 5.0_f32).into_dyn();
        ds.write_array("temperature", vec![0, 0], feb_temp.view()).await?;
        println!("feb_2024: wrote temperature (shared file with jan_2024)");

        // humidity: unique to feb, unchunked
        ds.define_array::<f32>(
            "humidity",
            vec!["lat".into(), "lon".into()],
            vec![NLAT, NLON],
            None,
            None,
        )
        .await?;

        let humidity = Array2::<f32>::from_elem([NLAT, NLON], 78.5_f32).into_dyn();
        ds.write_array("humidity", vec![0, 0], humidity.view()).await?;

        ds.set_attribute("month", Attr::UInt32(2));
        ds.set_attribute("year", Attr::UInt32(2024));

        ds.flush().await?;
        println!("feb_2024: flushed");
    }

    // ── Dataset 3: sparse — only a time axis, no spatial grid ─────────────────

    {
        let mut ds = s.create_dataset("station_obs").await?;

        ds.define_array::<f64>(
            "wind_speed",
            vec!["time".into()],
            vec![NTIME],
            None,
            None,
        )
        .await?;

        let wind = Array1::from_iter((0..NTIME).map(|i| 3.0 + (i as f64) * 0.1)).into_dyn();
        ds.write_array("wind_speed", vec![0], wind.view()).await?;

        ds.set_attribute("lat", Attr::Float64(52.1));
        ds.set_attribute("lon", Attr::Float64(5.18));
        ds.set_attribute("name", Attr::String("De Bilt".into()));

        ds.flush().await?;
        println!("station_obs: flushed");
    }

    // ── Overview before reopening ─────────────────────────────────────────────

    let mut datasets = s.list_datasets();
    datasets.sort();
    println!("\nDatasets      : {:?}", datasets);
    println!("Physical arrays: {:?}", s.list_arrays());

    // ── Reopen and read back ──────────────────────────────────────────────────

    // Codec was saved in atlas.json — open() restores it without any argument.
    println!("\n─── Reopening store ───────────────────────────────────────────");
    let s2 = Atlas::open(store, prefix).await?;

    // --- Read jan_2024 ---
    let ds_jan = s2.open_dataset("jan_2024").await?;
    let meta = ds_jan.meta();

    println!("\njan_2024 attributes:");
    for (k, v) in &meta.attributes {
        println!("  {k} = {v:?}");
    }

    println!("\njan_2024 array schemas:");
    let mut array_names: Vec<&str> = meta.arrays.keys().map(|s| s.as_str()).collect();
    array_names.sort();
    for name in &array_names {
        let schema = &meta.arrays[*name];
        println!(
            "  {name}: dtype={:?}  shape={:?}  chunk_shape={:?}  dims={:?}  codec={:?}",
            schema.dtype, schema.shape, schema.chunk_shape, schema.dimension_names, schema.codec
        );
    }

    // Full read of temperature
    let temp_full = ds_jan
        .read_array::<f32>("temperature", vec![], vec![])
        .await?
        .unwrap();
    println!("\njan_2024 temperature (full [{NLAT}×{NLON}]):");
    println!("{temp_full:.1}");

    // Partial read — one chunk region
    let temp_chunk = ds_jan
        .read_array::<f32>("temperature", vec![0, 0], vec![CHUNK_LAT, CHUNK_LON])
        .await?
        .unwrap();
    println!("jan_2024 temperature chunk [0..{CHUNK_LAT}, 0..{CHUNK_LON}]:");
    println!("{temp_chunk:.1}");

    // Pressure — unchunked, full read
    let pressure = ds_jan
        .read_array::<f64>("pressure", vec![], vec![])
        .await?
        .unwrap();
    println!("\njan_2024 pressure (full, unchunked — first value = {:.2})", pressure[[0, 0]]);

    // Time — first 4 and last timestamp
    let time = ds_jan
        .read_array::<i64>("time", vec![], vec![])
        .await?
        .unwrap();
    println!(
        "jan_2024 time: first={}, last={}",
        time[[0]],
        time[[NTIME - 1]]
    );

    // --- Read feb_2024 ---
    let ds_feb = s2.open_dataset("feb_2024").await?;
    let feb_temp = ds_feb
        .read_array::<f32>("temperature", vec![], vec![])
        .await?
        .unwrap();
    println!("\nfeb_2024 temperature (first value = {:.1})", feb_temp[[0, 0]]);

    let feb_hum = ds_feb
        .read_array::<f32>("humidity", vec![], vec![])
        .await?
        .unwrap();
    println!("feb_2024 humidity  (first value = {:.1})", feb_hum[[0, 0]]);

    // --- Read station_obs ---
    let ds_obs = s2.open_dataset("station_obs").await?;
    println!("\nstation_obs attributes:");
    for (k, v) in &ds_obs.meta().attributes {
        println!("  {k} = {v:?}");
    }
    let wind = ds_obs
        .read_array::<f64>("wind_speed", vec![0], vec![4])
        .await?
        .unwrap();
    println!("station_obs wind_speed[0..4]: {wind:.2}");

    // Verify the shared temperature file holds both datasets independently
    let jan_val = ds_jan
        .read_array::<f32>("temperature", vec![0, 0], vec![1, 1])
        .await?
        .unwrap();
    let feb_val = ds_feb
        .read_array::<f32>("temperature", vec![0, 0], vec![1, 1])
        .await?
        .unwrap();
    assert_ne!(
        jan_val[[0, 0]], feb_val[[0, 0]],
        "jan and feb share the file but hold independent data"
    );
    println!(
        "\nShared 'temperature' file: jan[0,0]={:.1}  feb[0,0]={:.1}  (independent ✓)",
        jan_val[[0, 0]],
        feb_val[[0, 0]]
    );

    // Verify that schema dtype matches what we defined
    assert_eq!(meta.arrays["temperature"].dtype, DType::Float32);
    assert_eq!(meta.arrays["pressure"].dtype, DType::Float64);
    assert_eq!(meta.arrays["time"].dtype, DType::Int64);
    println!("dtype assertions passed ✓");

    // ── Statistics ────────────────────────────────────────────────────────────
    //
    // array_stats returns None until the first flush; after that the stats file
    // is persisted alongside the .af file and reloaded on open.

    println!("\n─── Statistics ────────────────────────────────────────────────");

    fn fmt_stat(v: &Option<StatValue>) -> String {
        match v {
            None => "n/a".into(),
            Some(StatValue::Float(f)) => format!("{f:.2}"),
            Some(StatValue::Int(i)) => i.to_string(),
            Some(StatValue::UInt(u)) => u.to_string(),
            Some(StatValue::Bytes(b)) => String::from_utf8_lossy(b).into_owned(),
        }
    }

    for array_name in &array_names {
        if let Some(stats) = ds_jan.array_stats(array_name).await? {
            println!(
                "jan_2024 / {array_name}: rows={} nulls={}  min={}  max={}",
                stats.row_count,
                stats.null_count,
                fmt_stat(&stats.min),
                fmt_stat(&stats.max),
            );
        }
    }

    // feb shares the 'temperature' file but has its own stats entry
    let feb_stats = ds_feb.array_stats("temperature").await?.unwrap();
    println!(
        "feb_2024 / temperature: rows={}  min={}  max={}",
        feb_stats.row_count,
        fmt_stat(&feb_stats.min),
        fmt_stat(&feb_stats.max),
    );

    // station_obs: wind_speed stats
    let wind_stats = ds_obs.array_stats("wind_speed").await?.unwrap();
    println!(
        "station_obs / wind_speed: rows={}  min={}  max={}",
        wind_stats.row_count,
        fmt_stat(&wind_stats.min),
        fmt_stat(&wind_stats.max),
    );

    Ok(())
}
