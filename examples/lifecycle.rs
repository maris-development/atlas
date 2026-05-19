//! Demonstrates the full mutation lifecycle: creating datasets, adding and
//! removing arrays, deleting datasets, and compacting reclaimed space.
//!
//! Key behaviours shown:
//! - Deleting an array removes it from the dataset but leaves the physical file
//!   intact if another dataset still uses it.
//! - Deleting a dataset removes it from the registry and tombstones its entries
//!   in every shared array file.
//! - Compact rewrites the array file, physically reclaiming tombstoned space.

use std::sync::Arc;

use atlas::{Atlas, Attr, StoreConfig};
use ndarray::{Array1, Array2};
use object_store::{local::LocalFileSystem, path::Path};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let store: Arc<dyn object_store::ObjectStore> = Arc::new(LocalFileSystem::new());
    let prefix = Path::from_absolute_path(tmp.path())?;

    // StoreConfig::default() uses Zstd compression. The codec is persisted in
    // atlas.json so that Atlas::open() below needs no codec argument.
    let mut s = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default()).await?;

    // ── Phase 1: create three datasets, two share "grid" ─────────────────────

    println!("=== Phase 1: create datasets ===");

    {
        let mut ds = s.create_dataset("north").await?;
        ds.define_array::<f32>("grid", vec!["x".into(), "y".into()], vec![4, 4], None, None)
            .await?;
        let data = Array2::<f32>::from_elem([4, 4], 1.0_f32).into_dyn();
        ds.write_array("grid", vec![0, 0], data.view()).await?;
        ds.define_array::<f64>("elevation", vec!["x".into(), "y".into()], vec![4, 4], None, None)
            .await?;
        let elev = Array2::<f64>::from_elem([4, 4], 100.0_f64).into_dyn();
        ds.write_array("elevation", vec![0, 0], elev.view()).await?;
        ds.set_attribute("region", Attr::String("north".into()));
    }
    s.flush().await?;

    {
        let mut ds = s.create_dataset("south").await?;
        ds.define_array::<f32>("grid", vec!["x".into(), "y".into()], vec![4, 4], None, None)
            .await?;
        let data = Array2::<f32>::from_elem([4, 4], 2.0_f32).into_dyn();
        ds.write_array("grid", vec![0, 0], data.view()).await?;
        ds.set_attribute("region", Attr::String("south".into()));
    }
    s.flush().await?;

    {
        let mut ds = s.create_dataset("scratch").await?;
        ds.define_array::<i32>("ids", vec!["n".into()], vec![8], None, None).await?;
        let ids = Array1::from_iter(0..8_i32).into_dyn();
        ds.write_array("ids", vec![0], ids.view()).await?;
    }
    s.flush().await?;

    let mut names = s.list_datasets();
    names.sort();
    println!("Datasets: {:?}", names);
    println!("Physical arrays: {:?}", s.list_arrays());

    // ── Phase 2: remove an array from a dataset ───────────────────────────────

    println!("\n=== Phase 2: delete 'elevation' from 'north' ===");

    {
        let mut north = s.open_dataset("north").await?;
        println!("north arrays before: {:?}", {
            let mut v = north.list_arrays();
            v.sort();
            v
        });

        north.delete_array("elevation").await?;
    }
    s.flush().await?;
    {
        let north = s.open_dataset("north").await?;

        println!("north arrays after:  {:?}", {
            let mut v = north.list_arrays();
            v.sort();
            v
        });
    }

    // Verify the physical file still exists (it holds data for other arrays too,
    // though in this case elevation was unique — the file stays, just tombstoned).
    let north2 = s.open_dataset("north").await?;
    assert!(north2.read_array::<f32>("grid", vec![], vec![]).await?.is_some());
    assert!(north2.read_array::<f64>("elevation", vec![], vec![]).await?.is_none());
    println!("'grid' still readable, 'elevation' gone from north ✓");

    // ── Phase 3: delete a whole dataset ──────────────────────────────────────

    println!("\n=== Phase 3: delete 'scratch' dataset ===");

    assert!(s.dataset_exists("scratch"));
    s.delete_dataset("scratch").await?;
    assert!(!s.dataset_exists("scratch"));

    let mut names = s.list_datasets();
    names.sort();
    println!("Datasets after delete: {:?}", names);
    // 'ids' physical file still exists on disk (tombstoned, not yet compacted)
    println!("Physical arrays still present: {:?}", s.list_arrays());

    // ── Phase 4: 'south' is deleted; 'grid' file is still used by 'north' ────

    println!("\n=== Phase 4: delete 'south'; 'grid' file survives ===");

    s.delete_dataset("south").await?;

    // north can still read its grid — south's entry was tombstoned, not north's
    let north3 = s.open_dataset("north").await?;
    let grid = north3.read_array::<f32>("grid", vec![], vec![]).await?.unwrap();
    assert_eq!(grid[[0, 0]], 1.0_f32, "north grid unaffected by south deletion");
    println!("north grid[0,0] = {:.1} (still 1.0, unaffected) ✓", grid[[0, 0]]);

    let mut names = s.list_datasets();
    names.sort();
    println!("Datasets remaining: {:?}", names);

    // ── Phase 5: compact to reclaim tombstoned space ──────────────────────────

    println!("\n=== Phase 5: compact 'north' ===");

    let north4 = s.open_dataset("north").await?;

    // Read the grid once more before compacting to confirm it still works
    let before = north4.read_array::<f32>("grid", vec![], vec![]).await?.unwrap();
    println!("grid[0,0] before compact = {:.1}", before[[0, 0]]);
    drop(north4);

    s.compact().await?;
    println!("compact done");

    // Data must be intact after compaction
    let north4 = s.open_dataset("north").await?;
    let after = north4.read_array::<f32>("grid", vec![], vec![]).await?.unwrap();
    assert_eq!(before, after, "data unchanged after compact");
    println!("grid[0,0] after  compact = {:.1} (unchanged ✓)", after[[0, 0]]);

    // ── Phase 6: reopen and verify final state ────────────────────────────────

    println!("\n=== Phase 6: reopen and verify ===");

    // Codec is restored from atlas.json — no StoreConfig needed.
    let s2 = Atlas::open(store, prefix).await?;

    assert!(s2.dataset_exists("north"));
    assert!(!s2.dataset_exists("south"));
    assert!(!s2.dataset_exists("scratch"));

    let north5 = s2.open_dataset("north").await?;
    assert_eq!(north5.get_attribute("region"), Some(Attr::String("north".into())));

    let final_grid = north5.read_array::<f32>("grid", vec![], vec![]).await?.unwrap();
    assert_eq!(final_grid[[0, 0]], 1.0_f32);

    println!("Only 'north' survives, grid and attributes intact ✓");
    println!("region = {:?}", north5.get_attribute("region").unwrap());

    Ok(())
}
