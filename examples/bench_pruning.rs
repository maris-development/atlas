//! Benchmark: rebuild the on-demand pruning index over a large collection.
//!
//! Builds `N` datasets (default 1,000,000) spread across `A` arrays so each
//! array is declared by ~1/A of the datasets (25% for A=4, disjoint subsets),
//! writes a tiny 1-element value per dataset, flushes, then times rebuilding
//! the flat pruning index — one column, all columns, and the full summaries.
//!
//! Run in release for realistic numbers:
//!     cargo run --release --example bench_pruning [N] [A]

use std::sync::Arc;
use std::time::Instant;

use atlas::{Atlas, ColumnKey, StoreConfig};
use object_store::{local::LocalFileSystem, path::Path};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let a: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(4);

    let tmp = tempfile::tempdir()?;
    let store: Arc<dyn object_store::ObjectStore> = Arc::new(LocalFileSystem::new());
    let prefix = Path::from_absolute_path(tmp.path())?;
    let mut s = Atlas::create(store.clone(), prefix.clone(), StoreConfig::default()).await?;

    let arrays: Vec<String> = (0..a).map(|i| format!("arr{i}")).collect();
    println!(
        "building {n} datasets across {a} arrays (~{:.0}% presence each)…",
        100.0 / a as f64
    );

    // ── Build ──────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    for i in 0..n {
        let mut ds = s.create_dataset(&format!("ds{i:08}")).await?;
        let array = &arrays[i % a];
        ds.define_array::<i64>(array, vec!["x".into()], vec![1], None, None).await?;
        let data = ndarray::Array1::from_vec(vec![i as i64]).into_dyn();
        ds.write_array(array, vec![0], data.view()).await?;
    }
    let build = t0.elapsed();

    let t1 = Instant::now();
    s.flush().await?;
    let flush = t1.elapsed();
    println!("  build: {build:.1?}   flush: {flush:.1?}");

    // ── Read: rebuild the pruning index ──────────────────────────────────────
    let one = ColumnKey::array(arrays[0].clone());
    // First call is "cold" (opens the array file); second is "warm" (cached).
    for label in ["cold", "warm"] {
        let t = Instant::now();
        let idx = s.pruning_index(std::slice::from_ref(&one)).await?;
        let el = t.elapsed();
        let view = idx.view(&one).unwrap();
        println!(
            "  pruning_index([{}]) {label}: {el:.2?}   (rows={}, present={})",
            arrays[0],
            idx.rows(),
            view.present_rows().len(),
        );
    }

    let all: Vec<ColumnKey> = arrays.iter().map(|s| ColumnKey::array(s.clone())).collect();
    let t = Instant::now();
    let idx = s.pruning_index(&all).await?;
    println!("  pruning_index(all {a} cols): {:.2?}   (rows={})", t.elapsed(), idx.rows());

    let t = Instant::now();
    let sums = s.column_summaries().await?;
    println!("  column_summaries ({} cols): {:.2?}", sums.len(), t.elapsed());

    Ok(())
}
