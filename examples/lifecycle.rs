//! The whole life of a collection: build it, read it, delete from it.
//!
//! There is no mutation phase, because the format has none. A collection is
//! written once. The only thing that changes afterwards is the deletion mask,
//! and that hides datasets rather than removing their bytes.
//!
//! Run with: `cargo run --example lifecycle`

use atlas::{Atlas, AtlasWriter, Attr, FillValue, WriterConfig};
use ndarray::{Array1, Array2};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;

    // ── Build ────────────────────────────────────────────────────────

    println!("=== Building the collection ===");
    let mut w = AtlasWriter::create_path(tmp.path(), WriterConfig::default()).await?;

    for (name, value) in [("north", 1.0f32), ("south", 2.0), ("east", 3.0)] {
        let mut ds = w.add_dataset(name)?;

        ds.define_array::<f32>(
            "grid",
            vec!["x".into(), "y".into()],
            vec![4, 4],
            Some(vec![2, 2]), // four chunks, so partial reads stay partial
            Some(FillValue::Float(f64::NAN)),
        )
        .await?;
        let grid = Array2::<f32>::from_elem([4, 4], value).into_dyn();
        ds.write_array("grid", vec![0, 0], grid.view()).await?;

        ds.define_array::<f64>("elevation", vec!["x".into()], vec![4], None, None)
            .await?;
        let elevation = Array1::from_vec(vec![10.0f64, 20.0, 30.0, 40.0]).into_dyn();
        ds.write_array("elevation", vec![0], elevation.view()).await?;

        ds.set_attribute("region", Attr::String(name.to_string()));
        ds.set_array_attribute("grid", "units", Attr::String("celsius".into()))?;

        // Until this line, nothing about the dataset has entered the file.
        ds.finish().await?;
        println!("  wrote dataset '{name}'");
    }

    // And until this line, nothing at all is readable.
    w.finish().await?;
    let size = std::fs::metadata(tmp.path().join("data.atlas"))?.len();
    println!("  data.atlas is {size} bytes\n");

    // ── Read ─────────────────────────────────────────────────────────

    println!("=== Reading ===");
    // Opening touches the footer only, however large the collection is.
    let atlas = Atlas::open_path(tmp.path()).await?;
    println!("  datasets: {:?}", atlas.list_datasets());
    println!("  arrays:   {:?}", atlas.list_arrays());

    let north = atlas.dataset("north")?;
    let meta = north.array_meta("grid").expect("grid is defined");
    println!(
        "  north/grid: {:?} shape {:?} chunks {:?}",
        meta.dtype, meta.shape, meta.chunk_shape
    );
    println!("  north attributes: {:?}", north.attributes());
    println!(
        "  north/grid units: {:?}",
        north.get_array_attribute("grid", "units")
    );
    println!(
        "  north segment bytes: {:?}  (a complete array-format file)",
        north.segment_range()
    );

    // The first call that needs data opens the segment. This one reads a
    // single chunk, not the whole array.
    let corner = north.read_array::<f32>("grid", vec![0, 0], vec![2, 2]).await?;
    println!("  north/grid[0..2, 0..2] = {:?}\n", corner.as_slice().unwrap());

    // ── Delete ───────────────────────────────────────────────────────

    println!("=== Deleting ===");
    atlas.delete_dataset("south").await?;
    println!("  after delete: {:?}", atlas.list_datasets());

    let reopened = Atlas::open_path(tmp.path()).await?;
    println!("  after reopen: {:?}", reopened.list_datasets());

    // The container did not move. Only a small mask file was written.
    let size_after = std::fs::metadata(tmp.path().join("data.atlas"))?.len();
    let mask = std::fs::metadata(tmp.path().join("deleted.mask"))?.len();
    println!("  data.atlas is still {size_after} bytes; deleted.mask is {mask}");

    // Ordinals never shift, so 'east' is where it always was.
    println!(
        "  'east' is still ordinal {}",
        reopened.dataset("east")?.ordinal()
    );
    println!("\nTo reclaim the deleted bytes, write a new collection.");

    Ok(())
}
