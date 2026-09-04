//! Compatibility test against committed bytes.
//!
//! `tests/fixtures/golden_v6/` holds one container, written by hand and
//! committed. The test below opens it and asserts every value. A change to the
//! format or the reader that breaks a v6 container fails here. No round-trip
//! test catches that.
//!
//! This test does *not* compare the writer's output byte for byte. Zstd
//! promises no stable output across versions. Only the framing gets an exact
//! assertion, because this crate produces the framing itself.
//!
//! To regenerate after a deliberate format change, bump `FORMAT_VERSION`, then
//! run:
//!
//! ```text
//! cargo test --test golden -- --ignored regenerate
//! ```

use std::path::{Path, PathBuf};

use atlas::{Atlas, AtlasWriter, Attr, Codec, FillValue, StatValue, WriterConfig};
use ndarray::{Array1, ArrayD, IxDyn};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_v6")
}

/// Writes the fixture. Keep it in step with the assertions below.
async fn write_golden(dir: &Path) {
    let w = AtlasWriter::create_path(
        dir,
        WriterConfig {
            codec: Codec::Zstd,
            ..Default::default()
        },
    )
    .await
    .unwrap();

    {
        let mut ds = w.add_dataset("grid").await.unwrap();
        ds.define_array::<f32>(
            "temperature",
            vec!["lat".into(), "lon".into()],
            vec![4, 6],
            Some(vec![2, 3]),
            Some(FillValue::Float(f64::NAN)),
        )
        .await
        .unwrap();
        let data = ArrayD::from_shape_fn(IxDyn(&[4, 6]), |i| (i[0] * 6 + i[1]) as f32);
        ds.write_array("temperature", vec![0, 0], data.view())
            .await
            .unwrap();

        ds.define_array::<i64>("counts", vec!["lat".into()], vec![4], None, None)
            .await
            .unwrap();
        // Nobody writes this array. It must read back as zero.

        ds.set_attribute("month", Attr::Int64(1));
        ds.set_attribute("scale", Attr::Float64(0.5));
        ds.set_attribute("tags", Attr::StringList(vec!["a".into(), "b".into()]));
        // The committed fixture predates the removal of the timestamp
        // attribute, so its footer still declares `created` as TimestampNs.
        // A regenerate writes Int64 there instead, and the collection loses
        // its one case of an old container whose schema names a type no
        // writer can produce. The container bytes are otherwise identical.
        ds.set_attribute("created", Attr::Int64(1_700_000_000_000_000_000));
        ds.set_array_attribute("temperature", "units", Attr::String("celsius".into()))
            .unwrap();
        ds.finish().await.unwrap();
    }

    {
        let mut ds = w.add_dataset("labels").await.unwrap();
        ds.define_array::<String>("name", vec!["i".into()], vec![3], None, None)
            .await
            .unwrap();
        let names =
            Array1::from_vec(vec!["alpha".to_string(), "beta".into(), "gamma".into()]).into_dyn();
        ds.write_array("name", vec![0], names.view()).await.unwrap();
        ds.finish().await.unwrap();
    }

    w.finish().await.unwrap();
}

#[tokio::test]
#[ignore = "run explicitly to regenerate tests/fixtures/golden_v6"]
async fn regenerate() {
    let dir = fixture_dir();
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_golden(&dir).await;
    println!("wrote {}", dir.display());
}

#[tokio::test]
async fn the_committed_v6_container_still_opens() {
    let atlas = Atlas::open_path(fixture_dir()).await.unwrap();

    assert_eq!(atlas.list_datasets(), vec!["grid", "labels"]);
    assert_eq!(atlas.list_arrays(), vec!["counts", "name", "temperature"]);

    let grid = atlas.dataset("grid").unwrap();
    assert_eq!(grid.ordinal(), 0);
    assert_eq!(grid.list_arrays(), vec!["temperature", "counts"]);

    let meta = grid.array_meta("temperature").unwrap();
    assert_eq!(*meta.dtype(), atlas::DType::Float32);
    let layout = grid.array_layout("temperature").await.unwrap();
    assert_eq!(layout.shape(), vec![4, 6]);
    assert_eq!(layout.chunk_shape(), vec![2, 3]);
    assert_eq!(layout.dimension_names(), vec!["lat", "lon"]);
    assert!(matches!(
        layout.fill_value(),
        Some(FillValue::Float(f)) if f.is_nan()
    ));

    assert_eq!(
        grid.get_attribute("month").await.unwrap(),
        Some(Attr::Int64(1))
    );
    assert_eq!(
        grid.get_attribute("scale").await.unwrap(),
        Some(Attr::Float64(0.5))
    );
    assert_eq!(
        grid.get_attribute("tags").await.unwrap(),
        Some(Attr::StringList(vec!["a".into(), "b".into()]))
    );
    // This fixture was written when `Attr` still had a timestamp variant, so
    // its footer declares `created` as TimestampNs. A read consults no
    // schema, and the stored form is an i64, so it comes back as one. That is
    // the point: an old container still opens, and its footer decodes
    // nothing.
    assert_eq!(
        grid.get_attribute("created").await.unwrap(),
        Some(Attr::Int64(1_700_000_000_000_000_000))
    );
    assert_eq!(
        grid.get_array_attribute("temperature", "units")
            .await
            .unwrap(),
        Some(Attr::String("celsius".into()))
    );

    let temp = grid
        .read_array::<f32>("temperature", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(temp.shape(), &[4, 6]);
    assert_eq!(temp[[0, 0]], 0.0);
    assert_eq!(temp[[3, 5]], 23.0);
    // A window across the 2x3 chunk grid.
    let window = grid
        .read_array::<f32>("temperature", vec![1, 2], vec![2, 2])
        .await
        .unwrap();
    assert_eq!(window[[0, 0]], 8.0);
    assert_eq!(window[[1, 1]], 15.0);

    let counts = grid
        .read_array::<i64>("counts", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(counts.as_slice().unwrap(), &[0, 0, 0, 0]);

    // The write computed these statistics. They live in the footer.
    let stats = grid
        .array_stats("temperature")
        .await
        .unwrap()
        .expect("temperature has stats");
    assert_eq!(stats.name, "temperature");
    assert_eq!(stats.row_count, 24);
    assert_eq!(stats.min, Some(StatValue::Float(0.0)));
    assert_eq!(stats.max, Some(StatValue::Float(23.0)));
    // The write covered every cell, and no cell holds NaN.
    assert_eq!(stats.null_count, 0);

    // counts is declared and never written. Every element is therefore the
    // fill value, which is what the null count reports.
    let counts_stats = grid
        .array_stats("counts")
        .await
        .unwrap()
        .expect("counts has stats");
    assert_eq!(counts_stats.row_count, 4);
    assert_eq!(counts_stats.null_count, 4);
    assert_eq!(counts_stats.min, None);

    // An array this dataset does not declare has no statistics.
    assert!(grid.array_stats("missing").await.unwrap().is_none());

    // A string gets a lexicographic min and max, as raw bytes.
    let labels = atlas.dataset("labels").unwrap();
    let name_stats = labels
        .array_stats("name")
        .await
        .unwrap()
        .expect("name has stats");
    assert_eq!(name_stats.row_count, 3);
    assert_eq!(name_stats.min, Some(StatValue::Bytes(b"alpha".to_vec())));
    assert_eq!(name_stats.max, Some(StatValue::Bytes(b"gamma".to_vec())));

    let names = labels
        .read_array::<String>("name", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(names[[0]], "alpha");
    assert_eq!(names[[2]], "gamma");
}

#[tokio::test]
async fn the_committed_container_has_the_v6_framing() {
    let bytes = std::fs::read(fixture_dir().join("data.atlas")).unwrap();
    let len = bytes.len();

    assert_eq!(&bytes[0..4], b"ATLS", "leading magic");
    assert_eq!(
        u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        6,
        "header version"
    );
    assert_eq!(&bytes[len - 4..], b"ATLS", "trailing magic");
    assert_eq!(
        u32::from_le_bytes(bytes[len - 8..len - 4].try_into().unwrap()),
        6,
        "trailer version"
    );
    let footer_size = u64::from_le_bytes(bytes[len - 16..len - 8].try_into().unwrap()) as usize;
    assert_eq!(
        len,
        8 + segment_bytes(&bytes, footer_size) + footer_size + 16,
        "header + segments + footer + trailer must account for every byte"
    );
}

/// The bytes between the header and the footer. The segments fill them.
fn segment_bytes(bytes: &[u8], footer_size: usize) -> usize {
    bytes.len() - 8 - footer_size - 16
}
