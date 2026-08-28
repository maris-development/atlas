//! Compatibility test against committed bytes.
//!
//! `tests/fixtures/golden_v1/` holds a container written once, by hand, and
//! checked in. The test below opens it and asserts every value. If a change to
//! the format or the reader breaks compatibility with a v1 container, this
//! fails, and no round-trip test would have noticed.
//!
//! The writer's output is deliberately *not* compared byte for byte. Zstd does
//! not promise stable output across versions, so only the framing — which this
//! crate produces itself — is asserted exactly.
//!
//! To regenerate after an intentional format change (which means bumping
//! `FORMAT_VERSION`):
//!
//! ```text
//! cargo test --test golden -- --ignored regenerate
//! ```

use std::path::{Path, PathBuf};

use atlas::{Atlas, AtlasWriter, Attr, Codec, FillValue, WriterConfig};
use ndarray::{Array1, ArrayD, IxDyn};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_v1")
}

/// Writes the fixture. Kept in sync with what the assertions below expect.
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
        // Deliberately never written: it must read back as zero.

        ds.set_attribute("month", Attr::Int64(1));
        ds.set_attribute("scale", Attr::Float64(0.5));
        ds.set_attribute("tags", Attr::StringList(vec!["a".into(), "b".into()]));
        ds.set_attribute(
            "created",
            Attr::TimestampNanoseconds(1_700_000_000_000_000_000),
        );
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
#[ignore = "run explicitly to regenerate tests/fixtures/golden_v1"]
async fn regenerate() {
    let dir = fixture_dir();
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_golden(&dir).await;
    println!("wrote {}", dir.display());
}

#[tokio::test]
async fn the_committed_v1_container_still_opens() {
    let atlas = Atlas::open_path(fixture_dir()).await.unwrap();

    assert_eq!(atlas.list_datasets(), vec!["grid", "labels"]);
    assert_eq!(atlas.list_arrays(), vec!["counts", "name", "temperature"]);

    let grid = atlas.dataset("grid").unwrap();
    assert_eq!(grid.ordinal(), 0);
    assert_eq!(grid.list_arrays(), vec!["temperature", "counts"]);

    let meta = grid.array_meta("temperature").unwrap();
    assert_eq!(meta.dtype, atlas::DType::Float32);
    assert_eq!(meta.shape, vec![4, 6]);
    assert_eq!(meta.chunk_shape, vec![2, 3]);
    assert_eq!(meta.dimension_names, vec!["lat", "lon"]);
    assert!(matches!(
        grid.array_fill_value("temperature"),
        Some(FillValue::Float(f)) if f.is_nan()
    ));

    assert_eq!(grid.get_attribute("month"), Some(Attr::Int64(1)));
    assert_eq!(grid.get_attribute("scale"), Some(Attr::Float64(0.5)));
    assert_eq!(
        grid.get_attribute("tags"),
        Some(Attr::StringList(vec!["a".into(), "b".into()]))
    );
    assert_eq!(
        grid.get_attribute("created"),
        Some(Attr::TimestampNanoseconds(1_700_000_000_000_000_000))
    );
    assert_eq!(
        grid.get_array_attribute("temperature", "units"),
        Some(Attr::String("celsius".into()))
    );

    let temp = grid
        .read_array::<f32>("temperature", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(temp.shape(), &[4, 6]);
    assert_eq!(temp[[0, 0]], 0.0);
    assert_eq!(temp[[3, 5]], 23.0);
    // A window straddling the 2x3 chunk grid.
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

    let labels = atlas.dataset("labels").unwrap();
    let names = labels
        .read_array::<String>("name", vec![], vec![])
        .await
        .unwrap();
    assert_eq!(names[[0]], "alpha");
    assert_eq!(names[[2]], "gamma");
}

#[tokio::test]
async fn the_committed_container_has_the_v1_framing() {
    let bytes = std::fs::read(fixture_dir().join("data.atlas")).unwrap();
    let len = bytes.len();

    assert_eq!(&bytes[0..4], b"ATLS", "leading magic");
    assert_eq!(
        u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        1,
        "header version"
    );
    assert_eq!(&bytes[len - 4..], b"ATLS", "trailing magic");
    assert_eq!(
        u32::from_le_bytes(bytes[len - 8..len - 4].try_into().unwrap()),
        1,
        "trailer version"
    );
    let footer_size = u64::from_le_bytes(bytes[len - 16..len - 8].try_into().unwrap()) as usize;
    assert_eq!(
        len,
        8 + segment_bytes(&bytes, footer_size) + footer_size + 16,
        "header + segments + footer + trailer must account for every byte"
    );
}

/// Bytes between the header and the footer, i.e. everything the segments
/// occupy.
fn segment_bytes(bytes: &[u8], footer_size: usize) -> usize {
    bytes.len() - 8 - footer_size - 16
}
