//! A consumer reaches the segments of a collection.

use atlas::{ArrayFile, Atlas, AtlasWriter, Attr, WriterConfig};

/// A collection of two datasets over one array, plus one array only the
/// second declares.
async fn collection(dir: &std::path::Path) {
    let w = AtlasWriter::create_path(dir, WriterConfig::default())
        .await
        .unwrap();
    for month in ["jan", "feb"] {
        let mut ds = w.add_dataset(month).await.unwrap();
        ds.define_array::<f32>("temperature", vec!["x".into()], vec![2], None, None)
            .await
            .unwrap();
        let data = ndarray::arr1(&[1.0f32, 2.0]).into_dyn();
        ds.write_array("temperature", vec![0], data.view())
            .await
            .unwrap();
        ds.set_attribute("month", Attr::Int64(1));
        ds.finish().await.unwrap();
    }
    w.finish().await.unwrap();
}

#[tokio::test]
async fn the_segments_of_a_collection_are_named_and_placed() {
    let tmp = tempfile::tempdir().unwrap();
    collection(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();

    // The container's own list. It names the reserved attribute segment too.
    let names = atlas.segment_names();
    assert!(names.contains(&"temperature"), "{names:?}");
    assert!(names.contains(&"_datasets"), "{names:?}");
    // `list_arrays` answers per declared array, so that one stays out.
    assert_eq!(atlas.list_arrays(), vec!["temperature".to_string()]);

    // The byte range of one segment, with no open.
    let entry = atlas.segment_entry("temperature").unwrap();
    assert!(entry.seg_len > 0);
    assert!(entry.seg_offset + entry.seg_len <= atlas.container_bytes());
    assert!(atlas.segment_entry("missing").is_none());
}

#[tokio::test]
async fn one_segment_answers_for_every_dataset() {
    let tmp = tempfile::tempdir().unwrap();
    collection(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();

    // The handle is the array-format file. It keys on the dataset name.
    let segment: &ArrayFile = atlas.segment("temperature").await.unwrap();
    let mut datasets: Vec<&str> = segment.stats().map(|s| s.name.as_str()).collect();
    datasets.sort_unstable();
    assert_eq!(datasets, ["feb", "jan"]);

    // A view opens the same handle, so the two are one file.
    let view = atlas.dataset("jan").unwrap();
    let same = view.segment("temperature").await.unwrap();
    assert!(std::ptr::eq(
        &**same,
        &**atlas.segment("temperature").await.unwrap()
    ));

    // A name the collection does not hold.
    assert!(atlas.try_segment("missing").await.unwrap().is_none());
    assert!(atlas.segment("missing").await.is_err());
}

#[tokio::test]
async fn the_array_format_crate_comes_through_atlas() {
    let tmp = tempfile::tempdir().unwrap();
    collection(tmp.path()).await;
    let atlas = Atlas::open_path(tmp.path()).await.unwrap();

    // Types the segment API returns, named through the re-export.
    let segment = atlas.segment("temperature").await.unwrap();
    let stats: &atlas::array_format::ArrayStats = segment.stats().next().unwrap();
    assert_eq!(stats.row_count, 2);
    let dtype: atlas::array_format::DType = atlas::DType::Float32;
    assert_eq!(
        *atlas
            .footer()
            .collection_schema()
            .array_dtypes("temperature")[0],
        dtype
    );
}
