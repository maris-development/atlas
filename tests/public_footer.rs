//! The footer types are part of the public API.

use atlas::{Atlas, AtlasWriter, CollectionFooter, WriterConfig};

#[tokio::test]
async fn a_consumer_reads_the_footer_and_its_pools() {
    let tmp = tempfile::tempdir().unwrap();
    let w = AtlasWriter::create_path(tmp.path(), WriterConfig::default())
        .await
        .unwrap();
    let mut ds = w.add_dataset("jan").await.unwrap();
    ds.define_array::<f32>("temperature", vec!["x".into()], vec![2], None, None)
        .await
        .unwrap();
    ds.finish().await.unwrap();
    w.finish().await.unwrap();

    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    let footer: &CollectionFooter = atlas.footer();
    assert_eq!(footer.version, atlas::FORMAT_VERSION);

    // The pools, the variables, and the schemas all resolve from outside.
    let id = footer.string_id("temperature").unwrap();
    let index = footer.variable_index(id).unwrap();
    let entry: &atlas::VariableEntry = &footer.variables[index];
    assert!(entry.seg_len > 0);

    let schema: &atlas::InternedSchema = footer.schema_of(footer.datasets["jan"]);
    assert_eq!(schema.arrays[0].0, id);
    assert_eq!(
        footer.dtype(schema.arrays[0].1),
        Some(&atlas::DType::Float32)
    );

    // Encode and decode work on the public type too.
    let bytes = footer.encode().unwrap();
    assert_eq!(&CollectionFooter::decode(&bytes).unwrap(), footer);
}

#[tokio::test]
async fn the_footer_answers_the_schema_of_the_whole_collection() {
    let tmp = tempfile::tempdir().unwrap();
    let w = AtlasWriter::create_path(tmp.path(), WriterConfig::default())
        .await
        .unwrap();
    for month in ["jan", "feb"] {
        let mut ds = w.add_dataset(month).await.unwrap();
        ds.define_array::<f32>("temperature", vec!["x".into()], vec![2], None, None)
            .await
            .unwrap();
        ds.set_array_attribute("temperature", "units", atlas::Attr::String("K".into()))
            .unwrap();
        ds.set_attribute("month", atlas::Attr::Int64(1));
        ds.finish().await.unwrap();
    }
    // A second array, in one dataset only.
    let mut ds = w.add_dataset("mar").await.unwrap();
    ds.define_array::<f64>("salinity", vec!["x".into()], vec![2], None, None)
        .await
        .unwrap();
    ds.finish().await.unwrap();
    w.finish().await.unwrap();

    let atlas = Atlas::open_path(tmp.path()).await.unwrap();
    let schema = atlas.footer().collection_schema();

    // Every array of the collection, once, whatever declares it.
    let mut arrays: Vec<&str> = schema.arrays.keys().copied().collect();
    arrays.sort_unstable();
    assert_eq!(arrays, ["salinity", "temperature"]);
    assert_eq!(schema.array_dtypes("temperature"), [&atlas::DType::Float32]);
    assert_eq!(schema.array_dtypes("salinity"), [&atlas::DType::Float64]);

    // The global attribute and the array attribute, with their types.
    assert_eq!(schema.attribute_dtypes("month"), [&atlas::DType::Int64]);
    assert_eq!(
        schema.array_attribute_dtypes("temperature", "units"),
        [&atlas::DType::String]
    );
    assert!(!schema.array_attributes.contains_key("salinity"));
}
