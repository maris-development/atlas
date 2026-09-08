//! One glob import reaches the whole API.

use atlas::prelude::*;

/// `AtlasResult` is this crate's alias. Plain `Result` still takes two
/// parameters, so the glob import left `std::result::Result` alone.
fn two_parameter_result() -> Result<(), std::fmt::Error> {
    Ok(())
}

#[tokio::test]
async fn the_prelude_covers_a_write_and_a_read() -> AtlasResult<()> {
    two_parameter_result().unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let w = AtlasWriter::create_path(tmp.path(), WriterConfig::default()).await?;
    let mut ds = w.add_dataset("jan").await?;
    ds.define_array::<f32>("temperature", vec!["x".into()], vec![2], None, None)
        .await?;
    ds.set_attribute("month", Attr::Int64(1));
    ds.finish().await?;
    w.finish().await?;

    let atlas: Atlas = Atlas::open_path(tmp.path()).await?;
    let view: DatasetView = atlas.dataset("jan")?;
    let schema: SchemaView = view.schema();
    assert_eq!(schema.names().collect::<Vec<_>>(), ["temperature"]);

    let layout: ArrayLayout = view.array_layout("temperature").await?;
    assert_eq!(layout.shape(), vec![2]);

    let meta: ArrayMeta = view.array_meta("temperature").unwrap();
    assert_eq!(*meta.dtype(), DType::Float32);

    // The footer types and the collection schema come along too.
    let footer: &CollectionFooter = atlas.footer();
    assert_eq!(footer.version, FORMAT_VERSION);
    let entry: &VariableEntry = &footer.variables[0];
    assert!(entry.seg_len > 0);
    let interned: &InternedSchema = footer.schema_of(footer.datasets["jan"]);
    assert_eq!(interned.arrays.len(), 1);
    let collection: CollectionSchema = footer.collection_schema();
    assert_eq!(collection.array_dtypes("temperature"), [&DType::Float32]);

    let stats: Option<ArrayStats> = atlas.array_stats("temperature").await?;
    assert!(stats.is_some());
    assert_eq!(atlas.codec(), Codec::Zstd);

    // An error still names the crate's own type.
    let missing: Error = atlas.dataset("nope").unwrap_err();
    assert!(matches!(missing, Error::DatasetNotFound(_)));
    Ok(())
}
