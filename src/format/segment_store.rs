//! An [`ObjectStore`] view of one segment inside the container.
//!
//! `array-format` opens a file through an [`ObjectStore`] and a path. A
//! segment sits inside a larger file. This adapter therefore presents the byte
//! range `container[offset .. offset + len]` as one standalone object named
//! [`segment_path`]. It translates each range request and forwards it. It
//! buffers nothing.
//!
//! Two behaviours matter as much as the translation:
//!
//! - `list` returns nothing. Sidecar discovery therefore finds no delta layers.
//!   A segment is always one compacted base.
//! - Any other path is [`NotFound`](object_store::Error::NotFound). The
//!   statistics probe for `seg<n>.stats` therefore comes back empty, and does
//!   not fail.
//!
//! Every write method is [`NotSupported`](object_store::Error::NotSupported).
//! A collection is immutable, so nothing must try.
//!
//! The virtual name carries the segment's ordinal. A block cache keys on
//! `(path, block_id)`. One name across two segments would let the blocks of one
//! dataset answer the reads of another.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use futures::StreamExt;
use futures::stream::BoxStream;
use object_store::path::Path as OsPath;
use object_store::{
    CopyOptions, GetOptions, GetRange, GetResult, GetResultPayload, ListResult, MultipartUpload,
    ObjectMeta, ObjectStore, ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload,
    PutResult,
};

/// The name `array-format` sees for the segment at `ordinal`. It must end in
/// `.af`, because `ArrayFile::open` expects that suffix.
pub(crate) fn segment_path(ordinal: u32) -> OsPath {
    OsPath::from(format!("seg{ordinal}.af"))
}

/// Presents one byte range of a container as a single-object store.
#[derive(Debug)]
pub(crate) struct SegmentStore {
    inner: Arc<dyn ObjectStore>,
    container: OsPath,
    name: OsPath,
    offset: u64,
    len: u64,
}

impl SegmentStore {
    /// Views `container[offset .. offset + len]` as the single object
    /// [`segment_path(ordinal)`](segment_path).
    pub(crate) fn new(
        inner: Arc<dyn ObjectStore>,
        container: OsPath,
        ordinal: u32,
        offset: u64,
        len: u64,
    ) -> Self {
        Self {
            inner,
            container,
            name: segment_path(ordinal),
            offset,
            len,
        }
    }

    /// The path callers must use to reach this segment.
    pub(crate) fn path(&self) -> OsPath {
        self.name.clone()
    }

    /// Resolves a requested range against the segment, in segment-local
    /// coordinates. A real object store behaves the same way. An end past the
    /// object clamps. A start past the object is an error.
    fn resolve(&self, range: Option<GetRange>) -> object_store::Result<Range<u64>> {
        let local = match range {
            None => 0..self.len,
            Some(GetRange::Bounded(r)) => r.start..r.end.min(self.len),
            Some(GetRange::Offset(o)) => o..self.len,
            Some(GetRange::Suffix(n)) => self.len.saturating_sub(n)..self.len,
        };
        if local.start >= local.end {
            return Err(object_store::Error::Generic {
                store: "SegmentStore",
                source: format!(
                    "requested range {}..{} is empty or starts past the {}-byte segment",
                    local.start, local.end, self.len
                )
                .into(),
            });
        }
        Ok(local)
    }

    fn meta(&self) -> ObjectMeta {
        ObjectMeta {
            location: self.name.clone(),
            // The container holds no per-segment timestamp. `array-format`
            // uses the size only.
            last_modified: chrono::DateTime::UNIX_EPOCH,
            size: self.len,
            e_tag: None,
            version: None,
        }
    }

    fn not_supported(op: &'static str) -> object_store::Error {
        object_store::Error::NotSupported {
            source: format!("{op} is not supported: an atlas collection is immutable").into(),
        }
    }
}

impl fmt::Display for SegmentStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SegmentStore({}[{}..{}])",
            self.container,
            self.offset,
            self.offset + self.len
        )
    }
}

#[async_trait::async_trait]
impl ObjectStore for SegmentStore {
    async fn put_opts(
        &self,
        _location: &OsPath,
        _payload: PutPayload,
        _opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        Err(Self::not_supported("put"))
    }

    async fn put_multipart_opts(
        &self,
        _location: &OsPath,
        _opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        Err(Self::not_supported("put_multipart"))
    }

    async fn get_opts(
        &self,
        location: &OsPath,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        if location != &self.name {
            return Err(object_store::Error::NotFound {
                path: location.to_string(),
                source: format!("a segment store holds only '{}'", self.name).into(),
            });
        }
        let local = self.resolve(options.range)?;
        if options.head {
            return Ok(GetResult {
                payload: GetResultPayload::Stream(Box::pin(futures::stream::empty())),
                meta: self.meta(),
                range: local,
                attributes: Default::default(),
            });
        }
        let absolute = (self.offset + local.start)..(self.offset + local.end);
        let bytes = self.inner.get_range(&self.container, absolute).await?;
        Ok(GetResult {
            payload: GetResultPayload::Stream(Box::pin(futures::stream::once(async move {
                Ok::<Bytes, object_store::Error>(bytes)
            }))),
            meta: self.meta(),
            range: local,
            attributes: Default::default(),
        })
    }

    /// Always empty. This tells `array-format` the segment has no sidecar
    /// layers.
    fn list(
        &self,
        _prefix: Option<&OsPath>,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        Box::pin(futures::stream::empty())
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<OsPath>>,
    ) -> BoxStream<'static, object_store::Result<OsPath>> {
        Box::pin(locations.map(|_| Err(Self::not_supported("delete"))))
    }

    async fn list_with_delimiter(
        &self,
        _prefix: Option<&OsPath>,
    ) -> object_store::Result<ListResult> {
        Ok(ListResult {
            common_prefixes: vec![],
            objects: vec![],
        })
    }

    async fn copy_opts(
        &self,
        _from: &OsPath,
        _to: &OsPath,
        _options: CopyOptions,
    ) -> object_store::Result<()> {
        Err(Self::not_supported("copy"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    /// A container whose bytes are `0..200`, with a segment at 50..150.
    async fn fixture() -> SegmentStore {
        let inner = Arc::new(InMemory::new());
        let body: Vec<u8> = (0..200u32).map(|i| i as u8).collect();
        let path = OsPath::from("data.atlas");
        inner.put(&path, Bytes::from(body).into()).await.unwrap();
        SegmentStore::new(inner, path, 0, 50, 100)
    }

    async fn read(store: &SegmentStore, range: Option<GetRange>) -> Vec<u8> {
        let r = store
            .get_opts(
                &store.path(),
                GetOptions {
                    range,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        r.bytes().await.unwrap().to_vec()
    }

    #[tokio::test]
    async fn head_reports_the_segment_length() {
        let s = fixture().await;
        let meta = s.head(&s.path()).await.unwrap();
        assert_eq!(meta.size, 100);
    }

    #[tokio::test]
    async fn a_full_read_returns_only_the_segment() {
        let s = fixture().await;
        let got = read(&s, None).await;
        assert_eq!(got.len(), 100);
        assert_eq!(got[0], 50);
        assert_eq!(got[99], 149);
    }

    #[tokio::test]
    async fn bounded_ranges_are_translated() {
        let s = fixture().await;
        assert_eq!(
            read(&s, Some(GetRange::Bounded(0..4))).await,
            [50, 51, 52, 53]
        );
        assert_eq!(
            read(&s, Some(GetRange::Bounded(96..100))).await,
            [146, 147, 148, 149]
        );
    }

    #[tokio::test]
    async fn suffix_and_offset_ranges_are_translated() {
        let s = fixture().await;
        // The trailer read that opens every array-format file.
        assert_eq!(
            read(&s, Some(GetRange::Suffix(4))).await,
            [146, 147, 148, 149]
        );
        assert_eq!(read(&s, Some(GetRange::Offset(98))).await, [148, 149]);
    }

    #[tokio::test]
    async fn an_end_past_the_segment_is_clamped_not_leaked() {
        let s = fixture().await;
        // Without the clamp, this read reaches into the next segment.
        let got = read(&s, Some(GetRange::Bounded(98..500))).await;
        assert_eq!(got, [148, 149]);
    }

    #[tokio::test]
    async fn a_start_past_the_segment_is_an_error() {
        let s = fixture().await;
        let r = s
            .get_opts(
                &s.path(),
                GetOptions {
                    range: Some(GetRange::Bounded(100..104)),
                    ..Default::default()
                },
            )
            .await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn any_other_path_is_not_found() {
        let s = fixture().await;
        // This is the stats probe. It must not fail the open.
        let r = s.head(&OsPath::from("seg0.stats")).await;
        assert!(matches!(r, Err(object_store::Error::NotFound { .. })));
    }

    #[tokio::test]
    async fn listing_is_empty_so_no_sidecars_are_discovered() {
        use futures::StreamExt;
        let s = fixture().await;
        assert_eq!(s.list(None).collect::<Vec<_>>().await.len(), 0);
        assert!(
            s.list_with_delimiter(None)
                .await
                .unwrap()
                .objects
                .is_empty()
        );
    }

    #[tokio::test]
    async fn writes_are_refused() {
        let s = fixture().await;
        let r = s.put(&s.path(), PutPayload::from_static(b"x")).await;
        assert!(matches!(r, Err(object_store::Error::NotSupported { .. })));
    }
}
