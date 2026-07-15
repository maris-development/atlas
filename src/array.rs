use std::sync::Arc;

use array_format::{ArrayFile, DeltaCache, FileConfig, Lz4Codec, NoCompression, ZstdCodec};
use object_store::{ObjectStore, ObjectStoreExt, path::Path as OsPath};
use tokio::sync::RwLock as AsyncRwLock;
use tracing::debug;

use crate::{Error, Result, config::Codec};

/// Lazy handle to a single physical array file. The first call to [`get`]
/// performs the `head` / open-or-create round-trip; subsequent calls reuse
/// the cached `Arc<RwLock<ArrayFile>>` via `tokio::sync::OnceCell`.
pub(crate) struct AtlasArray {
    store: Arc<dyn ObjectStore>,
    codec: Codec,
    array_name: String,
    delta_cache: Arc<DeltaCache>,
    inner: tokio::sync::OnceCell<Arc<AsyncRwLock<ArrayFile>>>,
}

impl AtlasArray {
    pub(crate) fn new(
        store: Arc<dyn ObjectStore>,
        codec: Codec,
        array_name: String,
        delta_cache: Arc<DeltaCache>,
    ) -> Self {
        Self {
            store,
            codec,
            array_name,
            delta_cache,
            inner: tokio::sync::OnceCell::new(),
        }
    }

    /// Force initialization (open existing file or create a new one) and return
    /// the shared `Arc<RwLock<ArrayFile>>`.
    pub(crate) async fn get(&self) -> Result<Arc<AsyncRwLock<ArrayFile>>> {
        self.inner
            .get_or_try_init(|| async {
                let path = OsPath::from(format!("{}/data.af", self.array_name));
                let file = match self.store.head(&path).await {
                    Ok(_) => {
                        debug!(array = %self.array_name, codec = ?self.codec, "opening existing array file");
                        open_array_file(self.store.clone(), path, &self.codec, &self.delta_cache)
                            .await?
                    }
                    Err(object_store::Error::NotFound { .. }) => {
                        debug!(array = %self.array_name, codec = ?self.codec, "creating new array file");
                        create_array_file(self.store.clone(), path, &self.codec, &self.delta_cache)
                            .await?
                    }
                    Err(e) => return Err(Error::ObjectStore(e)),
                };
                Ok(Arc::new(AsyncRwLock::new(file)))
            })
            .await
            .map(Arc::clone)
    }

    /// Returns the underlying file if it has already been initialized; never
    /// triggers initialization. Used by tests and any caller that wants to
    /// observe lazy state without forcing I/O.
    pub(crate) fn try_get(&self) -> Option<Arc<AsyncRwLock<ArrayFile>>> {
        self.inner.get().map(Arc::clone)
    }

    /// Opens the file if it already exists on disk, but — unlike [`get`] —
    /// never *creates* it. Returns `Ok(None)` when no file is present, so
    /// read-only paths (e.g. reading attributes that may never have been
    /// written) don't write an empty base file as a side effect.
    ///
    /// If the file exists it is cached like [`get`]; a concurrent racing open
    /// is harmless (one instance wins the cache slot, both observe the same
    /// on-disk state).
    pub(crate) async fn get_existing(&self) -> Result<Option<Arc<AsyncRwLock<ArrayFile>>>> {
        if let Some(arc) = self.inner.get() {
            return Ok(Some(Arc::clone(arc)));
        }
        let path = OsPath::from(format!("{}/data.af", self.array_name));
        match self.store.head(&path).await {
            Ok(_) => {
                debug!(array = %self.array_name, codec = ?self.codec, "opening existing array file (read-only)");
                let file =
                    open_array_file(self.store.clone(), path, &self.codec, &self.delta_cache).await?;
                let arc = Arc::new(AsyncRwLock::new(file));
                // Best-effort cache; if another task already populated the cell,
                // its instance wins and ours is dropped.
                let _ = self.inner.set(arc);
                Ok(self.inner.get().map(Arc::clone))
            }
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(Error::ObjectStore(e)),
        }
    }
}

pub(crate) async fn open_array_file(
    store: Arc<dyn ObjectStore>,
    path: OsPath,
    codec: &Codec,
    delta: &Arc<DeltaCache>,
) -> Result<ArrayFile> {
    Ok(match codec {
        Codec::Zstd => {
            ArrayFile::open(store, path, file_config(ZstdCodec::default(), delta)).await?
        }
        Codec::Lz4 => ArrayFile::open(store, path, file_config(Lz4Codec, delta)).await?,
        Codec::Uncompressed => {
            ArrayFile::open(store, path, file_config(NoCompression, delta)).await?
        }
    })
}

pub(crate) async fn create_array_file(
    store: Arc<dyn ObjectStore>,
    path: OsPath,
    codec: &Codec,
    delta: &Arc<DeltaCache>,
) -> Result<ArrayFile> {
    Ok(match codec {
        Codec::Zstd => {
            ArrayFile::create(store, path, file_config(ZstdCodec::default(), delta)).await?
        }
        Codec::Lz4 => ArrayFile::create(store, path, file_config(Lz4Codec, delta)).await?,
        Codec::Uncompressed => {
            ArrayFile::create(store, path, file_config(NoCompression, delta)).await?
        }
    })
}

fn file_config<C: array_format::CompressionCodec>(
    codec: C,
    delta: &Arc<DeltaCache>,
) -> FileConfig<C> {
    FileConfig {
        codec,
        block_target_size: 8 * 1024 * 1024,
        cache_capacity: 256 * 1024 * 1024,
        io_cache_capacity: 64 * 1024 * 1024,
        cache: Some(Arc::clone(delta)),
    }
}
