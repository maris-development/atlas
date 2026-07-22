//! Bulk cross-dataset reads: the same array slice from many datasets at once.
//!
//! One Rust call replaces N Python → Rust → tokio round trips, running the
//! per-dataset reads concurrently the way a tuned dask threadpool would.

use std::sync::Arc;

use tracing::instrument;

use super::Atlas;
use crate::{Error, Result, config::Codec};

impl Atlas {
    /// Bulk read the same slice of `array` from many datasets that share its
    /// physical file. Runs at most `num_cpus` reads concurrently — matching
    /// what a well-tuned dask threadpool would do — to keep
    /// `tokio::task::spawn_blocking`'s decompression pool from oversubscribing
    /// the actual CPU cores.
    ///
    /// This exists because `open_as_many_xarray_dataset` over N datasets used to incur N
    /// separate Python → Rust → tokio::block_on transitions plus Python-side
    /// dask graph overhead. One call here replaces all of that and gets the
    /// same parallelism dask was providing — but in pure Rust, with no GIL
    /// involvement until the results return.
    ///
    /// `start` and `shape` follow the same conventions as
    /// [`DatasetView::read_array`]: empty `start` + empty `shape` mean the
    /// full array. Per-dataset entries that don't declare `array` are
    /// returned as `None`.
    #[instrument(skip(self, dataset_names), fields(array = %array, n = dataset_names.len()))]
    pub async fn read_array_across<T: array_format::ArrayElement + Send + Sync + 'static>(
        &self,
        array: &str,
        dataset_names: &[String],
        start: Vec<usize>,
        shape: Vec<usize>,
    ) -> Result<Vec<Option<ndarray::ArcArray<T, ndarray::IxDyn>>>> {
        // Discover the codec for `array` from any dataset that defines it,
        // and pre-flight which dataset names declare it.
        let (codec, present): (Codec, Vec<bool>) = {
            let meta = self.meta.lock();
            let mut codec: Option<Codec> = None;
            let mut present: Vec<bool> = Vec::with_capacity(dataset_names.len());
            for name in dataset_names {
                let has = meta
                    .live_schema(name)
                    .and_then(|d| d.arrays.get(array))
                    .map(|schema| {
                        codec.get_or_insert(schema.codec);
                        true
                    })
                    .unwrap_or(false);
                present.push(has);
            }
            let codec = codec.ok_or_else(|| Error::ArrayNotFound(array.to_string()))?;
            (codec, present)
        };

        let handle = self.cache.get_or_insert(&self.store, array, &codec);
        let arc = handle.get().await?;

        // Spawn each per-dataset read as a top-level tokio task so the
        // multi-thread runtime distributes them across worker threads.
        // A semaphore caps in-flight tasks at `concurrency` (≈ num_cpus)
        // to keep `tokio::task::spawn_blocking`'s decompression pool from
        // oversubscribing the actual CPU cores.
        let concurrency = num_cpus::get().max(1);
        let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
        let mut joinset = tokio::task::JoinSet::new();
        for (idx, (name, &has)) in dataset_names.iter().zip(present.iter()).enumerate() {
            if !has {
                continue;
            }
            let permit = Arc::clone(&sem)
                .acquire_owned()
                .await
                .expect("semaphore never closed");
            let arc = Arc::clone(&arc);
            let name = name.clone();
            let start = start.clone();
            let shape = shape.clone();
            joinset.spawn(async move {
                let _permit = permit;
                let guard = arc.read().await;
                let res = guard.read_array::<T>(&name, start, shape).await;
                (idx, res)
            });
        }

        let mut out: Vec<Option<ndarray::ArcArray<T, ndarray::IxDyn>>> =
            (0..dataset_names.len()).map(|_| None).collect();
        while let Some(join_res) = joinset.join_next().await {
            let (idx, read_res) = join_res
                .map_err(|e| Error::Internal(format!("read task failed: {e}")))?;
            out[idx] = Some(read_res?);
        }
        Ok(out)
    }

    /// Like [`Atlas::read_array_across`] but returns one stacked
    /// `(len(dataset_names), *per_dataset_shape)` `ndarray::Array` instead of
    /// a `Vec` of per-dataset arrays.
    ///
    /// The output buffer is pre-allocated once; each parallel read writes its
    /// row in as the task completes, overlapping the serial copy with the
    /// remaining in-flight reads. Saves the ~5.7 GiB of memory copies that
    /// the Python-side `np.stack` on the per-dataset list would do on a
    /// 1000-dataset gridded workload.
    ///
    /// Errors if any listed dataset doesn't declare `array` — the stacked
    /// representation has no positional "missing" sentinel.
    #[instrument(skip(self, dataset_names), fields(array = %array, n = dataset_names.len()))]
    pub async fn read_array_across_stacked<
        T: array_format::ArrayElement + Send + Sync + Clone + 'static,
    >(
        &self,
        array: &str,
        dataset_names: &[String],
        start: Vec<usize>,
        shape: Vec<usize>,
    ) -> Result<ndarray::Array<T, ndarray::IxDyn>> {
        if dataset_names.is_empty() {
            return Err(Error::ArrayNotFound(array.to_string()));
        }

        // Discover the codec and verify ALL listed datasets declare the array.
        let codec: Codec = {
            let meta = self.meta.lock();
            let mut codec: Option<Codec> = None;
            for name in dataset_names {
                let schema = meta
                    .live_schema(name)
                    .and_then(|d| d.arrays.get(array))
                    .ok_or_else(|| {
                        Error::ArrayNotFound(format!("{array} (in dataset {name})"))
                    })?;
                codec.get_or_insert(schema.codec);
            }
            codec.expect("non-empty dataset_names, all schemas present")
        };

        let handle = self.cache.get_or_insert(&self.store, array, &codec);
        let arc_file = handle.get().await?;

        // Read the first dataset synchronously to discover the per-dataset
        // shape (after `start`/`shape` slicing) so we can pre-allocate the
        // stacked output. Then write its row in.
        let first_arr = {
            let guard = arc_file.read().await;
            guard
                .read_array::<T>(&dataset_names[0], start.clone(), shape.clone())
                .await?
        };
        let per_dataset_shape: Vec<usize> = first_arr.shape().to_vec();
        let n = dataset_names.len();
        let mut out_shape = Vec::with_capacity(per_dataset_shape.len() + 1);
        out_shape.push(n);
        out_shape.extend(&per_dataset_shape);

        // Allocate the output as a flat `Vec<T>` of N * per_dataset_elements
        // entries. We bypass `Array::default` so we don't pay an extra
        // zero-fill memset of the entire buffer — every slot will be written
        // by either the first-dataset read above or a spawned task below.
        let per_dataset_elements: usize = per_dataset_shape.iter().product();
        let total_elements = n * per_dataset_elements;
        let mut buf: Vec<T> = Vec::with_capacity(total_elements);
        // SAFETY: every element is written exactly once before we hand the
        // Vec to `Array::from_shape_vec`. Until then, the uninitialised
        // portion is never read.
        unsafe { buf.set_len(total_elements) };

        // Helper: copy a per-dataset ArcArray into row `idx` of the flat
        // buffer via `copy_from_slice` (memcpy). Both source and destination
        // are C-order contiguous (array-format's `assemble_nd` builds via
        // `Array::from_elem`, our Vec is contiguous by construction).
        fn write_row<T: array_format::ArrayElement + Clone>(
            buf: &mut [T],
            idx: usize,
            per_row: usize,
            src: &ndarray::ArcArray<T, ndarray::IxDyn>,
        ) -> Result<()> {
            let src_slice = src
                .as_slice()
                .ok_or_else(|| Error::Internal("per-dataset read returned non-contiguous array".into()))?;
            let dst = &mut buf[idx * per_row..(idx + 1) * per_row];
            dst.clone_from_slice(src_slice);
            Ok(())
        }

        write_row(&mut buf, 0, per_dataset_elements, &first_arr)?;
        drop(first_arr);

        // Spawn the remaining N-1 reads with the same concurrency-capped
        // pattern as `read_array_across`.
        let concurrency = num_cpus::get().max(1);
        let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
        let mut joinset = tokio::task::JoinSet::new();
        for (idx, name) in dataset_names.iter().enumerate().skip(1) {
            let permit = Arc::clone(&sem)
                .acquire_owned()
                .await
                .expect("semaphore never closed");
            let arc = Arc::clone(&arc_file);
            let name = name.clone();
            let start = start.clone();
            let shape = shape.clone();
            joinset.spawn(async move {
                let _permit = permit;
                let guard = arc.read().await;
                let res = guard.read_array::<T>(&name, start, shape).await;
                (idx, res)
            });
        }

        // As tasks complete, memcpy their row into the pre-allocated buffer.
        // The serial memcpy overlaps with the remaining in-flight parallel
        // reads happening on the runtime's other workers.
        while let Some(join_res) = joinset.join_next().await {
            let (idx, read_res) = join_res
                .map_err(|e| Error::Internal(format!("read task failed: {e}")))?;
            let arr = read_res?;
            write_row(&mut buf, idx, per_dataset_elements, &arr)?;
        }

        ndarray::Array::from_shape_vec(ndarray::IxDyn(&out_shape), buf)
            .map_err(|e| Error::Internal(format!("stacked output shape mismatch: {e}")))
    }
}
