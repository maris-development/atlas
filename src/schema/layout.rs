//! [`ArrayLayout`]: where one array's elements sit, read from its segment.

use array_format::FillValue;
use array_format::layout::ArrayMeta as StoredMeta;
use smallvec::SmallVec;
use smol_str::SmolStr;

/// Axes a layout holds inline. An array of a higher rank spills to the heap.
const INLINE_RANK: usize = 4;

/// Shape, chunking, dimension names, and fill value of one array.
///
/// The footer holds none of this. The segment that holds the data records it
/// already, so a reader opens that segment to answer. A segment covers one
/// variable across the whole collection, so one open answers for every
/// dataset that declares the array.
///
/// [`DatasetView::array_layout`](crate::DatasetView::array_layout) returns
/// one. Names and element types come from
/// [`schema`](crate::DatasetView::schema) instead, and cost nothing.
///
/// Every field is inline for the common rank, so to build one allocates
/// nothing. A layout exists per call, not per dataset, so the size it takes
/// costs nothing either.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrayLayout {
    shape: SmallVec<[usize; INLINE_RANK]>,
    chunk_shape: SmallVec<[usize; INLINE_RANK]>,
    dimension_names: SmallVec<[SmolStr; INLINE_RANK]>,
    fill_value: Option<FillValue>,
}

impl ArrayLayout {
    /// Reads the layout `array-format` recorded for one array.
    pub(crate) fn from_stored(meta: &StoredMeta) -> Self {
        Self {
            shape: meta.layout.shape.iter().map(|&s| s as usize).collect(),
            chunk_shape: meta
                .layout
                .storage
                .chunk_shape
                .iter()
                .map(|&s| s as usize)
                .collect(),
            dimension_names: meta
                .layout
                .dimension_names
                .iter()
                .map(SmolStr::new)
                .collect(),
            fill_value: meta.fill_value.clone(),
        }
    }

    /// Logical shape, one entry per axis.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// On-disk chunk shape, same rank as [`shape`](Self::shape). Equal to
    /// `shape` when the array is stored as one chunk.
    pub fn chunk_shape(&self) -> &[usize] {
        &self.chunk_shape
    }

    /// Named dimensions, one per axis, in the order of
    /// [`shape`](Self::shape).
    pub fn dimension_names(&self) -> Vec<&str> {
        self.dimension_names.iter().map(SmolStr::as_str).collect()
    }

    /// Value a read returns for an element nobody wrote. `None` when the
    /// array has no fill value.
    pub fn fill_value(&self) -> Option<&FillValue> {
        self.fill_value.as_ref()
    }

    /// How many elements the array holds across every axis.
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }
}
