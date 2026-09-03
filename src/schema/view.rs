//! Borrowed views over an interned schema.
//!
//! The footer stores a schema as pairs of indices into its string and dtype
//! pools. A view resolves them on demand, so to inspect a schema copies no
//! name and allocates nothing.
//!
//! A schema names things. It says which arrays a dataset declares and what
//! type each holds, and the same for its attributes. It says nothing about
//! shape or chunking. Ask
//! [`DatasetView::array_layout`](crate::DatasetView::array_layout) for those.

use array_format::DType;
use indexmap::IndexMap;

use crate::format::footer::{CollectionFooter, InternedSchema};
use crate::schema::DatasetSchema;

/// The index resolved, or a panic. `CollectionFooter::validate` runs at every
/// decode and proves every index in the footer resolves.
const VALIDATED: &str = "the footer validated at decode";

/// What one dataset declares: its arrays and its attribute keys, with types.
///
/// Datasets that declare the same things share one interned schema. To find
/// two of them equal therefore costs no compare.
#[derive(Clone, Copy)]
pub struct SchemaView<'a> {
    footer: &'a CollectionFooter,
    schema: &'a InternedSchema,
}

impl<'a> SchemaView<'a> {
    pub(crate) fn new(footer: &'a CollectionFooter, schema: &'a InternedSchema) -> Self {
        Self { footer, schema }
    }

    /// How many arrays the dataset declares.
    pub fn len(&self) -> usize {
        self.schema.arrays.len()
    }

    /// Whether the dataset declares no array.
    pub fn is_empty(&self) -> bool {
        self.schema.arrays.is_empty()
    }

    /// Every array, in definition order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = ArrayMeta<'a>> + 'a {
        let (footer, schema) = (self.footer, self.schema);
        (0..schema.arrays.len()).map(move |position| ArrayMeta {
            footer,
            schema,
            position,
        })
    }

    /// Array names, in definition order.
    pub fn names(&self) -> impl ExactSizeIterator<Item = &'a str> + 'a {
        let (footer, schema) = (self.footer, self.schema);
        schema
            .arrays
            .iter()
            .map(move |&(name, _)| footer.string(name).expect(VALIDATED))
    }

    /// Position of `array` in definition order. `None` if the dataset does
    /// not declare it.
    ///
    /// The search covers this dataset's arrays, not the whole string pool. A
    /// wide collection therefore costs no more here than a narrow one.
    pub fn index_of(&self, array: &str) -> Option<usize> {
        self.schema
            .arrays
            .iter()
            .position(|&(name, _)| self.footer.string(name) == Some(array))
    }

    /// One array by name. `None` if the dataset does not declare it.
    pub fn get(&self, array: &str) -> Option<ArrayMeta<'a>> {
        self.get_index(self.index_of(array)?)
    }

    /// The array at `position` in definition order.
    pub fn get_index(&self, position: usize) -> Option<ArrayMeta<'a>> {
        (position < self.schema.arrays.len()).then_some(ArrayMeta {
            footer: self.footer,
            schema: self.schema,
            position,
        })
    }

    /// Dataset-level attribute keys, in the order somebody set them.
    pub fn attribute_names(&self) -> impl ExactSizeIterator<Item = &'a str> + 'a {
        let (footer, schema) = (self.footer, self.schema);
        schema
            .attrs
            .iter()
            .map(move |&(key, _)| footer.string(key).expect(VALIDATED))
    }

    /// This dataset's attribute keys with their declared types.
    ///
    /// A reader needs the type to read the value back out of a segment, which
    /// stores an `i64` for both an integer and a timestamp.
    pub fn attribute_pairs(&self) -> impl Iterator<Item = (&'a str, &'a DType)> + 'a {
        let (footer, schema) = (self.footer, self.schema);
        schema.attrs.iter().map(move |&(key, dtype)| {
            (
                footer.string(key).expect(VALIDATED),
                footer.dtype(dtype).expect(VALIDATED),
            )
        })
    }

    /// The type of one dataset-level attribute. `None` for a key this dataset
    /// does not carry.
    pub fn attribute_dtype(&self, key: &str) -> Option<&'a DType> {
        let &(_, dtype) = self
            .schema
            .attrs
            .iter()
            .find(|&&(k, _)| self.footer.string(k) == Some(key))?;
        self.footer.dtype(dtype)
    }

    /// An owned copy, with every name resolved.
    pub fn to_owned_schema(&self) -> DatasetSchema {
        DatasetSchema {
            arrays: self
                .iter()
                .map(|a| (a.name().to_string(), a.dtype().clone()))
                .collect(),
            attributes: self
                .schema
                .attrs
                .iter()
                .map(|&(key, dtype)| {
                    (
                        self.footer.string(key).expect(VALIDATED).to_string(),
                        self.footer.dtype(dtype).expect(VALIDATED).clone(),
                    )
                })
                .collect(),
            array_attributes: self
                .iter()
                .filter(|a| !a.attribute_names().is_empty())
                .map(|a| {
                    let keyed: IndexMap<String, DType> = a
                        .attribute_pairs()
                        .map(|(key, dtype)| (key.to_string(), dtype.clone()))
                        .collect();
                    (a.name().to_string(), keyed)
                })
                .collect(),
        }
    }
}

/// Two views are equal when they declare the same things, in the same order.
///
/// One footer interns each schema once, so two equal views of it are one
/// entry. That settles the equal case with no compare. Anything else compares
/// by content, because a foreign footer may hold an entry twice.
impl PartialEq for SchemaView<'_> {
    fn eq(&self, other: &Self) -> bool {
        if std::ptr::eq(self.footer, other.footer) && std::ptr::eq(self.schema, other.schema) {
            return true;
        }
        self.len() == other.len()
            && self.iter().zip(other.iter()).all(|(a, b)| a == b)
            && self.attribute_names().eq(other.attribute_names())
    }
}

impl std::fmt::Debug for SchemaView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchemaView")
            .field(
                "arrays",
                &self
                    .iter()
                    .map(|a| (a.name(), a.dtype()))
                    .collect::<Vec<_>>(),
            )
            .field("attributes", &self.attribute_names().collect::<Vec<_>>())
            .finish()
    }
}

/// One named array, as its dataset declares it.
///
/// A name and an element type. Shape and chunking are not here, because the
/// footer does not hold them. Ask
/// [`DatasetView::array_layout`](crate::DatasetView::array_layout).
#[derive(Clone, Copy)]
pub struct ArrayMeta<'a> {
    footer: &'a CollectionFooter,
    schema: &'a InternedSchema,
    position: usize,
}

impl<'a> ArrayMeta<'a> {
    /// The array's name.
    pub fn name(&self) -> &'a str {
        self.footer
            .string(self.schema.arrays[self.position].0)
            .expect(VALIDATED)
    }

    /// Element type.
    pub fn dtype(&self) -> &'a DType {
        self.footer
            .dtype(self.schema.arrays[self.position].1)
            .expect(VALIDATED)
    }

    /// Position in the dataset's definition order. The footer keys statistics
    /// and attributes on it.
    pub fn position(&self) -> usize {
        self.position
    }

    /// This array's attribute keys, in the order somebody set them.
    pub fn attribute_names(&self) -> Vec<&'a str> {
        self.attribute_pairs().map(|(key, _)| key).collect()
    }

    /// The type of one of this array's attributes. `None` for a key it does
    /// not carry.
    pub fn attribute_dtype(&self, key: &str) -> Option<&'a DType> {
        self.attribute_pairs()
            .find(|&(k, _)| k == key)
            .map(|(_, dtype)| dtype)
    }

    /// This array's attribute keys with their declared types.
    ///
    /// A reader needs the type to read the value back out of the segment.
    pub fn attribute_pairs(&self) -> impl Iterator<Item = (&'a str, &'a DType)> + 'a {
        let (footer, schema, position) = (self.footer, self.schema, self.position as u32);
        schema
            .array_attrs
            .iter()
            .filter(move |(p, _)| *p == position)
            .flat_map(|(_, pairs)| pairs.iter())
            .map(move |&(key, dtype)| {
                (
                    footer.string(key).expect(VALIDATED),
                    footer.dtype(dtype).expect(VALIDATED),
                )
            })
    }
}

/// Two arrays are equal when the name, the type, and the attribute keys are.
impl PartialEq for ArrayMeta<'_> {
    fn eq(&self, other: &Self) -> bool {
        if std::ptr::eq(self.footer, other.footer)
            && std::ptr::eq(self.schema, other.schema)
            && self.position == other.position
        {
            return true;
        }
        self.name() == other.name()
            && self.dtype() == other.dtype()
            && self.attribute_names() == other.attribute_names()
    }
}

impl std::fmt::Debug for ArrayMeta<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayMeta")
            .field("name", &self.name())
            .field("dtype", self.dtype())
            .field("attributes", &self.attribute_names())
            .finish()
    }
}
