use atlas::DType;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// Parse a numpy/atlas-style dtype string into an `atlas::DType`.
///
/// Accepted forms (case-insensitive):
///   - `bool`
///   - `i8`/`int8`, `i16`/`int16`, `i32`/`int32`, `i64`/`int64`
///   - `u8`/`uint8`, `u16`/`uint16`, `u32`/`uint32`, `u64`/`uint64`
///   - `f32`/`float32`, `f64`/`float64`
///   - `string`/`str`
///   - `binary`/`bytes`
///   - `timestamp_ns`/`timestamp_nanoseconds`/`datetime64[ns]`
///   - `list[<inner>]`
///   - `fixed_size_list[<inner>,<n>]`
pub fn parse_dtype(s: &str) -> PyResult<DType> {
    parse_dtype_inner(s.trim())
        .ok_or_else(|| PyValueError::new_err(format!("unknown dtype string: {s:?}")))
}

fn parse_dtype_inner(s: &str) -> Option<DType> {
    let lower = s.to_ascii_lowercase();

    if let Some(inner) = strip_prefix_suffix(&lower, "list[", "]") {
        let child = parse_dtype_inner(inner)?;
        return Some(DType::List {
            child: Box::new(child),
        });
    }
    if let Some(inner) = strip_prefix_suffix(&lower, "fixed_size_list[", "]") {
        let (child_str, size_str) = inner.rsplit_once(',')?;
        let child = parse_dtype_inner(child_str.trim())?;
        let size: u32 = size_str.trim().parse().ok()?;
        return Some(DType::FixedSizeList {
            child: Box::new(child),
            size,
        });
    }

    Some(match lower.as_str() {
        "bool" => DType::Bool,
        "i8" | "int8" => DType::Int8,
        "i16" | "int16" => DType::Int16,
        "i32" | "int32" => DType::Int32,
        "i64" | "int64" => DType::Int64,
        "u8" | "uint8" => DType::UInt8,
        "u16" | "uint16" => DType::UInt16,
        "u32" | "uint32" => DType::UInt32,
        "u64" | "uint64" => DType::UInt64,
        "f32" | "float32" => DType::Float32,
        "f64" | "float64" => DType::Float64,
        "string" | "str" => DType::String,
        "binary" | "bytes" => DType::Binary,
        "timestamp_ns" | "timestamp_nanoseconds" | "datetime64[ns]" => DType::TimestampNs,
        _ => return None,
    })
}

fn strip_prefix_suffix<'a>(s: &'a str, prefix: &str, suffix: &str) -> Option<&'a str> {
    s.strip_prefix(prefix)?.strip_suffix(suffix)
}

/// Render a `DType` as a stable string (matches the parser input vocabulary).
pub fn dtype_to_string(dtype: &DType) -> String {
    match dtype {
        DType::Bool => "bool".into(),
        DType::Int8 => "int8".into(),
        DType::Int16 => "int16".into(),
        DType::Int32 => "int32".into(),
        DType::Int64 => "int64".into(),
        DType::UInt8 => "uint8".into(),
        DType::UInt16 => "uint16".into(),
        DType::UInt32 => "uint32".into(),
        DType::UInt64 => "uint64".into(),
        DType::Float32 => "float32".into(),
        DType::Float64 => "float64".into(),
        DType::String => "string".into(),
        DType::Binary => "binary".into(),
        DType::TimestampNs => "timestamp_nanoseconds".into(),
        DType::List { child } => format!("list[{}]", dtype_to_string(child)),
        DType::FixedSizeList { child, size } => {
            format!("fixed_size_list[{},{}]", dtype_to_string(child), size)
        }
    }
}

/// Expands `$cb!(Variant, RustType, "numpy_dtype_name")` for every numeric variant.
/// Used in read/write paths to generate a match expression over `atlas::DType`.
#[macro_export]
macro_rules! for_each_numeric_dtype {
    ($cb:ident) => {
        $cb!(Bool, bool);
        $cb!(Int8, i8);
        $cb!(Int16, i16);
        $cb!(Int32, i32);
        $cb!(Int64, i64);
        $cb!(UInt8, u8);
        $cb!(UInt16, u16);
        $cb!(UInt32, u32);
        $cb!(UInt64, u64);
        $cb!(Float32, f32);
        $cb!(Float64, f64);
    };
}
