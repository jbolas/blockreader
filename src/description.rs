//! Identifying detail about a byte source, for reports and diagnostics.

use std::fmt;
use std::path::PathBuf;

/// Where a byte source's data comes from.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Location {
    /// A file on a local filesystem.
    Path(PathBuf),
    /// A URL, such as `s3://bucket/key`. No backend in this release opens
    /// one; the variant exists so remote sources can arrive without a
    /// breaking change.
    Url(String),
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Path(p) => write!(f, "{}", p.display()),
            Self::Url(u) => f.write_str(u),
        }
    }
}

/// Identifying detail about a byte source, for reports and diagnostics.
///
/// Everything is optional: a backend reports what it knows. Consumers put
/// this in their output so an examiner can see what was read. The sector
/// size, and how it came to be known, is reported separately by
/// [`crate::BlockReader::sector_size`].
///
/// The struct is `#[non_exhaustive]`, so a backend outside this crate builds
/// one with [`SourceDescription::new`] or [`Default`].
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceDescription {
    /// Where the data came from.
    pub location: Option<Location>,
    /// The container format: `"raw"`, `"aff4"`, `"e01"`, or `"vmdk"`.
    pub format: Option<String>,
}

impl SourceDescription {
    /// A description with the given location and format.
    #[must_use]
    pub fn new(location: Option<Location>, format: Option<String>) -> Self {
        Self { location, format }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn default_description_knows_nothing() {
        let d = SourceDescription::default();
        assert_eq!(d.location, None);
        assert_eq!(d.format, None);
    }

    #[test]
    fn new_keeps_what_it_is_given() {
        let d = SourceDescription::new(
            Some(Location::Path(PathBuf::from("/evidence/disk.E01"))),
            Some("e01".to_string()),
        );
        assert_eq!(
            d.location,
            Some(Location::Path(PathBuf::from("/evidence/disk.E01")))
        );
        assert_eq!(d.format.as_deref(), Some("e01"));
    }

    #[test]
    fn locations_display_as_written() {
        assert_eq!(
            Location::Path(PathBuf::from("/evidence/disk.vmdk")).to_string(),
            "/evidence/disk.vmdk"
        );
        assert_eq!(
            Location::Url("s3://bucket/case/disk.E01".to_string()).to_string(),
            "s3://bucket/case/disk.E01"
        );
    }
}
