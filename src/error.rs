use std::fmt;

/// The result type for byte-source operations.
pub type Result<T> = std::result::Result<T, Error>;

/// A failure reading from a byte source.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The host filesystem failed.
    Io(std::io::Error),

    /// The source ended before [`crate::read_exact_at`] could fill its buffer.
    UnexpectedEof {
        /// Where the read began.
        offset: u64,
        /// How many bytes were wanted.
        wanted: usize,
        /// How many were obtained before the source ended.
        got: usize,
    },

    /// A sector size that cannot describe a device.
    ///
    /// Partition addressing multiplies by this value, so a zero or
    /// non-power-of-two would silently produce nonsensical offsets.
    InvalidSectorSize {
        /// The value offered.
        bytes: u32,
    },

    /// The content of a range is unknown, so no bytes can be returned for it.
    ///
    /// `offset` and `len` describe the whole unknown range containing the
    /// requested offset, as far as the backend knows it. A caller that wants
    /// to continue resumes reading at `offset + len`. See the rules on
    /// [`crate::BlockReader`].
    UnknownContent {
        /// Where the unknown range begins.
        offset: u64,
        /// How long it is.
        len: u64,
        /// Why its content is unknown.
        cause: UnknownCause,
    },

    /// A backend-specific failure.
    ///
    /// This is how a container format reports its own errors without this
    /// crate depending on it. An AFF4 backend puts an `aff4tools::Error`
    /// here; an E01 backend puts its own.
    Backend(Box<dyn std::error::Error + Send + Sync>),
}

/// Why the content of a range is unknown.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnknownCause {
    /// The acquisition never read this range, as with AFF4 `aff4:UnknownData`.
    NotAcquired,
    /// The source medium returned read errors when it was acquired, as with
    /// AFF4 `aff4:UnreadableData`.
    UnreadableAtAcquisition,
    /// The container's stored bytes fail an integrity check, as with an E01
    /// chunk that will not decompress or fails its checksum.
    FailedIntegrity,
}

impl fmt::Display for UnknownCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NotAcquired => "the acquisition did not read this range",
            Self::UnreadableAtAcquisition => {
                "the source medium could not be read here when it was acquired"
            }
            Self::FailedIntegrity => "the container's stored data fails an integrity check",
        })
    }
}

impl Error {
    /// Wrap a backend's own error.
    pub fn backend<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Backend(Box::new(source))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "read failed: {e}"),
            Self::UnexpectedEof {
                offset,
                wanted,
                got,
            } => write!(
                f,
                "source ended early at offset {offset}: wanted {wanted} bytes, got {got}"
            ),
            Self::InvalidSectorSize { bytes } => write!(
                f,
                "{bytes} is not a valid sector size; it must be a non-zero power of two"
            ),
            Self::UnknownContent { offset, len, cause } => write!(
                f,
                "content of bytes {offset}..{} is unknown: {cause}",
                offset.saturating_add(*len)
            ),
            Self::Backend(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Backend(e) => Some(e.as_ref()),
            Self::UnexpectedEof { .. }
            | Self::InvalidSectorSize { .. }
            | Self::UnknownContent { .. } => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Fake;

    impl fmt::Display for Fake {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "container is not readable")
        }
    }

    impl std::error::Error for Fake {}

    #[test]
    fn backend_errors_keep_their_message_and_chain() {
        let e = Error::backend(Fake);
        assert_eq!(e.to_string(), "container is not readable");
        assert!(
            std::error::Error::source(&e).is_some(),
            "backend cause stays reachable"
        );
    }

    #[test]
    fn messages_name_the_numbers_an_examiner_needs() {
        let e = Error::UnexpectedEof {
            offset: 4096,
            wanted: 32,
            got: 10,
        };
        let s = e.to_string();
        assert!(
            s.contains("4096") && s.contains("32") && s.contains("10"),
            "{s}"
        );

        let e = Error::UnknownContent {
            offset: 32_768,
            len: 32_768,
            cause: UnknownCause::FailedIntegrity,
        };
        let s = e.to_string();
        assert!(
            s.contains("32768") && s.contains("65536") && s.contains("integrity"),
            "{s}"
        );
    }

    #[test]
    fn every_cause_explains_itself() {
        for cause in [
            UnknownCause::NotAcquired,
            UnknownCause::UnreadableAtAcquisition,
            UnknownCause::FailedIntegrity,
        ] {
            assert!(!cause.to_string().is_empty());
        }
    }

    #[test]
    fn io_errors_convert_and_chain() {
        let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let e: Error = io.into();
        assert!(matches!(e, Error::Io(_)));
        assert!(std::error::Error::source(&e).is_some());
    }
}
