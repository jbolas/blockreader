//! The sector size of a byte source, and how it came to be known.

use crate::{Error, Result};

/// The sector size assumed when nobody has determined the real one.
pub const DEFAULT_SECTOR_SIZE: u32 = 512;

/// How a [`SectorSize`] came to be known.
///
/// Partition-table offsets are computed in sectors, so a report built on a
/// sector size must say whether that size was observed or assumed.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SectorSizeBasis {
    /// Nobody determined it: the [`DEFAULT_SECTOR_SIZE`] default.
    Assumed,
    /// The container records it, as E01's volume section does.
    Recorded,
    /// A caller declared it.
    Asserted,
}

/// A sector size in bytes, together with how it came to be known.
///
/// The size and its basis travel as one value, so they cannot disagree. The
/// size is always a non-zero power of two: [`SectorSize::new`] rejects
/// anything else, and partition arithmetic multiplies by it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SectorSize {
    bytes: u32,
    basis: SectorSizeBasis,
}

impl SectorSize {
    /// 512 bytes, assumed. What a backend reports when it knows nothing better.
    pub const DEFAULT: SectorSize = SectorSize {
        bytes: DEFAULT_SECTOR_SIZE,
        basis: SectorSizeBasis::Assumed,
    };

    /// A sector size of `bytes`, known on the given `basis`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSectorSize`] if `bytes` is zero or not a power
    /// of two.
    pub fn new(bytes: u32, basis: SectorSizeBasis) -> Result<Self> {
        if !bytes.is_power_of_two() {
            return Err(Error::InvalidSectorSize { bytes });
        }
        Ok(Self { bytes, basis })
    }

    /// The size in bytes.
    #[must_use]
    pub fn bytes(&self) -> u32 {
        self.bytes
    }

    /// How the size came to be known.
    #[must_use]
    pub fn basis(&self) -> SectorSizeBasis {
        self.basis
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::Error;

    #[test]
    fn default_is_512_and_assumed() {
        assert_eq!(SectorSize::DEFAULT.bytes(), 512);
        assert_eq!(SectorSize::DEFAULT.basis(), SectorSizeBasis::Assumed);
        assert_eq!(DEFAULT_SECTOR_SIZE, 512);
    }

    #[test]
    fn accepts_powers_of_two_and_keeps_the_basis() {
        for (bytes, basis) in [
            (512, SectorSizeBasis::Recorded),
            (4096, SectorSizeBasis::Asserted),
            (1, SectorSizeBasis::Assumed),
        ] {
            let s = SectorSize::new(bytes, basis).unwrap();
            assert_eq!(s.bytes(), bytes);
            assert_eq!(s.basis(), basis);
        }
    }

    #[test]
    fn rejects_zero_and_non_powers_of_two() {
        for bad in [0u32, 3, 500, 1000, 520] {
            let r = SectorSize::new(bad, SectorSizeBasis::Asserted);
            assert!(
                matches!(r, Err(Error::InvalidSectorSize { bytes }) if bytes == bad),
                "{bad} must be rejected, got {r:?}"
            );
        }
    }
}
