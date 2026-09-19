//! A [`BlockReader`] over a plain file: raw dd images and split-free captures.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::{BlockReader, DEFAULT_SECTOR_SIZE, Error, Result, SourceDescription};

/// A byte source backed by a plain file.
///
/// The trivial case of a raw `dd` image or any file whose bytes are
/// the disk's exact bytes. This allows consumers of [`BlockReader`] to be
/// tested without any container format, and so this crate has one working
/// implementation of its own trait.
///
/// # Sector size
///
/// A plain file carries no record of the sector size of the device it came
/// from, so the default is [`DEFAULT_SECTOR_SIZE`] and
/// [`SourceDescription::sector_size_assumed`] is set. A caller that knows
/// better sets it with [`with_sector_size`](Self::with_sector_size).
#[derive(Debug)]
pub struct FileSource {
    /// Seek and read must happen together, so the handle is guarded rather
    /// than the two calls being separately atomic.
    file: Mutex<File>,
    path: PathBuf,
    size: u64,
    sector_size: u32,
    sector_size_assumed: bool,
}

impl FileSource {
    /// Open `path` read-only.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be opened or its length
    /// cannot be determined.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let size = file.metadata()?.len();
        Ok(Self {
            file: Mutex::new(file),
            path,
            size,
            sector_size: DEFAULT_SECTOR_SIZE,
            sector_size_assumed: true,
        })
    }

    /// Declare the real sector size, overriding the assumed default.
    /// Clears [`SourceDescription::sector_size_assumed`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSectorSize`] if `bytes` is zero or not a power
    /// of two. Partition-table addressing multiplies by this value, so a
    /// nonsensical one would silently produce nonsensical offsets.
    pub fn with_sector_size(mut self, bytes: u32) -> Result<Self> {
        if bytes == 0 || !bytes.is_power_of_two() {
            return Err(Error::InvalidSectorSize { bytes });
        }
        self.sector_size = bytes;
        self.sector_size_assumed = false;
        Ok(self)
    }

    /// The path this source was opened from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// A second handle on the same file, with its own descriptor.
    ///
    /// Sharing one `FileSource` across threads is safe but serializes reads
    /// on its lock. Where genuine parallel throughput is wanted, give each
    /// worker its own handle from here.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the descriptor cannot be duplicated.
    pub fn try_clone(&self) -> Result<Self> {
        let file = self
            .file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .try_clone()?;
        Ok(Self {
            file: Mutex::new(file),
            path: self.path.clone(),
            size: self.size,
            sector_size: self.sector_size,
            sector_size_assumed: self.sector_size_assumed,
        })
    }
}

impl BlockReader for FileSource {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        if offset >= self.size || buf.is_empty() {
            return Ok(0);
        }
        // A poisoned lock means another thread panicked mid-read. The file
        // itself is unharmed, so recover the guard rather than propagating.
        let mut file = self
            .file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        file.seek(SeekFrom::Start(offset))?;
        // A single `read` may return short for reasons that are not errors.
        // Callers wanting a full buffer use `read_exact_at`.
        let n = file.read(buf)?;
        Ok(n)
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn sector_size(&self) -> u32 {
        self.sector_size
    }

    fn describe(&self) -> SourceDescription {
        SourceDescription {
            path: Some(self.path.clone()),
            format: Some("raw".to_string()),
            sector_size_assumed: self.sector_size_assumed,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::read_exact_at;
    use std::io::Write;

    /// Writes a temp file of `len` bytes with a recognisable pattern.
    fn temp_image(name: &str, len: usize) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("blockreader-test-{name}-{}", std::process::id()));
        let data: Vec<u8> = (0..len)
            .map(|i| u8::try_from(i % 251).unwrap_or(0))
            .collect();
        let mut f = File::create(&path).unwrap();
        f.write_all(&data).unwrap();
        f.sync_all().unwrap();
        path
    }

    #[test]
    fn reads_at_offset() {
        let path = temp_image("offset", 4096);
        let s = FileSource::open(&path).unwrap();
        assert_eq!(s.size(), 4096);

        let mut buf = [0u8; 16];
        read_exact_at(&s, 1000, &mut buf).unwrap();
        let expected: Vec<u8> = (1000..1016u32)
            .map(|i| u8::try_from(i % 251).unwrap_or(0))
            .collect();
        assert_eq!(buf.as_slice(), expected.as_slice());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_past_end_returns_zero_not_padding() {
        let path = temp_image("eof", 100);
        let s = FileSource::open(&path).unwrap();

        let mut buf = [0xAAu8; 32];
        assert_eq!(s.read_at(100, &mut buf).unwrap(), 0);
        assert_eq!(buf[0], 0xAA, "nothing written when nothing read");

        // Straddling the end is a short read, not an error and not padded.
        let n = s.read_at(90, &mut buf).unwrap();
        assert_eq!(n, 10);
        assert_eq!(buf[10], 0xAA, "no zero-fill past the real data");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_exact_at_past_end_is_an_error() {
        let path = temp_image("exact-eof", 100);
        let s = FileSource::open(&path).unwrap();
        let mut buf = [0u8; 32];
        let err = read_exact_at(&s, 90, &mut buf).unwrap_err();
        assert!(
            matches!(err, Error::UnexpectedEof { got: 10, .. }),
            "{err:?}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn sector_size_defaults_to_512_and_says_it_assumed() {
        let path = temp_image("sector-default", 512);
        let s = FileSource::open(&path).unwrap();
        assert_eq!(s.sector_size(), DEFAULT_SECTOR_SIZE);
        assert!(s.describe().sector_size_assumed);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn caller_can_set_sector_size() {
        let path = temp_image("sector-set", 512);
        let s = FileSource::open(&path)
            .unwrap()
            .with_sector_size(4096)
            .unwrap();
        assert_eq!(s.sector_size(), 4096);
        assert!(
            !s.describe().sector_size_assumed,
            "an asserted value is not an assumed one"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn nonsensical_sector_sizes_are_rejected() {
        let path = temp_image("sector-bad", 512);
        for bad in [0u32, 3, 500, 1000] {
            let r = FileSource::open(&path).unwrap().with_sector_size(bad);
            assert!(
                matches!(r, Err(Error::InvalidSectorSize { bytes: b }) if b == bad),
                "{bad} should be rejected as an invalid sector size"
            );
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn describe_reports_path_and_format() {
        let path = temp_image("describe", 64);
        let s = FileSource::open(&path).unwrap();
        let d = s.describe();
        assert_eq!(d.path.as_deref(), Some(path.as_path()));
        assert_eq!(d.format.as_deref(), Some("raw"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn clones_read_independently() {
        let path = temp_image("clone", 4096);
        let s = FileSource::open(&path)
            .unwrap()
            .with_sector_size(4096)
            .unwrap();
        let a = s;
        let b = a.try_clone().expect("a file source can always clone");

        // Interleaved reads at different offsets must not disturb each other.
        let mut buf_a = [0u8; 8];
        let mut buf_b = [0u8; 8];
        read_exact_at(&a, 0, &mut buf_a).unwrap();
        read_exact_at(&b, 2048, &mut buf_b).unwrap();
        read_exact_at(&a, 8, &mut buf_a).unwrap();

        let expect_a: Vec<u8> = (8..16u32)
            .map(|i| u8::try_from(i % 251).unwrap_or(0))
            .collect();
        let expect_b: Vec<u8> = (2048..2056u32)
            .map(|i| u8::try_from(i % 251).unwrap_or(0))
            .collect();
        assert_eq!(buf_a.as_slice(), expect_a.as_slice());
        assert_eq!(buf_b.as_slice(), expect_b.as_slice());

        assert_eq!(
            b.sector_size(),
            4096,
            "clone keeps the asserted sector size"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_file_is_an_io_error() {
        let err = FileSource::open("/nonexistent/blockreader/test/image.dd").unwrap_err();
        assert!(matches!(err, Error::Io(_)), "{err:?}");
    }

    #[test]
    fn empty_file_reads_nothing() {
        let path = temp_image("empty", 0);
        let s = FileSource::open(&path).unwrap();
        assert_eq!(s.size(), 0);
        let mut buf = [0u8; 8];
        assert_eq!(s.read_at(0, &mut buf).unwrap(), 0);
        std::fs::remove_file(&path).ok();
    }
}
