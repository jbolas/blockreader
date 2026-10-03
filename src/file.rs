//! A [`BlockReader`] over a plain file: raw dd images and split-free captures.

use std::fs::File;
use std::path::{Path, PathBuf};

use crate::{BlockReader, Error, Location, Result, SectorSize, SectorSizeBasis, SourceDescription};

#[cfg(not(any(unix, windows)))]
compile_error!("FileSource needs positioned reads, which std provides only on Unix and Windows");

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
/// from, so the default is [`SectorSize::DEFAULT`]: 512 bytes, assumed. A
/// caller that knows better sets it with
/// [`with_sector_size`](Self::with_sector_size), which records it as
/// asserted.
#[derive(Debug)]
pub struct FileSource {
    /// Read with positioned reads, which never move a shared cursor, so one
    /// handle serves concurrent readers without a lock.
    file: File,
    path: PathBuf,
    size: u64,
    sector_size: SectorSize,
}

impl FileSource {
    /// Open `path` read-only.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be opened or its length
    /// cannot be determined.
    ///
    /// Returns [`Error::Io`] of kind `InvalidInput` if `path` is not a
    /// regular file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let meta = file.metadata()?;
        // A device node's metadata reports a length of 0, so without this
        // check a block device would read as an empty image. Finding a
        // device's real size needs an ioctl on macOS, which this crate's
        // no-unsafe, no-dependency rules exclude, so devices are refused.
        if !meta.is_file() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "{} is not a regular file; FileSource reads only regular files, \
                     not directories or devices",
                    path.display()
                ),
            )));
        }
        Ok(Self {
            file,
            path,
            size: meta.len(),
            sector_size: SectorSize::DEFAULT,
        })
    }

    /// Declare the real sector size. It is reported as
    /// [`SectorSizeBasis::Asserted`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSectorSize`] if `bytes` is zero or not a power
    /// of two. Partition-table addressing multiplies by this value, so a
    /// nonsensical one would silently produce nonsensical offsets.
    pub fn with_sector_size(mut self, bytes: u32) -> Result<Self> {
        self.sector_size = SectorSize::new(bytes, SectorSizeBasis::Asserted)?;
        Ok(self)
    }

    /// The path this source was opened from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// A second handle on the same file, with its own descriptor.
    ///
    /// Reads need no lock, so sharing one `FileSource` across threads is
    /// already concurrent. A separate handle is still useful where a caller
    /// wants independent ownership.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the descriptor cannot be duplicated.
    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            file: self.file.try_clone()?,
            path: self.path.clone(),
            size: self.size,
            sector_size: self.sector_size,
        })
    }
}

impl BlockReader for FileSource {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        if offset >= self.size || buf.is_empty() {
            return Ok(0);
        }
        // A single positioned read may return short for reasons that are
        // not errors. Callers wanting a full buffer use `read_exact_at`.
        loop {
            match positioned_read(&self.file, buf, offset) {
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                other => return Ok(other?),
            }
        }
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn sector_size(&self) -> SectorSize {
        self.sector_size
    }

    fn describe(&self) -> SourceDescription {
        SourceDescription::new(
            Some(Location::Path(self.path.clone())),
            Some("raw".to_string()),
        )
    }
}

/// Read into `buf` at `offset` without moving any shared file cursor.
#[cfg(unix)]
fn positioned_read(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buf, offset)
}

/// Read into `buf` at `offset`. On Windows, `seek_read` moves the handle's
/// cursor, but every read names its offset, so concurrent reads are still
/// correct.
#[cfg(windows)]
fn positioned_read(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buf, offset)
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
        assert_eq!(s.sector_size(), SectorSize::DEFAULT);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn caller_can_set_sector_size() {
        let path = temp_image("sector-set", 512);
        let s = FileSource::open(&path)
            .unwrap()
            .with_sector_size(4096)
            .unwrap();
        assert_eq!(s.sector_size().bytes(), 4096);
        assert_eq!(
            s.sector_size().basis(),
            SectorSizeBasis::Asserted,
            "a declared value is asserted, not assumed"
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
    fn describe_reports_location_and_format() {
        let path = temp_image("describe", 64);
        let s = FileSource::open(&path).unwrap();
        let d = s.describe();
        assert_eq!(d.location, Some(Location::Path(path.clone())));
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
            SectorSize::new(4096, SectorSizeBasis::Asserted).unwrap(),
            "clone keeps the asserted sector size"
        );

        std::fs::remove_file(&path).ok();
    }

    // Unix only: on Windows, `File::open` on a directory already fails, with
    // a different error kind, before the regular-file check is reached.
    #[cfg(unix)]
    #[test]
    fn a_directory_is_refused() {
        let dir = std::env::temp_dir();
        let err = FileSource::open(&dir).unwrap_err();
        match err {
            Error::Io(e) => {
                assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
                assert!(e.to_string().contains("not a regular file"), "{e}");
            }
            other => panic!("expected Error::Io, got {other:?}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_device_node_is_refused_rather_than_read_as_empty() {
        // /dev/null is a character device on every Unix. Its metadata reports
        // a length of 0, which is exactly how a block device reads as empty.
        let err = FileSource::open("/dev/null").unwrap_err();
        assert!(
            matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::InvalidInput),
            "{err:?}"
        );
    }

    #[test]
    fn one_source_serves_concurrent_readers() {
        use std::sync::Arc;
        let path = temp_image("concurrent", 64 * 1024);
        let s = Arc::new(FileSource::open(&path).unwrap());
        let handles: Vec<_> = (0..8u64)
            .map(|i| {
                let s = Arc::clone(&s);
                std::thread::spawn(move || {
                    let offset = i * 8000;
                    let mut buf = [0u8; 16];
                    read_exact_at(s.as_ref(), offset, &mut buf).unwrap();
                    (offset, buf)
                })
            })
            .collect();
        for h in handles {
            let (offset, buf) = h.join().unwrap();
            let expected: Vec<u8> = (offset..offset + 16)
                .map(|i| u8::try_from(i % 251).unwrap_or(0))
                .collect();
            assert_eq!(buf.as_slice(), expected.as_slice(), "read at {offset}");
        }
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
