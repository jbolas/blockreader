//! A minimal, zero-dependency, read-only random-access byte source.
//!
//! This crate is the seam between forensic container formats (AFF4, E01, raw
//! dd, dmg, VMDK) and the parsers that read structure out of them (partition
//! tables, filesystems). Container formats *implement* [`BlockReader`];
//! filesystem parsers *consume* it. Neither side depends on the other.
//!
//! # Read-only
//!
//! There is no write method. Forensic evidence is not modified.
//!
//! # Reads take `&self`
//!
//! Reading does not change a byte source. An
//! implementation that caches uses interior mutability; the lock cost is
//! negligible beside the I/O and decompression it guards.
//!
//! This is what lets one `Arc<dyn BlockReader>` be shared by several workers
//! without cloning a handle per worker. Note that sharing is not the same as
//! parallelism: workers sharing one reader serialize on whatever lock it holds
//! internally. Genuine parallel throughput still wants a handle each, which
//! backends over seekable files can provide cheaply.
//!
//! # Short reads are honest
//!
//! [`BlockReader::read_at`] returns the number of bytes actually read and
//! never fabricates data. Implementations must not zero-fill, pad, or
//! substitute placeholder bytes for a region they could not read, because a
//! caller cannot then distinguish an unstored region from genuine zeros.
//! Callers wanting all-or-nothing use [`read_exact_at`]. A range whose content
//! is unknown is an error, [`Error::UnknownContent`], never filler; see
//! [`BlockReader`].
//!
//! # Sector size is reported with its basis
//!
//! [`BlockReader::sector_size`] returns a [`SectorSize`]: the size in bytes
//! together with how it came to be known, whether assumed, recorded by the
//! container, or asserted by a caller. The two cannot disagree, and a backend
//! that says nothing reports an assumed 512.
//!
//! # Composition
//!
//! Implementations nest. A volume inside a container is itself a
//! `BlockReader` over its own address space, as is a decrypting wrapper over
//! an encrypted volume. Nesting costs a virtual call, not a copy: `read_at`
//! writes into the caller's buffer at every level.

#![deny(unsafe_code)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unused_must_use
)]
#![warn(missing_docs, clippy::pedantic)]

mod description;
mod error;
mod file;
mod sector;

pub use description::{Location, SourceDescription};
pub use error::{Error, Result, UnknownCause};
pub use file::FileSource;
pub use sector::{DEFAULT_SECTOR_SIZE, SectorSize, SectorSizeBasis};

/// A read-only random-access source of bytes.
///
/// # Implementing
///
/// The required methods are [`read_at`](Self::read_at) and
/// [`size`](Self::size). Override [`sector_size`](Self::sector_size) when the
/// container records the real value, and [`describe`](Self::describe) to
/// identify the source in reports. The defaults claim nothing: an assumed
/// 512-byte sector and an empty description.
///
/// `read_at` takes `&self`, so an implementation that caches or holds a file
/// cursor needs interior mutability — a `Mutex` around the mutable part. That
/// cost is nanoseconds against the I/O it guards.
///
/// Implementations may cache. Callers should treat `read_at` as potentially
/// cheap when reads have locality, but never free.
///
/// # Unknown content
///
/// Some ranges of an image have no known content: the acquisition never read
/// them, the source medium could not be read there, or the container's stored
/// bytes fail an integrity check. A backend never returns placeholder bytes for
/// such a range. Instead:
///
/// 1. [`Error::UnknownContent`] carries the whole unknown range containing the
///    requested offset, as far as the backend knows it, and the
///    [`UnknownCause`].
/// 2. A read that starts before an unknown range returns the good bytes up to
///    where the range begins: an honest short read.
/// 3. A read that starts inside an unknown range returns
///    [`Error::UnknownContent`].
/// 4. [`read_exact_at`] therefore surfaces [`Error::UnknownContent`], not
///    [`Error::UnexpectedEof`]. A caller that wants to continue resumes at
///    `offset + len`.
/// 5. Content the format defines is not unknown. Unallocated VMDK grains, AFF4
///    `aff4:Zero` regions, and map gaps filled by rule are returned as data.
pub trait BlockReader: Send + Sync {
    /// Read into `buf`, starting at byte `offset`.
    ///
    /// Returns the number of bytes read, which may be fewer than
    /// `buf.len()`. A return of `0` means `offset` is at or past the end of
    /// the source.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the underlying source fails, or
    /// [`Error::Backend`] for a format-specific failure.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize>;

    /// The total addressable size, in bytes.
    fn size(&self) -> u64;

    /// The sector size, and how it came to be known.
    ///
    /// Defaults to [`SectorSize::DEFAULT`]: 512 bytes, assumed. A GPT header
    /// lives at LBA 1, so its byte offset depends on this value, and 512
    /// versus 4096 is not always determinable from the data. A backend that
    /// knows the real value, because its container records it, overrides
    /// this and reports [`SectorSizeBasis::Recorded`]. A backend that knows
    /// nothing better need not override it: the default already says the
    /// value is assumed.
    fn sector_size(&self) -> SectorSize {
        SectorSize::DEFAULT
    }

    /// Identifying detail for reports and diagnostics.
    fn describe(&self) -> SourceDescription {
        SourceDescription::default()
    }
}

impl<T: BlockReader + ?Sized> BlockReader for &T {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        (**self).read_at(offset, buf)
    }

    fn size(&self) -> u64 {
        (**self).size()
    }

    fn sector_size(&self) -> SectorSize {
        (**self).sector_size()
    }

    fn describe(&self) -> SourceDescription {
        (**self).describe()
    }
}

impl<T: BlockReader + ?Sized> BlockReader for Box<T> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        (**self).read_at(offset, buf)
    }

    fn size(&self) -> u64 {
        (**self).size()
    }

    fn sector_size(&self) -> SectorSize {
        (**self).sector_size()
    }

    fn describe(&self) -> SourceDescription {
        (**self).describe()
    }
}

impl<T: BlockReader + ?Sized> BlockReader for std::sync::Arc<T> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        (**self).read_at(offset, buf)
    }

    fn size(&self) -> u64 {
        (**self).size()
    }

    fn sector_size(&self) -> SectorSize {
        (**self).sector_size()
    }

    fn describe(&self) -> SourceDescription {
        (**self).describe()
    }
}

/// Fill `buf` entirely, or fail.
///
/// Repeats [`BlockReader::read_at`] until `buf` is full, since a backend is
/// permitted to return short reads for reasons that are not errors (a chunk
/// boundary, an internal buffer limit).
///
/// This is what most parsing code wants: a truncated structure is an error,
/// not a value to interpret.
///
/// # Errors
///
/// Returns [`Error::UnexpectedEof`] if the source ends before `buf` is full,
/// or whatever error the underlying reader produced.
pub fn read_exact_at<R: BlockReader + ?Sized>(
    reader: &R,
    offset: u64,
    buf: &mut [u8],
) -> Result<()> {
    let wanted = buf.len();
    let mut filled = 0usize;

    while filled < wanted {
        let at = offset.saturating_add(filled as u64);
        let n = reader.read_at(at, &mut buf[filled..])?;
        if n == 0 {
            return Err(Error::UnexpectedEof {
                offset,
                wanted,
                got: filled,
            });
        }
        filled += n;
    }

    Ok(())
}

/// Adapts a [`BlockReader`] to `std::io::Read` + `Seek`.
///
/// Some libraries take an `io::Read + Seek` device rather than a positioned
/// reader. This carries its own cursor, so several may be made over one
/// shared source without interfering.  
///
/// `Write` is implemented, but every call fails with
/// [`std::io::ErrorKind::PermissionDenied`]. It exists only because some
/// device traits require the bound; nothing is ever written.
pub struct IoAdapter<R: BlockReader> {
    inner: R,
    pos: u64,
}

impl<R: BlockReader> IoAdapter<R> {
    /// Wrap `inner`, positioned at zero.
    pub fn new(inner: R) -> Self {
        Self { inner, pos: 0 }
    }

    /// The wrapped reader.
    pub fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Unwrap, discarding the cursor.
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: BlockReader> std::fmt::Debug for IoAdapter<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IoAdapter")
            .field("pos", &self.pos)
            .field("size", &self.inner.size())
            .finish()
    }
}

impl<R: BlockReader> std::io::Read for IoAdapter<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read_at(self.pos, buf).map_err(into_io_error)?;
        self.pos = self.pos.saturating_add(n as u64);
        Ok(n)
    }
}

/// Convert a byte-source error for callers that speak `std::io`.
///
/// An I/O error passes through unchanged, keeping its kind. Unknown content
/// becomes `InvalidData`, wrapping the original, which a caller recovers with
/// `io::Error::get_ref` and `downcast_ref::<blockreader::Error>()`.
fn into_io_error(e: Error) -> std::io::Error {
    match e {
        Error::Io(io) => io,
        unknown @ Error::UnknownContent { .. } => {
            std::io::Error::new(std::io::ErrorKind::InvalidData, unknown)
        }
        other => std::io::Error::other(other),
    }
}

impl<R: BlockReader> std::io::Seek for IoAdapter<R> {
    fn seek(&mut self, from: std::io::SeekFrom) -> std::io::Result<u64> {
        use std::io::SeekFrom;
        let new = match from {
            SeekFrom::Start(n) => Some(n),
            SeekFrom::End(d) => self.inner.size().checked_add_signed(d),
            SeekFrom::Current(d) => self.pos.checked_add_signed(d),
        };
        let Some(new) = new else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid seek to a negative or overflowing position",
            ));
        };
        self.pos = new;
        Ok(new)
    }
}

impl<R: BlockReader> std::io::Write for IoAdapter<R> {
    fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
        Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "this source is read-only",
        ))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// A reader over an in-memory slice that returns at most `chunk` bytes
    /// per call, so short-read handling can be exercised.
    struct ChunkedSource {
        data: Vec<u8>,
        chunk: usize,
    }

    impl BlockReader for ChunkedSource {
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
            let Ok(start) = usize::try_from(offset) else {
                return Ok(0);
            };
            if start >= self.data.len() {
                return Ok(0);
            }
            let available = self.data.len() - start;
            let n = buf.len().min(self.chunk).min(available);
            buf[..n].copy_from_slice(&self.data[start..start + n]);
            Ok(n)
        }

        fn size(&self) -> u64 {
            self.data.len() as u64
        }
    }

    fn source(len: usize, chunk: usize) -> ChunkedSource {
        ChunkedSource {
            data: (0..len)
                .map(|i| u8::try_from(i % 251).unwrap_or(0))
                .collect(),
            chunk,
        }
    }

    /// A reader over an in-memory slice with one range whose content is
    /// unknown, following the rules in the trait documentation.
    struct UnknownRangeSource {
        data: Vec<u8>,
        unknown: std::ops::Range<u64>,
        cause: UnknownCause,
    }

    impl BlockReader for UnknownRangeSource {
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
            let size = self.data.len() as u64;
            if offset >= size || buf.is_empty() {
                return Ok(0);
            }
            if self.unknown.contains(&offset) {
                return Err(Error::UnknownContent {
                    offset: self.unknown.start,
                    len: self.unknown.end - self.unknown.start,
                    cause: self.cause,
                });
            }
            let mut end = offset.saturating_add(buf.len() as u64).min(size);
            if offset < self.unknown.start && end > self.unknown.start {
                end = self.unknown.start;
            }
            let start = usize::try_from(offset).unwrap();
            let n = usize::try_from(end - offset).unwrap();
            buf[..n].copy_from_slice(&self.data[start..start + n]);
            Ok(n)
        }

        fn size(&self) -> u64 {
            self.data.len() as u64
        }
    }

    fn with_unknown(cause: UnknownCause) -> UnknownRangeSource {
        UnknownRangeSource {
            data: (0..256u32)
                .map(|i| u8::try_from(i % 251).unwrap_or(0))
                .collect(),
            unknown: 100..150,
            cause,
        }
    }

    #[test]
    fn a_read_before_an_unknown_range_stops_at_its_start() {
        let s = with_unknown(UnknownCause::FailedIntegrity);
        let mut buf = [0xAAu8; 64];
        let n = s.read_at(80, &mut buf).unwrap();
        assert_eq!(n, 20, "a short read ending where the unknown range begins");
        assert_eq!(buf[20], 0xAA, "nothing written past the good bytes");
    }

    #[test]
    fn a_read_inside_an_unknown_range_reports_the_whole_range_and_cause() {
        let s = with_unknown(UnknownCause::UnreadableAtAcquisition);
        let mut buf = [0u8; 8];
        let err = s.read_at(120, &mut buf).unwrap_err();
        match err {
            Error::UnknownContent { offset, len, cause } => {
                assert_eq!((offset, len), (100, 50));
                assert_eq!(cause, UnknownCause::UnreadableAtAcquisition);
            }
            other => panic!("expected UnknownContent, got {other:?}"),
        }
    }

    #[test]
    fn read_exact_at_surfaces_unknown_content_not_eof() {
        let s = with_unknown(UnknownCause::NotAcquired);
        let mut buf = [0u8; 64];
        let err = read_exact_at(&s, 80, &mut buf).unwrap_err();
        assert!(
            matches!(
                err,
                Error::UnknownContent {
                    offset: 100,
                    len: 50,
                    cause: UnknownCause::NotAcquired
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn io_adapter_wraps_unknown_content_as_invalid_data() {
        use std::io::{Read, Seek, SeekFrom};
        let mut a = IoAdapter::new(with_unknown(UnknownCause::FailedIntegrity));
        a.seek(SeekFrom::Start(110)).unwrap();
        let mut buf = [0u8; 4];
        let e = a.read(&mut buf).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
        let inner = e
            .get_ref()
            .and_then(|inner| inner.downcast_ref::<Error>())
            .expect("the original error stays reachable");
        assert!(matches!(
            inner,
            Error::UnknownContent {
                cause: UnknownCause::FailedIntegrity,
                ..
            }
        ));
    }

    /// A reader whose every read fails with the given I/O error kind.
    struct FailingSource(std::io::ErrorKind);

    impl BlockReader for FailingSource {
        fn read_at(&self, _offset: u64, _buf: &mut [u8]) -> Result<usize> {
            Err(Error::Io(std::io::Error::new(self.0, "simulated")))
        }

        fn size(&self) -> u64 {
            1024
        }
    }

    #[test]
    fn io_adapter_keeps_the_io_error_kind() {
        use std::io::Read;
        let mut a = IoAdapter::new(FailingSource(std::io::ErrorKind::PermissionDenied));
        let mut buf = [0u8; 4];
        let e = a.read(&mut buf).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn read_at_returns_short_count_without_padding() {
        let s = source(100, 7);
        let mut buf = [0xAAu8; 32];
        let n = s.read_at(0, &mut buf).unwrap();
        assert_eq!(n, 7, "should return the backend's short count");
        assert_eq!(buf[7], 0xAA, "read_at must not pad beyond what it read");
    }

    #[test]
    fn read_at_past_end_returns_zero() {
        let s = source(10, 64);
        let mut buf = [0u8; 8];
        assert_eq!(s.read_at(10, &mut buf).unwrap(), 0);
        assert_eq!(s.read_at(9999, &mut buf).unwrap(), 0);
    }

    #[test]
    fn read_exact_at_loops_over_short_reads() {
        let s = source(100, 7);
        let mut buf = [0u8; 32];
        read_exact_at(&s, 0, &mut buf).unwrap();
        let expected: Vec<u8> = (0..32u32)
            .map(|i| u8::try_from(i % 251).unwrap_or(0))
            .collect();
        assert_eq!(buf.as_slice(), expected.as_slice());
    }

    #[test]
    fn read_exact_at_honors_offset() {
        let s = source(100, 3);
        let mut buf = [0u8; 10];
        read_exact_at(&s, 50, &mut buf).unwrap();
        let expected: Vec<u8> = (50..60u32)
            .map(|i| u8::try_from(i % 251).unwrap_or(0))
            .collect();
        assert_eq!(buf.as_slice(), expected.as_slice());
    }

    #[test]
    fn read_exact_at_errors_on_truncation() {
        let s = source(10, 4);
        let mut buf = [0u8; 32];
        let err = read_exact_at(&s, 0, &mut buf).unwrap_err();
        match err {
            Error::UnexpectedEof {
                offset,
                wanted,
                got,
            } => {
                assert_eq!(offset, 0);
                assert_eq!(wanted, 32);
                assert_eq!(got, 10, "reports how much it did get");
            }
            other => panic!("expected UnexpectedEof, got {other:?}"),
        }
    }

    #[test]
    fn read_exact_at_empty_buffer_is_ok() {
        let s = source(10, 4);
        read_exact_at(&s, 0, &mut []).unwrap();
        read_exact_at(&s, 9999, &mut []).unwrap();
    }

    #[test]
    fn defaults_are_conservative() {
        // A backend implementing only the two required methods must not
        // appear to know its sector size.
        let s = source(10, 4);
        assert_eq!(s.sector_size(), SectorSize::DEFAULT);
        assert_eq!(s.sector_size().basis(), SectorSizeBasis::Assumed);
        assert_eq!(s.describe(), SourceDescription::default());
    }

    #[test]
    fn works_through_dyn_dispatch() {
        let s = source(100, 7);
        let r: &dyn BlockReader = &s;
        let mut buf = [0u8; 16];
        read_exact_at(r, 0, &mut buf).unwrap();
        assert_eq!(buf[15], 15);
    }

    #[test]
    fn one_reader_serves_several_threads_without_cloning() {
        // The point of &self reads: share, don't clone.
        use std::sync::Arc;
        let s: Arc<dyn BlockReader> = Arc::new(source(4096, 4096));
        let handles: Vec<_> = (0..4u64)
            .map(|i| {
                let s = Arc::clone(&s);
                std::thread::spawn(move || {
                    let mut buf = [0u8; 8];
                    read_exact_at(s.as_ref(), i * 64, &mut buf).unwrap();
                    buf[0]
                })
            })
            .collect();
        for (i, h) in handles.into_iter().enumerate() {
            let got = h.join().unwrap();
            let expect = u8::try_from((i as u64 * 64) % 251).unwrap_or(0);
            assert_eq!(got, expect, "thread {i}");
        }
    }

    #[test]
    fn io_adapter_reads_and_seeks() {
        use std::io::{Read, Seek, SeekFrom};
        let mut a = IoAdapter::new(source(100, 64));
        let mut buf = [0u8; 4];
        a.read_exact(&mut buf).unwrap();
        assert_eq!(buf, [0, 1, 2, 3]);
        a.seek(SeekFrom::Start(50)).unwrap();
        a.read_exact(&mut buf).unwrap();
        assert_eq!(buf, [50, 51, 52, 53]);
        assert_eq!(a.seek(SeekFrom::End(0)).unwrap(), 100);
    }

    #[test]
    fn io_adapter_rejects_a_seek_before_the_start() {
        use std::io::{Seek, SeekFrom};
        let mut a = IoAdapter::new(source(100, 64));
        a.seek(SeekFrom::Start(5)).unwrap();
        let e = a.seek(SeekFrom::Current(-10)).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(
            a.stream_position().unwrap(),
            5,
            "a failed seek leaves the position alone"
        );
        let e = a.seek(SeekFrom::End(-101)).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn io_adapter_rejects_an_overflowing_seek() {
        use std::io::{Seek, SeekFrom};
        let mut a = IoAdapter::new(source(100, 64));
        a.seek(SeekFrom::Start(u64::MAX)).unwrap();
        let e = a.seek(SeekFrom::Current(1)).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn io_adapter_allows_a_seek_past_the_end_and_reads_nothing_there() {
        use std::io::{Read, Seek, SeekFrom};
        let mut a = IoAdapter::new(source(100, 64));
        assert_eq!(a.seek(SeekFrom::Start(500)).unwrap(), 500);
        let mut buf = [0u8; 4];
        assert_eq!(a.read(&mut buf).unwrap(), 0);
    }

    /// A reader with a recorded sector size and a description, so the tests
    /// can see whether a wrapper forwards the non-default methods.
    struct Described {
        inner: ChunkedSource,
    }

    impl BlockReader for Described {
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
            self.inner.read_at(offset, buf)
        }

        fn size(&self) -> u64 {
            self.inner.size()
        }

        fn sector_size(&self) -> SectorSize {
            SectorSize::new(4096, SectorSizeBasis::Recorded).unwrap()
        }

        fn describe(&self) -> SourceDescription {
            SourceDescription::new(None, Some("test".to_string()))
        }
    }

    fn described() -> Described {
        Described {
            inner: source(100, 64),
        }
    }

    /// Everything a wrapper must forward, checked through the trait alone.
    fn assert_forwards<R: BlockReader>(r: &R) {
        let mut buf = [0u8; 4];
        read_exact_at(r, 50, &mut buf).unwrap();
        assert_eq!(buf, [50, 51, 52, 53]);
        assert_eq!(r.size(), 100);
        assert_eq!(r.sector_size().bytes(), 4096);
        assert_eq!(r.sector_size().basis(), SectorSizeBasis::Recorded);
        assert_eq!(r.describe().format.as_deref(), Some("test"));
    }

    #[test]
    fn a_reference_forwards_everything() {
        let d = described();
        assert_forwards(&&d);
    }

    #[test]
    fn a_box_forwards_everything() {
        let b: Box<dyn BlockReader> = Box::new(described());
        assert_forwards(&b);
    }

    #[test]
    fn an_arc_forwards_everything() {
        let a: std::sync::Arc<dyn BlockReader> = std::sync::Arc::new(described());
        assert_forwards(&a);
    }

    #[test]
    fn io_adapter_wraps_a_shared_reader() {
        use std::io::{Read, Seek, SeekFrom};
        let shared: std::sync::Arc<dyn BlockReader> = std::sync::Arc::new(described());
        let mut a = IoAdapter::new(std::sync::Arc::clone(&shared));
        let mut b = IoAdapter::new(shared);
        a.seek(SeekFrom::Start(10)).unwrap();
        let mut ba = [0u8; 2];
        let mut bb = [0u8; 2];
        a.read_exact(&mut ba).unwrap();
        b.read_exact(&mut bb).unwrap();
        assert_eq!(ba, [10, 11]);
        assert_eq!(bb, [0, 1], "each adapter keeps its own cursor");
    }

    #[test]
    fn io_adapter_refuses_writes() {
        use std::io::Write;
        let mut a = IoAdapter::new(source(10, 10));
        let e = a.write(b"nope").unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied);
    }
}
