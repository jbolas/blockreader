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
//! Callers wanting all-or-nothing use [`read_exact_at`].
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

mod error;
mod file;

pub use error::{Error, Result};
pub use file::FileSource;

/// The default sector size assumed when a backend cannot determine the real
/// one. See [`BlockReader::sector_size`].
pub const DEFAULT_SECTOR_SIZE: u32 = 512;

/// Identifying detail about a byte source, for reports and diagnostics.
///
/// Everything is optional: a backend reports what it knows. Consumers put
/// this in their output so an examiner can see what was read and which
/// values were assumed rather than observed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceDescription {
    /// Where the data came from, when it has a filesystem path.
    pub path: Option<std::path::PathBuf>,
    /// The container format, e.g. `"aff4"`, `"raw"`, `"e01"`.
    pub format: Option<String>,
    /// True when [`BlockReader::sector_size`] is a default rather than a
    /// value the backend actually determined.
    ///
    /// Consumers should surface this: partition-table offsets depend on
    /// sector size, so an assumed value is a caveat on everything derived
    /// from it.
    pub sector_size_assumed: bool,
}

/// A read-only random-access source of bytes.
///
/// # Implementing
///
/// The required methods are [`read_at`](Self::read_at) and
/// [`size`](Self::size). Override [`sector_size`](Self::sector_size) when the
/// backend knows the real value, and [`describe`](Self::describe) to identify
/// the source in reports.
///
/// `read_at` takes `&self`, so an implementation that caches or holds a file
/// cursor needs interior mutability — a `Mutex` around the mutable part. That
/// cost is nanoseconds against the I/O it guards.
///
/// Implementations may cache. Callers should treat `read_at` as potentially
/// cheap when reads have locality, but never free.
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

    /// The sector size in bytes, for callers that need LBA addressing.
    ///
    /// Defaults to [`DEFAULT_SECTOR_SIZE`]. A GPT header lives at LBA 1, so
    /// its byte offset depends on this value and 512 versus 4096 is not
    /// always determinable from the data. Backends that know the real value
    /// override this; those that do not should also set
    /// [`SourceDescription::sector_size_assumed`] so consumers can say so.
    fn sector_size(&self) -> u32 {
        DEFAULT_SECTOR_SIZE
    }

    /// Identifying detail for reports and diagnostics.
    fn describe(&self) -> SourceDescription {
        SourceDescription::default()
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
        let n = self
            .inner
            .read_at(self.pos, buf)
            .map_err(std::io::Error::other)?;
        self.pos = self.pos.saturating_add(n as u64);
        Ok(n)
    }
}

impl<R: BlockReader> std::io::Seek for IoAdapter<R> {
    fn seek(&mut self, from: std::io::SeekFrom) -> std::io::Result<u64> {
        use std::io::SeekFrom;
        let new = match from {
            SeekFrom::Start(n) => n,
            SeekFrom::End(d) => self.inner.size().saturating_add_signed(d),
            SeekFrom::Current(d) => self.pos.saturating_add_signed(d),
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
        let s = source(10, 4);
        assert_eq!(s.sector_size(), DEFAULT_SECTOR_SIZE);
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
    fn io_adapter_refuses_writes() {
        use std::io::Write;
        let mut a = IoAdapter::new(source(10, 10));
        let e = a.write(b"nope").unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied);
    }
}
