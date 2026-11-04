//! Append-only storage one thread writes and others read.
//!
//! The walk materializes derivations one at a time and never changes one
//! once it is made, so a reader on another thread can follow any derivation
//! the walk has already handed it, while the walk goes on appending. An
//! [`Arena`] holds such records in chunks that never move: chunk `c` holds
//! `FIRST << c` of them, so `CHUNKS` chunks hold more than `u32::MAX`. One
//! [`Writer`] appends; any number of [`Reader`]s read an index once the
//! writer has published it, by a release store of the length that a reader
//! acquires before it reads past what it last saw.

use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;

const FIRST: usize = 1 << 10;
const CHUNKS: usize = 23;

struct Inner<T> {
    chunks: [AtomicPtr<T>; CHUNKS],
    len: AtomicUsize,
}

// SAFETY: elements are `Copy` plain data written once by the single writer
// before the release store that publishes them, and read only after an
// acquire load that sees that store.
unsafe impl<T: Copy + Send> Send for Inner<T> {}
unsafe impl<T: Copy + Send + Sync> Sync for Inner<T> {}

/// The chunk index `i` lives in and its offset there.
fn locate(i: usize) -> (usize, usize) {
    let c = (usize::BITS - 1 - (i / FIRST + 1).leading_zeros()) as usize;
    (c, i - FIRST * ((1 << c) - 1))
}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        for (c, chunk) in self.chunks.iter_mut().enumerate() {
            let p = *chunk.get_mut();
            if !p.is_null() {
                // SAFETY: allocated in `Writer::push` as a boxed slice of
                // exactly `FIRST << c` possibly uninitialized elements;
                // `T: Copy` needs no drop.
                unsafe {
                    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                        p.cast::<MaybeUninit<T>>(),
                        FIRST << c,
                    )));
                }
            }
        }
    }
}

/// An empty arena: its one writer and a reader.
pub(super) fn arena<T: Copy + Default>() -> (Writer<T>, Reader<T>) {
    let inner = Arc::new(Inner {
        chunks: std::array::from_fn(|_| AtomicPtr::new(std::ptr::null_mut())),
        len: AtomicUsize::new(0),
    });
    (
        Writer {
            inner: inner.clone(),
            len: 0,
        },
        Reader { inner, seen: 0 },
    )
}

/// The one thread that appends.
pub(super) struct Writer<T> {
    inner: Arc<Inner<T>>,
    len: usize,
}

impl<T: Copy + Default> Writer<T> {
    /// Append `x` and publish it; its index.
    pub(super) fn push(&mut self, x: T) -> u32 {
        let i = self.len;
        let (c, off) = locate(i);
        let mut p = self.inner.chunks[c].load(Ordering::Relaxed);
        if p.is_null() {
            // Uninitialized, so pages commit as records are written rather
            // than all at once; nothing reads a slot before its write.
            let chunk: Box<[MaybeUninit<T>]> = Box::new_uninit_slice(FIRST << c);
            p = Box::into_raw(chunk).cast::<T>();
            self.inner.chunks[c].store(p, Ordering::Release);
        }
        // SAFETY: `off < FIRST << c`, and no reader reads index `i` before
        // the release store below publishes it.
        unsafe { p.add(off).write(x) };
        self.len = i + 1;
        self.inner.len.store(self.len, Ordering::Release);
        u32::try_from(i).expect("fewer than 2^32 records")
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Bytes of the chunks allocated so far.
    pub(super) fn bytes(&self) -> usize {
        if self.len == 0 {
            return 0;
        }
        let (c, _) = locate(self.len - 1);
        FIRST * ((1 << (c + 1)) - 1) * std::mem::size_of::<T>()
    }

    /// Another reader over the same records.
    pub(super) fn reader(&self) -> Reader<T> {
        Reader {
            inner: self.inner.clone(),
            seen: 0,
        }
    }
}

/// A thread that reads what the writer has published.
pub(super) struct Reader<T> {
    inner: Arc<Inner<T>>,
    /// A length this reader has acquired: indices below it are safe to read
    /// without touching the shared length again.
    seen: usize,
}

impl<T: Copy> Reader<T> {
    /// The record at `i`. Panics unless the writer has published it.
    pub(super) fn get(&mut self, i: u32) -> T {
        let i = i as usize;
        if i >= self.seen {
            self.seen = self.inner.len.load(Ordering::Acquire);
            assert!(i < self.seen, "a record not yet published");
        }
        let (c, off) = locate(i);
        // SAFETY: `i` is below a length acquired from the writer's release
        // store, which follows the chunk's store and the slot's write.
        unsafe { *self.inner.chunks[c].load(Ordering::Acquire).add(off) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_map_to_chunks_without_gaps() {
        let mut next = 0;
        for c in 0..4 {
            for off in 0..(FIRST << c) {
                assert_eq!(locate(next), (c, off));
                next += 1;
            }
        }
        assert!(FIRST * ((1 << CHUNKS) - 1) > u32::MAX as usize);
    }

    #[test]
    fn a_reader_on_another_thread_sees_every_published_record() {
        let (mut w, r) = arena::<u64>();
        let (to, from) = std::sync::mpsc::channel::<u32>();
        let reader = std::thread::spawn(move || {
            let mut r = r;
            let mut n = 0u64;
            for i in from {
                assert_eq!(r.get(i), u64::from(i) * 3 + 1);
                n += 1;
            }
            n
        });
        let total = 3 * FIRST as u32 + 17;
        for i in 0..total {
            assert_eq!(w.push(u64::from(i) * 3 + 1), i);
            if i % 5 == 0 {
                to.send(i).unwrap();
            }
        }
        drop(to);
        assert_eq!(reader.join().unwrap(), u64::from(total.div_ceil(5)));
        assert_eq!(w.len(), total as usize);
    }
}
