//! An [`AwarePass`] kept on disk, every value bit for bit.
//!
//! The file opens with its format and the caller's key, then the weight, and
//! ends with an FNV-1a checksum of everything before it. A reader checks all
//! three and the length of every section against what is left of the file,
//! so a truncated, corrupt or foreign file is an error rather than a pass.

use super::{nuc_from_index, AwarePass, NodeIx, PKey};
use crate::constrained::ConstrainedFold;
use crate::dfa::{Node, NodeLayout};
use crate::hash::FxHashMap;
use crate::state::{Opening, PairKey};
use rna_mfe::Energy;
use std::io::Write;

/// The format, read first from every file.
const AWARE_FORMAT: &[u8] = b"derna-aware-pass-2";

/// Bytes of one `outside` entry: the 20-byte key and its value.
const OUTSIDE_ENTRY: usize = 28;
/// Bytes of one `through` entry: two nodes, the opening nucleotide, the
/// lane the opening lands in, the closing nucleotide and the value.
const THROUGH_ENTRY: usize = 21;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(FNV_PRIME);
    }
    h
}

fn malformed() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, "a malformed aware pass")
}

/// A writer that checksums what it forwards.
struct Summed<W: Write> {
    inner: W,
    sum: u64,
}

impl<W: Write> Summed<W> {
    fn put(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.sum = fnv(self.sum, bytes);
        self.inner.write_all(bytes)
    }

    fn u64(&mut self, v: u64) -> std::io::Result<()> {
        self.put(&v.to_le_bytes())
    }

    fn f64(&mut self, v: f64) -> std::io::Result<()> {
        self.u64(v.to_bits())
    }

    fn text(&mut self, s: &str) -> std::io::Result<()> {
        self.u64(s.len() as u64)?;
        self.put(s.as_bytes())
    }
}

fn node_bytes(node: Node) -> [u8; 5] {
    let mut out = [0u8; 5];
    out[..4].copy_from_slice(&{ node.pos }.to_le_bytes());
    out[4] = node.lane;
    out
}

impl AwarePass {
    /// Keep the pass in `file` under `key`, written whole to a temporary
    /// file and renamed so a reader never sees half of one.
    pub(crate) fn write(&self, file: &std::path::Path, key: u64) -> std::io::Result<()> {
        let tmp = file.with_extension(format!("tmp{}", std::process::id()));
        let written = self.write_to(&tmp, key);
        if written.is_err() {
            let _ = std::fs::remove_file(&tmp);
            return written;
        }
        std::fs::rename(&tmp, file)
    }

    fn write_to(&self, tmp: &std::path::Path, key: u64) -> std::io::Result<()> {
        let mut w = Summed {
            inner: std::io::BufWriter::new(std::fs::File::create(tmp)?),
            sum: FNV_OFFSET,
        };
        w.put(AWARE_FORMAT)?;
        w.u64(key)?;
        w.f64(self.lambda)?;
        w.f64(self.optimum)?;
        w.f64(self.ceiling)?;
        w.u64(self.folds as u64)?;
        match &self.design {
            None => w.put(&[0])?,
            Some(d) => {
                w.put(&[1])?;
                w.put(&d.mfe.0.to_le_bytes())?;
                w.f64(d.cai)?;
                w.text(&d.sequence)?;
                w.text(&d.structure)?;
            }
        }
        w.u64(self.outside.len() as u64)?;
        for column in &self.outside {
            w.u64(column.len() as u64)?;
            for (k, &v) in column {
                w.put(&k.open.0.to_le_bytes())?;
                w.put(&k.close.0.to_le_bytes())?;
                w.put(&[k.nuc_open, k.nuc_close, k.lane_open, k.lane_close])?;
                w.put(&k.q_open.to_le_bytes())?;
                w.put(&k.q_outer.to_le_bytes())?;
                w.f64(v)?;
            }
        }
        w.u64(self.through.len() as u64)?;
        for column in &self.through {
            w.u64(column.len() as u64)?;
            for (k, &v) in column {
                w.put(&node_bytes(k.open))?;
                w.put(&node_bytes(k.close))?;
                w.put(&[k.opening.nuc() as u8, k.opening.lane(), k.nuc_close as u8])?;
                w.f64(v)?;
            }
        }
        let sum = w.sum;
        let mut inner = w.inner;
        inner.write_all(&sum.to_le_bytes())?;
        inner.flush()
    }

    /// The pass at `lambda` kept in `file` under `key` by [`Self::write`].
    /// `secs` is how long reading it took.
    pub(crate) fn read(
        file: &std::path::Path,
        layout: NodeLayout,
        key: u64,
        lambda: f64,
    ) -> std::io::Result<Self> {
        let started = std::time::Instant::now();
        let bytes = std::fs::read(file)?;
        let split = bytes.len().checked_sub(8).ok_or_else(malformed)?;
        let (body, sum) = bytes.split_at(split);
        if fnv(FNV_OFFSET, body).to_le_bytes() != sum {
            return Err(malformed());
        }
        let mut r = Reader { bytes: body, at: 0 };
        if r.take(AWARE_FORMAT.len())? != AWARE_FORMAT
            || r.u64()? != key
            || r.u64()? != lambda.to_bits()
        {
            return Err(malformed());
        }
        let optimum = r.f64()?;
        let ceiling = r.f64()?;
        let folds = usize::try_from(r.u64()?).map_err(|_| malformed())?;
        let design = r.design()?;
        let outside = r.columns(OUTSIDE_ENTRY, Reader::outside_entry)?;
        let through = r.columns(THROUGH_ENTRY, Reader::through_entry)?;
        if r.at != body.len() {
            return Err(malformed());
        }
        Ok(Self {
            layout,
            lambda,
            optimum,
            ceiling,
            folds,
            secs: started.elapsed().as_secs_f64(),
            design,
            outside,
            through,
        })
    }
}

/// A kept pass's bytes, read from the front.
struct Reader<'b> {
    bytes: &'b [u8],
    at: usize,
}

impl<'b> Reader<'b> {
    fn take(&mut self, n: usize) -> std::io::Result<&'b [u8]> {
        let end = self.at.checked_add(n).ok_or_else(malformed)?;
        let out = self.bytes.get(self.at..end).ok_or_else(malformed)?;
        self.at = end;
        Ok(out)
    }

    fn array<const N: usize>(&mut self) -> std::io::Result<[u8; N]> {
        self.take(N)?.try_into().map_err(|_| malformed())
    }

    fn u64(&mut self) -> std::io::Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> std::io::Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn f64(&mut self) -> std::io::Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    /// A count of items of at least `each` bytes, no more than what is left
    /// of the file could hold.
    fn count(&mut self, each: usize) -> std::io::Result<usize> {
        let n = usize::try_from(self.u64()?).map_err(|_| malformed())?;
        if n > (self.bytes.len() - self.at) / each.max(1) {
            return Err(malformed());
        }
        Ok(n)
    }

    fn text(&mut self) -> std::io::Result<String> {
        let n = self.count(1)?;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| malformed())
    }

    fn design(&mut self) -> std::io::Result<Option<ConstrainedFold>> {
        match self.array::<1>()? {
            [0] => Ok(None),
            [1] => Ok(Some(ConstrainedFold {
                mfe: Energy(i32::from_le_bytes(self.array()?)),
                cai: self.f64()?,
                sequence: self.text()?,
                structure: self.text()?,
            })),
            _ => Err(malformed()),
        }
    }

    fn columns<K: std::hash::Hash + Eq>(
        &mut self,
        each: usize,
        entry: fn(&mut Self) -> std::io::Result<(K, f64)>,
    ) -> std::io::Result<Vec<FxHashMap<K, f64>>> {
        let n = self.count(8)?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let len = self.count(each)?;
            let mut column = FxHashMap::default();
            column.reserve(len);
            for _ in 0..len {
                let (k, v) = entry(self)?;
                if column.insert(k, v).is_some() {
                    return Err(malformed());
                }
            }
            out.push(column);
        }
        Ok(out)
    }

    fn outside_entry(&mut self) -> std::io::Result<(PKey, f64)> {
        let open = NodeIx(self.u32()?);
        let close = NodeIx(self.u32()?);
        let [nuc_open, nuc_close, lane_open, lane_close] = self.array()?;
        if nuc_open > 3 || nuc_close > 3 || lane_open > 1 || lane_close > 1 {
            return Err(malformed());
        }
        let key = PKey {
            open,
            close,
            nuc_open,
            nuc_close,
            lane_open,
            lane_close,
            q_open: self.u32()?,
            q_outer: self.u32()?,
        };
        Ok((key, self.f64()?))
    }

    fn node(&mut self) -> std::io::Result<Node> {
        let pos = self.u32()?;
        let [lane] = self.array()?;
        if lane > 1 {
            return Err(malformed());
        }
        Ok(Node::new(pos, lane))
    }

    fn through_entry(&mut self) -> std::io::Result<(PairKey, f64)> {
        let open = self.node()?;
        let close = self.node()?;
        let [nuc_open, lane_open, nuc_close] = self.array()?;
        if nuc_open > 3 || lane_open > 1 || nuc_close > 3 {
            return Err(malformed());
        }
        let key = PairKey {
            open,
            close,
            opening: Opening::new(nuc_from_index(nuc_open), lane_open),
            nuc_close: nuc_from_index(nuc_close),
        };
        Ok((key, self.f64()?))
    }
}
