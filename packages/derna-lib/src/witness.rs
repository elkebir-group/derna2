//! Cheap rejections for the modularity check.
//!
//! A design is modular when no free structure scores below its
//! constrained MFE. The exact free fold settles that; a design can also be
//! rejected without it by exhibiting *one* free structure below the constrained
//! MFE. This module builds such candidate structures out of structures already
//! seen — the free MFE structures of earlier rejected designs (*witnesses*) and
//! the design's own constrained structure — and scores them on the design's
//! sequence with the linear-time evaluator.
//!
//! A candidate counts only if it lies in the exact fold's structure space
//! ([`admissible`]): canonical pairs, hairpins of at least
//! [`MIN_HAIRPIN_LOOP`] unpaired bases, interior loops and bulges of at most
//! [`MAX_INTERIOR`] unpaired bases, no crossing pairs. Its evaluated energy is
//! then the energy the exact fold assigns that structure, so an upper bound on
//! the free MFE, and a candidate below the constrained MFE is a sound
//! rejection. Nothing here ever accepts a design.

use rna_mfe::{encode_seq, evaluate_encoded, EvalScratch, Params};

/// A structure as base pairs `(i, j)`, `i < j`, 0-based positions.
pub type Pairs = Vec<(u32, u32)>;

/// The exact folds' smallest hairpin loop.
pub const MIN_HAIRPIN_LOOP: u32 = 3;
/// The exact folds' largest interior loop or bulge, in unpaired bases.
pub const MAX_INTERIOR: u32 = 30;

/// Whether `a` and `b` (bytes of `ACGU`) form a canonical pair.
fn canonical(a: u8, b: u8) -> bool {
    matches!(
        (a, b),
        (b'A', b'U') | (b'U', b'A') | (b'G', b'C') | (b'C', b'G') | (b'G', b'U') | (b'U', b'G')
    )
}

/// Whether two pairs cannot coexist: they share a position or cross.
fn conflict(a: (u32, u32), b: (u32, u32)) -> bool {
    let shares = a.0 == b.0 || a.0 == b.1 || a.1 == b.0 || a.1 == b.1;
    let crosses = (a.0 < b.0 && b.0 < a.1 && a.1 < b.1) || (b.0 < a.0 && a.0 < b.1 && b.1 < a.1);
    shares || crosses
}

/// Every position's partner, or `None` when two pairs share a position or a
/// pair leaves the sequence.
fn partners(len: usize, pairs: &[(u32, u32)]) -> Option<Vec<u32>> {
    let mut partner = vec![u32::MAX; len];
    for &(i, j) in pairs {
        if i >= j || j as usize >= len {
            return None;
        }
        if partner[i as usize] != u32::MAX || partner[j as usize] != u32::MAX {
            return None;
        }
        partner[i as usize] = j;
        partner[j as usize] = i;
    }
    Some(partner)
}

/// Whether the pairs nest: reading left to right, each closing position closes
/// the innermost open pair.
fn nested(partner: &[u32]) -> bool {
    let mut open: Vec<u32> = Vec::new();
    for (pos, &p) in partner.iter().enumerate() {
        if p == u32::MAX {
            continue;
        }
        if (p as usize) > pos {
            open.push(pos as u32);
        } else if open.pop() != Some(p) {
            return false;
        }
    }
    open.is_empty()
}

/// The loop a pair `(i, j)` closes: its branch count and unpaired bases.
fn enclosed(partner: &[u32], i: u32, j: u32) -> (u32, u32) {
    let (mut branches, mut unpaired) = (0, 0);
    let mut k = i + 1;
    while k < j {
        let p = partner[k as usize];
        if p == u32::MAX {
            unpaired += 1;
            k += 1;
        } else {
            branches += 1;
            k = p + 1;
        }
    }
    (branches, unpaired)
}

/// Whether `pairs` is a structure the exact free fold of `seq` can produce.
pub fn admissible(seq: &[u8], pairs: &[(u32, u32)]) -> bool {
    let Some(partner) = partners(seq.len(), pairs) else {
        return false;
    };
    if !nested(&partner) {
        return false;
    }
    pairs.iter().all(|&(i, j)| {
        if !canonical(seq[i as usize], seq[j as usize]) {
            return false;
        }
        match enclosed(&partner, i, j) {
            (0, unpaired) => unpaired >= MIN_HAIRPIN_LOOP,
            (1, unpaired) => unpaired <= MAX_INTERIOR,
            _ => true,
        }
    })
}

/// Dot-bracket for `pairs` over `len` positions. The pairs must not share a
/// position.
pub fn dot_bracket(len: usize, pairs: &[(u32, u32)]) -> String {
    let mut s = vec![b'.'; len];
    for &(i, j) in pairs {
        s[i as usize] = b'(';
        s[j as usize] = b')';
    }
    String::from_utf8(s).expect("ASCII")
}

/// Pairs of a dot-bracket string.
pub fn pairs_of(structure: &str) -> Pairs {
    let mut open = Vec::new();
    let mut pairs = Vec::new();
    for (pos, c) in structure.bytes().enumerate() {
        match c {
            b'(' => open.push(pos as u32),
            b')' => pairs.push((open.pop().expect("balanced"), pos as u32)),
            _ => {}
        }
    }
    pairs.sort_unstable();
    pairs
}

/// The maximal stacks of a structure: runs of pairs `(i, j), (i+1, j-1), …`.
pub fn helices(pairs: &[(u32, u32)]) -> Vec<Pairs> {
    let mut sorted = pairs.to_vec();
    sorted.sort_unstable();
    let mut out: Vec<Pairs> = Vec::new();
    for p in sorted {
        let extends = out
            .last()
            .and_then(|h| h.last())
            .is_some_and(|&(i, j)| p.0 == i + 1 && p.1 + 1 == j);
        match out.last_mut() {
            Some(h) if extends => h.push(p),
            _ => out.push(vec![p]),
        }
    }
    out
}

/// One design's sequence, encoded once, with the evaluator's scratch and
/// position-indexed buffers reused from candidate to candidate.
pub struct Scorer<'a> {
    params: &'a Params,
    seq: &'a [u8],
    encoded: Vec<rna_mfe::Nucleotide>,
    scratch: EvalScratch,
    /// Every position's partner, `u32::MAX` between uses.
    partner: Vec<u32>,
    /// [`Scorer::splice`]'s insert closings (`u32::MAX` between uses), loop
    /// labels and open-pair stack.
    close: Vec<u32>,
    loop_of: Vec<u32>,
    open: Vec<(u32, u32)>,
}

impl<'a> Scorer<'a> {
    pub fn new(params: &'a Params, seq: &'a [u8]) -> Self {
        Self {
            params,
            seq,
            encoded: encode_seq(seq).expect("an ACGU sequence"),
            scratch: EvalScratch::with_capacity(seq.len()),
            partner: vec![u32::MAX; seq.len()],
            close: vec![u32::MAX; seq.len()],
            loop_of: vec![EXTERIOR; seq.len()],
            open: Vec::new(),
        }
    }

    /// The energy of `pairs` when it is admissible, for a structure built
    /// from nesting parts: a pool witness with the pairs this sequence cannot
    /// close dropped, or that spliced into a design's own structure. Those
    /// nest by construction, so of [`admissible`]'s tests this skips only
    /// the nesting pass and runs the rest — shared positions, canonical
    /// pairs, hairpin and interior loop sizes, the last being the one
    /// dropping and splicing can break — over reused buffers. Debug builds
    /// check it against [`admissible`].
    pub fn score_constructed(&mut self, pairs: &[(u32, u32)]) -> Option<i32> {
        let ok = self.loops_ok(pairs);
        debug_assert_eq!(ok, admissible(self.seq, pairs), "a constructed candidate");
        if !ok {
            return None;
        }
        let db = dot_bracket(self.seq.len(), pairs);
        let e = evaluate_encoded(self.params, &self.encoded, &db, &mut self.scratch)
            .expect("an admissible structure evaluates");
        Some(e.0)
    }

    /// [`admissible`] without the nesting pass, over the reused partner
    /// buffer: a pair that does not nest makes a loop walk step backwards,
    /// which fails the test rather than looping.
    fn loops_ok(&mut self, pairs: &[(u32, u32)]) -> bool {
        let (seq, partner) = (self.seq, &mut self.partner);
        let mut set = 0;
        let mut ok = true;
        for &(i, j) in pairs {
            let fits = i < j
                && (j as usize) < seq.len()
                && partner[i as usize] == u32::MAX
                && partner[j as usize] == u32::MAX
                && canonical(seq[i as usize], seq[j as usize]);
            if !fits {
                ok = false;
                break;
            }
            partner[i as usize] = j;
            partner[j as usize] = i;
            set += 1;
        }
        if ok {
            ok = pairs.iter().all(|&(i, j)| {
                let (mut branches, mut unpaired, mut k) = (0u32, 0u32, i + 1);
                while k < j {
                    let p = partner[k as usize];
                    if p == u32::MAX {
                        unpaired += 1;
                        k += 1;
                    } else if p > k && p < j {
                        branches += 1;
                        k = p + 1;
                    } else {
                        return false;
                    }
                }
                match branches {
                    0 => unpaired >= MIN_HAIRPIN_LOOP,
                    1 => unpaired <= MAX_INTERIOR,
                    _ => true,
                }
            });
        }
        for &(i, j) in &pairs[..set] {
            partner[i as usize] = u32::MAX;
            partner[j as usize] = u32::MAX;
        }
        ok
    }

    /// [`splice`] over the reused buffers. Pool witnesses and a design's own
    /// structure arrive sorted, so the kept base pairs and the insert are
    /// merged in one pass rather than sorted; unsorted inputs are sorted.
    pub fn splice(&mut self, base: &[(u32, u32)], insert: &[(u32, u32)]) -> Pairs {
        let nests = self.label_loops(base, insert);
        let loop_of = &self.loop_of;
        let kept: Pairs = if nests {
            base.iter()
                .copied()
                .filter(|&(i, j)| {
                    let (li, lj) = (loop_of[i as usize], loop_of[j as usize]);
                    li != PAIRED && li == lj
                })
                .collect()
        } else {
            base.iter()
                .copied()
                .filter(|&b| insert.iter().all(|&h| !conflict(b, h)))
                .collect()
        };
        if !(kept.is_sorted() && insert.is_sorted()) {
            let mut out = kept;
            out.extend_from_slice(insert);
            out.sort_unstable();
            return out;
        }
        let mut out = Vec::with_capacity(kept.len() + insert.len());
        let (mut a, mut b) = (kept.iter().peekable(), insert.iter().peekable());
        while let (Some(&&x), Some(&&y)) = (a.peek(), b.peek()) {
            if x <= y {
                out.push(x);
                a.next();
            } else {
                out.push(y);
                b.next();
            }
        }
        out.extend(a);
        out.extend(b);
        out
    }

    /// [`loops`] into `self.loop_of`, over the positions either structure
    /// touches; `false` when `insert` does not nest or leaves the sequence.
    fn label_loops(&mut self, base: &[(u32, u32)], insert: &[(u32, u32)]) -> bool {
        let Some(len) = base
            .iter()
            .chain(insert)
            .map(|&(_, j)| j as usize + 1)
            .max()
        else {
            return true;
        };
        if len > self.close.len() {
            return false;
        }
        let (close, loop_of, open) = (&mut self.close, &mut self.loop_of, &mut self.open);
        let mut set = 0;
        let mut nests = true;
        for &(i, j) in insert {
            if i >= j || close[i as usize] != u32::MAX {
                nests = false;
                break;
            }
            close[i as usize] = j;
            set += 1;
        }
        open.clear();
        if nests {
            for pos in 0..len as u32 {
                if open.last().is_some_and(|&(_, j)| j == pos) {
                    if close[pos as usize] != u32::MAX {
                        nests = false;
                        break;
                    }
                    open.pop();
                    loop_of[pos as usize] = PAIRED;
                    continue;
                }
                let j = close[pos as usize];
                if j != u32::MAX {
                    if open.last().is_some_and(|&(_, outer)| j > outer) {
                        nests = false;
                        break;
                    }
                    open.push((pos, j));
                    loop_of[pos as usize] = PAIRED;
                    continue;
                }
                loop_of[pos as usize] = open.last().map_or(EXTERIOR, |&(i, _)| i);
            }
        }
        for &(i, _) in &insert[..set] {
            close[i as usize] = u32::MAX;
        }
        nests && open.is_empty()
    }

    /// The energy of `pairs` in dekacalories when it is admissible on this
    /// sequence, else `None`.
    pub fn score(&mut self, pairs: &[(u32, u32)]) -> Option<i32> {
        if !admissible(self.seq, pairs) {
            return None;
        }
        let db = dot_bracket(self.seq.len(), pairs);
        let e = evaluate_encoded(self.params, &self.encoded, &db, &mut self.scratch)
            .expect("an admissible structure evaluates");
        Some(e.0)
    }

    /// The sequence being scored.
    pub fn seq(&self) -> &[u8] {
        self.seq
    }
}

/// `pairs` with every pair that is not canonical on `seq` dropped.
pub fn canonical_only(seq: &[u8], pairs: &[(u32, u32)]) -> Pairs {
    pairs
        .iter()
        .copied()
        .filter(|&(i, j)| canonical(seq[i as usize], seq[j as usize]))
        .collect()
}

/// `base` with `insert` added and every pair of `base` that conflicts with
/// one of `insert` dropped.
///
/// Linear in the span of the two structures when `insert` nests: a pair of
/// `base` fits alongside `insert` exactly when neither end is paired in
/// `insert` and both ends lie in the same loop of it, since a pair of `insert`
/// holding one end and not the other is a crossing. An `insert` that does not
/// nest is spliced pair by pair.
pub fn splice(base: &[(u32, u32)], insert: &[(u32, u32)]) -> Pairs {
    let mut out: Pairs = match loops(base, insert) {
        Some(loop_of) => base
            .iter()
            .copied()
            .filter(|&(i, j)| {
                let (li, lj) = (loop_of[i as usize], loop_of[j as usize]);
                li != PAIRED && li == lj
            })
            .collect(),
        None => base
            .iter()
            .copied()
            .filter(|&b| insert.iter().all(|&h| !conflict(b, h)))
            .collect(),
    };
    out.extend_from_slice(insert);
    out.sort_unstable();
    out
}

/// Marks a position paired in the insert, in [`loops`].
const PAIRED: u32 = u32::MAX;
/// Marks the exterior loop, in [`loops`].
const EXTERIOR: u32 = u32::MAX - 1;

/// For every position up to the last one either structure touches: the
/// opening position of the innermost `insert` pair enclosing it, [`EXTERIOR`]
/// outside all of them, or [`PAIRED`] when `insert` pairs it. `None` when
/// `insert` does not nest.
fn loops(base: &[(u32, u32)], insert: &[(u32, u32)]) -> Option<Vec<u32>> {
    let len = base
        .iter()
        .chain(insert)
        .map(|&(_, j)| j as usize + 1)
        .max()?;
    let mut close = vec![u32::MAX; len];
    for &(i, j) in insert {
        if i >= j || close[i as usize] != u32::MAX {
            return None;
        }
        close[i as usize] = j;
    }
    let mut loop_of = vec![EXTERIOR; len];
    let mut open: Vec<(u32, u32)> = Vec::new();
    for pos in 0..len as u32 {
        if open.last().is_some_and(|&(_, j)| j == pos) {
            if close[pos as usize] != u32::MAX {
                return None;
            }
            open.pop();
            loop_of[pos as usize] = PAIRED;
            continue;
        }
        let j = close[pos as usize];
        if j != u32::MAX {
            if open.last().is_some_and(|&(_, outer)| j > outer) {
                return None;
            }
            open.push((pos, j));
            loop_of[pos as usize] = PAIRED;
            continue;
        }
        loop_of[pos as usize] = open.last().map_or(EXTERIOR, |&(i, _)| i);
    }
    open.is_empty().then_some(loop_of)
}

/// Recent witnesses, newest first, capped at a fixed count.
///
/// [`WitnessPool::reject`] tries each witness on a new design two ways: the
/// witness alone, with the pairs the new sequence cannot form dropped; and the
/// same pairs spliced into the design's own constrained structure, dropping the
/// design's pairs they conflict with. The second is what rejects: consecutive
/// designs differ at a few codons, and the witness carries the pairs the
/// constraint forbids while the design's own structure carries the rest.
///
/// Witnesses are shared, so a clone costs a reference count per witness: a
/// caller can score against a snapshot while another adds to the original.
#[derive(Debug, Clone)]
pub struct WitnessPool {
    cap: usize,
    structures: std::collections::VecDeque<std::sync::Arc<Pairs>>,
}

impl WitnessPool {
    /// An empty pool holding at most `cap` witnesses; `0` never rejects.
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            structures: std::collections::VecDeque::with_capacity(cap),
        }
    }

    /// Add a free structure some design was rejected by. A structure already
    /// held is not added twice; past the cap the oldest is dropped.
    pub fn add(&mut self, pairs: Pairs) {
        if self.cap == 0 || self.structures.iter().any(|s| **s == pairs) {
            return;
        }
        self.structures.push_front(std::sync::Arc::new(pairs));
        self.structures.truncate(self.cap);
    }

    /// The `w`-th witness, in scoring order.
    pub fn witness(&self, w: usize) -> Option<std::sync::Arc<Pairs>> {
        self.structures.get(w).cloned()
    }

    /// Move `source` to the front, when the pool still holds it: a witness
    /// that most recently rejected a design is tried first on the next. Matched by
    /// identity, so a snapshot's witness finds itself in the live pool.
    pub fn promote(&mut self, source: &std::sync::Arc<Pairs>) {
        if let Some(pos) = self
            .structures
            .iter()
            .position(|s| std::sync::Arc::ptr_eq(s, source))
        {
            if let Some(s) = self.structures.remove(pos) {
                self.structures.push_front(s);
            }
        }
    }

    /// How many witnesses the pool holds.
    pub fn len(&self) -> usize {
        self.structures.len()
    }

    /// Whether the pool holds no witness.
    pub fn is_empty(&self) -> bool {
        self.structures.is_empty()
    }

    /// The lowest-energy admissible structure built from the pool for the
    /// design `seq` (bytes of `ACGU`) whose constrained structure is `own`,
    /// when it scores below `constrained` (dekacalories): a sound rejection,
    /// with its energy and pairs. `None` says nothing about the design.
    pub fn reject(
        &self,
        params: &Params,
        seq: &[u8],
        constrained: i32,
        own: &[(u32, u32)],
    ) -> Option<(i32, Pairs)> {
        self.reject_ranked(params, seq, constrained, own, 0)
            .map(|(e, pairs, _)| (e, pairs))
    }

    /// As [`WitnessPool::reject`], but the first candidate, in pool order, that
    /// scores below `constrained`, not the lowest: any such structure is a
    /// sound rejection, and the rest go unscored.
    pub fn reject_first(
        &self,
        params: &Params,
        seq: &[u8],
        constrained: i32,
        own: &[(u32, u32)],
    ) -> Option<(i32, Pairs)> {
        self.reject_ranked(params, seq, constrained, own, 1)
            .map(|(e, pairs, _)| (e, pairs))
    }

    /// The lowest rejecting candidate among the first `stop_after`, in
    /// scoring order, that score below `constrained`; the rest go unscored.
    /// `0` scores every candidate ([`WitnessPool::reject`]), `1` is
    /// [`WitnessPool::reject_first`]. Any rejecting candidate is sound; a
    /// larger `stop_after` costs more scoring and hands the pool a lower,
    /// usually more widely rejecting, structure. Also returns the kept
    /// candidate's index in scoring order: `2·w` for the `w`-th witness
    /// alone, `2·w + 1` for it spliced into `own`.
    pub fn reject_ranked(
        &self,
        params: &Params,
        seq: &[u8],
        constrained: i32,
        own: &[(u32, u32)],
        stop_after: usize,
    ) -> Option<(i32, Pairs, usize)> {
        let mut scorer = Scorer::new(params, seq);
        let mut best: Option<(i32, Pairs, usize)> = None;
        let mut rejecting = 0;
        for (w, structure) in self.structures.iter().enumerate() {
            let repaired = canonical_only(seq, structure);
            let spliced = scorer.splice(own, &repaired);
            for (kind, candidate) in [repaired, spliced].into_iter().enumerate() {
                let Some(e) = scorer.score_constructed(&candidate) else {
                    continue;
                };
                if e >= constrained {
                    continue;
                }
                if best.as_ref().is_none_or(|(b, _, _)| e < *b) {
                    best = Some((e, candidate, 2 * w + kind));
                }
                rejecting += 1;
                if rejecting == stop_after {
                    return best;
                }
            }
        }
        best
    }
}
