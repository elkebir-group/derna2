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

#[cfg(test)]
mod tests {
    use super::*;

    const SEQ: &[u8] = b"GGGAAACCCAGCUUCGGCUGGGCGCAAGCGCCAUAUGGCAUCCGAAAGGAUGCC";

    fn params() -> Params {
        Params::load_default().expect("params")
    }

    fn exact(params: &Params, seq: &[u8]) -> (Pairs, i32) {
        let (db, e) = rna_linearfold::fold(params, seq, 1 << 20).expect("folds");
        (pairs_of(&db), e.0)
    }

    /// A 60-nt chain of `A` with `G` at `gs` and `C` at `cs`.
    fn chain(gs: &[usize], cs: &[usize]) -> Vec<u8> {
        let mut s = vec![b'A'; 60];
        gs.iter().for_each(|&i| s[i] = b'G');
        cs.iter().for_each(|&i| s[i] = b'C');
        s
    }

    #[test]
    fn only_structures_the_exact_fold_can_return_are_admissible() {
        let s = chain(&[0, 1], &[10, 11]);
        assert!(admissible(&s, &[(0, 11), (1, 10)]), "a stacked hairpin");
        assert!(!admissible(&s, &[(0, 11), (2, 9)]), "A–A is not a pair");
        assert!(
            !admissible(&chain(&[0], &[3]), &[(0, 3)]),
            "a two-base hairpin"
        );
        assert!(!admissible(&s, &[(0, 11), (0, 10)]), "a shared position");
        let x = chain(&[0, 5], &[11, 20]);
        assert!(!admissible(&x, &[(0, 11), (5, 20)]), "crossing pairs");
        // Outer pair (0, 59), inner pair closing an interior loop of 30, then 31.
        let at_cap = chain(&[0, 21], &[48, 59]);
        assert!(
            admissible(&at_cap, &[(0, 59), (21, 48)]),
            "an interior loop of 30"
        );
        let past = chain(&[0, 22], &[48, 59]);
        assert!(
            !admissible(&past, &[(0, 59), (22, 48)]),
            "an interior loop of 31"
        );
        assert!(admissible(&past, &[(0, 59)]), "the same chain as a hairpin");
    }

    #[test]
    fn a_folded_structure_scores_its_fold_energy() {
        let params = params();
        let (pairs, e) = exact(&params, SEQ);
        assert!(!pairs.is_empty(), "the test sequence folds");
        assert_eq!(Scorer::new(&params, SEQ).score(&pairs), Some(e));
    }

    /// One mutation away from the witness's sequence, a design whose own
    /// structure is the open chain is rejected by the witness, and never
    /// below its exact free MFE.
    #[test]
    fn a_witness_rejects_a_neighbour_soundly() {
        let params = params();
        let (w, _) = exact(&params, SEQ);
        let mut design = SEQ.to_vec();
        design[4] = b'U';
        let (_, free) = exact(&params, &design);
        let mut pool = WitnessPool::new(4);
        pool.add(w);
        let (e, pairs) = pool.reject(&params, &design, 0, &[]).expect("rejects");
        assert!(e < 0 && e >= free, "{e} against free MFE {free}");
        assert_eq!(Scorer::new(&params, &design).score(&pairs), Some(e));
    }

    /// At the design's exact free MFE no witness can reject, however many the
    /// pool holds: the design must fall through to the exact fold.
    #[test]
    fn a_modular_design_falls_through() {
        let params = params();
        let (own, free) = exact(&params, SEQ);
        let mut pool = WitnessPool::new(16);
        assert!(pool.reject(&params, SEQ, free, &own).is_none());
        for k in 1..8 {
            let mut other = SEQ.to_vec();
            other.rotate_left(k * 7);
            pool.add(exact(&params, &other).0);
        }
        pool.add(own.clone());
        assert!(pool.len() > 1);
        assert!(pool.reject(&params, SEQ, free, &own).is_none());
        assert!(pool.reject_first(&params, SEQ, free, &own).is_none());
        // One dekacalorie above the free MFE, the same pool does reject, and
        // the first rejecting candidate scores what the evaluator gives it.
        assert!(pool.reject(&params, SEQ, free + 1, &own).is_some());
        let (e, pairs) = pool
            .reject_first(&params, SEQ, free + 1, &own)
            .expect("rejects");
        assert_eq!(e, free, "below free + 1 and never below the free MFE");
        assert_eq!(Scorer::new(&params, SEQ).score(&pairs), Some(e));
    }

    /// Promoting a witness moves it to the front, found by identity in a
    /// snapshot taken before; one the pool no longer holds changes nothing.
    #[test]
    fn promoting_moves_a_held_witness_to_the_front() {
        let mut pool = WitnessPool::new(3);
        for k in 0..3u32 {
            pool.add(vec![(k, k + 10)]);
        }
        let snapshot = pool.clone();
        let oldest = snapshot.witness(2).expect("three witnesses");
        assert_eq!(*oldest, vec![(0, 10)]);
        pool.promote(&oldest);
        assert_eq!(pool.witness(0).as_deref(), Some(&vec![(0, 10)]));
        assert_eq!(pool.witness(1).as_deref(), Some(&vec![(2, 12)]));
        let evicted = pool.witness(2).expect("three witnesses");
        pool.add(vec![(5, 15)]);
        pool.promote(&evicted);
        assert_eq!(pool.witness(0).as_deref(), Some(&vec![(5, 15)]));
        assert_eq!(pool.len(), 3);
    }

    /// Stopping after more rejecting candidates keeps a structure no higher:
    /// the kept energy falls (weakly) as `stop_after` grows, every kept
    /// structure scores what it reports, and scoring every candidate is the
    /// floor.
    #[test]
    fn stopping_later_keeps_a_lower_witness() {
        let params = params();
        let (own, free) = exact(&params, SEQ);
        let mut pool = WitnessPool::new(16);
        for k in 1..12 {
            let mut other = SEQ.to_vec();
            other.rotate_left((k * 5) % SEQ.len());
            pool.add(exact(&params, &other).0);
        }
        pool.add(own.clone());
        // A loose threshold, so that several candidates reject.
        let loose = free + 400;
        let all = pool
            .reject_ranked(&params, SEQ, loose, &own, 0)
            .expect("rejects")
            .0;
        let mut last = i32::MAX;
        for stop in 1..=8 {
            let (e, pairs, _) = pool
                .reject_ranked(&params, SEQ, loose, &own, stop)
                .expect("rejects");
            assert!(e <= last, "stop {stop}: {e} above {last}");
            assert!(e >= all, "stop {stop}: {e} below the full scan's {all}");
            assert_eq!(Scorer::new(&params, SEQ).score(&pairs), Some(e));
            last = e;
        }
    }

    #[test]
    fn the_pool_keeps_the_newest_distinct_witnesses() {
        let mut pool = WitnessPool::new(2);
        pool.add(vec![(0, 5)]);
        pool.add(vec![(0, 5)]);
        assert_eq!(pool.len(), 1);
        pool.add(vec![(1, 6)]);
        pool.add(vec![(2, 7)]);
        assert_eq!(pool.len(), 2);
        assert!(!pool.structures.iter().any(|s| **s == vec![(0, 5)]));
        let mut off = WitnessPool::new(0);
        off.add(vec![(0, 5)]);
        assert!(off.is_empty());
    }

    /// The linear splice keeps exactly the pairs the pair-by-pair conflict
    /// test keeps, over folded structures of unrelated sequences.
    #[test]
    fn the_linear_splice_is_the_pairwise_one() {
        let params = params();
        let folds: Vec<Pairs> = (0..8)
            .map(|k| {
                let mut s = SEQ.to_vec();
                s.rotate_left(k * 7);
                exact(&params, &s).0
            })
            .collect();
        let mut compared = 0;
        for base in &folds {
            for insert in &folds {
                let mut want: Pairs = base
                    .iter()
                    .copied()
                    .filter(|&b| insert.iter().all(|&h| !conflict(b, h)))
                    .chain(insert.iter().copied())
                    .collect();
                want.sort_unstable();
                assert_eq!(splice(base, insert), want);
                compared += usize::from(want.len() < base.len() + insert.len());
            }
        }
        assert!(compared > 0, "no splice dropped a pair");
        assert!(loops(&[], &[(0, 5), (2, 8)]).is_none(), "a crossing insert");
    }

    /// The scorer's buffered splice and its check for constructed candidates
    /// agree with [`splice`] and the full [`Scorer::score`] on witnesses from
    /// other sequences, repaired to this one and spliced into its own
    /// structure, buffers reused across every candidate; and a crossing
    /// insert falls back to the pairwise splice.
    #[test]
    fn the_buffered_scorer_matches_the_plain_one() {
        let params = params();
        let (own, _) = exact(&params, SEQ);
        let mut scorer = Scorer::new(&params, SEQ);
        let (mut scored, mut refused) = (0, 0);
        for k in 1..10 {
            let mut other = SEQ.to_vec();
            other.rotate_left((k * 7) % SEQ.len());
            let witness = exact(&params, &other).0;
            let repaired = canonical_only(SEQ, &witness);
            let spliced = scorer.splice(&own, &repaired);
            assert_eq!(spliced, splice(&own, &repaired), "witness {k}");
            for candidate in [repaired, spliced] {
                let fast = scorer.score_constructed(&candidate);
                assert_eq!(
                    fast,
                    Scorer::new(&params, SEQ).score(&candidate),
                    "witness {k}"
                );
                scored += usize::from(fast.is_some());
                refused += usize::from(fast.is_none());
            }
        }
        assert!(scored > 0, "no constructed candidate scored");
        // The one test dropping and splicing can fail: an interior loop past
        // the cap, as when a dropped pair merges two loops.
        let (at_cap, past) = (chain(&[0, 21], &[48, 59]), chain(&[0, 22], &[48, 59]));
        assert!(Scorer::new(&params, &at_cap)
            .score_constructed(&[(0, 59), (21, 48)])
            .is_some());
        assert_eq!(
            Scorer::new(&params, &past).score_constructed(&[(0, 59), (22, 48)]),
            None,
            "an interior loop of 31"
        );
        let crossing = [(0, 5), (2, 8)];
        assert_eq!(
            scorer.splice(&[(1, 9)], &crossing),
            splice(&[(1, 9)], &crossing)
        );
        eprintln!("constructed candidates: {scored} scored, {refused} refused");
    }

    #[test]
    fn splicing_drops_the_pairs_an_insert_conflicts_with() {
        let base = vec![(0, 20), (2, 10), (12, 18)];
        assert_eq!(splice(&base, &[(5, 15)]), vec![(0, 20), (5, 15)]);
        assert_eq!(helices(&[(0, 20), (1, 19), (2, 18), (5, 10)]).len(), 2);
    }
}
