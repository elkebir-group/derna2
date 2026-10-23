//! A Zuker fold that iterates only over the pairs it keeps, with an optional
//! Carrillo–Lipman cut that keeps fewer.
//!
//! # The sparsification (exact)
//!
//! The dense fold's multi-loop terms sum `M[i, k] + M1[k+1, j]` over every
//! `k`. Expanding `M1`'s own recurrence, the branches that end before `j` are
//! already covered by the right-extension cases, so only the branches whose
//! closing pair ends exactly at `j` need summing:
//!
//! ```text
//! M[i, j]  = min( V'[i, j],  M[i+1, j] + b,  M[i, j-1] + b,
//!                 min over s in [i+2, j]:  M[i, s-1] + V'[s, j] )
//! M2[i, j] = min( M2[i, j-1] + b,
//!                 min over s in [i+1, j]:  M[i, s-1] + V'[s, j] )
//! V_multi(i, j) = closing + M2[i+1, j-1]
//! F[j]     = min( F[j-1],  min over i:  F[i] + V[i, j-1] + au(i, j-1) )
//! ```
//!
//! with `V' = V + intern + au` and `b` the unpaired multi-loop base. `M2` is
//! the dense fold's `min over k: M + M1` closing term; `M1` is gone. Every sum
//! now runs over the pairs `(s, j)` *kept* in column `j`, and the internal-loop
//! search over the pairs kept in each column of its window. With nothing cut
//! this is the dense fold's MFE exactly.
//!
//! # The cut
//!
//! Given a threshold `t` and, for each pair `(i, j)`, a lower bound on the
//! energy of everything outside it, a pair whose `V[i, j] + bound(i, j)`
//! reaches `t` lies on no structure below `t`, so it is dropped as soon as its
//! `V` is known: no later cell reads it, and the sums skip it. The fold then
//! answers "is the MFE below `t`?" exactly, and below `t` returns the MFE and
//! an MFE structure; at or above `t` it says only that.
//!
//! A cut may also carry a lower bound on `V[i, j]` itself. A pair whose bound
//! plus its outside bound already reaches `t` is dropped before its `V` is
//! computed, which saves the hairpin, internal-loop and multi-loop terms the
//! post-hoc test pays for every pair it then drops.

use crate::loops::{
    bulge_energy, hairpin_energy, internal_energy, pair_from_nucleotides, stack_energy, BulgeLoop,
    Energy, HairpinLoop, InternalLoop, Nucleotide, StackPair,
};
use crate::params::{BasePair, Params};

use super::tables::{MAX_LOOP_SIZE, MIN_HAIRPIN_LOOP};

const INF: i32 = i32::MAX / 4;

/// What [`fold_sparse`] concluded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseFold {
    /// The MFE and an MFE structure in dot-bracket. With a cut, the MFE lies
    /// below its threshold.
    Below(Energy, String),
    /// With a cut: no structure lies below the threshold.
    AtOrAbove,
}

/// A Carrillo–Lipman cut: a threshold in dekacalories and, for each pair
/// `(i, j)`, a lower bound on the energy outside it. Return a very negative
/// number for a pair with no bound.
pub struct Cut<'a> {
    pub threshold: i32,
    pub outside: &'a dyn Fn(usize, usize) -> i32,
    /// For each pair `(i, j)`, a lower bound on `V[i, j]`, read before `V` is
    /// computed; a very negative number where there is none. `None` tests
    /// each pair only once its `V` is known.
    pub inside: Option<&'a dyn Fn(usize, usize) -> i32>,
}

/// Upper-triangle storage, `(i, j)` with `i <= j`, row-major by `i`.
struct Tri {
    n: usize,
    cells: Vec<i32>,
}

impl Tri {
    fn new(n: usize) -> Self {
        Self {
            n,
            cells: vec![INF; n * (n + 1) / 2],
        }
    }

    #[inline]
    fn at(&self, i: usize, j: usize) -> usize {
        debug_assert!(i <= j && j < self.n);
        i * self.n - i * (i + 1) / 2 + j
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> i32 {
        self.cells[self.at(i, j)]
    }

    #[inline]
    fn set(&mut self, i: usize, j: usize, v: i32) {
        let k = self.at(i, j);
        self.cells[k] = v;
    }

    /// Where row `i` starts, less `i`'s own offset: cell `(i, j)` is at
    /// `row(i) + j`.
    #[inline]
    fn row(&self, i: usize) -> usize {
        i * self.n - i * (i + 1) / 2
    }
}

/// A kept pair `(s, j)`, filed under its column `j`: its opening, its `V`,
/// and its `V` as a multi-loop branch, so the sums read no table.
#[derive(Clone, Copy)]
struct Kept {
    s: u32,
    v: i32,
    branch: i32,
}

struct Fold<'a> {
    params: &'a Params,
    seq: &'a [Nucleotide],
    n: usize,
    v: Tri,
    m: Tri,
    m2: Tri,
    f: Vec<i32>,
    /// For each column `j`, the pairs `(i, j)` kept, in decreasing `i` (the
    /// order the fold finishes them).
    kept: Vec<Vec<Kept>>,
    base: i32,
    closing: i32,
    /// Pairs dropped by the cut's inside bound before their `V` was computed.
    precut: u64,
    /// Cells copied from an earlier fold ([`Reuse`]).
    reused: u64,
    /// [`interior_bounds`] of the parameters.
    bounds: &'a [i32],
}

/// Side of the [`interior_bounds`] table: unpaired lengths `0..=MAX_LOOP_SIZE`.
const BOUND_SIDE: usize = MAX_LOOP_SIZE + 1;

/// For each pair of unpaired lengths `(u1, u2)`, at `u1 * BOUND_SIDE + u2`,
/// the least energy any stack (`0, 0`), bulge or interior loop with those
/// lengths can have, over every pair of canonical closing pairs and every
/// nucleotide at the loop's edges (the only nucleotides these energies read);
/// `INF` past the loop-size cap. Computed once per parameter set.
fn interior_bounds(params: &Params) -> &[i32] {
    params.derived.interior_bounds.get_or_init(|| {
        let nucs = [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U];
        let pairs: Vec<BasePair> = nucs
            .iter()
            .flat_map(|&a| nucs.iter().map(move |&b| pair_from_nucleotides(a, b)))
            .filter(|p| !matches!(p, BasePair::NN))
            .collect();
        let mut table = vec![INF; BOUND_SIDE * BOUND_SIDE];
        let (mut left, mut right) = (
            vec![Nucleotide::A; BOUND_SIDE],
            vec![Nucleotide::A; BOUND_SIDE],
        );
        for u1 in 0..BOUND_SIDE {
            for u2 in 0..BOUND_SIDE - u1 {
                let mut least = INF;
                for &outer in &pairs {
                    for &inner in &pairs {
                        if u1 == 0 && u2 == 0 {
                            least = least.min(stack_energy(params, &StackPair { outer, inner }).0);
                        } else if u1 == 0 || u2 == 0 {
                            let size = u1.max(u2);
                            least = least
                                .min(bulge_energy(params, &BulgeLoop { outer, inner, size }).0);
                        } else {
                            for &a in &nucs {
                                for &b in &nucs {
                                    for &c in &nucs {
                                        for &d in &nucs {
                                            left[0] = a;
                                            left[u1 - 1] = b;
                                            right[0] = c;
                                            right[u2 - 1] = d;
                                            let e = internal_energy(
                                                params,
                                                &InternalLoop {
                                                    outer,
                                                    inner,
                                                    left: &left[..u1],
                                                    right: &right[..u2],
                                                },
                                            );
                                            least = least.min(e.0);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                table[u1 * BOUND_SIDE + u2] = least;
            }
        }
        table
    })
}

/// A finished fold's tables, kept so that a later fold of a similar
/// sequence can copy the spans the two share ([`Reuse`]).
pub struct FoldTables {
    v: Tri,
    m: Tri,
    m2: Tri,
    /// The threshold the fold was cut at; `i32::MAX` without a cut.
    threshold: i32,
}

impl FoldTables {
    /// The threshold the fold was cut at; `i32::MAX` without a cut.
    pub fn threshold(&self) -> i32 {
        self.threshold
    }

    /// The length of the sequence folded.
    pub fn len(&self) -> usize {
        self.v.n
    }

    /// Whether the folded sequence was empty.
    pub fn is_empty(&self) -> bool {
        self.v.n == 0
    }
}

/// Spans an earlier fold's [`FoldTables`] can stand in for.
///
/// A span `[i, j]` may be copied only when its cells hold the same values in
/// both folds: the same nucleotides on `i..=j`, and the same cut test for
/// every pair inside it, so the same outside bound and a threshold no higher
/// than the earlier fold's. Then a pair the earlier fold cut is cut here too,
/// and a copied value lies between the true one and what this fold would
/// compute, because the earlier fold kept a superset of this one's pairs; so
/// the verdict, the MFE below the threshold and its traceback are exact. The
/// caller vouches for the bounds via `until`; the threshold is checked.
///
/// Each row may copy from a different earlier fold: a copied cell's value
/// depends only on its own span and that fold's threshold, so cells from
/// different folds can sit side by side.
pub struct Reuse<'a> {
    pub tables: &'a [&'a FoldTables],
    /// For each `i`, spans `[i, j]` with `j < until[i]` may be copied from
    /// `tables[from[i]]`.
    pub until: &'a [usize],
    pub from: &'a [usize],
}

fn au(params: &Params, pair: BasePair) -> i32 {
    if pair.applies_terminal_au() {
        params.misc.terminal_au_37
    } else {
        0
    }
}

impl<'a> Fold<'a> {
    fn new(params: &'a Params, seq: &'a [Nucleotide]) -> Self {
        let n = seq.len();
        Self {
            params,
            seq,
            n,
            v: Tri::new(n.max(1)),
            m: Tri::new(n.max(1)),
            m2: Tri::new(n.max(1)),
            f: vec![0; n + 1],
            kept: vec![Vec::new(); n],
            base: params.ml.base_37,
            closing: params.ml.closing_37,
            precut: 0,
            reused: 0,
            bounds: interior_bounds(params),
        }
    }

    /// Copy cell `(i, j)` from an earlier fold, re-testing a copied pair
    /// against this fold's cut.
    fn copy_cell(&mut self, cut: Option<&Cut<'_>>, from: &FoldTables, i: usize, j: usize) {
        let mut v = from.v.get(i, j);
        if v < INF {
            if let Some(c) = cut {
                if v.saturating_add((c.outside)(i, j)) >= c.threshold {
                    v = INF;
                }
            }
        }
        if v < INF {
            self.v.set(i, j, v);
            let branch = self.branch(i, j);
            self.kept[j].push(Kept {
                s: i as u32,
                v,
                branch,
            });
        }
        self.m.set(i, j, from.m.get(i, j));
        self.m2.set(i, j, from.m2.get(i, j));
        self.reused += 1;
    }

    fn into_tables(self, threshold: i32) -> FoldTables {
        FoldTables {
            v: self.v,
            m: self.m,
            m2: self.m2,
            threshold,
        }
    }

    #[inline]
    fn pair(&self, i: usize, j: usize) -> BasePair {
        pair_from_nucleotides(self.seq[i], self.seq[j])
    }

    /// `V'`: a kept pair's `V` as a multi-loop branch.
    #[inline]
    fn branch(&self, s: usize, j: usize) -> i32 {
        self.v.get(s, j) + self.params.ml.intern_37 + au(self.params, self.pair(s, j))
    }

    fn two_pair(&self, outer: BasePair, i: usize, j: usize, k: usize, l: usize) -> i32 {
        let inner = self.pair(k, l);
        let (u1, u2) = (k - i - 1, j - l - 1);
        let e = if u1 == 0 && u2 == 0 {
            stack_energy(self.params, &StackPair { outer, inner })
        } else if u1 == 0 || u2 == 0 {
            bulge_energy(
                self.params,
                &BulgeLoop {
                    outer,
                    inner,
                    size: u1.max(u2),
                },
            )
        } else {
            internal_energy(
                self.params,
                &InternalLoop {
                    outer,
                    inner,
                    left: &self.seq[i + 1..k],
                    right: &self.seq[l + 1..j],
                },
            )
        };
        e.0
    }

    /// The best internal loop, bulge or stack closed by `(i, j)`, over the
    /// inner pairs kept. An inner pair whose `V` plus the least loop energy
    /// its unpaired lengths allow ([`interior_bounds`]) already reaches the
    /// best found cannot improve it, so its loop energy is not computed.
    fn interior(&self, outer: BasePair, i: usize, j: usize) -> i32 {
        let l_lower = (i + MIN_HAIRPIN_LOOP + 2).max(j.saturating_sub(MAX_LOOP_SIZE + 1));
        let bounds = self.bounds;
        let mut best = INF;
        for l in (l_lower..j).rev() {
            let u2 = j - l - 1;
            let last_k = ((i + 1) + MAX_LOOP_SIZE.saturating_sub(u2))
                .min(l.saturating_sub(MIN_HAIRPIN_LOOP + 1));
            // Smallest opening first; the row being filled may already have
            // added `(i, l)`, which is not an inner pair.
            for kp in self.kept[l].iter().rev() {
                let k = kp.s as usize;
                if k <= i {
                    continue;
                }
                if k > last_k {
                    break;
                }
                if kp.v + bounds[(k - i - 1) * BOUND_SIDE + u2] >= best {
                    continue;
                }
                let total = kp.v + self.two_pair(outer, i, j, k, l);
                if total < best {
                    best = total;
                }
            }
        }
        best
    }

    fn compute_v(&self, i: usize, j: usize) -> i32 {
        if j - i < MIN_HAIRPIN_LOOP + 1 {
            return INF;
        }
        let pair = self.pair(i, j);
        if matches!(pair, BasePair::NN) {
            return INF;
        }
        let mut best = hairpin_energy(
            self.params,
            &HairpinLoop {
                closing: pair,
                loop_seq: &self.seq[i + 1..j],
            },
        )
        .0;
        best = best.min(self.interior(pair, i, j));
        if j >= i + 2 {
            let inner = self.m2.get(i + 1, j - 1);
            if inner < INF {
                let multi = inner + self.closing + self.params.ml.intern_37 + au(self.params, pair);
                best = best.min(multi);
            }
        }
        best
    }

    /// `min over kept (s, j): M[i, s-1] + V'[s, j]`, over `s >= i + 2` (the
    /// `M` split) and over `s >= i + 1` (the `M2` split), in one pass.
    fn branch_splits(&self, i: usize, j: usize) -> (i32, i32) {
        let row = self.m.row(i);
        let (mut m, mut m2) = (INF, INF);
        for kp in &self.kept[j] {
            let s = kp.s as usize;
            if s <= i {
                continue;
            }
            let left = self.m.cells[row + s - 1];
            if left < INF {
                let total = left + kp.branch;
                m2 = m2.min(total);
                if s >= i + 2 {
                    m = m.min(total);
                }
            }
        }
        (m, m2)
    }

    /// Whether the cut's inside bound drops `(i, j)` before its `V` is
    /// computed. Only pairs that can close are bounded.
    fn precut(&mut self, cut: Option<&Cut<'_>>, i: usize, j: usize) -> bool {
        let Some((c, inside)) = cut.and_then(|c| c.inside.map(|f| (c, f))) else {
            return false;
        };
        if j - i < MIN_HAIRPIN_LOOP + 1 || matches!(self.pair(i, j), BasePair::NN) {
            return false;
        }
        let dropped = inside(i, j).saturating_add((c.outside)(i, j)) >= c.threshold;
        self.precut += u64::from(dropped);
        dropped
    }

    fn fill(&mut self, cut: Option<&Cut<'_>>) {
        self.fill_reusing(cut, None);
    }

    /// Fill, copying the cells `reuse` allows from the tables `usable` marks.
    fn fill_reusing(&mut self, cut: Option<&Cut<'_>>, reuse: Option<(&Reuse<'_>, &[bool])>) {
        let n = self.n;
        for i in (0..n).rev() {
            let row = reuse.filter(|(r, usable)| usable[r.from[i]]);
            for j in i..n {
                if let Some((r, _)) = row.filter(|(r, _)| j < r.until[i]) {
                    self.copy_cell(cut, r.tables[r.from[i]], i, j);
                    continue;
                }
                let mut v = if j > i && !self.precut(cut, i, j) {
                    self.compute_v(i, j)
                } else {
                    INF
                };
                if v < INF {
                    if let Some(c) = cut {
                        debug_assert!(
                            c.inside.is_none_or(|f| f(i, j) <= v),
                            "an inside bound above the pair's V"
                        );
                        if v.saturating_add((c.outside)(i, j)) >= c.threshold {
                            v = INF;
                        }
                    }
                }
                let mut m = INF;
                if v < INF {
                    self.v.set(i, j, v);
                    m = self.branch(i, j);
                    self.kept[j].push(Kept {
                        s: i as u32,
                        v,
                        branch: m,
                    });
                }
                let mut m2 = INF;
                if j > i {
                    let left = self.m.get(i + 1, j);
                    if left < INF {
                        m = m.min(left + self.base);
                    }
                    let right = self.m.get(i, j - 1);
                    if right < INF {
                        m = m.min(right + self.base);
                    }
                    let (split, split2) = self.branch_splits(i, j);
                    m = m.min(split);
                    let ext = self.m2.get(i, j - 1);
                    if ext < INF {
                        m2 = ext + self.base;
                    }
                    m2 = m2.min(split2);
                }
                self.m.set(i, j, m);
                self.m2.set(i, j, m2);
            }
        }
        for j in 1..=n {
            let mut best = self.f[j - 1];
            for kp in &self.kept[j - 1] {
                let i = kp.s as usize;
                let total = self.f[i] + kp.v + au(self.params, self.pair(i, j - 1));
                best = best.min(total);
            }
            self.f[j] = best;
        }
    }

    // ---- traceback ---------------------------------------------------------

    fn trace_f(&self, mut j: usize, s: &mut [u8]) {
        while j > 0 {
            let target = self.f[j];
            if self.f[j - 1] == target {
                j -= 1;
                continue;
            }
            let i = self.kept[j - 1]
                .iter()
                .map(|kp| kp.s as usize)
                .find(|&i| {
                    self.f[i] + self.v.get(i, j - 1) + au(self.params, self.pair(i, j - 1))
                        == target
                })
                .expect("an F split");
            s[i] = b'(';
            s[j - 1] = b')';
            self.trace_v(i, j - 1, s);
            j = i;
        }
    }

    fn trace_v(&self, i: usize, j: usize, s: &mut [u8]) {
        let target = self.v.get(i, j);
        let pair = self.pair(i, j);
        let hairpin = hairpin_energy(
            self.params,
            &HairpinLoop {
                closing: pair,
                loop_seq: &self.seq[i + 1..j],
            },
        )
        .0;
        if hairpin == target {
            return;
        }
        let l_lower = (i + MIN_HAIRPIN_LOOP + 2).max(j.saturating_sub(MAX_LOOP_SIZE + 1));
        for l in (l_lower..j).rev() {
            let u2 = j - l - 1;
            let last_k = ((i + 1) + MAX_LOOP_SIZE.saturating_sub(u2))
                .min(l.saturating_sub(MIN_HAIRPIN_LOOP + 1));
            for kp in self.kept[l].iter().rev() {
                let k = kp.s as usize;
                if k <= i {
                    continue;
                }
                if k > last_k {
                    break;
                }
                if self.v.get(k, l) + self.two_pair(pair, i, j, k, l) == target {
                    s[k] = b'(';
                    s[l] = b')';
                    self.trace_v(k, l, s);
                    return;
                }
            }
        }
        // A multi-loop: closing + M2[i+1, j-1].
        self.trace_m2(i + 1, j - 1, s);
    }

    fn trace_m2(&self, i: usize, mut j: usize, s: &mut [u8]) {
        loop {
            let target = self.m2.get(i, j);
            if j > i && self.m2.get(i, j - 1) < INF && self.m2.get(i, j - 1) + self.base == target {
                j -= 1;
                continue;
            }
            let split = self.kept[j]
                .iter()
                .map(|kp| kp.s as usize)
                .find(|&x| {
                    x > i
                        && self.m.get(i, x - 1) < INF
                        && self.m.get(i, x - 1) + self.branch(x, j) == target
                })
                .expect("an M2 split");
            s[split] = b'(';
            s[j] = b')';
            self.trace_v(split, j, s);
            self.trace_m(i, split - 1, s);
            return;
        }
    }

    fn trace_m(&self, mut i: usize, mut j: usize, s: &mut [u8]) {
        loop {
            let target = self.m.get(i, j);
            if j > i && self.m.get(i, j - 1) < INF && self.m.get(i, j - 1) + self.base == target {
                j -= 1;
                continue;
            }
            if j > i && self.m.get(i + 1, j) < INF && self.m.get(i + 1, j) + self.base == target {
                i += 1;
                continue;
            }
            if self.v.get(i, j) < INF && self.branch(i, j) == target {
                s[i] = b'(';
                s[j] = b')';
                self.trace_v(i, j, s);
                return;
            }
            let split = self.kept[j]
                .iter()
                .map(|kp| kp.s as usize)
                .find(|&x| {
                    x >= i + 2
                        && self.m.get(i, x - 1) < INF
                        && self.m.get(i, x - 1) + self.branch(x, j) == target
                })
                .expect("an M split");
            s[split] = b'(';
            s[j] = b')';
            self.trace_v(split, j, s);
            j = split - 1;
        }
    }
}

/// Fold `seq` sparsely, optionally under a Carrillo–Lipman `cut`. Without a
/// cut the result is always [`SparseFold::Below`] with the exact MFE.
pub fn fold_sparse(params: &Params, seq: &[Nucleotide], cut: Option<&Cut<'_>>) -> SparseFold {
    let mut fold = Fold::new(params, seq);
    if seq.len() >= MIN_HAIRPIN_LOOP + 2 {
        fold.fill(cut);
    }
    let mfe = fold.f[seq.len()];
    if let Some(c) = cut {
        if mfe >= c.threshold {
            return SparseFold::AtOrAbove;
        }
    }
    let mut s = vec![b'.'; seq.len()];
    if seq.len() >= MIN_HAIRPIN_LOOP + 2 {
        fold.trace_f(seq.len(), &mut s);
    }
    SparseFold::Below(Energy(mfe), String::from_utf8(s).expect("ASCII"))
}

/// [`fold_sparse`], copying the spans `reuse` allows from an earlier fold,
/// and returning this fold's tables for the next one with the number of
/// cells copied. Rows naming tables folded at a lower threshold than this
/// cut's, or of another length, are computed afresh.
pub fn fold_sparse_reusing(
    params: &Params,
    seq: &[Nucleotide],
    cut: Option<&Cut<'_>>,
    reuse: Option<&Reuse<'_>>,
) -> (SparseFold, FoldTables, u64) {
    let threshold = cut.map_or(i32::MAX, |c| c.threshold);
    let reuse = reuse.filter(|r| r.until.len() == seq.len() && r.from.len() == seq.len());
    let usable: Vec<bool> = reuse.map_or(Vec::new(), |r| {
        r.tables
            .iter()
            .map(|t| t.threshold >= threshold && t.len() == seq.len())
            .collect()
    });
    let mut fold = Fold::new(params, seq);
    if seq.len() >= MIN_HAIRPIN_LOOP + 2 {
        fold.fill_reusing(cut, reuse.map(|r| (r, &usable[..])));
    }
    let mfe = fold.f[seq.len()];
    let reused = fold.reused;
    let result = if mfe >= threshold {
        SparseFold::AtOrAbove
    } else {
        let mut s = vec![b'.'; seq.len()];
        if seq.len() >= MIN_HAIRPIN_LOOP + 2 {
            fold.trace_f(seq.len(), &mut s);
        }
        SparseFold::Below(Energy(mfe), String::from_utf8(s).expect("ASCII"))
    };
    (result, fold.into_tables(threshold), reused)
}

/// How many pairs [`fold_sparse`] keeps under `cut`: the work its sums do.
pub fn kept_pairs(params: &Params, seq: &[Nucleotide], cut: Option<&Cut<'_>>) -> u64 {
    cut_work(params, seq, cut).0
}

/// What one [`fold_sparse`] under `cut` does, counted after the fill from
/// the pairs it kept: the loops its time goes to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FoldWork {
    /// Every cell `(i, j)`, `i <= j`: each pays the `M`/`M2` extensions.
    pub cells: u64,
    /// Pairs whose `V` was computed: a hairpin, the interior search and the
    /// multi-loop closing each.
    pub v_cells: u64,
    /// Inner pairs the interior-loop searches visited: one loop energy each.
    pub interior_visits: u64,
    /// Kept pairs the multi-loop splits visited: an add and a compare each.
    pub split_visits: u64,
    /// Pairs kept, and so the exterior-loop splits.
    pub kept: u64,
}

/// [`FoldWork`] of the fold of `seq` under `cut`.
pub fn fold_work(params: &Params, seq: &[Nucleotide], cut: Option<&Cut<'_>>) -> FoldWork {
    let mut fold = Fold::new(params, seq);
    let n = seq.len();
    let mut work = FoldWork {
        cells: (n * (n + 1) / 2) as u64,
        ..FoldWork::default()
    };
    if n < MIN_HAIRPIN_LOOP + 2 {
        return work;
    }
    fold.fill(cut);
    for j in 0..n {
        for kp in &fold.kept[j] {
            work.kept += 1;
            work.split_visits += u64::from(kp.s);
        }
    }
    for i in 0..n {
        for j in (i + MIN_HAIRPIN_LOOP + 1)..n {
            if matches!(fold.pair(i, j), BasePair::NN) {
                continue;
            }
            work.v_cells += 1;
            let l_lower = (i + MIN_HAIRPIN_LOOP + 2).max(j.saturating_sub(MAX_LOOP_SIZE + 1));
            for l in l_lower..j {
                let u2 = j - l - 1;
                let last_k = ((i + 1) + MAX_LOOP_SIZE.saturating_sub(u2))
                    .min(l.saturating_sub(MIN_HAIRPIN_LOOP + 1));
                work.interior_visits += fold.kept[l]
                    .iter()
                    .filter(|kp| (kp.s as usize) > i && (kp.s as usize) <= last_k)
                    .count() as u64;
            }
        }
    }
    work
}

/// The pairs [`fold_sparse`] keeps under `cut`, and those its inside bound
/// dropped before computing their `V`.
pub fn cut_work(params: &Params, seq: &[Nucleotide], cut: Option<&Cut<'_>>) -> (u64, u64) {
    let mut fold = Fold::new(params, seq);
    if seq.len() >= MIN_HAIRPIN_LOOP + 2 {
        fold.fill(cut);
    }
    (fold.kept.iter().map(|c| c.len() as u64).sum(), fold.precut)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{evaluate, fold};

    fn params() -> Params {
        Params::load_default().expect("params")
    }

    fn nts(s: &[u8]) -> Vec<Nucleotide> {
        s.iter()
            .map(|&b| Nucleotide::from_byte(b).expect("ACGU"))
            .collect()
    }

    /// A deterministic pseudo-random sequence of length `n`, GC-rich enough
    /// to fold multi-loops.
    fn random_seq(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                b"GCGCAUGC"[(x % 8) as usize]
            })
            .collect()
    }

    #[test]
    fn without_a_cut_the_sparse_fold_is_the_dense_fold() {
        let p = params();
        for (n, seed) in [(20, 1), (45, 2), (80, 3), (120, 4), (200, 5), (300, 6)] {
            for k in 0..6 {
                let seq = random_seq(n, seed * 100 + k);
                let (_, dense) = fold(&p, &seq).expect("folds");
                let SparseFold::Below(e, db) = fold_sparse(&p, &nts(&seq), None) else {
                    panic!("no cut, so below");
                };
                assert_eq!(e, dense, "n {n} seed {seed} k {k}");
                assert_eq!(
                    evaluate(&p, &seq, &db).expect("evaluates"),
                    e,
                    "its structure scores its MFE"
                );
            }
        }
    }

    #[test]
    fn a_cut_with_no_bound_gives_the_verdict_and_the_mfe_below_it() {
        let p = params();
        let open: &dyn Fn(usize, usize) -> i32 = &|_, _| -INF;
        for k in 0..20 {
            let seq = random_seq(150, 7_000 + k);
            let (_, dense) = fold(&p, &seq).expect("folds");
            for t in [dense.0 - 1, dense.0, dense.0 + 1, dense.0 + 200] {
                let got = fold_sparse(
                    &p,
                    &nts(&seq),
                    Some(&Cut {
                        threshold: t,
                        outside: open,
                        inside: None,
                    }),
                );
                if dense.0 < t {
                    assert!(
                        matches!(got, SparseFold::Below(e, _) if e == dense),
                        "t {t}"
                    );
                } else {
                    assert_eq!(got, SparseFold::AtOrAbove, "t {t}");
                }
            }
        }
    }

    /// An inside bound drops pairs before their `V` is computed and leaves
    /// every verdict and MFE as the cut alone gives them. The bounds are
    /// built from the uncut fold: `V − d` with a pseudo-random `d >= 0` is a
    /// lower bound on `V`, and `MFE − V` on the energy outside a pair, since a
    /// structure holding the pair costs at least the MFE.
    #[test]
    fn an_inside_bound_drops_pairs_early_and_changes_no_verdict() {
        let p = params();
        let mut fired = 0;
        for k in 0..12 {
            let seq = random_seq(140, 9_000 + k);
            let nt = nts(&seq);
            let mut exact = Fold::new(&p, &nt);
            exact.fill(None);
            let mfe = exact.f[nt.len()];
            let v = |i: usize, j: usize| exact.v.get(i, j);
            let slack = |i: usize, j: usize| ((i * 31 + j * 17) % 40) as i32;
            let outside = |i: usize, j: usize| {
                let x = v(i, j);
                if x < INF {
                    mfe - x
                } else {
                    -INF
                }
            };
            let inside = |i: usize, j: usize| {
                let x = v(i, j);
                if x < INF {
                    x - slack(i, j)
                } else {
                    -INF
                }
            };
            for t in [mfe - 30, mfe - 5, mfe, mfe + 1, mfe + 100] {
                let plain = Cut {
                    threshold: t,
                    outside: &outside,
                    inside: None,
                };
                let early = Cut {
                    threshold: t,
                    outside: &outside,
                    inside: Some(&inside),
                };
                assert_eq!(
                    fold_sparse(&p, &nt, Some(&early)),
                    fold_sparse(&p, &nt, Some(&plain)),
                    "k {k} t {t}"
                );
                let (_, dropped) = cut_work(&p, &nt, Some(&early));
                fired += dropped;
            }
        }
        assert!(fired > 0, "the inside bound dropped no pair");
    }

    /// The exact `V` table and MFE of `seq`, from an uncut fill.
    fn exact_v(p: &Params, nt: &[Nucleotide]) -> (Tri, i32) {
        let mut f = Fold::new(p, nt);
        f.fill(None);
        let mfe = f.f[nt.len()];
        (f.v, mfe)
    }

    /// A fold that copies the spans it shares with an earlier fold of a
    /// sequence a few substitutions away gives the cut fold's verdict and
    /// MFE, and an MFE structure. Both folds share one outside bound, valid
    /// for both: the lesser of each design's `MFE − V`.
    #[test]
    fn copying_shared_spans_from_an_earlier_fold_changes_no_verdict() {
        let p = params();
        let mut copied = 0;
        for k in 0..10u64 {
            let a = random_seq(140, 11_000 + k);
            let mut b = a.clone();
            let mut state = 0x2545_f491_4f6c_dd1du64 ^ k;
            for _ in 0..(1 + k % 3) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let pos = (state % b.len() as u64) as usize;
                b[pos] = if b[pos] == b'A' { b'G' } else { b'A' };
            }
            let (na, nb) = (nts(&a), nts(&b));
            let ((va, mfe_a), (vb, mfe_b)) = (exact_v(&p, &na), exact_v(&p, &nb));
            let bound = |v: &Tri, mfe: i32, i: usize, j: usize| {
                let x = v.get(i, j);
                if x < INF {
                    mfe - x
                } else {
                    INF
                }
            };
            let outside = |i: usize, j: usize| {
                let lo = bound(&va, mfe_a, i, j).min(bound(&vb, mfe_b, i, j));
                if lo < INF {
                    lo
                } else {
                    -INF
                }
            };
            // A span is shared when the two sequences agree on it.
            let n = a.len();
            let mut until = vec![0usize; n];
            let mut next_diff = n;
            for i in (0..n).rev() {
                if a[i] != b[i] {
                    next_diff = i;
                }
                until[i] = next_diff;
            }
            for t in [mfe_b - 10, mfe_b, mfe_b + 1, mfe_b + 60] {
                for slack in [0, 25] {
                    let earlier = Cut {
                        threshold: t + slack,
                        outside: &outside,
                        inside: None,
                    };
                    let (_, tables, _) = fold_sparse_reusing(&p, &na, Some(&earlier), None);
                    // Cut below this fold's threshold: never usable.
                    let lower = Cut {
                        threshold: t - 20,
                        outside: &outside,
                        inside: None,
                    };
                    let (_, too_low, _) = fold_sparse_reusing(&p, &na, Some(&lower), None);
                    let cut = Cut {
                        threshold: t,
                        outside: &outside,
                        inside: None,
                    };
                    // Odd rows name the unusable fold, even rows the usable one.
                    let from: Vec<usize> = (0..n).map(|i| i % 2).collect();
                    let refs = [&tables, &too_low];
                    let reuse = Reuse {
                        tables: &refs,
                        until: &until,
                        from: &from,
                    };
                    let (got, _, n_copied) = fold_sparse_reusing(&p, &nb, Some(&cut), Some(&reuse));
                    let want = fold_sparse(&p, &nb, Some(&cut));
                    match (&got, &want) {
                        (SparseFold::Below(e, db), SparseFold::Below(w, _)) => {
                            assert_eq!(e, w, "k {k} t {t} slack {slack}");
                            assert_eq!(
                                evaluate(&p, &b, db).expect("evaluates"),
                                *e,
                                "its structure scores its MFE"
                            );
                        }
                        _ => assert_eq!(got, want, "k {k} t {t} slack {slack}"),
                    }
                    copied += n_copied;
                }
            }
        }
        assert!(copied > 0, "no cell was copied");
    }
}
