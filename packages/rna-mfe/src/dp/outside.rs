//! Outside values of the MFE tables: for every pair `(i, j)`, the least
//! free energy of a whole-sequence structure that contains it.
//!
//! The fill computes `V[i, j]`, the best energy *inside* a closed pair. The
//! complementary quantity `V̂[i, j]` is the best energy of everything *around*
//! it — the exterior loop and every enclosing loop — given that `(i, j)` is
//! closed. Their sum is the MFE of the sequence with `(i, j)` forced, for every
//! pair at once, from one inside and one outside pass: the cost of a fold per
//! pair becomes the cost of about one fold.
//!
//! # Recurrences
//!
//! Each fill rule `head = Σ tails + w` contributes `outside(head) + w + Σ
//! other tails` to the outside value of each of its tails; a cell's outside is
//! the least contribution over every rule that consumes it. At `dangles = 0`
//! the consumers are:
//!
//! ```text
//! V̂[k, l]  ← F[k] + ext(k, l) + F3[l + 1]                    exterior stem
//!          ← V̂[i, j] + leaf((i, j), (k, l))                  stack / bulge / internal
//!          ← M̂1[k, l] + stem(k, l)                           M1 = branch
//!          ← M̂[k, l] + stem(k, l)                            M  = branch
//! M̂[i, j]  ← M̂[i − 1, j] + base | M̂[i, j + 1] + base         unpaired flanks
//!          ← M̂[i, q] + M1[j + 1, q]                          M[i, q] = M[i, j] + M1[j + 1, q]
//!          ← V̂[i − 1, q] + close(i − 1, q) + M1[j + 1, q − 1] V multi-loop closing
//! M̂1[i, j] ← M̂1[i, j + 1] + base                             trailing unpaired
//!          ← M̂[p, j] + M[p, i − 1]                           M[p, j] = M[p, i − 1] + M1[i, j]
//!          ← V̂[p, j + 1] + close(p, j + 1) + M[p + 1, i − 1] V multi-loop closing
//! ```
//!
//! `F3[l]` is the suffix analogue of the fill's prefix table `F`. Every rule's
//! tails have a span no longer than its head, and the only equal-span rules
//! read `V` into `M` and `M1`, so cells are visited by decreasing span with
//! `M̂` and `M̂1` before `V̂` within a cell.
//!
//! The index ranges, the minimum-hairpin and `MAX_LOOP_SIZE` caps and the
//! minimum multi-loop span are the fill's own, so `V[i, j] + V̂[i, j]` equals a
//! constrained fold that requires `(i, j)`, not a bound on it.

use super::fill::{ext_stem_penalty, score_two_pair, FillCtx, OuterClose, Span};
use super::multi::{branch_stem_cost, ml_base_cost, multi_closing_stem_cost, MIN_V_MULTI_SPAN};
use super::tables::{DpTables, TriangleTable, MAX_LOOP_SIZE, MIN_HAIRPIN_LOOP};
use crate::loops::{pair_from_nucleotides, Energy, Nucleotide};
use crate::params::{BasePair, Params};

const INF: i32 = TriangleTable::INF;

/// A dense `n × n` table of raw dekacalorie values, `INF` where empty.
struct Dense {
    n: usize,
    cells: Vec<i32>,
}

impl Dense {
    fn new(n: usize) -> Self {
        Self {
            n,
            cells: vec![INF; n * n],
        }
    }

    fn from_triangle(t: &TriangleTable, n: usize) -> Self {
        let mut d = Self::new(n);
        for i in 0..n {
            for j in i..n {
                d.cells[i * n + j] = t.get(i, j).map_or(INF, |e| e.0);
            }
        }
        d
    }

    #[inline]
    fn at(&self, i: usize, j: usize) -> i32 {
        self.cells[i * self.n + j]
    }

    #[inline]
    fn set(&mut self, i: usize, j: usize, v: i32) {
        self.cells[i * self.n + j] = v;
    }

    /// The transpose, so a column of `self` is a contiguous row.
    fn transposed(&self) -> Self {
        let n = self.n;
        let mut t = Self::new(n);
        for i in 0..n {
            for j in 0..n {
                t.cells[j * n + i] = self.cells[i * n + j];
            }
        }
        t
    }
}

#[inline]
fn add(a: i32, b: i32) -> i32 {
    if a >= INF || b >= INF {
        INF
    } else {
        a + b
    }
}

/// `min over k: a[k] + b[k]`, `INF` when no term is finite.
#[inline]
fn min_sum(a: &[i32], b: &[i32]) -> i32 {
    a.iter()
        .zip(b)
        .fold(INF, |best, (&x, &y)| best.min(add(x, y)))
}

/// The forced-pair energies of one sequence: `V[i, j] + V̂[i, j]` for every
/// pair, and the free MFE they are measured against.
pub struct ForcedPairs {
    n: usize,
    forced: Vec<i32>,
    /// The free MFE, `F[n]`.
    pub mfe: Energy,
}

impl ForcedPairs {
    /// The least free energy of a structure containing pair `(i, j)`, or
    /// `None` when no structure the grammar admits contains it.
    pub fn get(&self, i: usize, j: usize) -> Option<Energy> {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        let v = self.forced[i * self.n + j];
        (v < INF).then_some(Energy(v))
    }
}

/// Every pair's forced MFE from the filled tables of `seq`.
pub fn forced_pair_energies(params: &Params, seq: &[Nucleotide], tables: &DpTables) -> ForcedPairs {
    let n = seq.len();
    let mfe = tables.f[n].unwrap_or(Energy::ZERO);
    let v = Dense::from_triangle(&tables.v, n);
    let m = Dense::from_triangle(&tables.m, n);
    let m1 = Dense::from_triangle(&tables.m1, n);
    let m_t = m.transposed();
    let pair_at = |i: usize, j: usize| pair_from_nucleotides(seq[i], seq[j]);
    let stem = |i: usize, j: usize| match pair_at(i, j) {
        BasePair::NN => INF,
        p => branch_stem_cost(params, p).0,
    };

    let f3 = suffix_exterior(params, seq, &v);
    let mut vo = Dense::new(n);
    // `V̂ + close` at each multi-loop-capable pair, row- and column-major.
    let mut vc = Dense::new(n);
    let mut vc_t = Dense::new(n);
    let mut mo = Dense::new(n);
    let mut mo_t = Dense::new(n);
    let mut m1o = Dense::new(n);
    let ctx = FillCtx::new(params, seq, tables);

    for span in (0..n).rev() {
        for i in 0..n - span {
            let j = i + span;
            // M̂[i, j]
            let mut best = INF;
            if i >= 1 {
                best = best.min(add(mo.at(i - 1, j), ml_base_cost(params).0));
            }
            if j + 1 < n {
                best = best.min(add(mo.at(i, j + 1), ml_base_cost(params).0));
                // M[i, q] = M[i, j] + M1[j + 1, q], j in [i + 1, q − 1].
                if j > i {
                    let row = &mo.cells[i * n + j + 1..i * n + n];
                    let m1_row = &m1.cells[(j + 1) * n + j + 1..(j + 1) * n + n];
                    best = best.min(min_sum(row, m1_row));
                }
                // V(i − 1, q) closes M[i, j] + M1[j + 1, q − 1], q ≥ j + 2.
                if i >= 1 && j + 2 < n {
                    let p = i - 1;
                    let row = &vc.cells[p * n + j + 2..p * n + n];
                    let m1_row = &m1.cells[(j + 1) * n + j + 1..(j + 1) * n + n - 1];
                    best = best.min(min_sum(row, m1_row));
                }
            }
            mo.set(i, j, best);
            mo_t.set(j, i, best);

            // M̂1[i, j]
            let mut best = INF;
            if j + 1 < n {
                best = best.min(add(m1o.at(i, j + 1), ml_base_cost(params).0));
            }
            if i >= 2 {
                // M[p, j] = M[p, i − 1] + M1[i, j], p ≤ i − 2.
                let col = &mo_t.cells[j * n..j * n + i - 1];
                let m_col = &m_t.cells[(i - 1) * n..(i - 1) * n + i - 1];
                best = best.min(min_sum(col, m_col));
                // V(p, j + 1) closes M[p + 1, i − 1] + M1[i, j], p ≤ i − 2.
                if j + 1 < n {
                    let col = &vc_t.cells[(j + 1) * n..(j + 1) * n + i - 1];
                    let m_col = &m_t.cells[(i - 1) * n + 1..(i - 1) * n + i];
                    best = best.min(min_sum(col, m_col));
                }
            }
            m1o.set(i, j, best);

            // V̂[i, j]
            if v.at(i, j) >= INF {
                continue;
            }
            let pair = pair_at(i, j);
            let mut best = add(
                tables.f[i].map_or(INF, |e| e.0),
                add(ext_stem_penalty(params, pair).0, f3[j + 1]),
            );
            let st = stem(i, j);
            best = best.min(add(m1o.at(i, j), st));
            best = best.min(add(mo.at(i, j), st));
            best = best.min(enclosing_min(&ctx, &vo, (i, j), pair));
            vo.set(i, j, best);
            if span >= MIN_V_MULTI_SPAN {
                let c = add(best, multi_closing_stem_cost(params, pair).0);
                vc.set(i, j, c);
                vc_t.set(j, i, c);
            }
        }
    }

    let mut forced = vec![INF; n * n];
    for i in 0..n {
        for j in i..n {
            forced[i * n + j] = add(v.at(i, j), vo.at(i, j));
        }
    }
    ForcedPairs { n, forced, mfe }
}

/// `F3[l]`: the MFE of the suffix `seq[l..]`, with `F3[n] = 0`.
fn suffix_exterior(params: &Params, seq: &[Nucleotide], v: &Dense) -> Vec<i32> {
    let n = seq.len();
    let mut f3 = vec![0; n + 1];
    for i in (0..n).rev() {
        let mut best = f3[i + 1];
        for j in i + 1..n {
            let vij = v.at(i, j);
            if vij >= INF {
                continue;
            }
            let pair = pair_from_nucleotides(seq[i], seq[j]);
            best = best.min(add(vij + ext_stem_penalty(params, pair).0, f3[j + 1]));
        }
        f3[i] = best;
    }
    f3
}

/// `min over enclosing (p, q): V̂[p, q] + leaf((p, q), (i, j))` — the pair as
/// the inner pair of a stack, bulge or internal loop, under the fill's bounds.
fn enclosing_min(ctx: &FillCtx<'_>, vo: &Dense, (k, l): (usize, usize), inner: BasePair) -> i32 {
    let n = ctx.seq.len();
    let mut best = INF;
    for p in (0..k).rev() {
        let u1 = k - p - 1;
        if u1 > MAX_LOOP_SIZE {
            break;
        }
        for q in l + 1..n {
            let u2 = q - l - 1;
            if u1 + u2 > MAX_LOOP_SIZE {
                break;
            }
            let out = vo.at(p, q);
            if out >= INF || q - p < MIN_HAIRPIN_LOOP + 1 {
                continue;
            }
            let outer_pair = pair_from_nucleotides(ctx.seq[p], ctx.seq[q]);
            if matches!(outer_pair, BasePair::NN) {
                continue;
            }
            let outer = OuterClose {
                span: Span { i: p, j: q },
                pair: outer_pair,
            };
            let leaf = score_two_pair(ctx, outer, inner, Span { i: k, j: l });
            best = best.min(add(out, leaf.0));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dp::fill_dp_tables;

    fn encode(s: &str) -> Vec<Nucleotide> {
        s.bytes()
            .map(|b| Nucleotide::from_byte(b).unwrap())
            .collect()
    }

    /// Every pair's forced energy equals the least `evaluate` over all
    /// structures containing it, by brute force over a short sequence.
    #[test]
    fn forced_pairs_match_brute_force() {
        let params = Params::load_default().unwrap();
        // None of these folds through a multi-loop at its optimum, and a length
        // that does is past brute force; the multi-loop consumers are reached by
        // the identity test below.
        for s in [
            "GGGAAAUCCCAGCGAAAGCUGG",
            "GCGCUUCGGCGCAUGGAAACCAUG",
            "GGCGCGAGCGAAAGCAGCGAAAGCAGCGCC",
        ] {
            let seq = encode(s);
            let tables = fill_dp_tables(&params, &seq);
            let forced = forced_pair_energies(&params, &seq, &tables);
            let mut best = vec![vec![INF; seq.len()]; seq.len()];
            for structure in all_structures(&seq) {
                let Ok(e) = crate::evaluate(&params, s.as_bytes(), &structure) else {
                    continue;
                };
                for (i, j) in pairs(&structure) {
                    best[i][j] = best[i][j].min(e.0);
                }
            }
            for (i, row) in best.iter().enumerate() {
                for (j, &want) in row.iter().enumerate().skip(i + 1) {
                    let got = forced.get(i, j).map_or(INF, |e| e.0);
                    assert_eq!(got, want, "{s} pair ({i}, {j})");
                }
            }
        }
    }

    /// On a sequence whose MFE structure holds a multi-loop, every pair of that
    /// structure is forced at exactly the MFE and no pair is forced below it —
    /// which reaches the multi-loop consumers that a brute-forceable length
    /// cannot.
    #[test]
    fn mfe_pairs_are_forced_at_the_mfe_through_a_multiloop() {
        let params = Params::load_default().unwrap();
        // A three-way junction: an eight-pair outer stem around two UUCG hairpins.
        let s = "GAGGCCGCAGCGCUUCGGCGCAAGGACUUCGGUCCAGCGGCCUC";
        let seq = encode(s);
        let tables = fill_dp_tables(&params, &seq);
        let forced = forced_pair_energies(&params, &seq, &tables);
        let structure = crate::dp::traceback(&params, &seq, &tables);
        let mfe_pairs = pairs(&structure);
        let branches_of = |(i, j): (usize, usize)| {
            mfe_pairs
                .iter()
                .filter(|&&(k, l)| i < k && l < j)
                .filter(|&&(k, l)| {
                    !mfe_pairs
                        .iter()
                        .any(|&(a, b)| i < a && a < k && l < b && b < j)
                })
                .count()
        };
        assert!(
            mfe_pairs.iter().any(|&p| branches_of(p) >= 2),
            "{structure} has no multi-loop"
        );
        for &(i, j) in &mfe_pairs {
            assert_eq!(
                forced.get(i, j),
                Some(forced.mfe),
                "MFE pair ({i}, {j}) of {structure}"
            );
        }
        for i in 0..seq.len() {
            for j in i + 1..seq.len() {
                if let Some(e) = forced.get(i, j) {
                    assert!(e >= forced.mfe, "pair ({i}, {j}) forced below the MFE");
                }
            }
        }
    }

    fn pairs(structure: &str) -> Vec<(usize, usize)> {
        let mut stack = Vec::new();
        let mut out = Vec::new();
        for (k, c) in structure.bytes().enumerate() {
            match c {
                b'(' => stack.push(k),
                b')' => out.push((stack.pop().unwrap(), k)),
                _ => (),
            }
        }
        out
    }

    /// Every secondary structure of `seq` with canonical pairs and hairpins of
    /// at least three unpaired bases.
    fn all_structures(seq: &[Nucleotide]) -> Vec<String> {
        fn go(
            seq: &[Nucleotide],
            i: usize,
            j: usize,
            memo: &mut Vec<Vec<Option<Vec<String>>>>,
        ) -> Vec<String> {
            if i > j || j == usize::MAX {
                return vec![String::new()];
            }
            if let Some(v) = &memo[i][j] {
                return v.clone();
            }
            let mut out: Vec<String> = go(seq, i + 1, j, memo)
                .into_iter()
                .map(|s| format!(".{s}"))
                .collect();
            for k in i + MIN_HAIRPIN_LOOP + 1..=j {
                if matches!(pair_from_nucleotides(seq[i], seq[k]), BasePair::NN) {
                    continue;
                }
                let inner = go(seq, i + 1, k - 1, memo);
                let rest = if k == j {
                    vec![String::new()]
                } else {
                    go(seq, k + 1, j, memo)
                };
                for a in &inner {
                    for b in &rest {
                        out.push(format!("({a}){b}"));
                    }
                }
            }
            memo[i][j] = Some(out.clone());
            out
        }
        let n = seq.len();
        let mut memo = vec![vec![None; n + 1]; n + 1];
        go(seq, 0, n - 1, &mut memo)
    }
}
