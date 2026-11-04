//! The modularity check's fold under a Carrillo–Lipman cut.
//!
//! A design is rejected when its free MFE lies below its constrained MFE
//! `E_c`, which the walk knows before the fold starts. A pair `(i, j)` matters
//! to that verdict only if `V[i, j]` plus the energy of everything outside it
//! can stay below `E_c`. The outside bound here is shared by every design of
//! one lattice: the exact cost to go ([`crate::cost_to_go::Outside`]) of the
//! *unconstrained, λ = 1* design lattice, read at the cell the design's path
//! passes through — the best rest-of-molecule over every design through that
//! cell, so never above the design's own. One extra exact scan and outside
//! pass per lattice, then [`rna_mfe::dp::fold_sparse`] per design.
//!
//! The same scan's closed-pair cells hold the best `V` any design of the
//! lattice has at each pair, so never above the design's own. Kept
//! ([`FoldCut::build`] with `inside`), they let the fold drop a pair before
//! computing its `V` ([`rna_mfe::dp::Cut::inside`]), at the memory of the
//! scan's tables.

use std::sync::Arc;

use rna_mfe::dp::{
    cut_work, fold_sparse, fold_sparse_reusing, fold_work, Cut, FoldTables, FoldWork, Reuse,
    SparseFold,
};
use rna_mfe::{Nucleotide, Params};

use crate::cost_to_go::{fold_with_cost_to_go, Outside};
use crate::dfa::{Dfa, Node};
use crate::region::RegionConstraints;
use crate::state::PairKey;
use crate::tables::Tables;
use crate::{Beam, BeamError, CodonUsage, ScanConfig};

/// The lattice and its cost to go, built once for every design's check.
pub struct FoldCut {
    dfa: Dfa,
    outside: Outside,
    /// The scan's tables, kept for their closed-pair values when the fold
    /// also bounds each pair's `V` from below.
    inside: Option<Tables>,
}

/// The lattice node before each position of `seq` and the end node, along
/// the design's path; `None` when the lattice does not spell `seq`.
///
/// A lane cannot be chosen greedily — see [`Dfa::path_of`], which this is a
/// byte-slice front for.
pub(crate) fn path(dfa: &Dfa, seq: &[u8]) -> Option<Vec<Node>> {
    let nts: Vec<Nucleotide> = seq
        .iter()
        .map(|&b| Nucleotide::from_byte(b))
        .collect::<Option<_>>()?;
    dfa.path_of(&nts)
}

/// No bound: a value the cut never reaches its threshold with.
const UNBOUNDED: i32 = i32::MIN / 8;
/// An inside bound that shuts a pair whatever its outside bound
/// ([`FoldCut::fold_restricted`]).
const SHUT: i32 = i32::MAX / 2;

impl FoldCut {
    /// A cut from a lattice and its cost to go, as [`Self::build`] makes them,
    /// without an inside bound.
    pub fn from_parts(dfa: Dfa, outside: Outside) -> Self {
        Self {
            dfa,
            outside,
            inside: None,
        }
    }

    /// The unconstrained λ = 1 lattice of the walk's design space, folded exactly
    /// on `config.threads` threads, with its cost to go; with `inside`, its
    /// tables too, so each fold can bound a pair's `V` before computing it.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        params: &Params,
        protein: &str,
        usage: &CodonUsage,
        allowed: &[Vec<[u8; 3]>],
        utr5: &str,
        utr3: &str,
        config: ScanConfig,
        inside: bool,
    ) -> Result<Option<Self>, BeamError> {
        let exact = ScanConfig {
            beam: Beam::exact(),
            ..config
        };
        Ok(fold_with_cost_to_go(
            params,
            protein,
            usage,
            allowed,
            utr5,
            utr3,
            RegionConstraints::default(),
            exact,
        )?
        .map(|(dfa, tables, outside)| Self {
            dfa,
            outside,
            inside: inside.then_some(tables),
        }))
    }

    /// Fold `seq` (bytes of `ACGU`) under the cut at `threshold` dekacalories:
    /// [`SparseFold::Below`] with the free MFE and a structure when it lies
    /// below, else [`SparseFold::AtOrAbove`]. `None` when the lattice does
    /// not spell `seq`.
    pub fn fold(&self, params: &Params, seq: &[u8], threshold: i32) -> Option<SparseFold> {
        self.with_cut(seq, threshold, true, |nt, _, cut| {
            fold_sparse(params, nt, Some(cut))
        })
    }

    /// [`Self::fold`] over only the pairs `allowed` admits: the least free
    /// structure built from them, when below `threshold`. Every such structure
    /// is a free structure of `seq`, and the cut's outside bound holds for
    /// it, so the pruning stays exact over the restricted set.
    pub fn fold_restricted(
        &self,
        params: &Params,
        seq: &[u8],
        threshold: i32,
        allowed: &dyn Fn(usize, usize) -> bool,
    ) -> Option<SparseFold> {
        self.with_cut(seq, threshold, false, |nt, _, cut| {
            let inside = |i: usize, j: usize| if allowed(i, j) { UNBOUNDED } else { SHUT };
            let restricted = Cut {
                threshold,
                outside: cut.outside,
                inside: Some(&inside),
            };
            fold_sparse(params, nt, Some(&restricted))
        })
    }

    /// [`Self::fold`], copying from earlier folds in `earlier` the spans they
    /// share with `seq` (see [`CachedFold`]): from the one sharing the most,
    /// or with `per_row` each row from whichever reaches furthest. Returns
    /// the verdict, this fold for later ones, and the cells copied. An
    /// earlier fold cut at a lower threshold is not used.
    pub fn fold_reusing(
        &self,
        params: &Params,
        seq: &[u8],
        threshold: i32,
        earlier: &[Arc<CachedFold>],
        per_row: bool,
    ) -> Option<(SparseFold, CachedFold, u64)> {
        self.with_cut(seq, threshold, true, |nt, nodes, cut| {
            let usable: Vec<&CachedFold> = earlier
                .iter()
                .map(|c| &**c)
                .filter(|c| {
                    c.marks.is_empty()
                        && c.tables.threshold() >= threshold
                        && c.seq.len() == seq.len()
                })
                .collect();
            let reach: Vec<Vec<usize>> = usable
                .iter()
                .map(|c| shared_spans(&c.seq, &c.nodes, seq, nodes))
                .collect();
            let (until, from) = choose_rows(&reach, seq.len(), per_row);
            let tables: Vec<&FoldTables> = usable.iter().map(|c| &c.tables).collect();
            let reuse = (!tables.is_empty()).then(|| Reuse {
                tables: &tables,
                until: &until,
                from: &from,
            });
            let (verdict, tables, copied) =
                fold_sparse_reusing(params, nt, Some(cut), reuse.as_ref());
            let cached = CachedFold {
                seq: seq.to_vec(),
                nodes: nodes.to_vec(),
                marks: Vec::new(),
                tables,
            };
            (verdict, cached, copied)
        })
    }

    /// [`Self::fold_restricted`], copying from earlier restricted folds in
    /// `earlier` the spans they share with `seq`: spelled alike, on the same
    /// lattice nodes, and with the same `marks` at every position. `marks`
    /// must stand for what `allowed` depends on beyond a predicate common to
    /// every fold that shares this cache — two folds whose marks agree on a
    /// span must admit the same pairs inside it (a windowed fold: each
    /// position's partner in the design's own structure). Returns the
    /// verdict, this fold for later ones, and the cells copied.
    #[allow(clippy::too_many_arguments)]
    pub fn fold_restricted_reusing(
        &self,
        params: &Params,
        seq: &[u8],
        threshold: i32,
        allowed: &dyn Fn(usize, usize) -> bool,
        marks: &[u32],
        earlier: &[Arc<CachedFold>],
        per_row: bool,
    ) -> Option<(SparseFold, CachedFold, u64)> {
        self.with_cut(seq, threshold, false, |nt, nodes, cut| {
            let inside = |i: usize, j: usize| if allowed(i, j) { UNBOUNDED } else { SHUT };
            let restricted = Cut {
                threshold,
                outside: cut.outside,
                inside: Some(&inside),
            };
            let usable: Vec<&CachedFold> = earlier
                .iter()
                .map(|c| &**c)
                .filter(|c| {
                    c.marks.len() == seq.len()
                        && c.tables.threshold() >= threshold
                        && c.seq.len() == seq.len()
                })
                .collect();
            let reach: Vec<Vec<usize>> = usable
                .iter()
                .map(|c| shared_spans_marked(c, seq, nodes, marks))
                .collect();
            let (until, from) = choose_rows(&reach, seq.len(), per_row);
            let tables: Vec<&FoldTables> = usable.iter().map(|c| &c.tables).collect();
            let reuse = (!tables.is_empty()).then(|| Reuse {
                tables: &tables,
                until: &until,
                from: &from,
            });
            let (verdict, tables, copied) =
                fold_sparse_reusing(params, nt, Some(&restricted), reuse.as_ref());
            let cached = CachedFold {
                seq: seq.to_vec(),
                nodes: nodes.to_vec(),
                marks: marks.to_vec(),
                tables,
            };
            (verdict, cached, copied)
        })
    }

    /// What [`Self::fold`] does for `seq`, with or without the inside bound:
    /// the pairs it keeps, and those the inside bound drops before their `V`
    /// is computed ([`rna_mfe::dp::cut_work`]).
    pub fn work(
        &self,
        params: &Params,
        seq: &[u8],
        threshold: i32,
        inside: bool,
    ) -> Option<(u64, u64)> {
        self.with_cut(seq, threshold, inside, |nt, _, cut| {
            cut_work(params, nt, Some(cut))
        })
    }

    /// The cut's outside bound for every pair `(i, j)`, `i <= j`, of `seq`,
    /// row-major by `i` (the triangle [`rna_mfe::dp`] stores): what a fold
    /// of `seq` under this cut reads, for replaying it without the lattice.
    pub fn outside_table(&self, seq: &[u8]) -> Option<Vec<i32>> {
        self.with_cut(seq, 0, false, |nt, _, cut| {
            let n = nt.len();
            let mut out = Vec::with_capacity(n * (n + 1) / 2);
            for i in 0..n {
                for j in i..n {
                    out.push((cut.outside)(i, j));
                }
            }
            out
        })
    }

    /// Where [`Self::fold`]'s time goes for `seq` at `threshold`
    /// ([`rna_mfe::dp::fold_work`]).
    pub fn fold_work(&self, params: &Params, seq: &[u8], threshold: i32) -> Option<FoldWork> {
        self.with_cut(seq, threshold, true, |nt, _, cut| {
            fold_work(params, nt, Some(cut))
        })
    }

    /// Run `f` on `seq`'s nucleotides and lattice path under the cut at
    /// `threshold`, with the inside bound when `inside` and the tables were
    /// kept.
    fn with_cut<R>(
        &self,
        seq: &[u8],
        threshold: i32,
        inside: bool,
        f: impl FnOnce(&[Nucleotide], &[Node], &Cut<'_>) -> R,
    ) -> Option<R> {
        let nodes = path(&self.dfa, seq)?;
        let nt: Vec<Nucleotide> = seq
            .iter()
            .map(|&b| Nucleotide::from_byte(b))
            .collect::<Option<_>>()?;
        let key = |i: usize, j: usize| PairKey {
            open: nodes[i],
            close: nodes[j],
            opening: crate::state::Opening::new(nt[i], nodes[i + 1].lane),
            nuc_close: nt[j],
        };
        let outside = |i: usize, j: usize| {
            self.outside
                .pair_at(nodes[j + 1], key(i, j))
                .map_or(UNBOUNDED, |e| e.0)
        };
        let tables = self.inside.as_ref().filter(|_| inside);
        let lower = |i: usize, j: usize| {
            tables
                .and_then(|t| t.p_at(nodes[j + 1], key(i, j)))
                .map_or(UNBOUNDED, |s| s.mfe.0)
        };
        let cut = Cut {
            threshold,
            outside: &outside,
            inside: tables.map(|_| &lower as &dyn Fn(usize, usize) -> i32),
        };
        Some(f(&nt, &nodes, &cut))
    }
}

/// A finished fold under the cut, kept so that later folds of similar
/// designs can copy the spans they share with it.
///
/// A cell `[i, j]` holds the same value in two folds when both spell the
/// same nucleotides on `i..=j` (a loop's energy reads only nucleotides
/// inside its closing pair) and every pair inside it meets the same cut
/// test. The outside bound of a pair `(i', j')` is read at the lattice nodes
/// `i'`, `j'` and `j' + 1`, and a node can depend on nucleotides before the
/// span (mid-codon, it holds the codon so far), so the nodes must agree on
/// `i..=j + 1` as well. The threshold is checked by
/// [`rna_mfe::dp::fold_sparse_reusing`].
pub struct CachedFold {
    seq: Vec<u8>,
    nodes: Vec<Node>,
    /// Empty for a fold over every pair; for a restricted fold, what its
    /// pair restriction depends on, per position
    /// ([`FoldCut::fold_restricted_reusing`]).
    marks: Vec<u32>,
    tables: FoldTables,
}

/// Which earlier fold each row copies from, and how far: the one fold whose
/// rows reach furthest in all, or with `per_row` each row's furthest.
/// `reach[r][i]` is fold `r`'s [`shared_spans`].
fn choose_rows(reach: &[Vec<usize>], n: usize, per_row: bool) -> (Vec<usize>, Vec<usize>) {
    let (mut until, mut from) = (vec![0; n], vec![0; n]);
    if per_row {
        for (r, row) in reach.iter().enumerate() {
            for (i, &u) in row.iter().enumerate() {
                if u > until[i] {
                    until[i] = u;
                    from[i] = r;
                }
            }
        }
    } else {
        let score = |row: &Vec<usize>| -> usize {
            row.iter()
                .enumerate()
                .map(|(i, &u)| u.saturating_sub(i))
                .sum()
        };
        if let Some((r, row)) = reach.iter().enumerate().max_by_key(|(_, row)| score(row)) {
            until.clone_from(row);
            from.fill(r);
        }
    }
    (until, from)
}

/// For each `i`, the `until[i]` such that spans `[i, j]` with `j < until[i]`
/// have the same nucleotides in `a` and `b` and the same lattice nodes on
/// `i..=j + 1` ([`CachedFold`]).
fn shared_spans(a: &[u8], a_nodes: &[Node], b: &[u8], b_nodes: &[Node]) -> Vec<usize> {
    let n = b.len();
    debug_assert!(a.len() == n && a_nodes.len() == n + 1 && b_nodes.len() == n + 1);
    let mut until = vec![0; n];
    let (mut next_seq, mut next_node) = (n, n + 1);
    if a_nodes[n] != b_nodes[n] {
        next_node = n;
    }
    for i in (0..n).rev() {
        if a[i] != b[i] {
            next_seq = i;
        }
        if a_nodes[i] != b_nodes[i] {
            next_node = i;
        }
        until[i] = next_seq.min(next_node.saturating_sub(1));
    }
    until
}

/// [`shared_spans`] that also ends a span at a position whose mark differs.
fn shared_spans_marked(a: &CachedFold, b: &[u8], b_nodes: &[Node], b_marks: &[u32]) -> Vec<usize> {
    let mut until = shared_spans(&a.seq, &a.nodes, b, b_nodes);
    let mut next_mark = b.len();
    for i in (0..b.len()).rev() {
        if a.marks[i] != b_marks[i] {
            next_mark = i;
        }
        until[i] = until[i].min(next_mark);
    }
    until
}

#[cfg(test)]
mod tests {
    use super::*;
    use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};

    /// On a real lattice the inside bound drops pairs before their `V` is
    /// computed and leaves every verdict as the cut alone gives it. Under
    /// debug assertions every pair it keeps is also checked against its `V`.
    #[test]
    fn the_lattice_inside_bound_drops_pairs_and_changes_no_verdict() {
        let params = Params::load_default().expect("params");
        let usage = CodonUsage::load_default().expect("usage");
        let protein = "MLRSK";
        let allowed: Vec<Vec<[u8; 3]>> = protein
            .chars()
            .map(|c| {
                AMINO_ACIDS[aa_to_index(c).expect("a residue")]
                    .codons
                    .iter()
                    .map(|c| c.nuc)
                    .collect()
            })
            .collect();
        let (utr5, utr3) = ("GGGAUCCGAA", "AAGGAUCCC");
        let cut = FoldCut::build(
            &params,
            protein,
            &usage,
            &allowed,
            utr5,
            utr3,
            ScanConfig::default(),
            true,
        )
        .expect("builds")
        .expect("a lattice");
        let (mut dropped, mut checked) = (0, 0);
        // Every encoding of the middle three residues, the ends fixed.
        for a in &allowed[1] {
            for b in &allowed[2] {
                for c in &allowed[3] {
                    let mut seq = utr5.as_bytes().to_vec();
                    for codon in [&allowed[0][0], a, b, c, &allowed[4][0]] {
                        seq.extend(codon.iter().map(|&x| b"ACGU"[x as usize]));
                    }
                    seq.extend_from_slice(utr3.as_bytes());
                    let nt: Vec<Nucleotide> = seq
                        .iter()
                        .map(|&x| Nucleotide::from_byte(x).expect("ACGU"))
                        .collect();
                    let SparseFold::Below(mfe, _) = fold_sparse(&params, &nt, None) else {
                        unreachable!("no cut, so below")
                    };
                    for t in [mfe.0 - 150, mfe.0 - 20, mfe.0, mfe.0 + 1, mfe.0 + 300] {
                        let with = cut.fold(&params, &seq, t).expect("on the lattice");
                        let without = cut
                            .with_cut(&seq, t, false, |nt, _, c| fold_sparse(&params, nt, Some(c)))
                            .expect("on the lattice");
                        assert_eq!(with, without, "t {t}");
                        dropped += cut.work(&params, &seq, t, true).expect("on the lattice").1;
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 100, "{checked} folds checked");
        assert!(dropped > 0, "the inside bound dropped no pair");
    }

    /// Folds that copy spans from the best of the last eight folds of
    /// neighbouring encodings give the cut fold's verdict and MFE, and an
    /// MFE structure, and copy cells. The thresholds cycle, so the earlier
    /// fold's is sometimes lower and must not be used.
    #[test]
    fn folds_reusing_earlier_folds_change_no_verdict() {
        let params = Params::load_default().expect("params");
        let usage = CodonUsage::load_default().expect("usage");
        let protein = "MLRSK";
        let allowed: Vec<Vec<[u8; 3]>> = protein
            .chars()
            .map(|c| {
                AMINO_ACIDS[aa_to_index(c).expect("a residue")]
                    .codons
                    .iter()
                    .map(|c| c.nuc)
                    .collect()
            })
            .collect();
        let (utr5, utr3) = ("GGGAUCCGAA", "AAGGAUCCC");
        let cut = FoldCut::build(
            &params,
            protein,
            &usage,
            &allowed,
            utr5,
            utr3,
            ScanConfig::default(),
            false,
        )
        .expect("builds")
        .expect("a lattice");
        let mut recent: Vec<Arc<CachedFold>> = Vec::new();
        let (mut copied, mut folds) = ([0u64; 2], 0);
        for a in &allowed[1] {
            for b in &allowed[2] {
                for c in &allowed[3] {
                    let mut seq = utr5.as_bytes().to_vec();
                    for codon in [&allowed[0][0], a, b, c, &allowed[4][0]] {
                        seq.extend(codon.iter().map(|&x| b"ACGU"[x as usize]));
                    }
                    seq.extend_from_slice(utr3.as_bytes());
                    let nt: Vec<Nucleotide> = seq
                        .iter()
                        .map(|&x| Nucleotide::from_byte(x).expect("ACGU"))
                        .collect();
                    let SparseFold::Below(mfe, _) = fold_sparse(&params, &nt, None) else {
                        unreachable!("no cut, so below")
                    };
                    let t = mfe.0 + [-40, 0, 1, 120][folds % 4];
                    let want = cut.fold(&params, &seq, t).expect("on the lattice");
                    let mut keep = None;
                    for per_row in [false, true] {
                        let (got, cached, n) = cut
                            .fold_reusing(&params, &seq, t, &recent, per_row)
                            .expect("on the lattice");
                        match (&got, &want) {
                            (SparseFold::Below(e, db), SparseFold::Below(w, _)) => {
                                assert_eq!(e, w, "fold {folds} t {t} per row {per_row}");
                                assert_eq!(
                                    rna_mfe::evaluate(&params, &seq, db).expect("evaluates"),
                                    *e,
                                    "its structure scores its MFE"
                                );
                            }
                            _ => assert_eq!(got, want, "fold {folds} t {t} per row {per_row}"),
                        }
                        copied[usize::from(per_row)] += n;
                        keep = Some(cached);
                    }
                    folds += 1;
                    recent.insert(0, Arc::new(keep.expect("folded")));
                    recent.truncate(8);
                }
            }
        }
        assert!(copied[0] > 0, "no cell was copied from one fold");
        assert!(
            copied[1] >= copied[0],
            "rows from their furthest fold copy no fewer cells than one fold: {copied:?}"
        );
    }

    /// A windowed fold over the flanks' pairs and a design's own pairs, each
    /// design's own structure its free optimum: copying from earlier windowed
    /// folds where both designs spell alike and their own structures agree
    /// gives every fresh restricted fold's verdict and energy.
    #[test]
    fn restricted_folds_reusing_earlier_ones_change_no_verdict() {
        let params = Params::load_default().expect("params");
        let usage = CodonUsage::load_default().expect("usage");
        let protein = "MLRSK";
        let allowed: Vec<Vec<[u8; 3]>> = protein
            .chars()
            .map(|c| {
                AMINO_ACIDS[aa_to_index(c).expect("a residue")]
                    .codons
                    .iter()
                    .map(|c| c.nuc)
                    .collect()
            })
            .collect();
        let (utr5, utr3) = ("GGGAUCCGAA", "AAGGAUCCC");
        let cut = FoldCut::build(
            &params,
            protein,
            &usage,
            &allowed,
            utr5,
            utr3,
            ScanConfig::default(),
            false,
        )
        .expect("builds")
        .expect("a lattice");
        // One cache per window predicate, as in a run: odd folds also admit
        // every pair with an end in a UTR (`--window-utr`).
        let mut caches: [Vec<Arc<CachedFold>>; 2] = [Vec::new(), Vec::new()];
        let (mut copied, mut folds, mut below) = ([0u64; 2], 0usize, 0usize);
        for a in &allowed[1] {
            for b in &allowed[2] {
                for c in &allowed[3] {
                    let mut seq = utr5.as_bytes().to_vec();
                    for codon in [&allowed[0][0], a, b, c, &allowed[4][0]] {
                        seq.extend(codon.iter().map(|&x| b"ACGU"[x as usize]));
                    }
                    seq.extend_from_slice(utr3.as_bytes());
                    let n = seq.len();
                    let nt: Vec<Nucleotide> = seq
                        .iter()
                        .map(|&x| Nucleotide::from_byte(x).expect("ACGU"))
                        .collect();
                    let SparseFold::Below(mfe, own) = fold_sparse(&params, &nt, None) else {
                        unreachable!("no cut, so below")
                    };
                    let mut mate = vec![u32::MAX; n];
                    for (i, j) in crate::witness::pairs_of(&own) {
                        mate[i as usize] = j;
                        mate[j as usize] = i;
                    }
                    let near = |i: usize| i < utr5.len() + 3 || i >= n - utr3.len() - 3;
                    let utr = folds % 2 == 1;
                    let in_utr = |i: usize| i < utr5.len() || i >= n - utr3.len();
                    let window = |i: usize, j: usize| {
                        (near(i) && near(j))
                            || mate[i] as usize == j
                            || (utr && (in_utr(i) || in_utr(j)))
                    };
                    let t = mfe.0 + [-40, 0, 1, 120][folds % 4];
                    let want = cut
                        .fold_restricted(&params, &seq, t, &window)
                        .expect("on the lattice");
                    let (got, cached, k) = cut
                        .fold_restricted_reusing(
                            &params,
                            &seq,
                            t,
                            &window,
                            &mate,
                            &caches[usize::from(utr)],
                            true,
                        )
                        .expect("on the lattice");
                    match (&got, &want) {
                        (SparseFold::Below(e, db), SparseFold::Below(w, _)) => {
                            assert_eq!(e, w, "fold {folds} t {t}");
                            assert_eq!(
                                rna_mfe::evaluate(&params, &seq, db).expect("evaluates"),
                                *e,
                                "its structure scores its energy"
                            );
                            below += 1;
                        }
                        _ => assert_eq!(got, want, "fold {folds} t {t}"),
                    }
                    copied[usize::from(utr)] += k;
                    folds += 1;
                    let cache = &mut caches[usize::from(utr)];
                    cache.insert(0, Arc::new(cached));
                    cache.truncate(8);
                }
            }
        }
        assert!(
            copied[0] > 0 && copied[1] > 0,
            "no cell was copied: {copied:?}"
        );
        assert!(
            below > 0,
            "no windowed fold found a structure below its threshold"
        );
    }
}
