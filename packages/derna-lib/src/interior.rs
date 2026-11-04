//! Joint optimization of energy-determining loop interiors.
//!
//! A loop's free energy depends on more than its closing pair: a hairpin's
//! terminal mismatch (and, at sizes 3/4/6, its whole sequence via the
//! Triloop/Tetraloop/Hexaloop tables) and an internal loop's two terminal
//! mismatches all read interior nucleotides. Those nucleotides are codon
//! choices on the DFA, so the loop energy and the CAI of the interior must
//! be chosen *together* — picking the best-CAI interior and scoring its
//! energy after the fact is not jointly optimal and understates the MFE.
//!
//! Each helper enumerates only the interior nucleotides the energy model
//! actually reads, fills the energy-irrelevant remainder with the best-CAI
//! DFA path, and returns the combination minimizing the fixed-λ BRD
//! objective. The scan consumes the `(energy, cai)`; the traceback re-runs
//! the same deterministic choice and consumes the spelled nucleotides, so
//! the interior it spells always matches the one that was scored.

use rna_mfe::loops::{internal_energy, internal_energy_scalar, InternalLoop, InternalLoopScalar};
use rna_mfe::params::{BasePair, Nuc5, Triplet37};
use rna_mfe::{hairpin_energy, hairpin_energy_scalar, Energy, HairpinLoop, Nucleotide, Params};

use crate::dfa::{Dfa, Node};

/// A chosen hairpin interior: its loop free energy, the CAI log-weight of
/// its interior nucleotides, and the nucleotides themselves (5′→3′).
pub(crate) struct Interior {
    /// Loop free energy for the closing pair and this interior.
    pub energy: Energy,
    /// Accumulated CAI log-weight of the interior nucleotides.
    pub cai: f64,
    /// Interior nucleotides, 5′→3′.
    pub seq: Vec<Nucleotide>,
}

/// A chosen internal-loop interior: the spelled nucleotides of each
/// unpaired strand. The traceback consumes these to fill the loop flanks;
/// the energy and CAI it was selected by are recomputed by the scan.
pub(crate) struct SingleInterior {
    /// 5′-side unpaired strand, 5′→3′.
    pub left: Vec<Nucleotide>,
    /// 3′-side unpaired strand, 5′→3′.
    pub right: Vec<Nucleotide>,
}

/// One spelling of an unpaired strand: its CAI log-weight and nucleotides.
type Strand = (f64, Vec<Nucleotide>);

/// Fixed-λ BRD score `λ·MFE − (1−λ)·CAI` (lower is better); CAI is scaled
/// to dekacalories to match the energy unit, exactly as [`crate::state`].
fn brd(lambda: f64, energy: Energy, cai: f64) -> f64 {
    lambda * f64::from(energy.0) - (1.0 - lambda) * cai * 100.0
}

fn lower_brd(a: f64, b: f64) -> std::cmp::Ordering {
    a.partial_cmp(&b).expect("finite BRD scores")
}

/// The two closing pairs of an internal loop, in sequence order.
pub(crate) struct LoopPairs {
    /// The enclosing pair `(i, j)`.
    pub outer: BasePair,
    /// The enclosed pair `(k, l)`.
    pub inner: BasePair,
}

/// One unpaired strand of an internal loop as scored corners plus its
/// length — the energy model needs the length for its size dispatch.
pub(crate) struct ScoredStrand<'a> {
    /// Terminal-mismatch corners of the strand (see [`corner_scores`]).
    pub corners: &'a [Corner],
    /// Number of unpaired nucleotides in the strand.
    pub len: usize,
}

/// A node span `from → to` along the DFA: an unpaired strand's endpoints.
pub(crate) struct Span {
    /// The 5′ boundary node (immediately inside the loop's 5′ pair base).
    pub from: Node,
    /// The 3′ boundary node (the loop's 3′ pair base).
    pub to: Node,
}

/// The scan's cheap counterpart to [`best_hairpin_interior`]: the chosen
/// interior's `(energy, cai)` *without* spelling its nucleotides. The
/// forward scan stores only the score; the full interior is respelled
/// once, during traceback, by [`best_hairpin_interior`]. Both walk the
/// same candidates in the same order under the same BRD comparison, so
/// they agree on the winner — but this one skips the per-candidate
/// `O(span)` interior reconstruction that dominates the scan.
///
/// At the special-hairpin sizes (3/4/6) the whole loop selects a table
/// bonus, so the interior is still enumerated (cheap: the spans are ≤ 6
/// nt with bounded fan-out). At other sizes only the two terminal
/// mismatches reach the energy model, scored by [`hairpin_energy_scalar`]
/// over the corner candidates. Returns `None` when no interior path
/// connects the boundaries.
pub(crate) fn score_hairpin(
    params: &Params,
    dfa: &Dfa,
    lambda: f64,
    start: Node,
    end: Node,
    closing: BasePair,
) -> Option<(Energy, f64)> {
    let size = (end.pos - start.pos) as usize;
    if matches!(size, 3 | 4 | 6) {
        return dfa
            .paths_between(start, end)
            .into_iter()
            .map(|(cai, seq)| {
                (
                    hairpin_energy(
                        params,
                        &HairpinLoop {
                            closing,
                            loop_seq: &seq,
                        },
                    ),
                    cai,
                )
            })
            .min_by(|a, b| lower_brd(brd(lambda, a.0, a.1), brd(lambda, b.0, b.1)));
    }
    corner_scores(dfa, start, end)
        .into_iter()
        .map(|c| {
            (
                hairpin_energy_scalar(params, closing, size, c.first, c.last),
                c.cai,
            )
        })
        .min_by(|a, b| lower_brd(brd(lambda, a.0, a.1), brd(lambda, b.0, b.1)))
}

/// Per closing pair, whether its row of the `1×n` mismatch table is one value
/// over every pair of mismatched nucleotides, so that a `1×n` loop (`n ≥ 3`)
/// closed by it scores the same whatever its four terminal nucleotides.
/// Turner 2004's rows are: the table carries only a closure penalty per pair.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OneByNRows([bool; 7]);

impl OneByNRows {
    /// The rows of `params`' `1×n` mismatch table.
    pub(crate) fn of(params: &Params) -> Self {
        let table = &params.mismatch_interior_1n;
        let nucs = [Nuc5::A, Nuc5::C, Nuc5::G, Nuc5::U];
        Self(BasePair::ALL.map(|pair| {
            let first = table.get(pair, Nuc5::A, Nuc5::A);
            nucs.iter()
                .all(|&a| nucs.iter().all(|&b| table.get(pair, a, b) == first))
        }))
    }

    /// Whether every corner choice of an `(n1, n2)` loop between `pairs`
    /// scores the same energy.
    pub(crate) fn corner_free(&self, pairs: &LoopPairs, n1: usize, n2: usize) -> bool {
        matches!((n1, n2), (1, n) | (n, 1) if n >= 3)
            && self.0[pairs.outer.as_index()]
            && self.0[pairs.inner.reversed().as_index()]
    }
}

/// The scan's cheap counterpart to [`best_internal_interior`]: the chosen
/// internal-loop interior's `(energy, cai)` without spelling its strands.
/// Reads the four terminal mismatches off the strands' precomputed corner
/// candidates and scores them with [`internal_energy_scalar`], skipping
/// strand reconstruction. The traceback respells the winner via
/// [`best_internal_interior`]; both pick the same candidate.
///
/// A loop whose energy is the same at every corner choice
/// ([`OneByNRows::corner_free`]) is scored once and its corners compared by
/// CAI alone, in the same order and by the same arithmetic.
pub(crate) fn score_internal_from(
    params: &Params,
    lambda: f64,
    pairs: &LoopPairs,
    (left, right): (&ScoredStrand, &ScoredStrand),
    rows: &OneByNRows,
) -> Option<(Energy, f64)> {
    let energy_at = |l: &Corner, r: &Corner| {
        internal_energy_scalar(
            params,
            &InternalLoopScalar {
                outer: pairs.outer,
                inner: pairs.inner,
                boundary_left5p: l.first,
                boundary_left3p: l.last,
                boundary_right5p: r.first,
                boundary_right3p: r.last,
                n1: left.len,
                n2: right.len,
            },
        )
    };
    if rows.corner_free(pairs, left.len, right.len) {
        let energy = energy_at(left.corners.first()?, right.corners.first()?);
        return best_corners(lambda, (left.corners, right.corners), |_, _| energy);
    }
    if let Some(table) = mismatch_table(params, left.len, right.len) {
        // The loop's energy is a term of its shape plus its two terminal
        // mismatches: read the first off one corner pair, then add each
        // pair's mismatches — the energy model's own integer sum.
        let inner_rev = pairs.inner.reversed();
        let mismatches = |l: &Corner, r: &Corner| {
            table.get(pairs.outer, l.first.to_nuc5(), r.last.to_nuc5())
                + table.get(inner_rev, r.first.to_nuc5(), l.last.to_nuc5())
        };
        let (l0, r0) = (left.corners.first()?, right.corners.first()?);
        let shape = energy_at(l0, r0).0 - mismatches(l0, r0);
        return best_corners(lambda, (left.corners, right.corners), |l, r| {
            Energy(shape + mismatches(l, r))
        });
    }
    best_corners(lambda, (left.corners, right.corners), energy_at)
}

/// The mismatch table of an `(n1, n2)` internal loop whose energy is a term of
/// its shape alone plus its two terminal mismatches from that table, as
/// [`internal_energy_scalar`] dispatches: `1×n` (`n ≥ 3`), `2×3`, and the
/// generic branch (both strands at least two long, not a tabulated small
/// shape). `None` for the shapes scored from a whole-loop table.
fn mismatch_table(params: &Params, n1: usize, n2: usize) -> Option<&Triplet37> {
    match (n1, n2) {
        (1, n) | (n, 1) if n >= 3 => Some(&params.mismatch_interior_1n),
        (2, 3) | (3, 2) => Some(&params.mismatch_interior_23),
        (2, 2) => None,
        _ if n1 >= 2 && n2 >= 2 => Some(&params.mismatch_interior),
        _ => None,
    }
}

/// The least-BRD corner pair of two strands under `energy_at`, first on ties.
fn best_corners(
    lambda: f64,
    (lefts, rights): (&[Corner], &[Corner]),
    energy_at: impl Fn(&Corner, &Corner) -> Energy,
) -> Option<(Energy, f64)> {
    let mut best: Option<(f64, Energy, f64)> = None;
    for l in lefts {
        for r in rights {
            let energy = energy_at(l, r);
            let cai = l.cai + r.cai;
            let score = brd(lambda, energy, cai);
            if best.as_ref().is_none_or(|&(b, _, _)| score < b) {
                best = Some((score, energy, cai));
            }
        }
    }
    best.map(|(_, energy, cai)| (energy, cai))
}

/// Best interior of a hairpin closed by `(start − 1, end)` with closing
/// pair `closing`, where `start` is the node immediately inside the 5′ pair base
/// and `end` is the node of the 3′ pair base. The loop spans
/// `end.pos − start.pos` unpaired nucleotides.
///
/// At the special-hairpin sizes (3/4/6) the entire interior selects a
/// table bonus, so every interior spelling is enumerated. At other sizes
/// only the two terminal-mismatch nucleotides reach the energy model, so
/// they are enumerated while the energy-irrelevant middle is filled by the
/// best-CAI DFA path. Returns `None` when no interior path connects the
/// boundaries (an infeasible codon span).
pub(crate) fn best_hairpin_interior(
    params: &Params,
    dfa: &Dfa,
    lambda: f64,
    start: Node,
    end: Node,
    closing: BasePair,
) -> Option<Interior> {
    let strands = if matches!((end.pos - start.pos) as usize, 3 | 4 | 6) {
        dfa.paths_between(start, end)
    } else {
        corner_candidates(dfa, start, end)
    };
    strands
        .into_iter()
        .map(|(cai, seq)| {
            let energy = hairpin_energy(
                params,
                &HairpinLoop {
                    closing,
                    loop_seq: &seq,
                },
            );
            Interior { energy, cai, seq }
        })
        .min_by(|a, b| lower_brd(brd(lambda, a.energy, a.cai), brd(lambda, b.energy, b.cai)))
}

/// Best interior of an internal loop with the given closing pairs and
/// unpaired strands between them: the 5′ strand runs `left_from → left_to`
/// and the 3′ strand runs `right_from → right_to`. Both strands must be
/// non-empty (a stack or bulge reads no interior nucleotides and is scored
/// directly). The internal-loop energy reads only the four terminal
/// nucleotides, so those are enumerated and the middles filled by best-CAI.
pub(crate) fn best_internal_interior(
    params: &Params,
    dfa: &Dfa,
    lambda: f64,
    pairs: &LoopPairs,
    left: Span,
    right: Span,
) -> Option<SingleInterior> {
    let lefts = corner_candidates(dfa, left.from, left.to);
    let rights = corner_candidates(dfa, right.from, right.to);
    let mut best: Option<(f64, SingleInterior)> = None;
    for (lcai, lseq) in &lefts {
        for (rcai, rseq) in &rights {
            let energy = internal_energy(
                params,
                &InternalLoop {
                    outer: pairs.outer,
                    inner: pairs.inner,
                    left: lseq,
                    right: rseq,
                },
            );
            let score = brd(lambda, energy, lcai + rcai);
            if best.as_ref().is_none_or(|(b, _)| score < *b) {
                best = Some((
                    score,
                    SingleInterior {
                        left: lseq.clone(),
                        right: rseq.clone(),
                    },
                ));
            }
        }
    }
    best.map(|(_, si)| si)
}

/// The scan-side corner of an unpaired strand: its two terminal-mismatch
/// nucleotides (the only interior the energy model reads) and the total
/// CAI log-weight of the strand, including the best-CAI middle. The
/// energy-irrelevant middle is *not* spelled — that work is deferred to
/// [`corner_candidates`] during traceback.
#[derive(Clone, Copy)]
pub(crate) struct Corner {
    /// Total CAI log-weight of the strand (terminals plus best-CAI middle).
    pub cai: f64,
    /// First (5′) nucleotide of the strand.
    pub first: Nucleotide,
    /// Last (3′) nucleotide of the strand.
    pub last: Nucleotide,
}

/// Cheap counterpart to [`corner_candidates`]: the terminal mismatches
/// and CAI of every strand spelling `from → to`, without materializing
/// the middle. Enumerates the same candidates in the same order as
/// [`corner_candidates`], so the two agree on which corner a downstream
/// `min` selects. The middle's CAI is the `O(1)` [`Dfa::broken_codon_cai`]
/// lookup — the `O(span)` [`Dfa::broken_codon_seq`] is never called.
///
/// Precondition: `from.pos < to.pos` (a strand of length ≥ 1); the
/// hairpin and internal-loop callers never score an empty strand.
pub(crate) fn corner_scores(dfa: &Dfa, from: Node, to: Node) -> Vec<Corner> {
    let mut out = Vec::new();
    corner_scores_into(dfa, from, to, &mut out);
    out
}

/// [`corner_scores`] writing into a caller-owned buffer instead of a fresh
/// `Vec`. The per-candidate corner enumeration is on the hottest path (the
/// single-branch closure runs it ~10⁷–10⁸ times per scan), so the closures
/// reuse one scratch buffer across candidates rather than allocating each
/// time. `out` is cleared first; its prior capacity is retained.
pub(crate) fn corner_scores_into(dfa: &Dfa, from: Node, to: Node, out: &mut Vec<Corner>) {
    out.clear();
    let len = (to.pos - from.pos) as usize;
    if len == 1 {
        out.extend(
            dfa.right(from)
                .iter()
                .filter(|e| e.to == to)
                .map(|e| Corner {
                    cai: e.cai,
                    first: e.nuc,
                    last: e.nuc,
                }),
        );
        return;
    }
    for first in dfa.right(from) {
        for last in dfa.left(to) {
            let Some(mid_cai) = dfa.broken_codon_cai(first.to, last.to) else {
                continue;
            };
            out.push(Corner {
                cai: first.cai + mid_cai + last.cai,
                first: first.nuc,
                last: last.nuc,
            });
        }
    }
}

/// Every spelling of the unpaired strand `from → to` that pins its first
/// and last nucleotide (the terminal mismatches the energy model reads),
/// filling the energy-irrelevant middle with the best-CAI DFA path. A
/// length-1 strand has a single nucleotide that is both terminals. Returns
/// no candidates when the boundaries are not connectable.
pub(crate) fn corner_candidates(dfa: &Dfa, from: Node, to: Node) -> Vec<Strand> {
    let len = (to.pos - from.pos) as usize;
    if len == 0 {
        return vec![(0.0, Vec::new())];
    }
    if len == 1 {
        return dfa
            .right(from)
            .iter()
            .filter(|e| e.to == to)
            .map(|e| (e.cai, vec![e.nuc]))
            .collect();
    }
    let mut out = Vec::new();
    for first in dfa.right(from) {
        for last in dfa.left(to) {
            let Some(mid_cai) = dfa.broken_codon_cai(first.to, last.to) else {
                continue;
            };
            let mut seq = Vec::with_capacity(len);
            seq.push(first.nuc);
            seq.extend(dfa.broken_codon_seq(first.to, last.to));
            seq.push(last.nuc);
            out.push((first.cai + mid_cai + last.cai, seq));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    //! Scan ↔ traceback scoring equivalence.
    //!
    //! The forward scan scores a loop interior through the cheap corner
    //! path ([`score_hairpin`] / [`score_internal_from`]); the traceback
    //! respells it through the full materializing path
    //! ([`best_hairpin_interior`] / [`corner_candidates`] +
    //! [`internal_energy`]). The two must agree on the chosen `(energy,
    //! cai)` for every interior, or the recovered structure would carry a
    //! different energy than the scan stored. These tests lock that
    //! equivalence directly, independent of the end-to-end parity suite.

    use super::*;
    use crate::dfa::build;
    use derna_codon_dp::codon_table::aa_to_index;
    use derna_codon_dp::CodonUsage;

    /// The six canonical (Watson–Crick + wobble) closing pairs.
    const CANONICAL: [BasePair; 6] = [
        BasePair::AU,
        BasePair::UA,
        BasePair::CG,
        BasePair::GC,
        BasePair::GU,
        BasePair::UG,
    ];

    /// The λ values swept by both equivalence checks (low / mid / high).
    const LAMBDAS: [f64; 3] = [0.0001, 0.5, 0.99];

    /// A varied 20-residue protein whose DFA carries both single- and
    /// two-lane codon units, exercising the cross-lane corner enumeration.
    fn test_dfa() -> Dfa {
        let usage = CodonUsage::load_default().unwrap();
        let aa: Vec<usize> = "MARNDCQEGHILKMFPSTWYV"
            .chars()
            .map(|c| aa_to_index(c).unwrap())
            .collect();
        build(&aa, &usage)
    }

    /// CAI log-weights are summed in the same order on both paths, so they
    /// must agree to within floating-point noise.
    fn cai_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// Reference internal-loop score via the full materializing path: the
    /// same enumeration [`best_internal_interior`] runs, but returning the
    /// winner's `(energy, cai)` for comparison rather than its strands.
    fn full_internal(
        params: &Params,
        dfa: &Dfa,
        lambda: f64,
        pairs: &LoopPairs,
        left: Span,
        right: Span,
    ) -> Option<(Energy, f64)> {
        let lefts = corner_candidates(dfa, left.from, left.to);
        let rights = corner_candidates(dfa, right.from, right.to);
        let mut best: Option<(f64, Energy, f64)> = None;
        for (lcai, lseq) in &lefts {
            for (rcai, rseq) in &rights {
                let energy = internal_energy(
                    params,
                    &InternalLoop {
                        outer: pairs.outer,
                        inner: pairs.inner,
                        left: lseq,
                        right: rseq,
                    },
                );
                let cai = lcai + rcai;
                let score = brd(lambda, energy, cai);
                if best.as_ref().is_none_or(|&(b, _, _)| score < b) {
                    best = Some((score, energy, cai));
                }
            }
        }
        best.map(|(_, energy, cai)| (energy, cai))
    }

    /// Every `(from, to)` node span of the given lengths starting at the
    /// given positions, across all lanes present at each endpoint.
    fn spans(dfa: &Dfa, starts: &[u32], lens: &[u32]) -> Vec<Span> {
        let mut out = Vec::new();
        for &p in starts {
            for &len in lens {
                let q = p + len;
                if q >= dfa.seq_len {
                    continue;
                }
                for &from in dfa.nodes_at(p) {
                    for &to in dfa.nodes_at(q) {
                        out.push(Span { from, to });
                    }
                }
            }
        }
        out
    }

    #[test]
    fn score_hairpin_matches_full_path() {
        let params = Params::load_default().unwrap();
        let dfa = test_dfa();
        // Special-hairpin sizes (3/4/6) and generic sizes (5/7/9).
        for span in spans(
            &dfa,
            &(1..dfa.seq_len).collect::<Vec<_>>(),
            &[3, 4, 5, 6, 7, 9],
        ) {
            for &closing in &CANONICAL {
                for &lambda in &LAMBDAS {
                    let scan = score_hairpin(&params, &dfa, lambda, span.from, span.to, closing);
                    let full =
                        best_hairpin_interior(&params, &dfa, lambda, span.from, span.to, closing)
                            .map(|i| (i.energy, i.cai));
                    match (scan, full) {
                        (None, None) => {}
                        (Some((es, cs)), Some((ef, cf))) => {
                            assert_eq!(
                                es, ef,
                                "hairpin energy mismatch: span={:?}→{:?} closing={closing:?} λ={lambda}",
                                span.from, span.to,
                            );
                            assert!(
                                cai_eq(cs, cf),
                                "hairpin cai mismatch: scan={cs} full={cf} closing={closing:?} λ={lambda}",
                            );
                        }
                        _ => panic!(
                            "hairpin Some/None mismatch: span={:?}→{:?} closing={closing:?} λ={lambda}",
                            span.from, span.to,
                        ),
                    }
                }
            }
        }
    }

    #[test]
    fn score_internal_matches_full_path() {
        let params = Params::load_default().unwrap();
        let dfa = test_dfa();
        // Left and right strands of lengths 1..=3 spanning codon
        // boundaries and codon interiors, giving every (n1, n2) shape the
        // energy model dispatches on.
        let lefts = spans(&dfa, &[1, 4], &[1, 2, 3]);
        let rights = spans(&dfa, &[8, 11], &[1, 2, 3]);
        for &outer in &CANONICAL {
            for &inner in &CANONICAL {
                let pairs = LoopPairs { outer, inner };
                for left in &lefts {
                    for right in &rights {
                        for &lambda in &LAMBDAS {
                            let lc = corner_scores(&dfa, left.from, left.to);
                            let rc = corner_scores(&dfa, right.from, right.to);
                            let scan = score_internal_from(
                                &params,
                                lambda,
                                &pairs,
                                (
                                    &ScoredStrand {
                                        corners: &lc,
                                        len: (left.to.pos - left.from.pos) as usize,
                                    },
                                    &ScoredStrand {
                                        corners: &rc,
                                        len: (right.to.pos - right.from.pos) as usize,
                                    },
                                ),
                                &OneByNRows::of(&params),
                            );
                            let full = full_internal(
                                &params,
                                &dfa,
                                lambda,
                                &pairs,
                                Span {
                                    from: left.from,
                                    to: left.to,
                                },
                                Span {
                                    from: right.from,
                                    to: right.to,
                                },
                            );
                            match (scan, full) {
                                (None, None) => {}
                                (Some((es, cs)), Some((ef, cf))) => {
                                    assert_eq!(
                                        es, ef,
                                        "internal energy mismatch: outer={outer:?} inner={inner:?} λ={lambda}",
                                    );
                                    assert!(
                                        cai_eq(cs, cf),
                                        "internal cai mismatch: scan={cs} full={cf} λ={lambda}",
                                    );
                                }
                                _ => panic!(
                                    "internal Some/None mismatch: outer={outer:?} inner={inner:?}"
                                ),
                            }
                        }
                    }
                }
            }
        }
    }
}
