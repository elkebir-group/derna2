//! Counterexample-guided constraint generation over the design space the
//! bounded search partitions.
//!
//! Both this and [`crate::search`] look for a design whose *own* free fold
//! satisfies a region table, and both restrict the space with the same
//! primitive — a codon excluded at a protein position. What differs is what
//! they do with a counterexample, and that difference is the whole comparison.
//!
//! **The search partitions.** When a region's optimum folds dirty it is
//! replaced by parts that tile everything except that one solution, so nothing
//! is ever discarded and a bound can prove the walk finished. The answer is the
//! exact optimum over modular designs.
//!
//! **This cuts.** A dirty fold names the CDS positions spelling the pairs it
//! reinstated; the codon standing at one of them is excluded outright, and the
//! solve is repeated over what is left. Every sequence using that codon at that
//! position goes with it, the optimum included if it happened to be there. The
//! result is a design that is modular or nothing, never a design that is proven
//! best.
//!
//! **It terminates quickly, and that is the same fact as its weakness.** Each
//! round removes at least one codon from at least one position and never
//! restores it, so the loop ends within the total number of synonymous codons —
//! `O(m)` rounds against a partition tree that is exponential in the worst
//! case. What buys the bound is exactly what forfeits optimality.
//!
//! **A cut does not always exist, for two different reasons.** A reinstated
//! pair contributes a cuttable position only where an endpoint lies inside the
//! CDS; a pair with both endpoints in given flank sequence cannot be spelled
//! away by any codon assignment, and the loop reports [`CegarOutcome::NoCut`].
//! That is a property of the arrangement, and the partitioning search meets the
//! same wall as an empty modular set.
//!
//! The second reason is a limit of this loop rather than of the problem. Cuts
//! are read off the *recovered* free structure, and free optima are routinely
//! co-optimal, so a design can be violating by energy while the traceback hands
//! back one of the clean structures at that energy. Then no pair is named,
//! nothing can be blamed, and the loop reports [`CegarOutcome::Blind`]. The
//! energy verdict is still right; it is the localization that fails. The
//! partitioning search does not share this failure, because it branches on
//! positions rather than on blame — the violation only orders its branching,
//! and an empty violation list leaves the order arbitrary rather than the walk
//! stuck.

use std::time::Instant;

use derna_codon_dp::codon_table::AMINO_ACIDS;

use crate::ac::MotifWindow;
use crate::free_fold::{sequence_gap, CdsWindow};
use crate::search::{SearchError, SearchInputs};
use crate::{design_restricted_with_region, design_with_placed_motifs, Design};

/// How many counterexamples one round converts into cuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CutRule {
    /// Cut the codon at one violating position per round — the least the
    /// counterexample licenses, and the most conservative about what it
    /// discards along with it.
    #[default]
    Single,
    /// Cut the codon at every CDS position spelling a reinstated pair. Fewer
    /// rounds, and correspondingly more of the space thrown away per round.
    All,
    /// Cut the **junction** at one violating position: forbid the sequence the
    /// design spells over the shortest window that contains that position and
    /// **crosses a codon boundary**, at that window and nowhere else.
    ///
    /// The point is what it does *not* remove. A codon cut removes one synonym
    /// at one residue outright; a window crossing a boundary removes only the
    /// *combinations* of adjacent codons that spell that junction, so every
    /// codon involved stays available in every other context. That is the only
    /// reading under which a positional motif cut discards strictly less than
    /// a codon cut — two others were considered and both fail:
    ///
    /// - The design's 3-mer over the violating codon, at that codon's
    ///   positions, removes exactly that codon there. It **is** the codon cut,
    ///   spelled differently.
    /// - The single nucleotide at the violating position removes every synonym
    ///   matching that letter at that offset, and the design's own codon is one
    ///   of them — so it is a strict **superset** of the codon cut. Forbidding
    ///   `C` at a proline codon's first position removes all four of
    ///   `CCU/CCC/CCA/CCG` where the codon cut removes one.
    ///
    /// Positional in the sense the codon cut is: the cut holds at the place the
    /// counterexample occurred and nowhere else, which is what
    /// [`crate::ac::Ac::build_placed`] exists to express.
    Junction,
}

/// How the loop ended, and therefore what its answer is worth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CegarOutcome {
    /// A design whose free fold satisfies the table. Modular, and carrying no
    /// claim of optimality: earlier cuts may have removed better ones.
    Modular,
    /// A counterexample with no cuttable endpoint — every reinstated pair lies
    /// wholly inside given sequence. No codon assignment reaches it.
    NoCut,
    /// The design is violating by energy and the traceback named no forbidden
    /// pair, so there is nothing to cut *on*. Free optima are routinely
    /// co-optimal: the recovered structure can be one of the clean ones at the
    /// same energy while a violating structure exists at that energy too. The
    /// loop cannot localize a counterexample it cannot see, which is a limit of
    /// reading cuts off a traceback rather than a fact about the molecule.
    Blind,
    /// The cuts emptied some position's codon set. Says nothing about whether
    /// a modular design exists: the cuts are not a partition, so what they
    /// removed was never examined.
    Exhausted,
    /// The budget ran out while still cutting — wall-clock or the round cap,
    /// which say the same thing about the answer: the loop had not finished, so
    /// this is not evidence about whether a modular design exists.
    Budget,
}

impl CegarOutcome {
    /// Whether the loop ended holding a design that satisfies the table.
    pub fn is_modular(self) -> bool {
        self == CegarOutcome::Modular
    }
}

/// What one loop found and what it cost, in the units the bounded search
/// reports so the two can be read side by side.
#[derive(Debug, Clone)]
pub struct CegarReport {
    pub outcome: CegarOutcome,
    /// The modular design, when there is one. `None` on every other outcome —
    /// the last dirty design is not an answer to the question asked.
    pub design: Option<Design>,
    /// Rounds of solve-and-verify, each of which cost one design and one fold.
    pub rounds: usize,
    /// Codons excluded over the whole loop. Exceeds `rounds` under
    /// [`CutRule::All`].
    pub cuts: usize,
    /// Design solves. Equal to `rounds`: this loop has no lazy bound to refine,
    /// so a round is a solve.
    pub solves: usize,
    pub secs: f64,
    /// The counterexample the loop stopped on, when it stopped on one. Zeroed
    /// on a modular finish. `named` is how many pairs the verdict listed,
    /// `in_cds` how many of those had an endpoint a codon spells, and
    /// `cuttable` how many of *those* still had a codon left to exclude — so
    /// `named > 0, in_cds == 0` is the arrangement refusing, and
    /// `in_cds > 0, cuttable == 0` is the loop having already spent them.
    pub stuck: StuckDetail,
}

/// Why a round produced no cut, in the three counts that separate the reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StuckDetail {
    pub named: usize,
    pub in_cds: usize,
    pub cuttable: usize,
}

/// Run the cut loop until it is modular, stuck, or out of budget.
///
/// The objective is minimized over what the cuts have left, not over the modular
/// designs — so a returned design is modular but not comparable to
/// [`crate::search::bounded_search`]'s incumbent as an optimum. Comparing the
/// two objectives is the point of running both, and the gap is what the cuts
/// cost.
pub fn cegar_search(inputs: &SearchInputs<'_>, rule: CutRule) -> Result<CegarReport, SearchError> {
    cegar_search_capped(inputs, rule, usize::MAX)
}

/// [`cegar_search`] with a cap on **rounds** as well as on wall-clock.
///
/// A round is one design and one fold, so a caller pacing the loop by work
/// rather than by seconds caps rounds. That is what a chained remedy does: the
/// reason to try this loop before the exact search is that it is cheap, and
/// "cheap" is a count of solves rather than a number of seconds, which varies
/// with the instance and the machine.
///
/// Hitting the cap is [`CegarOutcome::Budget`] — the same outcome as running out
/// of wall-clock, and for the same reason: the loop was still cutting and says
/// nothing about whether a modular design exists.
pub fn cegar_search_capped(
    inputs: &SearchInputs<'_>,
    rule: CutRule,
    max_rounds: usize,
) -> Result<CegarReport, SearchError> {
    let started = Instant::now();
    let cds = CdsWindow::new(inputs.utr5.len(), inputs.protein.len());
    let mut cuts = Cuts::new(inputs, rule);
    let mut report = CegarReport {
        outcome: CegarOutcome::Budget,
        design: None,
        rounds: 0,
        cuts: 0,
        solves: 0,
        secs: 0.0,
        stuck: StuckDetail::default(),
    };

    loop {
        if started.elapsed() >= inputs.budget || report.rounds >= max_rounds {
            report.outcome = CegarOutcome::Budget;
            break;
        }
        report.rounds += 1;
        report.solves += 1;
        let Some(design) = cuts.solve(inputs)? else {
            report.outcome = CegarOutcome::Exhausted;
            break;
        };
        let gap = sequence_gap(inputs.params, &design.sequence, cds, inputs.constraints)?;
        if gap.is_modular() {
            report.outcome = CegarOutcome::Modular;
            report.design = Some(design);
            break;
        }
        report.stuck = StuckDetail::default();
        if gap.reinstated_pairs.is_empty() {
            // Violating by energy, but the traceback showed a clean co-optimal
            // structure. There is no pair to blame, so no cut follows.
            report.outcome = CegarOutcome::Blind;
            break;
        }
        let codons = cds_codons(inputs, &design);
        let targets = match rule {
            CutRule::Junction => cuts.junction_targets(inputs, &gap.reinstated_pairs, &design, cds),
            _ => cuttable(
                inputs,
                &gap.reinstated_pairs,
                cuts.excluded(),
                &codons,
                cds,
                rule,
            )
            .into_iter()
            .map(Target::Codon)
            .collect(),
        };
        if targets.is_empty() {
            report.stuck = StuckDetail {
                named: gap.reinstated_pairs.len(),
                in_cds: gap
                    .reinstated_pairs
                    .iter()
                    .flat_map(|&(i, j)| [i, j])
                    .filter(|&p| p >= cds.start && p < cds.end)
                    .count(),
                cuttable: 0,
            };
            report.outcome = CegarOutcome::NoCut;
            break;
        }
        for target in targets {
            cuts.add(target, &codons);
            report.cuts += 1;
        }
    }

    report.secs = started.elapsed().as_secs_f64();
    Ok(report)
}

/// The optimum over what the cuts have left, or `None` when some position has
/// no codon remaining.
fn solve(
    inputs: &SearchInputs<'_>,
    excluded: &[Vec<[u8; 3]>],
) -> Result<Option<Design>, SearchError> {
    let allowed: Vec<Vec<[u8; 3]>> = inputs
        .aa
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            AMINO_ACIDS[a]
                .codons
                .iter()
                .map(|c| c.nuc)
                .filter(|c| !excluded[i].contains(c))
                .collect()
        })
        .collect();
    Ok(design_restricted_with_region(
        inputs.params,
        inputs.protein,
        inputs.usage,
        inputs.lambda,
        &allowed,
        inputs.utr5,
        inputs.utr3,
        inputs.constraints,
        inputs.config,
    )?)
}

/// What the loop has cut so far, in whichever vocabulary its rule cuts in.
///
/// The two are different mechanisms and not two spellings of one: a codon
/// exclusion removes arcs from the codon lattice, and a placed motif threads an
/// automaton axis through node identity. They are held in one type so the loop
/// reads the same either way and the rule is consulted in exactly two places.
enum Cuts {
    /// Per-residue codon exclusions — [`CutRule::Single`] and [`CutRule::All`].
    Codons(Vec<Vec<[u8; 3]>>),
    /// Placed patterns, each forbidden at the window it was cut at —
    /// [`CutRule::Junction`].
    Placed(Vec<(String, MotifWindow)>),
}

/// One cut the loop is about to make: a residue whose codon goes, or a pattern
/// placed at a window.
enum Target {
    Codon(usize),
    Placed(String, MotifWindow),
}

impl Cuts {
    fn new(inputs: &SearchInputs<'_>, rule: CutRule) -> Self {
        match rule {
            CutRule::Junction => Self::Placed(Vec::new()),
            _ => Self::Codons(vec![Vec::new(); inputs.aa.len()]),
        }
    }

    /// The codon exclusions, for the rules that make them. Empty for a placed
    /// loop, which is what keeps [`cuttable`]'s already-cut check modular there.
    fn excluded(&self) -> &[Vec<[u8; 3]>] {
        match self {
            Self::Codons(excluded) => excluded,
            Self::Placed(_) => &[],
        }
    }

    fn add(&mut self, target: Target, codons: &[[u8; 3]]) {
        match (self, target) {
            (Self::Codons(excluded), Target::Codon(position)) => {
                excluded[position].push(codons[position]);
            }
            (Self::Placed(placed), Target::Placed(pattern, window)) => {
                placed.push((pattern, window));
            }
            _ => unreachable!("a rule's targets are in that rule's vocabulary"),
        }
    }

    /// Solve under everything cut so far.
    fn solve(&self, inputs: &SearchInputs<'_>) -> Result<Option<Design>, SearchError> {
        match self {
            Self::Codons(excluded) => solve(inputs, excluded),
            Self::Placed(placed) => Ok(design_with_placed_motifs(
                inputs.params,
                inputs.protein,
                inputs.usage,
                inputs.lambda,
                placed,
                inputs.utr5,
                inputs.utr3,
                inputs.constraints,
                inputs.config,
            )?),
        }
    }

    /// The junction cut a counterexample licenses: the first violating CDS
    /// position whose junction has not already been cut.
    ///
    /// One per round, as [`CutRule::Single`] is — the loop's own reason for
    /// cutting one at a time is that it is the least the counterexample
    /// licenses, and a finer cut has no reason to be greedier.
    fn junction_targets(
        &self,
        inputs: &SearchInputs<'_>,
        reinstated: &[(usize, usize)],
        design: &Design,
        cds: CdsWindow,
    ) -> Vec<Target> {
        let Self::Placed(placed) = self else {
            return Vec::new();
        };
        for &(i, j) in reinstated {
            for endpoint in [i, j] {
                if endpoint < cds.start || endpoint >= cds.end {
                    continue;
                }
                let Some(window) = junction_window(endpoint, cds, design.sequence.len()) else {
                    continue;
                };
                let pattern =
                    design.sequence[window.lo() as usize..window.hi() as usize].to_string();
                if placed.iter().any(|(p, w)| *p == pattern && *w == window) {
                    continue;
                }
                let _ = inputs;
                return vec![Target::Placed(pattern, window)];
            }
        }
        Vec::new()
    }
}

/// The shortest window containing `pos` that **crosses a codon boundary**.
///
/// A window inside one codon would forbid that codon wherever the pattern lands
/// — the codon cut, or worse. Crossing a boundary is what makes the cut remove
/// *combinations* of adjacent codons instead, so every codon involved stays
/// available in every other context.
///
/// Two candidates contain `pos` and cross: one reaching back over this codon's
/// left boundary, one forward over its right. The shorter wins, and ties go
/// left; a candidate that runs off the molecule is not available, and `None`
/// means neither is — a single-codon coding sequence with no flanks, where
/// nothing finer than a codon cut exists.
fn junction_window(pos: usize, cds: CdsWindow, seq_len: usize) -> Option<MotifWindow> {
    let offset = (pos - cds.start) % 3;
    let codon_start = pos - offset;
    let next_codon = codon_start + 3;
    // Reaching left: [codon_start - 1, pos + 1) covers the boundary and `pos`.
    let left = (codon_start >= 1).then(|| (codon_start - 1, pos + 1));
    // Reaching right: [pos, next_codon + 1) covers `pos` and the boundary.
    let right = (next_codon < seq_len).then_some((pos, next_codon + 1));
    let best = match (left, right) {
        (Some(l), Some(r)) if r.1 - r.0 < l.1 - l.0 => r,
        (Some(l), _) => l,
        (None, Some(r)) => r,
        (None, None) => return None,
    };
    Some(MotifWindow::new(best.0 as u32, best.1 as u32))
}
/// The protein positions a counterexample licenses cutting.
///
/// A reinstated pair's endpoints are whole-mRNA positions; only those inside
/// the CDS name a codon, which is why a pair joining two given flanks yields
/// nothing here. A position whose current codon is already excluded is dropped
/// too — it cannot be excluded twice, and re-cutting it would spend a round
/// without shrinking anything.
fn cuttable(
    inputs: &SearchInputs<'_>,
    reinstated: &[(usize, usize)],
    excluded: &[Vec<[u8; 3]>],
    codons: &[[u8; 3]],
    cds: CdsWindow,
    rule: CutRule,
) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for &(i, j) in reinstated {
        for endpoint in [i, j] {
            if endpoint < cds.start || endpoint >= cds.end {
                continue;
            }
            let position = (endpoint - cds.start) / 3;
            if position >= inputs.aa.len() || out.contains(&position) {
                continue;
            }
            if excluded[position].contains(&codons[position]) {
                continue;
            }
            out.push(position);
            if rule == CutRule::Single {
                return out;
            }
        }
    }
    out
}

/// The CDS codons a design spells, in the codon table's encoding.
///
/// [`derna_codon_dp::codon_table::Codon::nuc`] is a nucleotide-index triple
/// (A=0, C=1, G=2, U=3), not ASCII. Copying the sequence bytes across instead
/// yields a triple equal to no codon in the table, and an exclusion built from
/// one silently excludes nothing.
fn cds_codons(inputs: &SearchInputs<'_>, design: &Design) -> Vec<[u8; 3]> {
    let start = inputs.utr5.len();
    let bytes = design.sequence.as_bytes();
    (0..inputs.aa.len())
        .map(|i| {
            let at = |k: usize| {
                [b'A', b'C', b'G', b'U']
                    .iter()
                    .position(|&n| n == bytes[start + 3 * i + k])
                    .expect("the design spells A/C/G/U") as u8
            };
            [at(0), at(1), at(2)]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window always crosses a codon boundary, at every offset within a
    /// codon — which is the whole of what makes the cut finer than a codon cut.
    /// A window inside one codon would forbid that codon wherever the pattern
    /// lands, which is the codon cut or worse.
    #[test]
    fn a_junction_window_always_crosses_a_codon_boundary() {
        let cds = CdsWindow::new(6, 4); // flank 6 nt, four residues
        let seq_len = 6 + 12 + 6;
        for pos in cds.start..cds.end {
            let window = junction_window(pos, cds, seq_len).expect("a boundary is reachable");
            let (lo, hi) = (window.lo() as usize, window.hi() as usize);
            assert!(lo <= pos && pos < hi, "the window does not contain {pos}");
            // A boundary is a multiple of 3 from the CDS start, strictly inside
            // the window.
            let crosses = (lo + 1..hi).any(|b| b >= cds.start && (b - cds.start).is_multiple_of(3));
            assert!(
                crosses,
                "the window [{lo}, {hi}) for position {pos} lies inside one codon, \
                 so the pattern it carries would forbid that codon rather than a \
                 combination of adjacent ones",
            );
            assert!(hi - lo <= 4, "the window is not the shortest that crosses");
        }
    }

    /// The shortest crossing window is 2 or 3 wide, never the 3 positions of one
    /// codon.
    #[test]
    fn a_junction_window_is_the_shortest_that_crosses() {
        let cds = CdsWindow::new(3, 3);
        let seq_len = 3 + 9 + 3;
        // Offsets 0 and 2 reach a boundary in two positions; offset 1 needs
        // three, whichever way it reaches.
        let width = |pos: usize| {
            let w = junction_window(pos, cds, seq_len).expect("reachable");
            (w.hi() - w.lo()) as usize
        };
        assert_eq!(width(cds.start), 2, "offset 0 should reach left in two");
        assert_eq!(width(cds.start + 1), 3, "offset 1 needs three either way");
        assert_eq!(
            width(cds.start + 2),
            2,
            "offset 2 should reach right in two"
        );
    }

    /// A coding sequence with nothing on either side of its only codon has no
    /// boundary to cross, so there is nothing finer than a codon cut and the
    /// window says so rather than returning one inside the codon.
    #[test]
    fn a_lone_codon_has_no_junction() {
        let cds = CdsWindow::new(0, 1);
        assert!(junction_window(0, cds, 3).is_none());
        assert!(junction_window(2, cds, 3).is_none());
    }
}
