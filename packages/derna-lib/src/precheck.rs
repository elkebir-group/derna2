//! The flank-versus-flank pre-check: what two fixed UTR flanks do to each other
//! with no coding sequence between them.
//!
//! Every other instrument here folds a whole mRNA, so everything it reports is a
//! property of one CDS. This one removes the CDS entirely: the two flanks are
//! folded against each other, and the pairs the active table refuses are pairs
//! with **both endpoints in fixed sequence**. No codon assignment can change
//! either nucleotide, so no design can spell those pairs away.
//!
//! # What it can and cannot say
//!
//! It is live only for a table that enforces something on a pair of flank
//! positions — `5′UTR↔3′UTR`, an intra-flank class, or an accessibility window
//! reaching into the 5′ UTR — **and** only when both flanks are given rather
//! than designed. Under a table forbidding `5′UTR↔CDS` alone it can never fire:
//! every pair of that class has one endpoint in the designed CDS, so folding the
//! flanks in isolation says nothing about it.
//!
//! Where it is live, a finding is a **prediction, not a proof**. The pairs it
//! reports are available to every candidate molecule, because their nucleotides
//! are fixed; whether the whole-mRNA free fold actually forms them depends on
//! what the CDS offers as competition. A clear result is correspondingly weak
//! evidence — the CDS can introduce violations of its own. The instrument earns
//! its place by being the only one whose domain is CDS-independent, and it costs
//! a fold of the flanks alone.

use rna_mfe::Params;

use crate::free_fold::{measure, FreeFoldError, FreeFoldGap};
use crate::region::{AccessibilityWindow, PairClass, PairConstraint, RegionConstraints};

/// The shortest span a base pair can close: `j − i − 1 ≥ 3` nucleotides of loop.
/// Matches the folding recurrence, so a "live" entry is one some foldable pair
/// can actually reach.
const MIN_HAIRPIN_LOOP: usize = 3;

/// What the two fixed flanks do to each other under a table, and which entries
/// of that table had anything to say about it.
///
/// Produced by [`flank_precheck`]. Positions are **concatenated coordinates**
/// over `UTR₅′ + UTR₃′` with the CDS absent; [`Self::in_mrna`] maps one back to
/// the whole molecule.
#[derive(Debug, Clone, PartialEq)]
pub struct FlankPrecheck {
    /// Nucleotides in the 5′ flank — the region boundary of the concatenation.
    pub utr5_len: usize,
    /// Nucleotides in the 3′ flank.
    pub utr3_len: usize,
    /// The classes the table enforces that a pair of flank positions can fall
    /// in, ascending. Empty (with no live window) means the instrument has
    /// nothing to say about this table.
    pub live_classes: Vec<PairClass>,
    /// Whether the accessibility window reaches a pair of flank positions. Only
    /// its **upstream** extent does: the downstream half lies in the CDS, which
    /// is absent here.
    pub window_live: bool,
    /// The flanks folded against each other, or `None` when nothing is live and
    /// the fold would answer no question.
    pub fold: Option<FreeFoldGap>,
}

impl FlankPrecheck {
    /// Whether the table enforces anything reachable by a flank-versus-flank
    /// pair. When false, everything below is empty and says nothing about the
    /// design — not even that it is feasible.
    pub fn is_live(&self) -> bool {
        !self.live_classes.is_empty() || self.window_live
    }

    /// The pairs of the flank-only fold the table refuses — both endpoints in
    /// fixed sequence. Empty when the pre-check is not live.
    pub fn blocking_pairs(&self) -> &[(usize, usize)] {
        match &self.fold {
            Some(f) => &f.reinstated_pairs,
            None => &[],
        }
    }

    /// Whether the fixed flanks, left to themselves, prefer a structure the
    /// table refuses.
    pub fn blocks(&self) -> bool {
        !self.blocking_pairs().is_empty()
    }

    /// A concatenated-coordinate `pair` in whole-mRNA coordinates, given the CDS
    /// length in nucleotides: 3′ endpoints shift right by the whole CDS.
    pub fn in_mrna(&self, pair: (usize, usize), cds_nt: usize) -> (usize, usize) {
        let shift = |p: usize| if p < self.utr5_len { p } else { p + cds_nt };
        (shift(pair.0), shift(pair.1))
    }
}

/// Fold the two **fixed** flanks against each other with no CDS present, and
/// report the pairs `constraints` refuses.
///
/// `utr5` and `utr3` are nucleotide sequences (`A/C/G/U`, `T` read as `U`); a
/// designed flank has no sequence to fold and belongs to the not-live case,
/// which the caller expresses by not calling this. See the module docs for the
/// scope of a finding.
pub fn flank_precheck(
    params: &Params,
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
) -> Result<FlankPrecheck, FreeFoldError> {
    let table = flank_only_table(constraints);
    let constraint = PairConstraint::for_cds(&table, utr5.len() as u32, utr5.len() as u32);
    let total = utr5.len() + utr3.len();
    let mut report = FlankPrecheck {
        utr5_len: utr5.len(),
        utr3_len: utr3.len(),
        live_classes: live_classes(&table, &constraint, total),
        window_live: live_window(&table, &constraint, total),
        fold: None,
    };
    if report.is_live() {
        report.fold = Some(measure(
            params,
            format!("{utr5}{utr3}").as_bytes(),
            constraint,
        )?);
    }
    Ok(report)
}

/// `constraints` restricted to what a CDS-free molecule can model: the window's
/// **downstream** extent covers CDS positions, which the concatenation does not
/// have, so keeping it would enforce against the 3′ flank instead.
fn flank_only_table(constraints: RegionConstraints) -> RegionConstraints {
    match constraints.accessibility {
        Some(w) => constraints.with_window(AccessibilityWindow { downstream: 0, ..w }),
        None => constraints,
    }
}

/// The classes `table` enforces that some foldable pair of the concatenation
/// falls in — the entries this fold can put to a test.
fn live_classes(
    table: &RegionConstraints,
    constraint: &PairConstraint,
    total: usize,
) -> Vec<PairClass> {
    PairClass::ALL
        .into_iter()
        .filter(|&class| {
            table.enforcement(class).is_some()
                && any_foldable_pair(total, |i, j| constraint.class_of(i, j) == class)
        })
        .collect()
}

/// Whether the window reaches some foldable pair of the concatenation.
fn live_window(table: &RegionConstraints, constraint: &PairConstraint, total: usize) -> bool {
    table.accessibility.is_some() && any_foldable_pair(total, |i, j| constraint.in_window(i, j))
}

/// Whether any pair the folding recurrence admits over `total` positions
/// satisfies `pred`.
fn any_foldable_pair(total: usize, pred: impl Fn(u32, u32) -> bool) -> bool {
    (0..total).any(|i| ((i + MIN_HAIRPIN_LOOP + 1)..total).any(|j| pred(i as u32, j as u32)))
}

#[cfg(test)]
mod tests {
    //! Liveness — which tables the instrument has anything to say about — and
    //! the coordinate mapping. The end-to-end contracts are integration tests at
    //! `tests/flank_precheck.rs`.

    use super::*;
    use crate::Enforcement;

    const UTR5: &str = "GGGGGGGGGG";
    const UTR3: &str = "CCCCCCCCCC";

    fn precheck(constraints: RegionConstraints) -> FlankPrecheck {
        let params = Params::load_default().expect("default Turner 2004 parameters");
        flank_precheck(&params, UTR5, UTR3, constraints).expect("the flanks fold")
    }

    /// `5′↔CDS` alone is not reachable by a flank-versus-flank fold: that class
    /// always has an endpoint in the designed CDS. The instrument is dead for
    /// the `five-prime` table and says so.
    #[test]
    fn five_prime_isolation_alone_is_not_live() {
        let report = precheck(RegionConstraints::five_prime_isolation(Enforcement::Forbid));
        assert!(!report.is_live());
        assert!(report.fold.is_none(), "a dead table earns no fold");
        assert!(!report.blocks());
    }

    /// Forbidding the flank-to-flank class is exactly the case the instrument is
    /// for — and it is the shipped default, so the pre-check reports on an
    /// unconfigured run.
    #[test]
    fn the_shipped_default_is_live_through_the_clamp_class() {
        let report = precheck(RegionConstraints::no_clamp(Enforcement::Forbid));
        assert_eq!(report.live_classes, [PairClass::Utr5Utr3]);
        assert!(report.fold.is_some());
    }

    /// Full mutual isolation carries the flank-to-flank class, so it is live for
    /// that entry alone — its other two entries touch the absent CDS.
    #[test]
    fn full_isolation_is_live_on_its_flank_entry_alone() {
        let report = precheck(RegionConstraints::full_mutual_isolation(
            Enforcement::Forbid,
        ));
        assert_eq!(report.live_classes, [PairClass::Utr5Utr3]);
    }

    /// An intra-flank entry is live without any crossing entry at all, and a
    /// soft one counts: the table has something to say either way.
    #[test]
    fn an_intra_flank_entry_is_live() {
        let soft = Enforcement::Penalize(rna_mfe::Energy(50));
        let report = precheck(RegionConstraints::default().with(PairClass::Utr3Utr3, soft));
        assert_eq!(report.live_classes, [PairClass::Utr3Utr3]);
    }

    /// The window's upstream extent reaches fixed 5′ sequence; its downstream
    /// extent is CDS and is dropped, so a window with no upstream reach is dead.
    #[test]
    fn only_the_windows_upstream_extent_is_live() {
        let upstream = RegionConstraints::default().with_window(AccessibilityWindow::hard(6, 15));
        assert!(precheck(upstream).window_live);

        let downstream_only =
            RegionConstraints::default().with_window(AccessibilityWindow::hard(0, 15));
        assert!(!precheck(downstream_only).is_live());
    }

    /// An inert table has nothing to check.
    #[test]
    fn an_inert_table_is_not_live() {
        assert!(!precheck(RegionConstraints::default()).is_live());
    }

    /// Concatenated coordinates skip the CDS; whole-mRNA coordinates do not.
    #[test]
    fn in_mrna_shifts_only_the_three_prime_endpoint() {
        let report = precheck(RegionConstraints::default());
        assert_eq!(report.utr5_len, UTR5.len());
        // A 5′↔3′ pair over a 90-nt CDS: the 3′ endpoint moves, the 5′ one does
        // not.
        assert_eq!(report.in_mrna((2, 12), 90), (2, 102));
        // An intra-5′ pair is untouched.
        assert_eq!(report.in_mrna((1, 8), 90), (1, 8));
    }
}
