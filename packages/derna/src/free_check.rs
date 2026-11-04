//! Checking a design against the free fold of its own sequence.
//!
//! A design is **modular** when its free MFE equals its
//! constrained energy: the molecule's own preferred fold already respects the
//! constraint. The check is one free fold of the design's own sequence, plus a
//! rescore of the structure that fold emitted — not a second fold. The
//! constrained energy comes from the design, the free one from that fold, and
//! their difference is the **gap**.
//!
//! `--beam-first B` makes the refutation cheap. A beam fold returns a real
//! structure, so its energy is an **upper bound** on the free MFE; a bound
//! already below the constrained MFE proves a positive gap without the free
//! fold. It can only refute: a beam that fails to beat that energy has proved
//! nothing, and the exact fold runs. What the bound is measured against has to
//! be the sequence's constrained MFE rather than whatever energy the design
//! arrived with, so [`check`] folds that energy and confirms the two agree
//! before taking a refutation — and keeps the fold either way, since the exact
//! verdict is then the free fold against it. So the flag costs a beam fold and
//! saves a free one; it never pays for the same constrained fold twice.

use std::collections::VecDeque;

use derna_lib::{
    free_fold_gap, linearfold_fold, sequence_constrained_mfe_zuker, sequence_gap_given_constrained,
    sequence_gap_given_structure, sequence_gap_given_structure_reusing, CdsWindow, Design,
    FreeFoldGap, FreeFoldTables, Params,
};
use derna_lib::{Energy, RegionConstraints};

use crate::Result;

/// What the check established about one design.
#[derive(Debug)]
pub enum Checked {
    /// Both folds ran: the exact verdict.
    Exact(FreeFoldGap),
    /// A beam fold refuted the design without the exact fold: the free MFE is
    /// at most this, so the gap is at least `constrained − free`.
    Refuted {
        /// An upper bound on the free MFE, in kcal/mol — a structure's own
        /// energy.
        free_mfe_at_most: f64,
        /// The gap this bound establishes, in kcal/mol; positive.
        gap_at_least: f64,
    },
}

impl Checked {
    /// Whether the design's own fold respects the constraint. A refutation
    /// answers this with `false`, which is the whole point of running one.
    pub fn modular(&self) -> bool {
        match self {
            Self::Exact(gap) => gap.is_modular(),
            Self::Refuted { .. } => false,
        }
    }
}

/// Check `design` under `constraints` over the coding window `cds`.
///
/// With `beam_first` given, a beam fold of that width runs first and the free
/// fold is skipped where it refutes the design.
pub fn check(
    params: &Params,
    design: &Design,
    cds: CdsWindow,
    constraints: RegionConstraints,
    beam_first: Option<usize>,
) -> Result<Checked> {
    // A beam refutation is measured against the sequence's constrained MFE, and
    // the design's reported energy stands for it only where a fold agrees, so
    // the beam path needs the constrained fold the exact check would run anyway.
    // That fold is therefore done once here and handed on, rather than being
    // discarded when the refutation does not hold and recomputed below.
    if let Some(width) = beam_first {
        if constraints.is_hard() {
            if let Some(beam) = beats_the_claim(params, design, width)? {
                let zuker =
                    sequence_constrained_mfe_zuker(params, &design.sequence, cds, constraints)?;
                if (zuker - design.mfe_kcal_mol).abs() < 1e-9 {
                    return Ok(Checked::Refuted {
                        free_mfe_at_most: beam,
                        gap_at_least: design.mfe_kcal_mol - beam,
                    });
                }
                // The reported energy is not this sequence's constrained MFE, so
                // a structure below it bounds nothing about the gap. The exact
                // verdict is the free fold against the energy folded above.
                return Ok(Checked::Exact(sequence_gap_given_constrained(
                    params,
                    &design.sequence,
                    cds,
                    constraints,
                    zuker,
                )?));
            }
        }
    }
    Ok(Checked::Exact(exact(params, design, cds, constraints)?))
}

/// The exact check, with the constrained fold run only where the design's
/// energy cannot stand for it.
///
/// Where the design's structure re-scores to its energy and a hard table
/// allows it, that energy bounds the constrained MFE from above, and the gap
/// against it is read from the free fold alone
/// ([`sequence_gap_given_structure`]). It is the exact gap when the energy is
/// the constrained MFE, which holds when that gap is zero (the free MFE bounds
/// the constrained one from below), or when the constrained Zuker fold reaches
/// the energy too — that fold minimizes over the scan's own structure class, so
/// where both are exact they agree.
///
/// Where the structure could **not** stand for the energy,
/// [`sequence_gap_given_structure`] has already folded the sequence both ways
/// and the report it returns is the one the scan would produce again — the same
/// function of the same sequence under the same constraints — so it is handed
/// on rather than recomputed. The scan therefore runs only where the structure
/// stood, the gap is positive and the Zuker fold does not reach the energy:
/// the one case where what is already in hand is a free fold this cannot pass
/// to [`free_fold_gap`].
fn exact(
    params: &Params,
    design: &Design,
    cds: CdsWindow,
    constraints: RegionConstraints,
) -> Result<FreeFoldGap> {
    exact_with(params, design, cds, constraints, || {
        Ok(sequence_gap_given_structure(
            params,
            &design.sequence,
            &design.structure,
            design.mfe_kcal_mol,
            cds,
            constraints,
        )?)
    })
}

/// [`check`] without a beam, copying the free fold's cells from the folds in
/// `pool` and keeping this one's tables there, at most `keep` of them.
///
/// Consecutive points of one front differ in a few codons, so most of a free
/// fold's cells are a neighbour's. A copied cell holds the value the fresh
/// fold computes, so the verdict is [`check`]'s; only the time differs.
pub fn check_reusing(
    params: &Params,
    design: &Design,
    cds: CdsWindow,
    constraints: RegionConstraints,
    pool: &mut VecDeque<FreeFoldTables>,
    keep: usize,
) -> Result<Checked> {
    let gap = exact_with(params, design, cds, constraints, || {
        let held: Vec<&FreeFoldTables> = pool.iter().collect();
        let (gap, tables) = sequence_gap_given_structure_reusing(
            params,
            &design.sequence,
            &design.structure,
            design.mfe_kcal_mol,
            cds,
            constraints,
            &held,
        )?;
        if let Some(tables) = tables.filter(|_| keep > 0) {
            if pool.len() >= keep {
                pool.pop_front();
            }
            pool.push_back(tables);
        }
        Ok(gap)
    })?;
    Ok(Checked::Exact(gap))
}

/// [`exact`] with the structure's gap taken by `given`, which
/// [`sequence_gap_given_structure`] or its reusing form computes.
fn exact_with(
    params: &Params,
    design: &Design,
    cds: CdsWindow,
    constraints: RegionConstraints,
    given: impl FnOnce() -> Result<FreeFoldGap>,
) -> Result<FreeFoldGap> {
    if !constraints.is_hard() {
        return Ok(free_fold_gap(params, design, cds, constraints)?);
    }
    let gap = given()?;
    if gap.is_modular() {
        return Ok(gap);
    }
    // The structure stood for the energy exactly when the gap was measured
    // against the design's own, which is what `sequence_gap_given_constrained`
    // writes into `constrained_mfe_kcal_mol`. Where it did not, this report
    // already carries the scan's constrained optimum with a free fold beside
    // it, which is what the scan below would compute.
    if (gap.constrained_mfe_kcal_mol - design.mfe_kcal_mol).abs() >= 1e-9 {
        return Ok(gap);
    }
    let zuker = sequence_constrained_mfe_zuker(params, &design.sequence, cds, constraints)?;
    if (zuker - design.mfe_kcal_mol).abs() < 1e-9 {
        return Ok(gap);
    }
    Ok(free_fold_gap(params, design, cds, constraints)?)
}

/// A beam fold of the design's own sequence, and the bound it establishes on
/// the free MFE — `None` where it does not beat the design's reported energy
/// and so establishes nothing.
///
/// A structure strictly below the sequence's **constrained MFE** is one the
/// constraint does not allow, so the free optimum is below it too and the gap
/// is positive. Energies are integer dekacalories under the model, so "strictly
/// below" is a comparison of integers rather than of floats.
///
/// Whether the reported energy *is* that constrained MFE is not decided here:
/// the caller folds it and decides, because a reported energy above the
/// sequence's constrained optimum — a traceback at λ = 0, where every structure
/// is equally optimal, or a front built at a limited beam — would let a
/// structure lying between the two refute a design whose free and constrained
/// optima in fact agree.
fn beats_the_claim(params: &Params, design: &Design, width: usize) -> Result<Option<f64>> {
    let (_, beam_energy) = linearfold_fold(params, design.sequence.as_bytes(), width)?;
    let claimed = Energy((design.mfe_kcal_mol * 100.0).round() as i32);
    if beam_energy >= claimed {
        return Ok(None);
    }
    // The library's own conversion, so the dekacalorie scale is stated in one
    // place rather than re-derived here.
    Ok(Some(beam_energy.to_kcal_mol()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use derna_lib::{Enforcement, PairClass};

    fn design(sequence: &str, structure: &str, mfe: f64) -> Design {
        Design {
            sequence: sequence.to_string(),
            structure: structure.to_string(),
            mfe_kcal_mol: mfe,
            cai_log: 0.0,
            cai: 1.0,
            utr_profile_log: 0.0,
        }
    }

    /// A sequence whose own fold is strongly paired, so a narrow beam still
    /// finds a structure well below zero.
    const HAIRPIN: &str = "GGGGGAAAAACCCCC";

    /// The constraint that forbids this sequence's own stem: with the five
    /// leading G's in the 5' UTR, every pair of the hairpin but the outermost
    /// crosses the 5' UTR–CDS boundary.
    fn stem_forbidden() -> (CdsWindow, RegionConstraints) {
        (
            CdsWindow::new(5, 3),
            RegionConstraints::enforcing(&[PairClass::Utr5Cds], Enforcement::Forbid),
        )
    }

    #[test]
    fn a_beam_fold_refutes_a_design_whose_constrained_energy_it_beats() {
        let params = Params::load_default().unwrap();
        let (cds, constraints) = stem_forbidden();
        // The claimed energy is this sequence's own constrained MFE, so the
        // refutation's premise holds and the beam's structure — which the
        // constraint forbids — proves a positive gap.
        let constrained =
            sequence_constrained_mfe_zuker(&params, HAIRPIN, cds, constraints).unwrap();
        let claimed = design(HAIRPIN, ".".repeat(HAIRPIN.len()).as_str(), constrained);
        let checked = check(&params, &claimed, cds, constraints, Some(10)).unwrap();
        match checked {
            Checked::Refuted {
                free_mfe_at_most,
                gap_at_least,
            } => {
                assert!(free_mfe_at_most < constrained, "{free_mfe_at_most}");
                assert!(gap_at_least > 0.0, "{gap_at_least}");
                // The bound is the structure's energy, so the gap it
                // establishes is the difference from the constrained energy.
                assert!((gap_at_least - (constrained - free_mfe_at_most)).abs() < 1e-9);
            }
            Checked::Exact(_) => panic!("the beam should have refuted this design"),
        }
    }

    #[test]
    fn an_energy_above_the_constrained_optimum_is_not_refuted() {
        let params = Params::load_default().unwrap();
        // Nothing is forbidden here — the whole molecule is coding — so this
        // sequence's free and constrained optima agree and every design of it
        // is modular. The claimed energy is nonetheless above that
        // optimum, as a λ = 0 traceback or a beam-built front can leave it.
        let cds = CdsWindow::new(0, HAIRPIN.len() / 3);
        let constraints = RegionConstraints::enforcing(&[PairClass::Utr5Cds], Enforcement::Forbid);
        let optimum = sequence_constrained_mfe_zuker(&params, HAIRPIN, cds, constraints).unwrap();
        assert!(optimum < 0.0, "the fixture should fold: {optimum}");
        let claimed = design(HAIRPIN, ".".repeat(HAIRPIN.len()).as_str(), 0.0);

        let checked = check(&params, &claimed, cds, constraints, Some(10)).unwrap();
        // A beam structure sits between the claimed energy and the optimum, so
        // refuting against the claimed energy would report a positive gap for a
        // design whose own molecule respects the constraint.
        let gap = match checked {
            Checked::Exact(gap) => gap,
            Checked::Refuted { gap_at_least, .. } => {
                panic!("refuted a modular design, gap_at_least {gap_at_least}")
            }
        };
        assert!(
            gap.is_modular(),
            "gap {} against free MFE {}",
            gap.gap_kcal_mol,
            gap.free_mfe_kcal_mol
        );
        // And the exact check answered against a freshly folded constrained
        // energy rather than the one the design arrived with.
        assert!((gap.constrained_mfe_kcal_mol - optimum).abs() < 1e-9);
    }

    #[test]
    fn a_beam_that_cannot_beat_the_energy_leaves_the_exact_fold_to_run() {
        let params = Params::load_default().unwrap();
        // A design claiming an energy no structure of this sequence reaches:
        // the beam proves nothing and the exact check runs.
        let claimed = design(HAIRPIN, ".".repeat(HAIRPIN.len()).as_str(), -100.0);
        let checked = check(
            &params,
            &claimed,
            CdsWindow::new(0, HAIRPIN.len() / 3),
            RegionConstraints::default(),
            Some(10),
        )
        .unwrap();
        assert!(
            matches!(checked, Checked::Exact(_)),
            "the beam refuted nothing"
        );
    }

    #[test]
    fn a_soft_table_is_never_refuted_by_the_beam() {
        let params = Params::load_default().unwrap();
        // A penalized pair is not forbidden, so a structure below the design's
        // energy is not evidence that the constraint excludes it — and the
        // constrained Zuker fold a refutation would have to confirm against
        // takes hard tables only. The beam therefore establishes nothing and the
        // exact check runs, as it does with no beam at all.
        let soft =
            RegionConstraints::enforcing(&[PairClass::Utr5Cds], Enforcement::Penalize(Energy(500)));
        assert!(!soft.is_hard());
        let claimed = design(HAIRPIN, ".".repeat(HAIRPIN.len()).as_str(), 0.0);
        let checked = check(
            &params,
            &claimed,
            CdsWindow::new(0, HAIRPIN.len() / 3),
            soft,
            Some(10),
        )
        .unwrap();
        assert!(
            matches!(checked, Checked::Exact(_)),
            "a soft table was refuted on a beam bound"
        );
    }

    #[test]
    fn with_no_beam_the_check_is_the_exact_one() {
        let params = Params::load_default().unwrap();
        let claimed = design(HAIRPIN, ".".repeat(HAIRPIN.len()).as_str(), 0.0);
        let checked = check(
            &params,
            &claimed,
            CdsWindow::new(0, HAIRPIN.len() / 3),
            RegionConstraints::default(),
            None,
        )
        .unwrap();
        assert!(matches!(checked, Checked::Exact(_)));
    }

    /// The exact check reads what the two-fold check does — constrained and
    /// free energy, gap and the pairs put back — on optimal designs that pass
    /// and that fail, and on designs claiming more than their optimum, which
    /// the scan alone can price.
    #[test]
    fn the_exact_check_is_the_two_fold_one() {
        let params = Params::load_default().unwrap();
        let table = RegionConstraints::enforcing(
            &[PairClass::Utr5Cds, PairClass::CdsUtr3],
            Enforcement::Forbid,
        );
        let cds = CdsWindow::new(24, 18);
        let mut state = 0x510e_527f_ade6_82d1_u64;
        let (mut passed, mut failed, mut claimed_high) = (0, 0, 0);
        for k in 0..24 {
            let seq: String = (0..96)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    b"ACGU"[(state % 4) as usize] as char
                })
                .collect();
            let seq = if k % 2 == 0 {
                format!("GGGGCCCC{}", &seq[8..])
            } else {
                seq
            };
            let (structure, mfe) =
                derna_lib::sequence_constrained_fold_must(&params, &seq, cds, table, &[])
                    .unwrap()
                    .unwrap();
            // Every third design claims an unpaired structure instead: above
            // its optimum wherever the sequence folds at all.
            let claim = if k % 3 == 2 {
                claimed_high += 1;
                design(&seq, &".".repeat(seq.len()), 0.0)
            } else {
                design(&seq, &structure, mfe)
            };
            let got = exact(&params, &claim, cds, table).unwrap();
            let want = free_fold_gap(&params, &claim, cds, table).unwrap();
            assert_eq!(
                got.constrained_mfe_kcal_mol, want.constrained_mfe_kcal_mol,
                "{k}"
            );
            assert_eq!(got.free_mfe_kcal_mol, want.free_mfe_kcal_mol, "{k}");
            assert_eq!(got.gap_kcal_mol, want.gap_kcal_mol, "{k}");
            assert_eq!(got.reinstated_pairs, want.reinstated_pairs, "{k}");
            assert_eq!(got.free_structure, want.free_structure, "{k}");
            if got.is_modular() {
                passed += 1;
            } else {
                failed += 1;
            }
        }
        assert!(
            passed > 0 && failed > 0 && claimed_high > 0,
            "{passed} {failed}"
        );
    }

    /// A design whose structure cannot stand for its energy is priced by the
    /// scan, and the exact check reads what the two-fold check reads on it.
    ///
    /// This is the path where the structure test fails, so the first step has
    /// already folded the sequence both ways and its report is handed on
    /// instead of being recomputed. The energies, the gap, the free structure
    /// and the pairs put back must therefore be the two-fold check's, and the
    /// case is asserted to be the intended one: the gap positive, so nothing
    /// returns at the modularity test, and the constrained energy
    /// not the design's own, which is what a failed structure test leaves.
    #[test]
    fn a_structure_that_cannot_stand_for_its_energy_is_priced_by_the_scan() {
        let params = Params::load_default().unwrap();
        let (cds, table) = stem_forbidden();
        // The unpaired structure accounts for 0.0, so an energy below it is one
        // this structure cannot explain and the structure test refuses.
        let claim = design(HAIRPIN, &".".repeat(HAIRPIN.len()), -1.0);
        let got = exact(&params, &claim, cds, table).unwrap();
        let want = free_fold_gap(&params, &claim, cds, table).unwrap();

        assert!(
            !got.is_modular(),
            "the fixture must leave a positive gap, got {}",
            got.gap_kcal_mol
        );
        assert!(
            (got.constrained_mfe_kcal_mol - claim.mfe_kcal_mol).abs() >= 1e-9,
            "the structure test must have refused, constrained {} claimed {}",
            got.constrained_mfe_kcal_mol,
            claim.mfe_kcal_mol
        );
        assert_eq!(
            got,
            FreeFoldGap {
                emitted: None,
                ..want
            },
            "the scan's own report, with no re-score on this path"
        );
    }
}
