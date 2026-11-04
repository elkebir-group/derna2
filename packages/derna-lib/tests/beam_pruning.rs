//! Finite-`b` pruning contract.
//!
//! Per-column beam pruning is an opt-in speed knob over the exact
//! (`b → ∞`) lattice, which stays the default; correctness never depends on
//! pruning. Three properties anchor it: a width at or
//! above every column size leaves the exact optimum untouched; an arbitrary
//! finite width still returns a *valid* design (translatable to the target
//! protein, with the reported MFE matching an independent re-evaluation of
//! the recovered structure — the traceback never follows a pruned cell); and
//! a zero width is rejected.

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::{
    design, design_with_beam, design_with_config, Beam, BeamError, CodonUsage, Params, ScanConfig,
};
use rna_mfe::evaluate;

/// Proteins exercising every loop kind at sizes the default test run folds
/// quickly.
const PROTEINS: [&str; 3] = ["GGGGCCCC", "MKTAYIAKQR", "DEYWCNQHKR"];

/// λ values spanning CAI-priority to MFE-priority.
const LAMBDAS: [f64; 3] = [0.0001, 0.5, 0.99];

/// Whether `seq` is a valid coding sequence for `protein`.
fn translates(seq: &str, protein: &str) -> bool {
    let b = seq.as_bytes();
    if b.len() != protein.len() * 3 {
        return false;
    }
    let code = |c: u8| match c {
        b'A' => 0u8,
        b'C' => 1,
        b'G' => 2,
        _ => 3,
    };
    protein.chars().enumerate().all(|(i, ch)| {
        let aa = aa_to_index(ch).unwrap();
        let trip = [code(b[3 * i]), code(b[3 * i + 1]), code(b[3 * i + 2])];
        AMINO_ACIDS[aa].codons.iter().any(|c| c.nuc == trip)
    })
}

/// A width at or above every column size cannot prune anything, so the
/// design must be identical to the exact (`b → ∞`) lattice.
#[test]
fn wide_beam_equals_exact() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    for protein in PROTEINS {
        for &lambda in &LAMBDAS {
            let exact = design(&params, protein, &usage, lambda).unwrap();
            // These proteins are short enough that no CKY column approaches
            // 100_000 cells, so this width is effectively `b → ∞`.
            let wide = design_with_beam(&params, protein, &usage, lambda, 100_000).unwrap();
            assert_eq!(
                exact, wide,
                "wide beam diverged from exact [{protein} λ={lambda}]",
            );
        }
    }
}

/// Even an aggressively pruned beam returns a valid design: it translates to
/// the target protein and is phantom-free (the recovered structure
/// re-evaluates to the reported MFE). A finite width may select a worse
/// optimum, but never a malformed one — the prune-before-consume order keeps
/// every backpointer pointing at a surviving cell.
#[test]
fn narrow_beam_stays_valid() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    for protein in PROTEINS {
        for &lambda in &LAMBDAS {
            for beam in [1usize, 2, 8] {
                let d = design_with_beam(&params, protein, &usage, lambda, beam).unwrap();
                assert!(
                    translates(&d.sequence, protein),
                    "beam b={beam} produced a non-coding sequence [{protein} λ={lambda}]: {}",
                    d.sequence,
                );
                let reeval = evaluate(&params, d.sequence.as_bytes(), &d.structure)
                    .expect("recovered structure scores")
                    .dekacal() as f64
                    / 100.0;
                assert!(
                    (reeval - d.mfe_kcal_mol).abs() <= 0.005,
                    "phantom energy at b={beam} [{protein} λ={lambda}]: \
                     reported={:.2} reeval={reeval:.2} structure={}",
                    d.mfe_kcal_mol,
                    d.structure,
                );
            }
        }
    }
}

/// The per-open beam and the completion bound are new pruning shapes, so they
/// get the same phantom-free-validity constraint as the flat knob: at an aggressive
/// width every recovered design must still translate to the target protein and
/// re-evaluate to its reported MFE. The prune-before-consume order (now also
/// through the grouped per-open path) must never leave a backpointer at a
/// dropped cell.
#[test]
fn narrow_per_open_beam_stays_valid() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let narrow: [Beam; 6] = [
        Beam {
            per_open: 1,
            cap: usize::MAX,
            cai_bound: false,
            mfe_bound: false,
            ..Beam::exact()
        },
        Beam {
            per_open: 2,
            cap: 8,
            cai_bound: false,
            mfe_bound: false,
            ..Beam::exact()
        },
        Beam {
            per_open: 2,
            cap: 8,
            cai_bound: true,
            mfe_bound: false,
            ..Beam::exact()
        },
        Beam {
            per_open: usize::MAX,
            cap: 4,
            cai_bound: true,
            mfe_bound: false,
            ..Beam::exact()
        },
        // Exterior-MFE completion estimate, alone (flat cap) and stacked on the
        // per-open cap — the new grouped-prune paths get the same phantom constraint.
        Beam {
            per_open: usize::MAX,
            cap: 4,
            cai_bound: false,
            mfe_bound: true,
            ..Beam::exact()
        },
        Beam {
            per_open: 1,
            cap: 8,
            cai_bound: false,
            mfe_bound: true,
            ..Beam::exact()
        },
    ];
    for protein in PROTEINS {
        for &lambda in &LAMBDAS {
            for beam in narrow {
                let d = design_with_config(
                    &params,
                    protein,
                    &usage,
                    lambda,
                    ScanConfig {
                        beam,
                        ..ScanConfig::default()
                    },
                )
                .unwrap();
                assert!(
                    translates(&d.sequence, protein),
                    "beam {beam:?} produced a non-coding sequence [{protein} λ={lambda}]: {}",
                    d.sequence,
                );
                let reeval = evaluate(&params, d.sequence.as_bytes(), &d.structure)
                    .expect("recovered structure scores")
                    .dekacal() as f64
                    / 100.0;
                assert!(
                    (reeval - d.mfe_kcal_mol).abs() <= 0.005,
                    "phantom energy at {beam:?} [{protein} λ={lambda}]: \
                     reported={:.2} reeval={reeval:.2} structure={}",
                    d.mfe_kcal_mol,
                    d.structure,
                );
            }
        }
    }
}

/// The per-open cap, the CAI-half completion bound, and the exterior-MFE
/// completion estimate are additive over the flat knob: at a width above every
/// column size none prunes a cell, so — whatever the ranking — the design must
/// stay byte-for-byte the exact (`b → ∞`) lattice. This pins the invariant that
/// all three refinements leave the exact optimum untouched (the per-open cap
/// only reserves budget that is never scarce here, and the completion terms only
/// reorder the drop set, which is empty).
#[test]
fn wide_per_open_and_completion_bounds_equal_exact() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    // Every refinement combination at an effectively infinite width, including
    // the exterior-MFE estimate alone and stacked with the CAI bound.
    let wide: [Beam; 5] = [
        Beam {
            per_open: usize::MAX,
            cap: 100_000,
            cai_bound: true,
            mfe_bound: false,
            ..Beam::exact()
        },
        Beam {
            per_open: 2,
            cap: 100_000,
            cai_bound: false,
            mfe_bound: false,
            ..Beam::exact()
        },
        Beam {
            per_open: 2,
            cap: 100_000,
            cai_bound: true,
            mfe_bound: false,
            ..Beam::exact()
        },
        Beam {
            per_open: usize::MAX,
            cap: 100_000,
            cai_bound: false,
            mfe_bound: true,
            ..Beam::exact()
        },
        Beam {
            per_open: 2,
            cap: 100_000,
            cai_bound: true,
            mfe_bound: true,
            ..Beam::exact()
        },
    ];
    for protein in PROTEINS {
        for &lambda in &LAMBDAS {
            let exact = design(&params, protein, &usage, lambda).unwrap();
            for beam in wide {
                let got = design_with_config(
                    &params,
                    protein,
                    &usage,
                    lambda,
                    ScanConfig {
                        beam,
                        ..ScanConfig::default()
                    },
                )
                .unwrap();
                assert_eq!(
                    exact, got,
                    "wide beam {beam:?} diverged from exact [{protein} λ={lambda}]",
                );
            }
        }
    }
}

/// A zero beam width is meaningless (no cell could ever survive) and is
/// rejected rather than silently emptying the lattice.
#[test]
fn zero_beam_is_rejected() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    assert_eq!(
        design_with_beam(&params, "GGGGCCCC", &usage, 0.5, 0),
        Err(BeamError::ZeroBeamWidth),
    );
}
