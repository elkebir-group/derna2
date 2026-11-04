//! Motif-exclusion entry-point contract: additivity and clean infeasibility.
//!
//! Two properties anchor the motif-constrained design surface before the
//! constrained folding scan is wired. **Additivity:** an empty forbidden-motif
//! set is byte-identical to the unconstrained design — the constraint allocates
//! nothing and perturbs no result (the floor the whole feature must never
//! break). **Clean infeasibility:** a motif set that leaves no motif-free
//! encoding of the protein reports a typed error rather than panicking or
//! returning a garbage optimum.
//!
//! These mirror the `beam_pruning` `usize::MAX`-default contract shape: the
//! opt-in axis (here, a non-empty motif set; there, a finite beam) must reduce
//! to the exact default when it is not exercised.

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::ac::MotifError;
use derna_lib::{
    design, design_with_motifs, Beam, BeamError, CodonUsage, Design, Params, ScanConfig,
};
use rna_mfe::evaluate;

/// Proteins exercising a range of codon-lattice shapes, short enough to fold
/// quickly (shared with `beam_pruning`).
const PROTEINS: [&str; 3] = ["GGGGCCCC", "MKTAYIAKQR", "DEYWCNQHKR"];

/// Whether `seq` (ASCII `A/C/G/U`) is a valid coding sequence for `protein`.
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

/// λ values spanning CAI-priority to MFE-priority.
const LAMBDAS: [f64; 3] = [0.0001, 0.5, 0.99];

/// An empty forbidden-motif set routes through the unconstrained scan verbatim,
/// so the design must be byte-for-byte identical to [`design`] — the additivity
/// floor. `&[] as &[&str]` pins the element type the generic entry point infers.
#[test]
fn empty_motif_set_equals_unconstrained() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    for protein in PROTEINS {
        for &lambda in &LAMBDAS {
            let exact = design(&params, protein, &usage, lambda).unwrap();
            let empty = design_with_motifs(
                &params,
                protein,
                &usage,
                lambda,
                &[] as &[&str],
                ScanConfig::default(),
            )
            .unwrap();
            assert_eq!(
                exact, empty,
                "empty motif set diverged from unconstrained [{protein} λ={lambda}]",
            );
        }
    }
}

/// A motif set that forbids the protein's only encoding leaves no motif-free
/// coding sequence, so the design is infeasible and reported cleanly.
/// `MW` encodes uniquely as `AUGUGG` (Met → AUG, Trp → UGG are single-codon),
/// so forbidding `AUG` removes the sole encoding.
#[test]
fn over_constrained_motif_set_is_infeasible() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    for &lambda in &LAMBDAS {
        let got = design_with_motifs(
            &params,
            "MW",
            &usage,
            lambda,
            &["AUG"],
            ScanConfig::default(),
        );
        assert_eq!(
            got,
            Err(BeamError::Infeasible),
            "expected clean infeasibility when the only encoding is forbidden [λ={lambda}]",
        );
    }
}

/// `T` is read as `U`, so the RNA motif `AUG` and the DNA spelling `ATG` forbid
/// the same substring — the same infeasibility verdict on the uniquely-encoded
/// `MW`.
#[test]
fn dna_spelled_motif_matches_rna() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let got = design_with_motifs(&params, "MW", &usage, 0.5, &["ATG"], ScanConfig::default());
    assert_eq!(got, Err(BeamError::Infeasible));
}

/// A malformed motif (empty, or containing a non-IUPAC code) is a typed build
/// error, surfaced before any folding is attempted.
#[test]
fn malformed_motif_is_a_typed_error() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let empty_motif = design_with_motifs(&params, "MKT", &usage, 0.5, &[""], ScanConfig::default());
    assert_eq!(empty_motif, Err(BeamError::Motif(MotifError::Empty)));
    let bad_code = design_with_motifs(&params, "MKT", &usage, 0.5, &["ACZ"], ScanConfig::default());
    assert!(matches!(
        bad_code,
        Err(BeamError::Motif(MotifError::InvalidCode { .. })),
    ));
}

/// A feasible, genuinely-constraining motif set folds to a real design: the
/// entry point classifies it (pass-through vs infeasible vs constrained) and
/// routes it through the AC-threaded constrained fold. The recovered sequence
/// must be motif-free, translate to the target protein, and be phantom-free (its
/// structure re-evaluates to the reported MFE) — the end-to-end contract the
/// exhaustive fuzz oracle in `motif_blacklist_fuzz` checks on small proteins.
#[test]
fn feasible_constrained_set_folds_to_a_valid_design() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let protein = "MKTAYIAKQR";
    // A single rare 6-mer cannot make a 10-residue protein unencodable, so the
    // constraint is active but the set stays feasible.
    let motif = "ACGACG";
    let d = design_with_motifs(
        &params,
        protein,
        &usage,
        0.5,
        &[motif],
        ScanConfig::default(),
    )
    .expect("feasible constrained set folds");
    assert!(
        !d.sequence.contains(motif),
        "spelled a forbidden motif: {}",
        d.sequence,
    );
    assert!(
        translates(&d.sequence, protein),
        "spelled a non-coding sequence: {}",
        d.sequence,
    );
    let reeval = evaluate(&params, d.sequence.as_bytes(), &d.structure)
        .expect("recovered structure scores")
        .dekacal() as f64
        / 100.0;
    assert!(
        (reeval - d.mfe_kcal_mol).abs() <= 0.005,
        "phantom energy: reported={:.2} reeval={reeval:.2} structure={}",
        d.mfe_kcal_mol,
        d.structure,
    );
}

/// The fixed-λ BRD of a design in dekacalorie units — lower is better.
fn brd(d: &Design, lambda: f64) -> f64 {
    lambda * (d.mfe_kcal_mol * 100.0) - (1.0 - lambda) * (d.cai_log * 100.0)
}

/// The beam-pruning axis must reduce to the exact constrained fold when it is
/// not exercised: through the constrained path, every exact-equivalent [`Beam`]
/// (`per_open == MAX && cap == MAX`, whatever the completion-bound flags) yields
/// a design byte-identical to the default. This is the additivity floor for the
/// beam composition — the analogue of `empty_motif_set_equals_unconstrained`
/// on the beam axis rather than the motif axis.
#[test]
fn exact_beam_equals_default_on_constrained_path() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let protein = "MKTAYIAKQR";
    let motif = "ACGACG"; // active but leaves the protein encodable
    let exact_beams = [
        Beam::exact(),
        Beam::flat(usize::MAX),
        Beam {
            per_open: usize::MAX,
            cap: usize::MAX,
            cai_bound: true,
            mfe_bound: true,
            ..Beam::exact()
        },
    ];
    for &lambda in &LAMBDAS {
        let base = design_with_motifs(
            &params,
            protein,
            &usage,
            lambda,
            &[motif],
            ScanConfig::default(),
        )
        .expect("feasible constrained set folds");
        for beam in exact_beams {
            let got = design_with_motifs(
                &params,
                protein,
                &usage,
                lambda,
                &[motif],
                ScanConfig {
                    beam,
                    ..ScanConfig::default()
                },
            )
            .expect("feasible constrained set folds under an exact beam");
            assert_eq!(
                base, got,
                "exact-equivalent beam {beam:?} diverged from the default constrained fold \
                 [{protein} λ={lambda}]",
            );
        }
    }
}

/// A finite beam on the constrained path preserves feasibility and motif-freeness
/// by construction (the AC product removed the infeasible edges; the exterior
/// spine is never pruned), and — being a relaxation of the exact search — can
/// only match or worsen the BRD, never beat it. This pins the beam's correctness
/// story: pruning trades quality, never validity.
#[test]
fn finite_beam_stays_feasible_and_no_better_than_exact() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let protein = "MKTAYIAKQR";
    let motif = "ACGACG";
    let finite_beams = [
        Beam::flat(4),
        Beam {
            per_open: 1,
            cap: 64,
            cai_bound: false,
            mfe_bound: true,
            ..Beam::exact()
        },
        Beam {
            per_open: 2,
            cap: 64,
            cai_bound: false,
            mfe_bound: true,
            ..Beam::exact()
        },
    ];
    for &lambda in &LAMBDAS {
        let exact = design_with_motifs(
            &params,
            protein,
            &usage,
            lambda,
            &[motif],
            ScanConfig::default(),
        )
        .expect("feasible constrained set folds");
        for beam in finite_beams {
            let got = design_with_motifs(
                &params,
                protein,
                &usage,
                lambda,
                &[motif],
                ScanConfig {
                    beam,
                    ..ScanConfig::default()
                },
            )
            .expect("finite beam preserves feasibility");
            assert!(
                !got.sequence.contains(motif),
                "finite beam {beam:?} spelled a forbidden motif: {} [{protein} λ={lambda}]",
                got.sequence,
            );
            assert!(
                translates(&got.sequence, protein),
                "finite beam {beam:?} spelled a non-coding sequence: {} [{protein} λ={lambda}]",
                got.sequence,
            );
            assert!(
                brd(&got, lambda) >= brd(&exact, lambda) - 1e-6,
                "finite beam {beam:?} beat the exact optimum [{protein} λ={lambda}]: \
                 beam={} exact={}",
                brd(&got, lambda),
                brd(&exact, lambda),
            );
        }
    }
}
