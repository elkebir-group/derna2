//! `design_restricted_with_region_must`: the restricted, region-constrained exact
//! solve with positions required paired in the designed structure.
//!
//! 1. An empty requirement is `design_restricted_with_region`, byte for byte.
//! 2. A requirement binds: on a fixture whose unconstrained optimum leaves the
//!    required positions unpaired, every required position is paired in the
//!    result, and the objective is no better than the unconstrained one.

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::{
    design_restricted_with_region, design_restricted_with_region_must, Beam, CodonUsage, Design,
    Enforcement, Params, RegionConstraints, ScanConfig,
};

const PROTEIN: &str = "MKVLAGRSTEQWPLLAG";
const UTR5: &str = "GGACAUCCAGUC";
const UTR3: &str = "CUGAUCCAGG";
const LAMBDA: f64 = 0.5;

fn exact() -> ScanConfig {
    ScanConfig {
        beam: Beam::exact(),
        ..ScanConfig::default()
    }
}

fn full_lattice() -> Vec<Vec<[u8; 3]>> {
    PROTEIN
        .chars()
        .map(|c| {
            let a = aa_to_index(c).expect("a standard residue");
            AMINO_ACIDS[a].codons.iter().map(|c| c.nuc).collect()
        })
        .collect()
}

fn solve(must: &[u32]) -> Option<Design> {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    design_restricted_with_region_must(
        &params,
        PROTEIN,
        &usage,
        LAMBDA,
        &full_lattice(),
        (UTR5, UTR3),
        RegionConstraints::symmetric_isolation(Enforcement::Forbid),
        must,
        exact(),
    )
    .expect("solves")
}

fn objective(d: &Design) -> f64 {
    LAMBDA * (100.0 * d.mfe_kcal_mol).round() - (1.0 - LAMBDA) * 100.0 * d.cai_log
}

#[test]
fn an_empty_requirement_is_the_restricted_solve() {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let plain = design_restricted_with_region(
        &params,
        PROTEIN,
        &usage,
        LAMBDA,
        &full_lattice(),
        UTR5,
        UTR3,
        RegionConstraints::symmetric_isolation(Enforcement::Forbid),
        exact(),
    )
    .expect("solves")
    .expect("feasible");
    let must = solve(&[]).expect("feasible");
    assert_eq!(plain.sequence, must.sequence);
    assert_eq!(plain.structure, must.structure);
}

#[test]
fn required_positions_are_paired_and_cost_something() {
    let free = solve(&[]).expect("feasible");
    // The unconstrained optimum's unpaired positions inside the CDS.
    let start = UTR5.len();
    let unpaired: Vec<u32> = free.structure.as_bytes()[start..start + 3 * PROTEIN.len()]
        .iter()
        .enumerate()
        .filter(|(_, &c)| c == b'.')
        .map(|(k, _)| (start + k) as u32)
        .take(3)
        .collect();
    assert!(
        !unpaired.is_empty(),
        "the fixture's optimum pairs every CDS position"
    );
    let held = solve(&unpaired).expect("feasible");
    for &p in &unpaired {
        assert_ne!(
            held.structure.as_bytes()[p as usize],
            b'.',
            "required position {p} unpaired in {}",
            held.structure
        );
    }
    assert!(objective(&held) >= objective(&free) - 1e-9);
}

/// The constrained fold of a given sequence: with no requirement its energy is
/// the scan's constrained MFE; with one, every required position is paired
/// and the energy is no lower.
#[test]
fn the_constrained_fold_of_a_sequence_honours_the_requirement() {
    use derna_lib::{sequence_constrained_fold_must, sequence_constrained_mfe, CdsWindow};
    let params = Params::load_default().expect("params");
    let free = solve(&[]).expect("feasible");
    let seq = free.sequence.clone();
    let cds = CdsWindow::new(UTR5.len(), PROTEIN.len());
    let rc = RegionConstraints::symmetric_isolation(Enforcement::Forbid);
    let (g, e) = sequence_constrained_fold_must(&params, &seq, cds, rc, &[])
        .expect("folds")
        .expect("feasible");
    let scan = sequence_constrained_mfe(&params, &seq, cds, rc).expect("folds");
    assert!((e - scan).abs() < 1e-9, "{e} vs {scan}");
    let p = g
        .bytes()
        .position(|c| c == b'.')
        .expect("an unpaired position") as u32;
    let (held, e_held) = sequence_constrained_fold_must(&params, &seq, cds, rc, &[p])
        .expect("folds")
        .expect("feasible");
    assert_ne!(
        held.as_bytes()[p as usize],
        b'.',
        "position {p} unpaired in {held}"
    );
    assert!(e_held >= e - 1e-9);
}
