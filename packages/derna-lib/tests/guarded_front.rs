//! The whole front of a flanked molecule under a hard region constraint: every
//! Pareto design of (constrained energy, coding CAI), where a design's constrained
//! energy is its best structure among those forming no forbidden pair.
//!
//! Checked three ways, because the enumeration oracle folds each molecule
//! under the same region constraint the walk uses and so cannot see a fault in the
//! constraint itself: against that oracle; against a pair classifier written here
//! from the region boundaries alone; and on a flank built to pair with the
//! start of the coding sequence, so that the constraint is known to bind.

mod common;

use common::{eliminate, front_objectives, Objective};
use derna_codon_dp::codon_table::aa_to_index;
use derna_lib::free_fold::{sequence_gap, CdsWindow};
use derna_lib::{
    design_front_walk, design_front_walk_with_utr, dfa, BeamError, CodonUsage, Energy, Enforcement,
    FrontPoint, Params, PassGrid, RegionConstraints, UtrSpec, WalkOptions, WalkedFront,
};

/// A protein whose coding sequence starts `AUG UGG UGG`, which every encoding
/// shares.
const PROTEIN: &str = "MWWKFYCLMW";
/// The reverse complement of `AUGUGGUGG`: it can close nine Watson–Crick pairs
/// with the start of every coding sequence, so a 5′↔CDS constraint has something to
/// forbid on every encoding.
const FLANK5: &str = "CCACCACAU";
const FLANK3: &str = "AUCGA";

fn fixtures() -> (Params, CodonUsage) {
    (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    )
}

fn fixed(s: &str) -> UtrSpec {
    UtrSpec::Fixed(s.to_string())
}

/// The constraints checked: all three crossing classes, the two flank-to-CDS
/// classes, and the 5′ boundary alone.
fn constraints() -> [(&'static str, RegionConstraints); 3] {
    [
        (
            "full",
            RegionConstraints::full_mutual_isolation(Enforcement::Forbid),
        ),
        (
            "symmetric",
            RegionConstraints::symmetric_isolation(Enforcement::Forbid),
        ),
        (
            "five-prime",
            RegionConstraints::five_prime_isolation(Enforcement::Forbid),
        ),
    ]
}

/// Refined to adjacency on one and four workers, with every lever that changes
/// how a gap is cut, and with the pass grid seeding the sweep.
fn plans() -> [(usize, WalkOptions); 4] {
    let levers = WalkOptions {
        stop_slack: None,
        slice_slack: Some(5.0),
        slice_width: None,
        uncut_within: Some(30),
        chord_bracket: Some(1.0),
        cut_probes: true,
        ..WalkOptions::default()
    };
    // The pass grid cuts probes from its own scans and seeds the sweep from
    // their optima, so the constraint must reach those scans too.
    let gridded = WalkOptions {
        pass_grid: Some(PassGrid::two_step(0.1, 0.25, 0.3)),
        seed_sweep: true,
        cut_probes: true,
        ..WalkOptions::default()
    };
    [
        (1, WalkOptions::default()),
        (4, WalkOptions::default()),
        (4, levers),
        (4, gridded),
    ]
}

fn walk(
    params: &Params,
    usage: &CodonUsage,
    protein: &str,
    constraints: RegionConstraints,
    (threads, options): (usize, WalkOptions),
) -> WalkedFront {
    design_front_walk_with_utr(
        params,
        protein,
        usage,
        &fixed(FLANK5),
        &fixed(FLANK3),
        constraints,
        1e-4,
        0.999,
        options,
        threads,
    )
    .expect("the constrained walk answers")
}

/// Every encoding of `protein` with its coding CAI, as the lattice spells it.
fn encodings(usage: &CodonUsage, protein: &str) -> Vec<(f64, String)> {
    let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
    let graph = dfa::build(&aa, usage);
    graph
        .paths_between(graph.start(), graph.end())
        .into_iter()
        .map(|(cai_log, seq)| {
            let cds = seq
                .iter()
                .map(|&nt| match nt as u8 {
                    0 => 'A',
                    1 => 'C',
                    2 => 'G',
                    _ => 'U',
                })
                .collect();
            (cai_log, cds)
        })
        .collect()
}

/// The constrained front by enumeration: each whole molecule folded alone under
/// `constraints`, then Pareto-reduced.
fn oracle(
    params: &Params,
    usage: &CodonUsage,
    protein: &str,
    constraints: RegionConstraints,
) -> Vec<Objective> {
    let window = CdsWindow::new(FLANK5.len(), protein.len());
    let bag = encodings(usage, protein)
        .into_iter()
        .map(|(cai_log, cds)| {
            let molecule = format!("{FLANK5}{cds}{FLANK3}");
            let gap = sequence_gap(params, &molecule, window, constraints).unwrap();
            Objective {
                mfe: (gap.constrained_mfe_kcal_mol * 100.0).round() as i32,
                cai_log,
            }
        })
        .collect();
    eliminate(bag)
}

/// The pairs of a dot-bracket structure, `(i, j)` with `i < j`.
fn pairs(structure: &str) -> Vec<(usize, usize)> {
    let mut open = Vec::new();
    let mut out = Vec::new();
    for (j, c) in structure.chars().enumerate() {
        match c {
            '(' => open.push(j),
            ')' => out.push((open.pop().expect("balanced"), j)),
            _ => {}
        }
    }
    out
}

/// The pairs of `point` joining two different regions — 5′ flank, coding
/// sequence, 3′ flank — told apart by position alone, with no region constraint.
fn crossing_pairs(point: &FrontPoint, protein: &str) -> Vec<(usize, usize)> {
    let cds_start = FLANK5.len();
    let cds_end = cds_start + 3 * protein.len();
    let region = |p: usize| usize::from(p >= cds_start) + usize::from(p >= cds_end);
    pairs(&point.structure)
        .into_iter()
        .filter(|&(i, j)| region(i) != region(j))
        .collect()
}

#[test]
fn the_constrained_walk_is_exact_against_enumeration() {
    let (params, usage) = fixtures();
    // The two small lattices under every constraint; the largest under the full
    // constraint alone, since its enumeration is most of the test's cost.
    let cases = [("MWWKFYCLMW", 3), ("MCWFGGWCKM", 3), ("MYGKIIFVLM", 1)];
    for (protein, count) in cases {
        for (name, constraint) in constraints().into_iter().take(count) {
            let expected = oracle(&params, &usage, protein, constraint);
            for plan in plans() {
                let walked = walk(&params, &usage, protein, constraint, plan.clone());
                let label = format!("{protein}, {name} constraint, {plan:?}");
                common::assert_same(&label, &expected, &front_objectives(&walked.front));
            }
        }
    }
}

/// On fronts wide enough to have several hull gaps, the constrained walk refined
/// gap by gap equals the same walk solved as one gap from end to end — the
/// single-gap solve being what the enumeration checks above.
#[test]
fn the_constrained_walk_is_the_same_front_gap_by_gap_and_in_one_gap() {
    let (params, usage) = fixtures();
    let full = RegionConstraints::full_mutual_isolation(Enforcement::Forbid);
    let one_gap = WalkOptions {
        stop_slack: Some(1e12),
        ..WalkOptions::default()
    };
    let mut walked_gaps = 0;
    for protein in common::WIDE_CASES {
        let whole = walk(&params, &usage, protein, full, (1, one_gap.clone()));
        assert!(whole.gaps.len() <= 1, "{protein}: one gap end to end");
        let want = front_objectives(&whole.front);
        for plan in plans() {
            let walked = walk(&params, &usage, protein, full, plan.clone());
            let label = format!("{protein} gap by gap, {plan:?}");
            common::assert_same(&label, &want, &front_objectives(&walked.front));
            walked_gaps = walked_gaps.max(walked.gaps.len());
        }
    }
    assert!(walked_gaps > 1, "no constrained walk refined past one gap");
}

/// Under all three crossing classes no emitted structure joins two regions,
/// by a classifier that shares no code with the constraint, and each structure
/// evaluates to the energy its point reports.
#[test]
fn no_constrained_point_forms_a_forbidden_pair() {
    let (params, usage) = fixtures();
    let full = RegionConstraints::full_mutual_isolation(Enforcement::Forbid);
    for protein in [PROTEIN, "MWWKFYCLMWYGKIIF"] {
        for plan in plans() {
            let walked = walk(&params, &usage, protein, full, plan.clone());
            assert!(!walked.front.is_empty());
            for point in &walked.front {
                let crossing = crossing_pairs(point, protein);
                assert!(
                    crossing.is_empty(),
                    "{protein} {plan:?}: {} forms {crossing:?}",
                    point.structure
                );
                let evaluated =
                    rna_mfe::evaluate(&params, point.sequence.as_bytes(), &point.structure)
                        .unwrap();
                assert_eq!(
                    evaluated.0,
                    (point.mfe_kcal_mol * 100.0).round() as i32,
                    "{protein}: {}",
                    point.sequence
                );
            }
        }
    }
}

/// The constructed flank pairs with the coding sequence when nothing forbids
/// it, so the constraint binds on this case: the unconstrained front emits a 5′↔CDS
/// pair, and the constrained front differs from it.
#[test]
fn the_constraint_binds_on_the_constructed_flank() {
    let (params, usage) = fixtures();
    let free = walk(
        &params,
        &usage,
        PROTEIN,
        RegionConstraints::default(),
        (1, WalkOptions::default()),
    );
    let crossing: usize = free
        .front
        .iter()
        .map(|p| crossing_pairs(p, PROTEIN).len())
        .sum();
    assert!(
        crossing > 0,
        "the unconstrained front forms no crossing pair, so no constraint would bind"
    );
    let constrained = walk(
        &params,
        &usage,
        PROTEIN,
        RegionConstraints::full_mutual_isolation(Enforcement::Forbid),
        (1, WalkOptions::default()),
    );
    let (free, constrained) = (
        front_objectives(&free.front),
        front_objectives(&constrained.front),
    );
    assert_ne!(
        free.iter().map(|o| o.mfe).collect::<Vec<_>>(),
        constrained.iter().map(|o| o.mfe).collect::<Vec<_>>(),
        "the constraint changed nothing"
    );
    assert!(
        free[0].mfe < constrained[0].mfe,
        "the constrained optimum is not above the free one"
    );
}

/// The fold's MFE of `sequence` alone, by the free folder.
fn alone(params: &Params, sequence: &str) -> i32 {
    let gap = sequence_gap(
        params,
        sequence,
        CdsWindow::new(sequence.len(), 0),
        RegionConstraints::default(),
    )
    .unwrap();
    (gap.free_mfe_kcal_mol * 100.0).round() as i32
}

/// With all three crossing classes forbidden, no pair leaves its region, and
/// at `dangles = 0` the exterior loop scores its branches independently. So a
/// design's constrained energy is the 5′ flank's MFE alone, plus its coding
/// sequence's own, plus the 3′ flank's: the constrained front is the coding-only
/// front moved by a constant, point for point, with the same CAI.
#[test]
fn the_fully_isolated_front_is_the_coding_front_shifted_by_the_flanks() {
    let (params, usage) = fixtures();
    let full = RegionConstraints::full_mutual_isolation(Enforcement::Forbid);
    // Two flanks that fold on their own, so the shift is not zero.
    let (five, three) = ("GGGCGAAAGCCCA", "AGCGCAAAAGCGC");
    let shift = alone(&params, five) + alone(&params, three);
    assert!(
        shift < 0,
        "flanks that fold to zero would make this vacuous"
    );
    for protein in common::CASES.iter().chain(common::WIDE_CASES.iter()) {
        let coding = design_front_walk(
            &params,
            protein,
            &usage,
            1e-4,
            0.999,
            WalkOptions::default(),
            1,
        )
        .unwrap();
        let constrained = design_front_walk_with_utr(
            &params,
            protein,
            &usage,
            &fixed(five),
            &fixed(three),
            full,
            1e-4,
            0.999,
            WalkOptions::default(),
            1,
        )
        .unwrap();
        let want: Vec<Objective> = front_objectives(&coding.front)
            .into_iter()
            .map(|o| Objective {
                mfe: o.mfe + shift,
                cai_log: o.cai_log,
            })
            .collect();
        common::assert_same(protein, &want, &front_objectives(&constrained.front));
    }
}

/// A soft constraint is refused: the front fold has no penalty path.
#[test]
fn the_walk_refuses_a_soft_constraint() {
    let (params, usage) = fixtures();
    let refused = design_front_walk_with_utr(
        &params,
        PROTEIN,
        &usage,
        &fixed(FLANK5),
        &fixed(FLANK3),
        RegionConstraints::full_mutual_isolation(Enforcement::Penalize(Energy(100))),
        1e-4,
        0.999,
        WalkOptions::default(),
        1,
    );
    assert!(
        matches!(refused, Err(BeamError::SoftRegionUnsupported)),
        "a soft constraint must be refused"
    );
}
