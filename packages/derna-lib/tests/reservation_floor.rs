//! The per-open reservation floor, and what separates it from the ceiling.
//!
//! [`GroupPolicy::Limit`] reads `per_open` as a ceiling: at most that many cells
//! survive per 5′ open node, and the column-wide cap then applies among the
//! survivors, so a group's best cells are still evictable.
//! [`GroupPolicy::Reserve`] reads it as a floor: those cells survive the cap
//! unconditionally and the rest of the budget is filled by global rank.
//!
//! Three properties anchor the floor. A width above every column size leaves the
//! exact optimum untouched, as it must for any pruning policy. A finite width
//! never returns more cells than the cap allows, because reservations are
//! budgeted rather than added on top. And the two policies **actually differ** —
//! a mechanism indistinguishable from the one it replaces would make every other
//! test here vacuous, so the difference is asserted directly rather than assumed.

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::{design, design_with_config, Beam, CodonUsage, GroupPolicy, Params, ScanConfig};

/// A 120-residue prefix of the SARS-CoV-2 spike, the protein the ladder is cut
/// from. Long enough that an open-node group holds more cells than `per_open`,
/// which is the condition under which the policy does anything at all.
const REGRESSION_PROTEIN: &str = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQS";

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

fn reserve(cap: usize, per_open: usize) -> Beam {
    Beam {
        per_open,
        cap,
        cai_bound: false,
        mfe_bound: true,
        group: GroupPolicy::Reserve,
        ..Beam::exact()
    }
}

fn limit(cap: usize, per_open: usize) -> Beam {
    Beam {
        per_open,
        cap,
        cai_bound: false,
        mfe_bound: true,
        group: GroupPolicy::Limit,
        ..Beam::exact()
    }
}

fn solve(params: &Params, usage: &CodonUsage, protein: &str, lambda: f64, beam: Beam) -> String {
    design_with_config(
        params,
        protein,
        usage,
        lambda,
        ScanConfig {
            beam,
            ..ScanConfig::default()
        },
    )
    .expect("a finite width still returns a design")
    .sequence
}

/// The default policy is the ceiling, so an unchanged call prunes as it did.
#[test]
fn the_default_policy_is_the_ceiling() {
    assert_eq!(Beam::exact().group, GroupPolicy::Limit);
    assert_eq!(Beam::flat(100).group, GroupPolicy::Limit);
    assert_eq!(Beam::from(100usize).group, GroupPolicy::Limit);
}

/// A width above every column size prunes nothing, whichever reading of
/// `per_open` is in force.
#[test]
fn a_wide_reservation_equals_exact() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    for protein in ["GGGGCCCC", "MKTAYIAKQR", "DEYWCNQHKR"] {
        for lambda in LAMBDAS {
            let exact = design(&params, protein, &usage, lambda).unwrap();
            for per_open in [1usize, 2, 4] {
                let wide = solve(&params, &usage, protein, lambda, reserve(100_000, per_open));
                assert_eq!(
                    wide, exact.sequence,
                    "a reservation at an unreachable width moved the optimum \
                     ({protein}, λ={lambda}, per_open={per_open})"
                );
            }
        }
    }
}

/// A finite width still returns a valid design under the floor, at a length the
/// small-protein contract tests structurally cannot reach.
///
/// `beam_pruning.rs` runs 8-10 residues, where an open-node group rarely holds a
/// third cell and neither the per-open budget nor the group budget ever binds —
/// so it cannot observe the policy at all. At 100 residues both bind, which is
/// the regime the shipped mode operates in.
#[test]
fn a_finite_floor_returns_a_valid_design_at_length() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let protein = &REGRESSION_PROTEIN[..100];
    for (cap, per_open) in [(10usize, 2usize), (100, 2), (100, 4)] {
        for lambda in [0.5, 0.99] {
            let seq = solve(&params, &usage, protein, lambda, reserve(cap, per_open));
            assert!(
                translates(&seq, protein),
                "the floor returned an untranslatable design at cap={cap}, \
                 per_open={per_open}, λ={lambda}"
            );
        }
    }
}

/// The floor never beats the exact optimum, at a length where it binds.
///
/// A beam optimizes over a subset of the same feasible set, so a design better
/// than the exact one is a defect rather than a finding — it would mean the
/// pruned lattice admits something the full one does not. Asserted at 100
/// residues because that is where the policy actually engages, and the exact arm
/// is still affordable in a test.
#[test]
fn the_floor_never_beats_exact_at_length() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let protein = &REGRESSION_PROTEIN[..100];
    let lambda = 0.99;
    let exact = design(&params, protein, &usage, lambda).unwrap();
    let mut bound = false;
    for (cap, per_open) in [(10usize, 2usize), (100, 2)] {
        let d = design_with_config(
            &params,
            protein,
            &usage,
            lambda,
            ScanConfig {
                beam: reserve(cap, per_open),
                ..ScanConfig::default()
            },
        )
        .expect("a finite width still returns a design");
        let loss = lambda * (d.mfe_kcal_mol - exact.mfe_kcal_mol)
            - (1.0 - lambda) * (d.cai_log - exact.cai_log);
        assert!(
            loss >= -1e-9,
            "the floor beat the exact optimum by {loss} at cap={cap}, \
             per_open={per_open}, which is impossible over a subset"
        );
        if d.sequence != exact.sequence {
            bound = true;
        }
    }
    assert!(
        bound,
        "neither width changed the design at 100 residues, so this test would \
         pass against a policy that pruned nothing"
    );
}

/// The floor and the ceiling are different mechanisms, and the difference shows.
///
/// Without this every other assertion here could pass against a policy that
/// silently did nothing. The disagreement is looked for across a small grid
/// rather than pinned to one cell, because which cells differ depends on where
/// the cap binds — but that *some* cell differs is the property being asserted.
#[test]
fn the_floor_and_the_ceiling_disagree_somewhere() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let protein = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK";
    let mut differed = 0;
    for cap in [4usize, 10, 25] {
        for per_open in [2usize, 4] {
            for lambda in LAMBDAS {
                let a = solve(&params, &usage, protein, lambda, limit(cap, per_open));
                let b = solve(&params, &usage, protein, lambda, reserve(cap, per_open));
                assert!(translates(&a, protein) && translates(&b, protein));
                if a != b {
                    differed += 1;
                }
            }
        }
    }
    assert!(
        differed > 0,
        "the reservation floor never changed the design on any cell of the grid, \
         so it is indistinguishable from the ceiling it replaces and every other \
         assertion in this file is vacuous"
    );
}

/// An explicit group budget overrides the derived one, and the derived one is
/// what a cap and a per-open budget already imply together.
#[test]
fn the_group_budget_is_derived_unless_given() {
    assert_eq!(reserve(100, 2).group_budget(), 50);
    assert_eq!(reserve(100, 4).group_budget(), 25);
    // A cap narrower than one group's reservation still admits one group, so the
    // best group is protected rather than nothing being reserved at all.
    assert_eq!(reserve(3, 8).group_budget(), 1);
    let explicit = Beam {
        max_groups: 7,
        ..reserve(100, 2)
    };
    assert_eq!(explicit.group_budget(), 7, "an explicit budget wins");
    // The ceiling has no reservation to bound, and the exact lattice no limit.
    assert_eq!(Beam::exact().group_budget(), usize::MAX);
}

/// The length-scaled width reproduces the widths the calibration measured.
///
/// These are not free parameters: each is the narrowest width observed to hold a
/// design within 1.0 kcal/mol of the exact optimum at that length, on 56 ladder
/// lengths and again on a dense width grid. A change here is a claim about the
/// substrate and needs a re-measurement, not a re-tune.
#[test]
fn the_length_scaled_width_matches_the_calibration() {
    // Below the first breakpoint the cap must still bind: a wider width here
    // never prunes, and an explicitly approximate run then pays the exact cost.
    assert_eq!(derna_lib::recommended_cap(94), 100);
    assert_eq!(derna_lib::recommended_cap(139), 100);
    // The measured steps.
    assert_eq!(derna_lib::recommended_cap(174), 500);
    assert_eq!(derna_lib::recommended_cap(255), 1000);
    assert_eq!(derna_lib::recommended_cap(347), 2000);
    assert_eq!(derna_lib::recommended_cap(1273), 2000);
    assert_eq!(derna_lib::recommended_cap(1495), 5000);
    // Monotone: a longer protein never gets a narrower width.
    let mut last = 0;
    for n in (1..3000).step_by(7) {
        let c = derna_lib::recommended_cap(n);
        assert!(c >= last, "width fell from {last} to {c} at n={n}");
        last = c;
    }
}

/// The recommended operating point pairs that width with the ceiling reading of
/// the per-open budget, which the grid measured as the better of the two at the
/// widths this policy selects.
#[test]
fn the_recommended_point_is_the_ceiling() {
    let b = Beam::recommended(347);
    assert_eq!(b.cap, 2000);
    assert_eq!(b.per_open, 2);
    assert!(b.mfe_bound && !b.cai_bound);
    assert_eq!(b.group, GroupPolicy::Limit);
    assert!(!b.is_exact());
}
