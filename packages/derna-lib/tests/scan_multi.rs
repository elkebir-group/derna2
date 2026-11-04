//! One traversal at several weights fills each weight's tables exactly as
//! its own scan does, and the cost to go read off them is the same.
//!
//! Each lane must keep, in every cell, the `(mfe, cai)` its single scan keeps,
//! bit for bit: a weight's choice among candidates is made in its own lane, in
//! the single scan's order and under its tie rule.

use derna_lib::outside_multi::Instance;
use derna_lib::{Energy, Enforcement, PairClass, RegionConstraints};

/// Weights from the motif walk's pass grid, its two ends included.
const WEIGHTS: [f64; 6] = [0.0, 0.02, 0.16, 0.41, 0.71, 1.0];

fn lane_sets() -> Vec<Vec<f64>> {
    let mut sets: Vec<Vec<f64>> = WEIGHTS.iter().map(|&w| vec![w]).collect();
    sets.push(vec![WEIGHTS[0], WEIGHTS[5]]);
    sets.push(vec![WEIGHTS[2], WEIGHTS[3]]);
    sets.push(WEIGHTS[..4].to_vec());
    sets.push(WEIGHTS[2..].to_vec());
    sets
}

struct Case {
    protein: &'static str,
    utr5: &'static str,
    utr3: &'static str,
    constraints: RegionConstraints,
}

/// Coding sequences long enough to reach every skewed and one-sided
/// reduction, and flanked ones under a penalizing and a forbidding table.
fn cases() -> Vec<Case> {
    let penalize = RegionConstraints::symmetric_isolation(Enforcement::Penalize(Energy(50)));
    let forbid = RegionConstraints::five_prime_isolation(Enforcement::Forbid)
        .with(PairClass::Utr5Utr3, Enforcement::Forbid);
    let plain = RegionConstraints::default();
    let case = |protein, utr5, utr3, constraints| Case {
        protein,
        utr5,
        utr3,
        constraints,
    };
    vec![
        case("MYGKIIFVLM", "", "", plain),
        case("MKTAYIAKQRQISFVKSHFSRQ", "", "", plain),
        case("MAKKL", "GAGCGGACUUCGGUCCA", "ACAGGGAAACCUGGCUC", penalize),
        case("MAKKL", "GAGCGGACUUCGGUCCA", "ACAGGGAAACCUGGCUC", forbid),
    ]
}

fn check(case: &Case, lambdas: &[f64]) -> usize {
    let flanks = (case.utr5, case.utr3);
    let single = Instance::scan(case.protein, flanks, case.constraints, lambdas, 1)
        .expect("a valid instance")
        .expect("the lattice admits a design");
    let (together, _) = Instance::scan_together(case.protein, flanks, case.constraints, lambdas)
        .expect("a valid instance")
        .expect("the lattice admits a design");
    let what = format!("{} at {lambdas:?}", case.protein);
    if let Some(diff) = together.first_table_difference(&single) {
        panic!("{what}: {diff}");
    }
    together
        .same_cells()
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let (fused, _) = together.fused(1);
    for (l, (a, b)) in single.separate(1).iter().zip(&fused).enumerate() {
        if let Some(diff) = a.first_difference(b) {
            panic!("{what}, lane {l}: cost to go {diff}");
        }
    }
    single.table_cells()
}

#[test]
fn every_lane_fills_its_own_scans_tables() {
    let (mut cells, mut sweeps) = (0, 0);
    for case in cases() {
        for lambdas in lane_sets() {
            cells += check(&case, &lambdas);
            sweeps += 1;
        }
    }
    eprintln!("{sweeps} traversals, {cells} lane cells");
    assert_eq!(sweeps, 4 * 10);
    assert!(cells > 0);
}

/// The lanes are not one scan copied: two weights keep different values.
#[test]
fn different_weights_fill_different_tables() {
    let (a, _) = Instance::scan_together(
        "MKTAYIAKQRQISF",
        ("", ""),
        RegionConstraints::default(),
        &[0.16, 0.16],
    )
    .expect("a valid instance")
    .expect("the lattice admits a design");
    let (b, _) = Instance::scan_together(
        "MKTAYIAKQRQISF",
        ("", ""),
        RegionConstraints::default(),
        &[0.16, 0.71],
    )
    .expect("a valid instance")
    .expect("the lattice admits a design");
    assert!(a.first_table_difference(&b).is_some());
}
