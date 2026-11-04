//! One backward sweep over several weights' scans gives each weight the cost
//! to go of its own pass, bit for bit.
//!
//! The sweep shares the walk, the key lookups and the energy evaluations
//! across its lanes and keeps only the values apart; each lane must equal the
//! single pass at that lane's weight in every cell, infinities included.

use derna_lib::outside_multi::Instance;
use derna_lib::{Energy, Enforcement, PairClass, RegionConstraints};

/// Weights from the motif walk's pass grid, its two ends included.
const WEIGHTS: [f64; 6] = [0.0, 0.02, 0.16, 0.41, 0.71, 1.0];

/// The lane sets each case is swept with: every weight alone, two pairs and
/// two quadruples.
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

fn cases() -> Vec<Case> {
    let penalize = RegionConstraints::symmetric_isolation(Enforcement::Penalize(Energy(50)));
    let forbid = RegionConstraints::five_prime_isolation(Enforcement::Forbid)
        .with(PairClass::Utr5Utr3, Enforcement::Forbid);
    vec![
        Case {
            protein: "MYGKIIFVLM",
            utr5: "",
            utr3: "",
            constraints: RegionConstraints::default(),
        },
        Case {
            protein: "MKTAYIAKQRQISF",
            utr5: "",
            utr3: "",
            constraints: RegionConstraints::default(),
        },
        Case {
            protein: "MAKKL",
            utr5: "GAGCGGACUUCGGUCCA",
            utr3: "ACAGGGAAACCUGGCUC",
            constraints: penalize,
        },
        Case {
            protein: "MAKKL",
            utr5: "GAGCGGACUUCGGUCCA",
            utr3: "ACAGGGAAACCUGGCUC",
            constraints: forbid,
        },
    ]
}

/// Sweeps `case` over `lambdas` and checks every lane against the single
/// pass; returns the cells compared and those with a finite value.
fn check(case: &Case, lambdas: &[f64], threads: usize) -> (usize, usize) {
    let inst = Instance::scan(
        case.protein,
        (case.utr5, case.utr3),
        case.constraints,
        lambdas,
        1,
    )
    .expect("a valid instance")
    .expect("the lattice admits a design");
    assert!(inst.all_exact(), "{}: a scan is not exact", case.protein);
    inst.same_cells()
        .unwrap_or_else(|e| panic!("{} at {lambdas:?}: {e}", case.protein));
    let single = inst.separate(1);
    let (fused, _) = inst.fused(threads);
    assert_eq!(single.len(), fused.len());
    let (mut cells, mut finite) = (0, 0);
    for (l, (a, b)) in single.iter().zip(&fused).enumerate() {
        if let Some(diff) = a.first_difference(b) {
            panic!(
                "{} lanes {lambdas:?}, lane {l} (λ = {}): {diff}",
                case.protein, lambdas[l]
            );
        }
        cells += a.cells();
        finite += a.finite_cells();
    }
    (cells, finite)
}

#[test]
fn every_lane_equals_its_own_pass_bit_for_bit() {
    let (mut cells, mut finite, mut sweeps) = (0, 0, 0);
    for case in cases() {
        for lambdas in lane_sets() {
            let (c, f) = check(&case, &lambdas, 1);
            cells += c;
            finite += f;
            sweeps += 1;
        }
    }
    eprintln!("{sweeps} sweeps, {cells} lane cells, {finite} finite");
    assert_eq!(sweeps, 4 * 10);
    assert!(finite > 0, "no finite cell was compared");
}

/// Worker threads split only the single-branch consumers, and the minimum
/// makes their split invisible: the threaded sweep matches too.
#[test]
fn threaded_lanes_equal_their_own_passes() {
    for case in cases().iter().take(2) {
        check(case, &WEIGHTS[..4], 3);
    }
}

/// The lanes are not one value copied: two weights give different costs to
/// go, so a sweep that served every lane the first lane's values would fail.
#[test]
fn different_weights_give_different_lanes() {
    let case = &cases()[1];
    let inst = Instance::scan(
        case.protein,
        (case.utr5, case.utr3),
        case.constraints,
        &[0.16, 0.71],
        1,
    )
    .expect("a valid instance")
    .expect("the lattice admits a design");
    let (fused, _) = inst.fused(1);
    assert!(fused[0].first_difference(&fused[1]).is_some());
}
