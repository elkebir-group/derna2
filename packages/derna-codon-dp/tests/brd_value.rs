//! `brd_value` is exact arithmetic, and the DP depends on it being so.
//!
//! The objective is stored in a table and then minimized over, so a
//! quantized `brd_value` does not merely blur a reported score — it decides
//! which candidate wins. These tests fence that: the typical triples pin the
//! arithmetic, and `distinct_candidates_stay_distinguishable` pins the
//! property the arithmetic exists to provide.

use derna_codon_dp::testing::brd_value;

#[test]
fn computes_the_weighted_combination_exactly() {
    assert_eq!(brd_value(0.5, -1500, -200.0), -650.0);
    assert_eq!(brd_value(0.0, -1500, -200.0), 200.0);
    assert_eq!(brd_value(1.0, -1500, -200.0), -1500.0);
    assert_eq!(brd_value(0.25, -800, -100.0), -125.0);
    // Lands on a half: 0.75·(−1200) − 0.25·(−150) = −900 + 37.5.
    assert_eq!(brd_value(0.75, -1200, -150.0), -862.5);
}

#[test]
fn distinct_candidates_stay_distinguishable() {
    // Two candidates one dekacal apart in MFE. Quantizing the combination
    // maps both to the same stored value, and a `min` over the two then
    // picks by position rather than by objective — which is how a coarse
    // combination selects a design that is not the optimum.
    let lower = brd_value(0.5, 3, 0.0);
    let higher = brd_value(0.5, 5, 0.0);
    assert_eq!(lower, 1.5);
    assert_eq!(higher, 2.5);
    assert!(
        lower < higher,
        "candidates one dekacal apart must order strictly: {lower} vs {higher}",
    );
}

#[test]
fn halves_survive_on_both_signs() {
    assert_eq!(brd_value(0.5, 1, 0.0), 0.5);
    assert_eq!(brd_value(0.5, 7, 0.0), 3.5);
    assert_eq!(brd_value(0.5, -1, 0.0), -0.5);
    assert_eq!(brd_value(0.5, -3, 0.0), -1.5);
}

#[test]
fn the_cai_term_carries_its_fraction() {
    // log-CAI has no quantum of its own, so a fractional charge reaches the
    // objective intact rather than being snapped to a dekacal grid.
    assert_eq!(brd_value(0.5, 0, -1.5), 0.75);
    assert_eq!(brd_value(0.5, 0, -0.25), 0.125);
}

#[test]
fn lambda_extremes() {
    // λ = 0: value = −(1−0)·cai. λ = 1: value = mfe.
    assert_eq!(brd_value(0.0, -100_000, -42.0), 42.0);
    assert_eq!(brd_value(1.0, -100_000, -42.0), -100_000.0);
}
