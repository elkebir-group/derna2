//! The clamp table bounds every design's free fold: `free ≤ G + D(B)`, where
//! `G` is the constrained energy and `B` the exterior branch count of a constrained-
//! optimal structure of the coding region.
//!
//! The table comes from constrained folds of probe molecules; the check here folds
//! each whole molecule with the independent Zuker folder (`rna_mfe::fold`), so
//! the two share no code beyond the energy model. Random coding regions over the
//! full alphabet, three flank pairs that clamp. The bound must hold everywhere
//! and be attained somewhere: an entry off by one branch (0.9 kcal/mol too
//! strong) fails on the attaining designs.

use derna_lib::clamp::clamp_table;
use derna_lib::free_fold::{sequence_constrained_mfe, CdsWindow};
use derna_lib::{Enforcement, Params, RegionConstraints};

/// Unpaired flank bases a clamp may leave before the coding region; a design
/// whose leading run is within `30 - CAP_MARGIN` keeps the clamp's multiloop
/// leading run under the scan's cap for the flanks below.
const CAP_MARGIN: usize = 8;

fn params() -> Params {
    Params::load_default().expect("bundled parameters")
}

/// A deterministic coding region of `len` nucleotides.
fn random_cds(state: &mut u64, len: usize) -> String {
    (0..len)
        .map(|_| {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            b"ACGU"[(*state >> 62) as usize] as char
        })
        .collect()
}

fn exterior_branches(structure: &str) -> usize {
    let mut depth = 0usize;
    let mut branches = 0;
    for c in structure.chars() {
        match c {
            '(' => {
                branches += usize::from(depth == 0);
                depth += 1;
            }
            ')' => depth -= 1,
            _ => {}
        }
    }
    branches
}

/// Designs checked, and how many of them the bound attains.
///
/// Two folds per design: the free fold (Zuker, independent of the table's
/// folder), and the fold that allows only `5′↔3′` among the crossing pairs,
/// which isolates the clamp so that the bound is attained often enough to
/// catch a table that is too strong.
fn check(utr5: &str, utr3: &str, designs: usize, seed: u64) -> (usize, usize) {
    let p = params();
    let table = clamp_table(&p, utr5, utr3, 16).unwrap();
    let mfe = |s: &str| rna_mfe::fold(&p, s.as_bytes()).unwrap();
    let flanks = mfe(utr5).1 .0 + mfe(utr3).1 .0;
    let clamp_only = RegionConstraints::symmetric_isolation(Enforcement::Forbid);
    let mut state = seed;
    let (mut checked, mut tight) = (0, 0);
    for k in 0..designs {
        let cds = random_cds(&mut state, 24 + 3 * (k % 25));
        let (structure, coding) = mfe(&cds);
        let b = exterior_branches(&structure);
        let Some(Some(d)) = table.get(b).copied() else {
            continue;
        };
        let bound = flanks + coding.0 + (d * 100.0).round() as i32;
        let mol = format!("{utr5}{cds}{utr3}");
        let free = mfe(&mol).1 .0;
        assert!(
            free <= bound,
            "B = {b}, D = {d}: free {free} > bound {bound} for {cds}"
        );
        let window = CdsWindow::new(utr5.len(), cds.len() / 3);
        let clamped = sequence_constrained_mfe(&p, &mol, window, clamp_only).unwrap();
        let clamped = (clamped * 100.0).round() as i32;
        // The scan caps a multiloop's leading unpaired run at 30, so it cannot
        // build the clamp around a design whose own leading run, with the
        // flank's, exceeds that; the Zuker fold above has no cap and is the
        // check. Where the scan can build it, it must reach the bound too.
        let lead = structure.find('(').unwrap_or(cds.len());
        assert!(
            clamped <= bound || lead > 30 - CAP_MARGIN,
            "B = {b}, D = {d}: clamp-only {clamped} > {bound} for {cds}"
        );
        checked += 1;
        tight += usize::from(clamped == bound && d < 0.0);
    }
    (checked, tight)
}

#[test]
fn the_clamp_table_bounds_the_free_fold() {
    let cases = [
        // GAPDH: a 5′↔3′ clamp at every branch count.
        (
            "GCUCUCUGCUCCUCCUGUUCGACAGUCAGCCGCAUCUUCUUUUGCGUCGCCAGCCGAGCCACAUCGCUCAGACACC",
            "GACCCCUGGACCACCAGCCCCAGCAAGAGCACAAGAGGAAGAGAGAGACCCUCACUGCUGGGGAGUCCCUGCCACACUCAGUCCCCCACCACACUGAAUCUCCCCUCCUCACAGUUGCCAUGUAGACCCCUUGAAGAGGGGAGGGGCCUAGGGAGCCGCACCUUGUCAUGUACCAUCAAUAAAGUACCCUGUGCUCAACCA",
        ),
        // A synthetic pair whose clamp holds two flank branches inside.
        (
            "GGGGCCGGGGCCAAAAAAGGGAAACCCAAAAAAGGGAAACCCAA",
            "AAGGGAAACCCAAAAAAGGGAAACCCAAAAAAGGCCCCGGCCCC",
        ),
        // ACTB's leader against a GC-rich 3′ flank.
        (
            "ACCGCCGAGACCGCGUCCGCCCCGCGAGCACAGAGCCUCGCCUUUGCCGAUCCGCCGCCCGUCCACACCCGCCGCCAGCUCACC",
            "GGAGGCGGAGGCGGA",
        ),
    ];
    let mut tight = 0;
    for (k, (utr5, utr3)) in cases.iter().enumerate() {
        let (checked, t) = check(utr5, utr3, 150, 17 + k as u64);
        eprintln!("case {k}: {checked} designs checked, bound attained on {t}");
        assert!(checked >= 100, "only {checked} designs had a table entry");
        tight += t;
    }
    assert!(
        tight > 0,
        "the bound was attained nowhere; an off-by-one table would pass"
    );
}

/// HBA1's flanks, whose clamp the scan's cap once hid from the table.
const HBA1: (&str, &str) = (
    "ACUCUUCUGGUCCCCACAGACUCAGAGAGAACCCACC",
    "GCUGGAGCCUCGGUGGCCAUGCUUCUUGCCCCUUGGGCCUCCCCCCAGCCCCUCCUCCCCUUCCUGCACCCGUACCCCCGUGGUCUUUGAAUAAAGUCUGAGUGGGCGGCA",
);

/// The table is the **uncapped** clamp: on a probe whose Zuker optimum pairs
/// only `5′↔3′` across the boundaries, `D(B)` equals that optimum's gain. A
/// stand-in with a leading run longer than the scan's multiloop cap reports
/// `0` for HBA1 at `B = 1..=2` where the uncapped clamp gains `-0.30` and
/// `-1.20`, and fails here.
#[test]
fn the_clamp_table_sees_the_uncapped_clamp() {
    let p = params();
    let (utr5, utr3) = HBA1;
    let table = clamp_table(&p, utr5, utr3, 3).unwrap();
    let mfe = |s: &str| rna_mfe::fold(&p, s.as_bytes()).unwrap();
    for (b, d) in table.iter().enumerate().skip(1) {
        let d = d.expect("B >= 1 is always reported");
        let mut cds = String::new();
        for _ in 0..b {
            cds.push_str("GGGCGCGAAAGCGCCC");
            cds.push_str(&"A".repeat(33));
        }
        cds.push_str(&"A".repeat((3 - cds.len() % 3) % 3));
        let mol = format!("{utr5}{cds}{utr3}");
        let (structure, free) = mfe(&mol);
        // only 5′↔3′ crossings: every pair with one end in the stand-in has
        // the other in it too
        let (lo, hi) = (utr5.len(), utr5.len() + cds.len());
        let mut stack = Vec::new();
        for (j, c) in structure.chars().enumerate() {
            match c {
                '(' => stack.push(j),
                ')' => {
                    let i = stack.pop().unwrap();
                    assert_eq!((lo..hi).contains(&i), (lo..hi).contains(&j), "{structure}");
                }
                _ => {}
            }
        }
        let apart = mfe(utr5).1 .0 + mfe(&cds).1 .0 + mfe(utr3).1 .0;
        let gain = f64::from(free.0 - apart) / 100.0;
        assert!(
            gain < 0.0,
            "B = {b}: the uncapped clamp gains nothing ({gain})"
        );
        assert!(
            (d - gain).abs() < 1e-9,
            "B = {b}: table {d}, uncapped clamp {gain}"
        );
    }
}
