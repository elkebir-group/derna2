//! The single-branch loop cap at its boundary, against the reference minimizer.
//!
//! A single-branch (bulge / internal) loop may carry `n1 + n2 <= SINGLE_MAX_LEN`
//! unpaired bases in total — `rna_mfe`'s `MAX_LOOP_SIZE`, 30 by default.
//! The scan splits these loops across several families (the push closures, the
//! direct pull, the two one-sided reductions, the skewed size sweep), each with
//! its own bound on the flanks, so the cap is checked per split rather than once.
//!
//! The template `GGGGG · A^n1 · core · A^n2 · CCCCC` makes the outer helix close
//! a single-branch loop of exactly `n1 + n2` around a stable hairpin. For every
//! split at totals 29, 30 and 31 the scan must reach the reference minimizer's
//! cost: below the cap both admit the loop, above it neither does. A bound one
//! short of the cap reads the cost high at 30; one past it reads the cost low at
//! 31, which would break every certification that rests on the scan.

use derna_lib::free_fold::{sequence_constrained_mfe, CdsWindow};
use derna_lib::region::{Enforcement, RegionConstraints};
use derna_lib::{Params, DEFAULT_SINGLE_MAX_LEN};

/// A hairpin stable enough that the outer helix closes around it.
const CORE: &str = "GGGCGCGAAAGCGCCC";

/// The template at one split, padded with exterior `A`s to whole codons.
fn template(n1: usize, n2: usize) -> String {
    let mut s = format!("GGGGG{}{CORE}{}CCCCC", "A".repeat(n1), "A".repeat(n2));
    while s.len() % 3 != 0 {
        s.push('A');
    }
    s
}

fn scan_cost(params: &Params, seq: &str) -> i32 {
    let rc = RegionConstraints::full_mutual_isolation(Enforcement::Forbid);
    let kcal = sequence_constrained_mfe(params, seq, CdsWindow::new(0, seq.len() / 3), rc)
        .expect("scan folds");
    (kcal * 100.0).round() as i32
}

#[test]
fn the_scan_admits_exactly_the_reference_minimizers_loops() {
    let params = Params::load_default().expect("Turner 2004 parameters");
    let cap = DEFAULT_SINGLE_MAX_LEN;
    let mut used = 0;
    let mut mismatches = Vec::new();
    for total in [cap - 1, cap, cap + 1] {
        for n1 in 0..=total {
            let seq = template(n1, total - n1);
            let (structure, reference) =
                rna_mfe::fold(&params, seq.as_bytes()).expect("reference folds");
            used += usize::from(structure.starts_with("((((("));
            let scan = scan_cost(&params, &seq);
            if scan != reference.0 {
                mismatches.push(format!(
                    "n1={n1} n2={} scan={scan} reference={}",
                    total - n1,
                    reference.0
                ));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "scan and reference disagree:\n{}",
        mismatches.join("\n")
    );
    // The template is only a boundary test if the outer helix actually closes the
    // loop in a good share of the cases.
    assert!(
        used >= 40,
        "the outer helix closed the loop in only {used} cases"
    );
}

/// A branching loop's unpaired runs are energy-free and unbounded in the energy
/// model, on either side of its branches. The template
/// `GGGGG · A^lead · core · AA · core · A^trail · CCCCC` closes a two-branch loop
/// whose leading run crosses the single-branch cap; the scan must reach the
/// reference minimizer's cost whatever the leading run's length.
#[test]
fn a_branching_loops_leading_run_is_not_capped() {
    let params = Params::load_default().expect("Turner 2004 parameters");
    let mut used = 0;
    let mut mismatches = Vec::new();
    for trail in [2, 35] {
        for lead in 25..=40 {
            let mut seq = format!(
                "GGGGG{}{CORE}AA{CORE}{}CCCCC",
                "A".repeat(lead),
                "A".repeat(trail)
            );
            while seq.len() % 3 != 0 {
                seq.push('A');
            }
            let (structure, reference) =
                rna_mfe::fold(&params, seq.as_bytes()).expect("reference folds");
            used += usize::from(structure.starts_with("((((("));
            let scan = scan_cost(&params, &seq);
            if scan != reference.0 {
                mismatches.push(format!(
                    "lead={lead} trail={trail} scan={scan} reference={}",
                    reference.0
                ));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "scan and reference disagree:\n{}",
        mismatches.join("\n")
    );
    assert!(
        used >= 16,
        "the outer helix closed the loop in only {used} cases"
    );
}
