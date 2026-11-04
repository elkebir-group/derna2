//! The lazy walk with positions required paired, against exhaustive
//! enumeration.
//!
//! 1. The reference grammar's inside values equal the must-pair scan's, cell
//!    for cell, with the requirement on both sides.
//! 2. Walked to exhaustion, every encoding with a structure that pairs the
//!    required positions comes out exactly once, in the order of its best such
//!    structure, at that structure's energy; every structure handed out pairs
//!    them all.
//! 3. The first design that passes the check is the best one whose free MFE
//!    equals its best requirement-meeting structure's energy.
//!
//! Every case's requirement binds: it moves the optimum or removes encodings.

use std::collections::HashSet;

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::outside::lazy::{
    compare_inside_must, first_modular_must, walk_designs_must, Build, Verify, Walk,
};
use derna_lib::{
    design_restricted_with_region, design_restricted_with_region_must, sequence_free_mfe,
    CodonUsage, Design, Enforcement, Params, RegionConstraints, ScanConfig,
};

const TOL: f64 = 1e-6;

struct Case<'a> {
    protein: &'a str,
    utr5: &'a str,
    utr3: &'a str,
    lambda: f64,
    constraints: RegionConstraints,
    must: &'a [u32],
}

fn full_sets(protein: &str) -> Vec<Vec<[u8; 3]>> {
    protein
        .chars()
        .map(|c| {
            let a = aa_to_index(c).expect("a standard residue");
            AMINO_ACIDS[a].codons.iter().map(|c| c.nuc).collect()
        })
        .collect()
}

fn encodings(sets: &[Vec<[u8; 3]>]) -> Vec<Vec<[u8; 3]>> {
    let mut out = vec![Vec::new()];
    for set in sets {
        out = out
            .into_iter()
            .flat_map(|prefix| {
                set.iter().map(move |c| {
                    let mut e = prefix.clone();
                    e.push(*c);
                    e
                })
            })
            .collect();
    }
    out
}

fn brd(lambda: f64, d: &Design) -> f64 {
    lambda * (d.mfe_kcal_mol * 100.0) - (1.0 - lambda) * 100.0 * d.cai_log
}

/// `(objective, sequence, energy in dcal/mol)` of every encoding at its best
/// structure meeting `must` (or at its constrained MFE when `must` is empty),
/// ascending; encodings with no such structure are left out.
fn brute_force(
    params: &Params,
    usage: &CodonUsage,
    case: &Case<'_>,
    must: &[u32],
) -> Vec<(f64, String, i32)> {
    let mut out: Vec<(f64, String, i32)> = encodings(&full_sets(case.protein))
        .into_iter()
        .filter_map(|enc| {
            let pinned: Vec<Vec<[u8; 3]>> = enc.iter().map(|c| vec![*c]).collect();
            design_restricted_with_region_must(
                params,
                case.protein,
                usage,
                case.lambda,
                &pinned,
                (case.utr5, case.utr3),
                case.constraints,
                must,
                ScanConfig::default(),
            )
            .expect("a valid instance")
            .map(|d| {
                let e = (d.mfe_kcal_mol * 100.0).round() as i32;
                (brd(case.lambda, &d), d.sequence, e)
            })
        })
        .collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

/// The requirement changes the enumeration: some encoding loses every
/// structure or is scored higher.
fn assert_binds(params: &Params, usage: &CodonUsage, case: &Case<'_>) {
    let free = brute_force(params, usage, case, &[]);
    let must = brute_force(params, usage, case, case.must);
    let moved = must.len() < free.len()
        || must.iter().any(|(v, s, _)| {
            free.iter()
                .find(|(_, t, _)| t == s)
                .is_some_and(|(w, _, _)| v - w > TOL)
        });
    assert!(moved, "{}: the requirement binds nothing", case.protein);
    assert!(
        !must.is_empty(),
        "{}: no encoding meets the requirement",
        case.protein
    );
    eprintln!(
        "{}: {} of {} encodings meet {:?}",
        case.protein,
        must.len(),
        free.len(),
        case.must
    );
    // The empty requirement is the plain restricted solve.
    let plain = design_restricted_with_region(
        params,
        case.protein,
        usage,
        case.lambda,
        &full_sets(case.protein),
        case.utr5,
        case.utr3,
        case.constraints,
        ScanConfig::default(),
    )
    .expect("solves")
    .expect("a design");
    assert!((brd(case.lambda, &plain) - free[0].0).abs() < TOL);
}

fn cases() -> Vec<Case<'static>> {
    let sym = RegionConstraints::symmetric_isolation(Enforcement::Forbid);
    vec![
        // A UTR position and a CDS position.
        Case {
            protein: "MLR",
            utr5: "GGGAUC",
            utr3: "GAUCCC",
            lambda: 0.5,
            constraints: sym,
            must: &[2, 8],
        },
        // One CDS position.
        Case {
            protein: "MLR",
            utr5: "GGGAUC",
            utr3: "GAUCCC",
            lambda: 0.5,
            constraints: sym,
            must: &[10],
        },
        // Codons in loop interiors.
        Case {
            protein: "GLRS",
            utr5: "GGCGC",
            utr3: "GCGCC",
            lambda: 0.5,
            constraints: RegionConstraints::default(),
            must: &[6, 7],
        },
        // The coding sequence is a multi-loop's unpaired flank.
        Case {
            protein: "KK",
            utr5: "GAGCGGACUUCGGUCCA",
            utr3: "ACAGGGAAACCUGGCUC",
            lambda: 0.5,
            constraints: RegionConstraints::default(),
            must: &[17, 18],
        },
    ]
}

#[test]
fn reference_cells_equal_the_must_pair_scan() {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    for case in &cases() {
        assert_binds(&params, &usage, case);
        let cmp = compare_inside_must(
            &params,
            case.protein,
            &usage,
            case.lambda,
            &full_sets(case.protein),
            (case.utr5, case.utr3),
            case.constraints,
            case.must,
            ScanConfig::default(),
        )
        .expect("a valid instance")
        .expect("a lattice");
        assert!(cmp.identical(), "{}: {:#?}", case.protein, cmp.kinds);
    }
}

#[test]
fn a_must_pair_walk_exhausts_its_designs_in_order() {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let walks = [
        Walk::from(Build::Production),
        Walk {
            node_dedup: true,
            incremental_spell: true,
            ..Walk::from(Build::Production)
        },
    ];
    for case in &cases() {
        let want = brute_force(&params, &usage, case, case.must);
        for walk in walks {
            let mut got: Vec<(f64, String, i32)> = Vec::new();
            let stats = walk_designs_must(
                &params,
                case.protein,
                &usage,
                case.lambda,
                &full_sets(case.protein),
                (case.utr5, case.utr3),
                case.constraints,
                case.must,
                ScanConfig::default(),
                (u64::MAX, f64::INFINITY),
                walk,
                |obj, seq, energy, own| {
                    let paired: HashSet<u32> = own.iter().flat_map(|&(i, j)| [i, j]).collect();
                    for p in case.must {
                        assert!(paired.contains(p), "{seq}: position {p} unpaired");
                    }
                    got.push((obj, seq.to_string(), energy.expect("an energy at λ > 0")));
                    Ok(false)
                },
            )
            .expect("the walk runs")
            .expect("the lattice admits a design");
            assert_eq!(stats.stopped, "exhausted");
            assert_eq!(
                got.len(),
                want.len(),
                "{}: every design, once",
                case.protein
            );
            let seqs: HashSet<&String> = got.iter().map(|x| &x.1).collect();
            assert_eq!(seqs.len(), want.len(), "{}: no design twice", case.protein);
            for (rank, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g.0 - w.0).abs() < TOL,
                    "{} rank {rank}: walk {} against brute force {}",
                    case.protein,
                    g.0,
                    w.0
                );
            }
            for (_, seq, e) in &got {
                let w = want
                    .iter()
                    .find(|x| &x.1 == seq)
                    .expect("a brute-force design");
                assert_eq!(
                    *e, w.2,
                    "{seq}: the walk's energy is its best such structure's"
                );
            }
        }
    }
}

#[test]
fn the_first_passing_design_is_the_best_whose_must_structure_is_free_optimal() {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let mut answered = 0;
    for case in &cases() {
        let all = brute_force(&params, &usage, case, case.must);
        let want = all.iter().find(|(_, s, e)| {
            let free = sequence_free_mfe(&params, s).expect("folds");
            free.0 == *e
        });
        let report = first_modular_must(
            &params,
            case.protein,
            &usage,
            case.lambda,
            &full_sets(case.protein),
            (case.utr5, case.utr3),
            case.constraints,
            case.must,
            ScanConfig::default(),
            (u64::MAX, f64::INFINITY),
            Verify::folds(1),
            Walk::from(Build::Production),
        )
        .expect("the walk runs")
        .expect("a design exists");
        match (want, report.modular) {
            (Some(w), Some((got, seq, _))) => {
                assert!(
                    (w.0 - got).abs() < TOL,
                    "{}: walk {got} ({seq}) against brute force {} ({})",
                    case.protein,
                    w.0,
                    w.1
                );
                answered += 1;
            }
            (None, None) => {}
            (w, h) => panic!("{}: brute force {w:?} against walk {h:?}", case.protein),
        }
    }
    assert!(answered > 0, "no case has an answer");
}
