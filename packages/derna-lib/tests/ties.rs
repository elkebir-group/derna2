//! The tie walk against enumeration: at every point of the coding-only front,
//! the set of coding sequences the walk finds is exactly the set that spells
//! the point's energy and adaptation.

mod common;

use std::collections::BTreeSet;

use common::enumerate_all;
use derna_codon_dp::codon_table::aa_to_index;
use derna_lib::ties::{certify, front_point_permutation_ties, front_point_ties, Verdict, CAI_TIE};
use derna_lib::{design_front, dfa, CodonUsage, Energy, Params};

fn fixtures() -> (Params, CodonUsage) {
    (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    )
}

/// Every encoding's coding sequence, in `enumerate_all`'s order.
fn spellings(usage: &CodonUsage, protein: &str) -> Vec<String> {
    let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
    let graph = dfa::build(&aa, usage);
    graph
        .paths_between(graph.start(), graph.end())
        .into_iter()
        .map(|(_, seq)| {
            seq.iter()
                .map(|&nt| ['A', 'C', 'G', 'U'][nt as usize])
                .collect()
        })
        .collect()
}

#[test]
fn the_tie_walk_finds_exactly_the_tied_encodings() {
    let (params, usage) = fixtures();
    let mut tied_points = 0;
    for protein in common::CASES {
        let objectives = enumerate_all(&params, &usage, protein);
        let sequences = spellings(&usage, protein);
        for point in design_front(&params, protein, &usage).unwrap() {
            let energy = (point.mfe_kcal_mol * 100.0).round() as i32;
            let want: BTreeSet<String> = objectives
                .iter()
                .zip(&sequences)
                .filter(|(o, _)| o.mfe == energy && (o.cai_log - point.cai_log).abs() < CAI_TIE)
                .map(|(_, s)| s.clone())
                .collect();
            assert!(
                want.contains(&point.sequence),
                "{protein}: the front's own spelling"
            );
            let ties = front_point_ties(
                &params,
                protein,
                &usage,
                &point.sequence,
                Energy(energy),
                usize::MAX,
            )
            .unwrap();
            assert!(ties.exhausted);
            let got: BTreeSet<String> = ties.spellings.iter().cloned().collect();
            assert_eq!(
                got.len(),
                ties.spellings.len(),
                "{protein}: a spelling found twice"
            );
            assert_eq!(got, want, "{protein} at ({energy}, {})", point.cai_log);
            tied_points += usize::from(want.len() > 1);
        }
    }
    assert!(tied_points > 0, "no front point had a second spelling");
}

/// The adaptation ceiling the walk prunes with is on the front's own scale: a
/// region pinned to one design has that design's adaptation as its ceiling.
#[test]
fn a_pinned_region_has_its_design_adaptation_as_its_ceiling() {
    let (_, usage) = fixtures();
    let (params, _) = fixtures();
    for protein in common::CASES {
        let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
        for point in design_front(&params, protein, &usage).unwrap() {
            let pinned: Vec<Vec<[u8; 3]>> = point
                .sequence
                .as_bytes()
                .chunks_exact(3)
                .map(|c| {
                    let i = |b: u8| b"ACGU".iter().position(|&n| n == b).unwrap() as u8;
                    vec![[i(c[0]), i(c[1]), i(c[2])]]
                })
                .collect();
            let ceiling = dfa::cai_ceiling(&aa, &usage, &pinned).unwrap();
            assert!(
                (ceiling - point.cai_log).abs() < CAI_TIE,
                "{protein}: {ceiling} vs {}",
                point.cai_log
            );
        }
    }
}

/// With a cap, the walk stops and says so.
#[test]
fn a_capped_walk_reports_it_did_not_finish() {
    let (params, usage) = fixtures();
    let mut capped = 0;
    for protein in common::CASES {
        for point in design_front(&params, protein, &usage).unwrap() {
            let energy = Energy((point.mfe_kcal_mol * 100.0).round() as i32);
            let all = front_point_ties(
                &params,
                protein,
                &usage,
                &point.sequence,
                energy,
                usize::MAX,
            )
            .unwrap();
            if all.spellings.len() < 2 {
                continue;
            }
            let one =
                front_point_ties(&params, protein, &usage, &point.sequence, energy, 1).unwrap();
            assert!(!one.exhausted);
            assert_eq!(one.spellings, vec![point.sequence.clone()]);
            capped += 1;
        }
    }
    assert!(capped > 0);
}

/// A tied point whose first spelling is not modular between
/// these flanks and whose second is: the verdict must come from the second.
/// The flanks were found by search and are fixed here; the test checks the
/// split itself before relying on it.
#[test]
fn a_tie_is_certified_by_its_second_spelling() {
    let (params, usage) = fixtures();
    let protein = "MCWFGGWCKM";
    let (first, second) = (
        "AUGUGCUGGUUCGGGGGCUGGUGCAAGAUG",
        "AUGUGCUGGUUCGGAGGCUGGUGCAAGAUG",
    );
    let (utr5, utr3) = ("GUCCCUAA", "UGUUAACG");
    let energy = Energy(-780);
    let free = |m: &str| {
        rna_linearfold::fold(&params, m.as_bytes(), 1 << 20)
            .unwrap()
            .1
             .0
    };
    let constrained = energy.0 + free(utr5) + free(utr3);
    assert_ne!(
        free(&format!("{utr5}{first}{utr3}")),
        constrained,
        "the first spelling must not be in S"
    );
    assert_eq!(
        free(&format!("{utr5}{second}{utr3}")),
        constrained,
        "the second spelling must be in S"
    );

    let whole = certify(
        &params,
        protein,
        &usage,
        first,
        energy,
        utr5,
        utr3,
        usize::MAX,
    )
    .unwrap();
    assert_eq!(
        whole,
        Verdict::Certified {
            spelling: second.to_string(),
            checked: 2
        }
    );
    let capped = certify(&params, protein, &usage, first, energy, utr5, utr3, 1).unwrap();
    assert_eq!(capped, Verdict::Unresolved { checked: 1 });
    // Between flanks that admit neither spelling, the point is refuted.
    let mut seed: u64 = 11;
    let mut flank = || -> String {
        (0..8)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                b"ACGU"[(seed >> 33) as usize % 4] as char
            })
            .collect()
    };
    let refuted = (0..200).map(|_| (flank(), flank())).find_map(|(u5, u3)| {
        match certify(
            &params,
            protein,
            &usage,
            first,
            energy,
            &u5,
            &u3,
            usize::MAX,
        )
        .unwrap()
        {
            Verdict::Refuted { checked } => Some(checked),
            _ => None,
        }
    });
    assert_eq!(refuted, Some(2), "no flank pair refuted both spellings");
}

/// The permutation walk finds the same tie sets as the exact walk on every
/// enumerable case, and folds fewer regions.
#[test]
fn the_permutation_walk_agrees_with_the_exact_walk() {
    let (params, usage) = fixtures();
    let (mut exact_folds, mut permutation_folds) = (0, 0);
    for protein in common::CASES {
        for point in design_front(&params, protein, &usage).unwrap() {
            let energy = Energy((point.mfe_kcal_mol * 100.0).round() as i32);
            let exact = front_point_ties(
                &params,
                protein,
                &usage,
                &point.sequence,
                energy,
                usize::MAX,
            )
            .unwrap();
            let permuted = front_point_permutation_ties(
                &params,
                protein,
                &usage,
                &point.sequence,
                energy,
                usize::MAX,
            )
            .unwrap();
            let sorted = |t: &[String]| {
                let mut v = t.to_vec();
                v.sort();
                v
            };
            assert_eq!(
                sorted(&exact.spellings),
                sorted(&permuted.spellings),
                "{protein}"
            );
            exact_folds += exact.folds;
            permutation_folds += permuted.folds;
        }
    }
    assert!(
        permutation_folds < exact_folds,
        "{permutation_folds} folds against {exact_folds}"
    );
}
