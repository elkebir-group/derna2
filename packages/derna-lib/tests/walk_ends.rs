//! The walk's two ends are the front's exact ends, not the optima at its two
//! outermost weights.
//!
//! The **stable end** is the least MFE, and the greatest adaptation among the
//! designs reaching it; the **adapted end** is the greatest adaptation, and
//! the least MFE among the designs reaching it. Every case is checked against
//! the enumeration oracle (every encoding folded, then Pareto-reduced), both
//! ends included, under the plain options and under the grid-seeded ones
//! `derna front` runs.
//!
//! The near-tied usage is the case a weight inside `(0, 1)` cannot reach: two
//! synonymous codons whose weights differ by less than the outer weight times
//! the energy they trade, so the optimum at that weight takes the less adapted
//! codon and the most adapted design lies beyond it. The test asserts that the
//! scalar optimum at the walk's outer weight does miss the adapted end there,
//! so the case cannot pass vacuously.

mod common;

use common::{assert_same, eliminate, enumerate_all, front_objectives, Objective, CAI_EPS};
use derna_lib::{
    design, design_front_bounded_with_utr, design_front_walk, design_front_walk_with_utr, Bounded,
    Bounds, CodonUsage, Params, Prune, RegionConstraints, UtrProfile, UtrSpec, WalkOptions,
    WalkedFront,
};

/// The outer weights `derna front` walks between, `2⁻²³` and `1 − 2⁻²³`.
const EPSILON: f64 = 1.0 / (1u32 << 23) as f64;

/// Proteins small enough to enumerate, with enough synonymous choice that the
/// front has width at both ends.
const PROTEINS: [&str; 6] = [
    "MWWKFYCLMW",
    "MYGKIIFVLM",
    "MCWFGGWCKM",
    "MGAPGRW",
    "MPGAGPCW",
    "MRGAPW",
];

/// The human usage table with the glycine, alanine and proline families'
/// codons set to weights within `2e-8` of one another, distinct by more than
/// the front's dominance slop: a few kcal/mol of energy outweighs every
/// adaptation difference among them at `λ = 2⁻²³`.
fn near_tied_usage() -> CodonUsage {
    let base = include_str!("../../derna-codon-dp/data/codon_usage_human.csv");
    let tied = |aa: &str, codon: &str| -> Option<f64> {
        let rank = match (aa, codon) {
            ("G", "GGC") | ("A", "GCC") | ("P", "CCC") => 0.0,
            ("G", "GGG") | ("A", "GCG") | ("P", "CCG") => 1.0,
            ("G", "GGA") | ("A", "GCA") | ("P", "CCA") => 2.0,
            ("G", "GGU") | ("A", "GCU") | ("P", "CCU") => 3.0,
            _ => return None,
        };
        Some(0.25 * (1.0 - 5e-9 * rank))
    };
    let csv: String = base
        .lines()
        .map(|line| {
            let parts: Vec<&str> = line.split(',').collect();
            match parts.as_slice() {
                [aa, codon, _] => match tied(aa, codon) {
                    Some(w) => format!("{aa},{codon},{w:.17}\n"),
                    None => format!("{line}\n"),
                },
                _ => format!("{line}\n"),
            }
        })
        .collect();
    CodonUsage::load_from_str(&csv).expect("the near-tied table parses")
}

/// The two option sets checked: the walk's plain defaults, and the
/// grid-seeded sweep `derna front` runs.
fn plans(protein: &str) -> [(&'static str, WalkOptions); 2] {
    [
        ("plain", WalkOptions::default()),
        ("seeded", WalkOptions::for_residues(protein.len())),
    ]
}

/// The stable end of an oracle front: least energy, then greatest adaptation.
fn stable_end(bag: &[Objective]) -> Objective {
    let least = bag.iter().map(|o| o.mfe).min().expect("a non-empty bag");
    *bag.iter()
        .filter(|o| o.mfe == least)
        .max_by(|a, b| a.cai_log.total_cmp(&b.cai_log))
        .expect("a design at the least energy")
}

/// The adapted end of an oracle front: greatest adaptation (within the
/// front's slop), then least energy.
fn adapted_end(bag: &[Objective]) -> Objective {
    let best = common::best_adaptation(bag);
    *bag.iter()
        .filter(|o| o.cai_log >= best - common::CAI_DOM_EPS)
        .min_by_key(|o| o.mfe)
        .expect("a design at the greatest adaptation")
}

/// The walk's front matches the oracle's, and its first and last points are
/// the two exact ends.
fn assert_exact(label: &str, walked: &WalkedFront, bag: &[Objective]) {
    let got = front_objectives(&walked.front);
    assert_same(label, &eliminate(bag.to_vec()), &got);
    let (first, last) = (got.first().unwrap(), got.last().unwrap());
    let (stable, adapted) = (stable_end(bag), adapted_end(bag));
    assert_eq!(first.mfe, stable.mfe, "{label}: stable end's energy");
    assert!(
        (first.cai_log - stable.cai_log).abs() < CAI_EPS,
        "{label}: stable end's adaptation {} against {}",
        first.cai_log,
        stable.cai_log,
    );
    assert_eq!(last.mfe, adapted.mfe, "{label}: adapted end's energy");
    assert!(
        (last.cai_log - adapted.cai_log).abs() < CAI_EPS,
        "{label}: adapted end's adaptation {} against {}",
        last.cai_log,
        adapted.cai_log,
    );
    // The reported hull reaches both ends too.
    let (lo, hi) = (
        walked.vertices.first().unwrap(),
        walked.vertices.last().unwrap(),
    );
    assert_eq!(
        (lo.mfe_kcal_mol * 100.0).round() as i32,
        stable.mfe,
        "{label}: the hull's stable vertex"
    );
    assert_eq!(
        (hi.mfe_kcal_mol * 100.0).round() as i32,
        adapted.mfe,
        "{label}: the hull's adapted vertex"
    );
    assert!(
        (hi.adaptation_log - adapted.cai_log).abs() < CAI_EPS,
        "{label}: the hull's adapted vertex's adaptation"
    );
}

fn walk(params: &Params, usage: &CodonUsage, protein: &str, options: WalkOptions) -> WalkedFront {
    design_front_walk(params, protein, usage, EPSILON, 1.0 - EPSILON, options, 2)
        .expect("the walk answers")
}

/// Under the human usage table the walk recovers the whole front, both ends
/// included.
#[test]
fn the_walk_reaches_both_exact_ends() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    for protein in PROTEINS {
        let bag = enumerate_all(&params, &usage, protein);
        for (name, options) in plans(protein) {
            let walked = walk(&params, &usage, protein, options);
            assert_exact(&format!("{protein}, {name}"), &walked, &bag);
        }
    }
}

/// Under near-tied codons the optimum at `2⁻²³` stops short of the most
/// adapted design, and the walk still reaches it.
#[test]
fn the_walk_reaches_an_adapted_end_its_outer_weight_misses() {
    let params = Params::load_default().unwrap();
    let usage = near_tied_usage();
    let mut missed = 0;
    for protein in PROTEINS {
        let bag = enumerate_all(&params, &usage, protein);
        let outer = design(&params, protein, &usage, EPSILON).unwrap();
        if outer.cai_log < adapted_end(&bag).cai_log - common::CAI_DOM_EPS {
            missed += 1;
        }
        for (name, options) in plans(protein) {
            let walked = walk(&params, &usage, protein, options);
            assert_exact(&format!("{protein} near-tied, {name}"), &walked, &bag);
        }
    }
    assert!(
        missed > 0,
        "no case's optimum at the outer weight missed the adapted end, so the end \
         was never in question"
    );
}

/// The flanked walk reaches the same ends, against the flanked lattice's whole
/// front under no ceiling.
#[test]
fn the_flanked_walk_reaches_an_adapted_end_its_outer_weight_misses() {
    let params = Params::load_default().unwrap();
    let usage = near_tied_usage();
    let (five, three) = (
        UtrSpec::Fixed("GGGAAAUAAGAG".to_string()),
        UtrSpec::Fixed("UGAUAAUAGGCUGGAGCC".to_string()),
    );
    for protein in PROTEINS {
        let Bounded::Feasible(whole) = design_front_bounded_with_utr(
            &params,
            protein,
            &usage,
            &five,
            &three,
            UtrProfile::inert(),
            Prune::None,
            Bounds::UNBOUNDED,
            2,
        )
        .unwrap() else {
            panic!("an unbounded front is feasible");
        };
        let want = front_objectives(&whole.front);
        for (name, options) in plans(protein) {
            let walked = design_front_walk_with_utr(
                &params,
                protein,
                &usage,
                &five,
                &three,
                RegionConstraints::default(),
                EPSILON,
                1.0 - EPSILON,
                options,
                2,
            )
            .unwrap();
            let label = format!("{protein} near-tied with flanks, {name}");
            assert_same(&label, &want, &front_objectives(&walked.front));
        }
    }
}
