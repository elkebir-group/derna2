//! Shedding ends the walks running at the time and no others.
//!
//! `derna front --shed-on-memory` calls `outside::lazy::shed_walks` from its
//! memory watcher so that one walk whose frontier outgrows the budget ends
//! alone and the search goes on. A shed that latched like `stop_walks` would
//! end every later walk too and turn the flag back into a run-wide stop.
//!
//! Its own binary: the shed count is process-wide, and a walk running in a
//! sibling test at the moment of the shed would be ended by it.

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::outside::lazy::{shed_walks, walk_designs, walks_shed, Build, Walk};
use derna_lib::{CodonUsage, Enforcement, Params, RegionConstraints, ScanConfig};

fn every_codon(protein: &str) -> Vec<Vec<[u8; 3]>> {
    protein
        .bytes()
        .map(|b| {
            let a = aa_to_index(b as char).unwrap();
            AMINO_ACIDS[a].codons.iter().map(|c| c.nuc).collect()
        })
        .collect()
}

/// Walk to at most 20,000 derivations, calling `shed_walks` at design `shed_at`.
fn walk(shed_at: Option<usize>) -> (usize, &'static str) {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let (protein, utr5, utr3) = ("MLRSGK", "GGGAUCCG", "CGGAUCCC");
    let mut designs = 0usize;
    let stats = walk_designs(
        &params,
        protein,
        &usage,
        0.5,
        &every_codon(protein),
        utr5,
        utr3,
        RegionConstraints::five_prime_isolation(Enforcement::Forbid),
        ScanConfig::default(),
        20_000,
        f64::INFINITY,
        Walk::from(Build::Production),
        |_, _, _, _| {
            designs += 1;
            if Some(designs) == shed_at {
                shed_walks();
            }
            Ok(false)
        },
    )
    .expect("the walk runs")
    .expect("a lattice");
    (designs, stats.stopped)
}

#[test]
fn a_shed_ends_the_running_walk_and_not_the_next() {
    let (shed, why) = walk(Some(10));
    assert_eq!(why, "shed");
    assert!(shed < 50, "the shed walk stopped at once: {shed} designs");
    assert_eq!(walks_shed(), 1);
    let (after, why) = walk(None);
    assert_ne!(why, "shed", "a walk begun after the shed was ended by it");
    assert!(after > 50, "the next walk ran on: {after} designs");
}
