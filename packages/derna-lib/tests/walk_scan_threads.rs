//! The walk's own scan may run on several threads: the scan reaches the same
//! value in every cell on any number of them, and the walk reads only those
//! values, so a walk visits the same designs in the same order, and a window
//! walk reports the same front, whatever the scan's thread count.
//!
//! `DERNA_FILL_FAN_ALL` makes every fill site fan out on these small lattices,
//! as it does on long ones; without it they stay below the parallel threshold
//! and the threaded arm would run serial. The flag is process-wide, so this
//! file holds one test and runs as its own binary.

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::outside::lazy::window::{window_front, WindowReport};
use derna_lib::outside::lazy::{walk_designs, Build, Verify, Walk};
use derna_lib::{CodonUsage, Enforcement, PairClass, Params, RegionConstraints, ScanConfig};

fn full_sets(protein: &str) -> Vec<Vec<[u8; 3]>> {
    protein
        .chars()
        .map(|c| {
            let a = aa_to_index(c).expect("a standard residue");
            AMINO_ACIDS[a].codons.iter().map(|c| c.nuc).collect()
        })
        .collect()
}

const DEDUP: Walk = Walk {
    build: Build::Production,
    node_dedup: true,
    incremental_spell: true,
    pipelined_visit: false,
    spellers: 0,
    dedup_index: false,
    tight_lists: false,
};

fn config(threads: usize) -> ScanConfig {
    ScanConfig {
        threads,
        ..ScanConfig::default()
    }
}

#[test]
fn the_scans_thread_count_changes_no_walk() {
    std::env::set_var("DERNA_FILL_FAN_ALL", "1");
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let constraint = RegionConstraints::five_prime_isolation(Enforcement::Forbid)
        .with(PairClass::Utr5Utr3, Enforcement::Forbid);
    let (protein, utr5, utr3) = ("MLRSGK", "GGGAUCCG", "CGGAUCCC");
    let allowed = full_sets(protein);

    for lambda in [0.05, 0.5, 0.95] {
        let walk = |threads: usize| -> Vec<(String, Option<i32>)> {
            let mut seen = Vec::new();
            walk_designs(
                &params,
                protein,
                &usage,
                lambda,
                &allowed,
                utr5,
                utr3,
                constraint,
                config(threads),
                3000,
                f64::INFINITY,
                DEDUP,
                |_, sequence, energy, _| {
                    seen.push((sequence.to_string(), energy));
                    Ok(false)
                },
            )
            .expect("the walk runs")
            .expect("a lattice");
            seen
        };
        let one = walk(1);
        assert!(one.len() > 100, "the walk reaches designs at {lambda}");
        assert_eq!(one, walk(4), "at {lambda}");
    }

    // A window walk with the energy bound on: its pass at the walk's own
    // weight also supplies the walk's tables. Its check workers report finds
    // as they land, so how far the walk runs before its coverage completes
    // (`reached`, `last_g`) depends on their timing at any thread count; the
    // front it reports does not.
    let key = |r: &WindowReport| {
        (
            r.front
                .iter()
                .map(|p| (p.energy, p.sequence.clone()))
                .collect::<Vec<_>>(),
            r.complete,
        )
    };
    let mu = 30.0;
    let (protein, utr5, utr3) = ("GLRSA", "GGCGC", "GCGCC");
    // Unconstrained, every design is modular, so the walk has a
    // front to report.
    let isolation = RegionConstraints::default();
    let allowed = full_sets(protein);
    let window = |threads: usize| {
        window_front(
            &params,
            protein,
            &usage,
            &allowed,
            utr5,
            utr3,
            isolation,
            config(threads),
            (f64::NEG_INFINITY, f64::INFINITY),
            mu,
            &[],
            &[0.0, 0.5 * mu, mu, 2.0 * mu],
            u64::MAX,
            f64::INFINITY,
            Verify::folds(1),
            DEDUP,
            None,
        )
        .expect("the walk runs")
        .expect("a lattice")
    };
    let one = window(1);
    assert!(
        !one.front.is_empty(),
        "the window walk finds a front: reached {} offered {} found {} complete {}",
        one.reached,
        one.offered,
        one.found,
        one.complete
    );
    assert_eq!(key(&one), key(&window(4)));
    std::env::remove_var("DERNA_FILL_FAN_ALL");
}
