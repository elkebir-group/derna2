//! The single-writer column fill against the push scan. The fill is the `Lz`
//! default (`DERNA_PUSH_SCAN` selects the push) and opt-in in `Naive` mode
//! (`DERNA_PULL_FILL`); [`select`] sets both, so each check runs the path it
//! names in either mode, and one check runs with neither set to pin the
//! defaults.
//!
//! The fill computes every single-branch closure of a `P` cell in one unit that
//! owns the cell, so its contract differs from the push scan's in one place:
//! where several derivations tie, it may emit a different one. The checks are
//! therefore on what must not differ:
//!
//! 1. **The objective** equals the push scan's on every case.
//! 2. **The emitted design is modular**: its structure evaluates to the reported
//!    MFE on its own sequence.
//! 3. **Thread counts do not matter**: 1, 4 and 8 threads emit the same design,
//!    byte for byte — the property the push scan does not have. An extra
//!    8-thread arm sets `DERNA_FILL_FAN_ALL`, so every fill site fans out on
//!    these small lattices too, as it does on long ones.
//! 4. **Every cell agrees**: each table holds the same cells at the same BRD as
//!    the push scan's, so a closure family the optimum happens not to use is
//!    checked too — in both internal-loop modes, each fill against its own
//!    mode's push.
//!
//! What the cell comparison does not reach: a multi-loop whose leading
//! unpaired run is `SINGLE_MAX_LEN` or longer, which these lattices never
//! hold. `tests/single_loop_cap_boundary.rs` runs the default scan (the fill)
//! on such loops against the reference minimizer; dropping the fill's
//! long-flank sweep fails it and passes this file.
//!
//! The flags are read from the environment, which is process-wide, so this file
//! holds one test and runs as its own binary.

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::{
    design_must_paired, design_restricted, design_with_config, design_with_utr, dfa, scan, tables,
    Beam, CodonUsage, Design, Enforcement, InternalLoopMode, Params, RegionConstraints, ScanConfig,
};

const LAMBDAS: [f64; 3] = [0.0001, 0.5, 0.99];

fn config(threads: usize) -> ScanConfig {
    ScanConfig {
        beam: Beam::exact(),
        threads,
        ..ScanConfig::default()
    }
}

fn naive(threads: usize) -> ScanConfig {
    ScanConfig {
        mode: InternalLoopMode::Naive,
        ..config(threads)
    }
}

fn brd(d: &Design, lambda: f64) -> f64 {
    lambda * d.mfe_kcal_mol * 100.0 - (1.0 - lambda) * d.cai_log * 100.0
}

/// One design case: a label and how to solve it at a λ and a thread count.
type Case<'a> = (&'a str, Box<dyn Fn(f64, usize) -> Design + 'a>);

/// A scan configuration at a thread count, in one internal-loop mode.
type ConfigAt = fn(usize) -> ScanConfig;

#[test]
fn the_column_fill_matches_the_push_and_ignores_thread_count() {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let long = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHS";
    let short = "MLWKCLKYQ";
    let leu = aa_to_index('L').unwrap();
    let restricted: Vec<Vec<[u8; 3]>> = long
        .chars()
        .map(|c| {
            let a = aa_to_index(c).unwrap();
            if a == leu {
                vec![[3, 3, 0], [3, 3, 2]]
            } else {
                AMINO_ACIDS[a].codons.iter().map(|x| x.nuc).collect()
            }
        })
        .collect();
    let cases: Vec<Case> = vec![
        (
            "cds",
            Box::new(|l, t| design_with_config(&params, long, &usage, l, config(t)).unwrap()),
        ),
        (
            "cds, Naive",
            Box::new(|l, t| design_with_config(&params, long, &usage, l, naive(t)).unwrap()),
        ),
        (
            "flanked",
            Box::new(|l, t| {
                design_with_utr(
                    &params,
                    short,
                    &usage,
                    l,
                    "GGAUCCAGCUA",
                    "UAAGCGGC",
                    RegionConstraints::default(),
                    config(t),
                )
                .unwrap()
            }),
        ),
        (
            "region",
            Box::new(|l, t| {
                design_with_utr(
                    &params,
                    short,
                    &usage,
                    l,
                    "GGAUCCAGCUA",
                    "",
                    RegionConstraints::five_prime_isolation(Enforcement::Forbid),
                    config(t),
                )
                .unwrap()
            }),
        ),
        (
            "restricted",
            Box::new(|l, t| {
                design_restricted(&params, long, &usage, l, &restricted, "", "", config(t))
                    .unwrap()
                    .unwrap()
            }),
        ),
        (
            "must-pair",
            Box::new(|l, t| {
                design_must_paired(&params, long, &usage, l, &[30, 31, 90], config(t))
                    .unwrap()
                    .unwrap()
            }),
        ),
    ];

    let mut differs_from_push = 0;
    for (name, solve) in &cases {
        for lambda in LAMBDAS {
            select(false);
            let push = solve(lambda, 1);
            select(true);
            let mut pulled: Vec<Design> = [1, 4, 8].iter().map(|&t| solve(lambda, t)).collect();
            std::env::set_var("DERNA_FILL_FAN_ALL", "1");
            pulled.push(solve(lambda, 8));
            std::env::remove_var("DERNA_FILL_FAN_ALL");
            defaults();
            // With neither flag set, `Lz` runs the fill and `Naive` the push.
            let default = solve(lambda, 1);
            let expected = if name.contains("Naive") {
                &push
            } else {
                &pulled[0]
            };
            assert_eq!(&default, expected, "{name} λ={lambda}: the default path");

            let fill = &pulled[0];
            assert!(
                (brd(fill, lambda) - brd(&push, lambda)).abs() < 1e-6,
                "{name} λ={lambda}: objective {} against the push's {}",
                brd(fill, lambda),
                brd(&push, lambda)
            );
            let energy = rna_mfe::evaluate(&params, fill.sequence.as_bytes(), &fill.structure)
                .expect("a well-formed design");
            assert!(
                (energy.to_kcal_mol() - fill.mfe_kcal_mol).abs() < 1e-9,
                "{name} λ={lambda}: the structure evaluates to {} but reports {}",
                energy.to_kcal_mol(),
                fill.mfe_kcal_mol
            );
            for (t, other) in ["4", "8", "8, every site fanned"].iter().zip(&pulled[1..]) {
                assert_eq!(fill, other, "{name} λ={lambda}: 1 thread against {t}");
            }
            differs_from_push += usize::from(*fill != push);
        }
    }
    // A record, not a requirement: tie order is free to differ from the push.
    eprintln!("designs differing from the push on a tie: {differs_from_push}");

    // Every cell, not only the optimum's: a family the optimum never uses is
    // still checked. Same cells and the same BRD in every table, at 1 and 8
    // threads, on a CDS-only and a flanked lattice.
    let aa: Vec<usize> = long[..40]
        .chars()
        .map(|c| aa_to_index(c).unwrap())
        .collect();
    let flank: Vec<rna_mfe::Nucleotide> = b"GGAUCCAGCUA"
        .iter()
        .map(|&b| rna_mfe::Nucleotide::from_byte(b).unwrap())
        .collect();
    let graphs = [
        dfa::build(&aa, &usage),
        dfa::build_with_utr(&aa, &usage, &flank, &flank[..5]),
    ];
    let mut cells = 0;
    let modes: [(&str, ConfigAt); 2] = [("Lz", config), ("Naive", naive)];
    for (mode, cfg) in modes {
        for graph in &graphs {
            for lambda in LAMBDAS {
                select(false);
                let push = summarize(&scan::scan(&params, graph, lambda, cfg(1)), graph);
                select(true);
                for (threads, fan_all) in [(1, false), (8, false), (8, true)] {
                    if fan_all {
                        std::env::set_var("DERNA_FILL_FAN_ALL", "1");
                    }
                    let fill = summarize(&scan::scan(&params, graph, lambda, cfg(threads)), graph);
                    std::env::remove_var("DERNA_FILL_FAN_ALL");
                    assert_eq!(
                        push.len(),
                        fill.len(),
                        "{mode} λ={lambda} t={threads}: cell count"
                    );
                    for (a, b) in push.iter().zip(&fill) {
                        assert!(
                            a.0 == b.0 && (a.1 - b.1).abs() < 1e-6,
                            "{mode} λ={lambda} t={threads}: cell {} BRD {} against the push's {}",
                            b.0,
                            b.1,
                            a.1
                        );
                    }
                }
                select(false);
                cells += push.len();
            }
        }
    }
    eprintln!("cells compared per arm: {cells}");

    // The loop cap at its boundary, with the fill on: every split of a
    // 30-nucleotide single-branch loop is admitted, on molecules where nothing
    // competes with it, against the independent folder.
    select(true);
    for n1 in 0..=30 {
        let a = |k: usize| "A".repeat(k);
        let seq = format!("GGGCGGC{}GGCGCAAAAGCGCC{}GCCGCCC", a(n1), a(30 - n1));
        let (_, reference) = rna_mfe::fold(&params, seq.as_bytes()).expect("valid");
        let nts: Vec<rna_mfe::Nucleotide> = seq
            .bytes()
            .map(|b| rna_mfe::Nucleotide::from_byte(b).unwrap())
            .collect();
        let graph = dfa::build_fixed_mrna(&nts);
        for threads in [1, 8] {
            for cfg in [config, naive] {
                let t = scan::scan(&params, &graph, 1.0, cfg(threads));
                let got = t.c[graph.ix(graph.end())].expect("reaches the end").mfe;
                assert_eq!(got, reference, "({n1}, {}) t={threads}", 30 - n1);
            }
        }
    }
    defaults();
}

/// Select the column fill (`true`) or the push scan in both internal-loop
/// modes, whatever each mode's default.
fn select(fill: bool) {
    if fill {
        std::env::set_var("DERNA_PULL_FILL", "1");
        std::env::remove_var("DERNA_PUSH_SCAN");
    } else {
        std::env::remove_var("DERNA_PULL_FILL");
        std::env::set_var("DERNA_PUSH_SCAN", "1");
    }
}

/// Clear both selectors, so each mode runs its default path.
fn defaults() {
    std::env::remove_var("DERNA_PULL_FILL");
    std::env::remove_var("DERNA_PUSH_SCAN");
}

/// Every occupied cell of every table as `(label, BRD)`, sorted by label.
fn summarize(t: &tables::Tables, graph: &dfa::Dfa) -> Vec<(String, f64)> {
    let lambda = t.lambda();
    let mut out = Vec::new();
    for pos in 0..=graph.seq_len {
        for &o in graph.nodes_at(pos) {
            for (k, s) in t.p_states(o) {
                out.push((format!("P{o:?}{k:?}"), s.brd(lambda)));
            }
            for (k, s) in t.multi_states(o) {
                out.push((format!("Multi{o:?}{k:?}"), s.brd(lambda)));
            }
            for (i, s) in t.m_states(o) {
                out.push((format!("M{o:?}{i:?}"), s.brd(lambda)));
            }
            for (i, s) in t.m2_states(o) {
                out.push((format!("M2{o:?}{i:?}"), s.brd(lambda)));
            }
            if let Some(s) = t.c[graph.ix(o)] {
                out.push((format!("C{o:?}"), s.brd(lambda)));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}
