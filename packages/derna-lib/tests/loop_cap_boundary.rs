//! The single-branch loop cap at its boundary: a bulge or internal loop of
//! exactly `single_max_len` = 30 unpaired bases.
//!
//! The cap is ViennaRNA's `MAXLOOP`, a bound on the loop's **total** unpaired
//! count, `n1 + n2 <= 30`, split either way. `rna_mfe::fold`, the independent
//! Zuker folder validated against ViennaRNA, is the reference. The scan in
//! both internal-loop modes and every exact solver must agree with it at 30,
//! and below and above it.
//!
//! The molecules are built so nothing competes with the loop under test: a
//! 7-bp G/C helix, then the `n1 × n2` loop of `A`s around a 5-bp G/C helix
//! closing a hairpin of four `A`s. `A` pairs with nothing here, so the optimum
//! is the two helices joined by the loop when the loop is admitted, and about
//! 8 kcal/mol worse when it is not.

use derna_lib::{dfa, scan, Beam, InternalLoopMode, Params, ScanConfig};
use rna_mfe::Nucleotide;

/// The molecule and its intended structure for one `(n1, n2)` loop.
fn molecule(n1: usize, n2: usize) -> (String, String) {
    let a = |k: usize| "A".repeat(k);
    let seq = format!("GGGCGGC{}GGCGCAAAAGCGCC{}GCCGCCC", a(n1), a(n2));
    let structure = format!(
        "{}{}{}....{}{}{}",
        "(".repeat(7),
        ".".repeat(n1),
        "(".repeat(5),
        ")".repeat(5),
        ".".repeat(n2),
        ")".repeat(7)
    );
    (seq, structure)
}

/// The scan's λ = 1 minimum free energy of a fixed sequence, in kcal/mol.
fn scan_mfe(params: &Params, seq: &str, mode: InternalLoopMode) -> f64 {
    let nts: Vec<Nucleotide> = seq
        .bytes()
        .map(|b| Nucleotide::from_byte(b).expect("A/C/G/U"))
        .collect();
    let graph = dfa::build_fixed_mrna(&nts);
    let config = ScanConfig {
        beam: Beam::exact(),
        mode,
        ..ScanConfig::default()
    };
    let tables = scan::scan(params, &graph, 1.0, config);
    tables.c[graph.ix(graph.end())]
        .expect("the exterior reaches the end")
        .mfe
        .to_kcal_mol()
}

/// Every folder whose optimum on the `(n1, n2)` molecule differs from the
/// reference's, named.
fn disagreements(params: &Params, n1: usize, n2: usize) -> Vec<String> {
    let (seq, _) = molecule(n1, n2);
    let (_, reference) = rna_mfe::fold(params, seq.as_bytes()).expect("a valid sequence");
    let reference = reference.to_kcal_mol();
    // `rna_linearfold` is not here: it keeps the reference C++'s 5′ range,
    // which stops a 5′ flank one short of the cap (see its
    // `emit_p_to_helix_or_single`).
    [
        ("scan Lz", scan_mfe(params, &seq, InternalLoopMode::Lz)),
        (
            "scan Naive",
            scan_mfe(params, &seq, InternalLoopMode::Naive),
        ),
    ]
    .into_iter()
    .filter(|&(_, e)| (e - reference).abs() > 1e-9)
    .map(|(label, e)| format!("({n1}, {n2}) {label}: {e} against {reference}"))
    .collect()
}

/// Every split of a 30-nucleotide loop is admitted by every folder — and is
/// the reference's optimum, so the agreement is about the loop.
#[test]
fn every_thirty_nucleotide_loop_is_admitted() {
    let params = Params::load_default().expect("params");
    let mut wrong = Vec::new();
    for n1 in 0..=30 {
        let (seq, structure) = molecule(n1, 30 - n1);
        let (folded, _) = rna_mfe::fold(&params, seq.as_bytes()).expect("valid");
        assert_eq!(
            folded,
            structure,
            "({n1}, {}): the reference folds elsewhere",
            30 - n1
        );
        wrong.extend(disagreements(&params, n1, 30 - n1));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Below the cap every folder finds the loop, and above it none does — the
/// reference included, so the boundary is where the model puts it.
#[test]
fn the_folders_agree_off_the_boundary() {
    let params = Params::load_default().expect("params");
    let mut wrong = Vec::new();
    for (n1, n2) in [(0, 29), (29, 0), (1, 28), (28, 1), (14, 15)] {
        let (seq, structure) = molecule(n1, n2);
        let (folded, _) = rna_mfe::fold(&params, seq.as_bytes()).expect("valid");
        assert_eq!(folded, structure, "({n1}, {n2}) is under the cap");
        wrong.extend(disagreements(&params, n1, n2));
    }
    for (n1, n2) in [(0, 31), (31, 0), (1, 30), (30, 1), (16, 15)] {
        let (seq, structure) = molecule(n1, n2);
        let (folded, _) = rna_mfe::fold(&params, seq.as_bytes()).expect("valid");
        assert_ne!(folded, structure, "({n1}, {n2}) is over the cap");
        wrong.extend(disagreements(&params, n1, n2));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The other exact solvers on the same molecules: the molecule as a fixed 5′
/// flank before one lysine codon, so each design entry point can run on it.
/// Pinned to `AAA` the codon pairs with nothing, and the optimum is the
/// molecule's own plus three unpaired `A`s.
mod solvers {
    use super::*;
    use derna_lib::outside::{edge_marginals, edge_marginals_fast};
    use derna_lib::{
        design_front_bounded_with_utr, design_restricted, design_restricted_with_placed_motifs,
        design_with_utr, Bounded, Bounds, CodonUsage, MotifWindow, Prune, RegionConstraints,
        UtrProfile, UtrSpec,
    };

    const LYS_AAA: [u8; 3] = [0, 0, 0];
    const LYS_AAG: [u8; 3] = [0, 0, 2];

    fn exact() -> ScanConfig {
        ScanConfig {
            beam: Beam::exact(),
            ..ScanConfig::default()
        }
    }

    /// Every exact solver's minimum free energy on the flanked molecule, in
    /// kcal/mol, after the reference's.
    fn energies(
        params: &Params,
        usage: &CodonUsage,
        n1: usize,
        n2: usize,
    ) -> Vec<(&'static str, f64)> {
        let (flank, _) = molecule(n1, n2);
        let pinned = [vec![LYS_AAA]];
        let both = [vec![LYS_AAA, LYS_AAG]];
        let (_, reference) =
            rna_mfe::fold(params, format!("{flank}AAA").as_bytes()).expect("valid");
        let restricted = design_restricted(params, "K", usage, 1.0, &pinned, &flank, "", exact())
            .expect("valid")
            .expect("feasible");
        // Forbidding `AAG` inside the coding sequence leaves `AAA`, through the
        // motif-constrained folder. The flank spells `AAG` itself, so the window
        // is the codon alone.
        let cds = MotifWindow::new(flank.len() as u32, flank.len() as u32 + 3);
        let constrained = design_restricted_with_placed_motifs(
            params,
            "K",
            usage,
            1.0,
            &both,
            &[("AAG".to_string(), cds)],
            &flank,
            "",
            RegionConstraints::default(),
            exact(),
        )
        .expect("valid")
        .expect("feasible");
        let pass = |fast: bool| {
            let run = if fast {
                edge_marginals_fast
            } else {
                edge_marginals
            };
            run(
                params,
                "K",
                usage,
                1.0,
                &pinned,
                &flank,
                "",
                RegionConstraints::default(),
                exact(),
            )
            .expect("valid")
            .expect("feasible")
        };
        let (rules, fast) = (pass(false), pass(true));
        // The two backward passes' outside value of every closed pair. The
        // inner helix's runs through the loop, so a pass that does not admit
        // it disagrees there even though the optimum it reports is the scan's.
        let (ro, fo) = (rules.pair_outsides(), fast.pair_outsides());
        let outside_gap = if ro.len() == fo.len() {
            ro.iter()
                .zip(fo)
                .map(|(a, b)| {
                    if (a.0, a.1) == (b.0, b.1) {
                        (a.2 - b.2).abs()
                    } else {
                        f64::INFINITY
                    }
                })
                .fold(0.0, f64::max)
        } else {
            f64::INFINITY
        };
        vec![
            ("reference", reference.to_kcal_mol()),
            ("restricted scan", restricted.mfe_kcal_mol),
            ("constrained fold", constrained.mfe_kcal_mol),
            ("outside, rules", rules.optimum() / 100.0),
            ("outside, scan", fast.optimum() / 100.0),
            // Zero when the passes agree, shifted onto the reference so the
            // same comparison reads it.
            (
                "outside values, rules against scan",
                reference.to_kcal_mol() + outside_gap,
            ),
        ]
    }

    /// The front DP's lowest-energy point and the scan's λ = 1 optimum on the
    /// same unrestricted lattice: both exact, so they must agree.
    fn front_against_scan(params: &Params, usage: &CodonUsage, n1: usize, n2: usize) -> (f64, f64) {
        let (flank, _) = molecule(n1, n2);
        let scan = design_with_utr(
            params,
            "K",
            usage,
            1.0,
            &flank,
            "",
            RegionConstraints::default(),
            exact(),
        )
        .expect("valid");
        let front = match design_front_bounded_with_utr(
            params,
            "K",
            usage,
            &UtrSpec::Fixed(flank),
            &UtrSpec::Fixed(String::new()),
            UtrProfile::inert(),
            Prune::None,
            Bounds::UNBOUNDED,
            1,
        )
        .expect("the flanked fold answers")
        {
            Bounded::Feasible(f) => f.front,
            Bounded::Unreachable { .. } => unreachable!("no ceiling was set"),
        };
        let lowest = front
            .iter()
            .map(|p| p.mfe_kcal_mol)
            .fold(f64::INFINITY, f64::min);
        (scan.mfe_kcal_mol, lowest)
    }

    #[test]
    fn every_exact_solver_admits_the_thirty_nucleotide_loop() {
        let params = Params::load_default().expect("params");
        let usage = CodonUsage::load_default().expect("usage");
        let mut wrong = Vec::new();
        let shapes = [
            (0, 30),
            (30, 0),
            (1, 29),
            (29, 1),
            (2, 28),
            (28, 2),
            (15, 15),
            (14, 15),
            (16, 15),
        ];
        for (n1, n2) in shapes {
            let all = energies(&params, &usage, n1, n2);
            let reference = all[0].1;
            for &(label, e) in &all[1..] {
                if (e - reference).abs() > 1e-9 {
                    wrong.push(format!("({n1}, {n2}) {label}: {e} against {reference}"));
                }
            }
            let (scan, front) = front_against_scan(&params, &usage, n1, n2);
            if (scan - front).abs() > 1e-9 {
                wrong.push(format!(
                    "({n1}, {n2}) front DP: {front} against the scan's {scan}"
                ));
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }
}
