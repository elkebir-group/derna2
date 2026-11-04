//! The free-fold verdict's contracts.
//!
//! A region-constrained design is the exact optimum over structures that respect
//! the constraint; the molecule is under no such obligation. `free_fold_gap`
//! reports what that difference costs. What is pinned here:
//!
//! 1. **The inert floor** — with no constraints a design already *is* its own
//!    free fold, so the gap is zero and nothing is reinstated. Any nonzero
//!    reading on an unconstrained design would be a bug in the diagnostic rather
//!    than a finding about the design.
//! 2. **Hard isolation is consistent with itself** — the reported MFE carries no
//!    penalty, the gap is non-negative, and the emitted structure contains none
//!    of the pairs it forbade.
//! 3. **The constraint really can be undone by the free fold** — on a fixture
//!    built to make cross-region pairing overwhelmingly favourable, the free fold
//!    reinstates pairs and the gap is strictly positive. This is the case the
//!    diagnostic exists to surface; a suite where it never fires would prove
//!    nothing.
//! 4. **The soft-penalty split** — under a soft family the design's reported MFE
//!    is the emitted structure's true energy *plus* one penalty per penalized
//!    pair, exactly. The re-score is the physical number.
//! 5. **The verdict holds over the whole λ domain, `λ = 0` included.** The
//!    constrained energy comes from a real fold of the emitted sequence, so it
//!    does not inherit the arbitrariness of a traceback the objective never
//!    ranked.
//! 6. **The breakdown covers permitted classes**, which is what makes a
//!    structure the table never objected to visible at all.

use derna_lib::{
    design, design_with_utr, free_fold_gap, sequence_gap, sequence_gap_given_constrained,
    sequence_gap_given_structure, CdsWindow, CodonUsage, Design, Energy, Enforcement, PairClass,
    Params, RegionConstraints, ScanConfig,
};

/// Energy comparisons are exact integer dekacalories under the hood; this only
/// absorbs the `/100.0` float division.
const TOL: f64 = 1e-6;

/// The λ values swept where a contract should hold at every trade-off.
const LAMBDAS: [f64; 3] = [0.0001, 0.5, 0.99];

/// A 5′ UTR and a protein chosen so that cross-region pairing is the sequence's
/// overwhelming structural preference: a poly-G flank against a proline run,
/// whose every codon is `CCN`. Isolation must forbid the G–C stem the free fold
/// then rebuilds.
const STICKY_UTR5: &str = "GGGGGGGG";
const STICKY_PROTEIN: &str = "PPPPP";

fn setup() -> (Params, CodonUsage) {
    (
        Params::load_default().expect("default Turner 2004 parameters"),
        CodonUsage::load_default().expect("default human codon usage"),
    )
}

fn hard_isolation() -> RegionConstraints {
    RegionConstraints::symmetric_isolation(Enforcement::Forbid)
}

/// The pairs of a dot-bracket structure (a balanced-parens stack walk).
fn pairs_of(structure: &str) -> Vec<(usize, usize)> {
    let mut stack = Vec::new();
    let mut pairs = Vec::new();
    for (j, c) in structure.bytes().enumerate() {
        match c {
            b'(' => stack.push(j),
            b')' => pairs.push((stack.pop().expect("balanced"), j)),
            _ => {}
        }
    }
    pairs
}

/// How many pairs of `structure` cross the CDS boundary at `cds`.
fn cross_region_count(structure: &str, cds: CdsWindow) -> usize {
    let in_cds = |p: usize| p >= cds.start && p < cds.end;
    pairs_of(structure)
        .into_iter()
        .filter(|&(i, j)| in_cds(i) != in_cds(j))
        .count()
}

fn design_sticky(
    params: &Params,
    usage: &CodonUsage,
    lambda: f64,
    constraints: RegionConstraints,
) -> (Design, CdsWindow) {
    let design = design_with_utr(
        params,
        STICKY_PROTEIN,
        usage,
        lambda,
        STICKY_UTR5,
        "",
        constraints,
        ScanConfig::default(),
    )
    .expect("sticky fixture designs");
    (
        design,
        CdsWindow::new(STICKY_UTR5.len(), STICKY_PROTEIN.len()),
    )
}

/// (1) An unconstrained design is its own free fold: nothing to report.
#[test]
fn unconstrained_design_has_no_gap_and_no_reinstated_pairs() {
    let (params, usage) = setup();
    let protein = "MKTAYIAKQR";
    for lambda in LAMBDAS {
        let d = design(&params, protein, &usage, lambda).expect("designs");
        let cds = CdsWindow::new(0, protein.len());
        let report =
            free_fold_gap(&params, &d, cds, RegionConstraints::default()).expect("diagnostic runs");

        assert!(
            report.gap_kcal_mol.abs() < TOL,
            "λ={lambda}: an unconstrained design must equal its own free fold, got gap {}",
            report.gap_kcal_mol
        );
        assert!(
            !report.reinstates(),
            "λ={lambda}: an inert constraint set can reinstate nothing, got {:?}",
            report.reinstated_pairs
        );
        assert!(
            (rescore(&report).mfe_kcal_mol - d.mfe_kcal_mol).abs() < TOL,
            "λ={lambda}: no penalty is in play, so the re-score must match the report"
        );
    }
}

/// The design's own structure, checked against the energy model — present on
/// every report `free_fold_gap` produces.
fn rescore(report: &derna_lib::FreeFoldGap) -> derna_lib::EmittedRescore {
    report
        .emitted
        .expect("a report derived from a design carries its re-score")
}

/// (1b) The same floor with UTR flanks present but isolation switched off — the
/// whole-mRNA scan is still an unconstrained fold.
#[test]
fn utr_run_without_constraints_has_no_gap() {
    let (params, usage) = setup();
    let (d, cds) = design_sticky(&params, &usage, 0.5, RegionConstraints::default());
    let report =
        free_fold_gap(&params, &d, cds, RegionConstraints::default()).expect("diagnostic runs");

    assert!(
        report.gap_kcal_mol.abs() < TOL,
        "an unconstrained whole-mRNA design must equal its own free fold, got gap {}",
        report.gap_kcal_mol
    );
    assert!(!report.reinstates());
}

/// (2) Under hard isolation the reported energy is penalty-free, the gap cannot
/// be negative (the free fold is a global minimum over the same model), and the
/// emitted structure honours the constraint it was designed under.
#[test]
fn hard_isolation_reports_a_penalty_free_non_negative_gap() {
    let (params, usage) = setup();
    for lambda in LAMBDAS {
        let (d, cds) = design_sticky(&params, &usage, lambda, hard_isolation());
        let report = free_fold_gap(&params, &d, cds, hard_isolation()).expect("diagnostic runs");

        assert!(
            (rescore(&report).mfe_kcal_mol - d.mfe_kcal_mol).abs() < TOL,
            "λ={lambda}: a hard family adds no penalty, so the reported MFE is the true one"
        );
        assert!(
            report.gap_kcal_mol > -TOL,
            "λ={lambda}: the free fold is a global minimum, so the gap cannot be negative, got {}",
            report.gap_kcal_mol
        );
        assert_eq!(
            cross_region_count(&d.structure, cds),
            0,
            "λ={lambda}: the designed structure must contain no cross-region pair"
        );
        assert!(
            (report.gap_kcal_mol - (report.constrained_mfe_kcal_mol - report.free_mfe_kcal_mol))
                .abs()
                < TOL,
            "λ={lambda}: the report must be internally consistent"
        );
        // λ > 0 is exactly where the retired shortcut was sound: the emitted
        // structure is constrained-optimal for its own sequence, so re-scoring
        // it agrees with the constrained fold the verdict now comes from.
        assert!(
            (rescore(&report).mfe_kcal_mol - report.constrained_mfe_kcal_mol).abs() < TOL,
            "λ={lambda}: the design's structure should be the constrained optimum of its sequence"
        );
        assert!(
            rescore(&report).phantom_kcal_mol.abs() < TOL,
            "λ={lambda}: the reported energy must be one the reported structure accounts for"
        );
    }
}

/// (3) The finding the diagnostic exists for: a sequence whose free fold rebuilds
/// exactly what isolation forbade, at a strictly positive energy cost.
#[test]
fn a_sticky_boundary_is_reinstated_by_the_free_fold() {
    let (params, usage) = setup();
    let (d, cds) = design_sticky(&params, &usage, 0.99, hard_isolation());
    let report = free_fold_gap(&params, &d, cds, hard_isolation()).expect("diagnostic runs");

    assert!(
        report.reinstates(),
        "the poly-G / poly-proline boundary must pair when unconstrained; free fold was {}",
        report.free_structure
    );
    assert!(
        report.gap_kcal_mol > TOL,
        "reinstated pairs mean the free fold is strictly better, got gap {}",
        report.gap_kcal_mol
    );
    for &(i, j) in &report.reinstated_pairs {
        let in_cds = |p: usize| p >= cds.start && p < cds.end;
        assert!(
            in_cds(i) != in_cds(j),
            "every reinstated pair under isolation alone must cross the boundary, got ({i}, {j})"
        );
    }
}

/// (5) The general modularity test agrees with the designer's own accounting. For a
/// hard constraint the emitted structure *is* a minimum-energy allowed structure
/// for that sequence, so re-deriving it from the sequence alone must reproduce
/// the same gap — which is what lets a candidate nobody designed be tested.
#[test]
fn sequence_gap_reproduces_the_designers_gap_under_hard_isolation() {
    let (params, usage) = setup();
    for lambda in LAMBDAS {
        let (d, cds) = design_sticky(&params, &usage, lambda, hard_isolation());
        let from_design =
            free_fold_gap(&params, &d, cds, hard_isolation()).expect("diagnostic runs");
        let from_sequence =
            sequence_gap(&params, &d.sequence, cds, hard_isolation()).expect("re-derives");

        assert!(
            (from_sequence.gap_kcal_mol - from_design.gap_kcal_mol).abs() < TOL,
            "λ={lambda}: sequence-only gap {} must match the design's {}",
            from_sequence.gap_kcal_mol,
            from_design.gap_kcal_mol
        );
        assert!(
            (from_sequence.free_mfe_kcal_mol - from_design.free_mfe_kcal_mol).abs() < TOL,
            "λ={lambda}: both fold the same sequence freely"
        );
        assert_eq!(
            from_sequence.reinstated_pairs, from_design.reinstated_pairs,
            "λ={lambda}: same free fold, same constraint, same classification"
        );
    }
}

/// A caller that already holds the constrained MFE gets the same verdict
/// without folding the constrained half again.
///
/// This is what lets a result line be emitted on two folds rather than three:
/// the fold that produced the structure has already returned the constrained
/// energy, and the scan behind it with no required position is the same scan
/// the gap would run. Every field the check reports must come out identical.
#[test]
fn the_gap_is_the_same_when_the_constrained_energy_is_supplied() {
    let (params, usage) = setup();
    for lambda in LAMBDAS {
        let (d, cds) = design_sticky(&params, &usage, lambda, hard_isolation());
        let folded = sequence_gap(&params, &d.sequence, cds, hard_isolation()).expect("folds");
        let given = sequence_gap_given_constrained(
            &params,
            &d.sequence,
            cds,
            hard_isolation(),
            folded.constrained_mfe_kcal_mol,
        )
        .expect("folds once");

        assert_eq!(
            given.constrained_mfe_kcal_mol, folded.constrained_mfe_kcal_mol,
            "λ={lambda}: the supplied energy is the one reported"
        );
        assert!(
            (given.free_mfe_kcal_mol - folded.free_mfe_kcal_mol).abs() < TOL,
            "λ={lambda}: the free fold is unaffected"
        );
        assert!(
            (given.gap_kcal_mol - folded.gap_kcal_mol).abs() < TOL,
            "λ={lambda}: same gap, {} against {}",
            given.gap_kcal_mol,
            folded.gap_kcal_mol
        );
        assert_eq!(
            given.reinstated_pairs, folded.reinstated_pairs,
            "λ={lambda}: same free fold and constraint, so the same classification"
        );
        assert_eq!(given.free_structure, folded.free_structure, "λ={lambda}");
    }
}

/// A caller holding an allowed structure and its energy gets the verdict of
/// the two-fold check from the free fold alone: on every design here the
/// structure-backed gap and the folded one agree field for field.
#[test]
fn the_gap_is_the_same_when_an_allowed_structure_is_supplied() {
    let (params, usage) = setup();
    for lambda in LAMBDAS {
        let (d, cds) = design_sticky(&params, &usage, lambda, hard_isolation());
        let folded = sequence_gap(&params, &d.sequence, cds, hard_isolation()).expect("folds");
        let given = sequence_gap_given_structure(
            &params,
            &d.sequence,
            &d.structure,
            folded.constrained_mfe_kcal_mol,
            cds,
            hard_isolation(),
        )
        .expect("folds once");

        assert!(
            (given.gap_kcal_mol - folded.gap_kcal_mol).abs() < TOL,
            "λ={lambda}: same gap, {} against {}",
            given.gap_kcal_mol,
            folded.gap_kcal_mol
        );
        assert_eq!(given.free_structure, folded.free_structure, "λ={lambda}");
        assert_eq!(
            given.reinstated_pairs, folded.reinstated_pairs,
            "λ={lambda}"
        );
    }
}

/// The supplied energy is used, not re-derived: an allowed structure above
/// the optimum (here the open chain, energy 0) is reported as the constrained
/// energy, and the gap it gives can only read high — a failed verdict, never a
/// false pass. Without this the test above would pass with the shortcut off.
#[test]
fn a_supplied_structure_above_the_optimum_reads_a_gap_too_high() {
    let (params, usage) = setup();
    let protein = "MKTAYIAKQRQISFVKSHFSRQ";
    let d = design(&params, protein, &usage, 0.99).expect("designs");
    let cds = CdsWindow::new(0, protein.len());
    let inert = RegionConstraints::default();
    let open = ".".repeat(d.sequence.len());
    let given = sequence_gap_given_structure(&params, &d.sequence, &open, 0.0, cds, inert)
        .expect("folds once");
    let folded = sequence_gap(&params, &d.sequence, cds, inert).expect("folds");

    assert_eq!(
        given.constrained_mfe_kcal_mol, 0.0,
        "the open chain's energy"
    );
    assert!(folded.constrained_mfe_kcal_mol < -TOL, "the fixture folds");
    assert!(given.gap_kcal_mol > folded.gap_kcal_mol + TOL);
    assert!(!given.is_modular());
}

/// A structure that does not prove its energy is not trusted. The free
/// fold's own structure, at its own energy, would read a zero gap — but it
/// reinstates pairs the table forbids, so the constrained half is folded and
/// the sticky fixture's real gap comes back. So does an energy the supplied
/// structure does not account for.
#[test]
fn a_structure_that_does_not_hold_falls_back_to_the_constrained_fold() {
    let (params, usage) = setup();
    let (d, cds) = design_sticky(&params, &usage, 0.5, hard_isolation());
    let folded = sequence_gap(&params, &d.sequence, cds, hard_isolation()).expect("folds");
    assert!(
        folded.reinstates() && folded.gap_kcal_mol > TOL,
        "the fixture fails"
    );

    let forbidden = sequence_gap_given_structure(
        &params,
        &d.sequence,
        &folded.free_structure,
        folded.free_mfe_kcal_mol,
        cds,
        hard_isolation(),
    )
    .expect("folds twice");
    let unaccounted = sequence_gap_given_structure(
        &params,
        &d.sequence,
        &d.structure,
        folded.free_mfe_kcal_mol,
        cds,
        hard_isolation(),
    )
    .expect("folds twice");

    for (what, r) in [("forbidden pair", forbidden), ("energy", unaccounted)] {
        assert_eq!(
            r.constrained_mfe_kcal_mol, folded.constrained_mfe_kcal_mol,
            "{what}: the constrained half is folded"
        );
        assert!(!r.is_modular(), "{what}: no false pass");
    }
}

/// **The check's free fold has the scan's grammar.** This molecule's free
/// optimum is a multiloop whose first branch follows 33 unpaired bases, a run
/// LinearFold caps at 30. Under an inert table the constrained fold *is* the
/// free fold, so the gap must be zero; a free fold that missed the multiloop
/// would read the free MFE above the constrained one.
#[test]
fn the_check_folds_a_multiloop_with_a_long_leading_run() {
    let params = Params::load_default().expect("default Turner 2004 parameters");
    let hairpin = "GCGCGAAAGCGC";
    let seq = format!("GGGGG{}{hairpin}AA{hairpin}AACCCCC", "A".repeat(33));
    let linear = derna_lib::sequence_free_mfe(&params, &seq).expect("folds");
    let zuker = derna_lib::sequence_free_fold_zuker(&params, &seq)
        .expect("folds")
        .1;
    assert!(
        zuker.0 < linear.0,
        "the fixture needs the long run: {zuker:?} against {linear:?}"
    );

    let cds = CdsWindow::new(seq.len(), 0);
    let gap = sequence_gap(&params, &seq, cds, RegionConstraints::default()).expect("folds");
    assert!(gap.gap_kcal_mol.abs() < TOL, "gap {}", gap.gap_kcal_mol);
    assert!((gap.free_mfe_kcal_mol - zuker.to_kcal_mol()).abs() < TOL);
    let (_, dense) = rna_mfe::fold(&params, seq.as_bytes()).expect("folds");
    assert_eq!(
        zuker, dense,
        "the check's sparse fold is the dense fold's MFE"
    );
}

/// (5b) The test applies to sequences no designer produced — the case the
/// verify-per-candidate route needs. An all-`A` mRNA cannot pair at all, so it is
/// trivially modular under any constraint.
#[test]
fn sequence_gap_accepts_an_unpairable_sequence_as_modular() {
    let params = Params::load_default().expect("params");
    let seq = "A".repeat(60);
    let cds = CdsWindow::new(10, 10);
    let report = sequence_gap(&params, &seq, cds, hard_isolation()).expect("folds");

    assert!(
        report.gap_kcal_mol.abs() < TOL,
        "poly-A has no structure to lose"
    );
    assert!(!report.reinstates());
}

/// (4) Soft isolation prices the pair rather than excluding it, so the design's
/// reported MFE is the structure's true energy plus exactly one penalty per
/// penalized pair. `emitted_mfe_kcal_mol` recovers the physical number.
#[test]
fn soft_isolation_reported_mfe_is_the_true_energy_plus_one_penalty_per_pair() {
    let (params, usage) = setup();
    let weight = Energy(10); // 0.1 kcal/mol — priced, but far too cheap to deter.
    let soft = RegionConstraints::symmetric_isolation(Enforcement::Penalize(weight));
    let (d, cds) = design_sticky(&params, &usage, 0.99, soft);
    let report = free_fold_gap(&params, &d, cds, soft).expect("diagnostic runs");

    let priced = cross_region_count(&d.structure, cds);
    assert!(
        priced > 0,
        "a 0.1 kcal/mol price must not deter the poly-G stem: {}",
        d.structure
    );
    let emitted = rescore(&report);
    let expected = emitted.mfe_kcal_mol + weight.to_kcal_mol() * priced as f64;
    assert!(
        (d.mfe_kcal_mol - expected).abs() < TOL,
        "reported {} should be the true energy {} plus {priced} × {} penalty",
        d.mfe_kcal_mol,
        emitted.mfe_kcal_mol,
        weight.to_kcal_mol()
    );
    assert!(
        emitted.mfe_kcal_mol < d.mfe_kcal_mol,
        "the penalty inflates the reported MFE above the physical one"
    );
    // The penalty the structure incurs is what closes that split, so the
    // phantom residual is zero even though the two energies differ.
    assert!(
        (emitted.penalty_kcal_mol - weight.to_kcal_mol() * priced as f64).abs() < TOL,
        "one penalty per priced pair"
    );
    assert!(emitted.phantom_kcal_mol.abs() < TOL);
}

/// (6) The domain edge the re-score shortcut could not serve. At λ=0 the MFE
/// term carries zero weight, so every structure of the emitted sequence is
/// equally optimal and the traceback returned is arbitrary — it is the
/// all-unpaired one. A verdict read off that structure charges the design for
/// every pair its sequence could have formed; a verdict from a real constrained
/// fold of the same sequence does not.
///
/// The fixture needs a CDS with structure of its own, so that the arbitrary
/// traceback and the constrained optimum are different structures at all: the
/// poly-proline fixture has no allowed pair to lose, which is exactly the case
/// where the shortcut happens to be right for the wrong reason.
#[test]
fn the_verdict_survives_lambda_zero() {
    let (params, usage) = setup();
    let protein = "MKTAYIAKQRQISFVKSHFSR";
    let d = design_with_utr(
        &params,
        protein,
        &usage,
        0.0,
        STICKY_UTR5,
        "",
        hard_isolation(),
        ScanConfig::default(),
    )
    .expect("designs");
    let cds = CdsWindow::new(STICKY_UTR5.len(), protein.len());
    let report = free_fold_gap(&params, &d, cds, hard_isolation()).expect("diagnostic runs");
    let emitted = rescore(&report);

    assert!(
        d.structure.bytes().all(|c| c == b'.'),
        "the premise of this test is that λ=0 returns an arbitrary structure: {}",
        d.structure
    );

    assert!(
        report.gap_kcal_mol > -TOL,
        "the gap cannot be negative at any λ, got {}",
        report.gap_kcal_mol
    );
    // The verdict is a property of the sequence, not of which co-optimal
    // structure λ=0 happened to return, so it must agree with the sequence-only
    // test — which is the whole content of the fix.
    let from_sequence = sequence_gap(&params, &d.sequence, cds, hard_isolation()).expect("folds");
    assert!(
        (report.gap_kcal_mol - from_sequence.gap_kcal_mol).abs() < TOL,
        "λ=0: gap {} must match the sequence-only gap {}",
        report.gap_kcal_mol,
        from_sequence.gap_kcal_mol
    );
    assert!(
        (report.constrained_mfe_kcal_mol - emitted.mfe_kcal_mol).abs() > TOL,
        "the fixture is only interesting if λ=0 really does return a structure that is not \
         constrained-optimal: re-score {}, constrained fold {}",
        emitted.mfe_kcal_mol,
        report.constrained_mfe_kcal_mol
    );
    // The re-score keeps its own job at λ=0: the reported energy still matches
    // the reported structure, arbitrary as that structure is.
    assert!(
        emitted.phantom_kcal_mol.abs() < TOL,
        "λ=0: reported {} vs re-scored {}",
        d.mfe_kcal_mol,
        emitted.mfe_kcal_mol
    );
}

/// (7) The breakdown describes the whole free fold, enforced classes and
/// permitted ones alike, and the classes the constraint refuses are exactly the
/// reinstated pairs.
#[test]
fn the_breakdown_covers_permitted_classes_and_agrees_with_the_verdict() {
    let (params, usage) = setup();
    let (d, cds) = design_sticky(&params, &usage, 0.99, hard_isolation());
    let report = free_fold_gap(&params, &d, cds, hard_isolation()).expect("diagnostic runs");

    assert_eq!(
        report.classes.total(),
        pairs_of(&report.free_structure).len(),
        "every pair of the free fold lands in exactly one class"
    );
    // Isolation forbids 5′↔CDS; the fixture's free fold rebuilds exactly that,
    // and the count attributes the failing verdict to the class responsible.
    assert_eq!(
        report.classes.count(PairClass::Utr5Cds),
        report.reinstated_pairs.len()
    );
    assert!(report.classes.count(PairClass::Utr5Cds) > 0);
    // A permitted class is reported all the same — this is the fixture's own
    // intra-CDS structure, which no table objected to.
    assert_eq!(
        report.classes.count(PairClass::CdsCds)
            + report.classes.count(PairClass::Utr5Utr5)
            + report.classes.count(PairClass::Utr5Cds),
        report.classes.total(),
        "with no 3′ flank the fixture can only reach the three 5′/CDS classes"
    );
}

/// The constrained Zuker fold (the sparse fold with every forbidden pair cut)
/// agrees with the scan's constrained MFE on sequences no designer produced,
/// under both hard tables, and the constraint binds on some of them: the gap is
/// positive there, so a fold that ignored the constraint would fail.
#[test]
fn the_constrained_zuker_fold_matches_the_scan() {
    use derna_lib::{sequence_constrained_mfe, sequence_constrained_mfe_zuker, sequence_free_mfe};
    let params = Params::load_default().expect("params");
    let tables = [
        hard_isolation(),
        RegionConstraints::full_mutual_isolation(Enforcement::Forbid),
    ];
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut bound = 0;
    for k in 0..12 {
        let seq: String = (0..96)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // A sticky 5′ flank on half the cases makes the constraint bind.
                b"ACGU"[(state % 4) as usize] as char
            })
            .collect();
        let seq = if k % 2 == 0 {
            format!("GGGGCCCC{}", &seq[8..])
        } else {
            seq
        };
        let cds = CdsWindow::new(24, 18);
        for rc in tables {
            let scan = sequence_constrained_mfe(&params, &seq, cds, rc).expect("scan folds");
            let zuker =
                sequence_constrained_mfe_zuker(&params, &seq, cds, rc).expect("zuker folds");
            assert!(
                (scan - zuker).abs() < TOL,
                "case {k}: scan {scan} vs zuker {zuker}"
            );
            let free = sequence_free_mfe(&params, &seq)
                .expect("free folds")
                .to_kcal_mol();
            if zuker > free + TOL {
                bound += 1;
            }
        }
    }
    assert!(
        bound > 0,
        "the constraint never bound: the test would pass with it ignored"
    );
    let soft = RegionConstraints::symmetric_isolation(Enforcement::Penalize(Energy(100)));
    let sticky = format!("{STICKY_UTR5}{}", "CCC".repeat(STICKY_PROTEIN.len()));
    assert!(
        sequence_constrained_mfe_zuker(&params, &sticky, CdsWindow::new(8, 5), soft).is_err(),
        "a soft table is refused, not folded as if allowed"
    );
}

/// The verdict alone agrees with the full gap on every case, hard, soft and
/// inert, with the constrained energy it keeps; and both verdicts occur, so a
/// check that always refuted, or always passed, would fail.
#[test]
fn the_verdict_alone_agrees_with_the_gap() {
    use derna_lib::sequence_modular_mfe;
    let params = Params::load_default().expect("params");
    let tables = [
        hard_isolation(),
        RegionConstraints::full_mutual_isolation(Enforcement::Forbid),
        RegionConstraints::symmetric_isolation(Enforcement::Penalize(Energy(100))),
        RegionConstraints::default(),
    ];
    let mut state = 0x6a09_e667_f3bc_c908_u64;
    let (mut passed, mut refuted) = (0, 0);
    for k in 0..16 {
        let seq: String = (0..96)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                b"ACGU"[(state % 4) as usize] as char
            })
            .collect();
        let seq = if k % 2 == 0 {
            format!("GGGGCCCC{}", &seq[8..])
        } else {
            seq
        };
        let cds = CdsWindow::new(24, 18);
        for rc in tables {
            let gap = sequence_gap(&params, &seq, cds, rc).expect("gap");
            let alone = sequence_modular_mfe(&params, &seq, cds, rc).expect("verdict");
            let expected = gap.is_modular().then_some(gap.constrained_mfe_kcal_mol);
            assert_eq!(alone, expected, "case {k}");
            if alone.is_some() {
                passed += 1;
            } else {
                refuted += 1;
            }
        }
    }
    assert!(
        passed > 0 && refuted > 0,
        "{passed} passed, {refuted} refuted"
    );
}

/// The verdict on a chain of sequences that each differ from the one before
/// near their 3′ end — as designs spelled one after another do — reads the
/// same copying the previous sequence's tables as folding fresh, under hard
/// tables.
///
/// Two regimes, because one alphabet cannot produce both verdicts. A random
/// molecule's free fold almost always reaches across the boundary, so every
/// design is refuted and only the Zuker folds' refuting path is exercised. A
/// molecule with no `U` at all and poly-A flanks admits no cross-region pair
/// — A pairs only with U — so no free fold can break the table, every design
/// is modular, and the path that runs the scan is exercised
/// instead. The tally spans both, so neither verdict can go untested.
#[test]
fn the_reusing_verdict_is_the_fresh_one() {
    use derna_lib::{sequence_modular_mfe, sequence_modular_mfe_reusing};
    let params = Params::load_default().expect("params");
    let cds = CdsWindow::new(24, 18);
    let mut state = 0x9b05_688c_2b3e_6c1f_u64;
    let mut draw = |n: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % n
    };
    let (mut passed, mut refuted) = (0, 0);
    for table in [
        hard_isolation(),
        RegionConstraints::full_mutual_isolation(Enforcement::Forbid),
    ] {
        for alphabet in [&b"ACGU"[..], &b"ACG"[..]] {
            // The U-free regime also needs flanks that cannot pair with the
            // coding window, which poly-A is when nothing is a U.
            let mut seq: Vec<u8> = (0..96)
                .map(|_| alphabet[draw(alphabet.len() as u64) as usize])
                .collect();
            if alphabet.len() == 4 {
                seq[..8].copy_from_slice(b"GGGGCCCC");
            } else {
                seq[..24].fill(b'A');
                seq[78..].fill(b'A');
                seq[30..36].copy_from_slice(b"GGGCCC");
            }
            let mut base = None;
            for k in 0..40 {
                // Re-spell one or two positions among the coding window's last
                // codons, as the next design in spelling order does.
                for _ in 0..1 + draw(2) {
                    let at = 24 + 36 + draw(18) as usize;
                    seq[at] = alphabet[draw(alphabet.len() as u64) as usize];
                }
                let s = String::from_utf8(seq.clone()).unwrap();
                let fresh = sequence_modular_mfe(&params, &s, cds, table).expect("fresh");
                let (reused, mfes) =
                    sequence_modular_mfe_reusing(&params, &s, cds, table, base.as_ref())
                        .expect("reused");
                assert_eq!(reused, fresh, "step {k} of alphabet {alphabet:?}");
                base = Some(mfes);
                if fresh.is_some() {
                    passed += 1;
                } else {
                    refuted += 1;
                }
            }
        }
    }
    assert!(
        passed > 0 && refuted > 0,
        "{passed} passed, {refuted} refuted: one verdict went untested"
    );
}

/// Folding a sequence that differs from a base in one stretch, copying the
/// base's cells whose spans avoid the stretch, gives the MFEs a fresh fold
/// does, constrained and free; and the change moves them on some cases, so a
/// fold that copied everything would fail.
#[test]
fn reused_zuker_folds_match_fresh_ones() {
    use derna_lib::sequence_mfes_reusing;
    let params = Params::load_default().expect("params");
    let cds = CdsWindow::new(24, 18);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut draw = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        b"ACGU"[(state % 4) as usize]
    };
    let mut moved = 0;
    for k in 0..10 {
        let mut base: Vec<u8> = (0..96).map(|_| draw()).collect();
        if k % 2 == 0 {
            base[..8].copy_from_slice(b"GGGGCCCC");
        }
        let mut next = base.clone();
        let at = 24 + 3 * (k % 6);
        for b in &mut next[at..at + 6] {
            *b = draw();
        }
        let (base, next) = (
            String::from_utf8(base).expect("ACGU"),
            String::from_utf8(next).expect("ACGU"),
        );
        for rc in [hard_isolation(), RegionConstraints::default()] {
            let held =
                sequence_mfes_reusing(&params, &base, cds, rc, None, &|_| true).expect("folds");
            let reused = sequence_mfes_reusing(&params, &next, cds, rc, Some(&held), &|_| true)
                .expect("folds");
            let fresh =
                sequence_mfes_reusing(&params, &next, cds, rc, None, &|_| true).expect("folds");
            assert!(
                (reused.constrained - fresh.constrained).abs() < TOL,
                "case {k}: constrained {} vs {}",
                reused.constrained,
                fresh.constrained
            );
            assert_eq!(reused.free, fresh.free, "case {k}: free");
            if (fresh.constrained - held.constrained).abs() > TOL {
                moved += 1;
            }
        }
    }
    assert!(
        moved > 0,
        "no change moved an MFE: copying everything would pass"
    );
}

/// The free fold copying cells from a pool of earlier folds — each row from
/// whichever fold differs from it least — returns the fresh fold's MFE and
/// structure, on a run of sequences that each re-spell a few positions
/// anywhere, as neighbouring front points do; and it does copy cells.
#[test]
fn the_free_fold_copying_from_a_pool_is_the_fresh_one() {
    use derna_lib::{sequence_free_fold_zuker, sequence_free_fold_zuker_reusing, FreeFoldTables};
    let params = Params::load_default().expect("params");
    let mut state = 0x5be0_cd19_137e_2179_u64;
    let mut draw = |n: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % n
    };
    let mut seq: Vec<u8> = (0..120).map(|_| b"ACGU"[draw(4) as usize]).collect();
    let mut pool: Vec<FreeFoldTables> = Vec::new();
    let mut copied = 0;
    for k in 0..40 {
        for _ in 0..1 + draw(3) {
            let at = 20 + draw(80) as usize;
            seq[at] = b"ACGU"[draw(4) as usize];
        }
        let s = String::from_utf8(seq.clone()).unwrap();
        let fresh = sequence_free_fold_zuker(&params, &s).expect("fresh");
        let held: Vec<&FreeFoldTables> = pool.iter().collect();
        let (reused, tables) =
            sequence_free_fold_zuker_reusing(&params, &s, &held).expect("reused");
        assert_eq!(reused, fresh, "step {k}");
        copied += tables.copied();
        if pool.len() == 3 {
            pool.remove(0);
        }
        pool.push(tables);
    }
    assert!(copied > 0, "no cell was copied");
}
