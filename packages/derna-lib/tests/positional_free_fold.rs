//! The free-fold verdict under a positional structure constraint:
//! `free_fold_gap_with_structure`.
//!
//! A positional statement binds the **designed** structure, so the emitted
//! sequence folded with nothing forbidden can re-pair a position marked
//! unpaired, take a partner the statement reserved for someone else, or decline
//! a pair it required. The verdict is the energy gap between the best fold the
//! whole statement allows and the free fold, as it is for a region table; the
//! two lists beside it attribute a failing verdict in the two directions a
//! positional statement has.
//!
//! # Both directions, per flag
//!
//! Each of `--unpaired`, `--forbid-pair`, `--helix` and `--structure` is
//! exercised twice against **one** constraint value, on two sequences: one the
//! verdict must fail and one it must pass. One direction alone passes
//! vacuously — a verdict wired to a constant reads correct on whichever half it
//! was built against.
//!
//! Where it can, the pair also shows the constraint **binds**. `--unpaired` and
//! `--forbid-pair` take the design the solver returns *without* the constraint
//! as the failing sequence and the design it returns *with* it as the passing
//! one: the statement changed what was emitted, so the zero on the second
//! sequence is not a zero against a constraint that forbade nothing. `--helix`
//! fails on the design emitted at `λ = 0`, where the objective is adaptation
//! alone, and passes on the one emitted at `λ = 1` under the same helix;
//! `--structure` fails on a design emitted for a different statement.
//!
//! Constraints are built through each flag's own grammar
//! (`PositionalSpec::add_unpaired` and its siblings, resolved against a
//! `Layout`), so a change to how a flag lowers to marks is covered here and not
//! only where the flag is parsed.

use derna_lib::{
    free_fold_gap, free_fold_gap_with_structure, CdsWindow, Design, FreeFoldError, FreeFoldGap,
    Layout, Params, PositionalSpec, RegionConstraints, StructureConstraints,
};

/// Thirty residues of `GLPE_HUMAN`, the protein every fixture below spells, with
/// no flanks: the CDS is the whole 90-nucleotide molecule.
const RESIDUES: u32 = 30;

/// The design at `λ = 1` under no positional constraint — the molecule the
/// solver prefers when nothing is asked of its structure.
const UNCONSTRAINED: &str =
    "AUGUAUGGCAAGAUCAUCUUCGUCCUGCUGCUCUCGGGGAUCGUGUCGAUCUCCGCGAGCAGCACGACGGGGGUGGCCAUGCAUACUUCU";

/// The design at `λ = 0`, where the objective is adaptation alone and the
/// structure is whatever the best-adapted codons happen to fold into.
const MAX_CAI: &str =
    "AUGUACGGCAAGAUCAUCUUCGUGCUGCUGCUGAGCGGCAUCGUGAGCAUCAGCGCCAGCAGCACCACCGGCGUGGCCAUGCACACCAGC";

/// The design at `λ = 1` under `--unpaired cds:5..cds:40`.
const UNDER_UNPAIRED: &str =
    "AUGUAUGGUAAAAUUAUUUUUGUUCUUCUUCUUUCUGGUAUCGUGAGCAUCUCCGCCUCCAGUACGACUGGGGUGGCGAUGCACACGAGU";

/// The design at `λ = 1` under `--forbid-pair 1:84`.
const UNDER_FORBID_PAIR: &str =
    "AUGUAUGGGAAGAUCAUCUUCGUCCUGCUGCUCUCGGGGAUCGUGUCGAUCUCCGCGAGCAGCACGACGGGGGUGGCUAUGCAUACAUCU";

/// The design at `λ = 1` under `--helix cds:6:cds:33:3`.
const UNDER_HELIX: &str =
    "AUGUACGGCAAGAUCAUUUUUGUGCUCUUGCUGUCUGGUAUCGUCUCCAUCUCGGCGUCGUCGACGACCGGGGUGGCGAUGCAUACCAGU";

fn params() -> Params {
    Params::load_default().expect("the built-in Turner 2004 table")
}

fn layout() -> Layout {
    Layout::new(0, RESIDUES, 0)
}

fn cds() -> CdsWindow {
    CdsWindow::new(0, RESIDUES as usize)
}

/// A positional statement built the way a flag builds it: `write` records the
/// flag's own value on a fresh spec, which is then resolved against the layout.
fn constraint(write: impl FnOnce(&mut PositionalSpec)) -> StructureConstraints {
    let mut spec = PositionalSpec::default();
    write(&mut spec);
    let resolved = spec.resolve(&layout()).expect("a satisfiable statement");
    assert!(
        !resolved.structure.is_inert(),
        "the flag lowered to an inert statement, so neither direction below \
         would be testing anything"
    );
    resolved.structure
}

/// The verdict for `sequence` under `structure` and no region table.
///
/// The emitted structure handed in is the all-unpaired one: it feeds only the
/// phantom-free re-score, which is a different check, and every fixture's own
/// designed structure is a property of the solve rather than of the verdict.
fn verdict(sequence: &str, structure: &StructureConstraints) -> FreeFoldGap {
    let design = Design {
        sequence: sequence.to_string(),
        structure: ".".repeat(sequence.len()),
        mfe_kcal_mol: 0.0,
        cai_log: 0.0,
        cai: 1.0,
        utr_profile_log: 0.0,
    };
    free_fold_gap_with_structure(
        &params(),
        &design,
        cds(),
        RegionConstraints::default(),
        structure,
    )
    .expect("both folds run")
}

/// Assert the verdict fails `sequence`, and return it.
fn fails(sequence: &str, structure: &StructureConstraints) -> FreeFoldGap {
    let v = verdict(sequence, structure);
    assert!(
        !v.is_modular(),
        "the verdict passed a sequence whose free fold breaks the statement: \
         gap {:+.2}, free {}",
        v.gap_kcal_mol,
        v.free_structure
    );
    assert!(
        v.gap_kcal_mol > 0.0,
        "a failing verdict is a positive gap, got {:+.2}",
        v.gap_kcal_mol
    );
    assert!(
        v.violates(),
        "a positive gap with neither list populated attributes the failure to \
         nothing: free {}",
        v.free_structure
    );
    v
}

/// The pairs of a dot-bracket, so an assertion can ask whether one named pair
/// is among them rather than read characters at positions.
fn pairs_of(structure: &str) -> Vec<(usize, usize)> {
    let mut open = Vec::new();
    let mut out = Vec::new();
    for (i, c) in structure.char_indices() {
        match c {
            '(' => open.push(i),
            ')' => out.push((open.pop().expect("a balanced dot-bracket"), i)),
            _ => {}
        }
    }
    out.sort_unstable();
    out
}

/// Assert the verdict passes `sequence`, and return it.
fn passes(sequence: &str, structure: &StructureConstraints) -> FreeFoldGap {
    let v = verdict(sequence, structure);
    assert!(
        v.is_modular(),
        "the verdict failed a sequence whose own fold already satisfies the \
         statement: gap {:+.2}, reinstated {:?}, unpaired required {:?}",
        v.gap_kcal_mol,
        v.reinstated_pairs,
        v.unpaired_required_positions
    );
    assert_eq!(
        v.constrained_mfe_kcal_mol, v.free_mfe_kcal_mol,
        "a zero gap is the two optima agreeing"
    );
    v
}

#[test]
fn an_unpaired_range_is_reported_against_the_free_fold_and_passes_a_design_that_keeps_it() {
    let range = constraint(|s| s.add_unpaired("cds:5..cds:40").expect("a range"));

    // The molecule the solver prefers with nothing asked of it folds straight
    // through the range, so the statement costs energy and the verdict says so.
    let broken = fails(UNCONSTRAINED, &range);
    assert!(
        !broken.reinstated_pairs.is_empty(),
        "an unpaired range only ever forbids, so its failure is reinstated pairs"
    );
    assert!(
        broken.reinstated_pairs.iter().all(|&(i, j)| {
            // `cds:5..cds:40` with no 5′ flank, read back as absolute 0-based
            // positions. An `x` is a claim about the *position*, so one endpoint
            // inside is enough — the partner may lie anywhere.
            let inside = |p: usize| (4..=39).contains(&p);
            inside(i) || inside(j)
        }),
        "every reinstated pair has an endpoint in the range: {:?}",
        broken.reinstated_pairs
    );
    assert!(
        broken.unpaired_required_positions.is_empty(),
        "an unpaired range requires no pair, so nothing can be left unpaired"
    );

    // The design made under the same range is modular: its own fold keeps the
    // range clear. The pair of verdicts is what shows the range bound — the
    // solver emitted a different molecule for it.
    let kept = passes(UNDER_UNPAIRED, &range);
    assert!(kept.reinstated_pairs.is_empty());
    assert_ne!(UNCONSTRAINED, UNDER_UNPAIRED);
}

#[test]
fn a_forbidden_pair_is_reported_against_the_free_fold_and_passes_a_design_that_avoids_it() {
    // The outermost pair of the unconstrained optimum, in whole-molecule
    // 1-based coordinates: position 1 against position 84.
    let forbidden = constraint(|s| s.add_forbidden_pair("1:84").expect("a pair"));

    let broken = fails(UNCONSTRAINED, &forbidden);
    assert!(
        broken.reinstated_pairs.contains(&(0, 83)),
        "the free fold forms the one pair the statement names: {:?}",
        broken.reinstated_pairs
    );
    assert!(
        broken.unpaired_required_positions.is_empty(),
        "--forbid-pair requires nothing of either endpoint"
    );

    // The design made under the same statement folds into something else at
    // that position — the pair is gone from its own optimum, not merely priced.
    let avoided = passes(UNDER_FORBID_PAIR, &forbidden);
    assert!(
        !pairs_of(&avoided.free_structure).contains(&(0, 83)),
        "the passing design's own fold still forms the named pair: {}",
        avoided.free_structure
    );
    assert_ne!(UNCONSTRAINED, UNDER_FORBID_PAIR);
}

#[test]
fn a_required_helix_is_reported_against_the_free_fold_and_passes_a_design_that_forms_it() {
    let helix = constraint(|s| s.add_helix("cds:6:cds:33:3").expect("a helix"));
    let required: Vec<(u32, u32)> = helix.required_pairs();
    assert_eq!(required.len(), 3, "three pairs: {required:?}");

    // At λ = 0 the objective is adaptation alone, so the emitted sequence only
    // has to *admit* the helix. Its own fold declines it.
    let broken = fails(MAX_CAI, &helix);
    assert!(
        !broken.unpaired_required_positions.is_empty(),
        "the require half's failure is a required position the free fold left \
         unpaired: free {}",
        broken.free_structure
    );
    assert!(
        broken.unpaired_required_positions.iter().all(|&p| required
            .iter()
            .any(|&(i, j)| p as u32 == i || p as u32 == j)),
        "every position named is one the helix requires: {:?} of {required:?}",
        broken.unpaired_required_positions
    );

    // The design made under the same helix folds into it unaided.
    let formed = passes(UNDER_HELIX, &helix);
    assert!(
        formed.unpaired_required_positions.is_empty(),
        "a modular design leaves no requirement unmet"
    );
    assert!(
        required.iter().all(|&(i, j)| {
            let b = formed.free_structure.as_bytes();
            b[i as usize] == b'(' && b[j as usize] == b')'
        }),
        "and the free fold exhibits the helix rather than merely tying its \
         energy: {}",
        formed.free_structure
    );
}

#[test]
fn a_structure_string_is_reported_against_the_free_fold_and_passes_a_design_that_honours_it() {
    // One pair of the unconstrained optimum's own outer helix, written as the
    // whole-mRNA string the flag takes. Positions 2 and 83 in the string's own
    // 1-based reading, which the brackets name by where they sit.
    let written = format!(".({}){}", ".".repeat(80), ".".repeat(7));
    assert_eq!(written.len(), 3 * RESIDUES as usize);
    let string = constraint(|s| s.set_structure(written.clone()));
    assert_eq!(string.required_pairs(), vec![(1, 82)]);

    // A design made for a different statement can spell the pair and folds
    // without it, which is the require half failing through the general form.
    let broken = fails(UNDER_UNPAIRED, &string);
    assert!(
        !broken.unpaired_required_positions.is_empty(),
        "free {}",
        broken.free_structure
    );

    let honoured = passes(UNCONSTRAINED, &string);
    assert!(honoured.unpaired_required_positions.is_empty());
    assert!(pairs_of(&honoured.free_structure).contains(&(1, 82)));
}

#[test]
fn the_structure_string_and_the_unpaired_range_lower_to_one_statement() {
    // `--unpaired` is sugar over the string, so the verdict cannot read
    // differently for the two spellings of one request. Asserted on the lowered
    // statement rather than on a pair of verdicts, which would also pass if both
    // spellings were inert.
    let range = constraint(|s| s.add_unpaired("cds:5..cds:40").expect("a range"));
    let written = format!("{}{}{}", ".".repeat(4), "x".repeat(36), ".".repeat(50));
    assert_eq!(written.len(), 3 * RESIDUES as usize);
    let string = constraint(|s| s.set_structure(written));
    assert_eq!(range, string);
}

#[test]
fn an_inert_statement_is_the_region_verdict_unchanged() {
    // The floor the positional entry point is measured against: a run that names
    // no positional flag must get the report it got before there was one, field
    // for field — including the two lists, which are then empty rather than
    // absent.
    let params = params();
    let design = Design {
        sequence: UNCONSTRAINED.to_string(),
        structure: ".".repeat(UNCONSTRAINED.len()),
        mfe_kcal_mol: 0.0,
        cai_log: 0.0,
        cai: 1.0,
        utr_profile_log: 0.0,
    };
    let table = RegionConstraints::default();
    let inert = StructureConstraints::none(layout().len());
    assert!(inert.is_inert());
    let with = free_fold_gap_with_structure(&params, &design, cds(), table, &inert)
        .expect("both folds run");
    let without = free_fold_gap(&params, &design, cds(), table).expect("both folds run");
    assert_eq!(with, without);
    assert!(with.unpaired_required_positions.is_empty());
}

#[test]
fn a_requirement_no_structure_of_the_sequence_can_meet_is_refused_rather_than_answered() {
    // The instrument's failing case: the constrained half has no optimum to
    // report, because the two positions the statement pairs cannot pair in this
    // sequence. A number returned here would be a verdict against a constraint
    // the sequence cannot satisfy at all.
    let mut spec = PositionalSpec::default();
    spec.add_helix("cds:1:cds:30:1").expect("a helix");
    let one_pair = spec
        .resolve(&layout())
        .expect("a well-formed statement")
        .structure;
    // Positions 1 and 30 of this CDS spell A and G, which is not a canonical
    // pair, so no structure of this sequence carries the one it requires.
    let seq = MAX_CAI;
    assert_eq!(&seq[0..1], "A");
    assert_eq!(&seq[29..30], "G");
    let design = Design {
        sequence: seq.to_string(),
        structure: ".".repeat(seq.len()),
        mfe_kcal_mol: 0.0,
        cai_log: 0.0,
        cai: 1.0,
        utr_profile_log: 0.0,
    };
    let err = free_fold_gap_with_structure(
        &params(),
        &design,
        cds(),
        RegionConstraints::default(),
        &one_pair,
    )
    .expect_err("no structure of this sequence pairs them");
    assert!(matches!(err, FreeFoldError::NoSatisfyingStructure), "{err}");
}
