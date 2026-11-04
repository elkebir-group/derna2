//! The motif rule `derna design` enforces, from both sides.
//!
//! Two claims, and they need different tests.
//!
//! 1. **Where the two window readings agree, `derna` agrees with the
//!    instrument.** With no flanks the coding sequence is the whole molecule,
//!    so there is no straddler to disagree about and the overlapping window
//!    over the CDS is the instrument's global enforcement. That is an equality
//!    against the standalone per-weight design binary, and it is what says the
//!    motif path is the same path.
//! 2. **Where they differ, `derna` forbids what the instrument permits.** With
//!    flanks, an occurrence straddling a CDS/UTR boundary is permitted by the
//!    entirely-inside reading and forbidden here. The test constructs that
//!    occurrence rather than hoping for one: it reads the design the
//!    unconstrained run emits, takes the pattern that design spells across the
//!    boundary, and forbids it.
//!
//! Without the second test the rule would be untested — every case in the
//! first one passes under either reading, by construction.

use derna_cli::design::{self, Solved};
use derna_cli::molecule::Molecule;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fasta() -> String {
    root()
        .join("packages/derna-codon-dp/tests/data/GLPE_HUMAN_20.fasta")
        .to_str()
        .unwrap()
        .to_string()
}

fn hbb25() -> (String, String) {
    let text =
        std::fs::read_to_string(root().join("packages/derna/tests/data/flank-pairs.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let flanks = &v["hbb25"];
    (
        flanks["utr5"].as_str().unwrap().to_string(),
        flanks["utr3"].as_str().unwrap().to_string(),
    )
}

/// Design at one weight under the given words.
fn design_once(words: &[String]) -> Vec<Solved> {
    let args = design::parse(words.to_vec()).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    design::solve_constrained(&m, &args.lambdas, 1, &args.sequence, None).unwrap()
}

fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

#[test]
fn with_no_flanks_the_motif_set_is_the_instruments_global_one() {
    // The fixture is the per-weight design binary's output for
    // `--motif GGGG --lambda 0.5`, with no flanks and no region, so both sides
    // enforce over the whole molecule.
    let expected: serde_json::Value = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/design-motif-global.jsonl"),
    )
    .unwrap()
    .lines()
    .map(|l| serde_json::from_str(l).unwrap())
    .next()
    .unwrap();

    let got = design_once(&words(&[
        "--protein",
        &fasta(),
        "--lambda",
        "0.5",
        "--motif",
        "GGGG",
    ]));
    assert_eq!(got.len(), 1);
    let design = &got[0].design;
    assert_eq!(expected["rna_seq"].as_str().unwrap(), design.sequence);
    assert_eq!(expected["structure"].as_str().unwrap(), design.structure);
    assert_eq!(
        expected["mfe_kcal_mol"].as_f64().unwrap(),
        design.mfe_kcal_mol
    );
    assert_eq!(expected["cai_log"].as_f64().unwrap(), design.cai_log);
    assert!(!design.sequence.contains("GGGG"), "the motif survived");
}

#[test]
fn a_motif_the_design_spells_across_the_boundary_is_forbidden() {
    let (utr5, utr3) = hbb25();
    let free = words(&[
        "--protein",
        &fasta(),
        "--lambda",
        "0.5",
        "--stop",
        "none",
        "--utr5",
        &utr5,
        "--utr3",
        &utr3,
        "--forbid",
        "none",
    ]);
    let unconstrained = design_once(&free);
    let sequence = unconstrained[0].design.sequence.clone();

    // The pattern the unconstrained design spells across the 3′ boundary: the
    // last two coding nucleotides and the first two of the fixed flank. The 3′
    // boundary is the one to use — a 5′ straddler reaches into the start
    // codon, which the protein fixes, so no design could avoid it and the
    // space would be empty rather than merely narrower. That case has a test
    // of its own below.
    let boundary = sequence.len() - utr3.len();
    let coding = &sequence[utr5.len()..boundary];
    // Widen the straddler until it occurs nowhere inside the coding sequence,
    // so that forbidding it is a claim about the straddling occurrence alone.
    // The narrowest such pattern is the one to use: a wider one forbids more
    // than the boundary.
    let straddler = (4..=8)
        .map(|len| sequence[boundary + 2 - len..boundary + 2].to_string())
        .find(|pattern| !coding.contains(pattern))
        .expect("a straddling pattern the coding sequence does not also spell");

    // The entirely-inside reading permits it: the occurrence is not wholly
    // within the coding sequence. The overlapping reading forbids it.
    let mut constrained = free.clone();
    constrained.extend(words(&["--motif", &straddler]));
    let repaired = design_once(&constrained);
    let got = repaired[0].design.sequence.clone();
    assert!(
        got.starts_with(&utr5) && got.ends_with(&utr3),
        "a fixed flank changed, which no design may do"
    );
    assert!(
        !got[boundary - 3..].contains(&straddler),
        "the straddling occurrence survived: {straddler} is still spelled at the boundary"
    );
}

#[test]
fn an_occurrence_inside_a_fixed_utr_is_reported_rather_than_refused() {
    let (utr5, utr3) = hbb25();
    // A pattern taken from the middle of the fixed 5′ UTR: no codon assignment
    // can remove it, so the run must proceed.
    let inside_flank = &utr5[5..11];
    let args = design::parse(words(&[
        "--protein",
        &fasta(),
        "--lambda",
        "0.5",
        "--stop",
        "none",
        "--utr5",
        &utr5,
        "--utr3",
        &utr3,
        "--forbid",
        "none",
        "--motif",
        inside_flank,
    ]))
    .unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let reported = args.sequence.unremovable(&m).unwrap();
    assert!(
        reported.iter().any(|(side, motif, at)| *side == "5′ UTR"
            && motif == inside_flank
            && utr5[*at - 1..].starts_with(inside_flank)),
        "the occurrence in the fixed flank was not reported: {reported:?}"
    );
    // And the design still comes back, with the flank untouched.
    let solved = design::solve_constrained(&m, &args.lambdas, 1, &args.sequence, None).unwrap();
    assert!(solved[0].design.sequence.starts_with(&utr5));
}

#[test]
fn a_window_over_the_whole_coding_sequence_is_no_window_at_all() {
    // `--motif-window 1..-1` names every coding position, which is what a run
    // with no window enforces over. The two must therefore give the same
    // design — the cheapest check there is on the window arithmetic, since an
    // off-by-one at either end would show up here and in no equality (the
    // instrument's window rule is a different rule).
    let base = words(&["--protein", &fasta(), "--lambda", "0.5", "--motif", "GGC"]);
    let mut windowed = base.clone();
    windowed.extend(words(&["--motif-window", "1..-1"]));

    let without = design_once(&base);
    let with = design_once(&windowed);
    assert_eq!(without[0].design.sequence, with[0].design.sequence);
    assert_eq!(without[0].design.mfe_kcal_mol, with[0].design.mfe_kcal_mol);
    assert_eq!(without[0].design.cai_log, with[0].design.cai_log);

    // And the motif really is enforced in both, or the equality is between
    // two unconstrained runs.
    assert!(!without[0].design.sequence.contains("GGC"));
    assert!(!with[0].design.sequence.contains("GGC"));
}

#[test]
fn a_motif_window_narrows_enforcement_to_its_coding_positions() {
    // A four-nucleotide pattern the design spells once, near the 5′ end of the
    // coding sequence. Four rather than two: a dinucleotide can be
    // unavoidable — glycine is `GGx`, so `GG` empties the space for this
    // protein — and a window test needs a pattern the design can give up.
    let free = words(&["--protein", &fasta(), "--lambda", "0.5"]);
    let sequence = design_once(&free)[0].design.sequence.clone();
    let pattern = sequence[3..7].to_string();
    let occurrences = sequence.matches(&pattern).count();
    assert_eq!(
        occurrences, 1,
        "{pattern} occurs {occurrences} times, not once"
    );

    // A window past the occurrence leaves it alone...
    let mut away = free.clone();
    away.extend(words(&["--motif", &pattern, "--motif-window", "10..-1"]));
    assert_eq!(
        design_once(&away)[0].design.sequence,
        sequence,
        "a window over positions 10.. changed a design whose only occurrence is at 4..7"
    );

    // ...and one over it does not.
    let mut over = free.clone();
    over.extend(words(&["--motif", &pattern, "--motif-window", "1..-1"]));
    let narrowed = design_once(&over)[0].design.sequence.clone();
    assert_ne!(narrowed, sequence);
    assert!(!narrowed.contains(&pattern));
}

#[test]
fn a_straddler_over_fixed_sequence_empties_the_space_and_says_so() {
    // The rule's sharp edge. An occurrence straddling the 5′ boundary reaches
    // into the start codon, which the protein fixes, and the flank is fixed
    // too, so such a pattern cannot be designed away at all. It is not wholly
    // inside a UTR, so it is forbidden rather than reported, and the run is
    // refused naming the set. This is worth pinning: a reader of the warning
    // path above might expect a warning here instead.
    let (utr5, utr3) = hbb25();
    let free = words(&[
        "--protein",
        &fasta(),
        "--lambda",
        "0.5",
        "--stop",
        "none",
        "--utr5",
        &utr5,
        "--utr3",
        &utr3,
        "--forbid",
        "none",
    ]);
    let sequence = design_once(&free)[0].design.sequence.clone();
    let boundary = utr5.len();
    let straddler = sequence[boundary - 2..boundary + 2].to_string();

    let mut constrained = free.clone();
    constrained.extend(words(&["--motif", &straddler]));
    let args = design::parse(constrained).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let refused = design::solve_constrained(&m, &args.lambdas, 1, &args.sequence, None);
    let message = refused.expect_err("the space is empty").to_string();
    assert!(message.contains("no encoding"), "{message}");
    assert!(message.contains(&straddler), "{message}");
}

#[test]
fn a_pin_is_spelled_by_the_design() {
    // Residue 3 of this protein is glycine, whose codons are `GGx`, so
    // position 9 — its third nucleotide — is one a pin can choose. Position 4
    // is not: residue 2 is tyrosine (`UAU`/`UAC`), and a pin there naming
    // anything but U has no codon to keep.
    let unpinned = design_once(&words(&["--protein", &fasta(), "--lambda", "0.5"]))[0]
        .design
        .sequence
        .clone();
    let pinned = design_once(&words(&[
        "--protein",
        &fasta(),
        "--lambda",
        "0.5",
        "--pin",
        "9:A",
    ]));
    let got = pinned[0].design.sequence.clone();
    assert_eq!(
        &got[8..9],
        "A",
        "the pin at coding position 9 was not spelled"
    );
    // The pin narrows the feasible set, so its optimum cannot be better than
    // the unpinned one at the same weight. Asserting that other positions do
    // not move would be wrong: a design is a global optimum, and fixing one
    // nucleotide can re-choose any codon.
    let objective = |s: &Solved| 0.5 * s.design.mfe_kcal_mol - 0.5 * s.design.cai_log;
    let free = design_once(&words(&["--protein", &fasta(), "--lambda", "0.5"]));
    assert!(
        objective(&pinned[0]) >= objective(&free[0]) - 1e-9,
        "the pinned optimum beat the unpinned one"
    );
    assert_eq!(unpinned.len(), got.len());
}

#[test]
fn a_pin_no_codon_can_spell_is_refused_before_the_solve() {
    // The start codon is AUG; nothing else spells methionine, so pinning its
    // first position to C leaves the residue with no codon.
    let args = design::parse(words(&[
        "--protein",
        &fasta(),
        "--lambda",
        "0.5",
        "--pin",
        "1:C",
    ]))
    .unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let refused = design::solve_constrained(&m, &args.lambdas, 1, &args.sequence, None);
    let message = refused
        .expect_err("a pin no codon can spell is refused")
        .to_string();
    // The refusal names the flag, the residue and the coordinate: a message
    // saying only that the space is empty would leave the caller to work out
    // which pin emptied it.
    assert!(message.contains("--pin"), "{message}");
    assert!(message.contains("residue 1 (M)"), "{message}");
    assert!(message.contains("cds:1"), "{message}");
}

#[test]
fn an_infeasible_motif_set_names_the_motifs() {
    // Every codon of every residue contains at least one of these, so no
    // encoding survives.
    let args = design::parse(words(&[
        "--protein",
        "MKV",
        "--lambda",
        "0.5",
        "--motif",
        "A",
        "--motif",
        "C",
        "--motif",
        "G",
        "--motif",
        "U",
    ]))
    .unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let refused = design::solve_constrained(&m, &args.lambdas, 1, &args.sequence, None);
    let message = refused
        .expect_err("an empty design space is refused")
        .to_string();
    assert!(message.contains("no encoding"), "{message}");
    assert!(message.contains('A') && message.contains('U'), "{message}");
}

#[test]
fn the_header_records_the_sequence_constraints_the_run_was_under() {
    // A later command needs this: a design made under motifs or pins may be
    // certified but not searched from, and nothing else in the file says so.
    let (utr5, _) = hbb25();
    let inside_flank = utr5[5..11].to_string();
    let argv = words(&[
        "--protein",
        &fasta(),
        "--lambda",
        "0.5",
        "--stop",
        "none",
        "--utr5",
        &utr5,
        "--forbid",
        "none",
        "--motif",
        &inside_flank,
        "--motif-window",
        "4..-4",
        "--pin",
        "9:A",
    ]);
    let args = design::parse(argv).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let fields = args.sequence.header_fields(&m).unwrap();

    assert_eq!(fields["motifs"], serde_json::json!([inside_flank]));
    assert_eq!(fields["motif_window"], "coding positions 4..-4");
    assert_eq!(fields["pins"], serde_json::json!(["9:A"]));
    // The occurrence the run warned about is recorded, not only printed.
    let unremovable = fields["motifs_in_fixed_flanks"].as_array().unwrap();
    assert_eq!(unremovable.len(), 1, "{unremovable:?}");
    assert_eq!(unremovable[0]["side"], "5′ UTR");
    assert_eq!(unremovable[0]["motif"], inside_flank);

    // A run under no sequence constraint records none of these fields rather
    // than recording them empty, so an old reader sees the file it expects.
    let plain = design::parse(words(&["--protein", &fasta(), "--lambda", "0.5"])).unwrap();
    let plain_m = Molecule::resolve(&plain.shared).unwrap();
    assert!(plain.sequence.header_fields(&plain_m).unwrap().is_empty());
}

#[test]
fn the_front_takes_motifs_and_refuses_pins() {
    // `front` takes the motif surface and walks the motif-free front; a pin
    // needs the restricted lattice in the walk, so it is refused with that
    // reason rather than read as an unknown flag.
    let motif = derna_cli::front::parse(words(&["--protein", "MK", "--motif", "GGGG"])).unwrap();
    assert_eq!(motif.sequence.patterns(), ["GGGG"]);
    let window = derna_cli::front::parse(words(&[
        "--protein",
        "MK",
        "--motif",
        "GGGG",
        "--motif-window",
        "1..3",
    ]))
    .unwrap();
    assert_eq!(window.sequence.motif_window, Some((1, 3)));
    let message = derna_cli::front::parse(words(&["--protein", "MK", "--pin", "1:A"]))
        .expect_err("front refuses a pin")
        .to_string();
    assert!(message.contains("--pin is not supported yet"), "{message}");
}
