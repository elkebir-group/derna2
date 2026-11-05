//! `derna check --sequences` scores designs this run did not make.
//!
//! The strong property is the same equality the rest of `check` is held to, pointed the
//! other way: a design `derna` made, handed back as a bare sequence with the molecule on
//! the command line, must be scored exactly as the run that made it scored it — the same
//! constrained energy, the same CAI, the same free fold and the same verdict. If the two
//! disagree, then what the command line says the molecule is and what the design was made
//! over are not the same thing, which is the one failure this route can have that a
//! result file's header rules out by construction.
//!
//! The refusals beside it are that construction's replacement: a sequence is scored only
//! where it is a molecule of the description given.

use derna_cli::free_check::Checked;
use derna_cli::molecule::Molecule;
use derna_cli::{check, design, given};
use derna_lib::Params;
use serde_json::Value;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

fn flanks(pair: &str) -> (String, String) {
    let text =
        std::fs::read_to_string(root().join("packages/derna/tests/data/flank-pairs.json")).unwrap();
    let v: Value = serde_json::from_str(&text).unwrap();
    (
        v[pair]["utr5"].as_str().unwrap().to_string(),
        v[pair]["utr3"].as_str().unwrap().to_string(),
    )
}

const PROTEIN: &str = "packages/derna-codon-dp/tests/data/GLPE_HUMAN_20.fasta";
const PAIR: &str = "hbb25";

/// The molecule these tests are about, and `derna`'s own designs of it.
fn designed() -> (Molecule, Vec<derna_cli::design::Solved>) {
    let (utr5, utr3) = flanks(PAIR);
    let fasta = root().join(PROTEIN);
    let argv = words(&[
        "--protein",
        fasta.to_str().unwrap(),
        "--lambda",
        "0.1,0.6",
        "--utr5",
        &utr5,
        "--utr3",
        &utr3,
    ]);
    let args = design::parse(argv).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let solved = design::solve(&m, &args.lambdas, 1).unwrap();
    (m, solved)
}

/// Write `sequences` where `given::points` can read them, FASTA or bare.
fn sequence_file(dir: &Path, name: &str, sequences: &[String], fasta: bool) -> PathBuf {
    let path = dir.join(name);
    let mut text = String::new();
    for (i, s) in sequences.iter().enumerate() {
        if fasta {
            text.push_str(&format!(">d{i}\n"));
        }
        text.push_str(s);
        text.push('\n');
    }
    std::fs::write(&path, text).unwrap();
    path
}

fn tmpdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("derna-given-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn a_design_handed_back_as_a_sequence_is_scored_as_it_was_designed() {
    let (m, solved) = designed();
    let dir = tmpdir("roundtrip");
    let sequences: Vec<String> = solved.iter().map(|s| s.design.sequence.clone()).collect();
    let path = sequence_file(&dir, "designs.fa", &sequences, true);

    let points = given::points(&Params::load_default().unwrap(), &m, &path).unwrap();
    assert_eq!(points.len(), solved.len());

    for (point, s) in points.iter().zip(&solved) {
        assert_eq!(point["sequence"].as_str().unwrap(), s.design.sequence);
        // The constrained fold of the sequence is the one the design search reported:
        // the same molecule, the same constraint, the same energy model.
        assert_eq!(
            point["mfe"].as_f64().unwrap(),
            s.design.mfe_kcal_mol,
            "the constrained energy of a design it did not compute itself"
        );
        assert_eq!(point["cai"].as_f64().unwrap(), s.design.cai);
        assert!(
            (point["cai_log"].as_f64().unwrap() - s.design.cai_log).abs() < 1e-9,
            "log-CAI, which the objective is written over"
        );
    }

    // And the verdict that follows from them is the producing run's.
    let certified = check::certify(&m, &points, None).unwrap();
    for ((_, checked), s) in certified.iter().zip(&solved) {
        let Checked::Exact(gap) = checked else {
            panic!("no beam was asked for")
        };
        let made = s.checked.as_ref().expect("a constrained design is checked");
        let Checked::Exact(theirs) = made else {
            panic!("no beam was asked for")
        };
        assert_eq!(gap.free_mfe_kcal_mol, theirs.free_mfe_kcal_mol);
        assert_eq!(gap.gap_kcal_mol, theirs.gap_kcal_mol);
    }
}

#[test]
fn the_same_sequences_read_bare_or_as_fasta() {
    let (m, solved) = designed();
    let dir = tmpdir("shapes");
    let sequences: Vec<String> = solved.iter().map(|s| s.design.sequence.clone()).collect();
    let params = Params::load_default().unwrap();

    let as_fasta = sequence_file(&dir, "a.fa", &sequences, true);
    let as_lines = sequence_file(&dir, "b.txt", &sequences, false);
    let one = given::points(&params, &m, &as_fasta).unwrap();
    let two = given::points(&params, &m, &as_lines).unwrap();
    assert_eq!(one.len(), two.len());
    for (a, b) in one.iter().zip(&two) {
        assert_eq!(a["sequence"], b["sequence"]);
        assert_eq!(a["mfe"], b["mfe"]);
    }
    // The FASTA header is kept, because a refusal names the sequence it is about.
    assert_eq!(one[0]["name"].as_str().unwrap(), "d0");
    assert!(two[0].get("name").is_none());
}

#[test]
fn a_dna_spelling_is_read_as_rna() {
    let (m, solved) = designed();
    let dir = tmpdir("dna");
    let rna = solved[0].design.sequence.clone();
    let dna = rna.replace('U', "T");
    let params = Params::load_default().unwrap();
    let path = sequence_file(&dir, "dna.txt", &[dna], false);
    let points = given::points(&params, &m, &path).unwrap();
    assert_eq!(points[0]["sequence"].as_str().unwrap(), rna);
}

/// Every refusal below is a sequence that is *not* a molecule of the description given,
/// which is what this route has in place of a header that cannot disagree with itself.
#[test]
fn a_sequence_of_the_wrong_length_is_refused() {
    let (m, solved) = designed();
    let dir = tmpdir("length");
    let short = solved[0].design.sequence[..solved[0].design.sequence.len() - 3].to_string();
    let path = sequence_file(&dir, "short.txt", &[short], false);
    let err = given::points(&Params::load_default().unwrap(), &m, &path)
        .unwrap_err()
        .to_string();
    assert!(err.contains("nucleotides"), "{err}");
    assert!(
        err.contains("coding"),
        "the refusal says what the parts are: {err}"
    );
}

#[test]
fn a_sequence_without_the_given_flank_is_refused() {
    let (m, solved) = designed();
    let dir = tmpdir("flank");
    let mut wrong = solved[0].design.sequence.clone();
    // One nucleotide of the 5′ flank changed: the length still fits, the flank does not.
    let first = if wrong.starts_with('A') { 'C' } else { 'A' };
    wrong.replace_range(0..1, &first.to_string());
    let path = sequence_file(&dir, "flank.txt", &[wrong], false);
    let err = given::points(&Params::load_default().unwrap(), &m, &path)
        .unwrap_err()
        .to_string();
    assert!(err.contains("5′ UTR"), "{err}");
}

#[test]
fn a_sequence_that_codes_for_another_protein_is_refused() {
    let (m, solved) = designed();
    let dir = tmpdir("translation");
    let mut wrong = solved[0].design.sequence.clone();
    // The first codon of the coding window, made into one that cannot code for the
    // residue the molecule names: every residue's codons share their first two bases
    // only within a family, so a whole codon is replaced.
    let start = m.cds().start;
    wrong.replace_range(start..start + 3, "AAA");
    let path = sequence_file(&dir, "wrong.txt", &[wrong], false);
    let err = given::points(&Params::load_default().unwrap(), &m, &path)
        .unwrap_err()
        .to_string();
    assert!(err.contains("does not code for"), "{err}");
    assert!(
        err.contains("codon 1"),
        "the refusal names which codon: {err}"
    );
}

#[test]
fn a_file_with_no_sequence_is_refused() {
    let (m, _) = designed();
    let dir = tmpdir("empty");
    let path = sequence_file(&dir, "empty.txt", &[], false);
    let err = given::points(&Params::load_default().unwrap(), &m, &path)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no sequence"), "{err}");
}

#[test]
fn the_molecule_flags_are_taken_with_sequences_and_refused_without() {
    let fasta = root().join(PROTEIN);
    let fasta = fasta.to_str().unwrap();
    // Without --sequences the header carries the molecule, so the flag is refused.
    let err = check::parse(words(&["front.jsonl", "--protein", fasta]))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("cannot be pointed at a different molecule"),
        "{err}"
    );

    // With it, the same flag is how the command learns what the sequences are.
    let args = check::parse(words(&["--designs", "s.txt", "--protein", fasta])).unwrap();
    assert_eq!(args.designs.unwrap().to_str().unwrap(), "s.txt");
    assert!(args.input.is_none());
}

#[test]
fn sequences_without_a_protein_is_refused() {
    let err = check::parse(words(&["--designs", "s.txt"]))
        .unwrap_err()
        .to_string();
    assert!(err.contains("--protein"), "{err}");
}

#[test]
fn a_result_file_and_sequences_together_are_refused() {
    let fasta = root().join(PROTEIN);
    let err = check::parse(words(&[
        "front.jsonl",
        "--designs",
        "s.txt",
        "--protein",
        fasta.to_str().unwrap(),
    ]))
    .unwrap_err()
    .to_string();
    assert!(err.contains("no result file to check"), "{err}");
}
