//! `derna design` reproduces the standalone per-weight design binary it was
//! built over field for field, on every constraint spelling the fixtures
//! record, with the constraint written out on both sides.
//!
//! The fixtures are frozen output of that binary, which is not part of this
//! distribution.

use std::path::{Path, PathBuf};

use derna_cli::design::{self, Solved};
use derna_cli::molecule::Molecule;
use derna_cli::result;
use serde_json::Value;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn read_jsonl(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn derna_words(manifest: &Value, case: &Value) -> Vec<String> {
    // Each case names its own protein: the equality runs over two cells — a
    // second protein with a second flank pair — so that it is a property of
    // the command rather than of one molecule.
    let fasta = repo_root().join(case["fasta"].as_str().unwrap());
    let mut words = vec!["--protein".to_string(), fasta.to_str().unwrap().to_string()];
    let lambdas: Vec<String> = manifest["lambdas"]
        .as_array()
        .unwrap()
        .iter()
        .map(Value::to_string)
        .collect();
    words.extend(["--lambda".to_string(), lambdas.join(",")]);
    words.extend(
        case["derna"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_str().unwrap().to_string()),
    );
    words
}

fn solve(words: Vec<String>) -> (Molecule, Vec<Solved>) {
    let args = design::parse(words).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let solved = design::solve(&m, &args.lambdas, 1).unwrap();
    (m, solved)
}

fn assert_same_design(case: &str, expected: &Value, got: &Value) {
    let pairs = [
        ("rna_seq", "sequence"),
        ("structure", "structure"),
        ("mfe_kcal_mol", "mfe"),
        ("cai", "cai"),
        ("cai_log", "cai_log"),
        ("lambda", "lambda"),
    ];
    for (theirs, ours) in pairs {
        assert_eq!(expected[theirs], got[ours], "{case}: {ours}");
    }
}

fn assert_same_check(case: &str, expected: &Value, got: &Value) {
    let f = &expected["free_fold"];
    assert_eq!(f["free_mfe_kcal_mol"], got["free_mfe"], "{case}: free_mfe");
    assert_eq!(f["gap_kcal_mol"], got["gap"], "{case}: gap");
    assert_eq!(
        f["free_structure"], got["free_structure"],
        "{case}: free_structure"
    );
    assert_eq!(f["honest"], got["modular"], "{case}: modular");
    let shifted: Vec<Vec<u64>> = f["reinstated_pairs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            p.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() + 1)
                .collect()
        })
        .collect();
    assert_eq!(
        serde_json::to_value(shifted).unwrap(),
        got["violations"],
        "{case}: violations"
    );
}

#[test]
fn design_matches_every_recorded_case() {
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(data("design-cases.json")).unwrap()).unwrap();
    for case in manifest["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let expected = read_jsonl(&data(&format!("design-{name}.jsonl")));
        let (m, solved) = solve(derna_words(&manifest, case));
        assert_eq!(expected.len(), solved.len(), "{name}: one record per λ");
        for (e, s) in expected.iter().zip(&solved) {
            let got = result::checked_point(&s.design, Some(s.lambda), true, s.checked.as_ref());
            assert_same_design(name, e, &got);
            assert_eq!(
                m.constraint.is_inert(),
                got.get("gap").is_none(),
                "{name}: check present"
            );
            if !m.constraint.is_inert() {
                assert_same_check(name, e, &got);
            }
        }
    }
}

#[test]
fn every_cell_has_a_design_that_violates_the_constraint() {
    // Non-vacuity, per cell: the check comparison above would pass on
    // all-clean rows, so each cell must contain a design whose free fold
    // breaks the constraint. Asserted per cell rather than over the whole
    // corpus, because one cell carrying every failing design would leave the
    // other's check untested.
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(data("design-cases.json")).unwrap()).unwrap();
    for cell in manifest["cells"].as_array().unwrap() {
        let cell = cell.as_str().unwrap();
        let failing = manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|case| {
                case["name"]
                    .as_str()
                    .unwrap()
                    .starts_with(&format!("{cell}-"))
            })
            .flat_map(|case| {
                read_jsonl(&data(&format!(
                    "design-{}.jsonl",
                    case["name"].as_str().unwrap()
                )))
            })
            .filter(|record| record["free_fold"]["honest"] == Value::Bool(false))
            .count();
        assert!(
            failing > 0,
            "no design in cell {cell} violates its constraint, so that cell's check \
             comparison proves nothing"
        );
    }
}
