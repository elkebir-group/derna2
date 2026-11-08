//! The codon usage table a run scored CAI against, and the header that
//! records it.
//!
//! A few organisms are bundled under names and anything else is a path; the
//! bundled tables themselves are checked against their sources in
//! `codon_usage_tables.rs`. A result file embeds the whole table rather than
//! its name, so a later command — `check` scoring a replacement design, or a
//! reader comparing two files — scores CAI against exactly the table the run
//! used, not against whatever that name means in the binary that reads it.

use derna_cli::design;
use derna_cli::molecule::Molecule;
use derna_cli::result;
use serde_json::Value;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

/// The vendored human table, as text.
fn human_csv() -> String {
    std::fs::read_to_string(root().join("packages/derna-codon-dp/data/codon_usage_human.csv"))
        .unwrap()
}

fn solve(argv: Vec<String>) -> (Molecule, Vec<design::Solved>) {
    let args = design::parse(argv).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let solved = design::solve(&m, &args.lambdas, 1).unwrap();
    (m, solved)
}

#[test]
fn a_table_given_by_path_is_the_one_scored_against() {
    // A table that inverts the human fractions for leucine: the codon the
    // human table prefers becomes the rarest and vice versa, so a
    // CAI-dominated design must spell leucine differently.
    let mut rows: Vec<String> = Vec::new();
    let mut leucine: Vec<(String, f64)> = Vec::new();
    for line in human_csv().lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields[0] == "L" {
            leucine.push((fields[1].to_string(), fields[2].parse().unwrap()));
        }
    }
    let inverted: Vec<f64> = leucine.iter().rev().map(|(_, f)| *f).collect();
    rows.push("amino_acid,codon,fraction".to_string());
    for line in human_csv().lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields[0] == "L" {
            let i = leucine.iter().position(|(c, _)| c == fields[1]).unwrap();
            rows.push(format!("L,{},{}", fields[1], inverted[i]));
        } else {
            rows.push(line.trim().to_string());
        }
    }
    let path = std::env::temp_dir().join(format!("derna-usage-{}.csv", std::process::id()));
    std::fs::write(&path, rows.join("\n") + "\n").unwrap();

    // A protein of leucines alone, at a weight that is nearly all CAI.
    let protein = "MLLLLLLLL";
    let (_, with_human) = solve(words(&["--protein", protein, "--lambda", "0.001"]));
    let (m, with_file) = solve(words(&[
        "--protein",
        protein,
        "--lambda",
        "0.001",
        "--codon-usage",
        path.to_str().unwrap(),
    ]));
    std::fs::remove_file(&path).unwrap();

    let human_leucine = &with_human[0].design.sequence[3..6];
    let file_leucine = &with_file[0].design.sequence[3..6];
    assert_ne!(
        human_leucine, file_leucine,
        "the table given by path changed nothing, so it was not the one scored against"
    );
    // The header records the table it used, by source and by content.
    let header = result::header("design", &m, serde_json::Map::new());
    assert_eq!(
        header["codon_usage"]["source"].as_str().unwrap(),
        path.to_str().unwrap()
    );
    let table = header["codon_usage"]["table"].as_array().unwrap();
    assert_eq!(table.len(), 61, "every sense codon is recorded");
    let recorded: Vec<&Value> = table.iter().filter(|row| row[0] == "L").collect();
    assert_eq!(recorded.len(), 6);
    assert_eq!(
        recorded[0][2].as_f64().unwrap(),
        inverted[0],
        "the header's table is not the file's"
    );
}

#[test]
fn the_built_in_table_is_named_rather_than_pathed() {
    let (m, _) = solve(words(&["--protein", "MK", "--lambda", "0.5"]));
    let header = result::header("design", &m, serde_json::Map::new());
    assert_eq!(header["codon_usage"]["source"], "human");
    assert_eq!(header["codon_usage"]["table"].as_array().unwrap().len(), 61);
}

#[test]
fn an_unreadable_or_malformed_table_is_refused_naming_the_flag() {
    let missing = design::parse(words(&[
        "--protein",
        "MK",
        "--lambda",
        "0.5",
        "--codon-usage",
        "/nonexistent/table.csv",
    ]))
    .unwrap();
    let message = Molecule::resolve(&missing.shared)
        .expect_err("a missing table is refused")
        .to_string();
    assert!(message.contains("--codon-usage"), "{message}");

    let path = std::env::temp_dir().join(format!("derna-bad-usage-{}.csv", std::process::id()));
    std::fs::write(&path, "amino_acid,codon,fraction\nL,CUU,not-a-number\n").unwrap();
    let malformed = design::parse(words(&[
        "--protein",
        "MK",
        "--lambda",
        "0.5",
        "--codon-usage",
        path.to_str().unwrap(),
    ]))
    .unwrap();
    let message = Molecule::resolve(&malformed.shared)
        .expect_err("a malformed table is refused")
        .to_string();
    std::fs::remove_file(&path).unwrap();
    assert!(message.contains("--codon-usage"), "{message}");
}
