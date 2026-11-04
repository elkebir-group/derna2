//! Which format version each command's result declares, and what a reader
//! says about a version above its own.
//!
//! A reader refuses any version above its own, so a file declares version 2
//! only when it carries a field the rename introduced (`modular`,
//! `modular_front`). Every other file is version 1's content exactly and says
//! so, which keeps it readable by a derna built before the rename.

use std::path::{Path, PathBuf};

use serde_json::Value;

const PROTEIN: &str = "MKVLAKGIVLAGQWERS";
const UTR5: &str = "ACAUUUGCUU";
const UTR3: &str = "GGCCAAUUGG";

fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

/// A fresh directory for one test's files.
fn scratch_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "derna-format-version-{}-{test}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(&dir).unwrap() {
        std::fs::remove_file(entry.unwrap().path()).unwrap();
    }
    dir
}

fn lines(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// Run `derna <command>` on the cell into `out`, plus `extra`, and return the
/// file's lines.
fn run(command: &str, out: &Path, extra: &[&str]) -> Vec<Value> {
    let mut argv = words(&[
        command,
        "--protein",
        PROTEIN,
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--out",
    ]);
    argv.push(out.to_str().unwrap().to_string());
    argv.extend(words(extra));
    derna_cli::run(argv).unwrap();
    lines(out)
}

/// Whether any line holds a field under a name the rename introduced.
fn carries_renamed(v: &Value) -> bool {
    match v {
        Value::Object(m) => m
            .iter()
            .any(|(k, v)| k == "modular" || k == "modular_front" || carries_renamed(v)),
        Value::Array(a) => a.iter().any(carries_renamed),
        _ => false,
    }
}

/// `file` declares `version`, and carries a renamed field exactly when that
/// version is 2.
fn assert_version(file: &[Value], version: u64, what: &str) {
    assert_eq!(file[0]["format_version"], version, "{what}: {}", file[0]);
    assert_eq!(
        file.iter().any(carries_renamed),
        version == 2,
        "{what}: the version does not match the content"
    );
}

#[test]
fn files_without_a_renamed_field_are_version_1() {
    let dir = scratch_dir("one");
    // A constrained front certifies its points by default, and each verdict
    // carries `modular`; declining the check leaves the walk's own fields.
    let plain = run(
        "front",
        &dir.join("plain.jsonl"),
        &["--threads", "2", "--no-check"],
    );
    assert_version(&plain, 1, "an uncertified plain front");
    let motif = run(
        "front",
        &dir.join("motif.jsonl"),
        &["--threads", "2", "--motif", "GAAUUC", "--no-check"],
    );
    assert_version(&motif, 1, "an uncertified motif front");
    // With no constraint a design is not checked, so it carries no verdict.
    let design = run(
        "design",
        &dir.join("design.jsonl"),
        &["--lambda", "0.2,0.8", "--forbid", "none"],
    );
    assert_version(&design, 1, "an unchecked design");
}

#[test]
fn files_with_a_renamed_field_are_version_2() {
    let dir = scratch_dir("two");
    // A design under a constraint carries each point's check, `modular`
    // among it.
    let design = run(
        "design",
        &dir.join("design.jsonl"),
        &["--lambda", "0.2,0.8"],
    );
    assert_version(&design, 2, "a checked design");

    // A constrained front certifies its own points, so each carries `modular`.
    let plain = dir.join("plain.jsonl");
    let certified = run("front", &plain, &["--threads", "2"]);
    assert_version(&certified, 2, "a self-certified front");
    let out = dir.join("checked.jsonl");
    derna_cli::run(words(&[
        "check",
        plain.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]))
    .unwrap();
    assert_version(&lines(&out), 2, "check output");

    let out = dir.join("modular.jsonl");
    derna_cli::run(words(&[
        "front",
        "--protein",
        "MPRGLA",
        "--utr5",
        "CCGCGG",
        "--stop",
        "none",
        "--modular",
        "--out",
        out.to_str().unwrap(),
    ]))
    .unwrap();
    assert_version(&lines(&out), 2, "a modular front");
}

#[test]
fn a_later_version_is_refused_by_name_as_newer_not_damaged() {
    let dir = scratch_dir("three");
    let plain = dir.join("plain.jsonl");
    let mut file = run("front", &plain, &["--threads", "2"]);
    file[0]["format_version"] = 3.into();
    let later = dir.join("later.jsonl");
    let text: String = file.iter().map(|l| format!("{l}\n")).collect();
    std::fs::write(&later, text).unwrap();
    let said = |e: String| {
        assert!(e.contains(later.to_str().unwrap()), "{e}");
        assert!(e.contains("format version 3"), "{e}");
        assert!(e.contains("reads up to version 2"), "{e}");
        assert!(e.contains("intact"), "{e}");
        assert!(e.contains("newer derna"), "{e}");
    };
    said(match derna_cli::show::read(Some(&later)) {
        Ok(_) => panic!("a later version was read"),
        Err(e) => e.to_string(),
    });
    // The front readers go through the same gate: `--front-from` here.
    said(
        derna_cli::run(words(&[
            "front",
            "--protein",
            PROTEIN,
            "--utr5",
            UTR5,
            "--utr3",
            UTR3,
            "--modular",
            "--front-from",
            later.to_str().unwrap(),
            "--out",
            dir.join("again.jsonl").to_str().unwrap(),
        ]))
        .unwrap_err()
        .to_string(),
    );
}
