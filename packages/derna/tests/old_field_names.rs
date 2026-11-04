//! Files written before `structurally_optimal` became `modular` (and
//! `structurally_optimal_front` became `modular_front`) still read: every
//! reader of a result or progress file takes the old name as the new one.
//!
//! Each test plants a value that only the old field carries — a verdict the
//! computation did not reach, or a header field that changes which files are
//! accepted — so a reader that ignored the old name would fail rather than
//! pass on the value the file happens to share with a fresh run.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

/// A fresh directory for one test's files.
fn scratch_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "derna-old-field-names-{}-{test}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(&dir).unwrap() {
        std::fs::remove_file(entry.unwrap().path()).unwrap();
    }
    dir
}

fn read_lines(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn write_lines(path: &Path, lines: &[Value]) {
    let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
    std::fs::write(path, text).unwrap();
}

/// `line` with `new` moved to `old`, as a file of the earlier format has it.
fn spelled_old(mut line: Value, new: &str, old: &str) -> Value {
    let map = line.as_object_mut().unwrap();
    if let Some(v) = map.remove(new) {
        map.insert(old.into(), v);
    }
    line
}

/// A checked design result over three weights.
fn designs(out: &Path) -> Vec<Value> {
    derna_cli::run(words(&[
        "design",
        "--protein",
        "MKVLAKGIVLAGQWERS",
        "--utr5",
        "ACAUUUGCUU",
        "--utr3",
        "GGCCAAUUGG",
        "--lambda",
        "0.2,0.5,0.8",
        "--out",
        out.to_str().unwrap(),
    ]))
    .unwrap();
    read_lines(out)
}

/// Run `derna front --modular` on a small cell into `out`, plus `extra`.
fn modular(out: &Path, extra: &[&str]) -> derna_cli::Result<()> {
    let mut argv = words(&[
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
    ]);
    argv.extend(words(extra));
    derna_cli::run(argv)
}

#[test]
fn a_result_file_reads_its_old_verdict_field_as_modular() {
    let dir = scratch_dir("result");
    let fresh = designs(&dir.join("fresh.jsonl"));
    assert_eq!(fresh[0]["format_version"], 2);
    // Every verdict flipped, and spelled the old way in a version-1 file.
    let mut old = vec![{
        let mut h = fresh[0].clone();
        h["format_version"] = json!(1);
        h
    }];
    let mut planted = Vec::new();
    for p in &fresh[1..] {
        let flipped = !p["modular"]
            .as_bool()
            .expect("a design row carries its verdict");
        planted.push(flipped);
        let mut p = p.clone();
        p["modular"] = json!(flipped);
        old.push(spelled_old(p, "modular", "structurally_optimal"));
    }
    let path = dir.join("old.jsonl");
    write_lines(&path, &old);

    let read = derna_cli::show::read(Some(&path)).unwrap();
    let verdicts: Vec<bool> = read
        .points
        .iter()
        .map(|p| p["modular"].as_bool().unwrap())
        .collect();
    assert_eq!(verdicts, planted);
    assert!(read
        .points
        .iter()
        .all(|p| p.get("structurally_optimal").is_none()));

    // `show --modular` selects by the old field too: it keeps exactly the
    // points the planted verdicts pass.
    let args = derna_cli::show::parse(words(&[path.to_str().unwrap(), "--modular"])).unwrap();
    let all = derna_cli::show::parse(words(&[path.to_str().unwrap()])).unwrap();
    let kept = derna_cli::show::render(&args, &read).len();
    let every = derna_cli::show::render(&all, &read).len();
    let passing = planted.iter().filter(|&&v| v).count();
    let failing = planted.len() - passing;
    assert_eq!(every - kept, failing, "planted {planted:?}");
}

#[test]
fn a_check_progress_line_reads_its_old_verdict_field_as_modular() {
    let dir = scratch_dir("check-progress");
    let fresh = designs(&dir.join("fresh.jsonl"));
    let point = &fresh[1];
    let flipped = !point["modular"].as_bool().unwrap();
    let mut line = point.clone();
    line["modular"] = json!(flipped);
    let path = dir.join("old.check.jsonl");
    write_lines(
        &path,
        &[spelled_old(line, "modular", "structurally_optimal")],
    );
    let sequence = point["sequence"].as_str().unwrap();
    let of: HashMap<&str, usize> = [(sequence, 0)].into();
    let done = derna_cli::check::read_progress(&path, &of).unwrap();
    assert_eq!(done[sequence]["modular"], json!(flipped));
    assert!(done[sequence].get("structurally_optimal").is_none());
}

#[test]
fn a_search_log_reads_its_old_verdict_field_as_modular() {
    let dir = scratch_dir("search-log");
    let out = dir.join("so.jsonl");
    modular(&out, &["--checkpoint"]).unwrap();
    let mut log = read_lines(&dir.join("so.jsonl.search.jsonl"));
    let at = log
        .iter()
        .position(|l| l.get("checked").is_some())
        .expect("a search that checked a point");
    let sequence = log[at]["checked"].as_str().unwrap().to_string();
    let flipped = !log[at]["modular"].as_bool().unwrap();
    log[at]["modular"] = json!(flipped);
    let old: Vec<Value> = log
        .into_iter()
        .map(|l| spelled_old(l, "modular", "structurally_optimal"))
        .collect();
    let path = dir.join("old.search.jsonl");
    write_lines(&path, &old);
    let progress = derna_cli::modular_front::read_progress(&path).unwrap();
    assert_eq!(progress.checked[&sequence], flipped);
}

#[test]
fn front_from_reads_the_old_header_field_and_refuses_a_modular_front() {
    let dir = scratch_dir("front-from");
    let out = dir.join("so.jsonl");
    modular(&out, &[]).unwrap();
    let mut lines = read_lines(&out);
    assert_eq!(lines[0]["modular_front"], true);
    // A version-1 modular front: only the old header field says what it is.
    lines[0] = spelled_old(
        lines[0].clone(),
        "modular_front",
        "structurally_optimal_front",
    );
    lines[0]["format_version"] = json!(1);
    let old = dir.join("old.jsonl");
    write_lines(&old, &lines);
    let e = modular(
        &dir.join("again.jsonl"),
        &["--front-from", old.to_str().unwrap()],
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("takes a plain `derna front` result"), "{e}");
}
