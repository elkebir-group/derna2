//! From a front to a sequence to order: `pick`, the FASTA forms, the sequence
//! metrics every point carries, the table a run ends with, `show --plot`, and
//! a multi-record `--protein`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use derna_cli::metrics;
use serde_json::Value;

const PROTEIN: &str = "MKVLAKGIVLAGQWERS";
const UTR5: &str = "ACAUUUGCUU";
const UTR3: &str = "GGCCAAUUGG";

fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

/// A fresh directory for one test's files.
fn scratch_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("derna-practical-{}-{test}", std::process::id()));
    if dir.exists() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_file() {
                std::fs::remove_file(path).unwrap();
            }
        }
    }
    std::fs::create_dir_all(&dir).unwrap();
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

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn derna(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_derna"))
        .args(args)
        .output()
        .expect("the binary runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).unwrap()
}

/// A certified front of the small cell, written to `out` with `extra`.
fn front(out: &Path, extra: &[&str]) {
    let mut argv = words(&[
        "front",
        "--protein",
        PROTEIN,
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--threads",
        "2",
        "--out",
    ]);
    argv.push(s(out).to_string());
    argv.extend(words(extra));
    derna_cli::run(argv).unwrap();
}

/// Every point of `file` carries the metrics its own sequence gives.
fn assert_metrics(file: &[Value], what: &str) {
    let cds = metrics::coding_window(&file[0]);
    let points: Vec<&Value> = file.iter().filter(|l| l["record"] == "point").collect();
    assert!(!points.is_empty(), "{what}: no point");
    for p in points {
        let want = metrics::of(p["sequence"].as_str().unwrap(), cds.clone()).fields();
        for field in metrics::FIELDS {
            assert_eq!(p[field], want[field], "{what}: {field} of {p}");
        }
    }
}

#[test]
fn every_command_writes_the_metrics_on_every_point() {
    let dir = scratch_dir("metrics");
    let f = dir.join("front.jsonl");
    front(&f, &[]);
    let file = lines(&f);
    assert_metrics(&file, "front");
    // The designed stop is in the coding window: 3 nt per residue and 3 more.
    let cds = metrics::coding_window(&file[0]);
    assert_eq!(cds, UTR5.len()..UTR5.len() + 3 * PROTEIN.len() + 3);

    let d = dir.join("design.jsonl");
    derna_cli::run(words(&[
        "design",
        "--protein",
        PROTEIN,
        "--utr5",
        UTR5,
        "--lambda",
        "0.3,0.7",
        "--out",
        s(&d),
    ]))
    .unwrap();
    assert_metrics(&lines(&d), "design");

    let unchecked = dir.join("unchecked.jsonl");
    front(&unchecked, &["--no-check", "--stop", "UAA"]);
    let c = dir.join("checked.jsonl");
    derna_cli::run(words(&["check", s(&unchecked), "--out", s(&c)])).unwrap();
    assert_metrics(&lines(&c), "check");

    let m = dir.join("modular.jsonl");
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
        s(&m),
    ]))
    .unwrap();
    assert_metrics(&lines(&m), "modular front");

    let pooled = dir.join("joined.jsonl");
    derna_cli::run(words(&["pool", s(&c), "--out", s(&pooled)])).unwrap();
    assert_metrics(&lines(&dir.join("joined.union.jsonl")), "pool");
}

#[test]
fn a_file_written_without_metrics_is_still_read_and_picked() {
    let dir = scratch_dir("old");
    let f = dir.join("front.jsonl");
    front(&f, &[]);
    let fresh = lines(&f);
    let mut old = fresh.clone();
    for p in old.iter_mut().skip(1) {
        for field in metrics::FIELDS {
            p.as_object_mut().unwrap().remove(field);
        }
    }
    let path = dir.join("old.jsonl");
    std::fs::write(
        &path,
        old.iter().map(|l| format!("{l}\n")).collect::<String>(),
    )
    .unwrap();
    // Derived from the sequence, the table matches the file that carries them.
    let render = |p: &Path, w: &[&str]| {
        let mut argv = vec![s(p)];
        argv.extend(w);
        let args = derna_cli::show::parse(words(&argv)).unwrap();
        derna_cli::show::render(&args, &derna_cli::show::read(Some(p)).unwrap())
    };
    assert_eq!(
        render(&path, &["--tsv", "--sequences"]),
        render(&f, &["--tsv", "--sequences"])
    );
    let pick = |p: &Path| {
        let args = derna_cli::pick::parse(words(&[s(p), "--knee", "--json"])).unwrap();
        derna_cli::pick::picked(&args, &derna_cli::show::read(Some(p)).unwrap()).unwrap()
    };
    assert_eq!(pick(&path), pick(&f));
}

#[test]
fn pick_writes_one_record_by_its_rule() {
    let dir = scratch_dir("pick");
    let f = dir.join("front.jsonl");
    front(&f, &[]);
    let file = lines(&f);
    let points = &file[1..];
    let fasta = |w: &[&str]| -> Vec<String> {
        let args = derna_cli::pick::parse(words(w)).unwrap();
        derna_cli::pick::picked(&args, &derna_cli::show::read(Some(&f)).unwrap()).unwrap()
    };
    let best = fasta(&[s(&f), "--best-cai"]);
    assert_eq!(best.len(), 2);
    let top = points
        .iter()
        .map(|p| p["cai"].as_f64().unwrap())
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(best[0].contains(&format!("cai={top:.4}")), "{}", best[0]);
    assert!(best[0].starts_with(">derna point="), "{}", best[0]);
    assert!(best[0].contains(" gc="), "{}", best[0]);

    let low = fasta(&[s(&f), "--best-mfe", "--cds-only", "--dna"]);
    assert_eq!(low[1].len(), 3 * PROTEIN.len() + 3);
    assert!(
        low[1].starts_with("ATG") && !low[1].contains('U'),
        "{}",
        low[1]
    );

    let row = fasta(&[s(&f), "--knee", "--json"]);
    let row: Value = serde_json::from_str(&row[0]).unwrap();
    assert!(
        points.contains(&row),
        "the knee is one of the file's points"
    );

    // A filter that removes everything is named.
    let args = derna_cli::pick::parse(words(&[s(&f), "--best-cai", "--max-mfe", "-1000"])).unwrap();
    let e = derna_cli::pick::picked(&args, &derna_cli::show::read(Some(&f)).unwrap())
        .unwrap_err()
        .to_string();
    assert!(e.contains("--max-mfe removed every point"), "{e}");

    // --modular on a file nobody checked is refused, not answered with nothing.
    let unchecked = dir.join("unchecked.jsonl");
    front(&unchecked, &["--no-check"]);
    let args = derna_cli::pick::parse(words(&[s(&unchecked), "--knee", "--modular"])).unwrap();
    let e = derna_cli::pick::picked(&args, &derna_cli::show::read(Some(&unchecked)).unwrap())
        .unwrap_err()
        .to_string();
    assert!(e.contains("no point in this file carries one"), "{e}");

    // Through the binary: a FASTA record on stdout and a non-zero exit with a
    // message on a refusal.
    let out = derna(&["pick", s(&f), "--best-mfe"]);
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout).lines().count(), 2);
    let refused = derna(&["pick", s(&f), "--best-mfe", "--min-gc", "0.99"]);
    assert!(!refused.status.success());
    assert!(text(&refused.stderr).contains("--min-gc removed every point"));
}

/// A minimal XML well-formedness check: every tag closes in order, every
/// attribute value is quoted, and every `&` begins one of the five entities.
/// The SVG is written by hand, so this is what stands between it and a
/// viewer that refuses it.
fn assert_well_formed(xml: &str) {
    let mut stack: Vec<String> = Vec::new();
    let mut rest = xml;
    while let Some(open) = rest.find('<') {
        assert_entities(&rest[..open]);
        let close = rest[open..].find('>').expect("an unclosed tag") + open;
        let tag = &rest[open + 1..close];
        rest = &rest[close + 1..];
        if let Some(name) = tag.strip_prefix('/') {
            assert_eq!(
                stack.pop().as_deref(),
                Some(name.trim()),
                "mismatched </{name}>"
            );
            continue;
        }
        let self_closing = tag.ends_with('/');
        let body = tag.trim_end_matches('/');
        let name = body.split_whitespace().next().expect("a tag name");
        let mut attrs = body[name.len()..].trim();
        while !attrs.is_empty() {
            let eq = attrs.find('=').expect("an attribute without a value");
            let value = attrs[eq + 1..].trim_start();
            let quote = value.chars().next().expect("an empty attribute");
            assert!(
                quote == '"' || quote == '\'',
                "an unquoted attribute in <{tag}>"
            );
            let end = value[1..].find(quote).expect("an unterminated attribute") + 1;
            assert_entities(&value[1..end]);
            attrs = value[end + 1..].trim_start();
        }
        if !self_closing {
            stack.push(name.to_string());
        }
    }
    assert!(stack.is_empty(), "unclosed elements: {stack:?}");
}

fn assert_entities(text: &str) {
    assert!(!text.contains('<'));
    for (i, _) in text.match_indices('&') {
        let tail = &text[i..];
        assert!(
            ["&amp;", "&lt;", "&gt;", "&quot;", "&apos;"]
                .iter()
                .any(|e| tail.starts_with(e)),
            "a bare & in {text:?}"
        );
    }
}

#[test]
fn the_plot_is_xml_with_one_mark_per_selected_point() {
    let dir = scratch_dir("plot");
    let f = dir.join("front.jsonl");
    front(&f, &[]);
    let total = lines(&f).len() - 1;
    for (flags, kept) in [
        (vec![], total),
        (vec!["--modular"], {
            lines(&f)[1..]
                .iter()
                .filter(|p| p["modular"] == true)
                .count()
        }),
    ] {
        let svg = dir.join("front.svg");
        let mut argv = vec![s(&f), "--plot", s(&svg)];
        argv.extend(flags.iter());
        derna_cli::show::run(&derna_cli::show::parse(words(&argv)).unwrap()).unwrap();
        let xml = std::fs::read_to_string(&svg).unwrap();
        assert_well_formed(&xml);
        assert!(xml.starts_with("<svg") && xml.contains("MFE (kcal/mol)"));
        assert_eq!(xml.matches(r#"class="mark""#).count(), kept, "{flags:?}");
    }
    // The checker can fail: a dropped close tag is caught.
    let broken = std::panic::catch_unwind(|| assert_well_formed("<svg><g></svg>"));
    assert!(broken.is_err());
}

/// `--plot-title` replaces the record's name in the plot, and the default
/// still names the record. The title is escaped like any other text.
#[test]
fn the_plot_title_is_the_given_one_or_the_records_name() {
    let dir = scratch_dir("plot-title");
    let f = dir.join("front.jsonl");
    front(&f, &[]);

    let default_svg = dir.join("default.svg");
    derna_cli::show::run(
        &derna_cli::show::parse(words(&[s(&f), "--plot", s(&default_svg)])).unwrap(),
    )
    .unwrap();
    let default_xml = std::fs::read_to_string(&default_svg).unwrap();
    let header = &lines(&f)[0];
    let record = derna_cli::show::record_name(header);
    assert!(
        default_xml.contains(&format!(">{record}</text>")),
        "the default title names the record: {record:?}"
    );

    let titled_svg = dir.join("titled.svg");
    let title = "Pareto front <GLPE & UTRs>";
    derna_cli::show::run(
        &derna_cli::show::parse(words(&[
            s(&f),
            "--plot",
            s(&titled_svg),
            "--plot-title",
            title,
        ]))
        .unwrap(),
    )
    .unwrap();
    let titled_xml = std::fs::read_to_string(&titled_svg).unwrap();
    assert_well_formed(&titled_xml);
    assert!(
        titled_xml.contains("Pareto front &lt;GLPE &amp; UTRs&gt;"),
        "the given title is drawn, escaped: {titled_xml:?}"
    );
    assert!(
        !titled_xml.contains(&format!(">{record}</text>")),
        "the record's name is replaced, not added to"
    );
}

#[test]
fn a_run_ends_with_the_table_of_what_it_wrote_on_stderr() {
    let dir = scratch_dir("table");
    let f = dir.join("design.jsonl");
    let out = derna(&[
        "design",
        "--protein",
        "MKV",
        "--lambda",
        "0.5",
        "--out",
        s(&f),
    ]);
    assert!(out.status.success());
    let err = text(&out.stderr);
    assert!(err.contains(&format!("wrote {}:", f.display())), "{err}");
    assert!(
        err.contains("modular") && err.contains("1 point(s)"),
        "{err}"
    );

    // To stdout: the JSONL alone on stdout, the table still on stderr.
    let piped = derna(&["design", "--protein", "MKV", "--lambda", "0.5"]);
    let stdout = text(&piped.stdout);
    assert!(
        stdout
            .lines()
            .all(|l| serde_json::from_str::<Value>(l).is_ok()),
        "{stdout}"
    );
    assert!(text(&piped.stderr).contains("wrote stdout:"));

    let quiet = derna(&["design", "--protein", "MKV", "--lambda", "0.5", "--quiet"]);
    assert!(!text(&quiet.stderr).contains("wrote stdout:"));
}

#[test]
fn a_multi_record_protein_runs_each_record_into_a_directory() {
    let dir = scratch_dir("batch");
    let fasta = dir.join("proteins.fasta");
    // The second record holds a residue no codon spells, so it fails alone.
    std::fs::write(&fasta, ">one first\nMKVL\n>two|b\nMKXV\n>three\nMPRG\n").unwrap();
    let out = dir.join("results");
    let argv = |extra: &[&str]| {
        let mut a = vec![
            "design",
            "--protein",
            s(&fasta),
            "--lambda",
            "0.5",
            "--out",
            s(&out),
        ];
        a.extend(extra);
        a.iter().map(|w| w.to_string()).collect::<Vec<_>>()
    };
    let ran = derna(&argv(&[]).iter().map(String::as_str).collect::<Vec<_>>());
    assert!(!ran.status.success(), "a failed record fails the batch");
    let err = text(&ran.stderr);
    assert!(
        err.contains("batch of 3 record(s): one ok; two_b failed (") && err.contains("three ok"),
        "{err}"
    );
    let one = lines(&out.join("one.jsonl"));
    assert_eq!(one[0]["protein"]["sequence"], "MKVL");
    assert_eq!(one[0]["protein"]["name"], "one first");
    assert!(out.join("three.jsonl").is_file());
    assert!(!out.join("two_b.jsonl").exists());

    // A directory that holds files is refused, and --checkpoint skips the
    // finished records.
    let again = derna_cli::run(argv(&[]));
    assert!(again.unwrap_err().to_string().contains("--force"));
    let before = std::fs::metadata(out.join("one.jsonl"))
        .unwrap()
        .modified()
        .unwrap();
    let resumed = derna(
        &argv(&["--checkpoint"])
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
    );
    let err = text(&resumed.stderr);
    assert!(
        err.contains("one skipped (finished)") && err.contains("three skipped"),
        "{err}"
    );
    let after = std::fs::metadata(out.join("one.jsonl"))
        .unwrap()
        .modified()
        .unwrap();
    assert_eq!(before, after);

    // Duplicate ids are refused before anything runs.
    std::fs::write(&fasta, ">a x\nMK\n>a y\nMV\n").unwrap();
    let dup = derna_cli::run(argv(&["--force"])).unwrap_err().to_string();
    assert!(dup.contains("both have the id \"a\""), "{dup}");

    // A one-record file is a single run: --out is a file, as before.
    std::fs::write(&fasta, ">solo\nMKV\n").unwrap();
    let single = dir.join("single.jsonl");
    derna_cli::run(words(&[
        "design",
        "--protein",
        s(&fasta),
        "--lambda",
        "0.5",
        "--out",
        s(&single),
    ]))
    .unwrap();
    assert_eq!(lines(&single)[0]["protein"]["name"], "solo");
}

#[test]
fn a_front_batch_continues_each_record_under_checkpoint() {
    let dir = scratch_dir("front-batch");
    let fasta = dir.join("p.fasta");
    std::fs::write(&fasta, ">p1\nMKVLAK\n>p2\nMPRGLA\n").unwrap();
    let out = dir.join("fronts");
    let argv = |extra: &[&str]| {
        let mut a = words(&[
            "front",
            "--protein",
            s(&fasta),
            "--utr5",
            UTR5,
            "--out",
            s(&out),
            "--report",
            "walk.json",
        ]);
        a.extend(words(extra));
        a
    };
    derna_cli::run(argv(&[])).unwrap();
    for id in ["p1", "p2"] {
        assert!(out.join(format!("{id}.jsonl")).is_file());
        assert!(out.join(format!("{id}.report.json")).is_file());
    }
    // Remove one record's result: a rerun with --checkpoint redoes that one
    // and skips the other.
    std::fs::remove_file(out.join("p2.jsonl")).unwrap();
    std::fs::remove_file(out.join("p2.report.json")).unwrap();
    let before = std::fs::metadata(out.join("p1.jsonl"))
        .unwrap()
        .modified()
        .unwrap();
    derna_cli::run(argv(&["--checkpoint"])).unwrap();
    assert!(out.join("p2.jsonl").is_file());
    let after = std::fs::metadata(out.join("p1.jsonl"))
        .unwrap()
        .modified()
        .unwrap();
    assert_eq!(before, after, "the finished record was run again");
}
