//! `--stop each`: every stop codon's own front from one run.
//!
//! A front over a designed stop codon minimizes over the three stops, so a
//! design that is best under one stop can be dominated by a design under
//! another and leave the front. Properties the objectives do not see —
//! structural optimality above all — are then lost with it. `--stop each`
//! keeps the stops apart: one search over a lattice whose stop codon is part
//! of the state, read once per stop.
//!
//! The claim under test is that each file is what the pinned run writes. The
//! pinned runs are the oracle, and the sequences are compared as well as the
//! objectives, since the point of the surface is the designs it keeps rather
//! than the hull they lie on. The three fronts are also asserted to differ,
//! because a run that ignored the stop would otherwise pass.
//!
//! The cell is the stop slot's, so every run is small and complete without a
//! budget.

use serde_json::Value;

const STOPS: [&str; 3] = ["UAA", "UAG", "UGA"];
const PROTEIN: &str = "MPRGLA";
const UTR5: &str = "CCGCGG";

/// A temporary output path of this process, named by a counter: the tests run
/// in one process on several threads and each deletes its own files.
fn out_path(tag: &str) -> std::path::PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let nth = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "derna-stop-each-{tag}-{}-{nth}.jsonl",
        std::process::id()
    ))
}

/// `derna front` on the cell with `extra`, writing to `out`.
fn front(out: &std::path::Path, extra: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let mut argv: Vec<String> = [
        "front",
        "--protein",
        PROTEIN,
        "--utr5",
        UTR5,
        "--force",
        // One worker: the front is the same at any thread count, and a test
        // that says so on a shared host costs its neighbours nothing.
        "--threads",
        "1",
        "--out",
        out.to_str().unwrap(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    argv.extend(extra.iter().map(|s| s.to_string()));
    derna_cli::run(argv)
}

/// The lines of a written front, the file removed.
fn read(path: &std::path::Path) -> Vec<Value> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    std::fs::remove_file(path).unwrap();
    text.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// `path` with `codon` before its extension, as the surface names its files.
fn per_stop(path: &std::path::Path, codon: &str) -> std::path::PathBuf {
    path.with_file_name(format!(
        "{}.{codon}.jsonl",
        path.file_stem().unwrap().to_string_lossy()
    ))
}

/// One `--stop each` run's three fronts, in `STOPS` order.
fn each() -> Vec<Vec<Value>> {
    let out = out_path("each");
    front(&out, &["--stop", "each"]).expect("--stop each runs");
    STOPS.iter().map(|c| read(&per_stop(&out, c))).collect()
}

/// A point as this test compares it: the design and both objectives.
fn points(lines: &[Value]) -> Vec<(String, f64, f64)> {
    lines[1..]
        .iter()
        .map(|p| {
            (
                p["sequence"].as_str().unwrap().to_string(),
                p["mfe"].as_f64().unwrap(),
                p["cai_log"].as_f64().unwrap(),
            )
        })
        .collect()
}

/// Each `--stop each` file is the front its pinned run writes: the same
/// designs, the same objectives, in the same order.
#[test]
fn each_file_is_the_pinned_runs_front() {
    let got = each();
    for (codon, got) in STOPS.iter().zip(&got) {
        let out = out_path(codon);
        front(&out, &["--stop", codon]).expect("a pinned run runs");
        let want = read(&out);
        assert_eq!(
            points(got),
            points(&want),
            "--stop each wrote a different front than --stop {codon}"
        );
    }
}

/// The fronts differ between stops, so a run that minimized over the stops
/// instead of keeping them apart could not have written these three files.
#[test]
fn the_stops_fronts_differ() {
    let got = each();
    let first = points(&got[0]);
    assert!(
        got[1..].iter().any(|f| points(f) != first),
        "every stop's front is the same, so the oracle proves nothing"
    );
}

/// Each file's header names the stop codon it is the front of, so a reader
/// holding one file alone knows which stop it describes.
#[test]
fn each_header_names_its_stop() {
    for (codon, lines) in STOPS.iter().zip(each()) {
        assert_eq!(lines[0]["stop"].as_str(), Some(*codon));
        assert_eq!(lines[0]["record"].as_str(), Some("header"));
    }
}

/// Every design of a stop's file ends in that stop codon: the cell has no 3′
/// UTR, so the stop is the molecule's last three nucleotides.
#[test]
fn every_design_ends_in_its_files_stop() {
    for (codon, lines) in STOPS.iter().zip(each()) {
        for (sequence, _, _) in points(&lines) {
            assert_eq!(&sequence[sequence.len() - 3..], *codon, "{sequence}");
        }
    }
}

/// The surface writes one file per stop, so it needs a path to name them
/// from: it is refused without `--out`, before any folding.
#[test]
fn each_without_an_out_path_is_refused() {
    let err = derna_cli::run(
        [
            "front",
            "--protein",
            PROTEIN,
            "--utr5",
            UTR5,
            "--stop",
            "each",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
    )
    .expect_err("--stop each needs --out");
    assert!(err.to_string().contains("--out"), "{err}");
}

/// The flags `--stop each` does not yet carry are refused by name rather than
/// silently ignored or applied to one stop only.
#[test]
fn each_refuses_the_flags_it_does_not_carry() {
    for flag in ["--checkpoint", "--front-from", "--resume"] {
        let out = out_path("refuse");
        let extra: Vec<&str> = match flag {
            "--checkpoint" => vec!["--stop", "each", flag],
            _ => vec!["--stop", "each", flag, "/nonexistent/front.jsonl"],
        };
        let err = front(&out, &extra).expect_err("{flag} is refused");
        assert!(err.to_string().contains(flag), "{flag}: {err}");
    }
}

/// `derna design` solves one weight and writes one design, so it has nothing
/// to do with a front per stop: it names the flags it does take.
#[test]
fn design_refuses_each() {
    let err = derna_cli::run(
        [
            "design",
            "--protein",
            PROTEIN,
            "--utr5",
            UTR5,
            "--stop",
            "each",
            "--lambda",
            "0.5",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
    )
    .expect_err("design does not take --stop each");
    assert!(err.to_string().contains("each"), "{err}");
}

/// With `--modular` each file is the pinned run's modular front: the search that follows the walk runs per stop, over the
/// molecule whose stop that file's walk fixed.
#[test]
fn each_file_is_the_pinned_runs_modular_front() {
    let out = out_path("each-so");
    front(&out, &["--stop", "each", "--modular"]).expect("--stop each runs");
    for codon in STOPS {
        let got = read(&per_stop(&out, codon));
        let pinned = out_path(codon);
        front(&pinned, &["--stop", codon, "--modular"]).expect("a pinned run runs");
        let want = read(&pinned);
        assert_eq!(
            points(&got),
            points(&want),
            "--stop each wrote a different modular front than --stop {codon}"
        );
        assert_eq!(got[0]["complete"], want[0]["complete"], "{codon}");
        for p in &got[1..] {
            assert_eq!(p["modular"], true, "{codon}: {p}");
        }
    }
}
