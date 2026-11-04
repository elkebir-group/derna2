//! `--stop` with a comma list: a front for each stop codon the list names, and
//! none for the stops it leaves out.
//!
//! `--stop each` is the list of all three and is covered by `stop_each.rs`. What
//! only a list can get wrong is which stops it walks and in what spelling, so
//! the claims here are: each named stop's file is the front its pinned run
//! writes, the stop left out gets no file, and a list that is not a set of stop
//! codons is refused before any folding.
//!
//! The pinned runs are the oracle, as in `stop_each.rs`, and the cell is the
//! stop slot's, so every run is small and complete without a budget.

use serde_json::Value;

const PROTEIN: &str = "MPRGLA";
const UTR5: &str = "CCGCGG";

/// A temporary output path of this process, named by a counter: the tests run
/// in one process on several threads and each deletes its own files.
fn out_path(tag: &str) -> std::path::PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let nth = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "derna-stop-list-{tag}-{}-{nth}.jsonl",
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

/// `path` with `tag` before its extension, as the surface names its files.
fn tagged(path: &std::path::Path, tag: &str) -> std::path::PathBuf {
    path.with_file_name(format!(
        "{}.{tag}.jsonl",
        path.file_stem().unwrap().to_string_lossy()
    ))
}

/// Remove whatever a run over `out` may have left: the pooled fronts as well
/// as the arms, so a test that only reads some of them does not leak the rest.
fn clean(out: &std::path::Path) {
    for tag in ["UAA", "UAG", "UGA", "union", "modular"] {
        let _ = std::fs::remove_file(tagged(out, tag));
    }
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

/// A two-stop list writes the two named stops' fronts, each the front its
/// pinned run writes, and no file for the stop it leaves out.
///
/// The list is written out of `STOP_CODONS` order and in DNA spelling, lower
/// case, so a run that walked the lanes by position or matched the words
/// literally would write the wrong files or none.
#[test]
fn a_list_writes_the_named_stops_and_only_those() {
    let out = out_path("two");
    front(&out, &["--stop", "uga,TAA"]).expect("a two-stop list runs");
    assert!(
        !tagged(&out, "UAG").exists(),
        "UAG is not in the list and was written"
    );
    for codon in ["UGA", "UAA"] {
        let got = read(&tagged(&out, codon));
        assert_eq!(got[0]["stop"].as_str(), Some(codon));
        let pinned = out_path(codon);
        front(&pinned, &["--stop", codon]).expect("a pinned run runs");
        let want = read(&pinned);
        assert_eq!(
            points(&got),
            points(&want),
            "--stop uga,TAA wrote a different {codon} front than --stop {codon}"
        );
    }
    clean(&out);
}

/// A list that names something other than a stop codon, or one stop twice, is
/// refused by what the caller wrote, and nothing is written.
#[test]
fn a_list_that_is_not_a_set_of_stops_is_refused() {
    for (list, says) in [
        ("UAA,UGG", "\"UGG\""),
        ("UAA,free", "\"free\""),
        ("UAA,uaa", "UAA twice"),
    ] {
        let out = out_path("refused");
        let err = front(&out, &["--stop", list]).expect_err(list);
        assert!(err.to_string().contains(says), "{list}: {err}");
        for tag in ["UAA", "UAG", "UGA", "union", "modular"] {
            assert!(!tagged(&out, tag).exists(), "{list} wrote {tag}");
        }
    }
}
