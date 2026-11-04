//! The set of keys a front header carries, named per invocation.
//!
//! Flags have `help_names_every_flag`; header keys had no equivalent, so a key
//! renamed or removed was an output change with nothing asserting it.
//!
//! The lists below are deliberately hardcoded. The repo's spec rule says an
//! enumerated list rots and to name the container instead — that rule is about
//! *specs*, where rot is silent decay, and it inverts here: in a test the
//! enumeration *is* the assertion, and its rotting is the alarm firing. Do not
//! replace these with a container; that would delete the test.
//!
//! An addition fails this test too, which is the point: it forces the author to
//! decide whether readers may take the key as optional. For an addition they
//! may, and `format_version` is **not** bumped — that field means "this reader
//! would misread this file" (`show` refuses any version above its own), so
//! bumping it for a new key would make every existing `derna` refuse files it
//! reads correctly. A removal or a rename is what owes a bump, and only on the
//! files that carry the renamed key: `result::format_version` declares 2 on a
//! file holding `modular` or `modular_front` and 1 on every other, whose
//! content a version-1 reader takes unchanged (`tests/format_version.rs`).
//!
//! Why the sets are per invocation, and why they come from a fixture rather
//! than from real files: a header's key set is a function of **both** the mode
//! and the build. Measured on real corpora, one modular-front arm of 24 files
//! carried four distinct sets that partition one-to-one onto four build pins,
//! while a motif-front corpus of 104 files carried one set across seven builds
//! — because the keys that moved (`gave_up`, `epsilon`, `outcome`,
//! `cut_build_s`) are modular-front keys that a motif front never emits. So a key
//! set moves when a build changes keys that *that mode* emits: it is a function
//! of mode and build jointly, and neither alone. A set taken from production
//! asserts a build, and only a fixture pins the argv and the build together.
//!
//! That seven-build span is evidence the format does not drift, and it was
//! established by joining each output file to a build ledger — not through any
//! header field. `version` is the crate version and reads the same on every
//! file, so a reader who reproduces the span from the header gets one group and
//! concludes the span was never tested.

use derna_cli::front;
use derna_cli::show;

const PROTEIN: &str = "MKVLAKGIVLAGQWERS";
const UTR5: &str = "ACAUUUGCUU";
const UTR3: &str = "GGCCAAUUGG";

/// Every key a plain constrained front writes, for the argv in `plain()`.
const PLAIN_KEYS: [&str; 28] = [
    "access",
    "checkpoint",
    "codon_usage",
    "command",
    "complete",
    "deterministic",
    "forbid",
    "format",
    "format_version",
    "front_from",
    "gaps",
    "max_rss_gb",
    "max_secs",
    "open",
    "pass_grid",
    "peak_rss_kb",
    "points",
    "protein",
    "range",
    "record",
    "resumed",
    "stop",
    "stopped_by",
    "threads",
    "utr3",
    "utr5",
    "version",
    "wall_s",
];

/// Every key a modular front writes, for the argv in `so()`. It is
/// `PLAIN_KEYS` less `checkpoint`, plus the search's own seventeen.
const MODULAR_KEYS: [&str; 44] = [
    "access",
    "anchors",
    "certify_s",
    "codon_usage",
    "command",
    "complete",
    "constrained_open",
    "constrained_points",
    "cut_build_s",
    "deterministic",
    "empty_stop",
    "epsilon",
    "forbid",
    "format",
    "format_version",
    "front_from",
    "gaps",
    "gave_up",
    "hold_barren",
    "max_rss_gb",
    "max_secs",
    "modular_front",
    "open",
    "outcome",
    "pass_grid",
    "peak_rss_kb",
    "points",
    "protein",
    "range",
    "record",
    "resumed",
    "resumed_search",
    "seeds",
    "shed_on_memory",
    "stop",
    "stopped_by",
    "stretches",
    "threads",
    "unchecked",
    "utr3",
    "utr5",
    "version",
    "walks_shed",
    "wall_s",
];

fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("derna-front-header-keys-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    let _ = std::fs::remove_file(&path);
    path
}

fn run(extra: &[&str], out: &std::path::Path) -> show::Rendered {
    let mut argv = words(&[
        "--protein",
        PROTEIN,
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--threads",
        "2",
        "--out",
        out.to_str().unwrap(),
    ]);
    argv.extend(words(extra));
    front::run(&front::parse(argv).unwrap()).unwrap();
    show::read(Some(out)).unwrap()
}

/// The header's keys, sorted, so the comparison does not depend on the order
/// the fields were inserted in.
fn keys(r: &show::Rendered) -> Vec<String> {
    let mut k: Vec<String> = r.header.as_object().unwrap().keys().cloned().collect();
    k.sort();
    k
}

/// What a key-set mismatch should say: which keys appeared and which went
/// missing, rather than two long lists to diff by eye.
fn assert_keys(got: &[String], want: &[&str], what: &str) {
    let got: std::collections::BTreeSet<&str> = got.iter().map(String::as_str).collect();
    let want: std::collections::BTreeSet<&str> = want.iter().copied().collect();
    let added: Vec<_> = got.difference(&want).collect();
    let gone: Vec<_> = want.difference(&got).collect();
    assert!(
        added.is_empty() && gone.is_empty(),
        "{what} header keys moved: added {added:?}, removed {gone:?}. \
         A removal or rename also owes a FORMAT_VERSION bump; an addition does not."
    );
}

#[test]
fn a_plain_front_header_carries_exactly_these_keys() {
    let plain = run(&[], &scratch("plain.jsonl"));
    assert_keys(&keys(&plain), &PLAIN_KEYS, "plain front");
}

#[test]
fn a_modular_front_header_carries_exactly_these_keys() {
    let whole = scratch("whole.jsonl");
    run(&["--forbid", "5-cds,cds-3"], &whole);
    let so = run(
        &[
            "--forbid",
            "5-cds,cds-3",
            "--front-from",
            whole.to_str().unwrap(),
            "--modular",
            // The key set is written whatever the search found, so a short
            // budget asserts the same keys as a completed walk.
            "--max-secs",
            "2",
        ],
        &scratch("so.jsonl"),
    );
    assert_keys(&keys(&so), &MODULAR_KEYS, "modular front");
}
