//! Every fixture on disk is one a test reads, and every fixture a test reads
//! is on disk.
//!
//! Fixtures are generated output, so they drift silently: a case renamed in
//! the generator leaves its old file behind, and a reader of a file nobody
//! generates any more keeps passing against a stale baseline. Two orphans had
//! already accumulated here — one from an earlier naming of the equality
//! cases, one a sidecar the instrument writes beside its `--out` — and
//! neither affected a test, which is exactly why neither was noticed.

use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn manifest(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(data().join(name)).unwrap()).unwrap()
}

/// Fixtures a test names directly rather than through a manifest. Each is
/// listed here so that adding one is a deliberate act.
const NAMED_BY_A_TEST: [&str; 2] = ["design-motif-global.jsonl", "flank-pairs.json"];

#[test]
fn no_fixture_is_an_orphan_and_none_is_missing() {
    let mut expected: BTreeSet<String> = NAMED_BY_A_TEST.iter().map(|s| s.to_string()).collect();
    for (file, prefix) in [
        ("design-cases.json", "design"),
        ("front-cases.json", "front"),
    ] {
        expected.insert(file.to_string());
        for case in manifest(file)["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            expected.insert(format!("{prefix}-{name}.jsonl"));
            // A case names its own protein, and a cell's protein may be a
            // fixture here rather than elsewhere in the repository.
            if let Some(fasta) = case["fasta"].as_str() {
                if let Some(base) = Path::new(fasta).file_name() {
                    if data().join(base).exists() {
                        expected.insert(base.to_string_lossy().into_owned());
                    }
                }
            }
            if prefix == "front" {
                // A front case also records the walk's own report, which is
                // where the hull's vertices come from.
                expected.insert(format!("{prefix}-{name}.report.json"));
            }
        }
    }

    let present: BTreeSet<String> = std::fs::read_dir(data())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();

    let orphans: Vec<&String> = present.difference(&expected).collect();
    let missing: Vec<&String> = expected.difference(&present).collect();
    assert!(
        orphans.is_empty(),
        "fixtures on disk that no test reads (regenerate or delete them): {orphans:?}"
    );
    assert!(
        missing.is_empty(),
        "fixtures a test reads that are not on disk (run the generators): {missing:?}"
    );
}

#[test]
fn every_case_names_the_flags_both_sides_ran() {
    // A case with no `derna` words would compare the command against the
    // instrument's default rather than against a stated constraint, which is
    // the comparison the equalities exist to avoid.
    for file in ["design-cases.json", "front-cases.json"] {
        let spec = manifest(file);
        let cases = spec["cases"].as_array().unwrap();
        assert!(!cases.is_empty(), "{file} names no case");
        for case in cases {
            assert!(
                case["name"].as_str().is_some(),
                "{file}: a case has no name"
            );
            assert!(
                case["binary"].as_array().is_some() && case["derna"].as_array().is_some(),
                "{file}: {} does not say what both sides ran",
                case["name"]
            );
        }
    }
}
