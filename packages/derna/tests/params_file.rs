//! `--params PATH` folds with the energy parameters in a file.
//!
//! The built-in table written to a file gives the run the flag-less command
//! gives, point for point; a table that differs gives a different run, so the
//! flag is shown to reach the folds rather than being read and dropped; and a
//! file that cannot be read is refused by name.

use std::path::PathBuf;
use std::process::Command;

use derna_lib::Params;

const PROTEIN: &str = "MKVLAKGIVL";
const UTR5: &str = "ACAUUUGCUU";
const UTR3: &str = "GGCCAAUUGG";

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("derna-params-{}-{name}", std::process::id()))
}

/// The point records `derna design` prints at two weights, with `extra` words.
fn design_points(extra: &[&str]) -> Vec<String> {
    let out = Command::new(env!("CARGO_BIN_EXE_derna"))
        .args([
            "design",
            "--protein",
            PROTEIN,
            "--utr5",
            UTR5,
            "--utr3",
            UTR3,
            "--lambda",
            "0.3,0.7",
            "--quiet",
        ])
        .args(extra)
        .output()
        .expect("the binary runs");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let points: Vec<String> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .filter(|l| l.contains(r#""record":"point""#))
        .map(str::to_string)
        .collect();
    assert_eq!(points.len(), 2, "one point per weight");
    points
}

#[test]
fn the_built_in_table_from_a_file_gives_the_default_run() {
    let path = temp("built-in.params");
    std::fs::write(&path, Params::load_default().unwrap().dump_canonical()).unwrap();
    let from_file = design_points(&["--params", path.to_str().unwrap()]);
    std::fs::remove_file(&path).unwrap();
    assert_eq!(from_file, design_points(&[]));
}

#[test]
fn a_different_table_gives_a_different_run() {
    let mut params = Params::load_default().unwrap();
    for e in params.stack.flat.iter_mut() {
        *e /= 2;
    }
    let path = temp("halved-stacks.params");
    std::fs::write(&path, params.dump_canonical()).unwrap();
    let from_file = design_points(&["--params", path.to_str().unwrap()]);
    std::fs::remove_file(&path).unwrap();
    assert_ne!(from_file, design_points(&[]));
}

#[test]
fn an_unreadable_file_is_refused_by_name() {
    let out = Command::new(env!("CARGO_BIN_EXE_derna"))
        .args([
            "design",
            "--protein",
            PROTEIN,
            "--lambda",
            "0.5",
            "--params",
            "/nonexistent/table.params",
        ])
        .output()
        .expect("the binary runs");
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("/nonexistent/table.params"), "{err}");
}
