//! `derna front --modular --motif --max-rss-gb`: the memory ceiling reaches
//! the motif walk, although the time budget belongs to the search.
//!
//! One test, in a test binary of its own, because the ceiling is a
//! process-wide watcher: it stops this process's walks for good once it fires,
//! so any test that follows it in the same binary sees a front stopped by a
//! budget it never asked for.

#![cfg(target_os = "linux")]

use serde_json::Value;

/// Run the cell with these extra words and read back the lines it wrote.
fn run(extra: &[&str]) -> Vec<Value> {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let nth = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let out = std::env::temp_dir().join(format!(
        "derna-modular-ceiling-{}-{nth}.jsonl",
        std::process::id()
    ));
    let mut argv: Vec<String> = [
        "front",
        "--protein",
        "MPRGLA",
        "--utr5",
        "CCGCGG",
        "--stop",
        "none",
        "--modular",
        "--motif",
        "CGC",
        "--force",
        "--out",
        out.to_str().unwrap(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    argv.extend(extra.iter().map(|s| s.to_string()));
    derna_cli::run(argv).unwrap();
    let text = std::fs::read_to_string(&out).unwrap();
    std::fs::remove_file(&out).unwrap();
    text.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// Under `--modular` the time budget is the search's, but the memory ceiling
/// still reaches the motif walk: a ceiling every process exceeds stops the walk
/// with boxes open, so the search starts from fewer motif-free points than the
/// walk finds without one.
///
/// The uncapped run goes first: once the ceiling has fired, this process's
/// walks stay stopped.
#[test]
fn a_memory_ceiling_reaches_the_motif_walk_under_a_modular_run() {
    let walked = |lines: &[Value]| lines[0]["constrained_points"].as_u64().unwrap();
    let whole = run(&[]);
    let capped = run(&["--max-rss-gb", "0.001"]);
    assert!(
        walked(&capped) < walked(&whole),
        "{} vs {}",
        capped[0],
        whole[0]
    );
}
