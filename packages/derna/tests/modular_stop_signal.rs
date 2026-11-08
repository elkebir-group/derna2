//! A `front --modular` run that is ended from outside writes the front it
//! has, and says so: `complete` is `false` and `stopped_by` reads `signal`.
//!
//! The run under test is a real process. The defect this covers is a process
//! ended from outside — a budget it was never given, a core handed back, an
//! operator's `kill` — and an in-process call cannot be sent a signal without
//! ending the test binary along with it. It is also one test in a binary of
//! its own, because the stop it installs and the walks it ends are
//! process-wide: any test after it in the same binary would see a front
//! stopped by something it never asked for.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde_json::Value;

/// A cell whose certification is quick and whose search does not finish: the
/// band above its best-adapted seed has no cap, so the run keeps rounding
/// over it until something stops it. A run that finished on its own would
/// leave the signal nothing to interrupt.
const PROTEIN: &str = "MSLLLKVGQWER";
const UTR5: &str = "ACAUUUGCUUCUGACACAACUGUGUAUGUAUGGCAAGAUCAUCUUCGUG";
const UTR3: &str = "GCUCGCUUUCUUGCUGUCCAAUUUCUAUUAAAGGUUCCUUUGUUCCCUA";

/// How long the walk is given to settle its first units before the signal,
/// and how long the signalled run is then given to write its front.
const SETTLES_IN: Duration = Duration::from_secs(120);
const WRITES_IN: Duration = Duration::from_secs(300);

fn read_lines(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// The sequences a search has settled: the constrained-front points whose
/// check passed, and the designs the walk found.
fn settled(progress: &[Value]) -> (Vec<String>, Vec<String>) {
    let of = |key: &str, keep: &dyn Fn(&Value) -> bool| -> Vec<String> {
        progress
            .iter()
            .filter(|l| keep(l))
            .filter_map(|l| l[key].as_str().map(str::to_string))
            .collect()
    };
    (
        of("checked", &|l| l["modular"] == true),
        of("found", &|_| true),
    )
}

/// Wait until `path` holds at least `lines` whole lines, or give up. The file
/// is counted rather than parsed, because it is being appended to while this
/// reads it and the line in flight is not this test's business.
fn waiting_for(path: &Path, lines: usize) -> bool {
    let since = Instant::now();
    while since.elapsed() < SETTLES_IN {
        let whole = std::fs::read_to_string(path)
            .map(|text| text.bytes().filter(|&b| b == b'\n').count())
            .unwrap_or(0);
        if whole >= lines {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

fn scratch_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("derna-modular-signal-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `derna front --modular` on the cell, ended by `SIGTERM` once its search
/// has settled something, writes the front it had settled by then and labels
/// it as the partial it is.
#[test]
fn a_signalled_modular_run_writes_the_front_it_has() {
    let dir = scratch_dir();
    let (out, log) = (dir.join("so.jsonl"), dir.join("stderr.log"));
    let progress = {
        let mut name = out.clone().into_os_string();
        name.push(".search.jsonl");
        PathBuf::from(name)
    };
    let mut run = Command::new(env!("CARGO_BIN_EXE_derna"))
        .args([
            "front",
            "--protein",
            PROTEIN,
            "--utr5",
            UTR5,
            "--utr3",
            UTR3,
        ])
        .args(["--stop", "none", "--forbid", "5-cds,cds-3"])
        .args(["--modular", "--checkpoint", "--threads", "1"])
        .args(["--out", out.to_str().unwrap()])
        .stderr(std::fs::File::create(&log).unwrap())
        .spawn()
        .unwrap();

    // The run line and one settled check: enough to know the search is under
    // way. Everything after the signal is the run's own doing, so the test
    // does not depend on where in the search it lands.
    let settling = waiting_for(&progress, 2);
    let killed = Command::new("kill")
        .args(["-TERM", &run.id().to_string()])
        .status()
        .unwrap();
    assert!(settling, "the search settled nothing in {SETTLES_IN:?}");
    assert!(killed.success(), "the run was not signalled");

    let since = Instant::now();
    let status = loop {
        if let Some(status) = run.try_wait().unwrap() {
            break status;
        }
        if since.elapsed() >= WRITES_IN {
            // Leave no run of this cell behind: it has no budget of its own.
            run.kill().unwrap();
            run.wait().unwrap();
            panic!("the signalled run was still going after {WRITES_IN:?}");
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let stderr = std::fs::read_to_string(&log).unwrap();
    assert!(status.success(), "{status}: {stderr}");
    assert!(out.exists(), "a signalled run left no result: {stderr}");

    let result = read_lines(&out);
    let (header, points) = result.split_first().unwrap();
    assert_eq!(header["modular_front"], true, "{header}");
    assert_eq!(header["complete"], false, "{header}");
    assert_eq!(header["stopped_by"], "signal", "{header}");
    // Non-vacuity: the run was given no budget of any kind, so `signal` is
    // the only thing that could have stopped it short.
    assert_eq!(header["max_secs"], Value::Null, "{header}");
    assert_eq!(header["max_rss_gb"], Value::Null, "{header}");
    assert!(stderr.contains("stopped by SIGTERM"), "{stderr}");

    // The front it wrote is the work the progress file settled: every point
    // is a check that passed or a design the walk found, and the seeds are
    // exactly the checks that passed.
    let (passed, found) = settled(&read_lines(&progress));
    assert_eq!(header["seeds"].as_u64().unwrap() as usize, passed.len());
    assert_eq!(header["points"].as_u64().unwrap() as usize, points.len());
    assert!(!points.is_empty(), "{header}");
    for point in points {
        let sequence = point["sequence"].as_str().unwrap().to_string();
        assert!(
            passed.contains(&sequence) || found.contains(&sequence),
            "a point the progress file never settled: {point}"
        );
    }
}
