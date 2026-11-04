//! `derna front --max-secs` and `--max-rss-gb` on the plain constrained walk:
//! what the command writes when a budget stops it before the whole front.
//!
//! The library's own test asserts that the points of a stopped walk are front
//! points and that each open box names two of the walk's hull points. Here the
//! command's side is checked: the run succeeds, the file says it is
//! incomplete, names what stopped it and lists the open boxes, and a run
//! without a budget says the opposite.

use derna_cli::front;
use derna_cli::show;

/// A protein whose walk settles several gaps, so a budget leaves boxes open.
const PROTEIN: &str = "MKVLAKGIVLAGQWERS";
const UTR5: &str = "ACAUUUGCUU";
const UTR3: &str = "GGCCAAUUGG";

fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

/// A fresh path under the system's temporary directory for one test's file.
fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("derna-front-budget-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    let _ = std::fs::remove_file(&path);
    path
}

/// Run `derna front` with these words and read back what it wrote.
fn front_file(extra: &[&str], out: &std::path::Path) -> show::Rendered {
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

#[test]
fn a_budgeted_plain_front_says_what_it_left_open() {
    let whole = front_file(&[], &scratch("whole.jsonl"));
    assert_eq!(whole.header["complete"], true);
    assert_eq!(whole.header["stopped_by"], serde_json::Value::Null);
    assert!(whole.header["open"].as_array().unwrap().is_empty());

    let stopped = front_file(&["--max-secs", "0.000001"], &scratch("stopped.jsonl"));
    assert_eq!(stopped.header["complete"], false);
    assert_eq!(stopped.header["stopped_by"], "time");
    let open = stopped.header["open"].as_array().unwrap();
    assert!(!open.is_empty());
    // A budget that stopped the walk before it settled anything would leave
    // nothing to compare, and the box ends would be all the file says.
    assert!(stopped.points.len() < whole.points.len());

    // Every point of the stopped front is a point of the whole one.
    let key = |p: &serde_json::Value| {
        (
            (p["mfe"].as_f64().unwrap() * 100.0).round() as i64,
            p["cai_log"].as_f64().unwrap(),
        )
    };
    let whole_keys: Vec<(i64, f64)> = whole.points.iter().map(key).collect();
    for point in stopped.points.iter().map(key) {
        assert!(
            whole_keys
                .iter()
                .any(|q| q.0 == point.0 && (q.1 - point.1).abs() < 1e-9),
            "{point:?} is not a point of the whole front"
        );
    }

    // Each open box spans a real energy range, the more stable end first.
    for box_ in open {
        let (lo, hi) = (
            box_["mfe_lo"].as_f64().unwrap(),
            box_["mfe_hi"].as_f64().unwrap(),
        );
        assert!(lo < hi, "{lo} is not below {hi}");
    }

    // The walk folds its stable end first and every box's stable end is a
    // hull point it folded exactly, so the lower of the stopped front's least
    // energy and the boxes' least `mfe_lo` is the whole front's least energy:
    // a stopped front reports its own floor.
    let cents = |e: f64| (e * 100.0).round() as i64;
    let least = |es: &mut dyn Iterator<Item = f64>| es.map(cents).min().unwrap_or(i64::MAX);
    let whole_lo = least(&mut whole.points.iter().map(|p| p["mfe"].as_f64().unwrap()));
    let point_lo = least(&mut stopped.points.iter().map(|p| p["mfe"].as_f64().unwrap()));
    let box_lo = least(&mut open.iter().map(|b| b["mfe_lo"].as_f64().unwrap()));
    assert_eq!(point_lo.min(box_lo), whole_lo);
    // Without this the check passes on any stop that kept the stable end.
    assert!(point_lo > whole_lo, "the open boxes were not needed");
}

/// A stopped front is not a front to build on: the modular
/// search would certify its designs against a front still missing points.
/// A plain `front` would write a stopped walk out again without what it solved
/// in each box, so it refuses one; a modular search reads it with
/// its boxes (`modular_front.rs`).
#[test]
fn a_stopped_front_is_refused_as_a_plain_front_to_read() {
    let stopped = scratch("partial.jsonl");
    let read = front_file(&["--max-secs", "0.000001"], &stopped);
    assert_eq!(read.header["complete"], false);
    let argv = words(&[
        "--protein",
        PROTEIN,
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--front-from",
        stopped.to_str().unwrap(),
        "--out",
        scratch("refused.jsonl").to_str().unwrap(),
    ]);
    let e = front::run(&front::parse(argv).unwrap())
        .unwrap_err()
        .to_string();
    assert!(e.contains("box(es) open"), "{e}");
    assert!(e.contains("--resume continues it"), "{e}");
}

#[test]
fn a_budget_is_refused_on_a_front_that_is_read_back() {
    let whole = scratch("read-back.jsonl");
    front_file(&[], &whole);
    let e = front::parse(words(&[
        "--protein",
        PROTEIN,
        "--front-from",
        whole.to_str().unwrap(),
        "--max-secs",
        "10",
    ]))
    .unwrap_err()
    .to_string();
    assert!(e.contains("--front-from runs none"), "{e}");
}

/// A modular run on a front read back still searches, and the
/// budget is that search's: it is accepted, and the header records it.
#[test]
fn a_budget_reaches_the_search_on_a_front_that_is_read_back() {
    let whole = scratch("so-read-back.jsonl");
    front_file(&["--forbid", "5-cds,cds-3"], &whole);
    let out = scratch("so-budgeted.jsonl");
    let argv = words(&[
        "--protein",
        PROTEIN,
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--forbid",
        "5-cds,cds-3",
        "--front-from",
        whole.to_str().unwrap(),
        "--modular",
        "--max-secs",
        "600",
        "--threads",
        "2",
        "--out",
        out.to_str().unwrap(),
    ]);
    front::run(&front::parse(argv).unwrap()).unwrap();
    let read = show::read(Some(&out)).unwrap();
    assert_eq!(read.header["modular_front"], true);
    assert_eq!(read.header["max_secs"], 600.0);
}

/// The search's give-up policy is in the header, so an absent `gave_up` can be
/// read.
///
/// `gave_up` alone is ambiguous where it matters: a front that never carried
/// the policy and a front that carried it and never fired both report nothing,
/// and those are a missing flag and a heuristic that held — different
/// readings. The thresholds separate them without the run's command line.
#[test]
fn the_header_records_the_give_up_policy_whether_or_not_it_fired() {
    let whole = scratch("so-policy-read-back.jsonl");
    front_file(&["--forbid", "5-cds,cds-3"], &whole);
    let search = |name: &str, extra: &[&str]| {
        let out = scratch(name);
        let mut argv = words(&[
            "--protein",
            PROTEIN,
            "--utr5",
            UTR5,
            "--utr3",
            UTR3,
            "--forbid",
            "5-cds,cds-3",
            "--front-from",
            whole.to_str().unwrap(),
            "--modular",
            "--threads",
            "2",
            // The policy is recorded from the command line, not from how the
            // search ended, so a budget short enough to stop the walk exercises
            // the same fields and keeps the test cheap.
            "--max-secs",
            "2",
            "--out",
            out.to_str().unwrap(),
        ]);
        argv.extend(words(extra));
        front::run(&front::parse(argv).unwrap()).unwrap();
        show::read(Some(&out)).unwrap()
    };

    // Without the policy the fields say so, rather than being absent.
    let plain = search("so-policy-none.jsonl", &[]);
    assert_eq!(plain.header["empty_stop"], serde_json::Value::Null);
    assert_eq!(plain.header["hold_barren"], false);

    // With it, the thresholds are the ones asked for. Neither clause of the
    // stop is reached here, so `gave_up` stays null while the policy is set —
    // which is the reading the thresholds exist to disambiguate.
    let held = search(
        "so-policy-set.jsonl",
        &[
            "--empty-stop-rounds",
            "7",
            "--empty-stop-derivations",
            "1000000",
            "--hold-barren",
        ],
    );
    assert_eq!(held.header["empty_stop"]["rounds"], 7);
    assert_eq!(held.header["empty_stop"]["derivations"], 1_000_000);
    assert_eq!(held.header["hold_barren"], true);
    assert_eq!(held.header["gave_up"], serde_json::Value::Null);
}

/// A stopped plain front resumed is the whole front: the boxes it left are
/// walked, its own points are returned with theirs, and the header says what
/// it was resumed from.
#[test]
fn a_resumed_plain_front_is_the_whole_front() {
    let whole = front_file(&[], &scratch("whole-resume.jsonl"));
    let from = scratch("stopped-resume.jsonl");
    let stopped = front_file(&["--max-secs", "0.000001"], &from);
    // A stopped walk with nothing open would resume to itself.
    assert_eq!(stopped.header["complete"], false);
    assert!(!stopped.header["open"].as_array().unwrap().is_empty());

    let resumed = front_file(
        &["--resume", from.to_str().unwrap()],
        &scratch("resumed.jsonl"),
    );
    assert_eq!(resumed.header["complete"], true);
    assert_eq!(resumed.header["resumed"]["points"], stopped.points.len());
    assert_eq!(
        resumed.header["resumed"]["open"],
        stopped.header["open"].as_array().unwrap().len()
    );
    assert_eq!(resumed.points.len(), whole.points.len());
    for (a, b) in resumed.points.iter().zip(&whole.points) {
        assert_eq!(
            (a["mfe"].as_f64().unwrap() * 100.0).round(),
            (b["mfe"].as_f64().unwrap() * 100.0).round()
        );
        assert!(
            (a["cai_log"].as_f64().unwrap() - b["cai_log"].as_f64().unwrap()).abs() < 1e-9,
            "{a} against {b}"
        );
        assert_eq!(a["on_hull"], b["on_hull"]);
    }
}

/// A motif front is a different front, and resuming one without `--motif`
/// would return its points as this walk's.
#[test]
fn a_motif_front_is_refused_as_a_plain_resume() {
    let motif = scratch("motif.jsonl");
    let argv = words(&[
        "--protein",
        PROTEIN,
        "--utr5",
        UTR5,
        "--stop",
        "none",
        "--utr3",
        UTR3,
        "--motif",
        "GCC",
        "--max-secs",
        "0.000001",
        "--out",
        motif.to_str().unwrap(),
    ]);
    front::run(&front::parse(argv).unwrap()).unwrap();
    let argv = words(&[
        "--protein",
        PROTEIN,
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        // The same molecule as the run above, so the refusal below is about
        // the file being a motif front and not about the two stops differing.
        "--stop",
        "none",
        "--resume",
        motif.to_str().unwrap(),
        "--out",
        scratch("refused-motif.jsonl").to_str().unwrap(),
    ]);
    let e = front::run(&front::parse(argv).unwrap())
        .unwrap_err()
        .to_string();
    assert!(e.contains("it is a motif front"), "{e}");
}

/// A resume returns the file's own points beside the ones it folds.
///
/// A budget small enough to stop a walk deterministically stops it before any
/// box settles, so the file it writes carries no point. This one is built from
/// a whole front instead: one box's interior points removed and that box
/// marked open, which is what a walk stopped after other boxes writes. The
/// resume must fold that box and return the points the file kept.
#[test]
fn a_resume_returns_the_points_its_file_carries() {
    let out = scratch("whole-partial.jsonl");
    let whole = front_file(&[], &out);
    let hull: Vec<usize> = (0..whole.points.len())
        .filter(|&i| whole.points[i]["on_hull"] == true)
        .collect();
    // The box to reopen: the first pair of hull vertices with a point between
    // them, which is what only the file can supply for the others.
    let (lo, hi) = hull
        .windows(2)
        .map(|w| (w[0], w[1]))
        .find(|&(a, b)| b > a + 1)
        .expect("a front with a point inside a box");
    let kept: Vec<serde_json::Value> = whole
        .points
        .iter()
        .enumerate()
        .filter(|(i, _)| *i <= lo || *i >= hi)
        .map(|(_, p)| p.clone())
        .collect();
    assert!(kept.len() < whole.points.len());

    let mut header = whole.header.clone();
    let at = |i: usize| {
        (
            whole.points[i]["mfe"].as_f64().unwrap(),
            whole.points[i]["cai_log"].as_f64().unwrap(),
        )
    };
    let (a, b) = (at(lo), at(hi));
    header["complete"] = serde_json::json!(false);
    header["stopped_by"] = serde_json::json!("time");
    header["open"] = serde_json::json!([{
        "mfe_lo": a.0, "cai_log_lo": a.1, "mfe_hi": b.0, "cai_log_hi": b.1,
    }]);
    let partial = scratch("partial-built.jsonl");
    let mut text = header.to_string();
    for p in &kept {
        text.push('\n');
        text.push_str(&p.to_string());
    }
    text.push('\n');
    std::fs::write(&partial, text).unwrap();

    let resumed = front_file(
        &["--resume", partial.to_str().unwrap()],
        &scratch("resumed-built.jsonl"),
    );
    assert_eq!(resumed.header["resumed"]["points"], kept.len());
    assert_eq!(resumed.points.len(), whole.points.len());
    for (r, w) in resumed.points.iter().zip(&whole.points) {
        assert_eq!(
            (r["mfe"].as_f64().unwrap() * 100.0).round(),
            (w["mfe"].as_f64().unwrap() * 100.0).round()
        );
        assert!((r["cai_log"].as_f64().unwrap() - w["cai_log"].as_f64().unwrap()).abs() < 1e-9);
    }
}

/// `--checkpoint` on the plain front: the run leaves the whole result in
/// `--out` with nothing beside it, and every checkpoint the command writes
/// there is a result that resumes to the whole front.
///
/// The checkpoints are captured by walking under the command's own writer and
/// copying `--out` after each one, since the run's final result replaces the
/// last of them.
#[test]
fn every_checkpoint_of_a_plain_front_resumes_to_the_whole_front() {
    let whole = front_file(&[], &scratch("whole-checkpoint.jsonl"));
    let out = scratch("checkpointed.jsonl");
    let ended = front_file(&["--checkpoint"], &out);
    assert_eq!(ended.header["checkpoint"], false);
    assert_eq!(ended.header["complete"], true);
    let mut beside = out.clone().into_os_string();
    beside.push(".partial");
    assert!(
        !std::path::Path::new(&beside).exists(),
        "{beside:?} is left"
    );
    let same = |a: &[serde_json::Value], b: &[serde_json::Value], label: &str| {
        assert_eq!(a.len(), b.len(), "{label}");
        for (a, b) in a.iter().zip(b) {
            assert_eq!(
                (a["mfe"].as_f64().unwrap() * 100.0).round(),
                (b["mfe"].as_f64().unwrap() * 100.0).round(),
                "{label}"
            );
            assert!(
                (a["cai_log"].as_f64().unwrap() - b["cai_log"].as_f64().unwrap()).abs() < 1e-9,
                "{label}: {a} against {b}"
            );
            assert_eq!(a["on_hull"], b["on_hull"], "{label}");
        }
    };
    same(&ended.points, &whole.points, "the checkpointed run");

    let argv = words(&[
        "--protein",
        PROTEIN,
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--checkpoint",
        // `--out` holds the finished run above, which a rerun without
        // `--force` would refuse; these arguments only build the lines.
        "--force",
        "--out",
        out.to_str().unwrap(),
    ]);
    let args = front::parse(argv).unwrap();
    let m = derna_cli::molecule::Molecule::resolve(&args.shared).unwrap();
    let lines = front::PlainLines {
        args: &args,
        m: &m,
        threads: 2,
        started: std::time::Instant::now(),
        resume: None,
    };
    let files = std::sync::Mutex::new(Vec::new());
    let hand = |so_far: &derna_lib::WalkedFront| {
        lines.replace(&out, so_far, true).unwrap();
        files
            .lock()
            .unwrap()
            .push(std::fs::read_to_string(&out).unwrap());
    };
    let run = derna_lib::WalkRun {
        checkpoint: Some(&hand),
        ..derna_lib::WalkRun::default()
    };
    front::walk_run(&m, 2, Default::default(), run).unwrap();
    let files = files.into_inner().unwrap();

    let mut open = 0;
    for (i, text) in files.iter().enumerate() {
        let copy = scratch(&format!("checkpoint-{i}.jsonl"));
        std::fs::write(&copy, text).unwrap();
        let taken = show::read(Some(&copy)).unwrap();
        assert_eq!(taken.header["checkpoint"], true, "checkpoint {i}");
        let boxes = taken.header["open"].as_array().unwrap().len();
        assert_eq!(taken.header["complete"], boxes == 0, "checkpoint {i}");
        open += usize::from(boxes > 0);
        let resumed = front_file(
            &["--resume", copy.to_str().unwrap()],
            &scratch(&format!("resumed-checkpoint-{i}.jsonl")),
        );
        assert_eq!(resumed.header["complete"], true, "checkpoint {i}");
        same(
            &resumed.points,
            &whole.points,
            &format!("checkpoint {i}, resumed"),
        );
    }
    // A checkpoint with nothing open resumes to itself and tests nothing.
    assert!(
        open >= 2,
        "{open} of {} checkpoint(s) left a box open",
        files.len()
    );
    // The last checkpoint is taken after the last gap settles, so it is the
    // whole front as it stands, with nothing open: one taken before a gap's
    // points are recorded lacks them and still resumes.
    let last = scratch("checkpoint-last.jsonl");
    std::fs::write(&last, files.last().expect("a walk checkpoints")).unwrap();
    let last = show::read(Some(&last)).unwrap();
    assert_eq!(last.header["complete"], true, "the last checkpoint");
    same(&last.points, &whole.points, "the last checkpoint");
}
