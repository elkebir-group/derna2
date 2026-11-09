//! `derna front --modular` end to end: the front of designs that
//! pass their own check, complete without a budget and partial, and saying so,
//! with one.
//!
//! The cell is one the library's exhaustive oracle covers, chosen because its
//! front holds points only a walk finds beside the constrained front's seeds:
//! five points, three of them seeds.
//!
//! Every run here asks for `--stop none`, so the cell stays the molecule those
//! counts were calibrated against. The same cell with its stop codon left to
//! the search is a different molecule — one codon longer — and is covered by
//! `stop_slot.rs`.

use serde_json::Value;

/// Run `derna front --modular` on the cell and read its lines.
///
/// The tests run in one process on several threads and each deletes its own
/// output, so the file is named by a counter rather than by anything about the
/// run: two tests that pass the same flags must not share a path.
fn run(extra: &[&str]) -> Vec<Value> {
    run_cell(("MPRGLA", "CCGCGG"), extra)
}

/// [`run`] on another protein and 5′ flank.
fn run_cell((protein, utr5): (&str, &str), extra: &[&str]) -> Vec<Value> {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let nth = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let out =
        std::env::temp_dir().join(format!("derna-so-front-{}-{nth}.jsonl", std::process::id()));
    let mut argv: Vec<String> = [
        "front",
        "--protein",
        protein,
        "--utr5",
        utr5,
        // The stop is a lattice slot, so a front run that omits --stop DESIGNS one and
        // returns a different front. Every test here pins it explicitly; `none` keeps
        // these fronts comparable to the ones asserted before the slot existed.
        "--stop",
        "none",
        "--modular",
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

#[test]
fn the_whole_front_passes_its_check_and_is_certified() {
    let lines = run(&[]);
    let (header, points) = lines.split_first().unwrap();
    assert_eq!(header["modular_front"], true);
    assert_eq!(header["complete"], true);
    assert_eq!(header["stopped_by"], Value::Null);
    assert_eq!(header["seeds"], 3);
    assert!(header["certify_s"].as_f64().is_some_and(|s| s >= 0.0));
    let stretches = header["stretches"].as_array().unwrap();
    assert!(
        stretches
            .iter()
            .all(|s| s["round"].is_u64() && s["slope"].is_f64()),
        "{stretches:?}"
    );
    assert_eq!(points.len(), 5, "{points:?}");
    for p in points {
        assert_eq!(p["modular"], true, "{p}");
        assert_eq!(p["gap"], 0.0, "{p}");
        assert_eq!(p["status"], "certified", "{p}");
    }
    // A front: energy ascending, adaptation strictly ascending.
    for w in points.windows(2) {
        assert!(w[0]["mfe"].as_f64() <= w[1]["mfe"].as_f64());
        assert!(w[0]["cai_log"].as_f64() < w[1]["cai_log"].as_f64());
    }
}

#[test]
fn a_spent_budget_returns_the_partial_front_and_what_it_left_open() {
    let lines = run(&["--max-secs", "1e-9"]);
    let (header, points) = lines.split_first().unwrap();
    assert_eq!(header["complete"], false);
    assert_eq!(header["stopped_by"], "time");
    assert!(!header["open"].as_array().unwrap().is_empty(), "{header}");
    // The budget governs the seeds' certification too, so a spent one may
    // leave constrained points unchecked; whatever it returns passed its
    // check, and the header says how many were left.
    assert!(header["unchecked"].as_u64().is_some(), "{header}");
    assert!(points.iter().all(|p| p["modular"] == true));
}

/// A constrained front written by a plain run, for a search to read back.
fn constrained_front_file() -> std::path::PathBuf {
    let out = std::env::temp_dir().join(format!("derna-so-walked-{}.jsonl", std::process::id()));
    let argv: Vec<String> = [
        "front",
        "--protein",
        "MPRGLA",
        "--utr5",
        "CCGCGG",
        "--stop",
        "none",
        "--force",
        "--out",
        out.to_str().unwrap(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    derna_cli::run(argv).unwrap();
    out
}

#[test]
fn a_budget_that_covers_the_seeds_returns_every_seed() {
    let walked = constrained_front_file();
    let lines = run(&[
        "--front-from",
        walked.to_str().unwrap(),
        "--max-secs",
        "3600",
    ]);
    std::fs::remove_file(&walked).unwrap();
    let header = &lines[0];
    assert_eq!(header["unchecked"], 0, "{header}");
    assert_eq!(header["seeds"], 3, "{header}");
}

#[test]
fn the_header_counts_the_points_it_is_written_before() {
    // The points are written as they are emitted, so a run an outer limit
    // kills leaves a prefix of them under a header already on disk. What
    // makes that prefix readable rather than ambiguous is this: the header
    // says how many points the front holds, so a reader comparing it with
    // the lines present sees at once whether the file is whole.
    let lines = run(&[]);
    let (header, points) = lines.split_first().unwrap();
    assert_eq!(header["points"], points.len());
}

#[test]
fn the_front_does_not_depend_on_the_thread_count() {
    // Emitting in batches over workers must not change the answer: each line
    // reads nothing but its own point, and the batches are written in the
    // front's order.
    //
    // What a thread count may change is how a *tie* is spelled. Two designs
    // of equal MFE and equal CAI are equally good answers, and which one a
    // walk reaches first depends on the order its workers finish, so the
    // sequence and the structure are compared only where the objectives say
    // the point is not a tie. Asking for both determinism and threads is what
    // `--deterministic` is for, and the test below pins that.
    let one = run(&["--threads", "1"]);
    let many = run(&["--threads", "4"]);
    assert_eq!(one.len(), many.len(), "the front has a different size");

    // A point either came back spelled the same, and then nothing about it
    // may differ, or it is a tie spelled differently — and then everything
    // that is a function of the sequence differs with it, the GC and
    // homopolymer metrics included. What may never differ is the objective
    // the front is a front of, and the verdict the check reached.
    const SAME_FOR_A_TIE: [&str; 10] = [
        "record",
        "mfe",
        "cai",
        "cai_log",
        "modular",
        "status",
        "gap",
        "free_mfe",
        "violations",
        "fixed_violations",
    ];
    let mut ties = 0;
    for (a, b) in one[1..].iter().zip(&many[1..]) {
        if a["sequence"] == b["sequence"] {
            assert_eq!(a, b, "two runs spelled a point the same and differ anyway");
            continue;
        }
        ties += 1;
        for field in SAME_FOR_A_TIE {
            assert_eq!(
                a[field], b[field],
                "a design spelled differently at another thread count is not a \
                 tie: {field} differs"
            );
        }
    }
    // Ties are what makes this test worth running, and they are a race: a
    // run where none occurred has compared two identical fronts and says
    // nothing about the tie-breaking. It is not a failure — the front still
    // has to match, which is the other half — so it is reported, not asserted.
    eprintln!("tied points spelled differently across thread counts: {ties}");
}

/// `--deterministic` is what asks for byte-identical output whatever
/// `--threads` says, and it buys that by folding on one thread.
#[test]
fn deterministic_output_does_not_depend_on_the_thread_count() {
    let one = run(&["--threads", "1", "--deterministic"]);
    let many = run(&["--threads", "4", "--deterministic"]);
    assert_eq!(one[1..], many[1..]);
    assert_eq!(
        many[0]["threads"], 1,
        "a deterministic run records the one thread it folded on"
    );
}

#[test]
fn a_run_with_no_constraint_is_refused() {
    let argv: Vec<String> = ["front", "--protein", "MPRGLA", "--modular"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let e = derna_cli::run(argv).unwrap_err().to_string();
    assert!(e.contains("needs a constraint"), "{e}");
}

#[test]
fn the_report_says_when_each_task_ran_and_what_memory_it_left() {
    // A memory budget is set from which task raises the peak, so the report
    // carries each task's clock and the process's memory when it ended.
    let report = std::env::temp_dir().join(format!("derna-so-report-{}.json", std::process::id()));
    run(&["--report", report.to_str().unwrap()]);
    let r: Value = serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    std::fs::remove_file(&report).unwrap();
    let timeline = r["timeline"].as_array().unwrap();
    assert!(!timeline.is_empty());
    for t in timeline {
        let num = |k: &str| t[k].as_f64().unwrap_or_else(|| panic!("{k} missing: {t}"));
        assert!(num("start_s") <= num("end_s"), "{t}");
        assert!(num("cpu_s") >= 0.0, "{t}");
        assert!(num("workers") >= 1.0, "{t}");
        // The two memory fields are not a pair. The peak comes from
        // `getrusage` and is readable on every Unix; the current size is read
        // from `/proc`, so it is `null` wherever that is absent — which is
        // what `CURRENT_RSS_READABLE` says. Asserting they appear together
        // asserts a Linux shape, and `(null, 34000)` is the right answer on
        // macOS rather than a defect.
        let peak = t["peak_rss_kb"]
            .as_f64()
            .unwrap_or_else(|| panic!("a peak reading on any Unix: {t}"));
        assert!(peak > 0.0, "{t}");
        if derna_lib::task_clock::CURRENT_RSS_READABLE {
            let rss = t["rss_kb"]
                .as_f64()
                .unwrap_or_else(|| panic!("this platform reads /proc: {t}"));
            assert!(rss > 0.0, "{t}");
            assert!(peak >= rss, "the peak is below the current size: {t}");
        } else {
            assert!(t["rss_kb"].is_null(), "no /proc, so no current size: {t}");
        }
    }
}

/// Run the cell with `--checkpoint` into `out`, plus `extra`; the result's lines.
fn checkpointed(out: &std::path::Path, extra: &[&str]) -> Vec<Value> {
    let mut argv: Vec<String> = [
        "front",
        "--protein",
        "MPRGLA",
        "--utr5",
        "CCGCGG",
        "--stop",
        "none",
        "--modular",
        "--checkpoint",
        "--force",
        "--out",
        out.to_str().unwrap(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    argv.extend(extra.iter().map(|s| s.to_string()));
    derna_cli::run(argv).unwrap();
    std::fs::read_to_string(out)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// A point's fields a resumed run must reproduce: all but the molecule and
/// its structures. Two designs can be modular at one energy and
/// one adaptation, and a fold of one can be spelled two ways, so which of
/// several co-optimal answers a run reports depends on the order it met them
/// — which resuming changes. [`every_point_is_a_witness`] asserts that
/// whatever it reports is one.
fn settled(lines: &[Value]) -> Vec<Value> {
    lines[1..]
        .iter()
        .map(|p| {
            let mut p = p.clone();
            p.as_object_mut().unwrap().remove("structure");
            p.as_object_mut().unwrap().remove("free_structure");
            p.as_object_mut().unwrap().remove("sequence");
            p
        })
        .collect()
}

/// Every point carries a molecule that is modular at the energy
/// the point claims: a zero gap against its own free fold, with no pair the
/// table forbids. What [`settled`] drops the sequence for.
fn every_point_is_a_witness(lines: &[Value]) {
    for p in &lines[1..] {
        let at = &p["mfe"];
        assert_eq!(p["modular"], Value::Bool(true), "{p}");
        assert_eq!(p["gap"].as_f64(), Some(0.0), "{p}");
        assert_eq!(p["free_mfe"], *at, "{p}");
        assert_eq!(p["violations"].as_array().map(Vec::len), Some(0), "{p}");
        assert!(p["sequence"].as_str().is_some_and(|s| !s.is_empty()), "{p}");
    }
}

/// `--checkpoint` records the search as it settles, and `--resume-search`
/// from that record — whole, or cut short mid-line as a kill leaves it —
/// returns the front the uninterrupted run did, and a resumed run's own
/// record carries what it resumed, so it alone resumes a second time.
#[test]
fn a_search_resumed_from_its_checkpoint_returns_the_same_front() {
    let dir = std::env::temp_dir();
    let id = std::process::id();
    let first = dir.join(format!("derna-so-ckpt-{id}-a.jsonl"));
    let progress = dir.join(format!("derna-so-ckpt-{id}-a.jsonl.search.jsonl"));
    let _ = std::fs::remove_file(&progress);
    let whole = checkpointed(&first, &[]);
    let log = std::fs::read_to_string(&progress).unwrap();
    // Whole lines only: a line a kill cut short is not progress.
    let count = |text: &str| {
        text.lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter(|v| v.get("checked").is_some())
            .count()
    };
    let checked = count(&log);
    // A design this run checked carries the free fold its verdict was read
    // from, over the whole molecule.
    for v in log
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|v| v.get("checked").is_some())
    {
        let free = v["free_structure"]
            .as_str()
            .unwrap_or_else(|| panic!("{v}"));
        assert_eq!(free.len(), v["checked"].as_str().unwrap().len(), "{v}");
    }
    assert_eq!(
        checked as u64,
        whole[0]["constrained_points"].as_u64().unwrap()
    );
    assert!(log.lines().any(|l| l.contains("\"found\"")), "{log}");

    // Cut mid-line after half its lines, as a kill during a write leaves it.
    let lines: Vec<&str> = log.lines().collect();
    let keep = lines.len() / 2;
    let cut = dir.join(format!("derna-so-ckpt-{id}-cut.search.jsonl"));
    let last = lines[keep];
    std::fs::write(
        &cut,
        format!("{}\n{}", lines[..keep].join("\n"), &last[..last.len() / 2]),
    )
    .unwrap();

    for (k, from) in [&progress, &cut].into_iter().enumerate() {
        let again_out = dir.join(format!("derna-so-ckpt-{id}-r{k}.jsonl"));
        let again_log = dir.join(format!("derna-so-ckpt-{id}-r{k}.jsonl.search.jsonl"));
        let _ = std::fs::remove_file(&again_log);
        let again = checkpointed(&again_out, &["--resume-search", from.to_str().unwrap()]);
        assert_eq!(settled(&again), settled(&whole), "from {}", from.display());
        every_point_is_a_witness(&again);
        let resumed = &again[0]["resumed_search"];
        let had = count(&std::fs::read_to_string(from).unwrap());
        assert_eq!(
            resumed["checked"].as_u64().unwrap() as usize,
            had,
            "{resumed}"
        );
        let carried = std::fs::read_to_string(&again_log).unwrap();
        assert_eq!(count(&carried), checked, "{carried}");
        for f in [&again_out, &again_log] {
            std::fs::remove_file(f).unwrap();
        }
    }
    for f in [&first, &progress, &cut] {
        std::fs::remove_file(f).unwrap();
    }
}

/// With a motif, `--checkpoint` records the modular search beside `--out`
/// as it does without one, and `--resume-search` from that record returns
/// the front the uninterrupted run did, carrying every point it had checked.
#[test]
fn a_motif_search_resumed_from_its_checkpoint_returns_the_same_front() {
    let first = scratch("motif-ckpt");
    let progress = std::path::PathBuf::from(format!("{}.search.jsonl", first.display()));
    let _ = std::fs::remove_file(&progress);
    let whole = checkpointed(&first, &["--motif", "CGC"]);
    let log = std::fs::read_to_string(&progress)
        .unwrap_or_else(|e| panic!("no search record at {}: {e}", progress.display()));
    let count = |text: &str| {
        text.lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter(|v| v.get("checked").is_some())
            .count()
    };
    let checked = count(&log);
    assert!(checked > 0, "{log}");
    assert!(whole[0]["resumed_search"].is_null(), "{}", whole[0]);

    let again_out = scratch("motif-ckpt-resumed");
    let again_log = std::path::PathBuf::from(format!("{}.search.jsonl", again_out.display()));
    let _ = std::fs::remove_file(&again_log);
    let again = checkpointed(
        &again_out,
        &[
            "--motif",
            "CGC",
            "--resume-search",
            progress.to_str().unwrap(),
        ],
    );
    assert_eq!(designs(&again[1..]), designs(&whole[1..]));
    every_point_is_a_witness(&again);
    let resumed = &again[0]["resumed_search"];
    assert_eq!(
        resumed["checked"].as_u64(),
        Some(checked as u64),
        "{resumed}"
    );
    let carried = std::fs::read_to_string(&again_log).unwrap();
    assert_eq!(count(&carried), checked, "{carried}");
    for f in [&first, &progress, &again_out, &again_log] {
        std::fs::remove_file(f).unwrap();
    }
}

/// The header names the search's schedule: the regions a round walks side by
/// side and the first round's span, so a stopped front can be read against
/// the span its regions were given. These are the defaults, which the
/// `ModularLanes` and `ModularFirstRoundS` development settings override.
///
/// The settings are read through the library's reader rather than by name, so
/// this file names no environment variable: a build that reads no development
/// setting answers both as unset, which is when the defaults are what the
/// header must carry.
#[test]
fn the_header_names_the_search_schedule() {
    use derna_lib::dev_env::{var_os, Setting};
    if var_os(Setting::ModularLanes).is_some() || var_os(Setting::ModularFirstRoundS).is_some() {
        return;
    }
    let lines = run(&["--threads", "4"]);
    assert_eq!(lines[0]["lanes"].as_u64(), Some(2), "{}", lines[0]);
    assert_eq!(
        lines[0]["first_round_s"].as_f64(),
        Some(30.0),
        "{}",
        lines[0]
    );
}

#[test]
fn resume_search_needs_the_modular_front() {
    let argv: Vec<String> = [
        "front",
        "--protein",
        "MPRGLA",
        "--resume-search",
        "x.search.jsonl",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let e = derna_cli::run(argv).unwrap_err().to_string();
    assert!(e.contains("--resume-search"), "{e}");
}

/// Anchors checked from another result of the molecule — here its own
/// modular front, every point of which passes — seed the search
/// without changing the front it returns; another molecule's are refused.
#[test]
fn anchors_seed_the_search_without_changing_its_front() {
    let plain = run(&[]);
    let anchors =
        std::env::temp_dir().join(format!("derna-so-anchors-{}.jsonl", std::process::id()));
    let text: Vec<String> = plain.iter().map(Value::to_string).collect();
    std::fs::write(&anchors, text.join("\n") + "\n").unwrap();
    let seeded = run(&["--anchors-from", anchors.to_str().unwrap()]);
    let header = &seeded[0]["anchors"];
    assert_eq!(header["points"].as_u64(), Some((plain.len() - 1) as u64));
    assert_eq!(header["modular"], header["points"], "{header}");
    let strip = |lines: &[Value]| -> Vec<Value> {
        lines[1..]
            .iter()
            .map(|p| {
                let mut p = p.clone();
                p.as_object_mut().unwrap().remove("structure");
                p.as_object_mut().unwrap().remove("free_structure");
                p
            })
            .collect()
    };
    assert_eq!(strip(&seeded), strip(&plain));

    let argv: Vec<String> = [
        "front",
        "--protein",
        "MPRGLV",
        "--utr5",
        "CCGCGG",
        "--stop",
        "none",
        "--modular",
        "--anchors-from",
        anchors.to_str().unwrap(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let e = derna_cli::run(argv).unwrap_err().to_string();
    assert!(e.contains("another molecule"), "{e}");
    std::fs::remove_file(&anchors).unwrap();
}

#[test]
fn with_a_motif_the_front_is_over_the_motif_free_designs() {
    let lines = run(&["--motif", "CGC"]);
    let (header, points) = lines.split_first().unwrap();
    assert_eq!(header["modular_front"], true);
    assert_eq!(header["complete"], true);
    assert_eq!(header["motifs"], serde_json::json!(["CGC"]));
    assert!(!points.is_empty());
    let (lo, hi) = ("CCGCGG".len(), "CCGCGG".len() + 3 * "MPRGLA".len());
    for p in points {
        assert_eq!(p["modular"], true, "{p}");
        assert_eq!(p["status"], "certified", "{p}");
        let sequence = p["sequence"].as_str().unwrap();
        let spelled = (0..=sequence.len() - 3)
            .filter(|&s| sequence[s..].starts_with("CGC"))
            .any(|s| s < hi && s + 3 > lo);
        assert!(!spelled, "{sequence} spells the motif");
    }
    // The motif removes points of the plain modular front.
    let plain = run(&[]);
    let at =
        |ps: &[Value]| -> Vec<String> { ps.iter().map(|p| p["sequence"].to_string()).collect() };
    assert_ne!(at(points), at(&plain[1..]));
}

/// Leucine's third base to U leaves it only CUU, so the pin takes its most
/// adapted codon.
#[test]
fn with_a_motif_and_a_pin_every_point_spells_the_pin() {
    let lines = run(&["--motif", "CGC", "--pin", "15:U"]);
    let (header, points) = lines.split_first().unwrap();
    assert_eq!(header["modular_front"], true);
    assert_eq!(header["complete"], true);
    assert_eq!(header["pins"], serde_json::json!(["15:U"]));
    assert!(!points.is_empty());
    let at = "CCGCGG".len() + 14;
    for p in points {
        assert_eq!(p["status"], "certified", "{p}");
        assert_eq!(&p["sequence"].as_str().unwrap()[at..=at], "U", "{p}");
    }
    // The pin binds: the most adapted pinned point is less adapted.
    let best = |ps: &[Value]| {
        ps.iter()
            .map(|p| p["cai_log"].as_f64().unwrap())
            .fold(f64::MIN, f64::max)
    };
    let unpinned = run(&["--motif", "CGC"]);
    assert!(best(points) < best(&unpinned[1..]) - 1e-3);
}

/// A path for a test's own file, named by a counter so tests on other threads
/// cannot share it.
fn scratch(stem: &str) -> std::path::PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let nth = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "derna-so-{stem}-{}-{nth}.jsonl",
        std::process::id()
    ))
}

fn read_lines(path: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// The points' objectives, energy in integer dekacalories, and statuses,
/// which a resumed or checkpointed run must return as an uninterrupted one
/// does. A resumed walk may return a different design of the same objectives.
fn designs(points: &[Value]) -> Vec<(i64, String, String)> {
    points
        .iter()
        .map(|p| {
            (
                (p["mfe"].as_f64().unwrap() * 100.0).round() as i64,
                format!("{:.9}", p["cai_log"].as_f64().unwrap()),
                p["status"].to_string(),
            )
        })
        .collect()
}

#[test]
fn a_stopped_motif_front_resumes_to_the_same_front() {
    let stopped = scratch("stopped");
    let argv: Vec<String> = [
        "front",
        "--protein",
        "MPRGLA",
        "--utr5",
        "CCGCGG",
        "--stop",
        "none",
        "--motif",
        "CGC",
        "--max-secs",
        "0.000001",
        "--force",
        "--out",
    ]
    .iter()
    .map(|s| s.to_string())
    .chain([stopped.to_str().unwrap().to_string()])
    .collect();
    derna_cli::run(argv).unwrap();
    let partial = read_lines(&stopped);
    assert_eq!(partial[0]["complete"], false, "{}", partial[0]);

    let resumed = run(&["--motif", "CGC", "--resume", stopped.to_str().unwrap()]);
    let whole = run(&["--motif", "CGC"]);
    assert_eq!(resumed[0]["complete"], true);
    assert_eq!(resumed[0]["resumed"]["points"], partial.len() - 1);
    assert_eq!(designs(&resumed[1..]), designs(&whole[1..]));
    std::fs::remove_file(&stopped).unwrap();
}

/// A search from a stopped motif front, walking none of what it left open,
/// returns the whole front's modular points: the open boxes floor
/// the search in place of the points they hold.
#[test]
fn a_search_from_a_stopped_motif_front_is_the_whole_search() {
    let stopped = scratch("stopped-search");
    let argv: Vec<String> = [
        "front",
        "--protein",
        "MPRGLA",
        "--utr5",
        "CCGCGG",
        "--stop",
        "none",
        "--motif",
        "CGC",
        "--max-secs",
        "0.000001",
        "--force",
        "--out",
    ]
    .iter()
    .map(|s| s.to_string())
    .chain([stopped.to_str().unwrap().to_string()])
    .collect();
    derna_cli::run(argv).unwrap();
    assert_eq!(read_lines(&stopped)[0]["complete"], false);
    let searched = run(&[
        "--motif",
        "CGC",
        "--walk-secs",
        "0",
        "--resume",
        stopped.to_str().unwrap(),
    ]);
    let whole = run(&["--motif", "CGC"]);
    assert_eq!(searched[0]["complete"], true, "{}", searched[0]);
    assert_eq!(designs(&searched[1..]), designs(&whole[1..]));
    std::fs::remove_file(&stopped).unwrap();
}

/// A search read from a plain front its budget stopped returns the whole
/// search's modular points: the open boxes floor the search in
/// place of the points they hold. A plain `front` refuses the same file, which
/// it could only write out again without what its walk solved in each box.
#[test]
fn a_search_from_a_stopped_plain_front_file_is_the_whole_search() {
    let stopped = scratch("plain-stopped");
    let argv: Vec<String> = [
        "front",
        "--protein",
        "MPRGLA",
        "--utr5",
        "CCGCGG",
        "--stop",
        "none",
        "--max-secs",
        "0.000001",
        "--force",
        "--out",
    ]
    .iter()
    .map(|s| s.to_string())
    .chain([stopped.to_str().unwrap().to_string()])
    .collect();
    derna_cli::run(argv).unwrap();
    let partial = read_lines(&stopped);
    assert_eq!(partial[0]["complete"], false, "{}", partial[0]);
    let open = partial[0]["open"].as_array().map_or(0, Vec::len);
    assert!(open > 0, "{}", partial[0]);

    let searched = run(&["--front-from", stopped.to_str().unwrap()]);
    let whole = run(&[]);
    assert_eq!(searched[0]["complete"], true, "{}", searched[0]);
    assert_eq!(searched[0]["constrained_open"], open);
    assert_eq!(designs(&searched[1..]), designs(&whole[1..]));

    let plain: Vec<String> = [
        "front",
        "--protein",
        "MPRGLA",
        "--utr5",
        "CCGCGG",
        "--stop",
        "none",
        "--front-from",
    ]
    .iter()
    .map(|s| s.to_string())
    .chain([stopped.to_str().unwrap().to_string()])
    .collect();
    let e = derna_cli::run(plain).unwrap_err().to_string();
    assert!(e.contains("--resume continues it"), "{e}");
    std::fs::remove_file(&stopped).unwrap();
}

/// A search read from a motif front written earlier, stopped or complete,
/// returns the whole search's modular points, and says which file
/// it read.
#[test]
fn a_search_from_a_motif_front_file_is_the_whole_search() {
    let (stopped, complete) = (scratch("from-stopped"), scratch("from-complete"));
    for (path, extra) in [
        (&stopped, &["--max-secs", "0.000001"][..]),
        (&complete, &[][..]),
    ] {
        let mut argv: Vec<String> = [
            "front",
            "--protein",
            "MPRGLA",
            "--utr5",
            "CCGCGG",
            "--stop",
            "none",
            "--motif",
            "CGC",
            "--force",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        argv.extend(extra.iter().map(|s| s.to_string()));
        argv.extend(["--out".to_string(), path.to_str().unwrap().to_string()]);
        derna_cli::run(argv).unwrap();
    }
    assert_eq!(read_lines(&stopped)[0]["complete"], false);
    assert_eq!(read_lines(&complete)[0]["complete"], true);
    let whole = run(&["--motif", "CGC"]);
    for path in [&stopped, &complete] {
        let searched = run(&["--motif", "CGC", "--front-from", path.to_str().unwrap()]);
        assert_eq!(searched[0]["complete"], true, "{}", searched[0]);
        assert_eq!(searched[0]["front_from"], path.to_str().unwrap());
        assert_eq!(designs(&searched[1..]), designs(&whole[1..]));
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn a_motif_front_file_without_a_modular_run_is_refused() {
    let argv: Vec<String> = [
        "front",
        "--protein",
        "MPRGLA",
        "--motif",
        "CGC",
        "--front-from",
        "motif.jsonl",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let e = derna_cli::run(argv).unwrap_err().to_string();
    assert!(e.contains("--modular"), "{e}");
}

#[test]
fn walk_seconds_without_a_modular_motif_run_are_refused() {
    for extra in [&["--motif", "CGC"][..], &["--modular"]] {
        let mut argv: Vec<String> = ["front", "--protein", "MPRGLA"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        argv.extend(extra.iter().map(|s| s.to_string()));
        argv.extend(["--walk-secs", "5"].map(String::from));
        let e = derna_cli::run(argv).unwrap_err().to_string();
        assert!(e.contains("--walk-secs"), "{e}");
    }
}

#[test]
fn a_checkpointed_run_writes_the_motif_front_to_the_front_out() {
    let (checkpointed, plain) = (scratch("checkpointed"), scratch("motif"));
    let lines = run(&[
        "--motif",
        "CGC",
        "--checkpoint",
        "--front-out",
        checkpointed.to_str().unwrap(),
    ]);
    assert_eq!(
        designs(&lines[1..]),
        designs(&run(&["--motif", "CGC"])[1..])
    );
    let argv: Vec<String> = [
        "front",
        "--protein",
        "MPRGLA",
        "--utr5",
        "CCGCGG",
        "--stop",
        "none",
        "--motif",
        "CGC",
        "--force",
        "--out",
    ]
    .iter()
    .map(|s| s.to_string())
    .chain([plain.to_str().unwrap().to_string()])
    .collect();
    derna_cli::run(argv).unwrap();
    let (saved, walked) = (read_lines(&checkpointed), read_lines(&plain));
    assert_eq!(saved[0]["complete"], true);
    assert!(
        saved[0].get("checkpoint").is_none_or(|c| c == false),
        "{}",
        saved[0]
    );
    let mfe = |ps: &[Value]| -> Vec<String> { ps.iter().map(|p| p["mfe"].to_string()).collect() };
    assert_eq!(mfe(&saved[1..]), mfe(&walked[1..]));
    std::fs::remove_file(&checkpointed).unwrap();
    std::fs::remove_file(&plain).unwrap();
}

/// The header with what a run's timing, memory and scheduling decide removed:
/// two runs of one search agree on everything else.
///
/// A window walk's check runs on its own workers even at one thread, and the
/// designs they certify raise the floor the walk prunes against, so how far
/// the walk gets before the window is covered depends on when those finds
/// land. Its counters vary between runs; the points it returns do not.
/// `covered`, `bound_requeued` and the window bound's `outside_tested`,
/// `cut_outside`, `covered_tested` and `cut_covered` read that floor too and
/// are seen to differ between runs at one thread (7 and 8, 112 and 113,
/// 18,691 and 39,537 on this cell under a loaded host). The energy bound's
/// counts are zero on this cell, which ends in its first round, and are
/// compared, as is `outside`, which is zero unless a design sits within the
/// window bound's slack of its edge.
fn untimed(header: &Value) -> Value {
    let mut h = header.clone();
    let o = h.as_object_mut().unwrap();
    o.remove("peak_rss_kb");
    for s in o["stretches"].as_array_mut().unwrap() {
        let s = s.as_object_mut().unwrap();
        s.retain(|k, _| {
            ![
                "derivations",
                "reached",
                "covered",
                "offered",
                "folds",
                "bound_requeued",
                "outside_tested",
                "cut_outside",
                "covered_tested",
                "cut_covered",
            ]
            .contains(&k.as_str())
        });
        // A stretch's diagnostic blocks are nested objects, and the retain
        // above tests a stretch's own key names: `phases` holds per-phase wall
        // clocks and `memory` holds allocation high-water marks, and neither
        // name ends in `_s` or appears in the list, so every quantity inside
        // them reached the comparison. Both vary run to run — the header says
        // `deterministic: false` — which made this equality an equality over
        // two walks' instrumentation.
        //
        // Dropped by **shape** rather than by name: a stretch's nested objects
        // are instrumentation and its scalars are the answer, so a block added
        // later cannot put a nondeterministic quantity back in. What is
        // compared is every scalar a stretch carries — its region bounds,
        // point and design counts, slope, stop reason, thread count and
        // completeness.
        s.retain(|_, v| !v.is_object());
    }
    drop_seconds(&mut h);
    h
}

/// Remove every key ending in `_s`, at any depth: a timing nested in an
/// object a run adds later is still a timing.
fn drop_seconds(v: &mut Value) {
    match v {
        Value::Object(o) => {
            o.retain(|k, _| !k.ends_with("_s"));
            o.values_mut().for_each(drop_seconds);
        }
        Value::Array(a) => a.iter_mut().for_each(drop_seconds),
        _ => {}
    }
}

/// `--epsilon 0` is the run without it, line for line: the same header — its
/// counts, regions and stretches — and the same points.
#[test]
fn epsilon_zero_returns_what_a_run_without_it_returns() {
    let plain = run(&["--threads", "1"]);
    let zero = run(&["--threads", "1", "--epsilon", "0"]);
    assert_eq!(untimed(&plain[0]), untimed(&zero[0]));
    assert_eq!(plain[1..], zero[1..]);
    assert_eq!(zero[0]["epsilon"], 0.0);
    assert_eq!(zero[0]["outcome"], "tight");
    assert!(zero[1..].iter().all(|p| p["epsilon"] == 0.0));
}

/// The walk counters of every stretch, by name.
fn walk_counters(s: &Value) -> [u64; 6] {
    [
        "reached",
        "outside",
        "covered",
        "refused",
        "offered",
        "bound_requeued",
    ]
    .map(|k| {
        s[k].as_u64()
            .unwrap_or_else(|| panic!("stretch has no `{k}` count: {s}"))
    })
}

/// A modular front written before stretches carried the walk's counters
/// (`outside`, `covered`, `bound_requeued`) and the bound's cuts by reason
/// still reads: `show` and `check` take it, and it seeds
/// a search. What `show` reads back leaves the counters absent rather than
/// filling them in, so a reader can tell "not recorded" from zero.
#[test]
fn a_front_whose_stretches_predate_the_walk_counters_still_reads() {
    let mut lines = run(&[]);
    let new = [
        "outside",
        "covered",
        "bound_requeued",
        "outside_tested",
        "cut_outside",
        "covered_tested",
        "cut_covered",
        "energy_bound",
        "energy_tested",
        "energy_cut",
        "nodes_closed",
    ];
    for s in lines[0]["stretches"].as_array_mut().unwrap() {
        let s = s.as_object_mut().unwrap();
        for k in new {
            assert!(s.remove(k).is_some(), "the run wrote no `{k}`");
        }
    }
    let dir = std::env::temp_dir().join(format!("derna-so-old-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let old = dir.join("old.jsonl");
    let text: Vec<String> = lines.iter().map(Value::to_string).collect();
    std::fs::write(&old, text.join("\n") + "\n").unwrap();

    let read = derna_cli::show::read(Some(&old)).unwrap();
    for s in read.header["stretches"].as_array().unwrap() {
        for k in new {
            assert!(s.get(k).is_none(), "`{k}` read back as {}", s[k]);
        }
    }
    derna_cli::show::render(
        &derna_cli::show::parse(Vec::<String>::new()).unwrap(),
        &read,
    );

    let checked = dir.join("checked.jsonl");
    let argv: Vec<String> = [
        "check",
        old.to_str().unwrap(),
        "--out",
        checked.to_str().unwrap(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    derna_cli::run(argv).unwrap();

    let seeded = run(&["--anchors-from", old.to_str().unwrap()]);
    assert_eq!(
        seeded[0]["anchors"]["points"].as_u64(),
        Some((lines.len() - 1) as u64)
    );
    for f in [&old, &checked] {
        std::fs::remove_file(f).unwrap();
    }
    std::fs::remove_dir(&dir).unwrap();
}

/// Every walked stretch says where the whole designs its walk reached went,
/// under the keys a reader looks them up by, whether its walk completed or was
/// stopped: `outside`, `covered`, `refused` and `offered` partition `reached`,
/// less the one design that completed the window's coverage and ended the walk.
/// What the bound cut before it was a design is not in `reached` at all.
#[test]
fn every_walked_stretch_says_where_its_designs_went() {
    let sorted = |lines: &[Value]| -> Vec<(String, [u64; 6])> {
        let mut out = Vec::new();
        for s in lines[0]["stretches"].as_array().unwrap() {
            let c = walk_counters(s);
            let [reached, outside, covered, refused, offered, _] = c;
            if s["walked"] == false {
                assert_eq!(c, [0; 6], "{s}");
                continue;
            }
            let landed = outside + covered + refused + offered;
            assert!(
                reached == landed || (reached == landed + 1 && s["stopped"] == "visitor"),
                "{s}"
            );
            out.push((s["stopped"].as_str().unwrap().to_string(), c));
        }
        out
    };
    let whole = sorted(&run(&["--threads", "1"]));
    assert!(whole.iter().any(|(_, c)| c[0] > 0), "{whole:?}");
    let capped = sorted(&run(&["--threads", "1", "--max-derivations", "20"]));
    assert!(
        capped
            .iter()
            .any(|(stopped, c)| stopped == "derivations" && c[0] > 0),
        "no stopped walk reached a design: {capped:?}"
    );
}

/// The bound's counters, each beside the count that arrived at its test, and
/// each non-zero here, so none is satisfied by a zero.
///
/// The cell and cap were chosen by measuring: of five small cells, each run
/// uncapped and at caps of 200, 50 and 20 derivations, this is the cheapest
/// where all nine move. Uncapped, or at 200 or 50, the search ends in its
/// first round, which runs without the energy bound, so `energy_tested`,
/// `energy_cut` and `nodes_closed` read zero; a cap of 20 leaves regions open
/// for a second round, which runs it. The counts held across ten runs. The
/// `outside` counter is planted by the library's window test,
/// `outside_counts_the_designs_reached_beyond_the_window`.
#[test]
fn every_bound_counter_fires_beside_what_reached_its_test() {
    let lines = run(&["--threads", "1", "--max-derivations", "20"]);
    let stretches = lines[0]["stretches"].as_array().unwrap();
    let n = |s: &Value, k: &str| -> u64 {
        s[k].as_u64()
            .unwrap_or_else(|| panic!("stretch has no `{k}` count: {s}"))
    };
    for s in stretches {
        assert_eq!(
            n(s, "covered_tested"),
            n(s, "outside_tested") - n(s, "cut_outside"),
            "{s}"
        );
        assert!(
            n(s, "energy_tested") <= n(s, "covered_tested") - n(s, "cut_covered"),
            "{s}"
        );
        assert!(n(s, "energy_cut") <= n(s, "energy_tested"), "{s}");
    }
    for k in [
        "covered",
        "bound_requeued",
        "outside_tested",
        "cut_outside",
        "covered_tested",
        "cut_covered",
        "energy_tested",
        "energy_cut",
        "nodes_closed",
    ] {
        let sum: u64 = stretches.iter().map(|s| n(s, k)).sum();
        assert!(sum > 0, "`{k}` is zero on every stretch: {stretches:?}");
    }
    // The energy bound is in force on the rounds that run it and no other, and
    // its counters are zero wherever it was not.
    let mut by_flag = [false; 2];
    for s in stretches.iter().filter(|s| s["walked"] == true) {
        let on = s["energy_bound"].as_bool().expect("an `energy_bound` flag");
        assert_eq!(on, n(s, "round") % 2 == 1, "{s}");
        if !on {
            assert_eq!(
                [
                    n(s, "energy_tested"),
                    n(s, "energy_cut"),
                    n(s, "nodes_closed")
                ],
                [0; 3],
                "{s}"
            );
        }
        by_flag[usize::from(on)] = true;
    }
    assert_eq!(by_flag, [true; 2], "{stretches:?}");
}

/// Within an epsilon every point says so and has a gap at most it, and both
/// sides of the test fire on this cell: a point with a gap strictly between
/// zero and 0.5 is returned, and a run within 2.5 returns a design whose gap
/// is above 0.5, which the run within 0.5 does not.
#[test]
fn a_run_within_epsilon_admits_gaps_up_to_it_and_none_above() {
    let within = run(&["--epsilon", "0.5"]);
    let (header, points) = within.split_first().unwrap();
    assert_eq!(header["epsilon"], 0.5, "{header}");
    assert_eq!(header["outcome"], "within_epsilon", "{header}");
    assert_eq!(header["complete"], true, "{header}");
    let gap = |p: &Value| p["gap"].as_f64().unwrap();
    for p in points {
        assert_eq!(p["epsilon"], 0.5, "{p}");
        assert!(gap(p) <= 0.5 + 1e-9, "{p}");
        assert_eq!(p["modular"], gap(p) == 0.0, "{p}");
        assert_eq!(p["status"], "certified", "{p}");
        let violations = p["violations"].as_array().unwrap();
        for pair in p["fixed_violations"].as_array().unwrap() {
            assert!(violations.contains(pair), "{p}");
        }
        // A gap above zero means the free fold uses a forbidden pair; a zero
        // gap may still come with a co-optimal one, so only one way holds.
        assert!(gap(p) == 0.0 || !violations.is_empty(), "{p}");
    }
    assert!(
        points.iter().any(|p| gap(p) > 0.0),
        "nothing admitted: {points:?}"
    );
    let wide = run(&["--epsilon", "2.5"]);
    let above: Vec<&Value> = wide[1..].iter().filter(|p| gap(p) > 0.5).collect();
    assert!(!above.is_empty(), "nothing above 0.5 to reject");
    for p in above {
        assert!(
            points.iter().all(|q| q["sequence"] != p["sequence"]),
            "{p} is above 0.5"
        );
    }
}

/// The header names which answer an empty or a relaxed front is, on a cell
/// with no modular design and no gap below 0.70.
#[test]
fn the_header_says_whether_nothing_was_tight_or_nothing_was_within_epsilon() {
    let cell = ("MRPGAL", "GGGCGCCCG");
    let outcome = |lines: &[Value]| lines[0]["outcome"].as_str().unwrap().to_string();
    let exact = run_cell(cell, &[]);
    assert_eq!((outcome(&exact), exact.len()), ("none_tight".into(), 1));
    let narrow = run_cell(cell, &["--epsilon", "0.5"]);
    assert_eq!(
        (outcome(&narrow), narrow.len()),
        ("none_within_epsilon".into(), 1)
    );
    assert_eq!(narrow[0]["complete"], true);
    let wide = run_cell(cell, &["--epsilon", "1"]);
    assert_eq!(outcome(&wide), "within_epsilon");
    assert!(wide.len() > 1);
    assert!(wide[1..].iter().all(|p| p["modular"] == false));
}

/// `--epsilon` takes a gap of zero or more, once, and only for the
/// modular search.
#[test]
fn epsilon_is_refused_where_it_cannot_act() {
    let refused = |extra: &[&str]| {
        let mut argv: Vec<String> = ["front", "--protein", "MPRGLA", "--utr5", "CCGCGG"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        argv.extend(extra.iter().map(|s| s.to_string()));
        derna_cli::run(argv).unwrap_err().to_string()
    };
    assert!(refused(&["--epsilon", "0.1"]).contains("--modular"));
    let motif = refused(&["--modular", "--motif", "CGC", "--epsilon", "0.1"]);
    assert!(
        motif.contains("--epsilon") && motif.contains("--motif"),
        "{motif}"
    );
    for bad in ["-0.1", "inf", "NaN"] {
        let e = refused(&["--modular", "--epsilon", bad]);
        assert!(e.contains("--epsilon"), "{bad}: {e}");
    }
    let twice = refused(&["--modular", "--epsilon", "0.1", "--epsilon", "0.2"]);
    assert!(twice.contains("twice"), "{twice}");
}

/// `--anchors-from` seeds the modular search over every design,
/// and the search over motif-free designs reads no anchors, so the flag is
/// refused rather than accepted and ignored.
#[test]
fn anchors_from_is_refused_where_it_cannot_act() {
    let refused = |extra: &[&str]| {
        let mut argv: Vec<String> = ["front", "--protein", "MPRGLA", "--utr5", "CCGCGG"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        argv.extend(extra.iter().map(|s| s.to_string()));
        derna_cli::run(argv).unwrap_err().to_string()
    };
    let anchors = std::env::temp_dir().join("derna-anchors-refused.jsonl");
    let path = anchors.to_str().unwrap();
    assert!(refused(&["--anchors-from", path]).contains("--modular"));
    let motif = refused(&["--modular", "--motif", "CGC", "--anchors-from", path]);
    assert!(
        motif.contains("--anchors-from") && motif.contains("--motif"),
        "{motif}"
    );
}

/// A search checkpointed within one epsilon is not resumed within another,
/// and its progress lines carry the epsilon they were settled within.
#[test]
fn a_search_is_not_resumed_within_another_epsilon() {
    let dir = std::env::temp_dir();
    let id = std::process::id();
    let out = dir.join(format!("derna-so-eps-{id}.jsonl"));
    let progress = dir.join(format!("derna-so-eps-{id}.jsonl.search.jsonl"));
    let _ = std::fs::remove_file(&progress);
    checkpointed(&out, &["--epsilon", "0.5"]);
    let log = std::fs::read_to_string(&progress).unwrap();
    assert!(
        log.lines()
            .all(|l| serde_json::from_str::<Value>(l).unwrap()["epsilon"] == 0.5),
        "{log}"
    );
    let argv: Vec<String> = [
        "front",
        "--protein",
        "MPRGLA",
        "--utr5",
        "CCGCGG",
        // The molecule the progress was written for: its log names it, and a
        // log of another molecule is refused before its epsilon is read.
        "--stop",
        "none",
        "--modular",
        "--force",
        "--out",
        out.to_str().unwrap(),
        "--resume-search",
        progress.to_str().unwrap(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let e = derna_cli::run(argv).unwrap_err().to_string();
    assert!(e.contains("--epsilon 0.5"), "{e}");
    let again = checkpointed(
        &dir.join(format!("derna-so-eps-{id}-r.jsonl")),
        &[
            "--epsilon",
            "0.5",
            "--resume-search",
            progress.to_str().unwrap(),
        ],
    );
    assert_eq!(again[0]["epsilon"], 0.5);
    for f in [
        out.clone(),
        progress.clone(),
        dir.join(format!("derna-so-eps-{id}-r.jsonl")),
        dir.join(format!("derna-so-eps-{id}-r.jsonl.search.jsonl")),
    ] {
        let _ = std::fs::remove_file(f);
    }
}

/// A front within an epsilon goes through `check` and `show` saying so:
/// `check` carries the epsilon and decides each point's fixed violations
/// again from its own fold, and `show` prints the epsilon and how many
/// points sit at each gap.
#[test]
fn a_front_within_epsilon_reads_as_one_through_check_and_show() {
    let dir = std::env::temp_dir().join(format!("derna-so-eps-check-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (front, checked) = (dir.join("front.jsonl"), dir.join("checked.jsonl"));
    let argv = |words: &[&str]| words.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    derna_cli::run(argv(&[
        "front",
        "--protein",
        "MPRGLA",
        "--utr5",
        "CCGCGG",
        "--modular",
        "--epsilon",
        "0.5",
        "--out",
        front.to_str().unwrap(),
    ]))
    .unwrap();
    derna_cli::run(argv(&[
        "check",
        front.to_str().unwrap(),
        "--out",
        checked.to_str().unwrap(),
    ]))
    .unwrap();
    let lines = |path: &std::path::Path| -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    };
    let (before, after) = (lines(&front), lines(&checked));
    assert_eq!(after[0]["epsilon"], 0.5, "{}", after[0]);
    assert_eq!(before.len(), after.len());
    for (b, a) in before[1..].iter().zip(&after[1..]) {
        assert_eq!(a["epsilon"], 0.5, "{a}");
        // The check folds the same sequence with the same model.
        assert_eq!(a["fixed_violations"], b["fixed_violations"], "{a}");
        let violations = a["violations"].as_array().unwrap();
        for pair in a["fixed_violations"].as_array().unwrap() {
            assert!(violations.contains(pair), "{a}");
        }
    }
    let shown = derna_cli::show::render(
        &derna_cli::show::parse(Vec::<String>::new()).unwrap(),
        &derna_cli::show::read(Some(&checked)).unwrap(),
    )
    .join("\n");
    assert!(shown.contains("within 0.5 kcal/mol, by gap: "), "{shown}");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// `--top-search none` stops the search below the band above the best modular
/// design it holds, and the header says so: `top_band` names the band, and it
/// is what tells a capped front from an uncapped one. What it returns holds
/// every point the searched run has at or below the ceiling and none above
/// it; a point it holds that the searched run does not is a modular design
/// the band's own designs dominate, which is the half of the guarantee that
/// does not point the same way.
///
/// Non-vacuity: the same cell with the band searched walks a region this run
/// does not, so the ceiling takes a region away and the comparison is not
/// between two copies of one answer.
#[test]
fn stopping_below_the_top_band_says_so_and_keeps_the_front_under_it() {
    let whole = run(&[]);
    let capped = run(&["--top-search", "none"]);
    assert_eq!(whole[0]["top_band"], Value::Null, "{}", whole[0]);
    let band = &capped[0]["top_band"];
    let ceiling = band["cai_log_lo"].as_f64().unwrap();
    assert_eq!(
        band["cai_log_hi"],
        Value::Null,
        "the band runs to the lattice's own top"
    );
    assert_eq!(capped[0]["complete"], true, "{}", capped[0]);
    assert!(capped[0]["open"].as_array().unwrap().is_empty());
    type Point = (f64, f64);
    let axes =
        |p: &Value| -> Point { (p["mfe"].as_f64().unwrap(), p["cai_log"].as_f64().unwrap()) };
    let kept: Vec<Point> = capped[1..].iter().map(axes).collect();
    let searched: Vec<Point> = whole[1..].iter().map(axes).collect();
    let under: Vec<Point> = searched
        .iter()
        .copied()
        .filter(|p| p.1 <= ceiling + 1e-9)
        .collect();
    let over: Vec<Point> = searched
        .iter()
        .copied()
        .filter(|p| p.1 > ceiling + 1e-9)
        .collect();
    assert!(
        kept.iter().all(|p| p.1 <= ceiling + 1e-9),
        "a point above the ceiling: {kept:?}"
    );
    for p in &under {
        assert!(kept.contains(p), "{p:?} is missing from the capped front");
    }
    for p in &kept {
        assert!(
            under.contains(p)
                || over
                    .iter()
                    .any(|q| q.0 <= p.0 && q.1 >= p.1 && (q.0 < p.0 || q.1 > p.1)),
            "{p:?} is neither on the searched run's front nor dominated from the band"
        );
    }
    // Non-vacuity: the band is a region the searched run walks and this one
    // does not, so the ceiling removed a region rather than renaming one. A
    // walked stretch with no upper end is that region.
    let open_ended = |lines: &[Value]| {
        lines[0]["stretches"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|s| s["walked"] == true && s["cai_log_hi"] == Value::Null)
            .count()
    };
    assert!(
        open_ended(&whole) > 0 && open_ended(&capped) == 0,
        "the searched run walked {} open-ended stretch(es) and the capped one {}",
        open_ended(&whole),
        open_ended(&capped)
    );
    for p in &capped[1..] {
        assert_eq!(p["modular"], true, "{p}");
        assert_eq!(p["gap"], 0.0, "{p}");
    }
}
