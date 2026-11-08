//! `--checkpoint` means one thing on `design` and `front`: progress is saved
//! beside `--out` as it settles, and running the same command again continues
//! from it. `--force` starts over and removes the progress first.
//!
//! A resumed run that recomputed everything would also return the
//! uninterrupted run's answer, so equality alone cannot show that anything was
//! resumed. Each resume here is shown on a planted unit instead: a progress
//! line whose value the computation would never produce, which a resumed run
//! must carry into its result and a forced run must not.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

const PROTEIN: &str = "MKVLAKGIVLAGQWERS";
const UTR5: &str = "ACAUUUGCUU";
const UTR3: &str = "GGCCAAUUGG";
const LAMBDAS: &str = "0.2,0.5,0.8";

fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

/// A fresh directory for one test's files.
fn scratch_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "derna-checkpoint-rerun-{}-{test}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let _ = std::fs::set_permissions(&path, writable());
        std::fs::remove_file(path).unwrap();
    }
    dir
}

fn writable() -> std::fs::Permissions {
    use std::os::unix::fs::PermissionsExt;
    std::fs::Permissions::from_mode(0o644)
}

fn beside(out: &Path, suffix: &str) -> PathBuf {
    let mut name = out.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn read_lines(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn write_lines(path: &Path, lines: &[Value]) {
    let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
    std::fs::write(path, text).unwrap();
}

/// Run `derna design` on the cell at [`LAMBDAS`] into `out`, plus `extra`.
fn design(out: &Path, extra: &[&str]) -> derna_cli::Result<()> {
    let mut argv = words(&[
        "design",
        "--protein",
        PROTEIN,
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--lambda",
        LAMBDAS,
        "--out",
        out.to_str().unwrap(),
    ]);
    argv.extend(words(extra));
    derna_cli::run(argv)
}

/// A run that designs every weight and is then killed as it lands, which is
/// where a progress file holds every weight: the result file cannot be
/// written, so the run fails after the progress lines are out.
fn design_killed_at_landing(out: &Path) -> Vec<Value> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(out, "").unwrap();
    std::fs::set_permissions(out, std::fs::Permissions::from_mode(0o444)).unwrap();
    let e = design(out, &["--checkpoint", "--force"]).unwrap_err();
    assert!(e.to_string().to_lowercase().contains("permission"), "{e}");
    std::fs::set_permissions(out, writable()).unwrap();
    std::fs::remove_file(out).unwrap();
    read_lines(&beside(out, ".design.jsonl"))
}

/// The header's fields that change from one run to the next.
fn steady(header: &Value) -> Value {
    let mut h = header.clone();
    for k in ["wall_s", "peak_rss_kb", "resumed"] {
        h.as_object_mut().unwrap().remove(k);
    }
    h
}

#[test]
fn a_design_progress_file_holds_a_line_per_weight() {
    let dir = scratch_dir("design-lines");
    let out = dir.join("d.jsonl");
    let progress = design_killed_at_landing(&out);
    let (run, designs) = progress.split_first().unwrap();
    assert_eq!(run["record"], "run", "{run}");
    let lambdas: Vec<f64> = designs
        .iter()
        .map(|d| d["lambda"].as_f64().unwrap())
        .collect();
    assert_eq!(lambdas, [0.2, 0.5, 0.8]);
    for d in designs {
        assert_eq!(d["record"], "point", "{d}");
        assert!(d.get("modular").is_some(), "{d}");
    }
}

#[test]
fn a_resumed_design_writes_what_an_uninterrupted_design_writes() {
    let dir = scratch_dir("design-equal");
    let whole_out = dir.join("whole.jsonl");
    design(&whole_out, &[]).unwrap();
    let whole = read_lines(&whole_out);

    let out = dir.join("d.jsonl");
    let progress = design_killed_at_landing(&out);
    // Keep the run line and the middle weight only, as a run killed after it
    // leaves the file.
    let path = beside(&out, ".design.jsonl");
    write_lines(&path, &[progress[0].clone(), progress[2].clone()]);
    design(&out, &["--checkpoint"]).unwrap();
    let resumed = read_lines(&out);
    assert_eq!(resumed[0]["resumed"], 1);
    assert_eq!(steady(&resumed[0]), steady(&whole[0]));
    assert_eq!(resumed[1..], whole[1..]);
    assert!(!path.exists(), "the progress file outlived the result");
}

#[test]
fn a_resumed_weight_is_not_designed_again() {
    let dir = scratch_dir("design-planted");
    let whole_out = dir.join("whole.jsonl");
    design(&whole_out, &[]).unwrap();
    let whole = read_lines(&whole_out);

    let out = dir.join("d.jsonl");
    let progress = design_killed_at_landing(&out);
    let mut planted = progress[2].clone();
    assert_eq!(planted["lambda"], 0.5);
    planted["mfe"] = json!(123.0);
    planted["sequence"] = json!("PLANTED");
    let path = beside(&out, ".design.jsonl");
    write_lines(&path, &[progress[0].clone(), planted.clone()]);
    design(&out, &["--checkpoint"]).unwrap();
    let resumed = read_lines(&out);
    assert_eq!(resumed[0]["resumed"], 1);
    // The planted line is the result's row at its weight, in --lambda order.
    // Its sequence metrics are derived from its sequence as it is written, so
    // they are the planted sequence's and the rest is the line as planted.
    let mut row = resumed[2].clone();
    for field in derna_cli::metrics::FIELDS {
        row.as_object_mut().unwrap().remove(field);
    }
    assert_eq!(row, planted);
    assert_eq!(
        resumed[2]["max_homopolymer"], 1,
        "PLANTED has no repeated letter"
    );
    assert_eq!(resumed[1], whole[1]);
    assert_eq!(resumed[3], whole[3]);
}

#[test]
fn another_runs_design_progress_is_refused() {
    let dir = scratch_dir("design-foreign");
    let out = dir.join("d.jsonl");
    let progress = design_killed_at_landing(&out);
    let path = beside(&out, ".design.jsonl");

    let mut other = progress[0].clone();
    other["run"]["protein"]["sequence"] = json!("MKVLAKGIVLAGQWERT");
    write_lines(&path, &[other, progress[1].clone()]);
    let e = design(&out, &["--checkpoint"]).unwrap_err().to_string();
    assert!(e.contains("another run's progress"), "{e}");
    assert!(e.contains("protein"), "{e}");

    // A weight this run does not ask for is another run's too.
    let mut elsewhere = progress[1].clone();
    elsewhere["lambda"] = json!(0.3);
    write_lines(&path, &[progress[0].clone(), elsewhere]);
    let e = design(&out, &["--checkpoint"]).unwrap_err().to_string();
    assert!(e.contains("another run's progress"), "{e}");
    assert!(!out.exists(), "a refused run wrote a result");
}

#[test]
fn a_forced_design_starts_over() {
    let dir = scratch_dir("design-force");
    let out = dir.join("d.jsonl");
    let progress = design_killed_at_landing(&out);
    let mut planted = progress[2].clone();
    planted["sequence"] = json!("PLANTED");
    let path = beside(&out, ".design.jsonl");
    write_lines(&path, &[progress[0].clone(), planted]);
    design(&out, &["--checkpoint", "--force"]).unwrap();
    let lines = read_lines(&out);
    assert_eq!(lines[0]["resumed"], 0);
    assert!(
        lines.iter().all(|l| l["sequence"] != "PLANTED"),
        "the forced run carried the planted design"
    );
    assert!(!path.exists(), "the progress file outlived the result");
}

#[test]
fn a_design_checkpoint_needs_a_file() {
    for out in [None, Some("-"), Some("/dev/null")] {
        let mut argv = words(&["--protein", "MK", "--lambda", "0.5", "--checkpoint"]);
        if let Some(out) = out {
            argv.extend(words(&["--out", out]));
        }
        let e = derna_cli::design::parse(argv).unwrap_err().to_string();
        assert!(e.contains("--checkpoint keeps"), "{out:?}: {e}");
        assert!(
            e.contains(if out == Some("/dev/null") {
                "regular file"
            } else {
                "not stdout"
            }),
            "{out:?}: {e}"
        );
    }
}

/// Run `derna front` on the cell into `out`, plus `extra`.
fn front(out: &Path, extra: &[&str]) -> derna_cli::Result<()> {
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
        out.to_str().unwrap(),
    ]);
    argv.extend(words(extra));
    derna_cli::run(argv)
}

/// A checkpoint of the whole front `whole` as a killed walk leaves it: one
/// box between hull points reopened with its interior points dropped, and one
/// kept point planted with a structure no fold returns. The planted point's
/// index in `whole` and its structure.
fn planted_checkpoint(whole: &[Value], path: &Path) -> (usize, String) {
    let (header, points) = whole.split_first().unwrap();
    let hull: Vec<usize> = (0..points.len())
        .filter(|&i| points[i]["on_hull"] == true)
        .collect();
    let (lo, hi) = hull
        .windows(2)
        .map(|w| (w[0], w[1]))
        .find(|&(a, b)| b > a + 1)
        .expect("a front with a point inside a box");
    let plant = if lo > 0 { 0 } else { points.len() - 1 };
    let structure = ".".repeat(points[plant]["structure"].as_str().unwrap().len());
    let at = |i: usize| (points[i]["mfe"].clone(), points[i]["cai_log"].clone());
    let (a, b) = (at(lo), at(hi));
    let mut h = header.clone();
    h["complete"] = json!(false);
    h["checkpoint"] = json!(true);
    h["stopped_by"] = Value::Null;
    h["open"] = json!([{
        "mfe_lo": a.0, "cai_log_lo": a.1, "mfe_hi": b.0, "cai_log_hi": b.1,
    }]);
    let mut lines = vec![h];
    for (i, p) in points.iter().enumerate() {
        if i > lo && i < hi {
            continue;
        }
        let mut p = p.clone();
        if i == plant {
            p["structure"] = json!(structure);
        }
        lines.push(p);
    }
    write_lines(path, &lines);
    (plant, structure)
}

#[test]
fn a_front_rerun_over_its_checkpoint_continues_it() {
    let dir = scratch_dir("front-rerun");
    let whole_out = dir.join("whole.jsonl");
    front(&whole_out, &[]).unwrap();
    let whole = read_lines(&whole_out);

    let out = dir.join("f.jsonl");
    let (plant, structure) = planted_checkpoint(&whole, &out);
    front(&out, &["--checkpoint"]).unwrap();
    let resumed = read_lines(&out);
    assert_eq!(resumed[0]["complete"], true);
    assert_eq!(resumed[0]["checkpoint"], false);
    assert_eq!(resumed[0]["resumed"]["from"], out.to_str().unwrap());
    assert_eq!(resumed.len(), whole.len());
    assert_eq!(resumed[1 + plant]["structure"], structure.as_str());
    for (r, w) in resumed[1..].iter().zip(&whole[1..]) {
        assert_eq!(r["mfe"], w["mfe"]);
        assert!((r["cai_log"].as_f64().unwrap() - w["cai_log"].as_f64().unwrap()).abs() < 1e-9);
    }

    // Finished now, so the same command again is refused rather than redone.
    let e = front(&out, &["--checkpoint"]).unwrap_err().to_string();
    assert!(e.contains("finished"), "{e}");
    assert!(e.contains("--force"), "{e}");
}

#[test]
fn a_forced_front_starts_over() {
    let dir = scratch_dir("front-force");
    let whole_out = dir.join("whole.jsonl");
    front(&whole_out, &[]).unwrap();
    let whole = read_lines(&whole_out);

    let out = dir.join("f.jsonl");
    let (plant, structure) = planted_checkpoint(&whole, &out);
    front(&out, &["--checkpoint", "--force"]).unwrap();
    let lines = read_lines(&out);
    assert_eq!(lines[0]["resumed"], Value::Null);
    assert_eq!(lines[0]["complete"], true);
    assert_ne!(lines[1 + plant]["structure"], structure.as_str());
    assert_eq!(lines.len(), whole.len());
}

#[test]
fn a_front_rerun_refuses_a_file_it_did_not_write() {
    let dir = scratch_dir("front-foreign");
    let out = dir.join("f.jsonl");
    design(&out, &[]).unwrap();
    let e = front(&out, &["--checkpoint"]).unwrap_err().to_string();
    assert!(e.contains("not a front walk"), "{e}");
    assert!(e.contains("--force"), "{e}");
}

/// Run `derna front --modular` on the oracle cell into `out`, plus `extra`.
fn modular(out: &Path, extra: &[&str]) -> derna_cli::Result<()> {
    let mut argv = words(&[
        "front",
        "--protein",
        "MPRGLA",
        "--utr5",
        "CCGCGG",
        "--stop",
        "none",
        "--modular",
        "--checkpoint",
        "--out",
        out.to_str().unwrap(),
    ]);
    argv.extend(words(extra));
    derna_cli::run(argv)
}

/// A modular front's points without the fields a resumed search may spell
/// differently among co-optimal designs.
fn settled(lines: &[Value]) -> Vec<Value> {
    lines[1..]
        .iter()
        .map(|p| {
            let mut p = p.clone();
            for k in ["structure", "free_structure", "sequence"] {
                p.as_object_mut().unwrap().remove(k);
            }
            p
        })
        .collect()
}

#[test]
fn a_modular_rerun_continues_the_walk_and_the_search() {
    let dir = scratch_dir("modular-rerun");
    let out = dir.join("so.jsonl");
    let walked = dir.join("walked.jsonl");
    modular(&out, &["--front-out", walked.to_str().unwrap()]).unwrap();
    let whole = read_lines(&out);
    let (walk, search) = (beside(&out, ".walk.jsonl"), beside(&out, ".search.jsonl"));
    assert_eq!(whole[0]["complete"], true);
    assert!(whole[0]["gaps"].as_u64().unwrap() > 0, "{}", whole[0]);
    assert!(!walk.exists(), "a complete front kept its walk");
    let first = read_lines(&search);
    assert!(first[0].get("run").is_some(), "{}", first[0]);

    // Finished, so the same command again is refused.
    let e = modular(&out, &[]).unwrap_err().to_string();
    assert!(e.contains("finished modular front"), "{e}");

    // Killed after the walk and before the search landed: the walk kept beside
    // --out and the search's log. The rerun walks nothing — a walk read back
    // reports no gaps of its own — and checks no point again.
    std::fs::remove_file(&out).unwrap();
    let mut constrained = read_lines(&walked);
    constrained[0]["checkpoint"] = json!(true);
    write_lines(&walk, &constrained);
    modular(&out, &[]).unwrap();
    let again = read_lines(&out);
    assert_eq!(again[0]["gaps"], 0, "the rerun walked again");
    assert_eq!(again[0]["resumed_search"]["from"], search.to_str().unwrap());
    assert_eq!(
        again[0]["resumed_search"]["checked"],
        whole[0]["constrained_points"]
    );
    assert_eq!(settled(&again), settled(&whole));
    assert!(!walk.exists(), "a complete front kept its walk");
}

/// The resume the equality test above cannot establish. A run that threw the
/// progress file away and checked all 465 points again would write the same
/// front, so equality is consistent with no resume at all; the planted
/// verdict is not. One point that passed its check is recorded as a point
/// that did not — a verdict the check would never return for it — and the
/// resumed run must seed its search with one point fewer and never check that
/// point again.
#[test]
fn a_settled_point_a_resumed_modular_search_reads_is_not_checked_again() {
    let dir = scratch_dir("modular-planted");
    let out = dir.join("so.jsonl");
    modular(&out, &[]).unwrap();
    let whole = read_lines(&out);
    let seeds = whole[0]["seeds"].as_u64().unwrap();
    assert!(seeds > 0, "{}", whole[0]);

    let search = beside(&out, ".search.jsonl");
    let mut log = read_lines(&search);
    let planted = log
        .iter_mut()
        .find(|l| l["modular"] == true)
        .expect("a point whose check passed");
    planted["modular"] = json!(false);
    let demoted = planted["checked"].as_str().unwrap().to_string();
    write_lines(&search, &log);

    std::fs::remove_file(&out).unwrap();
    modular(&out, &[]).unwrap();
    let again = read_lines(&out);
    assert_eq!(
        again[0]["resumed_search"]["checked"],
        whole[0]["constrained_points"]
    );
    assert_eq!(
        again[0]["seeds"].as_u64().unwrap(),
        seeds - 1,
        "the resumed run seeded the search with the planted point"
    );
    // And the run never folded it: a point it checked itself would be in the
    // progress file with the verdict its own check returned.
    let checked_again = read_lines(&search)
        .iter()
        .any(|l| l["checked"] == demoted.as_str() && l["modular"] == true);
    assert!(!checked_again, "the resumed run checked {demoted} again");
}

#[test]
fn a_modular_rerun_refuses_another_molecules_search() {
    let dir = scratch_dir("modular-foreign");
    let out = dir.join("so.jsonl");
    modular(&out, &[]).unwrap();
    let search = beside(&out, ".search.jsonl");
    let mut log = read_lines(&search);
    log[0]["run"]["protein"]["sequence"] = json!("MPRGLV");
    write_lines(&search, &log);
    std::fs::remove_file(&out).unwrap();
    let e = modular(&out, &[]).unwrap_err().to_string();
    assert!(e.contains("another run's search progress"), "{e}");
    assert!(e.contains("protein"), "{e}");
}

#[test]
fn a_forced_modular_front_starts_its_search_over() {
    let dir = scratch_dir("modular-force");
    let out = dir.join("so.jsonl");
    modular(&out, &[]).unwrap();
    let search = beside(&out, ".search.jsonl");
    let mut log = read_lines(&search);
    log.push(json!({
        "checked": "PLANTED",
        "modular": true,
        "gap_kcal_mol": null,
        "blocking": [],
    }));
    write_lines(&search, &log);
    modular(&out, &["--force"]).unwrap();
    let lines = read_lines(&out);
    assert_eq!(lines[0]["resumed_search"], Value::Null);
    let after = std::fs::read_to_string(&search).unwrap();
    assert!(!after.contains("PLANTED"), "{after}");
}
