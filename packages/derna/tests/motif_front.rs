//! `derna front --motif`: the front of the designs that spell no forbidden
//! motif over the coding sequence.
//!
//! The library's own oracle checks the fold against exhaustive enumeration on
//! molecules small enough to enumerate. Here the command's walk is checked
//! against the library's whole fold on a longer molecule, against the motif
//! rule `design` enforces, and against the plain front it must differ from.

use derna_cli::front;
use derna_cli::molecule::Molecule;
use derna_cli::show;
use derna_lib::motif_front::{motif_front_exact, MotifFrontSpec, MotifResume, MotifRun, MotifWalk};
use derna_lib::{Params, UtrSpec};

const PROTEIN: &str = "MKVLAKGIV";
const UTR5: &str = "ACAUUUGCUU";
const UTR3: &str = "GGCCAAUUGG";
const MOTIF: &str = "GCC";
/// A protein whose walk settles several pairs, so a walk stopped or
/// checkpointed part way leaves boxes open to resume (17 points, 6 hull
/// vertices under `MOTIF` and these flanks).
const RESUMED: &str = "MKVLAKGIVLAGQWERS";

fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

fn fixed(spec: UtrSpec) -> String {
    match spec {
        UtrSpec::Fixed(s) => s,
        _ => unreachable!("the command's flanks are fixed"),
    }
}

#[test]
fn the_motif_front_avoids_its_motif_and_is_the_whole_fold() {
    let argv = words(&[
        "--protein",
        PROTEIN,
        "--stop",
        "none",
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--motif",
        MOTIF,
    ]);
    let args = front::parse(argv).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let walked = front::motif_walk(&m, &args.sequence, 2, Default::default(), None).unwrap();
    assert!(
        walked.front.len() >= 2,
        "a front of one point tests nothing"
    );

    // No point spells the motif with a position in the coding sequence.
    let (lo, hi) = (UTR5.len(), UTR5.len() + 3 * PROTEIN.len());
    for p in &walked.front {
        let spelled = (0..=p.sequence.len() - MOTIF.len())
            .filter(|&s| p.sequence[s..].starts_with(MOTIF))
            .any(|s| s < hi && s + MOTIF.len() > lo);
        assert!(!spelled, "{} spells {MOTIF}", p.sequence);
    }

    // The walk and the whole fold agree point for point.
    let params = Params::load_default().unwrap();
    let (utr5, utr3) = (fixed(m.utr5_spec()), fixed(m.utr3_spec()));
    let whole = motif_front_exact(
        &params,
        &m.protein.sequence,
        &m.codons.usage,
        &MotifFrontSpec {
            utr5: &utr5,
            utr3: &utr3,
            constraints: m.table(),
            motifs: &args.sequence.placed(&m).unwrap(),
            pins: None,
        },
    )
    .unwrap();
    let key = |p: &derna_lib::FrontPoint| ((p.mfe_kcal_mol * 100.0).round() as i64, p.cai_log);
    assert_eq!(walked.front.len(), whole.len());
    for (a, b) in walked.front.iter().zip(&whole) {
        assert_eq!(key(a).0, key(b).0);
        assert!((key(a).1 - key(b).1).abs() < 1e-9);
    }

    // And the motif binds: the plain front is a different set of points.
    let plain = front::walk(&m, 2).unwrap();
    let differs = plain.front.len() != walked.front.len()
        || plain
            .front
            .iter()
            .zip(&walked.front)
            .any(|(a, b)| key(a).0 != key(b).0 || (key(a).1 - key(b).1).abs() > 1e-9);
    assert!(differs, "{MOTIF} removes no point of the plain front here");
}

#[test]
fn the_motif_front_is_the_same_at_any_thread_count() {
    let argv = words(&[
        "--protein",
        PROTEIN,
        "--stop",
        "none",
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--motif",
        MOTIF,
    ]);
    let args = front::parse(argv).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let one = front::motif_walk(&m, &args.sequence, 1, Default::default(), None).unwrap();
    let three = front::motif_walk(&m, &args.sequence, 3, Default::default(), None).unwrap();
    assert_eq!(one.front, three.front);
    assert_eq!(one.vertices, three.vertices);
}

#[test]
fn a_budget_returns_front_points_and_names_what_is_open() {
    let argv = words(&[
        "--protein",
        PROTEIN,
        "--stop",
        "none",
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--motif",
        MOTIF,
        "--max-secs",
        "0.000001",
    ]);
    let args = front::parse(argv).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let stopped = front::motif_walk(&m, &args.sequence, 2, args.budget, None).unwrap();
    let whole = front::motif_walk(&m, &args.sequence, 2, Default::default(), None).unwrap();
    assert_eq!(stopped.stopped_by, Some("time"));
    assert!(!stopped.open.is_empty());
    assert!(whole.stopped_by.is_none() && whole.open.is_empty());
    // The ends always complete, and every point returned is on the front.
    assert!(stopped.front.len() >= 2, "{}", stopped.front.len());
    for p in &stopped.front {
        assert!(
            whole.front.contains(p),
            "{} is not a front point",
            p.sequence
        );
    }
}

#[test]
fn a_pin_is_refused_with_its_reason() {
    let e = front::parse(words(&["--protein", "MK", "--pin", "1:A"]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("--pin is not supported yet"), "{e}");
}

/// With motifs, `--front-from` reads a motif front for the modular
/// search, so it is refused without one.
#[test]
fn a_front_to_read_with_motifs_needs_the_modular_search() {
    let e = front::parse(words(&[
        "--protein",
        "MK",
        "--motif",
        "GG",
        "--front-from",
        "plain.jsonl",
    ]))
    .unwrap_err()
    .to_string();
    assert!(e.contains("needs --modular"), "{e}");
}

/// A checkpoint is written beside `--out` and renamed over it, so `--out`
/// must be a regular file: a device would fail only after the whole walk.
#[test]
fn a_checkpoint_to_a_device_is_refused() {
    let e = front::parse(words(&[
        "--protein",
        "MK",
        "--motif",
        "GG",
        "--checkpoint",
        "--out",
        "/dev/null",
        "--force",
    ]))
    .unwrap_err()
    .to_string();
    assert!(e.contains("needs a regular file"), "{e}");
}

#[test]
fn an_infeasible_motif_set_names_the_motifs() {
    let args = front::parse(words(&["--protein", "MWW", "--motif", "UGG"])).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let e = front::motif_walk(&m, &args.sequence, 1, Default::default(), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("no encoding") && e.contains("UGG"), "{e}");
}

/// The partly solved boxes of `walk` as a [`MotifResume`] reads them.
fn rests_of(walk: &MotifWalk) -> Vec<((f64, f64), (f64, f64))> {
    walk.rests
        .iter()
        .map(|r| {
            (
                (r.lo.to_kcal_mol(), r.hi.to_kcal_mol()),
                (r.from.to_kcal_mol(), r.floor),
            )
        })
        .collect()
}

/// A fresh path under the system's temporary directory for one test's file.
fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("derna-motif-front-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    let _ = std::fs::remove_file(&path);
    path
}

/// Run `derna front` with these words and read back what it wrote.
fn front_file(extra: &[&str], out: &std::path::Path) -> show::Rendered {
    let mut argv = words(&[
        "--protein",
        RESUMED,
        "--stop",
        "none",
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--motif",
        MOTIF,
        "--threads",
        "2",
        "--out",
        out.to_str().unwrap(),
    ]);
    argv.extend(words(extra));
    front::run(&front::parse(argv).unwrap()).unwrap();
    show::read(Some(out)).unwrap()
}

/// A point's objectives, energy in integer dekacalories, and whether it is a
/// hull vertex.
fn objectives(p: &serde_json::Value) -> (i64, f64, bool) {
    (
        (p["mfe"].as_f64().unwrap() * 100.0).round() as i64,
        p["cai_log"].as_f64().unwrap(),
        p["on_hull"].as_bool().unwrap(),
    )
}

#[test]
fn a_resumed_walk_returns_the_whole_front() {
    let (from, resumed, whole) = (
        scratch("stopped.jsonl"),
        scratch("resumed.jsonl"),
        scratch("whole.jsonl"),
    );
    let stopped = front_file(&["--max-secs", "0.000001"], &from);
    let whole = front_file(&[], &whole);
    // The stopped walk must leave something to resume, or the check is empty.
    assert_eq!(stopped.header["complete"], false);
    assert!(!stopped.header["open"].as_array().unwrap().is_empty());
    assert!(stopped.points.len() < whole.points.len());

    let resumed = front_file(&["--resume", from.to_str().unwrap()], &resumed);
    assert_eq!(resumed.header["complete"], true);
    assert_eq!(resumed.header["resumed"]["points"], stopped.points.len());
    assert_eq!(resumed.points.len(), whole.points.len());
    for (a, b) in resumed.points.iter().zip(&whole.points) {
        let (a, b) = (objectives(a), objectives(b));
        assert_eq!((a.0, a.2), (b.0, b.2));
        assert!((a.1 - b.1).abs() < 1e-9, "{a:?} against {b:?}");
    }
}

#[test]
fn a_resume_from_another_run_is_refused() {
    let stopped = scratch("other.jsonl");
    front_file(&["--max-secs", "0.000001"], &stopped);
    let argv = words(&[
        "--protein",
        RESUMED,
        "--stop",
        "none",
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--motif",
        "GCU",
        "--resume",
        stopped.to_str().unwrap(),
        "--out",
        scratch("refused.jsonl").to_str().unwrap(),
    ]);
    let e = front::run(&front::parse(argv).unwrap())
        .unwrap_err()
        .to_string();
    assert!(e.contains("motifs differs"), "{e}");
}

/// Leucine's third base to U leaves it CUU, so the pins take its best codon.
const PIN: &str = "12:U";

#[test]
fn a_pinned_motif_front_spells_its_pins() {
    let (pinned, plain) = (scratch("pinned.jsonl"), scratch("unpinned.jsonl"));
    let pinned = front_file(&["--pin", PIN], &pinned);
    let plain = front_file(&[], &plain);
    assert_eq!(pinned.header["pins"], serde_json::json!([PIN]));
    assert_eq!(pinned.header["complete"], true);
    let at = UTR5.len() + 11;
    for p in &pinned.points {
        assert_eq!(&p["sequence"].as_str().unwrap()[at..=at], "U", "{p}");
    }
    let best = |r: &show::Rendered| {
        r.points
            .iter()
            .map(|p| p["cai_log"].as_f64().unwrap())
            .fold(f64::MIN, f64::max)
    };
    // The pin binds: the most adapted pinned design gives up CUG for CUU.
    assert!(best(&pinned) < best(&plain) - 1e-3);
    assert!(pinned.points.iter().any(|p| p["on_hull"] == true));

    // A stopped pinned walk resumes to the whole pinned front.
    let (stopped, resumed) = (scratch("pinned-part.jsonl"), scratch("pinned-rest.jsonl"));
    let part = front_file(&["--pin", PIN, "--max-secs", "0.000001"], &stopped);
    assert!(!part.header["open"].as_array().unwrap().is_empty());
    let rest = front_file(
        &["--pin", PIN, "--resume", stopped.to_str().unwrap()],
        &resumed,
    );
    assert_eq!(rest.points.len(), pinned.points.len());
    for (a, b) in rest.points.iter().zip(&pinned.points) {
        let (a, b) = (objectives(a), objectives(b));
        assert_eq!((a.0, a.2), (b.0, b.2));
        assert!((a.1 - b.1).abs() < 1e-9, "{a:?} against {b:?}");
    }
}

#[test]
fn a_resume_across_pins_is_refused() {
    let (pinned, plain) = (scratch("pinned-stop.jsonl"), scratch("plain-stop.jsonl"));
    front_file(&["--pin", PIN, "--max-secs", "0.000001"], &pinned);
    front_file(&["--max-secs", "0.000001"], &plain);
    for (from, pin) in [(&pinned, None), (&plain, Some(PIN))] {
        let mut extra = vec!["--resume", from.to_str().unwrap()];
        if let Some(pin) = pin {
            extra.extend(["--pin", pin]);
        }
        let mut argv = words(&[
            "--protein",
            RESUMED,
            "--stop",
            "none",
            "--utr5",
            UTR5,
            "--utr3",
            UTR3,
            "--motif",
            MOTIF,
            "--out",
            scratch("refused-pins.jsonl").to_str().unwrap(),
        ]);
        argv.extend(words(&extra));
        let e = front::run(&front::parse(argv).unwrap())
            .unwrap_err()
            .to_string();
        assert!(e.contains("pin"), "{e}");
    }
}

/// A motif set is a set: an earlier run given the same motifs in another
/// order resumes.
#[test]
fn a_resume_takes_its_motifs_in_any_order() {
    let (from, to) = (scratch("ordered.jsonl"), scratch("reordered.jsonl"));
    front_file(&["--motif", "GCU", "--max-secs", "0.000001"], &from);
    let argv = words(&[
        "--protein",
        RESUMED,
        "--stop",
        "none",
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--motif",
        "GCU",
        "--motif",
        MOTIF,
        "--resume",
        from.to_str().unwrap(),
        "--out",
        to.to_str().unwrap(),
    ]);
    front::run(&front::parse(argv).unwrap()).unwrap();
    assert_eq!(show::read(Some(&to)).unwrap().header["complete"], true);
}

/// An open box whose ends are swapped would fold nothing, or no weight at
/// all; it is refused rather than reported settled.
#[test]
fn a_resume_with_a_reversed_box_is_refused() {
    let (from, swapped) = (scratch("unswapped.jsonl"), scratch("swapped.jsonl"));
    front_file(&["--max-secs", "0.000001"], &from);
    let text = std::fs::read_to_string(&from).unwrap();
    let mut lines: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let open = lines[0]["open"].as_array_mut().unwrap();
    let b = open
        .iter_mut()
        .find(|b| b["mfe_lo"] != b["mfe_hi"])
        .expect("a stopped walk leaves a box between two points");
    for (lo, hi) in [("mfe_lo", "mfe_hi"), ("cai_log_lo", "cai_log_hi")] {
        let (l, h) = (b[lo].clone(), b[hi].clone());
        b[lo] = h;
        b[hi] = l;
    }
    let edited: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
    std::fs::write(&swapped, edited.join("\n") + "\n").unwrap();
    let argv = words(&[
        "--protein",
        RESUMED,
        "--stop",
        "none",
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--motif",
        MOTIF,
        "--resume",
        swapped.to_str().unwrap(),
        "--out",
        scratch("from-swapped.jsonl").to_str().unwrap(),
    ]);
    let e = front::run(&front::parse(argv).unwrap())
        .unwrap_err()
        .to_string();
    assert!(e.contains("more stable then"), "{e}");
}

#[test]
fn every_checkpoint_resumes_to_the_whole_front() {
    let argv = words(&[
        "--protein",
        RESUMED,
        "--stop",
        "none",
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--motif",
        MOTIF,
    ]);
    let args = front::parse(argv).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let taken = std::sync::Mutex::new(Vec::new());
    let hand = |so_far: &MotifWalk| taken.lock().unwrap().push(so_far.clone());
    let run = MotifRun {
        checkpoint: Some(&hand),
        ..MotifRun::default()
    };
    let whole = front::motif_walk_run(&m, &args.sequence, 2, run).unwrap();
    let taken = taken.into_inner().unwrap();
    // A checkpoint with nothing open would resume to itself and test nothing.
    assert!(
        taken.iter().filter(|c| !c.open.is_empty()).count() >= 2,
        "{} checkpoint(s)",
        taken.len()
    );
    assert!(taken.last().unwrap().open.is_empty());

    let key = |p: &derna_lib::FrontPoint| ((p.mfe_kcal_mol * 100.0).round() as i64, p.cai_log);
    for so_far in &taken {
        let resume = MotifResume {
            points: so_far
                .front
                .iter()
                .map(|p| {
                    let vertex = so_far.vertices.iter().any(|v| {
                        (v.mfe.to_kcal_mol() - p.mfe_kcal_mol).abs() < 1e-9
                            && (v.cai - p.cai_log).abs() < 1e-9
                    });
                    (p.clone(), vertex)
                })
                .collect(),
            open: so_far
                .open
                .iter()
                .map(|(a, b)| ((a.mfe.to_kcal_mol(), a.cai), (b.mfe.to_kcal_mol(), b.cai)))
                .collect(),
            rests: rests_of(so_far),
            rehull: false,
            moves: Vec::new(),
        };
        let resumed =
            front::motif_walk(&m, &args.sequence, 2, Default::default(), Some(&resume)).unwrap();
        assert_eq!(resumed.front.len(), whole.front.len());
        for (a, b) in resumed.front.iter().zip(&whole.front) {
            assert_eq!(key(a).0, key(b).0);
            assert!((key(a).1 - key(b).1).abs() < 1e-9);
        }
        assert!(resumed.open.is_empty());
    }
}
