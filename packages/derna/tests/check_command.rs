//! `derna check` certifies a result's points, and gets the same
//! verdicts the run that produced them got.
//!
//! The strong property is an equality, not a smoke test: a design carries its
//! own check, and a check of that file folds the molecule again from the
//! header alone. If the two disagree, either the header lost something the
//! fold depends on — a flank, the constraint, the codon table — or the two
//! checks are not the same check. Either way the pipeline's halves cannot be
//! trusted to compose, which is the whole point of `front | check`.

use derna_cli::design;
use derna_cli::free_check::Checked;
use derna_cli::molecule::Molecule;
use derna_cli::{check, result, show};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

fn flanks(pair: &str) -> (String, String) {
    let text =
        std::fs::read_to_string(root().join("packages/derna/tests/data/flank-pairs.json")).unwrap();
    let v: Value = serde_json::from_str(&text).unwrap();
    (
        v[pair]["utr5"].as_str().unwrap().to_string(),
        v[pair]["utr3"].as_str().unwrap().to_string(),
    )
}

/// The cells this equality runs over: a protein and a flank pair each. Two,
/// for the same reason the design and front equalities have two — a property
/// of one molecule is not a property of the command.
const CELLS: [(&str, &str); 2] = [
    (
        "packages/derna-codon-dp/tests/data/GLPE_HUMAN_20.fasta",
        "hbb25",
    ),
    ("packages/derna/tests/data/p0dtc2-25.fasta", "gu-poor"),
];

/// The first cell's constrained design result, for the tests that need only
/// one molecule.
fn constrained_design() -> (Molecule, Vec<Value>) {
    design_in_cell(CELLS[0])
}

/// A constrained design result, as lines, with the molecule it was made over.
fn design_in_cell(cell: (&str, &str)) -> (Molecule, Vec<Value>) {
    let (protein, pair) = cell;
    let (utr5, utr3) = flanks(pair);
    let fasta = root().join(protein);
    let argv = words(&[
        "--protein",
        fasta.to_str().unwrap(),
        "--lambda",
        "0.1,0.6",
        "--utr5",
        &utr5,
        "--utr3",
        &utr3,
    ]);
    let args = design::parse(argv).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let solved = design::solve(&m, &args.lambdas, 1).unwrap();

    let mut lines = vec![result::header("design", &m, serde_json::Map::new())];
    lines.extend(
        solved
            .iter()
            .map(|s| result::checked_point(&s.design, Some(s.lambda), true, s.checked.as_ref())),
    );
    (m, lines)
}

#[test]
fn a_check_reproduces_the_check_the_producing_run_made() {
    for cell in CELLS {
        check_one_cell(cell);
    }
}

fn check_one_cell(cell: (&str, &str)) {
    let (_, lines) = design_in_cell(cell);
    let text: String = lines
        .iter()
        .map(|l| serde_json::to_string(l).unwrap() + "\n")
        .collect();
    let read = show::parse_lines(&text).unwrap();

    // The molecule comes from the header alone — no flags.
    let from_header = Molecule::from_header(&read.header).unwrap();
    let certified = check::certify(&from_header, &read.points, None).unwrap();
    assert_eq!(certified.len(), read.points.len());

    for (point, checked) in &certified {
        let (free, gap) = match checked {
            Checked::Exact(g) => (g.free_mfe_kcal_mol, g.gap_kcal_mol),
            Checked::Refuted { .. } => panic!("no beam was asked for"),
        };
        // The producing run's own check, from the file.
        let (protein, pair) = cell;
        assert_eq!(
            point["free_mfe"].as_f64().unwrap(),
            free,
            "{protein}/{pair}: free_mfe"
        );
        assert_eq!(point["gap"].as_f64().unwrap(), gap, "{protein}/{pair}: gap");
        assert_eq!(
            point["modular"].as_bool().unwrap(),
            checked.modular(),
            "the verdict"
        );
    }
    // Non-vacuity: a file of clean points would pass the equality without
    // exercising the failing path, which is the one `check` exists for.
    assert!(
        certified.iter().any(|(_, c)| !c.modular()),
        "no point in {cell:?} violates the constraint, so the equality proves little there"
    );
}

#[test]
fn a_certification_is_the_same_at_any_thread_count() {
    // Each point's check reads only its own sequence, and the results are
    // collected in the input's order, so the thread count changes the wall
    // clock and nothing a reader sees. That is what `--deterministic` would
    // otherwise be for here, and why check refuses it.
    let (_, lines) = constrained_design();
    let text: String = lines
        .iter()
        .map(|l| serde_json::to_string(l).unwrap() + "\n")
        .collect();
    let read = show::parse_lines(&text).unwrap();
    let m = Molecule::from_header(&read.header).unwrap();

    let serial = check::certify_with(&m, &read.points, None, 1).unwrap();
    let parallel = check::certify_with(&m, &read.points, None, 4).unwrap();
    assert_eq!(serial.len(), parallel.len());
    for ((point_a, a), (point_b, b)) in serial.iter().zip(&parallel) {
        // The same point, in the same place: a worker writing its result to
        // the wrong slot would reorder the front.
        assert_eq!(point_a["sequence"], point_b["sequence"]);
        match (a, b) {
            (Checked::Exact(x), Checked::Exact(y)) => {
                assert_eq!(x.free_mfe_kcal_mol, y.free_mfe_kcal_mol);
                assert_eq!(x.gap_kcal_mol, y.gap_kcal_mol);
                assert_eq!(x.free_structure, y.free_structure);
            }
            _ => panic!("no beam was asked for"),
        }
    }
}

#[test]
fn the_check_folds_the_molecule_instead_of_believing_the_file() {
    // The mutant this guards against: a `check` that reports what the
    // file already claims instead of folding. Every other test here compares
    // a check's verdicts against a design's, and both are computed the same
    // way — so a check that simply echoed the file would pass all of them.
    //
    // This one lies to it. The recorded verdict is flipped to "modular
    // " with a zero gap on a point that is neither, and the check has
    // to contradict the file it was handed.
    let (m, lines) = constrained_design();
    let mut points: Vec<Value> = lines[1..].to_vec();
    assert!(
        points.iter().any(|p| p["modular"] == Value::Bool(false)),
        "this cell has nothing for the lie to contradict"
    );
    let truth: Vec<(bool, f64)> = points
        .iter()
        .map(|p| (p["modular"].as_bool().unwrap(), p["gap"].as_f64().unwrap()))
        .collect();

    for point in &mut points {
        let object = point.as_object_mut().unwrap();
        object.insert("modular".into(), Value::Bool(true));
        object.insert("gap".into(), json!(0.0));
        object.insert("free_mfe".into(), object["mfe"].clone());
    }

    let checked = check::certify(&m, &points, None).unwrap();
    for ((point, check), (was_optimal, was_gap)) in checked.iter().zip(&truth) {
        let certified = check::certified_point(point, check);
        assert_eq!(
            certified["modular"],
            Value::Bool(*was_optimal),
            "the check took the file's word for the verdict"
        );
        let gap = certified["gap"].as_f64().unwrap();
        assert!(
            (gap - was_gap).abs() < 1e-9,
            "the check took the file's word for the gap: {gap} against {was_gap}"
        );
    }
}

#[test]
fn certifying_an_already_certified_result_changes_nothing() {
    // A checked file is a valid input (D33), and `check` on one
    // re-folds the same molecules: the verdicts must come back identical, and
    // the stale fields must be replaced rather than joined by a second set.
    // Idempotence is what makes resuming safe — a file can be handed back in
    // without the first pass's results decaying into a second opinion.
    let (_, lines) = constrained_design();
    let text: String = lines
        .iter()
        .map(|l| serde_json::to_string(l).unwrap() + "\n")
        .collect();
    let first = show::parse_lines(&text).unwrap();
    let m = Molecule::from_header(&first.header).unwrap();

    let once: Vec<Value> = check::certify(&m, &first.points, None)
        .unwrap()
        .iter()
        .map(|(p, c)| check::certified_point(p, c))
        .collect();
    let twice: Vec<Value> = check::certify(&m, &once, None)
        .unwrap()
        .iter()
        .map(|(p, c)| check::certified_point(p, c))
        .collect();
    assert_eq!(once, twice, "a second certification changed the result");

    // And the check's fields appear once, not twice.
    for point in &twice {
        let object = point.as_object().unwrap();
        assert_eq!(object.keys().filter(|k| *k == "gap").count(), 1);
        assert!(object.contains_key("free_mfe"));
        assert!(!object.contains_key("free_mfe_at_most"));
    }
}

#[test]
fn the_verdict_and_the_gap_say_the_same_thing() {
    // `modular` is defined as a zero gap, so the two cannot
    // disagree — a reporting bug that set the flag from something else (the
    // violating-pair list, say) would show up here and nowhere else, because
    // free optima are routinely co-optimal: a structure that violates the
    // constraint can exist beside an equally good one that does not.
    for cell in CELLS {
        let (_, lines) = design_in_cell(cell);
        let text: String = lines
            .iter()
            .map(|l| serde_json::to_string(l).unwrap() + "\n")
            .collect();
        let read = show::parse_lines(&text).unwrap();
        let m = Molecule::from_header(&read.header).unwrap();
        for (_, checked) in check::certify(&m, &read.points, None).unwrap() {
            let Checked::Exact(gap) = &checked else {
                panic!("no beam was asked for")
            };
            assert!(gap.gap_kcal_mol >= -1e-9, "{cell:?}: a negative gap");
            assert_eq!(
                checked.modular(),
                gap.gap_kcal_mol.abs() < 1e-9,
                "{cell:?}: the verdict and the gap disagree (gap {})",
                gap.gap_kcal_mol
            );
        }
    }
}

#[test]
fn the_molecule_from_a_header_is_the_molecule_the_flags_built() {
    let (from_flags, lines) = constrained_design();
    let from_header = Molecule::from_header(&lines[0]).unwrap();

    assert_eq!(from_header.protein.sequence, from_flags.protein.sequence);
    assert_eq!(
        from_header.utr5.as_ref().map(|u| &u.sequence),
        from_flags.utr5.as_ref().map(|u| &u.sequence)
    );
    assert_eq!(
        from_header.utr3.as_ref().map(|u| &u.sequence),
        from_flags.utr3.as_ref().map(|u| &u.sequence)
    );
    assert_eq!(from_header.constraint, from_flags.constraint);
    assert_eq!(from_header.cds(), from_flags.cds());
    // The table travels by content, so a check scores CAI against exactly
    // what the run used rather than against whatever `human` means here.
    assert_eq!(from_header.codons.rows, from_flags.codons.rows);
}

#[test]
fn a_beam_refutation_in_a_check_agrees_with_the_exact_verdict() {
    let (_, lines) = constrained_design();
    let text: String = lines
        .iter()
        .map(|l| serde_json::to_string(l).unwrap() + "\n")
        .collect();
    let read = show::parse_lines(&text).unwrap();
    let m = Molecule::from_header(&read.header).unwrap();

    let exact = check::certify(&m, &read.points, None).unwrap();
    let bounded = check::certify(&m, &read.points, Some(50)).unwrap();
    for ((_, a), (_, b)) in exact.iter().zip(&bounded) {
        assert_eq!(
            a.modular(),
            b.modular(),
            "the beam and the exact fold disagree on the verdict"
        );
        if let (
            Checked::Exact(g),
            Checked::Refuted {
                free_mfe_at_most, ..
            },
        ) = (a, b)
        {
            assert!(
                *free_mfe_at_most >= g.free_mfe_kcal_mol - 1e-9,
                "the bound {free_mfe_at_most} claims less than the exact free MFE {}",
                g.free_mfe_kcal_mol
            );
        }
    }
}

#[test]
fn an_unconstrained_result_has_nothing_to_certify() {
    let fasta = root().join("packages/derna-codon-dp/tests/data/GLPE_HUMAN_20.fasta");
    let args = design::parse(words(&[
        "--protein",
        fasta.to_str().unwrap(),
        "--lambda",
        "0.5",
    ]))
    .unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let header = result::header("design", &m, serde_json::Map::new());
    let rebuilt = Molecule::from_header(&header).unwrap();
    // No flanks, so no crossing class exists and the constraint is empty —
    // `check`'s own refusal reads this.
    assert!(rebuilt.constraint.is_inert());
}

/// Write `lines` as a result file under `dir`, and return its path.
fn result_file(dir: &Path, name: &str, lines: &[Value]) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(name);
    let text: String = lines
        .iter()
        .map(|l| serde_json::to_string(l).unwrap() + "\n")
        .collect();
    std::fs::write(&path, text).unwrap();
    path
}

fn checkpoint_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("derna-check-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `--checkpoint` keeps each verdict as it settles, and a run that takes them
/// back writes the file the uninterrupted run would have written.
///
/// This is the property the flag exists for: a killed check must be worth its
/// elapsed time, and resuming must not be a second, weaker answer. Both halves
/// are asserted — the progress file holds a line per point *while* the run is
/// on, and the resumed result is byte-equal to the unresumed one apart from
/// the header fields that record the run rather than the molecule.
#[test]
fn a_resumed_check_writes_what_an_uninterrupted_check_writes() {
    let dir = checkpoint_dir("resume");
    let (_, lines) = constrained_design();
    let input = result_file(&dir, "front.jsonl", &lines);

    let run_to = |name: &str, checkpoint: bool| -> (Value, Vec<Value>) {
        let out = dir.join(name);
        let mut argv = words(&[input.to_str().unwrap(), "--out", out.to_str().unwrap()]);
        if checkpoint {
            argv.push("--checkpoint".into());
        }
        check::run(&check::parse(argv).unwrap()).unwrap();
        let read = show::parse_lines(&std::fs::read_to_string(&out).unwrap()).unwrap();
        (read.header, read.points)
    };

    let (_, plain) = run_to("plain.jsonl", false);

    // A first run killed after its first point leaves that point behind: the
    // progress file is written as the check goes, not at the end.
    let partial = dir.join("resumed.jsonl");
    let progress = check::progress_path(&partial);
    let first = serde_json::to_string(&plain[0]).unwrap();
    std::fs::write(&progress, first + "\n").unwrap();

    let (header, resumed) = run_to("resumed.jsonl", true);
    assert_eq!(resumed, plain, "a resumed check settled different points");
    assert!(
        plain.len() > 1,
        "a one-point front cannot show that resuming skipped anything"
    );
    // The progress file was read, which the equality above cannot show. That
    // the point was then not folded again is a separate claim, and the
    // equality cannot show that either: see the planted-verdict test below.
    assert_eq!(
        header["resumed"],
        json!(1),
        "the planted verdict was not read back"
    );
    assert!(
        !progress.exists(),
        "the landed result makes the progress file resumable twice"
    );
}

/// A resumed point is the progress file's verdict, not a fresh fold of it.
///
/// The non-vacuity probe for the equality above, which cannot tell a taken
/// verdict from a recomputed one because both are right. A planted verdict
/// that disagrees with the fold can: carrying it out is the only way the
/// point was not folded again. This is what resuming *means* — the progress
/// file is believed — so the planted disagreement is the instrument, not a
/// behaviour to rely on.
#[test]
fn a_resumed_point_is_not_folded_again() {
    let dir = checkpoint_dir("skip");
    let (_, lines) = constrained_design();
    let input = result_file(&dir, "front.jsonl", &lines);
    let out = dir.join("checked.jsonl");

    let read = show::parse_lines(&std::fs::read_to_string(&input).unwrap()).unwrap();
    let m = Molecule::from_header(&read.header).unwrap();
    let certified = check::certify(&m, &read.points[..1], None).unwrap();
    let mut planted = check::certified_point(&certified[0].0, &certified[0].1);
    let real = planted["free_mfe"].as_f64().unwrap();
    planted["free_mfe"] = json!(real - 12.5);
    std::fs::write(
        check::progress_path(&out),
        serde_json::to_string(&planted).unwrap() + "\n",
    )
    .unwrap();

    let argv = words(&[
        input.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--checkpoint",
    ]);
    check::run(&check::parse(argv).unwrap()).unwrap();
    let points = show::parse_lines(&std::fs::read_to_string(&out).unwrap())
        .unwrap()
        .points;
    assert_eq!(
        points[0]["free_mfe"].as_f64().unwrap(),
        real - 12.5,
        "the resumed point was folded again instead of taken from the progress file"
    );
}

/// A progress file of another run is refused rather than trusted.
///
/// The one error resuming cannot recover from is certifying the wrong
/// molecule, so the instrument has to be able to fail: a verdict is matched to
/// a point by its sequence, and a sequence this result does not hold means the
/// file beside `--out` belongs to something else.
#[test]
fn a_progress_file_of_another_result_is_refused() {
    let dir = checkpoint_dir("foreign");
    let (_, mine) = design_in_cell(CELLS[0]);
    let (_, theirs) = design_in_cell(CELLS[1]);
    let input = result_file(&dir, "mine.jsonl", &mine);
    let out = dir.join("checked.jsonl");

    // Their first point, with the fields a verdict carries, under my --out.
    let read = show::parse_lines(
        &theirs
            .iter()
            .map(|l| serde_json::to_string(l).unwrap() + "\n")
            .collect::<String>(),
    )
    .unwrap();
    let m = Molecule::from_header(&read.header).unwrap();
    let certified = check::certify(&m, &read.points[..1], None).unwrap();
    let foreign = check::certified_point(&certified[0].0, &certified[0].1);
    std::fs::write(
        check::progress_path(&out),
        serde_json::to_string(&foreign).unwrap() + "\n",
    )
    .unwrap();

    let argv = words(&[
        input.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--checkpoint",
    ]);
    let err = check::run(&check::parse(argv).unwrap())
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("another run's progress"),
        "a foreign progress file was not refused: {err}"
    );
}

/// `--checkpoint` has nowhere to keep its verdicts without a file `--out`.
#[test]
fn checkpointing_to_stdout_is_refused() {
    let err = check::parse(words(&["front.jsonl", "--checkpoint"]))
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(err.contains("needs a file there"), "{err}");
}

/// `--force` starts the check over instead of resuming it.
///
/// `--checkpoint` means one thing in every command — save beside `--out`, and
/// rerunning the same command continues — so the way to ask for a clean run is
/// the flag that already means *overwrite what is there*. Both halves are
/// asserted: the planted verdict is **not** carried out, and the progress file
/// it sat in does not survive, since a later resume would otherwise read a
/// mixture of two runs of the same molecule, which keying by sequence cannot
/// detect.
#[test]
fn force_starts_a_check_over_rather_than_resuming_it() {
    let dir = checkpoint_dir("force");
    let (_, lines) = constrained_design();
    let input = result_file(&dir, "front.jsonl", &lines);
    let out = dir.join("checked.jsonl");

    let read = show::parse_lines(&std::fs::read_to_string(&input).unwrap()).unwrap();
    let m = Molecule::from_header(&read.header).unwrap();
    let certified = check::certify(&m, &read.points[..1], None).unwrap();
    let mut planted = check::certified_point(&certified[0].0, &certified[0].1);
    let real = planted["free_mfe"].as_f64().unwrap();
    planted["free_mfe"] = json!(real - 12.5);
    let progress = check::progress_path(&out);
    std::fs::write(&progress, serde_json::to_string(&planted).unwrap() + "\n").unwrap();

    let argv = words(&[
        input.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--checkpoint",
        "--force",
    ]);
    check::run(&check::parse(argv).unwrap()).unwrap();
    let got = show::parse_lines(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(
        got.points[0]["free_mfe"].as_f64().unwrap(),
        real,
        "--force resumed from the progress file instead of starting over"
    );
    assert_eq!(got.header["resumed"], json!(0), "--force reported a resume");
    assert!(
        !progress.exists(),
        "a started-over run left the verdicts it did not resume behind it"
    );
}
