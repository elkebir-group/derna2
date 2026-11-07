//! `derna check`: certify a result's points against their own free folds.
//!
//! The command takes a result file and needs nothing else: the header carries
//! the protein, the flanks, the constraint and the codon table, so a check
//! cannot be pointed at other flanks than the result was produced with.
//!
//! `--designs` is for designs that carry no such header, because this run did
//! not make them: the molecule comes from the command line instead, and every
//! sequence is refused unless it is a molecule of that description. See
//! [`crate::given`].
//!
//! Certifying a point is one free fold of the whole molecule against the
//! point's constrained energy — the same check `design` reports. A failing
//! point is reported, not replaced.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use derna_lib::Design;
use serde_json::{json, Map, Value};

use crate::args::{Flags, Shared};
use crate::free_check::{self, Checked};
use crate::given;
use crate::modular_front;
use crate::molecule::Molecule;
use crate::result;
use crate::show;
use crate::Result;

/// `derna check` as parsed.
#[derive(Debug, Default)]
pub struct CheckArgs {
    /// The result file, or `None` for stdin.
    pub input: Option<PathBuf>,
    /// `--designs PATH`: score the sequences in `PATH` as molecules of the
    /// description the molecule flags give, in place of a result file.
    pub designs: Option<PathBuf>,
    /// Keep every point's verdict as it settles, and take back the verdicts an
    /// earlier run of the same `--out` left.
    pub checkpoint: bool,
    pub shared: Shared,
}

/// Help text for `derna check`.
///
/// `--beam-first` is named only by a build that has it. A help text that
/// advertises a flag the parser refuses sends a reader to write a command line
/// that fails, which is the same defect as an undocumented flag pointing the
/// other way.
pub fn help() -> String {
    "derna check — certify a result's points against their own free folds\n\n\
     Usage: derna check RESULT.jsonl|- [options]\n\n\
     A point passes, and is modular, when its free MFE equals its constrained\n\
     energy; a failing point keeps its gap and the forbidden pairs its free fold\n\
     forms.\n\n\
     Execution\n  \
     --threads N               certify this many points at a time; the result\n                            \
     is the same at any count [default: 1]\n  \
     --params PATH             load energy parameters from a file [default:\n                            \
     the built-in Turner 2004 table]\n  \
     --out PATH                JSONL result [default: stdout]\n  \
     --force                   overwrite an existing --out\n  \
     --quiet                   do not print the table of what was written to\n                            \
     stderr at the end of the run\n\n\
     Designs this run did not make\n  \
     --designs PATH            score the sequences in PATH (FASTA, or one per\n                            \
     line) in place of a result file. They have no\n                            \
     header to read the molecule from, so it comes\n                            \
     from the molecule flags instead: --protein\n                            \
     (required), --utr5, --utr3, --stop, --forbid\n                            \
     and --codon-usage. A sequence that is not a\n                            \
     molecule of that description is refused\n\n\
     Save and resume\n  \
     --checkpoint              save each point's verdict to OUT.check.jsonl as\n                            \
     it settles, so running the same command again\n                            \
     folds only the points a killed run left.\n                            \
     --force starts over instead\n"
        .to_string()
}

/// Parse the words after `check`.
pub fn parse(words: impl IntoIterator<Item = String>) -> Result<CheckArgs> {
    let mut flags = Flags::new(words);
    let mut args = CheckArgs::default();
    while let Some(word) = flags.next_word() {
        if args.shared.take(&word, &mut flags)? {
            continue;
        }
        match word.as_str() {
            "--checkpoint" => args.checkpoint = true,
            "--designs" => {
                args.designs = Some(PathBuf::from(flags.value("--designs")?));
            }
            "--report" | "--max-secs" | "--max-rss-gb" => {
                return Err(format!(
                    "derna check: {word} describes a search, and this command does not \
                     search; certifying every point is one free fold each"
                )
                .into())
            }
            flag if flag.starts_with("--") => {
                return Err(format!("derna check: unknown flag {flag:?} (see --help)").into())
            }
            path if args.input.is_none() => args.input = Some(PathBuf::from(path)),
            extra => {
                return Err(format!("derna check: one result file at a time, got {extra:?}").into())
            }
        }
    }
    match (&args.input, &args.designs) {
        (None, None) => {
            return Err(
                "derna check: give a result file, or `-` for stdin, or --designs PATH \
                 with the molecule they are designs of"
                    .into(),
            )
        }
        (Some(input), Some(_)) => {
            return Err(format!(
                "derna check: --designs scores designs this run did not make, so there is \
                 no result file to check beside them; got {input:?} as well"
            )
            .into())
        }
        _ => {}
    }
    refuse_what_this_command_cannot_do(&args)?;
    Ok(args)
}

/// Refuse a flag this command cannot act on, each with its reason.
///
/// Two kinds: the molecule's flags, which the result file's header carries,
/// and determinism, which a certification already has.
fn refuse_what_this_command_cannot_do(args: &CheckArgs) -> Result<()> {
    // With --designs there is no header to contradict: the molecule flags are how the
    // command learns what the sequences are designs of, and they are required rather than
    // refused. What protects against scoring the wrong molecule there is that every
    // sequence must be one of that description (`crate::given`).
    if args.designs.is_some() {
        if args.shared.protein.is_none() {
            return Err(
                "derna check: --designs needs --protein, the molecule the sequences are \
                 designs of"
                    .into(),
            );
        }
        return refuse_what_no_input_shape_can_do(args);
    }
    for (flag, given) in [
        ("--protein", args.shared.protein.is_some()),
        ("--utr5", args.shared.utr5.is_some()),
        ("--utr3", args.shared.utr3.is_some()),
        ("--stop", args.shared.stop.is_some()),
        ("--forbid", args.shared.forbid.is_some()),
        ("--access", args.shared.access.is_some()),
        ("--codon-usage", args.shared.codon_usage.is_some()),
    ] {
        if given {
            return Err(format!(
                "derna check: {flag} is the result file's, not this command's — the header \
                 carries it, so a check cannot be pointed at a different molecule"
            )
            .into());
        }
    }
    refuse_what_no_input_shape_can_do(args)
}

/// The refusals that hold whichever shape the input takes: where a checkpoint may go,
/// and determinism, which a certification already has.
fn refuse_what_no_input_shape_can_do(args: &CheckArgs) -> Result<()> {
    match result::destination(args.shared.out.as_deref()) {
        None if args.checkpoint => {
            return Err(
                "derna check: --checkpoint keeps its verdicts beside --out, so it \
                        needs a file there"
                    .into(),
            );
        }
        // The progress file is written beside --out, which a device or a pipe
        // gives no place for.
        Some(path)
            if args.checkpoint
                && std::fs::metadata(path).is_ok_and(|m| !m.file_type().is_file()) =>
        {
            return Err(format!(
                "derna check: --checkpoint keeps its verdicts beside --out, so it needs a \
                 regular file, not {}",
                path.display()
            )
            .into());
        }
        _ => {}
    }
    if args.shared.deterministic {
        return Err(
            "derna check: --deterministic asks for what a certification already is — \
             each point's check reads only its own sequence, and the results are written \
             in the input's order at any --threads"
                .into(),
        );
    }
    Ok(())
}

/// A point rebuilt as a design, so the check sees what it sees elsewhere —
/// including the emitted structure, which it re-scores.
fn design_of(point: &Value) -> Result<Design> {
    let field = |name: &str| -> Result<f64> {
        point[name]
            .as_f64()
            .ok_or_else(|| format!("a point has no {name}").into())
    };
    let text = |name: &str| -> Result<String> {
        Ok(point[name]
            .as_str()
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                format!("a point has no {name}").into()
            })?
            .to_string())
    };
    Ok(Design {
        sequence: text("sequence")?,
        structure: text("structure")?,
        mfe_kcal_mol: field("mfe")?,
        cai_log: field("cai_log")?,
        cai: field("cai")?,
        // Zero because this command designs no flank position: a preference
        // over designed flanks is the one thing a `derna` result cannot
        // carry, so there is no field to lose here.
        utr_profile_log: 0.0,
    })
}

/// Certify every point against its own free fold, on one thread.
pub fn certify(
    m: &Molecule,
    points: &[Value],
    beam_first: Option<usize>,
) -> Result<Vec<(Value, Checked)>> {
    certify_with(m, points, beam_first, 1)
}

/// Certify every point against its own free fold.
///
/// A point's check folds **its own** sequence and reads nothing from
/// the other points, so the work divides with no shared state. `threads` only
/// changes how long it takes: the results are collected back in the input's
/// order, so the output is the same at any count — which is what lets a
/// certification be both parallel and reproducible, unlike a design, where
/// the order ties are merged in can pick a different spelling.
pub fn certify_with(
    m: &Molecule,
    points: &[Value],
    beam_first: Option<usize>,
    threads: usize,
) -> Result<Vec<(Value, Checked)>> {
    let params = m.params.clone();
    let table = m.table();
    let designs: Vec<Design> = points.iter().map(design_of).collect::<Result<_>>()?;
    let cds = m.cds();
    // A worker's result crosses a thread boundary, and the error a check
    // returns is not `Send`; its message is what a caller reads anyway.
    let one = |design: &Design| -> std::result::Result<Checked, String> {
        free_check::check(&params, design, cds, table, beam_first).map_err(|e| e.to_string())
    };

    let checks: Vec<Checked> = if threads <= 1 || designs.len() < 2 {
        designs
            .iter()
            .map(|design| one(design).map_err(Into::into))
            .collect::<Result<_>>()?
    } else {
        // Each worker takes every `threads`-th point, so a long point and a
        // short one do not land on one worker by position; the slots are
        // written back by index, so the order is the input's.
        let workers = threads.min(designs.len());
        let mut slots: Vec<Option<std::result::Result<Checked, String>>> =
            (0..designs.len()).map(|_| None).collect();
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for worker in 0..workers {
                let designs = &designs;
                handles.push(scope.spawn(move || {
                    (worker..designs.len())
                        .step_by(workers)
                        .map(|at| (at, one(&designs[at])))
                        .collect::<Vec<_>>()
                }));
            }
            for handle in handles {
                for (at, checked) in handle.join().expect("a certifying worker") {
                    slots[at] = Some(checked);
                }
            }
        });
        slots
            .into_iter()
            .map(|slot| slot.expect("every point was certified").map_err(Into::into))
            .collect::<Result<_>>()?
    };
    Ok(points.iter().cloned().zip(checks).collect())
}

/// [`certify_with`] without a beam, each free fold copying what it can from
/// the worker's last `keep` folds ([`free_check::check_reusing`]).
///
/// Reuse pays only between neighbours, so each worker takes runs of
/// consecutive points rather than every `threads`-th one; the runs are dealt
/// round-robin, about four to a worker, so a stretch of the front that checks
/// slowly is shared out rather than landing on one worker. The verdicts are
/// [`certify_with`]'s, in the input's order.
pub fn certify_reusing(
    m: &Molecule,
    points: &[Value],
    threads: usize,
    keep: usize,
) -> Result<Vec<(Value, Checked)>> {
    let params = m.params.clone();
    let table = m.table();
    let designs: Vec<Design> = points.iter().map(design_of).collect::<Result<_>>()?;
    let cds = m.cds();
    let n = designs.len();
    let workers = threads.clamp(1, n.max(1));
    let run = n.div_ceil(4 * workers).max(1);
    let runs = n.div_ceil(run);
    let mut slots: Vec<Option<std::result::Result<Checked, String>>> =
        (0..n).map(|_| None).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|worker| {
                let (designs, params) = (&designs, &params);
                scope.spawn(move || {
                    let mut pool = std::collections::VecDeque::new();
                    (worker..runs)
                        .step_by(workers)
                        .flat_map(|r| r * run..((r + 1) * run).min(n))
                        .map(|at| {
                            let checked = free_check::check_reusing(
                                params,
                                &designs[at],
                                cds,
                                table,
                                &mut pool,
                                keep,
                            )
                            .map_err(|e| e.to_string());
                            (at, checked)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for handle in handles {
            for (at, checked) in handle.join().expect("a certifying worker") {
                slots[at] = Some(checked);
            }
        }
    });
    let checks: Vec<Checked> = slots
        .into_iter()
        .map(|slot| slot.expect("every point was certified").map_err(Into::into))
        .collect::<Result<_>>()?;
    Ok(points.iter().cloned().zip(checks).collect())
}

/// One certified point: its own fields, the check's, and its status.
///
/// Public so that a test can assert the shape a checked file carries, which
/// is also the shape `show` renders.
pub fn certified_point(point: &Value, checked: &Checked) -> Value {
    let mut line = point.clone();
    if let Value::Object(fields) = &mut line {
        // The check's fields replace any the input carried: this run folded
        // the molecule, and a stale verdict beside a fresh one would leave a
        // reader to guess which fold produced which.
        for stale in [
            "free_mfe",
            "free_structure",
            "gap",
            "modular",
            "violations",
            "fixed_violations",
            "free_mfe_at_most",
            "gap_at_least",
            // `status` carries two senses of `certified`. A modular
            // front writes it for a point no open region can hold a
            // dominator of; this command writes it for a point that passed its
            // own check. Leaving the input's behind would let a point this run
            // refuted keep a `certified` from the other sense, in the column
            // `show` documents as this one's.
            "status",
        ] {
            fields.remove(stale);
        }
        fields.extend(result::check_fields(checked));
        // `mfe` is the energy the gap is measured against, not a free-standing
        // description of the molecule, so it is the check's to write too. Where
        // the input's energy could not stand for the sequence's constrained
        // optimum, the gap was measured against the optimum the scan found, and
        // leaving the input's number beside it emits a row whose own
        // `mfe - free_mfe` is not its `gap`. A refutation is only taken where a
        // fold agreed with the input, so that case writes the same value back.
        if let Checked::Exact(gap) = checked {
            fields.insert("mfe".into(), gap.constrained_mfe_kcal_mol.into());
        }
        // `certified` is the status of a point that passed; a failing point is
        // left without one, since certification does not decide what replaces it.
        if checked.modular() {
            fields.insert("status".into(), "certified".into());
        }
    }
    line
}

/// Where `--checkpoint` keeps the verdicts a check has settled: beside `out`,
/// as `<out>.check.jsonl`.
pub fn progress_path(out: &Path) -> PathBuf {
    let mut name = out.as_os_str().to_owned();
    name.push(".check.jsonl");
    PathBuf::from(name)
}

/// A check's verdicts as they settle: one certified point per line, appended
/// and written through, so a run killed at any moment leaves behind every
/// point it had folded rather than none of them.
///
/// A check's points are independent, so a line is complete the moment its
/// fold is: there is no state to reconcile and nothing to write at the end
/// that makes an earlier line mean something different.
struct ProgressLog {
    file: Mutex<std::fs::File>,
}

impl ProgressLog {
    /// Append to `path`, creating it — or, `fresh`, replace what it holds.
    ///
    /// A run starting over must not leave the verdicts it is not resuming
    /// behind it: a later run would then resume from a mixture of two runs,
    /// which is the one thing keying by sequence cannot detect, both runs
    /// being of the same molecule.
    fn open(path: &Path, fresh: bool) -> Result<Self> {
        let mut options = std::fs::OpenOptions::new();
        options.create(true);
        if fresh {
            options.write(true).truncate(true);
        } else {
            options.append(true);
        }
        let file = options
            .open(path)
            .map_err(|e| format!("derna check: cannot open {}: {e}", path.display()))?;
        Ok(Self {
            file: Mutex::new(file),
        })
    }

    /// Write one settled point. A failed write loses that line of progress,
    /// never the run, so it is reported and the check goes on.
    fn record(&self, line: &Value) {
        let mut file = self.file.lock().expect("the progress log");
        if let Err(err) = writeln!(file, "{line}") {
            eprintln!("derna check: a progress line was not written: {err}");
        }
    }
}

/// Read the verdicts `--checkpoint` wrote, by the sequence each is of.
///
/// Keyed by sequence rather than by position, because a point's check reads
/// only its own sequence: a verdict is the same wherever in the input that
/// sequence sits, so progress survives an input written out again in another
/// order. A line without a sequence, or whose sequence the input does not
/// hold, is a progress file of another run and is refused — the one error
/// this cannot recover is silently certifying the wrong molecule. The last
/// line may have been cut short by the kill that ended the run, and is then
/// left out.
pub fn read_progress(path: &Path, of: &HashMap<&str, usize>) -> Result<HashMap<String, Value>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("derna check: cannot read {}: {e}", path.display()))?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut done = HashMap::new();
    for (k, line) in lines.iter().enumerate() {
        let v: Value = match crate::result::parse_line(line) {
            Ok(v) => v,
            Err(_) if k + 1 == lines.len() && !text.ends_with('\n') => break,
            Err(e) => {
                return Err(format!("derna check: {} line {}: {e}", path.display(), k + 1).into())
            }
        };
        let sequence = v["sequence"].as_str().ok_or_else(|| {
            format!(
                "derna check: {} line {}: not a checked point",
                path.display(),
                k + 1
            )
        })?;
        if !of.contains_key(sequence) {
            return Err(format!(
                "derna check: {} line {} is of a sequence this result does not hold, so it \
                 is another run's progress",
                path.display(),
                k + 1
            )
            .into());
        }
        done.insert(sequence.to_string(), v);
    }
    Ok(done)
}

/// Certify every point, as the lines a result file carries.
///
/// Each line is built where its fold finishes, so `log` can write it then;
/// a point already in `done` is taken from there and never folded again.
/// The lines come back in the input's order whatever `threads` is and
/// whatever was resumed, so a resumed run's output is the uninterrupted
/// run's output.
fn certify_lines(
    m: &Molecule,
    points: &[Value],
    args: &CheckArgs,
    done: &HashMap<String, Value>,
    log: Option<&ProgressLog>,
) -> Result<Vec<Value>> {
    let params = m.params.clone();
    let table = m.table();
    let designs: Vec<Design> = points.iter().map(design_of).collect::<Result<_>>()?;
    let cds = m.cds();
    // A modular front's points name the violations no codon
    // choice removes; this run's fold decides them again.
    let fixed = modular_front::fixed_bases(m);
    let beam_first = args.shared.beam_first;
    let one = |at: usize| -> std::result::Result<Value, String> {
        if let Some(kept) = done.get(&designs[at].sequence) {
            return Ok(kept.clone());
        }
        let checked = free_check::check(&params, &designs[at], cds, table, beam_first)
            .map_err(|e| e.to_string())?;
        let mut line = certified_point(&points[at], &checked);
        if let (Some(_), Checked::Exact(gap)) = (points[at].get("fixed_violations"), &checked) {
            line["fixed_violations"] =
                modular_front::fixed_violations(&fixed, &gap.reinstated_pairs);
        }
        if let Some(log) = log {
            log.record(&line);
        }
        Ok(line)
    };

    let n = designs.len();
    let threads = args.shared.threads();
    if threads <= 1 || n < 2 {
        return (0..n).map(|at| one(at).map_err(Into::into)).collect();
    }
    // Each worker takes every `threads`-th point, so a long point and a short
    // one do not land on one worker by position; the slots are written back by
    // index, so the order is the input's.
    let workers = threads.min(n);
    let mut slots: Vec<Option<std::result::Result<Value, String>>> = (0..n).map(|_| None).collect();
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            let one = &one;
            handles.push(scope.spawn(move || {
                (worker..n)
                    .step_by(workers)
                    .map(|at| (at, one(at)))
                    .collect::<Vec<_>>()
            }));
        }
        for handle in handles {
            for (at, line) in handle.join().expect("a certifying worker") {
                slots[at] = Some(line);
            }
        }
    });
    slots
        .into_iter()
        .map(|slot| slot.expect("every point was certified").map_err(Into::into))
        .collect()
}

/// Run `derna check`.
pub fn run(args: &CheckArgs) -> Result<()> {
    let started = Instant::now();
    result::check_output(args.shared.out.as_deref(), args.shared.force)?;
    // Either the input carries the molecule and the points, or --designs gives the
    // molecule on the command line and the points are folded from the sequences.
    let (m, header, points) = match &args.designs {
        Some(path) => {
            let m = Molecule::resolve(&args.shared)?;
            let points = given::points(&m.params, &m, path)?;
            let header = json!({
                "command": "designs",
                "designs": path.display().to_string(),
                "points": points.len(),
            });
            (m, header, points)
        }
        None => {
            let read = show::read(args.input.as_deref())?;
            let mut m = Molecule::from_header(&read.header)?;
            m.params = args.shared.energy_params()?;
            (m, read.header, read.points)
        }
    };
    if m.constraint.is_inert() {
        return Err(
            "this result was produced with no constraint, so no point can violate \
                    one and there is nothing to certify"
                .into(),
        );
    }
    // A point's verdict is of its sequence, so progress is matched on that;
    // the position is what the result file's order needs, not the check.
    let at_of: HashMap<&str, usize> = points
        .iter()
        .enumerate()
        .filter_map(|(at, p)| p["sequence"].as_str().map(|s| (s, at)))
        .collect();
    let progress = result::destination(args.shared.out.as_deref())
        .filter(|_| args.checkpoint)
        .map(progress_path);
    // `--checkpoint` resumes on a rerun of the same command; `--force` is how
    // a caller asks for a clean one. One flag, one meaning, and the way to
    // start over is the flag that already means "overwrite what is there".
    let done = match &progress {
        Some(_) if args.shared.force => HashMap::new(),
        Some(path) if path.exists() => read_progress(path, &at_of)?,
        _ => HashMap::new(),
    };
    if !done.is_empty() {
        eprintln!("resuming {} point(s) already certified", done.len());
    }
    let log = progress
        .as_deref()
        .map(|path| ProgressLog::open(path, args.shared.force))
        .transpose()?;

    let certified = certify_lines(&m, &points, args, &done, log.as_ref())?;
    let (optimal, total) = (
        certified
            .iter()
            .filter(|line| line["modular"] == Value::Bool(true))
            .count(),
        certified.len(),
    );
    eprintln!("{optimal} of {total} point(s) modular");

    let mut lines = vec![check_header(
        &m,
        &header,
        args,
        started,
        optimal,
        done.len(),
    )];
    lines.extend(certified);
    let mut w = result::open_output(args.shared.out.as_deref(), args.shared.force)?;
    result::write_lines(w.as_mut(), &lines)?;
    // The result holds every verdict the progress file does, so the progress
    // file has nothing left to protect; a later run of the same --out would
    // otherwise resume from a check that has already landed.
    if let Some(path) = &progress {
        if let Err(err) = std::fs::remove_file(path) {
            eprintln!("derna check: {} was left behind: {err}", path.display());
        }
    }
    Ok(())
}

fn check_header(
    m: &Molecule,
    input: &Value,
    args: &CheckArgs,
    started: Instant,
    optimal: usize,
    resumed: usize,
) -> Value {
    let mut extra = Map::new();
    // What the input was, so a chain of commands is readable from its last
    // file: a checked front records the front it checked.
    extra.insert(
        "checked_from".into(),
        json!({"command": input["command"], "version": input["version"]}),
    );
    // The checked front's own verdict, which this command neither re-establishes
    // nor refutes and nothing else here records. A modular front
    // that stopped on a budget left regions open, so its points are the ones it
    // reached rather than the whole front, and a reader of the checked file has
    // no other way to learn that. It belongs to the front rather than to any one
    // point: a per-point copy cannot survive a re-check, because a `status` this
    // command writes is indistinguishable from the one a front writes.
    if let Some(front) = front_run(input) {
        extra.insert("front_run".into(), front);
    }
    extra.insert("modular".into(), optimal.into());
    extra.insert("threads".into(), args.shared.threads().into());
    // How many verdicts this run took back rather than folded, so `wall_s` is
    // not read as the cost of the whole file.
    if args.checkpoint {
        extra.insert("resumed".into(), resumed.into());
    }
    if let Some(width) = args.shared.beam_first {
        extra.insert("beam_first".into(), width.into());
    }
    // The inputs a check may not change travel from the input's header, so
    // they are carried rather than re-derived where the input recorded them.
    for carried in [
        "motifs",
        "motif_window",
        "motif_scopes",
        "pins",
        "motifs_in_fixed_flanks",
        "lambdas",
        "epsilon",
    ] {
        if let Some(value) = input.get(carried) {
            extra.insert(carried.into(), value.clone());
        }
    }
    extra.insert("wall_s".into(), started.elapsed().as_secs_f64().into());
    extra.insert("peak_rss_kb".into(), json!(result::peak_rss_kb()));
    result::header("check", m, extra)
}

/// The verdict of the front this input came from, if it records one.
///
/// Carried through unchanged when the input is itself a checked file, so that
/// checking a check does not lose the front the chain started from.
fn front_run(input: &Value) -> Option<Value> {
    if let Some(carried) = input.get("front_run") {
        return Some(carried.clone());
    }
    let mut front = Map::new();
    for field in ["complete", "stopped_by", "open", "seeds", "unchecked"] {
        if let Some(value) = input.get(field) {
            front.insert(field.into(), value.clone());
        }
    }
    (!front.is_empty()).then_some(Value::Object(front))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|p| p.to_string()).collect()
    }

    #[test]
    fn a_refuted_point_does_not_keep_the_inputs_certified_status() {
        // A modular front writes `status: certified` for a point no
        // open region can hold a dominator of. That is a different claim from
        // this command's, so a point this run refutes must not carry it out:
        // `show` renders the column as the check's verdict.
        let point = json!({
            "record": "point", "sequence": "GGGG", "mfe": -1.0, "cai": 1.0,
            "cai_log": 0.0, "structure": "....", "status": "certified",
            "modular": true, "gap": 0.0, "free_mfe": -1.0,
        });
        let line = certified_point(
            &point,
            &crate::free_check::Checked::Refuted {
                free_mfe_at_most: -3.0,
                gap_at_least: 2.0,
            },
        );
        assert_eq!(line["modular"], json!(false));
        assert!(
            line.get("status").is_none(),
            "a refuted point kept a status: {}",
            line["status"]
        );
        // The front's own claim is not kept per point. It is about the regions
        // that run left open, which is a property of the front and not of this
        // point, and `front_run` in the header carries it instead.
        assert!(line.get("front_status").is_none());
        // The bound replaces the exact pair the input carried.
        assert!(line.get("gap").is_none() && line.get("free_mfe").is_none());
        assert_eq!(line["gap_at_least"], json!(2.0));
    }

    #[test]
    fn the_checks_own_constrained_energy_replaces_the_inputs_mfe() {
        // `mfe` is the energy the gap is measured against. Where the input's
        // could not stand for the sequence's constrained optimum, the check
        // measured against the optimum it folded, and keeping the input's
        // number emits a row whose own `mfe - free_mfe` is not its `gap` —
        // which is how a reader recovers the margin.
        let point = json!({
            "record": "point", "sequence": "GGGG", "mfe": -1.0, "cai": 1.0,
            "cai_log": 0.0, "structure": "....",
        });
        let gap = derna_lib::FreeFoldGap {
            constrained_mfe_kcal_mol: -4.0,
            free_mfe_kcal_mol: -6.0,
            gap_kcal_mol: 2.0,
            free_structure: "(())".into(),
            reinstated_pairs: vec![(0, 3)],
            classes: derna_lib::ClassBreakdown::default(),
            emitted: None,
        };
        let line = certified_point(&point, &crate::free_check::Checked::Exact(gap));
        // Non-vacuous: the input's -1.0 and the check's -4.0 differ, so an
        // implementation that carried the input through would fail here.
        assert_eq!(line["mfe"], json!(-4.0));
        let (mfe, free, gap) = (
            line["mfe"].as_f64().expect("an energy"),
            line["free_mfe"].as_f64().expect("a free energy"),
            line["gap"].as_f64().expect("a gap"),
        );
        assert!(
            (mfe - free - gap).abs() < 1e-9,
            "the row's own fields disagree: {mfe} - {free} != {gap}"
        );
    }

    #[test]
    fn a_budget_stopped_fronts_verdict_travels_into_the_check() {
        // Every modular front on record stops on its budget with a
        // region open, so its points are what it reached and not the whole
        // front. A checked file that dropped this would read as a complete
        // front's check, and nothing else in the check's header says otherwise.
        let front = json!({
            "record": "header", "command": "front", "complete": false,
            "stopped_by": "time", "open": 1, "seeds": 0, "unchecked": 0,
        });
        let carried = front_run(&front).expect("a front that stopped has a verdict");
        assert_eq!(carried["complete"], json!(false));
        assert_eq!(carried["stopped_by"], json!("time"));
        assert_eq!(carried["open"], json!(1));
        assert_eq!(carried["seeds"], json!(0));
    }

    #[test]
    fn checking_a_checked_file_keeps_the_front_it_started_from() {
        // A check of a check must not lose the front: the chain's first input is
        // the only run whose completeness is in question, and its verdict is
        // carried rather than re-derived from an input that no longer has one.
        let checked = json!({
            "record": "header", "command": "check",
            "front_run": {"complete": false, "stopped_by": "memory", "open": 2},
        });
        let carried = front_run(&checked).expect("a checked file carries the front's verdict");
        assert_eq!(carried["stopped_by"], json!("memory"));
        assert_eq!(carried["open"], json!(2));
    }

    #[test]
    fn an_input_that_claims_nothing_about_a_front_adds_no_field() {
        // A constrained front reports no such verdict, and inventing an absent
        // one would read as a claim the input never made.
        let plain = json!({"record": "header", "command": "front", "points": 117});
        assert!(front_run(&plain).is_none());
    }

    #[test]
    fn the_input_is_positional_and_required() {
        assert_eq!(
            parse(words(&["front.jsonl"]))
                .unwrap()
                .input
                .unwrap()
                .to_str(),
            Some("front.jsonl")
        );
        assert!(parse(words(&[])).is_err());
        assert!(parse(words(&["a.jsonl", "b.jsonl"])).is_err());
    }

    #[test]
    fn a_molecule_flag_is_refused_because_the_header_carries_it() {
        for flag in [
            "--protein",
            "--utr5",
            "--utr3",
            "--stop",
            "--forbid",
            "--access",
            "--codon-usage",
        ] {
            let e = parse(words(&["front.jsonl", flag, "x"]))
                .unwrap_err()
                .to_string();
            assert!(e.contains("the header"), "{flag}: {e}");
        }
    }

    #[test]
    fn threads_are_spent_and_determinism_is_already_had() {
        let a = parse(words(&["front.jsonl", "--threads", "4"])).unwrap();
        assert_eq!(a.shared.threads(), 4);
        // A certification is reproducible at any count, so asking for both
        // says nothing and is refused rather than silently accepted.
        let e = parse(words(&["front.jsonl", "--deterministic"]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("already is"), "{e}");
    }
}
