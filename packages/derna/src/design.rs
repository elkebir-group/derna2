//! `derna design`: the best design at each given trade-off weight λ.
//!
//! Each design minimizes `λ·MFE − (1−λ)·log CAI` exactly over the codon
//! lattice, under the constraint. Under any constraint each design also carries
//! its check against the free fold of its own sequence; `design` reports the
//! check and never alters the design.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use derna_lib::{
    design_composite, BeamError, CompositeSpec, Design, MotifWindow, ScanConfig, UtrProfile,
};
use serde_json::{json, Map, Value};

use crate::args::{shared_help, Flags, Shared};
use crate::free_check::{self, Checked};
use crate::molecule::Molecule;
use crate::result;
use crate::sequence_constraint::{SequenceConstraints, SEQUENCE_HELP};
use crate::Result;

/// `derna design` as parsed.
#[derive(Debug, Default, Clone)]
pub struct DesignArgs {
    pub shared: Shared,
    pub lambdas: Vec<f64>,
    pub sequence: SequenceConstraints,
    /// Save each weight's design beside `--out` as it settles, and design only
    /// the weights a progress file there does not already hold.
    pub checkpoint: bool,
}

/// One λ's answer.
#[derive(Debug)]
pub struct Solved {
    pub lambda: f64,
    pub design: Design,
    /// The check, under any constraint; `None` where there is nothing to check.
    pub checked: Option<Checked>,
}

/// Help text for `derna design`.
pub fn help() -> String {
    let shared = shared_help();
    format!(
        "derna design — the best design at given trade-off weights λ\n\n\
         Usage: derna design --protein P --lambda L[,L...] [options]\n\n\
         Weights\n  \
         --lambda L[,L...]         trade-off weights in (0, 1), comma-separated\n\n\
         {shared}\n\
         Save and resume\n  \
         --checkpoint              save each weight's design to OUT.design.jsonl as\n                            \
         it settles, so running the same command again\n                            \
         designs only the weights a killed run left.\n                            \
         --force starts over instead\n\n\
         {SEQUENCE_HELP}"
    )
}

/// Parse the words after `design`.
pub fn parse(words: impl IntoIterator<Item = String>) -> Result<DesignArgs> {
    let mut flags = Flags::new(words);
    let mut args = DesignArgs::default();
    while let Some(flag) = flags.next_word() {
        if args.shared.take(&flag, &mut flags)? || args.sequence.take(&flag, &mut flags)? {
            continue;
        }
        match flag.as_str() {
            "--lambda" => {
                // One list, so there is one way to ask for several weights.
                if !args.lambdas.is_empty() {
                    return Err("--lambda is given twice; give the weights as one list, \
                                e.g. --lambda 0.2,0.5,0.8"
                        .into());
                }
                args.lambdas = parse_lambdas(&flags.value(&flag)?)?;
            }
            "--checkpoint" => args.checkpoint = true,
            // A flag the spec gives this command but the command cannot run
            // yet says what it waits on, rather than reading as a typo.
            "--strict" | "--max-secs" | "--max-rss-gb" => {
                return Err(format!(
                    "derna design: {flag} would replace a failing design in the same run, \
                     and a design at one weight has no such search — `derna front \
                     --modular` computes the front of designs that pass"
                )
                .into())
            }
            other => {
                return Err(format!("derna design: unknown flag {other:?} (see --help)").into())
            }
        }
    }
    if args.lambdas.is_empty() {
        return Err("derna design: give at least one --lambda".into());
    }
    // A batch's --out is a directory, and each record's file is checked when
    // the batch runs it.
    if args.checkpoint && !crate::batch::is_batch(&args.shared) {
        refuse_a_checkpoint_without_a_file(args.shared.out.as_deref())?;
    }
    args.sequence.validate()?;
    Ok(args)
}

/// The progress file is written beside `--out`, which stdout, a device or a
/// pipe gives no place for.
fn refuse_a_checkpoint_without_a_file(out: Option<&Path>) -> Result<()> {
    match result::destination(out) {
        None => Err(
            "derna design: --checkpoint keeps each weight's design beside --out, so it \
             needs a file there, not stdout"
                .into(),
        ),
        Some(path) if std::fs::metadata(path).is_ok_and(|m| !m.file_type().is_file()) => {
            Err(format!(
                "derna design: --checkpoint keeps each weight's design beside --out, so it \
                 needs a regular file, not {}",
                path.display()
            )
            .into())
        }
        Some(_) => Ok(()),
    }
}

/// A comma-separated list of weights, each in (0, 1), in the order given.
fn parse_lambdas(v: &str) -> Result<Vec<f64>> {
    v.split(',').map(|w| parse_lambda(w.trim())).collect()
}

/// A weight in the open interval (0, 1). The ends are refused: at λ = 0 the
/// objective ignores structure and at λ = 1 it ignores CAI, so neither returns
/// a point of the front reliably.
fn parse_lambda(v: &str) -> Result<f64> {
    match v.parse::<f64>() {
        Ok(l) if l > 0.0 && l < 1.0 => Ok(l),
        Ok(l) => Err(format!(
            "--lambda {l}: the weight must lie strictly between 0 and 1; \
             the ends of the front are `derna front`'s"
        )
        .into()),
        Err(_) => Err(format!("--lambda: cannot read {v:?} as a number").into()),
    }
}

/// Design at every λ, in the order given, under no sequence constraint, with
/// the exact check.
pub fn solve(m: &Molecule, lambdas: &[f64], threads: usize) -> Result<Vec<Solved>> {
    solve_constrained(m, lambdas, threads, &SequenceConstraints::default(), None)
}

/// Design at every λ, in the order given.
///
/// A motif set is forbidden over the coding sequence by the overlapping
/// reading, so a pattern cannot be spelled across a flank's boundary; a pin
/// keeps, at its residue, exactly the codons that spell it.
pub fn solve_constrained(
    m: &Molecule,
    lambdas: &[f64],
    threads: usize,
    sequence: &SequenceConstraints,
    beam_first: Option<usize>,
) -> Result<Vec<Solved>> {
    let mut solved = Vec::with_capacity(lambdas.len());
    solve_each(
        m,
        (lambdas, |_| false),
        threads,
        sequence,
        beam_first,
        |s| {
            solved.push(s);
            Ok(())
        },
    )?;
    Ok(solved)
}

/// Design at every λ of `lambdas` that `skip` does not take, in the order
/// given, handing each design to `each` as it settles.
fn solve_each(
    m: &Molecule,
    (lambdas, skip): (&[f64], impl Fn(f64) -> bool),
    threads: usize,
    sequence: &SequenceConstraints,
    beam_first: Option<usize>,
    mut each: impl FnMut(Solved) -> Result<()>,
) -> Result<()> {
    let params = m.params.clone();
    let (utr5, utr3) = (m.utr5_spec(), m.utr3_spec());
    let table = m.table();
    // `design` enforces one window over the set, so a pattern carrying its own
    // is refused rather than quietly widened to the set's. The front walk is
    // where a per-pattern window is enforced.
    if sequence.is_scoped() {
        return Err(
            "a motif's own window (--motif PATH@LO..HI, or a `@window` line in a \
                    set file) is enforced by `derna front`; `derna design` forbids its \
                    whole set over one window, named by --motif-window"
                .into(),
        );
    }
    let motifs: Vec<String> = sequence.patterns().iter().map(|p| p.to_string()).collect();
    let motif_window = if motifs.is_empty() {
        MotifWindow::all()
    } else {
        sequence.window(m)?
    };
    let positional = sequence
        .positional()?
        .resolve(&m.layout())
        .map_err(|e| format!("--pin: {e}"))?;
    let enforced_over = sequence.enforced_over();
    for (side, motif, at) in sequence.unremovable(m)? {
        eprintln!(
            "warning: {motif} occurs in the fixed {side} at position {at}; no codon \
             assignment can remove it, and the motifs are forbidden over {enforced_over}"
        );
    }
    let spec = CompositeSpec {
        utr5: &utr5,
        utr3: &utr3,
        constraints: table,
        profile: UtrProfile::inert(),
        motifs: &motifs,
        motif_window,
        flank_complement: None,
        complement_classes: Some(&m.constraint.forbid),
        positional: &positional,
    };
    let config = ScanConfig {
        threads,
        ..ScanConfig::default()
    };
    for (i, &lambda) in lambdas.iter().enumerate() {
        if skip(lambda) {
            continue;
        }
        let t = Instant::now();
        let design = design_composite(
            &params,
            &m.designed_protein(),
            &m.codons.usage,
            lambda,
            &spec,
            config,
        )
        .map_err(|e| infeasible(e, &motifs))?;
        let checked = if m.constraint.is_inert() {
            None
        } else {
            Some(free_check::check(
                &params,
                &design,
                m.cds(),
                table,
                beam_first,
            )?)
        };
        eprintln!(
            "{}",
            progress_line(i, lambdas.len(), lambda, &design, checked.as_ref(), t)
        );
        each(Solved {
            lambda,
            design,
            checked,
        })?;
    }
    Ok(())
}

/// Name the constraint that emptied the design space.
///
/// Only a sequence constraint can do it: a region table always admits the
/// all-unpaired structure, so a run that comes back infeasible came back
/// infeasible because of the motifs or the pins, and the message says which
/// set to look at rather than reporting an empty lattice.
fn infeasible(e: BeamError, motifs: &[String]) -> Box<dyn std::error::Error> {
    match e {
        BeamError::Infeasible if motifs.is_empty() => {
            "no encoding of the protein satisfies the pins".into()
        }
        BeamError::Infeasible => format!(
            "no encoding of the protein avoids every motif ({}) over the coding sequence",
            motifs.join(", ")
        )
        .into(),
        // A pin that leaves a residue with no codon is named by the library,
        // residue and coordinate included; the flag is added so the message
        // names what the caller wrote as well as what it did.
        BeamError::NoFeasibleDesign(reason) => format!("--pin: {reason}").into(),
        other => other.into(),
    }
}

fn progress_line(
    i: usize,
    n: usize,
    lambda: f64,
    d: &Design,
    checked: Option<&Checked>,
    t: Instant,
) -> String {
    let check = match checked {
        None => String::new(),
        Some(Checked::Exact(g)) if g.is_modular() => "  modular".into(),
        Some(Checked::Exact(g)) => format!(
            "  violates the constraint: free MFE {:+.2}, gap {:.2}, {} forbidden pair(s)",
            g.free_mfe_kcal_mol,
            g.gap_kcal_mol,
            g.reinstated_pairs.len()
        ),
        // A refutation is a bound, and the line says so: the exact free MFE
        // was never computed.
        Some(Checked::Refuted {
            free_mfe_at_most,
            gap_at_least,
        }) => format!(
            "  violates the constraint: free MFE at most {free_mfe_at_most:+.2}, \
             gap at least {gap_at_least:.2} (refuted by the beam)"
        ),
    };
    format!(
        "[{}/{n}] λ={lambda} mfe={:+.2} cai={:.4} ({:.2}s){check}",
        i + 1,
        d.mfe_kcal_mol,
        d.cai,
        t.elapsed().as_secs_f64()
    )
}

/// Run `derna design`.
pub fn run(args: &DesignArgs) -> Result<()> {
    let started = Instant::now();
    // `design` solves one weight and writes one design, so it has nothing to
    // do with a front per stop codon.
    if !crate::molecule::stop_arms(args.shared.stop.as_deref())?.is_empty() {
        return Err(format!(
            "derna design: --stop {} asks for a front per stop codon, which is for derna \
             front; design solves one weight, so give one of UAA, UAG, UGA, free or none",
            args.shared.stop.as_deref().unwrap_or_default()
        )
        .into());
    }
    result::check_output(args.shared.out.as_deref(), args.shared.force)?;
    let m = Molecule::resolve(&args.shared)?;
    let threads = args.shared.resolved_threads();
    let progress = result::destination(args.shared.out.as_deref())
        .filter(|_| args.checkpoint)
        .map(progress_path);
    let run_line = progress_run_line(&m, args)?;
    // `--force` starts over: a progress file left beside it would otherwise be
    // resumed by the next run without it, mixing two runs' designs.
    if let (Some(path), true) = (&progress, args.shared.force) {
        remove_progress(path)?;
    }
    let done = match &progress {
        Some(path) if path.exists() => read_progress(path, &run_line, &args.lambdas)?,
        _ => HashMap::new(),
    };
    if !done.is_empty() {
        eprintln!("resuming {} weight(s) already designed", done.len());
    }
    let log = progress
        .as_deref()
        .map(|path| ProgressLog::open(path, &run_line))
        .transpose()?;
    let mut fresh: HashMap<u64, Value> = HashMap::new();
    solve_each(
        &m,
        (&args.lambdas, |l: f64| done.contains_key(&l.to_bits())),
        threads,
        &args.sequence,
        args.shared.beam_first,
        |s| {
            // The line is built where the design settles, so the progress
            // file holds exactly the line the result will.
            let line = result::checked_point(&s.design, Some(s.lambda), true, s.checked.as_ref());
            if let Some(log) = &log {
                log.record(&line);
            }
            fresh.insert(s.lambda.to_bits(), line);
            Ok(())
        },
    )?;
    // The rows in the order --lambda gives them, whichever run designed each.
    let rows: Vec<Value> = args
        .lambdas
        .iter()
        .map(|l| {
            let key = l.to_bits();
            done.get(&key)
                .or_else(|| fresh.get(&key))
                .cloned()
                .expect("every weight was designed or resumed")
        })
        .collect();
    let resumed = args
        .lambdas
        .iter()
        .filter(|l| done.contains_key(&l.to_bits()))
        .count();
    // A closing summary, as `front` and `check` give: the per-λ lines above
    // scroll, and a caller wants the count and how many passed their check
    // without reading them back.
    let checked = rows.iter().filter(|r| r.get("modular").is_some()).count();
    let optimal = rows
        .iter()
        .filter(|r| r["modular"] == Value::Bool(true))
        .count();
    eprintln!(
        "{} design(s){}",
        rows.len(),
        match checked {
            0 => ", unconstrained — no check to give".to_string(),
            _ => format!(", {optimal} of {checked} modular"),
        }
    );
    let mut lines = vec![design_header(&m, args, threads, started, resumed)?];
    lines.extend(rows);
    write(&lines, args.shared.out.as_deref(), args.shared.force)?;
    // The result holds every design the progress file does, so the progress
    // file has nothing left to protect; a later run of the same --out would
    // otherwise resume from a run that has already landed.
    if let Some(path) = &progress {
        if let Err(err) = std::fs::remove_file(path) {
            eprintln!("derna design: {} was left behind: {err}", path.display());
        }
    }
    Ok(())
}

/// Where `--checkpoint` keeps the designs a run has settled: beside `out`, as
/// `<out>.design.jsonl`.
pub fn progress_path(out: &Path) -> PathBuf {
    let mut name = out.as_os_str().to_owned();
    name.push(".design.jsonl");
    PathBuf::from(name)
}

fn remove_progress(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(format!(
            "derna design: --force cannot remove {}: {e}",
            path.display()
        )
        .into()),
        _ => Ok(()),
    }
}

/// The first line of a progress file: what decides a design besides its
/// weight — the molecule, the sequence constraints and the check's beam — so a
/// progress file of another run is told apart from this one's.
fn progress_run_line(m: &Molecule, args: &DesignArgs) -> Result<Value> {
    let mut run = result::molecule_fields(m);
    run.extend(args.sequence.header_fields(m)?);
    run.insert("beam_first".into(), json!(args.shared.beam_first));
    Ok(json!({"record": "run", "command": "design", "run": run}))
}

/// A run's designs as they settle: one point line per weight, appended and
/// written through, so a run killed at any moment leaves behind every weight
/// it had designed rather than none of them.
struct ProgressLog {
    file: Mutex<std::fs::File>,
}

impl ProgressLog {
    /// Append to `path`, creating it with `run_line` first.
    fn open(path: &Path, run_line: &Value) -> Result<Self> {
        let fresh = std::fs::metadata(path).map_or(true, |m| m.len() == 0);
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("derna design: cannot open {}: {e}", path.display()))?;
        let log = Self {
            file: Mutex::new(file),
        };
        if fresh {
            log.record(run_line);
        }
        Ok(log)
    }

    /// Write one line in one write. A failed write loses that line of
    /// progress, never the run, so it is reported and the run goes on.
    fn record(&self, line: &Value) {
        let mut file = self.file.lock().expect("the progress log");
        if let Err(err) = file.write_all(format!("{line}\n").as_bytes()) {
            eprintln!("derna design: a progress line was not written: {err}");
        }
    }
}

/// Read the designs `--checkpoint` wrote, by the exact bits of each weight.
///
/// Keyed by the weight rather than by position, so a rerun whose --lambda list
/// is written in another order still takes up what was designed. The first
/// line must name this run — molecule, sequence constraints and check — and
/// every weight must be one this run asks for; anything else is another run's
/// progress and is refused, because the one error this cannot recover is
/// returning another molecule's design. The last line may have been cut short
/// by the kill that ended the run, and is then left out.
fn read_progress(path: &Path, run_line: &Value, lambdas: &[f64]) -> Result<HashMap<u64, Value>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("derna design: cannot read {}: {e}", path.display()))?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let foreign = |why: String| -> Box<dyn std::error::Error> {
        format!(
            "derna design: {} is another run's progress: {why}; pass --force to start over",
            path.display()
        )
        .into()
    };
    let mut done = HashMap::new();
    for (k, line) in lines.iter().enumerate() {
        let v: Value = match crate::result::parse_line(line) {
            Ok(v) => v,
            Err(_) if k + 1 == lines.len() && !text.ends_with('\n') => break,
            Err(e) => {
                return Err(format!("derna design: {} line {}: {e}", path.display(), k + 1).into())
            }
        };
        if k == 0 {
            if v["record"] != "run" || v["command"] != "design" {
                return Err(foreign("it does not start with a design run's line".into()));
            }
            if let Some(field) = differing_field(&v["run"], &run_line["run"]) {
                return Err(foreign(format!("its {field} differs from this run's")));
            }
            continue;
        }
        let lambda = v["lambda"]
            .as_f64()
            .ok_or_else(|| foreign(format!("line {} is not a design", k + 1)))?;
        if !lambdas.iter().any(|l| l.to_bits() == lambda.to_bits()) {
            return Err(foreign(format!(
                "line {} is at λ={lambda}, which this run does not ask for",
                k + 1
            )));
        }
        done.insert(lambda.to_bits(), v);
    }
    Ok(done)
}

/// The first field on which two run descriptions differ, either way round.
fn differing_field(theirs: &Value, ours: &Value) -> Option<String> {
    let (Value::Object(theirs), Value::Object(ours)) = (theirs, ours) else {
        return Some("run".into());
    };
    ours.iter()
        .find(|(k, v)| theirs.get(k.as_str()) != Some(v))
        .or_else(|| theirs.iter().find(|(k, _)| !ours.contains_key(k.as_str())))
        .map(|(k, _)| k.clone())
}

fn design_header(
    m: &Molecule,
    args: &DesignArgs,
    threads: usize,
    started: Instant,
    resumed: usize,
) -> Result<Value> {
    let mut extra = Map::new();
    extra.insert("lambdas".into(), json!(args.lambdas));
    extra.insert("threads".into(), threads.into());
    extra.insert("deterministic".into(), args.shared.deterministic.into());
    // How many designs this run took back rather than computed, so `wall_s`
    // is not read as the cost of the whole file.
    if args.checkpoint {
        extra.insert("resumed".into(), resumed.into());
    }
    if let Some(width) = args.shared.beam_first {
        extra.insert("beam_first".into(), width.into());
    }
    // The sequence constraints the run was under. A later command has to know:
    // a design made under motifs or pins may be certified but not searched
    // from, and nothing else in the file says so.
    extra.extend(args.sequence.header_fields(m)?);
    extra.insert("wall_s".into(), started.elapsed().as_secs_f64().into());
    extra.insert("peak_rss_kb".into(), json!(result::peak_rss_kb()));
    Ok(result::header("design", m, extra))
}

fn write(lines: &[Value], out: Option<&Path>, force: bool) -> Result<()> {
    let mut w = result::open_output(out, force)?;
    result::write_lines(w.as_mut(), lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &[&str]) -> Vec<String> {
        s.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn lambda_takes_a_list_of_open_weights() {
        let a = parse(words(&["--protein", "MK", "--lambda", "0.2,0.5, 0.8"])).unwrap();
        assert_eq!(a.lambdas, [0.2, 0.5, 0.8]);
        let twice = parse(words(&[
            "--protein",
            "MK",
            "--lambda",
            "0.2",
            "--lambda",
            "0.8",
        ]))
        .unwrap_err()
        .to_string();
        assert!(twice.contains("one list"), "{twice}");
        for bad in ["0", "1", "1.5", "x", "0.2,", "0.2,1"] {
            assert!(
                parse(words(&["--protein", "MK", "--lambda", bad])).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn a_design_needs_a_lambda() {
        assert!(parse(words(&["--protein", "MK"])).is_err());
    }

    #[test]
    fn an_instrument_flag_is_unknown() {
        let e = parse(words(&["--lambda", "0.5", "--mode", "beam"]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("unknown flag \"--mode\""), "{e}");
    }
}
