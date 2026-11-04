//! `design` and `front` over a multi-record `--protein` FASTA file.
//!
//! A file of one record runs as it always has. With more than one, `--out`
//! names a directory, created when absent, and each record writes
//! `<id>.jsonl` there, its id the first word of its FASTA header with every
//! character outside `A–Z a–z 0–9 . _ -` written `_`. Its progress files take
//! their names from that path as a single run's do from `--out`, and
//! `--report` and `--front-out` become `<id>.report.<ext>` and
//! `<id>.front.jsonl` beside it.
//!
//! Records run one after another, each with the flags given. A record that
//! fails is reported and the batch goes on to the next; the batch ends with a
//! line naming every record's outcome, and fails when any record did.
//!
//! A directory that holds files is refused unless `--force`, which starts
//! every record over, or `--checkpoint`, under which rerunning the same
//! command skips the records that finished and continues the one that did not.

use std::path::{Path, PathBuf};

use crate::args::Shared;
use crate::design::{self, DesignArgs};
use crate::front::{self, FrontArgs};
use crate::input::{self, NamedSequence};
use crate::{result, summary, Result};

/// What became of one record of a batch.
enum Outcome {
    Done,
    Skipped,
    Failed(String),
}

/// A record's file stem: the first word of its FASTA header, with every
/// character a file name should not carry written `_`.
pub fn sanitized_id(record: &NamedSequence, index: usize) -> String {
    let word = record
        .name
        .as_deref()
        .and_then(|n| n.split_whitespace().next())
        .unwrap_or_default();
    if word.is_empty() {
        return format!("record_{index}");
    }
    word.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// The records of `shared`'s `--protein`, when it is a file of more than one.
fn records(shared: &Shared) -> Result<Option<Vec<NamedSequence>>> {
    match shared.protein.as_deref() {
        Some(value) => input::protein_records(value),
        None => Ok(None),
    }
}

/// Whether `shared`'s `--protein` is a file of more than one record, which
/// makes the run a batch.
pub fn is_batch(shared: &Shared) -> bool {
    records(shared).is_ok_and(|r| r.is_some())
}

/// Run `derna design`, over every record of a multi-record `--protein`.
pub fn run_design(args: &DesignArgs) -> Result<()> {
    let Some(records) = records(&args.shared)? else {
        summary::start();
        design::run(args)?;
        let out = result::destination(args.shared.out.as_deref()).map(Path::to_path_buf);
        summary::print_tables(&[out], args.shared.quiet);
        return Ok(());
    };
    let dir = prepare_directory(&args.shared, args.checkpoint, &records)?;
    run_records(
        &records,
        &dir,
        args.checkpoint && !args.shared.force,
        |record, out| {
            let mut a = args.clone();
            a.shared.protein_record = Some(record.clone());
            a.shared.out = Some(out.to_path_buf());
            Job {
                // `design` writes its result only once every weight has landed, so
                // a result file on disk is a finished record.
                finished: out.is_file(),
                run: Box::new(move || {
                    summary::start();
                    design::run(&a)?;
                    summary::print_tables(&[a.shared.out.clone()], a.shared.quiet);
                    Ok(())
                }),
            }
        },
    )
}

/// Run `derna front`, over every record of a multi-record `--protein`.
pub fn run_front(args: &FrontArgs) -> Result<()> {
    let Some(records) = records(&args.shared)? else {
        summary::start();
        front::run(args)?;
        summary::print_tables(&front::outputs(args)?, args.shared.quiet);
        return Ok(());
    };
    for (flag, given) in [
        ("--resume", args.resume.is_some()),
        ("--resume-search", args.resume_search.is_some()),
        ("--front-from", args.front_from.is_some()),
        ("--narrow-from", args.narrow_from.is_some()),
        ("--anchors-from", args.anchors_from.is_some()),
    ] {
        if given {
            return Err(format!(
                "derna front: {flag} names one molecule's file, and --protein holds {} \
                 records; a batch continues each record with --checkpoint instead",
                records.len()
            )
            .into());
        }
    }
    let dir = prepare_directory(&args.shared, args.checkpoint, &records)?;
    run_records(
        &records,
        &dir,
        args.checkpoint && !args.shared.force,
        |record, out| {
            let mut a = args.clone();
            a.shared.protein_record = Some(record.clone());
            a.shared.out = Some(out.to_path_buf());
            a.report = args
                .report
                .as_deref()
                .map(|r| beside(out, "report", r, "json"));
            a.front_out = args
                .front_out
                .as_deref()
                .map(|f| beside(out, "front", f, "jsonl"));
            Job {
                finished: front::finished(&a, out),
                run: Box::new(move || {
                    // What this record left at and beside its own file, taken
                    // up as a single run's parse takes up what `--out` holds.
                    front::settle_rerun(&mut a)?;
                    summary::start();
                    front::run(&a)?;
                    summary::print_tables(&front::outputs(&a)?, a.shared.quiet);
                    Ok(())
                }),
            }
        },
    )
}

/// `<stem>.<tag>.<ext>` beside `out`, the extension taken from `given` and
/// `default` when it has none.
fn beside(out: &Path, tag: &str, given: &Path, default: &str) -> PathBuf {
    let ext = given
        .extension()
        .map_or_else(|| default.to_string(), |e| e.to_string_lossy().into_owned());
    let stem = out
        .file_stem()
        .map_or_else(Default::default, |s| s.to_string_lossy().into_owned());
    out.with_file_name(format!("{stem}.{tag}.{ext}"))
}

/// The batch's directory, checked and created; duplicate ids are refused
/// before anything runs.
fn prepare_directory(
    shared: &Shared,
    checkpoint: bool,
    records: &[NamedSequence],
) -> Result<PathBuf> {
    let n = records.len();
    let Some(dir) = result::destination(shared.out.as_deref()) else {
        return Err(format!(
            "--protein holds {n} records, so --out must name a directory to write one \
             result per record into"
        )
        .into());
    };
    let mut seen = std::collections::HashMap::new();
    for (i, record) in records.iter().enumerate() {
        let id = sanitized_id(record, i + 1);
        if let Some(first) = seen.insert(id.clone(), i + 1) {
            return Err(format!(
                "--protein: records {first} and {} both have the id {id:?}; each record \
                 writes <id>.jsonl, so ids must differ",
                i + 1
            )
            .into());
        }
    }
    match std::fs::metadata(dir) {
        Ok(m) if !m.is_dir() => {
            return Err(format!(
                "--out {}: --protein holds {n} records, so --out names a directory, and this \
                 is a file",
                dir.display()
            )
            .into())
        }
        Ok(_) => {
            let holds_files = std::fs::read_dir(dir)?.next().is_some();
            if holds_files && !shared.force && !checkpoint {
                return Err(format!(
                    "--out {}: the directory holds files; pass --force to start every record \
                     over, or --checkpoint to continue the batch",
                    dir.display()
                )
                .into());
            }
        }
        Err(_) => std::fs::create_dir_all(dir)
            .map_err(|e| format!("--out {}: cannot create the directory: {e}", dir.display()))?,
    }
    Ok(dir.to_path_buf())
}

/// One record's run, and whether its result is already finished.
struct Job<'a> {
    finished: bool,
    run: Box<dyn FnOnce() -> Result<()> + 'a>,
}

/// Run each record in turn, report each outcome, and fail when any record
/// failed.
fn run_records<'a>(
    records: &[NamedSequence],
    dir: &Path,
    skip_finished: bool,
    mut job: impl FnMut(&NamedSequence, &Path) -> Job<'a>,
) -> Result<()> {
    let n = records.len();
    let mut outcomes = Vec::with_capacity(n);
    for (i, record) in records.iter().enumerate() {
        let id = sanitized_id(record, i + 1);
        let out = dir.join(format!("{id}.jsonl"));
        let Job { finished, run } = job(record, &out);
        let outcome = if skip_finished && finished {
            eprintln!(
                "[{}/{n}] {id}: finished in {}, skipped",
                i + 1,
                out.display()
            );
            Outcome::Skipped
        } else {
            eprintln!("[{}/{n}] {id}", i + 1);
            match run() {
                Ok(()) => Outcome::Done,
                Err(e) => {
                    eprintln!("error: {id}: {e}");
                    Outcome::Failed(e.to_string())
                }
            }
        };
        outcomes.push((id, outcome));
    }
    let failed = outcomes
        .iter()
        .filter(|(_, o)| matches!(o, Outcome::Failed(_)))
        .count();
    let listed: Vec<String> = outcomes
        .iter()
        .map(|(id, o)| match o {
            Outcome::Done => format!("{id} ok"),
            Outcome::Skipped => format!("{id} skipped (finished)"),
            Outcome::Failed(why) => format!("{id} failed ({why})"),
        })
        .collect();
    eprintln!("batch of {n} record(s): {}", listed.join("; "));
    if failed > 0 {
        return Err(format!("{failed} of {n} record(s) failed").into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(name: Option<&str>) -> NamedSequence {
        NamedSequence {
            name: name.map(str::to_string),
            sequence: "MK".into(),
        }
    }

    #[test]
    fn an_id_is_the_first_word_made_safe_for_a_file_name() {
        assert_eq!(
            sanitized_id(&named(Some("sp|P1|A_HUMAN desc")), 1),
            "sp_P1_A_HUMAN"
        );
        assert_eq!(sanitized_id(&named(Some("GLPE.v2-1")), 1), "GLPE.v2-1");
        assert_eq!(sanitized_id(&named(None), 3), "record_3");
    }

    #[test]
    fn a_report_beside_a_record_takes_its_stem() {
        let out = Path::new("/d/A.jsonl");
        assert_eq!(
            beside(out, "report", Path::new("r.json"), "json"),
            Path::new("/d/A.report.json")
        );
        assert_eq!(
            beside(out, "front", Path::new("f"), "jsonl"),
            Path::new("/d/A.front.jsonl")
        );
    }
}
