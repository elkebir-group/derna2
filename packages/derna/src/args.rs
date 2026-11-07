//! The flags every computing subcommand shares, and the cursor that reads them.
//!
//! One parser, one spelling, one unit: energies are kcal/mol on every flag.

use std::collections::VecDeque;
use std::path::PathBuf;

use derna_lib::Params;

use crate::Result;

/// The remaining command-line words.
pub struct Flags {
    words: VecDeque<String>,
}

impl Flags {
    /// Wrap the words after the subcommand.
    pub fn new(words: impl IntoIterator<Item = String>) -> Self {
        Self {
            words: words.into_iter().collect(),
        }
    }

    /// The next word, if any.
    pub fn next_word(&mut self) -> Option<String> {
        self.words.pop_front()
    }

    /// The value following `flag`, or a refusal naming it.
    pub fn value(&mut self, flag: &str) -> Result<String> {
        self.words
            .pop_front()
            .ok_or_else(|| format!("{flag} expects a value").into())
    }

    /// The value following `flag`, parsed.
    pub fn parsed<T: std::str::FromStr>(&mut self, flag: &str) -> Result<T> {
        let v = self.value(flag)?;
        v.parse()
            .map_err(|_| format!("{flag}: cannot read {v:?}").into())
    }
}

/// The shared flags as written, before any file is read.
#[derive(Debug, Default, Clone)]
pub struct Shared {
    pub protein: Option<String>,
    /// One record of a multi-record `--protein` file, which a batch runs in
    /// place of reading the flag's value.
    pub protein_record: Option<crate::input::NamedSequence>,
    pub codon_usage: Option<String>,
    pub utr5: Option<String>,
    pub utr3: Option<String>,
    pub stop: Option<String>,
    pub forbid: Option<String>,
    pub access: Option<String>,
    pub threads: Option<usize>,
    pub deterministic: bool,
    /// `--beam-first B`: refute a point with a beam fold of width `B` before
    /// folding it exactly. Always `None` without the `beam-first` feature,
    /// where no flag sets it and the check folds exactly throughout.
    pub beam_first: Option<usize>,
    pub out: Option<PathBuf>,
    /// `--params PATH`: energy parameters loaded from a file in place of the
    /// built-in table.
    pub params: Option<PathBuf>,
    pub force: bool,
    /// Do not print the table of what the run wrote.
    pub quiet: bool,
}

impl Shared {
    /// Consume `flag` (and its value) if it is a shared flag; `false` if not.
    pub fn take(&mut self, flag: &str, flags: &mut Flags) -> Result<bool> {
        let slot = match flag {
            "--protein" => &mut self.protein,
            "--codon-usage" => &mut self.codon_usage,
            "--utr5" => &mut self.utr5,
            "--utr3" => &mut self.utr3,
            "--stop" => &mut self.stop,
            "--forbid" => &mut self.forbid,
            "--access" => &mut self.access,
            "--threads" => {
                let n: usize = flags.parsed(flag)?;
                if n == 0 {
                    return Err("--threads must be at least 1".into());
                }
                return set_once(flag, &mut self.threads, n).map(|()| true);
            }
            "--out" => {
                let v = PathBuf::from(flags.value(flag)?);
                return set_once(flag, &mut self.out, v).map(|()| true);
            }
            "--params" => {
                let v = PathBuf::from(flags.value(flag)?);
                return set_once(flag, &mut self.params, v).map(|()| true);
            }
            "--deterministic" => return Ok(set_flag(&mut self.deterministic)),
            "--force" => return Ok(set_flag(&mut self.force)),
            "--quiet" => return Ok(set_flag(&mut self.quiet)),
            _ => return Ok(false),
        };
        let v = flags.value(flag)?;
        set_once(flag, slot, v)?;
        Ok(true)
    }

    /// The energy parameters a run folds with: the file `--params` names, or
    /// the built-in table.
    pub fn energy_params(&self) -> Result<Params> {
        Ok(match &self.params {
            Some(path) => Params::from_file(path)?,
            None => Params::load_default()?,
        })
    }

    /// Worker threads as written; 1 unless given.
    pub fn threads(&self) -> usize {
        self.threads.unwrap_or(1)
    }

    /// Worker threads a run actually uses.
    ///
    /// `--deterministic` wins over `--threads`: a single-threaded fold is
    /// byte-identical run to run, while more threads may spell a tied design
    /// differently. The cost of asking for both is therefore the whole of the
    /// threading speed-up — a canonical tie-break is the alternative if a
    /// command ever needs both at once.
    pub fn resolved_threads(&self) -> usize {
        if self.deterministic {
            1
        } else {
            self.threads()
        }
    }
}

/// The refusal for a flag spelled as it was before a rename, naming the
/// spelling that replaced it.
pub fn renamed(flag: &str) -> String {
    match flag {
        "--structurally-optimal" => format!("{flag} is now --modular"),
        other => format!("{other} has been renamed (see --help)"),
    }
}

fn set_flag(slot: &mut bool) -> bool {
    *slot = true;
    true
}

/// Store a single-valued flag, refusing a repeat.
fn set_once<T>(flag: &str, slot: &mut Option<T>, value: T) -> Result<()> {
    if slot.is_some() {
        return Err(format!("{flag} is given twice").into());
    }
    *slot = Some(value);
    Ok(())
}

/// Shared help text: the flags every computing subcommand reads.
pub const SHARED_HELP: &str = "\
Input
  --protein PATH|SEQ|-      protein: a one-record FASTA file, a literal
                            amino-acid sequence, or `-` for stdin
  --codon-usage human|PATH  codon usage for CAI [default: human]
  --params PATH             load energy parameters from a file [default:
                            the built-in Turner 2004 table]

UTRs and constraint
  --utr5 PATH|SEQ|-         fixed 5′ UTR: a one-record FASTA file, a literal
                            sequence (T read as U), or `-` for stdin
  --utr3 PATH|SEQ|-         fixed 3′ UTR, same forms
  --stop UAA|UAG|UGA|free|each|none
                            give the stop codon after the last residue, let the
                            search design it (`free`, needs a flank), or ask for
                            none; coding sequence to the constraint and outside
                            CAI either way. A designed stop is the coding
                            sequence's last slot, chosen among the three, and
                            each point names its own. `each` walks every stop
                            codon's own front in one run and writes each as a
                            pinned-stop run would, to the `--out` path with the
                            codon before its extension; it is for `front` and
                            needs that path [default: free with a flank, none
                            without one]
  --forbid CLASSES|none     pair classes to forbid, comma list of
                            5-cds, cds-3, 5-3 [default: 5-cds and cds-3,
                            whichever flank is given; 5-3 only if asked]
  --access UP,DOWN          keep UP nt upstream to DOWN nt downstream of the
                            start codon unpaired [default: off]

Execution
  --threads N               [default: 1]
  --deterministic           byte-identical output at any --threads
  --beam-first B            in the check, refute a point with a beam fold of
                            width B before folding it exactly [default: off]
  --out PATH                JSONL result [default: stdout]
  --force                   overwrite an existing --out
  --quiet                   do not print the table of what was written to
                            stderr at the end of the run
";

/// [`shared_help`] for a build with no beam folder.
pub fn shared_help() -> String {
    shared_help_without_check()
}

/// The shared flags as a command without a check should list them.
///
/// `derna front` takes every shared flag but `--beam-first`, which speeds up
/// a *check* and so has nothing to do in a command that does not check its
/// points — it is refused there, and a help text that listed it would send a
/// reader to write a command line the command rejects. One place builds this,
/// so the two texts cannot drift.
pub fn shared_help_without_check() -> String {
    SHARED_HELP
        .lines()
        .filter(|line| {
            !line.trim_start().starts_with("--beam-first")
                && !line.contains("width B before folding it exactly")
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(words: &[&str]) -> Result<Shared> {
        let mut flags = Flags::new(words.iter().map(|s| s.to_string()));
        let mut shared = Shared::default();
        while let Some(flag) = flags.next_word() {
            if !shared.take(&flag, &mut flags)? {
                return Err(format!("unknown flag {flag}").into());
            }
        }
        Ok(shared)
    }

    #[test]
    fn shared_flags_are_read() {
        let s = parse(&["--protein", "MK", "--threads", "4", "--deterministic"]).unwrap();
        assert_eq!(s.protein.as_deref(), Some("MK"));
        assert_eq!(s.threads(), 4);
        assert!(s.deterministic);
    }

    #[test]
    fn a_repeated_flag_is_refused() {
        assert!(parse(&["--utr5", "A", "--utr5", "C"]).is_err());
    }

    #[test]
    fn a_missing_value_is_refused() {
        let e = parse(&["--forbid"]).unwrap_err().to_string();
        assert!(e.contains("--forbid expects a value"), "{e}");
    }

    #[test]
    fn zero_threads_is_refused() {
        assert!(parse(&["--threads", "0"]).is_err());
    }
}
