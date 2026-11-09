//! `derna front`: the exact Pareto front of (MFE, CAI), optionally with fixed
//! UTRs and a region constraint.
//!
//! The walk returns the Pareto designs between the designs optimal at two
//! weights, so the range decides how much of the front comes back. There is no
//! range flag: the walk runs the symmetric range [ε, 1 − ε] with ε = 2⁻²³,
//! which is below the weight at which a corner stops reaching the front's end
//! at every length a real protein has, and is exact in binary floating point.
//!
//! Under a constraint the front certifies its own points once the walk has
//! written them ([`FrontArgs::check_points`]): each gains the verdict `check`
//! would add, at about one free fold of the whole molecule per point.
//! `--no-check` declines it, and with no constraint there is nothing to
//! check. `--modular` returns the front of the designs that pass instead
//! (see `modular_front`), each certified by construction.
//!
//! The constrained walk is deterministic and is most of a modular
//! run's cost, so it can be kept and reused: `--front-out` writes it beside a
//! modular run, in the format a plain `front` writes, and
//! `--front-from` reads such a file in place of walking. The file's header
//! must name the same molecule — protein, flanks, stop, constraint and codon
//! table — or the run is refused.
//!
//! A modular run writes its points as it makes them, so a run an
//! outer limit kills leaves everything it had finished rather than nothing.
//!
//! `--motif` forbids motifs over the coding sequence as `design` does, and the
//! front is then the exact front of the designs that spell none
//! (`derna_lib::motif_front`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use derna_lib::motif_front::{
    motif_front_walk_run, MotifFrontSpec, MotifResume, MotifRun, MotifWalk, WalkBudget,
};
use derna_lib::{
    design_front_walk_per_stop, design_front_walk_run, BeamError, FrontPoint, HullPoint, PassGrid,
    UtrSpec, WalkOptions, WalkResume, WalkRun, WalkedFront,
};
use serde_json::{json, Map, Value};

use crate::args::{Flags, Shared};
use crate::modular_front::{self, Budget};
use crate::molecule::Molecule;
use crate::result;
use crate::sequence_constraint::SequenceConstraints;
use crate::{show, Result};
use derna_lib::modular_front::OpenBox as ModularBox;

/// The weight the walk's range stops ε short of each end. 2⁻²³ is exact in
/// single and double precision, and sits below the bound at which the MFE
/// corner stops reaching the front's end — about 2 × 10⁻⁶ at 2,828 residues,
/// falling with length — at every length a protein has.
pub const EPSILON: f64 = 1.0 / (1u32 << 23) as f64;

/// `derna front` as parsed.
#[derive(Debug, Default, Clone)]
pub struct FrontArgs {
    pub shared: Shared,
    /// Where the walk's diagnostics go, or `None` to leave them unwritten.
    pub report: Option<std::path::PathBuf>,
    /// Return the modular front instead of the constrained one.
    pub modular: bool,
    /// The modular front's budget.
    pub budget: Budget,
    /// Read the constrained front from this file instead of walking it.
    pub front_from: Option<PathBuf>,
    /// Also write the constrained front here (modular runs).
    pub front_out: Option<PathBuf>,
    /// Forbidden motifs and where they are forbidden, and the pins.
    pub sequence: SequenceConstraints,
    /// An earlier motif front of the same molecule to resume: only the gaps
    /// it left open are walked.
    pub resume: Option<std::path::PathBuf>,
    /// A motif front of the same molecule under a subset of these motifs:
    /// its points that spell none of them are kept, and only the stretches
    /// where it lost a point or left a box open are walked.
    pub narrow_from: Option<std::path::PathBuf>,
    /// Rewrite `--out` with the front found so far after each settled gap;
    /// with `--modular`, keep that walk beside it in `OUT.walk.jsonl` and
    /// append the search's progress to `OUT.search.jsonl`. Running the same
    /// command again continues from what is there.
    pub checkpoint: bool,
    /// A modular search's progress file to continue from.
    pub resume_search: Option<PathBuf>,
    /// This run continues what an earlier run of the same command left at and
    /// beside `--out`, so it replaces those files without `--force`.
    pub continuing: bool,
    /// How the modular search visits a region nothing caps.
    pub top_search: modular_front::TopSearch,
    /// When the modular search stops or holds back a region
    /// before its budget runs out.
    pub policy: modular_front::SearchPolicy,
    /// The largest gap, in kcal/mol, a point of the modular
    /// search may have; `None` is zero.
    pub epsilon: Option<f64>,
    /// A result of the same molecule whose modular points seed
    /// the search, under any constraint.
    pub anchors_from: Option<PathBuf>,
    /// Threads inside each fold of a motif front's walk.
    pub fold_threads: Option<usize>,
    /// With `--modular --motif`: how long the motif front's walk
    /// may run before the search starts from what it has.
    pub walk_secs: Option<f64>,
    /// Leave the front uncertified: do not fold each point against its own
    /// free fold after the walk.
    pub no_check: bool,
    /// Do not save the walk's progress to `--out` as it runs.
    pub no_checkpoint: bool,
    /// With the default check: how many of its last free folds each
    /// certifying worker keeps to copy cells from. Zero folds every point
    /// from scratch.
    pub check_pool: usize,
}

impl FrontArgs {
    /// Whether the run certifies the front it writes.
    ///
    /// A constrained front's points carry an energy their own free fold need
    /// not agree with, and the difference decides whether a design's structure
    /// can be trusted — so the walk certifies what it wrote unless there is
    /// nothing to learn or the caller declined:
    ///
    /// - `--no-check` declines it.
    /// - With no constraint every design's free fold respects it, so every
    ///   point is modular already and a fold would only confirm it. This is
    ///   the coding-sequence-only run, motifs or not: a motif constrains what
    ///   a design may spell, not which pairs its fold may form.
    /// - A modular search already folded every point it emitted, and each
    ///   has gap 0 by construction.
    pub fn check_points(&self, m: &Molecule) -> bool {
        !self.no_check && !m.constraint.is_inert() && !self.modular
    }
}

impl FrontArgs {
    /// Whether the run may replace the files it writes: under `--force`, or
    /// when it continues the run that wrote them.
    pub fn overwrite(&self) -> bool {
        self.shared.force || self.continuing
    }
}

/// The derivations an empty search's walks must have taken before
/// `--empty-stop-rounds` stops it, unless `--empty-stop-derivations` says.
const EMPTY_STOP_DERIVATIONS: u64 = 1_000_000;

/// The options only `derna front` takes, in the order its help lists them;
/// the shared input, constraint and execution flags follow.
const FRONT_HELP: &str = r#"derna front — the exact Pareto front of (MFE, CAI)

Usage: derna front --protein P [options] [--out FRONT.jsonl]

Output
  --report PATH             diagnostics: per-gap timing, task timeline
  --no-check                leave the front uncertified; by default a
                            constrained run folds each point against its own
                            free fold and records the gap
  --check-pool N            let each certifying worker copy cells from its
                            last N free folds; neighbours on a front differ
                            in a few codons. Same verdicts; costs N
                            whole-molecule fold tables per worker

Budget
  --max-secs S              stop the walk after S seconds and return the
                            partial front, naming the boxes left open
                            [default: no limit]
  --max-rss-gb G            stop the walk at G GB resident and return the
                            partial front (Linux only, resident memory being
                            read from /proc) [default: no limit]
  --max-gap-folds N         stop the walk after N gap folds and return the
                            partial front, naming the boxes left open: a
                            fixed amount of work on any machine, so --resume
                            continues a long front in pieces [default: no
                            limit]
  --shed-on-memory          (with --modular --max-rss-gb) at G GB end only
                            the walks running then, leaving their regions
                            open, and go on; stop only if memory stays over
                            G once they let go

Save and resume
  --checkpoint              save progress beside --out as it settles, so
                            running the same command again continues a
                            killed or budget-stopped run instead of starting
                            over: a walk rewrites --out with the front so
                            far; a --modular run keeps its walk in
                            OUT.walk.jsonl and appends each point checked,
                            design found and band slice finished to
                            OUT.search.jsonl. --force starts over instead.
                            Without it a walk still saves into --out, but a
                            rerun does not continue from it
  --no-checkpoint           do not save the walk's progress as it runs
  --resume FRONT.jsonl      continue the walk from this front of the same
                            molecule rather than from --out: walk only the
                            boxes it left open, and return its points with
                            theirs
  --resume-search PATH      (with --modular) continue the search this log
                            records rather than OUT.search.jsonl: checked
                            points are not checked again and the band starts
                            where it stopped

Modular front
  --modular                 return the front of the modular designs, those
                            whose own free fold respects the constraint; a
                            budget then stops this search and the
                            constrained walk before it runs whole
  --front-out PATH          also write the constrained front, as a plain
                            `front` writes it
  --top-search band|window|none
                            how a region no found point caps is searched:
                            every design, most adapted first; a window walk;
                            or not at all. `none` stops the search at the
                            best-adapted modular design it holds — a seed or
                            an --anchors-from design — and reports the band
                            above it as `top_band` instead of walking it. That
                            band is the region nothing can cap, so it is what
                            a budget is spent on; without it the search can
                            finish, and the front it returns is the modular
                            front of the designs at or below that adaptation.
                            Nothing under the ceiling is missing, nothing
                            above it is returned, and a point `top_band` could
                            hold a dominating design for is provisional rather
                            than certified [default: window]
  --empty-stop-rounds R     stop a search that has found no point once R
                            rounds are over and its walks have taken at least
                            --empty-stop-derivations N (default 1000000); the
                            front is then incomplete, stopped_by empty-rounds,
                            and says nothing about whether a point exists
  --hold-barren             give a region re-entered unchanged after a visit
                            that found nothing no more time than that visit
                            while another region is open; changes what a
                            budget finds
  --max-derivations N       end each window walk once it has taken N
                            derivations off its root, whatever time is
                            left of its share; the stretch then reads
                            stopped: derivations. A span of time measures
                            a machine as much as a build, so an A/B of two
                            builds under --max-secs alone compares runs
                            that walked different amounts; this gives both
                            the same amount to walk [default: uncapped]
  --epsilon E               return the front of the designs whose gap (how far
                            the free MFE sits below the constrained MFE) is at
                            most E kcal/mol (not with --motif),
                            with its own search: a design within E caps only
                            designs within E, and a modular design one within
                            E dominates is not on it. Energies are whole
                            hundredths of a kcal/mol, and on the cells
                            measured every gap was a whole tenth, so 0.1 is the
                            smallest E that admits anything beyond 0 there: the
                            energy grid's step, not a choice. Gaps clump (on
                            one cell 311 near misses all sat at 0.8), so the
                            front changes in steps as E grows; the run prints
                            how many points sit at each gap
                            [default: 0]
  --anchors-from PATH       check the points of a `derna` result for the
                            same molecule under any constraint (an
                            unconstrained front, say); those modular here cap
                            the search from the start
  --walk-secs S             with --motif: stop the motif front's walk after
                            S seconds (0 with --resume: walk nothing) and search
                            from the partial front; its open boxes floor the
                            search, which stays exact [default: walk it whole]

Reuse
  --front-from PATH         read the constrained front from a `front`
                            result for the same molecule instead of walking it;
                            with --modular it may be a stopped walk, whose
                            open boxes floor the search; a motif front with
                            --motif

Sequence constraints
  --motif PATH|SEQ          forbidden motif (IUPAC), or a file of one per
                            line; repeatable. The front is then over the
                            designs that spell none
  --motif-window LO..HI     forbid only over this coding sub-window
                            [default: the whole coding sequence]
  --max-homopolymer K       cap the longest run of one nucleotide at K, by
                            forbidding the four (K+1)-runs; not combinable
                            with --motif-window
  --narrow-from FRONT.jsonl walk from a motif front of the same molecule
                            under a subset of these motifs: keep its points
                            that spell none, walk only where it lost one
  --fold-threads N          threads inside each fold of the motif front's
                            walk, beside its --threads workers;
                            the front is the same at any N [default: 1]

"#;

/// Help text for `derna front`.
pub fn help() -> String {
    format!("{FRONT_HELP}{}", crate::args::shared_help_without_check())
}

/// Parse the words after `front`.
///
/// `--beam-first` is refused rather than ignored: it speeds up a check, and a
/// plain front carries none.
pub fn parse(words: impl IntoIterator<Item = String>) -> Result<FrontArgs> {
    let mut flags = Flags::new(words);
    let mut args = FrontArgs::default();
    let mut empty_stop_derivations = None;
    while let Some(flag) = flags.next_word() {
        if args.shared.take(&flag, &mut flags)? || args.sequence.take(&flag, &mut flags)? {
            continue;
        }
        // Accepting a flag the command cannot act on is worse than refusing
        // it: the run looks as though it did what was asked.
        match flag.as_str() {
            "--report" => {
                if args.report.is_some() {
                    return Err("--report is given twice".into());
                }
                args.report = Some(flags.value(&flag)?.into());
            }
            "--lambda" => {
                return Err("derna front walks the whole front and takes no --lambda; \
                            `derna design` answers at a weight"
                    .into())
            }
            "--modular" => args.modular = true,
            "--structurally-optimal" => return Err(crate::args::renamed(&flag).into()),
            "--max-secs" => args.budget.max_secs = Some(positive(&flag, &mut flags)?),
            "--max-rss-gb" => args.budget.max_rss_gb = Some(positive(&flag, &mut flags)?),
            "--max-gap-folds" => {
                let n: usize = flags.parsed(&flag)?;
                if n == 0 {
                    return Err(format!("{flag} {n}: give a count of 1 or more").into());
                }
                args.budget.max_gap_folds = Some(n);
            }
            "--shed-on-memory" => args.budget.shed_on_memory = true,
            "--front-from" => once(&mut args.front_from, &flag, &mut flags)?,
            "--front-out" => once(&mut args.front_out, &flag, &mut flags)?,
            "--resume" => once(&mut args.resume, &flag, &mut flags)?,
            "--narrow-from" => once(&mut args.narrow_from, &flag, &mut flags)?,
            "--checkpoint" => args.checkpoint = true,
            "--no-check" => args.no_check = true,
            "--no-checkpoint" => args.no_checkpoint = true,
            "--check-pool" => args.check_pool = flags.parsed(&flag)?,
            "--resume-search" => once(&mut args.resume_search, &flag, &mut flags)?,
            "--anchors-from" => once(&mut args.anchors_from, &flag, &mut flags)?,
            "--epsilon" => {
                let e: f64 = flags.parsed(&flag)?;
                if !(e >= 0.0 && e.is_finite()) {
                    return Err(format!("{flag} {e}: give a gap in kcal/mol, 0 or more").into());
                }
                if args.epsilon.replace(e).is_some() {
                    return Err("--epsilon is given twice".into());
                }
            }
            "--top-search" => {
                args.top_search = match flags.value(&flag)?.as_str() {
                    "band" => modular_front::TopSearch::Band,
                    "window" => modular_front::TopSearch::Window,
                    "none" => modular_front::TopSearch::None,
                    other => {
                        return Err(format!(
                            "derna front: --top-search is band, window or none, not {other:?}"
                        )
                        .into())
                    }
                }
            }
            "--empty-stop-rounds" => {
                let rounds: usize = flags.parsed(&flag)?;
                if args
                    .policy
                    .give_up_empty
                    .replace((rounds, EMPTY_STOP_DERIVATIONS))
                    .is_some()
                {
                    return Err("--empty-stop-rounds is given twice".into());
                }
            }
            "--empty-stop-derivations" => {
                let least: u64 = flags.parsed(&flag)?;
                if empty_stop_derivations.replace(least).is_some() {
                    return Err("--empty-stop-derivations is given twice".into());
                }
            }
            "--hold-barren" => args.policy.hold_barren = true,
            "--max-derivations" => {
                let n: u64 = flags.parsed(&flag)?;
                if n == 0 {
                    return Err(format!("{flag} {n}: give a count of 1 or more").into());
                }
                if args.policy.max_derivations.replace(n).is_some() {
                    return Err("--max-derivations is given twice".into());
                }
            }
            "--walk-secs" => {
                let s: f64 = flags.parsed(&flag)?;
                if !(s >= 0.0 && s.is_finite()) {
                    return Err("--walk-secs must be a non-negative number of seconds".into());
                }
                if args.walk_secs.replace(s).is_some() {
                    return Err("--walk-secs is given twice".into());
                }
            }
            "--fold-threads" => {
                let n: usize = flags.parsed(&flag)?;
                if n == 0 {
                    return Err("--fold-threads must be at least 1".into());
                }
                if args.fold_threads.replace(n).is_some() {
                    return Err("--fold-threads is given twice".into());
                }
            }
            "--strict" => {
                return Err(format!(
                    "derna front: {flag} is not supported yet — resuming a walk needs it to \
                     report its unfinished gaps first"
                )
                .into())
            }
            other => return Err(format!("derna front: unknown flag {other:?} (see --help)").into()),
        }
    }
    if args.fold_threads.is_some() && args.sequence.motifs.is_empty() {
        return Err(
            "derna front: --fold-threads reaches the motif front's walk only, so it \
                    needs --motif"
                .into(),
        );
    }
    if args.walk_secs.is_some() && (!args.modular || args.sequence.motifs.is_empty()) {
        return Err(
            "derna front: --walk-secs budgets the motif front's walk before a modular \
                    search, so it needs --modular and --motif"
                .into(),
        );
    }
    if args.narrow_from.is_some()
        && (args.resume.is_some() || args.front_from.is_some() || args.sequence.motifs.is_empty())
    {
        return Err(
            "derna front: --narrow-from walks a motif front from a wider one, so it needs \
                    --motif and neither --resume nor --front-from"
                .into(),
        );
    }
    if args.resume.is_some() && args.front_from.is_some() {
        return Err(
            "derna front: --resume continues a walk and --front-from reads a front \
                    instead of walking one"
                .into(),
        );
    }
    if args.anchors_from.is_some() && !args.modular {
        return Err(
            "derna front: --anchors-from seeds a modular search, so it needs \
                    --modular"
                .into(),
        );
    }
    if args.anchors_from.is_some() && !args.sequence.motifs.is_empty() {
        return Err(
            "derna front: --anchors-from seeds the search over every design, and the search \
                    over motif-free designs reads no anchors, so it cannot be given with \
                    --motif"
                .into(),
        );
    }
    if args.epsilon.is_some() && !args.modular {
        return Err(
            "derna front: --epsilon relaxes the modular search, so it needs \
                    --modular"
                .into(),
        );
    }
    if args.epsilon.is_some() && !args.sequence.motifs.is_empty() {
        return Err(
            "derna front: --epsilon relaxes the search over every design, and the search \
                    over motif-free designs takes none, so it cannot be given with --motif"
                .into(),
        );
    }
    if args.resume_search.is_some() && !args.modular {
        return Err(
            "derna front: --resume-search continues a modular search, so it \
                    needs --modular"
                .into(),
        );
    }
    if args.check_pool > 0 && (args.no_check || args.shared.beam_first.is_some()) {
        return Err(format!(
            "derna front: --check-pool reuses the default check's free folds, and {} \
             takes that check away",
            if args.no_check {
                "--no-check"
            } else {
                "--beam-first"
            }
        )
        .into());
    }
    if args.budget.max_rss_gb.is_some() && !derna_lib::task_clock::CURRENT_RSS_READABLE {
        return Err(
            "derna front: --max-rss-gb needs the process's resident memory, \
                    which this platform does not report; the limit would never bind"
                .into(),
        );
    }
    if args.checkpoint && args.no_checkpoint {
        return Err("derna front: --checkpoint and --no-checkpoint ask for opposite things".into());
    }
    if args.checkpoint && args.front_from.is_some() && !args.modular {
        return Err(
            "derna front: --checkpoint saves a walk as it runs, and --front-from runs \
                    none: it reads a front this command wrote earlier"
                .into(),
        );
    }
    // A batch's --out is a directory, and each record's file is checked when
    // the batch runs it.
    let batch = crate::batch::is_batch(&args.shared);
    match result::destination(args.shared.out.as_deref()).filter(|_| !batch) {
        None if args.checkpoint && !batch => {
            return Err(
                "derna front: --checkpoint saves progress beside --out, so it needs a file there"
                    .into(),
            );
        }
        // A checkpoint is written beside --out and renamed over it, which a
        // device or a pipe cannot take.
        Some(path)
            if args.checkpoint
                && std::fs::metadata(path).is_ok_and(|m| !m.file_type().is_file()) =>
        {
            return Err(format!(
                "derna front: --checkpoint saves progress beside --out, so it needs a regular \
                 file, not {}",
                path.display()
            )
            .into());
        }
        _ => {}
    }
    // With --modular the budget is the search's, which runs
    // whether the constrained front is walked or read back.
    let budgeted = args.budget.max_secs.is_some() || args.budget.max_rss_gb.is_some();
    if budgeted && args.front_from.is_some() && !args.modular {
        return Err(
            "derna front: --max-secs and --max-rss-gb budget a walk, and --front-from \
                    runs none: it reads a front this command wrote earlier"
                .into(),
        );
    }
    if args.budget.max_gap_folds.is_some() && (args.modular || args.front_from.is_some()) {
        return Err(
            "derna front: --max-gap-folds stops the walk of a front, and with --modular \
                    or --front-from the walk is not what the run returns"
                .into(),
        );
    }
    if args.shared.beam_first.is_some() {
        return Err(
            "derna front: --beam-first speeds up the check of a design, and a front's \
                    points are not checked — `derna check` certifies them"
                .into(),
        );
    }
    if !args.sequence.pins.is_empty() && args.sequence.motifs.is_empty() {
        return Err(
            "derna front: --pin is not supported yet without --motif — the restricted \
                    lattice a pin builds reaches the motif front's walk, not the plain one; \
                    `derna design` takes it"
                .into(),
        );
    }
    args.sequence.validate()?;
    if args.front_from.is_some() && !args.sequence.motifs.is_empty() && !args.modular {
        return Err(
            "derna front: --front-from with --motif reads a motif front to search from, so \
                    it needs --modular; --resume continues a motif front"
                .into(),
        );
    }
    if args.front_from.is_some() && args.walk_secs.is_some() {
        return Err(
            "derna front: --walk-secs budgets the motif front's walk, and --front-from \
                    runs none: it reads a front this command wrote earlier"
                .into(),
        );
    }
    if args.front_out.is_some() && !args.modular {
        return Err(
            "derna front: --front-out writes the constrained front beside a modular \
                    run; a plain front already writes it with --out"
                .into(),
        );
    }
    if let Some(least) = empty_stop_derivations {
        let Some((rounds, _)) = args.policy.give_up_empty else {
            return Err("--empty-stop-derivations needs --empty-stop-rounds".into());
        };
        args.policy.give_up_empty = Some((rounds, least));
    }
    if args.budget.shed_on_memory && args.budget.max_rss_gb.is_none() {
        return Err("--shed-on-memory needs --max-rss-gb".into());
    }
    if args.policy.max_derivations.is_some() && !args.modular {
        return Err(
            "derna front: --max-derivations caps the modular search's window walks; \
                    a plain front walks to the whole front and takes --max-secs"
                .into(),
        );
    }
    if !batch {
        settle_rerun(&mut args)?;
    }
    Ok(args)
}

/// Where a `--modular --checkpoint` run keeps its constrained walk: beside
/// `out`, as `<out>.walk.jsonl`, in the format a plain `front` writes.
pub fn walk_path(out: &Path) -> PathBuf {
    let mut name = out.as_os_str().to_owned();
    name.push(".walk.jsonl");
    PathBuf::from(name)
}

/// The header of a result file and how many point lines follow it. A last
/// line cut short by a kill is not counted; a first line that is not a header
/// is `None`.
fn result_head(path: &Path) -> Result<Option<(Value, usize)>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("derna front: cannot read {}: {e}", path.display()))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines
        .next()
        .and_then(|l| result::parse_line(l).ok())
        .filter(|h| h["record"] == "header");
    let points = lines
        .filter_map(|l| result::parse_line(l).ok())
        .filter(|p| p["record"] == "point")
        .count();
    Ok(header.map(|h| (h, points)))
}

/// Whether `out` already holds the finished result of a run of `args` — every
/// stop's file under `--stop each`. A batch rerun under `--checkpoint` skips
/// such a record, where a single run would refuse the finished `--out`.
pub fn finished(args: &FrontArgs, out: &Path) -> bool {
    let arms = crate::molecule::stop_arms(args.shared.stop.as_deref()).unwrap_or_default();
    let paths: Vec<PathBuf> = if arms.is_empty() {
        vec![out.to_path_buf()]
    } else {
        arms.iter().map(|c| stop_path(out, c)).collect()
    };
    paths.iter().all(|path| {
        path.is_file()
            && result_head(path).ok().flatten().is_some_and(|(h, points)| {
                h["complete"] == true
                    && if args.modular {
                        h["points"].as_u64() == Some(points as u64)
                    } else {
                        h["checkpoint"] != true
                    }
            })
    })
}

/// With `--checkpoint`, take up what an earlier run of the same command left
/// at and beside `--out`, so running it again continues rather than starts
/// over: a walk's front so far becomes `--resume`, and a modular search's
/// progress file `--resume-search`. A finished result, or a file this command
/// did not write, is refused. With `--force` nothing is taken up, and `run`
/// clears what is there instead; an explicit `--resume` or `--resume-search`
/// is kept as given.
///
/// Only the files are read here. That a front is of this molecule is checked
/// where it is read back as a resume, which refuses another molecule's.
pub fn settle_rerun(args: &mut FrontArgs) -> Result<()> {
    let Some(out) = result::destination(args.shared.out.as_deref())
        .filter(|_| args.checkpoint && !args.shared.force)
        .map(Path::to_path_buf)
    else {
        return Ok(());
    };
    let refuse = |why: &str| -> Box<dyn std::error::Error> {
        format!(
            "derna front: --out {} {why}; pass --force to start over",
            out.display()
        )
        .into()
    };
    let walking = args.front_from.is_none() && args.resume.is_none();
    if args.modular {
        if out.exists() {
            let Some((h, points)) = result_head(&out)? else {
                return Err(refuse("is not a `derna` result"));
            };
            if h["command"] != "front" || h["modular_front"] != true {
                return Err(refuse("is not a modular front this command was writing"));
            }
            if h["complete"] == true && h["points"].as_u64() == Some(points as u64) {
                return Err(refuse("holds a finished modular front"));
            }
            args.continuing = true;
        }
        let walk = walk_path(&out);
        if walking && walk.exists() {
            args.resume = Some(walk);
            args.narrow_from = None;
            args.continuing = true;
        }
        let search = modular_front::progress_path(&out);
        if args.resume_search.is_none() && search.exists() {
            args.resume_search = Some(search);
            args.continuing = true;
        }
        return Ok(());
    }
    if args
        .resume
        .as_deref()
        .is_some_and(|from| same_file(from, &out))
    {
        args.continuing = true;
        return Ok(());
    }
    if !walking || !out.exists() {
        return Ok(());
    }
    let Some((h, _)) = result_head(&out)? else {
        return Err(refuse("is not a `derna` result"));
    };
    if h["command"] != "front" || h.get("modular_front").is_some() {
        return Err(refuse("is not a front walk this command was writing"));
    }
    if h.get("open").is_none() {
        return Err(refuse(
            "records no open boxes, so no walk can continue from it",
        ));
    }
    if h["complete"] == true && h["checkpoint"] != true {
        return Err(refuse("holds a finished front"));
    }
    // The checkpoint already holds what a narrowed walk kept, so the rerun
    // continues it rather than narrowing again.
    args.resume = Some(out);
    args.narrow_from = None;
    args.continuing = true;
    Ok(())
}

/// Where the walk saves its progress as it settles, or `None` if it does not.
///
/// With `--checkpoint`: `--out` itself, or beside it under `--modular`, whose
/// `--out` is the search's result — and a rerun then continues from it
/// ([`settle_rerun`]).
///
/// Without it, a plain walk still saves into `--out`. A run killed from
/// outside — a scheduler's time limit, the kernel out of memory — would
/// otherwise leave nothing, because the result is written at the end, and at
/// these lengths a walk is hours. The save is a write beside `--out`, synced
/// and renamed over it, so the file is always a whole result. Only the saving
/// is the default: a rerun overwrites or refuses as it would without it, and
/// continues only when `--checkpoint` asks. There is no default save without
/// a regular file at `--out`, with `--front-from`, which walks nothing, under
/// `--modular`, whose walk is saved only when asked, or under
/// `--no-checkpoint`.
fn checkpoint_path(args: &FrontArgs) -> Option<PathBuf> {
    let out = result::destination(args.shared.out.as_deref())?;
    if args.checkpoint {
        return Some(if args.modular {
            walk_path(out)
        } else {
            out.to_path_buf()
        });
    }
    let saves = !args.no_checkpoint && args.front_from.is_none() && !args.modular;
    // A path that does not exist yet is a file once the walk writes it; one
    // that exists and is a device or a pipe cannot be renamed over.
    let renamable = std::fs::metadata(out).map_or(true, |m| m.file_type().is_file());
    (saves && renamable).then(|| out.to_path_buf())
}

/// With `--checkpoint --force`, remove what an earlier run left at and beside
/// `--out` before starting over, so a later run without `--force` cannot
/// continue from a mixture of two runs. A file an explicit `--resume` or
/// `--resume-search` names is read, not removed.
fn clear_checkpoint(args: &FrontArgs) -> Result<()> {
    let Some(out) = result::destination(args.shared.out.as_deref())
        .filter(|_| args.checkpoint && args.shared.force)
    else {
        return Ok(());
    };
    let named = |path: &Path| {
        [&args.resume, &args.resume_search]
            .into_iter()
            .flatten()
            .any(|from| same_file(from, path))
    };
    let regular = |path: &Path| std::fs::metadata(path).is_ok_and(|m| m.is_file());
    for path in [
        out.to_path_buf(),
        walk_path(out),
        modular_front::progress_path(out),
    ] {
        if regular(&path) && !named(&path) {
            std::fs::remove_file(&path).map_err(|e| {
                format!("derna front: --force cannot remove {}: {e}", path.display())
            })?;
        }
    }
    Ok(())
}

/// A path flag given at most once.
fn once(slot: &mut Option<PathBuf>, flag: &str, flags: &mut Flags) -> Result<()> {
    if slot.is_some() {
        return Err(format!("{flag} is given twice").into());
    }
    *slot = Some(flags.value(flag)?.into());
    Ok(())
}

/// A budget value: a positive number.
fn positive(flag: &str, flags: &mut Flags) -> Result<f64> {
    let v: f64 = flags.parsed(flag)?;
    if v > 0.0 && v.is_finite() {
        Ok(v)
    } else {
        Err(format!("{flag} {v}: give a positive number").into())
    }
}

/// The pass grid this molecule's length selects, unless a development build
/// gives its segments as `until:step,...`.
///
/// One function, read by the walk and by the header that reports it: a header
/// recomputing the choice could come to say what the walk did not do.
///
/// The standard grid less its weights 0.41 and 0.61
/// (`0.16:0.02,0.31:0.05,0.71:0.2,1:0.29`) takes 0.93x, 0.96x and 0.95x the
/// instructions on one protein each of 163, 400 and 417 residues, one plain
/// walk each, with the same objectives and hull; at 400 and 417 residues
/// some tied points are spelled differently. Dropping 0.51 and 0.71 as well
/// takes 1.15x at 400 residues: the folds the wider chords loosen cost more
/// than the passes saved.
pub fn pass_grid(m: &Molecule) -> PassGrid {
    // The grid is picked from the molecule's length rather than given: a
    // spacing that suits one length makes folds much slower at another.
    derna_lib::dev_env::var(derna_lib::dev_env::Setting::WalkPassGrid)
        .ok()
        .and_then(|g| PassGrid::parse(&g))
        .unwrap_or_else(|| PassGrid::for_residues(m.protein.sequence.len()))
}

/// Walk the whole front of `m`.
pub fn walk(m: &Molecule, threads: usize) -> Result<WalkedFront> {
    walk_within(m, threads, Budget::default(), None)
}

/// [`walk`] stopped short when `budget` runs out: the front points it has
/// settled, and the boxes it has not as [`WalkedFront::open`]. With `resume`,
/// only the boxes an earlier such walk left open are settled.
pub fn walk_within(
    m: &Molecule,
    threads: usize,
    budget: Budget,
    resume: Option<&WalkResume>,
) -> Result<WalkedFront> {
    walk_run(
        m,
        threads,
        budget,
        WalkRun {
            resume,
            ..WalkRun::default()
        },
    )
}

/// The plain walk's options for `m` under `budget`.
fn walk_options(m: &Molecule, budget: Budget) -> WalkOptions {
    WalkOptions {
        budget: WalkBudget {
            max_secs: budget.max_secs,
            max_rss_gb: budget.max_rss_gb,
        },
        gap_folds: budget.max_gap_folds.or_else(gap_folds),
        pass_grid: Some(pass_grid(m)),
        ..WalkOptions::for_residues(m.protein.sequence.len())
    }
}

/// [`walk_within`] under a [`WalkRun`], which may also checkpoint.
pub fn walk_run(
    m: &Molecule,
    threads: usize,
    budget: Budget,
    run: WalkRun<'_>,
) -> Result<WalkedFront> {
    let params = m.params.clone();
    let options = walk_options(m, budget);
    Ok(design_front_walk_run(
        &params,
        &m.designed_protein(),
        &m.codons.usage,
        &m.utr5_spec(),
        &m.utr3_spec(),
        m.table(),
        EPSILON,
        1.0 - EPSILON,
        options,
        threads,
        run,
    )?)
}

/// Walk the front of the designs of `m` that spell none of the motifs, or,
/// with `resume`, only the gaps an earlier such walk left open.
pub fn motif_walk(
    m: &Molecule,
    sequence: &SequenceConstraints,
    threads: usize,
    budget: Budget,
    resume: Option<&MotifResume>,
) -> Result<MotifWalk> {
    let run = MotifRun {
        budget: WalkBudget {
            max_secs: budget.max_secs,
            max_rss_gb: budget.max_rss_gb,
        },
        resume,
        checkpoint: None,
        gap_budget: None,
        slice_grow: None,
        gap_fit: None,
        bracket_cuts: None,
        resident_share: None,
        resident_passes: resident_passes(),
        stop_open_share: stop_open_share(),
        gap_folds: budget.max_gap_folds.or_else(gap_folds),
        probe_skip: None,
        probe_learn: None,
        fold_threads: 1,
        aware_upto: None,
        aware_seeds: motif_aware_seeds(),
        slice_width: motif_slice_width(),
        narrow_from: None,
        gap_secs: None,
    };
    motif_walk_run(m, sequence, threads, run)
}

/// [`motif_walk`] under a [`MotifRun`], which may also checkpoint.
pub fn motif_walk_run(
    m: &Molecule,
    sequence: &SequenceConstraints,
    threads: usize,
    run: MotifRun<'_>,
) -> Result<MotifWalk> {
    let params = m.params.clone();
    let fixed = |spec: UtrSpec| match spec {
        UtrSpec::Fixed(s) => Ok(s),
        _ => Err("derna front: a motif front takes fixed flanks only"),
    };
    let (utr5, utr3) = (fixed(m.utr5_spec())?, fixed(m.utr3_spec())?);
    let positional = sequence
        .positional()?
        .resolve(&m.layout())
        .map_err(|e| format!("--pin: {e}"))?;
    let placed = sequence.placed(m)?;
    let spec = MotifFrontSpec {
        utr5: &utr5,
        utr3: &utr3,
        constraints: m.table(),
        motifs: &placed,
        pins: (!sequence.pins.is_empty()).then_some(&positional.pins),
    };
    motif_front_walk_run(
        &params,
        &m.designed_protein(),
        &m.codons.usage,
        &spec,
        threads,
        run,
    )
    .map_err(|e| match e {
        BeamError::Infeasible => format!(
            "no encoding of the protein avoids every motif ({}) over {}{}",
            sequence.patterns().join(", "),
            sequence.enforced_phrase(),
            if sequence.pins.is_empty() {
                ""
            } else {
                " under the pins"
            }
        )
        .into(),
        e => e.into(),
    })
}

/// How close two objectives must be to be the same point. Energies are integer
/// dekacalories under the model, so they agree far inside this; an adaptation
/// is a sum of logarithms, and the hull's sweep and the gap folds can add its
/// terms in different orders, which moves the last bit (8.9 × 10⁻¹⁶ measured
/// on one vertex of a 17-point front). Matching exactly would call such a
/// vertex an off-hull point.
const SAME_POINT: f64 = 1e-9;

/// The constrained front of `m` read from a plain `front` result.
///
/// Refused unless the result is a plain front of the same molecule: a front
/// walked for other flanks, another constraint or another codon table is a
/// different front, and the modular front built on it would be
/// wrong without saying so. The points, and which of them are hull vertices,
/// are kept; the walk's gaps are not in the file, so the header reports none.
///
/// A front a budget stopped is read with the boxes it left open and what
/// stopped it. Its points are exact, and the boxes say where the missing
/// points may be, so a modular search that floors on both stays
/// exact; anything that took the points alone for the whole front would not.
pub fn read_front(m: &Molecule, path: &Path) -> Result<WalkedFront> {
    let read = show::read(Some(path))?;
    let h = &read.header;
    if h["command"] != "front" || h.get("modular_front").is_some() {
        return Err(format!(
            "{}: --front-from takes a plain `derna front` result",
            path.display()
        )
        .into());
    }
    let mismatch = molecule_mismatch(&result::molecule_fields(m), h);
    if let Some(field) = mismatch {
        return Err(format!(
            "{}: its {field} is not this run's, so its front is another molecule's",
            path.display()
        )
        .into());
    }
    let refuse = |why: String| format!("{}: {why}", path.display());
    let open = resume_boxes(h, &refuse)?
        .into_iter()
        .map(|(lo, hi)| {
            let end = |(mfe_kcal_mol, adaptation_log)| HullPoint {
                mfe_kcal_mol,
                adaptation_log,
            };
            (end(lo), end(hi))
        })
        .collect::<Vec<_>>();
    let stopped_by = match (h["complete"].as_bool(), h["stopped_by"].as_str()) {
        (Some(true), _) => None,
        (_, Some("memory")) => Some("memory"),
        (_, Some("time")) => Some("time"),
        _ => return Err(refuse("a stopped walk that does not say what stopped it".into()).into()),
    };
    if stopped_by.is_some() && open.is_empty() {
        return Err(refuse("its walk was stopped and left no box open".into()).into());
    }
    let front = read
        .points
        .iter()
        .map(front_point)
        .collect::<Result<Vec<_>>>()?;
    // The hull's vertices are the points the file marks as on it, so a front
    // read back reports the hull it was written with.
    let vertices: Vec<HullPoint> = front
        .iter()
        .zip(&read.points)
        .filter(|(_, line)| line["on_hull"].as_bool() == Some(true))
        .map(|(p, _)| HullPoint::from(p))
        .collect();
    Ok(WalkedFront {
        hull: vertices.clone(),
        vertices,
        gaps: Vec::new(),
        front,
        sweep_scans: 0,
        probes_cut: 0,
        grid_s: 0.0,
        grid_lanes: 1,
        stop_slack: None,
        sweep_s: 0.0,
        gaps_s: 0.0,
        rounds: 0,
        credit_s: 0.0,
        timeline: Vec::new(),
        refinements: Vec::new(),
        stopped_by,
        open,
        rests: Vec::new(),
    })
}

/// The first molecule field on which `header` differs from `fields`, names of
/// sequences aside (a file's protein may be named differently).
fn molecule_mismatch(fields: &Map<String, Value>, header: &Value) -> Option<String> {
    let strip = |v: &Value| match v {
        Value::Object(o) if o.contains_key("sequence") => o["sequence"].clone(),
        other => other.clone(),
    };
    fields
        .iter()
        .find(|(k, v)| strip(v) != strip(&header[k.as_str()]))
        .map(|(k, _)| k.clone())
}

/// One point line of a plain front, as the walk returned it.
fn front_point(point: &Value) -> Result<FrontPoint> {
    let number = |name: &str| -> Result<f64> {
        point[name]
            .as_f64()
            .ok_or_else(|| format!("a front point has no {name}").into())
    };
    let text = |name: &str| -> Result<String> {
        point[name]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("a front point has no {name}").into())
    };
    Ok(FrontPoint {
        sequence: text("sequence")?,
        structure: text("structure")?,
        mfe_kcal_mol: number("mfe")?,
        cai_log: number("cai_log")?,
        cai: number("cai")?,
        // A `derna` run designs no flank position, so there is no profile
        // weight to carry.
        utr_profile_log: 0.0,
    })
}

/// Whether each front point is a vertex of the hull — a point a weight sweep,
/// and so `design`, can reach. The rest are the points the gap folds find.
pub fn on_hull(walked: &WalkedFront) -> Vec<bool> {
    walked
        .front
        .iter()
        .map(|p| {
            let point = HullPoint::from(p);
            walked.hull.iter().any(|v| is_same_point(v, &point))
        })
        .collect()
}

/// Whether two hull points are the same point, to the precision the walk
/// computes them at.
/// How much energy the front spans, in kcal/mol, or `None` for a front with
/// nothing on it.
fn mfe_span(walked: &WalkedFront) -> Option<f64> {
    let energies: Vec<f64> = walked.front.iter().map(|p| p.mfe_kcal_mol).collect();
    let lo = energies.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (!energies.is_empty()).then_some(hi - lo)
}

fn is_same_point(a: &HullPoint, b: &HullPoint) -> bool {
    (a.mfe_kcal_mol - b.mfe_kcal_mol).abs() <= SAME_POINT
        && (a.adaptation_log - b.adaptation_log).abs() <= SAME_POINT
}

/// One front point as a result line.
///
/// The line names the stop codon the design ends in when the stop was the
/// search's to choose, and carries no such field when every design of the run
/// shares the header's: a reader of one line then knows the whole molecule
/// without consulting the others.
pub fn point(m: &Molecule, p: &FrontPoint, on_hull: bool) -> Value {
    let mut line = Map::new();
    line.insert("record".into(), "point".into());
    line.insert("sequence".into(), p.sequence.clone().into());
    line.insert("mfe".into(), p.mfe_kcal_mol.into());
    line.insert("cai".into(), p.cai.into());
    line.insert("cai_log".into(), p.cai_log.into());
    line.insert("structure".into(), p.structure.clone().into());
    if let Some(stop) = m.stop_of(&p.sequence) {
        line.insert("stop".into(), stop.into());
    }
    line.insert("on_hull".into(), on_hull.into());
    Value::Object(line)
}

/// The stop codons a stop-labelled lattice lanes, in its lane order.
use crate::molecule::STOP_CODONS;

/// `path` with `codon` before its extension: `front.jsonl` → `front.UAA.jsonl`.
pub fn stop_path(path: &Path, codon: &str) -> PathBuf {
    crate::pool::tagged_path(path, codon)
}

/// The result files a run of `args` writes: one per stop codon under
/// `--stop each` or a list, and `--out` otherwise, `None` standing for stdout.
pub fn outputs(args: &FrontArgs) -> Result<Vec<Option<PathBuf>>> {
    let arms = crate::molecule::stop_arms(args.shared.stop.as_deref())?;
    let out = result::destination(args.shared.out.as_deref()).map(Path::to_path_buf);
    Ok(match (&out, arms.is_empty()) {
        (Some(out), false) => arms.iter().map(|c| Some(stop_path(out, c))).collect(),
        _ => vec![out],
    })
}

/// `derna front` over two or more stop codons: one walk over the lattice that
/// keeps them apart, each arm's front written as `--stop UAA|UAG|UGA` writes
/// it, and the two fronts their union defines written beside them.
fn run_each(args: &FrontArgs, started: Instant, arms: &[String]) -> Result<()> {
    // What a front per stop codon cannot yet do, refused by name rather than
    // accepted and dropped. Every one of these works on a single stop's front,
    // so the way to have it is one run per stop joined by `derna pool` — which
    // is also the cheaper route, three independent walks being parallel where
    // one shared walk is serial.
    //
    // `--motif` needs the motif driver's product lattice to carry the stop
    // lanes. The other two need a per-arm walk run threaded through the shared
    // walk, which it does not take.
    let unsupported = [
        ("--motif", !args.sequence.motifs.is_empty()),
        ("--resume", args.resume.is_some()),
        ("--front-from", args.front_from.is_some()),
    ];
    if let Some((flag, _)) = unsupported.iter().find(|(_, given)| *given) {
        return Err(format!(
            "derna front: a front per stop codon does not take {flag} — run one stop at a \
             time, each with {flag}, and join them with `derna pool`"
        )
        .into());
    }
    if args.checkpoint {
        return Err(
            "derna front: --checkpoint is not carried by a front per stop codon, which \
                    cannot save its progress as it runs — run one stop at a time, which \
                    checkpoints by default, and join them with `derna pool`"
                .into(),
        );
    }
    let Some(out) = result::destination(args.shared.out.as_deref()) else {
        return Err(
            "derna front: a front per stop codon writes one file per stop and needs --out PATH"
                .into(),
        );
    };
    let per_stop: Vec<FrontArgs> = arms
        .iter()
        .map(|codon| {
            let mut a = args.clone();
            a.shared.stop = Some(codon.clone());
            a.shared.out = Some(stop_path(out, codon));
            a.front_out = args.front_out.as_deref().map(|p| stop_path(p, codon));
            a.report = args.report.as_deref().map(|p| stop_path(p, codon));
            a
        })
        .collect();
    for a in &per_stop {
        result::check_output(a.shared.out.as_deref(), a.overwrite())?;
        result::check_output(a.report.as_deref(), a.overwrite())?;
        result::check_output(a.front_out.as_deref(), a.overwrite())?;
    }
    let m = Molecule::resolve(&args.shared)?;
    if args.modular && m.constraint.is_inert() {
        return Err("derna front: --modular needs a constraint — with none, \
                    every design is modular and the front is the plain front"
            .into());
    }
    let threads = args.shared.resolved_threads();
    eprintln!(
        "walking each stop codon's front of {} residue(s) over [{EPSILON:e}, 1 − {EPSILON:e}] \
         on a grid of {} weights with {} thread(s)",
        m.protein.sequence.len(),
        pass_grid(&m).weights().len(),
        threads
    );
    // As for one stop, the budget stays on the modular search.
    let budget = if args.modular {
        Budget::default()
    } else {
        args.budget
    };
    let params = m.params.clone();
    let fronts = design_front_walk_per_stop(
        &params,
        &m.designed_protein(),
        &m.codons.usage,
        &m.utr5_spec(),
        &m.utr3_spec(),
        m.table(),
        EPSILON,
        1.0 - EPSILON,
        walk_options(&m, budget),
        threads,
    )?;
    let mut fronts = fronts;
    let mut written: Vec<PathBuf> = Vec::new();
    for (codon, a) in arms.iter().zip(&per_stop) {
        let lane = STOP_CODONS
            .iter()
            .position(|c| *c == codon.as_str())
            .ok_or_else(|| format!("derna front: {codon} is not a stop codon"))?;
        let Some(walked) = fronts[lane].take() else {
            eprintln!("{codon}: no design ends in this stop codon");
            continue;
        };
        let m = m.with_given_stop(codon);
        let plain = PlainLines {
            args: a,
            m: &m,
            threads,
            started,
            resume: None,
        };
        eprint!("{codon}: ");
        finish_walk(&plain, &walked, None, started)?;
        let Some(path) = a.shared.out.clone() else {
            continue;
        };
        // Each arm is certified as it is written, against its own molecule:
        // the gap is a property of the design's own fold, and the stop codon
        // is part of that design.
        if args.check_points(&m) {
            eprint!("{codon}: ");
            certify_written(&path, &m, args)?;
        }
        written.push(path);
    }
    // The arms are joined only once every one of them carries a verdict,
    // because the modular front is the non-dominated subset of the designs
    // that pass their check and that verdict cannot be recovered after the
    // points are filtered.
    if written.len() > 1 && args.check_points(&m) {
        crate::pool::run(&crate::pool::PoolArgs {
            inputs: written,
            out: out.to_path_buf(),
            force: true,
            quiet: true,
        })?;
    }
    Ok(())
}

/// Certify the front this run has written, in place: every point gains
/// its gap and its verdict, and the front's own header stays.
///
/// Not `derna check`'s write. That command replaces the header with its own and
/// carries the walk's under `front_run`, which is right for a file handed to it
/// and wrong for the file this run wrote: a front's `complete`,
/// `stopped_by` and `open` are what a reader of a front reads, and moving them
/// under another key would break every reader of a plain front run.
fn certify_written(path: &Path, m: &Molecule, args: &FrontArgs) -> Result<()> {
    let read = crate::show::read(Some(path))?;
    let from = &args.shared;
    let certified = if args.check_pool > 0 {
        crate::check::certify_reusing(m, &read.points, from.threads(), args.check_pool)?
    } else {
        crate::check::certify_with(m, &read.points, from.beam_first, from.threads())?
    };
    let optimal = certified.iter().filter(|(_, c)| c.modular()).count();
    eprintln!("{optimal} of {} point(s) modular", certified.len());
    // The header is left exactly as the walk wrote it. The count belongs to
    // the points and `derna show` reads it from them, so putting it in the
    // header would give a reader two places to find one number and a way for
    // them to disagree.
    let mut lines = vec![read.header];
    lines.extend(
        certified
            .iter()
            .map(|(p, c)| crate::check::certified_point(p, c)),
    );
    result::replace_lines(path, &lines)
}

/// Run `derna front`.
pub fn run(args: &FrontArgs) -> Result<()> {
    let started = Instant::now();
    let arms = crate::molecule::stop_arms(args.shared.stop.as_deref())?;
    if !arms.is_empty() {
        return run_each(args, started, &arms);
    }
    // Both files the command writes are refused before the walk rather than
    // after it: a walk is long, and a run that ends by refusing to write has
    // wasted all of it. `--report` is held to the same rule as `--out`,
    // since silently replacing a file is a loss either way.
    if let (Some(out), Some(report)) = (&args.shared.out, &args.report) {
        if out == report {
            return Err(
                "derna front: --out and --report name the same file, and the run writes both"
                    .into(),
            );
        }
    }
    result::check_output(args.shared.out.as_deref(), args.overwrite())?;
    result::check_output(args.report.as_deref(), args.overwrite())?;
    result::check_output(args.front_out.as_deref(), args.overwrite())?;
    let m = Molecule::resolve(&args.shared)?;
    if args.modular && m.constraint.is_inert() {
        return Err("derna front: --modular needs a constraint — with none, \
                    every design is modular and the front is the plain front"
            .into());
    }
    clear_checkpoint(args)?;
    let threads = args.shared.resolved_threads();
    if !args.sequence.motifs.is_empty() {
        return run_motifs(args, &m, threads, started);
    }
    eprintln!(
        "walking the front of {} residue(s) over [{EPSILON:e}, 1 − {EPSILON:e}]{}, \
         on a grid of {} weights with {} thread(s)",
        m.protein.sequence.len(),
        match m.constraint.forbid_names() {
            names if names.is_empty() => ", no constraint".to_string(),
            names => format!(", forbidding {}", names.join(",")),
        },
        // The grid is what decides the walk's shape, and the walk prints
        // nothing again until it is done: a reader deserves to know what it
        // is about to do rather than only what it did.
        pass_grid(&m).weights().len(),
        threads
    );
    let resume = match &args.resume {
        Some(path) => Some(read_plain_resume(path, &m)?),
        None => None,
    };
    if let Some(r) = &resume {
        eprintln!(
            "resuming {} point(s) and {} open box(es)",
            r.points.len(),
            r.open.len()
        );
    }
    let plain = PlainLines {
        args,
        m: &m,
        threads,
        started,
        resume: resume.as_ref(),
    };
    let out = checkpoint_path(args);
    let lines = &plain;
    let hand = out
        .as_deref()
        .map(|out| checkpoint_to(out, move |so_far| lines.replace(out, so_far, true)));
    let walked = match &args.front_from {
        Some(path) => {
            let read = read_front(&m, path)?;
            // Written out again the front would lose what its walk solved
            // inside each box; `--resume` continues it with that kept.
            if read.stopped_by.is_some() && !args.modular {
                return Err(format!(
                    "derna front: {} was stopped with {} box(es) open; --front-from reads \
                     a stopped walk only to search from, with --modular, and \
                     --resume continues it",
                    path.display(),
                    read.open.len()
                )
                .into());
            }
            read
        }
        // With `--modular` the budget stays on the search that
        // follows, as it was before the plain walk could be stopped: a
        // truncated constrained front would narrow that search silently,
        // since the header then reports the modular front's own
        // completeness rather than this walk's.
        //
        // Under `--checkpoint` the walk is kept beside --out as it settles and
        // whole once it is done, so a rerun continues it rather than walking
        // it again.
        None if args.modular => {
            let run = WalkRun {
                resume: resume.as_ref(),
                checkpoint: hand.as_deref(),
            };
            let walked = walk_run(&m, threads, Budget::default(), run)?;
            if let Some(path) = &out {
                plain.replace(path, &walked, false)?;
            }
            walked
        }
        None => {
            let run = WalkRun {
                resume: resume.as_ref(),
                checkpoint: hand.as_deref(),
            };
            walk_run(&m, threads, args.budget, run)?
        }
    };
    finish_walk(&plain, &walked, out.as_deref(), started)?;
    // A constrained front says what each design's energy is under the
    // constraint and nothing about whether its own fold respects it, so the
    // run folds its points unless there is nothing to learn ([`check_points`]).
    // A front written to stdout is not certified: the certification rewrites
    // the file it read, and there is no file.
    match args.shared.out.as_deref() {
        Some(path) if args.check_points(&m) => certify_written(path, &m, args),
        _ => Ok(()),
    }
}

/// What `derna front` does with a plain walk once it is done: the summary
/// line, the constrained front's file, the modular search or
/// the front's own file, and the report.
fn finish_walk(
    plain: &PlainLines<'_>,
    walked: &WalkedFront,
    out: Option<&Path>,
    started: Instant,
) -> Result<()> {
    let (args, m, threads) = (plain.args, plain.m, plain.threads);
    let hull = on_hull(walked);
    eprintln!(
        "{} point(s), {} on the hull, {} gap(s), {} sweep fold(s) ({:.2}s){}{}",
        walked.front.len(),
        hull.iter().filter(|&&h| h).count(),
        walked.gaps.len(),
        walked.sweep_scans,
        started.elapsed().as_secs_f64(),
        match walked.stopped_by {
            Some(why) => format!(
                "; stopped by the {why} budget with {} box(es) open",
                walked.open.len()
            ),
            None => String::new(),
        },
        // The energy the front spans, which is the quantity this walk's cost
        // tracks: the fold slices that range, so a front spanning half the
        // energy costs about half as much. A caller comparing two molecules'
        // runtimes wants it, and the walk has already paid for it.
        match mfe_span(walked) {
            Some(span) => format!(", spanning {span:.1} kcal/mol"),
            None => String::new(),
        }
    );
    if let Some(path) = &args.front_out {
        // Written before the modular walk, so a run stopped by
        // an outer limit still leaves the constrained front behind.
        let mut w = result::open_output(Some(path), args.overwrite())?;
        result::write_lines(w.as_mut(), &plain.lines(walked, false))?;
    }
    if args.modular {
        write_modular(m, walked, threads, args, started)?;
    } else if let Some(path) = out {
        // The checkpoints already replaced the file; the result does too.
        plain.replace(path, walked, false)?;
    } else {
        let mut w = result::open_output(args.shared.out.as_deref(), args.overwrite())?;
        result::write_lines(w.as_mut(), &plain.lines(walked, false))?;
    }
    if let Some(path) = &args.report {
        write_report(path, walked, args.overwrite())?;
    }
    Ok(())
}

/// `derna front --motif`: the walk over the motif-free designs, its header,
/// points and report.
fn run_motifs(args: &FrontArgs, m: &Molecule, threads: usize, started: Instant) -> Result<()> {
    let sequence = &args.sequence;
    for (side, motif, at) in sequence.unremovable(m)? {
        eprintln!(
            "warning: {motif} occurs in the fixed {side} at position {at}; no codon \
             assignment can remove it, and the motifs are forbidden over {}",
            sequence.enforced_phrase()
        );
    }
    if let Some(path) = &args.front_from {
        // A motif front written earlier, complete or stopped: its points and
        // open boxes are the search's floor, as a walk's would be.
        let read = read_motif_front(path, "--front-from", m, sequence)?;
        let front: Vec<FrontPoint> = read.points.into_iter().map(|(p, _)| p).collect();
        let boxes: Vec<ModularBox> = read
            .open
            .iter()
            .map(|&(lo, hi)| ModularBox {
                mfe_lo_kcal_mol: lo.0,
                cai_log_lo: lo.1,
                cai_log_hi: hi.1,
            })
            .collect();
        let Value::Object(mut header) = crate::show::read(Some(path))?.header else {
            return Err("a header is an object".into());
        };
        header.insert("front_from".into(), json!(path.display().to_string()));
        return write_motif_modular(
            args,
            m,
            (&front, boxes),
            Value::Object(header),
            (threads, started),
        );
    }
    eprintln!(
        "walking the front of {} residue(s) avoiding {} over {}{}, with {} thread(s)",
        m.protein.sequence.len(),
        sequence.patterns().join(","),
        sequence.enforced_phrase(),
        match m.constraint.forbid_names() {
            names if names.is_empty() => ", no constraint".to_string(),
            names => format!(", forbidding {}", names.join(",")),
        },
        threads
    );
    let resume = match &args.resume {
        Some(path) => Some(read_motif_front(path, "--resume", m, sequence)?),
        None => None,
    };
    let narrow = match &args.narrow_from {
        Some(path) => Some(read_wider_motif_front(path, m, sequence)?),
        None => None,
    };
    if let Some(w) = &narrow {
        eprintln!(
            "narrowing a front of {} point(s) with {} gap(s) open",
            w.points.len(),
            w.open.len()
        );
    }
    if let Some(r) = &resume {
        eprintln!(
            "resuming from {} point(s) with {} gap(s) open",
            r.points.len(),
            r.open.len()
        );
    }
    let run = MotifLines {
        args,
        m,
        threads,
        started,
        resume: resume.as_ref(),
    };
    let out = checkpoint_path(args);
    let lines = &run;
    let hand = out.as_deref().map(|out| {
        checkpoint_to(out, move |so_far: &MotifWalk| {
            lines
                .lines(so_far, true)
                .and_then(|lines| result::replace_lines(out, &lines))
        })
    });
    // With --modular the TIME budget is the search's, and the walk
    // runs whole unless --walk-secs stops it: the search reads the front and
    // its open boxes as its floor.
    //
    // The memory ceiling is not shared that way, because it is not only a stop.
    // A walk given no `max_rss_gb` also abandons the slice-room and
    // resident-share discipline it keeps when it has one, so it allocates as if
    // nothing bounded it and the cgroup is left to end the leg. Measured on
    // 1273 residues under nine motifs at four threads: no ceiling gives
    // 0 points and a 35.9 GB peak in 532 s, killed at a 36 GB cap, while the
    // same molecule's constrained arm with the ceiling set peaks at 16.9 GB
    // over 3.3 h. A stopped walk is a result and an OOM is not, so the walk
    // keeps the ceiling and reports a partial front the search can floor on.
    let budget = WalkBudget {
        max_secs: if args.modular {
            args.walk_secs
        } else {
            args.budget.max_secs
        },
        max_rss_gb: args.budget.max_rss_gb,
    };
    let walk_run = MotifRun {
        budget,
        resume: resume.as_ref(),
        checkpoint: hand.as_deref(),
        gap_budget: None,
        slice_grow: None,
        gap_fit: None,
        bracket_cuts: None,
        resident_share: None,
        resident_passes: resident_passes(),
        stop_open_share: stop_open_share(),
        gap_folds: args.budget.max_gap_folds.or_else(gap_folds),
        probe_skip: None,
        probe_learn: None,
        fold_threads: args.fold_threads.unwrap_or(1),
        aware_upto: None,
        aware_seeds: motif_aware_seeds(),
        slice_width: motif_slice_width(),
        narrow_from: narrow.as_ref(),
        gap_secs: None,
    };
    let walked = motif_walk_run(m, sequence, threads, walk_run)?;
    let hull = motif_hull(&walked);
    eprintln!(
        "{} point(s), {} on the hull, {} gap(s), {} hull fold(s) ({:.2}s){}",
        walked.front.len(),
        hull.iter().filter(|&&h| h).count(),
        walked.gaps.len(),
        walked.hull_folds,
        started.elapsed().as_secs_f64(),
        match walked.stopped_by {
            None => String::new(),
            Some(why) => format!(
                "; stopped by the {why} budget with {} gap(s) open",
                walked.open.len()
            ),
        }
    );
    let lines = run.lines(&walked, false)?;
    if args.modular {
        // The whole walk, kept for a rerun to continue the search from.
        if let Some(path) = &out {
            result::replace_lines(path, &lines)?;
        }
        if let Some(path) = &args.front_out {
            let mut w = result::open_output(Some(path), args.overwrite())?;
            result::write_lines(w.as_mut(), &lines)?;
        }
        let header = lines
            .into_iter()
            .next()
            .ok_or("a motif front has a header")?;
        let boxes = walked
            .open
            .iter()
            .map(|(lo, hi)| ModularBox {
                mfe_lo_kcal_mol: lo.mfe.to_kcal_mol(),
                cai_log_lo: lo.cai,
                cai_log_hi: hi.cai,
            })
            .collect();
        write_motif_modular(args, m, (&walked.front, boxes), header, (threads, started))?;
        if let Some(path) = &args.report {
            write_motif_report(path, &walked, args.overwrite())?;
        }
        return Ok(());
    }
    match &out {
        // The checkpoints already replaced the file; the result does too.
        Some(path) => result::replace_lines(path, &lines)?,
        None => {
            let mut w = result::open_output(args.shared.out.as_deref(), args.overwrite())?;
            result::write_lines(w.as_mut(), &lines)?;
        }
    }
    if let Some(path) = &args.report {
        write_motif_report(path, &walked, args.overwrite())?;
    }
    // A motif constrains what a design may spell, not which pairs its fold
    // may form, so a motif front under a pair constraint is certified as a
    // plain one is; a front written to stdout has no file to rewrite.
    match result::destination(args.shared.out.as_deref()) {
        Some(path) if args.check_points(m) => certify_written(path, m, args),
        _ => Ok(()),
    }
}

/// `derna front --motif --modular`: the modular
/// front over the motif-free designs, from a motif front's points and open
/// boxes, walked or read back, under the motif front's `header` with the
/// search's fields in place of the walk's.
fn write_motif_modular(
    args: &FrontArgs,
    m: &Molecule,
    (front_points, boxes): (&[FrontPoint], Vec<ModularBox>),
    header: Value,
    (threads, started): (usize, Instant),
) -> Result<()> {
    let sequence = &args.sequence;
    eprintln!(
        "walking the modular front from {} motif-free point(s), {} box(es) open{}",
        front_points.len(),
        boxes.len(),
        match args.budget.max_secs {
            Some(s) => format!(" within {s} s"),
            None => String::new(),
        }
    );
    let positional = sequence
        .positional()?
        .resolve(&m.layout())
        .map_err(|e| format!("--pin: {e}"))?;
    let pins = (!sequence.pins.is_empty()).then_some(&positional.pins);
    let placed = sequence.placed(m)?;
    let motifs = (placed.as_slice(), pins);
    // The motif search takes no epsilon, so its progress is recorded and read
    // back within zero: see the assertion `modular_front::compute` makes on the
    // motif branch.
    let epsilon = 0.0;
    let prior = prior_search(args, m, epsilon)?;
    let log = search_log(args, m, epsilon, prior.as_ref())?;
    let record = |e: modular_front::ModularEvent<'_>| {
        if let Some(log) = &log {
            log.record(e);
        }
    };
    let front = modular_front::compute(
        m,
        (front_points, &boxes),
        Some(motifs),
        args.budget,
        threads,
        (
            prior.as_ref(),
            log.is_some()
                .then_some(&record as modular_front::Record<'_>),
        ),
        (args.top_search, epsilon, args.policy),
    )?;
    let certified = front.points.iter().filter(|p| p.certified).count();
    eprintln!(
        "{} modular point(s), {certified} of them certified, from {} modular seed(s); {}",
        front.points.len(),
        front.seeds,
        modular_front::stop_line(&front),
    );
    if let Some(note) = modular_front::top_band_note(&front) {
        eprintln!("{note}");
    }
    let Value::Object(mut header) = header else {
        return Err("a header is an object".into());
    };
    header.remove("gaps");
    header.remove("checkpoint");
    header.extend(modular_front::header_fields(
        &front,
        front_points.len(),
        args.budget,
        args.policy,
    ));
    header.insert(
        "resumed_search".into(),
        resumed_search_field(args, prior.as_ref()),
    );
    header.insert("wall_s".into(), started.elapsed().as_secs_f64().into());
    header.insert("peak_rss_kb".into(), json!(result::peak_rss_kb()));
    let header = Value::Object(header);
    let mut w = result::open_output(args.shared.out.as_deref(), args.overwrite())?;
    result::write_lines(w.as_mut(), std::slice::from_ref(&header))?;
    let params = m.params.clone();
    for batch in front.points.chunks(EMIT_BATCH.max(threads)) {
        let lines = modular_front::points(m, &params, batch, (threads, 0.0))?;
        result::write_lines(w.as_mut(), &result::with_metrics(&header, lines))?;
    }
    landed(args, &front);
    Ok(())
}

/// A checkpoint that hands the walk so far to `write`, which replaces the
/// result file at `out` with it.
///
/// One that cannot be written is reported once and the walk goes on: its
/// result is still written at the end.
fn checkpoint_to<'a, W>(
    out: &'a Path,
    write: impl Fn(&W) -> Result<()> + Sync + 'a,
) -> Box<dyn Fn(&W) + Sync + 'a> {
    let unwritten = std::sync::atomic::AtomicBool::new(false);
    Box::new(move |so_far: &W| {
        if let Err(e) = write(so_far) {
            if !unwritten.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "warning: a checkpoint of {} was not written: {e}",
                    out.display()
                );
            }
        }
    })
}

/// What a plain front's result lines are written from, at the end of the
/// walk and at each checkpoint.
pub struct PlainLines<'a> {
    /// The command's arguments.
    pub args: &'a FrontArgs,
    /// The molecule walked.
    pub m: &'a Molecule,
    /// The threads the walk runs on.
    pub threads: usize,
    /// When the run started, for the header's `wall_s`.
    pub started: Instant,
    /// The earlier front the walk continues, if any.
    pub resume: Option<&'a WalkResume>,
}

impl PlainLines<'_> {
    /// The header and points of `walked`; `checkpoint` marks a walk still
    /// running, whose `open` boxes include every pair it has not settled.
    pub fn lines(&self, walked: &WalkedFront, checkpoint: bool) -> Vec<Value> {
        let mut extra = Map::new();
        extra.insert("checkpoint".into(), checkpoint.into());
        let header = front_header(
            self.m,
            walked,
            self.threads,
            self.args,
            self.started,
            extra,
            self.resume,
        );
        let hull = on_hull(walked);
        let mut lines = vec![header];
        lines.extend(
            walked
                .front
                .iter()
                .zip(&hull)
                .map(|(p, &h)| point(self.m, p, h)),
        );
        lines
    }

    /// Replace the file at `out` with the lines of `walked`, as
    /// [`result::replace_lines`] does: a reader finds the old file whole or
    /// the new one.
    ///
    /// # Errors
    ///
    /// When the file cannot be written.
    pub fn replace(&self, out: &Path, walked: &WalkedFront, checkpoint: bool) -> Result<()> {
        result::replace_lines(out, &self.lines(walked, checkpoint))
    }
}

/// Whether each point of a motif walk's front is one of its hull vertices.
fn motif_hull(walked: &MotifWalk) -> Vec<bool> {
    walked
        .front
        .iter()
        .map(|p| {
            walked.vertices.iter().any(|v| {
                (v.mfe.to_kcal_mol() - p.mfe_kcal_mol).abs() <= SAME_POINT
                    && (v.cai - p.cai_log).abs() <= SAME_POINT
            })
        })
        .collect()
}

/// What a motif front's result lines are written from, at the end of the
/// walk and at each checkpoint.
struct MotifLines<'a> {
    args: &'a FrontArgs,
    m: &'a Molecule,
    threads: usize,
    started: Instant,
    resume: Option<&'a MotifResume>,
}

impl MotifLines<'_> {
    /// The header and points of `walked`; `checkpoint` marks a walk still
    /// running, whose `open` boxes include the gaps not yet reached.
    fn lines(&self, walked: &MotifWalk, checkpoint: bool) -> Result<Vec<Value>> {
        let (args, m) = (self.args, self.m);
        let mut extra = args.sequence.header_fields(m)?;
        extra.insert("range".into(), json!([EPSILON, 1.0 - EPSILON]));
        extra.insert("pass_grid".into(), Value::Null);
        extra.insert("points".into(), walked.front.len().into());
        extra.insert("gaps".into(), walked.gaps.len().into());
        extra.insert("complete".into(), walked.open.is_empty().into());
        extra.insert("checkpoint".into(), checkpoint.into());
        extra.insert("stopped_by".into(), json!(walked.stopped_by));
        extra.insert("max_secs".into(), json!(args.budget.max_secs));
        extra.insert("max_rss_gb".into(), json!(args.budget.max_rss_gb));
        extra.insert(
            "resumed".into(),
            match (&args.resume, self.resume) {
                (Some(path), Some(r)) => json!({
                    "from": path.display().to_string(),
                    "points": r.points.len(),
                    "open": r.open.len(),
                }),
                _ => Value::Null,
            },
        );
        // A partly solved box also carries the energy its gap is solved up
        // to and the best adaptation there. Its missing points all lie above
        // `mfe_from`, so it leaves `mfe_hi - max(mfe_from, mfe_lo)` unsettled,
        // not `mfe_hi - mfe_lo`; `mfe_from` sits one hundredth below
        // `mfe_lo` where the box's probe ran and no slice was solved.
        let boxes: Vec<OpenBox> = walked
            .open
            .iter()
            .map(|(a, b)| ((a.mfe.to_kcal_mol(), a.cai), (b.mfe.to_kcal_mol(), b.cai)))
            .collect();
        if let Some(why) = unwritable_open_boxes(&boxes) {
            return Err(format!("derna front: refusing to write the front: {why}").into());
        }
        let open: Vec<Value> = walked
            .open
            .iter()
            .map(|(a, b)| {
                let mut open = json!({
                    "mfe_lo": a.mfe.to_kcal_mol(),
                    "mfe_hi": b.mfe.to_kcal_mol(),
                    "cai_log_lo": a.cai,
                    "cai_log_hi": b.cai,
                });
                let rest = walked.rests.iter().find(|r| r.lo == a.mfe && r.hi == b.mfe);
                if let Some(r) = rest {
                    open["mfe_from"] = r.from.to_kcal_mol().into();
                    open["cai_log_floor"] = r.floor.into();
                }
                // How many times the box's gap moved on under the per-gap
                // time cap, so a resume takes up its count; absent for none.
                let moves = walked.moves.iter().find(|m| m.0 == a.mfe && m.1 == b.mfe);
                if let Some(&(_, _, moves)) = moves {
                    open["moves"] = moves.into();
                }
                open
            })
            .collect();
        extra.insert("open".into(), open.into());
        if walked.rehull {
            extra.insert("vertices_from_front".into(), true.into());
        }
        extra.insert("threads".into(), self.threads.into());
        extra.insert("deterministic".into(), args.shared.deterministic.into());
        extra.insert("wall_s".into(), self.started.elapsed().as_secs_f64().into());
        extra.insert("peak_rss_kb".into(), json!(result::peak_rss_kb()));
        let mut lines = vec![result::header("front", m, extra)];
        let hull = motif_hull(walked);
        lines.extend(walked.front.iter().zip(&hull).map(|(p, &h)| point(m, p, h)));
        Ok(lines)
    }
}

/// An earlier motif front to resume, read from its result file.
///
/// It must be a motif front of this molecule, flanks, constraint, codon table
/// and motifs: its points are taken as front points and its vertices as hull
/// vertices without a fold, so a file of any other run would be returned as
/// though it were this one's.
fn read_motif_front(
    path: &Path,
    flag: &str,
    m: &Molecule,
    sequence: &SequenceConstraints,
) -> Result<MotifResume> {
    read_motif_front_under(path, flag, m, sequence, |theirs, ours| theirs == ours)
}

/// A motif front of the same molecule, flanks, constraint, codon table and
/// pins under a subset of this run's motifs, for `--narrow-from`: its points
/// are not taken as this run's until the walk has kept those that spell none
/// of this run's motifs.
fn read_wider_motif_front(
    path: &Path,
    m: &Molecule,
    sequence: &SequenceConstraints,
) -> Result<MotifResume> {
    read_motif_front_under(path, "--narrow-from", m, sequence, |theirs, ours| {
        theirs.iter().all(|t| ours.contains(t))
    })
}

/// [`read_motif_front`] with the file's motif set compared to this run's by
/// `motifs`, each as a sorted list.
fn read_motif_front_under(
    path: &Path,
    flag: &str,
    m: &Molecule,
    sequence: &SequenceConstraints,
    motifs: impl Fn(&[String], &[String]) -> bool,
) -> Result<MotifResume> {
    let read = crate::show::read(Some(path))?;
    let refuse = |why: String| format!("derna front: {flag} {}: {why}", path.display());
    let h = &read.header;
    if h["command"] != "front" || h.get("open").is_none() || h.get("motifs").is_none() {
        return Err(refuse("not a motif front's result".into()).into());
    }
    let mut ours = result::molecule_fields(m);
    ours.extend(sequence.header_fields(m)?);
    // A motif set is a set: the order the motifs were given in is not part of
    // the molecule.
    let as_set = |v: &Value| {
        let mut items: Vec<String> = v
            .as_array()
            .into_iter()
            .flatten()
            .map(Value::to_string)
            .collect();
        items.sort_unstable();
        items
    };
    for (key, value) in &ours {
        let same = match h.get(key) {
            // A motif set is a set, and so is the set of its patterns paired
            // with the windows they are forbidden inside: the order the
            // motifs were given in is not part of the molecule, while which
            // pattern carried which window is.
            Some(theirs) if key == "motifs" || key == "motif_scopes" => {
                motifs(&as_set(theirs), &as_set(value))
            }
            theirs => theirs == Some(value),
        };
        if !same {
            return Err(refuse(format!("its {key} differs from this run's")).into());
        }
    }
    // A field is written only when it constrains something, so one the
    // earlier run wrote and this run does not is a difference too.
    if h.get("pins").is_some() && !ours.contains_key("pins") {
        return Err(refuse("it was pinned and this run is not".into()).into());
    }
    // `motif_scopes` is written only when some pattern carries a window of its
    // own, and `motifs` with one `motif_window` is the same in both runs when
    // one of them scopes a pattern and the other does not — so the absence has
    // to be read rather than skipped.
    if h.get("motif_scopes").is_some() && !ours.contains_key("motif_scopes") {
        return Err(refuse(
            "it forbade some motif over a window of its own and this run forbids every \
             motif over the same window"
                .into(),
        )
        .into());
    }
    let points = resume_points(&read, &refuse)?;
    let vertex = |mfe: f64, cai_log: f64| {
        points.iter().any(|(p, on_hull)| {
            *on_hull
                && (p.mfe_kcal_mol - mfe).abs() <= SAME_POINT
                && (p.cai_log - cai_log).abs() <= SAME_POINT
        })
    };
    let open = resume_boxes(h, &refuse)?;
    let among = |mfe: f64, cai_log: f64| {
        points.iter().any(|(p, _)| {
            (p.mfe_kcal_mol - mfe).abs() <= SAME_POINT && (p.cai_log - cai_log).abs() <= SAME_POINT
        })
    };
    let rests = resume_rests(h, &refuse)?;
    for (lo, hi) in &open {
        // The motif walk writes a design for both ends of every box it
        // leaves open, so a box whose ends are not among them is not this
        // walk's box. Only a narrowed walk opens a box between points that
        // are not hull vertices, and it solves such a box without a probe,
        // from its rest.
        let rest = rests
            .iter()
            .any(|((a, b), _)| (a - lo.0).abs() <= SAME_POINT && (b - hi.0).abs() <= SAME_POINT);
        if !among(lo.0, lo.1) || !among(hi.0, hi.1) {
            return Err(refuse("an open gap's ends are not among its points".into()).into());
        }
        if !rest && (!vertex(lo.0, lo.1) || !vertex(hi.0, hi.1)) {
            return Err(refuse("an open gap's ends are not among its hull points".into()).into());
        }
    }
    let rehull = h["vertices_from_front"].as_bool() == Some(true);
    Ok(MotifResume {
        points,
        open,
        rests,
        rehull,
        moves: resume_moves(h),
    })
}

/// The boxes of a motif front's header whose gap moved on under the per-gap
/// time cap, as their ends' energies with how many times; a box without a
/// `moves` field moved on none.
fn resume_moves(h: &Value) -> Vec<((f64, f64), u32)> {
    h["open"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|b| {
            let moves = u32::try_from(b["moves"].as_u64()?).ok()?;
            Some(((b["mfe_lo"].as_f64()?, b["mfe_hi"].as_f64()?), moves))
        })
        .filter(|&(_, moves)| moves > 0)
        .collect()
}

/// One partly solved box, as its ends' energies, then the energy up to which
/// its gap is solved and the best adaptation there.
type OpenRest = ((f64, f64), (f64, f64));

/// The partly solved boxes of a motif front's header, as their ends'
/// energies with the energy up to which each gap is solved and the best
/// adaptation there.
fn resume_rests(h: &Value, refuse: &impl Fn(String) -> String) -> Result<Vec<OpenRest>> {
    let mut rests = Vec::new();
    for b in h["open"].as_array().into_iter().flatten() {
        let (Some(from), Some(floor)) = (b["mfe_from"].as_f64(), b["cai_log_floor"].as_f64())
        else {
            continue;
        };
        let (lo, hi) = (b["mfe_lo"].as_f64(), b["mfe_hi"].as_f64());
        let (Some(lo), Some(hi)) = (lo, hi) else {
            return Err(refuse("an open box has no mfe_lo or mfe_hi".into()).into());
        };
        // Solved from immediately below the stable end, and short of the adapted one.
        if !(lo - 0.01 - SAME_POINT <= from && from < hi) {
            return Err(refuse("a partly solved gap is solved past its box".into()).into());
        }
        rests.push(((lo, hi), (from, floor)));
    }
    Ok(rests)
}

/// The points of a result to resume from, each with whether it is marked as a
/// hull vertex.
fn resume_points(
    read: &crate::show::Rendered,
    refuse: &impl Fn(String) -> String,
) -> Result<Vec<(FrontPoint, bool)>> {
    let number = |p: &Value, key: &str| {
        p[key]
            .as_f64()
            .ok_or_else(|| refuse(format!("a point has no {key}")))
    };
    let text = |p: &Value, key: &str| {
        p[key]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| refuse(format!("a point has no {key}")))
    };
    let mut points = Vec::new();
    for p in &read.points {
        let point = FrontPoint {
            sequence: text(p, "sequence")?,
            structure: text(p, "structure")?,
            mfe_kcal_mol: number(p, "mfe")?,
            cai_log: number(p, "cai_log")?,
            cai: number(p, "cai")?,
            utr_profile_log: 0.0,
        };
        points.push((point, p["on_hull"].as_bool().unwrap_or(false)));
    }
    Ok(points)
}

/// One open box, as its two ends' objectives `(mfe kcal/mol, cai_log)`, the
/// more stable first.
type OpenBox = ((f64, f64), (f64, f64));

/// Whether an open box's ends are its more stable then its more adapted point;
/// a lone vertex is the gap from itself to itself.
fn box_ends_ordered(lo: (f64, f64), hi: (f64, f64)) -> bool {
    lo == hi || (lo.0 < hi.0 && lo.1 < hi.1)
}

/// What stops a result's open boxes being written: a box a resume would
/// refuse, or one listed twice. A file with either cannot be continued, and
/// the walk that holds them cannot say why, so none is written.
fn unwritable_open_boxes(open: &[OpenBox]) -> Option<String> {
    if let Some(&((lm, lc), (hm, hc))) = open.iter().find(|&&(lo, hi)| !box_ends_ordered(lo, hi)) {
        return Some(format!(
            "an open gap's ends are not its more stable then its more adapted point: \
             ({lm}, {lc}) then ({hm}, {hc})"
        ));
    }
    for (i, b) in open.iter().enumerate() {
        if open[..i].contains(b) {
            return Some(format!(
                "an open gap from ({}, {}) to ({}, {}) is listed twice",
                b.0 .0, b.0 .1, b.1 .0, b.1 .1
            ));
        }
    }
    None
}

/// The boxes a result's header leaves open, as their two ends' objectives.
fn resume_boxes(h: &Value, refuse: &impl Fn(String) -> String) -> Result<Vec<OpenBox>> {
    let number = |p: &Value, key: &str| {
        p[key]
            .as_f64()
            .ok_or_else(|| refuse(format!("an open box has no {key}")))
    };
    let mut open = Vec::new();
    for b in h["open"].as_array().into_iter().flatten() {
        let (lo, hi) = (
            (number(b, "mfe_lo")?, number(b, "cai_log_lo")?),
            (number(b, "mfe_hi")?, number(b, "cai_log_hi")?),
        );
        // The more stable end first, and the more adapted one second; a lone
        // vertex is the gap from itself to itself.
        if !box_ends_ordered(lo, hi) {
            return Err(refuse(
                "an open gap's ends are not its more stable then its more adapted point".into(),
            )
            .into());
        }
        open.push((lo, hi));
    }
    Ok(open)
}

/// The plain constrained front of `m` read from a `front` result to continue.
///
/// Refused unless it is a plain front of this molecule, flanks, constraint and
/// codon table: its points are taken as front points and its open boxes as
/// pairs of hull points without a fold, so a file of any other run would be
/// returned as though it were this one's. Unlike the motif front, a box's ends
/// need not be among the points: a stopped plain walk cannot write a design
/// for a hull vertex it only folded scalar.
fn read_plain_resume(path: &Path, m: &Molecule) -> Result<WalkResume> {
    let read = crate::show::read(Some(path))?;
    let refuse = |why: String| format!("derna front: --resume {}: {why}", path.display());
    let h = &read.header;
    if h["command"] != "front" || h.get("open").is_none() {
        return Err(refuse("not a `derna front` result with its open boxes".into()).into());
    }
    if h.get("motifs").is_some() {
        return Err(refuse(
            "it is a motif front, and a motif front is a different front; \
             --resume it with --motif"
                .into(),
        )
        .into());
    }
    if h.get("modular_front").is_some() {
        return Err(refuse("it is a modular front, not a plain one".into()).into());
    }
    if let Some(field) = molecule_mismatch(&result::molecule_fields(m), h) {
        return Err(refuse(format!("its {field} differs from this run's")).into());
    }
    Ok(WalkResume {
        points: resume_points(&read, &refuse)?,
        open: resume_boxes(h, &refuse)?,
        rests: resume_rests(h, &refuse)?,
    })
}

/// The motif walk's diagnostics: the hull's vertices and each gap.
fn write_motif_report(path: &Path, walked: &MotifWalk, force: bool) -> Result<()> {
    result::check_output(Some(path), force)?;
    let report = json!({
        "vertices": walked.vertices.iter().map(|v| json!({
            "mfe": v.mfe.to_kcal_mol(),
            "adaptation_log": v.cai,
        })).collect::<Vec<_>>(),
        "gaps": walked.gaps.iter().map(|g| json!({
            "lambda": g.lambda,
            "points": g.points,
            "folds": g.folds,
            "cut_s": g.cut_s,
            "fold_s": g.fold_s,
            "settled_s": g.settled_s,
            "aware_out": g.aware_out,
            "aware_dropped": g.aware_dropped,
            "aware_unkeyed": g.aware_unkeyed,
            "carried": g.carried,
            "sized": g.sized,
            "adopted": g.adopted,
            "adopted_points": g.adopted_points,
        })).collect::<Vec<_>>(),
        "passes": walked.passes.iter().map(|p| json!({
            "lambda": p.lambda,
            "secs": p.secs,
            // `null`, not 0, where the reading could not be taken: a pass
            // that held nothing and a reading that was not taken are
            // different facts, and 0 reads as the first.
            "rss_kb": p.rss_kb,
            "aware_secs": p.aware_secs,
            "aware_folds": p.aware_folds,
            "aware_margin": p.aware_margin,
            "aware_cells": p.aware_cells,
            "aware_pairs": p.aware_pairs,
            "aware_read": p.aware_read,
        })).collect::<Vec<_>>(),
        "probe_records": walked.probe_records.iter().map(|p| json!({
            "lambda": p.lambda,
            "secs": p.secs,
            "folds": p.folds,
            "pruned": p.pruned,
            "cut": p.cut,
            "aware_pruned": p.aware_pruned,
            "aware_cut": p.aware_cut,
        })).collect::<Vec<_>>(),
        "seeded": walked.seeded,
        "hull_folds": walked.hull_folds,
        "hull_s": walked.hull_s,
        "ends_s": walked.ends_s,
        "ends_peak_kb": walked.ends_peak_kb,
        "adapted_narrow": walked.adapted_narrow,
        "probe_s": walked.probe_s,
        "probes": walked.probes,
        "probe_folds": walked.probe_folds,
        "probes_under_chord": walked.probes_under_chord,
        "gaps_moved_on": walked.gaps_moved_on,
        "points": walked.front.len(),
        "phases": phase_seconds(walked),
    });
    std::fs::write(path, format!("{report}\n"))?;
    Ok(())
}

/// The seconds each phase of a motif walk took, summed over its records: the
/// plain passes, their passes over the constrained lattice, the gaps' front
/// folds, the probes and the two ends. Phases run side by side on threads, so
/// the sum can exceed the walk's wall time.
fn phase_seconds(walked: &MotifWalk) -> Value {
    json!({
        "plain_s": walked.passes.iter().map(|p| p.secs).sum::<f64>(),
        "aware_s": walked.passes.iter().map(|p| p.aware_secs).sum::<f64>(),
        "gap_fold_s": walked.gaps.iter().map(|g| g.fold_s).sum::<f64>(),
        "probe_s": walked.probe_s,
        "ends_s": walked.ends_s,
    })
}

/// How many points one batch of the emission writes.
///
/// A batch is what a kill can cost: the points are written as they are made,
/// so the bound on what an outer limit destroys is one batch rather than the
/// whole emission. It is large enough that every worker is handed points —
/// the batch is spread over the same threads that would emit the whole front
/// — and the lines do not depend on it, since each one reads nothing but its
/// own point and is written in the front's order.
const EMIT_BATCH: usize = 64;

/// Write the modular front: its header, then its points as they
/// are emitted.
///
/// Emitting the points is the long phase of a budgeted run and no budget
/// stops it — `--max-secs` governs the search that precedes it — so the
/// lines go to `--out` as they are made rather than at the end. A run killed
/// partway then leaves a readable file: the header is already written, and it
/// records how many points the front holds and whether the search was
/// complete, so a reader sees the points are a prefix by counting them.
/// The points of an anchors file, and those within the run's epsilon here as
/// (constrained energy in kcal/mol, coding log-adaptation, molecule).
struct Anchors {
    points: usize,
    found: Vec<(f64, f64, String)>,
}

/// Check every point of `path`, a `derna` result for this molecule under any
/// constraint, against this run's constraint and `epsilon`. A point that
/// passes is a member of the set searched here wherever it came from, so it
/// may cap the search; one that fails is dropped. The molecule — protein, flanks, stop
/// codon and codon table — must be this run's, or the designs are another
/// molecule's.
fn read_anchors(m: &Molecule, path: &Path, threads: usize, epsilon: f64) -> Result<Anchors> {
    let read = show::read(Some(path))?;
    let mut fields = result::molecule_fields(m);
    fields.remove("forbid");
    fields.remove("access");
    if let Some(field) = molecule_mismatch(&fields, &read.header) {
        return Err(format!(
            "{}: its {field} is not this run's, so its points are another molecule's",
            path.display()
        )
        .into());
    }
    let points: Vec<(String, f64)> = read
        .points
        .iter()
        .map(|p| -> Result<(String, f64)> {
            let seq = p["sequence"]
                .as_str()
                .ok_or("an anchor point has no sequence")?;
            let cai = p["cai_log"]
                .as_f64()
                .ok_or("an anchor point has no cai_log")?;
            Ok((seq.to_string(), cai))
        })
        .collect::<Result<_>>()?;
    let params = m.params.clone();
    let (cds, table) = (m.cds(), m.table());
    let workers = threads.clamp(1, points.len().max(1));
    // Each worker's anchors within `epsilon` as (mfe, cai_log, sequence).
    type Kept = std::result::Result<Vec<(f64, f64, String)>, String>;
    let checked: Vec<Kept> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                let (points, params) = (&points, &params);
                scope.spawn(move || {
                    let mut kept = Vec::new();
                    for (seq, cai) in points.iter().skip(w).step_by(workers) {
                        let so = derna_lib::sequence_mfe_within(params, seq, cds, table, epsilon)
                            .map_err(|e| e.to_string())?;
                        if let Some(mfe) = so {
                            kept.push((mfe, *cai, seq.clone()));
                        }
                    }
                    Ok(kept)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("an anchor worker"))
            .collect()
    });
    let mut found = Vec::new();
    for c in checked {
        found.extend(c?);
    }
    Ok(Anchors {
        points: points.len(),
        found,
    })
}

/// Whether two paths name one file.
fn same_file(a: &std::path::Path, b: &std::path::Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// The progress `--resume-search` carries into a modular search, read from
/// the file a stopped run left.
///
/// A search within another `epsilon` is refused rather than merged: a verdict
/// or a find admitted to another set would cap this search unsoundly. So is
/// another molecule's search.
fn prior_search(
    args: &FrontArgs,
    m: &Molecule,
    epsilon: f64,
) -> Result<Option<modular_front::ModularProgress>> {
    let Some(path) = &args.resume_search else {
        return Ok(None);
    };
    let p = modular_front::read_progress(path)?;
    refuse_another_runs_search(m, path)?;
    if p.epsilon != epsilon {
        return Err(format!(
            "derna front: {} was searched within {} kcal/mol and this run is within \
             {epsilon}; resume it with --epsilon {}",
            path.display(),
            p.epsilon,
            p.epsilon
        )
        .into());
    }
    eprintln!(
        "resuming the search: {} point(s) checked, {} design(s) found{}",
        p.checked.len(),
        p.found.len(),
        match p.band_down_to {
            Some(d) => format!(", the band checked down to {d:.4}"),
            None => String::new(),
        }
    );
    Ok(Some(p))
}

/// Where `--checkpoint` records a modular search, beside `--out`.
///
/// A resumed run's log holds what it resumed too, so one file is always enough
/// to resume from; appending to the file it read already does.
fn search_log(
    args: &FrontArgs,
    m: &Molecule,
    epsilon: f64,
    prior: Option<&modular_front::ModularProgress>,
) -> Result<Option<modular_front::ProgressLog>> {
    let Some(out) = result::destination(args.shared.out.as_deref()).filter(|_| args.checkpoint)
    else {
        return Ok(None);
    };
    let path = modular_front::progress_path(out);
    let log = modular_front::ProgressLog::open(&path, epsilon)?;
    log.identify(Value::Object(result::molecule_fields(m)));
    if let (Some(p), Some(from)) = (prior, &args.resume_search) {
        if !same_file(from, &path) {
            log.carry(p);
        }
    }
    Ok(Some(log))
}

/// What a run resumed, for the result header: the file it read and how much of
/// the search it carried in.
fn resumed_search_field(args: &FrontArgs, prior: Option<&modular_front::ModularProgress>) -> Value {
    match (&args.resume_search, prior) {
        (Some(path), Some(p)) => json!({
            "from": path.display().to_string(),
            "checked": p.checked.len(),
            "found": p.found.len(),
            "band_down_to": p.band_down_to,
        }),
        _ => Value::Null,
    }
}

fn write_modular(
    m: &Molecule,
    walked: &WalkedFront,
    threads: usize,
    args: &FrontArgs,
    started: Instant,
) -> Result<()> {
    eprintln!(
        "walking the {} from {} constrained point(s){}",
        modular_front::which_front(args.epsilon.unwrap_or(0.0)),
        walked.front.len(),
        match args.budget.max_secs {
            Some(s) => format!(" within {s} s"),
            None => String::new(),
        }
    );
    let epsilon = args.epsilon.unwrap_or(0.0);
    let prior = prior_search(args, m, epsilon)?;
    let anchors = match &args.anchors_from {
        Some(path) => Some(read_anchors(m, path, threads, epsilon)?),
        None => None,
    };
    let prior = match (prior, &anchors) {
        (prior, Some(a)) => {
            eprintln!(
                "{} of {} anchor point(s) {}",
                a.found.len(),
                a.points,
                modular_front::admitted(epsilon)
            );
            let mut p = prior.unwrap_or_else(|| modular_front::progress_within(epsilon));
            p.found.extend(a.found.iter().cloned());
            Some(p)
        }
        (prior, None) => prior,
    };
    let log = search_log(args, m, epsilon, prior.as_ref())?;
    let record = |e: modular_front::ModularEvent<'_>| {
        if let Some(log) = &log {
            log.record(e);
        }
    };
    // A constrained front read back from a stopped walk floors the search with
    // the boxes it left open as well as its points.
    let boxes: Vec<ModularBox> = walked
        .open
        .iter()
        .map(|(lo, hi)| ModularBox {
            mfe_lo_kcal_mol: lo.mfe_kcal_mol,
            cai_log_lo: lo.adaptation_log,
            cai_log_hi: hi.adaptation_log,
        })
        .collect();
    let front = modular_front::compute(
        m,
        (&walked.front, &boxes),
        None,
        args.budget,
        threads,
        (
            prior.as_ref(),
            log.is_some()
                .then_some(&record as modular_front::Record<'_>),
        ),
        (args.top_search, epsilon, args.policy),
    )?;
    // Three counts, and each says what it counts: the points the run emits,
    // how many of those are certified against what is still open, and the
    // constrained-front points that were modular and seeded the search. On a
    // run that certifies every point and finds nothing all three are equal,
    // so one word serving two of them reads correctly there and nowhere else.
    let certified = front.points.iter().filter(|p| p.certified).count();
    eprintln!(
        "{} point(s) on the {}, {certified} of them certified, from {} modular seed(s); {} ({:.2}s)",
        front.points.len(),
        modular_front::which_front(epsilon),
        front.seeds,
        modular_front::stop_line(&front),
        started.elapsed().as_secs_f64()
    );
    if let Some(note) = modular_front::top_band_note(&front) {
        eprintln!("{note}");
    }
    let mut extra =
        modular_front::header_fields(&front, walked.front.len(), args.budget, args.policy);
    extra.insert("constrained_open".into(), boxes.len().into());
    extra.insert(
        "anchors".into(),
        match (&args.anchors_from, &anchors) {
            (Some(path), Some(a)) => json!({
                "from": path.display().to_string(),
                "points": a.points,
                "modular": a.found.len(),
            }),
            _ => Value::Null,
        },
    );
    extra.insert(
        "resumed_search".into(),
        resumed_search_field(args, prior.as_ref()),
    );
    let header = front_header(m, walked, threads, args, started, extra, None);
    let mut w = result::open_output(args.shared.out.as_deref(), args.overwrite())?;
    result::write_lines(w.as_mut(), std::slice::from_ref(&header))?;
    let params = m.params.clone();
    let emitting = Instant::now();
    // The gaps are known only once each point is folded, after the header is
    // out, so their histogram goes to stderr; the points' own `gap` fields
    // give it again from the file.
    let mut gaps = modular_front::GapHistogram::default();
    for batch in front.points.chunks(EMIT_BATCH.max(threads)) {
        let lines = modular_front::points(m, &params, batch, (threads, epsilon))?;
        gaps.count(&lines);
        result::write_lines(w.as_mut(), &result::with_metrics(&header, lines))?;
    }
    eprintln!(
        "wrote {} point(s), by gap {gaps} ({:.2}s)",
        front.points.len(),
        emitting.elapsed().as_secs_f64()
    );
    landed(args, &front);
    Ok(())
}

/// A search progress file names the molecule it was written for; one of
/// another molecule or constraint would cap this search with verdicts that are
/// not this molecule's, so it is refused. A file that names no run predates
/// the field and is read as before.
fn refuse_another_runs_search(m: &Molecule, path: &Path) -> Result<()> {
    let Some(run) = modular_front::progress_run(path)? else {
        return Ok(());
    };
    match molecule_mismatch(&result::molecule_fields(m), &run) {
        Some(field) => Err(format!(
            "derna front: {} is another run's search progress: its {field} differs from \
             this run's; pass --force to start over",
            path.display()
        )
        .into()),
        None => Ok(()),
    }
}

/// Once a complete modular front has landed, the walk kept beside it under
/// `--checkpoint` has nothing left to protect. A stopped front keeps it, and
/// so does one that stopped below a band, so
/// running the command again continues the search from the same walk. The
/// search's own progress file stays either way: it is a record of what the
/// search checked as well as a way to resume it.
fn landed(args: &FrontArgs, front: &derna_lib::modular_front::ModularFront) {
    let Some(path) = checkpoint_path(args) else {
        return;
    };
    // A front that stopped below a band keeps it too: the band is unsearched,
    // so a rerun that searches it has work left, and throwing the walk away
    // would make that rerun walk the constrained front again.
    let whole = modular_front::stopped_by(front).is_none() && front.top_band.is_none();
    if whole && path.exists() {
        if let Err(err) = std::fs::remove_file(&path) {
            eprintln!("derna front: {} was left behind: {err}", path.display());
        }
    }
}

fn front_header(
    m: &Molecule,
    walked: &WalkedFront,
    threads: usize,
    args: &FrontArgs,
    started: Instant,
    so: Map<String, Value>,
    resume: Option<&WalkResume>,
) -> Value {
    let grid = pass_grid(m);
    let mut extra = Map::new();
    extra.insert("range".into(), json!([EPSILON, 1.0 - EPSILON]));
    extra.insert("pass_grid".into(), json!(grid.segments));
    extra.insert("points".into(), walked.front.len().into());
    extra.insert("gaps".into(), walked.gaps.len().into());
    // The same fields the motif front writes, so one reader covers both: what
    // the walk settled, what stopped it, and the boxes it left between hull
    // points.
    extra.insert("complete".into(), walked.open.is_empty().into());
    extra.insert("stopped_by".into(), json!(walked.stopped_by));
    extra.insert("max_secs".into(), json!(args.budget.max_secs));
    extra.insert("max_rss_gb".into(), json!(args.budget.max_rss_gb));
    extra.insert(
        "resumed".into(),
        match (&args.resume, resume) {
            (Some(path), Some(r)) => json!({
                "from": path.display().to_string(),
                "points": r.points.len(),
                "open": r.open.len(),
            }),
            _ => Value::Null,
        },
    );
    // A partly solved box also carries the energy its gap is solved up to
    // and the best adaptation there, as the motif front writes it.
    extra.insert(
        "open".into(),
        walked
            .open
            .iter()
            .map(|(a, b)| {
                let mut open = json!({
                    "mfe_lo": a.mfe_kcal_mol,
                    "mfe_hi": b.mfe_kcal_mol,
                    "cai_log_lo": a.adaptation_log,
                    "cai_log_hi": b.adaptation_log,
                });
                let (lo, hi) = (energy_units(a.mfe_kcal_mol), energy_units(b.mfe_kcal_mol));
                let rest = walked.rests.iter().find(|r| r.lo.0 == lo && r.hi.0 == hi);
                if let Some(r) = rest {
                    open["mfe_from"] = r.from.to_kcal_mol().into();
                    open["cai_log_floor"] = r.floor.into();
                }
                open
            })
            .collect::<Vec<Value>>()
            .into(),
    );
    extra.extend(so);
    extra.insert(
        "front_from".into(),
        json!(args.front_from.as_ref().map(|p| p.display().to_string())),
    );
    extra.insert("threads".into(), threads.into());
    extra.insert("deterministic".into(), args.shared.deterministic.into());
    extra.insert("wall_s".into(), started.elapsed().as_secs_f64().into());
    extra.insert("peak_rss_kb".into(), json!(result::peak_rss_kb()));
    result::header("front", m, extra)
}

/// An energy in the fold's integer units, from kcal/mol.
fn energy_units(kcal_mol: f64) -> i32 {
    (kcal_mol * 100.0).round() as i32
}

/// One gap's row of the report: the pair it spans, the weight it was solved
/// at, how much of the front it produced, and where its own time went.
///
/// The per-gap rows are the half of the report a cost question is asked of: a
/// walk's wall clock says how long it took, and these say which gaps it went
/// on.
fn gap_row(gap: &derna_lib::WalkGap) -> Value {
    json!({
        "from": {"mfe": gap.a.mfe_kcal_mol, "adaptation_log": gap.a.adaptation_log},
        "to": {"mfe": gap.b.mfe_kcal_mol, "adaptation_log": gap.b.adaptation_log},
        "lambda": gap.lambda,
        "points": gap.front.len(),
        "slices": gap.slices,
        "cut": format!("{:?}", gap.cut),
        "scalar_s": gap.clock.scalar_s,
        "fold_s": gap.clock.fold_s,
        "credit_scan_s": gap.clock.credit_scan_s,
        "credit_pass_s": gap.clock.credit_pass_s,
    })
}

/// The walk's diagnostics: the hull's vertices, each gap, and every task the
/// queue ran.
fn write_report(path: &Path, walked: &WalkedFront, force: bool) -> Result<()> {
    result::check_output(Some(path), force)?;
    let report = json!({
        "vertices": walked.vertices.iter().map(|v| json!({
            "mfe": v.mfe_kcal_mol,
            "adaptation_log": v.adaptation_log,
        })).collect::<Vec<_>>(),
        "gaps": walked.gaps.iter().map(gap_row).collect::<Vec<_>>(),
        "sweep_scans": walked.sweep_scans,
        "probes_cut": walked.probes_cut,
        "rounds": walked.rounds,
        // Derivable from the front, but the quantity this walk's cost tracks
        // deserves to be read rather than re-derived by everything that wants
        // to compare two runs.
        "mfe_span": mfe_span(walked),
        "points": walked.front.len(),
        "grid_s": walked.grid_s,
        "grid_lanes": walked.grid_lanes,
        "stop_slack": walked.stop_slack,
        "sweep_s": walked.sweep_s,
        "gaps_s": walked.gaps_s,
        "credit_s": walked.credit_s,
        // Each task's clock and the process's memory when it ended: the
        // running peak there says which task raised it, which is the
        // question a memory budget is set from.
        "timeline": walked.timeline.iter().map(|t| json!({
            "kind": format!("{:?}", t.kind),
            "lambda": t.lambda,
            "queued_s": t.queued_s,
            "start_s": t.start_s,
            "end_s": t.end_s,
            "workers": t.workers,
            "peak_workers": t.peak_workers,
            "cpu_s": t.cpu_s,
            "rss_kb": t.rss_kb,
            "peak_rss_kb": t.peak_rss_kb,
        })).collect::<Vec<_>>(),
    });
    std::fs::write(path, format!("{report}\n"))?;
    Ok(())
}

/// The most grid passes a motif walk holds without a memory budget: every
/// pass, unless a development build sets a number, meant beside its pass
/// cache, which reads a dropped pass back from disk.
fn resident_passes() -> Option<usize> {
    derna_lib::dev_env::var(derna_lib::dev_env::Setting::MotifPassResident)
        .ok()?
        .parse()
        .ok()
}

/// The share of the front's energy span left in unsettled boxes at which a
/// motif walk stops: none, running until done or out of budget, unless a
/// development build sets one.
fn stop_open_share() -> Option<f64> {
    derna_lib::dev_env::var(derna_lib::dev_env::Setting::MotifStopOpen)
        .ok()?
        .parse()
        .ok()
}

/// The widest first range of a motif walk's gap folds, in dekacalories, when
/// `DERNA_MOTIF_SLICE_WIDTH` does not set one. At 2.5 kcal/mol it took 7.5%
/// fewer instructions than whole-gap folds on a complete 349-residue motif
/// front, with the same front.
const MOTIF_SLICE_WIDTH: i32 = 250;

/// The widest first range of a motif walk's gap folds, in dekacalories:
/// `DERNA_MOTIF_SLICE_WIDTH`, [`MOTIF_SLICE_WIDTH`] when unset, and `0` to
/// fold each gap whole first.
fn motif_slice_width() -> Option<i32> {
    match std::env::var("DERNA_MOTIF_SLICE_WIDTH") {
        Err(_) => Some(MOTIF_SLICE_WIDTH),
        Ok(v) => v.parse().ok().filter(|&w| w > 0),
    }
}

/// How many gap folds a walk solves before it stops, keeping each stopped
/// gap's rest (a motif walk's ranges, a plain walk's slices), when
/// `--max-gap-folds` does not say: no such limit, unless a development build
/// sets one.
fn gap_folds() -> Option<usize> {
    derna_lib::dev_env::var(derna_lib::dev_env::Setting::MotifGapFolds)
        .ok()?
        .parse()
        .ok()
}

/// Whether a motif walk seeds its sweep with the optima its aware passes
/// certify: on unless `DERNA_MOTIF_AWARE_SEEDS=0`. On a complete 349-residue
/// motif front the seeds took 1.4% fewer instructions, with the same front.
fn motif_aware_seeds() -> bool {
    std::env::var("DERNA_MOTIF_AWARE_SEEDS").map_or(true, |v| v != "0")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &[&str]) -> Vec<String> {
        s.iter().map(|w| w.to_string()).collect()
    }

    /// A plain walk saves its progress wherever `--out` is a file, without
    /// being asked, and not where there is nothing to save to, nothing to
    /// save, or a modular run that did not ask.
    ///
    /// This is the decision only: that the walk then writes through the path
    /// it is handed is `every_checkpoint_of_a_plain_front_resumes_to_the_whole_front`
    /// in `tests/front_budget.rs`.
    #[test]
    fn a_walk_checkpoints_by_default_wherever_out_is_a_file() {
        let at = |extra: &[&str]| {
            let mut w = vec!["--protein", "MK"];
            w.extend_from_slice(extra);
            let a = parse(words(&w)).unwrap();
            checkpoint_path(&a)
        };
        let file =
            std::env::temp_dir().join(format!("derna-ckpt-default-{}.jsonl", std::process::id()));
        let name = file.to_str().unwrap();
        assert_eq!(at(&["--out", name]), Some(file.clone()));
        assert_eq!(at(&["--out", name, "--no-checkpoint"]), None);
        assert_eq!(at(&["--out", name, "--front-from", "plain.jsonl"]), None);
        assert_eq!(at(&["--out", "-"]), None);
        // A device cannot be renamed over, so it is written once at the end.
        assert_eq!(at(&["--out", "/dev/null"]), None);
        assert_eq!(at(&[]), None);
        // A modular run's walk is saved only when asked, and then beside
        // --out, which holds the search's result.
        assert_eq!(at(&["--out", name, "--modular"]), None);
        assert_eq!(
            at(&["--out", name, "--modular", "--checkpoint"]),
            Some(walk_path(&file))
        );
        assert_eq!(at(&["--out", name, "--checkpoint"]), Some(file.clone()));
    }

    /// `--checkpoint` and `--no-checkpoint` together are refused rather than
    /// one silently winning.
    #[test]
    fn checkpoint_and_no_checkpoint_are_refused_together() {
        let e = parse(words(&[
            "--protein",
            "MK",
            "--checkpoint",
            "--no-checkpoint",
        ]))
        .unwrap_err()
        .to_string();
        assert!(e.contains("opposite"), "{e}");
    }

    /// The default check runs on a constrained front, and not where the
    /// caller declined it, where there is no constraint, or after a
    /// modular search, which folded every point it emitted.
    #[test]
    fn the_check_runs_only_where_it_can_learn_something() {
        let check = |w: &[&str]| {
            let a = parse(words(w)).unwrap();
            let m = Molecule::resolve(&a.shared).unwrap();
            a.check_points(&m)
        };
        let constrained = ["--protein", "MPRGLA", "--utr5", "CCGCGG"];
        assert!(check(&constrained));
        assert!(!check(&[&constrained[..], &["--no-check"]].concat()));
        assert!(!check(&[&constrained[..], &["--modular"]].concat()));
        assert!(!check(&["--protein", "MPRGLA"]));
    }

    /// A front is never written with an open box its own resume would refuse,
    /// or one listed twice; boxes the reader accepts pass.
    #[test]
    fn open_boxes_a_resume_would_refuse_are_not_written() {
        let ok: OpenBox = ((-10.5, -0.5), (-8.0, -0.25));
        let lone: OpenBox = ((-8.0, -0.25), (-8.0, -0.25));
        assert_eq!(unwritable_open_boxes(&[ok, lone]), None);
        let reversed: OpenBox = (ok.1, ok.0);
        let why = unwritable_open_boxes(&[ok, reversed]).unwrap();
        assert!(
            why.contains("not its more stable then its more adapted"),
            "{why}"
        );
        let why = unwritable_open_boxes(&[ok, lone, ok]).unwrap();
        assert!(why.contains("listed twice"), "{why}");
        // What the writer accepts, the reader accepts.
        let refuse = |s: String| s;
        let header = json!({"open": [{"mfe_lo": -10.5, "mfe_hi": -8.0,
            "cai_log_lo": -0.5, "cai_log_hi": -0.25}]});
        assert!(resume_boxes(&header, &refuse).is_ok());
    }

    /// A box's `moves` is read back by its ends; a box without the field, as
    /// every file written before it, moved on none.
    #[test]
    fn an_open_box_without_moves_resumes_with_none() {
        let h = json!({"open": [
            {"mfe_lo": -10.5, "mfe_hi": -8.0, "cai_log_lo": -0.1, "cai_log_hi": -0.05},
            {"mfe_lo": -8.0, "mfe_hi": -6.25, "cai_log_lo": -0.05, "cai_log_hi": -0.01,
             "mfe_from": -7.5, "cai_log_floor": -0.04, "moves": 3},
        ]});
        assert_eq!(resume_moves(&h), vec![((-8.0, -6.25), 3)]);
        assert!(resume_moves(&json!({"open": []})).is_empty());
    }

    #[test]
    fn epsilon_is_exact_and_under_the_corner_bound() {
        assert_eq!(EPSILON, 2f64.powi(-23));
        assert_eq!(EPSILON, 1.1920928955078125e-7);
        // The bound falls with length; 2.0e-6 is its value at 2,828 residues,
        // the longest protein this command is meant for, and ε sits well under
        // it. Read from a variable so the comparison is not folded away.
        let corner_bound_at_2828_residues = 2.0e-6;
        assert!(EPSILON < corner_bound_at_2828_residues / 10.0);
        // Exact in binary: the range's ends round-trip.
        assert_eq!(1.0 - (1.0 - EPSILON), EPSILON);
    }

    #[test]
    fn the_front_takes_no_weight() {
        let e = parse(words(&["--protein", "MK", "--lambda", "0.5"]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("takes no --lambda"), "{e}");
    }

    #[test]
    fn a_budget_takes_a_walk_and_a_positive_number() {
        // The budget every platform takes is the clock. `--max-rss-gb` is
        // refused where resident memory cannot be read, so asserting that it
        // parses would assert a Linux fact everywhere; that refusal is
        // `a_memory_limit_is_taken_only_where_it_can_bind`'s subject.
        let budgets: &[&str] = if derna_lib::task_clock::CURRENT_RSS_READABLE {
            &["--max-secs", "--max-rss-gb"]
        } else {
            &["--max-secs"]
        };
        for &flag in budgets {
            // A budget stops a walk, and every front this command walks takes
            // one; a front read back from a file is not walked.
            assert!(parse(words(&["--protein", "MK", flag, "1"])).is_ok());
            let e = parse(words(&[
                "--protein",
                "MK",
                "--front-from",
                "plain.jsonl",
                flag,
                "1",
            ]))
            .unwrap_err()
            .to_string();
            assert!(e.contains("--front-from runs none"), "{flag}: {e}");
            let a = parse(words(&["--protein", "MK", "--modular", flag, "2.5"])).unwrap();
            assert!(a.modular);
            assert!(parse(words(&["--protein", "MK", "--modular", flag, "0"])).is_err());
        }
    }

    #[test]
    fn fold_threads_reach_the_motif_walk_only() {
        let e = parse(words(&["--protein", "MK", "--fold-threads", "4"]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("needs --motif"), "{e}");
        let a = parse(words(&[
            "--protein",
            "MK",
            "--motif",
            "GGGGG",
            "--fold-threads",
            "4",
        ]))
        .unwrap();
        assert_eq!(a.fold_threads, Some(4));
        assert!(parse(words(&[
            "--protein",
            "MK",
            "--motif",
            "GGGGG",
            "--fold-threads",
            "0"
        ]))
        .is_err());
    }

    #[test]
    fn a_flag_that_waits_on_the_walk_says_so() {
        // `--strict` still waits on the walk, and says so.
        let e = parse(words(&["--protein", "MK", "--strict"]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("not supported yet"), "{e}");
        assert!(parse(words(&["--protein", "MK", "--resume", "f.jsonl"])).is_ok());
        let checkpoint = ["--protein", "MK", "--checkpoint", "--out", "f.jsonl"];
        assert!(parse(words(&checkpoint)).is_ok());
        // Without the modular search, a checkpoint is the
        // constrained walk's front: a run that walks none has nothing to
        // checkpoint. With it, the search is checkpointed whether or not the
        // constrained front is walked.
        let mut argv = words(&checkpoint);
        argv.extend(words(&["--front-from", "g.jsonl"]));
        let e = parse(argv).unwrap_err().to_string();
        assert!(e.contains("--checkpoint"), "{e}");
        for other in [
            &["--modular"][..],
            &["--modular", "--front-from", "g.jsonl"],
        ] {
            let mut argv = words(&checkpoint);
            argv.extend(words(other));
            assert!(parse(argv).is_ok(), "{other:?}");
        }
    }

    /// The derivation cap reaches the policy the search reads, and the values
    /// that cannot mean anything are refused rather than taken and ignored.
    #[test]
    fn a_derivation_cap_reaches_the_policy_or_is_refused() {
        let args = parse(words(&[
            "--protein",
            "MK",
            "--modular",
            "--max-derivations",
            "7",
        ]))
        .expect("a modular run takes the cap");
        assert_eq!(args.policy.max_derivations, Some(7));

        let zero = parse(words(&[
            "--protein",
            "MK",
            "--modular",
            "--max-derivations",
            "0",
        ]))
        .unwrap_err()
        .to_string();
        assert!(zero.contains("1 or more"), "{zero}");

        let twice = parse(words(&[
            "--protein",
            "MK",
            "--modular",
            "--max-derivations",
            "7",
            "--max-derivations",
            "9",
        ]))
        .unwrap_err()
        .to_string();
        assert!(twice.contains("given twice"), "{twice}");

        // A plain front walks to the whole front, so the cap would be read
        // from the command line, reported in the header and applied to
        // nothing.
        let plain = parse(words(&["--protein", "MK", "--max-derivations", "7"]))
            .unwrap_err()
            .to_string();
        assert!(plain.contains("--max-derivations"), "{plain}");
        assert!(plain.contains("modular"), "{plain}");
    }

    #[test]
    fn a_memory_limit_is_taken_only_where_it_can_bind() {
        // `--max-rss-gb` is enforced by reading the process's resident memory
        // and comparing. Where that reading cannot be taken the comparison is
        // false every time, so the flag used to be accepted and then never
        // fire: the run continued past the limit it was given, reporting
        // nothing. The flag now tracks whether the reading exists.
        //
        // One of the two branches is dead on any one platform, which is the
        // point — the assertion is that the flag and the reading agree, and
        // it is checked wherever this runs rather than only where the refusal
        // happens to fire.
        let argv = words(&["--protein", "MK", "--max-rss-gb", "4"]);
        if derna_lib::task_clock::CURRENT_RSS_READABLE {
            assert!(
                parse(argv).is_ok(),
                "resident memory is readable here, so the limit can bind"
            );
        } else {
            let e = parse(argv).unwrap_err().to_string();
            assert!(e.contains("--max-rss-gb"), "{e}");
            assert!(e.contains("never bind"), "{e}");
        }
    }

    #[test]
    fn the_front_out_needs_the_modular_front() {
        let e = parse(words(&["--protein", "MK", "--front-out", "f.jsonl"]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("--front-out"), "{e}");
        let a = parse(words(&[
            "--protein",
            "MK",
            "--modular",
            "--front-out",
            "f.jsonl",
            "--front-from",
            "g.jsonl",
        ]))
        .unwrap();
        assert_eq!(a.front_out.as_deref(), Some(Path::new("f.jsonl")));
        assert_eq!(a.front_from.as_deref(), Some(Path::new("g.jsonl")));
        let e = parse(words(&[
            "--protein",
            "MK",
            "--front-from",
            "a",
            "--front-from",
            "b",
        ]))
        .unwrap_err()
        .to_string();
        assert!(e.contains("twice"), "{e}");
    }

    #[test]
    fn a_front_of_another_molecule_is_named_by_its_field() {
        let mut fields = Map::new();
        fields.insert("protein".into(), json!({"name": "a", "sequence": "MK"}));
        fields.insert("forbid".into(), json!(["5-cds", "cds-3"]));
        let same =
            json!({"protein": {"name": "b", "sequence": "MK"}, "forbid": ["5-cds", "cds-3"]});
        assert_eq!(molecule_mismatch(&fields, &same), None);
        let other = json!({"protein": {"name": "a", "sequence": "MK"}, "forbid": ["5-cds"]});
        assert_eq!(
            molecule_mismatch(&fields, &other).as_deref(),
            Some("forbid")
        );
    }

    #[test]
    fn the_shared_flags_are_the_same_parser() {
        let a = parse(words(&[
            "--protein",
            "MK",
            "--utr5",
            "ACGU",
            "--forbid",
            "none",
        ]))
        .unwrap();
        assert_eq!(a.shared.protein.as_deref(), Some("MK"));
        assert_eq!(a.shared.forbid.as_deref(), Some("none"));
        assert_eq!(a.shared.threads(), 1);
    }
}
