//! `derna front --modular`: the front of the designs whose own
//! free fold respects the constraint, within a time and memory budget.
//!
//! The constrained front is walked first, as a plain `front` walks it; the
//! modular front is then computed from it by the library. Under a
//! budget the answer is partial and says so: each point is `certified` (proven
//! on the front) or `provisional` (modular and undominated by
//! anything found, but a region the budget left open could still hold a
//! design dominating it), and the header lists those regions.
//!
//! With `--epsilon` the front is that of the designs whose gap is at most
//! epsilon, searched on its own terms (`derna_lib::modular_front`). Every point and
//! the header carry the epsilon, and the header's `outcome` says which answer
//! the run gave: the front holds a modular point, or only points
//! within epsilon, or it is empty because no design was modular,
//! or because none was within epsilon.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use derna_codon_dp::codon_table::{AMINO_ACIDS, STOP};
use derna_lib::ac::MotifWindow;
use derna_lib::free_fold::sequence_gap_given_constrained;
use derna_lib::modular_front::{
    modular_front_resuming, modular_motif_front_resuming, ModularFront, ModularPoint,
    ModularProgress, OpenBox,
};
pub use derna_lib::modular_front::{ModularEvent, Record};
pub use derna_lib::modular_front::{SearchPolicy, TopSearch};
use derna_lib::pins::Pins;
use derna_lib::{sequence_constrained_fold_must, FrontPoint, Params};
use serde_json::{json, Map, Value};

use crate::molecule::Molecule;
use crate::result;
use crate::Result;

/// Set when the memory budget, rather than the clock, stopped the walk.
static MEMORY_STOPPED: AtomicBool = AtomicBool::new(false);

/// The budget a run was given.
#[derive(Debug, Default, Clone, Copy)]
pub struct Budget {
    pub max_secs: Option<f64>,
    pub max_rss_gb: Option<f64>,
    /// At the memory budget, end the walks running then and go on, rather
    /// than end the search; the search ends only if memory stays over the
    /// budget once they have let go of theirs.
    pub shed_on_memory: bool,
}

/// The modular front of `m`, from its constrained front
/// `constrained`: over every design, or with `motifs` over the designs that
/// spell none of them where the window forbids them and spell the pins, from
/// the motif-free constrained front under the same pins. Without motifs and
/// with `epsilon` above zero it is the front of the designs within it; the
/// search over motif-free designs takes no `epsilon`.
///
/// With `prior` the search continues what an earlier one settled; with
/// `record` it reports what it settles as it settles it.
/// The motif constraint a modular search runs under: every
/// pattern with the window it is forbidden inside, and the pins. One pair per
/// pattern, because which pattern a window scoped is the constraint.
type MotifsAndPins<'a> = (&'a [(String, MotifWindow)], Option<&'a Pins>);

pub fn compute(
    m: &Molecule,
    (constrained, boxes): (&[FrontPoint], &[OpenBox]),
    motifs: Option<MotifsAndPins<'_>>,
    budget: Budget,
    threads: usize,
    (prior, record): (Option<&ModularProgress>, Option<Record<'_>>),
    (top, epsilon, policy): (TopSearch, f64, SearchPolicy),
) -> Result<ModularFront> {
    if let Some(gb) = budget.max_rss_gb {
        watch_memory(gb, budget.shed_on_memory);
    }
    let params = Params::load_default()?;
    let utr5 = m.utr5.as_ref().map_or("", |u| u.sequence.as_str());
    let utr3 = flank3(m);
    let budget_s = (budget.max_secs.unwrap_or(f64::INFINITY), threads);
    let protein = m.designed_protein();
    let (protein, usage) = (protein.as_str(), &m.codons.usage);
    Ok(match motifs {
        None => modular_front_resuming(
            &params,
            protein,
            usage,
            (utr5, &utr3),
            m.table(),
            (constrained, boxes),
            budget_s,
            (prior, record),
            (top, epsilon, policy),
        )?,
        Some(motifs) => {
            debug_assert_eq!(epsilon, 0.0, "the motif-free search takes no epsilon");
            modular_motif_front_resuming(
                &params,
                protein,
                usage,
                (utr5, &utr3),
                m.table(),
                motifs,
                (constrained, boxes),
                budget_s,
                (prior, record),
                (top, policy),
            )?
        }
    })
}

/// What a design the search keeps is, for the run's messages: modular
/// , or within `epsilon` of it.
pub fn admitted(epsilon: f64) -> String {
    if epsilon > 0.0 {
        format!("within {epsilon} kcal/mol of modular")
    } else {
        "modular".to_string()
    }
}

/// Which front the search returns, for the run's messages.
pub fn which_front(epsilon: f64) -> String {
    if epsilon > 0.0 {
        format!("front within {epsilon} kcal/mol of modular")
    } else {
        "modular front".to_string()
    }
}

/// Empty progress of a search within `epsilon`.
pub fn progress_within(epsilon: f64) -> ModularProgress {
    ModularProgress {
        epsilon,
        ..ModularProgress::default()
    }
}

/// Where `--checkpoint` keeps a modular search's progress:
/// beside `out`, as `<out>.search.jsonl`.
pub fn progress_path(out: &Path) -> PathBuf {
    let mut name = out.as_os_str().to_owned();
    name.push(".search.jsonl");
    PathBuf::from(name)
}

/// A search's progress as it settles: one JSON line per event, appended and
/// written through, so a run killed at any moment leaves every event it
/// reported before the last line. A search within an epsilon above zero
/// writes it on every line, so its progress is never read as another's.
pub struct ProgressLog {
    file: Mutex<std::fs::File>,
    epsilon: f64,
}

impl ProgressLog {
    /// Append to `path`, creating it, the progress of a search within
    /// `epsilon`.
    pub fn open(path: &Path, epsilon: f64) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("derna front: cannot open {}: {e}", path.display()))?;
        Ok(Self {
            file: Mutex::new(file),
            epsilon,
        })
    }

    /// Name the run whose progress this is, as the file's first line, so a
    /// rerun can tell its own progress from another molecule's. Written only
    /// to a file that is still empty.
    pub fn identify(&self, run: Value) {
        let mut file = self.file.lock().expect("the progress log");
        if file.metadata().is_ok_and(|m| m.len() == 0) {
            let mut line = json!({ "run": run });
            if self.epsilon > 0.0 {
                line["epsilon"] = self.epsilon.into();
            }
            if let Err(err) = writeln!(file, "{line}") {
                eprintln!("derna front: the progress file's first line was not written: {err}");
            }
        }
    }

    /// Write everything `p` holds, as the events that settled it.
    pub fn carry(&self, p: &ModularProgress) {
        for (sequence, &modular) in &p.checked {
            // A carried verdict has no gap and no pairs: the progress file
            // keeps what resume needs, which is the verdict alone.
            self.record(ModularEvent::Checked {
                sequence,
                modular,
                gap_kcal_mol: None,
                blocking: &[],
            });
        }
        for (mfe_kcal_mol, cai_log, sequence) in &p.found {
            self.record(ModularEvent::Found {
                mfe_kcal_mol: *mfe_kcal_mol,
                cai_log: *cai_log,
                sequence,
            });
        }
        if let Some(down_to) = p.band_down_to {
            self.record(ModularEvent::Band { down_to });
        }
    }

    /// Write one event. A failed write loses that line of progress, never the
    /// run, so it is reported and the search goes on.
    pub fn record(&self, e: ModularEvent<'_>) {
        let mut line = match e {
            ModularEvent::Checked {
                sequence,
                modular,
                gap_kcal_mol,
                blocking,
            } => json!({
                "checked": sequence,
                "modular": modular,
                "gap_kcal_mol": gap_kcal_mol,
                "blocking": blocking.iter().map(|&(i, j)| [i, j]).collect::<Vec<_>>(),
            }),
            ModularEvent::Found {
                mfe_kcal_mol,
                cai_log,
                sequence,
            } => json!({"found": sequence, "mfe": mfe_kcal_mol, "cai_log": cai_log}),
            ModularEvent::Band { down_to } => json!({"band_down_to": down_to}),
        };
        if self.epsilon > 0.0 {
            line["epsilon"] = self.epsilon.into();
        }
        let mut file = self.file.lock().expect("the progress log");
        if let Err(err) = writeln!(file, "{line}") {
            eprintln!("derna front: a progress line was not written: {err}");
        }
    }
}

/// Read a progress file `--checkpoint` wrote. Its last line may have been cut
/// short by the kill that ended the run, and is then left out; any other line
/// that does not read is an error, and so are lines of searches within two
/// epsilons.
pub fn read_progress(path: &Path) -> Result<ModularProgress> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("derna front: cannot read {}: {e}", path.display()))?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut p = ModularProgress::default();
    let mut first = true;
    for (k, line) in lines.iter().enumerate() {
        let v: Value = match crate::result::parse_line(line) {
            Ok(v) => v,
            Err(_) if k + 1 == lines.len() && !text.ends_with('\n') => break,
            Err(e) => {
                return Err(format!("derna front: {} line {}: {e}", path.display(), k + 1).into())
            }
        };
        // Strict on purpose: a line that is not one of the known kinds, or a
        // known kind missing a field it needs, is an error and never a
        // default. Skipping it instead would let a log written under other
        // field names resume as a search that found less — the run would
        // succeed and return a smaller front.
        let bad = || {
            format!(
                "derna front: {} line {}: not a progress line",
                path.display(),
                k + 1
            )
        };
        // The run the file is of, which [`progress_run`] reads.
        if v.get("run").is_some() {
            continue;
        }
        let epsilon = match v.get("epsilon") {
            None => 0.0,
            Some(e) => e.as_f64().ok_or_else(bad)?,
        };
        if std::mem::take(&mut first) {
            p.epsilon = epsilon;
        } else if epsilon != p.epsilon {
            return Err(format!(
                "derna front: {} line {}: a search within {epsilon} kcal/mol, after lines \
                 within {}",
                path.display(),
                k + 1,
                p.epsilon
            )
            .into());
        }
        if let Some(seq) = v["checked"].as_str() {
            let modular = v["modular"].as_bool().ok_or_else(bad)?;
            p.checked.insert(seq.to_string(), modular);
        } else if let Some(seq) = v["found"].as_str() {
            let mfe = v["mfe"].as_f64().ok_or_else(bad)?;
            let cai = v["cai_log"].as_f64().ok_or_else(bad)?;
            p.found.push((mfe, cai, seq.to_string()));
        } else if let Some(d) = v["band_down_to"].as_f64() {
            p.band_down_to = Some(p.band_down_to.map_or(d, |e: f64| e.min(d)));
        } else {
            return Err(bad().into());
        }
    }
    Ok(p)
}

/// The run a progress file names on its first line, or `None` for a file
/// written before progress files named their run.
pub fn progress_run(path: &Path) -> Result<Option<Value>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("derna front: cannot read {}: {e}", path.display()))?;
    Ok(text
        .lines()
        .find(|l| !l.trim().is_empty())
        .and_then(|l| crate::result::parse_line(l).ok())
        .and_then(|v| v.get("run").cloned()))
}

/// The 3′ flank as the molecule spells it: a given stop codon, then the UTR. A
/// designed stop is a slot of the lattice rather than flank sequence, so it is
/// not here — it reaches the search through the protein's trailing `*`.
fn flank3(m: &Molecule) -> String {
    match m.utr3_spec() {
        derna_lib::UtrSpec::Fixed(sequence) => sequence,
        // Only fixed flanks reach a modular run, which the
        // molecule's own resolver is what decides.
        other => unreachable!("a modular run takes fixed flanks: {other:?}"),
    }
}

/// How long memory may stay over its budget after the running walks are shed
/// before the search is stopped: their frontiers are returned to the system
/// over a few seconds, not at once.
const SHED_GRACE: Duration = Duration::from_secs(30);

/// Stop every walk once resident memory reaches `gb`, checked every second.
/// With `shed`, first end only the walks running then, and stop the search
/// only if memory is still over `gb` after [`SHED_GRACE`].
///
/// The constrained walk that comes first is not stopped by it: only the
/// modular front's walks read the stop.
fn watch_memory(gb: f64, shed: bool) {
    let limit = (gb * 1024.0 * 1024.0 * 1024.0) as u64;
    let over = move || rss_bytes().is_some_and(|b| b >= limit);
    std::thread::spawn(move || loop {
        if over() {
            if shed {
                derna_lib::outside::lazy::shed_walks();
                let since = std::time::Instant::now();
                while over() && since.elapsed() < SHED_GRACE {
                    std::thread::sleep(Duration::from_secs(1));
                }
            }
            if !shed || over() {
                MEMORY_STOPPED.store(true, Ordering::Relaxed);
                derna_lib::outside::lazy::stop_walks();
                return;
            }
        }
        std::thread::sleep(Duration::from_secs(1));
    });
}

fn rss_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * 4096)
}

/// Why the run stopped short, or `None` for a complete front: `empty-rounds`
/// when it found nothing in the rounds `--empty-stop-rounds` gave it.
pub fn stopped_by(front: &ModularFront) -> Option<&'static str> {
    match (front.complete, MEMORY_STOPPED.load(Ordering::Relaxed)) {
        (true, _) => None,
        _ if front.gave_up.is_some() => Some("empty-rounds"),
        (false, true) => Some("memory"),
        (false, false) => Some("time"),
    }
}

/// Every point as a result line, on `threads` workers, in the front's order.
///
/// Each line is two exact folds of one molecule and depends on nothing but its
/// own point, so the points divide over the workers with no shared state. The
/// stage is worth parallelizing because it is not a tail: on three runs of a
/// 200-residue molecule it was 47 %, 62 % and 78 % of the whole wall, while the
/// walk above it already had every thread the run was given.
///
/// The order is the front's, not the order the workers finish in, so the result
/// file does not depend on the thread count.
///
/// The parameters are the caller's, so emitting a front in batches costs no
/// more parameter loads than emitting it in one call.
pub fn points(
    m: &Molecule,
    params: &Params,
    points: &[ModularPoint],
    (threads, epsilon): (usize, f64),
) -> Result<Vec<Value>> {
    let workers = threads.max(1).min(points.len().max(1));
    if workers == 1 {
        return points
            .iter()
            .map(|p| point(m, params, p, epsilon))
            .collect();
    }
    let mut out: Vec<Option<Value>> = vec![None; points.len()];
    let slices: Vec<(usize, &mut [Option<Value>])> = {
        // One contiguous run of points each, which is what lets a worker
        // write into its own slots and the file keep the front's order.
        let mut chunks: Vec<&mut [Option<Value>]> = Vec::new();
        let mut rest = out.as_mut_slice();
        let per = points.len().div_ceil(workers);
        while !rest.is_empty() {
            let (head, tail) = rest.split_at_mut(per.min(rest.len()));
            chunks.push(head);
            rest = tail;
        }
        chunks
            .into_iter()
            .enumerate()
            .map(|(i, c)| (i * per, c))
            .collect()
    };
    // A worker's error crosses a thread boundary, and the crate's boxed error
    // is not `Send`, so it travels as its message and is re-boxed here.
    let results: Vec<std::result::Result<(), String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = slices
            .into_iter()
            .map(|(base, slot)| {
                scope.spawn(move || -> std::result::Result<(), String> {
                    for (k, cell) in slot.iter_mut().enumerate() {
                        let line = point(m, params, &points[base + k], epsilon)
                            .map_err(|e| e.to_string())?;
                        *cell = Some(line);
                    }
                    Ok(())
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("a point worker panicked".to_string()))
            })
            .collect()
    });
    for r in results {
        r?;
    }
    Ok(out
        .into_iter()
        .map(|v| v.expect("every point written"))
        .collect())
}

/// One point as a result line: the design, its check — which passes, by
/// construction, and is recorded so the line reads like any checked point —
/// the `epsilon` the search admitted, and whether it is proven on the front.
/// Within an epsilon above zero the check's gap is at most it, and the
/// point is modular only when the gap is zero.
///
/// The check comes first: a seed's was already run to decide it was a seed
/// ([`ModularPoint::check`]) and is reused; any other point's is the free fold
/// against the point's own energy. When the check passes and its free
/// structure puts back no constrained pair, that structure is an allowed one
/// at the free MFE, which the passing check makes the constrained MFE, so it
/// is the point's structure and no constrained fold runs. Otherwise the
/// constrained fold runs for the structure; its energy is not folded a second
/// time for the check — the fold has already returned it — because
/// `scan_with_region` and the `scan_with_constraints` behind
/// `sequence_constrained_fold_must` with no required position are the same
/// scan, so a second call could only repeat the first.
pub fn point(m: &Molecule, params: &Params, p: &ModularPoint, epsilon: f64) -> Result<Value> {
    let check = match &p.check {
        Some(check) => check.clone(),
        None => {
            sequence_gap_given_constrained(params, &p.sequence, m.cds(), m.table(), p.mfe_kcal_mol)?
        }
    };
    let (structure, mfe, gap) = if check.is_modular() && !check.reinstates() {
        let mfe = check.constrained_mfe_kcal_mol;
        (check.free_structure.clone(), mfe, check)
    } else {
        let (structure, mfe) =
            sequence_constrained_fold_must(params, &p.sequence, m.cds(), m.table(), &[])?
                .ok_or("a modular design has no allowed structure")?;
        let gap = if p.check.is_some() || check.constrained_mfe_kcal_mol == mfe {
            check
        } else {
            sequence_gap_given_constrained(params, &p.sequence, m.cds(), m.table(), mfe)?
        };
        (structure, mfe, gap)
    };
    let residues = m.protein.sequence.len().max(1) as f64;
    let mut line = Map::new();
    line.insert("record".into(), "point".into());
    line.insert("sequence".into(), p.sequence.clone().into());
    line.insert("mfe".into(), mfe.into());
    line.insert("cai".into(), (p.cai_log / residues).exp().into());
    line.insert("cai_log".into(), p.cai_log.into());
    line.insert("structure".into(), structure.into());
    if let Some(stop) = m.stop_of(&p.sequence) {
        line.insert("stop".into(), stop.into());
    }
    line.extend(result::verdict_fields(&gap));
    line.insert(
        "fixed_violations".into(),
        fixed_violations(&fixed_bases(m), &gap.reinstated_pairs),
    );
    line.insert("epsilon".into(), epsilon.into());
    let status = if p.certified {
        "certified"
    } else {
        "provisional"
    };
    line.insert("status".into(), status.into());
    Ok(Value::Object(line))
}

/// Whether each base of the molecule is the same in every design: a flank
/// base, a given stop codon's base, or a codon position where every codon
/// its slot may hold has the same base — every codon the table lists for its
/// residue (all of the initiator's, say), or at a designed stop every stop
/// codon, which agree only on their first base. A violation joining two such
/// bases is one no codon choice can remove. It is a fact about the pair, not
/// a proof that the design cannot reach a gap of zero.
pub fn fixed_bases(m: &Molecule) -> Vec<bool> {
    let cds = m.cds();
    let mut fixed = vec![true; cds.end + m.layout().utr3 as usize];
    let stops: Vec<&[u8]> = AMINO_ACIDS[STOP]
        .codons
        .iter()
        .map(|c| &c.nuc[..])
        .collect();
    for (r, residue) in m.designed_protein().chars().enumerate() {
        let codons: Vec<&[u8]> = if residue == '*' {
            stops.clone()
        } else {
            m.codons
                .rows
                .iter()
                .filter(|(aa, _, _)| aa.chars().eq([residue]))
                .map(|(_, codon, _)| codon.as_bytes())
                .collect()
        };
        // An empty family would read as fixed everywhere.
        assert!(!codons.is_empty(), "no codon for residue {residue}");
        for k in 0..3 {
            fixed[cds.start + 3 * r + k] = codons.windows(2).all(|w| w[0][k] == w[1][k]);
        }
    }
    fixed
}

/// The pairs of `violations` (0-based) whose two bases are both in `fixed`
/// ([`fixed_bases`]), 1-based as a point lists its `violations`.
pub fn fixed_violations(fixed: &[bool], violations: &[(usize, usize)]) -> Value {
    violations
        .iter()
        .filter(|&&(i, j)| fixed[i] && fixed[j])
        .map(|&(i, j)| json!([i + 1, j + 1]))
        .collect()
}

/// The header fields that say what the modular front is and how
/// far the budget reached.
pub fn header_fields(
    front: &ModularFront,
    constrained: usize,
    budget: Budget,
    policy: SearchPolicy,
) -> Map<String, Value> {
    let mut extra = Map::new();
    extra.insert("modular_front".into(), true.into());
    extra.insert("epsilon".into(), front.epsilon.into());
    extra.insert("outcome".into(), outcome(front).into());
    extra.insert("constrained_points".into(), constrained.into());
    extra.insert("seeds".into(), front.seeds.into());
    extra.insert("unchecked".into(), front.unchecked.into());
    extra.insert("certify_s".into(), front.certify_s.into());
    extra.insert("points".into(), front.points.len().into());
    extra.insert("complete".into(), front.complete.into());
    extra.insert("stopped_by".into(), json!(stopped_by(front)));
    extra.insert(
        "gave_up".into(),
        json!(front
            .gave_up
            .map(|(round, derivations)| json!({"round": round, "derivations": derivations}))),
    );
    // What the search was allowed to give up on, beside what it did give up
    // on. `gave_up` names the round the stop fired on and the derivations
    // taken by then, which is what both clauses look like from outside: a
    // stop at round 7 reads the same whether the rounds asked for 7 or the
    // derivation count only reached its threshold there. The policy says
    // which, so a front can be read without the run's command line.
    extra.insert(
        "empty_stop".into(),
        json!(policy.give_up_empty.map(|(rounds, derivations)| json!({
            "rounds": rounds,
            "derivations": derivations,
        }))),
    );
    extra.insert("hold_barren".into(), json!(policy.hold_barren));
    extra.insert("max_secs".into(), json!(budget.max_secs));
    extra.insert("max_rss_gb".into(), json!(budget.max_rss_gb));
    extra.insert("shed_on_memory".into(), json!(budget.shed_on_memory));
    extra.insert(
        "walks_shed".into(),
        json!(derna_lib::outside::lazy::walks_shed()),
    );
    let open: Vec<Value> = front
        .open
        .iter()
        .map(|o| {
            json!({
                "cai_log_lo": o.cai_log_lo,
                "cai_log_hi": o.cai_log_hi,
                "mfe_floor": o.mfe_floor_kcal_mol,
                "slope": o.slope_kcal_mol,
                "mfe_floor_constrained": o.mfe_floor_constrained_kcal_mol,
            })
        })
        .collect();
    extra.insert("open".into(), open.into());
    let stretches: Vec<Value> = front
        .stretches
        .iter()
        .map(|s| {
            json!({
                "round": s.round,
                "cai_log_lo": s.cai_log_lo,
                "cai_log_hi": s.cai_log_hi,
                "walked": s.walked,
                "complete": s.complete,
                "points": s.points,
                "modular_found": s.modular_found,
                "designs": s.designs,
                "slope": s.slope,
                "wall_s": s.wall_s,
                "threads": s.threads,
                "cut_s": s.cut_s,
                "energy_s": s.energy_s,
                "verify_s": s.verify_s,
                "folds": s.folds,
                "reached": s.reached,
                "offered": s.offered,
                "build_s": s.build_s,
                "sweep_s": s.sweep_s,
                "walk_s": s.walk_s,
                "derivations": s.derivations,
                "stopped": s.stopped,
            })
        })
        .collect();
    extra.insert("stretches".into(), stretches.into());
    extra.insert("certify_s".into(), front.certify_s.into());
    extra.insert("cut_build_s".into(), front.cut_build_s.into());
    extra
}

/// How many points sit at each gap, in whole hundredths of a kcal/mol.
/// Gaps clump, so this says which steps of epsilon admitted the front.
#[derive(Debug, Default)]
pub struct GapHistogram(BTreeMap<i64, usize>);

impl GapHistogram {
    /// Counts the `gap` of each point line.
    pub fn count(&mut self, lines: &[Value]) {
        for gap in lines.iter().filter_map(|l| l["gap"].as_f64()) {
            *self.0.entry((gap * 100.0).round() as i64).or_default() += 1;
        }
    }
}

/// `0.00×12 0.10×7`, least gap first; `none` for an empty front.
impl std::fmt::Display for GapHistogram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            return write!(f, "none");
        }
        let steps: Vec<String> = self
            .0
            .iter()
            .map(|(hundredths, n)| format!("{:.2}×{n}", *hundredths as f64 / 100.0))
            .collect();
        write!(f, "{}", steps.join(" "))
    }
}

/// Which answer the front is, so a reader need not infer it from the counts
/// and the epsilon: `tight` when it holds points and admitted only
/// modular designs, `within_epsilon` when it holds points within
/// an epsilon above zero, `none_tight` when it is empty and admitted only
/// modular designs, `none_within_epsilon` when it is empty within
/// an epsilon above zero. Within an epsilon a point may still be modular
/// , which its own `gap` says. An empty front is proven empty only when
/// the header's `complete` is true.
pub fn outcome(front: &ModularFront) -> &'static str {
    match (front.points.is_empty(), front.epsilon > 0.0) {
        (false, false) => "tight",
        (false, true) => "within_epsilon",
        (true, false) => "none_tight",
        (true, true) => "none_within_epsilon",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::NamedSequence;

    /// Flanks and the stop are fixed; inside the coding sequence a position
    /// is fixed when every codon of its residue agrees there.
    #[test]
    fn fixed_bases_are_the_ones_every_design_shares() {
        let named = |s: &str| NamedSequence {
            name: None,
            sequence: s.into(),
        };
        let m = Molecule {
            protein: named("ML"),
            utr5: Some(named("GG")),
            utr3: Some(named("C")),
            stop: crate::molecule::Stop::Given("UAA".into()),
            constraint: crate::constraint::resolve(None, None, true, true).unwrap(),
            codons: crate::input::read_codon_usage(None).unwrap(),
        };
        // GG | AUG | (C|U) U (U|C|A|G) | UAA | C: leucine is CUN or UUR.
        let fixed: Vec<bool> = "111110101111".chars().map(|c| c == '1').collect();
        assert_eq!(fixed_bases(&m), fixed);
        // A designed stop is UAA, UAG or UGA: only its first base is fixed.
        let free = Molecule {
            stop: crate::molecule::Stop::Free,
            ..m
        };
        let fixed: Vec<bool> = "111110101001".chars().map(|c| c == '1').collect();
        assert_eq!(fixed_bases(&free), fixed);
    }

    /// Gaps are counted by the hundredth they round to, least first, so float
    /// noise in a gap does not split a step.
    #[test]
    fn a_gap_histogram_counts_each_step_once() {
        let mut gaps = GapHistogram::default();
        assert_eq!(gaps.to_string(), "none");
        gaps.count(&[json!({"gap": 0.1}), json!({"gap": 0.0})]);
        gaps.count(&[json!({"gap": 0.30000000000000004}), json!({"gap": 0.1})]);
        assert_eq!(gaps.to_string(), "0.00×1 0.10×2 0.30×1");
    }
}
