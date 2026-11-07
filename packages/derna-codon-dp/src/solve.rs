//! Top-level codon-DP solvers.
//!
//! Houses [`solve_brd`] (BRD at one λ) and [`solve_pord`] (the λ-sweep
//! Pareto-front wrapper).
//!
//! Determinism is load-bearing: two runs of [`solve_pord`] with
//! identical args must produce the same `Vec<ParetoPoint>` bit-for-bit.
//! The implementation uses a FIFO [`std::collections::VecDeque`] for
//! the bisection queue and a hand-rolled insertion-ordered memo
//! (`Vec<(λ_bits, BrdSolution)>` keyed by `λ.to_bits()`); the final
//! sort is a stable `(mfe_kcal_mol, cai, rna_seq)` order.

use std::collections::VecDeque;
use std::time::Instant;

use rna_mfe::loops::Nucleotide;
use rna_mfe::Params;

use crate::cai::{compute_cai_tables, score_cai, standard_cai, CaiTables};
use crate::plan::{assemble_rna_into, CellIdx, EStorageMode, Plan, PlanInputs, TableLayout, INF};
use crate::recurrences::{fill_dp_tables, traceback::traceback_o};
use crate::{
    BrdOptions, BrdSolution, BrdSolveTelemetry, CodonUsage, FoldError, ParetoPoint, PordTelemetry,
    SweepBracketStopReason, SweepBracketTelemetry,
};

/// The λ-sweep's outer bracket-width threshold.
pub(crate) const DEFAULT_THRESHOLD: f64 = 0.0025;
/// The λ-sweep's bracket-width threshold near the endpoints.
pub(crate) const DEFAULT_THRESHOLD2: f64 = 0.00075;
/// The λ-sweep's epsilon — opens the sweep at `[ε, 1−ε]`
/// rather than `[0, 1]` to avoid the degenerate endpoints.
pub(crate) const DEFAULT_EPSILON: f64 = 1e-5;

// ---------------------------------------------------------------------------
// solve_brd
// ---------------------------------------------------------------------------

/// Bundled inputs for [`solve_brd_with_params`] — keeps the entry
/// point at 4 arguments while threading the same data through the
/// internal solver.
pub(crate) struct SolveInputs<'a> {
    pub protein: &'a str,
    pub codon_usage: &'a CodonUsage,
    pub lambda: f64,
    pub options: &'a BrdOptions,
    pub params: &'a Params,
}

/// Solve the BRD problem at fixed `λ`.
///
/// Builds the DP plan, fills every sub-table, picks the argmin over
/// `O[0, n_aa − 1, 0, 2, x, y]`, walks the traceback to recover codon
/// assignments + structure, then re-folds the assembled RNA via
/// [`rna_mfe::fold`] to report the self-consistent `(structure, mfe)`
/// pair. CAI is reported via [`standard_cai`].
///
/// The reported BRD value is recomputed from the assembled RNA in a
/// single round at the boundary — the DP cell accumulates per-step
/// `round()` calls which can differ by ±1 dekacal from the single-round
/// quantization at the boundary.
pub fn solve_brd(
    protein: &str,
    codon_usage: &CodonUsage,
    lambda: f64,
    options: &BrdOptions,
) -> Result<BrdSolution, FoldError> {
    let params = Params::load_default().map_err(FoldError::Params)?;
    solve_brd_with_params(SolveInputs {
        protein,
        codon_usage,
        lambda,
        options,
        params: &params,
    })
}

/// Variant of [`solve_brd`] that reuses an already-loaded [`Params`].
/// The λ-sweep wrapper calls this directly to avoid re-loading the
/// parameter file at every λ.
pub(crate) fn solve_brd_with_params(inputs: SolveInputs<'_>) -> Result<BrdSolution, FoldError> {
    let plan_inputs = PlanInputs {
        dangling_5_prime: inputs.options.dangling_5_prime,
        codon_mask: inputs.options.codon_mask.clone(),
        lambda_: inputs.lambda,
        e_storage_mode: EStorageMode::CompactMotifWinners,
        table_layout: TableLayout::PackedUpperTriangle,
    };
    let mut plan = Plan::new(
        inputs.protein,
        inputs.codon_usage,
        &plan_inputs,
        inputs.params,
    )?;
    fill_dp_tables(&mut plan);
    let (x, y, _top) = argmin_top_o(&plan);
    plan.trace.codon_assignments = plan.default_codon_idx.clone();
    let n_aa = plan.n_aa();
    plan.trace.codon_assignments[0] = x;
    plan.trace.codon_assignments[n_aa - 1] = y;
    let top_idx: CellIdx = (0, n_aa - 1, 0, 2, x, y);
    let mut scratch: Vec<Nucleotide> = Vec::with_capacity(plan.default_seq.len());
    traceback_o(&mut plan, top_idx, &mut scratch);
    let cai = compute_cai_tables(inputs.codon_usage);
    solution_from_trace(SolutionInputs {
        plan: &plan,
        params: inputs.params,
        cai: &cai,
        scratch: &mut scratch,
        lambda: inputs.lambda,
    })
}

pub(crate) struct BrdObjectiveDiagnosticInternal {
    pub lambda_: f64,
    pub top_o_score_dekacal: f64,
    pub top_o_x: usize,
    pub top_o_y: usize,
    pub top_o_tie_count: usize,
    pub top_o_within_1_count: usize,
    pub top_o_within_2_count: usize,
    pub top_o_next_score_dekacal: Option<f64>,
    pub solution: BrdSolution,
    pub posthoc_score_dekacal: f64,
    pub table_minus_posthoc_dekacal: f64,
}

pub(crate) fn solve_brd_objective_diagnostic_with_params(
    inputs: SolveInputs<'_>,
) -> Result<BrdObjectiveDiagnosticInternal, FoldError> {
    let plan_inputs = PlanInputs {
        dangling_5_prime: inputs.options.dangling_5_prime,
        codon_mask: inputs.options.codon_mask.clone(),
        lambda_: inputs.lambda,
        e_storage_mode: EStorageMode::CompactMotifWinners,
        table_layout: TableLayout::PackedUpperTriangle,
    };
    let mut plan = Plan::new(
        inputs.protein,
        inputs.codon_usage,
        &plan_inputs,
        inputs.params,
    )?;
    fill_dp_tables(&mut plan);
    let stats = top_o_stats(&plan);
    plan.trace.codon_assignments = plan.default_codon_idx.clone();
    let n_aa = plan.n_aa();
    plan.trace.codon_assignments[0] = stats.x;
    plan.trace.codon_assignments[n_aa - 1] = stats.y;
    let top_idx: CellIdx = (0, n_aa - 1, 0, 2, stats.x, stats.y);
    let mut scratch: Vec<Nucleotide> = Vec::with_capacity(plan.default_seq.len());
    traceback_o(&mut plan, top_idx, &mut scratch);
    let cai = compute_cai_tables(inputs.codon_usage);
    let solution = solution_from_trace(SolutionInputs {
        plan: &plan,
        params: inputs.params,
        cai: &cai,
        scratch: &mut scratch,
        lambda: inputs.lambda,
    })?;
    let posthoc_score_dekacal = solution.brd_value_kcal_mol * 100.0;
    Ok(BrdObjectiveDiagnosticInternal {
        lambda_: inputs.lambda,
        top_o_score_dekacal: stats.score,
        top_o_x: stats.x,
        top_o_y: stats.y,
        top_o_tie_count: stats.tie_count,
        top_o_within_1_count: stats.within_1_count,
        top_o_within_2_count: stats.within_2_count,
        top_o_next_score_dekacal: stats.next_score,
        solution,
        posthoc_score_dekacal,
        table_minus_posthoc_dekacal: stats.score - posthoc_score_dekacal,
    })
}

struct TopOStats {
    x: usize,
    y: usize,
    score: f64,
    tie_count: usize,
    within_1_count: usize,
    within_2_count: usize,
    next_score: Option<f64>,
}

fn top_o_stats(plan: &Plan<'_>) -> TopOStats {
    let (x, y, score) = first_top_o_argmin(plan);
    let gaps = top_o_gap_counts(plan, score);
    TopOStats {
        x,
        y,
        score,
        tie_count: gaps.tie_count,
        within_1_count: gaps.within_1_count,
        within_2_count: gaps.within_2_count,
        next_score: gaps.next_score,
    }
}

fn first_top_o_argmin(plan: &Plan<'_>) -> (usize, usize, f64) {
    let a = 0usize;
    let b = plan.n_aa() - 1;
    let s = 0usize;
    let t = 2usize;
    let mut best = INF;
    let mut bx = 0usize;
    let mut by = 0usize;
    for x in 0..plan.codon_lists[a].len() {
        for y in 0..plan.codon_lists[b].len() {
            let v = plan.tables.o[plan.tables.flat_index((a, b, s, t, x, y))];
            if v < best {
                best = v;
                bx = x;
                by = y;
            }
        }
    }
    (bx, by, best)
}

#[derive(Default)]
struct TopOGapCounts {
    tie_count: usize,
    within_1_count: usize,
    within_2_count: usize,
    next_score: Option<f64>,
}

impl TopOGapCounts {
    fn observe(&mut self, v: f64, best: f64) {
        let gap = v - best;
        self.tie_count += usize::from(gap == 0.0);
        self.within_1_count += usize::from(gap <= 1.0);
        self.within_2_count += usize::from(gap <= 2.0);
        if gap > 0.0 && self.next_score.is_none_or(|score| v < score) {
            self.next_score = Some(v);
        }
    }
}

fn top_o_gap_counts(plan: &Plan<'_>, best: f64) -> TopOGapCounts {
    let a = 0usize;
    let b = plan.n_aa() - 1;
    let s = 0usize;
    let t = 2usize;
    let mut counts = TopOGapCounts::default();
    for x in 0..plan.codon_lists[a].len() {
        for y in 0..plan.codon_lists[b].len() {
            let v = plan.tables.o[plan.tables.flat_index((a, b, s, t, x, y))];
            if v < INF {
                counts.observe(v, best);
            }
        }
    }
    counts
}

fn argmin_top_o(plan: &Plan<'_>) -> (usize, usize, f64) {
    let stats = top_o_stats(plan);
    (stats.x, stats.y, stats.score)
}

/// Convert a `Vec<Nucleotide>` to an owned ASCII `String` (A=0 → 'A',
/// C=1 → 'C', G=2 → 'G', U=3 → 'U').
fn ascii_string(seq: &[Nucleotide]) -> String {
    let bytes: Vec<u8> = seq
        .iter()
        .map(|n| match n {
            Nucleotide::A => b'A',
            Nucleotide::C => b'C',
            Nucleotide::G => b'G',
            Nucleotide::U => b'U',
        })
        .collect();
    String::from_utf8(bytes).expect("nucleotide bytes are valid ASCII")
}

/// Convert a `Vec<Nucleotide>` to nucleotide-index bytes (A=0, C=1, G=2,
/// U=3) for [`standard_cai`] / [`score_cai`] (both expect a `&[u8]` of
/// indices, not ASCII).
fn index_bytes(seq: &[Nucleotide]) -> Vec<u8> {
    seq.iter().map(|n| *n as u8).collect()
}

/// Bundled per-`solve_brd` inputs for [`solution_from_trace`] — folds
/// the five quantities the report assembly needs into one argument.
struct SolutionInputs<'a, 'p> {
    plan: &'a Plan<'p>,
    params: &'a Params,
    cai: &'a CaiTables,
    scratch: &'a mut Vec<Nucleotide>,
    lambda: f64,
}

fn solution_from_trace(inputs: SolutionInputs<'_, '_>) -> Result<BrdSolution, FoldError> {
    let SolutionInputs {
        plan,
        params,
        cai,
        scratch,
        lambda,
    } = inputs;
    let pinned: Vec<(usize, usize)> = plan
        .trace
        .codon_assignments
        .iter()
        .copied()
        .enumerate()
        .collect();
    assemble_rna_into(plan, &pinned, scratch);
    let rna_ascii = ascii_string(scratch);
    let (structure, energy) = rna_mfe::fold(params, rna_ascii.as_bytes())?;
    let rna_idx = index_bytes(scratch);
    let cai_value =
        standard_cai(&rna_idx, &plan.aa_indices, &cai.log_cai).map_err(FoldError::Cai)?;
    let log_cai_sum =
        score_cai(&rna_idx, &plan.aa_indices, &cai.log_cai).map_err(FoldError::Cai)?;
    let mfe_dekacal = energy.dekacal();
    let cai_dekacal = (log_cai_sum * 100.0).round_ties_even();
    let reported_brd_dekacal =
        (lambda * f64::from(mfe_dekacal) - (1.0 - lambda) * cai_dekacal).round_ties_even() as i32;
    Ok(BrdSolution {
        rna_seq: rna_ascii,
        structure,
        mfe_kcal_mol: energy.to_kcal_mol(),
        cai: cai_value,
        lambda_: lambda,
        brd_value_kcal_mol: f64::from(reported_brd_dekacal) / 100.0,
    })
}

// ---------------------------------------------------------------------------
// solve_pord
// ---------------------------------------------------------------------------

/// Callback handed each BRD solve of a sweep and its solve time in seconds.
pub type OnSolve<'a> = &'a dyn Fn(&BrdSolution, f64);

/// The CAI scale a sweep's λ thresholds are read on.
///
/// This solver quantizes CAI as `round(log_CAI · 100)`, and a solver on raw
/// `log_CAI` sees the same trade-off at a different λ, related by
/// `λ_q = s·λ_raw / (1 + (s−1)·λ_raw)` for `s = 100`. The sweep's thresholds
/// are **widths in λ**, so reading them on this solver's λ places a different
/// set of weights than the raw-scale solver places from the same numbers —
/// measured as 62 λ values against 21 at 32 residues, the extra ones
/// subdividing where the raw-scale sweep had already stopped. A sweep asked to
/// place another solver's weights therefore converts to that solver's λ before
/// it measures a width or bisects, and converts back to solve.
pub const REFERENCE_CAI_SCALE: f64 = 100.0;

/// λ on this solver's scale, as the solver with CAI scale `s` would spell it.
fn on_cai_scale(lambda: f64, scale: f64) -> f64 {
    if scale == 1.0 {
        return lambda;
    }
    lambda / (scale - (scale - 1.0) * lambda)
}

/// The inverse of [`on_cai_scale`]: a λ of the solver with CAI scale `s`, on
/// this solver's scale.
fn off_cai_scale(lambda: f64, scale: f64) -> f64 {
    if scale == 1.0 {
        return lambda;
    }
    scale * lambda / (1.0 + (scale - 1.0) * lambda)
}

/// Internal sweep context — keeps bisector signatures short.
struct SweepCtx<'a> {
    protein: &'a str,
    codon_usage: &'a CodonUsage,
    options: &'a BrdOptions,
    params: &'a Params,
    threshold: f64,
    threshold2: f64,
    epsilon: f64,
    /// The CAI scale the thresholds are read on; `1.0` reads them on this
    /// solver's own λ. See [`REFERENCE_CAI_SCALE`].
    cai_scale: f64,
    on_solve: Option<OnSolve<'a>>,
}

/// One queued bracket plus the cached endpoint solutions.
#[derive(Clone)]
struct Bracket {
    lo: f64,
    hi: f64,
    sol_lo: BrdSolution,
    sol_hi: BrdSolution,
}

/// Insertion-ordered memo keyed by `λ.to_bits()`. Linear-scan lookup is
/// O(n) but the sweep is small (typically ≤ 100 evaluations on the toy
/// fixtures), and bit-for-bit determinism is more important than
/// hash-map ergonomics.
#[derive(Default)]
struct Memo {
    entries: Vec<(u64, BrdSolution)>,
}

impl Memo {
    fn get(&self, lambda: f64) -> Option<&BrdSolution> {
        let key = lambda.to_bits();
        self.entries
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, sol)| sol)
    }

    fn insert(&mut self, lambda: f64, sol: BrdSolution) {
        self.entries.push((lambda.to_bits(), sol));
    }

    fn iter(&self) -> impl Iterator<Item = (f64, &BrdSolution)> {
        self.entries
            .iter()
            .map(|(k, sol)| (f64::from_bits(*k), sol))
    }
}

/// Memoized BRD solve at one λ. Insertion order is discovery order.
fn solve_at(
    ctx: &SweepCtx<'_>,
    lambda: f64,
    memo: &mut Memo,
    telemetry: &mut Option<&mut PordTelemetry>,
) -> Result<BrdSolution, FoldError> {
    if let Some(sol) = memo.get(lambda) {
        if let Some(t) = telemetry.as_deref_mut() {
            t.memo_hits += 1;
        }
        return Ok(sol.clone());
    }
    if let Some(t) = telemetry.as_deref_mut() {
        t.memo_misses += 1;
    }
    let t0 = Instant::now();
    let sol = solve_brd_with_params(SolveInputs {
        protein: ctx.protein,
        codon_usage: ctx.codon_usage,
        lambda,
        options: ctx.options,
        params: ctx.params,
    })?;
    let elapsed_seconds = t0.elapsed().as_secs_f64();
    if let Some(t) = telemetry.as_deref_mut() {
        t.brd_solves.push(BrdSolveTelemetry {
            lambda_: lambda,
            elapsed_seconds,
            mfe_kcal_mol: sol.mfe_kcal_mol,
            cai: sol.cai,
        });
    }
    if let Some(on_solve) = ctx.on_solve {
        on_solve(&sol, elapsed_seconds);
    }
    memo.insert(lambda, sol.clone());
    Ok(sol)
}

/// `(MFE, CAI)` differ by more than `epsilon` on either axis.
fn solutions_differ(a: &BrdSolution, b: &BrdSolution, epsilon: f64) -> bool {
    (a.mfe_kcal_mol - b.mfe_kcal_mol).abs() > epsilon || (a.cai - b.cai).abs() > epsilon
}

struct BracketDecision {
    mid: Option<f64>,
    width_threshold: f64,
    mfe_differs: bool,
    cai_differs: bool,
    stop_reason: SweepBracketStopReason,
}

fn classify_bracket(bracket: &Bracket, ctx: &SweepCtx<'_>) -> BracketDecision {
    let mfe_differs =
        (bracket.sol_lo.mfe_kcal_mol - bracket.sol_hi.mfe_kcal_mol).abs() > ctx.epsilon;
    let cai_differs = (bracket.sol_lo.cai - bracket.sol_hi.cai).abs() > ctx.epsilon;
    // Both the width and the midpoint are taken on the threshold's own scale,
    // so the weights land where that solver would put them; only the solve
    // happens on this one.
    let (lo, hi) = (
        on_cai_scale(bracket.lo, ctx.cai_scale),
        on_cai_scale(bracket.hi, ctx.cai_scale),
    );
    let width_threshold = if hi < ctx.threshold {
        ctx.threshold2
    } else {
        ctx.threshold
    };
    let stop_reason = if !solutions_differ(&bracket.sol_lo, &bracket.sol_hi, ctx.epsilon) {
        SweepBracketStopReason::SameSolution
    } else if (hi - lo).abs() > width_threshold {
        SweepBracketStopReason::Bisect
    } else if hi < ctx.threshold {
        SweepBracketStopReason::NearZeroWidthThreshold
    } else {
        SweepBracketStopReason::WidthThreshold
    };
    BracketDecision {
        mid: (stop_reason == SweepBracketStopReason::Bisect)
            .then_some(off_cai_scale(0.5 * (lo + hi), ctx.cai_scale)),
        width_threshold,
        mfe_differs,
        cai_differs,
        stop_reason,
    }
}

fn record_bracket(
    telemetry: &mut Option<&mut PordTelemetry>,
    bracket: &Bracket,
    decision: &BracketDecision,
) {
    if let Some(t) = telemetry.as_deref_mut() {
        t.sweep_brackets.push(SweepBracketTelemetry {
            lo: bracket.lo,
            hi: bracket.hi,
            mid: decision.mid,
            lo_mfe_kcal_mol: bracket.sol_lo.mfe_kcal_mol,
            lo_cai: bracket.sol_lo.cai,
            hi_mfe_kcal_mol: bracket.sol_hi.mfe_kcal_mol,
            hi_cai: bracket.sol_hi.cai,
            mfe_differs: decision.mfe_differs,
            cai_differs: decision.cai_differs,
            width: (bracket.hi - bracket.lo).abs(),
            width_threshold: decision.width_threshold,
            stop_reason: decision.stop_reason,
        });
    }
}

fn run_sweep(
    ctx: &SweepCtx<'_>,
    telemetry: &mut Option<&mut PordTelemetry>,
) -> Result<Memo, FoldError> {
    let mut memo = Memo::default();
    let lo = ctx.epsilon;
    let hi = 1.0 - ctx.epsilon;
    let sol_lo = solve_at(ctx, lo, &mut memo, telemetry)?;
    let sol_hi = solve_at(ctx, hi, &mut memo, telemetry)?;
    let mut queue: VecDeque<Bracket> = VecDeque::new();
    queue.push_back(Bracket {
        lo,
        hi,
        sol_lo,
        sol_hi,
    });
    while let Some(bracket) = queue.pop_front() {
        let decision = classify_bracket(&bracket, ctx);
        record_bracket(telemetry, &bracket, &decision);
        let Some(mid) = decision.mid else {
            continue;
        };
        let sol_mid = solve_at(ctx, mid, &mut memo, telemetry)?;
        queue.push_back(Bracket {
            lo: bracket.lo,
            hi: mid,
            sol_lo: bracket.sol_lo,
            sol_hi: sol_mid.clone(),
        });
        queue.push_back(Bracket {
            lo: mid,
            hi: bracket.hi,
            sol_lo: sol_mid,
            sol_hi: bracket.sol_hi,
        });
    }
    Ok(memo)
}

type PointKey = (String, String, u64, u64);

fn point_key(sol: &BrdSolution) -> PointKey {
    (
        sol.rna_seq.clone(),
        sol.structure.clone(),
        sol.mfe_kcal_mol.to_bits(),
        sol.cai.to_bits(),
    )
}

fn dedup_memo(memo: &Memo) -> Vec<ParetoPoint> {
    let mut seen_keys: Vec<PointKey> = Vec::new();
    let mut points: Vec<ParetoPoint> = Vec::new();
    for (lam, sol) in memo.iter() {
        let key = point_key(sol);
        if seen_keys.contains(&key) {
            continue;
        }
        seen_keys.push(key);
        points.push(ParetoPoint {
            rna_seq: sol.rna_seq.clone(),
            structure: sol.structure.clone(),
            mfe_kcal_mol: sol.mfe_kcal_mol,
            cai: sol.cai,
            lambda_: lam,
        });
    }
    points
}

fn sort_points(points: &mut [ParetoPoint]) {
    points.sort_by(|a, b| {
        a.mfe_kcal_mol
            .partial_cmp(&b.mfe_kcal_mol)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.cai
                    .partial_cmp(&b.cai)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.rna_seq.cmp(&b.rna_seq))
    });
}

fn sweep_ctx<'a>(
    protein: &'a str,
    codon_usage: &'a CodonUsage,
    options: &'a BrdOptions,
    params: &'a Params,
) -> SweepCtx<'a> {
    SweepCtx {
        protein,
        codon_usage,
        options,
        params,
        threshold: DEFAULT_THRESHOLD,
        threshold2: DEFAULT_THRESHOLD2,
        epsilon: DEFAULT_EPSILON,
        cai_scale: 1.0,
        on_solve: None,
    }
}

fn run_sweep_points(
    ctx: &SweepCtx<'_>,
    telemetry: &mut Option<&mut PordTelemetry>,
) -> Result<Vec<ParetoPoint>, FoldError> {
    let memo = run_sweep(ctx, telemetry)?;
    let mut points = dedup_memo(&memo);
    sort_points(&mut points);
    Ok(points)
}

/// PORD solver: returns the convex hull of `(MFE, CAI)` for the given
/// protein via recursive λ-bisection.
///
/// Output is sorted by `(mfe_kcal_mol, cai, rna_seq)` ascending. The
/// sweep is deterministic — identical inputs produce identical
/// `Vec<ParetoPoint>` outputs.
pub fn solve_pord(
    protein: &str,
    codon_usage: &CodonUsage,
    options: &BrdOptions,
) -> Result<Vec<ParetoPoint>, FoldError> {
    let params = Params::load_default().map_err(FoldError::Params)?;
    let ctx = sweep_ctx(protein, codon_usage, options, &params);
    let mut telemetry = None;
    run_sweep_points(&ctx, &mut telemetry)
}

/// PORD solver variant that also records per-λ solve timing and memo
/// accounting for sweep-scale profiling.
pub fn solve_pord_with_telemetry(
    protein: &str,
    codon_usage: &CodonUsage,
    options: &BrdOptions,
) -> Result<(Vec<ParetoPoint>, PordTelemetry), FoldError> {
    solve_pord_reporting(protein, codon_usage, options, None)
}

/// [`solve_pord_with_telemetry`] that also hands each BRD solve to
/// `on_solve` as soon as it finishes, in discovery order, with its solve
/// time in seconds. Memo hits are not reported. A caller that persists
/// these keeps every solve made so far when the sweep is stopped early.
pub fn solve_pord_reporting(
    protein: &str,
    codon_usage: &CodonUsage,
    options: &BrdOptions,
    on_solve: Option<OnSolve<'_>>,
) -> Result<(Vec<ParetoPoint>, PordTelemetry), FoldError> {
    solve_pord_on_cai_scale(protein, codon_usage, options, on_solve, 1.0)
}

/// [`solve_pord_reporting`], placing its λ values as the solver with CAI scale
/// `cai_scale` would place them (see [`REFERENCE_CAI_SCALE`]). `1.0` is
/// [`solve_pord_reporting`].
///
/// Only where the sweep *looks* changes: every solve is this solver's, on this
/// solver's λ, so each returned point is the same point the same λ gives
/// without the scale.
///
/// # Errors
///
/// As [`solve_pord_reporting`].
pub fn solve_pord_on_cai_scale(
    protein: &str,
    codon_usage: &CodonUsage,
    options: &BrdOptions,
    on_solve: Option<OnSolve<'_>>,
    cai_scale: f64,
) -> Result<(Vec<ParetoPoint>, PordTelemetry), FoldError> {
    let params = Params::load_default().map_err(FoldError::Params)?;
    let mut ctx = sweep_ctx(protein, codon_usage, options, &params);
    ctx.on_solve = on_solve;
    ctx.cai_scale = cai_scale;
    let mut telemetry = PordTelemetry::default();
    let points = {
        let mut telemetry_ref = Some(&mut telemetry);
        run_sweep_points(&ctx, &mut telemetry_ref)?
    };
    Ok((points, telemetry))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_driver_traceback_runs_with_compact_e_storage() {
        let codon_usage = CodonUsage::load_default().expect("load default codon usage");
        let params = Params::load_default().expect("load default Turner 2004 params");
        let options = BrdOptions {
            dangling_5_prime: false,
            codon_mask: None,
        };
        let plan_inputs = PlanInputs {
            dangling_5_prime: options.dangling_5_prime,
            codon_mask: options.codon_mask.clone(),
            lambda_: 0.5,
            e_storage_mode: EStorageMode::CompactMotifWinners,
            table_layout: TableLayout::PackedUpperTriangle,
        };
        let mut plan = Plan::new("GLAGPR", &codon_usage, &plan_inputs, &params).unwrap();
        assert_eq!(
            plan.tables.e_storage_mode(),
            EStorageMode::CompactMotifWinners
        );
        assert_eq!(plan.tables.layout(), TableLayout::PackedUpperTriangle);
        assert!(plan.tables.e_h.is_empty());
        assert!(plan.tables.e_s.is_empty());
        assert!(plan.tables.e_b.is_empty());
        assert!(plan.tables.e_i.is_empty());
        assert!(plan.tables.e_m.is_empty());

        fill_dp_tables(&mut plan);
        let (x, y, top) = argmin_top_o(&plan);
        assert!(top < INF);
        plan.trace.codon_assignments = plan.default_codon_idx.clone();
        let n_aa = plan.n_aa();
        plan.trace.codon_assignments[0] = x;
        plan.trace.codon_assignments[n_aa - 1] = y;
        let mut scratch: Vec<Nucleotide> = Vec::with_capacity(plan.default_seq.len());
        traceback_o(&mut plan, (0, n_aa - 1, 0, 2, x, y), &mut scratch);
        assert_eq!(plan.trace.codon_assignments.len(), n_aa);
    }
}
