//! `derna-codon-dp` — Pareto-optimal codon design via DERNA's BRD / PORD DPs.
//!
//! Rust port of `derna.codon_dp`, paper-faithful and at `dangles = 0` only:
//! the cell index `(a, b, s, t, x, y)` does not track the flanking codon
//! context dangles would need, which matches DERNA C++ exactly. Composes
//! against [`rna_mfe`] for per-loop Turner 2004 energies.
//!
//! ## Public surface
//!
//! - [`solve_brd`] — BRD solver at one lambda.
//! - [`solve_pord`] — PORD Pareto-front solver (λ-sweep convex hull).
//! - [`solve_pord_with_telemetry`] — PORD solver plus per-λ timing.
//! - [`CodonUsage`] / [`load_codon_usage`] — Kazusa CSV loader.
//! - [`BrdOptions`] — solver options (`dangling_5_prime`, `codon_mask`).
//! - [`BrdSolution`] / [`ParetoPoint`] — result types.
//! - [`codon_table`] / [`codon_usage`] / [`cai`] — substrate modules.
//!
//! Internal modules ([`brd_value`], [`plan`], [`recurrences`], [`solve`])
//! are `pub(crate)` and feed the test-only entry point [`testing`], which
//! exposes the DP tables for per-cell checks. Fixed-λ correctness is
//! anchored against the published DERNA C++ rather than against a
//! reimplementation of it: the fixed-λ objective is carried unrounded, as
//! the C++ carries it, so the two agree on the exact design.

pub mod cai;
pub mod codon_table;
pub mod codon_usage;

pub(crate) mod brd_value;
pub(crate) mod plan;
pub(crate) mod recurrences;
pub(crate) mod solve;

pub use codon_usage::{load_codon_usage, CodonUsage};
pub use solve::{solve_brd, solve_pord, solve_pord_with_telemetry};

use thiserror::Error;

// ---------------------------------------------------------------------------
// Public result + options types
// ---------------------------------------------------------------------------

/// Solver options for [`solve_brd`] and [`solve_pord`].
///
/// `dangling_5_prime` is always `false`, since the DP runs at `dangles = 0`; the field
/// is retained for shape-parity with the Python v1 `BrdOptions`.
/// `codon_mask`, if set, restricts the codon space at each protein position
/// via post-mask indices into `AMINO_ACIDS[aa].codons` — the test path used
/// to build reduced-codon Pareto fixtures (top-2 codons per amino acid by
/// Kazusa fraction, ties broken on the codon string).
#[derive(Debug, Clone)]
pub struct BrdOptions {
    pub dangling_5_prime: bool,
    pub codon_mask: Option<Vec<Vec<usize>>>,
}

/// Result of [`solve_brd`] at one lambda.
///
/// Field shapes mirror Python v1 `BrdSolution`.
#[derive(Debug, Clone)]
pub struct BrdSolution {
    pub rna_seq: String,
    pub structure: String,
    pub mfe_kcal_mol: f64,
    pub cai: f64,
    pub lambda_: f64,
    pub brd_value_kcal_mol: f64,
}

/// One point of the Pareto front returned by [`solve_pord`].
///
/// Field shapes mirror Python v1 `ParetoPoint`.
#[derive(Debug, Clone)]
pub struct ParetoPoint {
    pub rna_seq: String,
    pub structure: String,
    pub mfe_kcal_mol: f64,
    pub cai: f64,
    pub lambda_: f64,
}

/// One measured BRD solve inside a PORD λ-sweep.
#[derive(Debug, Clone)]
pub struct BrdSolveTelemetry {
    pub lambda_: f64,
    pub elapsed_seconds: f64,
    pub mfe_kcal_mol: f64,
    pub cai: f64,
}

/// Why the λ-sweep stopped or refined one bracket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepBracketStopReason {
    SameSolution,
    WidthThreshold,
    NearZeroWidthThreshold,
    Bisect,
}

impl SweepBracketStopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SameSolution => "same_solution",
            Self::WidthThreshold => "width_threshold",
            Self::NearZeroWidthThreshold => "near_zero_width_threshold",
            Self::Bisect => "bisect",
        }
    }
}

/// One bracket decision made by the PORD λ-sweep.
#[derive(Debug, Clone)]
pub struct SweepBracketTelemetry {
    pub lo: f64,
    pub hi: f64,
    pub mid: Option<f64>,
    pub lo_mfe_kcal_mol: f64,
    pub lo_cai: f64,
    pub hi_mfe_kcal_mol: f64,
    pub hi_cai: f64,
    pub mfe_differs: bool,
    pub cai_differs: bool,
    pub width: f64,
    pub width_threshold: f64,
    pub stop_reason: SweepBracketStopReason,
}

/// Timing and memo accounting for a PORD λ-sweep.
#[derive(Debug, Clone, Default)]
pub struct PordTelemetry {
    pub brd_solves: Vec<BrdSolveTelemetry>,
    pub sweep_brackets: Vec<SweepBracketTelemetry>,
    pub memo_hits: usize,
    pub memo_misses: usize,
}

impl PordTelemetry {
    pub fn total_solves(&self) -> usize {
        self.brd_solves.len()
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error type returned by [`solve_brd`] and [`solve_pord`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FoldError {
    #[error("invalid amino acid: {0:?}")]
    InvalidAminoAcid(String),
    #[error(transparent)]
    CodonUsage(#[from] codon_usage::CodonUsageError),
    #[error(transparent)]
    Plan(#[from] plan::PlanError),
    #[error("Turner 2004 parameter load failed: {0}")]
    Params(rna_mfe::ParseError),
    #[error("MFE folder failed: {0}")]
    Mfe(#[from] rna_mfe::FoldError),
    #[error(transparent)]
    Cai(cai::CaiError),
}

// ---------------------------------------------------------------------------
// Test entry points
// ---------------------------------------------------------------------------

/// Test-only fixture-emitting helpers. Hidden from rustdoc; exposed
/// because integration tests in `tests/` can only see `pub` items, and
/// the per-cell parity tests need direct access to the DP tables that
/// the single-loop fill driver populates.
#[doc(hidden)]
pub mod testing {
    use std::collections::BTreeMap;

    use rna_mfe::Params;

    use crate::codon_usage::CodonUsage;
    use crate::plan::{CellIdx, EStorageMode, Plan, PlanInputs, TableLayout};
    use crate::{BrdOptions, BrdSolution, FoldError};

    /// Snapshot of the nine 6-D DP tables, flat in `(a, b, s, t, x, y)`
    /// row-major order. Used by the per-cell parity tests.
    pub struct TablesSnapshot {
        pub n_aa: usize,
        pub max_codons: usize,
        pub e: Vec<f64>,
        pub e_h: Vec<f64>,
        pub e_s: Vec<f64>,
        pub e_b: Vec<f64>,
        pub e_i: Vec<f64>,
        pub e_m: Vec<f64>,
        pub m: Vec<f64>,
        pub n: Vec<f64>,
        pub o: Vec<f64>,
    }

    /// Re-export of the BRD value function for driver-level parity tests.
    pub fn brd_value(lambda_: f64, mfe_dekacal: i32, cai_dekacal: f64) -> f64 {
        crate::brd_value::brd_value(lambda_, mfe_dekacal, cai_dekacal)
    }

    /// Objective-accounting details for one fixed-λ BRD solve.
    pub struct BrdObjectiveDiagnostic {
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

    /// `INF` sentinel — re-exported for parity assertions on unfilled
    /// cells.
    pub const INF: f64 = crate::plan::INF;

    /// Inputs for [`fill_single_loop_tables`]. Bundled so the call site
    /// stays at one ergonomic argument.
    pub struct FillInputs<'a> {
        pub protein: &'a str,
        pub codon_usage: &'a CodonUsage,
        pub lambda_: f64,
        pub options: &'a BrdOptions,
        pub params: &'a Params,
    }

    /// Run the production fixed-λ BRD fill/traceback path and report the
    /// selected top-level table value alongside the reconstructed solution.
    pub fn solve_brd_objective_diagnostic(
        inputs: FillInputs<'_>,
    ) -> Result<BrdObjectiveDiagnostic, FoldError> {
        let diagnostic =
            crate::solve::solve_brd_objective_diagnostic_with_params(crate::solve::SolveInputs {
                protein: inputs.protein,
                codon_usage: inputs.codon_usage,
                lambda: inputs.lambda_,
                options: inputs.options,
                params: inputs.params,
            })?;
        Ok(BrdObjectiveDiagnostic {
            lambda_: diagnostic.lambda_,
            top_o_score_dekacal: diagnostic.top_o_score_dekacal,
            top_o_x: diagnostic.top_o_x,
            top_o_y: diagnostic.top_o_y,
            top_o_tie_count: diagnostic.top_o_tie_count,
            top_o_within_1_count: diagnostic.top_o_within_1_count,
            top_o_within_2_count: diagnostic.top_o_within_2_count,
            top_o_next_score_dekacal: diagnostic.top_o_next_score_dekacal,
            solution: diagnostic.solution,
            posthoc_score_dekacal: diagnostic.posthoc_score_dekacal,
            table_minus_posthoc_dekacal: diagnostic.table_minus_posthoc_dekacal,
        })
    }

    /// Run the single-loop fill driver and return a snapshot of every
    /// table. `M`, `N`, `O`, `E_m` stay at `INF` (filled by the
    /// multi-loop and exterior recurrences, not yet wired).
    pub fn fill_single_loop_tables(inputs: FillInputs<'_>) -> Result<TablesSnapshot, FoldError> {
        let mut plan = full_motif_plan(inputs)?;
        crate::recurrences::fill_single_loop_tables(&mut plan);
        Ok(snapshot(&plan))
    }

    /// Run the fill driver through the multi-loop tables (`M`, `E_m`,
    /// `N`) on top of the single-loop motif tables, and return a
    /// snapshot of every table. `O` stays at `INF` (the exterior
    /// recurrence is not wired here).
    pub fn fill_through_multi_loop_tables(
        inputs: FillInputs<'_>,
    ) -> Result<TablesSnapshot, FoldError> {
        let mut plan = full_motif_plan(inputs)?;
        crate::recurrences::fill_through_multi_loop_tables(&mut plan);
        Ok(snapshot(&plan))
    }

    fn full_motif_plan(inputs: FillInputs<'_>) -> Result<Plan<'_>, FoldError> {
        let plan_inputs = PlanInputs {
            dangling_5_prime: inputs.options.dangling_5_prime,
            codon_mask: inputs.options.codon_mask.clone(),
            lambda_: inputs.lambda_,
            e_storage_mode: EStorageMode::FullMotifTables,
            table_layout: TableLayout::DenseMaxRectangles,
        };
        Ok(Plan::new(
            inputs.protein,
            inputs.codon_usage,
            &plan_inputs,
            inputs.params,
        )?)
    }

    /// Counts of cells gated and admitted by the outer-pair gate in
    /// `fill_compact_e_cell`.
    pub struct OuterGateCounts {
        /// Cells skipped by the gate (non-canonical outer pair).
        pub gated: usize,
        /// Cells admitted by the gate (canonical outer pair).
        pub admitted: usize,
    }

    /// Count of one tied `O` case set at prefix cells.
    pub struct OPrefixTieSetCount {
        pub cases: Vec<u8>,
        pub count: usize,
    }

    /// Winning-case summary for legacy full-fill `O` prefix cells.
    pub struct OPrefixWinningCaseCounts {
        pub total_cells: usize,
        pub finite_cells: usize,
        pub inf_cells: usize,
        pub no_winner_cells: usize,
        pub case_at_min: [usize; 10],
        pub strict_winner_case: [usize; 10],
        pub tie_sets: Vec<OPrefixTieSetCount>,
    }

    impl OPrefixWinningCaseCounts {
        pub fn case_at_min(&self, case: u8) -> usize {
            self.case_at_min[usize::from(case)]
        }

        pub fn strict_winner_case(&self, case: u8) -> usize {
            self.strict_winner_case[usize::from(case)]
        }
    }

    impl OuterGateCounts {
        pub fn total(&self) -> usize {
            self.gated + self.admitted
        }

        pub fn gated_fraction(&self) -> f64 {
            self.gated as f64 / self.total() as f64
        }
    }

    /// Count cells gated vs admitted by the outer-pair canonicity check.
    ///
    /// Builds the plan (without filling any tables) and sweeps every
    /// `(a, b, s, t, x, y)` cell, reporting how many have a canonical
    /// outer closing pair.
    pub fn count_outer_gate_cells(inputs: FillInputs<'_>) -> Result<OuterGateCounts, FoldError> {
        use crate::recurrences::ST_BY_NT_SPAN;

        let plan_inputs = PlanInputs {
            dangling_5_prime: inputs.options.dangling_5_prime,
            codon_mask: inputs.options.codon_mask.clone(),
            lambda_: inputs.lambda_,
            e_storage_mode: EStorageMode::CompactMotifWinners,
            table_layout: TableLayout::PackedUpperTriangle,
        };
        let plan = Plan::new(
            inputs.protein,
            inputs.codon_usage,
            &plan_inputs,
            inputs.params,
        )?;
        let n_aa = plan.n_aa();
        let mut counts = OuterGateCounts {
            gated: 0,
            admitted: 0,
        };
        for span in 0..n_aa {
            for a in 0..(n_aa - span) {
                let b = a + span;
                for &st in &ST_BY_NT_SPAN {
                    count_ab_st_cells(&plan, (a, b), st, &mut counts);
                }
            }
        }
        Ok(counts)
    }

    /// Fill with the legacy all-interval `O` recurrence and summarize which
    /// cases attain the minimum at prefix cells (`a == 0 && s == 0`).
    pub fn count_o_prefix_winning_cases(
        inputs: FillInputs<'_>,
    ) -> Result<OPrefixWinningCaseCounts, FoldError> {
        let plan_inputs = PlanInputs {
            dangling_5_prime: inputs.options.dangling_5_prime,
            codon_mask: inputs.options.codon_mask.clone(),
            lambda_: inputs.lambda_,
            e_storage_mode: EStorageMode::CompactMotifWinners,
            table_layout: TableLayout::PackedUpperTriangle,
        };
        let mut plan = Plan::new(
            inputs.protein,
            inputs.codon_usage,
            &plan_inputs,
            inputs.params,
        )?;
        let mut counts = OPrefixWinningCaseAccumulator::default();
        crate::recurrences::fill_dp_tables_observing_legacy_o(&mut plan, |p, idx, scratch| {
            if idx.0 == 0 && idx.2 == 0 {
                counts.observe(p, idx, scratch);
            }
        });
        Ok(counts.finish())
    }

    fn count_ab_st_cells(
        plan: &Plan<'_>,
        ab: (usize, usize),
        st: (usize, usize),
        counts: &mut OuterGateCounts,
    ) {
        use rna_mfe::loops::pair_from_nucleotides;
        use rna_mfe::params::BasePair;

        use crate::plan::nuc_from_index;

        let (a, b) = ab;
        let (s, t) = st;
        let nx = plan.codon_lists[a].len();
        let ny = plan.codon_lists[b].len();
        for x in 0..nx {
            for y in 0..ny {
                let outer = pair_from_nucleotides(
                    nuc_from_index(plan.codon_lists[a][x].nuc[s]),
                    nuc_from_index(plan.codon_lists[b][y].nuc[t]),
                );
                if matches!(outer, BasePair::NN) {
                    counts.gated += 1;
                } else {
                    counts.admitted += 1;
                }
            }
        }
    }

    #[derive(Default)]
    struct OPrefixWinningCaseAccumulator {
        total_cells: usize,
        finite_cells: usize,
        inf_cells: usize,
        no_winner_cells: usize,
        case_at_min: [usize; 10],
        strict_winner_case: [usize; 10],
        tie_sets: BTreeMap<u16, usize>,
    }

    impl OPrefixWinningCaseAccumulator {
        fn observe(
            &mut self,
            plan: &Plan<'_>,
            idx: CellIdx,
            scratch: &mut Vec<rna_mfe::loops::Nucleotide>,
        ) {
            let target = plan.tables.o[plan.tables.flat_index(idx)];
            self.total_cells += 1;
            if target < crate::plan::INF {
                self.finite_cells += 1;
            } else {
                self.inf_cells += 1;
            }
            let mask = o_prefix_min_case_mask(plan, idx, target, scratch);
            if mask == 0 {
                self.no_winner_cells += 1;
                return;
            }
            for case in 1..=9 {
                if mask & (1 << case) != 0 {
                    self.case_at_min[case] += 1;
                }
            }
            if mask.count_ones() == 1 {
                let case = mask.trailing_zeros() as usize;
                self.strict_winner_case[case] += 1;
            } else {
                *self.tie_sets.entry(mask).or_default() += 1;
            }
        }

        fn finish(self) -> OPrefixWinningCaseCounts {
            OPrefixWinningCaseCounts {
                total_cells: self.total_cells,
                finite_cells: self.finite_cells,
                inf_cells: self.inf_cells,
                no_winner_cells: self.no_winner_cells,
                case_at_min: self.case_at_min,
                strict_winner_case: self.strict_winner_case,
                tie_sets: self
                    .tie_sets
                    .into_iter()
                    .map(|(mask, count)| OPrefixTieSetCount {
                        cases: mask_to_cases(mask),
                        count,
                    })
                    .collect(),
            }
        }
    }

    fn o_prefix_min_case_mask(
        plan: &Plan<'_>,
        idx: CellIdx,
        target: f64,
        scratch: &mut Vec<rna_mfe::loops::Nucleotide>,
    ) -> u16 {
        crate::recurrences::exterior::o_legacy_min_case_mask(plan, idx, target, scratch)
    }

    fn mask_to_cases(mask: u16) -> Vec<u8> {
        (1..=9)
            .filter(|&case| mask & (1 << case) != 0)
            .map(|case| case as u8)
            .collect()
    }

    fn snapshot(plan: &Plan<'_>) -> TablesSnapshot {
        TablesSnapshot {
            n_aa: plan.n_aa(),
            max_codons: plan.max_codons,
            e: plan.tables.dense_values(&plan.tables.e),
            e_h: plan.tables.dense_values(&plan.tables.e_h),
            e_s: plan.tables.dense_values(&plan.tables.e_s),
            e_b: plan.tables.dense_values(&plan.tables.e_b),
            e_i: plan.tables.dense_values(&plan.tables.e_i),
            e_m: plan.tables.dense_values(&plan.tables.e_m),
            m: plan.tables.dense_values(&plan.tables.m),
            n: plan.tables.dense_values(&plan.tables.n),
            o: plan.tables.dense_values(&plan.tables.o),
        }
    }
}
