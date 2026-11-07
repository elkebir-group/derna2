//! `derna-lib` — beam-search codon design.
//!
//! Designs an mRNA coding sequence for a target protein that minimizes the
//! fixed-λ BRD objective `λ·MFE − (1−λ)·CAI`, by running a
//! left-to-right (5′→3′) CKY scan over a **codon DFA** — a compact graph
//! of every RNA sequence encoding the protein. Folding and codon choice
//! happen in one pass: structure is parsed by the CKY grammar and codon
//! adaptation enters as additive weights on the DFA's edges. The scan runs
//! the exact (`b → ∞`) lattice; a finite beam width is an optional, opt-in
//! speed knob that can change the optimum.
//!
//! The crate is organized as the codon DFA ([`dfa`]), the six node-keyed
//! CKY tables and their state ([`tables`], [`state`]), the 5′→3′ transition
//! driver ([`scan`]), and the structure/sequence traceback ([`traceback`]).
//!
//! The validated entry point is [`fold_single_path`]: a pure-MFE (λ=1) scan
//! over a single-path DFA computes the sequence's exact-lattice MFE, which
//! checks the grammar and energy model against a plain fold of one sequence.
//! The designs themselves do not depend on that check.
//! [`design`] composes the full multi-path DFA with the same scan, jointly
//! optimizing each loop interior's energy and CAI over the codon choices.
//!
//! See [`docs/algorithm.md`](../docs/algorithm.md) for the full recurrence
//! treatment. The scan is a separate substrate from the interval DP of
//! `derna-codon-dp`, which stays the exact-optimum oracle it is validated
//! against at tractable lengths.

use std::sync::{Arc, Mutex, OnceLock};

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS, STOP};
use thiserror::Error;

pub mod ac;
pub mod batch_probe;
pub mod bound_probe;
pub mod bounded_motifs;
pub mod bounds;
pub mod box_census;
pub mod cegar;
pub mod cl_probe;
pub mod clamp;
pub mod class_bound;
pub mod complement;
mod constrained;
pub mod coords;
pub mod cost_to_go;
pub mod dev_env;
pub mod dfa;
pub mod end_pins;
pub mod energy_cut;
mod fan_detail;
pub mod fold_cut;
mod fold_pool;
pub mod free_fold;
pub mod front;
pub mod front_probe;
mod front_scan;
mod hash;
mod interior;
pub mod kbest;
pub mod lazy_motifs;
mod lz_index;
pub mod margin;
pub mod modular_front;
pub mod motif_front;
mod must_pair;
pub mod nuc_set;
pub mod outside;
#[doc(hidden)]
pub mod outside_multi;
mod outside_scan;
pub mod pareto;
pub mod pins;
pub mod positional;
pub mod precheck;
pub mod probe;
pub mod product;
pub mod profile;
pub mod realization;
mod refine_gate;
pub mod region;
pub mod repair;
pub mod reuse_probe;
pub mod rival_cert;
pub mod scan;
mod scan_multi;
pub mod search;
pub mod seed_loop;
mod serial_detail;
pub mod state;
pub mod structure;
pub mod tables;
pub mod tail_cut;
pub mod task_clock;
mod task_queue;
pub mod ties;
pub mod traceback;
mod utr_profile;
mod walk;
pub mod witness;

/// Foundational energy types, re-exported so callers need not depend on
/// `rna-mfe` directly. The beam accumulates MFE in these integer-dekacalorie
/// units to keep cross-crate arithmetic exact.
pub use rna_mfe::{Energy, Nucleotide, Params};
use walk::PROBE;

/// The opt-in front-pruning policy for the approximate front-DP. Re-exported so
/// callers of [`design_front_pruned`] can select the FPTAS ε-grid or the hard
/// cap without depending on the [`front`] module directly.
pub use front::Prune;

/// The feasibility box a front fold is solved under: how a ceiling is stated,
/// the window it resolves to, and the verdict when nothing can meet it.
pub use bounds::{Bounds, Ceiling, CreditScope, Slack, Unreachable, Windows};

/// The read-only observer [`design_front_probed`] takes, with the drop rules
/// and energy budgets it reports against. Re-exported so a caller can configure
/// one without depending on the [`front_probe`] module directly.
pub use front_probe::{Budget, CellTable, ComposeSite, DropRule, FrontProbe, ProbeSpec};

/// What a box around one band of the front would remove from the bounded
/// fold, counted by [`census_front_bounded`].
pub use box_census::{BoxCut, BoxCuts, BoxTally};

pub use fan_detail::{FanDetail, FanSite, FanTally};
/// What a fold's feasibility drop discarded, split by table.
pub use front_scan::{DropTally, FoldClock};
pub use serial_detail::{ComposeCount, SerialDetail};

/// The opt-in region-pair constraints for [`design_with_utr`]: CDS↔UTR
/// isolation and the start-proximal accessibility window, each hard or soft.
/// Re-exported so callers need not depend on the [`region`] module directly; a
/// default-constructed [`RegionConstraints`] is inert.
pub use region::{
    parse_class_list, AccessibilityWindow, ClassListError, Enforcement, PairClass, Region,
    RegionConstraints,
};

/// The nucleotides one designed flank position may spell — an arbitrary
/// non-empty subset of `{A, C, G, U}`, written as an IUPAC code in a
/// [`UtrSpec::Mixed`] string. Holding two flanks over one mutually non-pairing
/// subset makes a flank-to-flank helix **unspellable**, which is a constraint on
/// the molecule rather than on the structure a design is scored against.
pub use nuc_set::NucSet;

/// The range of positions a forbidden-motif set is enforced over. Re-exported
/// so callers of [`design_composite`] need not depend on the [`ac`] module
/// directly; [`MotifWindow::all`] is enforcement over the whole molecule, which
/// is what every path without a window does. **Read its documentation for which
/// occurrences a window forbids** — a window's reach ([`WindowReach`]: the
/// occurrences lying wholly inside it, or every occurrence touching it) is a
/// property of the constraint, not of the implementation.
pub use ac::{MotifWindow, WindowReach};

/// Deriving a forbidden-motif set from the **given** regions of a molecule so
/// its **designed** regions cannot spell a duplex with them — the instrument
/// that reaches the case no alphabet restriction can. Re-exported so callers of
/// [`design_composite`] need not depend on the [`complement`] module directly.
/// **Read the module documentation for what a derived set guarantees**: no
/// *ungapped* duplex of length `≥ k`, which is narrower than "no helix of length
/// `≥ k`", and which regions it holds between.
pub use complement::{
    derive_for_run, derive_from_flanks, derive_over, Complement, DerivedMotifs, FlankComplement,
};

/// The opt-in per-position nucleotide preference over **designed** UTR
/// positions — the structural analogue of CAI, and the only term in the
/// objective that mentions a designed flank's spelling. A default-constructed
/// [`UtrProfile`] is inert.
pub use utr_profile::UtrProfile;

/// The free-fold diagnostic for a region-constrained design: what the constraint
/// cost in energy, which constrained pairs the sequence's own unconstrained fold
/// puts back, and how that fold's pairs distribute over the region classes. A
/// report on a finished design, never a gate on producing one.
pub use free_fold::{
    free_fold_gap, sequence_constrained_mfe, sequence_constrained_mfe_zuker,
    sequence_free_fold_zuker, sequence_free_fold_zuker_reusing, sequence_gap,
    sequence_gap_given_constrained, sequence_gap_given_structure,
    sequence_gap_given_structure_reusing, sequence_mfe_within, sequence_mfe_within_reusing,
    sequence_mfes_reusing, sequence_modular_mfe, sequence_modular_mfe_reusing, stacked_runs,
    within_epsilon, CdsWindow, ClassBreakdown, EmittedRescore, FreeFoldError, FreeFoldGap,
    FreeFoldTables, StackedRun, ZukerMfes,
};

/// The flank-versus-flank pre-check: whether two *fixed* UTR flanks already
/// violate the table by themselves, with no CDS present. Live only for a table
/// enforcing something on a pair of flank positions — which the shipped default
/// is, through its `5′UTR↔3′UTR` entry, so it reports on a default run rather
/// than only under an opt-in table.
pub use precheck::{flank_precheck, FlankPrecheck};

/// Best-first enumeration of the design space. Re-exported so a caller
/// searching for a design whose own fold satisfies a predicate needs only the
/// crate root.
pub use kbest::{Enumerator, Problem};

/// Bounded best-first branch-and-bound for the optimal modular design
/// — the enumerator's cost-ordered walk with a bound that lets whole regions of
/// the design space be discarded unopened.
pub use search::{
    bounded_search, BranchOrder, BudgetReadoff, MarginalKey, SearchError, SearchInputs,
    SearchOutcome, SearchReport, SearchStats, SeedRejected, TrajectoryEvent, TrajectoryRecord,
};

/// Counterexample-guided constraint generation over the same space: cut the
/// codon a dirty fold blamed and re-solve, terminating in `O(m)` rounds and
/// forfeiting the optimality the partitioning search proves.
pub use cegar::{
    cegar_search, cegar_search_capped, CegarOutcome, CegarReport, CutRule, StuckDetail,
};

/// λ probing: the nearest λ whose design verifies free-fold modular, and what
/// moving there costs in MFE and CAI. The cheap route when λ may move, and
/// inapplicable when it is pinned.
pub use probe::{probe_grid, probe_lambda, ProbeError, ProbeInputs, ProbeReport, ProbeStep};

/// What a walk with [`WalkOptions::refine_loose`] decided for each bracket of
/// its grid.
pub use refine_gate::BracketRefinement;

/// Codon-usage table loader, re-exported from the exact-DP crate. The same
/// table feeds the DFA's per-edge CAI weights and the BRD scalarization.
pub use derna_codon_dp::CodonUsage;

/// Where a constraint lands: the whole-mRNA coordinate system and the
/// region-qualified spelling (`cds:12`, `5utr:-6`) that survives a flank-length
/// edit. Resolved to absolute positions at parse time, so nothing downstream
/// carries two coordinate systems.
pub use coords::{Coord, CoordError, CoordRegion, Layout};

/// The sequence axis of [`design_composite`]: nucleotides pinned at chosen
/// positions, written as IUPAC codes. **A pin constrains the nucleotide**; the
/// codon set kept at a CDS residue is its projection, and a residue left with no
/// codon is reported before any search runs.
pub use pins::{PinError, Pins};

/// The structure axis of [`design_composite`]: the hard-constraint alphabet over
/// whole-mRNA positions. **Read [`Mark`] for the `<` / `>` directions** — `<`
/// names a partner *downstream* — and for the rule that every character except
/// `|` also forbids the pairs conflicting with it.
pub use structure::{Mark, StructureConstraints, StructureError};

/// The two positional axes as one statement, and what it means for one to be
/// unsatisfiable. [`Infeasible`] names the **tier** that found out, because an
/// empty exact scan is a proof and an empty beam is not.
pub use positional::{
    ConstraintError, Infeasible, PositionalConstraints, PositionalSpec, UnpairablePair,
};

/// The realization verdict for a positionally-constrained design: whether the
/// molecule's own optimum honours what the design required of it, per required
/// pair and overall. **Decided by comparing optima, never by reading one
/// recovered structure** — free optima are routinely co-optimal. A report,
/// never a gate.
pub use realization::{RealizationVerdict, RequiredPairVerdict};

/// How far a fixed sequence's own optimum sits from the best structure of it
/// that forms a pairing it was constrained against — the complement of the
/// modularity verdict, which reports only whether the two coincide. Exact where the
/// whole violating set is folded, and an upper bound where it is truncated.
pub use margin::{cross_region_duplexes, robustness_margin, Duplex, Margin};

/// The best energy a fixed sequence reaches while honouring a positional
/// constraint set — the fold underneath the realization verdict, exposed so a
/// caller can ask what insisting on a named structural element costs the
/// molecule. Under a finite [`Beam`] the answer is an upper bound and its `+∞`
/// is a statement about one search; see [`realization::honouring_mfe`].
pub use realization::honouring_mfe;

/// How many CKY cells a scan materialized — the memory proxy that is a property
/// of the inputs rather than of the host, so two runs of the same design agree
/// on it whatever else the machine is doing.
pub use tables::CellCounts;

/// The crossing-chain bound of a segment certificate: what a chain is, what its
/// margin counts, and what [`crossing_composite`] reports.
pub use outside_scan::crossing::{
    ChainEnd, CrossingChain, CrossingReport, CrossingSpec, ShapeBest,
};

/// A completed design: the chosen coding sequence, its structure, and
/// the `(MFE, CAI)` pair it achieves at the solve's λ.
#[derive(Debug, Clone, PartialEq)]
pub struct Design {
    /// Coding sequence as ASCII `A/C/G/U`.
    pub sequence: String,
    /// Dot-bracket secondary structure.
    pub structure: String,
    /// Minimum free energy in kcal/mol.
    pub mfe_kcal_mol: f64,
    /// Accumulated additive log-CAI, `Σ log(g / g_max)`. The **CDS codons
    /// alone** — a UTR position never contributes here, whether given, designed,
    /// or priced by a [`UtrProfile`].
    pub cai_log: f64,
    /// Conventional CAI, `exp(cai_log / m)` in `[0, 1]`.
    pub cai: f64,
    /// Accumulated [`UtrProfile`] weight over the designed UTR positions,
    /// `Σ log(f / f_max)` scaled by the profile's strength. Reported beside
    /// `cai_log` rather than folded into it, so the CAI a caller reads is always
    /// the CDS-only adaptation. `0.0` under an inert profile and wherever no
    /// position is designed.
    pub utr_profile_log: f64,
}

/// One point on the recovered Pareto front: an `(MFE, CAI)` trade-off with the
/// coding sequence and structure that achieves it. The [`design_front`] analogue
/// of [`Design`] — a front is a `Vec<FrontPoint>`, `mfe`-ascending, each point
/// non-dominated (strictly higher `cai` than the point before it).
#[derive(Debug, Clone, PartialEq)]
pub struct FrontPoint {
    /// Coding sequence as ASCII `A/C/G/U`.
    pub sequence: String,
    /// Dot-bracket secondary structure.
    pub structure: String,
    /// Minimum free energy in kcal/mol.
    pub mfe_kcal_mol: f64,
    /// Accumulated additive log-CAI, `Σ log(g / g_max)`. The **coding codons
    /// alone** — a flank position never contributes here, whether given,
    /// designed, or priced by a [`UtrProfile`].
    pub cai_log: f64,
    /// Conventional CAI, `exp(cai_log / m)` in `[0, 1]`.
    pub cai: f64,
    /// Accumulated [`UtrProfile`] weight over the designed flank positions,
    /// `Σ log(f / f_max)` scaled by the profile's strength. Reported beside
    /// `cai_log` rather than folded into it, so the CAI a caller reads is always
    /// the coding-only adaptation. `0.0` under an inert profile and wherever no
    /// position is designed.
    ///
    /// The front's own dominance is over the *sum*: the fold carries one
    /// adaptation accumulator, so a front recovered under an active profile is
    /// the exact front of energy against `cai_log + utr_profile_log`. Splitting
    /// the two is a reporting step, by the same arithmetic the scalar path uses.
    pub utr_profile_log: f64,
}

/// Recover the **complete Pareto front** of the `(MFE, CAI)` trade-off for
/// `protein` — every non-dominated encoding, including the off-hull points the
/// scalar λ-sweep provably cannot reach — by folding a front-valued algebraic
/// DP over the codon DFA (the same six CKY tables as [`design`], with each cell
/// holding a dominance-reduced front rather than one BRD-optimum).
///
/// This is the **opt-in** full-front mode. It is a separate entry point, never
/// the scalar path: [`design`] and the beam modes are byte-identical to today,
/// and this function allocates the heavier front payload only when a caller
/// explicitly selects it. The fold is **exact and un-reduced** — it recovers
/// the provably-complete front where it fits (validated byte-exact against the
/// brute-force enumeration oracle at small `n`), and carries no approximation.
///
/// Each returned point's `(sequence, structure)` is walked from that point's own
/// provenance, folds to its reported MFE, and translates to `protein`.
pub fn design_front(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
) -> Result<Vec<FrontPoint>, BeamError> {
    design_front_pruned(params, protein, usage, Prune::None)
}

/// Recover an **approximate** Pareto front of the `(MFE, CAI)` trade-off for
/// `protein` under a front-pruning `policy` — the same front-valued algebraic DP
/// as [`design_front`], but capping each cell's front so the fold scales past
/// the exact-front crossover (a complete front needs terabytes at a few hundred
/// residues).
///
/// The recovered front carries the bound its [`Prune`] mode guarantees:
/// [`Prune::EpsGridMfe`] a provable `(1 + delta)^depth` relative-MFE
/// approximation (the FPTAS route), [`Prune::Cap`] a per-cell spread bound (the
/// memory-viable route). [`Prune::None`] recovers the exact complete front and
/// is identical to [`design_front`] — the lossless limit every mode reduces to,
/// so this shares [`design_front`]'s oracle validation where the fold is exact.
///
/// Like [`design_front`] this is opt-in and never the scalar path: [`design`]
/// and the beam modes are byte-identical regardless of `policy`.
pub fn design_front_pruned(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    policy: Prune,
) -> Result<Vec<FrontPoint>, BeamError> {
    design_front_pruned_threaded(params, protein, usage, policy, 1)
}

/// Recover the Pareto front exactly as [`design_front_pruned`], with the fat
/// production phases fanned across `threads` workers — the **opt-in** parallel
/// front fold. `threads = 1` (and `0`) is byte-for-byte the single-threaded
/// path; `threads > 1` reaches an identical front (every `(MFE, CAI)` point *and*
/// every recovered `(sequence, structure)`), since every provenance commit stays
/// serial and the per-chunk partials are folded in deterministic order.
///
/// Multi-threaded wall-clock is a separate metric and does not bear on the
/// single-thread runtime; the parallel fold exists purely to spread a long
/// fold's wall-clock over the host's cores.
pub fn design_front_pruned_threaded(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    policy: Prune,
    threads: usize,
) -> Result<Vec<FrontPoint>, BeamError> {
    Ok(design_front_probed(params, protein, usage, policy, threads, None)?.0)
}

/// Recover the Pareto front exactly as [`design_front_pruned_threaded`], with a
/// read-only observer attached.
///
/// The probe is written to and never read back by the recurrence, so the
/// returned front is byte-identical to the same call without one — every
/// `(MFE, CAI)` point and every recovered `(sequence, structure)`. What comes
/// back with it is what each of the probe's drop rules **would** have removed
/// had the fold been applying them, which is how the cost of a completion bound
/// is measured before one is built.
pub fn design_front_probed(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    policy: Prune,
    threads: usize,
    probe: Option<FrontProbe>,
) -> Result<(Vec<FrontPoint>, Option<FrontProbe>), BeamError> {
    let (front, probe, _) = fold_front(
        params,
        protein,
        usage,
        policy,
        Windows::UNBOUNDED,
        threads,
        probe,
    )?;
    Ok((front, probe))
}

/// Fold under a resolved feasibility window with a read-only observer attached
/// — the diagnostic entry point, for measuring what a ceiling removes and what
/// removing it is worth.
///
/// Takes the resolved [`Windows`] rather than a [`Bounds`] because a caller
/// measuring a ceiling has already resolved it against the achievable minimum
/// and needs the two arms to be folded under windows it chose, not under
/// windows re-derived per call. [`design_front_bounded`] is the entry point for
/// solving; this one is for measuring.
pub fn design_front_probed_bounded(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    policy: Prune,
    windows: Windows,
    threads: usize,
    probe: Option<FrontProbe>,
) -> Result<(Vec<FrontPoint>, Option<FrontProbe>, DropTally), BeamError> {
    fold_front(params, protein, usage, policy, windows, threads, probe)
}

/// The one place an unbounded or diagnostic front fold is run: resolve the
/// protein to a DFA, fold it under `policy` and `windows`, and lift the
/// recovered points into [`FrontPoint`]s. A solve under a ceiling goes through
/// [`solve_bounded`] instead.
fn fold_front(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    policy: Prune,
    windows: Windows,
    threads: usize,
    probe: Option<FrontProbe>,
) -> Result<(Vec<FrontPoint>, Option<FrontProbe>, DropTally), BeamError> {
    let aa = residues(protein)?;
    let graph = dfa::build(&aa, usage);
    let (points, probe, dropped) =
        front_scan::fold_probed(params, &graph, policy, windows, None, threads, probe);
    let front = front_points(points, &graph, UtrProfile::inert());
    Ok((front, probe, dropped))
}

/// The conventional CAI of a design on `graph` whose coding log-adaptation is
/// `cai_log`: the geometric mean over the residues' codons, `exp(cai_log / m)`
/// with `m` = [`dfa::Dfa::sense_codons`]. `0` for a lattice with no residue.
fn cai_of(cai_log: f64, graph: &dfa::Dfa) -> f64 {
    match graph.sense_codons() {
        0 => 0.0,
        m => (cai_log / m as f64).exp(),
    }
}

/// Present the fold's recovered points as [`FrontPoint`]s, splitting the single
/// adaptation accumulator back into the coding-only `cai_log` and the flank's
/// `utr_profile_log`.
///
/// The split is [`finish_with_profile`]'s arithmetic and exists for the same
/// reason: the fold minimizes over one accumulator, and a caller reads two
/// quantities. Under an inert profile the flank term is zero and the offsets
/// vanish, so the coding-only lattice keeps reporting exactly what it reported
/// before.
fn front_points(
    points: Vec<front_scan::RecoveredPoint>,
    graph: &dfa::Dfa,
    profile: UtrProfile,
) -> Vec<FrontPoint> {
    let offsets = graph.utr_profile_offset(profile) + graph.codon_offset();
    points
        .into_iter()
        .map(|p| {
            let utr_profile_log = profile_score(&p.sequence, graph, profile);
            let cai_log = p.cai_log - utr_profile_log + offsets;
            FrontPoint {
                sequence: p.sequence,
                structure: p.structure,
                mfe_kcal_mol: p.mfe.to_kcal_mol(),
                cai_log,
                cai: cai_of(cai_log, graph),
                utr_profile_log,
            }
        })
        .collect()
}

/// The outcome of a design asked for under an energy ceiling.
///
/// Unreachability is a first-class answer rather than an error: asking for a
/// molecule more stable than the sequence space allows is a reasonable question
/// with a definite answer, and the caller needs the achievable minimum in order
/// to ask a reachable one. It is settled before the fold, from the
/// energy-priority corner alone.
#[derive(Debug, Clone, PartialEq)]
pub enum Bounded<T> {
    /// Designs meeting the ceiling exist, and here they are.
    Feasible(T),
    /// Nothing folds within the ceiling. No search was run.
    Unreachable {
        /// The absolute ceiling asked for, kcal/mol.
        ceiling_kcal_mol: f64,
        /// The lowest energy any encoding of this protein reaches, kcal/mol.
        achievable_kcal_mol: f64,
    },
}

/// The exact Pareto front of designs meeting an energy ceiling, with what
/// reaching it cost.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedFront {
    /// The feasible sub-front, `mfe`-ascending and each point non-dominated —
    /// exactly the points of the unconstrained front at or below the ceiling,
    /// recovered without materializing the ones above it. Never empty on any
    /// path that solves anything: the energy-priority corner meets any ceiling
    /// reachable at all, and no admissible credit can discard it.
    pub front: Vec<FrontPoint>,
    /// The resolved absolute ceiling, kcal/mol, whichever form it was stated
    /// in; `None` when the fold was unbounded.
    pub ceiling_kcal_mol: Option<f64>,
    /// The achievable minimum free energy, kcal/mol — the front's first point.
    pub optimum_kcal_mol: f64,
    /// Derivations the in-fold feasibility drop discarded, by table.
    pub dropped: DropTally,
}

/// Recover the exact Pareto front of designs whose fold meets an energy
/// ceiling — every non-dominated `(MFE, CAI)` design with `MFE <= ceiling`,
/// including the ones off the front's convex hull that no weighting of the two
/// objectives can select.
///
/// The ceiling is carried into the fold rather than applied to its output, so
/// derivations that cannot complete within it are discarded as they are formed
/// (see [`crate::bounds`] for why that is exact). The answer is the same either
/// way; what differs is how much of the front has to be built to reach it.
///
/// [`Bounds::UNBOUNDED`] recovers the complete front and is byte-identical to
/// [`design_front_pruned_threaded`] — the drop is a no-op under it, not a filter
/// that happens to keep everything.
pub fn design_front_bounded(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    policy: Prune,
    bounds: Bounds,
    threads: usize,
) -> Result<Bounded<BoundedFront>, BeamError> {
    let aa = residues(protein)?;
    let graph = dfa::build(&aa, usage);
    solve_bounded(
        params,
        &graph,
        UtrProfile::inert(),
        || achievable_minimum(params, protein, usage),
        policy,
        bounds,
        threads,
        None,
    )
}

/// The single best-adapted design whose fold meets an energy ceiling.
///
/// The default deliverable, and the end of [`design_front_bounded`]'s staircase:
/// the feasible sub-front is `cai`-ascending, so its last point is the
/// maximum-CAI feasible design. Equal-CAI designs are tie-broken by the front's
/// own dominance rule — the lower `mfe` is kept — so the answer is unique.
pub fn design_bounded(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    ceiling: Ceiling,
) -> Result<Bounded<FrontPoint>, BeamError> {
    let bounded = design_front_bounded(
        params,
        protein,
        usage,
        Prune::None,
        Bounds::with_ceiling(ceiling),
        1,
    )?;
    Ok(match bounded {
        Bounded::Unreachable {
            ceiling_kcal_mol,
            achievable_kcal_mol,
        } => Bounded::Unreachable {
            ceiling_kcal_mol,
            achievable_kcal_mol,
        },
        Bounded::Feasible(bounded) => {
            let corner = Energy((bounded.optimum_kcal_mol * 100.0).round() as i32);
            Bounded::Feasible(bounded.front.into_iter().next_back().ok_or(
                BeamError::AchievableMinimumDisagrees {
                    corner,
                    folded: None,
                },
            )?)
        }
    })
}

/// [`design_front_bounded`] over a molecule carrying 5′ and 3′ flanks, each
/// supplied, designed, or per-position mixed, with an optional [`UtrProfile`]
/// pricing the designed positions.
///
/// The ceiling is on the **whole molecule's** folding energy, flanks included,
/// which is well defined because the energy accumulator spans the folded
/// molecule rather than the coding sequence alone. That is the asymmetry against
/// a CAI floor over the same molecule: the adaptation accumulator does not split
/// into flank and coding halves sequence-independently, and the energy one needs
/// no splitting at all.
///
/// A flank position credits no CAI whether it is given or designed, so the
/// recovered front's `cai` axis is the coding sequence's alone and means exactly
/// what it means without flanks. Under an **active** profile the fold's single
/// adaptation accumulator is `cai_log + utr_profile_log`, so the recovered front
/// is the exact front of energy against that sum, reported as its two halves —
/// the same scalarization the scalar path makes and for the same reason.
///
/// Empty flanks with an inert profile collapse the lattice to the coding-only
/// one, and this is then [`design_front_bounded`] field for field.
#[allow(clippy::too_many_arguments)]
pub fn design_front_bounded_with_utr(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    profile: UtrProfile,
    policy: Prune,
    bounds: Bounds,
    threads: usize,
) -> Result<Bounded<BoundedFront>, BeamError> {
    let aa = residues(protein)?;
    let five = utr5.parse()?;
    let three = utr3.parse()?;
    let graph = dfa::build_with_utr_profile(&aa, usage, five.chain(), three.chain(), profile);
    solve_bounded(
        params,
        &graph,
        profile,
        || {
            let none = RegionConstraints::default();
            achievable_minimum_with_utr(params, protein, usage, utr5, utr3, profile, none)
        },
        policy,
        bounds,
        threads,
        None,
    )
}

/// The bounded solve both entry points share: resolve the ceiling, fold `graph`
/// under it, and hold the emitted front to the contract.
///
/// `corner` is how the achievable minimum is found when nothing else supplies
/// it. Under [`Bounds::with_exact_credit`] with a ceiling, the exact `λ = 1`
/// fold the credit is built from supplies it instead — its optimum is the same
/// quantity over the same lattice object — so the minimum is not folded twice.
/// That fold is the credit's whole reason to exist, so it is run whether or not
/// the ceiling turns out to be reachable, and [`verify_corner`] still checks the
/// minimum it gives against the front the fold recovers.
#[allow(clippy::too_many_arguments)]
fn solve_bounded(
    params: &Params,
    graph: &dfa::Dfa,
    profile: UtrProfile,
    corner: impl FnOnce() -> Result<Energy, BeamError>,
    policy: Prune,
    bounds: Bounds,
    threads: usize,
    band: Option<&BoxSpec>,
) -> Result<Bounded<BoundedFront>, BeamError> {
    let exact = bounds.exact_credit() && bounds.ceiling().is_some();
    let (mut side, chains) = EnergySide::for_box(params, graph, corner, exact, band, threads)?;
    solve_on(
        params, graph, profile, &mut side, chains, policy, bounds, threads, band,
    )
    .map(|(solved, _)| solved)
}

/// Where a bounded or box solve's time went: the credit's two passes, each
/// cut's two, and the front fold, phase by phase.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SolveClock {
    /// The exact `λ = 1` scalar fold the credit is built over.
    pub credit_scan_s: f64,
    /// The credit's backward pass.
    pub credit_pass_s: f64,
    /// Each cut's weight, scalar fold and backward pass, in the order given.
    pub cuts: Vec<CutClock>,
    /// The wall of the credit's and the cuts' passes together. A box whose
    /// weights are known up front runs them side by side, so this is less than
    /// the sum of the per-pass times above; otherwise it is that sum.
    pub scalar_s: f64,
    /// The front fold's wall, traceback included.
    pub fold_s: f64,
    /// The front fold, phase by phase.
    pub fold: FoldClock,
    /// A sliced gap fold's solves in the order they ran; empty for a solve
    /// that is not sliced.
    pub slices: Vec<SliceClock>,
    /// The first slice a gap fold cut by a chord solved under an exact pass
    /// at its own tie weight instead (1-based), and the seconds the fold
    /// itself spent changing cut (the pass's wall when it ran the pass, about
    /// none when the pass ran as a task beside it), when it changed cut part
    /// way ([`WalkOptions::own_when_alone`], [`WalkOptions::own_when_costly`]).
    pub own_from: Option<(usize, f64)>,
    /// The slice before which the fold asked for that pass. A pass asked
    /// for and pooled only after the fold's last slice leaves this set and
    /// [`Self::own_from`] unset.
    pub own_asked: Option<usize>,
    /// Whether the fold asked for that pass while the slice before
    /// [`Self::own_asked`] was still solving ([`WalkOptions::own_watch_s`])
    /// rather than at its end.
    pub own_asked_mid_slice: bool,
    /// Slices a gap fold halved because their fold stopped: it stored more
    /// than [`WalkOptions::slice_points`], or the walk's memory budget ran
    /// out inside it and was back under it once the fold was freed.
    pub halved: usize,
}

/// One solve of a sliced gap fold: the energy ceiling it ran under, how many
/// slices it covered (more than one when slices ran at once), its wall and
/// CPU time, the workers it ran on and the points it returned.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SliceClock {
    /// The highest slice's energy ceiling, in kcal/mol.
    pub top_kcal_mol: f64,
    /// Slices solved together.
    pub count: usize,
    /// Wall seconds.
    pub wall_s: f64,
    /// CPU seconds of every thread it ran on.
    pub cpu_s: f64,
    /// Workers it ran on.
    pub workers: usize,
    /// Points it returned, before the sweep across slices.
    pub points: usize,
}

impl SliceClock {
    /// The record as a report row:
    /// `[top_kcal_mol, count, wall_s, cpu_s, workers, points]`.
    pub fn row(&self) -> [f64; 6] {
        [
            self.top_kcal_mol,
            self.count as f64,
            self.wall_s,
            self.cpu_s,
            self.workers as f64,
            self.points as f64,
        ]
    }
}

/// One box cut's cost: its weight, scalar fold and backward pass.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CutClock {
    /// The cut's weight.
    pub lambda: f64,
    /// Its exact scalar fold.
    pub scan_s: f64,
    /// Its λ-general backward pass.
    pub pass_s: f64,
}

/// [`design_front_bounded`] or, with `band`, [`design_front_box`], on the
/// coding-only lattice, with the solve's phase clock beside the answer. A box
/// solve is exact only with the exact credit, as `design_front_box` passes it.
#[doc(hidden)]
pub fn clock_front(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    bounds: Bounds,
    band: Option<&BoxSpec>,
    threads: usize,
) -> Result<(Bounded<BoundedFront>, SolveClock), BeamError> {
    let aa = residues(protein)?;
    let graph = dfa::build(&aa, usage);
    let exact = bounds.exact_credit() && bounds.ceiling().is_some();
    let corner = || achievable_minimum(params, protein, usage);
    let (mut side, chains) = EnergySide::for_box(params, &graph, corner, exact, band, threads)?;
    solve_on(
        params,
        &graph,
        UtrProfile::inert(),
        &mut side,
        chains,
        Prune::None,
        bounds,
        threads,
        band,
    )
}

/// The energy side of a bounded solve over one lattice: the achievable
/// minimum and, under the exact credit, the `λ = 1` fold its cost to go is
/// taken over. Neither depends on the ceiling — the credit's window is passed
/// per cell by the fold — so one serves every ceiling solved over the lattice.
struct EnergySide {
    optimum: Energy,
    /// The region constraint every fold over this side applies: the `λ = 1` fold,
    /// so the optimum and the credit are the constrained ones, and every front
    /// fold and cut solved against it. Inert for an unconstrained lattice.
    region: region::PairConstraint,
    /// The exact `λ = 1` fold, until the credit is built from it.
    fold: Option<tables::Tables>,
    credit: Option<bounds::ExactCredit>,
    /// Seconds for the `λ = 1` fold and for its backward pass.
    scan_s: f64,
    pass_s: f64,
    /// Seconds for the credit's and the cuts' passes together, when
    /// [`Self::with_cuts`] ran them side by side.
    scalar_s: Option<f64>,
}

/// One box cut's scalar chain, run before the ceiling it is taken against is
/// known: its weight, its λ-general backward pass and what the two cost.
struct CutChain {
    outside: Arc<cost_to_go::Outside>,
    clock: CutClock,
}

impl EnergySide {
    /// [`Self::with_cuts`] for a box solved under the exact credit, whose cut
    /// weights are then known before any pass runs; [`Self::new`] and no
    /// prepared chains otherwise.
    fn for_box(
        params: &Params,
        graph: &dfa::Dfa,
        corner: impl FnOnce() -> Result<Energy, BeamError>,
        exact: bool,
        band: Option<&BoxSpec>,
        threads: usize,
    ) -> Result<(Self, Vec<CutChain>), BeamError> {
        match band {
            Some(band) if exact => Self::with_cuts(params, graph, corner, &band.lambdas, threads),
            _ => Ok((
                Self::new(
                    params,
                    graph,
                    corner,
                    exact,
                    threads,
                    region::PairConstraint::inert(),
                )?,
                Vec::new(),
            )),
        }
    }

    /// The exact energy side with its credit built, and the scalar chain of a
    /// box cut at each of `lambdas`, all run side by side on `threads` workers
    /// in total.
    ///
    /// No chain reads another's output — a cut's fold and pass depend on its
    /// weight and the lattice, and the ceiling and floor are applied to them
    /// afterwards — and one scalar chain gains little from more than a few
    /// workers, so sharing the workers between the chains finishes all of them
    /// sooner than running each on all of them in turn. Every chain computes
    /// what it would alone: the credit's values do not depend on the worker
    /// count (see [`credit_config`]), and neither do a cut's, whose min-BRD
    /// merges see identically computed candidates in whatever order.
    fn with_cuts(
        params: &Params,
        graph: &dfa::Dfa,
        corner: impl FnOnce() -> Result<Energy, BeamError>,
        lambdas: &[f64],
        threads: usize,
    ) -> Result<(Self, Vec<CutChain>), BeamError> {
        let t = std::time::Instant::now();
        let mut chains = share_threads(threads, 1 + lambdas.len(), |job, workers| {
            let lambda = job.checked_sub(1).map_or(1.0, |cut| lambdas[cut]);
            scalar_chain(params, graph, lambda, workers, job == 0)
        })
        .into_iter();
        let scalar_s = t.elapsed().as_secs_f64();
        let credit = chains.next().expect("the credit's chain runs first");
        let cuts = chains
            .map(|chain| CutChain {
                outside: Arc::new(chain.outside.expect("a cut's pass always runs")),
                clock: CutClock {
                    lambda: chain.lambda,
                    scan_s: chain.scan_s,
                    pass_s: chain.pass_s,
                },
            })
            .collect();
        let (optimum, credit_outside) = match credit.optimum {
            Some(optimum) => (optimum, credit.outside),
            None => (corner()?, None),
        };
        let side = Self {
            optimum,
            region: region::PairConstraint::inert(),
            fold: None,
            credit: credit_outside.map(|outside| bounds::ExactCredit::new(outside, optimum)),
            scan_s: credit.scan_s,
            pass_s: credit.pass_s,
            scalar_s: Some(scalar_s),
        };
        Ok((side, cuts))
    }

    /// Fold the lattice at `λ = 1` when `exact`, and take the achievable
    /// minimum from that fold; otherwise, or when the fold reaches no
    /// exterior, from `corner`. The fold applies `region`, so the minimum and
    /// the credit built from it are the constrained ones, and `corner` must be
    /// taken under the same constraint.
    fn new(
        params: &Params,
        graph: &dfa::Dfa,
        corner: impl FnOnce() -> Result<Energy, BeamError>,
        exact: bool,
        threads: usize,
        region: region::PairConstraint,
    ) -> Result<Self, BeamError> {
        let t = std::time::Instant::now();
        // The fold feeds the credit's outside pass and reports the optimum; no
        // traceback reads it, so it keeps values alone (see `walk_scan`).
        let fold = exact.then(|| {
            scan::scan_values_with_constraints(
                params,
                graph,
                1.0,
                credit_config(threads),
                region.clone(),
                must_pair::MustPairConstraint::inert(),
            )
        });
        let scan_s = t.elapsed().as_secs_f64();
        let folded = fold
            .as_ref()
            .and_then(|tables| tables.c[graph.ix(graph.end())].map(|st| st.mfe));
        let (optimum, fold) = match folded {
            Some(optimum) => (optimum, fold),
            None => (corner()?, None),
        };
        Ok(Self {
            optimum,
            region,
            fold,
            credit: None,
            scan_s,
            pass_s: 0.0,
            scalar_s: None,
        })
    }

    /// Build the exact credit from the `λ = 1` fold, once; a no-op when no
    /// fold was taken or the credit already stands.
    fn build_credit(&mut self, params: &Params, graph: &dfa::Dfa, threads: usize) {
        if let Some(tables) = self.fold.take() {
            let t = std::time::Instant::now();
            let outside =
                cost_to_go::Outside::compute(params, graph, &tables, credit_config(threads));
            self.pass_s = t.elapsed().as_secs_f64();
            self.credit = Some(bounds::ExactCredit::new(outside, self.optimum));
        }
    }
}

/// [`solve_bounded`] against an energy side already built, so several
/// ceilings over one lattice share its `λ = 1` fold and backward pass.
///
/// `chains` are `band`'s cuts' scalar chains when [`EnergySide::with_cuts`]
/// already ran them, one per weight in `band.lambdas`' order; empty, a box's
/// cuts are folded and passed here, one after another.
#[allow(clippy::too_many_arguments)]
fn solve_on(
    params: &Params,
    graph: &dfa::Dfa,
    profile: UtrProfile,
    side: &mut EnergySide,
    chains: Vec<CutChain>,
    policy: Prune,
    bounds: Bounds,
    threads: usize,
    band: Option<&BoxSpec>,
) -> Result<(Bounded<BoundedFront>, SolveClock), BeamError> {
    let bounded = bounds
        .resolve(side.optimum)
        .is_ok_and(|windows| windows.is_bounded());
    if bounds.exact_credit() && bounded {
        side.build_credit(params, graph, threads);
    }
    solve_with(
        params, graph, profile, side, chains, policy, bounds, threads, band,
    )
}

/// [`solve_on`] against an energy side whose credit, where the solve wants
/// one, is already built; it reads the side and never changes it, so several
/// solves may share one.
#[allow(clippy::too_many_arguments)]
fn solve_with(
    params: &Params,
    graph: &dfa::Dfa,
    profile: UtrProfile,
    side: &EnergySide,
    chains: Vec<CutChain>,
    policy: Prune,
    bounds: Bounds,
    threads: usize,
    band: Option<&BoxSpec>,
) -> Result<(Bounded<BoundedFront>, SolveClock), BeamError> {
    solve_stoppable(
        params, graph, profile, side, chains, policy, bounds, threads, band, None,
    )
}

/// What a box fold reads at each position with the points it has stored, to
/// end there ([`BoxCuts::with_stop`]).
type FoldStop = Arc<dyn Fn(usize) -> bool + Send + Sync>;

/// [`solve_with`] under `stop`: a box fold that `stop` ends returns no point
/// and sets [`FoldClock::stopped`].
#[allow(clippy::too_many_arguments)]
fn solve_stoppable(
    params: &Params,
    graph: &dfa::Dfa,
    profile: UtrProfile,
    side: &EnergySide,
    chains: Vec<CutChain>,
    policy: Prune,
    bounds: Bounds,
    threads: usize,
    band: Option<&BoxSpec>,
    stop: Option<FoldStop>,
) -> Result<(Bounded<BoundedFront>, SolveClock), BeamError> {
    let optimum = side.optimum;
    let mut clock = SolveClock::default();
    let windows = match bounds.resolve(optimum) {
        Ok(windows) => windows,
        Err(unreachable) => {
            let solved = Bounded::Unreachable {
                ceiling_kcal_mol: unreachable.ceiling.to_kcal_mol(),
                achievable_kcal_mol: unreachable.achievable.to_kcal_mol(),
            };
            return Ok((solved, clock));
        }
    };
    let ceiling = bounds.ceiling().map(|c| c.resolve(optimum));
    let wants_credit = bounds.exact_credit() && windows.is_bounded();
    (clock.credit_scan_s, clock.credit_pass_s) = (side.scan_s, side.pass_s);
    let credit = side.credit.as_ref().filter(|_| wants_credit);

    let floor = band.map(|band| band.cai_floor_log);
    let cuts = match (band, ceiling) {
        (Some(band), Some(ceiling)) if !chains.is_empty() => Some(prepared_cuts(
            graph,
            profile,
            ceiling,
            band,
            chains,
            &mut clock.cuts,
        )),
        (Some(band), Some(ceiling)) => Some(box_cuts(
            params,
            graph,
            profile,
            ceiling,
            band,
            threads,
            false,
            &side.region,
            &mut clock.cuts,
        )),
        _ => None,
    };
    // The floor's own cut needs no pass: every weight of the plain lattice is
    // at most zero, so no completion adds adaptation.
    let cuts = match (band, ceiling) {
        (Some(band), Some(_)) if bounds.floor_cut() => {
            debug_assert!(graph.max_edge_cai() <= 0.0, "a positive edge weight");
            let floor = BoxCut::floor(cut_floor(graph, profile, band));
            Some(cuts.unwrap_or_else(|| BoxCuts::new(Vec::new())).with(floor))
        }
        _ => cuts,
    };
    let cuts = match stop {
        Some(stop) => cuts.map(|cuts| cuts.with_stop(stop)),
        None => cuts,
    };
    let passes: f64 = clock.cuts.iter().map(|c| c.scan_s + c.pass_s).sum();
    clock.scalar_s = side
        .scalar_s
        .unwrap_or(clock.credit_scan_s + clock.credit_pass_s + passes);
    let t = std::time::Instant::now();
    let (points, dropped, fold_clock) = match &cuts {
        Some(cuts) => {
            front_scan::fold_boxed(params, graph, windows, credit, cuts, threads, &side.region)
        }
        None => front_scan::fold_clocked(
            params,
            graph,
            policy,
            windows,
            credit,
            threads,
            &side.region,
        ),
    };
    (clock.fold_s, clock.fold) = (t.elapsed().as_secs_f64(), fold_clock);
    let mut front = front_points(points, graph, profile);

    // The in-fold drop keeps a cell's points within the window of that cell's
    // own best, which at the final exterior cell is the achievable minimum — so
    // the fold already stops at the ceiling there. Applying it once more to the
    // emitted front costs one pass and makes the output's contract a property of
    // this function rather than of which merge last touched that cell.
    if let Some(ceiling) = ceiling {
        front.retain(|p| energy_of(p).0 <= ceiling.0);
    }
    // A box's floor is applied to the output for the same reason, on the
    // adaptation accumulator the fold minimized over. The corner is below the
    // floor whenever the floor binds, so there is nothing to verify it against.
    //
    // With the floor's own cut among the weights the fold has already applied
    // it at the root, where the cost to go is zero, so nothing below the floor
    // may reach this filter. A floor converted to the fold's scale without the
    // lattice's offsets is looser than the true one — still sound, so no answer
    // moves — and this is where it shows.
    if let Some(band) = band.filter(|band| band.lambdas.contains(&0.0)) {
        debug_assert!(
            front
                .iter()
                .all(|p| p.cai_log + p.utr_profile_log >= band.cai_floor_log - FLOOR_EPSILON),
            "a point below the floor survived the floor's own cut at the root",
        );
    }
    match floor {
        Some(floor) => front.retain(|p| p.cai_log + p.utr_profile_log >= floor - FLOOR_EPSILON),
        None if policy.is_exact() => verify_corner(&front, optimum)?,
        None => {}
    }
    let solved = Bounded::Feasible(BoundedFront {
        front,
        ceiling_kcal_mol: ceiling.map(Energy::to_kcal_mol),
        optimum_kcal_mol: optimum.to_kcal_mol(),
        dropped,
    });
    Ok((solved, clock))
}

/// The box a [`census_front_bounded`] run tests every cell against, beside the
/// energy ceiling it folds under.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxSpec {
    /// The band's adaptation floor: the reported log-CAI of its lower-edge
    /// point, on the scale [`FrontPoint::cai_log`] carries.
    pub cai_floor_log: f64,
    /// The weights at which a cut through the box's corner is taken; `0` is
    /// the floor alone and `1` the energy ceiling, a control.
    pub lambdas: Vec<f64>,
}

/// What a [`census_front_bounded`] run found and what it cost.
pub struct BoxCensusReport {
    /// The feasible sub-front under the energy ceiling — the same front
    /// [`design_front_bounded`] recovers, since the census removes nothing.
    pub front: Vec<FrontPoint>,
    /// The resolved absolute ceiling.
    pub ceiling: Energy,
    /// The floor as the cuts compare it, on the fold's accumulated scale.
    pub cai_floor_accumulated: f64,
    /// The lattice's codon normalization offset, which separates the two
    /// scales.
    pub codon_offset: f64,
    /// Per table and per compose site, what each cut would have kept.
    pub tally: BoxTally,
    /// Seconds for the bounded solve with the exact credit: the `λ = 1` fold,
    /// its backward pass and the front fold, census counting included.
    pub wall_s: f64,
    /// Seconds for the cuts' own scalar folds and backward passes, which a
    /// fold applying the cuts would pay on top of `wall_s`.
    pub cuts_s: f64,
}

/// How far below the stated floor the cuts compare, in log-CAI: a floor copied
/// from a printed report is rounded, and the census must keep the point it was
/// copied from.
const CAI_FLOOR_SLACK: f64 = 1e-5;

/// Solve under the energy ceiling `optimum + above_optimum` with the exact
/// credit — exactly as [`design_front_bounded`] does — while counting what a
/// box around one band of the front would additionally remove: an adaptation
/// floor and linear cuts through the box's corner at each of `spec.lambdas`.
/// Nothing is removed. Coding-only lattice.
#[doc(hidden)]
pub fn census_front_bounded(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    above_optimum: Energy,
    threads: usize,
    spec: &BoxSpec,
) -> Result<BoxCensusReport, BeamError> {
    let aa = residues(protein)?;
    let graph = dfa::build(&aa, usage);
    let config = credit_config(threads);

    let t0 = std::time::Instant::now();
    let energy_fold = scan::scan(params, &graph, 1.0, config);
    let optimum = energy_fold.c[graph.ix(graph.end())]
        .map(|st| st.mfe)
        .ok_or(BeamError::AchievableMinimumDisagrees {
            corner: Energy(0),
            folded: None,
        })?;
    let ceiling = Energy(optimum.0 + above_optimum.0);
    let windows = Bounds::with_ceiling(Ceiling::Absolute(ceiling))
        .resolve(optimum)
        .expect("a ceiling at or above the optimum is reachable");
    let outside = cost_to_go::Outside::compute(params, &graph, &energy_fold, config);
    drop(energy_fold);
    let credit = bounds::ExactCredit::new(outside, optimum);
    let wall_before_cuts = t0.elapsed().as_secs_f64();

    let codon_offset = graph.codon_offset();
    let slackened = BoxSpec {
        cai_floor_log: spec.cai_floor_log - CAI_FLOOR_SLACK,
        lambdas: spec.lambdas.clone(),
    };
    let cai_floor = slackened.cai_floor_log - codon_offset;
    let t1 = std::time::Instant::now();
    let census = box_cuts(
        params,
        &graph,
        UtrProfile::inert(),
        ceiling,
        &slackened,
        threads,
        true,
        &region::PairConstraint::inert(),
        &mut Vec::new(),
    );
    let cuts_s = t1.elapsed().as_secs_f64();

    let t2 = std::time::Instant::now();
    let (points, _, tally) =
        front_scan::fold_census(params, &graph, windows, Some(&credit), threads, &census);
    let wall_s = wall_before_cuts + t2.elapsed().as_secs_f64();
    let mut front = front_points(points, &graph, UtrProfile::inert());
    front.retain(|p| energy_of(p).0 <= ceiling.0);
    verify_corner(&front, optimum)?;
    Ok(BoxCensusReport {
        front,
        ceiling,
        cai_floor_accumulated: cai_floor,
        codon_offset,
        tally,
        wall_s,
        cuts_s,
    })
}

/// Slack under a box's floor on the emitted front: the fold's accumulator and
/// the reported split of it are sums taken in different orders.
const FLOOR_EPSILON: f64 = 1e-7;

/// The cuts of the box under `ceiling` and `band`'s floor over `graph`: one
/// exact scalar fold and one λ-general backward pass per weight. The floor is
/// stated on the reported adaptation accumulator (`cai_log + utr_profile_log`)
/// and converted here to the fold's own, which the lattice's normalization
/// offsets separate. `keep_inside` keeps each scalar fold for the census's
/// scale check; each cut's two passes are timed into `clock`.
#[allow(clippy::too_many_arguments)]
fn box_cuts(
    params: &Params,
    graph: &dfa::Dfa,
    profile: UtrProfile,
    ceiling: Energy,
    band: &BoxSpec,
    threads: usize,
    keep_inside: bool,
    region: &region::PairConstraint,
    clock: &mut Vec<CutClock>,
) -> BoxCuts {
    let config = credit_config(threads);
    let floor = cut_floor(graph, profile, band);
    BoxCuts::new(
        band.lambdas
            .iter()
            .map(|&lambda| {
                let t = std::time::Instant::now();
                let tables = scan::scan_with_region(params, graph, lambda, config, region.clone());
                let scan_s = t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                let outside =
                    cost_to_go::Outside::compute_at_any_lambda(params, graph, &tables, config);
                clock.push(CutClock {
                    lambda,
                    scan_s,
                    pass_s: t.elapsed().as_secs_f64(),
                });
                let cut = BoxCut::new(lambda, outside, ceiling, floor);
                if keep_inside {
                    cut.with_inside(tables)
                } else {
                    cut
                }
            })
            .collect(),
    )
}

/// The cuts of the box under `ceiling` and `band`'s floor from `chains`, the
/// scalar chains already run at `band.lambdas`, in that order; each chain's
/// cost is recorded into `clock`.
fn prepared_cuts(
    graph: &dfa::Dfa,
    profile: UtrProfile,
    ceiling: Energy,
    band: &BoxSpec,
    chains: Vec<CutChain>,
    clock: &mut Vec<CutClock>,
) -> BoxCuts {
    let weights: Vec<f64> = chains.iter().map(|chain| chain.clock.lambda).collect();
    assert_eq!(weights, band.lambdas, "a cut chain per weight, in order");
    let floor = cut_floor(graph, profile, band);
    BoxCuts::new(
        chains
            .into_iter()
            .map(|chain| {
                clock.push(chain.clock);
                BoxCut::new(chain.clock.lambda, chain.outside, ceiling, floor)
            })
            .collect(),
    )
}

/// `band`'s floor on the fold's own adaptation accumulator: stated on the
/// reported one (`cai_log + utr_profile_log`), which the lattice's
/// normalization offsets separate from it.
fn cut_floor(graph: &dfa::Dfa, profile: UtrProfile, band: &BoxSpec) -> f64 {
    band.cai_floor_log - (graph.utr_profile_offset(profile) + graph.codon_offset())
}

/// One exact scalar chain over `graph` at `lambda` on `threads` workers: the
/// fold, its optimum, and its backward pass — the credit's (`λ = 1`), taken
/// only when the fold reaches the exterior, or a cut's λ-general one.
struct ScalarChain {
    lambda: f64,
    optimum: Option<Energy>,
    outside: Option<cost_to_go::Outside>,
    scan_s: f64,
    pass_s: f64,
}

fn scalar_chain(
    params: &Params,
    graph: &dfa::Dfa,
    lambda: f64,
    threads: usize,
    credit: bool,
) -> ScalarChain {
    let config = credit_config(threads);
    let t = std::time::Instant::now();
    let tables = scan::scan(params, graph, lambda, config);
    let scan_s = t.elapsed().as_secs_f64();
    let optimum = tables.c[graph.ix(graph.end())].map(|st| st.mfe);
    let t = std::time::Instant::now();
    let outside = if credit {
        optimum.map(|_| cost_to_go::Outside::compute(params, graph, &tables, config))
    } else {
        Some(cost_to_go::Outside::compute_at_any_lambda(
            params, graph, &tables, config,
        ))
    };
    ScalarChain {
        lambda,
        optimum,
        outside,
        scan_s,
        pass_s: t.elapsed().as_secs_f64(),
    }
}

/// Run `jobs` jobs on `threads` workers in all, returning their results in job
/// order. `min(threads, jobs)` runners each take the next job in turn and run
/// it on their share of the workers, so no more than `threads` are ever busy.
/// With one runner every job runs in turn on all of them, on this thread.
fn share_threads<T: Send>(
    threads: usize,
    jobs: usize,
    job: impl Fn(usize, usize) -> T + Sync,
) -> Vec<T> {
    let threads = threads.max(1);
    let runners = threads.min(jobs);
    if runners <= 1 {
        return (0..jobs).map(|i| job(i, threads)).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut done: Vec<(usize, T)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..runners)
            .map(|r| {
                let workers = threads / runners + usize::from(r < threads % runners);
                let (next, job) = (&next, &job);
                task_clock::spawn(scope, move || {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= jobs {
                            break out;
                        }
                        out.push((i, job(i, workers)));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a scalar chain panicked"))
            .collect()
    });
    done.sort_by_key(|&(i, _)| i);
    done.into_iter().map(|(_, result)| result).collect()
}

/// The Pareto front of designs inside a box: every non-dominated `(MFE, CAI)`
/// design with `MFE <= ceiling` and `cai_log >= band.cai_floor_log` — the band
/// of the front between the floor's point and the ceiling.
///
/// Solved as [`design_front_bounded`] with the exact credit, plus the box's
/// cuts applied at every merge: the floor (`λ = 0`) and linear cuts through the
/// box's corner at each other weight in `band.lambdas`. The answer is the
/// ceiling's front filtered to the floor whatever weights are given; the
/// weights decide only how much of the fold is spent outside the box. Each
/// costs one exact scalar fold and one backward pass before the front fold.
///
/// A band holding no design returns an empty front.
pub fn design_front_box(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    ceiling: Ceiling,
    band: &BoxSpec,
    threads: usize,
) -> Result<Bounded<BoundedFront>, BeamError> {
    let aa = residues(protein)?;
    let graph = dfa::build(&aa, usage);
    solve_bounded(
        params,
        &graph,
        UtrProfile::inert(),
        || achievable_minimum(params, protein, usage),
        Prune::None,
        Bounds::with_ceiling(ceiling).with_exact_credit(),
        threads,
        Some(band),
    )
}

/// [`design_front_box`] over a flanked molecule. Under an active profile the
/// floor is on the fold's single adaptation accumulator,
/// `cai_log + utr_profile_log`, as the front itself is.
#[allow(clippy::too_many_arguments)]
pub fn design_front_box_with_utr(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    profile: UtrProfile,
    ceiling: Ceiling,
    band: &BoxSpec,
    threads: usize,
) -> Result<Bounded<BoundedFront>, BeamError> {
    let aa = residues(protein)?;
    let five = utr5.parse()?;
    let three = utr3.parse()?;
    let graph = dfa::build_with_utr_profile(&aa, usage, five.chain(), three.chain(), profile);
    solve_bounded(
        params,
        &graph,
        profile,
        || {
            let none = RegionConstraints::default();
            achievable_minimum_with_utr(params, protein, usage, utr5, utr3, profile, none)
        },
        Prune::None,
        Bounds::with_ceiling(ceiling).with_exact_credit(),
        threads,
        Some(band),
    )
}

/// The Pareto front of designs whose MFE lies in a band: every non-dominated
/// design with `lower < MFE <= upper`, plus the band's lower edge — the
/// best-adapted design meeting `lower`.
///
/// # Why this is a box
///
/// Along the front, adaptation rises with MFE. So once `A`, the best-adapted
/// design at or below `lower`, is known, the front points above `lower` are
/// exactly the ones at least as adapted as `A`: no front point sits between
/// `A`'s energy and `lower` (it would dominate `A` or be dominated by it).
/// An MFE floor cannot prune inside the fold — it is a statement about every
/// structure of a sequence — but that CAI floor can, and it is the same band.
///
/// Two solves over one lattice: a bounded solve at `lower` for `A`, then a box
/// solve at `upper` with `A`'s adaptation as the floor, cut at `lambdas`. Both
/// read the same exact `λ = 1` credit, built once. A single weight of `0.5` is the measured choice: on the one
/// instance swept, it removed as much of the fold as five weights together,
/// and a cut at `0` (the floor alone) removed too little to repay its own
/// scalar fold. A `lower` below the achievable minimum leaves no edge, and the
/// answer is the whole front up to `upper`; a `lower` above `upper` leaves an
/// empty band.
pub fn design_front_band(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lower: Ceiling,
    upper: Ceiling,
    lambdas: &[f64],
    threads: usize,
) -> Result<Bounded<BoundedFront>, BeamError> {
    let aa = residues(protein)?;
    let graph = dfa::build(&aa, usage);
    solve_band(
        params,
        &graph,
        UtrProfile::inert(),
        || achievable_minimum(params, protein, usage),
        (lower, upper),
        lambdas,
        threads,
    )
}

/// [`design_front_band`] over a flanked molecule; under an active profile the
/// band's edge is compared on `cai_log + utr_profile_log`, the accumulator the
/// front is non-dominated over.
#[allow(clippy::too_many_arguments)]
pub fn design_front_band_with_utr(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    profile: UtrProfile,
    lower: Ceiling,
    upper: Ceiling,
    lambdas: &[f64],
    threads: usize,
) -> Result<Bounded<BoundedFront>, BeamError> {
    let aa = residues(protein)?;
    let five = utr5.parse()?;
    let three = utr3.parse()?;
    let graph = dfa::build_with_utr_profile(&aa, usage, five.chain(), three.chain(), profile);
    solve_band(
        params,
        &graph,
        profile,
        || {
            let none = RegionConstraints::default();
            achievable_minimum_with_utr(params, protein, usage, utr5, utr3, profile, none)
        },
        (lower, upper),
        lambdas,
        threads,
    )
}

/// The band's two solves over `graph`, sharing one energy side: the bounded
/// solve at `lower` finds the edge, and the box at `upper` floors on it. The
/// cuts' chains do not depend on the edge, so they run beside the credit's
/// before either solve.
#[allow(clippy::too_many_arguments)]
fn solve_band(
    params: &Params,
    graph: &dfa::Dfa,
    profile: UtrProfile,
    corner: impl FnOnce() -> Result<Energy, BeamError>,
    (lower, upper): (Ceiling, Ceiling),
    lambdas: &[f64],
    threads: usize,
) -> Result<Bounded<BoundedFront>, BeamError> {
    let (mut side, chains) = EnergySide::with_cuts(params, graph, corner, lambdas, threads)?;
    let mut solve = |ceiling, band: Option<&BoxSpec>, chains| {
        solve_on(
            params,
            graph,
            profile,
            &mut side,
            chains,
            Prune::None,
            Bounds::with_ceiling(ceiling).with_exact_credit(),
            threads,
            band,
        )
        .map(|(solved, _)| solved)
    };
    let at_lower = solve(lower, None, Vec::new())?;
    match band_edge(at_lower, lambdas) {
        Some(band) => solve(upper, Some(&band), chains),
        None => solve(upper, None, Vec::new()),
    }
}

/// One end of a gap in a λ-sweep's hull: a design's energy and its adaptation
/// accumulator, `cai_log + utr_profile_log` — the two quantities the sweep
/// scalarizes and the front is non-dominated over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HullPoint {
    /// Minimum free energy in kcal/mol.
    pub mfe_kcal_mol: f64,
    /// `cai_log + utr_profile_log`, on the scale [`FrontPoint`] reports.
    pub adaptation_log: f64,
}

impl From<&FrontPoint> for HullPoint {
    fn from(p: &FrontPoint) -> Self {
        Self {
            mfe_kcal_mol: p.mfe_kcal_mol,
            adaptation_log: p.cai_log + p.utr_profile_log,
        }
    }
}

impl HullPoint {
    /// The energy in the fold's integer units.
    fn energy(self) -> Energy {
        Energy((self.mfe_kcal_mol * 100.0).round() as i32)
    }
}

/// The weight at which `a` and `b` score the same under the sweep's
/// scalarization `λ·MFE − 100·(1 − λ)·log CAI` (energy in the fold's integer
/// units): the slope of the hull edge between them. `None` unless `a` is
/// strictly more stable and strictly less adapted than `b`.
pub fn tie_weight(a: HullPoint, b: HullPoint) -> Option<f64> {
    let de = f64::from(b.energy().0 - a.energy().0);
    let dl = 100.0 * (b.adaptation_log - a.adaptation_log);
    (de > 0.0 && dl > 0.0).then(|| dl / (de + dl))
}

/// The Pareto designs between two points of a λ-sweep: every non-dominated
/// design with `MFE <= MFE(b)` and adaptation at least `a`'s, `a` and `b`
/// included when they are front points.
///
/// When `a` and `b` are adjacent vertices of the front's convex hull, the
/// designs strictly between them are exactly the ones no weighting of the two
/// objectives selects. They lie in the triangle `a`–`b`–`Q`,
/// `Q = (MFE(b), adaptation(a))`, which is solved as the box with corner `Q`
/// and one cut through it at [`tie_weight`]`(a, b)`. The weight is fixed by the
/// pair: at it the chord `a`–`b` is level, so the cut removes everything whose
/// best completion scores above `Q` there. A non-adjacent pair is still exact,
/// only a wider box.
///
/// The cut is one exact scalar fold and one backward pass, a fixed price
/// against a front fold that grows steeply with the ceiling. For a gap whose
/// ceiling sits close to the achievable minimum, the fold under the ceiling
/// alone can be cheaper than the cut, and the cut then costs more than it
/// saves; the answer is the same either way.
///
/// # Errors
///
/// [`BeamError::UnorderedHullPair`] unless `a` is strictly more stable and
/// strictly less adapted than `b`: otherwise one weakly dominates the other and
/// there is no gap between them.
pub fn design_front_between(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    a: HullPoint,
    b: HullPoint,
    threads: usize,
) -> Result<Bounded<BoundedFront>, BeamError> {
    let (ceiling, band) = between_box(a, b)?;
    design_front_box(params, protein, usage, ceiling, &band, threads)
}

/// [`design_front_between`] with the cut's cost to go supplied: `outside` is
/// any lower bound on the λ-general cost to go at [`tie_weight`]`(a, b)` — the
/// exact pass, or a looser one — or `None` for no cut, which leaves the
/// ceiling, the credit and the floor. The answer is the same for every
/// admissible bound; only the fold's work moves.
///
/// # Errors
///
/// As [`design_front_between`].
#[doc(hidden)]
pub fn design_gap_with_cut(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    a: HullPoint,
    b: HullPoint,
    outside: Option<cost_to_go::Outside>,
    threads: usize,
) -> Result<(Bounded<BoundedFront>, SolveClock), BeamError> {
    let (ceiling, mut band) = between_box(a, b)?;
    let lambda = band.lambdas[0];
    let chains = match outside {
        Some(outside) => vec![CutChain {
            outside: Arc::new(outside),
            clock: CutClock {
                lambda,
                scan_s: 0.0,
                pass_s: 0.0,
            },
        }],
        None => {
            band.lambdas.clear();
            Vec::new()
        }
    };
    let aa = residues(protein)?;
    let graph = dfa::build(&aa, usage);
    let corner = || achievable_minimum(params, protein, usage);
    let mut side = EnergySide::new(
        params,
        &graph,
        corner,
        true,
        threads,
        region::PairConstraint::inert(),
    )?;
    side.build_credit(params, &graph, threads);
    solve_with(
        params,
        &graph,
        UtrProfile::inert(),
        &side,
        chains,
        Prune::None,
        Bounds::with_ceiling(ceiling).with_exact_credit(),
        threads,
        Some(&band),
    )
}

/// [`design_front_between`] over a flanked molecule; `a` and `b` carry the
/// adaptation accumulator `cai_log + utr_profile_log`, as the front does.
#[allow(clippy::too_many_arguments)]
pub fn design_front_between_with_utr(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    profile: UtrProfile,
    a: HullPoint,
    b: HullPoint,
    threads: usize,
) -> Result<Bounded<BoundedFront>, BeamError> {
    let (ceiling, band) = between_box(a, b)?;
    design_front_box_with_utr(
        params, protein, usage, utr5, utr3, profile, ceiling, &band, threads,
    )
}

/// One gap of a walked front: its two hull vertices (adjacent on the hull
/// unless the walk stopped refining them, see [`WalkOptions::stop_slack`]),
/// the weight at which they tie, the Pareto designs
/// between them, and what the gap's solve cost.
#[derive(Debug, Clone)]
pub struct WalkGap {
    /// The more stable vertex.
    pub a: HullPoint,
    /// The more adapted vertex.
    pub b: HullPoint,
    /// [`tie_weight`]`(a, b)`.
    pub lambda: f64,
    /// [`design_front_between`]`(a, b)`'s front.
    pub front: Vec<FrontPoint>,
    /// Energy slices the gap was solved in; 1 for one box.
    pub slices: usize,
    /// The cost to go its cut read.
    pub cut: GapCut,
    /// The gap's solve: its cut's scan (the sweep's own, which certified the
    /// pair adjacent) and backward pass, and its front fold, on its share of
    /// the workers. The credit's passes are paid once for the walk, in
    /// [`WalkedFront::credit_s`].
    pub clock: SolveClock,
}

/// The whole front between two weights, one hull gap at a time.
#[derive(Debug, Clone)]
pub struct WalkedFront {
    /// The hull's vertices between the two weights, most stable first.
    pub vertices: Vec<HullPoint>,
    /// Every vertex of the hull between the two weights, most stable first:
    /// [`Self::vertices`] and those inside a gap folded whole before it was
    /// refined to adjacency, which no probe found. The front's points are
    /// labelled on the hull from these.
    pub hull: Vec<HullPoint>,
    /// Every pair of consecutive vertices, solved, in the same order.
    pub gaps: Vec<WalkGap>,
    /// The gaps' fronts, joined at their shared vertices: every Pareto design
    /// from the first vertex to the last, most stable first.
    pub front: Vec<FrontPoint>,
    /// Scalar folds the hull sweep ran, the certifying ones included.
    pub sweep_scans: usize,
    /// How many of them were cut at their pair's tie value.
    pub probes_cut: usize,
    /// Seconds from the sweep's start until the pass grid's last pass was
    /// pooled; zero without a grid.
    pub grid_s: f64,
    /// The most grid passes taken in one traversal of the lattice: 1 when
    /// each was taken alone or there was no grid.
    pub grid_lanes: usize,
    /// The walk's [`WalkOptions::stop_slack`]: a run that set it may store a
    /// different co-optimal structure for a point than one that did not.
    pub stop_slack: Option<f64>,
    /// Wall seconds from the sweep's first fold to its last. Gaps already
    /// certified are solved in this time too, beside the sweep.
    pub sweep_s: f64,
    /// Wall seconds after the sweep's last fold, while the remaining gaps
    /// are solved.
    pub gaps_s: f64,
    /// The sweep's depth: the most refinements any pair took from the first.
    pub rounds: usize,
    /// Seconds for the credit's `λ = 1` fold and backward pass, built once.
    pub credit_s: f64,
    /// Every task the walk's queue ran, in the order each finished.
    pub timeline: Vec<TaskSpan>,
    /// With [`WalkOptions::refine_loose`], each watched bracket's decision,
    /// by ascending weight; empty otherwise.
    pub refinements: Vec<BracketRefinement>,
    /// What stopped the walk before its whole front, `"time"` or `"memory"`,
    /// or `None` for a front this walk finished.
    pub stopped_by: Option<&'static str>,
    /// The boxes a budget left unsolved, each between two hull points of
    /// `vertices`, the more stable first. Every point of `front` is on the
    /// front whatever is open; the open boxes are what it may still be
    /// missing.
    pub open: Vec<(HullPoint, HullPoint)>,
    /// The open boxes whose gap was solved part way, one per such box: its
    /// front points up to the rest's `from` are in `front`.
    pub rests: Vec<WalkRest>,
}

/// An open box whose gap fold stopped between two of its slices: every front
/// point up to `from` is among the walk's points. Resuming solves only the
/// energies above `from`, for designs more adapted than `floor`, the best
/// adaptation found at or below it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WalkRest {
    /// The box's two ends' energies, the more stable first.
    pub lo: Energy,
    pub hi: Energy,
    /// The energy up to which the gap is solved: one unit below `lo` when it
    /// was stopped before its first slice.
    pub from: Energy,
    /// The best adaptation at or below `from`, as the front's `cai_log`
    /// reports it.
    pub floor: f64,
}

/// One task of a walk's queue: what it was, when it was queued, started and
/// finished (seconds from the sweep's start), and on how many workers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaskSpan {
    /// What the task did.
    pub kind: TaskKind,
    /// The weight it ran at: a probe's or a gap's tie weight, a grid weight.
    pub lambda: f64,
    /// When it was queued.
    pub queued_s: f64,
    /// When it started.
    pub start_s: f64,
    /// When it finished.
    pub end_s: f64,
    /// The workers it ran on.
    pub workers: usize,
    /// The most workers it ran on at once, idle ones lent to it included
    /// ([`WalkOptions::lend_idle_workers`]).
    pub peak_workers: usize,
    /// The CPU seconds its threads spent ([`task_clock::TaskClock`]), which
    /// host load does not inflate the way wall seconds are.
    pub cpu_s: f64,
    /// The process's resident memory when it finished, in KiB; `None` where
    /// it cannot be read, which is not the same as holding nothing.
    pub rss_kb: Option<u64>,
    /// The process's peak resident memory so far when it finished, in KiB;
    /// `None` where it cannot be read. Readable on more platforms than
    /// [`Self::rss_kb`], the peak coming from `getrusage` where `/proc` is
    /// absent.
    pub peak_rss_kb: Option<u64>,
}

/// What a [`TaskSpan`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    /// An exact pass at a grid weight.
    Grid,
    /// A sweep probe, `cut` when a pooled bracket cut its scan.
    Probe {
        /// Whether its scan was cut.
        cut: bool,
    },
    /// A pair within [`WalkOptions::stop_slack`] under a pooled bracket,
    /// handed to its fold without a probe.
    Narrow,
    /// A certified gap's fold.
    Fold,
    /// The credit's `λ = 1` fold and backward pass
    /// ([`WalkOptions::overlap_credit`]).
    Credit,
    /// An end's fold ([`WalkOptions::overlap_credit`]).
    End,
}

/// How [`design_front_walk`] divides the front into solves and cuts them.
/// None changes the front's objectives; each trades one cost against another.
/// The default refines every pair to adjacency with one uncut box per gap,
/// and cuts at the bracket's passes too once a chord is asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct WalkOptions {
    /// Solve a pair as one gap, without refining it, once its cut leaves at
    /// most this slack: `λ*·ΔE`, the tie weight times the pair's energy gap in
    /// the fold's integer units. `None` refines every pair to adjacency. A
    /// pair whose gap reads a chord ([`Self::cut_probes`] under a pooled
    /// bracket) is not probed at all.
    pub stop_slack: Option<f64>,
    /// Solve a gap whose cut leaves more than this slack as consecutive
    /// energy slices of about this slack each, in place of one box. `None`
    /// solves every gap as one box. Read only while [`Self::slice_width`] is
    /// `None`.
    pub slice_slack: Option<f64>,
    /// Solve a gap wider than this energy window, in the fold's integer
    /// units, as consecutive energy slices about this wide, in place of
    /// [`Self::slice_slack`]'s count. A slice's fold cost follows its energy
    /// width, the distance its absolute ceiling admits, whatever the tie
    /// weight; sliced by slack, a wide gap at a low weight stays one box and a
    /// narrow one at a high weight is cut many times. `None` slices by slack.
    /// Default [`SLICE_WIDTH`].
    pub slice_width: Option<f64>,
    /// Solve a gap with no cut, and so without its backward pass, when its
    /// ceiling lies at most this far above the optimum, in the fold's integer
    /// units. Near the optimum the credit alone keeps the fold cheap, and how
    /// cheap follows the ceiling's distance above the optimum, not the gap's
    /// width. `None` cuts every gap.
    pub uncut_within: Option<i32>,
    /// Give a gap, in place of its own backward pass, the chord between two
    /// passes other gaps of the walk computed, when their weights bracket its
    /// tie weight within this width. The cost to go is concave in `λ`, so the
    /// chord bounds it from below and the cut stays admissible; the wider the
    /// bracket, the more the fold keeps. Every computed pass is held until the
    /// walk ends. `None` computes every cut gap's own pass.
    pub chord_bracket: Option<f64>,
    /// With [`Self::chord_bracket`], cut a probe whose tie weight the pool
    /// brackets at its pair's tie value, with the bracket's chord as the cost
    /// to go: it still finds the optimum when the optimum splits the pair, and
    /// its gap, if it certifies one, reads that chord anyway.
    pub cut_probes: bool,
    /// Fill the pool with exact passes at this grid of weights, queued ahead
    /// of every probe, so a probe or gap finds a bracket at most one grid step
    /// wide. The bracket width ([`Self::chord_bracket`]) is raised to the
    /// grid's widest step. Only a probe that starts before its bracket is
    /// pooled runs exact and gives its gap a pass of its own; the pool holds
    /// the grid and those passes.
    pub pass_grid: Option<PassGrid>,
    /// With [`Self::pass_grid`], start the sweep from the grid's own optima.
    /// A grid pass is an exact scan, so its optimum is a point of the hull,
    /// and the sweep need only refine the pairs between consecutive ones.
    /// That removes the probes at the top of the sweep, whose wide pairs a
    /// cut prunes least, and lets a pair's probes start as soon as the grid
    /// passes on either side of it are pooled, beside the rest of the grid.
    pub seed_sweep: bool,
    /// With [`Self::seed_sweep`], build the credit and fold the ends as
    /// tasks of the walk's queue (the stable end alone with
    /// [`Self::grid_ends`], which takes the adapted end from the grid), beside the grid's passes, rather than on
    /// every worker before the queue starts. Nothing in the grid reads them:
    /// the pairs wait for the credit, and the end pairs for their ends, as
    /// they wait for the grid passes on either side. One scalar chain gains
    /// little from more than a few workers, so the workers it would leave
    /// idle take grid passes instead. Ignored without a seeded grid. Default
    /// on: the same work, never slower in any walk measured, and the
    /// credit's tables are not held beside a second setup chain's.
    pub overlap_credit: bool,
    /// With [`Self::chord_bracket`], cut a gap that reads a chord also at up
    /// to this many pooled exact passes on each side of its tie weight, the
    /// nearest first (1: the two the chord is drawn between), each at its own
    /// weight. The box's corner bounds every design in it at any weight, so
    /// each cut is admissible; they add the directions a single cut at the tie
    /// weight cannot see, near the box's corners. `0` cuts at the tie weight
    /// alone. Default 1: never slower in any walk measured, and further passes
    /// add nothing. A tied design may be spelled differently from a walk at
    /// `0`; objectives and vertices are the same.
    pub bracket_cuts: usize,
    /// With a seeded grid and the credit built before the queue, take the
    /// adapted end from the grid rather than from a fold of its own: the
    /// grid's lowest weight (`0`) becomes the walk's `lo`, and that pass's
    /// optimum is the end's vertex, the same scan the end fold runs. The
    /// stable end is folded beside the credit, each on half the workers,
    /// since one scalar chain gains little from all of them; with
    /// [`Self::overlap_credit`], both are queue tasks instead. Ignored
    /// without a seeded grid. Default on: never slower in any walk measured
    /// and no larger in peak memory on a quiet host. A tied design may be
    /// spelled differently from a walk without it; objectives and vertices
    /// are the same.
    pub grid_ends: bool,
    /// Fold a gap whose energy width `ΔE` (in the fold's integer units)
    /// exceeds this ahead of the probes, the widest first, rather than after
    /// them. A fold's cost follows its gap's width far more closely than its
    /// slack, and a heavy fold queued behind the sweep can end up running
    /// alone on one worker while the rest stand idle. `None` folds after
    /// every probe, by slack.
    pub heavy_fold: Option<i32>,
    /// Take the queued folds widest energy window `ΔE` first rather than by
    /// slack `λ*·ΔE`. Slack scales the window by the tie weight, so a wide gap
    /// at a low weight — among the costliest folds of a walk — is taken among
    /// the last and can run alone on one worker while the rest stand idle.
    /// Ordering only: each fold's input is what it would be under either
    /// order once the pool holds its bracket. Default on: with
    /// [`SLICE_WIDTH`], the same objectives on every walk measured and never
    /// measurably slower; a tied design may be spelled differently from a walk
    /// by slack.
    pub fold_by_width: bool,
    /// Run at most this many tasks at once, each on a share of the workers,
    /// in place of one task per worker. The walk's peak memory is set by the
    /// grid's exact scans running side by side, one per worker, each holding
    /// its whole tables; fewer at once, on more threads each, hold fewer
    /// tables at a time at the cost of the threaded scan's efficiency.
    /// `None` runs up to one task per worker.
    pub max_running: Option<usize>,
    /// With a seeded grid, add a candidate grid's weights inside a bracket of
    /// the pass grid as exact passes when the bracket's gaps read a loose
    /// chord, decided once the bracket's probes are done and before its folds
    /// start. `None` keeps the pass grid as given; ignored without a seeded
    /// grid.
    pub refine_loose: Option<RefineLoose>,
    /// With a seeded grid, add exact passes inside a bracket of the pass grid,
    /// at held gaps' tie weights, when the fold work they are predicted to
    /// save exceeds their price, decided once the bracket's probes are done
    /// and before its folds start. Every bracket of the grid is watched.
    /// Takes the place of [`Self::refine_loose`] when both are set; ignored
    /// without a seeded grid.
    pub refine_priced: Option<RefinePriced>,
    /// Lend a gap fold the queue's idle workers at each of its slices while
    /// no task waits to start, so a long fold in the walk's tail widens onto
    /// workers that would otherwise idle. Each loan lasts one slice; the
    /// front's objectives do not change. On by default: it costs a few
    /// percent of CPU and saves wall wherever a fold runs alone.
    pub lend_idle_workers: bool,
    /// Spend a gap fold's loan of idle workers ([`Self::lend_idle_workers`],
    /// implied) on running its next slices at once, one more slice than
    /// workers lent, each with the floor known when they start, rather than
    /// widening one slice: a slice's solve gains little from more threads,
    /// and a slice with a lower floor is still exact. The front's objectives
    /// do not change.
    pub speculative_slices: bool,
    /// What stops the walk short, leaving the pairs and gaps it had not
    /// started as [`WalkedFront::open`]. The ends and the pass grid are never
    /// skipped, since a walk without them has nothing to resume from, so a
    /// budget already spent when they finish stops the walk at its first
    /// probe. Default no budget, which runs the walk to its whole front.
    pub budget: WalkBudget,
    /// Let a gap fold cut by a chord change to an exact pass at its own tie
    /// weight at a slice boundary, once it runs on every worker (nothing else
    /// runs or waits) with two or more slices left and its last slice's CPU
    /// times the slices left is at least this many of the walk's mean
    /// grid-pass CPU. The pass runs on the fold's workers and joins the pool;
    /// every slice is exact under either cut, so the front's objectives do
    /// not change. A chord is loose where a fold is long, so an own cut
    /// mostly costs a fraction of the rest. `None` keeps the chord.
    pub own_when_alone: Option<f64>,
    /// As [`Self::own_when_alone`], without waiting for the fold to hold
    /// every worker: the change is made once the rest of the fold, read off
    /// its last slice, is at least this many mean grid passes of CPU,
    /// whatever else runs. A fold holding every worker runs the pass itself;
    /// one beside other tasks queues it as a grid pass, keeps its chord, and
    /// takes the pass at the first slice boundary after it is pooled.
    pub own_when_costly: Option<f64>,
    /// With [`Self::own_when_costly`], read a running slice's CPU every this
    /// many seconds and queue the pass during the slice, once its CPU so far
    /// times the slices after it reaches the ratio, so the pass can be pooled
    /// by the slice's end rather than one slice later. The CPU read is the
    /// slice's calling thread alone, a lower bound on the slice, so it asks no
    /// earlier than the slice's whole CPU would. Not while the fold holds
    /// every worker, where the pass runs at the boundary itself. `None` asks
    /// only at slice boundaries.
    pub own_watch_s: Option<f64>,
    /// With [`Self::own_when_alone`] or [`Self::own_when_costly`], change only
    /// a fold whose chord is loose at the root by at least this many kcal/mol
    /// ([`chord_looseness`]), read when the fold starts. What an exact cut
    /// saves a fold follows that looseness, and a tight chord's fold keeps
    /// most of its cost under its own pass. `None` changes any chorded fold.
    pub own_min_looseness: Option<f64>,
    /// Also cut each gap's slices at the box's floor alone (`λ = 0`), with no
    /// pass ([`Bounds::with_floor_cut`]): a point already less adapted than the
    /// best design below the slice has no completion in it, since no edge adds
    /// adaptation. The front's objectives do not change.
    pub floor_cut: bool,
    /// Stop once this many gap slices have been solved, between one slice and
    /// the next, keeping each stopped gap's rest ([`WalkedFront::rests`]).
    /// Unlike the wall budget it stops every run at the same place.
    pub gap_folds: Option<usize>,
    /// Halve a gap slice whose fold stores more than this many points, in
    /// place of folding it whole ([`SolveClock::halved`]). `None` halves only
    /// a slice inside which the walk's memory budget runs out.
    pub slice_points: Option<usize>,
}

/// When [`WalkOptions::refine_loose`] adds passes inside a bracket of the pass
/// grid.
///
/// A gap's chord is loose at the root by
/// `[g(λ*) − (t·g(λ_lo) + (1 − t)·g(λ_hi))]/λ*` kcal/mol, with `g` the
/// scalarized optimum over the hull's vertices and `λ_lo`, `λ_hi` its chord's
/// ends. A bracket refines when the mean of that looseness over its gaps,
/// weighted by each gap's energy width, exceeds the threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct RefineLoose {
    /// The weights a refined bracket adds: those of this grid strictly inside
    /// it.
    pub candidates: PassGrid,
    /// The weighted mean looseness, kcal/mol, above which a bracket refines.
    pub threshold_kcal: f64,
}

/// When [`WalkOptions::refine_priced`] adds passes inside a bracket.
///
/// Each held gap's fold work is predicted in exact passes at the walk's
/// length ([`FoldCost`]) under its chord from the passes pooled now and under
/// the chord a pass at another held gap's tie weight would leave; a pass at a
/// gap's own tie weight cuts that gap exactly. Passes are chosen one at a
/// time, each the one predicted to save most, while the saving exceeds
/// `margin_passes`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefinePriced {
    /// The fold-work predictor.
    pub cost: FoldCost,
    /// The predicted saving, in passes, an added pass must exceed.
    pub margin_passes: f64,
    /// The most passes added inside one bracket.
    pub max_passes: usize,
}

impl Default for RefinePriced {
    fn default() -> Self {
        Self {
            cost: FOLD_COST,
            margin_passes: 1.0,
            max_passes: 1,
        }
    }
}

/// A gap fold's work in exact passes at the same length:
/// `exp(intercept + width·ln W + slices·ln S + loose·L + loose_lambda·L·λ*
/// + lambda·ln λ* + residues·ln(n/1000))`, with `W` the gap's energy width in
/// kcal/mol, `S` its slice count, `L` its root chord looseness in kcal/mol and
/// `n` the protein's length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoldCost {
    /// The constant term.
    pub intercept: f64,
    /// The coefficient of `ln W`.
    pub width: f64,
    /// The coefficient of `ln S`.
    pub slices: f64,
    /// The coefficient of `L`.
    pub loose: f64,
    /// The coefficient of `L·λ*`.
    pub loose_lambda: f64,
    /// The coefficient of `ln λ*`.
    pub lambda: f64,
    /// The coefficient of `ln(n/1000)`.
    pub residues: f64,
}

impl FoldCost {
    /// The predicted work, in passes, of a fold at tie weight `lambda`,
    /// `width_kcal` wide in `slices` slices, cut by a chord `loose_kcal` loose
    /// at the root, on a protein of `residues`.
    pub fn passes(
        &self,
        lambda: f64,
        width_kcal: f64,
        slices: usize,
        loose_kcal: f64,
        residues: usize,
    ) -> f64 {
        let loose = loose_kcal.max(0.0);
        (self.intercept
            + self.width * width_kcal.max(0.01).ln()
            + self.slices * (slices.max(1) as f64).ln()
            + self.loose * loose
            + self.loose_lambda * loose * lambda
            + self.lambda * lambda.ln()
            + self.residues * (residues as f64 / 1000.0).ln())
        .exp()
    }
}

/// [`FoldCost`] fitted on the folds of 36 whole walks of 14 proteins of
/// 872–1270 residues: fold CPU-seconds over the walk's mean grid-pass
/// CPU-seconds, by Poisson regression on every chord-cut fold, so that a
/// bracket's summed prediction is unbiased (predicted over actual: median
/// 1.10 over 224 brackets). It predicts a bracket's work, not what tightening
/// its chord saves: on 32 brackets where a walk with added passes can be
/// compared with one without, the saving it predicts calls 20 right against
/// the passes added (refit leaving each protein out), fewer than the
/// looseness threshold's 25.
pub const FOLD_COST: FoldCost = FoldCost {
    intercept: -4.124,
    width: 0.835,
    slices: 0.39,
    loose: 0.107,
    loose_lambda: 0.644,
    lambda: 0.114,
    residues: 1.591,
};

/// The threshold of [`RefineLoose::threshold_kcal`] fitted on whole walks.
///
/// Measured on two proteins (P0DTC2 at 1000, 1100 and 1273 residues, Q5VYS8 at
/// 1000 and 1100), standard grid against [`PassGrid::refined`]: across the 15
/// brackets of `[0.14, 0.26]`, those where the refined passes paid read
/// 3.42–4.16 kcal/mol, those where they did not 0.83–2.96, apart from one
/// bracket worth a tenth of its cost at 1.96. A fit to five walks, with a
/// margin of 1.16× on each side.
pub const REFINE_LOOSE_KCAL: f64 = 3.2;

/// The default of [`WalkOptions::slice_width`]: energy slices about 2.5
/// kcal/mol wide, in the fold's integer units.
///
/// With [`WalkOptions::fold_by_width`], measured on whole walks: 0.69× the
/// instructions of the slack order at 1000 residues (Q5VYS8), where a wide gap
/// at a low tie weight had run alone at the end of the walk; about 1% fewer at
/// 500 and 0.995× at 150. Objectives were the same on every walk, in sample
/// and on eight held-out proteins of 1006–1270 residues.
pub const SLICE_WIDTH: f64 = 250.0;

/// An earlier walk of the same molecule to continue: what it settled, and the
/// boxes its budget left open.
///
/// The walk that reads it folds those boxes and nothing else. Its hull is the
/// earlier walk's, so the ends and the sweep are not folded again; the credit
/// and the pass grid are, since every gap fold reads them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WalkResume {
    /// The earlier walk's points, each with whether it is a hull vertex.
    pub points: Vec<(FrontPoint, bool)>,
    /// Its open boxes as their two hull points `(mfe kcal/mol, cai_log)`, the
    /// more stable first. An end that is a vertex among `points` is taken
    /// from there, so a box resumed twice reads the same point.
    pub open: Vec<((f64, f64), (f64, f64))>,
    /// Its partly solved boxes as `(mfe_lo, mfe_hi)` kcal/mol, each the
    /// energies of an open box's ends, with the energy up to which the gap is
    /// solved and the best `cai_log` there ([`WalkRest`]).
    pub rests: Vec<((f64, f64), (f64, f64))>,
}

/// How a walk runs beyond its inputs: an earlier walk to continue, and what to
/// hand the front found so far.
#[derive(Clone, Copy, Default)]
pub struct WalkRun<'a> {
    /// An earlier walk of the same molecule whose open boxes this one
    /// settles, or `None` to walk the whole front.
    pub resume: Option<&'a WalkResume>,
    /// Called once both ends of the hull are known (folded, or taken from the
    /// walk resumed), before any pair settles, and after each pair of hull
    /// points settles, solved as a gap or refined into two pairs, with the
    /// walk so far: every point of its `front` is a front point, and every one
    /// it is missing lies in one of its `open` boxes, so it can be resumed
    /// from. Calls are made one at a time, each with a later state than the
    /// one before. Its `vertices`, `gaps`, `front`,
    /// `open`, `sweep_scans`, `probes_cut` and `rounds` are the walk's so far;
    /// its clocks are zero, its timeline and refinements empty, and its
    /// `stopped_by` is `None`.
    pub checkpoint: Option<&'a (dyn Fn(&WalkedFront) + Sync)>,
}

/// What may stop a walk short: a wall-clock deadline and a resident-memory
/// cap, each checked before a task starts; both also inside every gap fold of
/// the motif walk and before each of its slices.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WalkBudget {
    /// Seconds from the walk's start.
    pub max_secs: Option<f64>,
    /// Resident memory, in GB.
    pub max_rss_gb: Option<f64>,
}

/// The process's resident memory in kB, or `None` where it cannot be read.
///
/// One reader, in `task_clock`. Two readings of the same quantity drift: a
/// second peak reader that only knew `/proc` reported no peak at all on a
/// platform where the first one reported a figure.
pub(crate) fn rss_kb() -> Option<u64> {
    crate::task_clock::rss_kb()
}

impl Default for WalkOptions {
    fn default() -> Self {
        Self {
            stop_slack: None,
            slice_slack: None,
            slice_width: Some(SLICE_WIDTH),
            uncut_within: None,
            chord_bracket: None,
            cut_probes: false,
            pass_grid: None,
            seed_sweep: false,
            overlap_credit: true,
            bracket_cuts: 1,
            grid_ends: true,
            heavy_fold: None,
            fold_by_width: true,
            max_running: None,
            refine_loose: None,
            refine_priced: None,
            lend_idle_workers: true,
            speculative_slices: false,
            budget: WalkBudget::default(),
            own_when_alone: None,
            own_when_costly: None,
            own_watch_s: None,
            own_min_looseness: None,
            floor_cut: false,
            gap_folds: None,
            slice_points: None,
        }
    }
}

/// Weights from 0 to 1 at which [`WalkOptions::pass_grid`] takes exact
/// passes, in segments of their own step, and 1.
///
/// A grid weight costs an exact scan and a backward pass, and serves only the
/// probes and gaps whose tie weights it brackets. Where a bracket's looseness
/// is cheap (high `λ`, where gaps near the optimum take no cut and a cut
/// probe stays fast under a wide bracket) a wide step saves passes for
/// nothing; where it is costly (low `λ`, longer molecules) a narrow one keeps
/// the folds cheap.
#[derive(Debug, Clone, PartialEq)]
pub struct PassGrid {
    /// `(until, step)` with `until` ascending: a weight below `until` steps by
    /// `step` to the next, and a weight past the last `until` by the last step.
    pub segments: Vec<(f64, f64)>,
}

/// The residue count from which [`PassGrid::for_residues`] picks
/// [`PassGrid::refined`] over [`PassGrid::standard`].
///
/// Measured on one protein (P0DTC2), whole walks on both grids with
/// identical fronts: the refined grid costs 1.19× the standard grid's
/// instructions at 1000 residues, 0.86× at 1100 and 0.74× at 1273. Below the
/// threshold the standard grid's chords are already tight enough and the nine
/// added passes are overhead; above it the standard grid's 0.05 steps over
/// `[0.16, 0.26]` leave the gaps there cut loosely, and the looseness is paid in
/// the fold's serial bifurcation phase. The ratio interpolates through 1 near
/// 1050 residues. Where the break sits depends on the weights at which a
/// molecule's hull ties, so another protein's threshold may differ.
pub const REFINED_GRID_FROM_RESIDUES: usize = 1050;

impl PassGrid {
    /// 17 weights: every 0.02 to 0.16, every 0.05 to 0.31, every 0.1 to 0.71,
    /// and 1.
    pub fn standard() -> Self {
        Self {
            segments: vec![(0.16, 0.02), (0.31, 0.05), (0.71, 0.1), (1.0, 0.29)],
        }
    }

    /// [`Self::standard`] with every 0.01 over `[0.14, 0.26]`: 26 weights, the
    /// standard grid's 17 and nine between them.
    pub fn refined() -> Self {
        Self {
            segments: vec![
                (0.14, 0.02),
                (0.26, 0.01),
                (0.31, 0.05),
                (0.71, 0.1),
                (1.0, 0.29),
            ],
        }
    }

    /// The grid for a molecule of `residues` coding residues:
    /// [`Self::refined`] from [`REFINED_GRID_FROM_RESIDUES`] on, and
    /// [`Self::standard`] below it. The front is the same under either; only
    /// the cost differs.
    pub fn for_residues(residues: usize) -> Self {
        if residues >= REFINED_GRID_FROM_RESIDUES {
            Self::refined()
        } else {
            Self::standard()
        }
    }

    /// The grid whose segments `given` spells as `until:step,...`; `None` when
    /// a segment does not parse.
    pub fn parse(given: &str) -> Option<Self> {
        let segments = given
            .split(',')
            .map(|seg| {
                let (until, step) = seg.split_once(':')?;
                Some((until.trim().parse().ok()?, step.trim().parse().ok()?))
            })
            .collect::<Option<Vec<(f64, f64)>>>()?;
        Some(Self { segments })
    }

    /// Every `fine` up to `switch` and every `coarse` above it.
    pub fn two_step(fine: f64, coarse: f64, switch: f64) -> Self {
        Self {
            segments: vec![(switch, fine), (1.0, coarse)],
        }
    }

    /// The grid's weights, ascending, from 0 to 1 inclusive.
    ///
    /// # Panics
    ///
    /// When the grid has no segment or a step is not positive.
    pub fn weights(&self) -> Vec<f64> {
        let last = self.segments.last().map_or(0.0, |&(_, step)| step);
        assert!(
            self.segments.iter().all(|&(_, step)| step > 0.0) && last > 0.0,
            "a pass grid needs segments with positive steps"
        );
        let round = |w: f64| (w * 1e9).round() / 1e9;
        let mut weights = Vec::new();
        let mut w = 0.0;
        while w < 1.0 {
            weights.push(round(w));
            w += self
                .segments
                .iter()
                .find(|&&(until, _)| w < until - 1e-12)
                .map_or(last, |&(_, step)| step);
        }
        weights.push(1.0);
        weights
    }

    /// The widest step between neighbouring weights.
    pub fn widest(&self) -> f64 {
        self.weights()
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(0.0, f64::max)
    }
}

/// The cost to go a gap's cut read.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GapCut {
    /// The exact backward pass at the gap's tie weight.
    Exact,
    /// The chord between the exact passes at `lo` and `hi`, two other gaps'
    /// tie weights around this one's; `lo == hi` is that pass itself.
    Chord {
        /// The lower weight.
        lo: f64,
        /// The upper weight.
        hi: f64,
    },
    /// No cut: the ceiling, the credit and the floor alone.
    Uncut,
}

/// How many energy slices a gap is solved in: by its energy width when
/// `width` is set ([`WalkOptions::slice_width`]), else by its slack `λ*·ΔE`
/// when `slack` is set ([`WalkOptions::slice_slack`]), else one. Any count is
/// exact (see `solve_gap`); the count only sets the cost.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Slicing {
    /// Slack per slice, in the fold's integer units.
    pub slack: Option<f64>,
    /// Energy width per slice, in the fold's integer units.
    pub width: Option<f64>,
}

impl Slicing {
    /// The slice count of a gap from energy `e_a` to `e_b` (integer units)
    /// at tie weight `lambda`: the gap's width or slack over the per-slice
    /// amount, rounded up, and at most one slice per energy unit.
    pub fn count(self, lambda: f64, e_a: i32, e_b: i32) -> usize {
        let width = f64::from(e_b - e_a);
        let (amount, each) = match (self.width, self.slack) {
            (Some(each), _) => (width, each),
            (None, Some(each)) => (lambda * width, each),
            (None, None) => return 1,
        };
        if each > 0.0 && amount > each {
            ((amount / each).ceil() as usize).clamp(1, (e_b - e_a).max(1) as usize)
        } else {
            1
        }
    }
}

impl WalkOptions {
    /// The options `derna front` walks a molecule of `residues` coding
    /// residues under: the grid [`PassGrid::for_residues`] picks, the sweep
    /// started from its optima, and probes cut under its chords. The same
    /// front as the defaults, with the hull found in fewer and cheaper folds
    /// (3.3x and 5.3x less wall at 200 and 250 residues on 8 threads, 2.7x and
    /// 3.6x on one).
    ///
    /// A development build can also turn on [`Self::floor_cut`] and set
    /// [`Self::stop_slack`].
    pub fn for_residues(residues: usize) -> Self {
        Self {
            pass_grid: Some(PassGrid::for_residues(residues)),
            seed_sweep: true,
            cut_probes: true,
            floor_cut: crate::dev_env::var(crate::dev_env::Setting::WalkFloorCut)
                .is_ok_and(|v| v == "1"),
            stop_slack: crate::dev_env::var(crate::dev_env::Setting::WalkStopSlack)
                .ok()
                .and_then(|v| v.trim().parse().ok()),
            ..Self::default()
        }
    }

    /// The walk's slice rule.
    pub fn slicing(&self) -> Slicing {
        Slicing {
            slack: self.slice_slack,
            width: self.slice_width,
        }
    }
}

/// What a [`SliceWatch`] calls: the slice's CPU seconds so far, the slices
/// after it, the next slice's number and the workers it holds.
type SliceAsk<'a> = dyn Fn(f64, usize, usize, usize) + Sync + 'a;

/// A watch beside one running slice: [`Self::every`] calls its function each
/// period until the slice ends ([`SliceWatchEnd`]).
#[derive(Default)]
struct SliceWatch {
    over: Mutex<bool>,
    ended: std::sync::Condvar,
}

impl SliceWatch {
    fn every(&self, period: f64, mut f: impl FnMut()) {
        let period = std::time::Duration::from_secs_f64(period);
        loop {
            let over = self.over.lock().expect("the watch's lock");
            if *over {
                return;
            }
            let (over, _) = self
                .ended
                .wait_timeout(over, period)
                .expect("the watch's lock");
            if *over {
                return;
            }
            drop(over);
            f();
        }
    }
}

/// Ends a [`SliceWatch`] when dropped, so a slice that unwinds still lets
/// its watch, and the scope holding both, end.
struct SliceWatchEnd<'a>(&'a SliceWatch);

impl Drop for SliceWatchEnd<'_> {
    fn drop(&mut self) {
        // A poisoned lock still holds the flag; set it either way.
        let mut over = self.0.over.lock().unwrap_or_else(|e| e.into_inner());
        *over = true;
        self.0.ended.notify_all();
    }
}

/// Solve the gap between `a` and `b` (tie weight `lambda`) with `chain`, its
/// cut's scalar chain at that weight, or with no cut when `chain` is `None`:
/// as one box, or as consecutive energy slices, as many as `slicing` counts.
///
/// Slice `i` covers energies up to `e_i` (evenly spaced from `E_a` to
/// `E_b`) with the floor at the best adaptation the slices below it found,
/// `a`'s for the first, and the same cut weight. It is exact by dominance: a
/// Pareto design with `e_{i-1} < E ≤ e_i` is better adapted than every design
/// at lower energy, so it clears slice `i`'s floor and lies in slice `i`. The
/// slices below found every such lower design, by induction. A slice may
/// re-return the best design of the slice below, which is kept once.
/// Each slice runs on `workers` and the `lend()` count beside them, the
/// loan held for that slice alone. With `speculate`, a loan of `k` workers
/// runs the next `k + 1` slices at once instead, each with the floor known
/// when they start and a share of the workers: a lower floor only widens a
/// slice's box, which then still holds every front point of its energies,
/// and every point it returns is Pareto among the designs under its ceiling,
/// so sweeping the union for dominance gives the same front.
/// Before each slice, a cut gap asks `own` (with the last solve's clock,
/// `None` before the first, the slice's number, the slices left and the
/// workers it now holds) for a chain
/// at its weight to solve the rest under instead, once: a slice is exact
/// under any cut at its weight, so changing cut between slices keeps the
/// front. With `watch`, a cut gap that has not changed cut also calls its
/// function every period while a slice solves alone, with the slice's CPU so
/// far, the slices after it, the next slice's number and the workers it
/// holds, so the next chain can
/// be readied while the slice runs.
/// Returns the gap's front, its clock (the cut's chain once, the slices'
/// folds summed), the slice count, and `None`; or, when `stop` says so before
/// a slice, the front found so far with the gap's rest `(from, floor)`: every
/// front point up to energy `from` is in it and `floor` is the best adaptation
/// there. Given a `rest`, the gap is solved from there alone: its slices cover
/// the energies above `from`, with `floor` as the first slice's.
/// With `fold_stop`, a slice whose fold it ends is halved and its lower half
/// solved first, unless `stop` then says the walk is out of budget, or the
/// slice is one energy unit wide: either ends the gap at its rest. Any halving
/// is exact, by the same dominance argument as the slicing itself.
/// Where a gap's solve starts (the rest an earlier walk left, if any), whether
/// to stop before its next slice, and what to call once a slice is solved.
type GapControl<'a> = (Option<(Energy, f64)>, &'a dyn Fn() -> bool, &'a dyn Fn());

#[allow(clippy::too_many_arguments)]
fn solve_gap<L>(
    params: &Params,
    graph: &dfa::Dfa,
    side: &EnergySide,
    (a, b, lambda): (HullPoint, HullPoint, f64),
    (rest, stop, solved_one): GapControl<'_>,
    fold_stop: Option<FoldStop>,
    mut chain: Option<CutChain>,
    extra: Vec<CutChain>,
    (slicing, floor_cut): (Slicing, bool),
    workers: usize,
    speculate: bool,
    lend: impl Fn() -> (usize, L),
    own: impl Fn(Option<&SliceClock>, usize, usize, usize) -> Result<Option<CutChain>, BeamError>,
    watch: Option<(f64, &SliceAsk<'_>)>,
) -> Result<GapSolve, BeamError> {
    let e_b = b.energy().0;
    // A gap continued from its rest is the box above the rest's energy.
    let e_a = rest.map_or(a.energy().0, |(from, _)| from.0);
    let mut slices = slicing.count(lambda, e_a, e_b);
    // Halving needs one slice at a time.
    let speculate = speculate && fold_stop.is_none();
    let mut clock = SolveClock::default();
    let mut halved = 0;
    let mut front: Vec<FrontPoint> = Vec::new();
    let mut floor = rest.map_or(a.adaptation_log, |(_, floor)| floor);
    // Extra cuts ride only beside the gap's own: each is an exact pass at its
    // own weight, cut at the box corner's score there.
    let lambdas: Vec<f64> = if chain.is_some() {
        std::iter::once(lambda)
            .chain(extra.iter().map(|c| c.clock.lambda))
            .collect()
    } else {
        Vec::new()
    };
    let copy = |c: &CutChain| CutChain {
        outside: Arc::clone(&c.outside),
        clock: c.clock,
    };
    // Slice `i`'s ceiling is `tops[i - 1]`: `slices` equal parts of the gap,
    // the last at `e_b`.
    let mut tops: Vec<i32> = (1..=slices)
        .map(|k| {
            if k == slices {
                e_b
            } else {
                e_a + ((i64::from(e_b - e_a) * k as i64) / slices as i64) as i32
            }
        })
        .collect();
    let solve_slice = |top: i32, floor: f64, threads: usize, own: Option<CutChain>| {
        let bounds = Bounds::with_ceiling(Ceiling::Absolute(Energy(top))).with_exact_credit();
        let bounds = if floor_cut {
            bounds.with_floor_cut()
        } else {
            bounds
        };
        let chains: Vec<CutChain> = match own {
            Some(own) => std::iter::once(own).chain(extra.iter().map(copy)).collect(),
            None => Vec::new(),
        };
        let band = BoxSpec {
            cai_floor_log: floor,
            lambdas: lambdas.clone(),
        };
        solve_stoppable(
            params,
            graph,
            UtrProfile::inert(),
            side,
            chains,
            Prune::None,
            bounds,
            threads,
            Some(&band),
            fold_stop.clone(),
        )
    };
    let mut i = 1;
    while i <= slices {
        if stop() {
            // Solved up to the top of the slice before this one: one unit
            // below the gap's stable end, before any slice, so that a resume
            // folds the whole gap.
            let from = match (i, rest) {
                (1, Some((from, _))) => from.0,
                (1, None) => e_a - 1,
                _ => tops[i - 2],
            };
            clock.halved = halved;
            return Ok(GapSolve {
                front,
                clock,
                slices,
                rest: Some((Energy(from), floor)),
            });
        }
        let (lent, loan) = lend();
        if clock.own_from.is_none() {
            let started = std::time::Instant::now();
            if let Some(cut) = chain
                .as_ref()
                .map(|_| own(clock.slices.last(), i, slices - i + 1, workers + lent))
                .transpose()?
                .flatten()
            {
                chain = Some(cut);
                clock.own_from = Some((i, started.elapsed().as_secs_f64()));
            }
        }
        let batch = if speculate && lent > 0 {
            (lent + 1).min(slices - i + 1)
        } else {
            1
        };
        let at = i;
        let started = std::time::Instant::now();
        let cpu = task_clock::TaskClock::start();
        let solved: Vec<(Bounded<BoundedFront>, SolveClock)> = if batch == 1 {
            let this = if at == slices {
                chain.take()
            } else {
                chain.as_ref().map(copy)
            };
            // The watch reports the solving thread's CPU time from a thread
            // of its own, so it needs a clock naming another thread; where
            // there is none the reading would be the watcher's, which is
            // nearly zero, so the watch does not run at all rather than
            // report that. See `ThreadCpu::AVAILABLE`.
            let watched = watch.filter(|_| {
                this.is_some() && clock.own_from.is_none() && task_clock::ThreadCpu::AVAILABLE
            });
            match watched {
                Some((period, ask)) => {
                    let cpu = task_clock::ThreadCpu::here();
                    let watch = SliceWatch::default();
                    let after = slices - at;
                    let now = workers + lent;
                    std::thread::scope(|scope| {
                        scope.spawn(|| {
                            watch.every(period, || ask(cpu.seconds(), after, at + 1, now));
                        });
                        // Ends the watch on return or unwind alike.
                        let _end = SliceWatchEnd(&watch);
                        solve_slice(tops[at - 1], floor, now, this)
                    })
                    .map(|solved| vec![solved])?
                }
                None => vec![solve_slice(tops[at - 1], floor, workers + lent, this)?],
            }
        } else {
            // Speculative: the next `batch` slices at once, each with the
            // floor known now, on a share of the workers. A lower floor
            // only widens a slice's box, so each still holds every front
            // point of its energies; the sweep below keeps the front.
            let each = ((workers + lent) / batch).max(1);
            let owns: Vec<Option<CutChain>> =
                (0..batch).map(|_| chain.as_ref().map(copy)).collect();
            std::thread::scope(|scope| {
                let handles: Vec<_> = owns
                    .into_iter()
                    .enumerate()
                    .map(|(k, own)| {
                        let solve_slice = &solve_slice;
                        let top = tops[i + k - 1];
                        task_clock::spawn(scope, move || solve_slice(top, floor, each, own))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("a speculative slice panicked"))
                    .collect::<Result<Vec<_>, BeamError>>()
            })?
        };
        drop(loan);
        if solved.iter().any(|(_, c)| c.fold.stopped) {
            // The slice's fold stopped: halve it, unless it cannot be or the
            // walk is out of budget once its tables are freed, which ends the
            // gap at the top of the slice below.
            let below = if i == 1 { e_a } else { tops[i - 2] };
            let mid = below + (tops[i - 1] - below) / 2;
            clock.fold_s += solved.iter().map(|(_, c)| c.fold_s).sum::<f64>();
            if mid > below && !stop() {
                tops.insert(i - 1, mid);
                slices += 1;
                halved += 1;
                continue;
            }
            let from = match (i, rest) {
                (1, Some((from, _))) => from.0,
                (1, None) => e_a - 1,
                _ => tops[i - 2],
            };
            clock.halved = halved;
            return Ok(GapSolve {
                front,
                clock,
                slices,
                rest: Some((Energy(from), floor)),
            });
        }
        let mut span = SliceClock {
            top_kcal_mol: f64::from(tops[i + batch - 2]) / 100.0,
            count: batch,
            wall_s: started.elapsed().as_secs_f64(),
            cpu_s: cpu.finish(),
            workers: workers + lent,
            points: 0,
        };
        for (k, (solved, slice_clock)) in solved.into_iter().enumerate() {
            if i + k == 1 {
                let own_from = clock.own_from;
                clock = slice_clock.clone();
                clock.fold_s = 0.0;
                clock.own_from = own_from;
            } else {
                clock.fold.absorb(&slice_clock.fold);
            }
            clock.fold_s += slice_clock.fold_s;
            if let Bounded::Feasible(found) = solved {
                span.points += found.front.len();
                // A continued gap's first slice reaches below its rest, where
                // the earlier walk found every front point already.
                let above = |p: &FrontPoint| rest.is_none_or(|(from, _)| energy_of(p).0 > from.0);
                for p in found.front.into_iter().filter(above) {
                    let seen = front.last().is_some_and(|q| {
                        q.mfe_kcal_mol == p.mfe_kcal_mol && q.cai_log == p.cai_log
                    });
                    if !seen {
                        front.push(p);
                    }
                }
            }
        }
        clock.slices.push(span);
        for _ in 0..batch {
            solved_one();
        }
        if batch > 1 {
            front = pareto_sweep(front);
        }
        if let Some(best) = front
            .iter()
            .map(|p| p.cai_log + p.utr_profile_log)
            .max_by(f64::total_cmp)
        {
            floor = floor.max(best);
        }
        i += batch;
    }
    clock.halved = halved;
    Ok(GapSolve {
        front,
        clock,
        slices,
        rest: None,
    })
}

/// What [`solve_gap`] returns: the gap's front, its clock, its slice count,
/// and the rest it stopped at, if it stopped.
struct GapSolve {
    front: Vec<FrontPoint>,
    clock: SolveClock,
    slices: usize,
    rest: Option<(Energy, f64)>,
}

/// `points` reduced to their Pareto front, by ascending energy: a point is
/// kept when it is better adapted than every point at lower or equal energy,
/// one copy of each.
fn pareto_sweep(mut points: Vec<FrontPoint>) -> Vec<FrontPoint> {
    let adapted = |p: &FrontPoint| p.cai_log + p.utr_profile_log;
    points.sort_by(|p, q| {
        p.mfe_kcal_mol
            .total_cmp(&q.mfe_kcal_mol)
            .then_with(|| adapted(q).total_cmp(&adapted(p)))
    });
    let mut best = f64::NEG_INFINITY;
    points.retain(|p| {
        let keep = adapted(p) > best;
        best = best.max(adapted(p));
        keep
    });
    points
}

/// One gap of a walk, as its report names it: the hull points it lies
/// between, its tie weight, and the cut it read.
#[derive(Debug, Clone, PartialEq)]
pub struct GapSpec {
    /// The gap's more stable hull point.
    pub a: HullPoint,
    /// Its more adapted hull point.
    pub b: HullPoint,
    /// The weight the gap is solved at ([`tie_weight`] of `a` and `b`).
    pub lambda: f64,
    /// The weights of the pooled passes whose chord cut it; `None` for a gap
    /// cut by its own pass at `lambda`.
    pub chord: Option<(f64, f64)>,
    /// The weights of the further pooled passes it was cut at
    /// ([`WalkOptions::bracket_cuts`]); in a walk, the passes nearest each
    /// side of `lambda`.
    pub bracket: Vec<f64>,
    /// The slice (1-based) from which a chord-cut gap is solved under an
    /// exact pass at `lambda` instead, as a walk's fold changes cut part way
    /// ([`WalkOptions::own_when_alone`]); `None` keeps the chord throughout.
    pub own_from: Option<usize>,
}

/// A gap solved on its own by [`replay_gap`].
#[derive(Debug, Clone)]
pub struct ReplayedGap {
    /// The gap's Pareto points, `a` and `b` included when they are front points.
    pub front: Vec<FrontPoint>,
    /// The solve's clock, cut and fold.
    pub clock: SolveClock,
    /// How many energy slices it was solved in.
    pub slices: usize,
    /// Seconds to build the credit.
    pub credit_s: f64,
    /// Seconds for the cut passes, and the chord where one was read.
    pub passes_s: f64,
    /// CPU seconds of the gap's slices alone, every thread they ran on.
    pub fold_cpu_s: f64,
}

/// Solve one gap of a coding-only walk alone, cut as the walk cut it: the
/// credit, then an exact pass at each of the cut's weights, then the gap's
/// box. The points are the walk's for that gap; the clock is the gap's own
/// cost, with nothing of the walk's sweep around it.
pub fn replay_gap(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    spec: &GapSpec,
    slicing: Slicing,
    threads: usize,
) -> Result<ReplayedGap, BeamError> {
    replay_gap_lending(
        params,
        protein,
        usage,
        spec,
        slicing,
        threads,
        false,
        || (0, ()),
    )
}

/// [`replay_gap`] with a fixed loan of `loan` workers at every slice spent on
/// running `loan + 1` slices at once ([`WalkOptions::speculative_slices`]),
/// so the cost of slices run with a lower floor can be measured on one gap.
pub fn replay_gap_speculating(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    spec: &GapSpec,
    slicing: Slicing,
    threads: usize,
    loan: usize,
) -> Result<ReplayedGap, BeamError> {
    replay_gap_lending(params, protein, usage, spec, slicing, threads, true, || {
        (loan, ())
    })
}

/// [`replay_gap`] with each slice also run on the `lend()` count of workers,
/// as a walk's fold borrows idle ones ([`WalkOptions::lend_idle_workers`]),
/// and with `speculate` a loan spent on running slices at once
/// ([`WalkOptions::speculative_slices`]).
#[allow(clippy::too_many_arguments)]
fn replay_gap_lending<L>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    spec: &GapSpec,
    slicing: Slicing,
    threads: usize,
    speculate: bool,
    lend: impl Fn() -> (usize, L),
) -> Result<ReplayedGap, BeamError> {
    let aa = residues(protein)?;
    let (utr5, utr3) = (UtrSpec::Fixed(String::new()), UtrSpec::Fixed(String::new()));
    let constraints = RegionConstraints::enforcing(&[], Enforcement::Forbid);
    let profile = UtrProfile::inert();
    let graph = dfa::build_with_utr_profile(
        &aa,
        usage,
        utr5.parse()?.chain(),
        utr3.parse()?.chain(),
        profile,
    );
    let region = region::PairConstraint::new(&constraints, &graph);
    let corner =
        || achievable_minimum_with_utr(params, protein, usage, &utr5, &utr3, profile, constraints);
    let t = std::time::Instant::now();
    let mut side = EnergySide::new(params, &graph, corner, true, threads, region.clone())?;
    side.build_credit(params, &graph, threads);
    let credit_s = t.elapsed().as_secs_f64();

    let t = std::time::Instant::now();
    let config = credit_config(threads);
    let mut passes: Vec<(f64, Arc<cost_to_go::Outside>)> = Vec::new();
    let mut pass_at = |w: f64| -> Arc<cost_to_go::Outside> {
        if let Some((_, pass)) = passes.iter().find(|(seen, _)| *seen == w) {
            return Arc::clone(pass);
        }
        let tables = scan::scan_with_region(params, &graph, w, config, region.clone());
        let mut pass = cost_to_go::Outside::compute_at_any_lambda(params, &graph, &tables, config);
        if let Some((_, first)) = passes.first() {
            pass.share_keys(first);
        }
        let pass = Arc::new(pass);
        passes.push((w, Arc::clone(&pass)));
        pass
    };
    let outside = match spec.chord {
        None => pass_at(spec.lambda),
        Some((lo, hi)) => {
            let ends = ChordEnds {
                lo: (lo, pass_at(lo)),
                hi: (hi, pass_at(hi)),
            };
            Arc::new(ends.make(spec.lambda))
        }
    };
    let own = CutChain {
        outside,
        clock: CutClock {
            lambda: spec.lambda,
            ..CutClock::default()
        },
    };
    let extra: Vec<CutChain> = spec
        .bracket
        .iter()
        .map(|&w| CutChain {
            outside: pass_at(w),
            clock: CutClock {
                lambda: w,
                ..CutClock::default()
            },
        })
        .collect();
    let passes_s = t.elapsed().as_secs_f64();
    let first = passes.first().map(|(_, pass)| Arc::clone(pass));
    let switch = |_: Option<&SliceClock>, at: usize, _: usize, workers: usize| {
        if spec.chord.is_none() || spec.own_from != Some(at) {
            return Ok(None);
        }
        let config = credit_config(workers);
        let tables = scan::scan_with_region(params, &graph, spec.lambda, config, region.clone());
        let mut pass = cost_to_go::Outside::compute_at_any_lambda(params, &graph, &tables, config);
        if let Some(first) = &first {
            pass.share_keys(first);
        }
        Ok(Some(CutChain {
            outside: Arc::new(pass),
            clock: CutClock {
                lambda: spec.lambda,
                ..CutClock::default()
            },
        }))
    };

    let cpu = task_clock::TaskClock::start();
    let GapSolve {
        front,
        mut clock,
        slices,
        ..
    } = solve_gap(
        params,
        &graph,
        &side,
        (spec.a, spec.b, spec.lambda),
        (None, &|| false, &|| {}),
        None,
        Some(own),
        extra,
        (slicing, false),
        threads,
        speculate,
        lend,
        switch,
        None,
    )?;
    clock.own_asked = clock.own_from.map(|(slice, _)| slice);
    Ok(ReplayedGap {
        front,
        clock,
        slices,
        credit_s,
        passes_s,
        fold_cpu_s: cpu.finish(),
    })
}

/// A hull vertex as one exact scalar fold at `lambda` reports it, beside the
/// fold itself. The root state is the fold's optimum, and its accumulators are
/// the design's objectives, reported as [`finish`] reports them.
fn hull_vertex(
    params: &Params,
    graph: &dfa::Dfa,
    lambda: f64,
    config: ScanConfig,
    region: &region::PairConstraint,
) -> Result<(HullPoint, tables::Tables), BeamError> {
    let tables = walk_scan(params, graph, lambda, config, region);
    let best = tables.c[graph.ix(graph.end())].ok_or(BeamError::AchievableMinimumDisagrees {
        corner: Energy(0),
        folded: None,
    })?;
    let point = HullPoint {
        mfe_kcal_mol: best.mfe.to_kcal_mol(),
        adaptation_log: best.cai + graph.codon_offset(),
    };
    Ok((point, tables))
}

/// One exact scan of a flanked molecule as the walk runs it ([`walk_scan`],
/// under `constraints`), with its lattice: for a probe that measures what a
/// walk's scan holds. Not part of the walk.
///
/// # Errors
///
/// As [`design_front_walk_with_utr`] for a malformed protein or flank.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn walk_scan_for_census(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    constraints: RegionConstraints,
    lambda: f64,
    threads: usize,
) -> Result<(dfa::Dfa, tables::Tables), BeamError> {
    let aa = residues(protein)?;
    let five = utr5.parse()?;
    let three = utr3.parse()?;
    let graph =
        dfa::build_with_utr_profile(&aa, usage, five.chain(), three.chain(), UtrProfile::inert());
    let region = region::PairConstraint::new(&constraints, &graph);
    let tables = walk_scan(params, &graph, lambda, credit_config(threads), &region);
    Ok((graph, tables))
}

/// An exact scan as the walk uses one: its optimum is a hull vertex, and its
/// tables feed at most an outside pass, so no traceback reads them. They keep
/// values alone ([`scan::scan_values_with_constraints`] with no required
/// pair, which fills the same values as [`scan::scan_with_region`]): a
/// multi-branch cell's back-pointer is the largest part of a scan's tables,
/// and every worker's scan holds its own.
fn walk_scan(
    params: &Params,
    graph: &dfa::Dfa,
    lambda: f64,
    config: ScanConfig,
    region: &region::PairConstraint,
) -> tables::Tables {
    scan::scan_values_with_constraints(
        params,
        graph,
        lambda,
        config,
        region.clone(),
        must_pair::MustPairConstraint::inert(),
    )
}

/// How a certified gap will get its cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Plan {
    /// Its own backward pass, over the scan that certified it.
    Own,
    /// The chord between two passes in the pool.
    Chord,
    /// No cut.
    Uncut,
}

/// How a certified gap gets its cut: uncut when its ceiling lies within
/// [`WalkOptions::uncut_within`] of `optimum`; with
/// [`WalkOptions::chord_bracket`], the chord between pooled passes when two
/// bracket its weight within the width; otherwise its own pass. The pool
/// only grows, so a bracket found here is still there, or a tighter one, when
/// the gap's fold reads it.
fn plan_cut(
    b: HullPoint,
    lambda: f64,
    optimum: Energy,
    options: &WalkOptions,
    pool: &Mutex<PassPool>,
) -> Plan {
    if options
        .uncut_within
        .is_some_and(|within| b.energy().0 - optimum.0 <= within)
    {
        return Plan::Uncut;
    }
    let bracketed = options.chord_bracket.is_some_and(|width| {
        pool.lock()
            .expect("the pool's lock")
            .around(lambda, width)
            .is_some()
    });
    if bracketed {
        Plan::Chord
    } else {
        Plan::Own
    }
}

/// The exact cut passes a walk has computed, by ascending weight, which later
/// gaps bound their own cost to go from.
#[derive(Default)]
struct PassPool {
    passes: Vec<(f64, Arc<cost_to_go::Outside>)>,
}

/// The pooled passes a gap's chord is taken between.
struct ChordEnds {
    lo: (f64, Arc<cost_to_go::Outside>),
    hi: (f64, Arc<cost_to_go::Outside>),
}

impl ChordEnds {
    /// [`Self::make`], read cell by cell rather than built whole.
    fn bound(&self, lambda: f64) -> cost_to_go::CutBound {
        let ((lo, a), (hi, b)) = (&self.lo, &self.hi);
        if lo == hi {
            cost_to_go::CutBound::Pass(Arc::clone(a))
        } else {
            cost_to_go::CutBound::Chord {
                lo: Arc::clone(a),
                hi: Arc::clone(b),
                t: (hi - lambda) / (hi - lo),
            }
        }
    }

    /// The chord at `lambda`; a pass at `lambda` itself is that pass.
    fn make(&self, lambda: f64) -> cost_to_go::Outside {
        let ((lo, a), (hi, b)) = (&self.lo, &self.hi);
        if lo == hi {
            (**a).clone()
        } else {
            cost_to_go::Outside::chord(a, b, (hi - lambda) / (hi - lo))
        }
    }
}

impl PassPool {
    /// A pooled pass whose keys a new one can share, if any is pooled.
    fn first(&self) -> Option<Arc<cost_to_go::Outside>> {
        self.passes.first().map(|(_, pass)| Arc::clone(pass))
    }

    fn insert(&mut self, lambda: f64, outside: Arc<cost_to_go::Outside>) {
        let i = self.passes.partition_point(|(w, _)| *w < lambda);
        self.passes.insert(i, (lambda, outside));
    }

    /// Up to `reach` pooled passes strictly below `lambda` and as many strictly
    /// above it, nearest first.
    fn neighbours(&self, lambda: f64, reach: usize) -> Vec<(f64, Arc<cost_to_go::Outside>)> {
        let below = self.passes.partition_point(|(w, _)| *w < lambda);
        let above = self.passes.partition_point(|(w, _)| *w <= lambda);
        let end = (above + reach).min(self.passes.len());
        self.passes[below.saturating_sub(reach)..below]
            .iter()
            .rev()
            .chain(&self.passes[above..end])
            .map(|(w, pass)| (*w, Arc::clone(pass)))
            .collect()
    }

    /// The nearest passes at or below and at or above `lambda`, when they lie
    /// within `width` of each other, with the cut that reads their chord.
    fn around(&self, lambda: f64, width: f64) -> Option<(ChordEnds, GapCut)> {
        let i = self.passes.partition_point(|(w, _)| *w < lambda);
        let hi = self.passes.get(i)?;
        let lo = if hi.0 == lambda {
            hi
        } else {
            self.passes.get(i.checked_sub(1)?)?
        };
        if hi.0 - lo.0 > width {
            return None;
        }
        let cut = GapCut::Chord { lo: lo.0, hi: hi.0 };
        let ends = ChordEnds {
            lo: (lo.0, Arc::clone(&lo.1)),
            hi: (hi.0, Arc::clone(&hi.1)),
        };
        Some((ends, cut))
    }
}

/// The most grid passes the plain walk takes in one traversal of the lattice
/// ([`outside_multi::scan_lanes`]) without a memory budget. On a 163-residue
/// walk two passes to a traversal took 0.66 of the instructions of the walk
/// taking each alone, for a peak of 450 MB against 387 MB; four took 0.51
/// for 673 MB, the lanes' tables being held at once. Resident memory, not
/// instructions, is what bounds a long walk, so two is the default and a
/// development build can give more.
const GRID_LANES: usize = 2;

/// The lane count a development build gives, from 1 (each pass alone) to
/// [`outside_multi::MAX_LANES`], with or without a memory budget. A value
/// that is not a count is refused rather than read as unset, since a walk
/// that silently took the default would cost what the count was set to save.
fn grid_lanes() -> Option<usize> {
    let given = crate::dev_env::var(crate::dev_env::Setting::GridLanes).ok()?;
    let n: usize = given.trim().parse().unwrap_or_else(|_| {
        panic!(
            "{:?}={given:?} is not a lane count",
            crate::dev_env::Setting::GridLanes
        )
    });
    Some(n.clamp(1, outside_multi::MAX_LANES))
}

/// The grid weights whose passes are taken several to a traversal, each with
/// a slot that holds its optimum once its pass is pooled.
struct GridLanes {
    weights: Vec<f64>,
    slots: Vec<Mutex<Option<Option<HullPoint>>>>,
    /// The most passes in one traversal.
    lanes: usize,
}

impl GridLanes {
    /// Lanes over `weights`, or `None` where they take one pass at a time.
    fn new(weights: &[f64], lanes: usize) -> Option<Self> {
        (lanes > 1 && weights.len() > 1).then(|| Self {
            weights: weights.to_vec(),
            slots: weights.iter().map(|_| Mutex::new(None)).collect(),
            lanes,
        })
    }

    /// The index of `lambda` among the lane weights.
    fn slot(&self, lambda: f64) -> Option<usize> {
        self.weights.iter().position(|&w| w == lambda)
    }
}

/// The best design a grid pass's tables hold, on the walk's axes.
/// The optimum at each stop's terminal of a stop-labelled lattice, in
/// [`dfa::STOP_CODON_NUCS`] order; all `None` on any other lattice.
fn stop_optima(graph: &dfa::Dfa, tables: &tables::Tables) -> [Option<HullPoint>; dfa::STOP_LANES] {
    let mut out = [None; dfa::STOP_LANES];
    if graph.stop_exit().is_some() {
        for &end in graph.nodes_at(graph.seq_len) {
            out[usize::from(end.lane)] = tables.c[graph.ix(end)].map(|best| HullPoint {
                mfe_kcal_mol: best.mfe.to_kcal_mol(),
                adaptation_log: best.cai + graph.codon_offset(),
            });
        }
    }
    out
}

/// The least energy at each stop's terminal of a stop-labelled lattice, in
/// [`dfa::STOP_CODON_NUCS`] order; all `None` on any other lattice.
fn stop_energies(graph: &dfa::Dfa, tables: &tables::Tables) -> [Option<Energy>; dfa::STOP_LANES] {
    let mut out = [None; dfa::STOP_LANES];
    if graph.stop_exit().is_some() {
        for &end in graph.nodes_at(graph.seq_len) {
            out[usize::from(end.lane)] = tables.c[graph.ix(end)].map(|best| best.mfe);
        }
    }
    out
}

/// Grid passes shared by the walks at each stop of one stop-labelled
/// lattice: a pass completes to every stop there
/// ([`dfa::Dfa::completion_ends`]), so it is the same pass at every stop and
/// is taken once, with every stop's optimum at its weight.
/// One grid pass and, beside it, each stop's optimum at that pass's weight:
/// what a walk needs from a weight another stop's walk has already taken.
type GridPass = (cost_to_go::Outside, [Option<HullPoint>; dfa::STOP_LANES]);

/// The credit pass and each stop's least energy, the `λ = 1` counterpart of
/// [`GridPass`], whose optimum is an energy rather than a hull point.
type CreditPass = (cost_to_go::Outside, [Option<Energy>; dfa::STOP_LANES]);

#[derive(Default)]
struct SharedPasses {
    /// The credit's `λ = 1` pass, with every stop's least energy.
    credit: Mutex<Option<CreditPass>>,
    by_weight: Mutex<crate::hash::FxHashMap<u64, GridPass>>,
}

impl SharedPasses {
    /// The energy side of `graph`'s stop built from the shared credit pass,
    /// when one is taken and the stop has designs.
    fn credit(&self, graph: &dfa::Dfa, region: &region::PairConstraint) -> Option<EnergySide> {
        let k = usize::from(graph.terminal_stop()?);
        let held = self.credit.lock().expect("the shared credit's lock");
        let (outside, optima) = held.as_ref()?;
        let optimum = optima[k]?;
        Some(EnergySide {
            optimum,
            region: region.clone(),
            fold: None,
            credit: Some(bounds::ExactCredit::new(outside.clone(), optimum)),
            scan_s: 0.0,
            pass_s: 0.0,
            scalar_s: None,
        })
    }

    fn put_credit(&self, outside: &cost_to_go::Outside, optima: [Option<Energy>; dfa::STOP_LANES]) {
        let mut held = self.credit.lock().expect("the shared credit's lock");
        if held.is_none() {
            *held = Some((outside.clone(), optima));
        }
    }

    fn has(&self, lambda: f64) -> bool {
        self.by_weight
            .lock()
            .expect("the shared passes' lock")
            .contains_key(&lambda.to_bits())
    }

    /// The pass at `lambda` and `graph`'s stop's optimum there, when taken.
    fn get(
        &self,
        lambda: f64,
        graph: &dfa::Dfa,
    ) -> Option<(cost_to_go::Outside, Option<HullPoint>)> {
        let k = usize::from(graph.terminal_stop().expect("a lattice read at one stop"));
        let map = self.by_weight.lock().expect("the shared passes' lock");
        map.get(&lambda.to_bits())
            .map(|(outside, optima)| (outside.clone(), optima[k]))
    }

    fn put(
        &self,
        lambda: f64,
        outside: &cost_to_go::Outside,
        graph: &dfa::Dfa,
        tables: &tables::Tables,
    ) {
        self.put_optima(lambda, outside, &stop_optima(graph, tables));
    }

    fn put_optima(
        &self,
        lambda: f64,
        outside: &cost_to_go::Outside,
        optima: &[Option<HullPoint>; dfa::STOP_LANES],
    ) {
        self.by_weight
            .lock()
            .expect("the shared passes' lock")
            .entry(lambda.to_bits())
            .or_insert_with(|| (outside.clone(), *optima));
    }
}

fn grid_optimum(graph: &dfa::Dfa, tables: &tables::Tables) -> Option<HullPoint> {
    tables.c[graph.ix(graph.end())].map(|best| HullPoint {
        mfe_kcal_mol: best.mfe.to_kcal_mol(),
        adaptation_log: best.cai + graph.codon_offset(),
    })
}

/// Store `outside`'s keyed tables in the slots of a pass the pool already
/// holds, when the two hold the same keys, so the pool keeps one copy of the
/// keys however many passes it holds. Read outside the pool's lock: the
/// comparison visits every key.
fn share_pooled_keys(pool: &Mutex<PassPool>, outside: &mut cost_to_go::Outside) {
    let first = pool.lock().expect("the pool's lock").first();
    if let Some(first) = first {
        outside.share_keys(&first);
    }
}

/// Set the walk's credit, which is built once.
fn set_once(side: &OnceLock<EnergySide>, built: EnergySide) {
    assert!(side.set(built).is_ok(), "the walk's credit is built once");
}

/// The walk's credit, which every pair waits for.
fn credit_of(side: &OnceLock<EnergySide>) -> &EnergySide {
    side.get()
        .expect("no pair is handed out before the credit stands")
}

/// A certified gap's priority in the walk's queue, from what is known when it
/// is queued: its energy width `ΔE` (the fold's integer units) and tie weight.
///
/// Below every probe, by slack `λ*·ΔE`, or by `ΔE` alone with
/// [`WalkOptions::fold_by_width`]; above the probes, by `ΔE`, when `ΔE`
/// exceeds [`WalkOptions::heavy_fold`]. Ties run in queueing order.
fn fold_priority(a: HullPoint, b: HullPoint, lambda: f64, options: &WalkOptions) -> (u8, u64) {
    let width = b.energy().0 - a.energy().0;
    let wide = u64::try_from(width).unwrap_or(0);
    match options.heavy_fold {
        Some(most) if width > most => (PROBE.0, wide),
        _ if options.fold_by_width => (0, wide),
        _ => (0, (lambda * f64::from(width)) as u64),
    }
}

/// A certified gap waiting for its fold: its own pass when it took one.
struct GapJob {
    a: HullPoint,
    b: HullPoint,
    lambda: f64,
    plan: Plan,
    outside: Option<Arc<cost_to_go::Outside>>,
    clock: CutClock,
    /// When it was queued, in seconds from the sweep's start.
    queued: f64,
    /// The watched bracket of the probe that certified it.
    bracket: Option<usize>,
    /// Where an earlier walk stopped the gap, `(from, floor)`.
    rest: Option<(Energy, f64)>,
}

/// The probe of the pair whose more stable end is `a`, at their tie weight
/// `lambda`, cut at their tie value with the chord of `ends` as the cost to
/// go: the fold's optimum when it scores at most the tie value, which is the
/// only case in which the walk reads it; otherwise whatever the cut left, or
/// `None`.
fn probe_cut(
    params: &Params,
    graph: &dfa::Dfa,
    a: HullPoint,
    lambda: f64,
    ends: &ChordEnds,
    workers: usize,
    region: &region::PairConstraint,
) -> Option<HullPoint> {
    probe_cut_at(params, graph, a, lambda, ends, 0.0, workers, region)
}

/// [`probe_cut`] with the threshold moved by `offset`.
#[allow(clippy::too_many_arguments)]
fn probe_cut_at(
    params: &Params,
    graph: &dfa::Dfa,
    a: HullPoint,
    lambda: f64,
    ends: &ChordEnds,
    offset: f64,
    workers: usize,
    region: &region::PairConstraint,
) -> Option<HullPoint> {
    // The tie value on the fold's own adaptation scale, which the lattice's
    // codon offset separates from the reported one.
    let threshold = sweep_score(a, lambda) + 100.0 * (1.0 - lambda) * graph.codon_offset();
    let cut = tables::ScanCut {
        bound: ends.bound(lambda),
        threshold: threshold + offset,
    };
    let tables = scan::scan_cut(
        params,
        graph,
        lambda,
        credit_config(workers),
        cut,
        region.clone(),
    );
    tables.c[graph.ix(graph.end())].map(|best| HullPoint {
        mfe_kcal_mol: best.mfe.to_kcal_mol(),
        adaptation_log: best.cai + graph.codon_offset(),
    })
}

/// One scan of `graph` at `lambda` cut at `threshold` with `bound` as the cost
/// to go (see [`tables::ScanCut`]), timed: the best score it reaches at the
/// exterior, if any, its seconds and the cells it kept per table.
#[doc(hidden)]
pub fn timed_cut_scan(
    params: &Params,
    graph: &dfa::Dfa,
    lambda: f64,
    threads: usize,
    bound: cost_to_go::Outside,
    threshold: f64,
) -> (Option<f64>, f64, tables::CellCounts) {
    let t = std::time::Instant::now();
    let cut = tables::ScanCut {
        bound: cost_to_go::CutBound::Pass(Arc::new(bound)),
        threshold,
    };
    let tables = scan::scan_cut(
        params,
        graph,
        lambda,
        credit_config(threads),
        cut,
        region::PairConstraint::inert(),
    );
    let seconds = t.elapsed().as_secs_f64();
    let best = tables.c[graph.ix(graph.end())].map(|s| s.brd(lambda));
    (best, seconds, tables.cell_counts())
}

/// How far below a pair's tie value a point must score to split the pair.
const TOL: f64 = 1e-6;

/// The vertices of the lower hull of `points`, most stable first: each point
/// scoring below the chord of its neighbouring vertices at their tie weight by
/// [`TOL`], as a probe at that weight would split them. A point on a chord is
/// left out, since a sweep reaches it only at the weight where it ties an end,
/// and there the sweep returns that end.
fn lower_hull(points: impl Iterator<Item = HullPoint>) -> Vec<HullPoint> {
    let mut hull: Vec<HullPoint> = Vec::new();
    for p in points {
        while let [.., a, b] = hull[..] {
            if tie_weight(a, p).is_some_and(|l| sweep_score(b, l) < sweep_score(a, l) - TOL) {
                break;
            }
            hull.pop();
        }
        hull.push(p);
    }
    hull
}

/// `vertices` with the hull vertices inside each of `gaps` that no probe
/// found: a pair folded whole before it was refined to adjacency holds them.
/// A gap's ends were adjacent vertices when it was handed out, so the rest of
/// the hull lies outside it, and its own lower hull holds exactly the vertices
/// inside it.
fn with_gap_vertices(mut vertices: Vec<HullPoint>, gaps: &[WalkGap]) -> Vec<HullPoint> {
    for g in gaps {
        let mut within: Vec<HullPoint> = g
            .front
            .iter()
            .map(HullPoint::from)
            .filter(|p| g.a.energy() < p.energy() && p.energy() < g.b.energy())
            .collect();
        within.sort_by_key(|p| p.energy());
        let hull = lower_hull(std::iter::once(g.a).chain(within).chain([g.b]));
        for &v in &hull[1..hull.len() - 1] {
            if !vertices.iter().any(|u| u.energy() == v.energy()) {
                vertices.push(v);
            }
        }
    }
    vertices.sort_by_key(|v| v.energy());
    vertices
}

/// The sweep's score of `p` at `lambda`, `λ·E − 100·(1 − λ)·L`, energy in the
/// fold's integer units.
fn sweep_score(p: HullPoint, lambda: f64) -> f64 {
    lambda * f64::from(p.energy().0) - 100.0 * (1.0 - lambda) * p.adaptation_log
}

/// How loose the chord of the passes at `lo` and `hi` cuts at `lambda`, in
/// kcal/mol: `[g(λ) − (t·g(lo) + (1 − t)·g(hi))]/λ`, with `g` the scalarized
/// optimum over `vertices` and `t = (hi − λ)/(hi − lo)`; zero for a chord
/// with one end. Exact once every vertex optimal at the three weights is
/// among `vertices`.
fn chord_looseness(vertices: &[HullPoint], lambda: f64, lo: f64, hi: f64) -> f64 {
    if hi <= lo {
        return 0.0;
    }
    let g = |w: f64| {
        vertices
            .iter()
            .map(|&v| sweep_score(v, w))
            .fold(f64::INFINITY, f64::min)
            / 100.0
    };
    let t = (hi - lambda) / (hi - lo);
    (g(lambda) - (t * g(lo) + (1.0 - t) * g(hi))) / lambda
}

/// Every Pareto design between the λ-sweep designs at `lo` and `hi`, found one
/// hull gap at a time, on the coding-only lattice.
///
/// The hull is found by the chord method: fold at both weights, then at the
/// tie weight of each pair of consecutive vertices found so far. A design
/// scoring below the pair's tie value is a new vertex between them; none
/// certifies the pair adjacent. That certifying fold at the tie weight is the
/// exact scalar fold the gap's cut needs, so the gap is solved with it and
/// pays only the cut's backward pass and its front fold on top. Every gap
/// reads one exact `λ = 1` credit, built once.
///
/// With [`WalkOptions::stop_slack`], a pair whose cut leaves at most that much
/// slack (`λ*·ΔE`, the tie weight times the pair's energy gap in the fold's
/// integer units) is solved as one gap without refining it, whatever its probe
/// found: a narrow pair's fold is cheaper than the probes that would split it,
/// and [`design_front_between`] is exact for any ordered pair, adjacent or
/// not. With [`WalkOptions::slice_width`] (or [`WalkOptions::slice_slack`]),
/// a wide gap is solved as energy slices (see `solve_gap`), whose folds cost
/// far less than one wide box's.
///
/// The sweep and the gaps share one work queue on `threads` workers, with no
/// barrier between them. A pair's refinement depends only on its two ends, so
/// a probe that splits its pair queues the two halves' probes at once, and
/// the walk probes the same weights as a one-at-a-time sweep and finds the
/// same vertices. A probe that certifies its pair takes the gap's backward
/// pass while its scan is still held, so no more than `threads` scans are
/// ever held, and queues the gap's fold. Probes run before folds, and folds
/// run widest energy window first ([`WalkOptions::fold_by_width`]; widest
/// slack first without it), so a heavy gap starts early rather than last.
/// Which gaps a chord reaches depends on the order passes reach the pool, so
/// with [`WalkOptions::chord_bracket`] a tied design may be spelled
/// differently from run to run; the front's objectives never change.
///
/// Each gap's front is [`design_front_between`]'s for that pair, through the
/// same solve, so the walk is exact wherever `between` is. Gaps share their
/// end vertices, which appear once in [`WalkedFront::front`].
///
/// # Errors
///
/// [`BeamError::InvalidAminoAcid`] for a residue outside the alphabet, and any
/// error a gap's solve returns.
pub fn design_front_walk(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lo: f64,
    hi: f64,
    options: WalkOptions,
    threads: usize,
) -> Result<WalkedFront, BeamError> {
    let none = UtrSpec::Fixed(String::new());
    design_front_walk_with_utr(
        params,
        protein,
        usage,
        &none,
        &none,
        RegionConstraints::default(),
        lo,
        hi,
        options,
        threads,
    )
}

/// [`design_front_walk`] over a molecule carrying 5′ and 3′ flanks: every
/// Pareto design of (whole-molecule MFE, coding CAI) between the λ-sweep
/// designs at `lo` and `hi`.
///
/// The walk's hull folds, cuts, credit and gap solves all run over whatever
/// lattice they are given, so only the lattice and its corner change here.
/// The lattice is built with an inert [`UtrProfile`], so a flank position
/// credits nothing and the front's `cai` axis is the coding sequence's alone,
/// as in [`design_front_bounded_with_utr`]. The corner is the flanked
/// molecule's own minimum: a corner taken over the coding-only lattice would be
/// a completion credit stated over a different space, and so inadmissible.
///
/// `constraints` forbids pairs by region class (and by an accessibility
/// window, if one is set). Every fold the walk runs applies the same constraint —
/// the `λ = 1` fold the credit and optimum come from, every hull and probe
/// fold, every cut's pass, and every gap's front fold — so the walk recovers
/// the exact front of the **constrained** problem: each design scored at its best
/// structure among those that form no forbidden pair. That is not the design's
/// free MFE, which may pair across a forbidden class at lower energy.
///
/// Empty flanks with no constraint reproduce [`design_front_walk`] exactly,
/// which is that function.
///
/// # Errors
///
/// [`BeamError::SoftRegionUnsupported`] for a constraint that penalizes rather
/// than forbids: the front fold takes hard constraints only. Otherwise as
/// [`design_front_walk`], plus [`BeamError::InvalidNucleotide`] for a
/// malformed flank.
#[allow(clippy::too_many_arguments)]
pub fn design_front_walk_with_utr(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    constraints: RegionConstraints,
    lo: f64,
    hi: f64,
    options: WalkOptions,
    threads: usize,
) -> Result<WalkedFront, BeamError> {
    design_front_walk_run(
        params,
        protein,
        usage,
        utr5,
        utr3,
        constraints,
        lo,
        hi,
        options,
        threads,
        WalkRun::default(),
    )
}

/// [`design_front_walk_with_utr`] under a [`WalkRun`]: continuing an earlier
/// walk's open boxes, rather than walking the whole front.
///
/// # Errors
///
/// As [`design_front_walk_with_utr`].
#[allow(clippy::too_many_arguments)]
pub fn design_front_walk_run(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    constraints: RegionConstraints,
    lo: f64,
    hi: f64,
    options: WalkOptions,
    threads: usize,
    run: WalkRun<'_>,
) -> Result<WalkedFront, BeamError> {
    if !constraints.is_hard() {
        return Err(BeamError::SoftRegionUnsupported);
    }
    let aa = residues(protein)?;
    let five = utr5.parse()?;
    let three = utr3.parse()?;
    let profile = UtrProfile::inert();
    let graph = dfa::build_with_utr_profile(&aa, usage, five.chain(), three.chain(), profile);
    let region = region::PairConstraint::new(&constraints, &graph);
    let corner =
        || achievable_minimum_with_utr(params, protein, usage, utr5, utr3, profile, constraints);
    walk_lattice(
        params,
        (&graph, &region, aa.len()),
        &corner,
        (lo, hi),
        options,
        threads,
        (run, None),
    )
}

/// [`design_front_walk_run`] once per stop codon of a protein whose stop is
/// designed (it ends in `*`): one walk over a stop-labelled lattice read at
/// each stop ([`dfa::Dfa::for_stop`]), so every front is that stop's alone,
/// each equal to the walk with that stop written into the 3′ flank. The
/// fronts come in [`dfa::STOP_CODON_NUCS`] order, `None` for a stop no design
/// reaches.
///
/// # Errors
///
/// As [`design_front_walk_run`], and [`BeamError::StopNotDesigned`] when the
/// protein does not end in `*`.
#[allow(clippy::too_many_arguments)]
pub fn design_front_walk_per_stop(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    constraints: RegionConstraints,
    lo: f64,
    hi: f64,
    options: WalkOptions,
    threads: usize,
) -> Result<Vec<Option<WalkedFront>>, BeamError> {
    if !constraints.is_hard() {
        return Err(BeamError::SoftRegionUnsupported);
    }
    let aa = residues(protein)?;
    let five = utr5.parse()?;
    let three = utr3.parse()?;
    let labelled =
        dfa::build_stop_labelled(&aa, usage, five.chain(), three.chain(), UtrProfile::inert())
            .ok_or(BeamError::StopNotDesigned)?;
    let shared = SharedPasses::default();
    (0..dfa::STOP_LANES)
        .map(|k| {
            let graph = labelled.for_stop_sharing(k);
            let region = region::PairConstraint::new(&constraints, &graph);
            let corner = || {
                let tables = walk_scan(params, &graph, 1.0, ScanConfig::default(), &region);
                tables.c[graph.ix(graph.end())].map(|best| best.mfe).ok_or(
                    BeamError::AchievableMinimumDisagrees {
                        corner: Energy(0),
                        folded: None,
                    },
                )
            };
            if corner().is_err() {
                return Ok(None);
            }
            walk_lattice(
                params,
                (&graph, &region, aa.len()),
                &corner,
                (lo, hi),
                options.clone(),
                threads,
                (WalkRun::default(), Some(&shared)),
            )
            .map(Some)
        })
        .collect()
}

/// The plain walk over a built lattice, its terminal [`dfa::Dfa::end`].
fn walk_lattice(
    params: &Params,
    (graph, region, residues): (&dfa::Dfa, &region::PairConstraint, usize),
    corner: &(dyn Fn() -> Result<Energy, BeamError> + Sync),
    (lo, hi): (f64, f64),
    options: WalkOptions,
    threads: usize,
    (run, shared): (WalkRun<'_>, Option<&SharedPasses>),
) -> Result<WalkedFront, BeamError> {
    let mut kernel = PlainWalk::new(
        params,
        (graph, region, residues),
        options,
        (lo, hi),
        corner,
        threads,
        run.resume,
    );
    kernel.shared = shared;
    let start = kernel.start()?;
    kernel.t = std::time::Instant::now();
    let start = kernel.queue_start(start);
    let drive = walk::Drive {
        budget: walk::Budget::new(
            kernel.options.budget,
            kernel.t,
            kernel.options.gap_folds,
            None,
            false,
        ),
        solved_to: |_| 0,
        checkpoint: run.checkpoint,
        t: kernel.t,
        gap_secs: None,
        gap_double: false,
    };
    walk::run(kernel, start, drive, PlainStats::default(), threads)
}

/// The plain lattice as the walk's [`walk::Kernel`]: scalar scans of the codon
/// lattice for the hull, their backward passes for the cuts, and
/// [`solve_gap`] for each gap, all reading one exact `λ = 1` credit.
struct PlainWalk<'a> {
    params: &'a Params,
    graph: &'a dfa::Dfa,
    region: &'a region::PairConstraint,
    residues: usize,
    options: WalkOptions,
    threads: usize,
    lo: f64,
    hi: f64,
    corner: &'a (dyn Fn() -> Result<Energy, BeamError> + Sync),
    resume: Option<&'a WalkResume>,
    /// Whether the walk continues an earlier one's open boxes.
    resumed: bool,
    seeded: bool,
    overlap: bool,
    grid_ends: bool,
    /// Every reader of the credit runs on a pair, and no pair is handed out
    /// before the credit stands, so it is set once and read thereafter.
    side: OnceLock<EnergySide>,
    credit_clock: Mutex<f64>,
    /// The grid's passes are queued ahead of the probes rather than taken
    /// before them: a probe that starts before its bracket is pooled runs
    /// exact, and its gap takes its own pass as without a grid.
    pool: Mutex<PassPool>,
    grid_done: Mutex<f64>,
    /// The grid passes' CPU seconds and count, which price a fold's change of
    /// cut ([`WalkOptions::own_when_alone`]).
    grid_cpu: Mutex<(f64, usize)>,
    /// The grid weights whose passes are taken several to a traversal;
    /// `None` takes each alone.
    grid_lanes: Option<GridLanes>,
    /// Brackets between seed weights with candidate weights inside them,
    /// each refined by its own seed pair's probes. Priced, every bracket is
    /// watched and passes go where they pay.
    gate: Option<Mutex<refine_gate::Gate<GapJob>>>,
    one_from_credit: bool,
    ends_s: f64,
    /// Grid passes shared with the walks at the lattice's other stops, when
    /// its backward passes complete to every stop.
    shared: Option<&'a SharedPasses>,
    /// The walk's clock, which every span it reports is timed against.
    t: std::time::Instant,
}

/// What the plain walk counts of its sweep.
#[derive(Debug, Clone, Default)]
struct PlainStats {
    sweep_scans: usize,
    probes_cut: usize,
}

/// What the plain walk's setup leaves for the queue: the vertices and ends
/// folded before it, and the pairs to settle.
struct PlainStart {
    vertices: Vec<HullPoint>,
    open: Vec<(HullPoint, HullPoint)>,
    ends: (Option<HullPoint>, Option<HullPoint>),
}

impl<'a> PlainWalk<'a> {
    /// What ends a gap slice's fold, so the slice is halved: more stored
    /// points than [`WalkOptions::slice_points`], or resident memory past the
    /// walk's budget. `None` without either, so the fold reads nothing.
    fn fold_stop(&self) -> Option<FoldStop> {
        let points = self.options.slice_points;
        let cap_kb = self
            .options
            .budget
            .max_rss_gb
            .map(|gb| gb * 1024.0 * 1024.0);
        if points.is_none() && cap_kb.is_none() {
            return None;
        }
        Some(Arc::new(move |stored: usize| {
            points.is_some_and(|n| stored > n)
                || cap_kb.is_some_and(|cap| rss_kb().is_some_and(|kb| kb as f64 > cap))
        }))
    }

    fn new(
        params: &'a Params,
        (graph, region, residues): (&'a dfa::Dfa, &'a region::PairConstraint, usize),
        options: WalkOptions,
        (lo, hi): (f64, f64),
        corner: &'a (dyn Fn() -> Result<Energy, BeamError> + Sync),
        threads: usize,
        resume: Option<&'a WalkResume>,
    ) -> Self {
        let mut options = options;
        if let Some(grid) = &options.pass_grid {
            let widest = grid.widest() + 1e-9;
            options.chord_bracket = Some(options.chord_bracket.map_or(widest, |w| w.max(widest)));
        }
        let resumed = resume.is_some_and(|r| !r.points.is_empty() || !r.open.is_empty());
        let seeded = options.seed_sweep && options.pass_grid.is_some() && !resumed;
        let overlap = seeded && options.overlap_credit;
        let grid_ends = seeded && options.grid_ends;
        Self {
            params,
            graph,
            region,
            residues,
            options,
            threads,
            lo,
            hi,
            corner,
            resume,
            resumed,
            seeded,
            overlap,
            grid_ends,
            side: OnceLock::new(),
            credit_clock: Mutex::new(0.0),
            pool: Mutex::new(PassPool::default()),
            grid_done: Mutex::new(0.0),
            grid_cpu: Mutex::new((0.0, 0)),
            grid_lanes: None,
            gate: None,
            one_from_credit: false,
            ends_s: 0.0,
            shared: None,
            t: std::time::Instant::now(),
        }
    }

    /// The exact `λ = 1` fold and credit on `workers`.
    fn build_credit(&self, workers: usize) -> Result<EnergySide, BeamError> {
        if let Some(side) = self.shared.and_then(|s| s.credit(self.graph, self.region)) {
            return Ok(side);
        }
        let corner = || (self.corner)();
        let mut built = EnergySide::new(
            self.params,
            self.graph,
            corner,
            true,
            workers,
            self.region.clone(),
        )?;
        let optima = built.fold.as_ref().map(|t| stop_energies(self.graph, t));
        built.build_credit(self.params, self.graph, workers);
        if let (Some(shared), Some(optima), Some(credit)) = (self.shared, optima, &built.credit) {
            shared.put_credit(credit.outside(), optima);
        }
        Ok(built)
    }

    /// The grid pass at `lambda` alone, on `workers`, pooled; its optimum.
    fn grid_pass(
        &self,
        lambda: f64,
        workers: usize,
        queue: &walk::WalkQueue<Self>,
    ) -> Option<HullPoint> {
        let (params, graph) = (self.params, self.graph);
        if let Some((outside, optimum)) = self.shared.and_then(|s| s.get(lambda, graph)) {
            self.pool_pass(lambda, outside, queue);
            return optimum;
        }
        let cpu = task_clock::TaskClock::start();
        let config = credit_config(workers);
        let tables = walk_scan(params, graph, lambda, config, self.region);
        let optimum = grid_optimum(graph, &tables);
        let outside = cost_to_go::Outside::compute_at_any_lambda(params, graph, &tables, config);
        if let Some(shared) = self.shared {
            shared.put(lambda, &outside, graph, &tables);
        }
        drop(tables);
        self.count_grid_cpu(cpu.finish(), 1);
        self.pool_pass(lambda, outside, queue);
        optimum
    }

    /// The grid pass at the `k`th lane weight: taken in one traversal with
    /// those at the nearest weights no task holds or has taken, which are
    /// pooled as they are taken, or alone on more than one worker, where the
    /// one-worker traversal would idle the rest. The optimum, which a pass
    /// taken in an earlier task's traversal left in its slot.
    fn lane_pass(
        &self,
        k: usize,
        workers: usize,
        queue: &walk::WalkQueue<Self>,
    ) -> Option<HullPoint> {
        let lanes = self.grid_lanes.as_ref().expect("a lane weight has lanes");
        let mut slot = lanes.slots[k].lock().expect("a grid slot's lock");
        if let Some(optimum) = *slot {
            return optimum;
        }
        if self.shared.is_some_and(|s| s.has(lanes.weights[k])) {
            let optimum = self.grid_pass(lanes.weights[k], workers, queue);
            *slot = Some(optimum);
            return optimum;
        }
        let mut near: Vec<usize> = (0..lanes.weights.len()).filter(|&j| j != k).collect();
        near.sort_by_key(|&j| j.abs_diff(k));
        // A slot is locked while its pass is taken and filled when it is
        // pooled, so an unlocked empty one is taken by no one; `try_lock`
        // never waits, so no two tasks can each hold a slot the other waits
        // on.
        let mut held = Vec::new();
        for j in near {
            if workers != 1 || held.len() + 1 == lanes.lanes {
                break;
            }
            if let Ok(other) = lanes.slots[j].try_lock() {
                if other.is_none() {
                    held.push((j, other));
                }
            }
        }
        if held.is_empty() {
            let optimum = self.grid_pass(lanes.weights[k], workers, queue);
            *slot = Some(optimum);
            return optimum;
        }
        let (params, graph) = (self.params, self.graph);
        let ks: Vec<usize> = std::iter::once(k)
            .chain(held.iter().map(|(j, _)| *j))
            .collect();
        let lambdas: Vec<f64> = ks.iter().map(|&j| lanes.weights[j]).collect();
        let cpu = task_clock::TaskClock::start();
        let config = credit_config(1);
        let (tables, _) = outside_multi::scan_lanes(params, graph, &lambdas, config, self.region);
        let optima: Vec<Option<HullPoint>> =
            tables.iter().map(|t| grid_optimum(graph, t)).collect();
        let optima_tables: Vec<[Option<HullPoint>; dfa::STOP_LANES]> =
            tables.iter().map(|t| stop_optima(graph, t)).collect();
        let refs: Vec<&tables::Tables> = tables.iter().collect();
        let (costs, _) = outside_multi::cost_to_go_lanes(
            params,
            graph,
            &refs,
            self.region,
            config.single_max_len,
            1,
        );
        drop(refs);
        drop(tables);
        self.count_grid_cpu(cpu.finish(), ks.len());
        for ((&lambda, costs), t) in lambdas.iter().zip(costs).zip(&optima_tables) {
            let outside = cost_to_go::Outside::from_costs(graph.layout(), costs);
            if let Some(shared) = self.shared {
                shared.put_optima(lambda, &outside, t);
            }
            self.pool_pass(lambda, outside, queue);
        }
        for ((_, mut other), optimum) in held.into_iter().zip(&optima[1..]) {
            *other = Some(*optimum);
        }
        *slot = Some(optima[0]);
        optima[0]
    }

    /// The most grid passes one traversal takes in this walk.
    fn lanes_taken(&self) -> usize {
        self.grid_lanes.as_ref().map_or(1, |l| l.lanes)
    }

    fn count_grid_cpu(&self, secs: f64, passes: usize) {
        let mut spent = self.grid_cpu.lock().expect("the grid CPU's lock");
        spent.0 += secs;
        spent.1 += passes;
    }

    /// Pool the grid pass at `lambda` and release what waited for it.
    fn pool_pass(
        &self,
        lambda: f64,
        mut outside: cost_to_go::Outside,
        queue: &walk::WalkQueue<Self>,
    ) {
        share_pooled_keys(&self.pool, &mut outside);
        self.pool
            .lock()
            .expect("the pool's lock")
            .insert(lambda, Arc::new(outside));
        {
            let mut done = self.grid_done.lock().expect("the grid clock's lock");
            *done = done.max(self.t.elapsed().as_secs_f64());
        }
        if let Some(gate) = &self.gate {
            let at = self.t.elapsed().as_secs_f64();
            let held = gate.lock().expect("the gate's lock").pooled(lambda, at);
            Self::release_held(held, queue);
        }
    }

    /// The folds a walk takes before its queue starts: the credit, and the
    /// ends unless the queue folds them or the walk resumed an earlier hull.
    fn start(&mut self) -> Result<PlainStart, BeamError> {
        let (params, graph, region, threads) = (self.params, self.graph, self.region, self.threads);
        let (lo, hi) = (self.lo, self.hi);
        // An earlier walk's hull, in this walk's units, and the boxes it
        // left. Resuming takes the hull as given: its vertices were folded
        // once and a box's end is read from them, so the same box settles the
        // same way however often it is resumed.
        // A hull point carries no design, so a walk stopped before it settled
        // a box cannot write that box's ends among its points: the ends come
        // from the open boxes as well as from the points marked as vertices,
        // and their union is the hull the earlier walk had.
        let mut prior: Vec<HullPoint> = self
            .resume
            .iter()
            .flat_map(|r| {
                let vertices = r
                    .points
                    .iter()
                    .filter(|(_, on_hull)| *on_hull)
                    .map(|(p, _)| HullPoint::from(p));
                let ends = r.open.iter().flat_map(|&(a, b)| {
                    [a, b].map(|(mfe_kcal_mol, adaptation_log)| HullPoint {
                        mfe_kcal_mol,
                        adaptation_log,
                    })
                });
                vertices.chain(ends).collect::<Vec<_>>()
            })
            .collect();
        prior.sort_by_key(|p| p.energy());
        prior.dedup_by_key(|p| p.energy());
        let vertex_at = |at: (f64, f64)| {
            let point = HullPoint {
                mfe_kcal_mol: at.0,
                adaptation_log: at.1,
            };
            prior
                .iter()
                .copied()
                .find(|p| p.energy() == point.energy())
                .unwrap_or(point)
        };
        let credit_clock = &self.credit_clock;
        let set_clock = |s: f64| *credit_clock.lock().expect("the credit clock's lock") = s;
        let (vertices, open, ends, ends_s) = if self.resumed {
            // The hull is the earlier walk's, so neither end is folded again
            // and the pairs to settle are the boxes it left open. The credit
            // is built before the queue starts rather than in it: every probe
            // reads the credit's optimum, and a resumed walk's queue begins
            // with probes.
            let t = std::time::Instant::now();
            let built = self.build_credit(threads)?;
            set_clock(t.elapsed().as_secs_f64());
            set_once(&self.side, built);
            let boxes = self
                .resume
                .iter()
                .flat_map(|r| &r.open)
                .map(|&(a, b)| (vertex_at(a), vertex_at(b)))
                .filter(|(a, b)| a.energy() < b.energy())
                .collect();
            (prior.clone(), boxes, (None, None), 0.0)
        } else if self.overlap {
            (Vec::new(), Vec::new(), (None, None), 0.0)
        } else if self.grid_ends {
            let t = std::time::Instant::now();
            // The credit and the stable end side by side, on half the workers
            // each (one after the other on one).
            let mut setup = share_threads(threads, 2, |job, workers| {
                if job == 1 {
                    return (
                        None,
                        Some(hull_vertex(
                            params,
                            graph,
                            hi,
                            credit_config(workers),
                            region,
                        )),
                    );
                }
                let started = std::time::Instant::now();
                let built = self.build_credit(workers);
                set_clock(started.elapsed().as_secs_f64());
                (Some(built), None)
            })
            .into_iter();
            let built = setup.next().and_then(|s| s.0).expect("the credit's job");
            let stable = setup
                .next()
                .and_then(|s| s.1)
                .expect("the stable end's job");
            set_once(&self.side, built?);
            let (stable, _) = stable?;
            (
                vec![stable],
                Vec::new(),
                (Some(stable), None),
                t.elapsed().as_secs_f64(),
            )
        } else {
            let t = std::time::Instant::now();
            let built = self.build_credit(threads)?;
            set_clock(t.elapsed().as_secs_f64());
            set_once(&self.side, built);

            let t = std::time::Instant::now();
            let mut ends = share_threads(threads, 2, |k, workers| {
                let lambda = if k == 0 { hi } else { lo };
                hull_vertex(params, graph, lambda, credit_config(workers), region)
            })
            .into_iter();
            let ends_s = t.elapsed().as_secs_f64();
            let (stable, _) = ends.next().expect("two end folds")?;
            let (adapted, _) = ends.next().expect("two end folds")?;
            drop(ends);
            let mut vertices = vec![stable];
            let mut open = Vec::new();
            if stable.energy() < adapted.energy() {
                vertices.push(adapted);
                open.push((stable, adapted));
            }
            (vertices, open, (Some(stable), Some(adapted)), ends_s)
        };
        self.ends_s = ends_s;
        Ok(PlainStart {
            vertices,
            open,
            ends,
        })
    }

    /// The queue's first tasks, and the seeds and refine gate the walk reads,
    /// once the setup folds are done.
    fn queue_start(&mut self, start: PlainStart) -> walk::Start<Self> {
        let PlainStart {
            vertices,
            mut open,
            ends,
        } = start;
        let (lo, hi) = (self.lo, self.hi);
        let mut grid_weights = self
            .options
            .pass_grid
            .as_ref()
            .map(PassGrid::weights)
            .unwrap_or_default();
        // The grid's `λ = 1` pass is the credit's, built from the same exact
        // fold: pool a copy rather than scan and pass again. Overlapped, the
        // credit's task pools it.
        // The adapted end comes from the grid's lowest pass, moved up to
        // `lo`; a weight below it would serve no gap.
        if self.grid_ends {
            grid_weights.retain(|&w| w > lo);
            grid_weights.insert(0, lo);
        }
        self.one_from_credit = grid_weights.last() == Some(&1.0)
            && (self.overlap || self.side.get().is_some_and(|s| s.credit.is_some()));
        if self.one_from_credit {
            grid_weights.pop();
            if let Some(credit) = self.side.get().and_then(|s| s.credit.as_ref()) {
                self.pool
                    .lock()
                    .expect("the pool's lock")
                    .insert(1.0, Arc::new(credit.outside().clone()));
            }
        }
        // Lanes hold several passes' tables at once, so a walk under a memory
        // budget, whose folds stop on resident memory, takes each alone
        // unless the count is given.
        let lanes = grid_lanes().unwrap_or(if self.options.budget.max_rss_gb.is_some() {
            1
        } else {
            GRID_LANES
        });
        self.grid_lanes = GridLanes::new(&grid_weights, lanes);
        // Seeded, the sweep starts from the pairs of grid optima as each
        // pair's two scans are done, in place of the pair of ends.
        let seeds = self.seeded.then(|| {
            let mut seeds = walk::Seeds::new(
                (hi, ends.0),
                (lo, ends.1),
                &grid_weights,
                self.overlap,
                self.grid_ends.then_some(lo),
            );
            open = seeds.ready();
            seeds
        });
        self.gate = seeds
            .as_ref()
            .filter(|_| self.options.refine_priced.is_some() || self.options.refine_loose.is_some())
            .map(|seeds| {
                let slots: Vec<f64> = seeds.slots.iter().rev().map(|&(w, _)| w).collect();
                Mutex::new(
                    match (&self.options.refine_priced, &self.options.refine_loose) {
                        (Some(_), _) => refine_gate::Gate::<GapJob>::every(&slots),
                        (None, Some(rule)) => {
                            refine_gate::Gate::<GapJob>::new(&slots, &rule.candidates.weights())
                        }
                        (None, None) => unreachable!("filtered on a rule"),
                    },
                )
            });
        let setup = [
            walk::Task::Credit,
            walk::Task::End {
                lambda: hi,
                stable: true,
            },
            walk::Task::End {
                lambda: lo,
                stable: false,
            },
        ];
        // Overlapped with grid ends, the grid's lowest pass is the adapted
        // end.
        let (overlap, grid_ends) = (self.overlap, self.grid_ends);
        let tasks = setup
            .into_iter()
            .filter(|task| {
                overlap && !(grid_ends && matches!(task, walk::Task::End { stable: false, .. }))
            })
            .map(|task| (walk::SETUP, task))
            .chain(
                grid_weights
                    .into_iter()
                    .map(|w| (walk::GRID, walk::Task::Grid(w))),
            )
            .collect();
        // Each partly solved box by its ends' energies, as the earlier walk
        // reported them.
        let units = |e: f64| (e * 100.0).round() as i32;
        let rests = self
            .resume
            .iter()
            .flat_map(|r| &r.rests)
            .map(|&((lo, hi), (from, floor))| {
                ((units(lo), units(hi)), (Energy(units(from)), floor))
            })
            .collect();
        walk::Start {
            vertices,
            ends: [ends.0, ends.1],
            pairs: open,
            tasks,
            seeds,
            rests,
            moved: hash::FxHashMap::default(),
        }
    }

    /// A certified gap is queued at once, or held while its bracket is
    /// undecided or its added passes are not yet pooled.
    fn submit(&self, priority: (u8, u64), job: GapJob, queue: &walk::WalkQueue<Self>) {
        let held = match &self.gate {
            Some(gate) => gate
                .lock()
                .expect("the gate's lock")
                .hold(job.bracket, (priority, job)),
            None => Some((priority, job)),
        };
        if let Some((priority, job)) = held {
            queue.push(priority, walk::Task::Fold(Box::new(job)));
        }
    }

    fn release_held(held: Vec<refine_gate::Held<GapJob>>, queue: &walk::WalkQueue<Self>) {
        for (priority, job) in held {
            queue.push(priority, walk::Task::Fold(Box::new(job)));
        }
    }

    /// A watched bracket whose probes are done: the width-weighted mean of
    /// its held gaps' root chord looseness against the threshold. Every
    /// vertex optimal anywhere in the bracket is known by now, so `g` is
    /// exact there.
    fn decide(&self, i: usize, queue: &walk::WalkQueue<Self>, vertices: Vec<HullPoint>) {
        let Some(gate) = &self.gate else {
            return;
        };
        let options = &self.options;
        let width = options
            .chord_bracket
            .expect("a seeded grid sets the bracket width");
        let jobs: Vec<(HullPoint, HullPoint, f64)> = gate
            .lock()
            .expect("the gate's lock")
            .held(i)
            .filter(|job| job.plan == Plan::Chord)
            .map(|job| (job.a, job.b, job.lambda))
            .collect();
        let g = |w: f64| {
            vertices
                .iter()
                .map(|&v| sweep_score(v, w))
                .fold(f64::INFINITY, f64::min)
                / 100.0
        };
        let (items, pooled): (Vec<(f64, f64)>, Vec<f64>) = {
            let pool = self.pool.lock().expect("the pool's lock");
            let items = jobs
                .iter()
                .filter_map(|&(a, b, lambda)| {
                    let (_, GapCut::Chord { lo, hi }) = pool.around(lambda, width)? else {
                        return None;
                    };
                    let loose = chord_looseness(&vertices, lambda, lo, hi);
                    Some((loose, f64::from(b.energy().0 - a.energy().0)))
                })
                .collect();
            (items, pool.passes.iter().map(|&(w, _)| w).collect())
        };
        let at = self.t.elapsed().as_secs_f64();
        let decision = match (&options.refine_priced, &options.refine_loose) {
            (Some(rule), _) => {
                let pool = self.pool.lock().expect("the pool's lock");
                let held: Vec<refine_gate::HeldGap> = jobs
                    .iter()
                    .filter(|&&(_, _, lambda)| pool.around(lambda, width).is_some())
                    .map(|&(a, b, lambda)| refine_gate::HeldGap {
                        lambda,
                        width_kcal: f64::from(b.energy().0 - a.energy().0) / 100.0,
                        slices: options.slicing().count(lambda, a.energy().0, b.energy().0),
                    })
                    .collect();
                drop(pool);
                let cost = |gap: &refine_gate::HeldGap, loose: f64| {
                    rule.cost
                        .passes(gap.lambda, gap.width_kcal, gap.slices, loose, self.residues)
                };
                let placement = refine_gate::place(
                    &held,
                    &pooled,
                    g,
                    cost,
                    rule.margin_passes,
                    rule.max_passes,
                );
                gate.lock().expect("the gate's lock").decide_placed(
                    i,
                    refine_gate::weighted_looseness(&items),
                    placement,
                    at,
                )
            }
            (None, Some(rule)) => gate.lock().expect("the gate's lock").decide(
                i,
                &items,
                rule.threshold_kcal,
                at,
                |w| pooled.contains(&w),
            ),
            (None, None) => return,
        };
        match decision {
            refine_gate::Decision::Release(held) => Self::release_held(held, queue),
            refine_gate::Decision::Refine(weights) => {
                for w in weights {
                    queue.push(walk::GRID, walk::Task::Grid(w));
                }
            }
        }
    }
}

impl walk::Kernel for PlainWalk<'_> {
    type Vertex = HullPoint;
    type Job = GapJob;
    type Gap = WalkGap;
    /// Where a gap stopped part way continues: `(from, floor)`, the energy
    /// it is solved up to and the best `cai_log` there.
    type Rest = (Energy, f64);
    type Point = FrontPoint;
    /// Whether a pooled bracket cut the probe's scan.
    type Note = bool;
    type Stats = PlainStats;
    type Out = WalkedFront;

    fn energy(v: HullPoint) -> i32 {
        v.energy().0
    }

    fn tie(a: HullPoint, b: HullPoint) -> Option<f64> {
        tie_weight(a, b)
    }

    fn policy(&self) -> walk::Policy {
        walk::Policy {
            shares: task_queue::Shares::Spread,
            max_running: self.options.max_running.unwrap_or(self.threads),
            checkpoint_skips: false,
            ends_in_turn: false,
        }
    }

    fn tag(&self, a: HullPoint, b: HullPoint) -> Option<usize> {
        let lambda = tie_weight(a, b)?;
        self.gate
            .as_ref()?
            .lock()
            .expect("the gate's lock")
            .bracket_of(lambda)
    }

    fn probe_queued(&self, bracket: Option<usize>, root: bool) {
        if let Some(gate) = &self.gate {
            gate.lock()
                .expect("the gate's lock")
                .probe_queued(bracket, root);
        }
    }

    fn probe_done(
        &self,
        bracket: Option<usize>,
        queue: &walk::WalkQueue<Self>,
        vertices: &dyn Fn() -> Vec<HullPoint>,
    ) {
        let done = self
            .gate
            .as_ref()
            .and_then(|gate| gate.lock().expect("the gate's lock").probe_done(bracket));
        if let Some(i) = done {
            self.decide(i, queue, vertices());
        }
    }

    fn kind(note: Option<bool>) -> TaskKind {
        match note {
            Some(cut) => TaskKind::Probe { cut },
            None => TaskKind::Narrow,
        }
    }

    fn credit(&self, workers: usize, queue: &walk::WalkQueue<Self>) -> Result<(), BeamError> {
        let started = std::time::Instant::now();
        let built = self.build_credit(workers)?;
        *self.credit_clock.lock().expect("the credit clock's lock") =
            started.elapsed().as_secs_f64();
        let one = self.one_from_credit.then(|| {
            built
                .credit
                .as_ref()
                .map(|credit| Arc::new(credit.outside().clone()))
        });
        set_once(&self.side, built);
        match one {
            Some(Some(outside)) => {
                self.pool
                    .lock()
                    .expect("the pool's lock")
                    .insert(1.0, outside);
                let mut done = self.grid_done.lock().expect("the grid clock's lock");
                *done = done.max(self.t.elapsed().as_secs_f64());
            }
            // No exterior at `λ = 1`, so no credit pass: the grid takes it.
            Some(None) => queue.push(walk::GRID, walk::Task::Grid(1.0)),
            None => {}
        }
        Ok(())
    }

    fn end(&self, lambda: f64, _stable: bool, workers: usize) -> Result<HullPoint, BeamError> {
        let (vertex, _) = hull_vertex(
            self.params,
            self.graph,
            lambda,
            credit_config(workers),
            self.region,
        )?;
        Ok(vertex)
    }

    fn grid(
        &self,
        lambda: f64,
        seed: bool,
        workers: usize,
        queue: &walk::WalkQueue<Self>,
    ) -> Result<Option<HullPoint>, BeamError> {
        let optimum = match self.grid_lanes.as_ref().and_then(|l| l.slot(lambda)) {
            Some(k) => self.lane_pass(k, workers, queue),
            None => self.grid_pass(lambda, workers, queue),
        };
        // A slot without an optimum would hold its pairs back for good.
        if seed {
            return optimum
                .ok_or(BeamError::AchievableMinimumDisagrees {
                    corner: Energy(0),
                    folded: None,
                })
                .map(Some);
        }
        Ok(None)
    }

    fn probe(
        &self,
        (a, b): (HullPoint, HullPoint),
        (rest, _cap): (Option<(Energy, f64)>, Option<walk::GapCap>),
        bracket: Option<usize>,
        workers: usize,
        queue: &walk::WalkQueue<Self>,
        _budget: &walk::Budget,
        _vertices: &dyn Fn() -> Vec<HullPoint>,
    ) -> Result<walk::Probed<Self>, BeamError> {
        let (params, graph, region, options) =
            (self.params, self.graph, self.region, &self.options);
        let lambda = tie_weight(a, b).ok_or(BeamError::UnorderedHullPair)?;
        let started = std::time::Instant::now();
        let bound = options
            .chord_bracket
            .filter(|_| options.cut_probes)
            .and_then(|width| {
                let pool = self.pool.lock().expect("the pool's lock");
                pool.around(lambda, width).map(|(ends, _)| ends)
            });
        let slack = lambda * f64::from(b.energy().0 - a.energy().0);
        // A pair an earlier walk certified and began to fold is folded from
        // its rest: its probe's scan serves only the gap's own pass.
        let narrow = rest.is_some() || options.stop_slack.is_some_and(|most| slack <= most);
        // A narrow pair is folded as one gap whatever its probe finds, and
        // under a pooled bracket its gap reads the chord, not the probe's
        // tables: its scan would decide nothing.
        let skip = narrow && bound.is_some();
        let (p, tables) = match &bound {
            _ if skip => (None, None),
            None => {
                let (p, tables) =
                    hull_vertex(params, graph, lambda, credit_config(workers), region)?;
                (Some(p), Some(tables))
            }
            Some(ends) => (
                probe_cut(params, graph, a, lambda, ends, workers, region),
                None,
            ),
        };
        let scan_s = started.elapsed().as_secs_f64();
        let splits = p.filter(|&p| {
            !narrow
                && sweep_score(p, lambda) < sweep_score(a, lambda) - TOL
                && a.energy() < p.energy()
                && p.energy() < b.energy()
        });
        if let Some(p) = splits {
            drop(tables);
            return Ok(walk::Probed::Split(p, bound.is_some()));
        }
        let plan = plan_cut(
            b,
            lambda,
            credit_of(&self.side).optimum,
            options,
            &self.pool,
        );
        let mut clock = CutClock {
            lambda,
            scan_s,
            pass_s: 0.0,
        };
        // A cut probe ran only because the pool brackets its weight, and the
        // pool only grows, so its gap never needs a pass of its own: the cut
        // tables could not give one.
        let outside = (plan == Plan::Own).then(|| {
            let tables = tables
                .as_ref()
                .expect("a gap taking its own pass was scanned exactly");
            let started = std::time::Instant::now();
            let mut outside = cost_to_go::Outside::compute_at_any_lambda(
                params,
                graph,
                tables,
                credit_config(workers),
            );
            if options.chord_bracket.is_some() {
                share_pooled_keys(&self.pool, &mut outside);
            }
            clock.pass_s = started.elapsed().as_secs_f64();
            Arc::new(outside)
        });
        drop(tables);
        if let (Some(outside), Some(_)) = (&outside, options.chord_bracket) {
            self.pool
                .lock()
                .expect("the pool's lock")
                .insert(lambda, Arc::clone(outside));
        }
        let job = GapJob {
            a,
            b,
            lambda,
            plan,
            outside,
            clock,
            queued: self.t.elapsed().as_secs_f64(),
            bracket,
            rest,
        };
        let priority = fold_priority(a, b, lambda, options);
        self.submit(priority, job, queue);
        Ok(walk::Probed::Queued((!skip).then_some(bound.is_some())))
    }

    fn job_pair(job: &GapJob) -> (HullPoint, HullPoint, f64, f64) {
        (job.a, job.b, job.lambda, job.queued)
    }

    fn fold(
        &self,
        job: GapJob,
        workers: usize,
        queue: &walk::WalkQueue<Self>,
        budget: &walk::Budget,
        vertices: &dyn Fn() -> Vec<HullPoint>,
    ) -> Result<(walk::Settled<Self>, usize), BeamError> {
        let (params, graph, region, options) =
            (self.params, self.graph, self.region, &self.options);
        let (pool, grid_cpu, threads) = (&self.pool, &self.grid_cpu, self.threads);
        let GapJob {
            a,
            b,
            lambda,
            plan,
            outside,
            mut clock,
            queued: _,
            bracket: _,
            rest,
        } = job;
        let peak = std::cell::Cell::new(workers);
        let mut extra = Vec::new();
        let (chain, cut) = match (plan, outside) {
            (Plan::Uncut, _) => (None, GapCut::Uncut),
            (Plan::Own, Some(outside)) => (Some(CutChain { outside, clock }), GapCut::Exact),
            _ => {
                let started = std::time::Instant::now();
                let width = options
                    .chord_bracket
                    .expect("a chord is planned only with a bracket");
                let (ends, cut) = pool
                    .lock()
                    .expect("the pool's lock")
                    .around(lambda, width)
                    .expect("a chord is planned against passes the pool holds");
                let outside = Arc::new(ends.make(lambda));
                let near = pool
                    .lock()
                    .expect("the pool's lock")
                    .neighbours(lambda, options.bracket_cuts);
                for (weight, pass) in near {
                    extra.push(CutChain {
                        outside: pass,
                        clock: CutClock {
                            lambda: weight,
                            ..CutClock::default()
                        },
                    });
                }
                clock.pass_s = started.elapsed().as_secs_f64();
                (Some(CutChain { outside, clock }), cut)
            }
        };
        // A chord between two distinct passes may give way to the gap's own
        // pass once what is left of the fold outweighs enough passes. A fold
        // holding every worker runs the pass itself; one beside other tasks
        // queues it as a grid pass, keeps its chord meanwhile, and takes the
        // pass at the first slice boundary after it is pooled.
        let loose = match cut {
            GapCut::Chord { lo, hi } if lo < hi => {
                Some(chord_looseness(&vertices(), lambda, lo, hi))
            }
            _ => None,
        };
        let chorded =
            loose.is_some_and(|loose| options.own_min_looseness.is_none_or(|least| loose >= least));
        let slicing = options.slicing();
        let asked: Mutex<Option<(usize, bool)>> = Mutex::new(None);
        let own = |last: Option<&SliceClock>, at: usize, left: usize, now: usize| {
            let (spent, passes) = *grid_cpu.lock().expect("the grid CPU's lock");
            if !chorded || left < 2 {
                return Ok(None);
            }
            if asked.lock().expect("the ask's lock").is_some() {
                let pooled = pool
                    .lock()
                    .expect("the pool's lock")
                    .passes
                    .iter()
                    .find(|(w, _)| *w == lambda)
                    .map(|(_, pass)| Arc::clone(pass));
                return Ok(pooled.map(|outside| CutChain {
                    outside,
                    clock: CutClock {
                        lambda,
                        ..CutClock::default()
                    },
                }));
            }
            let whole = now >= threads.max(1);
            let wanted = match last {
                None => false,
                Some(_) if passes == 0 => false,
                Some(last) => {
                    let rest = last.cpu_s / last.count.max(1) as f64 * left as f64;
                    let outweighs = |ratio: f64| rest >= ratio * spent / passes as f64;
                    (whole && options.own_when_alone.is_some_and(outweighs))
                        || options.own_when_costly.is_some_and(outweighs)
                }
            };
            if !wanted {
                return Ok(None);
            }
            let started = std::time::Instant::now();
            *asked.lock().expect("the ask's lock") = Some((at, false));
            if !whole {
                queue.push(walk::GRID, walk::Task::Grid(lambda));
                return Ok(None);
            }
            let config = credit_config(now);
            let tables = walk_scan(params, graph, lambda, config, region);
            let scan_s = started.elapsed().as_secs_f64();
            let mut outside =
                cost_to_go::Outside::compute_at_any_lambda(params, graph, &tables, config);
            drop(tables);
            share_pooled_keys(pool, &mut outside);
            let outside = Arc::new(outside);
            pool.lock()
                .expect("the pool's lock")
                .insert(lambda, Arc::clone(&outside));
            Ok(Some(CutChain {
                outside,
                clock: CutClock {
                    lambda,
                    scan_s,
                    pass_s: started.elapsed().as_secs_f64() - scan_s,
                },
            }))
        };
        // Read during a slice: its CPU so far times the slices after it, once
        // that outweighs the ratio, queues the pass for the next boundary.
        let ask = |cpu: f64, after: usize, next: usize, now: usize| {
            let Some(ratio) = options.own_when_costly else {
                return;
            };
            let (spent, passes) = *grid_cpu.lock().expect("the grid CPU's lock");
            if after < 2 || passes == 0 || now >= threads.max(1) {
                return;
            }
            if cpu * after as f64 >= ratio * spent / passes as f64 {
                let mut asked = asked.lock().expect("the ask's lock");
                if asked.is_none() {
                    *asked = Some((next, true));
                    queue.push(walk::GRID, walk::Task::Grid(lambda));
                }
            }
        };
        let watch = options
            .own_watch_s
            .filter(|_| chorded && options.own_when_costly.is_some())
            .map(|period| (period, &ask as &SliceAsk<'_>));
        // A budget is read before each slice too, so a gap stopped part way
        // keeps the slices it solved.
        let stop = || budget.time_spent() || budget.folds_spent() || budget.memory_spent();
        let GapSolve {
            front,
            mut clock,
            slices,
            rest,
        } = solve_gap(
            params,
            graph,
            credit_of(&self.side),
            (a, b, lambda),
            (rest, &stop, &|| budget.fold_solved()),
            self.fold_stop(),
            chain,
            extra,
            (slicing, options.floor_cut),
            workers,
            options.speculative_slices,
            || {
                let lending = options.lend_idle_workers || options.speculative_slices;
                let loan = lending.then(|| queue.lend_idle());
                let lent = loan.as_ref().map_or(0, task_queue::Loan::count);
                peak.set(peak.get().max(workers + lent));
                (lent, loan)
            },
            own,
            watch,
        )?;
        if let Some((slice, mid)) = *asked.lock().expect("the ask's lock") {
            clock.own_asked = Some(slice);
            clock.own_asked_mid_slice = mid;
        }
        if let Some(rest) = rest {
            return Ok((walk::Settled::Rest(rest, front), peak.get()));
        }
        let gap = WalkGap {
            a,
            b,
            lambda,
            front,
            clock,
            slices,
            cut,
        };
        Ok((walk::Settled::Gap(gap), peak.get()))
    }

    fn lone(
        &self,
        _v: HullPoint,
        _budget: &walk::Budget,
    ) -> Result<walk::Settled<Self>, BeamError> {
        unreachable!("the plain walk solves a lone vertex once its queue ends")
    }

    fn note(stats: &mut PlainStats, cut: bool) {
        stats.sweep_scans += 1;
        stats.probes_cut += usize::from(cut);
    }

    fn settled_gap(_stats: &mut PlainStats, _gap: &mut WalkGap, _at_s: f64) {}

    fn snapshot(
        &self,
        walk: &walk::Walk<Self>,
        mut open: Vec<(HullPoint, HullPoint)>,
    ) -> WalkedFront {
        let mut gaps: Vec<WalkGap> = walk.gaps.iter().map(|(_, g)| g.clone()).collect();
        gaps.sort_by_key(|g| g.a.energy());
        let mut vertices = walk.vertices.clone();
        vertices.sort_by_key(|v| v.energy());
        let hull = with_gap_vertices(vertices.clone(), &gaps);
        open.sort_by_key(|(a, _)| a.energy());
        let mut front = Vec::new();
        join_points(&mut front, &gaps, &walk.rest_points, self.resume);
        let rests = walk_rests(walk, &open);
        WalkedFront {
            vertices,
            hull,
            gaps,
            front,
            sweep_scans: walk.stats.sweep_scans,
            probes_cut: walk.stats.probes_cut,
            grid_s: 0.0,
            grid_lanes: self.lanes_taken(),
            stop_slack: self.options.stop_slack,
            sweep_s: 0.0,
            gaps_s: 0.0,
            rounds: walk.depth,
            credit_s: 0.0,
            timeline: Vec::new(),
            refinements: Vec::new(),
            stopped_by: None,
            open,
            rests,
        }
    }

    fn finish(
        self,
        walk: walk::Walk<Self>,
        end: walk::End,
        threads: usize,
    ) -> Result<WalkedFront, BeamError> {
        let grid_lanes = self.lanes_taken();
        let walk::End {
            timeline,
            stopped_by,
            wall_s: wall,
        } = end;
        let refinements = self
            .gate
            .map(|gate| gate.into_inner().expect("the gate's lock").records())
            .unwrap_or_default();
        let grid_s = self.grid_done.into_inner().expect("the grid clock's lock");
        let credit_s = self
            .credit_clock
            .into_inner()
            .expect("the credit clock's lock");
        let mut side = self
            .side
            .into_inner()
            .expect("the credit is built before the queue ends");
        let probes_s = walk.last_probe.unwrap_or(0.0);
        let (sweep_s, gaps_s) = (self.ends_s + probes_s, wall - probes_s);
        // A resumed walk folds no end: it was handed the hull.
        let ends_folded = match (self.resumed, self.grid_ends) {
            (true, _) => 0,
            (false, true) => 1,
            (false, false) => 2,
        };
        let (sweep_scans, rounds) = (ends_folded + walk.stats.sweep_scans, walk.depth);
        // Whatever is still outstanding is what the walk did not settle: a
        // pair its budget skipped, and a gap a refine bracket still holds,
        // which a stopped walk never releases.
        let mut open = walk.outstanding.clone();
        open.sort_by_key(|(a, _)| a.energy());
        let rests = walk_rests(&walk, &open);
        let mut gaps: Vec<WalkGap> = walk.gaps.into_iter().map(|(_, g)| g).collect();
        gaps.sort_by_key(|g| g.a.energy());
        let mut vertices = walk.vertices;
        vertices.sort_by_key(|v| v.energy());
        let hull = with_gap_vertices(vertices.clone(), &gaps);
        let mut front: Vec<FrontPoint> = Vec::new();
        if gaps.is_empty() && stopped_by.is_none() {
            // One vertex and no gap: the front between the weights is that
            // design alone, the front under a ceiling at its energy.
            let (solved, _) = solve_on(
                self.params,
                self.graph,
                UtrProfile::inert(),
                &mut side,
                Vec::new(),
                Prune::None,
                Bounds::with_ceiling(Ceiling::Absolute(vertices[0].energy())).with_exact_credit(),
                threads,
                None,
            )?;
            if let Bounded::Feasible(f) = solved {
                front = f.front;
            }
        }
        join_points(&mut front, &gaps, &walk.rest_points, self.resume);
        if stopped_by.is_none() {
            // A whole front holds a design at the energy optimum. One that does
            // not was pruned by a credit that is not a bound, and would
            // otherwise be reported as a smaller front, or an empty one.
            verify_corner(&front, side.optimum)?;
        }
        Ok(WalkedFront {
            vertices,
            hull,
            gaps,
            front,
            sweep_scans,
            probes_cut: walk.stats.probes_cut,
            grid_s,
            grid_lanes,
            stop_slack: self.options.stop_slack,
            sweep_s,
            gaps_s,
            rounds,
            credit_s,
            timeline,
            refinements,
            stopped_by,
            open,
            rests,
        })
    }
}

/// Add to `front` the points of `gaps`, of gaps solved part way
/// (`rest_points`) and of an earlier walk being resumed,
/// by ascending energy, a point two of them share once.
///
/// A resumed walk's own points lie in the earlier walk's open boxes, so the
/// two fronts interleave by energy rather than follow one another.
fn join_points(
    front: &mut Vec<FrontPoint>,
    gaps: &[WalkGap],
    rest_points: &[FrontPoint],
    resume: Option<&WalkResume>,
) {
    let settled = gaps
        .iter()
        .flat_map(|gap| gap.front.iter().cloned())
        .chain(rest_points.iter().cloned());
    let earlier = resume
        .iter()
        .flat_map(|r| &r.points)
        .map(|(p, _)| p.clone());
    let mut points: Vec<FrontPoint> = settled.chain(earlier).collect();
    points.sort_by(|p, q| {
        (p.mfe_kcal_mol, p.cai_log)
            .partial_cmp(&(q.mfe_kcal_mol, q.cai_log))
            .expect("a front point's objectives are finite")
    });
    for p in points {
        let shared = front.last().is_some_and(|q: &FrontPoint| {
            q.mfe_kcal_mol == p.mfe_kcal_mol && q.cai_log == p.cai_log
        });
        if !shared {
            front.push(p);
        }
    }
}

/// The rests of the open boxes in `open` whose gaps `walk` solved part way.
fn walk_rests(walk: &walk::Walk<PlainWalk<'_>>, open: &[(HullPoint, HullPoint)]) -> Vec<WalkRest> {
    open.iter()
        .filter_map(|&(a, b)| {
            walk.rest_of(a, b).map(|(from, floor)| WalkRest {
                lo: a.energy(),
                hi: b.energy(),
                from,
                floor,
            })
        })
        .collect()
}

/// The box between `a` and `b`: ceiling at `b`'s energy, floor at `a`'s
/// adaptation, one cut at the pair's tie weight.
fn between_box(a: HullPoint, b: HullPoint) -> Result<(Ceiling, BoxSpec), BeamError> {
    let lambda = tie_weight(a, b).ok_or(BeamError::UnorderedHullPair)?;
    Ok((
        Ceiling::Absolute(b.energy()),
        BoxSpec {
            cai_floor_log: a.adaptation_log,
            lambdas: vec![lambda],
        },
    ))
}

/// The box a band's lower edge defines: the best-adapted design of the
/// front at `lower`, as a floor. `None` when nothing meets `lower`.
fn band_edge(at_lower: Bounded<BoundedFront>, lambdas: &[f64]) -> Option<BoxSpec> {
    let Bounded::Feasible(front) = at_lower else {
        return None;
    };
    front.front.last().map(|edge| BoxSpec {
        cai_floor_log: edge.cai_log + edge.utr_profile_log,
        lambdas: lambdas.to_vec(),
    })
}

/// The exact, `λ = 1` configuration the credit's two passes run under, on the
/// solve's own workers. A threaded scan may spell a tied traceback differently
/// but fills every cell with the same energy, and the cost to go reads energies
/// alone, so the credit does not depend on the worker count.
fn credit_config(threads: usize) -> ScanConfig {
    ScanConfig {
        threads,
        ..ScanConfig::default()
    }
}

/// `protein`'s slots as codon-family indices: each residue's amino acid, and a
/// trailing `*` as the stop family ([`STOP`]).
///
/// A trailing `*` is a **designed stop codon**: one more slot of the lattice,
/// after the last residue, whose codon the design chooses among `UAA`, `UAG`
/// and `UGA`. Every stop codon weighs the same, so the slot adds nothing to any
/// design's adaptation, and the CDS a lattice builds over it ends past the stop.
/// A `*` anywhere but last would be a stop codon inside the coding sequence,
/// which no design here has, and is refused.
///
/// # Errors
///
/// [`BeamError::InvalidAminoAcid`] for a letter that is not one of the twenty
/// or a `*` before the last slot.
pub fn residues(protein: &str) -> Result<Vec<usize>, BeamError> {
    let last = protein.len().saturating_sub(1);
    protein
        .bytes()
        .enumerate()
        .map(|(i, b)| match b {
            b'*' if i == last => Ok(STOP),
            _ => aa_to_index(b as char).map_err(|_| BeamError::InvalidAminoAcid(b)),
        })
        .collect()
}

/// [`achievable_minimum`] over a molecule carrying flanks — the same
/// energy-priority corner, minimized over the same lattice the fold will walk.
///
/// Minimizing over the *same* lattice is the whole of it: the corner is one half
/// of every in-fold completion credit, and a corner taken over a wider space
/// than the fold searches is an inadmissible credit rather than a loose one. The
/// agreement is checked on every exact solve by [`verify_corner`].
fn achievable_minimum_with_utr(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    profile: UtrProfile,
    constraints: RegionConstraints,
) -> Result<Energy, BeamError> {
    let corner = design_with_utr_profile(
        params,
        protein,
        usage,
        1.0,
        utr5,
        utr3,
        constraints,
        profile,
        ScanConfig::default(),
    )?;
    Ok(Energy((corner.mfe_kcal_mol * 100.0).round() as i32))
}

/// The lowest energy any encoding of `protein` reaches — the energy-priority
/// corner, which the scalar path already computes.
///
/// Wanted before the fold for two independent reasons: it decides whether a
/// ceiling is reachable at all, and it is one half of the completion credit
/// every in-fold drop consults.
///
/// Public because a caller stating a ceiling as a **fraction** of the molecule's
/// own optimum needs the optimum before it can name the ceiling, and asking the
/// solver for it after the fact would mean folding twice. [`Ceiling`] carries an
/// absolute threshold and a fixed offset above this value; a proportional one is
/// resolved here, by the caller, against exactly the number the fold will use.
pub fn achievable_minimum(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
) -> Result<Energy, BeamError> {
    let corner = design(params, protein, usage, 1.0)?;
    Ok(Energy((corner.mfe_kcal_mol * 100.0).round() as i32))
}

/// A recovered point's energy in the fold's own integer units.
fn energy_of(point: &FrontPoint) -> Energy {
    Energy((point.mfe_kcal_mol * 100.0).round() as i32)
}

/// Check the corner the completion credit was built on against the front the
/// fold recovered.
///
/// The credit is admissible only while the achievable minimum it is stated
/// against is no higher than the true one; too low is merely loose, too high
/// discards feasible designs. The recovered front's first point is that true
/// minimum — the globally optimal derivation is within any window of its own
/// cell's best at every cell it passes through, so no admissible drop can remove
/// it — which makes the one comparison below a direct check on the only
/// direction that can be wrong. A disagreement means the corner and the fold
/// were minimized over different spaces, and the answer cannot be trusted.
fn verify_corner(front: &[FrontPoint], optimum: Energy) -> Result<(), BeamError> {
    let folded = front.first().map(energy_of);
    if folded == Some(optimum) {
        return Ok(());
    }
    Err(BeamError::AchievableMinimumDisagrees {
        corner: optimum,
        folded,
    })
}

/// Which internal-loop closure the scan runs. Both reach the identical BRD
/// optimum (MFE + CAI byte-exact); they differ only in algorithmic complexity
/// and — on degenerate equal-BRD folds — in which tied structure the traceback
/// spells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InternalLoopMode {
    /// The `O(n²·L²)` generic-internal-loop enumeration: for each inner
    /// pair, every outer `(p, q)` flank combination is scored directly. The
    /// validated reference the [`InternalLoopMode::Lz`] reduction is checked
    /// against (`tests/lz_fuzz.rs`); also the `λ=1` fold-parity cross-check
    /// substrate ([`fold_single_path`]).
    Naive,
    /// The Lyngsø–Zuker `O(n²·L)` reduction of the generic case (the default
    /// and production path). It reaches the same BRD optimum as
    /// [`InternalLoopMode::Naive`] byte-exactly (sequence + MFE + CAI); on an
    /// equal-BRD tie it may spell a different — equally optimal — structure,
    /// since its size-diagonal reduction cannot reproduce `Naive`'s
    /// enumeration-order tie-break.
    #[default]
    Lz,
}

/// Design an mRNA coding sequence for `protein` (single-letter amino-acid
/// codes) that minimizes the fixed-λ BRD objective
/// `λ·MFE − (1−λ)·CAI`, by running the exact (`b → ∞`) codon-DFA × CKY
/// scan under the given codon-usage table.
///
/// The scan is the unpruned lattice; finite-`b` pruning is an optional,
/// opt-in speed knob (see [`design_with_beam`]) and is not plumbed here.
///
/// Each loop interior's energy-determining nucleotides (hairpin and
/// internal-loop terminal mismatches, special-hairpin sequences) are chosen
/// jointly with their CAI rather than reconstructed best-CAI-first, so the
/// returned `(MFE, CAI)` matches the exact interval DP within the
/// tolerance of the two objective surfaces, which can pick neighbouring cells
/// at the same λ (byte-exact in practice; see
/// `tests/brd_parity.rs`). The scan is `O(1)` per loop candidate; next-pair
/// sparsity for the largest corpus lengths is a later optimization.
pub fn design(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
) -> Result<Design, BeamError> {
    design_with_beam(params, protein, usage, lambda, usize::MAX)
}

/// The per-column pruning policy: how each CKY table column is reduced
/// before it is consumed.
///
/// A column's cells are ranked and the lowest-ranked kept, under two
/// independent limits plus one ranking option:
///
/// - `per_open` — the maximum cells kept per distinct **5′ open node** in a
///   `P` / `Multi` column (a beam fold's "top-`b` states per position" axis,
///   grouping on the pair's span-start / codon-phase boundary). Reserving
///   budget per open node stops one long-span open from starving the others —
///   the collapse mechanism the closed-pair table exhibits. `usize::MAX`
///   disables the per-open cap (flat ranking, the original knob). The dense
///   `M` / `M2` tables hold exactly one cell per open node, so the per-open
///   cap is a structural no-op there.
/// - `cap` — the maximum cells kept per column overall, applied after the
///   per-open cap. `usize::MAX` disables it.
/// - `cai_bound` — when set, the rank adds the exact **CAI-half completion
///   bound** (the best still-achievable adaptation of the codons outside the
///   cell's span, a per-codon-decomposable prefix/suffix maximum) to each
///   cell's partial BRD, so cells of different span length are ranked by
///   *partial BRD + best-possible remaining CAI* rather than inside score
///   alone. Optimistic and admissible for the CAI half. Keyed on the 5′ open
///   node, it discriminates *between* open-node groups; it is constant within
///   an open group and so cannot reorder a `per_open` selection. It only
///   reorders which cells a finite width drops and never changes a merge
///   outcome.
/// - `mfe_bound` — when set, the rank adds the **MFE-half completion estimate**:
///   the exact best left-exterior BRD `C[open]` of the prefix before the cell's
///   5′ open node, read live from the exterior table (finalized by the time the
///   cell's column is pruned). It ranks cells by *partial BRD + best exterior
///   completion of the prefix* — exact for the exterior-absorption use of a
///   cell, an optimistic estimate for its nested / multibranch uses. Keyed on
///   the 5′ open node, it discriminates *between* open-node groups (letting the
///   flat `cap` keep the short-span cells whose long favorable prefix the inside
///   score ignores); it is constant within an open group and so cannot reorder a
///   `per_open` selection. Like `cai_bound` it only reorders the drop set and
///   never changes a merge outcome.
///
/// `per_open == usize::MAX && cap == usize::MAX` is the exact (`b → ∞`)
/// lattice — nothing is pruned regardless of `cai_bound` / `mfe_bound`, so the
/// exact default is byte-for-byte preserved. The interface is scalar-ranked
/// today, and shaped so a future front-valued generalization can replace the
/// rank without touching the grouping / cap machinery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Beam {
    /// Cells kept per distinct 5′ open node in a `P` / `Multi` column. Read as a
    /// ceiling or as a floor depending on [`Beam::group`]; see [`GroupPolicy`].
    pub per_open: usize,
    /// Max cells kept per column overall.
    pub cap: usize,
    /// Fold the exact CAI-half completion bound into the prune ranking.
    pub cai_bound: bool,
    /// Fold the exterior-MFE completion estimate (`C[open]`) into the ranking.
    pub mfe_bound: bool,
    /// Whether `per_open` bounds a group from above or protects it from below.
    pub group: GroupPolicy,
    /// How many open-node groups may hold a reservation, under
    /// [`GroupPolicy::Reserve`]. `usize::MAX` derives it from the two widths.
    ///
    /// A column's budget and its group count are not independent: reserving
    /// `per_open` cells for each of `g` groups spends `g · per_open` of `cap`,
    /// so a cap is already an implicit group budget. This field exists so the
    /// two can be measured apart before either is shipped as the policy's form.
    /// Ignored under [`GroupPolicy::Limit`], which has no reservation to bound.
    pub max_groups: usize,
}

/// Whether [`Beam::per_open`] bounds an open-node group from above or protects
/// it from below.
///
/// The distinction is the whole of the difference between the two policies, and
/// it is not a tuning choice — it changes which cells a column can lose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GroupPolicy {
    /// **A ceiling.** At most `per_open` cells survive per group, and the flat
    /// `cap` then applies among those survivors — so a group's best cells are
    /// still evictable by the column-wide rank. The original semantics, and the
    /// default, so an unchanged call prunes exactly as it did.
    #[default]
    Limit,
    /// **A floor.** The best `per_open` cells of each admitted group survive the
    /// flat `cap` unconditionally, and the remaining budget is filled by global
    /// rank from everything else.
    ///
    /// **Reserving does not, on its own, buy breadth — measured, it costs it.**
    /// Reservations are rationed by [`Beam::group_budget`], which is
    /// `cap / per_open` unless given, so at a tight cap only a couple of groups
    /// reserve at all; the rest of the column then goes to the best remaining
    /// cells, which are concentrated in the groups that already reserved. The
    /// ceiling, by discarding each group's surplus before the flat cap runs,
    /// spreads the column over *more* open nodes than the floor does.
    ///
    /// Breadth comes from raising [`Beam::max_groups`] above the derived value,
    /// which is why the two are separable knobs rather than one width with a
    /// derived split.
    ///
    /// It is **not** expected to be lossless — losslessness would follow from an
    /// exact cost-to-go, and these modes rank with an optimistic one — so a
    /// residual floor surviving it is an anticipated outcome rather than a
    /// defect.
    Reserve,
}

impl Beam {
    /// The exact (`b → ∞`) lattice: no per-open cap, no flat cap, so no column
    /// is ever pruned.
    pub const fn exact() -> Self {
        Self {
            per_open: usize::MAX,
            cap: usize::MAX,
            cai_bound: false,
            mfe_bound: false,
            group: GroupPolicy::Limit,
            max_groups: usize::MAX,
        }
    }

    /// The original flat knob: keep the `cap` lowest-BRD cells per column, with
    /// no per-open reservation and no completion bound. `cap == usize::MAX` is
    /// [`Beam::exact`].
    pub const fn flat(cap: usize) -> Self {
        Self {
            per_open: usize::MAX,
            cap,
            cai_bound: false,
            mfe_bound: false,
            group: GroupPolicy::Limit,
            max_groups: usize::MAX,
        }
    }

    /// The recommended operating point for a protein of `residues` residues: the
    /// length-scaled width from [`recommended_cap`], a per-open budget of 2
    /// under the exterior-MFE prune rank, and the **ceiling** reading of that
    /// budget.
    ///
    /// The ceiling is deliberate rather than inherited. The floor was built and
    /// measured beside it, and it does remove a width-independent residual the
    /// ceiling carries — but only at widths above the ones this policy selects.
    /// At the widths it does select the ceiling is the better of the two, by
    /// 1.30 against 4.10 kcal/mol summed over the six lengths of the dense grid.
    /// See [`GroupPolicy`].
    pub const fn recommended(residues: usize) -> Self {
        Self {
            per_open: 2,
            cap: recommended_cap(residues),
            cai_bound: false,
            mfe_bound: true,
            group: GroupPolicy::Limit,
            max_groups: usize::MAX,
        }
    }

    /// How many open-node groups may hold a reservation.
    ///
    /// An explicit [`Beam::max_groups`] wins. Otherwise it is derived from the
    /// two widths as `cap / per_open` — the largest number of groups whose
    /// reservations fit in the column — which is the split that already exists
    /// implicitly whenever a cap and a per-open budget are set together. At
    /// least one group is always admitted, so a cap narrower than `per_open`
    /// still protects the best group rather than nothing.
    pub const fn group_budget(&self) -> usize {
        if self.max_groups != usize::MAX {
            return self.max_groups;
        }
        if self.per_open == 0 || self.per_open == usize::MAX {
            return usize::MAX;
        }
        let derived = self.cap / self.per_open;
        if derived == 0 {
            1
        } else {
            derived
        }
    }

    /// Whether this policy prunes nothing (both limits disabled) — the exact
    /// lattice, byte-for-byte, whatever `cai_bound` says.
    pub const fn is_exact(&self) -> bool {
        self.per_open == usize::MAX && self.cap == usize::MAX
    }

    /// Whether either width is zero (an empty beam, rejected as
    /// [`BeamError::ZeroBeamWidth`]).
    const fn is_empty(&self) -> bool {
        self.per_open == 0 || self.cap == 0
    }
}

/// The per-column width recommended for a protein of `residues` residues.
///
/// A constant cap is wrong in both directions. Below roughly 200 residues the
/// shipped 500 never binds, so an explicitly approximate run pays close to the
/// exact cost and buys nothing; above that it is far too tight, giving up
/// 5 kcal/mol by 347 residues and 20.8 at 1273.
///
/// # Why a step and not a power law
///
/// The width a length needs was measured as `cap*(n)` — the narrowest width
/// holding the design within **1.0 kcal/mol** of the exact optimum — on 56
/// ladder lengths and again on a dense width grid. It is **not** a power law.
/// It climbs steeply to about 440 residues, sits flat at 2000 from there to
/// roughly 1300, and steps up again beyond that; a single exponent fitted
/// through it reads anywhere from `n^1.07` to `n^2.03` depending only on how
/// much of the flat stretch the range includes. A tight power-law envelope that
/// never under-provisions over-provisions by **4.7×** at the short end, which
/// gives back the speed the width exists to buy.
///
/// The measurement is also quantized: `cap*` was only ever observed at the
/// widths the grids swept. Shipping the step reports what was measured instead
/// of interpolating a law the data does not support.
///
/// # What it is calibrated to
///
/// **A 1.0 kcal/mol absolute tolerance**, and the figure does not survive being
/// read at a tighter one: at 0.5 kcal/mol seven of the same ladder lengths need
/// a wider width than this returns. An absolute tolerance is used because a
/// relative one loosens as `|MFE|` grows with `n`.
///
/// Against the 56 committed ladder lengths this under-provisions **none** and
/// over-provisions three, by at most 2.5×.
pub const fn recommended_cap(residues: usize) -> usize {
    // Breakpoints sit immediately below the shortest length observed to need the wider
    // width, so a length at a boundary is provisioned rather than starved.
    if residues >= 1400 {
        5000
    } else if residues >= 320 {
        2000
    } else if residues >= 230 {
        1000
    } else if residues >= 145 {
        500
    } else {
        100
    }
}

impl Default for Beam {
    fn default() -> Self {
        Self::exact()
    }
}

impl From<usize> for Beam {
    /// A bare width is the flat knob (`per_open` disabled), preserving the
    /// scalar `beam: usize` semantics of [`design_with_beam`].
    fn from(cap: usize) -> Self {
        Self::flat(cap)
    }
}

/// Design as [`design`], but pruning each CKY table column to the `beam`
/// lowest-BRD cells. `beam == usize::MAX` is the exact (`b → ∞`) lattice
/// (identical to [`design`]); a finite width trades optimality for speed as
/// an opt-in beam-search heuristic; the exact lattice is the default.
/// Returns [`BeamError::ZeroBeamWidth`] for `beam == 0`.
pub fn design_with_beam(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    beam: usize,
) -> Result<Design, BeamError> {
    design_with_beam_mode(
        params,
        protein,
        usage,
        lambda,
        beam,
        InternalLoopMode::default(),
    )
}

/// Design as [`design_with_beam`], selecting the internal-loop closure
/// ([`InternalLoopMode`]). Exists so the differential-fuzz oracle can run
/// the same protein × λ through both closures and assert byte-exact BRD
/// equality; production callers use the [`InternalLoopMode::default`] (`Lz`)
/// via [`design`] / [`design_with_beam`]. Runs single-threaded; use
/// [`design_with_config`] to opt into the multi-threaded scan.
pub fn design_with_beam_mode(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    beam: usize,
    mode: InternalLoopMode,
) -> Result<Design, BeamError> {
    design_with_config(
        params,
        protein,
        usage,
        lambda,
        ScanConfig {
            beam: Beam::flat(beam),
            mode,
            ..ScanConfig::default()
        },
    )
}

/// Knobs for one [`design_with_config`] scan: the beam pruning policy, the
/// internal-loop closure, and the worker-thread count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanConfig {
    /// CKY column pruning policy; [`Beam::exact`] (the default) is the exact
    /// (`b → ∞`) lattice. See [`Beam`] and [`design_with_beam`].
    pub beam: Beam,
    /// Internal-loop closure. See [`InternalLoopMode`].
    pub mode: InternalLoopMode,
    /// Worker threads for the two fat inner phases. `1` (the default) runs
    /// the single-threaded path; `>1` is an **opt-in speed mode** that reaches
    /// the byte-identical MFE + CAI optimum (only the co-optimal traceback
    /// structure among equal-BRD ties may differ). Multi-threaded wall-clock
    /// is a separate metric and does not count toward the single-thread
    /// runtime claim. `0` is treated as `1`.
    pub threads: usize,
    /// Largest total unpaired count a single-branch (bulge / internal) loop may
    /// carry. [`DEFAULT_SINGLE_MAX_LEN`] is the default and the only value with
    /// oracle parity.
    ///
    /// **This is a change to the energy model, not a tuning knob.** The default
    /// caps loops at 30 unpaired bases, as `rna_mfe`'s `MAX_LOOP_SIZE` does,
    /// which is what makes the scan's optimum comparable to
    /// `derna_codon_dp::solve_brd` (BRD parity) and to an exact fold of one
    /// sequence (λ=1 fold parity). A narrower cap forbids long
    /// loop interiors outright, so the design it returns is optimal for a
    /// *different* model and neither parity is claimed.
    ///
    /// Narrowing is a runtime knob with a real cost profile: it is worth a
    /// constant factor (the enumeration per outer cell shrinks) and does not
    /// change the scan's growth rate in protein length.
    ///
    /// Accepted range is [`MIN_SINGLE_MAX_LEN`]`..=`[`DEFAULT_SINGLE_MAX_LEN`];
    /// see [`BeamError::LoopCapOutOfRange`]. The lower bound exists because the
    /// skewed-large reduction's size band starts at 10, so a cap at or below
    /// that silently empties it and moves work to a different code path.
    pub single_max_len: usize,
}

impl ScanConfig {
    /// Reject a configuration the scan cannot serve, before any work is done.
    ///
    /// Only [`ScanConfig::single_max_len`] is checked here; the beam's own
    /// zero-width rejection stays with its callers, which report it against the
    /// width they resolved rather than the one they were handed.
    pub fn validate(&self) -> Result<(), BeamError> {
        if !(MIN_SINGLE_MAX_LEN..=DEFAULT_SINGLE_MAX_LEN).contains(&self.single_max_len) {
            return Err(BeamError::LoopCapOutOfRange {
                cap: self.single_max_len,
                min: MIN_SINGLE_MAX_LEN,
                max: DEFAULT_SINGLE_MAX_LEN,
            });
        }
        Ok(())
    }

    /// Whether the loop cap is the parity-bearing default.
    ///
    /// Paths that fold through a component carrying its own fixed cap — the
    /// AC-product constrained folder, the front-valued DP — consult this and
    /// refuse a narrowed cap rather than silently mixing two energy models in
    /// one answer.
    pub fn loop_cap_is_default(&self) -> bool {
        self.single_max_len == DEFAULT_SINGLE_MAX_LEN
    }
}

/// The single-branch loop cap of 30 unpaired bases — the default, and
/// the only value at which BRD and fold parity are claimed.
pub const DEFAULT_SINGLE_MAX_LEN: usize = 30;

/// The version of the configuration set the scan admits, written into reports
/// so a result can be matched to the grammar that produced it. It changes
/// whenever the scan admits a different set of configurations.
///
/// - `1`: a single-branch loop's unpaired total stops one short of
///   [`DEFAULT_SINGLE_MAX_LEN`], and a branching loop's leading unpaired run is
///   capped at it.
/// - `2`: a single-branch loop admits an unpaired total of exactly
///   [`DEFAULT_SINGLE_MAX_LEN`], and a branching loop's leading run is
///   unbounded — the energy model's own set.
pub const GRAMMAR_VERSION: u32 = 2;

/// The narrowest single-branch loop cap [`ScanConfig::single_max_len`] accepts.
///
/// The skewed-large size band the Lyngsø–Zuker reduction sweeps begins at 10, so
/// a cap at or below it leaves that band empty and routes every generic closure
/// down the direct path. That may well be correct, but it is a different
/// decomposition than the one the differential oracle covers, so it is refused
/// rather than shipped unverified.
pub const MIN_SINGLE_MAX_LEN: usize = 11;

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            beam: Beam::exact(),
            mode: InternalLoopMode::Lz,
            threads: 1,
            single_max_len: DEFAULT_SINGLE_MAX_LEN,
        }
    }
}

/// Design as [`design_with_beam_mode`], with the full [`ScanConfig`] — the
/// entry point that exposes the opt-in multi-threaded scan (`config.threads`).
/// `threads = 1` is byte-for-byte the single-threaded path.
pub fn design_with_config(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    config: ScanConfig,
) -> Result<Design, BeamError> {
    design_with_config_counted(params, protein, usage, lambda, config).map(|(design, _)| design)
}

/// [`design_with_config`], additionally reporting how many CKY cells the scan
/// materialized ([`CellCounts`]) — the deterministic memory proxy, for a caller
/// measuring the lattice's size rather than only its optimum. The design is
/// byte-for-byte [`design_with_config`]'s.
pub fn design_with_config_counted(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    config: ScanConfig,
) -> Result<(Design, CellCounts), BeamError> {
    design_with_config_profiled(params, protein, usage, lambda, config)
        .map(|(design, counts, _)| (design, counts))
}

/// [`design_with_config_counted`], additionally reporting where the scan's
/// wall-clock went as a JSON object (see [`profile`]).
///
/// The attribution is [`serde_json::Value::Null`] unless the scan was asked to
/// record one, and the design is byte-for-byte [`design_with_config`]'s either
/// way — the timer changes what is observed, never what is computed.
pub fn design_with_config_profiled(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    config: ScanConfig,
) -> Result<(Design, CellCounts, serde_json::Value), BeamError> {
    if config.beam.is_empty() {
        return Err(BeamError::ZeroBeamWidth);
    }
    config.validate()?;
    let aa = residues(protein)?;
    let graph = dfa::build(&aa, usage);
    let tables = scan::scan(params, &graph, lambda, config);
    let counts = tables.cell_counts();
    let profile = tables.profile().to_json();
    Ok((finish(params, &graph, tables), counts, profile))
}

/// Design as [`design_with_config`], excluding a set of forbidden nucleotide
/// motifs from the output: the returned coding sequence contains none of
/// `motifs` (IUPAC strings, `T` read as `U`) as a contiguous substring, and is
/// the exact BRD optimum over that motif-free feasible set (hard exclusion, not
/// a soft penalty).
///
/// The constraint is additive and opt-in. An **empty** motif set is routed
/// through [`design_with_config`] verbatim, so it is byte-identical to the
/// unconstrained design — the constraint machinery allocates nothing and
/// perturbs no result. A motif set with no motif-free encoding of the protein
/// returns [`BeamError::Infeasible`] cleanly.
///
/// The mechanism is an Aho–Corasick automaton producted onto the codon DFA
/// ([`ac`], [`product`]): every path through the product spells a motif-free
/// sequence, and the folding scan optimizes BRD over exactly that set.
pub fn design_with_motifs<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    motifs: &[S],
    config: ScanConfig,
) -> Result<Design, BeamError> {
    // A CDS-only design is the composite one with no flanks, no class table and
    // no profile — routed through the same entry point rather than a parallel
    // one, so the two cannot drift.
    let (no_utr5, no_utr3) = (empty_utr_spec(), empty_utr_spec());
    design_composite(
        params,
        protein,
        usage,
        lambda,
        &CompositeSpec {
            utr5: &no_utr5,
            utr3: &no_utr3,
            constraints: RegionConstraints::default(),
            profile: UtrProfile::inert(),
            motifs,
            motif_window: MotifWindow::all(),
            flank_complement: None,
            complement_classes: None,
            positional: PositionalConstraints::empty(),
        },
        config,
    )
}

/// What the unconstrained outside would prune from a motif-constrained fold:
/// a measurement, not a solver. See [`motif_outside_census`].
#[doc(hidden)]
pub struct OutsidePruneCensus {
    pub p_cells: usize,
    pub prunable: usize,
    /// Cells the `through` pair floor rules out on its own.
    pub prunable_through: usize,
    /// Cells ruled out by either floor: the reach production actually has.
    pub prunable_either: usize,
    /// Cells the exact constrained completion at λ = 0 rules out; **0 unless
    /// `lambda` is exactly 0**, because above it the chain value is not
    /// comparable with the ceiling. See
    /// [`constrained::PruneCensus::prunable_chain`].
    pub prunable_chain: usize,
    /// Cells any floor rules out.
    pub prunable_any: usize,
    pub unbounded: usize,
    /// The constrained and unconstrained optima, on the lattice BRD scale.
    pub constrained_optimum: f64,
    pub free_optimum: f64,
    /// Seconds: the unconstrained scan, its outside pass, and the constrained
    /// fold.
    pub seconds: (f64, f64, f64),
}

/// Count the constrained `P` cells whose inside plus the **unconstrained**
/// outside of their base pair exceeds the constrained optimum (or `ceiling`):
/// what an exact prune on that bound could skip. The unconstrained completion
/// of a pair is no worse than any motif-free one, which is what would make
/// the bound admissible; this measures how much it would buy. CDS-only, exact
/// lattice. `None` when the set leaves nothing feasible.
#[doc(hidden)]
pub fn motif_outside_census<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    motifs: &[S],
    ceiling: Option<f64>,
) -> Result<Option<OutsidePruneCensus>, BeamError> {
    let aa = residues(protein)?;
    let graph = dfa::build(&aa, usage);
    let ac = ac::Ac::build(motifs)?;
    let product = product::ProductDfa::build(&graph, &ac);
    if !product.is_feasible() {
        return Ok(None);
    }
    let product = product.merge_equivalent();
    let constraint = region::PairConstraint::inert();
    let started = std::time::Instant::now();
    let tables = scan_or_infeasible(
        params,
        &graph,
        lambda,
        ScanConfig::default(),
        constraint.clone(),
        must_pair::MustPairConstraint::inert(),
    )?;
    let free_optimum = tables.c[graph.ix(graph.end())].map_or(f64::INFINITY, |s| s.brd(lambda));
    let scan_s = started.elapsed().as_secs_f64();
    let costs = outside_scan::OutsideScan::cost_to_go(
        params,
        &graph,
        &tables,
        &constraint,
        ScanConfig::default().single_max_len,
        1,
        false,
    );
    let outside_s = started.elapsed().as_secs_f64() - scan_s;
    // Production consults the pair floor as well as the outside table, so
    // headroom measured against the outside table alone credits a tighter
    // bound with cells the existing pair floor already removes.
    let through = bounded_motifs::through_table(&tables, &costs.p, lambda);
    // The exact constrained completion at λ = 0, two chain passes over the
    // product rather than a product-lattice outside pass. Built ONLY at
    // λ = 0, where a completion's structure costs nothing so the chain value
    // and the ceiling are on one scale. Above λ = 0 the chain value is
    // CAI-only while the ceiling carries energy, and comparing them directly
    // marks every cell prunable -- measured, 350,789 of 350,789 at λ = 0.0001.
    // Production never reads it alone for that reason: it chords the value
    // against an unconstrained anchor ([`constrained::ZeroChord`]), and the
    // chord is what makes the two commensurable.
    let chain = (lambda == 0.0).then(|| constrained::ChainCai::build(&product));
    let Some(c) = constrained::prune_census(
        &product,
        params,
        lambda,
        &costs.p,
        Some(&through),
        chain.as_ref(),
        ceiling,
    ) else {
        return Ok(None);
    };
    Ok(Some(OutsidePruneCensus {
        p_cells: c.p_cells,
        prunable: c.prunable,
        prunable_through: c.prunable_through,
        prunable_either: c.prunable_either,
        prunable_chain: c.prunable_chain,
        prunable_any: c.prunable_any,
        unbounded: c.unbounded,
        constrained_optimum: c.optimum,
        free_optimum,
        seconds: (
            scan_s,
            outside_s,
            started.elapsed().as_secs_f64() - scan_s - outside_s,
        ),
    }))
}

/// The empty flank — the `UtrSpec` a caller who named no UTR has.
fn empty_utr_spec() -> UtrSpec {
    UtrSpec::Fixed(String::new())
}

/// Everything a composite design is over, past the protein and λ: the two
/// flanks, the region-pair table, the designed-flank preference, and the
/// forbidden-motif set with the positions it is enforced over.
///
/// The members are one product rather than a widening argument list because
/// they are genuinely independent axes: any combination is meaningful, and each
/// has a neutral value that routes to a narrower path verbatim.
pub struct CompositeSpec<'a, S: AsRef<str> = String> {
    /// The 5′ flank — supplied, designed, or per-position mixed.
    pub utr5: &'a UtrSpec,
    /// The 3′ flank.
    pub utr3: &'a UtrSpec,
    /// The region-pair class table and accessibility window. A
    /// default-constructed [`RegionConstraints`] is inert.
    pub constraints: RegionConstraints,
    /// The per-position preference over designed flank positions.
    /// [`UtrProfile::inert`] prices nothing.
    pub profile: UtrProfile,
    /// Forbidden nucleotide motifs (IUPAC strings, `T` read as `U`). Empty is
    /// no constraint, and routes to the unconstrained scan verbatim.
    pub motifs: &'a [S],
    /// The positions the motif set is enforced over. [`MotifWindow::all`] is
    /// enforcement over the whole molecule; see [`MotifWindow`] for which
    /// occurrences a narrower window forbids.
    pub motif_window: MotifWindow,
    /// Derive the forbidden set from whichever regions are **given**, instead of
    /// supplying it — see [`FlankComplement`]. `None` is no derivation, which is
    /// every path that does not ask for one.
    ///
    /// This is not a third motif axis beside the two above; it **replaces**
    /// them. A derived set has source regions, so its enforcement window is
    /// determined rather than chosen — the designed regions — and a run that
    /// also named `motifs` or a `motif_window` would carry two sources of truth
    /// for one automaton. Either of those being non-neutral alongside a
    /// derivation is [`BeamError::Derivation`], with the message saying what to
    /// do instead.
    pub flank_complement: Option<FlankComplement>,
    /// Which region-pair classes the derivation acts across. `None` is the
    /// universal rule — every class — and is what a caller who has not stated a
    /// coverage means.
    ///
    /// A class names both of its endpoints, so this states both halves of the
    /// derivation at once: which region a set comes from and which it is
    /// enforced over. It is the same vocabulary `constraints` is written in, so
    /// a front end that lets one flag name the classes can state coverage once
    /// and pass it to both.
    ///
    /// Ignored when `flank_complement` is `None`, which is every path that does
    /// not ask for a derivation.
    pub complement_classes: Option<&'a [PairClass]>,
    /// What the caller already knows about the molecule: nucleotides pinned at
    /// chosen positions, and the fold constrained at chosen positions.
    /// [`PositionalConstraints::default`] is the empty statement and routes
    /// every path below verbatim.
    ///
    /// The two axes reach the lattice in different places — a pin removes DFA
    /// arcs, a structure character becomes a positional verdict at the
    /// `P`-write choke point or a bound on an unpaired range — which is what
    /// lets them compose with each other and with everything above.
    pub positional: &'a PositionalConstraints,
}

/// Design an mRNA under any combination of the composite constraints: flanks,
/// the region-pair table, a designed-flank preference, and a
/// position-scoped forbidden-motif set.
///
/// The general entry point [`design_with_utr_profile`] and [`design_with_motifs`]
/// are the no-motif and no-flank corners of. The four axes compose because each
/// is enforced in a different place: the flanks and the preference are graph
/// structure, the motif set is an Aho–Corasick axis on node identity
/// ([`product`]), and the class table is a verdict at pair formation. Combining
/// a motif set with a class table needs both to be consulted on the same path,
/// which is why the constrained folder takes the constraint rather than the two being
/// exclusive.
///
/// **Floors.** An empty motif set routes to the unconstrained scan verbatim, so
/// every flank / table / profile result is byte-identical to
/// [`design_with_utr_profile`]'s. Empty flanks with an inert table and an empty
/// motif set route to [`design_with_config`]. A [`MotifWindow::all`] window is
/// global enforcement, transition for transition.
///
/// A motif set with no motif-free encoding returns [`BeamError::Infeasible`]
/// cleanly — including when the infeasibility comes from the *given* half of a
/// flank, which is exactly what a window exists to avoid.
///
/// **Deriving the set instead of supplying it.** `spec.flank_complement` asks
/// for the forbidden segments to be derived from whichever regions are given and
/// enforced over whichever are designed, so no designed position can spell a
/// duplex with given sequence ([`complement`]). It pins the window rather than
/// taking one, so it is exclusive with `spec.motifs` / `spec.motif_window`; the
/// one arrangement it cannot serve — no given region to derive from — is
/// [`BeamError::Derivation`] rather than a silently self-defeating set. A
/// derivation whose every source is shorter than `k` yields no patterns and is
/// the run without it, byte for byte.
pub fn design_composite<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    spec: &CompositeSpec<'_, S>,
    config: ScanConfig,
) -> Result<Design, BeamError> {
    design_composite_counted(params, protein, usage, lambda, spec, config).map(|(design, _)| design)
}

/// [`design_composite`], additionally reporting how many CKY cells the scan
/// materialized — the deterministic memory proxy, machine-independent where
/// wall-clock and resident set size are not.
///
/// The counts are `None` when the run routed to the **AC-threaded** constrained
/// folder, which is a genuinely different lattice (a motif set threads an
/// automaton state through node identity) with its own instrumentation at
/// [`characterize_constrained`]. Reporting the two under one field would put
/// two substrates in one column.
///
/// The design is byte-for-byte [`design_composite`]'s on every path.
pub fn design_composite_counted<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    spec: &CompositeSpec<'_, S>,
    config: ScanConfig,
) -> Result<(Design, Option<CellCounts>), BeamError> {
    design_composite_routed(params, protein, usage, lambda, spec, config, None)
}

/// [`design_composite_counted`], with the constrained fold optionally run as
/// the outside-bounded threshold search ([`bounded_motifs`]) when `bounded`
/// asks for it and the lattice is exact. Every other path is unchanged.
pub(crate) fn design_composite_routed<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    spec: &CompositeSpec<'_, S>,
    config: ScanConfig,
    bounded: Option<&mut bounded_motifs::BoundReport>,
) -> Result<(Design, Option<CellCounts>), BeamError> {
    // A derivation supplies both the set and the window, so it is resolved
    // before either is read. An empty derivation — a source flank shorter than
    // `k` — yields no patterns and therefore the run without one.
    let derived = match spec.flank_complement {
        Some(fc) => Some(complement::derive_for_run(
            fc,
            spec.utr5,
            spec.utr3,
            protein.len(),
            spec.complement_classes,
            !spec.motifs.is_empty(),
            spec.motif_window,
        )?),
        None => None,
    };
    let motif_window = derived.as_ref().map_or(spec.motif_window, |d| d.window);
    // The window's reach decides how the automaton is built, and
    // [`ac::Ac::for_window`] is the one place that decides it, so the same
    // window cannot mean two constraints at two entry points. The arms differ
    // only in whose patterns they carry.
    let ac = match &derived {
        Some(d) => ac::Ac::for_window(&d.motifs, motif_window)?,
        None => ac::Ac::for_window(spec.motifs, motif_window)?,
    };
    // Empty UTRs and an inert constraint set collapse the scan domain, node set,
    // and objective to today's, so a run that is also unconstrained by its motif
    // set can take the CDS-only scan verbatim — byte-identical to the exact
    // default. A constraint set is well-defined even with no UTR (an in-CDS
    // accessibility window, say), so it takes the full path. A profile with no
    // designed position to price cannot change this, and neither can an empty
    // positional statement.
    let collapses_to_cds = spec.utr5.is_empty()
        && spec.utr3.is_empty()
        && spec.constraints.is_inert()
        && spec.positional.is_inert();
    // An automaton with nothing to match collapses the product to the base DFA
    // for every protein, so it needs neither the product nor the lattice to be
    // built before the decision. A set that *does* match something is the case
    // the product has to answer, below.
    if ac.is_pass_through() && collapses_to_cds {
        return design_with_config_counted(params, protein, usage, lambda, config)
            .map(|(design, counts)| (design, Some(counts)));
    }
    if config.beam.is_empty() {
        return Err(BeamError::ZeroBeamWidth);
    }
    config.validate()?;
    let structure = &spec.positional.structure;
    let Lowered {
        graph,
        region,
        must,
    } = lower_composite(protein, usage, spec)?;
    // Whether the run is constrained is a question about the **product**, not
    // about the automaton: a pattern set can match somewhere in the alphabet and
    // still remove nothing this protein can spell, and then the AC-threaded
    // lattice is the base lattice with an extra axis carried through it at
    // considerable cost. So the product is built first and the routing decision
    // reads it — see [`product::ProductDfa::dropped_edge`] for why that
    // predicate and not a codon census.
    let product = if ac.is_pass_through() {
        None
    } else {
        let built = product::ProductDfa::build_scoped(&graph, &ac, motif_window);
        if !built.is_feasible() {
            return Err(BeamError::Infeasible);
        }
        // On the exact lattice, fold over future-equivalence classes rather than
        // automaton states: the same feasible set with fewer cells. A finite
        // beam ranks cells, so merging would change which survive; it folds the
        // unmerged product as before.
        Some(if config.beam.is_exact() {
            built.merge_equivalent()
        } else {
            built
        })
    };
    let constrained_lattice = product.as_ref().is_some_and(|p| p.dropped_edge());
    // A set that forbade nothing here reaches the same CDS-only early return the
    // empty set does, which returns the unconstrained design verbatim. Routing
    // only at the fold's entry would recover part of that; routing here recovers
    // all of it.
    if !constrained_lattice && collapses_to_cds {
        return design_with_config_counted(params, protein, usage, lambda, config)
            .map(|(design, counts)| (design, Some(counts)));
    }
    let mut counts = None;
    let scanned = match product {
        Some(product) if constrained_lattice => {
            // A feasible, genuinely-constraining motif set: fold the AC-threaded
            // constrained CKY and recover its motif-free optimum. `config.beam`
            // prunes the constrained span columns exactly as it prunes the
            // unconstrained scan; `Beam::exact` (the default) is the exact
            // `b → ∞` constrained lattice.
            //
            // The constrained folder carries its own fixed single-branch loop
            // cap, so a narrowed one is refused here rather than applied to the
            // scan half of the answer alone. The inert-set fast path above is
            // the unconstrained scan and honours the setting normally.
            if !config.loop_cap_is_default() {
                return Err(BeamError::LoopCapUnsupported("forbidden-motif constraints"));
            }
            let fold = match bounded {
                Some(report) if config.beam.is_exact() => {
                    report.bounded = true;
                    bounded_motifs::fold(
                        params, &graph, &product, lambda, region, must, config, report,
                    )
                }
                _ => constrained::fold_with_beam(
                    &product,
                    params,
                    lambda,
                    config.beam,
                    region,
                    must,
                    config.threads,
                ),
            }
            .ok_or_else(|| motif_or_constraint_infeasible(structure, config))?;
            finish_constrained(&graph, fold, spec.profile)
        }
        _ => {
            let tables = scan_or_infeasible(params, &graph, lambda, config, region, must)?;
            counts = Some(tables.cell_counts());
            finish_with_profile(params, &graph, &tables, spec.profile)
        }
    };
    Ok((scanned, counts))
}

/// A composite statement lowered onto the lattice: the spliced DFA with
/// pins applied, and the two constraints the scan consults.
struct Lowered {
    graph: dfa::Dfa,
    region: region::PairConstraint,
    must: must_pair::MustPairConstraint,
}

/// Lower a composite statement's flanks, pins, class table and structure
/// constraints onto the lattice, refusing contradictions up front.
fn lower_composite<S: AsRef<str>>(
    protein: &str,
    usage: &CodonUsage,
    spec: &CompositeSpec<'_, S>,
) -> Result<Lowered, BeamError> {
    let aa = residues(protein)?;
    let layout = Layout::new(
        spec.utr5.len() as u32,
        aa.len() as u32,
        spec.utr3.len() as u32,
    );
    let pins = &spec.positional.pins;
    let structure = &spec.positional.structure;
    // The sequence axis, lowered: flank positions become per-position masks and
    // CDS positions become codon subsets. Both are arc removal — the lattice
    // shrinks and no DP state appears.
    let utr5 = pinned_flank(spec.utr5, &layout, pins, 0)?;
    let utr3 = pinned_flank(spec.utr3, &layout, pins, layout.cds_end())?;
    let allowed = pins
        .cds_codon_sets(&layout, &aa)
        .map_err(Infeasible::Pins)?;
    let graph = dfa::build_composite(
        &aa,
        usage,
        allowed.as_deref(),
        utr5.chain(),
        utr3.chain(),
        spec.profile,
    )
    .expect("cds_codon_sets leaves every residue at least one codon");
    // The structure axis, lowered: the forbid half joins the class constraint at the
    // `P`-write choke point, the require half becomes range bounds at the sites
    // that declare a position unpaired.
    let region =
        region::PairConstraint::new(&spec.constraints, &graph).with_mask(structure.pair_mask());
    positional::check_against_region_table(&region, &layout, structure)?;
    positional::precheck_required_pairs(&graph, &layout, structure)
        .map_err(Infeasible::PreCheck)?;
    let must = must_pair::MustPairConstraint::new(&structure.required_positions(), graph.seq_len);
    Ok(Lowered {
        graph,
        region,
        must,
    })
}

/// The best design whose structure carries a segment certificate's crossing
/// chain within a margin budget, on the exact lattice of a composite
/// statement — see [`CrossingSpec`] for what a chain is and what its margin
/// counts.
///
/// The statement is lowered exactly as [`design_composite`] lowers it and
/// folded by the same scan; the chain is then scored over that scan's own
/// inside values and its outside pass. A motif set or a derived complement
/// routes [`design_composite`] to a different lattice, so either is refused
/// here, as is any beam: the outside values are minima over every consumer
/// only on the exact lattice. Values are reported on the unrestricted
/// objective's scale, pins included, so they compare across runs exactly as
/// designs' objectives do. Nothing about [`design_composite`] changes.
pub fn crossing_composite<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    spec: &CompositeSpec<'_, S>,
    config: ScanConfig,
    crossing: &CrossingSpec,
) -> Result<CrossingReport, BeamError> {
    if !spec.motifs.is_empty() || spec.flank_complement.is_some() {
        return Err(BeamError::CrossingUnsupported(
            "a motif set or derived complement folds on a different lattice",
        ));
    }
    if !config.beam.is_exact() {
        return Err(BeamError::CrossingUnsupported(
            "a beam drops consumers, so its outside values are not minima",
        ));
    }
    config.validate()?;
    let Lowered {
        graph,
        region,
        must,
        ..
    } = lower_composite(protein, usage, spec)?;
    let tables = scan_or_infeasible(params, &graph, lambda, config, region, must)?;
    let mut report = outside_scan::OutsideScan::crossing(
        params,
        &graph,
        &tables,
        tables.region(),
        config.single_max_len,
        config.threads,
        crossing,
    );
    // The lattice prices a restricted residue against its best remaining codon;
    // the report puts every value back on the unrestricted scale, as a design's
    // report does.
    let offsets = graph.codon_offset() + graph.utr_profile_offset(spec.profile);
    report.rebase(-(1.0 - lambda) * 100.0 * offsets);
    Ok(report)
}

/// Run the unconstrained-alphabet scan under both constraints, turning an
/// unreachable end node into the typed infeasibility D8 asks for rather than
/// into a panic at traceback.
///
/// Only a positional constraint can empty this path: without one, the exterior
/// parse always reaches the end node.
fn scan_or_infeasible(
    params: &Params,
    graph: &dfa::Dfa,
    lambda: f64,
    config: ScanConfig,
    region: region::PairConstraint,
    must: must_pair::MustPairConstraint,
) -> Result<tables::Tables, BeamError> {
    let tables = scan::scan_with_constraints(params, graph, lambda, config, region, must);
    if tables.c[graph.ix(graph.end())].is_none() {
        return Err(scan_tier(config).into());
    }
    Ok(tables)
}

/// Which sentence an empty scan licenses. On the exact lattice it is a proof of
/// infeasibility; under a beam it is a statement about one search, and the two
/// must not be reported with the same words (D8).
fn scan_tier(config: ScanConfig) -> Infeasible {
    if config.beam.is_exact() {
        Infeasible::ExactScan
    } else {
        Infeasible::Beam
    }
}

/// An empty constrained fold has two possible authors — the motif set or the
/// positional constraints — and only the caller's input says which. With no
/// positional constraint active the motif set is the only candidate, so the
/// shipped message stands; otherwise the answer is path-dependent and typed.
fn motif_or_constraint_infeasible(
    structure: &StructureConstraints,
    config: ScanConfig,
) -> BeamError {
    if structure.is_inert() {
        return BeamError::Infeasible;
    }
    scan_tier(config).into()
}

/// One flank's chain, with any pins over it intersected in.
///
/// A flank no pin touches is parsed exactly as it is written, so every existing
/// flank path stays byte-identical. A pinned one becomes a per-position mask —
/// which is what [`UtrSpec::Mixed`] already is, so a pin on a designed flank
/// position and a hand-written subset there are the same object.
fn pinned_flank(
    spec: &UtrSpec,
    layout: &Layout,
    pins: &Pins,
    base: u32,
) -> Result<ParsedUtr, BeamError> {
    let parsed = spec.parse()?;
    if !pins.touches(base, parsed.len()) {
        return Ok(parsed);
    }
    let supplied: Vec<NucSet> = (0..parsed.len()).map(|i| parsed.set_at(i)).collect();
    let mask = pins
        .flank_mask(layout, base, &supplied)
        .map_err(Infeasible::Pins)?;
    Ok(ParsedUtr::Mixed(mask))
}

/// [`finish_with_profile`] for a constrained fold, which already carries its own
/// traceback: split the folder's single adaptation accumulator back into the
/// CDS-only `cai_log` and the flank's `utr_profile_log`, by the same arithmetic
/// and for the same reason.
fn finish_constrained(
    graph: &dfa::Dfa,
    fold: constrained::ConstrainedFold,
    profile: UtrProfile,
) -> Design {
    let utr_profile_log = profile_score(&fold.sequence, graph, profile);
    let cai_log =
        fold.cai - utr_profile_log + graph.utr_profile_offset(profile) + graph.codon_offset();
    Design {
        sequence: fold.sequence,
        structure: fold.structure,
        mfe_kcal_mol: fold.mfe.to_kcal_mol(),
        cai_log,
        cai: cai_of(cai_log, graph),
        utr_profile_log,
    }
}

/// Design an mRNA coding sequence for `protein` folded jointly with a **fixed**
/// 5′ and 3′ UTR: the whole mRNA `utr5 + CDS + utr3` folds as one structure,
/// and the design minimizes the fixed-λ objective `λ·MFE − (1−λ)·CAI` where the
/// MFE spans the whole mRNA and the CAI is summed over the **CDS codons alone**
/// (UTR nucleotides earn no CAI, entering only through folding free energy).
///
/// The UTR positions are lane-of-one nodes — one fixed nucleotide, no codon
/// choice — so this extends the scan domain from `0..3m` to
/// `0..(utr5.len() + 3m + utr3.len())` without changing the folding recurrence
/// (`constraint-extensions.md` § 5b). `utr5` / `utr3` are ASCII `A/C/G/U`
/// (`T` read as `U`); either may be empty.
///
/// Opt-in region-pair `constraints` (CDS↔UTR isolation and the start-proximal
/// accessibility window, each hard or soft) fold the whole mRNA under the
/// constraint — the constraint is enforced at pair formation, so the returned design
/// is the exact constrained BRD optimum, not an unconstrained fold filtered
/// afterward. A default-constructed (inert) [`RegionConstraints`] leaves the
/// design byte-identical to the unconstrained fold.
///
/// **What that optimum claims.** It is the best design *among those admitting a
/// structure the constraint allows* — a statement about the design space, not a
/// prediction about the molecule, which will fold as its energy dictates. Use
/// [`free_fold_gap`] to measure the difference on a finished design.
///
/// **A zero-length UTR with an inert constraint set is byte-identical to
/// [`design_with_config`]** — that case routes to the CDS-only substrate
/// verbatim, allocating no UTR nodes. The returned [`Design`]'s `sequence` and
/// `structure` span the whole mRNA; `mfe_kcal_mol` is the whole-mRNA MFE (plus
/// any soft region penalties) and `cai` / `cai_log` are the CDS-only adaptation.
///
/// `constraints` is a **separate** parameter rather than folded into
/// [`ScanConfig`]: the region-constraint axis and the beam/mode axis are
/// orthogonal, so they stay separate at the API.
///
/// To have the designer choose the UTR sequence instead of supplying it, use
/// [`design_with_utr_spec`]; this is its all-[`UtrSpec::Fixed`] case.
#[allow(clippy::too_many_arguments)]
pub fn design_with_utr(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    config: ScanConfig,
) -> Result<Design, BeamError> {
    design_with_utr_spec(
        params,
        protein,
        usage,
        lambda,
        &UtrSpec::Fixed(utr5.to_owned()),
        &UtrSpec::Fixed(utr3.to_owned()),
        constraints,
        config,
    )
}

/// What one UTR flank is: a sequence supplied by the caller, or a run of
/// positions for the designer to fill.
///
/// Either way the flank credits **no CAI** — the CAI half of the objective is
/// the CDS codons alone, and a UTR enters only through the whole-mRNA folding
/// energy. The two flanks are chosen independently, so a fixed 5′ UTR may be
/// paired with a designed 3′ one and vice versa.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UtrSpec {
    /// A caller-supplied sequence, ASCII `A/C/G/U` (`T` read as `U`). An empty
    /// string is a zero-length flank.
    Fixed(String),
    /// A designed flank of exactly this many nucleotides: each position is a
    /// free four-way choice the fold makes, unconstrained by any codon lattice.
    /// `Designed(0)` is a zero-length flank.
    Designed(usize),
    /// A flank written as an **IUPAC** string, one code per position, naming the
    /// nucleotides the designer may choose there: `A/C/G/U` (`T` read as `U`)
    /// pin a position, `N` leaves it free, and a degenerate code restricts it to
    /// that code's subset — `M` is `{A, C}`, `R` is `{A, G}`, `Y` is `{C, U}`.
    /// `"ACGNNNGU"` pins six nucleotides and designs three; `"MMMMM"` designs
    /// five positions over `{A, C}` alone.
    ///
    /// An all-nucleotide string is exactly [`Self::Fixed`] and an all-`N` string
    /// exactly [`Self::Designed`], so the two are its ends rather than separate
    /// mechanisms. Like them it earns no CAI.
    ///
    /// A restriction constrains the **sequence**, so it holds for the molecule
    /// rather than only for the design: two flanks held over one mutually
    /// non-pairing subset cannot spell a helix with each other under any fold.
    /// See [`nuc_set`] for which subsets have that property.
    Mixed(String),
}

impl UtrSpec {
    /// The number of nucleotide positions this flank occupies.
    pub fn len(&self) -> usize {
        match self {
            Self::Fixed(s) => s.len(),
            Self::Designed(n) => *n,
            Self::Mixed(s) => s.len(),
        }
    }

    /// Whether the flank occupies no positions.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resolve to the parsed form the DFA builder consumes, surfacing any
    /// non-nucleotide byte in a [`UtrSpec::Fixed`] flank.
    fn parse(&self) -> Result<ParsedUtr, BeamError> {
        match self {
            Self::Fixed(s) => parse_utr(s).map(ParsedUtr::Fixed),
            Self::Designed(n) => Ok(ParsedUtr::Designed(*n as u32)),
            Self::Mixed(s) => parse_utr_mask(s).map(ParsedUtr::Mixed),
        }
    }
}

/// How many synonymous codons this run's **sequence-space** constraint removes
/// from the lattice, and how many the protein has in total.
///
/// `None` when the run has no such constraint — no motif set and no derivation —
/// which is a different thing from a set that removed nothing, and is why this
/// is an `Option` rather than a zero.
///
/// This is the cheap half of *does the constraint bind*. It is a statement about
/// the **lattice**: the set took a codon away from some residue. It is not a
/// statement about the molecule — a set can remove codons the molecule could
/// never have spelled anyway, measured at one cell in twelve — so the two
/// readings are reported separately and never merged.
///
/// Costs one product build: no fold, no beam, no energy model.
pub fn removed_codons<S: AsRef<str>>(
    protein: &str,
    usage: &CodonUsage,
    spec: &CompositeSpec<'_, S>,
) -> Result<Option<(usize, usize)>, BeamError> {
    with_lattice_product(protein, usage, spec, |aa, product| {
        let surviving = product.surviving_codons();
        let mut removed = 0usize;
        let mut total = 0usize;
        for (residue, alive) in aa.iter().zip(&surviving) {
            total += AMINO_ACIDS[*residue].codons.len();
            removed += alive.iter().filter(|kept| !**kept).count();
        }
        (removed, total)
    })
}

/// Whether this run's sequence-space constraint removed any transition the
/// protein's lattice offers — the predicate the folder is routed on, reported so
/// a measurement can say which lattice a cell ran on rather than infer it from
/// wall clock.
///
/// `None` on the same condition as [`removed_codons`]: the run carries no such
/// constraint, which is not the same as a constraint that removed nothing.
///
/// It answers a **strictly finer** question than the codon census beside it. A
/// pattern straddling a codon boundary can remove a spelling while leaving every
/// codon usable somewhere, so a run can be `removed = 0` here and still be
/// genuinely constrained; the reverse cannot happen. Reporting the two together
/// is what keeps them from being read as one number.
///
/// Costs one product build: no fold, no beam, no energy model.
pub fn drops_lattice_edge<S: AsRef<str>>(
    protein: &str,
    usage: &CodonUsage,
    spec: &CompositeSpec<'_, S>,
) -> Result<Option<bool>, BeamError> {
    with_lattice_product(protein, usage, spec, |_, product| product.dropped_edge())
}

/// Build the AC × codon product a lattice reading is taken from and hand it to
/// `read`, or return `None` when the run carries no sequence-space constraint to
/// read.
///
/// The product borrows the lattice and the automaton, both built here, so it is
/// lent to the reader rather than returned. Shared so the census and the
/// dropped-edge flag are two questions asked of one construction rather than two
/// constructions that could drift.
fn with_lattice_product<S: AsRef<str>, T>(
    protein: &str,
    usage: &CodonUsage,
    spec: &CompositeSpec<'_, S>,
    read: impl FnOnce(&[usize], &product::ProductDfa) -> T,
) -> Result<Option<T>, BeamError> {
    let derived = match spec.flank_complement {
        Some(fc) => Some(complement::derive_for_run(
            fc,
            spec.utr5,
            spec.utr3,
            protein.len(),
            spec.complement_classes,
            !spec.motifs.is_empty(),
            spec.motif_window,
        )?),
        None => None,
    };
    let patterns: Vec<String> = match &derived {
        Some(d) => d.motifs.clone(),
        None => spec.motifs.iter().map(|m| m.as_ref().to_string()).collect(),
    };
    if patterns.is_empty() {
        return Ok(None);
    }
    let window = derived.as_ref().map_or(spec.motif_window, |d| d.window);
    let aa = residues(protein)?;
    let five = spec.utr5.parse()?;
    let three = spec.utr3.parse()?;
    let graph = dfa::build_with_utr_chain(&aa, usage, five.chain(), three.chain());
    let ac = ac::Ac::for_window(&patterns, window)?;
    let product = product::ProductDfa::build_scoped(&graph, &ac, window);
    Ok(Some(read(&aa, &product)))
}

/// What the feasibility pre-check found: whether the run's derived complement
/// admits any design at all, and the smallest duplex length that does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComplementFeasibility {
    /// The `k` the run asked for.
    pub k: usize,
    /// Whether some design of this protein, over these flanks, spells none of
    /// the derived patterns inside the enforcement window.
    pub feasible: bool,
    /// The smallest `k` that admits a design. `Some(k)` for every run, since a
    /// `k` longer than the source derives nothing and forbids nothing — the
    /// value is what a caller sets `--complement` to instead.
    pub smallest_feasible_k: Option<usize>,
    /// Patterns and distinct literals at the requested `k`, so a refusal says
    /// what it was refusing rather than only that it refused.
    pub motifs: usize,
    pub literals: usize,
    /// Reachability checks the bisection spent. Logarithmic in the source
    /// length, and reported because "free" is a claim rather than an adjective.
    pub checks: usize,
}

/// Does any design satisfy this run's derived complement?
///
/// **A region table can never be infeasible** — the all-unpaired structure
/// satisfies every one of them, so a design always exists. Only a complement can
/// empty the space, which is why this runs exactly when one is asked for.
///
/// Pure reachability over the Aho–Corasick × codon-graph product: no fold, no
/// beam, no energy model. It converts a doomed solve into an immediate refusal
/// naming the smallest `k` that would have worked.
///
/// **It checks the lattice the run will actually solve**, designed flanks
/// included. A check over a narrower lattice than the solve can pass while the
/// solve comes back infeasible, and a caller reads *feasible* as a claim about
/// their run. It takes the run's own [`CompositeSpec`] for the same reason: a
/// pre-check resolving a run differently from the design is a second source of
/// truth about what the run is.
///
/// The smallest feasible `k` is found by **bisection, and that is exact**.
/// Feasibility is monotone in `k`: the derived set is the reverse complements of
/// the source's `k`-windows, and a longer window matches fewer segments, so a
/// feasible `k` stays feasible for every larger one. Bisection therefore returns
/// the true boundary rather than a grid's resolution.
pub fn check_complement<S: AsRef<str>>(
    protein: &str,
    usage: &CodonUsage,
    spec: &CompositeSpec<'_, S>,
) -> Result<Option<ComplementFeasibility>, BeamError> {
    let Some(fc) = spec.flank_complement else {
        return Ok(None);
    };
    let aa = residues(protein)?;
    let five = spec.utr5.parse()?;
    let three = spec.utr3.parse()?;
    let graph = dfa::build_with_utr_chain(&aa, usage, five.chain(), three.chain());

    let mut checks = 0usize;
    let mut feasible_at = |k: usize| -> Result<(bool, usize, usize), BeamError> {
        let derived = complement::derive_for_run(
            FlankComplement { k, ..fc },
            spec.utr5,
            spec.utr3,
            protein.len(),
            spec.complement_classes,
            !spec.motifs.is_empty(),
            spec.motif_window,
        )?;
        checks += 1;
        if derived.is_empty() {
            // Nothing derived is nothing forbidden, which is the run without
            // the complement — feasible by construction, not by search.
            return Ok((true, 0, 0));
        }
        let ac = ac::Ac::for_window(&derived.motifs, derived.window)?;
        let feasible = product::ProductDfa::build_scoped(&graph, &ac, derived.window).is_feasible();
        Ok((feasible, derived.motifs.len(), derived.literals))
    };

    let (feasible, motifs, literals) = feasible_at(fc.k)?;
    // A `k` past the longest source derives nothing, so it is feasible and
    // bounds the search from above without a further check.
    let ceiling = spec.utr5.len().max(spec.utr3.len()) + 1;
    let smallest = if feasible {
        Some(smallest_feasible(fc.k, &mut feasible_at)?)
    } else if fc.k >= ceiling {
        None
    } else {
        Some(smallest_feasible_above(fc.k, ceiling, &mut feasible_at)?)
    };
    Ok(Some(ComplementFeasibility {
        k: fc.k,
        feasible,
        smallest_feasible_k: smallest,
        motifs,
        literals,
        checks,
    }))
}

/// The smallest feasible `k` at or below a `k` already known feasible.
fn smallest_feasible(
    known: usize,
    feasible_at: &mut impl FnMut(usize) -> Result<(bool, usize, usize), BeamError>,
) -> Result<usize, BeamError> {
    let (mut lo, mut hi) = (1usize, known);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if feasible_at(mid)?.0 {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    Ok(lo)
}

/// The smallest feasible `k` above a `k` known infeasible, given a ceiling that
/// is feasible by construction.
fn smallest_feasible_above(
    known_infeasible: usize,
    feasible_ceiling: usize,
    feasible_at: &mut impl FnMut(usize) -> Result<(bool, usize, usize), BeamError>,
) -> Result<usize, BeamError> {
    let (mut lo, mut hi) = (known_infeasible + 1, feasible_ceiling);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if feasible_at(mid)?.0 {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    Ok(lo)
}

/// A [`UtrSpec`] with its given nucleotides parsed — the owner of the slice
/// [`dfa::UtrChain`] borrows.
enum ParsedUtr {
    Fixed(Vec<Nucleotide>),
    Designed(u32),
    Mixed(Vec<NucSet>),
}

impl ParsedUtr {
    /// Positions this flank occupies.
    fn len(&self) -> u32 {
        self.chain().len()
    }

    /// The nucleotides position `i` of this flank may spell — what a pin over
    /// it is intersected against.
    fn set_at(&self, i: u32) -> NucSet {
        match self {
            Self::Fixed(nts) => NucSet::of(nts[i as usize]),
            Self::Designed(_) => NucSet::ALL,
            Self::Mixed(mask) => mask[i as usize],
        }
    }

    fn chain(&self) -> dfa::UtrChain<'_> {
        match self {
            Self::Fixed(nts) => dfa::UtrChain::Fixed(nts),
            Self::Designed(n) => dfa::UtrChain::Designed(*n),
            Self::Mixed(mask) => dfa::UtrChain::Mixed(mask),
        }
    }
}

/// Design an mRNA for `protein` folded jointly with its 5′ and 3′ UTRs, each
/// flank either **supplied** or **designed** ([`UtrSpec`]).
///
/// The generalization of [`design_with_utr`] from fixed flanks to designed ones.
/// The whole mRNA `UTR₅′ + CDS + UTR₃′` folds as one structure and the design
/// minimizes `λ·MFE − (1−λ)·CAI`, where the MFE spans the whole mRNA and the CAI
/// is summed over the **CDS codons alone**. A designed UTR position is a single
/// node with four parallel zero-CAI edges (`A`/`C`/`G`/`U`), so it extends the
/// scan domain without touching the folding recurrence.
///
/// Because a designed UTR earns no CAI, its bulk is driven purely by folding
/// energy — minimizing MFE makes it maximally structured. The start-proximal
/// accessibility window in `constraints` is the counter-term that keeps the
/// initiation region open; designing a 5′ UTR without one yields a start codon
/// buried in structure.
///
/// Opt-in region-pair `constraints` (CDS↔UTR isolation and the accessibility
/// window, each hard or soft) fold the whole mRNA *under* the constraint, so the
/// result is the exact constrained optimum rather than a filtered unconstrained
/// fold — the optimum over designs that *admit* an allowed structure, which is
/// weaker than a claim about how the molecule folds ([`free_fold_gap`] measures
/// the difference). An inert [`RegionConstraints`] leaves the design
/// unconstrained.
///
/// **Floors.** An all-[`UtrSpec::Fixed`] pair is byte-identical to
/// [`design_with_utr`] on the same strings; two zero-length flanks with an inert
/// constraint set are byte-identical to [`design_with_config`], routing to the
/// CDS-only substrate verbatim.
#[allow(clippy::too_many_arguments)]
pub fn design_with_utr_spec(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    constraints: RegionConstraints,
    config: ScanConfig,
) -> Result<Design, BeamError> {
    design_with_utr_profile(
        params,
        protein,
        usage,
        lambda,
        utr5,
        utr3,
        constraints,
        UtrProfile::inert(),
        config,
    )
}

/// [`design_with_utr_spec`] with a [`UtrProfile`] pricing the designed UTR
/// positions.
///
/// A designed UTR earns no CAI, so without this term nothing in the objective
/// mentions its nucleotides and they are settled by folding energy alone —
/// or, under a constraint set that forbids the flank every pair it could form,
/// by nothing at all, leaving every spelling exactly co-optimal. The profile is
/// the term that prices them, and the objective becomes
///
/// ```text
/// λ·MFE − (1−λ)·( CAI_log + U )
/// ```
///
/// with `U` summed over the designed UTR positions and `CAI_log` still the CDS
/// codons alone. That is still a fixed-λ scalarization of two quantities, so a
/// λ-sweep still traces the exact Pareto front — of folding energy against
/// `CAI_log + U`. The two halves are reported separately on [`Design`].
///
/// The profile states a **preference**, not a target: `U` is a sum of
/// independent per-position weights, so on its own it is maximized by putting
/// the profile's most-favoured nucleotide everywhere. Any intermediate
/// composition in an optimum comes from the folding energy it trades against.
///
/// **Floors.** An inert profile is byte-identical to [`design_with_utr_spec`] on
/// every path, and a flank with no designed position is unaffected by any
/// profile — a given position has one out-edge and no choice to price.
#[allow(clippy::too_many_arguments)]
pub fn design_with_utr_profile(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    utr5: &UtrSpec,
    utr3: &UtrSpec,
    constraints: RegionConstraints,
    profile: UtrProfile,
    config: ScanConfig,
) -> Result<Design, BeamError> {
    design_composite(
        params,
        protein,
        usage,
        lambda,
        &CompositeSpec {
            utr5,
            utr3,
            constraints,
            profile,
            motifs: &[] as &[String],
            motif_window: MotifWindow::all(),
            flank_complement: None,
            complement_classes: None,
            positional: PositionalConstraints::empty(),
        },
        config,
    )
}

/// Parse a UTR string (ASCII `A/C/G/U`, `T` accepted as `U`) into nucleotides,
/// surfacing any other byte as [`BeamError::InvalidNucleotide`].
fn parse_utr(utr: &str) -> Result<Vec<Nucleotide>, BeamError> {
    utr.bytes()
        .map(|b| {
            let b = if b == b'T' { b'U' } else { b };
            Nucleotide::from_byte(b).ok_or(BeamError::InvalidNucleotide(b))
        })
        .collect()
}

/// Parse a mixed-UTR mask: one IUPAC code per position, naming the nucleotides
/// that position may spell. `A/C/G/U` (`T` read as `U`) pin it, `N` leaves it
/// free, and a degenerate code restricts it to that subset. Any non-IUPAC byte
/// is [`BeamError::InvalidNucleotide`].
fn parse_utr_mask(utr: &str) -> Result<Vec<NucSet>, BeamError> {
    utr.bytes()
        .map(|b| NucSet::from_iupac(b).ok_or(BeamError::InvalidNucleotide(b)))
        .collect()
}

/// A recorded characterization of the exact (`b → ∞`) constrained fold for one
/// `(protein, motif-set, λ)`: the AC automaton size, the reachable-`(node, q)`
/// sparsity of the product layout, and the materialized CKY cell counts. It is
/// the deterministic backbone of a perf/memory measurement; a caller times a
/// fold around [`characterize_constrained`] to pair these counts with
/// wall-clock and memory.
#[derive(Debug, Clone, Copy)]
pub struct ConstrainedCharacterization {
    /// The AC automaton state count `|A_M|` — the nominal dense AC multiplier.
    pub ac_states: usize,
    /// Base codon-DFA nodes `(seq_len + 1) · 2` — the dense node axis the AC
    /// dimension multiplies.
    pub base_nodes: usize,
    /// Materialized product `(node, q)` nodes — the reachable-state total
    /// `Σ_node |reachable q|`. The sparse layout's actual size; compare to the
    /// dense `base_nodes · ac_states`.
    pub product_nodes: usize,
    /// The largest reachable-AC-state count at any single node.
    pub reachable_state_max: usize,
    /// Materialized closed-pair (`P`) span cells.
    pub p_cells: usize,
    /// Materialized in-progress multi-loop (`Multi`) span cells.
    pub multi_cells: usize,
    /// Materialized single-branch (`M`) multi-branch cells.
    pub m_cells: usize,
    /// Materialized `M2` multi-branch cells.
    pub m2_cells: usize,
    /// Materialized exterior (`C`) cells (one per reachable `(node, q)`).
    pub c_cells: usize,
}

impl ConstrainedCharacterization {
    /// Total materialized CKY cells across all five tables — the deterministic
    /// memory proxy (machine-independent, unlike wall-clock / RSS).
    pub fn total_cells(&self) -> usize {
        self.p_cells + self.multi_cells + self.m_cells + self.m2_cells + self.c_cells
    }

    /// Mean reachable AC states per base node — the sparsity figure. When it is
    /// `≪ |A_M|` the reachability-pruned layout is winning.
    pub fn mean_reachable_states(&self) -> f64 {
        self.product_nodes as f64 / self.base_nodes as f64
    }
}

/// Build the AC × product layout for `(protein, motifs)` and fold the exact
/// (`b → ∞`) constrained lattice, reporting its layout and cell-count
/// characterization. Returns `Ok(None)` when the motif set is unconstraining
/// (empty / pass-through) or makes the protein unencodable (infeasible) —
/// neither exercises the constrained folder.
pub fn characterize_constrained<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    motifs: &[S],
) -> Result<Option<ConstrainedCharacterization>, BeamError> {
    characterize_constrained_scoped(
        params,
        protein,
        usage,
        lambda,
        motifs,
        MotifWindow::all(),
        RegionConstraints::default(),
    )
}

/// [`characterize_constrained`] with the motif set enforced over `window` and
/// the region table `constraints` consulted at the `P`-write.
///
/// Both are measurements the unscoped form cannot make: whether a window
/// confines the layout blow-up to its own positions, and what the constraint call
/// costs the constrained fold. The graph is the CDS-only lattice either way, so
/// the window is a range of coding positions and the only class the table can
/// reach is `cds-cds`.
pub fn characterize_constrained_scoped<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    motifs: &[S],
    window: MotifWindow,
    constraints: RegionConstraints,
) -> Result<Option<ConstrainedCharacterization>, BeamError> {
    let ac = ac::Ac::for_window(motifs, window)?;
    if ac.is_pass_through() {
        return Ok(None);
    }
    let aa = residues(protein)?;
    let graph = dfa::build(&aa, usage);
    let product = product::ProductDfa::build_scoped(&graph, &ac, window);
    if !product.is_feasible() {
        return Ok(None);
    }
    let region = region::PairConstraint::new(&constraints, &graph);
    let Some(counts) = constrained::span_cell_counts(&product, params, lambda, region) else {
        return Ok(None);
    };
    let reachable_state_max = (0..=graph.seq_len)
        .flat_map(|pos| graph.nodes_at(pos))
        .map(|&node| product.reachable_states(node).len())
        .max()
        .unwrap_or(0);
    Ok(Some(ConstrainedCharacterization {
        ac_states: counts.ac_states,
        base_nodes: graph.slots(),
        product_nodes: counts.product_nodes,
        reachable_state_max,
        p_cells: counts.p_cells,
        multi_cells: counts.multi_cells,
        m_cells: counts.m_cells,
        m2_cells: counts.m2_cells,
        c_cells: counts.c_cells,
    }))
}

/// Fold a fixed coding sequence at λ=1 over its single-path DFA — the
/// fold-parity cross-check substrate. `codons` pins each position to one
/// nucleotide-index triple (`A=0,C=1,G=2,U=3`).
pub fn fold_single_path(params: &Params, protein: &[usize], codons: &[[u8; 3]]) -> Design {
    let graph = dfa::build_single_path(protein, codons);
    finish(
        params,
        &graph,
        scan::scan(
            params,
            &graph,
            1.0,
            ScanConfig {
                mode: InternalLoopMode::Naive,
                ..ScanConfig::default()
            },
        ),
    )
}

/// Design over a **restricted** codon lattice: `allowed[i]` is the set of codons
/// protein position `i` may take, and the flanks are fixed.
///
/// The unconstrained objective over part of the design space. This is what a
/// search that partitions the space by codon choice needs — pin a prefix,
/// forbid one codon at the branch position, leave the rest free, and re-solve.
/// Restricting every position to all of its synonymous codons reproduces the
/// unrestricted design exactly; restricting each to one codon folds that single
/// sequence.
///
/// No region constraint is involved. A design whose *own unconstrained fold* must
/// satisfy a structural predicate is not found by constraining the fold — the
/// predicate is a property of the result, so it is tested on the result
/// ([`sequence_gap`]) and the lattice search only has to enumerate in objective
/// order.
///
/// Returns `Ok(None)` when some position's codon set is empty — an infeasible
/// region of the space, reported rather than papered over.
#[allow(clippy::too_many_arguments)]
pub fn design_restricted(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    allowed: &[Vec<[u8; 3]>],
    utr5: &str,
    utr3: &str,
    config: ScanConfig,
) -> Result<Option<Design>, BeamError> {
    design_restricted_with_region(
        params,
        protein,
        usage,
        lambda,
        allowed,
        utr5,
        utr3,
        RegionConstraints::default(),
        config,
    )
}

/// Design under **placed** forbidden motifs — each pattern with the window it is
/// forbidden inside, rather than one window for the set.
///
/// The sibling of [`design_restricted_with_region`] for a caller whose
/// restriction is a set of patterns each holding at one place, rather than a
/// per-residue codon subset. Both exist for the cut loop, which is the only
/// caller that accumulates restrictions round by round and therefore the only
/// one that needs them to be positional without being global.
///
/// `Ok(None)` is an empty design space — the placed set admits no encoding —
/// which is the same contract [`design_restricted_with_region`] has for a codon
/// subset that empties a residue. Distinguishing it from an error is what lets
/// the loop report *exhausted* rather than *failed*.
///
/// A placed set gives up the AC axis's collapse outside the window (see
/// [`ac::Ac::build_placed`]), so this is for a handful of short patterns on a
/// short molecule. It is not a general design entry.
#[allow(clippy::too_many_arguments)]
pub fn design_with_placed_motifs(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    placed: &[(String, ac::MotifWindow)],
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    config: ScanConfig,
) -> Result<Option<Design>, BeamError> {
    if config.beam.is_empty() {
        return Err(BeamError::ZeroBeamWidth);
    }
    config.validate()?;
    let aa = residues(protein)?;
    let (five, three) = (parse_utr(utr5)?, parse_utr(utr3)?);
    let graph = dfa::build_with_utr_chain(
        &aa,
        usage,
        dfa::UtrChain::Fixed(&five),
        dfa::UtrChain::Fixed(&three),
    );
    solve_placed(params, &graph, lambda, placed, constraints, config)
}

/// Design over an **already-built** lattice under placed forbidden motifs.
///
/// The half of [`design_with_placed_motifs`] below the graph, shared with
/// [`design_restricted_with_placed_motifs`] so the two differ only in which
/// lattice they hand it — a full codon lattice or one a walk has restricted —
/// and cannot drift in what the placed set then means.
fn solve_placed(
    params: &Params,
    graph: &dfa::Dfa,
    lambda: f64,
    placed: &[(String, ac::MotifWindow)],
    constraints: RegionConstraints,
    config: ScanConfig,
) -> Result<Option<Design>, BeamError> {
    let region = region::PairConstraint::new(&constraints, graph);
    let must = must_pair::MustPairConstraint::new(&[], graph.seq_len);
    let ac = ac::Ac::build_placed(placed)?;
    // Whether the placed set constrains is a question about the **product**, as
    // it is on the composite path: a cut placed at a window this lattice cannot
    // spell the pattern in removes nothing, and the AC-threaded fold would carry
    // the whole automaton axis to rediscover that. A placement gives the
    // predicate one more way to come back false than an unplaced set has — the
    // pattern must be spellable *there*, not merely somewhere — though how often
    // it does on the cut sets a walk actually accumulates is unmeasured.
    let product = if ac.is_pass_through() {
        None
    } else {
        let built = product::ProductDfa::build_scoped(graph, &ac, ac::MotifWindow::all());
        if !built.is_feasible() {
            return Ok(None);
        }
        // Merge as the supplied-set path does. A placed set gains more from it:
        // its accept test is scoped per pattern while the automaton is stepped
        // everywhere, so the axis splits every node rather than only those
        // inside one window, and most of that splitting is between states with
        // the same completions.
        Some(if config.beam.is_exact() {
            built.merge_equivalent()
        } else {
            built
        })
    };
    let Some(product) = product.filter(|p| p.dropped_edge()) else {
        // Nothing was forbidden that this lattice offers, so the run is the run
        // without the set, through the same scan the unplaced path takes for an
        // empty one.
        let tables = scan_or_infeasible(params, graph, lambda, config, region, must)?;
        return Ok(Some(finish_with_profile(
            params,
            graph,
            &tables,
            UtrProfile::inert(),
        )));
    };
    // The constrained folder carries its own fixed single-branch loop cap; the
    // inert-set path above is the ordinary scan and honours a narrowed one.
    if !config.loop_cap_is_default() {
        return Err(BeamError::LoopCapUnsupported("forbidden-motif constraints"));
    }
    let Some(fold) = constrained::fold_with_beam(
        &product,
        params,
        lambda,
        config.beam,
        region,
        must,
        config.threads,
    ) else {
        return Ok(None);
    };
    Ok(Some(finish_constrained(graph, fold, UtrProfile::inert())))
}

/// [`design_with_placed_motifs`] over a **restricted** codon lattice — the two
/// restriction vocabularies a best-first walk needs at once.
///
/// A walk that partitions the design space states its regions as per-residue
/// codon subsets, and a walk that cuts a counterexample states the cut as a
/// pattern at the place it occurred. A walk doing both needs one solve honouring
/// both, which is this entry: `allowed` restricts which codons the lattice
/// offers and `placed` forbids patterns at their windows on top of it.
///
/// `Ok(None)` is an empty design space from either axis — a residue whose codon
/// set is empty, or a placed set admitting no encoding of what is left. Both are
/// reported rather than erroring, so a caller can distinguish *exhausted* from
/// *failed*.
///
/// **Floors.** An empty `placed` is byte-identical to
/// [`design_restricted_with_region`], and a full `allowed` is byte-identical to
/// [`design_with_placed_motifs`].
#[allow(clippy::too_many_arguments)]
pub fn design_restricted_with_placed_motifs(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    allowed: &[Vec<[u8; 3]>],
    placed: &[(String, ac::MotifWindow)],
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    config: ScanConfig,
) -> Result<Option<Design>, BeamError> {
    if config.beam.is_empty() {
        return Err(BeamError::ZeroBeamWidth);
    }
    config.validate()?;
    let aa = residues(protein)?;
    let (five, three) = (parse_utr(utr5)?, parse_utr(utr3)?);
    let Some(graph) = dfa::build_restricted_with_utr(&aa, usage, allowed, &five, &three) else {
        return Ok(None);
    };
    solve_placed(params, &graph, lambda, placed, constraints, config)
}

/// Design over a **restricted** codon lattice under active region constraints —
/// [`design_restricted`] with a constraint, and [`design_with_utr`] over part of the
/// design space.
///
/// The two axes are independent and compose here for the first time: the
/// restriction says which sequences the solve may choose among, the constraint says
/// which structures it may score them by. The result is the exact optimum of the
/// constrained objective over that part of the space — the lower bound a
/// branch-and-bound over regions of the design space needs at each node.
///
/// An inert `constraints` is byte-identical to [`design_restricted`], which
/// routes here.
///
/// `config.threads` is honoured, and defaults to `1`. A caller partitioning the
/// design space has two kinds of concurrency available and they do not contend:
/// this one fans out *inside* a single fold, while
/// [`SearchInputs::search_threads`](crate::SearchInputs::search_threads) examines
/// several regions side by side. The second is worth about five times the first
/// on the same hardware, so the one to reach for is the width; the case for this
/// one is a solve with no siblings to run beside it — a bounded search's root,
/// which gates the whole walk and is the only solve there is until it returns.
///
/// Fanning out does not move the answer **at exact width**, where a region's
/// solved optimum is the exact restricted optimum and a partitioning caller's
/// bound is valid at all: measured over a region-constrained whole-mRNA design at
/// 1/2/4/8/16 threads and two lengths, the emitted design is byte-identical at
/// every thread count. Under a beam the recovered structure is a different
/// matter, and so is the bound — see the caveat on
/// [`bounded_search`](crate::bounded_search).
///
/// Returns `Ok(None)` when some position's codon set is empty, exactly as
/// [`design_restricted`] does.
#[allow(clippy::too_many_arguments)]
pub fn design_restricted_with_region(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    allowed: &[Vec<[u8; 3]>],
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    config: ScanConfig,
) -> Result<Option<Design>, BeamError> {
    if config.beam.is_empty() {
        return Err(BeamError::ZeroBeamWidth);
    }
    config.validate()?;
    let aa = residues(protein)?;
    let (utr5, utr3) = (parse_utr(utr5)?, parse_utr(utr3)?);
    let Some(graph) = dfa::build_restricted_with_utr(&aa, usage, allowed, &utr5, &utr3) else {
        return Ok(None);
    };
    let region = region::PairConstraint::new(&constraints, &graph);
    Ok(Some(finish(
        params,
        &graph,
        scan::scan_with_region(params, &graph, lambda, config, region),
    )))
}

/// [`design_restricted_with_region`], additionally requiring every position in
/// `must_pair` (absolute, 0-based over the whole mRNA) to carry a base pair in
/// the designed structure ([`design_must_paired`]). The exact optimum over the
/// (sequence, structure) pairs that respect the lattice, the table and the
/// requirement; `Ok(None)` when none does. An empty `must_pair` is
/// [`design_restricted_with_region`].
#[allow(clippy::too_many_arguments)]
pub fn design_restricted_with_region_must(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    allowed: &[Vec<[u8; 3]>],
    (utr5, utr3): (&str, &str),
    constraints: RegionConstraints,
    must_pair: &[u32],
    config: ScanConfig,
) -> Result<Option<Design>, BeamError> {
    if config.beam.is_empty() {
        return Err(BeamError::ZeroBeamWidth);
    }
    config.validate()?;
    let aa = residues(protein)?;
    let (utr5, utr3) = (parse_utr(utr5)?, parse_utr(utr3)?);
    let Some(graph) = dfa::build_restricted_with_utr(&aa, usage, allowed, &utr5, &utr3) else {
        return Ok(None);
    };
    let region = region::PairConstraint::new(&constraints, &graph);
    let must = must_pair::MustPairConstraint::new(must_pair, graph.seq_len);
    let tables = scan::scan_with_constraints(params, &graph, lambda, config, region, must);
    if tables.c[graph.ix(graph.end())].is_none() {
        return Ok(None);
    }
    Ok(Some(finish(params, &graph, tables)))
}

/// The constrained fold of a given sequence with positions required paired: the
/// best structure `constraints` allow over the CDS window `cds` that pairs
/// every position of `must_pair` (absolute), and its energy in kcal/mol.
/// `Ok(None)` when no allowed structure pairs them all. An empty `must_pair`
/// is the structure behind [`sequence_constrained_mfe`].
pub fn sequence_constrained_fold_must(
    params: &Params,
    sequence: &str,
    cds: CdsWindow,
    constraints: RegionConstraints,
    must_pair: &[u32],
) -> Result<Option<(String, f64)>, BeamError> {
    let nts = parse_utr(sequence)?;
    let graph = dfa::build_fixed_mrna(&nts);
    let region = region::PairConstraint::for_cds(&constraints, cds.start as u32, cds.end as u32);
    let must = must_pair::MustPairConstraint::new(must_pair, graph.seq_len);
    let tables =
        scan::scan_with_constraints(params, &graph, 1.0, ScanConfig::default(), region, must);
    if tables.c[graph.ix(graph.end())].is_none() {
        return Ok(None);
    }
    let d = finish(params, &graph, tables);
    Ok(Some((d.structure, d.mfe_kcal_mol)))
}

/// Run the restricted, region-constrained scan of [`design_restricted_with_region`]
/// and keep what it built — the lattice, the filled tables and the constraint — for a
/// pass that reads the tables afterwards. `None` when the lattice admits nothing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_restricted_with_region(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    allowed: &[Vec<[u8; 3]>],
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    config: ScanConfig,
) -> Result<Option<(dfa::Dfa, tables::Tables, region::PairConstraint)>, BeamError> {
    if config.beam.is_empty() {
        return Err(BeamError::ZeroBeamWidth);
    }
    config.validate()?;
    let aa = residues(protein)?;
    let (utr5, utr3) = (parse_utr(utr5)?, parse_utr(utr3)?);
    let Some(graph) = dfa::build_restricted_with_utr(&aa, usage, allowed, &utr5, &utr3) else {
        return Ok(None);
    };
    let region = region::PairConstraint::new(&constraints, &graph);
    let tables = scan::scan_with_region(params, &graph, lambda, config, region.clone());
    Ok(Some((graph, tables, region)))
}

/// [`scan_restricted_with_region`] with every position of `must_pair`
/// (absolute, 0-based) required paired ([`scan::scan_with_constraints`]). An empty
/// `must_pair` fills the same tables.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_restricted_with_region_must(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    allowed: &[Vec<[u8; 3]>],
    (utr5, utr3): (&str, &str),
    constraints: RegionConstraints,
    must_pair: &[u32],
    config: ScanConfig,
) -> Result<Option<(dfa::Dfa, tables::Tables)>, BeamError> {
    if config.beam.is_empty() {
        return Err(BeamError::ZeroBeamWidth);
    }
    config.validate()?;
    let aa = residues(protein)?;
    let (utr5, utr3) = (parse_utr(utr5)?, parse_utr(utr3)?);
    let Some(graph) = dfa::build_restricted_with_utr(&aa, usage, allowed, &utr5, &utr3) else {
        return Ok(None);
    };
    let region = region::PairConstraint::new(&constraints, &graph);
    let must = must_pair::MustPairConstraint::new(must_pair, graph.seq_len);
    let tables = scan::scan_with_constraints(params, &graph, lambda, config, region, must);
    Ok(Some((graph, tables)))
}

/// [`design_restricted_with_region`], keeping the lattice, tables and constraint the
/// design was read from, for a pass over the tables afterwards.
#[allow(clippy::too_many_arguments)]
pub(crate) fn design_restricted_keeping(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    allowed: &[Vec<[u8; 3]>],
    utr5: &str,
    utr3: &str,
    constraints: RegionConstraints,
    config: ScanConfig,
) -> Result<Option<(Design, dfa::Dfa, tables::Tables, region::PairConstraint)>, BeamError> {
    let Some((graph, tables, constraint)) = scan_restricted_with_region(
        params,
        protein,
        usage,
        lambda,
        allowed,
        utr5,
        utr3,
        constraints,
        config,
    )?
    else {
        return Ok(None);
    };
    let design = finish_with_profile(params, &graph, &tables, UtrProfile::inert());
    Ok(Some((design, graph, tables, constraint)))
}

/// Design as [`design_with_config`], additionally requiring every position in
/// `must_pair` — absolute, 0-based over the coding sequence — to carry a base
/// pair in the returned structure.
///
/// The result is the exact BRD optimum over the (sequence, requirement-
/// respecting structure) pairs, not a repaired approximation. An **empty**
/// `must_pair` is byte-identical to [`design_with_config`]: the predicate is
/// inert and every scan site stays on its unconstrained path.
///
/// Unlike the region-pair families, a positive structural requirement has a
/// genuinely empty feasible set — no synonymous encoding need be able to spell
/// partners for the required positions, and a required position closer than four
/// nucleotides to both ends of the sequence has no legal partner at all. That is
/// reported as `Ok(None)` rather than as a garbage optimum.
///
/// The requirement binds the **designed** structure, not the molecule: the
/// emitted sequence folded unconstrained may leave a required position unpaired.
/// Measuring that gap is [`free_fold_gap`]'s job, not this entry's.
pub fn design_must_paired(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    must_pair: &[u32],
    config: ScanConfig,
) -> Result<Option<Design>, BeamError> {
    design_must_paired_routed(params, protein, usage, lambda, must_pair, config, true)
}

/// [`design_must_paired`] with the single-branch closures **pushed** from each
/// enclosed pair rather than pulled at their enclosing position. The two
/// routes leave every cell the same, so this exists to check the pull against
/// the route that preceded it; the pull is the faster and is what
/// [`design_must_paired`] runs.
#[doc(hidden)]
pub fn design_must_paired_pushing(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    must_pair: &[u32],
    config: ScanConfig,
) -> Result<Option<Design>, BeamError> {
    design_must_paired_routed(params, protein, usage, lambda, must_pair, config, false)
}

fn design_must_paired_routed(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    must_pair: &[u32],
    config: ScanConfig,
    pull: bool,
) -> Result<Option<Design>, BeamError> {
    if config.beam.is_empty() {
        return Err(BeamError::ZeroBeamWidth);
    }
    config.validate()?;
    let aa = residues(protein)?;
    let graph = dfa::build(&aa, usage);
    let route = if pull {
        scan::scan_with_constraints
    } else {
        scan::scan_with_constraints_pushing
    };
    let tables = route(
        params,
        &graph,
        lambda,
        config,
        region::PairConstraint::inert(),
        must_pair::MustPairConstraint::new(must_pair, graph.seq_len),
    );
    if tables.c[graph.ix(graph.end())].is_none() {
        return Ok(None);
    }
    Ok(Some(finish(params, &graph, tables)))
}

fn finish(params: &Params, graph: &dfa::Dfa, tables: tables::Tables) -> Design {
    finish_with_profile(params, graph, &tables, UtrProfile::inert())
}

/// [`finish`] splitting the scan's single adaptation accumulator back into its
/// two reported halves.
///
/// The scan sums one weight per traversed edge, so a designed UTR position's
/// profile weight lands in the same accumulator the CDS codons' CAI does — which
/// is exactly what makes the term additive, and exactly what a report must undo.
/// The emitted sequence names which positions were designed, so the term's total
/// is recomputed from it directly and subtracted, leaving `cai_log` the CDS-only
/// adaptation it is documented to be. Under an inert profile the subtraction is
/// zero and every field is byte-identical to [`finish`]'s.
///
/// The subtraction alone is short by a constant whenever a designed position is
/// restricted to a subset excluding the profile's argmax. Such a position's
/// edges carry `w(x) − c_i` rather than `w(x)`, so that the best weight still
/// reachable through it is exactly zero and the span closed form stays exact
/// (`dfa::splice_utr_chain`). The accumulator is then `CAI_cds + U(v) − Σ c_i`
/// while `profile_score` recomputes `U(v)` from the **unnormalized** weights, so
/// recovering the CDS-only adaptation adds `Σ c_i` back. `utr_profile_log` stays
/// the emitted flank's true preference score under the profile as stated, which
/// is the quantity a caller asked about; the normalization is an implementation
/// detail of the bound and never reaches the report.
fn finish_with_profile(
    params: &Params,
    graph: &dfa::Dfa,
    tables: &tables::Tables,
    profile: UtrProfile,
) -> Design {
    let best = tables.c[graph.ix(graph.end())].expect("exterior parse reaches the end node");
    let recovered = traceback::recover(params, graph, tables);
    let utr_profile_log = profile_score(&recovered.sequence, graph, profile);
    let cai_log =
        best.cai - utr_profile_log + graph.utr_profile_offset(profile) + graph.codon_offset();
    Design {
        sequence: recovered.sequence,
        structure: recovered.structure,
        mfe_kcal_mol: best.mfe.to_kcal_mol(),
        cai_log,
        cai: cai_of(cai_log, graph),
        utr_profile_log,
    }
}

/// The profile's total over an emitted mRNA: its weight at each designed UTR
/// position of `graph`. Zero under an inert profile, and zero for a graph with
/// no designed position.
fn profile_score(sequence: &str, graph: &dfa::Dfa, profile: UtrProfile) -> f64 {
    if profile.is_inert() {
        return 0.0;
    }
    sequence
        .bytes()
        .enumerate()
        .filter(|(i, _)| graph.is_designed_utr(*i as u32))
        .filter_map(|(_, b)| Nucleotide::from_byte(b))
        .map(|nt| profile.weight(nt))
        .sum()
}

/// Errors surfaced by the beam-search designer.
///
/// `#[non_exhaustive]` because the scan, DFA construction, and traceback
/// will add internal-invariant variants without breaking external pattern
/// matches.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum BeamError {
    /// The energy-priority corner disagrees with the front the fold recovered,
    /// so the two were minimized over different spaces. Raised rather than
    /// returned as an answer because the completion credit a bounded fold prunes
    /// with is built on that corner, and a corner above the true minimum
    /// discards feasible designs.
    #[error("achievable minimum disagrees: corner {corner:?}, folded {folded:?}")]
    AchievableMinimumDisagrees {
        /// What the energy-priority corner reported.
        corner: Energy,
        /// What the recovered front's first point reports, or `None` when the
        /// fold returned nothing at all.
        folded: Option<Energy>,
    },
    /// The beam width must be at least 1.
    #[error("beam width must be >= 1")]
    ZeroBeamWidth,
    /// [`ScanConfig::single_max_len`] is outside the supported range. The bounds
    /// travel in the payload so the message cannot drift from the constants.
    #[error("single-branch loop cap must be in {min}..={max}, got {cap}")]
    LoopCapOutOfRange {
        /// The rejected cap.
        cap: usize,
        /// [`MIN_SINGLE_MAX_LEN`].
        min: usize,
        /// [`DEFAULT_SINGLE_MAX_LEN`].
        max: usize,
    },
    /// A non-default [`ScanConfig::single_max_len`] was combined with a path
    /// that carries its own fixed cap, which would fold two different energy
    /// models into one answer.
    ///
    /// **Whether this fires depends on the protein as well as the constraint.**
    /// The constrained folder is entered only when the constraint actually drops
    /// a lattice edge, which is a property of the (protein, constraint) pair: the
    /// same flags succeed on one protein and are refused on another. The message
    /// says so, because otherwise that reads as a bug.
    #[error(
        "single-branch loop cap is not supported with {0}: that path folds through a component \
         carrying its own fixed cap. Whether it applies depends on the protein as well as the \
         constraint — a set that forbids nothing on this protein takes the ordinary scan and \
         honours the cap"
    )]
    LoopCapUnsupported(&'static str),
    /// The crossing-chain bound was asked for on a lattice it cannot read
    /// exactly; the message names why.
    #[error("crossing-chain bound not supported: {0}")]
    CrossingUnsupported(&'static str),
    /// An input residue is not one of the 20 standard amino acids.
    #[error("invalid amino-acid byte: {0:?}")]
    InvalidAminoAcid(u8),
    /// A per-stop walk was asked of a protein whose stop is not designed: it
    /// must end in `*`.
    #[error("the protein's stop is not designed; per-stop fronts need a protein ending in `*`")]
    StopNotDesigned,
    /// A UTR nucleotide is not one of `A/C/G/U` (`T` accepted as `U`).
    #[error("invalid UTR nucleotide byte: {0:?}")]
    InvalidNucleotide(u8),
    /// A [`UtrProfile`]'s frequencies or strength are outside the domain the
    /// log-odds are defined on.
    #[error("invalid UTR profile: {0}")]
    InvalidUtrProfile(&'static str),
    /// A forbidden motif was malformed (empty, or containing a non-IUPAC code).
    #[error("invalid forbidden motif: {0}")]
    Motif(#[from] ac::MotifError),
    /// The forbidden-motif set admits no encoding of the protein: every coding
    /// sequence contains at least one forbidden motif, so the constrained design
    /// is infeasible. Reported cleanly rather than as a panic or a garbage
    /// optimum.
    #[error("no motif-free encoding of the protein exists under the given motif set")]
    Infeasible,
    /// A [`FlankComplement`] derivation was asked for in a configuration it
    /// cannot serve: an arrangement with no given region to derive from, or
    /// alongside a hand-supplied set or window, which would need two enforcement
    /// windows where one run enforces one. The message says what to do instead.
    #[error("flank-complement derivation: {0}")]
    Derivation(&'static str),
    /// Two constraints contradict each other, or contradict the folding
    /// grammar. Raised before any design work, naming what resolves it (D5).
    #[error(transparent)]
    Constraint(#[from] ConstraintError),
    /// No design satisfies the positional constraints.
    ///
    /// The payload names **which tier** found out, which is the whole point:
    /// an empty scan on the exact lattice is a proof over the design space,
    /// while an empty scan under a beam is a statement about one search. A
    /// caller that collapses the two will discard feasible problems.
    #[error(transparent)]
    NoFeasibleDesign(#[from] Infeasible),
    /// The two ends of a hull gap are not ordered: the first must be strictly
    /// more stable and strictly less adapted than the second, or one weakly
    /// dominates the other and nothing lies between them.
    #[error(
        "hull pair not ordered: the first point must be strictly more stable and strictly \
         less adapted than the second"
    )]
    UnorderedHullPair,
    /// A front walk was given a region constraint that penalizes a pair class
    /// or window rather than forbidding it. The front fold drops forbidden
    /// pairs where they close and has no penalty path.
    #[error("the front walk takes hard region constraints only (forbid, not penalize)")]
    SoftRegionUnsupported,
}

/// Every constrained `P` cell beside the unconstrained outside of its base
/// pair at several λ: the raw material of a pruning-margin survival curve and
/// of a λ-transferred bound. Measurement, not a solver.
#[doc(hidden)]
pub struct MotifMarginDump {
    pub n: usize,
    pub lambda: f64,
    /// The constrained optimum's energy (dcal/mol) and log-CAI.
    pub optimum_mfe: i32,
    pub optimum_cai: f64,
    pub outside_lambdas: Vec<f64>,
    /// The unconstrained optimum at each outside λ, on the BRD scale.
    pub free_optima: Vec<f64>,
    /// Seconds for each outside λ's scan plus outside pass.
    pub outside_seconds: Vec<f64>,
    pub fold_seconds: f64,
    /// `(open, close, mfe, cai, outside per λ)` per `P` cell.
    pub rows: Vec<(u32, u32, i32, f64, Vec<f64>)>,
}

/// Fold the motif-constrained lattice exactly at each of `lambdas`, sharing one
/// set of outside passes, and record every `P` cell with the unconstrained outside of its base pair computed at each
/// of `outside_lambdas`. `None` when the pattern set is infeasible.
#[doc(hidden)]
pub fn motif_margin_dump<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambdas: &[f64],
    outside_lambdas: &[f64],
    motifs: &[S],
) -> Result<Option<Vec<MotifMarginDump>>, BeamError> {
    let aa = residues(protein)?;
    let graph = dfa::build(&aa, usage);
    let ac = ac::Ac::build(motifs)?;
    let product = product::ProductDfa::build(&graph, &ac);
    if !product.is_feasible() {
        return Ok(None);
    }
    let product = product.merge_equivalent();
    let constraint = region::PairConstraint::inert();
    let mut outsides = Vec::new();
    let mut free_optima = Vec::new();
    let mut outside_seconds = Vec::new();
    for &ol in outside_lambdas {
        let started = std::time::Instant::now();
        let tables = scan_or_infeasible(
            params,
            &graph,
            ol,
            ScanConfig::default(),
            constraint.clone(),
            must_pair::MustPairConstraint::inert(),
        )?;
        free_optima.push(tables.c[graph.ix(graph.end())].map_or(f64::INFINITY, |s| s.brd(ol)));
        let costs = outside_scan::OutsideScan::cost_to_go(
            params,
            &graph,
            &tables,
            &constraint,
            ScanConfig::default().single_max_len,
            1,
            false,
        );
        outsides.push(costs.p);
        outside_seconds.push(started.elapsed().as_secs_f64());
    }
    let refs: Vec<&[hash::FxHashMap<state::PairKey, f64>]> =
        outsides.iter().map(|p| p.as_slice()).collect();
    let mut dumps = Vec::new();
    for &lambda in lambdas {
        let started = std::time::Instant::now();
        let Some(dump) = constrained::margin_rows(&product, params, lambda, &refs) else {
            return Ok(None);
        };
        dumps.push(MotifMarginDump {
            n: aa.len(),
            lambda,
            optimum_mfe: dump.mfe.0,
            optimum_cai: dump.cai,
            outside_lambdas: outside_lambdas.to_vec(),
            free_optima: free_optima.clone(),
            outside_seconds: outside_seconds.clone(),
            fold_seconds: started.elapsed().as_secs_f64(),
            rows: dump
                .rows
                .into_iter()
                .map(|r| (r.open, r.close, r.mfe.0, r.cai, r.outside))
                .collect(),
        });
    }
    Ok(Some(dumps))
}

#[cfg(test)]
mod tests {
    use super::{
        design_front_walk, fold_priority, replay_gap, replay_gap_lending, share_threads,
        task_queue, CodonUsage, GapSpec, HullPoint, Params, Slicing, WalkOptions, PROBE,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// A gap's fold asks for a loan once per slice and runs each slice on the
    /// lent workers beside its own, and the front it returns is the front
    /// without the loan.
    #[test]
    fn a_fold_borrows_once_per_slice_and_keeps_its_front() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let protein = "MKTAYIAKQRQISFVKSHFSRQ";
        let walked = design_front_walk(
            &params,
            protein,
            &usage,
            1e-4,
            0.999,
            WalkOptions::default(),
            2,
        )
        .unwrap();
        let widest = walked
            .gaps
            .iter()
            .max_by(|x, y| {
                let width = |g: &&super::WalkGap| g.b.mfe_kcal_mol - g.a.mfe_kcal_mol;
                width(x).total_cmp(&width(y))
            })
            .unwrap();
        let spec = GapSpec {
            a: widest.a,
            b: widest.b,
            lambda: widest.lambda,
            chord: None,
            bracket: Vec::new(),
            own_from: None,
        };
        let slicing = Slicing {
            slack: None,
            width: Some(50.0),
        };
        let plain = replay_gap(&params, protein, &usage, &spec, slicing, 1).unwrap();
        let asked = AtomicUsize::new(0);
        let lent = replay_gap_lending(&params, protein, &usage, &spec, slicing, 1, false, || {
            asked.fetch_add(1, Ordering::SeqCst);
            (1, ())
        })
        .unwrap();
        assert!(plain.slices > 1, "the gap must be cut into slices");
        assert_eq!(asked.load(Ordering::SeqCst), lent.slices);
        assert_eq!(lent.slices, plain.slices);
        // Each slice is clocked once, in ascending ceilings up to the gap's
        // stable end, on the workers it was lent, and its CPU is the fold's.
        for (replayed, workers) in [(&plain, 1), (&lent, 2)] {
            let clocked = &replayed.clock.slices;
            assert_eq!(clocked.len(), replayed.slices);
            assert!(clocked.iter().all(|s| s.count == 1 && s.workers == workers));
            assert!(clocked
                .windows(2)
                .all(|w| w[0].top_kcal_mol < w[1].top_kcal_mol));
            assert_eq!(clocked.last().unwrap().top_kcal_mol, spec.b.mfe_kcal_mol);
            let cpu: f64 = clocked.iter().map(|s| s.cpu_s).sum();
            assert!(cpu > 0.0 && cpu <= replayed.fold_cpu_s + 1e-3);
            let points: usize = clocked.iter().map(|s| s.points).sum();
            assert!(points >= replayed.front.len());
        }
        let objectives = |front: &[super::FrontPoint]| -> Vec<(f64, f64)> {
            front.iter().map(|p| (p.mfe_kcal_mol, p.cai_log)).collect()
        };
        assert_eq!(objectives(&lent.front), objectives(&plain.front));
    }

    /// Slices run at once on a loan, each with the floor known when the batch
    /// starts, return the serial slices' front: on a loan of one worker the
    /// gap goes two slices at a time, on a large loan all at once.
    #[test]
    fn speculative_slices_keep_the_serial_front() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let protein = "MKTAYIAKQRQISFVKSHFSRQ";
        let walked = design_front_walk(
            &params,
            protein,
            &usage,
            1e-4,
            0.999,
            WalkOptions::default(),
            2,
        )
        .unwrap();
        let widest = walked
            .gaps
            .iter()
            .max_by(|x, y| {
                let width = |g: &&super::WalkGap| g.b.mfe_kcal_mol - g.a.mfe_kcal_mol;
                width(x).total_cmp(&width(y))
            })
            .unwrap();
        let spec = GapSpec {
            a: widest.a,
            b: widest.b,
            lambda: widest.lambda,
            chord: None,
            bracket: Vec::new(),
            own_from: None,
        };
        let slicing = Slicing {
            slack: None,
            width: Some(50.0),
        };
        let objectives = |front: &[super::FrontPoint]| -> Vec<(f64, f64)> {
            front.iter().map(|p| (p.mfe_kcal_mol, p.cai_log)).collect()
        };
        let plain = replay_gap(&params, protein, &usage, &spec, slicing, 1).unwrap();
        assert!(plain.slices > 2, "the gap must be cut into several slices");
        for loan in [1, 64] {
            let asked = AtomicUsize::new(0);
            let fast =
                replay_gap_lending(&params, protein, &usage, &spec, slicing, 1, true, || {
                    asked.fetch_add(1, Ordering::SeqCst);
                    (loan, ())
                })
                .unwrap();
            assert_eq!(fast.slices, plain.slices);
            assert_eq!(
                asked.load(Ordering::SeqCst),
                plain.slices.div_ceil(loan + 1)
            );
            assert_eq!(
                objectives(&fast.front),
                objectives(&plain.front),
                "loan {loan}"
            );
        }
    }

    /// The order one worker takes a probe and three folds in: a wide gap at a
    /// low weight (29.2 kcal/mol at 0.0147, slack 0.43), a narrow one at a
    /// high weight (5 at 0.3, slack 1.5) and a middling one (10 at 0.2,
    /// slack 2). By slack the wide gap goes last; by width, first; the probe
    /// leads either way, unless the wide gap is heavy.
    #[test]
    fn folds_queue_by_slack_or_by_width() {
        let gap = |width: f64, lambda: f64| {
            let a = HullPoint {
                mfe_kcal_mol: -100.0,
                adaptation_log: -1.0,
            };
            let b = HullPoint {
                mfe_kcal_mol: -100.0 + width,
                adaptation_log: -0.5,
            };
            (a, b, lambda)
        };
        let gaps = [gap(29.2, 0.0147), gap(5.0, 0.3), gap(10.0, 0.2)];
        let order = |options: &WalkOptions| {
            let mut initial = vec![(PROBE, "probe")];
            initial.extend(
                gaps.iter()
                    .zip(["wide", "narrow", "middling"])
                    .map(|(&(a, b, lambda), name)| (fold_priority(a, b, lambda, options), name)),
            );
            let taken = Mutex::new(Vec::new());
            task_queue::run::<&str, ()>(1, 1, initial, |name, _, _| {
                taken.lock().unwrap().push(name);
                Ok(())
            })
            .unwrap();
            taken.into_inner().unwrap()
        };
        let by_slack = WalkOptions {
            fold_by_width: false,
            ..WalkOptions::default()
        };
        assert_eq!(order(&by_slack), ["probe", "middling", "narrow", "wide"]);
        let by_width = WalkOptions::default();
        assert_eq!(order(&by_width), ["probe", "wide", "middling", "narrow"]);
        let heavy = WalkOptions {
            heavy_fold: Some(2000),
            ..by_width
        };
        assert_eq!(order(&heavy), ["wide", "probe", "middling", "narrow"]);
    }

    /// Slice counts in the fold's integer units. By slack, a wide gap at a
    /// low weight (29.2 kcal/mol at 0.0147, slack 0.43) stays one box and a
    /// narrow one at a high weight (2 kcal/mol at 0.9, slack 1.8) is cut in
    /// four; by width 2.5 kcal/mol, the wide gap is cut in twelve and the
    /// narrow one stays whole. Width overrides slack, and an exact multiple
    /// is not rounded up.
    #[test]
    fn slices_by_slack_or_by_width() {
        let (wide, narrow) = ((0.0147, -112_630, -109_710), (0.9, -50_000, -49_800));
        let count =
            |slicing: Slicing, (lambda, e_a, e_b): (f64, i32, i32)| slicing.count(lambda, e_a, e_b);
        let by_slack = Slicing {
            slack: Some(50.0),
            width: None,
        };
        let by_width = Slicing {
            slack: None,
            width: Some(250.0),
        };
        let both = Slicing {
            slack: Some(50.0),
            width: Some(250.0),
        };
        assert_eq!(count(Slicing::default(), wide), 1);
        assert_eq!(count(by_slack, wide), 1);
        assert_eq!(count(by_slack, narrow), 4);
        assert_eq!(count(by_width, wide), 12);
        assert_eq!(count(by_width, narrow), 1);
        assert_eq!(count(both, wide), 12);
        assert_eq!(count(both, narrow), 1);
        assert_eq!(count(by_width, (0.5, 0, 500)), 2);
        assert_eq!(count(by_width, (0.5, 0, 501)), 3);
        // Never more slices than energy units.
        let fine = Slicing {
            slack: None,
            width: Some(0.1),
        };
        assert_eq!(count(fine, (0.5, 0, 7)), 7);
    }

    /// Jobs come back in job order, and the workers of the jobs running at
    /// once never add up to more than the budget.
    #[test]
    fn shared_threads_keep_order_and_budget() {
        for (threads, jobs) in [(1, 3), (2, 2), (8, 2), (8, 5), (3, 7)] {
            let busy = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);
            let got = share_threads(threads, jobs, |job, workers| {
                let now = busy.fetch_add(workers, Ordering::SeqCst) + workers;
                peak.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(20));
                busy.fetch_sub(workers, Ordering::SeqCst);
                (job, workers)
            });
            let order: Vec<usize> = got.iter().map(|&(job, _)| job).collect();
            assert_eq!(order, (0..jobs).collect::<Vec<_>>(), "{threads}/{jobs}");
            assert!(got.iter().all(|&(_, w)| w >= 1), "{threads}/{jobs}");
            assert!(
                peak.load(Ordering::SeqCst) <= threads,
                "{threads}/{jobs}: {} busy",
                peak.load(Ordering::SeqCst)
            );
        }
    }

    /// A probe cut at its pair's tie value finds the exact probe's design
    /// whenever that design splits the pair, under the exact cost to go and
    /// under a chord between passes around the weight; on an adjacent pair it
    /// finds nothing below the tie value. A cut one unit below the splitting
    /// design's own score loses it, so the cut binds.
    #[test]
    fn a_cut_probe_finds_the_design_that_splits_its_pair() {
        use super::*;
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let protein = "MKTAYIAKQRQISFVKSHFSRQ";
        let graph = dfa::build(&residues(protein).unwrap(), &usage);
        let walked = design_front_walk(
            &params,
            protein,
            &usage,
            1e-4,
            0.999,
            WalkOptions::default(),
            1,
        )
        .unwrap();
        let v = &walked.vertices;
        assert!(v.len() >= 5, "{}", v.len());
        let config = credit_config(1);
        let pass = |lambda: f64| {
            let tables = scan::scan(&params, &graph, lambda, config);
            Arc::new(cost_to_go::Outside::compute_at_any_lambda(
                &params, &graph, &tables, config,
            ))
        };
        let (mut splits, mut lost) = (0, 0);
        for (i, j) in (0..v.len() - 1).flat_map(|i| [(i, i + 1), (i, (i + 3).min(v.len() - 1))]) {
            let (a, b) = (v[i], v[j]);
            let lambda = tie_weight(a, b).unwrap();
            let inert = region::PairConstraint::inert();
            let (exact, _) = hull_vertex(&params, &graph, lambda, config, &inert).unwrap();
            let split = sweep_score(exact, lambda) < sweep_score(a, lambda) - 1e-6;
            let (lo, hi) = ((lambda - 0.05).max(0.0), (lambda + 0.05).min(1.0));
            let bounds = [
                ChordEnds {
                    lo: (lambda, pass(lambda)),
                    hi: (lambda, pass(lambda)),
                },
                ChordEnds {
                    lo: (lo, pass(lo)),
                    hi: (hi, pass(hi)),
                },
            ];
            for ends in &bounds {
                let found = probe_cut(&params, &graph, a, lambda, ends, 1, &inert);
                let found_splits =
                    found.filter(|&p| sweep_score(p, lambda) < sweep_score(a, lambda) - 1e-6);
                if split {
                    assert_eq!(found_splits, Some(exact), "{i}:{j} at {lambda}");
                } else {
                    assert_eq!(found_splits, None, "{i}:{j} at {lambda}");
                }
            }
            if split {
                splits += 1;
                let below = sweep_score(exact, lambda) - sweep_score(a, lambda) - 1.0;
                let tight = probe_cut_at(&params, &graph, a, lambda, &bounds[0], below, 1, &inert);
                lost += usize::from(tight != Some(exact));
            }
        }
        assert!(splits > 0, "no pair split");
        assert_eq!(lost, splits, "a cut below the design's own score kept it");
    }

    /// A point below the chord of its neighbours by more than the probe's
    /// tolerance is a hull vertex and one closer to it is not, so the hull a
    /// whole-folded gap reports agrees with what probes would have found.
    #[test]
    fn a_hull_vertex_is_below_its_chord_by_the_probes_tolerance() {
        use super::{lower_hull, tie_weight, TOL};
        let a = HullPoint {
            mfe_kcal_mol: -100.0,
            adaptation_log: 0.0,
        };
        let b = HullPoint {
            mfe_kcal_mol: -90.0,
            adaptation_log: 0.1,
        };
        let lambda = tie_weight(a, b).expect("an ordered pair");
        // The point halfway between the ends in energy that scores `below`
        // under their tie value.
        let midway = |below: f64| HullPoint {
            mfe_kcal_mol: -95.0,
            adaptation_log: (lambda * 500.0 + below) / (100.0 * (1.0 - lambda)),
        };
        for (below, vertex) in [(10.0 * TOL, true), (TOL / 10.0, false), (0.0, false)] {
            let p = midway(below);
            let hull = lower_hull([a, p, b].into_iter());
            assert_eq!(hull.len() == 3, vertex, "{below} below the chord");
        }
    }

    /// A grid spelled `until:step,...` is the grid of those segments, and a
    /// spelling with a segment that does not parse is no grid.
    #[test]
    fn a_pass_grid_parses_from_its_segments() {
        let thin = super::PassGrid::parse("0.16:0.02, 0.31:0.05,0.71:0.2,1:0.29").unwrap();
        assert_eq!(
            thin.segments,
            vec![(0.16, 0.02), (0.31, 0.05), (0.71, 0.2), (1.0, 0.29)]
        );
        assert_eq!(&thin.weights()[11..], &[0.31, 0.51, 0.71, 1.0]);
        for bad in ["", "0.5", "0.5:x", "0.5:0.1,"] {
            assert_eq!(super::PassGrid::parse(bad), None, "{bad:?}");
        }
    }

    /// A pass grid runs from 0 to 1, each segment's step below its end and
    /// the last step past every end, and reports its widest step.
    #[test]
    fn a_pass_grid_spans_zero_to_one() {
        let grid = super::PassGrid::two_step(0.1, 0.25, 0.3);
        assert_eq!(grid.weights(), vec![0.0, 0.1, 0.2, 0.3, 0.55, 0.8, 1.0]);
        assert!((grid.widest() - 0.25).abs() < 1e-9);
        let placed = super::PassGrid {
            segments: vec![(0.16, 0.02), (0.31, 0.05), (0.71, 0.1), (1.0, 0.15)],
        };
        let weights = placed.weights();
        assert_eq!(weights.len(), 18);
        assert_eq!(&weights[7..12], &[0.14, 0.16, 0.21, 0.26, 0.31]);
        assert_eq!(&weights[14..], &[0.61, 0.71, 0.86, 1.0]);
        assert!((placed.widest() - 0.15).abs() < 1e-9);
        let open_ended = super::PassGrid {
            segments: vec![(0.2, 0.1), (0.5, 0.3)],
        };
        assert_eq!(open_ended.weights(), vec![0.0, 0.1, 0.2, 0.5, 0.8, 1.0]);
    }

    /// The refined grid is the standard grid plus nine weights, all inside
    /// `[0.14, 0.26]`, and the length rule switches between them at its
    /// threshold.
    #[test]
    fn the_length_rule_adds_nine_passes_inside_the_loose_window() {
        use super::{PassGrid, REFINED_GRID_FROM_RESIDUES};
        let standard = PassGrid::standard().weights();
        let refined = PassGrid::refined().weights();
        assert_eq!(standard.len(), 17);
        assert_eq!(refined.len(), 26);
        assert!(standard.iter().all(|w| refined.contains(w)));
        let added: Vec<f64> = refined
            .iter()
            .copied()
            .filter(|w| !standard.contains(w))
            .collect();
        assert_eq!(
            added,
            vec![0.15, 0.17, 0.18, 0.19, 0.2, 0.22, 0.23, 0.24, 0.25]
        );
        assert_eq!(
            PassGrid::for_residues(REFINED_GRID_FROM_RESIDUES - 1),
            PassGrid::standard()
        );
        assert_eq!(
            PassGrid::for_residues(REFINED_GRID_FROM_RESIDUES),
            PassGrid::refined()
        );
    }
}
