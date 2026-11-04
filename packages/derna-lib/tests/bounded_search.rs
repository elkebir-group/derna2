//! The bounded search against exhaustive enumeration.
//!
//! The search claims to return the **exact optimum over modular
//! designs**, and that claim rests on a prune whose soundness is easy to state
//! and easy to violate. The oracle here settles it by brute force: walk the
//! whole design space in objective order with [`Enumerator`], verify every
//! candidate, and take the best modular one. At the lengths used the space is
//! small enough to exhaust, so the comparison is against ground truth rather
//! than against another approximation.
//!
//! Both regimes are covered, because they exercise different halves of the
//! machinery. A **non-empty** modular set exercises the incumbent, the stop rule
//! and the prune; an **empty** one exercises exhaustion, which is the only way
//! emptiness can be established.

use std::time::Duration;

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::batch_probe::BatchProbeReport;
use derna_lib::bound_probe::{cheap_bound, BoundProbeReport, Probed};
use derna_lib::dfa;
use derna_lib::{
    bounded_search, design_restricted, design_restricted_with_region, design_with_utr,
    sequence_gap, BranchOrder, BudgetReadoff, CdsWindow, CodonUsage, Design, Enforcement,
    Enumerator, MarginalKey, Params, Problem, RegionConstraints, ScanConfig, SearchError,
    SearchInputs, SearchOutcome, SearchStats, SeedRejected, TrajectoryEvent,
};

/// A four-residue protein over 144 encodings — small enough to enumerate
/// exhaustively, degenerate enough that the modular set is a proper subset of
/// it. Every case with a non-empty modular set uses this one.
const PROTEIN: &str = "LSKF";

/// A three-residue protein over 24 encodings. The cases that must **exhaust**
/// the region tree use it: exhaustion visits every region rather than stopping
/// at an incumbent, so its cost is the size of the whole space and the fixture
/// has to be correspondingly smaller.
const SMALL_PROTEIN: &str = "LKF";

/// A four-encoding protein starting with methionine — the one residue besides
/// tryptophan with exactly one codon. Under `Positional` order the first
/// partition step excludes that single codon at position 0, leaving that
/// child's lattice empty before any DP work runs.
const MET_FIRST_PROTEIN: &str = "MKF";

/// Flanks that give the fold something to reach across the CDS for. Both are
/// fixed sequence, so what varies between candidates is only the CDS.
const UTR5: &str = "GGGGGG";
const UTR3: &str = "CCCCCC";

const LAMBDA: f64 = 0.5;

/// A budget no test here is meant to hit — every case is expected to finish by
/// the stop rule or by exhaustion, and an outcome saying otherwise is a failure
/// rather than a slow machine.
const AMPLE: Duration = Duration::from_secs(600);

struct Fixture {
    params: Params,
    usage: CodonUsage,
    protein: &'static str,
    aa: Vec<usize>,
    branch_order: BranchOrder,
    trajectory: bool,
    /// A modular design to start the incumbent at, when a case is about what a
    /// seeded walk does.
    seed: Option<String>,
    /// How many regions the walk examines side by side. `1` everywhere but the
    /// cases about the width itself.
    search_threads: usize,
    /// Price a cheap bound beside every exact one. Off everywhere but the cases
    /// about the probe itself.
    bound_probe: bool,
    /// Record what each batch of the width cost. Off everywhere but the cases
    /// about the batches themselves.
    batch_probe: bool,
    /// Key children on their parent's codon max-marginals. Off everywhere but
    /// the cases about that key.
    marginal_key: Option<MarginalKey>,
}

impl Fixture {
    fn new() -> Self {
        Self::over(PROTEIN)
    }

    fn small() -> Self {
        Self::over(SMALL_PROTEIN)
    }

    fn over(protein: &'static str) -> Self {
        Self {
            params: Params::load_default().expect("Turner 2004 parameters"),
            usage: CodonUsage::load_default().expect("the canonical human table"),
            protein,
            aa: protein
                .chars()
                .map(|c| aa_to_index(c).expect("a standard residue"))
                .collect(),
            branch_order: BranchOrder::default(),
            trajectory: false,
            seed: None,
            search_threads: 1,
            bound_probe: false,
            batch_probe: false,
            marginal_key: None,
        }
    }

    fn cds(&self) -> CdsWindow {
        CdsWindow::new(UTR5.len(), self.aa.len())
    }

    fn inputs<'a>(&'a self, constraints: RegionConstraints, budget: Duration) -> SearchInputs<'a> {
        SearchInputs {
            params: &self.params,
            protein: self.protein,
            aa: &self.aa,
            usage: &self.usage,
            lambda: LAMBDA,
            utr5: UTR5,
            utr3: UTR3,
            constraints,
            config: ScanConfig::default(),
            budget,
            branch_order: self.branch_order,
            trajectory: self.trajectory,
            seed: self.seed.as_deref(),
            search_threads: self.search_threads,
            bound_probe: self.bound_probe,
            batch_probe: self.batch_probe,
            marginal_key: self.marginal_key,
        }
    }

    /// The design DP's objective, on the dekacalorie scale the search's bounds
    /// and the enumerator's order both use.
    fn objective(&self, mfe_kcal_mol: f64, cai_log: f64) -> f64 {
        LAMBDA * (mfe_kcal_mol * 100.0).round() - (1.0 - LAMBDA) * (cai_log * 100.0).round()
    }

    /// Every modular design in the whole space, by exhaustion, each with its
    /// objective — the ground truth the search is scored against.
    ///
    /// The enumerator walks in *unconstrained* objective order. That is the
    /// right order to read this off in: for a modular design the constrained
    /// and unconstrained energies coincide, so over the modular set the two
    /// objectives are the same function.
    fn modular_designs(&self, constraints: RegionConstraints) -> Vec<(f64, Design)> {
        let mut walk = Enumerator::new(
            &self.params,
            Problem {
                protein: self.protein,
                aa: &self.aa,
                usage: &self.usage,
                lambda: LAMBDA,
                utr5: UTR5,
                utr3: UTR3,
                threads: 4,
            },
        )
        .expect("the enumerator starts")
        .expect("the protein admits an encoding");
        let mut modular = Vec::new();
        while let Some(design) = walk.next_design().expect("a rank") {
            let gap = sequence_gap(&self.params, &design.sequence, self.cds(), constraints)
                .expect("the verdict");
            if gap.is_modular() {
                modular.push((self.objective(design.mfe_kcal_mol, design.cai_log), design));
            }
        }
        modular
    }
}

/// The two CDS boundaries — the table whose modular set is a proper, non-empty
/// subset of this space.
fn symmetric() -> RegionConstraints {
    RegionConstraints::symmetric_isolation(Enforcement::Forbid)
}

/// Every crossing class. With both flanks a complementary homopolymer pair, no
/// codon assignment can spell the `5′↔3′` duplex away, so the modular set is
/// empty.
fn full() -> RegionConstraints {
    RegionConstraints::full_mutual_isolation(Enforcement::Forbid)
}

#[test]
fn the_search_returns_the_exact_optimum_over_modular_designs() {
    let f = Fixture::new();
    let oracle = f.modular_designs(symmetric());
    assert!(
        !oracle.is_empty(),
        "the fixture is meant to have a non-empty modular set"
    );
    let best_by_oracle = oracle
        .iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .expect("non-empty");

    let report = bounded_search(&f.inputs(symmetric(), AMPLE)).expect("the search runs");

    assert_eq!(
        report.outcome,
        SearchOutcome::Proved,
        "an ample budget must end by the stop rule or exhaustion"
    );
    assert!(report.modular, "the answer must verify modular");
    assert_eq!(
        report.best_objective.expect("a modular incumbent has one"),
        best_by_oracle.0,
        "the search's optimum must match the exhaustive one"
    );
    assert_eq!(
        report.best.expect("an incumbent").sequence,
        best_by_oracle.1.sequence
    );
}

/// Branching order changes what the walk reaches first and nothing about what
/// it can reach, because the tiling is a partition for any ordering of a
/// region's free positions. That is the exactness half of every order on the
/// enum, and it is checkable rather than merely arguable: each must return the
/// design the shipped one does.
///
/// It says nothing about the *tree*, which every order changes — part `k`
/// restricts the first `k + 1` positions of the order, so the regions
/// themselves differ. Only the answer is invariant, which is why that is all
/// this asserts.
#[test]
fn no_branch_order_changes_the_answer() {
    let guided = bounded_search(&Fixture::new().inputs(symmetric(), AMPLE)).expect("it runs");
    // Without a partition the order is never consulted and the comparison below
    // is between a walk and itself.
    assert!(
        guided.stats.regions_pushed > 1,
        "the root was never branched, so no order was exercised"
    );
    let expected = guided.best.as_ref().expect("an incumbent").sequence.clone();

    for order in [
        BranchOrder::Positional,
        BranchOrder::Descending,
        BranchOrder::MoleculeEnds,
    ] {
        let f = Fixture {
            branch_order: order,
            ..Fixture::new()
        };
        let got = bounded_search(&f.inputs(symmetric(), AMPLE)).expect("the search runs");
        assert_eq!(got.best_objective, guided.best_objective, "{order:?}");
        assert_eq!(got.modular, guided.modular, "{order:?}");
        assert_eq!(got.outcome, guided.outcome, "{order:?}");
        assert_eq!(
            got.best.expect("an incumbent").sequence,
            expected,
            "{order:?}"
        );
    }
}

/// [`BranchOrder::MoleculeEnds`] measures distance to an end of the **mRNA**,
/// not of the coding sequence, and takes from whichever end is nearer.
///
/// Worth pinning as behaviour rather than trusting to the sort key, because the
/// order this produces is the reason the placement lever is weaker than it
/// looks: the two are indistinguishable wherever one end dominates, so a
/// version that measured the CDS's ends, or that swept one end only, would pass
/// every other test in this file. Here the UTRs are equal, so the root's first
/// two branch positions must be one from each end.
#[test]
fn the_molecule_end_order_takes_from_whichever_end_is_nearer() {
    let f = Fixture {
        branch_order: BranchOrder::MoleculeEnds,
        bound_probe: true,
        ..Fixture::new()
    };
    let report = bounded_search(&f.inputs(symmetric(), AMPLE)).expect("the search runs");
    let probe = report.bound_probe.expect("the probe was asked for");
    let last = f.aa.len() - 1;
    let first: Vec<&[usize]> = probe
        .rows
        .iter()
        .take(2)
        .map(|r| r.restricted.as_slice())
        .collect();
    assert_eq!(
        first,
        vec![vec![0].as_slice(), vec![0, last].as_slice()],
        "the root's first two children must restrict each end of the molecule"
    );
}

/// The prune is the part that can silently discard the answer, so it gets its
/// own assertion: nothing in the modular set may beat what the search returned.
#[test]
fn no_modular_design_beats_the_search_result() {
    let f = Fixture::new();
    let report = bounded_search(&f.inputs(symmetric(), AMPLE)).expect("the search runs");
    let incumbent = report.best_objective.expect("a modular incumbent");
    for (objective, design) in f.modular_designs(symmetric()) {
        assert!(
            objective >= incumbent,
            "{} is modular at objective {objective}, below the search's {incumbent}",
            design.sequence
        );
    }
}

/// The rank-1 design of this fixture is *not* modular, so the search cannot be
/// passing by never leaving the root.
#[test]
fn the_fixture_is_dirty_at_rank_one() {
    let f = Fixture::new();
    let rank1 = design_with_utr(
        &f.params,
        f.protein,
        &f.usage,
        LAMBDA,
        UTR5,
        UTR3,
        symmetric(),
        ScanConfig::default(),
    )
    .expect("the constrained design");
    let gap = sequence_gap(&f.params, &rank1.sequence, f.cds(), symmetric()).expect("the verdict");
    assert!(!gap.is_modular(), "the fixture must need the search");

    let report = bounded_search(&f.inputs(symmetric(), AMPLE)).expect("the search runs");
    assert_ne!(
        report.best.expect("an incumbent").sequence,
        rank1.sequence,
        "the answer must differ from the design Layer 0 returns"
    );
    assert!(
        report.stats.solves > 1,
        "a dirty root must cost more than the root solve"
    );
}

#[test]
fn an_empty_modular_set_is_established_by_exhaustion() {
    let f = Fixture::small();
    assert!(
        f.modular_designs(full()).is_empty(),
        "the fixture is meant to have an empty modular set"
    );

    let report = bounded_search(&f.inputs(full(), AMPLE)).expect("the search runs");

    assert_eq!(
        report.outcome,
        SearchOutcome::Proved,
        "exhausting the heap proves the modular set empty"
    );
    assert!(!report.modular);
    assert_eq!(report.best_objective, None, "there is no modular optimum");
    assert!(
        report.best.is_some(),
        "an empty modular set still returns the constrained optimum as a best effort"
    );
}

/// Against an empty modular set the incumbent never rises above `+∞`, so no
/// region is ever discarded by the bound. Every pushed region must therefore be
/// solved, which is the case the search's own docs call a region-granular
/// filter rather than a branch-and-bound.
#[test]
fn nothing_is_pruned_before_an_incumbent_exists() {
    let f = Fixture::small();
    let report = bounded_search(&f.inputs(full(), AMPLE)).expect("the search runs");
    assert_eq!(
        report.stats.regions_pruned_unsolved, 0,
        "with no incumbent the bound cannot fire"
    );
    assert_eq!(report.stats.regions_pushed, report.stats.solves);
}

/// A deterministic non-zero floor for [`derna_lib::SearchStats::regions_empty`]:
/// under `Positional` order the very first partition step excludes the start
/// codon's only codon, so that child's lattice is empty on the first refine.
/// `full()` forces exhaustion, guaranteeing the walk reaches and refines that
/// region rather than leaving it queued.
#[test]
fn an_empty_lattice_is_counted_apart_from_a_real_solve() {
    let f = Fixture {
        branch_order: BranchOrder::Positional,
        ..Fixture::over(MET_FIRST_PROTEIN)
    };
    let report = bounded_search(&f.inputs(full(), AMPLE)).expect("the search runs");
    assert_eq!(report.outcome, SearchOutcome::Proved);
    assert!(
        report.stats.regions_empty >= 1,
        "excluding the start codon's only codon must empty a region"
    );
    assert!(
        report.stats.regions_empty < report.stats.solves,
        "an empty-lattice result is one particular kind of solve() call, not all of them"
    );
}

/// The non-vacuity control for the case above. Emptiness is not particular to
/// a single-codon residue — it is what accumulating exclusions across enough
/// tree depth eventually does to *any* position, which is why exhausting an
/// empty modular set turns up plenty of it (`LKF`/`full()` alone hits 26 of 50
/// solves). The genuine zero is a search that proves quickly via the stop
/// rule, over a non-empty modular set, without exhausting anything.
#[test]
fn a_quick_proof_over_a_non_empty_modular_set_hits_no_empty_lattice() {
    let f = Fixture::new();
    let report = bounded_search(&f.inputs(symmetric(), AMPLE)).expect("the search runs");
    assert_eq!(report.outcome, SearchOutcome::Proved);
    assert_eq!(
        report.stats.regions_empty, 0,
        "a quick stop-rule proof should never need to exhaust a position's codons"
    );
}

#[test]
fn an_exhausted_budget_returns_an_unproven_best_effort() {
    let f = Fixture::new();
    let report = bounded_search(&f.inputs(symmetric(), Duration::ZERO)).expect("the search runs");
    assert_eq!(report.outcome, SearchOutcome::BudgetNotModular);
    assert!(!report.modular);
    assert!(
        !report.outcome.is_proved(),
        "a budget exit certifies nothing"
    );
    assert!(
        report.best.is_some(),
        "the budget returns the best design found, not nothing"
    );
    assert_eq!(report.stats.solves, 1, "only the root was ever solved");
}

/// A table nothing violates makes every design modular, so the root's optimum is
/// already the answer and the stop rule fires on the first child popped.
#[test]
fn a_table_with_no_dirty_designs_is_answered_at_the_root() {
    let f = Fixture::new();
    let inert = RegionConstraints::default();
    let report = bounded_search(&f.inputs(inert, AMPLE)).expect("the search runs");

    assert_eq!(report.outcome, SearchOutcome::Proved);
    assert!(report.modular);
    assert_eq!(
        report.stats.solves, 1,
        "Layer 0 answers: one solve, and the bound stops the rest"
    );
    let unconstrained = design_with_utr(
        &f.params,
        f.protein,
        &f.usage,
        LAMBDA,
        UTR5,
        UTR3,
        inert,
        ScanConfig::default(),
    )
    .expect("the design");
    assert_eq!(
        report.best.expect("an incumbent").sequence,
        unconstrained.sequence
    );
    assert!(
        report.stats.regions_pruned_unsolved > 0,
        "the children the stop rule left queued are pruned unsolved"
    );
}

/// The restricted solve gained a region constraint; with an inert table it must be
/// the function it was before, which is what keeps the enumerator's walk — and
/// every measurement taken over it — unchanged.
#[test]
fn a_restricted_solve_under_an_inert_table_is_the_unconstrained_one() {
    let f = Fixture::new();
    let allowed: Vec<Vec<[u8; 3]>> =
        f.aa.iter()
            .map(|&a| AMINO_ACIDS[a].codons.iter().map(|c| c.nuc).collect())
            .collect();
    let plain = design_restricted(
        &f.params,
        f.protein,
        &f.usage,
        LAMBDA,
        &allowed,
        UTR5,
        UTR3,
        ScanConfig::default(),
    )
    .expect("the solve")
    .expect("a feasible lattice");
    let constrained = design_restricted_with_region(
        &f.params,
        f.protein,
        &f.usage,
        LAMBDA,
        &allowed,
        UTR5,
        UTR3,
        RegionConstraints::default(),
        ScanConfig::default(),
    )
    .expect("the solve")
    .expect("a feasible lattice");
    assert_eq!(plain.sequence, constrained.sequence);
    assert_eq!(plain.structure, constrained.structure);
    assert_eq!(plain.mfe_kcal_mol, constrained.mfe_kcal_mol);
    assert_eq!(plain.cai_log, constrained.cai_log);
}

/// An unrestricted lattice under an active table must reproduce the ordinary
/// constrained design — the other end of the same floor, and the reason a
/// region bound is comparable with an incumbent at all.
#[test]
fn an_unrestricted_lattice_under_a_table_is_the_ordinary_constrained_design() {
    let f = Fixture::new();
    let allowed: Vec<Vec<[u8; 3]>> =
        f.aa.iter()
            .map(|&a| AMINO_ACIDS[a].codons.iter().map(|c| c.nuc).collect())
            .collect();
    let restricted = design_restricted_with_region(
        &f.params,
        f.protein,
        &f.usage,
        LAMBDA,
        &allowed,
        UTR5,
        UTR3,
        symmetric(),
        ScanConfig::default(),
    )
    .expect("the solve")
    .expect("a feasible lattice");
    let direct = design_with_utr(
        &f.params,
        f.protein,
        &f.usage,
        LAMBDA,
        UTR5,
        UTR3,
        symmetric(),
        ScanConfig::default(),
    )
    .expect("the design");
    assert_eq!(restricted.sequence, direct.sequence);
    assert_eq!(restricted.mfe_kcal_mol, direct.mfe_kcal_mol);
    assert_eq!(restricted.cai_log, direct.cai_log);
}

/// The trajectory must not change what the search decides.
///
/// This is the assertion the read-off rests on. Every budget below the one a
/// run was given is reported from that run's trajectory rather than measured by
/// a run of its own, and that substitution is only legitimate if recording
/// changes nothing about the walk. Anything on the logging path that
/// recomputed, re-folded or re-scored would show up here.
#[test]
fn the_trajectory_does_not_change_the_verdict() {
    let logged = bounded_search(
        &Fixture {
            trajectory: true,
            ..Fixture::new()
        }
        .inputs(symmetric(), AMPLE),
    )
    .expect("the search runs");
    let unlogged =
        bounded_search(&Fixture::new().inputs(symmetric(), AMPLE)).expect("the search runs");

    assert_eq!(logged.outcome, unlogged.outcome);
    assert_eq!(logged.modular, unlogged.modular);
    assert_eq!(logged.best_objective, unlogged.best_objective);
    assert_eq!(
        logged.best.expect("an incumbent").sequence,
        unlogged.best.expect("an incumbent").sequence
    );
    assert!(
        unlogged.trajectory.is_empty(),
        "a run that did not ask for a trajectory must not carry one"
    );
}

/// The shape a read-off depends on: the root solve first, then improvements, in
/// clock order and strictly better each time.
///
/// Without strict improvement a read-off would have to say which of two equal
/// incumbents the walk was holding; without clock order the prefix at a budget
/// would not be a prefix.
#[test]
fn the_trajectory_is_ordered_and_strictly_improving() {
    let report = bounded_search(
        &Fixture {
            trajectory: true,
            ..Fixture::new()
        }
        .inputs(symmetric(), AMPLE),
    )
    .expect("the search runs");

    let (first, rest) = report
        .trajectory
        .split_first()
        .expect("the root solve is always recorded");
    assert_eq!(first.event, TrajectoryEvent::RootSolved);
    assert!(
        !first.modular,
        "the root's optimum is a best effort, not an incumbent"
    );
    assert!(
        !rest.is_empty(),
        "the fixture is meant to reach a modular incumbent"
    );

    for pair in report.trajectory.windows(2) {
        assert!(
            pair[1].elapsed_s >= pair[0].elapsed_s,
            "instants come from one monotone clock"
        );
        assert!(
            pair[1].regions_popped >= pair[0].regions_popped,
            "work only accumulates"
        );
    }
    for r in rest {
        assert_eq!(r.event, TrajectoryEvent::Incumbent);
        assert!(r.modular, "only a modular design becomes the incumbent");
    }
    for pair in rest.windows(2) {
        assert!(
            pair[1].objective < pair[0].objective,
            "the incumbent's single update site is constrained by strict improvement"
        );
    }
    assert_eq!(
        rest.last().expect("an incumbent").objective,
        report.best_objective.expect("a modular incumbent has one"),
        "the last improvement is what the run finished holding"
    );
}

/// At and above the budget actually spent, a read-off is the terminal report —
/// and below it, never `Proved`.
///
/// `Proved` is a statement about the whole space, known only when the stop rule
/// fires or the heap empties. An instant mid-walk cannot report it however good
/// the incumbent it holds, so a curve of read-offs must show the proof arriving
/// exactly once, at the end.
#[test]
fn a_read_off_reports_the_terminal_verdict_only_at_the_end() {
    let report = bounded_search(
        &Fixture {
            trajectory: true,
            ..Fixture::new()
        }
        .inputs(symmetric(), AMPLE),
    )
    .expect("the search runs");
    assert_eq!(report.outcome, SearchOutcome::Proved);

    let terminal = BudgetReadoff::Held {
        outcome: report.outcome,
        objective: report.best_objective,
        modular: report.modular,
        regions_popped: report.stats.regions_popped,
    };
    assert_eq!(report.at_budget(report.elapsed_s), terminal);
    assert_eq!(report.at_budget(report.elapsed_s * 2.0), terminal);

    let last = *report.trajectory.last().expect("an incumbent");
    // Cut off after its last improvement, the walk held the same design — it
    // simply had not yet finished proving it optimal.
    assert_eq!(
        report.at_budget(last.elapsed_s),
        BudgetReadoff::Held {
            outcome: SearchOutcome::BudgetModular,
            objective: Some(last.objective),
            modular: true,
            regions_popped: last.regions_popped,
        }
    );
    assert_eq!(
        last.objective,
        report.best_objective.expect("a modular incumbent has one"),
        "the design is the same one; only the certificate differs"
    );
}

/// A budget below the root solve's cost was never honoured, and that is not the
/// same fact as the search failing in the time given.
///
/// `run()` solves the root before entering the loop and never consults the
/// budget during it, so a run given less than the root costs still pays for the
/// whole root solve. A column of read-offs that called such an entry a failure
/// would be reporting the root solve's price as the search's.
#[test]
fn a_budget_below_the_root_solve_was_never_honoured() {
    let f = Fixture {
        trajectory: true,
        ..Fixture::new()
    };
    let ample = bounded_search(&f.inputs(symmetric(), AMPLE)).expect("the search runs");
    let root = *ample
        .trajectory
        .first()
        .expect("the root solve is recorded");
    assert!(root.elapsed_s > 0.0, "the root solve costs something");
    assert_eq!(
        ample.at_budget(root.elapsed_s / 2.0),
        BudgetReadoff::NotHonoured
    );

    // What such a run actually does, measured rather than argued: it pays for
    // the root solve, records it, and only then finds the budget gone.
    let zero = bounded_search(&f.inputs(symmetric(), Duration::ZERO)).expect("the search runs");
    assert_eq!(zero.outcome, SearchOutcome::BudgetNotModular);
    assert_eq!(zero.stats.solves, 1, "only the root was ever solved");
    assert_eq!(
        zero.trajectory.len(),
        1,
        "the root solve, and nothing after it"
    );
    assert!(
        zero.trajectory[0].elapsed_s > 0.0,
        "a zero budget still paid for the whole root solve"
    );
}

// --- Seeding -----------------------------------------------------------
//
// Until a modular design exists the incumbent is `+∞` and no bound can reach
// it, so a walk that never finds one prunes nothing. A seed removes that phase
// by starting the incumbent at a design somebody else found. What must not
// change is what the search certifies, and that is what these settle: a seeded
// walk is scored against the same exhaustive oracle as an unseeded one, and a
// seed that cannot legally be an incumbent is refused rather than trusted.

/// The modular designs of the fixture, best first — the oracle both the seeded
/// and the unseeded cases are scored against.
fn modular_ranked(f: &Fixture, constraints: RegionConstraints) -> Vec<(f64, Design)> {
    let mut modular = f.modular_designs(constraints);
    modular.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert!(modular.len() > 1, "a seeding case needs a choice of seeds");
    modular
}

/// A fixture seeded with `sequence`, everything else as [`Fixture::new`].
fn seeded(sequence: String) -> Fixture {
    Fixture {
        seed: Some(sequence),
        ..Fixture::new()
    }
}

/// The soundness claim, seeded: the answer is still the exhaustive optimum over
/// the modular set, not the seed the caller happened to hand in. The seed used is
/// the **worst** modular design in the space, so a walk that merely kept what it
/// was given would be visibly wrong rather than accidentally right.
#[test]
fn a_seeded_search_returns_the_exact_optimum_over_modular_designs() {
    let f = Fixture::new();
    let oracle = modular_ranked(&f, symmetric());
    let (best_objective, best) = oracle.first().expect("non-empty").clone();
    let worst = oracle.last().expect("non-empty").1.sequence.clone();
    assert_ne!(
        worst, best.sequence,
        "the seed must not already be the answer"
    );

    let g = seeded(worst);
    let report = bounded_search(&g.inputs(symmetric(), AMPLE)).expect("the search runs");

    assert!(report.stats.seeded, "the seed must have been adopted");
    assert_eq!(report.outcome, SearchOutcome::Proved);
    assert!(report.modular);
    assert_eq!(report.best_objective.expect("an incumbent"), best_objective);
    assert_eq!(report.best.expect("an incumbent").sequence, best.sequence);
}

/// Seeding changes what the walk reaches first and nothing about what it can
/// reach — the same claim the branch order carries, and checkable the same way.
#[test]
fn a_seed_does_not_change_the_answer() {
    let f = Fixture::new();
    let unseeded = bounded_search(&f.inputs(symmetric(), AMPLE)).expect("the search runs");
    let oracle = modular_ranked(&f, symmetric());

    for (_, candidate) in [oracle.first().unwrap(), oracle.last().unwrap()] {
        let g = seeded(candidate.sequence.clone());
        let report = bounded_search(&g.inputs(symmetric(), AMPLE)).expect("the search runs");
        assert_eq!(report.outcome, unseeded.outcome);
        assert_eq!(report.best_objective, unseeded.best_objective);
        assert_eq!(
            report.best.expect("an incumbent").sequence,
            unseeded.best.clone().expect("an incumbent").sequence,
            "seeded with {}, the walk returned a different design",
            candidate.sequence
        );
    }
}

/// An optimal seed is still only a seed: the search has to exhaust or reach the
/// stop rule before it can say `Proved`, and the bound is what gets it there.
/// The prune is asserted rather than assumed, because a walk that discarded
/// nothing would be reporting a proof it had not done.
#[test]
fn an_optimal_seed_is_proved_by_the_bound_rather_than_believed() {
    let f = Fixture::new();
    let oracle = modular_ranked(&f, symmetric());
    let (objective, best) = oracle.first().expect("non-empty").clone();

    let g = seeded(best.sequence.clone());
    let report = bounded_search(&g.inputs(symmetric(), AMPLE)).expect("the search runs");

    assert_eq!(report.outcome, SearchOutcome::Proved);
    assert_eq!(report.best_objective.expect("an incumbent"), objective);
    assert_eq!(report.best.expect("an incumbent").sequence, best.sequence);
    assert!(
        report.stats.regions_pruned_unsolved + report.stats.regions_discarded_after_refine > 0,
        "an optimal incumbent from the first pop must discard something: {:?}",
        report.stats
    );
}

/// The bound is a lower bound only over the *modular* members of a region, and a
/// violating incumbent is what makes it invalid — its free-fold objective can
/// sit below bounds computed on the constrained one, and regions holding the
/// true optimum would be discarded against it. So a violating seed is refused,
/// not quietly ignored: a caller who passes one has a broken design, and a
/// search that proceeded would return a wrong answer labelled `Proved`.
#[test]
fn a_violating_seed_is_refused() {
    let f = Fixture::new();
    let dirty = design_with_utr(
        &f.params,
        f.protein,
        &f.usage,
        LAMBDA,
        UTR5,
        UTR3,
        symmetric(),
        ScanConfig::default(),
    )
    .expect("the constrained design");
    assert!(
        !sequence_gap(&f.params, &dirty.sequence, f.cds(), symmetric())
            .expect("the verdict")
            .is_modular(),
        "this fixture's rank-1 design is the violating one"
    );

    let g = seeded(dirty.sequence.clone());
    let err = bounded_search(&g.inputs(symmetric(), AMPLE)).expect_err("a violating seed");
    assert!(
        matches!(
            err,
            SearchError::Seed(SeedRejected::Violating { gap }) if gap > 0.0
        ),
        "expected a violating-seed rejection, got {err}"
    );
}

/// A seed is a design of *this* space or it is nothing. Its energy is compared
/// against bounds computed over this protein, these flanks and this codon
/// table, so a sequence from a different instance is refused on the way in
/// rather than compared on the way through.
#[test]
fn a_seed_from_another_problem_is_refused() {
    let f = Fixture::new();
    let modular = modular_ranked(&f, symmetric())
        .first()
        .expect("non-empty")
        .1
        .sequence
        .clone();
    let cds = f.cds();

    // Too short by one codon: not a molecule of this space at all.
    let truncated = modular[..modular.len() - 3].to_string();
    // This protein's residues, spelled in a different instance's flanks.
    let reflanked = format!("AAAAAA{}", &modular[UTR5.len()..]);
    // A codon this protein's first residue does not admit.
    let mut mistranslated = modular.clone().into_bytes();
    mistranslated[cds.start..cds.start + 3].copy_from_slice(b"AAA");
    let mistranslated = String::from_utf8(mistranslated).expect("ASCII");
    // Not a nucleotide.
    let mut nonsense = modular.clone().into_bytes();
    nonsense[cds.start] = b'X';
    let nonsense = String::from_utf8(nonsense).expect("ASCII");

    for (label, seed) in [
        ("truncated", truncated),
        ("reflanked", reflanked),
        ("mistranslated", mistranslated),
        ("nonsense", nonsense),
    ] {
        let g = seeded(seed);
        let err = bounded_search(&g.inputs(symmetric(), AMPLE))
            .expect_err(&format!("a {label} seed must be refused"));
        assert!(
            matches!(err, SearchError::Seed(_)),
            "a {label} seed gave {err}"
        );
    }
    assert_eq!(
        UTR5.len(),
        6,
        "the reflanked case assumes this flank length"
    );

    // The control: the unmodified sequence is accepted, so the four refusals
    // above are the modifications and not the fixture.
    let g = seeded(modular);
    assert!(bounded_search(&g.inputs(symmetric(), AMPLE)).is_ok());
}

/// A `T`-spelled seed is the same molecule, so it is read rather than refused —
/// the design DP's own flank parsing reads `T` as `U` and a seed comes from the
/// same world as a flank.
#[test]
fn a_dna_spelled_seed_is_the_same_seed() {
    let f = Fixture::new();
    let modular = modular_ranked(&f, symmetric())
        .first()
        .expect("non-empty")
        .1
        .sequence
        .clone();
    let rna = bounded_search(&seeded(modular.clone()).inputs(symmetric(), AMPLE))
        .expect("the search runs");
    let dna = bounded_search(&seeded(modular.replace('U', "T")).inputs(symmetric(), AMPLE))
        .expect("the search runs");
    assert_eq!(rna.best_objective, dna.best_objective);
    assert_eq!(
        rna.best.expect("an incumbent").sequence,
        dna.best.expect("an incumbent").sequence
    );
}

/// The read-off's `NotHonoured` rule is about the **root solve**, which a
/// seeded walk still pays for in full: the root is solved before the seed is
/// adopted and the budget is not consulted during it. A budget below that
/// instant therefore stays un-honoured however good the seed handed in, and the
/// walk's first recorded instant is still the root's.
#[test]
fn a_seed_does_not_honour_a_budget_below_the_root_solve() {
    let f = Fixture::new();
    let modular = modular_ranked(&f, symmetric())
        .first()
        .expect("non-empty")
        .1
        .sequence
        .clone();
    let g = Fixture {
        seed: Some(modular),
        trajectory: true,
        ..Fixture::new()
    };
    let report = bounded_search(&g.inputs(symmetric(), AMPLE)).expect("the search runs");

    assert_eq!(
        report.trajectory.first().expect("a recorded walk").event,
        TrajectoryEvent::RootSolved,
        "the root's instant precedes the seed's"
    );
    assert_eq!(report.at_budget(-1.0), BudgetReadoff::NotHonoured);
    assert!(
        report
            .trajectory
            .iter()
            .any(|r| r.event == TrajectoryEvent::Incumbent && r.modular),
        "the adopted seed is an instant on the walk"
    );
}

// --- Width -------------------------------------------------------------
//
// The regions on the heap are independent restricted solves, so examining
// several side by side is the one lever on the wall-clock of a *proof*: a
// best-first walk must open every region whose bound falls below the optimum,
// a set fixed by the bound and the answer, so neither a better incumbent nor a
// better order can shrink it. What must not move is the answer, and these
// settle that it does not — every decision is made after the examining, in the
// order the entries were popped, so a wide walk passes through exactly the
// incumbents a narrow one does.

/// A fixture examining `width` regions at a time, everything else as
/// [`Fixture::new`].
fn at_width(width: usize) -> Fixture {
    Fixture {
        search_threads: width,
        ..Fixture::new()
    }
}

/// What the width cannot reach: same design, same objective, same verdict at
/// every width, on the fixture whose modular set is a proper non-empty subset of
/// the space — so the walk genuinely branches, prunes and stops by the rule.
#[test]
fn the_width_does_not_change_the_answer() {
    let serial = bounded_search(&at_width(1).inputs(symmetric(), AMPLE)).expect("the search runs");
    assert_eq!(
        serial.outcome,
        SearchOutcome::Proved,
        "the control must prove, or the case is not about the width"
    );
    for width in [2, 3, 4, 8, 16] {
        let wide =
            bounded_search(&at_width(width).inputs(symmetric(), AMPLE)).expect("the search runs");
        assert_eq!(wide.outcome, serial.outcome, "at width {width}");
        assert_eq!(
            wide.best_objective, serial.best_objective,
            "at width {width}"
        );
        assert_eq!(wide.modular, serial.modular, "at width {width}");
        assert_eq!(
            wide.best.expect("an incumbent").sequence,
            serial.best.clone().expect("an incumbent").sequence,
            "at width {width}"
        );
    }
}

/// Exhaustion is the other terminal exit, and it is the one a width could break
/// differently: there is no incumbent to compare against, so every region in the
/// tree is examined and the walk ends only when the heap empties. An empty
/// modular set must still be established, and still by exhausting.
#[test]
fn the_width_does_not_change_an_exhaustive_verdict() {
    let serial = bounded_search(&Fixture::small().inputs(full(), AMPLE)).expect("the search runs");
    assert_eq!(serial.outcome, SearchOutcome::Proved);
    assert!(
        !serial.modular,
        "the fixture's modular set is meant to be empty"
    );
    for width in [2, 5] {
        let wide = bounded_search(&SearchInputs {
            search_threads: width,
            ..Fixture::small().inputs(full(), AMPLE)
        })
        .expect("the search runs");
        assert_eq!(wide.outcome, serial.outcome, "at width {width}");
        assert_eq!(wide.modular, serial.modular, "at width {width}");
        assert_eq!(wide.stats.solves, serial.stats.solves, "at width {width}");
    }
}

/// A width of 1 is the serial walk *byte for byte*, counters included — which is
/// the claim that lets the default stay unmeasured-but-safe. Wider walks are
/// allowed to differ in the counters and are checked separately for it below.
#[test]
fn a_width_of_one_is_the_serial_walk() {
    let default = bounded_search(&Fixture::new().inputs(symmetric(), AMPLE)).expect("runs");
    let explicit = bounded_search(&at_width(1).inputs(symmetric(), AMPLE)).expect("runs");
    assert_eq!(default.stats, explicit.stats);
    assert_eq!(default.best_objective, explicit.best_objective);
}

/// What a width *is* allowed to change, stated so it is not mistaken for a bug:
/// a batch is taken against one incumbent, so a wide walk can solve a region a
/// serial walk would have stopped before reaching. Those land in
/// `regions_discarded_after_refine`, so the counters may grow with the width
/// while the answer does not — which is why the width is opt-in.
#[test]
fn a_width_may_cost_solves_a_serial_walk_would_not_have_made() {
    let serial = bounded_search(&at_width(1).inputs(symmetric(), AMPLE)).expect("runs");
    let wide = bounded_search(&at_width(8).inputs(symmetric(), AMPLE)).expect("runs");
    assert!(
        wide.stats.solves >= serial.stats.solves,
        "a wide walk cannot examine fewer regions than a serial one: {:?} against {:?}",
        wide.stats,
        serial.stats
    );
    assert_eq!(
        wide.best.expect("an incumbent").sequence,
        serial.best.expect("an incumbent").sequence,
        "and it pays that in counters, never in the answer"
    );
}

/// The two refinements compose: a seeded walk at width still returns the
/// exhaustive optimum. They touch the same state from opposite sides — a seed
/// sets the incumbent before the loop, a width defers when the loop reads it —
/// so the pair is worth one case of its own.
#[test]
fn a_seeded_walk_at_width_still_returns_the_exact_optimum() {
    let f = Fixture::new();
    let oracle = modular_ranked(&f, symmetric());
    let (best_objective, best) = oracle.first().expect("non-empty").clone();
    let worst = oracle.last().expect("non-empty").1.sequence.clone();

    let report = bounded_search(&SearchInputs {
        seed: Some(&worst),
        search_threads: 4,
        ..Fixture::new().inputs(symmetric(), AMPLE)
    })
    .expect("the search runs");

    assert!(report.stats.seeded);
    assert_eq!(report.outcome, SearchOutcome::Proved);
    assert_eq!(report.best_objective.expect("an incumbent"), best_objective);
    assert_eq!(report.best.expect("an incumbent").sequence, best.sequence);
}

// ── Pricing a cheap bound ──────────────────────────────────────────────────
//
// A region's key costs one whole-mRNA solve, and a bound readable from the
// region itself could reject it before that solve — the only remaining way to
// make a proof shorter, since the set of regions a proof must open is fixed by
// the bound and the answer. These settle that the instrument measuring that
// route is trustworthy: that it does not disturb the walk it measures, that
// the adaptation ceiling is on the objective's own scale, and that the bounds
// composed from a real floor are under the exact bounds they claim to be.

/// A fixture pricing a cheap bound beside every exact one, everything else as
/// [`Fixture::new`].
fn probed() -> Fixture {
    Fixture {
        bound_probe: true,
        ..Fixture::new()
    }
}

/// The probe of a walk that branches, prunes and stops by the rule.
fn probe_of(fixture: &Fixture) -> BoundProbeReport {
    bounded_search(&fixture.inputs(symmetric(), AMPLE))
        .expect("the search runs")
        .bound_probe
        .expect("a probed walk reports its rows")
}

/// Probing records a walk; it does not change one. The answer, the verdict and
/// every counter come back identical — which is what lets a probed run price a
/// bound for the walk that ships rather than for a different one.
#[test]
fn probing_a_walk_changes_nothing_about_it() {
    let plain = bounded_search(&Fixture::new().inputs(symmetric(), AMPLE)).expect("runs");
    let probed = bounded_search(&probed().inputs(symmetric(), AMPLE)).expect("runs");

    assert_eq!(probed.stats, plain.stats);
    assert_eq!(probed.outcome, plain.outcome);
    assert_eq!(probed.best_objective, plain.best_objective);
    assert_eq!(
        probed.best.expect("an incumbent").sequence,
        plain.best.expect("an incumbent").sequence
    );
    assert!(
        plain.bound_probe.is_none(),
        "and stays off unless asked for"
    );
}

/// Every bound composed from a real floor sits under the exact bound the same
/// region's solve returned. This is the property a prune would rest on: a
/// cheap bound above a region's own optimum discards that optimum.
///
/// The parent's trade-off optimum is deliberately not asserted on. It is not a
/// floor, and whether it behaves like one is the measurement, not an invariant.
#[test]
fn a_bound_composed_from_a_floor_never_exceeds_the_exact_one() {
    let probe = probe_of(&probed());
    let mut solved = 0;
    for row in &probe.rows {
        let Probed::Solved { exact, .. } = row.probed else {
            continue;
        };
        solved += 1;
        let bounds = row
            .bounds(LAMBDA)
            .expect("a solved region admits an encoding");
        assert!(
            bounds.space <= exact,
            "the space's floor composed to {} against an exact {exact}",
            bounds.space
        );
        assert!(
            bounds.parent <= exact,
            "the parent's floor composed to {} against an exact {exact}",
            bounds.parent
        );
    }
    assert!(
        solved > 1,
        "the case has to solve regions below the root, or it asserts nothing: {solved}"
    );
    assert!(probe.floor_solves > 0, "and has to have paid for a floor");
}

/// The adaptation ceiling detects an empty region in one pass over the
/// positions, where the walk detects it by spending a solve. Any region the
/// walk found empty must have had no ceiling to report.
#[test]
fn an_empty_region_has_no_adaptation_ceiling() {
    let fixture = Fixture {
        branch_order: BranchOrder::Positional,
        bound_probe: true,
        ..Fixture::over(MET_FIRST_PROTEIN)
    };
    let probe = bounded_search(&fixture.inputs(full(), AMPLE))
        .expect("the search runs")
        .bound_probe
        .expect("a probed walk reports its rows");

    let empty: Vec<_> = probe
        .rows
        .iter()
        .filter(|row| row.probed == Probed::Empty)
        .collect();
    assert!(
        !empty.is_empty(),
        "the case has to reach an empty lattice, or it asserts nothing"
    );
    for row in empty {
        assert_eq!(row.cai_ceiling, None);
    }
}

/// The adaptation ceiling is on the objective's own scale, which is what makes
/// a composed bound comparable to an incumbent at all.
///
/// At λ = 0 the objective *is* adaptation, so a region's exact bound and the
/// ceiling over it agree to within the dekacalorie the composition rounds away.
/// The case pins one position to its worst synonymous codon so the ceiling is
/// strictly negative: over an unrestricted lattice it is zero and the agreement
/// would hold by arithmetic rather than by the ceiling being right.
#[test]
fn the_adaptation_ceiling_is_the_exact_bound_where_the_objective_is_adaptation() {
    let f = Fixture::new();
    let mut allowed: Vec<Vec<[u8; 3]>> =
        f.aa.iter()
            .map(|&aa| AMINO_ACIDS[aa].codons.iter().map(|c| c.nuc).collect())
            .collect();
    let first = f.aa[0];
    let worst = (0..AMINO_ACIDS[first].codons.len())
        .min_by(|&a, &b| {
            f.usage
                .fraction(first, a)
                .total_cmp(&f.usage.fraction(first, b))
        })
        .expect("a residue has at least one codon");
    allowed[0] = vec![AMINO_ACIDS[first].codons[worst].nuc];

    let ceiling = dfa::cai_ceiling(&f.aa, &f.usage, &allowed).expect("a non-empty lattice");
    assert!(
        ceiling < 0.0,
        "the restriction has to cost adaptation, or the case is vacuous: {ceiling}"
    );

    let design = design_restricted_with_region(
        &f.params,
        f.protein,
        &f.usage,
        0.0,
        &allowed,
        UTR5,
        UTR3,
        symmetric(),
        ScanConfig::default(),
    )
    .expect("the solve runs")
    .expect("a non-empty lattice");

    let exact = -(design.cai_log * 100.0).round();
    let cheap = cheap_bound(0.0, design.mfe_kcal_mol, ceiling);
    assert!(cheap <= exact, "{cheap} is not under {exact}");
    assert!(
        exact - cheap <= 1.0,
        "the ceiling is the exact bound here, up to the dekacalorie the          composition rounds away: {cheap} against {exact}"
    );
}

// ── What the width loses ───────────────────────────────────────────────────
//
// Each worker takes exactly one entry, so there is no schedule inside a batch
// to get wrong. What the arrangement costs is the barrier — a batch takes as
// long as its slowest entry — and thin batches, where the heap held fewer
// entries than there were cores. These settle that the instrument separating
// those two records the walk it is timing, and does not disturb it.

/// A fixture recording what each batch cost, at a width that makes batches
/// possible at all.
fn batch_probed(width: usize) -> Fixture {
    Fixture {
        search_threads: width,
        batch_probe: true,
        ..Fixture::new()
    }
}

/// Timing the batches does not change them. The probe adds two clock reads per
/// entry to examining that already happens, and nothing to any decision.
#[test]
fn probing_the_batches_changes_nothing_about_the_walk() {
    let plain = bounded_search(&at_width(4).inputs(symmetric(), AMPLE)).expect("runs");
    let probed = bounded_search(&batch_probed(4).inputs(symmetric(), AMPLE)).expect("runs");

    assert_eq!(probed.stats, plain.stats);
    assert_eq!(probed.outcome, plain.outcome);
    assert_eq!(probed.best_objective, plain.best_objective);
    assert!(
        plain.batch_probe.is_none(),
        "and stays off unless asked for"
    );
}

/// Every entry the walk popped belongs to exactly one batch, so the rows
/// account for the whole walk rather than a sample of it. Without that the
/// ratio they are read for would be over an unknown denominator.
#[test]
fn the_batches_account_for_every_entry_the_walk_popped() {
    let report = bounded_search(&batch_probed(4).inputs(symmetric(), AMPLE)).expect("runs");
    let probe: BatchProbeReport = report
        .batch_probe
        .expect("a probed walk reports its batches");

    assert!(!probe.rows.is_empty(), "the case has to examine something");
    let taken: usize = probe.rows.iter().map(|r| r.taken).sum();
    assert_eq!(
        taken, report.stats.regions_popped,
        "an unseeded walk pops only through batches"
    );
    for r in &probe.rows {
        assert!(
            r.taken <= r.width,
            "a batch cannot exceed the width it was filled to"
        );
        assert!(
            r.taken >= 1,
            "an empty batch ends the walk rather than being examined"
        );
        assert!(r.slowest_s <= r.total_s + f64::EPSILON);
    }
}

/// The two losses are told apart by different fields, and this fixture has
/// both: a walk narrow enough to fill some batches and starved on others.
#[test]
fn a_thin_batch_is_reported_apart_from_a_slow_one() {
    let report = bounded_search(&batch_probed(4).inputs(symmetric(), AMPLE)).expect("runs");
    let probe = report
        .batch_probe
        .expect("a probed walk reports its batches");

    let thin: usize = probe
        .rows
        .iter()
        .filter(|r| r.idle_for_want_of_work() > 0)
        .count();
    assert!(
        thin > 0,
        "a heap this small has to starve the width somewhere, or the case says nothing \
         about the field that reports it"
    );
    for r in &probe.rows {
        assert_eq!(r.idle_for_want_of_work(), r.width - r.taken);
        assert!(
            r.cost_factor() >= 1.0 - 1e-9,
            "a batch cannot beat a fed pool"
        );
    }
}

fn keyed(protein: &'static str) -> Fixture {
    Fixture {
        marginal_key: Some(MarginalKey::default()),
        ..Fixture::over(protein)
    }
}

/// Keying children on their parent's codon max-marginals changes which regions
/// are folded and nothing the walk returns: the same design, objective and
/// verdict as the unkeyed walk and as exhaustive enumeration, on the fixture
/// whose modular set is a proper non-empty subset of the space. And it is not
/// vacuous there: children are lifted and the proof folds fewer regions.
#[test]
fn a_marginal_key_does_not_change_the_answer() {
    let f = Fixture::new();
    let best_by_oracle = f
        .modular_designs(symmetric())
        .into_iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .expect("the fixture has a non-empty modular set");
    let plain = bounded_search(&f.inputs(symmetric(), AMPLE)).expect("the search runs");
    let k = keyed(PROTEIN);
    let lifted = bounded_search(&k.inputs(symmetric(), AMPLE)).expect("the search runs");
    assert_eq!(lifted.outcome, SearchOutcome::Proved);
    assert!(lifted.modular);
    assert_eq!(lifted.best_objective, Some(best_by_oracle.0));
    assert_eq!(lifted.best_objective, plain.best_objective);
    assert_eq!(
        lifted.best.expect("an incumbent").sequence,
        best_by_oracle.1.sequence
    );
    assert!(lifted.stats.marginal_passes > 0, "no outside pass ran");
    assert!(lifted.stats.children_lifted > 0, "no child's key moved");
    assert!(
        lifted.stats.solves - lifted.stats.regions_empty
            < plain.stats.solves - plain.stats.regions_empty,
        "the key lifted children and folded no fewer regions: {} vs {}",
        lifted.stats.solves - lifted.stats.regions_empty,
        plain.stats.solves - plain.stats.regions_empty,
    );
}

/// The other terminal exit: an empty modular set is still established, by
/// exhausting every region that holds a design.
#[test]
fn a_marginal_key_does_not_change_an_exhaustive_verdict() {
    let report =
        bounded_search(&keyed(SMALL_PROTEIN).inputs(full(), AMPLE)).expect("the search runs");
    assert_eq!(report.outcome, SearchOutcome::Proved);
    assert!(!report.modular);
    assert_eq!(report.best_objective, None);
}

/// The key a region enters the heap under is a floor under the exact bound its
/// own solve returns, on every region the walk solved. For a part-0 child the
/// marginal's region *is* the child, so there the floor sits within the key's
/// rounding of the exact bound — checked in both directions, so a key that
/// fell far below would fail as surely as one that rose above.
#[test]
fn a_marginal_key_is_a_floor_under_every_region_it_keys() {
    let f = Fixture {
        bound_probe: true,
        ..keyed(PROTEIN)
    };
    let report = bounded_search(&f.inputs(symmetric(), AMPLE)).expect("the search runs");
    let probe = report.bound_probe.expect("a probed walk returns rows");
    let slack = 0.5 * (1.0 - LAMBDA) + 1e-6;
    let mut part0 = 0;
    for row in &probe.rows {
        let Probed::Solved { exact, .. } = row.probed else {
            continue;
        };
        assert!(
            row.inherited <= exact,
            "a region entered under {} but solves to {exact}",
            row.inherited
        );
        if row.part == 0 {
            part0 += 1;
            assert!(
                exact - row.inherited <= 2.0 * slack,
                "a part-0 child's key {} is looser than rounding below its bound {exact}",
                row.inherited
            );
        }
    }
    assert!(
        part0 > 0,
        "no part-0 child was solved: the equality half is vacuous"
    );
}

/// The kept-table budget moves what a walk costs and nothing it decides: a
/// rebuilt table is the same table, so every key, counter and answer of a walk
/// that keeps nothing equals one that keeps everything — except the count of
/// tables rebuilt, which is every branching in the one and none in the other.
#[test]
fn the_kept_table_budget_changes_no_decision() {
    let run = |kept_bytes| {
        let f = Fixture {
            marginal_key: Some(MarginalKey { kept_bytes }),
            ..Fixture::new()
        };
        bounded_search(&f.inputs(symmetric(), AMPLE)).expect("the search runs")
    };
    let ample = run(usize::MAX);
    let none = run(0);
    assert_eq!(
        ample.stats.tables_rebuilt, 0,
        "an ample budget rebuilds nothing"
    );
    assert!(ample.stats.marginal_passes > 0, "the case never branched");
    assert_eq!(
        none.stats.tables_rebuilt, none.stats.marginal_passes,
        "a zero budget rebuilds every parent it branches"
    );
    let decisions = |s: SearchStats| SearchStats {
        tables_rebuilt: 0,
        ..s
    };
    assert_eq!(decisions(none.stats), decisions(ample.stats));
    assert_eq!(none.outcome, ample.outcome);
    assert_eq!(none.best_objective, ample.best_objective);
    assert_eq!(
        none.best.expect("an incumbent").sequence,
        ample.best.expect("an incumbent").sequence
    );
}

/// A keyed walk's outside passes run on the examining workers, beside each
/// region's verdict. The keys are applied in pop order all the same, so the
/// width reaches the answer no more than it does unkeyed.
#[test]
fn the_width_does_not_change_a_keyed_answer() {
    let serial =
        bounded_search(&keyed(PROTEIN).inputs(symmetric(), AMPLE)).expect("the search runs");
    assert_eq!(serial.outcome, SearchOutcome::Proved);
    for width in [2, 4, 8] {
        let wide = bounded_search(
            &Fixture {
                search_threads: width,
                ..keyed(PROTEIN)
            }
            .inputs(symmetric(), AMPLE),
        )
        .expect("the search runs");
        assert_eq!(wide.outcome, serial.outcome, "at width {width}");
        assert_eq!(
            wide.best_objective, serial.best_objective,
            "at width {width}"
        );
        assert_eq!(
            wide.best.expect("an incumbent").sequence,
            serial.best.clone().expect("an incumbent").sequence,
            "at width {width}"
        );
        assert!(wide.stats.marginal_passes > 0, "at width {width}");
    }
}

/// A modular region's children are all discarded by the stop rule before any
/// of their keys is read, so a keyed walk runs no outside pass for it: on a
/// case answered at the root it runs none at all.
#[test]
fn a_modular_region_runs_no_outside_pass() {
    let (_, constraints) = (0, RegionConstraints::default());
    let report =
        bounded_search(&keyed(PROTEIN).inputs(constraints, AMPLE)).expect("the search runs");
    assert_eq!(report.outcome, SearchOutcome::Proved);
    assert!(report.modular);
    assert_eq!(
        report.stats.solves, 1,
        "an inert table is answered at the root"
    );
    assert_eq!(report.stats.marginal_passes, 0);
}
