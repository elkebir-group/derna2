//! What the cut loop guarantees, and what it gives up to get there.
//!
//! The loop is a heuristic, so most of what can be asserted is a relation to the
//! partitioning search rather than a property of its own output. Each test
//! either pins a claim the loop makes unconditionally, or compares it against
//! `bounded_search` on the same instance, which is the only oracle available for
//! what was optimal.
//!
//! **Two fixtures, because one of them cuts and the other does not.** A flank
//! that happens not to pair into the coding sequence leaves the first solve
//! already modular, and a loop that never cuts satisfies every claim about
//! cutting vacuously. `sticky` pairs a G-only flank with a proline-rich protein,
//! whose codons are all `CCN`, so the coding sequence cannot avoid spelling C
//! and the table has something real to forbid. Tests about cutting assert that
//! cutting happened.

use std::time::Duration;

use derna_codon_dp::codon_table::AMINO_ACIDS;
use derna_lib::free_fold::{sequence_gap, CdsWindow};
use derna_lib::region::{Enforcement, PairClass, RegionConstraints};
use derna_lib::{
    bounded_search, cegar_search, CegarOutcome, CodonUsage, CutRule, Params, ScanConfig,
    SearchInputs, SearchOutcome,
};

/// A flank that cannot pair much into a mixed coding sequence.
const MILD5: &str = "GGGAGGCGGCGGAGGCGG";
/// A flank of nothing but G, so every C the coding sequence spells is a partner.
const STICKY5: &str = "GGGGGGGGGGGGGGGGGG";
const UTR3: &str = "CCUCCGCCGCCUCCC";

fn aa_indices(protein: &str) -> Vec<usize> {
    protein
        .chars()
        .map(|c| {
            AMINO_ACIDS
                .iter()
                .position(|a| a.code == c)
                .unwrap_or_else(|| panic!("{c} is not an amino acid"))
        })
        .collect()
}

struct Fixture {
    params: Params,
    usage: CodonUsage,
    aa: Vec<usize>,
    protein: String,
    utr5: &'static str,
}

impl Fixture {
    fn new(protein: &str, utr5: &'static str) -> Self {
        Self {
            params: Params::load_default().expect("bundled energy parameters"),
            usage: CodonUsage::load_default().expect("bundled usage table"),
            aa: aa_indices(protein),
            protein: protein.to_string(),
            utr5,
        }
    }

    /// An instance the five-prime table genuinely bites on.
    fn sticky() -> Self {
        Self::new("MPPPPPPPPPPPP", STICKY5)
    }

    /// An instance the partitioning search proves quickly *and* on which the
    /// modular set is non-empty — both halves matter, since a proof that no
    /// modular design exists gives the comparison nothing to compare.
    fn provable() -> Self {
        Self::new("MPPAPPA", MILD5)
    }

    fn inputs(&self, constraints: RegionConstraints) -> SearchInputs<'_> {
        SearchInputs {
            params: &self.params,
            protein: &self.protein,
            aa: &self.aa,
            usage: &self.usage,
            lambda: 0.5,
            utr5: self.utr5,
            utr3: UTR3,
            constraints,
            config: ScanConfig::default(),
            budget: Duration::from_secs(120),
            branch_order: Default::default(),
            trajectory: false,
            seed: None,
            search_threads: 1,
            bound_probe: false,
            batch_probe: false,
            marginal_key: None,
        }
    }
}

fn five_prime() -> RegionConstraints {
    RegionConstraints::five_prime_isolation(Enforcement::Forbid)
}

fn objective(mfe: f64, cai: f64) -> f64 {
    0.5 * (mfe * 100.0).round() - 0.5 * (cai * 100.0).round()
}

/// The sticky fixture is the premise of every cutting claim below, so it is
/// checked once here rather than assumed by each of them.
#[test]
fn the_sticky_fixture_actually_forces_cuts() {
    let f = Fixture::sticky();
    let report = cegar_search(&f.inputs(five_prime()), CutRule::Single).expect("loop runs");
    assert!(
        report.cuts > 0,
        "fixture never cut ({:?} in {} rounds) — every cutting test over it is vacuous",
        report.outcome,
        report.rounds
    );
}

/// The one claim the loop makes without qualification: a design it calls modular
/// folds modular, checked with an independent verdict rather than its bookkeeping.
#[test]
fn a_modular_verdict_survives_an_independent_fold() {
    for rule in [CutRule::Single, CutRule::All] {
        let f = Fixture::sticky();
        let report = cegar_search(&f.inputs(five_prime()), rule).expect("loop runs");
        let Some(design) = report.design else {
            assert!(
                !report.outcome.is_modular(),
                "modular outcome without a design"
            );
            continue;
        };
        let cds = CdsWindow::new(f.utr5.len(), f.protein.len());
        let gap = sequence_gap(&f.params, &design.sequence, cds, five_prime()).expect("folds");
        assert!(
            gap.is_modular(),
            "{rule:?} reported modular but the free fold gives {:.4} kcal/mol",
            gap.gap_kcal_mol
        );
    }
}

/// Each round removes a codon and never restores it, so the loop cannot outlast
/// the codons it can remove. This bound is what distinguishes it from the
/// partitioning search, so it is asserted rather than assumed.
#[test]
fn rounds_are_bounded_by_the_codons_available_to_cut() {
    let f = Fixture::sticky();
    let report = cegar_search(&f.inputs(five_prime()), CutRule::Single).expect("loop runs");
    let ceiling: usize =
        f.aa.iter()
            .map(|&a| AMINO_ACIDS[a].codons.len())
            .sum::<usize>()
            + 1;
    assert!(
        report.rounds <= ceiling,
        "{} rounds against a ceiling of {ceiling}",
        report.rounds
    );
    assert_eq!(
        report.solves, report.rounds,
        "the loop has no lazy bound, so a round is a solve"
    );
}

/// Every round but the last one cuts. A round that neither cut nor stopped would
/// have re-solved an unchanged space.
#[test]
fn every_round_but_the_last_makes_progress() {
    for rule in [CutRule::Single, CutRule::All] {
        let f = Fixture::sticky();
        let report = cegar_search(&f.inputs(five_prime()), rule).expect("loop runs");
        assert!(
            report.cuts >= report.rounds.saturating_sub(1),
            "{rule:?}: {} cuts over {} rounds — some round changed nothing",
            report.cuts,
            report.rounds
        );
    }
}

/// Cutting every blamed position at once cannot need more rounds than cutting
/// one at a time. The two rules are the same loop under different appetites.
#[test]
fn cutting_all_blamed_positions_needs_no_more_rounds_than_cutting_one() {
    let f = Fixture::sticky();
    let inputs = f.inputs(five_prime());
    let single = cegar_search(&inputs, CutRule::Single).expect("loop runs");
    let all = cegar_search(&inputs, CutRule::All).expect("loop runs");
    assert!(single.cuts > 0, "vacuous — the single-cut rule never cut");
    assert!(
        all.rounds <= single.rounds,
        "all-cut took {} rounds against single-cut's {}",
        all.rounds,
        single.rounds
    );
}

/// The comparison the loop exists for. Where the partitioning search proves an
/// optimum, a modular cut-loop design cannot beat it — that would falsify the
/// search's claim — and is allowed to be worse, which is what cutting costs.
#[test]
fn the_loop_never_beats_a_proven_optimum() {
    let f = Fixture::provable();
    let inputs = f.inputs(five_prime());
    let proved = bounded_search(&inputs).expect("search runs");
    // The oracle must actually answer. A budgeted search returns an incumbent
    // that is not an optimum, and a violating one is a proof that the modular
    // set is empty — either way there is nothing to price the loop against, and
    // returning quietly here is how this test passed while comparing nothing.
    assert_eq!(
        proved.outcome,
        SearchOutcome::Proved,
        "the oracle did not finish, so the comparison below would be vacuous"
    );
    assert!(
        proved.modular,
        "the modular set is empty on this instance, so there is no optimum to compare"
    );
    let best = proved
        .best
        .expect("a proved modular walk holds its incumbent");
    let report = cegar_search(&inputs, CutRule::Single).expect("loop runs");
    let found = report
        .design
        .expect("the loop should find a modular design where one provably exists");
    let optimum = objective(best.mfe_kcal_mol, best.cai_log);
    let got = objective(found.mfe_kcal_mol, found.cai_log);
    assert!(
        got >= optimum - 1e-6,
        "cut loop found {got:.1} against a proven optimum of {optimum:.1}"
    );
}

/// An inert table forbids nothing, so the first solve is modular and the loop
/// never cuts. A cut here would be a cut on an invented counterexample.
#[test]
fn an_inert_table_costs_one_round_and_no_cuts() {
    let f = Fixture::new("MWWKRFYQMLHD", MILD5);
    let report =
        cegar_search(&f.inputs(RegionConstraints::default()), CutRule::Single).expect("loop runs");
    assert_eq!(report.outcome, CegarOutcome::Modular);
    assert_eq!(
        report.rounds, 1,
        "an inert table is modular on the first solve"
    );
    assert_eq!(report.cuts, 0);
}

/// A table forbidding only flank-to-flank pairing has no cuttable endpoint —
/// both ends of every pair it refuses sit in given sequence. The loop must say
/// so rather than cut something irrelevant. What it may never be is Exhausted,
/// which would mean it cut its way out of a space no cut could reach.
#[test]
fn a_flank_to_flank_table_is_never_cut_its_way_out_of() {
    let f = Fixture::sticky();
    let clamp = RegionConstraints::default().with(PairClass::Utr5Utr3, Enforcement::Forbid);
    let report = cegar_search(&f.inputs(clamp), CutRule::Single).expect("loop runs");
    assert!(
        matches!(report.outcome, CegarOutcome::NoCut | CegarOutcome::Modular),
        "expected NoCut or Modular, got {:?} after {} cuts",
        report.outcome,
        report.cuts
    );
    assert_eq!(
        report.cuts, 0,
        "no pair this table forbids has an endpoint a codon can reach"
    );
}
