//! The free-fold diagnostic: how far a region-constrained design is from the
//! way its own sequence would fold unconstrained.
//!
//! A hard region constraint ([`crate::RegionConstraints`]) is enforced at pair
//! formation, so the design it returns is the exact optimum over structures that
//! respect it. The molecule is under no such obligation. Folded with no constraint,
//! the emitted sequence may reinstate exactly the pairs the design forbade —
//! and nothing in the constrained solve can notice, because the forbidden pairs
//! never entered its energy accounting.
//!
//! [`free_fold_gap`] measures that directly. It folds the emitted sequence
//! twice — once under the constraints and once with no constraint at all — and
//! reports:
//!
//! - **the gap** — how much energy the constraint cost, and the verdict: a
//!   sequence whose two optima coincide folds the way it was designed to,
//! - **the reinstated pairs** — which constrained pairs the free fold puts back,
//! - **the per-class breakdown** — how the free fold's pairs distribute over the
//!   six [`crate::PairClass`]es and the start-proximal window, the classes the
//!   table permits included.
//!
//! The gap is the verdict because it relates the two *optima*: a single
//! recovered free structure is not enough, since free-optimal structures are
//! routinely co-optimal and one traceback may show a violating structure while
//! an equally optimal satisfying one exists. The reinstated pairs and the
//! breakdown then attribute a failing verdict — which interaction the molecule
//! insisted on — and describe what it did with the classes nobody forbade.
//!
//! Deriving the constrained energy by **re-scoring the design's own structure**
//! instead is a shortcut that holds only where that structure is
//! constrained-optimal for the emitted sequence, which needs the objective to
//! carry the folding energy at all: at `λ = 0` every structure is equally
//! optimal and the traceback returned is arbitrary. The re-score is retained for
//! the separate job it is sound at — checking that a design's reported energy is
//! one its reported structure accounts for ([`EmittedRescore`]).
//!
//! This is a **report, not a gate**. A design whose free fold reinstates pairs is
//! still the true optimum of the constrained problem; what the report supplies is
//! the evidence to judge whether that problem was the right one. Gating on the
//! gap would also manufacture an infeasibility path where the substrate has none
//! — the all-unpaired structure satisfies every region constraint, so a
//! constrained solve always succeeds.
//!
//! The MFE structure is a single-structure proxy for an ensemble question. The
//! faithful quantity is the probability of a cross-region pair over the Boltzmann
//! ensemble — a ratio of partition functions, and non-additive, so it cannot ride
//! the design DP as an objective. It would occupy this same position in the
//! pipeline: an evaluator applied to a finished design.

use rna_mfe::{evaluate, Energy, EvalError, Params};
use thiserror::Error;

use crate::region::{PairClass, PairConstraint, Region, RegionConstraints, Verdict};
use crate::structure::StructureConstraints;
use crate::Design;

/// The CDS's position window within a whole-mRNA sequence: `[start, end)` in
/// nucleotides, where `start` is the length of the 5′ UTR and `end - start` is
/// three times the protein length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CdsWindow {
    /// First CDS position (`= |UTR₅′|`).
    pub start: usize,
    /// First position past the CDS.
    pub end: usize,
}

impl CdsWindow {
    /// The window for a `protein_len`-residue CDS starting at `utr5_len`.
    pub fn new(utr5_len: usize, protein_len: usize) -> Self {
        Self {
            start: utr5_len,
            end: utr5_len + 3 * protein_len,
        }
    }
}

/// How a structure's pairs distribute over the six [`PairClass`]es and the
/// start-proximal window.
///
/// **Descriptive, never a verdict.** For a class the table enforces the count
/// explains a failing verdict — which interaction the molecule insisted on. For
/// a class it permits the count carries no verdict at all, and is the only thing
/// that surfaces a structure the table never objected to: a flank paired
/// base-for-base to the opposite flank is legal under a table that permits
/// `5′UTR↔3′UTR`, and invisible to any report that looks only at enforced
/// classes.
///
/// The window count **overlaps** the class counts — every pair joins two
/// regions whether or not an endpoint sits in the window — so the six class
/// counts alone sum to [`Self::total`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ClassBreakdown {
    /// Pairs per class, indexed by [`PairClass`] declaration order.
    classes: [usize; 6],
    /// Pairs with at least one endpoint in the start-proximal window; zero when
    /// no window is set.
    window: usize,
    /// Longest ungapped run of stacked `5′UTR↔3′UTR` pairs. See
    /// [`Self::flank_run`].
    flank_run: usize,
}

impl ClassBreakdown {
    /// Pairs falling in `class`.
    pub fn count(&self, class: PairClass) -> usize {
        self.classes[class as usize]
    }

    /// Pairs with at least one endpoint in the start-proximal window.
    pub fn window(&self) -> usize {
        self.window
    }

    /// The longest **ungapped** run of stacked pairs joining the 5′ flank to the
    /// 3′ one: the largest `n` for which some chain
    /// `(i, j), (i+1, j−1), …, (i+n−1, j−n+1)` is present with every pair in the
    /// `5′UTR↔3′UTR` class.
    ///
    /// It qualifies [`Self::count`] for that class rather than duplicating it,
    /// and the two answer different questions. The count says how much of the
    /// two flanks is paired to the other; the run says whether any of it forms a
    /// stack. Nine pairs in three runs of three is a different molecule from
    /// nine pairs in one run of nine, and only the run distinguishes them.
    ///
    /// Reported because a sequence-space constraint that forbids complementary
    /// runs is stated over exactly this quantity — a run shorter than the
    /// constraint's length is the instrument working, not failing — while the
    /// pair count beside it can be large at the same time. Without both numbers
    /// a caller cannot tell one from the other.
    pub fn flank_run(&self) -> usize {
        self.flank_run
    }

    /// Pairs in the structure, over every class.
    pub fn total(&self) -> usize {
        self.classes.iter().sum()
    }

    /// Every class with its count, in [`PairClass::ALL`] order.
    pub fn counts(&self) -> impl Iterator<Item = (PairClass, usize)> + '_ {
        PairClass::ALL.into_iter().map(|c| (c, self.count(c)))
    }
}

/// A design's own structure re-scored under the energy model — the phantom-free
/// check that a reported energy is one the reported structure accounts for.
///
/// Not the modularity verdict, and not a source for it: a structure the designer
/// returned is constrained-optimal for its sequence only where the objective
/// weighs folding energy, which fails at `λ = 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmittedRescore {
    /// The emitted structure's energy under the plain energy model — **without**
    /// any soft region penalty. Under a hard-only constraint set this equals the
    /// design's reported `mfe_kcal_mol`; under a soft one it is lower by the
    /// penalty the design paid, and is the physically meaningful of the two.
    pub mfe_kcal_mol: f64,
    /// The soft penalty the emitted structure incurs under the active table:
    /// one weight per penalized pair. Zero under a hard-only table.
    pub penalty_kcal_mol: f64,
    /// The design's reported energy minus what its own structure accounts for
    /// (`mfe_kcal_mol + penalty_kcal_mol`). Zero when the report is
    /// phantom-free; anything else is energy the report claims and the structure
    /// does not explain.
    pub phantom_kcal_mol: f64,
}

/// What a region-constrained design costs against the free fold of its own
/// sequence.
///
/// Produced by [`free_fold_gap`] and [`sequence_gap`]. All energies are
/// kcal/mol.
#[derive(Debug, Clone, PartialEq)]
pub struct FreeFoldGap {
    /// `E_iso` — the best energy over the structures of this sequence the
    /// constraints allow, from a real constrained fold of it. Under a soft table
    /// it carries the penalty the optimum pays, since that is what the
    /// constrained problem minimizes.
    pub constrained_mfe_kcal_mol: f64,
    /// `E_free` — the MFE of the same sequence folded with no region constraint at
    /// all.
    pub free_mfe_kcal_mol: f64,
    /// `E_iso − E_free`: the energy the constraint costs this sequence, and the
    /// modularity verdict. Non-negative, and zero exactly when the sequence's
    /// unconstrained optimum already satisfies the constraint.
    pub gap_kcal_mol: f64,
    /// Dot-bracket of the free fold, over the whole mRNA.
    pub free_structure: String,
    /// The pairs of the free fold that the constraints would not have allowed —
    /// `(i, j)` positions, ascending. Non-empty means the recovered preferred
    /// structure rebuilds what the design excluded; because free optima are
    /// routinely co-optimal, an empty list is weaker evidence than a zero gap.
    pub reinstated_pairs: Vec<(usize, usize)>,
    /// The positions the constraint requires to carry a pair and the free fold
    /// leaves unpaired — absolute, 0-based, ascending.
    ///
    /// The attribution for the half a region table does not have. A region
    /// table, an unpaired range and a forbidden pair only ever *forbid*, so this
    /// is empty for them and [`Self::reinstated_pairs`] carries everything. A
    /// required pair or a required position adds the other direction: the
    /// molecule's own fold can decline to form what was required, which no pair
    /// of the free structure witnesses. A required position the free fold pairs
    /// with the *wrong* partner is not here — it is paired, and the pair it took
    /// is one the constraint forbids, so it appears in
    /// [`Self::reinstated_pairs`] instead.
    ///
    /// Attribution, not the verdict: [`Self::is_modular`] is the energy
    /// comparison, as it is for the forbid half, because free optima are
    /// routinely co-optimal and a required pair missing from one traceback may
    /// be present in an equally optimal structure.
    pub unpaired_required_positions: Vec<usize>,
    /// How the free fold's pairs distribute over the classes and the window,
    /// permitted classes included.
    pub classes: ClassBreakdown,
    /// The design's own structure, checked against the energy model. `None` when
    /// the report was derived from a sequence alone ([`sequence_gap`]), where
    /// there is no reported structure to check.
    pub emitted: Option<EmittedRescore>,
}

impl FreeFoldGap {
    /// Whether the free fold reinstates any constrained pair.
    pub fn reinstates(&self) -> bool {
        !self.reinstated_pairs.is_empty()
    }

    /// Whether the free fold does anything the constraint refuses — a pair it
    /// forbids, or a required position it leaves unpaired.
    ///
    /// The two lists together, for a caller describing *what* the molecule did;
    /// still not the verdict, which is [`Self::is_modular`].
    pub fn violates(&self) -> bool {
        self.reinstates() || !self.unpaired_required_positions.is_empty()
    }

    /// The verdict: whether the constrained and unconstrained optima of this
    /// sequence coincide. Energies are integer dekacalories under the model, so
    /// the comparison is exact up to the `/100.0` float division.
    pub fn is_modular(&self) -> bool {
        self.gap_kcal_mol.abs() < 1e-9
    }

    /// Whether the gap is at most `epsilon_kcal_mol` ([`within_epsilon`]).
    /// At zero this is [`Self::is_modular`] exactly; above it,
    /// it admits designs that are not modular, so it is a
    /// different set and not a looser test of the same one.
    pub fn is_within(&self, epsilon_kcal_mol: f64) -> bool {
        within_epsilon(self.gap_kcal_mol, epsilon_kcal_mol)
    }
}

/// Whether a gap of `gap_kcal_mol` is at most `epsilon_kcal_mol`. A zero gap
/// always is, by the test [`FreeFoldGap::is_modular`] applies;
/// with `epsilon_kcal_mol` zero nothing else is.
pub fn within_epsilon(gap_kcal_mol: f64, epsilon_kcal_mol: f64) -> bool {
    gap_kcal_mol.abs() < 1e-9 || (epsilon_kcal_mol > 0.0 && gap_kcal_mol <= epsilon_kcal_mol + 1e-9)
}

/// `epsilon_kcal_mol` in whole dekacalories, the unit energies are scored in:
/// a design is within it when its free MFE is at least its constrained MFE
/// less this many. Rounded down, since a gap is a whole number of them.
pub fn epsilon_dekacal(epsilon_kcal_mol: f64) -> i32 {
    (epsilon_kcal_mol * 100.0 + 1e-6).floor().max(0.0) as i32
}

/// Errors surfaced by [`free_fold_gap`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FreeFoldError {
    /// The emitted structure could not be re-scored under the energy model.
    #[error("re-scoring the emitted structure failed: {0}")]
    Evaluate(#[from] EvalError),
    /// A byte of the sequence is not a nucleotide, so nothing can be folded.
    /// The message is the one the fold itself reported when the parse was
    /// reached through it.
    #[error("free fold of the emitted sequence failed: invalid nucleotide byte: {0:?}")]
    InvalidNucleotide(u8),
    /// The Zuker fold of the emitted sequence failed.
    #[error("Zuker fold of the emitted sequence failed: {0}")]
    Zuker(#[from] rna_mfe::FoldError),
    /// A soft constraint reached the constrained Zuker fold, which can only
    /// drop pairs.
    #[error("the constrained Zuker fold takes hard constraints only")]
    Soft,
    /// No structure of the sequence satisfies the positional constraint, so
    /// there is no constrained optimum for the free fold to be compared with.
    ///
    /// Unreachable for a design emitted under that constraint — its own
    /// structure is such a structure — and reachable for any other sequence the
    /// same constraint is applied to.
    #[error("no structure of the sequence carries every required pair")]
    NoSatisfyingStructure,
}

/// Measure `design` against the unconstrained fold of its own sequence, under
/// the `constraints` it was designed with and the CDS window `cds` it spans.
///
/// [`sequence_gap`] plus the phantom-free check the design's own structure makes
/// possible. Costs two exact folds of a single sequence against the design's
/// fold of a whole lattice — the same asymptotics with a much smaller constant,
/// the share falling as the lattice grows. See the module docs for what the
/// reported quantities do and do not establish.
///
/// With an inert `constraints` the report is trivially clean: the constrained
/// fold *is* the free fold, so the gap is zero and no pair is reinstated. The
/// per-class breakdown is still meaningful, and is then a plain description of
/// how the sequence folds.
pub fn free_fold_gap(
    params: &Params,
    design: &Design,
    cds: CdsWindow,
    constraints: RegionConstraints,
) -> Result<FreeFoldGap, FreeFoldError> {
    let constraint = PairConstraint::for_cds(&constraints, cds.start as u32, cds.end as u32);
    let mut report = measure(params, design.sequence.as_bytes(), constraint.clone())?;
    report.emitted = Some(rescore(params, design, &constraint)?);
    Ok(report)
}

/// [`free_fold_gap`] under a positional structure constraint beside the region
/// table — the verdict for `--structure`, `--helix`, `--unpaired` and
/// `--forbid-pair`.
///
/// The same quantity over a larger constraint set, not a second measurement. The
/// constrained half folds the emitted sequence under both statements at once:
/// the mask drops the pairs either of them forbids
/// ([`StructureConstraints::pair_mask`]), and every position the statement
/// requires to pair is held paired, which is the half a region table has no
/// form for. The free half is the same unconstrained fold.
///
/// So `gap_kcal_mol` is non-negative and zero exactly when some free-optimal
/// structure satisfies both statements: the constrained set is a subset of every
/// structure of the sequence whichever half shrank it, and the constrained
/// optimum is itself a structure, so a zero gap exhibits one.
/// [`FreeFoldGap::is_modular`] therefore reads on a positional constraint what
/// it reads on a table.
///
/// The attribution gains [`FreeFoldGap::unpaired_required_positions`] beside
/// [`FreeFoldGap::reinstated_pairs`], which the mask now contributes to.
///
/// An inert `structure` is [`free_fold_gap`] exactly, by the same call.
///
/// This is **the same gap** [`crate::realization_verdict`] reports, with the
/// region table folded into the constrained half rather than left out of it —
/// both constrained halves are [`fixed_mrna_mfe`], so with an inert
/// `constraints` the two numbers agree. The two differ in what they buy beside
/// the verdict: that one prices each required pair on its own, at one further
/// fold per pair, and names the element the molecule declined; this one is the
/// region report's fields over the whole constraint, at the region report's
/// cost.
///
/// Costs one free fold and one constrained fold, as [`free_fold_gap`] does.
pub fn free_fold_gap_with_structure(
    params: &Params,
    design: &Design,
    cds: CdsWindow,
    constraints: RegionConstraints,
    structure: &StructureConstraints,
) -> Result<FreeFoldGap, FreeFoldError> {
    if structure.is_inert() {
        return free_fold_gap(params, design, cds, constraints);
    }
    let constraint = PairConstraint::for_cds(&constraints, cds.start as u32, cds.end as u32)
        .with_mask(structure.pair_mask());
    let mut report = measure_requiring(
        params,
        design.sequence.as_bytes(),
        constraint.clone(),
        &structure.required_positions(),
    )?;
    report.emitted = Some(rescore(params, design, &constraint)?);
    Ok(report)
}

/// Measure an arbitrary `sequence` — one nobody designed, or one designed under
/// different constraints — against `constraints` over the CDS window `cds`.
///
/// The modularity test in its general form: fold the sequence twice, once freely
/// and once under the constraint, and report the difference. A sequence with
/// `gap_kcal_mol == 0` is one whose *unconstrained* optimum already satisfies the
/// constraint, which is a claim about the molecule rather than about a design
/// space.
///
/// Comparing the two fold *energies* is the correct test; inspecting a single
/// recovered structure is not, because free-optimal structures are routinely
/// co-optimal and one traceback may show a violating structure while an equally
/// optimal satisfying one exists.
///
/// Costs two exact folds of one sequence.
pub fn sequence_gap(
    params: &Params,
    sequence: &str,
    cds: CdsWindow,
    constraints: RegionConstraints,
) -> Result<FreeFoldGap, FreeFoldError> {
    let constraint = PairConstraint::for_cds(&constraints, cds.start as u32, cds.end as u32);
    measure(params, sequence.as_bytes(), constraint)
}

/// [`sequence_gap`] for a caller that already holds the constrained MFE, in
/// kcal/mol: the free fold and the comparison, without folding the constrained
/// half again.
///
/// The constrained energy is the expensive half and a caller that has already
/// folded the sequence under the same constraint has it already. The test the
/// gap performs is unaffected: it compares the constrained energy against an
/// **independently** computed free one, and only the duplicate constrained fold
/// is skipped.
///
/// `constrained_mfe_kcal_mol` must be the constrained MFE of this `sequence`
/// under these `constraints`; passing anything else reports a gap that is not
/// the molecule's.
///
/// Costs one exact fold of one sequence.
pub fn sequence_gap_given_constrained(
    params: &Params,
    sequence: &str,
    cds: CdsWindow,
    constraints: RegionConstraints,
    constrained_mfe_kcal_mol: f64,
) -> Result<FreeFoldGap, FreeFoldError> {
    let constraint = PairConstraint::for_cds(&constraints, cds.start as u32, cds.end as u32);
    let (free_structure, free_energy) = check_free_fold(params, sequence.as_bytes())?;
    let free = free_energy.to_kcal_mol();
    let (classes, reinstated_pairs) = classify(&free_structure, &constraint);
    Ok(FreeFoldGap {
        constrained_mfe_kcal_mol,
        free_mfe_kcal_mol: free,
        gap_kcal_mol: constrained_mfe_kcal_mol - free,
        reinstated_pairs,
        unpaired_required_positions: Vec::new(),
        classes,
        free_structure,
        emitted: None,
    })
}

/// [`sequence_gap`] for a caller that holds an allowed `structure` of
/// `sequence` and its energy `mfe_kcal_mol`, as a front point does: the free
/// fold against that energy, without the constrained fold, whenever the
/// structure proves it.
///
/// The structure is re-scored under the energy model and every pair checked
/// against a hard `constraints`. When both hold, an allowed structure of that
/// energy exists, so the constrained MFE is at most `mfe_kcal_mol`; the free
/// MFE is at most the constrained one, so a zero gap against `mfe_kcal_mol`
/// is a zero gap of the molecule. A structure that only *ties* the optimum is
/// all that is needed, which every point of an exact constrained front is: a
/// lower-energy allowed structure of the same sequence would dominate it.
///
/// Where either test fails — a soft table, a pair the table forbids, an
/// energy the structure does not account for — this is [`sequence_gap`]. So
/// it is sound on any input, and an energy that is merely too high costs a
/// verdict at most, never a false pass.
///
/// Costs one exact fold of one sequence where the structure holds, two
/// otherwise.
pub fn sequence_gap_given_structure(
    params: &Params,
    sequence: &str,
    structure: &str,
    mfe_kcal_mol: f64,
    cds: CdsWindow,
    constraints: RegionConstraints,
) -> Result<FreeFoldGap, FreeFoldError> {
    if structure_holds(params, sequence, structure, mfe_kcal_mol, cds, constraints) {
        sequence_gap_given_constrained(params, sequence, cds, constraints, mfe_kcal_mol)
    } else {
        sequence_gap(params, sequence, cds, constraints)
    }
}

/// The check's free fold of one sequence, kept so that a later fold of a
/// sequence of the same length can copy from it
/// ([`sequence_gap_given_structure_reusing`]).
pub struct FreeFoldTables {
    seq: Vec<u8>,
    tables: rna_mfe::dp::FoldTables,
    copied: u64,
}

impl FreeFoldTables {
    /// How many cells this fold copied rather than computed.
    pub fn copied(&self) -> u64 {
        self.copied
    }
}

/// [`sequence_gap_given_structure`], copying the free fold's cells from the
/// earlier folds in `pool`: each row `i` takes, from whichever fold copies
/// most of it, every cell `(i, j)` whose span avoids the stretch where that
/// fold's sequence differs from `sequence`. A copied cell holds the value
/// this fold would compute (same nucleotides on its span, no cut), so the
/// tables, the free MFE and the free structure are the fresh fold's.
///
/// Returns this fold's tables for later folds, or `None` where the structure
/// does not hold and the check falls back to [`sequence_gap`].
pub fn sequence_gap_given_structure_reusing(
    params: &Params,
    sequence: &str,
    structure: &str,
    mfe_kcal_mol: f64,
    cds: CdsWindow,
    constraints: RegionConstraints,
    pool: &[&FreeFoldTables],
) -> Result<(FreeFoldGap, Option<FreeFoldTables>), FreeFoldError> {
    if !structure_holds(params, sequence, structure, mfe_kcal_mol, cds, constraints) {
        return Ok((sequence_gap(params, sequence, cds, constraints)?, None));
    }
    let ((free_structure, free_energy), tables) =
        sequence_free_fold_zuker_reusing(params, sequence, pool)?;
    let constraint = PairConstraint::for_cds(&constraints, cds.start as u32, cds.end as u32);
    let (classes, reinstated_pairs) = classify(&free_structure, &constraint);
    let free = free_energy.to_kcal_mol();
    let gap = FreeFoldGap {
        constrained_mfe_kcal_mol: mfe_kcal_mol,
        free_mfe_kcal_mol: free,
        gap_kcal_mol: mfe_kcal_mol - free,
        reinstated_pairs,
        unpaired_required_positions: Vec::new(),
        classes,
        free_structure,
        emitted: None,
    };
    Ok((gap, Some(tables)))
}

/// Whether `structure` proves `mfe_kcal_mol` an allowed energy of `sequence`
/// under a hard `constraints`: it re-scores to that energy and has no pair the
/// table forbids ([`sequence_gap_given_structure`]).
fn structure_holds(
    params: &Params,
    sequence: &str,
    structure: &str,
    mfe_kcal_mol: f64,
    cds: CdsWindow,
    constraints: RegionConstraints,
) -> bool {
    if !(constraints.is_hard() && structure.len() == sequence.len()) {
        return false;
    }
    // Re-scored first: a structure the model evaluates is balanced, which the
    // pair walk below assumes.
    let accounted = evaluate(params, sequence.as_bytes(), structure)
        .is_ok_and(|e| (e.to_kcal_mol() - mfe_kcal_mol).abs() < 1e-9);
    let forbidden = constraints.pair_forbidden(cds.start as u32, cds.end as u32);
    let mut allowed = accounted;
    if accounted {
        for_each_pair(structure, |i, j| allowed &= !forbidden(i as u32, j as u32));
    }
    allowed
}

/// [`sequence_free_fold_zuker`], copying cells from the folds in `pool` as
/// [`sequence_gap_given_structure_reusing`] does, with this fold's tables.
pub fn sequence_free_fold_zuker_reusing(
    params: &Params,
    sequence: &str,
    pool: &[&FreeFoldTables],
) -> Result<((String, Energy), FreeFoldTables), FreeFoldError> {
    use rna_mfe::dp::{fold_sparse_reusing, Reuse, SparseFold};
    let bytes = sequence.as_bytes();
    let nts = bytes
        .iter()
        .map(|&b| rna_mfe::Nucleotide::from_byte(b).ok_or(rna_mfe::FoldError::InvalidNucleotide(b)))
        .collect::<Result<Vec<_>, _>>()?;
    let pool: Vec<&FreeFoldTables> = pool
        .iter()
        .copied()
        .filter(|b| b.seq.len() == bytes.len())
        .collect();
    let tables: Vec<&rna_mfe::dp::FoldTables> = pool.iter().map(|b| &b.tables).collect();
    let rows = (!pool.is_empty()).then(|| {
        let seqs: Vec<&[u8]> = pool.iter().map(|b| b.seq.as_slice()).collect();
        reuse_rows(bytes, &seqs)
    });
    let reuse = rows.as_ref().map(|(until, from)| Reuse {
        tables: &tables,
        until,
        from,
    });
    let (fold, tables, copied) = fold_sparse_reusing(params, &nts, None, reuse.as_ref());
    match fold {
        SparseFold::Below(energy, structure) => Ok((
            (structure, energy),
            FreeFoldTables {
                seq: bytes.to_vec(),
                tables,
                copied,
            },
        )),
        SparseFold::AtOrAbove => unreachable!("with no cut the sparse fold returns its MFE"),
    }
}

/// For each row `i` of a fold of `seq`, how far it may copy (`until`: spans
/// `[i, j]` with `j < until[i]`) and from which of `bases` (`from`), each a
/// sequence of the same length. A base copies a row past the stretch where it
/// differs from `seq` whole, a row before it up to the stretch, and nothing of
/// a row inside it; each row takes the base that copies most.
fn reuse_rows(seq: &[u8], bases: &[&[u8]]) -> (Vec<usize>, Vec<usize>) {
    let n = seq.len();
    let (mut until, mut from) = (vec![0usize; n], vec![0usize; n]);
    for (b, base) in bases.iter().enumerate() {
        let a = (0..n).find(|&k| base[k] != seq[k]).unwrap_or(n);
        let z = (0..n).rev().find(|&k| base[k] != seq[k]).unwrap_or(0);
        for i in 0..n {
            let reach = if i > z || a == n {
                n
            } else if i < a {
                a
            } else {
                i
            };
            if reach > until[i] {
                until[i] = reach;
                from[i] = b;
            }
        }
    }
    (until, from)
}

/// The verdict of [`sequence_gap`] alone, for a caller that keeps only the
/// modular: the constrained MFE in kcal/mol when `sequence` is
/// modular, `None` when it is not.
///
/// Under a hard table the two Zuker folds refute first. The constrained Zuker
/// fold's grammar holds the scan's, so its MFE lies between the free MFE and
/// the scan's constrained one; when it is above the free MFE the gap is
/// positive and the scan is never run. Only a sequence the Zuker folds cannot
/// refute pays for the scan, whose energy then decides as in
/// [`sequence_gap`]. A soft table is [`sequence_gap`].
///
/// Costs two sparse folds of one sequence for a refuted design, those and
/// the scan otherwise.
pub fn sequence_modular_mfe(
    params: &Params,
    sequence: &str,
    cds: CdsWindow,
    constraints: RegionConstraints,
) -> Result<Option<f64>, FreeFoldError> {
    sequence_mfe_within(params, sequence, cds, constraints, 0.0)
}

/// [`sequence_modular_mfe`] for the designs whose gap is at most
/// `epsilon_kcal_mol` ([`within_epsilon`]): the constrained MFE when it is,
/// `None` when it is not. The Zuker refutation stays sound, since the scan's
/// constrained MFE is at least the constrained Zuker fold's and so its gap
/// at least as large.
pub fn sequence_mfe_within(
    params: &Params,
    sequence: &str,
    cds: CdsWindow,
    constraints: RegionConstraints,
    epsilon_kcal_mol: f64,
) -> Result<Option<f64>, FreeFoldError> {
    let verdict =
        |constrained: f64, free: f64| within_epsilon(constrained - free, epsilon_kcal_mol);
    if !constraints.is_hard() {
        let gap = sequence_gap(params, sequence, cds, constraints)?;
        return Ok(gap
            .is_within(epsilon_kcal_mol)
            .then_some(gap.constrained_mfe_kcal_mol));
    }
    let free = check_free_fold(params, sequence.as_bytes())?
        .1
        .to_kcal_mol();
    let zuker = sequence_constrained_mfe_zuker(params, sequence, cds, constraints)?;
    if !verdict(zuker, free) {
        return Ok(None);
    }
    let constraint = PairConstraint::for_cds(&constraints, cds.start as u32, cds.end as u32);
    let constrained = constrained_mfe(params, sequence.as_bytes(), constraint)?;
    Ok(verdict(constrained, free).then_some(constrained))
}

/// [`sequence_modular_mfe`] under a hard table, with both Zuker
/// folds copying from `base` — an earlier result for a sequence of the same
/// length under the same `cds` and `constraints` — every cell whose span
/// avoids the stretch where the two differ ([`sequence_mfes_reusing`]). The
/// verdict comes with this sequence's tables, to be the next one's `base`.
/// Designs that differ only near one end share most of their cells.
///
/// # Errors
///
/// [`FreeFoldError::Soft`] under a table that penalizes rather than forbids.
pub fn sequence_modular_mfe_reusing(
    params: &Params,
    sequence: &str,
    cds: CdsWindow,
    constraints: RegionConstraints,
    base: Option<&ZukerMfes>,
) -> Result<(Option<f64>, ZukerMfes), FreeFoldError> {
    sequence_mfe_within_reusing(params, sequence, cds, constraints, base, 0.0)
}

/// [`sequence_modular_mfe_reusing`] for the designs whose gap is
/// at most `epsilon_kcal_mol`, as [`sequence_mfe_within`] decides it.
///
/// # Errors
///
/// [`FreeFoldError::Soft`] under a table that penalizes rather than forbids.
pub fn sequence_mfe_within_reusing(
    params: &Params,
    sequence: &str,
    cds: CdsWindow,
    constraints: RegionConstraints,
    base: Option<&ZukerMfes>,
    epsilon_kcal_mol: f64,
) -> Result<(Option<f64>, ZukerMfes), FreeFoldError> {
    let mfes = sequence_mfes_reusing(params, sequence, cds, constraints, base, &|_| true)?;
    let free = mfes
        .free
        .expect("the free fold runs when asked for every sequence");
    if !within_epsilon(mfes.constrained - free, epsilon_kcal_mol) {
        return Ok((None, mfes));
    }
    let constraint = PairConstraint::for_cds(&constraints, cds.start as u32, cds.end as u32);
    let constrained = constrained_mfe(params, sequence.as_bytes(), constraint)?;
    let so = within_epsilon(constrained - free, epsilon_kcal_mol).then_some(constrained);
    Ok((so, mfes))
}

/// The exact unconstrained MFE of `sequence` and its MFE structure, in
/// dot-bracket, by the Zuker DP: interior loops are capped at 30 unpaired
/// bases, while a multiloop's leading unpaired run is not capped, so a
/// multiloop with more than 30 unpaired bases before its first branch is
/// reachable. A fold that caps that run can only read the same MFE or a higher
/// one. Where the MFE is degenerate, two exact folds may return different
/// co-optimal structures.
///
/// Run as the sparse fold with no cut ([`rna_mfe::dp::fold_sparse`]), which
/// is the dense fold's ([`rna_mfe::fold`]) MFE exactly while iterating only
/// over the pairs it keeps: about 2.9x faster than the dense fold at 4,348 nt.
///
/// This is the free half of the check ([`sequence_gap`] and its variants).
pub fn sequence_free_fold_zuker(
    params: &Params,
    sequence: &str,
) -> Result<(String, Energy), FreeFoldError> {
    check_free_fold(params, sequence.as_bytes())
}

/// The check's free fold: the Zuker DP, whose grammar is the constrained
/// scan's with nothing forbidden. A free fold that missed a structure the
/// scan can form would read the free MFE high and could pass a design whose
/// molecule folds lower; this one cannot.
fn check_free_fold(params: &Params, seq: &[u8]) -> Result<(String, Energy), FreeFoldError> {
    use rna_mfe::dp::{fold_sparse, SparseFold};
    let nts = seq
        .iter()
        .map(|&b| rna_mfe::Nucleotide::from_byte(b).ok_or(rna_mfe::FoldError::InvalidNucleotide(b)))
        .collect::<Result<Vec<_>, _>>()?;
    match fold_sparse(params, &nts, None) {
        SparseFold::Below(energy, structure) => Ok((structure, energy)),
        SparseFold::AtOrAbove => unreachable!("with no cut the sparse fold returns its MFE"),
    }
}

/// The constrained half of [`sequence_gap`] alone: the best energy, in
/// kcal/mol, over structures of `sequence` that `constraints` allow over the
/// CDS window `cds`. One exact fold.
pub fn sequence_constrained_mfe(
    params: &Params,
    sequence: &str,
    cds: CdsWindow,
    constraints: RegionConstraints,
) -> Result<f64, FreeFoldError> {
    let constraint = PairConstraint::for_cds(&constraints, cds.start as u32, cds.end as u32);
    constrained_mfe(params, sequence.as_bytes(), constraint)
}

/// [`sequence_constrained_mfe`] by the Zuker DP: the sparse fold with every
/// pair `constraints` forbid dropped by a cut. Its grammar is the scan's —
/// interior loops capped at 30 unpaired bases, a multi-loop's leading unpaired
/// run unbounded, which is the energy model's own set and what
/// [`GRAMMAR_VERSION`](crate::GRAMMAR_VERSION) 2 records. So the two minimize over one structure class
/// and agree wherever both are exact.
/// Hard constraints only: a pair the constraints would penalize is an error
/// ([`FreeFoldError::Soft`]).
pub fn sequence_constrained_mfe_zuker(
    params: &Params,
    sequence: &str,
    cds: CdsWindow,
    constraints: RegionConstraints,
) -> Result<f64, FreeFoldError> {
    Ok(sequence_mfes_reusing(params, sequence, cds, constraints, None, &|_| false)?.constrained)
}

/// Both Zuker MFEs of one sequence, and the tables that let a later fold of
/// a sequence differing in one stretch copy the rest
/// ([`sequence_mfes_reusing`]).
pub struct ZukerMfes {
    /// The constrained MFE, kcal/mol.
    pub constrained: f64,
    /// The free MFE, kcal/mol, when it was folded.
    pub free: Option<f64>,
    seq: Vec<u8>,
    constrained_tables: rna_mfe::dp::FoldTables,
    free_tables: Option<rna_mfe::dp::FoldTables>,
}

/// The constrained Zuker MFE of `sequence` ([`sequence_constrained_mfe_zuker`])
/// and, when `fold_free` holds of it (kcal/mol), its free Zuker MFE, copying from `base` (an earlier
/// result for a sequence of the same length, under the same `cds` and
/// `constraints`) every cell whose span avoids the stretch where the two
/// sequences differ. A copied cell holds the value this fold would compute:
/// same nucleotides on its span, same cut test on every pair inside it.
pub fn sequence_mfes_reusing(
    params: &Params,
    sequence: &str,
    cds: CdsWindow,
    constraints: RegionConstraints,
    base: Option<&ZukerMfes>,
    fold_free: &dyn Fn(f64) -> bool,
) -> Result<ZukerMfes, FreeFoldError> {
    use rna_mfe::dp::{fold_sparse_reusing, Cut, Reuse, SparseFold};
    let bytes = sequence.as_bytes();
    let nts = bytes
        .iter()
        .map(|&b| rna_mfe::Nucleotide::from_byte(b).ok_or(rna_mfe::FoldError::InvalidNucleotide(b)))
        .collect::<Result<Vec<_>, _>>()?;
    let n = nts.len();
    let constraint = PairConstraint::for_cds(&constraints, cds.start as u32, cds.end as u32);
    let soft = std::cell::Cell::new(false);
    // Any V a real pair can take lies far inside +-2^28 dekacalories, so a
    // forbidden pair's V plus 2^29 always reaches the threshold and an allowed
    // pair's never does.
    let outside = |i: usize, j: usize| match constraint.verdict(i as u32, j as u32) {
        Verdict::Allow => 0,
        Verdict::Forbid => 1 << 29,
        Verdict::Penalize(_) => {
            soft.set(true);
            0
        }
    };
    let cut = Cut {
        threshold: 1 << 28,
        outside: &outside,
        inside: None,
    };
    // Spans `[i, j]` that avoid the differing stretch `a..=b`: every `j` for a
    // row past it, `j < a` for a row before it.
    let base = base.filter(|b| b.seq.len() == n);
    let until: Option<Vec<usize>> = base.map(|b| {
        let a = (0..n).find(|&k| b.seq[k] != bytes[k]).unwrap_or(n);
        let z = (0..n).rev().find(|&k| b.seq[k] != bytes[k]).unwrap_or(0);
        (0..n)
            .map(|i| {
                if i > z {
                    n
                } else if i < a {
                    a
                } else {
                    i
                }
            })
            .collect()
    });
    let from = vec![0usize; n];
    let energy = |f: SparseFold| match f {
        SparseFold::Below(e, _) => e.to_kcal_mol(),
        SparseFold::AtOrAbove => 0.0,
    };
    let (c_fold, constrained_tables, _) = {
        let held: Option<[&rna_mfe::dp::FoldTables; 1]> = base.map(|b| [&b.constrained_tables]);
        let reuse = held.as_ref().zip(until.as_deref()).map(|(t, u)| Reuse {
            tables: t,
            until: u,
            from: &from,
        });
        fold_sparse_reusing(params, &nts, Some(&cut), reuse.as_ref())
    };
    if soft.get() {
        return Err(FreeFoldError::Soft);
    }
    let constrained = energy(c_fold);
    let (free, free_tables) = if fold_free(constrained) {
        let held: Option<[&rna_mfe::dp::FoldTables; 1]> =
            base.and_then(|b| b.free_tables.as_ref()).map(|t| [t]);
        let reuse = held.as_ref().zip(until.as_deref()).map(|(t, u)| Reuse {
            tables: t,
            until: u,
            from: &from,
        });
        let (f, t, _) = fold_sparse_reusing(params, &nts, None, reuse.as_ref());
        (Some(energy(f)), Some(t))
    } else {
        (None, None)
    };
    Ok(ZukerMfes {
        constrained,
        free,
        seq: bytes.to_vec(),
        constrained_tables,
        free_tables,
    })
}

/// The verdict and everything read off the two folds of one sequence: the
/// constrained and free optima, the free structure, its pairs the constraint would
/// have refused, and their distribution over the classes.
pub(crate) fn measure(
    params: &Params,
    seq: &[u8],
    constraint: PairConstraint,
) -> Result<FreeFoldGap, FreeFoldError> {
    measure_requiring(params, seq, constraint, &[])
}

/// [`measure`] with `required` positions held paired in the constrained fold —
/// the require half, which only a positional statement has.
///
/// With `required` empty this is [`measure`] by the same code path, so the
/// positional entry points cost a constraint that asks for nothing exactly what
/// the class-only ones do.
fn measure_requiring(
    params: &Params,
    seq: &[u8],
    constraint: PairConstraint,
    required: &[u32],
) -> Result<FreeFoldGap, FreeFoldError> {
    let (free_structure, free_energy) = check_free_fold(params, seq)?;
    let constrained = constrained_mfe_requiring(params, seq, constraint.clone(), required)?;
    let free = free_energy.to_kcal_mol();
    let (classes, reinstated_pairs) = classify(&free_structure, &constraint);
    Ok(FreeFoldGap {
        constrained_mfe_kcal_mol: constrained,
        free_mfe_kcal_mol: free,
        gap_kcal_mol: constrained - free,
        reinstated_pairs,
        unpaired_required_positions: left_unpaired(&free_structure, required),
        classes,
        free_structure,
        emitted: None,
    })
}

/// The positions of `required` that `structure` leaves unpaired, ascending.
///
/// A position past the end of `structure` counts as unpaired: a requirement the
/// structure does not reach is one it does not meet.
fn left_unpaired(structure: &str, required: &[u32]) -> Vec<usize> {
    if required.is_empty() {
        return Vec::new();
    }
    let mut paired = vec![false; structure.len()];
    for_each_pair(structure, |i, j| {
        paired[i] = true;
        paired[j] = true;
    });
    let mut out: Vec<usize> = required
        .iter()
        .map(|&p| p as usize)
        .filter(|&p| !paired.get(p).copied().unwrap_or(false))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Re-score the design's own structure and compare it with the energy the design
/// reported — the phantom-free check.
fn rescore(
    params: &Params,
    design: &Design,
    constraint: &PairConstraint,
) -> Result<EmittedRescore, FreeFoldError> {
    let mfe = evaluate(params, design.sequence.as_bytes(), &design.structure)?.to_kcal_mol();
    let penalty = soft_penalty(&design.structure, constraint);
    Ok(EmittedRescore {
        mfe_kcal_mol: mfe,
        penalty_kcal_mol: penalty,
        phantom_kcal_mol: design.mfe_kcal_mol - (mfe + penalty),
    })
}

/// The soft penalty `structure` incurs under `constraint`: one weight per penalized
/// pair. A hard-forbidden pair contributes nothing — under a hard table the
/// design cannot contain one, and a caller re-scoring a structure against a
/// table it was not designed under is asking about energy, not legality.
fn soft_penalty(structure: &str, constraint: &PairConstraint) -> f64 {
    let mut total = Energy(0);
    for_each_pair(structure, |i, j| {
        if let Verdict::Penalize(w) = constraint.verdict(i as u32, j as u32) {
            total = Energy(total.0 + w.0);
        }
    });
    total.to_kcal_mol()
}

/// The best energy over structures of `seq` that `constraint` allows, via a
/// lane-of-one scan at `λ = 1` (pure MFE — the CAI half is identically zero over
/// a graph with no codon lattice).
fn constrained_mfe(
    params: &Params,
    seq: &[u8],
    constraint: PairConstraint,
) -> Result<f64, FreeFoldError> {
    constrained_mfe_requiring(params, seq, constraint, &[])
}

/// [`constrained_mfe`] with every position of `required` (absolute, 0-based)
/// holding a pair.
///
/// A forbid-only constraint always admits the all-unpaired structure, so the
/// exterior parse always reaches the end node and the fold always has an
/// answer. A requirement can be unsatisfiable for a given sequence, which is
/// what [`FreeFoldError::NoSatisfyingStructure`] reports — the one way this fold
/// has no optimum to return.
fn constrained_mfe_requiring(
    params: &Params,
    seq: &[u8],
    constraint: PairConstraint,
    required: &[u32],
) -> Result<f64, FreeFoldError> {
    fixed_mrna_mfe(params, seq, constraint, required, crate::Beam::exact())?
        .ok_or(FreeFoldError::NoSatisfyingStructure)
}

/// The best energy over the structures of a **fixed** mRNA `seq` that
/// `constraint` allows and that pair every position of `required` — `None` when
/// there is none.
///
/// One lane-of-one scan at `λ = 1`: pure MFE, since the adaptation half of the
/// objective is identically zero over a graph with no codon lattice. The single
/// place the constrained half of every verdict on an already-spelled sequence is
/// folded, whether the constraint reached it as a region table
/// ([`sequence_constrained_mfe`]), as a positional statement
/// ([`crate::honouring_mfe`]) or as both ([`free_fold_gap_with_structure`]), so
/// the three cannot drift apart.
///
/// `beam` is the CKY column policy. On [`crate::Beam::exact`] the energy is the
/// optimum and `None` is a proof that no allowed structure of this sequence
/// pairs them all — a real answer for a fixed sequence, unlike the design case
/// where infeasibility is a claim about a whole lattice. Under a finite width
/// the energy is an **upper bound** on that optimum and `None` is a statement
/// about one search.
pub(crate) fn fixed_mrna_mfe(
    params: &Params,
    seq: &[u8],
    constraint: PairConstraint,
    required: &[u32],
    beam: crate::Beam,
) -> Result<Option<f64>, FreeFoldError> {
    let bad_byte = || FreeFoldError::InvalidNucleotide(seq.first().copied().unwrap_or(b'?'));
    let nts = crate::parse_utr(std::str::from_utf8(seq).map_err(|_| bad_byte())?)
        .map_err(|_| bad_byte())?;
    let graph = crate::dfa::build_fixed_mrna(&nts);
    let must = crate::must_pair::MustPairConstraint::new(required, graph.seq_len);
    let tables = crate::scan::scan_with_constraints(
        params,
        &graph,
        1.0,
        crate::ScanConfig {
            beam,
            ..crate::ScanConfig::default()
        },
        constraint,
        must,
    );
    Ok(tables.c[graph.ix(graph.end())].map(|best| best.mfe.to_kcal_mol()))
}

/// One walk of `structure`, classifying every pair against `constraint`: how the
/// pairs distribute over the classes and the window, and which of them the constraint
/// would have forbidden or penalized (ascending).
pub(crate) fn classify(
    structure: &str,
    constraint: &PairConstraint,
) -> (ClassBreakdown, Vec<(usize, usize)>) {
    let mut breakdown = ClassBreakdown::default();
    let mut hits = Vec::new();
    let mut flank_pairs = Vec::new();
    for_each_pair(structure, |i, j| {
        let (open, close) = (i as u32, j as u32);
        let class = constraint.class_of(open, close);
        breakdown.classes[class as usize] += 1;
        breakdown.window += usize::from(constraint.in_window(open, close));
        if class == PairClass::Utr5Utr3 {
            flank_pairs.push((i, j));
        }
        if constraint.verdict(open, close) != Verdict::Allow {
            hits.push((i, j));
        }
    });
    breakdown.flank_run = longest_stacked_run(&mut flank_pairs);
    hits.sort_unstable();
    (breakdown, hits)
}

/// The longest chain `(i, j), (i+1, j−1), …` all of whose pairs are in `pairs`.
fn longest_stacked_run(pairs: &mut [(usize, usize)]) -> usize {
    maximal_runs(pairs)
        .iter()
        .map(|run| run.len)
        .max()
        .unwrap_or(0)
}

/// Every maximal chain `(i, j), (i+1, j−1), …` all of whose pairs are in
/// `pairs`, as `(outermost pair, stacked pairs)`.
///
/// Sorts in place and probes by binary search rather than hashing: the input is
/// one class’s pairs from a single structure, tens of entries at most, and a
/// sorted probe keeps the walk allocation-free and its result order-independent.
///
/// **Maximal within the set it is given.** A chain is cut wherever the next
/// stacked pair is absent from `pairs`, so passing the pairs of one class makes
/// a run that crosses into another class end at the crossing — which is what
/// lets a run carry a single class rather than a list of them.
fn maximal_runs(pairs: &mut [(usize, usize)]) -> Vec<Run> {
    pairs.sort_unstable();
    let has = |a: usize, b: usize| pairs.binary_search(&(a, b)).is_ok();
    let mut out = Vec::new();
    for &(i, j) in pairs.iter() {
        // Walk each maximal chain once, from its 5′ end — the pair with no
        // stacked predecessor. Without this every chain is re-walked from each
        // of its members, which costs work and changes no answer.
        if i > 0 && has(i - 1, j + 1) {
            continue;
        }
        let (mut a, mut b, mut len) = (i, j, 0);
        while b > a && has(a, b) {
            len += 1;
            a += 1;
            b -= 1;
        }
        out.push(Run {
            open: i,
            close: j,
            len,
        });
    }
    out
}

/// One maximal chain, before it is told which class it joins.
struct Run {
    open: usize,
    close: usize,
    len: usize,
}

/// One maximal ungapped run of stacked pairs a folded structure holds, every
/// pair of which joins the same two regions.
///
/// `(open, close)` is the outermost pair and `len` counts the stacked pairs, so
/// the innermost pair is `(open + len − 1, close − len + 1)` and the two sides
/// occupy `[open, open + len)` and `[close − len + 1, close + 1)`. That is the
/// naming [`crate::Duplex`] uses, and [`crate::Duplex::from_run`] converts.
///
/// The distinction from [`crate::cross_region_duplexes`] is what the object is a
/// fact about. That census reads a **sequence** and reports the duplexes it
/// could spell; this one reads a **structure** and reports the ones a fold
/// actually formed. A sequence-space constraint is stated over the first and
/// what a molecule does is the second, so a set claimed to prevent a duplex is
/// checked against the first and a residue is measured on the second.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackedRun {
    /// The 5′ endpoint of the outermost pair, absolute and 0-based.
    pub open: usize,
    /// Its 3′ endpoint.
    pub close: usize,
    /// Stacked pairs in the run.
    pub len: usize,
    /// The two regions every pair of the run joins.
    pub class: PairClass,
}

impl StackedRun {
    /// The positions the run occupies inside `cds`, when one of its two sides
    /// lies there — `None` for a run with no coding endpoint, and for one whose
    /// **both** sides are coding, where naming “the” coding side is meaningless.
    ///
    /// This is what a constraint against the run is written over: the coding
    /// side is the half a codon assignment can respell, and the given side is
    /// not.
    pub fn cds_side(&self, cds: CdsWindow) -> Option<std::ops::Range<usize>> {
        let sides = [
            self.open..self.open + self.len,
            self.close + 1 - self.len..self.close + 1,
        ];
        let coding = |side: &std::ops::Range<usize>| side.start >= cds.start && side.end <= cds.end;
        match (coding(&sides[0]), coding(&sides[1])) {
            (true, false) => Some(sides[0].clone()),
            (false, true) => Some(sides[1].clone()),
            _ => None,
        }
    }
}

/// Every maximal ungapped run of stacked pairs in `structure`, over a molecule
/// whose coding sequence spans `cds`.
///
/// Runs are grouped by class before they are chained, so every run carries one
/// class and a chain that crosses a region boundary ends at the crossing.
///
/// Returned **longest first**, ties by position — so the first element is the
/// longest run, ties leftmost, which is the one a loop cutting a single
/// counterexample per round is defined over.
pub fn stacked_runs(structure: &str, cds: CdsWindow) -> Vec<StackedRun> {
    let region = |p: usize| {
        if p < cds.start {
            Region::Utr5
        } else if p < cds.end {
            Region::Cds
        } else {
            Region::Utr3
        }
    };
    let mut by_class: [Vec<(usize, usize)>; 6] = Default::default();
    for_each_pair(structure, |i, j| {
        by_class[PairClass::of(region(i), region(j)) as usize].push((i, j));
    });
    let mut out: Vec<StackedRun> = Vec::new();
    for class in PairClass::ALL {
        out.extend(
            maximal_runs(&mut by_class[class as usize])
                .into_iter()
                .map(|run| StackedRun {
                    open: run.open,
                    close: run.close,
                    len: run.len,
                    class,
                }),
        );
    }
    out.sort_by(|a, b| {
        b.len
            .cmp(&a.len)
            .then(a.open.cmp(&b.open))
            .then(a.close.cmp(&b.close))
    });
    out
}

/// Call `visit(i, j)` for every base pair of a balanced dot-bracket `structure`.
pub(crate) fn for_each_pair(structure: &str, mut visit: impl FnMut(usize, usize)) {
    let mut stack = Vec::new();
    for (j, c) in structure.bytes().enumerate() {
        match c {
            b'(' => stack.push(j),
            b')' => visit(stack.pop().expect("balanced dot-bracket"), j),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit coverage for the pair-classification walk; the end-to-end contracts
    //! (inert cleanliness, the soft-penalty split, hard-isolation agreement) are
    //! integration tests at `tests/free_fold_gap.rs`.

    use super::*;
    use crate::{AccessibilityWindow, Enforcement};

    /// The 14-position fixture every case below classifies, over a 4-nt 5′ UTR,
    /// a two-codon CDS at `[4, 10)`, and a 4-nt 3′ UTR:
    ///
    /// ```text
    /// positions  0123456789•••
    /// structure  ((..((....))))
    /// ```
    ///
    /// Pairs `(0, 13)` and `(1, 12)` join the two UTRs; `(4, 11)` and `(5, 10)`
    /// join the CDS to the 3′ UTR.
    const STRUCTURE: &str = "((..((....))))";

    /// Isolation over `[4, 10)` — a 4-nt 5′ UTR and a two-codon CDS.
    fn isolation_constraint() -> PairConstraint {
        let rc = RegionConstraints::symmetric_isolation(Enforcement::Forbid);
        PairConstraint::for_cds(&rc, 4, 10)
    }

    /// The pairs `constraint` would have refused, the half of [`classify`] most cases
    /// here are about.
    fn refused(structure: &str, constraint: &PairConstraint) -> Vec<(usize, usize)> {
        classify(structure, constraint).1
    }

    #[test]
    fn classify_picks_out_cross_region_pairs_only() {
        assert_eq!(
            refused(STRUCTURE, &isolation_constraint()),
            [(4, 11), (5, 10)]
        );
    }

    #[test]
    fn classify_refuses_nothing_under_an_inert_constraint() {
        let constraint = PairConstraint::for_cds(&RegionConstraints::default(), 4, 10);
        assert!(refused(STRUCTURE, &constraint).is_empty());
    }

    #[test]
    fn classify_counts_soft_pairs_as_refused() {
        // A soft family still "would not have allowed" the pair unpriced, so it
        // is reported — the gap tells the caller what the price bought.
        let rc = RegionConstraints::symmetric_isolation(Enforcement::Penalize(Energy(50)));
        let constraint = PairConstraint::for_cds(&rc, 4, 10);
        assert_eq!(refused(STRUCTURE, &constraint), [(4, 11), (5, 10)]);
    }

    #[test]
    fn classify_refuses_window_pairs() {
        // Window [2, 6) straddling the start at 4; isolation off, so only the
        // window fires: (1,12) is outside it, (4,11) and (5,10) are inside.
        let rc = RegionConstraints::default().with_window(AccessibilityWindow::hard(2, 2));
        let constraint = PairConstraint::for_cds(&rc, 4, 10);
        assert_eq!(refused(STRUCTURE, &constraint), [(4, 11), (5, 10)]);
    }

    /// The breakdown describes the whole structure, not the enforced part of it:
    /// the same counts come back whether the table forbids the classes or
    /// permits them.
    #[test]
    fn breakdown_covers_permitted_classes_too() {
        for constraint in [
            isolation_constraint(),
            PairConstraint::for_cds(&RegionConstraints::default(), 4, 10),
        ] {
            let (b, _) = classify(STRUCTURE, &constraint);
            assert_eq!(b.count(PairClass::Utr5Utr3), 2);
            assert_eq!(b.count(PairClass::CdsUtr3), 2);
            assert_eq!(b.count(PairClass::Utr5Cds), 0);
            assert_eq!(b.total(), 4, "every pair lands in exactly one class");
            assert_eq!(b.window(), 0, "no window is set on either constraint");
        }
    }

    /// The window count is an overlay on the class counts rather than a seventh
    /// class: a windowed pair is still counted in the classes it joins.
    #[test]
    fn breakdown_window_overlaps_the_classes() {
        let rc = RegionConstraints::default().with_window(AccessibilityWindow::hard(2, 2));
        let (b, _) = classify(STRUCTURE, &PairConstraint::for_cds(&rc, 4, 10));
        assert_eq!(b.window(), 2);
        assert_eq!(b.total(), 4);
        assert_eq!(b.count(PairClass::CdsUtr3), 2);
    }

    /// The two flank-to-flank pairs of the fixture stack, so the run is 2 —
    /// equal to the count here, which is why the cases below separate them.
    #[test]
    fn breakdown_reports_the_stacked_flank_run() {
        let (b, _) = classify(STRUCTURE, &isolation_constraint());
        assert_eq!(b.count(PairClass::Utr5Utr3), 2);
        assert_eq!(b.flank_run(), 2);
    }

    /// **The reason the field exists.** A bulged flank-to-flank duplex carries
    /// more pairs than its longest stack, so the count alone cannot say whether
    /// a constraint stated over run length held. Here nine positions of a 12-nt
    /// 5′ flank pair to a 12-nt 3′ flank in three runs of three, separated by
    /// single unpaired nucleotides: the count reads 9 and the run reads 3.
    #[test]
    fn a_bulged_flank_duplex_has_a_run_far_below_its_pair_count() {
        //  0..11 = 5′ UTR, 12..17 = CDS (two codons), 18..29 = 3′ UTR.
        //  Pairs (1,29) (2,28) (3,27) | (5,25) (6,24) (7,23) | (9,21) (10,20) (11,19)
        let structure = ".(((.(((.(((.......))).))).)))";
        let constraint = PairConstraint::for_cds(&RegionConstraints::default(), 12, 18);
        let (b, _) = classify(structure, &constraint);
        assert_eq!(b.count(PairClass::Utr5Utr3), 9, "nine flank-to-flank pairs");
        assert_eq!(b.flank_run(), 3, "in three stacked runs of three");
    }

    /// The same nine pairs in one unbroken stack: the count is unchanged and the
    /// run tells the two molecules apart. Without this pairing the previous test
    /// would pass against an implementation that returned the pair count.
    #[test]
    fn an_ungapped_flank_duplex_has_a_run_equal_to_its_pair_count() {
        //  0..11 = 5′ UTR, 12..17 = CDS, 18..29 = 3′ UTR; pairs (0,29)…(8,21).
        let structure = "(((((((((............)))))))))";
        let constraint = PairConstraint::for_cds(&RegionConstraints::default(), 12, 18);
        let (b, _) = classify(structure, &constraint);
        assert_eq!(b.count(PairClass::Utr5Utr3), 9);
        assert_eq!(b.flank_run(), 9);
    }

    /// Only flank-to-flank pairs count toward the run. A CDS-to-3′ stack is a
    /// stack, and it is not the thing the number is a claim about.
    #[test]
    fn the_run_ignores_stacks_in_other_classes() {
        let constraint = PairConstraint::for_cds(&RegionConstraints::default(), 4, 10);
        let (b, _) = classify("....((....))..", &constraint);
        assert_eq!(b.count(PairClass::CdsUtr3), 2, "a stacked cds-3 pair");
        assert_eq!(b.count(PairClass::Utr5Utr3), 0);
        assert_eq!(b.flank_run(), 0);
    }

    /// Every class the breakdown reports is a class the enumeration knows about,
    /// so a new class cannot be silently dropped from a report.
    #[test]
    fn breakdown_counts_enumerate_every_class() {
        let (b, _) = classify(STRUCTURE, &isolation_constraint());
        let listed: Vec<_> = b.counts().collect();
        assert_eq!(listed.len(), PairClass::ALL.len());
        assert_eq!(listed.iter().map(|&(_, n)| n).sum::<usize>(), b.total());
    }

    /// The 5′-flank-to-CDS duplex of the fixture: `(4, 11)` and `(5, 10)` stack,
    /// and the two flank-to-flank pairs stack as their own run. Every run
    /// carries one class, and the longest comes first.
    #[test]
    fn stacked_runs_groups_a_structure_into_runs_of_one_class_each() {
        let runs = stacked_runs(STRUCTURE, CdsWindow::new(4, 2));
        assert_eq!(
            runs,
            [
                StackedRun {
                    open: 0,
                    close: 13,
                    len: 2,
                    class: PairClass::Utr5Utr3,
                },
                StackedRun {
                    open: 4,
                    close: 11,
                    len: 2,
                    class: PairClass::CdsUtr3,
                },
            ]
        );
    }

    /// **A chain crossing a region boundary is two runs, not one.** The pairs
    /// `(3, 12)` and `(4, 11)` stack, but the first joins the 5′ UTR to the 3′
    /// one and the second joins the CDS to the 3′ UTR — so a constraint stated
    /// over one class sees a run of one, and reporting a run of two would credit
    /// it with a length no single class holds.
    #[test]
    fn a_run_ends_where_it_crosses_a_region_boundary() {
        //  0..3 = 5′ UTR, 4..9 = CDS (two codons), 10..13 = 3′ UTR. The pairs
        //  (3, 11) and (4, 10) stack across the 5′ boundary.
        let structure = "...((.....))..";
        let runs = stacked_runs(structure, CdsWindow::new(4, 2));
        assert_eq!(runs.iter().map(|r| r.len).collect::<Vec<_>>(), [1, 1]);
        assert_eq!(
            runs.iter().map(|r| r.class).collect::<Vec<_>>(),
            [PairClass::Utr5Utr3, PairClass::CdsUtr3]
        );
    }

    /// The census is ordered so its first element is the longest run, ties
    /// leftmost — the one a loop cutting a single counterexample per round is
    /// defined over.
    #[test]
    fn stacked_runs_puts_the_longest_run_first() {
        //  0..5 = 5′ UTR, 6..11 = CDS, 12..17 = 3′ UTR. One flank-to-flank pair
        //  outside a three-pair CDS-to-3′ stack.
        let structure = "(.....(((...)))..)";
        let runs = stacked_runs(structure, CdsWindow::new(6, 2));
        assert_eq!(runs[0].len, 3);
        assert_eq!(runs[0].class, PairClass::CdsUtr3);
        assert_eq!(runs.last().expect("two runs").len, 1);
    }

    /// The coding side is the half a codon assignment can respell, and it is
    /// read off the run rather than assumed to be either end: it is the 3′ side
    /// of a 5′UTR↔CDS run and the 5′ side of a CDS↔3′UTR one.
    #[test]
    fn the_cds_side_is_whichever_side_lies_in_the_coding_sequence() {
        let cds = CdsWindow::new(4, 2); // CDS at [4, 10)
        let cds_to_three = StackedRun {
            open: 4,
            close: 11,
            len: 2,
            class: PairClass::CdsUtr3,
        };
        assert_eq!(cds_to_three.cds_side(cds), Some(4..6));
        let five_to_cds = StackedRun {
            open: 0,
            close: 9,
            len: 2,
            class: PairClass::Utr5Cds,
        };
        assert_eq!(five_to_cds.cds_side(cds), Some(8..10));
    }

    /// A run with no coding endpoint has no coding side, and neither does one
    /// with two — no codon assignment reaches the first, and naming “the” side
    /// of the second is meaningless. Both are `None` rather than a guess.
    #[test]
    fn a_run_with_no_single_cds_side_reports_none() {
        let cds = CdsWindow::new(4, 2);
        let flank_to_flank = StackedRun {
            open: 0,
            close: 13,
            len: 2,
            class: PairClass::Utr5Utr3,
        };
        assert_eq!(flank_to_flank.cds_side(cds), None);
        let inside_the_cds = StackedRun {
            open: 4,
            close: 9,
            len: 1,
            class: PairClass::CdsCds,
        };
        assert_eq!(inside_the_cds.cds_side(cds), None);
    }

    #[test]
    fn cds_window_spans_three_nt_per_residue() {
        assert_eq!(CdsWindow::new(4, 2), CdsWindow { start: 4, end: 10 });
        assert_eq!(CdsWindow::new(0, 3), CdsWindow { start: 0, end: 9 });
    }
}
