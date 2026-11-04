//! How much energy separates a molecule's own optimum from the best structure
//! that forms a pairing it was constrained against.
//!
//! The modularity verdict ([`crate::free_fold_gap`]) answers whether a design's
//! constrained and free optima coincide. It is zero exactly when they do, so
//! **every clean design reads identically and none says by how much**. This
//! module measures the complement: `E_viol − E_free`, the energy a sequence must
//! give up to do the forbidden thing. Zero means a violating structure is
//! co-optimal and the guarantee holds over strings without holding over the
//! ensemble; a large value means the molecule would have to pay for it.
//!
//! # Why a minimum over a disjunction is computable here
//!
//! `E_viol` is a minimum over the structures that contain **at least one**
//! forbidden duplex. A folding recurrence cannot express that disjunction, which
//! is what makes the quantity look like it needs an outside pass.
//!
//! It does not, because on a **fixed** sequence the disjunction is finite and
//! enumerable. The ungapped duplexes of a given length joining one region to
//! another are found by scanning the two regions against the pairing relation
//! ([`cross_region_duplexes`]), and insisting on one of them is a positional
//! constraint the folder already honours ([`crate::honouring_mfe`]). So
//!
//! ```text
//! E_viol = min over enumerated duplexes d of  MFE(best fold containing d)
//! ```
//!
//! is an **equality**: every violating structure contains some enumerated `d`,
//! so the left side is at least the right; and each right-hand value is attained
//! by a structure that violates, so the right side is at least the left.
//!
//! Two things bound it rather than compute it, and [`Margin`] reports both.
//! Folding only part of the candidate set makes the minimum an **upper** bound —
//! the direction that can still settle a question phrased over margins near
//! zero — and the candidates are ordered so a truncated minimum is over the ones
//! most likely to be cheap. A finite [`Beam`] makes each fold an upper bound on
//! its own optimum, so a beam-mode margin is a difference of two same-policy
//! upper bounds and is not an exact margin.
//!
//! # What counts as a violation, and what deliberately does not
//!
//! Only **ungapped** runs count: a helix interrupted by a bulge is a different
//! object, and a constraint stated over ungapped duplexes does not forbid it.
//! And the coding side must lie **entirely inside** the enforcement window,
//! because an occurrence straddling either edge is permitted. A test asserting
//! that no cross-region pairing occurs at all therefore fails against a correct
//! implementation, and the failure is the test's.

use std::ops::Range;

use rna_mfe::{Nucleotide, Params};

use crate::complement::can_pair;
use crate::coords::Layout;
use crate::free_fold::{FreeFoldError, StackedRun};
use crate::structure::StructureConstraints;
use crate::{Beam, Complement};

/// One ungapped duplex joining two regions of a fixed sequence, named by the
/// outermost pair it forms and how many stacked pairs it carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Duplex {
    /// The 5′ endpoint of the outermost pair, absolute and 0-based.
    pub open: u32,
    /// Its 3′ endpoint.
    pub close: u32,
    /// Stacked pairs, so the innermost pair is `(open + len − 1,
    /// close − len + 1)`.
    pub len: u32,
    /// How firmly the duplex holds together: `G·C` = 3, `A·U` = 2, `G·U` = 1,
    /// summed over its pairs.
    ///
    /// Not an energy, and not used as one. It orders the candidates so that a
    /// fold budget too small to cover them all is spent on the ones a molecule
    /// is most likely to form for free — which is what keeps a truncated
    /// minimum tight instead of arbitrary.
    pub stability: u32,
}

impl Duplex {
    /// The duplex a folded structure's run names, scored for stability against
    /// the sequence that formed it.
    ///
    /// The two censuses report the same object about different things — a
    /// [`StackedRun`] is a duplex a fold **formed**, a [`Duplex`] from
    /// [`cross_region_duplexes`] one a sequence can **spell** — so carrying them
    /// in one type is what lets a report put them in one column and a caller
    /// hand either to [`Duplex::required`].
    pub fn from_run(seq: &[Nucleotide], run: StackedRun) -> Self {
        let (open, len) = (run.open as u32, run.len as u32);
        Self {
            open,
            close: run.close as u32,
            len,
            stability: stability(seq, open, run.close as u32 + 1 - len, len),
        }
    }

    /// The constraint set that insists on this duplex, or `None` when the
    /// folding grammar admits no structure containing it — its two endpoints
    /// are closer than the shortest hairpin allows.
    pub fn required(&self, layout: &Layout) -> Option<StructureConstraints> {
        let mut sc = StructureConstraints::none(layout.len());
        sc.require_helix(layout, self.open, self.close, self.len)
            .ok()?;
        Some(sc)
    }
}

/// Every ungapped duplex of exactly `run` pairs that `seq` spells between the
/// `source` positions and the `target` positions, under `complement`.
///
/// Read off the sequence rather than off whatever produced it, so a set claimed
/// to have made such duplexes unspellable is checked against the string itself.
/// Both endpoints lie wholly inside their range: a duplex straddling either
/// boundary is not enumerated, matching the reading that an occurrence is
/// constrained only where it lies entirely inside the window it is constrained
/// over.
///
/// Returned most-stable-first, then in position order, so the sequence is
/// deterministic and a prefix of it is the tightest prefix available.
pub fn cross_region_duplexes(
    seq: &[Nucleotide],
    source: Range<u32>,
    target: Range<u32>,
    run: u32,
    complement: Complement,
) -> Vec<Duplex> {
    let mut out = Vec::new();
    if run == 0 || source.len() < run as usize || target.len() < run as usize {
        return out;
    }
    for i in source.start..=source.end - run {
        for j in target.start..=target.end - run {
            if stacks(seq, i, j, run, complement) {
                out.push(duplex(seq, i, j, run));
            }
        }
    }
    out.sort_by(|a, b| {
        b.stability
            .cmp(&a.stability)
            .then(a.open.cmp(&b.open))
            .then(a.close.cmp(&b.close))
    });
    out
}

/// The duplex whose two ends begin at `i` and `j`. Pairing is antiparallel, so
/// the outermost pair joins the two outer ends and the innermost joins the two
/// inner ones.
fn duplex(seq: &[Nucleotide], i: u32, j: u32, run: u32) -> Duplex {
    let (open, close) = if i < j {
        (i, j + run - 1)
    } else {
        (j, i + run - 1)
    };
    Duplex {
        open,
        close,
        len: run,
        stability: stability(seq, i, j, run),
    }
}

/// Whether the window at `i` and the window at `j` pair base for base along
/// their whole length.
fn stacks(seq: &[Nucleotide], i: u32, j: u32, run: u32, complement: Complement) -> bool {
    (0..run).all(|s| {
        can_pair(
            seq[(i + s) as usize],
            seq[(j + run - 1 - s) as usize],
            complement,
        )
    })
}

/// `G·C` = 3, `A·U` = 2, anything else canonical = 1, summed over the duplex.
fn stability(seq: &[Nucleotide], i: u32, j: u32, run: u32) -> u32 {
    (0..run)
        .map(
            |s| match (seq[(i + s) as usize], seq[(j + run - 1 - s) as usize]) {
                (Nucleotide::G, Nucleotide::C) | (Nucleotide::C, Nucleotide::G) => 3,
                (Nucleotide::A, Nucleotide::U) | (Nucleotide::U, Nucleotide::A) => 2,
                _ => 1,
            },
        )
        .sum()
}

/// What a sequence gives up to form the pairing it was constrained against, and
/// the magnitudes that say whether the instrument registered anything.
#[derive(Debug, Clone, PartialEq)]
pub struct Margin {
    /// Duplexes enumerated — the size of the violating set.
    pub candidates: usize,
    /// Of those, the ones the folding grammar cannot form at all. Excluded from
    /// the minimum, since no structure contains them.
    pub unformable: usize,
    /// Duplexes folded.
    pub folded: usize,
    /// Duplexes the fold budget did not reach. Non-zero makes the minimum an
    /// upper bound rather than the margin.
    pub dropped: usize,
    /// `E_free` — the sequence's own optimum under the same fold policy.
    pub free_mfe_kcal_mol: f64,
    /// `MFE(best fold containing d) − E_free` for each folded duplex `d`, in the
    /// order the duplexes were folded.
    ///
    /// The **minimum** is the margin. The rest is the magnitude beside it: a
    /// minimum of zero from a set whose other members cost eight is a different
    /// finding from a minimum of zero over an instrument that registered nothing
    /// anywhere.
    pub costs: Vec<f64>,
}

impl Margin {
    /// Whether the minimum is over the whole violating set rather than a prefix
    /// of it — the difference between the margin and an upper bound on it.
    pub fn is_exact(&self) -> bool {
        self.dropped == 0
    }

    /// The margin: the least a violating structure costs. `None` when the
    /// violating set is empty, where the separation is unbounded rather than
    /// zero.
    pub fn least(&self) -> Option<f64> {
        self.costs.iter().copied().reduce(f64::min)
    }

    /// The most a violating structure costs — the magnitude that says the
    /// instrument was on.
    pub fn greatest(&self) -> Option<f64> {
        self.costs.iter().copied().reduce(f64::max)
    }

    /// The median cost over the folded duplexes.
    pub fn median(&self) -> Option<f64> {
        if self.costs.is_empty() {
            return None;
        }
        let mut sorted = self.costs.clone();
        sorted.sort_by(f64::total_cmp);
        Some(sorted[sorted.len() / 2])
    }
}

/// Fold `sequence` once free and once per candidate duplex, and report what
/// insisting on each of them costs it.
///
/// Costs `1 + min(cap, |candidates|)` folds of one fixed sequence and no design
/// work. `cap` bounds how many duplexes are folded; passing [`usize::MAX`] folds
/// every one of them, which is what makes the result exact rather than an upper
/// bound.
///
/// `cap = 0` folds **nothing at all**, the free arm included, and returns the
/// violating set's census with no energies. That is the cheap half of the
/// measurement: how many forbidden duplexes a sequence spells is a property of
/// the string and answerable at lengths where folding it is not, and it is what
/// says whether a constraint had anything to prevent.
pub fn robustness_margin(
    params: &Params,
    sequence: &str,
    layout: &Layout,
    candidates: &[Duplex],
    beam: Beam,
    cap: usize,
) -> Result<Margin, FreeFoldError> {
    if cap == 0 {
        return Ok(census(candidates, layout));
    }
    let inert = StructureConstraints::none(layout.len());
    let free_mfe_kcal_mol = crate::honouring_mfe(params, sequence, layout, &inert, beam)?;
    let mut out = Margin {
        candidates: candidates.len(),
        unformable: 0,
        folded: 0,
        dropped: 0,
        free_mfe_kcal_mol,
        costs: Vec::new(),
    };
    for (index, duplex) in candidates.iter().enumerate() {
        let Some(constraints) = duplex.required(layout) else {
            out.unformable += 1;
            continue;
        };
        if out.folded >= cap {
            out.dropped = candidates.len() - index;
            break;
        }
        let held = crate::honouring_mfe(params, sequence, layout, &constraints, beam)?;
        out.folded += 1;
        if held.is_finite() {
            out.costs.push(held - free_mfe_kcal_mol);
        }
    }
    Ok(out)
}

/// The violating set counted rather than priced — what `cap = 0` returns.
fn census(candidates: &[Duplex], layout: &Layout) -> Margin {
    let unformable = candidates
        .iter()
        .filter(|d| d.required(layout).is_none())
        .count();
    Margin {
        candidates: candidates.len(),
        unformable,
        folded: 0,
        dropped: candidates.len() - unformable,
        free_mfe_kcal_mol: f64::NAN,
        costs: Vec::new(),
    }
}
