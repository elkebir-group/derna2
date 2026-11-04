//! Enumerate designs in increasing objective order.
//!
//! The design DP returns one optimum. A search for a design whose *own* fold has
//! some property cannot be expressed as a constraint on that DP — the property
//! is a fact about the result, not a cost along a structure — so the only exact
//! route is to walk candidates best-first and test each. This module supplies
//! the walk.
//!
//! **Lawler/Murty partitioning.** Solve the whole space; that optimum is rank 1.
//! To enumerate the rest, split "everything except this solution" into disjoint
//! regions: one per free codon position `j`, pinning positions before `j` to the
//! solution and forbidding the solution's codon *at* `j`. Every other sequence
//! differs from the solution at some first position, so it lies in exactly one
//! region — the tiling is a partition, and no sequence is enumerated twice or
//! skipped. Each region is one [`crate::design_restricted`] solve, and the best
//! unexamined region's optimum is the next rank.
//!
//! Cost is one solve per region, and a rank spawns one region per *free*
//! position — `m` of them at the root, but fewer as the walk descends and the
//! pinned prefix lengthens. The `m`-per-rank figure is therefore an upper bound
//! reached only at the start; measured over a 78-residue protein the amortized
//! cost is closer to `0.3·m`. That is the price of keeping memory flat: the
//! alternative — k-best lists in every CKY cell — pays `O(k)` memory across the
//! whole lattice instead, which at corpus length is the binding limit rather
//! than time.
//!
//! Regions within a rank are independent, so they are solved concurrently while
//! each individual solve stays single-threaded. That keeps every recovered
//! structure deterministic, which matters when the caller's predicate reads the
//! structure rather than only the energy.
//!
//! **Cuts are an optional accelerator, and they are not part of the exactness
//! argument.** [`Enumerator::cut`] forbids a pattern at a window for the rest of
//! the walk, so a caller that has rejected a candidate can stop the walk
//! respawning the family it came from. The partition is what makes the walk
//! exact; a cut only removes sequences from inside the regions it already
//! tiles — and whether the removed ones are all sequences the caller would have
//! rejected is a property of the caller's predicate, not of this module. Running
//! the same walk with and without cuts and comparing the two answers is what
//! prices that.

use std::collections::BinaryHeap;

use derna_codon_dp::codon_table::AMINO_ACIDS;

use crate::ac::MotifWindow;
use crate::{
    design_restricted, design_restricted_with_placed_motifs, BeamError, CodonUsage, Design, Params,
    RegionConstraints, ScanConfig,
};

/// One region of the design space: positions before `prefix.len()` are pinned to
/// `prefix`, the position at `prefix.len()` may not take any codon in
/// `excluded`, and every later position is free.
#[derive(Clone, Debug)]
struct Region {
    prefix: Vec<[u8; 3]>,
    excluded: Vec<[u8; 3]>,
}

impl Region {
    fn root() -> Self {
        Self {
            prefix: Vec::new(),
            excluded: Vec::new(),
        }
    }

    /// The per-position codon sets this region admits.
    fn allowed(&self, aa: &[usize]) -> Vec<Vec<[u8; 3]>> {
        let branch = self.prefix.len();
        aa.iter()
            .enumerate()
            .map(|(i, &a)| {
                let all = || AMINO_ACIDS[a].codons.iter().map(|c| c.nuc);
                match i.cmp(&branch) {
                    std::cmp::Ordering::Less => vec![self.prefix[i]],
                    std::cmp::Ordering::Equal => {
                        all().filter(|c| !self.excluded.contains(c)).collect()
                    }
                    std::cmp::Ordering::Greater => all().collect(),
                }
            })
            .collect()
    }

    /// The regions tiling everything in `self` except `solution`.
    fn partition(&self, solution: &[[u8; 3]]) -> Vec<Region> {
        let branch = self.prefix.len();
        (branch..solution.len())
            .map(|j| {
                let mut excluded = if j == branch {
                    self.excluded.clone()
                } else {
                    Vec::new()
                };
                excluded.push(solution[j]);
                Region {
                    prefix: solution[..j].to_vec(),
                    excluded,
                }
            })
            .collect()
    }
}

/// A solved region, ordered so the heap yields the least objective first.
struct Solved {
    objective: f64,
    design: Design,
    region: Region,
    /// How many cuts were in force when this region was solved. A heap entry
    /// stamped below the current count holds an optimum a later cut may have
    /// removed, so it is re-solved before it can be emitted.
    stamp: usize,
}

impl PartialEq for Solved {
    fn eq(&self, other: &Self) -> bool {
        self.objective == other.objective
    }
}
impl Eq for Solved {}
impl PartialOrd for Solved {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Solved {
    /// Reversed: [`BinaryHeap`] is a max-heap, and the search wants the minimum.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.objective.total_cmp(&self.objective)
    }
}

/// What to enumerate over: the protein, its flanks, and the objective weight.
///
/// The flanks are folded jointly with the CDS but are not part of the design
/// space, so they only enter through the energy.
pub struct Problem<'a> {
    pub protein: &'a str,
    pub aa: &'a [usize],
    pub usage: &'a CodonUsage,
    pub lambda: f64,
    pub utr5: &'a str,
    pub utr3: &'a str,
    /// Worker threads for the independent per-rank solves. Each solve stays
    /// single-threaded regardless, so results do not depend on this.
    pub threads: usize,
}

/// A best-first walk over the design space.
///
/// [`Enumerator::next_design`] returns designs in non-decreasing objective
/// order, each a distinct coding sequence, until the space is exhausted.
pub struct Enumerator<'a> {
    params: &'a Params,
    problem: Problem<'a>,
    heap: BinaryHeap<Solved>,
    solves: usize,
    /// Patterns forbidden at the windows they were cut at, in the order they
    /// were placed. Empty on every walk that never calls [`Enumerator::cut`],
    /// which routes each solve through [`design_restricted`] verbatim.
    cuts: Vec<(String, MotifWindow)>,
}

impl<'a> Enumerator<'a> {
    /// Start the walk. `Ok(None)` when the protein admits no encoding at all.
    pub fn new(params: &'a Params, problem: Problem<'a>) -> Result<Option<Self>, BeamError> {
        let mut e = Enumerator {
            params,
            problem,
            heap: BinaryHeap::new(),
            solves: 0,
            cuts: Vec::new(),
        };
        match e.solve(&Region::root())? {
            Some(root) => {
                e.heap.push(root);
                Ok(Some(e))
            }
            None => Ok(None),
        }
    }

    /// How many restricted solves the walk has run — the real cost measure.
    pub fn solves(&self) -> usize {
        self.solves
    }

    /// Forbid `pattern` over `window` for the rest of the walk.
    ///
    /// The walk itself is exact without this: the regions tile the space, so
    /// every design is examined in objective order whether or not anything is
    /// cut. A cut is a **speed** move — it stops the walk respawning a family
    /// its caller has already rejected, at the price of re-solving the regions
    /// that were solved before it landed.
    ///
    /// That trade is sound only where every sequence the cut removes is one the
    /// caller's predicate would have rejected anyway. Nothing here checks that,
    /// because nothing here can: the predicate is the caller's, and it reads the
    /// finished molecule. A caller pricing the accelerator runs the same walk
    /// with and without cuts and compares the two answers — they agree exactly
    /// when no cut removed something it should not have.
    pub fn cut(&mut self, pattern: String, window: MotifWindow) {
        self.cuts.push((pattern, window));
    }

    /// How many cuts are in force.
    pub fn cuts(&self) -> usize {
        self.cuts.len()
    }

    /// The next design in objective order, or `None` when exhausted.
    pub fn next_design(&mut self) -> Result<Option<Design>, BeamError> {
        let top = loop {
            let Some(top) = self.heap.pop() else {
                return Ok(None);
            };
            if top.stamp == self.cuts.len() {
                break top;
            }
            // Solved before a cut landed, so its optimum may be one the cut
            // removed. Re-solving the region restores the invariant that a
            // popped entry is its region's optimum under everything cut so far;
            // a region the cuts have emptied drops out of the heap here.
            if let Some(fresh) = self.solve(&top.region)? {
                self.heap.push(fresh);
            }
        };
        let children = top.region.partition(&self.codons(&top.design));
        for solved in self.solve_all(&children)? {
            self.heap.push(solved);
        }
        Ok(Some(top.design))
    }

    fn objective(&self, d: &Design) -> f64 {
        let mfe_dk = (d.mfe_kcal_mol * 100.0).round();
        let cai_dk = (d.cai_log * 100.0).round();
        self.problem.lambda * mfe_dk - (1.0 - self.problem.lambda) * cai_dk
    }

    fn solve(&mut self, region: &Region) -> Result<Option<Solved>, BeamError> {
        self.solves += 1;
        let stamp = self.cuts.len();
        Ok(
            solve_region(self.params, &self.problem, region, &self.cuts)?.map(|design| Solved {
                objective: self.objective(&design),
                design,
                region: region.clone(),
                stamp,
            }),
        )
    }

    /// Solve `regions` concurrently, one single-threaded solve each.
    fn solve_all(&mut self, regions: &[Region]) -> Result<Vec<Solved>, BeamError> {
        self.solves += regions.len();
        let threads = self.problem.threads.max(1);
        let chunk = regions.len().div_ceil(threads).max(1);
        let (params, problem, cuts) = (self.params, &self.problem, &self.cuts);
        let stamp = cuts.len();
        let mut out = Vec::new();
        let batches: Vec<Result<Vec<Option<Design>>, BeamError>> = std::thread::scope(|s| {
            let handles: Vec<_> = regions
                .chunks(chunk)
                .map(|batch| {
                    crate::task_clock::spawn(s, move || {
                        batch
                            .iter()
                            .map(|r| solve_region(params, problem, r, cuts))
                            .collect::<Result<Vec<_>, _>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("solver thread"))
                .collect()
        });
        for (batch, regions) in batches.into_iter().zip(regions.chunks(chunk)) {
            for (design, region) in batch?.into_iter().zip(regions) {
                if let Some(design) = design {
                    out.push(Solved {
                        objective: self.objective(&design),
                        design,
                        region: region.clone(),
                        stamp,
                    });
                }
            }
        }
        Ok(out)
    }

    /// The CDS codons a design spells.
    fn codons(&self, d: &Design) -> Vec<[u8; 3]> {
        let start = self.problem.utr5.len();
        let end = start + 3 * self.problem.aa.len();
        let b = d.sequence.as_bytes();
        (start..end)
            .step_by(3)
            .map(|i| {
                let idx = |k: usize| {
                    [b'A', b'C', b'G', b'U']
                        .iter()
                        .position(|&n| n == b[k])
                        .expect("A/C/G/U") as u8
                };
                [idx(i), idx(i + 1), idx(i + 2)]
            })
            .collect()
    }
}

fn solve_region(
    params: &Params,
    problem: &Problem<'_>,
    region: &Region,
    cuts: &[(String, MotifWindow)],
) -> Result<Option<Design>, BeamError> {
    let allowed = region.allowed(problem.aa);
    if cuts.is_empty() {
        // The floor: a walk that cuts nothing takes the entry it always took, so
        // its designs and its solve count do not depend on this axis existing.
        return design_restricted(
            params,
            problem.protein,
            problem.usage,
            problem.lambda,
            &allowed,
            problem.utr5,
            problem.utr3,
            ScanConfig::default(),
        );
    }
    design_restricted_with_placed_motifs(
        params,
        problem.protein,
        problem.usage,
        problem.lambda,
        &allowed,
        cuts,
        problem.utr5,
        problem.utr3,
        RegionConstraints::default(),
        ScanConfig::default(),
    )
}
