//! Every coding sequence tied at one point of the coding-only Pareto front.
//!
//! A front point `(E, C)` is spelled by one design, but other designs may reach
//! exactly the same energy and adaptation. Which of them is in hand matters
//! whenever a property of the *sequence* is asked of the point — whether the
//! molecule's free fold attains its constrained energy, say — because the answer
//! can differ between spellings of one point.
//!
//! **The walk.** Lawler partitioning around each spelling found, the same
//! tiling [`crate::kbest`] uses: every other sequence first differs from a
//! found one at some codon position `j`, so it lies in exactly one region that
//! pins the positions before `j` and forbids the found codon at `j`. A region
//! holds a tied design iff its own front reaches `(E, C)`, and because `(E, C)`
//! is non-dominated over the whole space it is non-dominated over the region
//! too, so the region's front contains it whenever any of the region's designs
//! spells it. Each region is decided by, cheapest first:
//!
//! 1. **the adaptation ceiling** — the sum of each position's best remaining
//!    codon; below `C` and the region cannot reach `C`, with no fold at all;
//! 2. **a degenerate box** — the bounded front fold with its energy ceiling at
//!    `E` and its adaptation floor at `C`, under the exact credit. Its
//!    `λ = 1` fold decides first whether the region reaches `E` at all.
//!
//! Every region that yields a tied design is partitioned around it in turn, so
//! the walk enumerates each tied sequence exactly once and stops when no region
//! is left — or at a cap, when it reports how many it found and that it did not
//! finish.
//!
//! **Equality.** Energies are integers. Adaptation is a floating-point sum
//! whose order follows the codon positions, so two designs using the same
//! codons at different positions can differ in the last bits; two values within
//! [`CAI_TIE`] are the same point.

use derna_codon_dp::codon_table::AMINO_ACIDS;
use rna_mfe::{Energy, Params};

use crate::{
    dfa, residues, solve_bounded, BeamError, Bounded, Bounds, BoxSpec, Ceiling, CodonUsage, Prune,
    UtrProfile,
};

/// Adaptation values closer than this are one point.
pub const CAI_TIE: f64 = 1e-9;

/// What a tie walk found.
#[derive(Debug, Clone, PartialEq)]
pub struct Ties {
    /// Every tied coding sequence found, the given one first.
    pub spellings: Vec<String>,
    /// Whether the walk ran out of regions: then `spellings` is every design at
    /// the point. `false` means it stopped at the cap.
    pub exhausted: bool,
    /// Regions decided by a fold.
    pub folds: usize,
    /// Regions decided by the adaptation ceiling alone.
    pub skipped: usize,
}

/// One region of the design space: positions before `prefix.len()` pinned to
/// `prefix`, the next one forbidden every codon in `excluded`, the rest free.
#[derive(Clone, Debug)]
struct Region {
    prefix: Vec<[u8; 3]>,
    excluded: Vec<[u8; 3]>,
}

impl Region {
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

/// A coding sequence as codon index triples.
fn codons(sequence: &str) -> Result<Vec<[u8; 3]>, BeamError> {
    let index = |b: u8| match b {
        b'A' => Ok(0),
        b'C' => Ok(1),
        b'G' => Ok(2),
        b'U' | b'T' => Ok(3),
        other => Err(BeamError::InvalidNucleotide(other)),
    };
    sequence
        .as_bytes()
        .chunks_exact(3)
        .map(|c| Ok([index(c[0])?, index(c[1])?, index(c[2])?]))
        .collect()
}

/// The design in `region` at `(energy, cai_log)`, if any.
fn tied_in(
    params: &Params,
    aa: &[usize],
    usage: &CodonUsage,
    allowed: &[Vec<[u8; 3]>],
    energy: Energy,
    cai_log: f64,
) -> Result<Option<String>, BeamError> {
    let Some(graph) = dfa::build_restricted_with_utr(aa, usage, allowed, &[], &[]) else {
        return Ok(None);
    };
    let band = BoxSpec {
        cai_floor_log: cai_log - CAI_TIE,
        lambdas: vec![0.0],
    };
    let solved = solve_bounded(
        params,
        &graph,
        UtrProfile::inert(),
        || Err(BeamError::Infeasible),
        Prune::None,
        Bounds::with_ceiling(Ceiling::Absolute(energy)).with_exact_credit(),
        1,
        Some(&band),
    )?;
    Ok(match solved {
        Bounded::Unreachable { .. } => None,
        Bounded::Feasible(front) => front
            .front
            .into_iter()
            .find(|p| {
                (p.mfe_kcal_mol * 100.0).round() as i32 == energy.0
                    && (p.cai_log - cai_log).abs() < CAI_TIE
            })
            .map(|p| p.sequence),
    })
}

/// Every coding sequence of `protein` at the coding-only front point that
/// `spelling` reaches at `energy`, up to `cap` of them.
///
/// The point's adaptation is taken from `spelling` itself, on the scale the
/// fold accumulates, so a caller holding a rounded copy of it loses nothing.
/// `spelling` must spell a point of the front; the walk is exact only there,
/// where no region's design can dominate it.
///
/// # Errors
///
/// [`BeamError::InvalidAminoAcid`] or [`BeamError::InvalidNucleotide`] for
/// malformed input, and any error a region's solve returns.
pub fn front_point_ties(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    spelling: &str,
    energy: Energy,
    cap: usize,
) -> Result<Ties, BeamError> {
    let aa = residues(protein)?;
    let pinned: Vec<Vec<[u8; 3]>> = codons(spelling)?.into_iter().map(|c| vec![c]).collect();
    let cai_log = dfa::cai_ceiling(&aa, usage, &pinned).ok_or(BeamError::Infeasible)?;
    let mut skipped = 0;
    let mut ties = walk(&aa, spelling, cap, |allowed| {
        let ceiling = dfa::cai_ceiling(&aa, usage, allowed);
        if ceiling.is_none_or(|c| c < cai_log - CAI_TIE) {
            skipped += 1;
            return Ok(None);
        }
        tied_in(params, &aa, usage, allowed, energy, cai_log).map(Some)
    })?;
    ties.skipped = skipped;
    ties.folds -= skipped;
    Ok(ties)
}

/// [`front_point_ties`] restricted to **codon permutations**: tied spellings
/// whose codon weights, amino acid by amino acid, form the same multiset as
/// `spelling`'s.
///
/// Two such designs have the same adaptation exactly, whatever their order, so
/// every spelling this finds is a true tie. The restriction is what it can
/// miss: a design reaching the same adaptation from a *different* multiset of
/// weights, which needs the log-weights to satisfy an integer relation to
/// within [`CAI_TIE`]. On every case where both walks run they agree; a caller
/// citing this walk's count states the restriction.
///
/// The price is what it buys: a region that cannot hold the multiset — its
/// pinned codons overuse a weight, or its branch position can take no weight
/// the rest still needs — is decided by counting, with no fold.
///
/// # Errors
///
/// As [`front_point_ties`].
pub fn front_point_permutation_ties(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    spelling: &str,
    energy: Energy,
    cap: usize,
) -> Result<Ties, BeamError> {
    let aa = residues(protein)?;
    let found = codons(spelling)?;
    let pinned: Vec<Vec<[u8; 3]>> = found.iter().map(|&c| vec![c]).collect();
    let cai_log = dfa::cai_ceiling(&aa, usage, &pinned).ok_or(BeamError::Infeasible)?;
    // A codon's weight class at a residue: its log-weight's bit pattern.
    let class = |a: usize, c: [u8; 3]| -> u64 {
        dfa::cai_ceiling(&[a], usage, &[vec![c]])
            .unwrap_or(f64::NAN)
            .to_bits()
    };
    let mut target: std::collections::HashMap<(usize, u64), i64> = Default::default();
    for (&a, &c) in aa.iter().zip(&found) {
        *target.entry((a, class(a, c))).or_default() += 1;
    }
    let holds = |allowed: &[Vec<[u8; 3]>]| -> bool {
        let mut left = target.clone();
        let mut partial = Vec::new();
        for (&a, codons) in aa.iter().zip(allowed) {
            let full = AMINO_ACIDS[a].codons.len();
            if codons.len() == 1 {
                let slot = left.entry((a, class(a, codons[0]))).or_default();
                *slot -= 1;
                if *slot < 0 {
                    return false;
                }
            } else if codons.len() < full {
                partial.push((a, codons));
            }
        }
        // The walk leaves at most one partial position: the branch. It needs a
        // weight the rest still owes; free positions then absorb the remainder.
        partial.iter().all(|(a, codons)| {
            codons
                .iter()
                .any(|&c| left.get(&(*a, class(*a, c))).is_some_and(|&k| k > 0))
        })
    };
    let mut skipped = 0;
    let mut ties = walk(&aa, spelling, cap, |allowed| {
        if !holds(allowed) {
            skipped += 1;
            return Ok(None);
        }
        tied_in(params, &aa, usage, allowed, energy, cai_log).map(Some)
    })?;
    ties.skipped = skipped;
    ties.folds -= skipped;
    Ok(ties)
}

/// Whether a point of the fully isolated constrained front is modular
/// , over its tied spellings.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// This spelling's molecule folds freely to its constrained energy.
    Certified {
        /// The certifying coding sequence.
        spelling: String,
        /// Spellings folded, this one included.
        checked: usize,
    },
    /// Every tied spelling was folded and none is in S.
    Refuted {
        /// Spellings folded: all of them.
        checked: usize,
    },
    /// The tie walk stopped at its cap with no spelling in S.
    Unresolved {
        /// Spellings folded.
        checked: usize,
    },
}

/// The verdict for the front point `spelling` reaches at `energy`, between
/// the fixed flanks `utr5` and `utr3`, under the constraint forbidding all three
/// crossing classes.
///
/// Under that constraint a design's constrained energy is its coding sequence's own MFE
/// plus each flank's alone (no pair leaves its region, and at `dangles = 0`
/// the exterior loop scores its branches independently), so the constrained front
/// is the coding-only front shifted, and its ties are the coding-only ties. A
/// spelling is in S when its whole molecule's free MFE reaches that energy.
/// Spellings are tried in the order the tie walk finds them, `spelling` first,
/// up to `cap`.
///
/// # Errors
///
/// As [`front_point_ties`], plus a free fold's failure as
/// [`BeamError::InvalidNucleotide`].
#[allow(clippy::too_many_arguments)]
pub fn certify(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    spelling: &str,
    energy: Energy,
    utr5: &str,
    utr3: &str,
    cap: usize,
) -> Result<Verdict, BeamError> {
    let free = |molecule: &str| -> Result<i32, BeamError> {
        if molecule.is_empty() {
            return Ok(0);
        }
        rna_mfe::fold(params, molecule.as_bytes())
            .map(|(_, e)| e.0)
            .map_err(|_| BeamError::InvalidNucleotide(b'?'))
    };
    let constrained = energy.0 + free(utr5)? + free(utr3)?;
    let ties = front_point_ties(params, protein, usage, spelling, energy, cap)?;
    for (k, s) in ties.spellings.iter().enumerate() {
        if free(&format!("{utr5}{s}{utr3}"))? == constrained {
            return Ok(Verdict::Certified {
                spelling: s.clone(),
                checked: k + 1,
            });
        }
    }
    let checked = ties.spellings.len();
    Ok(if ties.exhausted {
        Verdict::Refuted { checked }
    } else {
        Verdict::Unresolved { checked }
    })
}

/// The partition walk around `spelling`, with `decide` saying which design,
/// if any, a region holds: `Ok(None)` for a region skipped without a fold,
/// `Ok(Some(None))` for one folded and found empty, `Ok(Some(Some(s)))` for one
/// holding `s`. `folds` counts every region `decide` was asked about.
fn walk(
    aa: &[usize],
    spelling: &str,
    cap: usize,
    mut decide: impl FnMut(&[Vec<[u8; 3]>]) -> Result<Option<Option<String>>, BeamError>,
) -> Result<Ties, BeamError> {
    let mut ties = Ties {
        spellings: vec![spelling.to_string()],
        exhausted: false,
        folds: 0,
        skipped: 0,
    };
    let mut open = Region {
        prefix: Vec::new(),
        excluded: Vec::new(),
    }
    .partition(&codons(spelling)?);
    while let Some(region) = open.pop() {
        if ties.spellings.len() >= cap {
            return Ok(ties);
        }
        ties.folds += 1;
        if let Some(Some(found)) = decide(&region.allowed(aa))? {
            open.extend(region.partition(&codons(&found)?));
            ties.spellings.push(found);
        }
    }
    ties.exhausted = true;
    Ok(ties)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every coding sequence a per-position codon list admits.
    fn expand(allowed: &[Vec<[u8; 3]>]) -> Vec<String> {
        let mut out = vec![String::new()];
        for codons in allowed {
            out = out
                .iter()
                .flat_map(|prefix| {
                    codons.iter().map(move |c| {
                        let mut s = prefix.clone();
                        s.extend(c.iter().map(|&n| ['A', 'C', 'G', 'U'][n as usize]));
                        s
                    })
                })
                .collect();
        }
        out
    }

    /// The walk finds exactly a planted set, each member once, when several
    /// members share the first region the walk opens: the second and later
    /// ones are reached only through the partition around the first found.
    #[test]
    fn the_walk_finds_every_planted_tie_exactly_once() {
        // K (2 codons) x5: 32 sequences.
        let aa: Vec<usize> = "KKKKK"
            .chars()
            .map(|c| derna_codon_dp::codon_table::aa_to_index(c).unwrap())
            .collect();
        let all = |_: usize| AMINO_ACIDS[aa[0]].codons.iter().map(|c| c.nuc).collect();
        let everything = expand(&(0..aa.len()).map(all).collect::<Vec<_>>());
        assert_eq!(everything.len(), 32);
        let start = everything[0].clone();
        let planted: Vec<String> = [5, 6, 7, 21, 30]
            .iter()
            .map(|&i| everything[i].clone())
            .collect();
        let ties = walk(&aa, &start, usize::MAX, |allowed| {
            let region = expand(allowed);
            Ok(Some(planted.iter().find(|p| region.contains(p)).cloned()))
        })
        .unwrap();
        assert!(ties.exhausted);
        let mut got = ties.spellings[1..].to_vec();
        got.sort();
        let mut want = planted.clone();
        want.sort();
        assert_eq!(got, want);
    }
}
