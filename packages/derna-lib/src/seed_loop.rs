//! Counterexample-guided seeding of tight items from the class scan.
//!
//! The class scan ([`crate::class_bound::class_optimum`]) returns, at one λ,
//! the item and in-class configuration of least scalarized score. That item is
//! a *seed*: tight when its free fold between the two contexts costs exactly
//! its constrained fold, `F = E`, and otherwise a counterexample. This loop reads
//! each counterexample for the reason it is not tight, turns the reason into a
//! cut, and solves the class scan again over what the cuts leave, until a seed
//! is tight, the class is empty over what is left, no cut applies, or the
//! round budget runs out.
//!
//! # Two mechanisms, and the cut each one licenses
//!
//! - **The seed's own optimum leaves the class** ([`Mechanism::OutsideClass`]).
//!   The item alone folds cheaper than its in-class configuration, so the
//!   rival is item-internal: the pairs of its own optimum the in-class
//!   configuration lacks, and the in-class stems the own optimum drops (the
//!   class needs a stem the item alone does not hold). The cut excludes the
//!   seed's choice at a position one of those pairs uses — nearest the symbols
//!   a context pairs first (the context binds where the item's configuration
//!   changed), then outermost first (the outermost loop is what the class
//!   constrains).
//! - **The seed is in the class but a cross pair beats it.** Pairs between a
//!   context and the item lower the free fold below the constrained one. Where an
//!   item position the context pairs has more than one choice, the cut excludes
//!   the seed's choice there — and every other choice still allowed there that
//!   could pair the same context symbol, unless that would leave none — one
//!   position per round, nearest an item end first ([`Mechanism::CrossChoice`]). Where every such position is fixed (a single
//!   choice — nothing there can be excluded), the cut must make the item's own
//!   configuration *pair* those positions instead: the class's leading run (for
//!   positions in the item's first half) or trailing run (second half) is
//!   capped so that the first (last) top-level pair reaches them
//!   ([`Mechanism::CrossFixed`]). A seed whose own optimum then still
//!   leaves them free is an [`Mechanism::OutsideClass`] counterexample of the
//!   capped class, and the next round cuts its rival stem.
//!
//! A counterexample matching neither — a cross pair at fixed positions the caps
//! already enclose, or a pairing of the two contexts alone — falls back to an
//! item position whose partner differs between the free fold and the item's
//! own optimum ([`Mechanism::Refold`]); with none, the loop reports
//! [`LoopOutcome::NoCut`].
//!
//! # What the loop guarantees
//!
//! - **Sound.** A returned item is tight by two exact folds, and the
//!   configurations realizing both energies are re-scored by
//!   [`rna_mfe::evaluate`] ([`Certificate`]); it is a valid upper-side point.
//! - **Not complete.** A codon exclusion removes every item using that choice at
//!   that position for the rest of the loop, valid ones included, and a cap
//!   removes every configuration outside it. The loop can end empty while a
//!   tight item exists.
//! - **Not optimal.** A cut can remove the cheapest tight item at this λ, so a
//!   returned item carries no claim of being the best at its weight.
//!
//! Every cut strictly shrinks a finite set (an excluded choice is one the seed
//! used, so it was still allowed; a cap only tightens), so the loop ends within
//! the total number of choices plus twice the item length rounds even without a
//! budget.

use std::time::Instant;

use derna_codon_dp::codon_table::AMINO_ACIDS;
use rna_mfe::Params;

use crate::class_bound::{class_optimum, ExteriorClass};
use crate::dfa::{self, UtrChain};
use crate::{scan, traceback, CodonUsage, ScanConfig, UtrProfile};

/// Why a seed was not tight, as the cut made from it names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mechanism {
    /// The item's own optimum is cheaper than its in-class configuration.
    OutsideClass,
    /// A cross pair at an item position with more than one choice.
    CrossChoice,
    /// Cross pairs at fixed item positions only; the class was capped.
    CrossFixed,
    /// Neither: an item position the free fold re-pairs.
    Refold,
}

impl Mechanism {
    /// A short label for reports.
    pub fn label(self) -> &'static str {
        match self {
            Self::OutsideClass => "outside-class",
            Self::CrossChoice => "cross-choice",
            Self::CrossFixed => "cross-fixed",
            Self::Refold => "refold",
        }
    }
}

/// One cut.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cut {
    /// The choice `codon` is excluded at `position` for the rest of the loop.
    Exclude { position: usize, codon: String },
    /// The class's leading unmatched run is capped at this many symbols.
    LeadCap(u32),
    /// The class's trailing unmatched run is capped at this many symbols.
    TrailCap(u32),
}

/// One round: the seed solved, its two folds, and the cut made from it.
#[derive(Debug, Clone)]
pub struct Round {
    /// The seed item.
    pub item: String,
    /// Its in-class configuration.
    pub class_structure: String,
    /// The in-class configuration's cost (centiunits, re-scored).
    pub class_cost: i32,
    /// The item's own optimum.
    pub own_structure: String,
    /// The free optimum of the whole molecule.
    pub free_structure: String,
    /// The item's own minimum (centiunits).
    pub own_cost: i32,
    /// Constrained cost `E` of the whole molecule (centiunits).
    pub constrained: i32,
    /// Free cost `F` of the whole molecule (centiunits).
    pub free: i32,
    /// The cuts made, all of one mechanism (at most the loop's width); empty
    /// on the last round.
    pub cuts: Vec<(Mechanism, Cut)>,
    /// Seconds in the class scan.
    pub scan_s: f64,
    /// Seconds in the two folds.
    pub fold_s: f64,
}

impl Round {
    /// `E − F >= 0`, in centiunits.
    pub fn gap(&self) -> i32 {
        self.constrained - self.free
    }
}

/// How the loop ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopOutcome {
    /// The last seed is tight and certified.
    Tight,
    /// The class is empty over the choices and caps the cuts left (or a
    /// position has no choice left). Says nothing about whether a tight item
    /// exists: the cuts are not a partition.
    Exhausted,
    /// The last seed is not tight and no cut applies to it.
    NoCut,
    /// The round budget ran out while still cutting.
    Budget,
}

impl LoopOutcome {
    /// A short label for reports.
    pub fn label(self) -> &'static str {
        match self {
            Self::Tight => "tight",
            Self::Exhausted => "exhausted",
            Self::NoCut => "no-cut",
            Self::Budget => "budget",
        }
    }
}

/// The evidence that a returned item is tight, every energy re-scored by
/// [`rna_mfe::evaluate`] from the configuration that realizes it.
#[derive(Debug, Clone)]
pub struct Certificate {
    /// The whole molecule (context, item, context).
    pub molecule: String,
    /// The free optimum of the whole molecule.
    pub free_structure: String,
    /// The constrained configuration: each context's own optimum beside the item's.
    pub constrained_structure: String,
    /// `evaluate(molecule, free_structure)` (centiunits).
    pub free_rescored: i32,
    /// `evaluate(molecule, constrained_structure)` (centiunits).
    pub constrained_rescored: i32,
}

impl Certificate {
    /// Whether the two re-scored energies agree and match the folds.
    pub fn holds(&self, free: i32, constrained: i32) -> bool {
        self.free_rescored == free
            && self.constrained_rescored == constrained
            && free == constrained
    }
}

/// What one loop did.
#[derive(Debug, Clone)]
pub struct LoopReport {
    pub outcome: LoopOutcome,
    /// Every round, in order.
    pub rounds: Vec<Round>,
    /// On [`LoopOutcome::Tight`], the certificate of the last seed.
    pub certificate: Option<Certificate>,
    /// The class as the caps left it.
    pub class: ExteriorClass,
    pub secs: f64,
}

impl LoopReport {
    /// The tight item, when the loop returned one.
    pub fn item(&self) -> Option<&str> {
        match self.outcome {
            LoopOutcome::Tight => self.rounds.last().map(|r| r.item.as_str()),
            _ => None,
        }
    }

    /// The mechanism of each round's cuts, in order.
    pub fn mechanisms(&self) -> Vec<Mechanism> {
        self.rounds
            .iter()
            .filter_map(|r| r.cuts.first().map(|(m, _)| *m))
            .collect()
    }
}

/// Errors of the loop.
#[derive(Debug, thiserror::Error)]
pub enum SeedLoopError {
    #[error("fold failed: {0}")]
    Fold(#[from] rna_mfe::FoldError),
    #[error("re-scoring failed: {0}")]
    Eval(#[from] rna_mfe::EvalError),
    /// The class path's configuration does not re-score to the class value.
    #[error("the in-class configuration re-scores to {0}, the class value is {1}")]
    Rescore(i32, i32),
}

/// One protein between two fixed contexts, and how to solve its class scan.
pub struct SeedLoop<'a> {
    params: &'a Params,
    usage: &'a CodonUsage,
    aa: Vec<usize>,
    utr5: &'a str,
    utr3: &'a str,
    utr5_fold: (String, i32),
    utr3_fold: (String, i32),
    threads: usize,
    width: usize,
}

/// A configuration's pairs as `(open, close)`, with each pair's depth (0 for
/// a top-level pair).
fn pairs(structure: &str) -> Vec<(usize, usize, usize)> {
    let mut stack = Vec::new();
    let mut out = Vec::new();
    for (k, ch) in structure.bytes().enumerate() {
        match ch {
            b'(' => stack.push(k),
            b')' => {
                let i = stack.pop().expect("a balanced structure");
                out.push((i, k, stack.len()));
            }
            _ => {}
        }
    }
    out
}

/// Each symbol's partner, if paired.
fn partners(structure: &str) -> Vec<Option<usize>> {
    let mut out = vec![None; structure.len()];
    for (i, j, _) in pairs(structure) {
        out[i] = Some(j);
        out[j] = Some(i);
    }
    out
}

impl<'a> SeedLoop<'a> {
    /// A loop over `aa` (amino-acid indices) between `utr5` and `utr3`, its
    /// scans on `threads` threads.
    pub fn new(
        params: &'a Params,
        usage: &'a CodonUsage,
        aa: Vec<usize>,
        (utr5, utr3): (&'a str, &'a str),
        threads: usize,
    ) -> Result<Self, SeedLoopError> {
        let fold = |s: &str| -> Result<(String, i32), SeedLoopError> {
            let (st, e) = rna_mfe::fold(params, s.as_bytes())?;
            Ok((st, e.0))
        };
        Ok(Self {
            params,
            usage,
            utr5_fold: fold(utr5)?,
            utr3_fold: fold(utr3)?,
            aa,
            utr5,
            utr3,
            threads: threads.max(1),
            width: 1,
        })
    }

    /// Make up to `width` exclusions per round (default 1), all from the
    /// first mechanism that applies, in its own order. A wider round spends
    /// fewer class scans and discards more of the item set with each.
    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width.max(1);
        self
    }

    fn menu(&self, p: usize) -> &'static [derna_codon_dp::codon_table::Codon] {
        AMINO_ACIDS[self.aa[p]].codons
    }

    /// The class minimizer over the choices `excluded` leaves: item, in-class
    /// configuration, its re-scored cost. `None` when the class or a
    /// position's menu is empty.
    fn solve(
        &self,
        lambda: f64,
        class: &ExteriorClass,
        excluded: &[Vec<String>],
    ) -> Result<Option<(String, String, i32)>, SeedLoopError> {
        let allowed: Vec<Vec<[u8; 3]>> = (0..self.aa.len())
            .map(|p| {
                self.menu(p)
                    .iter()
                    .filter(|c| !excluded[p].iter().any(|x| x == c.name))
                    .map(|c| c.nuc)
                    .collect()
            })
            .collect();
        let Some(graph) = dfa::build_composite(
            &self.aa,
            self.usage,
            Some(&allowed),
            UtrChain::Fixed(&[]),
            UtrChain::Fixed(&[]),
            UtrProfile::inert(),
        ) else {
            return Ok(None);
        };
        let config = ScanConfig {
            threads: self.threads,
            ..ScanConfig::default()
        };
        let mut tables = scan::scan(self.params, &graph, lambda, config);
        let opt = class_optimum(self.params, &graph, &tables, class);
        let Some(best) = opt.best else {
            return Ok(None);
        };
        let mut column = vec![None; tables.c.len()];
        for &(node, st) in &opt.path {
            column[graph.ix(node)] = Some(st);
        }
        std::mem::swap(&mut tables.c, &mut column);
        let rec = traceback::recover(self.params, &graph, &tables);
        let cost = rna_mfe::evaluate(self.params, rec.sequence.as_bytes(), &rec.structure)?.0;
        if cost != best.mfe.0 {
            return Err(SeedLoopError::Rescore(cost, best.mfe.0));
        }
        Ok(Some((rec.sequence, rec.structure, cost)))
    }

    /// Run the loop at `lambda` from `class`, for at most `max_rounds` rounds.
    pub fn run(
        &self,
        lambda: f64,
        class: &ExteriorClass,
        max_rounds: usize,
    ) -> Result<LoopReport, SeedLoopError> {
        self.run_from(lambda, class, vec![Vec::new(); self.aa.len()], max_rounds)
    }

    /// The choices left at every position except `item`'s own in its first
    /// `w5` and last `w3` positions: the exclusions that pin an item's two end
    /// windows, as [`Self::run_from`] takes them. A pinned position has one
    /// choice, so no cut excludes it; a cross pair there is a fixed-position
    /// counterexample.
    pub fn pin_ends(&self, item: &str, w5: usize, w3: usize) -> Vec<Vec<String>> {
        let n = self.aa.len();
        (0..n)
            .map(|p| {
                if p < w5 || p + w3 >= n {
                    let own = &item[3 * p..3 * p + 3];
                    self.menu(p)
                        .iter()
                        .map(|c| c.name.to_string())
                        .filter(|c| c != own)
                        .collect()
                } else {
                    Vec::new()
                }
            })
            .collect()
    }

    /// [`Self::run`] over the choices `excluded` leaves from the start (its
    /// entries are codon names per position), e.g. [`Self::pin_ends`].
    pub fn run_from(
        &self,
        lambda: f64,
        class: &ExteriorClass,
        excluded: Vec<Vec<String>>,
        max_rounds: usize,
    ) -> Result<LoopReport, SeedLoopError> {
        let started = Instant::now();
        let mut class = class.clone();
        let mut excluded = excluded;
        let mut rounds: Vec<Round> = Vec::new();
        let mut outcome = LoopOutcome::Budget;
        let mut certificate = None;
        while rounds.len() < max_rounds {
            let t = Instant::now();
            let Some((item, class_structure, class_cost)) =
                self.solve(lambda, &class, &excluded)?
            else {
                outcome = LoopOutcome::Exhausted;
                break;
            };
            let scan_s = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let (own_structure, own) = rna_mfe::fold(self.params, item.as_bytes())?;
            let molecule = format!("{}{}{}", self.utr5, item, self.utr3);
            let (free_structure, free) = rna_mfe::fold(self.params, molecule.as_bytes())?;
            let fold_s = t.elapsed().as_secs_f64();
            let constrained = self.utr5_fold.1 + own.0 + self.utr3_fold.1;
            let mut round = Round {
                item,
                class_structure,
                class_cost,
                own_structure,
                free_structure,
                own_cost: own.0,
                constrained,
                free: free.0,
                cuts: Vec::new(),
                scan_s,
                fold_s,
            };
            if round.free == round.constrained {
                let constrained_structure = format!(
                    "{}{}{}",
                    self.utr5_fold.0, round.own_structure, self.utr3_fold.0
                );
                let rescore = |s: &str| rna_mfe::evaluate(self.params, molecule.as_bytes(), s);
                certificate = Some(Certificate {
                    free_rescored: rescore(&round.free_structure)?.0,
                    constrained_rescored: rescore(&constrained_structure)?.0,
                    free_structure: round.free_structure.clone(),
                    molecule,
                    constrained_structure,
                });
                rounds.push(round);
                outcome = LoopOutcome::Tight;
                break;
            }
            let cuts = self.cut(&round, &excluded, &class);
            for (_, cut) in &cuts {
                match cut {
                    Cut::Exclude { position, codon } => excluded[*position].push(codon.clone()),
                    Cut::LeadCap(a) => class.max_lead = Some(*a),
                    Cut::TrailCap(t) => class.max_trail = Some(*t),
                }
            }
            let stuck = cuts.is_empty();
            round.cuts = cuts;
            rounds.push(round);
            if stuck {
                outcome = LoopOutcome::NoCut;
                break;
            }
        }
        Ok(LoopReport {
            outcome,
            rounds,
            certificate,
            class,
            secs: started.elapsed().as_secs_f64(),
        })
    }

    /// An exclusion of the seed's choice at the codon spelling item symbol `k`,
    /// if that position has another choice left.
    fn exclusion(&self, item: &str, k: usize, excluded: &[Vec<String>]) -> Option<Cut> {
        let p = k / 3;
        let codon = &item[3 * p..3 * p + 3];
        let left = self
            .menu(p)
            .iter()
            .filter(|c| c.name != codon && !excluded[p].iter().any(|x| x == c.name))
            .count();
        (left > 0).then(|| Cut::Exclude {
            position: p,
            codon: codon.to_string(),
        })
    }

    /// The exclusions a cross pair at item symbol `k` licenses: every choice
    /// still allowed at its position whose symbol there could pair `partner`
    /// again (the seed's among them), so the next seed cannot re-form the pair
    /// by a synonym. When every allowed choice could, only the seed's.
    fn pairing_choices(
        &self,
        item: &str,
        k: usize,
        partner: u8,
        excluded: &[Vec<String>],
    ) -> Vec<Cut> {
        let (p, offset) = (k / 3, k % 3);
        let seed = &item[3 * p..3 * p + 3];
        let allowed: Vec<&str> = self
            .menu(p)
            .iter()
            .map(|c| c.name)
            .filter(|name| !excluded[p].iter().any(|x| x == name))
            .collect();
        let pairs_with = |a: u8| {
            matches!(
                (a, partner),
                (b'A', b'U')
                    | (b'U', b'A')
                    | (b'G', b'C')
                    | (b'C', b'G')
                    | (b'G', b'U')
                    | (b'U', b'G')
            )
        };
        let mut hit: Vec<&str> = allowed
            .iter()
            .copied()
            .filter(|name| pairs_with(name.as_bytes()[offset]))
            .collect();
        if hit.len() == allowed.len() || !hit.contains(&seed) {
            hit = vec![seed];
        }
        hit.into_iter()
            .map(|codon| Cut::Exclude {
                position: p,
                codon: codon.to_string(),
            })
            .collect()
    }

    /// Up to [`Self::width`] exclusions at the item symbols `ks`, in order,
    /// one per position.
    fn exclusions(
        &self,
        item: &str,
        ks: impl IntoIterator<Item = usize>,
        excluded: &[Vec<String>],
        mechanism: Mechanism,
    ) -> Vec<(Mechanism, Cut)> {
        let mut out: Vec<(Mechanism, Cut)> = Vec::new();
        for k in ks {
            if out.len() == self.width {
                break;
            }
            let Some(cut) = self.exclusion(item, k, excluded) else {
                continue;
            };
            if !out.iter().any(|(_, c)| *c == cut) {
                out.push((mechanism, cut));
            }
        }
        out
    }

    /// The cuts a non-tight seed licenses, from the first mechanism that
    /// applies.
    fn cut(
        &self,
        round: &Round,
        excluded: &[Vec<String>],
        class: &ExteriorClass,
    ) -> Vec<(Mechanism, Cut)> {
        let item = round.item.as_str();
        let (own_structure, free_structure) = (&round.own_structure, &round.free_structure);
        let len = item.len();
        // Cross pairs: item symbols paired to a context, nearest an item end
        // first.
        let start = self.utr5.len();
        let molecule = format!("{}{}{}", self.utr5, item, self.utr3);
        let mut cross: Vec<(usize, usize, u8)> = pairs(free_structure)
            .into_iter()
            .flat_map(|(i, j, _)| [(i, j), (j, i)])
            .filter(|&(k, q)| {
                (start..start + len).contains(&k) && !(start..start + len).contains(&q)
            })
            .map(|(k, q)| {
                let k = k - start;
                (k.min(len - 1 - k), k, molecule.as_bytes()[q])
            })
            .collect();
        cross.sort();
        if round.own_cost < round.class_cost {
            // The rival is item-internal: the own optimum's pairs the
            // in-class configuration lacks (the rival's stems) and the
            // in-class configuration's pairs the own optimum lacks (a stem the
            // class needs and the item alone does not hold). Nearest a cross
            // pair first — the context binds where the item's own
            // configuration changed — then outermost first.
            let own_pairs = pairs(own_structure);
            let class_pairs = pairs(&round.class_structure);
            let missing = |from: &[(usize, usize, usize)], to: &[(usize, usize, usize)]| {
                from.iter()
                    .copied()
                    .filter(|&(i, j, _)| !to.iter().any(|&(a, b, _)| a == i && b == j))
                    .collect::<Vec<_>>()
            };
            let near = |k: usize| cross.iter().map(|c| c.1.abs_diff(k)).min().unwrap_or(0);
            let mut differ = missing(&own_pairs, &class_pairs);
            differ.extend(missing(&class_pairs, &own_pairs));
            differ.sort_by_key(|&(i, j, depth)| (near(i).min(near(j)), depth, i));
            let ks = differ.into_iter().flat_map(|(i, j, _)| [i, j]);
            let cuts = self.exclusions(item, ks, excluded, Mechanism::OutsideClass);
            if !cuts.is_empty() {
                return cuts;
            }
        }
        let mut cuts: Vec<(Mechanism, Cut)> = Vec::new();
        let mut positions: Vec<usize> = Vec::new();
        for &(_, k, partner) in &cross {
            if positions.len() == self.width || positions.contains(&(k / 3)) {
                continue;
            }
            if self.exclusion(item, k, excluded).is_none() {
                continue;
            }
            positions.push(k / 3);
            cuts.extend(
                self.pairing_choices(item, k, partner, excluded)
                    .into_iter()
                    .map(|cut| (Mechanism::CrossChoice, cut)),
            );
        }
        if !cuts.is_empty() {
            return cuts;
        }
        // Every cross position is fixed: cap the class so that its
        // configuration encloses them — the leading run for those in the
        // item's first half, the trailing run for the rest.
        let half = len / 2;
        let lead = cross
            .iter()
            .filter(|c| c.1 < half)
            .map(|c| c.1 as u32)
            .min();
        if let Some(a) = lead {
            if class.max_lead.is_none_or(|cur| a < cur) {
                return vec![(Mechanism::CrossFixed, Cut::LeadCap(a))];
            }
        }
        let trail = cross
            .iter()
            .filter(|c| c.1 >= half)
            .map(|c| (len - 1 - c.1) as u32)
            .min();
        if let Some(t) = trail {
            if class.max_trail.is_none_or(|cur| t < cur) {
                return vec![(Mechanism::CrossFixed, Cut::TrailCap(t))];
            }
        }
        // Fallback: an item symbol the free fold pairs differently from the
        // item's own optimum, nearest a cross pair.
        let own = partners(own_structure);
        let free = partners(free_structure);
        let anchor: Vec<usize> = cross.iter().map(|c| c.1).collect();
        let mut moved: Vec<(usize, usize)> = (0..len)
            .filter(|&k| {
                let f = free[start + k].map(|q| q as isize - start as isize);
                f != own[k].map(|q| q as isize)
            })
            .map(|k| {
                let d = anchor.iter().map(|&a| a.abs_diff(k)).min().unwrap_or(0);
                (d, k)
            })
            .collect();
        moved.sort();
        self.exclusions(
            item,
            moved.into_iter().map(|(_, k)| k),
            excluded,
            Mechanism::Refold,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use derna_codon_dp::codon_table::aa_to_index;

    const P0DTC2_30: &str = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTN";
    const HBB25_5: &str = "ACAUUUGCUUCUGACACAACUGUGU";
    const HBB25_3: &str = "GCUCGCUUUCUUGCUGUCCAAUUUC";
    const GU_POOR: &str = "ACAACCACAACCAAACCACAACCAACACCAAACCACAACCAAACCACAAC";

    fn hbb25_class() -> ExteriorClass {
        ExteriorClass {
            min_branches: 2,
            max_branches: Some(2),
            max_lead: Some(2),
            ..ExteriorClass::default()
        }
    }

    fn run(protein: &str, flanks: (&str, &str), lambda: f64, class: &ExteriorClass) -> LoopReport {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
        let lp = SeedLoop::new(&params, &usage, aa, flanks, 2).unwrap();
        lp.run(lambda, class, 30).unwrap()
    }

    /// A returned item is tight by an independent re-score: the free optimum
    /// and the constrained configuration (each part's own optimum, side by side)
    /// evaluate to the same cost, and no fold of the molecule goes lower.
    fn assert_recertified(report: &LoopReport, flanks: (&str, &str)) {
        let params = Params::load_default().unwrap();
        assert_eq!(report.outcome, LoopOutcome::Tight);
        let item = report.item().unwrap();
        let cert = report.certificate.as_ref().unwrap();
        let molecule = format!("{}{}{}", flanks.0, item, flanks.1);
        assert_eq!(cert.molecule, molecule);
        let f = rna_mfe::evaluate(&params, molecule.as_bytes(), &cert.free_structure).unwrap();
        let e =
            rna_mfe::evaluate(&params, molecule.as_bytes(), &cert.constrained_structure).unwrap();
        assert_eq!(f, e, "the constraint costs the item something");
        assert_eq!(rna_mfe::fold(&params, molecule.as_bytes()).unwrap().1, f);
        let last = report.rounds.last().unwrap();
        assert!(cert.holds(last.free, last.constrained));
    }

    #[test]
    fn pairs_report_depth_and_partners_are_symmetric() {
        let p = pairs("((..))..(...)");
        assert_eq!(p, vec![(1, 4, 1), (0, 5, 0), (8, 12, 0)]);
        let q = partners("((..))..(...)");
        assert_eq!(q[0], Some(5));
        assert_eq!(q[5], Some(0));
        assert_eq!(q[6], None);
    }

    /// The seed's own optimum leaves the class (it folds cheaper than its
    /// in-class configuration); cutting the rival's stems alone reaches a
    /// tight item.
    #[test]
    fn an_item_internal_rival_is_cut_until_the_seed_is_tight() {
        let flanks = (HBB25_5, HBB25_3);
        let report = run(P0DTC2_30, flanks, 0.3, &hbb25_class());
        let first = &report.rounds[0];
        assert!(first.gap() > 0 && first.own_cost < first.class_cost);
        let mechanisms = report.mechanisms();
        assert!(!mechanisms.is_empty());
        assert!(mechanisms.iter().all(|m| *m == Mechanism::OutsideClass));
        assert_recertified(&report, flanks);
    }

    /// The seed is in the class, a cross pair at a position with other
    /// choices beats it, and excluding the seed's choice there (one position
    /// per round) reaches a tight item.
    #[test]
    fn a_cross_pair_at_a_free_position_is_cut_one_position_at_a_time() {
        let flanks = (GU_POOR, GU_POOR);
        let report = run(&P0DTC2_30[..20], flanks, 0.2, &ExteriorClass::default());
        let first = &report.rounds[0];
        assert!(first.gap() > 0 && first.own_cost == first.class_cost);
        assert!(report
            .mechanisms()
            .iter()
            .all(|m| *m == Mechanism::CrossChoice));
        for r in &report.rounds[..report.rounds.len() - 1] {
            let at: Vec<usize> = r
                .cuts
                .iter()
                .map(|(_, c)| match c {
                    Cut::Exclude { position, .. } => *position,
                    other => panic!("not an exclusion: {other:?}"),
                })
                .collect();
            assert!(!at.is_empty() && at.iter().all(|&p| p == at[0]), "{at:?}");
        }
        assert_recertified(&report, flanks);
    }

    /// The seed is in the class, but its own optimum leaves the start codon
    /// unpaired and a context pairs it; the start codon has one choice, so no
    /// exclusion applies there. The cut caps the leading run so the item's
    /// configuration must pair it, and the next seed is tight.
    #[test]
    fn a_cross_pair_at_the_fixed_start_codon_caps_the_leading_run() {
        let flanks = (GU_POOR, GU_POOR);
        let report = run(P0DTC2_30, flanks, 1.0, &ExteriorClass::default());
        let first = &report.rounds[0];
        assert!(first.gap() > 0 && first.own_cost == first.class_cost);
        assert!(
            first.own_structure.starts_with(".."),
            "{}",
            first.own_structure
        );
        assert!(matches!(
            first.cuts[..],
            [(Mechanism::CrossFixed, Cut::LeadCap(a))] if a < 3
        ));
        assert_eq!(report.class.max_lead, Some(1));
        assert_recertified(&report, flanks);
    }

    /// Pinned end windows hold in every round's seed, and no cut excludes a
    /// choice inside them.
    #[test]
    fn pinned_ends_hold_through_every_round() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let aa: Vec<usize> = P0DTC2_30.chars().map(|c| aa_to_index(c).unwrap()).collect();
        let lp = SeedLoop::new(&params, &usage, aa, (HBB25_5, HBB25_3), 2).unwrap();
        let free = lp.run(0.3, &hbb25_class(), 1).unwrap();
        let pin = free.rounds[0].item.clone();
        let (w5, w3) = (4, 3);
        let report = lp
            .run_from(0.3, &hbb25_class(), lp.pin_ends(&pin, w5, w3), 30)
            .unwrap();
        let n = pin.len();
        for r in &report.rounds {
            assert_eq!(r.item[..3 * w5], pin[..3 * w5]);
            assert_eq!(r.item[n - 3 * w3..], pin[n - 3 * w3..]);
            for (_, cut) in &r.cuts {
                if let Cut::Exclude { position, .. } = cut {
                    assert!(*position >= w5 && *position + w3 < n / 3, "{cut:?}");
                }
            }
        }
        if report.outcome == LoopOutcome::Tight {
            assert_recertified(&report, (HBB25_5, HBB25_3));
        }
    }

    /// A one-position item whose only choice is paired by its contexts: the
    /// cap that would make the item pair it leaves the class empty, and the
    /// loop ends empty-handed rather than returning a non-tight item.
    #[test]
    fn a_class_capped_to_empty_ends_the_loop_exhausted() {
        let report = run("M", ("GGG", "AAAACAUCCC"), 0.5, &ExteriorClass::default());
        assert_eq!(report.outcome, LoopOutcome::Exhausted);
        assert!(report.item().is_none());
        assert_eq!(
            report.mechanisms(),
            vec![Mechanism::CrossFixed],
            "{:?}",
            report.rounds
        );
        assert!(report.rounds[0].gap() > 0);
    }
}
