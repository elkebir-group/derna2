//! Local search over the codon lattice toward modular designs.
//!
//! A design's constrained energy separates under the three-class constraint at
//! `dangles = 0`: `E = MFE(5′) + MFE(CDS) + MFE(3′)`. Its free energy `F` is
//! the MFE of the whole molecule, and the design is modular when
//! `F = E` (`F ≤ E` always). Both are exact Zuker folds ([`rna_mfe::fold`]).
//!
//! From a start design, each round re-chooses the codon at every position the
//! free optimum pairs across a UTR boundary, and at its two neighbours: every
//! synonymous alternative there is folded, and the one with the smallest gap
//! `E − F` is taken (ties broken by [`Aim`]), as long as the gap does not
//! widen. A position once changed is not changed again (tabu), and a bounded
//! number of moves that keep the gap level are allowed when no move narrows it.
//! When the free optimum pairs no CDS position across a boundary — a clamp of
//! the two UTRs alone — the candidates are the codons at the ends of the CDS's
//! top-level pairs and its first and last two codons, which set the cost of
//! the loop the clamp closes around the CDS.
//!
//! A design is reported modular only when its two folds agree
//! exactly ([`Evaluated::is_tight`]); a narrowed gap is never enough.

use std::thread;

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use rna_mfe::Params;

use crate::CodonUsage;

/// A design's own and whole-molecule fold tables.
type Tables = (rna_mfe::dp::DpTables, rna_mfe::dp::DpTables);

/// A climb's record of single changes: (position, codon) -> the energy change
/// it made when last folded (`None`: not tight).
type Memo = std::collections::HashMap<(usize, String), Option<i32>>;

/// The per-position codon menus of a protein, each codon with its adaptation
/// weight `ln(fraction / best fraction for its amino acid)` (0 for the most
/// frequent synonym, negative otherwise).
#[derive(Debug, Clone)]
pub struct Menus {
    positions: Vec<Vec<(String, f64)>>,
}

impl Menus {
    /// The menus of `protein` (one-letter codes) under `usage`.
    pub fn new(protein: &str, usage: &CodonUsage) -> Result<Self, RepairError> {
        let positions = protein
            .chars()
            .map(|aa| {
                let idx = aa_to_index(aa).map_err(|_| RepairError::Residue(aa))?;
                let codons = AMINO_ACIDS[idx].codons;
                let fracs: Vec<f64> = (0..codons.len())
                    .map(|ci| usage.fraction(idx, ci))
                    .collect();
                let best = fracs.iter().cloned().fold(f64::MIN, f64::max);
                Ok(codons
                    .iter()
                    .zip(&fracs)
                    .map(|(c, &f)| (c.name.to_string(), (f / best).ln()))
                    .collect())
            })
            .collect::<Result<_, RepairError>>()?;
        Ok(Self { positions })
    }

    /// The number of positions.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the protein is empty.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Restrict the first `w` positions to the codons `cds` spells there, so a
    /// search over these menus never changes them. Weights are kept, so
    /// [`Self::weight`] still reads the whole design.
    pub fn freeze_head(&mut self, cds: &str, w: usize) -> Result<(), RepairError> {
        self.freeze_ends(cds, w, 0)
    }

    /// [`Self::freeze_head`] on both ends: the first `w5` and the last `w3`
    /// positions are held to what `cds` spells there. The two windows may
    /// overlap, which freezes the whole design.
    pub fn freeze_ends(&mut self, cds: &str, w5: usize, w3: usize) -> Result<(), RepairError> {
        self.check(cds)?;
        let n = self.len();
        for p in (0..n).filter(|p| *p < w5 || p + w3 >= n) {
            let codon = &cds[3 * p..3 * p + 3];
            self.positions[p].retain(|(c, _)| c == codon);
        }
        Ok(())
    }

    /// The codon menu at position `p`: each codon with its adaptation weight.
    pub fn options(&self, p: usize) -> &[(String, f64)] {
        &self.positions[p]
    }

    /// The summed adaptation weight of a CDS (`cai_log` in front files).
    pub fn weight(&self, cds: &str) -> Result<f64, RepairError> {
        self.check(cds)?;
        Ok((0..self.len())
            .map(|p| {
                self.option(p, &cds[3 * p..3 * p + 3])
                    .map_or(f64::NAN, |(_, w)| w)
            })
            .sum())
    }

    fn option(&self, p: usize, codon: &str) -> Option<(usize, f64)> {
        self.positions[p]
            .iter()
            .position(|(c, _)| c == codon)
            .map(|i| (i, self.positions[p][i].1))
    }

    /// Fail unless `cds` spells one menu codon per position.
    fn check(&self, cds: &str) -> Result<(), RepairError> {
        if cds.len() != 3 * self.len() {
            return Err(RepairError::Length(cds.len(), 3 * self.len()));
        }
        for p in 0..self.len() {
            if self.option(p, &cds[3 * p..3 * p + 3]).is_none() {
                return Err(RepairError::Codon(p, cds[3 * p..3 * p + 3].to_string()));
            }
        }
        Ok(())
    }

    /// Every design one synonymous change away from `cds`, in position order,
    /// each position's menu in its own order.
    pub fn neighbours(&self, cds: &str) -> Vec<String> {
        self.climb_neighbours(cds, Toward::Slope(0.0))
    }

    /// Every design one synonymous change away from `cds` that a climb toward
    /// `toward` considers: every change for a slope, only the changes that
    /// raise the weight for weight. In position order, each position's menu in
    /// its own order.
    fn climb_neighbours(&self, cds: &str, toward: Toward) -> Vec<String> {
        (0..self.len())
            .flat_map(|p| {
                let cur = &cds[3 * p..3 * p + 3];
                let here = self.option(p, cur).map_or(f64::NAN, |(_, w)| w);
                self.positions[p]
                    .iter()
                    .filter(move |(c, w)| match toward {
                        Toward::Slope(_) => c != cur,
                        Toward::Weight => *w > here + 1e-12,
                    })
                    .map(move |(c, _)| format!("{}{}{}", &cds[..3 * p], c, &cds[3 * p + 3..]))
            })
            .collect()
    }

    /// Every design one synonymous change at `p` away from `cds`.
    fn alternatives(&self, cds: &str, p: usize) -> Vec<String> {
        let cur = &cds[3 * p..3 * p + 3];
        self.positions[p]
            .iter()
            .filter(|(c, _)| c != cur)
            .map(|(c, _)| format!("{}{}{}", &cds[..3 * p], c, &cds[3 * p + 3..]))
            .collect()
    }
}

/// One design's two energies (in dekacalories) and its free optimum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evaluated {
    /// The CDS.
    pub cds: String,
    /// Constrained energy `E = MFE(5′) + MFE(CDS) + MFE(3′)`.
    pub constrained: i32,
    /// Free energy `F`, the whole molecule's MFE.
    pub free: i32,
    /// The free optimum of the whole molecule, in dot-bracket.
    pub free_structure: String,
    /// With a class ([`Searcher::with_class`]): the CDS's least in-class cost
    /// less its own MFE, `>= 0`; zero when its own optimum is in the class
    /// (necessary for tightness when the class is). Zero without a class, and
    /// filled in by a search only for the designs it ranks.
    pub class_gap: i32,
}

impl Evaluated {
    /// `E − F ≥ 0`, in dekacalories.
    pub fn gap(&self) -> i32 {
        self.constrained - self.free
    }

    /// Whether the design is modular: its two folds agree exactly.
    pub fn is_tight(&self) -> bool {
        self.free == self.constrained
    }
}

/// How ties between equally narrow moves are broken.
#[derive(Debug, Clone, Copy)]
pub enum Aim {
    /// Higher adaptation weight first, then lower constrained energy.
    Weight,
    /// Lower `E − μ·A`, with `μ` in dekacalories per unit weight.
    Slope(f64),
}

/// What a climb steps toward.
#[derive(Debug, Clone, Copy)]
pub enum Toward {
    /// Higher adaptation weight: only weight-raising changes are tried, and the
    /// step goes to the tight one of highest weight, then lowest constrained energy.
    Weight,
    /// Lower `E − μ·A`, with `μ` in dekacalories per unit weight: every change
    /// is tried, and the step goes to the tight one of least `E − μ·A`, if it is
    /// below the current design's.
    Slope(f64),
}

/// A tight design a climb met.
#[derive(Debug, Clone)]
pub struct Met {
    /// The CDS.
    pub cds: String,
    /// Its adaptation weight.
    pub weight: f64,
    /// Its constrained energy, equal to its free energy.
    pub constrained: i32,
    /// The step whose neighbourhood it was in (0 for the start's).
    pub step: usize,
}

/// The outcome of one climb.
#[derive(Debug, Clone)]
pub struct Climbed {
    /// The start design, evaluated.
    pub start: Evaluated,
    /// Every tight neighbour evaluated, in evaluation order (repeats possible
    /// across steps); empty when the start is not tight.
    pub met: Vec<Met>,
    /// The designs stepped to, in order; the climb ended on the last (on the
    /// start when empty).
    pub path: Vec<String>,
    /// The design the climb ended on.
    pub end: String,
    /// Steps taken.
    pub steps: usize,
}

/// The outcome of one search.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// The start design, evaluated.
    pub start: Evaluated,
    /// The design the search ended on.
    pub end: Evaluated,
    /// The end design's adaptation weight.
    pub weight: f64,
    /// Rounds taken.
    pub rounds: usize,
}

/// Errors of a search.
#[derive(Debug, thiserror::Error)]
pub enum RepairError {
    /// A residue outside the twenty amino acids.
    #[error("residue {0:?} has no codon menu")]
    Residue(char),
    /// A CDS of the wrong length.
    #[error("a CDS of {0} nucleotides for a protein needing {1}")]
    Length(usize, usize),
    /// A codon not on its position's menu.
    #[error("position {0}: codon {1} does not code for the residue there")]
    Codon(usize, String),
    /// A fold failed.
    #[error("fold failed: {0}")]
    Fold(#[from] rna_mfe::FoldError),
    /// A climb's per-step sink failed.
    #[error("climb sink: {0}")]
    Sink(#[from] std::io::Error),
}

/// A local search over one protein's codon lattice between fixed UTRs.
pub struct Searcher<'a> {
    params: &'a Params,
    utr5: &'a str,
    utr3: &'a str,
    flanks: i32,
    menus: Menus,
    threads: usize,
    aim: Aim,
    reach: usize,
    partners: bool,
    pairs: usize,
    class: Option<crate::class_bound::ExteriorClass>,
    batch: usize,
    max_batches: usize,
    descend: bool,
    recall: bool,
    stem_pairs: bool,
    climb_ends: usize,
    estimate: bool,
}

impl<'a> Searcher<'a> {
    /// A searcher folding on `threads` threads (at least one).
    pub fn new(
        params: &'a Params,
        (utr5, utr3): (&'a str, &'a str),
        menus: Menus,
        threads: usize,
        aim: Aim,
    ) -> Result<Self, RepairError> {
        let flanks = rna_mfe::fold(params, utr5.as_bytes())?.1 .0
            + rna_mfe::fold(params, utr3.as_bytes())?.1 .0;
        Ok(Self {
            params,
            utr5,
            utr3,
            flanks,
            menus,
            threads: threads.max(1),
            aim,
            reach: 1,
            partners: false,
            pairs: 0,
            class: None,
            batch: usize::MAX,
            max_batches: usize::MAX,
            descend: false,
            recall: false,
            stem_pairs: false,
            climb_ends: 0,
            estimate: false,
        })
    }

    /// Re-choose the codons within `reach` of each candidate site rather than
    /// within one (the default): a site at the start codon, which has no
    /// synonym, otherwise leaves a round only its right neighbour.
    pub fn with_reach(mut self, reach: usize) -> Self {
        self.reach = reach.max(1);
        self
    }

    /// Also re-choose the codons at the in-CDS partners of the symbols within
    /// three of each crossing site: a crossing pair often takes a symbol the
    /// CDS leaves unpaired next to one of its own helices, and extending that
    /// helix by a pair needs the codon across from it, not the one at the site.
    pub fn with_partners(mut self, partners: bool) -> Self {
        self.partners = partners;
        self
    }

    /// When no single change narrows the gap, also try every pair of changes
    /// among the first `k` codons, and among the last `k` (0, the default,
    /// turns this off). A pair at one end recomputes only that end's cells
    /// ([`Self::evaluate_near`]), so the search is cheap where the context
    /// rivals sit.
    pub fn with_pairs(mut self, k: usize) -> Self {
        self.pairs = k;
        self
    }

    /// Evaluate a climb step's neighbours `k` at a time, most promising first,
    /// and stop at the first batch holding a qualifying tight one (see
    /// [`Self::climb`]); 0 or `usize::MAX` (the default) evaluates them all.
    pub fn with_batch(mut self, k: usize) -> Self {
        self.batch = if k == 0 { usize::MAX } else { k };
        self
    }

    /// Evaluate at most `m` batches per climb step (0 or `usize::MAX`, the
    /// default: every batch): a step whose first `m` batches hold no
    /// qualifying tight neighbour ends the climb, leaving the rest of the
    /// neighbourhood unexamined, so the end need not be a local optimum.
    pub fn with_max_batches(mut self, m: usize) -> Self {
        self.max_batches = if m == 0 { usize::MAX } else { m };
        self
    }

    /// Order a batched step's neighbours by weight loss rather than
    /// gain: for a climb started above the slope's optimum, where the
    /// improving changes give up weight for a larger fall in energy.
    pub fn with_descent(mut self, descend: bool) -> Self {
        self.descend = descend;
        self
    }

    /// In a batched slope climb, put first the changes an earlier step found
    /// improving by the slope (by the energy change they made there, best
    /// first), then the untried ones in the usual order, then those found not
    /// improving or not tight: a change away from the last move usually
    /// shifts the energy by about what it did one step before. Only the order
    /// changes; every design moved to is still folded and tight.
    pub fn with_recall(mut self, recall: bool) -> Self {
        self.recall = recall;
        self
    }

    /// Add two-change moves to a climb step's neighbourhood: with `stems`,
    /// every pair of changes at two codons that the current design's own
    /// optimum pairs with each other (a pair of the optimum whose two symbols
    /// lie in different codons), and with `ends > 0`, every pair of changes
    /// inside the first or the last `ends` codons. A single change at one side
    /// of a stem pair breaks the pair unless the other side changes with it,
    /// so a design that no single change improves may still have a
    /// two-change improvement. A weight climb keeps only the pairs that raise
    /// the weight. The neighbourhood is the single changes, then the stem
    /// pairs (by codon pair, each menu in its own order), then the end pairs
    /// (sorted), without repeats. A batched slope climb with pairs orders it
    /// by the estimate ([`Self::with_estimate`]).
    pub fn with_climb_pairs(mut self, stems: bool, ends: usize) -> Self {
        self.stem_pairs = stems;
        self.climb_ends = ends;
        self
    }

    /// In a batched slope climb, order the neighbours by an upper bound on
    /// the change of `E − μ·A` before folding any: the current design's own
    /// optimum, rescored on the neighbour ([`rna_mfe::evaluate`]), bounds the
    /// neighbour's own minimum from above, so a neighbour whose bound falls
    /// is an improvement if it is tight. A neighbour that breaks pairs of that
    /// optimum is scored on it with those pairs opened — an estimate rather
    /// than a bound, since opening a pair can merge two loops past the span
    /// cap. Ties keep the usual order. Only the order changes; every design
    /// moved to is still folded and tight.
    pub fn with_estimate(mut self, estimate: bool) -> Self {
        self.estimate = estimate;
        self
    }

    /// Whether a climb step adds two-change moves.
    fn pairs_on(&self) -> bool {
        self.stem_pairs || self.climb_ends > 0
    }

    /// Whether a batched slope step orders its neighbours by the estimate.
    fn ordered(&self) -> bool {
        self.estimate || self.pairs_on()
    }

    /// Every design two changes away from `cds` at two codons that `own`,
    /// a configuration of `cds` alone, pairs with each other.
    fn stem_pairs(&self, cds: &str, own: &str) -> Vec<String> {
        let codons: std::collections::BTreeSet<(usize, usize)> = matching(own)
            .into_iter()
            .map(|(i, j)| (i / 3, j / 3))
            .filter(|(p, q)| p != q)
            .collect();
        let mut out = Vec::new();
        for (p, q) in codons {
            for one in self.menus.alternatives(cds, p) {
                out.extend(self.menus.alternatives(&one, q));
            }
        }
        out
    }

    /// A climb step's neighbourhood from `cds`, whose own optimum is `own`
    /// (see [`Self::with_climb_pairs`]).
    fn climb_moves(
        &self,
        cds: &str,
        own: &str,
        toward: Toward,
    ) -> Result<Vec<String>, RepairError> {
        let mut moves = self.menus.climb_neighbours(cds, toward);
        if !self.pairs_on() {
            return Ok(moves);
        }
        let mut doubles = Vec::new();
        if self.stem_pairs {
            doubles.extend(self.stem_pairs(cds, own));
        }
        if self.climb_ends > 0 {
            doubles.extend(self.end_pairs(cds, self.climb_ends));
        }
        if let Toward::Weight = toward {
            let here = self.menus.weight(cds)?;
            let mut raising = Vec::with_capacity(doubles.len());
            for c in doubles {
                if self.menus.weight(&c)? > here + 1e-12 {
                    raising.push(c);
                }
            }
            doubles = raising;
        }
        let mut seen: std::collections::HashSet<String> = moves.iter().cloned().collect();
        moves.extend(doubles.into_iter().filter(|c| seen.insert(c.clone())));
        Ok(moves)
    }

    /// The energy-change estimate of [`Self::with_estimate`] for each of
    /// `cds` from `base`, whose own optimum is `own` with energy `own_e`
    /// (`None` only if the rescoring fails).
    fn estimates(&self, base: &str, own: &str, own_e: i32, cds: &[String]) -> Vec<Option<i32>> {
        let partner = partners(own);
        let partner = &partner;
        let rescore = move |c: &String| -> Option<i32> {
            let (b, n) = (base.as_bytes(), c.as_bytes());
            let broken: Vec<usize> = (0..n.len())
                .filter(|&k| b[k] != n[k])
                .filter(|&k| partner[k] != usize::MAX && !can_pair(n[k], n[partner[k]]))
                .collect();
            let score = |structure: &str| {
                rna_mfe::evaluate(self.params, n, structure)
                    .ok()
                    .map(|e| e.0 - own_e)
            };
            if broken.is_empty() {
                return score(own);
            }
            let mut opened = own.as_bytes().to_vec();
            for k in broken {
                (opened[k], opened[partner[k]]) = (b'.', b'.');
            }
            score(std::str::from_utf8(&opened).expect("dot-bracket is ASCII"))
        };
        let chunk = cds.len().div_ceil(self.threads).max(1);
        thread::scope(|s| {
            let parts: Vec<_> = cds
                .chunks(chunk)
                .map(|part| s.spawn(move || part.iter().map(rescore).collect::<Vec<_>>()))
                .collect();
            parts
                .into_iter()
                .flat_map(|h| h.join().expect("an estimate thread panicked"))
                .collect()
        })
    }

    /// The order a batched step folds its neighbours `ups` in, from `cur`
    /// whose own optimum is `own` (see [`Self::with_batch`],
    /// [`Self::with_descent`], [`Self::with_recall`], [`Self::with_estimate`]).
    fn order_moves(
        &self,
        cur: &Evaluated,
        own: &str,
        ups: Vec<String>,
        toward: Toward,
        memo: &Memo,
    ) -> Result<Vec<String>, RepairError> {
        let here_w = self.menus.weight(&cur.cds)?;
        let mut keyed = ups
            .into_iter()
            .map(|c| Ok((self.menus.weight(&c)? - here_w, c)))
            .collect::<Result<Vec<_>, RepairError>>()?;
        if self.descend {
            keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
        } else {
            keyed.sort_by(|a, b| b.0.total_cmp(&a.0));
        }
        let Toward::Slope(mu) = toward else {
            return Ok(keyed.into_iter().map(|(_, c)| c).collect());
        };
        if self.ordered() {
            let cds: Vec<String> = keyed.into_iter().map(|(_, c)| c).collect();
            return Ok(self.by_estimate(cur, own, cds, mu));
        }
        if self.recall {
            // Stable: within a class the order above stands.
            keyed.sort_by(|a, b| {
                let (ra, rb) = (
                    recall_rank(memo, &cur.cds, a, mu),
                    recall_rank(memo, &cur.cds, b, mu),
                );
                ra.0.cmp(&rb.0).then(ra.1.total_cmp(&rb.1))
            });
        }
        Ok(keyed.into_iter().map(|(_, c)| c).collect())
    }

    /// `cds` sorted by the estimated change of `E − μ·A` from `cur`
    /// ([`Self::with_estimate`]); stable, the unbounded last.
    fn by_estimate(&self, cur: &Evaluated, own: &str, cds: Vec<String>, mu: f64) -> Vec<String> {
        let own_e = cur.constrained - self.flanks;
        let bounds = self.estimates(&cur.cds, own, own_e, &cds);
        let mut scored: Vec<(f64, String)> = bounds
            .into_iter()
            .zip(cds)
            .map(|(b, c)| {
                let dw = self.menus.weight(&c).unwrap_or(f64::NAN)
                    - self.menus.weight(&cur.cds).unwrap_or(f64::NAN);
                (b.map_or(f64::INFINITY, |de| f64::from(de) - mu * dw), c)
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        scored.into_iter().map(|(_, c)| c).collect()
    }

    /// The own optimum of `cds` from its filled own `tables`, in dot-bracket.
    fn own_structure(&self, cds: &str, tables: &rna_mfe::dp::DpTables) -> String {
        let nt: Vec<rna_mfe::Nucleotide> = cds
            .bytes()
            .map(|b| rna_mfe::Nucleotide::from_byte(b).expect("a checked CDS"))
            .collect();
        rna_mfe::dp::traceback(self.params, &nt, tables)
    }

    /// Every design two changes away from `cds` inside one end window of
    /// `k` codons.
    fn end_pairs(&self, cds: &str, k: usize) -> Vec<String> {
        let n = self.menus.len();
        let mut out = Vec::new();
        for window in [0..k.min(n), n.saturating_sub(k)..n] {
            let window: Vec<usize> = window.collect();
            for (a, &p) in window.iter().enumerate() {
                for &q in &window[a + 1..] {
                    for one in self.menus.alternatives(cds, p) {
                        out.extend(self.menus.alternatives(&one, q));
                    }
                }
            }
        }
        out.sort();
        out.dedup();
        out
    }

    /// Rank moves by the gap, then by the class gap ([`Evaluated::class_gap`]):
    /// a move that leaves the gap but brings the CDS's own optimum toward the
    /// class counts as progress, not a sideways move. Every evaluation then
    /// also scans the CDS alone on a single-path graph.
    pub fn with_class(mut self, class: crate::class_bound::ExteriorClass) -> Self {
        self.class = Some(class);
        self
    }

    /// The CDS's least in-class cost less its own minimum, by the scan on its
    /// single-path graph; 0 without a class.
    fn class_gap(&self, cds: &str) -> i32 {
        let Some(class) = &self.class else {
            return 0;
        };
        let seq: Vec<rna_mfe::Nucleotide> = cds
            .bytes()
            .map(|b| rna_mfe::Nucleotide::from_byte(b).expect("a checked CDS"))
            .collect();
        let dfa = crate::dfa::build_fixed_mrna(&seq);
        let tables = crate::scan::scan(self.params, &dfa, 1.0, crate::ScanConfig::default());
        let own = tables.c[dfa.ix(dfa.end())].map_or(0, |s| s.mfe.0);
        crate::class_bound::class_optimum(self.params, &dfa, &tables, class)
            .best
            .map_or(i32::MAX / 4, |s| s.mfe.0 - own)
    }

    /// The menus searched over.
    pub fn menus(&self) -> &Menus {
        &self.menus
    }

    /// Both energies of each CDS, folded across the searcher's threads.
    pub fn evaluate(&self, cds: &[String]) -> Result<Vec<Evaluated>, RepairError> {
        let chunk = cds.len().div_ceil(self.threads).max(1);
        thread::scope(|s| {
            let parts: Vec<_> = cds
                .chunks(chunk)
                .map(|part| {
                    s.spawn(move || {
                        part.iter()
                            .map(|c| self.evaluate_one(c))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            parts
                .into_iter()
                .flat_map(|h| h.join().expect("a fold thread panicked"))
                .collect()
        })
    }

    /// [`Self::evaluate`] for designs that each differ from `base` in one
    /// stretch: `base`'s two folds are filled once and every design is folded
    /// from them ([`rna_mfe::fold_from`]), recomputing only the cells that
    /// span its change. A change near either end of the CDS recomputes a
    /// small share of each table; the energies are those of a fold from
    /// scratch.
    pub fn evaluate_near(&self, base: &str, cds: &[String]) -> Result<Vec<Evaluated>, RepairError> {
        let (_, own, whole) = self.base_tables(base)?;
        self.evaluate_with(base, (&own, &whole), cds)
    }

    /// `base`'s own and whole-molecule folds, filled on two threads, with the
    /// base evaluated from them.
    fn base_tables(
        &self,
        base: &str,
    ) -> Result<(Evaluated, rna_mfe::dp::DpTables, rna_mfe::dp::DpTables), RepairError> {
        self.menus.check(base)?;
        let molecule = format!("{}{}{}", self.utr5, base, self.utr3);
        let (own, whole) = thread::scope(|s| {
            let own = s.spawn(|| rna_mfe::fold_tables(self.params, base.as_bytes()));
            let whole = rna_mfe::fold_tables(self.params, molecule.as_bytes());
            (own.join().expect("a fold thread panicked"), whole)
        });
        let ((_, own_e, own), (free_structure, free, whole)) = (own?, whole?);
        let e = Evaluated {
            cds: base.to_string(),
            constrained: self.flanks + own_e.0,
            free: free.0,
            free_structure,
            class_gap: 0,
        };
        Ok((e, own, whole))
    }

    /// The designs `cds`, each differing from `base` in one stretch, folded
    /// across the threads from `base`'s filled `tables`.
    fn evaluate_with(
        &self,
        base: &str,
        tables: (&rna_mfe::dp::DpTables, &rna_mfe::dp::DpTables),
        cds: &[String],
    ) -> Result<Vec<Evaluated>, RepairError> {
        let chunk = cds.len().div_ceil(self.threads).max(1);
        thread::scope(|s| {
            let parts: Vec<_> = cds
                .chunks(chunk)
                .map(|part| {
                    s.spawn(move || {
                        part.iter()
                            .map(|c| self.evaluate_from(base, tables, c))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            parts
                .into_iter()
                .flat_map(|h| h.join().expect("a fold thread panicked"))
                .collect()
        })
    }

    fn evaluate_from(
        &self,
        base: &str,
        (own, whole): (&rna_mfe::dp::DpTables, &rna_mfe::dp::DpTables),
        cds: &str,
    ) -> Result<Evaluated, RepairError> {
        self.menus.check(cds)?;
        let (b, c) = (base.as_bytes(), cds.as_bytes());
        let Some(lo) = (0..c.len()).find(|&k| b[k] != c[k]) else {
            return self.evaluate_one(cds);
        };
        let hi = (0..c.len())
            .rev()
            .find(|&k| b[k] != c[k])
            .expect("a difference");
        let own = rna_mfe::fold_from(self.params, own, c, lo, hi)?.1 .0;
        let molecule = format!("{}{}{}", self.utr5, cds, self.utr3);
        let l5 = self.utr5.len();
        let (free_structure, free) =
            rna_mfe::fold_from(self.params, whole, molecule.as_bytes(), l5 + lo, l5 + hi)?;
        Ok(Evaluated {
            cds: cds.to_string(),
            constrained: self.flanks + own,
            free: free.0,
            free_structure,
            class_gap: 0,
        })
    }

    /// [`Self::evaluate_with`] that also returns the own and whole-molecule
    /// tables of each tight design (the base for folding its own neighbours),
    /// dropping the others' as they are folded.
    fn evaluate_keeping(
        &self,
        base: &str,
        tables: (&rna_mfe::dp::DpTables, &rna_mfe::dp::DpTables),
        cds: &[String],
    ) -> Result<Vec<(Evaluated, Option<Tables>)>, RepairError> {
        let chunk = cds.len().div_ceil(self.threads).max(1);
        thread::scope(|s| {
            let parts: Vec<_> = cds
                .chunks(chunk)
                .map(|part| {
                    s.spawn(move || {
                        part.iter()
                            .map(|c| self.evaluate_kept(base, tables, c))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            parts
                .into_iter()
                .flat_map(|h| h.join().expect("a fold thread panicked"))
                .collect()
        })
    }

    /// [`Self::evaluate_from`], keeping the filled tables when the design is tight.
    fn evaluate_kept(
        &self,
        base: &str,
        (own, whole): (&rna_mfe::dp::DpTables, &rna_mfe::dp::DpTables),
        cds: &str,
    ) -> Result<(Evaluated, Option<Tables>), RepairError> {
        self.menus.check(cds)?;
        let (b, c) = (base.as_bytes(), cds.as_bytes());
        let Some(lo) = (0..c.len()).find(|&k| b[k] != c[k]) else {
            return Ok((self.evaluate_one(cds)?, None));
        };
        let hi = (0..c.len())
            .rev()
            .find(|&k| b[k] != c[k])
            .expect("a difference");
        let (_, own_e, own) = self.fill_from(own, c, (lo, hi), false)?;
        let molecule = format!("{}{}{}", self.utr5, cds, self.utr3);
        let l5 = self.utr5.len();
        let (free_structure, free, whole) =
            self.fill_from(whole, molecule.as_bytes(), (l5 + lo, l5 + hi), true)?;
        let e = Evaluated {
            cds: cds.to_string(),
            constrained: self.flanks + own_e,
            free,
            free_structure,
            class_gap: 0,
        };
        let kept = e.is_tight().then_some((own, whole));
        Ok((e, kept))
    }

    /// The tables of `seq`, refilled from `base` over the change `lo..=hi`
    /// ([`rna_mfe::dp::fill_dp_tables_from`]), with the minimum and, when
    /// `trace`, the optimum's dot-bracket (empty otherwise).
    fn fill_from(
        &self,
        base: &rna_mfe::dp::DpTables,
        seq: &[u8],
        (lo, hi): (usize, usize),
        trace: bool,
    ) -> Result<(String, i32, rna_mfe::dp::DpTables), RepairError> {
        let nt = seq
            .iter()
            .map(|&b| {
                rna_mfe::Nucleotide::from_byte(b).ok_or(rna_mfe::FoldError::InvalidNucleotide(b))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tables = rna_mfe::dp::fill_dp_tables_from(self.params, &nt, base, lo, hi);
        let energy = tables.f[nt.len()].map_or(0, |e| e.0);
        let structure = if trace {
            rna_mfe::dp::traceback(self.params, &nt, &tables)
        } else {
            String::new()
        };
        Ok((structure, energy, tables))
    }

    fn evaluate_one(&self, cds: &str) -> Result<Evaluated, RepairError> {
        self.menus.check(cds)?;
        let own = rna_mfe::fold(self.params, cds.as_bytes())?.1 .0;
        let molecule = format!("{}{}{}", self.utr5, cds, self.utr3);
        let (free_structure, free) = rna_mfe::fold(self.params, molecule.as_bytes())?;
        Ok(Evaluated {
            cds: cds.to_string(),
            constrained: self.flanks + own,
            free: free.0,
            free_structure,
            class_gap: 0,
        })
    }

    /// Fill in the class gap of every design in `scored` whose gap is the
    /// least there, the only ones it can rank (scans run across the threads).
    fn rank_ties(&self, scored: &mut [Evaluated]) {
        if self.class.is_none() {
            return;
        }
        let Some(least) = scored.iter().map(Evaluated::gap).min() else {
            return;
        };
        let mut tied: Vec<&mut Evaluated> =
            scored.iter_mut().filter(|e| e.gap() == least).collect();
        let chunk = tied.len().div_ceil(self.threads).max(1);
        thread::scope(|s| {
            for part in tied.chunks_mut(chunk) {
                s.spawn(move || {
                    for e in part.iter_mut() {
                        e.class_gap = self.class_gap(&e.cds);
                    }
                });
            }
        });
    }

    /// The codon positions a round re-chooses (see the module docs).
    fn candidates(&self, e: &Evaluated) -> Vec<usize> {
        let (l5, c) = (self.utr5.len(), e.cds.len());
        let region = |p: usize| -> u8 {
            if p < l5 {
                0
            } else if p < l5 + c {
                1
            } else {
                2
            }
        };
        let mut out = std::collections::BTreeSet::new();
        let reach = self.reach;
        let add = |out: &mut std::collections::BTreeSet<usize>, codon: usize| {
            out.extend(codon.saturating_sub(reach)..=codon + reach);
        };
        let mut stack = Vec::new();
        let mut partner = vec![usize::MAX; e.free_structure.len()];
        let mut sites = Vec::new();
        for (j, ch) in e.free_structure.bytes().enumerate() {
            match ch {
                b'(' => stack.push(j),
                b')' => {
                    let i = stack.pop().expect("balanced");
                    (partner[i], partner[j]) = (j, i);
                    if region(i) != region(j) {
                        for p in [i, j] {
                            if region(p) == 1 {
                                add(&mut out, (p - l5) / 3);
                                sites.push(p);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        if self.partners {
            for &p in &sites {
                let lo = p.saturating_sub(3);
                for (q, &r) in partner
                    .iter()
                    .enumerate()
                    .take((p + 3).min(partner.len() - 1) + 1)
                    .skip(lo)
                {
                    if q != p && region(q) == 1 && r != usize::MAX && region(r) == 1 {
                        add(&mut out, (r - l5) / 3);
                    }
                }
            }
        }
        if out.is_empty() {
            let mut depth = 0usize;
            for (j, ch) in e.free_structure.bytes().skip(l5).take(c).enumerate() {
                match ch {
                    b'(' => {
                        if depth == 0 {
                            add(&mut out, j / 3);
                        }
                        depth += 1;
                    }
                    b')' => {
                        depth -= 1;
                        if depth == 0 {
                            add(&mut out, j / 3);
                        }
                    }
                    _ => {}
                }
            }
            let n = c / 3;
            for q in [0, 1, n.saturating_sub(2), n.saturating_sub(1)] {
                out.insert(q);
            }
        }
        out.into_iter().filter(|&q| q < c / 3).collect()
    }

    /// The tie-break value of a move: smaller is better.
    fn value(&self, e: &Evaluated) -> (f64, f64) {
        let w = self.menus.weight(&e.cds).unwrap_or(f64::NAN);
        match self.aim {
            Aim::Weight => (-w, f64::from(e.constrained)),
            Aim::Slope(mu) => (f64::from(e.constrained) - mu * w, 0.0),
        }
    }

    /// Search from `start` for at most `rounds` rounds, allowing `sideways`
    /// gap-preserving moves when stuck.
    pub fn run(&self, start: &str, rounds: usize, sideways: usize) -> Result<Outcome, RepairError> {
        let first = self.evaluate(&[start.to_string()])?.remove(0);
        self.run_evaluated(first, rounds, sideways)
    }

    /// [`Self::run`] from a start already evaluated (e.g. one of many folded
    /// together by [`Self::evaluate`], which keeps every thread busy).
    pub fn run_evaluated(
        &self,
        first: Evaluated,
        rounds: usize,
        sideways: usize,
    ) -> Result<Outcome, RepairError> {
        let mut cur = first.clone();
        cur.class_gap = self.class_gap(&cur.cds);
        let (mut taken, mut level, mut tabu) = (0, 0, std::collections::BTreeSet::new());
        while !cur.is_tight() && taken < rounds {
            taken += 1;
            let variants: Vec<String> = self
                .candidates(&cur)
                .into_iter()
                .filter(|p| !tabu.contains(p))
                .flat_map(|p| self.menus.alternatives(&cur.cds, p))
                .collect();
            if variants.is_empty() {
                break;
            }
            let mut scored = self.evaluate_near(&cur.cds, &variants)?;
            self.rank_ties(&mut scored);
            let narrows = scored
                .iter()
                .any(|e| (e.gap(), e.class_gap) < (cur.gap(), cur.class_gap));
            if !narrows && self.pairs > 0 {
                let doubles = self.end_pairs(&cur.cds, self.pairs);
                scored.extend(self.evaluate_near(&cur.cds, &doubles)?);
                self.rank_ties(&mut scored);
            }
            let best = scored
                .into_iter()
                .min_by(|a, b| {
                    (a.gap(), a.class_gap).cmp(&(b.gap(), b.class_gap)).then(
                        self.value(a)
                            .partial_cmp(&self.value(b))
                            .expect("finite values"),
                    )
                })
                .expect("non-empty");
            let (here, there) = ((cur.gap(), cur.class_gap), (best.gap(), best.class_gap));
            if there >= here {
                if there > here || level >= sideways {
                    break;
                }
                level += 1;
            }
            tabu.extend(
                (0..self.menus.len())
                    .filter(|&p| cur.cds[3 * p..3 * p + 3] != best.cds[3 * p..3 * p + 3]),
            );
            cur = best;
        }
        let weight = self.menus.weight(&cur.cds)?;
        Ok(Outcome {
            start: first,
            end: cur,
            weight,
            rounds: taken,
        })
    }

    /// The flanks' own minima, `MFE(5′) + MFE(3′)`: the constant a design's
    /// constrained energy adds to its CDS's own MFE.
    pub fn flanks(&self) -> i32 {
        self.flanks
    }

    /// Climb from the tight design `start` for at most `steps` steps, staying
    /// tight: each step evaluates the neighbours [`Toward`] admits, records
    /// the tight ones, and moves to the best of them (see [`Toward`]); it stops
    /// when no tight neighbour qualifies. Every tight neighbour evaluated is a
    /// certified point, not only the end of the path. A start that is not
    /// tight is returned unclimbed.
    ///
    /// With a batch ([`Self::with_batch`]) smaller than the neighbourhood, a
    /// step orders the neighbours by the most each can gain before it is
    /// folded — its weight gain, which for a slope `μ > 0` orders `μ·ΔA` too —
    /// and evaluates them a batch at a time, moving to the best qualifying
    /// tight neighbour of the first batch that holds one (first improvement).
    /// A batch covering the neighbourhood evaluates it in its own order and
    /// takes the best over all of it.
    pub fn climb(&self, start: &str, toward: Toward, steps: usize) -> Result<Climbed, RepairError> {
        self.climb_with(start, toward, steps, |_| Ok(()))
    }

    /// [`Self::climb`], handing each step's newly met tight designs to `sink`
    /// as the step ends (so a long climb's finds can be kept as it goes).
    pub fn climb_with(
        &self,
        start: &str,
        toward: Toward,
        steps: usize,
        mut sink: impl FnMut(&[Met]) -> Result<(), RepairError>,
    ) -> Result<Climbed, RepairError> {
        let (first, mut own, mut whole) = self.base_tables(start)?;
        let (mut met, mut path) = (Vec::new(), Vec::new());
        let (mut cur, mut taken) = (first.clone(), 0);
        // (position, codon) of a change -> its energy change when last folded (None: not tight).
        let mut memo = Memo::new();
        if !first.is_tight() {
            return Ok(Climbed {
                start: first,
                met,
                path,
                end: start.to_string(),
                steps: 0,
            });
        }
        for step in 0..steps {
            let own_opt = if self.ordered() {
                self.own_structure(&cur.cds, &own)
            } else {
                String::new()
            };
            let mut ups = self.climb_moves(&cur.cds, &own_opt, toward)?;
            if ups.is_empty() {
                break;
            }
            let here_w = self.menus.weight(&cur.cds)?;
            if self.batch < ups.len() {
                ups = self.order_moves(&cur, &own_opt, ups, toward, &memo)?;
            }
            let here = (cur.constrained, here_w);
            let from = met.len();
            // A batched step keeps the tables of its batch's tight designs, so
            // the one it moves to is not folded again as the next base.
            let keep = self.batch < ups.len();
            let (mut next, mut kept) = (None, Vec::new());
            for part in ups.chunks(self.batch.min(ups.len())).take(self.max_batches) {
                let evaluated = if keep {
                    self.evaluate_keeping(&cur.cds, (&own, &whole), part)?
                } else {
                    self.evaluate_with(&cur.cds, (&own, &whole), part)?
                        .into_iter()
                        .map(|e| (e, None))
                        .collect()
                };
                let mut tight = Vec::new();
                for (e, tables) in evaluated {
                    if changed(&cur.cds, &e.cds).len() == 1 {
                        let de = e.is_tight().then_some(e.constrained - cur.constrained);
                        memo.insert(change(&cur.cds, &e.cds), de);
                    }
                    if e.is_tight() {
                        let weight = self.menus.weight(&e.cds)?;
                        met.push(Met {
                            cds: e.cds.clone(),
                            weight,
                            constrained: e.constrained,
                            step,
                        });
                        kept.extend(tables.map(|t| (e.cds.clone(), t)));
                        tight.push((weight, e));
                    }
                }
                next = best_step(toward, here, tight);
                if next.is_some() {
                    break;
                }
                kept.clear();
            }
            sink(&met[from..])?;
            let Some((_, e)) = next else {
                break;
            };
            let moved = changed(&cur.cds, &e.cds);
            memo.retain(|k, _| !moved.contains(&k.0));
            path.push(e.cds.clone());
            taken += 1;
            if step + 1 < steps {
                (own, whole) = match kept.into_iter().find(|(c, _)| *c == e.cds) {
                    Some((_, tables)) => tables,
                    None => {
                        let (_, o, w) = self.base_tables(&e.cds)?;
                        (o, w)
                    }
                };
            }
            cur = e;
        }
        Ok(Climbed {
            start: first,
            met,
            path,
            end: cur.cds,
            steps: taken,
        })
    }
}

/// The codon positions at which `cds` differs from `base`.
fn changed(base: &str, cds: &str) -> Vec<usize> {
    (0..base.len() / 3)
        .filter(|&p| base[3 * p..3 * p + 3] != cds[3 * p..3 * p + 3])
        .collect()
}

/// A recalled single change's class and key (smaller first): found improving
/// by the slope `mu` (by how much), untried, or found not improving or not
/// tight (see [`Searcher::with_recall`]).
fn recall_rank(memo: &Memo, base: &str, (dw, c): &(f64, String), mu: f64) -> (u8, f64) {
    match memo.get(&change(base, c)) {
        Some(&Some(de)) if mu * dw - f64::from(de) > 1e-9 => (0, f64::from(de) - mu * dw),
        None => (1, 0.0),
        _ => (2, 0.0),
    }
}

/// The pairs `(i, j)`, `i < j`, of a dot-bracket configuration.
fn matching(structure: &str) -> Vec<(usize, usize)> {
    let (mut stack, mut out) = (Vec::new(), Vec::new());
    for (j, ch) in structure.bytes().enumerate() {
        match ch {
            b'(' => stack.push(j),
            b')' => out.push((stack.pop().expect("balanced"), j)),
            _ => {}
        }
    }
    out
}

/// Each symbol's partner in a dot-bracket configuration (`usize::MAX` when
/// unpaired).
fn partners(structure: &str) -> Vec<usize> {
    let mut partner = vec![usize::MAX; structure.len()];
    for (i, j) in matching(structure) {
        (partner[i], partner[j]) = (j, i);
    }
    partner
}

/// Whether two symbols can close a pair (Watson-Crick or GU wobble).
fn can_pair(a: u8, b: u8) -> bool {
    matches!(
        (a, b),
        (b'A', b'U') | (b'U', b'A') | (b'G', b'C') | (b'C', b'G') | (b'G', b'U') | (b'U', b'G')
    )
}

/// The position and new codon of the one-codon change from `base` to `cds`.
fn change(base: &str, cds: &str) -> (usize, String) {
    let p = base
        .bytes()
        .zip(cds.bytes())
        .position(|(a, b)| a != b)
        .map_or(0, |k| k / 3);
    (p, cds[3 * p..3 * p + 3].to_string())
}

/// The tight design a climb step moves to among `tight` (weight, design)
/// pairs, from a design of constrained energy and weight `here` (see [`Toward`]).
fn best_step(
    toward: Toward,
    (here_e, here_w): (i32, f64),
    tight: Vec<(f64, Evaluated)>,
) -> Option<(f64, Evaluated)> {
    let score = |constrained: i32, w: f64, mu: f64| f64::from(constrained) - mu * w;
    match toward {
        Toward::Weight => tight.into_iter().reduce(|best, t| {
            let better = t.0 > best.0 || (t.0 == best.0 && t.1.constrained < best.1.constrained);
            if better {
                t
            } else {
                best
            }
        }),
        Toward::Slope(mu) => {
            let here = score(here_e, here_w, mu);
            tight
                .into_iter()
                .filter(|(w, e)| score(e.constrained, *w, mu) < here - 1e-9)
                .reduce(|best, t| {
                    if score(t.1.constrained, t.0, mu) < score(best.1.constrained, best.0, mu) {
                        t
                    } else {
                        best
                    }
                })
        }
    }
}
