//! The band above the best-adapted modular design found: every
//! design there is visited, most adapted first.
//!
//! Nothing caps the band. The best-adapted modular design is on
//! the front whatever its energy, so no energy bound rules a design out, and
//! the band's points are found only by checking its designs. A walk in score
//! order reaches them through every structure of each design; enumerating
//! the designs directly reaches each once.
//!
//! The band is taken in slices of adaptation from the top down. A slice is
//! finished only when every design in it has been checked, so a budget that
//! runs out leaves exactly the adaptations below the last finished slice
//! open.
//!
//! Where forbidden motifs narrow the molecule, the enumeration spells only
//! the designs that are free of them. A forbidden-motif automaton is carried
//! alongside the spelling and a codon that would complete an occurrence is
//! never pushed, so no derivation is spent below one — see [`Reach`]. The
//! alternative, spelling every design and refusing the ones that spell a
//! motif, charges the whole subtree below each refusal to the enumeration.

use std::time::Instant;

use derna_codon_dp::cai::compute_cai_tables;
use derna_codon_dp::codon_table::{get_codon_index, AMINO_ACIDS};

use crate::ac::ScopedAc;
use crate::free_fold::{
    sequence_mfe_within, sequence_mfe_within_reusing, CdsWindow, FreeFoldError, ZukerMfes,
};
use crate::region::RegionConstraints;
use crate::{CodonUsage, Nucleotide, Params};

/// How many slices the band is cut into.
const SLICES: usize = 16;

/// Designs spelled before they go to the workers together.
const CHUNK: usize = 1024;

/// How close two adaptations are to count as equal.
const CAI_TOL: f64 = 1e-9;

/// A modular design: constrained energy in dekacalories, coding
/// log-adaptation and the whole molecule.
pub(super) type Kept = (f64, f64, String);

/// One residue's codon choices: spelling and log-adaptation.
type Choices = Vec<([u8; 3], f64)>;

/// The nucleotides in [`Nucleotide`] index order, which is the order a codon's
/// three bytes are in.
const BASES: [Nucleotide; 4] = [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U];

/// What the molecule's band enumeration needs.
pub(super) struct Band<'a> {
    pub params: &'a Params,
    pub utr5: &'a str,
    pub utr3: &'a str,
    pub constraints: RegionConstraints,
    pub threads: usize,
    /// The largest gap, in kcal/mol, a design kept may have.
    epsilon: f64,
    /// Keeps a whole molecule as a design; a design it refuses is not
    /// checked. A caller that also hands over the automaton behind the test
    /// never reaches it with a refused design, because none is spelled; it
    /// stands for the test alone, for a caller that supplies no automaton.
    admit: Option<&'a super::Admit<'a>>,
    choices: Vec<Choices>,
    /// What each prefix can still reach, and which codons it may be extended
    /// by.
    reach: Reach<'a>,
}

/// What a coding prefix can still reach: the forbidden-motif automaton the
/// enumeration must stay clean of, and the best and worst log-adaptation
/// reachable from each position under it.
///
/// The automaton reads the **whole molecule**, so a motif may be completed
/// across either flank boundary as well as inside the coding sequence. Both
/// boundaries are held here rather than at the leaf: the state is seeded with
/// the 5' flank, and whether the 3' flank joins cleanly is folded into the
/// table's last row. A coding prefix whose completion would spell a motif
/// across either boundary therefore has no reachable adaptation at all, and
/// the enumeration abandons it where it abandons any branch that cannot reach
/// the adaptation window.
///
/// **Why the bounds are indexed by automaton state.** The best and worst
/// adaptation the residues below a position can add are read in comparisons
/// that *prune*, so they have to bound what is reachable under the
/// restriction, not what is reachable without it. A motif narrows a position's
/// codons, and which ones it narrows depends on what the prefix spelled — that
/// is the automaton state — so a bound held per position alone is the wrong
/// quantity: it is the unrestricted one, which overstates the best and
/// understates the worst. Indexing by state makes both exact over the
/// restricted lattice, which is what keeps the prune from cutting an
/// admissible design and is also what lets it cut a branch with no admissible
/// completion at all.
///
/// The tables cost `(residues + 1) × states` of each bound, with one state
/// when nothing is forbidden, where every entry is the unrestricted bound.
struct Reach<'a> {
    /// The automaton every design must be clean of; `None` when nothing is
    /// forbidden, and then one state and every codon admissible.
    ac: Option<ScopedAc<'a>>,
    /// Where the 5' flank leaves the automaton, or `None` when the flank
    /// itself already completes a forbidden occurrence — which no coding
    /// sequence can undo, so no design is admissible.
    seed: Option<u32>,
    /// Where the coding sequence starts in the molecule, which is the
    /// position the automaton reads the first codon at.
    cds_start: usize,
    /// How many automaton states the tables are indexed by.
    states: usize,
    /// `best[i * states + q]`: the most log-adaptation the residues from `i`
    /// on can add to a prefix that leaves the automaton in state `q`, or
    /// [`f64::NEG_INFINITY`] when none of them completes an admissible
    /// molecule.
    best: Vec<f64>,
    /// As `best`, the least; [`f64::INFINITY`] where nothing completes.
    worst: Vec<f64>,
}

/// What a band enumeration found and how far it got.
pub(super) struct Enumerated {
    /// Modular designs: (constrained energy in dekacalories,
    /// coding log-adaptation, whole molecule).
    pub found: Vec<Kept>,
    /// The log-adaptation down to which every design was checked, or `None`
    /// when the whole band was.
    pub open_below: Option<f64>,
    pub checked: u64,
}

impl<'a> Reach<'a> {
    /// What `choices` can reach inside `utr5 · coding · utr3` under `ac`.
    fn new(ac: Option<ScopedAc<'a>>, (utr5, utr3): (&str, &str), choices: &[Choices]) -> Self {
        let states = ac.map_or(1, |ac| ac.num_states());
        let residues = choices.len();
        let mut reach = Self {
            ac,
            seed: Some(0),
            cds_start: utr5.len(),
            states,
            best: vec![f64::NEG_INFINITY; (residues + 1) * states],
            worst: vec![f64::INFINITY; (residues + 1) * states],
        };
        reach.seed = reach.read(0, 0, utr5);
        // The last row is the 3' join: a complete coding sequence adds
        // nothing more, and reaches a molecule at all only where the flank
        // after it completes no occurrence.
        for q in 0..states as u32 {
            if reach
                .read(reach.cds_start + 3 * residues, q, utr3)
                .is_some()
            {
                reach.best[residues * states + q as usize] = 0.0;
                reach.worst[residues * states + q as usize] = 0.0;
            }
        }
        for i in (0..residues).rev() {
            for q in 0..states as u32 {
                let (mut best, mut worst) = (f64::NEG_INFINITY, f64::INFINITY);
                for &(codon, weight) in &choices[i] {
                    if let Some(next) = reach.step(i, q, codon) {
                        best = best.max(weight + reach.best_at(i + 1, next));
                        worst = worst.min(weight + reach.worst_at(i + 1, next));
                    }
                }
                reach.best[i * states + q as usize] = best;
                reach.worst[i * states + q as usize] = worst;
            }
        }
        reach
    }

    /// Read `seq` from state `q`, its first base at molecule position `pos`;
    /// `None` when it completes a forbidden occurrence, or when it is not
    /// spelled in nucleotides.
    ///
    /// A flank that is not nucleotides admits no molecule, which is how the
    /// whole-molecule test this stands in for reads it.
    fn read(&self, pos: usize, q: u32, seq: &str) -> Option<u32> {
        let Some(ac) = self.ac else {
            return Some(0);
        };
        let mut q = q;
        for (k, byte) in seq.bytes().enumerate() {
            q = ac.step_at((pos + k) as u32, q, Nucleotide::from_byte(byte)?)?;
        }
        Some(q)
    }

    /// Read residue `i`'s `codon` from state `q`; `None` when one of its three
    /// bases completes a forbidden occurrence, so the codon may not be spelled
    /// there.
    fn step(&self, i: usize, q: u32, codon: [u8; 3]) -> Option<u32> {
        let Some(ac) = self.ac else {
            return Some(0);
        };
        let mut q = q;
        let base = self.cds_start + 3 * i;
        for (k, &n) in codon.iter().enumerate() {
            q = ac.step_at((base + k) as u32, q, BASES[n as usize])?;
        }
        Some(q)
    }

    /// The most log-adaptation the residues from `i` on can add from state
    /// `q`.
    fn best_at(&self, i: usize, q: u32) -> f64 {
        self.best[i * self.states + q as usize]
    }

    /// The least log-adaptation the residues from `i` on can add from state
    /// `q`.
    fn worst_at(&self, i: usize, q: u32) -> f64 {
        self.worst[i * self.states + q as usize]
    }
}

impl<'a> Band<'a> {
    pub fn new(
        params: &'a Params,
        protein: &str,
        usage: &CodonUsage,
        (utr5, utr3): (&'a str, &'a str),
        (constraints, allowed): (RegionConstraints, &[Vec<[u8; 3]>]),
        (threads, admit, motifs): (usize, Option<&'a super::Admit<'a>>, Option<ScopedAc<'a>>),
    ) -> Self {
        let tables = compute_cai_tables(usage);
        let choices: Vec<Choices> = crate::residues(protein)
            .expect("a protein the lattice accepted")
            .into_iter()
            .zip(allowed)
            .map(|(a, allowed)| {
                let mut c: Choices = AMINO_ACIDS[a]
                    .codons
                    .iter()
                    .filter(|c| allowed.contains(&c.nuc))
                    .map(|c| {
                        let i = get_codon_index(a, c.nuc).expect("a codon of its residue");
                        (c.nuc, tables.log_cai[a][i])
                    })
                    .collect();
                // Most adapted first, so the enumeration meets the top first.
                c.sort_by(|x, y| y.1.total_cmp(&x.1));
                c
            })
            .collect();
        let reach = Reach::new(motifs, (utr5, utr3), &choices);
        Self {
            params,
            utr5,
            utr3,
            constraints,
            threads: threads.max(1),
            epsilon: 0.0,
            admit,
            choices,
            reach,
        }
    }

    /// The band keeping the designs whose gap is at most `epsilon` kcal/mol
    /// rather than the modular ones.
    pub fn within(self, epsilon: f64) -> Self {
        Self { epsilon, ..self }
    }

    /// The best log-adaptation any design has, or [`f64::NEG_INFINITY`] when
    /// no design is admissible.
    pub fn top(&self) -> f64 {
        self.reach
            .seed
            .map_or(f64::NEG_INFINITY, |q| self.reach.best_at(0, q))
    }

    /// The worst log-adaptation any design has, or [`f64::INFINITY`] when no
    /// design is admissible. A search names the range it wants, so only a
    /// caller that wants the whole band reads this.
    #[cfg(test)]
    pub fn floor(&self) -> f64 {
        self.reach
            .seed
            .map_or(f64::INFINITY, |q| self.reach.worst_at(0, q))
    }

    /// Check every design with log-adaptation in `[lo, hi]`, slice by slice
    /// from the top, until `deadline` — or until a slice finds a
    /// modular design, whichever comes first. A band an earlier
    /// visit finished down to `hi` resumes there. Each finished slice is
    /// handed to `settled`: the log-adaptation down to which every design has
    /// now been checked, and the modular designs the slice found.
    ///
    /// The band exists because nothing bounds an uncapped region in energy, so
    /// every design above the best modular one must be checked
    /// one at a time. A design it finds ends that: the range below the find is
    /// capped by it, and a capped range is what a window walk searches, in
    /// energy order and with the bound pruning whole subtrees. So the first
    /// find returns the rest of the range as open rather than enumerating it,
    /// and the caller's next round walks it as a window.
    ///
    /// Returning the range as open is the same path the deadline takes, and
    /// the find itself is returned, so nothing is lost: the region is searched
    /// by the round after, with the cap the band has earned.
    pub fn enumerate(
        &self,
        lo: f64,
        hi: f64,
        deadline: Instant,
        settled: &mut dyn FnMut(f64, &[Kept]),
    ) -> Result<Enumerated, FreeFoldError> {
        let top = self.top().min(hi);
        let mut out = Enumerated {
            found: Vec::new(),
            open_below: None,
            checked: 0,
        };
        if lo > top + CAI_TOL {
            return Ok(out);
        }
        let width = (top - lo) / SLICES as f64;
        for k in 0..SLICES {
            let hi = top - width * k as f64;
            let below = if k + 1 == SLICES {
                lo
            } else {
                top - width * (k + 1) as f64
            };
            // The top slice holds its upper end itself.
            let ceiling = if k == 0 { hi + 2.0 * CAI_TOL } else { hi };
            let before = out.found.len();
            if Instant::now() >= deadline || !self.slice(below, ceiling, deadline, &mut out)? {
                out.open_below = Some(hi);
                return Ok(out);
            }
            settled(below, &out.found[before..]);
            // This slice found one: everything below is capped by it now, so
            // hand that range back for a window walk instead of enumerating it.
            if out.found.len() > before && below > lo + CAI_TOL {
                out.open_below = Some(below);
                return Ok(out);
            }
        }
        Ok(out)
    }

    /// Check every design with log-adaptation in `[lo, hi)`; `false` when the
    /// deadline came first.
    fn slice(
        &self,
        lo: f64,
        hi: f64,
        deadline: Instant,
        out: &mut Enumerated,
    ) -> Result<bool, FreeFoldError> {
        // No state the 5' flank can leave the automaton in: the flank itself
        // spells a forbidden motif, so the slice holds no design.
        let Some(seed) = self.reach.seed else {
            return Ok(true);
        };
        let mut pending: Vec<(String, f64)> = Vec::with_capacity(CHUNK);
        let mut prefix = String::with_capacity(3 * self.choices.len());
        let mut finished = true;
        self.spell((0, seed), 0.0, (lo, hi), &mut prefix, &mut |cds, cai| {
            pending.push((cds.to_string(), cai));
            if pending.len() < CHUNK {
                return Ok(true);
            }
            let batch = std::mem::take(&mut pending);
            finished = self.check(&batch, deadline, out)? && Instant::now() < deadline;
            Ok(finished)
        })?;
        if finished && !pending.is_empty() {
            finished = self.check(&pending, deadline, out)?;
        }
        Ok(finished)
    }

    /// Spell every coding sequence from position `i`, the automaton in state
    /// `q`, with log-adaptation in `window`, handing each to `visit`; stop
    /// when `visit` returns `false`.
    ///
    /// A codon that would complete a forbidden motif is never pushed, so no
    /// derivation below it is spent and no design that spells one is ever
    /// handed on. The bounds the adaptation window is tested against are the
    /// restricted ones, so they cut a branch exactly when it holds no design
    /// in the window — including the branch that holds no admissible design
    /// at all.
    fn spell(
        &self,
        (i, q): (usize, u32),
        cai: f64,
        window: (f64, f64),
        prefix: &mut String,
        visit: &mut dyn FnMut(&str, f64) -> Result<bool, FreeFoldError>,
    ) -> Result<bool, FreeFoldError> {
        let (best, worst) = (self.reach.best_at(i, q), self.reach.worst_at(i, q));
        // Nothing below here completes a molecule the automaton admits. Both
        // bounds are infinite together, since they are the extremes of one
        // set of completions.
        if !best.is_finite() {
            return Ok(true);
        }
        if cai + best < window.0 - CAI_TOL || cai + worst >= window.1 - CAI_TOL {
            return Ok(true);
        }
        if i == self.choices.len() {
            return visit(prefix, cai);
        }
        for &(codon, w) in &self.choices[i] {
            let Some(next) = self.reach.step(i, q, codon) else {
                continue;
            };
            for &n in &codon {
                prefix.push(b"ACGU"[n as usize] as char);
            }
            let go_on = self.spell((i + 1, next), cai + w, window, prefix, visit)?;
            prefix.truncate(prefix.len() - 3);
            if !go_on {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Check a batch of coding sequences on the band's workers; `false` when
    /// the deadline came before every design was checked. A worker reads the
    /// clock before each design, so the deadline is overrun by at most one
    /// check per worker, however long the batch.
    fn check(
        &self,
        batch: &[(String, f64)],
        deadline: Instant,
        out: &mut Enumerated,
    ) -> Result<bool, FreeFoldError> {
        let cds = CdsWindow {
            start: self.utr5.len(),
            end: self.utr5.len() + batch.first().map_or(0, |b| b.0.len()),
        };
        let workers = self.threads.min(batch.len()).max(1);
        // Each worker takes a contiguous run of the batch: designs spelled one
        // after another differ in their last codons, so under a hard table
        // each design's folds copy most cells from the one before.
        let per = batch.len().div_ceil(workers);
        let hard = self.constraints.is_hard();
        let results: Vec<Result<(Vec<Kept>, u64), FreeFoldError>> = std::thread::scope(|scope| {
            let handles: Vec<_> = batch
                .chunks(per)
                .map(|run| {
                    scope.spawn(move || {
                        let mut kept = Vec::new();
                        let mut checked = 0;
                        let mut base: Option<ZukerMfes> = None;
                        for (coding, cai) in run {
                            if Instant::now() >= deadline {
                                break;
                            }
                            checked += 1;
                            let whole = format!("{}{coding}{}", self.utr5, self.utr3);
                            if self.admit.is_some_and(|admit| !admit(&whole)) {
                                continue;
                            }
                            let so = if hard {
                                let (so, mfes) = sequence_mfe_within_reusing(
                                    self.params,
                                    &whole,
                                    cds,
                                    self.constraints,
                                    base.as_ref(),
                                    self.epsilon,
                                )?;
                                base = Some(mfes);
                                so
                            } else {
                                sequence_mfe_within(
                                    self.params,
                                    &whole,
                                    cds,
                                    self.constraints,
                                    self.epsilon,
                                )?
                            };
                            if let Some(mfe) = so {
                                kept.push(((mfe * 100.0).round(), *cai, whole));
                            }
                        }
                        Ok((kept, checked))
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("a band worker"))
                .collect()
        });
        let before = out.checked;
        for r in results {
            let (kept, checked) = r?;
            out.found.extend(kept);
            out.checked += checked;
        }
        Ok(out.checked - before == batch.len() as u64)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::ac::{Ac, MotifWindow};
    use crate::region::{Enforcement, PairClass};

    /// A 5' flank ending in `CC`, a coding sequence, and a 3' flank starting
    /// with `CC`, so a motif can be spelled across either boundary and across
    /// neither flank on its own.
    const FIVE: &str = "AAAAAAAAAACC";
    const THREE: &str = "CCAAAAAAAAAA";

    /// Methionine, leucine, glycine: one codon, six and four, so the lattice
    /// holds 24 designs and both of the many-codon residues sit beside a
    /// flank.
    const PROTEIN: &str = "MLG";

    /// Two patterns that each need bases from a flank and bases from the
    /// coding sequence, so neither is visible to an automaton reading the
    /// coding sequence alone.
    ///
    /// `CCAUGC` needs the 5' flank's `CC`, the start codon and a leucine codon
    /// starting `C`. `GGGCC` needs the 3' flank's `CC` and completes at two
    /// places: the glycine codon `GGG` before the flank, and a leucine codon
    /// ending `G` followed by `GGC`.
    const ACROSS_BOTH_FLANKS: [&str; 2] = ["CCAUGC", "GGGCC"];

    /// [`ACROSS_BOTH_FLANKS`] and one more, `GGUCC`, which forbids the glycine
    /// codon `GGU` before the 3' flank's `CC`. The set removes the most
    /// adapted leucine codon and the least adapted glycine codon, so the
    /// restriction moves the best *and* the worst reachable adaptation —
    /// `ACROSS_BOTH_FLANKS` alone leaves the worst where the lattice has it.
    const MOVES_BOTH_BOUNDS: [&str; 3] = ["CCAUGC", "GGGCC", "GGUCC"];

    /// Whether `cds` spells the occurrence that crosses the 5' boundary.
    fn across_five(cds: &str) -> bool {
        codon(cds, 1).starts_with('C')
    }

    /// Whether `cds` spells either occurrence that crosses the 3' boundary.
    fn across_three(cds: &str) -> bool {
        codon(cds, 2) == "GGG" || (codon(cds, 1).ends_with('G') && codon(cds, 2) == "GGC")
    }

    /// The whole-molecule test the band's `admit` is built from, stated here
    /// as the band's caller states it, so a comparison against it is a
    /// comparison against the test the enumeration replaces.
    fn admits(scoped: &ScopedAc<'_>, whole: &str) -> bool {
        let nucs: Option<Vec<Nucleotide>> = whole.bytes().map(Nucleotide::from_byte).collect();
        nucs.is_some_and(|nucs| scoped.is_clean(&nucs))
    }

    /// Every coding sequence the band spells in `window`, with its
    /// log-adaptation, in the order it spells them.
    fn spelled(band: &Band<'_>, window: (f64, f64)) -> Vec<(String, f64)> {
        let mut out = Vec::new();
        let Some(seed) = band.reach.seed else {
            return out;
        };
        let mut prefix = String::new();
        band.spell((0, seed), 0.0, window, &mut prefix, &mut |cds, cai| {
            out.push((cds.to_string(), cai));
            Ok(true)
        })
        .expect("spelling asks for no fold");
        out
    }

    /// Every design of the lattice, in the order the band spells them.
    fn every(band: &Band<'_>) -> Vec<(String, f64)> {
        spelled(band, (f64::NEG_INFINITY, f64::INFINITY))
    }

    /// Residue `i`'s codon of a coding sequence.
    fn codon(cds: &str, i: usize) -> &str {
        &cds[3 * i..3 * i + 3]
    }

    /// An empty motif set restricts nothing, so the enumeration is the
    /// lattice's own: the same designs in the same order, with the same
    /// bounds at every position. This is the floor a behaviour-preserving
    /// change has to hold — the restricted bounds reduce to the unrestricted
    /// ones, rather than being a second quantity that happens to agree.
    #[test]
    fn an_empty_motif_set_leaves_the_enumeration_as_the_lattice() {
        let params = Params::load_default().expect("params");
        let usage = CodonUsage::load_default().expect("usage");
        // Nothing is folded here: the enumeration is what is under test.
        let table = RegionConstraints::default();
        let allowed = super::super::synonymous_codons(PROTEIN).expect("residues");
        let plain = Band::new(
            &params,
            PROTEIN,
            &usage,
            (FIVE, THREE),
            (table, &allowed),
            (1, None, None),
        );
        let ac = Ac::build::<&str>(&[]).expect("an empty motif set");
        assert!(ac.is_pass_through(), "an empty set forbids nothing");
        let scoped = ScopedAc::new(&ac, MotifWindow::all());
        let empty = Band::new(
            &params,
            PROTEIN,
            &usage,
            (FIVE, THREE),
            (table, &allowed),
            (1, None, Some(scoped)),
        );
        assert_eq!(every(&plain), every(&empty), "the designs differ");
        assert_eq!(plain.top(), empty.top(), "the best adaptation differs");
        assert_eq!(plain.floor(), empty.floor(), "the worst adaptation differs");
        for i in 0..=PROTEIN.len() {
            assert_eq!(
                (plain.reach.best_at(i, 0), plain.reach.worst_at(i, 0)),
                (empty.reach.best_at(i, 0), empty.reach.worst_at(i, 0)),
                "the bounds at residue {i} differ"
            );
        }
    }

    /// The enumeration spells exactly the designs the whole-molecule test
    /// admits, in the same order — the refusal moved from the check to the
    /// spelling without changing which designs reach the check.
    ///
    /// The comparison is against the test itself over the unrestricted
    /// lattice, which is what the band did before, so it is the whole claim:
    /// nothing admissible is lost and nothing refused is spelled.
    #[test]
    fn the_enumeration_spells_what_the_whole_molecule_test_admits() {
        let params = Params::load_default().expect("params");
        let usage = CodonUsage::load_default().expect("usage");
        // Nothing is folded here: the enumeration is what is under test.
        let table = RegionConstraints::default();
        let allowed = super::super::synonymous_codons(PROTEIN).expect("residues");
        let ac = Ac::build(&ACROSS_BOTH_FLANKS).expect("motifs");
        let scoped = ScopedAc::new(&ac, MotifWindow::all());
        let plain = Band::new(
            &params,
            PROTEIN,
            &usage,
            (FIVE, THREE),
            (table, &allowed),
            (1, None, None),
        );
        let restricted = Band::new(
            &params,
            PROTEIN,
            &usage,
            (FIVE, THREE),
            (table, &allowed),
            (1, None, Some(scoped)),
        );
        let filtered: Vec<(String, f64)> = every(&plain)
            .into_iter()
            .filter(|(cds, _)| admits(&scoped, &format!("{FIVE}{cds}{THREE}")))
            .collect();
        assert_eq!(filtered, every(&restricted), "the admitted designs differ");
        // Non-vacuity: the comparison would hold just as well on an automaton
        // that forbade nothing and on one that forbade everything, so both
        // populations are named.
        assert_eq!(filtered.len(), 5, "the admissible designs");
        assert_eq!(every(&plain).len(), 24, "the lattice's designs");
    }

    /// A motif spelled across a flank boundary is never spelled. The
    /// automaton reads the whole molecule, so the enumeration is seeded with
    /// the 5' flank and joined with the 3' flank; an enumeration over the
    /// coding sequence alone sees neither occurrence and spells all 24
    /// designs.
    ///
    /// The designs the restriction removes are named by which occurrence
    /// removes them rather than counted in one lump: 16 spell the 5'
    /// occurrence, 8 a 3' one and 5 both, so 19 of 24 go and 5 are left.
    #[test]
    fn a_motif_across_a_flank_boundary_is_never_spelled() {
        let params = Params::load_default().expect("params");
        let usage = CodonUsage::load_default().expect("usage");
        // Nothing is folded here: the enumeration is what is under test.
        let table = RegionConstraints::default();
        let allowed = super::super::synonymous_codons(PROTEIN).expect("residues");
        let ac = Ac::build(&ACROSS_BOTH_FLANKS).expect("motifs");
        let scoped = ScopedAc::new(&ac, MotifWindow::all());
        let restricted = Band::new(
            &params,
            PROTEIN,
            &usage,
            (FIVE, THREE),
            (table, &allowed),
            (1, None, Some(scoped)),
        );
        let spelled: Vec<String> = every(&restricted).into_iter().map(|(cds, _)| cds).collect();
        assert!(
            spelled.iter().all(|cds| !across_five(cds)),
            "a design completing CCAUGC across the 5' flank was spelled: {spelled:?}"
        );
        assert!(
            spelled.iter().all(|cds| !across_three(cds)),
            "a design completing GGGCC across the 3' flank was spelled: {spelled:?}"
        );
        // Each occurrence removes designs, and between them they remove every
        // design the restriction removes. The two classes overlap, so the
        // overlap is named too rather than left to the arithmetic.
        let plain = Band::new(
            &params,
            PROTEIN,
            &usage,
            (FIVE, THREE),
            (table, &allowed),
            (1, None, None),
        );
        let all: Vec<String> = every(&plain).into_iter().map(|(cds, _)| cds).collect();
        let five = all.iter().filter(|c| across_five(c)).count();
        let three = all.iter().filter(|c| across_three(c)).count();
        let both = all
            .iter()
            .filter(|c| across_five(c) && across_three(c))
            .count();
        assert_eq!(
            (all.len(), five, three, both),
            (24, 16, 8, 5),
            "the lattice and the designs each occurrence removes"
        );
        assert_eq!(
            all.len() - (five + three - both),
            spelled.len(),
            "what is left"
        );
    }

    /// Both adaptation bounds the prune reads are the restricted ones, and
    /// the restriction moves both. The unrestricted bounds overstate the
    /// band's top and understate its floor, because the set removes the most
    /// adapted leucine codon and the least adapted glycine codon; a prune
    /// reading them would cut at a ceiling no design reaches and open slices
    /// that hold nothing.
    ///
    /// Each restricted bound is exact, not merely different: it equals the
    /// extreme log-adaptation the enumeration actually spells. An offset of
    /// zero would make the whole comparison a no-op, so both are asserted
    /// non-zero — and the two-pattern set is not enough for that, since it
    /// leaves the floor where the lattice has it.
    #[test]
    fn the_restricted_adaptation_bounds_are_tighter_than_the_lattices() {
        let params = Params::load_default().expect("params");
        let usage = CodonUsage::load_default().expect("usage");
        // Nothing is folded here: the enumeration is what is under test.
        let table = RegionConstraints::default();
        let allowed = super::super::synonymous_codons(PROTEIN).expect("residues");
        let ac = Ac::build(&MOVES_BOTH_BOUNDS).expect("motifs");
        let scoped = ScopedAc::new(&ac, MotifWindow::all());
        let plain = Band::new(
            &params,
            PROTEIN,
            &usage,
            (FIVE, THREE),
            (table, &allowed),
            (1, None, None),
        );
        let restricted = Band::new(
            &params,
            PROTEIN,
            &usage,
            (FIVE, THREE),
            (table, &allowed),
            (1, None, Some(scoped)),
        );
        let offset = plain.top() - restricted.top();
        assert!(
            offset > 0.0,
            "the restriction left the bound where it was: offset {offset}, \
             unrestricted top {}, restricted top {}",
            plain.top(),
            restricted.top()
        );
        let floor_offset = restricted.floor() - plain.floor();
        assert!(
            floor_offset > 0.0,
            "the restriction left the floor where it was: offset {floor_offset}, \
             unrestricted floor {}, restricted floor {}",
            plain.floor(),
            restricted.floor()
        );
        // Each restricted bound is the adaptation the enumeration reaches, so
        // it is the exact extreme over the restricted lattice rather than some
        // other number that merely differs from the unrestricted one.
        let designs = every(&restricted);
        let reached = designs.iter().map(|(_, cai)| *cai);
        let highest = reached.clone().fold(f64::NEG_INFINITY, f64::max);
        let lowest = reached.fold(f64::INFINITY, f64::min);
        assert!(
            (restricted.top() - highest).abs() < CAI_TOL
                && (restricted.floor() - lowest).abs() < CAI_TOL,
            "the restricted bounds ({}, {}) are not the adaptations reached \
             ({highest}, {lowest})",
            restricted.top(),
            restricted.floor()
        );
        // What the top offset costs a search that reads the unrestricted
        // bound: the slice it would cut from the unrestricted top down holds
        // nothing, while the slice from the restricted top down holds the one
        // design that reaches it.
        assert!(
            spelled(&restricted, (plain.top(), f64::INFINITY)).is_empty(),
            "the unrestricted top is reached after all"
        );
        assert_eq!(
            spelled(&restricted, (restricted.top(), f64::INFINITY)).len(),
            1,
            "the restricted top is not reached by exactly one design"
        );
        println!(
            "top offset {offset} nats (unrestricted {}, restricted {}); \
             floor offset {floor_offset} nats (unrestricted {}, restricted {})",
            plain.top(),
            restricted.top(),
            plain.floor(),
            restricted.floor()
        );
    }

    /// A 5' flank that spells a forbidden motif by itself admits no design,
    /// which no coding sequence can undo. The enumeration spells nothing
    /// rather than spelling every design and refusing each.
    #[test]
    fn a_flank_that_spells_a_motif_leaves_no_design() {
        let params = Params::load_default().expect("params");
        let usage = CodonUsage::load_default().expect("usage");
        // Nothing is folded here: the enumeration is what is under test.
        let table = RegionConstraints::default();
        let allowed = super::super::synonymous_codons(PROTEIN).expect("residues");
        let ac = Ac::build(&["AAAAAAAAAA"]).expect("a motif the 5' flank spells");
        let scoped = ScopedAc::new(&ac, MotifWindow::all());
        let band = Band::new(
            &params,
            PROTEIN,
            &usage,
            (FIVE, THREE),
            (table, &allowed),
            (1, None, Some(scoped)),
        );
        assert!(band.reach.seed.is_none(), "the flank left a live state");
        assert_eq!(band.top(), f64::NEG_INFINITY, "a design was reachable");
        assert!(every(&band).is_empty(), "a design was spelled");
        let far = Instant::now() + Duration::from_secs(600);
        let got = band
            .enumerate(-1e9, 0.0, far, &mut |_, _| {})
            .expect("checks");
        assert!(got.found.is_empty(), "a design was kept");
        assert_eq!(got.checked, 0, "a design was folded");
    }

    /// A deadline that lands inside a batch stops the batch there, not after
    /// it: far fewer designs than a batch holds are checked, and the band is
    /// left open.
    #[test]
    fn a_deadline_inside_a_batch_stops_it() {
        let params = Params::load_default().expect("params");
        let usage = CodonUsage::load_default().expect("usage");
        let table = RegionConstraints::enforcing(
            &[PairClass::Utr5Cds, PairClass::CdsUtr3],
            Enforcement::Forbid,
        );
        let allowed =
            super::super::synonymous_codons("MSLRLSPRAVLLLGLAAALLVLGGSRRSLAEQ").expect("residues");
        let band = Band::new(
            &params,
            "MSLRLSPRAVLLLGLAAALLVLGGSRRSLAEQ",
            &usage,
            ("GGGAAAUAAGAGAGAAAAGAAGAGUAAG", "UGAUAAUAGGCUGGAGCCUCGGUGGC"),
            (table, &allowed),
            (1, None, None),
        );
        let lo = band.floor();
        let deadline = Instant::now() + Duration::from_millis(50);
        let got = band
            .enumerate(lo, band.top(), deadline, &mut |_, _| {})
            .expect("checks");
        assert!(got.open_below.is_some(), "the band cannot finish in 50 ms");
        assert!(
            got.checked < CHUNK as u64,
            "{} designs checked: the deadline waited for the batch",
            got.checked
        );
    }

    /// A band that finds a modular design stops there and hands
    /// the range below it back: everything below the find is capped by it, and
    /// a capped range belongs to a window walk. Without the hand-back the band
    /// enumerates every remaining slice down to the floor, so the two differ
    /// in how far down they check and in how much they check to get there.
    #[test]
    fn a_band_that_finds_one_hands_the_rest_back() {
        let params = Params::load_default().expect("params");
        let usage = CodonUsage::load_default().expect("usage");
        // Flanks with no partner for the coding window: A pairs only with U,
        // so no cross-region pair can form and the best-adapted designs are
        // modular. The first slice therefore finds one.
        let table = RegionConstraints::enforcing(
            &[PairClass::Utr5Cds, PairClass::CdsUtr3],
            Enforcement::Forbid,
        );
        let allowed = super::super::synonymous_codons("MPGAPGAPGPAGMP").expect("residues");
        let band = Band::new(
            &params,
            "MPGAPGAPGPAGMP",
            &usage,
            ("AAAAAAAAAAAA", "AAAAAAAAAAAA"),
            (table, &allowed),
            (1, None, None),
        );
        let lo = band.floor();
        let far = Instant::now() + Duration::from_secs(600);
        let mut settled = 0;
        let got = band
            .enumerate(lo, band.top(), far, &mut |_, _| settled += 1)
            .expect("checks");
        assert!(!got.found.is_empty(), "no design was modular");
        let handed = got
            .open_below
            .expect("the band found one, so the rest is open");
        assert!(
            handed > lo + CAI_TOL,
            "handed back {handed} at the floor {lo}: nothing was left for a window"
        );
        assert!(
            settled < SLICES,
            "{settled} slices settled of {SLICES}: the band did not stop at its find"
        );
    }
}
