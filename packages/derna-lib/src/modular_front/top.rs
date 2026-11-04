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

use std::time::Instant;

use derna_codon_dp::cai::compute_cai_tables;
use derna_codon_dp::codon_table::{get_codon_index, AMINO_ACIDS};

use crate::free_fold::{
    sequence_mfe_within, sequence_mfe_within_reusing, CdsWindow, FreeFoldError, ZukerMfes,
};
use crate::region::RegionConstraints;
use crate::{CodonUsage, Params};

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

/// What the molecule's band enumeration needs.
pub(super) struct Band<'a> {
    pub params: &'a Params,
    pub utr5: &'a str,
    pub utr3: &'a str,
    pub constraints: RegionConstraints,
    pub threads: usize,
    /// The largest gap, in kcal/mol, a design kept may have.
    epsilon: f64,
    /// Keeps a whole molecule as a design; a design it refuses is not checked.
    admit: Option<&'a super::Admit<'a>>,
    choices: Vec<Choices>,
    /// Per position, the best and worst log-adaptation the residues from it
    /// to the end can add.
    best_after: Vec<f64>,
    worst_after: Vec<f64>,
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

impl<'a> Band<'a> {
    pub fn new(
        params: &'a Params,
        protein: &str,
        usage: &CodonUsage,
        (utr5, utr3): (&'a str, &'a str),
        (constraints, allowed): (RegionConstraints, &[Vec<[u8; 3]>]),
        (threads, admit): (usize, Option<&'a super::Admit<'a>>),
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
        let mut best_after = vec![0.0; choices.len() + 1];
        let mut worst_after = vec![0.0; choices.len() + 1];
        for i in (0..choices.len()).rev() {
            best_after[i] = best_after[i + 1] + choices[i].first().map_or(0.0, |c| c.1);
            worst_after[i] = worst_after[i + 1] + choices[i].last().map_or(0.0, |c| c.1);
        }
        Self {
            params,
            utr5,
            utr3,
            constraints,
            threads: threads.max(1),
            epsilon: 0.0,
            admit,
            choices,
            best_after,
            worst_after,
        }
    }

    /// The band keeping the designs whose gap is at most `epsilon` kcal/mol
    /// rather than the modular ones.
    pub fn within(self, epsilon: f64) -> Self {
        Self { epsilon, ..self }
    }

    /// The best log-adaptation any design has.
    pub fn top(&self) -> f64 {
        self.best_after[0]
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
        let mut pending: Vec<(String, f64)> = Vec::with_capacity(CHUNK);
        let mut prefix = String::with_capacity(3 * self.choices.len());
        let mut finished = true;
        self.spell(0, 0.0, (lo, hi), &mut prefix, &mut |cds, cai| {
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

    /// Spell every coding sequence from position `i` with log-adaptation in
    /// `window`, handing each to `visit`; stop when `visit` returns `false`.
    fn spell(
        &self,
        i: usize,
        cai: f64,
        window: (f64, f64),
        prefix: &mut String,
        visit: &mut dyn FnMut(&str, f64) -> Result<bool, FreeFoldError>,
    ) -> Result<bool, FreeFoldError> {
        if cai + self.best_after[i] < window.0 - CAI_TOL
            || cai + self.worst_after[i] >= window.1 - CAI_TOL
        {
            return Ok(true);
        }
        if i == self.choices.len() {
            return visit(prefix, cai);
        }
        for &(codon, w) in &self.choices[i] {
            for &n in &codon {
                prefix.push(b"ACGU"[n as usize] as char);
            }
            let go_on = self.spell(i + 1, cai + w, window, prefix, visit)?;
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
    use crate::region::{Enforcement, PairClass};

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
            (1, None),
        );
        let lo = band.worst_after[0];
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
            (1, None),
        );
        let lo = band.worst_after[0];
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
