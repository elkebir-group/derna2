//! The clamp table: what a pair of fixed UTR flanks gains by closing a
//! `5′UTR↔3′UTR` helix around a coding region that presents `B` exterior
//! branches.
//!
//! # Why `B` is all that matters
//!
//! Under a constraint forbidding every crossing class at `dangles = 0`, a design's
//! constrained energy separates: `G = MFE(5′) + E(T) + MFE(3′)`, where `T` is the
//! coding region's own structure and its `B` top-level pairs are branches of the
//! exterior loop, each scored by its terminal-AU term alone. Take any structure
//! of the two flanks that pairs `5′` to `3′` and leaves the coding region inside
//! one multiloop. Put `T` there unchanged. Every branch of `T` now pays
//! `ML.intern` on top of the same terminal-AU term, unpaired nucleotides cost
//! `ML.base = 0`, and nothing inside `T` changes. So the free MFE satisfies
//!
//! ```text
//! free ≤ E(T) + Σ flank loops + B · ML.intern = G + D(B),
//! ```
//!
//! where `D(B)` is the least energy of such a flank structure plus
//! `B · ML.intern`, less `MFE(5′) + MFE(3′)`. Nothing in it depends on the
//! coding sequence beyond `B`. A design some constrained-optimal structure of which
//! has `B` branches with `D(B) < 0` is therefore **not modular**.
//! If `D(B) < 0` for every `B` a design can present, no design is.
//!
//! # How it is computed
//!
//! As the constrained fold of a **probe molecule**: the two flanks around a
//! stand-in coding region of `B` stable GC hairpins, each followed by a run of
//! `A` longer than the interior-loop cap. The constraint forbids the two
//! `UTR↔CDS` classes and allows `5′↔3′`, so the probe's fold is the best of
//! the separate folds and every clamp around the stand-in. The trailing `A`
//! run after the last hairpin cannot pair and puts more than the interior-loop
//! cap of unpaired bases between any clamp pair and the hairpins, so every
//! clamp the probe scores closes a multiloop around them — exactly the
//! structures the argument above transplants onto a real design. The table is
//! the probe's fold less its separate fold.
//!
//! The first hairpin starts right after the 5′ flank, with no leading run.
//! Transplanted around a real design, the clamp's leading run grows by the
//! design's own leading unpaired run, which costs nothing in a multiloop
//! (`ML.base = 0`) and is unbounded in the model and in the scan.
//!
//! Because that run is unbounded, a stand-in of many hairpins can fold into one
//! multiloop of its own, its spacers serving as the loop's unpaired runs. Its
//! separate fold then drops below `B` hairpins' worth and the probe reports
//! less gain than the clamp has. The argument above fixes the gain's growth —
//! past one branch, exactly `ML.intern` per branch — so each entry from `B = 2`
//! on is also capped by the entry before it plus `ML.intern` whenever that
//! entry gains; where the probe is the weaker of the two, the extension is the
//! table's value.
//!
//! A multiloop's unpaired runs are free and unbounded, so a stand-in of many
//! hairpins can fold into one multiloop of its own, its spacers serving as the
//! loop's unpaired runs. Its separate fold then drops below `B` hairpins' worth
//! and the probe reports less gain than the clamp has. The argument above fixes
//! the gain's growth — past one branch, exactly `ML.intern` per branch — so each
//! entry from `B = 2` on is also capped by the entry before it plus `ML.intern`
//! whenever that entry gains; where the probe is the weaker of the two, the
//! extension is the table's value.
//!
//! Interior-loop clamps (`B = 1` with no flank branch inside) are left out:
//! they depend on the coding flanks' lengths and terminal mismatches. Leaving a
//! valid structure out only weakens the bound, never its soundness.
//!
//! `B = 0` admits one more clamp, a hairpin closed over the bare coding region,
//! whose energy depends on the region's length and on the nucleotides beside
//! the closing pair. The table keeps only the multiloop value there: it folds
//! the probe at two lengths of bare region, both too long for an interior loop,
//! and reports the entry only when they agree. A hairpin's energy grows
//! strictly with its loop, so agreement means a multiloop clamp attains the
//! value, and that clamp fits any coding region. Otherwise the entry is `None`.

use rna_mfe::Params;

use crate::free_fold::{sequence_constrained_mfe, CdsWindow, FreeFoldError};
use crate::region::{Enforcement, RegionConstraints};

/// One stand-in branch: a six-pair GC stem closed by a GNRA tetraloop.
const HAIRPIN: &str = "GGGCGCGAAAGCGCCC";

/// A run of `A` longer than the 30-nucleotide interior-loop cap, so no
/// interior loop can span it. It follows each hairpin, never precedes the
/// first.
const SPACER: usize = 33;

/// The two bare-region lengths whose probes must agree for `B = 0`.
const BARE: [usize; 2] = [300, 1200];

/// The stand-in coding region for `branches` exterior branches: hairpins,
/// each followed by a spacer, padded with `A` to a whole number of codons.
/// No leading spacer: see the module docs on the multiloop's leading run.
fn stand_in(branches: usize) -> String {
    let mut cds = String::new();
    for _ in 0..branches {
        cds.push_str(HAIRPIN);
        cds.push_str(&"A".repeat(SPACER));
    }
    let len = cds.len().div_ceil(3) * 3;
    cds.push_str(&"A".repeat(len - cds.len()));
    cds
}

/// The clamp gain `D(B)` in kcal/mol for `B = 0..=b_max`, clipped at zero:
/// a negative entry is the best clamp's gain over folding the flanks apart,
/// and `0.0` means no clamp beats folding them apart. The `B = 0` entry is
/// `None` when only a hairpin over the bare region attains it (see the module
/// docs); every other entry is `Some` and holds at any coding length.
pub fn clamp_table(
    params: &Params,
    utr5: &str,
    utr3: &str,
    b_max: usize,
) -> Result<Vec<Option<f64>>, FreeFoldError> {
    let gain = |cds: &str| clamp_gain(params, utr5, utr3, cds);
    let short = gain(&"A".repeat(BARE[0]))?;
    let long = gain(&"A".repeat(BARE[1]))?;
    let intern = f64::from(params.ml.intern_37) / 100.0;
    let mut table = vec![(short == long).then_some(short)];
    for b in 1..=b_max {
        let probed = gain(&stand_in(b))?;
        // Past one branch, each more adds exactly `ML.intern` to a multiloop
        // clamp that gains (module docs), so an entry is never weaker than the
        // one before it plus `ML.intern`; the probe alone can be, once its
        // stand-in folds into a multiloop of its own.
        let extended = match table[b - 1] {
            Some(prev) if b >= 2 && prev < 0.0 => ((prev + intern) * 100.0).round() / 100.0,
            _ => probed,
        };
        table.push(Some(probed.min(extended)));
    }
    Ok(table)
}

/// `D` for one stand-in coding region `cds` (a multiple of three long).
fn clamp_gain(params: &Params, utr5: &str, utr3: &str, cds: &str) -> Result<f64, FreeFoldError> {
    let probe = format!("{utr5}{cds}{utr3}");
    let window = CdsWindow::new(utr5.len(), cds.len() / 3);
    let fold = |rc| sequence_constrained_mfe(params, &probe, window, rc);
    let clamped = fold(RegionConstraints::symmetric_isolation(Enforcement::Forbid))?;
    let apart = fold(RegionConstraints::full_mutual_isolation(
        Enforcement::Forbid,
    ))?;
    Ok(((clamped - apart) * 100.0).round() / 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Params {
        Params::load_default().expect("bundled parameters")
    }

    #[test]
    fn stand_in_is_whole_codons_with_one_hairpin_per_branch() {
        for b in 0..4 {
            let s = stand_in(b);
            assert_eq!(s.len() % 3, 0);
            assert_eq!(s.matches(HAIRPIN).count(), b);
            assert!(b == 0 || s.starts_with(HAIRPIN), "no leading run: {s}");
        }
    }

    #[test]
    fn flanks_that_cannot_pair_have_no_clamp() {
        let t = clamp_table(&params(), "ACAACCACAACC", "ACAACCACAACC", 4).unwrap();
        assert!(t.iter().all(|&d| d == Some(0.0)), "{t:?}");
    }

    #[test]
    fn complementary_flanks_clamp_and_each_branch_adds_ml_intern() {
        let utr5 = "GGGGCCGGGGCCAAAAAAGGGAAACCCAAAAAAGGGAAACCCAA";
        let utr3 = "AAGGGAAACCCAAAAAAGGGAAACCCAAAAAAGGCCCCGGCCCC";
        let t: Vec<f64> = clamp_table(&params(), utr5, utr3, 5).unwrap()[1..]
            .iter()
            .map(|d| d.expect("B >= 1 is always reported"))
            .collect();
        assert!(t[1] < 0.0, "{t:?}");
        let intern = params().ml.intern_37 as f64 / 100.0;
        for b in 1..4 {
            assert!((t[b + 1] - t[b] - intern).abs() < 1e-9, "{t:?}");
        }
    }
}
