//! Shared helpers for the two-pair recurrences (stack, bulge, internal).
//!
//! Mirrors v1's `_MotifSite` / `_iter_two_pair_inner` / `_motif_mfe` from
//! `derna.codon_dp._recurrences`. Each two-pair recurrence yields
//! candidates per enclosed-pair position `(nt_p, nt_q)` and per inner-cell
//! codon assignment `(h, r)`. The candidate's BRD value adds the closing
//! pair's per-loop energy to the inner cell's `E` value, with CAI charged
//! only for AAs whose codon ends at this level.

use rna_mfe::loops::pair_from_nucleotides;
use rna_mfe::params::BasePair;

use crate::plan::{aa_of_nt, consistent_choices, nuc_at, nuc_from_index, CellIdx, Plan, INF};

/// Bundled geometry of a stacking / bulge / internal-loop candidate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MotifSite {
    pub idx: CellIdx,
    /// `(nt_i, nt_j, nt_p, nt_q)` — closing then enclosed pair positions.
    pub nt_pairs: (usize, usize, usize, usize),
    /// `(c, s', d, t')` — enclosed-pair AA indices and their codon offsets.
    pub enclosed: (usize, usize, usize, usize),
}

/// Build a [`MotifSite`] if the enclosed-pair AAs satisfy `c < d`.
/// Mirrors v1's `_make_site` + `_enclosed_aa_indices`.
pub(crate) fn make_site(idx: CellIdx, nt_pairs: (usize, usize, usize, usize)) -> Option<MotifSite> {
    let (_nt_i, _nt_j, nt_p, nt_q) = nt_pairs;
    let c = aa_of_nt(nt_p);
    let d = aa_of_nt(nt_q);
    if c >= d {
        return None;
    }
    Some(MotifSite {
        idx,
        nt_pairs,
        enclosed: (c, nt_p % 3, d, nt_q % 3),
    })
}

/// Closing-pair CAI in dekacal int, charged only when the closing AA's
/// codon "ends" at this level — i.e., the enclosed-pair AA differs from
/// the closing-pair AA. Mirrors v1's `_closing_pair_cai`.
pub(crate) fn closing_pair_cai(plan: &Plan<'_>, site: MotifSite) -> f64 {
    let (a, b, _s, _t, x, y) = site.idx;
    let (c, _sp, d, _tp) = site.enclosed;
    let log_cai = &plan.log_cai_dekacal_per_pos;
    let left = if c != a { log_cai[a][x] } else { 0.0 };
    let right = if d != b { log_cai[b][y] } else { 0.0 };
    left + right
}

/// One inner-cell pointer: `(h, r)` codon assignments at the enclosed-pair
/// AAs and the inner cell's `E` value.
#[derive(Debug, Clone, Copy)]
pub(crate) struct InnerRef {
    pub h: usize,
    pub r: usize,
    pub inner_val: f64,
}

/// Yield `InnerRef` records with a finite `E` value at the inner cell.
/// Inner cells with `E[idx] == INF` are skipped (mirrors v1's `if
/// inner_val < INF`).
pub(crate) fn iter_two_pair_inner<F: FnMut(InnerRef)>(
    plan: &Plan<'_>,
    site: MotifSite,
    mut emit: F,
) {
    let (a, b, _s, _t, x, y) = site.idx;
    let (c, s_prime, d, t_prime) = site.enclosed;
    let base: [(usize, usize); 2] = [(a, x), (b, y)];
    for h in consistent_choices(plan, c, &base) {
        let extended: [(usize, usize); 3] = [(a, x), (b, y), (c, h)];
        for r in consistent_choices(plan, d, &extended) {
            let inner_idx = (c, d, s_prime, t_prime, h, r);
            let i = plan.tables.flat_index(inner_idx);
            let inner_val = plan.tables.e[i];
            if inner_val < INF {
                emit(InnerRef { h, r, inner_val });
            }
        }
    }
}

/// Read the closing and inner base pairs directly from the codons pinned
/// at the outer (`x`, `y`) and enclosed (`h`, `r`) AA positions. Returns
/// `None` if either pair is non-canonical.
///
/// Outer-pair canonicity is checked at the codon-DP layer because Vienna's
/// `E_IntLoop` returns its `INF` sentinel (plus shape-dependent base
/// penalties) when the closing pair is non-canonical — semantically the
/// closure is impossible. Filtering here keeps the cell at `INF` rather
/// than chasing Vienna's exact sentinel-plus-penalty arithmetic.
pub(crate) fn pair_at(
    plan: &Plan<'_>,
    site: MotifSite,
    h: usize,
    r: usize,
) -> Option<(BasePair, BasePair)> {
    let (a, b, s, t, x, y) = site.idx;
    let (c, s_prime, d, t_prime) = site.enclosed;
    let outer = pair_from_nucleotides(
        nuc_from_index(nuc_at(plan, a, x, s)),
        nuc_from_index(nuc_at(plan, b, y, t)),
    );
    if matches!(outer, BasePair::NN) {
        return None;
    }
    let inner = pair_from_nucleotides(
        nuc_from_index(nuc_at(plan, c, h, s_prime)),
        nuc_from_index(nuc_at(plan, d, r, t_prime)),
    );
    if matches!(inner, BasePair::NN) {
        return None;
    }
    Some((outer, inner))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codon_usage::CodonUsage;
    use crate::plan::{EStorageMode, PlanInputs, TableLayout};
    use rna_mfe::Params;

    fn seeded_plan<'p>(protein: &str, params: &'p Params) -> Plan<'p> {
        let codon_usage = CodonUsage::load_default().unwrap();
        let inputs = PlanInputs {
            dangling_5_prime: false,
            codon_mask: None,
            lambda_: 0.5,
            e_storage_mode: EStorageMode::FullMotifTables,
            table_layout: TableLayout::DenseMaxRectangles,
        };
        let mut plan = Plan::new(protein, &codon_usage, &inputs, params).unwrap();
        // Seed the E table with distinct finite values per flat index so
        // any iteration-order change surfaces as a divergent per-emission
        // `inner_val` sequence (not only as a length difference).
        for (i, slot) in plan.tables.e.iter_mut().enumerate() {
            *slot = f64::from((i as i32) % 100);
        }
        plan
    }

    fn outer_cells(n_aa: usize) -> impl Iterator<Item = (usize, usize, usize, usize)> {
        (0..n_aa).flat_map(move |a| {
            ((a + 2)..n_aa)
                .flat_map(move |b| (0..3).flat_map(move |s| (0..3).map(move |t| (a, b, s, t))))
        })
    }

    fn push_motif_sites_for_outer(
        plan: &Plan<'_>,
        outer: (usize, usize, usize, usize),
        sites: &mut Vec<MotifSite>,
    ) {
        let (a, b, s, t) = outer;
        let (nt_i, nt_j) = (3 * a + s, 3 * b + t);
        if nt_j == 0 || nt_i + 1 >= nt_j - 1 {
            return;
        }
        let nt_pairs = (nt_i, nt_j, nt_i + 1, nt_j - 1);
        for x in 0..plan.codon_lists[a].len() {
            for y in 0..plan.codon_lists[b].len() {
                if let Some(site) = make_site((a, b, s, t, x, y), nt_pairs) {
                    sites.push(site);
                }
            }
        }
    }

    /// Enumerate `MotifSite`s for every legal `(a, b, s, t, x, y)` cell
    /// with the enclosed-pair nts placed canonically immediately inside the
    /// outer pair (`nt_p = nt_i + 1`, `nt_q = nt_j - 1`).
    fn motif_sites(plan: &Plan<'_>) -> Vec<MotifSite> {
        let mut sites = Vec::new();
        for outer in outer_cells(plan.n_aa()) {
            push_motif_sites_for_outer(plan, outer, &mut sites);
        }
        sites
    }

    fn direct_emissions(plan: &Plan<'_>, site: MotifSite) -> Vec<(usize, usize, f64)> {
        let mut emissions = Vec::new();
        iter_two_pair_inner(plan, site, |inner| {
            emissions.push((inner.h, inner.r, inner.inner_val));
        });
        emissions
    }

    /// Reference emission sequence — emulates a materializing `.collect()`
    /// shape (two materializations) over the same nested
    /// [`consistent_choices`] walks the live body uses.
    fn reference_emissions(plan: &Plan<'_>, site: MotifSite) -> Vec<(usize, usize, f64)> {
        let (a, b, _s, _t, x, y) = site.idx;
        let (c, s_prime, d, t_prime) = site.enclosed;
        let base: [(usize, usize); 2] = [(a, x), (b, y)];
        let h_choices: Vec<usize> = consistent_choices(plan, c, &base).collect();
        let mut emissions = Vec::new();
        for h in h_choices {
            let extended: [(usize, usize); 3] = [(a, x), (b, y), (c, h)];
            let r_choices: Vec<usize> = consistent_choices(plan, d, &extended).collect();
            for r in r_choices {
                let inner_idx = (c, d, s_prime, t_prime, h, r);
                let i = plan.tables.flat_index(inner_idx);
                let inner_val = plan.tables.e[i];
                if inner_val < INF {
                    emissions.push((h, r, inner_val));
                }
            }
        }
        emissions
    }

    /// Falsifiability anchor for `iter_two_pair_inner`'s emission order.
    ///
    /// The candidate enumeration at a two-pair site is two nested
    /// [`consistent_choices`] walks (over the enclosed-pair codons `h`
    /// and `r`). Walks the same two-iterator shape with the reference
    /// `.collect()` materializations and with the live
    /// [`iter_two_pair_inner`] body, and asserts the emitted `(h, r,
    /// inner_val)` sequence is byte-identical across a broad sweep of
    /// `MotifSite`s on a small protein. Future refactors that
    /// inadvertently reorder, dedup, or skip emissions will fire here
    /// at the unit level before the per-cell parity suite runs.
    #[test]
    fn iter_two_pair_inner_emission_byte_identical() {
        let params = Params::load_default().unwrap();
        let plan = seeded_plan("GLAGPR", &params);
        let sites = motif_sites(&plan);
        assert!(
            !sites.is_empty(),
            "no MotifSites exercised by the test sweep"
        );
        let mut emissions_seen = 0usize;
        for site in sites {
            let direct = direct_emissions(&plan, site);
            let reference = reference_emissions(&plan, site);
            assert_eq!(
                direct, reference,
                "iter_two_pair_inner emission diverged at site {site:?}",
            );
            emissions_seen += direct.len();
        }
        assert!(
            emissions_seen > 0,
            "no InnerRef emissions observed; test would pass trivially",
        );
    }
}
