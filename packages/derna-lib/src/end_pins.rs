//! Pinning an item's end windows: per-residue codon menus that fix the first
//! and last residues to a given item's own codons and leave every other residue
//! its full synonymous menu.
//!
//! A scan over the lattice these menus build ([`crate::dfa::build_composite`]
//! with `Some(menus)`) optimizes the free middle with the ends held to the
//! item's choices. Restricting a residue to one codon is the one-choice corner
//! of the restricted lattice, so nothing here touches the scan: the menus are
//! the whole mechanism. The restricted lattice moves each residue's best
//! available adaptation out of its edges, so a reported weight is the scan's
//! own plus [`crate::dfa::Dfa::codon_offset`].

use derna_codon_dp::codon_table::AMINO_ACIDS;

/// Why an item cannot pin a protein's ends.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EndPinError {
    /// The item is not three symbols per residue.
    #[error("item has {got} symbols, the protein needs {want}")]
    Length {
        /// Symbols in the item.
        got: usize,
        /// Three per residue.
        want: usize,
    },
    /// A pinned residue's codon in the item is not one of its synonymous codons.
    #[error("residue {residue}: codon {codon} is not on its menu")]
    OffMenu {
        /// Zero-based residue index.
        residue: usize,
        /// The item's three symbols there.
        codon: String,
    },
}

/// The codon menus pinning residues `0..w5` and `n - w3..n` of `protein`
/// (DERNA amino-acid indices) to `item`'s codons (`A/C/G/U`, three per
/// residue); every other residue keeps all its synonymous codons. Windows
/// longer than the protein, or overlapping, pin every residue they cover.
pub fn end_pinned_menus(
    protein: &[usize],
    item: &str,
    w5: usize,
    w3: usize,
) -> Result<Vec<Vec<[u8; 3]>>, EndPinError> {
    let n = protein.len();
    if item.len() != 3 * n {
        return Err(EndPinError::Length {
            got: item.len(),
            want: 3 * n,
        });
    }
    let tail = n.saturating_sub(w3);
    protein
        .iter()
        .enumerate()
        .map(|(i, &aa)| {
            let codons = AMINO_ACIDS[aa].codons;
            if i >= w5 && i < tail {
                return Ok(codons.iter().map(|c| c.nuc).collect());
            }
            let spelled = &item[3 * i..3 * i + 3];
            codons
                .iter()
                .find(|c| c.name == spelled)
                .map(|c| vec![c.nuc])
                .ok_or_else(|| EndPinError::OffMenu {
                    residue: i,
                    codon: spelled.to_string(),
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dfa::{self, Dfa, UtrChain};
    use crate::scan;
    use crate::state::State;
    use crate::traceback;
    use crate::{CodonUsage, Params, ScanConfig, UtrProfile};
    use derna_codon_dp::codon_table::aa_to_index;

    const PROTEIN: &str = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPD";

    struct Fixture {
        params: Params,
        usage: CodonUsage,
        aa: Vec<usize>,
    }

    fn fixture() -> Fixture {
        Fixture {
            params: Params::load_default().expect("params"),
            usage: CodonUsage::load_default().expect("usage"),
            aa: PROTEIN
                .chars()
                .map(|c| aa_to_index(c).expect("residue"))
                .collect(),
        }
    }

    fn restricted(f: &Fixture, menus: &[Vec<[u8; 3]>]) -> Dfa {
        dfa::build_composite(
            &f.aa,
            &f.usage,
            Some(menus),
            UtrChain::Fixed(&[]),
            UtrChain::Fixed(&[]),
            UtrProfile::inert(),
        )
        .expect("every residue keeps a codon")
    }

    /// The optimum at `lambda` with its weight restored, and its item.
    fn solve(f: &Fixture, graph: &Dfa, lambda: f64) -> (State, String) {
        let tables = scan::scan(&f.params, graph, lambda, ScanConfig::default());
        let mut best = tables.c[graph.ix(graph.end())].expect("a parse");
        best.cai += graph.codon_offset();
        (best, traceback::recover(&f.params, graph, &tables).sequence)
    }

    fn weight(f: &Fixture, item: &str) -> f64 {
        let menus = end_pinned_menus(&f.aa, item, f.aa.len(), 0).expect("on the lattice");
        restricted(f, &menus).codon_offset()
    }

    #[test]
    fn pinning_every_residue_returns_the_item_and_its_reference_fold() {
        let f = fixture();
        let (_, item) = solve(&f, &dfa::build(&f.aa, &f.usage), 0.3);
        let graph = restricted(&f, &end_pinned_menus(&f.aa, &item, f.aa.len(), 0).unwrap());
        let (best, got) = solve(&f, &graph, 0.7);
        assert_eq!(got, item);
        let (_, reference) = rna_mfe::fold(&f.params, item.as_bytes()).expect("folds");
        assert_eq!(best.mfe, reference);
        assert!((best.cai - weight(&f, &item)).abs() < 1e-9);
    }

    #[test]
    fn pinning_nothing_reproduces_the_unpinned_scan() {
        let f = fixture();
        let any = "A".repeat(3 * f.aa.len());
        let menus = end_pinned_menus(&f.aa, &any, 0, 0).unwrap();
        for lambda in [0.1, 0.5, 0.9] {
            let (free, free_item) = solve(&f, &dfa::build(&f.aa, &f.usage), lambda);
            let (pinned, pinned_item) = solve(&f, &restricted(&f, &menus), lambda);
            assert_eq!(pinned.mfe, free.mfe);
            assert!((pinned.cai - free.cai).abs() < 1e-9);
            assert_eq!(pinned_item, free_item);
        }
    }

    #[test]
    fn pinned_ends_keep_the_items_ends_and_never_beat_the_free_optimum() {
        let f = fixture();
        let (w5, w3) = (6, 5);
        let (_, base) = solve(&f, &dfa::build(&f.aa, &f.usage), 0.9);
        let item = with_other_end_codons(&f.aa, &base, w5, w3);
        let graph = restricted(&f, &end_pinned_menus(&f.aa, &item, w5, w3).unwrap());
        let lambda = 0.2;
        let (free, free_item) = solve(&f, &dfa::build(&f.aa, &f.usage), lambda);
        let (pinned, got) = solve(&f, &graph, lambda);
        let n3 = item.len();
        assert_eq!(got[..3 * w5], item[..3 * w5]);
        assert_eq!(got[n3 - 3 * w3..], item[n3 - 3 * w3..]);
        assert!(pinned.brd(lambda) >= free.brd(lambda) - 1e-6);
        assert!((pinned.cai - weight(&f, &got)).abs() < 1e-9);
        // The free optimum does not spell the pinned ends, so the pin binds.
        assert_ne!(free_item[..3 * w5], item[..3 * w5]);
        assert!(pinned.brd(lambda) > free.brd(lambda));
    }

    /// `item` with each residue in the two end windows moved to a different
    /// synonymous codon where it has one.
    fn with_other_end_codons(aa: &[usize], item: &str, w5: usize, w3: usize) -> String {
        let n = aa.len();
        let mut out = item.to_string();
        for i in (0..w5).chain(n - w3..n) {
            let here = &item[3 * i..3 * i + 3];
            if let Some(c) = AMINO_ACIDS[aa[i]].codons.iter().find(|c| c.name != here) {
                out.replace_range(3 * i..3 * i + 3, c.name);
            }
        }
        out
    }

    #[test]
    fn an_off_menu_codon_is_refused_by_residue() {
        let f = fixture();
        let bad = "UUU".repeat(f.aa.len());
        assert_eq!(
            end_pinned_menus(&f.aa, &bad, 1, 0),
            Err(EndPinError::OffMenu {
                residue: 0,
                codon: "UUU".into()
            })
        );
        assert!(matches!(
            end_pinned_menus(&f.aa, "AUG", 1, 0),
            Err(EndPinError::Length { .. })
        ));
    }
}
