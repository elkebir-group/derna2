//! A non-empty subset of `{A, C, G, U}` — what one designed flank position is
//! allowed to spell.
//!
//! # Why a subset is a constraint on the molecule
//!
//! A structure-space constraint restricts the structure a design is *scored
//! against*, so the design can be the exact optimum of its constrained problem
//! and still fold differently once synthesized. A constraint on the **sequence**
//! escapes that: the fold stays unrestricted, so whatever sequence survives, the
//! returned structure is its true minimum.
//!
//! The subsets that matter here are the ones no two members of which can pair.
//! Over the canonical pairs — including the wobble `G·U` — the mutually
//! non-pairing subsets of size two are exactly `{A, C}`, `{A, G}` and `{C, U}`,
//! and **two is the maximum**: any third letter reintroduces a pair. Holding
//! two flanks over one such subset makes every flank-to-flank helix unspellable,
//! which is a property of the molecule rather than of the search.
//!
//! # The spelling
//!
//! Sets are written as **IUPAC** codes, the same grammar the forbidden-motif
//! automaton reads: `A`/`C`/`G`/`U` are the singletons, `M` is `{A, C}`, `R` is
//! `{A, G}`, `Y` is `{C, U}`, `N` is the full alphabet. A singleton set is a
//! given position — one out-edge, nothing to choose — and the full set is a
//! freely designed one, so the two ends of the existing pinned-or-free axis are
//! both ordinary members of this one.

use rna_mfe::Nucleotide;

/// A non-empty subset of `{A, C, G, U}`, held as a four-bit mask keyed by
/// [`Nucleotide`]'s discriminant.
///
/// Construction guarantees non-emptiness: a position that can spell nothing has
/// no out-edge, which would silently disconnect the graph rather than report an
/// infeasible constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NucSet(u8);

impl NucSet {
    /// The full alphabet — a freely designed position.
    pub const ALL: Self = Self(0b1111);

    /// The singleton `{nt}` — a given position.
    pub fn of(nt: Nucleotide) -> Self {
        Self(1 << nt as u8)
    }

    /// The set of the listed nucleotides, or `None` when the list is empty.
    /// Duplicates are absorbed.
    pub fn from_nucleotides(nts: &[Nucleotide]) -> Option<Self> {
        let mask = nts.iter().fold(0u8, |m, nt| m | 1 << *nt as u8);
        (mask != 0).then_some(Self(mask))
    }

    /// The set an IUPAC code denotes (`T` read as `U`), or `None` for any byte
    /// that is not one.
    pub fn from_iupac(code: u8) -> Option<Self> {
        Self::from_nucleotides(crate::ac::iupac_class(code))
    }

    /// Whether `nt` may be spelled here.
    pub fn contains(self, nt: Nucleotide) -> bool {
        self.0 & 1 << nt as u8 != 0
    }

    /// How many nucleotides the position may spell — the node's out-degree.
    /// Never zero; the type has no empty value, which is why this is `size`
    /// rather than a `len` implying an `is_empty` that could never be true.
    pub fn size(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Whether the position has no choice to make. Such a position is a given
    /// one: one out-edge, no preference to price, and no re-normalization to
    /// account for.
    pub fn is_pinned(self) -> bool {
        self.size() == 1
    }

    /// Whether every nucleotide is available — the freely designed position the
    /// full-alphabet byte-identity floor turns on.
    pub fn is_full(self) -> bool {
        self == Self::ALL
    }

    /// The set's members in `A`, `C`, `G`, `U` order.
    pub fn iter(self) -> impl Iterator<Item = Nucleotide> {
        [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U]
            .into_iter()
            .filter(move |nt| self.contains(*nt))
    }

    /// Whether any two members — including a member with itself — can form a
    /// canonical pair, wobble included. A set that answers `false` cannot spell
    /// a helix against another position drawn from the same set, which is the
    /// property the flank restriction exists for.
    pub fn is_self_pairing(self) -> bool {
        self.iter().any(|a| {
            self.iter().any(|b| {
                rna_mfe::loops::pair_from_nucleotides(a, b) != rna_mfe::params::BasePair::NN
            })
        })
    }
}

#[cfg(test)]
mod tests {
    //! The subset algebra, and the claim the scope rests on: `{A,C}`, `{A,G}`
    //! and `{C,U}` are exactly the mutually non-pairing pairs, and no larger set
    //! is non-pairing.

    use super::*;
    use rna_mfe::Nucleotide::{A, C, G, U};

    #[test]
    fn an_empty_list_has_no_set() {
        assert_eq!(NucSet::from_nucleotides(&[]), None);
    }

    #[test]
    fn the_ends_of_the_axis_are_ordinary_members() {
        assert!(NucSet::of(G).is_pinned());
        assert_eq!(NucSet::of(G).iter().collect::<Vec<_>>(), vec![G]);
        assert!(NucSet::ALL.is_full());
        assert_eq!(NucSet::ALL.size(), 4);
        assert_eq!(NucSet::ALL.iter().collect::<Vec<_>>(), vec![A, C, G, U]);
    }

    #[test]
    fn iupac_codes_denote_the_expected_sets() {
        assert_eq!(NucSet::from_iupac(b'N'), Some(NucSet::ALL));
        assert_eq!(NucSet::from_iupac(b'M'), NucSet::from_nucleotides(&[A, C]));
        assert_eq!(NucSet::from_iupac(b'R'), NucSet::from_nucleotides(&[A, G]));
        assert_eq!(NucSet::from_iupac(b'Y'), NucSet::from_nucleotides(&[C, U]));
        assert_eq!(NucSet::from_iupac(b'T'), Some(NucSet::of(U)));
        assert_eq!(NucSet::from_iupac(b'Z'), None);
    }

    #[test]
    fn exactly_three_two_letter_sets_cannot_pair_with_themselves() {
        let letters = [A, C, G, U];
        let mut non_pairing = Vec::new();
        for (i, a) in letters.iter().enumerate() {
            for b in &letters[i + 1..] {
                let set = NucSet::from_nucleotides(&[*a, *b]).expect("two letters");
                if !set.is_self_pairing() {
                    non_pairing.push(set);
                }
            }
        }
        let expected = [[A, C], [A, G], [C, U]]
            .map(|s| NucSet::from_nucleotides(&s).expect("two letters"))
            .to_vec();
        assert_eq!(non_pairing, expected);
    }

    #[test]
    fn no_three_letter_set_avoids_pairing() {
        for triple in [[A, C, G], [A, C, U], [A, G, U], [C, G, U]] {
            let set = NucSet::from_nucleotides(&triple).expect("three letters");
            assert!(
                set.is_self_pairing(),
                "{triple:?} should reintroduce a pair"
            );
        }
        assert!(NucSet::ALL.is_self_pairing());
    }

    #[test]
    fn a_singleton_pairs_with_itself_only_where_the_model_allows() {
        // No nucleotide pairs with a copy of itself under the canonical set, so
        // a pinned position is trivially non-self-pairing — which is true and
        // uninformative, since one position spells no helix on its own.
        for nt in [A, C, G, U] {
            assert!(!NucSet::of(nt).is_self_pairing());
        }
    }
}
