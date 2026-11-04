//! The structure axis: what the designed fold must and must not do at each
//! position.
//!
//! One character per nucleotide over the whole mRNA, in the alphabet
//! `RNAfold -C` reads:
//!
//! | character | meaning |
//! |---|---|
//! | `.` | no constraint |
//! | `x` | the position is **unpaired** |
//! | `\|` | the position is **paired**, with any partner |
//! | `<` | the position is paired with a partner **downstream** (`j > i`) |
//! | `>` | the position is paired with a partner **upstream** (`j < i`) |
//! | `(` `)` | the two positions are paired **with each other** |
//!
//! The `<` / `>` directions are ViennaRNA's, and they are the ones that read
//! backwards: `<` names a partner to the *right*. Taking them the other way
//! round produces a fold that is self-consistent and wrong, so the direction is
//! stated here, on the type, and in the run header rather than left in the
//! parser.
//!
//! Every character except `|` also **forbids** the pairs that conflict with it,
//! which is ViennaRNA's rule and the reason the two halves below are not
//! symmetric: `|` contributes a requirement and no prohibition.
//!
//! # Two halves, two mechanisms
//!
//! - The **forbid** half — everything that says *this pair may not form* — is a
//!   verdict on a candidate pair's two endpoint positions, consulted at the
//!   scan's single `P`-write choke point. It is the shipped region constraint with a
//!   positional key instead of a class key ([`PairMask`]).
//! - The **require** half — everything that says *this position must pair* — is
//!   not a property any cell carries. A position is left unpaired by whichever
//!   site *declares* it so, and each such site declares a range. Requiring a
//!   position therefore lands as a bound on those ranges
//!   ([`crate::must_pair::MustPairConstraint`]), never as a filter.
//!
//! Both are predicates on absolute position, so neither needs DP state.
//!
//! # What a required pair claims
//!
//! It binds the **designed** structure — the structure the returned dot-bracket
//! exhibits — and not the molecule. The emitted sequence folded unconstrained may
//! leave a required position unpaired, because "the target is the MFE structure"
//! is a condition on the optimum rather than a term the recurrence can carry.
//! [`crate::free_fold_gap`] is where that difference is measured; nothing here
//! gates on it.

use crate::coords::Layout;
use crate::hash::FxHashSet;

/// The shortest hairpin the folding grammar admits is three unpaired
/// nucleotides, so a pair's endpoints are at least this far apart. A required
/// pair closer than this can never form under any sequence.
const MIN_PAIR_SPAN: u32 = 4;

/// What one position's constraint character says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark {
    /// `.` — no constraint.
    Free,
    /// `x` — the position carries no pair, whatever its partner would have been.
    Unpaired,
    /// `|` — the position carries a pair with some partner. Forbids nothing.
    Paired,
    /// `<` — the position pairs with a partner **downstream** of it (`j > i`),
    /// so it may only be a pair's 5′ endpoint.
    PairsDownstream,
    /// `>` — the position pairs with a partner **upstream** of it (`j < i`), so
    /// it may only be a pair's 3′ endpoint.
    PairsUpstream,
    /// `(` or `)` — the position pairs with exactly the named partner.
    PairsWith(u32),
}

impl Mark {
    /// Whether this mark requires the position to carry a pair.
    fn requires_pair(self) -> bool {
        !matches!(self, Self::Free | Self::Unpaired)
    }

    /// The character this mark is written with. A [`Mark::PairsWith`] is written
    /// as the bracket facing its partner.
    fn as_char(self, at: u32) -> char {
        match self {
            Self::Free => '.',
            Self::Unpaired => 'x',
            Self::Paired => '|',
            Self::PairsDownstream => '<',
            Self::PairsUpstream => '>',
            Self::PairsWith(p) if p > at => '(',
            Self::PairsWith(_) => ')',
        }
    }
}

/// The structure constraints over one mRNA: a mark per position, plus pairs
/// named as forbidden outright.
///
/// A default-constructed value over any length is **inert** — every position is
/// [`Mark::Free`] and no pair is forbidden — so the design is byte-identical to
/// the unconstrained one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructureConstraints {
    marks: Vec<Mark>,
    forbidden: FxHashSet<(u32, u32)>,
}

/// Why a structure constraint set could not be built. Every variant is raised
/// **at parse time**, before any design work, and names the position or pair
/// that has to change.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StructureError {
    /// A character outside the `-C` alphabet.
    #[error("structure character {found:?} at {position} is not one of . x | < > ( )")]
    BadCharacter {
        /// The offending character.
        found: char,
        /// Its position, region-qualified.
        position: String,
    },
    /// The constraint string is not as long as the mRNA. Padding it silently
    /// would make a shortened string mean something different from what it says.
    #[error("structure string is {found} characters but the mRNA is {expected} nt")]
    WrongLength {
        /// Characters supplied.
        found: usize,
        /// Nucleotides in the mRNA.
        expected: u32,
    },
    /// A `(` with no `)` after it, or a `)` with no `(` before it.
    #[error("unbalanced bracket at {position}")]
    UnbalancedBracket {
        /// The bracket's position, region-qualified.
        position: String,
    },
    /// A position carries two constraints that cannot both hold.
    #[error("{position} is constrained {first} and {second} at once")]
    Contradiction {
        /// The position, region-qualified.
        position: String,
        /// The constraint already held there.
        first: String,
        /// The one that contradicts it.
        second: String,
    },
    /// Two required pairs cross. The folding grammar is nested-only, as
    /// ViennaRNA's is, so a pseudoknotted target is refused rather than
    /// silently dropped.
    #[error("required pairs ({a0}, {a1}) and ({b0}, {b1}) cross — the grammar is nested-only")]
    Crossing {
        /// 5′ endpoint of the first pair, region-qualified.
        a0: String,
        /// 3′ endpoint of the first pair.
        a1: String,
        /// 5′ endpoint of the second pair.
        b0: String,
        /// 3′ endpoint of the second pair.
        b1: String,
    },
    /// A required pair whose endpoints are closer than a hairpin loop allows.
    #[error("required pair ({open}, {close}) leaves fewer than 3 unpaired nucleotides between its endpoints, which no structure can do")]
    TooClose {
        /// The 5′ endpoint, region-qualified.
        open: String,
        /// The 3′ endpoint.
        close: String,
    },
    /// A pair is required and forbidden at once.
    #[error("pair ({open}, {close}) is both required and forbidden")]
    RequiredAndForbidden {
        /// The 5′ endpoint, region-qualified.
        open: String,
        /// The 3′ endpoint.
        close: String,
    },
    /// A coordinate lands outside the mRNA.
    #[error("position {0} is outside the mRNA")]
    OutOfRange(u32),
}

impl StructureConstraints {
    /// An inert constraint set over an `mrna_len`-nucleotide mRNA.
    pub fn none(mrna_len: u32) -> Self {
        Self {
            marks: vec![Mark::Free; mrna_len as usize],
            forbidden: FxHashSet::default(),
        }
    }

    /// Parse a whole-mRNA constraint string, one character per nucleotide.
    ///
    /// The string must be exactly as long as the mRNA: a short string is an
    /// error rather than an implied `.`-padding, so what is written is what is
    /// enforced.
    pub fn parse(layout: &Layout, s: &str) -> Result<Self, StructureError> {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() != layout.len() as usize {
            return Err(StructureError::WrongLength {
                found: chars.len(),
                expected: layout.len(),
            });
        }
        let mut sc = Self::none(layout.len());
        let mut open_stack: Vec<u32> = Vec::new();
        for (i, &c) in chars.iter().enumerate() {
            let at = i as u32;
            match c {
                '.' => {}
                'x' => sc.set(layout, at, Mark::Unpaired)?,
                '|' => sc.set(layout, at, Mark::Paired)?,
                '<' => sc.set(layout, at, Mark::PairsDownstream)?,
                '>' => sc.set(layout, at, Mark::PairsUpstream)?,
                '(' => open_stack.push(at),
                ')' => {
                    let open =
                        open_stack
                            .pop()
                            .ok_or_else(|| StructureError::UnbalancedBracket {
                                position: layout.describe(at),
                            })?;
                    sc.require_pair(layout, open, at)?;
                }
                other => {
                    return Err(StructureError::BadCharacter {
                        found: other,
                        position: layout.describe(at),
                    })
                }
            }
        }
        if let Some(&unclosed) = open_stack.first() {
            return Err(StructureError::UnbalancedBracket {
                position: layout.describe(unclosed),
            });
        }
        Ok(sc)
    }

    /// Whether nothing is constrained — the byte-identical floor.
    pub fn is_inert(&self) -> bool {
        self.forbidden.is_empty() && self.marks.iter().all(|m| *m == Mark::Free)
    }

    /// The mark at `pos`, [`Mark::Free`] past the end.
    pub fn mark(&self, pos: u32) -> Mark {
        self.marks.get(pos as usize).copied().unwrap_or(Mark::Free)
    }

    /// This constraint set written back out as a `-C` string.
    pub fn to_string_over(&self, _layout: &Layout) -> String {
        self.marks
            .iter()
            .enumerate()
            .map(|(i, m)| m.as_char(i as u32))
            .collect()
    }

    /// Require `[from, to]` (inclusive, absolute) to carry no pair — the `x`
    /// half, and the general form of a keep-this-window-accessible request.
    pub fn require_unpaired(
        &mut self,
        layout: &Layout,
        from: u32,
        to: u32,
    ) -> Result<(), StructureError> {
        for pos in from..=to {
            self.set(layout, pos, Mark::Unpaired)?;
        }
        Ok(())
    }

    /// Require the pair `(open, close)` — the `( )` half.
    pub fn require_pair(
        &mut self,
        layout: &Layout,
        open: u32,
        close: u32,
    ) -> Result<(), StructureError> {
        let (open, close) = (open.min(close), open.max(close));
        if close.saturating_sub(open) < MIN_PAIR_SPAN {
            return Err(StructureError::TooClose {
                open: layout.describe(open),
                close: layout.describe(close),
            });
        }
        self.set(layout, open, Mark::PairsWith(close))?;
        self.set(layout, close, Mark::PairsWith(open))?;
        self.check_nesting(layout, open, close)
    }

    /// Require the helix pairing `(open + s, close − s)` for `s = 0 … len − 1` —
    /// the sugar the raw string cannot express for an element mid-construct.
    pub fn require_helix(
        &mut self,
        layout: &Layout,
        open: u32,
        close: u32,
        len: u32,
    ) -> Result<(), StructureError> {
        for s in 0..len {
            let (i, j) = (
                open + s,
                close
                    .checked_sub(s)
                    .ok_or(StructureError::OutOfRange(close))?,
            );
            self.require_pair(layout, i, j)?;
        }
        Ok(())
    }

    /// Forbid the pair `(i, j)` without requiring anything of either endpoint —
    /// the negative direction, and the positional key of the shipped class
    /// constraint's verdict.
    pub fn forbid_pair(&mut self, layout: &Layout, i: u32, j: u32) -> Result<(), StructureError> {
        let (i, j) = (i.min(j), i.max(j));
        if j as usize >= self.marks.len() {
            return Err(StructureError::OutOfRange(j));
        }
        if self.mark(i) == Mark::PairsWith(j) {
            return Err(StructureError::RequiredAndForbidden {
                open: layout.describe(i),
                close: layout.describe(j),
            });
        }
        self.forbidden.insert((i, j));
        Ok(())
    }

    /// Record `mark` at `pos`, refusing a mark that contradicts what is held.
    fn set(&mut self, layout: &Layout, pos: u32, mark: Mark) -> Result<(), StructureError> {
        if pos as usize >= self.marks.len() {
            return Err(StructureError::OutOfRange(pos));
        }
        let held = self.marks[pos as usize];
        let merged = merge(held, mark, pos).ok_or_else(|| StructureError::Contradiction {
            position: layout.describe(pos),
            first: describe_mark(held, pos),
            second: describe_mark(mark, pos),
        })?;
        // A `( )` pair may not also be forbidden — the two statements are the
        // contradiction D5 exists to catch, and only this direction is
        // reachable here (the other is checked as the forbid is recorded).
        if let Mark::PairsWith(partner) = merged {
            let key = (pos.min(partner), pos.max(partner));
            if self.forbidden.contains(&key) {
                return Err(StructureError::RequiredAndForbidden {
                    open: layout.describe(key.0),
                    close: layout.describe(key.1),
                });
            }
        }
        self.marks[pos as usize] = merged;
        Ok(())
    }

    /// Refuse a required pair that crosses one already required. The grammar is
    /// nested-only, so a crossing set has no structure at all — reported here
    /// rather than as an empty scan the caller cannot diagnose.
    fn check_nesting(&self, layout: &Layout, open: u32, close: u32) -> Result<(), StructureError> {
        for (i, mark) in self.marks.iter().enumerate() {
            let Mark::PairsWith(partner) = *mark else {
                continue;
            };
            let (a, b) = (i as u32, partner);
            if a >= b || (a, b) == (open, close) {
                continue;
            }
            let strictly_inside = |p: u32| p > open && p < close;
            if strictly_inside(a) != strictly_inside(b) {
                return Err(StructureError::Crossing {
                    a0: layout.describe(a),
                    a1: layout.describe(b),
                    b0: layout.describe(open),
                    b1: layout.describe(close),
                });
            }
        }
        Ok(())
    }

    /// The positions required to carry a pair — the require half, as the
    /// absolute position list [`crate::must_pair::MustPairConstraint`] resolves.
    ///
    /// Public because a report on what a run constrained is read off this
    /// rather than re-derived from the caller's own flags.
    pub fn required_positions(&self) -> Vec<u32> {
        self.marks
            .iter()
            .enumerate()
            .filter(|(_, m)| m.requires_pair())
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// The pairs required outright, 5′ endpoint first — what the pre-check tests
    /// for spellability and what a realization verdict later reports on.
    pub fn required_pairs(&self) -> Vec<(u32, u32)> {
        self.marks
            .iter()
            .enumerate()
            .filter_map(|(i, m)| match *m {
                Mark::PairsWith(p) if p > i as u32 => Some((i as u32, p)),
                _ => None,
            })
            .collect()
    }

    /// The forbid half, lowered to the positional verdict the `P`-write choke
    /// point consults.
    pub(crate) fn pair_mask(&self) -> PairMask {
        let sites = self
            .marks
            .iter()
            .map(|m| match *m {
                Mark::Free | Mark::Paired => Site::OPEN,
                Mark::Unpaired => Site {
                    forbid_all: true,
                    ..Site::OPEN
                },
                // `<` names a partner downstream, so the position may only be a
                // pair's 5′ endpoint — it may not close one.
                Mark::PairsDownstream => Site {
                    forbid_as_close: true,
                    ..Site::OPEN
                },
                Mark::PairsUpstream => Site {
                    forbid_as_open: true,
                    ..Site::OPEN
                },
                Mark::PairsWith(p) => Site {
                    only_partner: Some(p),
                    ..Site::OPEN
                },
            })
            .collect::<Vec<_>>();
        let active = sites.iter().any(|s| *s != Site::OPEN) || !self.forbidden.is_empty();
        PairMask {
            sites,
            forbidden: self.forbidden.clone(),
            active,
        }
    }
}

/// What one position forbids of a candidate pair. [`Site::OPEN`] forbids
/// nothing, which is what makes an inert mask byte-identical to no mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Site {
    /// `x`: no pair may have an endpoint here.
    forbid_all: bool,
    /// The position may not be a pair's 5′ endpoint.
    forbid_as_open: bool,
    /// The position may not be a pair's 3′ endpoint.
    forbid_as_close: bool,
    /// The only partner this position may take.
    only_partner: Option<u32>,
}

impl Site {
    const OPEN: Self = Self {
        forbid_all: false,
        forbid_as_open: false,
        forbid_as_close: false,
        only_partner: None,
    };
}

/// The positional pair mask, resolved against a concrete mRNA: an `O(1)`
/// verdict on a candidate pair's two endpoint positions.
///
/// Structurally the shipped region constraint with a positional key instead of a
/// class key, and consulted at the same single `P`-write choke point. A mask
/// built from an inert constraint set answers "allowed" for every pair.
#[derive(Debug, Clone, Default)]
pub(crate) struct PairMask {
    sites: Vec<Site>,
    forbidden: FxHashSet<(u32, u32)>,
    active: bool,
}

impl PairMask {
    /// Whether the mask forbids nothing at all.
    pub(crate) fn is_inert(&self) -> bool {
        !self.active
    }

    /// Whether a pair with 5′ endpoint `i` and 3′ endpoint `j` may form.
    ///
    /// The predicate is a claim about each **position**, not about a span: an
    /// `x` at `i` forbids every pair with an endpoint there whatever its
    /// partner, including one whose partner lies outside any constrained
    /// element. That is what makes "keep this window unpaired" mean what it
    /// says when the window's would-be partner is elsewhere in the molecule.
    pub(crate) fn allows(&self, i: u32, j: u32) -> bool {
        if !self.active {
            return true;
        }
        let (open, close) = (i.min(j), i.max(j));
        let (a, b) = (self.site(open), self.site(close));
        if a.forbid_all || b.forbid_all || a.forbid_as_open || b.forbid_as_close {
            return false;
        }
        if a.only_partner.is_some_and(|p| p != close) || b.only_partner.is_some_and(|p| p != open) {
            return false;
        }
        !self.forbidden.contains(&(open, close))
    }

    fn site(&self, pos: u32) -> Site {
        self.sites.get(pos as usize).copied().unwrap_or(Site::OPEN)
    }
}

/// The two marks a position can hold at once, or `None` when they contradict.
///
/// The merges that succeed are the ones where one mark is strictly stronger:
/// `|` is subsumed by every other requirement, a direction agrees with a
/// concrete partner that lies that way, and identical marks are idempotent.
fn merge(held: Mark, new: Mark, at: u32) -> Option<Mark> {
    use Mark::*;
    match (held, new) {
        (Free, m) | (m, Free) => Some(m),
        (a, b) if a == b => Some(a),
        (Unpaired, _) | (_, Unpaired) => None,
        (Paired, m) | (m, Paired) => Some(m),
        (PairsWith(p), PairsDownstream) | (PairsDownstream, PairsWith(p)) => {
            (p > at).then_some(PairsWith(p))
        }
        (PairsWith(p), PairsUpstream) | (PairsUpstream, PairsWith(p)) => {
            (p < at).then_some(PairsWith(p))
        }
        // Two different concrete partners, or two opposing directions.
        _ => None,
    }
}

/// How a mark reads in an error message.
fn describe_mark(mark: Mark, at: u32) -> String {
    match mark {
        Mark::Free => "unconstrained".into(),
        Mark::Unpaired => "unpaired (x)".into(),
        Mark::Paired => "paired (|)".into(),
        Mark::PairsDownstream => "paired downstream (<)".into(),
        Mark::PairsUpstream => "paired upstream (>)".into(),
        Mark::PairsWith(p) => format!("paired with position {} ({})", p + 1, mark.as_char(at)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(n: u32) -> Layout {
        Layout::new(0, n / 3, 0)
    }

    #[test]
    fn an_inert_set_forbids_nothing_and_requires_nothing() {
        let l = layout(30);
        let sc = StructureConstraints::none(l.len());
        assert!(sc.is_inert());
        assert!(sc.required_positions().is_empty());
        let mask = sc.pair_mask();
        assert!(mask.is_inert());
        for i in 0..30 {
            for j in i + 1..30 {
                assert!(mask.allows(i, j));
            }
        }
    }

    #[test]
    fn the_alphabet_round_trips_through_parse_and_back() {
        let l = layout(30);
        let s = "..((((....))))..xx||<....>....";
        let sc = StructureConstraints::parse(&l, s).expect("valid");
        assert_eq!(sc.to_string_over(&l), s);
    }

    /// The direction that reads backwards, pinned as a test rather than as a
    /// comment: `<` names a partner **downstream**, so the position may open a
    /// pair and may not close one.
    #[test]
    fn angle_brackets_take_viennarnas_directions() {
        let l = layout(30);
        let sc = StructureConstraints::parse(&l, &format!("{}<{}", ".".repeat(10), ".".repeat(19)))
            .expect("valid");
        let mask = sc.pair_mask();
        assert!(mask.allows(10, 20), "10 may pair with a downstream partner");
        assert!(!mask.allows(2, 10), "10 may not close a pair");
        assert_eq!(sc.required_positions(), vec![10]);

        let sc = StructureConstraints::parse(&l, &format!("{}>{}", ".".repeat(10), ".".repeat(19)))
            .expect("valid");
        let mask = sc.pair_mask();
        assert!(mask.allows(2, 10), "10 may pair with an upstream partner");
        assert!(!mask.allows(10, 20), "10 may not open a pair");
    }

    /// `x` is a claim about the position: every pair with an endpoint there is
    /// forbidden, including one whose partner is nowhere near the constraint.
    #[test]
    fn x_forbids_every_pair_touching_the_position() {
        let l = layout(30);
        let mut sc = StructureConstraints::none(l.len());
        sc.require_unpaired(&l, 12, 14).unwrap();
        let mask = sc.pair_mask();
        for k in 12..=14 {
            for other in 0..30u32 {
                if other.abs_diff(k) >= MIN_PAIR_SPAN {
                    assert!(!mask.allows(k.min(other), k.max(other)), "({k}, {other})");
                }
            }
        }
        assert!(
            mask.allows(0, 20),
            "positions outside the window are untouched"
        );
        assert!(sc.required_positions().is_empty(), "x requires no pairing");
    }

    /// `|` requires pairing and forbids nothing — ViennaRNA's one exception to
    /// "constraints disallow all conflicting pairs".
    #[test]
    fn a_bar_requires_pairing_without_forbidding_anything() {
        let l = layout(30);
        let sc = StructureConstraints::parse(&l, &format!("{}|{}", ".".repeat(7), ".".repeat(22)))
            .expect("valid");
        assert_eq!(sc.required_positions(), vec![7]);
        assert!(sc.pair_mask().is_inert());
    }

    #[test]
    fn a_named_pair_forbids_every_other_partner_for_both_endpoints() {
        let l = layout(30);
        let mut sc = StructureConstraints::none(l.len());
        sc.require_pair(&l, 5, 20).unwrap();
        let mask = sc.pair_mask();
        assert!(mask.allows(5, 20));
        assert!(!mask.allows(5, 21));
        assert!(!mask.allows(0, 20));
        assert_eq!(sc.required_pairs(), vec![(5, 20)]);
        assert_eq!(sc.required_positions(), vec![5, 20]);
    }

    #[test]
    fn a_helix_is_the_pair_sugar_applied_along_its_length() {
        let l = layout(60);
        let mut sc = StructureConstraints::none(l.len());
        sc.require_helix(&l, 10, 40, 4).unwrap();
        assert_eq!(
            sc.required_pairs(),
            vec![(10, 40), (11, 39), (12, 38), (13, 37)]
        );
    }

    #[test]
    fn contradictions_are_refused_at_parse_time() {
        let l = layout(30);
        let mut sc = StructureConstraints::none(l.len());
        sc.require_unpaired(&l, 5, 5).unwrap();
        assert!(matches!(
            sc.require_pair(&l, 5, 20),
            Err(StructureError::Contradiction { .. })
        ));

        let mut sc = StructureConstraints::none(l.len());
        sc.require_pair(&l, 5, 20).unwrap();
        assert!(matches!(
            sc.require_pair(&l, 5, 21),
            Err(StructureError::Contradiction { .. })
        ));
    }

    /// A direction and a concrete partner agree when the partner lies that way
    /// and contradict when it does not — the merge that is not simply "equal or
    /// bust".
    #[test]
    fn a_direction_and_a_concrete_partner_must_agree() {
        let l = layout(30);
        let mut sc = StructureConstraints::none(l.len());
        sc.set(&l, 5, Mark::PairsDownstream).unwrap();
        sc.require_pair(&l, 5, 20).expect("20 is downstream of 5");

        let mut sc = StructureConstraints::none(l.len());
        sc.set(&l, 5, Mark::PairsUpstream).unwrap();
        assert!(matches!(
            sc.require_pair(&l, 5, 20),
            Err(StructureError::Contradiction { .. })
        ));
    }

    #[test]
    fn crossing_pairs_are_refused_rather_than_dropped() {
        let l = layout(60);
        let mut sc = StructureConstraints::none(l.len());
        sc.require_pair(&l, 5, 25).unwrap();
        assert!(matches!(
            sc.require_pair(&l, 15, 40),
            Err(StructureError::Crossing { .. })
        ));
        // Nested and disjoint pairs are both fine.
        let mut sc = StructureConstraints::none(l.len());
        sc.require_pair(&l, 5, 25).unwrap();
        sc.require_pair(&l, 10, 20).expect("nested");
        sc.require_pair(&l, 30, 50).expect("disjoint");
    }

    #[test]
    fn a_pair_too_close_to_close_a_hairpin_is_refused() {
        let l = layout(30);
        let mut sc = StructureConstraints::none(l.len());
        assert!(matches!(
            sc.require_pair(&l, 5, 8),
            Err(StructureError::TooClose { .. })
        ));
        sc.require_pair(&l, 5, 9)
            .expect("exactly three unpaired between");
    }

    #[test]
    fn requiring_and_forbidding_one_pair_is_refused_from_either_side() {
        let l = layout(30);
        let mut sc = StructureConstraints::none(l.len());
        sc.require_pair(&l, 5, 20).unwrap();
        assert!(matches!(
            sc.forbid_pair(&l, 5, 20),
            Err(StructureError::RequiredAndForbidden { .. })
        ));

        let mut sc = StructureConstraints::none(l.len());
        sc.forbid_pair(&l, 5, 20).unwrap();
        assert!(matches!(
            sc.require_pair(&l, 5, 20),
            Err(StructureError::RequiredAndForbidden { .. })
        ));
    }

    #[test]
    fn a_forbidden_pair_stops_exactly_that_pair() {
        let l = layout(30);
        let mut sc = StructureConstraints::none(l.len());
        sc.forbid_pair(&l, 5, 20).unwrap();
        let mask = sc.pair_mask();
        assert!(!mask.allows(5, 20));
        assert!(mask.allows(5, 21));
        assert!(mask.allows(4, 20));
        assert!(sc.required_positions().is_empty());
    }

    #[test]
    fn malformed_strings_are_refused_with_their_position() {
        let l = layout(30);
        assert!(matches!(
            StructureConstraints::parse(&l, &".".repeat(29)),
            Err(StructureError::WrongLength { found: 29, .. })
        ));
        assert!(matches!(
            StructureConstraints::parse(&l, &format!("{}Z{}", ".".repeat(4), ".".repeat(25))),
            Err(StructureError::BadCharacter { found: 'Z', .. })
        ));
        assert!(matches!(
            StructureConstraints::parse(&l, &format!("{}({}", ".".repeat(4), ".".repeat(25))),
            Err(StructureError::UnbalancedBracket { .. })
        ));
        assert!(matches!(
            StructureConstraints::parse(&l, &format!("{}){}", ".".repeat(4), ".".repeat(25))),
            Err(StructureError::UnbalancedBracket { .. })
        ));
    }
}
