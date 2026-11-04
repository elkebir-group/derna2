//! CKY-table state and the BRD scalarization the scan optimizes.
//!
//! Every partial parse carries two separate accumulators — minimum free
//! energy (integer dekacalories, exact) and log-CAI (`f64`, summed
//! cost-to-go edge weights) — plus a [`Back`] pointer that lets the
//! traceback rebuild the chosen sequence and structure without re-running
//! the recurrence. Equivalent derivations reaching the same table cell
//! merge, keeping the one with the lowest combined BRD score.

use rna_mfe::{Energy, Nucleotide};

use crate::dfa::Node;

/// A closed-pair table key: the 5' boundary node together with the
/// boundary nucleotides of the pair. The 3' boundary is the table's own
/// outer index, so it need not be stored here. This is the DFA analogue
/// of the interval DP's endpoint codon choice, recorded as the two
/// boundary nucleotides: a flank codon that affects only the loop energy and
/// the CAI is recovered at traceback from those nucleotides, so the key need
/// not carry the codon itself.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PairKey {
    /// 5' boundary node (before the opening nucleotide).
    pub open: Node,
    /// 3' boundary node (the closing nucleotide's own node, before its
    /// edge). Distinguishes the parallel lanes at the 3' position so the
    /// loop interior is reconstructed identically by the scan and the
    /// traceback — the node-pair half of the closed-stem key.
    pub close: Node,
    /// The opening edge's nucleotide and the lane it lands in.
    pub opening: Opening,
    /// Nucleotide at the pair's 3' slot.
    pub nuc_close: Nucleotide,
}

/// An opening key `key` — a node and the nucleotide its edge spells —
/// refined by the lane `lane` the edge lands in: `key * 2 + lane % 2`.
///
/// Two edges out of one node that spell one nucleotide differ only at a stop
/// unit's first node, whose `U` edges land in lanes 0 and 1; every other
/// `(node, nucleotide)` has one edge and so one lane. The low lane bit therefore
/// tells apart every pair of edges a key could confuse, and lanes 0 and 1 keep
/// the value a lattice without a third lane has always used.
#[inline]
pub(crate) fn lane_key(key: usize, lane: u8) -> usize {
    key * 2 + usize::from(lane & 1)
}

/// An opening edge's nucleotide and the lane of the node it lands in, in one
/// byte. A designed stop codon's first node has two out-edges spelling `U`
/// (the `UA` and `UG` lanes), so the open node and nucleotide alone do not
/// name the opening edge there; the closing edge is named by the table's
/// outer index.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Opening(u8);

impl Opening {
    /// The opening spelling `nuc` into lane `lane` (0 to 3).
    pub fn new(nuc: Nucleotide, lane: u8) -> Self {
        debug_assert!(lane < 4, "a lane fits two bits");
        Self(nuc as u8 | lane << 2)
    }

    /// The nucleotide the opening edge spells.
    pub fn nuc(self) -> Nucleotide {
        match self.0 & 3 {
            0 => Nucleotide::A,
            1 => Nucleotide::C,
            2 => Nucleotide::G,
            _ => Nucleotide::U,
        }
    }

    /// The lane of the node the opening edge lands in.
    pub fn lane(self) -> u8 {
        self.0 >> 2
    }
}

impl std::fmt::Debug for Opening {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}/{}", self.nuc(), self.lane())
    }
}

/// Field by field as the key reads, the lane last: the lazy outside index
/// orders cells as their `Debug` strings.
impl std::fmt::Debug for PairKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PairKey")
            .field("open", &{ self.open })
            .field("close", &{ self.close })
            .field("nuc_open", &self.nuc_open())
            .field("nuc_close", &self.nuc_close)
            .field("lane_open", &self.lane_open())
            .finish()
    }
}

/// The lane is left out of the hash: everywhere but a designed stop's first
/// node it follows from `open` and the nucleotide, so hashing it would only
/// reorder every table a key indexes, and the two lanes there share a bucket.
impl std::hash::Hash for PairKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.open.hash(state);
        self.close.hash(state);
        (self.nuc_open() as u8).hash(state);
        (self.nuc_close as u8).hash(state);
    }
}

impl PairKey {
    /// Nucleotide at the pair's 5' slot.
    pub fn nuc_open(&self) -> Nucleotide {
        self.opening.nuc()
    }

    /// Lane of the node the opening edge lands in.
    pub fn lane_open(&self) -> u8 {
        self.opening.lane()
    }

    /// The edge that opens the pair: out of `open`, spelling `nuc_open`, into
    /// lane `lane_open`.
    pub fn open_edge(&self, dfa: &crate::dfa::Dfa) -> Option<crate::dfa::Edge> {
        dfa.right(self.open)
            .iter()
            .find(|e| e.nuc == self.nuc_open() && e.to.lane == self.lane_open())
            .copied()
    }

    /// The edge that closes the pair into `outer`: out of `close`, spelling
    /// `nuc_close`.
    pub fn close_edge(&self, dfa: &crate::dfa::Dfa, outer: Node) -> Option<crate::dfa::Edge> {
        dfa.right(self.close)
            .iter()
            .find(|e| e.nuc == self.nuc_close && e.to == outer)
            .copied()
    }
}

// Closed-pair keys are most of what the tables hash and store, so their width
// is held here: two 5-byte nodes, the opening, and the closing nucleotide.
const _: () = assert!(std::mem::size_of::<PairKey>() == 12);

/// Key for the in-progress multi-loop table `multi_open`: the multi-loop's
/// 5' opening node and the nucleotide on its opening edge. The 3' frontier
/// is supplied at closing time (it is not part of the key — one opening is
/// shared across every closing frontier), and the closing nucleotide is
/// chosen when the cell is closed into `Multi`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiOpenKey {
    /// 5' opening node of the multi-loop.
    pub open: Node,
    /// Nucleotide on the multi-loop's opening edge.
    pub nuc_open: Nucleotide,
    /// Lane of the node the opening edge lands in, as in [`Opening`].
    pub lane_open: u8,
    /// The stops the opening's core can still reach ([`crate::dfa::Dfa::reach_class`]):
    /// `0` on every lattice but a stop-labelled one. Openings whose cores
    /// reach different stops are kept apart, since a frontier on one stop's
    /// lane closes only the openings that reach it. Not hashed, so a lattice
    /// that never sets it hashes every key as before.
    pub reach: u8,
}

impl std::hash::Hash for MultiOpenKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.open.hash(state);
        (self.nuc_open as u8).hash(state);
    }
}

/// A cell's two accumulators without its provenance: what a closure reads of
/// the inner pair it builds on, whose own back-pointer names that pair rather
/// than copying the pair's.
#[derive(Debug, Clone, Copy)]
pub struct Value {
    /// Accumulated free energy (integer dekacalories).
    pub mfe: Energy,
    /// Accumulated log-CAI (natural log).
    pub cai: f64,
}

impl Value {
    /// [`State::brd`] on the accumulators alone.
    pub fn brd(&self, lambda: f64) -> f64 {
        lambda * f64::from(self.mfe.0) - (1.0 - lambda) * (self.cai * 100.0)
    }
}

impl From<State> for Value {
    fn from(st: State) -> Self {
        Self {
            mfe: st.mfe,
            cai: st.cai,
        }
    }
}

/// One CKY table cell: the best partial parse for its `(table, outer,
/// key)` coordinate.
#[derive(Debug, Clone, Copy)]
pub struct State {
    /// Accumulated free energy (integer dekacalories; lower = better).
    pub mfe: Energy,
    /// Accumulated log-CAI (natural log; higher = better adaptation).
    pub cai: f64,
    /// How this state was derived — drives the traceback walk.
    pub back: Back,
}

impl State {
    /// Construct a state with the given accumulators and back-pointer.
    pub fn new(mfe: Energy, cai: f64, back: Back) -> Self {
        Self { mfe, cai, back }
    }

    /// Combined BRD score `λ·MFE − (1−λ)·CAI` (lower = better). The CAI
    /// term is scaled to dekacalories to match the energy unit and the
    /// interval DP's `brd_value` formula (pre-rounding).
    pub fn brd(&self, lambda: f64) -> f64 {
        lambda * f64::from(self.mfe.0) - (1.0 - lambda) * (self.cai * 100.0)
    }
}

/// How a CKY-table cell was derived. Stores the predecessor cell
/// coordinates (and any locally chosen nucleotides) the backward walk
/// needs to rebuild the `(sequence, structure)` pair.
#[derive(Debug, Clone, Copy)]
pub enum Back {
    /// Exterior base case: empty prefix at the start node.
    ExteriorEmpty,
    /// Exterior extends by one unpaired nucleotide: `C[p+1] ← C[p]`.
    ExteriorUnpaired {
        /// Predecessor exterior node `p`.
        prev: Node,
        /// Nucleotide spelled at slot `p`.
        nuc: Nucleotide,
    },
    /// Exterior absorbs a closed pair: `C[b+1] ← C[a] + P`.
    ExteriorPair {
        /// Exterior prefix node `a` (the pair's 5' boundary).
        prefix: Node,
        /// The closed-pair cell `(outer = b+1, key)`.
        pair: PairRef,
    },
    /// Hairpin closure: `P` formed from a loop with no inner pair.
    Hairpin,
    /// Stack / bulge / internal closure around one inner pair.
    Single {
        /// The enclosed closed-pair cell.
        inner: PairRef,
    },
    /// Multi-loop closure: `P ← Multi`.
    MultiClose {
        /// The multi-in-progress cell at the same `(outer, key)`.
        multi: PairRef,
    },
    /// Multi-loop assembly: `Multi` wraps an `M2` core with unpaired
    /// flanks.
    MultiWrap {
        /// The inner `M2` cell.
        core: SpanRef,
    },
    /// First multi-branch: `M ← P`.
    MFromP {
        /// The closed-pair cell forming the branch.
        pair: PairRef,
    },
    /// `M ← M2`.
    MFromM2 {
        /// The `M2` cell.
        m2: SpanRef,
    },
    /// `M` extends by one unpaired nucleotide on the right.
    MUnpaired {
        /// Predecessor `M` cell.
        prev: SpanRef,
        /// Nucleotide spelled at the extended slot.
        nuc: Nucleotide,
    },
    /// Multi-branch bifurcation: `M2 ← M + P`.
    M2Bifurcation {
        /// The `M` part.
        left: SpanRef,
        /// The `P` part.
        right: PairRef,
    },
    /// Not recorded: a values-only scan keeps a multi-branch cell's
    /// accumulators and not how it was reached (see
    /// [`crate::tables::Tables::values_only`]). No traceback reads one.
    Unrecorded,
}

/// Reference to a closed-pair (`P` / `Multi`) cell: its outer (3'
/// boundary) node and its key.
#[derive(Debug, Clone, Copy)]
pub struct PairRef {
    /// Outer table index (the `b+1` boundary node).
    pub outer: Node,
    /// The closed-pair key.
    pub key: PairKey,
}

/// Reference to a multi-branch (`M` / `M2`) cell: its outer and inner
/// boundary nodes.
#[derive(Debug, Clone, Copy)]
pub struct SpanRef {
    /// Outer (3' boundary) node.
    pub outer: Node,
    /// Inner (5' boundary) node.
    pub inner: Node,
}
