//! The codon DFA: a compact graph of every RNA sequence encoding a
//! protein.
//!
//! Each amino acid compiles to a small lattice of its synonymous codons,
//! grouped by shared nucleotide prefixes; the lattices splice end-to-end
//! into a DFA over nucleotide positions `0..=3m`. Every directed path
//! from `(0, 0)` to `(3m, 0)` spells exactly one coding sequence, and
//! every such sequence is exactly one path. Folding and codon choice
//! then happen in one pass by running a CKY scan over the DFA's nodes.
//!
//! A node is `(position, lane)`: synonymous codons that share a
//! nucleotide prefix collapse onto the same lane, so each position layer
//! carries at most two lanes. Nucleotides ride on the edges; codon
//! adaptation rides on the edges as additive log-weights, distributed by
//! a cost-to-go normalization so that a partial path's accumulated weight
//! is the best CAI still reachable through it. Summed along a complete
//! codon path the weights telescope to that codon's `log(g / g_max)`.

use crate::hash::FxHashMap;
use crate::nuc_set::NucSet;
use crate::state::Opening;
use crate::UtrProfile;
use derna_codon_dp::cai::EPSILON;
use derna_codon_dp::codon_table::{AMINO_ACIDS, STOP};
use derna_codon_dp::CodonUsage;
use rna_mfe::loops::pair_from_nucleotides;
use rna_mfe::params::BasePair;
use rna_mfe::Nucleotide;

/// One UTR flank's shape in the DFA.
///
/// Every variant occupies one node per nucleotide position and credits **no
/// CAI** — the adaptation half of the objective is the CDS codons alone. They
/// differ only in which nucleotides each position may spell. A position with a
/// choice carries a [`UtrProfile`] weight on each out-edge; a position with one
/// out-edge has nothing to choose and carries none.
#[derive(Debug, Clone, Copy)]
pub enum UtrChain<'a> {
    /// Given nucleotides: each position is a lane-of-one node with a single
    /// out-edge spelling the supplied nucleotide.
    Fixed(&'a [Nucleotide]),
    /// A designed run of `len` positions: each is one node with **four**
    /// parallel out-edges (`A`/`C`/`G`/`U`) to the next position, so the
    /// nucleotide is chosen by the fold rather than supplied. The free-
    /// nucleotide analogue of a codon's third-position choice — no lane
    /// expansion, since a closed pair keys on its endpoint nucleotides.
    Designed(u32),
    /// A per-position [`NucSet`]: each position spells one of an arbitrary
    /// non-empty subset of `{A, C, G, U}`, so its out-degree runs one to four.
    /// [`Self::Fixed`] and [`Self::Designed`] are its all-singleton and
    /// all-full ends, and it reduces to each of them exactly.
    ///
    /// A subset of size two or three is what makes a flank-to-flank helix
    /// *unspellable* rather than merely unscored — see [`crate::nuc_set`].
    Mixed(&'a [NucSet]),
}

impl UtrChain<'_> {
    /// The number of nucleotide positions this flank occupies.
    pub fn len(&self) -> u32 {
        match *self {
            Self::Fixed(nts) => nts.len() as u32,
            Self::Designed(len) => len,
            Self::Mixed(mask) => mask.len() as u32,
        }
    }

    /// Whether the flank occupies no positions — the case that collapses the
    /// flanked graph back to the CDS-only one.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The nucleotides position `i` of this flank may spell. The one place the
    /// three variants differ, so the splice below is variant-blind.
    fn at(&self, i: u32) -> NucSet {
        match *self {
            Self::Fixed(nts) => NucSet::of(nts[i as usize]),
            Self::Designed(_) => NucSet::ALL,
            Self::Mixed(mask) => mask[i as usize],
        }
    }
}

/// A DFA node: a nucleotide position together with the lane that
/// distinguishes parallel synonymous-prefix paths at that position.
///
/// `pos` runs `0..=3m` for a protein of length `m`. `lane` is `0` or `1`
/// — the coding wheel never splits an amino acid into more than two
/// prefix groups.
///
/// Packed to five bytes: a node sits in every pair key and back-pointer, and
/// at its natural alignment of eight it padded each by three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(C, packed)]
pub struct Node {
    /// Nucleotide position, `0..=3m`.
    pub pos: u32,
    /// Parallel-path lane within the position layer (`0` or `1`).
    pub lane: u8,
}

impl Node {
    /// Construct a node at `pos` on `lane`.
    pub const fn new(pos: u32, lane: u8) -> Self {
        Self { pos, lane }
    }

    /// The position, by value: a reference to a field of the packed node
    /// would be unaligned, so a caller that needs one (a map lookup, a
    /// formatting macro) borrows this copy.
    #[must_use]
    pub const fn pos(self) -> u32 {
        self.pos
    }
}

/// One outgoing DFA edge: the nucleotide consumed and the cost-to-go
/// CAI log-weight credited when the edge is traversed.
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    /// Node the edge leads to (always one position further along).
    pub to: Node,
    /// Nucleotide spelled by this edge.
    pub nuc: Nucleotide,
    /// `nuc` and the lane of the node the edge lands in, as a pair key
    /// stores them. In a left adjacency list, where `to` is the node the
    /// edge leaves, this is still the landing lane, so turning a left edge
    /// back into its right edge only replaces `to`. It sits in the padding
    /// before `cai`.
    pub opening: Opening,
    /// Cost-to-go CAI log-weight (natural log; λ applied later at the
    /// BRD scalarization, not here).
    pub cai: f64,
}

/// The spliced codon DFA for one protein, plus the per-amino-acid
/// broken-codon CAI tables consulted whenever a transition spans a
/// partial codon.
#[derive(Clone)]
pub struct Dfa {
    /// Amino-acid indices (DERNA ordering) of the encoded protein.
    pub protein: Vec<usize>,
    /// The number of nucleotide positions; the terminal node is
    /// `(seq_len, 0)`. For a CDS-only graph this is `3 * protein.len()`; a
    /// UTR-flanked graph extends it to `|UTR₅′| + 3·protein.len() + |UTR₃′|`.
    pub seq_len: u32,
    /// Position of the first CDS nucleotide — the start-codon boundary. `0`
    /// for a CDS-only graph; `|UTR₅′|` when a fixed 5′ UTR is prepended. The
    /// CDS codons occupy `cds_start .. cds_start + 3·protein.len()`; positions
    /// outside that window are UTR nodes carrying no CAI.
    cds_start: u32,
    /// The position of a stop-labelled lattice's stop exit: from here to the
    /// end every position carries one lane per stop codon, in
    /// [`STOP_CODONS`](derna_codon_dp::codon_table) order, and a lane never
    /// meets another. `None` on every other lattice.
    stop_exit: Option<u32>,
    /// Dense-index slots per position: the most lanes any position may carry.
    /// Every table keyed by node is laid out `pos * stride + lane`
    /// ([`Dfa::ix`]), and some of them are triangles over that index, so a
    /// lattice keeps the stride as small as its widest position allows.
    stride: usize,
    /// The one stop lane a stop-labelled lattice is read at
    /// ([`Dfa::for_stop`]), or `None` to read every terminal.
    terminal: Option<u8>,
    /// Whether backward passes complete to every terminal rather than to
    /// [`Self::ends`] ([`Self::completion_ends`]).
    shared_completion: bool,
    /// `nodes_at[pos]` lists every node at that position.
    nodes_at: Vec<Vec<Node>>,
    /// Outgoing edges, indexed by [`Dfa::ix`].
    right: Vec<Vec<Edge>>,
    /// Incoming edges, indexed by [`Dfa::ix`].
    left: Vec<Vec<Edge>>,
    /// Codon-unit tables, indexed by codon (amino-acid) position.
    units: Vec<CodonUnit>,
    /// Next-pair index: `next_partner[pos][open]` is the smallest position
    /// `>= pos` carrying an out-edge whose nucleotide can canonically pair
    /// with the opening nucleotide `open` (`0..=3` = A/C/G/U), or the
    /// sentinel `seq_len + 1` ("none past here"). Lets the CKY q-scans jump
    /// over closing positions that offer no pairing partner across any lane
    /// — next-pair sparsity generalized to the lane'd DFA.
    next_partner: Vec<[u32; 4]>,
    /// `whole_prefix[p]`: the summed best weight of every whole unit — a CDS
    /// codon or a UTR position — lying entirely in `[0, p)`, indexed by unit
    /// boundary (a position inside a codon carries its codon start's value).
    /// The middle term of [`Dfa::broken_codon_cai`]: a span's whole interior
    /// units weigh `whole_prefix[codon_start(end)] − whole_prefix[next boundary
    /// after start]`. Every entry is exactly zero while each unit's best
    /// available weight is zero, which the builders arrange; the table keeps
    /// the closed form exact when they do not.
    whole_prefix: Vec<f64>,
    /// Per node, by [`Dfa::ix`], the two halves of
    /// [`Dfa::broken_codon_cai`] for a span crossing a unit boundary:
    /// [`Dfa::span_head_cai`] when the node starts the span and
    /// [`Dfa::span_tail_cai`] when it ends one, prefix terms included. NaN
    /// where no partial path exists.
    exit_cai: Vec<f64>,
    entry_cai: Vec<f64>,
}

/// How a lattice lays its nodes out in a dense index: `pos * stride + lane`.
///
/// A copy of the lattice's stride for the tables that index nodes without
/// holding the lattice. Two tables over one lattice agree on every index
/// because they hold the same layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeLayout {
    stride: usize,
}

impl NodeLayout {
    /// The layout of every lattice whose positions carry at most two lanes —
    /// each codon unit's prefix groups — for a table built without a lattice
    /// at hand.
    pub const PLAIN: Self = Self { stride: 2 };

    /// Dense index of `node`.
    #[inline]
    pub const fn ix(self, node: Node) -> usize {
        node.pos as usize * self.stride + node.lane as usize
    }

    /// [`Self::ix`] with the stride given as a constant, for a caller
    /// monomorphized over one lattice shape: the multiply is then a constant
    /// and every lane-dependent test beside it folds away.
    #[inline]
    pub const fn ix_s<const S: usize>(node: Node) -> usize {
        node.pos as usize * S + node.lane as usize
    }

    /// [`Self::node_of`] with the stride given as a constant.
    #[inline]
    pub const fn node_of_s<const S: usize>(ix: usize) -> Node {
        Node::new((ix / S) as u32, (ix % S) as u8)
    }

    /// The node whose dense index is `ix` — the inverse of [`Self::ix`].
    #[inline]
    pub const fn node_of(self, ix: usize) -> Node {
        Node::new((ix / self.stride) as u32, (ix % self.stride) as u8)
    }

    /// Dense-index slots per position.
    #[inline]
    pub const fn stride(self) -> usize {
        self.stride
    }
}

impl Dfa {
    /// Dense table index of `node`, `pos * stride + lane`: the key of every
    /// flat `Vec`-backed table over this lattice's nodes and of its adjacency
    /// lists.
    #[inline]
    pub const fn ix(&self, node: Node) -> usize {
        node.pos as usize * self.stride + node.lane as usize
    }

    /// The node whose dense index is `ix` — the inverse of [`Dfa::ix`].
    #[inline]
    pub const fn node_of(&self, ix: usize) -> Node {
        self.layout().node_of(ix)
    }

    /// This lattice's dense-index layout, for a table that keys nodes without
    /// holding the lattice.
    #[inline]
    pub const fn layout(&self) -> NodeLayout {
        NodeLayout {
            stride: self.stride,
        }
    }

    /// The length of a table keyed by [`Dfa::ix`]: one slot per lane of every
    /// position `0..=seq_len`.
    #[inline]
    pub const fn slots(&self) -> usize {
        (self.seq_len as usize + 1) * self.stride
    }

    /// Dense-index slots per position ([`Dfa::ix`]).
    #[inline]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// The terminal nodes: one, or one per stop codon on a stop-labelled
    /// lattice, lane `k` ending every design whose stop is
    /// [`STOP_CODON_NUCS`]`[k]`.
    pub fn ends(&self) -> &[Node] {
        let all = self.nodes_at(self.seq_len);
        match self.terminal {
            None => all,
            Some(k) => &all[usize::from(k)..=usize::from(k)],
        }
    }

    /// This stop-labelled lattice read at one stop: the same nodes and edges,
    /// so a fold over it fills every stop's cells, but its terminal
    /// ([`Self::end`], [`Self::ends`]) is lane `k` alone, so every optimum,
    /// traceback and completion read from it is that stop's.
    ///
    /// # Panics
    ///
    /// When the lattice is not stop-labelled, or `k` names no stop.
    pub fn for_stop(&self, k: usize) -> Dfa {
        assert!(self.stop_exit.is_some(), "a stop-labelled lattice");
        assert!(k < STOP_LANES, "a stop lane");
        let mut view = self.clone();
        view.terminal = Some(k as u8);
        view
    }

    /// The terminals a backward pass completes from: [`Self::ends`], or every
    /// stop's terminal on a lattice read at one stop with
    /// [`Self::for_stop_sharing`], so the pass is the same at every stop and
    /// its completion is the least over the stops: a looser bound than the
    /// stop's own, and still a bound.
    pub fn completion_ends(&self) -> &[Node] {
        if self.shared_completion {
            self.nodes_at(self.seq_len)
        } else {
            self.ends()
        }
    }

    /// [`Self::for_stop`] whose backward passes complete to every stop
    /// ([`Self::completion_ends`]), so one pass serves each stop's walk.
    pub fn for_stop_sharing(&self, k: usize) -> Dfa {
        let mut view = self.for_stop(k);
        view.shared_completion = true;
        view
    }

    /// The stop lane this lattice is read at ([`Self::for_stop`]).
    pub const fn terminal_stop(&self) -> Option<u8> {
        self.terminal
    }

    /// The position from which a stop-labelled lattice keeps one lane per stop,
    /// or `None` when the lattice is not stop-labelled.
    pub const fn stop_exit(&self) -> Option<u32> {
        self.stop_exit
    }

    /// The stop lanes `node` still reaches, as a bit set over
    /// [`STOP_CODON_NUCS`], or `0` for "every one": always `0` on a lattice
    /// that is not stop-labelled, and before the stop codon's second
    /// nucleotide on one that is.
    #[inline]
    pub fn reach_class(&self, node: Node) -> u8 {
        match self.stop_exit {
            None => 0,
            Some(exit) => self.reach_class_past(exit, node),
        }
    }

    /// [`Self::reach_class`] on a stop-labelled lattice whose stop exit is
    /// `exit`.
    #[inline(never)]
    fn reach_class_past(&self, exit: u32, node: Node) -> u8 {
        if node.pos + 2 < exit {
            0
        } else if node.pos >= exit {
            1 << node.lane
        } else {
            (0..STOP_LANES as u8)
                .filter(|&k| self.connected(node, Node::new(exit, k)))
                .fold(0, |set, k| set | 1 << k)
        }
    }

    /// Whether a span whose 5′ side reaches the stops `reach`
    /// ([`Self::reach_class`]) can end at `node`.
    #[inline]
    pub fn reaches(&self, reach: u8, node: Node) -> bool {
        reach == 0
            || self.stop_exit.is_none_or(|exit| node.pos < exit)
            || reach & 1 << node.lane != 0
    }

    /// Whether a strand into `to` may be scored as two halves minimized apart,
    /// one beside each end with the whole codons between them free.
    ///
    /// Every lattice but a stop-labelled one allows it. On a stop-labelled one,
    /// a strand that reaches into the stop codon or past it ends on one stop's
    /// lane, and its two halves would each pick a stop of their own, so such a
    /// strand is scored whole.
    #[inline]
    pub fn halves_combine(&self, to: Node) -> bool {
        self.stop_exit.is_none_or(|exit| to.pos + 3 <= exit)
    }

    /// Whether some path of the lattice runs from `a` to `b`, for `a` at or
    /// before `b`.
    ///
    /// On every lattice but a stop-labelled one this is a question within one
    /// codon unit only, which [`Self::broken_codon_cai`] answers from the unit.
    /// A stop-labelled lattice adds the one case no unit sees: `b` past the stop
    /// exit is on its stop's lane, which `a` reaches only from before the stop,
    /// from the stop's own prefix lane (`UA` for `UAA` and `UAG`, `UG` for
    /// `UGA`), or from the same lane.
    #[inline]
    pub fn connected(&self, a: Node, b: Node) -> bool {
        match self.stop_exit {
            None => true,
            Some(exit) => Self::connected_past_stop(exit, a, b),
        }
    }

    /// [`Self::connected`] on a stop-labelled lattice whose stop exit is `exit`.
    #[inline(never)]
    fn connected_past_stop(exit: u32, a: Node, b: Node) -> bool {
        if b.pos < exit || a.pos + 2 < exit {
            return true;
        }
        if a.pos >= exit {
            return a.lane == b.lane;
        }
        // `a` is inside the stop unit, after its `U`: lane 0 is `UA`, lane 1 `UG`.
        let prefix = STOP_CODON_NUCS[usize::from(b.lane)][1];
        let lane_prefix = if a.lane == 0 {
            Nucleotide::A
        } else {
            Nucleotide::G
        };
        prefix == lane_prefix
    }

    /// Nodes at position `pos`.
    pub fn nodes_at(&self, pos: u32) -> &[Node] {
        &self.nodes_at[pos as usize]
    }

    /// Outgoing edges of `node`.
    pub fn right(&self, node: Node) -> &[Edge] {
        &self.right[self.ix(node)]
    }

    /// The largest adaptation weight any edge carries; at most zero on a
    /// lattice whose weights are log-ratios to each residue's best codon.
    pub fn max_edge_cai(&self) -> f64 {
        self.right
            .iter()
            .flatten()
            .map(|e| e.cai)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Smallest position `>= from` that carries an out-edge whose nucleotide
    /// can canonically pair with `open`, or `seq_len + 1` (a sentinel past
    /// every real position) when none exists. The CKY q-scans iterate
    /// `q = next_partner(open, q + 1)` to skip closing positions with no
    /// pairing partner, which would otherwise be visited and discarded.
    pub fn next_partner(&self, open: Nucleotide, from: u32) -> u32 {
        let i = (from as usize).min(self.next_partner.len() - 1);
        self.next_partner[i][open as usize]
    }

    /// Incoming edges of `node`.
    pub fn left(&self, node: Node) -> &[Edge] {
        &self.left[self.ix(node)]
    }

    /// The entry node `(0, 0)`.
    pub fn start(&self) -> Node {
        Node::new(0, 0)
    }

    /// The terminal node: `(seq_len, 0)`, or `(seq_len, k)` on a lattice read
    /// at stop `k` ([`Self::for_stop`]).
    pub fn end(&self) -> Node {
        Node::new(self.seq_len, self.terminal.unwrap_or(0))
    }

    /// The first CDS position — the start-codon boundary. `0` for a CDS-only
    /// graph; `|UTR₅′|` when a fixed 5′ UTR is prepended. The CDS codons occupy
    /// `cds_start .. cds_end`; positions below `cds_start` are the 5′ UTR.
    pub fn cds_start(&self) -> u32 {
        self.cds_start
    }

    /// The first position past the CDS — `cds_start + 3·protein.len()`. The
    /// CDS codons occupy `cds_start .. cds_end`; positions below `cds_start`
    /// are the 5′ UTR and positions at/above `cds_end` the 3′ UTR. A designed
    /// stop codon is the CDS's last slot, so the CDS ends past it.
    pub fn cds_end(&self) -> u32 {
        self.cds_start + 3 * self.protein.len() as u32
    }

    /// How many of the codon slots are residues rather than a designed stop:
    /// the count the conventional CAI, `exp(cai_log / m)`, is a mean over. A
    /// stop codon weighs nothing, so leaving its slot out keeps a design's CAI
    /// what it is with its stop fixed.
    pub fn sense_codons(&self) -> usize {
        derna_codon_dp::cai::sense_codons(&self.protein)
    }

    /// The unit-start position of `pos`: the codon-aligned boundary inside the
    /// CDS, or `pos` itself for a UTR position (each is its own lane-of-one
    /// unit). Reduces to `(pos / 3) * 3` for a CDS-only graph (`cds_start = 0`).
    pub fn codon_start(&self, pos: u32) -> u32 {
        if pos < self.cds_start || pos >= self.cds_end() {
            pos
        } else {
            self.cds_start + ((pos - self.cds_start) / 3) * 3
        }
    }

    /// The next unit boundary at or after `pos`: the next codon boundary inside
    /// the CDS, or `pos` for a UTR position. Reduces to `pos.div_ceil(3) * 3`
    /// for a CDS-only graph (`cds_start = 0`).
    pub fn next_codon_boundary(&self, pos: u32) -> u32 {
        if pos <= self.cds_start || pos >= self.cds_end() {
            pos
        } else {
            self.cds_start + (pos - self.cds_start).div_ceil(3) * 3
        }
    }

    /// Append the best-CAI nucleotides of every whole boundary-aligned unit in
    /// `[from, to)` to `out`: three nucleotides per whole CDS codon (its highest
    /// -CAI synonymous choice) and one per UTR position. `from` and `to` are
    /// unit boundaries, so the walk never splits a codon.
    ///
    /// A designed UTR position is filled by [`Self::best_nuc`] — its
    /// highest-weight nucleotide under the profile, which is the choice the
    /// span closed form already priced. This walk only fills spans the energy
    /// model does not read (the loop interiors it does read are respelled by
    /// [`crate::interior`], which enumerates the endpoints), so the weight is
    /// the only thing distinguishing the choices.
    fn spell_whole_units(&self, from: u32, to: u32, out: &mut Vec<Nucleotide>) {
        let e = self.cds_end();
        let mut pos = from;
        while pos < to {
            if pos >= self.cds_start && pos < e {
                let ci = ((pos - self.cds_start) / 3) as usize;
                out.extend(self.units[ci].best_whole_codon.iter().copied());
                pos += 3;
            } else {
                out.extend(self.best_nuc(pos));
                pos += 1;
            }
        }
    }

    /// The highest-weight nucleotide a node can spell: the one nucleotide
    /// available at a given UTR position or a single-path CDS node, and the
    /// most-favoured choice at a designed UTR position. `None` at the terminal,
    /// which has no out-edge.
    ///
    /// Ties keep the earliest edge, so under an inert profile — every weight
    /// zero — this is the first out-edge and the spelling is unchanged.
    fn best_nuc(&self, pos: u32) -> Option<Nucleotide> {
        self.right(Node::new(pos, 0))
            .iter()
            .reduce(|best, e| if e.cai > best.cai { e } else { best })
            .map(|e| e.nuc)
    }

    /// Whether `pos` is a designed UTR position — outside the CDS and carrying
    /// a choice of nucleotide. The positions a [`UtrProfile`] prices, and the
    /// ones whose weight a report separates from the CDS-only adaptation. A
    /// position restricted to a single nucleotide has no choice, so it is a
    /// given position however it was spelled.
    pub fn is_designed_utr(&self, pos: u32) -> bool {
        (pos < self.cds_start || pos >= self.cds_end()) && self.right(Node::new(pos, 0)).len() > 1
    }

    /// `Σ c_i` — the total of the per-position constants `splice_utr_chain`
    /// subtracted from the designed flanks' edge weights so that each restricted
    /// position's best available weight is exactly zero.
    ///
    /// The scan sums the re-normalized edges, so its adaptation accumulator is
    /// `CAI_cds + U(v) − Σ c_i` where `U` is the profile's total under its
    /// **unnormalized** weights. A report recomputing `U(v)` from the emitted
    /// sequence must therefore add this constant back to recover the CDS-only
    /// `CAI_cds`.
    ///
    /// It is a constant of the graph, not of the path: `c_i` depends only on
    /// which nucleotides position `i` admits. Zero unless some designed position
    /// excludes the profile's argmax — in particular zero for every unrestricted
    /// flank and every inert profile, which is what keeps those paths
    /// byte-identical.
    pub fn utr_profile_offset(&self, profile: UtrProfile) -> f64 {
        if profile.is_inert() {
            return 0.0;
        }
        (0..self.seq_len)
            .filter(|&pos| self.is_designed_utr(pos))
            .map(|pos| {
                self.right(Node::new(pos, 0))
                    .iter()
                    .map(|e| profile.weight(e.nuc))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .sum()
    }

    /// The adaptation the codon restrictions moved out of the edge weights,
    /// `Σ c_i` over the CDS units — the constant a report adds back.
    ///
    /// Zero for an unrestricted lattice, so every unrestricted path reports
    /// exactly what it reported before. See [`CodonUnit::offset`] for why the
    /// weights are shifted at all.
    pub fn codon_offset(&self) -> f64 {
        self.units.iter().map(|u| u.offset).sum()
    }

    /// Best cost-to-go CAI log-weight for the best DFA path from `start` to
    /// `end`: the partial units at the span's ends plus every whole unit in
    /// between, each at its best available weight, with the whole units
    /// summed rather than assumed zero.
    ///
    /// A span crossing a unit boundary separates as
    /// [`Self::span_head_cai`]`(start) + `[`Self::span_tail_cai`]`(end)`, each
    /// half reading one end alone. The scan's shared reductions (the one-sided
    /// corner factorization, the Lyngsø–Zuker index, the multi-loop opening
    /// pool) rest on that split and call the halves directly.
    ///
    /// A span crossing a unit boundary reads its two halves from per-node
    /// tables; the rest go the long way ([`Self::broken_codon_cai_walk`]),
    /// which debug builds also check the tables against.
    #[inline]
    pub fn broken_codon_cai(&self, start: Node, end: Node) -> Option<f64> {
        self.broken_codon_cai_s::<0>(start, end)
    }

    /// [`Self::broken_codon_cai`] on a lattice whose lane stride is the
    /// constant `S`, for a caller monomorphized over one lattice shape: at
    /// stride 2 no position holds a stop lane, so the connectivity test is
    /// known to hold and compiles away. `S = 0` asks the lattice.
    #[inline]
    pub fn broken_codon_cai_s<const S: usize>(&self, start: Node, end: Node) -> Option<f64> {
        if S != 2 && !self.connected(start, end) {
            return None;
        }
        let (s, e) = (self.cds_start, self.cds_end());
        let crosses = start != end
            && start.pos < end.pos
            && !(start.pos >= s && end.pos < e && (start.pos - s) / 3 == (end.pos - s) / 3);
        if !crosses {
            return self.broken_codon_cai_walk(start, end);
        }
        let exit = self.exit_cai[self.ix(start)];
        let entry = self.entry_cai[self.ix(end)];
        let fast = if exit.is_nan() || entry.is_nan() {
            None
        } else {
            let mut cai = 0.0;
            cai += exit;
            cai += entry;
            Some(cai)
        };
        debug_assert_eq!(
            fast.map(f64::to_bits),
            self.broken_codon_cai_walk(start, end).map(f64::to_bits),
            "the per-node halves disagree with the walk"
        );
        fast
    }

    /// The two per-node halves of a boundary-crossing
    /// [`Self::broken_codon_cai`], for every node index.
    fn cai_halves(&self) -> (Vec<f64>, Vec<f64>) {
        (0..self.right.len())
            .map(|idx| {
                let node = self.node_of(idx);
                (
                    self.head_value(node).unwrap_or(f64::NAN),
                    self.tail_value(node).unwrap_or(f64::NAN),
                )
            })
            .unzip()
    }

    /// [`Self::broken_codon_cai`] computed from the codon units directly, for
    /// `start` connected to `end`.
    fn broken_codon_cai_walk(&self, start: Node, end: Node) -> Option<f64> {
        if start == end {
            return Some(0.0);
        }
        // Two distinct nodes at the same position are parallel codon lanes
        // with no connecting DFA edge: bridging them would splice
        // nucleotides from different codons into one position. The span is
        // structurally impossible, so `None` prunes the derivation rather
        // than crediting a free zero-cost hop between lanes.
        if start.pos >= end.pos {
            return None;
        }
        let (s, e) = (self.cds_start, self.cds_end());
        // Span confined to a single CDS codon: one partial path between two
        // interior relative nodes. Two different lanes within a codon are not
        // connected by any path, so the span is infeasible — `None` signals the
        // caller to skip this (structurally impossible) derivation.
        if start.pos >= s && end.pos < e && (start.pos - s) / 3 == (end.pos - s) / 3 {
            let ci = ((start.pos - s) / 3) as usize;
            let from = RelNode::layer((start.pos - s) % 3, start.lane);
            let to = RelNode::layer((end.pos - s) % 3, end.lane);
            return self.units[ci].best_path_cai(from, to);
        }
        // Span crosses a unit boundary: every (layer, lane) reaches its codon's
        // exit and the entry reaches every (layer, lane), so the partial CDS
        // codons at the ends are always connectable, and the whole units in
        // between contribute their best weights through the prefix terms each
        // half carries.
        let mut cai = 0.0;
        cai += self.head_value(start)?;
        cai += self.tail_value(end)?;
        Some(cai)
    }

    /// The 5′ half of a boundary-crossing span from `start`: the best partial
    /// codon from `start` to its codon's exit, less the whole units before that
    /// exit. Zero partial when `start` sits on a unit boundary.
    ///
    /// `broken_codon_cai(a, b) = span_head_cai(a) + span_tail_cai(b)` for any
    /// `a`, `b` in different units, so a reduction minimizing over one end can
    /// fold this half in once and share it across every other end.
    pub fn span_head_cai(&self, start: Node) -> Option<f64> {
        let v = self.exit_cai[self.ix(start)];
        (!v.is_nan()).then_some(v)
    }

    /// [`Self::span_head_cai`] computed from the units and the prefix table.
    fn head_value(&self, start: Node) -> Option<f64> {
        let mut cai = 0.0;
        cai += self.head_partial(start)?;
        Some(cai - self.whole_prefix[self.next_codon_boundary(start.pos) as usize])
    }

    /// The 3′ half of a boundary-crossing span into `end`: the whole units up
    /// to `end`'s codon start, plus the best partial codon from that start to
    /// `end`. See [`Self::span_head_cai`].
    pub fn span_tail_cai(&self, end: Node) -> Option<f64> {
        let v = self.entry_cai[self.ix(end)];
        (!v.is_nan()).then_some(v)
    }

    /// [`Self::span_tail_cai`] computed from the units and the prefix table.
    fn tail_value(&self, end: Node) -> Option<f64> {
        let partial = self.tail_partial(end)?;
        Some(self.whole_prefix[self.codon_start(end.pos) as usize] + partial)
    }

    /// The best partial path from `start` to its codon's exit, `0` when
    /// `start` is not inside a CDS codon.
    fn head_partial(&self, start: Node) -> Option<f64> {
        let (s, e) = (self.cds_start, self.cds_end());
        if start.pos > s && start.pos < e && !(start.pos - s).is_multiple_of(3) {
            let ci = ((start.pos - s) / 3) as usize;
            let from = RelNode::layer((start.pos - s) % 3, start.lane);
            return self.units[ci].best_path_cai(from, RelNode::EXIT);
        }
        Some(0.0)
    }

    /// The best partial path from `end`'s codon entry to `end`, `0` when `end`
    /// is not inside a CDS codon.
    fn tail_partial(&self, end: Node) -> Option<f64> {
        let (s, e) = (self.cds_start, self.cds_end());
        if end.pos > s && end.pos < e && !(end.pos - s).is_multiple_of(3) {
            let ci = ((end.pos - s) / 3) as usize;
            let to = RelNode::layer((end.pos - s) % 3, end.lane);
            return self.units[ci].best_path_cai(RelNode::ENTRY, to);
        }
        Some(0.0)
    }

    /// The prefix table behind the closed form's middle term: each whole unit's
    /// best weight, summed boundary to boundary.
    fn build_whole_prefix(&self) -> Vec<f64> {
        let mut prefix = vec![0.0; self.seq_len as usize + 1];
        if middle_term_skipped() {
            return prefix;
        }
        let (s, e) = (self.cds_start, self.cds_end());
        let mut pos = 0;
        while pos < self.seq_len {
            let (next, best) = if pos >= s && pos < e {
                let unit = &self.units[((pos - s) / 3) as usize];
                (pos + 3, unit.best_whole_cai())
            } else {
                let best = self
                    .right(Node::new(pos, 0))
                    .iter()
                    .map(|edge| edge.cai)
                    .fold(f64::NEG_INFINITY, f64::max);
                (pos + 1, best)
            };
            let here = prefix[pos as usize];
            // A codon's interior positions read its start's value.
            prefix[pos as usize + 1..next as usize].fill(here);
            prefix[next as usize] = here + best;
            pos = next;
        }
        prefix
    }

    /// The first unit boundary strictly after `pos`: the end of `pos`'s codon
    /// inside the CDS, or the next position in a flank, clamped to the end.
    fn unit_end_after(&self, pos: u32) -> u32 {
        let (s, e) = (self.cds_start, self.cds_end());
        if pos < s || pos >= e {
            (pos + 1).min(self.seq_len)
        } else {
            s + ((pos - s) / 3 + 1) * 3
        }
    }

    /// Whether any path from `at` spells `seq[i..end]` (nucleotide indices).
    fn spells(&self, at: Node, seq: &[u8], i: usize, end: usize) -> bool {
        i >= end
            || self
                .right(at)
                .iter()
                .any(|e| e.nuc as u8 == seq[i] && self.spells(e.to, seq, i + 1, end))
    }

    /// The out-edge of `at` that spells `seq[i]` *and* lies on a path spelling
    /// the rest of `at`'s codon. `seq` holds nucleotide indices (`A=0, C=1,
    /// G=2, U=3`), as [`Edge::nuc`] does.
    ///
    /// # A lane cannot be chosen greedily
    ///
    /// Two codon lanes can offer the same nucleotide out of one node, so taking
    /// the first out-edge that spells `seq[i]` can commit to a lane the rest of
    /// the sequence cannot leave — and then report a design the lattice does
    /// spell as one it does not. The twenty amino acids hide this, because each
    /// one's two prefix groups differ in their **first** nucleotide; the stop
    /// family's groups are `UA` and `UG`, which do not. The lanes converge at
    /// the end of the unit, so looking that far ahead — three steps at most —
    /// decides between them, and a sequence is exactly one path.
    pub fn step_on_path(&self, at: Node, seq: &[u8], i: usize) -> Option<Edge> {
        let edges = self.right(at);
        let mut matching = edges.iter().filter(|e| e.nuc as u8 == seq[i]);
        let first = *matching.next()?;
        if matching.next().is_none() {
            return Some(first);
        }
        let end = (self.unit_end_after(at.pos) as usize).min(seq.len());
        edges
            .iter()
            .copied()
            .find(|e| e.nuc as u8 == seq[i] && self.spells(e.to, seq, i + 1, end))
    }

    /// The edges of a path from `at` spelling `nucs` (nucleotide indices) and,
    /// when `end` is given, ending at that node; `None` when there is none.
    ///
    /// The span form of [`Self::step_on_path`]: a lane cannot be chosen
    /// greedily, so a branch is kept only while the rest of `nucs` — and
    /// `end`'s lane, where a span stops inside a unit — can still be reached
    /// from it. Only the stop family branches, so the search backtracks at
    /// most once per stop slot.
    pub fn spell_to(&self, at: Node, nucs: &[u8], end: Option<Node>) -> Option<Vec<Edge>> {
        let mut path = Vec::with_capacity(nucs.len());
        self.extend_spelling(at, nucs, end, &mut path)
            .then_some(path)
    }

    fn extend_spelling(
        &self,
        at: Node,
        nucs: &[u8],
        end: Option<Node>,
        path: &mut Vec<Edge>,
    ) -> bool {
        let Some((&first, rest)) = nucs.split_first() else {
            return end.is_none_or(|end| end == at);
        };
        for e in self.right(at).iter().filter(|e| e.nuc as u8 == first) {
            path.push(*e);
            if self.extend_spelling(e.to, rest, end, path) {
                return true;
            }
            path.pop();
        }
        false
    }

    /// The path spelling the whole molecule `seq`: the node before each
    /// position, then the terminal — `seq.len() + 1` nodes. `None` when this
    /// lattice does not spell `seq`, a `seq` of the wrong length included.
    pub fn path_of(&self, seq: &[Nucleotide]) -> Option<Vec<Node>> {
        self.walk_path(seq).map(|(nodes, _)| nodes)
    }

    /// The cost-to-go CAI log-weight `seq` accumulates along its own path, or
    /// `None` when this lattice does not spell it. The adaptation of a design
    /// the lattice carries, read back from the sequence alone.
    pub fn path_cai(&self, seq: &[Nucleotide]) -> Option<f64> {
        self.walk_path(seq).map(|(_, cai)| cai)
    }

    /// The path and its accumulated weight, behind [`Self::path_of`] and
    /// [`Self::path_cai`] so the two cannot disagree about what a sequence
    /// spells.
    fn walk_path(&self, seq: &[Nucleotide]) -> Option<(Vec<Node>, f64)> {
        if seq.len() as u32 != self.seq_len {
            return None;
        }
        let idx: Vec<u8> = seq.iter().map(|&n| n as u8).collect();
        let mut node = self.start();
        let mut nodes = Vec::with_capacity(seq.len() + 1);
        let mut cai = 0.0;
        nodes.push(node);
        for i in 0..idx.len() {
            let e = self.step_on_path(node, &idx, i)?;
            cai += e.cai;
            node = e.to;
            nodes.push(node);
        }
        (node == self.end()).then_some((nodes, cai))
    }

    /// Every DFA path from `start` to the downstream node `end`, each as
    /// its accumulated CAI log-weight and the nucleotides spelled along it.
    ///
    /// Used to enumerate the energy-determining interior of a small loop
    /// (the special-hairpin sizes 3/4/6, where the whole loop sequence
    /// selects a Triloop/Tetraloop/Hexaloop bonus): the loop energy and the
    /// CAI must be optimized jointly, so the scan considers each interior
    /// spelling rather than reconstructing one by best-CAI alone. The path
    /// count is bounded by the DFA's two-lanes-per-position fan-out, so this
    /// stays cheap for the short spans it is called on.
    pub fn paths_between(&self, start: Node, end: Node) -> Vec<(f64, Vec<Nucleotide>)> {
        let mut out = Vec::new();
        if start == end {
            out.push((0.0, Vec::new()));
            return out;
        }
        let mut stack: Vec<(Node, f64, Vec<Nucleotide>)> = vec![(start, 0.0, Vec::new())];
        while let Some((node, cai, seq)) = stack.pop() {
            if node == end {
                out.push((cai, seq));
                continue;
            }
            if node.pos >= end.pos {
                continue;
            }
            for e in self.right(node) {
                let mut next = seq.clone();
                next.push(e.nuc);
                stack.push((e.to, cai + e.cai, next));
            }
        }
        out
    }

    /// Best-CAI nucleotide path spelling the codons between `start` and
    /// `end`, used by traceback to fill loop interiors.
    pub fn broken_codon_seq(&self, start: Node, end: Node) -> Vec<Nucleotide> {
        if start.pos >= end.pos {
            return Vec::new();
        }
        let (s, e) = (self.cds_start, self.cds_end());
        // Span confined to a single CDS codon: one partial path.
        if start.pos >= s && end.pos < e && (start.pos - s) / 3 == (end.pos - s) / 3 {
            let ci = ((start.pos - s) / 3) as usize;
            let from = RelNode::layer((start.pos - s) % 3, start.lane);
            let to = RelNode::layer((end.pos - s) % 3, end.lane);
            return self.units[ci].best_path_seq(from, to);
        }
        let mut out = Vec::with_capacity((end.pos - start.pos) as usize);
        // Left partial CDS codon (start lands mid-codon).
        if start.pos > s && start.pos < e && !(start.pos - s).is_multiple_of(3) {
            let ci = ((start.pos - s) / 3) as usize;
            let from = RelNode::layer((start.pos - s) % 3, start.lane);
            out.extend(self.units[ci].best_path_seq(from, RelNode::EXIT));
        }
        // Whole units strictly inside the span (best-CAI whole CDS codons and
        // forced 5′/3′ UTR nucleotides), walked boundary to boundary.
        self.spell_whole_units(
            self.next_codon_boundary(start.pos),
            self.codon_start(end.pos),
            &mut out,
        );
        // Right partial CDS codon (end lands mid-codon).
        if end.pos > s && end.pos < e && !(end.pos - s).is_multiple_of(3) {
            let ci = ((end.pos - s) / 3) as usize;
            let to = RelNode::layer((end.pos - s) % 3, end.lane);
            out.extend(self.units[ci].best_path_seq(RelNode::ENTRY, to));
        }
        self.spell_labelled_stop(start, end, &mut out);
        out
    }

    /// On a stop-labelled lattice, respell the stop codon inside `out` — the
    /// fill of the span `start → end` — as the stop of `end`'s lane: the unit's
    /// best codon is a choice among all three, and a span ending past the stop
    /// is on one of them. `start` reaches that lane ([`Self::connected`]), so
    /// the part of the codon before the span agrees with it.
    fn spell_labelled_stop(&self, start: Node, end: Node, out: &mut [Nucleotide]) {
        let Some(exit) = self.stop_exit else {
            return;
        };
        if end.pos < exit || start.pos >= exit {
            return;
        }
        let codon = STOP_CODON_NUCS[usize::from(end.lane)];
        for pos in start.pos.max(exit - 3)..exit {
            out[(pos - start.pos) as usize] = codon[(pos + 3 - exit) as usize];
        }
    }
}

/// Build the codon DFA for `protein` (a slice of DERNA amino-acid
/// indices) under the given codon-usage table.
pub fn build(protein: &[usize], usage: &CodonUsage) -> Dfa {
    let units = protein
        .iter()
        .map(|&aa| CodonUnit::build(aa, usage))
        .collect();
    splice_units(protein, units)
}

/// Build a single-path codon DFA: each protein position is pinned to one
/// chosen codon (a slice of nucleotide-index triples, DERNA `A=0,C=1,G=2,
/// U=3`). The unique path spells exactly that coding sequence, so a λ=1
/// scan over it folds a fixed RNA — the fold-parity cross-check substrate.
pub fn build_single_path(protein: &[usize], codons: &[[u8; 3]]) -> Dfa {
    let units = codons.iter().map(|c| CodonUnit::single(*c)).collect();
    splice_units(protein, units)
}

fn splice_units(protein: &[usize], units: Vec<CodonUnit>) -> Dfa {
    splice_full(
        protein,
        units,
        (UtrChain::Fixed(&[]), UtrChain::Fixed(&[])),
        UtrProfile::inert(),
        false,
    )
}

/// Build the full-mRNA DFA with each protein position restricted to its own
/// subset of synonymous codons — the substrate for searching a *part* of the
/// design space.
///
/// `allowed[i]` lists the codons position `i` may take. Restricting a position
/// to one codon pins it; restricting it to all of them is the unrestricted
/// lattice. Returns `None` if any position's subset is empty, which is how an
/// over-restricted region of the space reports itself infeasible rather than
/// returning a garbage optimum.
///
/// CAI weights are unchanged by the restriction, so optima over different
/// subsets are comparable — the property any search that orders subproblems by
/// objective depends on.
pub fn build_restricted_with_utr(
    protein: &[usize],
    usage: &CodonUsage,
    allowed: &[Vec<[u8; 3]>],
    utr5: &[Nucleotide],
    utr3: &[Nucleotide],
) -> Option<Dfa> {
    build_composite(
        protein,
        usage,
        Some(allowed),
        UtrChain::Fixed(utr5),
        UtrChain::Fixed(utr3),
        UtrProfile::inert(),
    )
}

/// The best adaptation a restricted codon lattice admits:
/// `Σ_i max_{c ∈ allowed_i} log CAI(c)`, or `None` when some position admits no
/// codon at all.
///
/// This is [`Dfa::codon_offset`] without the automaton — the same per-position
/// constant, derived from the same weights, for a caller that wants the number
/// and not the lattice. It is a *ceiling*: adaptation is separable, so no
/// structure can make a position spell better than the best codon left to it,
/// and no encoding the restriction admits scores above this sum. Zero for an
/// unrestricted lattice, and strictly negative as soon as a restriction takes a
/// family's best codon away.
pub fn cai_ceiling(protein: &[usize], usage: &CodonUsage, allowed: &[Vec<[u8; 3]>]) -> Option<f64> {
    protein
        .iter()
        .zip(allowed)
        .map(|(&aa, codons)| {
            let mut options = build_options_over(aa, usage, Some(codons));
            (!options.is_empty()).then(|| renormalize(&mut options))
        })
        .sum()
}

/// The general builder every other one is a corner of: a codon lattice
/// optionally restricted per residue, between two flank chains of any kind,
/// priced by any profile.
///
/// `allowed` is `None` for the unrestricted lattice — not an all-codons
/// restriction, which would merely reproduce it through a different code path.
/// `Some` restricts each residue to its listed codons and returns `None` when a
/// residue's list is empty, which is how an over-restricted design space reports
/// itself infeasible rather than returning a garbage optimum.
///
/// CAI weights are unchanged by the restriction, so optima over different
/// subsets stay comparable — the property any search ordering subproblems by
/// objective depends on.
pub fn build_composite(
    protein: &[usize],
    usage: &CodonUsage,
    allowed: Option<&[Vec<[u8; 3]>]>,
    utr5: UtrChain<'_>,
    utr3: UtrChain<'_>,
    profile: UtrProfile,
) -> Option<Dfa> {
    let units: Vec<CodonUnit> = match allowed {
        None => protein
            .iter()
            .map(|&aa| CodonUnit::build(aa, usage))
            .collect(),
        // Every restricted unit — down to the one-codon case — is compiled by
        // [`CodonUnit::restricted`], which shifts its edges so the best
        // *available* codon sits at weight zero and records what it removed.
        // That keeps the middle term of [`Dfa::broken_codon_cai`] exactly zero,
        // and it means no parse can claim adaptation it never spelled: a pinned
        // residue's edges are zero and its constant reaches the report through
        // [`Dfa::codon_offset`] instead.
        Some(sets) => protein
            .iter()
            .zip(sets)
            .map(|(&aa, set)| CodonUnit::restricted(aa, usage, set))
            .collect::<Option<Vec<_>>>()?,
    };
    Some(splice_full(protein, units, (utr5, utr3), profile, false))
}

/// Build a lane-of-one DFA over an already-decided whole mRNA: every position
/// has exactly one out-edge and no CAI is credited anywhere, so a scan over it
/// folds that fixed sequence and nothing else.
///
/// The codon lattice is empty by construction, which is the point — this is the
/// substrate for scoring a *given* sequence rather than choosing one, as when a
/// candidate design is checked against a constraint it was not designed under.
/// The region window such a check needs is supplied to the constraint directly rather
/// than read off this graph, since the graph carries no CDS.
pub fn build_fixed_mrna(seq: &[Nucleotide]) -> Dfa {
    splice_full(
        &[],
        Vec::new(),
        (UtrChain::Fixed(seq), UtrChain::Fixed(&[])),
        UtrProfile::inert(),
        false,
    )
}

/// Build the full-mRNA codon DFA for `protein` under `usage`, with fixed 5′
/// and 3′ UTR nucleotides folded jointly with the designed CDS. Shorthand for
/// [`build_with_utr_chain`] over two [`UtrChain::Fixed`] flanks.
pub fn build_with_utr(
    protein: &[usize],
    usage: &CodonUsage,
    utr5: &[Nucleotide],
    utr3: &[Nucleotide],
) -> Dfa {
    build_with_utr_chain(protein, usage, UtrChain::Fixed(utr5), UtrChain::Fixed(utr3))
}

/// Build the full-mRNA codon DFA for `protein` under `usage`, flanked by the
/// given 5′ and 3′ UTR chains. The whole mRNA folds as one structure: the scan
/// domain grows from `0..3m` to `0..(|UTR₅′| + 3m + |UTR₃′|)`, and UTR positions
/// credit no CAI whether they are fixed or designed. Empty flanks collapse to
/// exactly the CDS-only graph [`build`] produces.
pub fn build_with_utr_chain(
    protein: &[usize],
    usage: &CodonUsage,
    utr5: UtrChain<'_>,
    utr3: UtrChain<'_>,
) -> Dfa {
    build_with_utr_profile(protein, usage, utr5, utr3, UtrProfile::inert())
}

/// [`build_with_utr_chain`] with a [`UtrProfile`] pricing the flanks' designed
/// positions. An inert profile is byte-identical to `build_with_utr_chain`, and
/// a flank with no designed position is unaffected by any profile.
pub fn build_with_utr_profile(
    protein: &[usize],
    usage: &CodonUsage,
    utr5: UtrChain<'_>,
    utr3: UtrChain<'_>,
    profile: UtrProfile,
) -> Dfa {
    build_composite(protein, usage, None, utr5, utr3, profile)
        .expect("an unrestricted lattice has every codon at every residue")
}

/// Lanes per position of a stop-labelled lattice's 3′ flank: one per stop codon.
pub const STOP_LANES: usize = 3;

/// The stop codons in the order their lanes take: the stop family's own order.
pub const STOP_CODON_NUCS: [[Nucleotide; 3]; STOP_LANES] = {
    use Nucleotide::{A, G, U};
    [[U, A, A], [U, A, G], [U, G, A]]
};

/// [`build_with_utr_profile`] with the designed stop codon's identity kept to
/// the end of the molecule: a **stop-labelled** lattice.
///
/// `protein` ends in the stop family ([`STOP`]). Its three codons do not merge
/// at the unit's exit as every other unit's codons do: each lands on a lane of
/// its own, the 3′ flank is spliced once per lane, and the lattice ends in one
/// terminal node per stop ([`Dfa::ends`]). A fold over it keeps every cell
/// whose span reaches past the stop apart by stop, so its optimum at a terminal
/// is the optimum with that stop codon pinned. Returns `None` when `protein`
/// does not end in a designed stop.
pub fn build_stop_labelled(
    protein: &[usize],
    usage: &CodonUsage,
    utr5: UtrChain<'_>,
    utr3: UtrChain<'_>,
    profile: UtrProfile,
) -> Option<Dfa> {
    if protein.last() != Some(&STOP) {
        return None;
    }
    let units = protein
        .iter()
        .map(|&aa| CodonUnit::build(aa, usage))
        .collect();
    Some(splice_full(protein, units, (utr5, utr3), profile, true))
}

/// Splice `units` (the CDS codon lattices) between the `utr5` and `utr3` UTR
/// chains, pricing designed flank positions by `profile`. With empty flanks
/// `cds_start = 0` and the node/edge set is byte-identical to the CDS-only
/// graph.
fn splice_full(
    protein: &[usize],
    units: Vec<CodonUnit>,
    (utr5, utr3): (UtrChain<'_>, UtrChain<'_>),
    profile: UtrProfile,
    labelled: bool,
) -> Dfa {
    let cds_start = utr5.len();
    let seq_len = cds_start + 3 * protein.len() as u32 + utr3.len();
    let labelled = labelled && protein.last() == Some(&STOP);
    let stride = if labelled { STOP_LANES } else { 2 };
    let n_nodes = (seq_len as usize + 1) * stride;
    let mut dfa = Dfa {
        protein: protein.to_vec(),
        seq_len,
        cds_start,
        stop_exit: labelled.then(|| cds_start + 3 * protein.len() as u32),
        stride,
        terminal: None,
        shared_completion: false,
        nodes_at: vec![Vec::new(); seq_len as usize + 1],
        right: vec![Vec::new(); n_nodes],
        left: vec![Vec::new(); n_nodes],
        units,
        next_partner: Vec::new(),
        whole_prefix: Vec::new(),
        exit_cai: Vec::new(),
        entry_cai: Vec::new(),
    };
    let cds_end = dfa.cds_end();
    let tail_lanes = if labelled { STOP_LANES as u8 } else { 1 };
    for lane in 0..tail_lanes {
        add_node(&mut dfa, Node::new(seq_len, lane));
    }
    splice_utr_chain(&mut dfa, 0, utr5, profile, 0);
    let codons = protein.len() as u32;
    for i in 0..codons {
        if labelled && i + 1 == codons {
            splice_stop_unit(&mut dfa, cds_start, i);
        } else {
            splice_codon_unit(&mut dfa, cds_start, i);
        }
    }
    for lane in 0..tail_lanes {
        splice_utr_chain(&mut dfa, cds_end, utr3, profile, lane);
    }
    dfa.next_partner = build_next_partner(&dfa);
    dfa.whole_prefix = dfa.build_whole_prefix();
    (dfa.exit_cai, dfa.entry_cai) = dfa.cai_halves();
    dfa
}

/// Splice a UTR run as single-lane nodes starting at `base`. Each position gets
/// one out-edge per nucleotide its [`NucSet`] admits, all to the same successor,
/// so the fold chooses among them under a stated preference rather than under
/// folding energy alone. The node at `base + utr.len()` (the run's downstream
/// boundary) is added by its owner — the first CDS codon, the terminal, or
/// nothing when the run is empty — so it is not added here.
///
/// # The edge weight is re-normalized per position
///
/// [`Dfa::broken_codon_cai`] is an `O(1)` closed form returning the **best
/// still-reachable** weight over a span; the whole units in the span's interior
/// enter it through a prefix sum of each unit's best weight. A profile's raw
/// weights are `≤ 0` with a maximum of exactly `0` *over the full alphabet*, so
/// an unrestricted designed position adds exactly nothing to that sum.
///
/// Restrict a position to a subset that excludes the profile's argmax and its
/// best available weight is strictly below zero. Subtracting the position's own
/// best, `c_i = max_{y ∈ subset} w(y) ≤ 0`, restores a maximum of exactly zero
/// over the *available* subset, so the prefix sum stays exactly zero and no
/// span's value picks up rounding from it. The closed form is exact either way;
/// the shift is what keeps a restricted graph's span values bit-identical to an
/// unrestricted one's.
///
/// The scan therefore accumulates `CAI_cds + U(v) − Σ c_i`, and the report adds
/// `Σ c_i` back — see [`Dfa::utr_profile_offset`], which recovers the constant,
/// and `finish_with_profile`, which applies it. On a full-alphabet position
/// `c_i` is `0` and every weight is unchanged; on a pinned position `c_i` is the
/// one available weight, so its single edge lands at exactly `0.0` and a given
/// flank is untouched by any profile.
fn splice_utr_chain(dfa: &mut Dfa, base: u32, utr: UtrChain<'_>, profile: UtrProfile, lane: u8) {
    for i in 0..utr.len() {
        let (from, to) = (Node::new(base + i, lane), Node::new(base + i + 1, lane));
        add_node(dfa, from);
        let set = utr.at(i);
        let best = if normalization_skipped() {
            0.0
        } else {
            best_available_weight(set, profile)
        };
        for nt in set.iter() {
            add_edge(dfa, from, to, nt, profile.weight(nt) - best);
        }
    }
}

/// The best weight `profile` offers over `set` — the `c_i` the position's edges
/// are re-normalized by. Zero whenever the profile's argmax is available, which
/// includes every unrestricted designed position and every inert profile.
fn best_available_weight(set: NucSet, profile: UtrProfile) -> f64 {
    set.iter()
        .map(|nt| profile.weight(nt))
        .fold(f64::NEG_INFINITY, f64::max)
}

#[cfg(test)]
thread_local! {
    /// The mutant the closed-form oracle must catch: restricted units and
    /// flank positions keep their raw weights instead of being shifted so their
    /// best available weight is zero. Set only by
    /// `closed_form_oracle::with_normalization_skipped`.
    static SKIP_NORMALIZATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The closed form's own mutant: the whole-unit middle term reads zero, as
    /// it would if the form assumed every whole unit's best is zero. Set only by
    /// `closed_form_oracle::with_middle_term_skipped`.
    static SKIP_MIDDLE_TERM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the per-unit normalization is switched off — always `false` outside
/// the test build.
fn normalization_skipped() -> bool {
    #[cfg(test)]
    return SKIP_NORMALIZATION.with(std::cell::Cell::get);
    #[cfg(not(test))]
    false
}

/// Whether the closed form's middle term is switched off — always `false`
/// outside the test build.
fn middle_term_skipped() -> bool {
    #[cfg(test)]
    return SKIP_MIDDLE_TERM.with(std::cell::Cell::get);
    #[cfg(not(test))]
    false
}

/// Precompute the next-pair index: for each opening nucleotide and each
/// position, the next position at or after it that offers a canonical
/// pairing partner on some out-edge. A reverse sweep so each cell either
/// holds its own position (a partner exists here) or inherits the next
/// position downstream that does (`seq_len + 1` past the end).
fn build_next_partner(dfa: &Dfa) -> Vec<[u32; 4]> {
    let none = dfa.seq_len + 1;
    let mut np = vec![[none; 4]; (dfa.seq_len + 2) as usize];
    for pos in (0..=dfa.seq_len).rev() {
        for open in 0u8..4 {
            let here = position_has_partner(dfa, pos, nuc(open));
            np[pos as usize][open as usize] = if here {
                pos
            } else {
                np[pos as usize + 1][open as usize]
            };
        }
    }
    np
}

/// Whether some node at `pos` has an out-edge whose nucleotide canonically
/// pairs with `open`.
fn position_has_partner(dfa: &Dfa, pos: u32, open: Nucleotide) -> bool {
    dfa.nodes_at(pos)
        .iter()
        .flat_map(|&n| dfa.right(n))
        .any(|e| !matches!(pair_from_nucleotides(open, e.nuc), BasePair::NN))
}

/// Splice codon position `i`'s lattice into the protein DFA at nucleotide
/// positions `cds_start + 3i .. cds_start + 3i+3`; `cds_start` is the
/// 5′-UTR offset (`0` for a CDS-only graph).
fn splice_codon_unit(dfa: &mut Dfa, cds_start: u32, i: u32) {
    let base = cds_start + i * 3;
    add_node(dfa, Node::new(base, 0));
    for o in 0..dfa.units[i as usize].options.len() {
        let opt = dfa.units[i as usize].options[o].clone();
        let lane = o as u8;
        let layer1 = Node::new(base + 1, lane);
        let layer2 = Node::new(base + 2, lane);
        add_node(dfa, layer1);
        add_node(dfa, layer2);
        add_edge(dfa, Node::new(base, 0), layer1, opt.first, opt.w_first);
        add_edge(dfa, layer1, layer2, opt.second, 0.0);
        for &(third, w_third) in &opt.thirds {
            add_edge(dfa, layer2, Node::new(base + 3, 0), third, w_third);
        }
    }
}

/// Splice the stop family's unit at codon slot `i` with its three codons kept
/// apart: each third-nucleotide edge lands on the exit lane of its own stop, in
/// [`STOP_CODONS`](derna_codon_dp::codon_table) order, instead of on one shared
/// exit node. The unit's two prefix lanes (`UA`, `UG`) are as in
/// [`splice_codon_unit`].
fn splice_stop_unit(dfa: &mut Dfa, cds_start: u32, i: u32) {
    let base = cds_start + i * 3;
    add_node(dfa, Node::new(base, 0));
    for o in 0..dfa.units[i as usize].options.len() {
        let opt = dfa.units[i as usize].options[o].clone();
        let lane = o as u8;
        let layer1 = Node::new(base + 1, lane);
        let layer2 = Node::new(base + 2, lane);
        add_node(dfa, layer1);
        add_node(dfa, layer2);
        add_edge(dfa, Node::new(base, 0), layer1, opt.first, opt.w_first);
        add_edge(dfa, layer1, layer2, opt.second, 0.0);
        for &(third, w_third) in &opt.thirds {
            let stop = stop_lane([opt.first, opt.second, third]);
            add_edge(dfa, layer2, Node::new(base + 3, stop), third, w_third);
        }
    }
}

/// The exit lane of the stop codon `codon` on a stop-labelled lattice: its
/// index in the stop family.
fn stop_lane(codon: [Nucleotide; 3]) -> u8 {
    STOP_CODON_NUCS
        .iter()
        .position(|&c| c == codon)
        .expect("a stop unit spells only stop codons") as u8
}

fn add_node(dfa: &mut Dfa, node: Node) {
    dfa.nodes_at[node.pos as usize].push(node);
}

fn add_edge(dfa: &mut Dfa, from: Node, to: Node, nuc: Nucleotide, cai: f64) {
    let opening = Opening::new(nuc, to.lane);
    let (from_ix, to_ix) = (dfa.ix(from), dfa.ix(to));
    dfa.right[from_ix].push(Edge {
        to,
        nuc,
        opening,
        cai,
    });
    dfa.left[to_ix].push(Edge {
        to: from,
        nuc,
        opening,
        cai,
    });
}

/// A node relative to one codon unit: `layer ∈ {0,1,2}` plus an explicit
/// `EXIT` (layer 3). Lane is `0` or `1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RelNode {
    layer: u8,
    lane: u8,
}

impl RelNode {
    const ENTRY: RelNode = RelNode { layer: 0, lane: 0 };
    const EXIT: RelNode = RelNode { layer: 3, lane: 0 };

    fn layer(layer: u32, lane: u8) -> Self {
        Self {
            layer: layer as u8,
            lane,
        }
    }

    /// Dense index `layer * 2 + lane` into a codon unit's
    /// [`CodonUnit::best_cai`] table. `layer ∈ {0,1,2,3}`, `lane ∈ {0,1}`, so
    /// the index is in `0..8`.
    const fn index(self) -> usize {
        (self.layer * 2 + self.lane) as usize
    }
}

/// Number of distinct [`RelNode`] indices (`layer ∈ 0..4` × `lane ∈ 0..2`):
/// the side length of [`CodonUnit::best_cai`].
const REL_NODES: usize = 8;

/// One amino acid's synonymous-codon lattice: its prefix-grouped options
/// (with cost-to-go CAI weights), the best-CAI path between any two
/// relative nodes (for the broken-codon bookkeeping), and the highest-CAI
/// whole codon (for traceback over whole codons inside a span).
#[derive(Clone)]
struct CodonUnit {
    options: Vec<UnitOption>,
    /// `best_path[(a, b)] = (max cai-weight sum, nucleotides on that path)`.
    /// Retained for the traceback's `O(span)` interior respelling
    /// ([`Self::best_path_seq`]), which is cold.
    best_path: FxHashMap<(RelNode, RelNode), (f64, Vec<Nucleotide>)>,
    /// Dense `[from][to]` mirror of `best_path`'s weights, indexed by
    /// [`RelNode::index`]; `NaN` marks an unreachable `(from, to)`. The
    /// broken-codon CAI lookup ([`Self::best_path_cai`]) is on the scan's
    /// hottest path (~10⁷ calls/scan via `corner_scores`), so it indexes this
    /// array instead of probing the hash map.
    best_cai: [[f64; REL_NODES]; REL_NODES],
    best_whole_codon: Vec<Nucleotide>,
    /// The adaptation a restriction moved out of the edge weights, `c_i`.
    ///
    /// An unrestricted unit's best path weighs exactly zero by construction —
    /// the argmax codon's weights telescope to `log(g_max / g_max)` — while a
    /// restricted one's best available codon is generally worse than the
    /// family's. A restricted unit subtracts its own best available weight from
    /// every edge, restoring a best of exactly zero over the *available* set,
    /// and records the constant here for the report to add back
    /// ([`Dfa::codon_offset`]). Zero for an unrestricted unit, which is what
    /// keeps every unrestricted path byte-identical.
    ///
    /// With every unit's best at zero, the whole-unit middle term of
    /// [`Dfa::broken_codon_cai`] is exactly zero. The closed form does not rely
    /// on it — the term is summed, not assumed — but the report does: a
    /// restricted design's `cai_log` is the scan's accumulator plus this
    /// constant.
    offset: f64,
}

/// One prefix group (`first`, `second`) of an amino acid's codons: a lane
/// of the codon unit. `thirds` lists the third nucleotides closing each
/// synonymous codon in the group, with that codon's closing CAI weight.
#[derive(Clone)]
struct UnitOption {
    first: Nucleotide,
    second: Nucleotide,
    w_first: f64,
    thirds: Vec<(Nucleotide, f64)>,
}

impl CodonUnit {
    fn best_path_cai(&self, from: RelNode, to: RelNode) -> Option<f64> {
        if from == to {
            return Some(0.0);
        }
        let w = self.best_cai[from.index()][to.index()];
        (!w.is_nan()).then_some(w)
    }

    /// The best weight of a whole codon through this unit, entry to exit.
    fn best_whole_cai(&self) -> f64 {
        self.options
            .iter()
            .flat_map(|o| o.thirds.iter().map(move |&(_, w)| o.w_first + w))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    fn best_path_seq(&self, from: RelNode, to: RelNode) -> Vec<Nucleotide> {
        self.best_path
            .get(&(from, to))
            .map_or_else(Vec::new, |(_, s)| s.clone())
    }

    /// Compile amino acid `aa`'s codon unit: group its codons by
    /// `(first, second)` prefix, assign cost-to-go CAI weights, and
    /// precompute the best-CAI path between relative nodes.
    fn build(aa: usize, usage: &CodonUsage) -> Self {
        Self::from_options(build_options(aa, usage), 0.0)
    }

    /// Compile the precomputed tables over an already-normalized option set.
    /// Shared by the restricted and unrestricted constructors so both derive
    /// every table from the same weights.
    fn from_options(options: Vec<UnitOption>, offset: f64) -> Self {
        let best_whole_codon = best_whole_codon(&options);
        let best_path = build_best_paths(&options);
        let best_cai = dense_cai(&best_path);
        Self {
            options,
            best_path,
            best_cai,
            best_whole_codon,
            offset,
        }
    }

    /// Compile a codon unit over a **subset** of the amino acid's synonymous
    /// codons. CAI weights are the standard ones — the subset restricts which
    /// codons are spellable, not what they are worth — so the optimum over a
    /// restricted lattice is directly comparable to the optimum over the full
    /// one. Returns `None` when the subset is empty (an infeasible position).
    fn restricted(aa: usize, usage: &CodonUsage, allowed: &[[u8; 3]]) -> Option<Self> {
        let mut options = build_options_over(aa, usage, Some(allowed));
        if options.is_empty() {
            return None;
        }
        let offset = renormalize(&mut options);
        Some(Self::from_options(options, offset))
    }

    /// Compile a single-codon unit (one lane, zero CAI weights), pinning
    /// this position to exactly the given codon.
    ///
    /// Its [`CodonUnit::offset`] is zero rather than the codon's adaptation:
    /// this is the substrate for scoring an already-decided sequence, where no
    /// CAI is credited or reported at all. A *restricted* position that happens
    /// to admit one codon goes through [`Self::restricted`] instead, which
    /// zeroes the same edges but records the constant so the report can add it
    /// back.
    fn single(codon: [u8; 3]) -> Self {
        let options = vec![UnitOption {
            first: nuc(codon[0]),
            second: nuc(codon[1]),
            w_first: 0.0,
            thirds: vec![(nuc(codon[2]), 0.0)],
        }];
        Self::from_options(options, 0.0)
    }
}

/// Shift every option's weights so the **best available** codon path is exactly
/// zero again, and return the constant removed.
///
/// A cost-to-go weight is `log(g / g_max)` with `g_max` over the *full*
/// synonym family, so the best path through an unrestricted unit is zero.
/// Restrict the unit and its best available codon is generally worse than the
/// family's; shifting it back to zero keeps the whole-unit middle term of
/// [`Dfa::broken_codon_cai`] exactly zero on every graph.
///
/// The shift lands on `w_first` because each option's third-edge weights are
/// already relative to their own group's best. On an unrestricted unit the
/// shift is zero and nothing moves.
fn renormalize(options: &mut [UnitOption]) -> f64 {
    let best = options
        .iter()
        .map(|o| {
            o.w_first
                + o.thirds
                    .iter()
                    .map(|&(_, w)| w)
                    .fold(f64::NEG_INFINITY, f64::max)
        })
        .fold(f64::NEG_INFINITY, f64::max);
    if best == 0.0 || normalization_skipped() {
        return 0.0;
    }
    for opt in options.iter_mut() {
        opt.w_first -= best;
    }
    best
}

/// Project [`build_best_paths`]'s `(from, to) → (weight, seq)` map onto a
/// dense `[from][to]` weight table keyed by [`RelNode::index`], with `NaN` in
/// every unreachable cell. The weights are copied verbatim, so the dense
/// lookup returns byte-identical values to the map probe it replaces.
fn dense_cai(
    best_path: &FxHashMap<(RelNode, RelNode), (f64, Vec<Nucleotide>)>,
) -> [[f64; REL_NODES]; REL_NODES] {
    let mut table = [[f64::NAN; REL_NODES]; REL_NODES];
    for (&(from, to), &(w, _)) in best_path {
        table[from.index()][to.index()] = w;
    }
    table
}

/// Group amino acid `aa`'s codons into prefix options and assign each
/// edge its cost-to-go CAI log-weight. The first edge of a group carries
/// `log(best-in-group / g_max)`, the second edge carries `0`, and each
/// third edge carries `log(g(codon) / best-in-group)`; summed along a
/// codon these telescope to `log(g(codon) / g_max) = log_cai`.
fn build_options(aa: usize, usage: &CodonUsage) -> Vec<UnitOption> {
    build_options_over(aa, usage, None)
}

/// [`build_options`] restricted to `allowed`, a subset of the amino acid's
/// synonymous codons (`None` = all of them).
///
/// **`g_max` still ranges over the full codon set**, so every edge keeps its
/// standard `log(g / g_max)` weight. Renormalizing to the subset would make the
/// CAI — and therefore the objective — mean something different in each
/// restriction, and objectives from different restrictions must stay comparable
/// for a search that orders subproblems by them.
fn build_options_over(
    aa: usize,
    usage: &CodonUsage,
    allowed: Option<&[[u8; 3]]>,
) -> Vec<UnitOption> {
    let codons = AMINO_ACIDS[aa].codons;
    let g = |ci: usize| usage.fraction(aa, ci).max(EPSILON);
    let g_max = (0..codons.len()).map(g).fold(EPSILON, f64::max);

    let mut options: Vec<UnitOption> = Vec::new();
    for (ci, codon) in codons.iter().enumerate() {
        if allowed.is_some_and(|set| !set.contains(&codon.nuc)) {
            continue;
        }
        let first = nuc(codon.nuc[0]);
        let second = nuc(codon.nuc[1]);
        let third = nuc(codon.nuc[2]);
        let group = options
            .iter_mut()
            .find(|o| o.first == first && o.second == second);
        match group {
            Some(o) => o.thirds.push((third, g(ci))),
            None => options.push(UnitOption {
                first,
                second,
                w_first: 0.0,
                thirds: vec![(third, g(ci))],
            }),
        }
    }

    for opt in &mut options {
        let best_in_group = opt.thirds.iter().map(|&(_, gv)| gv).fold(EPSILON, f64::max);
        opt.w_first = (best_in_group / g_max).ln();
        for (_, w) in &mut opt.thirds {
            *w = (*w / best_in_group).ln();
        }
    }
    options
}

/// The nucleotides of the highest-frequency (highest-CAI) whole codon —
/// the cost-to-go-optimal completion when a whole codon sits inside a
/// broken span.
fn best_whole_codon(options: &[UnitOption]) -> Vec<Nucleotide> {
    let mut best: (f64, Vec<Nucleotide>) = (f64::NEG_INFINITY, Vec::new());
    for opt in options {
        for &(third, w_third) in &opt.thirds {
            let w = opt.w_first + w_third;
            if w > best.0 {
                best = (w, vec![opt.first, opt.second, third]);
            }
        }
    }
    best.1
}

/// Precompute the best-CAI path (weight + nucleotides) between every pair
/// of relative nodes used by the broken-codon bookkeeping: the partial
/// codons (`entry→layer1`, `entry→layer2`, `layer1→exit`, `layer2→exit`)
/// and the within-layer steps needed when a span sits inside one codon.
fn build_best_paths(
    options: &[UnitOption],
) -> FxHashMap<(RelNode, RelNode), (f64, Vec<Nucleotide>)> {
    let mut map: FxHashMap<(RelNode, RelNode), (f64, Vec<Nucleotide>)> = FxHashMap::default();
    let mut relax = |a: RelNode, b: RelNode, w: f64, seq: Vec<Nucleotide>| {
        let entry = map.entry((a, b)).or_insert((f64::NEG_INFINITY, Vec::new()));
        if w > entry.0 {
            *entry = (w, seq);
        }
    };
    for (o, opt) in options.iter().enumerate() {
        let lane = o as u8;
        let l1 = RelNode::layer(1, lane);
        let l2 = RelNode::layer(2, lane);
        // Single-edge steps.
        relax(RelNode::ENTRY, l1, opt.w_first, vec![opt.first]);
        relax(l1, l2, 0.0, vec![opt.second]);
        for &(third, w_third) in &opt.thirds {
            relax(l2, RelNode::EXIT, w_third, vec![third]);
        }
        // Two-edge steps.
        relax(RelNode::ENTRY, l2, opt.w_first, vec![opt.first, opt.second]);
        for &(third, w_third) in &opt.thirds {
            relax(l1, RelNode::EXIT, w_third, vec![opt.second, third]);
        }
    }
    map
}

fn nuc(b: u8) -> Nucleotide {
    match b {
        0 => Nucleotide::A,
        1 => Nucleotide::C,
        2 => Nucleotide::G,
        _ => Nucleotide::U,
    }
}

#[cfg(test)]
mod closed_form_oracle;

#[cfg(test)]
mod tests {
    //! The UTR substrate's graph shape.
    //!
    //! Two floors. A UTR-flanked graph with empty flanks must be
    //! indistinguishable from the CDS-only graph: same domain, same `cds_start`,
    //! and the same broken-codon CAI on every span (the API-level byte-identity
    //! floor lives in `tests/fixed_utr_fold.rs`). And a *designed* UTR position
    //! must offer exactly the four nucleotides at zero CAI — the whole substrate
    //! change behind designed-UTR support, pinned here rather than through the
    //! folding oracle, which cannot see an alphabet the optimum never needs
    //! (`A`/`U` stacks are too weak for a small-case optimum to spell a paired
    //! `U`, so a three-edge node would still fold to the right answer there).

    use super::*;
    use derna_codon_dp::codon_table::aa_to_index;

    fn protein_indices(protein: &str) -> Vec<usize> {
        protein.chars().map(|c| aa_to_index(c).unwrap()).collect()
    }

    /// Every synonymous codon of every position — the unrestricted lattice
    /// written as a restriction, which is what a region of the design space
    /// narrows down from.
    fn every_codon(protein: &[usize]) -> Vec<Vec<[u8; 3]>> {
        protein
            .iter()
            .map(|&aa| AMINO_ACIDS[aa].codons.iter().map(|c| c.nuc).collect())
            .collect()
    }

    /// The codon the usage table likes least at `aa`. Restricting a position to
    /// it is the cheapest way to make an adaptation ceiling strictly negative.
    fn worst_codon(aa: usize, usage: &CodonUsage) -> [u8; 3] {
        let codons = AMINO_ACIDS[aa].codons;
        let ci = (0..codons.len())
            .min_by(|&a, &b| usage.fraction(aa, a).total_cmp(&usage.fraction(aa, b)))
            .expect("a residue has at least one codon");
        codons[ci].nuc
    }

    /// The standalone adaptation ceiling is the restricted lattice's own
    /// normalization constant — the same number from the same weights, so the
    /// two cannot drift apart.
    ///
    /// The case pins one position to its worst synonymous codon, which makes
    /// the constant strictly negative. Over an unrestricted lattice both sides
    /// are zero and the equality would hold whatever either one computed.
    #[test]
    fn the_adaptation_ceiling_is_the_restricted_lattices_own_offset() {
        let usage = CodonUsage::load_default().expect("the canonical human table");
        let protein = protein_indices("LSKF");
        let mut allowed = every_codon(&protein);
        allowed[0] = vec![worst_codon(protein[0], &usage)];

        let ceiling = cai_ceiling(&protein, &usage, &allowed).expect("a non-empty lattice");
        assert!(
            ceiling < 0.0,
            "the restriction has to cost adaptation, or the case is vacuous: {ceiling}"
        );
        let lattice = build_restricted_with_utr(&protein, &usage, &allowed, &[], &[])
            .expect("a non-empty lattice");
        assert_eq!(ceiling, lattice.codon_offset());
    }

    /// An unrestricted lattice gives every position its family's best codon, so
    /// the ceiling is exactly zero — the normalization the closed forms assume.
    #[test]
    fn an_unrestricted_lattice_has_a_zero_adaptation_ceiling() {
        let usage = CodonUsage::load_default().expect("the canonical human table");
        let protein = protein_indices("LSKF");
        let allowed = every_codon(&protein);
        assert_eq!(cai_ceiling(&protein, &usage, &allowed), Some(0.0));
    }

    /// A position left no codon has no ceiling to report, which is the same
    /// emptiness the lattice builder declines to build over.
    #[test]
    fn a_position_left_no_codon_has_no_adaptation_ceiling() {
        let usage = CodonUsage::load_default().expect("the canonical human table");
        let protein = protein_indices("LSKF");
        let mut allowed = every_codon(&protein);
        allowed[1] = Vec::new();

        assert_eq!(cai_ceiling(&protein, &usage, &allowed), None);
        assert!(build_restricted_with_utr(&protein, &usage, &allowed, &[], &[]).is_none());
    }

    #[test]
    fn a_span_through_the_stop_slot_is_spelled_on_its_own_lane() {
        // The stop slot's two lanes, `UA` and `UG`, both leave its first node
        // on a `U`: the case every greedy walk got wrong. Without both edges
        // the test would pass whatever `spell_to` did.
        let usage = CodonUsage::load_default().unwrap();
        let aa = crate::residues("MA*").unwrap();
        let dfa = build(&aa, &usage);
        let slot = dfa.nodes_at(6)[0];
        let u_edges: Vec<Edge> = dfa
            .right(slot)
            .iter()
            .copied()
            .filter(|e| e.nuc as u8 == 3)
            .collect();
        assert_eq!(u_edges.len(), 2, "{u_edges:?}");
        for codon in [[3, 0, 0], [3, 0, 2], [3, 2, 0]] {
            let path = dfa
                .spell_to(slot, &codon, None)
                .expect("a stop codon the slot admits");
            let spelled: Vec<u8> = path.iter().map(|e| e.nuc as u8).collect();
            assert_eq!(spelled, codon);
        }
        // A span that stops after the shared `U` is told apart by its end alone.
        let ends =
            |path: Option<Vec<Edge>>| path.map(|p| p.iter().map(|e| e.to).collect::<Vec<_>>());
        for e in &u_edges {
            assert_eq!(ends(dfa.spell_to(slot, &[3], Some(e.to))), Some(vec![e.to]));
        }
        assert!(dfa.spell_to(slot, &[3, 2], Some(u_edges[0].to)).is_none());
    }

    #[test]
    fn empty_utr_collapses_to_cds_only() {
        let usage = CodonUsage::load_default().unwrap();
        let aa = protein_indices("MARNDCQEGHILKMFPSTWYV");
        let cds = build(&aa, &usage);
        let flanked = build_with_utr(&aa, &usage, &[], &[]);

        assert_eq!(flanked.seq_len, cds.seq_len);
        assert_eq!(flanked.cds_start, 0);
        assert_eq!(flanked.cds_end(), cds.seq_len);
        // Every node's out-edge multiset matches, so the folding domain is
        // identical (the scan reads the graph only through these edges).
        for pos in 0..=cds.seq_len {
            for &node in cds.nodes_at(pos) {
                let edges = |d: &Dfa| {
                    let mut v: Vec<(u8, usize)> = d
                        .right(node)
                        .iter()
                        .map(|e| (e.nuc as u8, d.ix(e.to)))
                        .collect();
                    v.sort_unstable();
                    v
                };
                assert_eq!(
                    edges(&cds),
                    edges(&flanked),
                    "edge set diverges at {node:?}"
                );
            }
        }
        // Broken-codon CAI agrees on a sweep of spans (the hot CAI bookkeeping).
        for a in 0..cds.seq_len {
            for len in 1..=7 {
                let b = a + len;
                if b > cds.seq_len {
                    continue;
                }
                for &from in cds.nodes_at(a) {
                    for &to in cds.nodes_at(b) {
                        assert_eq!(
                            cds.broken_codon_cai(from, to),
                            flanked.broken_codon_cai(from, to),
                            "broken_codon_cai diverges {from:?}→{to:?}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn designed_utr_node_offers_all_four_nucleotides() {
        let usage = CodonUsage::load_default().unwrap();
        let aa = protein_indices("MKTAY");
        let (len5, len3) = (4u32, 3u32);
        let dfa = build_with_utr_chain(
            &aa,
            &usage,
            UtrChain::Designed(len5),
            UtrChain::Designed(len3),
        );
        assert_eq!(dfa.cds_start(), len5);
        assert_eq!(dfa.seq_len, len5 + 3 * aa.len() as u32 + len3);

        let utr_positions = (0..len5).chain(dfa.cds_end()..dfa.seq_len);
        for pos in utr_positions {
            // One node per designed position — the four choices ride the edges,
            // so no lane expansion happens.
            assert_eq!(dfa.nodes_at(pos), [Node::new(pos, 0)], "lanes at {pos}");
            let node = Node::new(pos, 0);
            let mut nucs: Vec<u8> = dfa.right(node).iter().map(|e| e.nuc as u8).collect();
            nucs.sort_unstable();
            assert_eq!(nucs, [0, 1, 2, 3], "designed alphabet at {pos}");
            for e in dfa.right(node) {
                assert_eq!(e.to, Node::new(pos + 1, 0), "successor at {pos}");
                assert_eq!(e.cai, 0.0, "designed UTR edge earns CAI at {pos}");
            }
            // Every opening nucleotide finds a partner here, since all four
            // closing bases are available.
            for open in 0u8..4 {
                assert_eq!(dfa.next_partner(nuc(open), pos), pos, "partner at {pos}");
            }
        }

        // A span between designed positions is always connectable and free.
        assert_eq!(
            dfa.broken_codon_cai(Node::new(0, 0), Node::new(len5, 0)),
            Some(0.0),
        );
    }

    /// A `G`-favouring profile with a unique argmax and two distinct
    /// non-maximal weights, so a `{A, C}` restriction both excludes the argmax
    /// and leaves the two survivors distinguishable.
    const G_FAVOURING: [f64; 4] = [0.15, 0.20, 0.45, 0.20];
    const STRENGTH: f64 = 2.0;

    /// A flank of `len` positions restricted to `{A, C}`.
    fn ac_mask(len: usize) -> Vec<NucSet> {
        vec![NucSet::from_nucleotides(&[Nucleotide::A, Nucleotide::C]).unwrap(); len]
    }

    #[test]
    fn a_restricted_position_offers_only_its_subset() {
        let usage = CodonUsage::load_default().unwrap();
        let aa = protein_indices("MKTAY");
        let mask = ac_mask(4);
        let dfa = build_with_utr_chain(&aa, &usage, UtrChain::Mixed(&mask), UtrChain::Designed(0));
        for pos in 0..4 {
            let mut nucs: Vec<u8> = dfa
                .right(Node::new(pos, 0))
                .iter()
                .map(|e| e.nuc as u8)
                .collect();
            nucs.sort_unstable();
            assert_eq!(nucs, [0, 1], "restricted alphabet at {pos}");
        }
        // The property the restriction exists for, read straight off the
        // next-pair index: neither member of `{A, C}` finds a partner anywhere in
        // a flank drawn from `{A, C}`, so no pair inside it is spellable at all.
        // (`G` and `U` still do — they pair with `C` and `A` — which is why the
        // guarantee is about two flanks over one such subset, not about the
        // flank against the CDS.)
        for open in [Nucleotide::A, Nucleotide::C] {
            assert!(
                dfa.next_partner(open, 0) >= 4,
                "{open:?} found a partner inside an {{A,C}} flank",
            );
        }
    }

    /// The D3 property, stated where it is used: a restricted flank position
    /// is shifted so its best available weight is exactly zero, so a span
    /// across it closes at exactly zero — the same value an unrestricted flank
    /// gives, with no rounding from the closed form's middle term.
    #[test]
    fn a_restricted_span_still_closes_at_zero() {
        let usage = CodonUsage::load_default().unwrap();
        let aa = protein_indices("MKTAY");
        let profile = UtrProfile::new(G_FAVOURING, STRENGTH).unwrap();
        let mask = ac_mask(4);
        let dfa = build_with_utr_profile(
            &aa,
            &usage,
            UtrChain::Mixed(&mask),
            UtrChain::Designed(3),
            profile,
        );
        // The restriction genuinely excludes the argmax, so the case is the one
        // D3 is about rather than a no-op that reads as coverage.
        let best_raw = [Nucleotide::A, Nucleotide::C]
            .map(|nt| profile.weight(nt))
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(best_raw < 0.0, "the restriction must exclude the argmax");

        for pos in 0..4 {
            let best = dfa
                .right(Node::new(pos, 0))
                .iter()
                .map(|e| e.cai)
                .fold(f64::NEG_INFINITY, f64::max);
            assert_eq!(
                best, 0.0,
                "restricted position {pos} does not close at zero"
            );
        }
        assert_eq!(
            dfa.broken_codon_cai(Node::new(0, 0), Node::new(4, 0)),
            Some(0.0),
        );
        // And the constant that re-normalization moved out of the edges is
        // recoverable, since the report has to add it back.
        assert!((dfa.utr_profile_offset(profile) - 4.0 * best_raw).abs() < 1e-12);
    }

    /// The unrestricted flank's offset is exactly zero, which is what leaves
    /// every previously-shipped path byte-identical: the argmax is available at
    /// every designed position, so nothing is subtracted anywhere.
    #[test]
    fn an_unrestricted_flank_needs_no_normalization() {
        let usage = CodonUsage::load_default().unwrap();
        let aa = protein_indices("MKTAY");
        let profile = UtrProfile::new(G_FAVOURING, STRENGTH).unwrap();
        let free = build_with_utr_profile(
            &aa,
            &usage,
            UtrChain::Designed(4),
            UtrChain::Designed(3),
            profile,
        );
        assert_eq!(free.utr_profile_offset(profile), 0.0);
        for pos in (0..4).chain(free.cds_end()..free.seq_len) {
            for e in free.right(Node::new(pos, 0)) {
                assert_eq!(e.cai, profile.weight(e.nuc), "edge reweighted at {pos}");
            }
        }
    }

    /// A pinned position is a given one however it was spelled: one out-edge at
    /// weight zero, no preference to price, and nothing in the offset. This is
    /// what makes an all-singleton mask reduce to `Fixed` exactly.
    #[test]
    fn a_pinned_position_is_a_given_one() {
        let usage = CodonUsage::load_default().unwrap();
        let aa = protein_indices("MKTAY");
        let profile = UtrProfile::new(G_FAVOURING, STRENGTH).unwrap();
        let given = [Nucleotide::A, Nucleotide::C, Nucleotide::U];
        let mask: Vec<NucSet> = given.iter().map(|nt| NucSet::of(*nt)).collect();
        let pinned = build_with_utr_profile(
            &aa,
            &usage,
            UtrChain::Mixed(&mask),
            UtrChain::Designed(0),
            profile,
        );
        let fixed = build_with_utr_profile(
            &aa,
            &usage,
            UtrChain::Fixed(&given),
            UtrChain::Designed(0),
            profile,
        );
        assert_eq!(pinned.utr_profile_offset(profile), 0.0);
        for pos in 0..3 {
            assert!(
                !pinned.is_designed_utr(pos),
                "pinned {pos} reads as designed"
            );
            let edges = |d: &Dfa| {
                d.right(Node::new(pos, 0))
                    .iter()
                    .map(|e| (e.nuc as u8, e.cai))
                    .collect::<Vec<_>>()
            };
            assert_eq!(edges(&pinned), edges(&fixed), "edges diverge at {pos}");
            assert_eq!(edges(&pinned), vec![(given[pos as usize] as u8, 0.0)]);
        }
    }
}
