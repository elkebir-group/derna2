//! The exact Pareto front of (constrained MFE, CAI) over motif-free designs.
//!
//! [`crate::design_front`] folds the front over the plain codon lattice, whose
//! nodes carry no automaton state. Under a forbidden-motif set the feasible
//! designs are the paths of the Aho–Corasick product
//! ([`crate::product::ProductDfa`]), and a base node is no longer enough to
//! say which nucleotides may follow it. This module folds the same
//! front-valued recurrence over the product: every cell is keyed by product
//! nodes (a base node and an automaton state), so every path a cell composes is
//! motif-free by construction.
//!
//! ## The recurrence
//!
//! Positions are nucleotide slots; a product edge spells one slot. A closed
//! pair is keyed by its two edges `(a, b)`: `a` spells the 5′ base and `b` the
//! 3′ base, and the cell holds every non-dominated `(energy, adaptation)` of
//! the structures on that stretch closed by that pair. The remaining tables are
//! keyed by product-node pairs `(u, v)` and cover the stretch between them:
//!
//! - `X(u, v)`: a closed pair spanning exactly `u..v`, with its terminal-AU
//!   correction — the stem as the exterior loop and a multi-loop both see it;
//! - `M(u, v)`: one or more multi-loop branches, the first starting at `u`,
//!   then an unpaired run to `v`;
//! - `M2(u, v)`: two or more branches, the first starting at `u` and the last
//!   ending at `v`;
//! - `T(u, v)`: `M2` followed by an unpaired run to `v`;
//! - `L(u, v)`: an unpaired run from `u`, then `T` — a multi-loop's whole
//!   interior once its closing pair is known.
//!
//! `M` and `M2` are the scalar scan's multi-branch cells, so the scan's
//! outside pass bounds them directly. A branch is added at the node where its
//! stem starts, so a cell's work is over the stems ending at its 3′ end rather
//! than over every node between its ends.
//!
//! The exterior is a front per product node, grown left to right. Loop
//! interiors that do not reach the energy model are filled best-adaptation
//! along product paths, so the AC state threads through them too; the energy
//! terms read only the loop's boundary nucleotides, except the hairpins of 3, 4
//! and 6 unpaired bases, whose whole spelling selects a table bonus and which
//! are enumerated.
//!
//! The energy model is the scalar scan's: Turner 2004 at `dangles = 0`,
//! single-branch loops of at most 30 unpaired bases, unbounded multi-loop and
//! exterior flanks, and the region constraint applied at every closed pair.
//!
//! On its own ([`motif_front_exact`]) the fold keeps every non-dominated point
//! of every cell, and its cost grows quickly with length.
//!
//! ## The walk
//!
//! [`motif_front_walk`] finds the front's hull with scalar folds of the
//! product lattice and solves each gap between neighbouring vertices with the
//! fold, cut to the gap's box. Every cut is a floor on what a cell's
//! completion can add, taken where a motif-free completion cannot do better:
//!
//! - closed pairs and the multi-branch cells `M`, `M2` read the plain
//!   lattice's outside pass (a motif-free completion is a plain one), as the
//!   chord of the two passes on a fixed grid of weights around the gap's tie
//!   weight — a floor, since each value is concave in the weight;
//! - the box's energy reads the plain pass at weight 1, and its adaptation is
//!   exact for any cell, since adaptation does not depend on structure;
//! - `L` and `T` read floors propagated over every cell they feed.
//!
//! A cut may drop a cell only by a floor over every cell built from it; a cut
//! that read only the cell's own direct uses would drop derivations of cells
//! that extend it. A gap whose fold grows too large is solved again in energy
//! slices, unless slicing stops shrinking it, when the slice is folded again
//! with a larger budget. Any cut or slicing returns the same points; they
//! decide only the cost.
//!
//! A walk stopped short returns exact points and the boxes it left open, and
//! [`MotifRun`] resumes from such a result or hands one over after every
//! settled pair, so a run can be continued, or recovered after a kill.

use rna_mfe::loops::{
    bulge_energy, hairpin_energy, hairpin_energy_scalar, internal_energy_scalar,
    pair_from_nucleotides, stack_energy, BulgeLoop, HairpinLoop, InternalLoopScalar, StackPair,
};
use rna_mfe::params::BasePair;
use rna_mfe::{Energy, Nucleotide, Params};
use smallvec::SmallVec;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::ac::{Ac, MotifWindow};
use crate::constrained::outside::AwarePass;
use crate::constrained::{
    AwareChord, AwareSide, BoundCounts, ChainCai, PKey, PairColumn, PairLines, ZeroChord,
};
use crate::coords::Layout;
use crate::dfa::{self, Dfa, Node, NodeLayout};
use crate::fold_pool::Pool;
use crate::front::{Front, Point};
use crate::hash::FxHashMap;
use crate::pins::Pins;
use crate::positional::Infeasible;
use crate::product::ProductDfa;
use crate::region::{PairConstraint, RegionConstraints, Verdict};
use crate::state::PairKey;
use crate::walk;
pub use crate::WalkBudget;
use crate::{rss_kb, BeamError, CodonUsage, FrontPoint, NucSet};

/// No-sharp-turn rule: a hairpin has at least this many unpaired bases.
const MIN_HAIRPIN_LOOP: u32 = 3;
/// The most unpaired bases a stack, bulge or internal loop may hold.
const SINGLE_MAX_LEN: u32 = 30;

/// Entries of [`Fold::loop_floors`]: an internal loop's two closing pairs and
/// its two side lengths, each up to [`SINGLE_MAX_LEN`].
const LOOP_FLOORS: usize =
    BasePair::ALL.len() * BasePair::ALL.len() * (SINGLE_MAX_LEN as usize + 1).pow(2);

/// What a motif-excluded front is asked over: the molecule's flanks, the
/// pair constraint, the forbidden motifs with the window they are enforced
/// over (in whole-molecule positions), and any pinned nucleotides.
pub struct MotifFrontSpec<'a, S: AsRef<str>> {
    /// The 5′ flank as an IUPAC string: `A/C/G/U` (`T` read as `U`) gives a
    /// position, and any other code designs it over that code's nucleotides,
    /// `N` over all four ([`crate::UtrSpec::Mixed`]). Designed positions earn
    /// no CAI, and the motifs are forbidden through them as through the rest.
    pub utr5: &'a str,
    /// The 3′ flank, read as [`Self::utr5`].
    pub utr3: &'a str,
    /// The pairs the structure may not form. Hard constraints only.
    pub constraints: RegionConstraints,
    /// The forbidden motifs, IUPAC allowed, each with the window it is
    /// forbidden inside (in whole-molecule positions). Patterns sharing one
    /// window are enforced over that window exactly as one shared window
    /// always was; patterns with different windows are enforced per pattern
    /// through [`Ac::for_windows`].
    pub motifs: &'a [(S, MotifWindow)],
    /// Pinned nucleotides, in whole-molecule positions, or `None`. A pinned
    /// residue keeps only the synonymous codons that spell its pins; a pin on
    /// a flank must agree with the flank.
    pub pins: Option<&'a Pins>,
}

/// A flank as the walk reads it: given nucleotides, or positions designed over
/// a nucleotide set each.
enum Flank {
    Fixed(Vec<Nucleotide>),
    Designed(Vec<NucSet>),
}

impl Flank {
    /// `s` read as IUPAC: given when every position names one nucleotide.
    fn parse(s: &str) -> Result<Self, BeamError> {
        let masks = crate::parse_utr_mask(s)?;
        let one = |m: &NucSet| {
            let mut it = m.iter();
            it.next().filter(|_| it.next().is_none())
        };
        Ok(match masks.iter().map(one).collect::<Option<Vec<_>>>() {
            Some(nucleotides) => Self::Fixed(nucleotides),
            None => Self::Designed(masks),
        })
    }

    fn chain(&self) -> dfa::UtrChain<'_> {
        match self {
            Self::Fixed(n) => dfa::UtrChain::Fixed(n),
            Self::Designed(m) => dfa::UtrChain::Mixed(m),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Fixed(n) => n.len(),
            Self::Designed(m) => m.len(),
        }
    }

    /// Each position's nucleotides.
    fn masks(&self) -> Vec<NucSet> {
        match self {
            Self::Fixed(n) => n.iter().map(|&n| NucSet::of(n)).collect(),
            Self::Designed(m) => m.clone(),
        }
    }
}

/// The codon lattice a spec is folded over: the plain one, or the one its pins
/// restrict. A restricted residue's best remaining codon sits at adaptation
/// zero, and [`Dfa::codon_offset`] carries what the restriction removed, so
/// every point adds it back.
fn spec_graph<S: AsRef<str>>(
    aa: &[usize],
    usage: &CodonUsage,
    spec: &MotifFrontSpec<'_, S>,
    five: &Flank,
    three: &Flank,
) -> Result<Dfa, BeamError> {
    let build = |allowed: Option<&[Vec<[u8; 3]>]>| {
        dfa::build_composite(
            aa,
            usage,
            allowed,
            five.chain(),
            three.chain(),
            crate::UtrProfile::inert(),
        )
    };
    let Some(pins) = spec.pins else {
        return Ok(build(None).expect("an unrestricted lattice"));
    };
    let layout = Layout::new(five.len() as u32, aa.len() as u32, three.len() as u32);
    for (flank, base) in [(five, 0), (three, layout.cds_end())] {
        pins.flank_mask(&layout, base, &flank.masks())
            .map_err(Infeasible::Pins)?;
    }
    let allowed = pins.cds_codon_sets(&layout, aa).map_err(Infeasible::Pins)?;
    Ok(build(allowed.as_deref()).expect("cds_codon_sets leaves every residue at least one codon"))
}

/// The exact front of (constrained MFE, coding CAI) over the designs that spell
/// no forbidden motif inside the window, most stable first. Each point's
/// sequence is the whole molecule, flanks included.
///
/// # Errors
///
/// [`BeamError::Infeasible`] when every encoding spells a motif;
/// [`BeamError::SoftRegionUnsupported`] for a constraint that penalizes rather
/// than forbids; otherwise a malformed protein, flank or motif.
pub fn motif_front_exact<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    spec: &MotifFrontSpec<'_, S>,
) -> Result<Vec<FrontPoint>, BeamError> {
    motif_front_within(params, protein, usage, spec, None)
}

/// [`motif_front_exact`] restricted to the designs whose energy is at most
/// `ceiling`: exactly the points of the full front at or below it.
///
/// # Errors
///
/// As [`motif_front_exact`].
pub fn motif_front_within<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    spec: &MotifFrontSpec<'_, S>,
    ceiling: Option<Energy>,
) -> Result<Vec<FrontPoint>, BeamError> {
    motif_front_tallied(params, protein, usage, spec, ceiling).map(|(points, _)| points)
}

/// How many loops of each kind the returned designs' structures hold, summed
/// over the front: what a test reads to say which rules its cells exercise.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuleTally {
    /// Hairpins scored from their two boundary nucleotides.
    pub hairpins: usize,
    /// Hairpins of 3, 4 or 6 unpaired bases, scored from their whole spelling.
    pub spelled_hairpins: usize,
    /// Stacked pairs.
    pub stacks: usize,
    /// Bulges.
    pub bulges: usize,
    /// Internal loops.
    pub internals: usize,
    /// Multi-loops.
    pub multis: usize,
    /// Stems on the exterior loop.
    pub exterior_stems: usize,
}

/// [`motif_front_within`] with the [`RuleTally`] of the returned designs.
///
/// # Errors
///
/// As [`motif_front_exact`].
pub fn motif_front_tallied<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    spec: &MotifFrontSpec<'_, S>,
    ceiling: Option<Energy>,
) -> Result<(Vec<FrontPoint>, RuleTally), BeamError> {
    if !spec.constraints.is_hard() {
        return Err(BeamError::SoftRegionUnsupported);
    }
    let aa = crate::residues(protein)?;
    let five = Flank::parse(spec.utr5)?;
    let three = Flank::parse(spec.utr3)?;
    let graph = spec_graph(&aa, usage, spec, &five, &three)?;
    let region = PairConstraint::new(&spec.constraints, &graph);
    let (ac, scope) = Ac::for_windows(spec.motifs)?;
    let product = ProductDfa::build_scoped(&graph, &ac, scope);
    if !product.is_feasible() {
        return Err(BeamError::Infeasible);
    }
    let product = product.merge_equivalent();
    let lattice = Lattice::new(&product);
    let fold = Fold::run(
        params,
        &lattice,
        &region,
        ceiling,
        ([None; 4], None, &[]),
        (None, Stop::default(), 1),
    );
    let m = derna_codon_dp::cai::sense_codons(&aa);
    let offset = graph.codon_offset();
    let mut tally = RuleTally::default();
    let points = fold
        .recover(&mut tally)
        .into_iter()
        .map(|(mfe, cai, sequence, structure)| {
            let cai_log = cai + offset;
            FrontPoint {
                sequence,
                structure,
                mfe_kcal_mol: mfe.to_kcal_mol(),
                cai_log,
                cai: if m == 0 {
                    0.0
                } else {
                    (cai_log / m as f64).exp()
                },
                utr_profile_log: 0.0,
            }
        })
        .collect();
    Ok((points, tally))
}

/// One product edge: it spells `nuc` at its source's position.
#[derive(Clone, Copy)]
struct PEdge {
    from: u32,
    to: u32,
    nuc: Nucleotide,
    cai: f64,
}

/// The product lattice flattened to dense node and edge ids.
struct Lattice {
    /// The base lattice's dense node layout.
    layout: NodeLayout,
    /// Each node's position.
    pos: Vec<u32>,
    /// The nodes at each position, `0..=seq_len`.
    at: Vec<Vec<u32>>,
    edges: Vec<PEdge>,
    /// Out- and in-edge ids per node.
    out: Vec<Vec<u32>>,
    inn: Vec<Vec<u32>>,
    start: u32,
    ends: Vec<u32>,
    seq_len: u32,
    /// Each node's base-lattice node, for the plain lattice's tables.
    base: Vec<Node>,
    /// Each node's automaton states: several where equivalent states were
    /// merged into one node.
    states: Vec<SmallVec<[u32; 2]>>,
    /// `fill[u * n + v]`: the best adaptation over paths from `u` to `v`,
    /// `-∞` when none.
    fill: Vec<f64>,
    /// The best adaptation from the start to each node, and from each node
    /// to the end.
    prefix: Vec<f64>,
    suffix: Vec<f64>,
}

impl Lattice {
    fn new(product: &ProductDfa<'_>) -> Self {
        let dfa: &Dfa = product.dfa();
        let ac = product.lattice_ac();
        let n = product.num_product_nodes();
        let mut pos = vec![0; n];
        let mut base = vec![dfa.start(); n];
        let mut states: Vec<SmallVec<[u32; 2]>> = vec![SmallVec::new(); n];
        let mut at = vec![Vec::new(); dfa.seq_len as usize + 1];
        let mut edges = Vec::new();
        let mut out = vec![Vec::new(); n];
        let mut inn = vec![Vec::new(); n];
        for p in 0..=dfa.seq_len {
            for &node in dfa.nodes_at(p) {
                for &q in product.reachable_states(node) {
                    let id = product.product_id(node, q).expect("a reachable state") as u32;
                    states[id as usize].push(q);
                    pos[id as usize] = p;
                    base[id as usize] = node;
                    at[p as usize].push(id);
                    for &e in dfa.right(node) {
                        let Some(next) = ac.step_edge(node, q, e) else {
                            continue;
                        };
                        let Some(to) = product.product_id(e.to, next) else {
                            continue;
                        };
                        let k = edges.len() as u32;
                        edges.push(PEdge {
                            from: id,
                            to: to as u32,
                            nuc: e.nuc,
                            cai: e.cai,
                        });
                        out[id as usize].push(k);
                        inn[to].push(k);
                    }
                }
            }
        }
        let start = product
            .product_id(dfa.start(), product.ac().start())
            .expect("the start is reachable") as u32;
        let ends = product
            .reachable_states(dfa.end())
            .iter()
            .map(|&q| product.product_id(dfa.end(), q).expect("a reachable state") as u32)
            .collect();
        let mut lattice = Self {
            layout: dfa.layout(),
            pos,
            at,
            edges,
            out,
            inn,
            start,
            ends,
            seq_len: dfa.seq_len,
            base,
            states,
            fill: Vec::new(),
            prefix: Vec::new(),
            suffix: Vec::new(),
        };
        lattice.fill_paths();
        lattice.prefix = (0..n as u32)
            .map(|u| lattice.fill_at(lattice.start, u))
            .collect();
        lattice.suffix = (0..n as u32)
            .map(|u| {
                lattice
                    .ends
                    .iter()
                    .map(|&e| lattice.fill_at(u, e))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        lattice
    }

    fn n(&self) -> usize {
        self.pos.len()
    }

    fn fill_at(&self, u: u32, v: u32) -> f64 {
        self.fill[u as usize * self.n() + v as usize]
    }

    /// The best adaptation between every ordered pair of nodes, swept from the
    /// 3′ end so each entry is one step from entries already final. Every fold
    /// over the lattice reads it, so it is built once.
    fn fill_paths(&mut self) {
        let n = self.n();
        let mut fill = vec![f64::NEG_INFINITY; n * n];
        for p in (0..=self.seq_len).rev() {
            for &u in &self.at[p as usize] {
                let row = u as usize * n;
                fill[row + u as usize] = 0.0;
                for &e in &self.out[u as usize] {
                    let edge = self.edge(e);
                    let next = edge.to as usize * n;
                    for v in 0..n {
                        let rest = fill[next + v];
                        if rest == f64::NEG_INFINITY {
                            continue;
                        }
                        let cand = edge.cai + rest;
                        if cand > fill[row + v] {
                            fill[row + v] = cand;
                        }
                    }
                }
            }
        }
        self.fill = fill;
    }

    fn edge(&self, e: u32) -> PEdge {
        self.edges[e as usize]
    }

    /// The plain lattice's key for the pair `(ea, eb)`, with the column it
    /// closes into.
    fn pair_key(&self, ea: PEdge, eb: PEdge) -> (usize, PairKey) {
        let base = &self.base;
        (
            self.layout.ix(base[eb.to as usize]),
            PairKey {
                open: base[ea.from as usize],
                close: base[eb.from as usize],
                opening: crate::state::Opening::new(ea.nuc, base[ea.to as usize].lane),
                nuc_close: eb.nuc,
            },
        )
    }

    fn pos_of_edge(&self, e: u32) -> u32 {
        self.pos[self.edge(e).from as usize]
    }
}

fn terminal_au(params: &Params, pair: BasePair) -> Energy {
    if pair.applies_terminal_au() {
        Energy(params.misc.terminal_au_37)
    } else {
        Energy::ZERO
    }
}

/// How a front point was derived: the rule and the child points it composed,
/// by index into each child's final front.
#[derive(Clone, Copy)]
enum Prov {
    Start,
    /// A hairpin whose interior is filled best-adaptation between the two
    /// boundary nucleotides.
    Hairpin {
        first: u8,
        last: u8,
    },
    /// A hairpin whose interior spelling is stored whole, entry `id` of its
    /// closing pair's list.
    HairpinSpelled {
        id: u32,
    },
    /// A stack, bulge or internal loop around the inner pair `(c, d)`, point
    /// `k`, with the corner nucleotides of each non-empty flank: `(c, d)` as
    /// [`pack_ids`] and the corners as [`pack_corners`].
    Single {
        cd: u32,
        cd_hi: u16,
        corners: u8,
        k: u32,
    },
    /// A multi-loop closed around `T(to a, from b)`, point `k`.
    Multi {
        k: u32,
    },
    /// `X(u, v)` from the pair `(a, b)`, point `k`, the pair as
    /// [`pack_ids`].
    XPair {
        ab: u32,
        ab_hi: u16,
        k: u32,
    },
    /// `M(u, v)` from the branch `X(u, v)`, point `k`.
    MBranch {
        k: u32,
    },
    /// `M(u, v)` from `M(u, from e)`, point `k`, and the unpaired edge `e`.
    MUnpaired {
        e: u32,
        k: u32,
    },
    /// `M(u, v)` from `M2(u, v)`, point `k`.
    MFromM2 {
        k: u32,
    },
    /// `M2(u, v)` from `M(u, w)`, point `l`, and the branch `X(w, v)`, point
    /// `r`, the node `w` as [`pack_id`].
    Split {
        w: u16,
        w_hi: u8,
        l: u32,
        r: u32,
    },
    /// `L(u, v)` from `T(u, v)`, point `k`.
    LCore {
        k: u32,
    },
    /// `L(u, v)` from the unpaired edge `e` and `L(to e, v)`, point `k`.
    LUnpaired {
        e: u32,
        k: u32,
    },
    /// `T(u, v)` from `M2(u, v)`, point `k`.
    TCore {
        k: u32,
    },
    /// `T(u, v)` from `T(u, from e)`, point `k`, and the unpaired edge `e`.
    TUnpaired {
        e: u32,
        k: u32,
    },
    /// The exterior grows by the unpaired edge `e` from `Z(from e)`, point `k`.
    ZUnpaired {
        e: u32,
        k: u32,
    },
    /// The exterior absorbs `X(x, v)`, point `r`, after `Z(x)`, point `l`,
    /// the node `x` as [`pack_id`].
    ZPair {
        x: u16,
        x_hi: u8,
        l: u32,
        r: u32,
    },
    /// The front at the terminal node `v`, point `k`.
    End {
        v: u32,
        k: u32,
    },
}

// A fold stores hundreds of millions of points at 1000 residues, so a point
// is held to 24 bytes: its energy, its adaptation and a 12-byte `Prov`.
const _: () = assert!(std::mem::size_of::<Prov>() == 12);
const _: () = assert!(std::mem::size_of::<Point<Prov>>() == 24);

/// Bits of an edge or node index a [`Prov`] holds.
const ID_BITS: u32 = 24;

/// A node index in a [`Prov`]: its low 16 bits and the 8 above.
fn pack_id(id: u32) -> (u16, u8) {
    debug_assert!(id < 1 << ID_BITS);
    (id as u16, (id >> 16) as u8)
}

fn unpack_id(lo: u16, hi: u8) -> u32 {
    u32::from(lo) | u32::from(hi) << 16
}

/// Two edge indexes in a [`Prov`]: `a` and the low 8 bits of `b` in one
/// word, and the 16 bits of `b` above those.
fn pack_ids(a: u32, b: u32) -> (u32, u16) {
    debug_assert!(a < 1 << ID_BITS && b < 1 << ID_BITS);
    (a | b << ID_BITS, (b >> (32 - ID_BITS)) as u16)
}

fn unpack_ids(lo: u32, hi: u16) -> (u32, u32) {
    let a = lo & ((1 << ID_BITS) - 1);
    (a, lo >> ID_BITS | u32::from(hi) << (32 - ID_BITS))
}

/// A single loop's four corner nucleotides, two bits each.
fn pack_corners(corners: [u8; 4]) -> u8 {
    debug_assert!(corners.iter().all(|&c| c < 4));
    corners
        .iter()
        .enumerate()
        .fold(0, |acc, (i, &c)| acc | c << (2 * i))
}

fn unpack_corners(bits: u8) -> [u8; 4] {
    std::array::from_fn(|i| bits >> (2 * i) & 3)
}

type Cells = Store<Front<Prov>>;

/// Values keyed by an ordered pair of ids, each held at a slot numbered in
/// insertion order, so that a caller holding a slot reads its value with no
/// lookup. Keys over a lattice's nodes are indexed densely at `u * n + v`, a
/// lookup in a fold's inner loops being one array read; other keys are
/// hashed.
#[derive(Clone)]
struct Store<T> {
    /// One more than each key's slot at `u * n + v`, `0` where absent;
    /// empty when the keys are hashed.
    dense: Vec<u32>,
    n: usize,
    hashed: FxHashMap<(u32, u32), u32>,
    keys: Vec<(u32, u32)>,
    items: Vec<T>,
}

impl<T> Default for Store<T> {
    fn default() -> Self {
        Self {
            dense: Vec::new(),
            n: 0,
            hashed: FxHashMap::default(),
            keys: Vec::new(),
            items: Vec::new(),
        }
    }
}

impl<T> Store<T> {
    /// A store keyed by pairs of ids below `n`, indexed densely.
    fn dense(n: usize) -> Self {
        Self {
            dense: vec![0; n * n],
            n,
            ..Self::default()
        }
    }

    fn slot(&self, (u, v): (u32, u32)) -> Option<u32> {
        if self.n == 0 {
            return self.hashed.get(&(u, v)).copied();
        }
        self.dense[u as usize * self.n + v as usize].checked_sub(1)
    }

    fn get(&self, key: &(u32, u32)) -> Option<&T> {
        self.slot(*key).map(|s| &self.items[s as usize])
    }

    /// The value at `slot`, as [`Self::insert`] returned it.
    fn at(&self, slot: u32) -> &T {
        &self.items[slot as usize]
    }

    /// Store `value` under `key`, replacing any value there, and return its
    /// slot.
    fn insert(&mut self, key: (u32, u32), value: T) -> u32 {
        if let Some(s) = self.slot(key) {
            self.items[s as usize] = value;
            return s;
        }
        let s = self.items.len() as u32;
        if self.n == 0 {
            self.hashed.insert(key, s);
        } else {
            self.dense[key.0 as usize * self.n + key.1 as usize] = s + 1;
        }
        self.keys.push(key);
        self.items.push(value);
        s
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn values(&self) -> std::slice::Iter<'_, T> {
        self.items.iter()
    }

    #[cfg(test)]
    fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }
}

impl<T> std::ops::Index<&(u32, u32)> for Store<T> {
    type Output = T;

    fn index(&self, key: &(u32, u32)) -> &T {
        self.get(key).expect("a stored key")
    }
}

impl<'s, T> IntoIterator for &'s Store<T> {
    type Item = (&'s (u32, u32), &'s T);
    type IntoIter = std::iter::Zip<std::slice::Iter<'s, (u32, u32)>, std::slice::Iter<'s, T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.keys.iter().zip(&self.items)
    }
}

/// Corner nucleotides of a flank, and the best adaptation over its fills.
type Corners = SmallVec<[(u8, u8, f64); 8]>;

/// A staircase's least score at each of a fold's cut weights, as
/// `(weight, score)`.
type Lows = SmallVec<[(f64, f64); 2]>;

/// A closed pair with a non-empty cell, kept by its 5′ position: its 3′
/// position, the pair, its cell's slot in the fold's `P` store, its cell's
/// best adaptation, and its cell's least score at each cut's weight.
type PairAt = (u32, (u32, u32), u32, f64, Lows);

/// A hairpin interior stored whole, as its edges.
type Spelling = SmallVec<[u32; 6]>;

/// The cuts a fold holds every cell to, each with its floors under the `L`
/// and `T` cells.
type Cuts<'a> = [Option<(&'a Cut<'a>, &'a Floors)>; 4];

/// A box of a narrowed walk: its two ends as `(mfe, cai)` and whether it
/// keeps the wider walk's probe and rest.
type NarrowBox = ((f64, f64), (f64, f64), bool);

struct Fold<'a> {
    params: &'a Params,
    l: &'a Lattice,
    region: &'a PairConstraint,
    ceiling: Option<Energy>,
    /// The cuts every cell is held to: the tie-weight cut through a gap's
    /// box, and the energy cut at weight 1 under the box's top energy, each
    /// with its floors under the `L` and `T` cells.
    cuts: Cuts<'a>,
    /// The split loops' rule, read once so that every cell of the fold is
    /// stored and read under the same one.
    prune: SplitPrune,
    gap_box: Option<GapBox>,
    /// Further floors under closed pairs at the tie-weight cut's weight,
    /// from the exact motif-free adaptation at weight 0 (see [`ZeroChord`]).
    zero: &'a [ZeroChord<'a>],
    corners: Store<Corners>,
    /// The least energy of each internal loop over its four boundary
    /// nucleotides, by closing pairs and side lengths, found the first time
    /// a loop of that shape is met (see [`Self::loop_floor`]).
    loop_floors: Vec<std::sync::OnceLock<Energy>>,
    /// The hairpin interiors stored whole, by closing pair; a
    /// [`Prov::HairpinSpelled`] point indexes its pair's list.
    spelled: FxHashMap<(u32, u32), Vec<Spelling>>,
    p: Cells,
    /// The closed pairs with a non-empty cell, by 5′ position, each with
    /// its 3′ position and its cell's least score at each cut's weight.
    /// Spans are stored shortest first, so each list rises in 3′ position.
    p_at: Vec<Vec<PairAt>>,
    x: Cells,
    /// Each `X` cell's points in score order at the cuts' weights, by the
    /// cell's slot, under the `score` split rule, sorted the first time a
    /// split reads them: most stored cells never meet a split.
    x_score: Vec<std::sync::OnceLock<ByScore>>,
    /// The same orders of each `M` cell, by its slot.
    m_score: Vec<std::sync::OnceLock<ByScore>>,
    /// Each `X` and `M` cell's best adaptation and least score at each
    /// cut's weight, by slot, beside its orders: what a split must clear
    /// before either order is read.
    x_low: Vec<(f64, Lows)>,
    m_low: Vec<(f64, Lows)>,
    z_score: Vec<Option<ByScore>>,
    /// The 5′ nodes of the non-empty `X` cells, by 3′ node, each with its
    /// cell's slot.
    x_into: Vec<Vec<(u32, u32)>>,
    m: Cells,
    lead: Cells,
    m2: Cells,
    t: Cells,
    z: Vec<Front<Prov>>,
    end: Front<Prov>,
    /// Seconds per phase, read by the census: closed pairs, of which the
    /// single-branch loops, the multi-loop tables, the exterior.
    clock: [f64; 4],
    /// What the constrained-lattice chords alone ruled out, as [`CellOut`]
    /// counts it, over every cell stored.
    aware: [usize; 3],
    /// Points stored so far, and the most the fold may store before it
    /// gives up (`aborted`); `usize::MAX` for no limit.
    stored: usize,
    budget: usize,
    /// Threads filling the cells of one span.
    threads: usize,
    aborted: bool,
    /// The span, and the index among its cells, at which filling continues.
    next: (u32, usize),
    /// Points stored by the end of each span filled, in span order.
    filled: Vec<usize>,
    /// What stops the fold short, whether it did, and the cells stored since
    /// memory was last read.
    stop: Stop<'a>,
    stopped: bool,
    /// Whether the stop was the gap's own time cap ([`Stop::gap_deadline`])
    /// rather than the walk's deadline or memory budget.
    capped: bool,
    unread: usize,
    /// Points adopted from a fold that gave up ([`Fold::adopt`]).
    adopted: usize,
    /// The `M2` and exterior split points, under the split census ([`split_census`]).
    split: [SplitTally; 2],
}

/// What stops a gap's folds short: the walk's deadline, read after every
/// cell, and its memory budget, read before every span and every
/// [`MEMORY_EVERY`] cells, and both before every slice of a gap.
#[derive(Clone, Copy, Default)]
struct Stop<'a> {
    /// Whether the walk's deadline has passed.
    deadline: Option<&'a (dyn Fn() -> bool + Sync)>,
    /// Whether the walk's memory budget is spent.
    memory: Option<&'a (dyn Fn() -> bool + Sync)>,
    /// The gap folds that solved their range so far, and how many may.
    folds: Option<(&'a std::sync::atomic::AtomicUsize, usize)>,
    /// The seconds one gap's fold may take before it moves on, read between
    /// its slices ([`solve_gap`]), with which visit to the gap this is.
    gap_secs: Option<walk::GapCap>,
    /// When the gap's turn ends, read after every cell like the walk's
    /// deadline, so the cap stops a fold in flight rather than only between
    /// slices: a slice that outlasts the cap, the gap's first above all,
    /// would otherwise hold the walk's other boxes for as long as it runs.
    /// Set by [`solve_gap`] unless [`gap_midslice`] says otherwise.
    gap_deadline: Option<std::time::Instant>,
}

impl Stop<'_> {
    /// Whether the count of gap folds that solved their range is spent.
    fn folds_spent(&self) -> bool {
        self.folds
            .is_some_and(|(run, n)| run.load(std::sync::atomic::Ordering::Relaxed) >= n)
    }

    /// Counts one more gap fold that solved its range.
    fn fold_solved(&self) {
        if let Some((run, _)) = self.folds {
            run.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn timed_out(&self) -> bool {
        self.deadline.is_some_and(|passed| passed())
    }

    /// Whether the gap's own turn is over ([`Stop::gap_deadline`]).
    fn gap_due(&self) -> bool {
        self.gap_deadline
            .is_some_and(|due| std::time::Instant::now() >= due)
    }

    fn out_of_memory(&self) -> bool {
        self.memory.is_some_and(|spent| spent())
    }
}

/// Give the cell at `slot` an unsorted score order, the next one or in
/// place of the one a replaced cell had.
fn fresh_order(orders: &mut Vec<std::sync::OnceLock<ByScore>>, slot: u32) {
    match orders.get_mut(slot as usize) {
        Some(order) => *order = std::sync::OnceLock::new(),
        None => {
            debug_assert_eq!(orders.len(), slot as usize, "orders follow slots");
            orders.push(std::sync::OnceLock::new());
        }
    }
}

/// Store a cell's best adaptation and least scores at `slot`, as
/// [`fresh_order`] keeps its orders.
fn fresh_low(lows: &mut Vec<(f64, Lows)>, slot: u32, low: (f64, Lows)) {
    match lows.get_mut(slot as usize) {
        Some(at) => *at = low,
        None => {
            debug_assert_eq!(lows.len(), slot as usize, "lows follow slots");
            lows.push(low);
        }
    }
}

/// A front's best adaptation.
fn best_cai(front: &Front<Prov>) -> f64 {
    front
        .points()
        .iter()
        .map(|p| p.cai)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Cells stored between two reads of the memory budget inside a fold.
const MEMORY_EVERY: usize = 4096;

/// The cells over one `u..v`, computed from cells already final and stored
/// by [`Fold::commit`].
#[derive(Default)]
struct CellOut {
    /// Closed pairs, in the order they were found.
    p: Vec<((u32, u32), Front<Prov>)>,
    spelled: Vec<((u32, u32), Vec<Spelling>)>,
    x: Option<Front<Prov>>,
    m: Option<Front<Prov>>,
    m2: Option<Front<Prov>>,
    t: Option<Front<Prov>>,
    lead: Option<Front<Prov>>,
    /// Flank corners computed for this cell and not yet in the fold's memo.
    corners: FxHashMap<(u32, u32), Corners>,
    /// Seconds in closed pairs, single-branch loops and multi-loop tables.
    clock: [f64; 3],
    /// Closed pairs the constrained-lattice chords alone ruled out, points
    /// their limits alone dropped from a pair's front, and pairs no chord
    /// could read because a node held more than one automaton state.
    aware: [usize; 3],
    /// The `M2` split points, under the split census ([`split_census`]).
    split: SplitTally,
}

/// Cells of one span filled per hand-out to the threads; the budget and the
/// deadline are checked after each cell as it is stored.
const CELLS_PER_THREAD: usize = 16;

fn canonical(a: Nucleotide, b: Nucleotide) -> Option<BasePair> {
    match pair_from_nucleotides(a, b) {
        BasePair::NN => None,
        pair => Some(pair),
    }
}

fn nuc_of(i: u8) -> Nucleotide {
    match i {
        0 => Nucleotide::A,
        1 => Nucleotide::C,
        2 => Nucleotide::G,
        _ => Nucleotide::U,
    }
}

/// Merge the staircase `add` into the staircase `acc`, keeping exactly the
/// points [`Front::merge`] keeps. A cell is gathered this way, one sorted
/// batch at a time, in place of sorting every candidate at once.
fn merge_into(acc: &mut Vec<Point<Prov>>, add: impl Iterator<Item = Point<Prov>>) {
    let old = std::mem::take(acc);
    let mut add = add.peekable();
    let mut old_iter = old.iter().copied().peekable();
    let mut out: Vec<Point<Prov>> = Vec::with_capacity(old.len() + add.size_hint().0);
    loop {
        let take_old = match (old_iter.peek(), add.peek()) {
            (Some(l), Some(r)) => (l.mfe.0, r.cai) <= (r.mfe.0, l.cai),
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        let next = if take_old {
            old_iter.next()
        } else {
            add.next()
        }
        .expect("a point");
        match out.last() {
            Some(last) if last.mfe <= next.mfe && last.cai >= next.cai - CAI_EPS => {}
            _ => out.push(next),
        }
    }
    *acc = out;
}

/// Log-adaptation within which two points count as equal, as in
/// [`crate::front`].
const CAI_EPS: f64 = 1e-9;

/// What a point must clear to survive the cuts its cell is held to once
/// gathered: an adaptation floor and, per cut, a most the point may score at
/// the cut's weight. Each is read a little looser than the cut itself.
///
/// A point a cut drops is dropped with every point it dominates, since its
/// score at any weight is no higher than theirs. So a staircase gathered from
/// the admitted points alone, then cut, is the staircase gathered from every
/// point, then cut: a point the gathering drops is dominated by one kept or by
/// one the cut drops too.
#[derive(Clone)]
struct Admit {
    c_min: f64,
    /// Each cut's weight, and the most a point may score there.
    lines: SmallVec<[(f64, f64); 4]>,
}

impl Admit {
    /// Admits every point.
    fn all() -> Self {
        Self {
            c_min: f64::NEG_INFINITY,
            lines: SmallVec::new(),
        }
    }

    /// Also hold points to score at most `limit` at `lambda`.
    fn line(&mut self, lambda: f64, limit: f64) {
        self.lines.push((lambda, limit + ADMIT_SLACK));
    }

    fn admits(&self, mfe: Energy, cai: f64) -> bool {
        cai >= self.c_min && self.lines.iter().all(|&(l, at)| brd(mfe, cai, l) <= at)
    }

    /// Whether no point of a staircase whose best adaptation is `cai` and
    /// whose least scores are `lows` is admitted once shifted by `(de, dc)`.
    /// Scores add under a shift, so each line's least shifted score is the
    /// staircase's least score plus the shift's own.
    fn rules_out(&self, (cai, lows): (f64, &Lows), (de, dc): (Energy, f64)) -> bool {
        cai + dc < self.c_min
            || self.lines.iter().any(|&(l, at)| {
                lows.iter()
                    .any(|&(w, low)| w == l && low + brd(de, dc, l) > at + RULE_OUT_SLACK)
            })
    }

    /// Whether no sum of a point of a staircase whose best adaptation and
    /// least scores are `left` and a point of one whose are `right`, shifted
    /// by `(de, dc)`, is admitted, read in the forms [`ByScore::rows`] and
    /// [`Self::admits`] read them, so that it rules out exactly the splits
    /// whose rows they leave empty. Rounding is monotone, so the best sum's
    /// adaptation is the best adaptations' sum, added in the order a sum is.
    /// A line's test is the negation of the rows' own `<=`, so that a split
    /// is ruled out exactly when that comparison fails.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn rules_out_sums(
        &self,
        (left_cai, left): (f64, &Lows),
        (right_cai, right): (f64, &Lows),
        (de, dc): (Energy, f64),
    ) -> bool {
        right_cai + (left_cai + dc) < self.c_min
            || self.lines.iter().any(|&(l, at)| {
                let low = |lows: &Lows| lows.iter().find(|&&(w, _)| w == l).map(|&(_, low)| low);
                low(left).zip(low(right)).is_some_and(|(left, right)| {
                    !(left <= at - right - brd(de, dc, l) + 2.0 * RULE_OUT_SLACK)
                })
            })
    }

    /// The first point of `front` that, shifted by `dc`, clears the
    /// adaptation floor: a staircase's adaptation rises with its energy.
    fn first(&self, front: &[Point<Prov>], dc: f64) -> usize {
        front.partition_point(|p| p.cai + dc < self.c_min)
    }
}

/// How much looser [`Admit`] reads each cut than the cut itself: more than
/// the rounding between the two forms of one comparison, and more than the
/// adaptation by which [`merge_into`] lets one point dominate another.
const ADMIT_SLACK: f64 = 1e-6;

/// How far past a line [`Admit::rules_out`] needs a staircase's least
/// shifted score: more than the rounding between a sum's score and the sum of
/// its terms' scores, so that it rules out only what [`Admit::admits`] drops.
const RULE_OUT_SLACK: f64 = 1e-7;

/// Merge the points of `front` shifted by `(de, dc)` that `admit` admits into
/// `acc`, recording each point's index through `back`.
fn merge_shift(
    acc: &mut Vec<Point<Prov>>,
    front: &Front<Prov>,
    (de, dc): (Energy, f64),
    admit: &Admit,
    back: impl Fn(u32) -> Prov,
) {
    let points = front.points();
    let first = admit.first(points, dc);
    merge_into(
        acc,
        points[first..]
            .iter()
            .enumerate()
            .map(|(k, p)| Point::new(p.mfe + de, p.cai + dc, back((first + k) as u32)))
            .filter(|p| admit.admits(p.mfe, p.cai)),
    );
}

/// Points gathered before they are merged into a cell, at most.
const GATHER_BATCH: usize = 1 << 16;

/// Merge the points of `gathered`, in any order, into `acc`, keeping the
/// first gathered of equal points, and empty `gathered`.
fn merge_gathered(acc: &mut Vec<Point<Prov>>, gathered: &mut Vec<Point<Prov>>) {
    if !merge_by_energy(acc, gathered) {
        merge_sorted(acc, gathered);
    }
    gathered.clear();
}

/// [`merge_gathered`] by sorting `gathered` on energy, then adaptation
/// highest first, keeping gathered order among equal points.
fn merge_sorted(acc: &mut Vec<Point<Prov>>, gathered: &mut [Point<Prov>]) {
    gathered.sort_by(|p, q| p.mfe.0.cmp(&q.mfe.0).then_with(|| q.cai.total_cmp(&p.cai)));
    merge_into(acc, gathered.iter().copied());
}

/// Flanks' best adaptations as read, by the node that tells the flanks of
/// one loop side apart.
type FlankBests = SmallVec<[(u32, f64); 4]>;

/// The best adaptation recorded for the flank at `node`, read by `best` and
/// recorded where it is not.
fn flank_best(bests: &mut FlankBests, node: u32, best: impl FnOnce() -> f64) -> f64 {
    if let Some(&(_, b)) = bests.iter().find(|&&(n, _)| n == node) {
        return b;
    }
    let b = best();
    bests.push((node, b));
    b
}

/// Gathered points below which [`merge_by_energy`] leaves them to the sort.
const BY_ENERGY_MIN: usize = 32;

thread_local! {
    /// Per energy above a batch's least, the index of the batch's best
    /// point there, reused across batches.
    static BEST_AT: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// [`merge_gathered`] without a sort, where the batch's energies span at
/// most four times as many values as it has points; `false`, merging
/// nothing, where they do not.
///
/// [`merge_into`] keeps at most the first point of each energy in its
/// merged order, which is the one of highest adaptation: any later point of
/// that energy has no more adaptation than the point before it in the
/// merged front. So of each energy only the batch's best point, the first
/// gathered among equals as the sort orders them, can survive, and those are
/// found by one pass that buckets the batch by energy, then read back in
/// energy order.
///
/// That holds where `acc` is in the order [`merge_into`] leaves it, energy
/// ascending and adaptation descending, and no adaptation is NaN: every
/// caller merges into a front built this way from points that
/// [`Admit::admits`], which admits no NaN.
fn merge_by_energy(acc: &mut Vec<Point<Prov>>, gathered: &[Point<Prov>]) -> bool {
    let n = gathered.len();
    if n < BY_ENERGY_MIN {
        return false;
    }
    let (lo, hi) = gathered.iter().fold((i32::MAX, i32::MIN), |(lo, hi), p| {
        (lo.min(p.mfe.0), hi.max(p.mfe.0))
    });
    let span = (i64::from(hi) - i64::from(lo)) as usize + 1;
    if span > 4 * n {
        return false;
    }
    BEST_AT.with(|best| {
        let best = &mut *best.borrow_mut();
        best.clear();
        best.resize(span, u32::MAX);
        for (i, p) in gathered.iter().enumerate() {
            let at = &mut best[(p.mfe.0 - lo) as usize];
            if *at == u32::MAX
                || p.cai.total_cmp(&gathered[*at as usize].cai) == std::cmp::Ordering::Greater
            {
                *at = i as u32;
            }
        }
        merge_into(
            acc,
            best.iter()
                .filter(|&&i| i != u32::MAX)
                .map(|&i| gathered[i as usize]),
        );
    });
    true
}

/// Gather the points of `front` shifted by `(de, dc)` that `admit` admits
/// into `gathered`, recording each point's index through `back`, and merge
/// them into `acc` once `gathered` holds a batch.
fn gather_shift(
    (acc, gathered): (&mut Vec<Point<Prov>>, &mut Vec<Point<Prov>>),
    front: &Front<Prov>,
    (de, dc): (Energy, f64),
    admit: &Admit,
    back: impl Fn(u32) -> Prov,
) {
    let points = front.points();
    let first = admit.first(points, dc);
    gathered.extend(
        points[first..]
            .iter()
            .enumerate()
            .map(|(k, p)| Point::new(p.mfe + de, p.cai + dc, back((first + k) as u32)))
            .filter(|p| admit.admits(p.mfe, p.cai)),
    );
    if gathered.len() >= GATHER_BATCH {
        merge_gathered(acc, gathered);
    }
}

/// Whether the multi-loop and exterior split points are counted and timed
/// (off unless a development build turns it on), printed per slice and per gap.
fn split_census() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::dev_env::var_os(crate::dev_env::Setting::SplitCensus).is_some())
}

/// What a fold's split points cost and what they kept: per table, the cells
/// with a split point, the split points enumerated and those with a point in
/// the cell's stored front, the sums tried over each, and seconds in the
/// split loops, in their merges, and in the cuts after them.
#[derive(Clone, Copy, Default)]
struct SplitTally {
    cells: usize,
    splits: usize,
    useful: usize,
    sums: u64,
    useful_sums: u64,
    /// Sums formed and tested, of `sums`: the rest a reference dominated.
    formed: u64,
    /// Sums a reference dominated and the split loop skipped unformed,
    /// counted with or without the census.
    skipped: u64,
    /// Splits ruled out from their cells' least scores before either
    /// cell's order is read.
    ruled_out: usize,
    /// An order-free digest of every stored cell's points.
    digest: u64,
    secs: f64,
    merge_secs: f64,
    post_secs: f64,
}

impl SplitTally {
    fn add(&mut self, o: &Self) {
        self.cells += o.cells;
        self.splits += o.splits;
        self.useful += o.useful;
        self.sums += o.sums;
        self.useful_sums += o.useful_sums;
        self.formed += o.formed;
        self.skipped += o.skipped;
        self.ruled_out += o.ruled_out;
        self.digest = self.digest.wrapping_add(o.digest);
        self.secs += o.secs;
        self.merge_secs += o.merge_secs;
        self.post_secs += o.post_secs;
    }

    /// The sums tried over split points with no point stored.
    fn skippable(&self) -> f64 {
        if self.sums == 0 {
            0.0
        } else {
            1.0 - self.useful_sums as f64 / self.sums as f64
        }
    }

    /// Record a cell's split points, as `(node, sums tried)`, against the
    /// nodes of those with a point in its stored front.
    fn cell(&mut self, enumerated: &[(u32, u64)], kept: &mut Vec<u32>) {
        if enumerated.is_empty() {
            return;
        }
        kept.sort_unstable();
        kept.dedup();
        self.cells += 1;
        self.splits += enumerated.len();
        for &(w, sums) in enumerated {
            self.sums += sums;
            if kept.binary_search(&w).is_ok() {
                self.useful += 1;
                self.useful_sums += sums;
            }
        }
    }

    fn line(&self, name: &str, wall: f64) -> String {
        format!(
            "{name} cells {} splits {} useful {} ({:.4}) sums {} useful {} skippable {:.4} formed {} ({:.4}) digest {:016x} secs {:.3} (merge {:.3}, cuts {:.3}) share {:.4} ceiling {:.4} ruled out {}",
            self.cells,
            self.splits,
            self.useful,
            self.useful as f64 / self.splits.max(1) as f64,
            self.sums,
            self.useful_sums,
            self.skippable(),
            self.formed,
            self.formed as f64 / self.sums.max(1) as f64,
            self.digest,
            self.secs,
            self.merge_secs,
            self.post_secs,
            self.secs / wall.max(1e-9),
            self.secs * self.skippable() / wall.max(1e-9),
            self.ruled_out,
        )
    }
}

/// Print a fold's or a gap's split census against `wall` seconds.
fn split_lines(what: &str, tallies: &[SplitTally; 2], wall: f64) {
    let mut both = tallies[0];
    both.add(&tallies[1]);
    eprintln!(
        "  split census {what} wall {wall:.3}: {} | {} | both share {:.4} skippable {:.4} ceiling {:.4}",
        tallies[0].line("M2", wall),
        tallies[1].line("Z", wall),
        both.secs / wall.max(1e-9),
        both.skippable(),
        (tallies[0].secs * tallies[0].skippable() + tallies[1].secs * tallies[1].skippable())
            / wall.max(1e-9),
    );
}

/// Which staircase the split loops read to skip sums it dominates
/// (`score` unless a development build picks another), which forms only
/// the sums some cut's line can admit, see [`ByScore`]; none (`off` or `0`);
/// the cell's merged points so far (`acc`); or the cell's path sums (`path`,
/// taken for any other value), see [`path_sums`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum SplitPrune {
    Off,
    Acc,
    Path,
    Score,
}

#[cfg(test)]
thread_local! {
    /// The rule the tests fold under in place of the development setting: 0
    /// reads the setting, 1, 2, 3 and 4 are off, `acc`, `path` and
    /// `score`. Each test thread holds its own, and a fold reads it on the
    /// thread that builds the fold.
    static SPLIT_PRUNE_TEST: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

fn split_prune() -> SplitPrune {
    #[cfg(test)]
    match SPLIT_PRUNE_TEST.with(std::cell::Cell::get) {
        1 => return SplitPrune::Off,
        2 => return SplitPrune::Acc,
        3 => return SplitPrune::Path,
        4 => return SplitPrune::Score,
        _ => {}
    }
    static ON: std::sync::OnceLock<SplitPrune> = std::sync::OnceLock::new();
    *ON.get_or_init(
        || match crate::dev_env::var(crate::dev_env::Setting::SplitPrune).as_deref() {
            Ok("off") | Ok("0") => SplitPrune::Off,
            Err(_) | Ok("score") => SplitPrune::Score,
            Ok("acc") => SplitPrune::Acc,
            Ok(_) => SplitPrune::Path,
        },
    )
}

/// The staircase a split loop skips the dominated sums of: none, the cell's
/// merged points so far, or a staircase of the cell's own admitted sums.
#[derive(Clone, Copy)]
enum Reference<'r> {
    None,
    Acc,
    Given(&'r [Point<Prov>]),
}

/// Whether a point of the staircase `reference` strictly dominates
/// `(mfe, cai)`: no more energy and at least as much adaptation, and not equal
/// in both. The staircase's adaptation rises with its energy, so its last
/// point at or below `mfe` is the one to test, and its adaptation is returned
/// as the least a later sum of the same row must reach to escape it.
fn dominated_by(reference: &[Point<Prov>], mfe: Energy, cai: f64) -> Option<f64> {
    let k = reference.partition_point(|q| q.mfe.0 <= mfe.0);
    let q = reference.get(k.checked_sub(1)?)?;
    (q.cai > cai || (q.cai == cai && q.mfe.0 < mfe.0)).then_some(q.cai)
}

/// A front's points in ascending score at each cut's weight, as
/// `(weight, [index], [score])`, equal scores in index order. The scores sit
/// beside the indexes so a search reads them in order rather than through
/// the front.
///
/// A score is affine in the point, so a sum's score at a weight is its two
/// terms' scores added. A row of sums `a + b` over the points `b` of this
/// front can meet a line `score <= at` only where `b` scores at most `at`
/// less `a`'s own score, and those points are a prefix of this order: a row
/// forms only that prefix, read past the line by the rounding between the two
/// forms of one score, and tests each sum against the whole [`Admit`] as the
/// full row would.
struct ByScore {
    /// The weights, one order each.
    weights: SmallVec<[f64; 2]>,
    /// Per weight in turn, the front's indexes in ascending score.
    order: Vec<u32>,
    /// The score of each index in `order`, at its weight.
    scores: Vec<f64>,
}

thread_local! {
    /// The scores and indexes [`ByScore::new`] sorts, reused across fronts.
    static SCORED: std::cell::RefCell<Vec<(f64, u32)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The score of the point `j` of `points` at the weight `lambda`.
fn score_of(points: &[Point<Prov>], j: u32, lambda: f64) -> f64 {
    let b = &points[j as usize];
    brd(b.mfe, b.cai, lambda)
}

impl ByScore {
    fn new(front: &Front<Prov>, weights: impl Iterator<Item = f64>) -> Self {
        let weights: SmallVec<[f64; 2]> = weights.collect();
        let n = front.len() * weights.len();
        let (mut order, mut scores) = (Vec::with_capacity(n), Vec::with_capacity(n));
        SCORED.with(|scored| {
            let scored = &mut *scored.borrow_mut();
            for &lambda in &weights {
                scored.clear();
                scored.extend(
                    front
                        .points()
                        .iter()
                        .enumerate()
                        .map(|(j, b)| (brd(b.mfe, b.cai, lambda), j as u32)),
                );
                scored.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
                order.extend(scored.iter().map(|&(_, j)| j));
                scores.extend(scored.iter().map(|&(score, _)| score));
            }
        });
        Self {
            weights,
            order,
            scores,
        }
    }

    /// The order at the weight `lambda` and its scores, if this holds one.
    fn at(&self, lambda: f64) -> Option<(&[u32], &[f64])> {
        let w = self.weights.iter().position(|&l| l == lambda)?;
        let n = self.order.len() / self.weights.len();
        let range = w * n..(w + 1) * n;
        Some((&self.order[range.clone()], &self.scores[range]))
    }

    /// The fewest points of the front a row shifted by `(ae, ac)` must form
    /// to meet every line of `admit`: the prefix of the most selective
    /// weight's order. `None` where no line has a weight here.
    fn candidates(&self, admit: &Admit, (ae, ac): (Energy, f64)) -> Option<&[u32]> {
        let mut best: Option<&[u32]> = None;
        for &(lambda, at) in &admit.lines {
            let Some((order, scores)) = self.at(lambda) else {
                continue;
            };
            let room = at - brd(ae, ac, lambda) + RULE_OUT_SLACK;
            let k = scores.partition_point(|&score| score <= room);
            if best.is_none_or(|b| k < b.len()) {
                best = Some(&order[..k]);
            }
        }
        best
    }

    /// The rows of a split, as indices into this front ascending, that can
    /// form a sum with some point of `right_by`'s front meeting every line of `admit`
    /// once shifted by `(de, dc)`: a prefix of the most selective weight's
    /// order, each row's own score plus the least score of `right` read past
    /// the line by the rounding of both. `None` where no line has a weight in
    /// both orders. `left` is the front this order was built over.
    fn rows(
        &self,
        left: &[Point<Prov>],
        right_by: &Self,
        admit: &Admit,
        (de, dc): (Energy, f64),
    ) -> Option<Rows> {
        let mut best: Option<(&[u32], f64, f64)> = None;
        for &(lambda, at) in &admit.lines {
            let (Some((order, scores)), Some((_, low))) = (self.at(lambda), right_by.at(lambda))
            else {
                continue;
            };
            let Some(&low) = low.first() else {
                return Some(Rows::Listed(SmallVec::new()));
            };
            let room = at - low - brd(de, dc, lambda) + 2.0 * RULE_OUT_SLACK;
            let k = scores.partition_point(|&score| score <= room);
            if best.is_none_or(|(b, ..)| k < b.len()) {
                best = Some((&order[..k], lambda, room));
            }
        }
        let (prefix, lambda, room) = best?;
        if prefix.len() * ROWS_BY_SCAN >= left.len() {
            return Some(Rows::Below(lambda, room));
        }
        let mut rows: SmallVec<[u32; 16]> = prefix.iter().copied().collect();
        rows.sort_unstable();
        Some(Rows::Listed(rows))
    }
}

/// A split's rows that can still meet the cut lines, ascending: listed, or
/// every row whose score at a weight is at most a room. The two name the
/// same rows, a prefix of the weight's order being exactly the rows scoring
/// at most its room; a long prefix is read by scanning the rows in place
/// rather than copied out of the order and sorted back.
enum Rows {
    Listed(SmallVec<[u32; 16]>),
    Below(f64, f64),
}

/// A prefix holding at least one row in this many is scanned for in place.
const ROWS_BY_SCAN: usize = 4;

/// Gather the admitted sums along each split's merge path into `gathered`:
/// from both staircases' first points to both last points, a step at a time
/// in whichever staircase rises least in energy. Each is a sum
/// [`gather_compose`] forms over the same split, with the same terms in the
/// same order, so a staircase of them holds only candidates of the cell.
fn path_sums(
    gathered: &mut Vec<Point<Prov>>,
    (left, right): (&Front<Prov>, &Front<Prov>),
    (de, dc): (Energy, f64),
    admit: &Admit,
    back: impl Fn(u32, u32) -> Prov,
) {
    let (l, r) = (left.points(), right.points());
    if l.is_empty() || r.is_empty() {
        return;
    }
    let (mut i, mut j) = (0, 0);
    loop {
        let (ae, ac) = (l[i].mfe + de, l[i].cai + dc);
        let sum = Point::new(r[j].mfe + ae, r[j].cai + ac, back(i as u32, j as u32));
        if admit.admits(sum.mfe, sum.cai) {
            gathered.push(sum);
        }
        let (more_l, more_r) = (i + 1 < l.len(), j + 1 < r.len());
        match (more_l, more_r) {
            (false, false) => break,
            (true, false) => i += 1,
            (false, true) => j += 1,
            (true, true) => {
                if l[i + 1].mfe.0 - l[i].mfe.0 <= r[j + 1].mfe.0 - r[j].mfe.0 {
                    i += 1;
                } else {
                    j += 1;
                }
            }
        }
    }
}

/// Gather every sum of a point of `left` and a point of `right`, shifted by
/// `(de, dc)`, that `admit` admits into `gathered`, recording both indices
/// through `back`, and merge them into `acc` a batch at a time. Returns the
/// sums the rows span past the adaptation floor, those formed and tested,
/// and the seconds in batch merges.
///
/// A cell's sums over all its split points are sorted together and merged in
/// one pass, a batch at a time. One merge per split point rebuilt `acc` per
/// split point, and one per point of `left` per point of a large `left`, as
/// the exterior's fronts are. Each sum adds its terms in the order that merge
/// did, and the first of equal sums in gathering order is kept, as there.
///
/// A sum some point of `reference` strictly dominates is skipped unformed,
/// with the rest of its row up to the first sum adapted at least as well as
/// that point. When `reference` holds only the cell's own admitted
/// candidates, the merge drops every such sum anyway: the dominating point is
/// merged ahead of it, and the adaptation the merge compares against never
/// falls, so the cell's points and their provenance are unchanged.
fn gather_compose(
    (acc, gathered): (&mut Vec<Point<Prov>>, &mut Vec<Point<Prov>>),
    (left, right): (&Front<Prov>, &Front<Prov>),
    (de, dc): (Energy, f64),
    admit: &Admit,
    (reference, (left_by, right_by)): (Reference, (Option<&ByScore>, Option<&ByScore>)),
    back: impl Fn(u32, u32) -> Prov,
) -> (u64, u64, f64) {
    let (mut sums, mut formed) = (0u64, 0u64);
    let mut merge_secs = 0.0;
    let points = right.points();
    let tried = |a: &Point<Prov>| row_tried(points, a.cai + dc, admit);
    // A row left out pushes nothing, so the batches merge where they would
    // have over every row.
    let rows = match (left_by, right_by) {
        (Some(l), Some(r)) => l.rows(left.points(), r, admit, (de, dc)),
        _ => None,
    };
    if rows.is_some() && count_split_sums() {
        sums = row_sums(left, right, dc, admit);
    }
    let listed = match &rows {
        Some(Rows::Listed(listed)) => Some(listed.as_slice()),
        _ => None,
    };
    let below = match &rows {
        Some(Rows::Below(lambda, room)) => Some((*lambda, *room)),
        _ => None,
    };
    let every = listed.is_none().then_some(0..left.len() as u32);
    for i in listed
        .iter()
        .copied()
        .flatten()
        .copied()
        .chain(every.into_iter().flatten())
    {
        if below.is_some_and(|(lambda, room)| score_of(left.points(), i, lambda) > room) {
            continue;
        }
        let a = &left.points()[i as usize];
        let (ae, ac) = (a.mfe + de, a.cai + dc);
        if rows.is_none() {
            sums += tried(a);
        }
        if let Some(order) = right_by.and_then(|o| o.candidates(admit, (ae, ac))) {
            // No two sums of one row are equal, so the order a row's sums are
            // gathered in leaves the merge's choice among equal sums alone.
            formed += order.len() as u64;
            for &j in order {
                let b = &points[j as usize];
                let (mfe, cai) = (b.mfe + ae, b.cai + ac);
                if admit.admits(mfe, cai) {
                    gathered.push(Point::new(mfe, cai, back(i, j)));
                }
            }
        } else {
            let against: Option<&[Point<Prov>]> = match reference {
                Reference::None => None,
                Reference::Acc => Some(acc.as_slice()),
                Reference::Given(r) => Some(r),
            };
            let mut j = admit.first(points, ac);
            while j < points.len() {
                let b = &points[j];
                let (mfe, cai) = (b.mfe + ae, b.cai + ac);
                if let Some(reach) = against.and_then(|r| dominated_by(r, mfe, cai)) {
                    j += 1 + points[j + 1..].partition_point(|b| b.cai + ac < reach);
                    continue;
                }
                formed += 1;
                if admit.admits(mfe, cai) {
                    gathered.push(Point::new(mfe, cai, back(i, j as u32)));
                }
                j += 1;
            }
        }
        if gathered.len() >= GATHER_BATCH {
            let t = std::time::Instant::now();
            merge_gathered(acc, gathered);
            merge_secs += t.elapsed().as_secs_f64();
        }
    }
    (sums, formed, merge_secs)
}

/// The sums a row shifted to adaptation `ac` spans past the adaptation
/// floor.
fn row_tried(points: &[Point<Prov>], ac: f64, admit: &Admit) -> u64 {
    points.len().saturating_sub(admit.first(points, ac)) as u64
}

/// The sums every row of a split spans past the adaptation floor.
fn row_sums(left: &Front<Prov>, right: &Front<Prov>, dc: f64, admit: &Admit) -> u64 {
    let points = right.points();
    left.points()
        .iter()
        .map(|a| row_tried(points, a.cai + dc, admit))
        .sum()
}

/// Whether a split loop that leaves rows out still counts the sums they
/// span, which costs the search per row it saves: in the tests only, so a
/// census of such a loop counts only the rows it reads.
fn count_split_sums() -> bool {
    cfg!(test)
}

/// An FNV-1a digest of one stored front, keyed by its table and cell, with
/// each split sum's provenance.
fn front_digest(key: (u32, u32, u32), points: &[Point<Prov>]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |x: u64| {
        for b in x.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    eat(u64::from(key.0));
    eat(u64::from(key.1));
    eat(u64::from(key.2));
    for p in points {
        eat(p.mfe.0 as u64);
        eat(p.cai.to_bits());
        match p.back {
            Prov::Split { w, w_hi, l, r } => {
                eat(u64::from(unpack_id(w, w_hi)));
                eat((u64::from(l) << 32) | u64::from(r));
            }
            Prov::ZPair { x, x_hi, l, r } => {
                eat(u64::from(unpack_id(x, x_hi)));
                eat((u64::from(l) << 32) | u64::from(r));
            }
            _ => eat(u64::MAX),
        }
    }
    h
}

/// A multi-loop table, for [`Fold::hold_table`].
#[derive(Clone, Copy)]
enum Table {
    M2,
    M,
    T,
    Lead,
}

/// The tables of a fold its budget gave up, kept so that a later fold under
/// a box inside its own can start from them ([`Fold::adopt`]).
struct Tables {
    corners: Store<Corners>,
    spelled: FxHashMap<(u32, u32), Vec<Spelling>>,
    p: Cells,
    p_at: Vec<Vec<PairAt>>,
    x: Cells,
    m: Cells,
    lead: Cells,
    m2: Cells,
    t: Cells,
    /// The span the fold was filling when it gave up: every shorter span is
    /// complete.
    span: u32,
}

/// For each cell adopted from [`Tables`], where each of its points went:
/// its index in the adopting fold's cell, or `u32::MAX` if it was dropped.
/// A cell absent here was dropped whole.
#[derive(Default)]
struct Moved {
    p: FxHashMap<(u32, u32), Vec<u32>>,
    x: FxHashMap<(u32, u32), Vec<u32>>,
    m: FxHashMap<(u32, u32), Vec<u32>>,
    lead: FxHashMap<(u32, u32), Vec<u32>>,
    m2: FxHashMap<(u32, u32), Vec<u32>>,
    t: FxHashMap<(u32, u32), Vec<u32>>,
}

/// Where the point `k` of the adopted cell `key` went, if it was kept.
fn moved(cells: &FxHashMap<(u32, u32), Vec<u32>>, key: (u32, u32), k: u32) -> Option<u32> {
    cells
        .get(&key)?
        .get(k as usize)
        .copied()
        .filter(|&n| n != u32::MAX)
}

/// The points of `from` that `hold` keeps and whose provenance `back`
/// rewrites into the adopting fold, in their order, with where each point
/// of `from` went ([`Moved`]).
fn sift(
    from: &Front<Prov>,
    hold: impl FnOnce(&mut Front<Prov>),
    back: impl Fn(Prov) -> Option<Prov>,
) -> (Front<Prov>, Vec<u32>) {
    // Each point carries its index in `from` through `hold`, which reads
    // only energies and adaptations.
    let mut k = 0;
    let mut front = from.shift_map(Energy::ZERO, 0.0, |_| {
        k += 1;
        Prov::Multi { k: k - 1 }
    });
    hold(&mut front);
    let index = |b: &Prov| match *b {
        Prov::Multi { k } => k as usize,
        _ => unreachable!("a sifted point carries its index"),
    };
    let mut went = vec![u32::MAX; from.len()];
    let mut kept = 0;
    front.retain_backs(|b| {
        let k = index(b);
        let Some(moved) = back(from.points()[k].back) else {
            return false;
        };
        went[k] = kept;
        kept += 1;
        *b = moved;
        true
    });
    (front, went)
}

impl<'a> Fold<'a> {
    /// Fold under the given cuts, starting from the complete spans of
    /// `from`, a fold under a box holding this one's ([`Self::adopt`]).
    fn run_from(
        params: &'a Params,
        l: &'a Lattice,
        region: &'a PairConstraint,
        ceiling: Option<Energy>,
        (cuts, gap_box, zero): (Cuts<'a>, Option<GapBox>, &'a [ZeroChord<'a>]),
        (budget, stop, threads): (Option<usize>, Stop<'a>, usize),
        from: Option<&Tables>,
    ) -> Self {
        assert!(
            l.edges.len() <= 1 << ID_BITS && l.pos.len() <= 1 << ID_BITS,
            "a lattice too large for a point's provenance"
        );
        let mut fold = Self {
            params,
            l,
            region,
            ceiling,
            cuts,
            prune: split_prune(),
            gap_box,
            zero,
            corners: Store::dense(l.pos.len()),
            loop_floors: (0..LOOP_FLOORS)
                .map(|_| std::sync::OnceLock::new())
                .collect(),
            spelled: FxHashMap::default(),
            p: Cells::default(),
            p_at: vec![Vec::new(); l.seq_len as usize + 1],
            x: Cells::dense(l.pos.len()),
            x_score: Vec::new(),
            m_score: Vec::new(),
            x_low: Vec::new(),
            m_low: Vec::new(),
            z_score: Vec::new(),
            x_into: vec![Vec::new(); l.pos.len()],
            m: Cells::dense(l.pos.len()),
            lead: Cells::dense(l.pos.len()),
            m2: Cells::default(),
            t: Cells::dense(l.pos.len()),
            z: vec![Front::empty(); l.pos.len()],
            end: Front::empty(),
            clock: [0.0; 4],
            aware: [0; 3],
            stored: 0,
            budget: budget.unwrap_or(usize::MAX),
            threads: threads.max(1),
            aborted: false,
            next: (1, 0),
            filled: Vec::new(),
            stop,
            stopped: false,
            capped: false,
            unread: 0,
            adopted: 0,
            split: [SplitTally::default(); 2],
        };
        if let Some(from) = from {
            fold.adopt(from);
        }
        fold.fill();
        fold
    }

    /// [`Self::run_from`] from nothing.
    fn run(
        params: &'a Params,
        l: &'a Lattice,
        region: &'a PairConstraint,
        ceiling: Option<Energy>,
        (cuts, gap_box, zero): (Cuts<'a>, Option<GapBox>, &'a [ZeroChord<'a>]),
        (budget, stop, threads): (Option<usize>, Stop<'a>, usize),
    ) -> Self {
        Self::run_from(
            params,
            l,
            region,
            ceiling,
            (cuts, gap_box, zero),
            (budget, stop, threads),
            None,
        )
    }

    /// Continue a fold its budget gave up, under `budget`, from the cell
    /// after the last one stored: the same cells in the same order, so the
    /// tables are those of a fold run under `budget` from the start.
    fn resume(&mut self, budget: usize) {
        debug_assert!(self.aborted && !self.stopped);
        self.budget = budget;
        self.aborted = false;
        self.fill();
    }

    /// Fill the spans from where the fold stands and, if they all fill, the
    /// exterior.
    fn fill(&mut self) {
        let finished = Pool::scoped(self.threads - 1, |pool| self.fill_spans(pool));
        if !finished {
            return;
        }
        let t = std::time::Instant::now();
        self.exterior();
        self.clock[3] = t.elapsed().as_secs_f64();
        if crate::dev_env::var_os(crate::dev_env::Setting::MotifFrontCensus).is_some() {
            self.census();
        }
    }

    /// Fill every span, shortest first, and say whether the fold finished.
    ///
    /// A cell over `u..v` reads only cells of shorter spans and its own, so the
    /// cells of one span are computed on the threads a hand-out at a time and
    /// stored in the one-thread order. The tables, and the cell at which the
    /// budget gives up, are the one-thread fold's for any thread count.
    fn fill_spans(&mut self, pool: &Pool) -> bool {
        let l = self.l;
        let hand_out = if self.threads > 1 {
            self.threads * CELLS_PER_THREAD
        } else {
            1
        };
        let mut cells = Vec::new();
        let (first, mut skip) = self.next;
        for span in first..=l.seq_len {
            if span > 1 && skip == 0 {
                self.filled.push(self.stored);
            }
            if self.out_of_memory() {
                return false;
            }
            self.span_cells(span, &mut cells);
            let mut at = skip;
            for batch in cells[skip..].chunks(hand_out) {
                for (&(u, v), out) in batch.iter().zip(self.cells(pool, batch)) {
                    self.commit(u, v, out);
                    at += 1;
                    if self.stored > self.budget {
                        self.aborted = true;
                        self.next = (span, at);
                        return false;
                    }
                    if self.stop.timed_out() {
                        self.aborted = true;
                        self.stopped = true;
                        return false;
                    }
                    if self.stop.gap_due() {
                        self.aborted = true;
                        self.stopped = true;
                        self.capped = true;
                        return false;
                    }
                    self.unread += 1;
                    if self.unread >= MEMORY_EVERY && self.out_of_memory() {
                        return false;
                    }
                }
            }
            skip = 0;
        }
        true
    }

    /// Start from the complete spans of `from`, the tables of a fold that
    /// gave up under a box holding this fold's: an energy ceiling, a
    /// ceiling per cut and an adaptation floor each at least as loose.
    ///
    /// Every test a cell holds a point to is one of those limits, read at
    /// the point's energy and adaptation, so it drops whatever the looser
    /// one drops. A cell's front in `from` therefore holds, for every point
    /// this fold's cell would hold, that point or one that dominates it.
    /// Each adopted cell is `from`'s held to this fold's tests, less every
    /// point composed from a point dropped below it: a point that cannot
    /// complete into the box cannot be part of one that does. A point this
    /// fold would hold and the adopted cell lacks is then dominated by a
    /// point that cannot complete into the box, so it cannot either, and
    /// the spans filled from here end on the same front. The cells are
    /// stored in the order [`Self::fill_spans`] stores them.
    fn adopt(&mut self, from: &Tables) {
        self.corners.clone_from(&from.corners);
        let mut moved = Moved::default();
        let mut cells = Vec::new();
        for span in 1..from.span {
            if span > 1 {
                self.filled.push(self.stored);
            }
            self.span_cells(span, &mut cells);
            for &(u, v) in &cells {
                let out = self.sifted(from, (u, v), &mut moved);
                self.commit(u, v, out);
            }
        }
        self.next = (from.span.max(1), 0);
        self.adopted = self.stored;
    }

    /// The cells over `u..v` of `from`, held to this fold's tests.
    fn sifted(&self, from: &Tables, (u, v): (u32, u32), moved: &mut Moved) -> CellOut {
        let mut out = CellOut::default();
        let (i, j1) = (self.l.pos[u as usize], self.l.pos[v as usize]);
        if let Some(close) = j1.checked_sub(1) {
            let at = &from.p_at[i as usize];
            let lo = at.partition_point(|&(c, ..)| c < close);
            let hi = at.partition_point(|&(c, ..)| c <= close);
            for &(_, (a, b), slot, ..) in &at[lo..hi] {
                let (ea, eb) = (self.l.edge(a), self.l.edge(b));
                if (ea.from, eb.to) != (u, v) {
                    continue;
                }
                let (o, key) = self.l.pair_key(ea, eb);
                if self.cuts().any(|(cut, _)| cut.rules_out_pair(o, &key)) {
                    continue;
                }
                let Some(aware) = self.aware_lines(ea, eb, &mut 0) else {
                    continue;
                };
                let zero_floors = self.zero_floors((ea, eb), o, &key);
                let (front, went) = sift(
                    from.p.at(slot),
                    |f| {
                        self.hold_floor(f, (u, v));
                        self.hold_pair(f, (o, &key), &zero_floors, &aware);
                    },
                    |back| self.moved_back(back, (u, v), (ea.to, eb.from), moved),
                );
                moved.p.insert((a, b), went);
                if front.is_empty() {
                    continue;
                }
                if let Some(spelled) = from.spelled.get(&(a, b)) {
                    out.spelled.push(((a, b), spelled.clone()));
                }
                out.p.push(((a, b), front));
            }
        }
        let cell = (u, v);
        let one = |cells: &Cells, table: Option<Table>, moved: &Moved| {
            let parent = cells.get(&cell)?;
            let (front, went) = sift(
                parent,
                |f| {
                    self.hold_floor(f, cell);
                    if let Some(table) = table {
                        self.hold_table(table, f, cell);
                    }
                },
                |back| self.moved_back(back, cell, cell, moved),
            );
            Some((front, went))
        };
        // In the order a cell's tables read each other: `X` from its pairs,
        // `M` from `X` and `M2`, `T` from `M2`, `L` from `T`.
        if let Some((f, went)) = one(&from.x, None, moved) {
            moved.x.insert(cell, went);
            out.x = Some(f).filter(|f| !f.is_empty());
        }
        if let Some((f, went)) = one(&from.m2, Some(Table::M2), moved) {
            moved.m2.insert(cell, went);
            out.m2 = Some(f).filter(|f| !f.is_empty());
        }
        if let Some((f, went)) = one(&from.m, Some(Table::M), moved) {
            moved.m.insert(cell, went);
            out.m = Some(f).filter(|f| !f.is_empty());
        }
        if let Some((f, went)) = one(&from.t, Some(Table::T), moved) {
            moved.t.insert(cell, went);
            out.t = Some(f).filter(|f| !f.is_empty());
        }
        if let Some((f, went)) = one(&from.lead, Some(Table::Lead), moved) {
            moved.lead.insert(cell, went);
            out.lead = Some(f).filter(|f| !f.is_empty());
        }
        out
    }

    /// The provenance `back` of a point adopted into the cell `u..v`, with
    /// its points' indexes moved to where [`Self::adopt`] put them; `None`
    /// where one of them was dropped. A closed pair's multi-loop reads `L`
    /// over `inside`.
    fn moved_back(
        &self,
        back: Prov,
        (u, v): (u32, u32),
        inside: (u32, u32),
        moved: &Moved,
    ) -> Option<Prov> {
        Some(match back {
            Prov::Start | Prov::Hairpin { .. } | Prov::HairpinSpelled { .. } => back,
            Prov::Single {
                cd,
                cd_hi,
                corners,
                k,
            } => Prov::Single {
                cd,
                cd_hi,
                corners,
                k: self::moved(&moved.p, unpack_ids(cd, cd_hi), k)?,
            },
            Prov::Multi { k } => Prov::Multi {
                k: self::moved(&moved.lead, inside, k)?,
            },
            Prov::XPair { ab, ab_hi, k } => Prov::XPair {
                ab,
                ab_hi,
                k: self::moved(&moved.p, unpack_ids(ab, ab_hi), k)?,
            },
            Prov::MBranch { k } => Prov::MBranch {
                k: self::moved(&moved.x, (u, v), k)?,
            },
            Prov::MUnpaired { e, k } => Prov::MUnpaired {
                e,
                k: self::moved(&moved.m, (u, self.l.edge(e).from), k)?,
            },
            Prov::MFromM2 { k } => Prov::MFromM2 {
                k: self::moved(&moved.m2, (u, v), k)?,
            },
            Prov::Split { w, w_hi, l, r } => {
                let at = unpack_id(w, w_hi);
                Prov::Split {
                    w,
                    w_hi,
                    l: self::moved(&moved.m, (u, at), l)?,
                    r: self::moved(&moved.x, (at, v), r)?,
                }
            }
            Prov::LCore { k } => Prov::LCore {
                k: self::moved(&moved.t, (u, v), k)?,
            },
            Prov::LUnpaired { e, k } => Prov::LUnpaired {
                e,
                k: self::moved(&moved.lead, (self.l.edge(e).to, v), k)?,
            },
            Prov::TCore { k } => Prov::TCore {
                k: self::moved(&moved.m2, (u, v), k)?,
            },
            Prov::TUnpaired { e, k } => Prov::TUnpaired {
                e,
                k: self::moved(&moved.t, (u, self.l.edge(e).from), k)?,
            },
            Prov::ZUnpaired { .. } | Prov::ZPair { .. } | Prov::End { .. } => {
                unreachable!("the exterior is never adopted")
            }
        })
    }

    /// The tables of a fold its budget gave up, for a later fold to adopt.
    fn into_tables(self) -> Tables {
        debug_assert!(self.aborted && !self.stopped);
        Tables {
            corners: self.corners,
            spelled: self.spelled,
            p: self.p,
            p_at: self.p_at,
            x: self.x,
            m: self.m,
            lead: self.lead,
            m2: self.m2,
            t: self.t,
            span: self.next.0,
        }
    }

    /// The cells of `span` this fold fills, in the order it fills them.
    fn span_cells(&self, span: u32, cells: &mut Vec<(u32, u32)>) {
        let l = self.l;
        cells.clear();
        for i in 0..=l.seq_len - span {
            for &u in &l.at[i as usize] {
                for &v in &l.at[(i + span) as usize] {
                    if self.reaches_box(u, v) {
                        cells.push((u, v));
                    }
                }
            }
        }
    }

    /// Whether some path joins `u` to `v` and, under a box, the best design
    /// through both still reaches its adaptation floor. Every table over
    /// `u..v` keeps only points [`Self::keep_span`] keeps, and none of them
    /// is more adapted than the best path, so a cell failing this would be
    /// empty in every table and is not computed.
    fn reaches_box(&self, u: u32, v: u32) -> bool {
        let fill = self.fill_at(u, v);
        if fill == f64::NEG_INFINITY {
            return false;
        }
        self.gap_box.is_none_or(|gap| {
            // Twice the slack: the path and a point's own sum add the same
            // terms in different orders.
            self.l.prefix[u as usize] + fill + self.l.suffix[v as usize]
                >= gap.c_min - 2.0 * BOX_SLACK
        })
    }

    /// Whether the memory budget is spent, marking the fold stopped if so.
    fn out_of_memory(&mut self) -> bool {
        self.unread = 0;
        let spent = self.stop.out_of_memory();
        if spent {
            self.aborted = true;
            self.stopped = true;
        }
        spent
    }

    /// The cells over `batch`, in its order, computed on the fold's threads.
    fn cells(&self, pool: &Pool, batch: &[(u32, u32)]) -> Vec<CellOut> {
        if batch.len() == 1 {
            return vec![self.cell(batch[0].0, batch[0].1)];
        }
        let next = AtomicUsize::new(0);
        let slots: Vec<Mutex<Option<CellOut>>> = batch.iter().map(|_| Mutex::default()).collect();
        pool.run_joined(self.threads, &|_| loop {
            let k = next.fetch_add(1, Ordering::Relaxed);
            let Some(&(u, v)) = batch.get(k) else {
                break;
            };
            *slots[k].lock().unwrap() = Some(self.cell(u, v));
        });
        slots
            .into_iter()
            .map(|slot| slot.into_inner().unwrap().expect("every cell computed"))
            .collect()
    }

    /// The closed pairs and the multi-loop tables over `u..v`.
    fn cell(&self, u: u32, v: u32) -> CellOut {
        let mut out = CellOut::default();
        let t = std::time::Instant::now();
        self.pairs(u, v, &mut out);
        let t2 = std::time::Instant::now();
        self.multi_tables(u, v, &mut out);
        out.clock[0] += (t2 - t).as_secs_f64();
        out.clock[2] += t2.elapsed().as_secs_f64();
        out
    }

    /// Store the cells over `u..v`, counting their points as kept, after
    /// every cut.
    /// Whether no sum over the split of the `M` cell `left` and the `X` cell
    /// `stem` is admitted, from the cells' best adaptations and least scores
    /// alone.
    fn split_ruled_out(&self, (left, stem): (u32, u32), intern: Energy, admit: &Admit) -> bool {
        let (Some((left_cai, left)), Some((stem_cai, stem))) =
            (self.m_low.get(left as usize), self.x_low.get(stem as usize))
        else {
            return false;
        };
        admit.rules_out_sums((*left_cai, left), (*stem_cai, stem), (intern, 0.0))
    }

    /// The score orders of the cell `key` of `cells`, sorted on first read.
    fn by_score<'s>(
        &'s self,
        orders: &'s [std::sync::OnceLock<ByScore>],
        cells: &'s Cells,
        slot: u32,
    ) -> Option<&'s ByScore> {
        let order = orders.get(slot as usize)?;
        Some(
            order.get_or_init(|| {
                ByScore::new(cells.at(slot), self.cuts().map(|(cut, _)| cut.lambda))
            }),
        )
    }

    /// Whether cells keep score orders: under the `score` rule, with a cut
    /// to order by.
    fn scored(&self) -> bool {
        self.prune == SplitPrune::Score && self.cuts().next().is_some()
    }

    /// A front's least score at each cut's weight.
    fn lows(&self, front: &Front<Prov>) -> Lows {
        self.cuts()
            .map(|(cut, _)| {
                let low = front
                    .points()
                    .iter()
                    .map(|p| brd(p.mfe, p.cai, cut.lambda))
                    .fold(f64::INFINITY, f64::min);
                (cut.lambda, low)
            })
            .collect()
    }

    fn commit(&mut self, u: u32, v: u32, out: CellOut) {
        let i = self.l.pos[u as usize] as usize;
        self.stored += out.p.iter().map(|(_, f)| f.len()).sum::<usize>()
            + [&out.x, &out.m, &out.m2, &out.t, &out.lead]
                .iter()
                .filter_map(|f| f.as_ref().map(Front::len))
                .sum::<usize>();
        for (total, n) in self.aware.iter_mut().zip(out.aware) {
            *total += n;
        }
        for (pair, front) in out.p {
            let lows = self.lows(&front);
            let cai = front.points().last().map_or(f64::NEG_INFINITY, |p| p.cai);
            let close = self.l.pos[self.l.edge(pair.1).from as usize];
            assert!(self.p_at[i].last().is_none_or(|&(last, ..)| last <= close));
            let slot = self.p.insert(pair, front);
            self.p_at[i].push((close, pair, slot, cai, lows));
        }
        self.spelled.extend(out.spelled);
        let scored = self.scored();
        if let Some(x) = out.x {
            let low = scored.then(|| (best_cai(&x), self.lows(&x)));
            let slot = self.x.insert((u, v), x);
            if let Some(low) = low {
                fresh_order(&mut self.x_score, slot);
                fresh_low(&mut self.x_low, slot, low);
            }
            self.x_into[v as usize].push((u, slot));
        }
        if let Some(m) = out.m {
            let low = scored.then(|| (best_cai(&m), self.lows(&m)));
            let slot = self.m.insert((u, v), m);
            if let Some(low) = low {
                fresh_order(&mut self.m_score, slot);
                fresh_low(&mut self.m_low, slot, low);
            }
        }
        let tables = [
            (&mut self.m2, out.m2),
            (&mut self.t, out.t),
            (&mut self.lead, out.lead),
        ];
        for (cells, front) in tables {
            if let Some(front) = front {
                cells.insert((u, v), front);
            }
        }
        for (key, corners) in out.corners {
            self.corners.insert(key, corners);
        }
        self.split[0].add(&out.split);
        for (total, add) in self.clock.iter_mut().zip(out.clock) {
            *total += add;
        }
    }

    /// Cells and points per table, to stderr.
    fn census(&self) {
        eprintln!(
            "  seconds: pairs {:.2} (singles {:.2}) multi tables {:.2} exterior {:.2}",
            self.clock[0], self.clock[1], self.clock[2], self.clock[3]
        );
        let count = |name: &str, cells: &Cells| {
            let points: usize = cells.values().map(Front::len).sum();
            eprintln!("  {name}: {} cells, {points} points", cells.len());
        };
        count("P", &self.p);
        count("X", &self.x);
        count("M", &self.m);
        count("L", &self.lead);
        count("M2", &self.m2);
        count("T", &self.t);
        let z: usize = self.z.iter().map(Front::len).sum();
        eprintln!(
            "  spelled hairpins {}, corner entries {}",
            self.spelled.values().map(Vec::len).sum::<usize>(),
            self.corners.len()
        );
        eprintln!(
            "  Z: {z} points; nodes {}, edges {}, point bytes {}",
            self.l.pos.len(),
            self.l.edges.len(),
            std::mem::size_of::<Point<Prov>>()
        );
    }

    fn n(&self) -> usize {
        self.l.pos.len()
    }

    fn fill_at(&self, u: u32, v: u32) -> f64 {
        self.l.fill_at(u, v)
    }

    /// The best fill from `u` to `v` per (first, last) nucleotide of the
    /// stretch; empty when `u == v` or no path joins them.
    fn corners(&self, u: u32, v: u32, out: &mut CellOut) -> Corners {
        if let Some(c) = self
            .corners
            .get(&(u, v))
            .or_else(|| out.corners.get(&(u, v)))
        {
            // A copy of the slice: `SmallVec`'s own clone goes element by
            // element, and this runs for every pair's every loop.
            return SmallVec::from_slice(c);
        }
        let mut best: Corners = SmallVec::new();
        let len = self.l.pos[v as usize] - self.l.pos[u as usize];
        if len >= 1 {
            for &e1 in &self.l.out[u as usize] {
                for &e2 in &self.l.inn[v as usize] {
                    let (a, b) = (self.l.edge(e1), self.l.edge(e2));
                    let cai = if len == 1 {
                        if e1 != e2 {
                            continue;
                        }
                        a.cai
                    } else {
                        let mid = self.fill_at(a.to, b.from);
                        if mid == f64::NEG_INFINITY {
                            continue;
                        }
                        a.cai + mid + b.cai
                    };
                    let key = (a.nuc as u8, b.nuc as u8);
                    match best.iter_mut().find(|c| (c.0, c.1) == key) {
                        Some(c) => c.2 = c.2.max(cai),
                        None => best.push((key.0, key.1, cai)),
                    }
                }
            }
        }
        out.corners.insert((u, v), SmallVec::from_slice(&best));
        best
    }

    /// The most adaptation any corner choice of the stretch from `u` to `v`
    /// adds, `NEG_INFINITY` when it has none, read from its corners in place.
    fn corner_best(&self, u: u32, v: u32, out: &mut CellOut) -> f64 {
        let best = |cs: &Corners| cs.iter().map(|c| c.2).fold(f64::NEG_INFINITY, f64::max);
        if let Some(c) = self
            .corners
            .get(&(u, v))
            .or_else(|| out.corners.get(&(u, v)))
        {
            return best(c);
        }
        best(&self.corners(u, v, out))
    }

    /// Every spelling of the stretch from `u` to `v`, as edge lists.
    fn spellings(&self, u: u32, v: u32) -> Vec<SmallVec<[u32; 6]>> {
        let target = self.l.pos[v as usize];
        let mut out = Vec::new();
        let mut path: SmallVec<[u32; 6]> = SmallVec::new();
        self.spell_from(u, v, target, &mut path, &mut out);
        out
    }

    fn spell_from(
        &self,
        u: u32,
        v: u32,
        target: u32,
        path: &mut SmallVec<[u32; 6]>,
        out: &mut Vec<SmallVec<[u32; 6]>>,
    ) {
        if self.l.pos[u as usize] == target {
            if u == v {
                out.push(path.clone());
            }
            return;
        }
        for &e in &self.l.out[u as usize] {
            let to = self.l.edge(e).to;
            if self.fill_at(to, v) == f64::NEG_INFINITY {
                continue;
            }
            path.push(e);
            self.spell_from(to, v, target, path, out);
            path.pop();
        }
    }

    /// A gathered staircase as a cell.
    fn keep(&self, points: Vec<Point<Prov>>) -> Front<Prov> {
        Front::from_points(points)
    }

    /// A gathered staircase as the cell spanning `u..v`, without the points
    /// the box's adaptation floor rules out: adaptation does not depend on
    /// structure, so the best completion of anything spanning `u..v` adds
    /// exactly the best path to `u` and the best path from `v`.
    fn keep_span(&self, mut points: Vec<Point<Prov>>, (u, v): (u32, u32)) -> Front<Prov> {
        if let Some(gap) = &self.gap_box {
            let need =
                gap.c_min - self.l.prefix[u as usize] - self.l.suffix[v as usize] - BOX_SLACK;
            points.retain(|p| p.cai >= need);
        }
        Front::from_points(points)
    }

    /// The adaptation floor [`Self::keep_span`] holds the cell spanning
    /// `u..v` to, as an [`Admit`].
    fn span_admit(&self, (u, v): (u32, u32)) -> Admit {
        let mut admit = Admit::all();
        if let Some(gap) = &self.gap_box {
            admit.c_min = gap.c_min
                - self.l.prefix[u as usize]
                - self.l.suffix[v as usize]
                - BOX_SLACK
                - ADMIT_SLACK;
        }
        admit
    }

    /// [`Self::span_admit`] with the floors `floors` puts under the cell
    /// `u..v` at each cut, as [`retain_floor`] reads them.
    fn floor_admit(&self, (u, v): (u32, u32), floors: impl Fn(&Floors) -> &[f64]) -> Admit {
        let mut admit = self.span_admit((u, v));
        for (cut, f) in self.cuts() {
            let floor = floors(f)[u as usize * self.n() + v as usize];
            if floor != f64::NEG_INFINITY {
                admit.line(cut.lambda, cut.ceiling + CUT_SLACK - floor);
            }
        }
        admit
    }

    /// [`Self::span_admit`] with each cut's outside floor from the plain
    /// table `table` picks, as [`Cut::retain_span`] reads it.
    fn table_admit(
        &self,
        (u, v): (u32, u32),
        table: impl Fn(&Pass) -> &Vec<Vec<f64>> + Copy,
    ) -> Admit {
        let mut admit = self.span_admit((u, v));
        for (cut, _) in self.cuts() {
            let inner = self.l.base[u as usize];
            let outer = self.l.base[v as usize];
            if let Some(outside) = cut.span_outside(table, inner, outer) {
                admit.line(cut.lambda, cut.ceiling + CUT_SLACK - outside);
            }
        }
        admit
    }

    /// [`Self::span_admit`] for the exterior up to `v`, with each cut's floor
    /// on the exterior's completion from `v`.
    fn exterior_admit(&self, v: u32) -> Admit {
        let mut admit = self.span_admit((self.l.start, v));
        for (cut, _) in self.cuts() {
            if let Some(outside) = cut.exterior_outside(self.l.base[v as usize]) {
                admit.line(cut.lambda, cut.ceiling + CUT_SLACK - outside);
            }
        }
        admit
    }

    fn terminal_au(&self, pair: BasePair) -> Energy {
        terminal_au(self.params, pair)
    }

    /// The floor `z` puts under the completion of the pair `(ea, eb)`: the
    /// chord of its pass's plain completion and the best motif-free
    /// adaptation around the pair at weight 0, over every automaton state
    /// merged into the pair's outer nodes. `None` where either is unknown.
    /// The limits the cuts' constrained-lattice chords put on the points of
    /// the pair `(ea, eb)` ([`AwareChord::pair_lines`]); `None` where one
    /// rules the pair out.
    ///
    /// The cell key is the pair's nodes with their automaton states. Both this
    /// fold and the aware pass run over the same product, so a lattice node's
    /// state is the cell's own state, not a stand-in for others: after
    /// [`ProductDfa::merge_equivalent`] it is the class representative the
    /// constrained fold keys its cells by. A node holding more than one state
    /// would name no single cell, so its pairs are held to no limit at all
    /// rather than to one state's, and counted in `unkeyed` where a cut
    /// carries a chord, so a lever silently off on them shows.
    fn aware_lines(
        &self,
        ea: PEdge,
        eb: PEdge,
        unkeyed: &mut usize,
    ) -> Option<SmallVec<[(f64, f64); 2]>> {
        let mut lines = SmallVec::new();
        let (open, outer) = (ea.from as usize, eb.to as usize);
        let ([q_open], [q_outer]) = (&self.l.states[open][..], &self.l.states[outer][..]) else {
            *unkeyed += usize::from(self.cuts().any(|(cut, _)| cut.aware.is_some()));
            return Some(lines);
        };
        let on = self.l.base[outer];
        let edge = |e: PEdge| {
            let to = self.l.base[e.to as usize];
            crate::dfa::Edge {
                to,
                nuc: e.nuc,
                opening: crate::state::Opening::new(e.nuc, to.lane),
                cai: e.cai,
            }
        };
        let key = PKey::of(
            (self.l.base[open], edge(ea), *q_open),
            (self.l.base[eb.from as usize], edge(eb), *q_outer),
        );
        for (cut, _) in self.cuts() {
            let Some(chord) = &cut.aware else { continue };
            match chord.pair_lines(
                (on, &key),
                cut.near.layout,
                cut.lambda,
                cut.ceiling + CUT_SLACK,
            ) {
                PairLines::Out => return None,
                PairLines::Hold(more) => lines.extend(more),
            }
        }
        Some(lines)
    }

    fn zero_floor(
        &self,
        z: &ZeroChord<'_>,
        (ea, eb): (PEdge, PEdge),
        o: usize,
        key: &PairKey,
    ) -> Option<f64> {
        let upper = z.upper.get(o)?.get(key)?.outside;
        let (open, outer) = (ea.from as usize, eb.to as usize);
        let (base_open, base_outer) = (self.l.base[open], self.l.base[outer]);
        let around = self.l.states[open]
            .iter()
            .flat_map(|&qa| {
                self.l.states[outer]
                    .iter()
                    .filter_map(move |&qb| z.chain.around(base_open, qa, base_outer, qb))
            })
            .reduce(f64::max)?;
        Some(z.weight * upper - (1.0 - z.weight) * 100.0 * around)
    }

    /// Every closed pair whose 5′ edge leaves `u` and whose 3′ edge enters
    /// `v`.
    fn pairs(&self, u: u32, v: u32, out: &mut CellOut) {
        let (i, j1) = (self.l.pos[u as usize], self.l.pos[v as usize]);
        if j1 < i + MIN_HAIRPIN_LOOP + 2 {
            return;
        }
        let j = j1 - 1;
        match self.region.verdict(i, j) {
            Verdict::Allow => {}
            Verdict::Forbid => return,
            Verdict::Penalize(_) => unreachable!("hard constraints only"),
        }
        let mut xs: Vec<Point<Prov>> = Vec::new();
        let x_admit = self.span_admit((u, v));
        for &a in &self.l.out[u as usize] {
            for &b in &self.l.inn[v as usize] {
                let (ea, eb) = (self.l.edge(a), self.l.edge(b));
                let Some(pair) = canonical(ea.nuc, eb.nuc) else {
                    continue;
                };
                if self.fill_at(ea.to, eb.from) == f64::NEG_INFINITY {
                    continue;
                }
                let (o, key) = self.l.pair_key(ea, eb);
                if self.cuts().any(|(cut, _)| cut.rules_out_pair(o, &key)) {
                    continue;
                }
                let Some(aware) = self.aware_lines(ea, eb, &mut out.aware[2]) else {
                    out.aware[0] += 1;
                    continue;
                };
                let mut admit = self.span_admit((ea.from, eb.to));
                for &(lambda, limit) in &aware {
                    admit.line(lambda, limit);
                }
                for (cut, _) in self.cuts() {
                    if let Some(outside) = cut.outside(o, &key) {
                        admit.line(cut.lambda, cut.ceiling + CUT_SLACK - outside);
                    }
                }
                let zero_floors = self.zero_floors((ea, eb), o, &key);
                if let Some((cut, _)) = self.cuts[0] {
                    for &floor in &zero_floors {
                        admit.line(cut.lambda, cut.ceiling + CUT_SLACK - floor);
                    }
                }
                let mut points = Vec::new();
                let mut spelled = Vec::new();
                self.hairpins(ea, eb, pair, (&mut points, &mut spelled), out);
                let t = std::time::Instant::now();
                self.singles(a, b, pair, (&mut points, &admit), out);
                out.clock[1] += t.elapsed().as_secs_f64();
                self.multis(ea, eb, pair, (&mut points, &admit));
                let mut front = self.keep_span(points, (ea.from, eb.to));
                out.aware[1] += self.hold_pair(&mut front, (o, &key), &zero_floors, &aware);
                if front.is_empty() {
                    continue;
                }
                let au = self.terminal_au(pair);
                let (ab, ab_hi) = pack_ids(a, b);
                merge_shift(&mut xs, &front, (au, 0.0), &x_admit, |k| Prov::XPair {
                    ab,
                    ab_hi,
                    k,
                });
                out.p.push(((a, b), front));
                if !spelled.is_empty() {
                    out.spelled.push(((a, b), spelled));
                }
            }
        }
        if !xs.is_empty() {
            out.x = Some(self.keep_span(xs, (u, v)));
        }
    }

    /// The floors [`Self::zero_floor`] puts under the pair `(ea, eb)`.
    fn zero_floors(&self, (ea, eb): (PEdge, PEdge), o: usize, key: &PairKey) -> SmallVec<[f64; 2]> {
        self.zero
            .iter()
            .filter_map(|z| self.zero_floor(z, (ea, eb), o, key))
            .collect()
    }

    /// Hold a closed pair's front, keyed `(o, key)`, to the cuts: each cut's
    /// floor on the pair's completion, the motif-free floors `zero_floors`
    /// at the tie-weight cut, and the constrained-lattice limits `aware`.
    /// Returns how many points the last alone dropped.
    fn hold_pair(
        &self,
        front: &mut Front<Prov>,
        (o, key): (usize, &PairKey),
        zero_floors: &[f64],
        aware: &[(f64, f64)],
    ) -> usize {
        for (cut, _) in self.cuts() {
            cut.retain(front, o, key);
        }
        if let Some((cut, _)) = self.cuts[0] {
            let (lambda, ceiling) = (cut.lambda, cut.ceiling + CUT_SLACK);
            for &floor in zero_floors {
                front.retain(|p| brd(p.mfe, p.cai, lambda) + floor <= ceiling);
            }
        }
        if aware.is_empty() {
            return 0;
        }
        let before = front.len();
        front.retain(|p| aware.iter().all(|&(l, at)| brd(p.mfe, p.cai, l) <= at));
        before - front.len()
    }

    /// Hold the multi-loop cell `table` over `u..v` to the cuts: by the
    /// plain tables' outside floors for `M2` and `M`, by the fold's own
    /// floors for `T` and `L`.
    fn hold_table(&self, table: Table, front: &mut Front<Prov>, (u, v): (u32, u32)) {
        let (inner, outer) = (self.l.base[u as usize], self.l.base[v as usize]);
        for (cut, floors) in self.cuts() {
            match table {
                Table::M2 => cut.retain_span(front, |p| &p.out_m2, inner, outer),
                Table::M => cut.retain_span(front, |p| &p.out_m, inner, outer),
                Table::T => retain_floor(cut, front, &floors.tail, (u, v, self.n())),
                Table::Lead => retain_floor(cut, front, &floors.lead, (u, v, self.n())),
            }
        }
    }

    /// Drop the points of a cell spanning `u..v` that the box's adaptation
    /// floor rules out, as [`Self::keep_span`] does.
    fn hold_floor(&self, front: &mut Front<Prov>, (u, v): (u32, u32)) {
        if let Some(gap) = &self.gap_box {
            let need =
                gap.c_min - self.l.prefix[u as usize] - self.l.suffix[v as usize] - BOX_SLACK;
            front.retain(|p| p.cai >= need);
        }
    }

    /// Hairpins closed by `(ea, eb)` into `points`, with the interiors spelled
    /// whole into `spelled`.
    fn hairpins(
        &self,
        ea: PEdge,
        eb: PEdge,
        closing: BasePair,
        (points, spelled): (&mut Vec<Point<Prov>>, &mut Vec<Spelling>),
        out: &mut CellOut,
    ) {
        let size = self.l.pos[eb.from as usize] - self.l.pos[ea.to as usize];
        if size < MIN_HAIRPIN_LOOP {
            return;
        }
        let outer = ea.cai + eb.cai;
        let mut found = Vec::new();
        if matches!(size, 3 | 4 | 6) {
            for path in self.spellings(ea.to, eb.from) {
                let seq: SmallVec<[Nucleotide; 6]> =
                    path.iter().map(|&e| self.l.edge(e).nuc).collect();
                let cai: f64 = path.iter().map(|&e| self.l.edge(e).cai).sum();
                let energy = hairpin_energy(
                    self.params,
                    &HairpinLoop {
                        closing,
                        loop_seq: &seq,
                    },
                );
                let id = spelled.len() as u32;
                spelled.push(path);
                found.push(Point::new(energy, outer + cai, Prov::HairpinSpelled { id }));
            }
        } else {
            for (first, last, cai) in self.corners(ea.to, eb.from, out) {
                let energy = hairpin_energy_scalar(
                    self.params,
                    closing,
                    size as usize,
                    nuc_of(first),
                    nuc_of(last),
                );
                found.push(Point::new(
                    energy,
                    outer + cai,
                    Prov::Hairpin { first, last },
                ));
            }
        }
        let found = Front::from_points(found);
        merge_into(points, found.points().iter().copied());
    }

    /// Stacks, bulges and internal loops closed by `(a, b)`.
    fn singles(
        &self,
        a: u32,
        b: u32,
        outer: BasePair,
        (points, admit): (&mut Vec<Point<Prov>>, &Admit),
        out: &mut CellOut,
    ) {
        let (ea, eb) = (self.l.edge(a), self.l.edge(b));
        let i = self.l.pos[ea.from as usize];
        let j = self.l.pos[eb.from as usize];
        // Every loop's shifted inner front is gathered and merged in one
        // pass, a batch at a time, in place of one merge per loop point.
        let mut gathered: Vec<Point<Prov>> = Vec::new();
        // Each loop's points and its staircase, in two buffers every inner
        // pair of this pair reuses.
        let (mut loop_points, mut loop_front) = (Vec::new(), Vec::new());
        // Each flank's best adaptation as first read: a left flank by the
        // node it ends at, for the inner position at hand, and a right flank
        // by the node it starts at, per its length. Many inner pairs share a
        // flank, and a flank's best does not change within the fold.
        let mut left_bests = FlankBests::new();
        let mut right_bests: [FlankBests; SINGLE_MAX_LEN as usize + 1] =
            std::array::from_fn(|_| FlankBests::new());
        for k in i + 1..j {
            let n1 = k - i - 1;
            if n1 > SINGLE_MAX_LEN {
                break;
            }
            left_bests.clear();
            // Only inner pairs closing within the loop's reach of `j`.
            let at = &self.p_at[k as usize];
            let reach = (j - 1).saturating_sub(SINGLE_MAX_LEN - n1);
            let from = at.partition_point(|&(close, ..)| close < reach);
            let to = at.partition_point(|&(close, ..)| close < j);
            for &(lpos, (c, d), slot, inner_cai, ref lows) in &at[from..to] {
                let (ec, ed) = (self.l.edge(c), self.l.edge(d));
                let n2 = j - lpos - 1;
                let inner = canonical(ec.nuc, ed.nuc).expect("a closed pair is canonical");
                // Each flank's best adaptation, read in place: most pairs are
                // ruled out on it, before their flanks' corners are copied.
                let left_best = if n1 == 0 {
                    if ea.to != ec.from {
                        continue;
                    }
                    0.0
                } else {
                    flank_best(&mut left_bests, ec.from, || {
                        self.corner_best(ea.to, ec.from, out)
                    })
                };
                let right_best = if n2 == 0 {
                    if ed.to != eb.from {
                        continue;
                    }
                    0.0
                } else {
                    flank_best(&mut right_bests[n2 as usize], ed.to, || {
                        self.corner_best(ed.to, eb.from, out)
                    })
                };
                // A flank with no corner choice has no best.
                if left_best == f64::NEG_INFINITY || right_best == f64::NEG_INFINITY {
                    continue;
                }
                let base = ea.cai + eb.cai;
                // No loop adds more adaptation than its best flanks.
                if inner_cai + base + left_best + right_best < admit.c_min - RULE_OUT_SLACK {
                    continue;
                }
                // No corner choice gives less energy than the loop's floor
                // or more adaptation than its best flanks, and a score rises
                // with energy and falls with adaptation, rounding included:
                // a pair the two rule out together leaves every loop point
                // ruled out below, so its energies are not computed.
                let best = base + (left_best + right_best);
                if admit.rules_out(
                    (inner_cai, lows),
                    (self.loop_floor(outer, inner, (n1, n2)), best),
                ) {
                    continue;
                }
                let lefts: Corners = if n1 == 0 {
                    SmallVec::from_elem((0, 0, 0.0), 1)
                } else {
                    self.corners(ea.to, ec.from, out)
                };
                let rights: Corners = if n2 == 0 {
                    SmallVec::from_elem((0, 0, 0.0), 1)
                } else {
                    self.corners(ed.to, eb.from, out)
                };
                // The loop's own trade-offs: its energy against the best fill
                // of its flanks, one point per corner choice that no other
                // choice beats on both.
                loop_points.clear();
                for &(lf, ll, lc) in &lefts {
                    for &(rf, rl, rc) in &rights {
                        let energy = match (n1, n2) {
                            (0, 0) => stack_energy(self.params, &StackPair { outer, inner }),
                            (0, _) | (_, 0) => bulge_energy(
                                self.params,
                                &BulgeLoop {
                                    outer,
                                    inner,
                                    size: (n1 + n2) as usize,
                                },
                            ),
                            _ => internal_energy_scalar(
                                self.params,
                                &InternalLoopScalar {
                                    outer,
                                    inner,
                                    boundary_left5p: nuc_of(lf),
                                    boundary_left3p: nuc_of(ll),
                                    boundary_right5p: nuc_of(rf),
                                    boundary_right3p: nuc_of(rl),
                                    n1: n1 as usize,
                                    n2: n2 as usize,
                                },
                            ),
                        };
                        loop_points.push(Point::new(energy, lc + rc, [lf, ll, rf, rl]));
                    }
                }
                // A loop point whose shift leaves even the inner cell's least
                // scores past a cut contributes nothing.
                loop_points
                    .retain(|lp| !admit.rules_out((inner_cai, lows), (lp.mfe, base + lp.cai)));
                if loop_points.is_empty() {
                    continue;
                }
                crate::front::staircase_into(&mut loop_points, &mut loop_front);
                let (cd, cd_hi) = pack_ids(c, d);
                let inner = self.p.at(slot).points();
                debug_assert_eq!(
                    inner.last().map_or(f64::NEG_INFINITY, |p| p.cai).to_bits(),
                    inner_cai.to_bits(),
                    "a closed pair's entry holds its cell's best adaptation"
                );
                for lp in &loop_front {
                    let corners = pack_corners(lp.back);
                    let (de, dc) = (lp.mfe, base + lp.cai);
                    let first = admit.first(inner, dc);
                    gathered.extend(
                        inner[first..]
                            .iter()
                            .enumerate()
                            .map(|(k, p)| {
                                let k = (first + k) as u32;
                                Point::new(
                                    p.mfe + de,
                                    p.cai + dc,
                                    Prov::Single {
                                        cd,
                                        cd_hi,
                                        corners,
                                        k,
                                    },
                                )
                            })
                            .filter(|p| admit.admits(p.mfe, p.cai)),
                    );
                }
                if gathered.len() >= GATHER_BATCH {
                    merge_gathered(points, &mut gathered);
                }
            }
        }
        merge_gathered(points, &mut gathered);
    }

    /// The least energy of a loop closed by `outer` around `inner` with
    /// `n1` and `n2` unpaired, over every choice of its boundary
    /// nucleotides: a stack's and a bulge's energy, which no boundary
    /// changes, and an internal loop's least over all four.
    fn loop_floor(&self, outer: BasePair, inner: BasePair, (n1, n2): (u32, u32)) -> Energy {
        match (n1, n2) {
            (0, 0) => stack_energy(self.params, &StackPair { outer, inner }),
            (0, _) | (_, 0) => bulge_energy(
                self.params,
                &BulgeLoop {
                    outer,
                    inner,
                    size: (n1 + n2) as usize,
                },
            ),
            _ => {
                let side = SINGLE_MAX_LEN as usize + 1;
                let at = ((outer as usize * BasePair::ALL.len() + inner as usize) * side
                    + n1 as usize)
                    * side
                    + n2 as usize;
                *self.loop_floors[at].get_or_init(|| {
                    (0..256u32)
                        .map(|c| {
                            let nuc = |shift: u32| nuc_of(((c >> shift) & 3) as u8);
                            internal_energy_scalar(
                                self.params,
                                &InternalLoopScalar {
                                    outer,
                                    inner,
                                    boundary_left5p: nuc(0),
                                    boundary_left3p: nuc(2),
                                    boundary_right5p: nuc(4),
                                    boundary_right3p: nuc(6),
                                    n1: n1 as usize,
                                    n2: n2 as usize,
                                },
                            )
                        })
                        .min()
                        .expect("256 boundary choices")
                })
            }
        }
    }

    fn multis(
        &self,
        ea: PEdge,
        eb: PEdge,
        closing: BasePair,
        (out, admit): (&mut Vec<Point<Prov>>, &Admit),
    ) {
        let Some(core) = self.lead.get(&(ea.to, eb.from)) else {
            return;
        };
        let ml = &self.params.ml;
        let energy = Energy(ml.closing_37 + ml.intern_37) + self.terminal_au(closing);
        merge_shift(out, core, (energy, ea.cai + eb.cai), admit, |k| {
            Prov::Multi { k }
        });
    }

    /// `M`, `M2`, `T` and `L` over `u..v`, from cells already final.
    fn multi_tables(&self, u: u32, v: u32, out: &mut CellOut) {
        let base = Energy(self.params.ml.base_37);
        let intern = Energy(self.params.ml.intern_37);
        let pu = self.l.pos[u as usize];
        let m2_admit = self.table_admit((u, v), |p| &p.out_m2);
        let m_admit = self.table_admit((u, v), |p| &p.out_m);
        let t_admit = self.floor_admit((u, v), |f| &f.tail);
        let lead_admit = self.floor_admit((u, v), |f| &f.lead);

        // Each table's candidates are gathered and merged a batch at a time,
        // in place of one merge per split point or edge.
        let census = split_census();
        let split_started = census.then(std::time::Instant::now);
        let mut enumerated: Vec<(u32, u64)> = Vec::new();
        let (mut m2, mut m2_got) = (Vec::new(), Vec::new());
        let splits = || {
            self.x_into[v as usize]
                .iter()
                .filter(move |&&(w, _)| self.l.pos[w as usize] > pu)
                .filter_map(|&(w, stem)| Some((w, self.m.slot((u, w))?, stem)))
        };
        let split_back = |w: u32| {
            move |l, r| {
                let (w, w_hi) = pack_id(w);
                Prov::Split { w, w_hi, l, r }
            }
        };
        let prune = self.prune;
        let mut path = Vec::new();
        if prune == SplitPrune::Path {
            let mut got = Vec::new();
            for (w, left, stem) in splits() {
                path_sums(
                    &mut got,
                    (self.m.at(left), self.x.at(stem)),
                    (intern, 0.0),
                    &m2_admit,
                    split_back(w),
                );
            }
            merge_gathered(&mut path, &mut got);
        }
        for (w, left, stem) in splits() {
            if prune == SplitPrune::Score && self.split_ruled_out((left, stem), intern, &m2_admit) {
                out.split.ruled_out += 1;
                let sums = if count_split_sums() {
                    row_sums(self.m.at(left), self.x.at(stem), 0.0, &m2_admit)
                } else {
                    0
                };
                out.split.skipped += sums;
                if census {
                    enumerated.push((w, sums));
                }
                continue;
            }
            let reference = match prune {
                SplitPrune::Off => Reference::None,
                SplitPrune::Acc => {
                    merge_gathered(&mut m2, &mut m2_got);
                    Reference::Acc
                }
                SplitPrune::Path => Reference::Given(&path),
                SplitPrune::Score => Reference::None,
            };
            let by_score = (
                self.by_score(&self.m_score, &self.m, left),
                self.by_score(&self.x_score, &self.x, stem),
            );
            let (sums, formed, merge_secs) = gather_compose(
                (&mut m2, &mut m2_got),
                (self.m.at(left), self.x.at(stem)),
                (intern, 0.0),
                &m2_admit,
                (reference, by_score),
                split_back(w),
            );
            out.split.skipped += sums.saturating_sub(formed);
            if census {
                enumerated.push((w, sums));
                out.split.formed += formed;
                out.split.merge_secs += merge_secs;
            }
        }
        if let Some(t) = split_started {
            let m = std::time::Instant::now();
            merge_gathered(&mut m2, &mut m2_got);
            out.split.merge_secs += m.elapsed().as_secs_f64();
            out.split.secs += t.elapsed().as_secs_f64();
        } else {
            merge_gathered(&mut m2, &mut m2_got);
        }
        let post_started = census.then(std::time::Instant::now);
        let (mut m, mut m_got) = (Vec::new(), Vec::new());
        if let Some(stem) = &out.x {
            gather_shift((&mut m, &mut m_got), stem, (intern, 0.0), &m_admit, |k| {
                Prov::MBranch { k }
            });
        }
        let (mut t, mut t_got) = (Vec::new(), Vec::new());
        if !m2.is_empty() {
            let mut m2 = self.keep_span(m2, (u, v));
            self.hold_table(Table::M2, &mut m2, (u, v));
            if !m2.is_empty() {
                let zero = (Energy::ZERO, 0.0);
                gather_shift((&mut m, &mut m_got), &m2, zero, &m_admit, |k| {
                    Prov::MFromM2 { k }
                });
                gather_shift((&mut t, &mut t_got), &m2, zero, &t_admit, |k| Prov::TCore {
                    k,
                });
                out.m2 = Some(m2);
            }
        }
        if let Some(t) = post_started {
            out.split.post_secs += t.elapsed().as_secs_f64();
            let mut kept: Vec<u32> = out
                .m2
                .iter()
                .flat_map(|f| f.points().iter())
                .filter_map(|p| match p.back {
                    Prov::Split { w, w_hi, .. } => Some(unpack_id(w, w_hi)),
                    _ => None,
                })
                .collect();
            out.split.cell(&enumerated, &mut kept);
            if let Some(f) = &out.m2 {
                out.split.digest = out
                    .split
                    .digest
                    .wrapping_add(front_digest((0, u, v), f.points()));
            }
        }
        for &e in &self.l.inn[v as usize] {
            let edge = self.l.edge(e);
            if let Some(rest) = self.m.get(&(u, edge.from)) {
                gather_shift(
                    (&mut m, &mut m_got),
                    rest,
                    (base, edge.cai),
                    &m_admit,
                    |k| Prov::MUnpaired { e, k },
                );
            }
            if let Some(rest) = self.t.get(&(u, edge.from)) {
                gather_shift(
                    (&mut t, &mut t_got),
                    rest,
                    (base, edge.cai),
                    &t_admit,
                    |k| Prov::TUnpaired { e, k },
                );
            }
        }
        merge_gathered(&mut m, &mut m_got);
        merge_gathered(&mut t, &mut t_got);
        if !m.is_empty() {
            let mut m = self.keep_span(m, (u, v));
            self.hold_table(Table::M, &mut m, (u, v));
            if !m.is_empty() {
                out.m = Some(m);
            }
        }
        if !t.is_empty() {
            let mut t = self.keep_span(t, (u, v));
            self.hold_table(Table::T, &mut t, (u, v));
            if !t.is_empty() {
                out.t = Some(t);
            }
        }

        let (mut lead, mut lead_got) = (Vec::new(), Vec::new());
        if let Some(core) = &out.t {
            gather_shift(
                (&mut lead, &mut lead_got),
                core,
                (Energy::ZERO, 0.0),
                &lead_admit,
                |k| Prov::LCore { k },
            );
        }
        for &e in &self.l.out[u as usize] {
            let edge = self.l.edge(e);
            if let Some(rest) = self.lead.get(&(edge.to, v)) {
                gather_shift(
                    (&mut lead, &mut lead_got),
                    rest,
                    (base, edge.cai),
                    &lead_admit,
                    |k| Prov::LUnpaired { e, k },
                );
            }
        }
        merge_gathered(&mut lead, &mut lead_got);
        if !lead.is_empty() {
            let mut lead = self.keep_span(lead, (u, v));
            self.hold_table(Table::Lead, &mut lead, (u, v));
            if !lead.is_empty() {
                out.lead = Some(lead);
            }
        }
    }

    /// The cuts this fold holds its cells to.
    fn cuts(&self) -> impl Iterator<Item = (&'a Cut<'a>, &'a Floors)> {
        self.cuts.into_iter().flatten()
    }

    fn exterior(&mut self) {
        self.z[self.l.start as usize] = Front::singleton(Energy::ZERO, 0.0, Prov::Start);
        for p in 1..=self.l.seq_len {
            for &v in &self.l.at[p as usize] {
                let (mut z, mut got) = (Vec::new(), Vec::new());
                let admit = self.exterior_admit(v);
                let census = split_census();
                let mut enumerated: Vec<(u32, u64)> = Vec::new();
                for &e in &self.l.inn[v as usize] {
                    let edge = self.l.edge(e);
                    gather_shift(
                        (&mut z, &mut got),
                        &self.z[edge.from as usize],
                        (Energy::ZERO, edge.cai),
                        &admit,
                        |k| Prov::ZUnpaired { e, k },
                    );
                }
                let split_started = census.then(std::time::Instant::now);
                let z_back = |x: u32| {
                    move |l, r| {
                        let (x, x_hi) = pack_id(x);
                        Prov::ZPair { x, x_hi, l, r }
                    }
                };
                let prune = self.prune;
                let mut path = Vec::new();
                if prune == SplitPrune::Path {
                    let mut sums = Vec::new();
                    for &(x, slot) in &self.x_into[v as usize] {
                        let pair = self.x.at(slot);
                        let left = &self.z[x as usize];
                        path_sums(
                            &mut sums,
                            (left, pair),
                            (Energy::ZERO, 0.0),
                            &admit,
                            z_back(x),
                        );
                    }
                    merge_gathered(&mut path, &mut sums);
                }
                for &(x, slot) in &self.x_into[v as usize] {
                    let pair = self.x.at(slot);
                    let reference = match prune {
                        SplitPrune::Off => Reference::None,
                        SplitPrune::Acc => {
                            merge_gathered(&mut z, &mut got);
                            Reference::Acc
                        }
                        SplitPrune::Path => Reference::Given(&path),
                        SplitPrune::Score => Reference::None,
                    };
                    let by_score = (
                        self.z_score.get(x as usize).and_then(Option::as_ref),
                        self.by_score(&self.x_score, &self.x, slot),
                    );
                    let (sums, formed, merge_secs) = gather_compose(
                        (&mut z, &mut got),
                        (&self.z[x as usize], pair),
                        (Energy::ZERO, 0.0),
                        &admit,
                        (reference, by_score),
                        z_back(x),
                    );
                    self.split[1].skipped += sums.saturating_sub(formed);
                    if census && !self.z[x as usize].is_empty() {
                        enumerated.push((x, sums));
                        self.split[1].formed += formed;
                        self.split[1].merge_secs += merge_secs;
                    }
                }
                if let Some(t) = split_started {
                    let m = std::time::Instant::now();
                    merge_gathered(&mut z, &mut got);
                    self.split[1].merge_secs += m.elapsed().as_secs_f64();
                    self.split[1].secs += t.elapsed().as_secs_f64();
                } else {
                    merge_gathered(&mut z, &mut got);
                }
                let mut z = self.keep_span(z, (self.l.start, v));
                for (cut, _) in self.cuts() {
                    if let Some(outside) = cut.exterior_outside(self.l.base[v as usize]) {
                        let (lambda, ceiling) = (cut.lambda, cut.ceiling + CUT_SLACK);
                        z.retain(|p| brd(p.mfe, p.cai, lambda) + outside <= ceiling);
                    }
                }
                if census {
                    let mut kept: Vec<u32> = z
                        .points()
                        .iter()
                        .filter_map(|p| match p.back {
                            Prov::ZPair { x, x_hi, .. } => Some(unpack_id(x, x_hi)),
                            _ => None,
                        })
                        .collect();
                    self.split[1].cell(&enumerated, &mut kept);
                    self.split[1].digest = self.split[1]
                        .digest
                        .wrapping_add(front_digest((1, 0, v), z.points()));
                }
                self.stored += z.len();
                if self.scored() {
                    if self.z_score.len() < self.z.len() {
                        self.z_score.resize_with(self.z.len(), || None);
                    }
                    let by = ByScore::new(&z, self.cuts().map(|(cut, _)| cut.lambda));
                    self.z_score[v as usize] = Some(by);
                }
                self.z[v as usize] = z;
            }
        }
        let mut end = Vec::new();
        for &v in &self.l.ends {
            merge_shift(
                &mut end,
                &self.z[v as usize],
                (Energy::ZERO, 0.0),
                &Admit::all(),
                |k| Prov::End { v, k },
            );
        }
        self.end = self.keep(end);
        // A cell's energy can still fall as it is closed, so the ceiling
        // applies to whole designs only.
        if let Some(c) = self.ceiling {
            self.end.truncate_mfe(c);
        }
    }

    /// Every point of the front with its sequence and structure.
    fn recover(&self, tally: &mut RuleTally) -> Vec<(Energy, f64, String, String)> {
        self.end
            .points()
            .iter()
            .map(|p| {
                let mut out = Spelled::new(self.l.seq_len as usize);
                self.walk_point(p.back, &mut out);
                add_tally(tally, &out.tally);
                (p.mfe, p.cai, out.sequence(), out.structure())
            })
            .collect()
    }

    fn walk_point(&self, back: Prov, out: &mut Spelled) {
        match back {
            Prov::End { v, k } => self.walk_z(v, k, out),
            _ => unreachable!("the front's points are terminal"),
        }
    }

    fn walk_z(&self, v: u32, k: u32, out: &mut Spelled) {
        match self.z[v as usize].points()[k as usize].back {
            Prov::Start => {}
            Prov::ZUnpaired { e, k } => {
                out.edge(self.l, e);
                self.walk_z(self.l.edge(e).from, k, out);
            }
            Prov::ZPair { x, x_hi, l, r } => {
                let x = unpack_id(x, x_hi);
                out.tally.exterior_stems += 1;
                self.walk_z(x, l, out);
                self.walk_x(x, v, r, out);
            }
            _ => unreachable!("an exterior point"),
        }
    }

    fn walk_x(&self, u: u32, v: u32, k: u32, out: &mut Spelled) {
        match self.x[&(u, v)].points()[k as usize].back {
            Prov::XPair { ab, ab_hi, k } => {
                let (a, b) = unpack_ids(ab, ab_hi);
                self.walk_p(a, b, k, out);
            }
            _ => unreachable!("a stem point"),
        }
    }

    fn walk_p(&self, a: u32, b: u32, k: u32, out: &mut Spelled) {
        let (ea, eb) = (self.l.edge(a), self.l.edge(b));
        out.edge(self.l, a);
        out.edge(self.l, b);
        out.pair(self.l.pos_of_edge(a), self.l.pos_of_edge(b));
        match self.p[&(a, b)].points()[k as usize].back {
            Prov::Hairpin { first, last } => {
                out.tally.hairpins += 1;
                self.walk_corner(ea.to, eb.from, first, last, out);
            }
            Prov::HairpinSpelled { id } => {
                out.tally.spelled_hairpins += 1;
                for &e in &self.spelled[&(a, b)][id as usize] {
                    out.edge(self.l, e);
                }
            }
            Prov::Single {
                cd,
                cd_hi,
                corners,
                k,
            } => {
                let (c, d) = unpack_ids(cd, cd_hi);
                let corners = unpack_corners(corners);
                let (ec, ed) = (self.l.edge(c), self.l.edge(d));
                match (ea.to == ec.from, ed.to == eb.from) {
                    (true, true) => out.tally.stacks += 1,
                    (false, false) => out.tally.internals += 1,
                    _ => out.tally.bulges += 1,
                }
                if ea.to != ec.from {
                    self.walk_corner(ea.to, ec.from, corners[0], corners[1], out);
                }
                if ed.to != eb.from {
                    self.walk_corner(ed.to, eb.from, corners[2], corners[3], out);
                }
                self.walk_p(c, d, k, out);
            }
            Prov::Multi { k } => {
                out.tally.multis += 1;
                self.walk_lead(ea.to, eb.from, k, out);
            }
            _ => unreachable!("a closed-pair point"),
        }
    }

    fn walk_m(&self, u: u32, v: u32, k: u32, out: &mut Spelled) {
        match self.m[&(u, v)].points()[k as usize].back {
            Prov::MBranch { k } => self.walk_x(u, v, k, out),
            Prov::MUnpaired { e, k } => {
                out.edge(self.l, e);
                self.walk_m(u, self.l.edge(e).from, k, out);
            }
            Prov::MFromM2 { k } => self.walk_m2(u, v, k, out),
            _ => unreachable!("a multi-branch point"),
        }
    }

    fn walk_m2(&self, u: u32, v: u32, k: u32, out: &mut Spelled) {
        match self.m2[&(u, v)].points()[k as usize].back {
            Prov::Split { w, w_hi, l, r } => {
                let w = unpack_id(w, w_hi);
                self.walk_m(u, w, l, out);
                self.walk_x(w, v, r, out);
            }
            _ => unreachable!("a two-branch point"),
        }
    }

    fn walk_t(&self, u: u32, v: u32, k: u32, out: &mut Spelled) {
        match self.t[&(u, v)].points()[k as usize].back {
            Prov::TCore { k } => self.walk_m2(u, v, k, out),
            Prov::TUnpaired { e, k } => {
                out.edge(self.l, e);
                self.walk_t(u, self.l.edge(e).from, k, out);
            }
            _ => unreachable!("a multi-loop core point"),
        }
    }

    fn walk_lead(&self, u: u32, v: u32, k: u32, out: &mut Spelled) {
        match self.lead[&(u, v)].points()[k as usize].back {
            Prov::LCore { k } => self.walk_t(u, v, k, out),
            Prov::LUnpaired { e, k } => {
                out.edge(self.l, e);
                self.walk_lead(self.l.edge(e).to, v, k, out);
            }
            _ => unreachable!("a multi-loop interior point"),
        }
    }

    /// Respell the best fill from `u` to `v` with the given corner
    /// nucleotides.
    fn walk_corner(&self, u: u32, v: u32, first: u8, last: u8, out: &mut Spelled) {
        let len = self.l.pos[v as usize] - self.l.pos[u as usize];
        let mut best: Option<(f64, u32, u32)> = None;
        for &e1 in &self.l.out[u as usize] {
            for &e2 in &self.l.inn[v as usize] {
                let (a, b) = (self.l.edge(e1), self.l.edge(e2));
                if a.nuc as u8 != first || b.nuc as u8 != last {
                    continue;
                }
                let cai = if len == 1 {
                    if e1 != e2 {
                        continue;
                    }
                    a.cai
                } else {
                    let mid = self.fill_at(a.to, b.from);
                    if mid == f64::NEG_INFINITY {
                        continue;
                    }
                    a.cai + mid + b.cai
                };
                if best.is_none_or(|(c, _, _)| cai > c) {
                    best = Some((cai, e1, e2));
                }
            }
        }
        let (_, e1, e2) = best.expect("a corner the fold used");
        out.edge(self.l, e1);
        if len > 1 {
            out.edge(self.l, e2);
            self.walk_fill(self.l.edge(e1).to, self.l.edge(e2).from, out);
        }
    }

    /// Respell the best fill from `u` to `v`.
    fn walk_fill(&self, mut u: u32, v: u32, out: &mut Spelled) {
        while u != v {
            let want = self.fill_at(u, v);
            let e = *self.l.out[u as usize]
                .iter()
                .find(|&&e| {
                    let edge = self.l.edge(e);
                    edge.cai + self.fill_at(edge.to, v) == want
                })
                .expect("the best fill's next edge");
            out.edge(self.l, e);
            u = self.l.edge(e).to;
        }
    }
}

/// Floors under what the multi-loop interior tables `L` and `T` can still
/// add once closed, at one cut's weight, `[u * n + v]`; `-∞` where none is
/// known. They depend on the cut's weight and passes, not on its ceiling, so
/// every fold under the same weight reads one set.
struct Floors {
    lead: Vec<f64>,
    tail: Vec<f64>,
}

impl Floors {
    /// The floors at `cut`'s weight.
    ///
    /// A cell is cut only by a floor over **every** cell built from it, since
    /// cutting it removes those derivations too. `L(u, v)` is closed by a pair
    /// whose 5′ edge enters `u` and 3′ edge leaves `v`, or is extended by one
    /// unpaired edge into `L(u₀, v)`; `T(x, y)` becomes `L(x, y)`, or is
    /// extended by one unpaired edge into `T(x, y′)`. So each floor is the
    /// least of its direct uses and its extensions' floors plus the edge, swept
    /// in the direction the extensions run.
    fn new(params: &Params, l: &Lattice, region: &PairConstraint, cut: &Cut<'_>) -> Self {
        let n = l.n();
        let base = Energy(params.ml.base_37);
        let lambda = cut.lambda;
        let unpaired = |e: u32| brd(base, l.edge(e).cai, lambda);
        let mut lead = vec![f64::INFINITY; n * n];
        let mut tail = vec![f64::INFINITY; n * n];
        let order: Vec<u32> = l.at.iter().flatten().copied().collect();
        for &v in &order {
            let pv = l.pos[v as usize];
            for &u in order.iter().take_while(|&&u| l.pos[u as usize] < pv) {
                let mut floor =
                    closing_floor(params, l, region, cut, u, v).unwrap_or(f64::NEG_INFINITY);
                for &e in &l.inn[u as usize] {
                    let from = l.edge(e).from;
                    floor = floor.min(unpaired(e) + lead[from as usize * n + v as usize]);
                }
                lead[u as usize * n + v as usize] = floor;
            }
        }
        for &x in &order {
            let px = l.pos[x as usize];
            for &y in order.iter().rev().take_while(|&&y| l.pos[y as usize] > px) {
                let mut floor = lead[x as usize * n + y as usize];
                for &e in &l.out[y as usize] {
                    let to = l.edge(e).to;
                    floor = floor.min(unpaired(e) + tail[x as usize * n + to as usize]);
                }
                tail[x as usize * n + y as usize] = floor;
            }
        }
        Self { lead, tail }
    }
}

/// The least a multi-loop interior from `u` to `v` adds once closed: the
/// best, over the pairs that can close it (5′ edge into `u`, 3′ edge out of
/// `v`), of the closing cost and the pair's plain completion, at the cut's
/// weight. `+∞` when no pair can close it; `None` where the plain passes do
/// not cover some closing pair, so no floor is known.
fn closing_floor(
    params: &Params,
    l: &Lattice,
    region: &PairConstraint,
    cut: &Cut<'_>,
    u: u32,
    v: u32,
) -> Option<f64> {
    let ml = &params.ml;
    let mut floor = f64::INFINITY;
    for &a in &l.inn[u as usize] {
        for &b in &l.out[v as usize] {
            let (ea, eb) = (l.edge(a), l.edge(b));
            let Some(pair) = canonical(ea.nuc, eb.nuc) else {
                continue;
            };
            let (i, j) = (l.pos[ea.from as usize], l.pos[eb.from as usize]);
            if matches!(region.verdict(i, j), Verdict::Forbid) {
                continue;
            }
            let (o, key) = l.pair_key(ea, eb);
            let outside = cut.chord(|p| Some(p.table.get(o)?.get(&key)?.outside))?;
            let closing = Energy(ml.closing_37 + ml.intern_37) + terminal_au(params, pair);
            floor = floor.min(brd(closing, ea.cai + eb.cai, cut.lambda) + outside);
        }
    }
    Some(floor)
}

/// Drop the points of the cell `u..v` that its floor under `cut` puts above
/// the cut's ceiling.
fn retain_floor(
    cut: &Cut<'_>,
    front: &mut Front<Prov>,
    floors: &[f64],
    (u, v, n): (u32, u32, usize),
) {
    let floor = floors[u as usize * n + v as usize];
    if floor == f64::NEG_INFINITY {
        return;
    }
    let (lambda, ceiling) = (cut.lambda, cut.ceiling + CUT_SLACK);
    front.retain(|p| brd(p.mfe, p.cai, lambda) + floor <= ceiling);
}

/// A design being respelled: the nucleotide at each position and the pairs.
struct Spelled {
    seq: Vec<Option<Nucleotide>>,
    pairs: Vec<(u32, u32)>,
    tally: RuleTally,
}

fn add_tally(sum: &mut RuleTally, t: &RuleTally) {
    sum.hairpins += t.hairpins;
    sum.spelled_hairpins += t.spelled_hairpins;
    sum.stacks += t.stacks;
    sum.bulges += t.bulges;
    sum.internals += t.internals;
    sum.multis += t.multis;
    sum.exterior_stems += t.exterior_stems;
}

impl Spelled {
    fn new(len: usize) -> Self {
        Self {
            seq: vec![None; len],
            pairs: Vec::new(),
            tally: RuleTally::default(),
        }
    }

    fn edge(&mut self, l: &Lattice, e: u32) {
        let edge = l.edge(e);
        let slot = &mut self.seq[l.pos[edge.from as usize] as usize];
        debug_assert!(
            slot.is_none_or(|n| n == edge.nuc),
            "a position spelled twice"
        );
        *slot = Some(edge.nuc);
    }

    fn pair(&mut self, i: u32, j: u32) {
        self.pairs.push((i, j));
    }

    fn sequence(&self) -> String {
        self.seq
            .iter()
            .map(|n| match n.expect("every position spelled") {
                Nucleotide::A => 'A',
                Nucleotide::C => 'C',
                Nucleotide::G => 'G',
                Nucleotide::U => 'U',
            })
            .collect()
    }

    fn structure(&self) -> String {
        let mut s = vec!['.'; self.seq.len()];
        for &(i, j) in &self.pairs {
            s[i as usize] = '(';
            s[j as usize] = ')';
        }
        s.into_iter().collect()
    }
}

/// Slack in the cut's comparison: the plain tables and this fold sum the same
/// terms in different orders, so equal objectives can differ in the last bits.
const CUT_SLACK: f64 = 1e-6;

/// The weighted objective of `(mfe, cai)` at `lambda`, in the scan's units.
fn brd(mfe: Energy, cai: f64, lambda: f64) -> f64 {
    lambda * f64::from(mfe.0) - (1.0 - lambda) * cai * 100.0
}

/// One plain-lattice pass at a fixed weight: the best completion of every
/// closed pair and multi-branch cell, and the best design through every
/// closed pair.
///
/// Each value is a minimum over designs of a quantity affine in the weight,
/// so it is concave in the weight, and the chord between two passes
/// bracketing a weight lies below its value there: a floor, which is what a
/// cut needs.
struct Pass {
    /// The lattice's dense node layout, which the outside tables are indexed
    /// by.
    layout: NodeLayout,
    lambda: f64,
    /// The plain optimum at this weight, on the BRD scale; infinite where
    /// the plain lattice has no design.
    optimum: f64,
    /// The plain best completion of each closed pair and the plain best
    /// design through it, by the column it closes into.
    table: Vec<PairColumn>,
    /// The plain best completion of each multi-branch cell with one or more
    /// branches, and with two or more, as `[outer][inner]`.
    out_m: Vec<Vec<f64>>,
    out_m2: Vec<Vec<f64>>,
    /// The plain best completion of the exterior up to each node.
    out_c: Vec<f64>,
    /// The same floors over the constrained lattice, at the low weights
    /// where what the motif set forbids is most of what the plain floors
    /// credit ([`aware_upto`]); `None` elsewhere.
    aware: Option<AwarePass>,
    /// Whether `aware` was read from the cache rather than taken.
    aware_read: bool,
}

impl Pass {
    fn new(
        params: &Params,
        graph: &Dfa,
        region: &PairConstraint,
        lambda: f64,
        threads: usize,
    ) -> Self {
        let config = crate::ScanConfig {
            threads,
            ..crate::ScanConfig::default()
        };
        let must = crate::must_pair::MustPairConstraint::inert();
        let tables =
            crate::scan::scan_with_constraints(params, graph, lambda, config, region.clone(), must);
        let costs = crate::outside_scan::OutsideScan::cost_to_go(
            params,
            graph,
            &tables,
            region,
            config.single_max_len,
            threads,
            false,
        );
        Self::of(graph, lambda, &tables, costs)
    }

    /// The passes at each of `lambdas`, from one traversal of the lattice and
    /// one backward sweep; each equals [`Pass::new`] at its weight with one
    /// thread.
    fn lanes(params: &Params, graph: &Dfa, region: &PairConstraint, lambdas: &[f64]) -> Vec<Self> {
        let config = crate::ScanConfig::default();
        let (tables, _) = crate::outside_multi::scan_lanes(params, graph, lambdas, config, region);
        let refs: Vec<&crate::tables::Tables> = tables.iter().collect();
        let (costs, _) = crate::outside_multi::cost_to_go_lanes(
            params,
            graph,
            &refs,
            region,
            config.single_max_len,
            1,
        );
        lambdas
            .iter()
            .zip(&tables)
            .zip(costs)
            .map(|((&lambda, tables), costs)| Self::of(graph, lambda, tables, costs))
            .collect()
    }

    /// The pass at `lambda` from its scan and its cost to go.
    fn of(
        graph: &Dfa,
        lambda: f64,
        tables: &crate::tables::Tables,
        costs: crate::outside_scan::CellCosts,
    ) -> Self {
        let optimum = tables.c[graph.ix(graph.end())].map_or(f64::INFINITY, |c| c.brd(lambda));
        let table = crate::constrained::pair_table(costs.p, |o, key| {
            tables.p[o].get(key).map(|s| s.brd(lambda))
        });
        Self {
            layout: graph.layout(),
            lambda,
            optimum,
            table,
            out_m: costs.m,
            out_m2: costs.m2,
            out_c: costs.c,
            aware: None,
            aware_read: false,
        }
    }
}

/// The highest grid weight whose pass also runs over the constrained
/// lattice ([`AwarePass`]): every grid weight above 0. A plain floor credits
/// completions the motif set forbids, most at low weight; on a complete
/// 349-residue motif front the passes at every weight took 4.1% fewer
/// instructions than those up to 0.07 alone, with the same front.
/// `DERNA_MOTIF_AWARE_UPTO` overrides it; 0 takes none.
const AWARE_UPTO: f64 = 1.0;

/// The most plain passes the grid takes in one traversal of the lattice. On
/// a 120-residue motif walk four passes in one traversal cost 0.42 of four
/// alone and raised the peak from 243 to 311 MB; eight cost 0.33 for a peak
/// of 468 MB, the lanes' tables being held at once.
const PASS_LANES: usize = 4;

/// [`PASS_LANES`], or the count a development build gives, from 1 (each pass alone)
/// to [`crate::outside_multi::MAX_LANES`].
fn pass_lanes() -> usize {
    crate::dev_env::var(crate::dev_env::Setting::MotifPassLanes)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(PASS_LANES)
        .clamp(1, crate::outside_multi::MAX_LANES)
}

fn aware_upto() -> f64 {
    std::env::var("DERNA_MOTIF_AWARE_UPTO")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(AWARE_UPTO)
}

/// How far above the floor on the constrained optimum an aware pass's
/// ceiling search starts, in BRD units: the pass is exact for every pair
/// with a design under the ceiling, so a wider one serves more of the folds
/// bracketing the weight, for a costlier fold. A development build can
/// override it.
const AWARE_MARGIN: f64 = 30.0;

fn aware_margin() -> f64 {
    crate::dev_env::var(crate::dev_env::Setting::MotifAwareMargin)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(AWARE_MARGIN)
}

/// How far above the constrained optimum an aware pass's ceiling aims, in BRD
/// units: after a failed fold the search lifts its ceilings by it, and a
/// search that certifies closer than half of it is folded once more at it
/// ([`crate::constrained::outside::aware_pass`]). Where the plain
/// floor sits far below the constrained optimum, as at the low weights of a
/// long molecule, the doubling search certifies close above the optimum, and
/// the gaps bracketing the weight read no aware floor below their ceilings.
/// A development build can override it.
const AWARE_HEADROOM: f64 = 30.0;

/// The widest box, in kcal/mol, that a pair of adjacent grid optima is
/// folded at once rather than probed: between two seeds a single gap fold
/// cost 0.13-0.66 of the probes and sub-gaps it replaces at 200 and 300
/// residues, where every such box spanned under 57 kcal/mol. Across boxes
/// that span several seeds the fold can cost far more: 7.4 times the probes
/// and sub-gaps over 54 kcal/mol holding 15 vertices at weights 0.02-0.06,
/// where its slices outgrew the store. Only adjacent grid optima are folded
/// whole for that reason.
/// A development build can override it.
const CELL_FOLD_DE: f64 = 60.0;

fn cell_fold_de() -> f64 {
    crate::dev_env::var(crate::dev_env::Setting::MotifCellFoldDe)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CELL_FOLD_DE)
}

/// The lowest tie weight at which a narrow box between adjacent grid optima
/// is folded whole; below it the box is probed. What a whole fold costs
/// against the probes and sub-gaps it replaces follows the weight, not the
/// box's span: on the first 300 residues of Q06187 the boxes folded whole
/// below a tie weight of 0.10 cost 2.58 times their split folds and those
/// above it 1.05 times, so folding every box under 60 kcal/mol whole cost
/// 1.40 times the instructions there and 1.50 times at 400 residues, while it
/// saved 13-17% on P47211.
/// A development build can override it.
const CELL_FOLD_TIE: f64 = 0.10;

fn cell_fold_tie() -> f64 {
    crate::dev_env::var(crate::dev_env::Setting::MotifCellFoldTie)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CELL_FOLD_TIE)
}

fn aware_headroom() -> f64 {
    crate::dev_env::var(crate::dev_env::Setting::MotifAwareHeadroom)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(AWARE_HEADROOM)
}

/// What a grid needs to take its aware passes: the constrained lattice, the
/// chain tables and the open chain's adaptation, which give the exact
/// motif-free floors at weight 0, and the weights and margin they are taken
/// at.
#[derive(Clone, Copy)]
struct AwareSource<'g> {
    product: &'g ProductDfa<'g>,
    chain: Option<(&'g ChainCai, f64)>,
    upto: f64,
    margin: f64,
    /// How far above its optimum each pass's ceiling reaches at least.
    headroom: f64,
    /// Whether each pass keeps a design at its optimum, to seed the walk.
    seed: bool,
}

impl AwareSource<'_> {
    /// Whether the pass at `lambda` also runs over the constrained lattice.
    fn covers(&self, lambda: f64) -> bool {
        lambda > 0.0 && lambda <= self.upto + 1e-12
    }

    /// The aware pass beside the plain `pass`: pruned as a probe at its own
    /// weight would be, on the plain pass and the chords to weight 0 against
    /// `top`, the pass at weight 1.
    fn pass(
        &self,
        params: &Params,
        region: &PairConstraint,
        (pass, top): (&Pass, &Pass),
        threads: usize,
    ) -> Option<AwarePass> {
        let (zero, floor) = zero_chords(self.chain, &[Some(top)], pass.lambda, pass.optimum);
        let on = crate::constrained::outside::AwareLattice {
            product: self.product,
            params,
            region: region.clone(),
        };
        crate::constrained::outside::aware_pass(
            &on,
            pass.lambda,
            |ceiling| crate::constrained::OutsideBound {
                outside: &pass.table,
                chord: None,
                through: true,
                zero: &zero,
                motif_free: None,
                aware: None,
                ceiling,
            },
            (floor, self.margin.max(first_margin(floor)), self.headroom),
            threads,
            self.seed,
        )
    }

    /// What an aware pass depends on beyond the plain lattice's
    /// [`fingerprint`], its weight and the weight of its `top`, as one
    /// number: every transition of the constrained lattice, both as the fold
    /// threads it and as the raw automaton reads it, from every state reachable
    /// at every node; the open chain's adaptation; and the margin, headroom
    /// and seeding the pass is taken with. The chain tables are built from the
    /// lattice alone, and the plain passes from what [`fingerprint`] covers.
    fn fingerprint(&self) -> u64 {
        let mut h = Fnv::new();
        h.eat(AWARE_PRINT);
        let product = self.product;
        let graph = product.dfa();
        let (lattice, ac) = (product.lattice_ac(), product.ac());
        h.eat(&[u8::from(product.is_merged()), u8::from(self.seed)]);
        h.eat(&(lattice.num_states() as u64).to_le_bytes());
        h.eat(&lattice.start().to_le_bytes());
        h.eat(format!("{:?}", product.window()).as_bytes());
        let step = |q: Option<u32>| q.map_or(u64::MAX, u64::from).to_le_bytes();
        for pos in 0..=graph.seq_len {
            for &node in graph.nodes_at(pos) {
                let states = product.reachable_states(node);
                h.eat(&[node.lane]);
                h.eat(&(states.len() as u64).to_le_bytes());
                for &q in states {
                    h.eat(&q.to_le_bytes());
                    for &e in graph.right(node) {
                        h.eat(&step(lattice.step_edge(node, q, e)));
                        h.eat(&step(ac.step_at(pos, q, e.nuc)));
                    }
                }
            }
        }
        match self.chain {
            Some((_, open_chain)) => h.eat(&open_chain.to_bits().to_le_bytes()),
            None => h.eat(b"no chain"),
        }
        h.eat(&self.margin.to_bits().to_le_bytes());
        h.eat(&self.headroom.to_bits().to_le_bytes());
        h.0
    }
}

/// FNV-1a over bytes, for the cache's keys.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn eat(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
}

/// What the aware passes' key starts from, so no aware key equals a plain
/// one by construction.
const AWARE_PRINT: &[u8] = b"derna-motif-aware-1";

/// The lattice a pass is taken over, as one number: every edge of `graph`,
/// the pair constraint and the energy parameters, by FNV-1a. A cached pass is
/// read only under the same fingerprint and weight.
fn fingerprint(params: &Params, graph: &Dfa, region: &PairConstraint) -> u64 {
    let mut h = Fnv::new();
    let mut eat = |bytes: &[u8]| h.eat(bytes);
    eat(PASS_FORMAT);
    eat(&graph.seq_len.to_le_bytes());
    for pos in 0..=graph.seq_len {
        for &node in graph.nodes_at(pos) {
            eat(&[node.lane]);
            for e in graph.right(node) {
                eat(&e.to.pos.to_le_bytes());
                eat(&[e.to.lane, e.nuc as u8]);
                eat(&e.cai.to_bits().to_le_bytes());
            }
        }
    }
    eat(format!("{region:?}").as_bytes());
    eat(format!("{params:?}").as_bytes());
    h.0
}

/// The cached pass's format, read first from every file and folded into the
/// fingerprint.
const PASS_FORMAT: &[u8] = b"derna-motif-pass-5";

impl Pass {
    /// Keep the pass in `file`, every value bit for bit, written whole to a
    /// temporary file and renamed so a reader never sees half of one.
    fn write(&self, file: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let tmp = file.with_extension(format!("tmp{}", std::process::id()));
        let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        let f64s = |w: &mut std::io::BufWriter<std::fs::File>, v: &[f64]| -> std::io::Result<()> {
            w.write_all(&(v.len() as u64).to_le_bytes())?;
            for x in v {
                w.write_all(&x.to_bits().to_le_bytes())?;
            }
            Ok(())
        };
        let maps =
            |w: &mut std::io::BufWriter<std::fs::File>, v: &[PairColumn]| -> std::io::Result<()> {
                w.write_all(&(v.len() as u64).to_le_bytes())?;
                for m in v {
                    w.write_all(&(m.len() as u64).to_le_bytes())?;
                    for (k, [outside, through]) in m.packed() {
                        w.write_all(&k.to_le_bytes())?;
                        w.write_all(&outside.to_le_bytes())?;
                        w.write_all(&through.to_le_bytes())?;
                    }
                }
                Ok(())
            };
        w.write_all(PASS_FORMAT)?;
        w.write_all(&self.lambda.to_bits().to_le_bytes())?;
        w.write_all(&self.optimum.to_bits().to_le_bytes())?;
        maps(&mut w, &self.table)?;
        for table in [&self.out_m, &self.out_m2] {
            w.write_all(&(table.len() as u64).to_le_bytes())?;
            for row in table {
                f64s(&mut w, row)?;
            }
        }
        f64s(&mut w, &self.out_c)?;
        w.flush()?;
        drop(w);
        std::fs::rename(&tmp, file)
    }

    /// The pass at `lambda` kept in `file` by [`Self::write`].
    fn read(file: &std::path::Path, layout: NodeLayout, lambda: f64) -> std::io::Result<Self> {
        let bytes = std::fs::read(file)?;
        let mut r = Cursor {
            bytes: &bytes,
            at: 0,
        };
        if r.take(PASS_FORMAT.len())? != PASS_FORMAT || r.u64()? != lambda.to_bits() {
            return Err(malformed());
        }
        let optimum = f64::from_bits(r.u64()?);
        let table = r.maps()?;
        let out_m = r.table()?;
        let out_m2 = r.table()?;
        let out_c = r.f64s()?;
        if r.at != bytes.len() {
            return Err(malformed());
        }
        Ok(Self {
            layout,
            lambda,
            optimum,
            table,
            out_m,
            out_m2,
            out_c,
            aware: None,
            aware_read: false,
        })
    }
}

fn malformed() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, "a malformed pass")
}

/// A cached pass's bytes, read from the front.
struct Cursor<'b> {
    bytes: &'b [u8],
    at: usize,
}

impl<'b> Cursor<'b> {
    fn take(&mut self, n: usize) -> std::io::Result<&'b [u8]> {
        let out = self.bytes.get(self.at..self.at + n).ok_or_else(malformed)?;
        self.at += n;
        Ok(out)
    }

    fn u64(&mut self) -> std::io::Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().map_err(|_| malformed())?,
        ))
    }

    fn u32(&mut self) -> std::io::Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().map_err(|_| malformed())?,
        ))
    }

    fn f64s(&mut self) -> std::io::Result<Vec<f64>> {
        let n = self.u64()? as usize;
        (0..n).map(|_| Ok(f64::from_bits(self.u64()?))).collect()
    }

    fn table(&mut self) -> std::io::Result<Vec<Vec<f64>>> {
        let rows = self.u64()? as usize;
        (0..rows).map(|_| self.f64s()).collect()
    }

    fn maps(&mut self) -> std::io::Result<Vec<PairColumn>> {
        let n = self.u64()? as usize;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let len = self.u64()? as usize;
            let mut m = Vec::with_capacity(len);
            for _ in 0..len {
                let key = self.u64()?;
                m.push((key, [self.u32()?, self.u32()?]));
            }
            out.push(PairColumn::from_packed(m));
        }
        Ok(out)
    }
}

/// The share of the memory budget below which the grid keeps every pass it
/// has taken.
const RESIDENT_SHARE: f64 = 0.5;

/// The plain lattice's passes at the grid's weights, each taken the first
/// time a fold needs it — by that fold, or by an idle worker ahead of it.
///
/// Under a residency limit no worker takes a pass ahead of need, and once the
/// process holds more than the limit's memory share, taking one first drops
/// the least recently read of the others, so at most that many are held at
/// once (besides any a running fold still reads). Below that share every pass
/// is kept: a walk well under its memory budget would otherwise take most
/// passes again and again. The pass at weight 1, which every box reads, is
/// never dropped. A dropped pass is taken again when a fold next needs it, and
/// is the same pass.
struct Grid<'g> {
    weights: Vec<f64>,
    slots: Vec<std::sync::Mutex<Option<std::sync::Arc<Pass>>>>,
    /// Whether a worker has started each pass, so prefetching skips it.
    claimed: Vec<std::sync::atomic::AtomicBool>,
    /// When each pass was last read, on `clock`.
    read_at: Vec<std::sync::atomic::AtomicU64>,
    clock: std::sync::atomic::AtomicU64,
    /// The most passes held once the process holds more than the kB beside
    /// it.
    resident: Option<(usize, f64)>,
    costs: std::sync::Mutex<Vec<PassCost>>,
    params: &'g Params,
    graph: &'g Dfa,
    region: &'g PairConstraint,
    /// Threads inside each pass, and inside each probe that reads the grid.
    fold_threads: usize,
    /// The floors under `L` and `T` at weight 1, which every gap's energy
    /// cut reads.
    energy_floors: std::sync::OnceLock<Floors>,
    /// Where passes are kept across walks, and the lattice's fingerprint.
    cache: Option<(std::path::PathBuf, u64)>,
    /// The aware passes' key in the cache, beside the plain fingerprint:
    /// `None` without a cache or without aware passes.
    aware_print: Option<u64>,
    /// Where the low weights' passes also run over the constrained lattice;
    /// `None` takes plain passes alone.
    aware: Option<AwareSource<'g>>,
    /// The most plain passes taken in one traversal ([`Pass::lanes`]).
    lanes: usize,
}

impl<'g> Grid<'g> {
    fn new(
        weights: Vec<f64>,
        resident: Option<(usize, f64)>,
        (params, graph, region): (&'g Params, &'g Dfa, &'g PairConstraint),
        fold_threads: usize,
        aware: Option<AwareSource<'g>>,
    ) -> Self {
        let cache: Option<(std::path::PathBuf, u64)> =
            crate::dev_env::var_os(crate::dev_env::Setting::MotifPassCache)
                .map(|dir| (dir.into(), fingerprint(params, graph, region)));
        let aware_print = cache.as_ref().zip(aware).map(|((_, plain), src)| {
            let mut h = Fnv::new();
            h.eat(&plain.to_le_bytes());
            h.eat(&src.fingerprint().to_le_bytes());
            h.0
        });
        Self {
            slots: weights
                .iter()
                .map(|_| std::sync::Mutex::default())
                .collect(),
            claimed: weights
                .iter()
                .map(|_| std::sync::atomic::AtomicBool::new(false))
                .collect(),
            read_at: weights
                .iter()
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect(),
            clock: std::sync::atomic::AtomicU64::new(0),
            resident,
            costs: std::sync::Mutex::default(),
            weights,
            params,
            graph,
            region,
            fold_threads,
            energy_floors: std::sync::OnceLock::new(),
            cache,
            aware_print,
            aware,
            lanes: pass_lanes(),
        }
    }

    /// The pass at the `k`th weight, taken now if no worker has it, or
    /// waited for if one is taking it. `keep` is a pass the caller is about
    /// to read beside it, which is not dropped to make room.
    fn pass(&self, k: usize, keep: Option<usize>) -> std::sync::Arc<Pass> {
        use std::sync::atomic::Ordering::Relaxed;
        self.claimed[k].store(true, Relaxed);
        self.read_at[k].store(self.clock.fetch_add(1, Relaxed) + 1, Relaxed);
        let mut slot = self.slots[k].lock().expect("a pass's lock");
        if let Some(pass) = slot.as_ref() {
            return std::sync::Arc::clone(pass);
        }
        if let Some((limit, above_kb)) = self.resident {
            // A threshold of zero asks for room from the start and needs no
            // reading. Above zero the question is whether memory has passed
            // it, which an absent reading does not answer: where the platform
            // reports no resident size, taking the absence for pressure would
            // drop a pass on every take however large the budget. The room
            // left is read the same way, as nothing resident.
            if above_kb <= 0.0 || rss_kb().is_some_and(|kb| kb as f64 > above_kb) {
                self.make_room(limit, [Some(k), keep]);
            }
        }
        if let Some(pass) = self.take_together(k) {
            *slot = Some(std::sync::Arc::clone(&pass));
            return pass;
        }
        let started = std::time::Instant::now();
        let mut pass = self.plain(k);
        let secs = started.elapsed().as_secs_f64();
        (pass.aware, pass.aware_read) = self.aware_beside(k, &pass);
        let pass = std::sync::Arc::new(pass);
        self.costs
            .lock()
            .expect("the pass costs' lock")
            .push(PassCost::of(&pass, secs));
        *slot = Some(std::sync::Arc::clone(&pass));
        pass
    }

    /// The pass at the `k`th weight taken in one traversal with those at the
    /// nearest weights no worker holds or is taking and the cache does not
    /// hold, which are stored, and kept in the cache, as they are taken;
    /// `None` where there is no such weight, where the cache holds `k`'s, where
    /// passes are threaded or limited in number, or where `k` is weight 1,
    /// whose slot the other passes' aware halves read.
    fn take_together(&self, k: usize) -> Option<std::sync::Arc<Pass>> {
        let top = self.weights.len() - 1;
        if self.lanes < 2
            || k == top
            || self.fold_threads != 1
            || self.resident.is_some()
            || self.cached(k)
        {
            return None;
        }
        let mut near: Vec<usize> = (0..top).filter(|&j| j != k && !self.cached(j)).collect();
        near.sort_by_key(|&j| j.abs_diff(k));
        // A slot is locked while its pass is taken, so an unlocked empty one
        // is taken by no one; `try_lock` never waits, so no two workers can
        // each hold a slot the other waits on.
        let mut held = Vec::new();
        for j in near {
            if held.len() + 1 == self.lanes {
                break;
            }
            if let Ok(slot) = self.slots[j].try_lock() {
                if slot.is_none() {
                    self.claimed[j].store(true, std::sync::atomic::Ordering::Relaxed);
                    held.push((j, slot));
                }
            }
        }
        if held.is_empty() {
            return None;
        }
        let ks: Vec<usize> = std::iter::once(k)
            .chain(held.iter().map(|(j, _)| *j))
            .collect();
        let lambdas: Vec<f64> = ks.iter().map(|&j| self.weights[j]).collect();
        let started = std::time::Instant::now();
        let passes = Pass::lanes(self.params, self.graph, self.region, &lambdas);
        let secs = started.elapsed().as_secs_f64() / ks.len() as f64;
        let mut slots = held.into_iter().map(|(_, slot)| slot);
        let mut first = None;
        for (j, mut pass) in ks.into_iter().zip(passes) {
            if let Some(f) = self.cache_file(j) {
                // A pass that cannot be kept is taken again next time.
                let _ = pass.write(&f);
            }
            (pass.aware, pass.aware_read) = self.aware_beside(j, &pass);
            let pass = std::sync::Arc::new(pass);
            self.costs
                .lock()
                .expect("the pass costs' lock")
                .push(PassCost::of(&pass, secs));
            if j == k {
                first = Some(pass);
            } else {
                *slots.next().expect("a held slot") = Some(pass);
            }
        }
        first
    }

    /// The plain pass at the `k`th weight: read from the cache where it holds
    /// one, else taken, and kept there when there is a cache.
    fn plain(&self, k: usize) -> Pass {
        let file = self.cache_file(k);
        if let Some(pass) = file
            .as_deref()
            .and_then(|f| Pass::read(f, self.graph.layout(), self.weights[k]).ok())
        {
            return pass;
        }
        let pass = Pass::new(
            self.params,
            self.graph,
            self.region,
            self.weights[k],
            self.fold_threads,
        );
        if let Some(f) = &file {
            // A pass that cannot be kept is taken again next time.
            let _ = pass.write(f);
        }
        pass
    }

    /// Where the cache keeps the plain pass at the `k`th weight; `None`
    /// without a cache.
    fn cache_file(&self, k: usize) -> Option<std::path::PathBuf> {
        self.cache.as_ref().map(|(dir, print)| {
            dir.join(format!(
                "{print:016x}-{:016x}.pass",
                self.weights[k].to_bits()
            ))
        })
    }

    /// Whether the cache holds a plain pass at the `k`th weight.
    fn cached(&self, k: usize) -> bool {
        self.cache_file(k).is_some_and(|f| f.is_file())
    }

    /// The constrained-lattice pass beside the plain `pass` at the `k`th
    /// weight, where the grid takes one there, and whether it was read from
    /// the cache. Where there is a cache it is read from there when the
    /// cache holds it, and kept there when taken.
    fn aware_beside(&self, k: usize, pass: &Pass) -> (Option<AwarePass>, bool) {
        let Some(src) = self.aware.filter(|a| a.covers(pass.lambda)) else {
            return (None, false);
        };
        let file = self.aware_file(k);
        if let Some(aware) = file
            .as_ref()
            .and_then(|(f, key)| AwarePass::read(f, pass.layout, *key, pass.lambda).ok())
        {
            return (Some(aware), true);
        }
        // The pass at weight 1 is its own top, and its slot is the one held
        // while it is taken, so it is never taken again.
        let aware = if k == self.weights.len() - 1 {
            src.pass(self.params, self.region, (pass, pass), self.fold_threads)
        } else {
            let top = self.energy();
            src.pass(self.params, self.region, (pass, &top), self.fold_threads)
        };
        if let (Some(aware), Some((f, key))) = (&aware, &file) {
            // A pass that cannot be kept is taken again next time.
            let _ = aware.write(f, *key);
        }
        (aware, false)
    }

    /// Where the cache keeps the aware pass at the `k`th weight, and the key
    /// written into it: the plain lattice's and the constrained lattice's
    /// fingerprints, the weight, and the weight of the pass it is taken
    /// against. `None` without a cache or without aware passes.
    fn aware_file(&self, k: usize) -> Option<(std::path::PathBuf, u64)> {
        let (dir, _) = self.cache.as_ref()?;
        let print = self.aware_print?;
        let top = self.weights.last()?.to_bits();
        let weight = self.weights[k].to_bits();
        let mut h = Fnv::new();
        for part in [print, weight, top] {
            h.eat(&part.to_le_bytes());
        }
        let file = dir.join(format!("{print:016x}-{weight:016x}-{top:016x}.aware"));
        Some((file, h.0))
    }

    /// Drop the least recently read passes until fewer than `limit` are
    /// held, sparing the pass at weight 1 and those in `spare`. A pass
    /// another worker is taking is skipped rather than waited for.
    fn make_room(&self, limit: usize, spare: [Option<usize>; 2]) {
        use std::sync::atomic::Ordering::Relaxed;
        let top = self.weights.len() - 1;
        loop {
            let mut held = 0;
            let mut oldest: Option<(u64, usize)> = None;
            for (k, slot) in self.slots.iter().enumerate() {
                if spare.contains(&Some(k)) {
                    continue;
                }
                let Ok(slot) = slot.try_lock() else { continue };
                if slot.is_none() {
                    continue;
                }
                held += 1;
                let at = self.read_at[k].load(Relaxed);
                if k != top && oldest.is_none_or(|(t, _)| at < t) {
                    oldest = Some((at, k));
                }
            }
            let held = held + spare.iter().flatten().count();
            match oldest {
                Some((_, k)) if held >= limit => {
                    if let Ok(mut slot) = self.slots[k].try_lock() {
                        slot.take();
                    }
                }
                _ => return,
            }
        }
    }

    /// The passes taken so far, in the order they were stored.
    fn costs(&self) -> Vec<PassCost> {
        self.costs.lock().expect("the pass costs' lock").clone()
    }

    /// Take one pass no worker has started; `false` when none is left, or
    /// under a residency limit, where nothing is taken ahead of need.
    fn prefetch(&self) -> bool {
        if self.resident.is_some() {
            return false;
        }
        let Some(k) = (0..self.weights.len())
            .rev()
            .find(|&k| !self.claimed[k].swap(true, std::sync::atomic::Ordering::Relaxed))
        else {
            return false;
        };
        self.pass(k, None);
        true
    }

    /// The passes bracketing `lambda`: the one at or below it and, unless
    /// `lambda` is its own weight, the one above.
    fn bracket(&self, lambda: f64) -> (std::sync::Arc<Pass>, Option<std::sync::Arc<Pass>>) {
        let hi = self
            .weights
            .iter()
            .position(|&w| w >= lambda)
            .expect("the grid spans every weight");
        if self.weights[hi] == lambda {
            return (self.pass(hi, None), None);
        }
        let lo = self.pass(hi - 1, Some(hi));
        (lo, Some(self.pass(hi, Some(hi - 1))))
    }

    /// The pass at weight 1: the best completion energy the box reads.
    fn energy(&self) -> std::sync::Arc<Pass> {
        self.pass(self.weights.len() - 1, None)
    }

    /// The floors under `L` and `T` at weight 1 over `lattice`, built the
    /// first time a gap reads them.
    fn energy_floors(&self, lattice: &Lattice) -> &Floors {
        self.energy_floors.get_or_init(|| {
            let top = self.energy();
            let cut = Cut::between(&top, None, 1.0, f64::INFINITY);
            Floors::new(self.params, lattice, self.region, &cut)
        })
    }
}

/// The tie-weight cut through a gap's box: no design scoring above `ceiling`
/// at `lambda` can lie in the box, and the chord of the two passes bracketing
/// `lambda` floors what surrounds each cell.
///
/// Every motif-free completion is also a completion on the plain lattice, so
/// the plain best completion is never worse than the motif-free one: a point
/// whose score plus that floor exceeds the ceiling is on no design in the box.
#[derive(Clone, Copy)]
struct Cut<'a> {
    lambda: f64,
    ceiling: f64,
    near: &'a Pass,
    /// The other bracketing pass and the weight on `near`, unless `lambda`
    /// is `near`'s own.
    far: Option<(&'a Pass, f64)>,
    /// The chord of the bracketing passes' constrained-lattice floors, where
    /// both sides have them; the probe and gap folds prune on it besides.
    aware: Option<AwareChord<'a>>,
}

impl<'a> Cut<'a> {
    /// The cut at `lambda` from the passes bracketing it: `lo` at or below
    /// it and, unless `lambda` is `lo`'s own weight, `hi` above it.
    fn between(lo: &'a Pass, hi: Option<&'a Pass>, lambda: f64, ceiling: f64) -> Self {
        Self {
            lambda,
            ceiling,
            near: lo,
            far: hi.map(|hi| (hi, (hi.lambda - lambda) / (hi.lambda - lo.lambda))),
            aware: None,
        }
    }

    /// This cut with the chord of its passes' constrained-lattice floors,
    /// where both sides have one: an aware pass, or at weight 0 the chain
    /// tables, exact there. At least one side must be a pass, so a grid
    /// without aware passes cuts as before.
    fn with_aware(self, chain: Option<&'a ChainCai>) -> Self {
        let side = |p: &'a Pass| match &p.aware {
            Some(a) => Some(AwareSide::Pass(a)),
            None if p.lambda == 0.0 => chain.map(AwareSide::Zero),
            None => None,
        };
        let has_pass =
            self.near.aware.is_some() || self.far.is_some_and(|(far, _)| far.aware.is_some());
        let aware = has_pass
            .then(|| {
                let near = side(self.near)?;
                let far = match self.far {
                    None => None,
                    Some((far, w)) => Some((side(far)?, w)),
                };
                Some(AwareChord { near, far })
            })
            .flatten();
        Self { aware, ..self }
    }

    /// The chord of a value both passes read off the same key; `None` where
    /// either lacks it.
    fn chord(&self, read: impl Fn(&Pass) -> Option<f64>) -> Option<f64> {
        let near = read(self.near)?;
        match self.far {
            None => Some(near),
            Some((far, w)) => Some(w * near + (1.0 - w) * read(far)?),
        }
    }

    /// A floor on the plain optimum at `lambda`, and so on every design's
    /// score there: the chord of the bracketing passes' optima.
    fn optimum(&self) -> f64 {
        self.chord(|p| Some(p.optimum)).unwrap_or(f64::INFINITY)
    }

    /// The `constrained` fold's view of the same bound, under `ceiling`,
    /// with the further floors `zero`.
    fn bound<'z>(
        &'z self,
        ceiling: f64,
        zero: &'z [ZeroChord<'z>],
    ) -> crate::constrained::OutsideBound<'z> {
        crate::constrained::OutsideBound {
            outside: &self.near.table,
            chord: self.far.map(|(far, w)| (&far.table[..], w)),
            through: true,
            aware: self.aware,
            ceiling,
            zero,
            motif_free: None,
        }
    }

    /// Whether the best design through the pair on the plain lattice already
    /// scores above the ceiling, so no motif-free one through it can do
    /// better. `false` where the plain passes do not cover the pair.
    fn rules_out_pair(&self, o: usize, key: &PairKey) -> bool {
        self.chord(|p| {
            let through = p.table.get(o)?.get(key)?.through;
            (through > f64::NEG_INFINITY).then_some(through)
        })
        .is_some_and(|floor| floor > self.ceiling + CUT_SLACK)
    }

    /// The floor on the completion of a multi-branch cell from `inner` to
    /// `outer`, by the plain table `table` picks.
    fn span_outside(
        &self,
        table: impl Fn(&Pass) -> &Vec<Vec<f64>>,
        inner: Node,
        outer: Node,
    ) -> Option<f64> {
        self.chord(|p| {
            table(p)
                .get(p.layout.ix(outer))?
                .get(p.layout.ix(inner))
                .copied()
        })
    }

    /// The floor on the completion of the exterior up to `node`.
    fn exterior_outside(&self, node: Node) -> Option<f64> {
        self.chord(|p| p.out_c.get(p.layout.ix(node)).copied())
            .filter(|o| o.is_finite())
    }

    /// The floor on the completion of the pair `key` closing into column `o`.
    fn outside(&self, o: usize, key: &PairKey) -> Option<f64> {
        self.chord(|p| Some(p.table.get(o)?.get(key)?.outside))
    }

    /// Drop the points of a multi-branch cell from `inner` to `outer` that no
    /// completion brings under the ceiling, by the plain table `table` picks.
    /// A cell the plain lattice cannot complete at all reads `+∞` there and
    /// loses every point: a motif-free completion would be a plain one.
    fn retain_span(
        &self,
        front: &mut Front<Prov>,
        table: impl Fn(&Pass) -> &Vec<Vec<f64>>,
        inner: Node,
        outer: Node,
    ) {
        let Some(outside) = self.span_outside(table, inner, outer) else {
            return;
        };
        let (lambda, ceiling) = (self.lambda, self.ceiling + CUT_SLACK);
        front.retain(|p| brd(p.mfe, p.cai, lambda) + outside <= ceiling);
    }

    /// Drop the points of the pair's front that no completion brings under
    /// the ceiling.
    fn retain(&self, front: &mut Front<Prov>, o: usize, key: &PairKey) {
        let Some(outside) = self.outside(o, key) else {
            return;
        };
        let (lambda, ceiling) = (self.lambda, self.ceiling + CUT_SLACK);
        front.retain(|p| brd(p.mfe, p.cai, lambda) + outside <= ceiling);
    }
}

/// Slack in the box's comparisons, for the same reason as [`CUT_SLACK`].
const BOX_SLACK: f64 = 1e-6;

/// A gap's adaptation floor: designs below `c_min` are outside its box. Its
/// energy side is the fold's energy cut.
#[derive(Clone, Copy)]
struct GapBox {
    c_min: f64,
}

/// A vertex of the hull of the motif-excluded front.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotifVertexPoint {
    /// Energy under the constraint, dekacalories.
    pub mfe: Energy,
    /// Accumulated coding log-adaptation, as the front's `cai_log` reports it.
    pub cai: f64,
}

/// One gap between adjacent hull vertices, solved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotifGap {
    /// The tie weight the gap was cut at.
    pub lambda: f64,
    /// Front points the gap holds, its two vertices included.
    pub points: usize,
    /// Front folds it took: one, or one per energy slice and one per fold
    /// abandoned over budget.
    pub folds: usize,
    /// Seconds for the plain scan and outside pass behind the cut.
    pub cut_s: f64,
    /// Seconds for the front fold.
    pub fold_s: f64,
    /// Seconds from the walk's start to the gap's settling.
    pub settled_s: f64,
    /// Closed pairs the constrained-lattice chords alone ruled out over the
    /// gap's folds, and points their limits alone dropped.
    pub aware_out: usize,
    pub aware_dropped: usize,
    /// Closed pairs under a chord that it could not read, a node holding more
    /// than one automaton state: 0 on a merged product, by construction.
    pub aware_unkeyed: usize,
    /// Ranges whose budget was raised to the growth times what the gap's
    /// last solved range stored ([`density_carry`]).
    pub carried: usize,
    /// Ranges taken in part, cut to [`slice_step`] times the width of the
    /// gap's last solved range once a range of it had given up.
    pub sized: usize,
    /// Folds that started from the complete spans of a fold that gave up,
    /// and the points they took from it ([`gap_reuse`]).
    pub adopted: usize,
    pub adopted_points: usize,
}

/// One plain-lattice pass the walk's cuts read: its weight, the seconds it
/// took, and the process's resident memory once it was stored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassCost {
    /// The pass's tie weight.
    pub lambda: f64,
    /// Seconds for its plain scan and outside pass.
    pub secs: f64,
    /// Resident memory once it was stored, in KiB; `None` where unread.
    pub rss_kb: Option<u64>,
    /// Seconds for its pass over the constrained lattice, and that pass's
    /// bounded folds, its ceiling's margin above the constrained optimum and
    /// the cells and pairs it holds; all 0 where it took none.
    pub aware_secs: f64,
    pub aware_folds: usize,
    pub aware_margin: f64,
    pub aware_cells: usize,
    pub aware_pairs: usize,
    /// Whether its constrained-lattice pass was read from the pass cache of
    /// a development build rather than taken; `aware_secs` is then
    /// the read's.
    pub aware_read: bool,
}

impl PassCost {
    /// The cost of `pass`, whose plain half took `secs`, with the process's
    /// resident memory now that it is stored.
    fn of(pass: &Pass, secs: f64) -> Self {
        let aware = pass.aware.as_ref();
        let (aware_cells, aware_pairs) = aware.map_or((0, 0), AwarePass::entries);
        Self {
            lambda: pass.lambda,
            secs,
            rss_kb: rss_kb(),
            aware_secs: aware.map_or(0.0, |a| a.secs),
            aware_folds: aware.map_or(0, |a| a.folds),
            aware_margin: aware.map_or(0.0, |a| a.ceiling - a.optimum),
            aware_cells,
            aware_pairs,
            aware_read: pass.aware_read,
        }
    }
}

/// One probe: its weight, seconds and folds, and what its bound's floors
/// ruled out over its folds — of those, what the constrained-lattice floors
/// alone did ([`AwareChord`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeRecord {
    pub lambda: f64,
    pub secs: f64,
    pub folds: usize,
    /// Closed pairs ruled out once scored, and closures skipped before.
    pub pruned: usize,
    pub cut: usize,
    /// Of those, the ones only the constrained-lattice floors ruled out.
    pub aware_pruned: usize,
    pub aware_cut: usize,
}

/// The motif-excluded front by a hull walk.
#[derive(Debug, Clone, PartialEq)]
pub struct MotifWalk {
    /// Every Pareto design from the most stable to the most adapted.
    pub front: Vec<FrontPoint>,
    /// The hull's vertices, most stable first.
    pub vertices: Vec<MotifVertexPoint>,
    /// Scalar folds the hull sweep ran.
    pub hull_folds: usize,
    /// Seconds from the walk's start until its last gap settled, 0 when none
    /// settled; a plain walk reports its sweep's seconds. The walk's phases
    /// overlap this span, so it is not the cost of any one of them.
    pub hull_s: f64,
    /// Seconds the two end folds took, side by side on two threads.
    pub ends_s: f64,
    /// Seconds in the probe folds, summed over threads.
    pub probe_s: f64,
    /// Probes run, the folds they took, and how many of them a ceiling below
    /// their pair's own score certified.
    pub probes: usize,
    pub probe_folds: usize,
    pub probes_under_chord: usize,
    /// Each probe, in the order they settled.
    pub probe_records: Vec<ProbeRecord>,
    /// Hull vertices the aware passes seeded ([`MotifRun::aware_seeds`]).
    pub seeded: usize,
    /// The process's peak resident memory once the two ends were folded, in
    /// KiB; 0 for a resumed walk or where unread.
    pub ends_peak_kb: Option<u64>,
    /// Whether the adapted end came from the fold over the most adapted
    /// codons alone; `false` when the whole lattice folded it, or the walk
    /// was resumed.
    pub adapted_narrow: bool,
    /// Whether `vertices` are the lower hull of `front` rather than the
    /// vertices the walk's probes found: a narrowed walk's boxes are solved
    /// without probes, so a walk resumed from this one takes its vertices
    /// the same way ([`MotifResume::rehull`]).
    pub rehull: bool,
    /// The passes taken, in the order they were stored.
    pub passes: Vec<PassCost>,
    /// The gaps solved, in the vertices' order.
    pub gaps: Vec<MotifGap>,
    /// What stopped the walk short — `"time"` or `"memory"` — or `None` for a
    /// complete front.
    pub stopped_by: Option<&'static str>,
    /// The boxes between neighbouring hull points that the budget left
    /// unsolved, most stable first. Every point of `front` is on the exact
    /// front regardless: a gap's points lie between its two vertices in both
    /// objectives, so nothing in another gap can dominate them. A box with a
    /// rest in `rests` is solved part way: its front points up to the rest's
    /// `from` are in `front`, and only those above it are missing.
    pub open: Vec<(MotifVertexPoint, MotifVertexPoint)>,
    /// The open boxes whose gap was partly solved, one per such box.
    pub rests: Vec<MotifRest>,
    /// Gap folds that moved on under the per-gap time cap
    /// ([`MotifRun::gap_secs`]), each queued again behind the walk's other
    /// tasks.
    pub gaps_moved_on: usize,
    /// The open boxes whose gap moved on under the per-gap time cap, as
    /// their two ends' energies, the more stable first, with how many times
    /// it did, counting the walks this one resumed. A box not listed moved on
    /// none.
    pub moves: Vec<(Energy, Energy, u32)>,
}

/// An open box whose gap fold stopped part way: its probe found no vertex,
/// and every front point up to `from` is among the walk's points. Resuming
/// solves only the energies above `from`, for designs more adapted than
/// `floor`, the best adaptation found at or below it.
///
/// What is missing from the box spans `hi` less the larger of `from` and
/// `lo`, not `hi` less `lo`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotifRest {
    /// The box's two ends' energies, the more stable first.
    pub lo: Energy,
    pub hi: Energy,
    /// The energy up to which the gap is solved: one unit below `lo` when
    /// the box's probe ran and its gap was stopped before a range was
    /// solved, so that a resume folds the whole gap without the probe.
    pub from: Energy,
    /// The best adaptation at or below `from`, as the front's `cai_log`
    /// reports it.
    pub floor: f64,
}

/// What an earlier walk of the same molecule, flanks, constraint and motifs
/// returned, to resume from: its points, which of them are hull vertices, and
/// the boxes it left open.
///
/// Every point of a walk stopped short is on the exact front, and every point
/// it is missing lies in an open box, so resuming settles the open boxes alone
/// and returns their points beside the earlier ones.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MotifResume {
    /// The earlier walk's points, each with whether it is a hull vertex.
    pub points: Vec<(FrontPoint, bool)>,
    /// Its open boxes as their two ends `(mfe kcal/mol, cai_log)`, the more
    /// stable first; each is among `points`, and a hull vertex unless
    /// `rehull`. A box with a rest in `rests` also holds points of `points` strictly inside it:
    /// every front point up to the rest's energy.
    pub open: Vec<((f64, f64), (f64, f64))>,
    /// Its partly solved boxes as `(mfe_lo, mfe_hi)` kcal/mol, each the
    /// energies of an open box's ends, with the energy up to which the gap is
    /// solved and the best `cai_log` there ([`MotifRest`]).
    pub rests: Vec<((f64, f64), (f64, f64))>,
    /// Whether the walk takes its hull vertices from its final front rather
    /// than from `points`' marks and its own probes: set where the boxes'
    /// ends need not be hull vertices ([`narrowed_resume`]).
    pub rehull: bool,
    /// Its boxes whose gap moved on under the per-gap time cap, as
    /// `(mfe_lo, mfe_hi)` kcal/mol with how many times ([`MotifWalk::moves`]);
    /// the resumed walk's first visit to such a gap is the one after these.
    pub moves: Vec<((f64, f64), u32)>,
}

/// The walk of `spec`'s front from `wider`, a front of the same molecule,
/// flanks, constraint and pins under a subset of `spec`'s motifs: every point
/// of `wider` that spells none of `spec`'s motifs, with a box left open
/// between the two kept neighbours of each run of lost points, and around
/// each box `wider` left open. `None` where `wider` lost an end, which only
/// a walk from the ends finds again.
///
/// Every design `spec` admits, `wider`'s motifs admit too, so each point q of
/// `spec`'s front is weakly dominated by a point p of `wider`'s. Where p is
/// kept, q is p's point. Where p is lost, q is at least as stable as p's next
/// kept neighbour and less stable than its previous one, since neither
/// dominates q, so q lies in p's box. Where p is missing from `wider`, it lies
/// in one of `wider`'s open boxes, widened here to kept ends.
///
/// A box between kept points that are not hull vertices is solved as a rest
/// from immediately below its stable end, without a probe: the gap fold's cuts hold
/// for any two consecutive front points, while a probe between them may find
/// a vertex outside the box. A box `wider` left open between two kept hull
/// vertices, with no point lost inside it, keeps its probe and its rest. The walk's hull vertices are then taken from
/// its front ([`MotifResume::rehull`]): a kept point can become a vertex, and
/// a box solved without a probe marks none.
///
/// # Errors
///
/// Where `spec`'s motifs or window do not build an automaton.
pub fn narrowed_resume<S: AsRef<str>>(
    spec: &MotifFrontSpec<'_, S>,
    wider: &MotifResume,
) -> Result<Option<MotifResume>, BeamError> {
    let (ac, scope) = Ac::for_windows(spec.motifs)?;
    let scoped = crate::ac::ScopedAc::new(&ac, scope);
    let admitted = |p: &FrontPoint| {
        let nucs: Option<Vec<Nucleotide>> = p.sequence.bytes().map(Nucleotide::from_byte).collect();
        nucs.is_some_and(|nucs| scoped.is_clean(&nucs))
    };
    let mut points: Vec<(&FrontPoint, bool, bool)> = wider
        .points
        .iter()
        .map(|(p, on_hull)| (p, *on_hull, admitted(p)))
        .collect();
    points.sort_by(|a, b| a.0.mfe_kcal_mol.total_cmp(&b.0.mfe_kcal_mol));
    let (Some(first), Some(last)) = (points.first(), points.last()) else {
        return Ok(None);
    };
    if !first.2 || !last.2 {
        return Ok(None);
    }
    let at = |p: &FrontPoint| (p.mfe_kcal_mol, p.cai_log);
    // The kept neighbours of an energy: the last kept point below it and the
    // first above it. Both ends are kept, so both exist strictly inside.
    let kept: Vec<&FrontPoint> = points.iter().filter(|p| p.2).map(|p| p.0).collect();
    let below = |e: f64| {
        *kept
            .iter()
            .rev()
            .find(|p| p.mfe_kcal_mol < e)
            .unwrap_or(&kept[0])
    };
    let above = |e: f64| {
        *kept
            .iter()
            .find(|p| p.mfe_kcal_mol > e)
            .unwrap_or(&kept[kept.len() - 1])
    };
    let is_kept = |e: f64| kept.iter().any(|p| (p.mfe_kcal_mol - e).abs() < 1e-9);
    let is_vertex = |e: f64| {
        points
            .iter()
            .any(|p| p.1 && p.2 && (p.0.mfe_kcal_mol - e).abs() < 1e-9)
    };
    // Each box as its ends and whether it keeps its probe and rest.
    let mut boxes: Vec<NarrowBox> = Vec::new();
    for w in points.windows(2) {
        if w[0].2 && !w[1].2 {
            let lo = w[0].0;
            boxes.push((at(lo), at(above(lo.mfe_kcal_mol)), false));
        }
    }
    for &((a, ac), (b, bc)) in &wider.open {
        let (a, ac, b, bc) = if a <= b {
            (a, ac, b, bc)
        } else {
            (b, bc, a, ac)
        };
        if is_vertex(a) && is_vertex(b) {
            boxes.push(((a, ac), (b, bc), true));
        } else {
            let lo = if is_kept(a) { (a, ac) } else { at(below(a)) };
            let hi = if is_kept(b) { (b, bc) } else { at(above(b)) };
            boxes.push((lo, hi, false));
        }
    }
    boxes.sort_by(|x, y| x.0 .0.total_cmp(&y.0 .0));
    let mut merged: Vec<NarrowBox> = Vec::new();
    for b in boxes {
        match merged.last_mut() {
            Some(m) if b.0 .0 < m.1 .0 => {
                if b.1 .0 > m.1 .0 {
                    m.1 = b.1;
                }
                m.2 = false;
            }
            _ => merged.push(b),
        }
    }
    // A box keeping its probe keeps the points its rest solved; any other is
    // folded whole, which finds its points again.
    let inside = |e: f64| {
        merged
            .iter()
            .any(|(lo, hi, probed)| !probed && e > lo.0 && e < hi.0)
    };
    let points = points
        .iter()
        .filter(|p| p.2 && !inside(p.0.mfe_kcal_mol))
        .map(|p| (p.0.clone(), p.1))
        .collect();
    let wider_rest = |lo: f64, hi: f64| {
        wider
            .rests
            .iter()
            .find(|((a, b), _)| (a - lo).abs() < 1e-9 && (b - hi).abs() < 1e-9)
            .copied()
    };
    let rests = merged
        .iter()
        .filter_map(|&(lo, hi, probed)| {
            if probed {
                wider_rest(lo.0, hi.0)
            } else {
                Some(((lo.0, hi.0), (lo.0 - 0.01, lo.1)))
            }
        })
        .collect();
    Ok(Some(MotifResume {
        points,
        open: merged.iter().map(|&(lo, hi, _)| (lo, hi)).collect(),
        rests,
        rehull: true,
        moves: Vec::new(),
    }))
}

/// The weights the walk's two ends are folded at, as `derna front` fixes them.
const END_EPSILON: f64 = 1.0 / (1u32 << 23) as f64;

/// The exact front of (constrained MFE, coding CAI) over motif-free designs by
/// a walk: the hull's vertices from scalar folds of the product lattice, then
/// each gap between adjacent vertices from the front fold under that gap's
/// tie-weight cut. The same points as [`motif_front_exact`], in a fraction of
/// the work for long molecules; `threads` gaps are solved side by side.
///
/// # Errors
///
/// As [`motif_front_exact`].
pub fn motif_front_walk<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    spec: &MotifFrontSpec<'_, S>,
    threads: usize,
) -> Result<MotifWalk, BeamError> {
    motif_front_walk_within(params, protein, usage, spec, threads, WalkBudget::default())
}

/// [`motif_front_walk`] stopped short when `budget` runs out: the points
/// returned are exactly the front's points in the gaps it solved, and in
/// each gap it solved part way up to that gap's rest ([`MotifWalk::rests`]),
/// with every hull vertex it found, and [`MotifWalk::open`] lists the gaps it
/// did not finish.
/// The ends always complete, and a probe fold once started runs to its end.
///
/// # Errors
///
/// As [`motif_front_exact`].
pub fn motif_front_walk_within<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    spec: &MotifFrontSpec<'_, S>,
    threads: usize,
    budget: WalkBudget,
) -> Result<MotifWalk, BeamError> {
    let run = MotifRun {
        budget,
        ..MotifRun::default()
    };
    motif_front_walk_run(params, protein, usage, spec, threads, run)
}

/// How a walk runs beyond its inputs: its budget, an earlier walk to resume,
/// and what to hand the front found so far.
#[derive(Clone, Copy, Default)]
pub struct MotifRun<'a> {
    /// What may stop the walk short.
    pub budget: WalkBudget,
    /// An earlier walk of the same molecule, flanks, constraint and motifs:
    /// only the boxes it left open are settled, and the front returned is its
    /// points with theirs. Its vertices are taken as given, without a fold.
    pub resume: Option<&'a MotifResume>,
    /// Called once the two ends are folded (or taken from the walk resumed),
    /// before any pair settles, and after each pair of hull points settles,
    /// with the walk so far: every
    /// point it holds is a front point, and every one it is missing lies in
    /// one of its `open` boxes, so it can be resumed from. Calls are made one
    /// at a time, each with a later state than the one before.
    pub checkpoint: Option<&'a (dyn Fn(&MotifWalk) + Sync)>,
    /// The most points a gap's fold may store before its energy range is
    /// split; `None` takes the default. A small one drives the slicing on
    /// molecules too short to reach the default.
    pub gap_budget: Option<usize>,
    /// What the rest of a gap may store, folded again as one range, as a
    /// multiple of what a slice too narrow to split stored past the gap
    /// budget; `None` takes the default, `Some(0)` keeps the slices.
    pub slice_grow: Option<usize>,
    /// The width a gap's first ranges are cut to under a memory budget;
    /// `None` takes the last cut gap fold that solved, once one has. A small
    /// one drives the cutting on molecules too short to need it.
    pub gap_fit: Option<i32>,
    /// Whether a gap's slices are also cut by each pass bracketing its weight
    /// on its own; `None` takes the default ([`default_bracket_cuts`]).
    pub bracket_cuts: Option<bool>,
    /// The share of [`WalkBudget::max_rss_gb`] past which the walk holds
    /// only the passes its workers read; `None` takes the default. A small
    /// one drives the dropping on molecules too short to reach the default.
    pub resident_share: Option<f64>,
    /// The most grid passes held at once without a memory budget, the pass
    /// at weight 1 among them; `None` holds every pass taken. A dropped pass
    /// is taken again when read, or, in a development build with a pass cache, read
    /// back from it, which makes holding few cheap.
    pub resident_passes: Option<usize>,
    /// Stop once the boxes not yet settled span at most this share of the
    /// energy between the two ends; `None` runs until every box settles or
    /// the budget is spent. A coverage stop leaves walks of sibling cells
    /// stopped at the same resolution, where a wall budget does not.
    pub stop_open_share: Option<f64>,
    /// Stop once this many gap folds have solved their range, between one
    /// gap fold and the next, keeping each stopped gap's rest. A fold that
    /// gives up is not counted, so every round that folds solves a range.
    /// Unlike the wall budget it stops every run at the same place.
    pub gap_folds: Option<usize>,
    /// Whether a probe whose doubled first margin reaches its pair's score
    /// folds at that score at once; `None` takes [`default_probe_skip`].
    /// The front is the same either way.
    pub probe_skip: Option<bool>,
    /// Whether a probe's first ceiling starts at the widest slack an earlier
    /// probe of the walk found between its floor and its optimum, and a probe
    /// whose fold found a design above its ceiling folds next halfway to that
    /// design; `None` takes [`default_probe_learn`]. The front is the same
    /// either way.
    pub probe_learn: Option<bool>,
    /// Threads inside each end, pass, probe and gap fold, beside the walk's
    /// own workers; `0` and `1` fold each on one thread. Every such fold reaches
    /// the same answer at any count, so the front does not depend on it. It
    /// spends spare cores where memory allows fewer workers than cores.
    pub fold_threads: usize,
    /// The highest grid weight whose pass also runs over the constrained
    /// lattice, for the probes and gap folds between such weights to prune
    /// on; `None` takes the default ([`AWARE_UPTO`], or
    /// `DERNA_MOTIF_AWARE_UPTO`), and 0 takes none. The front does not depend
    /// on it.
    pub aware_upto: Option<f64>,
    /// Seed the sweep with the constrained optimum at each grid weight
    /// whose pass runs over the constrained lattice: each is a hull vertex
    /// its aware pass already certified, so the pairs between them are
    /// probed from the start in place of the pair of ends. The front does
    /// not depend on it.
    pub aware_seeds: bool,
    /// Fold a gap first in even energy ranges at most this wide, in the
    /// fold's integer units, lowest first, each under its own ceiling and the
    /// floor found below it, in place of the whole gap; a range that outgrows
    /// the gap budget is still halved, and the rest still regrown. `None`
    /// folds the whole gap first. Any width returns the same points.
    pub slice_width: Option<i32>,
    /// A complete or stopped front of the same molecule, flanks, constraint
    /// and pins under a subset of these motifs, to narrow instead of walking
    /// from the ends ([`narrowed_resume`]). Ignored with `resume`.
    pub narrow_from: Option<&'a MotifResume>,
    /// The seconds a gap's fold may take in one task before the walk keeps
    /// its rest and moves on to its other boxes, taking the gap up again once
    /// they are done; a resumed box with a rest also waits behind every box
    /// without one. `None` takes [`gap_secs`], 300. The cap doubles each time
    /// the gap moves on, and stops a fold in flight, not only between slices
    /// ([`solve_gap`]).
    /// The front does not depend on it.
    pub gap_secs: Option<f64>,
}

/// [`motif_front_walk_within`] under a [`MotifRun`]: resumed, checkpointed,
/// or both.
///
/// # Errors
///
/// As [`motif_front_exact`].
pub fn motif_front_walk_run<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    spec: &MotifFrontSpec<'_, S>,
    threads: usize,
    run: MotifRun<'_>,
) -> Result<MotifWalk, BeamError> {
    let MotifRun {
        budget,
        resume,
        checkpoint,
        gap_budget,
        slice_grow,
        gap_fit,
        bracket_cuts,
        resident_share,
        resident_passes,
        stop_open_share,
        gap_folds,
        probe_skip,
        probe_learn,
        fold_threads,
        aware_upto: upto,
        aware_seeds,
        slice_width,
        narrow_from,
        gap_secs: gap_cap,
    } = run;
    let narrowed = match (resume, narrow_from) {
        (None, Some(wider)) => narrowed_resume(spec, wider)?,
        _ => None,
    };
    let resume = resume.or(narrowed.as_ref());
    let probe_rule = ProbeRule {
        skip: probe_skip.unwrap_or_else(default_probe_skip),
        learn: probe_learn.unwrap_or_else(default_probe_learn),
        slack: std::sync::atomic::AtomicU64::new(0),
    };
    let fold_threads = fold_threads.max(1);
    let slice_budget = SliceBudget {
        points: gap_budget.unwrap_or_else(default_gap_budget),
        grow: slice_grow.unwrap_or_else(default_slice_grow),
        max_rss_kb: budget.max_rss_gb.map(|gb| gb * 1024.0 * 1024.0),
        workers: threads.max(1),
        open_in_room: gap_room(),
        fit: gap_fit,
        bracket: bracket_cuts.unwrap_or_else(default_bracket_cuts),
        width: slice_width,
        reuse: gap_reuse(),
    };
    if !spec.constraints.is_hard() {
        return Err(BeamError::SoftRegionUnsupported);
    }
    let plain = || {
        let run = PlainRun {
            budget,
            resume,
            checkpoint,
        };
        plain_walk(params, protein, usage, spec, threads * fold_threads, run)
    };
    if spec.motifs.is_empty() && spec.pins.is_none() {
        return plain();
    }
    let aa = crate::residues(protein)?;
    let five = Flank::parse(spec.utr5)?;
    let three = Flank::parse(spec.utr3)?;
    let graph = spec_graph(&aa, usage, spec, &five, &three)?;
    let region = PairConstraint::new(&spec.constraints, &graph);
    let (ac, scope) = Ac::for_windows(spec.motifs)?;
    let product = ProductDfa::build_scoped(&graph, &ac, scope);
    if !product.is_feasible() {
        return Err(BeamError::Infeasible);
    }
    // A motif set that removes no edge of this protein's lattice leaves the
    // plain lattice, whose scan takes reductions the product's fold cannot
    // (see `constrained`): its front is the plain walk's.
    if spec.pins.is_none() && !product.dropped_edge() {
        return plain();
    }
    let most_adapted = product.most_adapted_codons();
    let product = product.merge_equivalent();
    let lattice = Lattice::new(&product);
    let chain = ChainCai::build(&product);
    let open_chain = crate::constrained::open_chain_cai(&product);

    let started = std::time::Instant::now();
    // Under a memory budget the walk folds the ends one after the other, and
    // past [`RESIDENT_SHARE`] of the budget holds per worker only the two
    // passes around its weight, beside the pass at weight 1. Without one,
    // `resident_passes` holds that few from the start.
    let capped = budget.max_rss_gb.is_some();
    let upto = upto.unwrap_or_else(aware_upto);
    let grid = Grid::new(
        motif_pass_grid(derna_codon_dp::cai::sense_codons(&aa)).weights(),
        budget
            .max_rss_gb
            .map(|gb| {
                (
                    2 * threads.max(1) + 1,
                    resident_share.unwrap_or(RESIDENT_SHARE) * gb * 1024.0 * 1024.0,
                )
            })
            .or_else(|| resident_passes.map(|k| (k.max(2 * threads.max(1) + 1), 0.0))),
        (params, &graph, &region),
        fold_threads,
        (upto > 0.0).then_some(AwareSource {
            product: &product,
            chain: open_chain.map(|c| (&chain, c)),
            upto,
            margin: aware_margin(),
            headroom: aware_headroom(),
            seed: aware_seeds,
        }),
    );
    // The grid weights whose aware passes seed the sweep.
    let seed_weights: Vec<f64> = if aware_seeds && upto > 0.0 {
        grid.weights
            .iter()
            .copied()
            .filter(|&w| w > 0.0 && w <= upto + 1e-12)
            .collect()
    } else {
        Vec::new()
    };
    // An earlier walk's vertices in the walk's own units. Its points are
    // returned as it wrote them, so a resumed front repeats them exactly.
    let offset = graph.codon_offset();
    let unit = |(mfe, cai_log): (f64, f64)| MotifVertexPoint {
        mfe: Energy((mfe * 100.0).round() as i32),
        cai: cai_log - offset,
    };
    let mut prior_vertices: Vec<MotifVertexPoint> = resume
        .iter()
        .flat_map(|r| &r.points)
        .filter(|(_, on_hull)| *on_hull)
        .map(|(p, _)| unit((p.mfe_kcal_mol, p.cai_log)))
        .collect();
    prior_vertices.sort_by_key(|v| v.mfe.0);
    let vertex_at = |at: (f64, f64)| {
        let v = unit(at);
        prior_vertices
            .iter()
            .copied()
            .find(|p| p.mfe == v.mfe)
            .unwrap_or(v)
    };
    let resumed = !prior_vertices.is_empty();
    let kcal = |e: f64| (e * 100.0).round() as i32;
    let kernel = ProductWalk {
        params,
        graph: &graph,
        product: &product,
        lattice: &lattice,
        region: &region,
        grid,
        chain: open_chain.map(|c| (&chain, c)),
        most_adapted: (most_adapted.as_ref(), (&aa, usage, &five, &three)),
        motifs: (&spec.constraints, &ac, scope),
        designs: Designs::default(),
        spans: std::sync::Mutex::new(Vec::new()),
        slice_budget,
        probe_rule,
        fold_threads,
        threads,
        capped,
        resume,
        offset,
        started,
        narrowed: std::sync::atomic::AtomicBool::new(false),
        seeded: AtomicUsize::new(0),
        ends: std::sync::Mutex::new((0, 0.0, None)),
        // A probe, once started, runs to its end, so one is not started when
        // the time left is shorter than the longest taken so far: the
        // deadline would pass while it ran. Before any probe has run, the
        // stable end's fold stands in for one: both are a bounded fold over
        // the whole lattice.
        longest_probe: std::sync::Mutex::new(0.0),
    };
    // Pairs of hull points still to settle; each either yields a vertex
    // between them or is a gap, solved under the cut its probe already paid
    // for. A lone vertex is the gap from itself to itself. A resumed walk
    // settles the boxes the earlier one left open, and nothing else.
    let start = match resume {
        Some(r) if resumed => walk::Start {
            vertices: prior_vertices.clone(),
            ends: [
                prior_vertices.first().copied(),
                prior_vertices.last().copied(),
            ],
            pairs: r
                .open
                .iter()
                .map(|&(a, b)| (vertex_at(a), vertex_at(b)))
                .collect(),
            tasks: Vec::new(),
            seeds: None,
            // Each partly solved box by its ends' energies: the energy its
            // gap is solved up to and the best adaptation there, in the walk's
            // units.
            rests: r
                .rests
                .iter()
                .map(|&((lo, hi), (from, floor))| {
                    ((kcal(lo), kcal(hi)), (Energy(kcal(from)), floor - offset))
                })
                .collect(),
            // Keyed as the rests are, by the box's ends and not where its gap
            // is solved from, which is the key the walk looks a gap up by.
            moved: r
                .moves
                .iter()
                .map(|&((lo, hi), moves)| ((kcal(lo), kcal(hi)), moves))
                .collect(),
        },
        // The two ends first, side by side, with any other worker taking
        // grid passes meanwhile; the rest of the grid is taken as probes ask
        // for it.
        _ => walk::Start {
            vertices: Vec::new(),
            ends: [None, None],
            pairs: Vec::new(),
            tasks: [true, false]
                .into_iter()
                .filter(|&stable| stable || !kernel.ends_in_turn())
                .map(|stable| {
                    let lambda = if stable {
                        1.0 - END_EPSILON
                    } else {
                        END_EPSILON
                    };
                    (walk::SETUP, walk::Task::End { lambda, stable })
                })
                .chain(
                    seed_weights
                        .iter()
                        .map(|&w| (walk::GRID, walk::Task::Grid(w))),
                )
                .collect(),
            // Seeded, the sweep starts from the pairs of neighbouring seeds as
            // each pair's two are known, in place of the pair of ends.
            seeds: (!seed_weights.is_empty()).then(|| {
                walk::Seeds::new(
                    (1.0 - END_EPSILON, None),
                    (END_EPSILON, None),
                    &seed_weights,
                    false,
                    None,
                )
            }),
            rests: FxHashMap::default(),
            moved: FxHashMap::default(),
        },
    };
    let drive = walk::Drive {
        budget: walk::Budget::new(budget, started, gap_folds, stop_open_share, true),
        solved_to: |rest: &(Energy, f64)| rest.0 .0,
        checkpoint,
        t: started,
        gap_secs: gap_cap.or_else(gap_secs),
        gap_double: gap_double(),
    };
    let stats = ProductStats {
        hull_folds: if resumed { 0 } else { 2 },
        ..ProductStats::default()
    };
    walk::run(kernel, start, drive, stats, threads)
}

/// What a motif walk taken over the plain lattice runs under.
struct PlainRun<'a> {
    budget: WalkBudget,
    resume: Option<&'a MotifResume>,
    checkpoint: Option<&'a (dyn Fn(&MotifWalk) + Sync)>,
}

/// The front of a spec whose motifs remove nothing from the lattice, by the
/// plain walk under `derna front`'s options on `threads` workers, reported as
/// a motif walk. The plain walk has no rests, gap-fold count or coverage
/// stop, so a run's [`MotifRun`] budget beyond time and memory is not read.
fn plain_walk<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    spec: &MotifFrontSpec<'_, S>,
    threads: usize,
    run: PlainRun<'_>,
) -> Result<MotifWalk, BeamError> {
    let options = crate::WalkOptions {
        budget: run.budget,
        ..crate::WalkOptions::for_residues(protein.len())
    };
    let resume = run.resume.map(|r| crate::WalkResume {
        points: r.points.clone(),
        open: r.open.clone(),
        rests: r.rests.clone(),
    });
    let hand = run
        .checkpoint
        .map(|hand| move |walked: &crate::WalkedFront| hand(&motif_walk_of(walked)));
    let walk_run = crate::WalkRun {
        resume: resume.as_ref(),
        checkpoint: hand
            .as_ref()
            .map(|h| h as &(dyn Fn(&crate::WalkedFront) + Sync)),
    };
    let walked = crate::design_front_walk_run(
        params,
        protein,
        usage,
        &flank_spec(spec.utr5),
        &flank_spec(spec.utr3),
        spec.constraints,
        END_EPSILON,
        1.0 - END_EPSILON,
        options,
        threads,
        walk_run,
    )?;
    Ok(motif_walk_of(&walked))
}

/// A spec's flank as the plain walk takes it: given when every position names
/// one nucleotide, else designed over each position's nucleotides.
fn flank_spec(s: &str) -> crate::UtrSpec {
    match Flank::parse(s) {
        Ok(Flank::Fixed(_)) | Err(_) => crate::UtrSpec::Fixed(s.to_string()),
        Ok(Flank::Designed(_)) => crate::UtrSpec::Mixed(s.to_string()),
    }
}

/// A plain walk's front as a motif walk reports it: the same points, hull and
/// open boxes; its probes are its sweep's folds and its passes its grid's.
fn motif_walk_of(walked: &crate::WalkedFront) -> MotifWalk {
    let vertex = |v: crate::HullPoint| MotifVertexPoint {
        mfe: v.energy(),
        cai: v.adaptation_log,
    };
    let span = |kind: crate::TaskKind| {
        walked
            .timeline
            .iter()
            .filter(move |s| s.kind == kind)
            .map(|s| (s.lambda, s.end_s - s.start_s, s.rss_kb))
    };
    MotifWalk {
        front: walked.front.clone(),
        vertices: walked.vertices.iter().copied().map(vertex).collect(),
        hull_folds: walked.sweep_scans,
        hull_s: walked.sweep_s,
        ends_s: 0.0,
        probe_s: 0.0,
        probes: walked.sweep_scans,
        probe_folds: walked.sweep_scans,
        probes_under_chord: walked.probes_cut,
        probe_records: Vec::new(),
        seeded: 0,
        ends_peak_kb: None,
        adapted_narrow: false,
        rehull: false,
        passes: span(crate::TaskKind::Grid)
            .map(|(lambda, secs, rss_kb)| PassCost {
                lambda,
                secs,
                rss_kb,
                aware_secs: 0.0,
                aware_folds: 0,
                aware_margin: 0.0,
                aware_cells: 0,
                aware_pairs: 0,
                aware_read: false,
            })
            .collect(),
        gaps: walked
            .gaps
            .iter()
            .map(|g| MotifGap {
                lambda: g.lambda,
                points: g.front.len(),
                folds: g.slices,
                cut_s: g.clock.scalar_s,
                fold_s: g.clock.fold_s,
                settled_s: 0.0,
                aware_out: 0,
                aware_dropped: 0,
                aware_unkeyed: 0,
                carried: 0,
                sized: 0,
                adopted: 0,
                adopted_points: 0,
            })
            .collect(),
        stopped_by: walked.stopped_by,
        open: walked
            .open
            .iter()
            .map(|&(a, b)| (vertex(a), vertex(b)))
            .collect(),
        rests: walked
            .rests
            .iter()
            .map(|r| MotifRest {
                lo: r.lo,
                hi: r.hi,
                from: r.from,
                floor: r.floor,
            })
            .collect(),
        gaps_moved_on: 0,
        moves: Vec::new(),
    }
}

/// The motif-free product lattice as the walk's [`walk::Kernel`]: bounded
/// folds of the product for the ends and probes, the plain lattice's passes
/// for the cuts, and [`solve_gap`]'s front fold for each gap.
struct ProductWalk<'a> {
    params: &'a Params,
    graph: &'a Dfa,
    product: &'a ProductDfa<'a>,
    lattice: &'a Lattice,
    region: &'a PairConstraint,
    grid: Grid<'a>,
    chain: Option<(&'a ChainCai, f64)>,
    /// The codons on some most adapted encoding, with the inputs a lattice
    /// of them alone is built from.
    #[allow(clippy::type_complexity)]
    most_adapted: (
        Option<&'a (f64, Vec<Vec<bool>>)>,
        (&'a [usize], &'a CodonUsage, &'a Flank, &'a Flank),
    ),
    motifs: (&'a RegionConstraints, &'a Ac, MotifWindow),
    designs: Designs,
    /// The probes' span memos, one per worker that has probed: a memo is
    /// the same at any weight, so a worker reuses whichever is free.
    spans: std::sync::Mutex<Vec<crate::constrained::SpanMemo>>,
    slice_budget: SliceBudget,
    probe_rule: ProbeRule,
    fold_threads: usize,
    threads: usize,
    capped: bool,
    resume: Option<&'a MotifResume>,
    offset: f64,
    started: std::time::Instant,
    narrowed: std::sync::atomic::AtomicBool,
    /// Hull vertices the aware passes seeded.
    seeded: AtomicUsize,
    /// Ends folded, the seconds from the walk's start until the second, and
    /// the peak resident memory then.
    ends: std::sync::Mutex<(usize, f64, Option<u64>)>,
    longest_probe: std::sync::Mutex<f64>,
}

/// What the product walk counts of its sweep.
#[derive(Debug, Clone, Default)]
struct ProductStats {
    hull_folds: usize,
    hull_end: f64,
    probe_s: f64,
    probes: usize,
    probe_folds: usize,
    probes_under_chord: usize,
    records: Vec<ProbeRecord>,
}

impl ProductWalk<'_> {
    /// Whether the ends are folded one after the other: under a memory
    /// budget, or on one worker.
    fn ends_in_turn(&self) -> bool {
        self.threads <= 1 || self.capped
    }

    /// What stops this walk's gap folds short, read from `budget`.
    fn stop<'b>(
        budget: &'b walk::Budget,
        time: &'b (dyn Fn() -> bool + Sync),
        memory: &'b (dyn Fn() -> bool + Sync),
    ) -> Stop<'b> {
        Stop {
            deadline: budget.has_deadline().then_some(time),
            memory: budget.has_memory().then_some(memory),
            folds: budget.folds.as_ref().map(|(run, n)| (run, *n)),
            gap_secs: None,
            gap_deadline: None,
        }
    }

    /// The ends have no incumbent to bound them, so each runs the bounded
    /// search `design` runs: ceilings raised from the plain optimum until one
    /// certifies.
    fn end_fold(&self, lambda: f64) -> Result<MotifVertexPoint, BeamError> {
        let mut report = crate::bounded_motifs::BoundReport::default();
        crate::bounded_motifs::fold(
            self.params,
            self.graph,
            self.product,
            lambda,
            self.region.clone(),
            crate::must_pair::MustPairConstraint::inert(),
            crate::ScanConfig {
                threads: self.fold_threads,
                ..crate::ScanConfig::default()
            },
            &mut report,
        )
        .map(|f| self.designs.keep(f))
        .ok_or(BeamError::Infeasible)
    }

    /// The adapted end is the most adapted motif-free design, with the best
    /// energy among those. Every such design spells only codons on some most
    /// adapted encoding, so it is folded first over the lattice of those
    /// codons alone, which is far smaller than the whole one. A fold there
    /// that reaches the best adaptation is the end; one that does not leaves
    /// the end to the whole lattice.
    fn adapted_end(&self) -> Result<MotifVertexPoint, BeamError> {
        let (most_adapted, inputs) = self.most_adapted;
        let narrow = most_adapted.and_then(|(best, codons)| {
            let fold =
                most_adapted_fold(self.params, inputs, self.motifs, codons, self.fold_threads)?;
            let cai = fold.cai - self.graph.codon_offset();
            ((cai - best).abs() <= 1e-6)
                .then_some(crate::constrained::ConstrainedFold { cai, ..fold })
        });
        match narrow {
            Some(f) => {
                self.narrowed
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(self.designs.keep(f))
            }
            None => self.end_fold(END_EPSILON),
        }
    }

    fn exact(&self, lambda: f64) -> Result<MotifVertexPoint, BeamError> {
        crate::constrained::fold_with_beam(
            self.product,
            self.params,
            lambda,
            crate::Beam::exact(),
            self.region.clone(),
            crate::must_pair::MustPairConstraint::inert(),
            self.fold_threads,
        )
        .map(|f| self.designs.keep(f))
        .ok_or(BeamError::Infeasible)
    }

    /// Settle `task` as [`settle`] does, under `budget`, its gap's fold
    /// moving on after `cap` seconds, if given.
    fn settle(
        &self,
        task: Task,
        (rest, cap): (Option<(Energy, f64)>, Option<walk::GapCap>),
        budget: &walk::Budget,
    ) -> Settled {
        let time = || budget.time_spent();
        let memory = || budget.memory_spent();
        let stop = Stop {
            gap_secs: cap,
            ..Self::stop(budget, &time, &memory)
        };
        let mut spans = self.spans.lock().expect("the span memos' lock").pop();
        let exact = |lambda: f64| self.exact(lambda);
        let settled = settle(
            (task, rest),
            self.params,
            (self.product, self.lattice, self.region),
            (&self.grid, self.chain),
            &mut spans,
            &exact,
            (&self.designs, stop, self.slice_budget, &self.probe_rule),
        );
        if let Some(spans) = spans {
            self.spans.lock().expect("the span memos' lock").push(spans);
        }
        settled
    }

    /// The walk so far as a [`MotifWalk`]: its vertices, the points of the
    /// gaps it solved and of every vertex, and as open every pair in `open`.
    fn assemble(
        &self,
        walked: &walk::Walk<Self>,
        mut open: Vec<(MotifVertexPoint, MotifVertexPoint)>,
        stopped_by: Option<&'static str>,
    ) -> MotifWalk {
        let kept = self.designs.0.lock().expect("the designs' lock");
        let mut vertices = walked.vertices.clone();
        vertices.sort_by_key(|v| v.mfe.0);
        let mut solved: Vec<&(i32, GapFront)> = walked.gaps.iter().collect();
        solved.sort_by_key(|(a, _)| *a);
        open.sort_by_key(|(a, _)| a.mfe.0);

        let mut gaps = Vec::new();
        let mut found = Vec::new();
        let mut inner = Vec::new();
        for (key, gap) in solved.iter().map(|g| (g.0, &g.1)) {
            gaps.push(gap.record);
            found.extend(gap.points.iter().cloned());
            // A gap folded whole marks its own vertices; a probed gap holds
            // none strictly inside it.
            let ends = vertices.iter().position(|v| v.mfe.0 == key);
            if let Some(&[a, b]) = ends.and_then(|i| vertices.get(i..i + 2)) {
                let within = gap
                    .points
                    .iter()
                    .filter(|p| p.0 > a.mfe && p.0 < b.mfe)
                    .map(|p| MotifVertexPoint { mfe: p.0, cai: p.1 });
                let hull = lower_hull(std::iter::once(a).chain(within).chain([b]));
                inner.extend_from_slice(&hull[1..hull.len() - 1]);
            }
        }
        vertices.extend(inner);
        vertices.sort_by_key(|v| v.mfe.0);
        found.extend(walked.rest_points.iter().cloned());
        // A vertex beside a gap no fold finished is returned from its own
        // scalar fold; one a finished gap holds is returned as that gap found
        // it.
        for v in &vertices {
            if let Some((cai, sequence, structure)) = kept.get(&v.mfe.0) {
                found.push((v.mfe, *cai, sequence.clone(), structure.clone()));
            }
        }
        drop(kept);
        found.sort_by_key(|p| p.0 .0);
        let m = self.lattice_residues();
        let mut front: Vec<FrontPoint> = Vec::new();
        for (mfe, cai, sequence, structure) in found {
            // Adjacent gaps share their vertex.
            if front
                .last()
                .is_some_and(|q| (q.mfe_kcal_mol - mfe.to_kcal_mol()).abs() < 1e-9)
            {
                continue;
            }
            let cai_log = cai + self.offset;
            front.push(FrontPoint {
                sequence,
                structure,
                mfe_kcal_mol: mfe.to_kcal_mol(),
                cai_log,
                cai: if m == 0 {
                    0.0
                } else {
                    (cai_log / m as f64).exp()
                },
                utr_profile_log: 0.0,
            });
        }
        // A resumed walk's own points sit in the earlier walk's open boxes,
        // so they meet its points only at a shared vertex, which keeps its
        // old line.
        if let Some(r) = self.resume {
            let energy = |p: &FrontPoint| (p.mfe_kcal_mol * 100.0).round() as i64;
            let old: std::collections::HashSet<i64> =
                r.points.iter().map(|(p, _)| energy(p)).collect();
            front.retain(|p| !old.contains(&energy(p)));
            front.extend(r.points.iter().map(|(p, _)| p.clone()));
            front.sort_by_key(energy);
            if r.rehull {
                vertices = lower_hull(front.iter().map(|p| MotifVertexPoint {
                    mfe: Energy(energy(p) as i32),
                    cai: p.cai_log - self.offset,
                }));
            }
        }
        // The walk works in the lattice's units; a caller reads the report's,
        // which differ by the offset a pin's restriction removed.
        let report = |v: MotifVertexPoint| MotifVertexPoint {
            cai: v.cai + self.offset,
            ..v
        };
        let (_, ends_s, ends_peak_kb) = *self.ends.lock().expect("the ends' lock");
        let stats = &walked.stats;
        MotifWalk {
            front,
            vertices: vertices.into_iter().map(report).collect(),
            hull_folds: stats.hull_folds,
            hull_s: stats.hull_end,
            ends_s,
            probe_s: stats.probe_s,
            probes: stats.probes,
            probe_folds: stats.probe_folds,
            probes_under_chord: stats.probes_under_chord,
            probe_records: stats.records.clone(),
            seeded: self.seeded.load(Ordering::Relaxed),
            ends_peak_kb,
            adapted_narrow: self.narrowed.load(std::sync::atomic::Ordering::Relaxed),
            rehull: self.resume.is_some_and(|r| r.rehull),
            passes: self.grid.costs(),
            gaps,
            stopped_by,
            rests: open
                .iter()
                .filter_map(|&(a, b)| {
                    walked.rest_of(a, b).map(|(from, floor)| MotifRest {
                        lo: a.mfe,
                        hi: b.mfe,
                        from,
                        floor: floor + self.offset,
                    })
                })
                .collect(),
            moves: open
                .iter()
                .map(|&(a, b)| (a.mfe, b.mfe, walked.moves_of(a, b)))
                .filter(|&(_, _, moves)| moves > 0)
                .collect(),
            open: open
                .into_iter()
                .map(|(a, b)| (report(a), report(b)))
                .collect(),
            gaps_moved_on: walked.moved_on,
        }
    }

    /// The sense codons, which a point's mean adaptation is over.
    fn lattice_residues(&self) -> usize {
        derna_codon_dp::cai::sense_codons(self.most_adapted.1 .0)
    }
}

impl walk::Kernel for ProductWalk<'_> {
    type Vertex = MotifVertexPoint;
    /// Product walks fold each gap in its probe's task.
    type Job = std::convert::Infallible;
    type Gap = GapFront;
    type Rest = (Energy, f64);
    type Point = (Energy, f64, String, String);
    type Note = Probe;
    type Stats = ProductStats;
    type Out = MotifWalk;

    fn energy(v: MotifVertexPoint) -> i32 {
        v.mfe.0
    }

    fn tie(a: MotifVertexPoint, b: MotifVertexPoint) -> Option<f64> {
        (b.mfe > a.mfe).then(|| tie(a, b))
    }

    fn policy(&self) -> walk::Policy {
        walk::Policy {
            shares: crate::task_queue::Shares::One,
            max_running: self.threads.max(1),
            checkpoint_skips: true,
            ends_in_turn: self.ends_in_turn(),
        }
    }

    /// The task at the highest tie weight goes first. A box's folds cost
    /// more the lower its weight — near the adapted end energy barely
    /// separates designs and the cuts prune little — so a walk stopped by its
    /// budget has spent it on the boxes that settle soonest. A development
    /// build can put the box spanning the most energy first instead
    /// ([`span_first`]).
    /// A narrow box between adjacent grid optima, both hull vertices, at a
    /// tie weight of at least [`CELL_FOLD_TIE`], is folded whole from just
    /// below its stable end: the gap fold's cuts hold between any two front
    /// points, and the box's vertices are read from its points
    /// ([`lower_hull`]).
    fn cell_rest(&self, a: MotifVertexPoint, b: MotifVertexPoint) -> Option<(Energy, f64)> {
        (f64::from(b.mfe.0 - a.mfe.0) <= 100.0 * cell_fold_de() && tie(a, b) >= cell_fold_tie())
            .then_some((Energy(a.mfe.0 - 1), a.cai))
    }

    fn probe_priority(&self, a: MotifVertexPoint, b: MotifVertexPoint) -> (u8, u64) {
        (walk::PROBE.0, task_weight(Task::Pair(a, b)).to_bits())
    }

    fn lone_priority(&self) -> (u8, u64) {
        let v = MotifVertexPoint {
            mfe: Energy(0),
            cai: 0.0,
        };
        (walk::PROBE.0, task_weight(Task::Lone(v)).to_bits())
    }

    fn probe_would_overrun(
        &self,
        _pair: (MotifVertexPoint, MotifVertexPoint),
        rest: bool,
        budget: &walk::Budget,
    ) -> bool {
        let longest = *self.longest_probe.lock().expect("the probes' lock");
        let overrun = !rest
            && budget.deadline().is_some_and(|d| {
                d.saturating_duration_since(std::time::Instant::now())
                    .as_secs_f64()
                    < longest
            });
        if overrun {
            budget.stop_for_time();
        }
        overrun
    }

    fn idle(&self, budget: &walk::Budget) -> bool {
        !budget.spent() && self.grid.prefetch()
    }

    fn adapted_weight(&self) -> f64 {
        END_EPSILON
    }

    fn kind(_note: Option<Probe>) -> crate::TaskKind {
        crate::TaskKind::Probe { cut: false }
    }

    fn credit(&self, _workers: usize, _queue: &walk::WalkQueue<Self>) -> Result<(), BeamError> {
        unreachable!("a product walk's gap folds read the grid's pass at weight 1")
    }

    fn end(
        &self,
        _lambda: f64,
        stable: bool,
        _workers: usize,
    ) -> Result<MotifVertexPoint, BeamError> {
        let v = if stable {
            let started = std::time::Instant::now();
            let v = self.end_fold(1.0 - END_EPSILON);
            let secs = started.elapsed().as_secs_f64();
            *self.longest_probe.lock().expect("the probes' lock") = secs;
            v
        } else {
            self.adapted_end()
        };
        let mut ends = self.ends.lock().expect("the ends' lock");
        ends.0 += 1;
        if ends.0 == 2 {
            ends.1 = self.started.elapsed().as_secs_f64();
            ends.2 = crate::task_clock::peak_rss_kb();
        }
        v
    }

    /// Only a seeded walk queues its grid's passes; an unseeded one takes
    /// them as its folds read them.
    fn grid(
        &self,
        lambda: f64,
        seed: bool,
        _workers: usize,
        _queue: &walk::WalkQueue<Self>,
    ) -> Result<Option<MotifVertexPoint>, BeamError> {
        let k = self
            .grid
            .weights
            .iter()
            .position(|&w| w == lambda)
            .expect("a seed is at a grid weight");
        let pass = self.grid.pass(k, None);
        if !seed {
            return Ok(None);
        }
        self.seeded.fetch_add(1, Ordering::Relaxed);
        // A pass that certified no optimum leaves its seed to an exact fold,
        // so every slot is filled and the pairs around it are handed out.
        match pass.aware.as_ref().and_then(|a| a.design.clone()) {
            Some(design) => Ok(Some(self.designs.keep(design))),
            None => self.end_fold(lambda).map(Some),
        }
    }

    fn probe(
        &self,
        (a, b): (MotifVertexPoint, MotifVertexPoint),
        rest: (Option<(Energy, f64)>, Option<walk::GapCap>),
        _bracket: Option<usize>,
        _workers: usize,
        _queue: &walk::WalkQueue<Self>,
        budget: &walk::Budget,
        _vertices: &dyn Fn() -> Vec<MotifVertexPoint>,
    ) -> Result<walk::Probed<Self>, BeamError> {
        let settled = self.settle(Task::Pair(a, b), rest, budget);
        if let Settled::Vertex(_, probe)
        | Settled::Gap(_, Some(probe))
        | Settled::Rest(_, Some(probe)) = settled
        {
            let mut longest = self.longest_probe.lock().expect("the probes' lock");
            *longest = longest.max(probe.secs);
        }
        Ok(match settled {
            Settled::Vertex(v, probe) => walk::Probed::Split(v, probe),
            Settled::Gap(gap, probe) => walk::Probed::Settled(walk::Settled::Gap(gap), probe),
            Settled::Rest(rest, probe) if rest.moved_on => walk::Probed::Settled(
                walk::Settled::MovedOn((rest.from, rest.floor), rest.points),
                probe,
            ),
            Settled::Rest(rest, probe) => walk::Probed::Settled(
                walk::Settled::Rest((rest.from, rest.floor), rest.points),
                probe,
            ),
            Settled::Open => walk::Probed::Settled(walk::Settled::Open, None),
        })
    }

    fn fold(
        &self,
        job: std::convert::Infallible,
        _workers: usize,
        _queue: &walk::WalkQueue<Self>,
        _budget: &walk::Budget,
        _vertices: &dyn Fn() -> Vec<MotifVertexPoint>,
    ) -> Result<(walk::Settled<Self>, usize), BeamError> {
        match job {}
    }

    fn job_pair(job: &std::convert::Infallible) -> (MotifVertexPoint, MotifVertexPoint, f64, f64) {
        match *job {}
    }

    fn lone(
        &self,
        v: MotifVertexPoint,
        budget: &walk::Budget,
    ) -> Result<walk::Settled<Self>, BeamError> {
        Ok(match self.settle(Task::Lone(v), (None, None), budget) {
            Settled::Gap(gap, _) => walk::Settled::Gap(gap),
            _ => walk::Settled::Open,
        })
    }

    fn note(stats: &mut ProductStats, probe: Probe) {
        stats.hull_folds += 1;
        stats.probe_s += probe.secs;
        stats.probes += 1;
        stats.probe_folds += probe.folds;
        stats.probes_under_chord += usize::from(probe.under_chord);
        stats.records.push(ProbeRecord {
            lambda: probe.lambda,
            secs: probe.secs,
            folds: probe.folds,
            pruned: probe.counts.pruned,
            cut: probe.counts.cut,
            aware_pruned: probe.counts.aware_pruned,
            aware_cut: probe.counts.aware_cut,
        });
    }

    fn settled_gap(stats: &mut ProductStats, gap: &mut GapFront, at_s: f64) {
        gap.record.settled_s = at_s;
        stats.hull_end = stats.hull_end.max(at_s);
    }

    fn snapshot(
        &self,
        walk: &walk::Walk<Self>,
        open: Vec<(MotifVertexPoint, MotifVertexPoint)>,
    ) -> MotifWalk {
        self.assemble(walk, open, None)
    }

    fn finish(
        self,
        walk: walk::Walk<Self>,
        end: walk::End,
        _threads: usize,
    ) -> Result<MotifWalk, BeamError> {
        let open = walk.outstanding.clone();
        Ok(self.assemble(&walk, open, end.stopped_by))
    }
}

/// The weight a task folds at: its pair's tie, or the stable end's.
fn task_weight(task: Task) -> f64 {
    match task {
        Task::Pair(a, b) if span_first() => f64::from(b.mfe.0 - a.mfe.0),
        Task::Pair(a, b) => tie(a, b),
        Task::Lone(_) if span_first() => f64::INFINITY,
        Task::Lone(_) => 1.0 - END_EPSILON,
    }
}

/// Whether the queue takes the box spanning the most energy first rather
/// than the one at the highest weight: no, unless a development build asks. Any
/// order settles the same boxes when the walk finishes; a walk stopped short
/// keeps different ones open.
fn span_first() -> bool {
    static SPAN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SPAN.get_or_init(|| {
        crate::dev_env::var(crate::dev_env::Setting::MotifGapOrder).is_ok_and(|o| o == "span")
    })
}

/// The weight at which `a` and `b` score alike.
fn tie(a: MotifVertexPoint, b: MotifVertexPoint) -> f64 {
    let de = f64::from(b.mfe.0 - a.mfe.0);
    let dl = 100.0 * (b.cai - a.cai);
    dl / (de + dl)
}

/// The vertices of the lower hull of `front`, most stable first: the points
/// a walk from the ends marks, each strictly below the chord between its
/// neighbouring vertices at their tie weight, by the probe's tolerance.
fn lower_hull(front: impl Iterator<Item = MotifVertexPoint>) -> Vec<MotifVertexPoint> {
    const TOL: f64 = 1e-6;
    let mut hull: Vec<MotifVertexPoint> = Vec::new();
    for p in front {
        while let [.., a, b] = hull[..] {
            let lambda = tie(a, p);
            if brd(b.mfe, b.cai, lambda) < brd(a.mfe, a.cai, lambda) - TOL {
                break;
            }
            hull.pop();
        }
        hull.push(p);
    }
    hull
}

/// One solved gap: its front points, most stable first, and its record.
struct GapFront {
    points: Vec<(Energy, f64, String, String)>,
    record: MotifGap,
}

/// A gap stopped part way: solved up to `from`, with `floor` the best
/// adaptation there and `points` the front points found; `moved_on` when the
/// per-gap time cap stopped it rather than the walk's budget.
struct GapRest {
    from: Energy,
    floor: f64,
    points: Vec<(Energy, f64, String, String)>,
    moved_on: bool,
}

/// The pass grid of a motif walk over `residues` coding residues:
/// [`crate::PassGrid::for_residues`], unless a development build gives its segments
/// as `until:step,...`.
fn motif_pass_grid(residues: usize) -> crate::PassGrid {
    crate::dev_env::var(crate::dev_env::Setting::MotifPassGrid)
        .ok()
        .and_then(|g| crate::PassGrid::parse(&g))
        .unwrap_or_else(|| crate::PassGrid::for_residues(residues))
}

/// Whether an on/off knob is on: `default` when `setting` is unset, otherwise
/// on unless it reads `0` or `off`.
fn env_switch(setting: crate::dev_env::Setting, default: bool) -> bool {
    crate::dev_env::var(setting).map_or(default, |v| !matches!(v.trim(), "0" | "off"))
}

/// A non-negative number of seconds from `setting`: `default` when unset, and
/// none when it reads `none`, `off` or anything that is not such a number.
fn env_secs(setting: crate::dev_env::Setting, default: f64) -> Option<f64> {
    match crate::dev_env::var(setting) {
        Err(_) => Some(default),
        Ok(s) => s
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|s| s.is_finite() && *s >= 0.0),
    }
}

/// The seconds a gap's fold may take in one task before the walk moves on to
/// its other boxes, keeping the gap's rest: 300, unless a development build sets
/// another or none.
fn gap_secs() -> Option<f64> {
    env_secs(crate::dev_env::Setting::MotifGapSecs, 300.0)
}

/// Whether the per-gap time cap doubles each time a gap moves on, so its
/// `k`-th return gets `2^k` times the cap: yes, unless a development build turns it
/// off.
fn gap_double() -> bool {
    env_switch(crate::dev_env::Setting::MotifGapDouble, true)
}

/// Whether the per-gap time cap also stops a fold in flight, before any
/// slice of the turn has solved: yes, unless a development build turns it off.
fn gap_midslice() -> bool {
    env_switch(crate::dev_env::Setting::MotifGapMidslice, true)
}

/// The fewest seconds a fold in flight runs before the mid-slice stop may end
/// it, when longer than the gap's cap: 3600, unless a development build sets another
/// or none. A slice that needs a little more than the cap then finishes, and
/// the gap moves on at the next slice boundary; only a fold past both is
/// lost.
fn gap_midslice_secs() -> Option<f64> {
    env_secs(crate::dev_env::Setting::MotifGapMidsliceSecs, 3600.0)
}

/// The mid-slice floor on a gap's `visit`-th task (the first being 1):
/// `least`, doubled once per earlier move as the cap is ([`gap_double`]), so a first slice longer than the floor is cut at most a
/// few times before a visit gives it room to finish.
fn midslice_floor(least: f64, visit: u32) -> f64 {
    if gap_double() {
        least * 2f64.powi(i32::try_from(visit.saturating_sub(1)).unwrap_or(i32::MAX))
    } else {
        least
    }
}

/// The most points a gap's fold may store before its energy range is split
/// in two and each half folded on its own, unless [`MotifRun::gap_budget`]
/// gives another; a development build can override it.
fn default_gap_budget() -> usize {
    crate::dev_env::var(crate::dev_env::Setting::MotifGapBudget)
        .ok()
        .and_then(|b| b.parse().ok())
        .unwrap_or(2_000_000)
}

/// The most a slice may store, as a share of what the fold of the range it
/// was cut from stored by the same span, for halving it again to pay.
/// A development build can override it.
fn slice_shrink() -> f64 {
    crate::dev_env::var(crate::dev_env::Setting::MotifSliceShrink)
        .ok()
        .and_then(|b| b.parse().ok())
        .unwrap_or(0.9)
}

/// The narrowest energy range, in the fold's units, that is split further.
const MIN_SLICE: i32 = 50;

/// The most a gap's next range may span on its first try, as a multiple of
/// the width of its last solved range, once a range of the gap has given up.
/// A range taken whole after the gap's budget grew gave up and halved its
/// way back down: on Q06187 (659 residues, manufacturing motifs) 5,107 of
/// the 10,436 s spent on ranges that gave up followed a budget change whose
/// first try spanned up to 128 times the last solved width, while the width
/// that solved was a median 1.76 times it. Where the next budget is smaller
/// than the solved range's, the width shrinks with it.
/// A development build can override it.
const SLICE_STEP: f64 = 2.0;

fn slice_step() -> f64 {
    crate::dev_env::var(crate::dev_env::Setting::MotifSliceStep)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(SLICE_STEP)
}

/// Whether a gap's later ranges carry forward what its most recently solved
/// range stored and how wide it was, at any width. Without a memory budget
/// and with the slice growth above 0, each later budgeted range of the gap
/// is folded with a budget of at least the growth times what that range
/// stored. Once a range of the gap has given up, a later range is cut by
/// [`slice_step`] times that range's width, as the slice sizing cuts.
///
/// A range that gives up costs about the time to store its budget: on
/// Q06187 (659 and 670 residues) a give-up took 4.5 s at a budget of 2
/// million points and 208 s at 50 million. A small gap budget is then a
/// cheap search for a width that solves, and what that search learned is
/// kept for the rest of the gap instead of being searched for again.
///
/// Nothing bounds the carried budget but the growth: each solved range can
/// raise the next one's to twice what it stored, so a fold can store more
/// than the same range would without the carry. On Q06187 F2 (resumed, one
/// hour, at about 40 bytes a point) the largest solved range held 87.2
/// million points against 50.0 million without it. `--max-rss-gb` bounds
/// the store and turns the carry off; a development build can turn it off alone.
fn density_carry() -> bool {
    crate::dev_env::var(crate::dev_env::Setting::MotifDensityCarry).map_or(true, |v| v != "0")
}

/// What becomes of the energies `(top, gap_top]` left of a gap once a slice
/// `width` wide, too narrow to split, stored `stored` points past its budget
/// `budget`: ranges with their budgets, the lowest last, or `None` to keep
/// them as they are.
///
/// They become one range with budget `grown` when that holds at least what
/// the slice stored. A smaller `grown` is the memory room binding: a merged
/// range would give up, and so would every wider range left, each halving
/// down to slices of the narrowest width. So they become slices `width` wide
/// at once, as wide as the one that fitted, with no budget to give up on.
///
/// With `even`, the slices share the rest equally, each at most a tenth
/// wider than `width`. A slice costs about the same at any width (at
/// P0DTC2[:1000] a slice of 1 dekacalorie stored 156 million points in
/// 314 s, one of 50 stored 182 million), so the narrow slice left over at
/// the bottom was a fold spent on nearly nothing.
fn rest_of_gap(
    (top, gap_top): (i32, i32),
    width: i32,
    stored: usize,
    (grown, budget): (usize, usize),
    even: bool,
) -> Option<Vec<(i32, i32, usize)>> {
    if grown >= stored {
        return (grown > budget).then(|| vec![(top, gap_top, grown)]);
    }
    let width = width.max(1);
    if even {
        let wide = (width + width / 10).max(width);
        return Some(
            even_ranges(top, gap_top, wide)
                .into_iter()
                .map(|(lo, hi)| (lo, hi, usize::MAX))
                .collect(),
        );
    }
    let mut rest = Vec::new();
    let mut hi = gap_top;
    while hi > top {
        let lo = (hi - width).max(top);
        rest.push((lo, hi, budget));
        hi = lo;
    }
    Some(rest)
}

/// The energies `(lo, hi]` cut into as few ranges at most `width` wide as
/// will do, of equal widths to a unit, the lowest last.
fn even_ranges(lo: i32, hi: i32, width: i32) -> Vec<(i32, i32)> {
    let len = hi - lo;
    if len <= 0 {
        return Vec::new();
    }
    let k = (len + width.max(1) - 1) / width.max(1);
    (0..k)
        .rev()
        .map(|i| (lo + len * i / k, lo + len * (i + 1) / k))
        .collect()
}

/// The most points a gap's fold may store, and how far the rest of a gap
/// folded again as one range may grow past it ([`solve_gap`]).
#[derive(Clone, Copy)]
struct SliceBudget {
    points: usize,
    /// The multiple of a narrowest slice's storage the rest may store.
    grow: usize,
    /// The walk's memory budget, in KiB: the rest stores no more than the
    /// room left under it allows.
    max_rss_kb: Option<f64>,
    /// The walk's workers, which share that room.
    workers: usize,
    /// Whether a gap's first fold may store its worker's share of the room
    /// rather than `points`.
    open_in_room: bool,
    /// The width a gap's first ranges are cut to, over [`GAP_FIT`].
    fit: Option<i32>,
    /// Whether the slices are also cut by each bracketing pass alone.
    bracket: bool,
    /// The widest a gap's first ranges may be, in the fold's units; `None`
    /// folds the whole gap first ([`MotifRun::slice_width`]).
    width: Option<i32>,
    /// How many folds that gave up a gap keeps for its later ranges to
    /// start from ([`gap_reuse`]); zero keeps none.
    reuse: usize,
}

impl SliceBudget {
    /// The points a fold that gave up at `stored` may still store, read from
    /// the resident memory it grew by since `before_kb`; none when that
    /// leaves less than a twentieth of what it stored.
    fn more_in_room(&self, before_kb: u64, stored: usize) -> Option<usize> {
        let max = self.max_rss_kb?;
        release_freed();
        let now = rss_kb()?;
        let per_point = (now.saturating_sub(before_kb) as f64 * 1024.0 / stored.max(1) as f64)
            .max(ROOM_PER_POINT / 2.0);
        let left = (max * (1.0 - ROOM_RESERVE) - now as f64).max(0.0) / self.workers.max(1) as f64;
        let more = (left * 1024.0 / per_point) as usize;
        (more > stored / 20).then_some(more)
    }

    /// The points a fold may store in the room left under the memory budget,
    /// once memory freed so far is handed back; unbounded without a budget.
    fn room(&self) -> usize {
        self.max_rss_kb.map_or(usize::MAX, |max| {
            release_freed();
            let left = (max - rss_kb().unwrap_or(0) as f64).max(0.0);
            (left * 1024.0 / ROOM_PER_POINT) as usize
        })
    }
}

/// Whether, under a memory budget, a gap's first folds may store its
/// worker's share of the room left rather than the gap budget (over ranges
/// sized from the last solved fold under [`gap_density`]): yes, unless
/// a development build turns it off. At P0DTC2[:1000] every gap's first two folds gave
/// up at the 2 million point budget, and its first slice was later folded
/// again by the range above it; a whole gap folded in the room gave up after
/// 262 s, where its halves then fit.
/// Whether a gap's first ranges in the room are sized from the points per
/// dekacalorie the last solved fold stored: no, unless a development build turns it
/// on. Off by default: a narrow fold stores far more per dekacalorie than a wide
/// one (at P0DTC2[:1000], 81 million points over 11 against 131 million over
/// 135), so after a narrow gap the next was cut into folds of 17 and 19
/// dekacalories, 133 and 145 s, where one fold fitted in the room.
fn gap_density() -> bool {
    crate::dev_env::var(crate::dev_env::Setting::MotifGapDensity).is_ok_and(|v| v == "1")
}

/// Points per dekacalorie the last solved gap fold stored, as `f64` bits;
/// zero before any. Only the size of a gap's first ranges reads it, so it
/// changes what a walk costs, never what it returns.
static GAP_DENSITY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many times the width of the last range that solved a gap's next
/// range may be: 1.5, unless a development build sets another of at least 1 or none.
/// Without a
/// limit, a slice that outgrew its budget merges the rest of the gap into one
/// range, which gives up and halves back down: at Q6PF06 under motif set I,
/// 27 of 31 give-ups were wider than the slice before them, 57% of the gap's
/// folding.
fn gap_carry() -> Option<f64> {
    match crate::dev_env::var(crate::dev_env::Setting::MotifGapCarry) {
        Err(_) => Some(1.5),
        Ok(s) => s.trim().parse::<f64>().ok().filter(|&k| k >= 1.0),
    }
}

/// Whether a gap's carry drops to 1 once a range wider than the last that
/// solved gives up, off unless a development build turns it on: the width that last
/// solved is then the most the gap's later ranges are given.
fn gap_carry_adapt() -> bool {
    crate::dev_env::var(crate::dev_env::Setting::MotifGapCarryAdapt).is_ok_and(|v| v == "1")
}

/// Whether a gap's folds are sized from the last that fitted: yes, unless
/// a development build turns it off. Under a memory budget a gap's first ranges are
/// as wide as the last cut fold that solved ([`GAP_FIT`]), the rest of a gap
/// is cut in equal slices ([`rest_of_gap`]), and a fold whose budget gives
/// up continues while the resident memory it measured leaves room and its
/// halves would store more than it ([`CONTINUE_KEPT`]). At
/// P0DTC2[:1000] a gap of 741 dekacalories at weight 0.41 gave up 4 times
/// (885 s) halving down to 46, where the gap before had solved at 50; over
/// one round, give-ups took 26% of the gap folds' time and slices left over
/// at a gap's bottom another 10%.
fn gap_fit() -> bool {
    crate::dev_env::var(crate::dev_env::Setting::MotifGapFit).map_or(true, |v| v != "0")
}

/// The width of the last gap fold that solved, in the room, a range cut from
/// its gap; zero before any. Like
/// [`GAP_DENSITY`], it changes what a walk costs, never what it returns.
static GAP_FIT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// The least share of its parent's points, at the last span both filled, at
/// which a fold that gave up continues in the room rather than halving: two
/// halves that each keep more than half store more than the parent did. At
/// PRT-04[:500] under a 2 GB budget the whole gap was continued from 2 to
/// 42.6 million points over 138 s before giving up, where its halves kept
/// 0.025 and 0.060 of it and solved in 2.5 and 4.8 s; at P0DTC2[:1000]
/// halves kept 0.49 to 0.84.
const CONTINUE_KEPT: f64 = 0.5;

/// The resident memory a fold under the memory budget may still grow by,
/// as a share of that budget kept back for what the fold holds while it
/// runs.
const ROOM_RESERVE: f64 = 0.05;

/// Whether a gap's slices are also cut by each pass bracketing the gap's
/// weight on its own, through the slice's corner at that pass's weight,
/// unless [`MotifRun::bracket_cuts`] says: yes, unless a development build turns it
/// off.
///
/// A cut at weight `mu` through the corner (top, floor) rules a point out
/// when `A(mu) = mu (E - top) - (1 - mu) 100 (C - floor) + out_mu > 0`, with
/// `out_mu` the pass's floor on its completion; it holds for every pass, as
/// for the chord. At the gap's weight `lambda = w lo + (1 - w) hi` the chord
/// cut's `A` is `w A(lo) + (1 - w) A(hi)`, since all but `out` is linear in
/// the weight, so each point the chord rules out one of the two passes rules
/// out alone. The two cost two floors tables per gap (n² each) and nothing
/// per point beyond the chord's two reads. At PRT-04[:500], on the gap at
/// weight 0.3422 between the passes at 0.31 and 0.41, the slices stored 6.57
/// million points against 8.59 million, in 5 folds against 7 and 16 s
/// against 19, with the same front.
fn default_bracket_cuts() -> bool {
    crate::dev_env::var(crate::dev_env::Setting::MotifBracketCuts).map_or(true, |v| v != "0")
}

/// How many folds that gave up a gap are kept for its later ranges to start
/// from: one, unless a development build sets another, 0 for none. A range of a gap is folded
/// under an energy ceiling at its top and an adaptation floor at the best
/// found below it, so a later range inside a range that gave up, at no
/// higher a top and with a floor that has only risen, folds under a box
/// inside that range's. It then starts from that fold's complete spans held
/// to its own box ([`Fold::adopt`]) rather than from nothing, and returns
/// the same points. The most recent folds kept come first; a kept fold is
/// freed once the gap is solved past its top.
fn gap_reuse() -> usize {
    crate::dev_env::var(crate::dev_env::Setting::MotifGapReuse)
        .map_or(Some(1), |s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// A fold that gave up a gap's range, kept for the ranges inside it: the
/// range's top, the adaptation floor it pruned on, and its tables.
struct GaveUp {
    top: i32,
    floor: f64,
    tables: Tables,
}

fn gap_room() -> bool {
    crate::dev_env::var(crate::dev_env::Setting::MotifGapRoom).map_or(true, |v| v != "0")
}

/// Bytes of resident memory a stored point is allowed when the room left
/// under the memory budget caps a merged range's budget: its 24 bytes and
/// the cells and maps around it. At P0DTC2[:1000] slices of 71-75 million
/// 32-byte points held 37 bytes per point once their fold was freed; the
/// rest is margin for what a fold holds while it runs.
const ROOM_PER_POINT: f64 = 40.0;

/// Hand the memory freed so far back to the system, so that resident memory
/// reads what is still held.
fn release_freed() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: `malloc_trim` only returns free heap pages to the system.
    unsafe {
        libc::malloc_trim(0);
    }
    if JEMALLOC.load(Ordering::Relaxed) {
        purge_jemalloc();
    }
}

/// Whether the process allocator is jemalloc, as a binary that installs it
/// says through [`allocator_is_jemalloc`]: only then are freed pages in its
/// arenas, and only then is it worth starting jemalloc to purge them.
static JEMALLOC: AtomicBool = AtomicBool::new(false);

/// Tell the fold that this process allocates through jemalloc, so that
/// handing freed memory back before a memory budget is read purges jemalloc's
/// arenas as well as the system heap.
pub fn allocator_is_jemalloc() {
    JEMALLOC.store(true, Ordering::Relaxed);
}

/// Hand jemalloc's freed pages back to the system, as `malloc_trim` does for
/// the system allocator: jemalloc keeps them resident until they decay, and
/// the resident size a memory budget reads next would count them as held.
/// Called through `tikv_jemalloc_sys` rather than through an `extern` block
/// written here. A hand-declared `extern` states a symbol this crate *needs*
/// without depending on the crate that *provides* it, and an rlib nothing
/// references is dropped from the link along with its `-l` directive — so every
/// target that links this library without instantiating
/// `tikv_jemallocator::Jemalloc` failed to find `_rjem_mallctl`, which is every
/// one of this crate's test targets and the `derna` binary unless its optional
/// `jemalloc` feature is on. Going through the sys crate makes the dependency
/// real, and leaves the symbol's `_rjem_` prefix for that crate to spell.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn purge_jemalloc() {
    // `arena.<MALLCTL_ARENAS_ALL>.purge`, where `MALLCTL_ARENAS_ALL` is 4096,
    // purges every arena; it reads and writes nothing.
    // SAFETY: the name is a NUL-terminated string and every other argument
    // is null or zero, as a purge takes them.
    unsafe {
        tikv_jemalloc_sys::mallctl(
            c"arena.4096.purge".as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        );
    }
}

/// [`purge_jemalloc`] where jemalloc is not built. Nothing installs it as the
/// allocator there, so [`JEMALLOC`] is never set and this is unreachable; it
/// exists so [`release_freed`] does not need a `cfg` of its own.
#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn purge_jemalloc() {}

/// What the rest of a gap may store when it is folded again as one range, as
/// a multiple of what a slice too narrow to split stored past its budget,
/// unless [`MotifRun::slice_grow`] gives another; zero keeps the slices.
/// A development build can override it.
fn default_slice_grow() -> usize {
    crate::dev_env::var(crate::dev_env::Setting::MotifSliceGrow)
        .ok()
        .and_then(|b| b.parse().ok())
        .unwrap_or(2)
}

/// Asserts the order a gap's ranges are solved in, as [`solve_gap`] takes
/// the range `(below, top]` with the gap solved up to `solved` and the
/// `left` ranges `(lo, hi]` still on its stack, the lowest last: the range
/// taken starts where the gap is solved, and those left tile
/// `(top, gap_top]` from the lowest up.
///
/// Every point found later then has an energy above every point found so
/// far, so none can dominate one already found, and a gap stopped here
/// holds every front point up to `solved`: its rest is sound. Splitting a
/// range and folding the rest of a gap again in other ranges must both keep
/// the order, so it is checked at every range taken, in every build.
fn assert_solved_from_below(
    left: impl DoubleEndedIterator<Item = (i32, i32)>,
    (below, top): (i32, i32),
    solved: i32,
    gap_top: i32,
) {
    assert!(
        below == solved && below < top,
        "a gap's range ({below}, {top}] was taken with the gap solved up to {solved}"
    );
    let mut from = top;
    for (lo, hi) in left.rev() {
        assert!(
            lo == from && lo < hi,
            "a gap's ranges left do not tile its energies: ({lo}, {hi}] follows {from}"
        );
        from = hi;
    }
    assert_eq!(from, gap_top, "a gap's ranges left end short of its top");
}

/// How a gap's fold is doing against the per-gap time cap: `elapsed` seconds
/// in this task, which started it solved up to `start` and has it solved up to
/// `solved` of `top`.
struct GapPace {
    elapsed: f64,
    start: i32,
    solved: i32,
    top: i32,
}

impl GapPace {
    /// The seconds this task would take for the whole of its range at the
    /// pace of its energy solved so far; infinite before any is.
    fn projected(&self) -> f64 {
        let done = f64::from(self.solved - self.start);
        if done <= 0.0 {
            return f64::INFINITY;
        }
        self.elapsed * f64::from(self.top - self.start) / done
    }

    /// Whether the fold moves on under a cap of `cap` seconds: once some of
    /// its energy is solved in this task, so that every task the cap ends
    /// moves the gap's rest up, when its time or its projected time passes
    /// the cap.
    fn moves_on(&self, cap: f64) -> bool {
        self.solved > self.start
            && self.solved < self.top
            && (self.elapsed > cap || self.projected() > cap)
    }
}

/// Reports a gap whose turn the per-gap time cap ended inside a fold, after
/// `elapsed` seconds against `cap` on the given visit, with `solved`
/// hundredths of a kcal/mol solved in the turn.
fn gap_capped(
    lambda: f64,
    (a, b): (MotifVertexPoint, MotifVertexPoint),
    elapsed: f64,
    (cap, visit): walk::GapCap,
    solved: i32,
) {
    let done = if solved > 0 {
        format!(
            "{:.2} kcal/mol solved this visit",
            f64::from(solved) / 100.0
        )
    } else {
        "nothing solved this visit".to_owned()
    };
    eprintln!(
        "  gap lambda {lambda:.4} ({}, {}] stopped mid-slice after {elapsed:.2}s against a cap of {cap}s (visit {visit}), {done}",
        a.mfe.0, b.mfe.0,
    );
}

/// The front points in the box `a`–`b`, by the fold under `cut`.
///
/// A gap whose fold outgrows the gap budget ([`MotifRun::gap_budget`]) is
/// solved in energy slices, the lower half before the upper, halving again as
/// needed. A front point in a slice must beat the best adaptation found below
/// it, so each slice's fold runs under its own energy ceiling, that adaptation
/// as its floor, and the tie-weight ceiling through the two — each tighter than
/// the whole gap's. Near the adaptation end those ceilings barely bind (the tie
/// weight is almost all adaptation), and a slice stores nearly as much as the
/// range it was cut from. So a slice that gives up is compared with that
/// range's fold at the last span both filled: if it stored more than
/// [`slice_shrink`] of what that fold had, halving again would not pay, and the
/// slice is folded again with twice the budget.
///
/// That comparison reads early spans, and can approve halvings that save
/// nothing by the last span: at 1,000 residues a gap was halved into slices of
/// 30 that each stored 71-77 million points against a budget of 2 million,
/// at 290-470 s each. A slice too narrow to split that stores past its budget
/// shows the budget is far below what any slice of the gap stores, so the rest
/// of the gap is folded again as one range, with [`MotifRun::slice_grow`] times that
/// slice's storage as its budget. Any slicing returns the same points; the
/// budget decides only the cost.
///
/// The ranges are solved from the lowest up ([`assert_solved_from_below`]),
/// so a gap stopped part way, between two folds or inside one, is solved up
/// to the bottom of the range it stopped in: that energy, the best
/// adaptation found up to it and the points found are returned as the gap's
/// rest, and a `rest` given continues from it. A stop inside a fold loses
/// that fold, so a gap whose first range outlasts the budget keeps nothing.
///
/// Under a per-gap time cap ([`Stop::gap_secs`]) the fold also stops between
/// two slices once the cap or the time projected for the whole gap passes it
/// ([`GapPace::moves_on`]), and returns its rest the same way, marked as
/// moved on. Unless [`gap_midslice`] says otherwise the cap also stops a fold in
/// flight, the give-up ladder included and before any slice has solved: that
/// fold is lost, as under a deadline, and the rest is where the turn left it,
/// so a gap whose first slice outlasts the cap moves on with the rest it came
/// with. [`gap_midslice_secs`] (3600) lets a fold in
/// flight run that long when it is longer than the cap, doubled per move as the cap is ([`midslice_floor`]).
/// A gap that keeps moving on gets a doubled
/// cap each turn ([`gap_double`]), so the time its stopped
/// folds waste stays within about twice what the gap costs.
fn solve_gap(
    params: &Params,
    region: &PairConstraint,
    lattice: &Lattice,
    (a, b, rest): (MotifVertexPoint, MotifVertexPoint, Option<(Energy, f64)>),
    (cut, energy, zero): (&Cut, (&Pass, &Floors), &[ZeroChord<'_>]),
    cut_s: f64,
    (stop, slices, threads): (Stop<'_>, SliceBudget, usize),
) -> Result<GapFront, GapRest> {
    let lambda = cut.lambda;
    let started = std::time::Instant::now();
    let stop = Stop {
        gap_deadline: stop
            .gap_secs
            .filter(|_| gap_midslice())
            .map(|(cap, visit)| {
                gap_midslice_secs().map_or(cap, |least| cap.max(midslice_floor(least, visit)))
            })
            .and_then(|secs| std::time::Duration::try_from_secs_f64(secs).ok())
            .and_then(|cap| started.checked_add(cap)),
        ..stop
    };
    let mut points: Vec<(Energy, f64, String, String)> = Vec::new();
    let mut best = rest.map_or(a.cai, |r| r.1);
    // The adaptation the slices prune on: `best`, or, held, where the gap
    // started, to price what the rising floor saves.
    let held = gap_floor_held().then_some(best);
    // Energy ranges `(below, top]` still to solve, the lowest on top, each
    // with its budget and what the fold of the range it was cut from stored
    // by the end of each span.
    let shrink = slice_shrink();
    let start = rest.map_or(a.mfe.0 - 1, |r| r.0 .0);
    let mut pending = Vec::new();
    if slices.open_in_room && slices.max_rss_kb.is_some() {
        // The room, cut into ranges as wide as the points per dekacalorie
        // the last solved fold stored say fit in four fifths of it. A
        // narrower fold stores more per dekacalorie, so the ranges err
        // narrow.
        // Until a cut fold has fitted, a gap opens at the gap budget, whose
        // give-up is cheap and measures what halving keeps; a fold sized to
        // the whole room gave up after 107 s at PRT-04[:500].
        let fit_known =
            slices.fit.is_some() || GAP_FIT.load(std::sync::atomic::Ordering::Relaxed) > 0;
        let room = if gap_fit() && !fit_known {
            slices.points
        } else {
            slices.points.max(slices.room() / slices.workers)
        };
        let per_unit = f64::from_bits(GAP_DENSITY.load(std::sync::atomic::Ordering::Relaxed));
        let fit = slices
            .fit
            .unwrap_or_else(|| GAP_FIT.load(std::sync::atomic::Ordering::Relaxed));
        if fit > 0 && gap_fit() {
            let fit = slices.width.map_or(fit, |w| fit.min(w));
            for (lo, hi) in even_ranges(start, b.mfe.0, fit) {
                pending.push((lo, hi, room, Rc::new(Vec::<usize>::new())));
            }
        } else {
            let width = if per_unit > 0.0 && gap_density() {
                ((0.8 * room as f64 / per_unit) as i32).max(1)
            } else {
                b.mfe.0 - start
            };
            let width = slices.width.map_or(width, |w| width.min(w));
            let mut hi = b.mfe.0;
            while hi > start {
                let lo = (hi - width).max(start);
                pending.push((lo, hi, room, Rc::new(Vec::<usize>::new())));
                hi = lo;
            }
        }
    } else {
        // Even ranges at most `width` wide, as the plain walk slices a gap,
        // or the whole gap.
        let span = b.mfe.0 - start;
        let count = slices
            .width
            .filter(|&w| w > 0 && span > w)
            .map_or(1, |w| ((span + w - 1) / w) as usize);
        for k in (0..count).rev() {
            let at = |k: usize| start + ((i64::from(span) * k as i64) / count as i64) as i32;
            pending.push((
                at(k),
                at(k + 1),
                slices.points,
                Rc::new(Vec::<usize>::new()),
            ));
        }
    }
    let mut folds = 0;
    let mut aware = [0usize; 3];
    let mut split = [SplitTally::default(); 2];
    let gap_split = |split: &[SplitTally; 2], how: &str| {
        if split_census() {
            split_lines(
                &format!("gap lambda {lambda:.4} ({}, {}] {how}", a.mfe.0, b.mfe.0),
                split,
                started.elapsed().as_secs_f64(),
            );
        }
    };
    // The gap's front is among `points` up to this energy: at first, where
    // the gap starts, read from the gap rather than from its stack, so the
    // first range taken is checked against the gap too.
    let mut solved = rest.map_or(a.mfe.0 - 1, |r| r.0 .0);
    // The floors under `L` and `T` depend on the weight, not the ceiling, so
    // every slice reads the same.
    let floors = Floors::new(params, lattice, region, cut);
    // Each pass bracketing the weight, read alone, with its own floors
    // ([`bracket_cuts`]); none at weight 1, which the energy cut holds.
    let bracket: Vec<(&Pass, Floors)> = match cut.far {
        Some((hi, _)) if slices.bracket => [cut.near, hi]
            .into_iter()
            .filter(|p| p.lambda < 1.0)
            .map(|p| {
                let alone = Cut::between(p, None, p.lambda, 0.0);
                (p, Floors::new(params, lattice, region, &alone))
            })
            .collect(),
        _ => Vec::new(),
    };
    // The width and first budget of the gap's last solved range wider than
    // the narrowest split, and whether a range of the gap has given up.
    let mut last: Option<(i32, usize)> = None;
    let mut gave_up = false;
    let step = slice_step();
    // The width of the gap's last solved range, at any width, and the points
    // it stored, when later ranges carry them forward ([`density_carry`]).
    let carry = density_carry() && slices.max_rss_kb.is_none() && slices.grow > 0;
    let mut seen: Option<(i32, usize)> = None;
    // Ranges whose budget the carry raised, and ranges the sizing cut.
    let (mut carried, mut sized_cuts) = (0, 0);
    // The width of the last range that solved, for [`gap_carry`].
    let mut last_solved = 0;
    let mut gap_k = gap_carry();
    // Folds that gave up, kept for the ranges inside them, the most recent
    // last ([`gap_reuse`]).
    let mut given_up: Vec<GaveUp> = Vec::new();
    let mut adopted = (0, 0);
    while let Some((below, mut top, mut budget, parent)) = pending.pop() {
        if let Some((_, stored)) = seen {
            let raised = budget.max(stored.saturating_mul(slices.grow));
            carried += usize::from(raised > budget);
            budget = raised;
        }
        // A range far wider than the last that solved is first taken in
        // part; the rest goes back on the stack as it was. A gap no range of
        // which gave up is sliced as before.
        let sized = if carry {
            seen.map(|(width, _)| (width, 1.0))
        } else {
            last.map(|(width, solved_budget)| {
                let scale = (budget as f64 / solved_budget.max(1) as f64).min(1.0);
                (width, scale)
            })
        };
        if let Some((width, scale)) = sized.filter(|_| gave_up && step > 0.0) {
            let cap = ((step * scale * f64::from(width)) as i32)
                .max(MIN_SLICE + 1)
                .min(top - below);
            if top - below > cap + MIN_SLICE {
                pending.push((below + cap, top, budget, parent.clone()));
                top = below + cap;
                sized_cuts += 1;
            }
        }
        // A range wider than `carry` times the last that solved is taken in
        // two, the lower part that wide, so a gap does not climb back to its
        // whole rest after each slice and halve down again.
        if let Some(most) = gap_k.map(|k| ((k * f64::from(last_solved)) as i32).max(1)) {
            if last_solved > 0 && top - below > most {
                let cut_at = below + most;
                pending.push((cut_at, top, budget, parent.clone()));
                top = cut_at;
            }
        }
        let first_budget = budget;
        assert_solved_from_below(
            pending.iter().map(|r| (r.0, r.1)),
            (below, top),
            solved,
            b.mfe.0,
        );
        if stop.out_of_memory() || stop.timed_out() || stop.folds_spent() {
            gap_split(&split, "returned early");
            return Err(GapRest {
                from: Energy(below),
                floor: best,
                points,
                moved_on: false,
            });
        }
        if let Some((cap, visit)) = stop.gap_secs {
            let pace = GapPace {
                elapsed: started.elapsed().as_secs_f64(),
                start,
                solved,
                top: b.mfe.0,
            };
            if !pace.moves_on(cap) && stop.gap_due() {
                gap_capped(lambda, (a, b), pace.elapsed, (cap, visit), solved - start);
                return Err(GapRest {
                    from: Energy(below),
                    floor: best,
                    points,
                    moved_on: true,
                });
            }
            if pace.moves_on(cap) {
                eprintln!(
                    "  gap lambda {lambda:.4} ({}, {}] moved on after {:.2}s: projected {:.2}s against a cap of {cap}s (visit {visit}), {:.2} kcal/mol solved, {:.2} left",
                    a.mfe.0,
                    b.mfe.0,
                    pace.elapsed,
                    pace.projected(),
                    f64::from(solved - start) / 100.0,
                    f64::from(b.mfe.0 - solved) / 100.0,
                );
                return Err(GapRest {
                    from: Energy(below),
                    floor: best,
                    points,
                    moved_on: true,
                });
            }
        }
        folds += 1;
        let pruning = held.unwrap_or(best);
        let slice_cut = Cut {
            ceiling: brd(Energy(top), pruning, lambda).max(brd(a.mfe, pruning, lambda)),
            ..*cut
        };
        // No design in the slice has energy above its top: the plain
        // lattice's best completion energy floors every cell's at weight 1.
        let energy_cut = Cut::between(energy.0, None, 1.0, f64::from(top));
        let alone: SmallVec<[Cut<'_>; 2]> = bracket
            .iter()
            .map(|(p, _)| {
                let ceiling =
                    brd(Energy(top), pruning, p.lambda).max(brd(a.mfe, pruning, p.lambda));
                Cut::between(p, None, p.lambda, ceiling)
            })
            .collect();
        let mut cuts: Cuts<'_> = [
            Some((&slice_cut, &floors)),
            Some((&energy_cut, energy.1)),
            None,
            None,
        ];
        for (slot, (c, (_, f))) in cuts[2..].iter_mut().zip(alone.iter().zip(&bracket)) {
            *slot = Some((c, f));
        }
        let gap_box = GapBox { c_min: pruning };
        let slice_started = std::time::Instant::now();
        let fit = gap_fit() && slices.max_rss_kb.is_some();
        let before_kb = if fit {
            release_freed();
            rss_kb()
        } else {
            None
        };
        // Ranges are taken from the lowest up, so a kept fold below this
        // range's top holds no later range either.
        given_up.retain(|g| g.top >= top);
        let from = given_up.iter().rev().find(|g| g.floor <= pruning);
        let mut fold = Fold::run_from(
            params,
            lattice,
            region,
            Some(Energy(top)),
            (cuts, Some(gap_box), zero),
            ((top - below > MIN_SLICE).then_some(budget), stop, threads),
            from.map(|g| &g.tables),
        );
        let reused = from.map(|g| (g.top, g.tables.span, fold.adopted));
        if from.is_some() {
            adopted.0 += 1;
            adopted.1 += fold.adopted;
        }
        let (mut budget, mut parent) = (budget, parent);
        loop {
            // What this fold stored against its parent's fold, at the last
            // span both filled; none for the gap's first fold.
            let kept = fold
                .filled
                .len()
                .min(parent.len())
                .checked_sub(1)
                .map(|k| fold.filled[k] as f64 / parent[k].max(1) as f64);
            if crate::dev_env::var_os(crate::dev_env::Setting::MotifFrontCensus).is_some() {
                let reused = reused.map_or(String::new(), |(from, span, n)| {
                    format!(
                        ", adopted {n} points over {} spans from the fold to {from}",
                        span.saturating_sub(1)
                    )
                });
                eprintln!(
                    "  slice lambda {lambda:.4} ({below}, {top}]: {} after {:.2}s, {} points stored, budget {budget}, spans {}, kept {kept:.3?}{reused}",
                    if fold.aborted { "gave up" } else { "solved" },
                    slice_started.elapsed().as_secs_f64(),
                    fold.stored,
                    fold.filled.len(),
                );
            }
            if split_census() {
                split_lines(
                    &format!(
                        "slice lambda {lambda:.4} ({below}, {top}] {}",
                        if fold.aborted { "gave up" } else { "solved" }
                    ),
                    &fold.split,
                    slice_started.elapsed().as_secs_f64(),
                );
            }
            if fold.stopped {
                for (total, t) in split.iter_mut().zip(&fold.split) {
                    total.add(t);
                }
                gap_split(&split, "stopped");
                if let Some(cap) = stop.gap_secs.filter(|_| fold.capped) {
                    let elapsed = started.elapsed().as_secs_f64();
                    gap_capped(lambda, (a, b), elapsed, cap, solved - start);
                }
                return Err(GapRest {
                    from: Energy(below),
                    floor: best,
                    points,
                    moved_on: fold.capped,
                });
            }
            if !fold.aborted {
                break;
            }
            if kept.is_some_and(|k| k > shrink) {
                // Halving would not pay, so the same range takes twice the
                // budget, continuing this fold rather than folding it again.
                budget = budget.saturating_mul(2);
            } else if let Some(more) = before_kb
                .filter(|_| kept.is_some_and(|k| k > CONTINUE_KEPT))
                .and_then(|before| slices.more_in_room(before, fold.stored))
            {
                // The budget read the room at a bound on a point's bytes;
                // the fold's own bytes per point say more still fit, and
                // halving would store more than continuing.
                budget = fold.stored.saturating_add(more);
            } else {
                break;
            }
            parent = Rc::new(Vec::new());
            folds += 1;
            fold.resume(budget);
        }
        for (total, n) in aware.iter_mut().zip(fold.aware) {
            *total += n;
        }
        for (total, t) in split.iter_mut().zip(&fold.split) {
            total.add(t);
        }
        if fold.aborted {
            gave_up = true;
            if gap_carry_adapt() && last_solved > 0 && top - below > last_solved {
                gap_k = gap_k.map(|_| 1.0);
            }
            let mid = below + (top - below) / 2;
            let filled = Rc::new(std::mem::take(&mut fold.filled));
            pending.push((mid, top, budget, filled.clone()));
            pending.push((below, mid, budget, filled));
            if slices.reuse > 0 {
                if given_up.len() >= slices.reuse {
                    given_up.remove(0);
                }
                given_up.push(GaveUp {
                    top,
                    floor: pruning,
                    tables: fold.into_tables(),
                });
            }
            continue;
        }
        stop.fold_solved();
        let stored = fold.stored;
        if fit {
            let width = top - below;
            // Only a cut fold that ran in the room says what fits in it. A
            // whole gap says only that its own width fits, and one cut to
            // the gap budget only what fits that budget: at PRT-04[:300]
            // under 1.5 GB a first gap of 11 dekacalories solved whole cut
            // every later gap into 2,215 folds against 76.
            if (below, top) != (start, b.mfe.0) && budget > slices.points && budget < usize::MAX {
                GAP_FIT.store(width, std::sync::atomic::Ordering::Relaxed);
            }
        }
        GAP_DENSITY.store(
            (stored as f64 / f64::from((top - below).max(1))).to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
        let outgrown = top - below <= MIN_SLICE && stored > budget;
        let mut tally = RuleTally::default();
        let floor = best;
        let found: Vec<_> = fold
            .recover(&mut tally)
            .into_iter()
            .filter(|&(mfe, cai, _, _)| {
                let beats = cai > floor + 1e-9 || (mfe == a.mfe && cai >= a.cai - 1e-9);
                mfe.0 > below && beats && cai <= b.cai + 1e-9
            })
            .collect();
        assert!(
            found.iter().all(|p| p.0 .0 <= top),
            "a slice of a gap returned a point above its top {top}"
        );
        for p in found {
            best = best.max(p.1);
            points.push(p);
        }
        solved = top;
        if top - below > MIN_SLICE {
            last = Some((top - below, first_budget));
        }
        if carry {
            seen = Some((top - below, stored));
        }
        last_solved = top - below;
        // Never past the room the memory budget leaves: a merged fold that
        // passes it would stop and drop the whole gap. The room is read once
        // this fold's tables are freed and handed back, or it would count
        // them against the next fold.
        let census = crate::dev_env::var_os(crate::dev_env::Setting::MotifFrontCensus).is_some();
        let held_kb = if census { rss_kb() } else { None };
        drop(fold);
        let room = slices.room();
        if census {
            if slices.max_rss_kb.is_none() {
                release_freed();
            }
            if let (Some(held), Some(freed)) = (held_kb, rss_kb()) {
                eprintln!(
                    "  slice memory: {:.2} GB resident with the fold, {:.2} GB once freed, {:.1} bytes per point stored, room {room}",
                    held as f64 / 1048576.0,
                    freed as f64 / 1048576.0,
                    held.saturating_sub(freed) as f64 * 1024.0 / stored.max(1) as f64,
                );
            }
        }
        let grown = stored.saturating_mul(slices.grow).min(room);
        // The ranges left are contiguous from this slice's top to the gap's,
        // the highest at the bottom of the stack.
        if outgrown && slices.grow > 0 {
            if let Some(&(_, gap_top, _, _)) = pending.first() {
                debug_assert_eq!(pending.last().map(|r| r.0), Some(top));
                if let Some(rest) =
                    rest_of_gap((top, gap_top), top - below, stored, (grown, budget), fit)
                {
                    pending.clear();
                    pending.extend(
                        rest.into_iter()
                            .map(|(lo, hi, b)| (lo, hi, b, Rc::new(Vec::new()))),
                    );
                }
            }
        }
    }
    assert_eq!(solved, b.mfe.0, "a gap's ranges ended short of its top");
    gap_split(&split, "solved");
    Ok(GapFront {
        record: MotifGap {
            lambda,
            points: points.len(),
            folds,
            cut_s,
            fold_s: started.elapsed().as_secs_f64(),
            settled_s: 0.0,
            aware_out: aware[0],
            aware_dropped: aware[1],
            aware_unkeyed: aware[2],
            carried,
            sized: sized_cuts,
            adopted: adopted.0,
            adopted_points: adopted.1,
        },
        points,
    })
}

/// A pair of hull points to settle, or a lone vertex to report.
#[derive(Clone, Copy, PartialEq)]
enum Task {
    Pair(MotifVertexPoint, MotifVertexPoint),
    Lone(MotifVertexPoint),
}

/// What settling a task found: a vertex between the pair, or the pair's gap
/// (whether a probe fold was run for it).
enum Settled {
    Vertex(MotifVertexPoint, Probe),
    Gap(GapFront, Option<Probe>),
    /// The budget ran out before the pair's gap was solved.
    Open,
    /// The budget ran out with the pair's gap solved part way.
    Rest(GapRest, Option<Probe>),
}

/// What a probe took: its weight, seconds and folds, whether a ceiling below
/// its pair's own score certified it, and what its folds' floors ruled out.
#[derive(Clone, Copy)]
struct Probe {
    lambda: f64,
    secs: f64,
    folds: usize,
    under_chord: bool,
    counts: BoundCounts,
}

/// Settle one task: for a pair, a probe fold at the pair's tie weight bounded
/// by the chord of the passes around it under the pair's own score (exact,
/// since both ends reach it), and, when the probe finds nothing better, the
/// gap's front fold under the same chord. A pair with a rest left by an
/// earlier walk has had its probe, and its gap continues from that rest.
fn settle(
    (task, rest): (Task, Option<(Energy, f64)>),
    params: &Params,
    (product, lattice, region): (&ProductDfa<'_>, &Lattice, &PairConstraint),
    (grid, chain): (&Grid<'_>, Option<(&ChainCai, f64)>),
    spans: &mut Option<crate::constrained::SpanMemo>,
    exact: &impl Fn(f64) -> Result<MotifVertexPoint, BeamError>,
    (designs, stop, slice_budget, rule): (&Designs, Stop<'_>, SliceBudget, &ProbeRule),
) -> Settled {
    const TOL: f64 = 1e-6;
    let (a, b, lambda) = match task {
        Task::Pair(a, b) => (a, b, tie(a, b)),
        Task::Lone(v) => (v, v, 1.0 - END_EPSILON),
    };
    let started = std::time::Instant::now();
    let corner = brd(b.mfe, a.cai, lambda).max(brd(a.mfe, a.cai, lambda));
    // The pass at weight 1 first: it is never dropped, so taking it after
    // the bracket could only drop a bracketing pass this box is reading.
    let top = grid.energy();
    let energy_floors = grid.energy_floors(lattice);
    let (lo, hi) = grid.bracket(lambda);
    let cut = Cut::between(&lo, hi.as_deref(), lambda, corner).with_aware(chain.map(|(c, _)| c));
    let (zero, floor) = zero_chords(chain, &[hi.as_deref(), Some(&*top)], lambda, cut.optimum());
    let cut_s = started.elapsed().as_secs_f64();
    let gap = |cut: &Cut<'_>, rest| {
        solve_gap(
            params,
            region,
            lattice,
            (a, b, rest),
            (cut, (&*top, energy_floors), &zero),
            cut_s,
            (stop, slice_budget, grid.fold_threads),
        )
    };
    if matches!(task, Task::Lone(_)) {
        return match gap(&cut, None) {
            Ok(front) => Settled::Gap(front, None),
            Err(_) => Settled::Open,
        };
    }
    // A rest's probe found no vertex in an earlier walk.
    if rest.is_some() {
        return match gap(&cut, rest) {
            Ok(front) => Settled::Gap(front, None),
            Err(rest) => Settled::Rest(rest, None),
        };
    }
    let score = brd(a.mfe, a.cai, lambda);
    let probing = std::time::Instant::now();
    // The bounded search the ends run: a ceiling slightly above the floor on the
    // plain optimum, raised until a fold certifies, and never above the
    // pair's own score, the ceiling a single fold would take. A wide box's
    // pair scores far above the hull between them, where a ceiling prunes
    // little.
    let mut cap = score + TOL;
    let (mut margin, learned) = rule.first_margin(floor);
    // Where the doubled margin already reaches the pair's score, a first
    // fold under the narrower ceiling is the cheaper fold only if it
    // certifies; at 1,000 residues it never did, and its fold was spent. A
    // learned margin is expected to certify.
    if rule.skip && !learned && floor + 2.0 * margin >= cap {
        margin = cap - floor;
    }
    let mut folds = 0;
    let mut counts = BoundCounts::default();
    let mut halved = false;
    let census = crate::dev_env::var_os(crate::dev_env::Setting::MotifFrontCensus).is_some();
    let (probe, under_chord) = loop {
        let ceiling = (floor + margin).min(cap);
        folds += 1;
        let fold_started = std::time::Instant::now();
        let (fold, ruled_out) = crate::constrained::fold_bounded(
            product,
            params,
            lambda,
            region.clone(),
            crate::must_pair::MustPairConstraint::inert(),
            grid.fold_threads,
            cut.bound(ceiling, &zero),
            spans,
        );
        counts += ruled_out;
        let found = fold.as_ref().map(|f| brd(f.mfe, f.cai, lambda));
        if census {
            eprintln!(
                "  probe lambda {lambda:.4} fold {folds}: ceiling {:.2} and score {:.2} above the floor, {:.2}s, found {:?}",
                ceiling - floor,
                score - floor,
                fold_started.elapsed().as_secs_f64(),
                found.map(|f| f - floor),
            );
        }
        if found.is_some_and(|f| f <= ceiling + crate::constrained::BOUND_SLACK) || ceiling >= cap {
            if let Some(f) = found {
                rule.saw(f - floor);
            }
            break (fold, ceiling < score + TOL);
        }
        // A design above the ceiling caps the next one, and only caps it.
        if let Some(f) = found {
            cap = cap.min(f);
        }
        margin *= 2.0;
        // The optimum lies between this ceiling and the design found above
        // it. A fold's cost climbs steeply with its ceiling, so one fold
        // halfway there is cheap beside the fold at the design's score.
        if rule.learn && !halved && found.is_some() {
            halved = true;
            margin = margin.min((ceiling + cap) / 2.0 - floor);
        }
    };
    let v = match probe {
        Some(f) => designs.keep(f),
        // Both ends score at the ceiling, so the bounded fold always finds
        // one; an empty result can only be rounding, settled exactly.
        None => exact(lambda).expect("a feasible pair has a fold"),
    };
    let probe = Probe {
        lambda,
        secs: probing.elapsed().as_secs_f64(),
        folds,
        under_chord,
        counts,
    };
    if brd(v.mfe, v.cai, lambda) < score - TOL {
        return Settled::Vertex(v, probe);
    }
    match gap(&cut, None) {
        Ok(front) => Settled::Gap(front, Some(probe)),
        Err(rest) => Settled::Rest(rest, Some(probe)),
    }
}

/// The chords to the motif-free outside at weight 0 a probe at `lambda`
/// prunes on, against each of `passes` above `lambda` (see [`ZeroChord`]),
/// and the floor they put under the probe's motif-free optimum, never below
/// `free`, the chord of the plain optima.
///
/// The motif-free optimum is concave in the weight, exactly
/// `−100 · open_chain` at 0 and at least a pass's plain optimum at its own
/// weight `ν`, so the chord of the two is below it on `[0, ν]`. Near weight
/// 0, where the plain passes count adaptation the motifs forbid, these
/// floors read the motif-free adaptation exactly; a chord whose root does not
/// raise `free` is left out, as it seldom raises any cell's floor.
fn zero_chords<'a>(
    chain: Option<(&'a ChainCai, f64)>,
    passes: &[Option<&'a Pass>],
    lambda: f64,
    free: f64,
) -> (Vec<ZeroChord<'a>>, f64) {
    let Some((chain, open_chain)) = chain else {
        return (Vec::new(), free);
    };
    let mut floor = free;
    let mut zero: Vec<ZeroChord<'a>> = Vec::new();
    for pass in passes.iter().flatten() {
        if pass.lambda <= lambda + 1e-12
            || zero.iter().any(|z| std::ptr::eq(z.upper, &pass.table[..]))
        {
            continue;
        }
        let weight = lambda / pass.lambda;
        let root = weight * pass.optimum - (1.0 - weight) * 100.0 * open_chain;
        if root > free + crate::constrained::BOUND_SLACK {
            floor = floor.max(root);
            zero.push(ZeroChord {
                chain,
                upper: &pass.table,
                weight,
            });
        }
    }
    (zero, floor)
}

/// The adapted end's fold over the lattice of the codons `codons` marks on
/// some most adapted encoding: the same bounded search as an end, at the same
/// weight, with its adaptation in the whole unrestricted lattice's units.
/// `None` when that lattice has no motif-free design.
fn most_adapted_fold(
    params: &Params,
    (aa, usage, five, three): (&[usize], &CodonUsage, &Flank, &Flank),
    (constraints, ac, window): (&RegionConstraints, &Ac, MotifWindow),
    codons: &[Vec<bool>],
    threads: usize,
) -> Option<crate::constrained::ConstrainedFold> {
    let allowed: Vec<Vec<[u8; 3]>> = aa
        .iter()
        .zip(codons)
        .map(|(&residue, on)| {
            derna_codon_dp::codon_table::AMINO_ACIDS[residue]
                .codons
                .iter()
                .zip(on)
                .filter(|&(_, &on)| on)
                .map(|(codon, _)| codon.nuc)
                .collect()
        })
        .collect();
    let graph = dfa::build_composite(
        aa,
        usage,
        Some(&allowed),
        five.chain(),
        three.chain(),
        crate::UtrProfile::inert(),
    )?;
    let region = PairConstraint::new(constraints, &graph);
    let product = ProductDfa::build_scoped(&graph, ac, window);
    if !product.is_feasible() {
        return None;
    }
    let product = product.merge_equivalent();
    let mut report = crate::bounded_motifs::BoundReport::default();
    let fold = crate::bounded_motifs::fold(
        params,
        &graph,
        &product,
        END_EPSILON,
        region,
        crate::must_pair::MustPairConstraint::inert(),
        crate::ScanConfig {
            threads,
            ..crate::ScanConfig::default()
        },
        &mut report,
    )?;
    Some(crate::constrained::ConstrainedFold {
        cai: fold.cai + graph.codon_offset(),
        ..fold
    })
}

/// Whether a gap's slices prune on the adaptation the gap started from
/// rather than on the best its lower slices found: no, unless
/// a development build turns it on. The front is the same either way; held,
/// each slice stores what it would store folded apart from the others.
fn gap_floor_held() -> bool {
    crate::dev_env::var(crate::dev_env::Setting::MotifGapFloorHeld).is_ok_and(|v| v == "1")
}

/// How a walk's probes choose their ceilings: [`MotifRun::probe_skip`],
/// [`MotifRun::probe_learn`], and the widest slack between a probe's floor
/// and its optimum seen so far, as the bits of a non-negative `f64`, whose
/// order the bits keep.
struct ProbeRule {
    skip: bool,
    learn: bool,
    slack: std::sync::atomic::AtomicU64,
}

impl ProbeRule {
    /// The first ceiling's margin above `floor`, and whether it was learned:
    /// [`first_margin`], or, when learning, a tenth more than the widest
    /// slack seen if that is wider.
    /// The floor's slack varies little between weights; at
    /// P0DTC2[:1000] it was 611-736 at weights 0.35-0.55, where the
    /// fraction [`first_margin`] takes gave 272-482.
    fn first_margin(&self, floor: f64) -> (f64, bool) {
        let plain = first_margin(floor);
        let seen = f64::from_bits(self.slack.load(std::sync::atomic::Ordering::Relaxed));
        if self.learn && 1.1 * seen > plain {
            (1.1 * seen, true)
        } else {
            (plain, false)
        }
    }

    /// Records a certified probe's slack between its floor and its optimum.
    fn saw(&self, slack: f64) {
        if slack.is_finite() && slack > 0.0 {
            self.slack
                .fetch_max(slack.to_bits(), std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Whether a probe's ceilings learn from the walk's earlier probes, unless
/// [`MotifRun::probe_learn`] says: yes, unless a development build turns it off.
/// The probe's answer is the same either way; only the folds it takes differ.
fn default_probe_learn() -> bool {
    env_switch(crate::dev_env::Setting::MotifProbeLearn, true)
}

/// Whether a probe whose doubled first margin reaches its pair's score folds
/// at that score at once, unless [`MotifRun::probe_skip`] says: yes, unless
/// a development build turns it off. The probe's answer is the same either way;
/// only the folds it takes differ. At P0DTC2[:1000] the fold it skips was
/// 25-29 s of probes of 68-82 s.
fn default_probe_skip() -> bool {
    crate::dev_env::var(crate::dev_env::Setting::MotifProbeSkip).map_or(true, |v| v != "0")
}

/// The first ceiling's margin above a floor on the plain optimum: a small
/// fraction of its magnitude, so the first fold prunes hard and a clean
/// optimum certifies at once.
fn first_margin(floor: f64) -> f64 {
    if floor.is_finite() {
        (floor.abs() * 0.005).max(1.0)
    } else {
        f64::INFINITY
    }
}

/// Each hull vertex's design, by energy: a vertex is a front point, so a
/// walk stopped short still returns it, beside a gap no fold finished.
#[derive(Default)]
struct Designs(std::sync::Mutex<FxHashMap<i32, (f64, String, String)>>);

impl Designs {
    /// Record a scalar fold's design and return its point.
    fn keep(&self, f: crate::constrained::ConstrainedFold) -> MotifVertexPoint {
        let v = MotifVertexPoint {
            mfe: f.mfe,
            cai: f.cai,
        };
        self.0
            .lock()
            .expect("the designs' lock")
            .entry(f.mfe.0)
            .or_insert((f.cai, f.sequence, f.structure));
        v
    }
}

#[cfg(test)]
mod tests {
    /// Merging gathered batches by energy buckets keeps exactly the points,
    /// in the order and with the provenance, that sorting them keeps: over
    /// runs of batches into one front, with repeated energies, equal
    /// adaptations, adaptations within `CAI_EPS` of each other, signed zeros
    /// and an infinite one, and batch sizes and energy spans either side of
    /// where the buckets give way to the sort.
    #[test]
    fn merging_by_energy_keeps_what_the_sort_keeps() {
        use super::{Point, Prov, BY_ENERGY_MIN};
        use rna_mfe::loops::Energy;
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move |m: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % m
        };
        let cais = [
            0.0,
            -0.0,
            -1e-10,
            -0.5,
            -0.5 + 5e-10,
            -0.5 - 2e-9,
            -1.0,
            -1.25,
            -2.0,
            f64::NEG_INFINITY,
        ];
        let key = |p: &Point<Prov>| {
            let Prov::Multi { k } = p.back else {
                unreachable!()
            };
            (p.mfe.0, p.cai.to_bits(), k)
        };
        let (mut by_energy, mut by_sort) = (0, 0);
        let mut k = 0_u32;
        for round in 0..200 {
            let mut sorted_acc = Vec::new();
            let mut bucket_acc = Vec::new();
            for batch in 0..4 {
                let n = match (round + batch) % 4 {
                    0 => BY_ENERGY_MIN - 1,
                    1 => BY_ENERGY_MIN,
                    _ => 2 + next(300) as usize,
                };
                // A span of exactly 4n or 4n + 1, or a random one.
                let spread = match round % 5 {
                    0 => 4 * n as i32,
                    1 => 4 * n as i32 + 1,
                    2 => 1 + next(2_000) as i32,
                    _ => 1 + next(60) as i32,
                };
                let offset = 40 * batch - spread / 2;
                let mut gathered: Vec<Point<Prov>> = (0..n)
                    .map(|i| {
                        let e = match i {
                            0 => 0,
                            1 => spread - 1,
                            _ => next(spread as u64) as i32,
                        };
                        let cai = cais[next(cais.len() as u64) as usize]
                            - f64::from(next(4) as u32) * 0.125;
                        k += 1;
                        Point::new(Energy(e + offset), cai, Prov::Multi { k })
                    })
                    .collect();
                super::merge_sorted(&mut sorted_acc, &mut gathered.clone());
                if super::merge_by_energy(&mut Vec::new(), &gathered) {
                    by_energy += 1;
                } else {
                    by_sort += 1;
                }
                super::merge_gathered(&mut bucket_acc, &mut gathered);
                assert!(gathered.is_empty());
                assert_eq!(
                    bucket_acc.iter().map(key).collect::<Vec<_>>(),
                    sorted_acc.iter().map(key).collect::<Vec<_>>(),
                    "round {round} batch {batch}"
                );
            }
        }
        assert!(
            by_energy > 200 && by_sort > 100,
            "{by_energy} batches by energy, {by_sort} by the sort"
        );
    }

    /// The rest of a gap is one range where the room holds what the outgrown
    /// slice stored, and slices as wide as that one where it does not.
    #[test]
    fn the_rest_of_a_gap_is_one_range_or_slices_as_wide_as_the_one_that_fitted() {
        assert_eq!(
            super::rest_of_gap((100, 400), 30, 1_000, (2_000, 500), false),
            Some(vec![(100, 400, 2_000)])
        );
        assert_eq!(
            super::rest_of_gap((100, 400), 30, 1_000, (1_000, 1_000), false),
            None
        );
        let rest = super::rest_of_gap((100, 170), 30, 1_000, (600, 500), false).unwrap();
        assert_eq!(
            rest,
            vec![(140, 170, 500), (110, 140, 500), (100, 110, 500)]
        );
        // Even: 70 over slices of at most 33 is three of 23 or 24, no budget.
        let rest = super::rest_of_gap((100, 170), 30, 1_000, (600, 500), true).unwrap();
        assert_eq!(
            rest,
            vec![
                (146, 170, usize::MAX),
                (123, 146, usize::MAX),
                (100, 123, usize::MAX)
            ]
        );
        // A remainder within a tenth of the width joins the slices.
        let rest = super::rest_of_gap((100, 201), 50, 1_000, (600, 500), true).unwrap();
        assert_eq!(rest, vec![(150, 201, usize::MAX), (100, 150, usize::MAX)]);
    }

    /// Even ranges cover the energies once, lowest last, none wider than asked.
    #[test]
    fn even_ranges_cover_the_energies_in_equal_widths() {
        assert_eq!(super::even_ranges(0, 10, 4), vec![(6, 10), (3, 6), (0, 3)]);
        assert_eq!(super::even_ranges(0, 10, 10), vec![(0, 10)]);
        assert_eq!(super::even_ranges(5, 5, 3), vec![]);
        for (lo, hi, w) in [(-186471, -185730, 46), (3, 4, 50), (0, 1000, 7)] {
            let r = super::even_ranges(lo, hi, w);
            assert_eq!(r.last().unwrap().0, lo);
            assert_eq!(r[0].1, hi);
            assert!(r.windows(2).all(|p| p[0].0 == p[1].1));
            assert!(r.iter().all(|&(a, b)| b > a && b - a <= w));
        }
    }

    use super::*;

    /// The lattice of `MKVLA` behind a 5′ UTR, with `GCC` forbidden over the
    /// coding sequence.
    fn small() -> (Params, PairConstraint, Lattice) {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let spec = MotifFrontSpec {
            utr5: "GGGAAACCC",
            utr3: "",
            constraints: RegionConstraints::default(),
            motifs: &[("GCC", MotifWindow::overlapping(9, 24))],
            pins: None,
        };
        let aa = crate::residues("MKVLA").unwrap();
        let five = Flank::parse(spec.utr5).unwrap();
        let three = Flank::parse(spec.utr3).unwrap();
        let graph = spec_graph(&aa, &usage, &spec, &five, &three).unwrap();
        let region = PairConstraint::new(&spec.constraints, &graph);
        let (ac, scope) = Ac::for_windows(spec.motifs).unwrap();
        let product = ProductDfa::build_scoped(&graph, &ac, scope).merge_equivalent();
        (params, region, Lattice::new(&product))
    }

    /// A pass kept on disk reads back bit for bit, and only at its weight.
    #[test]
    fn a_kept_pass_reads_back_bit_for_bit() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let aa = crate::residues("MKVLAW").unwrap();
        let five = crate::parse_utr("GGGAAA").unwrap();
        let three = crate::parse_utr("AAAGG").unwrap();
        let allowed: Vec<Vec<[u8; 3]>> = aa
            .iter()
            .map(|&a| {
                derna_codon_dp::codon_table::AMINO_ACIDS[a]
                    .codons
                    .iter()
                    .map(|c| c.nuc)
                    .collect()
            })
            .collect();
        let graph = dfa::build_restricted_with_utr(&aa, &usage, &allowed, &five, &three).unwrap();
        let constraints = RegionConstraints::symmetric_isolation(crate::Enforcement::Forbid);
        let region = PairConstraint::new(&constraints, &graph);
        let pass = Pass::new(&params, &graph, &region, 0.37, 1);
        let dir = std::env::temp_dir().join(format!("derna-pass-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("p.pass");
        pass.write(&file).unwrap();
        let back = Pass::read(&file, NodeLayout::PLAIN, 0.37).unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let maps = |v: &[PairColumn]| {
            v.iter()
                .map(|m| {
                    let mut e: Vec<_> = m.packed().collect();
                    e.sort();
                    e
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(pass.optimum.to_bits(), back.optimum.to_bits());
        assert!(pass.table.iter().any(|m| !m.is_empty()));
        assert_eq!(maps(&pass.table), maps(&back.table));
        assert_eq!(
            pass.out_m.iter().map(|r| bits(r)).collect::<Vec<_>>(),
            back.out_m.iter().map(|r| bits(r)).collect::<Vec<_>>()
        );
        assert_eq!(
            pass.out_m2.iter().map(|r| bits(r)).collect::<Vec<_>>(),
            back.out_m2.iter().map(|r| bits(r)).collect::<Vec<_>>()
        );
        assert_eq!(bits(&pass.out_c), bits(&back.out_c));
        assert!(Pass::read(&file, NodeLayout::PLAIN, 0.38).is_err());
        assert_ne!(
            fingerprint(&params, &graph, &region),
            fingerprint(
                &params,
                &graph,
                &PairConstraint::new(&RegionConstraints::default(), &graph)
            )
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A fold its budget gave up, continued under twice the budget until it
    /// finishes, stores what the unbudgeted fold stores and ends on its front,
    /// on one thread and on three.
    #[test]
    fn a_resumed_fold_is_the_unbudgeted_fold() {
        let (params, region, lattice) = small();
        let run = |budget, threads| {
            Fold::run(
                &params,
                &lattice,
                &region,
                None,
                ([None; 4], None, &[]),
                (budget, Stop::default(), threads),
            )
        };
        let whole = run(None, 1);
        let ends = |f: &Fold<'_>| -> Vec<(i32, u64)> {
            f.end
                .points()
                .iter()
                .map(|p| (p.mfe.0, p.cai.to_bits()))
                .collect()
        };
        for threads in [1, 3] {
            let mut fold = run(Some(20), threads);
            let mut resumed = 0;
            while fold.aborted {
                resumed += 1;
                let budget = fold.budget * 2;
                fold.resume(budget);
            }
            assert!(resumed >= 2, "{resumed}");
            assert_eq!(fold.stored, whole.stored);
            assert_eq!(fold.filled, whole.filled);
            assert_eq!(ends(&fold), ends(&whole));
        }
    }

    /// A fold whose memory budget reads spent at the start of its third span
    /// stops there with two spans filled; one that never reads spent fills
    /// every span. The budget is read before each span, so the stop does not
    /// wait for a task to end. A gap's own turn that is already over stops
    /// the fold at its first cell, marked as the gap's cap; one that is not
    /// over leaves it whole, and the walk's deadline passing with it is the
    /// walk's stop, not the gap's.
    #[test]
    fn a_fold_stops_where_its_memory_budget_is_spent() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let spec = MotifFrontSpec {
            utr5: "GGGAAACCC",
            utr3: "",
            constraints: RegionConstraints::default(),
            motifs: &[("GCC", MotifWindow::overlapping(9, 24))],
            pins: None,
        };
        let aa = crate::residues("MKVLA").unwrap();
        let five = Flank::parse(spec.utr5).unwrap();
        let three = Flank::parse(spec.utr3).unwrap();
        let graph = spec_graph(&aa, &usage, &spec, &five, &three).unwrap();
        let region = PairConstraint::new(&spec.constraints, &graph);
        let (ac, scope) = Ac::for_windows(spec.motifs).unwrap();
        let product = ProductDfa::build_scoped(&graph, &ac, scope);
        assert!(product.is_feasible());
        let product = product.merge_equivalent();
        let lattice = Lattice::new(&product);
        let reads = AtomicUsize::new(0);
        let third = || reads.fetch_add(1, Ordering::Relaxed) + 1 >= 3;
        let never = || false;
        let fold = |memory: Stop<'_>| {
            let (params, lattice, region) = (&params, &lattice, &region);
            let run = Fold::run(
                params,
                lattice,
                region,
                None,
                ([None; 4], None, &[]),
                (None, memory, 1),
            );
            (
                run.aborted,
                run.stopped && !run.capped,
                run.filled.len(),
                run.end.is_empty(),
                run.capped,
            )
        };
        let (aborted, stopped, filled, _, _) = fold(Stop {
            deadline: None,
            memory: Some(&third),
            folds: None,
            gap_secs: None,
            gap_deadline: None,
        });
        assert!(aborted && stopped);
        assert_eq!(filled, 2);
        let (aborted, stopped, filled, empty, _) = fold(Stop {
            deadline: None,
            memory: Some(&never),
            folds: None,
            gap_secs: None,
            gap_deadline: None,
        });
        assert!(!aborted && !stopped && !empty);
        assert_eq!(filled as u32 + 1, lattice.seq_len);
        let now = std::time::Instant::now();
        let over = Stop {
            gap_deadline: Some(now),
            ..Stop::default()
        };
        let (aborted, stopped, filled, _, capped) = fold(over);
        assert!(aborted && !stopped && capped);
        assert_eq!(filled, 0);
        let later = Stop {
            gap_deadline: now.checked_add(std::time::Duration::from_secs(3600)),
            ..Stop::default()
        };
        let (aborted, _, _, empty, capped) = fold(later);
        assert!(!aborted && !capped && !empty);
        let passed = || true;
        let (aborted, stopped, _, _, capped) = fold(Stop {
            deadline: Some(&passed),
            ..over
        });
        assert!(aborted && stopped && !capped);
    }

    /// The order check passes a stack that tiles a gap from where it is
    /// solved, as a split or a re-slicing leaves one, and fails a stack with
    /// a hole, one with its lowest range at the bottom, a range taken that
    /// starts above where the gap is solved, and a stack that ends short of
    /// the gap.
    #[test]
    fn the_order_check_passes_a_tiling_and_fails_a_hole_or_a_skip() {
        let holds = |left: &[(i32, i32)], taken: (i32, i32), solved: i32| {
            std::panic::catch_unwind(|| {
                assert_solved_from_below(left.iter().copied(), taken, solved, 400);
            })
            .is_ok()
        };
        assert!(holds(&[], (100, 400), 100));
        assert!(holds(&[(300, 400), (200, 300)], (100, 200), 100));
        assert!(!holds(&[(300, 400), (250, 300)], (100, 200), 100));
        assert!(!holds(&[(200, 300), (300, 400)], (100, 200), 100));
        assert!(!holds(&[(300, 400), (200, 300)], (150, 200), 100));
        assert!(!holds(&[(300, 380), (200, 300)], (100, 200), 100));
    }

    /// A gap stopped by its deadline or its memory budget, before its first
    /// slice is solved, between two slices or inside one, keeps exactly the
    /// gap's front up to its rest and the best adaptation there, and the gap
    /// resumed from that rest returns exactly the front above it.
    ///
    /// The clock and the memory reading are counters that read `true` from
    /// their `k`th read on, so the stops land at every kind of place one can,
    /// the same at any load. The gap is the whole front between the walk's two
    /// ends, under a budget of 20 points, so its rests fall in at least four
    /// distinct solved slices: at least three splits, each checked by
    /// [`assert_solved_from_below`].
    #[test]
    fn a_gap_stopped_by_time_or_memory_keeps_its_front_up_to_its_rest() {
        fn stop(memory: bool, read: &(dyn Fn() -> bool + Sync)) -> Stop<'_> {
            if memory {
                Stop {
                    memory: Some(read),
                    ..Stop::default()
                }
            } else {
                Stop {
                    deadline: Some(read),
                    ..Stop::default()
                }
            }
        }
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let spec = MotifFrontSpec {
            utr5: "GGGAAACCC",
            utr3: "",
            constraints: RegionConstraints::default(),
            motifs: &[("GCC", MotifWindow::overlapping(9, 24))],
            pins: None,
        };
        let protein = "MKVLAW";
        let aa = crate::residues(protein).unwrap();
        let five = Flank::parse(spec.utr5).unwrap();
        let three = Flank::parse(spec.utr3).unwrap();
        let graph = spec_graph(&aa, &usage, &spec, &five, &three).unwrap();
        let region = PairConstraint::new(&spec.constraints, &graph);
        let (ac, scope) = Ac::for_windows(spec.motifs).unwrap();
        let product = ProductDfa::build_scoped(&graph, &ac, scope).merge_equivalent();
        let lattice = Lattice::new(&product);
        let walk = motif_front_walk(&params, protein, &usage, &spec, 1).unwrap();
        let offset = graph.codon_offset();
        let unit = |v: &MotifVertexPoint| MotifVertexPoint {
            mfe: v.mfe,
            cai: v.cai - offset,
        };
        let a = unit(walk.vertices.first().unwrap());
        let b = unit(walk.vertices.last().unwrap());
        let lambda = tie(a, b);
        let grid = Grid::new(
            motif_pass_grid(derna_codon_dp::cai::sense_codons(&aa)).weights(),
            None,
            (&params, &graph, &region),
            1,
            None,
        );
        let top = grid.energy();
        let (lo, hi) = grid.bracket(lambda);
        let corner = brd(b.mfe, a.cai, lambda).max(brd(a.mfe, a.cai, lambda));
        let cut = Cut::between(&lo, hi.as_deref(), lambda, corner);
        let slices = SliceBudget {
            points: 20,
            grow: 0,
            max_rss_kb: None,
            workers: 1,
            open_in_room: false,
            fit: None,
            bracket: true,
            width: None,
            reuse: 0,
        };
        let gap = |stop: Stop<'_>, rest| {
            solve_gap(
                &params,
                &region,
                &lattice,
                (a, b, rest),
                (&cut, (&*top, grid.energy_floors(&lattice)), &[]),
                0.0,
                (stop, slices, 1),
            )
        };
        let Ok(whole) = gap(Stop::default(), None) else {
            panic!("a gap nothing stops is solved");
        };
        let near = |p: &[(Energy, f64, String, String)], q: &[(Energy, f64, String, String)]| {
            p.len() == q.len()
                && p.iter()
                    .zip(q)
                    .all(|(p, q)| p.0 == q.0 && (p.1 - q.1).abs() < 1e-9)
        };
        let front: Vec<_> = walk
            .front
            .iter()
            .map(|p| {
                let mfe = Energy((p.mfe_kcal_mol * 100.0).round() as i32);
                (mfe, p.cai_log - offset, String::new(), String::new())
            })
            .collect();
        assert!(
            near(&whole.points, &front),
            "the gap is not the walk's front"
        );
        for memory in [false, true] {
            let reads = AtomicUsize::new(0);
            let count = || {
                reads.fetch_add(1, Ordering::Relaxed);
                false
            };
            assert!(gap(stop(memory, &count), None).is_ok());
            let total = reads.load(Ordering::Relaxed);
            let (mut unsolved, mut solved_in) = (0, std::collections::BTreeSet::new());
            for k in (1..=total).step_by((total / 300).max(1)) {
                let reads = AtomicUsize::new(0);
                let spent = || reads.fetch_add(1, Ordering::Relaxed) + 1 >= k;
                let Err(rest) = gap(stop(memory, &spent), None) else {
                    panic!("a gap stopped at read {k} of {total} was solved");
                };
                let n = whole.points.partition_point(|p| p.0 <= rest.from);
                assert_eq!(rest.points, whole.points[..n], "kept at read {k}");
                let floor = rest.points.iter().map(|p| p.1).fold(a.cai, f64::max);
                assert_eq!(rest.floor.to_bits(), floor.to_bits(), "floor at read {k}");
                let Ok(more) = gap(Stop::default(), Some((rest.from, rest.floor))) else {
                    panic!("a gap resumed with nothing to stop it is solved");
                };
                assert!(near(&more.points, &whole.points[n..]), "resumed at {k}");
                if rest.from.0 < a.mfe.0 {
                    unsolved += 1;
                } else {
                    solved_in.insert(rest.from.0);
                }
            }
            assert!(unsolved > 0, "no stop before the first slice was solved");
            assert!(
                solved_in.len() >= 4,
                "rests in too few slices: {solved_in:?}, memory {memory}"
            );
        }
    }

    /// A fold over one range of a gap under its tie-weight cut, its energy
    /// cut and its adaptation floor, as [`solve_gap`] folds it, started from
    /// `from` if given.
    fn fold_range<'a>(
        (params, lattice, region): (&'a Params, &'a Lattice, &'a PairConstraint),
        cuts: [(&'a Cut<'a>, &'a Floors); 2],
        (top, floor): (i32, f64),
        budget: Option<usize>,
        from: Option<&Tables>,
    ) -> Fold<'a> {
        Fold::run_from(
            params,
            lattice,
            region,
            Some(Energy(top)),
            (
                [Some(cuts[0]), Some(cuts[1]), None, None],
                Some(GapBox { c_min: floor }),
                &[],
            ),
            (budget, Stop::default(), 1),
            from,
        )
    }

    /// Cells by table and key, as their points' energies and adaptation bits.
    type CellsBelow = std::collections::BTreeMap<(u8, (u32, u32)), Vec<(i32, u64)>>;

    /// Every non-empty cell a fold stored over spans shorter than `span`, by
    /// table and key, as its points' energies and adaptation bits.
    fn cells_below(fold: &Fold<'_>, span: u32) -> CellsBelow {
        let l = fold.l;
        let values = |front: &Front<Prov>| -> Vec<(i32, u64)> {
            front
                .points()
                .iter()
                .map(|p| (p.mfe.0, p.cai.to_bits()))
                .collect()
        };
        let mut out = std::collections::BTreeMap::new();
        for (&(a, b), front) in &fold.p {
            let s = l.pos[l.edge(b).to as usize] - l.pos[l.edge(a).from as usize];
            if s < span && !front.is_empty() {
                out.insert((0, (a, b)), values(front));
            }
        }
        let tables = [
            (1, &fold.x),
            (2, &fold.m),
            (3, &fold.m2),
            (4, &fold.t),
            (5, &fold.lead),
        ];
        for (table, cells) in tables {
            for (&(u, v), front) in cells {
                if l.pos[v as usize] - l.pos[u as usize] < span && !front.is_empty() {
                    out.insert((table, (u, v)), values(front));
                }
            }
        }
        out
    }

    /// A range inside one whose fold gave up, started from that fold's
    /// tables held to its own box, stores over the adopted spans exactly the
    /// cells a fold from nothing stores there and ends on the same designs;
    /// and a gap solved with its gave-up folds kept returns the points it
    /// returns without, with some of its folds started from one.
    #[test]
    fn a_range_inside_a_fold_that_gave_up_starts_from_its_tables() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let spec = MotifFrontSpec {
            utr5: "GGGAAACCC",
            utr3: "",
            constraints: RegionConstraints::default(),
            motifs: &[("GCC", MotifWindow::overlapping(9, 24))],
            pins: None,
        };
        let protein = "MKVLAW";
        let aa = crate::residues(protein).unwrap();
        let five = Flank::parse(spec.utr5).unwrap();
        let three = Flank::parse(spec.utr3).unwrap();
        let graph = spec_graph(&aa, &usage, &spec, &five, &three).unwrap();
        let region = PairConstraint::new(&spec.constraints, &graph);
        let (ac, scope) = Ac::for_windows(spec.motifs).unwrap();
        let product = ProductDfa::build_scoped(&graph, &ac, scope).merge_equivalent();
        let lattice = Lattice::new(&product);
        let walk = motif_front_walk(&params, protein, &usage, &spec, 1).unwrap();
        let offset = graph.codon_offset();
        let unit = |v: &MotifVertexPoint| MotifVertexPoint {
            mfe: v.mfe,
            cai: v.cai - offset,
        };
        let a = unit(walk.vertices.first().unwrap());
        let b = unit(walk.vertices.last().unwrap());
        let lambda = tie(a, b);
        let grid = Grid::new(
            motif_pass_grid(derna_codon_dp::cai::sense_codons(&aa)).weights(),
            None,
            (&params, &graph, &region),
            1,
            None,
        );
        let energy_pass = grid.energy();
        let (lo, hi) = grid.bracket(lambda);
        let corner = brd(b.mfe, a.cai, lambda).max(brd(a.mfe, a.cai, lambda));
        let cut = Cut::between(&lo, hi.as_deref(), lambda, corner);

        // The whole gap, with and without its gave-up folds kept.
        let gap = |reuse| {
            let slices = SliceBudget {
                points: 20,
                grow: 0,
                max_rss_kb: None,
                workers: 1,
                open_in_room: false,
                fit: None,
                bracket: true,
                width: None,
                reuse,
            };
            let Ok(gap) = solve_gap(
                &params,
                &region,
                &lattice,
                (a, b, None),
                (&cut, (&*energy_pass, grid.energy_floors(&lattice)), &[]),
                0.0,
                (Stop::default(), slices, 1),
            ) else {
                panic!("a gap nothing stops is solved");
            };
            gap
        };
        let values = |g: &GapFront| -> Vec<(i32, u64)> {
            g.points.iter().map(|p| (p.0 .0, p.1.to_bits())).collect()
        };
        let scratch = gap(0);
        assert_eq!(scratch.record.adopted, 0);
        assert!(scratch.record.folds > 3, "{}", scratch.record.folds);
        for reuse in [1, 3] {
            let kept = gap(reuse);
            assert!(
                kept.record.adopted > 0 && kept.record.adopted_points > 0,
                "no fold started from one that gave up, reuse {reuse}"
            );
            assert_eq!(values(&kept), values(&scratch), "reuse {reuse}");
        }

        // One range by hand: the whole gap gives up halfway through its
        // spans, and its lower half starts from its tables, under its own
        // floor and under a raised one.
        let floors = Floors::new(&params, &lattice, &region, &cut);
        let energy_floors = grid.energy_floors(&lattice);
        let slice = |top: i32, floor: f64| Cut {
            ceiling: brd(Energy(top), floor, lambda).max(brd(a.mfe, floor, lambda)),
            ..cut
        };
        let energy_cut = |top: i32| Cut::between(&energy_pass, None, 1.0, f64::from(top));
        let at = (&params, &lattice, &region);
        let (wide_top, wide_floor) = (b.mfe.0, a.cai);
        let (wide_cut, wide_energy) = (slice(wide_top, wide_floor), energy_cut(wide_top));
        let wide = |budget| {
            fold_range(
                at,
                [(&wide_cut, &floors), (&wide_energy, energy_floors)],
                (wide_top, wide_floor),
                budget,
                None,
            )
        };
        let whole = wide(None);
        let budget = whole.filled[whole.filled.len() / 2];
        let gave_up = wide(Some(budget));
        assert!(gave_up.aborted && !gave_up.stopped);
        let below: usize = cells_below(&gave_up, gave_up.next.0)
            .values()
            .map(Vec::len)
            .sum();
        let tables = gave_up.into_tables();
        assert!(tables.span > 3, "gave up at span {}", tables.span);
        let mid = a.mfe.0 + (b.mfe.0 - a.mfe.0) / 2;
        let raised = scratch
            .points
            .iter()
            .filter(|p| p.0 .0 <= a.mfe.0 + (mid - a.mfe.0) / 2)
            .map(|p| p.1)
            .fold(a.cai, f64::max);
        assert!(raised > a.cai);
        let mut dropped = 0;
        for floor in [a.cai, raised] {
            let (narrow_cut, narrow_energy) = (slice(mid, floor), energy_cut(mid));
            let cuts = [(&narrow_cut, &floors), (&narrow_energy, energy_floors)];
            let from_nothing = fold_range(at, cuts, (mid, floor), None, None);
            let adopted = fold_range(at, cuts, (mid, floor), None, Some(&tables));
            assert!(adopted.adopted > 0);
            dropped += below - adopted.adopted;
            assert_eq!(
                cells_below(&adopted, tables.span),
                cells_below(&from_nothing, tables.span),
                "floor {floor}"
            );
            assert_eq!(adopted.filled, from_nothing.filled, "floor {floor}");
            let mut tally = RuleTally::default();
            assert_eq!(
                adopted.recover(&mut tally),
                from_nothing.recover(&mut tally),
                "floor {floor}"
            );
        }
        assert!(dropped > 0, "the narrower boxes dropped nothing adopted");
    }

    /// A gap whose ranges start from gave-up folds and whose splits form
    /// only the sums the score orders leave room for returns the points it
    /// returns with neither; and a fold started from gave-up tables under
    /// the score orders stores the cells and recovers the designs a fold
    /// from nothing forming every sum does, while skipping sums.
    #[test]
    fn gave_up_tables_and_score_orders_together_store_the_same_points() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let spec = MotifFrontSpec {
            utr5: "GGGAAACCC",
            utr3: "",
            constraints: RegionConstraints::default(),
            motifs: &[("GCC", MotifWindow::overlapping(9, 24))],
            pins: None,
        };
        let protein = "MKVLAWGHRSTEPQ";
        let aa = crate::residues(protein).unwrap();
        let five = Flank::parse(spec.utr5).unwrap();
        let three = Flank::parse(spec.utr3).unwrap();
        let graph = spec_graph(&aa, &usage, &spec, &five, &three).unwrap();
        let region = PairConstraint::new(&spec.constraints, &graph);
        let (ac, scope) = Ac::for_windows(spec.motifs).unwrap();
        let product = ProductDfa::build_scoped(&graph, &ac, scope).merge_equivalent();
        let lattice = Lattice::new(&product);
        let walk = motif_front_walk(&params, protein, &usage, &spec, 1).unwrap();
        let offset = graph.codon_offset();
        let unit = |v: &MotifVertexPoint| MotifVertexPoint {
            mfe: v.mfe,
            cai: v.cai - offset,
        };
        let a = unit(walk.vertices.first().unwrap());
        let b = unit(walk.vertices.last().unwrap());
        let lambda = tie(a, b);
        let grid = Grid::new(
            motif_pass_grid(derna_codon_dp::cai::sense_codons(&aa)).weights(),
            None,
            (&params, &graph, &region),
            1,
            None,
        );
        let energy_pass = grid.energy();
        let (lo, hi) = grid.bracket(lambda);
        let corner = brd(b.mfe, a.cai, lambda).max(brd(a.mfe, a.cai, lambda));
        let cut = Cut::between(&lo, hi.as_deref(), lambda, corner);

        // The whole gap under (reuse, rule): 1 forms every sum, 4 reads the
        // score orders.
        let gap = |reuse, rule| {
            let slices = SliceBudget {
                points: 20,
                grow: 0,
                max_rss_kb: None,
                workers: 1,
                open_in_room: false,
                fit: None,
                bracket: true,
                width: None,
                reuse,
            };
            SPLIT_PRUNE_TEST.with(|r| r.set(rule));
            let solved = solve_gap(
                &params,
                &region,
                &lattice,
                (a, b, None),
                (&cut, (&*energy_pass, grid.energy_floors(&lattice)), &[]),
                0.0,
                (Stop::default(), slices, 1),
            );
            SPLIT_PRUNE_TEST.with(|r| r.set(0));
            let Ok(gap) = solved else {
                panic!("a gap nothing stops is solved");
            };
            let values: Vec<(i32, u64)> =
                gap.points.iter().map(|p| (p.0 .0, p.1.to_bits())).collect();
            (values, gap.record.adopted)
        };
        let (neither, none) = gap(0, 1);
        assert_eq!(none, 0);
        for (reuse, rule) in [(0, 4), (3, 1), (3, 4)] {
            let (points, adopted) = gap(reuse, rule);
            assert_eq!(points, neither, "reuse {reuse}, rule {rule}");
            assert_eq!(adopted > 0, reuse > 0, "reuse {reuse}, rule {rule}");
        }

        // By hand: the whole gap gives up halfway under the score orders, and
        // the whole range and its lower half start from its tables under them,
        // against folds from nothing that form every sum.
        let floors = Floors::new(&params, &lattice, &region, &cut);
        let energy_floors = grid.energy_floors(&lattice);
        let (top, floor) = (b.mfe.0, a.cai);
        let wide_cut = Cut {
            ceiling: brd(Energy(top), floor, lambda).max(brd(a.mfe, floor, lambda)),
            ..cut
        };
        let wide_energy = Cut::between(&energy_pass, None, 1.0, f64::from(top));
        let at = (&params, &lattice, &region);
        let cuts = [(&wide_cut, &floors), (&wide_energy, energy_floors)];
        let mid = a.mfe.0 + (b.mfe.0 - a.mfe.0) / 2;
        let narrow_cut = Cut {
            ceiling: brd(Energy(mid), floor, lambda).max(brd(a.mfe, floor, lambda)),
            ..cut
        };
        let narrow_energy = Cut::between(&energy_pass, None, 1.0, f64::from(mid));
        let narrow = [(&narrow_cut, &floors), (&narrow_energy, energy_floors)];
        SPLIT_PRUNE_TEST.with(|r| r.set(4));
        let whole = fold_range(at, cuts, (top, floor), None, None);
        let budget = whole.filled[whole.filled.len() / 2];
        let gave_up = fold_range(at, cuts, (top, floor), Some(budget), None);
        SPLIT_PRUNE_TEST.with(|r| r.set(0));
        assert!(gave_up.aborted && !gave_up.stopped);
        let tables = gave_up.into_tables();
        for (boxed, range) in [(cuts, (top, floor)), (narrow, (mid, floor))] {
            SPLIT_PRUNE_TEST.with(|r| r.set(1));
            let from_nothing = fold_range(at, boxed, range, None, None);
            SPLIT_PRUNE_TEST.with(|r| r.set(4));
            let adopted = fold_range(at, boxed, range, None, Some(&tables));
            SPLIT_PRUNE_TEST.with(|r| r.set(0));
            assert!(adopted.adopted > 0, "top {}", range.0);
            assert_eq!(from_nothing.split[0].skipped, 0);
            assert!(
                adopted.split[0].skipped > 0 && adopted.split[1].skipped > 0,
                "the score orders skipped no sums, top {}",
                range.0
            );
            assert!(
                adopted.split[0].ruled_out > 0,
                "no split was ruled out from its cells' least scores, top {}",
                range.0
            );
            assert_eq!(
                cells_below(&adopted, u32::MAX),
                cells_below(&from_nothing, u32::MAX),
                "top {}",
                range.0
            );
            assert_eq!(adopted.filled, from_nothing.filled, "top {}", range.0);
            let mut tally = RuleTally::default();
            assert_eq!(
                adopted.recover(&mut tally),
                from_nothing.recover(&mut tally),
                "top {}",
                range.0
            );
        }
    }

    /// Skipping the split sums a staircase of the cell's own candidates
    /// strictly dominates stores the same points with the same provenance in
    /// every multi-loop and exterior cell, under either reference, and skips
    /// some sums under each.
    #[test]
    fn split_sums_skipped_by_dominance_store_the_same_cells() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let spec = MotifFrontSpec {
            utr5: "GGGAAACCC",
            utr3: "",
            constraints: RegionConstraints::default(),
            motifs: &[("GCC", MotifWindow::overlapping(9, 24))],
            pins: None,
        };
        let aa = crate::residues("MKVLAWGHRSTEPQ").unwrap();
        let five = Flank::parse(spec.utr5).unwrap();
        let three = Flank::parse(spec.utr3).unwrap();
        let graph = spec_graph(&aa, &usage, &spec, &five, &three).unwrap();
        let region = PairConstraint::new(&spec.constraints, &graph);
        let (ac, scope) = Ac::for_windows(spec.motifs).unwrap();
        let product = ProductDfa::build_scoped(&graph, &ac, scope).merge_equivalent();
        let lattice = Lattice::new(&product);
        let cells = |rule: u8| {
            SPLIT_PRUNE_TEST.with(|r| r.set(rule));
            let fold = Fold::run(
                &params,
                &lattice,
                &region,
                None,
                ([None; 4], None, &[]),
                (None, Stop::default(), 1),
            );
            SPLIT_PRUNE_TEST.with(|r| r.set(0));
            let mut m2: Vec<_> = fold
                .m2
                .iter()
                .map(|(&(u, v), f)| front_digest((0, u, v), f.points()))
                .collect();
            m2.sort_unstable();
            let z: Vec<_> = (0..fold.z.len())
                .map(|v| front_digest((1, 0, v as u32), fold.z[v].points()))
                .collect();
            let skipped = [fold.split[0].skipped, fold.split[1].skipped];
            ((m2, z, fold.stored), skipped)
        };
        let (off, none) = cells(1);
        assert_eq!(none, [0, 0]);
        assert!(off.0.len() > 10, "too few multi-loop cells to test");
        for rule in [2, 3] {
            let (on, skipped) = cells(rule);
            assert!(on == off, "rule {rule} changed the stored cells");
            assert!(skipped[0] > 0, "rule {rule} skipped no multi-loop sums");
        }
    }

    /// Forming only the sums the score orders leave room for gathers the
    /// same staircase, with the same provenance, as forming every sum, over
    /// staircases, shifts and cut lines that admit anything from nothing to
    /// everything, and forms fewer sums where a line binds.
    #[test]
    fn split_sums_read_in_score_order_gather_the_same_staircase() {
        let seed = std::cell::Cell::new(0x9e37_79b9_7f4a_7c15u64);
        let next = |n: u64| {
            seed.set(
                seed.get()
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1),
            );
            (seed.get() >> 33) % n
        };
        let staircase = |len: usize| {
            let (mut e, mut c) = (-(next(3000) as i32), -(next(400) as f64) / 7.0);
            let points: Vec<_> = (0..len)
                .map(|k| {
                    e += 1 + next(250) as i32;
                    c += (1 + next(90)) as f64 / 13.0;
                    Point::new(Energy(e), c, Prov::Multi { k: k as u32 })
                })
                .collect();
            Front::from_points(points)
        };
        let weights = [0.31, 0.62];
        let (mut fewer, mut admitted) = (0, 0);
        for case in 0..200 {
            let left = staircase(1 + case % 23);
            let right = staircase(1 + (case * 7) % 31);
            let shift = (Energy(next(500) as i32 - 250), next(50) as f64 / 3.0 - 8.0);
            let mut admit = Admit::all();
            if case % 3 != 0 {
                admit.c_min = next(60) as f64 - 30.0;
            }
            for &w in &weights[..case % 3] {
                admit.line(w, next(4000) as f64 - 2500.0);
            }
            let back = |l, r| Prov::Split {
                w: 7,
                w_hi: 0,
                l,
                r,
            };
            let gather = |by: (Option<&ByScore>, Option<&ByScore>)| {
                let (mut acc, mut got) = (Vec::new(), Vec::new());
                let (sums, formed, _) = gather_compose(
                    (&mut acc, &mut got),
                    (&left, &right),
                    shift,
                    &admit,
                    (Reference::None, by),
                    back,
                );
                merge_gathered(&mut acc, &mut got);
                (front_digest((0, 0, 0), &acc), acc.len(), sums, formed)
            };
            let (left_by, right_by) = (
                ByScore::new(&left, weights.iter().copied()),
                ByScore::new(&right, weights.iter().copied()),
            );
            let all = gather((None, None));
            let read = gather((Some(&left_by), Some(&right_by)));
            assert_eq!(all.0, read.0, "case {case} gathered a different staircase");
            assert_eq!(all.2, read.2, "case {case} counted different sums");
            assert_eq!(all.2, all.3);
            fewer += usize::from(read.3 < all.3);
            admitted += usize::from(all.1 > 0);
        }
        assert!(fewer > 50, "the score orders bound too few cases: {fewer}");
        assert!(admitted > 50, "too few cases admit anything: {admitted}");
    }
}
