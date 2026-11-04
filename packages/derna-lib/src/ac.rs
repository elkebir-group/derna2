//! Aho–Corasick automaton over the RNA alphabet `{A, C, G, U}` for hard
//! motif exclusion.
//!
//! Built from a forbidden-motif set. IUPAC degenerate codes are expanded into
//! their literal ACGU realizations (the *union* automaton), `T` read as `U`;
//! literal expansion is chosen over class-labeled trie edges because it keeps
//! the transition function a plain per-nucleotide table with no per-edge class
//! test — byte-equivalent on the constraint, and only larger on
//! wildcard-dense motif sets.
//!
//! The automaton reads a linear nucleotide sequence left-to-right. A state is
//! *accepting* when some forbidden motif has completed as a suffix of the
//! text read so far (standard output-link propagation, so a shorter motif that
//! is a suffix of a longer one still triggers). The product-DFA construction
//! ([`crate::product`]) removes every transition into an accepting state, so no
//! path through the product spells a sequence containing a forbidden motif.
//!
//! The transition function `δ(q, x)` is materialized densely ([`Ac::step`]), so
//! each step is `O(1)`; failure links are folded into the dense table at build
//! time and not retained.
//!
//! Enforcement can be restricted to a contiguous range of sequence positions —
//! see [`MotifWindow`] for the range and [`ScopedAc`] for the automaton that
//! reads it.

use std::collections::VecDeque;

use rna_mfe::Nucleotide;
use thiserror::Error;

/// The four RNA nucleotides — the automaton's alphabet size (`A, C, G, U` in
/// [`Nucleotide`] index order).
const ALPHABET: usize = 4;

/// An error building the automaton from a motif set.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum MotifError {
    /// A motif string was empty. An empty motif would forbid every position,
    /// which is a malformed constraint rather than a meaningful blacklist.
    #[error("motif is empty")]
    Empty,
    /// A motif contained a byte that is not an IUPAC nucleotide code.
    #[error("motif {motif:?} contains a non-IUPAC code {code:?}")]
    InvalidCode {
        /// The offending motif.
        motif: String,
        /// The unrecognized byte, as a character.
        code: char,
    },
}

/// An Aho–Corasick automaton over `{A, C, G, U}`.
///
/// State `0` is the non-accepting root and the start state. The dense
/// transition table [`Ac::step`] maps `(state, nucleotide)` to the next state
/// in `O(1)`; [`Ac::is_accept`] flags the states a forbidden motif completes
/// on.
pub struct Ac {
    /// `goto[state][nuc]` — the next state, with failure links folded in so
    /// every `(state, nucleotide)` has a defined target.
    goto: Vec<[u32; ALPHABET]>,
    /// `accept[state]` — some forbidden motif ends here (including, via output
    /// links, a shorter motif that is a suffix of the text read so far).
    accept: Vec<bool>,
    /// `placements[state]` — the windows the motifs ending here are forbidden
    /// inside, when the automaton was built from **placed** patterns. Empty at
    /// every state otherwise, which is what [`Ac::is_placed`] reads.
    ///
    /// A placed automaton is enforced differently: it is stepped at every
    /// position rather than held at a root outside one shared window, and a
    /// state rejects only where some pattern ending there has a window covering
    /// the occurrence. That keeps the state an absolute function of the linear
    /// prefix — it is only the automaton state of the whole prefix — so
    /// [`crate::product`]'s bifurcation argument is unchanged. What it gives up
    /// is the collapse outside the window, so the axis splits every node; a
    /// placed set is for a handful of patterns rather than for a long molecule.
    placements: Vec<Vec<Placement>>,
}

impl Ac {
    /// Build the automaton from a forbidden-motif set. Each motif is an IUPAC
    /// string (`A C G U/T` plus degenerate codes `R Y S W K M B D H V N`);
    /// degenerate codes expand into the union of their literal ACGU
    /// realizations. An empty motif set yields the pass-through automaton (a
    /// single non-accepting root — see [`Ac::is_pass_through`]).
    pub fn build<S: AsRef<str>>(motifs: &[S]) -> Result<Self, MotifError> {
        let mut trie = vec![TrieNode::default()];
        for motif in motifs {
            for pattern in expand_motif(motif.as_ref())? {
                insert(&mut trie, &pattern);
            }
        }
        Ok(fold_failure_links(trie))
    }

    /// Build the automaton from **placed** patterns — each an IUPAC string with
    /// the window it is forbidden inside, rather than one window for the set.
    ///
    /// This is what expresses several constraints that each hold at one place: a
    /// loop cutting a counterexample at the position it occurred cannot state
    /// that with one shared window, because a second cut would inherit the
    /// first's scope.
    pub fn build_placed<S: AsRef<str>>(placed: &[(S, MotifWindow)]) -> Result<Self, MotifError> {
        let mut trie = vec![TrieNode::default()];
        for (motif, window) in placed {
            for pattern in expand_motif(motif.as_ref())? {
                let node = insert(&mut trie, &pattern) as usize;
                let placement = Placement {
                    len: pattern.len() as u32,
                    window: *window,
                };
                if !trie[node].placements.contains(&placement) {
                    trie[node].placements.push(placement);
                }
            }
        }
        Ok(fold_failure_links(trie))
    }

    /// Build the automaton a `window` needs.
    ///
    /// The reach and the mechanism go together, so the choice belongs here
    /// rather than at each call site: an overlapping window is built
    /// **placed**, because an unplaced automaton is held at its root until the
    /// window starts and cannot see an occurrence that begins before it; every
    /// other window is built plain. A caller that picked for itself could make
    /// one `MotifWindow` mean two different constraints depending on which
    /// entry point read it.
    /// Build the automaton patterns that each carry their own window need,
    /// with the window the product axis is scoped to.
    ///
    /// Two things have to agree and are decided here together, as
    /// [`Ac::for_window`] decides the reach and the mechanism together. A set
    /// whose patterns share one window is that window's set: it routes to
    /// [`Ac::for_window`] and the product is scoped to it, which is the run a
    /// caller naming one window has always had. Patterns with **different**
    /// windows have no shared range to scope the axis to, so the automaton is
    /// built placed and the product is scoped to [`MotifWindow::all`] — the
    /// placed path steps the automaton at every position and scopes the accept
    /// test per pattern, so a narrower product window would promise a collapse
    /// that the enforcement does not take.
    pub fn for_windows<S: AsRef<str>>(
        placed: &[(S, MotifWindow)],
    ) -> Result<(Self, MotifWindow), MotifError> {
        let mut windows = placed.iter().map(|(_, window)| *window);
        let shared = match windows.next() {
            None => Some(MotifWindow::all()),
            Some(first) => windows.all(|window| window == first).then_some(first),
        };
        match shared {
            Some(window) => {
                let patterns: Vec<&str> = placed.iter().map(|(m, _)| m.as_ref()).collect();
                Ok((Self::for_window(&patterns, window)?, window))
            }
            None => Ok((Self::build_placed(placed)?, MotifWindow::all())),
        }
    }

    pub fn for_window<S: AsRef<str>>(
        motifs: &[S],
        window: MotifWindow,
    ) -> Result<Self, MotifError> {
        match window.reach() {
            WindowReach::Inside => Self::build(motifs),
            WindowReach::Overlapping => {
                let placed: Vec<(&str, MotifWindow)> =
                    motifs.iter().map(|m| (m.as_ref(), window)).collect();
                Self::build_placed(&placed)
            }
        }
    }

    /// The start (root) state.
    pub fn start(&self) -> u32 {
        0
    }

    /// The state reached from `q` on reading nucleotide `x`.
    pub fn step(&self, q: u32, x: Nucleotide) -> u32 {
        self.goto[q as usize][x as usize]
    }

    /// Whether `q` is an accepting state (a forbidden motif has completed).
    pub fn is_accept(&self, q: u32) -> bool {
        self.accept[q as usize]
    }

    /// Whether the automaton was built from placed patterns, so each carries its
    /// own window and enforcement is per pattern rather than per set.
    pub fn is_placed(&self) -> bool {
        self.placements.iter().any(|p| !p.is_empty())
    }

    /// Whether reading position `pos` into state `q` completes an occurrence
    /// that lies entirely inside its **own** window.
    ///
    /// A state can end several patterns — its own and, through the failure
    /// chain, every suffix of it — and they need not share a window. The state
    /// rejects where any one of them does.
    pub fn rejects_at(&self, q: u32, pos: u32) -> bool {
        self.placements[q as usize]
            .iter()
            .any(|placement| placement.forbids_at(pos))
    }

    /// The number of automaton states, `|A_M|`.
    pub fn num_states(&self) -> usize {
        self.goto.len()
    }

    /// Whether the automaton imposes no constraint: a single non-accepting root
    /// whose every transition self-loops. Holds exactly for the empty motif
    /// set, and lets the product collapse to the base DFA (no AC dimension).
    pub fn is_pass_through(&self) -> bool {
        self.num_states() == 1 && !self.accept[0]
    }
}

/// One trie node during construction: child links per nucleotide, the failure
/// link, and whether a motif ends here.
#[derive(Clone, Default)]
struct TrieNode {
    children: [Option<u32>; ALPHABET],
    fail: u32,
    accept: bool,
    /// The placed patterns ending at this node, own and inherited. Empty for
    /// every node of an unplaced automaton.
    placements: Vec<Placement>,
}

/// One pattern's length and the window it is forbidden in.
///
/// The length is carried because the window test is over the occurrence, not
/// over the position an accept is detected at: a pattern accepted on reading
/// position `pos` occupies `[pos + 1 - len, pos + 1)`, and whether that range
/// is forbidden is the window's own reading of it ([`WindowReach`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    len: u32,
    window: MotifWindow,
}

impl Placement {
    /// Whether an occurrence detected on reading `pos` falls under this
    /// placement's window, by that window's reach.
    fn forbids_at(&self, pos: u32) -> bool {
        let end = pos + 1;
        if end < self.len {
            return false;
        }
        let start = end - self.len;
        match self.window.reach() {
            WindowReach::Inside => self.window.lo() <= start && end <= self.window.hi(),
            WindowReach::Overlapping => start < self.window.hi() && end > self.window.lo(),
        }
    }
}

/// The literal nucleotides an IUPAC code denotes (`T` read as `U`); empty for
/// any byte that is not an IUPAC code.
pub(crate) fn iupac_class(code: u8) -> &'static [Nucleotide] {
    use Nucleotide::{A, C, G, U};
    match code.to_ascii_uppercase() {
        b'A' => &[A],
        b'C' => &[C],
        b'G' => &[G],
        b'U' | b'T' => &[U],
        b'R' => &[A, G],
        b'Y' => &[C, U],
        b'S' => &[C, G],
        b'W' => &[A, U],
        b'K' => &[G, U],
        b'M' => &[A, C],
        b'B' => &[C, G, U],
        b'D' => &[A, G, U],
        b'H' => &[A, C, U],
        b'V' => &[A, C, G],
        b'N' => &[A, C, G, U],
        _ => &[],
    }
}

/// Expand an IUPAC motif into the literal `ACGU` spellings it denotes.
///
/// The public form of the expansion, so a caller outside this crate — matching
/// a motif against sequence it cannot design away, say — does not restate
/// which nucleotides `N` stands for.
///
/// **The expansion is exponential in the number of degenerate codes**, which a
/// caller working position by position avoids: expanding one code gives that
/// position's allowed letters in at most four spellings, and a scan over those
/// costs the motif's length rather than its number of spellings.
pub fn expand_iupac_motif(motif: &str) -> Result<Vec<String>, MotifError> {
    Ok(expand_motif(motif)?
        .into_iter()
        .map(|pattern| {
            pattern
                .into_iter()
                .map(|n| match n {
                    Nucleotide::A => 'A',
                    Nucleotide::C => 'C',
                    Nucleotide::G => 'G',
                    Nucleotide::U => 'U',
                })
                .collect()
        })
        .collect())
}

/// Expand an IUPAC motif into the literal ACGU patterns it denotes — the
/// Cartesian product of its per-position classes.
pub(crate) fn expand_motif(motif: &str) -> Result<Vec<Vec<Nucleotide>>, MotifError> {
    if motif.is_empty() {
        return Err(MotifError::Empty);
    }
    let mut patterns = vec![Vec::new()];
    for &code in motif.as_bytes() {
        let class = iupac_class(code);
        if class.is_empty() {
            return Err(MotifError::InvalidCode {
                motif: motif.to_string(),
                code: code as char,
            });
        }
        patterns = extend_patterns(patterns, class);
    }
    Ok(patterns)
}

/// Append every nucleotide of `class` to every pattern in `patterns` — one
/// factor of the Cartesian-product expansion.
fn extend_patterns(patterns: Vec<Vec<Nucleotide>>, class: &[Nucleotide]) -> Vec<Vec<Nucleotide>> {
    let mut out = Vec::with_capacity(patterns.len() * class.len());
    for prefix in &patterns {
        for &nuc in class {
            let mut pattern = prefix.clone();
            pattern.push(nuc);
            out.push(pattern);
        }
    }
    out
}

/// Insert one literal pattern into the trie, creating nodes as needed and
/// flagging the terminal node as accepting. Returns that node, which is where a
/// placed pattern's window is recorded.
fn insert(trie: &mut Vec<TrieNode>, pattern: &[Nucleotide]) -> u32 {
    let mut cur = 0usize;
    for &nuc in pattern {
        let slot = nuc as usize;
        cur = match trie[cur].children[slot] {
            Some(next) => next as usize,
            None => {
                let next = trie.len() as u32;
                trie.push(TrieNode::default());
                trie[cur].children[slot] = Some(next);
                next as usize
            }
        };
    }
    trie[cur].accept = true;
    cur as u32
}

/// Compute failure links by BFS and fold them into a dense `goto` table,
/// propagating accept flags along output (failure) links. Consumes the trie.
fn fold_failure_links(mut trie: Vec<TrieNode>) -> Ac {
    let mut goto = vec![[0u32; ALPHABET]; trie.len()];
    let mut queue = VecDeque::new();
    for (x, child) in trie[0].children.iter().enumerate() {
        if let Some(child) = *child {
            goto[0][x] = child;
            queue.push_back(child);
        }
    }
    while let Some(u) = queue.pop_front() {
        let (uu, fail) = (u as usize, trie[u as usize].fail as usize);
        trie[uu].accept |= trie[fail].accept;
        // A state ends every pattern its failure chain ends, so it inherits
        // their placements exactly as it inherits `accept`.
        let inherited = trie[fail].placements.clone();
        for placement in inherited {
            if !trie[uu].placements.contains(&placement) {
                trie[uu].placements.push(placement);
            }
        }
        let children = trie[uu].children;
        for (x, child) in children.into_iter().enumerate() {
            match child {
                Some(child) => {
                    trie[child as usize].fail = goto[fail][x];
                    goto[uu][x] = child;
                    queue.push_back(child);
                }
                None => goto[uu][x] = goto[fail][x],
            }
        }
    }
    let accept = trie.iter().map(|node| node.accept).collect();
    let placements = trie.into_iter().map(|node| node.placements).collect();
    Ac {
        goto,
        accept,
        placements,
    }
}

/// The range of sequence positions a forbidden-motif set is enforced over, as
/// the half-open interval `[lo, hi)`.
///
/// # What a window forbids
///
/// **An occurrence is forbidden exactly when it lies entirely inside the
/// window** — a motif occupying positions `[s, s + len)` is rejected iff
/// `lo <= s` and `s + len <= hi`. An occurrence that *straddles* either edge is
/// permitted, including one that begins before `lo` and ends inside.
///
/// That is a choice, and the alternative is coherent: forbidding every
/// occurrence whose *last* nucleotide falls inside the window would additionally
/// catch the straddlers on the left, at the price of reading `k_max - 1`
/// positions of sequence from outside the window the caller declared. The
/// entirely-inside reading is the one that makes a window mean *these positions
/// and no others*, which is what a caller deriving motifs from the sequence
/// immediately outside the window needs: were left-straddling occurrences
/// forbidden, the source sequence would be partly back inside the constraint it
/// generated.
///
/// The reading is a property of the constraint rather than of the
/// implementation, so it is named wherever a window is reported, not only here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotifWindow {
    lo: u32,
    hi: u32,
    reach: WindowReach,
}

/// Which occurrences a window forbids: those lying wholly within it, or those
/// touching it at all.
///
/// The two readings differ only on an occurrence straddling a boundary, and
/// which one is right follows from what lies outside the window.
///
/// - [`WindowReach::Inside`] is the default reading:
///   a window means *these positions and no others*. It is what a caller
///   deriving patterns from the sequence immediately outside the window needs,
///   since otherwise the source sequence would be partly back inside the
///   constraint it generated.
/// - [`WindowReach::Overlapping`] forbids an occurrence with any position in
///   the window, the straddlers included. It is the reading for a window whose
///   positions are *chosen* and whose outside is fixed: a straddling occurrence
///   then has a nucleotide the caller is choosing, so it is removable, and
///   leaving it permitted would let the molecule spell the pattern across the
///   boundary.
///
/// An overlapping window is enforced through the **placed** path
/// ([`Ac::build_placed`]), which steps the automaton at every position: a
/// straddler beginning before `lo` is invisible to an automaton held at the
/// root until `lo`, so the reach and the mechanism go together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowReach {
    /// Forbid an occurrence lying wholly inside the window.
    #[default]
    Inside,
    /// Forbid an occurrence with any position inside the window.
    Overlapping,
}

impl MotifWindow {
    /// The whole molecule: every occurrence at every position is forbidden,
    /// which is enforcement without a window. The two reaches agree here.
    pub fn all() -> Self {
        Self {
            lo: 0,
            hi: u32::MAX,
            reach: WindowReach::Inside,
        }
    }

    /// Enforce over `[lo, hi)`, forbidding the occurrences lying wholly inside
    /// it. An empty or inverted range (`hi <= lo`) enforces nothing, which is
    /// the faithful reading of a window with no positions in it rather than an
    /// error.
    pub fn new(lo: u32, hi: u32) -> Self {
        Self {
            lo,
            hi,
            reach: WindowReach::Inside,
        }
    }

    /// Enforce over `[lo, hi)`, forbidding every occurrence that **overlaps**
    /// it — see [`WindowReach`]. Enforced through [`Ac::build_placed`].
    pub fn overlapping(lo: u32, hi: u32) -> Self {
        Self {
            lo,
            hi,
            reach: WindowReach::Overlapping,
        }
    }

    /// Which occurrences this window forbids.
    pub fn reach(&self) -> WindowReach {
        self.reach
    }

    /// First enforced position.
    pub fn lo(&self) -> u32 {
        self.lo
    }

    /// First position past the enforced range.
    pub fn hi(&self) -> u32 {
        self.hi
    }

    /// Whether the window covers every position of a molecule `seq_len`
    /// nucleotides long, so enforcement is the unscoped global one.
    pub fn covers(&self, seq_len: u32) -> bool {
        self.lo == 0 && self.hi >= seq_len
    }
}

/// The single AC state held at every position outside the window.
///
/// Before `lo` the automaton is never stepped, so it sits at the root; past
/// `hi` the reachable set is *collapsed* onto one value so the axis does not
/// split the rest of the molecule. Both are the root state: they occupy
/// disjoint position ranges, so one value serves both and every reachable state
/// stays inside [`Ac`]'s own state space (in particular it is never accepting).
const OUTSIDE: u32 = 0;

/// An [`Ac`] enforced only over a [`MotifWindow`].
///
/// The automaton reads the linear sequence as before, with three position
/// regions:
///
/// | positions | AC state |
/// |---|---|
/// | `pos < lo` | held at the root — never stepped, so never left |
/// | `lo <= pos < hi` | stepped normally; a transition into an accepting state is removed |
/// | `pos >= hi` | collapsed to one sink value |
///
/// The left region needs no mechanism: not stepping is what keeps it at one
/// state. The right region does — at `hi` the reachable set is one state per
/// surviving in-window prefix, and collapsing it is what stops the axis
/// splitting every node from `hi` to the 3′ end. The collapse happens *after*
/// the accept test on the transition into `hi`, so an occurrence ending exactly
/// at `hi` is still inside the window and still rejected.
///
/// The scoped state stays an **absolute function of the linear prefix**, which
/// is the property [`crate::product`]'s bifurcation argument rests on: it is the
/// root below `lo`, the automaton state of `seq[lo..pos]` inside, and the sink
/// at or above `hi` — each determined by the prefix and the position alone. So a
/// CKY split node is still one product node both sub-derivations agree on, and
/// no boundary needs an independent entry/exit pair.
///
/// [`MotifWindow::all`] reduces this to plain global enforcement transition for
/// transition.
#[derive(Clone, Copy)]
pub struct ScopedAc<'a> {
    ac: &'a Ac,
    window: MotifWindow,
    /// [`Ac::is_placed`], read once: it scans every state's placements, and
    /// [`Self::step_at`] runs once per nucleotide the fold threads.
    placed: bool,
}

impl<'a> ScopedAc<'a> {
    /// Enforce `ac` over `window`.
    pub fn new(ac: &'a Ac, window: MotifWindow) -> Self {
        Self {
            ac,
            window,
            placed: ac.is_placed(),
        }
    }

    /// The start state — the root, which is also the state held everywhere
    /// outside the window.
    pub fn start(&self) -> u32 {
        OUTSIDE
    }

    /// Read the nucleotide at sequence position `pos` from state `q`, returning
    /// the next state, or `None` when the transition completes a forbidden
    /// occurrence and is therefore removed.
    ///
    /// The accept test and the transition are one call because they are one
    /// decision: outside the window there is no occurrence to complete, so a
    /// caller that stepped and then tested separately would have to remember to
    /// scope both.
    pub fn step_at(&self, pos: u32, q: u32, x: Nucleotide) -> Option<u32> {
        // A placed set carries a window per pattern, so there is no shared range
        // to hold outside of: the automaton is stepped everywhere and only the
        // *accept* test is scoped, per pattern. The state is then the plain
        // automaton state of the whole prefix, which is still an absolute
        // function of the prefix and the position — what the product's
        // bifurcation argument needs. What it gives up is the collapse, so the
        // axis splits every node rather than only those inside one window.
        if self.placed {
            let next = self.ac.step(q, x);
            return (!self.ac.rejects_at(next, pos)).then_some(next);
        }
        // An unplaced automaton holds the root until `lo`, so it has no memory
        // of the positions before the window and cannot see an occurrence
        // beginning there. That is sound for the inside reading, whose
        // occurrences lie within the window, and cannot express the overlapping
        // one, which is why such a window is built placed.
        debug_assert_eq!(
            self.window.reach(),
            WindowReach::Inside,
            "an overlapping window must be enforced through Ac::build_placed"
        );
        if pos < self.window.lo || pos >= self.window.hi {
            return Some(OUTSIDE);
        }
        let next = self.ac.step(q, x);
        if self.ac.is_accept(next) {
            return None;
        }
        // Reading position `pos` lands at position `pos + 1`; at or past `hi`
        // that is outside the window, and the reachable set collapses.
        Some(if pos + 1 >= self.window.hi {
            OUTSIDE
        } else {
            next
        })
    }

    /// The window enforcement is scoped to.
    pub fn window(&self) -> MotifWindow {
        self.window
    }

    /// The unscoped automaton.
    pub fn ac(&self) -> &'a Ac {
        self.ac
    }

    /// The automaton's state count `|A_M|`.
    pub fn num_states(&self) -> usize {
        self.ac.num_states()
    }

    /// Whether the automaton constrains nothing, so the window is moot. See
    /// [`Ac::is_pass_through`].
    pub fn is_pass_through(&self) -> bool {
        self.ac.is_pass_through()
    }

    /// Whether `seq` — read as a whole molecule starting at position 0 —
    /// contains an occurrence the window forbids, by its reach.
    ///
    /// The predicate stated directly, for callers checking a finished sequence
    /// rather than building a lattice.
    pub fn is_clean(&self, seq: &[Nucleotide]) -> bool {
        let mut q = self.start();
        for (pos, &x) in seq.iter().enumerate() {
            match self.step_at(pos as u32, q, x) {
                Some(next) => q = next,
                None => return false,
            }
        }
        true
    }
}
