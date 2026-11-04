//! A rival certificate for the emptiness of the tight set.
//!
//! # The certificate
//!
//! An item `x` is tight when its relaxed minimum `F(x)` equals its constrained
//! minimum `E(x) = Ep + e(x) + Eq` (the context blocks' own minima plus the
//! item's own minimum `e(x)`). A **rival** of a pair `(x, d)` — `d` a
//! configuration of `x` alone — is a configuration of the whole word whose cost
//! is at most a claimed value `c`; its **slack** is `c − (Ep + cost(d) + Eq)`.
//! If `x` is tight and `d` is one of its own optima, every rival has slack
//! `≥ 0`, because `c ≥ F(x) = E(x)`. So if every pair `(x, d)` a dynamic program
//! ranges over has some rival of negative slack, no item is tight.
//!
//! Soundness needs two things, both kept here: each rival is a genuine
//! configuration whose true cost is at most the claimed one (claimed gains are
//! never larger than true gains), and the pairs are filtered only on what the
//! program's state distinguishes, so that every `(x, own optimum)` survives
//! unless a rival kills it. The converse fails: a surviving pair proves nothing.
//!
//! # The rival family
//!
//! Everything below is decided at the outermost loop, which at `dangles = 0`
//! is additive: a configuration of a disjoint span keeps its own cost when it
//! is placed beside another one.
//!
//! - **Clamp** (`D(B)`, [`crate::clamp`]): the context blocks paired around `d`'s
//!   `B` top-level pairs. Slack `D(B)`.
//! - **Free 5′ end** (`g5(a)`): with `d`'s first `a` symbols unmatched, the
//!   prefix and those symbols folded together beside `d`. Slack `−g5(a)`, where
//!   `g5(a)` is the least gain over every spelling of the item's first `a`
//!   symbols, so it holds for any item. `g5` never falls as `a` grows (more
//!   unmatched symbols at the end of a span cost nothing), so a run longer than
//!   the enumerated cap reuses the cap's value.
//! - **Free 3′ end** (`g3(b)`): the same at the other end.
//! - **Window** (only for `d = ∅`): the own optimum of `m` consecutive positions
//!   placed in the otherwise empty configuration. Slack = the window's minimum.
//! - **Terminal loop under a clamp** (only for `d = ∅`): one terminal loop of a
//!   window, a one-branch configuration, wrapped by the one-branch clamp. Slack =
//!   the loop's cost `+ D(1)`.
//!
//! The first three are properties of `(B, a, b)` only, each monotone, so the
//! pairs they leave are an outermost-loop class ([`ExteriorClass`]) and the
//! least scalarized value over that class is [`class_optimum`](crate::class_bound::class_optimum) read over an
//! exact scan. The last two need the item's spelling in every window: a sweep
//! over positions keeps the distinct last-`(m − 1)`-choice suffixes of the
//! prefixes no window of which has a killing rival.
//!
//! # Exact end rivals
//!
//! The tabulated `g5`, `g3` are least gains over spellings, so they hold for
//! every item but are weak for a given one. [`rival_optimum`] instead prices
//! each end run by the item's **own** spelling: the exterior recurrence over
//! the scan's closed-pair cells, with the leading run up to `cap` symbols
//! enumerated as DFA paths (each kept only if the prefix gains nothing from
//! it) and the trailing run likewise. A run longer than `cap` must have a
//! gainless last (or first) `cap` symbols — a necessary condition, since a
//! longer free run never gains less — and nothing more is asked of it.
//!
//! The same recurrence prices one more rival, the **joint end clamp**: the
//! clamp table recomputed with the first and last few end-run symbols joined
//! to the context blocks, so a clamp helix may continue into the item's own
//! unmatched ends (see [`rival_optimum`]).

use std::cell::RefCell;
use std::collections::{BinaryHeap, HashMap};

use derna_codon_dp::cai::compute_cai_tables;
use derna_codon_dp::codon_table::AMINO_ACIDS;
use derna_codon_dp::CodonUsage;
use rna_mfe::Params;

use rna_mfe::loops::pair_from_nucleotides;
use rna_mfe::{Energy, Nucleotide};

use crate::clamp::clamp_table;
use crate::class_bound::{ClassOptimum, ExteriorClass};
use crate::dfa::{Dfa, Node};
use crate::free_fold::{sequence_constrained_mfe, CdsWindow, FreeFoldError};
use crate::region::{Enforcement, RegionConstraints};
use crate::state::{Back, PairKey, PairRef, State};
use crate::tables::Tables;
use crate::tail_cut::TailCut;

/// One piece of a stand-in probe ([`EndGains::standin_gain`]).
#[derive(Debug, Clone, Copy)]
pub enum Part<'a> {
    /// Exposed item symbols, in order.
    Run(&'a [Nucleotide]),
    /// A kept domain, as a stand-in closed on its outer pair's symbols.
    Stand((Nucleotide, Nucleotide)),
}

/// Rival gains of one context pair at one item length.
#[derive(Debug, Clone)]
pub struct RivalTables {
    /// `D(B)` in centiunits for `B = 0..=b_max`; `None` where only a clamp that
    /// depends on the item attains it (no rival is used there).
    pub clamp: Vec<Option<i32>>,
    /// `g5(a)` in centiunits for `a = 0..=cap`: the least gain over spellings.
    pub g5: Vec<i32>,
    /// `g3(b)` in centiunits for `b = 0..=cap`.
    pub g3: Vec<i32>,
}

/// The synonymous codons of each position, as symbol triples, with their
/// log-weights (the full menus: a superset of any restricted choice lattice).
#[derive(Debug, Clone)]
pub struct Menus {
    /// Per position, `(codon symbols, log-weight)`.
    pub choices: Vec<Vec<([u8; 3], f64)>>,
}

impl Menus {
    /// The full synonymous menus of `protein` (amino-acid indices).
    pub fn full(protein: &[usize], usage: &CodonUsage) -> Self {
        let cai = compute_cai_tables(usage);
        let choices = protein
            .iter()
            .map(|&aa| {
                AMINO_ACIDS[aa]
                    .codons
                    .iter()
                    .enumerate()
                    .map(|(ci, c)| (rna(c.nuc), cai.log_cai[aa][ci]))
                    .collect()
            })
            .collect();
        Self { choices }
    }

    /// Every distinct spelling of the first `len` symbols (`front`) or the
    /// last `len` symbols (`!front`).
    fn spellings(&self, len: usize, front: bool) -> Vec<String> {
        let k = len.div_ceil(3).min(self.choices.len());
        let span: Vec<&Vec<([u8; 3], f64)>> = if front {
            self.choices[..k].iter().collect()
        } else {
            self.choices[self.choices.len() - k..].iter().collect()
        };
        let mut words = vec![String::new()];
        for menu in span {
            words = words
                .iter()
                .flat_map(|w| {
                    menu.iter()
                        .map(move |(c, _)| format!("{w}{}", std::str::from_utf8(c).unwrap()))
                })
                .collect();
        }
        let mut cut: Vec<String> = words
            .into_iter()
            .map(|w| {
                let len = len.min(w.len());
                if front {
                    w[..len].to_string()
                } else {
                    w[w.len() - len..].to_string()
                }
            })
            .collect();
        cut.sort();
        cut.dedup();
        cut
    }
}

/// A codon's nucleotide indices (`A = 0, C = 1, G = 2, U = 3`) as symbols.
fn rna(c: [u8; 3]) -> [u8; 3] {
    c.map(|b| b"ACGU"[b as usize])
}

/// The reference minimizer's least cost of `seq`, in centiunits.
pub fn mfe(params: &Params, seq: &str) -> i32 {
    rna_mfe::fold(params, seq.as_bytes()).map_or(0, |(_, e)| e.0)
}

impl RivalTables {
    /// The tables for context blocks `utr5`, `utr3` around items spelled from
    /// `menus`, the clamp to `b_max` branches and the end gains to `cap` symbols.
    pub fn compute(
        params: &Params,
        utr5: &str,
        utr3: &str,
        menus: &Menus,
        b_max: usize,
        cap: usize,
    ) -> Result<Self, FreeFoldError> {
        let clamp = clamp_table(params, utr5, utr3, b_max)?
            .into_iter()
            .map(|d| d.map(|v| (v * 100.0).round() as i32))
            .collect();
        let gains = |front: bool| -> Vec<i32> {
            let ctx = if front { utr5 } else { utr3 };
            let alone = mfe(params, ctx);
            let mut out = vec![0];
            for len in 1..=cap {
                let least = menus
                    .spellings(len, front)
                    .iter()
                    .map(|s| {
                        let word = if front {
                            format!("{ctx}{s}")
                        } else {
                            format!("{s}{ctx}")
                        };
                        alone - mfe(params, &word)
                    })
                    .min()
                    .unwrap_or(0);
                // A longer free run never gains less (module docs); keep the
                // table monotone against round-off in the spellings' minimum.
                out.push(least.max(*out.last().unwrap()));
            }
            out
        };
        Ok(Self {
            clamp,
            g5: gains(true),
            g3: gains(false),
        })
    }

    /// Whether the clamp rival kills every configuration with `b` branches.
    fn clamp_kills(&self, b: usize) -> bool {
        self.clamp_at(b).is_some_and(|d| d < 0)
    }

    /// `D(b)`, extended past the table: past one branch a gaining clamp gains
    /// at least `ML.intern` more per branch, so a negative last entry stays
    /// negative; a zero last entry says nothing about larger counts.
    fn clamp_at(&self, b: usize) -> Option<i32> {
        match self.clamp.get(b) {
            Some(&d) => d,
            None => self.clamp.last().copied().flatten().filter(|&d| d < 0),
        }
    }

    /// The outermost-loop class the clamp and end rivals leave for `B >= 1`,
    /// or `None` when the clamp kills every branch count from one on.
    pub fn class(&self) -> Option<ExteriorClass> {
        let b_max = self.clamp.len() - 1;
        let alive: Vec<usize> = (1..=b_max).filter(|&b| !self.clamp_kills(b)).collect();
        let open_above = !self.clamp_kills(b_max + 1);
        if alive.is_empty() && !open_above {
            return None;
        }
        let lo = alive.first().copied().unwrap_or(b_max + 1);
        let hi = if open_above {
            None
        } else {
            alive.last().map(|&b| b as u8)
        };
        Some(ExteriorClass {
            min_branches: lo as u8,
            max_branches: hi,
            max_lead: last_free(&self.g5),
            first_close: None,
            max_trail: last_free(&self.g3),
        })
    }

    /// Whether the empty configuration survives the clamp rival.
    pub fn empty_survives_clamp(&self) -> bool {
        !self.clamp_kills(0)
    }

    /// The bottleneck value over `B >= 1`: the best of
    /// `min(D(B), −g5(a), −g3(b))` over the tabulated `B`, `a`, `b`
    /// (centiunits; `0` when some shape survives, given it is reachable).
    pub fn bottleneck(&self) -> i32 {
        let ends = (-self.g5[0]).min(-self.g3[0]);
        (1..self.clamp.len())
            .map(|b| self.clamp_at(b).unwrap_or(0).min(ends))
            .max()
            .unwrap_or(i32::MIN)
    }
}

/// The longest run whose gain is zero, or `None` when even the cap's is.
fn last_free(gains: &[i32]) -> Option<u32> {
    let cap = gains.len() - 1;
    let free = gains.iter().take_while(|&&g| g == 0).count() - 1;
    (free < cap).then_some(free as u32)
}

/// What the sweep over the empty configuration found.
#[derive(Debug, Clone)]
pub struct EmptySweep {
    /// The position at which no prefix survived, if one did not.
    pub emptied_at: Option<usize>,
    /// Surviving suffixes per position.
    pub level_sizes: Vec<usize>,
    /// The highest-weight surviving item and its log-weight, if any survived.
    pub best: Option<(String, f64)>,
    /// Windows folded (distinct spellings).
    pub folds: usize,
}

/// Sweep the empty configuration `d = ∅` against the window rival and, when
/// `clamp1` (`D(1)`, centiunits) is at most `-gate`, the terminal-loop-under-
/// clamp rival. Windows of `m` positions; shorter items are checked whole.
pub fn sweep_empty(
    params: &Params,
    menus: &Menus,
    m: usize,
    clamp1: Option<i32>,
    gate: i32,
    threads: usize,
) -> EmptySweep {
    let n = menus.choices.len();
    let keep = 3 * (m - 1);
    let width = 3 * m.min(n);
    let loop_rival = clamp1.filter(|&d| d <= -gate);
    // level: suffix (at most `keep` symbols) -> (best whole prefix, log-weight)
    let mut level: HashMap<String, (String, f64)> = HashMap::new();
    level.insert(String::new(), (String::new(), 0.0));
    let mut memo: HashMap<String, bool> = HashMap::new();
    let mut sizes = Vec::with_capacity(n);
    for (pos, menu) in menus.choices.iter().enumerate() {
        let mut cand: Vec<(String, String, f64)> = Vec::new();
        for (suffix, (whole, w)) in &level {
            for (c, cw) in menu {
                let c = std::str::from_utf8(c).unwrap();
                cand.push((format!("{suffix}{c}"), format!("{whole}{c}"), w + cw));
            }
        }
        let ready = |t: &str| t.len() == width || (pos + 1 == n && t.len() < width);
        let todo: Vec<String> = {
            let mut v: Vec<String> = cand
                .iter()
                .filter(|(t, _, _)| ready(t) && !memo.contains_key(t))
                .map(|(t, _, _)| t.clone())
                .collect();
            v.sort();
            v.dedup();
            v
        };
        for (t, killed) in kill_all(params, &todo, loop_rival, threads) {
            memo.insert(t, killed);
        }
        let mut next: HashMap<String, (String, f64)> = HashMap::new();
        for (t, whole, w) in cand {
            if ready(&t) && memo[&t] {
                continue;
            }
            let key = if t.len() > keep {
                t[t.len() - keep..].to_string()
            } else {
                t
            };
            match next.get(&key) {
                Some((_, old)) if *old >= w => {}
                _ => {
                    next.insert(key, (whole, w));
                }
            }
        }
        level = next;
        sizes.push(level.len());
        if level.is_empty() {
            return EmptySweep {
                emptied_at: Some(pos),
                level_sizes: sizes,
                best: None,
                folds: memo.len(),
            };
        }
    }
    let best = level.into_values().max_by(|a, b| a.1.total_cmp(&b.1));
    EmptySweep {
        emptied_at: None,
        level_sizes: sizes,
        best,
        folds: memo.len(),
    }
}

/// For each window, whether a rival of the empty configuration kills it.
fn kill_all(
    params: &Params,
    windows: &[String],
    loop_rival: Option<i32>,
    threads: usize,
) -> Vec<(String, bool)> {
    let threads = threads.max(1);
    let chunk = windows.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = windows
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    part.iter()
                        .map(|w| (w.clone(), window_kills(params, w, loop_rival)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("window worker"))
            .collect()
    })
}

/// Whether the empty configuration of a word containing `w` has a rival of
/// negative slack inside `w`.
pub fn window_kills(params: &Params, w: &str, loop_rival: Option<i32>) -> bool {
    if mfe(params, w) < 0 {
        return true;
    }
    loop_rival.is_some_and(|d1| least_terminal_loop(params, w).is_some_and(|h| h + d1 < 0))
}

/// The least cost of a configuration of `w` with exactly one pair (a single
/// terminal loop), scored by the reference evaluator.
pub fn least_terminal_loop(params: &Params, w: &str) -> Option<i32> {
    let s = w.as_bytes();
    let mut best: Option<i32> = None;
    let mut dots = vec![b'.'; s.len()];
    for i in 0..s.len() {
        for j in i + 4..s.len() {
            if !pairs(s[i], s[j]) {
                continue;
            }
            dots[i] = b'(';
            dots[j] = b')';
            let db = std::str::from_utf8(&dots).unwrap();
            if let Ok(e) = rna_mfe::evaluate(params, s, db) {
                best = Some(best.map_or(e.0, |b: i32| b.min(e.0)));
            }
            dots[i] = b'.';
            dots[j] = b'.';
        }
    }
    best
}

fn pairs(a: u8, b: u8) -> bool {
    matches!(
        (a, b),
        (b'A', b'U') | (b'U', b'A') | (b'G', b'C') | (b'C', b'G') | (b'G', b'U') | (b'U', b'G')
    )
}

/// The outermost-loop shape of a dot-bracket configuration: top-level pairs,
/// leading and trailing unmatched runs.
pub fn shape(structure: &str) -> (usize, usize, usize) {
    let mut depth = 0usize;
    let mut top = 0;
    for ch in structure.bytes() {
        match ch {
            b'(' => {
                top += usize::from(depth == 0);
                depth += 1;
            }
            b')' => depth -= 1,
            _ => {}
        }
    }
    let lead = structure.bytes().take_while(|&c| c == b'.').count();
    let trail = structure.bytes().rev().take_while(|&c| c == b'.').count();
    (top, lead, trail)
}

/// Whether an item whose own optimum has outermost shape `(b, a, t)` lies in
/// `class` (`None`: the clamp kills every `b >= 1`).
pub fn admits(class: Option<&ExteriorClass>, b: usize, a: usize, t: usize) -> bool {
    let Some(c) = class else {
        return false;
    };
    b >= c.min_branches as usize
        && c.max_branches.is_none_or(|m| b <= m as usize)
        && c.max_lead.is_none_or(|m| a <= m as usize)
        && c.max_trail.is_none_or(|m| t <= m as usize)
}

/// The gain a context block takes from an item end run of a given spelling,
/// memoized: `MFE(ctx) − MFE(ctx + run)` (5′) or `MFE(ctx) − MFE(run + ctx)` (3′).
pub struct EndGains<'a> {
    params: &'a Params,
    utr5: &'a str,
    utr3: &'a str,
    alone5: i32,
    alone3: i32,
    memo: HashMap<(bool, Vec<u8>), i32>,
    joint: HashMap<(String, String), Vec<i32>>,
    single: HashMap<(String, String, String), i32>,
    /// Split refolds scored ([`Self::split_gain`]) and the largest gain read.
    pub split_calls: usize,
    pub split_gain_best: Option<i32>,
}

impl<'a> EndGains<'a> {
    /// Gains against the context blocks `utr5` and `utr3`.
    pub fn new(params: &'a Params, utr5: &'a str, utr3: &'a str) -> Self {
        Self {
            params,
            utr5,
            utr3,
            alone5: mfe(params, utr5),
            alone3: mfe(params, utr3),
            memo: HashMap::new(),
            joint: HashMap::new(),
            single: HashMap::new(),
            split_calls: 0,
            split_gain_best: None,
        }
    }

    /// The gain of the run `nucs` at the 5′ end (`front`) or the 3′ end.
    pub fn gain(&mut self, front: bool, nucs: &[Nucleotide]) -> i32 {
        if nucs.is_empty() {
            return 0;
        }
        let key: Vec<u8> = nucs.iter().map(|n| *n as u8).collect();
        if let Some(&g) = self.memo.get(&(front, key.clone())) {
            return g;
        }
        let run: String = nucs.iter().map(|n| b"ACGU"[*n as usize] as char).collect();
        let g = if front {
            self.alone5 - mfe(self.params, &format!("{}{run}", self.utr5))
        } else {
            self.alone3 - mfe(self.params, &format!("{run}{}", self.utr3))
        };
        self.memo.insert((front, key), g);
        g
    }
}

impl EndGains<'_> {
    /// The clamp gain `D'(b)` (centiunits) with the end runs `l` and `r`
    /// joined to the context blocks: the clamp table of `(utr5 + l, r + utr3)`
    /// up to `b_max` branches, memoized per `(l, r)`. Past `b_max` the table's
    /// extension rule applies (a negative last entry stays negative).
    pub fn joint_clamp(
        &mut self,
        l: &[Nucleotide],
        r: &[Nucleotide],
        b: usize,
        b_max: usize,
    ) -> i32 {
        let spell = |s: &[Nucleotide]| -> String {
            s.iter().map(|n| b"ACGU"[*n as usize] as char).collect()
        };
        let key = (spell(l), spell(r));
        if !self.joint.contains_key(&key) {
            let u5 = format!("{}{}", self.utr5, key.0);
            let u3 = format!("{}{}", key.1, self.utr3);
            let table = branch_clamps(self.params, &u5, &u3, b_max.max(1));
            self.joint.insert(key.clone(), table);
        }
        let table = &self.joint[&key];
        match table.get(b) {
            Some(&d) => d,
            None => table.last().copied().filter(|&d| d < 0).unwrap_or(0),
        }
    }
}

/// The clamp table's entries for `B = 1..=b_max` (index 0 unused, `0`), in
/// centiunits, computed as [`clamp_table`] does but without its `B = 0` probes
/// over long bare regions: one stand-in of `B` terminal loops (each followed by
/// an unpairable run longer than the degree-2 span cap) between the blocks,
/// folded with only block-to-block pairs allowed, less the blocks and the
/// stand-in folded apart; past one branch an entry is also capped by the one
/// before plus the per-branch constant whenever that one gains. A probe the
/// folder rejects reads `0` (no rival), which only weakens the certificate.
pub fn branch_clamps(params: &Params, utr5: &str, utr3: &str, b_max: usize) -> Vec<i32> {
    const HAIRPIN: &str = "GGGCGCGAAAGCGCCC";
    const SPACER: usize = 33;
    let intern = params.ml.intern_37;
    let mut table = vec![0];
    for b in 1..=b_max {
        let mut cds = String::new();
        for _ in 0..b {
            cds.push_str(HAIRPIN);
            cds.push_str(&"A".repeat(SPACER));
        }
        let padded = cds.len().div_ceil(3) * 3;
        cds.push_str(&"A".repeat(padded - cds.len()));
        let probe = format!("{utr5}{cds}{utr3}");
        let window = CdsWindow::new(utr5.len(), cds.len() / 3);
        let fold = |rc| sequence_constrained_mfe(params, &probe, window, rc);
        let probed = match (
            fold(RegionConstraints::symmetric_isolation(Enforcement::Forbid)),
            fold(RegionConstraints::full_mutual_isolation(
                Enforcement::Forbid,
            )),
        ) {
            (Ok(clamped), Ok(apart)) => ((clamped - apart) * 100.0).round() as i32,
            _ => 0,
        };
        let prev = table[b - 1];
        let extended = if b >= 2 && prev < 0 {
            prev + intern
        } else {
            probed
        };
        table.push(probed.min(extended).min(0));
    }
    table
}

/// How a cell of [`rival_optimum`] was reached.
#[derive(Debug, Clone)]
enum Via {
    /// The leading run, spelled from the start node.
    Lead(Vec<Nucleotide>),
    /// One unmatched symbol after the same row's cell at `prev`.
    Unpaired { prev: Node },
    /// A top-level pair after the cell of row `row` at `prefix`.
    Pair { prefix: Node, row: usize },
    /// The first top-level pair, a head-cut class `class` of the cell `key`,
    /// after the leading run `lead`.
    Head {
        lead: Vec<Nucleotide>,
        key: PairKey,
        class: usize,
    },
}

type Cell = Option<(State, Via)>;

/// The best way found to end a parse: its value, cell and trailing run.
struct Closing {
    value: f64,
    class: usize,
    row: usize,
    node: Node,
    from_any: bool,
    trail: Vec<Nucleotide>,
    total: State,
    /// A last domain priced by the tail cut: its cell, class, the prefix row
    /// and the exterior cell it writes at `node`.
    tail: Option<(PairKey, usize, usize, State)>,
    /// When that domain is also the first and the head cut prices it: the
    /// head class whose value it took, where its floor was the higher.
    head_floor: Option<usize>,
    /// When that domain is also the first: the leading run it was priced
    /// after, traced directly (the table keeps only the best run's prefix).
    lead: Option<Vec<Nucleotide>>,
    /// A tail-priced last domain read after a junction run the wrap rival
    /// spared: the node the run starts at, whether that node's value is the
    /// `any` table's (else `paired`), and the run.
    junction: Option<Junction>,
}

/// A junction run before a last domain, and what precedes it.
#[derive(Debug, Clone)]
enum Junction {
    /// The run starts at a node whose value is the `any` table's (`true`)
    /// or the closed-pair table's.
    Table(Node, bool, Vec<Nucleotide>),
    /// The run follows a 5'-priced first domain: its cell (outer node and
    /// key), class, and the leading run before it.
    Head {
        u: Node,
        key: PairKey,
        class: usize,
        lead: Vec<Nucleotide>,
        run: Vec<Nucleotide>,
        /// With the junction cut: the cell's class under that cut, and
        /// whether its floor (not the head class's) set the value.
        cut: Option<(usize, bool)>,
        /// The last domain's class under the 5' junction cut whose floor set
        /// its value (above the caller's 3' cut class), if one did.
        w5: Option<usize>,
    },
    /// The run follows a domain priced by the junction cut at `u`: its cell,
    /// class (`None`: the cell is not a cut cell; the scan's value) and the
    /// prefix row before it.
    Cut {
        u: Node,
        key: PairKey,
        class: Option<usize>,
        row: usize,
        run: Vec<Nucleotide>,
        /// With the 5' wrap, a first domain (`row` 0) read after one
        /// enumerated leading run: that run, traced in place of the table's.
        lead: Option<Vec<Nucleotide>>,
        /// As for [`Junction::Head`]: the 5' junction cut's lifted class.
        w5: Option<usize>,
    },
}

impl Junction {
    fn run(&self) -> &[Nucleotide] {
        match self {
            Junction::Table(_, _, run) | Junction::Head { run, .. } | Junction::Cut { run, .. } => {
                run
            }
        }
    }
}

/// Nucleotide indices, as [`crate::dfa::Edge::nuc`] holds them, for
/// [`Dfa::spell_to`].
fn codes(nucs: &[Nucleotide]) -> Vec<u8> {
    nucs.iter().map(|&n| n as u8).collect()
}

/// The exterior path of a leading run from the start node, start first.
fn lead_path(dfa: &Dfa, lead: &[Nucleotide]) -> Vec<(Node, State)> {
    let mut nodes = vec![(
        dfa.start(),
        State::new(Energy::ZERO, 0.0, Back::ExteriorEmpty),
    )];
    let mut a = dfa.start();
    let mut cai = 0.0;
    for (e, &nuc) in dfa
        .spell_to(a, &codes(lead), None)
        .expect("lead edge")
        .into_iter()
        .zip(lead)
    {
        cai += e.cai;
        nodes.push((
            e.to,
            State::new(Energy::ZERO, cai, Back::ExteriorUnpaired { prev: a, nuc }),
        ));
        a = e.to;
    }
    nodes
}

/// The least class of the 5' cut cell `(o, key)` that no rival kills after
/// the leading run `lead` (`None`: every class falls) — the same test the
/// first-pair recurrence applies.
fn head_floor(
    hc: &TailCut,
    o: Node,
    key: &PairKey,
    lead: &[Nucleotide],
    ends: &mut EndGains<'_>,
    lambda: f64,
) -> Option<usize> {
    let classes = hc.classes(o, key)?;
    let mut order: Vec<usize> = (0..classes.len()).collect();
    order.sort_by(|&a, &b| {
        classes[a]
            .st
            .brd(lambda)
            .total_cmp(&classes[b].st.brd(lambda))
    });
    order.into_iter().find(|&i| {
        let c = &classes[i];
        let killed = hc
            .kills(c, lead, &mut |w| ends.gain(true, w), hc.max_open)
            .is_some()
            || hc.kills_invasion(c, lead, &mut |f, i, t| ends.invasion_gain(f, i, t))
            || (hc.split
                && hc
                    .kills_split(c, lead, &mut |f, first, ml| {
                        ends.split_gain(f, first, &[], ml)
                    })
                    .1)
            || (hc.own_repair
                && hc.kills_invasion(c, lead, &mut |f, i, t| ends.own_invasion_gain(f, i, t)));
        !killed
    })
}

/// Surviving 5' classes (index in the cell's list) with their least window
/// gain per depth.
type Survivors = Vec<(usize, Vec<Option<i32>>)>;

/// Every class of the 5' cut cell `(o, key)` that no rival kills after the
/// leading run `lead`, least first, each with its least window gain per
/// depth ([`TailCut::least_gains`]) — the input of the joint two-end test.
fn head_survivors(
    hc: &TailCut,
    o: Node,
    key: &PairKey,
    lead: &[Nucleotide],
    ends: &mut EndGains<'_>,
    lambda: f64,
) -> Survivors {
    let Some(classes) = hc.classes(o, key) else {
        return Vec::new();
    };
    let mut order: Vec<usize> = (0..classes.len()).collect();
    order.sort_by(|&a, &b| {
        classes[a]
            .st
            .brd(lambda)
            .total_cmp(&classes[b].st.brd(lambda))
    });
    let mut out = Vec::new();
    for i in order {
        let c = &classes[i];
        let killed = hc
            .kills(c, lead, &mut |w| ends.gain(true, w), hc.max_open)
            .is_some()
            || hc.kills_invasion(c, lead, &mut |f, i, t| ends.invasion_gain(f, i, t))
            || (hc.split
                && hc
                    .kills_split(c, lead, &mut |f, first, ml| {
                        ends.split_gain(f, first, &[], ml)
                    })
                    .1)
            || (hc.own_repair
                && hc.kills_invasion(c, lead, &mut |f, i, t| ends.own_invasion_gain(f, i, t)));
        if !killed {
            let g = hc.least_gains(c, lead, &mut |w| ends.gain(true, w), hc.max_open);
            out.push((i, g));
        }
    }
    out
}

/// Whether one removal kills every derivation in both the head class `h` and
/// the tail class `c` of a domain priced by both cuts.
///
/// On one derivation the pairs either cut removes are both a run of the
/// domain's outer chain from its closing pair inward, so their union is the
/// longer run, and its cost is at most the higher of the two classes'
/// removal bounds (`raw`, taken before the gain ceiling: a removal neither
/// window alone can pay for may be paid for by both). That removal frees the
/// first `t5` and the last `t3` symbols; the prefix refolds with the one, the
/// suffix with the other, side by side in the additive outermost loop, so the
/// gains add. A kill needs `max(raw_h, raw_c) < gain5 + gain3`, each gain the
/// least over the window's fills.
pub fn joint_kills(
    h: &crate::tail_cut::TailClass,
    g5: &[Option<i32>],
    c: &crate::tail_cut::TailClass,
    g3: &[Option<i32>],
) -> bool {
    joint_margin(h, g5, c, g3).is_some_and(|m| m < 0)
}

/// The least `max(raw_h, raw_c) - (gain5 + gain3)` over the depth pairs both
/// classes price and both windows can refold, or `None` when no pair is
/// priced; [`joint_kills`] is a negative margin.
pub fn joint_margin(
    h: &crate::tail_cut::TailClass,
    g5: &[Option<i32>],
    c: &crate::tail_cut::TailClass,
    g3: &[Option<i32>],
) -> Option<i32> {
    let unk = crate::tail_cut::UNKILLABLE;
    let ends = |raw: &[i32], g: &[Option<i32>]| -> Vec<(i32, i32)> {
        g.iter()
            .zip(raw)
            .filter_map(|(g, &r)| g.filter(|_| r != unk).map(|g| (r, g)))
            .collect()
    };
    let (e5, e3) = (ends(&h.raw, g5), ends(&c.raw, g3));
    e5.iter()
        .flat_map(|&(r5, a)| {
            e3.iter()
                .map(move |&(r3, b)| r5.max(r3) - a.saturating_add(b))
        })
        .min()
}

/// How the joint two-end rival met the closings it was offered.
#[derive(Clone, Copy, Debug, Default)]
pub struct JointStat {
    /// Closings that reached the joint test.
    pub reached: usize,
    /// Head survivors passed over (killed with the tail class before the
    /// least standing one).
    pub passed: usize,
    /// Closings refused outright (no head survivor stood; also counted in
    /// `tail_kills`).
    pub dead: usize,
    /// Reached closings whose least head survivor has no priced depth pair,
    /// by side: no 5' window refold, no 5' removal priced, and likewise 3'.
    pub no_gain5: usize,
    pub no_raw5: usize,
    pub no_gain3: usize,
    pub no_raw3: usize,
    /// The least margin ([`joint_margin`]) of a least head survivor.
    pub least_margin: Option<i32>,
}

impl JointStat {
    fn meet(
        &mut self,
        h: &crate::tail_cut::TailClass,
        g5: &[Option<i32>],
        c: &crate::tail_cut::TailClass,
        g3: &[Option<i32>],
    ) {
        let unk = crate::tail_cut::UNKILLABLE;
        let priced = |raw: &[i32], g: &[Option<i32>]| {
            g.iter().zip(raw).any(|(g, &r)| g.is_some() && r != unk)
        };
        self.reached += 1;
        self.no_gain5 += usize::from(g5.iter().all(Option::is_none));
        self.no_gain3 += usize::from(g3.iter().all(Option::is_none));
        self.no_raw5 += usize::from(g5.iter().any(Option::is_some) && !priced(&h.raw, g5));
        self.no_raw3 += usize::from(g3.iter().any(Option::is_some) && !priced(&c.raw, g3));
        if let Some(m) = joint_margin(h, g5, c, g3) {
            self.least_margin = Some(self.least_margin.map_or(m, |l| l.min(m)));
        }
    }
}

/// [`rival_optimum`]'s result: the optimum, and the closed-pair cells to
/// override before tracing it (a tail-cut class's derivation).
pub struct RivalOptimum {
    /// The optimum and its exterior path.
    pub opt: ClassOptimum,
    /// `(outer node, key, state)` to write into the scan's `P` table before
    /// [`crate::traceback::recover`] reads the path.
    pub overrides: Vec<(Node, PairKey, State)>,
    /// Closings refused because a tail cut killed them.
    pub tail_kills: usize,
    /// Of those, the ones only the context-free (own) invasion rival refused.
    pub own_kills: usize,
    /// The 5' cut class the optimum's first domain is: its cell, its index
    /// in the cell's list and the leading run before it.
    pub head_pick: Option<HeadPick>,
    /// Tail closings of a domain priced by both cuts whose head floor stood
    /// above the tail class (the lift was available, whether or not it won).
    pub head_floor_lifts: usize,
    /// Whether the optimum's own closing took its head floor.
    pub head_floor_optimum: bool,
    /// The 3' cut class the optimum's last domain closed through: its outer
    /// node, cell and index in the cell's list.
    pub tail_pick: Option<(Node, PairKey, usize)>,
    /// How the joint two-end rival met the closings it was offered.
    pub joint: JointStat,
    /// The split rival on one-domain items: rivals scored, closings it
    /// refused (no head class stood), and the largest `gain - delta` scored.
    pub split_tests: usize,
    pub split_kills: usize,
    pub split_best: Option<i32>,
    /// The junction the optimum's last domain was read after, if the wrap
    /// chose one.
    pub wrap_pick: Option<WrapPick>,
    /// Tail closings whose every junction run the 3' wrap rival killed.
    pub wrap_kills: usize,
    /// How the wrap rival met the junction runs it was offered.
    pub wrap_scored: WrapStat,
}

/// A 5' cut class on the optimum's route.
#[derive(Debug, Clone)]
pub struct HeadPick {
    /// The cell's outer node.
    pub outer: Node,
    /// The cell.
    pub key: PairKey,
    /// The class's index in the cell's list.
    pub class: usize,
    /// The leading run before the cell.
    pub lead: Vec<Nucleotide>,
}

/// The six symbol pairs a pair may close on, in the order a one-pair row
/// counts them.
const PAIR_TYPES: [(Nucleotide, Nucleotide); 6] = [
    (Nucleotide::A, Nucleotide::U),
    (Nucleotide::C, Nucleotide::G),
    (Nucleotide::G, Nucleotide::C),
    (Nucleotide::G, Nucleotide::U),
    (Nucleotide::U, Nucleotide::A),
    (Nucleotide::U, Nucleotide::G),
];

fn pair_type(a: Nucleotide, b: Nucleotide) -> usize {
    PAIR_TYPES
        .iter()
        .position(|&(x, y)| x == a && y == b)
        .unwrap_or(0)
}

/// Branch rows: `0` no pair yet; `1..=6` exactly one pair, by the symbols it
/// closes on (so the one-domain rival can read them); `b + 5` for `b >= 2`.
fn row_of(b: usize, t: usize) -> usize {
    match b {
        0 => 0,
        1 => 1 + t,
        b => b + 5,
    }
}

fn branches_of(row: usize) -> usize {
    match row {
        0 => 0,
        1..=6 => 1,
        r => r - 5,
    }
}

impl EndGains<'_> {
    /// The one-domain clamp's slack (centiunits) for an item whose own
    /// optimum is one domain closed on `closing`, with the whole leading run
    /// `l` and trailing run `r`: a probe `utr5 + l + domain + r + utr3` with a
    /// stand-in domain (the closing pair around a stable terminal loop) is
    /// folded; its optimum, with every pair touching the stand-in replaced by
    /// the stand-in's own optimum, is a configuration whose interior can be
    /// swapped for the item's own domain at the same closing pair and the
    /// same neighbouring symbols, so its cost less the stand-in's own minimum
    /// and the blocks' minima is the rival's slack. A stand-in that does not
    /// fold as one domain on `closing` gives no rival (`0`).
    pub fn single_domain(
        &mut self,
        l: &[Nucleotide],
        closing: (Nucleotide, Nucleotide),
        r: &[Nucleotide],
    ) -> i32 {
        const STEM: &str = "GCCGCGGCGAAAGCCGCGGC";
        let spell = |s: &[Nucleotide]| -> String {
            s.iter().map(|n| b"ACGU"[*n as usize] as char).collect()
        };
        let sym = |n: Nucleotide| b"ACGU"[n as usize] as char;
        let domain = format!("{}{STEM}{}", sym(closing.0), sym(closing.1));
        let key = (spell(l), domain.clone(), spell(r));
        if let Some(&g) = self.single.get(&key) {
            return g;
        }
        let g = (|| {
            let (own, own_e) = rna_mfe::fold(self.params, domain.as_bytes()).ok()?;
            if !(own.starts_with('(') && own.ends_with(')') && shape(&own).0 == 1) {
                return None;
            }
            let probe = format!("{}{}{domain}{}{}", self.utr5, key.0, key.2, self.utr3);
            let from = self.utr5.len() + key.0.len();
            let to = from + domain.len();
            let fold = fold_keeping(self.params, &probe, &[(from, to)])?;
            let mut dots: Vec<u8> = fold.into_bytes();
            let mut stack = Vec::new();
            let mut drop = Vec::new();
            for (j, &c) in dots.iter().enumerate() {
                match c {
                    b'(' => stack.push(j),
                    b')' => {
                        let i = stack.pop()?;
                        if (from..to).contains(&i) || (from..to).contains(&j) {
                            drop.push((i, j));
                        }
                    }
                    _ => {}
                }
            }
            for (i, j) in drop {
                dots[i] = b'.';
                dots[j] = b'.';
            }
            dots[from..to].copy_from_slice(own.as_bytes());
            let db = String::from_utf8(dots).ok()?;
            // Dropping pairs merges loops; a degree-2 loop past the span cap is
            // not a configuration of the model, so such a repair gives no rival.
            if !degree2_within_cap(&db) {
                return None;
            }
            let cost = rna_mfe::evaluate(self.params, probe.as_bytes(), &db).ok()?;
            Some((cost.0 - own_e.0 - self.alone5 - self.alone3).min(0))
        })()
        .unwrap_or(0);
        self.single.insert(key, g);
        g
    }
}

impl EndGains<'_> {
    /// The gain (centiunits) of the first-domain **invasion** refold: the
    /// prefix block folded with `five` (leading run + freed 5' strand), a
    /// stand-in domain closed on `inner`, and the freed 3' strand `three`, the
    /// pairs touching the stand-in replaced by its own optimum (as
    /// [`invasion_slack`]), less the stand-in's own minimum and the block's.
    /// `i32::MIN` (no rival) when the stand-in does not fold as one domain or
    /// the repaired probe breaks the degree-2 span cap.
    pub fn invasion_gain(
        &mut self,
        five: &[Nucleotide],
        inner: (Nucleotide, Nucleotide),
        three: &[Nucleotide],
    ) -> i32 {
        self.invasion_gain_against(self.utr5, "", self.alone5, "inv", five, inner, three)
    }

    /// The gain of the same invasion refold with **no context block**: the
    /// freed strands and the stand-in fold among themselves.
    ///
    /// In `d` the freed strands lie unmatched in the outermost loop and cost
    /// nothing, so the baseline is the stand-in's own minimum alone. A kill
    /// from this rival says `d` is not the item's own optimum, which holds
    /// whatever context the item sits in.
    ///
    /// Folding with a block is never worse than folding without one, so at the
    /// same `(five, inner, three)` this gain never exceeds
    /// [`Self::invasion_gain`] and kills a subset of what it kills. Where a
    /// block is adjacent it therefore adds only the positions at which the
    /// blocked probe was *refused* — the block lengthens the probe, which can
    /// break the degree-2 span cap the context-free one clears.
    pub fn own_invasion_gain(
        &mut self,
        five: &[Nucleotide],
        inner: (Nucleotide, Nucleotide),
        three: &[Nucleotide],
    ) -> i32 {
        self.invasion_gain_against("", "", 0, "own", five, inner, three)
    }

    /// The gain (centiunits) of the **3' wrap** refold, the 3' mirror of
    /// [`Self::invasion_gain`]: the strand `five` (exterior unmatched
    /// symbols of the item immediately 5' of its last domain), a stand-in domain
    /// closed on that domain's outer pair `inner`, the trailing run `three`
    /// and the suffix block, folded together; the pairs touching the
    /// stand-in replaced by its own optimum, less the stand-in's own minimum
    /// and the block's. The domain's inside can replace the stand-in's
    /// behind the same outer pair and neighbours, and `five` and `three`
    /// cost nothing unmatched in the outermost loop, so a positive gain is a
    /// rival of negative slack. `i32::MIN` (no rival) as for the invasion.
    pub fn wrap_gain(
        &mut self,
        five: &[Nucleotide],
        inner: (Nucleotide, Nucleotide),
        three: &[Nucleotide],
    ) -> i32 {
        self.invasion_gain_against("", self.utr3, self.alone3, "wrap", five, inner, three)
    }

    /// The gain (centiunits) of the **two-domain refold**: both context
    /// blocks folded with the whole two-domain item `lead · D1 · run · D2 ·
    /// trail`, each domain a stand-in closed on its outer pair (`first`,
    /// `last`) ([`Self::standin_gain`]). It holds the clamp around two
    /// branches, both wraps and every mix of them with the exposed runs.
    pub fn two_domain_gain(
        &mut self,
        lead: &[Nucleotide],
        first: (Nucleotide, Nucleotide),
        run: &[Nucleotide],
        last: (Nucleotide, Nucleotide),
        trail: &[Nucleotide],
    ) -> i32 {
        self.standin_gain(
            (true, true),
            &[
                Part::Run(lead),
                Part::Stand(first),
                Part::Run(run),
                Part::Stand(last),
                Part::Run(trail),
            ],
        )
    }

    /// The gain (centiunits) of the **split** refold of a one-domain item:
    /// both context blocks folded with `five` (the leading run and the freed
    /// 5' symbols), a stand-in on the first branch's outer pair `first`, a
    /// second stand-in, and the trailing run `trail`
    /// ([`crate::tail_cut::TailCut::kills_split`]). Nothing a pair of the
    /// probe can enclose without enclosing both stand-ins lies between them,
    /// so each sits in a loop with at least two branches, where at
    /// `dangles = 0` its outer pair adds only its terminal term, which its own
    /// minimum carries too: any branches, any count of two or more, and any
    /// unmatched symbols can take their place. Each further branch adds
    /// `ML.intern` (negative here) where the loop is a branching loop, so
    /// leaving them out only lowers the gain. With `trail` empty the probe
    /// leaves out everything after the stand-ins (the rest of the domain,
    /// every later domain, the trailing run): a probe pair around any of it
    /// opens before the stand-ins and so holds them both, the later domains
    /// keep their own cost, and the gain holds for every item and branch
    /// count behind the window.
    pub fn split_gain(
        &mut self,
        five: &[Nucleotide],
        first: (Nucleotide, Nucleotide),
        trail: &[Nucleotide],
        enclosed: bool,
    ) -> i32 {
        let g = self.standin_gain_if(
            (true, true),
            &[
                Part::Run(five),
                Part::Stand(first),
                Part::Stand((Nucleotide::G, Nucleotide::C)),
                Part::Run(trail),
            ],
            enclosed,
        );
        self.split_calls += 1;
        if g != i32::MIN {
            self.split_gain_best = Some(self.split_gain_best.map_or(g, |b| b.max(g)));
        }
        g
    }

    /// The gain (centiunits) of folding the item's exposed symbols `parts`
    /// with the prefix block (`with.0`) and/or the suffix block (`with.1`),
    /// each [`Part::Stand`] a stand-in domain closed on the given outer pair:
    /// the probe's optimum with the pairs touching a stand-in replaced by its
    /// own optimum, less the stand-ins' own minima and the blocks'. A domain
    /// whose outer pair and neighbours match a stand-in can replace its
    /// inside, and unmatched exterior symbols cost nothing, so a positive
    /// gain over the item's own cost of the same symbols is a rival of
    /// negative slack. `i32::MIN` (no rival) when a stand-in does not fold as
    /// one domain or the repaired probe breaks the degree-2 span cap.
    pub fn standin_gain(&mut self, with: (bool, bool), parts: &[Part<'_>]) -> i32 {
        self.standin_gain_if(with, parts, false)
    }

    /// [`Self::standin_gain`]; with `enclosed`, `i32::MIN` (no rival) unless
    /// the refold encloses every stand-in in a pair.
    fn standin_gain_if(&mut self, with: (bool, bool), parts: &[Part<'_>], enclosed: bool) -> i32 {
        const STEM: &str = "GCCGCGGCGAAAGCCGCGGC";
        let sym = |n: Nucleotide| b"ACGU"[n as usize] as char;
        let mut body = String::new();
        let mut spans = Vec::new();
        let pre = if with.0 { self.utr5 } else { "" };
        let post = if with.1 { self.utr3 } else { "" };
        for part in parts {
            match part {
                Part::Run(r) => body.extend(r.iter().map(|&n| sym(n))),
                Part::Stand((a, b)) => {
                    let from = pre.len() + body.len();
                    body.push(sym(*a));
                    body.push_str(STEM);
                    body.push(sym(*b));
                    spans.push((from, pre.len() + body.len()));
                }
            }
        }
        let key = (
            format!(
                "std:{}{}{}:{body}",
                u8::from(with.0),
                u8::from(with.1),
                u8::from(enclosed)
            ),
            String::new(),
            String::new(),
        );
        if let Some(&g) = self.single.get(&key) {
            return g;
        }
        let g = (|| {
            let probe = format!("{pre}{body}{post}");
            let mut own_total = 0;
            let mut owns = Vec::new();
            for &(a, b) in &spans {
                let (own, e) = rna_mfe::fold(self.params, &probe.as_bytes()[a..b]).ok()?;
                if !(own.starts_with('(') && own.ends_with(')') && shape(&own).0 == 1) {
                    return None;
                }
                own_total += e.0;
                owns.push(own);
            }
            let fold = fold_keeping(self.params, &probe, &spans)?;
            let mut dots = fold.into_bytes();
            let pf = partners(std::str::from_utf8(&dots).ok()?)?;
            let touches = |q: usize| spans.iter().any(|&(a, b)| (a..b).contains(&q));
            for i in 0..dots.len() {
                let j = pf[i];
                if j != usize::MAX && (touches(i) || touches(j)) {
                    dots[i] = b'.';
                    dots[j] = b'.';
                }
            }
            for (&(a, b), own) in spans.iter().zip(&owns) {
                dots[a..b].copy_from_slice(own.as_bytes());
            }
            let repaired = String::from_utf8(dots).ok()?;
            if !degree2_within_cap(&repaired) {
                return None;
            }
            if enclosed {
                let pr = partners(&repaired)?;
                let held =
                    |&(a, b): &(usize, usize)| (0..a).any(|i| pr[i] != usize::MAX && pr[i] >= b);
                if !spans.iter().all(held) {
                    return None;
                }
            }
            let cost = rna_mfe::evaluate(self.params, probe.as_bytes(), &repaired)
                .ok()?
                .0;
            let alone = if with.0 { self.alone5 } else { 0 } + if with.1 { self.alone3 } else { 0 };
            Some(alone + own_total - cost)
        })()
        .unwrap_or(i32::MIN);
        self.single.insert(key, g);
        g
    }

    #[allow(clippy::too_many_arguments)]
    fn invasion_gain_against(
        &mut self,
        block: &str,
        after: &str,
        alone: i32,
        kind: &str,
        five: &[Nucleotide],
        inner: (Nucleotide, Nucleotide),
        three: &[Nucleotide],
    ) -> i32 {
        const STEM: &str = "GCCGCGGCGAAAGCCGCGGC";
        let spell = |s: &[Nucleotide]| -> String {
            s.iter().map(|n| b"ACGU"[*n as usize] as char).collect()
        };
        let sym = |n: Nucleotide| b"ACGU"[n as usize] as char;
        let domain = format!("{}{STEM}{}", sym(inner.0), sym(inner.1));
        let key = (
            format!("{kind}:{}", spell(five)),
            domain.clone(),
            spell(three),
        );
        if let Some(&g) = self.single.get(&key) {
            return g;
        }
        let g = (|| {
            let (own, own_e) = rna_mfe::fold(self.params, domain.as_bytes()).ok()?;
            if !(own.starts_with('(') && own.ends_with(')') && shape(&own).0 == 1) {
                return None;
            }
            let probe = format!("{block}{}{domain}{}{after}", spell(five), spell(three));
            let from = block.len() + five.len();
            let to = from + domain.len();
            let fold = fold_keeping(self.params, &probe, &[(from, to)])?;
            let mut dots = fold.into_bytes();
            let pf = partners(std::str::from_utf8(&dots).ok()?)?;
            for i in 0..dots.len() {
                let j = pf[i];
                if j != usize::MAX && ((from..to).contains(&i) || (from..to).contains(&j)) {
                    dots[i] = b'.';
                    dots[j] = b'.';
                }
            }
            dots[from..to].copy_from_slice(own.as_bytes());
            let repaired = String::from_utf8(dots).ok()?;
            if !degree2_within_cap(&repaired) {
                return None;
            }
            let cost = rna_mfe::evaluate(self.params, probe.as_bytes(), &repaired)
                .ok()?
                .0;
            Some(alone + own_e.0 - cost)
        })()
        .unwrap_or(i32::MIN);
        self.single.insert(key, g);
        g
    }
}

/// The optimum of `probe` over the configurations in which no pair joins a
/// symbol of a stand-in span (`spans`, half-open) to a symbol outside it, in
/// dot-bracket: the stand-ins then sit in the loops around them as kept
/// domains do, so replacing their insides loses nothing. `None` when the probe
/// holds a non-nucleotide.
fn fold_keeping(params: &Params, probe: &str, spans: &[(usize, usize)]) -> Option<String> {
    fold_keeping_blind(params, probe, spans, &[])
}

/// [`fold_keeping`] with the symbols of the `blind` spans (half-open) left
/// unmatched: a gain read with a strand blind holds for every spelling of it
/// wherever an unmatched symbol costs nothing.
fn fold_keeping_blind(
    params: &Params,
    probe: &str,
    spans: &[(usize, usize)],
    blind: &[(usize, usize)],
) -> Option<String> {
    let nts: Vec<Nucleotide> = probe
        .bytes()
        .map(Nucleotide::from_byte)
        .collect::<Option<_>>()?;
    let span_of = |q: usize| spans.iter().position(|&(a, b)| (a..b).contains(&q));
    let is_blind = |q: usize| blind.iter().any(|&(a, b)| (a..b).contains(&q));
    let outside = |i: usize, j: usize| {
        if span_of(i) == span_of(j) && !is_blind(i) && !is_blind(j) {
            0
        } else {
            1 << 29
        }
    };
    let cut = rna_mfe::dp::Cut {
        threshold: 1 << 28,
        outside: &outside,
        inside: None,
    };
    match rna_mfe::dp::fold_sparse(params, &nts, Some(&cut)) {
        rna_mfe::dp::SparseFold::Below(_, fold) => Some(fold),
        rna_mfe::dp::SparseFold::AtOrAbove => None,
    }
}

/// The most unmatched symbols a degree-2 loop (one inner pair) may hold.
const DEGREE2_SPAN_CAP: usize = 30;

/// Whether every degree-2 loop of the dot-bracket configuration `db` holds at
/// most [`DEGREE2_SPAN_CAP`] unmatched symbols on its two sides together.
pub fn degree2_within_cap(db: &str) -> bool {
    let bytes = db.as_bytes();
    let mut partner = vec![usize::MAX; bytes.len()];
    let mut stack = Vec::new();
    for (j, &c) in bytes.iter().enumerate() {
        match c {
            b'(' => stack.push(j),
            b')' => {
                let Some(i) = stack.pop() else { return false };
                partner[i] = j;
                partner[j] = i;
            }
            _ => {}
        }
    }
    for i in 0..bytes.len() {
        if bytes[i] != b'(' {
            continue;
        }
        let j = partner[i];
        let (mut k, mut inner, mut count) = (i + 1, None, 0);
        while k < j {
            if bytes[k] == b'(' {
                count += 1;
                inner = Some((k, partner[k]));
                k = partner[k] + 1;
            } else {
                k += 1;
            }
        }
        if let (1, Some((p, q))) = (count, inner) {
            if (p - i - 1) + (j - q - 1) > DEGREE2_SPAN_CAP {
                return false;
            }
        }
    }
    true
}

/// The slack (centiunits) of the **cut rival** of an item `x` with own
/// configuration `d`: the prefix folded together with `x[..s]`, the suffix
/// folded together with `x[t..]`, and `d` kept on the middle with every pair
/// that touches `x[..s]` or `x[t..]` removed. The three parts are disjoint
/// spans side by side in the additive outermost loop, so this is a genuine
/// configuration of the whole word and its cost is the sum of the parts'
/// costs. `None` when removing those pairs leaves a degree-2 loop past the
/// span cap (not a configuration) or the reference evaluator rejects `d`.
pub fn cut_rival_slack(
    params: &Params,
    utr5: &str,
    utr3: &str,
    item: &str,
    d: &str,
    s: usize,
    t: usize,
) -> Option<i32> {
    if s > t || t > item.len() || d.len() != item.len() {
        return None;
    }
    let own = rna_mfe::evaluate(params, item.as_bytes(), d).ok()?.0;
    let mut kept: Vec<u8> = d.as_bytes().to_vec();
    let mut stack = Vec::new();
    for (j, &c) in d.as_bytes().iter().enumerate() {
        match c {
            b'(' => stack.push(j),
            b')' => {
                let i = stack.pop()?;
                if i < s || j >= t {
                    kept[i] = b'.';
                    kept[j] = b'.';
                }
            }
            _ => {}
        }
    }
    let kept = String::from_utf8(kept).ok()?;
    if !degree2_within_cap(&kept) {
        return None;
    }
    let middle = rna_mfe::evaluate(params, item.as_bytes(), &kept).ok()?.0;
    let head = mfe(params, &format!("{utr5}{}", &item[..s]));
    let tail = mfe(params, &format!("{}{utr3}", &item[t..]));
    Some(head + middle + tail - (mfe(params, utr5) + own + mfe(params, utr3)))
}

/// The partner of every position of a dot-bracket string (`usize::MAX` if
/// unmatched), or `None` when the brackets do not balance.
fn partners(db: &str) -> Option<Vec<usize>> {
    let mut p = vec![usize::MAX; db.len()];
    let mut stack = Vec::new();
    for (j, c) in db.bytes().enumerate() {
        match c {
            b'(' => stack.push(j),
            b')' => {
                let i = stack.pop()?;
                p[i] = j;
                p[j] = i;
            }
            _ => {}
        }
    }
    stack.is_empty().then_some(p)
}

/// The degree-2 chain of a domain: its closing pair, then while the current
/// pair closes a loop with exactly one inner pair, that inner pair.
fn outer_chain(db: &[u8], p: &[usize], open: usize) -> Vec<(usize, usize)> {
    let mut chain = vec![(open, p[open])];
    loop {
        let (i, j) = *chain.last().unwrap();
        let (mut k, mut inner) = (i + 1, Vec::new());
        while k < j {
            if db[k] == b'(' {
                inner.push((k, p[k]));
                k = p[k] + 1;
            } else {
                k += 1;
            }
        }
        match inner.as_slice() {
            [one] => chain.push(*one),
            _ => return chain,
        }
    }
}

/// Which end domain an [`invasion_slack`] rival opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndDomain {
    /// The first top-level domain, invaded by the prefix (the rest of the
    /// item and the suffix stay in the outermost loop).
    First,
    /// The last top-level domain, invaded by the suffix.
    Last,
    /// A one-domain item, invaded from both ends at once.
    Only,
}

/// The slack (centiunits) of the **invasion rival** at depth `depth` of an
/// item `x` with own configuration `d`: the outermost `depth` pairs of an end
/// domain's degree-2 chain are removed, the domain's remaining inner pair and
/// everything inside it are kept, and the context block(s) on that side fold
/// freely together with the freed strands and the end run around it.
///
/// Computed as [`EndGains::single_domain`] does: a probe with a stand-in
/// domain closed on the inner pair's symbols is folded, the pairs touching the
/// stand-in are replaced by its own optimum, and the loops outside the inner
/// pair see only its symbols and their outer neighbours, so swapping the
/// stand-in's inside for the real inside changes the cost by exactly
/// (real inside − stand-in inside). `None` when the domain's chain is shorter
/// than `depth`, the stand-in does not fold as one domain, or the repaired
/// probe has a degree-2 loop past the span cap.
pub fn invasion_slack(
    params: &Params,
    utr5: &str,
    utr3: &str,
    item: &str,
    d: &str,
    side: EndDomain,
    depth: usize,
) -> Option<i32> {
    const STEM: &str = "GCCGCGGCGAAAGCCGCGGC";
    let p = partners(d)?;
    let db = d.as_bytes();
    let tops: Vec<usize> = {
        let (mut v, mut k) = (Vec::new(), 0);
        while k < db.len() {
            if db[k] == b'(' {
                v.push(k);
                k = p[k] + 1;
            } else {
                k += 1;
            }
        }
        v
    };
    let open = match side {
        EndDomain::First => *tops.first()?,
        EndDomain::Last => *tops.last()?,
        EndDomain::Only if tops.len() == 1 => tops[0],
        EndDomain::Only => return None,
    };
    let chain = outer_chain(db, &p, open);
    let &(a, b) = chain.get(depth)?;
    let (j0, i0) = (p[open], open);
    // The span the rival refolds, as item coordinates [lo, hi), and the
    // context blocks it includes.
    let (lo, hi, with5, with3) = match side {
        EndDomain::First => (0, j0 + 1, true, false),
        EndDomain::Last => (i0, item.len(), false, true),
        EndDomain::Only => (0, item.len(), true, true),
    };
    let (c5, c3) = (if with5 { utr5 } else { "" }, if with3 { utr3 } else { "" });
    let x = item.as_bytes();
    let domain = format!("{}{STEM}{}", x[a] as char, x[b] as char);
    let (own, own_e) = rna_mfe::fold(params, domain.as_bytes()).ok()?;
    if !(own.starts_with('(') && own.ends_with(')') && shape(&own).0 == 1) {
        return None;
    }
    let probe = format!("{c5}{}{domain}{}{c3}", &item[lo..a], &item[b + 1..hi]);
    let from = c5.len() + (a - lo);
    let to = from + domain.len();
    let fold = fold_keeping(params, &probe, &[(from, to)])?;
    let mut dots = fold.into_bytes();
    let pf = partners(std::str::from_utf8(&dots).ok()?)?;
    for i in 0..dots.len() {
        let j = pf[i];
        if j != usize::MAX && ((from..to).contains(&i) || (from..to).contains(&j)) {
            dots[i] = b'.';
            dots[j] = b'.';
        }
    }
    dots[from..to].copy_from_slice(own.as_bytes());
    let repaired = String::from_utf8(dots).ok()?;
    if !degree2_within_cap(&repaired) {
        return None;
    }
    let probe_cost = rna_mfe::evaluate(params, probe.as_bytes(), &repaired)
        .ok()?
        .0;
    let inside = rna_mfe::evaluate(params, &x[a..=b], &d[a..=b]).ok()?.0;
    let span = rna_mfe::evaluate(params, &x[lo..hi], &d[lo..hi]).ok()?.0;
    let blocks =
        if with5 { mfe(params, utr5) } else { 0 } + if with3 { mfe(params, utr3) } else { 0 };
    Some(probe_cost - own_e.0 + inside - span - blocks)
}

/// The slack (centiunits) of the **split rival** of an item `x` with own
/// configuration `d`: every pair of an end domain's degree-2 outer chain is
/// removed together with the closing pair of the branching loop the chain
/// ends in, so that loop's branches stand beside each other in the loop
/// around them; the context block(s) on that side fold freely with the
/// freed strands, the loop's unmatched runs and the end run around the
/// branches.
///
/// Computed as [`invasion_slack`] is, with one stand-in per branch (closed on
/// the branch's outer pair) under [`fold_keeping_blind`]. With `blind >= 1`,
/// the runs between consecutive branches are left unmatched in the probe:
/// every pair around them also holds the stand-ins on both sides, so they lie
/// in a loop with at least two branches, where an unmatched symbol costs
/// nothing, and the gain read holds for every spelling of them. With
/// `blind >= 2`, so is the domain's interior on the side away from the cut
/// end (`First` / `Only`: from the last branch to the domain's closing symbol;
/// `Last`: from the domain's opening to the first branch), by the same
/// argument: a pair around it holds a stand-in, or none exists. With
/// `blind >= 3` (`First` / `Only`), everything after the branches is left out
/// (the rest of the domain, every later domain, the trailing run) and both
/// blocks refold: a probe pair around any of it opens before the stand-ins,
/// so it holds them both; the later domains add their own cost, terminal
/// term included, plus `ML.intern` (negative) where the loop branches. `None`
/// when the chain ends in a terminal loop, a stand-in does not fold as one
/// domain, or the repaired probe has a degree-2 loop past the span cap.
pub fn split_slack(
    params: &Params,
    utr5: &str,
    utr3: &str,
    item: &str,
    d: &str,
    side: EndDomain,
    blind: u8,
) -> Option<i32> {
    const STEM: &str = "GCCGCGGCGAAAGCCGCGGC";
    let p = partners(d)?;
    let db = d.as_bytes();
    let mut tops = Vec::new();
    let mut k = 0;
    while k < db.len() {
        if db[k] == b'(' {
            tops.push(k);
            k = p[k] + 1;
        } else {
            k += 1;
        }
    }
    let open = match side {
        EndDomain::First => *tops.first()?,
        EndDomain::Last => *tops.last()?,
        EndDomain::Only if tops.len() == 1 => tops[0],
        EndDomain::Only => return None,
    };
    let chain = outer_chain(db, &p, open);
    let &(ci, cj) = chain.last()?;
    let mut branches = Vec::new();
    let mut k = ci + 1;
    while k < cj {
        if db[k] == b'(' {
            branches.push((k, p[k]));
            k = p[k] + 1;
        } else {
            k += 1;
        }
    }
    if branches.len() < 2 {
        return None;
    }
    let (j0, i0) = (p[open], open);
    let open_rest = blind >= 3 && side != EndDomain::Last;
    let (lo, hi, with5, with3) = match side {
        _ if open_rest => (0, item.len(), true, true),
        EndDomain::First => (0, j0 + 1, true, false),
        EndDomain::Last => (i0, item.len(), false, true),
        EndDomain::Only => (0, item.len(), true, true),
    };
    let (c5, c3) = (if with5 { utr5 } else { "" }, if with3 { utr3 } else { "" });
    let x = item.as_bytes();
    let mut probe = String::from(c5);
    let mut spans = Vec::new();
    let mut blinds = Vec::new();
    let mut owns = Vec::new();
    let mut stand_e = 0;
    let mut at = lo;
    for (n, &(a, b)) in branches.iter().enumerate() {
        let from = probe.len();
        if n == 0 && blind >= 2 && side == EndDomain::Last {
            probe.push_str(&item[at..i0]);
            let from = probe.len();
            probe.push_str(&item[i0..a]);
            blinds.push((from, probe.len()));
        } else {
            probe.push_str(&item[at..a]);
        }
        if blind >= 1 && n > 0 {
            blinds.push((from, probe.len()));
        }
        let domain = format!("{}{STEM}{}", x[a] as char, x[b] as char);
        let (own, e) = rna_mfe::fold(params, domain.as_bytes()).ok()?;
        if !(own.starts_with('(') && own.ends_with(')') && shape(&own).0 == 1) {
            return None;
        }
        stand_e += e.0;
        spans.push((probe.len(), probe.len() + domain.len()));
        probe.push_str(&domain);
        owns.push(own);
        at = b + 1;
    }
    if open_rest {
        at = hi;
    } else if blind >= 2 && side != EndDomain::Last {
        let from = probe.len();
        probe.push_str(&item[at..=j0]);
        blinds.push((from, probe.len()));
        at = j0 + 1;
    }
    probe.push_str(&item[at..hi]);
    probe.push_str(c3);
    let fold = fold_keeping_blind(params, &probe, &spans, &blinds)?;
    let mut dots = fold.into_bytes();
    let pf = partners(std::str::from_utf8(&dots).ok()?)?;
    let touches = |q: usize| spans.iter().any(|&(a, b)| (a..b).contains(&q));
    for i in 0..dots.len() {
        let j = pf[i];
        if j != usize::MAX && (touches(i) || touches(j)) {
            dots[i] = b'.';
            dots[j] = b'.';
        }
    }
    for (&(a, b), own) in spans.iter().zip(&owns) {
        dots[a..b].copy_from_slice(own.as_bytes());
    }
    let repaired = String::from_utf8(dots).ok()?;
    if !degree2_within_cap(&repaired) {
        return None;
    }
    let probe_cost = rna_mfe::evaluate(params, probe.as_bytes(), &repaired)
        .ok()?
        .0;
    let mut inside = 0;
    for &(a, b) in &branches {
        inside += rna_mfe::evaluate(params, &x[a..=b], &d[a..=b]).ok()?.0;
    }
    let span = rna_mfe::evaluate(params, &x[lo..hi], &d[lo..hi]).ok()?.0;
    if open_rest {
        // Every other domain stays as it is in the outermost loop.
        inside += span - rna_mfe::evaluate(params, &x[i0..=j0], &d[i0..=j0]).ok()?.0;
    }
    let blocks =
        if with5 { mfe(params, utr5) } else { 0 } + if with3 { mfe(params, utr3) } else { 0 };
    Some(probe_cost - stand_e + inside - span - blocks)
}

/// `[lead class][branch row][node]`.
type Grid = Vec<Vec<Vec<Cell>>>;

/// Non-vacuity counters of the 3' wrap rival.
#[derive(Debug, Clone, Copy, Default)]
pub struct WrapStat {
    /// Junction runs scored (lazily, in value order).
    pub scored: usize,
    /// Candidates read after a 5'-priced first domain (second phase).
    pub head_cands: usize,
    /// Invasion strands of such a class tried with the wrap.
    pub inv_tests: usize,
    /// The largest `gain - delta` over those (positive kills).
    pub inv_best: Option<i32>,
    /// Spared candidates that came after a 5'-priced first domain.
    pub head_spared: usize,
    /// Junction-cut groups expanded, classes tested, and killed by the cut.
    pub cut_groups: usize,
    pub cut_tests: usize,
    pub cut_kills: usize,
    /// (Head class, junction-cut class) pairs' slip rivals scored, and the
    /// pairs they killed ([`crate::tail_cut::TailCut::slip`]).
    pub slip_tests: usize,
    pub slip_kills: usize,
    /// The largest `gain - delta` over those (positive kills).
    pub slip_best: Option<i32>,
    /// 5' wrap ([`crate::tail_cut::TailCut::wrap5`]): junction runs scored
    /// alone, the ones that killed, and the largest gain scored.
    pub w5_tests: usize,
    pub w5_kills: usize,
    pub w5_best: Option<i32>,
    /// 5' junction cut: candidates whose last domain it priced, and the ones
    /// every class of which it killed.
    pub w5_cut_tests: usize,
    pub w5_cut_kills: usize,
    /// Spared candidates the 5' junction cut's floor raised.
    pub w5_lifts: usize,
    /// The last domain's invasion rivals with both blocks: rivals scored and
    /// the largest `gain - delta` (positive kills).
    pub w5_inv_tests: usize,
    pub w5_inv_best: Option<i32>,
    /// The first domain's invasion rivals with both blocks: rivals scored,
    /// candidates killed, and the largest `gain - delta`.
    pub w5_inv1_tests: usize,
    pub w5_inv1_kills: usize,
    pub w5_inv1_best: Option<i32>,
    /// The two-domain refold: candidates scored, killed, largest gain.
    pub w5_both_tests: usize,
    pub w5_both_kills: usize,
    pub w5_both_best: Option<i32>,
    /// Expanded candidates below their group's bound (must stay 0: the
    /// best-first order assumes a recomputed cell never reads below the scan).
    pub order_breaks: usize,
}

/// The junction the optimum took: diagnostic only.
#[derive(Debug, Clone)]
pub struct WrapPick {
    /// `table`, `head` or `cut`.
    pub kind: &'static str,
    /// The junction's start and the cell before it (opening, closing).
    pub start: u32,
    pub cell: Option<(u32, u32)>,
    /// The junction-cut class taken, and whether its floor set the value.
    pub cut_class: Option<usize>,
    pub by_cut: bool,
    /// That class's value, its reasons per depth, and the scan's value of the
    /// cell (a class never reads below the scan unless something is off).
    pub class_brd: Option<f64>,
    pub why: Vec<&'static str>,
    pub scan_brd: Option<f64>,
}

/// What [`wrap_prefix`] found.
enum WrapPrefix {
    /// The least prefix whose junction run the wrap rival spares.
    Spared(State, Junction),
    /// The rival kills every junction run.
    Killed,
    /// None spared below the limit (some may be above it).
    NoBetter,
}

/// The junction cut's inputs: the model, the scan's tables, its depth, the
/// cuts already built (by junction start), the row count and the row a pair
/// moves a prefix row to (`None`: it may not follow).
struct JunctionCut<'a, 'b> {
    params: &'a Params,
    tables: &'a Tables,
    depth: usize,
    cuts: &'b mut HashMap<u32, TailCut>,
    rows: usize,
    next_row: &'b dyn Fn(usize, &PairKey) -> Option<usize>,
}

/// The 5' wrap's inputs: the longest junction run it scores alone, the 5'
/// junction cut's depth, the model, the scan's tables, the last domain's
/// outer node and the value the caller prices it at (its 3' cut class), and
/// the junction cuts built so far (by junction end).
struct Wrap5<'a, 'b> {
    run: usize,
    depth: usize,
    /// Also the two-domain refold, with this trailing run.
    both: Option<&'b [Nucleotide]>,
    /// With the junction cut: also each class's invasion rivals of the last
    /// domain, refolded with both blocks; the trailing run.
    inv: Option<&'b [Nucleotide]>,
    params: &'a Params,
    tables: &'a Tables,
    outer: Node,
    last_st: State,
    cuts: &'b mut HashMap<u32, TailCut>,
}

/// What [`Wrap5::kills`] found.
enum W5 {
    /// Every derivation of the candidate falls.
    Killed,
    /// Some stand: with the junction cut, the least surviving class of the
    /// last domain (value and index) when it sits above the caller's value.
    Spared(Option<(State, usize)>),
}

impl Wrap5<'_, '_> {
    /// Whether the 5' wrap kills a junction candidate: the first domain
    /// `key1` after the leading run `lead`, then the junction run `run`,
    /// then the last domain `last`.
    ///
    /// The prefix block folds with `lead`, a stand-in closed on `key1`'s
    /// outer pair and `run` ([`EndGains::invasion_gain`]); the first domain's
    /// inside can replace the stand-in's behind the same pair and neighbours,
    /// and the last domain, the trailing run and the suffix block stay as
    /// they are beside it, so a positive gain is a rival of negative slack.
    /// With the junction cut, `last` is priced by a 5' cut anchored at its
    /// opening: each class frees its first `t` symbols for `delta_t`, appended
    /// to `run` in the refold. Every derivation of `last` lies in one class,
    /// so the candidate falls when every class does; otherwise the least
    /// surviving class is a floor on the last domain, the same argument as
    /// the both-sided floor.
    #[allow(clippy::too_many_arguments)]
    fn kills(
        &mut self,
        dfa: &Dfa,
        lead: &[Nucleotide],
        key1: &PairKey,
        run: &[Nucleotide],
        last: &PairKey,
        ends: &mut EndGains<'_>,
        stat: &mut WrapStat,
    ) -> W5 {
        let inner = (key1.nuc_open(), key1.nuc_close);
        if let Some(trail) = self.both {
            stat.w5_both_tests += 1;
            let g =
                ends.two_domain_gain(lead, inner, run, (last.nuc_open(), last.nuc_close), trail);
            if g != i32::MIN {
                stat.w5_both_best = Some(stat.w5_both_best.map_or(g, |b| b.max(g)));
            }
            if g > 0 {
                stat.w5_both_kills += 1;
                return W5::Killed;
            }
        }
        if self.run > 0 && run.len() <= self.run {
            stat.w5_tests += 1;
            let g = ends.invasion_gain(lead, inner, run);
            if g != i32::MIN {
                stat.w5_best = Some(stat.w5_best.map_or(g, |b| b.max(g)));
            }
            if g > 0 {
                stat.w5_kills += 1;
                return W5::Killed;
            }
        }
        if self.depth == 0 {
            return W5::Spared(None);
        }
        let (tables, depth) = (self.tables, self.depth);
        let with_inv = self.inv.is_some();
        let cut = self.cuts.entry(last.open.pos).or_insert_with(|| {
            let mut c = TailCut::lazy_head_at(dfa, tables, depth, last.open.pos);
            c.max_open = DEFAULT_MAX_OPEN_WRAP;
            // Invasion rivals through loops with unmatched 3' symbols too:
            // those symbols are left open and filled by enumeration, so a
            // rival with more than the fills cover is not recorded. A small
            // group cap keeps the classes few (a capped group's floor is
            // never killed, which only weakens the cut).
            c.slip = with_inv;
            c.inv_open_cap = INV_FILL_OPEN;
            c.group_cap = Some(ANCHORED_GROUP_CAP);
            c
        });
        if !cut.is_tail(last) {
            return W5::Spared(None);
        }
        cut.price(self.params, dfa, tables, self.outer, *last);
        let Some(classes) = cut.classes(self.outer, last) else {
            return W5::Spared(None);
        };
        stat.w5_cut_tests += 1;
        let lambda = tables.lambda();
        let inv = self.inv;
        // Least value first: the first class standing is the floor, and the
        // candidate is spared, so nothing above it needs testing.
        let mut order: Vec<usize> = (0..classes.len()).collect();
        order.sort_by(|&a, &b| {
            classes[a]
                .st
                .brd(lambda)
                .total_cmp(&classes[b].st.brd(lambda))
        });
        let floor = order
            .into_iter()
            .map(|i| (i, &classes[i]))
            .find(|(_, c)| {
                let by_window = cut
                    .kills(
                        c,
                        &[],
                        &mut |w| {
                            let mut three = run.to_vec();
                            three.extend_from_slice(w);
                            ends.invasion_gain(lead, inner, &three)
                        },
                        cut.max_open,
                    )
                    .is_some();
                let by_inv = !by_window
                    && inv.is_some_and(|trail| {
                        let (tests, k, best) = last_invasion_kills(
                            cut,
                            c,
                            (lead, inner, run),
                            trail,
                            last.close.pos,
                            ends,
                        );
                        stat.w5_inv_tests += tests;
                        if let Some(b) = best {
                            stat.w5_inv_best = Some(stat.w5_inv_best.map_or(b, |m| m.max(b)));
                        }
                        k
                    });
                !by_window && !by_inv
            })
            .map(|(i, c)| (c.st, i));
        match floor {
            None => {
                stat.w5_cut_kills += 1;
                W5::Killed
            }
            Some(f) if f.0.brd(lambda) > self.last_st.brd(lambda) + 1e-9 => {
                stat.w5_lifts += 1;
                W5::Spared(Some(f))
            }
            Some(_) => W5::Spared(None),
        }
    }
}

/// The most unspelled symbols an invasion rival of the last domain may hold
/// and still be scored (each costs a factor of four in two refolds).
const INV_FILL_OPEN: usize = 2;

/// The per-spelling class cap of the 5' cut anchored at a junction.
const ANCHORED_GROUP_CAP: usize = 32;

/// Whether some invasion rival of the last domain's class `c` (a 5' cut
/// anchored at its opening) kills it after the first domain: the domain's
/// outer chain opened down to a kept pair at `x.a`, the freed 5' strand (the
/// class's window from the domain's opening to `a`) and 3' strand
/// (`x.three`, ending at the domain's closing symbol `close`, then `trail`)
/// refolded with the leading run, a stand-in on the first domain, the
/// junction run and a stand-in on the kept pair, against the prefix block
/// alone or both blocks; the kept pair's inside and the first domain stay
/// whole. Unspelled symbols take the least gain over their fills (at most
/// [`INV_FILL_OPEN`] of them). Returns the rivals scored, whether one
/// killed (`delta < gain`), and the largest `gain - delta` scored.
fn last_invasion_kills(
    cut: &TailCut,
    c: &crate::tail_cut::TailClass,
    (lead, first, run): (&[Nucleotide], (Nucleotide, Nucleotide), &[Nucleotide]),
    trail: &[Nucleotide],
    close: u32,
    ends: &mut EndGains<'_>,
) -> (usize, bool, Option<i32>) {
    const NUC: [Nucleotide; 4] = [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U];
    let (mut tests, mut best) = (0, None::<i32>);
    let start = cut.first();
    for x in &c.inv {
        if x.sealed || x.ml || x.delta == crate::tail_cut::UNKILLABLE || x.a < start {
            continue;
        }
        if x.delta < 0 {
            return (tests + 1, true, best);
        }
        let k = (x.a - start) as usize;
        let Some(five) = c.spell.get(..k) else {
            continue;
        };
        if five.contains(&crate::tail_cut::PAST) || x.inner.0 > 3 || x.inner.1 > 3 {
            continue;
        }
        // The 3' strand must reach the domain's closing symbol.
        if close + 1 < x.three.len() as u32 {
            continue;
        }
        let mut word: Vec<u8> = five.to_vec();
        word.extend_from_slice(&x.three);
        let open: Vec<usize> = (0..word.len()).filter(|&i| word[i] > 3).collect();
        if open.len() > INV_FILL_OPEN {
            continue;
        }
        let inner = (NUC[x.inner.0 as usize], NUC[x.inner.1 as usize]);
        let mut least = i32::MAX;
        for fill in 0..(1usize << (2 * open.len())) {
            let mut w = word.clone();
            for (b, &i) in open.iter().enumerate() {
                w[i] = ((fill >> (2 * b)) & 3) as u8;
            }
            let nucs: Vec<Nucleotide> = w.iter().map(|&v| NUC[v as usize]).collect();
            let (f5, f3) = nucs.split_at(k);
            let mut run5 = run.to_vec();
            run5.extend_from_slice(f5);
            let mut run3 = f3.to_vec();
            let prefix_only = ends.standin_gain(
                (true, false),
                &[
                    Part::Run(lead),
                    Part::Stand(first),
                    Part::Run(&run5),
                    Part::Stand(inner),
                    Part::Run(&run3),
                ],
            );
            run3.extend_from_slice(trail);
            let both = ends.standin_gain(
                (true, true),
                &[
                    Part::Run(lead),
                    Part::Stand(first),
                    Part::Run(&run5),
                    Part::Stand(inner),
                    Part::Run(&run3),
                ],
            );
            least = least.min(prefix_only.max(both));
            if least <= x.delta {
                break;
            }
        }
        if least == i32::MIN {
            continue;
        }
        tests += 1;
        best = Some(best.map_or(least - x.delta, |b| b.max(least - x.delta)));
        if x.delta < least {
            return (tests, true, best);
        }
    }
    (tests, false, best)
}

/// Whether some invasion rival of the first domain's head class `c` kills
/// it with the last domain kept: the first domain's outer chain opened to a
/// kept pair at `x.a`, the leading run, the freed 5' strand (`c`'s spelling
/// up to `a`), a stand-in on the kept pair, the freed 3' strand `x.three`,
/// the junction run, a stand-in on the last domain's outer pair `last` and
/// the trailing run, refolded against the prefix block alone (up to the
/// junction run) or both blocks. Unspelled symbols take the least gain over
/// their fills (at most [`INV_FILL_OPEN`]). Returns the rivals scored,
/// whether one killed (`delta < gain`) and the largest `gain - delta`.
fn first_invasion_kills(
    c: &crate::tail_cut::TailClass,
    (lead, run): (&[Nucleotide], &[Nucleotide]),
    last: (Nucleotide, Nucleotide),
    trail: &[Nucleotide],
    ends: &mut EndGains<'_>,
) -> (usize, bool, Option<i32>) {
    const NUC: [Nucleotide; 4] = [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U];
    let (mut tests, mut best) = (0, None::<i32>);
    for x in &c.inv {
        if x.ml || x.delta == crate::tail_cut::UNKILLABLE || x.inner.0 > 3 || x.inner.1 > 3 {
            continue;
        }
        if x.delta < 0 {
            return (tests + 1, true, best);
        }
        let a = x.a as usize;
        if a < lead.len() {
            continue;
        }
        if x.sealed {
            // The split ([`crate::tail_cut::TailCut::kills_split`]): the first
            // domain's chain and branching loop opened, its branches kept; the
            // window spelled up to the first unspelled symbol, two stand-ins,
            // and the domain's 3' side left out (each such symbol lies in a
            // loop holding both stand-ins).
            let mut head5 = lead.to_vec();
            head5.extend(
                c.spell[lead.len().min(c.spell.len())..a.min(c.spell.len())]
                    .iter()
                    .take_while(|&&v| v < 4)
                    .map(|&v| NUC[v as usize]),
            );
            let inner = (NUC[x.inner.0 as usize], NUC[x.inner.1 as usize]);
            let gc = (Nucleotide::G, Nucleotide::C);
            let prefix_only = ends.standin_gain(
                (true, false),
                &[
                    Part::Run(&head5),
                    Part::Stand(inner),
                    Part::Stand(gc),
                    Part::Run(run),
                ],
            );
            let both = ends.standin_gain(
                (true, true),
                &[
                    Part::Run(&head5),
                    Part::Stand(inner),
                    Part::Stand(gc),
                    Part::Run(run),
                    Part::Stand(last),
                    Part::Run(trail),
                ],
            );
            let g = prefix_only.max(both);
            if g == i32::MIN {
                continue;
            }
            tests += 1;
            best = Some(best.map_or(g - x.delta, |b| b.max(g - x.delta)));
            if x.delta < g {
                return (tests, true, best);
            }
            continue;
        }
        let Some(five) = c.spell.get(lead.len()..a) else {
            continue;
        };
        if five.contains(&crate::tail_cut::PAST) {
            continue;
        }
        let k = five.len();
        let mut word: Vec<u8> = five.to_vec();
        word.extend_from_slice(&x.three);
        let open: Vec<usize> = (0..word.len()).filter(|&i| word[i] > 3).collect();
        if open.len() > INV_FILL_OPEN {
            continue;
        }
        let inner = (NUC[x.inner.0 as usize], NUC[x.inner.1 as usize]);
        let mut least = i32::MAX;
        for fill in 0..(1usize << (2 * open.len())) {
            let mut w = word.clone();
            for (b, &i) in open.iter().enumerate() {
                w[i] = ((fill >> (2 * b)) & 3) as u8;
            }
            let nucs: Vec<Nucleotide> = w.iter().map(|&v| NUC[v as usize]).collect();
            let (f5, f3) = nucs.split_at(k);
            let mut head5 = lead.to_vec();
            head5.extend_from_slice(f5);
            let mut mid = f3.to_vec();
            mid.extend_from_slice(run);
            let prefix_only = ends.standin_gain(
                (true, false),
                &[Part::Run(&head5), Part::Stand(inner), Part::Run(&mid)],
            );
            let both = ends.standin_gain(
                (true, true),
                &[
                    Part::Run(&head5),
                    Part::Stand(inner),
                    Part::Run(&mid),
                    Part::Stand(last),
                    Part::Run(trail),
                ],
            );
            least = least.min(prefix_only.max(both));
            if least <= x.delta {
                break;
            }
        }
        if least == i32::MIN {
            continue;
        }
        tests += 1;
        best = Some(best.map_or(least - x.delta, |b| b.max(least - x.delta)));
        if x.delta < least {
            return (tests, true, best);
        }
    }
    (tests, false, best)
}

/// A heap key: least value first.
#[derive(PartialEq)]
struct Least(f64, usize);

impl Eq for Least {}

impl PartialOrd for Least {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for Least {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.0.total_cmp(&self.0).then(o.1.cmp(&self.1))
    }
}

/// A candidate prefix, or a group of them not yet expanded.
enum Entry {
    /// Every derivation ending in a pair closed at `u` and the run: bounded
    /// below by the closed-pair table.
    Group(Node, Vec<Nucleotide>, f64),
    /// One cell closing at the group's node after one prefix row: bounded
    /// below by the scan's value of the cell, priced only when popped.
    Cell(Node, PairKey, usize, Vec<Nucleotide>, f64),
    Cand(State, Junction),
    /// A candidate every rival spared, raised to the 5' junction cut's floor
    /// on the last domain: final when popped.
    Lifted(State, Junction),
    Done,
}

/// The least exterior prefix reaching the last domain `key` in row `from`
/// whose junction run the 3' wrap rival spares, after the trailing run
/// `trail`. Every derivation reaching `key.open` ends in a run of exactly
/// `r < depth` unmatched symbols after a closed pair (the `paired` table `r`
/// symbols back) or in at least `depth` of them (the `any` table `depth`
/// back), so the runs read here cover every derivation; each is scored on
/// its own symbols. With the junction cut, a closed-pair entry is a group,
/// expanded (only when it is the least entry left) into every cell closing
/// there, every prefix row and every class of the cell under a 3' cut
/// anchored at the junction; each class value is at least the group's, so
/// the order holds. Candidates are tried in order of value and the search
/// stops at the first spared one or at `limit`.
#[allow(clippy::too_many_arguments)]
fn wrap_prefix(
    dfa: &Dfa,
    (any, paired): (&[Vec<Cell>], &[Vec<Cell>]),
    head: Option<WrapHead<'_>>,
    mut jc: Option<JunctionCut<'_, '_>>,
    mut w5: Option<Wrap5<'_, '_>>,
    from: usize,
    key: &PairKey,
    trail: &[Nucleotide],
    depth: usize,
    lambda: f64,
    limit: Option<f64>,
    ends: &mut EndGains<'_>,
    scored: &mut WrapStat,
) -> WrapPrefix {
    const NUC: [Nucleotide; 4] = [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U];
    let open = key.open;
    let inner = (key.nuc_open(), key.nuc_close);
    let mut entries: Vec<Entry> = Vec::new();
    let mut heap: BinaryHeap<Least> = BinaryHeap::new();
    let push = |entries: &mut Vec<Entry>, heap: &mut BinaryHeap<Least>, v: f64, e: Entry| {
        heap.push(Least(v, entries.len()));
        entries.push(e);
    };
    for r in 0..=depth.min(open.pos as usize) {
        let (table, is_any) = if r == depth {
            (any, true)
        } else {
            (paired, false)
        };
        for &u in dfa.nodes_at(open.pos - r as u32) {
            let Some((ust, _)) = &table[from][dfa.ix(u)] else {
                continue;
            };
            for (cai, nucs) in dfa.paths_between(u, open) {
                let p = State::new(ust.mfe, ust.cai + cai, Back::ExteriorEmpty);
                let e = if !is_any && jc.is_some() {
                    Entry::Group(u, nucs, cai)
                } else {
                    Entry::Cand(p, Junction::Table(u, is_any, nucs))
                };
                push(&mut entries, &mut heap, p.brd(lambda), e);
            }
        }
    }
    // Without the junction cut, the second phase reads 5'-priced first
    // domains class by class (`paired` then leaves them out).
    if let (Some(h), None) = (&head, &jc) {
        for r in 0..depth.min(open.pos as usize + 1) {
            for &u in dfa.nodes_at(open.pos - r as u32) {
                let paths = dfa.paths_between(u, open);
                for key1 in h.tables.p[dfa.ix(u)].keys() {
                    if !h.cut.priced(u, key1)
                        || row_of(1, pair_type(key1.nuc_open(), key1.nuc_close)) != from
                    {
                        continue;
                    }
                    for (p, j) in head_cands(h, u, key1, &paths, ends, lambda) {
                        scored.head_cands += 1;
                        push(&mut entries, &mut heap, p.brd(lambda), Entry::Cand(p, j));
                    }
                }
            }
        }
    }
    while let Some(Least(v, idx)) = heap.pop() {
        if limit.is_some_and(|l| v >= l) {
            return WrapPrefix::NoBetter;
        }
        match std::mem::replace(&mut entries[idx], Entry::Done) {
            Entry::Done => {}
            Entry::Lifted(p, j) => {
                scored.head_spared += 1;
                return WrapPrefix::Spared(p, j);
            }
            Entry::Group(u, run, cai) => {
                let jc = jc.as_ref().expect("groups only with the junction cut");
                scored.cut_groups += 1;
                for (key1, st1) in jc.tables.p[dfa.ix(u)].iter() {
                    let term = if pair_from_nucleotides(key1.nuc_open(), key1.nuc_close)
                        .applies_terminal_au()
                    {
                        Energy(jc.params.misc.terminal_au_37)
                    } else {
                        Energy::ZERO
                    };
                    for (fp, prefix_row) in any.iter().enumerate().take(jc.rows) {
                        if (jc.next_row)(fp, key1) != Some(from) {
                            continue;
                        }
                        let Some((ps, _)) = &prefix_row[dfa.ix(key1.open)] else {
                            continue;
                        };
                        let p = State::new(
                            ps.mfe + st1.mfe + term,
                            ps.cai + st1.cai + cai,
                            Back::ExteriorEmpty,
                        );
                        push(
                            &mut entries,
                            &mut heap,
                            p.brd(lambda),
                            Entry::Cell(u, *key1, fp, run.clone(), cai),
                        );
                    }
                }
            }
            Entry::Cell(u, key1, fp, run, cai) => {
                let jc = jc.as_mut().expect("groups only with the junction cut");
                let slip = head.as_ref().is_some_and(|h| h.cut.slip);
                let cut = jc.cuts.entry(u.pos).or_insert_with(|| {
                    let mut c = TailCut::lazy_at(dfa, jc.tables, jc.depth, false, &[], u.pos);
                    c.max_open = DEFAULT_MAX_OPEN_WRAP;
                    c.slip = slip;
                    c
                });
                let st1 = *jc.tables.p[dfa.ix(u)].get(&key1).expect("scan cell");
                let paths = vec![(cai, run.clone())];
                {
                    if cut.is_tail(&key1) {
                        cut.price(jc.params, dfa, jc.tables, u, key1);
                    }
                    let term = if pair_from_nucleotides(key1.nuc_open(), key1.nuc_close)
                        .applies_terminal_au()
                    {
                        Energy(jc.params.misc.terminal_au_37)
                    } else {
                        Energy::ZERO
                    };
                    let head_cell = head.as_ref().is_some_and(|h| h.cut.priced(u, &key1));
                    {
                        let prefix_row = &any[fp];
                        if fp == 0 && head_cell {
                            let h = head.as_ref().expect("head cell");
                            let cls = cut.classes(u, &key1);
                            for (p, j) in head_cands(h, u, &key1, &paths, ends, lambda) {
                                scored.head_cands += 1;
                                let Some(cls) = cls else {
                                    push(&mut entries, &mut heap, p.brd(lambda), Entry::Cand(p, j));
                                    continue; // next head candidate
                                };
                                let Junction::Head { class: hci, .. } = &j else {
                                    unreachable!()
                                };
                                let hst = h.cut.classes(u, &key1).expect("priced")[*hci].st;
                                for (ci, c) in cls.iter().enumerate() {
                                    // The higher floor: the cut class's
                                    // value in place of the head class's.
                                    let pc = State::new(
                                        Energy(p.mfe.0 - hst.mfe.0 + c.st.mfe.0),
                                        p.cai - hst.cai + c.st.cai,
                                        Back::ExteriorEmpty,
                                    );
                                    let by_cut = pc.brd(lambda) > p.brd(lambda);
                                    let q = if by_cut { pc } else { p };
                                    let mut jj = j.clone();
                                    if let Junction::Head { cut, .. } = &mut jj {
                                        *cut = Some((ci, by_cut));
                                    }
                                    push(
                                        &mut entries,
                                        &mut heap,
                                        q.brd(lambda),
                                        Entry::Cand(q, jj),
                                    );
                                }
                            }
                            continue;
                        }
                        let Some((ps, _)) = &prefix_row[dfa.ix(key1.open)] else {
                            continue;
                        };
                        // With the 5' wrap, a first domain opening within the
                        // enumerated leading runs is read once per run, so the
                        // rival can spell it; the runs' least is the table's.
                        let leads: Vec<(f64, Vec<Nucleotide>)> = match (&w5, &head) {
                            (Some(_), Some(h)) if fp == 0 && key1.open.pos <= h.lead_cap => h
                                .leads
                                .get(&(h.class, h.tables.layout().ix(key1.open)))
                                .cloned()
                                .unwrap_or_default(),
                            _ => Vec::new(),
                        };
                        let mut add = |cst: State, class: Option<usize>| {
                            let mut push_one = |ps: State, lead: Option<Vec<Nucleotide>>| {
                                let p = State::new(
                                    ps.mfe + cst.mfe + term,
                                    ps.cai + cst.cai + cai,
                                    Back::ExteriorEmpty,
                                );
                                let j = Junction::Cut {
                                    u,
                                    key: key1,
                                    class,
                                    row: fp,
                                    run: run.clone(),
                                    lead,
                                    w5: None,
                                };
                                if p.brd(lambda) < v - 1e-6 {
                                    scored.order_breaks += 1;
                                }
                                push(&mut entries, &mut heap, p.brd(lambda), Entry::Cand(p, j));
                            };
                            if leads.is_empty() {
                                push_one(*ps, None);
                            }
                            for (lcai, lead) in &leads {
                                let ls = State::new(Energy::ZERO, *lcai, Back::ExteriorEmpty);
                                push_one(ls, Some(lead.clone()));
                            }
                        };
                        match cut.classes(u, &key1) {
                            Some(cls) => {
                                for (ci, c) in cls.iter().enumerate() {
                                    add(c.st, Some(ci));
                                }
                            }
                            None => add(st1, None),
                        }
                    }
                }
            }
            Entry::Cand(p, j) => {
                scored.scored += 1;
                let run = j.run().to_vec();
                let mut killed = ends.wrap_gain(&run, inner, trail) > 0;
                match (&j, killed) {
                    (
                        Junction::Head {
                            u,
                            key: key1,
                            class,
                            lead,
                            cut: hcut,
                            ..
                        },
                        false,
                    ) => {
                        if let (Some((ci, _)), Some(jc)) = (hcut, jc.as_ref()) {
                            let cut = &jc.cuts[&u.pos()];
                            let c = &cut.classes(*u, key1).expect("priced")[*ci];
                            scored.cut_tests += 1;
                            killed = cut
                                .kills(
                                    c,
                                    &[],
                                    &mut |w| {
                                        let mut five = w.to_vec();
                                        five.extend_from_slice(&run);
                                        ends.wrap_gain(&five, inner, trail)
                                    },
                                    cut.max_open,
                                )
                                .is_some();
                            scored.cut_kills += usize::from(killed);
                        }
                        let h = head.as_ref().expect("head candidate");
                        let c = &h.cut.classes(*u, key1).expect("priced")[*class];
                        killed = killed
                            || c.inv.iter().any(|x| {
                                if x.sealed
                                    || x.delta == crate::tail_cut::UNKILLABLE
                                    || x.three.iter().any(|&v| v > 3)
                                {
                                    return false;
                                }
                                let mut five: Vec<Nucleotide> =
                                    x.three.iter().map(|&v| NUC[v as usize]).collect();
                                five.extend_from_slice(&run);
                                let g = ends.wrap_gain(&five, inner, trail);
                                scored.inv_tests += 1;
                                if g != i32::MIN {
                                    scored.inv_best = Some(
                                        scored.inv_best.map_or(g - x.delta, |b| b.max(g - x.delta)),
                                    );
                                }
                                g > x.delta
                            });
                        // The slipped outer helix: the head class's invasion
                        // rivals with the 3' strand spelled by the junction cut.
                        if let (false, true, Some((ci, _)), Some(jc)) =
                            (killed, h.cut.slip, hcut, jc.as_ref())
                        {
                            let cut = &jc.cuts[&u.pos()];
                            let cc = &cut.classes(*u, key1).expect("priced")[*ci];
                            let (tests, k, best) = h.cut.kills_slip(
                                c,
                                lead,
                                key1.close.pos,
                                cut,
                                cc,
                                &mut |f, i, t| ends.invasion_gain(f, i, t),
                            );
                            scored.slip_tests += tests;
                            scored.slip_kills += usize::from(k);
                            if let Some(b) = best {
                                scored.slip_best = Some(scored.slip_best.map_or(b, |m| m.max(b)));
                            }
                            killed = k;
                        }
                    }
                    (
                        Junction::Cut {
                            u,
                            key: key1,
                            class: Some(ci),
                            ..
                        },
                        false,
                    ) => {
                        let jc = jc.as_ref().expect("cut candidate");
                        let cut = &jc.cuts[&u.pos()];
                        let c = &cut.classes(*u, key1).expect("priced")[*ci];
                        scored.cut_tests += 1;
                        killed = cut
                            .kills(
                                c,
                                &[],
                                &mut |w| {
                                    let mut five = w.to_vec();
                                    five.extend_from_slice(&run);
                                    ends.wrap_gain(&five, inner, trail)
                                },
                                cut.max_open,
                            )
                            .is_some();
                        scored.cut_kills += usize::from(killed);
                    }
                    _ => {}
                }
                // The 5' wrap across the same junction, after a first domain
                // whose leading run is spelled.
                let spelled = match &j {
                    Junction::Head { key, lead, .. }
                    | Junction::Cut {
                        key,
                        lead: Some(lead),
                        ..
                    } => Some((*key, lead.clone())),
                    _ => None,
                };
                // The first domain's own invasion rivals, with the last domain
                // kept and both blocks (a 5'-priced first domain only).
                if let (
                    false,
                    Some(Some(trail)),
                    Junction::Head {
                        u,
                        key: key1,
                        class,
                        lead,
                        ..
                    },
                    Some(h),
                ) = (killed, w5.as_ref().map(|w| w.inv), &j, head.as_ref())
                {
                    let c = &h.cut.classes(*u, key1).expect("priced")[*class];
                    let (tests, k, best) = first_invasion_kills(
                        c,
                        (lead, &run),
                        (key.nuc_open(), key.nuc_close),
                        trail,
                        ends,
                    );
                    scored.w5_inv1_tests += tests;
                    if let Some(b) = best {
                        scored.w5_inv1_best = Some(scored.w5_inv1_best.map_or(b, |m| m.max(b)));
                    }
                    scored.w5_inv1_kills += usize::from(k);
                    killed = k;
                }
                if let (false, Some(w5), Some((key1, lead))) = (killed, w5.as_mut(), &spelled) {
                    match w5.kills(dfa, lead, key1, &run, key, ends, scored) {
                        W5::Killed => killed = true,
                        W5::Spared(Some((f, ci5))) => {
                            // The higher floor: the candidate goes back in
                            // the order at the raised value.
                            let q = State::new(
                                Energy(p.mfe.0 + f.mfe.0 - w5.last_st.mfe.0),
                                p.cai + f.cai - w5.last_st.cai,
                                Back::ExteriorEmpty,
                            );
                            let mut jj = j.clone();
                            if let Junction::Head { w5, .. } | Junction::Cut { w5, .. } = &mut jj {
                                *w5 = Some(ci5);
                            }
                            push(&mut entries, &mut heap, q.brd(lambda), Entry::Lifted(q, jj));
                            continue;
                        }
                        W5::Spared(None) => {}
                    }
                }
                if !killed {
                    scored.head_spared += usize::from(matches!(j, Junction::Head { .. }));
                    return WrapPrefix::Spared(p, j);
                }
            }
        }
    }
    WrapPrefix::Killed
}

/// The most unspelled window symbols a junction-cut class may hold and still
/// be tried (each costs a factor of four in refolds).
const DEFAULT_MAX_OPEN_WRAP: usize = 3;

/// Candidates after the 5'-priced first domain `(u, key1)`: every class no
/// head rival kills after each gainless leading run, followed by each run.
fn head_cands(
    h: &WrapHead<'_>,
    u: Node,
    key1: &PairKey,
    paths: &[(f64, Vec<Nucleotide>)],
    ends: &mut EndGains<'_>,
    lambda: f64,
) -> Vec<(State, Junction)> {
    let mut out = Vec::new();
    let Some(classes) = h.cut.classes(u, key1) else {
        return out;
    };
    let term = if pair_from_nucleotides(key1.nuc_open(), key1.nuc_close).applies_terminal_au() {
        h.au
    } else {
        Energy::ZERO
    };
    for (lcai, lead) in h
        .leads
        .get(&(h.class, h.tables.layout().ix(key1.open)))
        .into_iter()
        .flatten()
    {
        let memo_key = (
            h.tables.layout().ix(u),
            *key1,
            lead.iter().map(|&n| n as u8).collect::<Vec<u8>>(),
        );
        let cached = h.survivors.borrow().get(&memo_key).cloned();
        let ids = match cached {
            Some(ids) => ids,
            None => {
                let ids: Vec<usize> = head_survivors(h.cut, u, key1, lead, ends, lambda)
                    .into_iter()
                    .map(|(i, _)| i)
                    .collect();
                h.survivors.borrow_mut().insert(memo_key, ids.clone());
                ids
            }
        };
        for i in ids {
            let c = &classes[i];
            for (pcai, run) in paths {
                let p = State::new(c.st.mfe + term, lcai + c.st.cai + pcai, Back::ExteriorEmpty);
                out.push((
                    p,
                    Junction::Head {
                        u,
                        key: *key1,
                        class: i,
                        lead: lead.clone(),
                        run: run.clone(),
                        cut: None,
                        w5: None,
                    },
                ));
            }
        }
    }
    out
}

/// What the wrap's second phase reads: the 5' cut, the scan's tables, the
/// gainless leading runs, the lead class and the terminal AU term.
struct WrapHead<'a> {
    cut: &'a TailCut,
    tables: &'a Tables,
    leads: &'a LeadRuns,
    /// The longest enumerated leading run: a first domain opening no later
    /// is reached by those runs alone.
    lead_cap: u32,
    class: usize,
    au: Energy,
    /// Surviving head classes per (outer node, cell, leading run), kept
    /// across calls: every group the wrap expands re-reads the same cells.
    survivors: &'a RefCell<SurvivorMemo>,
}

/// Head classes no 5' rival kills, by (outer node, cell, leading run).
type SurvivorMemo = HashMap<(usize, PairKey, Vec<u8>), Vec<usize>>;

/// Gainless leading runs by `(lead class, node)`: `(cai, symbols)`.
type LeadRuns = HashMap<(usize, usize), Vec<(f64, Vec<Nucleotide>)>>;

/// A leading run a both-cut domain is priced after: its prefix, and the head
/// class setting its floor with the run itself (`None`: the table's prefix).
type PricedRun = (State, Option<(usize, Vec<Nucleotide>)>);

/// The least combined score at `tables`' λ over every item and configuration
/// with `min_b..=max_b` top-level pairs (`max_b = None`: unbounded) whose
/// leading and trailing runs, in the item's own spelling, give the context
/// blocks no gain (up to `cap` symbols, then the necessary condition of the
/// module docs), and — when `joint > 0` — survive the **joint end clamp**: the
/// first `min(a, joint)` and last `min(b, joint)` symbols joined to the context
/// blocks, the clamp table of the joined blocks at the configuration's branch
/// count `D'` (the rest of the end runs sits unmatched inside the clamp's
/// branching loop, where unmatched symbols cost nothing). With gainless end
/// runs the rival's slack is exactly `D'`, so a pair survives only if `D' = 0`.
/// The leading `joint` symbols are carried as a lead class through the
/// recurrence. The path writes into a `C` column for
/// [`crate::traceback::recover`].
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn rival_optimum(
    params: &Params,
    dfa: &Dfa,
    tables: &Tables,
    min_b: u8,
    max_b: Option<u8>,
    ends: &mut EndGains<'_>,
    cap: u32,
    joint: usize,
    tail: Option<&TailCut>,
    head: Option<&TailCut>,
) -> RivalOptimum {
    let lambda = tables.lambda();
    let saturate = max_b.is_none();
    // Unbounded: the last counted row is "at least `top`", so keep exactly one
    // pair apart from it.
    let top = max_b.map_or(min_b.max(2) as usize, |m| m as usize);
    let n_rows = 7 + top.saturating_sub(1);
    let len = dfa.seq_len;
    let width = 2 * (len as usize + 1);
    let better = |slot: &Cell, cand: &State| {
        slot.as_ref()
            .is_none_or(|(old, _)| cand.brd(lambda) < old.brd(lambda))
    };
    // Gainless leading runs, grouped into classes by their first `joint` symbols.
    let lead_cap = cap.min(len);
    let mut classes: Vec<(Vec<Nucleotide>, bool)> = Vec::new();
    let mut leads: Vec<(Node, usize, f64, Vec<Nucleotide>)> = Vec::new();
    for pos in 0..=lead_cap {
        for &o in dfa.nodes_at(pos) {
            for (cai, nucs) in dfa.paths_between(dfa.start(), o) {
                if ends.gain(true, &nucs) != 0 {
                    continue;
                }
                let head = (nucs[..nucs.len().min(joint)].to_vec(), nucs.len() <= joint);
                let k = match classes.iter().position(|c| *c == head) {
                    Some(k) => k,
                    None => {
                        classes.push(head);
                        classes.len() - 1
                    }
                };
                leads.push((o, k, cai, nucs));
            }
        }
    }
    let mut any: Grid = vec![vec![vec![None; width]; n_rows]; classes.len()];
    let mut paired: Grid = any.clone();
    // The wrap's second phase reads 5'-priced first domains class by class,
    // so it needs the closed-pair table without them.
    let wrap_cut = tail.map_or(0, |tc| if tc.wrap > 0 { tc.wrap_cut } else { 0 });
    let wrap_free =
        head.is_some() && wrap_cut == 0 && tail.is_some_and(|tc| tc.wrap > 0 && tc.wrap_free);
    // Junction cuts built so far, by junction start.
    let mut jcuts: HashMap<u32, TailCut> = HashMap::new();
    // The 5' wrap across the same junctions (only after a 5'-priced first
    // domain, which the junction cut's candidates carry), and its junction
    // cuts by junction end.
    let (wrap5, wrap5_cut, wrap5_both, wrap5_inv) = match tail {
        Some(tc) if wrap_cut > 0 && head.is_some() => (
            tc.wrap5,
            tc.wrap5_cut,
            tc.wrap5_both,
            tc.wrap5_inv && tc.wrap5_cut > 0,
        ),
        _ => (0, 0, false, false),
    };
    let mut j5cuts: HashMap<u32, TailCut> = HashMap::new();
    let survivor_memo: RefCell<SurvivorMemo> = RefCell::new(HashMap::new());
    // The row a pair moves a prefix row to, as the recurrence does.
    let next_row = |fp: usize, key: &PairKey| -> Option<usize> {
        let b = branches_of(fp);
        match (b < top, saturate) {
            (true, _) => Some(row_of(b + 1, pair_type(key.nuc_open(), key.nuc_close))),
            (false, true) => Some(row_of(top, 0)),
            (false, false) => None,
        }
    };
    let mut paired_nh: Grid = if wrap_free { any.clone() } else { Vec::new() };
    let mut leads_at: LeadRuns = HashMap::new();
    if head.is_some() {
        for (o, k, cai, nucs) in &leads {
            leads_at
                .entry((*k, dfa.ix(*o)))
                .or_default()
                .push((*cai, nucs.clone()));
        }
    }
    let mut head_kills = 0;
    let mut own_kills = 0;
    for (o, k, cai, nucs) in leads {
        let cand = State::new(Energy::ZERO, cai, Back::ExteriorEmpty);
        if better(&any[k][0][dfa.ix(o)], &cand) {
            any[k][0][dfa.ix(o)] = Some((cand, Via::Lead(nucs)));
        }
    }
    let au = Energy(params.misc.terminal_au_37);
    for pos in 0..=len {
        for &o in dfa.nodes_at(pos) {
            for k in 0..classes.len() {
                for (key, st) in &tables.p[dfa.ix(o)] {
                    let pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
                    let term = if pair.applies_terminal_au() {
                        au
                    } else {
                        Energy::ZERO
                    };
                    if let Some(hc) = head.filter(|hc| hc.priced(o, key)) {
                        // The first pair, priced class by class against every
                        // gainless leading run that reaches it.
                        let to = row_of(1, pair_type(key.nuc_open(), key.nuc_close));
                        let runs = leads_at.get(&(k, dfa.ix(key.open)));
                        for (ci, c) in hc.classes(o, key).unwrap_or(&[]).iter().enumerate() {
                            for (cai, lead) in runs.into_iter().flatten() {
                                let cand = State::new(
                                    c.st.mfe + term,
                                    cai + c.st.cai,
                                    Back::ExteriorPair {
                                        prefix: key.open,
                                        pair: PairRef {
                                            outer: o,
                                            key: *key,
                                        },
                                    },
                                );
                                let fits_p = better(&paired[k][to][dfa.ix(o)], &cand);
                                let fits_a = better(&any[k][to][dfa.ix(o)], &cand);
                                if !fits_p && !fits_a {
                                    continue;
                                }
                                let by_block = hc
                                    .kills(c, lead, &mut |w| ends.gain(true, w), hc.max_open)
                                    .is_some()
                                    || hc.kills_invasion(c, lead, &mut |f, i, t| {
                                        ends.invasion_gain(f, i, t)
                                    })
                                    || (hc.split
                                        && hc
                                            .kills_split(c, lead, &mut |f, first, ml| {
                                                ends.split_gain(f, first, &[], ml)
                                            })
                                            .1);
                                // Counted apart: how much the context-free rival
                                // adds where a block was available anyway.
                                let by_own = !by_block
                                    && hc.own_repair
                                    && hc.kills_invasion(c, lead, &mut |f, i, t| {
                                        ends.own_invasion_gain(f, i, t)
                                    });
                                if by_block || by_own {
                                    head_kills += 1;
                                    own_kills += usize::from(by_own);
                                    continue;
                                }
                                let via = Via::Head {
                                    lead: lead.clone(),
                                    key: *key,
                                    class: ci,
                                };
                                let priced = tail.is_some_and(|tc| tc.priced(o, key));
                                if !priced && fits_p {
                                    paired[k][to][dfa.ix(o)] = Some((cand, via.clone()));
                                }
                                if fits_a {
                                    any[k][to][dfa.ix(o)] = Some((cand, via));
                                }
                            }
                        }
                    }
                    for from in 0..n_rows {
                        if from == 0 && head.is_some_and(|hc| hc.priced(o, key)) {
                            continue;
                        }
                        let Some((prefix, _)) = &any[k][from][dfa.ix(key.open)] else {
                            continue;
                        };
                        let b = branches_of(from);
                        let to = match (b < top, saturate) {
                            (true, _) => row_of(b + 1, pair_type(key.nuc_open(), key.nuc_close)),
                            (false, true) => row_of(top, 0),
                            (false, false) => continue,
                        };
                        let cand = State::new(
                            prefix.mfe + st.mfe + term,
                            prefix.cai + st.cai,
                            Back::ExteriorPair {
                                prefix: key.open,
                                pair: PairRef {
                                    outer: o,
                                    key: *key,
                                },
                            },
                        );
                        let via = Via::Pair {
                            prefix: key.open,
                            row: from,
                        };
                        let priced = tail.is_some_and(|tc| tc.priced(o, key));
                        if !priced && better(&paired[k][to][dfa.ix(o)], &cand) {
                            paired[k][to][dfa.ix(o)] = Some((cand, via.clone()));
                        }
                        if wrap_free && !priced && better(&paired_nh[k][to][dfa.ix(o)], &cand) {
                            paired_nh[k][to][dfa.ix(o)] = Some((cand, via.clone()));
                        }
                        if better(&any[k][to][dfa.ix(o)], &cand) {
                            any[k][to][dfa.ix(o)] = Some((cand, via));
                        }
                    }
                }
                for (r, row) in any[k].iter_mut().enumerate() {
                    if r == 0 && pos < lead_cap {
                        continue;
                    }
                    let Some((st, _)) = row[dfa.ix(o)].clone() else {
                        continue;
                    };
                    for e in dfa.right(o) {
                        let cand = State::new(
                            st.mfe,
                            st.cai + e.cai,
                            Back::ExteriorUnpaired {
                                prev: o,
                                nuc: e.nuc,
                            },
                        );
                        if better(&row[dfa.ix(e.to)], &cand) {
                            row[dfa.ix(e.to)] = Some((cand, Via::Unpaired { prev: o }));
                        }
                    }
                }
            }
        }
    }
    // Close each parse with a gainless trailing run that the joint clamp spares.
    let tail_from = len.saturating_sub(cap);
    let mut best: Option<Closing> = None;
    for k in 0..classes.len() {
        for row in 1..n_rows {
            let r = branches_of(row);
            if r < (min_b as usize).max(1) {
                continue;
            }
            for pos in tail_from..=len {
                for &o in dfa.nodes_at(pos) {
                    let (table, from_any) = if pos == tail_from && len > cap {
                        (&any, true)
                    } else {
                        (&paired, false)
                    };
                    let Some((st, _)) = &table[k][row][dfa.ix(o)] else {
                        continue;
                    };
                    for (cai, nucs) in dfa.paths_between(o, dfa.end()) {
                        if ends.gain(false, &nucs) != 0 {
                            continue;
                        }
                        let v = st.brd(lambda) - (1.0 - lambda) * cai * 100.0;
                        if best.as_ref().is_some_and(|b| v >= b.value) {
                            continue;
                        }
                        if joint > 0 {
                            let tail = &nucs[nucs.len() - nucs.len().min(joint)..];
                            if ends.joint_clamp(&classes[k].0, tail, r, top) < 0 {
                                continue;
                            }
                        }
                        // One domain inside a degree-2 loop of the clamp: only
                        // when both end runs are whole in the probe.
                        let single = (1..=6).contains(&row) && classes[k].1 && !from_any;
                        if single && nucs.len() <= joint {
                            let closing = PAIR_TYPES[row - 1];
                            if ends.single_domain(&classes[k].0, closing, &nucs) < 0 {
                                continue;
                            }
                        }
                        let total = State::new(st.mfe, st.cai + cai, st.back);
                        best = Some(Closing {
                            value: v,
                            class: k,
                            row,
                            node: o,
                            from_any,
                            trail: nucs,
                            total,
                            tail: None,
                            head_floor: None,
                            lead: None,
                            junction: None,
                        });
                    }
                }
            }
        }
    }
    // Last domains closing in the tail, class by class, each refused when a
    // tail cut kills it.
    let mut tail_kills = 0;
    let mut jstat = JointStat::default();
    let (mut wrap_kills, mut wrap_scored) = (0usize, WrapStat::default());
    let wrap = tail.map_or(0, |tc| tc.wrap);
    // A domain that is both the first and the last is priced by both cuts.
    // Every derivation of it falls in one head class and one tail class, so
    // one surviving both is worth at least the higher of the two floors: the
    // least tail class standing, and the least head class no rival kills
    // after this leading run. Memoized per cell and run; `None`: every head
    // class falls.
    let mut head_floor_lifts = 0;
    let mut head_floors: HashMap<(usize, PairKey, Vec<u8>), Option<usize>> = HashMap::new();
    // With the joint two-end rival: every surviving head class per cell and
    // run, with its window gains.
    let joint_ends = head.is_some_and(|h| h.joint_ends);
    let mut alive: HashMap<(usize, PairKey, Vec<u8>), Survivors> = HashMap::new();
    let (mut split_tests, mut split_kills, mut split_best) = (0usize, 0usize, None::<i32>);
    if let Some(tc) = tail {
        for k in 0..classes.len() {
            for pos in tc.first() + 1..=len {
                for &o in dfa.nodes_at(pos) {
                    let trails = dfa.paths_between(o, dfa.end());
                    for key in tables.p[dfa.ix(o)].keys() {
                        if !tc.priced(o, key) {
                            continue;
                        }
                        let Some(cls) = tc.classes(o, key) else {
                            continue;
                        };
                        let pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
                        let term = if pair.applies_terminal_au() {
                            au
                        } else {
                            Energy::ZERO
                        };
                        for from in 0..n_rows {
                            let Some((prefix, via)) = &any[k][from][dfa.ix(key.open)] else {
                                continue;
                            };
                            let b = branches_of(from);
                            let to = match (b < top, saturate) {
                                (true, _) => {
                                    row_of(b + 1, pair_type(key.nuc_open(), key.nuc_close))
                                }
                                (false, true) => row_of(top, 0),
                                (false, false) => continue,
                            };
                            let r = branches_of(to);
                            if r < (min_b as usize).max(1) {
                                continue;
                            }
                            // The leading runs to price it after. On a domain that is
                            // also the first, every gainless run reaching it: the head
                            // floor depends on the run, so the table's best prefix
                            // alone no longer gives the least total. Otherwise the
                            // table's prefix, whose value the class does not depend on.
                            let mut runs: Vec<PricedRun> = Vec::new();
                            match (head, via) {
                                (Some(hc), Via::Lead(_)) if hc.priced(o, key) => {
                                    for (cai, lead) in
                                        leads_at.get(&(k, dfa.ix(key.open))).into_iter().flatten()
                                    {
                                        let floor = *head_floors
                                            .entry((
                                                dfa.ix(o),
                                                *key,
                                                lead.iter().map(|&n| n as u8).collect(),
                                            ))
                                            .or_insert_with(|| {
                                                head_floor(hc, o, key, lead, ends, lambda)
                                            });
                                        // Every head class falls after this run: so
                                        // does the domain.
                                        if let Some(hci) = floor {
                                            let st =
                                                State::new(Energy::ZERO, *cai, Back::ExteriorEmpty);
                                            runs.push((st, Some((hci, lead.clone()))));
                                        }
                                    }
                                }
                                _ => runs.push((*prefix, None)),
                            }
                            for (prefix, run) in &runs {
                                // The split rival reads the trailing run, so on a
                                // one-domain item every head class standing after
                                // the leading run is kept, least first.
                                let split_sv = match (run, head) {
                                    (Some((_, lead)), Some(hc))
                                        if hc.split && !joint_ends && r == 1 =>
                                    {
                                        Some(
                                            alive
                                                .entry((
                                                    dfa.ix(o),
                                                    *key,
                                                    lead.iter().map(|&n| n as u8).collect(),
                                                ))
                                                .or_insert_with(|| {
                                                    head_survivors(hc, o, key, lead, ends, lambda)
                                                })
                                                .clone(),
                                        )
                                    }
                                    _ => None,
                                };
                                let survivors = match (joint_ends, run, head) {
                                    (true, Some((_, lead)), Some(hc)) => Some(
                                        alive
                                            .entry((
                                                dfa.ix(o),
                                                *key,
                                                lead.iter().map(|&n| n as u8).collect(),
                                            ))
                                            .or_insert_with(|| {
                                                head_survivors(hc, o, key, lead, ends, lambda)
                                            })
                                            .clone(),
                                    ),
                                    _ => None,
                                };
                                for (ci, c) in cls.iter().enumerate() {
                                    // The higher of the two floors stands for the class.
                                    let (st0, by_head0) = match run {
                                        Some((hci, _)) => {
                                            let h = head.and_then(|hc| hc.classes(o, key)).unwrap()
                                                [*hci]
                                                .st;
                                            if h.brd(lambda) > c.st.brd(lambda) {
                                                head_floor_lifts += 1;
                                                (h, Some(*hci))
                                            } else {
                                                (c.st, None)
                                            }
                                        }
                                        None => (c.st, None),
                                    };
                                    for (cai, nucs) in &trails {
                                        // Joint two-end rival: the least head class
                                        // that survives together with this tail class
                                        // after this trailing run.
                                        let (st, by_head) = match (&survivors, head) {
                                            (Some(sv), Some(hc)) => {
                                                // The joint choice only raises the value:
                                                // skip what cannot win before scoring fills.
                                                let low = prefix.brd(lambda)
                                                    + c.st.brd(lambda)
                                                    + term.0 as f64 * lambda
                                                    - (1.0 - lambda) * cai * 100.0;
                                                if ends.gain(false, nucs) != 0
                                                    || best.as_ref().is_some_and(|b| low >= b.value)
                                                {
                                                    continue;
                                                }
                                                let g3 = tc.least_gains(
                                                    c,
                                                    nucs,
                                                    &mut |w| ends.gain(false, w),
                                                    tc.max_open,
                                                );
                                                let hcls = hc.classes(o, key).unwrap();
                                                if let Some((i, g5)) = sv.first() {
                                                    jstat.meet(&hcls[*i], g5, c, &g3);
                                                }
                                                let Some(at) = sv.iter().position(|(i, g5)| {
                                                    !joint_kills(&hcls[*i], g5, c, &g3)
                                                }) else {
                                                    jstat.passed += sv.len();
                                                    jstat.dead += 1;
                                                    tail_kills += 1;
                                                    continue;
                                                };
                                                jstat.passed += at;
                                                let hci = sv[at].0;
                                                let h = hcls[hci].st;
                                                if h.brd(lambda) > c.st.brd(lambda) {
                                                    head_floor_lifts += 1;
                                                    (h, Some(hci))
                                                } else {
                                                    (c.st, None)
                                                }
                                            }
                                            _ => match (&split_sv, run, head) {
                                                (Some(sv), Some((_, lead)), Some(hc)) => {
                                                    let low = prefix.brd(lambda)
                                                        + c.st.brd(lambda)
                                                        + term.0 as f64 * lambda
                                                        - (1.0 - lambda) * cai * 100.0;
                                                    if ends.gain(false, nucs) != 0
                                                        || best
                                                            .as_ref()
                                                            .is_some_and(|b| low >= b.value)
                                                    {
                                                        continue;
                                                    }
                                                    let hcls = hc.classes(o, key).unwrap();
                                                    let standing = sv.iter().find(|(i, _)| {
                                                        let (tests, k, b) = hc.kills_split(
                                                            &hcls[*i],
                                                            lead,
                                                            &mut |f, first, ml| {
                                                                ends.split_gain(f, first, nucs, ml)
                                                            },
                                                        );
                                                        split_tests += tests;
                                                        if let Some(b) = b {
                                                            split_best = Some(
                                                                split_best.map_or(b, |m| m.max(b)),
                                                            );
                                                        }
                                                        !k
                                                    });
                                                    let Some(&(hci, _)) = standing else {
                                                        split_kills += 1;
                                                        tail_kills += 1;
                                                        continue;
                                                    };
                                                    let h = hcls[hci].st;
                                                    if h.brd(lambda) > c.st.brd(lambda) {
                                                        head_floor_lifts += 1;
                                                        (h, Some(hci))
                                                    } else {
                                                        (c.st, None)
                                                    }
                                                }
                                                _ => (st0, by_head0),
                                            },
                                        };
                                        let cell = State::new(
                                            prefix.mfe + st.mfe + term,
                                            prefix.cai + st.cai,
                                            Back::ExteriorPair {
                                                prefix: key.open,
                                                pair: PairRef {
                                                    outer: o,
                                                    key: *key,
                                                },
                                            },
                                        );
                                        let v = cell.brd(lambda) - (1.0 - lambda) * cai * 100.0;
                                        if best.as_ref().is_some_and(|b| v >= b.value) {
                                            continue;
                                        }
                                        if ends.gain(false, nucs) != 0 {
                                            continue;
                                        }
                                        if joint > 0 {
                                            let tl = &nucs[nucs.len() - nucs.len().min(joint)..];
                                            if ends.joint_clamp(&classes[k].0, tl, r, top) < 0 {
                                                continue;
                                            }
                                        }
                                        if (1..=6).contains(&to)
                                            && classes[k].1
                                            && nucs.len() <= joint
                                        {
                                            let closing = PAIR_TYPES[to - 1];
                                            if ends.single_domain(&classes[k].0, closing, nucs) < 0
                                            {
                                                continue;
                                            }
                                        }
                                        if tc
                                            .kills(
                                                c,
                                                nucs,
                                                &mut |w| ends.gain(false, w),
                                                tc.max_open,
                                            )
                                            .is_some()
                                        {
                                            tail_kills += 1;
                                            continue;
                                        }
                                        // The 3' wrap rival: the least prefix
                                        // whose junction run it spares.
                                        let (cell, v, junction) = if wrap > 0
                                            && run.is_none()
                                            && from >= 1
                                        {
                                            let limit = best.as_ref().map(|b| {
                                                b.value - (cell.brd(lambda) - prefix.brd(lambda))
                                                    + (1.0 - lambda) * cai * 100.0
                                            });
                                            let wh = ((wrap_free || wrap_cut > 0)
                                                && head.is_some())
                                            .then(|| WrapHead {
                                                cut: head.expect("wrap_free"),
                                                tables,
                                                leads: &leads_at,
                                                lead_cap,
                                                class: k,
                                                au,
                                                survivors: &survivor_memo,
                                            });
                                            let table = if wrap_free && wrap_cut == 0 {
                                                &paired_nh[k]
                                            } else {
                                                &paired[k]
                                            };
                                            let jc = (wrap_cut > 0).then_some(JunctionCut {
                                                params,
                                                tables,
                                                depth: wrap_cut,
                                                cuts: &mut jcuts,
                                                rows: n_rows,
                                                next_row: &next_row,
                                            });
                                            let w5 = (wrap5 > 0 || wrap5_cut > 0 || wrap5_both)
                                                .then(|| Wrap5 {
                                                    run: wrap5,
                                                    depth: wrap5_cut,
                                                    both: wrap5_both.then_some(nucs.as_slice()),
                                                    inv: wrap5_inv.then_some(nucs.as_slice()),
                                                    params,
                                                    tables,
                                                    outer: o,
                                                    last_st: st,
                                                    cuts: &mut j5cuts,
                                                });
                                            // The 5' rivals need the whole junction
                                            // run: spell runs up to their cap exactly
                                            // (the 3' wrap's gain only grows with a
                                            // longer run, so it loses nothing).
                                            let depth =
                                                if w5.is_some() { wrap.max(wrap5) } else { wrap };
                                            match wrap_prefix(
                                                dfa,
                                                (&any[k], table),
                                                wh,
                                                jc,
                                                w5,
                                                from,
                                                key,
                                                nucs,
                                                depth,
                                                lambda,
                                                limit,
                                                ends,
                                                &mut wrap_scored,
                                            ) {
                                                WrapPrefix::Killed => {
                                                    wrap_kills += 1;
                                                    continue;
                                                }
                                                WrapPrefix::NoBetter => continue,
                                                WrapPrefix::Spared(p, j) => {
                                                    let cell = State::new(
                                                        p.mfe + st.mfe + term,
                                                        p.cai + st.cai,
                                                        cell.back,
                                                    );
                                                    let v = cell.brd(lambda)
                                                        - (1.0 - lambda) * cai * 100.0;
                                                    if best.as_ref().is_some_and(|b| v >= b.value) {
                                                        continue;
                                                    }
                                                    (cell, v, Some(j))
                                                }
                                            }
                                        } else {
                                            (cell, v, None)
                                        };
                                        best = Some(Closing {
                                            value: v,
                                            class: k,
                                            row: to,
                                            node: o,
                                            from_any: false,
                                            trail: nucs.clone(),
                                            total: State::new(cell.mfe, cell.cai + cai, cell.back),
                                            tail: Some((*key, ci, from, cell)),
                                            head_floor: by_head,
                                            lead: run.as_ref().map(|(_, l)| l.clone()),
                                            junction,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let Some(Closing {
        class: k,
        row,
        node,
        from_any,
        trail,
        total,
        tail: tail_pick,
        head_floor: head_side,
        lead: lead_run,
        junction: junction_pick,
        ..
    }) = best
    else {
        return RivalOptimum {
            opt: ClassOptimum {
                best: None,
                by_branches: Vec::new(),
                path: Vec::new(),
            },
            overrides: Vec::new(),
            tail_kills: tail_kills + head_kills,
            wrap_kills,
            wrap_scored,
            own_kills,
            head_pick: None,
            head_floor_lifts,
            head_floor_optimum: false,
            tail_pick: None,
            joint: jstat,
            split_tests,
            split_kills,
            split_best,
            wrap_pick: None,
        };
    };
    // A domain whose head floor stood traces the head class instead, and
    // one the 5' junction cut's floor raised traces that class.
    let w5_pick = match &junction_pick {
        Some(Junction::Head { w5: Some(ci5), .. } | Junction::Cut { w5: Some(ci5), .. }) => {
            Some(*ci5)
        }
        _ => None,
    };
    let mut overrides = match (tail, head, &tail_pick, &head_side, w5_pick) {
        (_, _, Some((key, _, _, _)), _, Some(ci5)) => {
            j5cuts[&key.open.pos()].overrides(node, *key, ci5)
        }
        (_, Some(hc), Some((key, _, _, _)), Some(hci), _) => hc.overrides(node, *key, *hci),
        (Some(tc), _, Some((key, ci, _, _)), _, _) => tc.overrides(node, *key, *ci),
        _ => Vec::new(),
    };
    let (any, paired) = (&any[k], &paired[k]);
    let mut path = Vec::new();
    let head_floor_optimum = head_side.is_some();
    let mut head_pick = match (&tail_pick, head_side, &lead_run) {
        (Some((key, _, _, _)), Some(class), Some(lead)) => Some(HeadPick {
            outer: node,
            key: *key,
            class,
            lead: lead.clone(),
        }),
        _ => None,
    };
    // The trail, from `node` forward.
    let mut at = node;
    let mut st = match &tail_pick {
        Some((_, _, _, cell)) => *cell,
        None => {
            if from_any {
                any[row][dfa.ix(node)].clone()
            } else {
                paired[row][dfa.ix(node)].clone()
            }
            .expect("chosen cell")
            .0
        }
    };
    let mut tail_run = Vec::new();
    for (e, nuc) in dfa
        .spell_to(at, &codes(&trail), None)
        .expect("trail edge")
        .into_iter()
        .zip(&trail)
    {
        st = State::new(
            st.mfe,
            st.cai + e.cai,
            Back::ExteriorUnpaired {
                prev: at,
                nuc: *nuc,
            },
        );
        tail_run.push((e.to, st));
        at = e.to;
    }
    // Back from `node` to the start.
    let (mut cur, mut r, mut table_any) = (node, row, from_any);
    let mut junction_lead = false;
    let wrap_pick = junction_pick.as_ref().map(|j| {
        let note = |kind,
                    start: &Node,
                    cell: Option<(&Node, &PairKey)>,
                    cut_class: Option<usize>,
                    by_cut| {
            let class = cut_class.and_then(|ci| {
                cell.and_then(|(o, k)| {
                    jcuts
                        .get(&start.pos())
                        .and_then(|c| c.classes(*o, k))
                        .map(|l| &l[ci])
                })
            });
            WrapPick {
                kind,
                start: start.pos,
                cell: cell.map(|(_, k)| (k.open.pos, k.close.pos)),
                cut_class,
                by_cut,
                class_brd: class.map(|c| c.st.brd(lambda)),
                why: class
                    .map(|c| c.why.iter().map(|w| w.name()).collect())
                    .unwrap_or_default(),
                scan_brd: cell
                    .and_then(|(o, k)| tables.p[dfa.ix(*o)].get(k).map(|st| st.brd(lambda))),
            }
        };
        match j {
            Junction::Table(u, _, _) => note("table", u, None, None, false),
            Junction::Head { u, key, cut, .. } => note(
                "head",
                u,
                Some((u, key)),
                cut.map(|(ci, _)| ci),
                cut.is_some_and(|(_, b)| b),
            ),
            Junction::Cut { u, key, class, .. } => {
                note("cut", u, Some((u, key)), *class, class.is_some())
            }
        }
    });
    if let Some((key, _, from, cell)) = &tail_pick {
        path.push((node, *cell));
        cur = key.open;
        r = *from;
        table_any = true;
        // A junction run the wrap rival spared: walk it back to its start.
        if let Some(Junction::Table(u, u_any, run)) = &junction_pick {
            let table = if *u_any {
                &any[r]
            } else if paired_nh.is_empty() {
                &paired[r]
            } else {
                &paired_nh[k][r]
            };
            let (mut st, _) = table[dfa.ix(*u)].clone().expect("junction start");
            let mut at = *u;
            let mut nodes = Vec::new();
            for (e, nuc) in dfa
                .spell_to(at, &codes(run), None)
                .expect("junction edge")
                .into_iter()
                .zip(run)
            {
                st = State::new(
                    st.mfe,
                    st.cai + e.cai,
                    Back::ExteriorUnpaired {
                        prev: at,
                        nuc: *nuc,
                    },
                );
                nodes.push((e.to, st));
                at = e.to;
            }
            path.extend(nodes.into_iter().rev());
            cur = *u;
            table_any = *u_any;
        }
        // After a domain the junction cut priced: the run, that domain's cell
        // (its class traced by overrides), then the table from its prefix.
        if let Some(Junction::Cut {
            u,
            key,
            class,
            row: fp,
            run,
            lead,
            ..
        }) = &junction_pick
        {
            let cst = match class {
                Some(ci) => jcuts[&u.pos()].classes(*u, key).expect("priced")[*ci].st,
                None => *tables.p[dfa.ix(*u)].get(key).expect("scan cell"),
            };
            let term = if pair_from_nucleotides(key.nuc_open(), key.nuc_close).applies_terminal_au()
            {
                au
            } else {
                Energy::ZERO
            };
            let ps = match lead {
                Some(l) => {
                    let lcai = lead_path(dfa, l).last().map_or(0.0, |(_, s)| s.cai);
                    State::new(Energy::ZERO, lcai, Back::ExteriorEmpty)
                }
                None => any[*fp][dfa.ix(key.open)].clone().expect("prefix").0,
            };
            let first = State::new(
                ps.mfe + cst.mfe + term,
                ps.cai + cst.cai,
                Back::ExteriorPair {
                    prefix: key.open,
                    pair: PairRef {
                        outer: *u,
                        key: *key,
                    },
                },
            );
            let mut st = first;
            let mut at = *u;
            let mut nodes = Vec::new();
            for (e, nuc) in dfa
                .spell_to(at, &codes(run), None)
                .expect("junction edge")
                .into_iter()
                .zip(run)
            {
                st = State::new(
                    st.mfe,
                    st.cai + e.cai,
                    Back::ExteriorUnpaired {
                        prev: at,
                        nuc: *nuc,
                    },
                );
                nodes.push((e.to, st));
                at = e.to;
            }
            path.extend(nodes.into_iter().rev());
            path.push((*u, first));
            if let Some(ci) = class {
                overrides.extend(jcuts[&u.pos()].overrides(*u, *key, *ci));
            }
            cur = key.open;
            r = *fp;
            table_any = true;
            if let Some(l) = lead {
                path.extend(lead_path(dfa, l).into_iter().rev());
                junction_lead = true;
            }
        }
        // After a 5'-priced first domain: the run, that class's cell and its
        // leading run; nothing is left to read from the tables.
        if let Some(Junction::Head {
            u,
            key,
            class,
            lead,
            run,
            cut: hcut,
            ..
        }) = &junction_pick
        {
            let hc = head.expect("5' cut");
            let by_cut = hcut.and_then(|(ci, by)| by.then_some(ci));
            let c = match by_cut {
                Some(ci) => &jcuts[&u.pos()].classes(*u, key).expect("priced")[ci],
                None => &hc.classes(*u, key).expect("priced")[*class],
            };
            let term = if pair_from_nucleotides(key.nuc_open(), key.nuc_close).applies_terminal_au()
            {
                au
            } else {
                Energy::ZERO
            };
            let lcai: f64 = lead_path(dfa, lead).last().map_or(0.0, |(_, s)| s.cai);
            let mut st = State::new(
                c.st.mfe + term,
                lcai + c.st.cai,
                Back::ExteriorPair {
                    prefix: key.open,
                    pair: PairRef {
                        outer: *u,
                        key: *key,
                    },
                },
            );
            let first = st;
            let mut at = *u;
            let mut nodes = Vec::new();
            for (e, nuc) in dfa
                .spell_to(at, &codes(run), None)
                .expect("junction edge")
                .into_iter()
                .zip(run)
            {
                st = State::new(
                    st.mfe,
                    st.cai + e.cai,
                    Back::ExteriorUnpaired {
                        prev: at,
                        nuc: *nuc,
                    },
                );
                nodes.push((e.to, st));
                at = e.to;
            }
            path.extend(nodes.into_iter().rev());
            path.push((*u, first));
            match by_cut {
                Some(ci) => overrides.extend(jcuts[&u.pos()].overrides(*u, *key, ci)),
                None => overrides.extend(hc.overrides(*u, *key, *class)),
            }
            head_pick = Some(HeadPick {
                outer: *u,
                key: *key,
                class: *class,
                lead: lead.clone(),
            });
            path.extend(lead_path(dfa, lead).into_iter().rev());
            junction_lead = true;
        }
    }
    // A domain priced after a chosen leading run: that run, not the table's.
    if let (Some(_), Some(lead)) = (&tail_pick, &lead_run) {
        path.extend(lead_path(dfa, lead).into_iter().rev());
    }
    while !junction_lead && (lead_run.is_none() || tail_pick.is_none()) {
        let cell = if table_any {
            &any[r][dfa.ix(cur)]
        } else {
            &paired[r][dfa.ix(cur)]
        };
        let (st, via) = cell.clone().expect("path cell");
        match via {
            Via::Pair { prefix, row } => {
                path.push((cur, st));
                cur = prefix;
                r = row;
                table_any = true;
            }
            Via::Unpaired { prev } => {
                path.push((cur, st));
                cur = prev;
                table_any = true;
            }
            Via::Head { lead, key, class } => {
                path.push((cur, st));
                if let Some(hc) = head {
                    overrides.extend(hc.overrides(cur, key, class));
                }
                head_pick = Some(HeadPick {
                    outer: cur,
                    key,
                    class,
                    lead: lead.clone(),
                });
                let mut nodes = vec![(
                    dfa.start(),
                    State::new(Energy::ZERO, 0.0, Back::ExteriorEmpty),
                )];
                let mut a = dfa.start();
                let mut cai = 0.0;
                for (e, nuc) in dfa
                    .spell_to(a, &codes(&lead), None)
                    .expect("lead edge")
                    .into_iter()
                    .zip(lead)
                {
                    cai += e.cai;
                    nodes.push((
                        e.to,
                        State::new(Energy::ZERO, cai, Back::ExteriorUnpaired { prev: a, nuc }),
                    ));
                    a = e.to;
                }
                path.extend(nodes.into_iter().rev());
                break;
            }
            Via::Lead(nucs) => {
                let mut nodes = vec![(
                    dfa.start(),
                    State::new(Energy::ZERO, 0.0, Back::ExteriorEmpty),
                )];
                let mut a = dfa.start();
                let mut cai = 0.0;
                for (e, nuc) in dfa
                    .spell_to(a, &codes(&nucs), None)
                    .expect("lead edge")
                    .into_iter()
                    .zip(nucs)
                {
                    cai += e.cai;
                    nodes.push((
                        e.to,
                        State::new(Energy::ZERO, cai, Back::ExteriorUnpaired { prev: a, nuc }),
                    ));
                    a = e.to;
                }
                path.extend(nodes.into_iter().rev());
                break;
            }
        }
    }
    path.reverse();
    path.extend(tail_run);
    RivalOptimum {
        opt: ClassOptimum {
            best: Some(total),
            by_branches: Vec::new(),
            path,
        },
        overrides,
        tail_kills: tail_kills + head_kills,
        wrap_kills,
        wrap_scored,
        own_kills,
        head_pick,
        head_floor_lifts,
        head_floor_optimum,
        tail_pick: tail_pick.map(|(key, ci, _, _)| (node, key, ci)),
        joint: jstat,
        split_tests,
        split_kills,
        split_best,
        wrap_pick,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Params {
        Params::load_default().unwrap()
    }

    #[test]
    fn shape_reads_branches_and_end_runs() {
        assert_eq!(shape("..((...))..(...)."), (2, 2, 1));
        assert_eq!(shape("....."), (0, 5, 5));
    }

    #[test]
    fn a_single_terminal_loop_scores_as_its_configuration() {
        let p = params();
        let w = "GGGAAACCC";
        let h = least_terminal_loop(&p, w).unwrap();
        // Every single-pair configuration is a genuine configuration, so its
        // cost is never below the word's minimum over all configurations.
        assert!(h >= mfe(&p, w));
        assert_eq!(least_terminal_loop(&p, "AAAAAAA"), None);
    }

    /// The exact-end optimum is never below the same branch counts without
    /// end rivals, and its traced item and configuration re-score to its value
    /// with gainless end runs in the item's own spelling.
    #[test]
    fn exact_end_optimum_traces_to_a_gainless_in_class_configuration() {
        use crate::class_bound::class_optimum;
        use crate::{dfa, scan::scan, ScanConfig};
        use derna_codon_dp::codon_table::aa_to_index;
        let p = params();
        let usage = CodonUsage::load_default().unwrap();
        let aa: Vec<usize> = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
            .bytes()
            .map(|b| aa_to_index(b as char).unwrap())
            .collect();
        let graph = dfa::build(&aa, &usage);
        let mut tables = scan(&p, &graph, 0.5, ScanConfig::default());
        let (u5, u3) = ("GUGGUUGUGGUUGGUGUU", "UUGGUGUUGGUGGUUGUG");
        let plain = class_optimum(
            &p,
            &graph,
            &tables,
            &ExteriorClass {
                min_branches: 1,
                max_branches: Some(2),
                ..ExteriorClass::default()
            },
        )
        .best
        .unwrap();
        let mut ends = EndGains::new(&p, u5, u3);
        let opt = rival_optimum(&p, &graph, &tables, 1, Some(2), &mut ends, 9, 6, None, None).opt;
        let best = opt.best.unwrap();
        assert!(best.brd(0.5) >= plain.brd(0.5) - 1e-9);
        let mut column = vec![None; tables.c.len()];
        for &(node, st) in &opt.path {
            column[graph.ix(node)] = Some(st);
        }
        std::mem::swap(&mut tables.c, &mut column);
        let rec = crate::traceback::recover(&p, &graph, &tables);
        std::mem::swap(&mut tables.c, &mut column);
        let energy = rna_mfe::evaluate(&p, rec.sequence.as_bytes(), &rec.structure).unwrap();
        assert_eq!(energy, best.mfe);
        let (b, a, t) = shape(&rec.structure);
        assert!((1..=2).contains(&b), "{}", rec.structure);
        let nucs: Vec<Nucleotide> = rec
            .sequence
            .bytes()
            .filter_map(Nucleotide::from_byte)
            .collect();
        assert_eq!(ends.gain(true, &nucs[..a.min(9)]), 0);
        assert_eq!(ends.gain(false, &nucs[nucs.len() - t.min(9)..]), 0);
    }

    #[test]
    fn menus_spell_symbols() {
        let usage = CodonUsage::load_default().unwrap();
        let m = Menus::full(&[9, 13], &usage); // M, F
        assert_eq!(m.spellings(4, true), vec!["AUGU".to_string()]);
        assert_eq!(
            m.spellings(2, false),
            vec!["UC".to_string(), "UU".to_string()]
        );
    }

    /// An item whose every window folds below zero leaves no empty
    /// configuration; a word that cannot pair leaves its empty one.
    #[test]
    fn the_sweep_kills_exactly_the_negative_windows() {
        let p = params();
        let gc = Menus {
            choices: vec![
                vec![(*b"GGG", 0.0)],
                vec![(*b"GAA", 0.0)],
                vec![(*b"ACC", 0.0)],
                vec![(*b"CAA", 0.0)],
            ],
        };
        let killed = sweep_empty(&p, &gc, 3, None, 0, 1);
        let negative = mfe(&p, "GGGGAAACC") < 0 || mfe(&p, "GAAACCCAA") < 0;
        assert_eq!(negative, killed.emptied_at.is_some());
        let poor = Menus {
            choices: vec![vec![(*b"AAA", 0.0), (*b"ACA", -0.5)]; 5],
        };
        let kept = sweep_empty(&p, &poor, 3, None, 0, 1);
        assert_eq!(kept.emptied_at, None);
        assert_eq!(kept.best.unwrap().0, "AAA".repeat(5));
    }

    /// The branch-only table reproduces the clamp table's entries from one
    /// branch on.
    #[test]
    fn branch_clamps_match_the_clamp_table() {
        let p = params();
        let u5 = "ACAUUUGCUUCUGACACAACUGUGUUCACUAGCAACCUCAAACAGACACC";
        let u3 = "GCUCGCUUUCUUGCUGUCCAAUUUCUAUUAAAGGUUCCUUUGUUCCCUAAG";
        let full = clamp_table(&p, u5, u3, 4).unwrap();
        let fast = branch_clamps(&p, u5, u3, 4);
        for b in 1..=4 {
            let want = full[b].map(|v| (v * 100.0).round() as i32);
            assert_eq!(Some(fast[b]), want, "B = {b}");
        }
    }

    /// A degree-2 loop of 30 unmatched symbols is a configuration; one of 31
    /// is not; a branching loop of any size is.
    #[test]
    fn the_degree2_span_cap_is_enforced() {
        let loop_of =
            |n1: usize, n2: usize| format!("(({}((...)){}))", ".".repeat(n1), ".".repeat(n2));
        assert!(degree2_within_cap(&loop_of(30, 0)));
        assert!(degree2_within_cap(&loop_of(15, 15)));
        assert!(!degree2_within_cap(&loop_of(31, 0)));
        assert!(!degree2_within_cap(&loop_of(16, 15)));
        let branching = format!("({}((...))((...)){})", ".".repeat(40), ".".repeat(40));
        assert!(degree2_within_cap(&branching));
    }

    /// End rivals are genuine configurations: no claimed slack is below the
    /// word's real gap `F − E`, a cut that keeps everything has slack 0, and
    /// depth past the chain gives no rival.
    #[test]
    fn end_rivals_never_claim_more_than_the_real_gap() {
        let p = params();
        let (u5, u3) = ("GUGGUUGUGGUUGGUGUU", "UUGGUGUUGGUGGUUGUG");
        for item in [
            "AUGGCCAAGCUGGCCGCCAAGCUGCAGGCCUUCCGCAAGCUG",
            "AUGCCCGGGAAACCCGGGAUUUGCAGCUGCACCUGCAGCUG",
        ] {
            let (d, own) = rna_mfe::fold(&p, item.as_bytes()).unwrap();
            let e = mfe(&p, u5) + own.0 + mfe(&p, u3);
            let f = mfe(&p, &format!("{u5}{item}{u3}"));
            let floor = f - e;
            assert_eq!(
                cut_rival_slack(&p, u5, u3, item, &d, 0, item.len()),
                Some(0)
            );
            for s in 0..12 {
                for t in item.len() - 12..=item.len() {
                    if let Some(v) = cut_rival_slack(&p, u5, u3, item, &d, s, t) {
                        assert!(v >= floor, "cut {s} {t}: {v} < {floor}");
                    }
                }
            }
            let chain = 64;
            for side in [EndDomain::First, EndDomain::Last, EndDomain::Only] {
                for k in 0..8 {
                    if let Some(v) = invasion_slack(&p, u5, u3, item, &d, side, k) {
                        assert!(v >= floor, "{side:?} depth {k}: {v} < {floor}");
                    }
                }
                assert_eq!(invasion_slack(&p, u5, u3, item, &d, side, chain), None);
            }
        }
    }

    /// Dropping the context block can only weaken the invasion rival: folding
    /// the freed strands with a block is never worse than folding them alone,
    /// so `own_invasion_gain <= invasion_gain` at the same chain position.
    /// That inequality is why the context-free rival earns its place only
    /// where no block is adjacent to the freed strands, and a refactor that
    /// swapped either baseline would break it silently.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn the_context_free_invasion_gain_never_exceeds_the_blocked_one() {
        let p = params();
        let (u5, u3) = ("GGCGCAUAUGCGCA", "ACUACUGGCGG");
        let item = "GGCGCAAAAGCGCCAAUAA";
        let (d, _) = rna_mfe::fold(&p, item.as_bytes()).unwrap();
        let x: Vec<Nucleotide> = item.bytes().filter_map(Nucleotide::from_byte).collect();
        let pr = partners(&d).unwrap();
        let open = d.find('(').unwrap();
        let chain = outer_chain(d.as_bytes(), &pr, open);
        let j0 = pr[open];
        let mut ends = EndGains::new(&p, u5, u3);
        // Both gains defined on the same position, or the comparison is vacuous.
        let mut both = 0;
        for depth in 1..chain.len() {
            let (a, b) = chain[depth];
            let blocked = ends.invasion_gain(&x[..a], (x[a], x[b]), &x[b + 1..=j0]);
            let own = ends.own_invasion_gain(&x[..a], (x[a], x[b]), &x[b + 1..=j0]);
            if blocked == i32::MIN || own == i32::MIN {
                continue;
            }
            assert!(
                own <= blocked,
                "depth {depth}: own {own} > blocked {blocked}"
            );
            both += 1;
        }
        assert!(both > 0, "no chain position had both gains defined");
    }

    /// The invasion rival splits as the DP filter prices it: slack = (kept
    /// part's cost − domain's cost) − the refold's gain.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn invasion_slack_is_break_cost_less_refold_gain() {
        let p = params();
        let (u5, u3) = ("GGCGCAUAUGCGCA", "ACUACUGGCGG");
        let item = "GGCGCAAAAGCGCCAAUAA";
        let (d, _) = rna_mfe::fold(&p, item.as_bytes()).unwrap();
        let x: Vec<Nucleotide> = item.bytes().filter_map(Nucleotide::from_byte).collect();
        let pr = partners(&d).unwrap();
        let open = d.find('(').unwrap();
        let chain = outer_chain(d.as_bytes(), &pr, open);
        assert!(chain.len() >= 3, "{d}");
        let j0 = pr[open];
        let mut checked = 0;
        for depth in 1..chain.len() {
            let (a, b) = chain[depth];
            let Some(slack) = invasion_slack(&p, u5, u3, item, &d, EndDomain::First, depth) else {
                continue;
            };
            let inside = rna_mfe::evaluate(&p, &item.as_bytes()[a..=b], &d[a..=b])
                .unwrap()
                .0;
            let span = rna_mfe::evaluate(&p, &item.as_bytes()[..=j0], &d[..=j0])
                .unwrap()
                .0;
            let mut ends = EndGains::new(&p, u5, u3);
            let gain = ends.invasion_gain(&x[..a], (x[a], x[b]), &x[b + 1..=j0]);
            assert_eq!(slack, inside - span - gain, "depth {depth}");
            checked += 1;
        }
        assert!(checked > 0);
    }

    /// The 3' wrap rival never claims more than the word's real gap: with the
    /// last domain kept whole and nothing removed, a positive gain is a
    /// configuration of the whole word that much below `E`, so it is at most
    /// `E - F`. Non-vacuity: the suffix is built to pair with the junction
    /// run, so at least one strand length must gain.
    #[test]
    fn the_wrap_gain_never_exceeds_the_real_gap() {
        let p = params();
        // Two hairpin domains joined by a junction run the suffix complements.
        let d1 = "GGGCGCAAAGCGCCC";
        let junction = "AAGGACUCCA";
        let d2 = "GCCGCGGAAACCGCGGC";
        let trail = "AA";
        let item = format!("{d1}{junction}{d2}{trail}");
        let (u5, u3) = ("AUAUAUAUAUAU", "AAUGGAGUCCUUAUAUAU");
        let (d, own) = rna_mfe::fold(&p, item.as_bytes()).unwrap();
        let e = mfe(&p, u5) + own.0 + mfe(&p, u3);
        let f = mfe(&p, &format!("{u5}{item}{u3}"));
        let x: Vec<Nucleotide> = item.bytes().filter_map(Nucleotide::from_byte).collect();
        let a = d1.len() + junction.len();
        let b = a + d2.len() - 1;
        // The item's own optimum keeps the second hairpin as its last domain.
        assert_eq!(partners(&d).unwrap()[a], b, "{d}");
        let mut ends = EndGains::new(&p, u5, u3);
        let mut gained = 0;
        for k in 0..=junction.len() {
            let g = ends.wrap_gain(&x[a - k..a], (x[a], x[b]), &x[b + 1..]);
            assert!(g <= e - f, "k {k}: gain {g} > gap {}", e - f);
            gained += usize::from(g > 0);
        }
        assert!(gained > 0, "no junction length gained: {d}");
    }

    /// The 5' wrap and the two-domain refold never claim more than the
    /// word's real gap (both keep the domains whole and remove nothing).
    /// Non-vacuity: the prefix is built to pair with the junction run, so
    /// both gain.
    #[test]
    fn the_five_prime_and_two_domain_gains_never_exceed_the_real_gap() {
        let p = params();
        let d1 = "GGGCGCAAAGCGCCC";
        let junction = "AAGGACUCCA";
        let d2 = "GCCGCGGAAACCGCGGC";
        let trail = "AA";
        let item = format!("{d1}{junction}{d2}{trail}");
        let (u5, u3) = ("AUAUAUGGAGUCCUUAUAU", "AUAUAUAUAUAU");
        let (d, own) = rna_mfe::fold(&p, item.as_bytes()).unwrap();
        let e = mfe(&p, u5) + own.0 + mfe(&p, u3);
        let f = mfe(&p, &format!("{u5}{item}{u3}"));
        let x: Vec<Nucleotide> = item.bytes().filter_map(Nucleotide::from_byte).collect();
        let c1 = d1.len() - 1;
        let a = d1.len() + junction.len();
        let b = a + d2.len() - 1;
        let pr = partners(&d).unwrap();
        assert_eq!((pr[0], pr[a]), (c1, b), "{d}");
        let mut ends = EndGains::new(&p, u5, u3);
        let (outer1, outer2) = ((x[0], x[c1]), (x[a], x[b]));
        let w5 = ends.invasion_gain(&[], outer1, &x[c1 + 1..a]);
        let both = ends.two_domain_gain(&[], outer1, &x[c1 + 1..a], outer2, &x[b + 1..]);
        assert!(w5 > 0 && both > 0, "no gain: {w5} / {both}, {d}");
        assert!(
            w5 <= e - f && both <= e - f,
            "gains {w5} / {both} > gap {}",
            e - f
        );
    }

    /// The split rival never claims more than the word's real gap, per item at
    /// every blind level and as the class-level refold (two stand-ins, the
    /// domain's 3' side and every symbol past the first branch left out)
    /// against the exact removal cost. Items are drawn from a fixed generator
    /// with the suffix complementing the item's head, so the blocks can pair
    /// the freed 5' strand; non-vacuity: some split must gain.
    #[test]
    fn the_split_rivals_never_claim_more_than_the_real_gap() {
        let p = params();
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % m) as usize
        };
        let rc = |s: &str| -> String {
            s.bytes()
                .rev()
                .map(|c| match c {
                    b'A' => 'U',
                    b'C' => 'G',
                    b'G' => 'C',
                    _ => 'A',
                })
                .collect()
        };
        let (mut checked, mut killing) = (0, 0);
        for _ in 0..400 {
            let mut word = |len: usize| -> String {
                (0..len).map(|_| ['A', 'C', 'G', 'U'][next(4)]).collect()
            };
            let stem = word(5);
            let h1 = format!("GCGC{}GCGC", word(4));
            let h2 = format!("CGGC{}GCCG", word(4));
            let (r1, r2, r3) = (word(2), word(1), word(2));
            let item = format!("{stem}{r1}{h1}{r2}{h2}{r3}{}{}", rc(&stem), word(3));
            let u5 = word(12);
            let u3 = format!("{}{}", rc(&item[..6]), word(6));
            let (d, own) = rna_mfe::fold(&p, item.as_bytes()).unwrap();
            if shape(&d).0 != 1 {
                continue;
            }
            let e = mfe(&p, &u5) + own.0 + mfe(&p, &u3);
            let gap = e - mfe(&p, &format!("{u5}{item}{u3}"));
            for blind in 0..=3 {
                if let Some(s) = split_slack(&p, &u5, &u3, &item, &d, EndDomain::Only, blind) {
                    checked += 1;
                    killing += usize::from(s < 0);
                    assert!(
                        -s <= gap,
                        "blind {blind}: slack {s} below -gap {gap}: {item} {d}"
                    );
                }
            }
            // The class-level refold: two stand-ins, the 3' side left out.
            let pr = partners(&d).unwrap();
            let open = d.find('(').unwrap();
            let chain = outer_chain(d.as_bytes(), &pr, open);
            let &(ci, cj) = chain.last().unwrap();
            let Some(first) = (ci + 1..cj).find(|&k| d.as_bytes()[k] == b'(') else {
                continue;
            };
            let mut kept: Vec<u8> = d.as_bytes().to_vec();
            for &(i, j) in &chain {
                kept[i] = b'.';
                kept[j] = b'.';
            }
            let kept = String::from_utf8(kept).unwrap();
            if shape(&kept).0 < 2 {
                continue;
            }
            let delta = rna_mfe::evaluate(&p, item.as_bytes(), &kept).unwrap().0 - own.0;
            let x: Vec<Nucleotide> = item.bytes().filter_map(Nucleotide::from_byte).collect();
            let close = pr[open];
            let mut ends = EndGains::new(&p, &u5, &u3);
            for (stop, rest) in [
                (first, true),
                (first.min(3), true),
                (0, true),
                (first, false),
            ] {
                let trail = if rest { &x[close + 1..] } else { &x[..0] };
                let g = ends.split_gain(&x[..stop], (x[first], x[pr[first]]), trail, false);
                if g == i32::MIN {
                    continue;
                }
                checked += 1;
                killing += usize::from(delta < g);
                assert!(
                    g - delta <= gap,
                    "class split: gain {g} delta {delta} gap {gap}: {item} {d}"
                );
            }
            // The branch-count-free record: the removal with every branch's
            // `intern` charged back, against a refold enclosing the stand-ins.
            let c = shape(&kept).0 as i32;
            let delta_ml = delta + (c - 2) * p.ml.intern_37;
            // Also with no branch symbols read (two GC stand-ins) and the
            // window cut short of the first branch, as a first branch past the
            // window is scored.
            let gc = (Nucleotide::G, Nucleotide::C);
            for (stop, first_pair) in [
                (first, (x[first], x[pr[first]])),
                (first, gc),
                (first / 2, gc),
                (0, gc),
            ] {
                let g = ends.split_gain(&x[..stop], first_pair, &[], true);
                if g != i32::MIN {
                    checked += 1;
                    killing += usize::from(delta_ml < g);
                    assert!(
                        g - delta_ml <= gap,
                        "ml split: gain {g} delta {delta_ml} gap {gap}: {item} {d}"
                    );
                }
            }
        }
        assert!(
            checked > 50 && killing > 0,
            "checked {checked}, killing {killing}"
        );
        // Two domains: the first one split with everything after its branches
        // left out (the second domain and the trailing run kept as they are).
        let (mut checked, mut killing) = (0, 0);
        for _ in 0..400 {
            let mut word = |len: usize| -> String {
                (0..len).map(|_| ['A', 'C', 'G', 'U'][next(4)]).collect()
            };
            let stem = word(5);
            let h1 = format!("GCGC{}GCGC", word(4));
            let h2 = format!("CGGC{}GCCG", word(4));
            let d2 = format!("GGAC{}GUCC", word(4));
            let (r1, r2, r3, jn) = (word(2), word(1), word(2), word(3));
            let item = format!("{stem}{r1}{h1}{r2}{h2}{r3}{}{jn}{d2}{}", rc(&stem), word(2));
            let u5 = word(12);
            let u3 = format!("{}{}", rc(&item[..6]), word(6));
            let (d, own) = rna_mfe::fold(&p, item.as_bytes()).unwrap();
            if shape(&d).0 < 2 {
                continue;
            }
            let e = mfe(&p, &u5) + own.0 + mfe(&p, &u3);
            let gap = e - mfe(&p, &format!("{u5}{item}{u3}"));
            if let Some(s) = split_slack(&p, &u5, &u3, &item, &d, EndDomain::First, 3) {
                checked += 1;
                killing += usize::from(s < 0);
                assert!(
                    -s <= gap,
                    "first, level 3: slack {s} below -gap {gap}: {item} {d}"
                );
            }
            let pr = partners(&d).unwrap();
            let open = d.find('(').unwrap();
            let chain = outer_chain(d.as_bytes(), &pr, open);
            let &(ci, cj) = chain.last().unwrap();
            let Some(first) = (ci + 1..cj).find(|&k| d.as_bytes()[k] == b'(') else {
                continue;
            };
            let mut kept: Vec<u8> = d.as_bytes().to_vec();
            for &(i, j) in &chain {
                kept[i] = b'.';
                kept[j] = b'.';
            }
            let kept = String::from_utf8(kept).unwrap();
            if shape(&kept).0 < 3 {
                continue;
            }
            let delta = rna_mfe::evaluate(&p, item.as_bytes(), &kept).unwrap().0 - own.0;
            let x: Vec<Nucleotide> = item.bytes().filter_map(Nucleotide::from_byte).collect();
            let mut ends = EndGains::new(&p, &u5, &u3);
            for stop in [first, first.min(3), 0] {
                let g = ends.split_gain(&x[..stop], (x[first], x[pr[first]]), &[], false);
                if g == i32::MIN {
                    continue;
                }
                checked += 1;
                killing += usize::from(delta < g);
                assert!(
                    g - delta <= gap,
                    "first, class: gain {g} delta {delta} gap {gap}: {item} {d}"
                );
            }
        }
        assert!(
            checked > 50 && killing > 0,
            "two domains: checked {checked}, killing {killing}"
        );
    }

    #[test]
    fn free_run_bounds_stop_at_the_first_gain() {
        assert_eq!(last_free(&[0, 0, 0, 30, 30]), Some(2));
        assert_eq!(last_free(&[0, 0, 0]), None);
        assert_eq!(last_free(&[0, 10]), Some(0));
    }
}
