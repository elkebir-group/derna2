//! Counterexample-guided search with energy-inequality cuts over a fixed
//! designed structure.
//!
//! The designed configuration `d` is held fixed: the structure each region of
//! a start design folds to on its own (5′ UTR, CDS, 3′ UTR, side by side). The
//! codons are chosen to minimize `E(x, d) − μ·A(x)`, where `A` is the summed
//! adaptation weight. Each rival `r` met so far — a free optimum that pairs
//! across a region boundary — adds the cut `E(x, d) ≤ E(x, r)`. A cut removes
//! exactly the designs that rival beats, so it never removes a design that
//! realizes `d` and is modular.
//!
//! Every energy at `dangles = 0` is a sum of loop terms, and each term reads a
//! handful of nucleotides. The objective is therefore a sum of factors over
//! codon positions. The master problem keeps the positions any cut reads (the
//! *cut set*), eliminates every other position exactly by variable
//! elimination, and enumerates the cut set against the cuts. Each master step
//! is exact for the fixed `d`; the method as a whole is a heuristic only
//! because `d` is fixed.
//!
//! A step's design is checked by two exact folds ([`rna_mfe::fold`]): the
//! constrained energy `E` (the regions folded apart) and the free energy `F` (the
//! whole molecule). The design is modular when `F = E`; otherwise
//! the free optimum is the next rival.

use std::collections::HashSet;

use rna_mfe::loops::pair_from_nucleotides;
use rna_mfe::params::BasePair;
use rna_mfe::{
    bulge_energy, hairpin_energy, internal_energy, stack_energy, BulgeLoop, HairpinLoop,
    InternalLoop, Nucleotide, Params, StackPair,
};

use crate::repair::Menus;

/// One sequence-dependent term of a structure's energy, keyed so that the
/// same term in two structures compares equal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    /// The pair must be canonical (0, or infeasible).
    Pair(usize, usize),
    /// A hairpin closed by the pair.
    Hairpin(usize, usize),
    /// A stack, bulge or interior loop: outer pair, inner pair.
    TwoLoop(usize, usize, usize, usize),
    /// A multiloop's sequence-free part: closing pair and children.
    MultiConst(usize, usize, Vec<(usize, usize)>),
    /// The terminal-AU term a pair pays as a branch of a multiloop or of the
    /// exterior loop. The two loops charge the same penalty, so the term is
    /// keyed by the pair alone and cancels between structures that both
    /// branch there.
    BranchAu(usize, usize),
    /// The terminal-AU term a pair pays as the closing pair of a multiloop.
    ClosingAu(usize, usize),
}

impl Term {
    /// The sequence positions the term's energy reads.
    pub fn reads(&self) -> Vec<usize> {
        let mut at = match self {
            Term::Pair(i, j) | Term::BranchAu(i, j) | Term::ClosingAu(i, j) => {
                vec![*i, *j]
            }
            Term::Hairpin(i, j) => match j - i - 1 {
                3 | 4 | 6 => (*i..=*j).collect(),
                0..=2 => vec![*i, *j],
                _ => vec![*i, i + 1, j - 1, *j],
            },
            Term::TwoLoop(i, j, k, l) => {
                if k - i - 1 > 0 && j - l - 1 > 0 {
                    vec![*i, i + 1, k - 1, *k, *l, l + 1, j - 1, *j]
                } else {
                    vec![*i, *j, *k, *l]
                }
            }
            Term::MultiConst(..) => Vec::new(),
        };
        at.sort_unstable();
        at.dedup();
        at
    }

    /// The term's energy (dekacalories) on `seq`, or `None` for a
    /// non-canonical pair.
    pub fn energy(&self, params: &Params, seq: &[Nucleotide]) -> Option<i32> {
        let pair = |i: usize, j: usize| pair_from_nucleotides(seq[i], seq[j]);
        let au = |i: usize, j: usize| {
            if pair(i, j).applies_terminal_au() {
                params.misc.terminal_au_37
            } else {
                0
            }
        };
        Some(match self {
            Term::Pair(i, j) => {
                if pair(*i, *j) == BasePair::NN {
                    return None;
                }
                0
            }
            Term::Hairpin(i, j) => {
                hairpin_energy(
                    params,
                    &HairpinLoop {
                        closing: pair(*i, *j),
                        loop_seq: &seq[i + 1..*j],
                    },
                )
                .0
            }
            Term::TwoLoop(i, j, k, l) => two_loop(params, seq, (*i, *j), (*k, *l)),
            Term::MultiConst(i, j, children) => {
                let covered: usize = children.iter().map(|(k, l)| l - k + 1).sum();
                let unpaired = (j - i - 1 - covered) as i32;
                let stems = 1 + children.len() as i32;
                params.ml.closing_37 + params.ml.intern_37 * stems + params.ml.base_37 * unpaired
            }
            Term::BranchAu(p, q) | Term::ClosingAu(p, q) => au(*p, *q),
        })
    }
}

fn two_loop(
    params: &Params,
    seq: &[Nucleotide],
    (i, j): (usize, usize),
    (k, l): (usize, usize),
) -> i32 {
    let outer = pair_from_nucleotides(seq[i], seq[j]);
    let inner = pair_from_nucleotides(seq[k], seq[l]);
    let (n1, n2) = (k - i - 1, j - l - 1);
    if n1 == 0 && n2 == 0 {
        stack_energy(params, &StackPair { outer, inner }).0
    } else if n1 == 0 || n2 == 0 {
        bulge_energy(
            params,
            &BulgeLoop {
                outer,
                inner,
                size: n1.max(n2),
            },
        )
        .0
    } else {
        internal_energy(
            params,
            &InternalLoop {
                outer,
                inner,
                left: &seq[i + 1..k],
                right: &seq[l + 1..j],
            },
        )
        .0
    }
}

/// The partner table of a dot-bracket string.
pub fn pair_table(structure: &str) -> Result<Vec<Option<usize>>, CutError> {
    let mut stack = Vec::new();
    let mut pairs = vec![None; structure.len()];
    for (i, c) in structure.bytes().enumerate() {
        match c {
            b'(' => stack.push(i),
            b')' => {
                let o = stack.pop().ok_or(CutError::Structure(i))?;
                pairs[o] = Some(i);
                pairs[i] = Some(o);
            }
            b'.' => {}
            _ => return Err(CutError::Structure(i)),
        }
    }
    if let Some(o) = stack.pop() {
        return Err(CutError::Structure(o));
    }
    Ok(pairs)
}

/// The pairs directly enclosed by `lo..hi` (positions, half-open).
fn children(pairs: &[Option<usize>], lo: usize, hi: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut k = lo;
    while k < hi {
        match pairs[k] {
            Some(l) if l > k => {
                out.push((k, l));
                k = l + 1;
            }
            _ => k += 1,
        }
    }
    out
}

/// Every term of a structure's energy; their sum is [`rna_mfe::evaluate`].
pub fn terms(pairs: &[Option<usize>]) -> Vec<Term> {
    let mut out = Vec::new();
    for (p, q) in children(pairs, 0, pairs.len()) {
        out.push(Term::BranchAu(p, q));
    }
    for i in 0..pairs.len() {
        let Some(j) = pairs[i] else { continue };
        if j < i {
            continue;
        }
        out.push(Term::Pair(i, j));
        let kids = children(pairs, i + 1, j);
        match kids.len() {
            0 => out.push(Term::Hairpin(i, j)),
            1 => out.push(Term::TwoLoop(i, j, kids[0].0, kids[0].1)),
            _ => {
                out.push(Term::ClosingAu(i, j));
                for &(k, l) in &kids {
                    out.push(Term::BranchAu(k, l));
                }
                out.push(Term::MultiConst(i, j, kids));
            }
        }
    }
    out
}

/// The cross-region pairs of a structure over `utr5 | cds | utr3`.
pub fn cross_pairs(pairs: &[Option<usize>], cds: std::ops::Range<usize>) -> Vec<(usize, usize)> {
    let region = |p: usize| {
        if p < cds.start {
            0
        } else if p < cds.end {
            1
        } else {
            2
        }
    };
    (0..pairs.len())
        .filter_map(|i| {
            pairs[i]
                .filter(|&j| j > i && region(i) != region(j))
                .map(|j| (i, j))
        })
        .collect()
}

/// Errors of the cut loop.
#[derive(Debug, thiserror::Error)]
pub enum CutError {
    /// A malformed dot-bracket at the position.
    #[error("malformed structure at position {0}")]
    Structure(usize),
    /// A residue or codon outside the menus.
    #[error("bad design: {0}")]
    Design(String),
    /// An internal consistency check failed.
    #[error("consistency: {0}")]
    Check(String),
}

/// A table over a few codon positions (row-major, first variable slowest).
#[derive(Debug, Clone)]
struct Factor {
    vars: Vec<usize>,
    sizes: Vec<usize>,
    table: Vec<f64>,
}

impl Factor {
    fn value(&self, asg: &[usize]) -> f64 {
        let mut ix = 0;
        for (v, s) in self.vars.iter().zip(&self.sizes) {
            ix = ix * s + asg[*v];
        }
        self.table[ix]
    }
}

/// Visit every assignment of `vars` (writing into `asg`), first variable
/// slowest.
fn for_each_assignment(
    vars: &[usize],
    sizes: &[usize],
    asg: &mut [usize],
    mut f: impl FnMut(&[usize]),
) {
    for &v in vars {
        asg[v] = 0;
    }
    loop {
        f(asg);
        let mut d = vars.len();
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            asg[vars[d]] += 1;
            if asg[vars[d]] < sizes[d] {
                break;
            }
            asg[vars[d]] = 0;
        }
    }
}

/// One cut `Σ plus ≤ Σ minus` (each side a sum of factors and a constant).
#[derive(Debug, Clone)]
struct Cut {
    plus: Vec<Factor>,
    plus_const: f64,
    minus: Vec<Factor>,
    minus_const: f64,
}

impl Cut {
    fn holds(&self, asg: &[usize]) -> bool {
        let minus = self.minus_const + self.minus.iter().map(|f| f.value(asg)).sum::<f64>();
        if minus.is_infinite() {
            return true; // the rival cannot be spelled at all
        }
        let plus = self.plus_const + self.plus.iter().map(|f| f.value(asg)).sum::<f64>();
        plus <= minus
    }

    fn vars(&self) -> impl Iterator<Item = usize> + '_ {
        self.plus
            .iter()
            .chain(&self.minus)
            .flat_map(|f| f.vars.iter().copied())
    }
}

/// How a master step ended without a design.
#[derive(Debug, Clone, PartialEq)]
pub enum MasterFail {
    /// No assignment of the cut set satisfies every cut.
    Infeasible,
    /// A table would exceed the cap (its size).
    TooLarge(f64),
    /// The run's time limit passed.
    Time,
}

/// One master step's design and its size.
#[derive(Debug, Clone)]
struct MasterSolution {
    asg: Vec<usize>,
    objective: f64,
    cut_set: Vec<usize>,
    space: f64,
    width: f64,
}

/// How a master step treats the cuts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    /// Keep the cut set and enumerate it against the cuts: exact for the
    /// fixed `d`, and exponential in the cut set.
    Exact,
    /// Fold each cut into the objective with a multiplier, eliminate every
    /// position, and double the multipliers of violated cuts until all hold
    /// (at most `steps` doublings). Each solve is exact; the multiplier search
    /// is a heuristic, so a returned design is feasible but may not be the
    /// constrained optimum, and a failure proves nothing.
    Lagrange {
        /// Most multiplier doublings per round.
        steps: usize,
    },
}

/// The value given to a rival term that cannot be spelled (dekacalories),
/// standing in for the rival's infinite energy inside a Lagrangian, where an
/// infinity would swamp every other term.
const UNSPELLABLE: f64 = 500.0;

/// The fixed context of a run: parameters, flanks and codon menus.
pub struct Problem<'a> {
    params: &'a Params,
    utr5: Vec<Nucleotide>,
    utr3: Vec<Nucleotide>,
    menus: Vec<Vec<([Nucleotide; 3], f64)>>,
    cap: f64,
    mode: Mode,
    limit: Option<std::time::Duration>,
}

fn encode(s: &str) -> Result<Vec<Nucleotide>, CutError> {
    s.bytes()
        .map(|b| Nucleotide::from_byte(b).ok_or_else(|| CutError::Design(s.to_string())))
        .collect()
}

fn decode(s: &[Nucleotide]) -> String {
    s.iter()
        .map(|n| match n {
            Nucleotide::A => 'A',
            Nucleotide::C => 'C',
            Nucleotide::G => 'G',
            Nucleotide::U => 'U',
        })
        .collect()
}

fn fold(params: &Params, seq: &[Nucleotide]) -> (String, i32) {
    let (s, e) = rna_mfe::fold(params, decode(seq).as_bytes()).expect("encoded sequence folds");
    (s, e.0)
}

/// One round of the loop: the master step's size and the design's two folds.
#[derive(Debug, Clone)]
pub struct Round {
    /// Codon positions in the cut set.
    pub cut_set: Vec<usize>,
    /// Assignments of the cut set enumerated.
    pub space: f64,
    /// Largest table built while eliminating.
    pub width: f64,
    /// Sum of the cut multipliers the step ended at (0 in [`Mode::Exact`]).
    pub multiplier: f64,
    /// The master objective `E(x, d) − μ·A(x)` (dekacalories).
    pub objective: f64,
    /// Constrained energy of the design (dekacalories).
    pub constrained: i32,
    /// Free energy of the design (dekacalories).
    pub free: i32,
    /// Summed adaptation weight.
    pub weight: f64,
    /// Cross-region pairs of the design's free optimum (empty when tight).
    pub rival_cross: Vec<(usize, usize)>,
    /// The design's CDS.
    pub cds: String,
}

/// How a run ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// The last design is modular.
    Tight,
    /// The cuts admit no design realizing `d` ([`Mode::Exact`]), or the
    /// multiplier search found none ([`Mode::Lagrange`]).
    Infeasible,
    /// A master table exceeded the cap.
    TooLarge(f64),
    /// The round limit was reached.
    Rounds,
    /// The time limit passed.
    Time,
}

/// A whole run from one start.
#[derive(Debug, Clone)]
pub struct Run {
    /// Every round, in order.
    pub rounds: Vec<Round>,
    /// How it ended.
    pub outcome: Outcome,
    /// Constrained energy of the flanks alone (dekacalories).
    pub shift: i32,
    /// Codon positions read by the cuts in force when the run ended.
    pub final_cut_set: Vec<usize>,
}

impl<'a> Problem<'a> {
    /// A problem over `menus` between the flanks, with tables capped at `cap`
    /// entries.
    pub fn new(
        params: &'a Params,
        utr5: &str,
        utr3: &str,
        menus: &Menus,
        cap: f64,
    ) -> Result<Self, CutError> {
        let menus = (0..menus.len())
            .map(|p| {
                menus
                    .options(p)
                    .iter()
                    .map(|(c, w)| {
                        let n = encode(c)?;
                        Ok(([n[0], n[1], n[2]], *w))
                    })
                    .collect::<Result<Vec<_>, CutError>>()
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            params,
            utr5: encode(utr5)?,
            utr3: encode(utr3)?,
            menus,
            cap,
            mode: Mode::Exact,
            limit: None,
        })
    }

    /// The same problem with master steps in `mode`.
    pub fn with_mode(mut self, mode: Mode) -> Self {
        self.mode = mode;
        self
    }

    /// The same problem with each run stopped after `limit` of wall clock,
    /// checked between rounds and between multiplier steps.
    pub fn with_limit(mut self, limit: std::time::Duration) -> Self {
        self.limit = Some(limit);
        self
    }

    fn out_of_time(&self, since: std::time::Instant) -> bool {
        self.limit.is_some_and(|l| since.elapsed() > l)
    }

    fn cds_range(&self) -> std::ops::Range<usize> {
        self.utr5.len()..self.utr5.len() + 3 * self.menus.len()
    }

    fn molecule(&self, asg: &[usize]) -> Vec<Nucleotide> {
        let mut seq = self.utr5.clone();
        for (p, &c) in asg.iter().enumerate() {
            seq.extend_from_slice(&self.menus[p][c].0);
        }
        seq.extend_from_slice(&self.utr3);
        seq
    }

    fn weight(&self, asg: &[usize]) -> f64 {
        asg.iter()
            .enumerate()
            .map(|(p, &c)| self.menus[p][c].1)
            .sum()
    }

    /// The menu indices spelling `cds`.
    pub fn assignment(&self, cds: &str) -> Result<Vec<usize>, CutError> {
        if cds.len() != 3 * self.menus.len() {
            return Err(CutError::Design(format!("CDS length {}", cds.len())));
        }
        let nts = encode(cds)?;
        (0..self.menus.len())
            .map(|p| {
                self.menus[p]
                    .iter()
                    .position(|(c, _)| c[..] == nts[3 * p..3 * p + 3])
                    .ok_or_else(|| CutError::Design(format!("codon at {p}")))
            })
            .collect()
    }

    /// A term as a factor over the codon positions it reads, or a constant.
    fn factor(&self, term: &Term, template: &mut [Nucleotide]) -> Result<Factor, f64> {
        let off = self.utr5.len();
        let cds = self.cds_range();
        let mut vars: Vec<usize> = term
            .reads()
            .into_iter()
            .filter(|p| cds.contains(p))
            .map(|p| (p - off) / 3)
            .collect();
        vars.dedup();
        let value = |seq: &[Nucleotide]| {
            term.energy(self.params, seq)
                .map_or(f64::INFINITY, |e| e as f64)
        };
        if vars.is_empty() {
            return Err(value(template));
        }
        let sizes: Vec<usize> = vars.iter().map(|&v| self.menus[v].len()).collect();
        let mut table = Vec::with_capacity(sizes.iter().product());
        let mut asg = vec![0; self.menus.len()];
        for_each_assignment(&vars, &sizes, &mut asg, |a| {
            for &v in &vars {
                let at = off + 3 * v;
                template[at..at + 3].copy_from_slice(&self.menus[v][a[v]].0);
            }
            table.push(value(template));
        });
        Ok(Factor { vars, sizes, table })
    }

    /// The factors and constant of a sum of terms.
    fn factors(&self, terms: &[Term], template: &mut [Nucleotide]) -> (Vec<Factor>, f64) {
        let mut out = Vec::new();
        let mut constant = 0.0;
        for t in terms {
            match self.factor(t, template) {
                Ok(f) => out.push(f),
                Err(c) => constant += c,
            }
        }
        (out, constant)
    }

    /// The cut `E(x, d) ≤ E(x, r)`, keeping only the terms the two
    /// structures do not share.
    fn cut(&self, d: &[Term], r: &[Term], template: &mut [Nucleotide]) -> Cut {
        let ds: HashSet<&Term> = d.iter().collect();
        let rs: HashSet<&Term> = r.iter().collect();
        let only_d: Vec<Term> = d.iter().filter(|t| !rs.contains(t)).cloned().collect();
        let only_r: Vec<Term> = r.iter().filter(|t| !ds.contains(t)).cloned().collect();
        let (plus, plus_const) = self.factors(&only_d, template);
        let (minus, minus_const) = self.factors(&only_r, template);
        Cut {
            plus,
            plus_const,
            minus,
            minus_const,
        }
    }

    fn check_size(&self, size: f64) -> Result<(), MasterFail> {
        if size > self.cap {
            Err(MasterFail::TooLarge(size))
        } else {
            Ok(())
        }
    }

    /// Minimize `Σ objective` over designs satisfying every cut.
    fn master(&self, objective: &[Factor], cuts: &[Cut]) -> Result<MasterSolution, MasterFail> {
        let m = self.menus.len();
        let size_of = |v: usize| self.menus[v].len();
        let mut keep = vec![false; m];
        for c in cuts {
            for v in c.vars() {
                keep[v] = true;
            }
        }
        let cut_set: Vec<usize> = (0..m).filter(|&v| keep[v]).collect();
        let mut pool: Vec<Factor> = objective.to_vec();
        let mut steps: Vec<(usize, Vec<Factor>)> = Vec::new();
        let mut asg = vec![0usize; m];
        let mut width: f64 = 0.0;
        let mut left: Vec<usize> = (0..m).filter(|&v| !keep[v]).collect();
        while !left.is_empty() {
            // Greedy order: the variable whose elimination builds the smallest table.
            let scope = |v: usize, pool: &[Factor]| {
                let mut s: Vec<usize> = pool
                    .iter()
                    .filter(|f| f.vars.contains(&v))
                    .flat_map(|f| f.vars.iter().copied())
                    .filter(|&u| u != v)
                    .collect();
                s.sort_unstable();
                s.dedup();
                s
            };
            let (at, v, s) = left
                .iter()
                .enumerate()
                .map(|(at, &v)| {
                    let s = scope(v, &pool);
                    (at, v, s)
                })
                .min_by(|a, b| {
                    let sa: f64 = a.2.iter().map(|&u| size_of(u) as f64).product();
                    let sb: f64 = b.2.iter().map(|&u| size_of(u) as f64).product();
                    sa.total_cmp(&sb)
                })
                .expect("non-empty");
            left.swap_remove(at);
            let sizes: Vec<usize> = s.iter().map(|&u| size_of(u)).collect();
            let entries: f64 = sizes.iter().map(|&z| z as f64).product::<f64>() * size_of(v) as f64;
            self.check_size(entries)?;
            width = width.max(entries);
            let (with, without): (Vec<Factor>, Vec<Factor>) =
                pool.into_iter().partition(|f| f.vars.contains(&v));
            pool = without;
            let mut table = Vec::with_capacity(sizes.iter().product());
            for_each_assignment(&s, &sizes, &mut asg, |a| {
                let mut a = a.to_vec();
                let best = (0..size_of(v))
                    .map(|c| {
                        a[v] = c;
                        with.iter().map(|f| f.value(&a)).sum::<f64>()
                    })
                    .fold(f64::INFINITY, f64::min);
                table.push(best);
            });
            pool.push(Factor {
                vars: s,
                sizes,
                table,
            });
            steps.push((v, with));
        }
        let sizes: Vec<usize> = cut_set.iter().map(|&u| size_of(u)).collect();
        let space: f64 = sizes.iter().map(|&z| z as f64).product();
        self.check_size(space)?;
        let mut best: Option<(f64, Vec<usize>)> = None;
        for_each_assignment(&cut_set, &sizes, &mut asg, |a| {
            let value: f64 = pool.iter().map(|f| f.value(a)).sum();
            if value.is_finite()
                && best.as_ref().is_none_or(|(b, _)| value < *b)
                && cuts.iter().all(|c| c.holds(a))
            {
                best = Some((value, a.to_vec()));
            }
        });
        let (objective, mut asg) = best.ok_or(MasterFail::Infeasible)?;
        for (v, with) in steps.iter().rev() {
            let c = (0..size_of(*v))
                .map(|c| {
                    asg[*v] = c;
                    (c, with.iter().map(|f| f.value(&asg)).sum::<f64>())
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("non-empty menu")
                .0;
            asg[*v] = c;
        }
        Ok(MasterSolution {
            asg,
            objective,
            cut_set,
            space,
            width,
        })
    }

    /// The objective with each cut folded in at its multiplier.
    fn penalized(objective: &[Factor], cuts: &[Cut], multipliers: &[f64]) -> Vec<Factor> {
        let mut out = objective.to_vec();
        for (c, &l) in cuts.iter().zip(multipliers) {
            if l == 0.0 {
                continue;
            }
            for f in &c.plus {
                out.push(Factor {
                    table: f.table.iter().map(|v| l * v).collect(),
                    ..f.clone()
                });
            }
            for f in &c.minus {
                out.push(Factor {
                    table: f
                        .table
                        .iter()
                        .map(|v| -l * if v.is_finite() { *v } else { UNSPELLABLE })
                        .collect(),
                    ..f.clone()
                });
            }
        }
        out
    }

    /// A Lagrangian master step ([`Mode::Lagrange`]): the first design whose
    /// penalized optimum satisfies every cut, with the multipliers it took.
    fn master_lagrange(
        &self,
        objective: &[Factor],
        cuts: &[Cut],
        multipliers: &mut Vec<f64>,
        steps: usize,
        since: std::time::Instant,
    ) -> Result<(MasterSolution, f64), MasterFail> {
        multipliers.resize(cuts.len(), 0.0);
        for _ in 0..=steps {
            if self.out_of_time(since) {
                return Err(MasterFail::Time);
            }
            let mut sol = self.master(&Self::penalized(objective, cuts, multipliers), &[])?;
            let violated: Vec<usize> = (0..cuts.len())
                .filter(|&k| !cuts[k].holds(&sol.asg))
                .collect();
            if violated.is_empty() {
                let plain: f64 = objective.iter().map(|f| f.value(&sol.asg)).sum();
                sol.objective = plain;
                sol.cut_set = cuts.iter().flat_map(|c| c.vars()).collect();
                sol.cut_set.sort_unstable();
                sol.cut_set.dedup();
                let total: f64 = multipliers.iter().sum();
                return Ok((sol, total));
            }
            for k in violated {
                multipliers[k] = (2.0 * multipliers[k]).max(0.05);
            }
        }
        Err(MasterFail::Infeasible)
    }

    /// Run the loop from `start` (a CDS) at slope `mu` (dekacalories per unit
    /// of adaptation) for at most `max_rounds` rounds.
    pub fn run(&self, start: &str, mu: f64, max_rounds: usize) -> Result<Run, CutError> {
        let params = self.params;
        let start_asg = self.assignment(start)?;
        let cds = self.cds_range();
        let (s5, e5) = fold(params, &self.utr5);
        let (s3, e3) = fold(params, &self.utr3);
        let start_cds = self.molecule(&start_asg)[cds.clone()].to_vec();
        let (sc, _) = fold(params, &start_cds);
        let d = format!("{s5}{sc}{s3}");
        let d_terms = terms(&pair_table(&d)?);
        let mut template = self.molecule(&start_asg);
        let (mut objective, d_const) = self.factors(&d_terms, &mut template);
        for p in 0..self.menus.len() {
            objective.push(Factor {
                vars: vec![p],
                sizes: vec![self.menus[p].len()],
                table: self.menus[p].iter().map(|(_, w)| -mu * w).collect(),
            });
        }
        let mut cuts: Vec<Cut> = Vec::new();
        let mut multipliers: Vec<f64> = Vec::new();
        let mut rivals: Vec<String> = Vec::new();
        let mut rounds = Vec::new();
        let since = std::time::Instant::now();
        let outcome = loop {
            if rounds.len() >= max_rounds {
                break Outcome::Rounds;
            }
            if self.out_of_time(since) {
                break Outcome::Time;
            }
            let solved = match self.mode {
                Mode::Exact => self.master(&objective, &cuts).map(|s| (s, 0.0)),
                Mode::Lagrange { steps } => {
                    self.master_lagrange(&objective, &cuts, &mut multipliers, steps, since)
                }
            };
            let (sol, multiplier) = match solved {
                Ok(s) => s,
                Err(MasterFail::Infeasible) => break Outcome::Infeasible,
                Err(MasterFail::TooLarge(z)) => break Outcome::TooLarge(z),
                Err(MasterFail::Time) => break Outcome::Time,
            };
            let seq = self.molecule(&sol.asg);
            let weight = self.weight(&sol.asg);
            self.verify(&seq, &d, &rivals, sol.objective + d_const + mu * weight)?;
            let (_, ec) = fold(params, &seq[cds.clone()]);
            let constrained = e5 + ec + e3;
            let (free_s, free) = fold(params, &seq);
            let free_pairs = pair_table(&free_s)?;
            let tight = free == constrained;
            let rival_cross = if tight {
                Vec::new()
            } else {
                cross_pairs(&free_pairs, cds.clone())
            };
            if !tight && rival_cross.is_empty() {
                return Err(CutError::Check(
                    "free optimum below constrained with no cross pair".into(),
                ));
            }
            rounds.push(Round {
                cut_set: sol.cut_set,
                space: sol.space,
                width: sol.width,
                multiplier,
                objective: sol.objective + d_const,
                constrained,
                free,
                weight,
                rival_cross,
                cds: decode(&seq[cds.clone()]),
            });
            if tight {
                break Outcome::Tight;
            }
            if rivals.contains(&free_s) {
                return Err(CutError::Check("a cut rival returned".into()));
            }
            cuts.push(self.cut(&d_terms, &terms(&free_pairs), &mut template));
            rivals.push(free_s);
        };
        let mut final_cut_set: Vec<usize> = cuts.iter().flat_map(|c| c.vars()).collect();
        final_cut_set.sort_unstable();
        final_cut_set.dedup();
        Ok(Run {
            rounds,
            outcome,
            shift: e5 + e3,
            final_cut_set,
        })
    }

    /// Check a master design against the evaluator: its energy on `d` matches
    /// the master's account, and every cut holds on the evaluated energies.
    fn verify(
        &self,
        seq: &[Nucleotide],
        d: &str,
        rivals: &[String],
        expected: f64,
    ) -> Result<(), CutError> {
        let bytes = decode(seq);
        let ed = rna_mfe::evaluate(self.params, bytes.as_bytes(), d)
            .map_err(|e| CutError::Check(format!("{e:?}")))?
            .0;
        if (ed as f64 - expected).abs() > 1e-6 {
            return Err(CutError::Check(format!("E(x,d) {ed} vs master {expected}")));
        }
        for r in rivals {
            let pairs = pair_table(r)?;
            let spellable = (0..pairs.len()).all(|i| {
                pairs[i].is_none_or(|j| pair_from_nucleotides(seq[i], seq[j]) != BasePair::NN)
            });
            if !spellable {
                continue;
            }
            let er = rna_mfe::evaluate(self.params, bytes.as_bytes(), r)
                .map_err(|e| CutError::Check(format!("{e:?}")))?
                .0;
            if ed > er {
                return Err(CutError::Check(format!(
                    "cut broken: E(x,d) {ed} > E(x,r) {er}"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 33
    }

    #[test]
    fn terms_sum_to_the_evaluator() {
        let params = Params::load_default().unwrap();
        let mut seed = 7u64;
        let mut checked = 0;
        for _ in 0..200 {
            let n = 30 + (lcg(&mut seed) % 60) as usize;
            let seq: String = (0..n)
                .map(|_| b"ACGU"[(lcg(&mut seed) % 4) as usize] as char)
                .collect();
            let (s, e) = rna_mfe::fold(&params, seq.as_bytes()).unwrap();
            let nts = encode(&seq).unwrap();
            let total: i32 = terms(&pair_table(&s).unwrap())
                .iter()
                .map(|t| t.energy(&params, &nts).unwrap())
                .sum();
            assert_eq!(total, e.0, "{seq} {s}");
            checked += usize::from(s.contains('('));
        }
        assert!(checked > 100, "too few folded structures: {checked}");
    }

    #[test]
    fn terms_read_only_their_positions() {
        // Changing a position no term reads leaves every term's energy unchanged.
        let params = Params::load_default().unwrap();
        let mut seed = 11u64;
        for _ in 0..100 {
            let n = 40;
            let seq: String = (0..n)
                .map(|_| b"ACGU"[(lcg(&mut seed) % 4) as usize] as char)
                .collect();
            let (s, _) = rna_mfe::fold(&params, seq.as_bytes()).unwrap();
            let nts = encode(&seq).unwrap();
            for t in terms(&pair_table(&s).unwrap()) {
                let reads = t.reads();
                let base = t.energy(&params, &nts);
                for p in (0..n).filter(|p| !reads.contains(p)) {
                    for b in [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U] {
                        let mut alt = nts.clone();
                        alt[p] = b;
                        assert_eq!(t.energy(&params, &alt), base, "{t:?} reads {p}");
                    }
                }
            }
        }
    }

    /// Every design of a short protein between flanks: the brute-force side
    /// of the master test.
    fn every_design(problem: &Problem) -> Vec<Vec<usize>> {
        let m = problem.menus.len();
        let vars: Vec<usize> = (0..m).collect();
        let sizes: Vec<usize> = vars.iter().map(|&v| problem.menus[v].len()).collect();
        let mut out = Vec::new();
        let mut asg = vec![0; m];
        for_each_assignment(&vars, &sizes, &mut asg, |a| out.push(a.to_vec()));
        out
    }

    fn spells(seq: &str, structure: &str) -> bool {
        let p = pair_table(structure).unwrap();
        let nts = encode(seq).unwrap();
        (0..p.len())
            .all(|i| p[i].is_none_or(|j| pair_from_nucleotides(nts[i], nts[j]) != BasePair::NN))
    }

    #[test]
    fn master_matches_brute_force() {
        // A short protein between short flanks. The rival is the free fold of
        // the unconstrained master's own design, as in the loop, chosen on a
        // flank pair where that design loses to it — so the cut removes the
        // unconstrained optimum and must move the minimum.
        let params = Params::load_default().unwrap();
        let usage = crate::CodonUsage::load_default().unwrap();
        let menus = Menus::new("MLRSG", &usage).unwrap();
        let mu = 50.0;
        let flanks = [
            ("GGGAC", "GUCCC"),
            ("CCGCC", "GGCGG"),
            ("CAUCC", "GGAUG"),
            ("CCAGC", "GCUGG"),
            ("GCCGCCAUG", "CAUGGCGGC"),
            ("CGGCGCC", "GGCGCCG"),
        ];
        let mut exercised = 0;
        for (u5, u3) in flanks {
            let problem = Problem::new(&params, u5, u3, &menus, 1e9).unwrap();
            let all = every_design(&problem);
            let start = problem.molecule(&all[0]);
            let (sc, _) = fold(&params, &start[problem.cds_range()]);
            let (s5, _) = fold(&params, &problem.utr5);
            let (s3, _) = fold(&params, &problem.utr3);
            let d = format!("{s5}{sc}{s3}");
            let d_terms = terms(&pair_table(&d).unwrap());
            let mut template = start.clone();
            let (mut objective, d_const) = problem.factors(&d_terms, &mut template);
            for p in 0..problem.menus.len() {
                objective.push(Factor {
                    vars: vec![p],
                    sizes: vec![problem.menus[p].len()],
                    table: problem.menus[p].iter().map(|(_, w)| -mu * w).collect(),
                });
            }
            let free = problem.master(&objective, &[]).unwrap();
            let x0 = decode(&problem.molecule(&free.asg));
            let (r, er0) = rna_mfe::fold(&params, x0.as_bytes()).unwrap();
            let ed0 = rna_mfe::evaluate(&params, x0.as_bytes(), &d).unwrap();
            if ed0 <= er0 {
                continue; // the unconstrained optimum survives this rival
            }
            exercised += 1;
            let cuts = [problem.cut(&d_terms, &terms(&pair_table(&r).unwrap()), &mut template)];
            let mut least = Vec::new();
            for cuts in [&[][..], &cuts[..]] {
                let brute = all
                    .iter()
                    .filter_map(|a| {
                        let seq = decode(&problem.molecule(a));
                        if !spells(&seq, &d) {
                            return None;
                        }
                        let ed = rna_mfe::evaluate(&params, seq.as_bytes(), &d).unwrap().0;
                        if !cuts.is_empty() && spells(&seq, &r) {
                            let er = rna_mfe::evaluate(&params, seq.as_bytes(), &r).unwrap().0;
                            if ed > er {
                                return None;
                            }
                        }
                        Some(ed as f64 - mu * problem.weight(a))
                    })
                    .fold(f64::INFINITY, f64::min);
                let sol = problem.master(&objective, cuts).unwrap();
                let got = sol.objective + d_const;
                assert!(
                    (got - brute).abs() < 1e-6,
                    "{u5}/{u3}: master {got} vs brute {brute}"
                );
                least.push(brute);
            }
            assert!(
                least[1] > least[0],
                "{u5}/{u3}: the cut did not move the minimum"
            );
        }
        assert!(exercised > 0, "no flank pair exercised a binding cut");
    }
}
