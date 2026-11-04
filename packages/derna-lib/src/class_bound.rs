//! A lower bound on the scalarized objective over the items whose own optimal
//! configuration has a given outermost-loop shape.
//!
//! # Why the outermost loop is enough
//!
//! At `dangles = 0` the outermost loop is additive: a configuration's cost is
//! the sum of its top-level pairs' closed costs, each plus its terminal-AU term,
//! and unpaired symbols there cost nothing. The scan's `C` table is the least
//! combined score over every prefix parse, `C = C + unpaired | C + P`. Every
//! closed-pair cell `P` holds the least score of a pair *and everything inside
//! it*, and nothing inside a top-level pair sees the outermost loop. So a class
//! of configurations defined only by their outermost loop — how many top-level
//! pairs, how long the run before the first one, which symbols the first one
//! closes on — is minimized by the same exterior recurrence with a little more
//! state, read over the finished `P` cells of an exact scan.
//!
//! # What the bound is for
//!
//! Necessary conditions on a tight item (the clamp table's branch counts, the
//! leading-run gains) are statements about **every** optimal configuration of
//! the item alone: if one optimal configuration lies outside the class, a
//! context configuration beats the constrained cost and the item is not tight. So
//! for a tight item `x` with own optimum `d*`, `d*` is in the class, and
//!
//! ```text
//! min over (x, d), d in class, of λ·cost(d) − (1 − λ)·A(x)  ≤  λ·E(x) − (1 − λ)·A(x),
//! ```
//!
//! a lower bound on the tight front's scalarized value at every λ. It is never
//! below the relaxed value (a restriction of the same minimum), and equals it
//! when the class admits everything.

use rna_mfe::loops::pair_from_nucleotides;
use rna_mfe::{Energy, Nucleotide, Params};

use crate::dfa::{Dfa, Node};
use crate::state::{Back, PairRef, State};
use crate::tables::Tables;

/// An outermost-loop shape: the configurations of the item alone whose
/// top-level pairs satisfy every bound given. `ExteriorClass::default()` admits
/// every configuration.
#[derive(Debug, Clone, Default)]
pub struct ExteriorClass {
    /// Fewest top-level pairs.
    pub min_branches: u8,
    /// Most top-level pairs, if bounded.
    pub max_branches: Option<u8>,
    /// Longest unmatched run before the first top-level pair, if bounded. A
    /// configuration with no pair has a leading run of the whole item.
    pub max_lead: Option<u32>,
    /// Symbols the first top-level pair may close on (its 3′ symbol), if bounded.
    pub first_close: Option<Vec<Nucleotide>>,
    /// Longest unmatched run after the last top-level pair, if bounded. No
    /// end-local bound proves this one necessary, so a class that sets it
    /// shapes seeds, not a lower bound.
    pub max_trail: Option<u32>,
}

/// The best state reached at the item's end, overall and per branch count.
#[derive(Debug, Clone)]
pub struct ClassOptimum {
    /// The least score over the class, if the class is non-empty on this item set.
    pub best: Option<State>,
    /// Per counted branch number `b` (the last entry counts every `b` at or
    /// above it when the class has no upper bound), the least score with `b`
    /// top-level pairs, before the class's bounds on `b` are applied.
    pub by_branches: Vec<Option<State>>,
    /// The best parse's exterior cells from the start node to the end, each
    /// with its [`Back::ExteriorUnpaired`] / [`Back::ExteriorPair`] pointer:
    /// written into a `C` column, [`crate::traceback::recover`] rebuilds the
    /// minimizing item and its in-class configuration from it.
    pub path: Vec<(Node, State)>,
}

impl ExteriorClass {
    /// Branch counts the recurrence tracks: `0..=top`, the last saturating when
    /// the class has no upper bound. Count 0 always means "no pair yet", which
    /// the leading-run and first-pair bounds read.
    fn top(&self) -> u8 {
        self.max_branches.unwrap_or(self.min_branches.max(1))
    }

    fn admits_first(&self, nuc_close: Nucleotide) -> bool {
        self.first_close
            .as_ref()
            .is_none_or(|set| set.contains(&nuc_close))
    }
}

/// One cell of the class recurrence: the best state and the row of the cell
/// its back-pointer names. Rows `0..=top` count top-level pairs; with a
/// trailing-run bound, rows `top + 1 + b` hold the parses whose last pair is
/// placed (`b` pairs, only unmatched symbols after it).
type Cell = Option<(State, usize)>;

/// The least combined score at `tables`' λ over every item and every
/// configuration in `class`, read off the closed-pair cells of an exact scan.
///
/// With `ExteriorClass::default()` this reproduces `tables.c[dfa.end()]`'s
/// score (the scan's own exterior recurrence).
pub fn class_optimum(
    params: &Params,
    dfa: &Dfa,
    tables: &Tables,
    class: &ExteriorClass,
) -> ClassOptimum {
    let lambda = tables.lambda();
    let top = class.top() as usize;
    let saturate = class.max_branches.is_none();
    let width = 2 * (dfa.seq_len as usize + 1);
    let tail = top + 1;
    let tail_from = class.max_trail.map(|t| dfa.seq_len.saturating_sub(t));
    // c[row][node]: the best prefix parse ending at `node` in `row` (see [`Cell`]).
    let rows = if tail_from.is_some() { 2 * tail } else { tail };
    let mut c: Vec<Vec<Cell>> = vec![vec![None; width]; rows];
    c[0][dfa.ix(dfa.start())] = Some((State::new(Energy::ZERO, 0.0, Back::ExteriorEmpty), 0));
    let merge = |slot: &mut Cell, cand: State, from: usize| match slot {
        Some((old, _)) if old.brd(lambda) <= cand.brd(lambda) => {}
        _ => *slot = Some((cand, from)),
    };
    let au = Energy(params.misc.terminal_au_37);
    for pos in 0..=dfa.seq_len {
        for &o in dfa.nodes_at(pos) {
            for (key, st) in &tables.p[dfa.ix(o)] {
                let pair = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
                let term = if pair.applies_terminal_au() {
                    au
                } else {
                    Energy::ZERO
                };
                for b in 0..=top {
                    let Some((prefix, _)) = c[b][dfa.ix(key.open)] else {
                        continue;
                    };
                    if b == 0 && !class.admits_first(key.nuc_close) {
                        continue;
                    }
                    let to = match (b < top, saturate) {
                        (true, _) => b + 1,
                        (false, true) => top,
                        (false, false) => continue,
                    };
                    let back = Back::ExteriorPair {
                        prefix: key.open,
                        pair: PairRef {
                            outer: o,
                            key: *key,
                        },
                    };
                    let cand = State::new(prefix.mfe + st.mfe + term, prefix.cai + st.cai, back);
                    merge(&mut c[to][dfa.ix(o)], cand, b);
                    if tail_from.is_some_and(|t| pos >= t) {
                        merge(&mut c[tail + to][dfa.ix(o)], cand, b);
                    }
                }
            }
            for (r, row) in c.iter_mut().enumerate() {
                let Some((st, _)) = row[dfa.ix(o)] else {
                    continue;
                };
                if r == 0 && class.max_lead.is_some_and(|a| pos >= a) {
                    continue;
                }
                for e in dfa.right(o) {
                    let back = Back::ExteriorUnpaired {
                        prev: o,
                        nuc: e.nuc,
                    };
                    merge(
                        &mut row[dfa.ix(e.to)],
                        State::new(st.mfe, st.cai + e.cai, back),
                        r,
                    );
                }
            }
        }
    }
    let end = dfa.end();
    let read = if tail_from.is_some() { tail } else { 0 };
    let by_branches: Vec<Option<State>> = c[read..read + tail]
        .iter()
        .map(|col| col[dfa.ix(end)].map(|(s, _)| s))
        .collect();
    let mut best: Cell = None;
    for (b, st) in by_branches.iter().enumerate() {
        if let Some(st) = *st {
            if b >= class.min_branches as usize {
                merge(&mut best, st, read + b);
            }
        }
    }
    let mut path = Vec::new();
    if let Some((_, row)) = best {
        let (mut node, mut row) = (end, row);
        while let Some((st, from)) = c[row][dfa.ix(node)] {
            path.push((node, st));
            match st.back {
                Back::ExteriorUnpaired { prev, .. } => node = prev,
                Back::ExteriorPair { prefix, .. } => node = prefix,
                _ => break,
            }
            row = from;
        }
        path.reverse();
    }
    ClassOptimum {
        best: best.map(|(s, _)| s),
        by_branches,
        path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dfa;
    use crate::scan::scan;
    use crate::{CodonUsage, ScanConfig};
    use derna_codon_dp::codon_table::aa_to_index;

    const PROTEINS: [&str; 2] = ["MYGKIIFVLM", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"];

    fn score(class: &ExteriorClass, protein: &str, lambda: f64) -> (Option<f64>, f64) {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let aa: Vec<usize> = protein
            .bytes()
            .map(|b| aa_to_index(b as char).unwrap())
            .collect();
        let graph = dfa::build(&aa, &usage);
        let tables = scan(&params, &graph, lambda, ScanConfig::default());
        let relaxed = tables.c[graph.ix(graph.end())].unwrap().brd(lambda);
        let best = class_optimum(&params, &graph, &tables, class).best;
        (best.map(|s| s.brd(lambda)), relaxed)
    }

    fn exactly(b: u8) -> ExteriorClass {
        ExteriorClass {
            min_branches: b,
            max_branches: Some(b),
            ..ExteriorClass::default()
        }
    }

    #[test]
    fn the_unrestricted_class_reproduces_the_scan() {
        for protein in PROTEINS {
            for lambda in [0.1, 0.5, 1.0] {
                let (best, relaxed) = score(&ExteriorClass::default(), protein, lambda);
                assert_eq!(best, Some(relaxed), "{protein} at λ = {lambda}");
            }
        }
    }

    /// The classes "exactly b pairs" for b below a cutoff and "at least the
    /// cutoff" partition every configuration, so their least value is the
    /// relaxed one and none is below it.
    #[test]
    fn branch_counts_partition_the_optimum() {
        for protein in PROTEINS {
            for lambda in [0.2, 1.0] {
                let mut parts: Vec<ExteriorClass> = (0..4).map(exactly).collect();
                parts.push(ExteriorClass {
                    min_branches: 4,
                    ..ExteriorClass::default()
                });
                let values: Vec<f64> = parts
                    .iter()
                    .filter_map(|c| score(c, protein, lambda).0)
                    .collect();
                let relaxed = score(&ExteriorClass::default(), protein, lambda).1;
                let least = values.iter().copied().fold(f64::INFINITY, f64::min);
                assert!((least - relaxed).abs() < 1e-9, "{protein} at λ = {lambda}");
                assert!(values.iter().all(|&v| v >= relaxed - 1e-9));
            }
        }
    }

    /// A tighter leading-run cap never lowers the value; a cap of the whole
    /// item, or every symbol allowed at the first pair's 3′ end, is no bound.
    #[test]
    fn leading_run_and_first_pair_bounds_are_monotone_and_vacuous_at_their_limits() {
        let protein = PROTEINS[1];
        let lambda = 0.5;
        let relaxed = score(&ExteriorClass::default(), protein, lambda).1;
        let lead = |a: u32| {
            let class = ExteriorClass {
                max_lead: Some(a),
                ..ExteriorClass::default()
            };
            score(&class, protein, lambda).0.unwrap_or(f64::INFINITY)
        };
        let caps: Vec<f64> = [0, 1, 2, 5, 20].map(lead).to_vec();
        assert!(caps.windows(2).all(|w| w[0] >= w[1] - 1e-9), "{caps:?}");
        assert!((lead(3 * protein.len() as u32) - relaxed).abs() < 1e-9);
        let every = ExteriorClass {
            first_close: Some(vec![
                Nucleotide::A,
                Nucleotide::C,
                Nucleotide::G,
                Nucleotide::U,
            ]),
            ..ExteriorClass::default()
        };
        assert_eq!(score(&every, protein, lambda).0, Some(relaxed));
        let close_on_u = ExteriorClass {
            first_close: Some(vec![Nucleotide::U]),
            ..ExteriorClass::default()
        };
        assert!(score(&close_on_u, protein, lambda).0.unwrap() >= relaxed - 1e-9);
    }

    /// The path, written into a `C` column, traces back to an item and a
    /// configuration in the class whose energy, scored independently, is the
    /// class value.
    #[test]
    fn the_path_traces_back_to_an_in_class_configuration_of_the_class_value() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let aa: Vec<usize> = PROTEINS[1]
            .bytes()
            .map(|b| aa_to_index(b as char).unwrap())
            .collect();
        let graph = dfa::build(&aa, &usage);
        let mut tables = scan(&params, &graph, 0.5, ScanConfig::default());
        for class in [exactly(2), exactly(1)] {
            let opt = class_optimum(&params, &graph, &tables, &class);
            let mut column = vec![None; tables.c.len()];
            for &(node, st) in &opt.path {
                column[graph.ix(node)] = Some(st);
            }
            std::mem::swap(&mut tables.c, &mut column);
            let rec = crate::traceback::recover(&params, &graph, &tables);
            std::mem::swap(&mut tables.c, &mut column);
            let energy =
                rna_mfe::evaluate(&params, rec.sequence.as_bytes(), &rec.structure).unwrap();
            assert_eq!(energy, opt.best.unwrap().mfe, "{class:?}");
            let (mut top, mut depth) = (0, 0);
            for ch in rec.structure.bytes() {
                match ch {
                    b'(' => {
                        top += u8::from(depth == 0);
                        depth += 1;
                    }
                    b')' => depth -= 1,
                    _ => {}
                }
            }
            assert_eq!(Some(top), class.max_branches, "{}", rec.structure);
        }
    }

    /// A trailing-run bound of the whole item is no bound; a tighter one never
    /// lowers the value, and its traced configuration ends within the bound and
    /// re-scores to the class value.
    #[test]
    fn the_trailing_run_bound_is_monotone_and_its_path_respects_it() {
        let params = Params::load_default().unwrap();
        let usage = CodonUsage::load_default().unwrap();
        let aa: Vec<usize> = PROTEINS[1]
            .bytes()
            .map(|b| aa_to_index(b as char).unwrap())
            .collect();
        let graph = dfa::build(&aa, &usage);
        let mut tables = scan(&params, &graph, 0.5, ScanConfig::default());
        let with = |t: Option<u32>| ExteriorClass {
            max_trail: t,
            ..exactly(1)
        };
        let value = |tables: &Tables, class: &ExteriorClass| {
            class_optimum(&params, &graph, tables, class)
                .best
                .map_or(f64::INFINITY, |s| s.brd(0.5))
        };
        let open = value(&tables, &with(None));
        assert_eq!(value(&tables, &with(Some(graph.seq_len))), open);
        let caps: Vec<f64> = [0, 1, 3, 10]
            .map(|t| value(&tables, &with(Some(t))))
            .to_vec();
        assert!(caps.windows(2).all(|w| w[0] >= w[1] - 1e-9), "{caps:?}");
        for t in [0, 2] {
            let opt = class_optimum(&params, &graph, &tables, &with(Some(t)));
            let mut column = vec![None; tables.c.len()];
            for &(node, st) in &opt.path {
                column[graph.ix(node)] = Some(st);
            }
            std::mem::swap(&mut tables.c, &mut column);
            let rec = crate::traceback::recover(&params, &graph, &tables);
            std::mem::swap(&mut tables.c, &mut column);
            let trail = rec.structure.len() - rec.structure.trim_end_matches('.').len();
            assert!(trail <= t as usize, "{}", rec.structure);
            let energy =
                rna_mfe::evaluate(&params, rec.sequence.as_bytes(), &rec.structure).unwrap();
            assert_eq!(energy, opt.best.unwrap().mfe);
        }
    }
}
