//! Backward walk over the filled CKY tables to recover the chosen
//! coding sequence and its dot-bracket structure together.
//!
//! Starting from `C[end]`, each cell's [`Back`] pointer names its
//! predecessor cells and the locally chosen nucleotides; loop interiors
//! the forward scan jumped over are respelled by the best-CAI DFA path
//! between their boundaries. The walk writes nucleotides and brackets
//! directly into position-indexed buffers, since every CKY coordinate
//! lands at a unique nucleotide slot.

use rna_mfe::loops::pair_from_nucleotides;
use rna_mfe::{Nucleotide, Params};

use crate::dfa::{Dfa, Node};
use crate::interior::{best_hairpin_interior, best_internal_interior, LoopPairs, Span};
use crate::state::{Back, PairKey, PairRef, SpanRef};
use crate::tables::Tables;

/// Recovered design: the spelled RNA and its structure.
pub struct Recovered {
    /// Coding sequence as ASCII `A/C/G/U`.
    pub sequence: String,
    /// Dot-bracket structure, same length as the sequence.
    pub structure: String,
}

/// Walk back from `C[end]` to rebuild the `(sequence, structure)` pair.
pub fn recover(params: &Params, dfa: &Dfa, tables: &Tables) -> Recovered {
    recover_at(params, dfa, tables, dfa.end())
}

/// [`recover`] from the terminal `end`, one of [`Dfa::ends`]: on a
/// stop-labelled lattice, the design whose stop is `end`'s.
pub fn recover_at(params: &Params, dfa: &Dfa, tables: &Tables, end: Node) -> Recovered {
    let n = dfa.seq_len as usize;
    let mut w = Walk {
        params,
        dfa,
        tables,
        seq: vec![Nucleotide::A; n],
        dots: vec![b'.'; n],
    };
    w.walk_c(end);
    Recovered {
        sequence: w.seq.iter().map(|&nt| nuc_char(nt)).collect(),
        structure: String::from_utf8(w.dots).expect("dot-bracket bytes are ASCII"),
    }
}

struct Walk<'a> {
    params: &'a Params,
    dfa: &'a Dfa,
    tables: &'a Tables,
    seq: Vec<Nucleotide>,
    dots: Vec<u8>,
}

impl Walk<'_> {
    fn walk_c(&mut self, node: Node) {
        let mut cursor = Some(node);
        while let Some(node) = cursor {
            let Some(state) = self.tables.c[self.dfa.ix(node)] else {
                return;
            };
            match state.back {
                Back::ExteriorEmpty => return,
                Back::ExteriorUnpaired { prev, nuc } => {
                    self.seq[prev.pos as usize] = nuc;
                    cursor = Some(prev);
                }
                Back::ExteriorPair { prefix, pair } => {
                    self.walk_p(pair);
                    cursor = Some(prefix);
                }
                _ => return,
            }
        }
    }

    fn walk_p(&mut self, p: PairRef) {
        let (a, b) = self.bracket_pair(p.outer, p.key);
        let Some(state) = self.tables.p[self.dfa.ix(p.outer)].get(&p.key) else {
            return;
        };
        match state.back {
            Back::Hairpin => self.fill_hairpin(p.key, a + 1),
            Back::Single { inner } => self.walk_single(p, inner, a, b),
            Back::MultiClose { multi } => self.walk_multi(multi),
            _ => {}
        }
    }

    fn walk_single(&mut self, outer: PairRef, inner: PairRef, a: u32, b: u32) {
        let ib = inner.key.close.pos;
        let left_from = self.after_open(outer.key);
        let n1 = inner.key.open.pos - left_from.pos;
        let n2 = outer.key.close.pos - inner.outer.pos;
        if n1 > 0 && n2 > 0 {
            self.fill_internal(outer, inner, a + 1, ib + 1);
        } else {
            // Stack or bulge: interior is energy-irrelevant, best-CAI.
            self.fill_span(left_from, inner.key.open, a + 1);
            self.fill_span(inner.outer, outer.key.close, ib + 1);
        }
        let _ = b;
        self.walk_p(inner);
    }

    /// Respell both flanks of an internal loop, re-running the same joint
    /// optimization the scan used so the spelled terminal mismatches match
    /// the ones that were scored.
    fn fill_internal(&mut self, outer: PairRef, inner: PairRef, left_slot: u32, right_slot: u32) {
        let pairs = LoopPairs {
            outer: pair_from_nucleotides(outer.key.nuc_open(), outer.key.nuc_close),
            inner: pair_from_nucleotides(inner.key.nuc_open(), inner.key.nuc_close),
        };
        let Some(si) = best_internal_interior(
            self.params,
            self.dfa,
            self.tables.lambda(),
            &pairs,
            Span {
                from: self.after_open(outer.key),
                to: inner.key.open,
            },
            Span {
                from: inner.outer,
                to: outer.key.close,
            },
        ) else {
            return;
        };
        for (k, nt) in si.left.into_iter().enumerate() {
            self.seq[left_slot as usize + k] = nt;
        }
        for (k, nt) in si.right.into_iter().enumerate() {
            self.seq[right_slot as usize + k] = nt;
        }
    }

    fn walk_multi(&mut self, m: PairRef) {
        let Some(state) = self.tables.multi[self.dfa.ix(m.outer)].get(&m.key) else {
            return;
        };
        let Back::MultiWrap { core } = state.back else {
            return;
        };
        let a = m.key.open.pos;
        self.fill_span(self.after_open(m.key), core.inner, a + 1);
        self.fill_span(core.outer, m.key.close, core.outer.pos);
        self.walk_m2(core);
    }

    fn walk_m2(&mut self, span: SpanRef) {
        let Some(state) = self.tables.m2[self.dfa.ix(span.outer)].get(span.inner) else {
            return;
        };
        if let Back::M2Bifurcation { left, right } = state.back {
            self.walk_m(left);
            self.walk_p(right);
        }
    }

    fn walk_m(&mut self, span: SpanRef) {
        let mut cursor = Some(span);
        while let Some(s) = cursor {
            let Some(state) = self.tables.m[self.dfa.ix(s.outer)].get(s.inner) else {
                return;
            };
            match state.back {
                Back::MFromP { pair } => return self.walk_p(pair),
                Back::MFromM2 { m2 } => return self.walk_m2(m2),
                Back::MUnpaired { prev, nuc } => {
                    self.seq[prev.outer.pos as usize] = nuc;
                    cursor = Some(prev);
                }
                _ => return,
            }
        }
    }

    /// Emit the pair brackets and boundary nucleotides for a closed pair
    /// `(a, b)`, returning `(a, b)`.
    fn bracket_pair(&mut self, outer: Node, key: PairKey) -> (u32, u32) {
        let a = key.open.pos;
        let b = outer.pos - 1;
        self.seq[a as usize] = key.nuc_open();
        self.seq[b as usize] = key.nuc_close;
        self.dots[a as usize] = b'(';
        self.dots[b as usize] = b')';
        (a, b)
    }

    /// The DFA node one position past a pair's 5' open, following the
    /// edge that spells the opening nucleotide.
    fn after_open(&self, key: PairKey) -> Node {
        key.open_edge(self.dfa).map_or(key.open, |e| e.to)
    }

    /// Respell a hairpin loop interior, re-running the same joint
    /// energy/CAI optimization the scan used so the spelled nucleotides
    /// match the ones that were scored.
    fn fill_hairpin(&mut self, key: PairKey, start_slot: u32) {
        let from = self.after_open(key);
        let closing = pair_from_nucleotides(key.nuc_open(), key.nuc_close);
        let Some(interior) = best_hairpin_interior(
            self.params,
            self.dfa,
            self.tables.lambda(),
            from,
            key.close,
            closing,
        ) else {
            return;
        };
        for (k, nt) in interior.seq.into_iter().enumerate() {
            self.seq[start_slot as usize + k] = nt;
        }
    }

    /// Respell and mark unpaired the best-CAI interior between two
    /// boundary nodes, writing nucleotides from slot `start_slot`.
    fn fill_span(&mut self, from: Node, to: Node, start_slot: u32) {
        let interior = self.dfa.broken_codon_seq(from, to);
        for (k, nt) in interior.into_iter().enumerate() {
            self.seq[start_slot as usize + k] = nt;
        }
    }
}

fn nuc_char(nt: Nucleotide) -> char {
    match nt {
        Nucleotide::A => 'A',
        Nucleotide::C => 'C',
        Nucleotide::G => 'G',
        Nucleotide::U => 'U',
    }
}
