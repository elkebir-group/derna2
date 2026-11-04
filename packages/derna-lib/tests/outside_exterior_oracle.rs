//! Exactness oracle for the **right-exterior** outside `Cout`.
//!
//! `Cout[node]` is the best BRD of the suffix `[node, end)` folded as an
//! exterior parse: the backward mirror of the forward exterior table `C`.
//! Computed independently here (a second implementation, so a drift in either
//! the scan or the diagnostic's `Cout` is caught), it must satisfy the
//! inside/outside identity `Cout[start] == C[end].brd`: re-deriving the global
//! optimum from the 3′ side reproduces the forward optimum. This is the
//! correctness gate the outside must clear before it is trusted for ranking —
//! no outside estimate that fails it is admissible.
//!
//! The identity is asserted across the length ladder and a λ sweep, on the
//! exact (`b → ∞`) lattice where the forward optimum is the true global BRD.

use derna_codon_dp::codon_table::aa_to_index;
use derna_lib::dfa::{self, Dfa};
use derna_lib::scan::scan;
use derna_lib::tables::Tables;
use derna_lib::{CodonUsage, Params, ScanConfig};
use rna_mfe::loops::pair_from_nucleotides;
use rna_mfe::Nucleotide;

/// BRD of one unpaired exterior edge (CAI half only; energy is zero at
/// `dangles=0`).
fn edge_brd(cai: f64, lambda: f64) -> f64 {
    -(1.0 - lambda) * (cai * 100.0)
}

/// BRD of the terminal-AU penalty on a closing pair: `λ · terminal_au_37` for
/// AU / GU closures, zero otherwise.
fn terminal_au_brd(
    params: &Params,
    nuc_open: Nucleotide,
    nuc_close: Nucleotide,
    lambda: f64,
) -> f64 {
    if pair_from_nucleotides(nuc_open, nuc_close).applies_terminal_au() {
        lambda * f64::from(params.misc.terminal_au_37)
    } else {
        0.0
    }
}

/// The right-exterior completion `Cout[node]`, computed independently of the
/// diagnostic: `Cout[end] = 0`; each `Cout[o]` is the best of an unpaired
/// extension or absorbing a closed pair opening at `o`.
fn right_exterior(tables: &Tables, dfa: &Dfa, params: &Params, lambda: f64) -> Vec<f64> {
    let n = tables.c.len();
    let mut pairs_by_open: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for outer in 0..n {
        let node = dfa.node_of(outer);
        for (k, s) in tables.p_states(node) {
            let cost = s.brd(lambda) + terminal_au_brd(params, k.nuc_open(), k.nuc_close, lambda);
            pairs_by_open[dfa.ix(k.open)].push((outer, cost));
        }
    }
    let mut cout = vec![f64::INFINITY; n];
    cout[dfa.ix(dfa.end())] = 0.0;
    for pos in (0..=dfa.seq_len).rev() {
        for &o in dfa.nodes_at(pos) {
            let oi = dfa.ix(o);
            let mut best = cout[oi];
            for e in dfa.right(o) {
                best = best.min(edge_brd(e.cai, lambda) + cout[dfa.ix(e.to)]);
            }
            for &(outer, cost) in &pairs_by_open[oi] {
                let c = cout[outer];
                if c.is_finite() {
                    best = best.min(cost + c);
                }
            }
            cout[oi] = best;
        }
    }
    cout
}

fn assert_oracle(protein: &str, lambda: f64) {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let aa: Vec<usize> = protein
        .bytes()
        .map(|b| aa_to_index(b as char).unwrap())
        .collect();
    let graph = dfa::build(&aa, &usage);
    let tables = scan(&params, &graph, lambda, ScanConfig::default());

    let opt = tables.c[graph.ix(graph.end())]
        .expect("exterior reaches end")
        .brd(lambda);
    let cout = right_exterior(&tables, &graph, &params, lambda);
    let cout_start = cout[graph.ix(graph.start())];
    assert!(
        (cout_start - opt).abs() < 1e-6,
        "right-exterior oracle failed (n={}, λ={lambda}): Cout[start]={cout_start} vs C[end]={opt}",
        aa.len(),
    );
}

/// The backward right-exterior reproduces the forward optimum across a length
/// ladder and a λ sweep — the outside is exact for the exterior use.
#[test]
fn right_exterior_reproduces_forward_optimum() {
    // Short synthetic proteins spanning hairpin, stack, and multibranch folds;
    // kept small so the exact lattice runs fast in the test harness.
    let proteins = [
        "MRRWWFF",
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        "MAAAGGGCCCLLLKKKRRRWWWFFFYYYHHHDDDEEE",
    ];
    for protein in proteins {
        for &lambda in &[0.1, 0.5, 0.99, 0.99999] {
            assert_oracle(protein, lambda);
        }
    }
}
