//! Two weights scanned position by position in one loop
//! ([`scan::scan_lockstep`]) fill the same tables, cell for cell and back-pointer
//! for back-pointer, as two scans run one after the other. The lockstep scan is
//! an instrument for pricing what it costs to keep two weights' tables live
//! together, so it must be the same work in a different order and nothing else.

use derna_codon_dp::codon_table::aa_to_index;
use derna_lib::dfa::{self, Dfa};
use derna_lib::scan;
use derna_lib::tables::Tables;
use derna_lib::{CodonUsage, Params, ScanConfig};

const PROTEIN: &str = "MLWKCWMKYFVNQHRTGADESILPMKV";

/// Every cell of every table, as text, in a fixed order.
fn cells(graph: &Dfa, t: &Tables) -> Vec<String> {
    let mut out = Vec::new();
    for pos in 0..=graph.seq_len {
        for &node in graph.nodes_at(pos) {
            let mut at: Vec<String> = Vec::new();
            at.extend(t.p_states(node).iter().map(|c| format!("P {c:?}")));
            at.extend(t.multi_states(node).iter().map(|c| format!("Multi {c:?}")));
            at.extend(t.m_states(node).iter().map(|c| format!("M {c:?}")));
            at.extend(t.m2_states(node).iter().map(|c| format!("M2 {c:?}")));
            at.push(format!("C {:?}", t.c[graph.ix(node)]));
            at.sort();
            out.push(format!("{node:?}"));
            out.extend(at);
        }
    }
    out
}

#[test]
fn lockstep_tables_match_separate_scans() {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("codon usage");
    let aa: Vec<_> = PROTEIN
        .chars()
        .map(|c| aa_to_index(c).expect("residue"))
        .collect();
    let graph = dfa::build(&aa, &usage);
    let config = ScanConfig::default();
    let lambdas = [0.1, 0.6];
    let lock = scan::scan_lockstep(&params, &graph, lambdas, config);
    for (lambda, locked) in lambdas.into_iter().zip(&lock) {
        let alone = scan::scan(&params, &graph, lambda, config);
        let (a, b) = (cells(&graph, &alone), cells(&graph, locked));
        assert!(a.len() > 1000, "too few cells to be a check: {}", a.len());
        assert_eq!(a, b, "lockstep tables differ at lambda {lambda}");
    }
    // The two weights' optima differ, so a lockstep that served one weight's
    // tables to both would fail above.
    let end = graph.ix(graph.end());
    let (x, y) = (
        lock[0].c[end].expect("feasible"),
        lock[1].c[end].expect("feasible"),
    );
    assert!(x.mfe != y.mfe || x.cai != y.cai);
}
