//! The threaded scan fills every cell with the serial scan's value.
//!
//! The scan fans two phases out over workers and folds their buffered
//! candidates back with the min-BRD merge. That makes every cell's **BRD**
//! independent of the order candidates arrive in. On an exact BRD tie the merge
//! keeps the first arrival, so the tied candidates' `(mfe, cai)` split is
//! order-dependent. At `λ = 1` the BRD is the energy alone, and ties between
//! designs of equal energy and different adaptation are common: there the
//! stored CAI varies run to run, and only the energy is a value of the cell.
//! At `0 < λ < 1` a tie between different splits needs an exact float
//! coincidence, and this lattice has none.
//!
//! This checks the property every consumer of the scan's values rests on (the
//! exact credit reads energies at `λ = 1`; a box cut reads BRDs at its weight),
//! cell by cell and bit for bit at three worker counts. It is the property a
//! change to how the fan-out deals its work must keep.

use derna_codon_dp::codon_table::aa_to_index;
use derna_lib::dfa::{self, Dfa, Node};
use derna_lib::scan::scan;
use derna_lib::state::PairKey;
use derna_lib::tables::Tables;
use derna_lib::{CodonUsage, Params, ScanConfig};

/// Long enough that the internal-loop pull and the push both fan out at the
/// later positions (the pull's outer-cell count there is several times its
/// 256-cell serial threshold), short enough for a debug-profile test.
const PROTEIN: &str = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTN";

/// Every cell's `(mfe, cai bits)`, keyed by table and coordinates, sorted.
type Cells = Vec<(u8, Node, Node, Option<PairKey>, i32, u64)>;

fn cells(graph: &Dfa, tables: &Tables) -> Cells {
    let mut out = Cells::new();
    for pos in 0..=graph.seq_len {
        for &o in graph.nodes_at(pos) {
            if let Some(st) = tables.c[graph.ix(o)] {
                out.push((0, o, o, None, st.mfe.0, st.cai.to_bits()));
            }
            for (inner, st) in tables.m_states(o) {
                out.push((1, o, inner, None, st.mfe.0, st.cai.to_bits()));
            }
            for (inner, st) in tables.m2_states(o) {
                out.push((2, o, inner, None, st.mfe.0, st.cai.to_bits()));
            }
            for (key, st) in tables.p_states(o) {
                out.push((3, o, o, Some(key), st.mfe.0, st.cai.to_bits()));
            }
            for (key, st) in tables.multi_states(o) {
                out.push((4, o, o, Some(key), st.mfe.0, st.cai.to_bits()));
            }
        }
    }
    out.sort_by_key(|&(t, o, i, k, _, _)| {
        (
            t,
            graph.ix(o),
            graph.ix(i),
            k.map(|k| {
                (
                    graph.ix(k.open),
                    graph.ix(k.close),
                    k.nuc_open() as u8,
                    k.nuc_close as u8,
                )
            }),
        )
    });
    out
}

#[test]
fn every_cell_matches_the_serial_scan_at_any_worker_count() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let aa: Vec<usize> = PROTEIN.chars().map(|c| aa_to_index(c).unwrap()).collect();
    let graph = dfa::build(&aa, &usage);
    for lambda in [0.5, 1.0] {
        // At `λ = 1` only the energy is the cell's value; see the module note.
        let value = |cells: Cells| -> Cells {
            if lambda == 1.0 {
                cells
                    .into_iter()
                    .map(|(t, o, i, k, e, _)| (t, o, i, k, e, 0))
                    .collect()
            } else {
                cells
            }
        };
        let fold = |threads| {
            let config = ScanConfig {
                threads,
                ..ScanConfig::default()
            };
            value(cells(&graph, &scan(&params, &graph, lambda, config)))
        };
        let serial = fold(1);
        assert!(serial.len() > 1000, "a lattice with something to fan out");
        for threads in [2, 8] {
            let threaded = fold(threads);
            assert_eq!(
                serial.len(),
                threaded.len(),
                "λ = {lambda}: cell sets differ"
            );
            let differ = serial.iter().zip(&threaded).filter(|(a, b)| a != b).count();
            assert_eq!(
                differ,
                0,
                "λ = {lambda}, {threads} threads: {differ} of {} cells differ",
                serial.len()
            );
        }
    }
}
