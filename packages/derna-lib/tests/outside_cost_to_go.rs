//! The cost to go reproduces the forward optimum, matches an independent
//! derivation cell by cell, and is not the cheap credit in disguise.
//!
//! The first property is the inside/outside identity: re-deriving the global
//! optimum from the 3′ side must give back what the forward fold found from the
//! 5′ side. It is the correctness gate every table has to clear before anything
//! is pruned with it — a cost to go that fails it is not a bound at all.
//!
//! The identity is necessary and far from sufficient: it reads one cell. The
//! second property reads all of them, against the recorded hypergraph of
//! [`reference_cell_values`], which enumerates a multi-loop's 3′ flank per core
//! and per closing frontier rather than through the shared opening pool the
//! production pass mirrors.
//!
//! The last is the one that says the pass is worth having. The cheap credit
//! `optimum - cell_min` is *exactly* the true cost to go at cells the optimal
//! derivation passes through, so a test that only ever looked at those would
//! pass identically against a pass that returned the cheap credit everywhere.
//! The gap has to be shown where it exists.

use std::collections::HashMap;

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS, STOP, STOP_CODE};
use derna_lib::cost_to_go::{
    fold_with_cost_to_go, fold_with_cost_to_go_at_lambda, CellRef, Outside, SinglesTally,
};
use derna_lib::dfa::Dfa;
use derna_lib::outside::reference_cell_values;
use derna_lib::tables::Tables;
use derna_lib::{
    CodonUsage, Energy, Enforcement, PairClass, Params, RegionConstraints, ScanConfig,
};

/// One molecule: a protein between two given flanks, under a constraint table.
#[derive(Clone, Copy)]
struct Case<'a> {
    protein: &'a str,
    utr5: &'a str,
    utr3: &'a str,
    constraints: RegionConstraints,
}

impl<'a> Case<'a> {
    fn cds(protein: &'a str) -> Self {
        Self {
            protein,
            utr5: "",
            utr3: "",
            constraints: RegionConstraints::default(),
        }
    }
}

/// Coding sequences alone, spanning hairpin and stacked folds. **None of their
/// optima contains a multi-loop** — every `M`, `M2` and `Multi` cell in them
/// carries regret — so they exercise the multi-loop tables' values but never
/// put one on the answer. [`MULTILOOP`] is what does.
const CDS: [&str; 4] = [
    "MRRWWFF",
    "MYGKIIFVLM",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
    "MAAAGGGCCCLLLKKKRRRWWWFFFYYYHHHDDDEEE",
];

/// Two hairpins that cannot pair with each other in the 5′ flank, and a
/// closing helix joining the flank's first bases to the molecule's last, so the
/// optimum is a multi-loop. In the last three the coding sequence is the
/// multi-loop's 3′ unpaired flank — the stretch the scan closes through its
/// shared opening pool rather than one frontier at a time — and in the last the
/// closing strand ends mid-codon.
fn multiloop_cases(tight: &str, loose: &str) -> Vec<(String, String, String)> {
    let row = |p: &str, u5: &str, u3: &str| (p.to_string(), u5.to_string(), u3.to_string());
    vec![
        row("KK", "GAGCGGACUUCGGUCCA", "ACAGGGAAACCUGGCUC"),
        row("MA", "GGAGCGGACUUCGGUCCA", "ACAGGGAAACCUGGCUCC"),
        row("KKK", tight, "GCUC"),
        row("KKK", loose, "AGCUC"),
        row("KKKL", tight, "A"),
    ]
}

fn pool_flanks() -> (String, String) {
    let (a, b) = ("GGGACCUUCGGGUCCC", "CAUGCGGAAACGCAUG");
    (format!("GAGC{a}A{b}"), format!("GAGCA{a}AA{b}"))
}

/// The table a cell belongs to, for per-table reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Table {
    C,
    M,
    M2,
    P,
    Multi,
}

const TABLES: [Table; 5] = [Table::C, Table::M, Table::M2, Table::P, Table::Multi];

fn table(cell: CellRef) -> Table {
    match cell {
        CellRef::C(_) => Table::C,
        CellRef::M(..) => Table::M,
        CellRef::M2(..) => Table::M2,
        CellRef::P(..) => Table::P,
        CellRef::Multi(..) => Table::Multi,
    }
}

struct Folded {
    params: Params,
    graph: Dfa,
    outside: Outside,
    optimum: Energy,
    /// Every filled cell of every table, with its inside energy.
    cells: Vec<(CellRef, Energy)>,
}

fn full_sets(protein: &str) -> Vec<Vec<[u8; 3]>> {
    protein
        .chars()
        .map(|c| {
            // A designed stop is the stop family's slot, which has no letter.
            let a = match c {
                STOP_CODE => STOP,
                c => aa_to_index(c).expect("a standard residue"),
            };
            AMINO_ACIDS[a].codons.iter().map(|c| c.nuc).collect()
        })
        .collect()
}

fn fold(case: Case<'_>) -> Folded {
    fold_threaded(case, 1)
}

fn fold_threaded(case: Case<'_>, threads: usize) -> Folded {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let (graph, tables, outside) = fold_with_cost_to_go(
        &params,
        case.protein,
        &usage,
        &full_sets(case.protein),
        case.utr5,
        case.utr3,
        case.constraints,
        ScanConfig {
            threads,
            ..ScanConfig::default()
        },
    )
    .expect("a valid instance")
    .expect("the lattice admits a design");
    let optimum = tables.c[graph.ix(graph.end())]
        .expect("the exterior reaches the end")
        .mfe;
    let cells = filled_cells(&graph, &tables);
    Folded {
        params,
        graph,
        outside,
        optimum,
        cells,
    }
}

fn filled_cells(graph: &Dfa, tables: &Tables) -> Vec<(CellRef, Energy)> {
    let mut cells = Vec::new();
    for pos in 0..=graph.seq_len {
        for &o in graph.nodes_at(pos) {
            if let Some(st) = tables.c[graph.ix(o)] {
                cells.push((CellRef::C(o), st.mfe));
            }
            for (inner, st) in tables.m_states(o) {
                cells.push((CellRef::M(o, inner), st.mfe));
            }
            for (inner, st) in tables.m2_states(o) {
                cells.push((CellRef::M2(o, inner), st.mfe));
            }
            for (key, st) in tables.p_states(o) {
                cells.push((CellRef::P(o, key), st.mfe));
            }
            for (key, st) in tables.multi_states(o) {
                cells.push((CellRef::Multi(o, key), st.mfe));
            }
        }
    }
    cells
}

/// Every case below, with the constraint tables the constraint has to be mirrored
/// under: none, a forbidding table, and a penalizing one.
fn constrained(case: Case<'_>) -> [Case<'_>; 3] {
    let forbid = RegionConstraints::five_prime_isolation(Enforcement::Forbid)
        .with(PairClass::Utr5Utr3, Enforcement::Forbid);
    let penalize = RegionConstraints::symmetric_isolation(Enforcement::Penalize(Energy(50)));
    [
        case,
        Case {
            constraints: forbid,
            ..case
        },
        Case {
            constraints: penalize,
            ..case
        },
    ]
}

/// `Cout[start]` re-derives the global optimum from the 3′ side.
#[test]
fn the_exterior_cost_to_go_reproduces_the_forward_optimum() {
    for protein in CDS {
        let f = fold(Case::cds(protein));
        assert_eq!(
            f.outside.exterior_at(f.graph.start()),
            Some(f.optimum),
            "{protein}: the 3' derivation disagrees with the 5' one",
        );
    }
}

/// Assert every cell of `case` matches the recorded hypergraph, and return how
/// many cells of each table were compared and how the pass scored its loops.
fn check_against_reference(case: Case<'_>) -> (HashMap<Table, usize>, SinglesTally) {
    let f = fold(case);
    let reference = reference_cell_values(
        &f.params,
        &f.graph,
        1.0,
        case.constraints,
        ScanConfig::default(),
    );
    let ours: HashMap<CellRef, Energy> = f.cells.iter().copied().collect();
    let name = format!("{} utr=({},{})", case.protein, case.utr5, case.utr3);
    let mut per_table: HashMap<Table, usize> = HashMap::new();
    for &(cell, inside, out) in &reference {
        assert_eq!(
            ours.get(&cell).map(|e| f64::from(e.0)),
            Some(inside),
            "{name}: the reference fills {cell:?} differently from the scan",
        );
        let expected = out.is_finite().then_some(Energy(out as i32));
        assert_eq!(
            f.outside.at(cell),
            expected,
            "{name}: cost to go at {cell:?}"
        );
        if expected.is_some() {
            *per_table.entry(table(cell)).or_default() += 1;
        }
    }
    assert_eq!(
        reference.len(),
        f.cells.len(),
        "{name}: the scan fills cells the reference never derives",
    );
    (per_table, f.outside.singles_tally())
}

/// Sum two tallies.
fn plus(a: SinglesTally, b: SinglesTally) -> SinglesTally {
    SinglesTally {
        stack_bulge: a.stack_bulge + b.stack_bulge,
        split: a.split + b.split,
        corners: a.corners + b.corners,
        corners_generic: a.corners_generic + b.corners_generic,
        walked: a.walked + b.walked,
        corner_pairs: a.corner_pairs + b.corner_pairs,
        strands: a.strands + b.strands,
    }
}

/// Every loop family the single-branch consumers score was exercised: a family
/// no case reaches passes every equality check whatever it computes.
fn assert_every_family_scored(tally: SinglesTally) {
    assert!(
        tally.stack_bulge > 0,
        "no stack or bulge was scored: {tally:?}"
    );
    assert!(
        tally.split > 0,
        "no internal loop was scored from its halves: {tally:?}"
    );
    assert!(
        tally.corners > 0,
        "no internal loop was scored by corners: {tally:?}"
    );
}

/// Every cell's cost to go equals the recorded hypergraph's, and the two agree
/// on which cells have one at all — over coding sequences alone and over
/// flanked multi-loops, each under no constraint, a forbidding table and a
/// penalizing one.
#[test]
fn every_cell_matches_the_recorded_hypergraph() {
    let (tight, loose) = pool_flanks();
    let rows = multiloop_cases(&tight, &loose);
    let mut cases: Vec<Case<'_>> = CDS[..2].iter().map(|p| Case::cds(p)).collect();
    for (p, u5, u3) in &rows {
        cases.push(Case {
            protein: p,
            utr5: u5,
            utr3: u3,
            constraints: RegionConstraints::default(),
        });
    }
    let mut compared: HashMap<Table, usize> = HashMap::new();
    let mut tally = SinglesTally::default();
    for case in cases {
        for case in constrained(case) {
            let (per_table, t) = check_against_reference(case);
            for (t, n) in per_table {
                *compared.entry(t).or_default() += n;
            }
            tally = plus(tally, t);
        }
    }
    for t in TABLES {
        assert!(
            compared.get(&t).copied().unwrap_or(0) > 0,
            "no {t:?} cell with a finite cost to go was compared",
        );
    }
    assert_every_family_scored(tally);
}

/// The recorded hypergraph at a length where the multi-loop tables hold
/// thousands of cells. Separate from the test above because it is the slow one.
#[test]
fn every_cell_matches_the_recorded_hypergraph_at_thirty_three_residues() {
    let (counts, tally) = check_against_reference(Case::cds(CDS[2]));
    for t in TABLES {
        assert!(
            counts.get(&t).copied().unwrap_or(0) > 0,
            "no {t:?} cell with a finite cost to go was compared",
        );
    }
    assert_every_family_scored(tally);
}

/// At `λ < 1` every loop carries its strands' CAI, which the `λ = 1` checks
/// above never see: the pass's values match the recorded hypergraph's there
/// too, to rounding, over the flanked multi-loops under each constraint table.
#[test]
fn every_cell_matches_the_recorded_hypergraph_below_lambda_one() {
    const LAMBDA: f64 = 0.5;
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let (tight, loose) = pool_flanks();
    let rows = multiloop_cases(&tight, &loose);
    let mut tally = SinglesTally::default();
    let mut compared = 0;
    for (p, u5, u3) in &rows {
        let case = Case {
            protein: p,
            utr5: u5,
            utr3: u3,
            constraints: RegionConstraints::default(),
        };
        for case in constrained(case) {
            let (graph, _, outside) = fold_with_cost_to_go_at_lambda(
                &params,
                case.protein,
                &usage,
                LAMBDA,
                &full_sets(case.protein),
                case.utr5,
                case.utr3,
                case.constraints,
                ScanConfig::default(),
            )
            .expect("a valid instance")
            .expect("the lattice admits a design");
            let reference = reference_cell_values(
                &params,
                &graph,
                LAMBDA,
                case.constraints,
                ScanConfig::default(),
            );
            for &(cell, _, out) in &reference {
                let ours = outside.raw_at(cell);
                let agree = if out.is_finite() {
                    (ours - out).abs() <= 1e-6 * out.abs().max(1.0)
                } else {
                    !ours.is_finite()
                };
                assert!(
                    agree,
                    "{} utr=({},{}): cost to go at {cell:?} is {ours}, the reference {out}",
                    case.protein, case.utr5, case.utr3,
                );
                compared += usize::from(out.is_finite());
            }
            tally = plus(tally, outside.singles_tally());
        }
    }
    assert!(
        compared > 0,
        "no cell with a finite cost to go was compared"
    );
    assert_every_family_scored(tally);
}

/// Every internal loop the pass scores from its two halves equals its least
/// value over every pair of corners, at `λ = 1` and below.
///
/// The cell-by-cell checks above cannot see a split that is wrong only at a
/// consumer that never wins its cell, and such a split is easy to write:
/// splitting a loop whose strand stays inside one codon decouples that codon's
/// first and last nucleotides, mis-scoring 2×4 loops on these coding sequences
/// while every cell's value stays right. This check compares each consumer.
#[test]
fn every_split_loop_equals_its_corners() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let (tight, loose) = pool_flanks();
    let rows = multiloop_cases(&tight, &loose);
    let mut cases: Vec<Case<'_>> = ["MSLRSRLVVAG", "MLLSSRRGGAAP"]
        .into_iter()
        .map(Case::cds)
        .collect();
    for (p, u5, u3) in &rows {
        cases.push(Case {
            protein: p,
            utr5: u5,
            utr3: u3,
            constraints: RegionConstraints::default(),
        });
    }
    let mut split = 0;
    for case in cases {
        for lambda in [1.0, 0.5] {
            let (graph, tables, _) = fold_with_cost_to_go_at_lambda(
                &params,
                case.protein,
                &usage,
                lambda,
                &full_sets(case.protein),
                case.utr5,
                case.utr3,
                case.constraints,
                ScanConfig::default(),
            )
            .expect("a valid instance")
            .expect("the lattice admits a design");
            let checked =
                Outside::compute_validated(&params, &graph, &tables, ScanConfig::default());
            split += checked.singles_tally().split;
        }
    }
    assert!(split > 0, "no internal loop was scored from its halves");
}

/// For every cell, inside plus outside is a real molecule, so it can never beat
/// the optimum — and each table attains it somewhere.
#[test]
fn inside_plus_outside_is_a_molecule_and_the_optimum_is_attained() {
    let (tight, loose) = pool_flanks();
    let rows = multiloop_cases(&tight, &loose);
    let mut cases: Vec<Case<'_>> = CDS.iter().map(|p| Case::cds(p)).collect();
    for (p, u5, u3) in &rows {
        cases.push(Case {
            protein: p,
            utr5: u5,
            utr3: u3,
            constraints: RegionConstraints::default(),
        });
    }
    let mut attained: HashMap<Table, usize> = HashMap::new();
    for case in cases {
        let f = fold(case);
        for &(cell, inside) in &f.cells {
            let Some(outside) = f.outside.at(cell) else {
                continue;
            };
            let whole = inside + outside;
            assert!(
                whole >= f.optimum,
                "{}: a molecule through {cell:?} beats the optimum ({} < {})",
                case.protein,
                whole.0,
                f.optimum.0,
            );
            if whole == f.optimum {
                *attained.entry(table(cell)).or_default() += 1;
            }
        }
    }
    for t in TABLES {
        assert!(
            attained.get(&t).copied().unwrap_or(0) > 0,
            "no {t:?} cell lies on an optimal derivation of any case",
        );
    }
}

/// The pass is not the cheap credit in disguise.
///
/// `optimum - cell_min` equals the true cost to go exactly where the optimal
/// derivation runs, and underestimates it elsewhere by that cell's regret. If
/// no cell showed a gap, this pass would be worth nothing and every test above
/// would still pass, so the gap is asserted to exist and to point the one way
/// it can.
#[test]
fn the_cost_to_go_is_tighter_than_the_cheap_credit_and_never_looser() {
    let mut strictly_tighter: HashMap<Table, usize> = HashMap::new();
    for protein in CDS {
        let f = fold(Case::cds(protein));
        for &(cell, inside) in &f.cells {
            let Some(outside) = f.outside.at(cell) else {
                continue;
            };
            let cheap = Energy(f.optimum.0 - inside.0);
            assert!(
                outside >= cheap,
                "{protein}: the cost to go at {cell:?} is below the cheap \
                 credit ({} < {}), which would make the cheap credit unsound",
                outside.0,
                cheap.0,
            );
            if outside > cheap {
                *strictly_tighter.entry(table(cell)).or_default() += 1;
            }
        }
    }
    for t in TABLES {
        assert!(
            strictly_tighter.get(&t).copied().unwrap_or(0) > 0,
            "the cost to go never beat `optimum - cell_min` on a {t:?} cell",
        );
    }
}

/// The lever the cheap credit structurally cannot pull: cells no molecule
/// meeting the ceiling can pass through.
///
/// A cell's regret is `(cell_min + cost_to_go) - optimum`, and under the true
/// cost to go a cell is **empty** once its regret exceeds the window — no
/// completion brings anything in it within the ceiling. The cheap credit can
/// never empty a cell, since a cell's own best point is trivially within any
/// budget of itself, so this is a difference in kind rather than in degree and
/// it is what removes compose calls rather than merely stored points.
///
/// **Asserted on the closed-pair table, for a measured reason.** At this window
/// the multi-loop tables are wholly deletable on every case here, because no
/// optimum among them contains a multi-loop — true, and too easy to be the
/// evidence. The exterior is not deletable at all at seven residues. The
/// closed-pair table is the one that is non-trivially split on every case:
/// 134 of 216, 321 of 395, 6406 of 6855 and 6701 of 6980 cells deletable, in
/// length order. Asserting that some cell is deletable *and* some survives
/// keeps the test from passing against a window that deletes everything.
#[test]
fn a_tight_window_makes_some_cells_unreachable() {
    // 3.3 kcal/mol above the achievable minimum — the budget the window
    // ladders are anchored on.
    let window = Energy(330);
    for protein in CDS {
        let f = fold(Case::cds(protein));
        let (mut deletable, mut live) = (0usize, 0usize);
        for &(cell, inside) in &f.cells {
            if table(cell) != Table::P {
                continue;
            }
            match f.outside.at(cell) {
                Some(outside) if (inside + outside).0 - f.optimum.0 <= window.0 => live += 1,
                _ => deletable += 1,
            }
        }
        assert!(
            live > 0,
            "{protein}: the window deleted every closed pair, so the instance \
             says nothing about a real ceiling",
        );
        assert!(
            deletable > 0,
            "{protein}: no closed pair carries regret above {} dekacalories, so \
             the cell-deletion lever is empty here",
            window.0,
        );
    }
}

/// The pass spreads each position's single-branch consumers over workers; the
/// cost to go it returns must not depend on how many. Checked on every cell of
/// a coding sequence with thousands of multi-loop cells and of the flanked
/// cases whose optimum is a multi-loop closed through the shared pool.
#[test]
fn worker_count_does_not_change_the_cost_to_go() {
    let (tight, loose) = pool_flanks();
    let rows = multiloop_cases(&tight, &loose);
    let mut cases = vec![Case::cds(CDS[2])];
    for (p, u5, u3) in &rows {
        cases.push(Case {
            protein: p,
            utr5: u5,
            utr3: u3,
            constraints: RegionConstraints::default(),
        });
    }
    for case in cases {
        let serial = fold_threaded(case, 1);
        let parallel = fold_threaded(case, 8);
        assert_eq!(serial.cells.len(), parallel.cells.len());
        for &(cell, _) in &serial.cells {
            assert_eq!(
                serial.outside.at(cell),
                parallel.outside.at(cell),
                "{}: eight workers changed the cost to go at {cell:?}",
                case.protein,
            );
        }
    }
}

/// A designed stop's first node has two out-edges spelling `U`, the `UA` and
/// `UG` lanes, so a closed pair opening there is a cell per lane, and the pairs
/// each one encloses start past its own lane and must see it as their enclosing
/// pair. A 3′ flank
/// complementary to the coding sequence's end pairs the stop, and the optimum
/// runs through the `UG` lane.
#[test]
fn a_pair_opening_on_a_designed_stop_encloses_both_lanes() {
    let case = Case {
        protein: "MAKVLE*",
        utr5: "CCCUUUCCCUUU",
        utr3: "UUUCACUCUUU",
        constraints: RegionConstraints::default(),
    };
    let f = fold(case);
    // Otherwise the comparison below never reads a cell two lanes share.
    let shared = f
        .cells
        .iter()
        .filter(|(cell, _)| match cell {
            CellRef::P(_, key) => {
                let spelling = |e: &&derna_lib::dfa::Edge| e.nuc == key.nuc_open();
                f.graph.right(key.open).iter().filter(spelling).count() > 1
            }
            _ => false,
        })
        .count();
    assert!(shared > 0, "no closed pair opens on the stop's shared node");
    check_against_reference(case);
}
