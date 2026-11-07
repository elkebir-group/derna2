//! A stop-labelled lattice keeps the designed stop codon's identity to the end
//! of the molecule ([`dfa::build_stop_labelled`]), so one fold over it must give,
//! at the terminal of each stop, exactly the optimum of the lattice with that
//! stop codon given — and a design that really ends in that stop.
//!
//! The pinned reference spells the stop as the head of the 3′ flank, which is
//! how a given stop is folded everywhere else, so it shares no code with the
//! labelled lattice's stop unit.

use derna_codon_dp::codon_table::{aa_to_index, STOP};
use derna_lib::dfa::{self, Dfa, UtrChain, STOP_CODON_NUCS};
use derna_lib::scan;
use derna_lib::traceback;
use derna_lib::{CodonUsage, Params, ScanConfig, UtrProfile};
use rna_mfe::Nucleotide;

const AMINO: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";
const NUCS: [Nucleotide; 4] = [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U];
const LAMBDAS: [f64; 5] = [0.05, 0.3, 0.55, 0.8, 1.0];

/// A small deterministic generator, so every failure names a reproducible case.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

struct Case {
    protein: String,
    utr5: Vec<Nucleotide>,
    utr3: Vec<Nucleotide>,
}

fn case(seed: u64) -> Case {
    let mut r = Lcg(seed);
    let len = 3 + r.below(10);
    let protein = std::iter::once('M')
        .chain((1..len).map(|_| AMINO[r.below(AMINO.len())] as char))
        .collect();
    let utr = |r: &mut Lcg, max: usize| -> Vec<Nucleotide> {
        (0..r.below(max + 1)).map(|_| NUCS[r.below(4)]).collect()
    };
    let utr5 = utr(&mut r, 10);
    let utr3 = utr(&mut r, 14);
    Case {
        protein,
        utr5,
        utr3,
    }
}

/// A case built to put multi-loops past the stop: two GC hairpins in the 3′
/// flank, enclosed by a stem whose 5′ strand is the 5′ flank and whose 3′
/// strand closes the molecule, so a multi-loop opened before the stop has its
/// branches and its closing frontier on a stop lane.
fn structured_case(seed: u64) -> Case {
    use Nucleotide::{A, C, G, U};
    let mut r = Lcg(seed ^ 0x5eed);
    let len = 2 + r.below(5);
    let protein = std::iter::once('M')
        .chain((1..len).map(|_| AMINO[r.below(AMINO.len())] as char))
        .collect();
    let any =
        |r: &mut Lcg, n: usize| -> Vec<Nucleotide> { (0..n).map(|_| NUCS[r.below(4)]).collect() };
    let hairpin = |r: &mut Lcg| -> Vec<Nucleotide> {
        let stem: Vec<Nucleotide> = (0..3 + r.below(2)).map(|_| [G, C][r.below(2)]).collect();
        let pair = |n: &Nucleotide| match n {
            G => C,
            C => G,
            A => U,
            U => A,
        };
        let mut h = stem.clone();
        h.extend(any(r, 4));
        h.extend(stem.iter().rev().map(pair));
        h
    };
    let close: Vec<Nucleotide> = (0..4).map(|_| [G, C][r.below(2)]).collect();
    let n = r.below(3);
    let mut utr3 = any(&mut r, n);
    utr3.extend(hairpin(&mut r));
    utr3.extend({
        let n = 1 + r.below(2);
        any(&mut r, n)
    });
    utr3.extend(hairpin(&mut r));
    utr3.extend({
        let n = r.below(3);
        any(&mut r, n)
    });
    utr3.extend(close.iter().rev().map(|n| match n {
        G => C,
        _ => G,
    }));
    let n = r.below(3);
    let mut utr5 = any(&mut r, n);
    utr5.extend(close);
    Case {
        protein,
        utr5,
        utr3,
    }
}

fn residues(protein: &str) -> Vec<usize> {
    protein
        .chars()
        .map(|c| aa_to_index(c).expect("residue"))
        .collect()
}

fn spell(nts: &[Nucleotide]) -> String {
    nts.iter()
        .map(|n| match n {
            Nucleotide::A => 'A',
            Nucleotide::C => 'C',
            Nucleotide::G => 'G',
            Nucleotide::U => 'U',
        })
        .collect()
}

/// The lattice with stop `k` given: its nucleotides head the 3′ flank.
fn pinned(case: &Case, usage: &CodonUsage, k: usize) -> Dfa {
    let tail: Vec<Nucleotide> = STOP_CODON_NUCS[k]
        .iter()
        .chain(&case.utr3)
        .copied()
        .collect();
    dfa::build_with_utr(&residues(&case.protein), usage, &case.utr5, &tail)
}

fn labelled(case: &Case, usage: &CodonUsage) -> Dfa {
    let mut aa = residues(&case.protein);
    aa.push(STOP);
    dfa::build_stop_labelled(
        &aa,
        usage,
        UtrChain::Fixed(&case.utr5),
        UtrChain::Fixed(&case.utr3),
        UtrProfile::inert(),
    )
    .expect("the protein ends in a designed stop")
}

/// The free energy of `seq`'s minimum free energy structure, folded alone.
fn free_mfe(params: &Params, seq: &str) -> i32 {
    let nts: Vec<Nucleotide> = seq
        .bytes()
        .map(|b| NUCS["ACGU".bytes().position(|c| c == b).expect("ACGU")])
        .collect();
    let graph = dfa::build_fixed_mrna(&nts);
    let t = scan::scan(params, &graph, 1.0, ScanConfig::default());
    t.c[graph.ix(graph.end())]
        .expect("a fixed molecule folds")
        .mfe
        .0
}

/// Fold every case's labelled lattice at every weight and check each terminal
/// against its pinned-stop fold; returns how many (case, weight) pairs had
/// stops that differ, so a caller can require the cases to tell them apart.
fn check(cases: impl Iterator<Item = (u64, Case)>) -> usize {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("codon usage");
    let config = ScanConfig::default();
    let mut differing_stops = 0;
    for (seed, case) in cases {
        let graph = labelled(&case, &usage);
        assert_eq!(graph.ends().len(), STOP_CODON_NUCS.len(), "seed {seed}");
        for lambda in LAMBDAS {
            let t = scan::scan(&params, &graph, lambda, config);
            let mut optima = Vec::new();
            for (k, &end) in graph.ends().iter().enumerate() {
                let tag = format!("seed {seed} {} λ={lambda} stop {k}", case.protein);
                let reference = pinned(&case, &usage, k);
                let rt = scan::scan(&params, &reference, lambda, config);
                let want = rt.c[reference.ix(reference.end())].expect("pinned is feasible");
                let got = t.c[graph.ix(end)].unwrap_or_else(|| panic!("{tag}: no parse"));
                // At λ = 1 adaptation is not scored, so designs that tie on
                // energy may differ in it.
                let key =
                    |s: derna_lib::state::State| (s.mfe, (lambda < 1.0).then_some(s.cai.to_bits()));
                assert_eq!(
                    key(got),
                    key(want),
                    "{tag}: labelled terminal differs from the pinned fold"
                );
                let design = traceback::recover_at(&params, &graph, &t, end);
                let stop_at = case.utr5.len() + 3 * case.protein.len();
                assert_eq!(
                    &design.sequence[stop_at..stop_at + 3],
                    spell(&STOP_CODON_NUCS[k]),
                    "{tag}: the design does not end in its terminal's stop"
                );
                assert_eq!(
                    free_mfe(&params, &design.sequence),
                    got.mfe.0,
                    "{tag}: the design's energy does not re-evaluate"
                );
                optima.push(key(got));
            }
            optima.dedup();
            differing_stops += usize::from(optima.len() > 1);
        }
    }
    differing_stops
}

#[test]
fn each_terminal_is_the_pinned_stop_optimum() {
    let differing = check((0..40).map(|seed| (seed, case(seed))));
    // The terminals must not all agree, or a lattice that merged its stops
    // would pass this test.
    assert!(
        differing > 20,
        "only {differing} cases tell the stops apart"
    );
}

#[test]
fn multi_loops_past_the_stop_keep_their_stop() {
    let differing = check((0..24).map(|seed| (seed, structured_case(seed))));
    assert!(
        differing > 10,
        "only {differing} cases tell the stops apart"
    );
}

/// The cost to go over a stop-labelled lattice starts at every terminal, so the
/// start node's completion is the best design over every stop: the optimum of
/// the merged free-stop lattice, and the least of the terminals' optima. A pass
/// seeded at one terminal only would read the first stop's optimum instead.
#[test]
fn the_cost_to_go_completes_to_every_stop() {
    use derna_lib::cost_to_go::{CellRef, Outside};
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("codon usage");
    let config = ScanConfig::default();
    let mut below_first = 0;
    let cases = (0..12)
        .map(|seed| (seed, structured_case(seed)))
        .chain((0..12).map(|seed| (100 + seed, case(seed))));
    for (seed, case) in cases {
        let graph = labelled(&case, &usage);
        let mut aa = residues(&case.protein);
        aa.push(STOP);
        let merged = dfa::build_with_utr_profile(
            &aa,
            &usage,
            UtrChain::Fixed(&case.utr5),
            UtrChain::Fixed(&case.utr3),
            UtrProfile::inert(),
        );
        for lambda in [0.3, 1.0] {
            let t = scan::scan(&params, &graph, lambda, config);
            let ends: Vec<f64> = graph
                .ends()
                .iter()
                .map(|&e| t.c[graph.ix(e)].expect("feasible").brd(lambda))
                .collect();
            let best = ends.iter().copied().fold(f64::INFINITY, f64::min);
            let out = Outside::compute_validated(&params, &graph, &t, config);
            let start = out.raw_at(CellRef::C(graph.start()));
            assert!(
                (start - best).abs() < 1e-6,
                "seed {seed} λ={lambda}: {start} vs {best}"
            );
            let mt = scan::scan(&params, &merged, lambda, config);
            let free = mt.c[merged.ix(merged.end())].expect("feasible").brd(lambda);
            assert!(
                (free - best).abs() < 1e-6,
                "seed {seed} λ={lambda}: free {free} vs {best}"
            );
            below_first += usize::from(best < ends[0] - 1e-6);
        }
    }
    assert!(
        below_first > 5,
        "only {below_first} cases where a later stop is best"
    );
}

/// One backward sweep over several weights gives each weight its own outside
/// pass on a stop-labelled lattice, bit for bit, so the shared sweep keeps
/// stops apart wherever the single pass does.
#[test]
fn the_fused_cost_to_go_keeps_stops_apart() {
    use derna_lib::outside_multi::Instance;
    let usage = CodonUsage::load_default().expect("codon usage");
    let cases = (0..8)
        .map(|seed| (seed, structured_case(seed)))
        .chain((0..8).map(|seed| (100 + seed, case(seed))));
    for (seed, case) in cases {
        let instance = Instance::scan_lattice(labelled(&case, &usage), &[0.05, 0.3, 1.0]);
        let separate = instance.separate(1);
        let (fused, _) = instance.fused(1);
        for (lane, (one, all)) in separate.iter().zip(&fused).enumerate() {
            assert!(one.finite_cells() > 0, "seed {seed}: lane {lane} is empty");
            if let Some(diff) = one.first_difference(all) {
                panic!("seed {seed} lane {lane}: {diff}");
            }
        }
    }
}

/// A walk over the stop-labelled lattice read at each stop returns that
/// stop's front: point for point the front of the walk with the stop written
/// into the 3′ flank, each design ending in its own stop.
#[test]
fn each_stop_walk_is_the_pinned_stop_walk() {
    use derna_lib::{
        design_front_walk_per_stop, design_front_walk_with_utr, PassGrid, RegionConstraints,
        UtrSpec, WalkOptions,
    };
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("codon usage");
    let fixed = |s: &str| UtrSpec::Fixed(s.to_string());
    let cases = [
        ("MKFL", "GCAUUG", "AUCGA"),
        ("MAWQ", "GGAC", "GGGCGCAAGCGCCC"),
        ("MRSE", "", "CCUUCGGGAAAGC"),
    ];
    let mut distinct = 0;
    let gridded = |n: usize| WalkOptions {
        pass_grid: Some(PassGrid::for_residues(n)),
        ..WalkOptions::for_residues(n)
    };
    for ((protein, utr5, utr3), grid) in cases.iter().flat_map(|&c| [(c, false), (c, true)]) {
        let options = || {
            if grid {
                gridded(protein.len())
            } else {
                WalkOptions::default()
            }
        };
        let mut seen: Vec<Vec<(i64, f64)>> = Vec::new();
        let per_stop = design_front_walk_per_stop(
            &params,
            &format!("{protein}*"),
            &usage,
            &fixed(utr5),
            &fixed(utr3),
            RegionConstraints::default(),
            1e-4,
            0.999,
            options(),
            1,
        )
        .expect("the per-stop walk answers");
        assert_eq!(per_stop.len(), 3);
        for (k, got) in per_stop.iter().enumerate() {
            let stop: String = STOP_CODON_NUCS[k]
                .iter()
                .map(|&n| ['A', 'C', 'G', 'U'][n as usize])
                .collect();
            let want = design_front_walk_with_utr(
                &params,
                protein,
                &usage,
                &fixed(utr5),
                &fixed(&format!("{stop}{utr3}")),
                RegionConstraints::default(),
                1e-4,
                0.999,
                options(),
                1,
            )
            .expect("the pinned walk answers");
            let got = got.as_ref().expect("every stop has designs");
            let objectives = |f: &derna_lib::WalkedFront| -> Vec<(i64, f64)> {
                let mut v: Vec<(i64, f64)> = f
                    .front
                    .iter()
                    .map(|p| ((p.mfe_kcal_mol * 100.0).round() as i64, p.cai_log))
                    .collect();
                v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
                v
            };
            let (g, w) = (objectives(got), objectives(&want));
            assert_eq!(g.len(), w.len(), "{protein} stop {stop}: {g:?} vs {w:?}");
            for (a, b) in g.iter().zip(&w) {
                assert!(
                    a.0 == b.0 && (a.1 - b.1).abs() < 1e-6,
                    "{protein} stop {stop}: {a:?} vs {b:?}"
                );
            }
            let at = utr5.len() + 3 * protein.len();
            for p in &got.front {
                assert_eq!(&p.sequence[at..at + 3], stop, "{protein}: {}", p.sequence);
            }
            seen.push(g);
        }
        distinct += usize::from(seen.iter().any(|f| *f != seen[0]));
    }
    assert!(distinct > 0, "no case where the stops' fronts differ");
}
