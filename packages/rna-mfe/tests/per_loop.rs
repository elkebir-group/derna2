//! Per-loop fixture parity test.
//!
//! Loads the curated fixture corpus
//! (`tests/data/mfe_fixtures.json`), filters to the
//! small-motif `dangling_5_prime: false` records (hairpin / bulge /
//! internal loop), decomposes each structure into its constituent
//! loops, sums per-loop energies via the per-loop evaluators in
//! [`rna_mfe::loops`], and asserts agreement with the fixture's
//! `energy_kcal_mol` at the 0.001 kcal/mol tolerance. Multi-loop and
//! long-form fixtures are exercised by the whole-structure integration
//! test in [`whole_structure.rs`] and excluded here.
//!
//! The decomposer is hand-rolled and intentionally minimal — the
//! production whole-structure decomposer lives in
//! [`rna_mfe::decompose`]. This test pair-tables the dot-bracket, classifies
//! each closed loop by counting immediately enclosed children, and add
//! a `Misc.terminal_au_37` term per outermost AU / GU / UA / UG branch
//! (the exterior-loop contribution at `dangles=0`: the terminal-AU
//! penalty on every outermost pair other than `CG` / `GC`).

use rna_mfe::loops::pair_from_nucleotides;
use rna_mfe::params::BasePair;
use rna_mfe::{
    bulge_energy, hairpin_energy, internal_energy, stack_energy, BulgeLoop, Energy, HairpinLoop,
    InternalLoop, Nucleotide, Params, StackPair,
};
use serde::Deserialize;

const TOLERANCE_KCAL_PER_MOL: f64 = 0.001;

#[test]
fn small_motif_fixtures_match_evaluator() {
    let params = Params::load_default().expect("load Turner 2004");
    let fixtures = load_small_motif_fixtures();
    assert!(
        fixtures.len() >= 26,
        "expected at least the 26 original small-motif dangles=0 fixtures, got {}",
        fixtures.len(),
    );
    for fixture in &fixtures {
        check_fixture(&params, fixture);
    }
}

fn check_fixture(params: &Params, fixture: &Fixture) {
    let seq = parse_seq(&fixture.seq);
    let pairs = pair_table(&fixture.structure);
    let computed = Walker {
        params,
        seq: &seq,
        pairs: &pairs,
    }
    .evaluate();
    let delta = (computed.to_kcal_mol() - fixture.energy_kcal_mol).abs();
    assert!(
        delta < TOLERANCE_KCAL_PER_MOL,
        "fixture mismatch: motif={} closure={} seq={} expected={:.4} computed={:.4} delta={:.4}",
        fixture.motif,
        fixture.closure,
        fixture.seq,
        fixture.energy_kcal_mol,
        computed.to_kcal_mol(),
        delta,
    );
}

/// Load the curated corpus and keep only the small-motif
/// `dangling_5_prime: false` records (excluding multi-loop / long-form).
fn load_small_motif_fixtures() -> Vec<Fixture> {
    const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/mfe_fixtures.json",);
    let raw = std::fs::read_to_string(PATH).expect("read mfe_fixtures.json");
    let parsed: FixtureFile = serde_json::from_str(&raw).expect("parse mfe_fixtures.json");
    parsed
        .fixtures
        .into_iter()
        .filter(|f| !f.dangling_5_prime && is_small_motif(&f.motif))
        .collect()
}

fn is_small_motif(motif: &str) -> bool {
    matches!(motif, "hairpin" | "bulge" | "internal_loop")
}

#[derive(Debug, Deserialize)]
struct FixtureFile {
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    closure: String,
    dangling_5_prime: bool,
    energy_kcal_mol: f64,
    motif: String,
    seq: String,
    structure: String,
}

fn parse_seq(seq: &str) -> Vec<Nucleotide> {
    seq.bytes()
        .map(|b| Nucleotide::from_byte(b).unwrap_or_else(|| panic!("non-ACGU byte: {b}")))
        .collect()
}

/// Build a position → partner-position table from a dot-bracket string.
/// Unbalanced brackets panic — the corpus is curated and never carries
/// malformed structures, so this stays a hard assertion.
fn pair_table(structure: &str) -> Vec<Option<usize>> {
    let mut stack = Vec::new();
    let mut pairs = vec![None; structure.len()];
    for (i, c) in structure.bytes().enumerate() {
        match c {
            b'(' => stack.push(i),
            b')' => {
                let j = stack.pop().expect("unbalanced dot-bracket");
                pairs[j] = Some(i);
                pairs[i] = Some(j);
            }
            b'.' => (),
            _ => panic!("unexpected dot-bracket char: {c}"),
        }
    }
    debug_assert!(stack.is_empty(), "unbalanced opening brackets");
    pairs
}

/// Walks the dot-bracket structure to compute the total energy as the
/// sum of all closed-loop energies plus the exterior-loop contribution
/// per outermost branch.
struct Walker<'a> {
    params: &'a Params,
    seq: &'a [Nucleotide],
    pairs: &'a [Option<usize>],
}

impl<'a> Walker<'a> {
    fn evaluate(&self) -> Energy {
        let mut energy = Energy::ZERO;
        for i in 0..self.pairs.len() {
            let Some(j) = self.pairs[i] else { continue };
            if j <= i {
                continue;
            }
            energy += self.loop_closed_by(i, j);
            if !self.is_enclosed(i, j) {
                energy += self.exterior_branch(i);
            }
        }
        energy
    }

    /// Energy of the loop closed by the pair `(i, j)`: 0 enclosed
    /// pairs → hairpin; 1 enclosed pair → stack / bulge / internal
    /// based on the unpaired counts. Multi-loop fixtures are filtered
    /// out before reaching this path.
    fn loop_closed_by(&self, i: usize, j: usize) -> Energy {
        let outer = self.pair_at(i);
        let children = self.enclosed_children(i, j);
        match children.as_slice() {
            [] => hairpin_energy(
                self.params,
                &HairpinLoop {
                    closing: outer,
                    loop_seq: &self.seq[i + 1..j],
                },
            ),
            &[(k, l)] => self.degree_two_loop(DegreeTwo {
                outer,
                outer_pair: (i, j),
                inner_pair: (k, l),
            }),
            _ => panic!(
                "multi-loop fixture leaked through small-motif filter: pair ({i}, {j}) has {} children",
                children.len(),
            ),
        }
    }

    fn degree_two_loop(&self, d: DegreeTwo) -> Energy {
        let (i, j) = d.outer_pair;
        let (k, l) = d.inner_pair;
        let inner = self.pair_at(k);
        let n1 = k - i - 1;
        let n2 = j - l - 1;
        if n1 == 0 && n2 == 0 {
            return stack_energy(
                self.params,
                &StackPair {
                    outer: d.outer,
                    inner,
                },
            );
        }
        if n1 == 0 || n2 == 0 {
            return bulge_energy(
                self.params,
                &BulgeLoop {
                    outer: d.outer,
                    inner,
                    size: n1.max(n2),
                },
            );
        }
        internal_energy(
            self.params,
            &InternalLoop {
                outer: d.outer,
                inner,
                left: &self.seq[i + 1..k],
                right: &self.seq[l + 1..j],
            },
        )
    }

    /// The exterior-stem contribution at `dangles=0` reduces to the
    /// terminal-AU penalty per non-`CG`/`GC` branch — no `mismatch_ext`
    /// lookup, no `dangle*`.
    fn exterior_branch(&self, i: usize) -> Energy {
        if self.pair_at(i).applies_terminal_au() {
            Energy(self.params.misc.terminal_au_37)
        } else {
            Energy::ZERO
        }
    }

    /// Immediately enclosed pairs `(k, l)` with `i < k < l < j`. Walks
    /// once in O(j - i) by jumping over each child's interval.
    fn enclosed_children(&self, i: usize, j: usize) -> Vec<(usize, usize)> {
        let mut children = Vec::new();
        let mut cursor = i + 1;
        while cursor < j {
            match self.pairs[cursor] {
                Some(end) if end > cursor && end < j => {
                    children.push((cursor, end));
                    cursor = end + 1;
                }
                _ => cursor += 1,
            }
        }
        children
    }

    fn pair_at(&self, i: usize) -> BasePair {
        let j = self.pairs[i].expect("position is paired");
        pair_from_nucleotides(self.seq[i], self.seq[j])
    }

    fn is_enclosed(&self, i: usize, j: usize) -> bool {
        self.pairs[..i]
            .iter()
            .any(|partner| matches!(partner, &Some(b) if b > j))
    }
}

/// Bundle the closing-pair / enclosed-pair indices needed by
/// [`Walker::degree_two_loop`] without overflowing the
/// too-many-arguments lint.
struct DegreeTwo {
    outer: BasePair,
    outer_pair: (usize, usize),
    inner_pair: (usize, usize),
}
