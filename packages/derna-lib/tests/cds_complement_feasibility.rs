//! A complement derived from **given** flanks and enforced over the **coding
//! sequence**, checked against exhaustive enumeration.
//!
//! The flank-targeted arrangement — one flank given, the other designed — has
//! its own oracle. This is the other arrangement: **both** flanks given and the
//! coding sequence designed, so the forbidden set derives from the flanks and is
//! enforced over the codons. The construction is the same one; what changes is
//! which region it is enforced over, and the coding region is the one the
//! protein has already constrained.
//!
//! Everything a census of that arrangement reports rests on one predicate —
//! whether the product automaton admits any encoding — evaluated tens of
//! thousands of times. So the predicate is checked against brute force before
//! it is trusted at length, and checked on the quantity a census actually
//! reports rather than only on the boolean:
//!
//! 1. **The path count is the encoding count.** A start-to-terminal path of the
//!    unconstrained product is one synonymous assignment, so the counts the
//!    later checks compare are counts of the same thing.
//! 2. **Feasibility and the surviving-encoding count**, against enumeration, on
//!    two proteins × three derivation arms × three duplex lengths. A single
//!    agreement is consistent with an oracle sharing a bug with what it checks,
//!    which is why neither the protein nor the arm is held fixed.
//! 3. **The codon census** ([`ProductDfa::surviving_codons`]), against the same
//!    enumeration. This is the column that separates *the constraint bound and
//!    the protein had room* from *the constraint forbade nothing that could
//!    occur*, and a census that is merely plausible makes every feasible cell
//!    unreadable.
//! 4. **The vacuity floor.** A window narrower than the duplex length can hold
//!    no occurrence entirely, so it forbids nothing and is feasible for no
//!    reason. Asserted from both sides — nothing removed below the floor, and
//!    something removed above it, or the check passes on an inert instrument.
//! 5. **Anti-monotonicity in the window width**, which a bisection for the
//!    widest feasible width depends on, and **in the protein prefix**, which
//!    lets a length ladder be read as a collapse rather than as noise.
//!
//! The forbidden set is derived here from the **pairing relation** rather than
//! from the shipped pattern derivation: a `k`-mer is forbidden when it can stack
//! against some window of the source flank, base by base. So a disagreement
//! localizes to the derivation, the degenerate expansion, the automaton, the
//! window reading or the product — not to a shared helper. The window predicate
//! and the IUPAC semantics are likewise re-stated rather than imported.
//!
//! **What the checks deliberately do not assert.** Not *zero cross-region
//! pairing*: a derived complement forbids only **ungapped** duplexes, so a
//! bulged helix survives it by construction and an assertion to the contrary
//! would be a defect in this file rather than in the code. Not *every
//! occurrence*: one straddling a window edge is permitted, which is the window's
//! stated reading and is asserted positively in check 4.

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::ac::Ac;
use derna_lib::complement::pairing_motifs;
use derna_lib::dfa::{self, Dfa, UtrChain};
use derna_lib::product::ProductDfa;
use derna_lib::{CodonUsage, Complement, MotifWindow};
use rna_mfe::Nucleotide;

/// The first 25 nucleotides of the human β-globin 5′ UTR.
const HBB5: &str = "ACAUUUGCUUCUGACACAACUGUGU";
/// The first 25 nucleotides of the human β-globin 3′ UTR.
const HBB3: &str = "GCUCGCUUUCUUGCUGUCCAAUUUC";

/// Nine residues over 6912 synonymous assignments.
const PEP_A: &str = "MLLLAADWM";
/// Nine residues over 55296 — a different codon mix, so an agreement on one is
/// not an agreement on one lattice shape.
const PEP_B: &str = "MFVFLVLLP";

/// Which given flank(s) the forbidden set derives from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Five,
    Three,
    Both,
}

impl Arm {
    /// The flanks this arm derives from, in 5′-then-3′ order.
    fn sources<'a>(self, utr5: &'a [Nucleotide], utr3: &'a [Nucleotide]) -> Vec<&'a [Nucleotide]> {
        match self {
            Self::Five => vec![utr5],
            Self::Three => vec![utr3],
            Self::Both => vec![utr5, utr3],
        }
    }
}

// ─── check 1: a path is an encoding ──────────────────────────────────────────

#[test]
fn an_unconstrained_path_is_one_synonymous_assignment() {
    for protein in [PEP_A, PEP_B] {
        let case = Case::new(protein, HBB5, HBB3);
        let ac = Ac::build::<String>(&[]).expect("the empty set builds");
        let product = ProductDfa::build_scoped(&case.graph, &ac, MotifWindow::all());
        let expected = case.assignments();
        assert_eq!(
            count_paths(&product),
            expected,
            "{protein}: the unconstrained product should carry one path per \
             synonymous assignment, so a constrained path count is a count of \
             surviving encodings"
        );
        assert_eq!(case.enumerate().len() as u128, expected, "{protein}");
    }
}

// ─── check 2: feasibility and the surviving count ────────────────────────────

#[test]
fn the_automaton_agrees_with_exhaustive_enumeration() {
    let mut bound_somewhere = false;
    for protein in [PEP_A, PEP_B] {
        let case = Case::new(protein, HBB5, HBB3);
        let enumeration = case.enumerate();
        for arm in [Arm::Five, Arm::Three, Arm::Both] {
            for k in [4usize, 5, 6] {
                for complement in [Complement::Wobble, Complement::Strict] {
                    let cell = Cell::new(&case, arm, k, complement, case.cds_len());
                    let product = cell.product(&case);
                    let surviving = enumeration.survivors(&cell);
                    assert_eq!(
                        product.is_feasible(),
                        !surviving.is_empty(),
                        "{protein} {arm:?} k={k} {:?}: the automaton and brute force \
                         disagree on whether any encoding survives",
                        complement
                    );
                    assert_eq!(
                        count_paths(&product),
                        surviving.len() as u128,
                        "{protein} {arm:?} k={k} {:?}: the automaton and brute force \
                         disagree on how many encodings survive",
                        complement
                    );
                    bound_somewhere |= !surviving.is_empty() && surviving.len() < enumeration.len();
                }
            }
        }
    }
    assert!(
        bound_somewhere,
        "no swept cell both admitted an encoding and removed one, so the \
         agreement above is an agreement about an inert constraint"
    );
}

// ─── check 3: the codon census ───────────────────────────────────────────────

#[test]
fn the_codon_census_agrees_with_exhaustive_enumeration() {
    let mut saw_partial_removal = false;
    for protein in [PEP_A, PEP_B] {
        let case = Case::new(protein, HBB5, HBB3);
        let enumeration = case.enumerate();
        for arm in [Arm::Five, Arm::Three, Arm::Both] {
            for k in [4usize, 5, 6] {
                let cell = Cell::new(&case, arm, k, Complement::Wobble, case.cds_len());
                let surviving = enumeration.survivors(&cell);
                let expected = enumeration.codon_census(&surviving);
                assert_eq!(
                    cell.product(&case).surviving_codons(),
                    expected,
                    "{protein} {arm:?} k={k}: the reachability census and brute \
                     force disagree on which codons survive"
                );
                let removed = removed_count(&expected);
                saw_partial_removal |= removed > 0 && !surviving.is_empty();
            }
        }
    }
    assert!(
        saw_partial_removal,
        "every swept cell either removed nothing or removed everything, so the \
         census was never checked on the case it exists to report"
    );
}

// ─── check 4: the vacuity floor ──────────────────────────────────────────────

#[test]
fn a_window_narrower_than_k_forbids_nothing() {
    let case = Case::new(PEP_B, HBB5, HBB3);
    let enumeration = case.enumerate();
    for (arm, k) in [(Arm::Three, 4usize), (Arm::Both, 5), (Arm::Both, 6)] {
        for width in 0..k as u32 {
            let cell = Cell::new(&case, arm, k, Complement::Wobble, width);
            assert!(
                cell.product(&case).is_feasible(),
                "{arm:?} k={k}: width {width} is below the duplex length, so no \
                 occurrence can lie entirely inside it and the run must be feasible"
            );
            assert_eq!(
                removed_count(&cell.product(&case).surviving_codons()),
                0,
                "{arm:?} k={k}: width {width} is below the duplex length, so the \
                 set forbids nothing and must remove no codon"
            );
            assert_eq!(
                enumeration.survivors(&cell).len(),
                enumeration.len(),
                "{arm:?} k={k}: at width {width} brute force should agree that \
                 every encoding survives"
            );
        }
    }
    // Without this the assertions above pass on an instrument that never fires.
    // The control has to be *feasible* and still remove something: an infeasible
    // cell also removes codons, but it removes all of them, which is a different
    // statement from "this set binds and the protein had room".
    let control = Cell::new(&case, Arm::Three, 4, Complement::Wobble, case.cds_len());
    let census = control.product(&case).surviving_codons();
    assert!(
        control.product(&case).is_feasible() && removed_count(&census) > 0,
        "the control cell is not a control: a set that is feasible and still \
         removes a codon at full width is what makes the widths below the floor \
         a floor rather than a property of an inert set"
    );
}

// ─── check 5: the two anti-monotonicities ────────────────────────────────────

#[test]
fn feasibility_is_anti_monotone_in_the_window_width() {
    let case = Case::new(PEP_B, HBB5, HBB3);
    for arm in [Arm::Five, Arm::Three, Arm::Both] {
        for k in [3usize, 4, 5, 6] {
            let mut previously_infeasible = false;
            for width in 0..=case.cds_len() {
                let cell = Cell::new(&case, arm, k, Complement::Wobble, width);
                let feasible = cell.product(&case).is_feasible();
                assert!(
                    !(previously_infeasible && feasible),
                    "{arm:?} k={k}: feasibility returned at width {width} after a \
                     narrower width was infeasible, so widening the window did \
                     not forbid a superset and a bisection for the widest \
                     feasible width would be unsound"
                );
                previously_infeasible |= !feasible;
            }
        }
    }
}

#[test]
fn feasibility_is_anti_monotone_in_the_protein_prefix() {
    let long = "MFVFLVLLPLVSSQ";
    for k in [4usize, 5, 6] {
        let mut previously_infeasible = false;
        for len in 1..=long.len() {
            let case = Case::new(&long[..len], HBB5, HBB3);
            let cell = Cell::new(&case, Arm::Both, k, Complement::Wobble, case.cds_len());
            let feasible = cell.product(&case).is_feasible();
            assert!(
                !(previously_infeasible && feasible),
                "k={k}: feasibility returned at {len} residues after a shorter \
                 prefix was infeasible, so a length ladder cannot be read as a \
                 collapse and the sparse confirmation points above it are unsound"
            );
            previously_infeasible |= !feasible;
        }
    }
}

// ─── the enumerated molecule ─────────────────────────────────────────────────

/// One protein between one given flank pair.
///
/// Enumeration is [`Case::enumerate`]d on demand rather than at construction:
/// the assignment count is the product of the synonym counts, so it passes ten
/// million within a couple of dozen residues, and the checks that only need the
/// automaton must not pay for it.
struct Case {
    residues: Vec<usize>,
    utr5: Vec<Nucleotide>,
    utr3: Vec<Nucleotide>,
    graph: Dfa,
}

impl Case {
    fn new(protein: &str, utr5: &str, utr3: &str) -> Self {
        let residues: Vec<usize> = protein
            .chars()
            .map(|c| aa_to_index(c).expect("a fixture residue"))
            .collect();
        let utr5 = nucleotides(utr5);
        let utr3 = nucleotides(utr3);
        let usage = CodonUsage::load_default().expect("the shipped codon usage");
        let graph = dfa::build_with_utr_chain(
            &residues,
            &usage,
            UtrChain::Fixed(&utr5),
            UtrChain::Fixed(&utr3),
        );
        Self {
            residues,
            utr5,
            utr3,
            graph,
        }
    }

    fn cds_len(&self) -> u32 {
        3 * self.residues.len() as u32
    }

    /// How many synonymous assignments the protein has.
    fn assignments(&self) -> u128 {
        self.residues
            .iter()
            .map(|&aa| AMINO_ACIDS[aa].codons.len() as u128)
            .product()
    }

    /// Every synonymous assignment, materialized.
    fn enumerate(&self) -> Enumeration {
        Enumeration::of(self)
    }
}

/// Every synonymous assignment of one case, as whole molecules and as the
/// per-residue codon indices that produced them.
struct Enumeration {
    molecules: Vec<Vec<Nucleotide>>,
    choices: Vec<Vec<usize>>,
    /// Synonym counts per residue, so a census can be shaped without the case.
    widths: Vec<usize>,
}

impl Enumeration {
    fn of(case: &Case) -> Self {
        let mut molecules = vec![case.utr5.clone()];
        let mut choices = vec![Vec::new()];
        for &aa in &case.residues {
            let codons = AMINO_ACIDS[aa].codons;
            let mut next_molecules = Vec::with_capacity(molecules.len() * codons.len());
            let mut next_choices = Vec::with_capacity(choices.len() * codons.len());
            for (molecule, choice) in molecules.iter().zip(&choices) {
                for (index, codon) in codons.iter().enumerate() {
                    let mut extended = molecule.clone();
                    extended.extend(codon.nuc.iter().map(|&b| nucleotide(b)));
                    next_molecules.push(extended);
                    let mut taken = choice.clone();
                    taken.push(index);
                    next_choices.push(taken);
                }
            }
            molecules = next_molecules;
            choices = next_choices;
        }
        for molecule in &mut molecules {
            molecule.extend_from_slice(&case.utr3);
        }
        Self {
            molecules,
            choices,
            widths: case
                .residues
                .iter()
                .map(|&aa| AMINO_ACIDS[aa].codons.len())
                .collect(),
        }
    }

    fn len(&self) -> usize {
        self.molecules.len()
    }

    /// Which assignments carry no forbidden occurrence entirely inside the
    /// window.
    fn survivors(&self, cell: &Cell) -> Vec<usize> {
        (0..self.molecules.len())
            .filter(|&i| !cell.forbids(&self.molecules[i]))
            .collect()
    }

    /// Which synonymous codons some surviving assignment uses, per residue.
    fn codon_census(&self, surviving: &[usize]) -> Vec<Vec<bool>> {
        let mut census: Vec<Vec<bool>> = self
            .widths
            .iter()
            .map(|&width| vec![false; width])
            .collect();
        for &i in surviving {
            for (residue, &codon) in self.choices[i].iter().enumerate() {
                census[residue][codon] = true;
            }
        }
        census
    }
}

// ─── one measured configuration ──────────────────────────────────────────────

/// One `(arm, k, complement, window width)` configuration over a case: the
/// automaton built from the shipped derivation, and — separately — the
/// forbidden `k`-mers implied by the pairing relation.
struct Cell {
    /// The automaton the product threads, owned here so a product can borrow it
    /// for as long as a check needs.
    ac: Ac,
    /// Indexed by the `k`-mer's base-4 code; true when it can stack against some
    /// window of a source flank.
    forbidden: Vec<bool>,
    k: usize,
    window: MotifWindow,
}

impl Cell {
    fn new(case: &Case, arm: Arm, k: usize, complement: Complement, width: u32) -> Self {
        let sources = arm.sources(&case.utr5, &case.utr3);
        let mut motifs: Vec<String> = sources
            .iter()
            .flat_map(|flank| pairing_motifs(flank, k, complement))
            .collect();
        motifs.sort();
        motifs.dedup();
        let lo = case.graph.cds_start();
        Self {
            ac: Ac::build(&motifs).expect("the derived set builds"),
            forbidden: forbidden_kmers(&sources, k, complement),
            k,
            window: MotifWindow::new(lo, lo + width.min(case.cds_len())),
        }
    }

    /// The product over this case's codon graph, enforced over this window.
    fn product<'a>(&'a self, case: &'a Case) -> ProductDfa<'a> {
        ProductDfa::build_scoped(&case.graph, &self.ac, self.window)
    }

    /// Whether `molecule` carries a forbidden `k`-mer lying **entirely inside**
    /// the window — the window's stated reading, re-implemented rather than
    /// imported, so an occurrence straddling either edge is permitted.
    fn forbids(&self, molecule: &[Nucleotide]) -> bool {
        let (lo, hi) = (self.window.lo() as usize, self.window.hi() as usize);
        if hi < lo + self.k {
            return false;
        }
        (lo..=hi - self.k).any(|start| self.forbidden[kmer_code(&molecule[start..start + self.k])])
    }
}

/// Every `k`-mer that can form an ungapped, perfectly stacked duplex with some
/// window of some source flank.
///
/// Pairing is antiparallel: a run joining a flank window `w₀…w_{k−1}` to a
/// segment `d₀…d_{k−1}` pairs `w_t` with `d_{k−1−t}`. Stated from the relation
/// and enumerated over the whole `k`-mer space, so the shipped derivation's
/// reversal, its degenerate expansion and its deduplication are all on the other
/// side of the comparison.
fn forbidden_kmers(sources: &[&[Nucleotide]], k: usize, complement: Complement) -> Vec<bool> {
    let mut out = vec![false; 4usize.pow(k as u32)];
    for (code, slot) in out.iter_mut().enumerate() {
        let segment = kmer_of_code(code, k);
        *slot = sources.iter().any(|flank| {
            flank
                .windows(k)
                .any(|w| (0..k).all(|t| pairs(w[t], segment[k - 1 - t], complement)))
        });
    }
    out
}

/// The canonical pair relation, as a table rather than as a call into the code
/// the derivation is built from. `G·U` is a pair under wobble and not under
/// strict; every other canonical pair holds under both, in either orientation.
fn pairs(flank: Nucleotide, segment: Nucleotide, complement: Complement) -> bool {
    use Nucleotide::{A, C, G, U};
    match (flank, segment) {
        (A, U) | (U, A) | (C, G) | (G, C) => true,
        (G, U) | (U, G) => complement == Complement::Wobble,
        _ => false,
    }
}

// ─── small helpers ───────────────────────────────────────────────────────────

/// Start-to-terminal paths of the product, which check 1 pins to the number of
/// surviving encodings.
///
/// [`Dfa::ix`] is `pos * stride + lane`, so ascending position is a topological
/// order and one forward sweep suffices.
fn count_paths(product: &ProductDfa) -> u128 {
    let graph = product.dfa();
    let ac = product.ac();
    let mut counts = vec![0u128; product.num_product_nodes()];
    let start = graph.start();
    let Some(entry) = product.product_id(start, ac.start()) else {
        return 0;
    };
    counts[entry] = 1;
    for pos in 0..graph.seq_len {
        for &node in graph.nodes_at(pos) {
            for &q in product.reachable_states(node) {
                let here = counts[product.product_id(node, q).expect("a reachable state")];
                if here == 0 {
                    continue;
                }
                for edge in graph.right(node) {
                    let Some(next) = ac.step_at(pos, q, edge.nuc) else {
                        continue;
                    };
                    let id = product
                        .product_id(edge.to, next)
                        .expect("a step from a reachable state stays reachable");
                    counts[id] += here;
                }
            }
        }
    }
    let end = graph.end();
    product
        .reachable_states(end)
        .iter()
        .map(|&q| counts[product.product_id(end, q).expect("a reachable state")])
        .sum()
}

/// Codons a census marks unusable.
fn removed_count(census: &[Vec<bool>]) -> usize {
    census
        .iter()
        .map(|residue| residue.iter().filter(|&&usable| !usable).count())
        .sum()
}

/// Base-4 code of a `k`-mer, most significant position first.
fn kmer_code(segment: &[Nucleotide]) -> usize {
    segment.iter().fold(0usize, |acc, &n| acc * 4 + n as usize)
}

/// The inverse of [`kmer_code`].
fn kmer_of_code(mut code: usize, k: usize) -> Vec<Nucleotide> {
    let mut out = vec![Nucleotide::A; k];
    for slot in out.iter_mut().rev() {
        *slot = nucleotide((code % 4) as u8);
        code /= 4;
    }
    out
}

/// The codon table's nucleotide encoding (`A=0, C=1, G=2, U=3`), which is
/// [`Nucleotide`]'s own discriminant.
fn nucleotide(index: u8) -> Nucleotide {
    match index {
        0 => Nucleotide::A,
        1 => Nucleotide::C,
        2 => Nucleotide::G,
        _ => Nucleotide::U,
    }
}

fn nucleotides(s: &str) -> Vec<Nucleotide> {
    s.bytes()
        .map(|b| Nucleotide::from_byte(if b == b'T' { b'U' } else { b }).expect("a fixture flank"))
        .collect()
}
