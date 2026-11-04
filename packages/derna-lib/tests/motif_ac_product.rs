//! Construction-level verification of the AC × codon-DFA product (no folding).
//!
//! These tests exercise the Aho–Corasick automaton and the product lift in
//! isolation — before any CKY table threads the AC state — and pin the
//! load-bearing threading property: the AC state at a boundary node is an
//! *absolute* function of the linear prefix, so a single state per boundary
//! node suffices (the `|A_M|` hypothesis, not the `|A_M|²` entry/exit-pair
//! trap). The oracle here is direct AC simulation over enumerated encodings;
//! the full folded brute-force oracle lives in `motif_blacklist_fuzz.rs`.
//!
//! Reading conventions: proteins are single-letter amino-acid strings, motifs
//! and sequences are `A/C/G/U` (with `T` accepted as `U` on input); a sequence
//! is *motif-free* when running the automaton over it linearly never enters an
//! accepting state.

use std::collections::BTreeSet;

use derna_codon_dp::codon_table::aa_to_index;
use derna_lib::ac::Ac;
use derna_lib::dfa::{self, Dfa};
use derna_lib::product::ProductDfa;
use derna_lib::{CodonUsage, Nucleotide};

/// The human codon-usage table. The product construction reads only edge
/// nucleotides, so the CAI weights are irrelevant here — any table serves.
fn usage() -> CodonUsage {
    CodonUsage::load_default().expect("default codon usage")
}

/// Build the base codon DFA for a single-letter amino-acid protein.
fn dfa_for(protein: &str) -> Dfa {
    let aa: Vec<usize> = protein
        .chars()
        .map(|c| aa_to_index(c).expect("standard amino acid"))
        .collect();
    dfa::build(&aa, &usage())
}

/// Every coding sequence the DFA spells, as `A/C/G/U` strings.
fn all_encodings(graph: &Dfa) -> Vec<String> {
    graph
        .paths_between(graph.start(), graph.end())
        .into_iter()
        .map(|(_, seq)| to_str(&seq))
        .collect()
}

fn to_str(seq: &[Nucleotide]) -> String {
    seq.iter()
        .map(|n| match n {
            Nucleotide::A => 'A',
            Nucleotide::C => 'C',
            Nucleotide::G => 'G',
            Nucleotide::U => 'U',
        })
        .collect()
}

fn seq_of(s: &str) -> Vec<Nucleotide> {
    s.bytes()
        .map(|b| Nucleotide::from_byte(b).expect("A/C/G/U"))
        .collect()
}

/// Whether running the automaton over `seq` linearly never enters an accepting
/// state — the reference definition of "contains no forbidden motif".
fn motif_free(ac: &Ac, seq: &[Nucleotide]) -> bool {
    let mut q = ac.start();
    for &x in seq {
        q = ac.step(q, x);
        if ac.is_accept(q) {
            return false;
        }
    }
    true
}

/// The subset of `encodings` the automaton considers motif-free.
fn motif_free_set(ac: &Ac, encodings: &[String]) -> BTreeSet<String> {
    encodings
        .iter()
        .filter(|s| motif_free(ac, &seq_of(s)))
        .cloned()
        .collect()
}

/// The subset of `encodings` the product actually spells.
fn spellable_set(product: &ProductDfa, encodings: &[String]) -> BTreeSet<String> {
    encodings
        .iter()
        .filter(|s| product.spells(&seq_of(s)))
        .cloned()
        .collect()
}

/// Build the automaton and product for `protein` × `motifs`, returning the
/// pieces the assertions consume.
fn build_case(protein: &str, motifs: &[&str]) -> (Dfa, Ac, Vec<String>) {
    let graph = dfa_for(protein);
    let ac = Ac::build(motifs).expect("valid motif set");
    let encodings = all_encodings(&graph);
    (graph, ac, encodings)
}

// ---------------------------------------------------------------------------
// The core equivalence: spellable == motif-free, across the hard cases.
// ---------------------------------------------------------------------------

/// The product-level analogue of the full fold oracle: enumerate every encoding of a
/// tiny protein, and assert the product spells *exactly* the motif-free ones.
/// The matrix covers a codon-boundary straddle, an interior straddle, a
/// whole-length motif, overlapping/shared-suffix motifs, and IUPAC codes.
#[test]
fn product_spells_exactly_the_motif_free_encodings() {
    let cases: &[(&str, &[&str])] = &[
        ("KF", &[]),              // empty set: everything is spellable
        ("KF", &["GU"]),          // straddles the codon boundary (…G|U…)
        ("KF", &["AAG"]),         // wholly inside the first codon
        ("KF", &["AAAUUU"]),      // a motif as long as the whole sequence
        ("KF", &["AAG", "GUU"]),  // overlapping, share the boundary G
        ("KF", &["GN"]),          // IUPAC: G then anything
        ("MF", &["UGU", "GUU"]),  // shared-suffix motifs across the split
        ("KFK", &["GUU", "UUA"]), // two straddles in a longer protein
    ];
    for &(protein, motifs) in cases {
        let (graph, ac, encodings) = build_case(protein, motifs);
        let product = ProductDfa::build(&graph, &ac);
        let expected = motif_free_set(&ac, &encodings);
        let actual = spellable_set(&product, &encodings);
        assert_eq!(
            actual, expected,
            "spellable set diverged from motif-free set for {protein} × {motifs:?}",
        );
        // Feasibility is exactly "some encoding survives".
        assert_eq!(
            product.is_feasible(),
            !expected.is_empty(),
            "feasibility disagreed with the survivor set for {protein} × {motifs:?}",
        );
    }
}

// ---------------------------------------------------------------------------
// (a) Each boundary node carries one absolute AC state along any concrete path;
//     a bifurcation split is a single shared state both sides agree on.
// ---------------------------------------------------------------------------

/// The AC state at position `k` is a function of the prefix `x₀…x_{k-1}` alone,
/// so threading the prefix and *continuing* from the state it lands in
/// reproduces the whole-sequence trajectory — for every split `k`. This is the
/// property that makes a CKY bifurcation boundary a single shared product node:
/// both sub-derivations reach the split at the one absolute state, never an
/// entry/exit pair.
#[test]
fn ac_state_threads_continuously_through_every_split() {
    let (_graph, ac, encodings) = build_case("KFK", &["GUU", "AAA"]);
    for enc in &encodings {
        let seq = seq_of(enc);
        let whole = trajectory(&ac, ac.start(), &seq);
        for k in 0..=seq.len() {
            let q_k = trajectory(&ac, ac.start(), &seq[..k]);
            let split = trajectory(&ac, *q_k.last().unwrap(), &seq[k..]);
            let stitched: Vec<u32> = q_k.iter().chain(split.iter().skip(1)).copied().collect();
            assert_eq!(
                stitched, whole,
                "state trajectory for {enc} was not continuous across split {k}",
            );
        }
    }
}

/// The sequence of AC states visited reading `seq` from `start`, length
/// `seq.len() + 1` (the entry state, then one per nucleotide).
fn trajectory(ac: &Ac, start: u32, seq: &[Nucleotide]) -> Vec<u32> {
    let mut q = start;
    let mut out = vec![q];
    for &x in seq {
        q = ac.step(q, x);
        out.push(q);
    }
    out
}

// ---------------------------------------------------------------------------
// (b) A motif straddling a chosen boundary is unspellable.
// ---------------------------------------------------------------------------

/// A motif placed across the codon boundary is excluded, and when it kills the
/// last survivor the product reports infeasibility rather than spelling it.
#[test]
fn motif_straddling_the_codon_boundary_is_unspellable() {
    // "MM" has the single encoding AUGAUG; "GAU" straddles the boundary at
    // position 3 (…AU G | A U G…), so no encoding survives.
    let (graph, ac, encodings) = build_case("MM", &["GAU"]);
    let product = ProductDfa::build(&graph, &ac);
    assert_eq!(encodings, vec!["AUGAUG".to_string()]);
    assert!(!product.spells(&seq_of("AUGAUG")));
    assert!(!product.is_feasible());

    // "KF" with a boundary-straddling "GU" excludes exactly the encodings whose
    // first codon ends in G; the G-free encodings still spell.
    let (graph, ac, _) = build_case("KF", &["GU"]);
    let product = ProductDfa::build(&graph, &ac);
    assert!(product.is_feasible());
    assert!(!product.spells(&seq_of("AAGUUU")));
    assert!(!product.spells(&seq_of("AAGUUC")));
    assert!(product.spells(&seq_of("AAAUUU")));
    assert!(product.spells(&seq_of("AAAUUC")));
}

/// A straddling motif is only a meaningful exclusion if the base DFA could
/// otherwise spell it — guard against a vacuously-passing test.
#[test]
fn the_straddle_case_is_not_vacuous() {
    let graph = dfa_for("KF");
    let encodings = all_encodings(&graph);
    assert!(
        encodings.iter().any(|s| s.contains("GU")),
        "no unconstrained encoding contains the straddling motif — the exclusion would be vacuous",
    );
}

// ---------------------------------------------------------------------------
// (c) The reachable-AC-state set matches direct enumeration.
// ---------------------------------------------------------------------------

/// The product's per-node reachable AC states equal what an enumeration-driven
/// walk observes: for every encoding, thread the automaton down every base lane
/// and record the state at each node reached by a motif-free prefix. This
/// validates the reachability material the sparse layout consumes.
#[test]
fn reachable_states_match_enumeration() {
    let (graph, ac, encodings) = build_case("KFK", &["GUU", "AAA"]);
    let product = ProductDfa::build(&graph, &ac);
    let observed = observed_reachable(&graph, &ac, &encodings);
    for pos in 0..=graph.seq_len {
        for &node in graph.nodes_at(pos) {
            let from_product: BTreeSet<u32> =
                product.reachable_states(node).iter().copied().collect();
            assert_eq!(
                from_product,
                observed[graph.ix(node)],
                "reachable AC states disagreed at node {node:?}",
            );
        }
    }
}

/// Enumeration-driven oracle for the reachable `(node, q)` set: replay every
/// encoding through the product's frontier rule and collect the states seen at
/// each node. Independent of the product's own BFS.
fn observed_reachable(graph: &Dfa, ac: &Ac, encodings: &[String]) -> Vec<BTreeSet<u32>> {
    let n_nodes = graph.slots();
    let mut observed = vec![BTreeSet::new(); n_nodes];
    observed[graph.ix(graph.start())].insert(ac.start());
    for enc in encodings {
        let mut frontier = vec![(graph.start(), ac.start())];
        for x in seq_of(enc) {
            let mut next = Vec::new();
            for (node, q) in frontier {
                let next_q = ac.step(q, x);
                if ac.is_accept(next_q) {
                    continue;
                }
                for edge in graph.right(node) {
                    if edge.nuc == x {
                        observed[graph.ix(edge.to)].insert(next_q);
                        next.push((edge.to, next_q));
                    }
                }
            }
            frontier = next;
        }
    }
    observed
}

// ---------------------------------------------------------------------------
// Empty-motif-set pass-through: structurally identical to the base DFA.
// ---------------------------------------------------------------------------

/// With no motifs the automaton is a single non-accepting root, no edge is
/// removed, and every reachable node carries exactly the root state — the
/// product collapses to the base DFA.
#[test]
fn empty_motif_set_is_pass_through() {
    let graph = dfa_for("KFK");
    let ac = Ac::build::<&str>(&[]).expect("empty set");
    assert!(ac.is_pass_through());
    let product = ProductDfa::build(&graph, &ac);
    assert!(product.is_pass_through());

    let mut base_nodes = 0usize;
    for pos in 0..=graph.seq_len {
        for &node in graph.nodes_at(pos) {
            assert_eq!(
                product.reachable_states(node),
                &[ac.start()],
                "node {node:?} should carry exactly the root AC state",
            );
            base_nodes += 1;
        }
    }
    assert_eq!(product.num_product_nodes(), base_nodes);

    // Every encoding is spellable; nothing is excluded.
    let encodings = all_encodings(&graph);
    assert_eq!(
        spellable_set(&product, &encodings),
        encodings.iter().cloned().collect::<BTreeSet<_>>(),
    );
}

// ---------------------------------------------------------------------------
// Whether the constraint removed anything *here*: the routing predicate.
// ---------------------------------------------------------------------------

/// An empty set removes nothing, so the product reports no dropped edge and a
/// caller may route the run to the base scan.
#[test]
fn empty_motif_set_drops_no_edge() {
    let graph = dfa_for("KFK");
    let ac = Ac::build::<&str>(&[]).expect("empty set");
    assert!(!ProductDfa::build(&graph, &ac).dropped_edge());
}

/// A pattern set that matches somewhere in the alphabet but nowhere this
/// protein can spell removes nothing either. This is the case a check on the
/// *automaton* gets wrong: the automaton is not pass-through, yet the product
/// is the base lattice, and folding it costs the AC axis for nothing.
#[test]
fn unspellable_motif_drops_no_edge() {
    let graph = dfa_for("KFK");
    for motifs in [&["ACGUACGUACGU"][..], &["CCUCCC", "GCCGCC"][..]] {
        let ac = Ac::build(motifs).expect("valid motifs");
        assert!(
            !ac.is_pass_through(),
            "{motifs:?} should build a real automaton"
        );
        let product = ProductDfa::build(&graph, &ac);
        assert!(
            !product.dropped_edge(),
            "{motifs:?} spells nothing KFK can spell, so no edge is removed",
        );
        // The lattices agree because nothing was removed, which is what makes
        // the reroute byte-exact rather than merely cheap.
        let encodings = all_encodings(&graph);
        assert_eq!(
            spellable_set(&product, &encodings),
            encodings.iter().cloned().collect::<BTreeSet<_>>(),
        );
    }
}

/// A pattern the protein *can* spell sets the flag, and — the case that makes
/// the codon census the wrong predicate — it sets the flag while every codon
/// remains usable somewhere.
///
/// `FF` spells `UUU`/`UUC` twice. Forbidding `UUUUUU` removes exactly one of
/// the four encodings; each of the four codon choices still appears in a
/// survivor, so a per-codon census reads inert and the constraint is not. A
/// caller routing on that census would send a genuinely constrained run to the
/// base scan and emit a sequence the set forbids.
#[test]
fn boundary_straddling_motif_drops_an_edge_with_every_codon_still_usable() {
    let graph = dfa_for("FF");
    let ac = Ac::build(&["UUUUUU"]).expect("valid motif");
    let product = ProductDfa::build(&graph, &ac);

    assert!(product.dropped_edge());
    assert!(
        product
            .surviving_codons()
            .iter()
            .all(|r| r.iter().all(|&c| c)),
        "every codon should still lie on some motif-free encoding",
    );
    assert_eq!(
        spellable_set(&product, &all_encodings(&graph)),
        ["UUCUUC", "UUCUUU", "UUUUUC"]
            .iter()
            .map(|s| s.to_string())
            .collect::<BTreeSet<_>>(),
    );
}

/// The flag is conservative in the safe direction: a set whose only removals
/// are on paths that die anyway still reports a dropped edge, so the run takes
/// the constrained path. Being wrong the other way would emit a forbidden
/// sequence with no error.
#[test]
fn dropped_edge_is_set_even_when_the_removal_is_on_a_dead_path() {
    // Trp is `UGG`-only, so every encoding of `W` is forbidden: nothing is
    // spellable, and every removal is on a path to nowhere.
    let graph = dfa_for("W");
    let ac = Ac::build(&["UGG"]).expect("valid motif");
    let product = ProductDfa::build(&graph, &ac);
    assert!(!product.is_feasible());
    assert!(product.dropped_edge());
}

// ---------------------------------------------------------------------------
// Sparse-layout addressing: the dense product-id numbering.
// ---------------------------------------------------------------------------

/// The dense product-id numbering is a bijection onto `0..num_product_nodes()`:
/// every reachable `(node, q)` maps to a distinct id, `base_node_of` inverts the
/// node component, and unreachable states have no id. This is the addressing the
/// sparse CKY layout indexes by, materializing only the reachable `(node, q)`
/// cells rather than the nominal `|nodes| · |A_M|`.
#[test]
fn product_ids_are_a_dense_bijection() {
    // A curated set with several AC states so per-node reachable sets exceed 1.
    let (graph, ac, _) = build_case("KFK", &["GUU", "AAG", "UUC"]);
    let product = ProductDfa::build(&graph, &ac);

    let mut seen = BTreeSet::new();
    for pos in 0..=graph.seq_len {
        for &node in graph.nodes_at(pos) {
            for &q in product.reachable_states(node) {
                let id = product
                    .product_id(node, q)
                    .expect("reachable state has an id");
                assert!(id < product.num_product_nodes(), "id {id} out of range");
                assert!(seen.insert(id), "duplicate product id {id}");
                assert_eq!(
                    product.base_node_of(id),
                    node,
                    "base_node_of inverts the node"
                );
            }
            // An AC state that is not reachable at this node has no id.
            let unreachable = ac.num_states() as u32;
            assert_eq!(product.product_id(node, unreachable), None);
        }
    }
    // The ids cover the whole range with no gaps.
    assert_eq!(seen.len(), product.num_product_nodes());
    assert_eq!(
        seen.iter().next_back().copied(),
        Some(product.num_product_nodes() - 1)
    );
}

// ---------------------------------------------------------------------------
// Automaton behaviors: IUPAC expansion, T→U, suffix-motif output links, errors.
// ---------------------------------------------------------------------------

/// Reading a single nucleotide from the root, is it accepted? A one-shot probe
/// of `δ`(root, x) landing in an accept state.
fn accepts_single(ac: &Ac, x: Nucleotide) -> bool {
    ac.is_accept(ac.step(ac.start(), x))
}

#[test]
fn iupac_codes_expand_to_their_literal_classes() {
    let r = Ac::build(&["R"]).unwrap(); // R = {A, G}
    assert!(accepts_single(&r, Nucleotide::A));
    assert!(accepts_single(&r, Nucleotide::G));
    assert!(!accepts_single(&r, Nucleotide::C));
    assert!(!accepts_single(&r, Nucleotide::U));

    // N accepts every nucleotide; the expansion adds the whole class.
    let n = Ac::build(&["N"]).unwrap();
    for x in [Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::U] {
        assert!(accepts_single(&n, x));
    }
}

#[test]
fn t_is_read_as_u() {
    let t = Ac::build(&["T"]).unwrap();
    assert!(accepts_single(&t, Nucleotide::U));
    assert!(!accepts_single(&t, Nucleotide::A));
}

#[test]
fn suffix_motifs_are_caught_via_output_links() {
    // "UG" is a suffix of "CUG"; the failure link must still flag acceptance
    // when the longer path was never a motif on its own.
    let ac = Ac::build(&["AUG", "UG"]).unwrap();
    assert!(!motif_free(&ac, &seq_of("CUG"))); // caught by the short motif "UG"
    assert!(!motif_free(&ac, &seq_of("AUG"))); // caught by the long motif "AUG"
    assert!(motif_free(&ac, &seq_of("ACC"))); // neither motif present
}

#[test]
fn invalid_and_empty_motifs_are_rejected() {
    use derna_lib::ac::MotifError;
    assert!(matches!(Ac::build(&[""]), Err(MotifError::Empty)));
    assert!(matches!(
        Ac::build(&["AXG"]),
        Err(MotifError::InvalidCode { code: 'X', .. }),
    ));
}
