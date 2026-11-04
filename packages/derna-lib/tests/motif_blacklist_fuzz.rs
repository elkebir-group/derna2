//! Brute-force fuzz oracle for the motif-constrained fold.
//!
//! The constrained folder ([`design_with_motifs`]) claims to return the exact
//! BRD-optimum over *only* the coding sequences that contain none of a forbidden
//! motif set. This harness proves that claim the only way it can be proven: by
//! enumerating every encoding of a small protein, dropping the ones a forbidden
//! motif touches, folding each survivor's true minimum free energy, and taking
//! the BRD-argmin — the brute-force reference the constrained substrate must
//! match **byte-exact** (MFE + CAI), sequence
//! motif-free and translatable, structure phantom-free. It mirrors the
//! `lz_fuzz` / `brd_parity` exact-lattice oracle shape.
//!
//! This is where the load-bearing correctness property — that an
//! Aho–Corasick state threads continuously through a CKY bifurcation split, so a
//! motif straddling that split is caught — goes from construction-verified to
//! optimum-proven on the folded result. Every fuzz case whose optimum bifurcates
//! is a straddle test: were the AC state ever dropped across a split, the folder
//! would either spell a forbidden motif in its optimum or undercut the brute
//! reference by folding an infeasible-but-cheaper structure, and one of the
//! motif-free / byte-exact assertions would fire. The dedicated multiloop case
//! ([`constrained_multiloop_is_optimal`]) forces a genuine 3-way junction so the
//! multibranch traceback (`walk_multi` / `walk_m2` / `walk_m`) is exercised on a
//! proven-optimal fold.
//!
//! ## Tractability
//!
//! The constrained fold is deliberately un-reduced (correct, slower), and the
//! reference enumerates *every* encoding. Both stay cheap only for short
//! proteins with a bounded encoding count (the product of per-residue codon
//! counts), so the fuzz generator caps that product ([`ENCODING_CAP`]) and keeps
//! lengths small — many tiny cases, not a few large ones.

use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS};
use derna_lib::ac::Ac;
use derna_lib::dfa;
use derna_lib::{
    design, design_with_motifs, fold_single_path, BeamError, CodonUsage, Nucleotide, Params,
    ScanConfig,
};
use rna_mfe::evaluate;

/// λ values spanning the BRD front: CAI-dominated, balanced, MFE-dominated.
const LAMBDAS: &[f64] = &[0.0001, 0.5, 0.99];

/// The maximum encoding count (product of per-residue codon counts) a fuzz case
/// may have. Keeps both the enumeration reference and the un-reduced constrained
/// fold well under a second.
const ENCODING_CAP: u128 = 4096;

// ===========================================================================
// Deterministic PRNG (xorshift64, matching `lz_fuzz` / `brd_parity`).
// ===========================================================================

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[lo, hi]`.
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize) % (hi - lo + 1)
    }
}

// ===========================================================================
// Sequence / motif helpers.
// ===========================================================================

/// The single-letter amino acids whose codon degeneracy stays small, so a random
/// protein's encoding count is likely to stay under [`ENCODING_CAP`]. Excludes
/// the six-codon L / R / S; keeps the one- and two-codon residues plus a few
/// three/four-codon ones for lattice variety.
const FUZZ_ALPHABET: &[u8] = b"MWCDEFHKNQYAGIPTV";

/// The product of per-residue codon counts — the number of distinct encodings
/// the reference must enumerate.
fn encoding_count(protein: &str) -> u128 {
    protein
        .chars()
        .map(|c| AMINO_ACIDS[aa_to_index(c).unwrap()].codons.len() as u128)
        .product()
}

/// The nucleotide character (`A/C/G/U`) for an alphabet index (`0=A,1=C,2=G,3=U`).
fn nuc_char(i: u8) -> char {
    match i {
        0 => 'A',
        1 => 'C',
        2 => 'G',
        _ => 'U',
    }
}

/// One random encoding of `protein` as an `A/C/G/U` string — a random codon per
/// residue. Used to seed motifs that are guaranteed to appear in at least one
/// encoding (so the constraint genuinely bites).
fn random_encoding(rng: &mut Rng, protein: &str) -> String {
    let mut s = String::with_capacity(protein.len() * 3);
    for c in protein.chars() {
        let codons = &AMINO_ACIDS[aa_to_index(c).unwrap()].codons;
        let k = (rng.next_u64() as usize) % codons.len();
        for &b in &codons[k].nuc {
            s.push(nuc_char(b));
        }
    }
    s
}

/// Whether running the automaton over `seq` never enters an accepting state —
/// the defensive motif-free re-check on the spelled optimum.
fn is_motif_free(ac: &Ac, seq: &[Nucleotide]) -> bool {
    let mut q = ac.start();
    for &x in seq {
        q = ac.step(q, x);
        if ac.is_accept(q) {
            return false;
        }
    }
    true
}

/// Whether `seq` (ASCII `A/C/G/U`) is a valid coding sequence for `protein`.
fn translates(seq: &str, protein: &str) -> bool {
    let b = seq.as_bytes();
    if b.len() != protein.len() * 3 {
        return false;
    }
    let code = |c: u8| match c {
        b'A' => 0u8,
        b'C' => 1,
        b'G' => 2,
        _ => 3,
    };
    protein.chars().enumerate().all(|(i, ch)| {
        let aa = aa_to_index(ch).unwrap();
        let trip = [code(b[3 * i]), code(b[3 * i + 1]), code(b[3 * i + 2])];
        AMINO_ACIDS[aa].codons.iter().any(|c| c.nuc == trip)
    })
}

/// Whether a dot-bracket structure contains a genuine multi-branch loop: a
/// closing pair that directly encloses two or more sibling stems (a 3-way-or-more
/// junction). The exterior loop's sibling stems do not count — only an *enclosed*
/// junction drives the multibranch traceback.
fn has_multiloop(structure: &str) -> bool {
    let b = structure.as_bytes();
    let mut partner = vec![usize::MAX; b.len()];
    let mut stack = Vec::new();
    for (i, &c) in b.iter().enumerate() {
        match c {
            b'(' => stack.push(i),
            b')' => {
                let j = stack.pop().expect("balanced dot-bracket");
                partner[i] = j;
                partner[j] = i;
            }
            _ => {}
        }
    }
    (0..b.len()).any(|i| b[i] == b'(' && direct_children(b, &partner, i) >= 2)
}

/// The number of stems directly enclosed by the pair opening at `i` (its 3′
/// partner is `partner[i]`), skipping over each child's whole span.
fn direct_children(b: &[u8], partner: &[usize], i: usize) -> usize {
    let j = partner[i];
    let mut k = i + 1;
    let mut children = 0;
    while k < j {
        if b[k] == b'(' {
            children += 1;
            k = partner[k] + 1;
        } else {
            k += 1;
        }
    }
    children
}

// ===========================================================================
// The brute-force reference and the oracle comparison.
// ===========================================================================

/// The brute-force constrained optimum: enumerate every encoding, keep the
/// motif-free ones, fold each survivor's true MFE (via [`fold_single_path`], the
/// λ=1 exact-lattice fold of the fixed sequence), and take the minimum BRD in
/// dekacalorie units. `None` when no encoding is motif-free (infeasible). This is
/// the sole correctness authority for the constraint.
fn brute_force(
    params: &Params,
    usage: &CodonUsage,
    protein: &str,
    motifs: &[&str],
    lambda: f64,
) -> Option<f64> {
    let aa: Vec<usize> = protein.chars().map(|c| aa_to_index(c).unwrap()).collect();
    let graph = dfa::build(&aa, usage);
    let ac = Ac::build(motifs).expect("valid motifs");
    let mut best: Option<f64> = None;
    for (cai, seq) in graph.paths_between(graph.start(), graph.end()) {
        if !is_motif_free(&ac, &seq) {
            continue;
        }
        let codons: Vec<[u8; 3]> = seq
            .chunks_exact(3)
            .map(|c| [c[0] as u8, c[1] as u8, c[2] as u8])
            .collect();
        let mfe_dekacal = fold_single_path(params, &aa, &codons).mfe_kcal_mol * 100.0;
        let brd = lambda * mfe_dekacal - (1.0 - lambda) * cai * 100.0;
        best = Some(best.map_or(brd, |b: f64| b.min(brd)));
    }
    best
}

/// The BRD of a folded design in dekacalorie units, matching [`brute_force`].
fn design_brd(mfe_kcal_mol: f64, cai_log: f64, lambda: f64) -> f64 {
    lambda * (mfe_kcal_mol * 100.0) - (1.0 - lambda) * (cai_log * 100.0)
}

/// What an oracle check exercised — tracked so a sweep can assert it covers the
/// branches that matter rather than silently degenerating to all-trivial cases.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Outcome {
    /// Feasible, and the constraint genuinely bit: the *unconstrained* optimum
    /// contains a forbidden motif, so the folder had to find a different one.
    FeasibleActive,
    /// Feasible, but the unconstrained optimum was already motif-free (the
    /// constraint removed only sub-optimal encodings).
    FeasibleInactive,
    /// No motif-free encoding of the protein exists.
    Infeasible,
}

/// The oracle assertion: the folder's result must agree with the brute-force
/// reference on feasibility, and — when feasible — spell a motif-free coding
/// sequence, fold phantom-free to its reported MFE, and achieve the brute-force
/// optimal BRD byte-exact. Returns which branch it exercised.
fn assert_oracle(
    params: &Params,
    usage: &CodonUsage,
    protein: &str,
    motifs: &[&str],
    lambda: f64,
) -> Outcome {
    let ctx = format!("protein={protein} motifs={motifs:?} λ={lambda}");
    let want = brute_force(params, usage, protein, motifs, lambda);
    let got = design_with_motifs(
        params,
        protein,
        usage,
        lambda,
        motifs,
        ScanConfig::default(),
    );
    let ac = Ac::build(motifs).expect("valid motifs");
    match (want, got) {
        (Some(w), Ok(d)) => {
            let seq = parse_seq(&d.sequence);
            assert!(
                is_motif_free(&ac, &seq),
                "spelled a forbidden motif: {ctx}: {}",
                d.sequence,
            );
            assert!(
                translates(&d.sequence, protein),
                "spelled a non-coding sequence: {ctx}: {}",
                d.sequence,
            );
            let reeval = evaluate(params, d.sequence.as_bytes(), &d.structure)
                .expect("recovered structure scores")
                .dekacal() as f64
                / 100.0;
            assert!(
                (reeval - d.mfe_kcal_mol).abs() <= 0.005,
                "phantom energy: {ctx}: reported={:.2} reeval={reeval:.2} structure={}",
                d.mfe_kcal_mol,
                d.structure,
            );
            let got_brd = design_brd(d.mfe_kcal_mol, d.cai_log, lambda);
            assert!(
                (w - got_brd).abs() < 1e-6,
                "constrained fold sub-optimal: {ctx}: brute={w} fold={got_brd}",
            );
            // The constraint "bit" iff the unconstrained optimum was forbidden.
            let unconstrained = design(params, protein, usage, lambda).expect("unconstrained");
            if is_motif_free(&ac, &parse_seq(&unconstrained.sequence)) {
                Outcome::FeasibleInactive
            } else {
                Outcome::FeasibleActive
            }
        }
        (None, Err(BeamError::Infeasible)) => Outcome::Infeasible,
        (want, got) => panic!("feasibility mismatch: {ctx}: brute={want:?} folder={got:?}"),
    }
}

/// Parse an `A/C/G/U` string into nucleotides.
fn parse_seq(seq: &str) -> Vec<Nucleotide> {
    seq.bytes()
        .map(|c| Nucleotide::from_byte(c).expect("A/C/G/U"))
        .collect()
}

// ===========================================================================
// Pass-through and infeasibility contracts.
// ===========================================================================

/// The additivity floor: an empty forbidden-motif set routes through the
/// unconstrained scan verbatim, so [`design_with_motifs`] is byte-identical to
/// [`design`]. The oracle also holds for it (the brute reference enumerates the
/// whole set), so both are checked.
#[test]
fn pass_through_is_byte_identical() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    for protein in ["MW", "KFKF", "DEYWCN", "MKTAYI"] {
        for &lambda in LAMBDAS {
            let exact = design(&params, protein, &usage, lambda).unwrap();
            let empty = design_with_motifs(
                &params,
                protein,
                &usage,
                lambda,
                &[] as &[&str],
                ScanConfig::default(),
            )
            .unwrap();
            assert_eq!(
                exact, empty,
                "empty motif set diverged: {protein} λ={lambda}"
            );
            assert_oracle(&params, &usage, protein, &[], lambda);
        }
    }
}

/// Infeasibility is reported cleanly: a motif set that forbids every
/// encoding of the protein returns [`BeamError::Infeasible`], and the brute
/// reference agrees there is no motif-free encoding.
#[test]
fn infeasible_is_reported_cleanly() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    // Uniquely-encoded proteins whose sole codons are forbidden outright.
    let cases: &[(&str, &[&str])] = &[
        ("MW", &["AUG"]),  // Met is AUG-only, so no encoding survives.
        ("MW", &["ATG"]),  // T read as U — same verdict as AUG.
        ("MWM", &["UGG"]), // Trp is UGG-only.
        ("KF", &["AAR"]),  // K∈{AAA,AAG}=AA[AG]=AAR forbidden ⇒ K unencodable.
    ];
    for &(protein, motifs) in cases {
        for &lambda in LAMBDAS {
            assert_eq!(
                brute_force(&params, &usage, protein, motifs, lambda),
                None,
                "expected no motif-free encoding: {protein} × {motifs:?}",
            );
            assert_eq!(
                design_with_motifs(
                    &params,
                    protein,
                    &usage,
                    lambda,
                    motifs,
                    ScanConfig::default()
                ),
                Err(BeamError::Infeasible),
                "expected clean infeasibility: {protein} × {motifs:?} λ={lambda}",
            );
        }
    }
}

// ===========================================================================
// The named hard-case matrix.
// ===========================================================================

/// The hard cases: a motif spanning a codon boundary, overlapping /
/// shared-prefix motifs (AC failure/output links), a motif longer than a hairpin
/// span (multi-codon), and IUPAC-degenerate codes. Each is verified byte-exact
/// against the brute-force reference across the λ front. The named coverage sits
/// on top of the randomized sweep so a regression in any one shape is legible.
#[test]
fn d4_named_hard_cases() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let cases: &[(&str, &[&str])] = &[
        // Codon-boundary: the motif forms only across the residue boundary, so
        // it bites one codon pairing while leaving the residue encodable.
        // K∈{AAA,AAG}, F∈{UUU,UUC}: "GUU" appears only as ...AAG|UU... .
        ("KF", &["GUU"]),
        // F∈{UUU,UUC}, K∈{AAA,AAG}: "CAA" appears only as ...UUC|AA... .
        ("FK", &["CAA"]),
        // Overlapping / shared-prefix motifs — exercises AC failure & output
        // links. H∈{CAU,CAC}, M=AUG: "CAC" and "ACA" both bite H=CAC (the
        // latter across H|M), leaving H=CAU.
        ("HMHM", &["CAC", "ACA"]),
        // Long, multi-codon motifs (longer than a minimal hairpin span): the
        // 6-mer bites the AAG|UUU pairing across K|F, leaving AAA|UUU.
        ("KFKF", &["AAGUUU"]),
        // IUPAC degenerate code that bites a pairing but not the whole residue:
        // "AAGUUN" (N=any) forbids K=AAG followed by either F codon, so K=AAA
        // survives.
        ("KFKF", &["AAGUUN"]),
        // A deliberately infeasible IUPAC case: "GN" (G then any) forbids every
        // G-initial dinucleotide, and D∈{GAU,GAC} both start with G, so D is
        // unencodable — clean infeasibility, not a panic.
        ("DFK", &["GN"]),
        // A long motif that forbids a uniquely-encoded protein outright: M=AUG,
        // W=UGG, so every MW spells AUGUGG — infeasible.
        ("MWMW", &["AUGUGG"]),
    ];
    let (mut active, mut inactive, mut infeasible) = (0u32, 0u32, 0u32);
    for &(protein, motifs) in cases {
        assert!(
            encoding_count(protein) <= ENCODING_CAP,
            "named case {protein} exceeds the enumeration cap",
        );
        for &lambda in LAMBDAS {
            match assert_oracle(&params, &usage, protein, motifs, lambda) {
                Outcome::FeasibleActive => active += 1,
                Outcome::FeasibleInactive => inactive += 1,
                Outcome::Infeasible => infeasible += 1,
            }
        }
    }
    eprintln!("motif oracle coverage: active={active} inactive={inactive} infeasible={infeasible}");
    // The matrix must exercise both an active constraint and a clean
    // infeasibility, or it has stopped testing the hard shapes.
    assert!(
        active > 0 && infeasible > 0,
        "motif oracle matrix went vacuous: {active}/{inactive}/{infeasible}"
    );
}

// ===========================================================================
// The randomized fuzz sweep.
// ===========================================================================

/// A random protein of `len` residues over [`FUZZ_ALPHABET`] whose encoding
/// count stays within [`ENCODING_CAP`]. Retries with the running seed until the
/// cap holds (short proteins over this alphabet almost always fit at once).
fn random_bounded_protein(rng: &mut Rng, len: usize) -> String {
    loop {
        let protein: String = (0..len)
            .map(|_| FUZZ_ALPHABET[(rng.next_u64() as usize) % FUZZ_ALPHABET.len()] as char)
            .collect();
        if encoding_count(&protein) <= ENCODING_CAP {
            return protein;
        }
    }
}

/// A biting motif: a random substring (length 2..=6) of a random encoding of the
/// protein, so it is guaranteed to appear in at least one encoding. With
/// probability ~½ one interior position is generalized to an IUPAC code covering
/// its base, exercising the degenerate-expansion path while staying biting.
fn biting_motif(rng: &mut Rng, protein: &str) -> String {
    let enc = random_encoding(rng, protein);
    let bytes = enc.as_bytes();
    let max_len = bytes.len().min(6);
    let len = rng.range(2, max_len.max(2));
    let start = if bytes.len() > len {
        (rng.next_u64() as usize) % (bytes.len() - len + 1)
    } else {
        0
    };
    let mut motif: Vec<u8> = bytes[start..start + len].to_vec();
    if rng.next_u64() & 1 == 0 {
        let pos = (rng.next_u64() as usize) % motif.len();
        motif[pos] = iupac_cover(motif[pos], rng);
    }
    String::from_utf8(motif).unwrap()
}

/// An IUPAC code whose realization set contains `base` (`A/C/G/U`), so replacing
/// `base` with it keeps the motif matching the encoding it was drawn from.
fn iupac_cover(base: u8, rng: &mut Rng) -> u8 {
    let options: &[u8] = match base {
        b'A' => b"ARWMN",
        b'C' => b"CYSMN",
        b'G' => b"GRSKN",
        _ => b"UYWKN", // U
    };
    options[(rng.next_u64() as usize) % options.len()]
}

/// The core sweep: random short proteins × biting motif sets × the λ front,
/// each asserted byte-exact against the brute-force reference. Deterministic and
/// CI-stable — a fixed case count seeded from `base_seed`, not a wall-clock loop.
fn fuzz_sweep(base_seed: u64, cases: usize, lo: usize, hi: usize) {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let (mut active, mut inactive, mut infeasible) = (0u32, 0u32, 0u32);
    for i in 0..cases {
        let seed = base_seed.wrapping_add(i as u64);
        let mut rng = Rng::new(seed);
        let len = rng.range(lo, hi);
        let protein = random_bounded_protein(&mut rng, len);
        let n_motifs = rng.range(1, 3);
        let motifs: Vec<String> = (0..n_motifs)
            .map(|_| biting_motif(&mut rng, &protein))
            .collect();
        let motif_refs: Vec<&str> = motifs.iter().map(String::as_str).collect();
        for &lambda in LAMBDAS {
            match assert_oracle(&params, &usage, &protein, &motif_refs, lambda) {
                Outcome::FeasibleActive => active += 1,
                Outcome::FeasibleInactive => inactive += 1,
                Outcome::Infeasible => infeasible += 1,
            }
        }
    }
    eprintln!("fuzz coverage: active={active} inactive={inactive} infeasible={infeasible}");
    // The sweep is worthless if the constraint never reshapes an optimum; a
    // biting motif set makes the active branch the common one.
    assert!(
        active > 0,
        "no fuzz case had an active constraint (active={active} inactive={inactive} infeasible={infeasible})",
    );
}

/// Fast default gate: a spread of short proteins with biting motif sets across
/// the λ front. Runs in the standard `cargo test` pass.
#[test]
fn fuzz_matches_brute_force_quick() {
    fuzz_sweep(0x5C0F_1701, 60, 2, 5);
}

/// Heavier sweep — more cases and slightly longer proteins — behind `--ignored`,
/// mirroring the `lz_fuzz` rigorous oracle. Run with
/// `cargo test -p derna-lib --release --test motif_blacklist_fuzz -- --ignored`.
#[test]
#[ignore = "rigorous brute-force sweep; run with --ignored --release"]
fn fuzz_matches_brute_force_rigorous() {
    fuzz_sweep(0x5C0F_1702, 240, 4, 7);
}

// ===========================================================================
// The multibranch-traceback gate: a proven-optimal 3-way junction.
// ===========================================================================

/// A protein whose constrained optimum folds to a genuine multi-branch loop
/// under an active motif, with a small enough encoding set to enumerate. This is
/// the case the tiny folds above cannot reach: the multibranch traceback
/// (`walk_multi` / `walk_m2` / `walk_m`) fires only inside an enclosed 3-way
/// junction, which the Turner model favors only for longer sequences.
const MULTILOOP_PROTEIN: &str = "NWWFQWMMWHWMCMMFMQDQ";
/// A motif that prunes part of [`MULTILOOP_PROTEIN`]'s encoding set (128 of its
/// 512 encodings stay motif-free) but is absent from the multiloop optimum, so
/// the constraint is genuinely active while the optimum stays a 3-way junction.
const MULTILOOP_MOTIF: &str = "AAU";

/// The multibranch-traceback gate. The constrained optimum of a longer protein
/// is a genuine 3-way junction: `has_multiloop` confirms the enclosed junction,
/// `assert_oracle` confirms it is motif-free, phantom-free, and byte-exact
/// against the brute-force reference. Together they prove the multibranch
/// recurrences and their traceback are optimum-correct under the constraint — the
/// coverage reachable only at this length. Heavier than the tiny cases (the
/// un-reduced fold at multiloop length), so it runs behind `--ignored`.
#[test]
#[ignore = "un-reduced multiloop fold; run with --ignored --release"]
fn constrained_multiloop_is_optimal() {
    let params = Params::load_default().unwrap();
    let usage = CodonUsage::load_default().unwrap();
    let protein = MULTILOOP_PROTEIN;
    let motifs: &[&str] = &[MULTILOOP_MOTIF];
    // MFE-priority λ, where the structure drives the optimum.
    let lambda = 0.99;
    let d = design_with_motifs(
        &params,
        protein,
        &usage,
        lambda,
        motifs,
        ScanConfig::default(),
    )
    .expect("multiloop case is feasible");
    assert!(
        has_multiloop(&d.structure),
        "expected a 3-way junction, got {}",
        d.structure,
    );
    assert_oracle(&params, &usage, protein, motifs, lambda);
}
