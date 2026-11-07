//! The single-branch loop cap as a per-scan setting.
//!
//! The cap bounds how much unpaired sequence a bulge or internal loop may
//! carry. It is an energy-model parameter rather than a search knob: interiors
//! longer than it are unreachable, not penalized. The default is 30 unpaired
//! bases, the energy model's loop cap, and is the only value at which the scan's
//! optimum is comparable to the exact-DP and fold oracles. The cap is inclusive —
//! a loop of exactly 30 unpaired is admitted — which `single_loop_cap_boundary.rs`
//! checks against the reference minimizer at every split of the boundary.
//!
//! Five properties anchor it. The default is the energy model's and is what a caller
//! naming no cap gets; naming it explicitly is the same run. Values outside the
//! supported range are refused rather than folded. A narrowed cap is refused on
//! the paths that carry their own fixed cap, so two energy models never combine
//! into one answer. And a narrowed cap does reach the scan — it is a real
//! setting, not an accepted-and-ignored one.

use derna_lib::{
    design_with_config, design_with_motifs, BeamError, CodonUsage, Params, ScanConfig,
    DEFAULT_SINGLE_MAX_LEN, MIN_SINGLE_MAX_LEN,
};

/// Long enough to admit interiors either cap could bind on.
const PROTEIN: &str = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ";

/// Glycine is spelled `GG[ACGU]`, so adjacent glycines can spell `GGGG` and the
/// motif below necessarily drops a lattice edge here. Whether a motif set binds
/// is a property of the (protein, set) **pair**, not of the set, so a constraint that
/// fires only on a binding set needs a protein chosen to make it bind.
const G_RICH: &str = "MGGKGGQGGRGG";

/// Binding on [`G_RICH`] by construction.
const BINDING_MOTIF: &str = "GGGG";

/// The two 100-residue prefixes a sweep of loop caps found to fold a 21-nt
/// single-branch loop — the only loops over 20 in that sweep, and so the only
/// cells where a cap of 20 is known to bind.
const P0DTC2_100: &str =
    "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNI";
const Q5VYS8_100: &str =
    "MGDTAKPYFVKRTKDRGTMDDDDFRRGHPQQDYLIIDDHAKGHGSKMEKGLQKKKITPGNYGNTPRKGPCAVSSNPYAFKNPIYSQPAWMNDSHKDQSKR";

const LAMBDA: f64 = 0.99999;

/// CAI-priority. The cap binds here and nowhere else measured: at the MFE corner
/// the optimizer is buying helices and will not spend a long interior.
const CAI_CORNER: f64 = 0.0001;

fn fixtures() -> (Params, CodonUsage) {
    (
        Params::load_default().expect("Turner 2004 parameters"),
        CodonUsage::load_default().expect("canonical human codon usage"),
    )
}

fn config(single_max_len: usize) -> ScanConfig {
    ScanConfig {
        single_max_len,
        ..ScanConfig::default()
    }
}

#[test]
fn the_default_is_the_energy_models_loop_cap() {
    assert_eq!(DEFAULT_SINGLE_MAX_LEN, 30);
    assert_eq!(ScanConfig::default().single_max_len, DEFAULT_SINGLE_MAX_LEN);
    assert!(ScanConfig::default().loop_cap_is_default());
}

/// Naming the default explicitly is the same run as not naming it. This is the
/// property that lets the setting ship without re-certifying every parity claim.
#[test]
fn naming_the_default_changes_nothing() {
    let (params, usage) = fixtures();
    let implicit = design_with_config(&params, PROTEIN, &usage, LAMBDA, ScanConfig::default())
        .expect("default config designs");
    let explicit = design_with_config(
        &params,
        PROTEIN,
        &usage,
        LAMBDA,
        config(DEFAULT_SINGLE_MAX_LEN),
    )
    .expect("explicit default designs");

    assert_eq!(implicit.sequence, explicit.sequence);
    assert_eq!(implicit.structure, explicit.structure);
    assert_eq!(implicit.mfe_kcal_mol, explicit.mfe_kcal_mol);
    assert_eq!(implicit.cai, explicit.cai);
}

/// The bounds are refused rather than clamped: a caller who asks for a cap the
/// scan cannot serve is told so, and the error carries the range so its message
/// cannot drift from the constants.
#[test]
fn a_cap_outside_the_supported_range_is_refused() {
    let (params, usage) = fixtures();
    for cap in [0, 1, MIN_SINGLE_MAX_LEN - 1, DEFAULT_SINGLE_MAX_LEN + 1, 64] {
        let err = design_with_config(&params, PROTEIN, &usage, LAMBDA, config(cap))
            .expect_err("out-of-range cap is refused");
        match err {
            BeamError::LoopCapOutOfRange { cap: got, min, max } => {
                assert_eq!(got, cap);
                assert_eq!(min, MIN_SINGLE_MAX_LEN);
                assert_eq!(max, DEFAULT_SINGLE_MAX_LEN);
            }
            other => panic!("expected LoopCapOutOfRange for {cap}, got {other:?}"),
        }
    }
}

/// Both ends of the supported range are accepted, so the refusal above is a
/// statement about the bounds rather than about the setting.
#[test]
fn both_ends_of_the_range_are_accepted() {
    let (params, usage) = fixtures();
    for cap in [MIN_SINGLE_MAX_LEN, DEFAULT_SINGLE_MAX_LEN] {
        assert!(
            design_with_config(&params, PROTEIN, &usage, LAMBDA, config(cap)).is_ok(),
            "cap {cap} is inside the supported range"
        );
    }
}

/// A narrowed cap actually reaches the scan, on an instance where it is known
/// to bind.
///
/// Two weaker assertions were tried and rejected. Asserting the *design* moves
/// on an arbitrary instance fails, because the cap only binds where the optimum
/// wanted a long interior. Asserting the materialized *cell count* shrinks also
/// fails, because a cell reachable by an internal loop is usually reachable by a
/// stack or hairpin too — the cap removes candidates, not cells.
///
/// So the instance is chosen rather than arbitrary. At the CAI corner the design
/// is pure best-synonym (CAI comes back exactly 1.0000), structure goes
/// unoptimized, and whatever long interiors the sequence admits are folded. The
/// two cells below are the ones a sweep of loop caps found to carry a 21-nt
/// single-branch loop, the only loops over 20 anywhere in that sweep.
///
/// The expected deltas are theirs, measured by re-scoring a fixed molecule under
/// a cap-20 *folder*; this test reaches them by *designing* under a cap-20 scan,
/// which is a different code path. Agreement to the hundredth cross-checks that
/// both instruments mean the same thing by the cap.
///
/// It shows nothing about whether a narrowed *search* finds what a narrowed
/// *score* keeps. Both cells report CAI = 1.0000, so every codon is its
/// residue's top-CAI synonym and the sequence is pinned by the codon table —
/// no search could have differed here, and the agreement is forced rather than
/// observed. Separating the two would need a cell where the cap binds at
/// CAI < 1.0000, which the sweep did not contain.
#[test]
fn a_narrowed_cap_reaches_the_scan() {
    let (params, usage) = fixtures();
    for (protein, wide_mfe, narrow_mfe) in
        [(P0DTC2_100, -94.50, -94.30), (Q5VYS8_100, -77.50, -76.70)]
    {
        let wide = design_with_config(
            &params,
            protein,
            &usage,
            CAI_CORNER,
            config(DEFAULT_SINGLE_MAX_LEN),
        )
        .expect("default cap designs");
        let narrow = design_with_config(&params, protein, &usage, CAI_CORNER, config(20))
            .expect("cap 20 designs");

        assert!(
            (wide.mfe_kcal_mol - wide_mfe).abs() < 1e-6,
            "cap-{DEFAULT_SINGLE_MAX_LEN} MFE moved: expected {wide_mfe}, got {}",
            wide.mfe_kcal_mol
        );
        assert!(
            (narrow.mfe_kcal_mol - narrow_mfe).abs() < 1e-6,
            "cap-20 MFE moved: expected {narrow_mfe}, got {}",
            narrow.mfe_kcal_mol
        );
        // The tighter cap forbids structures the default admits, so its optimum
        // can only be worse — never better, which would mean the cap had somehow
        // enlarged the search.
        assert!(
            narrow.mfe_kcal_mol > wide.mfe_kcal_mol,
            "a cap of 20 forbids a 21-nt interior here, so its optimum must be strictly worse"
        );
    }
}

/// The AC-product constrained folder carries its own fixed cap. Mixing it with a
/// narrowed scan cap would fold one energy model and scan under another, so the
/// combination is refused by name rather than silently served.
#[test]
fn a_narrowed_cap_is_refused_under_a_binding_motif_set() {
    let (params, usage) = fixtures();
    let err = design_with_motifs(
        &params,
        G_RICH,
        &usage,
        LAMBDA,
        &[BINDING_MOTIF],
        config(MIN_SINGLE_MAX_LEN),
    )
    .expect_err("a narrowed cap under a binding motif set is refused");
    assert!(
        matches!(err, BeamError::LoopCapUnsupported(_)),
        "expected LoopCapUnsupported, got {err:?}"
    );
}

/// The default cap is served under motifs exactly as before, so the refusal
/// above is about the narrowing rather than about the constrained path.
#[test]
fn the_default_cap_is_served_under_a_motif_set() {
    let (params, usage) = fixtures();
    assert!(
        design_with_motifs(
            &params,
            G_RICH,
            &usage,
            LAMBDA,
            &[BINDING_MOTIF],
            ScanConfig::default(),
        )
        .is_ok(),
        "the default cap is unaffected by the constrained path's own cap"
    );
}
