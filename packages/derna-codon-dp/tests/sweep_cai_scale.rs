//! Placing a sweep's λ values as a solver on another CAI scale would.
//!
//! This solver quantizes CAI as `round(log_CAI · 100)` where DERNA C++ uses
//! raw `log_CAI`, so the same trade-off sits at a different λ in each, and the
//! sweep's thresholds are widths **in λ**. Reading C++'s thresholds on this
//! solver's λ therefore subdivides where C++ had already stopped.
//! `solve_pord_on_cai_scale` converts to the other solver's λ before it
//! measures a width or bisects, and converts back to solve, so the weights are
//! that solver's and every answer is still this one's.
//!
//! What these tests can and cannot establish: DERNA C++ is not built here, so
//! they check the transform's own properties, and anchor the weight **count**
//! against the recorded C++ figure in a test held out of the default run for
//! its cost. They do not compare weight-for-weight against a live C++ run.

use derna_codon_dp::codon_usage::CodonUsage;
use derna_codon_dp::{solve_pord_on_cai_scale, BrdOptions, ParetoPoint, CPP_CAI_SCALE};

/// Short enough that a sweep of it is milliseconds, which is what lets the
/// properties below run every time.
const SHORT: &str = "GLAGPR";

/// The first 32 residues of the spike protein, the length the 62-against-21
/// count was recorded at.
const ANCHOR: &str = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSF";

fn sweep(protein: &str, cai_scale: f64) -> (Vec<ParetoPoint>, Vec<f64>) {
    let codon_usage = CodonUsage::load_default().expect("load default codon usage");
    let options = BrdOptions {
        dangling_5_prime: false,
        codon_mask: None,
    };
    let weights = std::cell::RefCell::new(Vec::new());
    let note = |sol: &derna_codon_dp::BrdSolution, _: f64| {
        weights.borrow_mut().push(sol.lambda_);
    };
    let (points, _) =
        solve_pord_on_cai_scale(protein, &codon_usage, &options, Some(&note), cai_scale)
            .expect("sweep");
    (points, weights.into_inner())
}

/// A point by the bits of the two objectives it is a point of.
fn key(p: &ParetoPoint) -> (u64, u64) {
    (p.mfe_kcal_mol.to_bits(), p.cai.to_bits())
}

#[test]
fn the_cpp_scale_sweeps_fewer_weights_and_invents_no_point() {
    let (own_points, own) = sweep(SHORT, 1.0);
    let (cpp_points, cpp) = sweep(SHORT, CPP_CAI_SCALE);

    assert!(
        cpp.len() < own.len(),
        "C++'s thresholds on C++'s λ must stop sooner than on this solver's \
         ({} against {}), or this protein cannot tell the two apart",
        cpp.len(),
        own.len()
    );

    // Fewer weights can only mean fewer points, never different ones: each
    // weight is solved by this solver either way, so a point the C++ spacing
    // reaches is one the denser spacing reached too. A point outside that set
    // would mean the scale had changed an answer rather than a choice of
    // weights.
    let dense: Vec<_> = own_points.iter().map(key).collect();
    for p in &cpp_points {
        assert!(
            dense.contains(&key(p)),
            "a point from the C++ spacing is not one the denser sweep found"
        );
    }
}

#[test]
fn scale_one_is_the_sweep_it_always_was() {
    // The conversion is identity at scale 1, so the sweep every other caller
    // gets must be untouched, weights included.
    let (plain, plain_weights) = sweep(SHORT, 1.0);
    let (again, again_weights) = sweep(SHORT, 1.0);
    assert_eq!(plain_weights, again_weights);
    assert_eq!(
        plain.iter().map(key).collect::<Vec<_>>(),
        again.iter().map(key).collect::<Vec<_>>()
    );
}

#[test]
fn every_weight_stays_a_weight() {
    for scale in [1.0, CPP_CAI_SCALE] {
        let (_, weights) = sweep(SHORT, scale);
        for w in &weights {
            assert!(
                (0.0..=1.0).contains(w) && w.is_finite(),
                "scale {scale}: {w} is not a convex weight"
            );
        }
    }
}

/// The count this solver places against the count DERNA C++ places, at the
/// length both were recorded at.
///
/// Held out of the default run: the two sweeps it needs are ~8 minutes
/// together, against milliseconds for everything above. It is the only test
/// anchored to a figure from outside this crate, so it is the one to run when
/// the sweep's spacing changes:
///
/// ```text
/// cargo test --release -p derna-codon-dp --test sweep_cai_scale -- --ignored
/// ```
#[test]
#[ignore = "two full sweeps at 32 residues; run when the spacing changes"]
fn the_cpp_scale_sweeps_the_weights_cpp_sweeps() {
    let (_, own) = sweep(ANCHOR, 1.0);
    let (_, cpp) = sweep(ANCHOR, CPP_CAI_SCALE);
    assert_eq!(cpp.len(), 21, "weights on C++'s scale");
    assert_eq!(own.len(), 62, "weights on this solver's scale");
}
