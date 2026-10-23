//! Folding a sequence from another's tables, recomputing only the cells that
//! span the change, gives exactly the fold from scratch.

use rna_mfe::{fold, fold_from, fold_tables, Params};

/// A deterministic pseudo-random ACGU sequence.
fn sequence(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            b"ACGU"[(x >> 62) as usize]
        })
        .collect()
}

/// Every symbol in `lo..=hi` replaced by a different one.
fn changed(seq: &[u8], lo: usize, hi: usize) -> Vec<u8> {
    let mut out = seq.to_vec();
    for s in &mut out[lo..=hi] {
        *s = match *s {
            b'A' => b'G',
            b'C' => b'U',
            b'G' => b'C',
            _ => b'A',
        };
    }
    out
}

#[test]
fn a_fold_from_base_tables_equals_the_fold_from_scratch() {
    let params = Params::load_default().expect("parameters");
    for (n, seed) in [(40, 1), (150, 2), (320, 3)] {
        let base = sequence(n, seed);
        let (_, _, tables) = fold_tables(&params, &base).expect("base fold");
        for (lo, hi) in [
            (0, 2),
            (3, 5),
            (n / 2, n / 2 + 2),
            (n - 6, n - 4),
            (n - 3, n - 1),
            (n - 1, n - 1),
        ] {
            let seq = changed(&base, lo, hi);
            let fresh = fold(&params, &seq).expect("fold");
            let reused = fold_from(&params, &tables, &seq, lo, hi).expect("fold from");
            assert_eq!(reused, fresh, "n = {n}, change {lo}..={hi}");
        }
    }
}

/// The non-vacuity control: telling `fold_from` the wrong span leaves stale
/// cells, so the fold it returns differs from the fold from scratch.
///
/// Release-only, because a stale table is also an inconsistent one: the
/// traceback's `debug_assert!`s hold for tables the recurrence filled itself,
/// and this test deliberately hands it tables that no sequence produced, so one
/// of them fires before the fold is returned. Under `--release` those
/// assertions are compiled out, the fold completes, and the control asserts
/// what it is for. Gating it is what keeps a debug `cargo test --workspace`
/// from reporting a failure that says nothing about the code under test.
#[cfg(not(debug_assertions))]
#[test]
fn a_span_that_misses_the_change_gives_a_different_fold() {
    let params = Params::load_default().expect("parameters");
    let base = sequence(150, 7);
    let (_, _, tables) = fold_tables(&params, &base).expect("base fold");
    let seq = changed(&base, 0, 20);
    let fresh = fold(&params, &seq).expect("fold");
    let wrong = fold_from(&params, &tables, &seq, 140, 149).expect("fold from");
    assert_ne!(wrong, fresh);
}
