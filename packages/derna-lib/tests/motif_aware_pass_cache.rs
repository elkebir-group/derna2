//! The motif walk's pass cache (`DERNA_MOTIF_PASS_CACHE`) keeps the
//! constrained-lattice (aware) passes beside the plain ones. A walk over a
//! cold cache, over a warm one and without one returns the same front; the
//! warm walk reads every aware pass it needs from disk; and a damaged aware
//! file is taken again rather than read.
//!
//! The read counts are what this test rests on, not the front comparison:
//! an aware pass read at the wrong weight only loosens or tightens the floors
//! an exact walk prunes on, and on these cells a walk reading each weight's
//! neighbour's pass returned the same front.
//!
//! The variable is process-wide, so these walks live in their own test binary
//! and run one after the other inside a single test.

use derna_lib::ac::MotifWindow;
use derna_lib::free_fold::CdsWindow;
use derna_lib::motif_front::{motif_front_walk_run, MotifFrontSpec, MotifRun, MotifWalk};
use derna_lib::{CodonUsage, Enforcement, Params, RegionConstraints};

const CACHE: &str = "DERNA_MOTIF_PASS_CACHE";

/// A cell whose motif set removes front points, and one with flanks and a
/// region constraint.
fn cells() -> Vec<(
    &'static str,
    &'static str,
    &'static str,
    RegionConstraints,
    &'static [&'static str],
)> {
    vec![
        ("MRLGSA", "", "", RegionConstraints::default(), &["GGC"]),
        (
            "MRLGSA",
            "GGCAUCCAC",
            "GUGGAUGCC",
            RegionConstraints::symmetric_isolation(Enforcement::Forbid),
            &["CCA", "UCC"],
        ),
    ]
}

/// Everything a walk returns that a cache could change, as text.
fn outcome(w: &MotifWalk) -> String {
    format!(
        "{:?} {:?} {:?} {:?}",
        w.front, w.vertices, w.open, w.stopped_by
    )
}

/// Passes with a constrained-lattice half, and of those the ones read from
/// the cache.
fn aware_counts(w: &MotifWalk) -> (usize, usize) {
    let aware: Vec<_> = w.passes.iter().filter(|p| p.aware_folds > 0).collect();
    (aware.len(), aware.iter().filter(|p| p.aware_read).count())
}

fn aware_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "aware"))
        .collect();
    out.sort();
    out
}

#[test]
fn aware_passes_read_from_the_cache_give_the_same_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("motif-aware-pass-cache-{}", std::process::id()));
    for (i, (protein, utr5, utr3, constraints, motifs)) in cells().into_iter().enumerate() {
        let cds = CdsWindow::new(utr5.len(), protein.len());
        let window = MotifWindow::overlapping(cds.start as u32, cds.end as u32);
        let motifs: Vec<_> = motifs.iter().map(|&m| (m, window)).collect();
        let spec = MotifFrontSpec {
            utr5,
            utr3,
            constraints,
            motifs: &motifs,
            pins: None,
        };
        let walk = || {
            motif_front_walk_run(&params, protein, &usage, &spec, 1, MotifRun::default())
                .expect("a feasible cell")
        };
        let dir = root.join(i.to_string());
        std::fs::create_dir_all(&dir).unwrap();

        std::env::remove_var(CACHE);
        let plain = walk();
        let (aware, _) = aware_counts(&plain);
        assert!(aware > 0, "{protein}: the walk takes no aware pass");

        std::env::set_var(CACHE, &dir);
        let cold = walk();
        let files = aware_files(&dir);
        assert_eq!(aware_counts(&cold), (aware, 0), "{protein}: cold");
        assert_eq!(files.len(), aware, "{protein}: one file per aware pass");

        let warm = walk();
        let (warm_aware, read) = aware_counts(&warm);
        eprintln!(
            "{protein}: {aware} aware passes, {} files, warm read {read} of {warm_aware}, sizes {:?}",
            files.len(),
            files
                .iter()
                .map(|f| std::fs::metadata(f).unwrap().len())
                .collect::<Vec<_>>()
        );
        // Plain passes read from the cache are not taken several to a
        // traversal, so a warm walk can need fewer passes than a cold one.
        assert!(
            warm_aware > 0,
            "{protein}: the warm walk takes no aware pass"
        );
        assert_eq!(read, warm_aware, "{protein}: warm reads");

        // Every file truncated or with a flipped byte: each is taken again.
        for (n, f) in files.iter().enumerate() {
            let mut bytes = std::fs::read(f).unwrap();
            if n % 2 == 0 {
                bytes.truncate(bytes.len() / 2);
            } else {
                let mid = bytes.len() / 2;
                bytes[mid] ^= 0x10;
            }
            std::fs::write(f, &bytes).unwrap();
        }
        let damaged = walk();
        let (damaged_aware, damaged_read) = aware_counts(&damaged);
        assert!(
            damaged_aware > 0,
            "{protein}: the damaged walk takes no aware pass"
        );
        assert_eq!(damaged_read, 0, "{protein}: a damaged file was read");
        std::env::remove_var(CACHE);

        for (name, w) in [("cold", &cold), ("warm", &warm), ("damaged", &damaged)] {
            assert_eq!(outcome(w), outcome(&plain), "{protein}: {name}");
        }
        for f in std::fs::read_dir(&dir).unwrap() {
            std::fs::remove_file(f.unwrap().path()).unwrap();
        }
        std::fs::remove_dir(&dir).unwrap();
    }
    let _ = std::fs::remove_dir(&root);
}
