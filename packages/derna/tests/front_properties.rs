//! What a front must satisfy whatever the molecule is.
//!
//! A front is defined by these properties, so they are worth asserting
//! directly rather than only through an equality against the walk: a front
//! equal to the walk's would still be wrong if the walk's were.
//!
//! - the points ascend in energy and in adaptation together, and no point
//!   dominates another — that is what "Pareto" means here;
//! - every point encodes the requested protein between the fixed flanks;
//! - the points marked `on_hull` are the vertices of a **convex** lower
//!   boundary: walking them, the slope between consecutive vertices turns one
//!   way only. A hull that is not convex is not a hull, and `on_hull` is what
//!   tells a caller which points a weight sweep can reach;
//! - the front's ends are vertices, since each is optimal at a weight.
//!
//! The protein is short on purpose: a front is a solve, and these properties
//! do not become truer at length.

use derna_cli::front;
use derna_cli::molecule::Molecule;
use std::collections::HashMap;

const PROTEIN: &str = "MKVLAKGIV";
const UTR5: &str = "ACAUUUGCUU";
const UTR3: &str = "GGCCAAUUGG";

fn words(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

fn balanced(structure: &str) -> bool {
    let mut open = 0i64;
    for c in structure.chars() {
        match c {
            '(' => open += 1,
            ')' => open -= 1,
            '.' => {}
            _ => return false,
        }
        if open < 0 {
            return false;
        }
    }
    open == 0
}

#[test]
fn a_front_satisfies_its_own_definition() {
    let argv = words(&["--protein", PROTEIN, "--utr5", UTR5, "--utr3", UTR3]);
    let args = front::parse(argv).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let walked = front::walk(&m, 1).unwrap();
    let hull = front::on_hull(&walked);
    assert!(
        walked.front.len() >= 2,
        "a front of one point tests nothing"
    );

    let codon_of: HashMap<String, char> = m
        .codons
        .rows
        .iter()
        .map(|(aa, codon, _)| (codon.clone(), aa.chars().next().unwrap()))
        .collect();

    // Ordered, and no point dominated by another.
    for pair in walked.front.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        assert!(
            b.mfe_kcal_mol >= a.mfe_kcal_mol - 1e-9,
            "the front is not energy-ascending: {} then {}",
            a.mfe_kcal_mol,
            b.mfe_kcal_mol
        );
        assert!(
            b.cai_log > a.cai_log - 1e-9,
            "a point is dominated: {} at {} then {} at {}",
            a.cai_log,
            a.mfe_kcal_mol,
            b.cai_log,
            b.mfe_kcal_mol
        );
    }

    // Every point is a molecule encoding the protein between the flanks, and
    // ending the coding sequence in a stop codon: the flanks are given and no
    // `--stop` is, so the search designs one as the coding sequence's last
    // slot.
    for point in &walked.front {
        assert!(point.sequence.starts_with(UTR5) && point.sequence.ends_with(UTR3));
        assert_eq!(
            point.sequence.len(),
            UTR5.len() + 3 * PROTEIN.len() + 3 + UTR3.len()
        );
        let stop_at = UTR5.len() + 3 * PROTEIN.len();
        let stop = &point.sequence[stop_at..stop_at + 3];
        assert!(
            ["UAA", "UAG", "UGA"].contains(&stop),
            "a front point ends its coding sequence in {stop}"
        );
        assert!(balanced(&point.structure), "{}", point.structure);
        let cds = &point.sequence[UTR5.len()..UTR5.len() + 3 * PROTEIN.len()];
        let read: String = (0..PROTEIN.len())
            .map(|i| codon_of[&cds[3 * i..3 * i + 3]])
            .collect();
        assert_eq!(read, PROTEIN, "a front point does not encode the protein");
    }

    // The ends are vertices: each is optimal at a weight.
    assert!(hull[0], "the front's first point is not on the hull");
    assert!(
        *hull.last().unwrap(),
        "the front's last point is not on the hull"
    );

    // The vertices form a convex boundary: the slope between consecutive ones
    // turns one way only. The objective is `λ·MFE − (1−λ)·adaptation`, so the
    // hull is the lower boundary in (energy, adaptation) and its slopes
    // decrease.
    let vertices: Vec<_> = walked
        .front
        .iter()
        .zip(&hull)
        .filter_map(|(p, &on)| on.then_some((p.mfe_kcal_mol, p.cai_log + p.utr_profile_log)))
        .collect();
    assert!(
        vertices.len() >= 2,
        "fewer than two vertices to be convex about"
    );
    let slopes: Vec<f64> = vertices
        .windows(2)
        .map(|w| {
            let ((e0, a0), (e1, a1)) = (w[0], w[1]);
            assert!(e1 > e0, "two vertices at the same energy: {e0} and {e1}");
            (a1 - a0) / (e1 - e0)
        })
        .collect();
    for pair in slopes.windows(2) {
        assert!(
            pair[1] <= pair[0] + 1e-6,
            "the hull is not convex: slope {} then {}",
            pair[0],
            pair[1]
        );
    }
}

#[test]
fn the_accessibility_window_reaches_the_walk() {
    // `--access` has no counterpart on the instrument, so no equality covers
    // it: the only way it is known to do anything on a front is a measurement
    // like this one. Forbidding every pair with an endpoint around the start
    // codon has to change the front — if it did not, the flag would be
    // accepted and inert, which is the failure this branch refuses elsewhere.
    let walk_with = |extra: &[&str]| {
        let mut argv = words(&["--protein", PROTEIN, "--utr5", UTR5, "--utr3", UTR3]);
        argv.extend(words(extra));
        let args = front::parse(argv).unwrap();
        let m = Molecule::resolve(&args.shared).unwrap();
        assert_eq!(
            m.constraint.access.is_some(),
            extra.contains(&"--access"),
            "the window did not reach the resolved constraint"
        );
        let walked = front::walk(&m, 1).unwrap();
        walked
            .front
            .iter()
            .map(|p| {
                (
                    format!("{:.2}", p.mfe_kcal_mol),
                    format!("{:.6}", p.cai_log),
                )
            })
            .collect::<Vec<_>>()
    };

    let free = walk_with(&[]);
    let windowed = walk_with(&["--access", "10,10"]);
    assert!(
        free != windowed,
        "a 10,10 window around the start codon changed neither point nor objective, \
         so the flag is inert on the front"
    );
    // And it can only cost: the window forbids pairs, so no energy improves.
    let best = |front: &[(String, String)]| {
        front
            .iter()
            .map(|(mfe, _)| mfe.parse::<f64>().unwrap())
            .fold(f64::MAX, f64::min)
    };
    assert!(
        best(&windowed) >= best(&free) - 1e-9,
        "the windowed front reached a better energy than the free one, which forbidding \
         pairs cannot do"
    );
}

#[test]
fn an_off_hull_point_lies_above_the_hull_it_is_off() {
    // `on_hull` is a claim about position, not a label: a point that is not a
    // vertex must sit strictly inside the hull's chord at its own energy —
    // worse-adapted than the boundary. That is also why the front is more than
    // a weight sweep: these are the points no λ can reach.
    //
    // The constraint is written out rather than left to the default: this
    // test needs a front with off-hull points on it at all, which is a
    // property of how tightly the molecule is constrained, and a default is
    // free to change. The count assertion at the end is what says so.
    let argv = words(&[
        "--protein",
        PROTEIN,
        "--utr5",
        UTR5,
        "--utr3",
        UTR3,
        "--forbid",
        "5-cds,cds-3,5-3",
    ]);
    let args = front::parse(argv).unwrap();
    let m = Molecule::resolve(&args.shared).unwrap();
    let walked = front::walk(&m, 1).unwrap();
    let hull = front::on_hull(&walked);

    let vertices: Vec<(f64, f64)> = walked
        .front
        .iter()
        .zip(&hull)
        .filter_map(|(p, &on)| on.then_some((p.mfe_kcal_mol, p.cai_log + p.utr_profile_log)))
        .collect();
    let mut off = 0;
    for (point, &on) in walked.front.iter().zip(&hull) {
        if on {
            continue;
        }
        off += 1;
        let (e, a) = (point.mfe_kcal_mol, point.cai_log + point.utr_profile_log);
        // The chord of the two vertices bracketing this energy.
        let after = vertices.iter().position(|&(ve, _)| ve > e);
        let Some(i) = after.filter(|&i| i > 0) else {
            panic!("an off-hull point at {e} lies outside the vertices' range");
        };
        let ((e0, a0), (e1, a1)) = (vertices[i - 1], vertices[i]);
        let chord = a0 + (a1 - a0) * (e - e0) / (e1 - e0);
        assert!(
            a <= chord + 1e-6,
            "an off-hull point at energy {e} is above the hull's chord: {a} > {chord}"
        );
    }
    assert!(
        off > 0,
        "every point is a vertex here, so the off-hull claim is untested — \
         a longer protein or a constraint would give the gap folds something to find"
    );
}
