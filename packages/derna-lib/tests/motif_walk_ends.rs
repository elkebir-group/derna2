//! The motif walk's two ends are the motif-free front's exact ends, not the
//! optima at its two outermost weights.
//!
//! The **stable end** is the least MFE among motif-free designs, and the
//! greatest adaptation among those reaching it; the **adapted end** is the
//! greatest adaptation among motif-free designs, and the least MFE among those
//! reaching it. Every case is checked against exhaustive enumeration: every
//! molecule the case admits is folded, the ones spelling a forbidden motif are
//! dropped, and the rest Pareto-reduced.
//!
//! The near-tied usage is the case a weight inside `(0, 1)` cannot reach: two
//! synonymous codons whose weights differ by less than the outer weight times
//! the energy they trade, so the optimum at that weight takes the less adapted
//! codon and the most adapted motif-free design lies beyond it. The test
//! asserts that the scalar optimum at the walk's outer weight misses the
//! adapted end on some case with flanks and on some case without, so neither
//! arrangement can pass vacuously. That optimum is the whole lattice's;
//! which cases put the walk's *own* end fold in question is the narrower
//! question `CASES` answers.
//!
//! A case carries the two flanks and the pins as well as the protein, because
//! the adapted end is folded over a lattice of its own — the most adapted
//! codons alone, with the flanks and the pins intersected in — and a flank
//! position a pin narrows is the same object as a designed flank position. A
//! pin that narrows a residue's codons also moves the lattice's adaptation
//! origin, which is the quantity the end is compared against, so cases with
//! and without one are both enumerated here.

use derna_codon_dp::cai::compute_cai_tables;
use derna_codon_dp::codon_table::{aa_to_index, get_codon_index, AMINO_ACIDS};
use derna_lib::ac::{expand_iupac_motif, MotifWindow};
use derna_lib::free_fold::{sequence_constrained_mfe, CdsWindow};
use derna_lib::motif_front::{motif_front_walk, MotifFrontSpec, MotifWalk};
use derna_lib::pins::Pins;
use derna_lib::{
    design_composite, BeamError, CodonUsage, CompositeSpec, Layout, Params, PositionalConstraints,
    RegionConstraints, ScanConfig, UtrProfile, UtrSpec,
};

/// The weight the walk's adapted end starts from, `2⁻²³`.
const EPSILON: f64 = 1.0 / (1u32 << 23) as f64;

/// Two log-adaptations within this are one trade-off, as the front's own
/// dominance reads them.
const SLOP: f64 = 1e-9;

/// (energy in dekacalories, coding log-adaptation) of a design.
type Point = (i64, f64);

/// A case: a protein, the motifs forbidden over the whole molecule, the two
/// flanks, and the nucleotides pinned over it.
struct Case {
    protein: &'static str,
    motifs: &'static [&'static str],
    /// The 5′ flank as an IUPAC string, one code per position: a nucleotide
    /// gives that position, and a degenerate code designs it over that code's
    /// nucleotides. Empty is no flank.
    utr5: &'static str,
    /// The 3′ flank, read as `utr5`.
    utr3: &'static str,
    /// Pinned positions as (whole-molecule position, the IUPAC code allowed
    /// there).
    pins: &'static [(u32, &'static str)],
}

/// The nucleotides an IUPAC code admits.
fn iupac(code: u8) -> &'static str {
    match code {
        b'A' => "A",
        b'C' => "C",
        b'G' => "G",
        b'U' | b'T' => "U",
        b'S' => "CG",
        b'W' => "AU",
        b'R' => "AG",
        b'Y' => "CU",
        b'M' => "AC",
        b'K' => "GU",
        b'N' => "ACGU",
        _ => panic!(
            "the case's flank spells {}, not an IUPAC code",
            code as char
        ),
    }
}

impl Case {
    /// Nucleotides in the whole molecule.
    fn len(&self) -> u32 {
        (self.utr5.len() + 3 * self.protein.len() + self.utr3.len()) as u32
    }

    /// Where the coding sequence starts.
    fn cds_start(&self) -> usize {
        self.utr5.len()
    }

    fn window(&self) -> MotifWindow {
        MotifWindow::overlapping(0, self.len())
    }

    fn layout(&self) -> Layout {
        Layout::new(
            self.utr5.len() as u32,
            self.protein.len() as u32,
            self.utr3.len() as u32,
        )
    }

    fn pins(&self) -> Option<Pins> {
        if self.pins.is_empty() {
            return None;
        }
        let mut pins = Pins::none(self.len());
        for &(at, letters) in self.pins {
            pins.pin(&self.layout(), at, letters)
                .expect("the pin lands");
        }
        Some(pins)
    }

    /// Whether either flank occupies any position.
    fn flanked(&self) -> bool {
        !self.utr5.is_empty() || !self.utr3.is_empty()
    }

    fn label(&self) -> String {
        format!(
            "{} {:?} 5′={:?} 3′={:?} pins={:?}",
            self.protein, self.motifs, self.utr5, self.utr3, self.pins
        )
    }

    /// Whether position `at` may spell `nucleotide` under the case's pins.
    fn pinned_to(pins: Option<&Pins>, at: usize, nucleotide: char) -> bool {
        pins.is_none_or(|p| {
            p.at(at as u32)
                .iter()
                .any(|n| b"ACGU"[n as usize] == nucleotide as u8)
        })
    }

    /// Every spelling of a flank, pins honoured.
    fn every_flank(&self, flank: &str, base: usize, pins: Option<&Pins>) -> Vec<String> {
        let mut out = vec![String::new()];
        for (i, &code) in flank.as_bytes().iter().enumerate() {
            let here: Vec<char> = iupac(code)
                .chars()
                .filter(|&c| Self::pinned_to(pins, base + i, c))
                .collect();
            out = out
                .iter()
                .flat_map(|d| here.iter().map(move |c| format!("{d}{c}")))
                .collect();
        }
        out
    }

    /// Every spelling of the coding sequence, pins honoured.
    fn every_cds(&self, pins: Option<&Pins>) -> Vec<String> {
        let base = self.cds_start();
        let mut out = vec![String::new()];
        for (residue, b) in self.protein.bytes().enumerate() {
            let a = aa_to_index(b as char).unwrap();
            let codons: Vec<String> = AMINO_ACIDS[a]
                .codons
                .iter()
                .map(|c| {
                    c.nuc
                        .iter()
                        .map(|&n| b"ACGU"[n as usize] as char)
                        .collect::<String>()
                })
                .filter(|codon| {
                    codon
                        .chars()
                        .enumerate()
                        .all(|(j, c)| Self::pinned_to(pins, base + 3 * residue + j, c))
                })
                .collect();
            out = out
                .iter()
                .flat_map(|d| codons.iter().map(move |c| format!("{d}{c}")))
                .collect();
        }
        out
    }

    /// Every molecule the case admits: the flanks' spellings crossed with the
    /// coding sequence's, pins honoured, motifs not yet applied.
    fn every_molecule(&self) -> Vec<String> {
        let pins = self.pins();
        let five = self.every_flank(self.utr5, 0, pins.as_ref());
        let cds = self.every_cds(pins.as_ref());
        let three = self.every_flank(
            self.utr3,
            self.cds_start() + 3 * self.protein.len(),
            pins.as_ref(),
        );
        let mut out = Vec::with_capacity(five.len() * cds.len() * three.len());
        for f in &five {
            for c in &cds {
                for t in &three {
                    out.push(format!("{f}{c}{t}"));
                }
            }
        }
        out
    }
}

/// Proteins rich in the near-tied families, with motifs that forbid some of
/// their most adapted codons or codon pairs. The first group is coding-only;
/// the second carries flanks, the arrangements being a designed 5′ flank, one
/// flank part given and part designed, two given flanks, a designed flank
/// under a pin, a pin that narrows a residue's codons, and a designed 3′
/// flank.
///
/// Most of the flanked cases carry a protein whose motifs forbid a *pair* of
/// residues from both spelling their most adapted codon, because that is what
/// makes the gap beyond the walk's most adapted vertex run rather than find
/// the end already reached. Two encodings then tie at the best adaptation,
/// differing in which of the pair gives way, so the lattice of the most
/// adapted codons alone — the one the adapted end is folded over — also
/// spells the design that takes the second codon at *both*, which is less
/// adapted and which the outer weight prefers wherever it is the more stable
/// by more than that weight prices the adaptation at. Where the pair is absent
/// the lattice spells nothing less adapted, and where it is present but the
/// less adapted design is no more stable the fold reaches the end by itself;
/// either way the case covers the arrangement without putting the end in
/// question, which is why the cases that do put it in question carry the pair.
const CASES: &[Case] = &[
    Case {
        protein: "MGGWGG",
        motifs: &["GGCGGC"],
        utr5: "",
        utr3: "",
        pins: &[],
    },
    Case {
        protein: "MGAPGRW",
        motifs: &["GCCC"],
        utr5: "",
        utr3: "",
        pins: &[],
    },
    Case {
        protein: "MPGAGPCW",
        motifs: &["CCCG"],
        utr5: "",
        utr3: "",
        pins: &[],
    },
    Case {
        protein: "MRGAPW",
        motifs: &["GGCG"],
        utr5: "",
        utr3: "",
        pins: &[],
    },
    Case {
        protein: "MCWFGGWCKM",
        motifs: &["GGCGGC"],
        utr5: "",
        utr3: "",
        pins: &[],
    },
    Case {
        protein: "MAAPPGW",
        motifs: &["GCCGCC", "CCCCCC"],
        utr5: "",
        utr3: "",
        pins: &[],
    },
    Case {
        protein: "MGPAGW",
        motifs: &["GGCC"],
        utr5: "",
        utr3: "",
        pins: &[],
    },
    Case {
        protein: "MAAPPW",
        motifs: &["GCCGCC", "CCCCCC"],
        utr5: "SS",
        utr3: "",
        pins: &[],
    },
    Case {
        protein: "MAAPPW",
        motifs: &["GCCGCC", "CCCCCC"],
        utr5: "GSC",
        utr3: "",
        pins: &[],
    },
    Case {
        protein: "MAAPPW",
        motifs: &["GCCGCC", "CCCCCC"],
        utr5: "GCC",
        utr3: "GGU",
        pins: &[],
    },
    Case {
        protein: "MAAPPW",
        motifs: &["GCCGCC", "CCCCCC"],
        utr5: "NN",
        utr3: "",
        pins: &[(0, "S")],
    },
    // Position 17 is the arginine's first base — two flank nucleotides and five
    // residues before it. Pinned to `A` the residue keeps `AGA` and `AGG`,
    // which drops its most adapted codon and so moves the lattice's adaptation
    // origin off zero.
    Case {
        protein: "MAAPPRW",
        motifs: &["GCCGCC", "CCCCCC"],
        utr5: "SS",
        utr3: "",
        pins: &[(17, "A")],
    },
    Case {
        protein: "MAAPPW",
        motifs: &["GCCGCC", "CCCCCC"],
        utr5: "",
        utr3: "SS",
        pins: &[],
    },
];

/// The human usage table with the glycine, alanine and proline families'
/// codons set to weights within `2e-8` of one another, distinct by more than
/// the front's dominance slop: a few kcal/mol of energy outweighs every
/// adaptation difference among them at `λ = 2⁻²³`.
fn near_tied_usage() -> CodonUsage {
    near_tied_usage_by([5e-9; 3])
}

/// The near-tied table with each family's adjacent codons `steps` apart in
/// relative weight, one step per family in the order glycine, alanine,
/// proline. Unequal steps give designs whose adaptations sit at several
/// distinct distances below the best, not at multiples of one quantum.
fn near_tied_usage_by(steps: [f64; 3]) -> CodonUsage {
    let base = include_str!("../../derna-codon-dp/data/codon_usage_human.csv");
    let tied = |aa: &str, codon: &str| -> Option<f64> {
        let (family, rank) = match (aa, codon) {
            ("G", "GGC") => (0, 0.0),
            ("A", "GCC") => (1, 0.0),
            ("P", "CCC") => (2, 0.0),
            ("G", "GGG") => (0, 1.0),
            ("A", "GCG") => (1, 1.0),
            ("P", "CCG") => (2, 1.0),
            ("G", "GGA") => (0, 2.0),
            ("A", "GCA") => (1, 2.0),
            ("P", "CCA") => (2, 2.0),
            ("G", "GGU") => (0, 3.0),
            ("A", "GCU") => (1, 3.0),
            ("P", "CCU") => (2, 3.0),
            _ => return None,
        };
        Some(0.25 * (1.0 - steps[family] * rank))
    };
    let csv: String = base
        .lines()
        .map(|line| {
            let parts: Vec<&str> = line.split(',').collect();
            match parts.as_slice() {
                [aa, codon, _] => match tied(aa, codon) {
                    Some(w) => format!("{aa},{codon},{w:.17}\n"),
                    None => format!("{line}\n"),
                },
                _ => format!("{line}\n"),
            }
        })
        .collect();
    CodonUsage::load_from_str(&csv).expect("the near-tied table parses")
}

/// The coding log-adaptation of `molecule`'s codons. A flank position earns
/// none, whether given or designed.
fn cai_log(case: &Case, usage: &CodonUsage, molecule: &str) -> f64 {
    let tables = compute_cai_tables(usage);
    let base = case.cds_start();
    case.protein
        .bytes()
        .enumerate()
        .map(|(i, b)| {
            let a = aa_to_index(b as char).unwrap();
            let codon: Vec<u8> = molecule.as_bytes()[base + 3 * i..base + 3 * i + 3]
                .iter()
                .map(|&c| b"ACGU".iter().position(|&x| x == c).unwrap() as u8)
                .collect();
            tables.log_cai[a][get_codon_index(a, [codon[0], codon[1], codon[2]]).unwrap()]
        })
        .sum()
}

/// Whether `molecule` spells a motif occurrence, read directly off the string.
fn spells_motif(case: &Case, molecule: &str) -> bool {
    case.motifs.iter().any(|m| {
        expand_iupac_motif(m)
            .unwrap()
            .iter()
            .any(|pattern| molecule.contains(pattern.as_str()))
    })
}

/// Every motif-free molecule's point, by folding each one.
fn motif_free_points(case: &Case, params: &Params, usage: &CodonUsage) -> Vec<Point> {
    let window = CdsWindow::new(case.cds_start(), case.protein.len());
    case.every_molecule()
        .into_iter()
        .filter(|molecule| !spells_motif(case, molecule))
        .map(|molecule| {
            let e =
                sequence_constrained_mfe(params, &molecule, window, RegionConstraints::default())
                    .unwrap();
            ((e * 100.0).round() as i64, cai_log(case, usage, &molecule))
        })
        .collect()
}

fn pareto(mut points: Vec<Point>) -> Vec<Point> {
    points.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
    let mut front: Vec<Point> = Vec::new();
    for p in points {
        if front.last().is_none_or(|q| p.1 > q.1 + SLOP) {
            front.push(p);
        }
    }
    front
}

/// The stable end: least energy, then greatest adaptation.
fn stable_end(bag: &[Point]) -> Point {
    let least = bag.iter().map(|p| p.0).min().unwrap();
    *bag.iter()
        .filter(|p| p.0 == least)
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
}

/// The adapted end: greatest adaptation (within the slop), then least energy.
fn adapted_end(bag: &[Point]) -> Point {
    let best = bag.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    *bag.iter()
        .filter(|p| p.1 >= best - SLOP)
        .min_by_key(|p| p.0)
        .unwrap()
}

fn walk(
    case: &Case,
    params: &Params,
    usage: &CodonUsage,
    threads: usize,
) -> Result<MotifWalk, BeamError> {
    let placed: Vec<(&str, MotifWindow)> =
        case.motifs.iter().map(|m| (*m, case.window())).collect();
    let pins = case.pins();
    let spec = MotifFrontSpec {
        utr5: case.utr5,
        utr3: case.utr3,
        constraints: RegionConstraints::default(),
        motifs: &placed,
        pins: pins.as_ref(),
    };
    motif_front_walk(params, case.protein, usage, &spec, threads)
}

/// The coding log-adaptation of the scalar optimum at the walk's outer weight,
/// over the same molecules the walk searches.
fn outer_weight_cai(case: &Case, params: &Params, usage: &CodonUsage) -> f64 {
    let motifs: Vec<String> = case.motifs.iter().map(|m| (*m).to_string()).collect();
    let (five, three) = (
        UtrSpec::Mixed(case.utr5.to_string()),
        UtrSpec::Mixed(case.utr3.to_string()),
    );
    let mut positional = PositionalConstraints::none(case.len());
    if let Some(pins) = case.pins() {
        positional.pins = pins;
    }
    let spec = CompositeSpec {
        utr5: &five,
        utr3: &three,
        constraints: RegionConstraints::default(),
        profile: UtrProfile::inert(),
        motifs: &motifs,
        motif_window: case.window(),
        flank_complement: None,
        complement_classes: None,
        positional: &positional,
    };
    design_composite(
        params,
        case.protein,
        usage,
        EPSILON,
        &spec,
        ScanConfig::default(),
    )
    .expect("the optimum at the outer weight")
    .cai_log
}

/// What is wrong with `walked` against the enumerated motif-free `bag`, if
/// anything: the whole front, then each end and the hull's end vertices.
fn mismatch(walked: &MotifWalk, bag: &[Point]) -> Option<String> {
    let got: Vec<Point> = walked
        .front
        .iter()
        .map(|p| ((p.mfe_kcal_mol * 100.0).round() as i64, p.cai_log))
        .collect();
    let truth = pareto(bag.to_vec());
    let same = got.len() == truth.len()
        && got
            .iter()
            .zip(&truth)
            .all(|(x, y)| x.0 == y.0 && (x.1 - y.1).abs() < 1e-6);
    if !same {
        return Some(format!("front\n got   {got:?}\n truth {truth:?}"));
    }
    let (stable, adapted) = (stable_end(bag), adapted_end(bag));
    let (first, last) = (got[0], got[got.len() - 1]);
    if first.0 != stable.0 || (first.1 - stable.1).abs() > SLOP {
        return Some(format!("stable end {first:?} against {stable:?}"));
    }
    if last.0 != adapted.0 || (last.1 - adapted.1).abs() > SLOP {
        return Some(format!("adapted end {last:?} against {adapted:?}"));
    }
    let (lo, hi) = (
        walked.vertices[0],
        walked.vertices[walked.vertices.len() - 1],
    );
    // The walk reports a vertex's adaptation in the front's own units, adding
    // back what a pin's restriction removed from the lattice's, so a vertex is
    // read as it stands. The stable vertex and the front's first point need
    // not be one design: two codons whose adaptations differ by less than the
    // stable fold resolves leave the vertex on the less adapted spelling, so
    // their difference is not an origin. A hull of one vertex has no adapted
    // vertex of its own: that vertex is the stable one, and the adapted end is
    // the front's last point, checked above.
    if i64::from(lo.mfe.0) != stable.0 {
        return Some(format!(
            "the hull's stable vertex {lo:?} against {stable:?}"
        ));
    }
    let distinct = walked.vertices.len() > 1;
    if i64::from(hi.mfe.0) != adapted.0 || (distinct && (hi.cai - adapted.1).abs() > SLOP) {
        return Some(format!(
            "the hull's adapted vertex {hi:?} against {adapted:?}"
        ));
    }
    None
}

/// A point of the emitted front that another emitted point dominates, if any:
/// at most as many dekacalories and at least the adaptation, within the slop.
fn dominated(walked: &MotifWalk) -> Option<String> {
    let pts: Vec<Point> = walked
        .front
        .iter()
        .map(|p| ((p.mfe_kcal_mol * 100.0).round() as i64, p.cai_log))
        .collect();
    for (i, p) in pts.iter().enumerate() {
        for (j, q) in pts.iter().enumerate() {
            if i != j && q.0 <= p.0 && q.1 >= p.1 - SLOP && (q.0 < p.0 || q.1 > p.1) {
                return Some(format!("emitted {p:?} is dominated by emitted {q:?}"));
            }
        }
    }
    None
}

/// Every case's walk against its enumerated front under `usage`, failing
/// with every case that misses rather than the first.
fn check_all(params: &Params, usage: &CodonUsage, tag: &str) {
    check_cases(CASES, params, usage, tag, 2);
}

/// [`check_all`] over `cases`, with every emitted front also free of a point
/// another emitted point dominates. `threads` is the walk's; a case that may
/// panic walks on one, because a panic in a worker of a scoped walk hangs it.
fn check_cases(cases: &[Case], params: &Params, usage: &CodonUsage, tag: &str, threads: usize) {
    let mut failures = Vec::new();
    for case in cases {
        let bag = motif_free_points(case, params, usage);
        assert!(!bag.is_empty(), "{}: no motif-free design", case.label());
        let walked = walk(case, params, usage, threads).expect("a feasible case walks");
        if let Some(why) = mismatch(&walked, &bag).or_else(|| dominated(&walked)) {
            failures.push(format!("{} {tag}: {why}", case.label()));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Under the human usage table the motif walk returns the whole motif-free
/// front, both exact ends included.
#[test]
fn the_motif_walk_reaches_both_exact_ends() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    check_all(&params, &usage, "human usage");
}

/// Under near-tied codons the motif-free optimum at `2⁻²³` stops short of the
/// most adapted motif-free design, and the walk still reaches it.
#[test]
fn the_motif_walk_reaches_an_adapted_end_its_outer_weight_misses() {
    let params = Params::load_default().unwrap();
    let usage = near_tied_usage();
    let mut missed = Vec::new();
    for case in CASES {
        let bag = motif_free_points(case, &params, &usage);
        if outer_weight_cai(case, &params, &usage) < adapted_end(&bag).1 - SLOP {
            missed.push(case);
        }
    }
    let labels: Vec<String> = missed.iter().map(|c| c.label()).collect();
    eprintln!("outer weight misses the adapted end on: {labels:?}");
    assert!(
        missed.iter().any(|c| !c.flanked()),
        "no coding-only case's optimum at the outer weight missed the adapted \
         end, so the end was never in question there"
    );
    assert!(
        missed.iter().any(|c| c.flanked()),
        "no flanked case's optimum at the outer weight missed the adapted end, \
         so the end was never in question with a flank"
    );
    check_all(&params, &usage, "near-tied usage");
}

/// The human usage table with the alanine, glycine and proline codons set to
/// `0.25` less the given offsets, each a few parts in `10⁹`: inside the walk's
/// former adapted-end tolerance, outside the documented slop. A codon absent
/// from `offsets` keeps `0.25`.
fn offset_usage(offsets: &[(&str, f64)]) -> CodonUsage {
    let base = include_str!("../../derna-codon-dp/data/codon_usage_human.csv");
    let csv: String = base
        .lines()
        .map(|line| {
            let parts: Vec<&str> = line.split(',').collect();
            match parts.as_slice() {
                [aa, codon, _] if matches!(*aa, "A" | "G" | "P") => {
                    let off = offsets
                        .iter()
                        .find(|(c, _)| c == codon)
                        .map_or(0.0, |&(_, o)| o);
                    format!("{aa},{codon},{:.17}\n", 0.25 - off)
                }
                _ => format!("{line}\n"),
            }
        })
        .collect();
    CodonUsage::load_from_str(&csv).expect("the offset table parses")
}

/// A tag, a case, and the codon weight offsets of its usage table.
type NarrowEnd = (&'static str, Case, &'static [(&'static str, f64)]);

/// Cases where the narrowed adapted-end fold, at weight `2⁻²³`, ends a hair
/// short of the most adapted design: by more than the slop, less than a looser
/// tolerance would accept. Chosen by measuring that accepting that end moves
/// the answer. The first case then panics in the grid's bracketing (a negative
/// tie weight from a dominated end); the second then emits a point that
/// another emitted point dominates.
const NARROW_END_CASES: &[NarrowEnd] = &[
    (
        "perturbed codons",
        Case {
            protein: "MAAPPGW",
            motifs: &["GCCGCC", "CCCCCC"],
            utr5: "",
            utr3: "",
            pins: &[],
        },
        &[
            ("GCA", 2e-9),
            ("GCG", 1e-9),
            ("GGG", 1e-9),
            ("CCU", 2e-9),
            ("CCA", 2e-9),
            ("CCG", 2e-9),
        ],
    ),
    (
        "stepped codons",
        Case {
            protein: "PPPPAA",
            motifs: &["CCCCCC"],
            utr5: "",
            utr3: "",
            pins: &[],
        },
        &[
            ("GCU", 3.75e-9),
            ("GCA", 2.5e-9),
            ("GCG", 1.25e-9),
            ("GGU", 3.75e-9),
            ("GGA", 2.5e-9),
            ("GGG", 1.25e-9),
            ("CCU", 3.75e-9),
            ("CCA", 2.5e-9),
            ("CCG", 1.25e-9),
        ],
    ),
];

/// One of [`NARROW_END_CASES`]: the walk's adapted end is the most adapted
/// motif-free design, never a narrowed fold's end a hair less adapted, and the
/// emitted front has no dominated point.
fn narrow_end_case(which: usize) {
    let params = Params::load_default().unwrap();
    let (tag, case, offsets) = &NARROW_END_CASES[which];
    let usage = offset_usage(offsets);
    check_cases(std::slice::from_ref(case), &params, &usage, tag, 1);
}

/// The case whose narrowed end is dominated by a more stable, more adapted
/// vertex.
#[test]
fn a_dominated_narrowed_end_does_not_give_a_negative_tie_weight() {
    narrow_end_case(0);
}

/// The case whose narrowed end ties the best adaptation and is less stable.
#[test]
fn a_less_stable_narrowed_end_is_not_emitted_beside_its_dominator() {
    narrow_end_case(1);
}

/// The gap from the walk's last hull vertex out to the adapted end holds every
/// front point between the two. Adaptation differences of `5e-9`, `2e-9` and
/// `3e-9` in the three families give designs at several distances below the
/// best, so a front point lies strictly between the last vertex and the end: a
/// gap cut off at the end's energy, or one that kept only its last point,
/// drops it.
#[test]
fn the_gap_to_the_adapted_end_keeps_the_front_points_inside_it() {
    let params = Params::load_default().unwrap();
    let usage = near_tied_usage_by([5e-9, 2e-9, 3e-9]);
    let case = Case {
        protein: "MAAPPGW",
        motifs: &["GCCGCC", "CCCCCC"],
        utr5: "",
        utr3: "",
        pins: &[],
    };
    let bag = motif_free_points(&case, &params, &usage);
    let walked = walk(&case, &params, &usage, 2).expect("a feasible case walks");
    assert!(
        walked.adapted_gap,
        "the walk's last vertex already was the adapted end, so no gap ran"
    );
    let k = walked.vertices.len();
    let (last, end) = (walked.vertices[k - 2], walked.vertices[k - 1]);
    let inside = walked
        .front
        .iter()
        .filter(|p| {
            let e = (p.mfe_kcal_mol * 100.0).round() as i32;
            last.mfe.0 < e && e < end.mfe.0
        })
        .count();
    assert!(
        inside >= 1,
        "no front point lies between the last vertex and the end, so a gap that dropped its interior would pass"
    );
    assert_eq!(mismatch(&walked, &bag), None);
}

/// The hull's stable vertex and the front's first point are two designs when
/// two codons' adaptations differ by less than the stable fold resolves: the
/// vertex sits on the spelling `2e-9` less adapted than the point, while the
/// adapted vertex is the front's last point exactly. Reading the adapted
/// vertex against an origin taken from that pair moves it by the `2e-9`, which
/// is over the slop, so the oracle must read it as it stands. Chosen by
/// measuring that: the vertex and the point differ by more than the slop here,
/// and the front equals enumeration, so only the adapted-vertex check could
/// fail.
#[test]
fn the_adapted_vertex_is_read_as_reported_when_the_stable_vertex_is_another_design() {
    let params = Params::load_default().unwrap();
    let usage = near_tied_usage_by([3e-9, 7e-9, 2e-9]);
    let case = Case {
        protein: "MGGPGGW",
        motifs: &["CCCGCC", "GCCGCC"],
        utr5: "",
        utr3: "",
        pins: &[],
    };
    let bag = motif_free_points(&case, &params, &usage);
    let walked = walk(&case, &params, &usage, 2).expect("a feasible case walks");
    let (first, lo) = (walked.front[0].cai_log, walked.vertices[0].cai);
    assert!(
        (first - lo).abs() > SLOP,
        "the stable vertex and the front's first point are one design, so the \
         case does not separate the two readings"
    );
    assert_eq!(mismatch(&walked, &bag), None);
}
