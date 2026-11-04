//! The modular front over motif-free designs against exhaustive
//! enumeration.
//!
//! On molecules small enough to fold every design, the front is the Pareto set
//! of (constrained energy, coding log-adaptation) over the designs that spell
//! no forbidden motif overlapping the coding sequence and whose free MFE
//! equals their constrained energy. A complete run from the motif-free
//! constrained front must return exactly that set; a run its budget stopped
//! must certify only points of it, and every point of it it does not return
//! must sit in a region it reported open. A pinned cell enumerates only the
//! designs spelling its pins.

use derna_codon_dp::cai::compute_cai_tables;
use derna_codon_dp::codon_table::{aa_to_index, get_codon_index, AMINO_ACIDS};
use derna_lib::ac::{expand_iupac_motif, MotifWindow};
use derna_lib::coords::Layout;
use derna_lib::free_fold::{sequence_constrained_mfe, sequence_gap, CdsWindow};
use derna_lib::modular_front::{modular_motif_front, ModularFront, OpenBox};
use derna_lib::motif_front::{motif_front_walk_run, MotifFrontSpec, MotifRun};
use derna_lib::pins::Pins;
use derna_lib::{CodonUsage, Enforcement, Params, RegionConstraints};

/// A cell: protein, flanks, constraint and motif set, forbidden over the
/// coding sequence by the overlapping reading, and pins as (whole-molecule
/// position, nucleotides spelled from it).
struct Cell {
    protein: &'static str,
    utr5: &'static str,
    utr3: &'static str,
    constraints: RegionConstraints,
    motifs: &'static [&'static str],
    pins: &'static [(u32, &'static str)],
}

impl Cell {
    fn cds(&self) -> CdsWindow {
        CdsWindow::new(self.utr5.len(), self.protein.len())
    }

    fn window(&self) -> MotifWindow {
        let cds = self.cds();
        MotifWindow::overlapping(cds.start as u32, cds.end as u32)
    }

    /// Every motif with the window it is forbidden inside, as a spec takes it:
    /// here one shared window, the whole coding sequence.
    fn placed(&self) -> Vec<(&'static str, MotifWindow)> {
        self.motifs.iter().map(|m| (*m, self.window())).collect()
    }

    fn pins(&self) -> Option<Pins> {
        if self.pins.is_empty() {
            return None;
        }
        let layout = Layout::new(
            self.utr5.len() as u32,
            self.protein.len() as u32,
            self.utr3.len() as u32,
        );
        let mut pins = Pins::none(layout.len());
        for &(at, letters) in self.pins {
            pins.pin(&layout, at, letters).unwrap();
        }
        Some(pins)
    }

    /// Whether `whole` spells every pin, read directly off the string.
    fn spells_pins(&self, whole: &str) -> bool {
        self.pins
            .iter()
            .all(|&(at, letters)| whole[at as usize..].starts_with(letters))
    }
}

/// (energy in dekacalories, coding log-adaptation) of a design.
type Point = (i64, f64);

fn every_design(cell: &Cell) -> Vec<String> {
    let mut designs = vec![String::new()];
    for b in cell.protein.bytes() {
        let a = aa_to_index(b as char).unwrap();
        let codons: Vec<String> = AMINO_ACIDS[a]
            .codons
            .iter()
            .map(|c| c.nuc.iter().map(|&n| b"ACGU"[n as usize] as char).collect())
            .collect();
        designs = designs
            .iter()
            .flat_map(|d| codons.iter().map(move |c| format!("{d}{c}")))
            .collect();
    }
    designs
}

fn cai_log(cell: &Cell, usage: &CodonUsage, cds: &str) -> f64 {
    let tables = compute_cai_tables(usage);
    cell.protein
        .bytes()
        .enumerate()
        .map(|(i, b)| {
            let a = aa_to_index(b as char).unwrap();
            let codon: Vec<u8> = cds.as_bytes()[3 * i..3 * i + 3]
                .iter()
                .map(|&c| b"ACGU".iter().position(|&x| x == c).unwrap() as u8)
                .collect();
            tables.log_cai[a][get_codon_index(a, [codon[0], codon[1], codon[2]]).unwrap()]
        })
        .sum()
}

/// Whether `whole` spells a motif occurrence overlapping the coding sequence,
/// read directly off the string.
fn spells_motif(cell: &Cell, whole: &str) -> bool {
    let cds = cell.cds();
    cell.motifs.iter().any(|m| {
        expand_iupac_motif(m).unwrap().iter().any(|pattern| {
            (0..=whole.len().saturating_sub(pattern.len())).any(|s| {
                whole[s..].starts_with(pattern.as_str())
                    && s < cds.end
                    && s + pattern.len() > cds.start
            })
        })
    })
}

fn pareto(mut points: Vec<Point>) -> Vec<Point> {
    points.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
    let mut front: Vec<Point> = Vec::new();
    for p in points {
        if front.last().is_none_or(|q| p.1 > q.1 + 1e-9) {
            front.push(p);
        }
    }
    front
}

/// The modular front by folding every design, over every design
/// (`motif_free = false`) or the motif-free ones.
fn exhaustive(cell: &Cell, params: &Params, usage: &CodonUsage, motif_free: bool) -> Vec<Point> {
    let cds = cell.cds();
    let optimal: Vec<Point> = every_design(cell)
        .into_iter()
        .filter_map(|c| {
            let whole = format!("{}{c}{}", cell.utr5, cell.utr3);
            if !cell.spells_pins(&whole) || motif_free && spells_motif(cell, &whole) {
                return None;
            }
            let gap = sequence_gap(params, &whole, cds, cell.constraints).unwrap();
            gap.is_modular().then(|| {
                let e = sequence_constrained_mfe(params, &whole, cds, cell.constraints).unwrap();
                ((e * 100.0).round() as i64, cai_log(cell, usage, &c))
            })
        })
        .collect();
    pareto(optimal)
}

fn computed(cell: &Cell, params: &Params, usage: &CodonUsage, seconds: f64) -> ModularFront {
    computed_from(cell, params, usage, seconds, false).0
}

/// [`computed`], from the walk's front with, when `reopen`, every gap between
/// neighbouring hull vertices put back as an open box: its inner points
/// dropped, as a walk stopped before that gap would return it. Also the
/// number of points dropped.
fn computed_from(
    cell: &Cell,
    params: &Params,
    usage: &CodonUsage,
    seconds: f64,
    reopen: bool,
) -> (ModularFront, usize) {
    let pins = cell.pins();
    let spec = MotifFrontSpec {
        utr5: cell.utr5,
        utr3: cell.utr3,
        constraints: cell.constraints,
        motifs: &cell.placed(),
        pins: pins.as_ref(),
    };
    let walked =
        motif_front_walk_run(params, cell.protein, usage, &spec, 1, MotifRun::default()).unwrap();
    assert!(walked.open.is_empty(), "{}: the walk stopped", cell.protein);
    let mut vertices = walked.vertices.clone();
    vertices.sort_by_key(|v| v.mfe.0);
    let mut boxes = Vec::new();
    let mut front = walked.front.clone();
    if reopen {
        for w in vertices.windows(2) {
            let (lo, hi) = (w[0].mfe.to_kcal_mol(), w[1].mfe.to_kcal_mol());
            let inner = |p: &derna_lib::FrontPoint| {
                p.mfe_kcal_mol > lo + 1e-9 && p.mfe_kcal_mol < hi - 1e-9
            };
            if walked.front.iter().any(inner) {
                front.retain(|p| !inner(p));
                boxes.push(OpenBox {
                    mfe_lo_kcal_mol: lo,
                    cai_log_lo: w[0].cai,
                    cai_log_hi: w[1].cai,
                });
            }
        }
    }
    let dropped = walked.front.len() - front.len();
    let so = modular_motif_front(
        params,
        cell.protein,
        usage,
        cell.utr5,
        cell.utr3,
        cell.constraints,
        (&cell.placed(), pins.as_ref()),
        (&front, &boxes),
        seconds,
        2,
    )
    .unwrap();
    (so, dropped)
}

fn points(front: &ModularFront) -> Vec<Point> {
    front
        .points
        .iter()
        .map(|p| ((p.mfe_kcal_mol * 100.0).round() as i64, p.cai_log))
        .collect()
}

fn same(a: &[Point], b: &[Point]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.0 == y.0 && (x.1 - y.1).abs() < 1e-6)
}

/// The modular search's cells with a motif each, chosen by
/// measuring that the motif removes points of the plain modular
/// front, so the filter decides the answer.
fn cells() -> Vec<Cell> {
    let five = RegionConstraints::five_prime_isolation(Enforcement::Forbid);
    let symmetric = RegionConstraints::symmetric_isolation(Enforcement::Forbid);
    vec![
        Cell {
            protein: "MRLGSA",
            utr5: "GGCAUCCAC",
            utr3: "GUGGAUGCC",
            constraints: symmetric,
            motifs: &["CCA", "UCC"],
            pins: &[],
        },
        Cell {
            protein: "MPRGLA",
            utr5: "CCGCGG",
            utr3: "",
            constraints: five,
            motifs: &["CGC"],
            pins: &[],
        },
        Cell {
            protein: "MLLRGA",
            utr5: "UCCGCGCAG",
            utr3: "",
            constraints: five,
            motifs: &["CUG"],
            pins: &[],
        },
        Cell {
            protein: "MRPGAL",
            utr5: "GGGCGCCCG",
            utr3: "",
            constraints: five,
            motifs: &["GCC"],
            pins: &[],
        },
        // A motif can straddle each UTR–CDS edge, and the straddlers change
        // the front: forbidding only occurrences wholly inside the coding
        // sequence admits more designs (found by scanning random cells).
        Cell {
            protein: "TAAAA",
            utr5: "UCAUUUG",
            utr3: "UCAC",
            constraints: symmetric,
            motifs: &["CUC"],
            pins: &[],
        },
        Cell {
            protein: "HTPSY",
            utr5: "AGAACGC",
            utr3: "UCCA",
            constraints: symmetric,
            motifs: &["ACUC"],
            pins: &[],
        },
    ]
}

#[test]
fn a_complete_run_is_the_exhaustive_motif_free_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut shapes = Vec::new();
    for cell in cells() {
        let truth = exhaustive(&cell, &params, &usage, true);
        let plain = exhaustive(&cell, &params, &usage, false);
        let got = computed(&cell, &params, &usage, 600.0);
        shapes.push((got.seeds, got.points.len(), !same(&plain, &truth)));
        assert!(got.complete, "{}: the run did not complete", cell.protein);
        assert!(got.points.iter().all(|p| p.certified), "{}", cell.protein);
        for p in &got.points {
            assert!(!spells_motif(&cell, &p.sequence), "{}", p.sequence);
        }
        assert!(
            same(&points(&got), &truth),
            "{}:\n got   {:?}\n truth {:?}\n plain {:?}",
            cell.protein,
            points(&got),
            truth,
            plain
        );
    }
    // The motifs change the front, and walks find points beside the seeds.
    assert!(shapes.iter().any(|&(_, _, moved)| moved), "{shapes:?}");
    assert!(
        shapes.iter().any(|&(seeds, n, _)| seeds > 0 && n > seeds),
        "{shapes:?}"
    );
}

#[test]
fn a_run_from_a_front_with_open_boxes_is_the_exhaustive_motif_free_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let mut dropped = 0;
    for cell in cells() {
        let truth = exhaustive(&cell, &params, &usage, true);
        let (got, d) = computed_from(&cell, &params, &usage, 600.0, true);
        dropped += d;
        assert!(got.complete, "{}: the run did not complete", cell.protein);
        assert!(got.points.iter().all(|p| p.certified), "{}", cell.protein);
        assert!(
            same(&points(&got), &truth),
            "{}:\n got   {:?}\n truth {:?}",
            cell.protein,
            points(&got),
            truth
        );
    }
    // Some gap held points, so some box was open.
    assert!(dropped > 0);
}

#[test]
fn a_stopped_run_certifies_only_motif_free_front_points_and_leaves_the_rest_open() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    for cell in cells() {
        let truth = exhaustive(&cell, &params, &usage, true);
        let got = computed(&cell, &params, &usage, 0.0);
        for p in got.points.iter().filter(|p| p.certified) {
            let at = ((p.mfe_kcal_mol * 100.0).round() as i64, p.cai_log);
            assert!(
                truth.iter().any(|t| same(&[*t], &[at])),
                "{}: certified {at:?} is not on the front {truth:?}",
                cell.protein
            );
        }
        let returned = points(&got);
        for t in &truth {
            if returned.iter().any(|r| same(&[*r], &[*t])) {
                continue;
            }
            let open = got.open.iter().any(|o| o.admits(t.0 as f64 / 100.0, t.1));
            assert!(open, "{}: {t:?} is neither returned nor open", cell.protein);
        }
    }
}

/// The first cell with leucine's third base pinned to U, which leaves it only
/// CUU: the pin takes the residue's most adapted codon, so the lattice's
/// adaptation sits below the front's by a non-zero offset.
fn pinned() -> Cell {
    let mut cell = cells().swap_remove(0);
    cell.pins = &[(9 + 8, "U")];
    cell
}

#[test]
fn a_pinned_run_is_the_exhaustive_pinned_motif_free_front() {
    let (params, usage) = (
        Params::load_default().unwrap(),
        CodonUsage::load_default().unwrap(),
    );
    let cell = pinned();
    let tables = compute_cai_tables(&usage);
    let leu = aa_to_index('L').unwrap();
    let cuu = get_codon_index(leu, [1, 3, 3]).unwrap();
    assert!(
        tables.log_cai[leu][cuu] < -1e-6,
        "the pin keeps the best codon"
    );
    let truth = exhaustive(&cell, &params, &usage, true);
    let unpinned = exhaustive(&cells().swap_remove(0), &params, &usage, true);
    assert!(!same(&truth, &unpinned), "the pin does not bind");
    let got = computed(&cell, &params, &usage, 600.0);
    assert!(got.complete);
    for p in &got.points {
        assert!(cell.spells_pins(&p.sequence), "{}", p.sequence);
        assert!(!spells_motif(&cell, &p.sequence), "{}", p.sequence);
    }
    assert!(
        same(&points(&got), &truth),
        "got   {:?}\n truth {:?}",
        points(&got),
        truth
    );
    let stopped = computed(&cell, &params, &usage, 0.0);
    for p in stopped.points.iter().filter(|p| p.certified) {
        let at = ((p.mfe_kcal_mol * 100.0).round() as i64, p.cai_log);
        assert!(truth.iter().any(|t| same(&[*t], &[at])), "{at:?}");
    }
}
