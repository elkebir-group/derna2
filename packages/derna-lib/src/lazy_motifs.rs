//! Lazy enforcement of a forbidden-motif set: solve under the motifs the
//! optimum actually spells, not under all of them.
//!
//! Forbidding a subset `S'` of the set `S` relaxes the problem — every
//! `S`-free encoding is `S'`-free — so the exact optimum over `S'`-free
//! encodings is a lower bound on the one over `S`-free encodings. When that
//! relaxed optimum happens to spell no motif of `S` it is feasible for the full
//! problem and meets the bound, so it **is** the exact `S`-constrained optimum.
//! Otherwise the motifs it spells join `S'` and the solve repeats. Each round
//! strictly grows `S'`, so the loop ends within `|S| + 1` solves, and the last
//! possible round is the full set, which cannot return a dirty design.
//!
//! The first round is `S' = ∅`, the unconstrained scan. When the free optimum
//! is already motif-free the whole answer costs one unconstrained fold; when it
//! is not, the rounds that follow fold under an automaton built from the
//! offending motifs alone, which is typically a small part of the set.
//!
//! **What the guarantee rests on, and so where it is offered.** The argument
//! needs every round to be an *exact* optimum over the same feasible set the
//! full run would search, minus the motif axis. That holds on the exact lattice
//! and fails under a finite beam, where a relaxed beam optimum bounds nothing.
//! It also needs the clean-check to read the same occurrences the automaton
//! forbids, which is whole-molecule enforcement over a CDS-only design, the
//! sequence [`Design::sequence`] spells. A run outside that — a finite beam, a
//! narrower [`MotifWindow`], designed or given flanks, or a derived set — takes
//! the full constrained fold unchanged and reports itself as not lazy.
//!
//! **Equal objective, not equal bytes.** The answer has the full fold's BRD
//! objective, but where several motif-free designs tie at it the two paths may
//! emit different ones: the unconstrained scan and the constrained folder break
//! ties independently.

use crate::ac::{Ac, MotifWindow};
use crate::{
    design_composite, BeamError, CodonUsage, CompositeSpec, Design, Nucleotide, Params, ScanConfig,
};

/// A design reached by lazy motif enforcement, with what it took.
#[derive(Debug, Clone)]
pub struct LazyDesign {
    /// The design. Its BRD objective is the full constrained optimum's.
    pub design: Design,
    /// Solves run, the first one unconstrained. `1` means the free optimum was
    /// already motif-free.
    pub rounds: usize,
    /// Indices into the caller's motif list of the motifs the final round
    /// enforced — empty when the free optimum was clean.
    pub enforced: Vec<usize>,
    /// Whether the run was eligible for lazy enforcement at all. `false` means
    /// it took the full constrained fold in a single round.
    pub lazy: bool,
}

/// Design under `spec`'s motif set by lazy enforcement where that is exact (see
/// the module docs), and by the full constrained fold otherwise.
pub fn design_motifs_lazy<S: AsRef<str>>(
    params: &Params,
    protein: &str,
    usage: &CodonUsage,
    lambda: f64,
    spec: &CompositeSpec<'_, S>,
    config: ScanConfig,
) -> Result<LazyDesign, BeamError> {
    if !eligible(spec, config) {
        let design = design_composite(params, protein, usage, lambda, spec, config)?;
        return Ok(LazyDesign {
            design,
            rounds: 1,
            enforced: (0..spec.motifs.len()).collect(),
            lazy: false,
        });
    }
    let matchers: Vec<Ac> = spec
        .motifs
        .iter()
        .map(|m| Ac::build(&[m.as_ref()]))
        .collect::<Result<_, _>>()?;
    let mut enforced: Vec<usize> = Vec::new();
    for rounds in 1.. {
        let active: Vec<&str> = enforced.iter().map(|&i| spec.motifs[i].as_ref()).collect();
        let relaxed = CompositeSpec {
            utr5: spec.utr5,
            utr3: spec.utr3,
            constraints: spec.constraints,
            profile: spec.profile,
            motifs: active.as_slice(),
            motif_window: spec.motif_window,
            flank_complement: spec.flank_complement,
            complement_classes: spec.complement_classes,
            positional: spec.positional,
        };
        // An infeasible relaxation is an infeasible full problem: its feasible
        // set contains the full one.
        let design = design_composite(params, protein, usage, lambda, &relaxed, config)?;
        let spelled = spelled_motifs(&matchers, &design.sequence);
        let fresh: Vec<usize> = spelled
            .into_iter()
            .filter(|i| !enforced.contains(i))
            .collect();
        if fresh.is_empty() {
            debug_assert!(
                spelled_motifs(&matchers, &design.sequence).is_empty(),
                "an enforced motif was spelled"
            );
            return Ok(LazyDesign {
                design,
                rounds,
                enforced,
                lazy: true,
            });
        }
        enforced.extend(fresh);
        enforced.sort_unstable();
    }
    unreachable!("the round counter is unbounded")
}

/// Whether the lazy argument is exact for this run: an exact lattice,
/// whole-molecule enforcement of a supplied set, and no flank.
fn eligible<S: AsRef<str>>(spec: &CompositeSpec<'_, S>, config: ScanConfig) -> bool {
    config.beam.is_exact()
        && spec.flank_complement.is_none()
        && spec.motif_window == MotifWindow::all()
        && spec.utr5.is_empty()
        && spec.utr3.is_empty()
}

/// Indices of the motifs whose automaton accepts somewhere along `sequence`.
fn spelled_motifs(matchers: &[Ac], sequence: &str) -> Vec<usize> {
    let seq: Vec<Nucleotide> = sequence
        .bytes()
        .map(|b| Nucleotide::from_byte(b).expect("a design spells A/C/G/U"))
        .collect();
    matchers
        .iter()
        .enumerate()
        .filter(|(_, ac)| {
            let mut q = ac.start();
            seq.iter().any(|&x| {
                q = ac.step(q, x);
                ac.is_accept(q)
            })
        })
        .map(|(i, _)| i)
        .collect()
}
