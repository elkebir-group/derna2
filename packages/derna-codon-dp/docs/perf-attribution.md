# Codon-DP perf attribution

Hot-path attribution and constant-factor comparison for the Rust `solve_brd`
and `solve_pord` λ-sweep against the [DERNA C++](https://github.com/elkebir-group/derna)
reference. The headline attribution is **per-cell time per recurrence**;
the cube is filled per-solve regardless of how many λs the sweep visits,
so a single-solve profile attributes hot-path time to the same recurrences
a full-sweep profile would.

> **Note on the "M" prefix in this doc.** "Apple Silicon Apple M1 Pro" and
> "Apple Silicon Apple M5 Max" name hardware. This doc describes the
> current code shape.

## Methodology — single-solve at n=20 for per-cell attribution, n=78 sweep for end-to-end

Two measurement regimes, used together:

- **Per-cell attribution: single `solve_brd(GLPE_HUMAN_20aa_cut, λ=0.5)`** on the
  20-AA prefix of GLPE_HUMAN. Wall-clock is small (sub-second after the
  current set of optimizations), so per-recurrence hot paths come out cleanly
  without λ-schedule or `solve_pord` bisection bookkeeping muddying the
  attribution. The 6-D DP cube is filled per-solve regardless of λ.

- **End-to-end: full `solve_pord` λ-sweep on GLPE-78** (78 AAs, ~25 λs
  visited). This is the ratio that matters for product perf vs. DERNA C++ at
  realistic protein sizes. Per-cell shares do not project linearly from n=20
  to n=78 — see [§ Sweep-scale lesson](#sweep-scale-lesson).

Build: workspace `--release` with `[profile.release].debug = "line-tables-only"`.
Tool: samply 0.13.1, 1 ms sampling.

## Top-line per-cell attribution (n=20, λ=0.5)

samply self-time on a single `solve_brd` at n=20, λ=0.5. Symbols are taken
from the latest steady-state profile.

| Rank | % self | Symbol |
|---|---:|---|
| 1 | 26.63 % | `derna_codon_dp::solve::solve_brd_with_params` |
| 2 | 19.05 % | `recurrences::internal::emit_internal_candidates::{{closure}}::{{closure}}` |
| 3 | 11.48 % | `recurrences::internal::compute_internal_energy` |
| 4 | 10.30 % | `recurrences::internal::read_base_pairs` |
| 5 |  7.81 % | `rna_mfe::params::parse::parse_params` |
| 6 |  7.34 % | `recurrences::internal::build_inner_scope` |

The structural hot path inside the codon DP is the **internal-loop
boundary-nt enumeration closure chain** (`emit_internal_candidates` +
`build_inner_scope` + `read_base_pairs`) plus `compute_internal_energy`.
`parse_params` is per-solve init cost; it amortises away at full-sweep
scale (C++ pays the same per-solve init cost).

## End-to-end: GLPE-78 full sweep

Sweep wall-clock and Pareto records on GLPE-78, Apple Silicon Apple M5 Max,
rustc 1.95.0:

| Side | Sweep wall-clock | Pareto records | Unique hull cells |
|------|----------------:|---------------:|-----------------:|
| C++ (`lambda_sweep_2`) | ~432 s (≈ 7.2 min) | 25 | 13 |
| Rust (`solve_pord`)    | ~3138 s (≈ 52 min) | 29 | 26 |

Rust / C++ sweep ratio: **~7.3×**. C++ per-solve at n=78: ~17.3 s. Rust
per-solve at n=78: ~108 s. Empirical exponent fit from per-cell time at
n=20 single-solve to per-cell time at n=78 sweep: ≈ n^3.6 (cube-only
prediction is n^3, so the constant-factor cost rises with `n` — see
[§ Sweep-scale lesson](#sweep-scale-lesson)).

### Hull comparison — `cpp ⊆ rust`

`packages/derna-codon-dp/scripts/cpp_parity/diff_hulls.py` at the standard
`tol_mfe = 0.02 kcal/mol`, `tol_cai = 1e-6` tolerances satisfies the
contract — `cpp-only orphans` is empty. Rust finds all of C++'s 13 hull
cells (12 exact + 1 co-optimal-tie) plus 11 additional non-dominated Pareto
vertices C++'s bisection schedule does not visit, plus 2 dominated
MFE-corner integer-dekacal tie-break duplicates (`mfe = -148.70`).

The 1 co-optimal-tie at `mfe = -118.80 kcal/mol`, `cai = 0.968451` is
informational under the parity contract: identical RNA sequence on both
sides; structure-string match perturbed beyond the shared 32-char prefix.

## BRD scaling and the `--cai-scale` remap

DERNA C++ and the Rust port use different BRD scalar conventions:

| Side | BRD scalar | CAI representation |
|------|-----------|---------------------|
| Rust port (faithful to v1) | `round(λ · MFE_dekacal − (1−λ) · CAI_dekacal)` | `round(log_CAI · 100)` ("CAI_DEKACAL_SCALE = 100") |
| DERNA C++ | `λ · MFE_dekacal − (1−λ) · log_CAI` (no CAI scaling) | raw `log_CAI` (sum of unscaled log-frequency-ratios) |

The 100× CAI scaling means Rust weights CAI ~100× more heavily than DERNA
C++ at the same λ. Two consequences:

1. **Naive per-λ parity Rust↔C++ cannot hold.** At identical λ, the two
   solvers minimize different objectives. The fix at the comparison layer
   is a closed-form λ remap (`λ_R = 100·λ_C / (1 + 99·λ_C)`) applied by
   `run_brd_per_lambda --cai-scale 100` before calling `solve_brd`, so
   per-λ comparison is apples-to-apples.
2. **The bisection schedules diverge.** Rust's finer CAI weighting makes
   its bisection discriminate between codon assignments at the same `mfe`
   cell where C++'s coarser weighting collapses them to a single argmin.
   This shows up as Rust finding additional non-dominated Pareto vertices
   on the full-sweep hull comparison.

For full-sweep `diff_hulls.py` comparison, no remap is needed — both sides
converge on the same `(MFE, CAI)` Pareto front modulo the integer-dekacal
precision residuals captured by the four-bucket comparison shape (see
[`packages/derna-codon-dp/scripts/cpp_parity/README.md`](../scripts/cpp_parity/README.md)).

## Correctness anchors

The current correctness invariants the per-cell perf attribution rests on:

- **Per-cell parity vs v1**: byte-identical at every cell across λ ∈ {0, 0.5, 1}
  (`tests/recurrences_single.rs`, `tests/recurrences_multi.rs`). The strongest
  correctness invariant. The Rust port faithfully implements the v1 reference
  at every motif site.
- **Per-λ Rust ↔ DERNA-C++ parity at n=20** via `--cai-scale 100` remap
  (`tests/solve_brd.rs`): 12 exact + 6 co-optimal-tie + 4 explainable diffs
  / 22.
- **Hull recovery vs brute-force enumeration fixtures** (`tests/solve_pord.rs`):
  Rust's hull matches the brute-force ground truth.
- **GLPE-78 hull `cpp ⊆ rust`** at the standard tolerances: Rust hull is
  a superset of C++'s.

<a id="sweep-scale-lesson"></a>

## Sweep-scale lesson

Per-cell hot-path % at n=20 does **not** project linearly to sweep-scale
wall-clock at n=78. The Rust/C++ per-solve ratio widens from ~1.4× at n=20
incl-init to ~6.2× at n=78. Two candidate explanations:

1. **L1 / L2 cache pressure scales worse for Rust at large `n`.** The 6-D
   `i32` DP table at n=78 is `(78 × 78 × 8 × 8 × 4 × 4) × 4 B` ≈ 25 MB per
   recurrence table — well above Apple Silicon L2 capacity (~16 MB shared).
   At n=20 the table fits in L2. C++'s table layout (column-major bit-packed
   in `Zuker.cpp`) may amortise the L3 miss cost differently than Rust's
   flat-backed layout.
2. **`emit_internal_candidates` closure chain inlines differently at sweep
   scale.** LLVM's inlining heuristics depend on caller-callee shape; the
   sweep driver (`solve_pord` + `solve_brd` repeated) may de-inline some
   paths that the single-solve profile didn't see.

**Methodological consequence**: always include a sweep-scale ground-truth
in any perf DoD, not only a single-solve projection. Single-solve hot-path
% is "*what's slow per cell*"; sweep wall-clock is "*how slow it is
end-to-end*". The two need not project linearly when `n` changes table size
enough to shift the cache regime.

## Reproducing the samply attribution

```bash
cargo build --release --workspace
samply record \
    --save-only --unstable-presymbolicate \
    -o target/profile/solve_brd-n20-lambda05.json \
    -- ./target/release/profile_solve_brd \
        --fasta packages/derna-codon-dp/tests/data/GLPE_HUMAN_20.fasta \
        --codon-usage packages/derna-codon-dp/data/codon_usage_human.csv \
        --lambda 0.5
```

The `--unstable-presymbolicate` flag emits a `*.syms.json` sidecar holding
the rva → symbol table needed to resolve Rust mangled names without
launching the samply web UI.

## Reproducing the n=78 capstone

```bash
# Both sides — runs idempotently from a clean checkout:
bash packages/derna-codon-dp/scripts/cpp_parity/cpp/run_cpp.sh
bash packages/derna-codon-dp/scripts/cpp_parity/rust/run_rust.sh

# Hull diff:
python3 packages/derna-codon-dp/scripts/cpp_parity/diff_hulls.py \
    --cpp   packages/derna-codon-dp/scripts/cpp_parity/cpp/pareto_points.jsonl \
    --other packages/derna-codon-dp/scripts/cpp_parity/rust/pareto_points.jsonl
```

## Reproducing the per-λ parity check

```bash
# C++ side: regenerate pareto_points.jsonl at n=20
bash packages/derna-codon-dp/scripts/cpp_parity/cpp/run_cpp.sh \
    --fasta packages/derna-codon-dp/tests/data/GLPE_HUMAN_20.fasta

# Rust side: feed the C++ λ list through solve_brd with the cai-scale
# remap so the two sides minimize the same effective BRD scalar
cargo build --release --workspace
./target/release/run_brd_per_lambda \
    --fasta packages/derna-codon-dp/tests/data/GLPE_HUMAN_20.fasta \
    --codon-usage packages/derna-codon-dp/data/codon_usage_human.csv \
    --lambdas packages/derna-codon-dp/scripts/cpp_parity/cpp/pareto_points.jsonl \
    --out    packages/derna-codon-dp/scripts/cpp_parity/rust/per_lambda.jsonl \
    --cai-scale 100

# Diff
python3 packages/derna-codon-dp/scripts/cpp_parity/diff_per_lambda.py \
    --cpp   packages/derna-codon-dp/scripts/cpp_parity/cpp/pareto_points.jsonl \
    --other packages/derna-codon-dp/scripts/cpp_parity/rust/per_lambda.jsonl
```
