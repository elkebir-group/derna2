# DERNA C++ ↔ DERNA Rust parity harness

A reproducible diff between [DERNA C++](https://github.com/elkebir-group/derna)'s `lambda_sweep_2` and the Rust `solve_pord` at `packages/derna-codon-dp/` on the same protein. Both solvers are λ-sweep BRDs that produce the convex hull of the (MFE, CAI) Pareto front; the harness asserts the hulls agree. The Rust port is the live reference; the earlier Python implementation is retired.

## Layout

```
README.md                       this file
translate_codon_usage.py        long-form Kazusa CSV → C++ matrix-form CSV
diff_hulls.py                   load both JSONLs, assert hull equality, print diff
diff_per_lambda.py              per-λ comparison
cpp/
  run_cpp.sh                    build C++, run lambda_sweep_2, project CSV → JSONL
  pareto_points.jsonl           output (gitignored)
rust/
  run_rust.sh                   build derna-codon-dp release, run solve_pord, emit JSONL
  pareto_points.jsonl           output (gitignored)
```

## Fixed-λ anchor fixtures

`gen_anchor_fixtures.py` is the other half of this harness, and the one the
test suite depends on. Where `diff_hulls.py` compares whole sweeps ad hoc, the
generator records what the C++ returns at a handful of fixed weights into
`tests/data/cpp_anchor_*.json`, which `tests/cpp_anchor.rs` asserts against on
every run. The C++ is needed to *regenerate* those fixtures, never to run the
tests.

```bash
git clone --depth 1 https://github.com/elkebir-group/derna docs/reference/_cpp/derna-main
# build it (C++17), then:
python3 gen_anchor_fixtures.py --derna-bin /path/to/derna
```

Two things the generator gets right that a hand-run would not. It **translates
the codon table** through `translate_codon_usage.py` so both sides weigh the
committed Kazusa CSV rather than two copies of it — passing the long-form CSV
straight to the C++ fails with `no char found`. And it **reparametrizes the
weight**: the C++ pairs dekacal energies with raw log-CAI where this solver
scales log-CAI by 100, so the equivalent weight is `λ / (100 − 99λ)`. Each
fixture carries both coordinates. The Rust DP carries the fixed-λ objective
unrounded, as the C++ does, so the fixtures pin the exact sequence and
structure rather than a tolerance.

## Common output: `pareto_points.jsonl`

One record per λ visited by the sweep. Both sides emit this file in their own subdirectory.

```jsonc
{
  "lambda": 0.0025,                 // float — λ at which this point was the argmin
  "mfe_kcal_mol": -148.7,           // float — Vienna-style ΔG (negative = stable)
  "cai": 0.716842,                  // float in [0, 1] — geometric-mean (standard) CAI
  "rna_seq": "AUGUAUGGC...",        // str — RNA, length 3·|protein|, alphabet ACGU
  "structure": "(((((((((....",     // str — dot-bracket, length 3·|protein|
  "source": "cpp"                   // "cpp" or "v1"
}
```

`cai` is the **standard CAI** in [0, 1] (geometric mean of relative codon adaptation). On the C++ side this is the `sCAI` column — *not* `CAI`, which is the log-form scaled CAI used internally by the optimizer.

## Tolerances

| Field | Tolerance | Why |
|---|---|---|
| `mfe_kcal_mol` | `\|Δ\| ≤ 0.02` kcal/mol | v1 uses dekacal-int arithmetic (0.01 kcal/mol per term); C++ uses doubles. 0.02 covers boundary drift across the accumulation. |
| `cai` | `\|Δ\| ≤ 1e-6` | Both sides accumulate `Σ log(f_i / f_max)` then `exp(…/n)` in double precision; real drift is ≪ 1e-9. |

Tolerances are read by `diff_hulls.py` at diff time. The JSONL artifacts hold full precision, so re-running the diff with different tolerances is cheap.

## λ-matching policy: hull-vertex equality (Rust-superset framing)

Both implementations bisect from `(EPSILON, 1 - EPSILON) = (1e-5, 1 - 1e-5)` with the same `(threshold, threshold2) = (0.0025, 0.00075)` constants. They *should* produce comparable λ sequences but may diverge at float-vs-int boundaries — and they diverge **systematically** because of the BRD-scalar scaling difference between sides (see § *BRD scaling and the `--cai-scale` remap* below). The harness does **not** assert λ-set equality.

The diff acceptance contract is **hull-vertex equality with a Rust-superset framing**: after dominance-pruning the comparison cells, **C++'s hull must be a subset of Rust's hull** — Rust finds every C++ Pareto vertex, and may additionally find non-dominated vertices that C++'s bisection schedule doesn't visit at the same protein. The failure mode is C++ finding a Pareto vertex Rust missed. The diff acceptance shape is therefore:

1. Deduplicate each side's records to the set of `(mfe_kcal_mol, cai)` cells (within tolerance — multiple λs can witness the same vertex).
2. **Cell-set comparison reports four buckets:** *shared* (cells in both sides within tolerance); *cpp-only orphans* (cells in C++ only); *rust-only orphans (dominated by shared)* (Rust extras that integer-dekacal tie-break differently than C++ at the same hull cell — informational); *rust-only orphans (non-dominated)* (Rust extras that are genuine additional Pareto vertices — informational). Failure mode: **non-empty `cpp-only orphans`.** Rust-only orphans of either subtype **do not fail** the assertion; they're recorded for the perf-attribution doc.
3. For each shared cell, compare `(rna_seq, structure)`. Exact match → log "exact"; otherwise log "co-optimal-tie". Co-optimal ties **do not fail** the assertion — both sides found a valid argmin at this `(mfe, cai)` cell, they picked different RNAs.

Equivalently: **the diff fails iff Rust's hull is missing a Pareto vertex C++ found.** Rust finding extras is evidence of *more complete* Pareto-front exploration, not a parity violation. This framing absorbs the integer-dekacal precision artifact at MFE-corner ties (C++ floats discriminate near-tied CAI values; Rust's i32 dekacal arithmetic ties them and the traceback picks a different representative — both representatives are valid, neither dominates the other).

## BRD scaling and the `--cai-scale` remap

v1/Rust quantize CAI as `CAI_dekacal = round(log_CAI · 100)` (the `CAI_DEKACAL_SCALE = 100` factor in v1's `_dp.py`). DERNA C++ minimizes against raw `log_CAI` with no scaling. Both sides minimize `λ·MFE − (1−λ)·CAI_term`; since argmin is invariant under positive scaling of the objective, they pick the same `(MFE, CAI)` vertex iff their MFE-to-`log_CAI` weight ratios agree. Solving:

```
λ_R = s·λ_C / (1 + (s−1)·λ_C)      // s = CAI scale, 100 for the v1↔C++ pairing
```

`run_brd_per_lambda --cai-scale 100` applies this remap before calling `solve_brd`, so per-λ comparison is apples-to-apples. Output JSONL's `lambda` field carries the original C++ `λ_C` so `diff_per_lambda.py` can group cross-side records by it.

For full-sweep `diff_hulls.py` comparison, no remap is needed — both sides converge on the same `(MFE, CAI)` Pareto front modulo the integer-dekacal precision residuals captured by the four-bucket comparison shape above.

## C++ side

Source: `docs/reference/_cpp/derna-main/` (gitignored under the `_*` convention; fetch with `git clone https://github.com/elkebir-group/derna docs/reference/_cpp/derna-main`).

Build:

```bash
cmake -S docs/reference/_cpp/derna-main -B docs/reference/_cpp/derna-main/build
cmake --build docs/reference/_cpp/derna-main/build
```

Dependencies: any C++11 compiler. Apple Clang on macOS works.

Run (sweep mode):

```bash
docs/reference/_cpp/derna-main/build/derna \
  -i packages/derna-codon-dp/tests/data/GLPE_HUMAN.fasta \
  -o <run_log.txt> \
  -O <csv_basename>          \
  -m 1                       `# Zuker model`        \
  -s 3                       `# lambda-sweep mode`  \
  -t 0.0025                  `# threshold`          \
  -p 0.00075                 `# threshold2`         \
  -c <translated_codon_usage.csv>
```

Leave `-d` unset to use the baked-in Turner 2004 parameters from `src/default.cpp`. `EPSILON = 1e-5` is hard-coded at `src/params/constants.h:32`.

C++ writes `<csv_basename>.csv` with header:
```
lambda,gamma,mfe,CAI,sCAI,seq,bp,sub_mfe,sub_CAI,sub_sCAI,sub_seq,sub_bp
```

`run_cpp.sh` projects to JSONL with these rules:

- `lambda → lambda` (float, unchanged)
- **`mfe → mfe_kcal_mol` with `/100.0` conversion** — the C++ CSV `mfe` column is **dekacal-int** (the same internal representation v1 uses), not kcal/mol. `ZukerAlgorithm::calculate_W()` returns `int` in dekacal units (0.01 kcal/mol per Turner 2004 parameter); `lambda_sweep_2` writes that integer to the CSV directly. v1's `ParetoPoint.mfe_kcal_mol` already applies the `/100.0` conversion, so dividing on the C++ side too means both JSONLs share `kcal/mol` units.
- `sCAI → cai` (float in [0, 1] — **not** `CAI`, which is the log-form scaled CAI used internally by the optimizer)
- `seq → rna_seq` (string)
- `bp → structure` (string)
- All `sub_*` columns dropped

Rows with `gamma != "None"` (suboptimal samples) are filtered defensively; `subopt = false` is hardcoded at `Zuker.cpp:7142` so they don't appear in practice.

## Codon-usage translation

C++ reads codon usage as a matrix (one header row of codons per amino acid, one row of values). The Rust port reads long-form (`amino_acid,codon,fraction`).

`translate_codon_usage.py` reads `packages/derna-codon-dp/data/codon_usage_human.csv` (the canonical Kazusa *H. sapiens* table, transcribed from the C++ source) and writes the C++ matrix-form CSV. C++ normalizes input internally as `f / max(f)` per amino acid (`default.cpp:1701`), so passing fractions directly works — no need to reconstruct integer counts.

## Rust side

`packages/derna-codon-dp/src/bin/run_glpe78.rs` runs the `solve_pord` sweep and emits JSONL conformant to this contract:

```bash
cargo build --release --workspace
./target/release/run_glpe78 \
    --fasta packages/derna-codon-dp/tests/data/GLPE_HUMAN.fasta \
    --codon-usage packages/derna-codon-dp/data/codon_usage_human.csv \
    --out packages/derna-codon-dp/scripts/cpp_parity/rust/pareto_points.jsonl
```

`dangling_5_prime = false` is hardcoded — C++ uses `dangles = 0` and the Rust port only supports that mode: its cell index does not track the flanking-codon context dangles would need.

`run_rust.sh` wraps this with the standard FASTA + codon-usage defaults and a machine-snapshot sidecar.
The Rust sweep also writes `<out-stem>.telemetry.json` with per-λ BRD timing,
memo hit/miss counts, total solve count, Pareto record count, and sweep
wall-clock.
