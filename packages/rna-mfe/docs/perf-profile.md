# `rna-mfe` perf profile

This document captures `samply` profiles of `rna_mfe::fold` (at 800 nt and
3000 nt) and `rna_mfe::evaluate` (at 3000 nt), cross-referenced against
the ViennaRNA v2.7.0 source (<https://github.com/ViennaRNA/ViennaRNA>, tag `v2.7.0`). The current
profile reflects the post-fix steady state: `rna_mfe::fold` is now faster
than ViennaRNA across the corpus-relevant length ladder.

The samply profiles themselves (`fold_800.json`, `fold_3000.json`,
`evaluate_3000.json` plus `.syms.json` sidecars) are not committed —
they live under `profiles/` in the worktree where they were
recorded. To reproduce, see *Reproduction recipe* at the bottom.

## Profile setup

| Field | Value |
| --- | --- |
| Profiled at | commit `87d259f`, `examples/mfe_fold` / `examples/mfe_eval` binaries |
| Host | Apple Silicon, macOS 26.4.1 |
| Build | release with `[profile.release].debug = "line-tables-only"`, `lto = "thin"`, `codegen-units = 1` |
| Inputs | `synth_rna(len)`, xorshift64 seed=42 (matches `tests/bench.rs`) |
| Tool | samply 0.13.1, `--unstable-presymbolicate` for offline symbol resolution |
| Sample rate | 1000 Hz for fold profiles; 4000 Hz for evaluate (faster per-call kernel needs more samples) |

## Wall-clock vs Vienna

Median across N=5 trials per length:

| length | rust_mfe (ms) | Vienna (ms) | ratio |
|------:|--------:|-------:|------:|
| 800   | 181.8   | 308.6  | **0.59×** |
| 1500  | 792.8   | 1193.9 | **0.66×** |
| 3000  | 4712.4  | 5839.6 | **0.81×** |

`rna_mfe::fold` is faster than ViennaRNA across the ladder. The
performance target (≤1.0× Vienna at 800 and 3000 nt) is the steady-state
contract.

## Hot paths

samply attribution at full LTO is coarser than the call-graph would
suggest: many small DP helpers (`compute_v`, `compute_m`, `compute_m1`,
`compute_w`, `m_bifurcation`, `best_multi_split`, `ml_branch_total`,
extension cases) are inlined into the outer fill loop and reported
collectively as `fill_dp_tables`. Functions that escape inlining
(`internal_energy`, `mismatch_sum`) keep their own self-time.

### `fold` at 800 nt — 5 folds, 911 samples

| % self | path |
|-------:|------|
| 78.27% | `dp::fill::fill_dp_tables` |
| 12.18% | `loops::internal::mismatch_sum` |
|  9.00% | `loops::internal::internal_energy` |
|  0.00% | `dp::fill::two_pair_total` |

### `fold` at 3000 nt — 1 fold, 4715 samples

| % self | path |
|-------:|------|
| 87.25% | `dp::fill::fill_dp_tables` |
|  7.23% | `loops::internal::mismatch_sum` |
|  5.32% | `loops::internal::internal_energy` |
|  0.00% | `dp::fill::two_pair_total` |

`two_pair_total` carries 0% self-time at both lengths — its work is absorbed
into `fill_dp_tables` via the inlined `inner_min_at_l` kernel. The two-pair
leaf search is not a separable hot path; the dominant cost is the multi-loop
bifurcation + F-fill cubic loops inlined into `fill_dp_tables` (the
`zip_add_min` kernel plus `ml_branch_total`, `m_extend_*`, `m1_extend_right`).

`mismatch_sum` and `internal_energy` together account for ~13–20 % of
profile share. In absolute terms each is a small constant cost (~16–22 ms
at 800 nt); they appear as a larger fraction at 800 nt than at 3000 nt
because the cube dominates more strongly at long lengths.

### `evaluate` at 3000 nt — 30 000 calls

Inclusive ~95% of samples have a frame from the `parse_seq` → `Vec::collect`
→ allocator chain on the stack. **`evaluate` at 3000 nt is allocator-bound,
not compute-bound.** The fix to make `evaluate` competitive in
allocator-sensitive callers (e.g., codon-DP λ-sweep with O(n³ · |codons|²)
calls) is to use the `evaluate_encoded(&Params, &[Nucleotide], &PairTable)`
sibling API that takes pre-encoded inputs, bypassing the per-call sequence
and pair-table allocations.

## Identified asymmetries vs ViennaRNA — historical reference

The following asymmetries were the motivating analysis for the current
shape of `dp/tables.rs`, `dp/fill.rs`, and `lib.rs`. They are documented
here as the *what and why* of the current layout. The fixes have shipped;
the symbols mentioned (`PairTable`'s flat `Vec<i32>` with `INF` sentinels,
`evaluate_encoded`, the flat `inner_min_at_l` kernel, the outer-`l`
decreasing loop order) reflect the live code.

### A1. Stride-n column access in M1 and V cell reads — addressed by triangular `i32` PairTable

**What.** Vienna stores its DP matrices in a column-major triangular
layout: `c[indx[j] + i]` where `indx[j] = j*(j-1)/2`, so all of column
`j` lives contiguously in memory and varying `i` for fixed `j` is
stride-1.

A naive row-major-square `Vec<Option<Energy>>` indexed `i * n + j` would
be **stride-n** for fixed-`j` column access — at n = 3000 that's 24 KB
per index step, each crossing a new cache line. `rna_mfe` uses Vienna's
column-major triangular layout for the same reason.

**Where (Rust).** `dp/tables.rs` (`PairTable`) and the call sites in
`dp/fill.rs`, `dp/multi.rs` that read `V` / `M` / `M1` cells.

**Where (Vienna).** `mfe/mfe_multibranch.c::vrna_fun_zip_add_min` and
`mfe/mfe_exterior.c` F-fill — both use stride-1 access via the column-major
triangular layout.

### A2. `i32` sentinel instead of `Option<Energy>` — addressed by `INF` convention

**What.** `Option<Energy>` is **8 bytes** (4-byte tag + 4-byte payload)
because `i32` has no niche space. Vienna stores plain `int` (4 bytes) and
uses `INF = ~32767` as the "no structure" sentinel.

`PairTable` uses Vienna's convention: plain `i32` with `INF` sentinel.
This cuts memory 2× per cell vs. the `Option<Energy>` alternative; combined
with the triangular layout, the total per-table footprint at n=3000 is
~18 MB rather than ~72 MB, comparable to Vienna's.

Apple Silicon L1d is 128 KB / core; L2 is 16 MB shared. At 800 nt
four tables (~5 MB) fit in L2; at 3000 nt (~72 MB) exceed L2 by ~4×.
Cache-pressure scaling matches Vienna's.

### A3. Per-call sequence allocation in `evaluate` — addressed by `evaluate_encoded`

**What.** A naive `evaluate(seq: &[u8], structure: &str)` allocates a
fresh `Vec<Nucleotide>` of length n on every call (via `parse_seq`), plus
several pair-table allocations from `decompose`. For callers that evaluate
many structures over the same sequence, this allocator traffic dwarfs the
energy kernel (~95 % of inclusive time on a 3000 nt evaluate-heavy load).

**Fix shape.** `evaluate_encoded(&Params, &[Nucleotide], &PairTable)`
takes pre-encoded inputs. The public `evaluate(&[u8], &str)` is a thin
shell over `evaluate_encoded` that encodes once per call; allocator-sensitive
callers (codon DP, beam search) hold `&[Nucleotide]` + `&PairTable`
across calls and bypass the encoding work.

**Where (Vienna).** Vienna's `vrna_eval_structure_*` family takes a
pre-allocated `vrna_fold_compound_t` that holds `S[]` (the encoded
sequence, allocated once at compound construction time) and only allocates
a small per-call `pair_table` — `eval/eval_structures.c:215, 266, 283`.
The expected usage pattern is "one compound, many evals."

## Reproduction recipe

The recipe expects a ViennaRNA v2.7.0 checkout
(`git clone --depth 1 --branch v2.7.0 https://github.com/ViennaRNA/ViennaRNA.git`)
for the source references below, and `samply` (`cargo install samply`).
`gen_synth.py` (writes N random sequences of length L, one per line) and
`extract_hot.py` (ranks a saved samply profile by self time per function)
are small ad-hoc helpers that are not part of the repository.

```bash
cargo build --release --workspace --examples

mkdir -p profiles
python3 gen_synth.py 800 5  > profiles/seqs_800.txt
python3 gen_synth.py 3000 1 > profiles/seqs_3000.txt

# Reproducibility check (no samply, only /usr/bin/time)
/usr/bin/time -p target/release/examples/mfe_fold < profiles/seqs_800.txt  > /dev/null
/usr/bin/time -p target/release/examples/mfe_fold < profiles/seqs_3000.txt > /dev/null

# Profiles
samply record --save-only --unstable-presymbolicate --rate 1000 \
  -o profiles/fold_800.json -- \
  target/release/examples/mfe_fold < profiles/seqs_800.txt > /dev/null
samply record --save-only --unstable-presymbolicate --rate 1000 \
  -o profiles/fold_3000.json -- \
  target/release/examples/mfe_fold < profiles/seqs_3000.txt > /dev/null

# Build a (seq, structure) fixture for evaluate
target/release/examples/mfe_fold < profiles/seqs_3000.txt \
  | awk -F'\t' '{print $1}' > profiles/struct_3000.txt
python3 -c "
seq    = open('profiles/seqs_3000.txt').read().strip()
struct = open('profiles/struct_3000.txt').read().strip()
with open('profiles/seq_struct_3000_x30000.txt', 'w') as f:
    for _ in range(30_000):
        f.write(seq); f.write('\n'); f.write(struct); f.write('\n')
"

samply record --save-only --unstable-presymbolicate --rate 4000 \
  -o profiles/evaluate_3000.json -- \
  target/release/examples/mfe_eval < profiles/seq_struct_3000_x30000.txt > /dev/null

# Resolve hot paths offline
python3 extract_hot.py profiles/fold_800.json     --top 25 --no-filter
python3 extract_hot.py profiles/fold_3000.json    --top 25 --no-filter
python3 extract_hot.py profiles/evaluate_3000.json --top 30 --no-filter
```

`gen_synth.py` and `extract_hot.py` are kept in `profiles/`
(gitignored). They're 30–150 line scripts: the first reproduces the
xorshift64-seed-42 generator from `tests/bench.rs`; the second walks
the samply profile + presymbolicate sidecar to print self/inclusive
hot-path tables.
