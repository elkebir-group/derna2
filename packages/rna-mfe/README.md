# `rna-mfe`

Turner 2004 RNA minimum-free-energy evaluator and Zuker DP folder at `dangles=0`.

This crate is a Rust implementation validated against ViennaRNA v2.7.0
as the reference oracle (no `libRNA` runtime dep). Two public entry
points: `fold(seq) -> (structure, energy)` runs the Zuker DP;
`evaluate(seq, structure) -> energy` scores any user-provided structure
including ones with interior loops > 30 nt that the DP would never
produce.

## Toolchain

The Rust toolchain is pinned in the repository-root `rust-toolchain.toml` (currently
`1.95.0` stable). Install with the canonical rustup.rs installer:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
```

This installs `cargo`, `rustc`, and friends into `$HOME/.cargo/bin`,
which the rustup installer adds to your shell profile automatically.
If you skip that step (or are running in a shell without your
profile sourced), add it to `PATH` yourself:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
```

## Common commands

```bash
cargo build --release --workspace
cargo test --release -p rna-mfe
cargo fmt --all -- --check
cargo clippy --release --workspace --all-targets -- -D warnings
```

## Performance

The bench harness measures wall-clock per call (median of N=5 trials + p25/p75 IQR) across a length ladder of 100–3000 nt. Subprocess wrappers for ViennaRNA (`RNAfold -d 0 --noPS`, `RNAeval -d 0`) are invoked the same way for every length, with the sequence on stdin. Sequences are synthesised with a deterministic xorshift64 RNG (seed 42).

Run the bench and write `tests/data/perf_baseline.json`:

```bash
cargo test --release -p rna-mfe --test bench -- --ignored --nocapture
```

Baseline JSON path: `tests/data/perf_baseline.json`. The JSON is the source of truth for cross-version perf diffs; numbers are not pasted into prose here.

**Subprocess floor:** ~2.6 ms per call on Apple M-series (RNAfold and RNAeval). Ratios at lengths < ~400 nt are floor-distorted — the subprocess overhead is a significant fraction of the fold time. Ratios at 400 nt and above reflect genuine compute differences.

**Evaluator note:** `rust_mfe::evaluate` is measured with EVAL_BATCH=100 calls per trial (divided to per-call ms) because it is fast enough at small lengths to be dominated by `Instant::now()` overhead when called once. Subprocess `RNAeval` calls are not batched.

The performance target is `fold` at or below ViennaRNA's wall clock at 800 and 3000 nt; see [`docs/perf-profile.md`](docs/perf-profile.md) for the profile.
