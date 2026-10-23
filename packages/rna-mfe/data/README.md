# rna-mfe data

Static data files used by the `rna-mfe` Rust crate.

## `rna_turner2004.par`

A byte-identical copy of the upstream `rna_turner2004.par` that ships
with ViennaRNA. **No patch.** The file is vendored so the param-file
version is pinned to the repo (rather than to whatever each developer's
local Vienna install happens to provide), and the recorded SHA-256 acts
as an upgrade tripwire — any drift on a Vienna upgrade should be a
deliberate review, not a silent change.

This crate owns the canonical copy of the parameter file.

ViennaRNA is the validation oracle for the Rust evaluator, not a runtime
dependency: this crate parses the same parameter file Vienna does and is
validated against Vienna's own energies on the fixture corpus in
`tests/data/mfe_fixtures.json`.

### Provenance

| Field | Value |
|-------|-------|
| Upstream path | `/opt/homebrew/share/ViennaRNA/rna_turner2004.par` (Homebrew `brewsci/bio/viennarna`) |
| ViennaRNA version | `RNAfold 2.7.0` |
| Upstream SHA-256 | `0c671a7455f5686bc160fc655194b11888594ab7d506626cdad1470bcadd17d9` |
| Lines | 8142 |

### Contract

This file is byte-identical to the upstream `rna_turner2004.par`.

```
$ diff /opt/homebrew/share/ViennaRNA/rna_turner2004.par packages/rna-mfe/data/rna_turner2004.par
$ shasum -a 256 packages/rna-mfe/data/rna_turner2004.par
0c671a7455f5686bc160fc655194b11888594ab7d506626cdad1470bcadd17d9  packages/rna-mfe/data/rna_turner2004.par
```

If a Vienna upgrade changes the upstream file's bytes, that is a
deliberate-review event, not a silent re-pin: re-validate the Rust parser against the new
file (the round-trip test in `src/params/` is the first-line gate),
then update the SHA above.

### Runtime use

`Params::load_default()` in `src/params/` reads this file at compile
time via `include_str!` and parses it into static lookup tables, which
the per-loop energy evaluation reads.

### Maintenance

On a ViennaRNA upgrade:

1. `shasum -a 256 $(brew --prefix viennarna)/share/ViennaRNA/rna_turner2004.par` — if the SHA matches the table above, no change is needed.
2. If the SHA changed, re-validate against the new upstream by running the round-trip test in `src/params/dump.rs` against the new file. Bit-for-bit copy the new upstream file in and update the SHA above only after the round-trip passes.
