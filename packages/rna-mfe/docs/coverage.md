# Fixture coverage of Turner 2004 sections

This document describes which Turner 2004 sections each `dangling_5_prime=false`
record in [`packages/rna-mfe/tests/data/mfe_fixtures.json`](../tests/data/mfe_fixtures.json)
exercises, plus a per-section coverage summary. Generated from a mechanical
decomposition of every `dangling_5_prime=false` record in the fixture file,
cross-referenced with the section list of the Turner 2004 parameter file
(`data/rna_turner2004.par`).

## Source

- **Fixtures.** `packages/rna-mfe/tests/data/mfe_fixtures.json` holds
  **94 records total** (47 unique structures × 2 dangle settings). After
  filtering to `dangling_5_prime=false`, **47 records** are in scope.
- **Parameter file.** `rna_turner2004.par` from ViennaRNA 2.7.0, vendored
  byte-identical at `packages/rna-mfe/data/rna_turner2004.par`, SHA-256-pinned.
- **Decomposition rules.** Standard Vienna-style parse: each base pair `(i,j)`
  closes one loop, classified by the number of immediately-enclosed pairs
  (0 → hairpin; 1 → stack/bulge/internal by unpaired-side counts; ≥2 →
  multi-loop). The exterior loop is the parent of the outermost pairs.
  Stack pairs at dangles=0 contribute one `stack` lookup per pair-pair
  adjacency. The Rust-side decomposer that productionizes this lives at
  [`packages/rna-mfe/src/decompose.rs`](../src/decompose.rs); see also the
  integration test [`tests/whole_structure.rs`](../tests/whole_structure.rs).

## Per-fixture coverage

One row per fixture. `sections used` is the union over every loop in the
structure, not only the named "primary" motif — the closing stem stacks
count, the hairpin at the bottom of every fixture counts, and so on. AU/GU
end penalties surface as `Misc.terminal_AU`. Special-sequence hairpin
bonuses (`Triloops`/`Tetraloops`/`Hexaloops`) are listed when the loop
sequence would *trigger a lookup* even if the actual lookup misses. The
`Triloops` / `Tetraloops` / `Hexaloops` bonus tables each have at least
one fixture that hits a non-zero entry (see
[`§G3` resolution](#g3-special-hairpin-bonus-tables)).

| #  | motif         | len | closure | E (kcal/mol) | sections used |
|---:|---------------|----:|--------|-------------:|---------------|
| 0  | hairpin       | 9   | GC     | -1.20        | `stack`, `hairpin[3]`, `Triloops` |
| 1  | hairpin       | 12  | GC     | -5.40        | `stack`, `hairpin[4]`, `mismatch_hairpin`, `Tetraloops` |
| 2  | hairpin       | 12  | GC     | -6.80        | `stack`, `hairpin[4]`, `mismatch_hairpin`, `Tetraloops` |
| 3  | hairpin       | 12  | GC     | -5.40        | `stack`, `hairpin[4]`, `mismatch_hairpin`, `Tetraloops` |
| 4  | hairpin       | 14  | GC     | -9.20        | `stack`, `hairpin[4]`, `mismatch_hairpin`, `Tetraloops` |
| 5  | hairpin       | 14  | GC     | -7.90        | `stack`, `hairpin[4]`, `mismatch_hairpin`, `Tetraloops` |
| 6  | hairpin       | 13  | AU     | -6.10        | `stack`, `hairpin[3]`, `Triloops`, `Misc.terminal_AU` |
| 7  | hairpin       | 15  | AU     | -6.30        | `stack`, `hairpin[3]`, `Triloops`, `Misc.terminal_AU` |
| 8  | hairpin       | 15  | AU     | -6.20        | `stack`, `hairpin[3]`, `Triloops`, `Misc.terminal_AU` |
| 9  | hairpin       | 13  | GU     | -5.40        | `stack`, `hairpin[3]`, `Triloops`, `Misc.terminal_AU` |
| 10 | hairpin       | 15  | GC     | -10.40       | `stack`, `hairpin[3]`, `Triloops` |
| 11 | bulge         | 16  | GC     | -7.30        | `stack`, `bulge[1]`, `hairpin[3]`, `Triloops` |
| 12 | bulge         | 17  | GC     | -9.60        | `stack`, `bulge[1]`, `hairpin[4]`, `mismatch_hairpin`, `Tetraloops` |
| 13 | bulge         | 15  | GC     | -5.30        | `stack`, `bulge[1]`, `hairpin[4]`, `mismatch_hairpin`, `Tetraloops` |
| 14 | bulge         | 17  | GC     | -8.60        | `stack`, `bulge[1]`, `hairpin[4]`, `mismatch_hairpin`, `Tetraloops` |
| 15 | bulge         | 19  | GC     | -8.30        | `stack`, `bulge[2]`, `hairpin[3]`, `Triloops` |
| 16 | bulge         | 20  | GC     | -7.90        | `stack`, `bulge[3]`, `hairpin[3]`, `Triloops` |
| 17 | bulge         | 20  | GC     | -3.80        | `stack`, `bulge[5]`, `hairpin[3]`, `Triloops` |
| 18 | bulge         | 16  | AU     | -5.60        | `stack`, `bulge[1]`, `hairpin[3]`, `Triloops`, `Misc.terminal_AU` |
| 19 | bulge         | 17  | GU     | -7.10        | `stack`, `bulge[1]`, `hairpin[4]`, `mismatch_hairpin`, `Tetraloops`, `Misc.terminal_AU` |
| 20 | internal_loop | 17  | GC     | -6.90        | `stack`, `int11`, `hairpin[3]`, `Triloops` |
| 21 | internal_loop | 19  | GC     | -6.50        | `stack`, `int22`, `hairpin[3]`, `Triloops` |
| 22 | internal_loop | 19  | GC     | -5.50        | `stack`, `interior[4]`, `mismatch_interior_1n`, `NINIO`, `hairpin[3]`, `Triloops` |
| 23 | internal_loop | 21  | GC     | -4.60        | `stack`, `interior[6]`, `mismatch_interior`, `NINIO`, `hairpin[3]`, `Triloops` |
| 24 | internal_loop | 19  | AU     | -7.70        | `stack`, `int11`, `hairpin[3]`, `Triloops`, `Misc.terminal_AU` |
| 25 | internal_loop | 19  | GU     | -7.70        | `stack`, `int11`, `hairpin[3]`, `Triloops`, `Misc.terminal_AU` |
| 26 | multi_loop    | 35  | GC     | -14.00       | `stack`, `ML_params`, `hairpin[3]`, `Triloops` |
| 27 | multi_loop    | 35  | mixed  | -12.30       | `stack`, `ML_params`, `Misc.terminal_AU`, `hairpin[3]`, `Triloops` |
| 28 | long_form     | 234 | AU     | -89.40       | `stack`, `hairpin[4]`, `hairpin[8]`, `mismatch_hairpin`, `Tetraloops`, `bulge[1]`, `bulge[2]`, `bulge[3]`, `int11`, `int21`, `interior[4]`, `interior[5]`, `interior[6]`, `mismatch_interior`, `mismatch_interior_1n`, `mismatch_interior_23`, `NINIO`, `ML_params`, `Misc.terminal_AU` |
| 29 | hairpin       | 13  | GU     | -3.20        | `stack` (UG↔CG, UA↔CG, GC↔UA, CG↔GC), `hairpin[3]`, `Triloops`, `Misc.terminal_AU` (closing CG → no fire; outer UG fires on exterior) — §G1.a |
| 30 | hairpin       | 15  | GU     | -5.30        | `stack` (GU↔AU, UA↔CG, GC↔UA, CG↔GC), `hairpin[3]`, `Triloops`, `Misc.terminal_AU` (outer GU exterior) — §G1.b |
| 31 | hairpin       | 15  | AU     | -4.40        | `stack` (AU↔AU, UA↔UA, UA↔CG, GC↔CG), `hairpin[3]`, `Triloops`, `Misc.terminal_AU` (outer UA) — §G1.c |
| 32 | hairpin       | 12  | GC     | -3.70        | `stack`, `hairpin[4]`, `mismatch_hairpin[AU,A,A]`, `Tetraloops` — §G2.a (AU-inner size-4 hairpin) |
| 33 | hairpin       | 12  | GC     | -3.70        | `stack`, `hairpin[4]`, `mismatch_hairpin[UA,A,A]`, `Tetraloops` — §G2.UA (UA-inner size-4 hairpin) |
| 34 | internal_loop | 21  | GC     | -6.40        | `stack`, `interior[5]`, `mismatch_interior_23[AU,A,U] + mismatch_interior_23[CG,A,A]` (AU on int-loop outer), `NINIO`, `hairpin[3]`, `Triloops` — §G2.c (2,3) AU int-closure |
| 35 | internal_loop | 21  | GC     | -7.40        | `stack`, `interior[5]`, `mismatch_interior_23` (AU on int-loop outer), `NINIO`, `hairpin[3]`, `Triloops` — §G2.d (3,2) AU int-closure |
| 36 | hairpin       | 13  | GC     | -6.50        | `stack`, **`Triloops` HIT (`CAACG`)** — size-table / mismatch / terminal_AU all skipped — §G3.a |
| 37 | hairpin       | 12  | GC     | -6.70        | `stack`, **`Tetraloops` HIT (`CCAAGG`)** — size-table / mismatch all skipped — §G3.b |
| 38 | hairpin       | 14  | GC     | -6.20        | `stack`, **`Hexaloops` HIT (`ACAGUACU`)** — §G3.c |
| 39 | internal_loop | 19  | GC     | -7.90        | `stack`, `int11[GU,GC,A,A]` (GU on int-loop outer), `hairpin[3]`, `Triloops`, `Misc.terminal_AU` — §G4.a |
| 40 | internal_loop | 20  | GC     | -6.40        | `stack`, `int21` (AU on int-loop outer), `hairpin[3]`, `Triloops` — §G4.b |
| 41 | internal_loop | 21  | GC     | -8.40        | `stack`, `int22[CG,CG,A,C,A,C]` (non-AA mismatch cell), `hairpin[3]`, `Triloops` — §G4.c |
| 42 | hairpin       | 15  | GC     | -7.80        | `stack`, `hairpin[5]`, `mismatch_hairpin` — §G5.a |
| 43 | hairpin       | 17  | GC     | -7.50        | `stack`, `hairpin[7]`, `mismatch_hairpin` — §G5.b |
| 44 | multi_loop    | 49  | GC     | -15.40       | `stack`, `ML_params` (3 branches), `hairpin[3]`, `hairpin[4]`, `Triloops` — §G6.a |
| 45 | multi_loop    | 63  | mixed  | -16.30       | `stack`, `ML_params` (4 branches), `Misc.terminal_AU` (multiple AU branch caps), `hairpin[3]` × 4, `Triloops` — §G6.b |
| 46 | bulge         | 16  | GC     | -2.80        | `stack` (AU/CG and AU/CG around the bulge), `bulge[1] + stack[AU,CG]`, `hairpin[3]`, `Triloops` — §OQ3 size-1 bulge |

### Long-form fixture (#28) — loop inventory

The 234-nt fixture is the only multi-motif single record. Its full loop
inventory (excluding the 62 individual stack lookups, which collectively
exercise the same `stack` table the small motifs do):

| Loop kind   | Count | Specifics |
|-------------|------:|-----------|
| exterior    | 1     | 1-branch (AU) |
| stack       | 62    | mixed pair types (see [§Stack table coverage](#stack-table-detail)) |
| hairpin     | 2     | size 8 (closing GC, loop `CAGCACCA`); size 4 (closing CG, loop `CAUG`) |
| bulge       | 7     | sizes 1×4, 2×2, 3×1; mixed closing/inner pairs |
| internal    | 10    | shapes (1,1)×3, (1,2)×1, (1,3)×1, (2,1)×2, (2,4)×1, (3,2)×1; closures span GC/CG/AU/UA/GU |
| multi       | 1     | branches=2, unpaired=10, closing AU, branch pairs UA + AU |

This is **the only fixture that exercises** `int21`, `mismatch_interior_23`,
`interior[5]`, and `hairpin[8]`. It is also the only fixture with a non-`GC`
inner pair feeding `int11` (carries `(GC,CG)` and `(AU,CG)` versions).

## Per-section coverage summary

Coverage tags follow the spec:
- **well-covered** — ≥3 fixtures exercise this section across varied parameters
- **thin** — 1–2 fixtures, OR ≥3 fixtures but with very narrow parameter coverage
- **gap** — no fixture exercises this section
- **n/a** — genuinely not needed at dangles=0 (the folder and the codon DP built on it run at dangles=0 only, because a codon-level cell index does not record the flanking codon that a dangle energy would read)

| Section | Coverage | Fires | Notes |
|---|---|--:|---|
| `stack` | well-covered | 211 + ~80 stack-loop fires across 26 of the 36 base-pair × base-pair combinations | Five non-canonical/`NN` rows untouched by design (no fixture has `NN`). Heavy GC/GC bias (~36% of all fires); see [§Stack table coverage](#stack-table-detail). |
| `mismatch_hairpin` | well-covered | 13 fixtures (12 small + long_form) | Closing pairs covered: GC, CG, AU (#32), UA (#33). **Outstanding:** no GU-closed size-≥4 hairpin — couldn't be landed as Vienna's MFE; see [§G2](#g2-mismatch-tables-closing-pair-coverage). |
| `mismatch_interior` | thin | 2 fixtures (#23 + long_form) | Generic-interior path; (2,4) and (3,2) pairs only. |
| `mismatch_interior_1n` | thin | 2 fixtures (#22 + long_form) | (1,3) and (1,3)/(3,1) family. See [Open Q §OQ1](#oq1-mismatch_interior_1n-cutoff) for the cutoff resolution. |
| `mismatch_interior_23` | well-covered | 3 fixtures (long_form + #34 + #35) | (3,2) at GC closure plus (2,3) and (3,2) at AU closure of int-loop. |
| `mismatch_multi` | n/a | — | dangles=2 only. |
| `mismatch_exterior` | n/a | — | dangles=2 only. |
| `dangle5`, `dangle3` | n/a | — | dangles=2 only. |
| `int11` | well-covered | 5 fixtures (#20, #24, #25, long_form, #39), 7 fires across 4 distinct (closing, inner) pairs | Closing pairs touched: GC, CG, AU, GU. Inner pair almost always GC. |
| `int21` | thin (1 small + long_form) | 2 fixtures (long_form, #40), 5 internal fires | #40 adds a (1,2) shape with AU on the int-loop outer. The 6×6×4×4×6 cell space is largely uncovered by design; surface-up specific cells if a bug emerges. See [§G4](#g4-int21-and-int22-undertested). |
| `int22` | thin (1 small + #21) | 2 fixtures (#21 GC/GC AA mismatches, #41 CG/CG AC mismatches) | Two distinct cells exercised; non-AA mismatch indexing now disambiguated from AA-only. |
| `hairpin` (size table) | well-covered | sizes 3, 4, 5, 6, 7, 8 | 24+ size-3 + 12 size-4 + 1 size-5 + 1 size-6 + 1 size-7 + 1 size-8 fires; size 8 still long-form only. **Outstanding:** sizes ≥ 9 + the `lxc` extrapolation path (size > 30). See [§G5](#g5-hairpin-size-coverage). |
| `bulge` (size table) | partial | sizes 1 (11), 2 (3), 3 (2), 5 (1) | #46 adds the AU/CG-flanked size-1 bulge cell (resolves [OQ3](#oq3-bulge1--stack-superposition)). **Outstanding:** size 4 and ≥6 untested. |
| `interior` (generic size table) | thin (size-5 ×3) | sizes 4, 5, 6 only — 2/3/2 fires | #34, #35 add size-5 fires at AU closure. **Outstanding:** sizes ≥7 untested. |
| `ML_params` | well-covered (2-/3-/4-branch) | 5 fixtures (#26, #27, long_form, #44 3-branch, #45 4-branch) | Branch counts 2, 3, 4 all exercised — the `init` vs `per_branch * (1+n)` netting bug class is caught. |
| `NINIO` | thin | 3 fixtures (#22, #23, long_form) plus 2 more fires from #34, #35 | Asymmetric internal loops only. |
| `Misc` (terminal AU/GU penalty) | well-covered | 17 fixtures with at least one AU/GU end | Includes outer GU/UA closures (#29, #30, #31), int-loop AU outers (#34, #35), multi-loop AU branches (#44, #45). |
| `Misc` (other constants — `lxc`, init terms) | thin / contextual | depends on definition | Size-dependent constants used in interior-loop extrapolation (`lxc`, the logarithmic factor used when interior size > 30). No fixture has a size-≥31 interior loop, so the `lxc` path is **gap** — low priority unless folder layers exercise very long sequences with degenerate loops. |
| `Tetraloops` | covered | 12 size-4 hairpin fires, **1 hit** (#37: `CCAAGG`) | #37 wraps the strongest Tetraloop bonus entry in a 4-bp GC stem. |
| `Triloops` | covered | 24+ size-3 hairpin fires, **1 hit** (#36: `CAACG`) | All other corpus triloops are `AAA`; #36 is the only non-`AAA` triloop. |
| `Hexaloops` | covered | 1 fire, **1 hit** (#38: `ACAGUACU`) | First and only size-6 hairpin in the corpus; wraps the strongest Hexaloop bonus. |
| All `*_enthalpies` | n/a | — | Folding runs at 37°C only. |

<a id="stack-table-detail"></a>
### Stack table detail

`stack` is a 7×7 table indexed by `(closing_pair, inner_pair)` over
`{CG, GC, GU, UG, AU, UA, NN}`. The 21 distinct entries the corpus
exercises (out of 36 canonical pair × pair combinations):

| Outer\Inner | CG | GC | GU | UG | AU | UA |
|---|---:|---:|---:|---:|---:|---:|
| **CG** | 10 | 13 | 2  | -  | 7  | 4  |
| **GC** | 25 | 82 | -  | 2  | 2  | 2  |
| **GU** | 2  | 5  | -  | 1  | -  | -  |
| **UG** | -  | 2  | -  | -  | 1  | -  |
| **AU** | 2  | 6  | -  | -  | 1  | 6  |
| **UA** | -  | 6  | -  | -  | -  | -  |

Empty cells (`-`) are gap entries. The GC↔GC homo-stack dominates (82
fires, 39% of all stack lookups), reflecting the fixture-design pattern
of "GC-rich stem to make energy values robust to noise." The diagonal NN
row in `rna_turner2004.par` (used for non-canonical mismatches) has no
corpus fires; that's expected since fixtures all use canonical pairs.

For an evaluator, the takeaway is: every `stack` cell with corpus presence
pins its parameter exactly via fixture comparison. The 15 missing cells
are not validated by fixtures alone — but at dangles=0 with the current
fixture set, the validated subset covers the cells folder layers are most
likely to traverse on RNA secondary structure search of GLPE-78-like
sequences.

## Closed gaps — fixture additions in §G1..§G7

Each named gap below records the Turner region the corpus once
under-covered, the fixture(s) that close it, and any residual outstanding
sub-case. The gap labels (G1..G7) are the addressing scheme used in the
per-fixture coverage column above.

### G1 — Stack table row coverage

Closed by fixtures #29..#31. Each exercises previously-empty stack-table
cells: #29 — `stack[UG,CG]` inside its 5-bp stem; #30 — `stack[GU,AU]`;
#31 — `stack[AU,AU] + stack[UA,UA]`. Five non-empty cells per outer-pair
row remain at one fire, but the `Misc.terminal_AU` cross-stem accounting
between G1 fixtures and the AU-/UA-closed hairpins of §G2 disambiguates
the residual ambiguity.

### G2 — Mismatch tables closing-pair coverage

Closed (partial) by fixtures #32, #33, #34, #35. Fixtures #32 and #33 are
size-4 hairpins with AU and UA closing pairs respectively — they activate
the previously-untouched `mismatch_hairpin[AU,A,A]` and
`mismatch_hairpin[UA,A,A]` cells. Fixtures #34 and #35 add (2,3) and (3,2)
internal loops at AU closure of the int-loop outer pair, exercising
`mismatch_interior_23` with non-GC closure.

**Outstanding:** GU-closed size-≥4 hairpin couldn't be landed naturally —
Vienna's MFE search consistently picks an alternative fold with GU stacked
one position above the bottom of the stem rather than as the
hairpin-closing pair (the Wobble pair loses ~0.3 kcal/mol per position
when at the helix terminus). An `RNAeval`-driven (rather than
`RNAfold`-driven) fixture would be needed if folder layers surface a bug
on the missing cell.

### G3 — Special hairpin bonus tables

Closed by fixtures #36, #37, #38. Fixture #36 wraps the `CAACG` Triloop
entry in a 4-bp GC stem; fixture #37 wraps the `CCAAGG` Tetraloop entry
in a 4-bp stem; fixture #38 wraps the `ACAGUACU` Hexaloop entry in a 4-bp
stem. Each is the first fixture in the corpus to fire a non-zero entry
from its respective table.

Vienna semantics for special hairpins: the bonus value **replaces** the
size-table penalty, the AU/GU `terminal_au_37`, and the `mismatch_hairpin`
lookup entirely — see `include/ViennaRNA/eval/hairpin.h::E_Hairpin` (v2.7.0
lines 376–404; the special_hp branch returns the bonus directly).
Implementation: [`packages/rna-mfe/src/loops/special_hairpin.rs`](../src/loops/special_hairpin.rs).

The `rna_turner2004.par` Tetraloop table has only 16 entries
(`CAACGG`–`CUUUGG`, all C-starting and G-ending), so common loops like
`GAAA` (the GNRA family) do **not** receive a Turner 2004 bonus from this
file. Additional tetraloop bonuses live in `rna_misc_special_hairpins.par`,
which is not loaded by default and is out of scope for the corpus.

### G4 — `int21` and `int22` undertested

Closed by fixtures #39, #40, #41. Fixture #39 is a 1×1 internal loop with
GU on the int-loop outer pair (new `int11[GU,GC,A,A]` cell). Fixture #40
is a 1×2 internal loop with AU on the int-loop outer pair (new `int21`
cell). Fixture #41 is a 2×2 internal loop with non-AA mismatches
(`A,C,A,C` — outer positions filled with deliberately non-canonical
letters so RNAfold can't fold them into an extended stem).

Five out of six int-table dimensions remain under-validated by single-loop
fixtures — the sheer cell count of `int22`'s 6×6×4⁴ = 9216 entries is not
a corpus goal. Surface specific cells via folder-layer comparison if a
bug emerges.

### G5 — Hairpin size coverage

Closed (partial) by fixtures #38, #42, #43. Fixture #38 (Hexaloop, size 6)
closes the size-6 hairpin gap. Fixtures #42 and #43 add size-5 and size-7
hairpins. Sizes ≥ 9 (and the `lxc`-extrapolation path at size > 30)
remain untested — add fixtures here if folder layers exercise them.

### G6 — Multi-loop branch fan-out

Closed by fixtures #44, #45. Fixture #44 is a 3-branch multi-loop and
fixture #45 is a 4-branch multi-loop. The sequence design (alternating
GC-rich and `GAUC`/`UACG` middle branches) deliberately uses *different*
canonical pair patterns at each branch's stem cap so RNAfold doesn't
collapse them into a single nested stem.

With 3- and 4-branch multi-loops in the corpus, an evaluator that nets
out `init` against `per_branch * (1+n)` at n=2 will disagree with at least
one fixture. The `ML_params` formula at dangles=0 is
`init + branches × per_branch + unpaired × per_unpaired`; without ≥3-branch
fixtures, an evaluator that gets `per_branch` or `init` wrong by a constant
offset (and the other one by the negation) silently passes.

### G7 — Corpus size vs. spec text

Resolved. The corpus holds 47 dangles=0 records (94 total).

## Open questions — resolutions

<a id="oq1-mismatch_interior_1n-cutoff"></a>
### OQ1 — `mismatch_interior_1n` cutoff

**Resolution.** Vienna applies `mismatch_interior_1n` for `(1, n)` only
when `n ≥ 3`. The `(1, 2)` and `(2, 1)` cases route to `int21` instead,
so the `1xn` mismatch path is reached only after the 1×1 (`int11`) and
1×2 / 2×1 (`int21`) shapes are dispatched. Source:
`src/ViennaRNA/eval/eval_internal.c::vrna_E_internal` `case 1:` branch
(v2.7.0, lines 141–159). The dispatch lives in
[`packages/rna-mfe/src/loops/internal.rs`](../src/loops/internal.rs) —
see the module docstring's "OQ1 resolution" note for the case-split
table.

The `(2, 3)` / `(3, 2)` boundary (the analogue question for
`mismatch_interior_23`) is resolved by the same `case 2: nl == 3:`
branch — both sides of `nl == 3` (i.e., `(2, 3)` and `(3, 2)`) use the
same `mismatch23I` table; `(2, 2)` uses `int22` and never touches
`mismatch_interior_23`.

### OQ2 — `Misc` block field semantics

**Resolution.** See `packages/rna-mfe/src/params/tables.rs` `MiscParams`
struct. The on-disk Misc block `410 360 50 370` maps to
`[DuplexInit37, DuplexInitdH, TerminalAU37, TerminalAUdH]`, per
ViennaRNA 2.7.0 `src/ViennaRNA/params/io.c` `case MISC:` (lines 675–684).
`lxc37` is **not** parsed from the file (it's a hardcoded `default.c`
constant); Vienna's *dumper* emits a 6-token Misc row that includes
`lxc37` and a trailing `0`, but the on-disk file ships only the four
integers. The round-trip test
`packages/rna-mfe/tests/round_trip.rs::misc_fields_match_vienna_assignment`
pins the field-by-field mapping.

<a id="oq3-bulge1--stack-superposition"></a>
### OQ3 — `bulge[1]` + `stack` superposition

**Resolution.** Fixture #46 (`GGAAAGGAAACCUUCC`, 16 nt) is a size-1 bulge
whose surrounding stack pairs are AU↔CG (outer of the bulge is `AU`,
inner is `GC`), breaking the GC/GC monoculture of the original 10 size-1
bulge fires. An evaluator that hardcodes `stack[GC,CG]` (or that
double-adds the stack) disagrees with the recorded energy.

### OQ4 — Long-form fixture as a regression target vs. a unit-test target

**Resolution.** Both, by construction. The per-loop suite
([`tests/per_loop.rs`](../tests/per_loop.rs)) covers 42 small-motif
fixtures and is the right tool for localizing which loop kind / which
lookup-table cell is buggy. The whole-structure suite
([`tests/whole_structure.rs`](../tests/whole_structure.rs)) covers all
47 dangles=0 fixtures including #28 and acts as the integration acceptance
gate — it surfaces decomposer-edge bugs (e.g., misclassifying a degree-2
loop with `n1==0 && n2==0` as a bulge instead of a stack) and cross-loop
accounting bugs (e.g., a multi-loop's per-stem terminal_AU not summing
with its closing pair's). Both suites assert at exact integer-dekacal
equality.

## How this was built

A Python decomposition script parses each fixture's dot-bracket into a
pair table, identifies each loop type by walking the immediate-children
relation, and tags the Turner 2004 sections by a per-motif rules table
(which loop type reads which parameter section). A companion script
tabulates the `stack` 7×7 cell hit counts and the `int11`/`int21`/`int22`
closing-pair distribution. Both are throwaway analysis tooling and are
not part of the repository.

Candidate fixtures for new corpus cells are iterated through
`RNAfold -d 0 -P rna_turner2004.par` and classified by motif counts to
confirm the candidate would land the intended cell. Once a candidate
clears the probe, it is appended to the fixture-generator script and the
corpus is regenerated. The Rust-side loop decomposer that productionizes
the walk lives at [`packages/rna-mfe/src/decompose.rs`](../src/decompose.rs);
the canonical fixture-parity test is
[`packages/rna-mfe/tests/whole_structure.rs`](../tests/whole_structure.rs)
which iterates every dangles=0 fixture (small + multi-loop + long-form)
and asserts exact integer-dekacal agreement.
