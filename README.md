# DERNA2

**Full Pareto fronts of modular mRNA designs with UTRs.**

Computational mRNA design optimizes a coding sequence (CDS) for two criteria:
minimum free energy (MFE), which approximates thermodynamic stability, and the
codon adaptation index (CAI), which approximates translation efficiency.
Existing methods combine the two in a weighted sum, which misses most of the
Pareto front; they do not exclude sequence motifs such as restriction sites
that are undesirable for synthesis and cloning; and a CDS designed in isolation
may base-pair with the 5′ and 3′ untranslated regions (UTRs) and hinder
translation.

DERNA2 is an exact algorithm for the **UTR-Aware Pareto Optimal RNA Design**
(UTR-PORD) problem: given a protein sequence, fixed 5′ and 3′ UTRs and a set of
forbidden motifs, it computes the Pareto front of the motif-free coding
sequences encoding that protein whose full mRNA has an MFE structure with no
CDS–UTR base pairs. A design with that property is called **modular**, and the
energy difference between a design's constrained and free folds is its **gap**.
Without UTRs the complete motif-free front is computed in polynomial time; with
UTRs the worst-case running time is exponential.

![DERNA2 finds all Pareto-optimal mRNAs without CDS–UTR base pairs or forbidden motifs](docs/overview.png)

**(a)** Human glycophorin-E (GLPE) with the UTRs of human β-globin and the
XhoI site `CUCGAG` forbidden. **(b)** Existing methods return only designs on
the convex hull of their fronts and miss most Pareto-optimal designs, none of
them modular. **(c)** A design computed without the UTRs forms 13 CDS–UTR base
pairs once they are added. **(d)** A method that penalizes such pairs still
forms 3 of them and spells the XhoI site. **(e)** The DERNA2 design has
neither.

## Install

DERNA2 is written in Rust. You need a Rust toolchain (1.95 or newer) and git.

1. Install Rust with [rustup](https://rustup.rs):

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

   The repository pins its toolchain in `rust-toolchain.toml`, so rustup
   fetches the right version on the first build.

2. Clone and build:

   ```bash
   git clone https://github.com/elkebir-group/derna2.git
   cd derna2
   cargo build --release -p derna
   ```

   The binary is `target/release/derna`. To put it on your `PATH`, run
   `cargo install --path packages/derna`.

3. Optionally, run the tests:

   ```bash
   cargo test --release --workspace
   ```

## The `derna` command

| Command | What it does |
|---|---|
| `derna design` | the best design at each given trade-off weight |
| `derna front` | the exact Pareto front of (MFE, CAI) |
| `derna check` | certify a result's points against their free folds |
| `derna pool` | join certified per-stop-codon fronts |
| `derna show` | render, filter, plot or export a result file |
| `derna pick` | choose one design and write it as FASTA |

Every energy is in kcal/mol. Results are JSONL files that record how they were
made, so `show`, `check` and `pick` can be run on them later.

## Examples

All examples below use a 20-residue test protein and a pair of 25-nt UTRs, and
each finishes in seconds. The outputs are real.

```bash
alias derna=./target/release/derna
P=packages/derna-codon-dp/tests/data/GLPE_HUMAN_20.fasta
U5=ACAUUUGCUUCUGACACAACUGUGU
U3=GCUCGCUUUCUUGCUGUCCAAUUUC
```

### One design at one trade-off weight

`--lambda` runs from 0 (only MFE matters) to 1 (only CAI matters), and takes a
list.

```bash
derna design --protein $P --lambda 0.5 --out design.jsonl
```

```text
[1/1] λ=0.5 mfe=-27.80 cai=0.8498 (0.07s)
1 design(s), unconstrained — no check to give

wrote design.jsonl:
design — 20 residue(s) of GLPE_HUMAN_20aa_cut
  5′ UTR absent   3′ UTR absent   stop none
  constraint none   access off
  codon usage human   derna 0.1.0   0.08s   peak 5 MB
  1 point(s), 1 on the hull, not checked

       mfe       cai     gc      gap  modular   hull    lambda  status
    -27.80    0.8498   0.52        —        —    yes  0.500000  —
```

### The whole Pareto front, with UTRs

With UTRs present, the default constraint forbids the coding sequence from
pairing with either UTR, and `front` certifies each point against the
molecule's own free fold. A point whose free MFE equals its constrained MFE is
**modular** and its gap is 0.

```bash
derna front --protein $P --utr5 $U5 --utr3 $U3 --out front.jsonl
```

```text
19 point(s), 9 on the hull, 8 gap(s), 11 sweep fold(s) (2.39s), spanning 11.2 kcal/mol
13 of 19 point(s) modular

wrote front.jsonl:
front — 20 residue(s) of GLPE_HUMAN_20aa_cut
  5′ UTR 25 nt   3′ UTR 25 nt   stop free
  constraint 5-cds,cds-3   access off
  codon usage human   derna 0.1.0   2.39s   peak 14 MB
  19 point(s), 9 on the hull, 13 of 19 modular

       mfe       cai     gc      gap  modular   hull    lambda  status
    -34.40    0.7657   0.49     0.70       no    yes         —  —
    -34.20    0.8041   0.49     0.70       no    yes         —  —
    -34.00    0.8092   0.47     0.00      yes     no         —  certified
    -33.90    0.8301   0.47     0.00      yes     no         —  certified
    -33.80    0.8498   0.47     0.00      yes    yes         —  certified
       ...
    -29.10    0.9719   0.49     0.00      yes    yes         —  certified
    -27.10    0.9857   0.50     2.50       no    yes         —  —
    -25.20    0.9860   0.50     6.10       no     no         —  —
    -24.90    0.9957   0.51     4.90       no    yes         —  —
    -23.20    1.0000   0.51    11.10       no    yes         —  —
```

Omitting `--stop` leaves the stop codon to the search (`stop free` above);
`--stop UAA` fixes it and `--stop none` asks for none.

### The front of modular designs only

`--modular` walks the front of designs that are modular by construction, rather
than certifying afterwards. It is the expensive mode, so it takes a budget;
here it hits the 60-second limit with one region of the front still open.

```bash
derna front --protein $P --utr5 $U5 --utr3 $U3 --modular --max-secs 60 --out modular.jsonl
```

```text
walking the front of 20 residue(s) over [1.1920928955078125e-7, 1 − 1.1920928955078125e-7], forbidding 5-cds,cds-3, on a grid of 17 weights with 1 thread(s)
19 point(s), 9 on the hull, 8 gap(s), 11 sweep fold(s) (2.13s), spanning 11.2 kcal/mol
walking the modular front from 19 constrained point(s) within 60 s
16 point(s) on the modular front, 16 certified, from 13 seed(s); stopped by the time budget with 1 region(s) open (63.37s)
wrote 16 point(s), by gap 0.00×16 (0.05s)
```

Add `--threads N` to parallelize it, and `--checkpoint`/`--resume` to continue
a walk that ran out of budget.

### Excluding motifs

`--motif` forbids a subsequence anywhere in the coding sequence; repeat it or
pass a file of patterns, and scope one with `--motif-window`.

```bash
derna front --protein $P --utr5 $U5 --utr3 $U3 --motif GCGGC --out motif.jsonl
```

```text
walking the front of 20 residue(s) avoiding GCGGC over the coding sequence, forbidding 5-cds,cds-3, with 1 thread(s)
24 point(s), 8 on the hull, 7 gap(s), 3 hull fold(s) (2.63s)
11 of 24 point(s) modular
```

This protein's CDS is only 60 nt, so it is too short to spell a six-base
restriction site by chance: `--motif GAAUUC` returns a front identical to the
unconstrained one. `GCGGC` is used here because this front does spell it, so
the constraint actually removes designs. On real coding sequences the motifs of
interest are restriction and cloning sites, as in the figure above.

### Reading a result back

`show` re-renders a result file and filters it, without folding anything.

```bash
derna show front.jsonl --modular --fasta | head -4
```

```text
>GLPE_HUMAN_20aa_cut point=3 mfe=-34.00 cai=0.8092 gc=0.47
ACAUUUGCUUCUGACACAACUGUGUAUGUAUGGCAAGAUCAUCUUUGUCCUGCUGCUGAGCGGGAUAGUGUCGAUCUCUGCCAGCUAAGCUCGCUUUCUUGCUGUCCAAUUUC
>GLPE_HUMAN_20aa_cut point=4 mfe=-33.90 cai=0.8301 gc=0.47
ACAUUUGCUUCUGACACAACUGUGUAUGUAUGGCAAGAUCAUCUUUGUGCUGCUGCUGAGCGGCAUAGUGUCGAUCUCUGCCAGUUAGGCUCGCUUUCUUGCUGUCCAAUUUC
```

`--min-gc`, `--max-gc` and `--max-homopolymer` filter on sequence metrics, and
`--sequences` adds the per-point GC, U and homopolymer columns.

### Plotting a front

`--plot` writes an SVG with CAI on the x axis and MFE on the y axis, marking
hull points and distinguishing modular from non-modular ones. It is titled
with the record's name, or with `--plot-title` instead.

```bash
derna show front.jsonl --plot front.svg --plot-title "GLPE-20 with β-globin UTRs"
```

![The Pareto front of the example](docs/front.svg)

### Picking one design to order

`pick` applies one selection rule and writes a single record. `--dna` spells T
for U and `--cds-only` drops the UTRs.

```bash
derna pick front.jsonl --best-cai --modular
```

```text
>GLPE_HUMAN_20aa_cut point=15 mfe=-29.10 cai=0.9719 gc=0.49
ACAUUUGCUUCUGACACAACUGUGUAUGUAUGGCAAGAUCAUCUUCGUGCUGCUGCUGAGCGGCAUCGUGAGCAUCUCUGCCAGCUAAGCUCGCUUUCUUGCUGUCCAAUUUC
```

```bash
derna pick front.jsonl --best-mfe --min-cai 0.9 --cds-only --dna
```

```text
>GLPE_HUMAN_20aa_cut point=9 mfe=-31.70 cai=0.9108 gc=0.47
ATGTATGGCAAGATCATCTTTGTGCTGCTGCTGAGCGGCATAGTGAGCATCTCTGCCAGCTAA
```

`--knee` takes the point farthest from the chord joining the front's two ends,
and `--json` writes the point's JSONL row instead of FASTA. A rule that selects
nothing exits non-zero and names the filter that emptied the set.

### Several proteins at once

`--protein` also takes a multi-record FASTA. `--out` then names a directory,
and each record is written to `<id>.jsonl` inside it.

```bash
derna design --protein two.fasta --lambda 0.5 --out batch/
```

```text
...
wrote batch/HBB20.jsonl:
design — 20 residue(s) of HBB20
  5′ UTR absent   3′ UTR absent   stop none
  constraint none   access off
  codon usage human   derna 0.1.0   0.03s   peak 5 MB
  1 point(s), 1 on the hull, not checked

       mfe       cai     gc      gap  modular   hull    lambda  status
    -44.10    0.7923   0.57        —        —    yes  0.500000  —

batch of 2 record(s): GLPE20 ok; HBB20 ok
```

A record that fails is reported and the batch continues, with a non-zero exit
at the end; rerunning with `--checkpoint` skips the records already finished.

### Certifying a result computed elsewhere

`front` certifies its own points, so `check` is for results produced with
`--no-check`, and for `design` output.

```bash
derna front --protein $P --utr5 $U5 --utr3 $U3 --no-check --out raw.jsonl
derna check raw.jsonl --out checked.jsonl
```

```text
13 of 19 point(s) modular

wrote checked.jsonl:
check — 20 residue(s) of GLPE_HUMAN_20aa_cut
  5′ UTR 25 nt   3′ UTR 25 nt   stop free
  constraint 5-cds,cds-3   access off
  codon usage human   derna 0.1.0   0.09s   peak 3 MB
  19 point(s), 9 on the hull, 13 of 19 modular
```

`check` saves progress as it goes and resumes where it stopped, since folding a
large front can take hours.

The full reference for every command and flag, what each one reports and what a
run costs is in [docs/guides/derna-cli.md](docs/guides/derna-cli.md).

## Repository layout

| Package | What it is |
|---|---|
| `packages/rna-mfe` | Turner 2004 energy evaluation and Zuker MFE folding (dangles = 0) |
| `packages/rna-linearfold` | LinearFold-style 5′→3′ beam-search folding over `rna-mfe` |
| `packages/derna-codon-dp` | Codon-design dynamic programs over the codon lattice |
| `packages/derna-lib` | The exact front, region and motif constraints, and certification |
| `packages/derna` | The `derna` command-line tool |

## Acknowledgements

- The Turner 2004 energy parameters in `packages/rna-mfe/data/rna_turner2004.par`
  are the parameter file distributed with the
  [ViennaRNA Package](https://www.tbi.univie.ac.at/RNA/), by the authors and the
  Institute for Theoretical Chemistry of the University of Vienna.
- `rna-linearfold` implements the LinearFold algorithm (Huang et al.,
  *Bioinformatics* 35, 2019).
- The codon automaton follows the construction in LinearDesign (Zhang et al.,
  2023).

We used Claude (Anthropic) to support code development. The authors reviewed
and tested all AI-assisted code and analyses, and take full responsibility for
the content.

## Citation

See [CITATION.cff](CITATION.cff).

## Authors

Xinyu Gu, Mark D. M. Leiserson and Mohammed El-Kebir.
