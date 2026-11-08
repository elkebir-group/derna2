# The `derna` command

One binary over the method: design a coding sequence at a given trade-off
weight, walk the exact Pareto front of (MFE, CAI), certify what either
returns against the molecule's own preferred fold, and read any of it back.

```text
derna design --protein P --lambda L[,L...]   the best design at each weight
derna front  --protein P                     the exact Pareto front
derna front  --protein P --modular   the front of designs that pass their check
derna front  --protein P --motif M           the front of designs that spell no motif
derna check  RESULT.jsonl                    certify a result's points
derna pool   FRONT.jsonl... --out PATH       join certified per-stop fronts
derna show   RESULT.jsonl                    render a result file
derna pick   RESULT.jsonl --knee             one design, as FASTA to order
```

Install it from the repository root with `cargo install --path packages/derna`,
or build it in place with `cargo build --release -p derna` and run
`target/release/derna`:

```bash
derna --help
derna design --help
derna --version      # the version a result file records
```

The binary runs on the system allocator. Built with `--features jemalloc`
(`cargo build --release -p derna --features jemalloc`) it runs on jemalloc
instead: the motif gap fold takes about 0.89× the user time and returns the
same fronts, but holds more memory resident, so a walk under `--max-rss-gb` can
stop earlier and return less than the default build would.

Every energy is kcal/mol, on every flag and in every output field. Every
coordinate is a coding-sequence position, 1-based from the start codon, with a
negative index counting back from the end.

## The molecule

The three sequence flags take the same three forms: a path to a one-record
FASTA file, a literal sequence, or `-` for stdin (one flag at a time).

```bash
derna design --protein hbb.fasta --lambda 0.5
derna design --protein MYGKIIFVLL --lambda 0.5
echo MYGKIIFVLL | derna design --protein - --lambda 0.5
```

`--protein` takes the 20 amino acids, case-insensitive. A `*` is refused with
a pointer to `--stop`, so there is one way to ask for a stop codon. A value
that is both an existing file name and a valid sequence is refused rather than
guessed; write `./NAME` for the file.

`--utr5` and `--utr3` are **fixed** sequences (`T` read as `U`). A degenerate
IUPAC code is refused: designed UTR positions are not supported, because the
objective gives such a position no reason to prefer one nucleotide.

`--stop UAA|UAG|UGA` gives the stop codon after the last residue, and
`--stop none` asks for none. **Omitting it with a flank present leaves the
codon to the search**, which designs it as one more slot of the coding
sequence: the lattice's last position chooses among `UAA`, `UAG` and `UGA`
alongside every residue's codon, in the same dynamic program and the same walk.
`--stop free` asks for the same thing by name, for a script that spells out
every flag. Without a flank the default is still no stop codon, because a stop
can only matter through a pair the constraint forbids and a forbidden pair needs
a flank; `--stop free` without one is refused.

Which stop codon is right is a property of the molecule, not a convention, and
it is worth not guessing. Measured on **one** 163-residue molecule, with
everything but the stop codon held fixed and each arm computing its own front:
two of the three stop codons returned no modular design at all and
the third returned 311, and on the plain front of that molecule the Pareto
union over the three held 465 designs of which **none** ended in `UAA`. Which
suffix works is cell-specific — a second molecule is rescued by a different one
— so neither the ordering nor the counts generalise, and that is the point: a
run that pins the stop risks publishing a front that designs of another stop
dominate. Hence the search is given the choice rather than a default.

`--stop each` keeps the three apart instead of choosing between them: one run
walks every stop codon's own front and writes each where a pinned run would,
to the `--out` path with the codon before its extension — `front.jsonl`
becomes `front.UAA.jsonl`, `front.UAG.jsonl` and `front.UGA.jsonl`, and
`--front-out` and `--report` are named the same way. Each file is the front
`--stop UAA|UAG|UGA` writes, design for design, and its header names its stop;
a stop no design reaches is reported and leaves no file. It is for `front` and
needs an `--out` path to name the files from.

The reason to want it is the property the objectives do not see. A front over a
designed stop minimizes over the three codons, so a design that is best under
one of them can be dominated by a design under another, drop off the front, and
take its modularity with it — the union of the three fronts holds
every design, but re-dominating that union is what loses them. Asking for each
stop's front keeps every stop's modular designs, and `--modular`
searches from each one.

**It is not the faster route.** The folds that do not depend on the stop are
taken once — the pass grid's folds and the `λ = 1` credit are the same at every
stop — and that saving is real: measured on one 163-residue molecule with both
flanks and one worker, 755 CPU seconds against 1,320 for three walks over the
same stop-labelled lattice that each take their own grid. But the baseline that
matters to a caller is three `--stop UAA|UAG|UGA` runs, and against those the
same molecule reads **592 CPU seconds against 559, and 795 seconds of wall
against 224, 228 and 261 run independently** — 1.06× the work and 3.0× the
wait. Two costs eat the saving: one lattice carrying every stop's lane is
dearer to fold than a lattice with the stop spelled in, and the completion
bounds a shared fold gives are the least over the three stops, so they prune
the per-stop gap folds less well. The fronts are identical either way.

So reach for `--stop each` when one invocation is simpler to drive than three,
not when you want the answer sooner. For speed, or for anything the flag
refuses, run the three stops separately and join them with
[`derna pool`](#pooling-the-per-stop-fronts) — which is also the only route
that keeps every stop's modular designs, since it filters after
certifying rather than before.

`--stop each` does not yet combine with `--resume`, `--checkpoint`,
`--front-from` or `--motif`, and refuses each by name, naming the pinned route
instead. (Checkpointing is otherwise the default, so a single-stop run needs no
`--checkpoint`; see [What a front costs](#what-a-front-costs).) `derna design`
solves one weight and writes one design, so it does not take it.

A given stop codon is spelled into the molecule between the designed codons and
the 3′ UTR and carries no synonymous choice; a designed one is the coding
sequence's last codon. Either way the stop occupies three **coding**
nucleotides at the same positions, so the constraint classifies every pair
alike: a pair from a codon to the stop is `cds-cds` and allowed, a pair from
the stop to the 3′ UTR is `cds-3`, and a pair from the stop to the 5′ UTR is
`5-cds`. That is the difference from spelling the codon into `--utr3` yourself
— without a constraint the two design the same molecule; under one, the 3′
boundary sits on the other side of the stop.

No stop codon is any part of CAI: all three carry one weight, so the slot adds
nothing to a design's `cai_log`, and the reported `cai` is the geometric mean
over the residues' codons alone. A given-stop front and a designed-stop front
are on one adaptation scale and may be compared directly.

`--pin` positions are coding-sequence positions, addressing the residues'
codons; a negative position counts back from the last residue.

**A sequence constraint covers a designed stop codon.** `--motif` and `--pin`
are enforced over the coding sequence, whose last codon a designed stop is, so
a motif spanning the last residue's codon and the stop is forbidden there too.
A given stop sits in the 3′ flank instead, where a motif is not enforced: a
free-stop motif run is therefore not the union of three pinned ones, but that
union kept to the designs whose whole coding sequence spells no motif.

Every point of a run whose stop was designed names its own in a `stop` field,
since the designs of one such front do not share a suffix; a run with a given
or absent stop carries no per-point field, because the header's applies to all
of them.

`--codon-usage` is a bundled table's name or a path in the
`amino_acid,codon,fraction` format. A result file embeds the whole table it was
scored against, so a reader never has to guess which one a run used. A value
that is neither a bundled name nor a readable file is refused in one message
naming both, since a misspelled name and a mistyped path look the same here.

Five tables are bundled, all from the Kazusa Codon Usage Database. The CDS and
codon counts are the basis each one was tabulated over, and they differ by more
than two orders of magnitude, so they belong beside any CAI a run reports:

| Name | Organism | CDS | Codons |
|---|---|---|---|
| `human` | *Homo sapiens* | not recorded | not recorded |
| `mouse` | *Mus musculus* | 53,036 | 24,533,776 |
| `cho` | *Cricetulus griseus* | 331 | 153,527 |
| `ecoli` | *Escherichia coli* | 8,087 | 2,330,943 |
| `yeast` | *Saccharomyces cerevisiae* | 14,411 | 6,534,504 |

Three things to know before comparing CAI across them.
`cho` rests on 331 coding sequences, the thinnest basis of the five by two
orders of magnitude, and it is the only *Cricetulus griseus* entry the database
has. `ecoli` is the database's species-level entry and claims no strain: the
entry labelled K-12 there holds 14 coding sequences and is not a reference set.
And `human` is an earlier, deliberately frozen snapshot than the other four,
which carry GenBank release 160.0 — it predates the counts the others report,
which is why it has none, and a CAI difference between `human` and one of the
others is therefore partly a difference of corpus vintage rather than of
organism.
`packages/derna-codon-dp/data/PROVENANCE.md` records each table's source entry,
retrieval date and how its numbers were checked.

`--params PATH` loads energy parameters from a file, on `design`, `front` and
`check`. Without it every fold uses the built-in Turner 2004 table. The file
is either in the built-in table's own layout or a ViennaRNA `.par` parameter
file; of a `.par` file, the free energies at 37 °C are read and the
enthalpy, dangle and exterior/multiloop mismatch sections are passed over. A
result file does not record which parameters it was folded with, so a `check`
of a run made with `--params` takes the same `--params`.

### Several proteins at once

`--protein` takes a FASTA file of more than one record on `design` and
`front`. The records run one after another, each with the flags given, and
`--out` then names a **directory**: it is created when absent, and each record
writes `<id>.jsonl` there, where the id is the first word of the record's
header with every character outside `A–Z a–z 0–9 . _ -` written `_`. A
record's progress files take their names from that path as a single run's do
from `--out`; `--report` and `--front-out` become `<id>.report.json` and
`<id>.front.jsonl` beside it.

```bash
derna front --protein panel.fasta --utr5 u5.fa --utr3 u3.fa --out fronts/
#   fronts/GLPE_HUMAN.jsonl, fronts/HBB_HUMAN.jsonl, …
```

A record that fails — a residue the codon table has no codon for, say — is
reported and the batch goes on to the next. The batch ends with one line
naming every record's outcome and exits 1 if any record failed. Two records
with the same id are refused before anything runs. A directory that already
holds files is refused unless `--force`, which starts every record over, or
`--checkpoint`, under which rerunning the same command skips each record whose
result is finished and continues the one that is not. A batch does not take
`--resume`, `--resume-search`, `--front-from`, `--narrow-from` or
`--anchors-from`, which each name one molecule's file. A file of one record
runs exactly as before, with `--out` a file; a multi-record FASTA on stdin is
still refused.

## The constraint

With flanks given, `derna` forbids the **crossing** pair classes that reach the
coding sequence — `5-cds` and `cds-3`, whichever flank is present. A `5-3`
pair has both endpoints in given flank sequence, so no codon choice can spell
it away: forbidding it would change a design's verdict without changing the
design.
That is the constraint the front and the modular front are defined
under, so it is the default rather than something to remember.

`5-3` is **not** in the default, and is forbidden only when asked for. Both
endpoints of such a pair lie in fixed flank sequence, so no codon assignment
spells it away: forbidding it leaves the design unchanged and its own free-fold
check failing — measured as 4 of 6 ladder points turning overruled, at gaps of
1.60–3.40 kcal/mol, and separately as 132 of 132 designs byte-identical with
and without the class. Whether the two flanks pair with each other is a
property of the flanks themselves, which `flank_precheck` reports before any
design work.

```bash
derna design --protein hbb.fasta --utr5 u5.fa --utr3 u3.fa --lambda 0.5
#   forbids 5-cds and cds-3

derna design --protein hbb.fasta --utr5 u5.fa --utr3 u3.fa \
    --forbid 5-cds,cds-3,5-3 --lambda 0.5
#   adds the flank-to-flank class, which the default leaves out

derna design --protein hbb.fasta --utr5 u5.fa --lambda 0.5
#   forbids 5-cds — the classes needing a 3′ UTR are dropped, not refused

derna design --protein hbb.fasta --utr5 u5.fa --utr3 u3.fa --forbid none --lambda 0.5
#   no constraint

derna design --protein hbb.fasta --utr5 u5.fa --utr3 u3.fa --forbid 5-cds,cds-3 --lambda 0.5
```

`--forbid` takes the crossing classes only. A within-region class (`5-5`,
`cds-cds`, `3-3`) is refused: the first and last forbid structure inside a
fixed flank that the molecule's own fold forms anyway, so the design would fail
its own check by construction, and `cds-cds` leaves MFE nothing to measure.
Naming a class whose flank is absent is refused too — it would be a constraint
on nothing.

`--access UP,DOWN` additionally keeps a window around the start codon unpaired:
every pair with an endpoint in it is forbidden, whatever the partner's region.

### What the check tells you

A design is **modular** when its free MFE equals its constrained
energy: the molecule's own preferred fold already respects the constraint.
Under any constraint every design carries that check — `free_mfe`,
`free_structure`, `gap` and `modular`, plus the `violations` its
free fold forms. `derna design` **reports** the check and never alters the
design.

## Sequence constraints

```bash
derna design --protein hbb.fasta --lambda 0.5 --motif GAAUUC --motif GGAUCC
derna design --protein hbb.fasta --lambda 0.5 --motif sites.txt
derna design --protein hbb.fasta --lambda 0.5 --motif GGGG --motif-window 10..-10
derna design --protein hbb.fasta --lambda 0.5 --pin 9:A --pin -6:GCCACC
```

These examples carry no flank, so they have no stop codon to design. With a
flank given, a `--motif` or `--pin` run must name its stop codon —
`--stop UAA|UAG|UGA` or `--stop none` — because a sequence constraint is not
carried over a designed one (see [The molecule](#the-molecule)).

`--motif` takes an IUPAC pattern or a file of one per line (`#` begins a
comment). Motifs are forbidden over the coding sequence, and **every
occurrence overlapping it is forbidden**, including one straddling a CDS/UTR
boundary, since the codons on the coding side are being chosen.
Two consequences are worth knowing before you hit them:

- An occurrence lying **wholly inside a fixed UTR** cannot be removed by any
  codon assignment. The run proceeds and warns, naming the side, the motif and
  the position.
- An occurrence straddling the **5′** boundary reaches into the start codon,
  which the protein fixes. If the flank spells the rest of it, no design can
  avoid it and the run is refused naming the motifs. The 3′ boundary has no
  such problem: the last codon is designed.

`--motif-window LO..HI` narrows enforcement to a coding sub-window; with no
`--motif` it is refused, since it would scope nothing.

**`--max-homopolymer K` caps the longest run of one nucleotide**, and is a
motif set and nothing more. A run longer than `K` contains a run of exactly
`K + 1`, and a run of `K + 1` is a run longer than `K`, so the cap is exactly
the four patterns `A…A`, `C…C`, `G…G`, `U…U` of length `K + 1` — over codon
boundaries and over the CDS/UTR boundary alike, since neither direction of that
argument mentions a boundary. The flag expands to those four and they are
forbidden on the same exact path as any other blacklist; it adds no machinery
and buys no approximation. The header records the four patterns rather than the
flag's value, because the patterns are what the run was under:

```bash
derna design --protein hbb.fasta --lambda 0.5 --max-homopolymer 6
derna front --protein hbb.fasta --motif sites.txt --max-homopolymer 9 --out front.jsonl
```

Four points to know:

- **It sits beside `--motif`**, adding four patterns to whatever that names.
- **`--max-homopolymer 0` is refused**: it would forbid every single
  nucleotide, and the run would come back infeasible without saying that the
  cap, not the protein, was impossible. A small cap can still be infeasible —
  `1` has no encoding of any protein holding `K` (`AAA`/`AAG`) or `F`
  (`UUU`/`UUC`) — and that run is refused naming the patterns.
- **`--motif-window` is refused with it.** `--motif-window` narrows the whole
  set, so the two together would narrow the cap; and not to the window either,
  since every occurrence *overlapping* the enforced span is forbidden, so a run
  that merely straddles the span's edge is forbidden too. Neither reading is a
  windowed cap, so the combination is refused rather than given one of them.
  A pattern's *own* window (`--motif PATH@LO..HI`) is a different flag and
  stays allowed: it scopes that pattern and leaves the cap over the whole
  molecule.
- **A cap at or above the molecule's longest unconstrained run forbids
  nothing**, and the run then emits the unconstrained design byte-identically.
  Read `max_homopolymer` on the result (see [Sequence
  metrics](#sequence-metrics)) to see what the design actually spells; a cap
  longer than the molecule is accepted and does nothing.

A run lying wholly inside a fixed UTR is the general motif rule's case: no
codon assignment can remove it, so the run proceeds and warns. A run straddling
the 5′ boundary whose coding side is forced refuses, as any such straddler
does.

`--max-homopolymer` on `show` and `pick` is the same quantity as a **filter**
over points already written, not a constraint on a run.

A **pattern may carry its own window** instead, so one run can forbid one
blacklist over the whole coding sequence and another only past a given coding
position. Two
spellings, and `--motif-window` is the default for whatever carries neither:

```bash
derna front --protein p.fa --utr5 u5.fa --utr3 u3.fa \
  --motif sites.txt --motif initiation.txt@5..-1
```

```
# initiation.txt — Kozak-like contexts around an *internal* AUG
@window 5..-1
RCCATGG
```

A `@window LO..HI` line scopes every pattern **below** it in that file, until
the next such line; a suffix on the `--motif` value scopes everything that
value names. A scope in the file is the one to prefer when the scope is a
property of the blacklist rather than of one run: a command line then cannot
leave it off, and a driver concatenating sets into one file carries it along.
It is a line of its own rather than a `#` comment because a comment is
stripped by every tool that reads these files, so a scope written that way
would be silently dropped by an older build instead of refused by it.

The window a set of internal-initiation contexts needs is the one that excludes
the **authentic** start. A strong initiation context ends `ACC` and the start
codon follows, so `RCCAUGG` is spelled by fixed flank sequence, the protein,
and the residue after the initiator methionine — and where that residue has
only `G`-initial codons (`A D E G V`) no design can avoid it and the unscoped
run is refused. Write that window `5..-1` and not a literal: the coding window
is `3n` whatever `--stop` says, since the stop codon sits at the head of the 3′
flank, so `5..24` means *the first eight codons* on a long molecule rather than
*everywhere but the junction*. `5` is the minimal left end — `4..-1` still
refuses.

The header records the pairing in `motif_scopes`, one `{motif, window}` object
per pattern, whenever some pattern carries a window of its own; `--resume` and
`--narrow-from` compare it as a set of pairs. `derna design` forbids its whole
set over one window, so a pattern carrying its own is refused there, naming
`front` and `--motif-window`.

`--pin K:LETTERS` fixes
coding nucleotides from position `K`, keeping for each residue exactly the
synonymous codons that spell them; a residue left with no codon is refused up
front, naming it, and a coordinate carrying a region prefix (`cds:9`,
`5utr:-6`) is refused so that there is one spelling. `design` takes all
four; `front` takes `--motif`, `--motif-window` and `--max-homopolymer` (see
[The motif-excluded front](#the-motif-excluded-front)), and `--pin` with
`--motif` only.

## Results

Every command writes JSONL: a header line recording the inputs — protein,
flanks, stop, constraint, the full codon table, the version, the wall clock and
the peak memory — then one line per design. Progress, warnings and the summary
go to stderr, so the commands compose:

```bash
derna front --protein hbb.fasta --utr5 u5.fa --utr3 u3.fa --out front.jsonl
derna show front.jsonl
derna show front.jsonl --min-cai 0.9 --sequences
derna show front.jsonl --tsv > front.tsv
derna show front.jsonl --modular --fasta > strict.fa
```

The result is the only thing on stdout, so a command is safe to redirect or
pipe. Once `design`, `front`, `check` or `pool` has written its result, it
prints `show`'s default table of each file it wrote to **stderr**, under a
`wrote PATH:` line — `wrote stdout:` when the JSONL went to stdout, which
still carries nothing else. `--quiet` turns the table off. Every command exits **0** on success and **1** on any refusal or
failure, with the reason on stderr — including a refusal to start, such as an
unreadable input or a flag this command does not take. `derna` with no command,
`--help` and `--version` print to stdout and exit 0.

`--out` refuses an existing **regular** file unless `--force` is given; `--out -`
means stdout and `--out /dev/null` discards, since neither has contents to
lose.

**`--checkpoint` means one thing on every command that computes:** save
progress beside `--out` as it settles, so that running the **same command
again** continues a killed or budget-stopped run instead of starting over.
`--force` starts over instead, and removes the progress first, so a later run
cannot continue from a mixture of two runs. `--checkpoint` needs a regular
file at `--out`; stdout, a device or a pipe is refused, since there is nowhere
beside it to save.

| Command | What is saved | A rerun |
|---------|---------------|---------|
| `design` | each weight's design, one line per λ, in `OUT.design.jsonl` | designs only the weights not in the file, matched by their exact value; the result is the uninterrupted run's, in `--lambda` order |
| `check` | each point's verdict in `OUT.check.jsonl` | folds only the points not in the file |
| `front` | the front so far, rewriting `--out` | continues the walk from it, as `--resume OUT` would |
| `front --modular` | the constrained walk in `OUT.walk.jsonl`; each point checked, design found and band slice finished in `OUT.search.jsonl` | continues the walk from the first and the search from the second, as `--resume` and `--resume-search` would |

A rerun refuses a file of another run — another molecule, constraint or
options — naming the field that differs, and refuses a finished `--out`
unless `--force` is given. A `design` progress file is removed once the result
has landed, and the header's `resumed` says how many weights were taken back
rather than designed. A complete `--modular` front removes `OUT.walk.jsonl`
and keeps `OUT.search.jsonl`, which also records every check the search made.
An explicit `--resume` or `--resume-search` still names the file to continue
from, and is never removed by `--force`.

```bash
derna design --protein hbb.fasta --utr5 u5.fa --utr3 u3.fa \
  --lambda 0.1,0.3,0.5,0.7,0.9 --checkpoint --out designs.jsonl
# killed after three weights; the same command designs the other two
derna design --protein hbb.fasta --utr5 u5.fa --utr3 u3.fa \
  --lambda 0.1,0.3,0.5,0.7,0.9 --checkpoint --out designs.jsonl
```

`show` folds nothing: every energy, CAI and verdict it prints comes from the
file, and a point with no check prints `—` in the `gap` and `modular` columns
while the summary says `not checked`. What it derives is sequence arithmetic:
the [sequence metrics](#sequence-metrics) of a file written before points
carried them, computed from each point's `sequence` and the coding window the
header implies, and the cuts and spellings below.

The columns are `mfe`, `cai`, `gc`, `gap`, `modular`, `on_hull`, `lambda` and
`status`; `--designs` adds each point's sequence, structure and the other
metrics (`gc_cds`, `u_cds`, `gc_window_min`, `gc_window_max`,
`max_homopolymer`) under its row, or as columns under `--tsv`. A gap a beam
refuted prints as `≥1.00` rather than `1.00`, so a bound is never read as a
value. `status` is `certified` for a point that passed `check`.

The filters are `--min-cai X`, `--max-mfe E`, `--modular`, `--min-gc X`,
`--max-gc X` and `--max-homopolymer N`. `--plot PATH.svg` also draws the
selected points: CAI on x, MFE in kcal/mol on y, titled with the record's
name, or with `--plot-title TEXT` instead. A point on the hull is a square and
one off it a circle; a modular point
is filled, a point that fails its check is open, and one nothing checked is
open in grey. The SVG is written by hand and needs no plotting library.

### Sequences to order

`--fasta` writes the selected points as FASTA, one record each, with a header
naming the protein's FASTA id, the point's 1-based index in the file, and its
`mfe`, `cai` and `gc` (and `lambda` for a `design` result):

```text
>GLPE_HUMAN_20aa_cut point=3 mfe=-34.00 cai=0.8092 gc=0.56
```

A result made from a literal sequence has no id and is named `derna`.
`--dna` writes T for U, and `--cds-only` writes the coding sequence alone —
start codon through stop codon, or through the last codon when the run has
none — without the UTRs. The default is the whole molecule.

`derna pick` writes **one** point of a `design`, `front`, `check` or `pool`
result, by exactly one rule:

```bash
derna pick front.jsonl --best-cai --max-mfe -30      # highest CAI at MFE ≤ −30
derna pick front.jsonl --best-mfe --min-cai 0.85     # lowest MFE at CAI ≥ 0.85
derna pick front.jsonl --knee --modular --dna > order.fa
derna pick front.jsonl --knee --json                 # the point's JSONL row
```

- `--best-cai`: the highest CAI;
- `--best-mfe`: the lowest MFE;
- `--knee`: the point farthest from the chord joining the selection's two
  ends — its lowest-MFE and its highest-CAI point — with MFE and CAI each
  normalized to [0, 1] over the selected points.

It takes `show`'s filters (`--min-cai`, `--max-mfe`, `--modular`, `--min-gc`,
`--max-gc`, `--max-homopolymer`) under any rule, and `--modular` on a file
whose points were never checked is refused, as `show --modular` notes. Ties go
to the lower MFE, then the higher CAI, then the earlier point in the file, so
the same file and flags always pick the same point. When no point qualifies it
exits 1 naming the filter that removed the last of them. The output is a FASTA
record as `show --fasta` writes it, `--dna` and `--cds-only` included, or the
point's JSONL row under `--json` (with its metrics derived when the file
predates them).

A target such as *the best MFE with CAI ≥ 0.9* is `--min-cai 0.9` and the top
row — exact, because the front is complete.

### The fields a result carries

The header line (`"record": "header"`):

| field | meaning |
|---|---|
| `format`, `format_version` | `derna-result`, and the version a reader must understand. A reader refuses a version above its own rather than rendering unknown fields as absent. Version 2 renamed `structurally_optimal` to `modular` and `structurally_optimal_front` to `modular_front`; every reader still takes the old names in a version-1 file. A file declares 2 only when it carries one of the renamed fields (a checked design, `check` output, a modular front) and 1 otherwise, so a plain or motif front stays readable by a `derna` built before the rename. |
| `command`, `version` | which subcommand wrote it, and the `derna` version |
| `protein`, `utr5`, `utr3` | `{name, sequence}` each; a flank is `null` when absent |
| `stop` | the given stop codon, `"free"` when the search chose it per design, or `null` for none |
| `forbid`, `access` | the forbidden classes as a list, and `[up, down]` or `null` |
| `codon_usage` | `{source, table}` — the whole table, so CAI is scored against exactly what the run used |
| `threads`, `deterministic` | what ran, not what was asked: a run under `--deterministic` records 1 |
| `motifs`, `motif_window`, `pins` | the sequence constraints the run was under; absent when it had none |
| `motifs_in_fixed_flanks` | the occurrences no codon assignment could remove, which the run warned about |
| `wall_s`, `peak_rss_kb` | the run's own cost — the memory figure is the process's peak resident size, not a sample, so it is a true peak. It is `null` where the platform reports no such figure, which is not a reading of zero |
| `lambdas` (design) · `range`, `pass_grid`, `points`, `gaps` (front) | the command's own inputs and shape |

A point line (`"record": "point"`), with energies in kcal/mol:

| field | meaning |
|---|---|
| `sequence`, `structure` | the whole molecule, UTRs included |
| `mfe` | the energy **under the constraint** — the objective |
| `cai`, `cai_log` | conventional CAI, and the additive log-CAI the objective uses |
| `stop` | the stop codon this design ends its coding sequence in — only when the search chose it, since otherwise the header's applies to every point |
| `on_hull` | whether a weight sweep can reach this point |
| `lambda` | the weight it was asked at — `design` results only |
| `free_mfe`, `free_structure`, `gap`, `modular`, `violations` | the check, under any constraint. `gap` = `mfe − free_mfe` ≥ 0, and `violations` are 1-based pairs |
| `status` | the verdict of whichever command wrote the file: `certified` or `provisional` from a modular `front` (see [that section](#the-modular-front)), `certified` from `check` for a point that passed. A check replaces the input's, so one file's column carries one sense |
| `gc`, `gc_cds`, `u_cds`, `gc_window_min`, `gc_window_max`, `max_homopolymer` | the [sequence metrics](#sequence-metrics), on every point every command writes |

A check's **header** carries `front_run` where its input recorded one: the
`complete`, `stopped_by`, `open`, `seeds` and `unchecked` of the front being
checked. That front's claim about the regions it left open is not one a check
re-establishes or refutes, and a modular front that stopped on a
budget emitted the points it reached rather than the whole front — so a reader
of the checked file needs the front's own verdict to know which of the two it
has. It sits in the header rather than on each point because it is a property of
the run, and because a per-point copy could not survive a re-check: a `status`
the check writes is indistinguishable from the one a front writes. Checking a
checked file carries `front_run` through unchanged, so a chain of checks keeps
the front it started from.

CAI is absent from the check because both folds are of one sequence, so its
codons — and therefore its CAI — are the same; only a different sequence
changes CAI.

### Sequence metrics

Every point `design`, `front`, `check` and `pool` write carries six numbers
read off its sequence alone — no folding:

| field | meaning |
|---|---|
| `gc` | GC fraction of the whole molecule |
| `gc_cds` | GC fraction of the coding sequence |
| `u_cds` | U fraction of the coding sequence |
| `gc_window_min`, `gc_window_max` | the least and greatest GC fraction over every 50-nt window of the whole molecule; the whole molecule's GC when it is shorter than 50 nt |
| `max_homopolymer` | the longest run of one nucleotide in the whole molecule — what [`--max-homopolymer`](#sequence-constraints) caps on a run, and what `show --max-homopolymer` filters on afterwards |

The coding sequence runs from the start codon, after the 5′ UTR, through the
stop codon — given or designed — or through the last residue's codon when the
run has none. The fields are additive, so they do not change the format
version: a file declares 2 only when it carries a renamed field, as before,
and a reader built before the fields reads a file that has them. `show` and
`pick` read a file written without them by deriving them the same way.

## The front

```bash
derna front --protein hbb.fasta --utr5 u5.fa --utr3 u3.fa --out front.jsonl --report walk.json
```

`front` has **no range flag**. The walk returns the Pareto designs between the
designs optimal at two weights, so the range decides how much of the front
comes back; the command fixes the symmetric range [2⁻²³, 1 − 2⁻²³], which is
below the weight at which a corner stops reaching the front's end at every
length a real protein has, and is exact in binary floating point. It picks its
pass grid from the molecule's length, starts the hull sweep from the grid's own
optima, and cuts each sweep fold under the grid's chords. Those two return the
same front as an unseeded, uncut sweep in a fraction of the time: on P0DTC2's
first 200 and 250 residues, 2.7× and 3.6× less wall on one thread and 3.3×
and 5.3× on eight, all of it in the sweep.

`--report` writes the walk's diagnostics: the hull's vertices, the grid the run
used, one row per gap — the pair it spans, the weight it was solved at, the
points it produced, the energy slices it took, the cut it read, and where its
seconds went — and every task the queue ran. The result file says what the
front *is*; the report says what the walk *did*, which is where a cost question
is asked.

Each point carries `on_hull`: whether it is a vertex of the convex hull, which
is to say a point a weight sweep — and so `derna design` — can reach. The rest
are the off-hull points the gap folds find, and they are the reason a front is
not a λ sweep.

**How much a sweep misses, measured.** The off-hull share rises with length:
about half the front at 20 residues, 85 % at 78, 86–87 % at 100, 90 % at 150,
**92 % at 200** and 95 % at 550 — and it is remarkably insensitive to
which protein: **four** different proteins agree within one and a half
percentage points at 100 residues (85.8–87.2 %) and **four** within
six-tenths of a point at 150 (89.4–90.0 %). One of them was held out from the
measurement that predicted it, at both lengths. At 200 residues a sweep can reach 58 of the
front's 763 designs however many weights you give it. So `design` is the right
command when you want a design at a weight you have chosen, and `front` is the
only one that can show you the trade-off you are actually choosing from.

**A budget stops the walk and says what it left.** `--max-secs` and
`--max-rss-gb` apply to every front this command walks: no probe or gap fold
starts once either runs out, and the run writes the points it has settled.
With `--motif` both are also read inside each gap fold, so a gap whose fold
outgrows the memory cap is left open rather than run until the host kills it.
Those points are exact, because each gap is solved between two hull points
that are front points themselves, so nothing an unsolved box holds can
dominate one of them. The header says `complete`, `stopped_by` (`time`,
`memory`, `folds` or `null`) and lists each box left `open` as its two hull
points.
A front read back with `--front-from` is not walked, so a budget is refused
there, and a front whose own header says it is incomplete is refused as
something to read: the search built on it would certify against a front still
missing designs. `--resume FRONT.jsonl` hands those boxes back: the walk folds them and
nothing else, returns the file's points with the ones it finds, and records
what it continued under `resumed`. The hull comes from the file, so neither
end is folded again — that is the saving — and the file is refused unless it
is a plain front of the same molecule, flanks, constraint and codon table. A
motif front is refused with its reason: it is a different front, and
`--motif --resume` continues that one.

**`--max-gap-folds N` is a stop that does not depend on the machine.** It ends
the walk after `N` gap folds, with or without `--motif`, and writes what it
has settled exactly as a budget does: the header says `stopped_by: folds` and
records each box it left `open`, and a box whose gap was solved part way
carries the energy its gap is solved up to (`mfe_from`) and the best
adaptation there (`cai_log_floor`), so `--resume` folds only the rest. Where
`--max-secs` stops after a span of time, and so at a different place on a
faster or a busier host, this stops after a fixed amount of work, whatever
the host's speed. That makes it the way to split a
long front into resumable pieces of a known size, each run taking the last
one's file with `--resume` and the same `--max-gap-folds`, until the header
says `complete`. It is refused with `--modular` and with `--front-from`, where
the walk is not what the run returns.

A resume is for a run that was killed, not a way to make a walk cheaper. It
folds neither end of the hull, and that is all it saves: the completion credit
and the pass grid are built again, because every gap fold reads them. Measured
on a 200-residue prefix with all three crossing classes forbidden, on two
threads, those took 73 s of a 134 s walk — so a budget below 73 s returns no point at all,
since no gap can start before the grid is pooled, and resuming from such a
file costs about as much as starting over. A stop at 114 s resumed in 97 s:
the same front, and 210 s of work against 134 s for one run.

**`--max-rss-gb` is Linux-only, and the memory it stands for is unguarded
elsewhere.** It needs the process's current resident size, which comes from
`/proc`; where that cannot be read the limit would never bind, so the flag is
refused outright rather than accepted as a cap that does nothing:

```text
error: derna front: --max-rss-gb needs the process's resident memory, which
this platform does not report; the limit would never bind
```

The quieter half is the worker count. A walk reads the memory available to it
and takes fewer workers where a round of folds would not fit; off Linux that
reading is absent too, and the walk falls back to one worker per core without
saying so. So on macOS a long walk has neither the cap nor the climbdown, and
only the first absence is reported. Bound such a run with `--max-secs` and
`--threads`, and size it with the memory figures under
[What a front costs](#what-a-front-costs) — `35 MB + 0.04 × residues²` for the
plain front — rather than relying on the run to stop itself. `--max-secs` has
no such dependency and binds on every platform.

**`--checkpoint` survives a kill** here as it does for the motif front (see
[The motif-excluded front](#the-motif-excluded-front)): `--out` is replaced
as soon as both ends of the hull are folded, and again after every pair of
hull points the walk settles, solved as a gap or refined into two, by the
front found so far, written beside it, synced and renamed over it. A
checkpoint carries `"checkpoint": true`, and its `open` boxes are every pair
the walk has handed out and not settled, together with the ranges between
grid optima it has not handed out yet, so `--resume` continues from it. No
checkpoint comes before both ends are folded, since before then there is no
box to name; the first comes right after, so a run killed before its first
pair settles still leaves a file to resume. On eGFP with fixed flanks on
three threads it landed at 22 s of a 163 s walk, where waiting for the first
settled pair put it at 109 s. Running the same command again over a
checkpoint continues from it, as `--resume OUT` does. `--checkpoint` is
refused with `--front-from` on a plain front, which walks nothing; with
`--modular`, whose `--out` holds another front, the walk's checkpoints go to
`OUT.walk.jsonl` instead.

A constrained `front` certifies its points after the walk (see [A constrained
front certifies itself](#a-constrained-front-certifies-itself)). Certifying one
costs a free fold of the whole molecule plus a rescore of the structure it
emitted — measured at about 1.2 s per point per thread at 100 residues — so on a
long front the check is a phase of its own, and `--no-check` leaves it to a
later `derna check`.

### What a front costs

**In short:** budget `35 MB + 0.04 × residues²` of memory — that is the figure
that decides whether a run is possible — expect seconds to minutes at tens of
residues and a scheduled job at hundreds, and do not try to predict how many
points come back. The rest of this section is the evidence.

The cost is **one constrained fold of the whole molecule per hull vertex and
per gap between vertices** — exactly `2 × vertices − 1` folds — and not one
per point of the answer. One protein's prefixes, with 25-nucleotide flanks and
the default constraint:

| residues | hull vertices | folds | **points returned** | peak memory | wall (2 threads) |
|---|---|---|---|---|---|
| 20 | 5 | 9 | 16 | 32 MB | 6.1 s |
| 50 | 17 | 33 | 80 | 131 MB | 68 s |
| 100 | 33 | 65 | 258 | 428 MB | 3.5 min |
| 150 | 46 | 91 | 462 | 912 MB | 14.4 min |
| 200 | 58 | 115 | 763 | 1.73 GB | 25.6 min |

Over 50 to 200 residues the fold count grows **roughly in proportion to the
length** — as `n^0.91` on this protein and `n^1.00` on another measured over
50 to 150, so expect something near linear and not much better. The front's
size grows faster (about `n^1.6` here) and peak memory as about `n^1.9`, the
DP table's shape.

A front with many points is not thereby expensive: at 200 residues the walk
returns 763 points for 115 folds. What costs is the number of folds — which
follows the hull — and the price of each fold.

**Your protein matters too, not only its length.** The same length on four
proteins, same flanks and constraint:

| protein (first 50 residues) | hull | folds | points | MFE span |
|---|---|---|---|---|
| Q5VYS8 | 14 | **27** | 41 | 26.4 |
| P0DTC2 | 17 | **33** | 80 | 47.6 |
| GLPE_HUMAN | 18 | **35** | 128 | 50.4 |
| P51805 | 20 | **39** | 69 | 40.6 |

So the fold count varies by about 1.4× between proteins of the same length,
and the number of points by 3×. These are exact counts, not timings.

Wall times are deliberately absent from that table: the measurements behind
this guide come from a shared, variably loaded host, and runs scheduled at
different times there differ by more than the work does — two of these
proteins ran *the same number of folds* and differed twofold in wall. Size a
run by its fold count and its memory, and treat any single wall time as an
upper bound on a quiet machine.

Neither the point count nor the hull is monotone in length, either. Over the
ladder above the count rose, fell twice and rose again, and on P51805 the
*hull* shrank from 20 vertices at 50 residues to 18 at 60 — so its fold count
went **down** as the protein got longer. Measure, don't extrapolate.

**Memory is the predictable part, and it is the figure that decides whether
you can run at all.** Peak memory is set by the DP table, so it follows the
length far more closely than the runtime does: three proteins' peaks sit
within 15 % of each other at a given length, where their runtimes differ by up
to three times. Budget roughly

```
peak ≈ 35 MB + 0.04 × (residues)² MB
```

— about 0.4 GB at 100 residues, 1 GB at 150, 10 GB at 500. Against two dozen
measured runs on five proteins the formula lands within **20 %** from 100
residues up, and the 550-residue walk's 14 GB sits in that band too; below
~50 residues the floor dominates and it reads as much as 1.6× high.

**If you have run your own protein once, scale from that instead.** The
formula's error turns out to be mostly a per-protein offset — each molecule
keeps to a lane of its own roughly a tenth wide while the lanes span a factor
of 1.3 — so one measurement of yours at any length from 100 residues up
predicts your other lengths to about **±10 %**, which is twice as tight as the
formula and costs a run you were going to make anyway.

For scale at the other end, a front at **550 residues** (no flanks, no
constraint, four threads) finished in **4 h 09 m**, having folded 277 times
over a 139-vertex hull to return 2,771 points — and it peaked at **14.5 GB of
memory**, which is the figure likeliest to decide whether you can run it at
all. The ladder's own identity holds there too: 277 = 2 × 139 − 1.

The wall times here come from a shared host at load 35–51, so read them as
upper bounds and as orders of magnitude; the fold counts are exact, being
deterministic.

The practical reading: tens of residues is seconds to minutes, hundreds is a
job you schedule rather than wait for. There is no budget flag yet, so a walk
that is too long for you yields **nothing** if you stop it — `derna design` at
a few weights is the way to get something at a length you cannot walk.

### The motif-excluded front

```bash
derna front --protein hbb.fasta --utr5 u5.fa --utr3 u3.fa --motif GGGGG --motif CCCCC --out front.jsonl
```

With `--motif`, the front is the exact Pareto front of (constrained MFE,
CAI) over the designs that spell **no** forbidden motif, under the same rule
`design` enforces: every occurrence overlapping the coding sequence (or the
`--motif-window`) is forbidden, and an occurrence wholly inside a fixed UTR is
warned about and left. It is not the plain front with the offending points
filtered out. Removing a design can expose points the plain front's
designs dominated, so the motif front can hold **more** points than the
plain one (98 against 80 for P0DTC2's first 50 residues below).

The walk is the plain one's in shape — the hull's vertices from scalar folds,
then a front fold for each gap between neighbouring vertices — run over the
motif lattice, whose nodes carry the motif automaton's state. Its cuts are
floors taken from the plain lattice, which is admissible because every
motif-free design is also a plain one. At low weight a plain floor credits
mostly completions the motif set forbids, so the pass at every grid weight
above 0 also runs over the motif lattice itself, and the scalar probes
between two such weights (or between weight 0, where the motif lattice's
floors are exact by construction, and the first) prune on those floors as
well. Each such pass also certifies the motif-free optimum at its weight, a
hull vertex, and the walk starts from those vertices rather than from its two
ends alone. On a complete 349-residue motif front the passes at every weight
took 4.1 % fewer instructions than those up to 0.07, and the seeds 1.4 % fewer.
The front depends on neither; `DERNA_MOTIF_AWARE_UPTO` sets the highest weight
with such a pass (`0` takes none) and `DERNA_MOTIF_AWARE_SEEDS=0` starts from
the ends alone. The header records `motifs`, `motif_window` and `motifs_in_fixed_flanks`
as `design`'s does, and `pass_grid` is `null`. `--report` writes the hull's
vertices; per gap, its weight, its points and its fold seconds; per pass, its
seconds and memory, with the seconds, folds and size of its motif-lattice
pass; and per probe, its weight, seconds and folds and what its floors ruled
out (`pruned` cells, `cut` pairs), of which `aware_pruned` and `aware_cut` the
motif-lattice floors alone. `phases` sums the seconds of each phase: `plain_s`,
`aware_s`, `gap_fold_s`, `probe_s` and `ends_s`. Phases share threads, so the
sum can exceed wall time. `hull_s` is the time from the walk's start to its
last settled gap, 0 when none settled, not the cost of a phase.

**What it costs, measured.** Two proteins' prefixes with the 25-nucleotide
flanks `GGGAAAUAAGAGAGAAAAGAAGAGU` / `UGAUAAUAGGCUGGAGCCUCGGUGG`,
all three crossing classes forbidden — the default when these were measured,
narrowed to `5-cds,cds-3`
since, which does not move these numbers: the fronts under the two tables are
identical at 100 and 200 residues on this molecule, the third class costing
about 4 % more time and 6 % more memory for nothing —
`--motif GGGGG --motif CCCCC` and two threads, each run beside the plain front
of the same molecule:

| protein, residues | plain points spelling a motif | points (motif / plain) | hull vertices | wall (motif / plain) | peak memory (motif / plain) |
|---|---|---|---|---|---|
| P0DTC2, 20 | 1 of 16 | 15 / 16 | 5 | 0.5 / 1.0 s | 27 / 35 MB |
| P0DTC2, 50 | 37 of 80 | 98 / 80 | 20 | 7.5 / 13 s | 137 / 129 MB |
| P0DTC2, 100 | 173 of 258 | 247 / 258 | 33 | 31 / 66 s | 458 / 436 MB |
| P0DTC2, 150 | 462 of 462 | 505 / 462 | 46 | 104 / 190 s | 857 / 840 MB |
| P0DTC2, 200 | 734 of 763 | 768 / 763 | 53 | 320 / 415 s | 1.54 / 1.53 GB |
| Q14442, 20 | 0 of 10 | 10 / 10 | 7 | 0.7 / 1.2 s | 30 / 34 MB |
| Q14442, 50 | 49 of 56 | 55 / 56 | 13 | 9.7 / 15 s | 150 / 144 MB |
| Q14442, 100 | 115 of 157 | 180 / 157 | 23 | 50 / 81 s | 497 / 485 MB |

The second column says the motifs bind: from 50 residues up they rule out
between 46 % and all of the plain front's designs (none at Q14442's 20, where
the two fronts coincide). The walk folds once per hull vertex and once per
gap, as the plain walk does — 2 × vertices − 1 scalar folds and vertices − 1
gap folds — plus one pass of the plain lattice at each of up to 17 fixed
weights, which every fold's cut reads. A gap whose fold grows past a fixed
size is re-solved in energy slices, so the report's per-gap `folds` can exceed
one. Each gap is folded first in even ranges of at most 2.5 kcal/mol, lowest
first; `DERNA_MOTIF_SLICE_WIDTH` sets another width in dekacalories, and `0`
folds each gap whole first. On a complete 349-residue motif front the ranges
took 7.5 % fewer instructions than whole-gap folds, with the same front. The walk's wall time was below the plain front's on every molecule here — half
to two thirds of it from 100 residues to 150, three quarters at 200 — and its peak
memory within 6 % of the plain front's. The host carried other tenants' load
of 3–8, so read the walls as upper bounds; the point and vertex counts are
exact.

Past 200 residues the cost climbs steeply. Under the same motifs, constraint
and 2 threads, P0DTC2's first 300 residues were stopped by a 3,000 s budget
with 83 points written and 14 gaps open (3.5 GB peak), and its first 400 by a
6,600 s budget with 240 points and 12 gaps open (6.3 GB). A set of 21
patterns (restriction sites, processing signals and runs) at 250 residues was
stopped by a 4,800 s budget with 624 points and 6 gaps open (2.5 GB). At
these lengths, run with `--checkpoint` and continue with `--resume`.

**A budget returns part of the front, all of it exact.** `--max-secs` and
`--max-rss-gb` stop the motif walk: no fold starts once either runs out, and a
gap's front fold already running stops at the deadline (the two end folds and
a hull probe, once started, run to completion, so no probe is started when
less time is left than the longest one so far took). Memory is read between folds, so
a run's peak passes `--max-rss-gb` by whatever the folds in flight hold — above
all the two end folds, which run before the first check and peak at 3.1 GB side
by side at 400 residues and 4.7 GB at 500. Every point the run writes is
a point of the exact front — the walk's gaps lie between hull vertices that
are front points themselves, so nothing in a gap left unsolved can dominate a
point found in another — and every hull vertex found is written, even beside a
gap no fold finished. The header says `complete`, `stopped_by` (`time`,
`memory` or `null`) and lists every gap left `open` as the box between its two
vertices (`mfe_lo`..`mfe_hi`, `cai_log_lo`..`cai_log_hi`): the missing points
are exactly the front's points inside those boxes. The memory budget is read
between folds, so a single fold can overshoot it. A result with `complete:
false` and `stopped_by: null` is a checkpoint (`"checkpoint": true`) that no
later write replaced: its run was killed from outside, or is still running.

**A gap stopped part way keeps what it solved.** A gap's fold is solved in
energy slices from its stable end up, so a stop by either budget keeps every
slice solved before it: their points are written as front points, and the box
stays `open` with two more keys, `mfe_from`, the energy up to which its gap is
solved, and `cai_log_floor`, the best `cai_log` at or below it. `--resume`
folds such a box from that rest, without the probe the earlier run already
paid for. A box's missing points all lie above `mfe_from`, so the energy it
leaves unsettled is `mfe_hi − max(mfe_from, mfe_lo)`. Reading it as
`mfe_hi − mfe_lo` counts a half-solved box as wholly open, and a coverage gate
that does so never reaches its target while every other column shows progress.
`mfe_from` sits one hundredth below `mfe_lo` where the box's probe ran and no
slice was solved before the stop. A reader that ignores the two keys treats
the box as wholly open and re-walks it, which loses no point.

Two limits remain. A stop inside a slice's fold loses that fold, so a gap
whose first slice outlasts the budget still keeps nothing: whether a round
banks anything at a given length is decided by the cost of one slice, not of
one gap. At 1,000 residues the narrowest slices took 154–470 s each across
two measurements, against rounds of hours. And the two end folds never read
the deadline: they run to completion whatever `--max-secs` says, so a run
whose ends outlast its budget stops only after them. With `--checkpoint` the
ends are written as soon as they finish (below), so they are paid for once.

**`--resume FRONT.jsonl` picks a stopped motif front up.** It reads the earlier
result, refuses it unless its molecule, flanks, constraint, codon table and
motifs are this run's, and walks only the boxes it left open: their points
are returned beside the earlier ones, which are repeated exactly as written,
and the header's `resumed` records the file and how many points and open
gaps it held. The earlier hull vertices are taken as given, without a fold,
so a stopped run can be continued under a larger budget, or in several
budgeted steps, for the cost of what it had left. Without `--motif`,
`--resume` continues a plain front instead.

```bash
derna front --protein p.fa --utr5 u5.fa --utr3 u3.fa --motif sites.txt --max-secs 3600 --out part.jsonl
derna front --protein p.fa --utr5 u5.fa --utr3 u3.fa --motif sites.txt --resume part.jsonl --out whole.jsonl
```

**`--checkpoint` survives a kill.** A budget stops the walk cleanly, but a
run killed from outside — a scheduler's time limit, the kernel out of memory —
writes nothing, since the result is written at the end. With `--checkpoint`,
`--out` is replaced by the front found so far once the two end folds finish,
and again after every settled pair of hull points. Each is written beside it,
synced and renamed over it, so the file is always either the previous
checkpoint or the next one. A checkpoint is a result like any other, with
`"checkpoint": true` and `complete: false`; its `open` boxes hold every gap
not yet reached, and running the same command again continues from it, as
`--resume` does. The first holds the two
ends and the box between them, so a run killed inside its first box, which
at a thousand residues can take most of an hour, resumes without folding the
ends again; a resumed run's first checkpoint is the state it resumed from.
The final result replaces the last checkpoint.

**`--fold-threads N` spends cores inside each fold.** `--threads` sets how
many boxes the walk works on at once, and each worker holds its own fold and
passes, so at long lengths memory allows fewer workers than there are cores.
`--fold-threads` runs each end, pass, probe and gap fold on N threads instead,
with the same answer at any N. The two compose:
`--threads 1 --fold-threads 8` is one worker using eight cores. It is refused
without `--motif`.

**`--pin` narrows the motif front.** With `--motif`, pins restrict each
residue to the synonymous codons that spell them, as in `design`, and the front
is over the motif-free designs that spell the pins. Each point's
`cai_log` is its design's own, so the front's adaptation is capped by the best
codons the pins leave. The header records `pins`, and
`--resume` refuses a file walked under other pins. Without `--motif` a pin is
refused, since the plain walk does not run over the restricted lattice.

**`--modular` with `--motif`** is the modular front
over the motif-free designs. The motif front is the search's floor: walked
whole first, walked for at most `--walk-secs` seconds, continued from
`--resume`, or read back with `--front-from` from a motif front written
earlier for the same molecule and motif set. A stopped motif front serves
too, since its open boxes floor the search and the search stays exact. The
search then runs as on a plain front and drops every design that spells a
motif before checking it. The header is the motif front's, with the search's
fields in place of the walk's, and `--front-out` writes the motif front.
With `--pin` both the motif front and the search run over the codons the pins
leave. Without `--modular`, `--front-from` is refused with
`--motif`: `--resume` is how a motif front continues.

## Certifying a result

```bash
derna check front.jsonl --out strict.jsonl
derna show strict.jsonl --modular
derna front --protein hbb.fasta --utr5 u5.fa --utr3 u3.fa | derna check - | derna show -
```

`check` takes the result file positionally and **nothing about the molecule**:
the header carries the protein, the flanks, the constraint and the codon
table, so a check cannot be pointed at a different molecule than the result
was produced over. `--protein`, `--utr5`, `--utr3`, `--forbid`, `--access` and
`--codon-usage` are refused for that reason. To score sequences that have no
such header — a design another tool made — see [Scoring a design this run did
not make](#scoring-a-design-this-run-did-not-make), where those same flags are
how the molecule is given.

It folds each point's own sequence freely and records the verdict:
a point that passes gets `status: certified`, and one that fails keeps its gap
and its violating pairs. The verdict is this run's: a check replaces whatever
verdict fields the input carried, `status` among them, so a point that fails
here carries no `status` even if the front it came from called it certified —
that is [the front's own sense](#the-modular-front) of the word and
a different claim. A result produced with no constraint is refused — no point
can violate a constraint that forbids nothing.

`--threads N` certifies N points at a time. Each point's check reads only its
own sequence and the results are written in the input's order, so the output
is the same at any count — which is why `--deterministic` is refused here
rather than needed: a certification already is.

It is worth giving it the threads. Certifying a 258-point front at 100
residues took 306 s on one thread and 58 s on eight (5.3×, measured on a
loaded shared host, so a quiet one should do at least as well) — about **1.2
seconds per point per thread** at that length. Budget roughly one free fold of
the whole molecule per point.

**`--checkpoint` makes a killed check worth its elapsed time.** Without it a
check writes only at the end, so a run killed at any point before that leaves
nothing — and a front of a few thousand points is hours of folding. With it,
every point's verdict is appended to `<out>.check.jsonl` the moment its fold
finishes, and **running the same command again** folds only the points the
killed run left, reporting how many it took back under `resumed` in its header.
`--checkpoint` means that one thing in every command that takes it, so the way
to ask for a clean run is `--force`, the flag that already means *overwrite what
is there*; it discards the progress file rather than continuing from it. The
result file is byte-identical to the one an uninterrupted run would have
written, and the progress file is removed once it has landed, so a resume can
never be taken twice.

A verdict is keyed by **sequence**, not by position: a point's check reads
only its own sequence, so the verdict is the same wherever in the input that
sequence sits and progress survives an input written out again in another
order. The cost of that choice is the one error a resume must not make
silently — a progress file holding a sequence the result does not is another
run's, and is refused rather than believed. `--checkpoint` needs a regular
file at `--out`, since the progress file is written beside it.

Resuming is for a run that was killed, not a way to make a check cheaper: a
point already folded is the only thing it saves, and the molecule and the
codon table are read again either way.

### Scoring a design this run did not make

A design another tool emitted, or one `derna` made without a constraint, carries
no header to read the molecule from — and the question asked of it is the same
one `check` answers: what does the constraint cost a sequence that was not made
under it. `--designs` takes those designs with the molecule on the command
line.

```bash
# What the UTR constraint costs designs made without it.
derna check --designs theirs.fa \
    --protein hbb.fasta --utr5 u5.fa --utr3 u3.fa --out scored.jsonl
```

The file holds whole molecules, flanks included, as FASTA or one per line
(`T` reads as `U`). The molecule flags are **required** here rather than
refused: with no header to contradict, they are how the command learns what the
sequences are designs of. What replaces the header's guarantee is that each
sequence must *be* a molecule of that description — the given flanks at either
end, the coding window translating to the protein, the given stop codon where
one is given — and a sequence that is not is refused, naming it and what is
wrong with it rather than scoring it against a molecule it is not. A design of
another protein, or the same protein read out of frame, cannot pass that.

Each sequence is folded under the constraint, scored for CAI on the table
`--codon-usage` names, and then certified exactly as a point from a result file
is, so the output is an ordinary result: `mfe` is the sequence's constrained
energy, `free_mfe` its own fold, `gap` the difference and `modular` the verdict.
A FASTA record's name is kept on its point. `--designs` and a result file
together are refused: there is no file to check beside them.

**`check` certifies; it does not search.** A failing point is reported, not
replaced. Replacing failing points one at a time is not the route to a
modular front — a search per point does not scale — so that front
is computed whole, by `front --modular`.

### A constrained front certifies itself

A `front` run under a constraint certifies its own points before writing them,
so its `--out` already carries the verdict fields `check` would add, and
running `check` over it afterwards is a second opinion rather than a first.
It costs about one free fold of the molecule per point, as `check` does.

It is skipped where it would say nothing: with no constraint (nothing can be
violated), and with `--modular` (where every point is certified by
construction). `--no-check` declines it — for when the extra fold per point is
not worth it, or when a reader downstream expects the older fields. The front's
own header is kept either way, `complete`, `stopped_by` and `open` included.
A motif front under a pair constraint is certified the same way, since a motif
limits what a design may spell, not which pairs its fold may form.

A certified point carries `modular`, a field the format's version 2 introduced,
so a certified front declares format version 2 and a `derna` built before that
version refuses to read it. Where a front must be read by such a build, for
example as `--front-from` of an older pinned binary, write it with `--no-check`:
an uncertified front stays version 1.

`--check-pool N` lets each certifying worker copy cells from its last `N` free
folds. Consecutive front points differ in a few codons, so much of a fold's
table is a neighbour's; the verdicts are the fresh folds', field for field. It
saves CPU rather than time, and less than the share of copyable cells suggests,
because the cells a change spoils are the wide, expensive ones:

| molecule | points, threads | pool | check CPU | peak RSS |
|---|---|---|---|---|
| 660 nt | 474, 8 | 0 / 2 / 4 / 16 | 1.00× / 1.26× / 1.30× / 1.34× less | 0.04 / 0.08 / 0.12 / 0.35 GiB |
| 3,322 nt | 256, 4 | 0 / 3 / 16 | 1.00× / 1.11× / 1.24× less | 0.39 / 1.13 / 4.51 GiB |

The pool holds `N` whole-molecule fold tables per worker, about `6 n²` bytes
each, so memory grows with `N × threads × n²` — at 8 threads and 3,322 nt a
pool of 16 is about 9 GiB. Wall-clock time was not shown to improve: on a
loaded host it moved both ways between arms. The default is 0, which folds
every point from scratch. The pool is refused with `--no-check`, where the
check it belongs to does not run.

A plain walk likewise saves its progress into `--out` by default wherever it is
a file, so a run killed from outside leaves a resumable front rather than
nothing; `--no-checkpoint` declines. Only the saving is the default: a rerun
without `--checkpoint` overwrites (under `--force`) or refuses as before, and
continues the saved front only when `--checkpoint` asks for it or `--resume`
names it.

### Pooling the per-stop fronts

```bash
# the three stops, run separately, then joined
for c in UAA UAG UGA; do
  derna front --protein p.fa --utr5 u5.fa --utr3 u3.fa --forbid 5-cds,cds-3 \
    --stop $c --out front.$c.jsonl
done
derna pool front.UAA.jsonl front.UAG.jsonl front.UGA.jsonl --out joined
```

`pool` joins certified per-stop fronts into two files: `joined.union.jsonl`,
every point of every arm with the dominated ones dropped, and
`joined.modular.jsonl`, the same filter applied **inside** the modular
designs alone.

The second file is the reason the command exists, and the order is the whole
point. A design that is modular under one stop can be dominated
in `(mfe, cai)` by a design under another, and then it is gone — so filtering
the pooled points first gives back exactly the front a designed stop would have
produced, with none of the per-stop designs that the objectives cannot see.
Filtering after certifying keeps them. Measured on `P54852__mrna1273`, 163
residues with both flanks: the UAA arm holds **85** modular
designs and the other two hold none, while pooling the three arms and dropping
dominated points leaves **0** of the 85 — that union is the designed-stop front
exactly. Across ten cells the same comparison is 1,491 against 0.

Two kinds of loss are at work, and both need the arms kept apart until after
certification: 925 of those 1,491 sit on no non-dominated point at all, and 566
tie *exactly* in `(mfe, cai)` with a non-modular design, where the filter has
no way to prefer the one that keeps the property.

`pool` refuses an uncertified front by name, and refuses fronts of different
molecules by comparing their headers. It reads each point's stop from the
sequence rather than trusting a label, since a fixed-stop front carries its
stop once in the header while a designed-stop front carries one per row.

## The modular front

```bash
derna front --protein hbb.fasta --utr5 u5.fa --utr3 u3.fa --modular \
  --max-secs 3600 --max-rss-gb 16 --threads 8 --out so.jsonl
derna show so.jsonl
```

`--modular` returns the Pareto front of the designs whose own
free fold respects the constraint — every point has `gap` 0 — rather than the
constrained front. It walks the constrained front first, exactly as a plain
`front` does, and builds on it:

1. **Seeds.** A constrained-front point that passes its check is on the
   modular front, since nothing has lower energy at its CAI or
   above. Checking them is two folds of the whole molecule per point, so at
   length it is a long stage of its own, and the budget governs it: points it
   does not reach are not seeds (the header's `unchecked` counts them), and
   the stretches around them stay open. They are checked in an order that
   spreads over the whole front, so a budget that stops part-way leaves seeds
   everywhere rather than at one end. A seed's check is reused for its point
   line rather than folded again.
2. **Stretches.** Between consecutive seeds, and below the first and above the
   last, the missing points are found by walking designs in order of energy
   and CAI and checking each, the seed above a stretch capping it. Two seeds
   adjacent on the constrained front leave nothing between them. The region
   above the best-adapted point found has no cap — the best-adapted
   modular design is on the front whatever its energy. By default
   it is walked like a stretch, and what the walk leaves uncovered is reported
   with an energy floor. `--top-search band` instead checks every design there,
   most adapted first; above a well-adapted point most of those fail their
   check, so the band spends a budget without finding points.
3. **Rounds.** A stretch that runs out of its share of the time is walked again
   in the next round, for twice as long, capped by everything found so far.

Two opt-in flags stop or hold back the rounds before the budget runs out. Both
change what a budget finds, and neither is on by default.

- `--empty-stop-rounds R` stops a search that has found no point once `R`
  rounds are over and its walks have taken at least `--empty-stop-derivations N`
  derivations (default 1,000,000). The header then says `stopped_by:
  "empty-rounds"` and `gave_up: {round, derivations}`, and the front is empty
  and incomplete: it does not say that no modular design exists.
  Rounds count from 0 and each takes twice the last's time, so `R = 7` stops
  after rounds 0–6, about 1.1 h into a run whose first round is 30 s. On the
  cells measured, a walk that found anything found its first point within
  four rounds.
- `--hold-barren` gives a region re-entered with unchanged bounds, after a
  visit that found nothing, no more time than that visit while some other
  region is open; the time goes to the regions after it.

**A budget returns a partial front, and says exactly how partial.**
`--max-secs` and `--max-rss-gb` bound this second part, and with
`--modular` the constrained walk before it still runs whole: a
walk stopped there would narrow this search without the header saying so.
When either runs out, the run still writes every
modular point it found, each with a `status`:

- `certified` — proven on the front: no region the budget left open can hold
  a design that dominates it;
- `provisional` — modular and dominated by nothing found, but a
  region still open could hold a design that dominates it.

The header says `complete`, `stopped_by` (`time`, `memory`, `signal`,
`empty-rounds` or `null`) and lists
every region left `open`: a CAI interval (`cai_log_lo`..`cai_log_hi`, `null` for
an open end) and the least energy a missing point there can have
(`mfe_floor` at `cai_log_lo`, rising by `slope` per unit of `cai_log`; `null`
when only the constrained front bounds it).

**A run ended from outside writes its front too.** `SIGINT`, `SIGTERM` and
`SIGHUP` stop the search where a spent `--max-secs` would have stopped it, so
a run killed by an operator, by a job manager or at the end of a lease leaves
the same partial, labelled answer rather than nothing: `complete: false` and
`stopped_by: "signal"`, with every region it had not closed `open`. The run
then exits 0, as a run stopped by its budget does; a second signal goes
through and ends it at once, which is what to send if the front it is writing
is not wanted. The stop reaches the search's walks, so a run signalled during
its certification stage certifies the points it has left — `certify_s`
measures that — before the front comes out. `SIGKILL` cannot be caught, and a
run ended by one leaves `OUT.search.jsonl` and no result, which is what
`--checkpoint` is for.

**Three counts, and the summary line names each.** They coincide on a run
that certifies every point and finds nothing, so one word for two of them
reads correctly there and nowhere else:

```text
300 point(s) on the modular front, 298 of them certified, from 293 modular seed(s)
```

| the count | where it is in the result | what it counts |
|---|---|---|
| **points on the modular front** | header `points` | the points the run emitted: one line each |
| **of them certified** | the points' own `status` field | those of the emitted points that are `certified` rather than `provisional` |
| **modular seed(s)** | header `seeds` | the constrained-front points whose own check passed, which seeded the search. `constrained_points` is how many were offered and `unchecked` how many the budget never reached |

So certification's share of a front is `seeds / points`, and the certified
count is a statement about what is still open rather than about what passed a
check.

### Keeping the constrained front

The constrained walk that runs first is deterministic and is most of what a
budgeted run spends, so it can be kept and reused instead of repeated:

```bash
# Walk once, keeping the constrained front beside the answer.
derna front --protein hbb.fasta --utr5 u5.fa --utr3 u3.fa --modular \
  --max-secs 600 --front-out relaxed.jsonl --out so.jsonl
# A later run of the same molecule starts from it.
derna front --protein hbb.fasta --utr5 u5.fa --utr3 u3.fa --modular \
  --max-secs 3600 --front-from relaxed.jsonl --out so2.jsonl
```

`--front-out` writes the constrained front in the format a plain `front`
writes, **before** the modular walk begins, so a run stopped by an
outer limit still leaves it behind. `--front-from` reads such a file in place of
walking, and is refused unless the file's header names the same molecule —
protein, flanks, stop codon, constraint and codon table — saying which field
differs, because a front walked for another molecule would silently answer a
different question. On one 200-residue molecule behind 25-nucleotide flanks the
walk takes 462 s and reading it back takes 0.01 s, for the same 757 points.

`--front-from` spares only the constrained front beneath the search. To keep
the search itself, run with `--checkpoint`: every constrained point checked,
every modular design found and every slice of the top band
finished is appended to `OUT.search.jsonl` as it settles. A killed run loses
only what was in flight. Running the same command again continues from it:
checked points are not checked again, found designs are kept, and the band
starts where it stopped. The rounds of window walks start again, capped by
what was found. A walked constrained front is kept in `OUT.walk.jsonl` as it
settles, so the rerun continues that walk too rather than walking it again.
`--resume-search PATH` continues from another run's log instead. A log names
the molecule it was written for on its first line, and one of another
molecule or constraint is refused.

```bash
derna front --protein hbb.fasta --utr5 u5.fa --utr3 u3.fa --modular \
  --front-from relaxed.jsonl --max-secs 3600 --checkpoint --out so.jsonl
# killed or stopped by the budget; the same command continues where it stopped
derna front --protein hbb.fasta --utr5 u5.fa --utr3 u3.fa --modular \
  --front-from relaxed.jsonl --max-secs 3600 --checkpoint --out so.jsonl
```

The header's `resumed_search` records what a run started from.

### Within ε of modular

Some molecules have no modular design at all, or none at the
adaptations wanted, and the front is then empty or short. `--epsilon E` asks a
different question: the front of the designs whose gap is at most `E` kcal/mol.

```bash
derna front --protein hbb.fasta --utr5 u5.fa --utr3 u3.fa --modular \
  --epsilon 0.1 --max-secs 3600 --out near.jsonl
```

**It is its own search, not the modular one with near misses
added.** Every step above is a domination argument, and a design dominates
nothing in a set it is not a member of, however small its gap. So `E` reaches
every check together: the constrained-front points' (seeds are the points with
gap at most `E`), the walks' and the band's. A design within `E` caps, covers
and certifies only designs within `E`, and a modular design that a
design within `E` dominates is not on this front. At `E = 0` the run is the
modular run, line for line. Search progress is tied to its `E`:
`OUT.search.jsonl` writes `epsilon` on every line, and `--resume-search` refuses
a file searched within another `E`. Anchors (`--anchors-from`) are checked
within the run's `E`.

**Choosing `E`.** Energies are whole hundredths of a kcal/mol, and on every cell
measured every gap was a whole tenth: no design had a gap strictly between 0
and 0.1. There, 0.1 is the smallest `E` that admits anything beyond `E = 0`. It
is the energy grid's step, not a tuning knob. Gaps also clump: on one cell 311
near-miss designs all sat at a gap of exactly 0.80. The front therefore changes
in steps as `E` grows, with plateaus between them, and the run's last stderr
line counts the points at each gap (`by gap 0.00×12 0.10×7`), which the points'
own `gap` fields give again.

**What it does not say.** A point within `E` says how close it came, not why it
stops short of 0. Every gap above 0 means the free fold uses a pair the
constraint forbids, and the point's `violations` lists those pairs. Some such
pairs no codon choice can remove: on one cell 86 designs sat 0.6 above
tightness, each blocked by one pair joining the initiator codon's first base to
the 3′ UTR. Neither base can move, since the initiator has one codon and the
flank is given. Those designs are out of reach at any `E` below 0.6. That
certificate is a separate claim from this front and is not computed here.
What each point does carry is `fixed_violations`: those of its `violations`
whose two bases are the same in every design. Such a base is a flank base, a
stop codon base, or a codon position where every codon of its residue agrees.
A pair listed there is one no codon choice can remove. Whether some other
design avoids it in its free fold is a further question this front does not
answer.

**What it reports.** Every point carries `epsilon` beside its `gap`, and its
`modular` is true only where the gap is 0. The header carries
`epsilon` and an `outcome`:

| `outcome` | the front |
|---|---|
| `tight` | holds points, and only modular designs were admitted (`E = 0`) |
| `within_epsilon` | holds points, each with gap at most `E > 0`; a point's own `gap` says whether it is tight |
| `none_tight` | is empty: no design was modular |
| `none_within_epsilon` | is empty: no design was within `E` (so none was modular either) |

An empty front is proven empty only when the header's `complete` is true;
with `complete` false it means nothing was found within the budget.

`derna check` carries the `epsilon` into its own header. It folds each point
again, so a point with a gap above 0 comes back not modular, as
its `gap` already said. Its `status` keeps the front's meaning: `certified`
says the point is proven on the front within `E`, not that it passed the
check. `derna show` prints the `epsilon` and the count of
points at each gap under its summary.

### What an outer limit leaves behind

A modular run writes its points to `--out` as it emits them, not
at the end, so a run killed by an outer limit — a queue's `timeout`, a
scheduler, a reboot — leaves everything it had finished. Reading such a file
needs one check: the header's `points` is how many the front holds, so a file
whose point lines are fewer than that is a prefix and the rest was never
written. A whole file's counts agree.

The emission is not governed by `--max-secs`. That budget bounds the search for
the front; emitting the points it found is the stage after it, and the
constrained walk is the stage before. A run that must finish inside a hard wall
needs its own limit on each stage — walk once with `--out`, then run the
modular pass with `--front-from` — rather than one budget for all
three.

The constrained front is also a floor for every open region: a missing point at
CAI `A` cannot cost less than the constrained front's least energy at CAI `A` or
above, so a region the run left with no `mfe_floor` is still bounded by it once
the front has been kept.

**What it costs, measured.** On the [worked example](#a-worked-example) —
20 residues behind 25-nucleotide flanks, where 13 of the constrained front's
19 points pass and seed the search — a 60-second budget on four threads
returns 16 certified points with one region left open (1.3 GB peak, a loaded
shared host). A 600-second budget returns the same 16 points with that region
still open, so the time beyond the first minute goes to the band above the
best-adapted point and finds nothing there. Correctness is tested against exhaustive
enumeration of every design on molecules small enough to fold them all,
including one with no seed and one with no modular design at all.
At hundreds of residues expect a budget to run out; the partial front is the
answer to plan around.

## A worked example

These commands run as written, from the repository root. The protein is a
20-residue fixture and the flanks are the HBB pair trimmed to 25 nucleotides
each. Every step but the last finishes in seconds. The last, the modular
front, is given a 60-second budget and **stops on it**. It is shown as a
stopped run, because that is what a budget returns.

```bash
U5=ACAUUUGCUUCUGACACAACUGUGU
U3=GCUCGCUUUCUUGCUGUCCAAUUUC
P=packages/derna-codon-dp/tests/data/GLPE_HUMAN_20.fasta
OUT=$(mktemp -d)                 # results land here, not in the repository

# One design, at a weight favouring structure, under the default constraint.
derna design --protein $P --utr5 $U5 --utr3 $U3 --lambda 0.1 \
  --out $OUT/design.jsonl
derna show $OUT/design.jsonl --sequences

# The whole front, with the walk's diagnostics beside it. Under a
# constraint the front certifies its own points.
derna front --protein $P --utr5 $U5 --utr3 $U3 \
  --out $OUT/front.jsonl --report $OUT/walk.json
derna show $OUT/front.jsonl --modular

# One sequence to order: the knee of the modular points.
derna pick $OUT/front.jsonl --knee --modular

# `check` is for a front written without its check.
derna front --protein $P --utr5 $U5 --utr3 $U3 --no-check \
  --out $OUT/plain.jsonl --quiet
derna check $OUT/plain.jsonl --out $OUT/checked.jsonl --quiet
derna show $OUT/checked.jsonl --modular

# The front of the designs that pass, within one minute, and its plot.
derna front --protein $P --utr5 $U5 --utr3 $U3 --modular \
  --max-secs 60 --threads 4 --out $OUT/so.jsonl
derna show $OUT/so.jsonl --plot $OUT/so.svg
derna pick $OUT/so.jsonl --best-cai --dna --cds-only
```

The design at λ = 0.1 comes back **violating** the constraint. Its own
preferred fold forms pairs the constraint forbids, and the check reports this
rather than repairing it:

```text
[1/1] λ=0.1 mfe=-27.10 cai=0.9857 (0.18s)  violates the constraint: free MFE -29.60, gap 2.50, 17 forbidden pair(s)
1 design(s), 0 of 1 modular
```

`design` then prints the table of what it wrote to stderr. `show --sequences`
prints the same table with the sequence, the structure and the metrics under
the row:

```text
       mfe       cai     gc      gap  modular   hull    lambda  status
    -27.10    0.9857   0.50     2.50       no    yes  0.100000  —
    seq  ACAUUUGCUUCUGACACAACUGUGUAUGUACGGCAAGAUCAUCUUCGUGCUGCUGCUGAGCGGCAUCGUGAGCAUCUCUGCCAGCUAGGCUCGCUUUCUUGCUGUCCAAUUUC
    dot  ............(((((((..(((....(((((((((((..((..(((((((((....))))))).)).))..)))).)))).).))....)))....))).)))).......
    gc   cds 0.556  u(cds) 0.254  window 0.440–0.640  homopolymer 3
```

The front has 19 points, and `front` certifies them itself because the
constraint forbids pairs. **13 of the 19 are modular**:

```text
19 point(s), 9 on the hull, 8 gap(s), 11 sweep fold(s) (2.60s), spanning 11.2 kcal/mol
13 of 19 point(s) modular
```

The violations sit at the two ends. The two lowest-energy points (−34.40 and
−34.20 kcal/mol) fail at a gap of 0.70. The four best-adapted points, from CAI
0.9857 to 1.0000, fail at gaps of 2.50 to 11.10. The thirteen points between
them, from −34.00 to −29.10 kcal/mol, all pass. `show --modular` lists those
thirteen, and `pick --knee --modular` takes one of them:

```text
>GLPE_HUMAN_20aa_cut point=7 mfe=-33.10 cai=0.8769 gc=0.45
ACAUUUGCUUCUGACACAACUGUGUAUGUAUGGCAAGAUUAUCUUUGUGCUGCUGCUGAGCGGCAUAGUGAGUAUCUCUGCCAGCUAAGCUCGCUUUCUUGCUGUCCAAUUUC
```

Running `check` on that front would only repeat its verdicts. `check` is for a
front written with `--no-check`. On this cell it gives the same `13 of 19
point(s) modular`. `show --modular` on the unchecked file shows no point and
adds a note saying so. Without the note, "0 shown" would read as *no point
passed*, when the truth is *no point was asked*.

The modular front starts from the 13 certified points as seeds. It finds three
more within the budget: −34.10 kcal/mol at CAI 0.7904 and −34.20 at 0.7657,
below the lowest seed's energy, and −24.80 at 0.9772, past the
best-adapted seed. Then the budget runs out:

```text
16 point(s) on the modular front, 16 of them certified, from 13 modular seed(s); stopped by the time budget with 1 region(s) open (62.67s)
```

All 16 are `certified`: no region the budget left open can hold a design that
dominates any of them. The three counts are the three of the table above, and
on this cell two of them read 16. The open region is the band above the
best-adapted point, where a better-adapted modular design could still exist.
A ten-minute budget (`--max-secs 600`) gives the same 16 points with the same
region still open, so the extra nine minutes buy nothing on this cell.
`show --plot` draws the 16 as filled circles. The last `pick` writes the best-adapted one's coding
sequence, start codon to stop, as DNA:

```text
>GLPE_HUMAN_20aa_cut point=16 mfe=-24.80 cai=0.9772 gc=0.50
ATGTACGGCAAGATCATCTTCGTGCTGCTGCTGAGCGGCATCGTGTCCATCTCTGCCTCCTAA
```

The header's `gc` is the whole molecule's, UTRs included.

## Not yet on the command

These are the gaps worth knowing about, not a complete inventory: the
command's own `--help` is the authority on what exists, and it is tested in
both directions — every flag the parser accepts appears in the help, and no
help text names a flag its command refuses.

- **`front --pin` without `--motif`**, which needs the plain walk to run over
  the pin's restricted lattice; with `--motif` the pins reach both the motif
  front and its modular front.
- **Positional structure constraints** (`--structure`, `--helix`, `--unpaired`,
  `--forbid-pair`): no design emitted under them carries a free-fold check, so
  a design whose own fold re-pairs an unpaired window would come back
  unflagged. They stay on the instruments until the check covers them.

The `derna-lib` library keeps every constraint this command withholds, for
callers who drive it directly.
