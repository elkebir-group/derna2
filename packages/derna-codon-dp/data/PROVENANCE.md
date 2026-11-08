# Codon-usage tables: where each one came from

Every file here is `amino_acid,codon,fraction` with one row per sense codon, 61
rows over the 20 amino acids, in the same family and within-family order. Stop
codons are not listed: a stop carries no adaptation, so the loader fills the
stop family's row with a weight of 1 after parsing.

`fraction` is the relative frequency of a codon among the synonyms of its amino
acid, rounded to two decimals as the source gives it. A family's fractions
therefore sum to 1 only up to that rounding.

## The tables

| File | `--codon-usage` | Organism, as the source names it | Source id | CDS | Total codons |
|---|---|---|---|---|---|
| `codon_usage_human.csv` | `human` | *Homo sapiens* | — (see below) | — | — |
| `codon_usage_mouse.csv` | `mouse` | *Mus musculus* [gbrod] | `species=10090` | 53,036 | 24,533,776 |
| `codon_usage_cho.csv` | `cho` | *Cricetulus griseus* [gbrod] | `species=10029` | 331 | 153,527 |
| `codon_usage_ecoli.csv` | `ecoli` | *Escherichia coli* [gbbct] | `species=37762` | 8,087 | 2,330,943 |
| `codon_usage_yeast.csv` | `yeast` | *Saccharomyces cerevisiae* | `species=4932` | 14,411 | 6,534,504 |

All five are from the Codon Usage Database at the Kazusa DNA Research Institute
(<https://www.kazusa.or.jp/codon/>), maintained by Yasukazu Nakamura. Please
cite:

> Nakamura, Y., Gojobori, T. & Ikemura, T. (2000). Codon usage tabulated from
> international DNA sequence databases: status for the year 2000. *Nucleic
> Acids Research*, **28**, 292.

The database states no licence or terms of use; it asks for the citation above.
`NOTICE` in the published distribution carries it.

## The set spans two snapshots

**This is the thing to know before comparing a figure across organisms.**

`codon_usage_human.csv` is the Nakamura-2000-era table, transcribed from the
DERNA C++ default rather than fetched, and it is **deliberately frozen**: every
committed Pareto fixture was generated against it, so re-pinning it would
invalidate all of them.

The other four were retrieved on **2026-11-07** from
`showcodon.cgi?species=<id>&aa=1&style=N`, and carry the release every page of
the database now reports: **NCBI-GenBank Flat File Release 160.0 [15 June
2007]**. The database has not been updated since.

The two snapshots differ. The live `species=9606` human entry (93,487 CDS,
40,662,582 codons) disagrees with the frozen `human` file by 0.01–0.02 on about
half the codons — Gln `CAA/CAG` reads 0.25/0.75 frozen against 0.27/0.73 live,
Phe 0.45/0.55 against 0.46/0.54, Ala `GCU` 0.26 against 0.27. So a CAI
difference between `human` and one of the others is partly a difference of
corpus vintage, not only of organism.

## Two things the table above does not say on its own

**CHO rests on 331 CDS and 153,527 codons** — two orders of magnitude thinner
than mouse, and the thinnest basis of the five. It is the only *Cricetulus
griseus* entry the database has, and it is the table the field cites, but it is
not a peer of the others and a result that turns on it should say so.

**`ecoli` is the species-level entry, and claims no strain.** The database
splits *Escherichia coli* across 54 entries keyed on whatever strain string
GenBank recorded, and the one labelled `Escherichia coli K12` (`species=83333`)
holds **14 CDS / 5,122 codons** — a fragment, not a reference set, with a stop
fraction that rounds to zero. `species=37762`, used here, is the unqualified
species-level entry and the largest basis. The alternative, if a K-12 lineage
specifically is wanted, is `Escherichia coli W3110` (`species=316407`, 4,332
CDS), a K-12 derivative; it is not bundled, and a user who wants it can pass the
file through `--codon-usage PATH`.

## How the four fetched tables were checked

Each `fraction` was recomputed from the occurrence-count column of the same
source entry — `count(codon) / Σ count(synonyms)`, rounded to two decimals — and
compared against the transcribed value. The count column shares no digits with
the fraction column, so a pair of transcription errors that cancel within a
family cannot survive the comparison, which a family-sum check would admit. Each
entry's counts were also checked to sum to the total codon count the page
reports. All 244 fractions reproduced exactly.

`codon_usage_tables` in `packages/derna/tests/` asserts the transcribed values
that the comparison rests on, so a later edit to one of these files has to
restate them.
