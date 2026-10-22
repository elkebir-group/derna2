"""Translate the long-form Kazusa human codon usage CSV into the matrix-form
CSV that DERNA C++ expects via its `-c` flag.

Long-form input (`amino_acid,codon,fraction`) → matrix-form output, two lines
per amino acid: a header row of codons for that aa and a data row of the
corresponding fractions. C++ normalizes the fractions to `f / max(f)` per
amino acid internally (`default.cpp:1701`), so passing Kazusa fractions
directly is correct — no need to reconstruct integer counts.

The C++ reader (`fill_codon` in `default.cpp:1604`) keys each value into
`codon_usage[aa_index][codon_slot]` by codon name, so neither the order of
amino acids nor the order of codons within an amino acid affects the result.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def translate(input_path: Path, output_path: Path) -> None:
    by_aa: dict[str, list[tuple[str, str]]] = defaultdict(list)
    aa_order: list[str] = []
    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aa = row["amino_acid"]
            if aa not in by_aa:
                aa_order.append(aa)
            by_aa[aa].append((row["codon"], row["fraction"]))

    with output_path.open("w") as f:
        for aa in aa_order:
            codons = [codon for codon, _ in by_aa[aa]]
            fractions = [fraction for _, fraction in by_aa[aa]]
            f.write("," + ",".join(codons) + "\n")
            f.write(aa + "," + ",".join(fractions) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", maxsplit=1)[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("packages/derna-codon-dp/data/codon_usage_human.csv"),
        help="Long-form Kazusa codon usage CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="C++ matrix-form CSV output path",
    )
    args = parser.parse_args()
    translate(args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
