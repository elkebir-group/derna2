//! The bundled codon usage tables, against the numbers their source gives.
//!
//! Five organisms ship inside the binary and `--codon-usage` takes any of
//! their names. A test that only checks a table parses would pass on the wrong
//! file, on the same file bundled five times, and on a digit typed wrong, so
//! each table here is asserted against values transcribed from its own source
//! entry — including one amino-acid family's relative adaptiveness, which is
//! what CAI actually scores against and which no structural check reaches.
//!
//! The transcribed values below are the record, and they are not all from the
//! same place, because the tables are not. `mouse`, `cho`, `ecoli` and `yeast`
//! were taken from the Codon Usage Database entry named for each in
//! `data/PROVENANCE.md` beside the CSVs, and every one was reproduced from that
//! entry's occurrence-count column — which shares no digits with the fraction
//! column — before it was committed. `human` is the older frozen snapshot that
//! predates those entries, so its values are the ones the vendored table has
//! always carried; `PROVENANCE.md` says why it stays frozen. A later edit to
//! any of these CSVs has to restate the values here, which is the point.

use derna_cli::input::{bundled_codon_usage_names, read_codon_usage};
use derna_codon_dp::cai::compute_cai_tables;
use derna_codon_dp::codon_table::{aa_to_index, AMINO_ACIDS, STOP};

/// The bundled names, in the order the flag's help lists them.
const NAMES: [&str; 5] = ["human", "mouse", "cho", "ecoli", "yeast"];

/// One amino-acid family's fractions, as its source entry gives them: the
/// table name, the amino acid, and each codon with its fraction.
struct Family {
    table: &'static str,
    amino_acid: char,
    codons: &'static [(&'static str, f64)],
}

/// A family per bundled table, chosen so that the family's values differ from
/// every other bundled table's for the same amino acid. Reading a different
/// organism's file under one of these names therefore fails the assertion
/// rather than passing on a table that happens to parse.
const FAMILIES: &[Family] = &[
    Family {
        table: "human",
        amino_acid: 'R',
        codons: &[
            ("CGU", 0.08),
            ("CGC", 0.19),
            ("CGA", 0.11),
            ("CGG", 0.21),
            ("AGA", 0.20),
            ("AGG", 0.20),
        ],
    },
    Family {
        table: "mouse",
        amino_acid: 'L',
        codons: &[
            ("CUU", 0.13),
            ("CUC", 0.20),
            ("CUA", 0.08),
            ("CUG", 0.39),
            ("UUA", 0.07),
            ("UUG", 0.13),
        ],
    },
    Family {
        table: "cho",
        amino_acid: 'A',
        codons: &[("GCU", 0.32), ("GCC", 0.37), ("GCA", 0.23), ("GCG", 0.07)],
    },
    Family {
        table: "ecoli",
        amino_acid: 'R',
        codons: &[
            ("CGU", 0.30),
            ("CGC", 0.26),
            ("CGA", 0.09),
            ("CGG", 0.15),
            ("AGA", 0.13),
            ("AGG", 0.07),
        ],
    },
    Family {
        table: "yeast",
        amino_acid: 'R',
        codons: &[
            ("CGU", 0.14),
            ("CGC", 0.06),
            ("CGA", 0.07),
            ("CGG", 0.04),
            ("AGA", 0.48),
            ("AGG", 0.21),
        ],
    },
];

/// Named codons outside the families above, so no table rests on one family.
/// `(table, amino acid, codon, fraction)`, transcribed from the same entries.
const SPOT_VALUES: &[(&str, char, &str, f64)] = &[
    ("human", 'Q', "CAG", 0.75),
    ("human", 'F', "UUU", 0.45),
    ("mouse", 'Q', "CAG", 0.74),
    ("mouse", 'K', "AAG", 0.61),
    ("mouse", 'S', "UCG", 0.05),
    ("cho", 'Q', "CAG", 0.76),
    ("cho", 'I', "AUC", 0.51),
    ("cho", 'P', "CCG", 0.08),
    ("ecoli", 'Q', "CAG", 0.65),
    ("ecoli", 'K', "AAA", 0.71),
    ("ecoli", 'Y', "UAU", 0.65),
    ("yeast", 'Q', "CAA", 0.69),
    ("yeast", 'G', "GGU", 0.47),
    ("yeast", 'E', "GAA", 0.70),
];

/// A bundled table's rows, as the file lists them.
fn rows_of(name: &str) -> Vec<(String, String, f64)> {
    read_codon_usage(Some(name))
        .expect("a bundled table resolves")
        .rows
}

/// Resolve `(amino acid, codon)` to the pair of indices the tables are keyed
/// on, by the codon's name rather than by its spelling.
fn indices(amino_acid: char, codon: &str) -> (usize, usize) {
    let aa = aa_to_index(amino_acid).expect("a known amino acid");
    let ci = AMINO_ACIDS[aa]
        .codons
        .iter()
        .position(|c| c.name == codon)
        .unwrap_or_else(|| panic!("{codon} is not a synonym of {amino_acid}"));
    (aa, ci)
}

/// (a) Every bundled table's CAI weights are the ones its source implies.
///
/// The weight CAI scores against is a codon's fraction over the largest in its
/// family, so this asserts the derived quantity rather than the parsed column —
/// a table whose rows were scaled, reordered or taken from another organism
/// gives different weights even where it gives the same row count.
#[test]
fn each_bundled_tables_cai_weights_reproduce_its_sources_numbers() {
    for family in FAMILIES {
        let table = read_codon_usage(Some(family.table)).expect("a bundled table resolves");
        let cai = compute_cai_tables(&table.usage);
        let largest = family
            .codons
            .iter()
            .map(|(_, f)| *f)
            .fold(f64::MIN, f64::max);

        for (codon, source_fraction) in family.codons {
            let (aa, ci) = indices(family.amino_acid, codon);
            assert!(
                (table.usage.fraction(aa, ci) - source_fraction).abs() < 1e-9,
                "{}: {}/{codon} is {} where its source gives {source_fraction}",
                family.table,
                family.amino_acid,
                table.usage.fraction(aa, ci),
            );
            let weight = cai.log_cai[aa][ci].exp();
            let from_source = source_fraction / largest;
            assert!(
                (weight - from_source).abs() < 1e-9,
                "{}: the CAI weight of {}/{codon} is {weight} where its source gives \
                 {from_source} ({source_fraction}/{largest})",
                family.table,
                family.amino_acid,
            );
        }

        // The family's most-adapted codon is the one the source makes largest,
        // which is what the DP's per-family reference weight is taken from.
        let (best, _) =
            family.codons.iter().fold(
                ("", f64::MIN),
                |acc, (c, f)| {
                    if *f > acc.1 {
                        (*c, *f)
                    } else {
                        acc
                    }
                },
            );
        let (aa, ci) = indices(family.amino_acid, best);
        assert_eq!(
            cai.max_cai_pos[aa], ci,
            "{}: {}'s most frequent codon is not {best}",
            family.table, family.amino_acid,
        );
    }

    for (name, amino_acid, codon, source_fraction) in SPOT_VALUES {
        let table = read_codon_usage(Some(name)).expect("a bundled table resolves");
        let (aa, ci) = indices(*amino_acid, codon);
        assert!(
            (table.usage.fraction(aa, ci) - source_fraction).abs() < 1e-9,
            "{name}: {amino_acid}/{codon} is {} where its source gives {source_fraction}",
            table.usage.fraction(aa, ci),
        );
    }
}

/// Every bundled name resolves to a *different* table.
///
/// Without this, bundling one organism's file under five names passes every
/// other test in this file that does not name a value: the row count, the
/// family structure and the fraction sums are all identical across organisms.
#[test]
fn the_five_bundled_names_are_five_different_tables() {
    for (i, a) in NAMES.iter().enumerate() {
        for b in &NAMES[i + 1..] {
            assert_ne!(rows_of(a), rows_of(b), "{a} and {b} are the same table");
        }
    }
}

/// (b) Every bundled table has the structure the loader requires: one row per
/// sense codon over the twenty amino acids, and the same order throughout, so
/// the row list a result file embeds is comparable from one table to the next.
#[test]
fn every_bundled_table_covers_the_twenty_families_in_one_order() {
    let reference = read_codon_usage(Some("human")).expect("resolves").rows;
    let order: Vec<(&String, &String)> = reference.iter().map(|(a, c, _)| (a, c)).collect();

    for name in NAMES {
        let table = read_codon_usage(Some(name)).expect("a bundled table resolves");
        assert_eq!(table.source, name, "{name} records itself as its source");
        assert_eq!(
            table.rows.len(),
            61,
            "{name} does not hold the 61 sense codons"
        );

        let mut families: Vec<&String> = table.rows.iter().map(|(aa, _, _)| aa).collect();
        families.dedup();
        assert_eq!(families.len(), 20, "{name} does not cover 20 amino acids");

        let seen: Vec<(&String, &String)> = table.rows.iter().map(|(a, c, _)| (a, c)).collect();
        assert_eq!(seen, order, "{name} lists its codons in another order");

        // No stop row is listed, and the stop family still carries one weight
        // for each stop codon: a stop carries no adaptation, so its CAI weight
        // is 1 and it adds nothing to any design's score.
        assert!(
            !table.rows.iter().any(|(aa, _, _)| aa == "*"),
            "{name} lists stop codons, which this format does not"
        );
        let cai = compute_cai_tables(&table.usage);
        for ci in 0..AMINO_ACIDS[STOP].codons.len() {
            assert_eq!(cai.log_cai[STOP][ci], 0.0, "{name}: a stop codon is scored");
        }
    }
}

/// (c) Every family's fractions behave as the loader and CAI require: they sum
/// to one up to the source's two-decimal rounding, and none is zero.
///
/// A zero would not fail anything — `compute_cai_tables` floors it at
/// `EPSILON` — it would quietly make that codon about eleven log units worse
/// than its rarest real synonym and so unusable, which is why it is asserted
/// here rather than left to the floor.
#[test]
fn every_family_sums_to_one_and_no_codon_has_zero_weight() {
    for name in NAMES {
        let table = read_codon_usage(Some(name)).expect("a bundled table resolves");
        let mut family = String::new();
        let mut total = 0.0f64;

        let close = |family: &str, total: f64| {
            if !family.is_empty() {
                assert!(
                    (total - 1.0).abs() <= 0.02,
                    "{name}: {family}'s fractions sum to {total}, not one"
                );
            }
        };

        for (aa, codon, fraction) in &table.rows {
            if *aa != family {
                close(&family, total);
                family = aa.clone();
                total = 0.0;
            }
            total += fraction;
            assert!(
                *fraction > 0.0,
                "{name}: {aa}/{codon} has zero weight, so CAI cannot use it"
            );
        }
        close(&family, total);
    }
}

/// (d) A name that is neither bundled nor a readable file is refused, and the
/// refusal names the bundled tables.
///
/// Before the tables were bundled every unknown name fell through to a file
/// read, so a typo reported only that a file of that name did not exist and
/// never that a table of nearly that name existed.
#[test]
fn an_unknown_table_name_is_refused_naming_the_bundled_ones() {
    let message = read_codon_usage(Some("mouse2"))
        .map(|_| ())
        .expect_err("an unknown name is refused")
        .to_string();
    assert!(message.contains("--codon-usage"), "{message}");
    assert!(message.contains("mouse2"), "{message}");
    for name in NAMES {
        assert!(
            message.contains(name),
            "the refusal does not name {name}: {message}"
        );
    }

    // A path that does not exist is refused the same way, since a misspelled
    // name and a mistyped path are the same input here.
    let message = read_codon_usage(Some("/nonexistent/table.csv"))
        .map(|_| ())
        .expect_err("an unreadable path is refused")
        .to_string();
    assert!(message.contains("--codon-usage"), "{message}");
    assert!(message.contains("/nonexistent/table.csv"), "{message}");

    assert_eq!(bundled_codon_usage_names(), NAMES.join(", "));
}
