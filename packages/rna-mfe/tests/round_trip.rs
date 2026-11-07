//! Parameter round trips, in both layouts the parser reads.
//!
//! - **Canonical layout.** The built-in table, `data/turner2004.params`, is
//!   written in the layout [`Params::dump_canonical`] emits, so it is a fixed
//!   point: parse, dump, and the bytes come back. A bug in any axis ordering
//!   or field assignment of the parser or the dumper surfaces as a diff.
//! - **`.par` layout.** The same values written as a `.par` file — `##`
//!   header, a `/* ... */` comment on every line, and every section this
//!   crate skips present with its full length — parse to the same `Params`.
//!   [`as_par`] builds that file from the dump, so the test needs no file
//!   from elsewhere.
//!
//! Spot checks pin a few values of the built-in table, so a table that
//! round-trips but holds the wrong numbers fails too.

use rna_mfe::params::{BasePair, Nuc5};
use rna_mfe::Params;

const BUILT_IN: &str = include_str!("../data/turner2004.params");

#[test]
fn the_built_in_table_is_a_fixed_point_of_parse_then_dump() {
    let dumped = Params::load_default().unwrap().dump_canonical();
    assert!(
        dumped == BUILT_IN,
        "dump_canonical(load_default()) differs from data/turner2004.params; first \
         differing line: {:?}",
        dumped.lines().zip(BUILT_IN.lines()).find(|(a, b)| a != b)
    );
}

#[test]
fn dump_round_trips_to_equal_params() {
    let built_in = Params::load_default().expect("built-in table parses");
    let reparsed = Params::parse(&built_in.dump_canonical()).expect("dump parses");
    assert_eq!(
        built_in, reparsed,
        "Params changed across parse → dump → parse cycle"
    );
}

#[test]
fn a_par_layout_file_reads_as_the_same_params() {
    let built_in = Params::load_default().unwrap();
    let par = as_par(&built_in.dump_canonical());
    // The conversion did something: a skipped section and a comment are there.
    assert!(par.contains("# dangle5\n") && par.contains("/*"));
    assert_eq!(Params::parse(&par).expect(".par layout parses"), built_in);
}

#[test]
fn from_file_reads_either_layout() {
    let dir = std::env::temp_dir().join(format!("rna-mfe-round-trip-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let built_in = Params::load_default().unwrap();
    let canonical = dir.join("canonical.params");
    let par = dir.join("table.par");
    std::fs::write(&canonical, built_in.dump_canonical()).unwrap();
    std::fs::write(&par, as_par(&built_in.dump_canonical())).unwrap();
    assert_eq!(Params::from_file(&canonical).unwrap(), built_in);
    assert_eq!(Params::from_file(&par).unwrap(), built_in);
    std::fs::remove_file(&canonical).unwrap();
    std::fs::remove_file(&par).unwrap();
    std::fs::remove_dir(&dir).unwrap();
}

#[test]
fn from_file_names_a_file_it_cannot_read() {
    let e = Params::from_file("/nonexistent/turner.params").unwrap_err();
    assert!(e.to_string().contains("/nonexistent/turner.params"), "{e}");
}

#[test]
fn a_skipped_section_of_the_wrong_length_is_refused() {
    let par = as_par(&Params::load_default().unwrap().dump_canonical()).replacen(
        "# dangle5\n",
        "# dangle5\n0\n",
        1,
    );
    assert!(Params::parse(&par).is_err());
}

/// Misc: the terminal AU/GU penalty is 0.50 kcal/mol; duplex initiation and
/// the enthalpies, which a single-strand fold at 37 °C does not read, are 0.
#[test]
fn misc_fields_match_file_layout() {
    let m = Params::load_default().unwrap().misc;
    assert_eq!(
        (
            m.duplex_init_37,
            m.duplex_init_dh,
            m.terminal_au_37,
            m.terminal_au_dh
        ),
        (0, 0, 50, 0),
        "[DuplexInit37, DuplexInitdH, TerminalAU37, TerminalAUdH]",
    );
}

/// ML_params: per-unpaired (`base`), initiation (`closing`) and per-branch
/// (`intern`) energies of the linear multiloop model.
#[test]
fn ml_params_match_known_constants() {
    let p = Params::load_default().unwrap();
    assert_eq!(p.ml.base_37, 0, "ML_BASE37 (per-unpaired)");
    assert_eq!(p.ml.closing_37, 930, "ML_closing37 (init)");
    assert_eq!(p.ml.intern_37, -90, "ML_intern37 (per-branch)");
}

/// NINIO: `[ninio37, niniodH, MAX_NINIO]` = 0.6 per nucleotide of
/// asymmetry, capped at 3.0.
#[test]
fn ninio_fields_match_file_layout() {
    let p = Params::load_default().unwrap();
    assert_eq!(p.ninio.ninio_37, 60);
    assert_eq!(p.ninio.max_ninio, 300);
}

/// Stack table: `(CG, CG)` is `-240` and `(GC, GC)` is `-340`.
#[test]
fn stack_anchors_match_file() {
    let p = Params::load_default().unwrap();
    assert_eq!(p.stack.get(BasePair::CG, BasePair::CG), -240);
    assert_eq!(p.stack.get(BasePair::GC, BasePair::GC), -340);
    // Diagonal symmetry (the stack table is symmetric by construction).
    assert_eq!(
        p.stack.get(BasePair::CG, BasePair::GC),
        p.stack.get(BasePair::GC, BasePair::CG),
    );
}

/// Hairpin size table: `INF` for sizes 0/1/2 (no hairpin can be
/// shorter than 3 nt), then 540 at size 3, and 770 at size 30 (the last
/// explicit entry).
#[test]
fn hairpin_size_table_anchors() {
    let v = Params::load_default().unwrap().hairpin.values;
    let saw_inf = (0..3).all(|i| v[i] >= 1_000_000);
    assert_eq!(
        (saw_inf, v[3], v[30]),
        (true, 540, 770),
        "(sizes 0..3 = INF, size 3, size 30)",
    );
}

/// Special hairpin tables (16 tetraloop entries, 4 hexaloops, 2 triloops).
/// None of these is exercised by the property corpus, so this load test is
/// the round-trip suite's only validation of the special-loop parser.
#[test]
fn special_hairpin_tables_load() {
    let p = Params::load_default().unwrap();
    let counts = (
        p.tetraloops.entries.len(),
        p.hexaloops.entries.len(),
        p.triloops.entries.len(),
    );
    let lookups = (
        p.tetraloops.lookup_37(b"CCAAGG"),
        p.hexaloops.lookup_37(b"ACAGUACU"),
        p.triloops.lookup_37(b"CAACG"),
        p.tetraloops.lookup_37(b"GAAA"),
    );
    assert_eq!(
        (counts, lookups),
        ((16, 4, 2), (Some(330), Some(280), Some(680), None)),
        "(table sizes, anchor lookups)",
    );
}

/// Mismatch table: `mismatch_hairpin[CG, N, N] = -80`, the top-left cell
/// of the first 7×5×5 plane. An `N` cell holds the largest value over the
/// bases it stands for.
#[test]
fn mismatch_hairpin_anchor() {
    let p = Params::load_default().unwrap();
    assert_eq!(p.mismatch_hairpin.get(BasePair::CG, Nuc5::N, Nuc5::N), -80);
}

/// int22's first row: `120 160 20 160` (closing CG, inner CG, mismatch
/// nucleotides A, A, A, then the fourth over A, C, G, U).
#[test]
fn int22_first_row_anchor() {
    let f = &Params::load_default().unwrap().int22.flat;
    assert_eq!([f[0], f[1], f[2], f[3]], [120, 160, 20, 160]);
}

/// The sections a `.par` file carries that this crate skips, after the read
/// section each follows, with their lengths.
const SKIPPED_AFTER: &[(&str, &[(&str, usize)])] = &[
    ("stack", &[("stack_enthalpies", 49)]),
    (
        "mismatch_interior_23",
        &[
            ("mismatch_multi", 175),
            ("mismatch_exterior", 175),
            ("dangle5", 35),
            ("dangle3", 35),
        ],
    ),
    ("int22", &[("int22_enthalpies", 9216)]),
    ("interior", &[("interior_enthalpies", 31)]),
];

/// The canonical dump rewritten in the `.par` layout: a `##` header, a
/// `/* ... */` comment closing every value line, and the skipped sections
/// [`SKIPPED_AFTER`] names, filled with zeros, after the section each
/// follows.
fn as_par(canonical: &str) -> String {
    let mut out = String::from("## parameter file v2.0\n");
    let mut current = "";
    for line in canonical.lines().skip(1) {
        if let Some(name) = line.strip_prefix("# ") {
            push_skipped_after(&mut out, current);
            current = name;
            out.push_str(line);
        } else if line.is_empty() || line.starts_with('#') {
            out.push_str(line);
        } else {
            out.push_str(&format!("{line}    /* {current} */"));
        }
        out.push('\n');
    }
    out
}

fn push_skipped_after(out: &mut String, section: &str) {
    let Some((_, skipped)) = SKIPPED_AFTER.iter().find(|(after, _)| *after == section) else {
        return;
    };
    for (name, len) in *skipped {
        out.push_str(&format!("\n# {name}\n"));
        out.push_str(&vec!["0"; *len].join(" "));
        out.push('\n');
    }
}
