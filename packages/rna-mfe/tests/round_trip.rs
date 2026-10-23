//! Parameter round-trip test: parse upstream `rna_turner2004.par`, dump
//! the parsed `Params` back to a canonical text form, parse the dump,
//! assert equality. This is the load-bearing test for "the parser reads the
//! file faithfully and the dumper is its inverse" — a bug in any
//! axis ordering or field assignment surfaces as a round-trip diff.
//!
//! The dumper emits a *subset* of the upstream file (energies-only,
//! skipping enthalpies / dangles / mismatch_multi / mismatch_exterior),
//! so this is a *semantic* round-trip rather than byte-for-byte
//! identity against upstream. Token-stream equality on the parsed
//! sections is verified separately by spot-checking known constants.

use rna_mfe::params::{BasePair, Nuc5};
use rna_mfe::Params;

#[test]
fn dump_round_trips_to_equal_params() {
    let upstream = Params::load_default().expect("upstream parses");
    let dumped = upstream.dump_canonical();
    let reparsed = Params::parse(&dumped).expect("dump parses");
    assert_eq!(
        upstream, reparsed,
        "Params changed across parse → dump → parse cycle"
    );
}

#[test]
fn dump_is_its_own_fixed_point() {
    let upstream = Params::load_default().unwrap();
    let dump1 = upstream.dump_canonical();
    let dump2 = Params::parse(&dump1).unwrap().dump_canonical();
    assert_eq!(dump1, dump2, "dumper is not idempotent");
}

/// OQ2 spot check: the four Misc fields have the values the upstream
/// file ships and the names ViennaRNA's `read_epars.c` (`case MISC:`)
/// assigns. The on-disk Misc block is `410 360 50 370`. Field order
/// in the tuple matches Vienna's `values[0..=3]` assignment exactly.
#[test]
fn misc_fields_match_vienna_assignment() {
    let m = Params::load_default().unwrap().misc;
    assert_eq!(
        (
            m.duplex_init_37,
            m.duplex_init_dh,
            m.terminal_au_37,
            m.terminal_au_dh
        ),
        (410, 360, 50, 370),
        "[DuplexInit37, DuplexInitdH, TerminalAU37, TerminalAUdH]",
    );
}

/// ML_params spot check: per-unpaired (`base`), init (`closing`), and
/// per-branch (`intern`) energies match the values DERNA C++'s reference
/// uses (`ML_closing = 930`, `ML_intern = -90`, `TerminalAU = 50` —
/// upstream Turner 2004 constants vendored at `data/rna_turner2004.par`).
#[test]
fn ml_params_match_known_constants() {
    let p = Params::load_default().unwrap();
    assert_eq!(p.ml.base_37, 0, "ML_BASE37 (per-unpaired)");
    assert_eq!(p.ml.closing_37, 930, "ML_closing37 (init)");
    assert_eq!(p.ml.intern_37, -90, "ML_intern37 (per-branch)");
}

/// NINIO spot check: per ViennaRNA `read_epars.c` the on-disk values
/// are `[ninio37, niniodH, MAX_NINIO]` = `[60, 320, 300]`.
#[test]
fn ninio_fields_match_vienna_assignment() {
    let p = Params::load_default().unwrap();
    assert_eq!(p.ninio.ninio_37, 60);
    assert_eq!(p.ninio.ninio_dh, 320);
    assert_eq!(p.ninio.max_ninio, 300);
}

/// Stack table: row 0 col 0 is the `(CG, CG)` entry, which the file
/// reports as `-240`. Row 1 col 1 is `(GC, GC)` = `-340`.
#[test]
fn stack_anchors_match_file() {
    let p = Params::load_default().unwrap();
    assert_eq!(p.stack.get(BasePair::CG, BasePair::CG), -240);
    assert_eq!(p.stack.get(BasePair::GC, BasePair::GC), -340);
    // Diagonal symmetry (Vienna's `check_symmetry` enforces this on load).
    assert_eq!(
        p.stack.get(BasePair::CG, BasePair::GC),
        p.stack.get(BasePair::GC, BasePair::CG),
    );
}

/// Hairpin size table: `INF` for sizes 0/1/2 (no hairpin can be
/// shorter than 3 nt), then 540 at size 3 (the size-3 minimum), and
/// 770 at size 30 (the last explicit entry).
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

/// Special hairpin tables ship the values listed in the file (16
/// tetraloop entries, 4 hexaloops, 2 triloops). Per coverage.md G3,
/// none of these will be exercised by the property corpus; this load
/// test is therefore the round-trip suite's only validation of the
/// special-loop parser.
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

/// Mismatch table sanity check: `mismatch_hairpin[CG, N=0, N=0] = -80`
/// (top-left cell of the first 7×5×5 plane).
#[test]
fn mismatch_hairpin_anchor() {
    let p = Params::load_default().unwrap();
    assert_eq!(p.mismatch_hairpin.get(BasePair::CG, Nuc5::N, Nuc5::N), -80);
}

/// int22's first row: `120 160 20 160` per the upstream file's first
/// `# int22` block (closing CG, inner CG, mismatch nts A,A,A, with
/// the 4 displayed columns being the remaining-mismatch-nt loop).
#[test]
fn int22_first_row_anchor() {
    let f = &Params::load_default().unwrap().int22.flat;
    assert_eq!([f[0], f[1], f[2], f[3]], [120, 160, 20, 160]);
}

/// Token-stream round-trip against the upstream file. Walks the
/// upstream `rna_turner2004.par` section by section, retains only the
/// active (non-skipped) sections, and asserts the resulting numeric
/// token sequence matches the dump's numeric token sequence.
///
/// This is the strongest available check that the parser reads every value
/// correctly: a single misplaced value (axis-ordering bug, off-by-one
/// in the int22 layout, swapped Misc fields) surfaces as a token-
/// stream diff, even when individual spot checks pass.
#[test]
fn upstream_active_sections_match_dump_token_stream() {
    const UPSTREAM: &str = include_str!("../data/rna_turner2004.par");

    let upstream_active = active_token_stream(UPSTREAM);
    let dump = Params::load_default().unwrap().dump_canonical();
    let dump_tokens = active_token_stream(&dump);

    assert_eq!(
        upstream_active.len(),
        dump_tokens.len(),
        "active token count differs: upstream={}, dump={}",
        upstream_active.len(),
        dump_tokens.len(),
    );
    for (i, (u, d)) in upstream_active.iter().zip(&dump_tokens).enumerate() {
        assert_eq!(u, d, "token #{i} differs: upstream={u}, dump={d}");
    }
}

/// Active sections (energies + ML/NINIO/Misc + special hairpins) in
/// upstream order. Skipped sections (dangles=0 only, 37°C only):
/// enthalpies, dangles, mismatch_multi, mismatch_exterior.
const ACTIVE_SECTIONS: &[&str] = &[
    "stack",
    "mismatch_hairpin",
    "mismatch_interior",
    "mismatch_interior_1n",
    "mismatch_interior_23",
    "int11",
    "int21",
    "int22",
    "hairpin",
    "bulge",
    "interior",
    "ML_params",
    "NINIO",
    "Misc",
    "Hexaloops",
    "Tetraloops",
    "Triloops",
];

fn active_token_stream(source: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current: Option<&'static str> = None;
    for line in source.lines() {
        let trimmed = line.trim_start();
        if let Some(name) = trimmed
            .strip_prefix('#')
            .filter(|r| !r.starts_with('#'))
            .map(str::trim)
        {
            current = ACTIVE_SECTIONS.iter().copied().find(|s| *s == name);
            continue;
        }
        let Some(_section) = current else { continue };
        for tok in strip_comment_line(line).split_whitespace() {
            tokens.push(tok.to_string());
        }
    }
    tokens
}

fn strip_comment_line(line: &str) -> String {
    if let Some(start) = line.find("/*") {
        let after = &line[start + 2..];
        if let Some(end_rel) = after.find("*/") {
            let mut out = String::with_capacity(line.len());
            out.push_str(&line[..start]);
            out.push_str(&after[end_rel + 2..]);
            return out;
        }
    }
    line.to_string()
}
