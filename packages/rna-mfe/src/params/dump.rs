//! Canonical dumper: [`Params`] → text that the parser round-trips.
//!
//! The output is a *subset* of upstream `rna_turner2004.par`: only the
//! sections this crate parses are emitted. Skipped sections
//! (enthalpies, dangles, mismatch_multi, mismatch_exterior) are
//! intentionally absent — feeding the dump back through the parser
//! produces the same `Params`, which is the load-bearing round-trip
//! invariant.
//!
//! The dump format is intentionally minimal — no row labels, no column
//! headers, only `# section_name` followed by whitespace-separated
//! integers — because the parser ignores comments and headers anyway,
//! and this keeps the dumper trivially correct. Comparing to upstream
//! requires tokenizing both sides; see `tests/round_trip.rs`.

use std::fmt::Write;

use super::special::SpecialHairpinTable;
use super::tables::{MiscParams, MultiLoopParams, NinioParams, SizeTable, INF};
use super::Params;

const VALUES_PER_LINE: usize = 10;

pub fn dump_canonical(params: &Params) -> String {
    let mut out = String::new();
    out.push_str("## RNAfold parameter file v2.0 (rna-mfe canonical dump)\n");

    write_int_section(&mut out, "stack", params.stack.flat.iter().copied());
    write_int_section(
        &mut out,
        "mismatch_hairpin",
        params.mismatch_hairpin.flat.iter().copied(),
    );
    write_int_section(
        &mut out,
        "mismatch_interior",
        params.mismatch_interior.flat.iter().copied(),
    );
    write_int_section(
        &mut out,
        "mismatch_interior_1n",
        params.mismatch_interior_1n.flat.iter().copied(),
    );
    write_int_section(
        &mut out,
        "mismatch_interior_23",
        params.mismatch_interior_23.flat.iter().copied(),
    );
    write_int_section(&mut out, "int11", params.int11.flat.iter().copied());
    write_int_section(&mut out, "int21", params.int21.flat.iter().copied());
    write_int_section(&mut out, "int22", params.int22.flat.iter().copied());
    write_int_section(&mut out, "hairpin", size_iter(&params.hairpin));
    write_int_section(&mut out, "bulge", size_iter(&params.bulge));
    write_int_section(&mut out, "interior", size_iter(&params.interior));
    write_int_section(&mut out, "ML_params", ml_iter(&params.ml));
    write_int_section(&mut out, "NINIO", ninio_iter(&params.ninio));
    write_int_section(&mut out, "Misc", misc_iter(&params.misc));
    write_special_section(&mut out, "Hexaloops", &params.hexaloops);
    write_special_section(&mut out, "Tetraloops", &params.tetraloops);
    write_special_section(&mut out, "Triloops", &params.triloops);

    out.push_str("\n#END\n");
    out
}

fn write_int_section<I>(out: &mut String, name: &str, values: I)
where
    I: IntoIterator<Item = i32>,
{
    write_section_header(out, name);
    let mut col = 0;
    for v in values {
        dump_int_value(out, v);
        col += 1;
        if col == VALUES_PER_LINE {
            out.push('\n');
            col = 0;
        } else {
            out.push(' ');
        }
    }
    if col != 0 {
        out.push('\n');
    }
}

fn write_special_section(out: &mut String, name: &str, table: &SpecialHairpinTable) {
    write_section_header(out, name);
    for entry in &table.entries {
        out.push_str(&entry.seq);
        out.push(' ');
        write!(out, "{}", entry.bonus_37).unwrap();
        out.push(' ');
        write!(out, "{}", entry.bonus_dh).unwrap();
        out.push('\n');
    }
}

fn write_section_header(out: &mut String, name: &str) {
    out.push_str("\n# ");
    out.push_str(name);
    out.push('\n');
}

fn dump_int_value(out: &mut String, v: i32) {
    if v == INF {
        out.push_str("INF");
    } else if v == -INF {
        out.push_str("-INF");
    } else {
        write!(out, "{}", v).unwrap();
    }
}

fn size_iter(t: &SizeTable) -> impl Iterator<Item = i32> + '_ {
    t.values.iter().copied()
}

fn ml_iter(ml: &MultiLoopParams) -> impl Iterator<Item = i32> {
    [
        ml.base_37,
        ml.base_dh,
        ml.closing_37,
        ml.closing_dh,
        ml.intern_37,
        ml.intern_dh,
    ]
    .into_iter()
}

fn ninio_iter(n: &NinioParams) -> impl Iterator<Item = i32> {
    [n.ninio_37, n.ninio_dh, n.max_ninio].into_iter()
}

fn misc_iter(m: &MiscParams) -> impl Iterator<Item = i32> {
    [
        m.duplex_init_37,
        m.duplex_init_dh,
        m.terminal_au_37,
        m.terminal_au_dh,
    ]
    .into_iter()
}
