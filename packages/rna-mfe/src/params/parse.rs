//! `rna_turner2004.par` → [`Params`] section parsers.
//!
//! Each section's expected token count is wired in from ViennaRNA's
//! `src/ViennaRNA/params/io.c` `case <SECTION>:` branches. Skipped
//! sections (enthalpies, dangles, mismatch_multi, mismatch_exterior)
//! are walked-and-discarded so the section order check still works.

use thiserror::Error;

use super::lex::{body_tokens, split_sections, Section};
use super::special::{SpecialHairpinEntry, SpecialHairpinTable};
use super::tables::{
    Int11Table, Int21Table, Int22Table, MiscParams, MultiLoopParams, NinioParams, SizeTable,
    Stack37, Triplet37, INF, INT11_LEN, INT21_LEN, INT22_LEN, N_NUCS, N_PAIRS, SIZE_TABLE_LEN,
    STACK_LEN, TRIPLET_LEN,
};
use super::Params;

/// Parsing failure modes. Round-tripping through [`super::Params::dump_canonical`]
/// produces input the parser accepts, so these error paths are
/// exercised only by malformed external inputs.
#[derive(Debug, Error)]
pub enum ParseError {
    #[error("section `{name}` has {found} tokens, expected {expected}")]
    SectionLength {
        name: String,
        found: usize,
        expected: usize,
    },
    #[error("section `{name}` has unparseable token `{token}`")]
    BadToken { name: String, token: String },
    #[error("missing required section `{0}`")]
    MissingSection(&'static str),
}

/// Per-section expected token count for the skipped sections that
/// share their shape with an active section's `_enthalpies` companion.
const DANGLE_LEN: usize = N_PAIRS * N_NUCS;

/// Top-level entry. Walks every section in source order, dispatches
/// known sections to their parsers, and discards skipped sections
/// after asserting their length matches Vienna's layout.
pub fn parse_params(source: &str) -> Result<Params, ParseError> {
    let sections = split_sections(source);
    let mut builder = Builder::default();
    for section in &sections {
        builder.dispatch(section)?;
    }
    builder.finish()
}

#[derive(Default)]
struct Builder {
    stack: Option<Stack37>,
    mismatch_hairpin: Option<Triplet37>,
    mismatch_interior: Option<Triplet37>,
    mismatch_interior_1n: Option<Triplet37>,
    mismatch_interior_23: Option<Triplet37>,
    int11: Option<Int11Table>,
    int21: Option<Int21Table>,
    int22: Option<Int22Table>,
    hairpin: Option<SizeTable>,
    bulge: Option<SizeTable>,
    interior: Option<SizeTable>,
    ml: Option<MultiLoopParams>,
    ninio: Option<NinioParams>,
    misc: Option<MiscParams>,
    triloops: Option<SpecialHairpinTable>,
    tetraloops: Option<SpecialHairpinTable>,
    hexaloops: Option<SpecialHairpinTable>,
}

impl Builder {
    fn dispatch(&mut self, section: &Section<'_>) -> Result<(), ParseError> {
        match section.name {
            "stack" => self.stack = Some(parse_stack(section)?),
            "mismatch_hairpin" => self.mismatch_hairpin = Some(parse_triplet(section)?),
            "mismatch_interior" => self.mismatch_interior = Some(parse_triplet(section)?),
            "mismatch_interior_1n" => self.mismatch_interior_1n = Some(parse_triplet(section)?),
            "mismatch_interior_23" => self.mismatch_interior_23 = Some(parse_triplet(section)?),
            "int11" => self.int11 = Some(parse_int11(section)?),
            "int21" => self.int21 = Some(parse_int21(section)?),
            "int22" => self.int22 = Some(parse_int22(section)?),
            "hairpin" => self.hairpin = Some(parse_size_table(section)?),
            "bulge" => self.bulge = Some(parse_size_table(section)?),
            "interior" => self.interior = Some(parse_size_table(section)?),
            "ML_params" => self.ml = Some(parse_ml_params(section)?),
            "NINIO" => self.ninio = Some(parse_ninio(section)?),
            "Misc" => self.misc = Some(parse_misc(section)?),
            "Triloops" => self.triloops = Some(parse_special(section)?),
            "Tetraloops" => self.tetraloops = Some(parse_special(section)?),
            "Hexaloops" => self.hexaloops = Some(parse_special(section)?),
            other => skip_unparsed(section, other)?,
        }
        Ok(())
    }

    fn finish(self) -> Result<Params, ParseError> {
        Ok(Params {
            stack: self.stack.ok_or(ParseError::MissingSection("stack"))?,
            mismatch_hairpin: self
                .mismatch_hairpin
                .ok_or(ParseError::MissingSection("mismatch_hairpin"))?,
            mismatch_interior: self
                .mismatch_interior
                .ok_or(ParseError::MissingSection("mismatch_interior"))?,
            mismatch_interior_1n: self
                .mismatch_interior_1n
                .ok_or(ParseError::MissingSection("mismatch_interior_1n"))?,
            mismatch_interior_23: self
                .mismatch_interior_23
                .ok_or(ParseError::MissingSection("mismatch_interior_23"))?,
            int11: self.int11.ok_or(ParseError::MissingSection("int11"))?,
            int21: self.int21.ok_or(ParseError::MissingSection("int21"))?,
            int22: self.int22.ok_or(ParseError::MissingSection("int22"))?,
            hairpin: self.hairpin.ok_or(ParseError::MissingSection("hairpin"))?,
            bulge: self.bulge.ok_or(ParseError::MissingSection("bulge"))?,
            interior: self
                .interior
                .ok_or(ParseError::MissingSection("interior"))?,
            ml: self.ml.ok_or(ParseError::MissingSection("ML_params"))?,
            ninio: self.ninio.ok_or(ParseError::MissingSection("NINIO"))?,
            misc: self.misc.ok_or(ParseError::MissingSection("Misc"))?,
            triloops: self
                .triloops
                .ok_or(ParseError::MissingSection("Triloops"))?,
            tetraloops: self
                .tetraloops
                .ok_or(ParseError::MissingSection("Tetraloops"))?,
            hexaloops: self
                .hexaloops
                .ok_or(ParseError::MissingSection("Hexaloops"))?,
            derived: super::Derived::default(),
        })
    }
}

/// Skipped sections: the parser runs at
/// 37°C and dangles=0 only, so enthalpies / dangles / mismatch_multi /
/// mismatch_exterior are read-and-discarded with their expected
/// length validated. An unrecognised section header is a parser bug
/// (or a Vienna upgrade introduced new sections).
fn skip_unparsed(section: &Section<'_>, name: &str) -> Result<(), ParseError> {
    let expected = match name {
        "stack_enthalpies" => STACK_LEN,
        "mismatch_hairpin_enthalpies"
        | "mismatch_interior_enthalpies"
        | "mismatch_interior_1n_enthalpies"
        | "mismatch_interior_23_enthalpies"
        | "mismatch_multi"
        | "mismatch_multi_enthalpies"
        | "mismatch_exterior"
        | "mismatch_exterior_enthalpies" => TRIPLET_LEN,
        "dangle5" | "dangle5_enthalpies" | "dangle3" | "dangle3_enthalpies" => DANGLE_LEN,
        "int11_enthalpies" => INT11_LEN,
        "int21_enthalpies" => INT21_LEN,
        "int22_enthalpies" => INT22_LEN,
        "hairpin_enthalpies" | "bulge_enthalpies" | "interior_enthalpies" => SIZE_TABLE_LEN,
        other => {
            return Err(ParseError::BadToken {
                name: "<top-level>".into(),
                token: other.into(),
            })
        }
    };
    collect_ints(section, expected)?;
    Ok(())
}

fn collect_ints(section: &Section<'_>, expected: usize) -> Result<Vec<i32>, ParseError> {
    let mut out = Vec::with_capacity(expected);
    for token in body_tokens(section) {
        out.push(parse_int_token(section.name, token)?);
    }
    if out.len() != expected {
        return Err(ParseError::SectionLength {
            name: section.name.into(),
            found: out.len(),
            expected,
        });
    }
    Ok(out)
}

fn parse_int_token(section: &str, token: &str) -> Result<i32, ParseError> {
    match token {
        "INF" => Ok(INF),
        "-INF" | "-INf" => Ok(-INF),
        "DEF" | "NST" => Ok(0),
        _ => token.parse::<i32>().map_err(|_| ParseError::BadToken {
            name: section.into(),
            token: token.into(),
        }),
    }
}

/// Read `N` tokens into a fixed-size array. Centralises the
/// flat-storage population path so per-section parsers don't repeat
/// the boilerplate.
fn read_array<const N: usize>(section: &Section<'_>) -> Result<[i32; N], ParseError> {
    let flat = collect_ints(section, N)?;
    let mut arr = [0_i32; N];
    arr.copy_from_slice(&flat);
    Ok(arr)
}

fn read_boxed_array<const N: usize>(section: &Section<'_>) -> Result<Box<[i32; N]>, ParseError> {
    let flat = collect_ints(section, N)?;
    let mut arr = Box::new([0_i32; N]);
    arr.copy_from_slice(&flat);
    Ok(arr)
}

fn parse_stack(section: &Section<'_>) -> Result<Stack37, ParseError> {
    Ok(Stack37 {
        flat: read_array::<STACK_LEN>(section)?,
    })
}

fn parse_triplet(section: &Section<'_>) -> Result<Triplet37, ParseError> {
    Ok(Triplet37 {
        flat: read_array::<TRIPLET_LEN>(section)?,
    })
}

fn parse_int11(section: &Section<'_>) -> Result<Int11Table, ParseError> {
    Ok(Int11Table {
        flat: read_boxed_array::<INT11_LEN>(section)?,
    })
}

fn parse_int21(section: &Section<'_>) -> Result<Int21Table, ParseError> {
    Ok(Int21Table {
        flat: read_boxed_array::<INT21_LEN>(section)?,
    })
}

fn parse_int22(section: &Section<'_>) -> Result<Int22Table, ParseError> {
    Ok(Int22Table {
        flat: read_boxed_array::<INT22_LEN>(section)?,
    })
}

fn parse_size_table(section: &Section<'_>) -> Result<SizeTable, ParseError> {
    Ok(SizeTable {
        values: read_array::<SIZE_TABLE_LEN>(section)?,
    })
}

fn parse_ml_params(section: &Section<'_>) -> Result<MultiLoopParams, ParseError> {
    let v = collect_ints(section, 6)?;
    Ok(MultiLoopParams {
        base_37: v[0],
        base_dh: v[1],
        closing_37: v[2],
        closing_dh: v[3],
        intern_37: v[4],
        intern_dh: v[5],
    })
}

fn parse_ninio(section: &Section<'_>) -> Result<NinioParams, ParseError> {
    let v = collect_ints(section, 3)?;
    Ok(NinioParams {
        ninio_37: v[0],
        ninio_dh: v[1],
        max_ninio: v[2],
    })
}

fn parse_misc(section: &Section<'_>) -> Result<MiscParams, ParseError> {
    let v = collect_ints(section, 4)?;
    Ok(MiscParams {
        duplex_init_37: v[0],
        duplex_init_dh: v[1],
        terminal_au_37: v[2],
        terminal_au_dh: v[3],
    })
}

fn parse_special(section: &Section<'_>) -> Result<SpecialHairpinTable, ParseError> {
    let mut entries = Vec::new();
    for line in &section.lines {
        let mut tokens = line.split_whitespace();
        let Some(seq) = tokens.next() else { continue };
        let bonus_37 = parse_required_int(section.name, &mut tokens)?;
        let bonus_dh = parse_required_int(section.name, &mut tokens)?;
        if tokens.next().is_some() {
            return Err(ParseError::BadToken {
                name: section.name.into(),
                token: format!("trailing tokens on `{}`", seq),
            });
        }
        entries.push(SpecialHairpinEntry {
            seq: seq.to_string(),
            bonus_37,
            bonus_dh,
        });
    }
    Ok(SpecialHairpinTable { entries })
}

fn parse_required_int<'a>(
    section: &str,
    tokens: &mut impl Iterator<Item = &'a str>,
) -> Result<i32, ParseError> {
    let token = tokens.next().ok_or_else(|| ParseError::BadToken {
        name: section.into(),
        token: "<missing>".into(),
    })?;
    parse_int_token(section, token)
}
