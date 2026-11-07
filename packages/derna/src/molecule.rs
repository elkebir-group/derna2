//! The molecule a run designs: the protein, its fixed flanks, the stop codon,
//! the constraint and the codon usage table, resolved from the shared flags.

use derna_lib::{CdsWindow, Layout, Params, RegionConstraints, UtrSpec};
use serde_json::{json, Value};

use crate::args::Shared;
use crate::constraint::{self, Constraint};
use crate::input::{self, CodonTable, NamedSequence};
use crate::Result;

/// Everything a design, a front or a check is over, past the weights.
#[derive(Debug, Clone)]
pub struct Molecule {
    pub protein: NamedSequence,
    pub utr5: Option<NamedSequence>,
    pub utr3: Option<NamedSequence>,
    /// What follows the last residue: a given stop codon, a designed one, or
    /// nothing.
    pub stop: Stop,
    pub constraint: Constraint,
    pub codons: CodonTable,
    /// The energy parameters every fold of the run uses: the built-in table,
    /// or the file `--params` names.
    pub params: Params,
}

/// The stop codon after the last residue.
///
/// A stop codon can only matter through a pair the constraint forbids, and a
/// forbidden pair needs a flank — so with flanks given and no `--stop` the
/// codon is left to the search ([`Self::Free`]), and without flanks it is
/// absent as before. `--stop UAA|UAG|UGA` gives it, and `--stop none` asks for
/// none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stop {
    /// No stop codon: the molecule ends at the last residue's codon.
    Absent,
    /// This codon, as given sequence at the head of the 3′ flank. It carries no
    /// synonymous choice and no adaptation, and the constraint counts it as
    /// coding sequence.
    Given(String),
    /// A designed stop codon: the coding sequence's last slot, whose codon the
    /// search chooses among `UAA`, `UAG` and `UGA` alongside every residue's.
    /// Every stop codon weighs the same, so the slot moves no design's
    /// adaptation and the fronts of the given and designed runs share one
    /// adaptation scale.
    Free,
}

impl Stop {
    /// Whether the search chooses the codon.
    pub fn is_free(&self) -> bool {
        matches!(self, Self::Free)
    }

    /// Whether there is no stop codon at all.
    pub fn is_absent(&self) -> bool {
        matches!(self, Self::Absent)
    }

    /// The given codon's nucleotides, empty when the stop is designed or
    /// absent. This is 3′-flank sequence, so a designed stop contributes none:
    /// its nucleotides are inside the coding sequence.
    pub(crate) fn given_nucleotides(&self) -> &str {
        match self {
            Self::Given(codon) => codon,
            Self::Absent | Self::Free => "",
        }
    }

    /// How the header records it: the codon, `"free"`, or nothing.
    fn field(&self) -> Value {
        match self {
            Self::Absent => Value::Null,
            Self::Given(codon) => json!(codon),
            Self::Free => json!("free"),
        }
    }
}

impl Molecule {
    /// This molecule with the stop codon given as `codon`.
    pub fn with_given_stop(&self, codon: &str) -> Self {
        Self {
            stop: Stop::Given(codon.to_string()),
            ..self.clone()
        }
    }

    /// Read and check the shared flags' inputs. `--protein` is required.
    pub fn resolve(shared: &Shared) -> Result<Self> {
        let protein = shared.protein.as_deref().ok_or("--protein is required")?;
        input::refuse_shared_stdin(&[
            ("--protein", Some(protein)),
            ("--utr5", shared.utr5.as_deref()),
            ("--utr3", shared.utr3.as_deref()),
        ])?;
        // A run with arms designs the stop as `free` does; the front command
        // then reads each arm's front from the one lattice ([`stop_arms`]).
        let free = shared
            .stop
            .as_deref()
            .is_some_and(|s| s.eq_ignore_ascii_case("free"))
            || !stop_arms(shared.stop.as_deref())?.is_empty();
        let given = if free {
            None
        } else {
            parse_stop(shared.stop.as_deref())?
        };
        let protein = match &shared.protein_record {
            Some(record) => input::checked_protein(record)?,
            None => input::read_protein(protein)?,
        };
        let utr5 = shared
            .utr5
            .as_deref()
            .map(|v| input::read_utr("--utr5", v))
            .transpose()?;
        let utr3 = shared
            .utr3
            .as_deref()
            .map(|v| input::read_utr("--utr3", v))
            .transpose()?;
        let flanked = utr5.is_some() || utr3.is_some();
        if free && !flanked {
            return Err(
                "--stop free needs a flank: a stop codon matters only through a pair \
                        the constraint forbids, and on a bare coding sequence there is none"
                    .into(),
            );
        }
        let stop = if free {
            Stop::Free
        } else {
            resolve_stop(given, shared.stop.is_some(), flanked)
        };
        let constraint = constraint::resolve(
            shared.forbid.as_deref(),
            shared.access.as_deref(),
            utr5.is_some(),
            utr3.is_some(),
        )?;
        let codons = input::read_codon_usage(shared.codon_usage.as_deref())?;
        Ok(Self {
            protein,
            utr5,
            utr3,
            stop,
            constraint,
            codons,
            params: shared.energy_params()?,
        })
    }

    /// Rebuild the molecule a result file was produced over, from its header.
    ///
    /// This is how a command that takes a result rather than flags — `check`
    /// — learns what to work on, and why it cannot be pointed at other flanks
    /// than the result was made with. The codon table comes from the header's
    /// own embedded rows, so CAI is scored against exactly the table the run
    /// used rather than against whatever `human` means here. The energy
    /// parameters are the built-in table; a header does not record them.
    pub fn from_header(header: &serde_json::Value) -> Result<Self> {
        let sequence = |field: &str| -> Result<Option<NamedSequence>> {
            match header.get(field) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(value) => Ok(Some(NamedSequence {
                    name: value["name"].as_str().map(str::to_string),
                    sequence: value["sequence"]
                        .as_str()
                        .ok_or_else(|| format!("the header's {field} has no sequence"))?
                        .to_string(),
                })),
            }
        };
        let protein = sequence("protein")?.ok_or("the header names no protein")?;
        let constraint = constraint_from_header(
            header,
            sequence("utr5")?.is_some(),
            sequence("utr3")?.is_some(),
        )?;
        let codons = input::codon_table_from_csv(
            header["codon_usage"]["source"]
                .as_str()
                .unwrap_or("the header's table")
                .to_string(),
            &codon_usage_csv(&header["codon_usage"]["table"])?,
        )?;
        // A header records exactly what the run was over, so it is read back
        // literally: an absent or null field is a run with no stop codon, and
        // not a run that would free one now because it has flanks.
        let stop = match header.get("stop") {
            None | Some(serde_json::Value::Null) => Stop::Absent,
            Some(value) if value.as_str() == Some("free") => Stop::Free,
            Some(value) => match parse_stop(value.as_str())? {
                Some(codon) => Stop::Given(codon),
                None => Stop::Absent,
            },
        };
        Ok(Self {
            protein,
            utr5: sequence("utr5")?,
            utr3: sequence("utr3")?,
            stop,
            constraint,
            codons,
            params: Params::load_default()?,
        })
    }

    /// The 5′ flank as the library takes it; empty when absent.
    pub fn utr5_spec(&self) -> UtrSpec {
        flank_spec(self.utr5.as_ref())
    }

    /// The 3′ flank as the library takes it, headed by a **given** stop codon
    /// when there is one; empty when both are absent.
    ///
    /// A given stop is fixed sequence, so it carries no synonymous choice and
    /// stays out of CAI; [`Self::table`] is what makes the constraint count it
    /// as coding sequence. A designed stop is not here at all — it is the last
    /// slot of [`Self::designed_protein`]'s lattice, inside the coding
    /// sequence.
    pub fn utr3_spec(&self) -> UtrSpec {
        let utr3 = self.utr3.as_ref().map_or("", |u| u.sequence.as_str());
        UtrSpec::Fixed(format!("{}{utr3}", self.stop.given_nucleotides()))
    }

    /// The codon slots a design chooses, as the library reads them: the
    /// protein's residues, and a trailing `*` for a designed stop codon.
    ///
    /// This is what every entry point that builds a lattice is given, in place
    /// of the protein alone. `derna_lib::residues` reads the `*` as the stop
    /// family, so the slot is one more codon position of the same lattice and
    /// the search chooses its codon as it chooses a residue's.
    pub fn designed_protein(&self) -> String {
        match self.stop {
            Stop::Free => format!("{}*", self.protein.sequence),
            Stop::Absent | Stop::Given(_) => self.protein.sequence.clone(),
        }
    }

    /// The constraint as the library takes it, with a given stop codon
    /// classified as coding sequence.
    ///
    /// The suffix is what extends the classified coding sequence past the
    /// designed codons, so it counts a **given** stop's three nucleotides and
    /// not a designed one's: a designed stop is already inside the window
    /// [`Self::cds`] reports. Either way the stop occupies three coding
    /// nucleotides at the same positions, so the two spellings classify every
    /// pair of a molecule alike.
    pub fn table(&self) -> RegionConstraints {
        self.constraint
            .table()
            .with_cds_suffix(self.stop.given_nucleotides().len() as u32)
    }

    /// The molecule's regions, which every coordinate resolves against. A given
    /// stop codon sits past the designed codons, at the head of the 3′ flank; a
    /// designed one is the last designed codon.
    pub fn layout(&self) -> Layout {
        let flank = |f: Option<&NamedSequence>| f.map_or(0, |u| u.sequence.len() as u32);
        Layout::new(
            flank(self.utr5.as_ref()),
            self.codon_slots() as u32,
            self.stop.given_nucleotides().len() as u32 + flank(self.utr3.as_ref()),
        )
    }

    /// The designed codons' window in the whole molecule. A given stop codon is
    /// not in it; a designed one is its last codon.
    pub fn cds(&self) -> CdsWindow {
        let utr5 = self.utr5.as_ref().map_or(0, |u| u.sequence.len());
        CdsWindow::new(utr5, self.codon_slots())
    }

    /// How the header records the stop: the given codon, `"free"` when the
    /// search chooses it, or nothing.
    pub fn stop_field(&self) -> Value {
        self.stop.field()
    }

    /// How many codons the design spells: one per residue, and one more for a
    /// designed stop codon.
    fn codon_slots(&self) -> usize {
        self.protein.sequence.len() + usize::from(self.stop.is_free())
    }

    /// The stop codon `sequence` — a whole molecule of this one — ends its
    /// coding sequence in, when the stop is the search's to choose; `None` when
    /// it is given or absent, since then every design of the run shares the
    /// header's.
    ///
    /// Read off the coding window rather than the sequence's end, so a molecule
    /// with a 3′ flank behind the stop reports the same thing as one without.
    pub fn stop_of(&self, sequence: &str) -> Option<String> {
        if !self.stop.is_free() {
            return None;
        }
        let cds = self.cds();
        sequence.get(cds.end - 3..cds.end).map(str::to_string)
    }
}

/// The constraint a header records, read back through the same resolver the
/// flags go through, so the two cannot come to disagree about what a class
/// list means.
fn constraint_from_header(
    header: &serde_json::Value,
    has_utr5: bool,
    has_utr3: bool,
) -> Result<Constraint> {
    let forbid: Vec<&str> = header["forbid"]
        .as_array()
        .map(|names| names.iter().filter_map(serde_json::Value::as_str).collect())
        .unwrap_or_default();
    let access = header["access"]
        .as_array()
        .and_then(|w| match w.as_slice() {
            [up, down] => Some(format!("{up},{down}")),
            _ => None,
        });
    // An empty list is the constraint turned off, which is `--forbid none`'s
    // spelling; the resolver takes one vocabulary.
    let forbidden = forbid.join(",");
    constraint::resolve(
        Some(if forbidden.is_empty() {
            "none"
        } else {
            &forbidden
        }),
        access.as_deref(),
        has_utr5,
        has_utr3,
    )
}

/// The header's embedded table, back in the CSV the loader reads.
///
/// Going through the text rather than reconstructing the parsed table keeps
/// one reader for the format: a table written by this command and one given
/// by a caller are loaded by the same code.
fn codon_usage_csv(table: &serde_json::Value) -> Result<String> {
    let rows = table
        .as_array()
        .ok_or("the header's codon usage table is not a list of rows")?;
    let mut csv = String::from("amino_acid,codon,fraction\n");
    for row in rows {
        match row.as_array().map(|r| r.as_slice()) {
            Some([aa, codon, fraction]) => csv.push_str(&format!(
                "{},{},{}\n",
                aa.as_str().unwrap_or_default(),
                codon.as_str().unwrap_or_default(),
                fraction
            )),
            _ => return Err(format!("the header has a malformed codon usage row: {row}").into()),
        }
    }
    Ok(csv)
}

fn flank_spec(flank: Option<&NamedSequence>) -> UtrSpec {
    UtrSpec::Fixed(flank.map_or_else(String::new, |f| f.sequence.clone()))
}

/// What follows the last residue, from the flag and whether the molecule has a
/// flank.
///
/// `--stop UAA|UAG|UGA` gives the codon and `--stop none` asks for none, at any
/// flank; `--stop free` is read before this and spells out what omitting the
/// flag means at a flank. Omitting the flag frees the codon when there is a flank, and leaves
/// the molecule without one when there is not: the stop can only matter through
/// a pair the constraint forbids, and a forbidden pair needs a flank, so on a
/// bare coding sequence a designed stop would add a slot that changes no
/// answer.
fn resolve_stop(given: Option<String>, flag_given: bool, flanked: bool) -> Stop {
    match given {
        Some(codon) => Stop::Given(codon),
        // `--stop none` is a request for no stop codon, not the default being
        // taken: it says so at any flank, where omitting the flag frees the
        // codon at one and leaves it absent without one.
        None if flanked && !flag_given => Stop::Free,
        None => Stop::Absent,
    }
}

/// The three stop codons, in the order a stop-labelled lattice lanes them.
pub const STOP_CODONS: [&str; 3] = ["UAA", "UAG", "UGA"];

/// The stop codons `--stop` asks for a front of each.
///
/// `each` names all three and a comma list names those written, in the order
/// written. Everything else — one codon, `free`, `none`, the flag absent — is
/// a single front, and the answer is empty.
///
/// Two or more stop codons are walked over one lattice that keeps them apart
/// rather than minimized over, because a front over a designed stop codon
/// minimizes over the stops and loses a design that is best under one of them
/// to a design under another.
pub fn stop_arms(stop: Option<&str>) -> Result<Vec<String>> {
    let Some(spec) = stop else {
        return Ok(Vec::new());
    };
    if spec.eq_ignore_ascii_case("each") {
        return Ok(STOP_CODONS.iter().map(|c| (*c).to_string()).collect());
    }
    if !spec.contains(',') {
        return Ok(Vec::new());
    }
    let mut arms: Vec<String> = Vec::new();
    for word in spec.split(',') {
        let codon = word.trim().to_ascii_uppercase().replace('T', "U");
        if !STOP_CODONS.contains(&codon.as_str()) {
            return Err(format!(
                "--stop: a list names stop codons, and {:?} is not one of UAA, UAG, UGA",
                word.trim()
            )
            .into());
        }
        if arms.contains(&codon) {
            return Err(format!("--stop names {codon} twice; each stop gets one front").into());
        }
        arms.push(codon);
    }
    Ok(arms)
}

/// `--stop`: one of the three stop codons, `T` read as `U`, or `none`.
fn parse_stop(stop: Option<&str>) -> Result<Option<String>> {
    let codon = stop.map(|s| s.to_ascii_uppercase().replace('T', "U"));
    match codon.as_deref() {
        None | Some("NONE") => Ok(None),
        Some(c @ ("UAA" | "UAG" | "UGA")) => Ok(Some(c.to_string())),
        _ => Err(format!(
            "--stop expects UAA, UAG, UGA, a comma list of them, free, each or none, got {:?}",
            stop.unwrap_or_default()
        )
        .into()),
    }
}
