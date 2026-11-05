//! Sequences this run did not design, scored as molecules of a given description.
//!
//! `derna check` normally takes a result file, whose header carries the molecule, and a
//! point whose energy and structure the run that wrote it already computed. A design
//! another tool emitted has neither. The question asked of such a design — what does the
//! constraint cost a sequence that was not made under it — needs the same two folds as any
//! other check, so what is missing is only the description of the molecule and the
//! sequence's own constrained fold.
//!
//! `--designs` supplies the first on the command line and computes the second here.
//! Every sequence is refused unless it *is* a molecule of that description: the flanks
//! as given, the coding window translating to the protein. That refusal is what replaces
//! the safety a result file's header provides, which is that a check cannot be pointed at
//! a different molecule than the one its points were designed over.

use std::path::Path;

use derna_codon_dp::cai::{compute_cai_tables, score_cai, standard_cai};
use derna_codon_dp::codon_table::{aa_to_index, get_codon_index, STOP, STOP_CODE};
use derna_lib::{sequence_constrained_fold_must, Params};
use serde_json::{json, Value};

use crate::molecule::Molecule;
use crate::Result;

/// One point per sequence in `path`, each folded under the molecule's constraint.
///
/// The points carry what a designed point carries — the sequence, its constrained
/// structure and energy, and its CAI — so everything downstream of here reads them the
/// way it reads a front's.
pub fn points(params: &Params, m: &Molecule, path: &Path) -> Result<Vec<Value>> {
    let sequences = read(path)?;
    if sequences.is_empty() {
        return Err(format!("--designs: {} holds no sequence", path.display()).into());
    }
    let protein = aa_indices(m)?;
    let log_cai = compute_cai_tables(&m.codons.usage).log_cai;
    let mut points = Vec::with_capacity(sequences.len());
    for (at, (name, sequence)) in sequences.into_iter().enumerate() {
        let where_ = Where {
            at,
            name: name.as_deref(),
            path,
        };
        let coding = coding_of(m, &sequence, where_)?;
        let rna = nucleotide_indices(&coding, where_)?;
        check_translation(&rna, &protein, where_)?;
        let (structure, mfe) =
            sequence_constrained_fold_must(params, &sequence, m.cds(), m.table(), &[])
                .map_err(|e| format!("{}: {e}", where_.what()))?
                .ok_or_else(|| {
                    format!(
                        "{}: the constraint allows this sequence no structure at all",
                        where_.what()
                    )
                })?;
        let cai_log = score_cai(&rna, &protein, &log_cai).map_err(|e| e.to_string())?;
        let cai = standard_cai(&rna, &protein, &log_cai).map_err(|e| e.to_string())?;
        let mut point = json!({
            "record": "point",
            "sequence": sequence,
            "structure": structure,
            "mfe": mfe,
            "cai": cai,
            "cai_log": cai_log,
        });
        if let Some(name) = name {
            point["name"] = json!(name);
        }
        points.push(point);
    }
    Ok(points)
}

/// Which sequence a refusal is about: a FASTA name where the file gives one, and the
/// position either way, so the message names something the caller can find.
#[derive(Clone, Copy)]
struct Where<'a> {
    at: usize,
    name: Option<&'a str>,
    path: &'a Path,
}

impl Where<'_> {
    fn what(&self) -> String {
        match self.name {
            Some(name) => format!("--designs: {name} (number {})", self.at + 1),
            None => format!("--designs: {} number {}", self.path.display(), self.at + 1),
        }
    }
}

/// The sequences of a FASTA file, or of a file of one sequence per line.
///
/// `T` reads as `U`, as it does for every sequence this command line takes.
fn read(path: &Path) -> Result<Vec<(Option<String>, String)>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("--designs: cannot read {}: {e}", path.display()))?;
    let mut out: Vec<(Option<String>, String)> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(header) = line.strip_prefix('>') {
            out.push((Some(header.trim().to_string()), String::new()));
            continue;
        }
        let piece = line.to_ascii_uppercase().replace('T', "U");
        match out.last_mut() {
            // A FASTA record's sequence may be wrapped over several lines.
            Some((Some(_), sequence)) => sequence.push_str(&piece),
            _ => out.push((None, piece)),
        }
    }
    Ok(out)
}

/// The coding window of `sequence`, with the molecule's flanks checked off either end.
fn coding_of(m: &Molecule, sequence: &str, where_: Where) -> Result<String> {
    let flank = |f: Option<&crate::input::NamedSequence>| {
        f.map(|u| u.sequence.as_str())
            .unwrap_or_default()
            .to_string()
    };
    let utr5 = flank(m.utr5.as_ref());
    let utr3 = flank(m.utr3.as_ref());
    let given_stop = m.stop.given_nucleotides().to_string();
    let coding_len = 3 * designed_codons(m);
    let want = utr5.len() + coding_len + given_stop.len() + utr3.len();
    if sequence.len() != want {
        return Err(format!(
            "{}: {} nucleotides, and this molecule is {want} \
             ({} 5′ + {coding_len} coding + {} stop + {} 3′)",
            where_.what(),
            sequence.len(),
            utr5.len(),
            given_stop.len(),
            utr3.len()
        )
        .into());
    }
    if !sequence.starts_with(&utr5) {
        return Err(format!("{}: does not start with the 5′ UTR", where_.what()).into());
    }
    if !sequence.ends_with(&utr3) {
        return Err(format!("{}: does not end with the 3′ UTR", where_.what()).into());
    }
    let coding = &sequence[utr5.len()..utr5.len() + coding_len + given_stop.len()];
    if !given_stop.is_empty() && !coding.ends_with(&given_stop) {
        return Err(format!(
            "{}: its coding sequence does not end in the given stop codon {given_stop}",
            where_.what()
        )
        .into());
    }
    Ok(coding[..coding_len].to_string())
}

/// How many codons the molecule's design spells: one per residue, and one more where the
/// stop codon is the search's to choose.
fn designed_codons(m: &Molecule) -> usize {
    m.designed_protein().chars().count()
}

/// `A=0, C=1, G=2, U=3`, the indices the CAI tables and the lattice are written over.
fn nucleotide_indices(coding: &str, where_: Where) -> Result<Vec<u8>> {
    coding
        .chars()
        .map(|c| match c {
            'A' => Ok(0),
            'C' => Ok(1),
            'G' => Ok(2),
            'U' => Ok(3),
            other => Err(format!("{}: {other:?} is not a nucleotide", where_.what()).into()),
        })
        .collect()
}

/// The molecule's protein as amino-acid indices, a designed stop codon included.
///
/// A designed stop is a slot of the coding sequence like any other, spelled `*` by
/// [`Molecule::designed_protein`] and indexed [`STOP`]: which of the three stop codons
/// the sequence ends in is the sequence's to say, and all three code for that slot.
fn aa_indices(m: &Molecule) -> Result<Vec<usize>> {
    m.designed_protein()
        .chars()
        .map(|aa| {
            if aa == STOP_CODE {
                return Ok(STOP);
            }
            aa_to_index(aa).map_err(|e| -> Box<dyn std::error::Error> {
                format!("--protein: {aa:?} is not an amino acid: {e}").into()
            })
        })
        .collect()
}

/// Refuse a sequence whose codons do not spell the molecule's protein.
///
/// This is the check that makes scoring a given sequence safe: a design of another
/// protein, or of the same protein read out of frame, is refused at the first codon
/// that does not code for the residue the molecule names.
fn check_translation(rna: &[u8], protein: &[usize], where_: Where) -> Result<()> {
    for (i, &aa) in protein.iter().enumerate() {
        let codon = [rna[3 * i], rna[3 * i + 1], rna[3 * i + 2]];
        if get_codon_index(aa, codon).is_err() {
            let spelled: String = codon
                .iter()
                .map(|&n| ['A', 'C', 'G', 'U'][n as usize])
                .collect();
            let residue = if aa == STOP {
                "a stop codon".to_string()
            } else {
                derna_codon_dp::codon_table::index_to_aa(aa)
                    .map(|c| c.to_string())
                    .unwrap_or_else(|_| format!("index {aa}"))
            };
            return Err(format!(
                "{}: codon {} is {spelled}, which does not code for {residue}",
                where_.what(),
                i + 1
            )
            .into());
        }
    }
    Ok(())
}
