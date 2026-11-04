//! `derna pool`: several certified fronts of one molecule joined into the two
//! fronts their union defines.
//!
//! A front over a designed stop codon minimizes over the three stop codons, so
//! a design that is best under one stop can be dominated by a design under
//! another and leave the front. A front per stop keeps them apart. Joining
//! those fronts again gives two different objects, and neither contains the
//! other:
//!
//! - the **masking front**, the non-dominated points of the pooled set. Over a
//!   molecule's three stops this is the front a designed stop walks, because a
//!   point non-dominated in the union is non-dominated in its own stop's front
//!   too, so the union of the fronts already holds it.
//! - the **modular front**, the non-dominated points among those whose own
//!   free fold respects the constraint. A modular design dominated by a design
//!   that violates the constraint is absent from the masking front and present
//!   here; a point of the masking front with no modular design at its energy
//!   and adaptation is present there and absent here.
//!
//! The order of the two steps is the whole point. A modular design belongs on
//! the modular front because nothing *modular* beats it, so the certification
//! has to come before the filter. Filtering first and reading the survivors'
//! verdicts answers a different question, and on a molecule whose modular
//! designs are all dominated it returns nothing at all. That is why this
//! command refuses a front it cannot see a verdict on rather than filtering it
//! and reporting what survived.

use crate::{result, show, Result};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

/// Two points of the same objectives are mutually non-dominated, so the
/// comparison needs a tolerance rather than float equality: a front's energies
/// are written in decimal and read back by a parser that need not round-trip
/// them to the same bits.
const EPS: f64 = 1e-9;

/// What `derna pool` was asked to do.
#[derive(Debug, Clone)]
pub struct PoolArgs {
    /// The fronts to join, in the order given.
    pub inputs: Vec<PathBuf>,
    /// The stem the two written fronts are named from.
    pub out: PathBuf,
    /// Overwrite either output if it exists.
    pub force: bool,
    /// Do not print the tables of what was written.
    pub quiet: bool,
}

/// `derna pool --help`.
pub fn help() -> String {
    "\
derna pool — join several certified fronts of one molecule

Usage: derna pool FRONT.jsonl... --out PATH [options]

Writes two fronts, named from PATH's stem:

  PATH.union.jsonl    the non-dominated points of every front pooled
  PATH.modular.jsonl  the non-dominated points among the modular ones

Each point keeps the stop codon of the front it came from. Every input must
have been certified by `derna check`, because a modular front is the
non-dominated subset of the designs that pass their check and the verdict
cannot be recovered after the points are filtered.

  --out PATH                name both written fronts from this path
  --force                   overwrite an existing output
  --quiet                   do not print the tables of what was written to
                            stderr at the end of the run
"
    .to_string()
}

/// Parse `derna pool`'s words.
pub fn parse(words: impl IntoIterator<Item = String>) -> Result<PoolArgs> {
    let mut inputs = Vec::new();
    let mut out = None;
    let mut force = false;
    let mut quiet = false;
    let mut words = words.into_iter();
    while let Some(word) = words.next() {
        match word.as_str() {
            "--out" => {
                out = Some(PathBuf::from(
                    words.next().ok_or("derna pool: --out needs a path")?,
                ));
            }
            "--force" => force = true,
            "--quiet" => quiet = true,
            other if other.starts_with('-') => {
                return Err(format!("derna pool: unknown option {other}").into());
            }
            other => inputs.push(PathBuf::from(other)),
        }
    }
    if inputs.is_empty() {
        return Err("derna pool: name at least one front to join".into());
    }
    let out = out.ok_or("derna pool: writes two fronts and needs --out PATH")?;
    Ok(PoolArgs {
        inputs,
        out,
        force,
        quiet,
    })
}

/// `path` with `tag` before its extension, as `--stop each` names its files.
pub fn tagged_path(path: &Path, tag: &str) -> PathBuf {
    let stem = path
        .file_stem()
        .map_or_else(Default::default, |s| s.to_string_lossy().into_owned());
    let name = match path.extension() {
        Some(ext) => format!("{stem}.{tag}.{}", ext.to_string_lossy()),
        None => format!("{stem}.{tag}"),
    };
    path.with_file_name(name)
}

/// One front read back, with the stop its points were designed over.
struct Arm {
    /// The file it came from, for every message that names an input.
    path: PathBuf,
    header: Value,
    points: Vec<Value>,
}

/// A point's energy and adaptation, the axes the front is taken over.
fn axes(point: &Value, path: &Path) -> Result<(f64, f64)> {
    let field = |name: &str| {
        point[name]
            .as_f64()
            .ok_or_else(|| format!("{}: a point has no {name}", path.display()))
    };
    Ok((field("mfe")?, field("cai_log")?))
}

/// Whether `b` dominates `a`: at least as good on both axes and better on one.
/// Energy is minimized and adaptation maximized.
fn dominates(b: (f64, f64), a: (f64, f64)) -> bool {
    let (b_mfe, b_cai) = b;
    let (a_mfe, a_cai) = a;
    b_cai >= a_cai - EPS && b_mfe <= a_mfe + EPS && (b_cai > a_cai + EPS || b_mfe < a_mfe - EPS)
}

/// The non-dominated members of `pool`, keeping every point of a tie: two
/// designs of the same energy and adaptation are different molecules, and one
/// can be modular where the other is not.
///
/// Quadratic in the pool, which a front of a few thousand points does not
/// notice and a reader can check against the definition above. The sweep that
/// would be linearithmic has to treat a tie and a domination differently on
/// the same comparison, and this command exists because that distinction was
/// got wrong elsewhere.
fn non_dominated<'a>(pool: &[(&'a Value, (f64, f64))]) -> Vec<&'a Value> {
    pool.iter()
        .filter(|(_, a)| !pool.iter().any(|(_, b)| dominates(*b, *a)))
        .map(|(p, _)| *p)
        .collect()
}

/// The fields of a header that say which molecule a front is of. The stop is
/// not among them: fronts of different stops are exactly what this joins.
const IDENTITY: [&str; 6] = ["protein", "utr5", "utr3", "forbid", "access", "codon_usage"];

/// Refuse fronts of different molecules, naming the file and the field. A
/// union over two molecules is not a front of either.
fn one_molecule(arms: &[Arm]) -> Result<()> {
    let (first, rest) = arms.split_first().ok_or("derna pool: no fronts to join")?;
    for arm in rest {
        for field in IDENTITY {
            if arm.header[field] != first.header[field] {
                return Err(format!(
                    "derna pool: {} and {} are fronts of different molecules — their {field} \
                     differs, and a union over two molecules is a front of neither",
                    first.path.display(),
                    arm.path.display()
                )
                .into());
            }
        }
    }
    Ok(())
}

/// Refuse a front whose points carry no verdict, naming the file. The modular
/// front is the non-dominated subset of the certified designs, so a front that
/// has not been certified cannot contribute to it — and filtering it first and
/// reading the survivors would answer a different question.
fn every_arm_certified(arms: &[Arm]) -> Result<()> {
    for arm in arms {
        if let Some(point) = arm.points.iter().find(|p| p.get("modular").is_none()) {
            let at = point["cai_log"]
                .as_f64()
                .map_or_else(|| "a point".to_string(), |c| format!("the point at {c:.4}"));
            return Err(format!(
                "derna pool: {} has not been certified — {at} carries no verdict. \
                 Run `derna check` on it first; a modular front cannot be taken over \
                 points whose own free fold has not been folded",
                arm.path.display()
            )
            .into());
        }
    }
    Ok(())
}

/// The stop codon a point spells, read from the molecule rather than from a
/// label: a fixed-stop front carries its stop in the header and a designed one
/// on every row, and a reader that trusts the wrong one silently relabels a
/// whole arm.
fn spelled_stop(point: &Value, header: &Value) -> Option<String> {
    let sequence = point["sequence"].as_str()?;
    let utr3 = header["utr3"]["sequence"].as_str().unwrap_or("").len();
    let end = sequence.len().checked_sub(utr3)?;
    let start = end.checked_sub(3)?;
    Some(sequence.get(start..end)?.to_string())
}

/// Every point of `arm`, labelled with the stop codon it spells.
///
/// The label is taken from the sequence and only checked against the file's
/// own, so a front whose header and rows disagree is refused rather than
/// pooled under the wrong stop.
fn labelled(arm: &Arm) -> Result<Vec<Value>> {
    let declared = |point: &Value| {
        point
            .get("stop")
            .or_else(|| arm.header.get("stop"))
            .and_then(Value::as_str)
            .map(str::to_owned)
    };
    arm.points
        .iter()
        .map(|point| {
            let mut row = point.clone();
            match (spelled_stop(point, &arm.header), declared(point)) {
                (Some(spelled), Some(said)) if said != spelled && said != "none" => {
                    return Err(format!(
                        "derna pool: {} says its stop is {said} and a point spells {spelled}",
                        arm.path.display()
                    )
                    .into());
                }
                (Some(spelled), _) => {
                    row["stop"] = json!(spelled);
                }
                (None, _) => {}
            }
            Ok(row)
        })
        .collect()
}

/// What each input contributed, so the two written fronts say where their
/// points came from and what was known about each arm.
fn arm_provenance(arms: &[Arm]) -> Vec<Value> {
    arms.iter()
        .map(|arm| {
            let modular = arm
                .points
                .iter()
                .filter(|p| p["modular"] == json!(true))
                .count();
            json!({
                "front": arm.path.display().to_string(),
                "stop": arm.header.get("stop").cloned().unwrap_or(Value::Null),
                "points": arm.points.len(),
                "modular": modular,
                // A front that stopped on a budget holds fewer points than the
                // whole front, and a pooled front cannot say more than its
                // least complete arm. Carried per arm rather than reduced to
                // one flag, because one settled arm beside two open ones is
                // not the same as three open ones.
                "complete": arm.header.get("complete").cloned().unwrap_or(Value::Null),
                "stopped_by": arm.header.get("stopped_by").cloned().unwrap_or(Value::Null),
            })
        })
        .collect()
}

/// The header of one written front.
fn pooled_header(arms: &[Arm], front: &str, kept: usize, pooled: usize) -> Value {
    let mut h = Map::new();
    h.insert("record".into(), "header".into());
    h.insert("format".into(), "derna-result".into());
    h.insert("format_version".into(), result::FORMAT_VERSION.into());
    h.insert("command".into(), "pool".into());
    h.insert("version".into(), env!("CARGO_PKG_VERSION").into());
    // The molecule is the arms' shared one, which `one_molecule` has checked.
    for field in IDENTITY {
        if let Some(v) = arms[0].header.get(field) {
            h.insert(field.into(), v.clone());
        }
    }
    let stops: Vec<Value> = arms
        .iter()
        .map(|a| a.header.get("stop").cloned().unwrap_or(Value::Null))
        .collect();
    h.insert("stop".into(), json!(stops));
    h.insert("front".into(), json!(front));
    h.insert("points".into(), json!(kept));
    h.insert("pooled".into(), json!(pooled));
    h.insert("arms".into(), json!(arm_provenance(arms)));
    Value::Object(h)
}

/// The two files `derna pool` writes for `args`.
pub fn outputs(args: &PoolArgs) -> [PathBuf; 2] {
    [
        tagged_path(&args.out, "union"),
        tagged_path(&args.out, "modular"),
    ]
}

/// Run `derna pool`.
pub fn run(args: &PoolArgs) -> Result<()> {
    let [union_path, modular_path] = outputs(args);
    result::check_output(Some(&union_path), args.force)?;
    result::check_output(Some(&modular_path), args.force)?;

    let mut arms = Vec::new();
    for path in &args.inputs {
        let read = show::read(Some(path))?;
        arms.push(Arm {
            path: path.clone(),
            header: read.header,
            points: read.points,
        });
    }
    one_molecule(&arms)?;
    every_arm_certified(&arms)?;

    let pooled: Vec<Value> = arms
        .iter()
        .map(labelled)
        .collect::<Result<Vec<_>>>()?
        .concat();
    let mut keyed = Vec::with_capacity(pooled.len());
    for point in &pooled {
        keyed.push((point, axes(point, &args.out)?));
    }
    let modular: Vec<(&Value, (f64, f64))> = keyed
        .iter()
        .filter(|(p, _)| p["modular"] == json!(true))
        .copied()
        .collect();

    let union = non_dominated(&keyed);
    let modular_front = non_dominated(&modular);
    eprintln!(
        "{} point(s) pooled from {} front(s): {} on the masking front, \
         {} of {} modular design(s) on the modular front",
        pooled.len(),
        arms.len(),
        union.len(),
        modular_front.len(),
        modular.len(),
    );

    for (path, front, kept) in [
        (&union_path, "masking", &union),
        (&modular_path, "modular", &modular_front),
    ] {
        let mut lines = vec![pooled_header(&arms, front, kept.len(), pooled.len())];
        lines.extend(kept.iter().map(|p| (*p).clone()));
        let mut w = result::open_output(Some(path), args.force)?;
        result::write_lines(w.as_mut(), &lines)?;
    }
    Ok(())
}
