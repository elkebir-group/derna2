//! `derna pick`: one design out of a result file, as a sequence to order.
//!
//! It reads a `design`, `front`, `check` or `pool` result, keeps the points
//! the filters admit, and writes the one point a rule chooses: as FASTA by
//! default, spelled as `show --fasta` spells a record, or as its JSONL row.
//! Like `show` it folds nothing; a file written before points carried their
//! sequence metrics gets them from each point's sequence.
//!
//! Ties break the same way under every rule — lower MFE, then higher CAI,
//! then the earlier point in the file — so the same file and flags always pick
//! the same point.

use std::cmp::Ordering;
use std::io::Write;
use std::path::PathBuf;

use serde_json::Value;

use crate::args::Flags;
use crate::show::{self, FastaStyle, Selection, FASTA_STYLE_HELP, SELECTION_HELP};
use crate::Result;

/// Which point a pick takes from those selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// The highest CAI.
    BestCai,
    /// The lowest MFE.
    BestMfe,
    /// The point farthest from the chord joining the selection's two ends,
    /// with MFE and CAI each normalized to [0, 1] over the selection.
    Knee,
}

/// `derna pick` as parsed.
#[derive(Debug)]
pub struct PickArgs {
    /// The result file, or `None` for stdin.
    pub path: Option<PathBuf>,
    pub rule: Rule,
    pub selection: Selection,
    pub style: FastaStyle,
    /// Write the point's JSONL row rather than FASTA.
    pub json: bool,
}

/// Help text for `derna pick`.
pub fn help() -> String {
    format!(
        "derna pick — one design out of a result file\n\n\
         Usage: derna pick RESULT.jsonl|- RULE [options]\n\n\
         Rule (exactly one)\n  \
         --best-cai                the highest CAI; with --max-mfe, under it\n  \
         --best-mfe                the lowest MFE; with --min-cai, above it\n  \
         --knee                    the point farthest from the chord joining the\n                            \
         selection's two ends, with MFE and CAI each\n                            \
         normalized to [0, 1] over the selected points\n\n\
         Ties go to the lower MFE, then the higher CAI, then the earlier point in\n\
         the file.\n\n\
         Selection\n\
         {SELECTION_HELP}\n\
         Output [default: FASTA]\n\
         {FASTA_STYLE_HELP}  \
         --json                    the point's JSONL row instead\n"
    )
}

/// Parse the words after `pick`.
pub fn parse(words: impl IntoIterator<Item = String>) -> Result<PickArgs> {
    let mut flags = Flags::new(words);
    let mut path = None;
    let mut rules = Vec::new();
    let mut selection = Selection::default();
    let mut style = FastaStyle::default();
    let mut json = false;
    while let Some(word) = flags.next_word() {
        if selection.take(&word, &mut flags)? || style.take(&word) {
            continue;
        }
        match word.as_str() {
            "--best-cai" => rules.push(Rule::BestCai),
            "--best-mfe" => rules.push(Rule::BestMfe),
            "--knee" => rules.push(Rule::Knee),
            "--json" => json = true,
            flag if flag.starts_with("--") => {
                return Err(format!("derna pick: unknown flag {flag:?} (see --help)").into())
            }
            p if path.is_none() => path = Some(PathBuf::from(p)),
            extra => {
                return Err(format!("derna pick: one result file at a time, got {extra:?}").into())
            }
        }
    }
    let path = path.ok_or("derna pick: give a result file, or `-` for stdin")?;
    let rule = match rules.as_slice() {
        [rule] => *rule,
        [] => return Err("derna pick: give a rule: --best-cai, --best-mfe or --knee".into()),
        _ => {
            return Err("derna pick: give exactly one of --best-cai, --best-mfe and --knee".into())
        }
    };
    if json && (style.dna || style.cds_only) {
        return Err("derna pick: --dna and --cds-only spell a FASTA record, not --json".into());
    }
    Ok(PickArgs {
        path: Some(path),
        rule,
        selection,
        style,
        json,
    })
}

fn mfe(p: &Value) -> f64 {
    p["mfe"].as_f64().unwrap_or(f64::INFINITY)
}

fn cai(p: &Value) -> f64 {
    p["cai"].as_f64().unwrap_or(f64::NEG_INFINITY)
}

/// The tie order: lower MFE, then higher CAI, then earlier in the file.
fn tie_order(a: &(usize, &Value), b: &(usize, &Value)) -> Ordering {
    mfe(a.1)
        .total_cmp(&mfe(b.1))
        .then(cai(b.1).total_cmp(&cai(a.1)))
        .then(a.0.cmp(&b.0))
}

/// Each point's distance from the chord joining the selection's two ends —
/// the lowest-MFE point and the highest-CAI point — in coordinates normalized
/// to [0, 1] over the selection. A coordinate the selection does not vary
/// in normalizes to 0.
fn knee_distances(points: &[(usize, &Value)]) -> Vec<f64> {
    let range = |f: fn(&Value) -> f64| {
        let lo = points
            .iter()
            .map(|(_, p)| f(p))
            .fold(f64::INFINITY, f64::min);
        let hi = points
            .iter()
            .map(|(_, p)| f(p))
            .fold(f64::NEG_INFINITY, f64::max);
        move |v: f64| if hi > lo { (v - lo) / (hi - lo) } else { 0.0 }
    };
    let (nx, ny) = (range(cai), range(mfe));
    let at = |p: &Value| (nx(cai(p)), ny(mfe(p)));
    let low_mfe = points.iter().min_by(|a, b| tie_order(a, b));
    let high_cai = points
        .iter()
        .min_by(|a, b| cai(b.1).total_cmp(&cai(a.1)).then(tie_order(a, b)));
    let (Some(a), Some(b)) = (low_mfe, high_cai) else {
        return Vec::new();
    };
    let ((x1, y1), (x2, y2)) = (at(a.1), at(b.1));
    let length = (x2 - x1).hypot(y2 - y1);
    points
        .iter()
        .map(|(_, p)| {
            let (x, y) = at(p);
            if length == 0.0 {
                0.0
            } else {
                ((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)).abs() / length
            }
        })
        .collect()
}

/// The point `rule` takes from `points`, each paired with its index in the
/// file; `None` when there is none.
pub fn choose<'p>(rule: Rule, points: &[(usize, &'p Value)]) -> Option<(usize, &'p Value)> {
    match rule {
        Rule::BestMfe => points.iter().min_by(|a, b| tie_order(a, b)).copied(),
        Rule::BestCai => points
            .iter()
            .min_by(|a, b| cai(b.1).total_cmp(&cai(a.1)).then(tie_order(a, b)))
            .copied(),
        Rule::Knee => {
            let d = knee_distances(points);
            (0..points.len())
                .min_by(|&i, &j| {
                    d[j].total_cmp(&d[i])
                        .then(tie_order(&points[i], &points[j]))
                })
                .map(|i| points[i])
        }
    }
}

/// The lines `pick` writes for `rendered`, or a refusal saying why no point
/// qualified.
pub fn picked(args: &PickArgs, rendered: &show::Rendered) -> Result<Vec<String>> {
    if rendered.points.is_empty() {
        return Err("derna pick: the result holds no point to pick".into());
    }
    if args.selection.modular_unanswerable(&rendered.points) {
        return Err(
            "derna pick: --modular keeps the points that pass their check, and no \
                    point in this file carries one — `derna check` adds them"
                .into(),
        );
    }
    let points = show::filled_points(rendered);
    let kept = args.selection.select(&points).map_err(|flag| {
        format!(
            "derna pick: no point qualifies — {flag} removed every point the filters \
             before it left"
        )
    })?;
    let (index, point) = choose(args.rule, &kept).ok_or("derna pick: no point qualifies")?;
    if args.json {
        return Ok(vec![point.to_string()]);
    }
    Ok(show::fasta_record(&rendered.header, index, point, args.style).to_vec())
}

/// Run `derna pick`.
pub fn run(args: &PickArgs) -> Result<()> {
    let rendered = show::read(args.path.as_deref())?;
    let lines = picked(args, &rendered)?;
    let mut out = std::io::stdout().lock();
    for line in lines {
        writeln!(out, "{line}")?;
    }
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn indexed(points: &[Value]) -> Vec<(usize, &Value)> {
        points.iter().enumerate().map(|(i, p)| (i + 1, p)).collect()
    }

    fn index_of(rule: Rule, points: &[Value]) -> usize {
        choose(rule, &indexed(points)).unwrap().0
    }

    #[test]
    fn each_rule_takes_its_end_of_the_front() {
        let front = [
            json!({"mfe": -10.0, "cai": 0.50}),
            json!({"mfe": -9.0, "cai": 0.80}),
            json!({"mfe": -5.0, "cai": 0.90}),
            json!({"mfe": -1.0, "cai": 0.95}),
        ];
        assert_eq!(index_of(Rule::BestMfe, &front), 1);
        assert_eq!(index_of(Rule::BestCai, &front), 4);
        // Normalized: (0, 0), (0.667, 0.111), (0.889, 0.556), (1, 1). The
        // chord is the diagonal and the second point sits farthest below it.
        assert_eq!(index_of(Rule::Knee, &front), 2);
    }

    #[test]
    fn ties_go_to_lower_mfe_then_higher_cai_then_file_order() {
        let tied_cai = [
            json!({"mfe": -2.0, "cai": 0.9}),
            json!({"mfe": -3.0, "cai": 0.9}),
        ];
        assert_eq!(index_of(Rule::BestCai, &tied_cai), 2);
        let tied_mfe = [
            json!({"mfe": -3.0, "cai": 0.8}),
            json!({"mfe": -3.0, "cai": 0.9}),
        ];
        assert_eq!(index_of(Rule::BestMfe, &tied_mfe), 2);
        let same = [
            json!({"mfe": -3.0, "cai": 0.9, "n": 1}),
            json!({"mfe": -3.0, "cai": 0.9, "n": 2}),
        ];
        for rule in [Rule::BestCai, Rule::BestMfe, Rule::Knee] {
            assert_eq!(index_of(rule, &same), 1, "{rule:?}");
        }
        // Two points: both lie on the chord, so the knee falls to the tie
        // order.
        let two = [
            json!({"mfe": -1.0, "cai": 0.9}),
            json!({"mfe": -3.0, "cai": 0.5}),
        ];
        assert_eq!(index_of(Rule::Knee, &two), 2);
    }

    #[test]
    fn exactly_one_rule_is_taken() {
        let w = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert!(parse(w(&["r.jsonl"])).is_err());
        assert!(parse(w(&["r.jsonl", "--knee", "--best-cai"])).is_err());
        assert!(parse(w(&["r.jsonl", "--knee", "--json", "--dna"])).is_err());
        let a = parse(w(&["r.jsonl", "--best-cai", "--max-mfe", "-3"])).unwrap();
        assert_eq!(a.rule, Rule::BestCai);
        assert_eq!(a.selection.max_mfe, Some(-3.0));
    }
}
