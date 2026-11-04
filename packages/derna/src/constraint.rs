//! The region constraint: which crossing pair classes are forbidden, and the
//! start-codon accessibility window.
//!
//! With fixed flanks given, the crossing classes that reach the coding
//! sequence — `5-cds` and `cds-3` — are forbidden unless `--forbid` says
//! otherwise. `5-3` is offered but not default, for the reason the
//! within-region classes are not offered at all: both its endpoints lie in
//! fixed flank sequence, so the free fold forms such a pair whatever the
//! codons spell, and forbidding it makes the design fail its own check
//! without changing the design.
//! The within-region classes (`5-5`, `cds-cds`, `3-3`) are not offered: the
//! first and last forbid structure inside a fixed flank that the free fold
//! forms anyway, and the middle one forbids all coding structure.

use derna_lib::{AccessibilityWindow, Enforcement, PairClass, RegionConstraints};

use crate::Result;

/// The crossing classes `--forbid` accepts, in output order.
pub const CROSSING: [PairClass; 3] = [PairClass::Utr5Cds, PairClass::CdsUtr3, PairClass::Utr5Utr3];

/// The classes the default forbids, of those whose flanks are present.
///
/// `5-3` is left out: both endpoints of such a pair lie in fixed flank
/// sequence, so no codon assignment spells it away. Forbidding it therefore
/// does not change which design is chosen — it changes the structure that
/// design is scored against, and leaves the design failing its own free-fold
/// check through nothing it could have done differently. That is the same
/// reason the within-region classes are not offered, and it is a choice the
/// caller makes rather than one a run makes for them.
pub const DEFAULT_FORBID: [PairClass; 2] = [PairClass::Utr5Cds, PairClass::CdsUtr3];

/// The constraint a run carries.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Constraint {
    /// Forbidden crossing classes, in [`CROSSING`] order.
    pub forbid: Vec<PairClass>,
    /// `(upstream, downstream)` nucleotides around the start codon kept
    /// unpaired, or `None`.
    pub access: Option<(u32, u32)>,
}

impl Constraint {
    /// Whether the constraint forbids nothing.
    pub fn is_inert(&self) -> bool {
        self.forbid.is_empty() && self.access.is_none()
    }

    /// The library table.
    pub fn table(&self) -> RegionConstraints {
        let table = RegionConstraints::enforcing(&self.forbid, Enforcement::Forbid);
        match self.access {
            Some((up, down)) => table.with_window(AccessibilityWindow::hard(up, down)),
            None => table,
        }
    }

    /// The classes by their `--forbid` names.
    pub fn forbid_names(&self) -> Vec<&'static str> {
        self.forbid.iter().map(|c| c.cli_name()).collect()
    }
}

/// Resolve `--forbid` and `--access` against which flanks are present.
pub fn resolve(
    forbid: Option<&str>,
    access: Option<&str>,
    has_utr5: bool,
    has_utr3: bool,
) -> Result<Constraint> {
    let present = |c: PairClass| missing_flank(c, has_utr5, has_utr3).is_none();
    let forbid = match forbid {
        None => DEFAULT_FORBID.into_iter().filter(|&c| present(c)).collect(),
        Some("none") => Vec::new(),
        Some(list) => parse_forbid_list(list, has_utr5, has_utr3)?,
    };
    let access = access.map(parse_access).transpose()?;
    Ok(Constraint { forbid, access })
}

/// The flank a crossing class needs and the run lacks, if any.
fn missing_flank(class: PairClass, has_utr5: bool, has_utr3: bool) -> Option<&'static str> {
    let needs5 = matches!(class, PairClass::Utr5Cds | PairClass::Utr5Utr3);
    let needs3 = matches!(class, PairClass::CdsUtr3 | PairClass::Utr5Utr3);
    match (needs5 && !has_utr5, needs3 && !has_utr3) {
        (true, true) => Some("no 5′ or 3′ UTR"),
        (true, false) => Some("no 5′ UTR"),
        (false, true) => Some("no 3′ UTR"),
        (false, false) => None,
    }
}

fn parse_forbid_list(list: &str, has_utr5: bool, has_utr3: bool) -> Result<Vec<PairClass>> {
    let mut named = Vec::new();
    for name in list.split(',').map(str::trim) {
        let class = CROSSING
            .into_iter()
            .find(|c| c.cli_name() == name)
            .ok_or_else(|| unknown_class(name))?;
        if let Some(missing) = missing_flank(class, has_utr5, has_utr3) {
            return Err(format!("--forbid {name}: {missing} to forbid pairs with").into());
        }
        named.push(class);
    }
    Ok(CROSSING.into_iter().filter(|c| named.contains(c)).collect())
}

fn unknown_class(name: &str) -> Box<dyn std::error::Error> {
    if PairClass::from_cli_name(name).is_some() {
        format!(
            "--forbid {name}: only the crossing classes 5-cds, cds-3 and 5-3 can be forbidden; \
             a within-region class forbids structure the free fold forms anyway"
        )
        .into()
    } else {
        format!("--forbid: unknown class {name:?} (5-cds, cds-3, 5-3, or none)").into()
    }
}

fn parse_access(value: &str) -> Result<(u32, u32)> {
    let parsed = value
        .split_once(',')
        .and_then(|(u, d)| Some((u.trim().parse().ok()?, d.trim().parse().ok()?)));
    match parsed {
        Some((0, 0)) => Err("--access 0,0 keeps nothing unpaired; omit the flag".into()),
        Some(window) => Ok(window),
        None => Err(format!("--access expects UP,DOWN nucleotide counts, got {value:?}").into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(c: &Constraint) -> Vec<&'static str> {
        c.forbid_names()
    }

    #[test]
    fn the_default_forbids_the_crossing_classes_that_reach_the_coding_sequence() {
        // Not `5-3`: both its endpoints are fixed flank sequence, so no codon
        // choice forms or avoids it.
        assert_eq!(
            names(&resolve(None, None, true, true).unwrap()),
            ["5-cds", "cds-3"]
        );
        assert_eq!(
            names(&resolve(Some("5-cds,cds-3,5-3"), None, true, true).unwrap()),
            ["5-cds", "cds-3", "5-3"],
            "5-3 is still available when asked for"
        );
        assert_eq!(names(&resolve(None, None, true, false).unwrap()), ["5-cds"]);
        assert_eq!(names(&resolve(None, None, false, true).unwrap()), ["cds-3"]);
        assert!(resolve(None, None, false, false).unwrap().is_inert());
    }

    #[test]
    fn the_default_is_the_explicit_list_it_stands_for() {
        // The default is a convenience, not a different constraint: writing
        // its classes out must resolve to the same table, or a result taken
        // with the default cannot be compared with one taken explicitly.
        let default = resolve(None, None, true, true).unwrap();
        let explicit = resolve(Some("5-cds,cds-3"), None, true, true).unwrap();
        assert_eq!(default, explicit);
        assert_eq!(default.table(), explicit.table());

        // With one flank the default drops what that flank's absence makes
        // empty, and writing the remainder out matches it.
        let one_flank = resolve(None, None, true, false).unwrap();
        assert_eq!(
            one_flank,
            resolve(Some("5-cds"), None, true, false).unwrap()
        );
    }

    #[test]
    fn the_order_of_a_class_list_does_not_matter() {
        // A list is a set: two spellings of the same set are one constraint,
        // and the resolved order is the one the output uses.
        let canonical = resolve(Some("5-cds,cds-3,5-3"), None, true, true).unwrap();
        for spelling in ["cds-3,5-3,5-cds", "5-3,5-cds,cds-3", "5-3,cds-3,5-cds"] {
            let other = resolve(Some(spelling), None, true, true).unwrap();
            assert_eq!(canonical, other, "{spelling}");
            assert_eq!(canonical.forbid_names(), other.forbid_names(), "{spelling}");
        }
    }

    #[test]
    fn none_turns_the_constraint_off() {
        assert!(resolve(Some("none"), None, true, true).unwrap().is_inert());
    }

    #[test]
    fn an_explicit_list_is_ordered_and_deduplicated() {
        let c = resolve(Some("5-3,5-cds,5-3"), None, true, true).unwrap();
        assert_eq!(names(&c), ["5-cds", "5-3"]);
    }

    #[test]
    fn a_class_without_its_flank_is_refused_when_explicit() {
        let e = resolve(Some("cds-3"), None, true, false)
            .unwrap_err()
            .to_string();
        assert!(e.contains("no 3′ UTR to forbid pairs with"), "{e}");
    }

    #[test]
    fn within_region_classes_are_refused() {
        let e = resolve(Some("5-5"), None, true, true)
            .unwrap_err()
            .to_string();
        assert!(e.contains("only the crossing classes"), "{e}");
        assert!(resolve(Some("5-x"), None, true, true).is_err());
    }

    #[test]
    fn access_takes_two_counts() {
        let c = resolve(Some("none"), Some("10,10"), true, true).unwrap();
        assert_eq!(c.access, Some((10, 10)));
        assert!(!c.is_inert());
        assert!(resolve(None, Some("10"), true, true).is_err());
        assert!(resolve(None, Some("0,0"), true, true).is_err());
    }
}
