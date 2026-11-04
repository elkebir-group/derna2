//! Sequence constraints on a design: forbidden motifs, and pinned
//! nucleotides.
//!
//! Both are read as patterns or as a file of one pattern per line, and both
//! are stated in the coding sequence's own coordinates: a position is
//! 1-based from the start codon, a negative one counts back from the CDS end,
//! and a range is inclusive. A UTR coordinate is refused — the flanks are
//! fixed, so there is nothing there to constrain.
//!
//! Motifs are forbidden **over the coding sequence**: every occurrence with a
//! position in the CDS, including one straddling a CDS/UTR boundary, since the
//! codons on the CDS side are being chosen. An occurrence lying wholly inside
//! a fixed UTR cannot be removed by any codon assignment; the run proceeds and
//! says so rather than refusing.
//!
//! A narrower span is named by `--motif-window` for the whole set, or per
//! pattern — `--motif PATH@LO..HI`, or a `@window LO..HI` line inside a set
//! file. One run can therefore forbid one blacklist everywhere and another
//! only past a given coding position, which is what a set targeting
//! *internal* occurrences of a pattern needs when the pattern can also be
//! spelled across the 5′ boundary.

use derna_lib::ac::expand_iupac_motif;
use derna_lib::{MotifWindow, PositionalSpec};

use crate::molecule::Molecule;
use crate::Result;

/// One forbidden motif with the window it is forbidden inside.
///
/// A pattern carries its own window so that one run can say *this blacklist
/// over the whole coding sequence, that one only from coding position 5
/// onward*. The two are one item rather than two parallel lists because the
/// pairing is the constraint: a reader given the patterns and the windows
/// separately cannot say which pattern was scoped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopedMotif {
    /// The IUPAC pattern.
    pub pattern: String,
    /// The window in CDS coordinates this pattern carries of its own, or
    /// `None` to take `--motif-window`'s.
    pub window: Option<(i64, i64)>,
}

/// The sequence constraints as written.
#[derive(Debug, Default, Clone)]
pub struct SequenceConstraints {
    /// Forbidden motifs, in the order given, each with the window it carries.
    pub motifs: Vec<ScopedMotif>,
    /// `--motif-window LO..HI`, in CDS coordinates, or `None` for the whole
    /// CDS. The default for every pattern that carries no window of its own.
    pub motif_window: Option<(i64, i64)>,
    /// `--pin` entries as written, each `K:LETTERS`.
    pub pins: Vec<String>,
}

impl SequenceConstraints {
    /// Consume `flag` (and its value) if it is a sequence-constraint flag.
    pub fn take(&mut self, flag: &str, flags: &mut crate::args::Flags) -> Result<bool> {
        match flag {
            "--motif" => {
                let value = flags.value(flag)?;
                self.motifs.extend(scoped_patterns(&value)?);
            }
            "--pin" => {
                let value = flags.value(flag)?;
                self.pins.extend(patterns(flag, &value)?);
            }
            "--motif-window" => {
                let value = flags.value(flag)?;
                if self.motif_window.is_some() {
                    return Err("--motif-window is given twice".into());
                }
                self.motif_window = Some(parse_window("--motif-window", &value)?);
            }
            "--motifs" | "--pins" => {
                return Err(format!(
                    "{flag} is not a flag: {} takes a pattern or a file of one per line",
                    flag.trim_end_matches('s')
                )
                .into())
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Whether nothing was asked for.
    pub fn is_empty(&self) -> bool {
        self.motifs.is_empty() && self.pins.is_empty() && self.motif_window.is_none()
    }

    /// Refuse a combination the command cannot act on.
    ///
    /// `--motif-window` narrows where a motif set is enforced, so with no set
    /// it changes nothing: accepted, it would be read as having scoped
    /// something, and its own bounds would never be checked against the
    /// molecule either.
    pub fn validate(&self) -> Result<()> {
        if self.motif_window.is_some() && self.motifs.is_empty() {
            return Err(
                "--motif-window narrows where the motifs are forbidden, and this run \
                        names no --motif"
                    .into(),
            );
        }
        Ok(())
    }

    /// The patterns alone, in the order given.
    pub fn patterns(&self) -> Vec<&str> {
        self.motifs.iter().map(|m| m.pattern.as_str()).collect()
    }

    /// The window the motifs are enforced over, resolved against `m`'s layout.
    ///
    /// Without `--motif-window` this is the whole coding sequence. The reach is
    /// overlapping either way, so a pattern cannot be spelled across the
    /// boundary of the window's CDS positions.
    pub fn window(&self, m: &Molecule) -> Result<MotifWindow> {
        self.span(m, self.motif_window, "--motif-window")
    }

    /// Every pattern with the window it is enforced over, resolved against
    /// `m`'s layout: its own where it carries one, `--motif-window`'s
    /// otherwise.
    ///
    /// This is the constraint the automaton is built from, so a caller reads it
    /// rather than the patterns and the windows separately.
    pub fn placed(&self, m: &Molecule) -> Result<Vec<(String, MotifWindow)>> {
        self.motifs
            .iter()
            .map(|motif| {
                let window = motif.window.or(self.motif_window);
                let what = match motif.window {
                    Some(_) => "a motif's own window",
                    None => "--motif-window",
                };
                Ok((motif.pattern.clone(), self.span(m, window, what)?))
            })
            .collect()
    }

    /// Whether some pattern carries a window of its own, so the set is not
    /// enforced over one span and no single phrase describes it.
    pub fn is_scoped(&self) -> bool {
        self.motifs.iter().any(|m| m.window.is_some())
    }

    /// `window` in whole-molecule positions, as the enforced span.
    fn span(&self, m: &Molecule, window: Option<(i64, i64)>, what: &str) -> Result<MotifWindow> {
        let cds = m.cds();
        let (lo, hi) = match window {
            None => (cds.start as u32, cds.end as u32),
            Some((from, to)) => {
                let coding = (cds.end - cds.start) as i64;
                let resolve = |k: i64| -> Result<u32> {
                    let offset = if k > 0 { k - 1 } else { coding + k };
                    if offset < 0 || offset >= coding {
                        return Err(format!(
                            "{what}: position {k} lies outside the {coding}-nucleotide \
                             coding sequence"
                        )
                        .into());
                    }
                    Ok(cds.start as u32 + offset as u32)
                };
                let (from, to) = (resolve(from)?, resolve(to)?);
                if from > to {
                    return Err(format!("{what} LO..HI: LO comes after HI").into());
                }
                // The written range is inclusive; the window's end is exclusive.
                (from, to + 1)
            }
        };
        Ok(MotifWindow::overlapping(lo, hi))
    }

    /// The pins, as the library's positional spec. Each is read in CDS
    /// coordinates.
    pub fn positional(&self) -> Result<PositionalSpec> {
        let mut spec = PositionalSpec::default();
        for pin in &self.pins {
            spec.add_pin(&cds_coordinate(pin)?)
                .map_err(|e| format!("--pin {pin}: {e}"))?;
        }
        Ok(spec)
    }

    /// The sequence constraints as a result header records them, so a reader —
    /// or a later command deciding what it may do with the design — knows
    /// what the run was under. Absent fields mean the run asked for none.
    pub fn header_fields(
        &self,
        m: &Molecule,
    ) -> Result<serde_json::Map<String, serde_json::Value>> {
        use serde_json::json;
        let mut fields = serde_json::Map::new();
        if self.motifs.is_empty() && self.pins.is_empty() {
            return Ok(fields);
        }
        if !self.motifs.is_empty() {
            fields.insert("motifs".into(), json!(self.patterns()));
            fields.insert("motif_window".into(), json!(self.enforced_over()));
            // A pattern's window is recorded beside the pattern, because the
            // pairing is what the constraint is: `motifs` and one
            // `motif_window` no longer determine it once two patterns carry
            // different windows, and a resume comparing those two fields alone
            // would read `{A: 5.., B: all}` as `{A: all, B: 5..}`.
            if self.is_scoped() {
                let scopes: Vec<serde_json::Value> = self
                    .motifs
                    .iter()
                    .map(|motif| {
                        json!({
                            "motif": motif.pattern,
                            "window": in_words(motif.window.or(self.motif_window)),
                        })
                    })
                    .collect();
                fields.insert("motif_scopes".into(), json!(scopes));
            }
            // The occurrences no codon assignment can remove, recorded beside
            // the set: the run warned about them, and a file read later has no
            // other way to learn the design was never asked to remove them.
            let unremovable: Vec<serde_json::Value> = self
                .unremovable(m)?
                .into_iter()
                .map(|(side, motif, at)| json!({"side": side, "motif": motif, "position": at}))
                .collect();
            fields.insert("motifs_in_fixed_flanks".into(), json!(unremovable));
        }
        if !self.pins.is_empty() {
            fields.insert("pins".into(), json!(self.pins));
        }
        Ok(fields)
    }

    /// Where the motifs are enforced, in words, for a warning or a header.
    ///
    /// This is the set-wide window — the whole coding sequence, or the
    /// sub-window `--motif-window` named — which is where every pattern
    /// carrying no window of its own is enforced. A set holding a pattern that
    /// does carry one is not enforced over a single span, and
    /// [`Self::is_scoped`] is what a caller about to name one span asks.
    pub fn enforced_over(&self) -> String {
        in_words(self.motif_window)
    }

    /// Where the motifs are enforced, in a phrase a warning can use whether or
    /// not a pattern carries its own window.
    pub fn enforced_phrase(&self) -> String {
        if self.is_scoped() {
            return format!(
                "{} except where a motif carries a window of its own",
                self.enforced_over()
            );
        }
        self.enforced_over()
    }

    /// Occurrences of the motifs lying wholly inside a fixed UTR, as
    /// `(side, motif, 1-based position in that UTR)`.
    ///
    /// No codon assignment can remove one, so it is reported rather than
    /// forbidden. The scan is over the flank's own sequence, so an occurrence
    /// straddling the boundary is not one of these — the window forbids those.
    pub fn unremovable(&self, m: &Molecule) -> Result<Vec<(&'static str, String, usize)>> {
        let mut found = Vec::new();
        for (side, flank) in [("5′ UTR", m.utr5.as_ref()), ("3′ UTR", m.utr3.as_ref())] {
            let Some(flank) = flank else { continue };
            for motif in &self.motifs {
                for at in occurrences(&motif.pattern, &flank.sequence)? {
                    found.push((side, motif.pattern.clone(), at + 1));
                }
            }
        }
        Ok(found)
    }
}

/// Every 0-based offset in `text` where the IUPAC `motif` occurs.
///
/// Matched one position at a time rather than by expanding the motif into its
/// literal spellings: an expansion is exponential in the number of degenerate
/// codes, and `--motif NNNNNNNNNN` is something a caller can type. Each
/// position's allowed letters still come from the library's own expansion of
/// that single code, so the IUPAC vocabulary is not restated here.
fn occurrences(motif: &str, text: &str) -> Result<Vec<usize>> {
    let classes: Vec<Vec<u8>> = motif
        .chars()
        .map(|code| {
            let letters = expand_iupac_motif(&code.to_string())
                .map_err(|e| format!("--motif {motif}: {e}"))?;
            Ok(letters.iter().filter_map(|l| l.bytes().next()).collect())
        })
        .collect::<Result<_>>()?;
    let (text, width) = (text.as_bytes(), classes.len());
    if width == 0 || text.len() < width {
        return Ok(Vec::new());
    }
    Ok((0..=text.len() - width)
        .filter(|&at| {
            classes
                .iter()
                .zip(&text[at..at + width])
                .all(|(allowed, letter)| allowed.contains(letter))
        })
        .collect())
}

/// A flag's value: a file of one pattern per line, or the pattern itself.
///
/// A value naming an existing file is read as a file; `#` begins a comment and
/// blank lines are skipped, so a list can be annotated.
fn patterns(flag: &str, value: &str) -> Result<Vec<String>> {
    if !std::path::Path::new(value).is_file() {
        return Ok(vec![value.to_string()]);
    }
    let text =
        std::fs::read_to_string(value).map_err(|e| format!("{flag}: cannot read {value}: {e}"))?;
    let lines: Vec<String> = text
        .lines()
        .map(|l| l.split('#').next().unwrap_or_default().trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    if lines.is_empty() {
        return Err(format!("{flag}: {value} holds no pattern").into());
    }
    Ok(lines)
}

/// A `--motif` value's patterns, each with the window it carries.
///
/// The value is a pattern or a file of one per line, as `--pin`'s is, and it
/// may name a window two ways. `PATH@LO..HI` or `SEQ@LO..HI` scopes everything
/// the value names; a `@window LO..HI` line inside a file scopes every pattern
/// **below** it, until the next such line. So a set file whose comment explains
/// that its pattern is about internal occurrences can carry that scope itself,
/// and a command line cannot leave it off.
///
/// A directive is a line of its own rather than a `#` comment because a
/// comment is stripped by every tool that reads these files, including this
/// one: a scope written as a comment would be silently dropped by an older
/// build instead of refused by it.
fn scoped_patterns(value: &str) -> Result<Vec<ScopedMotif>> {
    let (source, default) = match value.rsplit_once('@') {
        Some((head, window)) if !head.is_empty() => {
            (head, Some(parse_window("--motif PATH@LO..HI", window)?))
        }
        _ => (value, None),
    };
    if !std::path::Path::new(source).is_file() {
        return Ok(vec![ScopedMotif {
            pattern: source.to_string(),
            window: default,
        }]);
    }
    let text = std::fs::read_to_string(source)
        .map_err(|e| format!("--motif: cannot read {source}: {e}"))?;
    let mut out = Vec::new();
    let mut window = default;
    for (number, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        if let Some(range) = line.strip_prefix("@window") {
            let what = format!("--motif: @window on line {} of {source}", number + 1);
            window = Some(parse_window(&what, range.trim())?);
            continue;
        }
        out.push(ScopedMotif {
            pattern: line.to_string(),
            window,
        });
    }
    if out.is_empty() {
        return Err(format!("--motif: {source} holds no pattern").into());
    }
    Ok(out)
}

/// A window in words, for a warning or a header: the whole coding sequence, or
/// the sub-window a `LO..HI` named.
fn in_words(window: Option<(i64, i64)>) -> String {
    match window {
        None => "the coding sequence".to_string(),
        Some((lo, hi)) => format!("coding positions {lo}..{hi}"),
    }
}

/// `LO..HI`, both in CDS coordinates. `what` names where the range was
/// written, since a window reaches the parser from a flag, from a `--motif`
/// value's suffix, or from a line of a set file.
fn parse_window(what: &str, value: &str) -> Result<(i64, i64)> {
    let syntax = || -> Box<dyn std::error::Error> {
        format!(
            "{what} expects LO..HI in coding-sequence positions \
             (1-based, inclusive, negative counting back from the end), got {value:?}"
        )
        .into()
    };
    let (lo, hi) = value.split_once("..").ok_or_else(syntax)?;
    let parse = |s: &str| s.trim().parse::<i64>().ok().filter(|&k| k != 0);
    match (parse(lo), parse(hi)) {
        (Some(lo), Some(hi)) => Ok((lo, hi)),
        _ => Err(syntax()),
    }
}

/// A `K:LETTERS` pin rewritten in the library's coordinate syntax, which names
/// the region: every `derna` coordinate is a CDS one.
fn cds_coordinate(pin: &str) -> Result<String> {
    let (head, letters) = pin
        .rsplit_once(':')
        .ok_or_else(|| format!("--pin expects K:LETTERS, e.g. 4:G or -6:GCCACC, got {pin:?}"))?;
    let head = head.trim();
    if let Some((region, _)) = head.rsplit_once(':') {
        return Err(format!(
            "--pin {pin}: {region:?} names a region, and every derna coordinate is a \
             coding-sequence position — the UTRs are fixed sequence. Write {}:{letters}",
            head.rsplit(':').next().unwrap_or_default()
        )
        .into());
    }
    for prefix in ["5utr", "3utr", "mrna", "cds"] {
        if head.to_ascii_lowercase().starts_with(prefix) {
            return Err(format!(
                "--pin {pin}: a coordinate is a plain coding-sequence position; \
                 drop the {prefix:?} prefix"
            )
            .into());
        }
    }
    match head.parse::<i64>() {
        Ok(0) | Err(_) => Err(format!(
            "--pin {pin}: {head:?} is not a position (1-based, negative counts back from \
             the coding sequence's end)"
        )
        .into()),
        Ok(k) => Ok(format!("cds:{k}:{letters}")),
    }
}

/// `derna design`'s sequence-constraint flags. It forbids its whole motif
/// set over one window, so a pattern's own window is not offered here.
pub const SEQUENCE_HELP: &str = "\
Sequence constraints
  --motif PATH|SEQ          forbidden motif (IUPAC), or a file of one per
                            line; repeatable
  --motif-window LO..HI     forbid only over this coding sub-window
                            [default: the whole coding sequence]
  --pin PATH|K:LETTERS      fix coding nucleotides from position K (IUPAC),
                            or a file of one per line; repeatable
";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::Flags;

    fn parse(words: &[&str]) -> Result<SequenceConstraints> {
        let mut flags = Flags::new(words.iter().map(|s| s.to_string()));
        let mut sc = SequenceConstraints::default();
        while let Some(flag) = flags.next_word() {
            if !sc.take(&flag, &mut flags)? {
                return Err(format!("unknown flag {flag}").into());
            }
        }
        Ok(sc)
    }

    #[test]
    fn motifs_and_pins_repeat() {
        let sc = parse(&["--motif", "GAAUUC", "--motif", "GGGG", "--pin", "4:G"]).unwrap();
        assert_eq!(sc.patterns(), ["GAAUUC", "GGGG"]);
        assert_eq!(sc.pins, ["4:G"]);
    }

    #[test]
    fn the_plural_spellings_point_at_the_singular_ones() {
        for flag in ["--motifs", "--pins"] {
            let e = parse(&[flag, "list.txt"]).unwrap_err().to_string();
            assert!(e.contains("takes a pattern or a file"), "{flag}: {e}");
        }
    }

    #[test]
    fn a_window_without_a_motif_set_is_refused() {
        // Accepted, it would read as having scoped something, and its own
        // bounds would never be checked against the molecule either.
        let e = parse(&["--motif-window", "4..9"])
            .unwrap()
            .validate()
            .unwrap_err()
            .to_string();
        assert!(e.contains("names no --motif"), "{e}");
        assert!(parse(&["--motif", "GGGG", "--motif-window", "4..9"])
            .unwrap()
            .validate()
            .is_ok());
        assert!(parse(&["--pin", "9:A"]).unwrap().validate().is_ok());
    }

    #[test]
    fn a_window_is_read_as_a_closed_range() {
        let sc = parse(&["--motif-window", "4..9"]).unwrap();
        assert_eq!(sc.motif_window, Some((4, 9)));
        assert_eq!(
            parse(&["--motif-window", "-9..-4"]).unwrap().motif_window,
            Some((-9, -4))
        );
        for bad in ["4", "4..", "0..9", "4..0", "a..b", "4-9"] {
            assert!(parse(&["--motif-window", bad]).is_err(), "{bad}");
        }
        assert!(parse(&["--motif-window", "1..2", "--motif-window", "3..4"]).is_err());
    }

    #[test]
    fn a_pin_is_rewritten_as_a_coding_coordinate() {
        assert_eq!(cds_coordinate("4:G").unwrap(), "cds:4:G");
        assert_eq!(cds_coordinate("-6:GCCACC").unwrap(), "cds:-6:GCCACC");
    }

    #[test]
    fn a_utr_pin_is_refused_because_the_flanks_are_fixed() {
        let e = cds_coordinate("5utr:-6:GCCACC").unwrap_err().to_string();
        assert!(
            e.contains("every derna coordinate is a coding-sequence position"),
            "{e}"
        );
        // The library's own CDS prefix is refused too: there is one spelling.
        assert!(cds_coordinate("cds:4:G").is_err());
        assert!(cds_coordinate("0:G").is_err());
        assert!(cds_coordinate("G").is_err());
    }

    #[test]
    fn a_degenerate_motif_matches_without_being_expanded() {
        // `GNAC` matches GAAC, GCAC, GGAC and GUAC. Matched per position, so a
        // motif with many degenerate codes costs its length rather than its
        // number of spellings — `NNNNNNNNNNNNNNN` is a sixteen-millionfold
        // expansion and a caller can type it.
        assert_eq!(occurrences("GNAC", "AGGACGUACA").unwrap(), [1, 5]);
        assert_eq!(occurrences("N", "ACGU").unwrap(), [0, 1, 2, 3]);
        let wide = "N".repeat(15);
        assert_eq!(occurrences(&wide, "ACGU").unwrap(), Vec::<usize>::new());
        assert_eq!(occurrences(&wide, &"ACGU".repeat(4)).unwrap(), [0, 1]);
        // Overlapping occurrences are all reported: each is a pair of
        // positions a design would have to change.
        assert_eq!(occurrences("AA", "AAAA").unwrap(), [0, 1, 2]);
        assert!(occurrences("ACGX", "ACGU").is_err());
    }

    #[test]
    fn a_pattern_file_is_read_with_its_comments_stripped() {
        let path = std::env::temp_dir().join(format!("derna-motifs-{}.txt", std::process::id()));
        std::fs::write(&path, "# restriction sites\nGAAUUC\n\nGGAUCC  # BamHI\n").unwrap();
        let read = patterns("--motif", path.to_str().unwrap());
        std::fs::remove_file(&path).unwrap();
        assert_eq!(read.unwrap(), ["GAAUUC", "GGAUCC"]);
    }

    /// A set file written for one test, removed when it returns.
    struct SetFile(std::path::PathBuf);

    impl SetFile {
        fn new(name: &str, text: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("derna-set-{name}-{}.txt", std::process::id()));
            std::fs::write(&path, text).unwrap();
            Self(path)
        }

        fn path(&self) -> &str {
            self.0.to_str().unwrap()
        }
    }

    impl Drop for SetFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn a_window_in_a_motif_value_scopes_what_that_value_names() {
        let sc = parse(&["--motif", "GGAUCC@5..-1"]).unwrap();
        assert_eq!(sc.patterns(), ["GGAUCC"]);
        assert_eq!(sc.motifs[0].window, Some((5, -1)));
        // A file's suffix scopes every pattern the file names.
        let set = SetFile::new("suffix", "GAAUUC\nGGAUCC\n");
        let sc = parse(&["--motif", &format!("{}@5..-1", set.path())]).unwrap();
        assert_eq!(sc.patterns(), ["GAAUUC", "GGAUCC"]);
        assert!(sc.motifs.iter().all(|m| m.window == Some((5, -1))));
        // The set-wide flag keeps its own meaning beside them.
        assert_eq!(sc.motif_window, None);
    }

    #[test]
    fn a_window_directive_scopes_the_patterns_below_it() {
        // A pattern above the directive is unaffected, and a second directive
        // rebinds: the scope is positional in the file, so a set file can hold
        // one blacklist enforced everywhere and another enforced past a given
        // coding position.
        let set = SetFile::new(
            "directive",
            "AAAAAA\n@window 5..-1\nGGAUCC  # scoped\n@window 7..-4\nCCCCCC\n",
        );
        let sc = parse(&["--motif", set.path()]).unwrap();
        assert_eq!(sc.patterns(), ["AAAAAA", "GGAUCC", "CCCCCC"]);
        assert_eq!(sc.motifs[0].window, None);
        assert_eq!(sc.motifs[1].window, Some((5, -1)));
        assert_eq!(sc.motifs[2].window, Some((7, -4)));
        assert!(sc.is_scoped());
        assert!(!parse(&["--motif", "GGAUCC"]).unwrap().is_scoped());
    }

    #[test]
    fn a_directive_beats_the_value_suffix_and_the_suffix_beats_neither() {
        // Precedence, from the narrowest statement outwards: a pattern's own
        // directive, then the window the `--motif` value carries, then
        // `--motif-window`, then the whole coding sequence.
        let set = SetFile::new("precedence", "AAAAAA\n@window 7..-4\nGGAUCC\n");
        let sc = parse(&[
            "--motif",
            &format!("{}@5..-1", set.path()),
            "--motif-window",
            "9..-9",
        ])
        .unwrap();
        assert_eq!(sc.motifs[0].window, Some((5, -1)));
        assert_eq!(sc.motifs[1].window, Some((7, -4)));
        assert_eq!(sc.motif_window, Some((9, -9)));
        // A pattern carrying nothing takes `--motif-window`, and with neither
        // the whole coding sequence — which is `None` at both levels.
        let sc = parse(&["--motif", "GGAUCC", "--motif-window", "9..-9"]).unwrap();
        assert_eq!(sc.motifs[0].window, None);
        assert_eq!(sc.motif_window, Some((9, -9)));
        let sc = parse(&["--motif", "GGAUCC"]).unwrap();
        assert_eq!(sc.motifs[0].window, None);
        assert_eq!(sc.motif_window, None);
    }

    #[test]
    fn a_malformed_window_names_where_it_was_written() {
        let set = SetFile::new("malformed", "AAAAAA\n@window 5..zzz\n");
        let e = parse(&["--motif", set.path()]).unwrap_err().to_string();
        assert!(e.contains("@window on line 2"), "{e}");
        assert!(e.contains(set.path()), "{e}");
        let e = parse(&["--motif", "GGAUCC@5..zzz"])
            .unwrap_err()
            .to_string();
        assert!(e.contains("--motif PATH@LO..HI"), "{e}");
        // A window reaching the end has to say so: a position is 1-based and
        // never 0, and both ends are written.
        for bad in ["5..", "..-1", "0..-1", "5..0"] {
            assert!(
                parse(&["--motif", &format!("GGAUCC@{bad}")]).is_err(),
                "{bad}"
            );
        }
    }
}
