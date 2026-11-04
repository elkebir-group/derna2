//! Sequence metrics of a design: composition and runs, read off the sequence
//! alone. Nothing here folds.
//!
//! Every point a run writes carries them, and a reader of a file written
//! without them derives them the same way, from the point's `sequence` and the
//! coding window its header implies.

use std::ops::Range;

use serde_json::{Map, Value};

/// The width of the sliding window `gc_window_min` and `gc_window_max` are
/// taken over, in nucleotides.
pub const GC_WINDOW: usize = 50;

/// The fields the metrics are written under, in the order a point carries them.
pub const FIELDS: [&str; 6] = [
    "gc",
    "gc_cds",
    "u_cds",
    "gc_window_min",
    "gc_window_max",
    "max_homopolymer",
];

/// One design's sequence metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    /// GC fraction of the whole molecule.
    pub gc: f64,
    /// GC fraction of the coding sequence.
    pub gc_cds: f64,
    /// U fraction of the coding sequence.
    pub u_cds: f64,
    /// The least GC fraction of any [`GC_WINDOW`]-nt window of the whole
    /// molecule; the whole molecule's when it is shorter than one window.
    pub gc_window_min: f64,
    /// The greatest such fraction.
    pub gc_window_max: f64,
    /// The longest run of one nucleotide in the whole molecule.
    pub max_homopolymer: usize,
}

fn is_gc(b: u8) -> bool {
    matches!(b, b'G' | b'C' | b'g' | b'c')
}

fn is_u(b: u8) -> bool {
    matches!(b, b'U' | b'T' | b'u' | b't')
}

/// The share of `bytes` that `pick` selects; 0 for an empty slice.
fn fraction(bytes: &[u8], pick: fn(u8) -> bool) -> f64 {
    if bytes.is_empty() {
        return 0.0;
    }
    bytes.iter().filter(|&&b| pick(b)).count() as f64 / bytes.len() as f64
}

/// The least and greatest GC fraction over every `width`-nt window, counted
/// incrementally; the whole sequence's fraction when it is shorter.
fn gc_window_range(bytes: &[u8], width: usize) -> (f64, f64) {
    if bytes.len() <= width {
        let gc = fraction(bytes, is_gc);
        return (gc, gc);
    }
    let mut count = bytes[..width].iter().filter(|&&b| is_gc(b)).count();
    let (mut lo, mut hi) = (count, count);
    for i in width..bytes.len() {
        count += usize::from(is_gc(bytes[i]));
        count -= usize::from(is_gc(bytes[i - width]));
        lo = lo.min(count);
        hi = hi.max(count);
    }
    (lo as f64 / width as f64, hi as f64 / width as f64)
}

/// The longest run of one letter, case-insensitive; 0 for an empty sequence.
fn longest_run(bytes: &[u8]) -> usize {
    let mut best = 0;
    let mut run = 0;
    let mut last = None;
    for b in bytes.iter().map(u8::to_ascii_uppercase) {
        run = if last == Some(b) { run + 1 } else { 1 };
        last = Some(b);
        best = best.max(run);
    }
    best
}

/// The metrics of `sequence`, whose coding sequence is `cds` (clamped to the
/// sequence).
pub fn of(sequence: &str, cds: Range<usize>) -> Metrics {
    let bytes = sequence.as_bytes();
    let end = cds.end.min(bytes.len());
    let coding = &bytes[cds.start.min(end)..end];
    let (gc_window_min, gc_window_max) = gc_window_range(bytes, GC_WINDOW);
    Metrics {
        gc: fraction(bytes, is_gc),
        gc_cds: fraction(coding, is_gc),
        u_cds: fraction(coding, is_u),
        gc_window_min,
        gc_window_max,
        max_homopolymer: longest_run(bytes),
    }
}

impl Metrics {
    /// The metrics as point fields, named by [`FIELDS`].
    pub fn fields(&self) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert("gc".into(), self.gc.into());
        m.insert("gc_cds".into(), self.gc_cds.into());
        m.insert("u_cds".into(), self.u_cds.into());
        m.insert("gc_window_min".into(), self.gc_window_min.into());
        m.insert("gc_window_max".into(), self.gc_window_max.into());
        m.insert("max_homopolymer".into(), self.max_homopolymer.into());
        m
    }
}

/// The coding sequence's window in a molecule of the run `header` describes:
/// from the start codon, after the 5′ UTR, through the stop codon, or through
/// the last residue's codon when the run has none.
///
/// A given stop sits at the head of the 3′ flank and a designed one is the
/// coding sequence's last slot; either way it is three coding nucleotides
/// right after the residues' codons. A pooled header lists one stop per arm,
/// and every arm of a pool has one when any does.
pub fn coding_window(header: &Value) -> Range<usize> {
    let utr5 = header["utr5"]["sequence"].as_str().map_or(0, str::len);
    let residues = header["protein"]["sequence"].as_str().map_or(0, str::len);
    let stop = match &header["stop"] {
        Value::Null => false,
        Value::Array(stops) => stops.iter().any(|s| !s.is_null()),
        _ => true,
    };
    utr5..utr5 + 3 * residues + if stop { 3 } else { 0 }
}

/// `point` with its metrics, computed from its `sequence` over `cds`. A line
/// without a sequence is returned as it is.
pub fn add_to(point: &mut Value, cds: &Range<usize>) {
    let Some(sequence) = point["sequence"].as_str() else {
        return;
    };
    let fields = of(sequence, cds.clone()).fields();
    if let Value::Object(p) = point {
        p.extend(fields);
    }
}

/// `point` with any metric it lacks derived from its sequence, for a file
/// written before points carried them. A point that has every one is
/// returned untouched.
pub fn filled(point: &Value, header: &Value) -> Value {
    let mut p = point.clone();
    if FIELDS.iter().any(|f| p.get(f).is_none()) {
        let fields = p["sequence"]
            .as_str()
            .map(|s| of(s, coding_window(header)).fields());
        if let (Value::Object(map), Some(fields)) = (&mut p, fields) {
            for (k, v) in fields {
                map.entry(k).or_insert(v);
            }
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn composition_is_counted_by_hand() {
        // 5′ AAGG | AUG GCU UAA | CC 3′: 15 nt, with G or C at 1-based
        // positions 3, 4, 7, 8, 9, 14 and 15 → 7 of 15.
        let m = of("AAGGAUGGCUUAACC", 4..13);
        assert_eq!(m.gc, 7.0 / 15.0);
        // AUGGCUUAA: G, G, C → 3 of 9; U at 2, 6, 7 → 3 of 9.
        assert_eq!(m.gc_cds, 3.0 / 9.0);
        assert_eq!(m.u_cds, 3.0 / 9.0);
        // Shorter than a window: both ends are the whole molecule's.
        assert_eq!((m.gc_window_min, m.gc_window_max), (m.gc, m.gc));
        // AA, GG, UU, AA, CC: the longest run is 2.
        assert_eq!(m.max_homopolymer, 2);
    }

    #[test]
    fn windows_slide_one_nucleotide_at_a_time() {
        // 50 A then 50 G: the first window has no GC, the last is all GC, and
        // every window between moves by one.
        let s = format!("{}{}", "A".repeat(50), "G".repeat(50));
        let m = of(&s, 0..s.len());
        assert_eq!((m.gc_window_min, m.gc_window_max), (0.0, 1.0));
        assert_eq!(m.gc, 0.5);
        assert_eq!(m.max_homopolymer, 50);
        // Exactly one window wide: the one window is the whole molecule.
        let one = format!("{}{}", "C".repeat(10), "U".repeat(40));
        let w = of(&one, 0..50);
        assert_eq!((w.gc_window_min, w.gc_window_max), (0.2, 0.2));
        // 51 nt: two windows, CCCCCCCCCC U×40 and C×9 U×40 A → 0.2 and 0.18.
        let two = format!("{one}A");
        let w = of(&two, 0..51);
        assert_eq!((w.gc_window_min, w.gc_window_max), (9.0 / 50.0, 0.2));
    }

    #[test]
    fn a_homopolymer_is_the_longest_run_of_one_letter() {
        assert_eq!(longest_run(b""), 0);
        assert_eq!(longest_run(b"A"), 1);
        assert_eq!(longest_run(b"ACGU"), 1);
        assert_eq!(longest_run(b"AAACCCCGU"), 4);
        assert_eq!(longest_run(b"GUUUUU"), 5);
    }

    #[test]
    fn the_coding_window_follows_the_header() {
        let h = |stop: Value| json!({"protein": {"sequence": "MKV"}, "utr5": {"sequence": "ACGU"}, "stop": stop});
        assert_eq!(coding_window(&h(Value::Null)), 4..13);
        assert_eq!(coding_window(&h(json!("UAA"))), 4..16);
        assert_eq!(coding_window(&h(json!("free"))), 4..16);
        assert_eq!(coding_window(&h(json!(["UAA", "UGA"]))), 4..16);
        let bare = json!({"protein": {"sequence": "MK"}, "utr5": null, "stop": null});
        assert_eq!(coding_window(&bare), 0..6);
    }

    #[test]
    fn a_point_without_metrics_gains_them_and_one_with_them_keeps_its_own() {
        let header = json!({"protein": {"sequence": "M"}, "utr5": null, "stop": null});
        let old = json!({"record": "point", "sequence": "AUG"});
        let filled_in = filled(&old, &header);
        assert_eq!(filled_in["gc"], json!(1.0 / 3.0));
        assert_eq!(filled_in["u_cds"], json!(1.0 / 3.0));
        assert_eq!(filled_in["max_homopolymer"], json!(1));
        let mut kept = filled_in.clone();
        kept["gc"] = json!(0.9);
        assert_eq!(filled(&kept, &header)["gc"], json!(0.9));
    }
}
