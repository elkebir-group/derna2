//! `derna show --plot`: the selected points as an SVG scatter, CAI on x and
//! MFE on y, written by hand so the command needs no plotting dependency.
//!
//! Every point is one element of class `mark`: a square on the hull, a circle
//! off it; filled when the point is modular, open when it is not, and open in
//! grey when nothing checked it. The legend's shapes carry class `legend`, so a
//! reader counting marks counts points.

use std::fmt::Write as _;

use serde_json::Value;

const WIDTH: f64 = 640.0;
const HEIGHT: f64 = 440.0;
const LEFT: f64 = 78.0;
const RIGHT: f64 = 160.0;
const TOP: f64 = 44.0;
const BOTTOM: f64 = 58.0;
const TICKS: usize = 5;
const MARK: f64 = 4.5;
const INK: &str = "#1f4e8c";
const GREY: &str = "#8a8a8a";

/// `text` with the five XML special characters escaped.
pub fn escape(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '&' => "&amp;".to_string(),
            '<' => "&lt;".to_string(),
            '>' => "&gt;".to_string(),
            '"' => "&quot;".to_string(),
            '\'' => "&apos;".to_string(),
            c => c.to_string(),
        })
        .collect()
}

/// A closed range with room around it, and never empty, so a single point or
/// a column of equal values still lands inside the frame.
fn padded(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let (lo, hi) = values
        .filter(|v| v.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
    if !lo.is_finite() {
        return (0.0, 1.0);
    }
    let pad = if hi > lo {
        (hi - lo) * 0.05
    } else {
        0.5_f64.max(lo.abs() * 0.05)
    };
    (lo - pad, hi + pad)
}

/// How a point is drawn: its shape, its fill and its stroke.
fn mark(p: &Value, x: f64, y: f64) -> String {
    let (fill, stroke) = match p.get("modular").and_then(Value::as_bool) {
        Some(true) => (INK, INK),
        Some(false) => ("none", INK),
        None => ("none", GREY),
    };
    let style = format!(r#"fill="{fill}" stroke="{stroke}" stroke-width="1.5""#);
    if p["on_hull"] == Value::Bool(true) {
        format!(
            r#"<rect class="mark" x="{:.2}" y="{:.2}" width="{:.2}" height="{:.2}" {style}/>"#,
            x - MARK,
            y - MARK,
            2.0 * MARK,
            2.0 * MARK
        )
    } else {
        format!(r#"<circle class="mark" cx="{x:.2}" cy="{y:.2}" r="{MARK}" {style}/>"#)
    }
}

/// The SVG of `points`, titled `title`.
pub fn svg(title: &str, points: &[&Value]) -> String {
    let cai = |p: &&Value| p["cai"].as_f64().unwrap_or(f64::NAN);
    let mfe = |p: &&Value| p["mfe"].as_f64().unwrap_or(f64::NAN);
    let (x_lo, x_hi) = padded(points.iter().map(cai));
    let (y_lo, y_hi) = padded(points.iter().map(mfe));
    let plot_w = WIDTH - LEFT - RIGHT;
    let plot_h = HEIGHT - TOP - BOTTOM;
    let sx = |v: f64| LEFT + (v - x_lo) / (x_hi - x_lo) * plot_w;
    let sy = |v: f64| TOP + plot_h - (v - y_lo) / (y_hi - y_lo) * plot_h;

    let mut s = String::new();
    let _ = writeln!(
        s,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" font-family="sans-serif" font-size="12">"#
    );
    let _ = writeln!(
        s,
        r#"<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>"#
    );
    let _ = writeln!(
        s,
        r#"<text x="{:.1}" y="24" text-anchor="middle" font-size="15">{}</text>"#,
        LEFT + plot_w / 2.0,
        escape(title)
    );
    let _ = writeln!(
        s,
        r##"<rect x="{LEFT}" y="{TOP}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333"/>"##
    );
    for k in 0..=TICKS {
        let t = k as f64 / TICKS as f64;
        let (xv, yv) = (x_lo + t * (x_hi - x_lo), y_lo + t * (y_hi - y_lo));
        let (x, y) = (sx(xv), sy(yv));
        let _ = writeln!(
            s,
            r##"<line x1="{x:.2}" y1="{:.2}" x2="{x:.2}" y2="{:.2}" stroke="#333"/><text x="{x:.2}" y="{:.2}" text-anchor="middle">{xv:.3}</text>"##,
            TOP + plot_h,
            TOP + plot_h + 5.0,
            TOP + plot_h + 19.0
        );
        let _ = writeln!(
            s,
            r##"<line x1="{:.2}" y1="{y:.2}" x2="{LEFT}" y2="{y:.2}" stroke="#333"/><text x="{:.2}" y="{:.2}" text-anchor="end">{yv:.1}</text>"##,
            LEFT - 5.0,
            LEFT - 8.0,
            y + 4.0
        );
    }
    let _ = writeln!(
        s,
        r#"<text x="{:.1}" y="{:.1}" text-anchor="middle">CAI</text>"#,
        LEFT + plot_w / 2.0,
        HEIGHT - 16.0
    );
    let _ = writeln!(
        s,
        r#"<text x="18" y="{:.1}" text-anchor="middle" transform="rotate(-90 18 {:.1})">MFE (kcal/mol)</text>"#,
        TOP + plot_h / 2.0,
        TOP + plot_h / 2.0
    );
    for p in points {
        let (x, y) = (cai(p), mfe(p));
        if x.is_finite() && y.is_finite() {
            let _ = writeln!(s, "{}", mark(p, sx(x), sy(y)));
        }
    }
    legend(&mut s, WIDTH - RIGHT + 16.0, TOP + 8.0);
    s.push_str("</svg>\n");
    s
}

/// The key to the marks, beside the frame.
fn legend(s: &mut String, x: f64, y: f64) {
    let rows: [(&str, &str, &str, bool); 4] = [
        ("modular", INK, INK, false),
        ("not modular", "none", INK, false),
        ("not checked", "none", GREY, false),
        ("on the hull", "none", "#333", true),
    ];
    for (i, (label, fill, stroke, square)) in rows.iter().enumerate() {
        let cy = y + 20.0 * i as f64;
        let style = format!(r#"fill="{fill}" stroke="{stroke}" stroke-width="1.5""#);
        let shape = if *square {
            format!(
                r#"<rect class="legend" x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" {style}/>"#,
                x - MARK,
                cy - MARK,
                2.0 * MARK,
                2.0 * MARK
            )
        } else {
            format!(r#"<circle class="legend" cx="{x:.1}" cy="{cy:.1}" r="{MARK}" {style}/>"#)
        };
        let _ = writeln!(
            s,
            r#"{shape}<text x="{:.1}" y="{:.1}">{label}</text>"#,
            x + 12.0,
            cy + 4.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn each_point_is_one_mark_drawn_by_its_verdict_and_hull() {
        let points = [
            json!({"mfe": -3.0, "cai": 0.7, "on_hull": true, "modular": true}),
            json!({"mfe": -2.0, "cai": 0.8, "on_hull": false, "modular": false}),
            json!({"mfe": -1.0, "cai": 0.9, "on_hull": false}),
        ];
        let refs: Vec<&Value> = points.iter().collect();
        let svg = svg("A&B <x>", &refs);
        assert_eq!(svg.matches(r#"class="mark""#).count(), 3);
        assert_eq!(svg.matches(r#"<rect class="mark""#).count(), 1);
        assert!(svg.contains("A&amp;B &lt;x&gt;"), "{svg}");
        assert!(svg.contains("MFE (kcal/mol)") && svg.contains(">CAI<"));
    }

    #[test]
    fn an_empty_selection_is_still_a_frame() {
        let svg = svg("p", &[]);
        assert_eq!(svg.matches(r#"class="mark""#).count(), 0);
        assert!(svg.ends_with("</svg>\n"));
    }
}
