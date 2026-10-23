//! Tokenizer for `rna_turner2004.par`.
//!
//! The file is line-oriented. Sections start with `# section_name`
//! (single `#`); lines starting with `##` are comments (the file header
//! is `## RNAfold parameter file v2.0`); per-line C-style `/* ... */`
//! comments may appear anywhere on a line. The terminator is `#END`.
//!
//! See ViennaRNA 2.7.0 `src/ViennaRNA/params/io.c`:
//! - `ignore_comment` (line 1141): excise `/* ... */` from a line.
//! - `gettype` / `settype` (≈ line 1735–1850): the `# foo` → enum map.

/// A single section as parsed from the file: the header name (e.g.
/// `"stack"`, `"int22"`), and the body lines (with `/* ... */`
/// comments stripped) in source order.
#[derive(Debug)]
pub struct Section<'a> {
    pub name: &'a str,
    /// Body lines, in source order, with C-style comments removed.
    /// String-typed because comment-stripping rewrites the slice.
    pub lines: Vec<String>,
}

/// Walk the file, splitting it into sections. The caller dispatches
/// each section to its parser based on `Section::name`.
pub fn split_sections(source: &str) -> Vec<Section<'_>> {
    let mut sections: Vec<Section<'_>> = Vec::new();
    let mut current: Option<Section<'_>> = None;

    for raw in source.lines() {
        if let Some(name) = section_header(raw) {
            if let Some(prev) = current.take() {
                sections.push(prev);
            }
            if name == "END" {
                break;
            }
            current = Some(Section {
                name,
                lines: Vec::new(),
            });
        } else if let Some(sec) = current.as_mut() {
            sec.lines.push(strip_comment(raw));
        }
    }

    if let Some(sec) = current {
        sections.push(sec);
    }
    sections
}

/// If `line` starts with a `#` section header, return its name; else
/// `None`. `##` doc-comment lines are not section headers.
fn section_header(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix('#')?;
    if rest.starts_with('#') {
        // `##` is a file-level comment, not a section header.
        return None;
    }
    let name = rest.trim();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// Remove a single `/* ... */` C-style comment if present. Mirrors
/// ViennaRNA's `ignore_comment` (only one comment per line, no
/// nesting, no multi-line spans).
fn strip_comment(line: &str) -> String {
    if let Some(start) = line.find("/*") {
        let after_open = &line[start + 2..];
        if let Some(end_rel) = after_open.find("*/") {
            let mut out = String::with_capacity(line.len());
            out.push_str(&line[..start]);
            out.push_str(&after_open[end_rel + 2..]);
            return out;
        }
    }
    line.to_string()
}

/// Yield whitespace-separated tokens across a section's body, in source
/// order. Comment-stripping has already been done by the time this is
/// called.
pub fn body_tokens<'a>(section: &'a Section<'a>) -> impl Iterator<Item = &'a str> {
    section.lines.iter().flat_map(|l| l.split_whitespace())
}
