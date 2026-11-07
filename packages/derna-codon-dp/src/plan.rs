//! Per-solve context: protein indices, codon lists, CAI tables, DP tables,
//! and traceback accumulators.
//!
//! [`Plan`] bundles every read-only quantity the recurrences need plus the
//! [`Tables`] (mutable `f64` storage) and [`Trace`] (traceback
//! accumulators). Construct once via [`Plan::new`], then hand to the fill
//! driver. Internal scaffolding — not part of the public API.
//!
//! Each recurrence is a candidate generator plus a reduction, so the cell
//! shape is `Candidate { value: f64, trace: ... }`; the logical table shape
//! is `[n_aa, n_aa, 3, 3, max_codons, max_codons]`, the paper's
//! `(a, b, s, t, x, y)` indexing. Production storage can pack that logical
//! shape more tightly as long as dense exports remain available for parity
//! tests.

use rna_mfe::loops::Nucleotide;
use rna_mfe::Params;

use crate::cai::{compute_cai_tables, CaiTables};
use crate::codon_table::{aa_to_index, AminoAcidError, Codon, AMINO_ACIDS};
use crate::codon_usage::CodonUsage;

/// Sentinel for "no candidate filled this cell." Finite rather than
/// [`f64::INFINITY`] so that an unfilled cell entering an arithmetic
/// combination yields a large number rather than a `NaN` that would
/// silently lose every comparison it takes part in.
pub const INF: f64 = 1.0e9;

/// Multiplier placing log-CAI on the same scale as dekacal energies, so that
/// one λ weighs comparable magnitudes. It is a change of units and nothing is
/// rounded to it; leaving log-CAI unscaled gives the same objective under the
/// reparametrization `λ' = λ / (100 − 99λ)`.
pub const CAI_DEKACAL_SCALE: f64 = 100.0;

/// `TURN`: minimum hairpin loop size. Lifted from
/// [`rna_mfe::dp::tables::MIN_HAIRPIN_LOOP`]; aliased here so the codon
/// DP recurrences can read it without naming the substrate's `dp` module.
pub const TURN: usize = rna_mfe::dp::tables::MIN_HAIRPIN_LOOP;

/// `MAXLOOP`: largest interior-loop side count the size table
/// covers. Lifted from [`rna_mfe::dp::tables::MAX_LOOP_SIZE`].
pub const MAXLOOP: usize = rna_mfe::dp::tables::MAX_LOOP_SIZE;

/// Packed cell index `(a, b, s, t, x, y)` — the natural unit of work.
pub type CellIdx = (usize, usize, usize, usize, usize, usize);

/// Storage policy for the motif-specific `E` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EStorageMode {
    /// Retain `E_h`, `E_s`, `E_b`, `E_i`, and `E_m` as full dense value
    /// tables. Used by parity snapshot tests that compare every cell.
    FullMotifTables,
    /// Retain only `E` plus a compact per-cell winner kind. Traceback
    /// re-runs the selected motif generator against the `E` value.
    CompactMotifWinners,
}

/// Physical storage layout for DP tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TableLayout {
    /// Store the full logical `[n, n, 3, 3, max_codons, max_codons]` shape.
    /// This mode is retained for the reference table snapshots.
    DenseMaxRectangles,
    /// Store only `a <= b` endpoint pairs, and store each endpoint pair's
    /// actual `len(codons[a]) * len(codons[b])` codon rectangle.
    PackedUpperTriangle,
}

/// Motif kind that won the `E` reduction for one cell.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EMotifKind {
    Hairpin,
    Stack,
    Bulge,
    Internal,
    MultiLoop,
}

impl EMotifKind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Hairpin => "E_h",
            Self::Stack => "E_s",
            Self::Bulge => "E_b",
            Self::Internal => "E_i",
            Self::MultiLoop => "E_m",
        }
    }
}

/// Flat DP table storage for the logical cell index `(a, b, s, t, x, y)`.
///
/// Tables `m`, `n`, and `o` are filled by the multi-loop and exterior
/// recurrences. In full motif-table mode, `e_m` is also retained as a
/// dense table for snapshot parity; in compact mode, `e_winner` records
/// which motif produced each reduced `E` value.
#[derive(Debug)]
pub(crate) struct Tables {
    pub e: Vec<f64>,
    pub e_winner: Vec<EMotifKind>,
    pub e_h: Vec<f64>,
    pub e_s: Vec<f64>,
    pub e_b: Vec<f64>,
    pub e_i: Vec<f64>,
    pub e_m: Vec<f64>,
    pub m: Vec<f64>,
    pub n: Vec<f64>,
    pub o: Vec<f64>,
    e_storage_mode: EStorageMode,
    layout: TableLayout,
    codon_counts: Vec<usize>,
    pair_offsets: Vec<usize>,
    pair_rect_sizes: Vec<usize>,
    pair_y_strides: Vec<usize>,
    n_aa: usize,
    max_codons: usize,
}

impl Tables {
    fn new(codon_counts: Vec<usize>, max_codons: usize, inputs: TableInputs) -> Self {
        let n_aa = codon_counts.len();
        let layout_info = LayoutInfo::new(&codon_counts, max_codons, inputs.layout);
        let total = layout_info.total_len;
        let inf_vec = || vec![INF; total];
        let motif_vec = || match inputs.e_storage_mode {
            EStorageMode::FullMotifTables => inf_vec(),
            EStorageMode::CompactMotifWinners => Vec::new(),
        };
        Self {
            e: inf_vec(),
            e_winner: match inputs.e_storage_mode {
                EStorageMode::FullMotifTables => Vec::new(),
                EStorageMode::CompactMotifWinners => vec![EMotifKind::Hairpin; total],
            },
            e_h: motif_vec(),
            e_s: motif_vec(),
            e_b: motif_vec(),
            e_i: motif_vec(),
            e_m: motif_vec(),
            m: inf_vec(),
            n: inf_vec(),
            o: inf_vec(),
            e_storage_mode: inputs.e_storage_mode,
            layout: inputs.layout,
            n_aa,
            codon_counts,
            pair_offsets: layout_info.pair_offsets,
            pair_rect_sizes: layout_info.pair_rect_sizes,
            pair_y_strides: layout_info.pair_y_strides,
            max_codons,
        }
    }

    #[inline]
    pub(crate) fn e_storage_mode(&self) -> EStorageMode {
        self.e_storage_mode
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn layout(&self) -> TableLayout {
        self.layout
    }

    #[inline]
    pub(crate) fn flat_index(&self, idx: CellIdx) -> usize {
        match self.layout {
            TableLayout::DenseMaxRectangles => self.dense_flat_index(idx),
            TableLayout::PackedUpperTriangle => self.packed_flat_index(idx),
        }
    }

    #[inline]
    fn dense_flat_index(&self, idx: CellIdx) -> usize {
        let (a, b, s, t, x, y) = idx;
        let mc = self.max_codons;
        ((((a * self.n_aa + b) * 3 + s) * 3 + t) * mc + x) * mc + y
    }

    #[inline]
    fn packed_flat_index(&self, idx: CellIdx) -> usize {
        let (a, b, s, t, x, y) = idx;
        debug_assert!(a <= b, "packed table indexed below diagonal: {idx:?}");
        debug_assert!(x < self.codon_counts[a], "x out of range for {idx:?}");
        debug_assert!(y < self.codon_counts[b], "y out of range for {idx:?}");
        let pair = a * self.n_aa + b;
        let base = self.pair_offsets[pair];
        debug_assert_ne!(base, usize::MAX, "missing packed offset for {idx:?}");
        base + ((s * 3 + t) * self.pair_rect_sizes[pair]) + (x * self.pair_y_strides[pair]) + y
    }

    pub(crate) fn dense_values(&self, table: &[f64]) -> Vec<f64> {
        if table.is_empty() {
            return Vec::new();
        }
        match self.layout {
            TableLayout::DenseMaxRectangles => table.to_vec(),
            TableLayout::PackedUpperTriangle => self.expand_packed_values(table),
        }
    }

    fn expand_packed_values(&self, table: &[f64]) -> Vec<f64> {
        let mut dense = vec![INF; self.dense_len()];
        let mut export = DenseExport {
            tables: self,
            packed: table,
            dense: &mut dense,
        };
        for pair in self.upper_endpoint_pairs() {
            export.copy_pair(pair);
        }
        dense
    }

    fn upper_endpoint_pairs(&self) -> impl Iterator<Item = EndpointPair> + '_ {
        (0..self.n_aa).flat_map(|a| (a..self.n_aa).map(move |b| EndpointPair { a, b }))
    }

    fn dense_len(&self) -> usize {
        self.n_aa * self.n_aa * 3 * 3 * self.max_codons * self.max_codons
    }
}

#[derive(Debug, Clone, Copy)]
struct EndpointPair {
    a: usize,
    b: usize,
}

struct DenseExport<'t, 'd> {
    tables: &'t Tables,
    packed: &'t [f64],
    dense: &'d mut [f64],
}

impl DenseExport<'_, '_> {
    fn copy_pair(&mut self, pair: EndpointPair) {
        for st in 0..9 {
            self.copy_st_block(pair, st);
        }
    }

    fn copy_st_block(&mut self, pair: EndpointPair, st: usize) {
        for x in 0..self.tables.codon_counts[pair.a] {
            self.copy_codon_row(pair, st, x);
        }
    }

    fn copy_codon_row(&mut self, pair: EndpointPair, st: usize, x: usize) {
        let s = st / 3;
        let t = st % 3;
        for y in 0..self.tables.codon_counts[pair.b] {
            let idx = (pair.a, pair.b, s, t, x, y);
            self.dense[self.tables.dense_flat_index(idx)] =
                self.packed[self.tables.packed_flat_index(idx)];
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct TableInputs {
    e_storage_mode: EStorageMode,
    layout: TableLayout,
}

struct LayoutInfo {
    total_len: usize,
    pair_offsets: Vec<usize>,
    pair_rect_sizes: Vec<usize>,
    pair_y_strides: Vec<usize>,
}

impl LayoutInfo {
    fn new(codon_counts: &[usize], max_codons: usize, layout: TableLayout) -> Self {
        match layout {
            TableLayout::DenseMaxRectangles => Self {
                total_len: codon_counts.len()
                    * codon_counts.len()
                    * 3
                    * 3
                    * max_codons
                    * max_codons,
                pair_offsets: Vec::new(),
                pair_rect_sizes: Vec::new(),
                pair_y_strides: Vec::new(),
            },
            TableLayout::PackedUpperTriangle => packed_upper_triangle_offsets(codon_counts),
        }
    }
}

fn packed_upper_triangle_offsets(codon_counts: &[usize]) -> LayoutInfo {
    let n_aa = codon_counts.len();
    let mut pair_offsets = vec![usize::MAX; n_aa * n_aa];
    let mut pair_rect_sizes = vec![0usize; n_aa * n_aa];
    let mut pair_y_strides = vec![0usize; n_aa * n_aa];
    let mut total_len = 0usize;
    for a in 0..n_aa {
        for b in a..n_aa {
            let pair = a * n_aa + b;
            let rect = codon_counts[a] * codon_counts[b];
            pair_offsets[pair] = total_len;
            pair_rect_sizes[pair] = rect;
            pair_y_strides[pair] = codon_counts[b];
            total_len += 3 * 3 * rect;
        }
    }
    LayoutInfo {
        total_len,
        pair_offsets,
        pair_rect_sizes,
        pair_y_strides,
    }
}

/// Traceback accumulators. Empty at construction; the traceback walker
/// pushes `(nt_i, nt_j)` pairs and writes codon assignments per AA when
/// it lands. The fields ride along the [`Plan`] until then.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub(crate) struct Trace {
    pub pairs: Vec<(usize, usize)>,
    pub codon_assignments: Vec<usize>,
}

/// Per-solve context. Holds protein indices, post-mask codon lists,
/// per-position log-CAI in dekacal int, λ, the DP tables, the trace, and
/// a shared reference to the Turner 2004 [`Params`].
///
/// Some fields (`default_codon_idx`, `dangling_5_prime`, `trace`) are
/// populated at construction but read only by consumers that build on
/// the single-loop fill driver. `dangling_5_prime` flows through to the
/// eventual top-level solver, and the trace fills as the traceback
/// walker runs. They sit silenced here so the surface compiles clean
/// without re-shaping the plan when those consumers land.
#[allow(dead_code)]
pub(crate) struct Plan<'p> {
    pub aa_indices: Vec<usize>,
    pub codon_lists: Vec<Vec<Codon>>,
    pub default_codon_idx: Vec<usize>,
    /// Pre-encoded RNA at the per-position default codons. Mutated by
    /// candidate construction (splice in pinned codons → score → splice
    /// back). Length is `3 * n_aa`.
    pub default_seq: Vec<Nucleotide>,
    /// `log_cai_dekacal_per_pos[p][ci]` — `log_cai[aa_indices[p], full_ci] *
    /// 100` for the codon at post-mask index `ci`, unrounded: a weight the
    /// objective is minimized over, not a reported quantity.
    pub log_cai_dekacal_per_pos: Vec<Vec<f64>>,
    /// Per-position, per-boundary-nucleotide, per-codon-position table of
    /// the highest log-CAI on the dekacal scale among the synonyms whose
    /// nucleotide at the given codon position equals the given nt.
    /// Indexed by `[p][nt][position]` where `p ∈ 0..n_aa`, `nt ∈ 0..4`,
    /// `position ∈ 0..3`. Reads are valid only when the corresponding bit
    /// in [`Self::aa_boundary_nts`] is set; entries at unreachable
    /// `(nt, position)` slots hold `-INF` and the bitmask must be
    /// checked before reading.
    pub best_synonym_cai: Vec<[[f64; 3]; 4]>,
    /// Per-position, per-boundary-nucleotide, per-codon-position table of
    /// the codon index (into [`Self::codon_lists`] at the same position)
    /// matching the entry in [`Self::best_synonym_cai`]. Unreachable
    /// `(nt, position)` slots hold `usize::MAX`; callers must gate reads
    /// on [`Self::aa_boundary_nts`].
    pub best_synonym_idx: Vec<[[usize; 3]; 4]>,
    /// Per-position, per-codon-position reachability bitmask. Bit
    /// `nt ∈ 0..4` of `aa_boundary_nts[p][position]` is set iff some
    /// synonym at position `p` puts `nt` at codon offset `position`.
    /// The internal-loop recurrence uses this to gate boundary-nt
    /// enumeration so impossible nucleotides are skipped.
    pub aa_boundary_nts: Vec<[u8; 3]>,
    pub lambda_: f64,
    pub dangling_5_prime: bool,
    pub max_codons: usize,
    pub tables: Tables,
    pub trace: Trace,
    pub params: &'p Params,
    /// Multi-loop closing constant `A` in dekacal. Cached once at plan
    /// construction; sequence-independent so the multi-loop recurrences
    /// don't re-fetch it per candidate.
    pub ml_closing_const: i32,
    /// Multi-loop per-unpaired-nucleotide constant `B` in dekacal. Cached
    /// once at plan construction alongside [`Self::ml_closing_const`]; the N
    /// recurrence's unpaired-bump cases (2 / 3 / 4 / 5) pay this constant per
    /// step.
    pub ml_base_const: i32,
}

impl<'p> Plan<'p> {
    /// Construct a [`Plan`] for a given protein, codon-usage, solver
    /// inputs (`λ`, `dangling_5_prime`, optional codon mask), and shared
    /// [`Params`]. Returns `Err` for unknown amino-acid letters; an
    /// empty protein is rejected early.
    pub fn new(
        protein: &str,
        codon_usage: &CodonUsage,
        inputs: &PlanInputs,
        params: &'p Params,
    ) -> Result<Self, PlanError> {
        validate_protein(protein)?;
        validate_lambda(inputs.lambda_)?;
        let aa_indices = parse_aa_indices(protein)?;
        let codon_lists = resolve_codon_lists(&aa_indices, inputs.codon_mask.as_deref())?;
        let max_codons = codon_lists.iter().map(Vec::len).max().unwrap_or(1);
        let cai = compute_cai_tables(codon_usage);
        let log_cai_dekacal_per_pos = build_log_cai_per_pos(&aa_indices, &codon_lists, &cai);
        let default_codon_idx = default_codon_idx(&log_cai_dekacal_per_pos, &codon_lists);
        let default_seq = assemble_default_seq(&codon_lists, &default_codon_idx);
        let best_synonyms = build_best_synonym_tables(&codon_lists, &log_cai_dekacal_per_pos);
        let codon_counts = codon_lists.iter().map(Vec::len).collect();
        Ok(Self {
            aa_indices,
            codon_lists,
            default_codon_idx,
            default_seq,
            log_cai_dekacal_per_pos,
            best_synonym_cai: best_synonyms.cai,
            best_synonym_idx: best_synonyms.idx,
            aa_boundary_nts: best_synonyms.masks,
            lambda_: inputs.lambda_,
            dangling_5_prime: inputs.dangling_5_prime,
            max_codons,
            tables: Tables::new(
                codon_counts,
                max_codons,
                TableInputs {
                    e_storage_mode: inputs.e_storage_mode,
                    layout: inputs.table_layout,
                },
            ),
            trace: Trace::default(),
            params,
            ml_closing_const: params.ml.closing_37,
            ml_base_const: params.ml.base_37,
        })
    }

    #[inline]
    pub fn n_aa(&self) -> usize {
        self.aa_indices.len()
    }
}

/// Internal-only solver inputs bundled into a single struct so
/// [`Plan::new`] stays at four arguments. Public-facing options live on
/// [`crate::BrdOptions`]; `λ` rides in because the plan is built per-λ.
#[derive(Debug, Clone)]
pub(crate) struct PlanInputs {
    pub dangling_5_prime: bool,
    pub codon_mask: Option<Vec<Vec<usize>>>,
    pub lambda_: f64,
    pub e_storage_mode: EStorageMode,
    pub table_layout: TableLayout,
}

fn validate_protein(protein: &str) -> Result<(), PlanError> {
    if protein.is_empty() {
        Err(PlanError::EmptyProtein)
    } else {
        Ok(())
    }
}

fn validate_lambda(lambda_: f64) -> Result<(), PlanError> {
    if (0.0..=1.0).contains(&lambda_) {
        Ok(())
    } else {
        Err(PlanError::LambdaOutOfRange(lambda_))
    }
}

/// `nt_position(p, k) = 3·p + k` — the paper's σ function.
#[inline]
pub fn nt_position(p: usize, k: usize) -> usize {
    3 * p + k
}

/// Inverse of [`nt_position`]: protein index containing nucleotide `nt`.
#[inline]
pub fn aa_of_nt(nt: usize) -> usize {
    nt / 3
}

/// Codon-index iterator at AA `p` constrained by already-pinned `(aa, ci)`
/// pairs. If `p` is in `constraints`, yields exactly the pinned `ci`
/// (forced-assignment); otherwise yields every `ci` in `0..n_codons(p)`.
///
/// The first matching constraint short-circuits the result.
pub(crate) fn consistent_choices(
    plan: &Plan<'_>,
    p: usize,
    constraints: &[(usize, usize)],
) -> ConsistentIter {
    let pinned: Option<usize> = constraints
        .iter()
        .find_map(|&(pp, ci)| (pp == p).then_some(ci));
    let n = plan.codon_lists[p].len();
    ConsistentIter { pinned, n, next: 0 }
}

pub(crate) struct ConsistentIter {
    pinned: Option<usize>,
    n: usize,
    next: usize,
}

impl Iterator for ConsistentIter {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        match self.pinned {
            Some(pin) => {
                if self.next == 0 {
                    self.next = 1;
                    if pin < self.n {
                        Some(pin)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            None => {
                if self.next < self.n {
                    let ci = self.next;
                    self.next += 1;
                    Some(ci)
                } else {
                    None
                }
            }
        }
    }
}

/// Build a candidate RNA from the per-position default codons, then splice
/// in pinned codon choices. Writes into `dst` (caller-owned scratch) so
/// repeated calls inside a hot loop avoid heap traffic.
///
/// On entry `dst` is overwritten from `plan.default_seq`; pinned positions
/// overwrite the corresponding three nucleotides.
pub(crate) fn assemble_rna_into(
    plan: &Plan<'_>,
    pinned: &[(usize, usize)],
    dst: &mut Vec<Nucleotide>,
) {
    dst.clear();
    dst.extend_from_slice(&plan.default_seq);
    for &(p, ci) in pinned {
        let codon = &plan.codon_lists[p][ci];
        let off = 3 * p;
        dst[off] = nuc_from_index(codon.nuc[0]);
        dst[off + 1] = nuc_from_index(codon.nuc[1]);
        dst[off + 2] = nuc_from_index(codon.nuc[2]);
    }
}

#[inline]
pub(crate) fn nuc_from_index(idx: u8) -> Nucleotide {
    match idx {
        0 => Nucleotide::A,
        1 => Nucleotide::C,
        2 => Nucleotide::G,
        3 => Nucleotide::U,
        other => panic!("invalid nucleotide index: {other}"),
    }
}

/// Raw nucleotide index (`A=0, C=1, G=2, U=3`) at codon position `pos` of
/// codon `ci` for AA `aa`. Lets the pair-only hot paths read closing /
/// enclosed nts straight from `codon_lists` without staging a full RNA in
/// `assemble_rna_into`'s scratch.
#[inline]
pub(crate) fn nuc_at(plan: &Plan<'_>, aa: usize, ci: usize, pos: usize) -> u8 {
    plan.codon_lists[aa][ci].nuc[pos]
}

// ---------------------------------------------------------------------------
// Plan construction helpers
// ---------------------------------------------------------------------------

fn parse_aa_indices(protein: &str) -> Result<Vec<usize>, PlanError> {
    protein
        .chars()
        .map(|c| aa_to_index(c).map_err(PlanError::from))
        .collect()
}

fn resolve_codon_lists(
    aa_indices: &[usize],
    codon_mask: Option<&[Vec<usize>]>,
) -> Result<Vec<Vec<Codon>>, PlanError> {
    if let Some(mask) = codon_mask {
        if mask.len() != aa_indices.len() {
            return Err(PlanError::CodonMaskLength {
                expected: aa_indices.len(),
                actual: mask.len(),
            });
        }
        let mut out = Vec::with_capacity(aa_indices.len());
        for (p, &aa) in aa_indices.iter().enumerate() {
            let position = resolve_position_codons(p, aa, &mask[p])?;
            out.push(position);
        }
        Ok(out)
    } else {
        Ok(aa_indices
            .iter()
            .map(|&aa| AMINO_ACIDS[aa].codons.to_vec())
            .collect())
    }
}

fn resolve_position_codons(
    p: usize,
    aa: usize,
    allowed_idx: &[usize],
) -> Result<Vec<Codon>, PlanError> {
    if allowed_idx.is_empty() {
        return Err(PlanError::EmptyMaskPosition(p));
    }
    let full = AMINO_ACIDS[aa].codons;
    let mut out = Vec::with_capacity(allowed_idx.len());
    for &ci in allowed_idx {
        if ci >= full.len() {
            return Err(PlanError::MaskIndexOutOfRange { p, aa, ci });
        }
        out.push(full[ci]);
    }
    Ok(out)
}

fn build_log_cai_per_pos(
    aa_indices: &[usize],
    codon_lists: &[Vec<Codon>],
    cai: &CaiTables,
) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(aa_indices.len());
    for (p, codons) in codon_lists.iter().enumerate() {
        let aa = aa_indices[p];
        let mut row = Vec::with_capacity(codons.len());
        for codon in codons {
            let full_ci = AMINO_ACIDS[aa]
                .codons
                .iter()
                .position(|c| c.nuc == codon.nuc)
                .expect("codon list entry is a synonym of its AA");
            let log_cai_val = cai.log_cai[aa][full_ci];
            row.push(log_cai_val * CAI_DEKACAL_SCALE);
        }
        out.push(row);
    }
    out
}

fn default_codon_idx(
    log_cai_dekacal_per_pos: &[Vec<f64>],
    codon_lists: &[Vec<Codon>],
) -> Vec<usize> {
    log_cai_dekacal_per_pos
        .iter()
        .zip(codon_lists.iter())
        .map(|(row, codons)| {
            (0..codons.len())
                .max_by(|&l, &r| {
                    row[l]
                        .partial_cmp(&row[r])
                        .expect("log-CAI weights are finite")
                })
                .expect("every codon list has at least one entry")
        })
        .collect()
}

/// Bundles the three nt-level boundary lookup tables consumed by the
/// internal-loop recurrence: per-position max-CAI synonym CAI value,
/// matching codon index, and a reachability bitmask.
struct BestSynonymTables {
    cai: Vec<[[f64; 3]; 4]>,
    idx: Vec<[[usize; 3]; 4]>,
    masks: Vec<[u8; 3]>,
}

/// Build the three nt-level boundary lookup tables. Ties on CAI resolve
/// to the lowest codon index so traceback recovers a deterministic codon
/// assignment.
fn build_best_synonym_tables(
    codon_lists: &[Vec<Codon>],
    log_cai_dekacal_per_pos: &[Vec<f64>],
) -> BestSynonymTables {
    let mut tables = BestSynonymTables {
        cai: Vec::with_capacity(codon_lists.len()),
        idx: Vec::with_capacity(codon_lists.len()),
        masks: Vec::with_capacity(codon_lists.len()),
    };
    for (p, codons) in codon_lists.iter().enumerate() {
        let rows = best_synonym_rows_for_position(codons, &log_cai_dekacal_per_pos[p]);
        tables.cai.push(rows.cai);
        tables.idx.push(rows.idx);
        tables.masks.push(rows.mask);
    }
    tables
}

/// Per-position rows of [`BestSynonymTables`]: max-CAI synonym at each
/// `(nt, codon_position)` slot for one protein position's codon list.
struct PositionRows {
    cai: [[f64; 3]; 4],
    idx: [[usize; 3]; 4],
    mask: [u8; 3],
}

fn best_synonym_rows_for_position(codons: &[Codon], log_cai_dekacal: &[f64]) -> PositionRows {
    let mut rows = PositionRows {
        cai: [[-INF; 3]; 4],
        idx: [[usize::MAX; 3]; 4],
        mask: [0u8; 3],
    };
    for (ci, codon) in codons.iter().enumerate() {
        update_rows_with_codon(&mut rows, codon, ci, log_cai_dekacal[ci]);
    }
    rows
}

#[inline]
fn update_rows_with_codon(rows: &mut PositionRows, codon: &Codon, ci: usize, cai: f64) {
    for pos in 0..3 {
        let nt = codon.nuc[pos] as usize;
        rows.mask[pos] |= 1 << nt;
        let is_first = rows.idx[nt][pos] == usize::MAX;
        if is_first || cai > rows.cai[nt][pos] {
            rows.cai[nt][pos] = cai;
            rows.idx[nt][pos] = ci;
        }
    }
}

fn assemble_default_seq(
    codon_lists: &[Vec<Codon>],
    default_codon_idx: &[usize],
) -> Vec<Nucleotide> {
    let mut out = Vec::with_capacity(3 * codon_lists.len());
    for (p, codons) in codon_lists.iter().enumerate() {
        let codon = &codons[default_codon_idx[p]];
        out.push(nuc_from_index(codon.nuc[0]));
        out.push(nuc_from_index(codon.nuc[1]));
        out.push(nuc_from_index(codon.nuc[2]));
    }
    out
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("protein must be non-empty")]
    EmptyProtein,
    #[error("lambda_ must be in [0, 1]; got {0}")]
    LambdaOutOfRange(f64),
    #[error(transparent)]
    AminoAcid(#[from] AminoAcidError),
    #[error("codon_mask length {actual} != protein length {expected}")]
    CodonMaskLength { expected: usize, actual: usize },
    #[error("codon_mask[{0}] is empty; every position needs >= 1 codon")]
    EmptyMaskPosition(usize),
    #[error("codon_mask[{p}] index {ci} out of range for AA {aa}")]
    MaskIndexOutOfRange { p: usize, aa: usize, ci: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codon_usage::CodonUsage;
    use rna_mfe::Params;

    fn make_plan(protein: &str) -> Plan<'static> {
        make_plan_with_layout(protein, TableLayout::DenseMaxRectangles)
    }

    fn make_plan_with_layout(protein: &str, table_layout: TableLayout) -> Plan<'static> {
        // Leak the params / codon-usage so the returned `Plan` can live as
        // `'static` for tests; harmless within the test process.
        let codon_usage: &'static CodonUsage =
            Box::leak(Box::new(CodonUsage::load_default().unwrap()));
        let params: &'static Params = Box::leak(Box::new(Params::load_default().unwrap()));
        let inputs = PlanInputs {
            dangling_5_prime: false,
            codon_mask: None,
            lambda_: 0.5,
            e_storage_mode: EStorageMode::FullMotifTables,
            table_layout,
        };
        Plan::new(protein, codon_usage, &inputs, params).unwrap()
    }

    /// One slot key in the round-trip test: protein position plus
    /// `(codon_position, boundary_nt)`. Bundles three usize-like values
    /// so the helper assertions stay at a manageable arity.
    #[derive(Debug, Clone, Copy)]
    struct SlotKey {
        p: usize,
        pos: usize,
        nt: u8,
    }

    fn assert_round_trip(plan: &Plan<'_>) {
        for (p, codons) in plan.codon_lists.iter().enumerate() {
            for pos in 0..3 {
                for nt in 0..4u8 {
                    assert_round_trip_slot(plan, codons, SlotKey { p, pos, nt });
                }
            }
        }
    }

    fn assert_round_trip_slot(plan: &Plan<'_>, codons: &[Codon], slot: SlotKey) {
        let SlotKey { p, pos, nt } = slot;
        let mask = plan.aa_boundary_nts[p][pos];
        let reachable = (mask >> nt) & 1 == 1;
        let idx = plan.best_synonym_idx[p][nt as usize][pos];
        if !reachable {
            assert_eq!(idx, usize::MAX, "unreachable slot {p}/{pos}/{nt}");
            return;
        }
        let codon_nt = codons[idx].nuc[pos];
        assert_eq!(codon_nt, nt, "idx round-trip {p}/{pos}/{nt}");
        let recorded = plan.best_synonym_cai[p][nt as usize][pos];
        let cai_at_idx = plan.log_cai_dekacal_per_pos[p][idx];
        assert_eq!(cai_at_idx, recorded, "cai mismatch at idx {p}/{pos}/{nt}");
        let expected_max = true_max_cai(plan, codons, slot);
        assert_eq!(expected_max, recorded, "cai not true max {p}/{pos}/{nt}");
    }

    fn true_max_cai(plan: &Plan<'_>, codons: &[Codon], slot: SlotKey) -> f64 {
        codons
            .iter()
            .enumerate()
            .filter(|(_, c)| c.nuc[slot.pos] == slot.nt)
            .map(|(ci, _)| plan.log_cai_dekacal_per_pos[slot.p][ci])
            .fold(f64::NEG_INFINITY, f64::max)
    }

    #[test]
    fn packed_layout_uses_upper_triangle_actual_rectangles() {
        let dense = make_plan_with_layout("GLAGPR", TableLayout::DenseMaxRectangles);
        let packed = make_plan_with_layout("GLAGPR", TableLayout::PackedUpperTriangle);
        let mut expected_packed_len = 0usize;
        for a in 0..packed.n_aa() {
            for b in a..packed.n_aa() {
                expected_packed_len +=
                    9 * packed.codon_lists[a].len() * packed.codon_lists[b].len();
            }
        }

        assert_eq!(dense.tables.layout(), TableLayout::DenseMaxRectangles);
        assert_eq!(packed.tables.layout(), TableLayout::PackedUpperTriangle);
        assert_eq!(packed.tables.e.len(), expected_packed_len);
        assert!(packed.tables.e.len() < dense.tables.e.len());
    }

    #[test]
    fn packed_layout_exports_dense_parity_shape() {
        let dense = make_plan_with_layout("GLAGPR", TableLayout::DenseMaxRectangles);
        let mut packed = make_plan_with_layout("GLAGPR", TableLayout::PackedUpperTriangle);
        let idx = (0, packed.n_aa() - 1, 0, 2, 0, 0);
        let packed_flat = packed.tables.flat_index(idx);
        let dense_flat = dense.tables.flat_index(idx);
        packed.tables.e[packed_flat] = 42.0;

        let exported = packed.tables.dense_values(&packed.tables.e);
        assert_eq!(exported.len(), dense.tables.e.len());
        assert_eq!(exported[dense_flat], 42.0);
    }

    #[test]
    fn best_synonym_round_trip_glagpr() {
        assert_round_trip(&make_plan("GLAGPR"));
    }

    #[test]
    fn best_synonym_round_trip_six_synonym_aas() {
        // Exercise the 6-synonym AAs: Arg (R), Leu (L), Ser (S).
        assert_round_trip(&make_plan("RLSARLS"));
    }
}
