//! MFE DP fold — sequence → MFE structure.
//!
//! Composes against [`crate::loops`] for per-loop energy lookups and
//! returns the `dangles=0` MFE structure and energy. The DP shape and
//! recurrence notes live in [`fill`] (V leaf cases + W + F) and
//! [`multi`] (M / M1 cells + V multi-loop closing); the table types
//! live in [`tables`].

pub mod fill;
mod multi;
pub mod outside;
pub mod sparse;
pub mod tables;

pub use fill::{fill_dp_tables, fill_dp_tables_from, traceback};
pub use multi::zip_add_min_inf;
pub use outside::{forced_pair_energies, ForcedPairs};
pub use sparse::{
    cut_work, fold_sparse, fold_sparse_reusing, fold_work, kept_pairs, Cut, FoldTables, FoldWork,
    Reuse, SparseFold,
};
pub use tables::{DpTables, TriangleTable, MAX_LOOP_SIZE, MIN_HAIRPIN_LOOP};
