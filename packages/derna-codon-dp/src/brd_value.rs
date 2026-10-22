//! BRD scalar arithmetic.
//!
//! `brd_value(λ, mfe, cai) = λ·mfe − (1−λ)·cai`. MFE is a dekacal integer
//! (100× kcal/mol) per [`rna_mfe::Energy`], which is the resolution the Turner
//! parameter set is tabulated at. The CAI term is a real number: log-CAI
//! carries no comparable quantum, so scaling it to dekacal equivalents is a
//! change of units, not a change of resolution.
//!
//! **The combination is not rounded.** A recurrence stores this value in its
//! table and then minimizes over it, so rounding here would not merely blur a
//! reported score — it would decide which candidate wins. Rounding each stored
//! value and summing the results is not the same as summing and rounding once:
//! the error accumulates along whichever way the structure happens to be
//! decomposed, so the quantity minimized would depend on the parse rather than
//! on the molecule, and no formula over the emitted sequence would describe it.
//! Keeping the combination real makes the objective a function of the design
//! alone, which is what lets an optimum be stated at all.

/// `λ·mfe − (1−λ)·cai`, the objective the codon DP minimizes.
///
/// `cai_dekacal` is non-positive (log-CAI is non-positive by construction);
/// `mfe_dekacal` is typically negative for stable structures.
pub fn brd_value(lambda_: f64, mfe_dekacal: i32, cai_dekacal: f64) -> f64 {
    lambda_ * f64::from(mfe_dekacal) - (1.0 - lambda_) * cai_dekacal
}
