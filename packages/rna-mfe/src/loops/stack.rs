//! Stack pair: two adjacent base pairs with no unpaired nucleotides
//! between them. The lone Turner 2004 contribution is the 7×7
//! `stack[outer, inner_reversed]` cell.

use super::Energy;
use crate::params::{BasePair, Params};

/// A single base-pair-on-base-pair stack. Both pairs are stored in
/// sequence order (5'-side nucleotide first); the inner pair is
/// reversed inside [`stack_energy`] before consulting the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackPair {
    pub outer: BasePair,
    pub inner: BasePair,
}

/// Stack energy at `dangles=0`. The inner pair enters the lookup in
/// reversed orientation because the table indexes it from the loop
/// side, not from sequence.
pub fn stack_energy(params: &Params, s: &StackPair) -> Energy {
    Energy(params.stack.get(s.outer, s.inner.reversed()))
}
