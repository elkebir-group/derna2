# @derna/codon-dp

Rust port of the DERNA BRD / PORD codon-design dynamic programs. Composes
against [`rna-mfe`](../rna-mfe/) for per-loop Turner 2004 energies.

The DP runs at `dangles = 0` only, which is the published DERNA C++'s mode,
and is checked at fixed λ against that implementation's output.
