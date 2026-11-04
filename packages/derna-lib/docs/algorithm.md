# derna-lib — algorithm

This is the full technical treatment of DERNA's beam-search codon-design
solver: the implementable specification of the codon-DFA × CKY scan.

The reference implementation is LinearDesign (Zhang et al. 2023),
[LinearDesignSoftware/LinearDesign](https://github.com/LinearDesignSoftware/LinearDesign)
at tag `v1.0.0`. Source citations below point at files in that release.

## 1. The idea in one paragraph

The exact codon DP lifts a Zuker interval DP to carry codon identity at
each endpoint, giving `(a, b, s, t, x, y)` cells and cubic cost. Beam
search instead borrows LinearFold's linear-time machinery wholesale and
applies it not to a fixed RNA sequence but to a **graph of all RNA
sequences encoding the protein**. That graph — the codon DFA — is small:
each amino acid contributes a constant-bounded sub-lattice of its
synonymous codons. Running LinearFold's 5′→3′ CKY scan over the DFA's
nodes (instead of over sequence positions) folds and chooses codons
simultaneously. Codon adaptation enters as additive weights on the DFA's
edges, so the same scan optimizes the joint MFE + CAI objective.

## 2. Inputs and objective

Input: a protein `w = w₁…w_m` and a species codon-frequency table
`g : codons → [0,1]`. Output: an RNA `v ∈ {A,U,G,C}^{3m}` encoding `w`
and a secondary structure `P`, minimizing the fixed-λ
BRD objective

```
BRD_λ(v, P) = λ · MFE(v, P) − (1 − λ) · CAI̅(v, w)
```

where `CAI̅(v, w) = Σ_a log( g(v[3a−2 .. 3a]) / max_{x ∈ S(w_a)} g(x) )`
is the additive log-CAI and `S(w_a)` is the synonym set of `w_a`. Lower
is better. λ is fixed per solve. This is the same objective
the interval DP minimizes; the beam differs only in *how* it searches.

> LinearDesign parameterizes differently — it maximizes
> `−MFE + λ_LD · log CAI` with MFE weight fixed at 1 — and scales λ by 100
> internally (`linear_design.cpp`). Translating between λ and λ_LD so the
> two tools are comparable at "matched scoring" is a calibration step, not
> a structural one, and is handled when the comparison harness is built.

## 3. The codon DFA

### 3.1 Per-amino-acid lattice

A "coding wheel" lists, for each amino acid, its synonymous codons grouped
by shared nucleotide prefixes (LinearDesign's `coding_wheel.txt`;
e.g. `Leu → {CUG, CUC, CUU, CUA}, {UUG, UUA}`). Each amino acid compiles
to a small lattice with four position layers:

- node layer 0: the codon-unit entry (one node),
- layer 1: distinct first nucleotides,
- layer 2: distinct (first, second) prefixes,
- layer 3: the codon-unit exit (shared with the next codon's layer 0).

Edges are labeled by the nucleotide consumed. Codons that share a prefix
share layer-1 / layer-2 nodes, so the lattice has far fewer than
`3 × |S(w_a)|` edges. A node is `(position, lane)`: `lane` (the `NumType`
in `network.h`) distinguishes parallel paths within a layer.

### 3.2 Splicing into the protein DFA

`get_dfa` (`network.h`) concatenates the per-AA lattices: amino acid `i`
occupies nucleotide positions `3i .. 3i+3`, and its layer-3 exit node is
amino acid `i+1`'s layer-0 entry. The result is a DFA over positions
`0 .. 3m`. Every directed path from `(0,0)` to `(3m,0)` spells exactly one
RNA sequence encoding `w`, and every such sequence is exactly one path.
The DFA has `O(m)` nodes and `O(m)` edges (constant per codon).

### 3.3 CAI as edge weights

CAI is additive over codons (§2), so each codon's term distributes across
its three DFA edges. The reference assigns edge weights by a cost-to-go
normalization (`read_wheel_with_weights_log`): the weight of an edge is
`λ-scaled log(best-reachable-through-this-edge / best-reachable-from-source-node)`.
Summed along any complete codon path these telescope to the codon's true
`log(g(codon)/g_max)`; summed along a *partial* path they give that
prefix's best achievable CAI, so partial states ending at different nodes
in the same scan column remain comparable for pruning. The maximum-frequency
synonym contributes 0; every other codon contributes a negative penalty.

The DERNA implementation can either replicate this edge normalization or
carry the raw per-codon log-CAI and credit it at the codon's closing edge
(layer-2 → layer-3 transition), matching the interval DP's "credit the CAI
term once at the codon boundary crossing." The two agree on complete paths;
they differ only in how partial credit is distributed, which affects
finite-`b` pruning quality, not `b → ∞` correctness. The closing-edge
crediting is simpler and is the recommended starting point; adopt the
cost-to-go normalization only if pruning quality demands it.

## 4. The CKY scan over DFA nodes

### 4.1 Tables

LinearFold's six tables (see `rna-linearfold/src/state.rs`) are re-keyed from sequence positions to
DFA nodes. Writing `i`, `j` for DFA nodes:

| Table | Meaning | DERNA interval-DP analogue |
|---|---|---|
| `C[j]` | best exterior parse of the prefix ending at node `j` | `O` (exterior) |
| `H[i,j]` | hairpin candidate spanning `i..j` | the `E_h` motif feeding `E` |
| `P[i,j]` | best parse with `(i,j)` a closed pair | `E` (closed pair) |
| `Multi[i,j]` | multi-loop in progress (M2 core + unpaired flanks) | part of `M` |
| `M[i,j]` | multi-branch component, ≥1 branch | `N` (one arm) |
| `M2[i,j]` | multi-branch component, ≥2 branches | `M` bifurcation |

Each entry holds a `State`: accumulated MFE (`score`, integer dekacalories
from `rna-mfe`), accumulated CAI (`cai_score`), and a traceback parent
(`pre_node` + the manner that produced it). Equivalent derivations
reaching the same `(table, i, j)` merge, keeping the best by combined BRD
score — LinearFold's state-packing, now over DFA nodes.

The closing-pair tables (`H`, `P`, `Multi`) additionally key on the
**boundary nucleotides** of the pair (`NodeNucpair` in
`beam_cky_parser.h`): the same node pair `(i,j)` can close with different
nucleotide identities at `i` and `j`, and those identities set the Turner
loop energy. This is the DFA analogue of the interval DP's `(x, y)` codon
choice at the endpoints — but recorded as the two boundary nucleotides,
not the full codons: a flank codon that only affects the loop energy (through
its boundary nucleotide) and the decomposable CAI is recovered at traceback
as the best synonym for that nucleotide.

### 4.2 Transitions

The scan runs `j` over DFA nodes in position order. At each `j`, the
manners from `rna-linearfold/src/state.rs` apply, now traversing DFA edges
rather than incrementing a position. Let `e : a → b` be a DFA edge
labeled with nucleotide `nuc` and carrying CAI weight `w(e)`.

- **skip / extend-unpaired.** `C[b] ← C[a]` and `M[b] ← M[a]`, adding
  `w(e)` to `cai_score` and 0 to `score` (dangles=0, exterior/multi
  unpaired score ≡ 0; the codon DPs run at dangles = 0 only, as DERNA does). This consumes one nucleotide along the
  exterior or a multi-loop arm.
- **push / open a pair.** Begin a new `H` candidate at a node, recording
  the opening boundary nucleotide.
- **pop / close a pair (`P`).** A hairpin/stack/bulge/internal/multi
  closure fires, paying the loop free energy (via `rna-mfe`) computed from
  the boundary nucleotides, then combining the inner state with its outer
  context:
  - **hairpin close** `P[i,j] ← H[i,j]` + `hairpin_energy`.
  - **stack** `P[i,j] ← P[i′,j′]` + `stack_energy`, where `i′,j′` are the
    DFA nodes one edge inward on each side.
  - **single (bulge / internal)** `P[i,j] ← P[i′,j′]` + `bulge_energy` or
    `internal_energy`, with the unpaired flanks bounded by `SINGLE_MAX_LEN`
    (the MAXLOOP analogue; `beam_cky_parser.cc`).
  - **multi close** `P[i,j] ← Multi[i,j]` + multi-loop closing cost.
- **multi assembly.** `M[i,j] ← P[i,j]` (first branch, + branch cost);
  `M[b] ← M[a]` (unpaired extension); `M2[i,j] ← M[i,k] + P[k,j]`
  (bifurcation at an interior node `k`); `M[i,j] ← M2[i,j]`;
  `Multi[i,j] ← M2[i′,j′]` + unpaired flanks.
- **exterior close.** `C[j] ← C[i⁻] + P[i,j]` + terminal-AU term, where
  `i⁻` is the node immediately 5′ of the pair's open. The empty-prefix base case
  seeds `C[(0,0)] = 0`.

Each pop combines two stored states (inner + outer context), which is the
`O(b²)`-per-column step; LinearFold's k-best trick reduces a naive scan to
`O(b log b)` and is an optimization, not a correctness concern.

Pairing is gated by Watson–Crick / wobble compatibility of the two
boundary nucleotides and by a minimum hairpin span (`TURN`). Because the
DFA fixes which nucleotides are even reachable at each node, infeasible
codon/nucleotide combinations never enter the tables.

### 4.3 Broken codons

A base pair can split a codon: the pair's open or close can land at DFA
layer 1 or 2, mid-codon. The CAI of a codon straddling a structural
boundary must still be credited exactly once and exactly right. The
reference precomputes per-codon-unit best paths between any two interior
nodes (`best_path_in_one_codon_unit`) and a `get_broken_codon_score` map
keyed by node pairs, consulted whenever a transition spans a partial
codon. This bookkeeping is the one piece with no interval-DP analogue and
is the most error-prone part of the port; the reference's maps are the
shape to mirror.

## 5. Beam pruning

After all transitions at scan column `j` are applied, rank the live states
by combined BRD score and keep the top `b`; discard the rest permanently.
With constant `b` the per-column cost is bounded and the whole scan is
near-linear in `m`. Pruning is applied per table (`C`, `P`, `M`, `M2`,
`Multi`) as in LinearFold. Quickselect gives an `O(states)` partition
without a full sort.

`b → ∞` (or `b` ≥ the largest column population) disables pruning and runs
the exact lattice — which is what the released LinearDesign does by
default. The released tool ships *no* pruning on its scan path; `b` is
DERNA's addition for headroom at the longest lengths.

## 6. Traceback

From `C[(3m, 0)]` walk parents backward, emitting the nucleotide on each
consumed DFA edge and a dot-bracket symbol per manner (open `(`, close
`)`, unpaired `.`), exactly as `rna-linearfold/src/traceback.rs` does, with
the manner set extended for the multi-loop and DFA-edge cases. The walk
yields the RNA sequence and its structure together. A final check that the
sequence translates back to `w` (LinearDesign's `FINAL_CHECK`) is a cheap
invariant worth keeping.

## 7. Correctness contract

1. **Feasibility (any `b`).** Every DFA path encodes `w`, and pairing gates
   enforce valid non-crossing structure, so traceback always returns a
   legal `(v, P)`.
2. **Monotonicity in `b`.** A larger beam is a superset of a smaller one's
   survivors at every column, so the best returned score is non-increasing
   in `b`.
3. **Exactness at `b → ∞`.** The unpruned scan is a complete CKY parse of
   the codon DFA: it considers every (path, parse-tree) = (codon
   assignment, structure) pair. Its optimum therefore equals the interval
   DP's optimum **provided the two share the energy model and λ
   convention.** Both are dangles=0 Turner 2004; any residual disagreement
   at tractable lengths localizes a grammar/energy discrepancy between the
   LinearFold-style folding and the Zuker interval DP rather than hiding
   silently.
4. **Upper bound at finite `b`.** The returned score is ≥ the exact optimum
   (minimization), and the gap is attributable to a specific column where
   the optimal state was pruned — the diagnostic the backtrack-from-beam
   recovery-rate report measures.

The verifiable anchors are therefore: feasibility on every output;
monotone improvement as `b` grows; and equality with the interval DP and
the brute-force fixtures at `b → ∞` on tractable proteins (e.g. GLPE_HUMAN).

## 8. Complexity

| Quantity | Cost |
|---|---|
| DFA nodes / edges | `O(m)` |
| States per column | ≤ `b` after pruning |
| Per-column work | `O(b²)` naive pop combination, `O(b log b)` with k-best |
| Whole solve, pruned | `O(m · b log b)` ≈ linear in `m` |
| Whole solve, `b → ∞` | `O(m³)` worst case, fast in practice (compact DFA, sparse grammar) |

Space is `O(m · b)` pruned. This is the LinearFold profile, and it is the
regime in which LinearDesign is competitive at ≥3000 nt where the interval
DP cannot reach.

## 9. Relationship to the existing crates

- **`rna-mfe`** — per-loop Turner energies (hairpin, stack, bulge,
  internal, multi). Consumed exactly as `rna-linearfold` consumes it; the
  beam adds no new energy code.
- **`rna-linearfold`** — the structural template. Its `state.rs` manners,
  `dp.rs` scan, and `traceback.rs` walk are the blueprint; the beam
  generalizes node identity from a sequence position to a DFA node and adds
  the CAI accumulator. A λ=1 (pure-MFE) beam over a single-path DFA must
  reproduce `rna_linearfold::fold` — a cheap parity regression.
- **`derna-codon-dp`** — the codon-usage table loader (`CodonUsage`),
  CAI utilities (`cai`), the `brd_value` scalarization, and `Plan` /
  `Params` reuse. It also provides the exact-optimum oracle for §7.3.

`derna-lib` depends on all three. It does **not** reuse the interval DP's
recurrence/candidate-generator code: the traversal is LinearFold's, not the
interval DP's. The beam is a separate substrate, and the interval DP stays
the exact-optimum oracle it is validated against.

## 10. Suggested module layout (for the reference implementation)

A starting decomposition, not a contract:

- `dfa.rs` — codon DFA: per-AA lattice compilation from the coding wheel,
  splicing into the protein DFA, edge CAI weights.
- `state.rs` — `State` (MFE + CAI accumulators + traceback parent),
  node and node-nucpair keys, the manner enum (extended from
  `rna-linearfold`).
- `tables.rs` — the six node-keyed CKY tables and per-column beams.
- `scan.rs` — the 5′→3′ transition driver (§4.2), per-table pruning (§5).
- `broken_codon.rs` — partial-codon CAI bookkeeping (§4.3).
- `traceback.rs` — backward walk to `(sequence, structure)` (§6).
- `lib.rs` — `design(params, protein, codon_usage, lambda, beam_width)
  -> Result<Design, BeamError>` with `Design { sequence, structure, mfe,
  cai }`.

The public entry point should mirror `rna_linearfold::fold`'s shape — a
single function returning the chosen sequence, its structure, and its
(MFE, CAI) — so the comparison harness can call all solvers uniformly.
