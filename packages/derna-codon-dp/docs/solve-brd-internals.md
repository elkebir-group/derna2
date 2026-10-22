# Rust `solve_brd` internals — algorithm, cache, and perf attribution

This document is the source-of-truth reference for how
`derna-codon-dp`'s `solve_brd` works, what it currently costs, and where
the remaining per-solve gap to DERNA C++ lives.

It covers:

1. the current per-solve performance standing vs DERNA C++,
2. the algorithm pseudocode at a cache-relevant abstraction level,
3. complexity and access-pattern analysis under the current table
   layout, and
4. the ranked list of remaining opportunities to close the gap, with
   recently-closed ideas kept visible.

Source anchors:

- [`solve.rs`](../src/solve.rs): production `solve_brd` / `solve_pord`
  entry points and the default production storage mode.
- [`plan.rs`](../src/plan.rs): table modes, packed endpoint offsets,
  codon-list storage, CAI side tables, dense snapshot export, and the
  shared `nuc_at` / `nuc_from_index` readers used by every pair-only
  hot path.
- [`recurrences/mod.rs`](../src/recurrences/mod.rs): global fill order,
  `(s,t)` traversal orders, the cell-level outer-pair canonicity gate,
  compact `E` reduction, and full-motif snapshot reduction.
- [`recurrences/hairpin.rs`](../src/recurrences/hairpin.rs),
  [`stack.rs`](../src/recurrences/stack.rs),
  [`bulge.rs`](../src/recurrences/bulge.rs),
  [`internal.rs`](../src/recurrences/internal.rs),
  [`multi.rs`](../src/recurrences/multi.rs), and
  [`exterior.rs`](../src/recurrences/exterior.rs): recurrence-specific
  candidate axes.

Reader path: for the current gap to C++ and the live opportunities,
read sections 1 and 4. Section 2 is the source-faithful pseudocode
layer to audit when a later claim depends on a recurrence's exact
read/write pattern; section 3 is the supporting cache and access-
pattern analysis.

## 1. Problem Recap

For each scalar BRD solve at one `lambda`, the solver fills a codon-lifted RNA
DP over intervals of an amino-acid sequence. A logical DP cell is keyed by:

```text
idx = (a, b, s, t, x, y)
```

where:

- `(a,b)` are amino-acid endpoints,
- `(s,t)` are nucleotide offsets inside codons `a` and `b`, and
- `(x,y)` are codon choices at `a` and `b`.

The logical recurrence surface still has nine score tables:

```text
E, E_h, E_s, E_b, E_i, E_m, M, N, O
```

The physical production storage is smaller:

```text
E, M, N, O        i32 score tables (packed upper-triangle, actual codon rects)
E_winner          repr(u8) motif-winner side table
```

`E_h`, `E_s`, `E_b`, `E_i`, and `E_m` are not retained as production
value tables. Production solving computes each motif's candidate
reduction at the current cell, stores only the reduced `E` value plus
the winning motif kind, and lets traceback regenerate the selected
motif's candidate stream. Test-only parity snapshots still use the
full-motif dense layout so exported tables match the historical v1
fixture shape byte-for-byte.

### What's load-bearing for the perf gap

Correctness anchors remain stronger than the performance story:
per-cell parity against the prior reference holds, and the n=78 hull
comparison keeps the expected Rust-superset shape. The remaining
question is per-solve wall-clock vs DERNA C++ on the same hardware at
the same input — see § 1.5 for the current numbers and § 4 for the
ranked opportunities.

The analysis layer that matters for closing the gap is:

- production vs snapshot storage and the implied L-cache regimes,
- packed endpoint/codon-rectangle layout and index arithmetic,
- loop order and `(s,t)` traversal,
- stride and reuse distance for the hottest table reads,
- candidate-side allocation streams (a Rust-specific cost, since removed
  by streaming candidates; see § 1.5), and
- recurrence axes that cause distant table reads.

It intentionally omits error handling, traceback reconstruction
details, and Turner-energy internals unless they change the access
pattern.

## 1.5 Current Standing vs DERNA C++

Per-solve median wall-clock at GLPE-78 `lambda = 0.5`, release build,
same hardware (Apple M5 Max):

| Side | Median single-solve | Ratio |
|---|---:|---:|
| DERNA C++ `single_brd` | 16.44 s | 1.00× |
| Rust `solve_brd` | 15.636 s (N=5 median; spread 15.613 / 15.620 / 15.636 / 15.655 / 15.715 s) | **0.951× (~4.9 % faster)** |

Use this table for current standing. Use paired A/B runs on one build
tree for the effect of a single change. The candidate-streaming change's
quoted −17.5 % came from a paired N=5 run (`18.942 s` → `15.636 s`). The
pair-only direct reads' quoted −2.0 % came from a paired N=4 run
(`19.531 s` → `19.132 s`), not from mixing an earlier telemetry number
(`19.33 s`) with a later run.

The PORD sweep ratio at GLPE-78 is roughly `4.7×` solve-count
multiplier (the PORD scheduler visits more λs because the Rust and C++
sides optimize differently scaled objective surfaces, so at the same λ
they can select neighbouring cells) times the per-solve multiplier. The sweep ratio is
not the parity gate; per-solve is.

### Profile attribution (before and after streaming candidates, GLPE-78 λ=0.5)

The samply profile before candidate streaming, at 1 ms sampling, had 19,134 main-thread
samples over a 19.1 s single solve. It was one profile, not a
variance-bounded distribution, but it correctly identified candidate
materialization as a gap-sized target. Self-time roll-up by recurrence
area before candidate streaming:

| Self-time % | Bucket | Notes |
|---:|---|---|
| 52.9 % | internal-loop (`E_i`) | dominant; the largest single bucket |
| 18.6 % | `fill_compact_e_cell` driver | outer-pair gate + 5-motif dispatch + `select_e_motif` reduce + winner-tag write |
| **13.7 %** | **libsystem_malloc (heap)** | per-candidate `consistent_choices.collect::<Vec<_>>()`, candidate Vecs, scratch growth — Rust-specific overhead C++ does not pay |
| 8.4 % | multi (`E_m` / `M` / `N`) | rectangle sweep work |
| 4.4 % | memset / bzero / memmove | initialization + Vec resets; overlaps with allocator bucket |
| 0.4 % | `two_pair::pair_at` | negligible once pairs are read directly from codons |
| 0.2 % | bulge | negligible |
| ~0 % | stack, hairpin, exterior, traceback | invisible at GLPE-78 λ=0.5; the outer-pair gate drops these below threshold |

Top leaf functions (self-time %):

| % | Function |
|---:|---|
| 18.6 % | `recurrences::fill_compact_e_cell` |
| 14.6 % | `recurrences::internal::emit_internal_candidates::{{closure}}::{{closure}}` |
| 13.6 % | `recurrences::internal::compute_internal_energy` |
|  9.1 % | `recurrences::internal::emit_internal_candidates::{{closure}}` |
|  9.0 % | `recurrences::internal::build_inner_scope` |
|  6.6 % | `recurrences::internal::read_base_pairs` |
|  3.9 % | `recurrences::multi::m_candidates` |
|  3.5 % | `recurrences::multi::n_candidates` |
|  3.0 % | `__bzero` (libsystem) |
|  1.0 % | `_platform_memset` |

Candidate streaming passed `consistent_choices` through hairpin, multi, and
exterior recurrence paths, removed stack/bulge `Vec<InnerRef>`
materialization, streamed bulge/internal enclosed-pair positions, and
replaced exterior pinned candidate `Vec`s with stack-backed `SmallVec`.
The paired GLPE-78 λ=0.5 result was `18.942 s` → `15.636 s`
(**−17.5 %**), and the post-change samply profile recorded no
`libsystem_malloc`, `malloc`, `memset`, or `bzero` strings in the saved
sample table. By that evidence, allocator self-time fell below sample
resolution.

### Where future perf work plausibly lives

Per-solve parity with DERNA C++ is met. If more exact-solver margin is
needed, the largest remaining area is still internal-loop recurrence
work: before candidate streaming it was 52.9 % of self-time. Even a 10 %
win inside internal-loop candidate emission or energy evaluation is worth
several percentage points of total per-solve cost. That target should be
decomposed into measurable sub-targets before it is attempted.

The driver self-time of 18.6 % in `fill_compact_e_cell` is largely
load-bearing — outer-pair canonicity gate, five motif dispatches, motif
reduction, winner-tag write — and is not a productive target on its own.

## 2. Pseudocode

### 2.1 Notation

Let:

- `n` = amino-acid length.
- `m` = `max_codons`, the maximum synonym count over positions in this solve.
  With the standard genetic code, `m <= 6`.
- `c[p] = len(codon_lists[p])`, the actual synonym count at amino-acid
  position `p`.
- `R[a,b] = c[a] * c[b]`, the actual codon-rectangle size for endpoint pair
  `(a,b)`.
- `nt(a,s) = 3a + s`.
- `aa(k) = floor(k / 3)`.
- `pos(k) = k mod 3`.

The logical table shape remains:

```math
logical(n,m) = [n, n, 3, 3, m, m]
```

The source loops over actual synonym counts for `a` and `b`. Physical storage
has two modes:

```text
DenseMaxRectangles:
  all endpoint pairs (a,b), all max-codon slots m*m

PackedUpperTriangle:
  only endpoint pairs with a <= b,
  each endpoint pair stores its actual rectangle c[a]*c[b]
```

Production `solve_brd` uses `PackedUpperTriangle` plus
`CompactMotifWinners`. Test-only per-cell parity snapshots use
`DenseMaxRectangles` plus `FullMotifTables`.

### 2.2 Table Layout

Dense snapshot mode uses the historical row-major flat index:

```text
dense_flat(a,b,s,t,x,y) =
    (((((a * n + b) * 3 + s) * 3 + t) * m + x) * m + y)
```

Packed production mode precomputes one base offset per upper-triangle endpoint
pair:

```text
total = 0
for a in 0..n:
  for b in a..n:
    pair = a*n + b
    pair_offset[pair]    = total
    pair_rect_size[pair] = c[a] * c[b]
    pair_y_stride[pair]  = c[b]
    total += 9 * pair_rect_size[pair]
```

The packed flat index is:

```text
packed_flat(a,b,s,t,x,y) =
    pair_offset[a*n + b]
  + (s*3 + t) * pair_rect_size[a*n + b]
  + x * pair_y_stride[a*n + b]
  + y
```

Element strides inside a fixed packed endpoint pair:

| Axis changed by +1 | Element stride |
|---|---:|
| `y` | `1` |
| `x` | `c[b]` |
| `t` | `R[a,b]` |
| `s` | `3R[a,b]` |

Packed storage preserves the most important local property: `y` is stride-1,
and `x` is a short row jump bounded by the right endpoint's synonym count.
It changes the endpoint story: moving to another `(a,b)` pair jumps by a
variable precomputed block size rather than by a fixed dense stride.

Every packed table access also pays more index work than the dense formula:

```text
pair = a*n + b
base = pair_offset[pair]
rect = pair_rect_size[pair]
y_stride = pair_y_stride[pair]
flat = base + (s*3+t)*rect + x*y_stride + y
```

That cost sits in every hot recurrence read and write. The packed A/B result
shows that the memory-footprint win alone did not dominate the added indexing
and changed-locality costs for GLPE-78 `lambda = 0.5`.

Dense exports for tests allocate the full dense shape and copy every packed
upper-triangle actual rectangle into its corresponding dense positions, leaving
unused cells at `INF`.

### 2.3 Top-Level Fill

The full scalar solve allocates production tables and fills increasing
amino-acid spans:

```text
solve_brd(lambda):
  plan = Plan(
    e_storage_mode = CompactMotifWinners,
    table_layout   = PackedUpperTriangle,
  )
  fill_dp_tables(plan)
  top = argmin_x_y O[0,n-1,0,2,x,y]
  traceback through O and compact E_winner
  re-fold assembled RNA for reported structure/MFE
```

The fill itself is:

```text
for span in 0..n:
  for a in 0..n-span:
    b = a + span
    fill_ab_cell_full(a,b)
```

At each `(a,b)`, the implementation runs four phases:

```text
fill_ab_cell_full(a,b):
  sweep_cells(ST_BY_NT_SPAN):       fill_m(idx)
  sweep_cells(ST_BY_NT_SPAN):       fill_compact_or_full_e(idx)
  sweep_cells(ST_BY_NT_SPAN_FOR_N): fill_n(idx)
  sweep_cells(ST_BY_NT_SPAN_FOR_N): fill_o(idx)         # prefix-only: a=0, s=0
```

`fill_o` is prefix-only: only cells with `a == 0 && s == 0` are filled.
The exterior recovery target sits at `O[0, n-1, 0, t, x, y]`, so cells
outside the prefix slice are never read; suppressing their fill drops
the exterior work from O(n³) per solve to O(n²).

Each sweep is:

```text
for (s,t) in order:
  for x in 0..c[a]:
    for y in 0..c[b]:
      body(a,b,s,t,x,y)
```

### Cell-level outer-pair gate

`fill_compact_e_cell` and `fill_full_e_cell` short-circuit on
non-canonical outer pairs before any motif candidate runs:

```text
fill_compact_e_cell(plan, idx):
  if not outer_pair_is_canonical(plan, idx):
    E[idx] = INF
    E_winner[idx] = Hairpin       # arbitrary canonical tag
    return
  run five motif candidate streams; reduce to (winner, best)
  E[idx] = best
  E_winner[idx] = winner
```

`outer_pair_is_canonical` reads `codon_lists[a][x].nuc[s]` and
`codon_lists[b][y].nuc[t]` directly and checks
`pair_from_nucleotides(...)`. At GLPE-78, **38.7 % of E-table cells
are canonical** (243,567 gated / 153,999 admitted / 397,566 total) —
the gate drops the other 61.3 % before any candidate work executes.
The profile confirms this: stack, hairpin, exterior, and traceback
self-time are all below the 0.2 % print threshold post-gate.

`ST_BY_NT_SPAN` is ascending `t - s`, which makes stack inner cells
`(s+1,t-1)` available before their outer cells:

```text
(2,0),
(1,0), (2,1),
(0,0), (1,1), (2,2),
(0,1), (1,2),
(0,2)
```

`ST_BY_NT_SPAN_FOR_N` is descending `s` and ascending `t`, which makes `N` and
`O` same-cell self references `(s+1,t)` and `(s,t-1)` available:

```text
(2,0), (2,1), (2,2),
(1,0), (1,1), (1,2),
(0,0), (0,1), (0,2)
```

### 2.4 Shared Reduction Shape

Every recurrence is a candidate stream plus a `min` reduction:

```text
best = INF
for candidate in candidates(idx):
  best = min(best, candidate.value)
table[flat(idx)] = best
```

Full-motif snapshot mode stores each motif table, then reduces them:

```text
E_h[idx] = min(hairpin candidates)
E_s[idx] = min(stack candidates)
E_b[idx] = min(bulge candidates)
E_i[idx] = min(internal candidates)
E_m[idx] = min(multi-loop closure candidates)

E[idx] = min(E_h[idx], E_s[idx], E_b[idx], E_i[idx], E_m[idx])
```

Production compact mode never writes the five motif value tables:

```text
values = {
  h: min(hairpin candidates),
  s: min(stack candidates),
  b: min(bulge candidates),
  i: min(internal candidates),
  m: min(multi-loop closure candidates) or INF before multi-loop phase,
}
(winner, best) = first_min_in_order(h, s, b, i, m)
E[idx] = best
E_winner[idx] = winner
```

Tie order is stable: `E_h`, then `E_s`, then `E_b`, then `E_i`, then `E_m`.
Traceback reads `E_winner[idx]` and regenerates the selected motif's candidate
stream until it finds the stored `E[idx]` value.

This removes five full score tables from production storage and removes the
five-table cross-array `E` reduction stream. It does not remove the candidate
work: all five motif candidate streams still run when filling `E`.

### 2.5 Candidate Side Streams

Hairpin needs a full RNA substrate for its `loop_seq` slice; every
other pair-only path now reads nucleotide identities directly from
`codon_lists` through `plan::nuc_at` (and `nuc_from_index` for the
`u8 → Nucleotide` conversion). The hairpin assembly looks like:

```text
scratch = default_seq
for (aa, codon_index) in pinned codons:
  overwrite scratch[3*aa .. 3*aa+3] with codon_lists[aa][codon_index]
read loop sequence scratch[nt_i+1 .. nt_j]
```

Pair-only paths skip the assembly entirely:

```text
outer = pair_from_nucleotides(
  nuc_from_index(codon_lists[a][x].nuc[s]),
  nuc_from_index(codon_lists[b][y].nuc[t]),
)
```

The current recurrence split is:

| Recurrence path | Substrate / side work |
|---|---|
| `E_h` | assembles candidate RNA for `loop_seq`; outer pair already canonical by the cell-level outer-pair gate, so `closing_pair_of` in hairpin is defense-in-depth (unreachable on the live path) |
| `E_s`, `E_b` | read outer and inner pairs directly through `two_pair::pair_at(plan, site, h, r)` — no substrate assembly |
| `E_i` | reads pair and boundary nucleotides directly from pinned codons and side tables |
| `E_m` | reads closing pair directly through `multi::closing_pair_at(plan, idx)` — outer-pair check redundant behind the cell-level gate |
| `N` case 1 | reads branch closing pair directly through `multi::closing_pair_at`; this is the one multi-family path where the canonicity early-out is still load-bearing (N is filled outside the E gate) |
| `O` closure / bifurcations | read closing pairs directly through `exterior::e_with_ext_loop` + a `pinned_nuc_at(plan, pinned, nt)` resolver that maps an arbitrary nt position to its pinned codon (or the per-position `default_codon_idx` fallback) |

Hot candidate-side allocation is removed by streaming candidates:

- hairpin, `E_m`, `M`, `N`, and `O` stream `consistent_choices`
  directly through candidate loops,
- `stack` and `bulge` evaluate finite two-pair inner references directly
  in the iterator callback,
- `bulge` and `internal` stream bounded enclosed-pair positions, and
- `O` bifurcations carry pinned codons in stack-backed `SmallVec`s.

For cache analysis, the remaining memory streams are:

- large DP-table streams (`E`, `M`, `N`, `O`, and snapshot-only motif tables),
- one small repeated substrate stream (hairpin `default_seq` overwrite — every
  other pair-only path was migrated off this), and
- small stack-backed candidate-side streams.

The profile before candidate streaming attributed **13.7 %** self-time to
heap traffic; the profile after it no longer sampled allocator symbols. The exact
remaining hot split should be re-profiled only if future work needs
more exact-solver margin.

### 2.6 Single-Loop Recurrences

#### Hairpin `E_h`

```text
if hairpin loop is shorter than TURN:
  hairpin_value(idx) = INF
else:
  c = aa(nt(a,s) + 1)
  d = aa(nt(b,t) - 1)
  for h in codons at c consistent with outer pins:
    for r in codons at d consistent with outer pins and h:
      assemble candidate RNA
      if closing pair is canonical:
        value = brd_value(lambda, hairpin_energy(loop sequence), cai(a,b,c,d))
        best = min(best, value)
  hairpin_value(idx) = best
```

Production writes only `E[idx]` after all motif values have been reduced.
Snapshot mode writes `E_h[idx]`. `E_h` reads no DP tables; its runtime is
candidate enumeration, scratch assembly, hairpin-energy evaluation, and the
final table write.

#### Shared two-pair inner read

Stack, bulge, and internal loops all read enclosed-pair `E` cells through the
same shape:

```text
for h in codons at c consistent with outer pins:
  for r in codons at d consistent with outer pins and h:
    inner = (c,d,s_prime,t_prime,h,r)
    inner_val = E[inner]
    if inner_val < INF:
      emit(h,r,inner_val)
```

Under packed production storage, for a fixed enclosed `(c,d,s_prime,t_prime,h)`,
reads over `r` are stride-1. Advancing `h` jumps by `c[d]` elements. Moving to
another enclosed endpoint pair jumps to another variable packed pair block.

#### Stack `E_s`

```text
nt_p = nt(a,s) + 1
nt_q = nt(b,t) - 1
site = enclosed cell for nt_p,nt_q
inners = finite two_pair_inner(site)
for (h,r,inner_val) in inners:
  (outer, inner) = pair_at(plan, site, h, r)   # direct codon reads
  if outer and inner pairs are canonical:
    value = inner_val + brd_value(lambda, stack_energy(outer, inner), closing_cai)
    best = min(best, value)
stack_value(idx) = best
```

The important DP locality is the two-pair inner rectangle: good over
`r`, small jump over `h`, larger variable jumps when the enclosed
endpoint changes. The implementation evaluates stack candidates directly
from `iter_two_pair_inner` (the
`iter_two_pair_inner_emission_byte_identical` test pins this emission
order). Behind the cell-level outer-pair gate the outer-pair check inside
`pair_at` is redundant for production cells; the inner-pair check still
filters non-canonical `(h, r)` candidates.

#### Bulge `E_b`

```text
positions = enclosed positions bounded by MAXLOOP and bulge shape
for nt_p,nt_q in positions:
  nt_i = nt(a,s)
  nt_j = nt(b,t)
  site = enclosed cell for nt_p,nt_q
  inners = finite two_pair_inner(site)
  for (h,r,inner_val) in inners:
    (outer, inner) = pair_at(plan, site, h, r)   # direct codon reads
    if outer and inner pairs are canonical:
      value = inner_val + brd_value(lambda, bulge_energy(max(n1,n2)), closing_cai)
      best = min(best, value)
bulge_value(idx) = best
```

`E_b` is bounded by `MAXLOOP`, so the enclosed-position count is a
large constant rather than an n-scale split. Its `E` reads have the
same local two-pair shape as stack. The current implementation streams
both the enclosed positions and per-site finite-inner references.

#### Internal loop `E_i`

```text
positions = enclosed positions bounded by MAXLOOP and internal-loop shape
for nt_p,nt_q in positions:
  nt_i = nt(a,s)
  nt_j = nt(b,t)
  site = enclosed cell for nt_p,nt_q
  flanks = (aa(nt_i+1), aa(nt_j-1), aa(nt_p-1), aa(nt_q+1))

  for (h,r,inner_val) in two_pair_inner(site):
    read outer and inner pairs directly from pinned codons
    if outer and inner pairs are canonical:
      classify flank slots as Fixed(nt) or Free
      for (xi,hi) in left boundary nucleotide enumeration:
        for (kj,yj) in right boundary nucleotide enumeration:
          energy = internal_energy_scalar(outer, inner, xi, hi, kj, yj, n1, n2)
          cai = closing_cai + best_synonym_cai_for_boundary_nts
          value = inner_val + brd_value(lambda, energy, cai)
          best = min(best, value)
internal_value(idx) = best
```

The current implementation already uses boundary-nucleotide enumeration for
the flanks. That avoids enumerating dominated flank codons and avoids scratch
RNA assembly for internal loops. Its remaining DP-table stream is the shared
two-pair `E` read; its additional work is enclosed-position allocation,
boundary-nt enumeration, side-table reads, and `internal_energy_scalar`.

### 2.7 Multi-Loop Recurrences

#### Multi-loop closure `E_m`

`E_m` reads `M` and pays the multi-loop closing cost. The closing pair
is read directly from `codon_lists` (no substrate assembly); behind the
cell-level gate the outer-pair check is redundant on the live `fill_compact_e_cell`
path but is retained for diagnostic symmetry:

```text
if closing pair is canonical:
  case 1, no boundary cross:
    if a < b and s < 2 and t > 0:
      inner = (a,b,s+1,t-1,x,y)
      candidate = M[inner] + closing_cost

  case 2, left boundary cross:
    if a+1 < b and s == 2 and t > 0:
      for x_prime at a+1:
        inner = (a+1,b,0,t-1,x_prime,y)
        candidate = M[inner] + closing_cost + left_cai

  case 3, right boundary cross:
    if a+1 < b and s < 2 and t == 0:
      for y_prime at b-1:
        inner = (a,b-1,s+1,2,x,y_prime)
        candidate = M[inner] + closing_cost + right_cai

  case 4, both boundaries cross:
    if a+2 < b and s == 2 and t == 0:
      for x_prime at a+1:
        for y_prime at b-1:
          inner = (a+1,b-1,0,2,x_prime,y_prime)
          candidate = M[inner] + closing_cost + left_cai + right_cai

e_m_value(idx) = min(candidates)
```

Case 1 is nearby in the same packed `(a,b)` block. Boundary-cross cases jump
to smaller-span endpoint blocks. Production does not store `E_m`; it folds the
value into compact `E` immediately.

#### Multi-loop body `M`

`M` bifurcates into two `N` fragments:

```text
if a < b:
  for split nucleotide k in nt(a,s)+1 .. nt(b,t)-2:
    c = aa(k)
    d = aa(k+1)
    if a < c and d < b:
      for y_prime at c:
        xp_choices = codons at d consistent with outer pins and y_prime
        left = N[a,c,s,pos(k),x,y_prime]
        if left finite:
          for x_prime in xp_choices:
            right = N[d,b,pos(k+1),t,x_prime,y]
            value = left + right
            if c == d:
              value += shared-AA CAI correction
            best = min(best, value)
M[idx] = best
```

The current implementation hoists the left `N` read out of the `x_prime` loop.
For fixed split and `y_prime`, the left cell is reused across all
`x_prime` choices. The right read advances by the packed right endpoint's
`x` stride (`c[b]` elements) as `x_prime` changes. Advancing the split moves
both endpoint blocks and usually loses reuse.

#### Multi-loop fragment `N`

`N` has seven cases:

`N` is filled outside the cell-level outer-pair gate, so its case 1
canonicity check is the one multi-family path where the early-out is
still load-bearing. The closing pair is read directly via
`multi::closing_pair_at(plan, idx)`.

```text
if a < b:
  case 1:
    if E[idx] finite and closing pair is canonical:
      close a branch via E[idx]

  case 2:
    if s < 2:
      read N[a,b,s+1,t,x,y]

  case 3:
    if t > 0:
      read N[a,b,s,t-1,x,y]

  case 4:
    if s == 2 and a+1 < b:
      for x_prime at a+1:
        read N[a+1,b,0,t,x_prime,y]

  case 5:
    if t == 0 and a+1 < b:
      for y_prime at b-1:
        read N[a,b-1,s,2,x,y_prime]

  case 6:
    if a+1 < b:
      for c in a+1 .. b-1:
        for k in 0..2:
          for y_prime at c:
            read N[a,c,s,k,x,y_prime]
            read N[c,b,k+1,t,y_prime,y]

  case 7:
    if a+2 < b:
      for c in a+1 .. b-2:
        for y_prime at c:
          xp_choices = codons at c+1 consistent with pins and y_prime
          left = N[a,c,s,2,x,y_prime]
          if left finite:
            for x_prime at c+1:
              read N[c+1,b,0,t,x_prime,y]
```

Cases 2 and 3 are nearby same-cell reads supplied by the special `(s,t)` order.
Cases 6 and 7 read two `N` cells per candidate. Case 7 now hoists the left
read out of the `x_prime` loop, matching the `M` reuse shape.

### 2.8 Exterior Recurrence `O`

`O` is filled prefix-only: production `fill_o` runs only for cells
with `a == 0 && s == 0`. The exterior recovery target sits at
`O[0, n-1, 0, t, x, y]`, so non-prefix cells are never read. The
all-interval fill is retained as a diagnostic-only path
(`fill_o_legacy_all_cases`) for the winning-case audit.

Within the prefix slice, `O` has base cases at `a == b`, shift/cross
cases, bifurcation cases, and a closure through `E`:

```text
if a == b:
  O[idx] = brd_value(lambda, 0, log_cai[a][x]) if x == y else INF
else:
  case 3:
    if s < 2:
      read O[a,b,s+1,t,x,y]

  case 4:
    if t > 0:
      read O[a,b,s,t-1,x,y]

  case 5:
    if s == 2:
      for x_prime at a+1:
        read O[a+1,b,0,t,x_prime,y]

  case 6:
    if t == 0:
      for y_prime at b-1:
        read O[a,b-1,s,2,x,y_prime]

  case 7:
    for c in a .. b-1:
      for t_prime in 0..2:
        for x_prime at c:
          read O[a,c,s,t_prime,x,x_prime]
          read E[c,b,t_prime+1,t,x_prime,y] plus exterior stem credit

  case 8:
    if a+1 < b:
      for c in a .. b-2:
        for y_prime at c:
          xp_choices = codons at c+1 consistent with pins and y_prime
          left = O[a,c,s,2,x,y_prime]
          if left finite:
            for x_prime at c+1:
              read E[c+1,b,0,t,x_prime,y] plus exterior stem credit

  case 9:
    if E[idx] finite and closing pair is canonical:
      close through E[idx]
```

Cases 7 and 8 mix an `O` left stream with an `E` right stream. Case 8
hoists the left `O` read out of the `x_prime` loop. `E` closure reads
run through `e_with_ext_loop`, which now reads the closing pair
directly via the `pinned_nuc_at(plan, pinned, nt)` resolver — no
substrate assembly. After the prefix-only fill, total `e_with_ext_loop`
calls per solve are O(n²) rather than O(n³).

## 3. Complexity And Cache Analysis

### 3.1 Storage Complexity

Dense snapshot cells per table:

```math
D(n,m) = 9n^2m^2
```

Packed production cells per table:

```math
P(c) = 9 \sum_{0 \le a \le b < n} c[a]c[b]
```

Current production storage is:

```math
bytes_{prod} = 4 \cdot P(c) \cdot sizeof(i32) + P(c) \cdot sizeof(EMotifKind)
             = 17P(c)
```

because production stores four `i32` tables (`E`, `M`, `N`, `O`) and one
`repr(u8)` winner table. Dense full-motif snapshots store nine `i32` tables:

```math
bytes_{snapshot} = 9 \cdot D(n,m) \cdot sizeof(i32) = 36D(n,m)
```

Concrete footprints:

| Input | One dense table | Dense full snapshot | Dense compact production | Packed compact production |
|---|---:|---:|---:|---:|
| GLPE-20 (`n=20`) | 0.49 MiB | 4.45 MiB | 2.10 MiB | 0.51 MiB |
| GLPE-78 (`n=78`) | 7.52 MiB | 67.68 MiB | 31.96 MiB | 6.45 MiB |
| `n=1000`, dense worst-case `m=6` | 1.21 GiB | 10.86 GiB | 5.13 GiB | input-composition dependent |
| `n=1000`, GLPE-like packed ratio | n/a | n/a | n/a | about 1.03 GiB |

For GLPE-78:

```text
D = 1,971,216 table elements
P =   397,566 table elements
P / D = 0.2017
```

Interpretation:

- The old "all nine dense tables" production model is no longer accurate.
- Compact `E` removed the five full motif score tables from production.
- Packed upper-triangle actual rectangles cut the remaining production element
  count to about 20% of the dense max-rectangle count for GLPE-78.
- The packed production plan is now small enough that "whole-table footprint
  exceeds LLC" is not the right first explanation for the one-solve timing.
  The live suspects are hot-loop table-access count, packed-index arithmetic,
  irregular endpoint jumps, candidate allocation, and repeated substrate
  assembly.

### 3.2 Time Complexity

With synonym counts and `MAXLOOP` treated as biological constants:

| Recurrence family | Per fixed `idx` | Whole solve in `n` | Why |
|---|---:|---:|---|
| Hairpin | O(1) | O(n²) | bounded flank codon choices |
| Stack | O(1) | O(n²) | one enclosed pair, bounded codons |
| Bulge | O(1) | O(n²) | `MAXLOOP`-bounded enclosed positions |
| Internal | O(1) | O(n²) | `MAXLOOP`-bounded positions and boundary nts |
| `E_m` | O(1) | O(n²) | four local/cross cases |
| `M` | O(span) | O(n³) | split nucleotide `k` across interval |
| `N` | O(span) | O(n³) | bifurcation cases 6 and 7 |
| `O` | O(span) | O(n³) | bifurcation cases 7 and 8 |

Keeping codon counts visible, the cell count already includes the outer
`(x,y)` endpoint rectangle. Candidate loops can add up to another bounded
codon factor on recurrences that choose inner or split codons. For example, a
bifurcation recurrence is `O(n³ c⁴)` in a dense-codon model, but `c <= 6`.
At GLPE-78, constant factors and memory behavior dominate the asymptotic
notation.

Packed layout does not change the recurrence count. It changes:

- how many physical cells are initialized and touched,
- how much index arithmetic each access performs, and
- how endpoint-changing reads map to memory.

### 3.3 Access-Pattern Summary

| Recurrence | Production writes | DP reads | Locality summary |
|---|---|---|---|
| `E_h` | compact `E` only after motif reduction | none | candidate-loop + hairpin substrate dominated; reaches this stream only when the cell-level outer-pair gate admits the cell |
| `E_s` | compact `E` only after motif reduction | `E[inner]` | enclosed `E` reads stride-1 over `r`; stack streams finite inners; closing/inner pairs read directly from `codon_lists` |
| `E_b` | compact `E` only after motif reduction | `E[inner]` | streams bounded positions and inners; each codon rectangle stride-1 over `r`; closing/inner pairs read directly |
| `E_i` | compact `E` only after motif reduction | `E[inner]` | same two-pair `E` reads plus boundary-nt side tables; pair reads via direct codon lookup |
| `E_m` | compact `E` only after motif reduction | `M[inner]` | local case nearby; cross cases jump to smaller packed endpoint blocks; closing pair read directly |
| `M` | `M[idx]` | two `N` cells | left read hoisted across `x_prime`; split endpoints still move widely |
| `N` | `N[idx]` | `E[idx]`, `N` | same-cell reads nearby; bifurcations read two `N` cells, with case-7 left hoist; case 1's closing pair read directly |
| `O` (prefix-only) | `O[idx]` for `a=0, s=0` only | `O`, `E` | bifurcations mix `O` and `E`; case-8 left hoist; case-9 closing pair read directly via `pinned_nuc_at` |
| compact `E` | `E[idx]`, `E_winner[idx]` | motif candidate streams | the cell-level gate short-circuits 61.3 % of cells before any motif generator runs; admitted cells execute all five generators |

The strongest table-stream cache-risk zones remain `M` and `N`,
because their bifurcation cases read two table cells per candidate and
those cells usually move through different endpoint axes. The packed
A/B result means table footprint alone is not a sufficient
explanation. If table access were still dominant the profile would
show it; instead the profile before candidate streaming showed internal-loop work and
allocator traffic on top, with `M` and `N` together at ~8 % self-time.
With allocator traffic removed, layout-level table specialization
remains a smaller future-margin target than internal-loop work — see
§ 4.

### 3.4 Reuse-Distance Notes

Reuse distance here means "how much intervening table/candidate work happens
before the same cache line is likely to be read again."

| Pattern | Stride / reuse under packed production |
|---|---|
| Table writes over `y` | stride-1, best-case cache-line use |
| Table writes over `x` | jump by `c[b]` elements; still small for `c[b] <= 6` |
| Stack same-cell inner `(s+1,t-1)` | offset `+2R[c,d]` within the enclosed pair block |
| `N`/`O` same-cell `(s+1,t)` | offset `+3R[a,b]` within the same pair block |
| `N`/`O` same-cell `(s,t-1)` | offset `-R[a,b]` within the same pair block |
| Two-pair `E` reads over `r` | stride-1 |
| Two-pair `E` reads over `h` | stride `c[d]` |
| Boundary-cross read changing `b` | jump to another variable packed endpoint block |
| Boundary-cross read changing `a` | jump to another variable packed endpoint block, often farther in offset order |
| `M` split, fixed left over `x_prime` | left cell reused across inner `x_prime` loop |
| `M` split, right over `x_prime` | right cell advances by stride `c[b]` |
| `N` case 7 / `O` case 8 fixed left over `x_prime` | left cell reused across inner `x_prime` loop (`O` paths only fire on prefix cells, since `O` is prefix-only) |
| `N`/`O` bifurcation next split | both endpoints move; reuse is unlikely without scratch/layout support |
| Compact `E` reduction | no five-table reduction stream; motif candidate work is repeated directly |

This suggests a useful distinction:

- **Good local streams:** writes over `y`, two-pair reads over `r`, and
  same-cell shifts inside one packed endpoint block.
- **Potentially expensive streams:** endpoint-changing bifurcations,
  repeated packed-index computation, and candidate-side helper work.
  Hairpin is the only remaining substrate-assembly stream and it sits
  below the profile's print threshold (the cell-level outer-pair gate filters most
  cells before `fill_e_h` runs).

## 4. Remaining Opportunities

This section is profile-driven (see § 1.5). Entries are grouped as
**Future** (possible exact-solver margin work after re-profiling),
**Closed** (delivered or attempted-and-regressed — kept visible so prior
outcomes are not re-invented), and **Out of Scope** (algorithmic-shape
changes that belong to a different solver than this exact DP).

Per-solve parity with DERNA C++ is met. The remaining entries are
retained as future exact-solver margin ideas, not as required work.

### 4.1 Closed — Remove Hot Candidate Materialization (delivered)

The profile before candidate streaming attributed **13.7 % of self-time
to `libsystem_malloc`** plus another ~4 % to `__bzero` /
`_platform_memset`. Streaming candidates removed the hot materialization
sites and closed the per-solve gap.

Delivered changes:

- streamed `consistent_choices` in hairpin, `E_m`, `M`, `N`, and `O`,
- removed `Vec<InnerRef>` in stack and bulge,
- streamed bounded enclosed-pair positions in bulge/internal, and
- moved short `O` bifurcation pinned lists to stack-backed `SmallVec`.

Measured outcome:

- paired GLPE-78 λ=0.5 N=5 timing: `18.942 s` → `15.636 s`
  (**−17.5 %**),
- standing vs DERNA C++ `16.44 s`: **0.951×**,
- solution fields unchanged: `mfe_kcal_mol=-143.4000`,
  `cai=0.829243`, and
- post-change samply profile had allocator symbols below sample
  resolution.

The result validates that profile as directionally useful, and leaves
internal-loop work as optional margin rather than a parity requirement.

### 4.2 Future — Internal-Loop Micro-Optimization (needs decomposition)

Internal-loop recurrence work is **52.9 % of per-solve self-time**.
This is the largest bucket, but the targets below are research moves,
not ready-to-implement spikes.
Within that bucket:

| % | Function | Sub-target |
|---:|---|---|
| 14.6 % | `emit_internal_candidates::{{closure}}::{{closure}}` | inner-most candidate emission; check for redundant work inside the candidate closure or inlining opportunities |
|  9.1 % | `emit_internal_candidates::{{closure}}` | outer candidate iteration over enclosed positions |
| 13.6 % | `compute_internal_energy` | per-candidate energy evaluation; check `internal_energy_scalar` inlining and the `rna_mfe` `int11/int21/int22` lookup paths |
|  9.0 % | `build_inner_scope` | per-candidate flank scope construction; candidate for hoisting invariants out of the inner enumeration |
|  6.6 % | `read_base_pairs` | per-candidate pair-read setup; should already be fast after the direct pair reads — verify no unnecessary work remains |

**Open questions** (these are research moves, not ready-to-implement
spikes):

- How much of `build_inner_scope`'s 9.0 % is invariant across the
  `(h, r)` loop and could be hoisted out?
- With candidates streamed, what share of `compute_internal_energy` remains once
  allocator traffic is below sample resolution?
- Are there inlining wins between `emit_internal_candidates`'s
  closures? Two nested closure frames at the top of the profile
  suggests possible inlining left on the table.
- Is `internal_energy_scalar` (in `rna_mfe`) hot enough to need an
  inline annotation or a specialized path for the dominant case
  shape?

**Expected leverage:** before candidate streaming, even a 10 % improvement inside the
internal-loop bucket was worth about 5 percentage points of total
per-solve cost. Larger wins are plausible if `build_inner_scope`
hoisting or candidate-closure inlining lands, but this is future margin
work and should be re-profiled before implementation.

### 4.3 Future — Specialize Table Access For The Current Packed Layout

Every production table access currently goes through
`Tables::flat_index(idx)`. In packed mode that means:

```text
match layout
pair = a*n + b
load pair_offset[pair]
load pair_rect_size[pair]
load pair_y_stride[pair]
compute st block and codon-row offset
```

The dense layout had a larger footprint but a simpler fixed-stride
formula. The packed-versus-dense A/B showed memory-footprint
win without per-solve time win — consistent with this per-access cost.

The profile bucket for `M` + `N` is only ~8 %, so the leverage here is
smaller than § 4.1 / § 4.2, but candidate interventions are:

- precompute a small `PairLayout { base, rect, y_stride }` at the
  outer `(a,b)` sweep and pass it into same-pair accesses,
- precompute `(a,b,s,t)` block bases inside the sweep so writes and
  same-cell reads only add `x*y_stride + y`,
- add specialized helpers for common access shapes
  (same-pair `(s+1,t)`, `(s,t-1)`, two-pair inner rectangles), or
- split dense/snapshot and packed/production paths enough that hot
  production code does not pay the layout `match`.

**Expected leverage:** small but cheap to test if exact-solver margin is
needed later.

### 4.4 Future — Scratch-Buffer Split Reads In `M` And `N`

`O` is prefix-only; it carries no split work. `M` and
`N` remain as the O(n³) recurrences and share the split-and-codon-row
shape:

```text
for fixed split and y_prime:
  left = table[left_idx]
  for x_prime:
    right = table[right_idx(x_prime)]
    use left + right
```

Scalar hoists (case-7 left hoist in `N`, `m_split_iter` hoists in `M`)
already exploit part of this pattern. A more substantive version would:

- materialize a tiny right-side row for the current split when it is
  read repeatedly by adjacent outer cells,
- reorder an inner loop only when it improves the right-side table
  stride without changing candidate tie order, or
- add recurrence-specific row readers that compute one packed base
  once and walk the small codon rectangle directly.

**Expected leverage:** small (`M` + `N` together were ~8 % of self-time
before candidate streaming). Re-profile before promoting.

### 4.5 Closed — Pair-Only Direct Reads (delivered)

Stack / bulge `pair_at`, multi `closing_pair_at` (E_m closure + N case
1), and exterior `e_with_ext_loop` were migrated from substrate
assembly to direct `nuc_at` / `pinned_nuc_at` codon reads.
Hairpin still assembles for `loop_seq` (its closing-pair check is
dead in production behind the cell-level outer-pair gate). Paired
GLPE-78 λ=0.5 result: 19.531 s → 19.132 s (−2.0 %, byte-identical
MFE/CAI). Do not compute that −2.0 % from the separate cross-run
`19.33 s` → `19.13 s` headline; that comparison is about −1 %. See
§ 2.5 / § 2.6 / § 2.7 / § 2.8 for the current shape.

### 4.6 Closed — Per-Cell `E`-then-`N` Fusion (attempted; regressed)

Fusing each `(s, t, x, y)` cell's `M → E → N` work back-to-back
(instead of three separate rectangle sweeps) was attempted to collapse
case-N1's read of `E[idx]` from full-rectangle reuse distance to zero.
Both the full M/E/N variant and an E+N-only variant produced
byte-identical parity and regressed ~3–4 % at GLPE-78 λ=0.5. The
working hypothesis is I-cache pressure from interleaving E's five
motif kernels with N's seven cases per cell, instead of letting each
sweep stay resident in I-cache for the whole rectangle. No perf-counter
or samply evidence has confirmed the I-cache mechanism; the measured
claim is the regression, not the cause.

**Do not re-attempt without new evidence.** If a future restructuring
collapses the motif kernels (e.g., a unified candidate dispatch with
better inlining behavior), revisit then; absent that, the per-cell
fusion path is closed.

### 4.7 Closed — State Representation Is Not The C++ Gap

Investigated as a first-principles hypothesis: maybe the `(a, b, s, t, x, y)` state shape is over-specified
and a sparser codon-boundary representation would be cheaper. The
parallel C++ writeup showed DERNA C++ uses the same
`(a, b, i, j, x, y)` state shape and was 7.5× faster before the
cell-level outer-pair gate, so the
state shape is not the differentiator for the Rust-vs-C++ gap. This
does not prove the state shape is globally optimal for the problem; it
does mean a state rewrite does not close the per-solve gap. **Do not
spend a state-rewrite cycle here.**

### 4.8 Out of Scope — Avoid Repeating Full Scalar DPs For The Same Frontier

The current PORD driver treats each `lambda` as an independent scalar
BRD solve, leading to many full table fills landing on already-seen
solution cells. A multiobjective or parametric DP that exposed the
Pareto frontier directly could cut the sweep-count multiplier — but
the per-solve gap (which is what defines parity, since the two sides'
λ schedules differ by construction) would not move.

**Out of scope for this solver.** This is algebraic-DP territory, which
the `derna-lib` front scan covers, and fundamentally addresses the
sweep-count axis, not the per-solve axis.

### 4.9 Out of Scope — Stop Requiring Exact Cubic DP For Long CDS Inputs

For proteins in the thousands of amino acids the exact codon-lifted
Zuker DP is not the right tool — beam search or windowed folding will
own production-scale runtime. The exact solver remains the correctness
reference.

**Out of scope for this solver** — the `derna-lib` beam search covers it.

### 4.10 Out of Scope — Fuse Recurrences Across Paper Sections

The full cell-fusion attempt failed (see § 4.6). A more aggressive
restructure — combining `M/N/O` split scans into shared split
iterators, or generating recurrence kernels for common shape classes
— is a larger redesign than a parity-gap change. Worth revisiting
after § 4.1 + § 4.2 land and the picture clarifies.

### 4.11 What To Measure Before Or After Each Move

- **Before § 4.2:** add per-recurrence-family inclusive timing
  (counters around `internal::emit_internal_candidates` /
  `compute_internal_energy`) so individual sub-targets can be
  attributed.
- **For any future exact-solver margin change:** re-run GLPE-78
  λ=0.5 single-solve with N≥5 (paired pre/post on one build tree);
  sweep totals remain informational regression-protect only, not the
  per-solve parity gate.

