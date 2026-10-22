# DERNA C++ pseudocode and cache analysis

This document describes the DERNA C++ `Zuker` implementation at the same
cache-relevant abstraction level as the Rust codon-DP pseudocode draft. It is
not a line-by-line source walkthrough. It preserves the loop order, table
layout, table reads and writes, reuse-distance shape, and working-set size
needed for a structural comparison against the Rust solver.

Source anchors:

- [`Zuker.cpp`](../../../docs/reference/_cpp/derna-main/src/Zuker.cpp):
  constructor storage layout, `lambda_sweep_2`, `reinit`, `calculate_CAI_O`,
  `calculate_CAI_E`, `hairpin_CAI`, `internal_CAI`, `multi_loop_CAI`, and
  `calculate_CAI_M`.
- [`Zuker.h`](../../../docs/reference/_cpp/derna-main/src/Zuker.h): flat index
  and table accessors.
- [`default.cpp`](../../../docs/reference/_cpp/derna-main/src/default.cpp):
  amino-acid synonym counts and codon tables.
- [`utils.cpp`](../../../docs/reference/_cpp/derna-main/src/utils.cpp):
  amino-acid encoding and FASTA parsing.

Reader path: read sections 1, 2.2, 2.3, 3, and 4 first for the cache diagnosis.
Use the recurrence subsections in 2.4-2.10 when a comparison claim depends on an
exact C++ loop shape.

## 1. Problem Recap

The C++ reference `lambda_sweep_2` repeatedly solves a scalar BRD objective for
different `lambda` values. Each scalar solve fills codon-lifted dynamic
programming tables over intervals of an amino-acid sequence. A DP state is
keyed by:

```text
idx = (a, b, i, j, x, y)
```

where:

- `(a,b)` are amino-acid endpoints,
- `(i,j)` are nucleotide offsets inside codons `a` and `b`, and
- `(x,y)` are codon choices at `a` and `b`.

The lambda-mode implementation uses these dense scalar arrays:

```text
O, E1, M1, TM
Z2, E2, M2, TM2
Z_CAI, E_CAI, M_CAI, TM_CAI
OB, EB, MB
```

The first four are the objective-score tables used by the recurrence logic:

| C++ table | Recurrence role | Closest Rust table |
|---|---|---|
| `E1` | paired interval aggregate | `E` |
| `M1` | multi-loop fragment | `N` |
| `TM` | multi-loop bifurcation subtable | `M` |
| `O` | exterior/prefix aggregate | `O` |

`Z2/E2/M2/TM2` and `*_CAI` are companion tables that store the decomposed MFE
and CAI contributions for traceback and reporting. `OB/EB/MB` store compact
case tags. `O_bt/E_bt/M_bt/TM_bt` are `vector<vector<int>>` backtrace side
tables with one C++ vector object per DP slot.

The C++ implementation differs structurally from the Rust implementation in
three ways that matter for a cache comparison:

- Tables are packed by the actual synonym counts for each endpoint pair rather
  than by one global `max_codons x max_codons` rectangle.
- Paired-cell motif cases update one aggregate `E1` table directly; C++ does
  not keep separate full `E_h`, `E_s`, `E_b`, `E_i`, and `E_m` score tables.
- `O` is filled only for the prefix exterior states reachable from
  `(a=0,i=0)`, even though the array is allocated for all endpoint pairs.

## 2. Pseudocode

### 2.1 Notation

Let:

- `n` = amino-acid length.
- `c[a] = n_codon[protein[a]]`, the number of codons for amino acid `a`.
- `S = sum_a c[a]`.
- `nt(a,i) = 3a + i`.
- `aa(k) = floor(k / 3)`.
- `pos(k) = k mod 3`.

The constructor allocates every endpoint pair `(a,b)`, including pairs that are
not later filled by a recurrence. For each pair it reserves nine `(i,j)` blocks
and an actual codon rectangle `c[a] * c[b]`:

```text
slots_per_pair(a,b) = 9 * c[a] * c[b]
last_idx = sum_{a,b} slots_per_pair(a,b)
         = 9 * (sum_a c[a])^2
```

If every amino acid had `m` codons, this would match the Rust dense shape
`9n^2m^2`. For real proteins, C++ packs away the unused max-codon padding.

### 2.2 Table Layout

All scalar tables use the same flat index:

```text
start_index[a,b] =
  sum over endpoint pairs before (a,b) in row-major (a then b):
    9 * c[p] * c[q]

index(a,b,i,j,x,y) =
  start_index[a,b] + c[a] * c[b] * (3i + j) + c[b] * x + y
```

Within a fixed endpoint pair, the element strides are:

| Axis changed by +1 | Element stride | Byte stride in `double` table |
|---|---:|---:|
| `y` | `1` | `8 B` |
| `x` | `c[b]` | `8c[b] B` |
| `j` | `c[a]c[b]` | `8c[a]c[b] B` |
| `i` | `3c[a]c[b]` | `24c[a]c[b] B` |
| `b` | variable endpoint-block jump | variable |
| `a` | variable row jump near `9c[a]S` elements | variable |

The hot loops put `y` in the innermost loop and `x` one level outside it, so
writes are contiguous over `y` and then jump by `c[b]` elements over `x`.
Reads that change endpoint pairs jump through the row-major packed endpoint
layout. The jump is smaller than a max-codon layout for most proteins, but it
is no longer a fixed affine stride in `n` and `m`.

### 2.3 Top-Level Lambda Sweep And Fill Order

The adaptive sweep driver is:

```text
lambda_sweep_2(threshold, threshold2):
  queue = [(EPSILON, 1 - EPSILON)]
  processed = {}

  while queue not empty:
    (left, right) = queue.pop()

    for lambda in [left, right]:
      if lambda not in processed:
        reinit all DP arrays
        result[lambda] = solve_one_lambda(lambda)
        processed.add(lambda)

    if left and right recover different MFE and CAI:
      if lambda gap exceeds threshold:
        mid = (left + right) / 2
        queue.push(left, mid)
        queue.push(mid, right)
```

The scalar solve is:

```text
solve_one_lambda(lambda):
  calculate_CAI_O(lambda)
  traceback_B2(lambda)
  emit recovered RNA/structure metrics
```

`calculate_CAI_O` owns the full DP fill:

```text
calculate_CAI_O(lambda):
  if lambda != 1:
    initialize one-amino-acid diagonal CAI base cases

  calculate_CAI_E(lambda)

  # Exterior prefix only: a = 0, i = 0.
  for nucleotide length len in 1..3n-1:
    b = floor(len / 3)
    j = len mod 3
    if b == n: continue

    for x in codons[0]:
      for y in codons[b]:
        fill O[0,b,0,j,x,y]
```

The paired interval and multi-fragment fill is:

```text
calculate_CAI_E(lambda):
  for nucleotide length len in 4..3n-1:
    for a in 0..n-floor(len/3)-1:
      for i in 0..2:
        j = (i + len) mod 3
        b = floor((3a + i + len) / 3)
        if b == n: continue

        for x in codons[a]:
          for y in codons[b]:
            if closing pair is not canonical:
              calculate_CAI_M(a,b,i,j,x,y,lambda)
              continue

            best = INF
            best = min(best, hairpin_CAI(...))
            best = min(best, internal_CAI(...))
            best = min(best, multi_loop_CAI(...))
            E1[idx] = best

            calculate_CAI_M(a,b,i,j,x,y,lambda)
```

The fill order is global nucleotide-span order, not `(a,b)` span order with a
separate `(i,j)` schedule. Same-endpoint reads such as `(i+1,j)` and `(i,j-1)`
refer to shorter nucleotide spans and are already available from earlier `len`
iterations. Endpoint-crossing reads also shorten the nucleotide span.

### 2.4 Shared Reduction Shape

C++ uses a direct aggregate update shape:

```text
best = INF
best = min(best, hairpin candidate)
best = min(best, stack/bulge/internal candidate)
best = min(best, multi-loop closure candidate)
E1[idx] = best
```

There are no full score tables for hairpin, stack, bulge, internal, or
multi-loop closure. The helper that wins also writes the corresponding
decomposed values and backtrace side table:

```text
E2[idx] = winning_mfe_component
E_CAI[idx] = winning_cai_component
E_bt[idx] = winning_backtrace_payload
EB[idx] = winning_case_tag
```

This means the paired-cell recurrence streams write one aggregate score table
plus companion metadata, rather than writing five motif tables and later
reducing them.

### 2.5 Hairpin

`hairpin_CAI` reads no DP score tables. Its work is boundary codon/nucleotide
enumeration plus Turner hairpin lookup:

```text
hairpin_CAI(a,b,i,j,x,y):
  la = nt(a,i)
  lb = nt(b,j)

  if i == 2 and j == 0:
    for x1 in codons[a+1]:
      for y1 in codons[b-1]:
        choose inner boundary nts xi_, _yj from x1/y1
        if short same-codon boundary is incompatible: continue
        value = special_hairpin_or_general_hairpin(...)
        best = min(best, value)

  else if i == 2:
    for x1 in codons[a+1]:
      choose xi_ from x1
      _yj = nucleotide before j in outer codon y
      value = special_hairpin_or_general_hairpin(...)
      best = min(best, value)

  else if j == 0:
    xi_ = nucleotide after i in outer codon x
    for y1 in codons[b-1]:
      choose _yj from y1
      value = special_hairpin_or_general_hairpin(...)
      best = min(best, value)

  else:
    xi_ = nucleotide after i in outer codon x
    _yj = nucleotide before j in outer codon y
    value = special_hairpin_or_general_hairpin(...)
    best = min(best, value)

  if best improves E1[idx]:
    E1[idx] = best
    E2[idx] = mfe_part
    E_CAI[idx] = cai_part
    E_bt[idx] = backtrace_payload
```

The cache footprint is mostly the write stream into `E1/E2/E_CAI/E_bt` and
small static parameter/codon tables. The only large-array write is at the
current flat index.

### 2.6 Stack, Bulge, And Internal Loops

The C++ implementation combines stack, bulge, and internal loops in
`internal_CAI`:

```text
internal_CAI(a,b,i,j,x,y,la,lb,xi,yj):
  for lc in la+1 .. min(lb-5, la+MAXLOOP+1):
    ll = lc - la
    c = aa(lc)
    i1 = pos(lc)
    min_ld = max(lc + 4, lb - la + lc - MAXLOOP - 2)

    for ld in reverse(lb-1 .. min_ld):
      lr = lb - ld
      d = aa(ld)
      j1 = pos(ld)

      for xh in codons[c]:
        if c == a and xh != x: continue
        hi = nucleotide(c,xh,i1)

        for xk in codons[d]:
          if d == b and xk != y: continue
          kj = nucleotide(d,xk,j1)
          if hi/kj is not canonical: continue

          if ll == 1 and lr == 1:
            loop = stack_energy(xi,yj,hi,kj)

          else if ll == 1 or lr == 1:
            loop = bulge_energy(xi,yj,hi,kj,max(ll,lr)-1)

          else:
            loop = best_internal_loop_boundary_codons(...)

          candidate = E1[c,d,i1,j1,xh,xk] + loop
          best = min(best, candidate)

  if best improves E1[idx]:
    E1[idx] = best
    E2[idx] = enclosed_mfe + loop_mfe
    E_CAI[idx] = enclosed_cai + loop_cai
```

The enclosed-position loops are `MAXLOOP` bounded. The large DP read is
`E1[c,d,i1,j1,xh,xk]`. For a fixed enclosed site and fixed `xh`, reads over
`xk` are stride-1 because `xk` occupies the flat `y` axis. Advancing `xh` jumps
by `c[d]` elements. Moving to another enclosed position changes `(c,d,i1,j1)`
and usually jumps to another endpoint block.

For true internal loops, C++ enumerates missing boundary codon choices according
to which neighboring nucleotides are already fixed by `i`, `j`, `i1`, and `j1`.
The source implements this as a switch over a bitmask-like code `tt`. This
adds small codon-table loops, but it does not add another large DP table read;
the DP read remains the enclosed `E1` cell.

### 2.7 Multi-Loop Closure Into `E1`

`multi_loop_CAI` closes a paired interval around a multi-loop bifurcation table
`TM`:

```text
multi_loop_CAI(a,b,i,j,x,y):
  best = INF

  if a < b-2 and i == 2 and j == 0:
    for x1 in codons[a+1]:
      for y1 in codons[b-1]:
        inner = (a+1,b-1,0,2,x1,y1)
        best = min(best, TM[inner] + left_cai + right_cai)

  if a < b-1 and i == 2 and j >= 1:
    for x1 in codons[a+1]:
      inner = (a+1,b,0,j-1,x1,y)
      best = min(best, TM[inner] + left_cai)

  if a < b-1 and i <= 1 and j == 0:
    for y1 in codons[b-1]:
      inner = (a,b-1,i+1,2,x,y1)
      best = min(best, TM[inner] + right_cai)

  if a < b and i <= 1 and j >= 1:
    inner = (a,b,i+1,j-1,x,y)
    best = min(best, TM[inner])

  best += AU_closing_pair + ML_closing + ML_intern

  if best improves E1[idx]:
    E1[idx] = best
    E2[idx] = TM2[inner] + closing_mfe
    E_CAI[idx] = TM_CAI[inner] + closing_cai
```

The best locality case is the both-boundaries-cross case: `y1` is innermost,
so reads over `TM[a+1,b-1,0,2,x1,y1]` are stride-1 for a fixed `x1`. The
single-boundary cases also read compact codon rectangles. Endpoint-crossing
still jumps to another packed endpoint block.

### 2.8 Multi-Loop Fragment `M1` And Bifurcation `TM`

`calculate_CAI_M` fills `M1[idx]` and also records the pure bifurcation minimum
in `TM[idx]`:

```text
calculate_CAI_M(a,b,i,j,x,y):
  best = INF

  if closing pair is canonical:
    best = min(best, E1[idx] + ML_intern + AU_closing)

  if i <= 1:
    best = min(best, M1[a,b,i+1,j,x,y] + ML_BASE)

  if i == 2 and a+1 < b:
    for x1 in codons[a+1]:
      best = min(best, M1[a+1,b,0,j,x1,y] + ML_BASE + left_cai)

  if j >= 1:
    best = min(best, M1[a,b,i,j-1,x,y] + ML_BASE)

  if j == 0 and a < b-1:
    for y1 in codons[b-1]:
      best = min(best, M1[a,b-1,i,2,x,y1] + ML_BASE + right_cai)

  bi = INF
  for lc in la+5 .. lb-4:
    c = aa(lc)
    i1 = pos(lc)

    for hx in codons[c]:
      if i1 >= 1:
        left  = M1[a,c,i,i1-1,x,hx]
        right = M1[c,b,i1,j,hx,y]
        bi = min(bi, left + right - shared_cai(hx))

      else:
        for ky in codons[c-1]:
          left  = M1[a,c-1,i,2,x,ky]
          right = M1[c,b,0,j,hx,y]
          bi = min(bi, left + right)

  best = min(best, bi)
  M1[idx] = best
  TM[idx] = min(TM[idx], bi)
```

The same-cell shifts `(i+1,j)` and `(i,j-1)` are shorter nucleotide spans, not
same-span self references under the global `len` loop. In bifurcation case
`i1 >= 1`, the left read changes along its `y` axis as `hx` changes, while the
right read changes along its `x` axis with stride `c[b]`. In the `i1 == 0`
case, the inner `ky` loop makes the left read stride-1 over `ky`; the right read
is fixed for that inner loop.

### 2.9 Exterior Prefix `O`

`calculate_CAI_O` fills only states anchored at the left end:

```text
for len in 1..3n-1:
  a = 0
  i = 0
  b = floor(len / 3)
  j = len mod 3

  for x in codons[0]:
    for y in codons[b]:
      best = INF

      if closing pair is canonical:
        candidate_mfe = E2[0,b,0,j,x,y] + AU_closing
        candidate_cai = E_CAI[0,b,0,j,x,y]
        best = min(best, candidate_mfe + candidate_cai)

      if j >= 1:
        candidate_mfe = Z2[0,b,0,j-1,x,y]
        candidate_cai = Z_CAI[0,b,0,j-1,x,y]
        best = min(best, candidate_mfe + candidate_cai)

      if j == 0:
        if b > 1:
          for q in codons[b-1]:
            candidate_mfe = Z2[0,b-1,0,2,x,q]
            candidate_cai = Z_CAI[0,b-1,0,2,x,q] + right_cai
            best = min(best, candidate_mfe + candidate_cai)
        else:
          candidate_mfe = Z2[0,b-1,0,2,x,x]
          candidate_cai = Z_CAI[0,b-1,0,2,x,x] + right_cai
          best = min(best, candidate_mfe + candidate_cai)

      for lc in 1 .. nt(b,j)-4:
        c = aa(lc)
        i1 = pos(lc)

        for hx in codons[c]:
          hi = nucleotide(c,hx,i1)

          if i1 >= 1:
            left_mfe  = Z2[0,c,0,i1-1,x,hx]
            right_mfe = E2[c,b,i1,j,hx,y]
            left_cai  = Z_CAI[0,c,0,i1-1,x,hx]
            right_cai = E_CAI[c,b,i1,j,hx,y]
            best = min(
              best,
              left_mfe + right_mfe + AU(hi,yj)
                + left_cai + right_cai - shared_cai(hx)
            )

          else:
            for ky in codons[c-1]:
              left_mfe  = Z2[0,c-1,0,2,x,ky]
              right_mfe = E2[c,b,0,j,hx,y]
              left_cai  = Z_CAI[0,c-1,0,2,x,ky]
              right_cai = E_CAI[c,b,0,j,hx,y]
              best = min(
                best,
                left_mfe + right_mfe + AU(hi,yj) + left_cai + right_cai
              )

      O[0,b,0,j,x,y] = best
      OB[0,b,0,j,x,y] = winning_case_tag
      O_bt[0,b,0,j,x,y] = winning_backtrace_payload
      Z2[0,b,0,j,x,y] = winning_mfe_part
      Z_CAI[0,b,0,j,x,y] = winning_cai_part
```

This is not a full all-interval exterior table fill. It is a prefix dynamic
program anchored at `(a=0,i=0)`. The allocated `O` array has slots for every
`(a,b,i,j,x,y)`, but the hot write stream touches only the prefix states needed
to recover the final full-sequence objective. The hot read stream is not the
aggregate `O`/`E1` pair, however. C++ reconstructs each candidate objective
from companion tables: prefix/exterior reads use `Z2 + Z_CAI`, and paired
closure reads use `E2 + E_CAI`. `O` is written as the aggregate result after
the winning candidate is chosen.

### 2.10 Candidate Substrate Construction Side Stream

The C++ hot helpers generally read nucleotide identities directly from codon
and nucleotide tables:

```text
xi = nucleotides[protein[a]][x][i]
yj = nucleotides[protein[b]][y][j]
xi_ = nucleotides[protein[a+1]][x1][0]
```

Hairpin, internal-loop, multi-loop, and exterior candidates pass the needed
nucleotide values into the energy helper or use direct table lookups such as
`rightCodon(...)`. They do not build a full `3n` scratch RNA sequence inside
these recurrence hot helpers. This is a comparison axis: Rust's repeated
candidate-substrate assembly is a side stream that is mostly absent from the
C++ recurrence implementation.

## 3. Complexity And Cache Analysis

### 3.1 Storage Complexity

The packed slot count is:

```math
slots(n, c) = 9(\sum_a c[a])^2
```

A `double` table uses `8 * slots` bytes. An `int` tag table uses `4 * slots`
bytes. The lambda-mode scalar arrays include 12 `double` tables and 3 `int`
tag tables. The four `*_bt` side tables are `vector<vector<int>>`; on the
current 64-bit C++ ABI, the empty vector header is typically 24 bytes, so the
four backtrace side tables have a header footprint around `96 * slots` bytes
before payload allocations.

The first two rows below use the actual GLPE prefix/full sequence codon counts.
The n=1000 row uses the GLPE-78 average synonym count (`S/n = 3.782`) as a
scale reference; worst-case six-codon proteins would be about 2.52x larger.

| `n` | `S = sum c[a]` | One `double` table | Four hot score tables | 12 `double` tables | 3 `int` tags | Scalar arrays | `*_bt` headers |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 81 | 0.45 MiB | 1.80 MiB | 5.41 MiB | 0.68 MiB | 6.08 MiB | 5.41 MiB |
| 78 | 295 | 5.98 MiB | 23.90 MiB | 71.71 MiB | 8.96 MiB | 80.67 MiB | 71.71 MiB |
| 1000 | 3782 | 982.17 MiB | 3.84 GiB | 11.51 GiB | 1.44 GiB | 12.95 GiB | 11.51 GiB |

Interpretation:

- C++ packing makes one GLPE-78 `double` table about 5.98 MiB despite the
  8-byte element type. The equivalent all-six-codon `double` table would be
  about 15.04 MiB.
- The four hot score tables at GLPE-78 are about 23.90 MiB. That is above a
  16 MiB L2-scale cache but below a 40 MiB LLC-scale cache.
- The full scalar allocation at GLPE-78 is larger than Rust's nine `i32` score
  tables, but C++ touches fewer full score tables in the paired-cell reduction
  and only fills prefix `O` states.
- The `vector<vector<int>>` backtrace headers are a large allocation footprint.
  Most recurrence hot paths write only the current payload, but the headers
  still consume memory and cache/TLB reach.

### 3.2 Time Complexity

With codon counts and `MAXLOOP` treated as biological constants:

| Recurrence family | Per fixed state | Whole solve in `n` | Why |
|---|---:|---:|---|
| Hairpin | O(1) | O(n^2) | bounded boundary codon choices |
| Stack | O(1) | O(n^2) | one enclosed pair |
| Bulge | O(1) | O(n^2) | `MAXLOOP`-bounded enclosed positions |
| Internal | O(1) | O(n^2) | `MAXLOOP`-bounded positions and boundary codons |
| Multi closure into `E1` | O(1) | O(n^2) | at most two boundary codon loops |
| `M1` shift/branch cases | O(1) | O(n^2) | shorter-span shifts and branch close |
| `M1`/`TM` bifurcation | O(span) | O(n^3) | split nucleotide `lc` across interval |
| Prefix `O` | O(prefix length) | O(n^2) | only `(a=0,i=0)` prefixes are filled |

The whole scalar solve remains O(n^3) because `M1/TM` bifurcation is cubic.
The exterior recurrence is not cubic in C++ because it is not filled for every
interval.

### 3.3 Access-Pattern Summary

| Recurrence | Writes | DP reads | Locality summary |
|---|---|---|---|
| Hairpin | `E1/E2/E_CAI/E_bt[idx]` | none | current-index writes; codon/energy side tables only |
| Stack | `E1/E2/E_CAI/E_bt[idx]` | `E1[inner]` | enclosed reads stride-1 over inner right codon |
| Bulge | `E1/E2/E_CAI/E_bt[idx]` | `E1[inner]` | bounded positions; each codon rectangle is compact |
| Internal | `E1/E2/E_CAI/E_bt[idx]` | `E1[inner]` | same enclosed read plus boundary codon enumeration |
| Multi closure | `E1/E2/E_CAI/E_bt[idx]` | `TM[inner]` | boundary-cross reads into compact `TM` rectangles |
| `M1` fragment | `M1/M2/M_CAI/M_bt[idx]` | `E1`, `M1` | shifts nearby by nucleotide span; bifurcation reads two `M1` cells |
| `TM` bifurcation | `TM/TM2/TM_CAI/TM_bt[idx]` | `M1` | same bifurcation stream as `M1`, saved separately |
| Prefix `O` | `O/OB/Z2/Z_CAI/O_bt[idx]` | `Z2`, `Z_CAI`, `E2`, `E_CAI` | prefix-only left companion streams plus paired right companion streams |

The strongest cache-risk zone is `M1/TM` bifurcation: it reads two `M1` cells
per split candidate and the two cells move through different endpoint blocks.
The prefix `O` bifurcation has the same split shape, but each side is a
two-table companion stream (`Z2/Z_CAI` on the prefix side and `E2/E_CAI` on the
paired side), and it is filled only for prefixes rather than all intervals.

### 3.4 Reuse-Distance Notes

| Pattern | Stride / reuse |
|---|---|
| Current table writes over `y` | stride-1 |
| Current table writes over `x` | stride `c[b]`, smaller than or equal to 6 |
| Same-pair `(i+1,j)` read | offset `+3c[a]c[b]` elements within the pair |
| Same-pair `(i,j-1)` read | offset `-c[a]c[b]` elements within the pair |
| Enclosed `E1` reads over inner right codon | stride-1 |
| Enclosed `E1` reads over inner left codon | stride `c[d]` |
| Boundary-cross read changing `b` | jump by one packed endpoint block, size `9c[a]c[b']` |
| Boundary-cross read changing `a` | jump by a row-scale packed block, near `9c[a]S` elements |
| `M1` bifurcation, `i1 >= 1`, left side over `hx` | left read advances over `y`, stride-1 |
| `M1` bifurcation, `i1 >= 1`, right side over `hx` | right read advances over `x`, stride `c[b]` |
| `M1` bifurcation, `i1 == 0`, inner `ky` | left read stride-1; right read reused across `ky` |
| Prefix `O` closure | `E2/E_CAI` at the same prefix-pair index, then aggregate `O` write |
| Prefix `O` shift/cross reads | `Z2/Z_CAI` prefix companion reads; aggregate `O` is not read |
| Prefix `O` bifurcation | left `Z2/Z_CAI` prefix streams plus right `E2/E_CAI` paired streams |
| Paired-cell motif reduction | no cross-table motif reduction; helpers update `E1` directly |

The useful distinction for the comparison is:

- **Good local streams:** current writes over `y`, enclosed-pair reads over
  inner right codons, and many boundary codon loops inside packed rectangles.
- **Potentially expensive streams:** `M1/TM` bifurcation endpoint jumps and the
  four-companion-table prefix-`O` bifurcation stream.
- **Work avoided relative to a full dense layout:** no max-codon padding, no
  separate full motif score tables, and no all-interval `O` fill.

## 4. Open Questions For The Comparison

These are comparison prompts for the Rust-vs-C++ diff, not implementation
recommendations by themselves.

### 4.1 How Much Of The n=78 Gap Is Max-Codon Padding?

C++ stores `c[a] * c[b]` codon rectangles. Rust stores `m * m` for every
endpoint pair. For GLPE-78, `S = 295`, so the packed endpoint-codon slot count
is:

```text
C++ slots = 9 * 295^2 = 783,225
Rust max-codon slots = 9 * 78^2 * 6^2 = 1,971,216
```

The C++ scalar score table has about 39.7 percent as many slots as the Rust
max-codon table, before accounting for element width or the different number
of tables. This is a first structural candidate for the n=20 to n=78 widening.

### 4.2 Is Rust Paying For Full Motif Tables That C++ Avoids?

C++ updates `E1` directly from hairpin, internal/stack/bulge, and multi-loop
closure helpers. Rust keeps separate full `E_h`, `E_s`, `E_b`, `E_i`, and
`E_m` score tables and reduces them into `E`.

A comparison should separate two costs:

- the memory footprint of keeping the motif tables resident, and
- the write/read traffic of filling and reducing them.

The C++ reference shows that a single aggregate paired table plus compact
winner metadata is sufficient for its traceback contract.

### 4.3 Is The All-Interval Exterior Fill Necessary In Rust?

C++ `O` is a prefix DP anchored at `(a=0,i=0)`. It allocates full `O` storage
but only fills states needed by the final exterior solve. If Rust fills `O` for
every `(a,b,s,t,x,y)`, that is a structural O(n^3) exterior recurrence where
C++ pays O(n^2) for exterior work.

This prefix-only conclusion survives the companion-table correction, but the
cache model is wider than an `O/E1` aggregate-table read model: C++ exterior
candidates read `Z2/Z_CAI` for prefix subproblems and `E2/E_CAI` for paired
subproblems before writing aggregate `O`. A comparison should verify whether Rust's more
general `O` states are needed for PORD/hull functionality or whether they are a
carry-over from a more general recurrence shape, while counting the C++
reference as a four-companion-stream exterior read rather than a two-aggregate-
stream read.

### 4.4 Does Per-Cell E-Then-M Fusion Improve Locality?

C++ fills `E1` for a state and immediately calls `calculate_CAI_M` for the same
state. `calculate_CAI_M` can then read `E1[idx]` while it is still local.

A comparison should compare this with Rust's phase order. If Rust fills an entire table
phase before a fragment phase reads it, the algorithm may be increasing reuse
distance between paired-cell aggregate writes and multi-fragment branch-close
reads.

### 4.5 Are Backtrace Side Tables A Red Herring For The Perf Gap?

C++ allocates large `vector<vector<int>>` side tables, and the empty headers at
GLPE-78 are roughly the size of the 12 double companion tables. Despite that,
C++ is faster at sweep scale. The likely explanation is that header footprint
is not the dominant hot stream, or that the other structural savings dominate.

A comparison should keep backtrace metadata in the memory accounting, but it should not
assume total allocation size alone explains wall-clock behavior.

### 4.6 Is Candidate-Substrate Assembly A Rust-Only Side Stream?

The C++ recurrence helpers mostly read needed nucleotides directly from
`nucleotides`, `protein`, and codon-choice indices. Rust's pseudocode calls out
repeated full-substrate assembly for several candidate paths. A comparison should compare
that side stream mechanically: if the Rust paths rebuild a `3n` scratch
sequence where C++ does direct indexed reads, the cost is not a DP-table
footprint issue, but it can still contribute repeated candidate-loop work.
