# Vienna's `fun_zip_add_min_default` on Apple Silicon

This document captures the ARM64 disassembly of ViennaRNA's
`fun_zip_add_min_default` (the inner kernel for the multi-loop
bifurcation cubic), as compiled into `/opt/homebrew/lib/libRNA.a`. It is
the falsification target for `rna_mfe`'s flat scalar
`zip_add_min_inf` kernel: any Rust kernel that lands meaningfully more
ARM64 instructions per iteration than this reference is not done.

## Platform

| Field | Value |
|---|---|
| Host arch | aarch64-apple-darwin (Apple M-series) |
| ViennaRNA tag | v2.7.0 |
| Source | ViennaRNA v2.7.0 `src/ViennaRNA/utils/higher_order_functions.c` lines 124–140 |
| Binary | `/opt/homebrew/lib/libRNA.a` (Homebrew-provided build) |
| Tool | `/usr/bin/otool -tvV` |

## Why this is the kernel

Vienna dispatches `vrna_fun_zip_add_min` via a function pointer
(`higher_order_functions.c:55`); the dispatcher (line 93) checks
`vrna_cpu_simd_capabilities()` for AVX-512 / SSE4.1 and falls back to
`fun_zip_add_min_default` if neither is available. Both SIMD paths are
**x86-only** (`#if VRNA_WITH_SIMD_AVX512` / `#if VRNA_WITH_SIMD_SSE41`).
On Apple Silicon there is no NEON path; the dispatcher unconditionally
selects the scalar default. So whatever throughput Vienna achieves on
M-series for the multi-loop bifurcation, it comes from this scalar
loop, not from SIMD.

## Disassembly

```
_fun_zip_add_min_default:
0000000000000014  mov   w9, #0x9680
0000000000000018  movk  w9, #0x98, lsl #16        ; w9 = INF (0x989680 = 10_000_000)
000000000000001c  cmp   w2, #0x1
0000000000000020  b.lt  0x64                       ; skip if count < 1
0000000000000024  mov   w10, w2                    ; w10 = remaining count
0000000000000028  mov   w8, #0x9680
000000000000002c  movk  w8, #0x98, lsl #16         ; w8 = decomp = INF
;     <-- loop top, body starts here -->
0000000000000030  ldr   w11, [x0], #0x4            ; w11 = e1[i], post-inc x0
0000000000000034  cmp   w11, w9                    ; e1[i] == INF?
0000000000000038  b.eq  0x54                       ; if INF, skip body
000000000000003c  ldr   w12, [x1]                  ; w12 = e2[i]
0000000000000040  add   w11, w12, w11              ; w11 = e1[i] + e2[i]
0000000000000044  cmp   w8, w11                    ; decomp < sum?
0000000000000048  csel  w11, w8, w11, lt           ; w11 = min(decomp, sum)
000000000000004c  cmp   w12, w9                    ; e2[i] == INF?
0000000000000050  csel  w8, w8, w11, eq            ; update decomp only if e2 != INF
0000000000000054  add   x1, x1, #0x4               ; x1 += 4 (advance e2)
0000000000000058  subs  x10, x10, #0x1             ; --remaining
000000000000005c  b.ne  0x30                       ; loop back
;     <-- loop body ends -->
0000000000000060  b     0x6c                       ; jump to epilogue
0000000000000064  mov   w8, #0x9680
0000000000000068  movk  w8, #0x98, lsl #16         ; (count<1 path) w8 = INF
000000000000006c  mov   x0, x8
0000000000000070  ret
```

## Loop body instruction count

The loop body — the path executed once per iteration — is the **eleven**
instructions from address `0x30` to `0x5c`:

| # | mnemonic | role |
|---|----------|------|
| 1 | `ldr w11, [x0], #0x4` | load `e1[i]` and advance pointer |
| 2 | `cmp w11, w9` | compare `e1[i]` to INF |
| 3 | `b.eq 0x54` | skip body if INF |
| 4 | `ldr w12, [x1]` | load `e2[i]` |
| 5 | `add w11, w12, w11` | sum |
| 6 | `cmp w8, w11` | compare `decomp` to `sum` |
| 7 | `csel w11, w8, w11, lt` | branchless `min(decomp, sum)` |
| 8 | `cmp w12, w9` | compare `e2[i]` to INF |
| 9 | `csel w8, w8, w11, eq` | branchless: keep old `decomp` if `e2` was INF, else take new min |
| 10 | `add x1, x1, #0x4` | advance `e2` pointer |
| 11 | `subs x10, x10, #0x1; b.ne 0x30` | decrement counter, branch (counted as one for budgeting purposes — `subs+b.ne` is the standard ARM64 loop close) |

The actual instruction-cycle count is 12 (`subs` and `b.ne` are
separate); the **shape** is two loads, one add, two compares, two
`csel`s for branchless min, one branch on the INF early-skip, and the
loop-close.

Notable shape characteristics:

- **No bounds-checking** — clang trusts the C `for (i = 0; i < count; i++)`
  loop variable.
- **Two `csel` instructions** implement the INF-aware min in a
  branchless form (besides the `b.eq` early-skip on `e1[i] == INF`,
  which can be predicted reliably for finite-cell-dense slices).
- **No SIMD, no unrolling.** clang did not auto-vectorize this either.

## Falsification target for `rna_mfe::zip_add_min_inf`

The Rust kernel in `packages/rna-mfe/src/dp/multi.rs` (or wherever it
lands) must produce ARM64 codegen with:

- **≤15 instructions per iteration** in the inner loop body.
- **No `bl` / call into a panic helper** (i.e., no bounds checks per
  iteration).
- **At least one `csel`** (or equivalent branchless min) for the
  finite-pair update.
- **Two `ldr` loads** per iteration.

If the Rust codegen lands ≥20 instructions or has visible bounds-check
calls, the kernel is not done — fix it before benchmarking.

## Reproduction

```bash
/usr/bin/otool -tvV /opt/homebrew/lib/libRNA.a 2>/dev/null \
  | awk '/_fun_zip_add_min_default:/{flag=1} flag{print; if(flag>1 && /^_[a-zA-Z]/) exit; flag++}'
```
