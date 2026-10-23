//! Codegen-verification harness for `dp::zip_add_min_inf`.
//!
//! The kernel is `#[inline]` for production hot-path inlining, so its
//! body never appears as a standalone symbol in `librna_mfe.rlib`
//! compiled with `lto = "thin"`. This example provides a
//! `#[inline(never)]` marker wrapper that anchors the kernel as the
//! body of an outlined function in `target/release/examples/kernel_disasm`,
//! making it disassemblable for the codegen falsification gate
//! (`packages/rna-mfe/docs/vienna-kernel-disasm.md` § Falsification target).
//!
//! Verification recipe (see `packages/rna-mfe/docs/vienna-kernel-disasm.md`
//! for the Vienna reference):
//!
//! ```bash
//! cargo build --release --workspace --examples
//! /usr/bin/objdump -d target/release/examples/kernel_disasm \
//!   | awk '/<zip_add_min_inf_marker>:/{flag=1} flag{print; if(/^[0-9a-f]+ <_/&&!/zip_add_min_inf_marker/)exit}'
//! ```

use rna_mfe::dp::zip_add_min_inf;

#[inline(never)]
#[no_mangle]
pub fn zip_add_min_inf_marker(a: &[i32], b: &[i32]) -> i32 {
    zip_add_min_inf(a, b)
}

fn main() {
    let inf = 10_000_000;
    let a = [1, 2, inf, 4, 5, 6, 7, 8];
    let b = [10, 20, 30, inf, 50, 60, 70, 80];
    let result = zip_add_min_inf_marker(&a, &b);
    println!("{result}");
}
