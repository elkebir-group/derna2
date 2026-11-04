//! The `derna` binary: everything it does lives in `derna_cli`, so the
//! command surface can be tested without a process.
//!
//! The two things that cannot be: a refusal reaches the caller on **stderr**
//! with an `error:` prefix, and any failure exits **1**. Both are part of the
//! contract (`docs/guides/derna-cli.md`) and are tested by running this
//! binary.

/// `--features jemalloc` builds the binary on jemalloc; see that feature in
/// `Cargo.toml` for what it changes.
#[cfg(feature = "jemalloc")]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() {
    #[cfg(feature = "jemalloc")]
    derna_lib::motif_front::allocator_is_jemalloc();
    if let Err(e) = derna_cli::run(std::env::args().skip(1).collect()) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
