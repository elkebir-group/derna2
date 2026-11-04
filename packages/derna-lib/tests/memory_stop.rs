//! A modular search whose walks were stopped must return, even
//! with no time budget.
//!
//! A caller watching memory stops the walks rather than the clock
//! (`outside::lazy::stop_walks`, which `derna front --max-rss-gb` calls from
//! its watcher thread). A stopped walk returns at once and finds nothing, so
//! a round loop whose only exits are "no region is open" and "the clock ran
//! out" rounds forever when the clock is `--max-secs`-less — which is what
//! `--max-rss-gb` alone used to do: a 40-residue cell that completes in 215 s
//! ran 1,725 s at 100% CPU before it was killed.
//!
//! This test has its own binary on purpose. `stop_walks` sets a process-wide
//! latch with no reset, so a test that calls it poisons every other test that
//! shares its binary. Nothing else belongs in this file.
//!
//! Note the failure mode: before the fix this test does not fail, it hangs.
//! That is why it is alone here — it can only hang itself.

use derna_lib::modular_front::modular_front;
use derna_lib::outside::lazy::{stop_walks, walks_stopped};
use derna_lib::{
    design_front_walk_with_utr, CodonUsage, Enforcement, Params, RegionConstraints, UtrSpec,
    WalkOptions,
};

const EPSILON: f64 = 1.0 / (1u32 << 23) as f64;

#[test]
fn a_search_whose_walks_were_stopped_returns_without_a_time_budget() {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let (protein, utr5, utr3) = ("MSLRLSPRAV", "GGGAAAUAAGAGAG", "UGAUAAUAGGCUGG");
    let table = RegionConstraints::symmetric_isolation(Enforcement::Forbid);
    let walked = design_front_walk_with_utr(
        &params,
        protein,
        &usage,
        &UtrSpec::Fixed(utr5.into()),
        &UtrSpec::Fixed(utr3.into()),
        table,
        EPSILON,
        1.0 - EPSILON,
        WalkOptions::default(),
        1,
    )
    .expect("the constrained front");

    stop_walks();
    assert!(walks_stopped(), "the latch did not take");

    // No time budget at all: termination has to come from the stop.
    let got = modular_front(
        &params,
        protein,
        &usage,
        utr5,
        utr3,
        table,
        &walked.front,
        f64::INFINITY,
        1,
    )
    .expect("the search returns");

    // It returned, which is the point. What it returns is a partial answer
    // that says so: the walks found nothing, so whatever the seeds did not
    // settle is still open.
    assert!(
        !got.complete || got.open.is_empty(),
        "a complete front must leave nothing open"
    );
}
