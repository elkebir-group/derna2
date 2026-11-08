//! A stop fired inside a round reaches the waves that come after it.
//!
//! The search tests `outside::lazy::walks_stopped` at the top of each round, so
//! a stop firing between two waves of one round is already behind that test.
//! Every wave left then pays its region's fold cut and its window's outside
//! passes before a stop can reach a walk, which is resident memory a caller
//! watching it has already asked the search to give back. `derna front
//! --max-rss-gb` sets exactly this latch from a watcher thread; the latch is
//! set here from the search's own report instead, so the wave it fires between
//! is fixed rather than left to when a memory reading crosses a limit.
//!
//! The control arm is the same search with no stop at all: it says how many of
//! the round's regions there were to lose, so a run that loses none is not
//! reported as a pass on a round that had only one wave in it. The round needs
//! more than one open region to have a second wave, and a dense constrained
//! front leaves only the top band open — every stretch between two adjacent
//! seeds holds nothing. The open boxes here are what makes the stretches
//! between those seeds walkable; their energy floor is far below the front, so
//! they bound nothing.
//!
//! This test has its own binary on purpose. `stop_walks` sets a process-wide
//! latch with no reset, so a test that calls it poisons every other test that
//! shares its binary. Nothing else belongs in this file.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use derna_lib::modular_front::{
    modular_front_resuming, ModularEvent, ModularFront, OpenBox, SearchPolicy, TopSearch,
};
use derna_lib::outside::lazy::{stop_walks, walks_stopped};
use derna_lib::{
    design_front_walk_with_utr, CodonUsage, Enforcement, FrontPoint, Params, RegionConstraints,
    UtrSpec, WalkOptions,
};

const EPSILON: f64 = 1.0 / (1u32 << 23) as f64;
const PROTEIN: &str = "MSLRLSPRAV";
const UTR5: &str = "GGGAAAUAAGAGAG";
const UTR3: &str = "UGAUAAUAGGCUGG";
const CONSTRAINTS: fn() -> RegionConstraints =
    || RegionConstraints::symmetric_isolation(Enforcement::Forbid);
/// Long enough for the first round to reach every region it opened, and short
/// enough that the control arm is not the test's cost.
const SECONDS: f64 = 20.0;

/// Resident memory now, in KiB, where `/proc` reports it.
fn rss_kb() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * 4)
}

/// The molecule's constrained front, which the search starts from.
fn constrained_front(params: &Params, usage: &CodonUsage) -> Vec<FrontPoint> {
    design_front_walk_with_utr(
        params,
        PROTEIN,
        usage,
        &UtrSpec::Fixed(UTR5.into()),
        &UtrSpec::Fixed(UTR3.into()),
        CONSTRAINTS(),
        EPSILON,
        1.0 - EPSILON,
        WalkOptions::default(),
        1,
    )
    .expect("the constrained front")
    .front
}

/// A box over every other pair of adjacent front points, floored far below the
/// front: the stretch between that pair is then walked rather than passed over
/// as holding nothing, which is what gives the first round several regions.
fn boxes_between(constrained: &[FrontPoint]) -> Vec<OpenBox> {
    constrained
        .windows(2)
        .step_by(2)
        .map(|w| OpenBox {
            mfe_lo_kcal_mol: -1000.0,
            cai_log_lo: w[0].cai_log,
            cai_log_hi: w[1].cai_log,
        })
        .collect()
}

/// The modular search of the molecule, reporting what it settles to `record`.
/// One thread, so a round's waves are its regions one at a time and the wave
/// that triggers a stop is the only one that can have been walking when it
/// fired. The top band goes first, which is the wave that reports.
fn search(
    params: &Params,
    usage: &CodonUsage,
    (constrained, boxes): (&[FrontPoint], &[OpenBox]),
    record: Option<&(dyn Fn(ModularEvent<'_>) + Sync)>,
) -> ModularFront {
    modular_front_resuming(
        params,
        PROTEIN,
        usage,
        (UTR5, UTR3),
        CONSTRAINTS(),
        (constrained, boxes),
        (SECONDS, 1),
        (None, record),
        (TopSearch::Band, 0.0, SearchPolicy::default()),
    )
    .expect("the search returns")
}

/// How many of the first round's regions were walked.
fn walked_in_the_first_round(front: &ModularFront) -> usize {
    front
        .stretches
        .iter()
        .filter(|s| s.round == 0 && s.walked)
        .count()
}

#[test]
fn a_stop_inside_a_round_reaches_the_waves_after_it() {
    let params = Params::load_default().expect("params");
    let usage = CodonUsage::load_default().expect("usage");
    let constrained = constrained_front(&params, &usage);
    let boxes = boxes_between(&constrained);

    // The control arm, before the latch is ever set: how many regions the
    // first round walks when nothing stops it. Without this the stopped arm
    // below is satisfied by a round that only ever had one wave.
    let control = search(&params, &usage, (&constrained, &boxes), None);
    let control_waves = walked_in_the_first_round(&control);
    assert!(
        control_waves >= 2,
        "the first round walked {control_waves} regions, so no wave could be lost to a stop"
    );

    // The stop fires as the first wave reports what it settled, which is after
    // that wave's walk and before the next wave's setup — the one place the
    // round's own test at the top cannot see.
    let fired = AtomicBool::new(false);
    let rss_at_stop = AtomicU64::new(0);
    let trigger = |event: ModularEvent<'_>| {
        let reports = matches!(
            event,
            ModularEvent::Found { .. } | ModularEvent::Band { .. }
        );
        if reports && !fired.swap(true, Ordering::Relaxed) {
            rss_at_stop.store(rss_kb().unwrap_or(0), Ordering::Relaxed);
            stop_walks();
        }
    };
    let stopped = search(&params, &usage, (&constrained, &boxes), Some(&trigger));
    let after = rss_kb().unwrap_or(0);
    let at_stop = rss_at_stop.load(Ordering::Relaxed);

    assert!(fired.load(Ordering::Relaxed), "the stop never fired");
    assert!(walks_stopped(), "the latch did not take");
    assert!(
        !stopped.complete,
        "a search stopped in its first round returned a complete front"
    );
    let stopped_waves = walked_in_the_first_round(&stopped);
    // The magnitude beside the count, so a run that walks nothing past the stop
    // still says what the walks it skipped were worth in the resource the stop
    // was watching.
    eprintln!(
        "walked past the stop: {} of {control_waves}; resident {at_stop} KiB at the stop, \
         {after} KiB on return",
        stopped_waves.saturating_sub(1),
    );
    assert_eq!(
        stopped_waves, 1,
        "regions of the first round were walked past the stop, of the {control_waves} \
         the control walked"
    );
    // The waves left pay their setup in resident memory, which is what the
    // caller asked the search to give back. With none of them started there is
    // nothing left to allocate, so the reading at the stop is the run's. This
    // is a backstop, not the discriminating assertion: on a molecule this small
    // the walks that used to run past the stop were worth less than its slack.
    if at_stop > 0 && after > 0 {
        assert!(
            after <= at_stop + 64 * 1024,
            "resident memory went from {at_stop} KiB at the stop to {after} KiB"
        );
    }
}
