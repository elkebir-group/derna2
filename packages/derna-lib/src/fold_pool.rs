//! Worker threads kept for a whole front fold.
//!
//! The fold fans out up to four times per position. Spawning fresh threads
//! for each fan-out costs a thread start and exit every time, and under a
//! thread-caching allocator each new thread also starts with a cold cache and
//! hands its cache back on exit. A pool started once per fold keeps its
//! threads, and their caches, warm across every fan-out.
//!
//! [`Pool::run`] hands one job to the first `n` workers and blocks until each
//! has returned. That blocking is what makes it sound to run a job that
//! borrows the caller's stack: no worker can still hold the job once `run`
//! returns.
//!
//! [`Pool::run_joined`] is the form for a job whose participants take work
//! off a shared counter: the caller runs it too, and a worker that has not
//! been scheduled by the time the caller's own share is done is released
//! without running it. On a loaded host a woken worker can wait milliseconds
//! for a core; with the caller working, the job never waits for it.

use std::panic::{self, AssertUnwindSafe};
use std::sync::{Condvar, Mutex};
use std::thread::Scope;

/// A job as the workers hold it: borrowed from the caller of [`Pool::run`],
/// with its lifetime erased.
type Job = &'static (dyn Fn(usize) + Sync);

struct State {
    /// Bumped once per [`Pool::run`]; a worker runs each epoch at most once.
    epoch: u64,
    job: Option<Job>,
    /// Workers `0..wanted` run the current job.
    wanted: usize,
    /// Of those, the ones still running it.
    running: usize,
    /// The current job came from [`Pool::run_joined`]: a worker counts itself
    /// into `running` when it takes the job, and the caller retracts the job
    /// once its own share is done.
    joined: bool,
    /// A worker's job panicked during this epoch.
    panicked: bool,
    shutdown: bool,
}

/// Worker threads that outlive each fan-out, inside one [`std::thread::scope`].
pub(crate) struct Pool {
    workers: usize,
    state: Mutex<State>,
    start: Condvar,
    done: Condvar,
}

/// Stops the pool when dropped, so the scope can join its workers even when
/// the body unwinds.
struct StopOnDrop<'p>(&'p Pool);

impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        self.0.stop();
    }
}

impl Pool {
    fn new(workers: usize) -> Self {
        Pool {
            workers,
            state: Mutex::new(State {
                epoch: 0,
                job: None,
                wanted: 0,
                running: 0,
                joined: false,
                panicked: false,
                shutdown: false,
            }),
            start: Condvar::new(),
            done: Condvar::new(),
        }
    }

    /// Run `body` with a pool of `workers` threads, which are joined when
    /// `body` returns or unwinds.
    pub(crate) fn scoped<R>(workers: usize, body: impl FnOnce(&Pool) -> R) -> R {
        let pool = Pool::new(workers);
        std::thread::scope(|scope| {
            pool.spawn(scope);
            let _stop = StopOnDrop(&pool);
            body(&pool)
        })
    }

    fn spawn<'scope, 'env>(&'env self, scope: &'scope Scope<'scope, 'env>) {
        for index in 0..self.workers {
            crate::task_clock::spawn(scope, move || self.work(index));
        }
    }

    /// Run `job(i)` on workers `i = 0..n` (at most the pool's size) and wait
    /// until every one has returned. A panic in any of them is re-raised here.
    pub(crate) fn run(&self, n: usize, job: &(dyn Fn(usize) + Sync)) {
        let n = n.min(self.workers);
        if n == 0 {
            return;
        }
        // SAFETY: the erased reference is stored only in `State::job` for this
        // epoch. Every worker that reads it runs it before decrementing
        // `running`, and this call does not return until `running` is zero and
        // the slot has been cleared, so no worker can reach `job` after the
        // borrow it came from ends.
        let erased: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(job) };
        let mut state = self.state.lock().unwrap();
        state.epoch += 1;
        state.job = Some(erased);
        state.wanted = n;
        state.running = n;
        state.joined = false;
        state.panicked = false;
        self.start.notify_all();
        while state.running > 0 {
            state = self.done.wait(state).unwrap();
        }
        state.job = None;
        let panicked = state.panicked;
        drop(state);
        assert!(!panicked, "a front-fold worker panicked");
    }

    /// Run `job(0)` on the caller and `job(i)` on worker `i − 1` for
    /// `i = 1..n` (at most the pool's size plus one), counting only the workers
    /// that take the job before the caller's own `job(0)` returns. The caller
    /// then retracts the job and waits for those workers alone; a worker
    /// woken later finds no job and goes back to sleep. Every participant
    /// index is therefore distinct, but which of `1..n` run is decided by the
    /// scheduler, so `job` must not depend on any of them running. A panic in
    /// any participant is re-raised here.
    pub(crate) fn run_joined(&self, n: usize, job: &(dyn Fn(usize) + Sync)) {
        let helpers = n.saturating_sub(1).min(self.workers);
        if helpers == 0 {
            if n > 0 {
                job(0);
            }
            return;
        }
        // SAFETY: as in `run`. A worker reads the erased reference only while
        // it is set, and counts itself into `running` under the same lock;
        // this call clears the slot and then waits for `running` to reach
        // zero, so no worker can reach `job` after the borrow ends.
        let erased: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(job) };
        {
            let mut state = self.state.lock().unwrap();
            state.epoch += 1;
            state.job = Some(erased);
            state.wanted = helpers;
            state.running = 0;
            state.joined = true;
            state.panicked = false;
        }
        self.start.notify_all();
        let mine = panic::catch_unwind(AssertUnwindSafe(|| job(0)));
        let mut state = self.state.lock().unwrap();
        state.job = None;
        while state.running > 0 {
            state = self.done.wait(state).unwrap();
        }
        let panicked = state.panicked;
        drop(state);
        if let Err(cause) = mine {
            panic::resume_unwind(cause);
        }
        assert!(!panicked, "a pool worker panicked");
    }

    /// Release the workers so the enclosing scope can join them.
    fn stop(&self) {
        self.state.lock().unwrap().shutdown = true;
        self.start.notify_all();
    }

    fn work(&self, index: usize) {
        let mut seen = 0;
        loop {
            let (job, at) = {
                let mut state = self.state.lock().unwrap();
                loop {
                    if state.shutdown {
                        return;
                    }
                    if state.epoch != seen {
                        seen = state.epoch;
                        if index < state.wanted {
                            if !state.joined {
                                break (state.job.expect("a job is set while workers run"), index);
                            }
                            // A joined job may already be retracted.
                            if let Some(job) = state.job {
                                state.running += 1;
                                break (job, index + 1);
                            }
                        }
                    }
                    state = self.start.wait(state).unwrap();
                }
            };
            let ok = panic::catch_unwind(AssertUnwindSafe(|| job(at))).is_ok();
            let mut state = self.state.lock().unwrap();
            state.panicked |= !ok;
            state.running -= 1;
            if state.running == 0 {
                self.done.notify_one();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn each_run_reaches_exactly_the_workers_asked_for() {
        Pool::scoped(4, |pool| {
            for n in [1, 3, 4, 9, 2] {
                let hits: Vec<AtomicUsize> = (0..4).map(|_| AtomicUsize::new(0)).collect();
                pool.run(n, &|i| {
                    hits[i].fetch_add(1, Ordering::Relaxed);
                });
                let got: Vec<usize> = hits.iter().map(|h| h.load(Ordering::Relaxed)).collect();
                let want: Vec<usize> = (0..4).map(|i| usize::from(i < n.min(4))).collect();
                assert_eq!(got, want, "n = {n}");
            }
        });
    }

    #[test]
    fn a_job_may_borrow_the_callers_stack() {
        Pool::scoped(3, |pool| {
            for round in 0..50 {
                let local = [round; 3];
                let sum = AtomicUsize::new(0);
                pool.run(3, &|i| {
                    sum.fetch_add(local[i], Ordering::Relaxed);
                });
                assert_eq!(sum.load(Ordering::Relaxed), 3 * round);
            }
        });
    }

    #[test]
    fn a_joined_run_finishes_a_shared_counter_and_never_repeats_an_index() {
        Pool::scoped(3, |pool| {
            for round in 0..200 {
                let total = 1000 + round;
                let next = AtomicUsize::new(0);
                let done = AtomicUsize::new(0);
                let seen: Vec<AtomicUsize> = (0..4).map(|_| AtomicUsize::new(0)).collect();
                let local = vec![1usize; total];
                pool.run_joined(4, &|i| {
                    seen[i].fetch_add(1, Ordering::Relaxed);
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        if k >= total {
                            break;
                        }
                        done.fetch_add(local[k], Ordering::Relaxed);
                    }
                });
                assert_eq!(done.load(Ordering::Relaxed), total);
                assert_eq!(seen[0].load(Ordering::Relaxed), 1, "the caller always runs");
                assert!(seen.iter().all(|s| s.load(Ordering::Relaxed) <= 1));
            }
        });
    }

    #[test]
    fn a_joined_run_reraises_a_panic_and_the_pool_survives() {
        Pool::scoped(2, |pool| {
            let caught = panic::catch_unwind(AssertUnwindSafe(|| {
                pool.run_joined(3, &|i| assert!(i != 0, "the caller fails"));
            }));
            assert!(caught.is_err());
            let ran = AtomicUsize::new(0);
            pool.run(2, &|_| {
                ran.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(ran.load(Ordering::Relaxed), 2);
        });
    }

    #[test]
    fn a_panicking_job_is_reraised_and_the_pool_survives() {
        Pool::scoped(2, |pool| {
            let caught = panic::catch_unwind(AssertUnwindSafe(|| {
                pool.run(2, &|i| assert!(i != 1, "worker one fails"));
            }));
            assert!(caught.is_err());
            let ran = AtomicUsize::new(0);
            pool.run(2, &|_| {
                ran.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(ran.load(Ordering::Relaxed), 2);
        });
    }
}
