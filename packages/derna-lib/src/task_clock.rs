//! CPU time and memory charged to the task that ran them.
//!
//! A walk's queue runs several tasks at once, each fanning out onto threads
//! of its own, so neither a wall clock nor the process's CPU clock says what
//! one task cost. A [`TaskClock`] charges a task the CPU time of the thread
//! that started it plus that of every thread spawned, directly or through
//! further spawns, by [`spawn`] while it ran: each such thread inherits the
//! charge of the thread that spawned it and adds its own CPU time to it when
//! it ends. Scoped threads end before their scope does, so every thread a
//! task spawned has been charged by the time the task returns. Thread CPU
//! time, unlike wall time, is not inflated by other tenants of the host.
//!
//! [`rss_kb`] and [`peak_rss_kb`] read the process's resident memory, now and
//! at its highest, so a task's end can mark what memory it left behind.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{Builder, Scope, ScopedJoinHandle};

thread_local! {
    /// Where this thread's CPU time is charged when it ends, if anywhere.
    static CHARGE: RefCell<Option<Arc<AtomicU64>>> = const { RefCell::new(None) };
}

/// This thread's CPU time so far, in nanoseconds.
pub fn thread_cpu_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable `timespec` for the call's duration.
    let ok = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    if ok != 0 {
        return 0;
    }
    u64::try_from(ts.tv_sec).unwrap_or(0) * 1_000_000_000 + u64::try_from(ts.tv_nsec).unwrap_or(0)
}

/// One thread's CPU time since [`ThreadCpu::here`], readable from any thread
/// while that thread runs: what a watcher reads of a task in progress, whose
/// spawned threads pay their time only when they end.
#[derive(Debug, Clone, Copy)]
pub struct ThreadCpu {
    clock: libc::clockid_t,
    start: u64,
}

impl ThreadCpu {
    /// Whether a thread's CPU time can be read from another thread here.
    ///
    /// It takes a clock id that names a *particular* thread, which
    /// `pthread_getcpuclockid` returns and which not every platform has —
    /// macOS does not. `CLOCK_THREAD_CPUTIME_ID` is not a substitute: it
    /// names whichever thread reads it, so a watcher asking for the solving
    /// thread's CPU time would be told its own, and a watcher that only
    /// sleeps reads nearly zero. That is a wrong number rather than a missing
    /// one, so callers test this and do without the reading instead.
    pub const AVAILABLE: bool = cfg!(all(target_os = "linux", target_env = "gnu"));

    /// The calling thread's CPU clock, from now.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    pub fn here() -> Self {
        let mut clock: libc::clockid_t = libc::CLOCK_THREAD_CPUTIME_ID;
        // SAFETY: `clock` is a valid, writable `clockid_t`; on failure it is
        // left as the calling-thread clock, which reads the reader instead.
        unsafe { libc::pthread_getcpuclockid(libc::pthread_self(), &mut clock) };
        ThreadCpu {
            clock,
            start: clock_ns(clock),
        }
    }

    /// [`Self::here`] where no clock names another thread. [`Self::AVAILABLE`]
    /// is false, so nothing should reach this; it keeps the type available to
    /// code that is compiled on every platform and gated at the point of use.
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    pub fn here() -> Self {
        ThreadCpu { clock: 0, start: 0 }
    }

    /// The thread's CPU seconds since [`Self::here`]; read only while the
    /// thread runs, since a clock outlives its thread as a stale id.
    ///
    /// `NaN` where [`Self::AVAILABLE`] is false — a reading that cannot be
    /// taken is not a reading of zero, and a caller that ignored the flag
    /// should see that rather than a plausible small number.
    pub fn seconds(&self) -> f64 {
        if !Self::AVAILABLE {
            return f64::NAN;
        }
        clock_ns(self.clock).saturating_sub(self.start) as f64 * 1e-9
    }
}

fn clock_ns(clock: libc::clockid_t) -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable `timespec` for the call's duration.
    if unsafe { libc::clock_gettime(clock, &mut ts) } != 0 {
        return 0;
    }
    u64::try_from(ts.tv_sec).unwrap_or(0) * 1_000_000_000 + u64::try_from(ts.tv_nsec).unwrap_or(0)
}

/// Charges one task: the calling thread's CPU time from [`TaskClock::start`]
/// to [`TaskClock::finish`], plus that of every thread [`spawn`]ed under it.
pub struct TaskClock {
    charge: Arc<AtomicU64>,
    own_start: u64,
    outer: Option<Arc<AtomicU64>>,
}

impl TaskClock {
    /// Start charging this thread's spawns to a new task.
    pub fn start() -> Self {
        let charge = Arc::new(AtomicU64::new(0));
        let outer = CHARGE.with(|c| c.replace(Some(Arc::clone(&charge))));
        TaskClock {
            charge,
            own_start: thread_cpu_ns(),
            outer,
        }
    }

    /// The task's CPU seconds so far.
    pub fn finish(self) -> f64 {
        self.charged_ns() as f64 * 1e-9
    }

    fn charged_ns(&self) -> u64 {
        let own = thread_cpu_ns().saturating_sub(self.own_start);
        own + self.charge.load(Ordering::Acquire)
    }
}

/// Restores whatever this thread charged before, paying into it the time of
/// the threads spawned under the task: a task that ends early (an error)
/// leaves no charge behind. The calling thread's own time is not paid, since
/// whatever charges this thread counts it already (an enclosing clock reads
/// the thread's CPU time; a spawned thread pays all of its own when it ends).
impl Drop for TaskClock {
    fn drop(&mut self) {
        let spawned = self.charge.load(Ordering::Acquire);
        let outer = self.outer.take();
        if let Some(outer) = &outer {
            outer.fetch_add(spawned, Ordering::AcqRel);
        }
        CHARGE.with(|c| c.replace(outer));
    }
}

/// Wrap `f` so the thread running it inherits this thread's charge and pays
/// its CPU time into it when it returns.
fn charged<T>(f: impl FnOnce() -> T) -> impl FnOnce() -> T {
    let charge = CHARGE.with(|c| c.borrow().clone());
    move || {
        let Some(charge) = charge else {
            return f();
        };
        CHARGE.with(|c| c.replace(Some(Arc::clone(&charge))));
        let out = f();
        charge.fetch_add(thread_cpu_ns(), Ordering::AcqRel);
        out
    }
}

/// [`Scope::spawn`], with the thread's CPU time charged to the spawning
/// thread's task.
pub fn spawn<'scope, 'env, T: Send + 'scope>(
    scope: &'scope Scope<'scope, 'env>,
    f: impl FnOnce() -> T + Send + 'scope,
) -> ScopedJoinHandle<'scope, T> {
    scope.spawn(charged(f))
}

/// [`Builder::spawn_scoped`], with the thread's CPU time charged to the
/// spawning thread's task.
pub fn spawn_built<'scope, 'env, T: Send + 'scope>(
    builder: Builder,
    scope: &'scope Scope<'scope, 'env>,
    f: impl FnOnce() -> T + Send + 'scope,
) -> std::io::Result<ScopedJoinHandle<'scope, T>> {
    builder.spawn_scoped(scope, charged(f))
}

/// Whether the process's resident memory **now** can be read here.
///
/// It comes from `/proc`, which only Linux has — and which a Linux process can
/// still be without, in a container or a sandbox that does not mount it, so
/// this says what the platform offers and the reading itself still answers
/// whether it was taken. A memory budget that cannot read resident memory is a
/// budget that never binds: the reading comes back absent, the comparison is
/// false, and the run continues past the limit it was given. A caller refuses
/// such a limit rather than accepting one that does nothing.
///
/// The **peak** is a different question, and [`peak_rss_kb`] answers it on
/// every Unix.
pub const CURRENT_RSS_READABLE: bool = cfg!(target_os = "linux");

/// A field of `/proc/self/status` in KiB, or `None` where it cannot be read —
/// which is not the same as a process holding nothing.
fn status_kb(field: &str) -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find(|l| l.starts_with(field))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
}

/// The process's resident set size now, in KiB (`VmRSS`), or `None` where it
/// cannot be read.
pub fn rss_kb() -> Option<u64> {
    status_kb("VmRSS:")
}

/// The process's peak resident set size, in KiB, or `None` where it cannot be
/// read.
///
/// `VmHWM` where `/proc` has it. Elsewhere `getrusage`'s `ru_maxrss`, which
/// every Unix reports and which is the same quantity — the high-water mark of
/// resident memory — so a peak is readable on macOS even though the current
/// size is not. The units are not the same: Linux gives kilobytes and Darwin
/// gives bytes, which is the kind of difference that silently reports a run as
/// 1024x smaller than it was.
pub fn peak_rss_kb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        status_kb("VmHWM:")
    }
    #[cfg(not(target_os = "linux"))]
    {
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: `usage` is a valid, writable `rusage` for the call.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
            return None;
        }
        let max = u64::try_from(usage.ru_maxrss).ok()?;
        // Darwin reports bytes, every other Unix kilobytes.
        Some(if cfg!(target_vendor = "apple") {
            max / 1024
        } else {
            max
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn burn(rounds: u64) -> u64 {
        let mut x = 1u64;
        for i in 0..rounds {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(i);
        }
        std::hint::black_box(x)
    }

    #[test]
    fn a_task_is_charged_its_own_and_its_spawned_threads_cpu_time() {
        let clock = TaskClock::start();
        let before = thread_cpu_ns();
        burn(20_000_000);
        let own = thread_cpu_ns() - before;
        let spawned: u64 = std::thread::scope(|s| {
            let handles: Vec<_> = (0..3)
                .map(|_| {
                    spawn(s, || {
                        let start = thread_cpu_ns();
                        burn(20_000_000);
                        thread_cpu_ns() - start
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });
        let charged_s = clock.finish();
        let floor = (own + spawned) as f64 * 1e-9;
        assert!(charged_s >= floor, "charged {charged_s} s, below {floor} s");
        assert!(
            charged_s < floor + 0.5,
            "charged {charged_s} s, far above {floor} s"
        );
    }

    #[test]
    fn a_spawn_under_a_spawn_is_charged_to_the_same_task() {
        let clock = TaskClock::start();
        let inner: u64 = std::thread::scope(|s| {
            spawn(s, || {
                std::thread::scope(|s2| {
                    spawn(s2, || {
                        let start = thread_cpu_ns();
                        burn(30_000_000);
                        thread_cpu_ns() - start
                    })
                    .join()
                    .unwrap()
                })
            })
            .join()
            .unwrap()
        });
        assert!(inner > 0);
        assert!(clock.finish() >= inner as f64 * 1e-9);
    }

    #[test]
    fn a_thread_spawned_outside_any_task_is_charged_nowhere() {
        assert!(CHARGE.with(|c| c.borrow().is_none()));
        let value = std::thread::scope(|s| spawn(s, || 7).join().unwrap());
        assert_eq!(value, 7);
        assert!(CHARGE.with(|c| c.borrow().is_none()));
    }

    #[test]
    fn finishing_restores_the_outer_charge() {
        let outer = TaskClock::start();
        let inner = TaskClock::start();
        burn(5_000_000);
        let inner_s = inner.finish();
        assert!(CHARGE.with(|c| c.borrow().is_some()));
        assert!(outer.finish() >= inner_s);
        assert!(CHARGE.with(|c| c.borrow().is_none()));
    }

    #[test]
    fn a_clock_inside_a_task_charges_the_task_each_cpu_second_once() {
        let outer = TaskClock::start();
        let before = thread_cpu_ns();
        let inner = TaskClock::start();
        burn(20_000_000);
        let spawned: u64 = std::thread::scope(|s| {
            spawn(s, || {
                let start = thread_cpu_ns();
                burn(20_000_000);
                thread_cpu_ns() - start
            })
            .join()
            .unwrap()
        });
        let inner_s = inner.finish();
        let own = thread_cpu_ns() - before;
        let outer_s = outer.finish();
        let once = (own + spawned) as f64 * 1e-9;
        assert!(inner_s >= spawned as f64 * 1e-9);
        assert!(outer_s >= once, "charged {outer_s} s, below {once} s");
        // Counting the inner clock's own-thread time twice would add `inner_s`
        // minus the spawned time again, at least the first burn.
        assert!(
            outer_s < once + 0.5 * (inner_s - spawned as f64 * 1e-9),
            "charged {outer_s} s against {once} s: the inner clock was counted twice"
        );
    }

    #[test]
    fn a_thread_cpu_clock_reads_its_thread_from_another() {
        std::thread::scope(|s| {
            let (tx, rx) = std::sync::mpsc::channel();
            let (burned_tx, burned_rx) = std::sync::mpsc::channel::<u64>();
            // Every channel is created inside the scope so that an assertion
            // below drops its sender. The thread outlives the read by waiting
            // on `done`, and a sender held in the enclosing frame could not
            // be dropped until the scope returned — which is the scope
            // waiting for a thread waiting for the scope. A test that hangs
            // on failure reports a timeout instead of a reason.
            let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
            s.spawn(move || {
                let cpu = ThreadCpu::here();
                tx.send(cpu).unwrap();
                let start = thread_cpu_ns();
                burn(40_000_000);
                burned_tx.send(thread_cpu_ns() - start).unwrap();
                // A disconnect says the reader is gone, which is this
                // thread's cue to end as much as a message is.
                let _ = done_rx.recv();
            });
            let mine = thread_cpu_ns();
            let cpu = rx.recv().unwrap();
            let burned = burned_rx.recv().unwrap() as f64 * 1e-9;
            let seen = cpu.seconds();
            let own = (thread_cpu_ns() - mine) as f64 * 1e-9;
            if ThreadCpu::AVAILABLE {
                // The reader blocked while the other thread burned: the clock
                // it read holds the burn, far more than the reader's own time.
                assert!(
                    seen >= burned,
                    "read {seen} s, the thread burned {burned} s"
                );
                assert!(own < 0.5 * burned, "the reader spent {own} s of {burned} s");
            } else {
                // Where no clock names another thread, the reading is not a
                // number. That is the contract callers gate on, and it is
                // what this test has to assert instead: `NaN >= burned` is
                // false, so the reading-is-the-burn assertion would fail here
                // and say the clock was wrong rather than absent.
                assert!(
                    seen.is_nan(),
                    "no clock names another thread here, so there is no reading: {seen}"
                );
            }
            // The thread is still waiting; a disconnect would end it too, so
            // nothing is asserted on the send.
            let _ = done_tx.send(());
        });
    }

    #[test]
    fn memory_marks_read_the_process() {
        // The current size is read first and the peak second, because the
        // order is what makes the comparison sound: a peak read at the later
        // moment is the maximum over everything up to it, so it bounds a
        // current size read before it whatever else allocated in between.
        // Read the other way round, a test allocating in parallel between the
        // two reads leaves a current size above the earlier peak, and the
        // assertion fails on a process that never did anything wrong.
        // The peak is readable on every Unix; the current size needs `/proc`.
        let current = rss_kb();
        let peak = peak_rss_kb().expect("a peak reading on any Unix");
        assert!(peak > 0, "a running process has held some memory");
        match current {
            Some(rss) => {
                assert!(rss > 0, "a reading was taken and came back as nothing held");
                assert!(peak >= rss, "the peak is below the current size");
            }
            // Absent is the answer where no `/proc` holds it, and a defect
            // where one does. Written as a guard rather than as an assertion
            // over the constant, which is constant on any one platform.
            None if CURRENT_RSS_READABLE => {
                panic!("this platform reports resident memory and the reading was still absent")
            }
            None => {}
        }
    }
}
