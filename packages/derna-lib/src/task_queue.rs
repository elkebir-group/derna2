//! A priority work queue whose tasks share a fixed number of workers.
//!
//! A task is handed a worker count when it starts and may queue more tasks
//! while it runs. Tasks run highest priority first. A task takes the free
//! workers shared out over itself and the tasks still waiting, at least one,
//! so a lone task runs on every worker and a full queue runs one task per
//! worker. No more than `threads` workers are ever claimed at once, and the
//! queue ends when it is empty and nothing runs.
//!
//! There are no barriers: a task that finishes frees its workers for the next
//! one at once, whatever the others are doing. That is the point of the
//! queue: work whose cost varies widely — one task can cost a hundred times
//! another — packs onto the workers without waiting for the slowest of a
//! batch.

use std::collections::BinaryHeap;
use std::sync::{Condvar, Mutex};

/// A task with its priority; higher runs first, ties in queueing order.
struct Queued<T> {
    priority: (u8, u64),
    seq: u64,
    task: T,
}

impl<T> PartialEq for Queued<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.seq == other.seq
    }
}

impl<T> Eq for Queued<T> {}

impl<T> PartialOrd for Queued<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Queued<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

struct State<T> {
    queue: BinaryHeap<Queued<T>>,
    free: usize,
    running: usize,
    /// How many tasks may run at once.
    max_running: usize,
    seq: u64,
    failed: bool,
}

/// The queue's handle inside a running task: it queues more tasks.
pub(crate) struct Queue<T> {
    state: Mutex<State<T>>,
    changed: Condvar,
}

impl<T> Queue<T> {
    /// Queue `task` at `priority`: the class first, then a weight within it.
    pub(crate) fn push(&self, priority: (u8, u64), task: T) {
        let mut state = self.state.lock().expect("the queue's lock");
        let seq = state.seq;
        state.seq += 1;
        state.queue.push(Queued {
            priority,
            seq,
            task,
        });
        drop(state);
        self.changed.notify_all();
    }
}

impl<T> Queue<T> {
    /// Whether the running task asking is the only work left: nothing waits
    /// to start and no other task runs.
    pub(crate) fn alone(&self) -> bool {
        let state = self.state.lock().expect("the queue's lock");
        state.queue.is_empty() && state.running <= 1
    }
}

/// Workers a running task borrowed from the queue's idle ones, handed back
/// when the loan is dropped.
pub(crate) struct Loan<'a, T> {
    queue: &'a Queue<T>,
    count: usize,
}

impl<T> Loan<'_, T> {
    /// How many workers were lent.
    pub(crate) fn count(&self) -> usize {
        self.count
    }
}

impl<T> Drop for Loan<'_, T> {
    fn drop(&mut self) {
        if self.count == 0 {
            return;
        }
        self.queue.state.lock().expect("the queue's lock").free += self.count;
        self.queue.changed.notify_all();
    }
}

impl<T> Queue<T> {
    /// Lend a running task every free worker while no task waits to start,
    /// and none while one does: a task in the queue's tail can widen onto
    /// workers that would otherwise idle. A task queued while the loan is
    /// out waits for a worker to come free, the loan's at the latest, so a
    /// loan should be held for one bounded step of the task's work.
    pub(crate) fn lend_idle(&self) -> Loan<'_, T> {
        let mut state = self.state.lock().expect("the queue's lock");
        let count = if state.queue.is_empty() {
            state.free
        } else {
            0
        };
        state.free -= count;
        Loan { queue: self, count }
    }
}

/// Run `initial` and every task they queue on `threads` workers in all,
/// highest priority first, at most `max_running` tasks at once; `run` gets
/// each task, its worker count and the queue. A task takes the free workers
/// shared out over the running slots left and the tasks waiting, whichever
/// are fewer, so `threads` workers over `max_running` slots give each task
/// `threads / max_running` of them at least. The first error stops the queue:
/// tasks not yet started are dropped, running ones finish, and that error is
/// returned.
#[cfg(test)]
pub(crate) fn run<T: Send, E: Send>(
    threads: usize,
    max_running: usize,
    initial: Vec<((u8, u64), T)>,
    run: impl Fn(T, usize, &Queue<T>) -> Result<(), E> + Sync,
) -> Result<(), E> {
    run_with(threads, max_running, Shares::Spread, initial, || false, run)
}

/// How a task's workers are counted when it starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shares {
    /// The free workers shared out over the running slots left and the tasks
    /// waiting, as [`run`] describes.
    Spread,
    /// One worker per task, whatever is free: a task that spends threads of
    /// its own beside the queue's.
    One,
}

/// [`run`] with each task's workers counted by `shares`, and `idle` offered
/// the time of a worker that finds nothing waiting while other tasks run.
/// The worker sleeps only once `idle` returns `false`; while it runs `idle`
/// it holds no worker of the queue's.
pub(crate) fn run_with<T: Send, E: Send>(
    threads: usize,
    max_running: usize,
    shares: Shares,
    initial: Vec<((u8, u64), T)>,
    idle: impl Fn() -> bool + Sync,
    run: impl Fn(T, usize, &Queue<T>) -> Result<(), E> + Sync,
) -> Result<(), E> {
    let threads = threads.max(1);
    let queue = Queue {
        state: Mutex::new(State {
            queue: BinaryHeap::new(),
            free: threads,
            running: 0,
            max_running: max_running.clamp(1, threads),
            seq: 0,
            failed: false,
        }),
        changed: Condvar::new(),
    };
    for (priority, task) in initial {
        queue.push(priority, task);
    }
    let error: Mutex<Option<E>> = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                while let Some((task, workers)) = take(&queue, shares, &idle) {
                    let outcome = run(task, workers, &queue);
                    let mut state = queue.state.lock().expect("the queue's lock");
                    state.free += workers;
                    state.running -= 1;
                    if let Err(e) = outcome {
                        state.failed = true;
                        state.queue.clear();
                        error.lock().expect("the error's lock").get_or_insert(e);
                    }
                    drop(state);
                    queue.changed.notify_all();
                }
            });
        }
    });
    match error.into_inner().expect("the error's lock") {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// The next task and its worker count, waiting while every worker is
/// claimed or the queue is empty with tasks still running; `None` once the
/// queue is done.
fn take<T>(queue: &Queue<T>, shares: Shares, idle: &impl Fn() -> bool) -> Option<(T, usize)> {
    let mut state = queue.state.lock().expect("the queue's lock");
    loop {
        if state.failed || (state.queue.is_empty() && state.running == 0) {
            return None;
        }
        if state.free > 0 && state.running < state.max_running {
            if let Some(next) = state.queue.pop() {
                let workers = match shares {
                    Shares::Spread => {
                        let slots = state.max_running - state.running;
                        let share = state.free / (state.queue.len() + 1).min(slots);
                        share.clamp(1, state.free)
                    }
                    Shares::One => 1,
                };
                state.free -= workers;
                state.running += 1;
                return Some((next.task, workers));
            }
        }
        if state.queue.is_empty() {
            drop(state);
            if idle() {
                state = queue.state.lock().expect("the queue's lock");
                continue;
            }
            state = queue.state.lock().expect("the queue's lock");
            if !(state.queue.is_empty() && state.running > 0 && !state.failed) {
                continue;
            }
        }
        state = queue.changed.wait(state).expect("the queue's lock");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Every queued task runs once, spawned ones included, and the workers
    /// claimed at once never exceed the count: a lone task gets all of them.
    #[test]
    fn every_task_runs_once_within_the_worker_count() {
        let (claimed, peak, ran) = (
            AtomicUsize::new(0),
            AtomicUsize::new(0),
            AtomicUsize::new(0),
        );
        let first = Mutex::new(None);
        run::<u32, ()>(4, 4, vec![((1, 0), 0)], |depth, workers, queue| {
            let now = claimed.fetch_add(workers, Ordering::SeqCst) + workers;
            peak.fetch_max(now, Ordering::SeqCst);
            first.lock().unwrap().get_or_insert(workers);
            if depth < 5 {
                queue.push((1, 0), depth + 1);
                queue.push((0, u64::from(depth)), depth + 1);
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
            ran.fetch_add(1, Ordering::SeqCst);
            claimed.fetch_sub(workers, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
        assert_eq!(ran.load(Ordering::SeqCst), (1 << 6) - 1);
        assert!(peak.load(Ordering::SeqCst) <= 4);
        assert_eq!(*first.lock().unwrap(), Some(4));
    }

    /// Capped, no more than the cap run at once, and the first tasks share
    /// the workers over the cap's slots rather than one per task.
    #[test]
    fn a_cap_bounds_the_tasks_running_at_once_and_shares_the_workers() {
        let (running, peak) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let first = Mutex::new(Vec::new());
        let tasks = (0..12).map(|i| ((0, 0), i)).collect();
        run::<u32, ()>(4, 2, tasks, |i, workers, _| {
            let now = running.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            if i < 2 {
                first.lock().unwrap().push(workers);
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
            running.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
        assert!(peak.load(Ordering::SeqCst) <= 2);
        assert_eq!(*first.lock().unwrap(), vec![2, 2]);
    }

    /// Two tasks start on half the workers each; once one ends, the other
    /// borrows the two it freed, borrows none while a task waits, and the
    /// waiting task starts once the loans end, on the returned workers.
    #[test]
    fn idle_workers_are_lent_only_while_nothing_waits() {
        let seen = Mutex::new(Vec::new());
        let tasks = vec![((0, 0), 0), ((0, 0), 1)];
        run::<u32, ()>(4, 4, tasks, |i, workers, queue| {
            match i {
                0 => {
                    let first = loop {
                        let loan = queue.lend_idle();
                        if loan.count() > 0 {
                            break loan;
                        }
                        drop(loan);
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    };
                    queue.push((0, 0), 2);
                    let second = queue.lend_idle();
                    seen.lock()
                        .unwrap()
                        .push(("borrowed", workers, first.count(), second.count()));
                }
                1 => {}
                _ => seen.lock().unwrap().push(("waited", workers, 0, 0)),
            }
            Ok(())
        })
        .unwrap();
        // The waiting task starts on the loan's two workers at least, or on
        // all four when the lender has also ended by the time it is woken.
        let seen = seen.into_inner().unwrap();
        assert_eq!(seen[0], ("borrowed", 2, 2, 0));
        assert_eq!(seen.len(), 2);
        assert!(seen[1].0 == "waited" && seen[1].1 >= 2, "{seen:?}");
    }

    /// The first error stops the queue and is returned.
    #[test]
    fn an_error_stops_the_queue() {
        let ran = AtomicUsize::new(0);
        let tasks = (0..100).map(|i| ((0, 0), i)).collect();
        let got = run(2, 2, tasks, |i: u32, _, _| {
            ran.fetch_add(1, Ordering::SeqCst);
            if i == 3 {
                Err(i)
            } else {
                Ok(())
            }
        });
        assert_eq!(got, Err(3));
        assert!(ran.load(Ordering::SeqCst) < 100);
    }
}
