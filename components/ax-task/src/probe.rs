//! [wake-hop] Temporary wakeup-delivery forensics for the port-base layered
//! measurement. Stages record last-event timestamps plus histograms; the
//! StarryOS card0 fx report reads and resets everything. This module is
//! worktree-only instrumentation and must never be committed.
//!
//! Stage chain for one unix-stream peer wake: `record_wake_enq` fires when a
//! waker enqueues the coroutine (CAS winner only; coalesced duplicates do not
//! re-record), `record_thread_wake` when a parked owner got a scheduler-level
//! wake, `record_thread_resume` when that owner leaves the OS park inside the
//! executor run loop, `record_pick` when the coroutine is dequeued from the
//! ready inbox, and `record_poll_duration` after each coroutine poll (long
//! samples are non-yielding compute segments).

use core::sync::atomic::{AtomicU64, Ordering};

use crate::runtime::task_runtime;

/// Bucket edges (µs) for small transfer-path costs (publish, poll duration).
pub const SMALL_EDGES: [u64; 10] = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000];
/// Bucket edges (µs) for delivery delays (OS pickup, enqueue-to-pick).
pub const DELAY_EDGES: [u64; 10] = [5, 10, 25, 50, 100, 250, 500, 1000, 2000, 4000];

const fn zeroed_buckets() -> [AtomicU64; 11] {
    [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ]
}

/// Index of `latency_ns` in a histogram whose `edges` are ascending µs bounds.
pub fn bucket_index_us(latency_ns: u64, edges: &[u64]) -> usize {
    let us = latency_ns / 1000;
    let mut index = 0;
    while index < edges.len() && us >= edges[index] {
        index += 1;
    }
    index
}

fn monotonic_nanos() -> u64 {
    task_runtime::monotonic_now().as_nanos()
}

/// Instrumentation-side timestamp source (monotonic ktime domain).
pub fn now_ns() -> u64 {
    monotonic_nanos()
}

// -- stage 1: coroutine wake enqueue ------------------------------------- //
pub static WAKE_ENQ_N: AtomicU64 = AtomicU64::new(0);
/// Per-victim last wake-enqueue state (victim id + monotonic ns), avoiding
/// the global last-event survivor bias when many wakeups interleave.
pub const VICTIM_SLOTS: usize = 64;
pub static WAKE_SLOT_VICTIM: [AtomicU64; VICTIM_SLOTS] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const ZERO: AtomicU64 = AtomicU64::new(0);
    [ZERO; VICTIM_SLOTS]
};
pub static WAKE_SLOT_ENQ_NS: [AtomicU64; VICTIM_SLOTS] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const ZERO: AtomicU64 = AtomicU64::new(0);
    [ZERO; VICTIM_SLOTS]
};

// -- stage 2: scheduler-level thread wake (owner was PARKED) -------------- //
pub static THREAD_WAKE_N: AtomicU64 = AtomicU64::new(0);
pub static TW_SLOT_VICTIM: [AtomicU64; VICTIM_SLOTS] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const ZERO: AtomicU64 = AtomicU64::new(0);
    [ZERO; VICTIM_SLOTS]
};
pub static TW_SLOT_WAKE_NS: [AtomicU64; VICTIM_SLOTS] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const ZERO: AtomicU64 = AtomicU64::new(0);
    [ZERO; VICTIM_SLOTS]
};

// -- stage 3: owner thread resumed from the OS park ----------------------- //
pub static THREAD_RESUME_N: AtomicU64 = AtomicU64::new(0);
pub static RESUME_DELAY_N: AtomicU64 = AtomicU64::new(0);
pub static RESUME_DELAY_SUM_NS: AtomicU64 = AtomicU64::new(0);
pub static RESUME_DELAY_MAX_NS: AtomicU64 = AtomicU64::new(0);
pub static RESUME_DELAY_BUCKETS: [AtomicU64; 11] = zeroed_buckets();

// -- stage 4: coroutine picked from the ready inbox ----------------------- //
pub static PICK_N: AtomicU64 = AtomicU64::new(0);
pub static LAST_PICK_NS: AtomicU64 = AtomicU64::new(0);
pub static LAST_PICK_VICTIM: AtomicU64 = AtomicU64::new(0);
/// wake→pick samples attributed by victim-thread match.
pub static ENQ2PICK_N: AtomicU64 = AtomicU64::new(0);
pub static ENQ2PICK_SUM_NS: AtomicU64 = AtomicU64::new(0);
pub static ENQ2PICK_MAX_NS: AtomicU64 = AtomicU64::new(0);
pub static ENQ2PICK_BUCKETS: [AtomicU64; 11] = zeroed_buckets();

// -- stage 5: single coroutine poll duration ------------------------------ //
pub static POLL_DUR_N: AtomicU64 = AtomicU64::new(0);
pub static POLL_DUR_SUM_NS: AtomicU64 = AtomicU64::new(0);
pub static POLL_DUR_MAX_NS: AtomicU64 = AtomicU64::new(0);
pub static POLL_DUR_BUCKETS: [AtomicU64; 11] = zeroed_buckets();

/// Records one actual coroutine enqueue (RUN_QUEUED CAS winner).
///
/// Like every waker callback this may in principle run in hard IRQ; the
/// StarryOS workloads under measurement never wake coroutines from IRQ
/// (IRQ producers only notify a fixed service thread), so the monotonic
/// provider sample here is task-context in practice.
pub fn record_wake_enq(victim_thread: u64) {
    let slot = (victim_thread as usize) % VICTIM_SLOTS;
    WAKE_SLOT_VICTIM[slot].store(victim_thread, Ordering::Relaxed);
    WAKE_SLOT_ENQ_NS[slot].store(monotonic_nanos(), Ordering::Relaxed);
    WAKE_ENQ_N.fetch_add(1, Ordering::Relaxed);
}

/// Records that a parked owner thread received a scheduler-level wake.
pub fn record_thread_wake(victim_thread: u64) {
    let slot = (victim_thread as usize) % VICTIM_SLOTS;
    TW_SLOT_VICTIM[slot].store(victim_thread, Ordering::Relaxed);
    TW_SLOT_WAKE_NS[slot].store(monotonic_nanos(), Ordering::Relaxed);
    THREAD_WAKE_N.fetch_add(1, Ordering::Relaxed);
}

/// Records the owner thread returning from the OS park, attributed when this
/// thread owns the last recorded thread wake in its slot.
pub fn record_thread_resume(victim_thread: u64) {
    let now = monotonic_nanos();
    let slot = (victim_thread as usize) % VICTIM_SLOTS;
    let last_wake = TW_SLOT_WAKE_NS[slot].load(Ordering::Relaxed);
    if last_wake != 0
        && victim_thread == TW_SLOT_VICTIM[slot].load(Ordering::Relaxed)
        && now >= last_wake
    {
        let delay = now - last_wake;
        RESUME_DELAY_SUM_NS.fetch_add(delay, Ordering::Relaxed);
        RESUME_DELAY_MAX_NS.fetch_max(delay, Ordering::Relaxed);
        RESUME_DELAY_BUCKETS[bucket_index_us(delay, &DELAY_EDGES)].fetch_add(1, Ordering::Relaxed);
        RESUME_DELAY_N.fetch_add(1, Ordering::Relaxed);
    }
    THREAD_RESUME_N.fetch_add(1, Ordering::Relaxed);
}

/// Records one coroutine pick, attributing wake→pick only when the picked
/// coroutine's owner matches the last enqueue's victim thread.
pub fn record_pick(victim_thread: u64) {
    let now = monotonic_nanos();
    LAST_PICK_NS.store(now, Ordering::Relaxed);
    LAST_PICK_VICTIM.store(victim_thread, Ordering::Relaxed);
    PICK_N.fetch_add(1, Ordering::Relaxed);
    let slot = (victim_thread as usize) % VICTIM_SLOTS;
    let enq = WAKE_SLOT_ENQ_NS[slot].load(Ordering::Relaxed);
    if enq != 0 && victim_thread == WAKE_SLOT_VICTIM[slot].load(Ordering::Relaxed) && now >= enq {
        let delay = now - enq;
        ENQ2PICK_SUM_NS.fetch_add(delay, Ordering::Relaxed);
        ENQ2PICK_MAX_NS.fetch_max(delay, Ordering::Relaxed);
        ENQ2PICK_N.fetch_add(1, Ordering::Relaxed);
        ENQ2PICK_BUCKETS[bucket_index_us(delay, &DELAY_EDGES)].fetch_add(1, Ordering::Relaxed);
    }
}

/// Records one completed coroutine poll (owner-side, exact per sample).
pub fn record_poll_duration(duration_ns: u64) {
    POLL_DUR_SUM_NS.fetch_add(duration_ns, Ordering::Relaxed);
    POLL_DUR_MAX_NS.fetch_max(duration_ns, Ordering::Relaxed);
    POLL_DUR_BUCKETS[bucket_index_us(duration_ns, &SMALL_EDGES)].fetch_add(1, Ordering::Relaxed);
    POLL_DUR_N.fetch_add(1, Ordering::Relaxed);
}
