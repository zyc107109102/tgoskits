//! Minimal Linux `sync_file` fd object (UAPI alignment of
//! `drivers/dma-buf/sync_file.c`).
//!
//! The only consumer is card0's EXECBUFFER fence path. Guest submits are
//! fire-and-forget (the fork's `submit_3d` enqueues and returns; the fence
//! flag rides on the command), so an out-fence must be a *real* fence: it
//! starts unsignaled and flips once the host completed the submit (the
//! used-ring pop advances `completed_fence_id`). This matches Linux
//! `VIRTGPU_EXECBUF_FENCE_FD_OUT` (`virtgpu_ioctl.c`): the kernel wraps the
//! dma-fence in a sync_file and `sync_file_poll` reports POLLIN when the
//! fence fires.
//!
//! UAPI reference (`include/uapi/linux/sync_file.h`, Linux master; opcodes
//! 0-2 were burned by the sync-framework v1→v2 revert):
//! - `SYNC_IOC_WAIT` = `_IOW('>', 0, struct sync_wait_data)` in the v2 ABI;
//!   v1 used `_IOW('>', 0, __s32)`. Both place the millisecond timeout in
//!   the first four bytes, so both are matched by (type `0x3e`, nr `0`).
//!   Negative waits forever, zero only tests, positive bounds the wait;
//!   expiry reports `ETIMEDOUT` (sync_file_ioctl_wait → -ETIME).
//! - `SYNC_IOC_FILE_INFO` = `_IOWR('>', 4, struct sync_file_info)`;
//!   `status` is 1 signaled / 0 active, `num_fences == 0` publishes the
//!   fence count, a non-null `sync_fence_info` buffer receives one entry.
//! - `SYNC_IOC_MERGE` / `SYNC_IOC_SET_DEADLINE` have no consumer here and
//!   keep the generic `ENOTTY`.
//! - `poll`/`epoll` report `POLLIN` once signaled.
//!
//! Wakeups: a [`PollSet`] drives poll/epoll sleepers. Completion is observed
//! three ways: waiter-driven refresh (the WAIT ioctl loop, poll levels), a
//! background refresher task (the device's completion IRQ is not delivered in
//! this environment, so a guest blocked in `poll()` needs the refresher to
//! pump the used ring and wake it), and the display-completion IRQ handler as
//! a fast path when the IRQ does fire. The guest's libsync fence wait is
//! `poll(fd, POLLIN, timeout)`, not the SYNC_IOC_WAIT ioctl.

use alloc::{
    borrow::Cow,
    string::String,
    sync::{Arc, Weak},
    vec::Vec,
};
use core::{
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::Duration,
};

use ax_display::gpu3d_fence_completed;
use ax_runtime::{hal::time::monotonic_time, task::sync::WaitQueue};
use axpoll::{IoEvents, Pollable};
use axpoll_set::PollSet;

use crate::{
    StarryError, StarryResult,
    file::FileLike,
    mm::{VmMutPtr, VmPtr},
    sync::IrqMutex,
    task::yield_now,
};

/// ioctl type byte for every sync_file command (`#define SYNC_IOC_MAGIC '>'`).
const SYNC_IOC_MAGIC: u32 = b'>' as u32;

/// `SYNC_IOC_WAIT`: wait for the fence, timeout in the first `__s32`.
const SYNC_IOC_WAIT_NR: u32 = 0;
/// `SYNC_IOC_FILE_INFO`: describe fence status/count (opcode 4 in the v2 ABI).
const SYNC_IOC_FILE_INFO_NR: u32 = 4;

const FENCE_NAME: &str = "starry-fence";
const FENCE_DRIVER_NAME: &str = "starry-card0";

/// A sync_file backed by one GPU submit fence.
///
/// Signaled state is published before pollers are woken (`Release` on the
/// swap, `Acquire` on the loads), so a woken waiter always observes the
/// completion it was woken for.
pub struct SyncFile {
    /// The `submit_3d` fence id whose host completion signals this file.
    fence_id: u64,
    signaled: AtomicBool,
    poll_set: PollSet,
    /// Whether a poll/epoll waiter has installed a waker in `poll_set`.
    ///
    /// The new [`Pollable`] contract has no deregistration hook (the
    /// registrar owns the lease), so this is conservative: it clears only on
    /// [`Drop`]. A polled fd keeps the refresher at the active cadence until
    /// its fd closes — for per-submit fence fds that is one frame.
    has_poller: AtomicBool,
    /// [wake-hop run6] monotonic ns when `register()` installed this fence
    /// (creation of the guest-visible fence latency interval).
    registered_ns: core::sync::atomic::AtomicU64,
}

/// Live out-fence registry. The host completion of a fence is only observable
/// as a level (`completed_fence_id`); a guest *blocked in poll()* cannot
/// re-check that level by itself, so a background refresher task pumps the
/// used ring and wakes matching pollers ([`refresher_loop`]). The
/// display-completion IRQ handler (card0) also refreshes it as a fast path
/// when the IRQ does fire. Entries are `Weak`; dead ones are pruned by the
/// same scan. The lock is an IRQ-save mutex so the IRQ path can never spin
/// on it (all holders hold it with local IRQs disabled).
static FENCE_WAITERS: IrqMutex<Vec<(u64, Weak<SyncFile>)>> = IrqMutex::new(Vec::new());

/// One-shot guard so the refresher task is spawned exactly once (on the first
/// registered out-fence).
static REFRESHER_SPAWNED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Number of live out-fences that currently have a registered poll/epoll
/// waker (`has_poller`). The refresher's 1 ms service cadence is only needed
/// while this is non-zero: `signal` exists to wake pollers, and every other
/// wait path re-checks the fence level itself (WAIT ioctl and in-fence waits
/// refresh in their yield loop; epoll re-polls the file on every wait).
static FENCE_POLLERS: AtomicUsize = AtomicUsize::new(0);

/// Parks the refresher while no pollers exist. A 0→1 poller transition in
/// `Pollable::register_shared` notifies it so the first waiter is served
/// immediately instead of after one idle backstop tick.
static REFRESHER_WAKE: WaitQueue = WaitQueue::new();

/// Idle backstop tick while no pollers exist: prunes dead registry entries
/// and covers a missed 0→1 notification. 50 ms is far below any fence-wait
/// timeout a guest tolerates, and 20 wakes/s do not contend for the scheduler.
const REFRESHER_IDLE_TICK: Duration = Duration::from_millis(50);

/// [wake-hop run6] out-fence latency forensics: register() → signal() in ns.
/// Bucket edges (µs): 10/25/50/100/250/500/1m/2m/4m/8m + overflow.
pub const FENCE_LAT_EDGES_US: [u64; 10] = [10, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000];
pub static FENCE_LAT_N: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
pub static FENCE_LAT_SUM_NS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
pub static FENCE_LAT_MAX_NS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
pub static FENCE_LAT_BUCKETS: [core::sync::atomic::AtomicU64; 11] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const ZERO: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
    [ZERO; 11]
};

impl SyncFile {
    /// Creates a sync_file for `fence_id`, initially unsignaled.
    pub fn new(fence_id: u64) -> Self {
        Self {
            fence_id,
            signaled: AtomicBool::new(false),
            poll_set: PollSet::new(),
            has_poller: AtomicBool::new(false),
            registered_ns: core::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Registers this out-fence in the completion registry. Call right after
    /// the `Arc` is created (card0's EXECBUFFER FENCE_FD_OUT path).
    ///
    /// `lock_irqsave` on both sides (here and the scans) guarantees the
    /// registry is never held with IRQs enabled, so the completion IRQ can
    /// never spin on it.
    pub fn register(self: &Arc<Self>) {
        self.registered_ns.store(
            monotonic_time().as_nanos() as u64,
            core::sync::atomic::Ordering::Relaxed,
        );
        FENCE_WAITERS
            .lock()
            .push((self.fence_id, Arc::downgrade(self)));
        ensure_refresher();
    }

    /// Publishes the signaled state and wakes poll/epoll sleepers.
    ///
    /// Task context only: [`PollSet::wake`] must not run in hard IRQ. All
    /// call sites (WAIT ioctl, poll levels, in-fence waits) are task context.
    fn signal(&self) {
        if !self.signaled.swap(true, Ordering::Release) {
            // [wake-hop run6] fence latency: register() → signal() covers
            // host processing + the refresher cadence that actually notices
            // the completion. This is the guest-visible out-fence latency
            // Mesa's sync_file poll experiences.
            let registered = self.registered_ns.load(Ordering::Relaxed);
            if registered != 0 {
                let now = monotonic_time().as_nanos() as u64;
                if now >= registered {
                    let dt = now - registered;
                    FENCE_LAT_SUM_NS.fetch_add(dt, Ordering::Relaxed);
                    FENCE_LAT_MAX_NS.fetch_max(dt, Ordering::Relaxed);
                    FENCE_LAT_N.fetch_add(1, Ordering::Relaxed);
                    let mut idx = 0usize;
                    let us = dt / 1000;
                    while idx < FENCE_LAT_EDGES_US.len() && us >= FENCE_LAT_EDGES_US[idx] {
                        idx += 1;
                    }
                    FENCE_LAT_BUCKETS[idx].fetch_add(1, Ordering::Relaxed);
                }
            }
            // SAFETY: task context; readiness was published by the `swap`
            // above before any woken thread reloads it.
            unsafe { self.poll_set.wake(IoEvents::IN) };
        }
    }

    /// Polls the underlying GPU fence and returns the current signaled state.
    pub fn refresh(&self) -> bool {
        if !self.signaled.load(Ordering::Acquire)
            && gpu3d_fence_completed(self.fence_id).is_ok_and(|done| done)
        {
            self.signal();
        }
        self.signaled.load(Ordering::Acquire)
    }

    /// Blocks until signaled.
    ///
    /// `timeout == None` waits forever (EXECBUFFER in-fence semantics);
    /// otherwise the wait is bounded. Cooperative: the loop yields between
    /// completion checks, mirroring the driver's `wait_fence` spin.
    pub fn wait_signaled(&self, timeout: Option<Duration>) -> StarryResult<()> {
        let deadline = timeout.map(|t| monotonic_time() + t);
        loop {
            if self.refresh() {
                return Ok(());
            }
            if let Some(deadline) = deadline
                && monotonic_time() >= deadline
            {
                return Err(StarryError::TimedOut);
            }
            yield_now();
        }
    }

}

impl Drop for SyncFile {
    fn drop(&mut self) {
        // A polled fd can be dropped without a matching unregister return
        // (the new registration lease is owned by the poll/epoll side);
        // drop the poller count it still holds so the refresher's gate
        // cannot leak upward.
        if self.has_poller.swap(false, Ordering::AcqRel) {
            FENCE_POLLERS.fetch_sub(1, Ordering::Release);
        }
        // Fence ids are unique among live out-fences (one SyncFile per
        // submit), so removing every entry with this id is exact.
        // `lock_irqsave`: see `register`.
        FENCE_WAITERS.lock().retain(|(id, _)| *id != self.fence_id);
    }
}

/// Prunes dead out-fence entries. Called from the display-completion IRQ
/// handler (card0), right after the device IRQ was acked and completions
/// pumped.
///
/// The new [`PollSet`] wake contract is task-context-only (there is no
/// `wake_from_irq`), so the IRQ path must neither publish the signaled state
/// nor wake pollers: publication without wake would make a later
/// task-context `signal` a no-op and strand the waiter until its poll
/// timeout. Waking stays with the refresher's task-context service tick.
pub(crate) fn refresh_fence_waiters_from_irq() {
    FENCE_WAITERS.lock().retain(|(_, w)| w.strong_count() > 0);
}

/// [wake-hop run7] Burst window after an execbuffer kick: the refresher
/// re-pumps the used ring at 50µs cadence so a just-submitted fence signals
/// within ~one host round-trip instead of waiting for the next 1ms tick.
static REFRESHER_BURST_UNTIL_NS: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(0);
const REFRESHER_BURST_WINDOW_NS: u64 = 500_000;
const REFRESHER_BURST_TICK: Duration = Duration::from_micros(50);

/// Kicks the refresher into burst mode. Called (task context) right after an
/// execbuffer submit that registered an out-fence; the host completes the
/// fenced command ~tens of µs later, and burst pumping signals the fence as
/// soon as that completion reaches the used ring.
pub(crate) fn kick_refresher() {
    REFRESHER_BURST_UNTIL_NS.store(
        monotonic_time().as_nanos() as u64 + REFRESHER_BURST_WINDOW_NS,
        Ordering::Relaxed,
    );
    REFRESHER_WAKE.notify_one();
}

/// Returns the number of fences that transitioned to signaled.
fn refresh_all_fences() -> usize {
    // Snapshot under the registry lock, then refresh *outside* it: holding
    // the IRQ-save registry lock across `lock_display()` would deadlock on
    // smp=1 if the display lock were held by a preempted task (local IRQs
    // disabled, so the holder can never be rescheduled).
    let snapshot: Vec<(u64, Weak<SyncFile>)> = {
        let waiters = FENCE_WAITERS.lock();
        if waiters.is_empty() {
            return 0;
        }
        waiters.clone()
    };
    let mut signaled = 0;
    for (_, w) in &snapshot {
        if let Some(sf) = w.upgrade() {
            let before = sf.signaled.load(Ordering::Acquire);
            sf.refresh();
            if !before && sf.signaled.load(Ordering::Acquire) {
                signaled += 1;
            }
        }
    }
    signaled
}

/// Removes dead entries (dropped `SyncFile`s) from the registry. This keeps
/// the `Vec` bounded in environments where the completion IRQ never fires
/// and [`refresh_all_fences`] would otherwise rescan the accumulated dead
/// `Weak`s on every service tick.
fn prune_dead_waiters() {
    FENCE_WAITERS.lock().retain(|(_, w)| w.strong_count() > 0);
}

/// Background fence waiter: while any out-fence is registered, periodically
/// pump + refresh so poll-blocked guests observe host completions (the
/// device's completion IRQ is not delivered in this environment). Sleeps long
/// when the registry is empty. Runs forever; spawned once by
/// [`ensure_refresher`].
fn refresher_loop() -> ! {
    loop {
        if FENCE_POLLERS.load(Ordering::Acquire) > 0 {
            let signaled = refresh_all_fences();
            let now = monotonic_time().as_nanos() as u64;
            let in_burst = now < REFRESHER_BURST_UNTIL_NS.load(Ordering::Relaxed);
            if in_burst && signaled == 0 {
                // A kicked submit's completion is imminent; re-check quickly
                // so the fence signals within one host round-trip.
                crate::task::sleep(REFRESHER_BURST_TICK);
            } else {
                if signaled > 0 {
                    REFRESHER_BURST_UNTIL_NS.store(0, Ordering::Relaxed);
                }
                REFRESHER_WAKE.wait_timeout_until(Duration::from_millis(1), || {
                    FENCE_POLLERS.load(Ordering::Acquire) == 0
                });
            }
        } else {
            prune_dead_waiters();
            REFRESHER_WAKE.wait_timeout_until(REFRESHER_IDLE_TICK, || {
                FENCE_POLLERS.load(Ordering::Acquire) > 0
            });
        }
    }
}

/// Spawns the refresher task once. Called from [`SyncFile::register`].
fn ensure_refresher() {
    if REFRESHER_SPAWNED.swap(true, Ordering::Relaxed) {
        return;
    }
    crate::task::kernel_thread_builder(String::from("fence-wait-refresher"))
        .spawn(|| refresher_loop())
        .expect("failed to spawn fence-wait-refresher");
}

impl FileLike for SyncFile {
    fn path(&self) -> Cow<'_, str> {
        "anon_inode:sync_file".into()
    }

    fn validate_write_access(&self) -> StarryResult {
        Err(StarryError::InvalidInput)
    }

    fn ioctl(
        &self,
        current: &crate::task::UserTaskRef,
        cmd: u32,
        arg: usize,
    ) -> StarryResult<usize> {
        let (ty, nr) = (cmd >> 8 & 0xff, cmd & 0xff);
        if ty != SYNC_IOC_MAGIC {
            return Err(StarryError::NotATty);
        }
        match nr {
            SYNC_IOC_WAIT_NR => {
                // Both v1 (`__s32 timeout`) and v2 (`struct sync_wait_data`)
                // read the timeout from the first four bytes.
                let timeout_ms: i32 = (arg as *const i32)
                    .vm_read(current)
                    .map_err(|_| StarryError::BadAddress)?;
                match timeout_ms {
                    n if n < 0 => self.wait_signaled(None)?,
                    0 => {
                        if !self.refresh() {
                            return Err(StarryError::TimedOut);
                        }
                    }
                    n => self.wait_signaled(Some(Duration::from_millis(n as u64)))?,
                }
                Ok(0)
            }
            SYNC_IOC_FILE_INFO_NR => {
                let ptr = arg as *mut SyncFileInfo;
                let mut info: SyncFileInfo =
                    ptr.vm_read(current).map_err(|_| StarryError::BadAddress)?;
                if info.num_fences == 0 {
                    info.num_fences = 1;
                } else if info.fence_info_ptr != 0 {
                    // Capacity was promised; fill the single fence entry.
                    let entry = SyncFenceInfo {
                        obj_name: name_bytes(FENCE_NAME),
                        driver_name: name_bytes(FENCE_DRIVER_NAME),
                        status: if self.refresh() { 1 } else { 0 },
                        flags: 0,
                        timestamp_ns: 0,
                    };
                    (info.fence_info_ptr as *mut SyncFenceInfo)
                        .vm_write(current, entry)
                        .map_err(|_| StarryError::BadAddress)?;
                }
                info.name = name_bytes(FENCE_NAME);
                info.status = if self.refresh() { 1 } else { 0 };
                info.flags = 0;
                info.pad = 0;
                ptr.vm_write(current, info).map_err(|_| StarryError::BadAddress)?;
                Ok(0)
            }
            _ => Err(StarryError::NotATty),
        }
    }
}

impl Pollable for SyncFile {
    fn poll(&self) -> IoEvents {
        if self.refresh() {
            IoEvents::IN
        } else {
            IoEvents::empty()
        }
    }

    unsafe fn register_shared(
        &self,
        sink: &mut dyn axpoll::SharedRegistrationSink,
        events: IoEvents,
    ) {
        if self.refresh() || (events & IoEvents::IN).is_empty() {
            return;
        }
        // SAFETY: poll registration runs in task context; the lease is owned
        // by the registrar and cancels the exact entry on drop.
        unsafe { sink.register_shared(&self.poll_set, IoEvents::IN) };
        if !self.has_poller.swap(true, Ordering::AcqRel) {
            // 0→1 transition: the refresher may be parked on the idle
            // backstop; wake it so this first waiter is served immediately.
            FENCE_POLLERS.fetch_add(1, Ordering::AcqRel);
            REFRESHER_WAKE.notify_one();
        }
    }

    unsafe fn register_exclusive(
        &self,
        sink: &mut dyn axpoll::ExclusiveRegistrationSink,
        events: IoEvents,
    ) {
        if self.refresh() || (events & IoEvents::IN).is_empty() {
            return;
        }
        // SAFETY: as above.
        unsafe { sink.register_exclusive(&self.poll_set, IoEvents::IN) };
    }
}

/// `struct sync_file_info` (UAPI layout, 56 bytes on 64-bit).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SyncFileInfo {
    name: [u8; 32],
    status: i32,
    flags: u32,
    num_fences: u32,
    pad: u32,
    fence_info_ptr: u64,
}

/// `struct sync_fence_info` (UAPI layout, 80 bytes on 64-bit).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SyncFenceInfo {
    obj_name: [u8; 32],
    driver_name: [u8; 32],
    status: i32,
    flags: u32,
    timestamp_ns: u64,
}

/// NUL-padded 32-byte name field as the UAPI defines it.
fn name_bytes(name: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    let bytes = name.as_bytes();
    let len = bytes.len().min(31);
    out[..len].copy_from_slice(&bytes[..len]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_info_layout_matches_uapi() {
        assert_eq!(size_of::<SyncFileInfo>(), 56);
        assert_eq!(size_of::<SyncFenceInfo>(), 80);
    }

    #[test]
    fn name_is_nul_terminated_and_truncated() {
        let long = name_bytes(&"x".repeat(64));
        assert_eq!(long[31], 0);
        let short = name_bytes("ab");
        assert_eq!(&short[..3], b"ab\0");
    }

    #[test]
    fn wait_ioctl_matches_both_abi_variants() {
        // v2 `_IOW('>', 0, struct sync_wait_data)` and v1 `_IOW('>', 0, s32)`
        // must both resolve to (type 0x3e, nr 0).
        assert_eq!(SYNC_IOC_MAGIC, 0x3e);
        assert_eq!(SYNC_IOC_WAIT_NR, 0);
        assert_eq!(SYNC_IOC_FILE_INFO_NR, 4);
    }
}
