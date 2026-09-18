use core::{
    marker::PhantomData,
    ptr::NonNull,
    sync::atomic::{AtomicPtr, Ordering},
};

use ax_alloc::{UsageKind, global_allocator};
use ax_memory_addr::PAGE_SIZE_4K;
#[cfg(feature = "virtio-net")]
use virtio_drivers::Error as VirtIoError;
use virtio_drivers::{
    BufferDirection, Hal as VirtIoHal, PhysAddr as VirtIoPhysAddr,
    transport::{DeviceType, Transport, mmio::MmioTransport},
};

#[cfg(feature = "virtio-gpu")]
pub mod display;
#[cfg(feature = "virtio-input")]
pub mod input;
#[cfg(feature = "virtio-net")]
pub mod net;
#[cfg(feature = "virtio-socket")]
pub mod vsock;

pub const MMIO_DEVICE_NAME: &str = "virtio-mmio";

#[cfg(any(
    feature = "virtio-net",
    feature = "virtio-gpu",
    feature = "virtio-input",
    feature = "virtio-socket",
))]
crate::model_register!(
    name: "VirtIO MMIO",
    level: ProbeLevel::PostKernel,
    priority: ProbePriority::DEFAULT,
    probe_kinds: &[ProbeKind::Fdt {
        compatibles: &["virtio,mmio"],
        on_probe: probe_fdt
    }],
);

#[cfg(any(
    feature = "virtio-net",
    feature = "virtio-gpu",
    feature = "virtio-input",
    feature = "virtio-socket",
))]
fn probe_fdt(probe: rdrive::register::ProbeFdt<'_>) -> Result<(), rdrive::probe::OnProbeError> {
    let (info, platform_device) = probe.into_parts();
    let (device_type, transport) = probe_fdt_mmio_device(&info)?;
    #[cfg(feature = "virtio-net")]
    if device_type == DeviceType::Network {
        return net::register_fdt_transport(&info, platform_device, transport);
    }
    #[cfg(feature = "virtio-socket")]
    if device_type == DeviceType::Socket {
        return vsock::register_fdt_transport(&info, platform_device, transport);
    }
    register_static_transport(platform_device, device_type, transport)
}

pub struct VirtIoHalImpl(PhantomData<()>);

pub const fn has_static_mmio_drivers() -> bool {
    cfg!(any(
        feature = "virtio-net",
        feature = "virtio-gpu",
        feature = "virtio-input",
        feature = "virtio-socket",
    ))
}

/// Device-visible staging for virtio buffers that live outside the linear DMA
/// window.
///
/// `Hal::share` must return an address the device can actually read. Kernel
/// virtual addresses are only a linear offset of their physical frames inside
/// the linear map window; task stacks (vmap'd, physically scattered) and other
/// dynamic mappings break that identity, so a formula-based translation would
/// silently hand the device the wrong frames. [`share`](VirtIoHal::share)
/// therefore translates through the kernel page tables and shares zero-copy
/// whenever the buffer is physically contiguous; scattered buffers are copied
/// into staging memory owned by this driver:
///
/// * a pool of fixed slots for buffers up to [`BOUNCE_SLOT_SIZE`], and
/// * dedicated tracked pages ([`BOUNCE_OVERSIZE`]) for larger buffers and as
///   the backstop when the pool has no free slot.
///
/// `unshare` copies device-written responses back into the caller's original
/// buffer and recycles the staging memory. Staging never falls back to the
/// offset formula: `Hal::share` cannot report failure, and handing the device
/// an unverified address would corrupt memory, so a staging allocation
/// failure panics.
///
/// This mirrors Linux's `dma_map_single` + swiotlb bounce design: drivers keep
/// their zero-copy submission model; the mapping layer owns addressability.
/// Two platform assumptions apply, both shared with the pre-existing
/// `dma_alloc` path: the device is coherent (no cache maintenance around the
/// staging copies), and kernel frames are [`PAGE_SIZE_4K`]. A platform whose
/// [`Klib`](axklib::Klib) exposes no page-table query reports `None` for every
/// translation, so every borrowed buffer is staged — one copy is the price of
/// not guessing.
const BOUNCE_SLOT_SIZE: usize = 16 * 1024;
const BOUNCE_POOL_SLOTS: usize = 64;

struct BouncePool {
    /// Linear-window virtual base of the pool.
    vaddr: usize,
    /// Physical base handed to the device.
    paddr: usize,
    /// Free slot ids.
    free: ax_sync::SpinLock<alloc::vec::Vec<u32>>,
}

/// Initialized pool handle, `null` until the first scattered buffer needs
/// staging. `Release`/`Acquire` publish the fully constructed pool; `unshare`
/// loads it read-only so completing direct-mapped descriptors never
/// initializes or locks anything.
static BOUNCE_POOL: AtomicPtr<BouncePool> = AtomicPtr::new(core::ptr::null_mut());
/// Oversize staging: device paddr → (staging vaddr, staged length).
static BOUNCE_OVERSIZE: ax_sync::SpinLock<alloc::collections::BTreeMap<usize, (usize, usize)>> =
    ax_sync::SpinLock::new(alloc::collections::BTreeMap::new());

/// Returns the initialized pool without ever creating one (for `unshare`).
fn bounce_pool_ready() -> Option<&'static BouncePool> {
    let pool = BOUNCE_POOL.load(Ordering::Acquire);
    (!pool.is_null()).then(|| unsafe { &*pool })
}

/// Returns the pool, initializing it on first use. Returns `None` when the
/// pool allocation fails; callers stage per-buffer instead. Unlike a permanent
/// failed state this can be retried — a failed boot-time allocation must not
/// disable staging for the lifetime of the kernel.
fn bounce_pool_init() -> Option<&'static BouncePool> {
    if let Some(pool) = bounce_pool_ready() {
        return Some(pool);
    }
    let total_pages = (BOUNCE_SLOT_SIZE * BOUNCE_POOL_SLOTS).div_ceil(PAGE_SIZE_4K);
    let Ok(vaddr) = global_allocator().alloc_pages(total_pages, PAGE_SIZE_4K, UsageKind::Dma)
    else {
        log::debug!("virtio: bounce pool allocation failed; staging per buffer");
        return None;
    };
    let pool = alloc::boxed::Box::new(BouncePool {
        vaddr,
        paddr: axklib::mem::virt_to_phys(vaddr.into()).as_usize(),
        free: ax_sync::SpinLock::new((0..BOUNCE_POOL_SLOTS as u32).rev().collect()),
    });
    let pool = alloc::boxed::Box::into_raw(pool);
    match BOUNCE_POOL.compare_exchange(
        core::ptr::null_mut(),
        pool,
        Ordering::Release,
        Ordering::Acquire,
    ) {
        Ok(_) => Some(unsafe { &*pool }),
        Err(winner) => {
            // Lost the one-shot init race; the winner's pool serves everyone.
            let losing = unsafe { alloc::boxed::Box::from_raw(pool) };
            global_allocator().dealloc_pages(losing.vaddr, total_pages, UsageKind::Dma);
            drop(losing);
            Some(unsafe { &*winner })
        }
    }
}

mod direct_map;

/// Walks `vaddr..vaddr + len` through the kernel page tables and returns the
/// buffer's physical base when it can be direct-mapped (all pages
/// translatable, physically contiguous); `None` otherwise.
fn contiguous_dma_paddr(vaddr: usize, len: usize) -> Option<usize> {
    direct_map::contiguous_dma_paddr_with(axklib::mem::virt_to_phys_checked, vaddr, len)
}

/// Copies caller data into staging memory. `DriverToDevice`/`Both` copy the
/// bytes in; `DeviceToDriver` zeroes the staging first so an under-writing
/// device cannot leak a previous stager's bytes back through `unshare`.
///
/// # Safety
///
/// `dst` must be valid for `buffer.len()` bytes and must not overlap the
/// caller's buffer.
unsafe fn stage_into(dst: *mut u8, buffer: NonNull<[u8]>, direction: BufferDirection) {
    let len = buffer.len();
    let src = buffer.as_ptr() as *mut u8;
    // SAFETY: the caller's buffer and the staging memory are valid for `len`
    // bytes and do not overlap.
    unsafe {
        match direction {
            BufferDirection::DriverToDevice | BufferDirection::Both => {
                core::ptr::copy_nonoverlapping(src, dst, len);
            }
            BufferDirection::DeviceToDriver => core::ptr::write_bytes(dst, 0, len),
        }
    }
}

/// Stages `buffer` into a free pool slot. Returns the device-visible paddr,
/// or `None` when the pool has no free slot (the caller stages fresh pages).
///
/// # Safety
///
/// See [`stage_into`]; the pool slot satisfies it for any buffer up to
/// [`BOUNCE_SLOT_SIZE`].
unsafe fn bounce_stage(
    pool: &BouncePool,
    buffer: NonNull<[u8]>,
    direction: BufferDirection,
) -> Option<usize> {
    let slot = pool.free.lock().pop()?;
    let off = slot as usize * BOUNCE_SLOT_SIZE;
    // SAFETY: pool-owned linear-window memory, disjoint from the caller's
    // buffer.
    unsafe { stage_into((pool.vaddr + off) as *mut u8, buffer, direction) };
    Some(pool.paddr + off)
}

/// Stages `buffer` into dedicated linear-window pages tracked in
/// [`BOUNCE_OVERSIZE`]. Serves buffers larger than a pool slot and backs up an
/// exhausted pool. Returns the device-visible paddr, or `None` when the
/// allocation fails.
///
/// # Safety
///
/// See [`stage_into`]; the fresh allocation satisfies it.
unsafe fn fresh_stage(buffer: NonNull<[u8]>, direction: BufferDirection) -> Option<usize> {
    let len = buffer.len();
    let pages = len.div_ceil(PAGE_SIZE_4K);
    let Ok(dst) = global_allocator().alloc_pages(pages, PAGE_SIZE_4K, UsageKind::Dma) else {
        return None;
    };
    let paddr = axklib::mem::virt_to_phys(dst.into()).as_usize();
    // SAFETY: freshly allocated linear-window pages, disjoint from the
    // caller's buffer.
    unsafe { stage_into(dst as *mut u8, buffer, direction) };
    BOUNCE_OVERSIZE.lock().insert(paddr, (dst, len));
    Some(paddr)
}

/// Copies device-written data back into the caller's original buffer and
/// recycles the staging memory. Addresses outside the staging range belong to
/// direct-mapped buffers, which were never copied and need no work.
///
/// # Safety
///
/// `paddr` must be the address [`bounce_stage`] or [`fresh_stage`] returned
/// for `buffer`, and the staging memory must not overlap the caller's buffer.
unsafe fn bounce_unstage(
    pool: &BouncePool,
    paddr: usize,
    buffer: NonNull<[u8]>,
    direction: BufferDirection,
) {
    let len = buffer.len();
    let dst = buffer.as_ptr() as *mut u8;

    if paddr >= pool.paddr && paddr < pool.paddr + BOUNCE_SLOT_SIZE * BOUNCE_POOL_SLOTS {
        let off = paddr - pool.paddr;
        debug_assert_eq!(off % BOUNCE_SLOT_SIZE, 0);
        if matches!(
            direction,
            BufferDirection::DeviceToDriver | BufferDirection::Both
        ) {
            // SAFETY: the pool slot and the caller's buffer are valid for
            // `len` bytes and do not overlap.
            unsafe { core::ptr::copy_nonoverlapping((pool.vaddr + off) as *const u8, dst, len) };
        }
        pool.free.lock().push((off / BOUNCE_SLOT_SIZE) as u32);
        return;
    }
    if let Some((vaddr, staged_len)) = BOUNCE_OVERSIZE.lock().remove(&paddr) {
        debug_assert_eq!(staged_len, len);
        // Clamp so a length-contract violation cannot read past the staging
        // allocation in release builds; the deallocation always covers the
        // full staged allocation.
        let copy_len = staged_len.min(len);
        if matches!(
            direction,
            BufferDirection::DeviceToDriver | BufferDirection::Both
        ) {
            // SAFETY: the staging pages and the caller's buffer are valid for
            // `copy_len` bytes and do not overlap.
            unsafe { core::ptr::copy_nonoverlapping(vaddr as *const u8, dst, copy_len) };
        }
        global_allocator().dealloc_pages(vaddr, staged_len.div_ceil(PAGE_SIZE_4K), UsageKind::Dma);
    }
}

unsafe impl VirtIoHal for VirtIoHalImpl {
    fn dma_alloc(
        pages: usize,
        _direction: BufferDirection,
        _access_platform: bool,
    ) -> (VirtIoPhysAddr, NonNull<u8>) {
        let Ok(vaddr) = global_allocator().alloc_pages(pages, 0x1000, UsageKind::Dma) else {
            return (0, NonNull::dangling());
        };
        unsafe {
            core::ptr::write_bytes(vaddr as *mut u8, 0, pages * 0x1000);
        }
        let paddr = axklib::mem::virt_to_phys(vaddr.into()).as_usize() as VirtIoPhysAddr;
        let ptr = NonNull::new(vaddr as _).expect("DMA allocator returned null");
        (paddr, ptr)
    }

    unsafe fn dma_dealloc(
        _paddr: VirtIoPhysAddr,
        vaddr: NonNull<u8>,
        pages: usize,
        _access_platform: bool,
    ) -> i32 {
        global_allocator().dealloc_pages(vaddr.as_ptr() as usize, pages, UsageKind::Dma);
        0
    }

    unsafe fn mmio_phys_to_virt(paddr: VirtIoPhysAddr, size: usize) -> NonNull<u8> {
        axklib::mmio::ioremap_raw((paddr as usize).into(), size)
            .map(|mmio| mmio.as_nonnull_ptr())
            .expect("failed to map VirtIO MMIO")
    }

    unsafe fn share(
        buffer: NonNull<[u8]>,
        direction: BufferDirection,
        _access_platform: bool,
    ) -> VirtIoPhysAddr {
        let vaddr = buffer.as_ptr() as *mut u8 as usize;
        // Direct path: buffers whose pages are physically contiguous are
        // shared zero-copy at their true physical address (no staging, no
        // copy). The page-table query replaces the linear-offset formula,
        // which silently mistranslates vmap-window addresses.
        if let Some(paddr) = contiguous_dma_paddr(vaddr, buffer.len()) {
            return paddr as VirtIoPhysAddr;
        }
        // Scattered (e.g. vmap'd task stacks): a pool slot first, dedicated
        // pages as the backstop. There is deliberately no formula fallback:
        // `Hal::share` cannot report failure, and handing the device an
        // unverified address would silently corrupt kernel memory.
        let staged = unsafe {
            bounce_pool_init()
                .and_then(|pool| bounce_stage(pool, buffer, direction))
                .or_else(|| fresh_stage(buffer, direction))
        };
        match staged {
            Some(paddr) => paddr as VirtIoPhysAddr,
            None => panic!(
                "virtio: cannot DMA-map borrowed buffer at {vaddr:#x}: staging allocation failed \
                 and an unverified device address would corrupt memory"
            ),
        }
    }

    unsafe fn unshare(
        paddr: VirtIoPhysAddr,
        buffer: NonNull<[u8]>,
        direction: BufferDirection,
        _access_platform: bool,
    ) {
        // Read-only load: never initializes the pool, so completing a queue
        // full of direct-mapped descriptors never allocates or locks.
        if let Some(pool) = bounce_pool_ready() {
            unsafe { bounce_unstage(pool, paddr as usize, buffer, direction) };
        }
    }
}

pub fn probe_mmio_device(
    reg_base: *mut u8,
    reg_size: usize,
) -> Option<(DeviceType, MmioTransport<'static>)> {
    if reg_base.is_null() || reg_size == 0 {
        return None;
    }

    let header = NonNull::new(reg_base as *mut virtio_drivers::transport::mmio::VirtIOHeader)?;
    let transport = unsafe { MmioTransport::new(header, reg_size) }.ok()?;
    Some((transport.device_type(), transport))
}

pub fn register_static_mmio(
    plat_dev: rdrive::PlatformDevice,
    base: usize,
    size: usize,
) -> Result<(), rdrive::probe::OnProbeError> {
    if !has_static_mmio_drivers() {
        return Err(rdrive::probe::OnProbeError::NotMatch);
    }

    let mmio = axklib::mmio::ioremap_raw(base.into(), size).map_err(|err| {
        rdrive::probe::OnProbeError::other(alloc::format!(
            "failed to map virtio-mmio {base:#x}: {err:?}",
        ))
    })?;
    let Some((ty, transport)) = probe_mmio_device(mmio.as_ptr(), size) else {
        return Err(rdrive::probe::OnProbeError::NotMatch);
    };
    register_static_transport(plat_dev, ty, transport)
}

#[cfg(any(
    feature = "virtio-net",
    feature = "virtio-gpu",
    feature = "virtio-input",
    feature = "virtio-socket",
))]
pub fn register_static_transport<T: Transport + 'static>(
    _plat_dev: rdrive::PlatformDevice,
    ty: DeviceType,
    _transport: T,
) -> Result<(), rdrive::probe::OnProbeError> {
    match ty {
        #[cfg(feature = "virtio-net")]
        DeviceType::Network => net::register_transport(_plat_dev, _transport),
        #[cfg(feature = "virtio-gpu")]
        DeviceType::GPU => display::register_transport(_plat_dev, _transport),
        #[cfg(feature = "virtio-input")]
        DeviceType::Input => input::register_transport(_plat_dev, _transport),
        #[cfg(feature = "virtio-socket")]
        DeviceType::Socket => Err(rdrive::probe::OnProbeError::other(
            "virtio-socket requires an explicit IRQ binding",
        )),
        _ => Err(rdrive::probe::OnProbeError::NotMatch),
    }
}

#[cfg(not(any(
    feature = "virtio-net",
    feature = "virtio-gpu",
    feature = "virtio-input",
    feature = "virtio-socket",
)))]
pub fn register_static_transport<T: Transport + 'static>(
    _plat_dev: rdrive::PlatformDevice,
    _ty: DeviceType,
    _transport: T,
) -> Result<(), rdrive::probe::OnProbeError> {
    Err(rdrive::probe::OnProbeError::NotMatch)
}

pub fn probe_fdt_mmio_device(
    info: &rdrive::register::FdtInfo<'_>,
) -> Result<(DeviceType, MmioTransport<'static>), rdrive::probe::OnProbeError> {
    let base_reg = info.node.regs().into_iter().next().ok_or_else(|| {
        rdrive::probe::OnProbeError::other(alloc::format!("[{}] has no reg", info.node.name()))
    })?;

    let mmio_size = base_reg.size.unwrap_or(0x1000) as usize;
    log::info!(
        "probing virtio-mmio node {} at PA {:#x}, size {:#x}",
        info.node.name(),
        base_reg.address,
        mmio_size
    );
    let mmio_base = crate::mmio::iomap(base_reg.address as usize, mmio_size)?.as_ptr();
    log::info!("mapped virtio-mmio at VA {mmio_base:p}");
    probe_mmio_device(mmio_base, mmio_size).ok_or(rdrive::probe::OnProbeError::NotMatch)
}

#[cfg(feature = "virtio-net")]
pub fn map_virtio_error(err: VirtIoError) -> &'static str {
    match err {
        VirtIoError::QueueFull => "virtio queue full",
        VirtIoError::NotReady => "virtio device not ready",
        VirtIoError::WrongToken => "virtio queue returned a wrong token",
        VirtIoError::AlreadyUsed => "virtio resource is already used",
        VirtIoError::InvalidParam => "virtio invalid parameter",
        VirtIoError::DmaError => "virtio DMA error",
        VirtIoError::IoError => "virtio I/O error",
        VirtIoError::Unsupported => "virtio operation unsupported",
        VirtIoError::ConfigSpaceTooSmall => "virtio config space too small",
        VirtIoError::ConfigSpaceMissing => "virtio config space missing",
        VirtIoError::SocketDeviceError(_) => "virtio socket device error",
    }
}

#[cfg(feature = "virtio-net")]
pub trait VirtIoTransport: Transport + 'static {}

#[cfg(feature = "virtio-net")]
impl<T: Transport + 'static> VirtIoTransport for T {}
