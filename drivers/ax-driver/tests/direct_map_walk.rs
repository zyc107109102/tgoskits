//! Host tests for the virtio direct-mapping walk.
//!
//! These compile the production module verbatim via `#[path]` (the crate's
//! virtio code links kernel-only ax-sync internals, so it cannot be exercised
//! as a lib unit test on the host). Each case feeds a fake page-table
//! translator modelling either the linear map window or a vmap'd task stack.

#[path = "../src/virtio/direct_map.rs"]
mod direct_map;

use ax_memory_addr::{PAGE_SIZE_4K, PhysAddr, VirtAddr};
use direct_map::contiguous_dma_paddr_with;

const LINEAR_BASE: usize = 0xffff_8880_0000_0000;
const VMAP_BASE: usize = 0xffff_c000_0000_0000;

/// Linear map window: page tables agree with the offset formula, so any span
/// inside it is physically contiguous.
fn linear_translate(va: VirtAddr) -> Option<PhysAddr> {
    Some(PhysAddr::from_usize(va.as_usize() - LINEAR_BASE))
}

/// A vmap'd task stack: every queried page is materialized at whatever frame
/// the allocator handed out — information the offset formula cannot recover.
fn vmap_stack_translate(
    stack_base: usize,
    frames: &[usize],
) -> impl Fn(VirtAddr) -> Option<PhysAddr> + '_ {
    move |va: VirtAddr| {
        let idx = (va.as_usize() - stack_base) / PAGE_SIZE_4K;
        let in_page = va.as_usize() % PAGE_SIZE_4K;
        frames
            .get(idx)
            .map(|&frame| PhysAddr::from_usize(frame + in_page))
    }
}

#[test]
fn linear_window_spans_map_direct_at_their_physical_base() {
    let va = LINEAR_BASE + 0x123_000;
    assert_eq!(
        contiguous_dma_paddr_with(linear_translate, va, 90),
        Some(va - LINEAR_BASE)
    );
    assert_eq!(
        contiguous_dma_paddr_with(linear_translate, va, 3 * PAGE_SIZE_4K + 7),
        Some(va - LINEAR_BASE)
    );
}

#[test]
fn single_page_stack_buffer_maps_direct_at_its_true_frame() {
    // Regression: a formula-based translation returns `va - LINEAR_BASE`
    // here — a frame that does not back the buffer at all, which is the Mesa
    // CONTEXT_INIT corruption. The page-table walk must keep this sub-page
    // stack buffer zero-copy at its true physical frame.
    let frames = [0x02f3_1000];
    let va = VMAP_BASE + 0x450;
    assert_eq!(
        contiguous_dma_paddr_with(vmap_stack_translate(VMAP_BASE, &frames), va, 90),
        Some(0x02f3_1000 + 0x450)
    );
}

#[test]
fn scattered_multi_page_stack_buffers_reject_direct_mapping() {
    // vmap'd stack pages are individually allocated; adjacent virtual pages
    // are physically disjoint, so a multi-page span must bounce.
    let frames = [0x1000_0000, 0x2000_0000, 0x3000_0000];
    assert_eq!(
        contiguous_dma_paddr_with(
            vmap_stack_translate(VMAP_BASE, &frames),
            VMAP_BASE,
            2 * PAGE_SIZE_4K
        ),
        None
    );
}

#[test]
fn unmaterialized_pages_reject_direct_mapping() {
    assert_eq!(
        contiguous_dma_paddr_with(|_: VirtAddr| Option::<PhysAddr>::None, VMAP_BASE, 16),
        None
    );
}

#[test]
fn zero_length_spans_reject_direct_mapping() {
    assert_eq!(
        contiguous_dma_paddr_with(linear_translate, LINEAR_BASE, 0),
        None
    );
}

#[test]
fn page_crossing_spans_check_every_page() {
    // Starts 16 bytes before a page boundary and crosses it: the walk must
    // step across and still verify the second page.
    let va = LINEAR_BASE + 0x0123_f000 - 0x10;
    assert_eq!(
        contiguous_dma_paddr_with(linear_translate, va, 0x100),
        Some(va - LINEAR_BASE)
    );
    let frames = [0x1000_0000, 0x2000_0000];
    assert_eq!(
        contiguous_dma_paddr_with(
            vmap_stack_translate(VMAP_BASE, &frames),
            VMAP_BASE + PAGE_SIZE_4K - 0x10,
            0x100
        ),
        None
    );
}
