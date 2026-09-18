//! Direct-mappability of borrowed DMA buffers.
//!
//! A virtio descriptor is a single flat bus address plus a length, so a
//! buffer can be shared zero-copy only when its pages are physically
//! contiguous. This module walks borrowed buffers through `translate` —
//! expected to be the kernel page-table query, never the linear-offset
//! formula, which silently mistranslates vmap-window addresses such as task
//! stacks — and reports whether a zero-copy share is possible.
//!
//! Kept free of kernel-runtime dependencies so the host integration test
//! (`tests/direct_map_walk.rs`) can include this file verbatim.

use ax_memory_addr::{PAGE_SIZE_4K, PhysAddr, VirtAddr};

/// Walks `vaddr..vaddr + len` through `translate` page by page and returns
/// the buffer's physical base when every page resolves and the physical
/// frames are contiguous — the condition for a single flat descriptor.
///
/// `None` means the buffer cannot be direct-mapped: some page is not
/// translatable, or the frames are physically scattered (vmap-style).
/// Zero-length spans cannot be direct-mapped and return `None` as well.
pub(super) fn contiguous_dma_paddr_with(
    translate: impl Fn(VirtAddr) -> Option<PhysAddr>,
    vaddr: usize,
    len: usize,
) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let mut base = None;
    let mut off = 0usize;
    while off < len {
        let va = vaddr + off;
        let pa = translate(va.into())?.as_usize();
        match base {
            None => base = Some(pa),
            Some(first) => {
                if pa != first + off {
                    return None;
                }
            }
        }
        let in_page = va % PAGE_SIZE_4K;
        off += core::cmp::min(PAGE_SIZE_4K - in_page, len - off);
    }
    base
}
