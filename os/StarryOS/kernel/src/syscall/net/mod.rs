mod addr;
mod cmsg;
mod io;
mod name;
mod opt;
mod socket;

pub use self::{cmsg::*, io::*, name::*, opt::*, socket::*};

/// [run6g] monotonic ns of the last unix-stream peer wake (read by do_poll
/// to measure wake-to-return latency).
pub fn unix_stream_last_peer_wake_ns() -> u64 {
    ax_net::unix::last_peer_wake_ns()
}
