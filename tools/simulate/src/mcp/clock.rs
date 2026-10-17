//! A shared atomic clock makes receipt timestamps independent of RPC polling.
use color_eyre::Result;
use std::{
    fs::OpenOptions,
    path::Path,
    sync::atomic::{AtomicI64, Ordering},
};

pub(super) struct SharedClock(memmap2::MmapMut);
impl SharedClock {
    pub fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        file.set_len(8)?;
        // The private session file remains eight bytes long for both processes.
        // mmap is page aligned, so the atomic below is aligned on supported 64-bit hosts.
        Ok(Self(unsafe {
            memmap2::MmapOptions::new().len(8).map_mut(&file)?
        }))
    }
    fn atomic(&self) -> &AtomicI64 {
        // Every reader and writer accesses these bytes only through this atomic.
        unsafe { &*self.0.as_ptr().cast::<AtomicI64>() }
    }
    pub fn load(&self) -> i64 {
        self.atomic().load(Ordering::Acquire)
    }
    pub fn store(&self, time: i64) {
        self.atomic().store(time, Ordering::Release);
    }
}
