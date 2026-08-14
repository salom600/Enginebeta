//! Memory budget tracker — caps the total bytes a single `AssetStore` may use.

/// Tracks bytes allocated / deallocated against a hard cap.
#[derive(Copy, Clone, Debug)]
pub struct MemoryBudget {
    pub limit_bytes: usize,
    pub used_bytes: usize,
}

impl MemoryBudget {
    pub fn new(limit_bytes: usize) -> Self {
        Self {
            limit_bytes,
            used_bytes: 0,
        }
    }

    /// Default: 256 MiB per asset category.
    pub fn default_256mb() -> Self {
        Self::new(256 * 1024 * 1024)
    }

    pub fn allocate(&mut self, bytes: usize) {
        self.used_bytes += bytes;
    }

    pub fn deallocate(&mut self, bytes: usize) {
        self.used_bytes = self.used_bytes.saturating_sub(bytes);
    }

    /// True if allocating `bytes` would push us over the limit.
    pub fn would_exceed(&self, bytes: usize) -> bool {
        self.used_bytes + bytes > self.limit_bytes
    }

    pub fn used(&self) -> usize {
        self.used_bytes
    }

    pub fn available(&self) -> usize {
        self.limit_bytes.saturating_sub(self.used_bytes)
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::default_256mb()
    }
}
