//! Generic asset loader trait — implementations decode bytes into an `A`.

use std::path::Path;

/// A loader for a specific asset type `A`. Implementations decode bytes
/// (optionally using the file extension) into the asset plus its byte size.
pub trait AssetLoader<A>: Send + Sync {
    fn load(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<(A, usize)>;
    /// File extensions this loader accepts (e.g. `["png", "jpg"]`).
    fn extensions(&self) -> &'static [&'static str];
}
