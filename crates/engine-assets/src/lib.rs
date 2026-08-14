//! engine-assets — asset loading + memory management for EngineBeta.
//!
//! Responsibilities:
//! - Load bytes from disk (or embedded) and cache them by handle.
//! - Reference-count assets so unused ones can be evicted.
//! - Hot-reload: watch files for changes and refresh the cache.
//! - Memory budget: track bytes used per category and reject loads over budget.
//!
//! The asset store is generic over the asset type `A`. Different stores are
//! used for different asset kinds (textures, meshes, sounds, scenes).

use anyhow::Context as _;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

pub mod handle;
pub mod loader;
pub mod memory;

pub use handle::{AssetHandle, AssetPath};
pub use loader::AssetLoader;
pub use memory::MemoryBudget;

/// A loaded asset stored in the cache. Each entry tracks its byte size and
/// last-modified time on disk (for hot reload).
pub struct AssetEntry<A> {
    pub asset: A,
    pub bytes: usize,
    pub source_mtime: Option<SystemTime>,
    pub source_path: Option<PathBuf>,
}

/// The asset store. One per asset kind (textures, sounds, etc.).
pub struct AssetStore<A> {
    inner: Arc<RwLock<AssetStoreInner<A>>>,
}

struct AssetStoreInner<A> {
    entries: HashMap<AssetPath, Arc<AssetEntry<A>>>,
    budget: MemoryBudget,
}

impl<A> AssetStore<A> {
    pub fn new(budget: MemoryBudget) -> Self {
        Self {
            inner: Arc::new(RwLock::new(AssetStoreInner {
                entries: HashMap::new(),
                budget,
            })),
        }
    }

    /// Look up an asset by handle. Returns a snapshot `Arc` to the entry.
    pub fn get(&self, handle: &AssetHandle) -> Option<Arc<AssetEntry<A>>> {
        self.inner.read().entries.get(&handle.path).cloned()
    }

    /// Insert (or replace) an asset in the cache. Returns a handle to it.
    pub fn insert(&self, path: AssetPath, asset: A, bytes: usize) -> AssetHandle {
        let mut inner = self.inner.write();
        // If we're over budget, evict the oldest entry (very simple LRU-ish —
        // for real production you'd track access time).
        if inner.budget.would_exceed(bytes) {
            if let Some(victim) = inner.entries.keys().next().cloned() {
                if let Some(entry) = inner.entries.remove(&victim) {
                    inner.budget.deallocate(entry.bytes);
                    log::debug!("evicted asset {:?} to free {} bytes", victim, entry.bytes);
                }
            }
        }
        inner.budget.allocate(bytes);
        let entry = Arc::new(AssetEntry {
            asset,
            bytes,
            source_mtime: None,
            source_path: None,
        });
        inner.entries.insert(path.clone(), entry);
        AssetHandle { path }
    }

    /// Insert an asset that was loaded from `disk_path`, capturing the mtime
    /// so hot-reload can detect changes.
    pub fn insert_from_disk(
        &self,
        disk_path: PathBuf,
        asset: A,
        bytes: usize,
    ) -> AssetHandle {
        let mtime = std::fs::metadata(&disk_path)
            .and_then(|m| m.modified())
            .ok();
        let path = AssetPath::from_disk(disk_path.clone());
        let mut inner = self.inner.write();
        if inner.budget.would_exceed(bytes) {
            if let Some(victim) = inner.entries.keys().next().cloned() {
                if let Some(entry) = inner.entries.remove(&victim) {
                    inner.budget.deallocate(entry.bytes);
                    log::debug!("evicted asset {:?} to free {} bytes", victim, entry.bytes);
                }
            }
        }
        inner.budget.allocate(bytes);
        let entry = Arc::new(AssetEntry {
            asset,
            bytes,
            source_mtime: mtime,
            source_path: Some(disk_path),
        });
        inner.entries.insert(path.clone(), entry);
        AssetHandle { path }
    }

    /// Remove an asset from the cache (drops it if no other Arcs reference it).
    pub fn remove(&self, path: &AssetPath) {
        let mut inner = self.inner.write();
        if let Some(entry) = inner.entries.remove(path) {
            inner.budget.deallocate(entry.bytes);
        }
    }

    /// Number of assets currently in the cache.
    pub fn len(&self) -> usize {
        self.inner.read().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total bytes used by all cached assets.
    pub fn bytes_used(&self) -> usize {
        self.inner.read().budget.used()
    }

    /// Hot-reload check: re-read files whose mtime changed since load.
    /// Returns the number of assets reloaded.
    ///
    /// `reload_fn` decodes the bytes into a new `A` (e.g. `image::load_from_memory`
    /// for textures).
    pub fn hot_reload<F>(&self, reload_fn: F) -> usize
    where
        F: Fn(&Path, &[u8]) -> anyhow::Result<(A, usize)>,
    {
        let mut inner = self.inner.write();
        let mut reloaded = 0;
        let paths: Vec<AssetPath> = inner.entries.keys().cloned().collect();
        for path in paths {
            let Some(disk_path) = path.as_disk_path() else {
                continue;
            };
            let Ok(meta) = std::fs::metadata(disk_path) else {
                continue;
            };
            let Ok(mtime) = meta.modified() else {
                continue;
            };
            let prev_mtime = inner
                .entries
                .get(&path)
                .and_then(|e| e.source_mtime)
                .unwrap_or(SystemTime::UNIX_EPOCH);
            if mtime <= prev_mtime {
                continue;
            }
            // Reload.
            let bytes = match std::fs::read(disk_path) {
                Ok(b) => b,
                Err(e) => {
                    log::warn!("hot reload read failed for {:?}: {e}", path);
                    continue;
                }
            };
            match reload_fn(disk_path, &bytes) {
                Ok((asset, size)) => {
                    let old_bytes = inner
                        .entries
                        .get(&path)
                        .map(|e| e.bytes)
                        .unwrap_or(0);
                    inner.budget.deallocate(old_bytes);
                    inner.budget.allocate(size);
                    let entry = Arc::new(AssetEntry {
                        asset,
                        bytes: size,
                        source_mtime: Some(mtime),
                        source_path: Some(disk_path.to_path_buf()),
                    });
                    inner.entries.insert(path, entry);
                    reloaded += 1;
                }
                Err(e) => {
                    log::warn!("hot reload decode failed for {:?}: {e}", path);
                }
            }
        }
        reloaded
    }
}

/// Convenience: load bytes from disk into an [`AssetStore`] via a loader function.
pub fn load_file<A, F>(
    store: &AssetStore<A>,
    path: impl Into<PathBuf>,
    loader: F,
) -> anyhow::Result<AssetHandle>
where
    F: FnOnce(&[u8]) -> anyhow::Result<(A, usize)>,
{
    let path_buf: PathBuf = path.into();
    let bytes = std::fs::read(&path_buf).with_context(|| {
        format!("failed to read asset file: {}", path_buf.display())
    })?;
    let (asset, size) = loader(&bytes)?;
    Ok(store.insert_from_disk(path_buf, asset, size))
}
