//! Asset handles — typed references into the asset store.

use std::path::PathBuf;

/// A typed handle to an asset in the cache. Cheap to clone (just a path).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct AssetHandle {
    pub path: AssetPath,
}

/// An asset key — either a path on disk or a synthetic name for embedded data.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum AssetPath {
    /// Loaded from a file on disk.
    Disk(PathBuf),
    /// Embedded / synthetic — looked up by string name.
    Embedded(String),
}

impl AssetPath {
    pub fn from_disk(p: impl Into<PathBuf>) -> Self {
        Self::Disk(p.into())
    }
    pub fn embedded(s: impl Into<String>) -> Self {
        Self::Embedded(s.into())
    }

    /// If this is a `Disk` path, return the underlying `&Path`.
    pub fn as_disk_path(&self) -> Option<&std::path::Path> {
        match self {
            AssetPath::Disk(p) => Some(p.as_path()),
            AssetPath::Embedded(_) => None,
        }
    }
}

impl std::fmt::Display for AssetPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssetPath::Disk(p) => write!(f, "disk:{}", p.display()),
            AssetPath::Embedded(s) => write!(f, "embedded:{s}"),
        }
    }
}
