//! Script registry — name → factory mapping for data-driven instantiation.

use crate::{Script, ScriptFactory};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// A thread-safe registry of script factories.
#[derive(Default, Clone)]
pub struct ScriptRegistry {
    inner: Arc<RwLock<HashMap<String, Arc<ScriptFactory>>>>,
}

impl ScriptRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a script factory under a string name. The factory is called
    /// whenever a scene file references the script by name.
    pub fn register<F>(&self, name: impl Into<String>, build: F)
    where
        F: Fn() -> Box<dyn Script> + Send + Sync + 'static,
    {
        self.inner.write().insert(
            name.into(),
            Arc::new(ScriptFactory {
                build: Box::new(build),
            }),
        );
    }

    /// Instantiate a script by name. Returns `None` if no factory is registered
    /// for `name`.
    pub fn instantiate(&self, name: &str) -> Option<Box<dyn Script>> {
        let inner = self.inner.read();
        inner.get(name).map(|f| (f.build)())
    }

    /// All registered script names — useful for tooling / editor autocomplete.
    pub fn names(&self) -> Vec<String> {
        self.inner.read().keys().cloned().collect()
    }
}
