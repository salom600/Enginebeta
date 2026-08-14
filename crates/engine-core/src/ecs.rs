//! Tiny, fast entity–component store.
//!
//! Components are stored in typed [`Column`]s (one `Vec<T>` per component type).
//! This is the same "archetype-lite" pattern used by smaller ECS engines — fast
//! iteration, cache-friendly access, no per-entity allocations.
//!
//! Because the borrow on a [`World`] is always exclusive (`&mut World`), columns
//! do NOT need their own internal locking — exclusive access is already enforced
//! by the Rust borrow checker at the system boundary.

use std::any::{Any, TypeId};
use std::collections::HashMap;

/// A lightweight entity handle. Just an index + generation (to detect stale IDs).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Entity {
    pub id: u32,
    pub gen: u32,
}

/// Trait implemented for every component type. Any `'static + Send + Sync` type
/// can be a component — there's nothing to derive.
pub trait Component: 'static + Send + Sync {}

impl<T: 'static + Send + Sync> Component for T {}

/// A typed column of component data, indexed by entity id.
pub struct Column<T> {
    pub sparse: Vec<Option<u32>>, // entity id -> dense index
    pub dense: Vec<T>,            // dense component storage
    pub entities: Vec<u32>,       // dense index -> entity id
}

impl<T> Column<T> {
    pub fn new() -> Self {
        Self {
            sparse: Vec::new(),
            dense: Vec::new(),
            entities: Vec::new(),
        }
    }

    fn ensure_capacity(&mut self, id: u32) {
        if self.sparse.len() <= id as usize {
            self.sparse.resize(id as usize + 1, None);
        }
    }

    pub fn insert(&mut self, id: u32, value: T) {
        self.ensure_capacity(id);
        if let Some(dense_idx) = self.sparse[id as usize] {
            self.dense[dense_idx as usize] = value;
        } else {
            let dense_idx = self.dense.len() as u32;
            self.dense.push(value);
            self.entities.push(id);
            self.sparse[id as usize] = Some(dense_idx);
        }
    }

    pub fn get(&self, id: u32) -> Option<&T> {
        self.sparse
            .get(id as usize)
            .and_then(|s| *s)
            .map(|i| &self.dense[i as usize])
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut T> {
        let dense_idx = *self.sparse.get(id as usize)?.as_ref()?;
        Some(&mut self.dense[dense_idx as usize])
    }

    pub fn remove(&mut self, id: u32) -> Option<T> {
        let dense_idx = self.sparse.get(id as usize).copied().flatten()?;
        self.sparse[id as usize] = None;
        let last = self.dense.len() - 1;
        let value = self.dense.swap_remove(dense_idx as usize);
        let moved_entity = self.entities.swap_remove(dense_idx as usize);
        if dense_idx as usize != last {
            self.sparse[moved_entity as usize] = Some(dense_idx);
        }
        Some(value)
    }

    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> {
        self.entities.iter().copied().zip(self.dense.iter())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u32, &mut T)> {
        self.entities.iter().copied().zip(self.dense.iter_mut())
    }

    pub fn len(&self) -> usize {
        self.dense.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }
}

impl<T> Default for Column<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Type-erased column so the world can call `remove_entity` without knowing T.
pub trait ComponentColumn: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn remove_entity(&mut self, id: u32);
    fn len(&self) -> usize;
}

impl<T: Component> ComponentColumn for Column<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn remove_entity(&mut self, id: u32) {
        let _ = self.remove(id);
    }
    fn len(&self) -> usize {
        Column::len(self)
    }
}

/// The world owns all entities and their components.
pub struct World {
    free_ids: Vec<u32>,
    generations: Vec<u32>,
    live: Vec<u32>,
    columns: HashMap<TypeId, Box<dyn ComponentColumn>>,
    pub shutdown: bool,
    /// Arbitrary per-world user data (engine modules can stash state here).
    pub resources: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl World {
    pub fn new() -> Self {
        Self {
            free_ids: Vec::new(),
            generations: Vec::new(),
            live: Vec::new(),
            columns: HashMap::new(),
            shutdown: false,
            resources: HashMap::new(),
        }
    }

    /// Spawn a new entity and return its handle.
    pub fn spawn(&mut self) -> Entity {
        let id = if let Some(id) = self.free_ids.pop() {
            id
        } else {
            let id = self.generations.len() as u32;
            self.generations.push(0);
            id
        };
        self.live.push(id);
        Entity {
            id,
            gen: self.generations[id as usize],
        }
    }

    /// Despawn an entity. Components for that entity are removed from every column.
    pub fn despawn(&mut self, e: Entity) -> bool {
        if !self.is_alive(e) {
            return false;
        }
        self.generations[e.id as usize] += 1;
        self.free_ids.push(e.id);
        if let Some(pos) = self.live.iter().position(|id| *id == e.id) {
            self.live.swap_remove(pos);
        }
        for (_tid, col) in self.columns.iter_mut() {
            col.remove_entity(e.id);
        }
        true
    }

    pub fn is_alive(&self, e: Entity) -> bool {
        self.generations
            .get(e.id as usize)
            .map(|g| *g == e.gen)
            .unwrap_or(false)
    }

    /// Insert a component onto an entity. The world must own that entity.
    pub fn insert<T: Component>(&mut self, e: Entity, value: T) -> bool {
        if !self.is_alive(e) {
            return false;
        }
        self.column_mut::<T>().insert(e.id, value);
        true
    }

    /// Borrow a component for an entity (clones out of the column).
    pub fn get<T: Component + Clone>(&self, e: Entity) -> Option<T> {
        let col = self.column::<T>()?;
        col.get(e.id).cloned()
    }

    /// Mutate a component in place via a closure.
    pub fn with<T: Component, R, F: FnOnce(Option<&mut T>) -> R>(&mut self, e: Entity, f: F) -> R {
        f(self.column_mut::<T>().get_mut(e.id))
    }

    /// Register a component type explicitly. Not strictly required (insert auto-registers).
    pub fn register<T: Component>(&mut self) {
        self.column_mut::<T>();
    }

    fn column<T: Component>(&self) -> Option<&Column<T>> {
        let any = self.columns.get(&TypeId::of::<T>())?;
        any.as_any().downcast_ref::<Column<T>>()
    }

    fn column_mut<T: Component>(&mut self) -> &mut Column<T> {
        let entry = self.columns.entry(TypeId::of::<T>());
        let boxed = entry.or_insert_with(|| Box::new(Column::<T>::new()));
        boxed
            .as_any_mut()
            .downcast_mut::<Column<T>>()
            .expect("type mismatch in column storage")
    }

    /// Borrow a whole column mutably for system-level iteration.
    pub fn column_write<T: Component>(&mut self) -> &mut Column<T> {
        self.column_mut::<T>()
    }

    /// Borrow a whole column immutably for system-level iteration.
    pub fn column_read<T: Component>(&self) -> Option<&Column<T>> {
        self.column::<T>()
    }

    /// Get a raw mutable pointer to a column. Allows multiple columns to be
    /// borrowed simultaneously when the caller can prove the types are distinct.
    ///
    /// # Safety
    /// The returned pointer is valid as long as `&mut self` is held. The caller
    /// MUST NOT alias the same `T` (i.e. call `column_ptr::<T>()` twice and
    /// dereference both) — that would create overlapping `&mut` references and
    /// is undefined behavior. Different `T`s are always distinct memory regions
    /// (keyed by `TypeId`), so borrowing `Column<A>` and `Column<B>` simultaneously
    /// (where `A != B`) is sound.
    pub fn column_ptr<T: Component>(&mut self) -> *mut Column<T> {
        self.column_mut::<T>() as *mut Column<T>
    }

    /// Borrow two columns mutably at the same time.
    ///
    /// # Panics
    /// Panics at runtime if `A` and `B` are the same type (would create aliasing).
    pub fn columns2<A: Component, B: Component, R, F: FnOnce(&mut Column<A>, &mut Column<B>) -> R>(
        &mut self,
        f: F,
    ) -> R {
        assert!(
            TypeId::of::<A>() != TypeId::of::<B>(),
            "columns2 called with the same component type twice"
        );
        let a = self.column_ptr::<A>();
        let b = self.column_ptr::<B>();
        // SAFETY: A and B are different types -> different HashMap entries -> disjoint memory.
        unsafe { f(&mut *a, &mut *b) }
    }

    /// Borrow three columns mutably at the same time. Same rules as [`World::columns2`].
    pub fn columns3<
        A: Component,
        B: Component,
        C: Component,
        R,
        F: FnOnce(&mut Column<A>, &mut Column<B>, &mut Column<C>) -> R,
    >(
        &mut self,
        f: F,
    ) -> R {
        assert!(
            TypeId::of::<A>() != TypeId::of::<B>()
                && TypeId::of::<A>() != TypeId::of::<C>()
                && TypeId::of::<B>() != TypeId::of::<C>(),
            "columns3 called with duplicate component types"
        );
        let a = self.column_ptr::<A>();
        let b = self.column_ptr::<B>();
        let c = self.column_ptr::<C>();
        // SAFETY: all three are different types -> disjoint memory regions.
        unsafe { f(&mut *a, &mut *b, &mut *c) }
    }

    /// Iterate all currently-alive entities.
    pub fn entities_alive(&self) -> impl Iterator<Item = Entity> + '_ {
        self.live.iter().copied().map(move |id| Entity {
            id,
            gen: self.generations[id as usize],
        })
    }

    /// Number of currently-spawned entities.
    pub fn entity_count(&self) -> usize {
        self.live.len()
    }

    /// Store arbitrary per-world data (engine modules use this to stash their state).
    pub fn insert_resource<R: 'static + Send + Sync>(&mut self, r: R) {
        self.resources.insert(TypeId::of::<R>(), Box::new(r));
    }

    pub fn resource<R: 'static + Send + Sync>(&self) -> Option<&R> {
        self.resources.get(&TypeId::of::<R>())?.downcast_ref::<R>()
    }

    pub fn resource_mut<R: 'static + Send + Sync>(&mut self) -> Option<&mut R> {
        self.resources
            .get_mut(&TypeId::of::<R>())?
            .downcast_mut::<R>()
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_and_is_alive() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.is_alive(e));
        assert_eq!(w.entity_count(), 1);
    }

    #[test]
    fn despawn_bumps_generation() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.despawn(e));
        assert!(!w.is_alive(e));
        // The id may be reused, but the old handle is stale.
        let e2 = w.spawn();
        assert_ne!(e.gen, e2.gen);
    }

    #[test]
    fn insert_and_get_component() {
        #[derive(Copy, Clone, Debug, PartialEq)]
        struct Health(i32);

        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Health(42));
        assert_eq!(w.get::<Health>(e), Some(Health(42)));
    }

    #[test]
    fn columns2_allows_two_mut_borrows() {
        #[derive(Copy, Clone, Debug, PartialEq)]
        struct A(i32);
        #[derive(Copy, Clone, Debug, PartialEq)]
        struct B(i32);

        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        w.insert(e1, A(1));
        w.insert(e1, B(10));
        w.insert(e2, A(2));
        w.insert(e2, B(20));

        w.columns2::<A, B, _, _>(|col_a, col_b| {
            assert_eq!(col_a.get(e1.id).unwrap().0, 1);
            assert_eq!(col_b.get(e2.id).unwrap().0, 20);
            // Mutate both
            col_a.get_mut(e1.id).unwrap().0 = 100;
            col_b.get_mut(e2.id).unwrap().0 = 200;
        });
        assert_eq!(w.get::<A>(e1).unwrap().0, 100);
        assert_eq!(w.get::<B>(e2).unwrap().0, 200);
    }

    #[test]
    fn despawn_removes_components() {
        #[derive(Copy, Clone, Debug, PartialEq)]
        struct Marker(u32);

        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Marker(7));
        assert_eq!(w.get::<Marker>(e), Some(Marker(7)));
        w.despawn(e);
        // Column should no longer have this entity.
        let col = w.column_read::<Marker>().unwrap();
        assert!(col.get(e.id).is_none());
    }
}
