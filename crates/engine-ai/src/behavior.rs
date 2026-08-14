//! Minimal behavior tree — Sequence, Selector, and Action nodes.

use engine_core::{Entity, Time, World};

/// Result of a single behavior tick.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BehaviorResult {
    /// The node finished successfully.
    Success,
    /// The node finished in failure.
    Failure,
    /// The node is still running — tick again next frame.
    Running,
}

/// A behavior tree node.
pub trait Behavior: Send + Sync {
    fn tick(&mut self, world: &mut World, entity: Entity, time: &Time) -> BehaviorResult;
    fn name(&self) -> &str {
        "behavior"
    }
}

/// Convenience alias for older code that referenced `BehaviorStatus`.
pub type BehaviorStatus = BehaviorResult;

/// Sequence: runs children in order. Stops on first `Failure` or `Running`.
/// Returns `Success` only when all children return `Success`.
pub struct Sequence {
    pub children: Vec<Box<dyn Behavior>>,
    pub current: usize,
}

impl Sequence {
    pub fn new(children: Vec<Box<dyn Behavior>>) -> Self {
        Self {
            children,
            current: 0,
        }
    }
}

impl Behavior for Sequence {
    fn tick(&mut self, world: &mut World, entity: Entity, time: &Time) -> BehaviorResult {
        while self.current < self.children.len() {
            let r = self.children[self.current].tick(world, entity, time);
            match r {
                BehaviorResult::Success => {
                    self.current += 1;
                }
                _ => return r,
            }
        }
        self.current = 0;
        BehaviorResult::Success
    }
    fn name(&self) -> &str {
        "sequence"
    }
}

/// Selector: runs children in order. Stops on first `Success`. Returns `Failure`
/// only when all children fail. (a.k.a. "fallback" node.)
pub struct Selector {
    pub children: Vec<Box<dyn Behavior>>,
    pub current: usize,
}

impl Selector {
    pub fn new(children: Vec<Box<dyn Behavior>>) -> Self {
        Self {
            children,
            current: 0,
        }
    }
}

impl Behavior for Selector {
    fn tick(&mut self, world: &mut World, entity: Entity, time: &Time) -> BehaviorResult {
        // Try each child; if any returns Success, we're done. If Running, pause.
        for i in 0..self.children.len() {
            let r = self.children[i].tick(world, entity, time);
            match r {
                BehaviorResult::Failure => continue,
                _ => {
                    self.current = i;
                    return r;
                }
            }
        }
        self.current = 0;
        BehaviorResult::Failure
    }
    fn name(&self) -> &str {
        "selector"
    }
}

/// Wrap a closure into a `Behavior`.
pub struct Action<F>
where
    F: FnMut(&mut World, Entity, &Time) -> BehaviorResult + Send + Sync,
{
    pub func: F,
    pub name: &'static str,
}

impl<F> Action<F>
where
    F: FnMut(&mut World, Entity, &Time) -> BehaviorResult + Send + Sync,
{
    pub fn new(name: &'static str, func: F) -> Self {
        Self { func, name }
    }
}

impl<F> Behavior for Action<F>
where
    F: FnMut(&mut World, Entity, &Time) -> BehaviorResult + Send + Sync,
{
    fn tick(&mut self, world: &mut World, entity: Entity, time: &Time) -> BehaviorResult {
        (self.func)(world, entity, time)
    }
    fn name(&self) -> &str {
        self.name
    }
}
