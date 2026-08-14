//! engine-ai — pathfinding, behavior trees, steering, and perception.
//!
//! v0.2.0 additions:
//! - `perception` — vision (FOV + line-of-sight) and hearing sensors
//! - `steering::arrive`, `pursue`, `orbit`, `smooth_velocity` — smoother movement
//!
//! All algorithms are pure-Rust, no allocations beyond the open/closed sets.

pub mod behavior;
pub mod pathfinding;
pub mod perception;
pub mod steering;

pub use behavior::{Behavior, BehaviorResult, BehaviorStatus, Sequence, Selector};
pub use pathfinding::{astar_grid, astar_world, Grid};
pub use perception::{
    can_hear, can_see, perception_system, Alerted, HearingSensor, LastKnownPosition,
    Perceivable, SoundEvent, VisionSensor,
};
pub use steering::{arrive, flee, orbit, pursue, seek, smooth_velocity, wander};
