//! engine-ai — pathfinding and behavior trees for EngineBeta.
//!
//! Includes:
//! - [`pathfinding::astar_grid`] — A* on a 2D grid (good for top-down games)
//! - [`pathfinding::astar_world`] — A* on a navigation graph
//! - [`behavior`] — minimal behavior tree (Sequence / Selector / Action)
//! - [`steering`] — steering behaviors (seek / flee / wander)
//!
//! All algorithms are pure-Rust, no allocations beyond the open/closed sets.

pub mod behavior;
pub mod pathfinding;
pub mod steering;

pub use behavior::{Behavior, BehaviorResult, BehaviorStatus, Sequence, Selector};
pub use pathfinding::{astar_grid, astar_world, Grid};
pub use steering::{flee, seek, wander};
