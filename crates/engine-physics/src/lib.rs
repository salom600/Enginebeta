//! engine-physics — rigid bodies, gravity, and sphere–sphere collision resolution.
//!
//! This is a deliberately small physics core: enough to make things feel "game-y"
//! without pulling in a 50k-LOC dependency. It runs on the fixed timestep
//! provided by [`engine_core::App`].
//!
//! Components:
//! - [`RigidBody`] — linear + angular velocity, mass, gravity scale
//! - [`ColliderSphere`] — spherical collider
//! - [`ColliderAabb`] — axis-aligned box collider
//!
//! Systems:
//! - [`step_gravity`] — applies gravity to every rigid body
//! - [`integrate`] — moves each rigid body by `velocity * dt`
//! - [`resolve_sphere_sphere`] — pushes overlapping pairs apart and applies
//!   a simple impulse response (no rotation, no friction)
//!
//! For real production physics, swap this module out for `rapier3d` — the
//! rest of the engine doesn't care.

pub mod body;
pub mod collider;
pub mod systems;

pub use body::{IntegrationMode, RigidBody};
pub use collider::{ColliderAabb, ColliderSphere};
pub use systems::{floor_clamp, integrate, resolve_sphere_sphere, step_gravity, step_world};

use glam::Vec3;

/// Default gravitational acceleration (m/s²). Matches Earth's gravity, -Y direction.
pub const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

/// Marker component: entities with this are skipped by the physics integrator
/// (e.g. the player camera, triggers, lights).
#[derive(Copy, Clone, Debug, Default)]
pub struct PhysicsIgnore;
