//! engine-physics — rigid bodies, gravity, collisions, and force generators.
//!
//! Collisions:
//! - Sphere–sphere (with impulse response)
//! - AABB–AABB (minimum translation vector on smallest axis)
//! - Sphere–AABB (closest-point-on-box)
//!
//! Force generators:
//! - Wind (with turbulence)
//! - Explosion (radial impulse with distance + time decay)
//! - Point gravity (inverse-square)
//!
//! For real production physics, swap this module out for `rapier3d` — the
//! rest of the engine doesn't care.

pub mod body;
pub mod collider;
pub mod forces;
pub mod systems;

pub use body::{IntegrationMode, RigidBody};
pub use collider::{ColliderAabb, ColliderSphere};
pub use forces::{ExplosionForce, ForceGenerator, ForceRegistry, PointGravityForce, WindForce};
pub use systems::{
    floor_clamp, integrate, resolve_aabb_aabb, resolve_sphere_aabb, resolve_sphere_sphere,
    sphere_aabb, step_gravity, step_world,
};

use glam::Vec3;

/// Default gravitational acceleration (m/s²). Matches Earth's gravity, -Y direction.
pub const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

/// Marker component: entities with this are skipped by the physics integrator
/// (e.g. the player camera, triggers, lights).
#[derive(Copy, Clone, Debug, Default)]
pub struct PhysicsIgnore;
