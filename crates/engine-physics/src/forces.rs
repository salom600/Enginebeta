//! Force generators — apply forces to rigid bodies over time.
//!
//! Built-in generators:
//! - [`WindForce`] — constant directional force (with optional turbulence)
//! - [`ExplosionForce`] — radial impulse that decays with distance + time
//! - [`PointGravityForce`] — pulls bodies toward a point (e.g. a black hole)
//!
//! A [`ForceRegistry`] pairs force generators with the entities they affect.
//! Call `update(world, dt)` each fixed step to apply all registered forces.

use engine_core::{Entity, World};
use glam::Vec3;

use crate::{IntegrationMode, RigidBody};

/// A force generator produces a force (in Newtons) on a body at a given time.
pub trait ForceGenerator: Send + Sync {
    fn force_on(&self, body: &RigidBody, position: Vec3, time_secs: f32) -> Vec3;
}

/// Constant directional force (wind). Optionally turbulence adds noise.
pub struct WindForce {
    pub direction: Vec3,
    pub strength: f32,
    pub turbulence: f32,
}

impl WindForce {
    pub fn new(direction: Vec3, strength: f32) -> Self {
        Self {
            direction: direction.normalize(),
            strength,
            turbulence: 0.0,
        }
    }

    pub fn with_turbulence(mut self, t: f32) -> Self {
        self.turbulence = t;
        self
    }
}

impl ForceGenerator for WindForce {
    fn force_on(&self, _body: &RigidBody, _position: Vec3, time_secs: f32) -> Vec3 {
        // Sinusoidal turbulence modulates the strength over time.
        let turb = 1.0 + self.turbulence * (time_secs * 3.0).sin();
        self.direction * self.strength * turb
    }
}

/// Radial impulse from a point in space. Decays with distance squared and
/// fades to zero after `duration` seconds.
pub struct ExplosionForce {
    pub origin: Vec3,
    pub strength: f32,
    pub radius: f32,
    pub start_time: f32,
    pub duration: f32,
}

impl ExplosionForce {
    pub fn new(origin: Vec3, strength: f32, radius: f32, start_time: f32) -> Self {
        Self {
            origin,
            strength,
            radius,
            start_time,
            duration: 0.5,
        }
    }
}

impl ForceGenerator for ExplosionForce {
    fn force_on(&self, _body: &RigidBody, position: Vec3, time_secs: f32) -> Vec3 {
        let elapsed = time_secs - self.start_time;
        if elapsed < 0.0 || elapsed > self.duration {
            return Vec3::ZERO;
        }
        let to_body = position - self.origin;
        let dist = to_body.length();
        if dist > self.radius || dist < 1e-3 {
            return Vec3::ZERO;
        }
        // Falloff: linear from full strength at origin to 0 at radius.
        let falloff = 1.0 - (dist / self.radius);
        // Time decay: starts at full strength, fades to 0 over `duration`.
        let time_decay = 1.0 - (elapsed / self.duration);
        let magnitude = self.strength * falloff * time_decay;
        to_body / dist * magnitude
    }
}

/// Pulls bodies toward a fixed point in space (like a gravity well).
pub struct PointGravityForce {
    pub origin: Vec3,
    pub strength: f32,
}

impl PointGravityForce {
    pub fn new(origin: Vec3, strength: f32) -> Self {
        Self { origin, strength }
    }
}

impl ForceGenerator for PointGravityForce {
    fn force_on(&self, _body: &RigidBody, position: Vec3, _time: f32) -> Vec3 {
        let to_center = self.origin - position;
        let dist_sq = to_center.length_squared();
        if dist_sq < 1e-3 {
            return Vec3::ZERO;
        }
        let dist = dist_sq.sqrt();
        // Inverse-square falloff (clamped to avoid infinite force at distance 0).
        to_center / dist * (self.strength / dist_sq.max(0.1))
    }
}

/// Pairs a force generator with the entities it affects.
pub struct ForceRegistry {
    entries: Vec<(Entity, Box<dyn ForceGenerator>)>,
}

impl ForceRegistry {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register a force generator onto an entity.
    pub fn add(&mut self, entity: Entity, gen: Box<dyn ForceGenerator>) {
        self.entries.push((entity, gen));
    }

    /// Apply all registered forces to the bodies. Uses the world's current
    /// transform positions and rigid bodies.
    pub fn update(&mut self, world: &mut World, dt: f32, time_secs: f32) {
        // Snapshot (entity, position, body, generator-index) using both columns
        // at once via `columns2` (which uses TypeId-keyed disjoint borrows).
        let mut snapshots: Vec<(u32, Vec3, RigidBody, usize)> =
            Vec::with_capacity(self.entries.len());
        world.columns2::<engine_core::Transform, RigidBody, _, _>(|transforms, bodies| {
            for (i, (entity, _gen)) in self.entries.iter().enumerate() {
                let pos = transforms
                    .get(entity.id)
                    .map(|t| t.position)
                    .unwrap_or(Vec3::ZERO);
                let body = bodies.get(entity.id).copied().unwrap_or_default();
                snapshots.push((entity.id, pos, body, i));
            }
        });
        // Compute forces and snapshot the resulting velocity deltas.
        let mut velocity_deltas: Vec<(u32, Vec3)> = Vec::new();
        for (id, pos, body, gen_idx) in &snapshots {
            if body.mode != IntegrationMode::Dynamic {
                continue;
            }
            let gen = &self.entries[*gen_idx].1;
            let f = gen.force_on(body, *pos, time_secs);
            if f.length_squared() > 1e-6 {
                let inv_m = 1.0 / body.mass.max(0.0001);
                velocity_deltas.push((*id, f * inv_m * dt));
            }
        }
        // Apply velocity deltas.
        let mut bodies = world.column_write::<RigidBody>();
        for (id, dv) in velocity_deltas {
            if let Some(body) = bodies.get_mut(id) {
                if body.mode == IntegrationMode::Dynamic {
                    body.velocity += dv;
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for ForceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wind_force_is_constant() {
        let w = WindForce::new(Vec3::new(1.0, 0.0, 0.0), 10.0);
        let body = RigidBody::default();
        let f = w.force_on(&body, Vec3::ZERO, 0.0);
        assert!((f - Vec3::new(10.0, 0.0, 0.0)).length() < 0.01);
    }

    #[test]
    fn explosion_decays_with_distance() {
        let e = ExplosionForce::new(Vec3::ZERO, 100.0, 5.0, 0.0);
        let body = RigidBody::default();
        let near = e.force_on(&body, Vec3::new(1.0, 0.0, 0.0), 0.0);
        let far = e.force_on(&body, Vec3::new(4.0, 0.0, 0.0), 0.0);
        assert!(near.length() > far.length());
    }

    #[test]
    fn explosion_fades_after_duration() {
        let e = ExplosionForce::new(Vec3::ZERO, 100.0, 5.0, 0.0);
        let body = RigidBody::default();
        let after = e.force_on(&body, Vec3::new(1.0, 0.0, 0.0), 1.0);
        assert_eq!(after, Vec3::ZERO);
    }

    #[test]
    fn point_gravity_pulls_toward_origin() {
        let g = PointGravityForce::new(Vec3::ZERO, 10.0);
        let body = RigidBody::default();
        let f = g.force_on(&body, Vec3::new(5.0, 0.0, 0.0), 0.0);
        // Force should point toward origin (negative X direction).
        assert!(f.x < 0.0);
    }
}
