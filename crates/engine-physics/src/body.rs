//! Rigid body component — linear velocity, mass, gravity scale.

use glam::Vec3;

/// How the body should be integrated by the physics step.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IntegrationMode {
    /// Dynamic body — affected by gravity and collisions.
    Dynamic,
    /// Static body — does not move, but other bodies collide with it.
    Static,
    /// Kinematic body — moved by gameplay code, not by physics; pushes dynamic bodies.
    Kinematic,
}

impl Default for IntegrationMode {
    fn default() -> Self {
        Self::Dynamic
    }
}

/// A rigid body. Pair with a [`crate::ColliderSphere`] or [`crate::ColliderAabb`]
/// for collisions. Pair with an [`engine_core::Transform`] for world position.
#[derive(Copy, Clone, Debug)]
pub struct RigidBody {
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub mass: f32,
    pub gravity_scale: f32,
    pub linear_damping: f32,
    pub mode: IntegrationMode,
    /// True if the body was resting on something last step (used for ground checks).
    pub on_ground: bool,
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            gravity_scale: 1.0,
            linear_damping: 0.01,
            mode: IntegrationMode::Dynamic,
            on_ground: false,
        }
    }
}

impl RigidBody {
    pub fn dynamic() -> Self {
        Self::default()
    }
    pub fn static_body() -> Self {
        Self {
            mode: IntegrationMode::Static,
            ..Default::default()
        }
    }
    pub fn kinematic() -> Self {
        Self {
            mode: IntegrationMode::Kinematic,
            ..Default::default()
        }
    }

    pub fn with_velocity(mut self, v: Vec3) -> Self {
        self.velocity = v;
        self
    }
    pub fn with_mass(mut self, m: f32) -> Self {
        self.mass = m.max(0.0001);
        self
    }
    pub fn with_gravity_scale(mut self, s: f32) -> Self {
        self.gravity_scale = s;
        self
    }

    /// Apply an instantaneous impulse `j` (in kg·m/s) to the body.
    pub fn apply_impulse(&mut self, j: Vec3) {
        if self.mode != IntegrationMode::Dynamic {
            return;
        }
        let inv_m = 1.0 / self.mass.max(0.0001);
        self.velocity += j * inv_m;
    }
}
