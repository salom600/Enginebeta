//! Collider shapes.

/// A sphere centered on the entity's transform position.
#[derive(Copy, Clone, Debug)]
pub struct ColliderSphere {
    pub radius: f32,
    pub restitution: f32, // bounciness in [0, 1]
}

impl Default for ColliderSphere {
    fn default() -> Self {
        Self {
            radius: 0.5,
            restitution: 0.2,
        }
    }
}

impl ColliderSphere {
    pub fn new(radius: f32) -> Self {
        Self {
            radius,
            restitution: 0.2,
        }
    }
}

/// An axis-aligned box collider centered on the entity's transform position.
/// Half-extents on each axis (so total width = 2 * half.x).
#[derive(Copy, Clone, Debug)]
pub struct ColliderAabb {
    pub half: glam::Vec3,
    pub restitution: f32,
}

impl Default for ColliderAabb {
    fn default() -> Self {
        Self {
            half: glam::Vec3::splat(0.5),
            restitution: 0.2,
        }
    }
}

impl ColliderAabb {
    pub fn new(half: glam::Vec3) -> Self {
        Self {
            half,
            restitution: 0.2,
        }
    }
    pub fn cube(half_extent: f32) -> Self {
        Self::new(glam::Vec3::splat(half_extent))
    }
}
