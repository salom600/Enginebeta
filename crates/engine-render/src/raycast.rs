//! Raycasting — convert a mouse position into a world-space ray and test it
//! against scene geometry.
//!
//! Usage:
//! 1. [`Raycast::from_screen`] builds a ray from a mouse position + camera + viewport.
//! 2. [`Raycast::test_aabb`] tests that ray against axis-aligned boxes.

use engine_core::{Aabb, Ray};
use glam::{Mat4, Vec2, Vec3};

/// A world-space ray with the camera matrices that produced it.
pub struct Raycast {
    pub ray: Ray,
}

impl Raycast {
    /// Build a ray from a screen position (in pixels, origin top-left) and the
    /// inverse view-projection matrix.
    pub fn from_screen(
        mouse: Vec2,
        viewport: Vec2,
        inv_view_proj: Mat4,
    ) -> Self {
        // Convert mouse to NDC: x in [-1, 1], y in [-1, 1] (y flipped because
        // screen Y grows downward but NDC Y grows upward).
        let ndc_x = (mouse.x / viewport.x) * 2.0 - 1.0;
        let ndc_y = 1.0 - (mouse.y / viewport.y) * 2.0;

        // Two points on the near and far planes in clip space.
        let near = inv_view_proj * Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let far = inv_view_proj * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        // Perspective divide.
        let near_p = Vec3::new(near.x, near.y, near.z) / near.w;
        let far_p = Vec3::new(far.x, far.y, far.z) / far.w;
        let dir = (far_p - near_p).normalize_or_zero();
        Self {
            ray: Ray::new(near_p, dir),
        }
    }

    /// Build a ray directly from a camera position and direction (no screen coords).
    pub fn from_camera(eye: Vec3, dir: Vec3) -> Self {
        Self {
            ray: Ray::new(eye, dir),
        }
    }

    /// Test the ray against an AABB. Returns the hit distance `t` and the
    /// hit position if there's an intersection within `max_dist`.
    pub fn test_aabb(&self, aabb: &Aabb, max_dist: f32) -> Option<RaycastHit> {
        let t = aabb.ray_cast(&self.ray)?;
        if t > max_dist {
            return None;
        }
        let point = self.ray.at(t);
        let normal = compute_aabb_normal(aabb, point);
        Some(RaycastHit { distance: t, point, normal })
    }

    /// Test against a sphere (defined by center + radius). Returns the nearest
    /// hit distance if the ray intersects the sphere.
    pub fn test_sphere(&self, center: Vec3, radius: f32, max_dist: f32) -> Option<RaycastHit> {
        let oc = self.ray.origin - center;
        let a = self.ray.dir.dot(self.ray.dir);
        let b = 2.0 * oc.dot(self.ray.dir);
        let c = oc.dot(oc) - radius * radius;
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return None;
        }
        let sq = disc.sqrt();
        let t1 = (-b - sq) / (2.0 * a);
        let t2 = (-b + sq) / (2.0 * a);
        let t = if t1 >= 0.0 {
            t1
        } else if t2 >= 0.0 {
            t2
        } else {
            return None;
        };
        if t > max_dist {
            return None;
        }
        let point = self.ray.at(t);
        let normal = (point - center).normalize_or_zero();
        Some(RaycastHit { distance: t, point, normal })
    }
}

/// Result of a successful raycast hit.
#[derive(Copy, Clone, Debug)]
pub struct RaycastHit {
    pub distance: f32,
    pub point: Vec3,
    pub normal: Vec3,
}

/// Compute the closest face normal of an AABB at a given surface point.
fn compute_aabb_normal(aabb: &Aabb, point: Vec3) -> Vec3 {
    let center = aabb.center();
    let half = aabb.half_extents();
    let local = (point - center) / half; // in [-1, 1]
    let abs = local.abs();
    if abs.x >= abs.y && abs.x >= abs.z {
        Vec3::new(local.x.signum(), 0.0, 0.0)
    } else if abs.y >= abs.x && abs.y >= abs.z {
        Vec3::new(0.0, local.y.signum(), 0.0)
    } else {
        Vec3::new(0.0, 0.0, local.z.signum())
    }
}

use glam::Vec4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_hits_sphere() {
        let r = Raycast::from_camera(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = r.test_sphere(Vec3::ZERO, 1.0, 100.0).unwrap();
        assert!((hit.distance - 4.0).abs() < 0.01);
        assert!((hit.point.z + 1.0).abs() < 0.01); // hit at z = -1
        assert!((hit.normal.z + 1.0).abs() < 0.01); // normal pointing -Z
    }

    #[test]
    fn ray_misses_sphere() {
        let r = Raycast::from_camera(Vec3::new(5.0, 5.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(r.test_sphere(Vec3::ZERO, 1.0, 100.0).is_none());
    }
}
