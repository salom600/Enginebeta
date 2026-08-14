//! Geometric primitives shared across the engine: rays, AABBs, planes.
//! Used by the renderer (raycasting / mouse picking), physics (broadphase),
//! and AI (line-of-sight checks).

use glam::{Vec3, Vec4};

/// A ray with an origin and a normalized direction.
#[derive(Copy, Clone, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, dir: Vec3) -> Self {
        Self {
            origin,
            dir: dir.normalize_or_zero(),
        }
    }

    /// Point on the ray at parameter `t` (origin + t * dir).
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

/// An axis-aligned bounding box defined by its min/max corners.
#[derive(Copy, Clone, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Build an AABB from a center point and half-extents.
    pub fn from_center_half(center: Vec3, half: Vec3) -> Self {
        Self {
            min: center - half,
            max: center + half,
        }
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// Expand the AABB to include point `p`.
    pub fn enclose_point(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    /// Test whether `other` overlaps this AABB on all three axes.
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Test whether point `p` lies inside this AABB.
    pub fn contains_point(&self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    /// Ray-vs-AABB intersection using the slab method. Returns the entry
    /// distance `t` if the ray hits the box (and `t >= 0`), else `None`.
    pub fn ray_cast(&self, ray: &Ray) -> Option<f32> {
        let inv_d = Vec3::new(
            if ray.dir.x.abs() > 1e-9 { 1.0 / ray.dir.x } else { f32::INFINITY },
            if ray.dir.y.abs() > 1e-9 { 1.0 / ray.dir.y } else { f32::INFINITY },
            if ray.dir.z.abs() > 1e-9 { 1.0 / ray.dir.z } else { f32::INFINITY },
        );
        let t1 = (self.min - ray.origin) * inv_d;
        let t2 = (self.max - ray.origin) * inv_d;
        let tmin = t1.min(t2);
        let tmax = t1.max(t2);
        let t_enter = tmin.x.max(tmin.y).max(tmin.z);
        let t_exit = tmax.x.min(tmax.y).min(tmax.z);
        if t_enter <= t_exit && t_exit >= 0.0 {
            Some(t_enter.max(0.0))
        } else {
            None
        }
    }
}

impl Default for Aabb {
    fn default() -> Self {
        Self::new(Vec3::ZERO, Vec3::ZERO)
    }
}

/// A plane defined by its normal and distance from origin along the normal.
#[derive(Copy, Clone, Debug)]
pub struct Plane {
    pub normal: Vec3,
    pub d: f32,
}

impl Plane {
    pub fn from_normal_point(normal: Vec3, point: Vec3) -> Self {
        let n = normal.normalize_or_zero();
        Self {
            normal: n,
            d: -n.dot(point),
        }
    }

    /// Signed distance from `p` to the plane. Positive = above (in front of) the plane.
    pub fn distance(&self, p: Vec3) -> f32 {
        self.normal.dot(p) + self.d
    }

    /// Ray-vs-plane intersection. Returns `t` along the ray, or `None`.
    pub fn ray_cast(&self, ray: &Ray) -> Option<f32> {
        let denom = self.normal.dot(ray.dir);
        if denom.abs() < 1e-6 {
            return None;
        }
        let t = -(self.normal.dot(ray.origin) + self.d) / denom;
        if t >= 0.0 {
            Some(t)
        } else {
            None
        }
    }
}

/// Linear interpolation helper for Vec4 (used by material color blending).
pub fn lerp_v4(a: Vec4, b: Vec4, t: f32) -> Vec4 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_intersects() {
        let a = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let b = Aabb::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(2.0, 2.0, 2.0));
        assert!(a.intersects(&b));
        let c = Aabb::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(3.0, 3.0, 3.0));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn ray_hits_aabb() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(aabb.ray_cast(&ray), Some(4.0));
    }

    #[test]
    fn ray_misses_aabb() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Vec3::new(5.0, 5.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(aabb.ray_cast(&ray).is_none());
    }

    #[test]
    fn plane_ray_cast() {
        let plane = Plane::from_normal_point(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 0.0));
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        assert_eq!(plane.ray_cast(&ray), Some(5.0));
    }
}
