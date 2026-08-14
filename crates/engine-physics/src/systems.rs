//! Physics systems: gravity, integration, sphere–sphere + sphere–AABB + AABB–AABB collisions.

use crate::{ColliderAabb, ColliderSphere, GRAVITY, IntegrationMode, RigidBody};
use engine_core::{Aabb, Time, Transform, World};
use glam::Vec3;

/// Apply gravity to every dynamic rigid body. Run this in the fixed step.
pub fn step_gravity(world: &mut World, time: &Time) {
    let g = GRAVITY * time.fixed_secs();
    let col = world.column_write::<RigidBody>();
    for (_id, body) in col.iter_mut() {
        if body.mode == IntegrationMode::Dynamic {
            body.velocity += g * body.gravity_scale;
            body.on_ground = false;
        }
    }
}

/// Integrate velocity into the transform position. Run after gravity, before
/// collision resolution.
pub fn integrate(world: &mut World, time: &Time) {
    let dt = time.fixed_secs();
    let updates: Vec<(u32, Vec3, IntegrationMode)> = {
        let bodies = world.column_write::<RigidBody>();
        bodies
            .iter()
            .map(|(id, b)| (id, b.velocity, b.mode))
            .collect()
    };
    world.columns2::<Transform, RigidBody, _, _>(|transforms, bodies| {
        for (id, v, mode) in &updates {
            if *mode == IntegrationMode::Static {
                continue;
            }
            if let Some(t) = transforms.get_mut(*id) {
                t.position += *v * dt;
            }
        }
        for (_id, body) in bodies.iter_mut() {
            if body.mode == IntegrationMode::Dynamic {
                let damp = (1.0 - body.linear_damping).max(0.0);
                body.velocity *= damp;
            }
        }
    });
}

/// Push overlapping sphere pairs apart and apply a simple impulse response.
pub fn resolve_sphere_sphere(world: &mut World, _time: &Time) {
    #[derive(Copy, Clone)]
    struct S {
        id: u32,
        pos: Vec3,
        radius: f32,
        vel: Vec3,
        mass: f32,
        mode: IntegrationMode,
        restitution: f32,
    }

    let snapshot: Vec<S> = {
        let mut out = Vec::new();
        world.columns3::<ColliderSphere, Transform, RigidBody, _, _>(
            |spheres, transforms, bodies| {
                for (id, s) in spheres.iter() {
                    let pos = transforms.get(id).map(|t| t.position).unwrap_or(Vec3::ZERO);
                    let body = bodies.get(id).copied().unwrap_or_default();
                    out.push(S {
                        id,
                        pos,
                        radius: s.radius,
                        vel: body.velocity,
                        mass: body.mass,
                        mode: body.mode,
                        restitution: s.restitution,
                    });
                }
            },
        );
        out
    };

    let mut corrections: Vec<(u32, Vec3, Vec3)> = Vec::new();
    for i in 0..snapshot.len() {
        for j in (i + 1)..snapshot.len() {
            let a = &snapshot[i];
            let b = &snapshot[j];
            if a.mode == IntegrationMode::Static && b.mode == IntegrationMode::Static {
                continue;
            }
            let delta = b.pos - a.pos;
            let dist_sq = delta.length_squared();
            let min_dist = a.radius + b.radius;
            if dist_sq < min_dist * min_dist && dist_sq > 1e-9 {
                let dist = dist_sq.sqrt();
                let normal = delta / dist;
                let overlap = min_dist - dist;
                let inv_a = if a.mode == IntegrationMode::Static { 0.0 } else { 1.0 / a.mass };
                let inv_b = if b.mode == IntegrationMode::Static { 0.0 } else { 1.0 / b.mass };
                let inv_sum = inv_a + inv_b;
                if inv_sum <= 0.0 {
                    continue;
                }
                let correction = normal * overlap;
                corrections.push((a.id, -correction * (inv_a / inv_sum), Vec3::ZERO));
                corrections.push((b.id, correction * (inv_b / inv_sum), Vec3::ZERO));
                let rel_v = b.vel - a.vel;
                let vel_along_normal = rel_v.dot(normal);
                if vel_along_normal < 0.0 {
                    let e = a.restitution.min(b.restitution);
                    let j_impulse = -(1.0 + e) * vel_along_normal / inv_sum;
                    let impulse = normal * j_impulse;
                    corrections.push((a.id, Vec3::ZERO, -impulse * inv_a));
                    corrections.push((b.id, Vec3::ZERO, impulse * inv_b));
                }
            }
        }
    }
    world.columns2::<Transform, RigidBody, _, _>(|transforms, bodies| {
        for (id, pos_corr, vel_corr) in corrections {
            if pos_corr != Vec3::ZERO {
                if let Some(t) = transforms.get_mut(id) {
                    t.position += pos_corr;
                }
            }
            if vel_corr != Vec3::ZERO {
                if let Some(b) = bodies.get_mut(id) {
                    if b.mode == IntegrationMode::Dynamic {
                        b.velocity += vel_corr;
                    }
                }
            }
        }
    });
}

/// Resolve AABB-vs-AABB collisions. Treats every entity with a [`ColliderAabb`]
/// as a static or dynamic box; pairs that overlap are pushed apart by minimum
/// translation vector (MTV) along the smallest overlap axis.
pub fn resolve_aabb_aabb(world: &mut World, _time: &Time) {
    #[derive(Copy, Clone)]
    struct B {
        id: u32,
        pos: Vec3,
        half: Vec3,
        vel: Vec3,
        mass: f32,
        mode: IntegrationMode,
        restitution: f32,
    }

    let snapshot: Vec<B> = {
        let mut out = Vec::new();
        world.columns3::<ColliderAabb, Transform, RigidBody, _, _>(
            |boxes, transforms, bodies| {
                for (id, b) in boxes.iter() {
                    let pos = transforms.get(id).map(|t| t.position).unwrap_or(Vec3::ZERO);
                    let body = bodies.get(id).copied().unwrap_or_default();
                    out.push(B {
                        id,
                        pos,
                        half: b.half,
                        vel: body.velocity,
                        mass: body.mass,
                        mode: body.mode,
                        restitution: b.restitution,
                    });
                }
            },
        );
        out
    };

    let mut corrections: Vec<(u32, Vec3, Vec3)> = Vec::new();
    for i in 0..snapshot.len() {
        for j in (i + 1)..snapshot.len() {
            let a = &snapshot[i];
            let b = &snapshot[j];
            if a.mode == IntegrationMode::Static && b.mode == IntegrationMode::Static {
                continue;
            }
            // Compute overlap on each axis.
            let a_min = a.pos - a.half;
            let a_max = a.pos + a.half;
            let b_min = b.pos - b.half;
            let b_max = b.pos + b.half;
            let overlap_x = (a_max.x.min(b_max.x) - a_min.x.max(b_min.x)).max(0.0);
            let overlap_y = (a_max.y.min(b_max.y) - a_min.y.max(b_min.y)).max(0.0);
            let overlap_z = (a_max.z.min(b_max.z) - a_min.z.max(b_min.z)).max(0.0);
            if overlap_x <= 0.0 || overlap_y <= 0.0 || overlap_z <= 0.0 {
                continue; // No overlap.
            }
            // Smallest axis is the MTV (minimum translation vector).
            let (mtv_axis, mtv_dist) = if overlap_x <= overlap_y && overlap_x <= overlap_z {
                (Vec3::X, overlap_x)
            } else if overlap_y <= overlap_x && overlap_y <= overlap_z {
                (Vec3::Y, overlap_y)
            } else {
                (Vec3::Z, overlap_z)
            };
            // Direction from a to b.
            let dir = b.pos - a.pos;
            let sign = if dir.dot(mtv_axis) >= 0.0 { 1.0 } else { -1.0 };
            let normal = mtv_axis * sign;
            let inv_a = if a.mode == IntegrationMode::Static { 0.0 } else { 1.0 / a.mass };
            let inv_b = if b.mode == IntegrationMode::Static { 0.0 } else { 1.0 / b.mass };
            let inv_sum = inv_a + inv_b;
            if inv_sum <= 0.0 {
                continue;
            }
            let correction = normal * mtv_dist;
            corrections.push((a.id, -correction * (inv_a / inv_sum), Vec3::ZERO));
            corrections.push((b.id, correction * (inv_b / inv_sum), Vec3::ZERO));
            // Velocity response along the collision normal.
            let rel_v = b.vel - a.vel;
            let vel_along_normal = rel_v.dot(normal);
            if vel_along_normal < 0.0 {
                let e = a.restitution.min(b.restitution);
                let j_impulse = -(1.0 + e) * vel_along_normal / inv_sum;
                let impulse = normal * j_impulse;
                corrections.push((a.id, Vec3::ZERO, -impulse * inv_a));
                corrections.push((b.id, Vec3::ZERO, impulse * inv_b));
            }
        }
    }
    world.columns2::<Transform, RigidBody, _, _>(|transforms, bodies| {
        for (id, pos_corr, vel_corr) in corrections {
            if pos_corr != Vec3::ZERO {
                if let Some(t) = transforms.get_mut(id) {
                    t.position += pos_corr;
                }
            }
            if vel_corr != Vec3::ZERO {
                if let Some(b) = bodies.get_mut(id) {
                    if b.mode == IntegrationMode::Dynamic {
                        b.velocity += vel_corr;
                    }
                }
            }
        }
    });
}

/// Resolve sphere-vs-AABB collisions. Pushes the sphere out of the box and
/// applies an impulse response.
pub fn resolve_sphere_aabb(world: &mut World, _time: &Time) {
    #[derive(Copy, Clone)]
    struct Sphere {
        id: u32,
        pos: Vec3,
        radius: f32,
        vel: Vec3,
        mass: f32,
        mode: IntegrationMode,
        restitution: f32,
    }
    #[derive(Copy, Clone)]
    struct Box {
        id: u32,
        pos: Vec3,
        half: Vec3,
        vel: Vec3,
        mass: f32,
        mode: IntegrationMode,
        restitution: f32,
    }

    let spheres: Vec<Sphere> = {
        let mut out = Vec::new();
        world.columns3::<ColliderSphere, Transform, RigidBody, _, _>(
            |spheres, transforms, bodies| {
                for (id, s) in spheres.iter() {
                    let pos = transforms.get(id).map(|t| t.position).unwrap_or(Vec3::ZERO);
                    let body = bodies.get(id).copied().unwrap_or_default();
                    out.push(Sphere {
                        id,
                        pos,
                        radius: s.radius,
                        vel: body.velocity,
                        mass: body.mass,
                        mode: body.mode,
                        restitution: s.restitution,
                    });
                }
            },
        );
        out
    };
    let boxes: Vec<Box> = {
        let mut out = Vec::new();
        world.columns3::<ColliderAabb, Transform, RigidBody, _, _>(
            |boxes, transforms, bodies| {
                for (id, b) in boxes.iter() {
                    let pos = transforms.get(id).map(|t| t.position).unwrap_or(Vec3::ZERO);
                    let body = bodies.get(id).copied().unwrap_or_default();
                    out.push(Box {
                        id,
                        pos,
                        half: b.half,
                        vel: body.velocity,
                        mass: body.mass,
                        mode: body.mode,
                        restitution: b.restitution,
                    });
                }
            },
        );
        out
    };

    let mut corrections: Vec<(u32, Vec3, Vec3)> = Vec::new();
    for s in &spheres {
        for b in &boxes {
            if s.id == b.id {
                continue;
            }
            if s.mode == IntegrationMode::Static && b.mode == IntegrationMode::Static {
                continue;
            }
            // Closest point on the AABB to the sphere center.
            let closest = Vec3::new(
                s.pos.x.clamp(b.pos.x - b.half.x, b.pos.x + b.half.x),
                s.pos.y.clamp(b.pos.y - b.half.y, b.pos.y + b.half.y),
                s.pos.z.clamp(b.pos.z - b.half.z, b.pos.z + b.half.z),
            );
            let delta = s.pos - closest;
            let dist_sq = delta.length_squared();
            if dist_sq >= s.radius * s.radius {
                continue;
            }
            let dist = dist_sq.sqrt().max(1e-6);
            let normal = delta / dist;
            let overlap = s.radius - dist;
            let inv_s = if s.mode == IntegrationMode::Static { 0.0 } else { 1.0 / s.mass };
            let inv_b = if b.mode == IntegrationMode::Static { 0.0 } else { 1.0 / b.mass };
            let inv_sum = inv_s + inv_b;
            if inv_sum <= 0.0 {
                continue;
            }
            let correction = normal * overlap;
            corrections.push((s.id, correction * (inv_s / inv_sum), Vec3::ZERO));
            corrections.push((b.id, -correction * (inv_b / inv_sum), Vec3::ZERO));
            let rel_v = s.vel - b.vel;
            let vel_along_normal = rel_v.dot(normal);
            if vel_along_normal < 0.0 {
                let e = s.restitution.min(b.restitution);
                let j_impulse = -(1.0 + e) * vel_along_normal / inv_sum;
                let impulse = normal * j_impulse;
                corrections.push((s.id, Vec3::ZERO, impulse * inv_s));
                corrections.push((b.id, Vec3::ZERO, -impulse * inv_b));
            }
        }
    }
    world.columns2::<Transform, RigidBody, _, _>(|transforms, bodies| {
        for (id, pos_corr, vel_corr) in corrections {
            if pos_corr != Vec3::ZERO {
                if let Some(t) = transforms.get_mut(id) {
                    t.position += pos_corr;
                }
            }
            if vel_corr != Vec3::ZERO {
                if let Some(b) = bodies.get_mut(id) {
                    if b.mode == IntegrationMode::Dynamic {
                        b.velocity += vel_corr;
                    }
                }
            }
        }
    });
}

/// Simple AABB-vs-floor test: clamp every body with a sphere collider above a
/// floor plane at y = `floor_y`. Useful for the demo scene so things don't fall
/// through the world.
pub fn floor_clamp(world: &mut World, floor_y: f32) {
    // Clamp spheres.
    let sphere_radii: Vec<(u32, f32)> = world
        .column_write::<ColliderSphere>()
        .iter()
        .map(|(id, s)| (id, s.radius))
        .collect();
    world.columns2::<Transform, RigidBody, _, _>(|transforms, bodies| {
        for (id, radius) in sphere_radii {
            if let Some(t) = transforms.get_mut(id) {
                let min_y = floor_y + radius;
                if t.position.y < min_y {
                    t.position.y = min_y;
                    if let Some(b) = bodies.get_mut(id) {
                        if b.velocity.y < 0.0 {
                            b.velocity.y = -b.velocity.y * 0.3;
                        }
                        if b.velocity.y.abs() < 0.5 {
                            b.velocity.y = 0.0;
                            b.on_ground = true;
                        }
                    }
                }
            }
        }
    });
    // Clamp AABBs.
    let aabb_halves: Vec<(u32, f32)> = world
        .column_write::<ColliderAabb>()
        .iter()
        .map(|(id, b)| (id, b.half.y))
        .collect();
    world.columns2::<Transform, RigidBody, _, _>(|transforms, bodies| {
        for (id, half_y) in aabb_halves {
            if let Some(t) = transforms.get_mut(id) {
                let min_y = floor_y + half_y;
                if t.position.y < min_y {
                    t.position.y = min_y;
                    if let Some(b) = bodies.get_mut(id) {
                        if b.velocity.y < 0.0 {
                            b.velocity.y = -b.velocity.y * 0.3;
                        }
                        if b.velocity.y.abs() < 0.5 {
                            b.velocity.y = 0.0;
                            b.on_ground = true;
                        }
                    }
                }
            }
        }
    });
}

/// Build an AABB for a sphere (for broadphase / picking).
pub fn sphere_aabb(center: Vec3, radius: f32) -> Aabb {
    Aabb::from_center_half(center, Vec3::splat(radius))
}

/// Convenience: run the full physics step (gravity → integrate → collisions
/// → floor clamp). Wire this into [`engine_core::Stage::FixedUpdate`].
pub fn step_world(floor_y: f32) -> impl FnMut(&mut World, &Time) + Send + 'static {
    move |world, time| {
        step_gravity(world, time);
        integrate(world, time);
        resolve_sphere_sphere(world, time);
        resolve_aabb_aabb(world, time);
        resolve_sphere_aabb(world, time);
        floor_clamp(world, floor_y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{Time, Transform, World};
    use std::time::Duration;

    fn make_time(dt_secs: f32) -> Time {
        let mut t = Time::default();
        t.fixed = Duration::from_secs_f32(dt_secs);
        t
    }

    #[test]
    fn gravity_accelerates_dynamic_body() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Transform::default());
        w.insert(e, RigidBody::dynamic());
        w.insert(e, ColliderSphere::new(0.5));

        let t = make_time(0.016);
        step_gravity(&mut w, &t);
        let body = w.get::<RigidBody>(e).unwrap();
        assert!(body.velocity.y < 0.0);
        assert!((body.velocity.y - (-9.81 * 0.016)).abs() < 0.01);
    }

    #[test]
    fn floor_clamp_stops_falling_body() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Transform::from_position(Vec3::new(0.0, -5.0, 0.0)));
        w.insert(e, RigidBody::dynamic().with_velocity(Vec3::new(0.0, -0.3, 0.0)));
        w.insert(e, ColliderSphere::new(1.0));

        floor_clamp(&mut w, 0.0);
        let t = w.get::<Transform>(e).unwrap();
        assert!((t.position.y - 1.0).abs() < 0.001);
        let b = w.get::<RigidBody>(e).unwrap();
        assert!(b.on_ground);
    }

    #[test]
    fn sphere_sphere_resolution_separates() {
        let mut w = World::new();
        let a = w.spawn();
        w.insert(a, Transform::from_position(Vec3::new(0.0, 0.0, 0.0)));
        w.insert(a, RigidBody::dynamic().with_velocity(Vec3::ZERO));
        w.insert(a, ColliderSphere::new(1.0));
        let b = w.spawn();
        w.insert(b, Transform::from_position(Vec3::new(1.0, 0.0, 0.0)));
        w.insert(b, RigidBody::dynamic().with_velocity(Vec3::ZERO));
        w.insert(b, ColliderSphere::new(1.0));

        let t = make_time(0.016);
        resolve_sphere_sphere(&mut w, &t);
        let ta = w.get::<Transform>(a).unwrap();
        let tb = w.get::<Transform>(b).unwrap();
        let dist = (tb.position - ta.position).length();
        assert!(dist >= 2.0 - 0.01, "expected separation, got dist={dist}");
    }

    #[test]
    fn aabb_aabb_resolution_separates() {
        let mut w = World::new();
        let a = w.spawn();
        w.insert(a, Transform::from_position(Vec3::new(0.0, 0.0, 0.0)));
        w.insert(a, RigidBody::dynamic().with_velocity(Vec3::ZERO));
        w.insert(a, ColliderAabb::cube(0.5));
        let b = w.spawn();
        w.insert(b, Transform::from_position(Vec3::new(0.8, 0.0, 0.0)));
        w.insert(b, RigidBody::dynamic().with_velocity(Vec3::ZERO));
        w.insert(b, ColliderAabb::cube(0.5));

        let t = make_time(0.016);
        resolve_aabb_aabb(&mut w, &t);
        let ta = w.get::<Transform>(a).unwrap();
        let tb = w.get::<Transform>(b).unwrap();
        let dist = (tb.position - ta.position).x.abs();
        // Each half-extent is 0.5, so they need to be ≥ 1.0 apart on X.
        assert!(dist >= 1.0 - 0.01, "expected separation, got dist={dist}");
    }

    #[test]
    fn sphere_aabb_resolution_separates() {
        let mut w = World::new();
        let s = w.spawn();
        w.insert(s, Transform::from_position(Vec3::new(0.0, 0.0, 0.0)));
        w.insert(s, RigidBody::dynamic().with_velocity(Vec3::ZERO));
        w.insert(s, ColliderSphere::new(1.0));
        let b = w.spawn();
        // Place box such that its surface is inside the sphere (overlap).
        w.insert(b, Transform::from_position(Vec3::new(0.8, 0.0, 0.0)));
        w.insert(b, RigidBody::static_body());
        w.insert(b, ColliderAabb::cube(0.5));

        let t = make_time(0.016);
        resolve_sphere_aabb(&mut w, &t);
        let ts = w.get::<Transform>(s).unwrap();
        let tb = w.get::<Transform>(b).unwrap();
        // After resolution the sphere should be pushed left so its right edge
        // touches the box's left edge (which stays at x = 0.8 - 0.5 = 0.3).
        // So sphere.x should be ≤ 0.3 - 1.0 = -0.7.
        assert!(
            ts.position.x <= -0.69,
            "expected sphere pushed to x ≤ -0.7, got x={}",
            ts.position.x
        );
        // Box is static — should not have moved.
        assert!((tb.position.x - 0.8).abs() < 0.001);
    }
}
