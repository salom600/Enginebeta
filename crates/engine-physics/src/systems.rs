//! Physics systems: gravity, integration, sphere–sphere resolution.

use crate::{ColliderSphere, GRAVITY, IntegrationMode, RigidBody};
use engine_core::{Time, Transform, World};
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
    // Snapshot (id, velocity, mode) — single column borrow, released when the
    // closure returns.
    let updates: Vec<(u32, Vec3, IntegrationMode)> = {
        let bodies = world.column_write::<RigidBody>();
        bodies
            .iter()
            .map(|(id, b)| (id, b.velocity, b.mode))
            .collect()
    };
    // Apply movement and damping using both columns at once.
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
/// O(n²) — fine for the MVP's entity counts. Swap for a broadphase if you
/// outgrow it.
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
        world.columns3::<ColliderSphere, Transform, RigidBody, _, _>(|spheres, transforms, bodies| {
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
        });
        out
    };

    // Pairwise check
    let mut corrections: Vec<(u32, Vec3, Vec3)> = Vec::new(); // (id, positional, delta_v)
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

    // Apply corrections
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
    // Snapshot sphere radii.
    let radii: Vec<(u32, f32)> = world
        .column_write::<ColliderSphere>()
        .iter()
        .map(|(id, s)| (id, s.radius))
        .collect();

    world.columns2::<Transform, RigidBody, _, _>(|transforms, bodies| {
        for (id, radius) in radii {
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
}

/// Convenience: run the full physics step (gravity → integrate → sphere-sphere
/// → floor clamp). Wire this into [`engine_core::Stage::FixedUpdate`].
pub fn step_world(floor_y: f32) -> impl FnMut(&mut World, &Time) + Send + 'static {
    move |world, time| {
        step_gravity(world, time);
        integrate(world, time);
        resolve_sphere_sphere(world, time);
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
        // After one step, y velocity should be -9.81 * 0.016 ≈ -0.157.
        assert!(body.velocity.y < 0.0);
        assert!((body.velocity.y - (-9.81 * 0.016)).abs() < 0.01);
    }

    #[test]
    fn floor_clamp_stops_falling_body() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Transform::from_position(Vec3::new(0.0, -5.0, 0.0)));
        // Small downward velocity so the bounce is below the 0.5 threshold
        // and on_ground gets set.
        w.insert(e, RigidBody::dynamic().with_velocity(Vec3::new(0.0, -0.3, 0.0)));
        w.insert(e, ColliderSphere::new(1.0));

        floor_clamp(&mut w, 0.0);
        let t = w.get::<Transform>(e).unwrap();
        // Floor is at y=0, radius is 1, so body should be clamped to y=1.
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
        // After resolution, the distance between a and b should be >= 2.0 (1+1).
        let ta = w.get::<Transform>(a).unwrap();
        let tb = w.get::<Transform>(b).unwrap();
        let dist = (tb.position - ta.position).length();
        assert!(dist >= 2.0 - 0.01, "expected separation, got dist={dist}");
    }
}
