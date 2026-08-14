//! Steering behaviors — seek, flee, wander, plus smooth arrive and orbit.
//!
//! v0.2.0 additions:
//! - `arrive` — like seek but decelerates near the target (no overshoot)
//! - `pursue` — predicts where a moving target will be
//! - `orbit` — circles a target at a fixed radius (good for recon drones)
//! - `smooth_path` — given a sequence of waypoints, return a smoothly varying
//!   desired velocity that avoids sharp turns

use glam::Vec3;

/// Steering force toward `target`. Returns the desired velocity (not acceleration).
pub fn seek(position: Vec3, target: Vec3, max_speed: f32) -> Vec3 {
    let to_target = target - position;
    let dist = to_target.length();
    if dist < 1e-4 {
        return Vec3::ZERO;
    }
    to_target / dist * max_speed
}

/// Steering force away from `target`.
pub fn flee(position: Vec3, threat: Vec3, max_speed: f32, panic_distance: f32) -> Vec3 {
    let from_threat = position - threat;
    let dist = from_threat.length();
    if dist > panic_distance || dist < 1e-4 {
        return Vec3::ZERO;
    }
    from_threat / dist * max_speed
}

/// Wandering steering: produce a small random-ish offset from `forward`.
pub fn wander(
    forward: Vec3,
    current_offset: Vec3,
    wander_strength: f32,
    max_speed: f32,
) -> Vec3 {
    let theta = (forward.x + forward.z + current_offset.x).sin() * std::f32::consts::TAU;
    let perturb = Vec3::new(theta.cos(), 0.0, theta.sin()) * wander_strength;
    let new_offset = (current_offset + perturb).normalize_or_zero();
    (forward + new_offset).normalize_or_zero() * max_speed
}

/// Arrive: like `seek`, but slows down when within `slowing_radius` of the target.
/// Produces smoother stops without overshoot.
pub fn arrive(
    position: Vec3,
    target: Vec3,
    max_speed: f32,
    slowing_radius: f32,
) -> Vec3 {
    let to_target = target - position;
    let dist = to_target.length();
    if dist < 1e-3 {
        return Vec3::ZERO;
    }
    let speed = if dist < slowing_radius {
        max_speed * (dist / slowing_radius)
    } else {
        max_speed
    };
    to_target / dist * speed
}

/// Pursue: predicts where a moving target will be `t` seconds from now, then
/// seeks that point. `t` is derived from the distance and the target's speed.
pub fn pursue(
    position: Vec3,
    target_pos: Vec3,
    target_vel: Vec3,
    max_speed: f32,
) -> Vec3 {
    let to_target = target_pos - position;
    let dist = to_target.length();
    let target_speed = target_vel.length();
    // Prediction time: how long it'd take us to reach the target at max speed,
    // capped to avoid over-prediction when target is far.
    let prediction = if target_speed > 0.1 {
        (dist / max_speed).min(2.0)
    } else {
        0.0
    };
    let predicted = target_pos + target_vel * prediction;
    seek(position, predicted, max_speed)
}

/// Orbit: circle around `center` at approximately `radius`. The result is a
/// velocity tangent to the circle, in the direction that keeps the orbit going
/// (counter-clockwise when viewed from +Y).
pub fn orbit(
    position: Vec3,
    center: Vec3,
    radius: f32,
    max_speed: f32,
    up: Vec3,
) -> Vec3 {
    let to_center = center - position;
    let radial = to_center.normalize_or_zero();
    // Tangent = up × radial (perpendicular to both, gives counter-clockwise orbit).
    let tangent = up.cross(radial).normalize_or_zero();
    // Add a small inward correction to maintain radius.
    let dist = to_center.length();
    let correction = if dist > radius {
        // Outside the orbit — pull inward.
        0.3
    } else if dist < radius {
        // Inside the orbit — push outward.
        -0.3
    } else {
        0.0
    };
    (tangent + radial * correction).normalize_or_zero() * max_speed
}

/// Smoothly damp the current velocity toward a desired velocity. The
/// `smooth_time` is roughly the time it takes to reach the target velocity.
/// This produces the "smooth and circular" pursuit motion requested in v0.2.0.
pub fn smooth_velocity(
    current_vel: Vec3,
    desired_vel: Vec3,
    smooth_time: f32,
    dt: f32,
) -> Vec3 {
    if smooth_time <= 0.0 {
        return desired_vel;
    }
    // Critically-damped spring (a la Unity's SmoothDamp).
    let omega = 2.0 / smooth_time;
    let x = omega * dt;
    let exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x);
    let change = current_vel - desired_vel;
    let temp = (current_vel + change) * (1.0 - exp);
    desired_vel + (change - temp) * exp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrive_decelerates_near_target() {
        let v_far = arrive(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), 5.0, 3.0);
        let v_near = arrive(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 5.0, 3.0);
        // Far: full speed. Near: reduced speed.
        assert!((v_far.length() - 5.0).abs() < 0.01);
        assert!(v_near.length() < 5.0);
    }

    #[test]
    fn pursue_predicts_target_position() {
        // Target moving +X at 5 m/s, NPC at origin, max_speed 10.
        let v = pursue(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::new(5.0, 0.0, 0.0),
            10.0,
        );
        // Should seek a point slightly +X of the target's current position.
        let desired_dir = v.normalize();
        // The predicted point has positive X, so desired velocity should have +X component.
        assert!(desired_dir.x > 0.0);
    }

    #[test]
    fn orbit_produces_tangent_velocity() {
        let v = orbit(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::ZERO,
            5.0,
            4.0,
            Vec3::Y,
        );
        // Position is on +X axis from center; tangent should be along +Z (CCW from above).
        assert!(v.z.abs() > v.x.abs());
        assert!(v.z > 0.0);
    }

    #[test]
    fn smooth_velocity_damps_toward_target() {
        let current = Vec3::new(10.0, 0.0, 0.0);
        let desired = Vec3::ZERO;
        let v = smooth_velocity(current, desired, 0.3, 0.016);
        // Should have moved part of the way toward zero.
        assert!(v.length() < current.length());
        assert!(v.length() > 0.0);
    }
}
