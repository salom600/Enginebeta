//! Steering behaviors: seek, flee, wander.

use glam::Vec3;

/// Steering force toward `target`. The resulting velocity change is computed by
/// the caller (this just returns the direction * max_speed).
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
/// `wander_strength` in `[0, 1]` controls how jittery the wander is.
pub fn wander(
    forward: Vec3,
    current_offset: Vec3,
    wander_strength: f32,
    max_speed: f32,
) -> Vec3 {
    // Tiny perturbation on the current offset (deterministic-ish; for real
    // randomness, pass in a RNG and sample here).
    let theta = (forward.x + forward.z + current_offset.x).sin() * std::f32::consts::TAU;
    let perturb = Vec3::new(theta.cos(), 0.0, theta.sin()) * wander_strength;
    let new_offset = (current_offset + perturb).normalize_or_zero();
    (forward + new_offset).normalize_or_zero() * max_speed
}
