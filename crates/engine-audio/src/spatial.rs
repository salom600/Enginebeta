//! Software 3D positional mixing: distance attenuation + stereo pan.

use crate::AudioListener;
use glam::Vec3;

/// Inverse-square-ish distance attenuation: 1.0 within `inner_radius`,
/// linearly fading to 0.0 at `outer_radius`.
pub fn distance_attenuation(distance: f32, inner: f32, outer: f32) -> f32 {
    if distance <= inner {
        1.0
    } else if distance >= outer {
        0.0
    } else {
        1.0 - (distance - inner) / (outer - inner)
    }
}

/// Stereo pan in `[0, 1]` (0 = full left, 0.5 = center, 1 = full right) based
/// on the emitter's position relative to the listener's facing direction.
pub fn stereo_pan(emitter: Vec3, listener: &AudioListener) -> f32 {
    let to_emitter = emitter - listener.position;
    let right = listener.up.cross(listener.forward).normalize_or_zero();
    let lateral = to_emitter.dot(right);
    // Map [-1, 1] to [0, 1] with a slight compression to avoid hard panning.
    let raw = (lateral + 1.0) * 0.5;
    raw.clamp(0.1, 0.9)
}

/// Combined (volume, pan) for an emitter relative to a listener.
pub fn compute_positional_mix(emitter: Vec3, listener: &AudioListener) -> (f32, f32) {
    let distance = (emitter - listener.position).length();
    let vol = distance_attenuation(distance, 1.0, 25.0);
    let pan = stereo_pan(emitter, listener);
    (vol, pan)
}
