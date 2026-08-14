//! Positional audio components.

use glam::Vec3;

/// An entity that emits sound. Pair with a [`engine_core::Transform`].
#[derive(Copy, Clone, Debug, Default)]
pub struct AudioEmitter {
    /// Sound name to play (must be loaded into the [`crate::AudioEngine`] cache).
    pub sound: [u8; 32],
    /// Base volume before distance attenuation.
    pub volume: f32,
    /// Inner radius (no attenuation within this distance).
    pub inner_radius: f32,
    /// Outer radius (silence beyond this distance).
    pub outer_radius: f32,
    /// Whether to loop while the entity is alive.
    pub looping: bool,
}

impl AudioEmitter {
    pub fn new(sound_name: &str) -> Self {
        let mut buf = [0u8; 32];
        let bytes = sound_name.as_bytes();
        let n = bytes.len().min(32);
        buf[..n].copy_from_slice(&bytes[..n]);
        Self {
            sound: buf,
            volume: 1.0,
            inner_radius: 1.0,
            outer_radius: 25.0,
            looping: false,
        }
    }

    pub fn name(&self) -> &str {
        let nul = self.sound.iter().position(|b| *b == 0).unwrap_or(32);
        std::str::from_utf8(&self.sound[..nul]).unwrap_or("")
    }
}

/// An entity that listens to sounds. Usually the player camera.
#[derive(Copy, Clone, Debug, Default)]
pub struct AudioListener {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
}

impl AudioListener {
    pub fn new(position: Vec3, forward: Vec3, up: Vec3) -> Self {
        Self {
            position,
            forward: forward.normalize_or_zero(),
            up,
        }
    }
}

// Suppress unused import warning when Component isn't derived here.
#[allow(dead_code)]
fn _unused(_: AudioEmitter) {}
