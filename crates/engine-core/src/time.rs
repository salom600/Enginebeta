//! Frame timing and fixed-step accumulator.

use std::time::Duration;

/// Timing info passed to every system each frame.
#[derive(Copy, Clone, Debug, Default)]
pub struct Time {
    /// Variable delta time of the current frame (clamped to `max_frame_delta`).
    pub delta: Duration,
    /// Total elapsed time since the app started.
    pub elapsed: Duration,
    /// Fixed timestep used by the simulation (e.g. physics).
    pub fixed: Duration,
    /// Total time accumulated by fixed steps.
    pub fixed_elapsed: Duration,
    /// Monotonically increasing frame counter (1-based).
    pub frame: u64,
}

impl Time {
    pub fn delta_secs(&self) -> f32 {
        self.delta.as_secs_f32()
    }

    pub fn delta_secs_f64(&self) -> f64 {
        self.delta.as_secs_f64()
    }

    pub fn elapsed_secs(&self) -> f32 {
        self.elapsed.as_secs_f32()
    }

    pub fn fixed_secs(&self) -> f32 {
        self.fixed.as_secs_f32()
    }

    /// Smooth interpolation factor in `[0, 1]` for the current frame — the
    /// fraction of the next fixed step that has accumulated. Use this to
    /// interpolate render state between two fixed simulation steps.
    pub fn alpha(&self, accumulator: Duration) -> f32 {
        if self.fixed.is_zero() {
            return 0.0;
        }
        (accumulator.as_secs_f32() / self.fixed.as_secs_f32()).clamp(0.0, 1.0)
    }
}
