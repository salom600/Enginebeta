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

    /// Frames-per-second computed from the current frame delta. Will be `INF`
    /// on the very first frame (delta = 0); callers should clamp.
    pub fn fps(&self) -> f32 {
        let s = self.delta.as_secs_f32();
        if s > 0.0 {
            1.0 / s
        } else {
            0.0
        }
    }
}

/// Rolling-average FPS counter. Call `update(delta)` every frame and read
/// `fps()` for a smoothed value that doesn't jitter.
#[derive(Clone, Debug)]
pub struct FpsCounter {
    samples: std::collections::VecDeque<f32>,
    capacity: usize,
    sum: f32,
}

impl FpsCounter {
    pub fn new(window: usize) -> Self {
        Self {
            samples: std::collections::VecDeque::with_capacity(window),
            capacity: window,
            sum: 0.0,
        }
    }

    pub fn update(&mut self, delta: Duration) {
        let s = delta.as_secs_f32();
        if s <= 0.0 {
            return;
        }
        let fps = 1.0 / s;
        if self.samples.len() == self.capacity {
            if let Some(old) = self.samples.pop_front() {
                self.sum -= old;
            }
        }
        self.samples.push_back(fps);
        self.sum += fps;
    }

    pub fn fps(&self) -> f32 {
        if self.samples.is_empty() {
            0.0
        } else {
            self.sum / self.samples.len() as f32
        }
    }

    pub fn frame_time_ms(&self) -> f32 {
        let f = self.fps();
        if f > 0.0 {
            1000.0 / f
        } else {
            0.0
        }
    }
}

impl Default for FpsCounter {
    fn default() -> Self {
        Self::new(60)
    }
}
