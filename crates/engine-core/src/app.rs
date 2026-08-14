//! The application loop and stage-based scheduler.

use crate::ecs::World;
use crate::time::Time;
use log::info;
use std::time::{Duration, Instant};

/// A logical pipeline stage. Systems run in the order they were inserted.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Stage {
    /// Runs once at startup before the first frame.
    Startup,
    /// Runs every frame, before fixed updates.
    PreUpdate,
    /// Runs at a fixed timestep (default 60 Hz) for physics / simulation.
    FixedUpdate,
    /// Runs every frame after fixed updates.
    Update,
    /// Runs every frame, last — rendering, presentation, end-of-frame bookkeeping.
    PostUpdate,
}

type SystemFn = Box<dyn FnMut(&mut World, &Time) + Send + 'static>;

/// Builder for [`App`]. Insert systems stage-by-stage, then call [`App::run`].
pub struct AppBuilder {
    startup: Vec<SystemFn>,
    pre_update: Vec<SystemFn>,
    fixed_update: Vec<SystemFn>,
    update: Vec<SystemFn>,
    post_update: Vec<SystemFn>,
    fixed_timestep: Duration,
    max_frame: Duration,
    world: World,
}

impl AppBuilder {
    pub fn new() -> Self {
        Self {
            startup: Vec::new(),
            pre_update: Vec::new(),
            fixed_update: Vec::new(),
            update: Vec::new(),
            post_update: Vec::new(),
            fixed_timestep: Duration::from_secs_f32(1.0 / 60.0),
            max_frame: Duration::from_millis(100),
            world: World::new(),
        }
    }

    /// Insert a system into a stage.
    pub fn add_system<F>(&mut self, stage: Stage, system: F) -> &mut Self
    where
        F: FnMut(&mut World, &Time) + Send + 'static,
    {
        let bucket = match stage {
            Stage::Startup => &mut self.startup,
            Stage::PreUpdate => &mut self.pre_update,
            Stage::FixedUpdate => &mut self.fixed_update,
            Stage::Update => &mut self.update,
            Stage::PostUpdate => &mut self.post_update,
        };
        bucket.push(Box::new(system));
        self
    }

    /// Set the fixed simulation timestep (default 1/60 s).
    pub fn fixed_timestep(&mut self, dur: Duration) -> &mut Self {
        self.fixed_timestep = dur;
        self
    }

    /// Set the maximum allowed frame delta (default 100 ms). Larger deltas are
    /// clamped to avoid the "spiral of death" after a hitch.
    pub fn max_frame_delta(&mut self, dur: Duration) -> &mut Self {
        self.max_frame = dur;
        self
    }

    /// Take ownership of the inner world (e.g. to seed it before run).
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    pub fn build(self) -> App {
        App {
            startup: self.startup,
            pre_update: self.pre_update,
            fixed_update: self.fixed_update,
            update: self.update,
            post_update: self.post_update,
            fixed_timestep: self.fixed_timestep,
            max_frame: self.max_frame,
            world: self.world,
            time: Time::default(),
            accumulator: Duration::ZERO,
            last: None,
            running: true,
        }
    }
}

impl Default for AppBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The running application. Call [`App::run`] to enter the main loop; a system can
/// request shutdown by setting `app.world_mut().shutdown = true`.
pub struct App {
    startup: Vec<SystemFn>,
    pre_update: Vec<SystemFn>,
    fixed_update: Vec<SystemFn>,
    update: Vec<SystemFn>,
    post_update: Vec<SystemFn>,
    fixed_timestep: Duration,
    max_frame: Duration,
    world: World,
    time: Time,
    accumulator: Duration,
    last: Option<Instant>,
    running: bool,
}

impl App {
    pub fn builder() -> AppBuilder {
        AppBuilder::new()
    }

    pub fn world(&self) -> &World {
        &self.world
    }
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }
    pub fn time(&self) -> &Time {
        &self.time
    }

    /// Run the startup stage once. Useful to call explicitly in tests.
    pub fn run_startup(&mut self) {
        for sys in self.startup.iter_mut() {
            sys(&mut self.world, &self.time);
        }
    }

    /// Run one frame manually (startup + accumulator). Returns `false` if the app
    /// has been asked to shut down.
    ///
    /// This is the entry point used by `engine-launcher` (which owns the window /
    /// event loop) and by integration tests.
    pub fn tick_once(&mut self) -> bool {
        if !self.running {
            return false;
        }

        let now = Instant::now();
        let delta = match self.last {
            None => self.fixed_timestep,
            Some(prev) => {
                let d = now - prev;
                if d > self.max_frame {
                    self.max_frame
                } else {
                    d
                }
            }
        };
        self.last = Some(now);

        self.time.delta = delta;
        self.time.elapsed += delta;
        self.time.frame += 1;

        // Startup runs on the first tick.
        if self.time.frame == 1 {
            self.run_startup();
        }

        // Pre-update every frame.
        for sys in self.pre_update.iter_mut() {
            sys(&mut self.world, &self.time);
        }

        // Fixed update may run 0..N times per frame.
        self.accumulator += delta;
        let step = self.fixed_timestep;
        while self.accumulator >= step {
            self.time.fixed = step;
            self.time.fixed_elapsed += step;
            for sys in self.fixed_update.iter_mut() {
                sys(&mut self.world, &self.time);
            }
            self.accumulator -= step;
        }

        // Variable update.
        for sys in self.update.iter_mut() {
            sys(&mut self.world, &self.time);
        }

        // Post update (rendering happens here, driven by engine-render).
        for sys in self.post_update.iter_mut() {
            sys(&mut self.world, &self.time);
        }

        if self.world.shutdown {
            self.running = false;
            info!("App received shutdown request — exiting main loop.");
        }
        true
    }

    /// Headless run loop — no window, just simulation. Useful for tests and
    /// dedicated servers. Returns when shutdown is requested or `max_frames`
    /// ticks have been processed (whichever comes first; pass `usize::MAX` for
    /// "run until shutdown").
    pub fn run_headless(&mut self, max_frames: usize) {
        self.run_startup();
        for _ in 0..max_frames {
            if !self.tick_once() {
                break;
            }
        }
    }
}
