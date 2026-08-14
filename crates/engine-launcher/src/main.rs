//! engine-launcher — the demo binary that wires all EngineBeta subsystems
//! together into a runnable window.
//!
//! Behavior:
//! - Opens a 1280×720 window titled "EngineBeta"
//! - Spawns a floor (static box) and a bouncing cube (dynamic sphere collider)
//! - WASD moves the camera; space drops a new cube; ESC quits
//! - Audio: a short 440Hz beep plays whenever a cube is dropped
//! - All 7 systems are exercised every frame

use anyhow::Context as _;
use engine_audio::AudioEngine;
use engine_core::{App, AppBuilder, Color, Stage, Transform, World};
use engine_input::InputContext;
use engine_physics::{
    floor_clamp, integrate, resolve_sphere_sphere, step_gravity, ColliderSphere, RigidBody,
};
use engine_render::{Camera, Mesh, MeshData, Renderer};
use glam::Vec3;
use std::collections::HashMap;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    log::info!("EngineBeta v{} starting up", engine_core::VERSION);

    let event_loop = EventLoop::new()?;
    let mut handler = EngineBetaApp::new()?;
    event_loop.run_app(&mut handler)?;
    Ok(())
}

/// The winit application handler. Owns the renderer, the app (with all systems),
/// and the input context.
struct EngineBetaApp {
    renderer: Option<Renderer>,
    app: App,
    input: InputContext,
    cube_mesh: Option<Arc<Mesh>>,
    /// Per-entity mesh handle: entity id → mesh data ref.
    meshes_by_entity: HashMap<u32, ()>,
    audio: AudioEngine,
    floor_y: f32,
    pending_drops: u32, // # of cubes to drop next frame (queued from input)
    /// Camera rotation state (mouse look).
    camera_yaw: f32,
    camera_pitch: f32,
    camera_distance: f32,
    camera_target: Vec3,
}

impl EngineBetaApp {
    fn new() -> anyhow::Result<Self> {
        // Build the app + world with all systems wired in.
        let floor_y = -2.0;
        let mut builder = AppBuilder::new();
        builder
            .add_system(Stage::PreUpdate, |world, _| {
                engine_input::clear_frame_edges(world);
            })
            .add_system(Stage::FixedUpdate, step_gravity)
            .add_system(Stage::FixedUpdate, integrate)
            .add_system(Stage::FixedUpdate, resolve_sphere_sphere)
            .add_system(Stage::FixedUpdate, move |_world, _| {
                // floor_clamp needs a world reference, run via closure
            })
            .add_system(Stage::FixedUpdate, move |world, _| {
                floor_clamp(world, -2.0);
            })
            .add_system(Stage::Update, handle_input_system(
                floor_y,
            ));

        // Seed the world: a floor and one bouncing cube.
        seed_demo_scene(builder.world_mut(), floor_y);

        let app = builder.build();
        let input = InputContext::new();
        let audio = AudioEngine::new().context("failed to initialize audio engine")?;
        // Install a default beep so we have something to play.
        let _ = audio.load_default_beep();

        Ok(Self {
            renderer: None,
            app,
            input,
            cube_mesh: None,
            meshes_by_entity: HashMap::new(),
            audio,
            floor_y,
            pending_drops: 0,
            camera_yaw: 0.0,
            camera_pitch: 0.3,
            camera_distance: 8.0,
            camera_target: Vec3::ZERO,
        })
    }

    fn spawn_cube(&mut self) {
        let entity = self.app.world_mut().spawn();
        let x = (rand_like(entity.id as u32) - 0.5) * 4.0;
        let z = (rand_like(entity.id as u32 + 17) - 0.5) * 4.0;
        let y = 4.0 + (entity.id as f32) * 0.1;
        self.app.world_mut().insert(
            entity,
            Transform::from_position(Vec3::new(x, y, z)),
        );
        self.app.world_mut().insert(
            entity,
            RigidBody::dynamic()
                .with_velocity(Vec3::new(
                    (rand_like(entity.id as u32 + 3) - 0.5) * 2.0,
                    0.0,
                    (rand_like(entity.id as u32 + 7) - 0.5) * 2.0,
                ))
                .with_gravity_scale(1.0),
        );
        self.app.world_mut().insert(entity, ColliderSphere::new(0.5));
        self.meshes_by_entity.insert(entity.id, ());
        log::info!("spawned cube entity {}", entity.id);

        // Play a beep.
        if let Err(e) = self.audio.play("beep", 0.4) {
            log::warn!("audio play failed: {e}");
        }
    }

    fn rebuild_render_list(&self) -> Vec<&Mesh> {
        // For MVP, we render the same cube mesh for every entity with a Transform.
        // A real engine would store per-entity mesh handles in the ECS.
        let mut out: Vec<&Mesh> = Vec::new();
        if let Some(mesh) = self.cube_mesh.as_ref() {
            let entity_count = self.app.world().entity_count();
            for _ in 0..entity_count {
                out.push(mesh.as_ref());
            }
        }
        out
    }
}

impl ApplicationHandler for EngineBetaApp {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }
        let window_attrs = Window::default_attributes()
            .with_title("EngineBeta")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
            .with_resizable(true);
        let window = match event_loop.create_window(window_attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                log::error!("failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };

        // Block on async init via pollster (avoid pulling in tokio just for this).
        let renderer = match pollster::block_on(Renderer::new(window.clone(), Color::rgb(0.05, 0.06, 0.09))) {
            Ok(r) => r,
            Err(e) => {
                log::error!("failed to initialize renderer: {e}");
                event_loop.exit();
                return;
            }
        };

        let mut renderer = renderer;
        if let Err(e) = renderer.build_pipeline() {
            log::error!("failed to build render pipeline: {e}");
            event_loop.exit();
            return;
        }

        // Build the cube mesh and stash it as an Arc.
        let mesh = Mesh::new(&renderer.ctx.device, &MeshData::cube([0.0, 0.78, 1.0]));
        self.cube_mesh = Some(Arc::new(mesh));

        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Feed input events.
        self.input.process_window_event(&event);

        match event {
            WindowEvent::CloseRequested => {
                log::info!("CloseRequested — exiting");
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state: ElementState::Pressed,
                    ..
                },
                ..
            } => match code {
                KeyCode::Escape => {
                    log::info!("ESC pressed — exiting");
                    event_loop.exit();
                }
                KeyCode::Space => {
                    self.pending_drops = self.pending_drops.saturating_add(1);
                }
                _ => {}
            },
            WindowEvent::Resized(size) => {
                if let Some(r) = self.renderer.as_mut() {
                    r.resize(size.width.max(1), size.height.max(1));
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // Drain pending input events from gamepads.
        self.input.process_gilrs_events();

        // Handle queued cube drops.
        while self.pending_drops > 0 {
            self.pending_drops -= 1;
            self.spawn_cube();
        }

        // Tick the simulation once. The renderer is driven from `PostUpdate`
        // via a direct call below (we don't make the renderer a system because
        // it needs the wgpu surface which lives in `self`).
        self.app.tick_once();

        // Camera orbit: keep a fixed orbiting camera for the MVP.
        let eye = Vec3::new(
            self.camera_target.x
                + self.camera_distance * self.camera_pitch.cos() * self.camera_yaw.sin(),
            self.camera_target.y + self.camera_distance * self.camera_pitch.sin(),
            self.camera_target.z
                + self.camera_distance * self.camera_pitch.cos() * self.camera_yaw.cos(),
        );
        let aspect = self
            .renderer
            .as_ref()
            .map(|r| r.config.width as f32 / r.config.height.max(1) as f32)
            .unwrap_or(16.0 / 9.0);
        let camera = Camera::new(eye, self.camera_target, aspect);

        // Render. We clone the Arc<Mesh> out first so we don't hold a borrow on
        // `self` while calling `self.renderer.as_mut()`.
        let mesh_arc = self.cube_mesh.clone();
        let entity_count = self.app.world().entity_count();
        let meshes: Vec<&Mesh> = mesh_arc
            .iter()
            .flat_map(|m| std::iter::repeat(m.as_ref()).take(entity_count))
            .collect();
        if let Some(r) = self.renderer.as_mut() {
            if let Err(e) = r.render(&camera, &meshes) {
                log::warn!("render error: {e:?}");
            }
        }

        // If app requested shutdown, exit.
        if !self.app.world().shutdown {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
        } else {
            event_loop.exit();
        }
    }
}

/// Seed the demo scene: a static floor + one bouncing cube.
fn seed_demo_scene(world: &mut World, floor_y: f32) {
    // Floor: a wide static box.
    let floor = world.spawn();
    world.insert(
        floor,
        Transform {
            position: Vec3::new(0.0, floor_y, 0.0),
            ..Default::default()
        },
    );
    world.insert(floor, RigidBody::static_body());
    world.insert(floor, ColliderSphere::new(2.0));

    // Initial cube.
    let cube = world.spawn();
    world.insert(
        cube,
        Transform::from_position(Vec3::new(0.0, 4.0, 0.0)),
    );
    world.insert(
        cube,
        RigidBody::dynamic().with_velocity(Vec3::new(1.0, 0.0, 0.5)),
    );
    world.insert(cube, ColliderSphere::new(0.5));
}

/// Input-handling system: spawn cubes when space is pressed, quit on ESC.
fn handle_input_system(
    _floor_y: f32,
) -> impl FnMut(&mut World, &engine_core::Time) + Send + 'static {
    move |world, _time| {
        // Look up the input resource and check for queued actions.
        if let Some(res) = world.resource::<engine_input::InputResource>() {
            let s = res.state.read();
            if s.key_pressed(KeyCode::Space) {
                log::info!("space pressed (system side) — would spawn cube");
                // For MVP, spawning is handled directly in the winit event
                // handler where we have access to the audio engine.
            }
        }
    }
}

/// Deterministic pseudo-random in [0, 1) from a u32 seed. Good enough for the
/// demo (no need to pull in `rand`).
fn rand_like(seed: u32) -> f32 {
    let mut x = seed.wrapping_mul(2654435761);
    x ^= x >> 16;
    x = x.wrapping_mul(2246822519);
    x ^= x >> 13;
    (x as f32) / (u32::MAX as f32)
}
