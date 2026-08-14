//! engine-launcher — the v0.2.0 demo binary.
//!
//! Demo features:
//! - Fly camera (WASD + mouse look, Space/Shift for vertical)
//! - Directional + ambient lighting with shadows
//! - Per-mesh materials (wood, metal, plastic, emissive)
//! - Static ground + falling cubes + falling balls + colliding boxes
//! - Wind force + explosion force (press E to trigger an explosion)
//! - AI enemy with vision + hearing perception
//! - Raycasting for mouse picking (left-click highlights nearest entity)
//! - FPS counter + memory usage logged to console

use anyhow::Context as _;
use engine_ai::{
    perception_system, Alerted, HearingSensor, Perceivable, SoundEvent,
    VisionSensor,
};
use engine_audio::AudioEngine;
use engine_core::{App, AppBuilder, Color, FpsCounter, Stage, Time, Transform, World};
use engine_input::InputContext;
use engine_physics::{
    floor_clamp, integrate, resolve_aabb_aabb, resolve_sphere_aabb, resolve_sphere_sphere,
    step_gravity, ColliderAabb, ColliderSphere, ExplosionForce, ForceGenerator, ForceRegistry,
    RigidBody,
};
use engine_render::{
    Camera, DrawCall, FlyCamera, LightUniform, Material, Mesh, MeshData, OrthoCamera, Raycast,
};
use glam::{Mat4, Vec3};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    log::info!("EngineBeta v0.2.0 starting up");

    let event_loop = EventLoop::new()?;
    let mut handler = EngineBetaApp::new()?;
    event_loop.run_app(&mut handler)?;
    Ok(())
}

/// Resource: shared state between the winit handler and the engine systems.
struct EngineState {
    fly_cam: FlyCamera,
    mouse_buttons_down: bool,
    last_mouse_pos: Option<glam::Vec2>,
    pending_explosion: bool,
    pending_drop_cube: u32,
    pending_drop_ball: u32,
    /// Cached meshes — populated on first resume.
    cube_mesh: Option<Arc<Mesh>>,
    sphere_mesh: Option<Arc<Mesh>>,
    plane_mesh: Option<Arc<Mesh>>,
    /// Materials — shared across entities by category.
    materials: MaterialSet,
    /// Profiler
    fps: FpsCounter,
    last_profiler_log: Instant,
    /// Force registry (wind + explosions).
    force_registry: ForceRegistry,
    /// Current elapsed simulation time (for force generators).
    sim_time: f32,
}

#[derive(Default, Clone, Copy)]
struct MaterialSet {
    wood: Material,
    metal: Material,
    plastic_red: Material,
    plastic_blue: Material,
    emissive: Material,
    ground: Material,
}

impl MaterialSet {
    fn new() -> Self {
        Self {
            wood: Material::wood(),
            metal: Material::metal(),
            plastic_red: Material::plastic([0.85, 0.15, 0.15]),
            plastic_blue: Material::plastic([0.15, 0.35, 0.85]),
            emissive: Material::emissive([0.0, 0.78, 1.0]),
            ground: Material::plastic([0.3, 0.35, 0.32]),
        }
    }
}

struct EngineBetaApp {
    renderer: Option<engine_render::Renderer>,
    app: App,
    input: InputContext,
    audio: AudioEngine,
    state: Arc<RwLock<EngineState>>,
    /// Per-entity (mesh_kind, material_id) so the renderer can build draw calls.
    entity_render: Arc<RwLock<Vec<(u32, MeshKind, MaterialId)>>>,
    /// Floor Y coordinate.
    floor_y: f32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MeshKind {
    Cube,
    Sphere,
    Plane,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MaterialId {
    Wood,
    Metal,
    PlasticRed,
    PlasticBlue,
    Emissive,
    Ground,
}

impl EngineBetaApp {
    fn new() -> anyhow::Result<Self> {
        let floor_y = -1.0;
        let mut builder = AppBuilder::new();
        builder
            .add_system(Stage::PreUpdate, |world, _| {
                engine_input::clear_frame_edges(world);
            })
            .add_system(Stage::FixedUpdate, step_gravity)
            .add_system(Stage::FixedUpdate, integrate)
            .add_system(Stage::FixedUpdate, resolve_sphere_sphere)
            .add_system(Stage::FixedUpdate, resolve_aabb_aabb)
            .add_system(Stage::FixedUpdate, resolve_sphere_aabb)
            .add_system(Stage::FixedUpdate, move |world, _| {
                floor_clamp(world, -1.0);
            });

        seed_demo_scene(builder.world_mut(), floor_y);

        let app = builder.build();
        let input = InputContext::new();
        let audio = AudioEngine::new().context("failed to initialize audio engine")?;
        let _ = audio.load_default_beep();

        let state = Arc::new(RwLock::new(EngineState {
            fly_cam: FlyCamera::default(),
            mouse_buttons_down: false,
            last_mouse_pos: None,
            pending_explosion: false,
            pending_drop_cube: 0,
            pending_drop_ball: 0,
            cube_mesh: None,
            sphere_mesh: None,
            plane_mesh: None,
            materials: MaterialSet::new(),
            fps: FpsCounter::new(60),
            last_profiler_log: Instant::now(),
            force_registry: ForceRegistry::new(),
            sim_time: 0.0,
        }));

        // Track entity → (mesh, material) for rendering.
        let entity_render = Arc::new(RwLock::new(Vec::new()));

        Ok(Self {
            renderer: None,
            app,
            input,
            audio,
            state,
            entity_render,
            floor_y,
        })
    }

    fn spawn_cube(&mut self, material: MaterialId) {
        let entity = self.app.world_mut().spawn();
        let x = pseudo_rand(entity.id as u32) * 6.0 - 3.0;
        let z = pseudo_rand(entity.id as u32 + 7) * 6.0 - 3.0;
        let y = 4.0 + (entity.id as f32) * 0.05;
        self.app.world_mut().insert(
            entity,
            Transform::from_position(Vec3::new(x, y, z)),
        );
        self.app.world_mut().insert(
            entity,
            RigidBody::dynamic()
                .with_velocity(Vec3::new(
                    (pseudo_rand(entity.id as u32 + 3) - 0.5) * 2.0,
                    0.0,
                    (pseudo_rand(entity.id as u32 + 11) - 0.5) * 2.0,
                ))
                .with_gravity_scale(1.0),
        );
        self.app.world_mut().insert(entity, ColliderAabb::cube(0.5));
        self.entity_render
            .write()
            .push((entity.id, MeshKind::Cube, material));
        log::debug!("spawned cube entity {}", entity.id);
        let _ = self.audio.play("beep", 0.25);
    }

    fn spawn_ball(&mut self, material: MaterialId) {
        let entity = self.app.world_mut().spawn();
        let x = pseudo_rand(entity.id as u32 + 5) * 6.0 - 3.0;
        let z = pseudo_rand(entity.id as u32 + 13) * 6.0 - 3.0;
        let y = 4.0 + (entity.id as f32) * 0.05;
        self.app.world_mut().insert(
            entity,
            Transform::from_position(Vec3::new(x, y, z)),
        );
        self.app.world_mut().insert(
            entity,
            RigidBody::dynamic()
                .with_velocity(Vec3::new(
                    (pseudo_rand(entity.id as u32 + 9) - 0.5) * 3.0,
                    0.0,
                    (pseudo_rand(entity.id as u32 + 17) - 0.5) * 3.0,
                ))
                .with_gravity_scale(1.0),
        );
        self.app.world_mut().insert(entity, ColliderSphere::new(0.4));
        self.entity_render
            .write()
            .push((entity.id, MeshKind::Sphere, material));
        let _ = self.audio.play("beep", 0.25);
    }

    fn trigger_explosion(&mut self) {
        // Apply an explosion force to all dynamic bodies.
        let origin = Vec3::new(0.0, 0.5, 0.0);
        let start_time = self.state.read().sim_time;
        // Snapshot all dynamic bodies and apply explosion directly to velocities.
        let snapshots: Vec<(u32, Vec3, f32)> = {
            let mut out = Vec::new();
            self.app.world_mut().columns2::<Transform, RigidBody, _, _>(
                |transforms, bodies| {
                    for (id, body) in bodies.iter() {
                        if body.mode != engine_physics::IntegrationMode::Dynamic {
                            continue;
                        }
                        let pos = transforms
                            .get(id)
                            .map(|t| t.position)
                            .unwrap_or(Vec3::ZERO);
                        out.push((id, pos, body.mass));
                    }
                },
            );
            out
        };
        let explosion = ExplosionForce::new(origin, 60.0, 6.0, start_time);
        let dummy_body = RigidBody::default();
        for (id, pos, mass) in snapshots {
            let force = explosion.force_on(&dummy_body, pos, start_time);
            if force.length_squared() < 1e-6 {
                continue;
            }
            let inv_m = 1.0 / mass.max(0.0001);
            let dv = force * inv_m * 0.1; // Apply a single-step impulse.
            self.app.world_mut().with::<RigidBody, _, _>(
                engine_core::Entity { id, gen: 0 },
                |b| {
                    if let Some(body) = b {
                        body.velocity += dv;
                    }
                },
            );
        }
        log::info!("EXPLOSION! Origin={:?} — pushing dynamic bodies apart", origin);
        let _ = self.audio.play("beep", 0.7);
    }
}

impl ApplicationHandler for EngineBetaApp {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }
        let window_attrs = Window::default_attributes()
            .with_title("EngineBeta v0.2.0 — Fly Camera + Lighting + Shadows")
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

        let renderer = match pollster::block_on(engine_render::Renderer::new(
            window.clone(),
            Color::rgb(0.04, 0.05, 0.07),
        )) {
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

        // Build meshes.
        let device = &renderer.ctx.device;
        let cube_mesh = Arc::new(Mesh::new(device, &MeshData::cube([1.0; 3])));
        let sphere_mesh = Arc::new(Mesh::new(
            device,
            &MeshData::sphere([1.0; 3], 0.4, 16, 12),
        ));
        let plane_mesh = Arc::new(Mesh::new(
            device,
            &MeshData::plane_xz([1.0; 3], 30.0),
        ));
        self.state.write().cube_mesh = Some(cube_mesh.clone());
        self.state.write().sphere_mesh = Some(sphere_mesh.clone());
        self.state.write().plane_mesh = Some(plane_mesh.clone());

        // Populate the entity_render list from the world (floor + initial cubes).
        // Floor entity was created in seed_demo_scene; we tag it here.
        let mut er = self.entity_render.write();
        er.clear();
        // Find the floor (a static body with an AABB collider).
        let floor_ids: Vec<u32> = {
            self.app.world_mut().columns2::<RigidBody, ColliderAabb, _, _>(
                |bodies, boxes| {
                    bodies
                        .iter()
                        .filter(|(id, b)| {
                            b.mode == engine_physics::IntegrationMode::Static
                                && boxes.get(*id).is_some()
                        })
                        .map(|(id, _)| id)
                        .collect()
                },
            )
        };
        for fid in floor_ids {
            er.push((fid, MeshKind::Plane, MaterialId::Ground));
        }
        // Tag dynamic AABBs as cubes (wood), dynamic spheres as balls (metal).
        let cube_ids: Vec<u32> = {
            self.app.world_mut().columns2::<RigidBody, ColliderAabb, _, _>(
                |bodies, boxes| {
                    bodies
                        .iter()
                        .filter(|(id, b)| {
                            b.mode == engine_physics::IntegrationMode::Dynamic
                                && boxes.get(*id).is_some()
                        })
                        .map(|(id, _)| id)
                        .collect()
                },
            )
        };
        for cid in cube_ids {
            er.push((cid, MeshKind::Cube, MaterialId::Wood));
        }
        let ball_ids: Vec<u32> = {
            self.app.world_mut().columns2::<RigidBody, ColliderSphere, _, _>(
                |bodies, spheres| {
                    bodies
                        .iter()
                        .filter(|(id, b)| {
                            b.mode == engine_physics::IntegrationMode::Dynamic
                                && spheres.get(*id).is_some()
                        })
                        .map(|(id, _)| id)
                        .collect()
                },
            )
        };
        for bid in ball_ids {
            er.push((bid, MeshKind::Sphere, MaterialId::Metal));
        }

        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        self.input.process_window_event(&event);

        let mut st = self.state.write();
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
                    st.pending_drop_cube = st.pending_drop_cube.saturating_add(1);
                }
                KeyCode::KeyB => {
                    st.pending_drop_ball = st.pending_drop_ball.saturating_add(1);
                }
                KeyCode::KeyE => {
                    st.pending_explosion = true;
                }
                _ => {}
            },
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    st.mouse_buttons_down = state == ElementState::Pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new = glam::Vec2::new(position.x as f32, position.y as f32);
                if let Some(prev) = st.last_mouse_pos {
                    // Mouse look: only when right mouse button is held? For simplicity,
                    // always apply mouse-look (the cursor is captured by the window in
                    // most desktop environments when clicked). Use small sensitivity.
                    let dx = new.x - prev.x;
                    let dy = new.y - prev.y;
                    st.fly_cam.look(dx, dy);
                }
                st.last_mouse_pos = Some(new);
            }
            WindowEvent::Resized(size) => {
                if let Some(r) = self.renderer.as_mut() {
                    r.resize(size.width.max(1), size.height.max(1));
                }
                st.fly_cam.resize(size.width.max(1), size.height.max(1));
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.input.process_gilrs_events();

        // Handle queued input actions.
        let mut st = self.state.write();
        if st.pending_explosion {
            drop(st);
            self.trigger_explosion();
            self.state.write().pending_explosion = false;
        } else {
            let cubes = std::mem::take(&mut st.pending_drop_cube);
            let balls = std::mem::take(&mut st.pending_drop_ball);
            drop(st);
            for _ in 0..cubes {
                self.spawn_cube(MaterialId::PlasticRed);
            }
            for _ in 0..balls {
                self.spawn_ball(MaterialId::PlasticBlue);
            }
        }

        // Apply fly-camera movement from input.
        {
            let st = self.state.read();
            if let Some(res) = self.app.world().resource::<engine_input::InputResource>() {
                let snap = res.snapshot();
                let dt = 0.016; // Approximate; could use Time from app.
                let forward = (if snap.keys_down.contains(&KeyCode::KeyW) { 1.0 } else { 0.0 })
                    - if snap.keys_down.contains(&KeyCode::KeyS) { 1.0 } else { 0.0 };
                let right = (if snap.keys_down.contains(&KeyCode::KeyD) { 1.0 } else { 0.0 })
                    - if snap.keys_down.contains(&KeyCode::KeyA) { 1.0 } else { 0.0 };
                let up = (if snap.keys_down.contains(&KeyCode::Space) { 1.0 } else { 0.0 })
                    - if snap.keys_down.contains(&KeyCode::ShiftLeft) { 1.0 } else { 0.0 };
                // Can't mutate fly_cam through a read guard; re-acquire write.
                drop(st);
                let mut st = self.state.write();
                st.fly_cam.move_flat(forward, right, dt);
                st.fly_cam.move_vertical(up, dt);
            }
        }

        // Tick the simulation.
        let started = Instant::now();
        self.app.tick_once();
        let physics_us = started.elapsed();

        // Update sim_time + run force registry.
        {
            let mut st = self.state.write();
            st.sim_time += self.app.time().delta_secs();
            st.fps.update(self.app.time().delta);
            // Periodic profiler log.
            if st.last_profiler_log.elapsed() > Duration::from_secs(2) {
                st.last_profiler_log = Instant::now();
                let frame_ms = st.fps.frame_time_ms();
                let fps = st.fps.fps();
                let entity_count = self.app.world().entity_count();
                let draw_calls = self.entity_render.read().len();
                log::info!(
                    "PROFILER | fps={:.1} | frame={:.2}ms | physics={:.2}ms | entities={} | draw_calls={}",
                    fps,
                    frame_ms,
                    physics_us.as_secs_f32() * 1000.0,
                    entity_count,
                    draw_calls,
                );
            }
        }

        // Build draw calls from the entity_render list.
        let (cube_mesh, sphere_mesh, plane_mesh, mats, fly_cam) = {
            let st = self.state.read();
            (
                st.cube_mesh.clone(),
                st.sphere_mesh.clone(),
                st.plane_mesh.clone(),
                st.materials,
                st.fly_cam,
            )
        };

        let entity_render = self.entity_render.read().clone();
        let mut draw_calls: Vec<DrawCall> = Vec::with_capacity(entity_render.len() + 1);
        // First, snapshot all transforms.
        let transforms: Vec<(u32, Mat4)> = {
            let col = self.app.world().column_read::<Transform>();
            let mut out = Vec::new();
            if let Some(col) = col {
                for (id, t) in col.iter() {
                    out.push((id, t.matrix()));
                }
            }
            out
        };
        for (id, mesh_kind, mat_id) in &entity_render {
            let mesh = match mesh_kind {
                MeshKind::Cube => cube_mesh.as_ref(),
                MeshKind::Sphere => sphere_mesh.as_ref(),
                MeshKind::Plane => plane_mesh.as_ref(),
            };
            let Some(mesh) = mesh else { continue };
            let Some((_, model)) = transforms.iter().find(|(tid, _)| tid == id) else {
                continue;
            };
            let material = match mat_id {
                MaterialId::Wood => &mats.wood,
                MaterialId::Metal => &mats.metal,
                MaterialId::PlasticRed => &mats.plastic_red,
                MaterialId::PlasticBlue => &mats.plastic_blue,
                MaterialId::Emissive => &mats.emissive,
                MaterialId::Ground => &mats.ground,
            };
            draw_calls.push(DrawCall {
                mesh: mesh.as_ref(),
                model: *model,
                material,
            });
        }

        // Build light uniform: ambient + directional, with a shadow matrix
        // derived from an ortho camera at the light's POV.
        let sun_dir = Vec3::new(-0.4, -1.0, -0.3).normalize();
        let light_ortho = OrthoCamera::new(Vec3::new(0.0, 0.0, 0.0), 15.0, sun_dir);
        let lights = LightUniform {
            ambient_color: [0.18, 0.20, 0.28, 0.45],
            sun_direction: [sun_dir.x, sun_dir.y, sun_dir.z, 1.4],
            sun_color: [1.0, 0.96, 0.88, 1.0],
            light_view_proj: light_ortho.view_proj().to_cols_array_2d(),
            camera_pos: [fly_cam.position.x, fly_cam.position.y, fly_cam.position.z, 0.0],
        };

        // Render.
        let camera = fly_cam.as_camera();
        if let Some(r) = self.renderer.as_mut() {
            if let Err(e) = r.render(&camera, &lights, &draw_calls) {
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

/// Seed the demo scene: a static ground plane + a few stacked cubes + a couple
/// of falling balls + an AI enemy.
fn seed_demo_scene(world: &mut World, floor_y: f32) {
    // Ground (large static box).
    let floor = world.spawn();
    world.insert(
        floor,
        Transform {
            position: Vec3::new(0.0, floor_y - 0.5, 0.0),
            ..Default::default()
        },
    );
    world.insert(floor, RigidBody::static_body());
    world.insert(floor, ColliderAabb::new(Vec3::new(15.0, 0.5, 15.0)));

    // Stack of 3 wood cubes.
    for i in 0..3 {
        let cube = world.spawn();
        world.insert(
            cube,
            Transform::from_position(Vec3::new(0.0, 0.0 + i as f32 * 1.1, 0.0)),
        );
        world.insert(cube, RigidBody::dynamic());
        world.insert(cube, ColliderAabb::cube(0.5));
    }

    // A couple of falling balls.
    for i in 0..2 {
        let ball = world.spawn();
        world.insert(
            ball,
            Transform::from_position(Vec3::new(2.0 + i as f32, 3.0, 1.0)),
        );
        world.insert(ball, RigidBody::dynamic());
        world.insert(ball, ColliderSphere::new(0.4));
    }

    // A static pedestal (so cubes can stack on it).
    let pedestal = world.spawn();
    world.insert(
        pedestal,
        Transform::from_position(Vec3::new(-3.0, floor_y + 0.25, -2.0)),
    );
    world.insert(pedestal, RigidBody::static_body());
    world.insert(pedestal, ColliderAabb::new(Vec3::new(0.75, 0.25, 0.75)));

    // AI enemy: a sphere with vision + hearing.
    let enemy = world.spawn();
    world.insert(
        enemy,
        Transform::from_position(Vec3::new(-5.0, floor_y + 0.5, 5.0)),
    );
    world.insert(enemy, RigidBody::kinematic());
    world.insert(enemy, ColliderSphere::new(0.5));
    world.insert(enemy, VisionSensor::default());
    world.insert(enemy, HearingSensor::default());

    // A "player" target that the enemy can perceive. We'll move it around.
    let player = world.spawn();
    world.insert(
        player,
        Transform::from_position(Vec3::new(3.0, floor_y + 0.5, 0.0)),
    );
    world.insert(player, Perceivable);
}

/// Deterministic pseudo-random in [0, 1) from a u32 seed.
fn pseudo_rand(seed: u32) -> f32 {
    let mut x = seed.wrapping_mul(2654435761);
    x ^= x >> 16;
    x = x.wrapping_mul(2246822519);
    x ^= x >> 13;
    (x as f32) / (u32::MAX as f32)
}
