//! engine-input — keyboard, mouse, and gamepad input for EngineBeta.
//!
//! The engine keeps a single [`InputState`] resource that gameplay code polls
//! each frame. Window events (from winit) and gamepad events (from gilrs) are
//! fed in by [`InputContext::process_window_event`] and
//! [`InputContext::process_gilrs_events`].

use engine_core::World;
use gilrs::{Event as GilrsEvent, EventType, Gilrs};
use glam::Vec2;
use parking_lot::RwLock;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};

/// A single input axis (e.g. WASD on the keyboard, a thumbstick on a gamepad).
#[derive(Copy, Clone, Debug, Default)]
pub struct AxisState {
    pub x: f32,
    pub y: f32,
}

/// Snapshot of all input devices at a moment in time. Stored as a resource in
/// the [`World`] via [`InputContext::install`].
#[derive(Default)]
pub struct InputState {
    /// Keys currently held down (logical key codes).
    pub keys_down: HashSet<winit::keyboard::KeyCode>,
    /// Keys pressed this frame (transitioned from up → down).
    pub keys_pressed: HashSet<winit::keyboard::KeyCode>,
    /// Keys released this frame (transitioned from down → up).
    pub keys_released: HashSet<winit::keyboard::KeyCode>,
    /// Mouse buttons currently held down.
    pub mouse_buttons_down: HashSet<MouseButton>,
    pub mouse_position: Vec2,
    pub mouse_delta: Vec2,
    pub mouse_scroll: Vec2,
    /// Left gamepad stick (in [-1, 1]²).
    pub left_stick: AxisState,
    /// Right gamepad stick (in [-1, 1]²).
    pub right_stick: AxisState,
    /// Left and right triggers (in [0, 1]).
    pub left_trigger: f32,
    pub right_trigger: f32,
    /// Gamepad buttons pressed this frame.
    pub gamepad_buttons_pressed: HashSet<gilrs::Button>,
    pub gamepad_buttons_down: HashSet<gilrs::Button>,
}

impl InputState {
    /// Was this key held down at the start of this frame?
    pub fn key(&self, k: winit::keyboard::KeyCode) -> bool {
        self.keys_down.contains(&k)
    }
    /// Was this key pressed this frame (edge)?
    pub fn key_pressed(&self, k: winit::keyboard::KeyCode) -> bool {
        self.keys_pressed.contains(&k)
    }
    /// Was this mouse button held down?
    pub fn mouse(&self, b: MouseButton) -> bool {
        self.mouse_buttons_down.contains(&b)
    }
    /// WASD-style movement vector from keyboard (Y = forward = W).
    pub fn wasd(&self) -> Vec2 {
        let mut v = Vec2::ZERO;
        if self.key(winit::keyboard::KeyCode::KeyW) {
            v.y += 1.0;
        }
        if self.key(winit::keyboard::KeyCode::KeyS) {
            v.y -= 1.0;
        }
        if self.key(winit::keyboard::KeyCode::KeyA) {
            v.x -= 1.0;
        }
        if self.key(winit::keyboard::KeyCode::KeyD) {
            v.x += 1.0;
        }
        v
    }
    /// Combined movement vector (keyboard WASD + left gamepad stick).
    pub fn movement(&self) -> Vec2 {
        let mut v = self.wasd();
        v.x += self.left_stick.x;
        v.y += self.left_stick.y;
        // Clamp to unit circle to avoid diagonal speed-up.
        if v.length() > 1.0 {
            v = v.normalize();
        }
        v
    }
}

/// Owns the input state and (optionally) a `Gilrs` gamepad context wrapped in
/// a `Mutex` (Gilrs is `Send` but not `Sync` because it contains an mpsc Receiver).
pub struct InputContext {
    pub state: Arc<RwLock<InputState>>,
    pub gilrs: Option<Mutex<Gilrs>>,
}

impl InputContext {
    /// Create a new input context. Tries to initialize gilrs; failure is logged
    /// and the engine continues with keyboard/mouse only.
    pub fn new() -> Self {
        let gilrs = match Gilrs::new() {
            Ok(g) => {
                log::info!("gilrs initialized");
                Some(Mutex::new(g))
            }
            Err(e) => {
                log::warn!("gilrs failed to initialize: {e}");
                None
            }
        };
        Self {
            state: Arc::new(RwLock::new(InputState::default())),
            gilrs,
        }
    }

    /// Install this context as a resource on the [`World`] so gameplay systems
    /// can read the input state.
    pub fn install(self, world: &mut World) {
        world.insert_resource(InputResource {
            state: self.state.clone(),
        });
        // Stash the gilrs handle in a separate resource.
        world.insert_resource(GilrsResource {
            gilrs: self.gilrs,
        });
    }

    /// Feed a winit window event into the input state.
    pub fn process_window_event(&self, event: &WindowEvent) {
        let mut s = self.state.write();
        match event {
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key,
                    state,
                    ..
                },
                ..
            } => {
                if let winit::keyboard::PhysicalKey::Code(code) = physical_key {
                    match state {
                        ElementState::Pressed => {
                            if s.keys_down.insert(*code) {
                                s.keys_pressed.insert(*code);
                            }
                        }
                        ElementState::Released => {
                            s.keys_down.remove(code);
                            s.keys_released.insert(*code);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => match state {
                ElementState::Pressed => {
                    s.mouse_buttons_down.insert(*button);
                }
                ElementState::Released => {
                    s.mouse_buttons_down.remove(button);
                }
            },
            WindowEvent::CursorMoved { position, .. } => {
                let new = Vec2::new(position.x as f32, position.y as f32);
                s.mouse_delta = new - s.mouse_position;
                s.mouse_position = new;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                s.mouse_scroll = match delta {
                    MouseScrollDelta::LineDelta(x, y) => Vec2::new(*x, *y),
                    MouseScrollDelta::PixelDelta(p) => Vec2::new(p.x as f32, p.y as f32),
                };
            }
            _ => {}
        }
    }

    /// Drain pending gilrs events and apply them to the input state.
    pub fn process_gilrs_events(&mut self) {
        let Some(gilrs_mtx) = self.gilrs.as_ref() else {
            return;
        };
        let mut gilrs = match gilrs_mtx.lock() {
            Ok(g) => g,
            Err(e) => {
                log::warn!("gilrs mutex poisoned: {e}");
                return;
            }
        };
        let mut s = self.state.write();
        // Poll events.
        while let Some(GilrsEvent { event, .. }) = gilrs.next_event() {
            match event {
                EventType::ButtonPressed(b, _) => {
                    s.gamepad_buttons_pressed.insert(b);
                    s.gamepad_buttons_down.insert(b);
                }
                EventType::ButtonReleased(b, _) => {
                    s.gamepad_buttons_down.remove(&b);
                }
                EventType::AxisChanged(axis, value, _) => match axis {
                    gilrs::Axis::LeftStickX => s.left_stick.x = value,
                    gilrs::Axis::LeftStickY => s.left_stick.y = value,
                    gilrs::Axis::RightStickX => s.right_stick.x = value,
                    gilrs::Axis::RightStickY => s.right_stick.y = value,
                    gilrs::Axis::LeftZ => s.left_trigger = (value + 1.0) * 0.5,
                    gilrs::Axis::RightZ => s.right_trigger = (value + 1.0) * 0.5,
                    _ => {}
                },
                _ => {}
            }
        }
        // Also refresh current axis values from active gamepads in case events
        // were missed.
        for (_id, pad) in gilrs.gamepads() {
            s.left_stick.x = pad.value(gilrs::Axis::LeftStickX);
            s.left_stick.y = pad.value(gilrs::Axis::LeftStickY);
            s.right_stick.x = pad.value(gilrs::Axis::RightStickX);
            s.right_stick.y = pad.value(gilrs::Axis::RightStickY);
            s.left_trigger = (pad.value(gilrs::Axis::LeftZ) + 1.0) * 0.5;
            s.right_trigger = (pad.value(gilrs::Axis::RightZ) + 1.0) * 0.5;
            break; // first pad only for MVP
        }
    }
}

/// Resource stored in [`World`] so gameplay systems can read the input state.
pub struct InputResource {
    pub state: Arc<RwLock<InputState>>,
}

impl InputResource {
    /// Snapshot the input state for this frame. Cheap (Arc clone + small RwLock read).
    pub fn snapshot(&self) -> InputStateSnapshot {
        let s = self.state.read();
        InputStateSnapshot {
            keys_down: s.keys_down.clone(),
            keys_pressed: s.keys_pressed.clone(),
            mouse_buttons_down: s.mouse_buttons_down.clone(),
            mouse_position: s.mouse_position,
            mouse_delta: s.mouse_delta,
            left_stick: s.left_stick,
            right_stick: s.right_stick,
        }
    }
}

/// Cheap snapshot of input state for use within a single system.
pub struct InputStateSnapshot {
    pub keys_down: HashSet<winit::keyboard::KeyCode>,
    pub keys_pressed: HashSet<winit::keyboard::KeyCode>,
    pub mouse_buttons_down: HashSet<MouseButton>,
    pub mouse_position: Vec2,
    pub mouse_delta: Vec2,
    pub left_stick: AxisState,
    pub right_stick: AxisState,
}

/// Resource holding the gilrs context (so it can be polled each frame).
pub struct GilrsResource {
    pub gilrs: Option<Mutex<Gilrs>>,
}

/// System: clear per-frame input edges (pressed / released / delta / scroll).
/// Insert into [`engine_core::Stage::PreUpdate`].
pub fn clear_frame_edges(world: &mut World) {
    if let Some(res) = world.resource::<InputResource>() {
        let mut s = res.state.write();
        s.keys_pressed.clear();
        s.keys_released.clear();
        s.mouse_delta = Vec2::ZERO;
        s.mouse_scroll = Vec2::ZERO;
        s.gamepad_buttons_pressed.clear();
    }
}
