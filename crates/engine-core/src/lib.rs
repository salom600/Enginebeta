//! engine-core — the foundation of EngineBeta.
//!
//! Provides:
//! - [`App`] — the main loop / scheduler
//! - [`World`] — a tiny, fast entity–component store
//! - [`Time`] — frame timing and fixed-step accumulator
//! - [`Transform`] — position / rotation / scale
//! - [`Color`] — linear RGBA
//!
//! Every other engine-* crate builds on top of this.

pub mod app;
pub mod color;
pub mod ecs;
pub mod geom;
pub mod time;
pub mod transform;

pub use app::{App, AppBuilder, Stage};
pub use color::Color;
pub use ecs::{Component, Entity, World};
pub use geom::{Aabb, Plane, Ray};
pub use time::{FpsCounter, Time};
pub use transform::Transform;

/// Re-export of `glam` vector / matrix types so the whole engine uses one math backend.
pub use glam::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};

/// Semantic version of the engine.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
