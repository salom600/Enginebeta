//! Light components: ambient, directional, point, spot.
//!
//! Lights are stored as ECS components so the renderer can iterate them.
//! For the MVP we support:
//! - [`AmbientLight`] — flat fill light applied to every pixel
//! - [`DirectionalLight`] — sun-like parallel rays, casts shadows
//! - [`PointLight`] — radial falloff (no shadows in MVP)

use glam::Vec3;

/// Soft, scene-wide fill light. Adds a constant color to every fragment.
/// Without this, shadowed areas would be pure black.
#[derive(Copy, Clone, Debug)]
pub struct AmbientLight {
    pub color: [f32; 3],
    pub intensity: f32,
}

impl Default for AmbientLight {
    fn default() -> Self {
        Self {
            color: [0.18, 0.20, 0.28],
            intensity: 0.35,
        }
    }
}

/// Sun-like parallel light. Direction is in world space (pointing AWAY from
/// the light source, towards the scene). The single directional light in the
/// scene casts shadows into a single shadow map.
#[derive(Copy, Clone, Debug)]
pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: [f32; 3],
    pub intensity: f32,
    /// Whether this light casts shadows. Only the first shadow-casting
    /// directional light is rendered to the shadow map.
    pub cast_shadows: bool,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(-0.4, -1.0, -0.3).normalize(),
            color: [1.0, 0.96, 0.88],
            intensity: 1.2,
            cast_shadows: true,
        }
    }
}

/// Radial point light with linear + quadratic falloff.
#[derive(Copy, Clone, Debug)]
pub struct PointLight {
    pub position: Vec3,
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            radius: 10.0,
        }
    }
}

/// CPU-side light uniform block uploaded to the shader each frame.
/// Mirrors the `SceneUniform` struct in the WGSL shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    pub ambient_color: [f32; 4],   // rgb + intensity
    pub sun_direction: [f32; 4],   // xyz + intensity
    pub sun_color: [f32; 4],       // rgb + cast_shadows (1.0 or 0.0)
    pub light_view_proj: [[f32; 4]; 4], // shadow-space matrix
    pub camera_pos: [f32; 4],      // xyz + pad
}

impl Default for LightUniform {
    fn default() -> Self {
        Self {
            ambient_color: [0.18, 0.20, 0.28, 0.35],
            sun_direction: [-0.4, -1.0, -0.3, 1.2],
            sun_color: [1.0, 0.96, 0.88, 1.0],
            light_view_proj: [[0.0; 4]; 4],
            camera_pos: [0.0; 4],
        }
    }
}
