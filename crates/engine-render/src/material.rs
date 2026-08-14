//! Material system — surface properties passed to the fragment shader.
//!
//! A material defines how light interacts with a surface:
//! - `albedo` — base color (also called diffuse)
//! - `metallic` — 0 = dielectric (wood, plastic), 1 = metal
//! - `roughness` — 0 = mirror-smooth, 1 = totally matte
//! - `emissive` — self-illumination (added on top of lit color)
//!
//! The renderer uploads a [`MaterialUniform`] per draw call via a dynamic
//! uniform offset, so each mesh can have its own material.

use bytemuck::{Pod, Zeroable};

/// CPU-side material. Stored as a component on entities that have a mesh.
#[derive(Copy, Clone, Debug)]
pub struct Material {
    pub albedo: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo: [0.8, 0.8, 0.8],
            metallic: 0.0,
            roughness: 0.6,
            emissive: [0.0, 0.0, 0.0],
        }
    }
}

impl Material {
    /// Matte wood-like surface: warm tan, fully dielectric, fairly rough.
    pub fn wood() -> Self {
        Self {
            albedo: [0.55, 0.40, 0.25],
            metallic: 0.0,
            roughness: 0.85,
            emissive: [0.0; 3],
        }
    }
    /// Polished metal: bright silver, fully metallic, low roughness.
    pub fn metal() -> Self {
        Self {
            albedo: [0.85, 0.85, 0.88],
            metallic: 1.0,
            roughness: 0.25,
            emissive: [0.0; 3],
        }
    }
    /// Glowing accent: ignores lighting, always emits.
    pub fn emissive(rgb: [f32; 3]) -> Self {
        Self {
            albedo: [0.0; 3],
            metallic: 0.0,
            roughness: 1.0,
            emissive: rgb,
        }
    }
    /// Rubber-like matte plastic.
    pub fn plastic(rgb: [f32; 3]) -> Self {
        Self {
            albedo: rgb,
            metallic: 0.0,
            roughness: 0.7,
            emissive: [0.0; 3],
        }
    }
}

/// GPU-side material uniform. Mirrors `MaterialUniform` in WGSL.
/// Padded to 48 bytes (3 × vec4) to satisfy std140 alignment.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MaterialUniform {
    pub albedo: [f32; 4],        // rgb + metallic
    pub roughness_emissive: [f32; 4], // roughness + emissive.rgb + pad
}

impl From<&Material> for MaterialUniform {
    fn from(m: &Material) -> Self {
        Self {
            albedo: [m.albedo[0], m.albedo[1], m.albedo[2], m.metallic],
            roughness_emissive: [m.roughness, m.emissive[0], m.emissive[1], m.emissive[2]],
        }
    }
}
