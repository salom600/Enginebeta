//! Vertex layout for the PBR-ish pipeline: position + normal + albedo tint.

use bytemuck::{Pod, Zeroable};

/// A vertex with position, normal, and per-vertex albedo tint.
/// Normals are required for the lighting calculation.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3, // position
            1 => Float32x3, // normal
            2 => Float32x3, // color
        ],
    };

    pub fn new(pos: [f32; 3], normal: [f32; 3], color: [f32; 3]) -> Self {
        Self {
            position: pos,
            normal,
            color,
        }
    }
}
