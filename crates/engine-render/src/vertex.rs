//! Vertex layout for the default unlit pipeline.

use bytemuck::{Pod, Zeroable};

/// A simple unlit vertex: position + per-vertex color.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
        ],
    };

    pub fn new(pos: [f32; 3], color: [f32; 3]) -> Self {
        Self {
            position: pos,
            color,
        }
    }

    pub fn rgb(r: f32, g: f32, b: f32) -> [f32; 3] {
        [r, g, b]
    }
}
