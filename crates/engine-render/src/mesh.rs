//! GPU-resident mesh: vertex + index buffer + draw count.

use wgpu::util::DeviceExt;

/// A mesh that has been uploaded to the GPU.
pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

impl Mesh {
    /// Upload `data` to the GPU as a new mesh.
    pub fn new(device: &wgpu::Device, data: &MeshData) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("enginebeta_mesh_vb"),
            contents: bytemuck::cast_slice(&data.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("enginebeta_mesh_ib"),
            contents: bytemuck::cast_slice(&data.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertex_buffer,
            index_buffer,
            index_count: data.indices.len() as u32,
        }
    }
}

/// CPU-side mesh data used to construct a [`Mesh`].
pub struct MeshData {
    pub vertices: Vec<crate::Vertex>,
    pub indices: Vec<u16>,
}

impl MeshData {
    /// Build a unit cube (side length 1) centered on the origin, with per-face
    /// colors so the unlit shader still shows orientation.
    pub fn cube(face_color: [f32; 3]) -> Self {
        let c = face_color;
        let p = |x: f32, y: f32, z: f32| [x, y, z];
        let positions = [
            p(-0.5, -0.5, 0.5),  // 0
            p(0.5, -0.5, 0.5),   // 1
            p(0.5, 0.5, 0.5),    // 2
            p(-0.5, 0.5, 0.5),   // 3
            p(-0.5, -0.5, -0.5), // 4
            p(0.5, -0.5, -0.5),  // 5
            p(0.5, 0.5, -0.5),   // 6
            p(-0.5, 0.5, -0.5),  // 7
        ];
        let vertices: Vec<crate::Vertex> = positions
            .iter()
            .map(|p| crate::Vertex::new(*p, c))
            .collect();
        let indices: Vec<u16> = vec![
            0, 1, 2, 0, 2, 3, // +Z
            5, 4, 7, 5, 7, 6, // -Z
            4, 0, 3, 4, 3, 7, // -X
            1, 5, 6, 1, 6, 2, // +X
            3, 2, 6, 3, 6, 7, // +Y
            4, 5, 1, 4, 1, 0, // -Y
        ];
        Self { vertices, indices }
    }

    /// Build a flat quad in the XY plane (good for sprites / UI when projected ortho).
    pub fn quad(color: [f32; 3]) -> Self {
        let c = color;
        let vertices: Vec<crate::Vertex> = vec![
            crate::Vertex::new([-0.5, -0.5, 0.0], c),
            crate::Vertex::new([0.5, -0.5, 0.0], c),
            crate::Vertex::new([0.5, 0.5, 0.0], c),
            crate::Vertex::new([-0.5, 0.5, 0.0], c),
        ];
        let indices: Vec<u16> = vec![0, 1, 2, 0, 2, 3];
        Self { vertices, indices }
    }
}
