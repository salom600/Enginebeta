//! CPU-side mesh data and GPU-resident mesh.

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
    /// Build a unit cube (side length 1) centered on the origin, with proper
    /// face normals and a per-vertex color tint.
    pub fn cube(color: [f32; 3]) -> Self {
        let c = color;
        // 24 vertices (4 per face, so each face has its own flat normal).
        // 6 faces × 2 triangles × 3 indices = 36 indices.
        let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
            // +Z face: normal (0, 0, 1)
            (
                [0.0, 0.0, 1.0],
                [
                    [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                ],
            ),
            // -Z face
            (
                [0.0, 0.0, -1.0],
                [
                    [0.5, -0.5, -0.5],
                    [-0.5, -0.5, -0.5],
                    [-0.5, 0.5, -0.5],
                    [0.5, 0.5, -0.5],
                ],
            ),
            // +X face
            (
                [1.0, 0.0, 0.0],
                [
                    [0.5, -0.5, 0.5],
                    [0.5, -0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [0.5, 0.5, 0.5],
                ],
            ),
            // -X face
            (
                [-1.0, 0.0, 0.0],
                [
                    [-0.5, -0.5, -0.5],
                    [-0.5, -0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                    [-0.5, 0.5, -0.5],
                ],
            ),
            // +Y face (top)
            (
                [0.0, 1.0, 0.0],
                [
                    [-0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, -0.5],
                    [-0.5, 0.5, -0.5],
                ],
            ),
            // -Y face (bottom)
            (
                [0.0, -1.0, 0.0],
                [
                    [-0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5],
                    [0.5, -0.5, 0.5],
                    [-0.5, -0.5, 0.5],
                ],
            ),
        ];

        let mut vertices: Vec<crate::Vertex> = Vec::with_capacity(24);
        let mut indices: Vec<u16> = Vec::with_capacity(36);
        for (normal, corners) in faces.iter() {
            let base = vertices.len() as u16;
            for &p in corners.iter() {
                vertices.push(crate::Vertex::new(p, *normal, c));
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        Self { vertices, indices }
    }

    /// Build a flat quad in the XZ plane (good for floors / ground).
    pub fn plane_xz(color: [f32; 3], size: f32) -> Self {
        let c = color;
        let h = size * 0.5;
        let normal = [0.0, 1.0, 0.0];
        let vertices: Vec<crate::Vertex> = vec![
            crate::Vertex::new([-h, 0.0, h], normal, c),
            crate::Vertex::new([h, 0.0, h], normal, c),
            crate::Vertex::new([h, 0.0, -h], normal, c),
            crate::Vertex::new([-h, 0.0, -h], normal, c),
        ];
        let indices: Vec<u16> = vec![0, 1, 2, 0, 2, 3];
        Self { vertices, indices }
    }

    /// Build a UV sphere of given radius, centered on origin.
    pub fn sphere(color: [f32; 3], radius: f32, segments: u32, rings: u32) -> Self {
        let c = color;
        let mut vertices: Vec<crate::Vertex> = Vec::new();
        let mut indices: Vec<u16> = Vec::new();
        for ring in 0..=rings {
            let phi = std::f32::consts::PI * (ring as f32 / rings as f32); // 0..PI
            let y = radius * phi.cos();
            let r = radius * phi.sin();
            for seg in 0..=segments {
                let theta = std::f32::consts::TAU * (seg as f32 / segments as f32);
                let x = r * theta.cos();
                let z = r * theta.sin();
                let normal = [x / radius, y / radius, z / radius];
                vertices.push(crate::Vertex::new([x, y, z], normal, c));
            }
        }
        let segs = (segments + 1) as u16;
        for ring in 0..rings as u16 {
            for seg in 0..segments as u16 {
                let a = ring * segs + seg;
                let b = (ring + 1) * segs + seg;
                indices.extend_from_slice(&[a, b, a + 1, b, b + 1, a + 1]);
            }
        }
        Self { vertices, indices }
    }
}
