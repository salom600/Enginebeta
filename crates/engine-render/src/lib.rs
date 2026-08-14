//! engine-render — wgpu-based renderer for EngineBeta (v0.2.0).
//!
//! v0.2.0 additions:
//! - Directional + ambient lighting (Blinn-Phong)
//! - Shadow map for the directional light
//! - Per-mesh materials (albedo, metallic, roughness, emissive)
//! - Per-mesh model matrices (so each cube can have its own transform)
//! - Raycasting for mouse picking
//! - FlyCamera for first-person navigation
//!
//! v0.2.1 hotfix: replaced `push_constant` (unsupported on DX12) with a
//! dynamic-offset uniform buffer for per-draw-call model matrices.

pub mod camera;
pub mod light;
pub mod material;
pub mod mesh;
pub mod raycast;
pub mod vertex;

pub use camera::{Camera, FlyCamera, OrthoCamera};
pub use light::{AmbientLight, DirectionalLight, LightUniform, PointLight};
pub use material::{Material, MaterialUniform};
pub use mesh::{Mesh, MeshData};
pub use raycast::{Raycast, RaycastHit};
pub use vertex::Vertex;

use crate::context::RenderContext;
use engine_core::Color;
use glam::Mat4;
use wgpu::SurfaceError;

/// Size of one ModelUniform entry in the dynamic uniform buffer.
/// Wgpu requires uniform buffer offsets to be aligned to `min_uniform_buffer_offset_alignment`,
/// which is 256 on virtually every backend. We pad to 256 to be safe.
pub const MODEL_UNIFORM_STRIDE: u64 = 256;
/// Same for materials (32 bytes content, padded to 256).
pub const MATERIAL_UNIFORM_STRIDE: u64 = 256;
/// Max draw calls supported per frame.
///
/// Capped at 64 (not 256) so the total uniform buffer binding stays at
/// 64 × 256 = 16384 bytes — exactly DX12's `max_*_buffer_binding_size` limit
/// (inherited from `D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT` = 4096 floats).
/// Going above 64 draw calls per frame would require switching to a storage
/// buffer, which has different alignment rules and is overkill for the MVP.
pub const MAX_DRAW_CALLS: usize = 64;

/// A single draw call: a mesh + its world-space transform + its material.
pub struct DrawCall<'a> {
    pub mesh: &'a Mesh,
    pub model: Mat4,
    pub material: &'a Material,
}

/// Top-level renderer. Owns the pipeline, shadow map, and per-frame uniforms.
pub struct Renderer {
    pub ctx: RenderContext,
    pub clear_color: Color,
    pub depth_format: wgpu::TextureFormat,
    pub shadow_format: wgpu::TextureFormat,
    pub config: wgpu::SurfaceConfiguration,
    pipeline: Option<wgpu::RenderPipeline>,
    shadow_pipeline: Option<wgpu::RenderPipeline>,
    scene_uniform_buffer: Option<wgpu::Buffer>,
    scene_bind_group: Option<wgpu::BindGroup>,
    scene_bind_group_layout: Option<wgpu::BindGroupLayout>,
    material_uniform_buffer: Option<wgpu::Buffer>,
    material_bind_group: Option<wgpu::BindGroup>,
    material_bind_group_layout: Option<wgpu::BindGroupLayout>,
    model_uniform_buffer: Option<wgpu::Buffer>,
    model_bind_group: Option<wgpu::BindGroup>,
    model_bind_group_layout: Option<wgpu::BindGroupLayout>,
    shadow_texture: Option<wgpu::Texture>,
    shadow_view: Option<wgpu::TextureView>,
    shadow_sampler: Option<wgpu::Sampler>,
}

impl Renderer {
    /// Create a renderer bound to a winit window.
    pub async fn new(
        window: std::sync::Arc<winit::window::Window>,
        clear_color: Color,
    ) -> anyhow::Result<Self> {
        let size = window.inner_size();
        let ctx = RenderContext::new(window).await?;
        let surface_caps = ctx.surface.get_capabilities(&ctx.adapter);
        let format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        ctx.surface.configure(&ctx.device, &config);

        Ok(Self {
            ctx,
            clear_color,
            depth_format: wgpu::TextureFormat::Depth32Float,
            shadow_format: wgpu::TextureFormat::Depth32Float,
            pipeline: None,
            shadow_pipeline: None,
            scene_uniform_buffer: None,
            scene_bind_group: None,
            scene_bind_group_layout: None,
            material_uniform_buffer: None,
            material_bind_group: None,
            material_bind_group_layout: None,
            model_uniform_buffer: None,
            model_bind_group: None,
            model_bind_group_layout: None,
            shadow_texture: None,
            shadow_view: None,
            shadow_sampler: None,
            config,
        })
    }

    /// Build the main render pipeline + shadow pipeline + uniform buffers.
    /// Call once after construction.
    pub fn build_pipeline(&mut self) -> anyhow::Result<()> {
        // --- Shaders ---
        let main_shader = self
            .ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("enginebeta_main_shader"),
                source: wgpu::ShaderSource::Wgsl(MAIN_SHADER_SRC.into()),
            });
        let shadow_shader = self
            .ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("enginebeta_shadow_shader"),
                source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER_SRC.into()),
            });

        // --- Scene uniform bind group (camera + lights + shadow matrix) ---
        let scene_bgl =
            self.ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("scene_bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Depth,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                            count: None,
                        },
                    ],
                });

        let scene_ubo = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene_ubo"),
            size: std::mem::size_of::<SceneUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shadow map texture (2048x2048 depth-only).
        let shadow_texture = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_map"),
            size: wgpu::Extent3d {
                width: 2048,
                height: 2048,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.shadow_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_sampler = self.ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let scene_bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_bg"),
            layout: &scene_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
            ],
        });

        // --- Material uniform bind group (dynamic offset) ---
        let material_bgl =
            self.ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("material_bgl"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        // Stash MAX_DRAW_CALLS materials, each MATERIAL_UNIFORM_STRIDE bytes.
        let material_ubo = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material_ubo"),
            size: MAX_DRAW_CALLS as u64 * MATERIAL_UNIFORM_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let material_bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_bg"),
            layout: &material_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                // Bind a single-stride window (256 bytes) into the UBO. The
                // dynamic offset at `set_bind_group` time shifts this window
                // across the 16 KB buffer. Binding the entire buffer with
                // `as_entire_binding()` would leave zero room for offsets.
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &material_ubo,
                    offset: 0,
                    size: Some(wgpu::BufferSize::new(MATERIAL_UNIFORM_STRIDE).unwrap()),
                }),
            }],
        });

        // --- Model uniform bind group (dynamic offset, replaces push constants) ---
        // DX12 does not support push constants in wgpu 22, so we use a dynamic-
        // offset uniform buffer. Each draw call writes its model matrix at
        // offset = i * MODEL_UNIFORM_STRIDE.
        let model_bgl =
            self.ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("model_bgl"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let model_ubo = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("model_ubo"),
            size: MAX_DRAW_CALLS as u64 * MODEL_UNIFORM_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let model_bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model_bg"),
            layout: &model_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                // Same pattern as material_bg: bind a 256-byte window so the
                // dynamic offset can shift it across the 16 KB buffer.
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &model_ubo,
                    offset: 0,
                    size: Some(wgpu::BufferSize::new(MODEL_UNIFORM_STRIDE).unwrap()),
                }),
            }],
        });

        // --- Main pipeline layout (3 bind groups: scene + model + material) ---
        let pipeline_layout =
            self.ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("main_pipeline_layout"),
                    bind_group_layouts: &[&scene_bgl, &model_bgl, &material_bgl],
                    push_constant_ranges: &[],
                });

        let pipeline = self
            .ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("main_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &main_shader,
                    entry_point: "vs_main",
                    buffers: &[Vertex::LAYOUT],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &main_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: self.depth_format,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        // --- Shadow pipeline (vertex-only, scene + model bind groups) ---
        let shadow_pipeline_layout = self
            .ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_pipeline_layout"),
                bind_group_layouts: &[&scene_bgl, &model_bgl],
                push_constant_ranges: &[],
            });
        let shadow_pipeline = self
            .ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shadow_pipeline"),
                layout: Some(&shadow_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shadow_shader,
                    entry_point: "vs_shadow",
                    buffers: &[Vertex::LAYOUT],
                    compilation_options: Default::default(),
                },
                fragment: None,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: self.shadow_format,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 2,
                        slope_scale: 2.0,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        self.pipeline = Some(pipeline);
        self.shadow_pipeline = Some(shadow_pipeline);
        self.scene_uniform_buffer = Some(scene_ubo);
        self.scene_bind_group = Some(scene_bg);
        self.scene_bind_group_layout = Some(scene_bgl);
        self.material_uniform_buffer = Some(material_ubo);
        self.material_bind_group = Some(material_bg);
        self.material_bind_group_layout = Some(material_bgl);
        self.model_uniform_buffer = Some(model_ubo);
        self.model_bind_group = Some(model_bg);
        self.model_bind_group_layout = Some(model_bgl);
        self.shadow_texture = Some(shadow_texture);
        self.shadow_view = Some(shadow_view);
        self.shadow_sampler = Some(shadow_sampler);
        Ok(())
    }

    /// Recreate the swapchain when the window resizes.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        // Borrow the device through the context — split the borrow to avoid
        // simultaneous `&self.config` and `&mut self.ctx` conflicts.
        let config = self.config.clone();
        self.ctx.surface.configure(&self.ctx.device, &config);
    }

    /// Render one frame: shadow pass → main pass.
    pub fn render(
        &mut self,
        camera: &Camera,
        lights: &LightUniform,
        draw_calls: &[DrawCall],
    ) -> Result<(), SurfaceError> {
        // --- Upload scene uniforms (camera + lights + shadow matrix) ---
        let scene_uniform = SceneUniform {
            view_proj: camera.view_proj().to_cols_array_2d(),
            camera_pos: [camera.eye.x, camera.eye.y, camera.eye.z, 0.0],
            ambient_color: lights.ambient_color,
            sun_direction: lights.sun_direction,
            sun_color: lights.sun_color,
            light_view_proj: lights.light_view_proj,
        };
        if let Some(buf) = &self.scene_uniform_buffer {
            self.ctx
                .queue
                .write_buffer(buf, 0, bytemuck::bytes_of(&scene_uniform));
        }

        // --- Upload model matrices + materials (one per draw call) ---
        // Both buffers use 256-byte strides to satisfy wgpu's
        // min_uniform_buffer_offset_alignment on every backend.
        let mut model_bytes = vec![0u8; MAX_DRAW_CALLS * MODEL_UNIFORM_STRIDE as usize];
        let mut material_bytes = vec![0u8; MAX_DRAW_CALLS * MATERIAL_UNIFORM_STRIDE as usize];
        let count = draw_calls.len().min(MAX_DRAW_CALLS);
        for (i, dc) in draw_calls.iter().enumerate().take(count) {
            let model_cols = dc.model.to_cols_array();
            let m = ModelUniform::from_cols_array(&model_cols);
            let model_off = i * MODEL_UNIFORM_STRIDE as usize;
            model_bytes[model_off..model_off + std::mem::size_of::<ModelUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&m));

            let mat = MaterialUniform::from(dc.material);
            let mat_off = i * MATERIAL_UNIFORM_STRIDE as usize;
            material_bytes[mat_off..mat_off + std::mem::size_of::<MaterialUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&mat));
        }
        if let Some(buf) = &self.model_uniform_buffer {
            self.ctx.queue.write_buffer(buf, 0, &model_bytes);
        }
        if let Some(buf) = &self.material_uniform_buffer {
            self.ctx.queue.write_buffer(buf, 0, &material_bytes);
        }

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("enginebeta_encoder"),
            });

        // --- Shadow pass: render depth from the light's POV ---
        if let (Some(shadow_pipeline), Some(shadow_view), Some(scene_bg), Some(model_bg)) = (
            &self.shadow_pipeline,
            &self.shadow_view,
            &self.scene_bind_group,
            &self.model_bind_group,
        ) {
            let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            shadow_pass.set_pipeline(shadow_pipeline);
            shadow_pass.set_bind_group(0, scene_bg, &[]);
            for (i, dc) in draw_calls.iter().enumerate().take(count) {
                let offset = (i as u32) * MODEL_UNIFORM_STRIDE as u32;
                shadow_pass.set_bind_group(1, model_bg, &[offset]);
                shadow_pass.set_vertex_buffer(0, dc.mesh.vertex_buffer.slice(..));
                shadow_pass.set_index_buffer(
                    dc.mesh.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                shadow_pass.draw_indexed(0..dc.mesh.index_count, 0, 0..1);
            }
        }

        // --- Main pass: render the scene with lighting + shadows ---
        let frame = self.ctx.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("main_depth"),
            size: wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.depth_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.clear_color.r as f64,
                            g: self.clear_color.g as f64,
                            b: self.clear_color.b as f64,
                            a: self.clear_color.a as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if let (Some(pipeline), Some(scene_bg), Some(model_bg), Some(mat_bg)) = (
                &self.pipeline,
                &self.scene_bind_group,
                &self.model_bind_group,
                &self.material_bind_group,
            ) {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, scene_bg, &[]);
                for (i, dc) in draw_calls.iter().enumerate().take(count) {
                    let model_offset = (i as u32) * MODEL_UNIFORM_STRIDE as u32;
                    let mat_offset = (i as u32) * MATERIAL_UNIFORM_STRIDE as u32;
                    pass.set_bind_group(1, model_bg, &[model_offset]);
                    pass.set_bind_group(2, mat_bg, &[mat_offset]);
                    pass.set_vertex_buffer(0, dc.mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(
                        dc.mesh.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint16,
                    );
                    pass.draw_indexed(0..dc.mesh.index_count, 0, 0..1);
                }
            }
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}

pub mod context;

/// GPU-side scene uniform. Mirrors `SceneUniform` in WGSL.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniform {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub ambient_color: [f32; 4],
    pub sun_direction: [f32; 4],
    pub sun_color: [f32; 4],
    pub light_view_proj: [[f32; 4]; 4],
}

/// GPU-side model matrix uniform. Stored at offset `i * 256` in the dynamic
/// uniform buffer. The 64-byte mat4 is followed by padding so each entry is
/// exactly 256 bytes (matches wgpu's `min_uniform_buffer_offset_alignment` on
/// every backend). The padding is expressed as a `[f32; 48]` array because
/// bytemuck can derive Pod/Zeroable for fixed-size f32 arrays (but not [u8; N>32]).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelUniform {
    pub model: [[f32; 4]; 4],
    pub _pad: [f32; 48], // 48 floats × 4 bytes = 192 bytes → total struct = 256 bytes
}

impl ModelUniform {
    pub fn from_cols_array(arr: &[f32; 16]) -> Self {
        let mut model = [[0f32; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                model[r][c] = arr[r * 4 + c];
            }
        }
        Self {
            model,
            _pad: [0.0; 48],
        }
    }
}

const MAIN_SHADER_SRC: &str = r#"
struct SceneUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    ambient_color: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    light_view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> scene: SceneUniform;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;

struct MaterialUniform {
    albedo_metallic: vec4<f32>,
    roughness_emissive: vec4<f32>,
};
@group(2) @binding(0) var<uniform> material: MaterialUniform;

// Per-draw-call model matrix — dynamic-offset uniform buffer.
// (Replaces `var<push_constant>` which DX12 doesn't support in wgpu 22.)
@group(1) @binding(0) var<uniform> model: mat4x4<f32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) albedo: vec3<f32>,
    @location(3) shadow_coord: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = model * vec4<f32>(in.position, 1.0);
    out.clip_position = scene.view_proj * world_pos;
    let world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
    out.world_normal = world_normal;
    out.world_pos = world_pos.xyz;
    out.albedo = in.color;
    let shadow = scene.light_view_proj * world_pos;
    out.shadow_coord = vec4<f32>(
        shadow.x * 0.5 + 0.5,
        shadow.y * 0.5 + 0.5,
        shadow.z,
        shadow.w,
    );
    return out;
}

fn sample_shadow(shadow_coord: vec4<f32>) -> f32 {
    let coord = shadow_coord.xyz / shadow_coord.w;
    if (coord.x < 0.0 || coord.x > 1.0 || coord.y < 0.0 || coord.y > 1.0) {
        return 1.0;
    }
    var visibility: f32 = 0.0;
    let texel_size = 1.0 / 2048.0;
    let bias = 0.003;
    for (var y: i32 = -1; y <= 1; y++) {
        for (var x: i32 = -1; x <= 1; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let s = textureSampleCompare(
                shadow_map, shadow_sampler,
                coord.xy + offset, coord.z - bias
            );
            visibility += s;
        }
    }
    return visibility / 9.0;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let L = normalize(-scene.sun_direction.xyz);
    let V = normalize(scene.camera_pos.xyz - in.world_pos);
    let H = normalize(L + V);

    let NdotL = max(dot(N, L), 0.0);
    let NdotH = max(dot(N, H), 0.0);
    let _NdotV = max(dot(N, V), 0.0);

    let shininess = mix(8.0, 256.0, 1.0 - material.roughness_emissive.x);
    let spec_strength = mix(0.1, 1.0, material.albedo_metallic.w);
    let spec = spec_strength * pow(NdotH, shininess) * NdotL;

    let shadow = sample_shadow(in.shadow_coord);
    let ambient = scene.ambient_color.rgb * scene.ambient_color.w * material.albedo_metallic.rgb;
    let diffuse = scene.sun_color.rgb * scene.sun_color.w * NdotL * shadow;
    let specular = scene.sun_color.rgb * scene.sun_color.w * spec * shadow;
    let emissive = material.roughness_emissive.yzw;

    let final_rgb = ambient + in.albedo * diffuse + specular + emissive;
    return vec4<f32>(final_rgb, 1.0);
}
"#;

const SHADOW_SHADER_SRC: &str = r#"
struct SceneUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    ambient_color: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    light_view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> scene: SceneUniform;

// Per-draw-call model matrix — dynamic-offset uniform buffer.
@group(1) @binding(0) var<uniform> model: mat4x4<f32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

@vertex
fn vs_shadow(in: VertexInput) -> @builtin(position) vec4<f32> {
    let world_pos = model * vec4<f32>(in.position, 1.0);
    return scene.light_view_proj * world_pos;
}
"#;
