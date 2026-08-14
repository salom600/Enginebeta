//! engine-render — wgpu-based renderer for EngineBeta.
//!
//! Provides:
//! - [`RenderContext`] — owns the wgpu device/queue/surface
//! - [`Renderer`] — pipeline + frame orchestration
//! - [`Camera`] — perspective camera with eye / target / up
//! - [`Mesh`] — vertex+index buffer bundle
//! - [`MeshData`], [`Vertex`] — CPU-side geometry uploads
//!
//! The renderer is intentionally minimal: clear the surface to a color, draw a
//! single pipeline of unlit colored triangles. That keeps the MVP small while
//! still demonstrating the full device → surface → pipeline → submit loop.

pub mod camera;
pub mod context;
pub mod mesh;
pub mod pipeline;
pub mod vertex;

pub use camera::Camera;
pub use context::RenderContext;
pub use mesh::{Mesh, MeshData};
pub use pipeline::RenderPipelineHandle;
pub use vertex::Vertex;

use engine_core::Color;
use wgpu::SurfaceError;

/// Top-level renderer. Owns the pipeline and one frame's worth of state.
pub struct Renderer {
    pub ctx: RenderContext,
    pub clear_color: Color,
    pub depth_format: wgpu::TextureFormat,
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    uniform_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    /// Surface configuration (width/height/format). Public so the launcher
    /// can read width/height for camera aspect.
    pub config: wgpu::SurfaceConfiguration,
}

impl Renderer {
    /// Create a renderer bound to a winit window. Configures the surface to the
    /// window's current size; call [`Renderer::resize`] when the window changes.
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
            pipeline: None,
            bind_group_layout: None,
            uniform_buffer: None,
            bind_group: None,
            config,
        })
    }

    /// Build (or rebuild) the render pipeline. Call after `new` and after any
    /// shader change. Returns a handle so future work can swap pipelines.
    pub fn build_pipeline(&mut self) -> anyhow::Result<()> {
        let shader = self.ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("enginebeta_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let bind_group_layout =
            self.ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("enginebeta_bgl"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let pipeline_layout =
            self.ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("enginebeta_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = self
            .ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("enginebeta_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[Vertex::LAYOUT],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
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

        let uniform_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("enginebeta_uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("enginebeta_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        self.pipeline = Some(pipeline);
        self.bind_group_layout = Some(bind_group_layout);
        self.uniform_buffer = Some(uniform_buffer);
        self.bind_group = Some(bind_group);
        Ok(())
    }

    /// Recreate the swapchain when the window resizes.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.ctx.surface.configure(&self.ctx.device, &self.config);
    }

    /// Render one frame: upload camera, clear, draw all submitted meshes.
    pub fn render(&mut self, camera: &Camera, meshes: &[&Mesh]) -> Result<(), SurfaceError> {
        let frame = self.ctx.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let cam_uniform = CameraUniform::from(camera);
        if let Some(buf) = &self.uniform_buffer {
            self.ctx.queue.write_buffer(buf, 0, bytemuck::bytes_of(&cam_uniform));
        }

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("enginebeta_encoder"),
            });

        // Depth texture — recreated every frame for simplicity in MVP.
        let depth_texture = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("enginebeta_depth"),
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
                label: Some("enginebeta_pass"),
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

            if let (Some(pipeline), Some(bg)) = (&self.pipeline, &self.bind_group) {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bg, &[]);
                for mesh in meshes {
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}

const SHADER_SRC: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl From<&Camera> for CameraUniform {
    fn from(c: &Camera) -> Self {
        Self {
            view_proj: c.view_proj().to_cols_array_2d(),
        }
    }
}
