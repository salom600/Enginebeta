//! engine-render — wgpu-based renderer for EngineBeta (v0.2.0).
//!
//! v0.2.0 additions:
//! - Directional + ambient lighting (Blinn-Phong)
//! - Shadow map for the directional light
//! - Per-mesh materials (albedo, metallic, roughness, emissive)
//! - Per-mesh model matrices (so each cube can have its own transform)
//! - Raycasting for mouse picking
//! - FlyCamera for first-person navigation

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
    shadow_texture: Option<wgpu::Texture>,
    shadow_view: Option<wgpu::TextureView>,
    shadow_sampler: Option<wgpu::Sampler>,
    shadow_bind_group: Option<wgpu::BindGroup>,
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
            shadow_texture: None,
            shadow_view: None,
            shadow_sampler: None,
            shadow_bind_group: None,
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

        // Shadow map texture (1024x1024 depth-only).
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
        // Stash 256 materials worth of uniforms (32 bytes each → 8 KiB).
        let material_ubo = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material_ubo"),
            size: 256 * std::mem::size_of::<MaterialUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let material_bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_bg"),
            layout: &material_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: material_ubo.as_entire_binding(),
            }],
        });

        // --- Main pipeline layout ---
        let pipeline_layout =
            self.ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("main_pipeline_layout"),
                    bind_group_layouts: &[&scene_bgl, &material_bgl],
                    push_constant_ranges: &[wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::VERTEX,
                        range: 0..64, // mat4 model matrix (64 bytes)
                    }],
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

        // --- Shadow pipeline (vertex-only, writes to shadow depth) ---
        let shadow_pipeline_layout = self
            .ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_pipeline_layout"),
                bind_group_layouts: &[&scene_bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::VERTEX,
                    range: 0..64,
                }],
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
        self.shadow_texture = Some(shadow_texture);
        self.shadow_view = Some(shadow_view);
        self.shadow_sampler = Some(shadow_sampler);
        // shadow pass uses the same scene bind group as the main pass; we just
        // reference it via `&scene_bind_group` at draw time, so we don't need a
        // separate field here. The field is kept for future per-light shadow maps.
        self.shadow_bind_group = None;
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

        // --- Upload materials (one per draw call, stashed in the dynamic buffer) ---
        let mat_uniform_size = std::mem::size_of::<MaterialUniform>() as u64;
        if let Some(buf) = &self.material_uniform_buffer {
            for (i, dc) in draw_calls.iter().enumerate() {
                if i >= 256 {
                    break;
                }
                let m = MaterialUniform::from(dc.material);
                self.ctx.queue.write_buffer(
                    buf,
                    i as u64 * mat_uniform_size,
                    bytemuck::bytes_of(&m),
                );
            }
        }

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("enginebeta_encoder"),
            });

        // --- Shadow pass: render depth from the light's POV ---
        if let (Some(shadow_pipeline), Some(shadow_view), Some(scene_bg)) = (
            &self.shadow_pipeline,
            &self.shadow_view,
            &self.scene_bind_group,
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
            for dc in draw_calls {
                let model_bytes = dc.model.to_cols_array();
                shadow_pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytemuck::cast_slice(&model_bytes));
                shadow_pass.set_vertex_buffer(0, dc.mesh.vertex_buffer.slice(..));
                shadow_pass.set_index_buffer(dc.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
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

            if let (Some(pipeline), Some(scene_bg), Some(mat_bg)) = (
                &self.pipeline,
                &self.scene_bind_group,
                &self.material_bind_group,
            ) {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, scene_bg, &[]);
                for (i, dc) in draw_calls.iter().enumerate() {
                    if i >= 256 {
                        break;
                    }
                    let model_bytes = dc.model.to_cols_array();
                    pass.set_push_constants(
                        wgpu::ShaderStages::VERTEX,
                        0,
                        bytemuck::cast_slice(&model_bytes),
                    );
                    let offset = i as u32 * mat_uniform_size as u32;
                    pass.set_bind_group(1, mat_bg, &[offset]);
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
@group(1) @binding(0) var<uniform> material: MaterialUniform;

var<push_constant> model: mat4x4<f32>;

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
    // Normal matrix (for uniform scale, model's upper 3x3 works)
    let world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
    out.world_normal = world_normal;
    out.world_pos = world_pos.xyz;
    out.albedo = in.color;
    // Shadow space: bias the depth into [0, 1] for the depth comparison.
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
    // Basic PCF 3x3 shadow sampling.
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
    let NdotV = max(dot(N, V), 0.0);

    // Specular: Blinn-Phong, modulated by material's metallic + roughness.
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

var<push_constant> model: mat4x4<f32>;

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
