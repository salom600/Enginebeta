//! Owning wrapper around wgpu device / queue / surface / adapter.
//!
//! The window is held as `Arc<Window>` so the surface can borrow it for `'static`.

use anyhow::Context as _;
use std::sync::Arc;
use winit::window::Window;

pub struct RenderContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    // Keep the Arc<Window> alive for the surface's lifetime.
    _window: Arc<Window>,
}

impl RenderContext {
    /// Create the wgpu device/queue/surface bound to `window`.
    ///
    /// The renderer holds an `Arc<Window>` internally so the surface can borrow
    /// the window for `'static`. Callers should pass a cloned `Arc`, not the
    /// original `Arc` they intend to keep.
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::default(),
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::from(window.clone()))
            .context("failed to create wgpu surface")?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("no suitable GPU adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("enginebeta_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .context("failed to acquire GPU device")?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface,
            _window: window,
        })
    }

    /// Convenience accessor: the inner window's inner size (for surface config).
    pub fn window_inner_size(&self) -> winit::dpi::PhysicalSize<u32> {
        self._window.inner_size()
    }
}
