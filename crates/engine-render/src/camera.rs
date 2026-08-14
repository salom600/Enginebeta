//! Perspective camera (for the main view) and orthographic camera (for shadow mapping).

use glam::{Mat4, Vec3};

/// A perspective camera with eye / target / up and an adjustable FOV.
#[derive(Copy, Clone, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub aspect: f32,
    pub fovy: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            eye: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            aspect: 16.0 / 9.0,
            fovy: 45.0f32.to_radians(),
            near: 0.1,
            far: 100.0,
        }
    }
}

impl Camera {
    pub fn new(eye: Vec3, target: Vec3, aspect: f32) -> Self {
        Self {
            eye,
            target,
            aspect,
            ..Default::default()
        }
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.target, self.up)
    }

    pub fn projection(&self) -> Mat4 {
        Mat4::perspective_rh(self.fovy, self.aspect, self.near, self.far)
    }

    pub fn view_proj(&self) -> Mat4 {
        self.projection() * self.view()
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if height > 0 {
            self.aspect = width as f32 / height as f32;
        }
    }

    /// Forward vector (direction the camera is looking, -Z in camera space).
    pub fn forward(&self) -> Vec3 {
        (self.target - self.eye).normalize_or_zero()
    }

    /// Right vector (perpendicular to forward and up).
    pub fn right(&self) -> Vec3 {
        self.forward().cross(self.up).normalize_or_zero()
    }
}

/// A fly camera — first-person style controlled by yaw/pitch and a position.
/// WASD moves on the camera's forward/right plane; mouse sets yaw/pitch.
#[derive(Copy, Clone, Debug)]
pub struct FlyCamera {
    pub position: Vec3,
    pub yaw: f32,   // rotation around Y, radians
    pub pitch: f32, // rotation around X, radians
    pub aspect: f32,
    pub fovy: f32,
    pub near: f32,
    pub far: f32,
    /// Move speed in m/s (multiplied by delta time).
    pub move_speed: f32,
    /// Mouse sensitivity (radians per pixel of mouse delta).
    pub look_sensitivity: f32,
}

impl Default for FlyCamera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 2.0, 8.0),
            yaw: -std::f32::consts::FRAC_PI_2, // looking towards -Z initially
            pitch: -0.15,
            aspect: 16.0 / 9.0,
            fovy: 60.0f32.to_radians(),
            near: 0.1,
            far: 200.0,
            move_speed: 6.0,
            look_sensitivity: 0.0025,
        }
    }
}

impl FlyCamera {
    pub fn forward(&self) -> Vec3 {
        let cos_p = self.pitch.cos();
        Vec3::new(
            self.yaw.sin() * cos_p,
            self.pitch.sin(),
            -self.yaw.cos() * cos_p, // -Z is forward
        )
        .normalize()
    }

    pub fn right(&self) -> Vec3 {
        // Right = forward × world_up (with the Y component zeroed to keep movement horizontal).
        let f = self.forward();
        Vec3::new(f.z, 0.0, -f.x).normalize()
    }

    /// Forward projected onto the XZ plane — used for WASD movement so the
    /// camera doesn't fly up/down when looking up.
    pub fn forward_flat(&self) -> Vec3 {
        let f = self.forward();
        Vec3::new(f.x, 0.0, f.z).normalize_or_zero()
    }

    pub fn target(&self) -> Vec3 {
        self.position + self.forward()
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target(), Vec3::Y)
    }

    pub fn projection(&self) -> Mat4 {
        Mat4::perspective_rh(self.fovy, self.aspect, self.near, self.far)
    }

    pub fn view_proj(&self) -> Mat4 {
        self.projection() * self.view()
    }

    /// Apply mouse-look delta (in pixels).
    pub fn look(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * self.look_sensitivity;
        self.pitch -= dy * self.look_sensitivity;
        // Clamp pitch to avoid flipping.
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-limit, limit);
    }

    /// Move along the horizontal forward/right plane (WASD-style).
    pub fn move_flat(&mut self, forward: f32, right: f32, dt: f32) {
        let f = self.forward_flat() * forward;
        let r = self.right() * right;
        self.position += (f + r) * self.move_speed * dt;
    }

    /// Move vertically (space / shift for up/down).
    pub fn move_vertical(&mut self, amount: f32, dt: f32) {
        self.position.y += amount * self.move_speed * dt;
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if height > 0 {
            self.aspect = width as f32 / height as f32;
        }
    }

    /// Convert to a [`Camera`] (for code that expects the legacy API).
    pub fn as_camera(&self) -> Camera {
        Camera {
            eye: self.position,
            target: self.target(),
            up: Vec3::Y,
            aspect: self.aspect,
            fovy: self.fovy,
            near: self.near,
            far: self.far,
        }
    }
}

/// Orthographic camera used for shadow-map rendering of the directional light.
#[derive(Copy, Clone, Debug)]
pub struct OrthoCamera {
    pub center: Vec3,
    pub size: f32, // half-extent of the orthographic box on each axis
    pub near: f32,
    pub far: f32,
    pub direction: Vec3,
}

impl OrthoCamera {
    /// Build an ortho camera that looks from `direction` back at `center`.
    pub fn new(center: Vec3, size: f32, direction: Vec3) -> Self {
        Self {
            center,
            size,
            near: -50.0,
            far: 50.0,
            direction: direction.normalize(),
        }
    }

    pub fn view(&self) -> Mat4 {
        let eye = self.center - self.direction * 20.0;
        Mat4::look_at_rh(eye, self.center, Vec3::Y)
    }

    pub fn projection(&self) -> Mat4 {
        let s = self.size;
        Mat4::orthographic_rh(-s, s, -s, s, self.near, self.far)
    }

    pub fn view_proj(&self) -> Mat4 {
        self.projection() * self.view()
    }
}
