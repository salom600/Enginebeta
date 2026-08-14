//! The hierarchical transform used by every renderable / physical entity.

use glam::{Mat4, Quat, Vec3};

/// Position / rotation / scale. Stored as separate fields (not a matrix) so
/// systems can manipulate each axis cheaply.
#[derive(Copy, Clone, Debug)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn from_position(p: Vec3) -> Self {
        Self {
            position: p,
            ..Default::default()
        }
    }

    pub fn from_pos_rot(p: Vec3, r: Quat) -> Self {
        Self {
            position: p,
            rotation: r,
            ..Default::default()
        }
    }

    /// Local-to-world matrix for this transform.
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_translation(self.position)
            * Mat4::from_quat(self.rotation)
            * Mat4::from_scale(self.scale)
    }

    /// Right vector (local +X in world space).
    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }
    /// Up vector (local +Y in world space).
    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }
    /// Forward vector (local -Z in world space, matches Unity/Godot convention).
    pub fn forward(&self) -> Vec3 {
        self.rotation * -Vec3::Z
    }

    pub fn translate(&mut self, v: Vec3) {
        self.position += v;
    }

    pub fn rotate_local_y(&mut self, angle_radians: f32) {
        self.rotation = Quat::from_rotation_y(angle_radians) * self.rotation;
    }
}
