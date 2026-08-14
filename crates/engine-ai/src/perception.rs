//! Enemy perception — vision (FOV cone + line of sight) and hearing (sound radius).
//!
//! The system is **event-driven + polling-friendly**:
//! - `VisionSensor` + `HearingSensor` components live on NPC entities.
//! - Each frame, `perception_system` checks every NPC against every entity
//!   tagged with `Perceivable` (typically the player).
//! - When an NPC notices a target, it gains an `Alerted` component; when it
//!   loses sight, `Alerted` is removed and `LastKnownPosition` is updated.

use engine_core::{Entity, World};
use glam::Vec3;

/// Configuration for an NPC's vision: a forward-facing cone.
#[derive(Copy, Clone, Debug)]
pub struct VisionSensor {
    /// Field of view in radians (half-angle on each side, so total FOV = 2 * this).
    pub half_fov: f32,
    /// Maximum view distance.
    pub range: f32,
    /// Whether line-of-sight is required (i.e. blocked by obstacles).
    pub require_los: bool,
    /// How long the NPC keeps memory of a seen target after losing sight (seconds).
    pub memory_duration: f32,
}

impl Default for VisionSensor {
    fn default() -> Self {
        Self {
            half_fov: 45.0f32.to_radians(), // 90° total
            range: 20.0,
            require_los: true,
            memory_duration: 3.0,
        }
    }
}

/// Configuration for an NPC's hearing: a sphere of radius `range`.
/// Louder sounds (higher `loudness`) propagate further.
#[derive(Copy, Clone, Debug)]
pub struct HearingSensor {
    /// Hearing sensitivity — sounds with `loudness * sensitivity >= distance`
    /// are detected.
    pub sensitivity: f32,
    /// Maximum hearing distance (hard cap).
    pub range: f32,
}

impl Default for HearingSensor {
    fn default() -> Self {
        Self {
            sensitivity: 1.0,
            range: 15.0,
        }
    }
}

/// Tag placed on entities that can be perceived (player, projectiles, noisy props).
#[derive(Copy, Clone, Debug, Default)]
pub struct Perceivable;

/// A sound event emitted at a world position. The perception system consumes
/// these each frame. `loudness` is in arbitrary units; `range = loudness * sensitivity`.
#[derive(Copy, Clone, Debug)]
pub struct SoundEvent {
    pub position: Vec3,
    pub loudness: f32,
    /// Tag so listeners can filter (e.g. "footstep", "gunshot").
    pub kind: u16,
}

/// Component added to an NPC when it currently perceives a target.
#[derive(Copy, Clone, Debug)]
pub struct Alerted {
    pub target: Entity,
    pub last_known_position: Vec3,
    pub time_since_seen: f32,
}

/// Component storing the last known position of a target (persists after losing sight).
#[derive(Copy, Clone, Debug, Default)]
pub struct LastKnownPosition {
    pub position: Vec3,
    pub target_id: u32,
    pub age: f32,
}

/// Check whether `npc_pos` (looking in `npc_forward` direction) can see
/// `target_pos`. Returns `true` if the target is within range and inside the FOV cone.
///
/// For line-of-sight, supply a `los_check` closure that returns `true` if the
/// ray from `npc_pos` to `target_pos` is unobstructed. Pass `None` to skip LoS.
pub fn can_see(
    npc_pos: Vec3,
    npc_forward: Vec3,
    target_pos: Vec3,
    sensor: &VisionSensor,
    los_check: Option<&dyn Fn(Vec3, Vec3) -> bool>,
) -> bool {
    let to_target = target_pos - npc_pos;
    let dist = to_target.length();
    if dist > sensor.range || dist < 1e-3 {
        return false;
    }
    let dir = to_target / dist;
    let cos_half_fov = sensor.half_fov.cos();
    let cos_angle = dir.dot(npc_forward.normalize_or_zero());
    if cos_angle < cos_half_fov {
        return false;
    }
    if sensor.require_los {
        if let Some(check) = los_check {
            if !check(npc_pos, target_pos) {
                return false;
            }
        }
    }
    true
}

/// Check whether `npc_pos` can hear a sound at `sound_pos` with the given `loudness`.
pub fn can_hear(
    npc_pos: Vec3,
    sound_pos: Vec3,
    loudness: f32,
    sensor: &HearingSensor,
) -> bool {
    let dist = (sound_pos - npc_pos).length();
    if dist > sensor.range {
        return false;
    }
    // A sound of loudness L can be heard up to L * sensitivity meters away.
    dist <= loudness * sensor.sensitivity
}

/// One-shot perception system. Iterates every NPC with a VisionSensor (or
/// HearingSensor), checks against every Perceivable target + SoundEvent, and
/// updates Alerted / LastKnownPosition components accordingly.
///
/// `los_check` is an optional closure that tests whether the line between two
/// points is unobstructed. The closure is called for vision checks only.
pub fn perception_system(
    world: &mut World,
    dt: f32,
    los_check: Option<&dyn Fn(Vec3, Vec3) -> bool>,
) {
    // Snapshot NPCs (id, position, forward, vision, hearing).
    #[derive(Copy, Clone)]
    struct NpcSnap {
        id: u32,
        pos: Vec3,
        forward: Vec3,
        vision: VisionSensor,
        hearing: HearingSensor,
    }
    let npcs: Vec<NpcSnap> = {
        let mut out = Vec::new();
        world.columns3::<VisionSensor, HearingSensor, engine_core::Transform, _, _>(
            |visions, hearings, transforms| {
                for (id, v) in visions.iter() {
                    let pos = transforms.get(id).map(|t| t.position).unwrap_or(Vec3::ZERO);
                    let forward = transforms.get(id).map(|t| t.forward()).unwrap_or(Vec3::Z);
                    let h = hearings.get(id).copied().unwrap_or_default();
                    out.push(NpcSnap {
                        id,
                        pos,
                        forward,
                        vision: *v,
                        hearing: h,
                    });
                }
            },
        );
        out
    };

    // Snapshot perceivable targets.
    let targets: Vec<(u32, Vec3)> = {
        let mut out = Vec::new();
        world.columns2::<Perceivable, engine_core::Transform, _, _>(|perceivable, transforms| {
            for (id, _) in perceivable.iter() {
                if let Some(t) = transforms.get(id) {
                    out.push((id, t.position));
                }
            }
        });
        out
    };

    // Snapshot sound events.
    let sounds: Vec<SoundEvent> = world
        .column_write::<SoundEvent>()
        .iter()
        .map(|(_, s)| *s)
        .collect();
    // Clear sound events (they're consumed each frame).
    {
        let mut col = world.column_write::<SoundEvent>();
        let ids: Vec<u32> = col.iter().map(|(id, _)| id).collect();
        for id in ids {
            col.remove(id);
        }
    }

    // For each NPC, check vision + hearing against every target + sound.
    let mut alerts: Vec<(u32, Entity, Vec3)> = Vec::new(); // (npc_id, target, last_known_pos)
    let mut last_known_updates: Vec<(u32, Entity, Vec3)> = Vec::new();
    for npc in &npcs {
        let mut saw_target: Option<(Entity, Vec3)> = None;
        let mut heard_target_pos: Option<Vec3> = None;
        for (tid, tpos) in &targets {
            if *tid == npc.id {
                continue;
            }
            if can_see(npc.pos, npc.forward, *tpos, &npc.vision, los_check) {
                saw_target = Some((Entity { id: *tid, gen: 0 }, *tpos));
                break;
            }
        }
        if saw_target.is_none() {
            for s in &sounds {
                if can_hear(npc.pos, s.position, s.loudness, &npc.hearing) {
                    heard_target_pos = Some(s.position);
                    break;
                }
            }
        }
        if let Some((target, pos)) = saw_target {
            alerts.push((npc.id, target, pos));
            last_known_updates.push((npc.id, target, pos));
        } else if let Some(pos) = heard_target_pos {
            // Hearing alone: NPC is alerted but at the sound's position (no entity).
            last_known_updates.push((npc.id, Entity { id: u32::MAX, gen: 0 }, pos));
        }
    }

    // Apply: add/update Alerted components, increment time_since_seen for NPCs
    // that didn't see anything this frame, drop Alerted after memory_duration.
    // First, snapshot existing alerts so we can update them.
    let existing_alerts: Vec<(u32, Alerted)> = world
        .column_write::<Alerted>()
        .iter()
        .map(|(id, a)| (id, *a))
        .collect();
    for (id, mut alert) in existing_alerts {
        // Did we see the target this frame?
        let still_seen = alerts.iter().any(|(nid, _, _)| *nid == id);
        if still_seen {
            // Will be overwritten below — skip.
            continue;
        } else {
            alert.time_since_seen += dt;
            // Look up this NPC's vision memory_duration.
            let memory = world
                .column_read::<VisionSensor>()
                .and_then(|c| c.get(id).map(|v| v.memory_duration))
                .unwrap_or(3.0);
            if alert.time_since_seen > memory {
                // Drop Alerted, keep LastKnownPosition.
                world.column_write::<Alerted>().remove(id);
            } else {
                // Re-insert updated alert.
                world.column_write::<Alerted>().insert(id, alert);
            }
        }
    }
    // Insert fresh alerts for NPCs that saw something this frame.
    for (npc_id, target, pos) in &alerts {
        let alert = Alerted {
            target: *target,
            last_known_position: *pos,
            time_since_seen: 0.0,
        };
        world.column_write::<Alerted>().insert(*npc_id, alert);
    }
    // Update LastKnownPosition components.
    for (npc_id, target, pos) in &last_known_updates {
        world.column_write::<LastKnownPosition>().insert(
            *npc_id,
            LastKnownPosition {
                position: *pos,
                target_id: target.id,
                age: 0.0,
            },
        );
    }
    // Age existing LastKnownPosition entries.
    let lkp_ids: Vec<u32> = world
        .column_write::<LastKnownPosition>()
        .iter()
        .map(|(id, _)| id)
        .collect();
    for id in lkp_ids {
        if let Some(lkp) = world.column_write::<LastKnownPosition>().get_mut(id) {
            lkp.age += dt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_sees_target_in_cone() {
        let sensor = VisionSensor::default();
        // NPC at origin looking +Z, target 5m ahead.
        assert!(can_see(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 5.0),
            &sensor,
            None,
        ));
    }

    #[test]
    fn vision_misses_target_outside_fov() {
        let sensor = VisionSensor::default();
        // Target directly behind NPC.
        assert!(!can_see(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -5.0),
            &sensor,
            None,
        ));
    }

    #[test]
    fn vision_misses_target_outside_range() {
        let sensor = VisionSensor {
            range: 5.0,
            ..Default::default()
        };
        assert!(!can_see(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 10.0),
            &sensor,
            None,
        ));
    }

    #[test]
    fn vision_respects_los() {
        let sensor = VisionSensor {
            require_los: true,
            ..Default::default()
        };
        // LoS check returns false → can't see.
        let result = can_see(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 5.0),
            &sensor,
            Some(&|_a, _b| false),
        );
        assert!(!result);
    }

    #[test]
    fn hearing_detects_nearby_sound() {
        let sensor = HearingSensor::default();
        assert!(can_hear(
            Vec3::ZERO,
            Vec3::new(3.0, 0.0, 0.0),
            5.0,
            &sensor,
        ));
    }

    #[test]
    fn hearing_misses_distant_sound() {
        let sensor = HearingSensor::default();
        assert!(!can_hear(
            Vec3::ZERO,
            Vec3::new(20.0, 0.0, 0.0),
            5.0,
            &sensor,
        ));
    }
}
