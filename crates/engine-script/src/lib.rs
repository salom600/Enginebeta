//! engine-script — native Rust scripting for EngineBeta.
//!
//! Instead of embedding a VM (Lua, Rhai, etc.) we expose a small trait
//! [`Script`] that gameplay code implements in Rust. This keeps the loop tight:
//! no FFI, no GC pauses, no parse step. Scripts are registered with a
//! [`ScriptRegistry`] and run as ordinary ECS systems.
//!
//! For data-driven behavior (designers tweaking values without recompiling),
//! see [`ScriptData`] — a JSON-deserializable property bag.

use engine_core::{Entity, Time, World};

pub mod registry;

pub use registry::ScriptRegistry;

/// A script instance. One per entity that uses it.
pub trait Script: Send + Sync {
    /// Called once when the script is attached to an entity.
    fn on_start(&mut self, _world: &mut World, _entity: Entity) {}
    /// Called every frame on the variable update stage.
    fn on_update(&mut self, _world: &mut World, _entity: Entity, _time: &Time) {}
    /// Called at the fixed step.
    fn on_fixed_update(&mut self, _world: &mut World, _entity: Entity, _time: &Time) {}
    /// Called once when the script is removed or the entity is despawned.
    fn on_stop(&mut self, _world: &mut World, _entity: Entity) {}
    /// Display name (useful for tooling).
    fn name(&self) -> &str {
        "script"
    }
}

/// A script component: holds one boxed [`Script`] instance plus its entity id.
pub struct ScriptInstance {
    pub entity: Entity,
    pub inner: Box<dyn Script>,
    pub started: bool,
}

impl ScriptInstance {
    pub fn new(entity: Entity, script: Box<dyn Script>) -> Self {
        Self {
            entity,
            inner: script,
            started: false,
        }
    }
}

/// System: tick all script instances. Insert into [`engine_core::Stage::Update`]
/// for variable update, or [`engine_core::Stage::FixedUpdate`] for fixed step.
///
/// Implementation note: scripts can mutate arbitrary components in the world,
/// so we can't hold a borrow on the `ScriptInstance` column while running them.
/// We snapshot the list of script-bearing entity ids, then take each instance
/// out, run it, and put it back.
pub fn run_scripts(stage: engine_core::Stage) -> impl FnMut(&mut World, &Time) + Send + 'static {
    move |world, time| {
        // Snapshot which entities have scripts.
        let ids: Vec<u32> = world
            .column_write::<ScriptInstance>()
            .iter()
            .map(|(id, _)| id)
            .collect();

        for id in ids {
            // Take the instance out (releases the &mut World borrow).
            let mut inst_opt: Option<ScriptInstance> =
                world.column_write::<ScriptInstance>().remove(id);
            if let Some(mut inst) = inst_opt.take() {
                if !inst.started {
                    inst.inner.on_start(world, inst.entity);
                    inst.started = true;
                }
                match stage {
                    engine_core::Stage::Update
                    | engine_core::Stage::PreUpdate
                    | engine_core::Stage::PostUpdate => {
                        inst.inner.on_update(world, inst.entity, time);
                    }
                    engine_core::Stage::FixedUpdate => {
                        inst.inner.on_fixed_update(world, inst.entity, time);
                    }
                    engine_core::Stage::Startup => {}
                }
                // Put it back.
                world.column_write::<ScriptInstance>().insert(id, inst);
            }
        }
    }
}

/// Data-driven property bag — gameplay designers can ship these as JSON files
/// and the script reads them at startup.
#[derive(Debug, Clone, Default)]
pub struct ScriptData {
    pub fields: std::collections::HashMap<String, ScriptValue>,
}

#[derive(Debug, Clone)]
pub enum ScriptValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Vec3([f32; 3]),
}

impl ScriptData {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        let fields: std::collections::HashMap<String, ScriptValue> = serde_json::from_str(s)?;
        Ok(Self { fields })
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.fields).unwrap_or_default()
    }
}

impl<'de> serde::Deserialize<'de> for ScriptValue {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let v = serde_json::Value::deserialize(d)?;
        match v {
            serde_json::Value::Bool(b) => Ok(ScriptValue::Bool(b)),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(ScriptValue::Int(i))
                } else {
                    Ok(ScriptValue::Float(n.as_f64().unwrap_or(0.0)))
                }
            }
            serde_json::Value::String(s) => Ok(ScriptValue::String(s)),
            serde_json::Value::Array(arr) if arr.len() == 3 => {
                let mut out = [0.0f32; 3];
                for (i, e) in arr.into_iter().enumerate() {
                    out[i] = e.as_f64().unwrap_or(0.0) as f32;
                }
                Ok(ScriptValue::Vec3(out))
            }
            _ => Err(Error::custom("unsupported ScriptValue")),
        }
    }
}

impl serde::Serialize for ScriptValue {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        match self {
            ScriptValue::Bool(b) => s.serialize_bool(*b),
            ScriptValue::Int(i) => s.serialize_i64(*i),
            ScriptValue::Float(f) => s.serialize_f64(*f),
            ScriptValue::String(st) => s.serialize_str(st),
            ScriptValue::Vec3(v) => {
                let mut m = s.serialize_map(Some(3))?;
                m.serialize_entry("x", &v[0])?;
                m.serialize_entry("y", &v[1])?;
                m.serialize_entry("z", &v[2])?;
                m.end()
            }
        }
    }
}

/// A script factory — used by [`ScriptRegistry`] to instantiate scripts by name.
pub struct ScriptFactory {
    pub build: Box<dyn Fn() -> Box<dyn Script> + Send + Sync>,
}

impl std::fmt::Debug for ScriptFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptFactory").finish()
    }
}
