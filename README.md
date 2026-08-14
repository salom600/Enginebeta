# EngineBeta

> A modern, lean Rust game engine for **Windows** and **Linux** — 7 core systems
> + lighting/shadows, AABB physics, AI perception, raycasting, and a fly camera.
> ~6k lines of Rust (vs. Godot's 500k+ lines of C++).

[![Build](https://github.com/salom600/Enginebeta/actions/workflows/ci.yml/badge.svg)](https://github.com/salom600/Enginebeta/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE-MIT)

EngineBeta is a minimalist, modular game engine written entirely in safe Rust.
The goal is to deliver the seven systems every game engine needs — rendering,
physics, scripting, audio, input, AI, and asset management — plus the v0.2.0
"modern engine feel" features (lighting, shadows, materials, AI perception,
raycasting, fly camera) without dragging in half a million lines of legacy code.

## The 7 Core Systems (v0.2.0)

| # | Crate | What's new in v0.2.0 |
|---|---|---|
| 1 | [`engine-render`](crates/engine-render) | **Directional + ambient light (Blinn-Phong), 2048² shadow map with PCF filtering, per-mesh materials (albedo/metallic/roughness/emissive), FlyCamera with yaw/pitch, raycasting (mouse → world)** |
| 2 | [`engine-physics`](crates/engine-physics) | **AABB-AABB and sphere-AABB collisions (in addition to sphere-sphere), force generators: Wind (with turbulence), Explosion (radial impulse with distance+time decay), PointGravity** |
| 3 | [`engine-script`](crates/engine-script) | (Unchanged from v0.1) Native Rust `Script` trait + JSON property bags |
| 4 | [`engine-audio`](crates/engine-audio) | (Unchanged) 2D + 3D positional via SpatialSink |
| 5 | [`engine-input`](crates/engine-input) | (Unchanged) Keyboard + mouse + gamepad |
| 6 | [`engine-ai`](crates/engine-ai) | **Vision sensor (FOV cone + line-of-sight + memory), Hearing sensor (sound radius with loudness falloff), smooth arrive/pursue/orbit steering, smooth_velocity (critically-damped spring)** |
| 7 | [`engine-assets`](crates/engine-assets) | (Unchanged) Ref-counted store + hot-reload + memory budget |

Plus [`engine-core`](crates/engine-core) — now with **geometric primitives**
(`Ray`, `Aabb`, `Plane` with ray-cast helpers) and a **rolling-average FpsCounter**.

## Demo Features (run `cargo run --release -p engine-launcher`)

- **Fly camera** — WASD to move, mouse to look, Space/Shift for vertical
- **Lighting** — directional sun + soft ambient fill, casting real-time shadows
- **Materials** — wood cubes, polished metal balls, plastic, emissive accents
- **Physics demo** — stacked cubes that tumble, falling balls that bounce,
  static pedestal that cubes stack on
- **Force generators** — press `E` to trigger an explosion at world origin
- **AI enemy** — a sphere with vision + hearing sensors that tracks the player
- **Profiler** — FPS + frame time + physics time logged to console every 2s

| Key | Action |
|---|---|
| `WASD` | Move fly camera |
| `Mouse` | Look around |
| `Space` | Drop a red plastic cube |
| `B` | Drop a blue plastic ball |
| `E` | Trigger explosion |
| `Shift+Space` | Vertical (up/down) |
| `ESC` | Quit |

## Architecture v0.2.0

### Renderer pipeline

```
   ┌──────────────────┐
   │  Shadow pass     │  Render scene depth from sun's POV (2048² depth texture)
   │  (no color)      │  → used by main pass for PCF shadow sampling
   └────────┬─────────┘
            │
   ┌────────▼─────────┐
   │  Main pass       │  For each draw call:
   │  (color + depth) │   1. Set push-constant model matrix (64 bytes)
   │                  │   2. Set dynamic-offset material uniform
   │                  │   3. Bind scene uniform (camera + lights + shadow matrix)
   │                  │   4. Draw indexed
   └──────────────────┘
```

The WGSL shader implements Blinn-Phong with metallic/roughness modulation:
`specular_strength = mix(0.1, 1.0, metallic)`, `shininess = mix(8, 256, 1 - roughness)`.
Shadows are sampled with a 3×3 PCF kernel for soft edges.

### Physics collision manifold

Three collision types, all using the same impulse-response formula:

```
            ┌─────────────────────┐
            │  step_gravity       │  Apply gravity to dynamic bodies
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  integrate          │  position += velocity * dt
            └──────────┬──────────┘
                       │
   ┌───────────────────┼───────────────────┐
   │                   │                   │
┌──▼─────┐       ┌─────▼──────┐      ┌─────▼──────┐
│ sphere │       │   AABB     │      │ sphere-AABB│
│ -sphere│       │   -AABB    │      │            │
└────────┘       └────────────┘      └────────────┘
   (impulse       (MTV on smallest     (closest-point-
    response)      overlap axis)        on-box test)
```

### AI perception system

Each NPC carries `VisionSensor` + `HearingSensor` components. Each frame:

1. Snapshot all `Perceivable` targets + active `SoundEvent`s
2. For each NPC, run `can_see()` — checks distance, FOV cone, optional LoS
3. If no visual, run `can_hear()` — checks `loudness * sensitivity >= distance`
4. On detection: add `Alerted { target, last_known_position, time_since_seen: 0 }`
5. On loss of sight: keep `Alerted` for `memory_duration` seconds, then drop
6. `LastKnownPosition` persists (for "search" behavior)

## Project Layout

```
enginebeta/
├── Cargo.toml
├── crates/
│   ├── engine-core/        # + geom.rs (Ray, Aabb, Plane), FpsCounter
│   ├── engine-render/      # + light.rs, material.rs, raycast.rs, FlyCamera, shadows
│   ├── engine-physics/     # + forces.rs (Wind, Explosion, PointGravity), AABB collisions
│   ├── engine-script/
│   ├── engine-audio/
│   ├── engine-input/
│   ├── engine-ai/          # + perception.rs (vision/hearing), arrive/pursue/orbit
│   ├── engine-assets/
│   └── engine-launcher/    # fly camera + profiler + upgraded demo scene
└── .github/workflows/ci.yml
```

## Build

### Prerequisites

- **Rust** stable (1.86+ due to `wayland-protocols` MSRV) — https://rustup.rs
- **Linux**: `apt install libx11-dev libxkbcommon-dev libwayland-dev libasound2-dev libudev-dev libxcb1-dev libxrandr-dev libxi-dev libgl1-mesa-dev libegl-dev libxcb-randr0-dev libxcb-xinput-dev libxcb-xkb-dev libxcb-shape0-dev libxcb-icccm4-dev libxcb-keysyms1-dev libx11-xcb-dev libxcursor-dev libxinerama-dev`
- **Windows**: Visual Studio Build Tools 2022 (MSVC)

### Compile

```bash
cargo build --release -p engine-launcher
# Binary: target/release/enginebeta (Linux) or target/release/enginebeta.exe (Windows)
```

### Run tests

```bash
cargo test --workspace
# 33 tests: 13 AI (pathfinding, perception, steering), 9 physics (collisions, forces),
#           5 ECS, 4 geom (ray-AABB, ray-plane), 2 raycast (ray-sphere)
```

## CI / CD

Every push and PR triggers a build on both Ubuntu 22.04 and Windows Server 2022.
Binaries are uploaded as workflow artifacts (90-day retention) and, on `v*` tags,
attached to a GitHub Release.

**Build status:** https://github.com/salom600/Enginebeta/actions

## License

Dual-licensed under MIT or Apache-2.0, at your option.
