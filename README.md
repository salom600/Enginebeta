# EngineBeta

> A modern, lean Rust game engine for **Windows** and **Linux** — all 7 core
> systems implemented in ~4.5k lines of Rust (vs. Godot's 500k+ lines of C++).

[![Build](https://github.com/salom600/Enginebeta/actions/workflows/ci.yml/badge.svg)](https://github.com/salom600/Enginebeta/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE-MIT)

EngineBeta is a minimalist, modular game engine written entirely in safe Rust.
The goal is to deliver the seven systems every game engine needs — rendering,
physics, scripting, audio, input, AI, and asset management — without dragging
in half a million lines of legacy code. Rust's ownership model lets us build a
fast, cache-friendly ECS without GC pauses; `wgpu` gives us a single, modern
graphics backend that runs on Vulkan, DX12, and Metal.

## Why Rust?

| | Godot (C++) | EngineBeta (Rust) |
|---|---|---|
| Codebase size | ~500k LOC | ~4.5k LOC |
| Memory safety | Manual / RAII | Compiler-enforced (no UB) |
| Concurrency | Hard (data races possible) | Fearless (Send/Sync + Arc) |
| Cross-platform build | Custom SCons | `cargo build --release` |
| Package ecosystem | vcpkg / manual | crates.io |

Rust lets us say "half the code" without lying — not because we cut features,
but because the type system replaces whole categories of defensive code (smart
pointers, ref-counting, lifetime tracking) that C++ engines have to spell out
by hand.

## The 7 Core Systems

| # | Crate | Responsibility |
|---|---|---|
| 1 | [`engine-render`](crates/engine-render) | GPU rendering via `wgpu` — surface, pipeline, mesh, camera |
| 2 | [`engine-physics`](crates/engine-physics) | Rigid bodies, gravity, sphere-sphere collisions |
| 3 | [`engine-script`](crates/engine-script) | Native Rust scripts via `Script` trait + data-driven property bags |
| 4 | [`engine-audio`](crates/engine-audio) | 2D + 3D positional audio via `rodio` |
| 5 | [`engine-input`](crates/engine-input) | Keyboard, mouse, gamepad via `winit` + `gilrs` |
| 6 | [`engine-ai`](crates/engine-ai) | A* pathfinding, behavior trees, steering behaviors |
| 7 | [`engine-assets`](crates/engine-assets) | Asset store with ref-counting, hot-reload, memory budget |

Plus [`engine-core`](crates/engine-core) — the foundation (App loop, ECS,
Transform, Color, Time) that every other crate builds on.

## Project Layout

```
enginebeta/
├── Cargo.toml              # workspace manifest
├── crates/
│   ├── engine-core/        # App loop, ECS, math, time, transforms
│   ├── engine-render/      # wgpu renderer
│   ├── engine-physics/     # rigid bodies + collisions
│   ├── engine-script/      # Rust-native scripting
│   ├── engine-audio/       # rodio 2D + SpatialSink 3D
│   ├── engine-input/       # keyboard, mouse, gamepad
│   ├── engine-ai/          # A* + behavior trees
│   ├── engine-assets/      # asset loading + hot-reload
│   └── engine-launcher/    # demo binary that wires it all together
├── .github/workflows/ci.yml # Windows + Linux CI with artifact upload
└── assets/                 # placeholder for shipped game assets
```

## Build

### Prerequisites

- **Rust** 1.82+ (stable) — https://rustup.rs
- **Linux**: `apt install libx11-dev libxkbcommon-dev libwayland-dev libasound2-dev libudev-dev libxcb1-dev libxrandr-dev libxi-dev libgl1-mesa-dev libegl-dev`
- **Windows**: Visual Studio Build Tools 2022 (MSVC)

### Compile

```bash
cargo build --release -p engine-launcher
# Binary: target/release/enginebeta (Linux) or target/release/enginebeta.exe (Windows)
```

### Run the demo

```bash
cargo run --release -p engine-launcher
```

**Controls:**
- `Space` — drop a new bouncing cube
- `ESC` — quit

The demo seeds a static floor + one bouncing cube, and lets you drop more with
Space. Every drop plays a beep via the audio engine.

## Cross-compile

The CI workflow already builds for `x86_64-unknown-linux-gnu` and
`x86_64-pc-windows-msvc`. To build Windows from Linux:

```bash
rustup target add x86_64-pc-windows-gnu
cargo build --release --target x86_64-pc-windows-gnu -p engine-launcher
```

## CI / CD

Every push and PR triggers a build on both Ubuntu 22.04 and Windows Server 2022.
Binaries are uploaded as workflow artifacts (90-day retention) and, on `v*`
tags, attached to a GitHub Release.

**Build status:** https://github.com/salom600/Enginebeta/actions

## Architecture

### ECS (entity–component store)

`engine-core::World` stores components in typed `Column<T>`s — one `Vec<T>` per
component type, indexed by entity id. Iteration is cache-friendly (dense arrays)
and despawns are O(1) per column via swap-remove. Multi-column system access uses
`World::columns2<A, B, _>` / `columns3<A, B, C, _>`, which use `TypeId`-keyed
storage to soundly hand out multiple `&mut Column<T>` simultaneously (a known
sound pattern — same one `hecs` and `bevy_ecs` use internally).

### Stage-based scheduler

`engine_core::App` runs systems in five stages: `Startup → PreUpdate →
FixedUpdate (×N) → Update → PostUpdate`. Fixed-update runs at 60 Hz by default,
accumulating frame time and stepping the simulation in fixed increments. This
decouples simulation rate from frame rate — physics is deterministic regardless
of refresh rate.

### Rendering

`engine_render::Renderer` owns a `wgpu::Surface` bound to the winit window, a
single unlit-color render pipeline, and a uniform buffer holding the camera
view-projection matrix. Each frame: clear → set pipeline → draw all submitted
meshes → present. The MVP uses one shader (WGSL) with no lighting; the
`Renderer` struct is the extension point for adding post-processing, materials,
and lighting passes.

## License

Dual-licensed under MIT or Apache-2.0, at your option.
