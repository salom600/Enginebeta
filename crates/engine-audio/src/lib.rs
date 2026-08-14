//! engine-audio — 2D and 3D positional audio on top of `rodio`.
//!
//! The engine keeps a single `OutputStream` (rodio requires this) and exposes:
//! - [`AudioEngine`] — top-level owner of the output stream + sound cache
//! - [`Sound`] — a decoded sound ready to play
//! - [`AudioEmitter`] / [`AudioListener`] — components for positional mixing
//!
//! 3D positional mixing uses rodio's built-in `SpatialSink`.

use anyhow::Context as _;
use glam::Vec3;
use parking_lot::RwLock;
use rodio::buffer::SamplesBuffer;
use rodio::decoder::Decoder;
use rodio::source::Source;
use rodio::{OutputStream, OutputStreamHandle, Sink, SpatialSink};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

pub mod assets;
pub mod listener;
pub mod spatial;

pub use listener::{AudioEmitter, AudioListener};

/// Top-level audio engine. Owns the rodio output stream and a cache of decoded
/// sounds. Cloneable so multiple systems can share it (the underlying stream
/// is wrapped in `Arc`).
#[derive(Clone)]
pub struct AudioEngine {
    inner: Arc<AudioEngineInner>,
}

struct AudioEngineInner {
    /// OutputStream must stay alive for the lifetime of any Sink created from
    /// its handle, so we keep it here.
    _stream: OutputStream,
    handle: OutputStreamHandle,
    sounds: RwLock<HashMap<String, Arc<Sound>>>,
}

impl AudioEngine {
    /// Construct a new audio engine. May fail if there is no audio device.
    pub fn new() -> anyhow::Result<Self> {
        let (stream, handle) = OutputStream::try_default()
            .context("failed to open default audio output stream")?;
        Ok(Self {
            inner: Arc::new(AudioEngineInner {
                _stream: stream,
                handle,
                sounds: RwLock::new(HashMap::new()),
            }),
        })
    }

    /// Decode WAV bytes and cache them under `name`. Subsequent plays use the cache.
    pub fn load_wav(&self, name: impl Into<String>, bytes: &'static [u8]) -> anyhow::Result<()> {
        let name = name.into();
        let cursor = Cursor::new(bytes);
        let decoder = Decoder::new_wav(cursor).context("failed to decode WAV")?;
        let sample_rate = decoder.sample_rate();
        let channels = decoder.channels();
        let samples: Vec<f32> = decoder.convert_samples().collect();
        let sound = Sound {
            samples,
            sample_rate,
            channels,
        };
        self.inner.sounds.write().insert(name, Arc::new(sound));
        Ok(())
    }

    /// Play a cached sound at `volume` (0..1). Returns a `Sink` so the caller
    /// can stop / fade / loop the source.
    pub fn play(&self, name: &str, volume: f32) -> anyhow::Result<Sink> {
        let sound = {
            let sounds = self.inner.sounds.read();
            sounds
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("sound '{name}' not in cache"))?
        };
        let sink = Sink::try_new(&self.inner.handle).context("failed to create sink")?;
        sink.append(sound.source());
        sink.set_volume(volume.clamp(0.0, 1.0));
        Ok(sink)
    }

    /// Play a sound positioned in 3D world space relative to `listener`.
    /// Uses rodio's SpatialSink for distance attenuation + stereo pan.
    pub fn play_3d(
        &self,
        name: &str,
        emitter_pos: Vec3,
        listener: &AudioListener,
        base_volume: f32,
    ) -> anyhow::Result<SpatialSink> {
        let sound = {
            let sounds = self.inner.sounds.read();
            sounds
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("sound '{name}' not in cache"))?
        };
        // Compute ear positions: ~0.1m left/right of listener position, along
        // the listener's right vector (perpendicular to forward and up).
        let right = listener
            .up
            .cross(listener.forward)
            .normalize_or_zero();
        let ear_offset = 0.1;
        let left_ear = listener.position - right * ear_offset;
        let right_ear = listener.position + right * ear_offset;
        let sink = SpatialSink::try_new(
            &self.inner.handle,
            [emitter_pos.x, emitter_pos.y, emitter_pos.z],
            [left_ear.x, left_ear.y, left_ear.z],
            [right_ear.x, right_ear.y, right_ear.z],
        )
        .context("failed to create SpatialSink")?;
        sink.append(sound.source());
        sink.set_volume(base_volume.clamp(0.0, 1.0));
        Ok(sink)
    }

    /// Number of currently cached sounds.
    pub fn cached_count(&self) -> usize {
        self.inner.sounds.read().len()
    }
}

/// A decoded sound stored as interleaved f32 samples.
pub struct Sound {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl Sound {
    /// Build a rodio `Source` from this sound's samples. Wraps the samples in a
    /// `SamplesBuffer` so rodio knows the sample rate and channel count.
    pub fn source(&self) -> SamplesBuffer<f32> {
        SamplesBuffer::new(self.channels, self.sample_rate, self.samples.clone())
    }
}

/// A short hand-rolled beep used as a fallback when no real asset is available.
/// Generates `duration_secs` seconds of `freq` Hz sine at 44.1kHz mono.
pub fn beep_sine(freq: f32, duration_secs: f32) -> Vec<f32> {
    let sr = 44_100.0f32;
    let n = (duration_secs * sr) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            // Apply a quick fade-in / fade-out to avoid clicks.
            let fade = (i as f32 / 200.0).min(1.0).min((n - i) as f32 / 200.0);
            (2.0 * std::f32::consts::PI * freq * t).sin() * 0.25 * fade
        })
        .collect()
}
