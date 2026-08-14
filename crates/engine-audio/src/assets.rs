//! Asset-loading helpers for audio (WAV bytes → [`crate::Sound`]).
//!
//! Real engines stream audio from disk; this MVP just holds bytes in memory.

use crate::AudioEngine;

impl AudioEngine {
    /// Convenience: load a tiny in-memory test beep so demos always have audio.
    pub fn load_default_beep(&self) -> anyhow::Result<()> {
        let samples = crate::beep_sine(440.0, 0.4);
        // Build a WAV file from samples in memory.
        let wav = wav_from_mono_f32(&samples, 44_100);
        // WAV bytes need to be 'static for the cursor; leak them.
        let leaked: &'static [u8] = Box::leak(wav.into_boxed_slice());
        self.load_wav("beep", leaked)
    }
}

/// Encode a mono f32 PCM buffer as a 16-bit PCM WAV file.
fn wav_from_mono_f32(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let n = samples.len();
    let bytes_per_sample = 2i32; // 16-bit
    let byte_count = (n as i32) * bytes_per_sample;
    let chunk_size = 36 + byte_count;
    let mut out = Vec::with_capacity(44 + n * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&chunk_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16i32.to_le_bytes()); // subchunk1 size
    out.extend_from_slice(&1i16.to_le_bytes()); // PCM
    out.extend_from_slice(&1i16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate as i32 * bytes_per_sample).to_le_bytes()); // byte rate
    out.extend_from_slice(&(bytes_per_sample as i16).to_le_bytes()); // block align
    out.extend_from_slice(&16i16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&byte_count.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
