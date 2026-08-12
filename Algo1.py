"""
denoiser_dsp.py
================
Production-grade adaptive spectral-gating noise cancellation.

Runs with only numpy + scipy (no torch / internet / pretrained weights
required), so it works immediately in any Python environment.

Algorithm (multi-stage, tuned to avoid the two classic failure modes of
naive spectral subtraction: "musical noise" and over-suppression of speech):

  1. STFT the signal.
  2. Automatically locate the noise-only region (lowest-energy frames) OR
     accept an explicit noise clip.
  3. Build a per-frequency-bin noise statistical profile (mean + std, in dB).
  4. Compute a smooth, soft attenuation mask (not a hard gate) using a
     sigmoid transition around an adaptive threshold. This is the key
     difference from textbook spectral subtraction and is what removes
     musical-noise artifacts.
  5. Smooth the mask across both time and frequency (separable 2D
     convolution) so gain changes are gradual -> no clicking / warbling.
  6. Apply a spectral floor so the residual noise fades naturally instead
     of being gated to hard silence (which sounds unnatural / robotic).
  7. Re-synthesize via inverse STFT with overlap-add, using the ORIGINAL
     phase (phase is inaudible-error-tolerant; magnitude is what matters).
  8. Optional second pass at lower strength for stubborn residual hiss.

Works on mono or stereo, any input sample rate/bit depth (16/24/32-bit PCM
or float WAV).
"""

from __future__ import annotations
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft
from scipy.ndimage import uniform_filter1d, convolve1d


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def load_wav(path: str):
    """Load a WAV file and return (float32 samples in [-1, 1], sample_rate,
    original_dtype). Samples shape: (n,) mono or (n, channels) stereo."""
    sr, data = wavfile.read(path)
    orig_dtype = data.dtype

    if orig_dtype == np.uint8:
        # 8-bit PCM WAV is unsigned with 128 as the zero point (unlike every
        # other PCM width, which is signed). Handle it explicitly or the
        # whole signal comes out with a +1.0 DC offset.
        audio = (data.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(orig_dtype, np.integer):
        max_val = float(np.iinfo(orig_dtype).max)
        audio = data.astype(np.float32) / max_val
    else:
        audio = data.astype(np.float32)

    return audio, sr, orig_dtype


def save_wav(path: str, audio: np.ndarray, sr: int, orig_dtype=np.int16):
    """Write float32 [-1, 1] samples back out, clipped and cast to the
    requested dtype (defaults to 16-bit PCM, the safe universal choice)."""
    audio = np.clip(audio, -1.0, 1.0)

    if np.issubdtype(orig_dtype, np.integer):
        max_val = float(np.iinfo(orig_dtype).max)
        out = (audio * max_val).astype(orig_dtype)
    else:
        out = audio.astype(np.float32)

    wavfile.write(path, sr, out)


# --------------------------------------------------------------------------- #
# Core single-channel denoiser
# --------------------------------------------------------------------------- #

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def denoise_channel(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_ratio: float = 0.25,
    n_std_thresh: float = 1.4,
    prop_decrease: float = 0.95,
    freq_smooth_hz: float = 250.0,
    time_smooth_ms: float = 60.0,
    spectral_floor_db: float = -32.0,
    noise_clip: np.ndarray | None = None,
    two_pass: bool = True,
) -> np.ndarray:
    """Denoise a single mono float32 channel and return a same-length
    float32 array."""

    # Trivially short / empty input: nothing meaningful to denoise.
    if len(y) < 64:
        return y.astype(np.float32)

    # Guard against clips shorter than the requested FFT window (e.g. very
    # short recordings, or the tail-end second pass on a tiny signal).
    # Shrink n_fft to the next-smaller power of two that fits, with a
    # sane floor so we still get a usable frequency resolution.
    if len(y) < n_fft:
        n_fft = max(64, 1 << int(np.floor(np.log2(max(len(y), 64)))))

    hop = max(1, int(n_fft * hop_ratio))
    if hop >= n_fft:
        hop = n_fft // 2
    win = "hann"

    f, t, Z = stft(y, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop,
                    padded=True, boundary="zeros")
    mag = np.abs(Z)
    phase = np.angle(Z)

    mag_db = 20 * np.log10(np.maximum(mag, 1e-10))

    # --- 1. Noise profile ---------------------------------------------------
    if noise_clip is not None and len(noise_clip) > n_fft:
        _, _, Zn = stft(noise_clip, fs=sr, window=win, nperseg=n_fft,
                         noverlap=n_fft - hop, padded=True, boundary="zeros")
        noise_mag_db = 20 * np.log10(np.maximum(np.abs(Zn), 1e-10))
    else:
        # Auto-detect noise-only frames: lowest-energy 15% of frames,
        # by RMS in dB. Works well for typical recordings that have some
        # silence/background-only stretches (start, gaps between speech).
        frame_energy = mag_db.mean(axis=0)
        n_noise_frames = max(6, int(0.15 * mag_db.shape[1]))
        noise_frame_idx = np.argsort(frame_energy)[:n_noise_frames]
        noise_mag_db = mag_db[:, noise_frame_idx]

    noise_mean_db = noise_mag_db.mean(axis=1, keepdims=True)
    noise_std_db = noise_mag_db.std(axis=1, keepdims=True) + 1e-6

    # --- 2. Soft threshold / sigmoid mask -----------------------------------
    threshold_db = noise_mean_db + n_std_thresh * noise_std_db
    # distance above threshold, normalized by std -> smooth sigmoid gate
    margin = (mag_db - threshold_db) / (noise_std_db + 1e-6)
    soft_gate = _sigmoid(margin)                    # 0 (noise) .. 1 (signal)
    gain = spectral_floor_db_to_lin(spectral_floor_db) + \
        (1 - spectral_floor_db_to_lin(spectral_floor_db)) * soft_gate
    gain = 1 - prop_decrease * (1 - gain)            # blend by strength

    # --- 3. Smooth the mask (time + frequency) to kill musical noise -------
    freq_res = sr / n_fft
    freq_smooth_bins = max(1, int(round(freq_smooth_hz / freq_res)))
    time_res_ms = (hop / sr) * 1000.0
    time_smooth_frames = max(1, int(round(time_smooth_ms / time_res_ms)))

    gain = uniform_filter1d(gain, size=freq_smooth_bins, axis=0, mode="nearest")
    gain = uniform_filter1d(gain, size=time_smooth_frames, axis=1, mode="nearest")
    gain = np.clip(gain, spectral_floor_db_to_lin(spectral_floor_db), 1.0)

    # --- 4. Apply mask, reconstruct ----------------------------------------
    Z_clean = gain * mag * np.exp(1j * phase)
    _, y_clean = istft(Z_clean, fs=sr, window=win, nperseg=n_fft,
                        noverlap=n_fft - hop)
    y_clean = y_clean[: len(y)]
    if len(y_clean) < len(y):
        y_clean = np.pad(y_clean, (0, len(y) - len(y_clean)))

    # --- 5. Optional gentle second pass for residual hiss -------------------
    if two_pass:
        y_clean = denoise_channel(
            y_clean, sr, n_fft=n_fft, hop_ratio=hop_ratio,
            n_std_thresh=n_std_thresh + 0.3, prop_decrease=prop_decrease * 0.5,
            freq_smooth_hz=freq_smooth_hz, time_smooth_ms=time_smooth_ms,
            spectral_floor_db=spectral_floor_db, noise_clip=noise_clip,
            two_pass=False,
        )

    return y_clean.astype(np.float32)


def spectral_floor_db_to_lin(db: float) -> float:
    return float(10 ** (db / 20))


def match_loudness(original: np.ndarray, processed: np.ndarray,
                    percentile: float = 90.0, max_gain_db: float = 6.0) -> np.ndarray:
    """
    Spectral gating inherently reduces overall level (it attenuates the
    noise floor between/under speech, which lowers average energy even
    though *speech* itself is barely touched). Left uncorrected this makes
    denoised output sound muffled/quiet compared to the source.

    This restores perceptual loudness by matching the high-percentile
    ("loud"/speech-dominant) sample energy of the processed signal to that
    of the original, with a capped makeup gain so we never amplify residual
    noise beyond what's reasonable -- and never clip.
    """
    orig_level = np.percentile(np.abs(original), percentile) + 1e-8
    proc_level = np.percentile(np.abs(processed), percentile) + 1e-8
    gain = orig_level / proc_level
    max_gain = 10 ** (max_gain_db / 20)
    gain = float(np.clip(gain, 1.0 / max_gain, max_gain))

    # Never let the makeup gain push peaks past the original file's own
    # peak level -- avoids clipping/distortion introduced by this step.
    proc_peak = np.max(np.abs(processed)) + 1e-8
    orig_peak = np.max(np.abs(original)) + 1e-8
    safe_gain = orig_peak / proc_peak
    gain = min(gain, max(safe_gain, 1.0 / max_gain))

    return processed * gain


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def denoise_wav(
    input_path: str,
    output_path: str = "denoised_audio.wav",
    strength: float = 0.95,
    noise_clip_path: str | None = None,
    n_std_thresh: float = 1.4,
    debug: bool = False,
):
    """
    Denoise `input_path` and write the result to `output_path`.

    strength: 0.0 (no denoising) .. 1.0 (maximum). Default 0.95 is tuned
              for strong noise removal while keeping speech natural.
    noise_clip_path: optional WAV file containing ONLY the noise (e.g. a
              recording of room tone / hum with no speech). If provided,
              the noise profile is built from it instead of being
              auto-detected, which gives even more precise results.
    n_std_thresh: how many standard deviations above the noise floor a
              bin must be to be treated as signal. LOWER = more aggressive
              (removes more, risks eating quiet speech). Default 1.4.
              Try 0.5-1.0 if the default isn't removing enough.
    debug: print before/after stats so you can verify the algorithm is
              actually doing something on your specific file.
    """
    audio, sr, dtype = load_wav(input_path)

    noise_clip = None
    if noise_clip_path:
        noise_clip, noise_sr, _ = load_wav(noise_clip_path)
        if noise_sr != sr:
            raise ValueError(
                f"Noise clip sample rate ({noise_sr}) must match input "
                f"audio sample rate ({sr})."
            )
        if noise_clip.ndim > 1:
            noise_clip = noise_clip.mean(axis=1)

    if debug:
        dur = len(audio) / sr
        print(f"[debug] loaded: {input_path}")
        print(f"[debug]   sample_rate={sr}  duration={dur:.2f}s  "
              f"channels={'stereo' if audio.ndim > 1 else 'mono'}  "
              f"orig_dtype={dtype}")
        print(f"[debug]   input peak={np.max(np.abs(audio)):.4f}  "
              f"input RMS={np.sqrt(np.mean(audio.astype(np.float64) ** 2)):.4f}")

    if audio.ndim == 1:
        clean = denoise_channel(audio, sr, prop_decrease=strength,
                                 n_std_thresh=n_std_thresh, noise_clip=noise_clip)
        clean = match_loudness(audio, clean)
    else:
        # Process each channel independently, preserving stereo image.
        channels = []
        for c in range(audio.shape[1]):
            ch_clean = denoise_channel(audio[:, c], sr, prop_decrease=strength,
                                        n_std_thresh=n_std_thresh, noise_clip=noise_clip)
            ch_clean = match_loudness(audio[:, c], ch_clean)
            channels.append(ch_clean)
        clean = np.stack(channels, axis=1)

    if debug:
        diff = np.abs(clean.astype(np.float64) - audio.astype(np.float64))
        print(f"[debug]   output peak={np.max(np.abs(clean)):.4f}  "
              f"output RMS={np.sqrt(np.mean(clean.astype(np.float64) ** 2)):.4f}")
        print(f"[debug]   max sample difference={diff.max():.6f}  "
              f"mean sample difference={diff.mean():.6f}")
        if diff.max() < 1e-4:
            print("[debug]   WARNING: output is essentially identical to input. "
                  "Try a lower --n_std_thresh (e.g. 0.5-0.8) or check the file "
                  "actually contains a noise floor distinguishable from the signal.")

    save_wav(output_path, clean, sr, dtype)
    return output_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Spectral-gating noise cancellation for WAV files")
    p.add_argument("input", help="Path to noisy input .wav")
    p.add_argument("-o", "--output", default="denoised_audio.wav", help="Path to write denoised .wav")
    p.add_argument("-s", "--strength", type=float, default=0.95, help="Denoise strength 0-1 (default 0.95)")
    p.add_argument("-n", "--noise_clip", default=None, help="Optional WAV of noise-only reference")
    p.add_argument("-t", "--n_std_thresh", type=float, default=1.4,
                    help="Sensitivity: lower = more aggressive removal (default 1.4, try 0.5-0.8 for more)")
    p.add_argument("--debug", action="store_true", help="Print before/after diagnostics")
    args = p.parse_args()

    out = denoise_wav(args.input, args.output, strength=args.strength,
                       noise_clip_path=args.noise_clip, n_std_thresh=args.n_std_thresh,
                       debug=args.debug)
    print(f"Wrote denoised audio to: {out}")
