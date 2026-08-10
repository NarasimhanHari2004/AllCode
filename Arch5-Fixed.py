#!/usr/bin/env python3
"""
DeepFilterNet3 Real-Time-Capable Audio Denoiser with ANC Diagnostics
======================================================================
Architecture-5: Uses DeepFilterNet3 (deep filtering in the ERB + complex
domain) to enhance noisy speech. DeepFilterNet3 works natively at 48kHz,
so unlike the Demucs-based architectures there is no manual chunking loop
needed for typical file lengths -- the library handles streaming internally.

Optimized for Microsoft Visual Studio. Computes ANC performance metrics and
automatically opens the generated Architecture-5 dashboard figure.
"""

import argparse
import os
import time
import glob
import librosa
import librosa.display  # FIX: required explicitly — librosa.display.specshow()
                         # is not guaranteed to be available just from `import librosa`.
                         # Without this the script crashes with AttributeError right
                         # at the plotting step. Same recurring bug as Architecture-1/2/3/4.
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# COMPAT SHIM: deepfilternet 0.5.6 still imports the pre-2.1 torchaudio path
# `torchaudio.backend.common.AudioMetaData`, but that module was removed from
# newer torchaudio releases (the class moved to `torchaudio.AudioMetaData`).
# Without this shim, `from df.enhance import ...` below raises:
#   ModuleNotFoundError: No module named 'torchaudio.backend'
# on any torchaudio version that dropped the legacy backend module. Recreating
# just the tiny bit of the old namespace deepfilternet needs avoids having to
# pin/downgrade torchaudio in the environment.
import sys
import types
import torchaudio

if "torchaudio.backend.common" not in sys.modules:
    try:
        _target = torchaudio.AudioMetaData
    except AttributeError as e:
        raise ImportError(
            "Could not locate AudioMetaData on this torchaudio install; "
            "the compatibility shim for deepfilternet needs updating."
        ) from e

    _backend_mod = types.ModuleType("torchaudio.backend")
    _common_mod = types.ModuleType("torchaudio.backend.common")
    _common_mod.AudioMetaData = _target
    _backend_mod.common = _common_mod
    sys.modules["torchaudio.backend"] = _backend_mod
    sys.modules["torchaudio.backend.common"] = _common_mod
    torchaudio.backend = _backend_mod

from df.enhance import enhance, init_df, load_audio, save_audio

# Force a non-interactive file-writing engine so it never relies on broken window popups
import matplotlib
matplotlib.use('Agg')

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def load_model():
    """Loads the pretrained DeepFilterNet3 model + its internal DF state config."""
    print("[INFO] Loading DeepFilterNet3 model weights...")
    model, df_state, _ = init_df()
    print(f"[INFO] Model ready. Internal processing sample rate: {df_state.sr()} Hz")
    return model, df_state


def calculate_metrics(noisy_np, clean_np, sr, frame_ms=20):
    """
    Reference-free denoising metrics (no paired clean ground-truth file exists,
    so nothing here can be a *true* SNR delta or true THD -- see notes below).

    - level_change_db: overall RMS power change between noisy input and the
      denoised output. This is NOT "noise removed" -- some of that power change
      is speech energy the model altered too. Kept as a general strength-of-effect
      indicator, relabeled to stop implying it isolates noise specifically.

    - snr_improvement_db: estimated using a noise-floor method. We estimate a
      noise floor independently in the noisy and denoised signals from each
      one's own quietest short-time frames, turn that into an SNR estimate for
      each, and report the difference. Not a substitute for a paired-reference
      metric (e.g. against a known-clean recording), but it's an honest
      before/after comparison rather than a raw output-only number.

    - spectral_flatness: a well-defined measure for broadband, non-tonal audio
      like speech (0 = tonal/peaky spectrum, 1 = noise-like/flat spectrum).
    """
    eps = 1e-10

    min_len = min(len(noisy_np), len(clean_np))
    noisy_np = noisy_np[:min_len]
    clean_np = clean_np[:min_len]

    power_noisy = np.mean(noisy_np ** 2)
    power_clean = np.mean(clean_np ** 2)
    level_change_db = 10 * np.log10((power_noisy + eps) / (power_clean + eps))

    def estimate_snr(signal):
        frame_len = max(1, int(sr * frame_ms / 1000))
        n_frames = len(signal) // frame_len
        if n_frames == 0:
            return 0.0
        frames = signal[:n_frames * frame_len].reshape(n_frames, frame_len)
        frame_power = np.mean(frames ** 2, axis=1)
        noise_floor = np.percentile(frame_power, 10) + eps
        active_power = np.mean(frame_power) + eps
        return 10 * np.log10(active_power / noise_floor)

    snr_improvement_db = estimate_snr(clean_np) - estimate_snr(noisy_np)

    fft_clean = np.abs(np.fft.rfft(clean_np)) + eps
    geo_mean = np.exp(np.mean(np.log(fft_clean)))
    arith_mean = np.mean(fft_clean)
    spectral_flatness = float(geo_mean / arith_mean)

    return {
        "level_change_db": level_change_db,
        "snr_improvement_db": snr_improvement_db,
        "spectral_flatness": spectral_flatness,
    }


def save_and_show_plots(noisy_np, clean_np, sr, plot_path, metrics):
    """Generates visualization dashboard panel and writes it directly to disk storage."""
    plt.close('all')

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Architecture-5: DeepFilterNet3 Denoising Pipeline for ANC", fontsize=16, fontweight='bold')

    min_len = min(len(noisy_np), len(clean_np))
    noisy_np = noisy_np[:min_len]
    clean_np = clean_np[:min_len]
    time_axis = np.linspace(0, min_len / sr, num=min_len)

    plt.subplot(3, 2, 1)
    plt.plot(time_axis, noisy_np, color='crimson', alpha=0.7)
    plt.title("Noisy Signal Waveform")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(3, 2, 2)
    plt.plot(time_axis, clean_np, color='dodgerblue', alpha=0.7)
    plt.title("Enhanced (DeepFilterNet3) Waveform")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(3, 2, 3)
    D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_np)), ref=np.max)
    librosa.display.specshow(D_noisy, sr=sr, x_axis='time', y_axis='linear', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Noisy Spectrogram")

    plt.subplot(3, 2, 4)
    D_clean = librosa.amplitude_to_db(np.abs(librosa.stft(clean_np)), ref=np.max)
    librosa.display.specshow(D_clean, sr=sr, x_axis='time', y_axis='linear', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("DeepFilterNet3 Spectrogram")

    plt.subplot(3, 1, 3)
    plt.axis('off')
    text_str = (
        f"PERFORMANCE SUMMARY METRICS (ARCHITECTURE-5 - DEEPFILTERNET3):\n"
        f"----------------------------------------------------------------------\n"
        f"\u2022 Overall Level Change: {metrics['level_change_db']:.2f} dB\n"
        f"\u2022 Estimated SNR Improvement: {metrics['snr_improvement_db']:.2f} dB\n"
        f"\u2022 Spectral Flatness (denoised): {metrics['spectral_flatness']:.3f}\n"
    )
    plt.text(0.05, 0.3, text_str, fontsize=13, family='monospace',
             bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=1'))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(plot_path, dpi=150)
    print(f"[INFO] -> Architecture-5 Metrics Dashboard Saved: {plot_path}")

    try:
        os.startfile(plot_path)
        print("[INFO] Launched default system image viewer to display charts.")
    except Exception as e:
        print(f"[WARN] Script couldn't open image viewer automatically: {str(e)}")


def process_file(model, df_state, in_path, out_path):
    """Orchestrates individual file lifecycle: load -> enhance -> save -> diagnose."""
    print(f"\n[INFO] Processing: {in_path}")
    try:
        model_sr = df_state.sr()

        # DeepFilterNet's own loader resamples to the model's native rate for inference.
        noisy_audio, _ = load_audio(in_path, sr=model_sr)

        # Keep a librosa-loaded copy at the same rate purely for the "before" side of
        # the metrics/plots, so both signals are directly comparable sample-for-sample.
        noisy_np, _ = librosa.load(in_path, sr=model_sr)

        t0 = time.time()
        enhanced_audio = enhance(model, df_state, noisy_audio)
        t1 = time.time()

        dur = noisy_audio.shape[-1] / model_sr
        print(f"[INFO] Track Length: {dur:.1f} seconds")

        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

        save_audio(out_path, enhanced_audio, model_sr)
        print(f"[INFO] -> Saved Clean File: {out_path} (Inference took: {t1 - t0:.2f}s)")

        clean_np = enhanced_audio.squeeze(0).numpy() if hasattr(enhanced_audio, "numpy") else np.asarray(enhanced_audio).squeeze()

        metrics = calculate_metrics(noisy_np, clean_np, model_sr)
        print(f"       [METRIC] Overall Level Change: {metrics['level_change_db']:.2f} dB")
        print(f"       [METRIC] Est. SNR Improvement: {metrics['snr_improvement_db']:.2f} dB")
        print(f"       [METRIC] Spectral Flatness: {metrics['spectral_flatness']:.3f}")

        base_name, _ = os.path.splitext(out_path)
        plot_path = os.path.abspath(str(base_name) + "_architecture5_metrics.png")
        save_and_show_plots(noisy_np, clean_np, model_sr, plot_path, metrics)

    except Exception as e:
        print(f"[ERROR] Failed to process {in_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="DeepFilterNet3 Local Audio Denoiser.")
    parser.add_argument("--input", "-i", default="test_input.wav", help="Path to a noisy audio file")
    parser.add_argument("--output", "-o", default="cleaned_output_dfn3_clean.wav", help="Path to save the cleaned audio file")
    parser.add_argument("--batch", action="store_true", help="Batch mode")
    args = parser.parse_args()

    model, df_state = load_model()

    if args.batch:
        if not os.path.isdir(args.input):
            print(f"[ERROR] Batch mode active but input folder not found: {args.input}")
            return
        files = [f for f in glob.glob(os.path.join(args.input, "*")) if f.lower().endswith(AUDIO_EXTENSIONS)]
        if not files:
            print(f"[ERROR] No compatible audio files found in {args.input}")
            return
        os.makedirs(args.output, exist_ok=True)
        for f in files:
            out_name = os.path.splitext(os.path.basename(f))[0] + "_dfn3_clean.wav"
            process_file(model, df_state, f, os.path.join(args.output, out_name))
    else:
        if not os.path.isfile(args.input):
            print(f"[ERROR] Input file not found: {args.input}. Make sure 'test_input.wav' is in your Metrics folder!")
            return
        process_file(model, df_state, args.input, args.output)

    print("\n[DONE] All processing complete.")
    input("\nPress ENTER to close the terminal window...")


if __name__ == "__main__":
    main()