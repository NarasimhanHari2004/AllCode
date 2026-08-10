#!/usr/bin/env python3
"""
State-of-the-Art Time-Frequency Domain Audio Denoiser
=====================================================
Architecture-4: Utilizes the ClearVoice FRCRN network architecture
to enhance noisy speech inside 2D spectrogram matrices.

Optimized for Microsoft Visual Studio. Computes ANC performance metrics and
automatically opens the integrated Architecture-4 dashboard figure.
"""

import argparse
import os
import time
import librosa
import librosa.display  # FIX: required explicitly — librosa.display.specshow()
                         # is not guaranteed to be available just from `import librosa`.
                         # Without this the script crashes with AttributeError right
                         # at the plotting step, after inference has already finished.
                         # Same recurring bug as in Architecture-1/2/3.
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from clearvoice import ClearVoice

# Force a non-interactive file-writing engine so it never relies on broken window popups
import matplotlib
matplotlib.use('Agg')

def calculate_metrics(noisy_np, clean_np, sr, frame_ms=20):
    """
    Reference-free denoising metrics (no paired clean ground-truth file exists,
    so nothing here can be a *true* SNR delta or true THD — see notes below).

    - level_change_db: overall RMS power change between noisy input and the
      denoised output. This is NOT "noise removed" — some of that power change
      is speech energy the model altered too. Kept as a general strength-of-effect
      indicator, relabeled to stop implying it isolates noise specifically.

    - snr_improvement_db: previously computed as clean_power / (noisy-clean)_power
      and mislabeled "improvement" — that's really just an *output* SNR estimate
      (and it silently assumes the denoised signal is pure speech with zero
      residual noise, which inflates the number). To get an actual before/after
      *improvement*, we instead estimate a noise floor independently in the noisy
      and denoised signals from each one's own quietest short-time frames, turn
      that into an SNR estimate for each, and report the difference.

    - spectral_flatness: replaces the old "thd_percent". The old formula treated
      the single loudest FFT bin as "the fundamental" and everything else as
      "harmonics" — that's only meaningful for a periodic tone, not broadband
      speech, so it was never really measuring THD. Spectral flatness (0 = tonal,
      1 = noise-like) is a well-defined, honest stand-in for broadband audio.
    """
    eps = 1e-10

    # Ensure both arrays are completely flat and matched in size
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

    # Explicit Title Header Target Configuration for Architecture-4
    fig.suptitle("Architecture-4: ClearVoice FRCRN Spectrogram Pipeline for ANC", fontsize=16, fontweight='bold')

    # Ensure audio length alignment for time indexing
    min_len = min(len(noisy_np), len(clean_np))
    noisy_np = noisy_np[:min_len]
    clean_np = clean_np[:min_len]

    time_axis = np.linspace(0, min_len / sr, num=min_len)

    # 1. Waveform Render Columns
    plt.subplot(3, 2, 1)
    plt.plot(time_axis, noisy_np, color='crimson', alpha=0.7)
    plt.title("Noisy Signal Waveform")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(3, 2, 2)
    plt.plot(time_axis, clean_np, color='dodgerblue', alpha=0.7)
    plt.title("Enhanced (FRCRN Cleaned) Waveform")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.5)

    # 2. Short-Time Fourier Transform Matrices (Spectrogram Rows)
    plt.subplot(3, 2, 3)
    D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_np)), ref=np.max)
    librosa.display.specshow(D_noisy, sr=sr, x_axis='time', y_axis='linear', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Noisy Spectrogram")

    plt.subplot(3, 2, 4)
    D_clean = librosa.amplitude_to_db(np.abs(librosa.stft(clean_np)), ref=np.max)
    librosa.display.specshow(D_clean, sr=sr, x_axis='time', y_axis='linear', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("FRCRN Spectrogram")

    # 3. Parameter Performance Data Display Window
    plt.subplot(3, 1, 3)
    plt.axis('off')
    text_str = (
        f"PERFORMANCE SUMMARY METRICS (ARCHITECTURE-4 - TIME-FREQUENCY DOMAIN):\n"
        f"----------------------------------------------------------------------\n"
        f"• Overall Level Change: {metrics['level_change_db']:.2f} dB\n"
        f"• Estimated SNR Improvement: {metrics['snr_improvement_db']:.2f} dB\n"
        f"• Spectral Flatness (denoised): {metrics['spectral_flatness']:.3f}\n"
    )
    plt.text(0.05, 0.3, text_str, fontsize=13, family='monospace',
             bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=1'))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(plot_path, dpi=150)
    print(f"[INFO] -> Architecture-4 Metrics Dashboard Saved: {plot_path}")

    # Force Desktop Image Display Workaround using absolute file path parameters
    try:
        os.startfile(plot_path)
        print("[INFO] Launched default system image viewer to display charts.")
    except Exception as e:
        print(f"[WARN] Script couldn't open image viewer automatically: {str(e)}")

def process_file(engine, target_sr, in_path, out_path):
    """Orchestrates individual file lifecycles with advanced diagnostics."""
    print(f"\n[INFO] Processing: {in_path}")
    try:
        # Load raw sample sound vector for input metrics tracking
        noisy_signal, _ = librosa.load(in_path, sr=target_sr)

        t0 = time.time()
        # 1. Execute Time-Frequency Transmutation Engine
        output_array = engine(input_path=in_path, online_write=False)
        t1 = time.time()

        # 2. Extract single channel index track structure
        if output_array.ndim > 1:
            clean_signal = output_array[0, :]
        else:
            clean_signal = output_array

        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

        # 3. Write target audio to project location
        sf.write(out_path, clean_signal, target_sr)
        print(f"[INFO] -> Saved Clean Audio File: {out_path} (Inference took: {t1 - t0:.2f}s)")

        # Calculate diagnostics
        metrics = calculate_metrics(noisy_signal, clean_signal, target_sr)
        print(f"       [METRIC] Overall Level Change: {metrics['level_change_db']:.2f} dB")
        print(f"       [METRIC] Est. SNR Improvement: {metrics['snr_improvement_db']:.2f} dB")
        print(f"       [METRIC] Spectral Flatness: {metrics['spectral_flatness']:.3f}")

        # Explicitly convert path objects into strict string targets before extending text names
        base_name, _ = os.path.splitext(out_path)
        plot_path = os.path.abspath(str(base_name) + "_architecture4_metrics.png")

        # Launch visualization panel builders
        save_and_show_plots(noisy_signal, clean_signal, target_sr, plot_path, metrics)

    except Exception as e:
        print(f"[ERROR] Failed to process {in_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Time-Frequency Speech Enhancement.")
    # Fallbacks protect script execution within Visual Studio isolated environments
    parser.add_argument("--input", "-i", default="test_input.wav", help="Input noisy audio track (.wav)")
    parser.add_argument("--output", "-o", default="cleaned_output_frcrn_clean.wav", help="Output clean audio path (.wav)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] Input audio file not found: {args.input}. Make sure 'test_input.wav' is in your Metrics folder!")
        return

    # ClearVoice processing specifications look for 16kHz audio tracking paths
    target_sr = 16000

    print("[INFO] Loading FRCRN Time-Frequency Neural Layers to GPU...")
    engine = ClearVoice(task='speech_enhancement', model_names=['FRCRN_SE_16K'])

    process_file(engine, target_sr, args.input, args.output)

    print("\n[DONE] All processing complete.")
    input("\nPress ENTER to close the terminal window...")

if __name__ == "__main__":
    main()
