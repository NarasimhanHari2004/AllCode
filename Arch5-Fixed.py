import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from df.enhance import enhance, init_df, load_audio, save_audio


def calculate_metrics(noisy_np, clean_np, sr, frame_ms=20):
    """
    Reference-free denoising metrics (no paired clean ground-truth file exists,
    so nothing here can be a *true* SNR delta or true THD).

    - level_change_db: overall RMS power change between noisy input and the
      denoised output. Not "noise removed" specifically -- speech energy
      changes contribute too. General strength-of-effect indicator.

    - snr_improvement_db: estimates a noise floor independently in the noisy
      and denoised signals from each one's own quietest short-time frames,
      converts that into an SNR estimate for each, and reports the difference.

    - spectral_flatness: 0 = tonal, 1 = noise-like. Honest stand-in for
      broadband audio (replaces a bogus single-bin-FFT "THD" calculation).
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
    """Generates visualization dashboard panel and writes it to disk."""
    plt.close('all')

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Architecture-5: DeepFilterNet Denoising Pipeline", fontsize=16, fontweight='bold')

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
    plt.title("Enhanced (DeepFilterNet) Waveform")
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
    plt.title("DeepFilterNet Spectrogram")

    plt.subplot(3, 1, 3)
    plt.axis('off')
    text_str = (
        f"PERFORMANCE SUMMARY METRICS (ARCHITECTURE-5 - DEEPFILTERNET):\n"
        f"----------------------------------------------------------------------\n"
        f"- Overall Level Change: {metrics['level_change_db']:.2f} dB\n"
        f"- Estimated SNR Improvement: {metrics['snr_improvement_db']:.2f} dB\n"
        f"- Spectral Flatness (denoised): {metrics['spectral_flatness']:.3f}\n"
    )
    plt.text(0.05, 0.3, text_str, fontsize=13, family='monospace',
              bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=1'))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plot_path, dpi=150)
    print(f"[INFO] -> Architecture-5 Metrics Dashboard Saved: {plot_path}")

    try:
        os.startfile(plot_path)  # Windows only
        print("[INFO] Launched default system image viewer to display charts.")
    except Exception as e:
        print(f"[WARN] Script couldn't open image viewer automatically: {str(e)}")


if __name__ == "__main__":
    noisy_path = "test_input.wav"          # reuse your existing test file
    output_wav = "a5_output.wav"
    output_plot = "a5_output_architecture5_metrics.png"

    # Load DeepFilterNet's default pretrained model
    model, df_state, _ = init_df()

    # Load and enhance your noisy audio
    audio, _ = load_audio(noisy_path, sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)

    # Save enhanced audio next to your other Arch outputs
    save_audio(output_wav, enhanced, df_state.sr())

    # Flatten to 1D numpy for the metrics/plotting functions
    noisy_np = audio.numpy().flatten()
    clean_np = enhanced.numpy().flatten()
    sr = df_state.sr()

    metrics = calculate_metrics(noisy_np, clean_np, sr)
    save_and_show_plots(noisy_np, clean_np, sr, output_plot, metrics)
