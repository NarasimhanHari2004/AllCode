#!/usr/bin/env python3
"""
Visual Studio Tailored Speech Denoising Script
=============================================
Optimized for Microsoft Visual Studio (Purple Logo).
Calculates ANC metrics and handles plot rendering safely within the IDE.
"""

import sys
import torch

import argparse
import os
import time
import glob
import librosa
import librosa.display  # FIX: required explicitly — librosa.display.specshow()
                         # is not guaranteed to be available just from `import librosa`
                         # in current librosa releases. Without this the script
                         # crashes with AttributeError right at the plotting step,
                         # after inference has already finished.
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# Force Visual Studio to use an interactive UI backend for plotting
import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    pass

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
        print("[WARN] No GPU detected — falling back to CPU. This will be slower.")
    return device

def load_model(model_name, device):
    from denoiser.pretrained import dns64, dns48, master64
    model_map = {"dns64": dns64, "dns48": dns48, "master64": master64}
    if model_name not in model_map:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(model_map.keys())}")
    print(f"[INFO] Loading pretrained model: {model_name} ...")
    try:
        model = model_map[model_name](pretrained=True)
    except AttributeError as e:
        # FIX: if this is genuinely hit (very old torch calling the deprecated
        # torch.set_default_tensor_type), handle it locally and explicitly
        # instead of globally neutering the function for the whole process.
        if "set_default_tensor_type" in str(e):
            raise RuntimeError(
                "Model loading failed due to a deprecated torch API call "
                "(torch.set_default_tensor_type) inside the denoiser package. "
                "Upgrade/downgrade torch or the denoiser package to compatible "
                "versions rather than silently patching torch's API."
            ) from e
        raise
    model = model.to(device)
    model.eval()
    return model

def load_audio(path, target_sr):
    data, sr = librosa.load(path, sr=target_sr)
    wav = torch.from_numpy(data)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav

def denoise_chunked(model, wav, device, chunk_len_sec=10):
    sr = model.sample_rate
    chunk_samples = chunk_len_sec * sr
    total_samples = wav.shape[-1]
    enhanced_chunks = []

    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = wav[:, start:end]

        pad_len = 0
        if chunk.shape[-1] < sr:
            pad_len = sr - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_len))

        chunk = chunk.to(device)

        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out_chunk = model(chunk.unsqueeze(0))
            else:
                out_chunk = model(chunk.unsqueeze(0))

        out_chunk = out_chunk.squeeze(0).detach().cpu()
        if pad_len > 0:
            out_chunk = out_chunk[:, :-pad_len]
        enhanced_chunks.append(out_chunk)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(enhanced_chunks, dim=-1)

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
    plt.close('all')

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Architecture-1: Demucs Speech Denoising Pipeline for ANC", fontsize=16, fontweight='bold')
    time_axis = np.linspace(0, len(noisy_np) / sr, num=len(noisy_np))

    plt.subplot(3, 2, 1)
    plt.plot(time_axis, noisy_np, color='crimson', alpha=0.7)
    plt.title("Noisy Signal Waveform")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(3, 2, 2)
    plt.plot(time_axis, clean_np, color='dodgerblue', alpha=0.7)
    plt.title("Enhanced (Cleaned) Waveform")
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
    plt.title("Clean Spectrogram")

    plt.subplot(3, 1, 3)
    plt.axis('off')
    text_str = (
        f"PERFORMANCE SUMMARY METRICS:\n"
        f"----------------------------------------------------------------------\n"
        f"• Overall Level Change: {metrics['level_change_db']:.2f} dB\n"
        f"• Estimated SNR Improvement: {metrics['snr_improvement_db']:.2f} dB\n"
        f"• Spectral Flatness (denoised): {metrics['spectral_flatness']:.3f}\n"
    )
    plt.text(0.05, 0.3, text_str, fontsize=13, family='monospace',
             bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=1'))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(plot_path, dpi=150)
    print(f"[INFO] -> Metrics Dashboard Saved: {plot_path}")

    print("[INFO] Displaying interactive plot window. Close it to proceed.")
    plt.show()

def process_file(model, device, target_sr, in_path, out_path):
    print(f"\n[INFO] Processing: {in_path}")
    try:
        wav = load_audio(in_path, target_sr)
        dur = wav.shape[-1] / target_sr
        print(f"[INFO] Track Length: {dur:.1f} seconds")

        t0 = time.time()
        enhanced = denoise_chunked(model, wav, device, chunk_len_sec=10)
        t1 = time.time()

        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

        noisy_np = wav.squeeze(0).numpy()
        clean_np = enhanced.squeeze(0).numpy()

        sf.write(out_path, clean_np, target_sr)
        print(f"[INFO] -> Saved Clean File: {out_path} (Inference took: {t1 - t0:.2f}s)")

        metrics = calculate_metrics(noisy_np, clean_np, target_sr)
        print(f"       [METRIC] Overall Level Change: {metrics['level_change_db']:.2f} dB")
        print(f"       [METRIC] Est. SNR Improvement: {metrics['snr_improvement_db']:.2f} dB")
        print(f"       [METRIC] Spectral Flatness: {metrics['spectral_flatness']:.3f}")

        plot_path = os.path.splitext(out_path)[0] + "_metrics.png"
        save_and_show_plots(noisy_np, clean_np, target_sr, plot_path, metrics)

    except Exception as e:
        print(f"[ERROR] Failed to process {in_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Memory-Safe Local Audio Denoiser.")
    # We remove 'required=True' so Visual Studio stops giving you errors
    parser.add_argument("--input", "-i", default="test_input.wav", help="Path to a noisy audio file")
    parser.add_argument("--output", "-o", default="cleaned_output.wav", help="Path to save the cleaned audio file")
    parser.add_argument("--model", "-m", default="dns48", choices=["dns64", "dns48", "master64"])
    parser.add_argument("--batch", action="store_true", help="Batch mode")
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.model, device)
    target_sr = model.sample_rate

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
            out_name = os.path.splitext(os.path.basename(f))[0] + "_clean.wav"
            process_file(model, device, target_sr, f, os.path.join(args.output, out_name))
    else:
        if not os.path.isfile(args.input):
            print(f"[ERROR] Input file not found: {args.input}. Make sure 'test_input.wav' is in your Metrics folder!")
            return
        process_file(model, device, target_sr, args.input, args.output)

    print("\n[DONE] All processing complete.")
    input("\nPress ENTER to close the terminal window...")

if __name__ == "__main__":
    main()
