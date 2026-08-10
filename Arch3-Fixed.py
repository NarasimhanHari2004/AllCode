#!/usr/bin/env python3
"""
Advanced Architectural Audio Denoiser (Premium Production Build)
================================================================
Architecture-3: Introduces a dynamic Dry-Wet Blending Valve framework
(90% Clean + 10% Original Noisy Atmosphere) to eliminate robotic artifacts.

Optimized for Microsoft Visual Studio. Computes ANC performance metrics and
automatically forces Windows to open the generated dashboard image file.
"""

import sys
import torch

# Guarded patch: only fires if set_default_tensor_type doesn't exist at all,
# which in practice means this branch is dead on any real torch install.
if not hasattr(torch, 'set_default_tensor_type'):
    torch.set_default_tensor_type = lambda *args, **kwargs: None

import argparse
import os
import time
import glob
import librosa
import librosa.display  # FIX: required explicitly — librosa.display.specshow()
                         # is not guaranteed to be available just from `import librosa`.
                         # Without this the script crashes with AttributeError right
                         # at the plotting step. Same recurring bug as Architecture-1/2.
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# Force a non-interactive file-writing engine so it never relies on broken window popups
import matplotlib
matplotlib.use('Agg')

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

def get_device():
    """Pick the fastest available device and configure it for max throughput."""
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
    """Downloads and loads the pretrained Demucs denoiser model weights."""
    from denoiser.pretrained import dns64, dns48, master64

    model_map = {"dns64": dns64, "dns48": dns48, "master64": master64}
    if model_name not in model_map:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(model_map.keys())}")

    print(f"[INFO] Loading pretrained model: {model_name} ...")
    model = model_map[model_name](pretrained=True)
    model = model.to(device)
    model.eval()
    return model

def load_audio(path, target_sr):
    """Loads an audio file safely using librosa to completely bypass torchcodec engine bugs."""
    data, sr = librosa.load(path, sr=target_sr)
    wav = torch.from_numpy(data)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav


def denoise_overlap_add(model, wav, device, chunk_len_sec=8, overlap_sec=2):
    """
    ADVANCED ARCHITECTURE: Processes audio using overlapping segments
    and uses a linear crossfade window to blend them back together seamlessly.
    """
    sr = model.sample_rate
    chunk_samples = chunk_len_sec * sr
    overlap_samples = overlap_sec * sr
    step_samples = chunk_samples - overlap_samples

    # FIX: guard against a zero/negative step (e.g. overlap_sec >= chunk_len_sec),
    # which would otherwise crash range() or silently misbehave.
    if step_samples <= 0:
        raise ValueError(
            f"overlap_sec ({overlap_sec}s) must be smaller than "
            f"chunk_len_sec ({chunk_len_sec}s) so that step_samples stays positive."
        )

    total_samples = wav.shape[-1]

    # Initialize output array and tracking weight map
    output_audio = np.zeros(total_samples, dtype=np.float32)
    weight_mask = np.zeros(total_samples, dtype=np.float32)

    # Create a linear crossfade window matrix
    window = np.ones(chunk_samples, dtype=np.float32)
    if overlap_samples > 0:
        fade_in = np.linspace(0, 1, overlap_samples, dtype=np.float32)
        fade_out = np.linspace(1, 0, overlap_samples, dtype=np.float32)
        window[:overlap_samples] = fade_in
        window[-overlap_samples:] = fade_out

    # Sliding Window Loop
    for start in range(0, total_samples, step_samples):
        end = start + chunk_samples

        if start >= total_samples:
            break

        # Extract chunk and pad if it overshoots the end of the file
        pad_len = 0
        if end > total_samples:
            pad_len = end - total_samples
            chunk = wav[:, start:total_samples]
            chunk = torch.nn.functional.pad(chunk, (0, pad_len))
        else:
            chunk = wav[:, start:end]

        # Push to NVIDIA GPU VRAM
        chunk = chunk.to(device)

        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out_chunk = model(chunk.unsqueeze(0))
            else:
                out_chunk = model(chunk.unsqueeze(0))

        # Immediately extract to CPU and convert to flat numpy array
        out_chunk_np = out_chunk.squeeze(0).squeeze(0).detach().cpu().numpy()

        # Truncate padding if it was added
        if pad_len > 0:
            out_chunk_np = out_chunk_np[:-pad_len]
            current_chunk_len = total_samples - start
        else:
            current_chunk_len = chunk_samples

        # Apply the blending window matrix
        active_window = window[:current_chunk_len]

        # Accumulate the segments into our output map
        output_audio[start:start+current_chunk_len] += out_chunk_np * active_window
        weight_mask[start:start+current_chunk_len] += active_window

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Normalize by the weight mask to ensure perfectly uniform volume balance
    weight_mask[weight_mask == 0] = 1.0
    final_audio = output_audio / weight_mask

    return torch.from_numpy(final_audio).unsqueeze(0)


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
    """Generates visualization dashboard panel and writes it directly to disk storage."""
    plt.close('all')

    fig = plt.figure(figsize=(14, 9))

    # Title adjusted exactly to denote Architecture-3 parameters
    fig.suptitle("Architecture-3: Demucs Speech Denoising Pipeline for ANC with OLA and 90-10 valves", fontsize=16, fontweight='bold')

    time_axis = np.linspace(0, len(noisy_np) / sr, num=len(noisy_np))

    # 1. Waveforms
    plt.subplot(3, 2, 1)
    plt.plot(time_axis, noisy_np, color='crimson', alpha=0.7)
    plt.title("Noisy Signal Waveform")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(3, 2, 2)
    plt.plot(time_axis, clean_np, color='dodgerblue', alpha=0.7)
    plt.title("Enhanced (Blended) Waveform")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.5)

    # 2. Spectrograms
    plt.subplot(3, 2, 3)
    D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_np)), ref=np.max)
    librosa.display.specshow(D_noisy, sr=sr, x_axis='time', y_axis='linear', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Noisy Spectrogram")

    plt.subplot(3, 2, 4)
    D_clean = librosa.amplitude_to_db(np.abs(librosa.stft(clean_np)), ref=np.max)
    librosa.display.specshow(D_clean, sr=sr, x_axis='time', y_axis='linear', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Blended Clean Spectrogram")

    # 3. Text Panel
    plt.subplot(3, 1, 3)
    plt.axis('off')
    text_str = (
        f"PERFORMANCE SUMMARY METRICS (ARCHITECTURE-3 - 90/10 BLEND):\n"
        f"----------------------------------------------------------------------\n"
        f"• Overall Level Change: {metrics['level_change_db']:.2f} dB\n"
        f"• Estimated SNR Improvement: {metrics['snr_improvement_db']:.2f} dB\n"
        f"• Spectral Flatness (denoised): {metrics['spectral_flatness']:.3f}\n"
    )
    plt.text(0.05, 0.3, text_str, fontsize=13, family='monospace',
             bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=1'))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(plot_path, dpi=150)
    print(f"[INFO] -> Architecture-3 Metrics Dashboard Saved: {plot_path}")

    # Force Desktop Open Workaround
    try:
        os.startfile(plot_path)
        print("[INFO] Launched default system image viewer to display charts.")
    except Exception as e:
        print(f"[WARN] Script couldn't open image viewer automatically: {str(e)}")

def process_file(model, device, target_sr, in_path, out_path):
    """Orchestrates individual file lifecycles with advanced artifact suppression."""
    print(f"\n[INFO] Processing: {in_path}")
    try:
        # Load the original noisy track
        original_wav = load_audio(in_path, target_sr)
        dur = original_wav.shape[-1] / target_sr
        print(f"[INFO] Track Length: {dur:.1f} seconds")

        t0 = time.time()
        # Run the deep learning denoiser chunk loop
        enhanced_tensor = denoise_overlap_add(model, original_wav, device, chunk_len_sec=8, overlap_sec=2)
        t1 = time.time()

        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

        # Convert tensors to clean numpy formats
        clean_np = enhanced_tensor.squeeze(0).numpy()
        noisy_np = original_wav.squeeze(0).numpy()

        # ======================================================================
        # ADVANCED ARTIFACT SUPPRESSION: THE DRY-WET BLEND VALVE
        # ======================================================================
        # 0.90 (90% AI Cleaned Signal) + 0.10 (10% Original Noisy Atmosphere)
        final_blend = (clean_np * 0.90) + (noisy_np * 0.10)

        # Final peak normalization safety check
        max_peak = np.max(np.abs(final_blend))
        if max_peak > 1.0:
            final_blend /= max_peak

        # Save the file using soundfile
        sf.write(out_path, final_blend, target_sr)
        print(f"[INFO] -> Saved Premium Clean File: {out_path} (Inference took: {t1 - t0:.2f}s)")

        # Calculate diagnostics using the final blended output array
        metrics = calculate_metrics(noisy_np, final_blend, target_sr)
        print(f"       [METRIC] Overall Level Change: {metrics['level_change_db']:.2f} dB")
        print(f"       [METRIC] Est. SNR Improvement: {metrics['snr_improvement_db']:.2f} dB")
        print(f"       [METRIC] Spectral Flatness: {metrics['spectral_flatness']:.3f}")

        # FIX: Force base path into a strict string format before appending extension text
        base_name, _ = os.path.splitext(out_path)
        plot_path = os.path.abspath(str(base_name) + "_architecture3_metrics.png")

        # Run visualization dashboard pipeline
        save_and_show_plots(noisy_np, final_blend, target_sr, plot_path, metrics)

    except Exception as e:
        print(f"[ERROR] Failed to process {in_path}: {str(e)}")



def main():
    parser = argparse.ArgumentParser(description="Advanced Local Audio Denoiser.")
    # Fallbacks protect script execution within Visual Studio isolated environments
    parser.add_argument("--input", "-i", default="test_input.wav", help="Path to a noisy audio file")
    parser.add_argument("--output", "-o", default="cleaned_output_blend_clean.wav", help="Path to save the cleaned audio file")
    parser.add_argument("--model", "-m", default="dns48", choices=["dns64", "dns48", "master64"])
    parser.add_argument("--batch", action="store_true", help="Batch mode")
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.model, device)
    target_sr = model.sample_rate

    if args.batch:
        if not os.path.isdir(args.input):
            print(f"[ERROR] Input folder not found: {args.input}")
            return
        files = [f for f in glob.glob(os.path.join(args.input, "*")) if f.lower().endswith(AUDIO_EXTENSIONS)]
        os.makedirs(args.output, exist_ok=True)
        for f in files:
            # FIX: os.path.splitext() returns a (root, ext) tuple, not a string.
            # The original code did `os.path.splitext(...) + "_architecture3_clean.wav"`,
            # which raises TypeError: can only concatenate tuple (not "str") to tuple.
            # Need to take [0] (the root) before concatenating, same as the other scripts.
            out_name = os.path.splitext(os.path.basename(f))[0] + "_architecture3_clean.wav"
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
