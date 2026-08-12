"""
End-to-end noise-cancellation pipeline for PhysicsAttentionDenoiser.

Usage:
    python run_denoise.py --input noisy_input.wav --output output.wav \
        [--checkpoint model.pt] [--clean_ref clean.wav]

Produces:
    - <output>.wav                 : denoised audio
    - <output>_report.png          : waveform / spectrogram / metrics figure
    - metrics printed to stdout

IMPORTANT: PhysicsAttentionDenoiser, as defined in model.py, is initialized with
RANDOM weights. It has NOT been trained on any speech/noise data. Running it
as-is will NOT perform meaningful denoising -- it will apply an essentially
random time-frequency mask. You must load a trained checkpoint (--checkpoint)
for the metrics below to reflect real denoising performance. See the note
printed at the top of the run.
"""
import argparse

import numpy as np
import torch
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import PhysicsAttentionDenoiser
from metrics import compute_all_metrics, TARGET_SR


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def load_mono_16k(path):
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if sr != TARGET_SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    return x, sr


def run(input_path, output_path, checkpoint=None, clean_ref_path=None, device="cpu"):
    noisy, sr = load_mono_16k(input_path)

    model = PhysicsAttentionDenoiser()
    trained = False
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state)
        trained = True
    model.to(device).eval()

    if not trained:
        print("=" * 78)
        print("WARNING: no --checkpoint given. PhysicsAttentionDenoiser is running")
        print("with RANDOM, UNTRAINED weights. Output below is NOT real denoising")
        print("-- it is included only to prove the code runs end-to-end. Do not")
        print("present these numbers as model performance.")
        print("=" * 78)

    with torch.no_grad():
        wav_t = torch.from_numpy(noisy).unsqueeze(0).to(device)
        out_t = model(wav_t)
        denoised = out_t.squeeze(0).cpu().numpy()

    # peak-normalize output to avoid clipping on write, preserve relative level info separately
    peak = np.max(np.abs(denoised)) + 1e-9
    denoised_write = denoised / peak * 0.99 if peak > 1.0 else denoised
    sf.write(output_path, denoised_write.astype(np.float32), sr)

    clean_ref = None
    if clean_ref_path is not None:
        clean_ref, _ = load_mono_16k(clean_ref_path)

    metrics = compute_all_metrics(noisy, denoised, clean_ref, sr)
    metrics["trained_checkpoint_used"] = trained

    print("\n---- METRICS ----")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    n = min(len(noisy), len(denoised))
    noisy_m, denoised_m = noisy[:n], denoised[:n]
    _make_report_figure(noisy_m, denoised_m, sr, metrics,
                         out_png=output_path.rsplit(".", 1)[0] + "_report.png")

    return metrics


def _make_report_figure(noisy, denoised, sr, metrics, out_png, clean_ref=None):
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))

    t_n = np.arange(len(noisy)) / sr
    t_d = np.arange(len(denoised)) / sr
    axes[0, 0].plot(t_n, noisy, linewidth=0.5)
    axes[0, 0].set_title("Noisy input waveform")
    axes[0, 0].set_xlabel("Time (s)")

    axes[0, 1].plot(t_d, denoised, linewidth=0.5, color="darkorange")
    axes[0, 1].set_title("Model output waveform")
    axes[0, 1].set_xlabel("Time (s)")

    axes[1, 0].specgram(noisy, Fs=sr, NFFT=512, noverlap=256, cmap="magma")
    axes[1, 0].set_title("Noisy spectrogram")
    axes[1, 0].set_xlabel("Time (s)"); axes[1, 0].set_ylabel("Hz")

    axes[1, 1].specgram(denoised, Fs=sr, NFFT=512, noverlap=256, cmap="magma")
    axes[1, 1].set_title("Output spectrogram")
    axes[1, 1].set_xlabel("Time (s)"); axes[1, 1].set_ylabel("Hz")

    # metrics bar panel
    ax = axes[2, 0]
    labels, values = [], []
    labels.append("Level change\n(dB)"); values.append(metrics["overall_level_change_db"])
    labels.append(f"Est. SNR\nimprovement (dB)\n[{metrics['snr_improvement_mode']}]")
    values.append(metrics["estimated_snr_improvement_db"])
    if metrics["pesq"] is not None:
        labels.append("PESQ"); values.append(metrics["pesq"])
    if metrics["stoi"] is not None:
        labels.append("STOI"); values.append(metrics["stoi"])
    colors = ["seagreen" if v >= 0 else "firebrick" for v in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Summary metrics")
    ax.tick_params(axis="x", labelsize=8)

    ax2 = axes[2, 1]
    sf_labels = ["Noisy", "Denoised"]
    sf_vals = [metrics["spectral_flatness_noisy"], metrics["spectral_flatness_denoised"]]
    ax2.bar(sf_labels, sf_vals, color=["slategray", "darkorange"])
    ax2.set_ylim(0, 1)
    ax2.set_title("Spectral flatness (0=tonal, 1=noise-like)")

    status = "TRAINED CHECKPOINT" if metrics["trained_checkpoint_used"] else "UNTRAINED / RANDOM WEIGHTS"
    fig.suptitle(f"PhysicsAttentionDenoiser report  —  {status}", fontsize=13,
                 color=("black" if metrics["trained_checkpoint_used"] else "red"))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--clean_ref", default=None,
                     help="Optional clean reference wav for ground-truth PESQ/STOI/SNR")
    args = ap.parse_args()
    run(args.input, args.output, args.checkpoint, args.clean_ref)
