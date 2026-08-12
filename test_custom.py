"""
Test a trained PhysicsAttentionDenoiser checkpoint on YOUR custom-recorded
speech -- separate from Valentini validation, which happens automatically
during train.py.

Two modes, auto-detected per file:

1. Blind mode (most likely what you have): just noisy real-world recordings,
   no matched clean reference. Runs overall level change, blind SNR-floor
   improvement estimate, and spectral flatness. PESQ/STOI are skipped (they
   are defined-only relative to a clean reference and cannot be computed
   blind -- any blind PESQ/STOI number would be fabricated).

2. Reference mode: if you also have a clean recording of the same utterance
   (e.g. recorded quiet-room clean take + separately recorded/mixed noisy
   take), pass --clean_dir with matching filenames and you get the full
   metric set including PESQ/STOI/ground-truth SNR improvement, same as
   run_denoise.py.

Usage:
    python test_custom.py --checkpoint checkpoints/best.pt \
        --noisy_dir ./my_recordings \
        --out_dir ./custom_test_results \
        [--clean_dir ./my_recordings_clean]   # optional, enables PESQ/STOI

Produces, per input file <name>.wav:
    <out_dir>/denoised/<name>.wav
    <out_dir>/reports/<name>_report.png

Plus one aggregate summary across all files:
    <out_dir>/summary.csv
    <out_dir>/summary.png
"""
import argparse
import csv
import os

import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import PhysicsAttentionDenoiser
from metrics import compute_all_metrics, TARGET_SR


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


def make_per_file_report(noisy, denoised, sr, metrics, fname, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].specgram(noisy, Fs=sr, NFFT=512, noverlap=256, cmap="magma")
    axes[0, 0].set_title("Noisy input"); axes[0, 0].set_ylabel("Hz")

    axes[0, 1].specgram(denoised, Fs=sr, NFFT=512, noverlap=256, cmap="magma")
    axes[0, 1].set_title("Denoised output")

    t = np.arange(len(noisy)) / sr
    axes[1, 0].plot(t, noisy, linewidth=0.4)
    axes[1, 0].plot(np.arange(len(denoised)) / sr, denoised, linewidth=0.4, alpha=0.7)
    axes[1, 0].set_title("Waveform overlay (blue=noisy, orange=denoised)")
    axes[1, 0].set_xlabel("Time (s)")

    ax = axes[1, 1]
    labels = ["Level\nchange (dB)",
              f"Est. SNR\nimprove (dB)\n[{metrics['snr_improvement_mode']}]"]
    values = [metrics["overall_level_change_db"], metrics["estimated_snr_improvement_db"]]
    if metrics["pesq"] is not None:
        labels.append("PESQ"); values.append(metrics["pesq"])
    if metrics["stoi"] is not None:
        labels.append("STOI"); values.append(metrics["stoi"])
    colors = ["seagreen" if v >= 0 else "firebrick" for v in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Metrics")

    fig.suptitle(fname)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def make_summary_figure(rows, out_png):
    names = [r["file"] for r in rows]
    level = [r["overall_level_change_db"] for r in rows]
    snr = [r["estimated_snr_improvement_db"] for r in rows]
    have_pesq = [r["pesq"] for r in rows if r["pesq"] is not None]
    have_stoi = [r["stoi"] for r in rows if r["stoi"] is not None]

    n_panels = 2 + (1 if have_pesq else 0) + (1 if have_stoi else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    axes[0].bar(range(len(names)), level, color="steelblue")
    axes[0].set_title("Overall level change (dB) per file")
    axes[0].set_xticks(range(len(names))); axes[0].set_xticklabels(names, rotation=75, fontsize=7)

    axes[1].bar(range(len(names)), snr, color="darkorange")
    axes[1].set_title("Estimated SNR improvement (dB) per file")
    axes[1].set_xticks(range(len(names))); axes[1].set_xticklabels(names, rotation=75, fontsize=7)

    idx = 2
    if have_pesq:
        vals = [r["pesq"] if r["pesq"] is not None else np.nan for r in rows]
        axes[idx].bar(range(len(names)), vals, color="seagreen")
        axes[idx].set_title(f"PESQ per file (mean {np.nanmean(vals):.2f})")
        axes[idx].set_xticks(range(len(names))); axes[idx].set_xticklabels(names, rotation=75, fontsize=7)
        idx += 1
    if have_stoi:
        vals = [r["stoi"] if r["stoi"] is not None else np.nan for r in rows]
        axes[idx].bar(range(len(names)), vals, color="mediumpurple")
        axes[idx].set_title(f"STOI per file (mean {np.nanmean(vals):.2f})")
        axes[idx].set_xticks(range(len(names))); axes[idx].set_xticklabels(names, rotation=75, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--noisy_dir", required=True, help="folder of your custom noisy recordings")
    ap.add_argument("--clean_dir", default=None,
                     help="optional: folder of matching clean recordings (same filenames) "
                          "to enable PESQ/STOI/ground-truth SNR")
    ap.add_argument("--out_dir", default="./custom_test_results")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(args.out_dir, "denoised"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "reports"), exist_ok=True)

    model = PhysicsAttentionDenoiser().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    files = sorted(f for f in os.listdir(args.noisy_dir) if f.lower().endswith(".wav"))
    if not files:
        raise RuntimeError(f"No .wav files found in {args.noisy_dir}")
    print(f"Found {len(files)} recordings to test.")

    rows = []
    for fname in files:
        noisy, sr = load_mono_16k(os.path.join(args.noisy_dir, fname))

        with torch.no_grad():
            wav_t = torch.from_numpy(noisy).unsqueeze(0).to(device)
            denoised = model(wav_t).squeeze(0).cpu().numpy()

        peak = np.max(np.abs(denoised)) + 1e-9
        write_wav = denoised / peak * 0.99 if peak > 1.0 else denoised
        sf.write(os.path.join(args.out_dir, "denoised", fname), write_wav.astype(np.float32), sr)

        clean_ref = None
        clean_path = os.path.join(args.clean_dir, fname) if args.clean_dir else None
        if clean_path and os.path.exists(clean_path):
            clean_ref, _ = load_mono_16k(clean_path)
        elif args.clean_dir:
            print(f"  [{fname}: no matching clean file found in --clean_dir, running blind metrics]")

        metrics = compute_all_metrics(noisy, denoised, clean_ref, sr)
        metrics["file"] = fname
        rows.append(metrics)

        n = min(len(noisy), len(denoised))
        make_per_file_report(noisy[:n], denoised[:n], sr, metrics, fname,
                              os.path.join(args.out_dir, "reports", fname.rsplit(".", 1)[0] + "_report.png"))

        print(f"{fname}: level_change={metrics['overall_level_change_db']:.2f} dB | "
              f"snr_improve={metrics['estimated_snr_improvement_db']:.2f} dB "
              f"[{metrics['snr_improvement_mode']}] | "
              f"pesq={metrics['pesq']} | stoi={metrics['stoi']}")

    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["file", "overall_level_change_db", "estimated_snr_improvement_db",
                      "snr_improvement_mode", "spectral_flatness_noisy",
                      "spectral_flatness_denoised", "pesq", "stoi", "has_clean_ref"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    make_summary_figure(rows, os.path.join(args.out_dir, "summary.png"))

    level_vals = [r["overall_level_change_db"] for r in rows]
    snr_vals = [r["estimated_snr_improvement_db"] for r in rows]
    print("\n---- AGGREGATE across all files ----")
    print(f"Mean level change: {np.mean(level_vals):.2f} dB")
    print(f"Mean estimated SNR improvement: {np.mean(snr_vals):.2f} dB")
    pesq_vals = [r["pesq"] for r in rows if r["pesq"] is not None]
    stoi_vals = [r["stoi"] for r in rows if r["stoi"] is not None]
    if pesq_vals:
        print(f"Mean PESQ: {np.mean(pesq_vals):.3f} (n={len(pesq_vals)})")
    if stoi_vals:
        print(f"Mean STOI: {np.mean(stoi_vals):.3f} (n={len(stoi_vals)})")
    if not pesq_vals and not stoi_vals:
        print("No PESQ/STOI -- pass --clean_dir with matching clean recordings to get these.")
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {os.path.join(args.out_dir, 'summary.png')}")


if __name__ == "__main__":
    main()
