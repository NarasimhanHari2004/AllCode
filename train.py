"""
Train PhysicsAttentionDenoiser on the Valentini (VoiceBank-DEMAND) dataset,
or any dataset laid out the same way: two directories of clean wavs and two
directories of noisy wavs, with matching filenames between clean/noisy pairs
(this is how Valentini ships, e.g. clean_trainset_28spk_wav/p226_001.wav <->
noisy_trainset_28spk_wav/p226_001.wav).

Usage:
    python train.py \
        --train_clean /path/to/clean_trainset_28spk_wav \
        --train_noisy /path/to/noisy_trainset_28spk_wav \
        --test_clean  /path/to/clean_testset_wav \
        --test_noisy  /path/to/noisy_testset_wav \
        --epochs 60 --batch_size 8 --segment_seconds 2.0 \
        --out_dir ./checkpoints

Produces:
    <out_dir>/best.pt        : best checkpoint by validation STOI
    <out_dir>/last.pt        : most recent checkpoint
    <out_dir>/history.json   : per-epoch train/val metrics
    <out_dir>/curves.png     : loss / PESQ / STOI curves over training

Then evaluate/run the trained model with:
    python run_denoise.py --input noisy.wav --output out.wav \
        --checkpoint checkpoints/best.pt --clean_ref clean.wav
"""
import argparse
import json
import os
import random
import time
import warnings

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import PhysicsAttentionDenoiser

TARGET_SR = 16000


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class PairedDenoiseDataset(Dataset):
    """Pairs clean/noisy wavs by filename. Random-crops (train) or
    center-crops/pads (eval) to a fixed segment length."""

    def __init__(self, clean_dir, noisy_dir, segment_len, train=True, sr=TARGET_SR):
        self.sr = sr
        self.segment_len = segment_len
        self.train = train

        clean_files = {f for f in os.listdir(clean_dir) if f.lower().endswith(".wav")}
        noisy_files = {f for f in os.listdir(noisy_dir) if f.lower().endswith(".wav")}
        common = sorted(clean_files & noisy_files)
        if not common:
            raise RuntimeError(
                f"No matching filenames between {clean_dir} and {noisy_dir}. "
                "Valentini pairs clean/noisy files by identical basename -- "
                "double check these are the right directories."
            )
        missing_c = noisy_files - clean_files
        missing_n = clean_files - noisy_files
        if missing_c or missing_n:
            warnings.warn(
                f"{len(missing_c)} noisy files with no clean match, "
                f"{len(missing_n)} clean files with no noisy match -- skipped."
            )

        self.clean_dir, self.noisy_dir = clean_dir, noisy_dir
        self.files = common

    def __len__(self):
        return len(self.files)

    def _load(self, path):
        x, sr = sf.read(path, always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != self.sr:
            import librosa
            x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=self.sr)
        return x.astype(np.float32)

    def __getitem__(self, idx):
        fname = self.files[idx]
        clean = self._load(os.path.join(self.clean_dir, fname))
        noisy = self._load(os.path.join(self.noisy_dir, fname))
        n = min(len(clean), len(noisy))
        clean, noisy = clean[:n], noisy[:n]

        L = self.segment_len
        if n >= L:
            if self.train:
                start = random.randint(0, n - L)
            else:
                start = (n - L) // 2
            clean = clean[start:start + L]
            noisy = noisy[start:start + L]
        else:
            pad = L - n
            clean = np.pad(clean, (0, pad))
            noisy = np.pad(noisy, (0, pad))

        return torch.from_numpy(noisy), torch.from_numpy(clean)


# --------------------------------------------------------------------------- #
# Loss: time-domain L1 + log-magnitude spectral L1 (standard, stable combo for
# mask-based speech enhancement -- avoids needing extra loss-specific deps)
# --------------------------------------------------------------------------- #
class DenoiseLoss(nn.Module):
    def __init__(self, stft_module, spectral_weight=1.0, eps=1e-6):
        super().__init__()
        self.stft = stft_module
        self.spectral_weight = spectral_weight
        self.eps = eps

    def forward(self, est, target):
        time_loss = nn.functional.l1_loss(est, target)

        est_spec = self.stft.STFT(est)
        tgt_spec = self.stft.STFT(target)
        est_mag = torch.log(est_spec.abs() + self.eps)
        tgt_mag = torch.log(tgt_spec.abs() + self.eps)
        spec_loss = nn.functional.l1_loss(est_mag, tgt_mag)

        return time_loss + self.spectral_weight * spec_loss, time_loss.item(), spec_loss.item()


# --------------------------------------------------------------------------- #
# Metrics (lightweight, for per-epoch validation logging)
# --------------------------------------------------------------------------- #
def si_sdr(est, target, eps=1e-8):
    target = target - target.mean()
    est = est - est.mean()
    s_target = (torch.sum(est * target) / (torch.sum(target ** 2) + eps)) * target
    e_noise = est - s_target
    return 10 * torch.log10((torch.sum(s_target ** 2) + eps) / (torch.sum(e_noise ** 2) + eps))


def try_pesq_stoi(clean_np, est_np, sr=TARGET_SR):
    pesq_v, stoi_v = None, None
    try:
        from pesq import pesq
        pesq_v = pesq(sr, clean_np, est_np, "wb")
    except Exception:
        pass
    try:
        from pystoi import stoi as stoi_fn
        stoi_v = stoi_fn(clean_np, est_np, sr, extended=False)
    except Exception:
        pass
    return pesq_v, stoi_v


# --------------------------------------------------------------------------- #
# Train / validate
# --------------------------------------------------------------------------- #
def validate(model, loader, device, max_batches=None):
    model.eval()
    si_sdrs, pesqs, stois = [], [], []
    with torch.no_grad():
        for i, (noisy, clean) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            noisy, clean = noisy.to(device), clean.to(device)
            est = model(noisy)
            for b in range(est.shape[0]):
                si_sdrs.append(si_sdr(est[b], clean[b]).item())
                p, s = try_pesq_stoi(clean[b].cpu().numpy(), est[b].cpu().numpy())
                if p is not None:
                    pesqs.append(p)
                if s is not None:
                    stois.append(s)
    model.train()
    return {
        "si_sdr": float(np.mean(si_sdrs)) if si_sdrs else None,
        "pesq": float(np.mean(pesqs)) if pesqs else None,
        "stoi": float(np.mean(stois)) if stois else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_clean", required=True)
    ap.add_argument("--train_noisy", required=True)
    ap.add_argument("--test_clean", required=True)
    ap.add_argument("--test_noisy", required=True)
    ap.add_argument("--out_dir", default="./checkpoints")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4,
                     help="default is conservative for ~4GB VRAM cards; raise if you have headroom")
    ap.add_argument("--segment_seconds", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--val_batches", type=int, default=20,
                     help="cap validation batches per epoch (PESQ is slow)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda / cpu / auto (default)")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--amp", action="store_true", default=True,
                     help="mixed precision training (default on; matters a lot on 4GB cards)")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--grad_accum_steps", type=int, default=4,
                     help="accumulate gradients over N steps for a larger effective "
                          "batch size without more VRAM, e.g. batch_size=4 --grad_accum_steps=4 "
                          "behaves like batch_size=16")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB")
    use_amp = args.amp and device == "cuda"
    print(f"Mixed precision (AMP): {use_amp} | grad_accum_steps: {args.grad_accum_steps}")
    os.makedirs(args.out_dir, exist_ok=True)

    seg_len = int(args.segment_seconds * TARGET_SR)
    train_ds = PairedDenoiseDataset(args.train_clean, args.train_noisy, seg_len, train=True)
    val_ds = PairedDenoiseDataset(args.test_clean, args.test_noisy, seg_len, train=False)
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    pin_mem = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_mem)

    model = PhysicsAttentionDenoiser().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    loss_fn = DenoiseLoss(model.stft)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_stoi = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running, running_t, running_s, nb = 0.0, 0.0, 0.0, 0
        opt.zero_grad()
        for step, (noisy, clean) in enumerate(train_loader):
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            try:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    est = model(noisy)
                    loss, tl, sl = loss_fn(est, clean)
                    loss_scaled = loss / args.grad_accum_steps

                scaler.scale(loss_scaled).backward()

                if (step + 1) % args.grad_accum_steps == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad()
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(f"  [OOM at step {step}, batch skipped -- lower --batch_size, "
                      f"--segment_seconds, or raise --grad_accum_steps instead]")
                continue

            running += loss.item(); running_t += tl; running_s += sl; nb += 1

        if device == "cuda":
            torch.cuda.empty_cache()

        train_loss = running / max(nb, 1)
        epoch_rec = {"epoch": epoch, "train_loss": train_loss,
                     "train_time_loss": running_t / max(nb, 1),
                     "train_spec_loss": running_s / max(nb, 1),
                     "elapsed_sec": time.time() - t0}

        if epoch % args.val_every == 0:
            val_metrics = validate(model, val_loader, device, max_batches=args.val_batches)
            epoch_rec.update({f"val_{k}": v for k, v in val_metrics.items()})
            if val_metrics["stoi"] is not None:
                sched.step(val_metrics["stoi"])
                if val_metrics["stoi"] > best_stoi:
                    best_stoi = val_metrics["stoi"]
                    torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))
                    epoch_rec["saved_best"] = True

        torch.save(model.state_dict(), os.path.join(args.out_dir, "last.pt"))
        history.append(epoch_rec)
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        print(f"epoch {epoch:3d}/{args.epochs} | loss {train_loss:.4f} | "
              f"val_si_sdr {epoch_rec.get('val_si_sdr')} | "
              f"val_pesq {epoch_rec.get('val_pesq')} | "
              f"val_stoi {epoch_rec.get('val_stoi')} | "
              f"{epoch_rec['elapsed_sec']:.1f}s")

    _plot_curves(history, os.path.join(args.out_dir, "curves.png"))
    print(f"Done. Best checkpoint: {os.path.join(args.out_dir, 'best.pt')} (val_stoi={best_stoi:.3f})")


def _plot_curves(history, out_png):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history])
    axes[0].set_title("Train loss"); axes[0].set_xlabel("epoch")

    val_pesq = [h.get("val_pesq") for h in history]
    if any(v is not None for v in val_pesq):
        axes[1].plot(epochs, val_pesq)
    axes[1].set_title("Val PESQ"); axes[1].set_xlabel("epoch")

    val_stoi = [h.get("val_stoi") for h in history]
    if any(v is not None for v in val_stoi):
        axes[2].plot(epochs, val_stoi)
    axes[2].set_title("Val STOI"); axes[2].set_xlabel("epoch")

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
