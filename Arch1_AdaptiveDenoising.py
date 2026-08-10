#!/usr/bin/env python3
"""
Visual Studio Tailored Speech Denoising Script
=============================================
Optimized for Microsoft Visual Studio (Purple Logo).
Calculates ANC metrics and handles plot rendering safely within the IDE.
"""

import sys
import torch

# Patch: Bypasses an outdated legacy tensor function call in facebookresearch/denoiser
torch.set_default_tensor_type = lambda *args, **kwargs: None  

import argparse
import os
import time
import glob
import librosa
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
    model = model_map[model_name](pretrained=True)
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

def calculate_metrics(noisy_np, clean_np):
    eps = 1e-10
    power_noisy = np.mean(noisy_np ** 2)
    power_clean = np.mean(clean_np ** 2)
    noise_reduction_db = 10 * np.log10((power_noisy + eps) / (power_clean + eps))
    
    removed_noise = noisy_np - clean_np
    power_removed = np.mean(removed_noise ** 2)
    snr_imp_db = 10 * np.log10((power_clean + eps) / (power_removed + eps))

    fft_clean = np.abs(np.fft.rfft(clean_np))
    idx_fundamental = np.argmax(fft_clean)
    fundamental_amp = fft_clean[idx_fundamental]
    
    harmonics_sum = np.sum(fft_clean**2) - (fundamental_amp**2)
    thd = np.sqrt(max(0, harmonics_sum)) / (fundamental_amp + eps)
    thd_percentage = min(thd * 100, 100.0)
    
    return {
        "noise_reduction_db": noise_reduction_db,
        "snr_improvement_db": snr_imp_db,
        "thd_percent": thd_percentage
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
        f"• Broad Noise Attenuation: {metrics['noise_reduction_db']:.2f} dB\n"
        f"• Estimated SNR Improvement: {metrics['snr_improvement_db']:.2f} dB\n"
        f"• Total Harmonic Distortion (THD): {metrics['thd_percent']:.2f}%\n"
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
        
        metrics = calculate_metrics(noisy_np, clean_np)
        print(f"       [METRIC] Noise Reduction: {metrics['noise_reduction_db']:.2f} dB")
        print(f"       [METRIC] SNR Gain: {metrics['snr_improvement_db']:.2f} dB")
        print(f"       [METRIC] Output THD: {metrics['thd_percent']:.2f}%")

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

