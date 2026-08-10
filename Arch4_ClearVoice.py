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
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from clearvoice import ClearVoice

# Force a non-interactive file-writing engine so it never relies on broken window popups
import matplotlib
matplotlib.use('Agg') 

def calculate_metrics(noisy_np, clean_np):
    """Calculates engineering metrics comparing noisy and clean audio arrays."""
    eps = 1e-10
    
    # Ensure both arrays are completely flat and matched in size
    min_len = min(len(noisy_np), len(clean_np))
    noisy_np = noisy_np[:min_len]
    clean_np = clean_np[:min_len]
    
    # 1. Total Noise Reduction (dB Attenuation)
    power_noisy = np.mean(noisy_np ** 2)
    power_clean = np.mean(clean_np ** 2)
    noise_reduction_db = 10 * np.log10((power_noisy + eps) / (power_clean + eps))
    
    # 2. Estimated Residual Profile (Noisy minus Clean to isolate removed noise)
    removed_noise = noisy_np - clean_np
    power_removed = np.mean(removed_noise ** 2)
    snr_imp_db = 10 * np.log10((power_clean + eps) / (power_removed + eps))

    # 3. Total Harmonic Distortion (THD) of clean output via Fast Fourier Transform
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
        f"• Broad Noise Attenuation: {metrics['noise_reduction_db']:.2f} dB\n"
        f"• Estimated SNR Improvement: {metrics['snr_improvement_db']:.2f} dB\n"
        f"• Total Harmonic Distortion (THD): {metrics['thd_percent']:.2f}%\n"
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
        metrics = calculate_metrics(noisy_signal, clean_signal)
        print(f"       [METRIC] Noise Reduction: {metrics['noise_reduction_db']:.2f} dB")
        print(f"       [METRIC] SNR Gain: {metrics['snr_improvement_db']:.2f} dB")
        print(f"       [METRIC] Output THD: {metrics['thd_percent']:.2f}%")

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
