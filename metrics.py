"""
Shared metric functions for the PhysicsAttentionDenoiser pipeline.
Used by run_denoise.py (single file) and test_custom.py (batch of your own
recordings). Kept in one place so the two scripts can't drift out of sync.
"""
import warnings

import numpy as np

TARGET_SR = 16000


def rms(x, eps=1e-12):
    return np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps)


def db(x, eps=1e-12):
    return 20.0 * np.log10(x + eps)


def overall_level_change_db(noisy, denoised):
    """Change in overall RMS level, in dB, output vs input."""
    return db(rms(denoised)) - db(rms(noisy))


def estimate_snr_db(signal, noise, eps=1e-12):
    """Simple time-domain SNR estimate: 10*log10(P_signal / P_noise)."""
    p_s = np.mean(signal.astype(np.float64) ** 2)
    p_n = np.mean(noise.astype(np.float64) ** 2)
    return 10.0 * np.log10((p_s + eps) / (p_n + eps))


def _noise_floor_power(x, frame=320, hop=160, percentile=10):
    n_frames = 1 + (len(x) - frame) // hop
    powers = np.array([
        np.mean(x[i * hop: i * hop + frame].astype(np.float64) ** 2)
        for i in range(max(n_frames, 1))
    ])
    return np.percentile(powers, percentile)


def estimate_snr_improvement_db(noisy, denoised, clean_ref=None):
    """
    If a clean reference is available: SNR_out - SNR_in, both computed against
    the true clean signal (the rigorous definition) -> mode 'ground_truth'.

    If NOT available (your custom recordings, real-world deployment case):
    falls back to a blind proxy using a minimum-statistics noise-floor
    estimate -- i.e. how much the estimated noise floor power dropped.
    This is an ESTIMATE, not a ground-truth SNR improvement, and is labeled
    'blind_estimate' so it's never confused with the real thing.
    """
    n = min(len(noisy), len(denoised))
    noisy, denoised = noisy[:n], denoised[:n]

    if clean_ref is not None:
        clean_ref = clean_ref[:n]
        noise_in = noisy - clean_ref
        noise_out = denoised - clean_ref
        snr_in = estimate_snr_db(clean_ref, noise_in)
        snr_out = estimate_snr_db(clean_ref, noise_out)
        return snr_out - snr_in, snr_in, snr_out, "ground_truth"

    floor_in = _noise_floor_power(noisy)
    floor_out = _noise_floor_power(denoised)
    improvement = db(floor_in) - db(floor_out)  # positive = noise floor went down
    return improvement, db(floor_in), db(floor_out), "blind_estimate"


def spectral_flatness(x, frame=1024, hop=512, eps=1e-12):
    """Wiener entropy: geometric_mean(power) / arithmetic_mean(power), per frame,
    averaged. 0 = tonal/peaky, 1 = flat/noise-like. Lower is generally better
    after denoising (less residual broadband noise), assuming speech content."""
    n_frames = max(1 + (len(x) - frame) // hop, 1)
    window = np.hanning(frame)
    vals = []
    for i in range(n_frames):
        seg = x[i * hop: i * hop + frame]
        if len(seg) < frame:
            seg = np.pad(seg, (0, frame - len(seg)))
        spec = np.abs(np.fft.rfft(seg * window)) ** 2 + eps
        gm = np.exp(np.mean(np.log(spec)))
        am = np.mean(spec)
        vals.append(gm / am)
    return float(np.mean(vals))


def safe_pesq(clean_ref, denoised, sr=TARGET_SR):
    try:
        from pesq import pesq
        mode = "wb" if sr == 16000 else "nb"
        return pesq(sr, clean_ref, denoised, mode)
    except Exception as e:
        warnings.warn(f"PESQ failed: {e}")
        return None


def safe_stoi(clean_ref, denoised, sr=TARGET_SR):
    try:
        from pystoi import stoi
        return stoi(clean_ref, denoised, sr, extended=False)
    except Exception as e:
        warnings.warn(f"STOI failed: {e}")
        return None


def compute_all_metrics(noisy, denoised, clean_ref=None, sr=TARGET_SR):
    """Runs every mandatory metric and returns a flat dict. clean_ref is
    optional -- PESQ/STOI and ground-truth SNR improvement only run if it's
    given; otherwise SNR improvement falls back to the blind estimate."""
    n = min(len(noisy), len(denoised))
    noisy, denoised = noisy[:n], denoised[:n]
    if clean_ref is not None:
        n2 = min(n, len(clean_ref))
        noisy, denoised, clean_ref = noisy[:n2], denoised[:n2], clean_ref[:n2]

    level_change_db = overall_level_change_db(noisy, denoised)
    snr_improve, snr_in, snr_out, snr_mode = estimate_snr_improvement_db(noisy, denoised, clean_ref)
    flat_noisy = spectral_flatness(noisy)
    flat_denoised = spectral_flatness(denoised)

    pesq_score, stoi_score = None, None
    if clean_ref is not None:
        pesq_score = safe_pesq(clean_ref, denoised, sr)
        stoi_score = safe_stoi(clean_ref, denoised, sr)

    return {
        "overall_level_change_db": level_change_db,
        "estimated_snr_improvement_db": snr_improve,
        "snr_improvement_mode": snr_mode,
        "spectral_flatness_noisy": flat_noisy,
        "spectral_flatness_denoised": flat_denoised,
        "pesq": pesq_score,
        "stoi": stoi_score,
        "has_clean_ref": clean_ref is not None,
    }
