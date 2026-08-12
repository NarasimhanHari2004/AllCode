import math
import torch
import torch.nn as nn
import numpy as np


class WAVSTFT(nn.Module):
    def __init__(self, win_size=320):
        super().__init__()
        window = torch.from_numpy(np.hanning(win_size).astype(np.float32))
        self.window_size = window.shape[-1]
        self.hop_length = self.window_size // 2
        window = window.unsqueeze(0).unsqueeze(-1)

        divisor = torch.ones(1, 1, 1, self.window_size * 4)
        divisor = nn.functional.unfold(divisor, (1, self.window_size), stride=self.hop_length)
        divisor = divisor * window.pow(2.0)
        divisor = nn.functional.fold(
            divisor, (1, self.window_size * 4), (1, self.window_size), stride=self.hop_length
        )[:, 0, 0, :]
        divisor = divisor[0, self.window_size:2 * self.window_size].unsqueeze(0).unsqueeze(-1)

        self.register_buffer("window", window)
        self.register_buffer("divisor", divisor)

    def add_window(self, x, divisor):
        return x * self.window / divisor

    def frame(self, x):
        assert x.dim() == 2, x.shape
        out = x.unsqueeze(1).unsqueeze(1)
        out = nn.functional.pad(out, (self.window_size, self.window_size), "constant", 0)
        out = nn.functional.unfold(out, (1, self.window_size), stride=self.hop_length)
        return out

    def overlap_and_add(self, x, length):
        assert x.dim() == 3, x.shape
        out = nn.functional.fold(
            x, (1, length + 2 * self.window_size), (1, self.window_size), stride=self.hop_length
        )[:, 0, 0, :]
        return out[:, self.window_size:-self.window_size]

    def rfft(self, x):
        assert x.dim() == 3, x.shape
        return torch.fft.rfft(x, dim=1)

    def irfft(self, x):
        assert x.dim() == 3, x.shape
        return torch.fft.irfft(x, dim=1)

    def STFT(self, x):
        assert x.dim() == 2, x.shape
        out = self.frame(x)
        out = self.add_window(out, 1)
        return self.rfft(out)

    def iSTFT(self, x, length):
        assert x.dim() == 3, x.shape
        out = self.irfft(x)
        out = self.add_window(out, self.divisor)
        return self.overlap_and_add(out, length=length)


class WienerPrior(nn.Module):
    def __init__(self, alpha_oversubtract=1.3, gain_floor=0.05,
                 noise_smooth=0.95, min_smooth=0.98, eps=1e-8):
        super().__init__()
        self.alpha = alpha_oversubtract
        self.floor = gain_floor
        self.noise_smooth = noise_smooth
        self.min_smooth = min_smooth
        self.eps = eps

    @torch.no_grad()
    def forward(self, power_spec):
        B, T, F = power_spec.shape
        device = power_spec.device
        noise_est = power_spec[:, 0:1, :].clone()
        running_min = power_spec[:, 0:1, :].clone()
        gains = torch.empty_like(power_spec)

        for t in range(T):
            py_t = power_spec[:, t:t + 1, :]
            running_min = torch.where(
                py_t < running_min,
                py_t,
                self.min_smooth * running_min + (1 - self.min_smooth) * py_t,
            )
            is_noise_like = (py_t <= running_min * 2.0).float()
            noise_est = (
                is_noise_like * (self.noise_smooth * noise_est + (1 - self.noise_smooth) * py_t)
                + (1 - is_noise_like) * noise_est
            )
            g = (py_t - self.alpha * noise_est) / (py_t + self.eps)
            g = torch.clamp(g, min=self.floor, max=1.0)
            gains[:, t, :] = g[:, 0, :]

        return gains, noise_est.expand(-1, T, -1) if noise_est.shape[1] == 1 else noise_est


class TimeCompression(nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, steps):
        super().__init__()
        self.dim1, self.dim2, self.dim3, self.dim4, self.steps = dim1, dim2, dim3, dim4, steps
        self.trans1 = nn.Conv2d(dim1 * steps, dim2, 1, bias=False)
        self.trans2 = nn.Conv2d(dim3, dim4, 1, bias=False)
        self.pad = 0

    def forward(self, x, inverse):
        if inverse:
            B, C, T, F = x.shape
            x = self.trans2(x).reshape(B, -1, 1, T, F).permute(0, 1, 3, 2, 4).contiguous()
            x = x.repeat(1, 1, 1, self.steps, 1).reshape(B, -1, T * self.steps, F)
            x = nn.functional.pad(x, (0, 0, self.steps - 1, 0), "constant", 0)
            x = x[:, :, :-self.steps + 1, :]
            if self.pad > 0:
                x = x[:, :, self.pad:, :]
            return x
        else:
            B, C, T, F = x.shape
            if x.shape[-2] % self.steps == 0:
                self.pad = 0
            else:
                self.pad = self.steps - x.shape[-2] % self.steps
                x = nn.functional.pad(x, (0, 0, self.pad, 0), "constant", 0)
            x = x.reshape(B, C, -1, self.steps, F).permute(0, 1, 3, 2, 4).contiguous()
            x = x.reshape(B, C * self.steps, -1, F)
            return self.trans1(x)


class FreqCompression(nn.Module):
    def __init__(self, nfreq, nfilters, in_dim, hidden_dim, out_dim, sample_rate=16000):
        super().__init__()
        self.nfreq, self.nfilters, self.sample_rate = nfreq, nfilters, sample_rate
        self.in_dim, self.hidden_dim, self.out_dim = in_dim, hidden_dim, out_dim

        all_freqs = torch.linspace(0, sample_rate // 2, nfreq)
        m_min = self._hz_to_mel(0)
        m_max = self._hz_to_mel(sample_rate / 2.0)
        m_pts = torch.linspace(m_min, m_max, nfilters + 2)
        f_pts = self._mel_to_hz(m_pts)
        self.bounds = [0]
        for i in range(1, len(f_pts) - 1):
            self.bounds.append((all_freqs > f_pts[i]).float().argmax().item())
        self.bounds.append(nfreq)

        self.trans1 = nn.ModuleList()
        self.trans2 = nn.ModuleList()
        for i in range(nfilters):
            width = self.bounds[i + 2] - self.bounds[i]
            self.trans1.append(nn.Linear(width * in_dim, hidden_dim, bias=False))
            self.trans2.append(nn.Conv1d(hidden_dim, width * out_dim, 1))

    @staticmethod
    def _hz_to_mel(freq):
        return 2595.0 * math.log10(1.0 + (freq / 700.0))

    @staticmethod
    def _mel_to_hz(mels: torch.Tensor):
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    def forward(self, x, inverse):
        if inverse:
            out = torch.zeros(
                [x.shape[0], self.out_dim, self.nfreq, x.shape[2]],
                dtype=x.dtype, layout=x.layout, device=x.device,
            )
            for i in range(self.nfilters):
                lo, hi = self.bounds[i], self.bounds[i + 2]
                out[:, :, lo:hi, :] = out[:, :, lo:hi, :] + self.trans2[i](x[:, :, :, i]).reshape(
                    x.shape[0], self.out_dim, -1, x.shape[-2]
                )
            out[:, :, self.bounds[1]:self.bounds[-2], :] /= 2.0
            return out.permute(0, 1, 3, 2).contiguous().tanh()
        else:
            x = x.reshape(x.shape[0], self.in_dim, *x.shape[-2:]).permute(0, 2, 1, 3).contiguous()
            x = torch.stack(
                [self.trans1[i](x[:, :, :, self.bounds[i]:self.bounds[i + 2]].flatten(start_dim=2))
                 for i in range(self.nfilters)], -1
            )
            return x.permute(0, 2, 1, 3).contiguous()


class UltraDualPath(nn.Module):
    def __init__(self, nfreq, in_dim, hidden_dim, out_dim, freq_cprs_ratio, time_cprs_ratio):
        super().__init__()
        self.compress_modules = nn.ModuleList([
            TimeCompression(in_dim, in_dim * 2, out_dim, out_dim, time_cprs_ratio),
            FreqCompression(nfreq, nfreq // freq_cprs_ratio, in_dim * 2, hidden_dim, out_dim),
        ])

    def forward(self, x, inverse):
        out = x
        modules = reversed(self.compress_modules) if inverse else self.compress_modules
        for m in modules:
            out = m(out, inverse)
        return out


class DynamicLayerNorm(nn.Module):
    def forward(self, x):
        return nn.functional.layer_norm(x, [x.shape[-1]])


class GRUBlock(nn.Module):
    def __init__(self, hidden_size, causal):
        super().__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, 1,
                           bidirectional=not causal, batch_first=True)
        self.linear = nn.Linear(hidden_size if causal else hidden_size * 2, hidden_size)
        self.norm = DynamicLayerNorm()
        self.activation = nn.ReLU()

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.linear(self.activation(out))
        out = x + out
        return self.norm(out)


class FreqAxisAttention(nn.Module):
    def __init__(self, hidden_size, n_heads=4, ff_mult=2, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ff_mult),
            nn.GELU(),
            nn.Linear(hidden_size * ff_mult, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class TimeAxisCausalAttention(nn.Module):
    def __init__(self, hidden_size, n_heads=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        T = x.shape[1]
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1
        )
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        return self.norm(x + attn_out)


class AttentionDualPathBackbone(nn.Module):
    def __init__(self, hidden_size, num_layers, n_heads=4, causal=True, use_attention=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_attention = use_attention

        self.row_gru = nn.ModuleList([GRUBlock(hidden_size, causal) for _ in range(num_layers)])
        self.col_gru = nn.ModuleList([GRUBlock(hidden_size, False) for _ in range(num_layers)])
        if use_attention:
            self.row_attn = nn.ModuleList(
                [TimeAxisCausalAttention(hidden_size, n_heads) for _ in range(num_layers)]
            )
            self.col_attn = nn.ModuleList(
                [FreqAxisAttention(hidden_size, n_heads) for _ in range(num_layers)]
            )

    def forward(self, x):
        b, _, T, F = x.shape
        out = x
        for i in range(len(self.row_gru)):
            row_in = out.permute(0, 3, 2, 1).contiguous().view(b * F, T, -1)
            row_out = self.row_gru[i](row_in)
            if self.use_attention:
                row_out = self.row_attn[i](row_out)
            out = row_out.view(b, F, T, -1).permute(0, 3, 2, 1).contiguous()

            col_in = out.permute(0, 2, 3, 1).contiguous().view(b * T, F, -1)
            col_out = self.col_gru[i](col_in)
            if self.use_attention:
                col_out = self.col_attn[i](col_out)
            out = col_out.view(b, T, F, -1).permute(0, 3, 1, 2).contiguous()

        return out


class PhysicsAttentionDenoiser(nn.Module):
    def __init__(self, win_size=320, hidden_size=48, freq_cprs_ratio=4, time_cprs_ratio=4,
                 num_dual_path_blocks=3, n_heads=4, use_attention=True, use_physics_prior=True):
        super().__init__()
        self.win_size = win_size
        self.use_physics_prior = use_physics_prior
        nfreq = win_size // 2 + 1

        self.stft = WAVSTFT(win_size)
        self.physics_prior = WienerPrior() if use_physics_prior else None

        in_ch = 2 + (1 if use_physics_prior else 0)

        self.ultra_compress = UltraDualPath(nfreq, in_ch, hidden_size, hidden_size,
                                             freq_cprs_ratio, time_cprs_ratio)
        self.backbone = AttentionDualPathBackbone(
            hidden_size, num_dual_path_blocks, n_heads=n_heads,
            causal=True, use_attention=use_attention,
        )
        self.mask_head = nn.Conv2d(hidden_size, 2, 1)

    def forward(self, wav):
        B, N = wav.shape
        Y = self.stft.STFT(wav)
        Y = Y.permute(0, 2, 1)
        real, imag = Y.real, Y.imag
        power = real.pow(2) + imag.pow(2)

        feats = [real.unsqueeze(1), imag.unsqueeze(1)]
        if self.use_physics_prior:
            with torch.no_grad():
                gain, _ = self.physics_prior(power)
            feats.append(gain.unsqueeze(1))

        x = torch.cat(feats, dim=1)

        latent = self.ultra_compress(x, inverse=False)
        latent = self.backbone(latent)
        latent = self.ultra_compress(latent, inverse=True)
        mask = self.mask_head(latent).tanh()

        m_r, m_i = mask[:, 0], mask[:, 1]
        est_r = real * m_r - imag * m_i
        est_i = real * m_i + imag * m_r

        est_spec = torch.complex(est_r, est_i).permute(0, 2, 1)
        out_wav = self.stft.iSTFT(est_spec, N)
        return out_wav
