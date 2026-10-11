# -*- coding: utf-8 -*-
"""
원본 오디오 직접 보정(enhance) — 분리·재연주 없이 원곡을 그대로 손봐서
노이즈/하스함을 줄이고 명료도·음량을 올린다. 스테레오·원본 샘플레이트 유지.

AI 생성 음원(Suno 등)의 전형적 문제: 넓게 깔린 생성 노이즈/헤이즈, 상단 중역 하스함,
저중역 머드, 공기감 부족. 분리 기반 리메이크는 원본보다 나빠지므로, "더 나은 소리"는
이 enhance 경로가 맞다.
"""
import numpy as np
import soundfile as sf
from pedalboard import (
    Pedalboard, HighpassFilter, LowShelfFilter, HighShelfFilter,
    PeakFilter, Compressor, Limiter,
)


def _denoise_stereo(y, sr, prop_decrease=0.55):
    """채널별 스펙트럼 게이팅 디노이즈(비정상 노이즈). 과하면 워터리해지므로 보수적."""
    import noisereduce as nr
    out = np.empty_like(y)
    for ch in range(y.shape[0]):
        out[ch] = nr.reduce_noise(y=y[ch], sr=sr, stationary=False,
                                  prop_decrease=prop_decrease,
                                  n_fft=2048, freq_mask_smooth_hz=500)
    return out


def _lufs_mono(y, sr):
    from . import loudness
    mono = y.mean(axis=0) if y.ndim > 1 else y
    return loudness.measure_lufs(mono, sr)


def _true_peak_stereo(y, oversample=4):
    try:
        from scipy.signal import resample_poly
        up = resample_poly(y, oversample, 1, axis=-1)
        return float(np.max(np.abs(up)))
    except Exception:
        return float(np.max(np.abs(y)))


def enhance(in_path, out_path, target_lufs=-14.0, denoise=0.55,
            start=None, dur=None):
    """원본을 보정해 out_path(스테레오 wav)로 저장. start/dur(초) 주면 그 구간만."""
    y, sr = sf.read(in_path, always_2d=True)      # (N, ch)
    y = y.T.astype(np.float32)                     # (ch, N)
    if y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    if start is not None:
        a = int(start * sr); b = int((start + (dur or 30)) * sr)
        y = y[:, a:b]

    # 1) 디노이즈(생성 노이즈·헤이즈 저감)
    if denoise and denoise > 0:
        y = _denoise_stereo(y, sr, prop_decrease=denoise)

    # 2) 톤·다이내믹 보정(스테레오 일괄)
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=28),          # 서브소닉 제거
        LowShelfFilter(cutoff_frequency_hz=120, gain_db=1.0),   # 저역 무게 약간
        PeakFilter(cutoff_frequency_hz=350, gain_db=-1.5, q=1.0),  # 저중역 머드 정리
        PeakFilter(cutoff_frequency_hz=3200, gain_db=-1.5, q=1.4),  # 상단 중역 하스함 완화
        HighShelfFilter(cutoff_frequency_hz=11000, gain_db=2.0),  # 공기감
        Compressor(threshold_db=-18, ratio=1.8, attack_ms=15, release_ms=160),  # 글루
    ])
    y = np.asarray(board(y, sr), dtype=np.float32)

    # 3) 라우드니스 정규화(-14 LUFS) + 트루피크 천장
    cur = _lufs_mono(y, sr)
    if cur > -70:
        gain_db = float(np.clip(target_lufs - cur, -18, 18))
        y = (y * (10 ** (gain_db / 20.0))).astype(np.float32)
    ceil = 10 ** (-1.0 / 20.0)
    lim = Pedalboard([Limiter(threshold_db=-1.6, release_ms=90)])
    y = np.asarray(lim(y, sr), dtype=np.float32)
    tp = _true_peak_stereo(y) or 1.0
    if tp > ceil:
        y = (y * (ceil / tp)).astype(np.float32)

    sf.write(out_path, y.T, sr)                     # 스테레오 유지
    return {"sr": sr, "channels": y.shape[0], "lufs_before": round(cur, 1),
            "lufs_after": round(_lufs_mono(y, sr), 1), "true_peak": round(_true_peak_stereo(y), 3)}
