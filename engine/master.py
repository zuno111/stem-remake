# -*- coding: utf-8 -*-
"""
보컬 정제 + 재믹스 + 마스터링. pedalboard 기반(실제 오디오 처리).
제품 빌드에서는 마스터링을 Matchering(레퍼런스 기반)으로 교체 가능.
"""
import numpy as np
from pedalboard import (
    Pedalboard, Compressor, Gain, HighpassFilter, LowShelfFilter,
    HighShelfFilter, PeakFilter, Limiter, NoiseGate, Reverb,
)


def _true_peak(y, oversample=4):
    """인터샘플(트루) 피크 추정. 4배 오버샘플 후 최대 절댓값."""
    try:
        from scipy.signal import resample_poly
        up = resample_poly(y.astype(np.float64), oversample, 1)
        return float(np.max(np.abs(up)))
    except Exception:
        return float(np.max(np.abs(y)))


def _true_peak_limit(y, sr, ceil_db=-1.0, oversample=4):
    """오버샘플 도메인에서 리미팅해 인터샘플 피크를 천장 아래로 제어.
    전체 스케일 다운(라우드니스 손실) 대신 피크만 눌러 체감 음량을 보존한다."""
    ceil = 10 ** (ceil_db / 20.0)
    try:
        from scipy.signal import resample_poly
        up = resample_poly(y.astype(np.float64), oversample, 1)
        # 다운샘플 리플 여유를 위해 내부 리미터는 약간 낮게(-0.6dB) 건다
        lim = Pedalboard([Limiter(threshold_db=ceil_db - 0.6, release_ms=80)])
        up = np.asarray(lim(up.astype(np.float32), sr * oversample), dtype=np.float64)
        down = resample_poly(up, 1, oversample)
        down = down[: len(y)] if len(down) >= len(y) else np.pad(down, (0, len(y) - len(down)))
        # 최종 보증은 '트루피크' 기준(샘플피크 아님) — 인터샘플 초과분까지 제거
        tp = _true_peak(down) or 1.0
        if tp > ceil:
            down = down * (ceil / tp)
        return down.astype(np.float32)
    except Exception:
        tp = _true_peak(y) or 1.0
        if tp > ceil:
            return (y * (ceil / tp)).astype(np.float32)
        return np.asarray(y, dtype=np.float32)


def clean_vocal(y, sr):
    """보컬 스템 정제: 저역 잡음 제거 + 디에서성 + 살짝의 공간감."""
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=90),
        NoiseGate(threshold_db=-45, ratio=2.0, attack_ms=2, release_ms=120),
        Compressor(threshold_db=-20, ratio=2.5, attack_ms=5, release_ms=120),
        PeakFilter(cutoff_frequency_hz=6000, gain_db=-3, q=2.0),   # 치찰음 완화
        HighShelfFilter(cutoff_frequency_hz=9000, gain_db=2),       # 공기감
        Reverb(room_size=0.12, wet_level=0.08, dry_level=0.92),
    ])
    return board(y, sr)


def clean_instrument(y, sr, bright=True):
    """악기 스템 정제(폴리포닉 파트 유지용): 저역 정리 + 가벼운 EQ. 리버브 없음."""
    chain = [
        HighpassFilter(cutoff_frequency_hz=35),
        Compressor(threshold_db=-22, ratio=2.0, attack_ms=8, release_ms=140),
    ]
    if bright:
        chain.append(HighShelfFilter(cutoff_frequency_hz=8000, gain_db=1.5))
    return Pedalboard(chain)(y, sr)


def _denoise_mono(y, sr, prop_decrease=0.6):
    """분리 스템 잔여 잡음/히스 제거(정적 스펙트럴 게이팅). 실패 시 원본 반환."""
    try:
        import noisereduce as nr
        out = nr.reduce_noise(y=y.astype(np.float32), sr=sr,
                              stationary=True, prop_decrease=float(prop_decrease))
        return np.asarray(out, dtype=np.float32)
    except Exception:
        return y


def mix_and_master(stems, sr, gains=None, target_lufs=-14.0, denoise=0.0, gentle=False):
    """
    재연주 악기 + 정제 보컬을 믹스 후 마스터링.
    stems: {name: np.ndarray}. gains: {name: dB} 선택.
    target_lufs: 적분 라우드니스 목표(스트리밍 -14 LUFS). None이면 라우드니스 정규화 생략.
    denoise: >0 이면 믹스에 디노이즈 적용(음색 유지본처럼 분리 잡음이 섞인 경우). 0~1.
    gentle: True면 고역 부스트/피크 부스트를 약하게(분리 스템 잡음 증폭 방지).
    """
    gains = gains or {"vocals": 0.0, "drums": -1.0, "bass": -2.0, "other": -3.0}
    length = max(len(a) for a in stems.values())
    mix = np.zeros(length, dtype=np.float32)
    for name, audio in stems.items():
        g = 10 ** (gains.get(name, 0.0) / 20.0)
        a = np.zeros(length, dtype=np.float32)
        a[: len(audio)] = audio.astype(np.float32)
        mix += a * g

    # 분리 잔여 잡음 제거(마스터링으로 증폭되기 전에)
    if denoise and denoise > 0:
        mix = _denoise_mono(mix, sr, prop_decrease=float(denoise))

    peak = np.max(np.abs(mix)) or 1.0
    mix = mix / peak * 0.9

    # gentle: 분리 스템은 고역/피크 부스트가 히스·아티팩트를 키우므로 약하게
    air_db = 0.8 if gentle else 2.0
    peak_db = 0.6 if gentle else 1.5
    master = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30),
        LowShelfFilter(cutoff_frequency_hz=100, gain_db=1.5),
        PeakFilter(cutoff_frequency_hz=3000, gain_db=peak_db, q=1.0),
        HighShelfFilter(cutoff_frequency_hz=10000, gain_db=air_db),
        Compressor(threshold_db=-14, ratio=2.0, attack_ms=10, release_ms=150),
        Limiter(threshold_db=-1.0, release_ms=100),
        Gain(gain_db=0.0),
    ])
    out = np.asarray(master(mix, sr), dtype=np.float32)

    # 라우드니스 정규화: 트랙 간 체감 음량 일정화(-14 LUFS 기본)
    if target_lufs is not None:
        try:
            from . import loudness as _ld
            out, _cur, _g = _ld.normalize_lufs(out, sr, target_lufs=target_lufs)
        except Exception:
            pass

    # 최종 안전 천장: 트루피크 -1dBTP 리미팅(라우드니스 보존, 인터샘플 클리핑 방지)
    out = _true_peak_limit(out, sr, ceil_db=-1.0)
    return out
