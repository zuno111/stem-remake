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


def mix_and_master(stems, sr, gains=None):
    """
    재연주 악기 + 정제 보컬을 믹스 후 마스터링.
    stems: {name: np.ndarray}. gains: {name: dB} 선택.
    """
    gains = gains or {"vocals": 0.0, "drums": -1.0, "bass": -2.0, "other": -3.0}
    length = max(len(a) for a in stems.values())
    mix = np.zeros(length, dtype=np.float32)
    for name, audio in stems.items():
        g = 10 ** (gains.get(name, 0.0) / 20.0)
        a = np.zeros(length, dtype=np.float32)
        a[: len(audio)] = audio.astype(np.float32)
        mix += a * g

    peak = np.max(np.abs(mix)) or 1.0
    mix = mix / peak * 0.9

    master = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30),
        LowShelfFilter(cutoff_frequency_hz=100, gain_db=1.5),
        PeakFilter(cutoff_frequency_hz=3000, gain_db=1.5, q=1.0),
        HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2.0),
        Compressor(threshold_db=-14, ratio=2.0, attack_ms=10, release_ms=150),
        Limiter(threshold_db=-1.0, release_ms=100),
        Gain(gain_db=0.0),
    ])
    out = np.asarray(master(mix, sr), dtype=np.float32)
    # 최종 안전 천장: -1dBFS 초과분 제거(진짜 클리핑 방지)
    peak = float(np.max(np.abs(out))) or 1.0
    ceil = 10 ** (-1.0 / 20.0)  # ≈0.891
    if peak > ceil:
        out = out * (ceil / peak)
    return out
