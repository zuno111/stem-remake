# -*- coding: utf-8 -*-
"""
BS.1770 기반 간이 라우드니스(LUFS) 측정 + 정규화.

K-weighting(프리-필터 고역 쉘프 + RLB 하이패스) 적용 후 400ms 블록으로
게이팅된 적분 라우드니스를 구해 목표 LUFS로 게인 보정한다.
스트리밍 표준(-14 LUFS)에 맞추면 트랙 간 체감 음량이 일정해진다.

모노 입력 기준(이 파이프라인은 모노 마스터). scipy만 사용.
"""
import numpy as np
from scipy.signal import lfilter


def _k_weighting_coeffs(sr):
    """BS.1770-4 K-weighting 2단 필터 계수(48k 기준식을 sr로 치환)."""
    # 1단: 고역 쉘프(머리 그림자 보정)
    f0 = 1681.974450955533
    G = 3.999843853973347
    Q = 0.7071752369554196
    K = np.tan(np.pi * f0 / sr)
    Vh = 10 ** (G / 20.0)
    Vb = Vh ** 0.4996667741545416
    a0 = 1.0 + K / Q + K * K
    b0 = (Vh + Vb * K / Q + K * K) / a0
    b1 = 2.0 * (K * K - Vh) / a0
    b2 = (Vh - Vb * K / Q + K * K) / a0
    a1 = 2.0 * (K * K - 1.0) / a0
    a2 = (1.0 - K / Q + K * K) / a0
    stage1_b = [b0, b1, b2]
    stage1_a = [1.0, a1, a2]

    # 2단: RLB 하이패스
    f0 = 38.13547087602444
    Q = 0.5003270373238773
    K = np.tan(np.pi * f0 / sr)
    a0 = 1.0 + K / Q + K * K
    b0 = 1.0
    b1 = -2.0
    b2 = 1.0
    a1 = 2.0 * (K * K - 1.0) / a0
    a2 = (1.0 - K / Q + K * K) / a0
    stage2_b = [b0, b1, b2]
    stage2_a = [1.0, a1, a2]
    return (stage1_b, stage1_a), (stage2_b, stage2_a)


def measure_lufs(y, sr):
    """게이팅된 적분 라우드니스(LUFS) 추정. 무음/초단편은 -inf 방지용 하한 반환."""
    y = np.asarray(y, dtype=np.float64)
    if y.size < int(0.4 * sr):
        return -70.0
    (b1, a1), (b2, a2) = _k_weighting_coeffs(sr)
    yk = lfilter(b1, a1, y)
    yk = lfilter(b2, a2, yk)

    block = int(0.4 * sr)            # 400ms
    step = int(0.1 * sr)             # 75% 겹침(100ms hop)
    if step < 1:
        return -70.0
    powers = []
    for s in range(0, len(yk) - block + 1, step):
        seg = yk[s:s + block]
        powers.append(np.mean(seg * seg) + 1e-12)
    if not powers:
        powers = [np.mean(yk * yk) + 1e-12]
    powers = np.array(powers)
    loud = -0.691 + 10.0 * np.log10(powers)   # 블록별 라우드니스

    # 절대 게이트(-70 LUFS)
    keep = loud > -70.0
    if not keep.any():
        return -70.0
    mean_pow = np.mean(powers[keep])
    abs_gated = -0.691 + 10.0 * np.log10(mean_pow)
    # 상대 게이트(적분값 -10 LU)
    rel = abs_gated - 10.0
    keep2 = loud > rel
    if keep2.any():
        mean_pow = np.mean(powers[keep2])
    return float(-0.691 + 10.0 * np.log10(mean_pow))


def normalize_lufs(y, sr, target_lufs=-14.0, max_gain_db=18.0):
    """측정 LUFS를 목표로 게인 보정. 과도한 증폭은 max_gain_db로 제한.
    반환: (정규화 오디오, 측정 LUFS, 적용 게인 dB)."""
    cur = measure_lufs(y, sr)
    if cur <= -70.0:
        return np.asarray(y, dtype=np.float32), cur, 0.0
    gain_db = float(np.clip(target_lufs - cur, -max_gain_db, max_gain_db))
    out = (np.asarray(y, dtype=np.float64) * (10 ** (gain_db / 20.0))).astype(np.float32)
    return out, cur, gain_db
