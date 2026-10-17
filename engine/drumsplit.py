# -*- coding: utf-8 -*-
"""
드럼 스템을 타격 종류별(킥·스네어·하이햇·탐·심벌) 개별 트랙으로 분해.

demucs는 드럼을 한 덩어리로만 분리하므로, 그 드럼 스템 위에서
온셋 분류(transcribe.transcribe_drums)로 '언제 어떤 조각이 쳤는지'를 알아낸 뒤,
조각별 특성 대역을 band-pass 하고 해당 타격 구간만 게이트로 떼어낸다.

완벽한 분리는 아니지만(겹친 타격·브로드밴드 성분은 번짐), 킥/스네어/하이햇을
따로 뽑아 리믹스·샘플링에 쓰기엔 충분한 근사 분해를 제공한다.
"""
import numpy as np
from scipy.signal import butter, sosfilt


# 조각 → (필터 스펙, 게이트 길이초). 분류 라벨을 5개 조각으로 묶는다.
_LABEL_TO_PIECE = {
    "kick": "kick", "snare": "snare",
    "hat": "hat", "open_hat": "hat",
    "tom": "tom", "crash": "cymbal", "ride": "cymbal",
}
_PIECE_ORDER = ["kick", "snare", "hat", "tom", "cymbal"]
_PIECE_BAND = {
    # (type, cutoff(s))  type: 'low'|'high'|'band'
    "kick":   ("low", 130.0, 0.30),
    "snare":  ("band", (160.0, 3500.0), 0.25),
    "hat":    ("high", 6000.0, 0.12),
    "tom":    ("band", (80.0, 320.0), 0.30),
    "cymbal": ("high", 4000.0, 0.70),
}


def _bandfilter(y, sr, spec):
    kind = spec[0]
    if kind == "low":
        sos = butter(4, spec[1], btype="low", fs=sr, output="sos")
    elif kind == "high":
        sos = butter(4, spec[1], btype="high", fs=sr, output="sos")
    else:
        lo, hi = spec[1]
        hi = min(hi, sr / 2 - 100)
        sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfilt(sos, y)


def _gate_envelope(n, sr, hits_t, hold):
    """해당 조각 타격 시각마다 raised-cosine 어택 + 지수 감쇠 게이트를 OR 합성."""
    env = np.zeros(n, dtype=np.float32)
    atk = max(1, int(0.004 * sr))
    hold_n = max(atk + 1, int(hold * sr))
    # 하나의 타격 게이트 템플릿(어택 상승 → 지수 하강)
    t = np.arange(hold_n) / sr
    tmpl = np.exp(-t * (3.0 / max(hold, 0.05))).astype(np.float32)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, atk))).astype(np.float32)
    tmpl[:atk] *= ramp
    for ts in hits_t:
        s = int(ts * sr)
        if s < 0:
            s = 0
        if s >= n:
            continue
        e = min(s + hold_n, n)
        np.maximum(env[s:e], tmpl[: e - s], out=env[s:e])
    return env


def split_drum_stem(y, sr, hits):
    """드럼 스템 y(모노)와 분류된 타격 hits([{time,drum,vel}]) → {piece: np.ndarray}.

    hits가 비었거나 특정 조각이 없으면 그 조각은 생략(무음 트랙 안 만듦).
    """
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    # 조각별 타격 시각 모으기
    times = {p: [] for p in _PIECE_ORDER}
    for h in hits or []:
        p = _LABEL_TO_PIECE.get(h.get("drum"))
        if p:
            times[p].append(float(h.get("time", 0.0)))
    out = {}
    for p in _PIECE_ORDER:
        ts = times[p]
        if not ts:
            continue
        spec = _PIECE_BAND[p]
        band = _bandfilter(y, sr, (spec[0], spec[1])).astype(np.float32)
        env = _gate_envelope(n, sr, ts, spec[2])
        piece = band * env
        peak = float(np.max(np.abs(piece))) if piece.size else 0.0
        if peak < 1e-5:
            continue
        out[p] = piece
    return out
