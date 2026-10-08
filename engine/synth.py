# -*- coding: utf-8 -*-
"""
재연주용 신디사이저. numpy만으로 깨끗한 악기 소리를 합성한다.
전사된 노트/드럼 이벤트를 받아 오디오 버퍼로 렌더.

제품 빌드에서는 이 자리를 SoundFont(fluidsynth) 또는 VST(pedalboard 호스팅)로 교체해
"가능한 비슷한 소리"의 실제 악기 음색을 낸다.
"""
import numpy as np
from scipy.signal import butter, sosfilt


def _midi_to_hz(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def _adsr(n, sr, a=0.005, d=0.05, s=0.7, r=0.08):
    env = np.ones(n)
    ai, di, ri = int(a * sr), int(d * sr), int(r * sr)
    ai, di, ri = min(ai, n), min(di, n), min(ri, n)
    if ai:
        env[:ai] = np.linspace(0, 1, ai)
    if di:
        env[ai:ai + di] = np.linspace(1, s, di)
    env[ai + di:n - ri] = s
    if ri:
        env[n - ri:] = np.linspace(env[max(n - ri - 1, 0)], 0, ri)
    return env


def _saw(freq, n, sr, detune=0.0):
    t = np.arange(n) / sr
    o = 2 * (t * freq % 1.0) - 1
    if detune:
        o += 2 * (t * freq * (1 + detune) % 1.0) - 1
        o += 2 * (t * freq * (1 - detune) % 1.0) - 1
        o /= 3
    return o


def render_notes(notes, sr, duration, kind="bass"):
    """단선율 노트열을 악기 음색으로 렌더."""
    buf = np.zeros(int(duration * sr) + sr)
    for nt in notes:
        f = _midi_to_hz(nt["midi"])
        start = int(nt["start"] * sr)
        n = max(1, int((nt["end"] - nt["start"]) * sr))
        amp = nt["vel"] / 127.0
        t = np.arange(n) / sr
        if kind == "bass":
            wave = 0.7 * np.sin(2 * np.pi * f * t) + 0.3 * _saw(f, n, sr)
            env = _adsr(n, sr, a=0.004, d=0.06, s=0.75, r=0.06)
        else:  # keys / other: 디튠 saw + 저역 필터
            wave = _saw(f, n, sr, detune=0.008)
            sos = butter(2, min(f * 6, sr / 2 - 100), btype="low", fs=sr, output="sos")
            wave = sosfilt(sos, wave)
            env = _adsr(n, sr, a=0.01, d=0.1, s=0.6, r=0.12)
        seg = (wave * env * amp).astype(np.float32)
        end = min(start + n, len(buf))
        buf[start:end] += seg[:end - start]
    return buf[: int(duration * sr)]


def _kick(sr):
    n = int(0.25 * sr); t = np.arange(n) / sr
    f = np.linspace(120, 45, n)
    w = np.sin(2 * np.pi * np.cumsum(f) / sr)
    w *= np.exp(-t * 18)
    return (w * 0.9).astype(np.float32)


def _snare(sr):
    n = int(0.2 * sr); t = np.arange(n) / sr
    noise = np.random.randn(n) * np.exp(-t * 22)
    tone = np.sin(2 * np.pi * 180 * t) * np.exp(-t * 25)
    return ((noise * 0.7 + tone * 0.4) * 0.7).astype(np.float32)


def _hat(sr):
    n = int(0.06 * sr); t = np.arange(n) / sr
    noise = np.random.randn(n) * np.exp(-t * 90)
    sos = butter(4, 6000, btype="high", fs=sr, output="sos")
    return (sosfilt(sos, noise) * 0.5).astype(np.float32)


def _tom(sr):
    n = int(0.3 * sr); t = np.arange(n) / sr
    f = np.linspace(180, 110, n)
    w = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 9)
    return (w * 0.7).astype(np.float32)


def _crash(sr):
    n = int(0.8 * sr); t = np.arange(n) / sr
    noise = np.random.randn(n) * np.exp(-t * 3.5)
    sos = butter(4, 4000, btype="high", fs=sr, output="sos")
    return (sosfilt(sos, noise) * 0.45).astype(np.float32)


def render_drums(hits, sr, duration):
    buf = np.zeros(int(duration * sr) + sr)
    samples = {"kick": _kick(sr), "snare": _snare(sr), "hat": _hat(sr),
               "tom": _tom(sr), "crash": _crash(sr)}
    for h in hits:
        s = samples[h["drum"]] * (h["vel"] / 127.0)
        start = int(h["time"] * sr)
        end = min(start + len(s), len(buf))
        buf[start:end] += s[: end - start]
    return buf[: int(duration * sr)]
