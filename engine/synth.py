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
    if n <= 0:
        return env
    # 짧은 노트에서도 ai+di+ri <= n 보장(독립 클램프 시 합이 n 초과 → 브로드캐스트 오류)
    ai = min(int(a * sr), n)
    ri = min(int(r * sr), n - ai)
    di = min(int(d * sr), n - ai - ri)
    if ai:
        env[:ai] = np.linspace(0, 1, ai)
    if di:
        env[ai:ai + di] = np.linspace(1, s, di)
    sus_end = n - ri
    if sus_end > ai + di:
        env[ai + di:sus_end] = s
    if ri:
        start_val = env[sus_end - 1] if sus_end > 0 else s
        env[sus_end:] = np.linspace(start_val, 0, ri)
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
        if start < 0:
            start = 0
        if start >= len(buf):   # 버퍼를 벗어나는 노트는 건너뜀(브로드캐스트 에러 방지)
            continue
        n = max(1, int((nt["end"] - nt["start"]) * sr))
        amp = nt["vel"] / 127.0
        t = np.arange(n) / sr
        if kind == "bass":
            # 기음 + saw 질감 + 한 옥타브 아래 서브로 무게감
            sub = 0.25 * np.sin(2 * np.pi * (f * 0.5) * t)
            wave = 0.62 * np.sin(2 * np.pi * f * t) + 0.28 * _saw(f, n, sr) + sub
            cut = min(f * (3.0 + 5.0 * amp), sr / 2 - 100)
            sos = butter(2, max(cut, f * 2), btype="low", fs=sr, output="sos")
            wave = sosfilt(sos, wave)
            env = _adsr(n, sr, a=0.004, d=0.06, s=0.75, r=0.06)
        elif kind == "guitar":
            # 플럭(뜯는) 음색: 밝은 saw+square, 빠른 감쇠 + 약한 몸통
            wave = 0.55 * _saw(f, n, sr, detune=0.004) + 0.25 * np.sign(np.sin(2 * np.pi * f * t))
            cut = min(f * (5.0 + 7.0 * amp), sr / 2 - 100)
            sos = butter(2, max(cut, f * 2), btype="low", fs=sr, output="sos")
            wave = sosfilt(sos, wave)
            env = np.exp(-t * (4.5 + 3.0 * (1 - amp)))          # 플럭 감쇠
            env *= _adsr(n, sr, a=0.002, d=0.02, s=1.0, r=0.05)  # 어택/릴리즈 다듬기
        elif kind == "piano":
            # 타현(두드리는) 음색: 1·2·3 배음 합 + 지수 감쇠(배음 먼저 꺼짐)
            h1 = np.sin(2 * np.pi * f * t)
            h2 = 0.5 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 6)
            h3 = 0.25 * np.sin(2 * np.pi * 3 * f * t) * np.exp(-t * 10)
            wave = h1 + h2 + h3
            env = np.exp(-t * (2.2 + 1.5 * (1 - amp)))          # 피아노식 긴 감쇠
            env *= _adsr(n, sr, a=0.002, d=0.02, s=1.0, r=0.08)
        elif kind == "lead" or kind == "synthlead":
            # 리드: 밝은 square+saw + 비브라토
            vib = 1.0 + 0.004 * np.sin(2 * np.pi * 5.5 * t)
            ph = 2 * np.pi * f * np.cumsum(vib) / sr
            wave = 0.5 * np.sign(np.sin(ph)) + 0.5 * (2 * ((f * np.cumsum(vib) / sr) % 1.0) - 1)
            cut = min(f * (6.0 + 6.0 * amp), sr / 2 - 100)
            sos = butter(2, max(cut, f * 2), btype="low", fs=sr, output="sos")
            wave = sosfilt(sos, wave)
            env = _adsr(n, sr, a=0.02, d=0.08, s=0.8, r=0.12)
        elif kind == "pad":
            # 패드: 여러 겹 디튠 saw + 느린 어택/릴리즈(풍성·부드러움)
            wave = (_saw(f, n, sr, detune=0.012) + _saw(f, n, sr, detune=0.02)
                    + 0.6 * np.sin(2 * np.pi * f * t)) / 2.6
            cut = min(f * 4.0, sr / 2 - 100)
            sos = butter(2, max(cut, f * 2), btype="low", fs=sr, output="sos")
            wave = sosfilt(sos, wave)
            env = _adsr(n, sr, a=0.08, d=0.2, s=0.75, r=0.3)
        elif kind == "organ":
            # 오르간: 1·2·4·8배음 합(드로바 느낌), 거의 플랫 서스테인
            wave = (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 2 * f * t)
                    + 0.35 * np.sin(2 * np.pi * 4 * f * t) + 0.2 * np.sin(2 * np.pi * 8 * f * t)) / 2.05
            env = _adsr(n, sr, a=0.006, d=0.02, s=0.95, r=0.06)
        elif kind == "fmbell":
            # FM 벨/전자음: 비배음 금속성 음색 + 감쇠
            idx = 3.0 + 4.0 * amp
            wave = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * 2.01 * t))
            env = np.exp(-t * 3.0) * _adsr(n, sr, a=0.002, d=0.02, s=1.0, r=0.1)
        else:  # keys / other: 2겹 디튠 saw 패드
            wave = 0.6 * _saw(f, n, sr, detune=0.008) + 0.4 * _saw(f * 2, n, sr, detune=0.006)
            cut = min(f * (4.0 + 6.0 * amp), sr / 2 - 100)
            sos = butter(2, max(cut, f * 2), btype="low", fs=sr, output="sos")
            wave = sosfilt(sos, wave)
            env = _adsr(n, sr, a=0.01, d=0.1, s=0.6, r=0.14)
        seg = (wave * env * amp).astype(np.float32)
        end = min(start + n, len(buf))
        if end <= start:
            continue
        buf[start:end] += seg[: end - start]
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
