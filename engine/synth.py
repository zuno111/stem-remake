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
        elif kind == "sine":
            # 순음(플루트계): 기음 중심 + 약한 2배음, 부드러운 어택
            wave = np.sin(2 * np.pi * f * t) + 0.12 * np.sin(2 * np.pi * 2 * f * t)
            env = _adsr(n, sr, a=0.03, d=0.08, s=0.85, r=0.12)
        elif kind == "brass":
            # 브라스: 톱니 + 세기/시간에 따라 열리는 필터(스웰감) + 밝은 배음
            wave = _saw(f, n, sr, detune=0.004)
            cut0 = f * 2.5
            cut1 = min(f * (7.0 + 6.0 * amp), sr / 2 - 100)
            # 노트 길이에 걸쳐 컷오프 상승(스웰) — 근사: 2구간 필터 혼합
            w_lo = sosfilt(butter(2, max(cut0, f * 2), btype="low", fs=sr, output="sos"), wave)
            w_hi = sosfilt(butter(2, max(cut1, f * 2), btype="low", fs=sr, output="sos"), wave)
            ramp = np.clip(np.linspace(0, 1, n) * 1.5, 0, 1)
            wave = w_lo * (1 - ramp) + w_hi * ramp
            env = _adsr(n, sr, a=0.03, d=0.06, s=0.85, r=0.1)
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


def _open_hat(sr):
    # 오픈 하이햇: 클로즈보다 길게 울림(감쇠 완만)
    n = int(0.3 * sr); t = np.arange(n) / sr
    noise = np.random.randn(n) * np.exp(-t * 14)
    # 금속 쉬머: 비하모닉 부분음 몇 개를 얹어 '쇳소리'가 아닌 '심벌'처럼
    shimmer = sum(np.sin(2 * np.pi * fq * t) for fq in (7200, 9400, 11300)) * 0.04 * np.exp(-t * 10)
    sos = butter(4, 5000, btype="high", fs=sr, output="sos")
    return (sosfilt(sos, noise + shimmer) * 0.42).astype(np.float32)


def _ride(sr):
    # 라이드: 톤성 '핑' + 부드러운 쉬머(길게)
    n = int(0.6 * sr); t = np.arange(n) / sr
    ping = (np.sin(2 * np.pi * 520 * t) + 0.5 * np.sin(2 * np.pi * 1040 * t)) * np.exp(-t * 6)
    noise = np.random.randn(n) * np.exp(-t * 4)
    sos = butter(4, 3500, btype="high", fs=sr, output="sos")
    return ((sosfilt(sos, noise) * 0.18 + ping * 0.12)).astype(np.float32)


def _pitch_jit(s, factor):
    # 아주 가벼운 피치/길이 변조(타격마다 동일음 반복되는 기계음 느낌 제거)
    if abs(factor - 1.0) < 1e-3:
        return s
    n = len(s)
    idx = np.clip(np.arange(n) * factor, 0, n - 1)
    return np.interp(idx, np.arange(n), s).astype(np.float32)


def _variant_bank(fn, sr, k=3):
    return [fn(sr) for _ in range(k)]   # 서로 다른 노이즈 시드 → 반복 타격이 미묘하게 다름


def render_drums(hits, sr, duration, seed=1234):
    """온셋 히트열 → 합성 드럼. 타격마다 레벨·음색·피치를 미세하게 흔들어 기계음 느낌 제거."""
    rng = np.random.RandomState(seed)
    buf = np.zeros(int(duration * sr) + sr)
    # 음색별 변주 뱅크(노이즈성은 여러 개, 톤성은 1개로 충분)
    banks = {
        "kick": _variant_bank(_kick, sr, 2),
        "snare": _variant_bank(_snare, sr, 3),
        "hat": _variant_bank(_hat, sr, 3),
        "open_hat": _variant_bank(_open_hat, sr, 2),
        "tom": _variant_bank(_tom, sr, 2),
        "crash": _variant_bank(_crash, sr, 2),
        "ride": _variant_bank(_ride, sr, 2),
    }
    # 알 수 없는 라벨은 가까운 음색으로 대체(KeyError 방지)
    alias = {"hihat": "hat", "closed_hat": "hat", "openhat": "open_hat",
             "cymbal": "crash", "clap": "snare"}
    jit_ok = {"kick", "tom"}   # 피치 변조는 피치성 드럼에만
    for h in hits:
        d = h.get("drum", "snare")
        d = alias.get(d, d)
        bank = banks.get(d) or banks["snare"]
        s = bank[rng.randint(len(bank))]
        if d in jit_ok:
            s = _pitch_jit(s, 1.0 + rng.uniform(-0.03, 0.03))
        lvl = (h["vel"] / 127.0) * (1.0 + rng.uniform(-0.08, 0.08))   # ±8% 레벨 흔들기
        s = s * float(np.clip(lvl, 0.05, 1.2))
        start = int(h["time"] * sr)
        if start < 0:
            start = 0
        end = min(start + len(s), len(buf))
        if end > start:
            buf[start:end] += s[: end - start]
    out = buf[: int(duration * sr)]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:                       # 겹침으로 1.0 초과 시에만 안전 스케일
        out = out * (0.99 / peak)
    return out.astype(np.float32)
