# -*- coding: utf-8 -*-
"""
원곡 샘플 기반 재연주(concatenative/auto-sampling).

합성음·VST로 음색을 '흉내'내지 않고, 원곡 스템에서 그 악기의 실제 소리 조각을
떼어 샘플 뱅크를 만든 뒤, 전사한 음정·타이밍에 맞춰 그 샘플을 피치시프트해 배치한다.
→ 재연주본이 원곡 음색 그대로 난다.

- 멜로디(bass/other/guitar/piano): 음표별로 스템 구간을 떼어 뱅크 구성,
  타깃 음정과 가장 가까운 샘플을 골라 반음차만큼 피치시프트 후 배치.
- 드럼: 분류된 타격(kick/snare/hat/tom/cymbal)마다 실제 히트 조각을 떼어
  클래스별 뱅크 구성, 패턴에 맞춰 원곡 드럼 소리로 재배치.
"""
import numpy as np
import librosa


def _fade(x, sr, fin=0.005, fout=0.02):
    n = len(x)
    a = int(fin * sr); b = int(fout * sr)
    if a > 0 and a * 2 < n:
        x[:a] *= np.linspace(0, 1, a, dtype=np.float32)
    if b > 0 and b * 2 < n:
        x[-b:] *= np.linspace(1, 0, b, dtype=np.float32)
    return x


def _slice(y, sr, start, end, pre=0.01, post=0.06):
    s = max(0, int((start - pre) * sr))
    e = min(len(y), int((end + post) * sr))
    if e <= s:
        return None
    seg = np.array(y[s:e], dtype=np.float32)
    return _fade(seg, sr)


def build_melodic_bank(y, sr, notes, max_per_pitch=3):
    """음표별 스템 구간을 떼어 {midi: [샘플,...]} 뱅크 구성. 에너지 큰 것 우선."""
    y = np.asarray(y, dtype=np.float32)
    ns = sorted(notes or [], key=lambda n: n["start"])
    bank = {}
    for i, n in enumerate(ns):
        end = n["end"]
        if i + 1 < len(ns):
            end = min(end, ns[i + 1]["start"])   # 다음 음 전까지(겹침 방지)
        seg = _slice(y, sr, n["start"], end)
        if seg is None or seg.size < int(0.02 * sr):
            continue
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < 1e-4:
            continue
        m = int(n["midi"])
        bank.setdefault(m, []).append((rms, seg))
    # 피치별 에너지 큰 순 상위 N개만
    for m in list(bank.keys()):
        bank[m] = [s for _, s in sorted(bank[m], key=lambda t: -t[0])[:max_per_pitch]]
    return bank


def _nearest_sample(bank, midi):
    if not bank:
        return None, 0
    if midi in bank:
        return bank[midi][0], 0
    best = min(bank.keys(), key=lambda k: abs(k - midi))
    return bank[best][0], midi - best   # (샘플, 반음차)


def render_sampler(notes, y, sr, duration, rng_seed=7):
    """멜로디 음표열을 원곡 샘플로 재연주. 뱅크가 비면 원본 스템 그대로 반환."""
    y = np.asarray(y, dtype=np.float32)
    bank = build_melodic_bank(y, sr, notes)
    out = np.zeros(int(duration * sr) + sr, dtype=np.float32)
    if not bank:
        n = min(len(y), len(out))
        out[:n] = y[:n]
        return out[: int(duration * sr)]
    rng = np.random.RandomState(rng_seed)
    for n in sorted(notes or [], key=lambda x: x["start"]):
        samp, steps = _nearest_sample(bank, int(n["midi"]))
        if samp is None:
            continue
        s = samp
        if steps != 0:
            try:
                s = librosa.effects.pitch_shift(s, sr=sr, n_steps=float(steps))
            except Exception:
                pass
        # 음 길이에 맞춰 자르기(길면 자연 감쇠 유지, 짧으면 그대로) + 페이드
        dur_n = max(int((n["end"] - n["start"]) * sr), int(0.05 * sr))
        s = s[:dur_n] if len(s) > dur_n else s
        s = _fade(np.array(s, dtype=np.float32), sr)
        # 세기: 원곡 샘플 레벨을 기준으로 vel 반영(과증폭 방지)
        amp = 0.6 + 0.4 * (n.get("vel", 90) / 127.0)
        s = s * float(amp)
        st = int(n["start"] * sr)
        if st < 0:
            st = 0
        e = min(st + len(s), len(out))
        if e > st:
            out[st:e] += s[: e - st]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out * (0.99 / peak)
    return out[: int(duration * sr)].astype(np.float32)


# ── 드럼 샘플러 ─────────────────────────────────────────────────────
_DRUM_WIN = {  # 클래스별 떼어낼 길이(초)
    "kick": 0.28, "snare": 0.25, "hat": 0.12, "open_hat": 0.3,
    "tom": 0.3, "crash": 0.8, "ride": 0.6,
}


def build_drum_bank(y, sr, hits, max_per_class=6):
    """분류된 타격마다 실제 히트 조각을 떼어 {class: [샘플,...]} 뱅크 구성."""
    y = np.asarray(y, dtype=np.float32)
    bank = {}
    for h in hits or []:
        d = h.get("drum", "snare")
        win = _DRUM_WIN.get(d, 0.25)
        s = max(0, int(h["time"] * sr))
        e = min(len(y), s + int(win * sr))
        if e <= s:
            continue
        seg = _fade(np.array(y[s:e], dtype=np.float32), sr, fin=0.001, fout=0.01)
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < 1e-4:
            continue
        bank.setdefault(d, []).append((rms, seg))
    for d in list(bank.keys()):
        bank[d] = [s for _, s in sorted(bank[d], key=lambda t: -t[0])[:max_per_class]]
    return bank


def render_drum_sampler(hits, y, sr, duration, rng_seed=11):
    """드럼 패턴을 원곡의 실제 드럼 히트 소리로 재배치. 뱅크 비면 원본 반환."""
    y = np.asarray(y, dtype=np.float32)
    bank = build_drum_bank(y, sr, hits)
    out = np.zeros(int(duration * sr) + sr, dtype=np.float32)
    if not bank:
        n = min(len(y), len(out))
        out[:n] = y[:n]
        return out[: int(duration * sr)]
    rng = np.random.RandomState(rng_seed)
    # 클래스 대체(알 수 없는 라벨 → 가까운 것)
    alias = {"open_hat": "hat", "ride": "crash"}
    for h in hits or []:
        d = h.get("drum", "snare")
        pool = bank.get(d) or bank.get(alias.get(d, ""), None) or bank.get("snare") or next(iter(bank.values()))
        s = pool[rng.randint(len(pool))]
        amp = 0.6 + 0.4 * (h.get("vel", 100) / 127.0)
        s = np.array(s, dtype=np.float32) * float(amp)
        st = max(0, int(h["time"] * sr))
        e = min(st + len(s), len(out))
        if e > st:
            out[st:e] += s[: e - st]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out * (0.99 / peak)
    return out[: int(duration * sr)].astype(np.float32)
