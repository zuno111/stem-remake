# -*- coding: utf-8 -*-
"""
전사 노트 사후 정리. 실제 곡에서 흔한 전사 오류를 보수적으로 줄인다.
- remove_short: 너무 짧은 유령 노트 제거
- merge_adjacent: 간격이 거의 없는 같은 음 병합
- fix_octave_spikes: 한 음만 ±12 근처로 튀었다 바로 돌아오는 옥타브 스파이크 교정
순서: octave → merge → remove 가 자연스럽다.
"""
import numpy as np


def remove_short(notes, min_dur=0.09):
    return [n for n in notes if (n["end"] - n["start"]) >= min_dur]


def merge_adjacent(notes, gap_tol=0.03):
    """시간순 정렬 후, 같은 midi이고 간격이 gap_tol 이하인 노트를 이어붙임."""
    if not notes:
        return notes
    ns = sorted(notes, key=lambda n: n["start"])
    out = [dict(ns[0])]
    for n in ns[1:]:
        last = out[-1]
        if n["midi"] == last["midi"] and n["start"] - last["end"] <= gap_tol:
            last["end"] = max(last["end"], n["end"])
            last["vel"] = max(last.get("vel", 90), n.get("vel", 90))
        else:
            out.append(dict(n))
    return out


def fix_octave_spikes(notes, win=5):
    """
    한 음이 주변 median에서 ±12(±1옥타브) 근처로 튄 경우, 주변 median 옥타브로 당김.
    선율의 진짜 옥타브 도약은 보통 연속 유지되므로, 고립된 1음 스파이크만 교정.
    """
    if len(notes) < 3:
        return notes
    ns = sorted(notes, key=lambda n: n["start"])
    midis = [n["midi"] for n in ns]
    out = []
    for i, n in enumerate(ns):
        lo = max(0, i - win); hi = min(len(midis), i + win + 1)
        neigh = midis[lo:i] + midis[i + 1:hi]
        if not neigh:
            out.append(dict(n)); continue
        med = int(np.median(neigh))
        diff = n["midi"] - med
        # ±1옥타브(±(12±1)) 근처로 고립되게 튄 경우만 교정
        if 11 <= abs(diff) <= 13:
            octs = round(diff / 12.0)
            m = dict(n); m["midi"] = n["midi"] - 12 * octs
            out.append(m)
        else:
            out.append(dict(n))
    return out


def clean_mono(notes, min_dur=0.09, gap_tol=0.03, fix_octave=True):
    out = notes
    if fix_octave:
        out = fix_octave_spikes(out)
    out = merge_adjacent(out, gap_tol=gap_tol)
    out = remove_short(out, min_dur=min_dur)
    return out


def dedup_drums(hits, min_gap=0.04):
    """같은 드럼 종류가 min_gap 이내 연속되면 뒤엣것 제거(중복 온셋)."""
    if not hits:
        return hits
    hs = sorted(hits, key=lambda h: h["time"])
    out = [dict(hs[0])]
    last_by = {out[0]["drum"]: out[0]["time"]}
    for h in hs[1:]:
        t0 = last_by.get(h["drum"])
        if t0 is not None and h["time"] - t0 < min_gap:
            continue
        out.append(dict(h))
        last_by[h["drum"]] = h["time"]
    return out
