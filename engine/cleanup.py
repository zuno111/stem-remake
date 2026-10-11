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


def fix_octave_spikes(notes, win=2):
    """
    '주변이 한 음으로 안정적인데 한 음만 ±1옥타브로 튄' 진짜 글리치만 교정한다.

    교대(E1↔E2)·옥타브 도약 같은 정상 선율을 망가뜨리지 않도록 보수적으로:
    인접한 최대 4개 이웃 중 과반(>=3)이 한 음(±1반음)으로 모일 때만 그 음을 '지배음'으로
    보고, 현재 음이 지배음에서 ±(11~13)반음이면 지배음 옥타브로 당긴다. 그 외는 그대로 둔다.
    """
    if len(notes) < 3:
        return notes
    ns = sorted(notes, key=lambda n: n["start"])
    midis = [n["midi"] for n in ns]
    out = []
    for i, n in enumerate(ns):
        lo = max(0, i - win); hi = min(len(midis), i + win + 1)
        neigh = midis[lo:i] + midis[i + 1:hi]
        corrected = False
        if len(neigh) >= 2:
            # 이웃 중 ±1반음 내로 가장 많이 모이는 음(지배음)과 그 표수
            best_pitch, best_count = None, 0
            for cand in neigh:
                cnt = sum(1 for m in neigh if abs(m - cand) <= 1)
                if cnt > best_count:
                    best_count, best_pitch = cnt, cand
            # 이웃 중 3개 이상이 한 음(±1반음)으로 모일 때만 지배음으로 인정(경계 오교정 방지)
            dominant = best_count >= 3
            if dominant and best_pitch is not None and 11 <= abs(n["midi"] - best_pitch) <= 13:
                octs = round((n["midi"] - best_pitch) / 12.0)
                m = dict(n); m["midi"] = n["midi"] - 12 * octs
                out.append(m); corrected = True
        if not corrected:
            out.append(dict(n))
    return out


def clean_mono(notes, min_dur=0.09, gap_tol=0.03, fix_octave=True):
    out = notes
    if fix_octave:
        out = fix_octave_spikes(out)
    out = merge_adjacent(out, gap_tol=gap_tol)
    out = remove_short(out, min_dur=min_dur)
    return out


def remove_octave_overtones(notes, vel_margin=25, overlap_tol=0.05):
    """다성 전사의 옥타브 배음 유령음 제거(보수적).

    현재 음이, 시간상 겹치는 더 낮은 음(기음)에서 정확히 +12/+24 반음 위이고
    벨로시티가 vel_margin 이상 낮으면 배음 유령으로 보고 제거한다.
    실제 옥타브 중복(기음과 비슷한 세기)이나 비옥타브 음은 건드리지 않는다.
    """
    if len(notes) < 2:
        return notes
    out = []
    for n in notes:
        ns, ne, nm, nv = n["start"], n["end"], n["midi"], n.get("vel", 90)
        is_overtone = False
        for m in notes:
            if m is n:
                continue
            # 시간 겹침
            if min(ne, m["end"]) - max(ns, m["start"]) <= overlap_tol:
                continue
            d = nm - m["midi"]
            if d in (12, 24) and m.get("vel", 90) - nv >= vel_margin:
                is_overtone = True
                break
        if not is_overtone:
            out.append(dict(n))
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
