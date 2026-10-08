# -*- coding: utf-8 -*-
"""
전사 결과 사후 양자화(박자 그리드 스냅).
BPM과 비트 시작점으로 그리드를 만들고, 노트/드럼 이벤트의 시작·끝을
가장 가까운 분할선(기본 16분음표)으로 스냅한다. strength로 강약 조절.
"""
import numpy as np


def _grid(bpm, duration, beat_offset=0.0, division=4):
    """division: 한 박당 분할 수(4=16분음표). 그리드 시각 배열 반환."""
    if not bpm or bpm <= 0:
        return None
    step = (60.0 / bpm) / division
    n = int(np.ceil((duration + step) / step)) + 1
    return beat_offset + np.arange(n) * step, step


def _snap(t, grid, step, strength):
    idx = int(round((t - grid[0]) / step))
    idx = max(0, min(idx, len(grid) - 1))
    target = grid[idx]
    return t + (target - t) * strength


def quantize_notes(notes, bpm, duration, beat_offset=0.0, division=4, strength=0.8, min_len=0.03):
    """멜로디/베이스 노트 start·end를 그리드에 스냅."""
    g = _grid(bpm, duration, beat_offset, division)
    if g is None:
        return notes
    grid, step = g
    out = []
    for n in notes:
        s = _snap(n["start"], grid, step, strength)
        e = _snap(n["end"], grid, step, strength)
        if e - s < min_len:
            e = s + step  # 너무 짧아지면 최소 한 칸
        m = dict(n)
        m["start"], m["end"] = float(s), float(e)
        out.append(m)
    return out


def quantize_drums(hits, bpm, duration, beat_offset=0.0, division=4, strength=0.9):
    """드럼 타격 time을 그리드에 스냅."""
    g = _grid(bpm, duration, beat_offset, division)
    if g is None:
        return hits
    grid, step = g
    out = []
    for h in hits:
        m = dict(h)
        m["time"] = float(_snap(h["time"], grid, step, strength))
        out.append(m)
    return out
