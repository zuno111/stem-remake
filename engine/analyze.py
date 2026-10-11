# -*- coding: utf-8 -*-
"""
전체 곡 분석: BPM, 키(조성), 박자 그리드.
librosa 기반으로 torch 없이 동작.
"""
import numpy as np
import librosa

# 키 추정용 Krumhansl-Schmuckler 프로파일
_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    prof = chroma.mean(axis=1)
    prof = prof / (prof.sum() + 1e-9)
    best = (-1.0, "C", "major")
    for i in range(12):
        maj = np.corrcoef(np.roll(_MAJ, i), prof)[0, 1]
        minr = np.corrcoef(np.roll(_MIN, i), prof)[0, 1]
        if maj > best[0]:
            best = (maj, _NOTES[i], "major")
        if minr > best[0]:
            best = (minr, _NOTES[i], "minor")
    return {"key": best[1], "mode": best[2], "confidence": round(float(best[0]), 3)}


def _fold_tempo(bpm, lo=75.0, hi=150.0):
    """옥타브 오류(절반/두 배 템포) 보정: 선호 범위로 ×2/÷2 접기.
    librosa beat_track가 하프타임/더블타임에 락되는 흔한 오류를 완화한다.
    극단적 실제 템포는 UI의 수동 BPM 입력으로 대응."""
    if bpm <= 0:
        return bpm
    b = float(bpm)
    for _ in range(4):
        if b < lo:
            b *= 2.0
        elif b >= hi:
            b /= 2.0
        else:
            break
    return b


def analyze(path, sr=44100):
    y, sr = librosa.load(path, sr=sr, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo_raw = float(np.atleast_1d(tempo)[0])
    tempo = _fold_tempo(tempo_raw)
    beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
    key = estimate_key(y, sr)
    return {
        "bpm": round(tempo, 2),
        "bpm_raw": round(tempo_raw, 2),
        "key": key["key"],
        "mode": key["mode"],
        "key_confidence": key["confidence"],
        "duration_sec": round(len(y) / sr, 2),
        "beat_times": beat_times,
        "sr": sr,
    }
