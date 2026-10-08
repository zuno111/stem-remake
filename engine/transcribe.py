# -*- coding: utf-8 -*-
"""
스템 전사(audio -> 노트/이벤트).

- transcribe_mono: 베이스/멜로디처럼 단선율 스템을 pYIN으로 MIDI 노트열로 전사.
- transcribe_drums: 타악 스템을 온셋 검출 + 스펙트럴 센트로이드로 kick/snare/hat 분류.

제품 빌드에서는 transcribe_mono 자리에 basic-pitch(다성 전사)를 끼운다.
반환 노트 형식: {"start","end","midi","vel"} 리스트.
"""
import numpy as np
import librosa


def transcribe_mono(y, sr, fmin=41.0, fmax=2000.0, min_dur=0.08):
    """pYIN 단선율 전사. 베이스/리드 멜로디용."""
    # fmin에서 두 주기 이상 들어가도록 frame_length 확보(저역 정확도)
    frame_length = max(2048, int(2 ** np.ceil(np.log2(2.5 * sr / fmin))))
    f0, voiced, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=256
    )
    hop = 256
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    midi = np.where(voiced & np.isfinite(f0), librosa.hz_to_midi(np.nan_to_num(f0, nan=0.0)), np.nan)
    midi = np.round(midi)

    # 에너지(세기) 추출
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    rms = np.interp(np.arange(len(midi)), np.linspace(0, len(midi) - 1, len(rms)), rms)

    notes = []
    cur = None
    for i, m in enumerate(midi):
        if np.isnan(m):
            if cur:
                notes.append(cur); cur = None
            continue
        if cur and abs(m - cur["midi"]) < 0.5:
            cur["end"] = times[i]
            cur["_e"].append(rms[i])
        else:
            if cur:
                notes.append(cur)
            cur = {"start": float(times[i]), "end": float(times[i]), "midi": int(m), "_e": [rms[i]]}
    if cur:
        notes.append(cur)

    out = []
    emax = max((max(n["_e"]) for n in notes), default=1e-9) or 1e-9
    for n in notes:
        if n["end"] - n["start"] < min_dur:
            continue
        vel = int(np.clip(90 * (np.mean(n["_e"]) / emax) + 30, 20, 127))
        out.append({"start": n["start"], "end": n["end"], "midi": n["midi"], "vel": vel})
    return out


def transcribe_drums(y, sr):
    """온셋별 kick/snare/hat 분류."""
    hop = 256
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    emax = float(rms.max() or 1e-9)

    hits = []
    for f, t in zip(onset_frames, onset_times):
        fi = min(f, len(cent) - 1)
        c = cent[fi]
        if c < 800:
            drum = "kick"
        elif c < 3500:
            drum = "snare"
        else:
            drum = "hat"
        vel = int(np.clip(90 * (rms[min(f, len(rms) - 1)] / emax) + 30, 20, 127))
        hits.append({"time": float(t), "drum": drum, "vel": vel})
    return hits
