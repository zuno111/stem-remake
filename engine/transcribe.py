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


def transcribe_mono(y, sr, fmin=41.0, fmax=2000.0, min_dur=0.08, conf_thresh=0.45):
    """pYIN 단선율 전사. 베이스/리드 멜로디용.

    conf_thresh: pYIN voiced_prob(유성음 신뢰도) 하한. 이 값 미만 프레임은 전사에서 제외해
    노이즈/무성 구간의 유령 음을 줄인다. 0이면 신뢰도 게이트 끔.
    """
    # fmin에서 두 주기 이상 들어가도록 frame_length 확보(저역 정확도)
    frame_length = max(2048, int(2 ** np.ceil(np.log2(2.5 * sr / fmin))))
    hop = 256
    f0, voiced, vprob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop
    )
    if vprob is None:
        vprob = np.where(voiced, 1.0, 0.0)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)

    # 유효 프레임: 유성 + 유한 f0 + 양수 주파수 + 신뢰도 임계 통과
    valid = voiced & np.isfinite(f0) & (np.nan_to_num(f0, nan=0.0) > 0.0)
    if conf_thresh > 0:
        valid = valid & (vprob >= conf_thresh)
    # 유효 프레임에서만 hz_to_midi 계산 → log2(0) divide-by-zero 경고 제거
    midi = np.full(len(f0), np.nan)
    if valid.any():
        midi[valid] = np.round(librosa.hz_to_midi(f0[valid]))

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
            cur["_c"].append(vprob[i])
        else:
            if cur:
                notes.append(cur)
            cur = {"start": float(times[i]), "end": float(times[i]), "midi": int(m),
                   "_e": [rms[i]], "_c": [vprob[i]]}
    if cur:
        notes.append(cur)

    out = []
    emax = max((max(n["_e"]) for n in notes), default=1e-9) or 1e-9
    for n in notes:
        if n["end"] - n["start"] < min_dur:
            continue
        # 평균 신뢰도가 임계의 절반에도 못 미치는 짧은 노트는 유령음으로 보고 제외
        if conf_thresh > 0 and float(np.mean(n["_c"])) < conf_thresh * 0.9 \
                and (n["end"] - n["start"]) < min_dur * 2:
            continue
        vel = int(np.clip(90 * (np.mean(n["_e"]) / emax) + 30, 20, 127))
        out.append({"start": n["start"], "end": n["end"], "midi": n["midi"], "vel": vel})
    return out


def transcribe_drums(y, sr):
    """
    온셋별 드럼 분류. kick/snare/hat 외에 tom/crash 세분화.
    스펙트럴 센트로이드 + 저역/고역 에너지 비율 + 감쇠 길이로 판별.
    """
    hop = 256
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    # 대역 에너지(저역 <150Hz, 고역 >6kHz)
    S = np.abs(librosa.stft(y, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr)
    low_e = S[freqs < 150].sum(axis=0)
    high_e = S[freqs > 6000].sum(axis=0)
    total_e = S.sum(axis=0) + 1e-9
    emax = float(rms.max() or 1e-9)

    def decay_frames(f):
        # 온셋 후 에너지가 초기의 30%로 떨어지는 데 걸리는 프레임(심벌류는 길다)
        if f >= len(rms):
            return 0
        peak = rms[f] + 1e-9
        c = 0
        for i in range(f, min(f + 60, len(rms))):
            if rms[i] < peak * 0.3:
                break
            c += 1
        return c

    hits = []
    for f, t in zip(onset_frames, onset_times):
        fi = min(f, len(cent) - 1)
        c = cent[fi]
        lo = low_e[fi] / total_e[fi]
        hi = high_e[fi] / total_e[fi]
        dec = decay_frames(f)

        if lo > 0.35 and c < 300:
            drum = "kick"
        elif hi > 0.45 and dec > 25:
            drum = "crash"            # 고역 풍부 + 긴 감쇠
        elif c > 4500 and dec <= 20:
            drum = "hat"              # 고역 + 짧은 감쇠
        elif 150 <= c < 800 and lo > 0.15:
            drum = "tom"              # 중저역 피치성
        elif c < 3500:
            drum = "snare"
        else:
            drum = "hat"
        vel = int(np.clip(90 * (rms[min(f, len(rms) - 1)] / emax) + 30, 20, 127))
        hits.append({"time": float(t), "drum": drum, "vel": vel})
    return hits
