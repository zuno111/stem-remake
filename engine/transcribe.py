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


def transcribe_mono(y, sr, fmin=41.0, fmax=2000.0, min_dur=0.08, conf_thresh=0.1):
    """pYIN 단선율 전사. 베이스/리드 멜로디용.

    conf_thresh: pYIN voiced_prob(유성음 신뢰도) 하한. 이 값 미만 프레임은 전사에서 제외해
    노이즈/무성 구간의 유령 음을 줄인다. 0이면 신뢰도 게이트 끔.
    주의: 저역(베이스)은 pYIN 신뢰도가 본래 낮으므로(0.2~0.5) 기본값을 낮게 둔다.
    pYIN의 voiced 플래그가 이미 노이즈를 걸러주어, 이 게이트는 경계 프레임만 보수적으로 제거한다.
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

    - 맨 앞(t≈0) 온셋을 놓치지 않도록 앞에 짧은 무음을 패딩해 검출(그 뒤 시간 보정).
    - 고역 그룹(hat/snare/crash)은 센트로이드만으로 가르면 밝은 스네어를 하이햇으로
      오분류하므로 감쇠 길이 + 고역 비율로 구분한다.
    """
    hop = 256
    pad_s = 0.1                     # 앞 100ms 무음 패딩 → 시작 지점 온셋 검출 보장
    pad_n = int(pad_s * sr)
    yp = np.concatenate([np.zeros(pad_n, dtype=y.dtype), y])

    onset_frames = librosa.onset.onset_detect(y=yp, sr=sr, hop_length=hop, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop) - pad_s
    cent = librosa.feature.spectral_centroid(y=yp, sr=sr, hop_length=hop)[0]
    rms = librosa.feature.rms(y=yp, hop_length=hop)[0]
    # 대역 에너지(저역 <150Hz, 중역 200~3kHz=스네어 몸통, 고역 >6kHz)
    S = np.abs(librosa.stft(yp, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr)
    low_e = S[freqs < 150].sum(axis=0)
    mid_e = S[(freqs >= 200) & (freqs <= 3000)].sum(axis=0)
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
        md = mid_e[fi] / total_e[fi]
        hi = high_e[fi] / total_e[fi]
        dec = decay_frames(f)
        vel = int(np.clip(90 * (rms[min(f, len(rms) - 1)] / emax) + 30, 20, 127))
        tt = float(max(0.0, t))

        # 한 온셋에 여러 악기가 동시 타격될 수 있어 다중 라벨로 분해한다.
        labels = []
        if c < 800:
            # 저역 중심: kick 또는 tom(피치성·긴 감쇠). 그 위에 올라탄 밝은 햇은 별도.
            if 150 <= c and 0.12 < lo <= 0.45 and dec > 20:
                labels.append("tom")
            else:
                labels.append("kick")
            if hi > 0.35 and md < 0.09:
                labels.append("hat")
        else:
            # 고역 중심: hat / snare / crash 중 주 음색 하나 + 밑에 깔린 kick
            if dec > 25 and hi > 0.5:
                labels.append("crash")
            elif md > 0.09:
                labels.append("snare")   # 중역 몸통 있음
            else:
                labels.append("hat")     # 몸통 거의 없는 고역 트랜지언트
            if lo > 0.08:
                labels.append("kick")    # 밑에 깔린 킥

        for d in labels:
            hits.append({"time": tt, "drum": d, "vel": vel})
    return hits
