# -*- coding: utf-8 -*-
"""
다성(polyphonic) 전사 — basic-pitch(Spotify) 래퍼. 제품 빌드 v2에서 사용.
basic_pitch 미설치 환경에서는 import 자체를 하지 않으므로 안전.
반환: pipeline이 쓰는 노트 형식 리스트 {"start","end","midi","vel"}.
"""
import os
import tempfile
import numpy as np
import soundfile as sf


def transcribe_polyphonic(y, sr, fmin=40.0, fmax=3000.0,
                          onset_threshold=0.55, frame_threshold=0.33,
                          min_note_ms=130.0):
    """basic-pitch 다성 전사. 악기 음역(fmin~fmax)과 임계값을 받아 유령 노트를 줄인다.

    - fmin/fmax: 해당 파트 음역 밖 스퓨리어스 노트 제거(기타외·기타·피아노에 맞춰 전달)
    - onset/frame_threshold: 기본값보다 약간 높여 약한 오검출 억제
    - min_note_ms: 너무 짧은 블립 제거
    설치 버전이 일부 인자를 받지 않으면 기본 호출로 안전 폴백.
    """
    from basic_pitch.inference import predict

    tmp = os.path.join(tempfile.gettempdir(), f"bp_{os.getpid()}.wav")
    sf.write(tmp, np.asarray(y, dtype=np.float32), sr)
    try:
        kwargs = dict(onset_threshold=onset_threshold, frame_threshold=frame_threshold,
                      minimum_note_length=min_note_ms,
                      minimum_frequency=fmin, maximum_frequency=fmax)
        try:
            _, _, note_events = predict(tmp, **kwargs)
        except TypeError:
            # 설치 버전이 해당 kwargs를 모르면 기본 호출로 폴백
            _, _, note_events = predict(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    notes = []
    for ev in note_events:
        start, end, pitch, amp = ev[0], ev[1], ev[2], ev[3]
        notes.append({
            "start": float(start), "end": float(end),
            "midi": int(pitch), "vel": int(np.clip(amp * 127, 20, 127)),
        })
    return notes
