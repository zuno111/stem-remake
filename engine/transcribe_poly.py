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


def transcribe_polyphonic(y, sr):
    from basic_pitch.inference import predict

    tmp = os.path.join(tempfile.gettempdir(), f"bp_{os.getpid()}.wav")
    sf.write(tmp, np.asarray(y, dtype=np.float32), sr)
    try:
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
