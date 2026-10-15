# -*- coding: utf-8 -*-
"""
음색 전사(timbre transfer) 렌더러.

고정 합성 악기로 바꾸는 대신, 원본 스템에서 음색(스펙트럼 포락선)을 떠서
목표 음정의 하모닉 여기신호에 입힌다. 전자음·신스·특이 음색을 고정 악기 몇 종으로
욱여넣지 않고, 원래 성격을 유지한 채 음정·타이밍만 전사 결과로 다시 연주한다.

소스-필터 교차합성:
  여기신호(목표 음정 톱니, 배음 풍부) 크기 스펙트럼 × 원본 포락선(켑스트럼 리프터) → ISTFT
"""
import numpy as np
import librosa


def _midi_to_hz(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def _spectral_envelope(S_mag, lifter=40):
    """켑스트럼 리프터링으로 피치 미세구조 제거 → 음색 포락선만 남김."""
    logS = np.log(S_mag + 1e-8)
    cep = np.fft.irfft(logS, axis=0)
    if lifter < cep.shape[0]:
        cep[lifter:cep.shape[0] - lifter] = 0.0
    env = np.exp(np.fft.rfft(cep, axis=0).real)
    return env


def _transfer_note(seg, sr, target_hz, n_fft=2048, hop=512):
    """원본 조각 seg의 음색을 목표 음정으로 전사한 오디오."""
    n = len(seg)
    if n < n_fft:
        seg = np.pad(seg, (0, n_fft - n))
    S = librosa.stft(seg, n_fft=n_fft, hop_length=hop)
    mag = np.abs(S)
    env = _spectral_envelope(mag)
    env /= (env.max(axis=0, keepdims=True) + 1e-8)

    t = np.arange(max(n, n_fft)) / sr
    exc = (2 * ((t * target_hz) % 1.0) - 1).astype(np.float32)
    E = librosa.stft(exc, n_fft=n_fft, hop_length=hop)
    emag, ephase = np.abs(E), np.angle(E)

    f = min(env.shape[1], emag.shape[1])
    out = librosa.istft(emag[:, :f] * env[:, :f] * np.exp(1j * ephase[:, :f]),
                        hop_length=hop, length=max(n, n_fft))

    # 원본 진폭(음량) 포락선 반영
    rms_src = librosa.feature.rms(y=seg, hop_length=hop)[0]
    rms_out = librosa.feature.rms(y=out, hop_length=hop)[0] + 1e-8
    idx = np.arange(len(out))
    gs = np.interp(idx, np.arange(len(rms_src)) * hop, rms_src)
    go = np.interp(idx, np.arange(len(rms_out)) * hop, rms_out)
    out = out * np.clip(gs / go, 0.0, 8.0)
    return out[:n].astype(np.float32)


def render_notes(notes, stem_y, sr, duration):
    """
    전사된 노트열을 원본 스템(stem_y)의 음색으로 다시 연주.
    notes: [{start,end,midi,vel}]. stem_y: 해당 파트 원본 오디오(모노).
    """
    total = int(duration * sr) + sr
    buf = np.zeros(total, dtype=np.float32)
    if stem_y is None or len(stem_y) == 0 or not notes:
        return buf[: int(duration * sr)]
    for nt in notes:
        start = max(0, int(nt["start"] * sr))
        end = int(nt["end"] * sr)
        if end <= start or start >= len(stem_y):
            continue
        src_end = min(end, len(stem_y))
        seg = stem_y[start:src_end].astype(np.float32)
        if len(seg) < 256:
            continue
        try:
            voice = _transfer_note(seg, sr, _midi_to_hz(nt["midi"]))
        except Exception:
            continue
        amp = 0.4 + 0.6 * (nt.get("vel", 90) / 127.0)
        # 짧은 페이드로 클릭 방지
        fade = min(256, len(voice) // 4)
        if fade > 0:
            voice[:fade] *= np.linspace(0, 1, fade)
            voice[-fade:] *= np.linspace(1, 0, fade)
        e = min(start + len(voice), total)
        buf[start:e] += voice[: e - start] * amp
    return buf[: int(duration * sr)]
