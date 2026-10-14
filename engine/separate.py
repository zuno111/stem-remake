# -*- coding: utf-8 -*-
"""
스템 분리 모듈.

두 가지 백엔드:
- demucs: torch + Demucs(htdemucs) 모델. 고품질. 제품 빌드(GitHub Actions)에서 사용.
- lite:   librosa 기반 DSP 분리(REPET-SIM + HPSS). torch 없이 동작. 샌드박스/폴백용.

둘 다 입력 오디오를 vocals / drums / bass / other 4스템으로 나눠 wav로 저장한다.
backend="auto"면 demucs가 설치돼 있고 모델을 받을 수 있으면 demucs, 아니면 lite.
"""
import os
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt

STEMS = ["vocals", "drums", "bass", "other"]
STEMS_6 = ["vocals", "drums", "bass", "guitar", "piano", "other"]


def _have_demucs():
    try:
        import torch  # noqa: F401
        import demucs  # noqa: F401
        return True
    except Exception:
        return False


def separate_demucs(input_path, out_dir, model_name="htdemucs"):
    """Demucs로 스템 분리. model_name='htdemucs_6s'면 6스템(guitar·piano 포함).

    오디오 로딩은 demucs 기본 AudioFile(ffmpeg 외부 실행파일 의존) 대신
    librosa(순수 파이썬/libsndfile)로 한다. 윈도우 등에서 ffmpeg가 없어도
    demucs가 동작하도록(ffmpeg 미설치 시 WinError 2로 폴백되던 문제 해결).
    """
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model(model_name)
    model.eval()
    ch = int(getattr(model, "audio_channels", 2))

    # ffmpeg 없이 로드: librosa(소프트웨어 디코딩). 모델 샘플레이트로 리샘플.
    y, _ = librosa.load(input_path, sr=model.samplerate, mono=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = np.stack([y] * ch, axis=0)        # 모노 → 필요한 채널 수로 복제
    elif y.shape[0] == 1 and ch > 1:
        y = np.repeat(y, ch, axis=0)
    elif y.shape[0] > ch:
        y = y[:ch]
    elif y.shape[0] < ch:
        y = np.vstack([y] + [y[-1:]] * (ch - y.shape[0]))

    wav = torch.from_numpy(np.ascontiguousarray(y))   # (channels, samples)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(model, wav[None], device="cpu", progress=True)[0]
    sources = sources * ref.std() + ref.mean()

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for stem in model.sources:  # 모델이 내놓는 스템 전부 저장
        idx = model.sources.index(stem)
        audio = sources[idx].cpu().numpy().T  # (samples, channels)
        p = os.path.join(out_dir, f"{stem}.wav")
        sf.write(p, audio, model.samplerate)
        paths[stem] = p
    return paths, model.samplerate


def separate_lite(input_path, out_dir, sr=44100):
    """
    torch 없이 도는 DSP 분리.
    1) REPET-SIM(nn_filter)로 보컬/반주 분리
    2) 반주를 HPSS로 타악(drums)/화성(harmonic) 분리
    3) 화성을 저역 필터로 bass / other 분리
    고품질은 아니지만 파이프라인 전체를 실제로 돌려 검증하는 용도.
    """
    y, sr = librosa.load(input_path, sr=sr, mono=True)

    S_full, phase = librosa.magphase(librosa.stft(y))
    n_frames = S_full.shape[-1]
    # nn_filter width는 1 이상, n_frames//2 - 1 이하여야 함(짧은 클립 크래시 방지)
    win = max(1, int(librosa.time_to_frames(2.0, sr=sr)))
    # librosa 제약: width <= (n_frames-1)//2. 경계 포함 오류 방지 위해 1 더 뺌.
    win = min(win, max(1, (n_frames - 1) // 2 - 1))
    S_filter = librosa.decompose.nn_filter(
        S_full, aggregate=np.median, metric="cosine", width=win
    )
    S_filter = np.minimum(S_full, S_filter)
    margin_i, margin_v, power = 2, 10, 2
    mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)
    mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=power)
    vocals = librosa.istft(mask_v * S_full * phase)
    accomp = librosa.istft(mask_i * S_full * phase)

    harm, perc = librosa.effects.hpss(accomp)
    drums = perc
    sos = butter(4, 250, btype="low", fs=sr, output="sos")
    bass = sosfilt(sos, harm)
    other = harm - bass

    os.makedirs(out_dir, exist_ok=True)
    stems = {"vocals": vocals, "drums": drums, "bass": bass, "other": other}
    paths = {}
    for stem, audio in stems.items():
        p = os.path.join(out_dir, f"{stem}.wav")
        sf.write(p, audio.astype(np.float32), sr)
        paths[stem] = p
    return paths, sr


def separate(input_path, out_dir, backend="auto", sr=44100, model_name="htdemucs"):
    """(paths, sr, used_backend) 반환. used_backend는 실제로 성공한 백엔드."""
    if backend == "auto":
        backend = "demucs" if _have_demucs() else "lite"
    if backend == "demucs":
        try:
            paths, msr = separate_demucs(input_path, out_dir, model_name=model_name)
            return paths, msr, "demucs"
        except Exception as e:
            print(f"[separate] demucs 실패 → lite 폴백: {e}")
            backend = "lite"
    paths, msr = separate_lite(input_path, out_dir, sr=sr)
    return paths, msr, "lite"
