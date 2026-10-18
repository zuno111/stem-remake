# -*- coding: utf-8 -*-
"""
MDX23C DrumSep(aufr33-jarredou) 모델로 드럼 스템을 조각으로 분리.

audio-separator로 실제 학습 모델을 돌려 킥/스네어/탐/하이햇/심벌을 분리한다.
band-pass 근사(drumsplit.py)보다 훨씬 깨끗하다. 모델·ffmpeg가 없으면 None을 반환해
호출부가 band-pass 폴백을 쓰게 한다.

- 모델 파일 위치: STEMREMAKE_MODELS_DIR 환경변수(번들된 경로) 우선, 없으면 기본 캐시.
- ffmpeg: audio-separator가 내부적으로 필요. 번들 exe는 시작 시 PATH에 ffmpeg를 추가함.
"""
import os
import glob

_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"

# DrumSep 출력 스템 이름(부분일치) → 우리 조각 라벨
_NAME_MAP = [
    ("kick", "kick"), ("snare", "snare"),
    ("tom", "tom"),
    ("hh", "hat"), ("hihat", "hat"), ("hi-hat", "hat"), ("hat", "hat"),
    ("ride", "cymbal"), ("crash", "cymbal"), ("cymbal", "cymbal"),
]


def available():
    try:
        import audio_separator  # noqa: F401
        return True
    except Exception:
        return False


def _map_piece(stem_name):
    low = stem_name.lower()
    for key, piece in _NAME_MAP:
        if key in low:
            return piece
    return None


def split_drums_model(drum_wav_path, out_dir):
    """드럼 wav를 MDX23C DrumSep로 분리. 반환 {piece: wav경로} 또는 None(실패/미설치)."""
    try:
        from audio_separator.separator import Separator
    except Exception as e:
        print(f"[drumsep_model] audio-separator 미설치 → 폴백: {e}")
        return None
    try:
        model_dir = os.environ.get("STEMREMAKE_MODELS_DIR") or os.path.join(out_dir, "_models")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        try:
            sep = Separator(output_dir=out_dir, model_file_dir=model_dir,
                            log_level=40, output_format="WAV")
        except TypeError:
            sep = Separator(output_dir=out_dir, model_file_dir=model_dir, log_level=40)
        sep.load_model(model_filename=_MODEL)
        files = sep.separate(drum_wav_path)
        out = {}
        for f in files:
            p = f if os.path.isabs(f) else os.path.join(out_dir, f)
            if not os.path.exists(p):
                # 일부 버전은 파일명만 반환 — out_dir에서 매칭 탐색
                cand = glob.glob(os.path.join(out_dir, "*" + os.path.basename(f) + "*"))
                p = cand[0] if cand else p
            base = os.path.splitext(os.path.basename(p))[0]
            piece = _map_piece(base)
            if piece and os.path.exists(p):
                out[piece] = p
        return out or None
    except Exception as e:
        print(f"[drumsep_model] 분리 실패 → 폴백: {e}")
        return None
