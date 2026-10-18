# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 스펙 — 스템 리메이크 단독 실행 빌드(onedir).
import os
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

# 무거운 패키지는 데이터·바이너리·서브모듈을 통째로 수집
# numpy를 명시적으로 먼저 수집 — 미포함 시 numpy._core._exceptions 누락으로 시작 크래시
for pkg in [
    "numpy",
    "librosa", "soundfile", "pedalboard", "pretty_midi", "mido",
    "demucs", "torch", "torchaudio", "julius", "openunmix",
    "dora", "lameenc", "einops", "scipy", "audioread",
    "soxr", "numba", "llvmlite", "sklearn", "flask", "waitress",
    "basic_pitch", "onnxruntime", "mir_eval", "resampy",
    "noisereduce",
    # DrumSep(audio-separator) + 의존 — 드럼 조각 실제 모델 분리
    "audio_separator", "onnx2torch", "ml_collections",
    "rotary_embedding_torch", "beartype", "pydub", "samplerate",
]:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f"[spec] collect_all skip {pkg}: {e}")

# 앱 정적 리소스
ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
datas += [(os.path.join(ROOT, "app", "static"), os.path.join("app", "static"))]

# ffmpeg 동봉(audio-separator가 내부적으로 필요). 번들 루트에 두고 런처가 PATH에 추가.
_ff = os.path.join(ROOT, "build", "ffmpeg", "ffmpeg.exe")
if os.path.isfile(_ff):
    binaries += [(_ff, ".")]
    print(f"[spec] ffmpeg 번들: {_ff}")
else:
    print(f"[spec] ffmpeg 없음(드럼 조각 모델 분리는 band-pass 폴백): {_ff}")

# 사전 다운로드한 Demucs 모델 번들 → 오프라인 실행.
# STEMREMAKE_REQUIRE_MODELS=1 이면 모델 누락 시 빌드를 실패시켜 조용한 미포함을 막는다.
models_dir = os.path.join(ROOT, "models")
_th_files = []
if os.path.isdir(models_dir):
    for _root, _dirs, _files in os.walk(models_dir):
        _th_files += [f for f in _files if f.endswith((".th", ".ckpt", ".pt", ".bin"))]
print(f"[spec] models_dir={models_dir} exists={os.path.isdir(models_dir)} weight_files={len(_th_files)}")
if _th_files:
    datas += [(models_dir, "models")]
elif os.environ.get("STEMREMAKE_REQUIRE_MODELS") == "1":
    raise SystemExit(f"[spec] 모델 가중치가 {models_dir} 에 없습니다. 빌드 중단(오프라인 번들 보장).")

a = Analysis(
    [os.path.join(ROOT, "app", "main.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["engine", "engine.pipeline", "engine.enhance", "app.server"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PyQt5", "PySide2", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)

# STEMREMAKE_ONEFILE=1 이면 단일 실행파일(.exe 하나)로 빌드 — 압축 풀기 불필요,
# 윈도우 경로 260자 제한/압축 풀기 실패 회피. 아니면 기존 onedir.
_onefile = os.environ.get("STEMREMAKE_ONEFILE") == "1"
if _onefile:
    exe = EXE(
        pyz, a.scripts, a.binaries, a.datas, [],
        name="StemRemake",
        console=True,
        disable_windowed_traceback=False,
        upx=False,
        icon=None,
    )
else:
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name="StemRemake",
        console=True,
        disable_windowed_traceback=False,
        icon=None,
    )
    coll = COLLECT(exe, a.binaries, a.datas, name="StemRemake")

# build: 20261012T154852Z

# build trigger: 20261012T185501Z

# retry: 185819Z

# trigger: 20261013T082714Z

# public build: 20261013T083452Z
