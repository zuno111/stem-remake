# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 스펙 — 스템 리메이크 단독 실행 빌드(onedir).
import os
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

# 무거운 패키지는 데이터·바이너리·서브모듈을 통째로 수집
for pkg in [
    "librosa", "soundfile", "pedalboard", "pretty_midi", "mido",
    "demucs", "torch", "torchaudio", "julius", "openunmix",
    "dora", "lameenc", "einops", "scipy", "audioread",
    "soxr", "numba", "llvmlite", "sklearn", "flask", "waitress",
    "basic_pitch", "onnxruntime", "mir_eval", "resampy",
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
    hiddenimports=hiddenimports + ["engine", "engine.pipeline", "app.server"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PyQt5", "PySide2", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="StemRemake",
    console=True,
    disable_windowed_traceback=False,
    icon=None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="StemRemake")
