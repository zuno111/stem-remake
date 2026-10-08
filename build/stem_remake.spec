# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 스펙 — 스템 리메이크 단독 실행 빌드(onedir).
import os
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

# 무거운 패키지는 데이터·바이너리·서브모듈을 통째로 수집
for pkg in [
    "librosa", "soundfile", "pedalboard", "pretty_midi",
    "demucs", "torch", "torchaudio", "julius", "openunmix",
    "dora", "lameenc", "einops", "scipy", "audioread",
    "soxr", "numba", "llvmlite", "sklearn", "flask", "waitress",
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

# 사전 다운로드한 Demucs 모델(있으면 번들 → 오프라인 실행)
models_dir = os.path.join(ROOT, "models")
if os.path.isdir(models_dir):
    datas += [(models_dir, "models")]

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
