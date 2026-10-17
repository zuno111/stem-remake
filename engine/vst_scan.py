# -*- coding: utf-8 -*-
"""
설치된 VST3 악기 자동 탐지.

표준 VST3 설치 폴더를 훑어 *.vst3(파일 또는 번들 폴더)를 찾는다.
큐베이스/Steinberg 기본 악기(HALion Sonic SE, Groove Agent SE 등)를 알아보고
악기/드럼 추천 태그를 붙여 UI 드롭다운에 바로 쓰게 한다.

이 모듈은 사용자 PC(플러그인이 실제 설치된 곳)에서 동작한다.
"""
import os
import sys


# 알려진 Steinberg/큐베이스 악기 — 소문자 부분일치
_KNOWN_INSTR = [
    "halion sonic", "halion", "groove agent", "retrologue", "padshop",
    "prologue", "spector", "mystic", "embracer", "model c",
]
_DRUM_HINT = ["groove agent", "drum", "beat"]
# 명백한 이펙트류(악기 아님) — 추천에서 제외
_FX_HINT = ["reverb", "delay", "eq", "compressor", "limiter", "chorus",
            "flanger", "phaser", "gate", "maximizer", "distortion", "amp"]


def _scan_dirs():
    dirs = []
    if sys.platform.startswith("win"):
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pfx = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        local = os.environ.get("LOCALAPPDATA", "")
        dirs += [
            os.path.join(pf, "Common Files", "VST3"),
            os.path.join(pfx, "Common Files", "VST3"),
        ]
        if local:
            dirs.append(os.path.join(local, "Programs", "Common", "VST3"))
    elif sys.platform == "darwin":
        home = os.path.expanduser("~")
        dirs += [
            "/Library/Audio/Plug-Ins/VST3",
            os.path.join(home, "Library/Audio/Plug-Ins/VST3"),
            "/Library/Audio/Plug-Ins/Components",   # AU
            os.path.join(home, "Library/Audio/Plug-Ins/Components"),
        ]
    else:
        # 리눅스(개발/검증용): 환경변수로 지정 가능
        p = os.environ.get("STEMREMAKE_VST3_DIR")
        if p:
            dirs += p.split(os.pathsep)
    return [d for d in dirs if d and os.path.isdir(d)]


def _classify(name):
    low = name.lower()
    is_fx = any(h in low for h in _FX_HINT)
    is_known = any(k in low for k in _KNOWN_INSTR)
    is_drum = any(h in low for h in _DRUM_HINT)
    # known > (not fx) > fx
    kind = "instrument" if is_known else ("effect" if is_fx else "unknown")
    return kind, is_drum, is_known


def scan_vst3(extra_dirs=None):
    """설치된 VST3/AU 목록. [{name, path, kind, drum, known}] (known 우선 정렬)."""
    seen = set()
    out = []
    dirs = _scan_dirs()
    if extra_dirs:
        dirs += [d for d in extra_dirs if d and os.path.isdir(d)]
    for d in dirs:
        try:
            for entry in sorted(os.listdir(d)):
                if not (entry.lower().endswith(".vst3") or entry.lower().endswith(".component")):
                    continue
                path = os.path.join(d, entry)
                if path in seen:
                    continue
                seen.add(path)
                name = os.path.splitext(entry)[0]
                kind, drum, known = _classify(name)
                out.append({"name": name, "path": path, "kind": kind,
                            "drum": bool(drum), "known": bool(known)})
        except Exception:
            continue
    # known 악기 → 일반 악기 → unknown → effect 순, 이름 알파벳
    order = {"instrument": 0, "unknown": 1, "effect": 2}
    out.sort(key=lambda x: (0 if x["known"] else 1, order.get(x["kind"], 3), x["name"].lower()))
    return out
