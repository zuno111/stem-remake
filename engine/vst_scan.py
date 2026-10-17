# -*- coding: utf-8 -*-
"""
설치된 VST3 악기 자동 탐지.

표준 VST3 설치 폴더를 훑어 *.vst3(파일 또는 번들 폴더)를 찾는다.
큐베이스/Steinberg 기본 악기(HALion Sonic SE, Groove Agent SE 등)뿐 아니라
널리 쓰는 무료 VST3 악기(Vital, Surge, Spitfire LABS, MT Power Drum 등)도
알아보고 악기/드럼/신스 추천 태그와 우선순위를 붙여 UI 드롭다운에 바로 쓰게 한다.

이 모듈은 사용자 PC(플러그인이 실제 설치된 곳)에서 동작한다.
"""
import os
import sys


# ── 알려진 악기 사전 ────────────────────────────────────────────────
# 각 항목: 소문자 부분일치 키 → (pref, is_synth)
#   pref  : 자동배정 우선순위(작을수록 먼저). 같은 역할이면 품질/범용성 높은 걸 먼저.
#   synth : 전자음·리드·패드 등 '기타외' 파트에 우선 배정할 신스 계열인지.
# Steinberg/큐베이스 기본 악기
_STEINBERG = {
    "halion sonic": (15, False),   # 범용 워크스테이션 — 베이스/멜로디에 무난
    "halion": (16, False),
    "groove agent": (10, False),   # 드럼(아래 _DRUM_HINT로 드럼 태그)
    "retrologue": (25, True),      # 버추얼 아날로그 신스
    "padshop": (30, True),         # 그래뉼러 신스
    "prologue": (32, True),
    "spector": (34, True),
    "mystic": (34, True),
    "embracer": (40, False),
    "model c": (40, False),        # 오르간
}
# 무료/서드파티 인기 악기 (사용자가 깔면 자동 인식·배정)
_FREE_INSTR = {
    # 신스
    "vital": (10, True),
    "surge": (11, True),           # Surge XT
    "diva": (12, True),            # u-he Diva
    "hive": (13, True),            # u-he Hive
    "zebralette": (14, True),
    "podolski": (14, True),
    "tyrell": (14, True),
    "helm": (15, True),
    "dexed": (16, True),           # FM 신스(DX7)
    "noisemaker": (17, True),      # TAL-NoiseMaker
    "tal-": (17, True),
    "obxd": (18, True),
    "ob-xd": (18, True),
    "odin": (18, True),
    "synth1": (18, True),
    "synth 1": (18, True),
    "crystal": (19, True),
    "charlatan": (19, True),
    "u-he": (12, True),
    # 샘플러·어쿠스틱(범용 멜로디)
    "labs": (20, False),           # Spitfire LABS (피아노·현악·질감)
    "spitfire": (20, False),
    "sine player": (21, False),    # Orchestral Tools SINE
    "sforzando": (22, False),      # SFZ 샘플러
    "decent sampler": (22, False),
    "decentsampler": (22, False),
    "tx16wx": (23, False),
    # 드럼(아래 _DRUM_HINT로도 드럼 태그)
    "mt power": (15, False),
    "mt-power": (15, False),
    "powerdrumkit": (15, False),
    "power drum": (15, False),
    "drumgizmo": (16, False),
    "sitala": (17, False),
    "mdrummer": (17, False),
}
_KNOWN = {}
_KNOWN.update(_STEINBERG)
_KNOWN.update(_FREE_INSTR)

_DRUM_HINT = ["groove agent", "drum", "beat", "mt power", "mt-power",
              "powerdrumkit", "power drum", "drumgizmo", "sitala", "mdrummer",
              "kit", "percussion"]
# 명백한 이펙트류(악기 아님) — 추천에서 제외
_FX_HINT = ["reverb", "delay", "eq", "compressor", "limiter", "chorus",
            "flanger", "phaser", "gate", "maximizer", "distortion", "amp",
            "saturator", "exciter", "deesser", "de-esser", "widener"]

_PREF_DEFAULT = 100


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
    pref = _PREF_DEFAULT
    is_synth = False
    is_known = False
    for key, (p, syn) in _KNOWN.items():
        if key in low:
            is_known = True
            if p < pref:
                pref = p
            if syn:
                is_synth = True
    is_drum = any(h in low for h in _DRUM_HINT)
    # known 악기면 이펙트 키워드가 이름에 있어도 악기로 본다(예: 'Groove Agent'는 아님)
    kind = "instrument" if is_known else ("effect" if is_fx else "unknown")
    return kind, is_drum, is_known, is_synth, pref


def scan_vst3(extra_dirs=None):
    """설치된 VST3/AU 목록.

    반환: [{name, path, kind, drum, known, synth, pref}]
    정렬: known 우선 → pref 오름차순 → kind → 이름. 리스트 앞쪽이 '더 좋은' 추천.
    """
    seen = set()
    out = []
    dirs = _scan_dirs()
    if extra_dirs:
        dirs += [d for d in extra_dirs if d and os.path.isdir(d)]
    for d in dirs:
        try:
            for entry in sorted(os.listdir(d)):
                le = entry.lower()
                if not (le.endswith(".vst3") or le.endswith(".component")):
                    continue
                path = os.path.join(d, entry)
                if path in seen:
                    continue
                seen.add(path)
                name = os.path.splitext(entry)[0]
                kind, drum, known, synth, pref = _classify(name)
                out.append({"name": name, "path": path, "kind": kind,
                            "drum": bool(drum), "known": bool(known),
                            "synth": bool(synth), "pref": int(pref)})
        except Exception:
            continue
    kind_order = {"instrument": 0, "unknown": 1, "effect": 2}
    out.sort(key=lambda x: (0 if x["known"] else 1,
                            x["pref"],
                            kind_order.get(x["kind"], 3),
                            x["name"].lower()))
    return out
