# -*- coding: utf-8 -*-
"""
전체 파이프라인 오케스트레이터.

업로드 음원 -> 스템 분리 -> 분석 -> 파트별 처리(재연주/정제유지/스킵) -> 재믹스·마스터 -> 결과 wav

프리셋:
- instrumental(연주곡): 보컬 스킵, 베이스·드럼 재연주, other(폴리포닉) 정제 유지
- vocal(보컬곡):       보컬 정제 유지, 베이스·드럼 재연주, other 정제 유지

처리 정책(policy) 값:
- "retrack"    : 전사 후 악기로 재연주
- "clean_keep" : 분리된 원 스템을 정제해서 그대로 사용(폴리포닉/보컬)
- "skip"       : 사용 안 함

실행:
    python -m engine.pipeline <input> <out_dir> [--backend auto|demucs|lite] [--preset instrumental|vocal]
"""
import os
import sys
import json
import numpy as np
import soundfile as sf
import librosa

from . import separate as sep
from . import analyze as ana
from . import transcribe as tr
from . import synth
from . import master as mst

PRESETS = {
    "instrumental": {"vocals": "skip", "drums": "retrack", "bass": "retrack", "other": "clean_keep"},
    "vocal": {"vocals": "clean_keep", "drums": "retrack", "bass": "retrack", "other": "clean_keep"},
}

# 제품 빌드에서 basic-pitch(다성 전사)가 있으면 other도 retrack으로 승격
def _upgrade_policy_if_polyphonic(policy):
    try:
        import basic_pitch  # noqa: F401
        policy = dict(policy)
        if policy.get("other") == "clean_keep":
            policy["other"] = "retrack_poly"
        return policy
    except Exception:
        return policy


def _load(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def _progress(cb, pct, msg):
    print(f"[{pct:3d}%] {msg}")
    if cb:
        try:
            cb(pct, msg)
        except Exception:
            pass


def run(input_path, out_dir, backend="auto", preset="instrumental", progress_cb=None):
    os.makedirs(out_dir, exist_ok=True)
    stems_dir = os.path.join(out_dir, "stems")
    retrack_dir = os.path.join(out_dir, "retrack")
    os.makedirs(retrack_dir, exist_ok=True)

    policy = _upgrade_policy_if_polyphonic(PRESETS.get(preset, PRESETS["instrumental"]))

    _progress(progress_cb, 5, f"스템 분리 (backend={backend}, preset={preset})")
    stem_paths, sr = sep.separate(input_path, stems_dir, backend=backend)

    _progress(progress_cb, 35, "곡 분석 (BPM/키)")
    report = ana.analyze(input_path, sr=sr)
    dur = report["duration_sec"]
    report["preset"] = preset
    report["policy"] = policy

    _progress(progress_cb, 45, f"분석 완료: BPM {report['bpm']}, Key {report['key']} {report['mode']}")

    produced = {}
    tr_summary = {}

    # bass
    if policy.get("bass") == "retrack":
        _progress(progress_cb, 55, "베이스 전사·재연주")
        y = _load(stem_paths["bass"], sr)
        notes = tr.transcribe_mono(y, sr, fmin=41, fmax=400)
        produced["bass"] = synth.render_notes(notes, sr, dur, kind="bass")
        tr_summary["bass_notes"] = len(notes)

    # drums
    if policy.get("drums") == "retrack":
        _progress(progress_cb, 65, "드럼 전사·재구성")
        y = _load(stem_paths["drums"], sr)
        hits = tr.transcribe_drums(y, sr)
        produced["drums"] = synth.render_drums(hits, sr, dur)
        tr_summary["drum_hits"] = len(hits)

    # other
    pol_other = policy.get("other")
    if pol_other == "retrack":
        _progress(progress_cb, 75, "기타외 전사·재연주(단선율)")
        y = _load(stem_paths["other"], sr)
        notes = tr.transcribe_mono(y, sr, fmin=80, fmax=2000)
        produced["other"] = synth.render_notes(notes, sr, dur, kind="keys")
        tr_summary["other_notes"] = len(notes)
    elif pol_other == "retrack_poly":
        _progress(progress_cb, 75, "기타외 다성 전사·재연주(basic-pitch)")
        from . import transcribe_poly as trp  # 제품 빌드에만 존재
        y = _load(stem_paths["other"], sr)
        notes = trp.transcribe_polyphonic(y, sr)
        produced["other"] = synth.render_notes(notes, sr, dur, kind="keys")
        tr_summary["other_notes"] = len(notes)
    elif pol_other == "clean_keep":
        _progress(progress_cb, 75, "기타외 정제 유지(폴리포닉)")
        y = _load(stem_paths["other"], sr)
        produced["other"] = np.asarray(mst.clean_instrument(y, sr), dtype=np.float32)

    # vocals
    pol_voc = policy.get("vocals")
    if pol_voc == "clean_keep":
        _progress(progress_cb, 85, "보컬 정제 유지")
        y = _load(stem_paths["vocals"], sr)
        produced["vocals"] = np.asarray(mst.clean_vocal(y, sr), dtype=np.float32)

    # 재연주 스템 저장
    for name, audio in produced.items():
        sf.write(os.path.join(retrack_dir, f"{name}.wav"), np.asarray(audio, dtype=np.float32), sr)

    report["transcription"] = tr_summary

    _progress(progress_cb, 92, "재믹스 + 마스터링")
    result = mst.mix_and_master(produced, sr)
    result_path = os.path.join(out_dir, "result.wav")
    sf.write(result_path, np.asarray(result, dtype=np.float32), sr)

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _progress(progress_cb, 100, f"완료 → {os.path.basename(result_path)}")
    return {"result": result_path, "report": report, "stems": stem_paths, "retrack_dir": retrack_dir}


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print("usage: python -m engine.pipeline <input> <out_dir> [--backend auto|demucs|lite] [--preset instrumental|vocal]")
        sys.exit(1)
    backend = args[args.index("--backend") + 1] if "--backend" in args else "auto"
    preset = args[args.index("--preset") + 1] if "--preset" in args else "instrumental"
    run(args[0], args[1], backend=backend, preset=preset)
