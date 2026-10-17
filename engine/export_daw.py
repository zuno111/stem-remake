# -*- coding: utf-8 -*-
"""
DAW 내보내기(모드 2). 큐베이스·로직 등 어떤 DAW에서도 import 가능한 패키지 생성.
- 파트별 MIDI (베이스·드럼·기타외)
- 통합 MIDI (한 파일에 전체)
- 분리 스템 wav (참고/대체용)
- info.txt (BPM·키), import_guide.txt
DAW 무관. 사용자는 이 MIDI를 DAW에서 열고 자기 악기(큐베이스 HALion, 로직 악기 등)를 꽂아 완성.
"""
import os
import shutil
import pretty_midi

# GM 드럼 노트
_GM_DRUM = {"kick": 36, "snare": 38, "hat": 42, "open_hat": 46,
            "tom": 45, "crash": 49, "ride": 51}


def _add_notes(inst, notes):
    for n in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=int(n["vel"]), pitch=int(n["midi"]),
            start=float(n["start"]), end=float(max(n["end"], n["start"] + 0.05)),
        ))


def _drum_instrument(hits):
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for h in hits:
        pitch = _GM_DRUM.get(h["drum"], 38)
        inst.notes.append(pretty_midi.Note(
            velocity=int(h["vel"]), pitch=pitch,
            start=float(h["time"]), end=float(h["time"] + 0.1),
        ))
    return inst


def _shift(items, key_pairs, offset):
    """offset(초)만큼 당겨 첫 박을 0에 맞춤. key_pairs: 시간 필드명들."""
    if not offset:
        return items
    out = []
    for it in items:
        m = dict(it)
        for k in key_pairs:
            if k in m:
                m[k] = max(0.0, m[k] - offset)
        out.append(m)
    return out


def _new_pm(bpm):
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm) if bpm else 120.0)
    # 4/4 박자표 추가 → DAW에서 마디 정렬
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0))
    return pm


def build_midis(out_dir, bpm, parts, beat_offset=0.0):
    """
    parts: {"bass": notes, "other": notes, "guitar", "piano", "drums": hits}
    beat_offset: 첫 박 시각(초). 이만큼 당겨 마디 1 박 1에 정렬.
    파트별 .mid + 통합 all.mid 생성. 정확한 BPM + 4/4 박자표 포함.
    """
    midi_dir = os.path.join(out_dir, "midi")
    os.makedirs(midi_dir, exist_ok=True)
    made = []

    programs = {"bass": 33, "other": 4, "guitar": 27, "piano": 0}  # finger bass, e.piano, clean guitar, a.piano
    combined = _new_pm(bpm)

    # 멜로디/화성 파트
    for name in ["bass", "other", "guitar", "piano"]:
        notes = parts.get(name)
        if not notes:
            continue
        notes = _shift(notes, ("start", "end"), beat_offset)
        inst = pretty_midi.Instrument(program=programs[name], name=name)
        _add_notes(inst, notes)
        pm = _new_pm(bpm)
        pm.instruments.append(inst)
        p = os.path.join(midi_dir, f"{name}.mid")
        pm.write(p)
        made.append(p)
        combined.instruments.append(inst)

    # 드럼 파트
    hits = parts.get("drums")
    if hits:
        hits = _shift(hits, ("time",), beat_offset)
        dinst = _drum_instrument(hits)
        pm = _new_pm(bpm)
        pm.instruments.append(dinst)
        p = os.path.join(midi_dir, "drums.mid")
        pm.write(p)
        made.append(p)
        combined.instruments.append(dinst)

    if combined.instruments:
        p = os.path.join(midi_dir, "all.mid")
        combined.write(p)
        made.append(p)
    return made


def build_package(out_dir, report, stem_paths, parts):
    """
    DAW 내보내기 패키지 폴더 구성 후 zip 생성. zip 경로 반환.
    """
    pkg = os.path.join(out_dir, "daw_package")
    if os.path.exists(pkg):
        shutil.rmtree(pkg)
    os.makedirs(pkg, exist_ok=True)

    # MIDI (첫 박 정렬)
    bt = report.get("beat_times") or []
    beat_offset = float(bt[0]) if bt else 0.0
    build_midis(pkg, report.get("bpm"), parts, beat_offset=beat_offset)

    # 스템 복사
    stems_out = os.path.join(pkg, "stems")
    os.makedirs(stems_out, exist_ok=True)
    for name, p in stem_paths.items():
        if p and os.path.exists(p):
            shutil.copy(p, os.path.join(stems_out, f"{name}.wav"))

    # 정보
    with open(os.path.join(pkg, "info.txt"), "w", encoding="utf-8") as f:
        f.write(f"BPM: {report.get('bpm')}\n")
        f.write(f"Key: {report.get('key')} {report.get('mode')}\n")
        f.write(f"Duration: {report.get('duration_sec')}s\n")

    with open(os.path.join(pkg, "import_guide.txt"), "w", encoding="utf-8") as f:
        f.write(
            "DAW 가져오기 안내\n"
            "==================\n\n"
            "1. DAW에서 새 프로젝트를 만들고 템포를 info.txt의 BPM으로 설정한다.\n"
            "2. midi/ 폴더의 파트별 MIDI(bass.mid, other.mid, drums.mid) 또는\n"
            "   all.mid를 트랙으로 import 한다.\n"
            "3. 각 MIDI 트랙에 원하는 악기를 꽂는다.\n"
            "   - 큐베이스: HALion Sonic / Groove Agent SE 등\n"
            "   - 로직: 로직 내장 악기 또는 보유한 플러그인\n"
            "4. drums.mid는 GM 드럼 매핑(킥36·스네어38·하이햇42) 기준이다.\n"
            "   드럼 악기의 매핑에 맞게 조정한다.\n"
            "5. stems/ 는 원곡 분리 스템이다. 참고용 또는 보컬 등 그대로 쓸 파트로 활용한다.\n"
        )

    zip_base = os.path.join(out_dir, "daw_package")
    zip_path = shutil.make_archive(zip_base, "zip", pkg)
    return zip_path
