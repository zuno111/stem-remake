# -*- coding: utf-8 -*-
"""
전체 파이프라인 오케스트레이터.

업로드 음원 -> 스템 분리 -> 분석 -> 파트별 전사/렌더 -> 재믹스·마스터 -> 결과 wav
+ (옵션) DAW 내보내기 패키지(스템+MIDI)

렌더 우선순위(파트별):
1. instrument_map에 플러그인이 지정됨 -> VST/AU 악기로 렌더 (모드 3)  [실패 시 2로 폴백]
2. 정책이 retrack/retrack_poly -> numpy 합성 악기
3. 정책이 clean_keep -> 분리 스템을 정제해서 유지
4. 정책이 skip -> 사용 안 함

instrument_map 예: {"bass": {"path": "C:/.../HALion.vst3", "preset": None}, "drums": {...}}
daw_export=True 이면 out_dir/daw_package.zip 생성(큐베이스·로직 등에서 import).

실행:
    python -m engine.pipeline <input> <out_dir> [--backend auto|demucs|lite]
        [--preset instrumental|vocal] [--daw-export]
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
from . import vst_render as vst
from . import export_daw

PRESETS = {
    "instrumental": {"vocals": "skip", "drums": "retrack", "bass": "retrack", "other": "clean_keep"},
    "vocal": {"vocals": "clean_keep", "drums": "retrack", "bass": "retrack", "other": "clean_keep"},
    # 음색 유지: 재연주 없이 분리 스템을 정제만 해서 원곡 음색 최대 보존(보컬 제외=연주곡)
    "preserve": {"vocals": "skip", "drums": "clean_keep", "bass": "clean_keep", "other": "clean_keep"},
    # 음색 유지(보컬 포함)
    "preserve_vocal": {"vocals": "clean_keep", "drums": "clean_keep", "bass": "clean_keep", "other": "clean_keep"},
}


def _poly_available():
    try:
        import basic_pitch  # noqa: F401
        return True
    except Exception:
        return False


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


def _render_melodic(name, stem_path, notes, policy, sr, dur, instrument_map):
    """베이스/기타외 렌더. (audio, used_backend)"""
    im = (instrument_map or {}).get(name)
    if im and im.get("path"):
        try:
            audio = vst.render_notes_vst(notes, sr, dur, im["path"], im.get("preset"),
                                         is_drum=False, program=im.get("program"))
            return audio, "vst"
        except Exception as e:
            print(f"[{name}] VST 실패 → 합성 폴백: {e}")
    if policy in ("retrack", "retrack_poly"):
        kind = {"bass": "bass", "guitar": "guitar", "piano": "piano",
                "lead": "lead"}.get(name, "keys")
        return synth.render_notes(notes, sr, dur, kind=kind), "synth"
    # clean_keep
    y = _load(stem_path, sr)
    return np.asarray(mst.clean_instrument(y, sr), dtype=np.float32), "clean_keep"


def _maybe_quantize_notes(notes, report, quant):
    if not quant or not notes:
        return notes
    from . import quantize as qz
    return qz.quantize_notes(notes, report["bpm"], report["duration_sec"],
                             beat_offset=report.get("beat_times", [0.0])[0] if report.get("beat_times") else 0.0,
                             division=quant.get("division", 4), strength=quant.get("strength", 0.8))


def _maybe_quantize_drums(hits, report, quant):
    if not quant or not hits:
        return hits
    from . import quantize as qz
    return qz.quantize_drums(hits, report["bpm"], report["duration_sec"],
                             beat_offset=report.get("beat_times", [0.0])[0] if report.get("beat_times") else 0.0,
                             division=quant.get("division", 4), strength=quant.get("strength", 0.9))


def _maybe_keysnap(notes, report, keysnap_on):
    if not keysnap_on or not notes:
        return notes
    from . import keysnap as ks
    snapped, _ = ks.snap_notes(notes, report.get("key", "C"), report.get("mode", "major"),
                               strength=1.0, max_shift=1)
    return snapped


def run(input_path, out_dir, backend="auto", preset="instrumental",
        instrument_map=None, daw_export=False, six_stem=False, quantize=None,
        bpm_override=None, key_override=None, gains=None, keysnap=False,
        compare=False, progress_cb=None):
    os.makedirs(out_dir, exist_ok=True)
    stems_dir = os.path.join(out_dir, "stems")
    retrack_dir = os.path.join(out_dir, "retrack")
    os.makedirs(retrack_dir, exist_ok=True)

    policy = dict(PRESETS.get(preset, PRESETS["instrumental"]))
    poly = _poly_available()
    model_name = "htdemucs_6s" if six_stem else "htdemucs"
    # 기타외는 자동 믹스에선 정제 유지(합성보다 품질 좋음). 다성 전사는 내보내기/VST에만 사용.

    _progress(progress_cb, 5, f"스템 분리 (backend={backend}, preset={preset}, model={model_name})")
    stem_paths, sr, used_backend = sep.separate(input_path, stems_dir, backend=backend, model_name=model_name)

    _progress(progress_cb, 30, "곡 분석 (BPM/키)")
    report = ana.analyze(input_path, sr=sr)
    dur = report["duration_sec"]
    # 수동 지정(override): 분석값 대신 사용자 값 사용
    if bpm_override:
        try:
            report["bpm"] = float(bpm_override)
            report["bpm_source"] = "manual"
        except (TypeError, ValueError):
            pass
    if key_override:
        parts_k = str(key_override).split()
        report["key"] = parts_k[0]
        report["mode"] = parts_k[1] if len(parts_k) > 1 else report.get("mode", "major")
        report["key_source"] = "manual"
    report["preset"] = preset
    report["backend_used"] = used_backend
    report["policy"] = policy
    report["six_stem"] = bool(stem_paths.get("guitar") or stem_paths.get("piano"))
    report["quantize"] = quantize or None
    report["keysnap"] = bool(keysnap)
    report["vst_used"] = {}

    _progress(progress_cb, 40, f"분석: BPM {report['bpm']}, Key {report['key']} {report['mode']}")

    produced = {}
    parts = {}  # 내보내기용 전사 결과

    # 보컬 접기: 연주곡(보컬 없음) 모드에서 분리기가 리드 멜로디를 vocal 스템으로
    # 흘려보내므로, vocal을 버리지 않고 other에 합쳐 멜로디를 보존한다.
    # (에너지만으로는 '진짜 보컬'과 '새어든 리드'를 구분할 수 없어 사용자의 모드 선택을 신뢰)
    def _rms(p):
        y = _load(p, sr)
        return float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    if "vocals" in stem_paths:
        vr = _rms(stem_paths["vocals"])
        tot = sum(_rms(stem_paths[k]) for k in stem_paths) + 1e-9
        report["vocal_rms"] = round(vr, 5)
        report["vocal_ratio"] = round(vr / tot, 3)
        if policy.get("vocals") == "skip" and vr > 1e-4:
            yv = _load(stem_paths["vocals"], sr)
            yo = _load(stem_paths["other"], sr)
            n = max(len(yv), len(yo))
            a = np.zeros(n, dtype=np.float32); b = np.zeros(n, dtype=np.float32)
            a[:len(yo)] = yo; b[:len(yv)] = yv
            fp = os.path.join(stems_dir, "other_folded.wav")
            sf.write(fp, (a + b).astype(np.float32), sr)
            stem_paths["other"] = fp
            stem_paths.pop("vocals", None)
            report["vocal_folded"] = True

    # bass (항상 전사)
    _progress(progress_cb, 50, "베이스 전사")
    from . import cleanup as _cl
    bass_notes = tr.transcribe_mono(_load(stem_paths["bass"], sr), sr, fmin=41, fmax=400)
    bass_notes = _cl.clean_mono(bass_notes)
    bass_notes = _maybe_keysnap(bass_notes, report, keysnap)
    bass_notes = _maybe_quantize_notes(bass_notes, report, quantize)
    parts["bass"] = bass_notes
    produced["bass"], bk = _render_melodic("bass", stem_paths["bass"], bass_notes, policy["bass"], sr, dur, instrument_map)
    report["vst_used"]["bass"] = bk

    # drums: 전사(재연주/내보내기용). preserve 정책이면 음색 유지로 분리 스템 정제.
    _progress(progress_cb, 60, "드럼 전사")
    drum_hits = tr.transcribe_drums(_load(stem_paths["drums"], sr), sr)
    drum_hits = _cl.dedup_drums(drum_hits)
    drum_hits = _maybe_quantize_drums(drum_hits, report, quantize)
    parts["drums"] = drum_hits
    im_d = (instrument_map or {}).get("drums")
    if policy.get("drums") == "clean_keep" and not (im_d and im_d.get("path")):
        # 음색 유지: 원본 드럼 스템 정제해 사용
        produced["drums"] = np.asarray(mst.clean_instrument(_load(stem_paths["drums"], sr), sr, bright=False),
                                       dtype=np.float32)
        report["vst_used"]["drums"] = "clean_keep"
    elif im_d and im_d.get("path"):
        try:
            # 드럼은 GM 노트로 변환해 VST에 전달
            dn = [{"start": h["time"], "end": h["time"] + 0.1,
                   "midi": export_daw._GM_DRUM.get(h["drum"], 38), "vel": h["vel"]} for h in drum_hits]
            produced["drums"] = vst.render_notes_vst(dn, sr, dur, im_d["path"], im_d.get("preset"),
                                                     is_drum=True, program=im_d.get("program"))
            report["vst_used"]["drums"] = "vst"
        except Exception as e:
            print(f"[drums] VST 실패 → 합성 폴백: {e}")
            produced["drums"] = synth.render_drums(drum_hits, sr, dur)
            report["vst_used"]["drums"] = "synth"
    else:
        produced["drums"] = synth.render_drums(drum_hits, sr, dur)
        report["vst_used"]["drums"] = "synth"

    # other
    pol_other = policy.get("other")
    need_other_notes = daw_export or bool((instrument_map or {}).get("other")) or pol_other in ("retrack", "retrack_poly")
    other_notes = None
    if need_other_notes:
        yo = _load(stem_paths["other"], sr)
        if poly:
            _progress(progress_cb, 70, "기타외 다성 전사 (basic-pitch)")
            from . import transcribe_poly as trp
            other_notes = trp.transcribe_polyphonic(yo, sr, fmin=40.0, fmax=3000.0)
            other_notes = _cl.remove_octave_overtones(other_notes)
            report["other_transcribe"] = "poly"
        else:
            _progress(progress_cb, 70, "기타외 전사 (단선율)")
            other_notes = tr.transcribe_mono(yo, sr, fmin=80, fmax=2000)
            report["other_transcribe"] = "mono"
        other_notes = _cl.clean_mono(other_notes, fix_octave=False)
        other_notes = _maybe_keysnap(other_notes, report, keysnap)
        other_notes = _maybe_quantize_notes(other_notes, report, quantize)
        parts["other"] = other_notes
    if pol_other != "skip":
        _progress(progress_cb, 78, "기타외 렌더")
        produced["other"], ok = _render_melodic("other", stem_paths["other"], other_notes or [], pol_other, sr, dur, instrument_map)
        report["vst_used"]["other"] = ok

    # guitar / piano (6스템일 때만): 연주곡 모드면 재연주(합성), 아니면 정제 유지
    retrack_extras = (preset == "instrumental")
    for extra in ["guitar", "piano"]:
        if extra not in stem_paths:
            continue
        _progress(progress_cb, 80, f"{extra} 처리")
        ye = _load(stem_paths[extra], sr)
        en = None
        im_e = (instrument_map or {}).get(extra)
        if daw_export or bool(im_e) or retrack_extras:
            if poly:
                from . import transcribe_poly as trp
                # 파트 음역: 기타 70~2500Hz, 피아노 30~4000Hz
                fr = (70.0, 2500.0) if extra == "guitar" else (30.0, 4000.0)
                en = trp.transcribe_polyphonic(ye, sr, fmin=fr[0], fmax=fr[1])
                en = _cl.remove_octave_overtones(en)
            else:
                en = tr.transcribe_mono(ye, sr, fmin=60, fmax=2000)
            en = _cl.clean_mono(en, fix_octave=False)
            en = _maybe_keysnap(en, report, keysnap)
            en = _maybe_quantize_notes(en, report, quantize)
            parts[extra] = en
        if im_e and im_e.get("path"):
            try:
                produced[extra] = vst.render_notes_vst(en or [], sr, dur, im_e["path"],
                                                        im_e.get("preset"), program=im_e.get("program"))
                report["vst_used"][extra] = "vst"
            except Exception as e:
                print(f"[{extra}] VST 실패 → 정제 유지: {e}")
                produced[extra] = np.asarray(mst.clean_instrument(ye, sr), dtype=np.float32)
                report["vst_used"][extra] = "clean_keep"
        elif retrack_extras:
            produced[extra] = np.asarray(synth.render_notes(en or [], sr, dur, kind=extra),
                                         dtype=np.float32)
            report["vst_used"][extra] = "synth"
        else:
            produced[extra] = np.asarray(mst.clean_instrument(ye, sr), dtype=np.float32)
            report["vst_used"][extra] = "clean_keep"

    # vocals
    if policy.get("vocals") == "clean_keep":
        _progress(progress_cb, 84, "보컬 정제 유지")
        produced["vocals"] = np.asarray(mst.clean_vocal(_load(stem_paths["vocals"], sr), sr), dtype=np.float32)

    for name, audio in produced.items():
        sf.write(os.path.join(retrack_dir, f"{name}.wav"), np.asarray(audio, dtype=np.float32), sr)

    report["transcription"] = {"bass_notes": len(bass_notes), "drum_hits": len(drum_hits),
                               "other_notes": len(other_notes or [])}

    _progress(progress_cb, 90, "재믹스 + 마스터링")
    result = mst.mix_and_master(produced, sr, gains=gains)
    report["gains"] = gains or None
    result_path = os.path.join(out_dir, "result.wav")
    sf.write(result_path, np.asarray(result, dtype=np.float32), sr)

    # 비교 모드: 같은 분리/분석으로 '음색 유지본'(재연주 없이 분리 스템 정제)도 함께 렌더
    preserve_path = None
    if compare:
        _progress(progress_cb, 92, "음색 유지본 렌더(비교용)")
        keep_vocals = policy.get("vocals") == "clean_keep"
        preserve_mix = {}
        for nm, p in stem_paths.items():
            if nm == "vocals" and not keep_vocals:
                continue
            ys = _load(p, sr)
            if nm == "vocals":
                preserve_mix[nm] = np.asarray(mst.clean_vocal(ys, sr), dtype=np.float32)
            elif nm == "drums":
                preserve_mix[nm] = np.asarray(mst.clean_instrument(ys, sr, bright=False), dtype=np.float32)
            else:
                preserve_mix[nm] = np.asarray(mst.clean_instrument(ys, sr), dtype=np.float32)
        # 음색 유지본은 분리 잔여 잡음이 섞이므로 디노이즈 + 고역 부스트 완화
        preserve_result = mst.mix_and_master(preserve_mix, sr, gains=gains, denoise=0.6, gentle=True)
        preserve_path = os.path.join(out_dir, "result_preserve.wav")
        sf.write(preserve_path, np.asarray(preserve_result, dtype=np.float32), sr)
        report["compare"] = True

    daw_zip = None
    if daw_export:
        _progress(progress_cb, 95, "DAW 내보내기 패키지 생성 (스템+MIDI)")
        daw_zip = export_daw.build_package(out_dir, report, stem_paths, parts)

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _progress(progress_cb, 100, "완료")
    return {"result": result_path, "result_preserve": preserve_path, "report": report,
            "stems": stem_paths, "retrack_dir": retrack_dir, "daw_zip": daw_zip}


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print("usage: python -m engine.pipeline <input> <out_dir> [--backend ...] [--preset ...] [--daw-export]")
        sys.exit(1)
    backend = args[args.index("--backend") + 1] if "--backend" in args else "auto"
    preset = args[args.index("--preset") + 1] if "--preset" in args else "instrumental"
    daw = "--daw-export" in args
    six = "--6stem" in args
    quant = {"division": 4, "strength": 0.85} if "--quantize" in args else None
    run(args[0], args[1], backend=backend, preset=preset, daw_export=daw,
        six_stem=six, quantize=quant, keysnap=("--keysnap" in args))
