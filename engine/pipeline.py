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
from . import timbre_transfer as tt

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


# 파트별 기본 음색 매핑(합성 모드에서 악기 미지정 시)
_DEFAULT_KIND = {"bass": "bass", "guitar": "guitar", "piano": "piano", "lead": "lead"}


def _mode_needs_notes(mode):
    """해당 렌더 방식이 전사 노트를 필요로 하는가(원본유지는 불필요)."""
    return bool(mode) and mode != "keep"


def _render_part(name, stem_path, notes, mode, sr, dur, instrument_map):
    """
    파트 렌더. mode: 'keep'(원본 음색 유지) | 'transfer'(음색 전사) | 'synth[:악기]'(합성).
    VST(instrument_map)가 지정되면 최우선. (audio, used)
    """
    im = (instrument_map or {}).get(name)
    if im and im.get("path"):
        try:
            audio = vst.render_notes_vst(notes or [], sr, dur, im["path"], im.get("preset"),
                                         is_drum=False, program=im.get("program"))
            return np.asarray(audio, dtype=np.float32), "vst"
        except Exception as e:
            print(f"[{name}] VST 실패 → 폴백: {e}")
    y = _load(stem_path, sr)
    if mode == "transfer":
        return np.asarray(tt.render_notes(notes or [], y, sr, dur), dtype=np.float32), "transfer"
    if mode and mode.startswith("synth"):
        inst = mode.split(":", 1)[1] if ":" in mode else _DEFAULT_KIND.get(name, "keys")
        return np.asarray(synth.render_notes(notes or [], sr, dur, kind=inst), dtype=np.float32), "synth:" + inst
    # 'keep' 또는 기본: 원본 음색 유지
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
        compare=False, render_modes=None, drum_split=True, progress_cb=None):
    os.makedirs(out_dir, exist_ok=True)
    stems_dir = os.path.join(out_dir, "stems")
    retrack_dir = os.path.join(out_dir, "retrack")
    pieces_dir = os.path.join(out_dir, "pieces")
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

    # 파트별 렌더 방식(원본유지/음색전사/합성). 기본: 전자음·특이음색 보존을 위해
    # 멜로디·화성 파트는 원본 유지, 베이스는 음색 전사, 드럼은 합성.
    _dmodes = {"drums": "keep", "bass": "transfer", "other": "keep", "guitar": "keep", "piano": "keep"}
    rmodes = dict(_dmodes)
    if render_modes:
        rmodes.update({k: v for k, v in render_modes.items() if v})
    report["render_modes"] = rmodes
    report["six_stem"] = bool(stem_paths.get("guitar") or stem_paths.get("piano"))
    report["quantize"] = quantize or None
    report["keysnap"] = bool(keysnap)
    report["vst_used"] = {}
    # 어떤 파트에 VST 악기가 지정돼 있었는지(UI에서 'VST 로드 실패→합성 폴백' 판별용)
    report["vst_requested"] = {k: bool((v or {}).get("path"))
                               for k, v in (instrument_map or {}).items()}

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
    produced["bass"], bk = _render_part("bass", stem_paths["bass"], bass_notes, rmodes.get("bass"), sr, dur, instrument_map)
    report["vst_used"]["bass"] = bk

    # drums: 파트별 방식(기본=원본 유지). 합성 모드면 전사→합성 드럼 키트.
    _progress(progress_cb, 60, "드럼 처리")
    im_d = (instrument_map or {}).get("drums")
    drum_mode = rmodes.get("drums", "keep")
    drum_hits = []
    if daw_export or (im_d and im_d.get("path")) or drum_mode == "synth":
        drum_hits = tr.transcribe_drums(_load(stem_paths["drums"], sr), sr)
        drum_hits = _cl.dedup_drums(drum_hits)
        drum_hits = _maybe_quantize_drums(drum_hits, report, quantize)
    parts["drums"] = drum_hits
    if im_d and im_d.get("path"):
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
    elif drum_mode == "synth":
        produced["drums"] = synth.render_drums(drum_hits, sr, dur)
        report["vst_used"]["drums"] = "synth"
    else:  # keep: 원본 드럼 스템 정제해 유지(합성 키트로 안 바꿈)
        produced["drums"] = np.asarray(mst.clean_instrument(_load(stem_paths["drums"], sr), sr, bright=False),
                                       dtype=np.float32)
        report["vst_used"]["drums"] = "clean_keep"

    # 드럼 분해: 드럼 스템을 킥·스네어·하이햇·탐·심벌 개별 트랙으로 추출(믹스와 무관, 추출물)
    extra_stems = {}
    if drum_split and "drums" in stem_paths:
        try:
            _progress(progress_cb, 64, "드럼 분해 (킥·스네어·하이햇·탐·심벌)")
            from . import drumsplit as _ds
            dy = _load(stem_paths["drums"], sr)
            split_hits = drum_hits if drum_hits else _cl.dedup_drums(tr.transcribe_drums(dy, sr))
            pieces = _ds.split_drum_stem(dy, sr, split_hits)
            if pieces:
                os.makedirs(pieces_dir, exist_ok=True)
                for pname, audio in pieces.items():
                    pp = os.path.join(pieces_dir, f"drum_{pname}.wav")
                    sf.write(pp, np.asarray(audio, dtype=np.float32), sr)
                    extra_stems[f"drum_{pname}"] = pp
                report["drum_pieces"] = list(pieces.keys())
        except Exception as e:
            print(f"[drumsplit] 실패(건너뜀): {e}")

    # other
    pol_other = policy.get("other")
    other_mode = rmodes.get("other")
    need_other_notes = (daw_export or bool((instrument_map or {}).get("other"))
                        or pol_other in ("retrack", "retrack_poly") or _mode_needs_notes(other_mode))
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
        produced["other"], ok = _render_part("other", stem_paths["other"], other_notes or [], other_mode, sr, dur, instrument_map)
        report["vst_used"]["other"] = ok

    # guitar / piano (6스템일 때만): 파트별 렌더 방식(기본=원본 유지)
    for extra in ["guitar", "piano"]:
        if extra not in stem_paths:
            continue
        _progress(progress_cb, 80, f"{extra} 처리")
        ye = _load(stem_paths[extra], sr)
        en = None
        im_e = (instrument_map or {}).get(extra)
        e_mode = rmodes.get(extra, "keep")
        if daw_export or bool(im_e) or _mode_needs_notes(e_mode):
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
        produced[extra], ek = _render_part(extra, stem_paths[extra], en or [], e_mode, sr, dur, instrument_map)
        report["vst_used"][extra] = ek

    # vocals
    if policy.get("vocals") == "clean_keep":
        _progress(progress_cb, 84, "보컬 정제 유지")
        produced["vocals"] = np.asarray(mst.clean_vocal(_load(stem_paths["vocals"], sr), sr), dtype=np.float32)

    for name, audio in produced.items():
        sf.write(os.path.join(retrack_dir, f"{name}.wav"), np.asarray(audio, dtype=np.float32), sr)

    report["transcription"] = {"bass_notes": len(bass_notes), "drum_hits": len(drum_hits),
                               "other_notes": len(other_notes or [])}

    _progress(progress_cb, 90, "재믹스 + 마스터링")
    # 게인 미지정 시: 재연주 파트를 원곡 각 스템 레벨에 맞춰 원곡 밸런스 유지
    mix_gains = gains
    if mix_gains is None:
        try:
            orig_lv = {nm: _load(stem_paths[nm], sr) for nm in produced if nm in stem_paths}
            # 보컬은 살짝 앞, 베이스는 살짝 받쳐주는 미세 틸트
            mix_gains = mst.match_gains(orig_lv, produced,
                                        tilt={"vocals": 1.0, "bass": 0.5})
            report["auto_balance"] = {k: round(v, 1) for k, v in mix_gains.items()}
        except Exception as e:
            print(f"[mix] 자동 밸런스 실패 → 기본 게인: {e}")
            mix_gains = None
    result = mst.mix_and_master(produced, sr, gains=mix_gains)
    report["gains"] = mix_gains or None
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
                v = np.asarray(mst.clean_vocal(ys, sr), dtype=np.float32)
                preserve_mix[nm] = mst.denoise_adaptive(v, sr, prop_decrease=0.3)
            elif nm == "drums":
                # 드럼은 트랜지언트 보존 위해 디노이즈 제외(정제만)
                preserve_mix[nm] = np.asarray(mst.clean_instrument(ys, sr, bright=False), dtype=np.float32)
            else:
                ci = np.asarray(mst.clean_instrument(ys, sr), dtype=np.float32)
                preserve_mix[nm] = mst.denoise_adaptive(ci, sr, prop_decrease=0.35)
        # 스템별 적응형 디노이즈를 이미 적용 → 믹스 단계 디노이즈는 생략, 고역 부스트만 완화
        preserve_result = mst.mix_and_master(preserve_mix, sr, gains=gains, denoise=0.0, gentle=True)
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
            "stems": stem_paths, "extra_stems": extra_stems,
            "retrack_dir": retrack_dir, "daw_zip": daw_zip}


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
