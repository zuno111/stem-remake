# -*- coding: utf-8 -*-
"""
스템 리메이크 로컬 서버(Flask).
업로드 → 백그라운드로 파이프라인 실행 → 진행률 폴링 → 결과/스템 재생·다운로드.
"""
import os
import sys
import uuid
import threading
import tempfile
import traceback

from flask import Flask, request, jsonify, send_file, send_from_directory, abort

# 엔진 임포트(패키징/소스 양쪽 지원)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
from engine import pipeline  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
WORK_ROOT = os.path.join(tempfile.gettempdir(), "stem_remake_jobs")
os.makedirs(WORK_ROOT, exist_ok=True)

app = Flask(__name__, static_folder=None)
JOBS = {}  # job_id -> {status, pct, msg, result, stems, error, input}


def _run_job(job_id, input_path, job_dir, preset, backend, daw_export, instrument_map,
             six_stem=False, quantize=None, bpm_override=None, key_override=None, gains=None,
             keysnap=False, compare=False, enh=None, render_modes=None, drum_split=True):
    job = JOBS[job_id]

    def cb(pct, msg):
        job["pct"] = pct
        job["msg"] = msg

    try:
        job["status"] = "running"
        # 원본 보정 모드: 분리·재연주 없이 원본을 직접 손봐 음질 업그레이드(스테레오 유지)
        if preset == "enhance":
            from engine import enhance as _enh
            cb(10, "원본 보정 (디노이즈·EQ·마스터)")
            out_path = os.path.join(job_dir, "result.wav")
            e = enh or {}
            rep = _enh.enhance(
                input_path, out_path,
                denoise=float(e.get("denoise", 0.55)),
                low_gain=float(e.get("low_gain", 1.0)),
                mud_cut=float(e.get("mud_cut", -1.5)),
                deharsh=float(e.get("deharsh", -1.5)),
                air=float(e.get("air", 2.0)),
                comp=float(e.get("comp", 1.8)),
                target_lufs=float(e.get("target_lufs", -14.0)),
            )
            cb(100, "완료")
            job["result"] = out_path
            job["result_preserve"] = None
            job["stems"] = {}
            job["retrack_dir"] = job_dir
            job["daw_zip"] = None
            job["job_dir"] = job_dir
            job["report"] = {"mode": "enhance", "enhance": rep}
            job["status"] = "done"; job["pct"] = 100; job["msg"] = "완료"
            return
        out = pipeline.run(input_path, job_dir, backend=backend, preset=preset,
                           daw_export=daw_export, instrument_map=instrument_map,
                           six_stem=six_stem, quantize=quantize,
                           bpm_override=bpm_override, key_override=key_override, gains=gains,
                           keysnap=keysnap, compare=compare, render_modes=render_modes,
                           drum_split=drum_split, progress_cb=cb)
        job["result"] = out["result"]
        job["result_preserve"] = out.get("result_preserve")
        job["stems"] = out["stems"]
        job["extra_stems"] = out.get("extra_stems") or {}
        job["retrack_dir"] = out["retrack_dir"]
        job["daw_zip"] = out.get("daw_zip")
        job["job_dir"] = job_dir
        job["report"] = out["report"]
        job["status"] = "done"
        job["pct"] = 100
        job["msg"] = "완료"
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"{e}"
        job["trace"] = traceback.format_exc()
        print(job["trace"])


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/static/<path:p>")
def static_files(p):
    return send_from_directory(STATIC_DIR, p)


@app.post("/api/process")
def process():
    if "file" not in request.files:
        return jsonify({"error": "파일이 없습니다"}), 400
    f = request.files["file"]
    preset = request.form.get("preset", "instrumental")
    backend = request.form.get("backend", "auto")
    daw_export = request.form.get("daw_export", "false").lower() == "true"
    six_stem = request.form.get("six_stem", "false").lower() == "true"
    import json as _json
    imap_raw = request.form.get("instrument_map", "")
    try:
        instrument_map = _json.loads(imap_raw) if imap_raw.strip() else None
    except Exception:
        instrument_map = None
    quant_raw = request.form.get("quantize", "")
    try:
        quantize = _json.loads(quant_raw) if quant_raw.strip() else None
    except Exception:
        quantize = None
    keysnap = request.form.get("keysnap", "false").lower() == "true"
    compare = request.form.get("compare", "false").lower() == "true"
    drum_split = request.form.get("drum_split", "true").lower() == "true"
    enh_raw = request.form.get("enh", "")
    try:
        enh = _json.loads(enh_raw) if enh_raw.strip() else None
    except Exception:
        enh = None
    bpm_override = request.form.get("bpm_override", "").strip() or None
    key_override = request.form.get("key_override", "").strip() or None
    gains_raw = request.form.get("gains", "")
    try:
        gains = _json.loads(gains_raw) if gains_raw.strip() else None
    except Exception:
        gains = None
    rmodes_raw = request.form.get("render_modes", "")
    try:
        render_modes = _json.loads(rmodes_raw) if rmodes_raw.strip() else None
    except Exception:
        render_modes = None

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(WORK_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)
    in_path = os.path.join(job_dir, "input" + os.path.splitext(f.filename or "a.wav")[1].lower())
    f.save(in_path)

    JOBS[job_id] = {"status": "queued", "pct": 0, "msg": "대기", "input": in_path,
                    "result": None, "error": None, "daw_zip": None}
    threading.Thread(target=_run_job,
                     args=(job_id, in_path, job_dir, preset, backend, daw_export, instrument_map,
                           six_stem, quantize, bpm_override, key_override, gains, keysnap, compare,
                           enh, render_modes, drum_split),
                     daemon=True).start()
    return jsonify({"job_id": job_id})


@app.get("/api/vst_scan")
def vst_scan():
    """설치된 VST3/AU 악기 자동 탐지(큐베이스 기본 악기 등)."""
    try:
        from engine import vst_scan as vs
        return jsonify({"ok": True, "items": vs.scan_vst3()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "items": []})


@app.get("/api/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "no such job"}), 404
    r = {"status": job["status"], "pct": job["pct"], "msg": job["msg"]}
    if job["status"] == "done":
        r["report"] = job.get("report")
        all_names = ["vocals", "drums", "bass", "guitar", "piano", "other"]
        r["files"] = ["original", "result"] + \
            (["result_preserve"] if job.get("result_preserve") else []) + \
            [f"retrack:{n}" for n in all_names
             if os.path.exists(os.path.join(job.get("retrack_dir", ""), f"{n}.wav"))] + \
            [f"stem:{n}" for n in (job.get("stems") or {}).keys()] + \
            [f"stem:{n}" for n in (job.get("extra_stems") or {}).keys()]
        r["daw"] = bool(job.get("daw_zip"))
        r["compare"] = bool(job.get("result_preserve"))
    if job["status"] == "error":
        r["error"] = job.get("error")
    return jsonify(r)


@app.get("/api/file/<job_id>/<kind>")
def get_file(job_id, kind):
    job = JOBS.get(job_id)
    if not job:
        abort(404)
    if kind == "original":
        return send_file(job["input"])
    if kind == "result":
        return send_file(job["result"])
    if kind == "result_preserve":
        if job.get("result_preserve"):
            return send_file(job["result_preserve"])
        abort(404)
    if kind == "daw":
        if job.get("daw_zip"):
            return send_file(job["daw_zip"], as_attachment=True, download_name="daw_package.zip")
        abort(404)
    if ":" in kind:
        grp, name = kind.split(":", 1)
        if grp == "retrack":
            return send_file(os.path.join(job["retrack_dir"], f"{name}.wav"))
        if grp == "stem":
            if name in (job.get("stems") or {}):
                return send_file(job["stems"][name])
            if name in (job.get("extra_stems") or {}):
                return send_file(job["extra_stems"][name])
    abort(404)


@app.post("/api/open_daw/<job_id>")
def open_daw(job_id):
    """로컬에서 all.mid를 DAW(또는 OS 기본 연결)로 연다. 앱이 사용자 PC에서 실행되므로 동작."""
    job = JOBS.get(job_id)
    if not job or not job.get("job_dir"):
        return jsonify({"error": "no job"}), 404
    midi = os.path.join(job["job_dir"], "daw_package", "midi", "all.mid")
    if not os.path.exists(midi):
        return jsonify({"error": "DAW 내보내기를 먼저 체크하고 실행하세요"}), 400
    daw_path = (request.get_json(silent=True) or {}).get("daw_path", "").strip()
    try:
        import subprocess, platform
        sysname = platform.system()
        if daw_path:
            if sysname == "Darwin":
                subprocess.Popen(["open", "-a", daw_path, midi])
            elif sysname == "Windows":
                subprocess.Popen([daw_path, midi])
            else:
                subprocess.Popen([daw_path, midi])
        else:
            if sysname == "Darwin":
                subprocess.Popen(["open", midi])
            elif sysname == "Windows":
                os.startfile(midi)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", midi])
        return jsonify({"ok": True, "opened": midi})
    except Exception as e:
        return jsonify({"error": f"열기 실패: {e}"}), 500


@app.get("/daw")
def daw_page():
    return send_from_directory(STATIC_DIR, "daw.html")


@app.post("/api/export_midi")
def export_midi():
    """피아노롤에서 편집한 파트 노트를 .mid로 써서 반환."""
    data = request.get_json(silent=True) or {}
    notes = data.get("notes", [])
    part = str(data.get("part", "part"))
    bpm = float(data.get("bpm") or 120)
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        is_drum = part == "drums"
        inst = pretty_midi.Instrument(program=0, is_drum=is_drum, name=part)
        for n in notes:
            try:
                s = float(n["start"]); e = max(s + 0.02, float(n["end"]))
                inst.notes.append(pretty_midi.Note(
                    velocity=int(max(1, min(127, n.get("vel", 90)))),
                    pitch=int(max(0, min(127, n["midi"]))), start=s, end=e))
            except Exception:
                continue
        pm.instruments.append(inst)
        tmp = os.path.join(tempfile.gettempdir(), f"pr_{uuid.uuid4().hex[:8]}.mid")
        pm.write(tmp)
        return send_file(tmp, as_attachment=True, download_name=f"{part}.mid")
    except Exception as e:
        return jsonify({"error": f"midi 생성 실패: {e}"}), 500


@app.get("/api/tracks/<job_id>")
def api_tracks(job_id):
    """DAW 편집기용: 잡의 오디오 트랙 목록 + 전사 MIDI(파트별 노트) + 길이."""
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "no job"}), 404
    base = f"/api/file/{job_id}/"
    tracks = []
    # 분리 스템(있으면)
    for n in (job.get("stems") or {}).keys():
        tracks.append({"name": n, "kind": "stem", "url": base + f"stem:{n}"})
    # 재연주 스템(있으면)
    rt = job.get("retrack_dir", "")
    for n in ["vocals", "drums", "bass", "guitar", "piano", "other"]:
        if rt and os.path.exists(os.path.join(rt, f"{n}.wav")):
            tracks.append({"name": f"재연주 {n}", "kind": "retrack", "url": base + f"retrack:{n}"})
    # 보정/결과 단일 트랙(enhance 모드 등)
    rep = job.get("report") or {}
    if rep.get("mode") == "enhance":
        tracks = [{"name": "보정본", "kind": "result", "url": base + "result"}]
    # MIDI 파트 파싱(daw_package 있으면)
    midi = {}
    mdir = os.path.join(job.get("job_dir", ""), "daw_package", "midi")
    if os.path.isdir(mdir):
        try:
            import pretty_midi
            GM = {36: "kick", 38: "snare", 42: "hat", 45: "tom", 49: "crash"}
            for part in ["bass", "other", "guitar", "piano", "drums"]:
                p = os.path.join(mdir, f"{part}.mid")
                if not os.path.exists(p):
                    continue
                pm = pretty_midi.PrettyMIDI(p)
                notes = []
                for inst in pm.instruments:
                    for nt in inst.notes:
                        notes.append({"start": round(float(nt.start), 4),
                                      "end": round(float(nt.end), 4),
                                      "midi": int(nt.pitch), "vel": int(nt.velocity),
                                      "drum": GM.get(int(nt.pitch)) if inst.is_drum else None})
                if notes:
                    midi[part] = notes
        except Exception as e:
            print(f"[tracks] midi parse 실패: {e}")
    return jsonify({"tracks": tracks, "midi": midi,
                    "bpm": rep.get("bpm"), "duration": rep.get("duration_sec")})


def create_app():
    return app
