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


def _run_job(job_id, input_path, job_dir, preset, backend):
    job = JOBS[job_id]

    def cb(pct, msg):
        job["pct"] = pct
        job["msg"] = msg

    try:
        job["status"] = "running"
        out = pipeline.run(input_path, job_dir, backend=backend, preset=preset, progress_cb=cb)
        job["result"] = out["result"]
        job["stems"] = out["stems"]
        job["retrack_dir"] = out["retrack_dir"]
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

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(WORK_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)
    in_path = os.path.join(job_dir, "input" + os.path.splitext(f.filename or "a.wav")[1].lower())
    f.save(in_path)

    JOBS[job_id] = {"status": "queued", "pct": 0, "msg": "대기", "input": in_path,
                    "result": None, "error": None}
    threading.Thread(target=_run_job, args=(job_id, in_path, job_dir, preset, backend), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.get("/api/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "no such job"}), 404
    r = {"status": job["status"], "pct": job["pct"], "msg": job["msg"]}
    if job["status"] == "done":
        r["report"] = job.get("report")
        r["files"] = ["original", "result"] + \
            [f"retrack:{n}" for n in ["bass", "drums", "other", "vocals"]
             if os.path.exists(os.path.join(job.get("retrack_dir", ""), f"{n}.wav"))] + \
            [f"stem:{n}" for n in ["vocals", "drums", "bass", "other"]]
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
    if ":" in kind:
        grp, name = kind.split(":", 1)
        if grp == "retrack":
            return send_file(os.path.join(job["retrack_dir"], f"{name}.wav"))
        if grp == "stem":
            return send_file(job["stems"][name])
    abort(404)


def create_app():
    return app
