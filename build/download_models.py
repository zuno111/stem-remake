# -*- coding: utf-8 -*-
"""
빌드용 Demucs 모델 다운로더(진단 강화).
htdemucs(4스템)·htdemucs_6s(6스템) 가중치를 받아 <models_dir>/hub/checkpoints 로 모은다.
demucs가 TORCH_HOME을 무시하고 ~/.cache/torch 등에 받을 수 있으므로 넓게 탐색해 복사한다.
진단을 stdout에 찍고, 마지막 줄에 RESULT 라인을 남겨 워크플로가 어노테이션으로 노출한다.
가중치 0개면 비정상 종료(빌드 게이트는 spec의 REQUIRE_MODELS가 담당하지만 여기서도 명확히 실패).
"""
import os
import sys
import glob
import shutil
import traceback

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

EXTS = ("*.th", "*.ckpt", "*.pt", "*.bin")


def _count_th(root):
    if not root or not os.path.isdir(root):
        return []
    out = []
    for ext in EXTS:
        out += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return sorted(set(out))


def main(models_dir):
    os.environ["TORCH_HOME"] = models_dir
    os.makedirs(models_dir, exist_ok=True)

    try:
        import torch  # noqa
        from demucs.pretrained import get_model
    except Exception:
        traceback.print_exc()
        print("RESULT weight_files=0 total_mb=0 reason=import_failed")
        sys.exit(2)

    errors = []
    for name in ["htdemucs", "htdemucs_6s"]:
        try:
            m = get_model(name)
            print(f"[dl] {name} OK sources={getattr(m,'sources',None)}")
        except Exception as e:
            traceback.print_exc()
            msg = f"{name}:{type(e).__name__}:{str(e)[:160]}"
            errors.append(msg)
            print(f"[dl] {name} DOWNLOAD FAILED {msg}")

    try:
        import torch.hub as hub
        hub_dir = hub.get_dir()
    except Exception:
        hub_dir = os.path.join(models_dir, "hub")

    home = os.path.expanduser("~")
    search_roots = [
        models_dir,
        hub_dir,
        os.path.join(home, ".cache", "torch"),
        os.path.join(home, "Library", "Caches", "torch"),
        os.path.join(home, ".cache", "demucs"),
        os.path.join(home, ".cache"),
    ]
    print("[dl] 탐색 경로별 가중치 수:")
    all_found = []
    for r in search_roots:
        fs = _count_th(r)
        print(f"   {r}  exists={os.path.isdir(r)}  weights={len(fs)}")
        all_found += fs
    all_found = sorted(set(all_found))

    dest = os.path.join(models_dir, "hub", "checkpoints")
    os.makedirs(dest, exist_ok=True)
    copied, total = 0, 0
    for f in all_found:
        try:
            tgt = os.path.join(dest, os.path.basename(f))
            if os.path.abspath(f) != os.path.abspath(tgt):
                shutil.copy2(f, tgt)
            total += os.path.getsize(tgt)
            copied += 1
        except Exception:
            traceback.print_exc()

    print(f"[dl] 최종 번들 대상: {dest}")
    err_s = (" err=" + " | ".join(errors)) if errors else ""
    print(f"RESULT weight_files={copied} total_mb={total/1e6:.1f} dest={dest}{err_s}")
    if copied == 0:
        sys.exit(3)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "models"))
