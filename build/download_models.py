# -*- coding: utf-8 -*-
"""
빌드용 Demucs 모델 다운로더.
htdemucs(4스템)·htdemucs_6s(6스템) 가중치를 받아 <models_dir> 아래에 모은다.
torch.hub가 어디에 받든 실제 .th 파일을 찾아 models_dir/hub/checkpoints 로 복사해
PyInstaller 번들 경로를 결정적으로 만든다. 가중치가 하나도 없으면 비정상 종료.

usage: python build/download_models.py <models_dir>
"""
import os
import sys
import glob
import shutil
import ssl

# macOS 등에서 인증서 검증 실패 방지
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass


def main(models_dir):
    os.environ["TORCH_HOME"] = models_dir
    os.makedirs(models_dir, exist_ok=True)
    import torch  # noqa
    from demucs.pretrained import get_model

    for name in ["htdemucs", "htdemucs_6s"]:
        m = get_model(name)
        print(f"[dl] {name} sources={m.sources}")

    # torch hub 체크포인트 위치
    try:
        import torch.hub as hub
        hub_dir = hub.get_dir()
    except Exception:
        hub_dir = os.path.join(models_dir, "hub")
    print(f"[dl] torch hub dir = {hub_dir}")

    dest = os.path.join(models_dir, "hub", "checkpoints")
    os.makedirs(dest, exist_ok=True)

    # 가능한 모든 캐시 위치에서 가중치 수집
    search_roots = [
        models_dir,
        hub_dir,
        os.path.expanduser("~/.cache/torch"),
        os.path.expanduser("~/Library/Caches/torch"),
        os.path.expanduser("~/.cache/demucs"),
    ]
    found = []
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        for ext in ("*.th", "*.ckpt", "*.pt", "*.bin"):
            found += glob.glob(os.path.join(root, "**", ext), recursive=True)
    found = sorted(set(found))

    copied = 0
    total = 0
    for f in found:
        try:
            tgt = os.path.join(dest, os.path.basename(f))
            if os.path.abspath(f) != os.path.abspath(tgt):
                shutil.copy2(f, tgt)
            total += os.path.getsize(tgt)
            copied += 1
        except Exception as e:
            print(f"[dl] copy skip {f}: {e}")

    print(f"[dl] weight files collected: {copied}, total {total/1e6:.1f} MB → {dest}")
    if copied == 0:
        print("[dl] ERROR: 가중치를 하나도 찾지 못함", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "models"))
