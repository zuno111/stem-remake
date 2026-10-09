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
    # 1) 평소 경로 시도(캐시되면 이득)
    for name in ["htdemucs", "htdemucs_6s"]:
        try:
            m = get_model(name)
            print(f"[dl] get_model {name} OK sources={getattr(m,'sources',None)}")
        except Exception as e:
            traceback.print_exc()
            errors.append(f"getmodel:{name}:{type(e).__name__}:{str(e)[:120]}")

    # 2) 결정적 직접 다운로드: demucs 패키지의 remote 목록에서 URL을 뽑아 체크포인트로 저장
    import re
    import torch.hub as hub
    dest0 = os.path.join(hub.get_dir(), "checkpoints")
    os.makedirs(dest0, exist_ok=True)
    try:
        from demucs import pretrained as _pt
        remote_root = getattr(_pt, "REMOTE_ROOT", None)
        # demucs가 파일명 앞에 붙이는 베이스 URL
        root_url = getattr(_pt, "ROOT_URL", None) or "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/"
        if not root_url.endswith("/"):
            root_url += "/"
        print(f"[dl] REMOTE_ROOT={remote_root} ROOT_URL={root_url}")
        if remote_root and os.path.isdir(str(remote_root)):
            rr = str(remote_root)
            files_txt = os.path.join(rr, "files.txt")
            sig_to_url = {}
            if os.path.exists(files_txt):
                for line in open(files_txt, encoding="utf-8"):
                    name = line.strip()
                    if not name:
                        continue
                    base = name.split("/")[-1]
                    # 전체 URL이 아니면 ROOT_URL을 붙인다
                    url = name if name.startswith(("http://", "https://")) else (root_url + base)
                    sig = base.split("-")[0]
                    sig_to_url[sig] = url
            print(f"[dl] files.txt urls={len(sig_to_url)}")
            # 필요한 시그니처: 두 bag yaml에서 추출
            need = set()
            for bag in ["htdemucs", "htdemucs_6s"]:
                y = os.path.join(rr, bag + ".yaml")
                if os.path.exists(y):
                    txt = open(y, encoding="utf-8").read()
                    for m in re.findall(r"[0-9a-f]{8}", txt):
                        need.add(m)
            print(f"[dl] 필요 시그니처: {sorted(need)}")
            targets = []
            if need:
                targets = [(s, sig_to_url[s]) for s in need if s in sig_to_url]
            else:
                targets = list(sig_to_url.items())  # 폴백: 전부
            for sig, url in targets:
                base = url.split("/")[-1]
                out = os.path.join(dest0, base)
                if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
                    print(f"[dl] 이미 있음 {base}")
                    continue
                print(f"[dl] 다운로드 {url}")
                hub.download_url_to_file(url, out, progress=False)
        else:
            errors.append("no_remote_root")
    except Exception as e:
        traceback.print_exc()
        errors.append(f"direct:{type(e).__name__}:{str(e)[:160]}")

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

    # 진단: 가중치가 안 잡히면 HOME·workspace에서 1MB 넘는 파일 census
    big = []
    if copied == 0:
        census_roots = [home, os.getcwd(), models_dir, hub_dir]
        seen = set()
        for cr in census_roots:
            if not cr or not os.path.isdir(cr) or cr in seen:
                continue
            seen.add(cr)
            for r, _d, fs in os.walk(cr):
                for f in fs:
                    try:
                        p = os.path.join(r, f)
                        sz = os.path.getsize(p)
                        if sz > 1_000_000:
                            big.append((sz, p))
                    except Exception:
                        pass
        big.sort(reverse=True)
        print("[dl] 1MB+ 파일 census(상위 20):")
        for sz, p in big[:20]:
            print(f"   {sz/1e6:8.1f}MB  {p}")

    print(f"[dl] 최종 번들 대상: {dest}")
    err_s = (" err=" + " | ".join(errors)) if errors else ""
    top3 = "; ".join(f"{sz//1_000_000}MB:{os.path.basename(p)}@{os.path.dirname(p)}" for sz, p in big[:3])
    diag = f" hub_dir={hub_dir} TORCH_HOME={os.environ.get('TORCH_HOME')} top={top3}" if copied == 0 else ""
    print(f"RESULT weight_files={copied} total_mb={total/1e6:.1f} dest={dest}{err_s}{diag}")
    if copied == 0:
        sys.exit(3)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "models"))
