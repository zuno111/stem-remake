# -*- coding: utf-8 -*-
"""
스템 리메이크 단독 실행 런처.
로컬 서버(waitress)를 띄우고 기본 브라우저를 자동으로 연다.
PyInstaller 엔트리포인트.

시작 중 예외가 나면 창이 바로 닫히지 않도록 로그 파일(stemremake_error.log)에
전체 트레이스백을 남기고 콘솔을 열어둔다.
"""
import os
import sys
import socket
import threading
import traceback
import webbrowser

# PyInstaller onefile/onedir 대응: 번들 루트를 import 경로에 추가
if getattr(sys, "frozen", False):
    BASE = sys._MEIPASS  # type: ignore[attr-defined]
    sys.path.insert(0, BASE)
    _models = os.path.join(BASE, "models")
    if os.path.isdir(_models):
        os.environ.setdefault("TORCH_HOME", _models)
    _LOG_DIR = os.path.dirname(sys.executable)
else:
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, BASE)
    _LOG_DIR = BASE


def _free_port(preferred=8777):
    # 환경변수로 포트 고정(CI 스모크 테스트 등)
    env_port = os.environ.get("STEMREMAKE_PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            pass
    for p in [preferred, 8778, 8779, 0]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", p))
            port = s.getsockname()[1]
            s.close()
            return port
        except OSError:
            continue
    return preferred


def _run():
    from app.server import create_app  # 무거운 의존성 import — 여기서 실패 가능
    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    app = create_app()

    # 실제 접속 URL을 파일로 남겨 둠(CI 스모크 테스트가 읽어 헬스체크, 사용자 참고용)
    try:
        with open(os.path.join(_LOG_DIR, "stemremake_url.txt"), "w", encoding="utf-8") as f:
            f.write(url)
    except Exception:
        pass

    def open_browser():
        import time
        time.sleep(1.5)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    # 헤드리스(CI 스모크 테스트 등)에서는 브라우저 자동 열기 생략
    if os.environ.get("STEMREMAKE_NO_BROWSER") != "1":
        threading.Thread(target=open_browser, daemon=True).start()
    print("=" * 52)
    print("  스템 리메이크 실행 중")
    print(f"  브라우저에서 열기: {url}")
    print("  이 창을 닫으면 종료됩니다.")
    print("=" * 52, flush=True)

    # 자가 진단 모드: 서버를 띄우고 스스로 접속 확인 후 종료(CI/로컬 검증용)
    if os.environ.get("STEMREMAKE_SELFTEST") == "1":
        import urllib.request
        from waitress.server import create_server
        srv = create_server(app, host="127.0.0.1", port=port, threads=4)
        threading.Thread(target=srv.run, daemon=True).start()
        import time
        ok = False
        for _ in range(20):
            time.sleep(0.5)
            try:
                r = urllib.request.urlopen(url, timeout=2)
                if r.status == 200:
                    ok = True
                    break
            except Exception:
                pass
        print(f"SELFTEST {'PASS' if ok else 'FAIL'} url={url}", flush=True)
        os._exit(0 if ok else 2)

    from waitress import serve
    serve(app, host="127.0.0.1", port=port, threads=4)


def main():
    try:
        _run()
    except SystemExit:
        raise
    except BaseException:
        tb = traceback.format_exc()
        try:
            logp = os.path.join(_LOG_DIR, "stemremake_error.log")
            with open(logp, "w", encoding="utf-8") as f:
                f.write(tb)
        except Exception:
            logp = "(로그 파일 기록 실패)"
        sys.stderr.write("\n" + "=" * 52 + "\n")
        sys.stderr.write("  스템 리메이크 시작 중 오류가 발생했습니다.\n")
        sys.stderr.write(f"  오류 로그: {logp}\n")
        sys.stderr.write("=" * 52 + "\n")
        sys.stderr.write(tb + "\n")
        sys.stderr.flush()
        try:
            input("엔터를 누르면 창이 닫힙니다...")
        except Exception:
            import time
            time.sleep(60)
        sys.exit(1)


if __name__ == "__main__":
    main()
