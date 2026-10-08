# -*- coding: utf-8 -*-
"""
스템 리메이크 단독 실행 런처.
로컬 서버(waitress)를 띄우고 기본 브라우저를 자동으로 연다.
PyInstaller 엔트리포인트.
"""
import os
import sys
import socket
import threading
import webbrowser

# PyInstaller onefile 대응: 번들 루트를 import 경로에 추가
if getattr(sys, "frozen", False):
    BASE = sys._MEIPASS  # type: ignore[attr-defined]
    sys.path.insert(0, BASE)
    # 번들된 Demucs 모델을 오프라인으로 쓰도록 TORCH_HOME 지정
    _models = os.path.join(BASE, "models")
    if os.path.isdir(_models):
        os.environ.setdefault("TORCH_HOME", _models)
else:
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, BASE)

from app.server import create_app  # noqa: E402


def _free_port(preferred=8777):
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


def main():
    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    app = create_app()

    def open_browser():
        import time
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()
    print("=" * 48)
    print("  스템 리메이크 실행 중")
    print(f"  브라우저에서 열기: {url}")
    print("  창을 닫으면 종료됩니다.")
    print("=" * 48)

    from waitress import serve
    serve(app, host="127.0.0.1", port=port, threads=4)


if __name__ == "__main__":
    main()
