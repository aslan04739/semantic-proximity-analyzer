#!/usr/bin/env python3
"""
Embedded macOS app that runs the Flask dashboard locally and displays it in a WebView window.
"""

import threading
import time
import socket
from typing import Tuple

import webview
from werkzeug.serving import make_server

from dashboard_app import app


def get_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()
    return port


class ServerThread(threading.Thread):
    def __init__(self, host: str, port: int):
        super().__init__(daemon=True)
        self.server = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self) -> None:
        self.server.serve_forever()

    def shutdown(self) -> None:
        self.server.shutdown()


def wait_for_server(host: str, port: int, timeout_s: float = 10.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def main() -> None:
    host = "127.0.0.1"
    port = get_free_port()

    server_thread = ServerThread(host, port)
    server_thread.start()

    if not wait_for_server(host, port):
        raise RuntimeError("Failed to start local dashboard server.")

    url = f"http://{host}:{port}"
    window = webview.create_window(
        "Semantic Proximity Analyzer",
        url=url,
        width=1280,
        height=800,
        resizable=True,
    )

    def on_closed() -> None:
        server_thread.shutdown()

    window.events.closed += on_closed
    webview.start(gui="cocoa", debug=False)


if __name__ == "__main__":
    main()
