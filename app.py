"""FocusFeed — native macOS app launcher
Starts the FastAPI server in a background thread, then opens the UI
in a native WKWebView window (no browser needed).

Run with:
    python app.py
"""
import socket
import threading
import time

import uvicorn
import webview


def _wait_for_server(host: str = "127.0.0.1", port: int = 8000, timeout: int = 30) -> bool:
    """Poll until the server accepts TCP connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def _run_server() -> None:
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    # Start FastAPI + MLX in a daemon thread
    threading.Thread(target=_run_server, daemon=True).start()

    # Wait for server to be ready (usually < 1 s)
    if not _wait_for_server():
        raise RuntimeError("Server did not start within 30 s.")

    # Open a native macOS window using WebKit (WKWebView)
    webview.create_window(
        title="FocusFeed",
        url="http://127.0.0.1:8000",
        width=580,
        height=920,
        resizable=True,
        min_size=(480, 700),
        background_color="#09090f",   # matches dark bg — no white flash on load
    )
    webview.start()
