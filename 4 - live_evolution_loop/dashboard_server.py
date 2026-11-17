"""Lightweight, zero-dependency HTTP server daemon for AlphaEvolve Architecture Search Console.

Serves static HTML/CSS/JS dashboard assets, provides runs.json polling and history endpoints,
and implements dynamic port hunting and browser launching.
"""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
import socket
import threading
from typing import Any, Optional
import urllib.parse
import webbrowser

logger = logging.getLogger(__name__)


def find_available_port(host: str = "127.0.0.1", start_port: int = 8080, max_attempts: int = 10) -> int:
    """Finds an available TCP port starting from start_port, incrementing on collision."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError(f"Could not find an available port in range {start_port} to {start_port + max_attempts}")


class DashboardHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Request handler serving the Single-Page Application and JSON endpoints."""

    def __init__(self, *args, directory=None, session_manager=None, **kwargs):
        self.session_manager = session_manager
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress standard HTTP request logging to keep terminal CLI output clean."""
        logger.debug(format, *args)

    def end_headers(self) -> None:
        """Add anti-caching and CORS headers to JSON and SPA requests."""
        if self.path.endswith(".json") or self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_GET(self) -> None:
        """Routes dashboard endpoints and static assets."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path in ("", "/", "/index.html"):
            index_path = os.path.join(self.directory, "index.html")
            if os.path.exists(index_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(index_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "index.html not found")
                return

        # 1. Active runs.json endpoint
        if path == "/runs.json":
            if self.session_manager and os.path.exists(self.session_manager.runs_path):
                try:
                    with open(self.session_manager.runs_path, "rb") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(content)
                    return
                except Exception as e:
                    logger.debug("Error reading runs.json: %s", e)
            self.send_error(404, "runs.json not found")
            return

        # 2. Available sessions list endpoint
        if path == "/api/sessions":
            if self.session_manager:
                sessions_list = self.session_manager.get_history_sessions()
                payload = json.dumps(
                    {
                        "active_session_id": self.session_manager.session_id,
                        "sessions": sessions_list,
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(500, "Session manager not configured")
            return

        # 3. Specific historical session snapshot endpoint
        if path.startswith("/api/sessions/"):
            session_id = path.replace("/api/sessions/", "").strip()
            if self.session_manager and session_id:
                history_file = os.path.join(self.session_manager.history_dir, f"{session_id}.json")
                if os.path.exists(history_file):
                    with open(history_file, "rb") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(content)
                    return
            self.send_error(404, "Historical session not found")
            return

        # Default static file handler (assets, CSS, JS)
        super().do_GET()


class DashboardServer:
    """Manages the background HTTP daemon thread and browser auto-launcher."""

    def __init__(
        self,
        dashboard_dir: str,
        session_manager: Any,
        host: str = "127.0.0.1",
        start_port: int = 8080,
        auto_open: bool = True,
    ):
        self.dashboard_dir = os.path.abspath(dashboard_dir)
        self.session_manager = session_manager
        self.host = host
        self.port = find_available_port(host=self.host, start_port=start_port)
        self.auto_open = auto_open

        handler = lambda *args, **kwargs: DashboardHTTPRequestHandler(
            *args, directory=self.dashboard_dir, session_manager=self.session_manager, **kwargs
        )
        self.httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.url = f"http://{self.host}:{self.port}"

    def start(self) -> None:
        """Starts the HTTP server daemon and optionally opens the browser."""
        self.server_thread.start()
        logger.info("AlphaEvolve Dashboard server live at %s", self.url)

        if self.auto_open:
            def _open():
                try:
                    webbrowser.open_new_tab(self.url)
                except Exception as e:
                    logger.debug("Could not launch web browser: %s", e)

            threading.Thread(target=_open, daemon=True).start()

    def stop(self) -> None:
        """Stops the HTTP daemon."""
        self.httpd.shutdown()
        self.httpd.server_close()
        logger.debug("Dashboard server stopped.")


def start_dashboard_server(
    dashboard_dir: Optional[str] = None,
    session_manager: Optional[Any] = None,
    port: int = 8080,
    auto_open: bool = True,
) -> DashboardServer:
    """Convenience factory function to start and return a DashboardServer instance."""
    base_dir = os.path.dirname(__file__)
    if dashboard_dir is None:
        dashboard_dir = os.path.join(base_dir, "dashboard")
    if session_manager is None:
        from session_manager import DashboardSessionManager
        session_manager = DashboardSessionManager(
            output_dir=os.path.join(base_dir, "evolved_output")
        )
    server = DashboardServer(
        dashboard_dir=dashboard_dir,
        session_manager=session_manager,
        start_port=port,
        auto_open=auto_open,
    )
    server.start()
    return server
