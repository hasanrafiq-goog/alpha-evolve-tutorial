"""Unit tests for DashboardServer daemon, port hunting, and HTTP endpoints."""

import json
import os
import socket
import tempfile
import urllib.request
import pytest

from dashboard_server import DashboardServer, find_available_port
from session_manager import DashboardSessionManager


def test_find_available_port_with_conflict():
    # Bind an existing socket to an ephemeral free port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    occupied_port = sock.getsockname()[1]
    sock.listen(1)

    try:
        # Hunting starting from occupied_port should return occupied_port + 1
        found_port = find_available_port(start_port=occupied_port, max_attempts=5)
        assert found_port > occupied_port
    finally:
        sock.close()


def test_dashboard_server_endpoints_and_asset_serving():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "evolved_output")
        dashboard_dir = os.path.join(tmpdir, "dashboard")
        assets_dir = os.path.join(dashboard_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        # Create dummy index.html and style.css
        index_file = os.path.join(dashboard_dir, "index.html")
        with open(index_file, "w") as f:
            f.write("<html><body>AlphaEvolve Test Console</body></html>")

        css_file = os.path.join(assets_dir, "style.css")
        with open(css_file, "w") as f:
            f.write("body { background: #fff; }")

        session_mgr = DashboardSessionManager(
            output_dir=output_dir,
            target_params=4000,
            target_accuracy=0.94,
            max_programs=10,
        )

        server = DashboardServer(
            dashboard_dir=dashboard_dir,
            session_manager=session_mgr,
            start_port=8900,
            auto_open=False,
        )
        server.start()

        try:
            base_url = server.url
            assert base_url.startswith("http://127.0.0.1:") or base_url.startswith("http://localhost:")

            # 1. Test GET /
            with urllib.request.urlopen(f"{base_url}/") as res:
                assert res.status == 200
                html = res.read().decode("utf-8")
                assert "AlphaEvolve Test Console" in html

            # 2. Test GET /assets/style.css
            with urllib.request.urlopen(f"{base_url}/assets/style.css") as res:
                assert res.status == 200
                css = res.read().decode("utf-8")
                assert "background: #fff" in css

            # 3. Test GET /runs.json
            with urllib.request.urlopen(f"{base_url}/runs.json") as res:
                assert res.status == 200
                data = json.loads(res.read().decode("utf-8"))
                assert data["target_params"] == 4000
                assert data["status"] == "INITIALIZING"

            # 4. Test GET /api/sessions
            with urllib.request.urlopen(f"{base_url}/api/sessions") as res:
                assert res.status == 200
                sessions = json.loads(res.read().decode("utf-8"))
                assert "sessions" in sessions
                assert len(sessions["sessions"]) >= 1
                assert sessions["sessions"][0]["id"] == "active"

        finally:
            server.stop()
