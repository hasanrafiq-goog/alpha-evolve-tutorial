"""Unit tests validating HTML, CSS, and JS dashboard structure against specification."""

import os
import re
import pytest

DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard")
INDEX_PATH = os.path.join(DASHBOARD_DIR, "index.html")
CSS_PATH = os.path.join(DASHBOARD_DIR, "assets", "style.css")
JS_PATH = os.path.join(DASHBOARD_DIR, "assets", "dashboard.js")


def test_dashboard_files_exist():
    assert os.path.exists(INDEX_PATH), "index.html must exist"
    assert os.path.exists(CSS_PATH), "style.css must exist"
    assert os.path.exists(JS_PATH), "dashboard.js must exist"


def test_index_html_contains_required_dom_elements():
    with open(INDEX_PATH, "r") as f:
        html = f.read()

    # Header elements
    assert 'id="session-status-badge"' in html
    assert 'id="budget-badge"' in html
    assert 'id="metric-badge"' in html
    assert 'id="session-select"' in html

    # 4 KPI cards
    assert 'id="kpi-top-acc"' in html
    assert 'id="kpi-optimal-params"' in html
    assert 'id="kpi-efficiency"' in html
    assert 'id="kpi-exploration-rate"' in html

    # Table and counts
    assert 'id="runs-table-body"' in html
    assert 'id="runs-count-badge"' in html

    # Canvas elements
    assert 'id="trajectory-chart"' in html
    assert 'id="pareto-chart"' in html

    # Inspection modal
    assert 'id="candidate-modal"' in html
    assert 'id="modal-code"' in html
    assert 'id="modal-metrics"' in html
    assert 'id="modal-insights"' in html


def test_javascript_contains_polling_and_render_functions():
    with open(JS_PATH, "r") as f:
        js = f.read()

    assert "fetch" in js
    assert "pollSessionState" in js or "poll" in js.lower()
    assert "updateKPIs" in js or "kpi" in js.lower()
    assert "renderTable" in js or "table" in js.lower()
    assert "openModal" in js or "modal" in js.lower()
