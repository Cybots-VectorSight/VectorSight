"""Tests for API endpoints (no LLM calls -- test pipeline integration only)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import FILLED_COMPLEX_SVG, HOME_SVG

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["transforms_registered"] == 14


def test_analyze_filled_svg():
    response = client.post("/api/analyze", json={"svg": FILLED_COMPLEX_SVG})
    assert response.status_code == 200
    data = response.json()
    assert data["enrichment"]["element_count"] >= 1
    assert data["transforms_completed"] >= 1
    assert data["processing_time_ms"] > 0


def test_analyze_home_svg():
    response = client.post("/api/analyze", json={"svg": HOME_SVG})
    assert response.status_code == 200
    data = response.json()
    assert data["enrichment"]["canvas"] == [24.0, 24.0]


def test_analyze_enrichment_has_text():
    response = client.post("/api/analyze", json={"svg": FILLED_COMPLEX_SVG})
    assert response.status_code == 200
    data = response.json()
    assert len(data["enrichment"]["enrichment_text"]) > 0
    assert len(data["enrichment"]["ascii_grid_positive"]) > 0


def test_analyze_empty_svg():
    response = client.post("/api/analyze", json={"svg": "<svg></svg>"})
    assert response.status_code == 200
    data = response.json()
    assert data["enrichment"]["element_count"] == 0


def test_chat_without_api_key():
    """Chat endpoint should return graceful message without API key."""
    response = client.post(
        "/api/chat",
        json={
            "svg": FILLED_COMPLEX_SVG,
            "question": "What shape is this?",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
