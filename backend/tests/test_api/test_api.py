"""Tests for API endpoints (no LLM calls — test pipeline integration only)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import CIRCLE_SVG, SMILEY_SVG


client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["transforms_registered"] == 61


def test_analyze_circle():
    response = client.post("/api/analyze", json={"svg": CIRCLE_SVG})
    assert response.status_code == 200
    data = response.json()
    assert data["enrichment"]["element_count"] == 1
    assert data["transforms_completed"] >= 50
    assert data["processing_time_ms"] > 0


def test_analyze_smiley():
    response = client.post("/api/analyze", json={"svg": SMILEY_SVG})
    assert response.status_code == 200
    data = response.json()
    assert data["enrichment"]["element_count"] >= 3
    assert data["enrichment"]["ascii_grid_positive"] != ""
    assert "VECTORSIGHT ENRICHMENT" in data["enrichment"]["enrichment_text"]


def test_analyze_invalid_svg():
    response = client.post("/api/analyze", json={"svg": "<not-svg>"})
    assert response.status_code == 200
    data = response.json()
    # Should handle gracefully (0 elements)
    assert data["enrichment"]["element_count"] == 0


def test_chat_without_api_key():
    """Chat endpoint should return graceful message without API key."""
    response = client.post("/api/chat", json={
        "svg": CIRCLE_SVG,
        "question": "What shape is this?",
    })
    assert response.status_code == 200
    data = response.json()
    # Without API key, should get the not-configured message
    assert "answer" in data
