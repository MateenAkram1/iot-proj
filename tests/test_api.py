"""API tests with Firebase and ML mocked."""

from __future__ import annotations

from unittest.mock import patch
import pytest

import app as app_module
from app import app as flask_app


def _minimal_vitals(fall_detected: bool = False) -> dict:
    return {
        "accelX": 0.1,
        "accelY": 0.2,
        "accelZ": 1.0,
        "gyroX": -0.5,
        "gyroY": 1.0,
        "gyroZ": -1.0,
        "timestamp": 99,
        "fallDetected": fall_detected,
    }


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def reset_state():
    with app_module._lock:
        app_module._last_prediction = None
        app_module._last_written_fall = None

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_root(client):
    r = client.get("/")
    body = r.get_json()
    assert body["service"] == "iot-fall-detection"
    assert body["vitals_path"]


def test_status_shape(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.get_json()
    assert "vitals_path" in body
    assert "last_prediction" in body


def test_sync_skips_when_fall_already_flagged(client):
    with patch("app.fetch_vitals", return_value=_minimal_vitals(True)):
        r = client.get("/api/sync")
    assert r.status_code == 200
    assert r.get_json()["reason"] == "fall_already_flagged"


def test_sync_runs_inference_and_patches_fall(client):
    preds_walk = {
        "fall_class": 2,
        "fall_label": "light",
        "fall_confidence": 0.9,
    }
    with (
        patch("app.fetch_vitals", return_value=_minimal_vitals(False)),
        patch("app.load_models"),
        patch("app.predict_fall_from_flat", return_value=preds_walk),
        patch("app.vitals_patch_fall_detected") as patch_fire,
    ):
        r = client.get("/api/sync?force=1")
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] is True
    assert j["predictions"] == preds_walk
    patch_fire.assert_not_called()


def test_sync_patches_when_model_says_fall(client):
    preds_fall = {
        "fall_class": 1,
        "fall_label": "fall",
        "fall_confidence": 0.95,
    }
    with (
        patch("app.fetch_vitals", return_value=_minimal_vitals(False)),
        patch("app.load_models"),
        patch("app.predict_fall_from_flat", return_value=preds_fall),
        patch("app.vitals_patch_fall_detected") as patch_fire,
    ):
        r = client.get("/api/sync?force=1")
    assert r.status_code == 200
    j = r.get_json()
    assert j["fallDetected_written"] is True
    patch_fire.assert_called_once_with(True)


def test_vitals_proxy(client):
    vit = _minimal_vitals(False)
    with patch("app.fetch_vitals", return_value=vit):
        r = client.get("/api/vitals")
    assert r.status_code == 200
    assert r.get_json()["accelX"] == 0.1


def test_cors_preflight(client):
    r = client.open(
        "/api/sync",
        method="OPTIONS",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code in (200, 204)


def test_post_sync_force_json(client):
    preds_walk = {
        "fall_class": 5,
        "fall_label": "walk",
        "fall_confidence": 0.8,
    }
    with (
        patch("app.fetch_vitals", return_value=_minimal_vitals(False)),
        patch("app.load_models"),
        patch("app.predict_fall_from_flat", return_value=preds_walk),
    ):
        r = client.post("/api/sync", json={"force": True})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
