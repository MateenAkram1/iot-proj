"""Fall detection server using latest Firebase IMU sample only."""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import (
    AUTOSTART_POLLER,
    FALL_MODEL_TIMESTEPS,
    POLL_INTERVAL_SEC,
    FIREBASE_HOST,
    FIREBASE_VITALS_PATH,
)
from firebase_rtdb import FirebaseError, fetch_vitals
from firebase_rtdb import vitals_patch_fall_detected
from inference import extract_six_axis, load_models, model_indicates_fall, predict_fall_from_flat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_logger = logging.getLogger("fall_service")

logging.getLogger("werkzeug").setLevel(logging.WARNING)

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers="*",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)

_lock = threading.Lock()
_last_sync_ms: int | None = None
_last_error: str | None = None
_last_prediction: dict[str, Any] | None = None
_last_written_fall: bool | None = None


@app.after_request
def add_cors_headers(response):  # noqa: ANN201
    response.headers.setdefault("Access-Control-Allow-Origin", "*")
    response.headers.setdefault(
        "Access-Control-Allow-Methods",
        "GET, POST, PUT, PATCH, DELETE, OPTIONS",
    )
    response.headers.setdefault(
        "Access-Control-Allow-Headers",
        request.headers.get("Access-Control-Request-Headers", "*"),
    )
    return response


def _read_fall_flag(vitals: dict[str, Any]) -> bool:
    for key in ("fallDetected", "falldetected", "fall_detected"):
        if key in vitals:
            raw = vitals[key]
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, (int, float)):
                return bool(raw)
            if isinstance(raw, str):
                return raw.strip().lower() in ("1", "true", "yes")
    return False


def _imu_line(vitals: dict[str, Any]) -> str:
    pairs = []
    for k in ("accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ"):
        v = vitals.get(k)
        pairs.append(f"{k}={v if v is not None else 'null'}")
    return " ".join(pairs)


def _log_vitals_banner(vitals: dict[str, Any]) -> None:
    _logger.info(
        "Firebase snapshot | RTDB=%s node=%s | timestamp=%r spO2=%r heartRate=%r temp=%r alert=%r",
        FIREBASE_HOST,
        FIREBASE_VITALS_PATH,
        vitals.get("timestamp"),
        vitals.get("spO2"),
        vitals.get("heartRate"),
        vitals.get("temperature"),
        vitals.get("alertStatus"),
    )
    _logger.info("Firebase vitals flags | fallDetected=%r", vitals.get("fallDetected"))
    _logger.info("Model inputs (live) | %s", _imu_line(vitals))


def process_pipeline(*, force: bool = False) -> dict[str, Any]:  # noqa: ARG001
    global _last_sync_ms, _last_error, _last_prediction
    global _last_written_fall

    _logger.info(
        "Step: GET %s JSON (vitals)",
        FIREBASE_VITALS_PATH,
    )

    try:
        vitals = fetch_vitals()
    except FirebaseError as exc:
        with _lock:
            _last_error = str(exc)
        _logger.error("Firebase read failed | %s", exc)
        return {"ok": False, "error": str(exc), "stage": "firebase_read"}

    if vitals is None:
        with _lock:
            _last_error = "no vitals at path"
        _logger.warning("No data at path %s", FIREBASE_VITALS_PATH)
        return {"ok": True, "skipped": True, "reason": "no_vitals"}

    _log_vitals_banner(vitals)

    if _read_fall_flag(vitals):
        with _lock:
            _last_prediction = None
            _last_error = None
            _last_sync_ms = int(time.time() * 1000)
        _logger.info("fallDetected=TRUE in DB → skip model and PATCH.")
        return {
            "ok": True,
            "skipped": True,
            "reason": "fall_already_flagged",
            "fallDetected_server": True,
        }

    _logger.info(
        "fallDetected=FALSE → run model (poll %.1fs repeats until flag set if fall).",
        POLL_INTERVAL_SEC,
    )

    try:
        row6 = extract_six_axis(vitals)
    except ValueError as exc:
        with _lock:
            _last_error = str(exc)
        _logger.error("Parse IMU failed | %s", exc)
        return {"ok": False, "error": str(exc), "stage": "parse_imu"}

    _logger.info("Live 6-vector (accelXYZ, gyroXYZ) | %s", row6.tolist())

    flat = row6.reshape(-1)
    _logger.info("CNN+LSTM inference | live_rows=1 | flat_len=%s", flat.size)

    try:
        load_models()
        preds = predict_fall_from_flat(flat)
    except Exception as exc:  # noqa: BLE001
        _logger.exception("Inference failed")
        with _lock:
            _last_error = str(exc)
            _last_prediction = None
        return {"ok": False, "error": str(exc), "stage": "inference"}

    fall_hit = model_indicates_fall(preds)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    with _lock:
        _last_prediction = {
            **preds,
            "model_fall": fall_hit,
            "evaluated_at": now,
            "window_rows_effective": FALL_MODEL_TIMESTEPS,
            "window": {"window_mode": "firebase_single_row", "live_rows": 1, "model_timesteps": FALL_MODEL_TIMESTEPS},
        }
        _last_sync_ms = int(time.time() * 1000)
        _last_error = None

    _logger.info(
        "Model | class=%s label=%r confidence=%.4f -> fall_alarm=%s",
        preds.get("fall_class"),
        preds.get("fall_label"),
        float(preds.get("fall_confidence") or 0.0),
        fall_hit,
    )

    if fall_hit:
        try:
            _logger.info(
                'PATCH %s merge {{"fallDetected": true}}',
                FIREBASE_VITALS_PATH,
            )
            vitals_patch_fall_detected(True)
            _last_written_fall = True
            _logger.info("PATCH ok: fallDetected=true")
        except FirebaseError as exc:
            with _lock:
                _last_error = str(exc)
            _logger.error("PATCH failed | %s", exc)
            return {"ok": False, "error": str(exc), "stage": "firebase_patch", "predictions": preds}
        return {
            "ok": True,
            "skipped": False,
            "predictions": preds,
            "fallDetected_written": True,
        }

    _last_written_fall = False
    _logger.info("No fall → leave fallDetected false; next poll in %.1fs", POLL_INTERVAL_SEC)
    return {"ok": True, "skipped": False, "predictions": preds, "fallDetected_written": False}


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "service": "iot-fall-detection",
            "vitals_path": FIREBASE_VITALS_PATH,
            "poll_interval_sec": POLL_INTERVAL_SEC,
            "model_timesteps": FALL_MODEL_TIMESTEPS,
            "endpoints": {
                "health": "/health",
                "status": "/api/status",
                "vitals_proxy": "/api/vitals",
                "sync": "/api/sync",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/status", methods=["GET"])
def api_status():
    with _lock:
        return jsonify(
            {
                "last_sync_ms": _last_sync_ms,
                "last_error": _last_error,
                "last_prediction": _last_prediction,
                "last_written_model_fall": _last_written_fall,
                "vitals_path": FIREBASE_VITALS_PATH,
            }
        )


@app.route("/api/vitals", methods=["GET"])
def api_vitals_proxy():
    try:
        data = fetch_vitals()
    except FirebaseError as exc:
        return jsonify({"error": str(exc)}), 502
    if data is None:
        return jsonify({"error": "vitals node empty"}), 404
    return jsonify(data)


@app.route("/api/sync", methods=["GET", "POST"])
def api_sync():
    force = False
    if request.method == "POST" and request.is_json:
        raw = request.get_json(silent=True)
        body = raw if isinstance(raw, dict) else {}
        force = bool(body.get("force"))
    force = force or request.args.get("force") in ("1", "true", "yes")
    result = process_pipeline(force=force)
    status = 500 if not result.get("ok", True) else 200
    return jsonify(result), status


def _poll_loop() -> None:
    cycle = 0
    while True:
        cycle += 1
        _logger.info(
            "%s poll #%s (interval=%ss) %s",
            "=" * 12,
            cycle,
            POLL_INTERVAL_SEC,
            "=" * 12,
        )
        try:
            out = process_pipeline(force=False)
            if not out.get("ok", True):
                _logger.warning("Poll #%s failed | %s", cycle, out)
            else:
                _logger.info(
                    "Poll #%s | skipped=%s reason=%s fallDetected_written=%s",
                    cycle,
                    out.get("skipped"),
                    out.get("reason"),
                    out.get("fallDetected_written"),
                )
        except Exception:  # noqa: BLE001
            _logger.exception("Poll #%s exception", cycle)
        time.sleep(POLL_INTERVAL_SEC)


_poller_lock = threading.Lock()
_poller_running = False


def _start_poller_once() -> None:
    global _poller_running
    with _poller_lock:
        if _poller_running:
            return
        _poller_running = True
        threading.Thread(target=_poll_loop, name="vitals-poller", daemon=True).start()
        _logger.info("Poller started (%ss)", POLL_INTERVAL_SEC)


def _maybe_start_poller() -> None:
    if AUTOSTART_POLLER:
        _start_poller_once()


_maybe_start_poller()


if __name__ == "__main__":
    _logger.info(
        "Flask 0.0.0.0:%s | poll=%ss | vitals=%s",
        os.environ.get("PORT", "5000"),
        POLL_INTERVAL_SEC,
        FIREBASE_VITALS_PATH,
    )
    _start_poller_once()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
