"""Inference for CNN+LSTM model using scaler + label encoder."""

from __future__ import annotations

import logging
import os
from typing import Any

import joblib
import numpy as np

from config import (
    FALL_KERAS_MODEL_PATH,
    FALL_ACTIVITY_LABELS,
    FALL_MODEL_TIMESTEPS,
    FALL_POSITIVE_CLASS_INDICES,
    LABEL_ENCODER_PATH,
    SCALER_PATH,
)

_logger = logging.getLogger(__name__)

_model = None
_label_encoder = None
_scaler = None
_loaded = False
_positive_mc_indices: frozenset[int] = frozenset()


def _require_file(path: str, what: str) -> None:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {what} at {path}")


def _multiclass_positive_indices() -> frozenset[int]:
    assert _label_encoder is not None
    classes = list(getattr(_label_encoder, "classes_", []))
    out: set[int] = set()
    if FALL_POSITIVE_CLASS_INDICES is not None:
        return FALL_POSITIVE_CLASS_INDICES
    for i, c in enumerate(classes):
        if str(c).strip().lower() in FALL_ACTIVITY_LABELS:
            out.add(i)
    return frozenset(out)


def load_models(force: bool = False) -> None:
    global _model, _label_encoder, _scaler, _loaded, _positive_mc_indices

    if _loaded and not force:
        return

    _require_file(FALL_KERAS_MODEL_PATH, "keras model (fall_detection_model.keras)")
    _require_file(LABEL_ENCODER_PATH, "label encoder (label_encoder.pkl)")
    _require_file(SCALER_PATH, "scaler (scaler.pkl)")

    from tensorflow.keras.models import load_model

    _model = load_model(FALL_KERAS_MODEL_PATH)
    _label_encoder = joblib.load(LABEL_ENCODER_PATH)
    _scaler = joblib.load(SCALER_PATH)

    _positive_mc_indices = _multiclass_positive_indices()
    _logger.info(
        "Loaded CNN+LSTM model + scaler + encoder | timesteps=%s | fall indices=%s",
        FALL_MODEL_TIMESTEPS,
        sorted(_positive_mc_indices),
    )
    _loaded = True


def extract_six_axis(vitals: dict[str, Any]) -> np.ndarray:
    keys_accel = ("accelX", "accelY", "accelZ")
    keys_gyro = ("gyroX", "gyroY", "gyroZ")
    missing = [k for k in keys_accel + keys_gyro if vitals.get(k) is None]
    if missing:
        raise ValueError(f"Missing IMU scalars at vitals root: {missing}")
    vals = []
    for k in keys_accel + keys_gyro:
        vals.append(float(vitals[k]))
    return np.asarray(vals, dtype=np.float64)


def _prepare_sequence(flat: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Convert flat IMU stream to shape (1, FALL_MODEL_TIMESTEPS, 6), scale rows with scaler.
    If fewer than FALL_MODEL_TIMESTEPS rows are provided, tile the single/latest row.
    """
    assert _scaler is not None
    if flat.size % 6 != 0:
        raise ValueError(f"Expected flat size multiple of 6; got {flat.size}")
    if flat.size == 0:
        raise ValueError("Empty IMU input")

    rows = flat.reshape(-1, 6)
    scaled = _scaler.transform(rows)
    timesteps = FALL_MODEL_TIMESTEPS
    n = scaled.shape[0]

    if n >= timesteps:
        seq = scaled[-timesteps:]
        mode = "last_timesteps"
    else:
        pad = np.tile(scaled[-1].reshape(1, -1), (timesteps - n, 1))
        seq = np.vstack([pad, scaled])
        mode = "tiled_short_input"
    return seq.reshape(1, timesteps, 6), mode


def predict_fall_from_flat(flat2400: np.ndarray) -> dict[str, Any]:
    """Run CNN+LSTM on flattened IMU rows (multiple of 6)."""
    load_models()
    assert _model is not None and _label_encoder is not None

    flat = np.asarray(flat2400, dtype=np.float64).reshape(-1)
    x, feature_mode = _prepare_sequence(flat)

    probs = _model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = str(_label_encoder.inverse_transform(np.asarray([idx]))[0])
    conf = float(np.max(probs))

    out: dict[str, Any] = {
        "fall_class": idx,
        "fall_label": label,
        "fall_confidence": conf,
        "model_kind": "cnn_lstm_keras",
        "feature_mode": feature_mode,
        "timesteps": FALL_MODEL_TIMESTEPS,
    }

    return out


def model_indicates_fall(pred: dict[str, Any]) -> bool:
    """Classify fall from multiclass index / label."""
    load_models()

    idx = int(pred.get("fall_class", -999))
    lab = str(pred.get("fall_label", "")).strip().lower()

    if _positive_mc_indices and idx in _positive_mc_indices:
        return True
    if lab in FALL_ACTIVITY_LABELS:
        return True
    return False
