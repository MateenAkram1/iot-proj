"""Application configuration via environment variables."""

from __future__ import annotations

import os

_FIREBASE_HOST_DEFAULT = "elderly-healthcare-monit-4decf-default-rtdb.firebaseio.com"


def _load_dotenv() -> None:
    """Minimal .env loader for local development."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

FIREBASE_HOST = os.getenv("FIREBASE_HOST", _FIREBASE_HOST_DEFAULT).strip().rstrip("/")
FIREBASE_SECRET = os.getenv("FIREBASE_SECRET", "").strip()

FIREBASE_VITALS_PATH = os.getenv(
    "FIREBASE_VITALS_PATH",
    "patient/vitals",
).strip().strip("/")

POLL_INTERVAL_SEC = float(os.getenv("POLL_INTERVAL_SEC", "4"))
AUTOSTART_POLLER = os.getenv("ENABLE_POLLER", "").lower() in ("1", "true", "yes")

# CNN+LSTM (Keras) artifacts.
_keras_override = os.getenv("FALL_KERAS_MODEL_PATH", "").strip()
FALL_KERAS_MODEL_PATH = _keras_override or os.path.join(
    MODELS_DIR, "fall_detection_model.keras"
)
_encoder_override = os.getenv("LABEL_ENCODER_PATH", "").strip()
LABEL_ENCODER_PATH = _encoder_override or os.path.join(
    MODELS_DIR, "label_encoder.pkl"
)
_scaler_override = os.getenv("SCALER_PATH", "").strip()
SCALER_PATH = _scaler_override or os.path.join(MODELS_DIR, "scaler.pkl")

FALL_MODEL_TIMESTEPS = int(os.getenv("FALL_MODEL_TIMESTEPS", "200"))
FALL_CONFIDENCE_THRESHOLD = float(os.getenv("FALL_CONFIDENCE_THRESHOLD", "0.90"))

# Multiclass fallback: these exact labels in acc_gyr.csv count as fall-related activity.
FALL_ACTIVITY_LABELS = frozenset(
    s.strip().lower()
    for s in os.getenv("FALL_ACTIVITY_LABELS", "fall,lfall,rfall").split(",")
    if s.strip()
)

_fpi_raw = os.getenv("FALL_POSITIVE_CLASS_INDICES", "").strip()
FALL_POSITIVE_CLASS_INDICES: frozenset[int] | None
if _fpi_raw:
    FALL_POSITIVE_CLASS_INDICES = frozenset(int(x.strip()) for x in _fpi_raw.split(",") if x.strip())
else:
    FALL_POSITIVE_CLASS_INDICES = None


def firebase_base_url() -> str:
    if FIREBASE_HOST.startswith("http"):
        return FIREBASE_HOST.rstrip("/")
    return f"https://{FIREBASE_HOST}"
