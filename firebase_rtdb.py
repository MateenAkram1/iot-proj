"""Firebase Realtime Database REST client (auth query param)."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

import requests

from config import FIREBASE_SECRET, firebase_base_url


class FirebaseError(Exception):
    pass


def _url(path: str) -> str:
    path = path.strip().strip("/")
    safe = "/".join(quote(segment, safe="") for segment in path.split("/"))
    base = firebase_base_url()
    return f"{base}/{safe}.json"


def get_json(path: str, timeout: float = 30.0) -> Any:
    if not FIREBASE_SECRET:
        raise FirebaseError("FIREBASE_SECRET is not set")
    r = requests.get(
        _url(path),
        params={"auth": FIREBASE_SECRET},
        timeout=timeout,
    )
    if r.status_code >= 400:
        raise FirebaseError(f"GET {path} failed: {r.status_code} {r.text}")
    if r.text in ("", "null"):
        return None
    return r.json()


def patch_json(path: str, partial: dict[str, Any], timeout: float = 30.0) -> Any:
    """Merge `partial` into the node at `path` without replacing sibling keys."""
    if not FIREBASE_SECRET:
        raise FirebaseError("FIREBASE_SECRET is not set")
    r = requests.patch(
        _url(path),
        params={"auth": FIREBASE_SECRET},
        data=json.dumps(partial),
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    if r.status_code >= 400:
        raise FirebaseError(f"PATCH {path} failed: {r.status_code} {r.text}")
    return r.json() if r.text and r.text != "null" else None


def fetch_vitals() -> dict[str, Any] | None:
    """Load `patient/vitals`-style snapshot from configured path."""
    from config import FIREBASE_VITALS_PATH

    raw = get_json(FIREBASE_VITALS_PATH)
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    raise FirebaseError(f"Vitals node must be an object at {FIREBASE_VITALS_PATH!r}")


def vitals_patch_fall_detected(flag: bool) -> None:
    """Set camelCase flag used by the mobile export."""
    from config import FIREBASE_VITALS_PATH

    patch_json(FIREBASE_VITALS_PATH, {"fallDetected": bool(flag)})

