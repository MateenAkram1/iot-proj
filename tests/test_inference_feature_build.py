"""Sequence prep for CNN+LSTM input."""

import numpy as np
import pytest

import inference
from inference import _prepare_sequence


class _IdentityScaler:
    def transform(self, x):
        return np.asarray(x, dtype=np.float64)


def test_prepare_sequence_long_input(monkeypatch):
    monkeypatch.setattr(inference, "_scaler", _IdentityScaler(), raising=False)
    monkeypatch.setattr(inference, "FALL_MODEL_TIMESTEPS", 200)
    flat = np.arange(400 * 6, dtype=np.float64)
    x, mode = _prepare_sequence(flat)
    assert mode == "last_timesteps"
    assert x.shape == (1, 200, 6)
    assert np.array_equal(x.reshape(-1, 6)[-1], np.asarray([2394, 2395, 2396, 2397, 2398, 2399], dtype=np.float64))


def test_prepare_sequence_short_input(monkeypatch):
    monkeypatch.setattr(inference, "_scaler", _IdentityScaler(), raising=False)
    monkeypatch.setattr(inference, "FALL_MODEL_TIMESTEPS", 200)
    row = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    x, mode = _prepare_sequence(row)
    assert mode == "tiled_short_input"
    assert x.shape == (1, 200, 6)
    assert np.array_equal(x.reshape(-1, 6)[0], row)
    assert np.array_equal(x.reshape(-1, 6)[-1], row)


def test_prepare_sequence_invalid_size(monkeypatch):
    monkeypatch.setattr(inference, "_scaler", _IdentityScaler(), raising=False)
    with pytest.raises(ValueError, match="multiple of 6"):
        _prepare_sequence(np.zeros(7))
