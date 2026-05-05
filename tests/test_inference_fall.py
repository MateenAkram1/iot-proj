"""Fall decision logic (label/index fall classes)."""

import inference as infer


def test_multiclass_label_fallback(monkeypatch):
    monkeypatch.setattr(infer, "_loaded", True)
    monkeypatch.setattr(infer, "_positive_mc_indices", frozenset(), raising=False)
    assert infer.model_indicates_fall({"fall_label": "fall", "fall_class": 0})
    assert not infer.model_indicates_fall({"fall_label": "walk", "fall_class": 6})
