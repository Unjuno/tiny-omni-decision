from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_cls_feature_cache_does_not_retain_all_token_hidden_states() -> None:
    from tiny_omni_decision.tensor_features import compact_cls_features

    hidden = torch.zeros((2, 7, 4), dtype=torch.float32)

    cls_features = compact_cls_features(hidden)

    assert cls_features.shape == (2, 4)
    assert cls_features.untyped_storage().nbytes() == cls_features.numel() * 4
