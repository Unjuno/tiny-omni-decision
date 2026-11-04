import json

import numpy as np
import pytest

pytest.importorskip("torch")

from scripts.audit_correspondence_validation import sha  # noqa: E402
from scripts.train_av_correspondence import load_features  # noqa: E402


def test_development_feature_loader_fails_closed(tmp_path):
    path = tmp_path / "feature-000.npy"
    np.save(path, np.ones(384, dtype=np.float32))
    report = {
        "status": "PASS",
        "windows_sha256": "fixed",
        "features": [{"id": "clip", "sha256": sha(path)}],
    }
    manifest = tmp_path / "result.json"
    manifest.write_text(json.dumps(report), encoding="utf-8")
    assert set(load_features(tmp_path, 384, "fixed", {"clip"}, {})) == {"clip"}
    with pytest.raises(ValueError, match="provenance"):
        load_features(tmp_path, 384, "changed", {"clip"}, {})
    with pytest.raises(ValueError, match="incomplete"):
        load_features(tmp_path, 384, "fixed", {"clip", "missing"}, {})
    with pytest.raises(ValueError, match="invalid feature"):
        load_features(tmp_path, 768, "fixed", {"clip"}, {})
    np.save(path, np.zeros(384, dtype=np.float32))
    with pytest.raises(ValueError, match="changed/duplicate"):
        load_features(tmp_path, 384, "fixed", {"clip"}, {})
