import hashlib
import json

import pytest

from scripts.acquire_perception_av_event_fresh_holdout import validate_plan
from scripts.plan_perception_av_event_fresh_holdout import TRAIN_ARCHIVE_PIN


def _write_plan(path, *, test_opened=False):
    rows = [
        {
            "video_id": f"video_{index}",
            "member": f"videos/video_{index}.mp4",
            "bytes": 1,
            "compressed_bytes": 1,
            "crc32": 0,
            "metadata": {"split": "train", "video_id": f"video_{index}"},
        }
        for index in range(200)
    ]
    plan = {
        "status": "PASS_FRESH_TRAIN_HOLDOUT_PLAN",
        "official_split": "TRAIN only; not a blind audit",
        "training": False,
        "checkpoint_selection": False,
        "test_opened": test_opened,
        "source_revision": "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf",
        "archive_pin": TRAIN_ARCHIVE_PIN,
        "rows": rows,
        "total_uncompressed_bytes": 200,
    }
    path.write_text(json.dumps(plan), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_acquisition_plan_requires_exact_hash_and_train_only_contract(tmp_path):
    path = tmp_path / "plan.json"
    digest = _write_plan(path)
    assert len(validate_plan(path, digest, pinned_sha256=digest)["rows"]) == 200
    with pytest.raises(ValueError, match="preregistered"):
        validate_plan(path, digest)
    with pytest.raises(ValueError, match="plan hash differs"):
        validate_plan(path, "0" * 64, pinned_sha256=digest)


def test_acquisition_plan_rejects_test_access(tmp_path):
    path = tmp_path / "plan.json"
    digest = _write_plan(path, test_opened=True)
    with pytest.raises(ValueError, match="fixed, inference-only"):
        validate_plan(path, digest, pinned_sha256=digest)
