import hashlib
from pathlib import Path

import yaml


def test_split_summary_matches_preregistered_byte_hash():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/pretrained_reuse/path_a_starss23_events.yaml").read_text(
            encoding="utf-8-sig"
        )
    )
    payload = (root / config["split_summary"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == config["split_summary_sha256"]


def test_archived_window_result_preserves_original_byte_hash():
    root = Path(__file__).resolve().parents[1] / "docs/pretrained_reuse"
    payload = (root / "R2_STARSS23_WINDOWS_20261011.json").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "7ed1bf595a36ae9dd334c191806bad8e8d2b9f4574f051c5a40b2f8b79f4c3f1"
    )
