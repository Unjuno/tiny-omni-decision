"""Ensure the runtime probe fails when a worker returns stale observation features."""

from pathlib import Path

import pytest

from scripts.verify_path_a_observation_updates import verify_updates
from tiny_omni_decision.dataset import sha256_file


@pytest.mark.parametrize("bug", [None, "path_cache", "accept_stale_hash"])
def test_update_probe_detects_stale_cache_and_identity_errors(tmp_path: Path, bug) -> None:
    specs = []
    for index, data in enumerate((b"A", b"B")):
        path = tmp_path / f"source-{index}.png"
        path.write_bytes(data)
        payload = {"image_path": str(path), "expected_media_sha256": sha256_file(path)}
        reference = {
            "sample_id": str(index), "prediction_index": index,
            "probabilities": [0.9, 0.1] if index == 0 else [0.1, 0.9],
        }
        specs.append({
            "task": "image", "payload": payload, "reference": reference,
            "warm_payload": payload, "warm_reference": reference,
        })

    class Worker:
        def __init__(self):
            self.cache = {}

        def dispatch(self, task, payload):
            path = Path(payload["image_path"])
            digest = sha256_file(path)
            if digest != payload["expected_media_sha256"] and bug != "accept_stale_hash":
                raise RuntimeError("image hash differs from the supplied frozen identity")
            key = str(path) if bug == "path_cache" else digest
            hit = key in self.cache
            if not hit:
                self.cache[key] = [0.9, 0.1] if path.read_bytes() == b"A" else [0.1, 0.9]
            probabilities = self.cache[key]
            return {
                "probabilities": probabilities,
                "prediction_index": probabilities.index(max(probabilities)),
                "encoder_metrics": {"cache_hit": hit},
            }

    output = tmp_path / "output"
    output.mkdir()
    if bug is None:
        events = verify_updates(Worker(), specs, output)
        assert len(events) == 7
        assert all(event["passed"] for event in events)
    else:
        with pytest.raises(AssertionError):
            verify_updates(Worker(), specs, output)
    assert (tmp_path / "source-0.png").read_bytes() == b"A"
    assert (tmp_path / "source-1.png").read_bytes() == b"B"
