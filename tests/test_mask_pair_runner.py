import hashlib
import json

import pytest

pytest.importorskip("torch")

from scripts import train_mask_pair_association as runner  # noqa: E402


def test_smoke_actual_config_and_saved_artifacts_must_match(tmp_path):
    cfg = tmp_path / "effective-config.json"
    cfg.write_bytes(b'{"seed":17}\n')
    cfg_sha = hashlib.sha256(cfg.read_bytes()).hexdigest()
    finals = []
    for arm, parameters in [("geometry", 449), ("appearance", 98753)]:
        weights = tmp_path / (arm + ".npz")
        predictions = tmp_path / (arm + "-predictions.npz")
        weights.write_bytes(b"fixed weights")
        predictions.write_bytes(b"fixed predictions")
        finals.append(
            dict(
                arm=arm,
                parameters=parameters,
                reload_exact=True,
                weights_path=str(weights),
                weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                predictions_path=str(predictions),
                predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
            )
        )
    report = dict(
        status="PASS_MASK_PAIR_SMOKE_NOT_QUALITY",
        all_final_reload_exact=True,
        config_sha256=cfg_sha,
        pair_corpus_sha256="corpus",
        steps_per_fit=8,
        completed_fits=2,
        final=finals,
    )
    (tmp_path / "result.json").write_text(json.dumps(report), encoding="utf-8")
    pins = runner.require_smoke_binding(tmp_path, cfg_sha, "corpus")
    assert pins[str(cfg)] == cfg_sha
    cfg.write_bytes(b'{"seed":18}\n')
    with pytest.raises(ValueError):
        runner.require_smoke_binding(tmp_path, cfg_sha, "corpus")
    cfg.write_bytes(b'{"seed":17}\n')
    (tmp_path / "appearance.npz").write_bytes(b"changed")
    with pytest.raises(ValueError):
        runner.require_smoke_binding(tmp_path, cfg_sha, "corpus")
