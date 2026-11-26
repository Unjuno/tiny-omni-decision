import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts import train_fine_grouped_qa as runner  # noqa: E402
from scripts.audit_correspondence_validation import sha  # noqa: E402


def test_smoke_is_exact_main_prefix_without_heldout_metrics():
    main = runner.epoch_order(torch.Generator().manual_seed(17), 79)
    smoke = runner.epoch_order(torch.Generator().manual_seed(17), 79)
    assert main == smoke and len(set(smoke[:16])) == 16


def test_mismatch_moves_whole_state_and_zero_preserves_questions():
    pairs = [dict(video_id="a"), dict(video_id="b")]
    d = dict(
        pairs=pairs,
        text=np.arange(4 * 3 * 1536, dtype=np.float32).reshape(4, 3, 1536),
        targets=np.array([0, 1, 2, 0]),
    )
    points = {k: np.full((256, 512), i + 1, np.float32) for i, k in enumerate(["a", "b"])}
    contexts = {
        "coarse16": {
            k: (np.full((2, 4, 384), i + 1, np.float32), np.zeros((2, 4, 3), np.float32))
            for i, k in enumerate(["a", "b"])
        }
    }
    p, t, c, y = runner.batch(points, contexts, d, [0], "coarse16", "cpu", "mismatch")
    assert p.min() == 2 and c[0, 0, 0] == 2
    assert torch.equal(t, torch.from_numpy(d["text"][:2].reshape(1, 6, 1536)))
    assert y.tolist() == [0, 1]
    p, t, c, y = runner.batch(points, contexts, d, [0], "coarse16", "cpu", "zero")
    assert not p.any() and not c.any() and y.tolist() == [0, 1]


def test_smoke_binding_checks_contract_and_real_saved_bytes(tmp_path):
    (tmp_path / "split.json").write_text("bound split")
    contract = {"seed": 17, "split_sha256": sha(tmp_path / "split.json"), "epochs": 50}
    config = dict(contract=contract, mode="smoke")
    (tmp_path / "config.json").write_text(json.dumps(config))
    artifact = tmp_path / "weights.pt"
    artifact.write_bytes(b"weights")
    evidence = dict(
        status="PASS_TRAIN_ONLY_SMOKE",
        contract=contract,
        contract_sha256=runner.digest(contract),
        config_sha256=sha(tmp_path / "config.json"),
        projected_main_seconds=30,
        arms={
            a: dict(
                parameters=232385,
                initial_sha256="init",
                prefix_orders=list(range(16)),
                prefix_state_sha256="different-" + a,
                hashes={"weights.pt": sha(artifact)},
            )
            for a in ["coarse16", "fine32"]
        },
    )
    for a in evidence["arms"]:
        (tmp_path / a).mkdir()
        (tmp_path / a / "weights.pt").write_bytes(b"weights")
    (tmp_path / "result.json").write_text(json.dumps(evidence))
    runner.verify_smoke(tmp_path, contract)
    with pytest.raises(ValueError):
        runner.verify_smoke(tmp_path, dict(contract, epochs=100))
    (tmp_path / "config.json").write_text("changed")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        runner.verify_smoke(tmp_path, contract)
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "split.json").write_text("changed")
    with pytest.raises(ValueError):
        runner.verify_smoke(tmp_path, contract)
    (tmp_path / "split.json").write_text("bound split")
    evidence["arms"]["fine32"]["hashes"]["weights.pt"] = sha(tmp_path / "fine32/weights.pt")
    refs = runner.evidence_pins(tmp_path, evidence, "smoke")
    assert str(tmp_path / "config.json") in refs and str(tmp_path / "result.json") in refs
    with pytest.raises(ValueError):
        runner.evidence_pins(tmp_path, evidence, "main")
    (tmp_path / "fine32/weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError):
        runner.verify_smoke(tmp_path, contract)
