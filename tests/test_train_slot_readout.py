import numpy as np
import pytest

pytest.importorskip("torch")
from scripts.train_slot_readout import validate_feature  # noqa: E402


def fixture():
    row = dict(
        video_id="v",
        split="train",
        media_sha256="hash",
        frame_count=32,
        frame_indices=list(range(32)),
    )
    value = np.zeros((224, 129), np.float32)
    value[:, -1] = np.repeat(np.linspace(0, 1, 32, dtype=np.float32), 7)
    return row, value, dict(sha256="hash")


def test_feature_admission():
    validate_feature(*fixture())


def test_rejects_changed_time():
    row, value, media = fixture()
    value[0, -1] = 1
    with pytest.raises(ValueError, match="time"):
        validate_feature(row, value, media)


def test_rejects_wrong_media():
    row, value, media = fixture()
    row["media_sha256"] = "other"
    with pytest.raises(ValueError, match="media"):
        validate_feature(row, value, media)


def test_incomplete_extraction_fails_before_loading_corpus(tmp_path):
    import json

    from scripts.train_slot_readout import load_train

    (tmp_path / "result.json").write_text(
        json.dumps({"status": "RUNNING", "weights_unchanged": False})
    )
    with pytest.raises(ValueError, match="complete"):
        load_train(tmp_path)


def test_fixed_fit_cpu_smoke_and_reload(tmp_path):
    import json

    import torch

    from scripts.train_slot_readout import fit

    torch.manual_seed(17)
    data = dict(
        pairs=[dict(video_id="a"), dict(video_id="b")],
        text=torch.randn(4, 3, 1536),
        targets=torch.tensor([0, 1, 2, 2]),
    )
    bank = {key: torch.randn(224, 129) for key in ["a", "b"]}
    output = tmp_path / "run"
    fit(bank, data, output, {"synthetic": "not-real-experiment"})
    result = json.loads((output / "result.json").read_text())
    assert result["reload_exact"]
    assert result["updates"] == 50
    assert result["unique_questions"] == 4
    assert result["example_presentations"] == 200
    assert len(json.loads((output / "history.json").read_text())) == 51
    assert not (output / "best.pt").exists()


def test_self_consistent_other_experiment_is_rejected(tmp_path):
    import hashlib
    import json

    from scripts.train_slot_readout import load_train

    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    (tmp_path / "result.json").write_text(
        json.dumps(
            dict(
                status="PASS_TRAIN_ONLY_SLOT_EXTRACTION",
                weights_unchanged=True,
                protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
            )
        )
    )
    with pytest.raises(ValueError, match="experiment protocol"):
        load_train(tmp_path)
