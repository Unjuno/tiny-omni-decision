import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.evaluate_lombard_grid_readouts import (  # noqa: E402
    check_artifact_hashes,
    joint_controls,
    report_identity,
)
from scripts.lombard_readout_core import OPTIONS, SlotHead, fit_normalizer  # noqa: E402
from scripts.lombard_readout_data import sha  # noqa: E402
from scripts.train_lombard_grid_readouts import state_sha  # noqa: E402


def test_checkpoint_and_normalizer_bytes_must_match_recorded_hashes(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"best")
    (tmp_path / "final.pt").write_bytes(b"final")
    (tmp_path / "normalization.npz").write_bytes(b"norm")
    report = dict(
        best_sha256=sha(tmp_path / "best.pt"),
        final_sha256=sha(tmp_path / "final.pt"),
        normalizer_sha256=sha(tmp_path / "normalization.npz"),
    )
    check_artifact_hashes(tmp_path, report)
    (tmp_path / "best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError):
        check_artifact_hashes(tmp_path, report)


def test_controls_keep_frozen_weights_temperatures_sources_and_full_native_vocab(tmp_path):
    rng = np.random.default_rng(17)
    audio, video = (rng.normal(size=(6, w)).astype("float32") for w in [384, 768])
    rows = [
        dict(
            source_id=str(i),
            speaker_group=str(i // 2),
            speaker=str(i // 2),
            condition="p" if i % 2 == 0 else "l",
            targets={s: o[-1] for s, o in OPTIONS.items()},
        )
        for i in range(6)
    ]
    model = SlotHead().eval().requires_grad_(False)
    before = state_sha(model)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = joint_controls(
            model,
            audio,
            video,
            fit_normalizer(audio, video),
            rows,
            np.ones(6),
            tmp_path / "controls",
            device="cpu",
        )
    finally:
        torch.set_num_threads(old_threads)
    assert state_sha(model) == before and result["model_unchanged"]
    assert set(result["conditions"]) == {
        "matched",
        "audio_mismatch",
        "video_mismatch",
        "audio_missing",
        "video_missing",
    }
    for name, values in result["conditions"].items():
        assert values["metrics"]["macro"]["count"] == 36
        records = json.loads((tmp_path / "controls" / (name + "-predictions.json")).read_text())
        assert [r["source_id"] for r in records] == [r["source_id"] for r in rows]
        assert all(len(r["probabilities"]["letter"]) == 25 for r in records)
    with pytest.raises(ValueError):
        joint_controls(
            model.train(),
            audio,
            video,
            fit_normalizer(audio, video),
            rows,
            np.ones(6),
            tmp_path / "invalid",
            device="cpu",
        )


def test_report_rejects_partial_controller_before_source_loading(tmp_path, monkeypatch):
    from scripts import evaluate_lombard_grid_readouts as driver

    folder = tmp_path / "lombard-grid-pooled-readouts-01"
    folder.mkdir()
    (folder / "result.json").write_text(json.dumps({"status": "RUNNING", "smoke": False}))
    monkeypatch.setattr(driver, "ROOT", tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("source/model evaluation before completed three-arm run")

    monkeypatch.setattr(driver, "load_data", forbidden)
    with pytest.raises(ValueError, match="completed"):
        driver.main()
    assert not (tmp_path / "lombard-grid-pooled-readouts-report-01").exists()


def test_artificial_fixture_report_is_never_identified_as_real_quality():
    status, scope = report_identity({"test_fixture_only": True})
    assert status == "PASS_MECHANICAL_FIXTURE_REPORT"
    assert "artificial" in scope and "not model quality" in scope
    assert report_identity({})[0] == "PASS_FIXED_MODEL_REPORT"
