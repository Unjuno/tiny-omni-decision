import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scripts.train_spatial_trajectory_readout import paired_batch, paired_logits  # noqa: E402


def test_one_encoded_state_routes_two_questions_and_original_targets(tmp_path):
    from scripts.spatial_readout_inputs import context_tokens
    from tiny_omni_decision.spatial_trajectory_readout import SpatialTrajectoryReadout

    outputs = {}
    for name, value in (
        ("appearance", np.ones((32, 256, 384), np.float32)),
        ("positions", np.zeros((32, 256, 3), np.float32)),
    ):
        path = tmp_path / (name + ".npy")
        np.save(path, value)
        outputs[name] = {"path": str(path)}
    spatial = {"v0": {"outputs": outputs}, "v1": {"outputs": outputs}}
    bank = {"v0": torch.randn(256, 512), "v1": torch.randn(256, 512)}
    data = dict(
        pairs=[{"video_id": "v0"}, {"video_id": "v1"}],
        text=torch.randn(4, 3, 1536),
        targets=torch.tensor([0, 1, 2, 0]),
    )
    points, candidates, context, target = paired_batch(
        bank, spatial, data, [1, 0], "temporal_appearance", "cpu"
    )
    assert points.shape == (2, 256, 512) and candidates.shape == (2, 6, 1536)
    torch.testing.assert_close(target, torch.tensor([2, 0, 0, 1]))
    model = SpatialTrajectoryReadout(hidden=8, latent_count=2, heads=2).eval()
    with torch.no_grad():
        logits, roles = paired_logits(model, points, candidates, context)
        state = model.encode_state(points, context)
        a = model.score_candidates(state, candidates[:, :3])
        b = model.score_candidates(state, candidates[:, 3:])
    torch.testing.assert_close(logits.reshape(2, 2, 3), torch.stack((a, b), 1))
    assert roles.shape == (2, 256)
    np.testing.assert_array_equal(
        context[0].numpy(), context_tokens(spatial["v1"], "temporal_appearance")
    )


def test_zero_control_does_not_modify_cached_points(tmp_path):
    outputs = {}
    for name, shape in (("appearance", (32, 256, 384)), ("positions", (32, 256, 3))):
        path = tmp_path / (name + ".npy")
        np.save(path, np.ones(shape, np.float32))
        outputs[name] = {"path": str(path)}
    bank = {"v": torch.ones(256, 512)}
    data = dict(
        pairs=[{"video_id": "v"}], text=torch.ones(2, 3, 1536), targets=torch.tensor([0, 2])
    )
    values = paired_batch(
        bank, {"v": {"outputs": outputs}}, data, [0], "temporal_appearance", "cpu", "zero"
    )
    assert values[0].sum() == 0 and values[2].sum() == 0
    assert bank["v"].sum() == 256 * 512
    assert values[1].sum() == 6 * 1536


def test_paired_bootstrap_preserves_two_question_video_unit():
    from scripts.train_spatial_trajectory_readout import paired_video_bootstrap

    truth = np.array([0, 1, 1, 2])
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    p, logp = logits.softmax(-1).numpy(), logits.log_softmax(-1).numpy()
    identity = paired_video_bootstrap(p, logp, p, logp, truth)
    assert identity["repetitions"] == 10000 and identity["seed"] == 17
    assert identity["videos"] == 2
    for metric in ("accuracy", "nll", "brier"):
        assert identity[metric]["difference"] == 0
        assert identity[metric]["interval95"] == [0, 0]
    worse = torch.zeros(4, 3).softmax(-1).numpy()
    changed = paired_video_bootstrap(worse, np.log(worse), p, logp, truth)
    assert changed["accuracy"]["difference"] == 0.75
    assert changed["nll"]["difference"] < 0 and changed["brier"]["difference"] < 0


def test_final_resource_gate_rejects_overrun_and_memory(monkeypatch):
    from scripts.train_spatial_trajectory_readout import require_limits

    monkeypatch.setattr("scripts.train_spatial_trajectory_readout.time.perf_counter", lambda: 5)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 10)
    config = dict(max_seconds=10, peak_cap_bytes=20)
    require_limits(0, config)
    with pytest.raises(RuntimeError):
        require_limits(-10, config)
    with pytest.raises(MemoryError):
        require_limits(0, dict(max_seconds=10, peak_cap_bytes=5))
