import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("transformers")
pytest.importorskip("av")

from scripts.evaluate_frozen_video_readouts import (  # noqa: E402
    CandidateScorer,
    _feature_map,
    _load_scorer,
    _validate_unseen_scenes,
)
from scripts.extract_frozen_vjepa_video_features import _extract  # noqa: E402


def test_frozen_readout_evaluation_rejects_previous_validation_scenes():
    validation = [{"scene_index": 15}, {"scene_index": 16}]
    exclusions = {"train": [1, 2], "validation": [3, 16]}

    with pytest.raises(ValueError, match="previously used video scene"):
        _validate_unseen_scenes(validation, exclusions)


def test_frozen_readout_evaluation_accepts_new_scene_ids():
    validation = [{"scene_index": 15}, {"scene_index": 16}]
    exclusions = {"train": [1, 2], "validation": [3, 4]}

    _validate_unseen_scenes(validation, exclusions)


def test_vjepa_feature_extraction_reuses_one_forward_per_video(tmp_path, monkeypatch):
    import scripts.extract_frozen_vjepa_video_features as extractor

    class Encoder:
        def __init__(self):
            self.calls = 0

        def __call__(self, clip):
            assert not torch.is_grad_enabled()
            self.calls += 1
            return (
                torch.arange(768, dtype=torch.float32)
                .reshape(1, 1, 768)
                .expand(clip.shape[0], 2, 768)
            )

    rows = [
        {"id": "q1", "media_sha256": "same", "media_path": "video.mp4"},
        {"id": "q2", "media_sha256": "same", "media_path": "video.mp4"},
    ]
    monkeypatch.setattr(
        extractor, "_decode_video", lambda _path: np.zeros((8, 2, 2, 3), dtype=np.uint8)
    )
    encoder = Encoder()

    def transform(_frames):
        return (torch.zeros((3, 8, 2, 2)),)

    features, _ = _extract(rows, tmp_path, encoder, transform, torch.device("cpu"))

    assert encoder.calls == 1
    assert list(features) == ["same"]
    assert features["same"].shape == (768,)


@pytest.mark.parametrize(
    "exclusions",
    [
        {"train": [1]},
        {"train": [1], "validation": [2, 2]},
        {"train": [1], "validation": [True]},
    ],
)
def test_frozen_readout_evaluation_rejects_invalid_scene_exclusion_manifest(exclusions):
    with pytest.raises(ValueError):
        _validate_unseen_scenes([{"scene_index": 15}], exclusions)


def test_readout_loader_freezes_parameters_and_feature_cache_checks_identity(tmp_path):
    path = tmp_path / "readout.pt"
    expected = CandidateScorer(12)
    torch.save(expected.state_dict(), path)
    scorer, digest = _load_scorer(path, torch.device("cpu"))
    assert scorer.feature_size == 12
    assert digest
    assert all(not parameter.requires_grad for parameter in scorer.parameters())

    cache = tmp_path / "features.npz"
    np.savez(cache, media_sha256=np.asarray(["asset-a"]), features=np.ones((1, 768)))
    assert list(_feature_map(cache)) == ["asset-a"]

    np.savez(
        cache,
        media_sha256=np.asarray(["asset-a", "asset-a"]),
        features=np.ones((2, 768)),
    )
    with pytest.raises(ValueError, match="duplicate media identities"):
        _feature_map(cache)
