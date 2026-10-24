from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "scripts" / "prepare_clevr4_spatial_count.py"
MODEL_PATH = ROOT / "src" / "tiny_omni_decision" / "spatial_count.py"
EXTRACT_PATH = ROOT / "scripts" / "extract_clevr4_spatial_count_features.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"missing implementation module: {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spatial_count_split_is_deterministic_and_excludes_prior_ids_and_media() -> None:
    split = _load_module(SPLIT_PATH, "prepare_clevr4_spatial_count")
    rows = [{"id": f"image-{i}", "sha256": f"hash-{i}"} for i in range(10)]

    train, validation = split.deterministic_group_split(
        rows,
        train_count=5,
        validation_count=3,
        seed=17,
        excluded_ids={"image-0"},
        excluded_media_hashes={"hash-1"},
    )

    assert (
        train
        == split.deterministic_group_split(
            rows,
            train_count=5,
            validation_count=3,
            seed=17,
            excluded_ids={"image-0"},
            excluded_media_hashes={"hash-1"},
        )[0]
    )
    assert len(train) == 5
    assert len(validation) == 3
    assert not {row["id"] for row in train} & {row["id"] for row in validation}
    assert not {row["sha256"] for row in train} & {row["sha256"] for row in validation}
    assert not {row["id"] for row in train + validation} & {"image-0"}
    assert not {row["sha256"] for row in train + validation} & {"hash-1"}


def test_spatial_count_split_rejects_duplicate_asset_identity() -> None:
    split = _load_module(SPLIT_PATH, "prepare_clevr4_spatial_count")
    rows = [
        {"id": "image-1", "sha256": "same"},
        {"id": "image-2", "sha256": "same"},
    ]

    with pytest.raises(ValueError, match="duplicate media identity"):
        split.deterministic_group_split(
            rows,
            train_count=1,
            validation_count=1,
            seed=17,
            excluded_ids=set(),
            excluded_media_hashes=set(),
        )


def test_video_tokens_are_time_averaged_then_pooled_to_an_ordered_spatial_grid() -> None:
    torch = pytest.importorskip("torch")
    model = _load_module(MODEL_PATH, "spatial_count")
    tokens = torch.zeros((1, 8 * 16 * 16, 2), dtype=torch.float32)
    tokens[:, :, 0] = 1.0
    tokens[:, :, 1] = 3.0
    tokens[:, 0, 0] = 9.0

    pooled = model.pool_video_tokens(tokens, output_grid_size=8)

    assert pooled.shape == (1, 64, 2)
    assert torch.isfinite(pooled).all()
    assert pooled[0, 0, 0] > pooled[0, 63, 0]
    assert torch.allclose(pooled[..., 1], torch.full((1, 64), 3.0))


def test_extractor_retains_spatial_grid_and_averages_repeated_frame_axis() -> None:
    np = pytest.importorskip("numpy")
    extractor = _load_module(EXTRACT_PATH, "extract_spatial_count_features")
    tokens = np.zeros((1, 8 * 16 * 16, 768), dtype=np.float32)
    tokens[:, :, 1] = 3.0
    tokens[:, 0, 0] = 9.0

    pooled = extractor._pooled_spatial_tokens(tokens)

    assert pooled.shape == (1, 64, 768)
    assert pooled.dtype == np.float16
    assert pooled[0, 0, 0] > pooled[0, -1, 0]
    assert np.allclose(pooled[..., 1], 3.0)


def test_resume_prefix_stops_at_first_zero_feature_row() -> None:
    np = pytest.importorskip("numpy")
    extractor = _load_module(EXTRACT_PATH, "extract_spatial_count_features")
    features = np.zeros((5, 2, 3), dtype=np.float16)
    features[:3] = 1.0

    assert extractor._find_completed_prefix(features) == 3


def test_resume_prefix_rejects_nonzero_rows_after_first_unwritten_row() -> None:
    np = pytest.importorskip("numpy")
    extractor = _load_module(EXTRACT_PATH, "extract_spatial_count_features")
    features = np.zeros((5, 2, 3), dtype=np.float16)
    features[0] = 1.0
    features[2] = 1.0

    with pytest.raises(ValueError, match="non-contiguous"):
        extractor._find_completed_prefix(features)


def test_resume_feature_store_replays_rows_after_durable_checkpoint(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    extractor = _load_module(EXTRACT_PATH, "extract_spatial_count_features")
    path = tmp_path / "train-spatial-features.npy"
    saved = np.zeros((5, 2, 3), dtype=np.float16)
    saved[:2] = 1.0
    saved[2:] = 2.0
    np.save(path, saved)

    features, start = extractor._open_feature_store(
        path,
        expected_shape=(5, 2, 3),
        resume_start=2,
    )

    assert start == 2
    assert np.all(features[:2] == 1.0)
    assert np.all(features[2:] == 2.0)
    features[2:] = 3.0
    features.flush()
    reopened = np.load(path, mmap_mode="r")
    assert np.all(reopened[2:] == 3.0)


def test_resume_progress_checkpoint_is_written_atomically(tmp_path: Path) -> None:
    import json

    extractor = _load_module(EXTRACT_PATH, "extract_spatial_count_features")
    path = tmp_path / "resume-progress.json"

    extractor._write_json_atomically(path, {"train": 10_000, "validation": 0})

    assert json.loads(path.read_text(encoding="utf-8")) == {"train": 10_000, "validation": 0}
    assert not path.with_suffix(".json.tmp").exists()


def test_resume_metadata_rejects_checkpoint_or_split_mismatch(tmp_path: Path) -> None:
    extractor = _load_module(EXTRACT_PATH, "extract_spatial_count_features")
    metadata = {
        "status": "starting",
        "input_split_dir": str(tmp_path),
        "checkpoint_sha256_expected": extractor.CHECKPOINT_SHA256,
        "model_revision": extractor.MODEL_REVISION,
        "frame_count": 8,
        "output_shape_per_image": [64, 768],
        "output_dtype": "float16",
        "backend_policy": "CPU only; does not acquire GPU memory",
        "full_corpus": True,
        "smoke_image_limit_per_split": None,
    }

    extractor._validate_resume_metadata(metadata, split_dir=tmp_path)

    for changed in (
        {**metadata, "checkpoint_sha256_expected": "wrong"},
        {**metadata, "input_split_dir": str(tmp_path / "other")},
        {**metadata, "full_corpus": False},
    ):
        with pytest.raises(ValueError, match="resume metadata"):
            extractor._validate_resume_metadata(changed, split_dir=tmp_path)


def test_candidate_conditioned_spatial_scorer_is_option_permutation_equivariant() -> None:
    torch = pytest.importorskip("torch")
    model_module = _load_module(MODEL_PATH, "spatial_count")
    torch.manual_seed(17)
    model = model_module.CandidateConditionedSpatialScorer().eval()
    visual = torch.randn(2, 64, 768)
    options = torch.randn(2, 10, 384)
    permutation = torch.tensor([7, 0, 9, 1, 5, 4, 3, 6, 8, 2])

    with torch.inference_mode():
        scores = model(visual, options)
        shuffled = model(visual, options[:, permutation])

    assert scores.shape == (2, 10)
    assert torch.isfinite(scores).all()
    assert torch.allclose(shuffled, scores[:, permutation], atol=1e-6, rtol=1e-6)
    assert 0 < sum(parameter.numel() for parameter in model.parameters()) < 1_000_000
