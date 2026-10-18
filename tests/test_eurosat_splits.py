import pytest

from tiny_omni_decision.eurosat import (
    eurosat_class_from_filename,
    require_hotgpu_reservation,
    require_probability_replay,
    validate_spatial_splits,
)


def test_validate_spatial_splits_accepts_complete_disjoint_partition():
    splits = validate_spatial_splits(
        {
            "train": "Forest_1.jpg\nRiver_1.jpg\n",
            "validation": "Forest_2.jpg\n",
            "test": "River_2.jpg\n",
        },
        expected_members={"Forest_1.jpg", "Forest_2.jpg", "River_1.jpg", "River_2.jpg"},
    )

    assert splits == {
        "train": ("Forest_1.jpg", "River_1.jpg"),
        "validation": ("Forest_2.jpg",),
        "test": ("River_2.jpg",),
    }


@pytest.mark.parametrize(
    ("texts", "expected", "match"),
    [
        (
            {
                "train": "Forest_1.jpg\nForest_1.jpg\n",
                "validation": "River_1.jpg\n",
                "test": "River_2.jpg\n",
            },
            {"Forest_1.jpg", "River_1.jpg", "River_2.jpg"},
            "duplicate",
        ),
        (
            {"train": "Forest_1.jpg\n", "validation": "Forest_1.jpg\n", "test": "River_2.jpg\n"},
            {"Forest_1.jpg", "River_2.jpg"},
            "overlap",
        ),
        (
            {"train": "Forest_1.jpg\n", "validation": "River_1.jpg\n", "test": "River_2.jpg\n"},
            {"Forest_1.jpg", "River_1.jpg", "River_2.jpg", "Highway_1.jpg"},
            "missing",
        ),
        (
            {"train": "../Forest_1.jpg\n", "validation": "River_1.jpg\n", "test": "River_2.jpg\n"},
            {"../Forest_1.jpg", "River_1.jpg", "River_2.jpg"},
            "invalid basename",
        ),
    ],
)
def test_validate_spatial_splits_rejects_invalid_partitions(texts, expected, match):
    with pytest.raises(ValueError, match=match):
        validate_spatial_splits(texts, expected_members=expected)


def test_eurosat_class_from_filename_parses_multitoken_class_name():
    assert eurosat_class_from_filename("HerbaceousVegetation_2701.jpg") == "HerbaceousVegetation"


@pytest.mark.parametrize("filename", ["Forest_x.jpg", "Forest.jpg", "folder/Forest_1.jpg"])
def test_eurosat_class_from_filename_rejects_invalid_filename(filename):
    with pytest.raises(ValueError, match="invalid EuroSAT filename"):
        eurosat_class_from_filename(filename)


def test_probability_replay_returns_observed_max_delta_with_tolerance():
    assert require_probability_replay(
        [[0.6, 0.4]], [[0.6000013, 0.3999987]], atol=2e-6
    ) == pytest.approx(1.3e-6, abs=1e-8)


def test_hotgpu_reservation_requires_complete_valid_telemetry():
    environment = {
        "TINY_OMNI_GPU_GATE_RESERVED_MIB": "3200",
        "TINY_OMNI_GPU_GATE_FREE_MIB": "13000",
        "TINY_OMNI_GPU_GATE_TOTAL_MIB": "16384",
    }

    assert require_hotgpu_reservation(environment) == {
        "TINY_OMNI_GPU_GATE_RESERVED_MIB": 3200,
        "TINY_OMNI_GPU_GATE_FREE_MIB": 13000,
        "TINY_OMNI_GPU_GATE_TOTAL_MIB": 16384,
    }


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {
            "TINY_OMNI_GPU_GATE_RESERVED_MIB": "3200",
            "TINY_OMNI_GPU_GATE_FREE_MIB": "not-an-integer",
            "TINY_OMNI_GPU_GATE_TOTAL_MIB": "16384",
        },
        {
            "TINY_OMNI_GPU_GATE_RESERVED_MIB": "0",
            "TINY_OMNI_GPU_GATE_FREE_MIB": "13000",
            "TINY_OMNI_GPU_GATE_TOTAL_MIB": "16384",
        },
        {
            "TINY_OMNI_GPU_GATE_RESERVED_MIB": "3200",
            "TINY_OMNI_GPU_GATE_FREE_MIB": "13000",
            "TINY_OMNI_GPU_GATE_TOTAL_MIB": "12000",
        },
    ],
)
def test_hotgpu_reservation_fails_closed(environment):
    with pytest.raises(RuntimeError, match="HotUGPU"):
        require_hotgpu_reservation(environment)


@pytest.mark.parametrize(
    ("saved", "replayed", "atol", "match"),
    [
        ([[0.6, 0.4]], [[0.61, 0.39]], 2e-6, "exceeds"),
        ([[0.6, 0.4]], [[0.6]], 2e-6, "shape"),
        ([[0.6, float("nan")]], [[0.6, 0.4]], 2e-6, "finite"),
        ([[0.6, 0.4]], [[0.6, 0.4]], -1.0, "nonnegative"),
    ],
)
def test_probability_replay_rejects_mismatch(saved, replayed, atol, match):
    with pytest.raises(ValueError, match=match):
        require_probability_replay(saved, replayed, atol=atol)
