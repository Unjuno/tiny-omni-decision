import numpy as np
import pytest

from scripts.audit_fairplay_audio_pair_alignment import peak_normalized_correlation


def test_peak_correlation_finds_positive_lag_and_documents_delay_convention():
    rng = np.random.default_rng(17)
    embedded = rng.normal(size=2048)
    delay = 23
    paired = np.concatenate((np.zeros(delay), embedded[:-delay]))

    result = peak_normalized_correlation(embedded, paired, max_lag_samples=40)

    assert result["best_lag_samples"] == delay
    assert result["best_correlation"] == pytest.approx(1.0)


def test_peak_correlation_handles_negative_lag():
    rng = np.random.default_rng(18)
    embedded = rng.normal(size=2048)
    advance = 19
    paired = np.concatenate((embedded[advance:], np.zeros(advance)))

    result = peak_normalized_correlation(embedded, paired, max_lag_samples=40)

    assert result["best_lag_samples"] == -advance
    assert result["best_correlation"] == pytest.approx(1.0)


def test_peak_correlation_rejects_nonfinite_input():
    with pytest.raises(ValueError, match="finite"):
        peak_normalized_correlation([0.0, float("nan")], [0.0, 1.0], 0)


def test_audit_receipt_rejects_a_non_train_split_before_media_access():
    from scripts.audit_fairplay_audio_pair_alignment import audit_receipt

    with pytest.raises(ValueError, match="source and split"):
        audit_receipt({"source": "unknown", "split": "test"}, [], 500)


def test_audit_receipt_rejects_non_train_media_entry_before_media_access():
    from scripts.audit_fairplay_audio_pair_alignment import audit_receipt

    receipt = {
        "source": "facebookresearch/FAIR-Play@2666d6caf2d126ae43a84841e94e0c9ae24b3e2d",
        "split": "PseudoBinaural new_splits/split1 train only",
        "cpu_only": True,
        "gpu_used": False,
        "sealed_audit_loaded": False,
        "ids": [f"{i:06d}" for i in range(8)],
        "media": [
            {"id": f"{i:06d}", "split": "PseudoBinaural split1 train"}
            for i in range(8)
        ],
    }
    receipt["media"][0]["split"] = "PseudoBinaural split1 test"

    with pytest.raises(ValueError, match="not explicitly marked split1 train"):
        audit_receipt(receipt, [], 500)
