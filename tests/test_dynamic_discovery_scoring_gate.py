import numpy as np
import pytest

from scripts.score_dynamic_discovery import validate_proposal_bank


def bank():
    return dict(
        masks_packed=np.broadcast_to(np.zeros(1, np.uint8), (1024, 8192)),
        scores=np.zeros(1024, np.float32),
        boxes=np.zeros((1024, 4), np.float64),
        points=np.zeros((1024, 2), np.float32),
        nms_indices=np.array([], np.int64),
    )


def test_complete_raw_bank_preannotation_gate():
    validate_proposal_bank(bank())


@pytest.mark.parametrize("kind", ["missing", "shape", "dtype", "nan", "index"])
def test_incomplete_or_corrupt_raw_bank_cannot_pass_annotation_admission(kind):
    b = bank()
    if kind == "missing":
        del b["points"]
    elif kind == "shape":
        b["masks_packed"] = b["masks_packed"][:1023]
    elif kind == "dtype":
        b["scores"] = b["scores"].astype(np.float64)
    elif kind == "nan":
        b["boxes"][0, 0] = np.nan
    else:
        b["nms_indices"] = np.array([1024], np.int64)
    with pytest.raises(ValueError):
        validate_proposal_bank(b)
