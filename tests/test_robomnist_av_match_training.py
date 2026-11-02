import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.fit_robomnist_av_match_readout import _fit  # noqa: E402
from tiny_omni_decision.robomnist_av_match_readout import (  # noqa: E402
    BilinearAVMatchScorer,
)


def test_fixed_readout_train_loop_smoke_uses_train_rows_only():
    model = BilinearAVMatchScorer(audio_width=6, video_width=10, rank=4)
    audio = torch.randn(8, 6)
    video = torch.randn(8, 10)
    target = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])

    history = _fit(model, (audio, video), target, steps=2)

    assert [row["step"] for row in history] == [1, 2]
    assert all(np.isfinite(row["train_ce"]) for row in history)
