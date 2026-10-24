import math

import pytest

torch = pytest.importorskip("torch")

from scripts.train_perception_source_candidate_readout import (  # noqa: E402
    CrossModalCandidateScorer,
    evaluate,
    grouped_nll,
    score_batch,
)
from tiny_omni_decision.perception_source_candidate_readout import (  # noqa: E402
    CandidateFeatureExample,
)


def test_candidate_scores_are_invariant_to_candidate_order():
    torch.manual_seed(17)
    model = CrossModalCandidateScorer().eval()
    video = torch.randn(1, 4, 1536)
    audio = torch.randn(1, 384)
    mask = torch.ones((1, 4), dtype=torch.bool)
    permutation = torch.tensor([2, 0, 3, 1])

    with torch.inference_mode():
        original = score_batch(model, video, audio, mask)[0, :4]
        reordered = score_batch(model, video[:, permutation], audio, mask)[0, :4]

    torch.testing.assert_close(original[permutation], reordered, rtol=0, atol=1e-7)


def test_grouped_nll_assigns_mass_to_all_valid_candidates():
    logits = torch.tensor([[0.0, 0.0, 0.0]])
    multiple_targets = torch.tensor([[True, True, False]])
    none_target = torch.tensor([[False, False, True]])

    multiple_loss = grouped_nll(logits, multiple_targets)
    none_loss = grouped_nll(logits, none_target)

    torch.testing.assert_close(multiple_loss, torch.tensor([math.log(1.5)]))
    assert none_loss.item() > multiple_loss.item()


def test_evaluation_handles_visible_and_none_visible_records():
    rows = [
        CandidateFeatureExample(
            record_id="r1",
            video_id="v1",
            candidate_ids=("c1", "c2"),
            visual_mean_delta=torch.zeros((2, 1536)).numpy(),
            audio=torch.zeros(384).numpy(),
            offset_audio=torch.zeros(384).numpy(),
            donor_record_id="r2",
            target_candidate_ids=frozenset({"c1"}),
            none_visible=False,
        ),
        CandidateFeatureExample(
            record_id="r2",
            video_id="v2",
            candidate_ids=("c3", "c4"),
            visual_mean_delta=torch.zeros((2, 1536)).numpy(),
            audio=torch.zeros(384).numpy(),
            offset_audio=torch.zeros(384).numpy(),
            donor_record_id="r1",
            target_candidate_ids=frozenset(),
            none_visible=True,
        ),
    ]

    result = evaluate(CrossModalCandidateScorer().eval(), rows)

    assert result["records"] == 2
    assert result["none_visible_records"] == 1
    assert result["visible_source_records"] == 1
    assert 0 <= result["top1_set_accuracy"] <= 1
    assert result["mean_set_nll"] > 0
