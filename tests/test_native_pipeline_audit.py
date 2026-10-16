import copy

import numpy as np
import pytest

from scripts.native_pipeline_audit import (
    annotation_counts,
    crop_geometry,
    independent_feature_slice,
    label_coverage,
    pair_digest,
    semantic_text_target,
    validate_associations,
)


def row():
    return dict(id="r", split="train", observation="o", source="starss", task="bell",
                group="g", question="Bell?", options=["no", "yes"], option_ids=["n", "y"],
                target_index=0, queries=[pair_digest("Bell?", o) for o in ["no", "yes"]])


def test_query_target_join_and_reordered_contradiction():
    r = row()
    pairs = dict(zip(r["queries"], [["Bell?", o] for o in r["options"]], strict=True))
    obs = {"o": dict(split="train")}
    assert validate_associations([r], obs, pairs)["conflicting_identical_inputs"] == []
    altered = dict(r, id="r2", options=r["options"][::-1],
                   option_ids=r["option_ids"][::-1], queries=r["queries"][::-1])
    assert validate_associations([r, altered], obs, pairs)["conflicting_identical_inputs"]


@pytest.mark.parametrize("field,value", [("queries", ["wrong", "keys"]),
    ("target_index", -1), ("target_index", True), ("split", "sealed_audit"),
    ("option_ids", ["n", "n"])])
def test_invalid_association_fails(field, value):
    r = row()
    r[field] = value
    with pytest.raises(ValueError):
        validate_associations([r], {"o": dict(split="train")}, {})


def test_zero_positive_and_repeat_counts_are_not_hidden():
    r = row()
    result = label_coverage([r, r])["starss:bell"]
    assert result["positive"] == 0 and result["negative"] == 2
    assert result["unique_rows"] == 1 and result["always_no_accuracy"] == 1
    assert result["candidates_without_target"] == ["yes"]


def test_independent_oracle_and_unknown_legacy():
    r = dict(source="curriculum", state='[{"id":"A","cost":4},{"id":"B","cost":1}]',
             question="Select the item with the minimum cost value.", target="A")
    assert semantic_text_target(r) == "B"  # Deliberately corrupted target is not consulted.
    assert semantic_text_target(dict(source="original")) is None
    r["state"] = '[{"id":"A","cost":1},{"id":"B","cost":1}]'
    with pytest.raises(ValueError, match="tie"):
        semantic_text_target(r)


def test_annotations_merge_speech_and_respect_window_boundary():
    text = "49,0,1,0,0,1\n49,1,2,0,0,1\n50,11,0,0,0,1\n51,12,0,0,0,1"
    assert annotation_counts(text, 0, 5) == [1] + [0] * 11
    assert annotation_counts(text, 5, 10) == [0] * 10 + [1, 1]


def test_crop_exposes_horizontal_loss_without_changing_preprocessing():
    value = crop_geometry(480, 320)
    assert value["retained_fraction"] == pytest.approx(2 / 3)
    assert value["source_box"] == [80, 0, 400, 320]
    assert crop_geometry(256, 256)["retained_fraction"] == 1


def test_slice_rejects_negative_offsets_and_overrun():
    value = np.arange(2 * 4 * 384, dtype=np.float32).reshape(2, 4, 384)
    spec = dict(index=1, offset=1, tokens=2, kind="text")
    np.testing.assert_array_equal(independent_feature_slice(value, spec), value[1, 1:3])
    for change in (dict(offset=-1), dict(tokens=4), dict(index=-1)):
        with pytest.raises(ValueError):
            independent_feature_slice(value, spec | change)


def test_accumulated_permuted_loss_gradient_matches_mean_batch_loss():
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.native_candidate import NativeCandidateReadout, NativeTokenBlock

    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = NativeCandidateReadout()
    other = copy.deepcopy(model)
    blocks = [NativeTokenBlock("text", torch.randn(4, 384), torch.zeros(4, 3),
                               torch.ones(4, dtype=torch.bool))]
    qs = [torch.randn(3, 384), torch.randn(5, 384)]
    targets, perms = [1, 4], [[2, 0, 1], [4, 2, 1, 0, 3]]
    for q, target, perm in zip(qs, targets, perms, strict=True):
        (-model(blocks, q[perm]).log_softmax(0)[perm.index(target)] / 2).backward()
    losses = [-other(blocks, q).log_softmax(0)[t] for q, t in zip(qs, targets, strict=True)]
    torch.stack(losses).mean().backward()
    for a, b in zip(model.parameters(), other.parameters(), strict=True):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6)
