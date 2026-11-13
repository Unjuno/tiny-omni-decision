import copy

import pytest

from scripts.verify_joint_result import verify_joint_pair


def test_joint_comparison_rejects_changed_evidence():
    common = {k: None for k in (
        "train_ids", "validation_ids", "seed", "steps", "batch", "lr", "weight_decay",
        "clip", "dropout", "deterministic", "initialization", "checkpoint_sha256",
        "view_corpus_sha256", "input_hashes", "trainable_parameters", "total_parameters",
        "selector", "evidence")}
    common["parent_ids"] = list(range(5120)) + list(range(3072))
    common["tokenization"] = {group: dict(questions=1, state_tokens=10,
        kept_state_tokens=8, state_truncated=1)
        for group in ("train", "validation-0", "validation-1")}
    a, b = copy.deepcopy(common), copy.deepcopy(common)
    a["arm"], b["arm"] = "independent", "joint"
    result = dict(status="PASS_EXECUTION_ONLY", smoke=False, steps=1024,
                  reload_exact=True, best_step=0)
    h = [dict(step=i, selector=1+i, consumed=i*8, unique_parent_questions=5120,
              source_counts=dict(original=4096, curriculum=4096))
         for i in range(0, 1025, 128)]
    verify_joint_pair((a, result, h), (b, result, h))
    b["tokenization"]["train"]["kept_state_tokens"] = 7
    with pytest.raises(ValueError, match="evidence"):
        verify_joint_pair((a, result, h), (b, result, h))
