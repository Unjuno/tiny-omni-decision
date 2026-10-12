"""Matched evidence gates; initial logits intentionally differ by readout."""

from scripts.verify_wording_result import verify_run


def verify_joint_pair(left, right):
    for triple in (left, right):
        verify_run(*triple)
    a, _, _ = left
    b, _, _ = right
    if (a["arm"], b["arm"]) != ("independent", "joint"):
        raise ValueError("wrong arms")
    for field in ("parent_ids", "train_ids", "validation_ids", "seed", "steps", "batch",
                  "lr", "weight_decay", "clip", "dropout", "deterministic", "initialization",
                  "checkpoint_sha256", "view_corpus_sha256", "input_hashes",
                  "trainable_parameters", "total_parameters", "selector", "evidence"):
        if a[field] != b[field]:
            raise ValueError(f"mismatched condition: {field}")
    for group in ("train", "validation-0", "validation-1"):
        for field in ("questions", "state_tokens", "kept_state_tokens", "state_truncated"):
            if a["tokenization"][group][field] != b["tokenization"][group][field]:
                raise ValueError("mismatched state evidence accounting")
