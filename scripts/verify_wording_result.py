"""Reject incomplete or mismatched wording comparisons before reporting quality."""


def verify_run(config, result, history):
    if result.get("status") != "PASS_EXECUTION_ONLY" or result.get("smoke") is not False:
        raise ValueError("not a completed main run")
    if result.get("steps") != 1024 or result.get("reload_exact") is not True:
        raise ValueError("incomplete or unverified reload")
    if [r["step"] for r in history] != list(range(0, 1025, 128)):
        raise ValueError("missing scheduled evaluation")
    if result["best_step"] != min(history, key=lambda r: r["selector"])["step"]:
        raise ValueError("incorrect selected checkpoint")
    if len(config["parent_ids"]) != 8192 or len(set(config["parent_ids"])) != 5120:
        raise ValueError("parent exposure mismatch")
    last = history[-1]
    if (last["consumed"], last["unique_parent_questions"]) != (8192, 5120):
        raise ValueError("consumption mismatch")
    if last["source_counts"] != {"original": 4096, "curriculum": 4096}:
        raise ValueError("source imbalance")


def verify_pair(left, right):
    for triple in (left, right):
        verify_run(*triple)
    a, _, ah = left
    b, _, bh = right
    for field in (
        "parent_ids", "validation_ids", "seed", "steps", "batch", "lr", "weight_decay",
        "clip", "dropout", "deterministic", "initialization", "checkpoint_sha256",
        "view_corpus_sha256", "input_hashes", "trainable_parameters", "total_parameters",
        "selector", "attention_backend", "torch", "transformers", "cuda",
    ):
        if a[field] != b[field]:
            raise ValueError(f"comparison condition mismatch: {field}")
    if (a["arm"], b["arm"]) != ("control", "wording"):
        raise ValueError("wrong arms")
    # Both arms use view0 in their first 256 updates. Full metric equality is
    # expected before the intervention; prediction files must also be checked.
    for x, y in zip(ah[:3], bh[:3], strict=True):
        for field in ("selector", "original", "curriculum", "tasks"):
            if x[field] != y[field]:
                raise ValueError("pre-intervention metric mismatch")
