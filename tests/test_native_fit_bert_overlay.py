import hashlib
import json

import numpy as np
import pytest


def apply_overlay(*args):
    pytest.importorskip("torch", reason="BERT overlay integration requires the optional ML extra")
    from scripts.native_fit_runtime import apply_text_feature_overlay

    return apply_text_feature_overlay(*args)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def cache_fixture(root):
    states = {
        "state-train": {"file": "train-states.npy", "offset": 0, "tokens": 2, "split": "train"},
        "state-validation": {
            "file": "validation-states.npy",
            "offset": 0,
            "tokens": 1,
            "split": "validation",
        },
    }
    queries = {"query-a": {"offset": 0, "tokens": 4}, "query-b": {"offset": 1, "tokens": 4}}
    shapes = {
        "train-states.npy": [2, 768],
        "validation-states.npy": [1, 768],
        "queries.npy": [2, 768],
    }
    write(root / "index.json", {"states": states, "queries": queries, "shapes": shapes})
    write(root / "query-order.json", ["query-b", "query-a"])
    for name, shape in shapes.items():
        np.save(root / name, np.ones(shape, dtype=np.float32), allow_pickle=False)
    artifacts = {
        name: digest(root / name)
        for name in ["index.json", "query-order.json", *shapes]
    }
    write(
        root / "result.json",
        {
            "status": "PASS_NATIVE_BERT_FEATURES_CACHE",
            "unique_text_states": 2,
            "unique_query_pairs": 2,
            "artifact_hashes": artifacts,
        },
    )
    return digest(root / "result.json")


def fixture_metadata():
    observations = {
        "text:state-train": {"source": "text", "features": [{"kind": "text"}]},
        "text:state-validation": {"source": "text", "features": [{"kind": "text"}]},
        "image:sample": {"source": "image", "features": [{"kind": "image"}]},
    }
    banks = {"image/bank.npy": {"path": "image.npy"}}
    query_map = {
        "query-a": {"bank": "old.npy", "offset": 0},
        "query-b": {"bank": "old.npy", "offset": 1},
    }
    return observations, banks, query_map


def test_bert_overlay_replaces_text_and_all_query_banks_without_touching_other_modalities(
    tmp_path,
):
    receipt_sha = cache_fixture(tmp_path)
    observations, banks, query_map = fixture_metadata()

    updated_observations, updated_banks, updated_queries, width = apply_overlay(
        observations, banks, query_map, tmp_path, receipt_sha
    )

    assert width == 768
    assert updated_observations["text:state-train"]["features"] == [
        {"kind": "text", "bank": "bert/train-states.npy", "offset": 0, "tokens": 2}
    ]
    assert updated_observations["text:state-validation"]["features"][0]["bank"] == (
        "bert/validation-states.npy"
    )
    assert updated_observations["image:sample"] == observations["image:sample"]
    assert updated_queries == {
        "query-b": {"bank": "bert/queries.npy", "offset": 0},
        "query-a": {"bank": "bert/queries.npy", "offset": 1},
    }
    assert updated_banks["image/bank.npy"] == banks["image/bank.npy"]
    assert updated_banks["bert/queries.npy"]["shape"] == [2, 768]


def test_768_readout_reuses_original_seeded_weights_and_adds_only_the_text_lens():
    torch = pytest.importorskip(
        "torch", reason="BERT overlay integration requires the optional ML extra"
    )
    from tiny_omni_decision.native_candidate import NativeCandidateReadout

    torch.manual_seed(17)
    original = NativeCandidateReadout()
    candidate = NativeCandidateReadout(text_input_width=768)
    loaded = candidate.load_state_dict(original.state_dict(), strict=False)

    assert set(loaded.missing_keys) == {"text_lens.weight", "text_lens.bias"}
    assert not loaded.unexpected_keys
    assert sum(parameter.numel() for parameter in candidate.parameters()) == 1_184_641


@pytest.mark.parametrize("corruption", ["receipt", "array", "missing_query", "missing_state"])
def test_bert_overlay_rejects_stale_receipts_and_incomplete_identity_coverage(tmp_path, corruption):
    receipt_sha = cache_fixture(tmp_path)
    observations, banks, query_map = fixture_metadata()
    if corruption == "receipt":
        receipt_sha = "0" * 64
    elif corruption == "array":
        with (tmp_path / "queries.npy").open("ab") as handle:
            handle.write(b"tampered")
    elif corruption == "missing_query":
        query_map.pop("query-b")
    else:
        observations.pop("text:state-validation")

    with pytest.raises(ValueError):
        apply_overlay(observations, banks, query_map, tmp_path, receipt_sha)
