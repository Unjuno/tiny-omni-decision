import numpy as np
import pytest


def rows():
    result = []
    for mod in ("text", "image", "audio", "video"):
        for cohort in ("one", "two"):
            for task in ("a", "b"):
                for i in range(3):
                    ident = f"{mod}:{cohort}:{task}:{i}"
                    result.append(
                        dict(
                            id=ident,
                            split="train",
                            modality=mod,
                            cohort=cohort,
                            source=cohort,
                            task=task,
                            observation=ident,
                            group=ident,
                            options=["a", "b"],
                            target_index=i % 2,
                        )
                    )
    return result


def test_balanced_order_exhaustion_and_accounting():
    from scripts.native_fit import accounting, sample_order

    data = rows()
    order = sample_order(data, 24)
    assert order == sample_order(data, 24)
    assert len(order) == 24 and all(len(b) == 4 for b in order)
    for batch in order:
        assert {data[i]["modality"] for i in batch} == {"text", "image", "audio", "video"}
    first = [data[i] for b in order[:12] for i in b]
    assert len({r["id"] for r in first}) == len(data)
    counts = accounting([data[i] for b in order for i in b])
    assert counts["consumed"] == 96 and counts["unique_questions"] == 48
    assert counts["repeated_questions"] == 48
    assert counts["unique_observations"] == 48


def test_sampler_rejects_validation_and_duplicate_rows():
    from scripts.native_fit import sample_order

    data = rows()
    with pytest.raises(ValueError):
        sample_order(data + [data[0]], 2)
    data[0]["split"] = "validation"
    with pytest.raises(ValueError):
        sample_order(data, 2)


def test_selector_ties_and_lr_budget():
    from scripts.native_fit import choose_best, learning_rate

    assert choose_best(1.0, 1.0 - 1e-6) is False
    assert choose_best(1.0, 0.9) is True
    assert choose_best(None, 1.0) is True
    with pytest.raises(ValueError):
        choose_best(1.0, float("nan"))
    assert learning_rate(1) == pytest.approx(0.001 / 62)
    assert learning_rate(62) == 0.001
    assert 0 < learning_rate(512) < 0.001
    assert learning_rate(2048) == 0


def test_macro_weights_and_positive_event_coverage():
    from scripts.native_fit import summarize

    data = [
        dict(modality=m, source=m, task="a", options=["a", "b"], target_index=0)
        for m in ("text", "image", "audio", "video")
    ]
    scores = np.array([[3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    report = summarize(data, scores)
    assert report["macro"]["accuracy"] == 0.75
    assert report["minimum_modality_accuracy"] == 0
    repeat = summarize(data + data[:1] * 10, np.vstack([scores, scores[:1].repeat(10, 0)]))
    assert report["macro"] == repeat["macro"]
    event = [
        dict(modality="audio", source="starss", task="bell", options=["no", "yes"], target_index=0)
    ] * 3
    metrics = summarize(event, np.array([[3.0, 0.0]] * 3))["event_classes"]["bell"]
    assert metrics["positive_count"] == 0 and metrics["recall"] is None
    assert metrics["balanced_accuracy"] is None


def test_variable_candidate_metrics_reject_padding_or_invalid_target():
    from scripts.native_fit import summarize

    data = [dict(modality="text", source="text", task="a", options=["a", "b"], target_index=0)]
    with pytest.raises(ValueError):
        summarize(data, np.zeros((1, 3)))
    with pytest.raises(ValueError):
        summarize([dict(data[0], target_index=2)], np.zeros((1, 2)))


def test_feature_store_does_not_use_targets_or_task_names(tmp_path):
    torch = pytest.importorskip("torch")
    from scripts.native_fit_runtime import FeatureStore

    values = np.arange(3 * 384, dtype=np.float32).reshape(3, 384)
    queries = np.ones((2, 384), dtype=np.float32)
    np.save(tmp_path / "states.npy", values)
    np.save(tmp_path / "queries.npy", queries)
    banks = {
        "states": dict(path=str(tmp_path / "states.npy")),
        "language/queries.npy": dict(path=str(tmp_path / "queries.npy")),
    }
    observations = {"x": dict(features=[dict(kind="text", bank="states", offset=0, tokens=3)])}
    mapping = {
        "a": dict(bank="language/queries.npy", offset=0),
        "b": dict(bank="language/queries.npy", offset=1),
    }
    store = FeatureStore(observations, banks, mapping, device="cpu")
    row = dict(observation="x", queries=["a", "b"], target_index=0, task="a")
    blocks, q = store.get(row)
    other, q2 = store.get(dict(row, target_index=1, task="changed", options=["x", "y"]))
    torch.testing.assert_close(blocks[0].values, torch.from_numpy(values), atol=0, rtol=0)
    torch.testing.assert_close(blocks[0].values, other[0].values, atol=0, rtol=0)
    torch.testing.assert_close(q, q2, atol=0, rtol=0)
    assert not q.requires_grad and not blocks[0].values.requires_grad
