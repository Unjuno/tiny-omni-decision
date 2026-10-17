import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def fixture(arm):
    from scripts.greatest_hits_matched_runtime import create_model
    from scripts.greatest_hits_replacement import build_inventory
    from tiny_omni_decision.native_candidate import NativeTokenBlock

    torch.manual_seed(17)
    model = create_model(arm)
    rows, features, binding = [], {}, {}
    for task, count in (("action", 2), ("material", 3)):
        q = torch.randn(count, 384)
        for date in ("a", "b"):
            ident = task + date
            options = [str(i) for i in range(count)]
            rows.append(dict(id=ident, task=task, question=task, options=options,
                             split="validation", date_group=date, parent=date,
                             asset_sha256=date, event_id=ident, target=0))
            binding[ident] = dict(options=options, query_keys=options)
            features[ident] = ([NativeTokenBlock(
                kind, torch.randn(2, width), torch.randn(2, 3),
                torch.ones(2, dtype=torch.bool))
                for kind, width in (("speech", 384), ("event_audio", 384), ("video", 768))], q)

    class Store:
        def __init__(self):
            self.binding = {"questions": binding}

        def get(self, ident):
            return features[ident]

    return model, rows, Store(), build_inventory(rows)


@pytest.mark.parametrize("arm", ["C0", "E0G0", "E0G1", "E1G0", "E1G1"])
def test_all_views_real_model_no_updates_and_label_blind(arm):
    from scripts.greatest_hits_replacement_scoring import score

    model, rows, store, plan = fixture(arm)
    before = copy.deepcopy(model.state_dict())
    records = []
    result = score(model, store, rows, plan, emit=records.append)
    assert len(records) == result["model_forwards"] == 20
    assert result["self_replacements_exact"] == 8
    assert result["questions"] == 4
    assert set(result["sensitivity"]) == {"donor_audio", "donor_video"}
    for view in result["sensitivity"].values():
        assert view["overall"]["count"] == 4
        assert set(view["by_task"]) == {"action", "material"}
        assert set(view["by_date"]) == set(view["by_parent"]) == {"a", "b"}
        assert "accuracy" not in view and "nll" not in view
    for record in records:
        assert len(record["logits"]) == len(record["options"])
        assert "target" not in record
        assert (record["gates"] is None) == (arm == "C0")
        expected_donor = (record["donor_id"] if record["view"].startswith("donor_")
                          else record["id"] if record["view"].startswith("self_") else None)
        assert record["input_donor_id"] == expected_donor
        if arm == "E0G1":
            assert set(record["readout_logits"]) == {
                "baseline", "query_only", "full_residual"
            }
            blocks, queries = store.get(record["id"] if record["view"] in
                                         ("natural", "self_audio", "self_video")
                                         else record["input_donor_id"])
            if record["view"].startswith("donor_"):
                from scripts.greatest_hits_replacement import inputs

                blocks, queries = inputs(store, record["id"], record["input_donor_id"],
                                         record["view"].removeprefix("donor_"))
            expected_baseline = model.baseline(blocks, queries)
            q = model.baseline.query(queries)
            zeros = torch.zeros_like(q)
            query_only = expected_baseline + model.correction(
                model._features(q, zeros, zeros)
            ).squeeze(-1)
            np.testing.assert_array_equal(record["readout_logits"]["baseline"],
                                          expected_baseline.detach().numpy())
            np.testing.assert_array_equal(record["readout_logits"]["query_only"],
                                          query_only.detach().numpy())
            np.testing.assert_array_equal(record["readout_logits"]["full_residual"],
                                          record["logits"])
    if arm == "E0G1":
        assert set(result["ablation_sensitivity"]) == {"baseline", "query_only"}
        for per_mode in result["ablation_sensitivity"].values():
            assert set(per_mode) == {"donor_audio", "donor_video"}
            for view in per_mode.values():
                assert view["overall"]["count"] == 4
                assert set(view["by_parent"]) == {"a", "b"}
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, model.state_dict()[name])
        else:
            assert value == model.state_dict()[name]
    assert all(p.grad is None for p in model.parameters())
    for row in rows:
        row["target"] = 1
    second = []
    assert score(model, store, list(reversed(rows)), plan, emit=second.append) == result
    assert second == records


@pytest.mark.parametrize("change", ["donor", "hash", "missing", "views"])
def test_rejects_changed_inventory_before_forward(change):
    from scripts.greatest_hits_replacement_scoring import score

    model, rows, store, plan = fixture("C0")
    if change == "donor":
        plan["pairs"][0]["donor_id"] = plan["pairs"][0]["id"]
    if change == "hash":
        plan["pairs_sha256"] = "wrong"
    if change == "missing":
        plan["pairs"].pop()
    if change == "views":
        plan["views"].pop()
    records = []
    with pytest.raises(ValueError, match="inventory"):
        score(model, store, rows, plan, emit=records.append)
    assert not records


def test_natural_reference_identity_and_logits_must_match():
    from scripts.greatest_hits_replacement_scoring import score

    model, rows, store, plan = fixture("C0")
    records = []
    score(model, store, rows, plan, emit=records.append)
    expected = {r["id"]: np.asarray(r["logits"], dtype=np.float32)
                for r in records if r["view"] == "natural"}
    score(model, store, rows, plan, emit=lambda r: None, expected_natural=expected)
    expected[rows[0]["id"]][0] += 1
    with pytest.raises(ValueError, match="natural"):
        score(model, store, rows, plan, emit=lambda r: None, expected_natural=expected)
    expected.pop(rows[0]["id"])
    with pytest.raises(ValueError, match="identit"):
        score(model, store, rows, plan, emit=lambda r: None, expected_natural=expected)


def test_self_replacement_drift_fails_closed():
    from scripts.greatest_hits_replacement_scoring import score

    model, rows, store, plan = fixture("C0")
    forward = model.forward
    calls = 0

    def drifting(blocks, queries):
        nonlocal calls
        calls += 1
        return forward(blocks, queries) + calls

    model.forward = drifting
    with pytest.raises(ValueError, match="self replacement"):
        score(model, store, rows, plan, emit=lambda r: None)


def test_store_options_must_match_frozen_metadata_before_forward():
    from scripts.greatest_hits_replacement_scoring import score

    model, rows, store, plan = fixture("C0")
    for entry in store.binding["questions"].values():
        entry["options"] = list(reversed(entry["options"]))
    with pytest.raises(ValueError, match="store.*options"):
        score(model, store, rows, plan, emit=lambda r: None)


def test_gpu_process_snapshot_is_reused_only_within_bounded_poll_interval(monkeypatch):
    from scripts import score_greatest_hits_postnorm_replacements as runner

    snapshots = iter(({101}, {202}))
    calls = []

    def read_gpu_pids():
        calls.append(True)
        return next(snapshots)

    monkeypatch.setattr(runner, "active_gpu_pids", read_gpu_pids)
    first, checked_at = runner.refresh_gpu_pids(0.0, float("-inf"), set())
    cached, cached_at = runner.refresh_gpu_pids(1.0, checked_at, first)
    second, checked_at = runner.refresh_gpu_pids(2.0, cached_at, cached)

    assert first == cached == {101}
    assert second == {202}
    assert len(calls) == 2
    assert checked_at == 2.0


@pytest.mark.parametrize("change", ["nan", "short", "guard"])
def test_bad_scores_and_safety_abort_are_not_success(change):
    from scripts.greatest_hits_replacement_scoring import score

    model, rows, store, plan = fixture("C0")
    if change == "nan":
        model.forward = lambda b, q: torch.full((len(q),), float("nan"))
    if change == "short":
        model.forward = lambda b, q: torch.zeros(len(q) - 1)

    def guard():
        if change == "guard":
            raise ValueError("resource guard")

    records = []
    with pytest.raises(ValueError):
        score(model, store, rows, plan, emit=records.append, guard=guard)
    assert not records
