import json

import numpy as np
import pytest

from scripts.condition_binding_corpus import all_states, identity


def test_new_partition_is_deterministic_disjoint_and_excludes_history():
    from scripts.condition_contrast import partition_states

    excluded = all_states()[:30]
    p = partition_states(excluded, (16, 8, 8))
    assert p == partition_states(excluded[::-1], (16, 8, 8))
    groups = [{identity(s) for s in states} for states in p.values()]
    assert [len(g) for g in groups] == [16, 8, 8]
    assert len(set.union(*groups)) == 32
    assert not set.union(*groups) & {identity(s) for s in excluded}
    with pytest.raises(ValueError):
        partition_states(all_states(), (16, 8, 8))


def test_full_groups_semantic_oracle_and_distinct_syntax():
    from scripts.condition_contrast import build_rows, validate_group

    states = all_states()[::211][:16]
    train = build_rows(states, "train")
    selection = build_rows(states, "validation")
    transfer = build_rows(states, "transfer")
    assert len(train) == 64 and len(selection) == 128
    assert set(transfer) == {
        "known",
        "focus",
        "complement",
        "focus_descriptions",
        "complement_descriptions",
        "reverse_focus_descriptions",
        "no_question",
    }
    assert all(len(rows) == 64 for rows in transfer.values())
    assert len({r["language_family"] for r in train}) == 8
    for rows in [train, selection, *transfer.values()]:
        for i in range(0, len(rows), 4):
            validate_group(rows[i : i + 4])
            for r in rows[i : i + 4]:
                state = json.loads(r["state"])
                assert r["target_id"] == f"{r['task']}:{state[r['task']]}"
                assert len(r["options"]) == len(set(r["option_ids"])) == 40
    trq = {r["question"] for r in train}
    vaq = {r["question"] for r in selection}
    teq = {
        r["question"]
        for n, rows in transfer.items()
        if n not in ("known", "no_question")
        for r in rows
    }
    assert not trq & vaq and not trq & teq and not vaq & teq
    corrupt = [dict(r) for r in train[:4]]
    corrupt[0]["target_id"] = corrupt[1]["target_id"]
    with pytest.raises(ValueError):
        validate_group(corrupt)


def test_group_schedule_uses_every_question_once_and_keeps_replay_order():
    from scripts.condition_binding_corpus import schedule
    from scripts.condition_contrast import grouped_schedule

    old = schedule(2048, 2048)
    new = grouped_schedule(old)
    assert new == grouped_schedule(old)
    assert len(new) == 512
    for a, b in zip(old, new, strict=True):
        assert [v for v in a if v[0] == 1] == [v for v in b if v[0] == 1]
        indices = [i for s, i in b if s == 0]
        assert len(indices) == 4 and len({i // 4 for i in indices}) == 1
    assert sorted(i for b in new for s, i in b if s == 0) == list(range(2048))


def test_contrast_loss_normalization_gradient_and_permutation():
    torch = pytest.importorskip("torch")
    from scripts.condition_contrast import contrast_loss

    target = torch.tensor([1, 11, 21, 31])
    scores = torch.zeros(4, 40, requires_grad=True)
    loss = contrast_loss(scores.flatten(), [40] * 4, target)
    assert loss.item() == pytest.approx(np.log(4))
    loss.backward()
    assert all(scores.grad[i, target[i]] < 0 for i in range(4))
    assert all(scores.grad[j, target[i]] > 0 for i in range(4) for j in range(4) if i != j)
    logits = torch.arange(160, dtype=torch.float32).reshape(4, 40) / 13
    assert contrast_loss(logits.flatten(), [40] * 4, target).item() == pytest.approx(
        contrast_loss(
            (logits + torch.tensor([7.0, -2.0, 3.0, 1.0])[:, None]).flatten(), [40] * 4, target
        ).item(),
        abs=1e-6,
    )
    order = torch.tensor([2, 0, 3, 1])
    assert contrast_loss(logits[order].flatten(), [40] * 4, target[order]).item() == pytest.approx(
        contrast_loss(logits.flatten(), [40] * 4, target).item(), abs=1e-6
    )
    with pytest.raises(ValueError):
        contrast_loss(scores.flatten(), [40] * 4, torch.tensor([1, 1, 21, 31]))


def test_selection_keeps_step_zero_fallback_but_requires_legacy_retention():
    from scripts.condition_contrast import eligible

    assert eligible(0, False, 0.3)
    assert not eligible(128, True, 0.98)
    assert not eligible(128, False, 1.0)
    assert eligible(128, True, 0.99)


def test_checkpoint_deduplication_uses_tensor_content_not_zip_filename(tmp_path):
    torch = pytest.importorskip("torch")
    from scripts.condition_contrast import delta_digest

    a, b, c = (tmp_path / name for name in ("best.pt", "final.pt", "changed.pt"))
    torch.save({"weight": torch.arange(8).float()}, a)
    torch.save({"weight": torch.arange(8).float()}, b)
    torch.save({"weight": torch.arange(8).float() + 1}, c)
    assert a.read_bytes() != b.read_bytes()
    assert delta_digest(a) == delta_digest(b)
    assert delta_digest(a) != delta_digest(c)
