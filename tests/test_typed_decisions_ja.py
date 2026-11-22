from __future__ import annotations

import pytest
import torch

from scripts.compare_typed_decisions_ja_encoders import _compact_cls_features
from tiny_omni_decision.typed_decisions_ja import (
    normalized_state_sha256,
    parse_typed_decisions_ja_row,
)


def _row() -> dict:
    questions = {
        "choice_a": {
            "type": "choice",
            "instructions": "次の対応を選んでください。",
            "criteria": {"stop": "停止する", "continue": "続ける"},
        },
        "noul_a": {
            "type": "noul",
            "instructions": "この記録は危険ですか。",
            "criteria": {"false": "危険ではない", "true": "危険である"},
        },
        "noul_b": {"type": "noul", "instructions": "この内容は正しいですか。"},
        "score_a": {
            "type": "score",
            "instructions": "リスクを評価してください。",
            "criteria": ["低", "中", "高"],
        },
        "choice_b": {
            "type": "choice",
            "instructions": "結果を選んでください。",
            "criteria": {"success": "成功", "failure": "失敗"},
        },
    }
    labels = {
        "choice_a": ("stop", "continue"),
        "noul_a": ("false", "true"),
        "noul_b": ("false", "true"),
        "score_a": ("0", "1", "2"),
        "choice_b": ("success", "failure"),
    }
    gold = {
        key: {
            "type": questions[key]["type"],
            "label": values[0],
            "probabilities": {label: 1.0 if label == values[0] else 0.0 for label in values},
        }
        for key, values in labels.items()
    }
    return {
        "id": "tr_workflow_000001",
        "workflow": "agent_trace_observability",
        "split": "train",
        "state": {"task": "ログを確認してください", "steps": 3},
        "questions": questions,
        "gold": gold,
        "n_questions": 5,
        "back_translation_match": True,
    }


def test_parser_preserves_all_five_typed_soft_target_groups() -> None:
    groups = parse_typed_decisions_ja_row(_row())

    assert len(groups) == 5
    assert {group.question_type for group in groups} == {"choice", "noul", "score"}
    assert all(group.sample_id == "tr_workflow_000001" for group in groups)
    assert all(group.split == "train" for group in groups)
    assert all(sum(group.target_probabilities) == pytest.approx(1.0) for group in groups)
    missing_criteria_noul = next(group for group in groups if group.question_key == "noul_b")
    assert missing_criteria_noul.option_labels == ("false", "true")
    assert missing_criteria_noul.option_descriptions == ("文の内容は偽である", "文の内容は真である")


def test_parser_refuses_publisher_test_rows() -> None:
    row = _row()
    row["split"] = "test"

    with pytest.raises(ValueError, match="only publisher train"):
        parse_typed_decisions_ja_row(row)


def test_parser_refuses_probability_options_that_do_not_match_schema() -> None:
    row = _row()
    row["gold"]["choice_a"]["probabilities"] = {"stop": 0.5, "unknown": 0.5}

    with pytest.raises(ValueError, match="exactly match"):
        parse_typed_decisions_ja_row(row)


def test_state_identity_hash_normalizes_unicode_and_whitespace() -> None:
    left = {"state": "ＡＢＣ   value", "nested": ["ｅ́"]}
    right = {"nested": ["é"], "state": "ABC value"}

    assert normalized_state_sha256(left) == normalized_state_sha256(right)


def test_cls_feature_cache_does_not_retain_all_token_hidden_states() -> None:
    hidden = torch.zeros((2, 7, 4), dtype=torch.float32)

    cls_features = _compact_cls_features(hidden)

    assert cls_features.shape == (2, 4)
    assert cls_features.untyped_storage().nbytes() == cls_features.numel() * 4
