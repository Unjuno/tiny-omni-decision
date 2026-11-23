import pytest


class Tokenizer:
    def __call__(self, texts, *, text_pair=None, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        if text_pair is not None and isinstance(text_pair, str):
            text_pair = [text_pair]
        encoded = []
        for i, text in enumerate(texts):
            values = [1, *[3] * len(text.split())]
            if text_pair is not None:
                values.extend([2, *[4] * len(text_pair[i].split())])
            values.append(2)
            encoded.append(values)
        return {"input_ids": encoded}


def fixture(long_pair=False):
    from scripts.native_language_cache import pair_key, text_key

    state_id = text_key("A small state.")
    states = {state_id: "A small state."}
    question = "question " + "q " * 511 if long_pair else "question"
    candidate = "answer"
    pair = [question, candidate]
    key = pair_key(*pair)
    return dict(
        state_texts=states,
        state_splits={"train": {state_id}, "validation": {text_key("Another state.")}},
        state_texts_all={**states, text_key("Another state."): "Another state."},
        pairs={key: pair},
        query_map={key: dict(bank="old", offset=0)},
        text_observation_ids={"text:" + state_id, "text:" + text_key("Another state.")},
    )


def make_plan(values):
    from scripts.native_bert_feature_cache import build_plan

    return build_plan(
        state_texts=values["state_texts_all"],
        state_splits=values["state_splits"],
        pairs=values["pairs"],
        query_map=values["query_map"],
        text_observation_ids=values["text_observation_ids"],
        tokenizer=Tokenizer(),
    )


def test_plan_tracks_complete_state_and_query_shapes_and_long_pair():
    plan = make_plan(fixture(long_pair=True))
    assert plan["state_count"] == 2
    assert plan["query_count"] == 1
    assert plan["maximum_query_tokens"] == 516
    assert plan["query_over_512"] == 1
    assert plan["shapes"]["queries.npy"] == [1, 768]
    assert plan["tensor_bytes"] == sum(
        rows * width * 4 for rows, width in plan["shapes"].values()
    )


@pytest.mark.parametrize(
    "change",
    ["split_overlap", "missing_state", "observation_mismatch", "missing_pair", "bad_pair_hash"],
)
def test_plan_fails_closed_on_identity_and_split_errors(change):
    values = fixture()
    if change == "split_overlap":
        key = next(iter(values["state_splits"]["train"]))
        values["state_splits"]["validation"].add(key)
    elif change == "missing_state":
        values["state_splits"]["validation"].clear()
    elif change == "observation_mismatch":
        values["text_observation_ids"].clear()
    elif change == "missing_pair":
        values["query_map"].clear()
    else:
        key = next(iter(values["pairs"]))
        values["pairs"][key] = ["different question", "answer"]
    with pytest.raises(ValueError):
        make_plan(values)
