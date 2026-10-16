import copy

import numpy as np
import pytest

from scripts.native_language_cache import pair_key


def sample():
    pairs = {pair_key("question", str(i)): ["question", str(i)] for i in range(5)}
    keys = list(pairs)
    mapping = {k: dict(bank="new/queries.npy", offset=i) for i, k in enumerate(keys[:3])}
    mapping.update({k: dict(bank="language/queries.npy", offset=i) for i, k in enumerate(keys[3:])})
    return pairs, mapping, keys


def test_query_order_and_train_only_probe():
    from scripts.native_common_queries import query_schedule

    pairs, mapping, keys = sample()
    schedule = query_schedule(pairs, mapping, {keys[0], keys[2], keys[3]})
    assert schedule["all"] == keys[:3]
    assert schedule["smoke"] == [keys[0], keys[2]]
    assert schedule["reference"] == keys[3]
    assert (
        query_schedule(copy.deepcopy(pairs), copy.deepcopy(mapping), {keys[0], keys[2], keys[3]})
        == schedule
    )


@pytest.mark.parametrize("bad", ["key", "bank", "duplicate", "gap", "missing", "no_train"])
def test_query_mapping_rejects_mutations(bad):
    from scripts.native_common_queries import query_schedule

    pairs, mapping, keys = sample()
    train = set(keys)
    if bad == "key":
        pairs[keys[0]][1] = "changed"
    elif bad == "bank":
        mapping[keys[0]]["bank"] = "other.npy"
    elif bad == "duplicate":
        mapping[keys[1]]["offset"] = 0
    elif bad == "gap":
        mapping[keys[1]]["offset"] = 7
    elif bad == "missing":
        mapping.pop(keys[0])
    elif bad == "no_train":
        train = {keys[3]}
    with pytest.raises(ValueError):
        query_schedule(pairs, mapping, train)


@pytest.mark.parametrize("bad", ["shape", "dtype", "nan"])
def test_query_vector_gate(bad):
    from scripts.native_common_queries import validate_vectors

    value = np.ones((3, 384), dtype=np.float32)
    validate_vectors(value, 3)
    if bad == "shape":
        value = value[:2]
    elif bad == "dtype":
        value = value.astype(np.float64)
    elif bad == "nan":
        value[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_vectors(value, 3)
