import pytest

from scripts.clotho_audio_controls import source_derangement


def test_source_coherent_bijection_excludes_same_uploader_and_ignores_labels():
    rows = [
        dict(source_id=str(i), uploader="person" + str(i // 2), target=i % 2) for i in range(12)
    ]
    rows += [dict(rows[0], question="another", target=0)]
    mapping = source_derangement(rows)
    assert set(mapping) == set(mapping.values())
    assert len(mapping) == 12
    owners = {r["source_id"]: r["uploader"] for r in rows}
    assert all(owners[a] != owners[b] for a, b in mapping.items())
    assert source_derangement(list(reversed(rows))) == mapping
    assert source_derangement([dict(r, target=1 - r["target"]) for r in rows]) == mapping


def test_conflicting_source_owner_rejected():
    with pytest.raises(ValueError, match="conflicting"):
        source_derangement([dict(source_id="1", uploader="a"), dict(source_id="1", uploader="b")])


def test_impossible_cross_uploader_mapping_rejected():
    with pytest.raises(ValueError):
        source_derangement([dict(source_id=str(i), uploader="same") for i in range(3)])
