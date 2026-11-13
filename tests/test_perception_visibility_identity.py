import pytest

from scripts.audit_perception_visibility_identity import identity_audit


def test_conflicting_sources_need_query_identity():
    row = dict(
        metadata=dict(split="train"),
        sound_localisation=[
            dict(id=0, label="impact", timestamps=[0, 100], is_visible=1),
            dict(id=1, label="background", timestamps=[0, 100], is_visible=0),
        ],
    )
    assert identity_audit({"v": row})["contradictory_inputs"] == 1
    assert identity_audit({"v": row}, True)["contradictory_inputs"] == 0
    row["sound_localisation"][1]["label"] = "impact"
    assert identity_audit({"v": row}, True)["contradictory_inputs"] == 1
    row["metadata"]["split"] = "valid"
    with pytest.raises(ValueError):
        identity_audit({"v": row})
