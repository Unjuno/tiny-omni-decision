from scripts.inat_source_grouping import assign_components


def test_people_connected_across_roles_never_cross_split():
    rows = [
        {"id": 1, "observer_key": "a", "photographer_key": "b"},
        {"id": 2, "observer_key": "b", "photographer_key": "c"},
        {"id": 3, "observer_key": "x", "photographer_key": "y"},
    ]
    assigned, quarantine = assign_components(rows)
    assert not quarantine
    assert assigned[0]["component_key"] == assigned[1]["component_key"]
    assert assigned[0]["split"] == assigned[1]["split"]
    assert assigned[2]["component_key"] != assigned[0]["component_key"]
    assert assign_components(list(reversed(rows)))[0][2]["split"] == assigned[0]["split"]


def test_missing_photographer_is_retained_in_quarantine():
    assigned, quarantine = assign_components([
        {"id": 1, "observer_key": "a", "photographer_key": None}
    ])
    assert assigned == [] and quarantine[0]["id"] == 1
    assert quarantine[0]["quarantine_reason"] == "missing_identity"
