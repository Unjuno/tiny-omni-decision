import pytest


def _label_row(image_id, label, confidence, source="verification"):
    return {
        "ImageID": image_id,
        "Source": source,
        "LabelName": label,
        "Confidence": str(confidence),
    }


def test_builds_deepest_sibling_groups_from_hierarchy():
    from tiny_omni_decision.openimages_options import build_sibling_groups

    hierarchy = {
        "LabelName": "/root",
        "Subcategory": [
            {
                "LabelName": "/vehicles",
                "Subcategory": [
                    {"LabelName": "/car"},
                    {"LabelName": "/bus"},
                    {"LabelName": "/boat"},
                    {"LabelName": "/bicycle"},
                ],
            },
            {
                "LabelName": "/animals",
                "Subcategory": [
                    {"LabelName": "/bird"},
                    {"LabelName": "/cat"},
                    {"LabelName": "/dog"},
                    {"LabelName": "/horse"},
                ],
            },
            {"LabelName": "/other"},
            {"LabelName": "/trucks"},
        ],
    }

    groups = build_sibling_groups(hierarchy, minimum_alternatives=3)

    assert groups["/car"] == ("/bicycle", "/boat", "/bus")
    assert groups["/dog"] == ("/bird", "/cat", "/horse")
    assert groups["/vehicles"] == ("/animals", "/other", "/trucks")


def test_repeated_hierarchy_label_uses_its_deepest_eligible_parent():
    from tiny_omni_decision.openimages_options import build_sibling_groups

    hierarchy = {
        "LabelName": "/root",
        "Subcategory": [
            {"LabelName": "/target"},
            {"LabelName": "/fallback-a"},
            {"LabelName": "/fallback-b"},
            {"LabelName": "/fallback-c"},
            {
                "LabelName": "/deep-parent",
                "Subcategory": [
                    {"LabelName": "/target"},
                    {"LabelName": "/near-a"},
                    {"LabelName": "/near-b"},
                    {"LabelName": "/near-c"},
                ],
            },
        ],
    }

    groups = build_sibling_groups(hierarchy, minimum_alternatives=3)

    assert groups["/target"] == ("/near-a", "/near-b", "/near-c")


def test_builds_deterministic_four_choice_from_explicit_labels_only():
    from tiny_omni_decision.openimages_options import make_four_choice_example

    sibling_groups = {
        "/car": ("/boat", "/bicycle", "/bus"),
        "/bus": ("/boat", "/bicycle", "/car"),
        "/dog": ("/boat", "/bicycle", "/bus"),
    }
    kwargs = {
        "image_id": "img-001",
        "positive_labels": {"/car", "/dog"},
        "explicit_negative_labels": {"/boat", "/bicycle", "/bus"},
        "sibling_groups": sibling_groups,
        "target_scope": {"/car"},
        "display_names": {
            "/car": "Car",
            "/boat": "Boat",
            "/bicycle": "Bicycle",
            "/bus": "Bus",
        },
        "seed": 17,
    }

    example = make_four_choice_example(**kwargs)

    assert example == make_four_choice_example(**kwargs)
    assert example["sample_id"] == "openimages-v6:img-001"
    assert len(example["options"]) == len(set(example["options"])) == 4
    assert example["target"] == "Car"
    assert example["target_label"] == "/car"
    assert set(example["options"]) == {"Car", "Boat", "Bicycle", "Bus"}
    assert example["question"] == "Which object is present in the image?"


def test_does_not_treat_unmentioned_classes_as_negative():
    from tiny_omni_decision.openimages_options import make_four_choice_example

    example = make_four_choice_example(
        image_id="img-002",
        positive_labels={"/car"},
        explicit_negative_labels={"/bus"},
        sibling_groups={"/car": ("/boat", "/bicycle", "/bus")},
        target_scope={"/car"},
        seed=17,
    )

    assert example is None


def test_rejects_same_image_label_as_both_positive_and_negative():
    from tiny_omni_decision.openimages_options import make_four_choice_example

    with pytest.raises(ValueError, match="both positive and negative"):
        make_four_choice_example(
            image_id="img-003",
            positive_labels={"/car"},
            explicit_negative_labels={"/car", "/boat", "/bicycle", "/bus"},
            sibling_groups={"/car": ("/boat", "/bicycle", "/bus")},
            target_scope={"/car"},
            seed=17,
        )


def test_requires_nonempty_stable_image_identity():
    from tiny_omni_decision.openimages_options import make_four_choice_example

    with pytest.raises(ValueError, match="image_id"):
        make_four_choice_example(
            image_id="",
            positive_labels={"/car"},
            explicit_negative_labels={"/boat", "/bicycle", "/bus"},
            sibling_groups={"/car": ("/boat", "/bicycle", "/bus")},
            target_scope={"/car"},
            seed=17,
        )


def test_selects_object_targets_from_train_support_only_with_frozen_name_scope():
    from tiny_omni_decision.openimages_options import (
        EXCLUDED_OPENIMAGES_NAME_TERMS,
        MINIMUM_OPENIMAGES_TRAIN_SUPPORT,
        select_target_classes,
    )

    selected = select_target_classes(
        {"/car": 1000, "/dog": 900, "/hand": 5000, "/boat": 999},
        {
            "/car": "Car",
            "/dog": "Dog",
            "/hand": "Hand",
            "/boat": "Boat",
        },
        minimum_train_support=1000,
        excluded_name_terms={"hand"},
    )

    assert selected == {"/car": 1000}
    assert MINIMUM_OPENIMAGES_TRAIN_SUPPORT == 1000
    assert "person" in EXCLUDED_OPENIMAGES_NAME_TERMS


def test_target_scope_rejects_missing_or_invalid_training_support():
    from tiny_omni_decision.openimages_options import select_target_classes

    with pytest.raises(ValueError, match="positive"):
        select_target_classes(
            {"/car": 0},
            {"/car": "Car"},
            minimum_train_support=0,
            excluded_name_terms=set(),
        )

    with pytest.raises(ValueError, match="display name"):
        select_target_classes(
            {"/car": 100},
            {},
            minimum_train_support=1,
            excluded_name_terms=set(),
        )


def test_groups_verification_labels_by_image_and_rejects_unsorted_source():
    from tiny_omni_decision.openimages_options import group_verification_labels

    rows = [
        _label_row("a", "/car", 1),
        _label_row("a", "/boat", 1, source="crowdsource-verification"),
        _label_row("a", "/bus", 0),
        _label_row("b", "/boat", 1),
    ]

    assert list(group_verification_labels(rows)) == [
        {"image_id": "a", "positive_labels": {"/car"}, "negative_labels": {"/bus"}},
        {"image_id": "b", "positive_labels": {"/boat"}, "negative_labels": set()},
    ]

    with pytest.raises(ValueError, match="sorted by ImageID"):
        list(group_verification_labels([rows[3], rows[0]]))


def test_counts_supported_targets_once_per_image_and_only_for_allowlisted_images():
    from tiny_omni_decision.openimages_options import (
        count_supported_targets,
        group_verification_labels,
    )

    rows = []
    for image_id in ("a", "b"):
        rows.extend(
            [
                _label_row(image_id, "/car", 1),
                _label_row(image_id, "/boat", 0),
                _label_row(image_id, "/bicycle", 0),
                _label_row(image_id, "/bus", 0),
            ]
        )

    counts = count_supported_targets(
        group_verification_labels(rows),
        sibling_groups={"/car": ("/boat", "/bicycle", "/bus")},
        allowed_image_ids={"a"},
    )

    assert counts == {"/car": 1}


def test_balanced_candidate_selection_is_deterministic_and_image_unique():
    from tiny_omni_decision.openimages_options import select_balanced_candidates

    candidates = [
        {"image_id": "a", "target_label": "/car"},
        {"image_id": "b", "target_label": "/car"},
        {"image_id": "c", "target_label": "/car"},
        {"image_id": "a", "target_label": "/dog"},
        {"image_id": "d", "target_label": "/dog"},
        {"image_id": "e", "target_label": "/dog"},
    ]

    selected = select_balanced_candidates(
        candidates,
        target_labels=("/car", "/dog"),
        per_target_limit=2,
        seed=17,
    )

    assert selected == select_balanced_candidates(
        candidates,
        target_labels=("/car", "/dog"),
        per_target_limit=2,
        seed=17,
    )
    assert len({item["image_id"] for item in selected}) == len(selected)
    assert {
        label: sum(item["target_label"] == label for item in selected)
        for label in ("/car", "/dog")
    } == {
        "/car": 2,
        "/dog": 2,
    }


def test_balanced_candidate_selection_rejects_duplicate_image_target_pair():
    from tiny_omni_decision.openimages_options import select_balanced_candidates

    candidate = {"image_id": "a", "target_label": "/car"}
    with pytest.raises(ValueError, match="duplicate image/target"):
        select_balanced_candidates(
            [candidate, candidate],
            target_labels=("/car",),
            per_target_limit=1,
            seed=17,
        )


def test_emits_candidates_only_for_frozen_targets_with_three_explicit_siblings():
    from tiny_omni_decision.openimages_options import iter_target_candidates

    groups = [
        {
            "image_id": "a",
            "positive_labels": {"/car", "/dog"},
            "negative_labels": {"/boat", "/bicycle", "/bus"},
        },
        {
            "image_id": "b",
            "positive_labels": {"/car"},
            "negative_labels": {"/bus"},
        },
    ]

    candidates = list(
        iter_target_candidates(
            groups,
            target_labels=("/car",),
            sibling_groups={"/car": ("/boat", "/bicycle", "/bus")},
        )
    )

    assert candidates == [
        {
            "image_id": "a",
            "target_label": "/car",
            "positive_labels": {"/car", "/dog"},
            "negative_labels": {"/boat", "/bicycle", "/bus"},
        }
    ]


def test_builds_balanced_four_choice_examples_from_frozen_targets():
    from tiny_omni_decision.openimages_options import build_four_choice_examples

    sibling_groups = {
        "/car": ("/boat", "/bicycle", "/bus"),
        "/dog": ("/bird", "/cat", "/horse"),
    }
    names = {
        "/car": "Car",
        "/boat": "Boat",
        "/bicycle": "Bicycle",
        "/bus": "Bus",
        "/dog": "Dog",
        "/bird": "Bird",
        "/cat": "Cat",
        "/horse": "Horse",
    }
    groups = [
        {
            "image_id": "a",
            "positive_labels": {"/car"},
            "negative_labels": {"/boat", "/bicycle", "/bus"},
        },
        {
            "image_id": "b",
            "positive_labels": {"/dog"},
            "negative_labels": {"/bird", "/cat", "/horse"},
        },
    ]

    examples = build_four_choice_examples(
        groups,
        target_labels=("/car", "/dog"),
        sibling_groups=sibling_groups,
        display_names=names,
        per_target_limit=1,
        seed=17,
    )

    assert {item["target_label"] for item in examples} == {"/car", "/dog"}
    assert len({item["image_id"] for item in examples}) == len(examples) == 2
    assert all(len(item["options"]) == 4 for item in examples)
    assert all(item["options"][item["target_position"]] == item["target"] for item in examples)
