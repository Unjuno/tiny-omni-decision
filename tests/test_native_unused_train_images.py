import pytest


def test_unused_image_rows_excludes_whole_seen_assets_and_keeps_question_order():
    from scripts.evaluate_native_unused_train_images import unused_image_rows

    rows = [
        dict(id="a-color", source="clevr4", group="image-a", observation="image:a"),
        dict(id="a-count", source="clevr4", group="image-a", observation="image:a"),
        dict(id="b-color", source="clevr4", group="image-b", observation="image:b"),
        dict(id="text-c", source="text", group="text-c", observation="text:c"),
        dict(id="b-count", source="clevr4", group="image-b", observation="image:b"),
    ]

    held, seen = unused_image_rows(rows, [[0, 3]], selected_step=1)

    assert seen == {"image-a"}
    assert [row["id"] for row in held] == ["b-color", "b-count"]


def test_unused_image_rows_rejects_invalid_sampler_indices():
    from scripts.evaluate_native_unused_train_images import unused_image_rows

    rows = [dict(id="a", source="clevr4", group="image-a", observation="image:a")]
    with pytest.raises(ValueError, match="sampler"):
        unused_image_rows(rows, [[4]], selected_step=1)
