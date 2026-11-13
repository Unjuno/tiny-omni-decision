from scripts.publish_native_full_extent import comparison_row


def test_comparison_table_has_one_prefix_and_exactly_seven_columns():
    delta = {
        "macro": {"accuracy": 0.01234, "nll": 0.0456},
        "modalities": {
            key: {"accuracy": value}
            for key, value in zip(
                ("text", "image", "audio", "video"), (0.01, 0.02, -0.03, 0.04), strict=True
            )
        },
    }
    row = comparison_row("selected", delta)
    assert row == "| selected | 1.234 | 0.045600 | 1.000 | 2.000 | -3.000 | 4.000 |"
    assert len(row.split("|")) == 9
