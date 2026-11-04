from scripts.build_conditional_curriculum import build, verify
from scripts.build_vocabulary_transfer import VARIANTS, render


def test_vocabulary_transfer_pairing_and_unchanged_retention_tasks():
    for row in build(167, 2, "development"):
        for variant in VARIANTS:
            actual = render(row, variant)
            verify(actual)
            assert actual["options"] == row["options"]
            assert actual["target"] == row["target"]
            assert actual == render(row, variant)
        if row["task"] in ("count", "conditional_selection"):
            assert len({render(row, f"base-{v}")["question"] for v in range(5)}) == 1
        if row["task"] == "extremum" and row["oracle"]["field"] == "cost":
            expected = "cheapest" if row["oracle"]["direction"] == "minimum" else "priciest"
            assert expected in render(row, "base-4")["question"]


def test_domain_family_changes_only_question_within_numeric_regime():
    for row in build(167, 2, "development"):
        for regime in ("base", "expanded"):
            baseline = render(row, regime + "-0")
            for family in range(1, 5):
                actual = render(row, f"{regime}-{family}")
                for key in ("state", "options", "target", "oracle", "task", "group"):
                    assert actual[key] == baseline[key]
            domain = render(row, regime + "-4")
            if row["task"] in ("extremum", "temporal_order"):
                assert domain["question"].startswith(
                    "Use cost as price, risk as danger, and time as timestamp. "
                )
            else:
                assert domain["question"] == baseline["question"]
