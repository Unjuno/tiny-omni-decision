from pathlib import Path


def test_budget_runner_only_changes_declared_fields():
    root = Path(__file__).resolve().parents[1] / "scripts"
    original = (root / "train_vidimu_fusion_arm.py").read_text().strip()
    expected = original.replace(
        "Fixed gated fusion experiment; no search or legacy artifact changes.",
        "Fixed gated fusion budget experiment; legacy run stays immutable.")
    expected = expected.replace('"vidimu-gated-fusion-arm-01"',
                                '"vidimu-gated-fusion-budget-01"')
    expected = expected.replace('"epochs": 100,', '"epochs": 1000,')
    expected = expected.replace('range(101)', 'range(1001)')
    assert (root / "train_vidimu_gated_budget.py").read_text().strip() == expected
