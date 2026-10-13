import pytest

from scripts.clotho_acquisition_gate import validate_records


def fixture():
    wanted = {"1": dict(source_id="1", file_name="one.wav", uploader="u",
                        license="allowed", split="train")}
    record = dict(source_id="1", member="audio_files/one.wav", uploader="u",
                  license="allowed", split="train", sha256="a" * 64)
    return wanted, dict(status="PASS", files=[record])


def test_exact_coverage():
    wanted, report = fixture()
    assert validate_records(report, wanted) == report["files"]


@pytest.mark.parametrize("mutation", ["running", "missing", "duplicate", "split", "member"])
def test_invalid_report_rejected(mutation):
    wanted, report = fixture()
    if mutation == "running":
        report["status"] = "RUNNING"
    elif mutation == "missing":
        report["files"] = []
    elif mutation == "duplicate":
        report["files"] *= 2
    elif mutation == "split":
        report["files"][0]["split"] = "validation"
    else:
        report["files"][0]["member"] = "audio_files/wrong.wav"
    with pytest.raises(ValueError):
        validate_records(report, wanted)


def test_same_media_cannot_cross_splits():
    wanted, report = fixture()
    wanted["2"] = dict(wanted["1"], source_id="2", file_name="two.wav", split="validation")
    report["files"].append(dict(report["files"][0], source_id="2",
                                 member="audio_files/two.wav", split="validation"))
    with pytest.raises(ValueError, match="cross splits"):
        validate_records(report, wanted)
