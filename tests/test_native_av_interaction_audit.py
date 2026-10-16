import subprocess
import sys

import pytest


def test_av_presence_is_not_synchronization_or_supervision():
    from scripts.audit_native_av_interaction import observation_counts

    def obs(split, *kinds):
        return {"split": split, "features": [{"kind": k} for k in kinds]}

    result = observation_counts({
        "a": obs("train", "speech", "event_audio"),
        "v": obs("validation", "video"),
        "av": obs("train", "speech", "video"),
    })
    assert result["train"]["observations"] == 2
    assert result["train"]["audio_video_present"] == 1
    assert result["validation"]["audio_video_present"] == 0
    assert result["joint_training_admitted"] is False


@pytest.mark.parametrize("value", [
    {"x": {"split": "sealed_audit", "features": [{"kind": "video"}]}},
    {"x": {"split": "train", "features": []}},
    {"x": {"split": "train", "features": [{"kind": "bogus"}]}},
])
def test_unsupported_metadata_rejected(value):
    from scripts.audit_native_av_interaction import observation_counts

    with pytest.raises(ValueError):
        observation_counts(value)


def test_metadata_import_does_not_require_torch():
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; sys.modules['torch'] = None; "
        "from scripts.audit_native_av_interaction import observation_counts; "
        "assert observation_counts({})['joint_training_admitted'] is False",
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
