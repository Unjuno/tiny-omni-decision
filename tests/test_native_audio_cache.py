import numpy as np
import pytest


def module():
    from scripts import native_audio_cache

    return native_audio_cache


def fixture():
    row = dict(split="train", speech_tokens=2, speech_offset=1, event_offset=1)
    banks = {
        "train-speech.npy": np.zeros((5, 384), np.float32),
        "train-event.npy": np.zeros((3, 384), np.float32),
    }
    values = dict(speech=np.ones((2, 384), np.float32), event=np.full((1, 384), 2.0, np.float32))
    return banks, row, values


def test_cache_slices_and_readonly_replay():
    banks, row, values = fixture()
    assert module().store_features(banks, row, values, "cache") == 0
    assert np.array_equal(banks["train-speech.npy"][1:3], values["speech"])
    assert not banks["train-speech.npy"][0].any()
    assert not banks["train-speech.npy"][3:].any()
    for bank in banks.values():
        bank.flags.writeable = False
    assert module().store_features(banks, row, values, "replay") == 0


def test_replay_rejects_one_changed_float():
    banks, row, values = fixture()
    module().store_features(banks, row, values, "cache")
    banks["train-speech.npy"][1, 0] = np.nextafter(np.float32(1), np.float32(2))
    with pytest.raises(ValueError):
        module().store_features(banks, row, values, "replay")


@pytest.mark.parametrize(
    "change", ["missing", "dtype", "width", "nonfinite", "offset", "phase", "reserved"]
)
def test_storage_fails_closed(change):
    banks, row, values = fixture()
    phase = "cache"
    if change == "missing":
        del values["event"]
    elif change == "dtype":
        values["speech"] = values["speech"].astype(np.float64)
    elif change == "width":
        values["speech"] = np.zeros((2, 383), np.float32)
    elif change == "nonfinite":
        values["speech"][0, 0] = float("nan")
    elif change == "offset":
        row["speech_offset"] = 4
    elif change == "phase":
        phase = "resume"
    elif change == "reserved":
        row["split"] = "calibration"
    with pytest.raises(ValueError):
        module().store_features(banks, row, values, phase)


@pytest.mark.parametrize(
    "change", [None, "dtype", "max_examples", "batch_size", "disk_reserve_bytes"]
)
def test_fixed_profile(change):
    config = dict(
        seed=17,
        batch_size=1,
        dtype="float32",
        attention="eager",
        max_examples=7410,
        expected_tensor_bytes=1908900864,
        max_wall_seconds=3600,
        max_vram_bytes=3 * 2**30,
        max_rss_bytes=6 * 2**30,
        disk_reserve_bytes=2 * 2**30,
    )
    if change:
        config[change] = "float16" if change == "dtype" else 2
        with pytest.raises(ValueError):
            module().validate_profile(config)
    else:
        module().validate_profile(config)
