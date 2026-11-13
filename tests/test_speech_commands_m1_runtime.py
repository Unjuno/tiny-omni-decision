from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_m1_runtime_rejects_unknown_readout_hash_before_loading(tmp_path) -> None:
    from tiny_omni_decision.speech_commands_m1_runtime import SpeechCommandsM1Runtime

    class AudioRuntime:
        pass

    bad_readout = tmp_path / "bad-readout.pt"
    bad_readout.write_bytes(b"not-the-frozen-readout")
    with pytest.raises(ValueError, match="readout hash mismatch"):
        SpeechCommandsM1Runtime(AudioRuntime(), bad_readout)
