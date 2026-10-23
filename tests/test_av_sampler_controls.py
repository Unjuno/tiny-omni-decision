import pytest

torch = pytest.importorskip("torch")

from scripts.audit_av_sampler_controls import apply_control  # noqa: E402


def test_controls_copy_only_requested_stream_and_preserve_inputs():
    values = {k: torch.arange(6).reshape(3, 2).float() for k in ["audio", "video", "text"]}
    values["targets"] = torch.tensor([0, 1, 0])
    before = {k: v.clone() for k, v in values.items()}
    donors = torch.tensor([2, 0, 1])
    for name, kind in [("zero_audio", "audio"), ("zero_video", "video"),
                       ("shuffle_video", "video"), ("shuffle_question", "text")]:
        changed = apply_control(values, name, donors)
        expected = (torch.zeros_like(values[kind]) if name.startswith("zero")
                    else values[kind][donors])
        torch.testing.assert_close(changed[kind], expected)
        for key in values:
            torch.testing.assert_close(values[key], before[key])
            if key != kind:
                assert changed[key] is values[key]
    with pytest.raises(ValueError):
        apply_control(values, "unknown", donors)
