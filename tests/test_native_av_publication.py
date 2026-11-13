import pytest


def test_changed_profile_or_missing_strict_verification_cannot_publish():
    from scripts.report_native_av_correspondence import admit

    primary = dict(
        status="PASS_NATIVE_AV_CORRESPONDENCE_RUNTIME",
        profile_sha256="a" * 64,
        model_updates=0,
        gpu_used=False,
        sealed_audit_opened=False,
    )
    replay = dict(primary, fresh_process_replay=True)
    strict = dict(
        status="PASS_STRICT_RAW_AV_REPLAY_PROVENANCE",
        profile_checked_before_forward=True,
        expected_profile_sha256="a" * 64,
    )
    admit(primary, replay, strict)
    with pytest.raises(ValueError):
        admit(primary, dict(replay, profile_sha256="b" * 64), strict)
    with pytest.raises(ValueError):
        admit(primary, replay, dict(strict, profile_checked_before_forward=False))
