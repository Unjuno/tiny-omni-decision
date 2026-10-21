import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("av")
pytest.importorskip("transformers")

from scripts import probe_native_mask_memory as runner  # noqa: E402


def test_source_rgb_hash_uses_original_saved_256_pixel_view(monkeypatch):
    native = np.zeros((2, 3, 3), np.uint8)
    small = np.ones((256, 256, 3), np.uint8)

    class Frame:
        def to_ndarray(self, **kwargs):
            return native

        def reformat(self, **kwargs):
            assert kwargs == dict(width=256, height=256, format="rgb24")
            return type("Small", (), {"to_ndarray": lambda self: small})()

    class Container:
        streams = type(
            "Streams",
            (),
            {
                "video": [
                    type(
                        "Stream", (), {"codec_context": type("Context", (), {"thread_count": 0})()}
                    )()
                ]
            },
        )()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, **kwargs):
            return [Frame()] * 32

    monkeypatch.setattr(runner.av, "open", lambda path: Container())
    plan = dict(
        media_path="fixed",
        frame_indices=list(range(32)),
        selected_rgb_sha256=hashlib.sha256(np.stack([small] * 32).tobytes()).hexdigest(),
    )
    np.testing.assert_array_equal(runner.decode(plan), np.stack([native] * 32))


def test_reference_configuration_bytes_and_mode_are_mandatory(tmp_path):
    contract = {"seed": 17}
    config = dict(contract=contract, mode="smoke")
    (tmp_path / "config.json").write_text(json.dumps(config))
    result = dict(
        contract=contract,
        contract_sha256=runner.digest(contract),
        config_sha256=runner.sha(tmp_path / "config.json"),
        rows=[],
    )
    (tmp_path / "result.json").write_text(json.dumps(result))
    pins = runner.evidence(tmp_path, result, "smoke")
    assert str(tmp_path / "result.json") in pins and str(tmp_path / "config.json") in pins
    with pytest.raises(ValueError):
        runner.evidence(tmp_path, result, "main")
    (tmp_path / "config.json").write_text(json.dumps(dict(config, contract={"seed": 18})))
    with pytest.raises(ValueError):
        runner.evidence(tmp_path, result, "smoke")


def test_absent_reference_or_nonfinite_projection_never_admits_main(tmp_path):
    contract = {"seed": 17}
    r = dict(
        status="PASS_NATIVE_MEMORY_SMOKE_ONLY",
        contract=contract,
        projected_main_seconds=float("nan"),
        rows=[],
    )
    (tmp_path / "result.json").write_text(json.dumps(r))
    with pytest.raises(ValueError):
        runner.admit_smoke(tmp_path, contract)
    r["projected_main_seconds"] = 30
    (tmp_path / "result.json").write_text(json.dumps(r))
    with pytest.raises(ValueError):
        runner.admit_smoke(tmp_path, contract)


def test_inherited_settings_cannot_be_repinned(tmp_path):
    p = tmp_path / "processor.json"
    p.write_text("original")
    inherited = {str(p): runner.sha(p)}
    assert runner.verify_inherited(inherited) == inherited
    p.write_text("changed")
    with pytest.raises(ValueError, match="inherited"):
        runner.verify_inherited(inherited)


def test_replay_requires_complete_ordered_cohort_and_archive_schema(tmp_path):
    plans = [
        dict(video_id=v, seed_indices=list(range(n)), frame_indices=list(range(32)))
        for v, n in zip(runner.IDS, runner.COUNTS, strict=True)
    ]
    contract = dict(plans=plans)
    rows = []
    for plan, count in zip(plans, runner.COUNTS, strict=True):
        for arm in runner.ARMS:
            p = tmp_path / f"{plan['video_id']}-{arm}.npz"
            raw = np.array(["a" * 64] * count)
            np.savez_compressed(
                p,
                masks_packed=np.zeros((count, 32, 8192), np.uint8),
                boxes=np.zeros((count, 32, 4), np.float64),
                scores=np.zeros((count, 32), np.float32),
                empty_masks=np.ones((count, 32), bool),
                raw_sha256=raw,
                seed_indices=np.arange(count, dtype=np.int64),
                frame_indices=np.arange(32, dtype=np.int64),
            )
            rows.append(
                dict(
                    video_id=plan["video_id"],
                    arm=arm,
                    cache_size=32,
                    seeds=count,
                    native_object_frames=count * 32,
                    raw_prefix_sha256=raw[:2].tolist(),
                    artifact=dict(path=str(p), sha256=runner.sha(p)),
                )
            )
    prior = dict(
        status="PASS_FULL_NATIVE_MEMORY_EXECUTION_ONLY",
        contract=contract,
        contract_sha256=runner.digest(contract),
        rows=rows,
        parameters_frozen=True,
        rng_unchanged=True,
        state_before="fixed",
        state_after="fixed",
    )
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(dict(contract=contract, mode="main")))
    prior["config_sha256"] = runner.sha(cfg)

    def save():
        (tmp_path / "result.json").write_text(json.dumps(prior))

    save()
    runner.admit_main(tmp_path, contract)
    prior["rows"] = rows[:2]
    save()
    with pytest.raises(ValueError):
        runner.admit_main(tmp_path, contract)
    prior["rows"] = rows[::-1]
    save()
    with pytest.raises(ValueError):
        runner.admit_main(tmp_path, contract)
    prior["rows"] = rows
    rows[0]["raw_prefix_sha256"] = ["b" * 64] * 2
    save()
    with pytest.raises(ValueError):
        runner.admit_main(tmp_path, contract)
