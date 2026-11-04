import json
import sys

import pytest

from scripts import score_native_mask_memory_groups as scorer


def test_partial_inference_rejects_before_any_annotation_access(tmp_path, monkeypatch):
    main = tmp_path / "main"
    main.mkdir()
    (main / "result.json").write_text(json.dumps(dict(status="PREFLIGHT", rows=[])))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(dict(status="INCOMPLETE", protected_hashes={})))

    class Forbidden:
        def read_bytes(self):
            raise AssertionError("source annotations opened before complete inference")

    monkeypatch.setattr(scorer, "SOURCE", Forbidden())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--main",
            str(main),
            "--audit",
            str(audit),
            "--output",
            str(tmp_path / "score.json"),
        ],
    )
    with pytest.raises(ValueError, match="complete inference"):
        scorer.main()
    assert not (tmp_path / "score.json").exists()


def test_missing_fresh_evidence_cannot_pass_annotation_barrier(tmp_path):
    with pytest.raises(ValueError, match="fresh"):
        scorer.require_fresh_evidence(dict(protected_hashes={}), tmp_path, {})


def test_required_fresh_files_and_main_reference_binding(tmp_path):
    main = tmp_path / "main"
    fresh = tmp_path / "fresh"
    main.mkdir()
    fresh.mkdir()
    contract = {"fixed": True}
    cfg = main / "config.json"
    cfg.write_text(json.dumps(dict(contract=contract, mode="main")))
    archive = main / "all.npz"
    archive.write_bytes(b"complete-main")
    r = dict(
        contract=contract,
        state_before="fixed",
        config_sha256=scorer.sha(cfg),
        rows=[dict(artifact=dict(path=str(archive), sha256=scorer.sha(archive)))],
    )
    rp = main / "result.json"
    rp.write_text(json.dumps(r))
    primary = {
        str(rp): scorer.sha(rp),
        str(cfg): scorer.sha(cfg),
        str(archive): scorer.sha(archive),
    }
    fcfg = fresh / "config.json"
    fcfg.write_text(json.dumps(dict(contract=contract, mode="replay")))
    rows = []
    for arm in scorer.ARMS:
        p = fresh / f"{arm}.npz"
        p.write_bytes(arm.encode())
        rows.append(
            dict(
                video_id=scorer.IDS[0],
                arm=arm,
                seeds=2,
                native_object_frames=64,
                artifact=dict(path=str(p), sha256=scorer.sha(p)),
            )
        )
    f = dict(
        status="PASS_FRESH_FIXED_NATIVE_MEMORY_PREFIX_ONLY",
        contract=contract,
        rows=rows,
        parameters_frozen=True,
        rng_unchanged=True,
        state_before="fixed",
        state_after="fixed",
        config_sha256=scorer.sha(fcfg),
        reference_hashes=primary,
    )
    fp = fresh / "result.json"

    def save():
        fp.write_text(json.dumps(f))
        bindings = {
            str(fp): scorer.sha(fp),
            str(fcfg): scorer.sha(fcfg),
            **{v["artifact"]["path"]: v["artifact"]["sha256"] for v in rows},
        }
        return dict(
            validated_main_evidence=primary,
            validated_fresh_evidence=dict(folder=str(fresh), hashes=bindings),
            protected_hashes={**primary, **bindings},
        )

    audit = save()
    assert len(scorer.require_fresh_evidence(audit, main, r)) == 4
    audit["protected_hashes"].pop(rows[1]["artifact"]["path"])
    with pytest.raises(ValueError):
        scorer.require_fresh_evidence(audit, main, r)
    f["reference_hashes"] = {str(rp): scorer.sha(rp)}
    with pytest.raises(ValueError):
        scorer.require_fresh_evidence(save(), main, r)
