"""Publish bounded feasibility evidence, including failed scale admission."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    records = {}
    for name in [
        "native-mask-memory-smoke-01",
        "native-mask-memory-smoke-02",
        "native-mask-memory-scale-admission-01",
    ]:
        p = root / name / "result.json"
        records[name] = dict(
            path=str(p),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            result=json.loads(p.read_bytes()),
        )
    p = root / "native-mask-memory-independent-smoke-audit-01.json"
    audit = json.loads(p.read_bytes())
    r = records["native-mask-memory-smoke-02"]["result"]
    if (
        r["status"] != "PASS_NATIVE_MEMORY_SMOKE_ONLY"
        or r["projected_main_seconds"] < 7200
        or audit["status"] != "PASS_INDEPENDENT_RAW_SMOKE_AUDIT_ONLY"
        or audit["scale_gate_passed"]
        or records["native-mask-memory-scale-admission-01"]["result"]["status"] != "FAIL_INCOMPLETE"
    ):
        raise ValueError("not the recorded failed-scale / admitted-smoke experiment")
    report = dict(
        status="PASS_SMOKE_ONLY_SCALE_NOT_ADMITTED",
        product_adopted=False,
        full_cohort_inferred=False,
        fresh_model_replay_performed=False,
        semantic_quality_evaluated=False,
        annotations_used_in_inference=False,
        sealed_audit_opened=False,
        training=False,
        cloud_cost=0,
        source_commit=r["contract"]["source_commit"],
        records=records,
        independent_audit=dict(
            path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), result=audit
        ),
        scope=dict(
            unique_known_train_videos=1,
            unique_nms_seeds=2,
            variants=3,
            predictions=192,
            planned_videos=8,
            planned_seeds=696,
        ),
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        auditor_sha256=hashlib.sha256(
            Path("scripts/audit_native_mask_memory_smoke.py").read_bytes()
        ).hexdigest(),
    )
    target = Path("docs/pretrained_reuse/R2_NATIVE_MASK_MEMORY_SMOKE_RESULTS_20261014.json")
    target.write_bytes((json.dumps(report, indent=2) + "\n").encode())
    print(
        "published bytes",
        target.stat().st_size,
        "sha256",
        hashlib.sha256(target.read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    main()
