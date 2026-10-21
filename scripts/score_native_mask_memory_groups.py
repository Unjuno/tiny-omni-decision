"""Posthoc fixed TRAIN source-box proxy comparison after complete inference audit."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.audit_native_mask_memory_full import ARMS, COUNTS, IDS, require_reference_bindings, sha
from tiny_omni_decision.memory_group_proxy import evaluate_group_proxy

SOURCE = Path("C:/CodexArtifacts/pretrained-reuse-r2/grid-role-corpus-all-train-01/result.json")
SOURCE_SHA = "a897195ccba8214bfc0d3f91c30d7f33d70910e8ebf4ee211fd463a967b56fd2"


def require_fresh_evidence(audit, main_folder, main):
    evidence = audit.get("validated_fresh_evidence")
    if not evidence or not evidence.get("hashes"):
        raise ValueError("mandatory fresh model replay evidence absent")
    folder = Path(evidence["folder"])
    result = folder / "result.json"
    config = folder / "config.json"
    hashes = evidence["hashes"]
    required_headers = {str(p): sha(p) for p in [result, config]}
    require_reference_bindings(required_headers, hashes)
    require_reference_bindings(hashes, audit["protected_hashes"])
    fresh = json.loads(result.read_bytes())
    cfg = json.loads(config.read_bytes())
    if (
        fresh["status"] != "PASS_FRESH_FIXED_NATIVE_MEMORY_PREFIX_ONLY"
        or cfg["mode"] != "replay"
        or fresh["contract"] != main["contract"]
        or cfg["contract"] != main["contract"]
        or fresh["config_sha256"] != required_headers[str(config)]
        or not fresh["parameters_frozen"]
        or not fresh["rng_unchanged"]
        or fresh["state_before"] != main["state_before"]
        or fresh["state_after"] != fresh["state_before"]
        or [(r["video_id"], r["arm"]) for r in fresh["rows"]] != [(IDS[0], a) for a in ARMS]
        or any(r["seeds"] != 2 or r["native_object_frames"] != 64 for r in fresh["rows"])
    ):
        raise ValueError("complete fixed fresh replay metadata required before annotations")
    required = {
        **required_headers,
        **{r["artifact"]["path"]: r["artifact"]["sha256"] for r in fresh["rows"]},
    }
    if set(hashes) != set(required):
        raise ValueError("exact complete fresh headers/archives required")
    require_reference_bindings(required, hashes)
    primary = {
        str(main_folder / "result.json"): sha(main_folder / "result.json"),
        str(main_folder / "config.json"): main["config_sha256"],
        **{r["artifact"]["path"]: r["artifact"]["sha256"] for r in main["rows"]},
    }
    require_reference_bindings(primary, audit["validated_main_evidence"])
    require_reference_bindings(primary, fresh["reference_hashes"])
    for p, h in required.items():
        if sha(p) != h:
            raise ValueError("mandatory fresh archive changed before annotations")
    return required


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new proxy-score output required")
    tick = time.perf_counter()
    audit = json.loads(args.audit.read_bytes())
    result_path = args.main / "result.json"
    r = json.loads(result_path.read_bytes())
    pins = audit["protected_hashes"].copy()
    if (
        audit["status"] != "PASS_COMPLETE_INDEPENDENT_NATIVE_MASK_AUDIT_ONLY"
        or audit["native_object_frames"] != 44544
        or r["status"] != "PASS_FULL_NATIVE_MEMORY_EXECUTION_ONLY"
        or pins.get(str(result_path)) != sha(result_path)
        or [(x["video_id"], x["arm"]) for x in r["rows"]] != [(v, a) for v in IDS for a in ARMS]
        or sha("scripts/audit_native_mask_memory_full.py") != audit["auditor_sha256"]
    ):
        raise ValueError(
            "complete inference/fresh independent admission required before annotations"
        )
    pins.update(require_fresh_evidence(audit, args.main, r))
    for row in r["rows"]:
        if pins.get(row["artifact"]["path"]) != row["artifact"]["sha256"]:
            raise ValueError("audit does not bind every actual prediction archive")
    for p, h in pins.items():
        if sha(p) != h:
            raise ValueError("admitted protected source/product changed before scoring")
    pins[str(args.audit)] = sha(args.audit)
    pins[str(SOURCE)] = SOURCE_SHA
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError("pinned existing TRAIN proxies changed")
    # First annotation access: inference plus prefix replay are already complete.
    source = json.loads(SOURCE.read_bytes())
    if source["validation_arrays_or_annotations_opened"] or source["sealed_audit_opened"]:
        raise ValueError("TRAIN-only proxy provenance required")
    source_rows = {x["video_id"]: x for x in source["rows"] if x["video_id"] in IDS}
    rows = []
    for i, vid in enumerate(IDS):
        s = source_rows[vid]
        if sha(s["path"]) != s["sha256"]:
            raise ValueError("source proxy archive changed")
        pins[s["path"]] = s["sha256"]
        with np.load(s["path"], allow_pickle=False) as b:
            group_ids = np.array(s["roles_supervision_only"]["group"], np.int64)
            indexes = [int(np.flatnonzero(b["source_ids"] == x)[0]) for x in group_ids]
            src = b["source_boxes"][:, indexes].copy()
            available = b["source_available"][:, indexes].copy()
            exact = b["source_exact"][:, indexes].copy()
            frames = b["frames"].copy()
        if (
            frames.tolist() != r["contract"]["plans"][i]["frame_indices"]
            or src.shape != (32, len(group_ids), 4)
            or available.shape != (32, len(group_ids))
        ):
            raise ValueError("source proxy time/group coverage differs")
        banks = {}
        first_packed = None
        for arm in ARMS:
            rec = r["rows"][i * 2 + ARMS.index(arm)]
            with np.load(rec["artifact"]["path"], allow_pickle=False) as b:
                banks[arm] = b["boxes"].copy()
                if len(banks[arm]) != COUNTS[i]:
                    raise ValueError("complete original seed cohort required")
                if arm == "initial_repeat" and (
                    first_packed is None
                    or first_packed.tobytes() != b["masks_packed"][:, 0].tobytes()
                ):
                    raise ValueError("whole-cohort frame0 masks differ between controls")
                if arm == "actual":
                    first_packed = b["masks_packed"][:, 0].copy()
        score = evaluate_group_proxy(
            banks["actual"], banks["initial_repeat"], src, available, group_ids
        )
        rows.append(
            dict(
                video_id=vid,
                **score,
                source_available=int(available.sum()),
                source_exact=int((available & exact).sum()),
                source_interpolated=int((available & ~exact).sum()),
                source_visibility_verified=False,
            )
        )
    if (
        sum(x["source_groups"] for x in rows) != 24
        or sum(x["anchor_counts"]["total"] for x in rows) != 696
    ):
        raise ValueError("all24sourcegroups/all696seed accounting required")
    pooled = {}
    for arm in ARMS:
        counts = {
            k: sum(row["arms"][arm]["future_categories"][k] for row in rows)
            for k in rows[0]["arms"][arm]["future_categories"]
        }
        den = sum(x["arms"][arm]["eligible_seed_future_denominator"] for x in rows)
        gden = sum(x["arms"][arm]["source_group_future_denominator"] for x in rows)
        covered = sum(x["arms"][arm]["source_group_future_covered"] for x in rows)
        pooled[arm] = dict(
            future_categories=counts,
            eligible_seed_future_denominator=den,
            eligible_seed_future_rate=counts["same_unique_group"] / den if den else None,
            source_group_future_denominator=gden,
            source_group_future_covered=covered,
            source_group_future_rate=covered / gden,
            macro_scene_source_group_future_rate=float(
                np.mean([x["arms"][arm]["source_group_future_rate"] for x in rows])
            ),
        )
    diffs = np.array(
        [
            x["arms"]["actual"]["source_group_future_rate"]
            - x["arms"]["initial_repeat"]["source_group_future_rate"]
            for x in rows
        ]
    )
    rng = np.random.default_rng(17)
    boot = diffs[rng.integers(0, 8, (10000, 8))].mean(-1)
    uncertainty = dict(
        unit="known TRAIN video",
        metric="macro scene source group future proxy coverage",
        difference_actual_minus_repeat=float(diffs.mean()),
        percentile95=np.quantile(boot, [0.025, 0.975]).tolist(),
        resamples=10000,
        seed=17,
        blind_or_population_quality=False,
    )
    for p, h in pins.items():
        if sha(p) != h:
            raise ValueError("protected inference/audit/proxy changed during scoring")
    out = dict(
        status="PASS_POSTHOC_TRAIN_BOX_PROXY_COMPARISON_ONLY",
        rows=rows,
        pooled=pooled,
        scene_paired_bootstrap=uncertainty,
        training=False,
        gpu_seconds=0,
        sealed_audit_opened=False,
        annotations_used_in_neural_inference=False,
        semantic_decision_quality=False,
        source_visibility_verified=False,
        source_boxes_amodal_sparse_or_interpolated=True,
        threshold=0.5,
        anchor_selection="posthoc oracle unique group box at frame0",
        all_unknown_ambiguous_missing_preserved=True,
        protected_hashes=pins,
        scorer_sha256=sha(__file__),
        helper_sha256=sha("src/tiny_omni_decision/memory_group_proxy.py"),
        seconds=time.perf_counter() - tick,
    )
    args.output.open("x", encoding="utf-8").write(json.dumps(out, indent=2) + "\n")
    print(out["status"], pooled, flush=True)


if __name__ == "__main__":
    main()
