"""Audit both fixed batch experiments from saved evidence; no model or audit load."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.analyze_native_temporal_fit import bootstrap, per_example
from scripts.native_fit import accounting, choose_best, option_order, summarize
from scripts.native_matched_batches import learning_rate

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
OUT = ROOT / "native-batch-comparison-01"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    rows = [
        json.loads(s)
        for s in (ROOT / "native-common-corpus-02/validation.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    train = [
        json.loads(s)
        for s in (ROOT / "native-common-corpus-02/train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    parent_sampling = load(ROOT / "native-common-baseline-fit-01/sampling.json")
    pool = [i for b in parent_sampling["order"] for i in b]
    expected_permutations = load(ROOT / "native-matched-plan-01/permutations.json")
    assert [option_order(train[i], o) for o, i in enumerate(pool)] == expected_permutations
    original_counts = Counter(train[i]["id"] for i in pool)
    runs, curves, vectors, receipts, resources = {}, {}, {}, {}, {}
    probe_overlaps = {}
    for policy in ("independent", "matched"):
        fit = ROOT / f"native-batch-{policy}-fit-01"
        replay = ROOT / f"native-batch-{policy}-replay-01"
        smoke = ROOT / f"native-batch-{policy}-smoke-01"
        receipt, replay_receipt = load(fit / "result.json"), load(replay / "result.json")
        assert receipt["status"] == "PASS_NATIVE_MATCHED_BATCH_TRAIN"
        assert replay_receipt["status"] == "PASS_NATIVE_MATCHED_BATCH_REPLAY"
        assert receipt["step"] == replay_receipt["step"] == 1024
        assert replay_receipt["reference_sha256"] == sha(fit / "result.json")
        for folder, result in (
            (fit, receipt),
            (replay, replay_receipt),
            (smoke, load(smoke / "result.json")),
        ):
            for name, digest in result["artifacts"].items():
                assert sha(folder / name) == digest, (folder, name)
            resources[folder.name] = {
                k: result[k]
                for k in (
                    "seconds",
                    "training_seconds",
                    "evaluation_seconds",
                    "peak_vram_bytes",
                    "peak_working_set_bytes",
                )
            }
            resources[folder.name]["receipt_sha256"] = sha(folder / "result.json")
        assert load(fit / "validation-ids.json") == [
            dict(id=r["id"], option_ids=r["option_ids"], target_index=r["target_index"])
            for r in rows
        ]
        assert sha(fit / "effective-config.json") == receipt["config_sha256"]
        sampling = load(fit / "sampling.json")
        expected_plan = load(ROOT / f"native-matched-plan-01/{policy}.json")
        assert sampling["occurrence_plan"] == expected_plan
        occurrences = expected_plan["occurrences"]
        order = [[pool[o] for o in b] for b in occurrences]
        assert order == sampling["order"]
        history = [json.loads(s) for s in (fit / "training-history.jsonl").read_text().splitlines()]
        assert len(history) == 1024
        for step, h in enumerate(history, 1):
            assert h["step"] == step and h["lr"] == learning_rate(step)
            assert h["occurrence_ids"] == occurrences[step - 1]
            assert h["sample_ids"] == [train[i]["id"] for i in order[step - 1]]
            assert np.isfinite([h["ce"], h["gradient_norm_before_clip"]]).all()
        assert Counter(i for h in history for i in h["sample_ids"]) == original_counts
        assert receipt["actual_accounting"] == accounting([train[i] for b in order for i in b])
        curve = load(fit / "learning-curve.json")
        assert [h["step"] for h in curve] == list(range(0, 1025, 128))
        best, best_step = None, None
        for point in curve:
            scores = np.load(fit / f"validation-{point['step']:04}.npy", allow_pickle=False)
            calculated = summarize(rows, scores)
            assert calculated == point["validation"]
            take = choose_best(best, calculated["macro"]["nll"])
            assert take == point["selected"]
            if take:
                best, best_step = calculated["macro"]["nll"], point["step"]
            assert point["best_step"] == best_step
        assert best_step == receipt["best_step"]
        assert receipt["selected_accounting"] == accounting(
            [train[i] for b in order[:best_step] for i in b]
        )
        state = load(fit / "endpoint-state-metrics.json")
        np.testing.assert_array_equal(
            np.load(fit / "endpoint-state-pairs.npy"), np.load(replay / "endpoint-state-pairs.npy")
        )
        runs[policy] = {
            k: receipt[k]
            for k in (
                "source_commit",
                "config_sha256",
                "command",
                "started_utc",
                "finished_utc",
                "step",
                "best_step",
                "best_score",
                "actual_accounting",
                "selected_accounting",
                "selected_metrics",
                "final_metrics",
                "total_parameters",
                "trainable_parameters",
                "versions",
                "python",
                "cuda",
                "device",
                "dtype",
                "tf32",
            )
        }
        runs[policy].update(
            path=str(fit),
            receipt_sha256=sha(fit / "result.json"),
            state_contrasts=state,
            checkpoint_sha256={k: sha(fit / k) for k in ("initial.pt", "best.pt", "final.pt")},
        )
        receipts[policy], curves[policy] = receipt, curve
        vectors[policy] = {
            label: per_example(rows, np.load(fit / f"validation-{step:04}.npy"))
            for label, step in (("final", 1024), ("selected", best_step))
        }
        probe_ids = load(fit / "train-probe-ids.json")
        probe_overlaps[policy] = dict(
            total=len(probe_ids),
            consumed=len(set(probe_ids) & set(original_counts)),
            unconsumed=len(set(probe_ids) - set(original_counts)),
        )
    assert receipts["independent"]["validated_hashes"] == receipts["matched"]["validated_hashes"]
    for name in (
        "initial.pt",
        "effective-config.json",
        "validation-ids.json",
        "train-probe-ids.json",
    ):
        assert (
            receipts["independent"]["artifacts"][name] == receipts["matched"]["artifacts"][name]
        ), name
    assert receipts["independent"]["actual_accounting"] == receipts["matched"]["actual_accounting"]
    np.testing.assert_array_equal(
        np.load(ROOT / "native-batch-independent-fit-01/validation-0000.npy"),
        np.load(ROOT / "native-batch-matched-fit-01/validation-0000.npy"),
    )
    np.testing.assert_array_equal(
        np.load(ROOT / "native-common-baseline-fit-01/validation-0000.npy"),
        np.load(ROOT / "native-batch-matched-fit-01/validation-0000.npy"),
    )
    first, second = runs["independent"], runs["matched"]
    delta = {
        m: {
            k: second["final_metrics"]["modality_macro"][m][k]
            - first["final_metrics"]["modality_macro"][m][k]
            for k in ("accuracy", "nll", "brier", "ece")
        }
        for m in ("text", "image", "audio", "video")
    }
    rule = dict(
        macro_nll_lower=second["final_metrics"]["macro"]["nll"]
        < first["final_metrics"]["macro"]["nll"],
        consumed_text_both_correct_improves=second["state_contrasts"]["modalities"][
            "consumed:text"
        ]["both_correct"]
        > first["state_contrasts"]["modalities"]["consumed:text"]["both_correct"],
        consumed_video_both_correct_improves=second["state_contrasts"]["modalities"][
            "consumed:video"
        ]["both_correct"]
        > first["state_contrasts"]["modalities"]["consumed:video"]["both_correct"],
        no_material_modality_regression=all(
            d["accuracy"] >= -0.02 and d["nll"] <= 0.05 for d in delta.values()
        ),
    )
    uncertainty = {
        label: bootstrap(rows, vectors["independent"][label], vectors["matched"][label])
        for label in ("final", "selected")
    }
    uncertainty = json.loads(
        json.dumps(uncertainty).replace("lstm_minus_baseline", "matched_minus_independent")
    )
    report = dict(
        status="PASS_BATCH_COMPARISON_AUDIT",
        analysis_sha256=sha(__file__),
        runs=runs,
        curves=curves,
        resources=resources,
        probe_overlap=probe_overlaps,
        same_multiset_initialization_candidates_config_validation=True,
        endpoint_modality_delta=delta,
        research_rule=rule,
        supports_registered_useful_batching_hypothesis=all(rule.values()),
        video_uncertainty=uncertainty,
        total_process_seconds=sum(v["seconds"] for v in resources.values()),
        cloud_cost=0,
        limitations=[
            "One seed and development validation; not a blind audit.",
            "Grouping and sequence change together; additive CE is unchanged.",
            "TRAIN contrast pairs are selected for construction, not representative quality.",
            "Readout timings exclude frozen encoder extraction.",
            "No product promotion or capacity ceiling claim.",
        ],
    )
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "endpoint_modality_delta",
                    "research_rule",
                    "video_uncertainty",
                    "total_process_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
