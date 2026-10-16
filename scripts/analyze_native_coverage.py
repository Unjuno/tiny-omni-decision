"""Audit the preregistered native coverage comparison from saved development scores."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.analyze_native_temporal_fit import bootstrap, per_example
from scripts.native_consumed_evaluation import common_task_gap
from scripts.native_fit import accounting, choose_best, learning_rate, summarize
from scripts.native_state_dependence import summarize as summarize_pairs

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
OUT = ROOT / "native-coverage-comparison-01"


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
    train_by_id = {r["id"]: r for r in train}
    union = load(ROOT / "native-coverage-plan-02/union-train-ids.json")
    union_rows = [train_by_id[i] for i in union["union_ids"]]
    state_plan = load(ROOT / "native-coverage-plan-02/parent-state-pairs.json")
    runs, curves, receipts, resources, vectors, probe_overlaps = {}, {}, {}, {}, {}, {}
    for policy in ("repeat", "fresh"):
        fit = ROOT / f"native-coverage-{policy}-fit-01"
        replay = ROOT / f"native-coverage-{policy}-replay-01"
        smoke = ROOT / f"native-coverage-{policy}-smoke-01"
        receipt, replay_receipt = load(fit / "result.json"), load(replay / "result.json")
        assert receipt["status"] == "PASS_NATIVE_COVERAGE_TRAIN"
        assert replay_receipt["status"] == "PASS_NATIVE_COVERAGE_REPLAY"
        assert receipt["step"] == replay_receipt["step"] == 2048
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
                    "peak_vram_reserved_bytes",
                )
            }
            resources[folder.name]["receipt_sha256"] = sha(folder / "result.json")
        assert load(fit / "validation-ids.json") == [
            dict(id=r["id"], option_ids=r["option_ids"], target_index=r["target_index"])
            for r in rows
        ]
        assert sha(fit / "effective-config.json") == receipt["config_sha256"]
        sampling = load(fit / "sampling.json")
        frozen_pool = load(ROOT / f"native-coverage-plan-02/{policy}-pool.json")
        frozen_plan = load(ROOT / f"native-coverage-plan-02/{policy}-plan.json")
        assert sampling["coverage_pool"] == frozen_pool
        assert sampling["occurrence_plan"] == frozen_plan
        pool, occurrences = frozen_pool["pool"], frozen_plan["occurrences"]
        order = [[pool[o] for o in b] for b in occurrences]
        assert order == sampling["order"]
        history = [json.loads(s) for s in (fit / "training-history.jsonl").read_text().splitlines()]
        assert len(history) == 2048
        for step, h in enumerate(history, 1):
            assert h["step"] == step and h["lr"] == learning_rate(step)
            assert h["occurrence_ids"] == occurrences[step - 1]
            assert h["sample_ids"] == [train[i]["id"] for i in order[step - 1]]
            assert np.isfinite([h["ce"], h["gradient_norm_before_clip"]]).all()
        actual = Counter(i for h in history for i in h["sample_ids"])
        assert actual == Counter(train[i]["id"] for i in pool)
        label_counts = {}
        for i in pool:
            row = train[i]
            key = row["source"] + ":" + row["task"]
            label_counts.setdefault(key, Counter())[row["options"][row["target_index"]]] += 1
        unique_counts = {
            key: dict(Counter(train_by_id[i][key] for i in actual))
            for key in ("modality", "source")
        }
        assert receipt["actual_accounting"] == accounting([train[i] for b in order for i in b])
        curve = load(fit / "learning-curve.json")
        assert [h["step"] for h in curve] == list(range(0, 2049, 256))
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
            assert point["accounting"] == accounting(
                [train[i] for b in order[: point["step"]] for i in b]
            )
        assert best_step == receipt["best_step"]
        assert receipt["selected_accounting"] == accounting(
            [train[i] for b in order[:best_step] for i in b]
        )
        state = load(fit / "endpoint-state-metrics.json")
        state_scores = np.load(fit / "endpoint-state-pairs.npy", allow_pickle=False)
        np.testing.assert_array_equal(state_scores, np.load(replay / "endpoint-state-pairs.npy"))
        assert state == summarize_pairs(state_plan, state_scores)
        membership = load(fit / "endpoint-state-membership.json")
        assert membership == [
            dict(
                parent_stratum=p["stratum"],
                row_ids=p["row_ids"],
                consumed_current=[i in actual for i in p["row_ids"]],
            )
            for p in state_plan["pairs"]
        ]
        train_scores = np.load(fit / "endpoint-train-union.npy", allow_pickle=False)
        np.testing.assert_array_equal(train_scores, np.load(replay / "endpoint-train-union.npy"))
        train_metrics = load(fit / "endpoint-train-metrics.json")
        assert train_metrics["union"] == summarize(union_rows, train_scores)
        for label, ids in (
            ("parent", set(union["parent_ids"])),
            ("additional", set(union["additional_ids"])),
            ("consumed", set(actual)),
        ):
            indices = [i for i, r in enumerate(union_rows) if r["id"] in ids]
            assert train_metrics[label] == summarize(
                [union_rows[i] for i in indices], train_scores[indices]
            )
        consumed_indices = [i for i, r in enumerate(union_rows) if r["id"] in actual]
        final_scores = np.load(fit / "validation-2048.npy", allow_pickle=False)
        gap = common_task_gap(
            [union_rows[i] for i in consumed_indices],
            train_scores[consumed_indices],
            rows,
            final_scores,
        )
        event_indices = [i for i, r in enumerate(rows) if r["source"] == "starss"]
        events = dict(
            count=len(event_indices),
            predicted_positive=int((final_scores[event_indices].argmax(1) == 1).sum()),
            true_positive=int(sum(rows[i]["target_index"] == 1 for i in event_indices)),
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
            endpoint_train=train_metrics,
            common_task_gap=gap,
            endpoint_event_predictions=events,
            unique_counts=unique_counts,
            correct_candidate_counts_by_task=label_counts,
            state_pair_exposure=dict(
                Counter("/".join(str(int(v)) for v in p["consumed_current"]) for p in membership)
            ),
        )
        receipts[policy], curves[policy] = receipt, curve
        vectors[policy] = {
            label: per_example(rows, np.load(fit / f"validation-{step:04}.npy"))
            for label, step in (("final", 2048), ("selected", best_step))
        }
        probe_ids = load(fit / "train-probe-ids.json")
        probe_overlaps[policy] = dict(
            total=len(probe_ids),
            consumed=len(set(probe_ids) & set(actual)),
            unconsumed=len(set(probe_ids) - set(actual)),
        )
    assert receipts["repeat"]["validated_hashes"] == receipts["fresh"]["validated_hashes"]
    for name in (
        "initial.pt",
        "effective-config.json",
        "validation-ids.json",
        "train-probe-ids.json",
    ):
        assert receipts["repeat"]["artifacts"][name] == receipts["fresh"]["artifacts"][name], name
    for name in ("repeat", "fresh"):
        np.testing.assert_array_equal(
            np.load(ROOT / "native-common-baseline-fit-01/validation-0000.npy"),
            np.load(ROOT / f"native-coverage-{name}-fit-01/validation-0000.npy"),
        )
    first, second = runs["repeat"], runs["fresh"]
    for name in ("modalities", "sources", "tasks"):
        assert first["actual_accounting"][name] == second["actual_accounting"][name]
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
        macro_accuracy_higher=second["final_metrics"]["macro"]["accuracy"]
        > first["final_metrics"]["macro"]["accuracy"],
        no_material_modality_regression=all(
            d["accuracy"] >= -0.02 and d["nll"] <= 0.05 for d in delta.values()
        ),
    )
    uncertainty = {
        label: bootstrap(rows, vectors["repeat"][label], vectors["fresh"][label])
        for label in ("final", "selected")
    }
    uncertainty = json.loads(
        json.dumps(uncertainty).replace("lstm_minus_baseline", "fresh_minus_repeat")
    )
    report = dict(
        status="PASS_COVERAGE_COMPARISON_AUDIT",
        analysis_sha256=sha(__file__),
        runs=runs,
        curves=curves,
        resources=resources,
        probe_overlap=probe_overlaps,
        endpoint_modality_delta=delta,
        research_rule=rule,
        supports_registered_useful_coverage_hypothesis=all(rule.values()),
        video_uncertainty=uncertainty,
        total_process_seconds=sum(v["seconds"] for v in resources.values()),
        cloud_cost=0,
        limitations=[
            "One seed and observed development validation, not a blind audit.",
            "Exact modality/source/task/option-count quotas, not label or difficulty matching.",
            "No unused Video questions; both arms repeat the same512 eight times.",
            "Parent membership differs from current consumption; per-side exposure is recorded.",
            "TRAIN unions/common tasks are descriptive, not independent test estimates.",
            "Readout timing excludes encoder extraction. No promotion or capacity ceiling claim.",
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
                    "probe_overlap",
                    "total_process_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
