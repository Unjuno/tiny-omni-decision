"""Inference-only TRAIN replay or fixed transfer after a completed frozen comparison."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.condition_relevance_corpus import load_development
from scripts.condition_relevance_inference import evaluation_groups, score_rows
from scripts.condition_relevance_postfit_io import (
    DEVELOPMENT,
    _json,
    _no_redirect,
    _outside_protected,
    _verified_bytes,
    _verify_stream,
    read_frozen_transfer,
    verify_execution_source,
    verify_pair_files,
)
from scripts.condition_relevance_training import forward_with_relevance, load_trainable, row_key
from scripts.condition_relevance_transfer import transfer_metrics, validate_transfer_rows
from scripts.decision_semantic_training import render_pairs, trainable_names
from scripts.decision_supervision import digest, state_key
from scripts.semantic_state_diagnostic import log_probabilities


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with _no_redirect(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(comparison, comparison_sha, mode, output, smoke_reference=None, smoke_sha=None):
    comparison = _no_redirect(comparison)
    receipt = _json(comparison / "result.json", comparison_sha)
    if receipt.get("status") != "PASS_FROZEN_CONDITION_RELEVANCE_COMPARISON":
        raise ValueError("completed frozen comparison required")
    if any(
        receipt.get(k) is not False
        for k in ("transfer_loaded", "calibration_loaded", "sealed_audit_loaded", "model_loaded")
    ):
        raise ValueError("original pre-evaluation comparison receipt required")
    if mode not in ("train-smoke", "transfer"):
        raise ValueError("registered inference mode required")
    if mode == "train-smoke" and (smoke_reference or smoke_sha):
        raise ValueError("TRAIN replay does not accept transfer/other run overrides")
    origins = verify_execution_source(
        receipt["source_root"], __file__, "evaluate_condition_relevance_transfer",
        receipt["code_pins"],
    )
    checked = verify_pair_files(
        receipt["run_directories"], receipt["result_hashes"], receipt["source_root"]
    )
    freeze_sha = receipt["artifacts"]["comparison-freeze.json"]
    frozen = _json(comparison / "comparison-freeze.json", freeze_sha)
    if frozen != checked["frozen"]:
        raise ValueError("frozen checkpoint roles changed")
    analysis = _json(
        comparison / "validation-comparison.json",
        receipt["artifacts"]["validation-comparison.json"],
    )
    if set(analysis["fit_evidence"]) != {"ce", "relevance"} or any(
        v["updates"] != 1024 or not v["selected_reload_artifacts_exact"]
        for v in analysis["fit_evidence"].values()
    ):
        raise ValueError("both completed timeline and selector audits required")
    groups = evaluation_groups(frozen["roles"])
    if mode == "transfer":
        if not smoke_reference or not smoke_sha:
            raise ValueError("successful pinned TRAIN replay required before transfer")
        smoke_reference = _no_redirect(smoke_reference)
        smoke = _json(smoke_reference / "result.json", smoke_sha)
        if (
            smoke.get("status") != "PASS_CONDITION_RELEVANCE_TRAIN_REPLAY"
            or smoke.get("comparison_sha256") != comparison_sha
            or smoke.get("frozen_sha256") != freeze_sha
            or smoke.get("transfer_loaded") is not False
            or smoke.get("exact_replay_groups") != [g["sha256"] for g in groups]
        ):
            raise ValueError("same frozen roles must pass exact TRAIN replay")
        for name, expected in smoke["artifacts"].items():
            if Path(name).name != name:
                raise ValueError("invalid smoke artifact path")
            _verify_stream(smoke_reference / name, expected)
    config = checked["runs"]["ce"]["config"]
    output = _outside_protected(checked, output)
    if output.is_relative_to(comparison) or (
        smoke_reference and output.is_relative_to(smoke_reference)
    ):
        raise ValueError("new output must not be inside a comparison or smoke artifact")
    output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    result = dict(
        status="RUNNING",
        mode=mode,
        pid=os.getpid(),
        command=sys.argv,
        started_utc=datetime.now(UTC).isoformat(),
        comparison_sha256=comparison_sha,
        frozen_sha256=freeze_sha,
        sealed_audit_loaded=False,
        calibration_loaded=False,
        transfer_loaded=False,
        optimizer_updates=0,
        product_complete=False,
        cloud_cost=0,
        module_origins=origins,
        evaluation_source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=receipt["source_root"], text=True
        ).strip(),
    )
    write(output / "result.json", result)
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import psutil
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from scripts.train_text_crossencoder import frozen_hash

        result["module_origins"] = verify_execution_source(
            receipt["source_root"], __file__, "evaluate_condition_relevance_transfer",
            receipt["code_pins"],
        )

        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model, loading = AutoModelForSequenceClassification.from_pretrained(
            checked["model_root"],
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            output_loading_info=True,
        )
        if any(loading.values()) or model.config.label2id.get("entailment") != 0:
            raise ValueError("native base loading mismatch")
        names = trainable_names([n for n, _ in model.named_parameters()])
        for n, p in model.named_parameters():
            p.requires_grad_(n in names)
        model = model.cuda().eval()
        head = torch.nn.Linear(model.classifier.in_features, 1).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(
            checked["model_root"], local_files_only=True, trust_remote_code=False
        )
        environment = dict(
            python=sys.version,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            dtype=str(next(model.parameters()).dtype),
            attention=model.config._attn_implementation,
            packages={
                n: importlib.metadata.version(n)
                for n in ("transformers", "numpy", "tokenizers", "safetensors", "psutil")
            },
        )
        if (
            environment != checked["effective_configs"]["ce"]["environment"]
            or names != checked["effective_configs"]["ce"]["trainable_names"]
            or frozen_hash(model) != checked["runs"]["ce"]["frozen_sha256"]
            or sum(p.numel() for p in model.parameters()) != config["total_parameters"]
            or sum(p.numel() for p in head.parameters()) != config["auxiliary_parameters"]
        ):
            raise ValueError("runtime environment, parameter scope or frozen base mismatch")
        result["environment"] = environment
        torch.cuda.reset_peak_memory_stats()

        def guard():
            memory = psutil.Process().memory_info()
            if (
                time.perf_counter() - began > config["max_seconds"]
                or getattr(memory, "peak_wset", memory.rss) > config["max_host_bytes"]
                or torch.cuda.max_memory_allocated() > config["max_vram_bytes"]
                or shutil.disk_usage(output).free < config["min_free_disk_bytes"]
                or sum(p.stat().st_size for p in output.iterdir() if p.is_file())
                > config["max_output_bytes"]
            ):
                raise RuntimeError("inference resource guard; comparison incomplete")

        development = {
            p: load_development(checked["corpus_root"], p, config["corpus_sha256"])
            for p in DEVELOPMENT
        }
        if mode == "train-smoke":
            train = [r for part in DEVELOPMENT[:3] for r in development[part]]
            rows = []
            for task in sorted({row_key(r) for r in train}):
                rows += sorted(
                    (r for r in train if row_key(r) == task), key=lambda r: digest([17, r["id"]])
                )[:8]
            if len(rows) != 536:
                raise ValueError("original fixed TRAIN subset changed")
        else:
            raw = read_frozen_transfer(
                comparison / "comparison-freeze.json",
                freeze_sha,
                receipt["run_directories"],
                receipt["result_hashes"],
                receipt["source_root"],
            )
            result["transfer_loaded"] = True
            write(output / "result.json", result)
            rows = [json.loads(line) for line in raw.splitlines()]
            del raw
            reserved = {state_key(r["state"]) for values in development.values() for r in values}
            write(
                output / "transfer-accounting.json",
                validate_transfer_rows(rows, checked["payload"], reserved),
            )
            maximum = 0
            for row in rows:
                first, second = map(list, zip(*render_pairs(row), strict=True))
                encoded = tokenizer(first, second, truncation=False, padding=False)["input_ids"]
                maximum = max(maximum, max(map(len, encoded)))
            if maximum > config["max_length"]:
                raise ValueError("fixed transfer questions must not truncate")
            write(
                output / "token-preflight.json",
                dict(rows=len(rows), maximum_pair_tokens=maximum, truncated=0),
            )
        del development
        metrics_by_role, exact = {}, []
        pairs_completed = 0

        def forward(encoded):
            nonlocal pairs_completed
            guard()
            with torch.inference_mode():
                scores, _ = forward_with_relevance(model, head, encoded.to("cuda"))
                values = scores.cpu().numpy()
            pairs_completed += len(values)
            if pairs_completed % 4096 == 0:
                write(
                    output / "active.json",
                    dict(
                        phase="scoring",
                        pairs_completed=pairs_completed,
                        seconds=time.perf_counter() - began,
                    ),
                )
            return values

        for group in groups:
            role = group["canonical_role"]
            path = checked["run_directories"][group["arm"]] / group["filename"]
            _verify_stream(path, group["sha256"])
            load_trainable(model, head, names, path)
            guard()
            write(
                output / "active.json",
                dict(phase="scoring", role=role, pairs_completed=pairs_completed),
            )
            scores = score_rows(
                rows,
                tokenizer,
                forward,
                pair_batch=config["evaluation_pair_batch"],
                max_length=config["max_length"],
            )
            np.save(output / (role + "-scores.npy"), scores, allow_pickle=False)
            if mode == "train-smoke":
                name = f"fixed-train-{group['step']}-scores.npy"
                arm = group["arm"]
                reference = np.load(
                    io.BytesIO(
                        _verified_bytes(
                            checked["run_directories"][arm] / name,
                            checked["runs"][arm]["artifacts"][name],
                        )
                    ),
                    allow_pickle=False,
                )
                if not np.array_equal(scores, reference):
                    write(
                        output / (role + "-mismatch.json"),
                        dict(maximum_absolute_difference=float(np.max(np.abs(scores - reference)))),
                    )
                    raise ValueError("fixed TRAIN scores do not exactly replay")
                exact.append(group["sha256"])
            else:
                summary = transfer_metrics(rows, scores)
                write(output / (role + "-metrics.json"), summary)
                probabilities = np.exp(log_probabilities(scores))
                with (output / (role + "-predictions.jsonl")).open("x", encoding="utf-8") as stream:
                    for row, probability in zip(rows, probabilities, strict=True):
                        prediction = int(probability.argmax())
                        stream.write(
                            json.dumps(
                                dict(
                                    id=row["id"],
                                    group=row["group"],
                                    condition=row["condition"],
                                    field=row["field"],
                                    target=row["target"],
                                    target_id=row["target_id"],
                                    option_ids=row["option_ids"],
                                    prediction=row["options"][prediction],
                                    predicted_id=row["option_ids"][prediction],
                                    probabilities=probability.tolist(),
                                )
                            )
                            + "\n"
                        )
                for alias in group["roles"]:
                    metrics_by_role[alias] = dict(
                        metrics=summary, canonical_role=role, identity=frozen["roles"][alias]
                    )
            if frozen_hash(model) != checked["runs"]["ce"]["frozen_sha256"]:
                raise ValueError("frozen input embeddings changed during inference")
            guard()
        result["module_origins"] = verify_execution_source(
            receipt["source_root"], __file__, "evaluate_condition_relevance_transfer",
            receipt["code_pins"],
        )
        verify_pair_files(
            receipt["run_directories"], receipt["result_hashes"], receipt["source_root"]
        )
        result.update(
            status="PASS_CONDITION_RELEVANCE_TRAIN_REPLAY"
            if mode == "train-smoke"
            else "PASS_FIXED_CONDITION_RELEVANCE_TRANSFER",
            finished_utc=datetime.now(UTC).isoformat(),
            seconds=time.perf_counter() - began,
            rows=len(rows),
            unique_weights_evaluated=len(groups),
            all_roles=frozen["roles"],
            exact_replay_groups=exact,
            metrics_by_role=metrics_by_role,
            pair_forwards=pairs_completed,
            peak_cuda_allocated=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved=torch.cuda.max_memory_reserved(),
            peak_host_bytes=getattr(psutil.Process().memory_info(), "peak_wset", None),
            blind_evaluation=False,
            calibration_applied=False,
            checkpoint_reselected=False,
            artifacts={
                p.name: sha(p)
                for p in output.iterdir()
                if p.name not in ("result.json", "active.json")
            },
        )
        write(output / "result.json", result)
        write(output / "active.json", dict(phase="complete", seconds=time.perf_counter() - began))
        return result
    except Exception as exc:
        result.update(
            status="INCOMPLETE",
            error=str(exc),
            traceback=traceback.format_exc(),
            seconds=time.perf_counter() - began,
        )
        write(output / "result.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--comparison-sha256", required=True)
    parser.add_argument("--mode", choices=("train-smoke", "transfer"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-reference", type=Path)
    parser.add_argument("--smoke-sha256")
    args = parser.parse_args()
    result = run(
        args.comparison,
        args.comparison_sha256,
        args.mode,
        args.output,
        args.smoke_reference,
        args.smoke_sha256,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
