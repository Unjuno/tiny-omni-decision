"""Frozen, inference-only local teacher screen with bounded independent replay."""

import argparse
import functools
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.evaluate_semantic_state_bridge import read, sha, verify, write
from scripts.semantic_teacher_probe import (
    LABELS,
    build_probe,
    input_text,
    probabilities,
    summarize_rows,
)
from tiny_omni_decision.decision_math import label_token_ids_from_prompt

CONFIG = "configs/pretrained_reuse/semantic_teacher_probe_v1.json"
CODE = [
    CONFIG,
    "scripts/evaluate_semantic_teacher.py",
    "scripts/semantic_teacher_probe.py",
    "scripts/build_conditional_curriculum.py",
    "scripts/condition_binding_corpus.py",
    "scripts/semantic_state_diagnostic.py",
    "scripts/evaluate_semantic_state_bridge.py",
    "src/tiny_omni_decision/decision_math.py",
    "src/tiny_omni_decision/dataset.py",
    "tests/test_semantic_teacher_probe.py",
    "docs/pretrained_reuse/R2_SEMANTIC_TEACHER_PROTOCOL_20261015.md",
]


def code_pins():
    subprocess.run(["git", "ls-files", "--error-unmatch", *CODE], check=True, capture_output=True)
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", *CODE], check=True, capture_output=True
    )
    return {str(Path(p).resolve()): sha(p) for p in CODE}


def validate_reference(ref, config, code, model, inputs):
    if (
        ref.get("status") != "PASS_FULL_TRAIN_SCREEN"
        or ref.get("config") != config
        or ref.get("code_pins") != code
        or ref.get("model_pins") != model
        or ref.get("input_pins") != inputs
    ):
        raise ValueError("complete identical reference required")


def encode(processor, row, config):
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": input_text(row)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids, length = label_token_ids_from_prompt(
        processor.tokenizer, rendered, list(LABELS[: len(row["options"])])
    )
    encoded = dict(processor(text=rendered, return_tensors="pt"))
    tokens = encoded["input_ids"][0].tolist()
    if len(tokens) > config["max_sequence_length"] or len(tokens) != length:
        raise ValueError("truncation/tokenizer-processor mismatch")
    return encoded, ids, rendered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "smoke", "full", "replay"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = read(CONFIG)
    tick = time.perf_counter()
    result = dict(
        status="RUNNING",
        started_utc=datetime.now(UTC).isoformat(),
        stage=args.stage,
        command=[sys.executable, *sys.argv],
        model_updates=0,
        cloud_cost=0,
        validation_loaded=False,
        sealed_audit_loaded=False,
        config=config,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    write(args.output / "result.json", result)
    try:
        pins = code_pins()
        verify(config["train_files"])
        result["code_pins"] = pins
        result["input_pins"] = dict(config["train_files"])
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import psutil
        import torch
        import transformers.modeling_utils as modeling_utils
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        torch.set_num_threads(4)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        model_path = Path(config["model_path"])
        model_pins = {str(p): sha(p) for p in model_path.iterdir() if p.is_file()}
        if model_pins[str(model_path / "model.safetensors")] != config["model_weight_sha256"]:
            raise ValueError("pretrained weight hash mismatch")
        result["model_pins"] = model_pins
        result["environment"] = dict(
            python=sys.version,
            cuda=torch.version.cuda,
            packages={
                k: importlib.metadata.version(k)
                for k in ("torch", "transformers", "safetensors", "tokenizers", "numpy", "psutil")
            },
            platform=sys.platform,
            disk_free_bytes=shutil.disk_usage(args.output).free,
            gpu=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip(),
        )

        def guard():
            mem = psutil.Process().memory_info()
            size = sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())
            if (
                time.perf_counter() - tick > config["max_seconds"]
                or getattr(mem, "peak_wset", mem.rss) > config["max_host_bytes"]
                or torch.cuda.max_memory_allocated() > config["max_vram_bytes"]
                or size > config["max_output_bytes"]
                or shutil.disk_usage(args.output).free < config["min_free_disk_bytes"]
            ):
                raise RuntimeError("resource guard; incomplete, not capability failure")

        guard()
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        sources = [
            [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines()]
            for p in config["train_files"]
        ]
        rows = build_probe(*sources, config["groups_per_source"])
        if args.stage == "prepare":
            token_info = []
            for row in rows:
                encoded, ids, rendered = encode(processor, row, config)
                token_info.append(
                    dict(
                        row_id=row["id"],
                        tokens=encoded["input_ids"][0].tolist(),
                        label_ids=ids,
                        prompt=rendered,
                    )
                )
            write(args.output / "rows.json", rows)
            write(args.output / "tokens.json", token_info)
            result.update(
                status="PASS_PREPARED_TRAIN_ONLY",
                rows=len(rows),
                unique_states=len({(r["source"], r["group"]) for r in rows}),
                conditions=dict(Counter(f"{r['source']}/{r['condition']}" for r in rows)),
                max_tokens=max(len(r["tokens"]) for r in token_info),
            )
        else:
            if args.corpus is None:
                raise ValueError("frozen corpus required")
            corpus = read(args.corpus / "result.json")
            if (
                corpus["status"] != "PASS_PREPARED_TRAIN_ONLY"
                or corpus["config"] != config
                or corpus["code_pins"] != pins
                or corpus["model_pins"] != model_pins
            ):
                raise ValueError("corpus provenance mismatch")
            corpus_pins = {
                str(args.corpus / name): digest for name, digest in corpus["artifacts"].items()
            }
            verify(corpus_pins)
            result["input_pins"].update(corpus_pins)
            result["input_pins"][str(args.corpus / "result.json")] = sha(
                args.corpus / "result.json"
            )
            if rows != read(args.corpus / "rows.json"):
                raise ValueError("frozen rows mismatch")
            token_info = read(args.corpus / "tokens.json")
            # Every condition and source represented; chosen before any forward.
            first_groups = {
                s: list(dict.fromkeys(r["group"] for r in rows if r["source"] == s))[:1]
                for s in ("binding", "numeric")
            }
            subset = [i for i, r in enumerate(rows) if r["group"] in first_groups[r["source"]]]
            indexes = list(range(len(rows))) if args.stage == "full" else subset
            expected = None
            if args.stage == "replay":
                if args.reference is None:
                    raise ValueError("full reference required")
                ref = read(args.reference / "result.json")
                validate_reference(ref, config, pins, model_pins, result["input_pins"])
                rp = {
                    str(args.reference / "predictions.json"): ref["artifacts"]["predictions.json"]
                }
                verify(rp)
                result["input_pins"].update(rp)
                result["input_pins"][str(args.reference / "result.json")] = sha(
                    args.reference / "result.json"
                )
                expected = read(args.reference / "predictions.json")
            # Process-local reader choice; no installed dependency or legacy file edits.
            original_safe_open = modeling_utils.safe_open
            modeling_utils.safe_open = functools.partial(original_safe_open, backend="pread")
            load_tick = time.perf_counter()
            try:
                model, info = AutoModelForMultimodalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    dtype=torch.bfloat16,
                    device_map={"": "cuda:0"},
                    attn_implementation="sdpa",
                    output_loading_info=True,
                )
            finally:
                modeling_utils.safe_open = original_safe_open
            result["load_seconds"] = time.perf_counter() - load_tick
            result["loading_info"] = {
                key: list(value) if isinstance(value, set) else value for key, value in info.items()
            }
            if any(
                info.get(k)
                for k in ("missing_keys", "mismatched_keys", "error_msgs", "conversion_errors")
            ):
                raise ValueError("incomplete model load")
            model.requires_grad_(False).eval()
            result["total_parameters"] = sum(p.numel() for p in model.parameters())
            result["trainable_parameters"] = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            result["parameter_dtypes"] = dict(Counter(str(p.dtype) for p in model.parameters()))
            result["attention_implementation"] = str(model.config._attn_implementation)
            print(
                json.dumps(
                    {"loaded": result["total_parameters"], "seconds": result["load_seconds"]}
                ),
                flush=True,
            )
            outputs, differences = [], []
            inference_tick = time.perf_counter()
            with (args.output / "progress.jsonl").open("w", encoding="utf-8") as progress:
                with torch.inference_mode():
                    for index in indexes:
                        guard()
                        row = rows[index]
                        encoded, ids, rendered = encode(processor, row, config)
                        expected_tokens = token_info[index]
                        if (
                            expected_tokens["prompt"] != rendered
                            or ids != expected_tokens["label_ids"]
                            or encoded["input_ids"][0].tolist() != expected_tokens["tokens"]
                        ):
                            raise ValueError("prepared prompt/token mismatch")
                        encoded = {k: v.to("cuda:0") for k, v in encoded.items()}
                        output = model(**encoded, use_cache=False, logits_to_keep=1)
                        logits = output.logits[0, -1].float()
                        if not torch.isfinite(logits).all():
                            raise ValueError("nonfinite vocabulary logits")
                        values = logits[ids].cpu().numpy().astype(np.float64)
                        p = probabilities(values)
                        item = dict(
                            row_id=row["id"],
                            logits=values.tolist(),
                            probabilities=p.tolist(),
                            prediction_id=row["option_ids"][int(p.argmax())],
                            target_id=row["target_id"],
                            option_probability_mass=float(
                                torch.softmax(logits, -1)[ids].sum().item()
                            ),
                            unrestricted_top_token=int(logits.argmax()),
                            tokens=int(encoded["input_ids"].shape[-1]),
                        )
                        if expected is not None:
                            reference = expected[index]
                            if reference["row_id"] != item["row_id"]:
                                raise ValueError("replay identity mismatch")
                            difference = float(np.max(np.abs(values - reference["logits"])))
                            differences.append(difference)
                            if (
                                difference > config["replay_atol"]
                                or item["prediction_id"] != reference["prediction_id"]
                            ):
                                raise ValueError("bounded replay mismatch")
                        outputs.append(item)
                        progress.write(
                            json.dumps(
                                dict(
                                    index=index,
                                    row_id=row["id"],
                                    elapsed_seconds=time.perf_counter() - tick,
                                    prediction=item,
                                )
                            )
                            + "\n"
                        )
                        progress.flush()
                        del output, logits, encoded
                        if len(outputs) % 32 == 0:
                            print(
                                json.dumps(
                                    dict(
                                        completed=len(outputs),
                                        total=len(indexes),
                                        seconds=time.perf_counter() - tick,
                                    )
                                ),
                                flush=True,
                            )
            torch.cuda.synchronize()
            result["inference_seconds"] = time.perf_counter() - inference_tick
            selected_rows = [rows[i] for i in indexes]
            summary = summarize_rows(selected_rows, outputs)
            write(args.output / "predictions.json", outputs)
            write(args.output / "metrics.json", summary)
            result.update(
                status={
                    "smoke": "PASS_SMOKE",
                    "full": "PASS_FULL_TRAIN_SCREEN",
                    "replay": "PASS_BOUNDED_REPLAY",
                }[args.stage],
                rows=len(outputs),
                metrics=summary,
                peak_vram_bytes=torch.cuda.max_memory_allocated(),
                replay_max_abs_logit_difference=max(differences) if differences else None,
                model_updates=0,
                teacher_adopted=False,
                student_training_started=False,
            )
            if args.stage == "full":
                eligibility = {}
                for name in (
                    "binding/direct",
                    "binding/exclusion",
                    "binding/switched",
                    "numeric/original",
                ):
                    eligibility[name] = {
                        task: m["accuracy"] >= config["teacher_screen_accuracy_floor"]
                        and m["ece"] <= config["teacher_screen_ece_ceiling"]
                        for task, m in summary["conditions"][name]["tasks"].items()
                    }
                result["train_screen_eligible_by_task"] = eligibility
                result["reversal_pass"] = {
                    s: m["answer_agreement"] >= config["reversal_agreement_floor"]
                    for s, m in summary["reversal"].items()
                }
            guard()
        verify(result["input_pins"])
        verify(pins)
        verify(model_pins)
        result["old_inputs_unchanged"] = True
        result["peak_host_bytes"] = getattr(
            psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss
        )
        result["artifacts"] = {
            p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != "result.json"
        }
    except Exception:
        result.update(status="BLOCKED_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        result.update(
            ended_utc=datetime.now(UTC).isoformat(), elapsed_seconds=time.perf_counter() - tick
        )
        result["artifacts"] = {
            p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != "result.json"
        }
        write(args.output / "result.json", result)
        print(
            json.dumps({k: result.get(k) for k in ("status", "rows", "elapsed_seconds", "error")}),
            flush=True,
        )


if __name__ == "__main__":
    main()
