"""Post-selection matched contrast comparison and retained-reference evaluation."""

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.binding_language import quality_failures
from scripts.binding_transfer import control_summary
from scripts.build_condition_binding_corpus import lines
from scripts.condition_binding_training import score_summary
from scripts.condition_contrast import CONDITIONS, delta_digest
from scripts.evaluate_semantic_state_bridge import read, sha, tensor_digest, verify, write
from scripts.prepare_condition_contrast import CONFIG, validate_corpus
from scripts.semantic_state_diagnostic import log_probabilities
from scripts.summarize_condition_binding import score_difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--ce-fit", type=Path, required=True)
    parser.add_argument("--ce-sha256", required=True)
    parser.add_argument("--contrast-fit", type=Path, required=True)
    parser.add_argument("--contrast-sha256", required=True)
    parser.add_argument("--replay-reference", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if bool(args.replay_reference) != bool(args.replay_sha256):
        raise ValueError("invalid mode")
    args.output.mkdir(parents=True, exist_ok=False)
    tick = time.perf_counter()
    result = dict(
        status="RUNNING",
        command=sys.argv,
        started_utc=datetime.now(UTC).isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        smoke=False,
        model_updates=0,
        sealed_audit_loaded=False,
        old_transfer_scored=False,
        cloud_cost=0,
    )
    write(args.output / "result.json", result)
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import psutil
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from scripts.train_numeric_deterministic import configure_repair_policy

        config = read(CONFIG)
        pins, code = validate_corpus(args.corpus, args.corpus_sha256, config)
        fits = {}
        expected_training_pins, _ = validate_corpus(
            args.corpus, args.corpus_sha256, config, training_only=True
        )
        for arm, directory, digest in (
            ("ce", args.ce_fit, args.ce_sha256),
            ("contrast", args.contrast_fit, args.contrast_sha256),
        ):
            pins[str(directory / "result.json")] = digest
            verify(pins)
            fit = read(directory / "result.json")
            if (
                fit["status"] != "PASS_FIXED_CONDITION_CONTRAST"
                or fit["arm"] != arm
                or fit["steps"] != 512
                or fit["code_pins"] != code
                or not fit["selected_reload_exact"]
                or fit["input_pins"] != expected_training_pins
            ):
                raise ValueError("both complete matched fits required before transfer")
            fits[arm] = fit
            pins.update({str(directory / p): h for p, h in fit["artifacts"].items()})
            verify(pins)
        reference = None
        if args.replay_reference:
            verify({str(args.replay_reference / "result.json"): args.replay_sha256})
            reference = read(args.replay_reference / "result.json")
            if (
                reference["status"] != "PASS_FIXED_CONDITION_CONTRAST_TRANSFER"
                or reference["input_pins"] != pins
                or reference["code_pins"] != code
            ):
                raise ValueError("invalid full replay reference")
            verify({str(args.replay_reference / p): h for p, h in reference["artifacts"].items()})
        root, base = Path(config["root"]), Path(config["base_directory"])
        data = {name: lines(args.corpus / f"{name}.jsonl") for name in CONDITIONS}
        torch.set_num_threads(4)
        torch.manual_seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        tokenizer = AutoTokenizer.from_pretrained(base / "model", local_files_only=True)
        model = (
            AutoModelForSequenceClassification.from_pretrained(
                base / "model", local_files_only=True, attn_implementation="sdpa"
            )
            .cuda()
            .eval()
        )
        names = configure_repair_policy(model, "all_encoder_blocks")
        if sum(p.numel() for p in model.parameters()) != 22713601:
            raise ValueError("parameter count changed")
        model.requires_grad_(False)
        effective = dict(
            config=config,
            corpus_sha256=args.corpus_sha256,
            base_revision=read(base / "result-retry.json")["revision"],
            total_parameters=22713601,
            trainable_parameters=0,
            environment=dict(
                python=sys.version,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),
                numpy=np.__version__,
                dtype=str(next(model.parameters()).dtype),
                attention=model.config._attn_implementation,
                threads=4,
                packages={
                    n: importlib.metadata.version(n)
                    for n in ("transformers", "tokenizers", "safetensors", "psutil")
                },
            ),
        )
        write(args.output / "effective-config.json", effective)

        def guard():
            mem = psutil.Process().memory_info()
            if (
                time.perf_counter() - tick > config["max_seconds"]
                or getattr(mem, "peak_wset", mem.rss) > config["max_host_bytes"]
                or torch.cuda.max_memory_allocated() > config["max_vram_bytes"]
                or shutil.disk_usage(args.output).free < config["min_free_disk_bytes"]
                or sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())
                > config["max_output_bytes"]
            ):
                raise RuntimeError("resource guard; comparison incomplete")

        summaries, arrays, times, digests = {}, {}, {}, {}
        pair_calls, max_tokens = 0, 0
        models = [
            ("control", args.ce_fit / "final-delta.pt"),
            ("candidate", args.contrast_fit / "final-delta.pt"),
            ("retained_reference", root / "condition-binding-retention-fit-01/best-delta.pt"),
        ]
        selected_labels = {}
        for arm, label, directory in (
            ("ce", "control", args.ce_fit),
            ("contrast", "candidate", args.contrast_fit),
        ):
            path = directory / "best-delta.pt"
            selected_digest = delta_digest(path)
            existing = next(
                (name for name, p in models if delta_digest(p) == selected_digest), None
            )
            if existing is None:
                existing = label + "_selected"
                models.append((existing, path))
            selected_labels[arm] = existing
        for label, path in models:
            delta = torch.load(path, map_location="cpu", weights_only=True)
            if set(delta) != set(names):
                raise ValueError("delta coverage mismatch")
            loaded = model.load_state_dict(delta, strict=False)
            if loaded.unexpected_keys or set(loaded.missing_keys) != set(model.state_dict()) - set(
                names
            ):
                raise ValueError("partial load mismatch")
            del delta
            digests[label] = tensor_digest(model)
            summaries[label], arrays[label], times[label] = {}, {}, {}
            for name, rows in data.items():
                start = time.perf_counter()
                values = []
                with torch.inference_mode():
                    for offset in range(0, len(rows), config["question_batch_size"]):
                        guard()
                        batch = rows[offset : offset + config["question_batch_size"]]
                        first = [
                            f"Question: {r['question']}\nCandidate answer: {o}"
                            for r in batch
                            for o in r["options"]
                        ]
                        second = [f"State: {r['state']}" for r in batch for _ in r["options"]]
                        tokens = tokenizer(
                            first, second, padding=True, truncation=False, return_tensors="pt"
                        )
                        max_tokens = max(max_tokens, tokens["input_ids"].shape[1])
                        if max_tokens > config["max_tokens"]:
                            raise ValueError("truncation forbidden")
                        scores = model(**tokens.to("cuda")).logits[:, 0].cpu().numpy()
                        if not np.isfinite(scores).all():
                            raise FloatingPointError("nonfinite logits")
                        values.append(scores.reshape(len(batch), 40))
                        pair_calls += len(scores)
                values = np.concatenate(values)
                arrays[label][name] = values
                probability = np.exp(log_probabilities(values))
                np.save(args.output / f"{label}-{name}-scores.npy", values, allow_pickle=False)
                np.save(
                    args.output / f"{label}-{name}-probabilities.npy",
                    probability,
                    allow_pickle=False,
                )
                summaries[label][name] = score_summary(rows, values)
                with (args.output / f"{label}-{name}-predictions.jsonl").open(
                    "w", encoding="utf-8"
                ) as f:
                    for row, p in zip(rows, probability, strict=True):
                        f.write(
                            json.dumps(
                                dict(
                                    id=row["id"],
                                    group=row["group"],
                                    target_id=row["target_id"],
                                    option_ids=row["option_ids"],
                                    prediction_id=row["option_ids"][int(p.argmax())],
                                    probabilities=p.tolist(),
                                ),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                if tensor_digest(model) != digests[label] or any(
                    p.requires_grad or p.grad is not None for p in model.parameters()
                ):
                    raise ValueError("frozen model changed or gradients present")
                times[label][name] = time.perf_counter() - start
                print(
                    json.dumps(
                        dict(
                            model=label,
                            condition=name,
                            pairs=pair_calls,
                            seconds=time.perf_counter() - tick,
                        )
                    ),
                    flush=True,
                )
        expected = len(models) * len(CONDITIONS) * len(data["known"]) * 40
        if pair_calls != expected:
            raise ValueError("inference accounting mismatch")
        payloads = [p.name for p in args.output.glob("*.npy")]
        intervals = {
            name: score_difference(data[name], arrays["control"][name], arrays["candidate"][name])
            for name in CONDITIONS
        }
        for interval in intervals.values():
            interval["columns"] = [
                "accuracy_candidate512_minus_control512",
                "nll_candidate512_minus_control512",
            ]
        controls = {}
        for label in arrays:
            controls[label] = control_summary(
                arrays[label]["focus_descriptions"],
                arrays[label]["reverse_focus_descriptions"],
                arrays[label]["no_question"],
            )
            no = summaries[label]["no_question"]
            if (
                no["overall"]["accuracy"] > 0.25
                or no["all_four_correct"] != 0
                or no["pair_both_correct"] != 0
            ):
                raise ValueError("order/question-free control violation")
        write(
            args.output / "metrics.json",
            dict(
                models=summaries,
                selected_models=selected_labels,
                selected_steps={arm: fit["best_step"] for arm, fit in fits.items()},
                primary_comparison="CE512 versus contrast512; same data and budget",
                final_retention={
                    arm: read(directory / "history.json")[-1]["eligible"]
                    for arm, directory in (("ce", args.ce_fit), ("contrast", args.contrast_fit))
                },
                paired_intervals=intervals,
                controls=controls,
                transfer_gate_failures=quality_failures(
                    summaries["candidate"], summaries["control"], intervals
                ),
            ),
        )
        payloads += [p.name for p in args.output.glob("*-predictions.jsonl")] + ["metrics.json"]
        if reference:
            for file in payloads + ["effective-config.json"]:
                if sha(args.output / file) != reference["artifacts"].get(file):
                    raise ValueError(f"fresh full replay mismatch: {file}")
        verify(pins)
        verify(code)
        guard()
        result.update(
            status="PASS_FIXED_CONDITION_CONTRAST_TRANSFER",
            finished_utc=datetime.now(UTC).isoformat(),
            seconds=time.perf_counter() - tick,
            input_pins=pins,
            code_pins=code,
            frozen_tensor_digests=digests,
            sources_unchanged=True,
            condition_seconds=times,
            actual_candidate_pairs=pair_calls,
            max_tokens=max_tokens,
            logical_questions_per_model=sum(map(len, data.values())),
            unique_state_count=len(data["known"]) // 4,
            fresh_replay_files_exact=payloads + ["effective-config.json"] if reference else [],
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_host_bytes=getattr(psutil.Process().memory_info(), "peak_wset", None),
            free_disk_bytes=shutil.disk_usage(args.output).free,
            artifacts={
                p.name: sha(p)
                for p in args.output.iterdir()
                if p.is_file() and p.name != "result.json"
            },
        )
        write(args.output / "result.json", result)
        print(
            json.dumps({k: result[k] for k in ("status", "seconds", "actual_candidate_pairs")}),
            flush=True,
        )
    except Exception:
        result.update(
            status="INCOMPLETE",
            traceback=traceback.format_exc(),
            seconds=time.perf_counter() - tick,
            finished_utc=datetime.now(UTC).isoformat(),
        )
        write(args.output / "result.json", result)
        raise


if __name__ == "__main__":
    main()
