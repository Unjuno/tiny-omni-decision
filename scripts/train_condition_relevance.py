"""Matched CE/relevance fits; original semantic experiment remains read-only."""

import argparse
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.condition_binding_training import pack_scores
from scripts.condition_relevance_corpus import load_development
from scripts.condition_relevance_training import (
    accounting,
    bootstrap_relevance,
    forward_with_relevance,
    load_trainable,
    metric_summary,
    row_key,
    sample_order,
    save_trainable,
    training_objective,
)
from scripts.decision_semantic_training import learning_rate, render_pairs, trainable_names
from scripts.decision_supervision import digest, select_validation
from scripts.evaluate_semantic_state_bridge import read, sha, verify, write
from scripts.semantic_state_diagnostic import log_probabilities

CONFIG = "configs/pretrained_reuse/condition_relevance_fit_v1.json"
PROTOCOL = "docs/pretrained_reuse/R2_CONDITION_RELEVANCE_FIT_PROTOCOL_20261015.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm", choices=("ce", "relevance"), required=True)
    parser.add_argument("--replay-reference", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if bool(args.replay_reference) != bool(args.replay_sha256) or (
        args.smoke and args.replay_reference
    ):
        raise ValueError("invalid execution mode")
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    result = dict(
        status="RUNNING",
        started_utc=datetime.now(UTC).isoformat(),
        command=sys.argv,
        pid=os.getpid(),
        smoke=args.smoke,
        arm=args.arm,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        sealed_audit_loaded=False,
        transfer_loaded=False,
        product_complete=False,
        cloud_cost=0,
        exact_resume=False,
    )
    write(args.output / "result.json", result)
    pins, code_pins, train, order = {}, {}, [], []
    completed_updates = training_forwards = training_backwards = pair_forwards = 0
    resource_snapshot = None
    failure_phase = "initialization"
    try:
        config = read(CONFIG)
        coefficient = config["relevance_coefficients"][args.arm]
        if config["retention_kl_coefficient"] != 1.0:
            raise ValueError("the fixed legacy KL coefficient must remain one")
        result.update(config=config, config_sha256=sha(CONFIG), relevance_coefficient=coefficient)
        scope = ["*.py", "pyproject.toml", CONFIG, PROTOCOL]
        files = subprocess.check_output(["git", "ls-files", "--", *scope], text=True).splitlines()
        if "scripts/train_condition_relevance.py" not in files:
            raise ValueError("freeze source before execution")
        subprocess.run(
            ["git", "diff", "--exit-code", "HEAD", "--", *scope], check=True, capture_output=True
        )
        code_pins = {str(Path(p).resolve()): sha(p) for p in files}
        corpus, model_dir = Path(config["corpus"]), Path(config["model"])
        pins = {
            str(corpus / "result.json"): config["corpus_sha256"],
            str(model_dir.parent / "download.json"): config["model_manifest_sha256"],
        }
        verify(pins)
        manifest = read(corpus / "result.json")
        model_manifest = read(model_dir.parent / "download.json")
        if model_manifest["revision"] != config["revision"]:
            raise ValueError("base revision mismatch")
        pins.update({str(model_dir / p): h for p, h in model_manifest["hashes"].items()})
        partitions = ["binding_train", "numeric_train", "legacy_train"]
        if not args.smoke:
            partitions += ["validation", "old_validation", "legacy_validation"]
        data = {p: load_development(corpus, p, config["corpus_sha256"]) for p in partitions}
        pins.update(
            {str(corpus / (p + ".jsonl")): manifest["artifacts"][p + ".jsonl"] for p in partitions}
        )
        pins[str(corpus / "selection-schema.json")] = manifest["artifacts"]["selection-schema.json"]
        verify(pins)
        schema = read(corpus / "selection-schema.json")
        train = [data[p] for p in partitions[:3]]
        order = sample_order(train, sizes=config["questions_per_update"], seed=config["seed"])
        if len(order) != config["steps"] or len(schema) != 47:
            raise ValueError("fixed schedule or selection schema changed")
        ordered_rows = [train[s][i] for batch in order for s, i in batch]
        write(args.output / "sample-order.json", order)
        write(
            args.output / "sample-accounting.json",
            dict(
                planned_full_fit=accounting(ordered_rows),
                ordered_id_sha256=digest([r["id"] for r in ordered_rows]),
            ),
        )
        fixed_train = []
        for task in sorted({row_key(r) for r in ordered_rows}):
            fixed_train += sorted(
                (r for r in ordered_rows if row_key(r) == task), key=lambda r: digest([17, r["id"]])
            )[:8]
        fixed_schema = dict(Counter(row_key(r) for r in fixed_train))
        validation = (
            []
            if args.smoke
            else data["validation"] + data["old_validation"] + data["legacy_validation"]
        )
        reference = None
        if args.replay_reference:
            verify({str(args.replay_reference / "result.json"): args.replay_sha256})
            reference = read(args.replay_reference / "result.json")
            if (
                reference["status"] != "PASS_MATCHED_RELEVANCE_FIT"
                or reference["arm"] != args.arm
                or reference["smoke"]
                or reference["input_pins"] != pins
                or reference["code_pins"] != code_pins
            ):
                raise ValueError("completed same-code same-input fit required for replay")
            verify({str(args.replay_reference / p): h for p, h in reference["artifacts"].items()})

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import psutil
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from scripts.train_text_crossencoder import choice_loss, frozen_hash

        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model, loading = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            output_loading_info=True,
        )
        if any(loading.values()) or model.config.label2id.get("entailment") != 0:
            raise ValueError("strict native model loading or label failure")
        names = trainable_names([n for n, _ in model.named_parameters()])
        for n, p in model.named_parameters():
            p.requires_grad_(n in names)
        model = model.cuda().eval()
        parameters = [p for p in model.parameters() if p.requires_grad]
        if (
            sum(p.numel() for p in parameters) != config["trainable_parameters"]
            or sum(p.numel() for p in model.parameters()) != config["total_parameters"]
            or model.config._attn_implementation != "eager"
        ):
            raise ValueError("architecture or backend changed")
        head = torch.nn.Linear(model.classifier.in_features, 1).cuda().eval()
        if sum(p.numel() for p in head.parameters()) != config["auxiliary_parameters"]:
            raise ValueError("auxiliary head size changed")
        parameters += list(head.parameters())
        frozen = frozen_hash(model)
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False
        )
        torch.cuda.reset_peak_memory_stats()
        pair_forwards = 0
        evaluation_seconds = 0.0
        training_seconds = 0.0
        token_report = {}

        def guard():
            memory = psutil.Process().memory_info()
            if (
                time.perf_counter() - began > config["max_seconds"]
                or getattr(memory, "peak_wset", memory.rss) > config["max_host_bytes"]
                or torch.cuda.max_memory_allocated() > config["max_vram_bytes"]
                or shutil.disk_usage(args.output).free < config["min_free_disk_bytes"]
                or sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())
                > config["max_output_bytes"]
            ):
                raise RuntimeError("resource guard; experiment incomplete")

        def resource_snapshot():
            return dict(
                peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved=torch.cuda.max_memory_reserved(),
                peak_host_bytes=getattr(psutil.Process().memory_info(), "peak_wset", None),
                free_disk_bytes=shutil.disk_usage(args.output).free,
            )

        def phase(name, step=0):
            nonlocal failure_phase
            failure_phase = name
            write(
                args.output / "active.json",
                dict(
                    phase=name,
                    step=step,
                    pid=os.getpid(),
                    utc=datetime.now(UTC).isoformat(),
                    seconds=time.perf_counter() - began,
                ),
            )

        longest_train = None
        longest_train_length = 0
        for part, rows in data.items():
            phase("token_preflight_" + part)
            truncated = []
            maximum = 0
            for r in rows:
                pairs = render_pairs(r)
                first, second = map(list, zip(*pairs, strict=True))
                lengths = tokenizer(first, second, truncation=False, padding=False)["input_ids"]
                largest = max(map(len, lengths))
                maximum = max(maximum, largest)
                if largest > config["max_length"]:
                    if r["source"] in ("programmatic_decision_v1", "condition_relevance_v1"):
                        raise ValueError("new decision truncation forbidden")
                    short = tokenizer(first, truncation=False, padding=False)["input_ids"]
                    if max(map(len, short)) + 1 >= config["max_length"]:
                        raise ValueError("question/candidate leaves no state token budget")
                    truncated.append(
                        dict(
                            id=r["id"],
                            maximum_pair_tokens=largest,
                            pair_lengths=[len(v) for v in lengths],
                        )
                    )
                if part.endswith("train") and largest > longest_train_length:
                    longest_train, longest_train_length = r, largest
            token_report[part] = dict(
                maximum_pair_tokens=maximum, truncated_questions=truncated, rows=len(rows)
            )
            guard()
        write(args.output / "token-preflight.json", token_report)
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
        write(
            args.output / "effective-config.json",
            dict(
                config=config,
                config_sha256=sha(CONFIG),
                environment=environment,
                input_pins=pins,
                code_pins=code_pins,
                trainable_names=names,
                frozen_sha256=frozen,
                smoke=args.smoke,
                arm=args.arm,
                relevance_coefficient=coefficient,
                auxiliary_parameters=config["auxiliary_parameters"],
                initial_state="pinned_pretrained_no_delta",
                command=sys.argv,
            ),
        )

        def forward_pairs(pairs):
            nonlocal pair_forwards
            guard()
            first, second = map(list, zip(*pairs, strict=True))
            encoded = tokenizer(
                first,
                second,
                padding=True,
                truncation="only_second",
                max_length=config["max_length"],
                return_tensors="pt",
            ).to("cuda")
            scores, relevance = forward_with_relevance(model, head, encoded)
            pair_forwards += len(pairs)
            if not torch.isfinite(scores).all() or not torch.isfinite(relevance).all():
                raise FloatingPointError("nonfinite option logits")
            return scores, relevance

        def score_rows(rows):
            pairs = [pair for r in rows for pair in render_pairs(r)]
            values = []
            with torch.inference_mode():
                for offset in range(0, len(pairs), config["evaluation_pair_batch"]):
                    values.extend(
                        forward_pairs(pairs[offset : offset + config["evaluation_pair_batch"]])[0]
                        .cpu()
                        .numpy()
                        .tolist()
                    )
            result_scores, offset = [], 0
            for r in rows:
                result_scores.append(np.asarray(values[offset : offset + len(r["options"])]))
                offset += len(r["options"])
            return pack_scores(result_scores)

        def evaluate(rows, expected_schema, label):
            nonlocal evaluation_seconds
            started = time.perf_counter()
            scores = score_rows(rows)
            summary = metric_summary(rows, scores, expected_schema)
            probabilities = np.exp(log_probabilities(scores))
            np.save(args.output / (label + "-scores.npy"), scores, allow_pickle=False)
            write(args.output / (label + "-metrics.json"), summary)
            with (args.output / (label + "-predictions.jsonl")).open("w", encoding="utf-8") as f:
                for r, probability in zip(rows, probabilities, strict=True):
                    prediction = int(probability.argmax())
                    ids = r.get("option_ids", r["options"])
                    payload = dict(
                        id=r["id"],
                        group=r["group"],
                        task=row_key(r),
                        target=r["options"].index(r["target"]),
                        prediction=prediction,
                        predicted_id=ids[prediction],
                        option_ids=ids,
                        probabilities=probability[: len(r["options"])].tolist(),
                    )
                    if r.get("supervision", {}).get("fact_true") is not None:
                        payload["predicted_fact_true"] = r["supervision"]["fact_true"][prediction]
                        payload["predicted_request_relevant"] = r["supervision"][
                            "request_relevant"
                        ][prediction]
                    f.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
            evaluation_seconds += time.perf_counter() - started
            return scores, summary

        def save_delta(name):
            guard()
            save_trainable(model, head, names, args.output / name)

        def load_delta(path):
            load_trainable(model, head, names, path)

        history = []
        best_step = 0
        baseline = None

        def snapshot(step):
            nonlocal best_step, baseline
            phase("validation", step)
            _, m = evaluate(validation, schema, f"validation-{step}")
            _, tr = evaluate(fixed_train, fixed_schema, f"fixed-train-{step}")
            if step == 0:
                baseline = m["metrics"]
            entry = dict(
                step=step,
                purpose="validation",
                metrics=m["metrics"],
                macro=m["macro"],
                minimum_accuracy=m["minimum_accuracy"],
                fixed_train=tr,
                consumed=accounting([train[s][i] for batch in order[:step] for s, i in batch]),
                elapsed_seconds=time.perf_counter() - began,
            )
            history.append(entry)
            best_step = select_validation(history, baseline, schema)
            entry["best_step"] = best_step
            save_delta(f"step-{step}-delta.pt")
            if best_step == step:
                shutil.copyfile(
                    args.output / f"step-{step}-delta.pt", args.output / "best-delta.pt"
                )
            if frozen_hash(model) != frozen:
                raise ValueError("frozen parameters changed")
            write(args.output / "history.json", history)
            print(
                json.dumps(
                    dict(
                        step=step,
                        macro=m["macro"],
                        minimum=m["minimum_accuracy"],
                        best_step=best_step,
                        seconds=entry["elapsed_seconds"],
                    )
                ),
                flush=True,
            )

        updates = []
        if not reference:
            phase("native_forward_equivalence")
            probe_pairs = render_pairs(train[0][0])
            first, second = map(list, zip(*probe_pairs, strict=True))
            encoded = tokenizer(
                first,
                second,
                padding=True,
                truncation="only_second",
                max_length=config["max_length"],
                return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode():
                native = model(**encoded).logits[:, 0]
                pair_forwards += len(probe_pairs)
                actual, _ = forward_with_relevance(model, head, encoded)
                pair_forwards += len(probe_pairs)
            if not torch.equal(native, actual):
                raise ValueError("native inference path changed")
            result["native_forward_equivalence"] = True
            del native, actual, encoded
            phase("retention_teacher_cache")
            cache = score_rows(train[2])
            np.save(args.output / "initial-retention-scores.npy", cache, allow_pickle=False)
            save_delta("initial-delta.pt")
            if args.smoke:
                phase("longest_train_backward_smoke")
                s, _ = forward_pairs(render_pairs(longest_train))
                loss = choice_loss(
                    s,
                    [len(s)],
                    torch.tensor(
                        [longest_train["options"].index(longest_train["target"])], device="cuda"
                    ),
                )
                loss.backward()
                model.zero_grad(set_to_none=True)
                write(
                    args.output / "longest-train-probe.json",
                    dict(
                        id=longest_train["id"],
                        tokens_before_truncation=longest_train_length,
                        loss=float(loss.detach()),
                        peak_vram=torch.cuda.max_memory_allocated(),
                    ),
                )
            else:
                snapshot(0)
            optimizer = torch.optim.AdamW(
                parameters,
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            steps = 2 if args.smoke else config["steps"]
            for step in range(1, steps + 1):
                phase("training", step)
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                losses, kls, auxiliaries = [], [], []
                for source, index in order[step - 1]:
                    row = train[source][index]
                    scores, relevance = forward_pairs(render_pairs(row))
                    training_forwards += 1
                    teacher = (
                        torch.tensor(cache[index, : len(scores)], device="cuda")
                        if source == 2
                        else None
                    )
                    objective, statistics = training_objective(
                        scores, relevance, row, teacher, coefficient
                    )
                    if statistics["kl"] is not None:
                        kl = statistics["kl"]
                        if not np.isfinite(kl) or kl < -1e-5:
                            raise FloatingPointError("invalid retention KL")
                        if step == 1 and abs(kl) > 1e-5:
                            raise ValueError("initial retention cache mismatch")
                        kls.append(kl)
                    if statistics["relevance"] is not None:
                        auxiliaries.append(statistics["relevance"])
                    (objective / len(order[step - 1])).backward()
                    training_backwards += 1
                    losses.append(statistics["ce"])
                norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        parameters, config["gradient_clip"], error_if_nonfinite=True
                    )
                )
                if norm == 0 or any(
                    p.grad is not None for p in model.parameters() if not p.requires_grad
                ):
                    raise ValueError("gradient scope failure")
                auxiliary_norm = float(
                    torch.linalg.vector_norm(
                        torch.cat([p.grad.flatten() for p in head.parameters()])
                    )
                )
                if coefficient == 0 and auxiliary_norm != 0:
                    raise ValueError("zero auxiliary coefficient produced a gradient")
                lr = learning_rate(
                    step, config["steps"], config["warmup_updates"], config["learning_rate"]
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                completed_updates = step
                seconds = time.perf_counter() - started
                training_seconds += seconds
                update = dict(
                    step=step,
                    ce=float(np.mean(losses)),
                    kl=float(np.mean(kls)),
                    relevance_loss=float(np.mean(auxiliaries)),
                    learning_rate=lr,
                    gradient_norm=norm,
                    auxiliary_gradient_norm_after_clip=auxiliary_norm,
                    seconds=seconds,
                    consumed=step * len(order[step - 1]),
                )
                updates.append(update)
                with (args.output / "updates.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(update, allow_nan=False) + "\n")
                if step % 32 == 0:
                    print(json.dumps(update), flush=True)
                if not args.smoke and step % config["evaluation_interval"] == 0:
                    snapshot(step)
            save_delta("final-delta.pt")
        if args.smoke:
            phase("smoke_save_reload", 2)
            probe_rows = [train[p][i] for p, i in order[0]]
            before = score_rows(probe_rows)
            load_delta(args.output / "final-delta.pt")
            after = score_rows(probe_rows)
            if not np.array_equal(before, after):
                raise ValueError("smoke save/reload changed option logits")
            result.update(
                status="PASS_TRAIN_ONLY_RELEVANCE_SMOKE",
                steps=2,
                train_probe_reload_exact=True,
                consumed=accounting([train[s][i] for b in order[:2] for s, i in b]),
                measured_seconds_per_update=float(np.mean([r["seconds"] for r in updates])),
                projected_train_seconds=config["steps"]
                * float(np.mean([r["seconds"] for r in updates])),
            )
        else:
            selected_dir = args.replay_reference if reference else args.output
            if reference:
                history = read(selected_dir / "history.json")
                baseline = history[0]["metrics"]
                best_step = select_validation(history, baseline, schema)
                if best_step != reference["best_step"]:
                    raise ValueError("reference selector mismatch")
            selected = selected_dir / "best-delta.pt"
            selected_hash = sha(selected)
            load_delta(selected)
            phase("selected_reload", best_step)
            exact = []
            for label, rows, expected_schema in (
                ("validation", validation, schema),
                ("fixed-train", fixed_train, fixed_schema),
            ):
                actual, _ = evaluate(rows, expected_schema, "reload-" + label)
                expected = np.load(
                    selected_dir / f"{label}-{best_step}-scores.npy", allow_pickle=False
                )
                if not np.array_equal(actual, expected):
                    raise ValueError(f"selected reload is not exact: {label}")
                exact.append(label)
            if sha(selected) != selected_hash:
                raise ValueError("selected weights changed")
            if not reference:
                new_count = len(data["validation"])
                base_scores = np.load(args.output / "validation-0-scores.npy", allow_pickle=False)[
                    :new_count
                ]
                intervals = {}
                for label, step in (("selected", best_step), ("endpoint", config["steps"])):
                    candidate = np.load(
                        args.output / f"validation-{step}-scores.npy", allow_pickle=False
                    )[:new_count]
                    intervals[label] = bootstrap_relevance(
                        data["validation"], base_scores, candidate
                    )
                write(args.output / "uncertainty.json", intervals)
                selected_interval = intervals["selected"]
                result["within_arm_validation_advancement"] = bool(
                    best_step > 0
                    and selected_interval["difference"] >= 0.03 - 1e-12
                    and selected_interval["interval95"][0] > 0
                )
            result.update(
                status="PASS_SELECTED_RELEVANCE_REPLAY"
                if reference
                else "PASS_MATCHED_RELEVANCE_FIT",
                steps=0 if reference else config["steps"],
                best_step=best_step,
                selected_sha256=selected_hash,
                selected_reload_exact=exact,
                consumed=accounting([] if reference else ordered_rows),
                selected_consumed=accounting(
                    [train[s][i] for b in order[:best_step] for s, i in b]
                ),
            )
        if frozen_hash(model) != frozen:
            raise ValueError("frozen parameter integrity failure")
        verify(pins)
        verify(code_pins)
        guard()
        result.update(
            finished_utc=datetime.now(UTC).isoformat(),
            seconds=time.perf_counter() - began,
            input_pins=pins,
            code_pins=code_pins,
            frozen_sha256=frozen,
            frozen_and_sources_unchanged=True,
            trainable_parameters=config["trainable_parameters"],
            total_parameters=config["total_parameters"],
            auxiliary_training_only_parameters=config["auxiliary_parameters"],
            optimization_parameters=config["trainable_parameters"] + config["auxiliary_parameters"],
            training_seconds=training_seconds,
            evaluation_seconds=evaluation_seconds,
            candidate_pair_forwards=pair_forwards,
            peak_cuda_allocated=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved=torch.cuda.max_memory_reserved(),
            peak_host_bytes=getattr(psutil.Process().memory_info(), "peak_wset", None),
            free_disk_bytes=shutil.disk_usage(args.output).free,
            artifacts={
                p.name: sha(p)
                for p in args.output.iterdir()
                if p.is_file() and p.name not in ("result.json", "active.json")
            },
        )
        write(args.output / "result.json", result)
        phase("complete", result["steps"])
        guard()
        print(
            json.dumps(
                {
                    k: result.get(k)
                    for k in (
                        "status",
                        "steps",
                        "best_step",
                        "within_arm_validation_advancement",
                        "seconds",
                    )
                }
            ),
            flush=True,
        )
    except Exception:
        failure_traceback = traceback.format_exc()
        result.update(
            status="INCOMPLETE",
            traceback=failure_traceback,
            failure_phase=failure_phase,
            input_pins=pins,
            code_pins=code_pins,
            actual_progress=dict(
                optimizer_updates_completed=completed_updates,
                completed_question_backwards=training_backwards,
                consumed=accounting([train[s][i] for b in order for s, i in b][:training_forwards]),
                partial_update_forwards=training_forwards
                - sum(map(len, order[:completed_updates])),
            ),
            successful_candidate_pair_forwards=pair_forwards,
            seconds=time.perf_counter() - began,
            finished_utc=datetime.now(UTC).isoformat(),
        )
        if resource_snapshot is not None:
            result.update(resource_snapshot())
        result["artifacts"] = {
            p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != "result.json"
        }
        write(args.output / "result.json", result)
        raise


if __name__ == "__main__":
    main()
