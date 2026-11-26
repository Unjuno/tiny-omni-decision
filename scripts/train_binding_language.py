"""One fixed instruction-diversity comparison; all other training conditions retained."""

import argparse
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

from scripts.build_condition_binding_corpus import lines
from scripts.condition_binding_corpus import retention_ok, select_checkpoint
from scripts.condition_binding_retention import checked_logits, retention_kl
from scripts.condition_binding_training import pack_scores, score_summary
from scripts.evaluate_semantic_state_bridge import read, sha, verify, write
from scripts.prepare_binding_language import CONFIG, validate_corpus
from scripts.semantic_state_diagnostic import log_probabilities

CORPUS_SHA = "957e2ce3ade65c5d2caa26a1cbb0e95a3adeee5cf5b312110f8c2a7e3cfed971"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--replay-reference", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if bool(args.replay_reference) != bool(args.replay_sha256) or (
        args.smoke and args.replay_reference
    ):
        raise ValueError("invalid mode")
    args.output.mkdir(parents=True, exist_ok=False)
    tick = time.perf_counter()
    result = dict(
        status="RUNNING",
        command=sys.argv,
        started_utc=datetime.now(UTC).isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        smoke=args.smoke,
        exact_resume=False,
        optimizer_reset=not bool(args.replay_reference),
        cloud_cost=0,
        sealed_audit_loaded=False,
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
        from scripts.train_text_crossencoder import choice_loss, frozen_hash

        config = read(CONFIG)
        root, base = Path(config["root"]), Path(config["base_directory"])
        original_config = read("configs/pretrained_reuse/condition_binding_parent_kl_v1.json")
        extra = {
            "experiment_id",
            "language_policy",
            "fit_result_sha256",
            "old_corpus_sha256",
            "previous_transfer_corpus_sha256",
            "question_batch_size",
            "max_tokens",
        }
        if {k: v for k, v in config.items() if k not in extra} != {
            k: v for k, v in original_config.items() if k != "experiment_id"
        }:
            raise ValueError("unplanned training difference")
        pins, code_pins = validate_corpus(
            args.corpus, args.corpus_sha256, config, training_only=True
        )
        control_path = root / "condition-binding-retention-fit-01"
        corpus = root / "condition-binding-corpus-01"
        reference = None
        if args.replay_reference:
            verify({str(args.replay_reference / "result.json"): args.replay_sha256})
            reference = read(args.replay_reference / "result.json")
            if reference["status"] != "PASS_FIXED_BINDING_LANGUAGE":
                raise ValueError("only a completed primary run can be replayed")
            verify({str(args.replay_reference / p): h for p, h in reference["artifacts"].items()})
            if reference["input_pins"] != pins or reference["code_pins"] != code_pins:
                raise ValueError("replay provenance mismatch")

        def guard():
            memory = psutil.Process().memory_info()
            size = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
            if (
                time.perf_counter() - tick > config["max_seconds"]
                or getattr(memory, "peak_wset", memory.rss) > config["max_host_bytes"]
                or torch.cuda.max_memory_allocated() > config["max_vram_bytes"]
                or size > config["max_output_bytes"]
                or shutil.disk_usage(args.output).free < config["min_free_disk_bytes"]
            ):
                raise RuntimeError("resource guard; experiment incomplete")

        torch.set_num_threads(4)
        torch.manual_seed(17)
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
        initial_path = root / "text-coverage-training-01/best-delta.pt"

        def load_delta(path):
            delta = torch.load(path, map_location="cpu", weights_only=True)
            if set(delta) != set(names):
                raise ValueError("delta target coverage changed")
            loaded = model.load_state_dict(delta, strict=False)
            if loaded.unexpected_keys or set(loaded.missing_keys) != set(model.state_dict()) - set(
                names
            ):
                raise ValueError("delta load mismatch")

        load_delta(initial_path)
        parameters = [p for p in model.parameters() if p.requires_grad]
        if (
            sum(p.numel() for p in parameters) != config["trainable_parameters"]
            or sum(p.numel() for p in model.parameters()) != config["total_parameters"]
        ):
            raise ValueError("parameter budget changed")
        frozen = frozen_hash(model)
        environment = dict(
            python=sys.version,
            torch=torch.__version__,
            numpy=np.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            packages={
                name: importlib.metadata.version(name)
                for name in ("transformers", "tokenizers", "safetensors", "psutil")
            },
            attention=model.config._attn_implementation,
            dtype=str(next(model.parameters()).dtype),
            threads=4,
        )
        write(
            args.output / "effective-config.json",
            dict(
                config=config,
                environment=environment,
                corpus_sha256=args.corpus_sha256,
                original_corpus_sha256=CORPUS_SHA,
                input_pins=pins,
                code_pins=code_pins,
                initialized_from=str(initial_path),
                frozen_parameter_sha256=frozen,
                trainable_names=names,
                smoke=args.smoke,
            ),
        )
        train = [lines(args.corpus / "train.jsonl"), lines(corpus / "replay.jsonl")]
        order = read(corpus / "sample-order.json")
        validation = (
            {}
            if args.smoke
            else {
                name: lines(corpus / f"{file}.jsonl")
                for name, file in (
                    ("binding", "validation"),
                    ("original", "original"),
                    ("curriculum", "curriculum"),
                )
            }
        )
        fixed_train = train[0][:256]
        pair_calls = 0
        truncated_presentations = 0
        truncated_ids = set()
        max_tokens = 0

        def forward(rows):
            nonlocal pair_calls, truncated_presentations, max_tokens
            guard()
            first = [
                f"Question: {r['question']}\nCandidate answer: {o}"
                for r in rows
                for o in r["options"]
            ]
            second = [f"State: {r['state']}" for r in rows for _ in r["options"]]
            lengths = tokenizer(first, second, padding=False, truncation=False)["input_ids"]
            offset = 0
            for row in rows:
                block = lengths[offset : offset + len(row["options"])]
                too_long = sum(len(v) > 512 for v in block)
                if too_long:
                    if row["source"] == "binding":
                        raise ValueError("binding truncation forbidden")
                    truncated_presentations += too_long
                    truncated_ids.add(row["id"])
                offset += len(row["options"])
            token = tokenizer(
                first,
                second,
                padding=True,
                truncation="only_second",
                max_length=512,
                return_tensors="pt",
            ).to("cuda")
            max_tokens = max(max_tokens, token["input_ids"].shape[1])
            scores = model(**token).logits[:, 0]
            if not torch.isfinite(scores).all():
                raise FloatingPointError("nonfinite model scores")
            pair_calls += len(first)
            return scores, [len(r["options"]) for r in rows]

        def evaluate(rows, name):
            values = []
            with torch.inference_mode():
                for offset in range(0, len(rows), 4):
                    scores, widths = forward(rows[offset : offset + 4])
                    values.extend(s.cpu().numpy() for s in scores.split(widths))
            packed = pack_scores(values)
            summary = score_summary(rows, packed)
            np.save(args.output / f"{name}-scores.npy", packed, allow_pickle=False)
            np.save(
                args.output / f"{name}-probabilities.npy",
                np.exp(log_probabilities(packed)),
                allow_pickle=False,
            )
            write(args.output / f"{name}-metrics.json", summary)
            return packed, summary

        def save_delta(name):
            guard()
            path = args.output / name
            temporary = args.output / (name + ".tmp")
            torch.save(
                {
                    n: p.detach().cpu().clone()
                    for n, p in model.named_parameters()
                    if p.requires_grad
                },
                temporary,
            )
            temporary.replace(path)

        teacher_cache = {}
        cache_summary = None
        if not reference:
            cached = read(control_path / "teacher-cache.json")
            required = {
                i
                for batch in order[: 2 if args.smoke else 512]
                for source, i in batch
                if source == 1
            }
            if (
                cached["summary"]["parent_sha256"] != sha(initial_path)
                or cached["summary"]["temperature"] != 1.0
            ):
                raise ValueError("cache parent or temperature mismatch")
            teacher_cache = {i: cached["entries"][str(i)] for i in required}
            for index, entry in teacher_cache.items():
                checked_logits(train[1][index], entry)
            cache_summary = dict(
                questions=len(teacher_cache),
                seconds=0.0,
                pair_presentations=0,
                reused_from=str(control_path / "teacher-cache.json"),
                source="original TRAIN only",
                teacher_gradient=False,
                parent_sha256=sha(initial_path),
                temperature=1.0,
            )
            write(
                args.output / "teacher-cache.json",
                dict(summary=cache_summary, entries=teacher_cache),
            )
            torch.manual_seed(17)

        history = []
        gradient_rows = []
        best_step = 0
        baseline_metrics, baseline_tasks = None, None

        def snapshot(step):
            nonlocal best_step, baseline_metrics, baseline_tasks
            started = time.perf_counter()
            summaries = {
                name: evaluate(rows, f"validation-{step}-{name}")[1]
                for name, rows in validation.items()
            }
            training = evaluate(fixed_train, f"fixed-train-{step}")[1]
            old = {name: summaries[name]["overall"] for name in ("original", "curriculum")}
            tasks = {
                name + ":" + task: m["accuracy"]
                for name in old
                for task, m in summaries[name]["tasks"].items()
            }
            if step == 0:
                baseline_metrics, baseline_tasks = old, tasks
                for name in validation:
                    filename = (
                        "fixed-train-0-scores.npy"
                        if name == "fixed_train"
                        else f"validation-0-{name}-scores.npy"
                    )
                    if not np.array_equal(
                        np.load(args.output / filename, allow_pickle=False),
                        np.load(control_path / filename, allow_pickle=False),
                    ):
                        raise ValueError(f"control baseline changed: {name}")
            eligible = retention_ok(old, tasks, baseline_metrics, baseline_tasks)
            used = [train[s][i] for batch in order[:step] for s, i in batch]
            row = dict(
                step=step,
                eligible=eligible,
                binding_nll=summaries["binding"]["overall"]["nll"],
                validation=summaries,
                fixed_train=training,
                consumed=len(used),
                unique_questions=len({r["id"] for r in used}),
                unique_states_by_source={
                    s: len({r["group"] for r in used if r["source"] == s})
                    for s in ("binding", "original")
                },
                source_counts=dict(Counter(r["source"] for r in used)),
                language_family_counts=dict(
                    Counter(r["language_family"] for r in used if r["source"] == "binding")
                ),
                task_counts=dict(Counter(r["source"] + ":" + r["task"] for r in used)),
                evaluation_seconds=time.perf_counter() - started,
                elapsed_seconds=time.perf_counter() - tick,
            )
            history.append(row)
            best_step = select_checkpoint(history)
            row["best_step"] = best_step
            if step == best_step:
                save_delta("best-delta.pt")
            if frozen_hash(model) != frozen:
                raise ValueError("frozen parameters changed")
            write(args.output / "history.json", history)
            print(
                json.dumps(
                    dict(
                        step=step,
                        binding_accuracy=summaries["binding"]["overall"]["accuracy"],
                        binding_nll=row["binding_nll"],
                        eligible=eligible,
                        best_step=best_step,
                        seconds=row["elapsed_seconds"],
                    )
                ),
                flush=True,
            )

        if not reference:
            optimizer = torch.optim.AdamW(
                parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"]
            )
            if not args.smoke:
                snapshot(0)
            steps = 2 if args.smoke else config["steps"]
            for step in range(1, steps + 1):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                kl_value = 0.0
                began = time.perf_counter()
                for source in range(2):
                    batch = [train[s][i] for s, i in order[step - 1] if s == source]
                    scores, widths = forward(batch)
                    target = torch.tensor(
                        [r["options"].index(r["target"]) for r in batch], device="cuda"
                    )
                    loss = choice_loss(scores, widths, target)
                    objective = loss
                    if source == 1:
                        indices = [i for s, i in order[step - 1] if s == 1]
                        teacher = torch.tensor(
                            [
                                v
                                for i, row in zip(indices, batch, strict=True)
                                for v in checked_logits(row, teacher_cache[i])
                            ],
                            device=scores.device,
                            dtype=scores.dtype,
                        )
                        kl = retention_kl(scores, widths, teacher)
                        if not torch.isfinite(kl) or kl.item() < -1e-5:
                            raise FloatingPointError("invalid retention KL")
                        kl_value = float(kl.detach())
                        if step == 1 and abs(kl_value) > 1e-6:
                            raise ValueError("initial parent/cache KL is not zero")
                        objective = objective + config["retention_kl_coefficient"] * kl
                    (0.5 * objective).backward()
                    losses.append(float(loss.detach()))
                norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        parameters, config["gradient_clip"], error_if_nonfinite=True
                    )
                )
                if not any(
                    p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in parameters
                ):
                    raise ValueError("no trainable gradient")
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise ValueError("frozen gradient")
                optimizer.step()
                gradient_rows.append(
                    dict(
                        step=step,
                        binding_ce=losses[0],
                        replay_ce=losses[1],
                        retention_kl=kl_value,
                        lr=config["learning_rate"],
                        gradient_norm=norm,
                        seconds=time.perf_counter() - began,
                    )
                )
                if step % 128 == 0 or step == steps:
                    write(args.output / "updates.json", gradient_rows)
                    if not args.smoke:
                        snapshot(step)
            save_delta("final-delta.pt")
        if args.smoke:
            if frozen_hash(model) != frozen:
                raise ValueError("smoke frozen mismatch")
            result.update(status="PASS_TRAIN_ONLY_LANGUAGE_SMOKE", steps=2)
        else:
            selected_dir = args.replay_reference if reference else args.output
            if reference:
                history = read(selected_dir / "history.json")
                best_step = reference["best_step"]
                if best_step != select_checkpoint(history):
                    raise ValueError("saved selector mismatch")
            selected_path = selected_dir / "best-delta.pt"
            selected_hash = sha(selected_path)
            load_delta(selected_path)
            reloaded = []
            for name, rows in {**validation, "fixed_train": fixed_train}.items():
                actual, _ = evaluate(rows, f"reload-{name}")
                expected_file = (
                    f"fixed-train-{best_step}-scores.npy"
                    if name == "fixed_train"
                    else f"validation-{best_step}-{name}-scores.npy"
                )
                expected = np.load(selected_dir / expected_file, allow_pickle=False)
                if not np.array_equal(actual, expected):
                    raise ValueError(f"selected reload mismatch: {name}")
                reloaded.append(expected_file)
            replay_exact = []
            if reference:
                for name in (*validation, "fixed_train"):
                    for suffix in ("scores.npy", "probabilities.npy", "metrics.json"):
                        file = f"reload-{name}-{suffix}"
                        if sha(args.output / file) != reference["artifacts"][file]:
                            raise ValueError(f"fresh selected replay mismatch: {file}")
                        replay_exact.append(file)
            if sha(selected_path) != selected_hash or frozen_hash(model) != frozen:
                raise ValueError("selected or frozen weights changed")
            result.update(
                status="PASS_SELECTED_LANGUAGE_REPLAY"
                if reference
                else "PASS_FIXED_BINDING_LANGUAGE",
                steps=0 if reference else 512,
                best_step=best_step,
                selected_sha256=selected_hash,
                selected_reload_exact=reloaded,
                fresh_replay_exact=replay_exact,
                transfer_evaluated=False,
            )
        result.update(teacher_cache=cache_summary)
        verify(pins)
        verify(code_pins)
        guard()
        result.update(
            finished_utc=datetime.now(UTC).isoformat(),
            seconds=time.perf_counter() - tick,
            input_pins=pins,
            code_pins=code_pins,
            sources_unchanged=True,
            frozen_parameter_sha256=frozen,
            candidate_pair_forward_presentations=pair_calls,
            truncated_pair_presentations=truncated_presentations,
            unique_truncated_question_ids=sorted(truncated_ids),
            max_tokens=max_tokens,
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
            json.dumps({k: result.get(k) for k in ("status", "steps", "best_step", "seconds")}),
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
