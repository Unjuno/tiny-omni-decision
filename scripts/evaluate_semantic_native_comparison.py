"""Prospective observed-image comparison; execute only after the semantic fit is verified."""

import argparse
import hashlib
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

from scripts.evaluate_semantic_state_bridge import read, sha, tensor_digest, verify, write
from scripts.semantic_native_comparison import (
    attribution,
    compare_public_pairs,
    paired_accuracy_interval,
    permutation_check,
    replay_file_names,
)
from scripts.semantic_state_diagnostic import (
    FIELDS,
    QUESTIONS,
    log_probabilities,
    metrics,
    options,
    summarize,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/pretrained_reuse/semantic_native_comparison_v1.json"


def tracked_source_names(repo, config_path, protocol_path):
    # Return the inventory on stdout, never as thousands of command-line arguments.
    # NUL separation also preserves spaces and non-ASCII names without Git quoting.
    tracked = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z"], text=True, encoding="utf-8"
    ).split("\0")
    tracked = [name for name in tracked if name]
    required = [
        "scripts/evaluate_semantic_native_comparison.py",
        "scripts/semantic_native_comparison.py",
        config_path,
        "pyproject.toml",
        protocol_path,
    ]
    if not set(required).issubset(tracked):
        raise ValueError("producer, config and protocol must be tracked and frozen")
    return [name for name in tracked if name.endswith(".py")] + required[2:]


def admit_native_sources(root, pins):
    old = root / "semantic-state-bridge-01"
    payload_names = (
        "selection.json",
        "states.json",
        "targets.npy",
        "native-logits.npy",
        "effective-config.json",
    )
    required = [
        old / "result.json",
        *[old / n for n in payload_names],
        root / "semantic-native-bridge-admission-01.json",
        root / "semantic-native-bridge-state-check-01.json",
        root / "semantic-native-control-inputs-01.json",
    ]
    if any(str(p) not in pins for p in required):
        raise ValueError("unpinned native input")
    verify({str(p): pins[str(p)] for p in required})
    historical = read(old / "result.json")
    if historical["status"] != "PASS_FIXED_TRAIN_DIAGNOSIS" or historical["smoke"]:
        raise ValueError("completed original diagnostic required")
    for name in payload_names:
        if pins[str(old / name)] != historical["artifacts"][name]:
            raise ValueError("old native payload differs from historical result")
    admission = read(root / "semantic-native-bridge-admission-01.json")
    exclusion = read(root / "semantic-native-bridge-state-check-01.json")
    control = read(root / "semantic-native-control-inputs-01.json")
    if (
        admission["status"] != "PASS_EXISTING_DIAGNOSTIC_NATIVE_ADMISSION"
        or not admission["all_selected_states_in_frozen_historical_exclusions"]
        or exclusion["status"] != "PASS_NATIVE_BRIDGE_STATE_EXCLUSION_CHECK"
        or exclusion["states_outside_historical_exclusions"] != 0
        or not exclusion["predicted_states_match_native_argmax"]
        or exclusion["annotation_targets_checked"] != 1024
        or control["status"] != "PASS_FIXED_NATIVE_CONTROL_INPUTS"
    ):
        raise ValueError("native admission/exclusion/control prerequisites failed")
    for record in (admission, exclusion, control):
        verify(record["input_pins"])
    verify(admission["media_pins"])
    selection, states = read(old / "selection.json"), read(old / "states.json")
    target = np.load(old / "targets.npy", allow_pickle=False)
    native = np.load(old / "native-logits.npy", allow_pickle=False)
    if (
        len(selection) != 256
        or len({r["label"]["id"] for r in selection}) != 256
        or len({r["media"]["sha256"] for r in selection}) != 256
        or any(r["label"]["split"] != "train" or r["media"]["split"] != "train" for r in selection)
        or any(len(states[k]) != 256 for k in ("oracle", "predicted"))
        or target.shape != (256, 4)
        or native.shape != (256, 4, 10)
    ):
        raise ValueError("fixed native input dimensions/identity changed")
    def canonical(state):
        return json.dumps(state, sort_keys=True, separators=(",", ":"))
    oracle_counts = Counter(canonical(s) for s in states["oracle"])
    if (
        len(oracle_counts) != 252
        or Counter(oracle_counts.values()) != {1: 248, 2: 4}
        or len({canonical(s) for s in states["predicted"]}) != 254
        or len({canonical(s) for kind in states.values() for s in kind}) != 308
        or int((target != np.roll(target, -1, axis=0)).sum()) != 912
        or control["donor_changed_question_count"] != 912
        or control["annotation_state_clusters"] != 252
    ):
        raise ValueError("frozen native state/control accounting changed")
    return dict(
        selection=selection,
        states=states,
        target=target,
        native=native,
        effective=read(old / "effective-config.json"),
        admission=admission,
        dependencies=[record["input_pins"] for record in (admission, exclusion, control)]
        + [admission["media_pins"]],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replay-reference", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if bool(args.replay_reference) != bool(args.replay_sha256) or (
        args.smoke and args.replay_reference
    ):
        raise ValueError("invalid mode")
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    config = read(CONFIG)
    result = dict(
        status="RUNNING",
        started_utc=datetime.now(UTC).isoformat(),
        pid=os.getpid(),
        command=sys.argv,
        smoke=args.smoke,
        optimizer_updates=0,
        cloud_cost=0,
        transfer_loaded=False,
        sealed_audit_loaded=False,
        product_complete=False,
    )
    write(args.output / "result.json", result)
    pins, code = {}, {}
    dependency_pins = []
    total_pairs, max_tokens = 0, 0
    current_arm, current_condition = None, "prerequisites"
    try:
        # Verify terminal prerequisites before importing torch or initializing CUDA.
        pins = config["input_pins"]
        verify(pins)
        root = Path(config["artifact_root"])
        primary = root / "decision-semantic-fit-01"
        replay = root / "decision-semantic-selected-replay-01"
        independent = root / "decision-semantic-independent-verification-01.json"
        fit, selected_replay, verification = (
            read(primary / "result.json"),
            read(replay / "result.json"),
            read(independent),
        )
        for path in (primary / "result.json", replay / "result.json", independent):
            if str(path) not in pins:
                raise ValueError("terminal prerequisite is not pinned")
        if (
            fit["status"] != "PASS_FIXED_SEMANTIC_FIT"
            or fit["steps"] != 1024
            or selected_replay["status"] != "PASS_SELECTED_SEMANTIC_REPLAY"
            or verification["status"] != "PASS_INDEPENDENT_SEMANTIC_VERIFICATION"
            or verification["result_sha256"] != sha(primary / "result.json")
            or verification["replay_sha256"] != sha(replay / "result.json")
        ):
            raise ValueError(
                "completed fit, fresh selected replay and independent verification required"
            )
        for folder, terminal in ((primary, fit), (replay, selected_replay)):
            if read(folder / "active.json")["phase"] != "complete":
                raise ValueError("prerequisite process is not terminal")
            artifact_pins = {str(folder / name): h for name, h in terminal["artifacts"].items()}
            verify(artifact_pins)
            dependency_pins.append(artifact_pins)
        admitted = admit_native_sources(root, pins)
        dependency_pins.extend(admitted["dependencies"])
        subprocess.run(
            ["git", "-C", str(REPO), "diff", "--exit-code", "HEAD"], check=True, capture_output=True
        )
        source_commit = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
        names = tracked_source_names(
            REPO, CONFIG.relative_to(REPO).as_posix(), config["protocol_path"]
        )
        code = {str(REPO / name): sha(REPO / name) for name in names}
        result["source_commit"] = source_commit
        previous = None
        if args.replay_reference:
            verify({str(args.replay_reference / "result.json"): args.replay_sha256})
            previous = read(args.replay_reference / "result.json")
            if previous["status"] != "PASS_FIXED_SEMANTIC_NATIVE_COMPARISON" or previous["smoke"]:
                raise ValueError("completed primary native comparison required")
            if previous["input_pins"] != pins or previous["code_pins"] != code:
                raise ValueError("fresh native replay pins differ")
            verify(
                {str(args.replay_reference / name): h for name, h in previous["artifacts"].items()}
            )

        import psutil

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(4)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

        def guard():
            memory = psutil.Process().memory_info()
            size = sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())
            if (
                time.perf_counter() - began > config["max_seconds"]
                or getattr(memory, "peak_wset", memory.rss) > config["max_host_bytes"]
                or torch.cuda.max_memory_allocated() > config["max_vram_bytes"]
                or size > config["max_output_bytes"]
                or shutil.disk_usage(args.output).free < config["min_free_disk_bytes"]
            ):
                raise RuntimeError("resource guard; incomplete comparison")

        selection, states = admitted["selection"], admitted["states"]
        old_effective = admitted["effective"]
        if old_effective["questions"] != list(QUESTIONS) or old_effective["options"] != options():
            raise ValueError("original public questions/options changed")
        n = config["smoke_images"] if args.smoke else config["images"]
        if len(selection) != 256 or config["images"] != 256 or not 2 <= n <= 256:
            raise ValueError("fixed cohort required")
        selection = selection[:n]
        states = {k: v[:n] for k, v in states.items()}
        target = admitted["target"][:n].copy()
        native = admitted["native"][:n].copy()
        if target.shape != (n, 4) or native.shape != (n, 4, 10):
            raise ValueError("old native alignment mismatch")
        admission = admitted["admission"]
        if (
            admission["status"] != "PASS_EXISTING_DIAGNOSTIC_NATIVE_ADMISSION"
            or not admission["all_selected_states_in_frozen_historical_exclusions"]
        ):
            raise ValueError("native source admission missing")
        verify(admission["media_pins"])
        # Validate no label routing: ground truth below is for metrics only.
        descriptions, option_ids = (
            [o["description"] for o in options()],
            [o["id"] for o in options()],
        )
        for i in range(n):
            for q, field in enumerate(FIELDS):
                if option_ids[target[i, q]] != field + ":" + str(
                    selection[i]["label"][field]
                ) or states["oracle"][i][field] != str(selection[i]["label"][field]):
                    raise ValueError("annotation identity mismatch")
                field_ids = option_ids[q * 10 : (q + 1) * 10]
                if (
                    states["predicted"][i][field]
                    != field_ids[int(native[i, q].argmax())].split(":", 1)[1]
                ):
                    raise ValueError("predicted state does not match native features")
        write(args.output / "selection.json", selection)
        write(args.output / "states.json", states)
        np.save(args.output / "targets.npy", target, allow_pickle=False)
        np.save(args.output / "native-logits.npy", native, allow_pickle=False)
        groups = [
            hashlib.sha256(
                json.dumps(s, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for s in states["oracle"]
        ]

        model_path = Path(fit["config"]["model"])
        model_pins = {p: h for p, h in fit["input_pins"].items() if str(model_path) in p}
        verify(model_pins)
        dependency_pins.append(model_pins)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        model, info = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        if any(info.values()):
            raise ValueError("pretrained loading mismatch")
        model = model.cuda().eval().requires_grad_(False)
        if sum(p.numel() for p in model.parameters()) != 70830722:
            raise ValueError("model parameter count changed")
        trainable = set(read(primary / "effective-config.json")["trainable_names"])
        if len(trainable) != 199:
            raise ValueError("unexpected trained scope")

        candidates = [
            ("step0", "step-0-delta.pt"),
            ("endpoint", "step-1024-delta.pt"),
            ("selected", "best-delta.pt"),
        ]
        aliases, fingerprints, arms = {}, {}, []
        for name, filename in candidates:
            path = primary / filename
            verify({str(path): fit["artifacts"][filename]})
            delta = torch.load(path, map_location="cpu", weights_only=True)
            if set(delta) != trainable or sum(v.numel() for v in delta.values()) != 21639554:
                raise ValueError("checkpoint coverage changed")
            digest = hashlib.sha256()
            for key, value in sorted(delta.items()):
                if value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise ValueError("checkpoint dtype/finite mismatch")
                digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
                digest.update(value.contiguous().numpy().tobytes())
            fingerprint = digest.hexdigest()
            if fingerprint in fingerprints:
                aliases[name] = fingerprints[fingerprint]
            else:
                fingerprints[fingerprint] = name
                aliases[name] = name
                arms.append((name, path))
            del delta
        effective = dict(
            config=config,
            input_pins=pins,
            code_pins=code,
            aliases=aliases,
            checkpoint_tensor_fingerprints=fingerprints,
            questions=list(QUESTIONS),
            options=options(),
            selection_sha256=sha(args.output / "selection.json"),
            parameters=70830722,
            trainable_parameters=0,
            dtype="float32",
            attention="eager",
            base_revision=fit["config"]["revision"],
            selected_step=fit["best_step"],
            native_features_reused=True,
            fresh_image_encoder_forward=False,
            environment=dict(
                python=sys.version,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),
                packages={
                    k: importlib.metadata.version(k)
                    for k in ("transformers", "numpy", "tokenizers", "safetensors", "psutil")
                },
            ),
        )
        write(args.output / "effective-config.json", effective)

        def forward(pairs, batch_size=32):
            nonlocal total_pairs, max_tokens
            outputs = []
            with torch.inference_mode():
                for offset in range(0, len(pairs), batch_size):
                    guard()
                    batch = pairs[offset : offset + batch_size]
                    token = tokenizer(
                        [x[0] for x in batch],
                        [x[1] for x in batch],
                        padding=True,
                        truncation=False,
                        return_tensors="pt",
                    )
                    max_tokens = max(max_tokens, token["input_ids"].shape[1])
                    if max_tokens > 512:
                        raise ValueError("truncation forbidden")
                    scores = model(**token.to("cuda")).logits[:, 0].cpu().numpy()
                    if not np.isfinite(scores).all():
                        raise FloatingPointError("nonfinite scores")
                    outputs.append(scores)
                    total_pairs += len(batch)
            return np.concatenate(outputs).astype(np.float64)

        def pairs_for(facts, questions, choices=descriptions):
            return [
                p
                for state in facts
                for question in questions
                for p in compare_public_pairs(dict(state=state, question=question, options=choices))
            ]

        summaries, all_arrays, controls = {}, {}, {}
        for name, path in arms:
            current_arm, current_condition = name, "load_checkpoint"
            write(args.output / "active.json", dict(phase="inference", arm=name, pid=os.getpid()))
            verify({str(path): fit["artifacts"][path.name]})
            delta = torch.load(path, map_location="cpu", weights_only=True)
            loaded = model.load_state_dict(delta, strict=False)
            if (
                loaded.unexpected_keys
                or set(loaded.missing_keys) != set(model.state_dict()) - trainable
            ):
                raise ValueError("strict trained checkpoint loading failed")
            del delta
            before = tensor_digest(model)
            arrays = {}
            for kind in ("oracle", "predicted"):
                current_condition = kind
                arrays[kind] = forward(pairs_for(states[kind], QUESTIONS)).reshape(n, 4, 40)
            current_condition = "no_evidence"
            arrays["no_evidence"] = np.repeat(
                forward(pairs_for([{}], QUESTIONS)).reshape(1, 4, 40), n, axis=0
            )
            current_condition = "no_question"
            arrays["no_question"] = np.repeat(
                forward(pairs_for(states["oracle"], [""])).reshape(n, 1, 40), 4, axis=1
            )
            arrays["donor"] = np.roll(arrays["oracle"], -1, axis=0).copy()
            donor_target = np.roll(target, -1, axis=0)
            permutation = np.random.default_rng(17).permutation(40)
            current_condition = "permutation_probe"
            permuted = forward(
                pairs_for(states["oracle"][:1], QUESTIONS, [descriptions[i] for i in permutation])
            ).reshape(1, 4, 40)
            order_check = permutation_check(
                arrays["oracle"][:1],
                permuted,
                permutation,
                config["probe_score_absolute_tolerance"],
            )
            current_condition = "singleton_probe"
            single = forward(pairs_for(states["oracle"][:1], QUESTIONS[:1]), 1)
            permutation_diff = order_check["max_abs"]
            singleton_diff = float(abs(single - arrays["oracle"][0, 0]).max())
            if (
                max(permutation_diff, singleton_diff) > config["probe_score_absolute_tolerance"]
                or single.argmax() != arrays["oracle"][0, 0].argmax()
            ):
                raise ValueError("candidate-order or singleton probe mismatch")
            controls[name] = dict(
                permutation_score_max_abs=permutation_diff,
                singleton_score_max_abs=singleton_diff,
                predictions_exact=True,
                all_candidates_retained=True,
            )
            summary = {kind: summarize(scores, target) for kind, scores in arrays.items()}
            summary["donor_corrected"] = summarize(arrays["donor"], donor_target)
            summary["donor_target_changed_questions"] = int((target != donor_target).sum())
            summary["error_attribution"] = attribution(
                native, arrays["oracle"], arrays["predicted"], target
            )
            current_condition = "save_payloads"
            if (
                summary["no_question"]["overall"]["accuracy"] > 0.25
                or summary["no_question"]["changed_question_pair_both_correct"] != 0
            ):
                raise ValueError("question-free control violation")
            for kind, scores in arrays.items():
                np.save(args.output / f"{name}-{kind}-scores.npy", scores, allow_pickle=False)
                probability = np.exp(log_probabilities(scores))
                with (args.output / f"{name}-{kind}-predictions.jsonl").open(
                    "w", encoding="utf-8"
                ) as f:
                    for i, asset in enumerate(selection):
                        for q, field in enumerate(FIELDS):
                            p = int(probability[i, q].argmax())
                            payload = dict(
                                id=asset["label"]["id"] + ":" + field,
                                state_group=groups[i],
                                option_ids=option_ids,
                                target=int(target[i, q]),
                                prediction=p,
                                predicted_id=option_ids[p],
                                logits=scores[i, q].tolist(),
                                probabilities=probability[i, q].tolist(),
                            )
                            if kind == "donor":
                                payload["donor_corrected_target"] = int(donor_target[i, q])
                            f.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
            if tensor_digest(model) != before or any(
                p.requires_grad or p.grad is not None for p in model.parameters()
            ):
                raise ValueError("inference changed parameters or gradients")
            summaries[name], all_arrays[name] = summary, arrays
            print(
                json.dumps(
                    dict(
                        arm=name,
                        oracle_accuracy=summary["oracle"]["overall"]["accuracy"],
                        predicted_accuracy=summary["predicted"]["overall"]["accuracy"],
                    )
                ),
                flush=True,
            )
        if total_pairs != len(arms) * (n * 360 + 160 + 200):
            raise ValueError("candidate forward accounting mismatch")
        intervals = {
            kind: paired_accuracy_interval(
                all_arrays[aliases["step0"]][kind],
                all_arrays[aliases["endpoint"]][kind],
                target,
                groups,
            )
            for kind in ("oracle", "predicted")
        }
        write(
            args.output / "metrics.json",
            dict(
                arms=summaries,
                aliases=aliases,
                controls=controls,
                endpoint_minus_step0=intervals,
                native_ten_option={
                    field: metrics(native[:, i], target[:, i] % 10)
                    for i, field in enumerate(FIELDS)
                },
            ),
        )
        verify(pins)
        verify(code)
        for dependencies in dependency_pins:
            verify(dependencies)
        guard()
        replay_files = []
        if previous:
            actual_names = [
                p.name
                for p in args.output.iterdir()
                if p.is_file() and p.name not in ("result.json", "active.json")
            ]
            for name in replay_file_names(actual_names, previous["artifacts"]):
                path = args.output / name
                if sha(path) != previous["artifacts"].get(path.name):
                    raise ValueError("fresh full replay mismatch: " + path.name)
                replay_files.append(path.name)
        result.update(
            status="PASS_SEMANTIC_NATIVE_SMOKE"
            if args.smoke
            else "PASS_FIXED_SEMANTIC_NATIVE_COMPARISON",
            finished_utc=datetime.now(UTC).isoformat(),
            seconds=time.perf_counter() - began,
            input_pins=pins,
            code_pins=code,
            consumed_dependency_pins=dependency_pins,
            aliases=aliases,
            images=n,
            questions=n * 4,
            annotation_state_clusters=len(set(groups)),
            actual_candidate_pairs=total_pairs,
            max_tokens=max_tokens,
            peak_cuda_allocated=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved=torch.cuda.max_memory_reserved(),
            peak_host_bytes=getattr(psutil.Process().memory_info(), "peak_wset", None),
            free_disk_bytes=shutil.disk_usage(args.output).free,
            all_model_parameters_frozen=True,
            observed_train_diagnostic=True,
            full_omni_quality=False,
            fresh_replay_files_exact=replay_files,
            artifacts={
                p.name: sha(p)
                for p in args.output.iterdir()
                if p.is_file() and p.name not in ("result.json", "active.json")
            },
        )
        write(args.output / "result.json", result)
        write(args.output / "active.json", dict(phase="complete", pid=os.getpid()))
        guard()
        print(
            json.dumps(
                {k: result[k] for k in ("status", "images", "seconds", "actual_candidate_pairs")}
            ),
            flush=True,
        )
    except Exception:
        result.update(
            status="INCOMPLETE",
            traceback=traceback.format_exc(),
            input_pins=pins,
            code_pins=code,
            consumed_dependency_pins=dependency_pins,
            seconds=time.perf_counter() - began,
            current_arm=current_arm,
            failure_phase=current_condition,
            successful_candidate_pairs=total_pairs,
            max_tokens=max_tokens,
            artifacts={
                p.name: sha(p)
                for p in args.output.iterdir()
                if p.is_file() and p.name != "result.json"
            },
        )
        if "psutil" in locals():
            memory = psutil.Process().memory_info()
            result["peak_host_bytes"] = getattr(memory, "peak_wset", memory.rss)
        if "torch" in locals() and torch.cuda.is_initialized():
            result["peak_cuda_allocated"] = torch.cuda.max_memory_allocated()
            result["peak_cuda_reserved"] = torch.cuda.max_memory_reserved()
        result["free_disk_bytes"] = shutil.disk_usage(args.output).free
        write(args.output / "result.json", result)
        raise


if __name__ == "__main__":
    main()
