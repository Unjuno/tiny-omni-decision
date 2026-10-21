"""Fit the conditional AV readout ablation on the frozen synthetic CLEVRER bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.build_clevrer_synthetic_av_timing_features import (
    TEXT_MODEL_SHA256,
    TEXT_PARAMETER_COUNT,
    sha256_file,
)
from scripts.train_clevrer_synthetic_av_temporal_decision import (
    _gpu_gate,
    _load_inputs,
    _metrics,
    _prediction_rows,
    _read_prediction_jsonl,
    _scene_pair_metrics,
)
from tiny_omni_decision.native_av_interaction import ConditionalAVReadout
from tiny_omni_decision.native_candidate import NativeCandidateReadout

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = ("sound before collision", "sound after collision")
TIMING_FEATURES = ROOT / "scripts" / "build_clevrer_synthetic_av_timing_features.py"
V2_CONFIG = ROOT / "configs" / "pretrained_reuse" / "clevrer_synthetic_av_temporal_decision_v2.json"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/pretrained_reuse/conditional_clevrer_synthetic_av_readout_v1.json",
    )
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--text-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--max-updates", type=int, help="test-only shorter budget; recorded in effective config"
    )
    return parser.parse_args()


def _make_positions(batch: int, tokens: int) -> torch.Tensor:
    timepoints = torch.linspace(0, 1, tokens, dtype=torch.float32)
    positions = torch.zeros(tokens, 3, dtype=torch.float32)
    positions[:, 2] = timepoints
    return positions.unsqueeze(0).expand(batch, -1, -1).contiguous()


def _batch_blocks(audio: torch.Tensor, question: torch.Tensor, video: torch.Tensor):
    if audio.ndim != 3 or audio.shape[-1] != 1:
        raise ValueError("synthetic audio energy must have shape [batch, tokens, 1]")
    if question.ndim != 2 or question.shape[-1] != 384:
        raise ValueError("question embeddings must have shape [batch, 384]")
    if video.ndim != 3 or video.shape[-1] != 768 or video.shape[:2] != audio.shape[:2]:
        raise ValueError("video tokens must align with audio tokens and have width 768")
    if len(question) != len(audio):
        raise ValueError("question and AV batch sizes differ")
    batch, tokens = audio.shape[:2]
    event_audio = torch.zeros(batch, tokens, 384, dtype=audio.dtype, device=audio.device)
    event_audio[..., 0] = audio[..., 0]
    audio_positions = _make_positions(batch, tokens).to(audio.device, dtype=audio.dtype)
    video_positions = _make_positions(batch, tokens).to(video.device, dtype=video.dtype)
    text_positions = torch.zeros(batch, 1, 3, dtype=question.dtype, device=question.device)
    return {
        "text": (
            question.unsqueeze(1),
            text_positions,
            torch.ones(batch, 1, dtype=torch.bool, device=question.device),
        ),
        "video": (
            video,
            video_positions,
            torch.ones(batch, tokens, dtype=torch.bool, device=video.device),
        ),
        "event_audio": (
            event_audio,
            audio_positions,
            torch.ones(batch, tokens, dtype=torch.bool, device=audio.device),
        ),
    }


def _encode_options(model_dir: Path) -> tuple[torch.Tensor, dict[str, str | int]]:
    weight_path = model_dir / "model.safetensors"
    if sha256_file(weight_path) != TEXT_MODEL_SHA256:
        raise ValueError("candidate MiniLM checkpoint does not match the pinned feature encoder")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_dir, local_files_only=True).cpu().eval()
    parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
    if parameter_count != TEXT_PARAMETER_COUNT:
        raise ValueError(f"unexpected candidate encoder parameter count: {parameter_count}")
    tokens = tokenizer(
        list(OPTIONS), padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    with torch.inference_mode():
        hidden = encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
    return embeddings.float(), {
        "model_id": "sentence-transformers/paraphrase-MiniLM-L3-v2",
        "revision": "4ca70771034acceecb2e72475f72050fcdde4ddc",
        "model_weight_sha256": TEXT_MODEL_SHA256,
        "parameter_count": parameter_count,
    }


def _baseline_batched(model: NativeCandidateReadout, blocks, queries) -> torch.Tensor:
    memories, masks = [], []
    for kind, (values, positions, valid) in blocks.items():
        kind_index = list(model.projections).index(kind)
        memories.append(
            model.projections[kind](values)
            + model.position(positions)
            + model.modality.weight[kind_index]
        )
        masks.append(valid)
    memory = model.memory_norm(torch.cat(memories, dim=1))
    mask = torch.cat(masks, dim=1)
    query = model.query(queries)
    attended = model.attention(query, memory, memory, key_padding_mask=~mask, need_weights=False)[0]
    scores = model.score(
        torch.cat([query, attended, query * attended, (query - attended).abs()], -1)
    ).squeeze(-1)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("native baseline produced nonfinite scores")
    return scores


def _paired_scene_deltas(rows, candidate_logits, baseline_logits, targets, *, repetitions, seed):
    scenes = sorted({str(row["scene_group_id"]) for row in rows})
    scene_indexes: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        scene_indexes[str(row["scene_group_id"])].append(index)
    candidate = candidate_logits.detach().cpu().float()
    baseline = baseline_logits.detach().cpu().float()
    targets = targets.detach().cpu()
    candidate_prob = torch.softmax(candidate, -1)
    baseline_prob = torch.softmax(baseline, -1)
    one_hot = F.one_hot(targets, num_classes=2).float()
    per_example = {
        "accuracy": (candidate.argmax(-1) == targets).float().numpy()
        - (baseline.argmax(-1) == targets).float().numpy(),
        "nll": F.cross_entropy(candidate, targets, reduction="none").numpy()
        - F.cross_entropy(baseline, targets, reduction="none").numpy(),
        "brier": (candidate_prob - one_hot).square().sum(-1).numpy()
        - (baseline_prob - one_hot).square().sum(-1).numpy(),
    }
    rng = np.random.default_rng(seed)
    resampled = rng.integers(0, len(scenes), size=(repetitions, len(scenes)))
    output = {}
    for metric, values in per_example.items():
        per_scene = np.asarray(
            [np.mean([values[index] for index in scene_indexes[scene]]) for scene in scenes]
        )
        distribution = per_scene[resampled].mean(1)
        output[metric] = {
            "difference_direction": "candidate minus C0"
            if metric == "accuracy"
            else "candidate minus C0; negative favors candidate",
            "point_difference": float(per_scene.mean()),
            "percentile_95_ci": [
                float(np.quantile(distribution, 0.025)),
                float(np.quantile(distribution, 0.975)),
            ],
        }
    return {
        "method": "paired bootstrap over validation scenes; per-scene mean example difference",
        "cluster_count": len(scenes),
        "repetitions": repetitions,
        "seed": seed,
        "metrics": output,
    }


def _logits(model, blocks, queries):
    if isinstance(model, ConditionalAVReadout):
        return model.forward_batched(blocks, queries)
    return _baseline_batched(model, blocks, queries)


def _evaluate(model, values, targets, rows, train_count, queries, batch_size, device):
    model.eval()
    all_logits = []
    with torch.inference_mode():
        for start in range(0, len(targets), batch_size):
            stop = min(start + batch_size, len(targets))
            source = slice(
                start + (0 if len(targets) == train_count else train_count),
                stop + (0 if len(targets) == train_count else train_count),
            )
            blocks = _batch_blocks(
                values["audio"][source], values["text"][source], values["video"][source]
            )
            batch_queries = queries.unsqueeze(0).expand(stop - start, -1, -1)
            all_logits.append(_logits(model, blocks, batch_queries))
    logits = torch.cat(all_logits, dim=0).float()
    return logits


def _make_baseline(seed: int) -> NativeCandidateReadout:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return NativeCandidateReadout()


def _run_candidate(
    name,
    policy,
    config,
    train_rows,
    validation_rows,
    tensors,
    queries,
    device,
    output_dir,
    epoch_orders,
):
    training = config["training"]
    seed = int(config["seed"])
    baseline = _make_baseline(seed)
    if name == "c0_native_attention":
        model = baseline
    else:
        torch.manual_seed(seed)
        model = ConditionalAVReadout(
            baseline,
            exchange=policy["exchange"],
            gated=policy["gated"],
            scalar_event_audio=bool(config["readout"].get("scalar_event_audio", False)),
        )
    model = model.to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    values = {key: value.to(device) for key, value in tensors.items()}
    candidate_queries = queries.to(device)
    train_count = len(train_rows)
    batch_size = int(training["batch_size"])
    total_updates = int(training["max_optimizer_updates_per_candidate"])
    batches_per_epoch = math.ceil(train_count / batch_size)
    history_path = output_dir / f"training-history-{name}.jsonl"
    best_score, best_step, best_state = math.inf, 0, None
    consumed, seen, losses, grad_norms = 0, set(), [], []
    training_seconds = evaluation_seconds = 0.0

    def evaluate(step):
        nonlocal best_score, best_step, best_state, evaluation_seconds
        started = time.perf_counter()
        train_logits = _evaluate(
            model,
            values,
            values["targets"][:train_count],
            train_rows,
            train_count,
            candidate_queries,
            batch_size,
            device,
        )
        val_logits = _evaluate(
            model,
            values,
            values["targets"][train_count:],
            validation_rows,
            train_count,
            candidate_queries,
            batch_size,
            device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evaluation_seconds += time.perf_counter() - started
        train_metrics = _metrics(train_logits, values["targets"][:train_count])
        validation_metrics = _metrics(val_logits, values["targets"][train_count:])
        score = float(validation_metrics["nll"])
        selected = score < best_score
        if selected:
            best_score, best_step = score, step
            best_state = {
                key: item.detach().cpu().clone() if isinstance(item, torch.Tensor) else item
                for key, item in model.state_dict().items()
            }
            torch.save(
                {
                    "candidate": name,
                    "step": step,
                    "policy": policy,
                    "model_state": best_state,
                    "config": config,
                },
                output_dir / f"best-{name}.pt",
            )
        record = {
            "candidate": name,
            "optimizer_updates": step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_cross_entropy_recent_mean": float(
                np.mean(losses[-int(training["evaluation_interval"]) :])
            )
            if losses
            else None,
            "gradient_norm_recent_mean": float(
                np.mean(grad_norms[-int(training["evaluation_interval"]) :])
            )
            if grad_norms
            else None,
            "cumulative_consumed_examples": consumed,
            "unique_train_examples_consumed": len(seen),
            "train": train_metrics,
            "validation": validation_metrics,
            "best_step": best_step,
            "best_validation_nll": best_score,
            "selected_now": selected,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        }
        with history_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        return train_logits, val_logits, record

    latest_train, latest_val, _ = evaluate(0)
    for update in range(total_updates):
        model.train()
        order = epoch_orders[update // batches_per_epoch]
        batch_index = update % batches_per_epoch
        indexes = order[batch_index * batch_size : (batch_index + 1) * batch_size]
        if not len(indexes):
            raise RuntimeError("deterministic sampler produced an empty batch")
        seen.update(int(index) for index in indexes.tolist())
        consumed += len(indexes)
        indexes = indexes.to(device)
        blocks = _batch_blocks(
            values["audio"].index_select(0, indexes),
            values["text"].index_select(0, indexes),
            values["video"].index_select(0, indexes),
        )
        batch_queries = candidate_queries.unsqueeze(0).expand(len(indexes), -1, -1)
        targets = values["targets"].index_select(0, indexes)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        logits = _logits(model, blocks, batch_queries)
        loss = F.cross_entropy(logits.float(), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss in {name} update {update + 1}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, float(training["gradient_clip_norm"])
        )
        if not torch.isfinite(grad_norm) or any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in trainable_parameters
        ):
            raise FloatingPointError(f"nonfinite gradient in {name} update {update + 1}")
        optimizer.step()
        losses.append(float(loss.detach().item()))
        grad_norms.append(float(grad_norm.detach().item()))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds += time.perf_counter() - started
        if (update + 1) % int(training["evaluation_interval"]) == 0 or update + 1 == total_updates:
            latest_train, latest_val, _ = evaluate(update + 1)

    torch.save(
        {
            "candidate": name,
            "step": total_updates,
            "best_step": best_step,
            "policy": policy,
            "model_state": model.state_dict(),
            "config": config,
        },
        output_dir / f"final-{name}.pt",
    )
    reloaded = _make_baseline(seed)
    if name != "c0_native_attention":
        reloaded = ConditionalAVReadout(
            reloaded,
            exchange=policy["exchange"],
            gated=policy["gated"],
            scalar_event_audio=bool(config["readout"].get("scalar_event_audio", False)),
        )
    reloaded.load_state_dict(
        torch.load(output_dir / f"best-{name}.pt", map_location="cpu", weights_only=False)[
            "model_state"
        ],
        strict=True,
    )
    reloaded = reloaded.to(device).eval()
    best_val = _evaluate(
        reloaded,
        values,
        values["targets"][train_count:],
        validation_rows,
        train_count,
        candidate_queries,
        batch_size,
        device,
    )
    if abs(float(_metrics(best_val, values["targets"][train_count:])["nll"]) - best_score) > 1e-6:
        raise RuntimeError(f"best checkpoint reload changed selector score for {name}")
    best_predictions = _prediction_rows(validation_rows, best_val, values["targets"][train_count:])
    final_predictions = _prediction_rows(
        validation_rows, latest_val, values["targets"][train_count:]
    )
    (output_dir / f"predictions-best-{name}.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in best_predictions),
        encoding="utf-8",
    )
    (output_dir / f"predictions-final-{name}.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in final_predictions),
        encoding="utf-8",
    )
    return {
        "candidate": name,
        "policy": policy,
        "trainable_parameter_count": parameter_count,
        "optimizer_updates": total_updates,
        "consumed_training_examples": consumed,
        "unique_train_examples_consumed": len(seen),
        "best_step": best_step,
        "best_validation": _metrics(best_val, values["targets"][train_count:]),
        "best_validation_pairs": _scene_pair_metrics(
            validation_rows, best_val, values["targets"][train_count:]
        ),
        "final_train": _metrics(latest_train, values["targets"][:train_count]),
        "final_validation": _metrics(latest_val, values["targets"][train_count:]),
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "best_checkpoint_sha256": sha256_file(output_dir / f"best-{name}.pt"),
        "final_checkpoint_sha256": sha256_file(output_dir / f"final-{name}.pt"),
    }


def main() -> None:
    args = _args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_config = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    if (
        config["experiment_id"] not in {
            "conditional-clevrer-synthetic-av-readout-v1",
            "conditional-clevrer-synthetic-av-scalar-audio-v1",
        }
        or config["source_experiment_id"] != base_config["experiment_id"]
    ):
        raise ValueError("conditional AV config must identify the frozen v2 source")
    if config["readout"]["options"] != list(OPTIONS):
        raise ValueError("candidate options differ from frozen CLEVRER labels")
    if args.max_updates is not None:
        if args.max_updates < 1:
            raise ValueError("max updates must be positive")
        config["training"]["max_optimizer_updates_per_candidate"] = args.max_updates
    train_rows, validation_rows, tensors, input_report = _load_inputs(
        base_config,
        args.feature_dir.resolve(strict=True),
        args.feature_report.resolve(strict=True),
        args.sample_dir.resolve(strict=True),
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    gpu_gate = _gpu_gate(device)
    option_vectors, option_encoder = _encode_options(args.text_model_dir.resolve(strict=True))
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing attempt: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    batch_size = int(config["training"]["batch_size"])
    train_count = len(train_rows)
    updates = int(config["training"]["max_optimizer_updates_per_candidate"])
    batches_per_epoch = math.ceil(train_count / batch_size)
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    epoch_orders = [
        torch.randperm(train_count, generator=generator)
        for _ in range(math.ceil(updates / batches_per_epoch))
    ]
    sample_order = []
    for update in range(updates):
        order = epoch_orders[update // batches_per_epoch]
        batch = update % batches_per_epoch
        sample_order.extend(order[batch * batch_size : (batch + 1) * batch_size].tolist())
    order_hash = hashlib.sha256(np.asarray(sample_order, dtype="<i8").tobytes()).hexdigest()
    effective = {
        **config,
        "resolved": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "source_v2_config_sha256": sha256_file(V2_CONFIG),
            "feature_cache_sha256": input_report["feature_report"]["feature_cache_sha256"],
            "feature_report_sha256": sha256_file(args.feature_report.resolve(strict=True)),
            "train_id_order_sha256": input_report["split"]["train_id_order_sha256"],
            "validation_id_order_sha256": input_report["split"]["validation_id_order_sha256"],
            "sampler_order_sha256": order_hash,
            "scheduled_examples_per_candidate": len(sample_order),
            "candidate_encoder": option_encoder,
            "candidate_query_sha256": hashlib.sha256(
                option_vectors.numpy().astype("<f4").tobytes()
            ).hexdigest(),
            "device": str(device),
            "gpu_gate": gpu_gate,
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "repository_commit": __import__("subprocess")
            .run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, check=True, text=True)
            .stdout.strip(),
            "training_script_sha256": sha256_file(Path(__file__).resolve()),
            "readout_sha256": sha256_file(ROOT / "src/tiny_omni_decision/native_av_interaction.py"),
            "synthetic_audio_caveat": "880 Hz generated tone energy, not natural recorded audio",
        },
    }
    (output_dir / "effective-config.json").write_text(
        json.dumps(effective, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "status": "running",
        "experiment_id": config["experiment_id"],
        "sealed_audit_opened": False,
        "start_time_utc": __import__("datetime")
        .datetime.now(__import__("datetime").UTC)
        .isoformat(),
        "input_integrity": input_report["split"],
    }
    (output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    policies = {
        "c0_native_attention": {"exchange": False, "gated": False, "residual_enabled": False},
        "no_exchange_no_gate": {"exchange": False, "gated": False, "residual_enabled": True},
        "no_exchange_gate": {"exchange": False, "gated": True, "residual_enabled": True},
        "exchange_no_gate": {"exchange": True, "gated": False, "residual_enabled": True},
        "exchange_gate": {"exchange": True, "gated": True, "residual_enabled": True},
    }
    results = []
    device_tensors = {name: tensor.to(device) for name, tensor in tensors.items()}
    for candidate in config["training"]["candidate_order"]:
        results.append(
            _run_candidate(
                candidate,
                policies[candidate],
                config,
                train_rows,
                validation_rows,
                device_tensors,
                option_vectors,
                device,
                output_dir,
                epoch_orders,
            )
        )
    predictions = {
        item["candidate"]: torch.tensor(
            [
                [prob["probability"] for prob in row["probabilities"]]
                for row in _read_prediction_jsonl(
                    output_dir / f"predictions-best-{item['candidate']}.jsonl"
                )
            ],
            dtype=torch.float32,
        )
        .clamp_min(1e-8)
        .log()
        for item in results
    }
    deltas = {}
    for name, logits in predictions.items():
        if name != "c0_native_attention":
            deltas[name] = _paired_scene_deltas(
                validation_rows,
                logits,
                predictions["c0_native_attention"],
                device_tensors["targets"][train_count:].cpu(),
                repetitions=int(config["validation"]["bootstrap_repetitions"]),
                seed=int(config["seed"]),
            )
    final = {
        "status": "complete_conditional_clevrer_synthetic_av_readout_development_comparison",
        "experiment_id": config["experiment_id"],
        "results": results,
        "scene_clustered_deltas_vs_c0": deltas,
        "sealed_audit_opened": False,
        "quality_scope": (
            "synthetic paired tone-energy timing benchmark; not natural AV or product evidence"
        ),
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata.update(
        {
            "status": "complete",
            "end_time_utc": __import__("datetime")
            .datetime.now(__import__("datetime").UTC)
            .isoformat(),
            "completion_reason": (
                "all configured candidates completed; best checkpoints reloaded "
                "and selector metrics verified"
            ),
            "results_sha256": sha256_file(result_path),
        }
    )
    (output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "output_dir": str(output_dir),
                "results": [
                    {
                        "candidate": result["candidate"],
                        "best_step": result["best_step"],
                        "validation": result["best_validation"],
                    }
                    for result in results
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
