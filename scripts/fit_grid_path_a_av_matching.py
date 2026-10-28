"""Fit a small frozen-feature audiovisual utterance-matching readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.fit_grid_path_a_readouts import (
    ARRAY_HASHES,
    COHORT,
    FEATURE_RESULT_SHA,
    FEATURES,
    sha,
    validate_inputs,
)

SEED = 17
EPOCHS = 50
QUARTETS_PER_BATCH = 8
EVAL_SEED = 1701
STALE_SEED = 1702


class AVMatchReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio = nn.Sequential(nn.Linear(384, 64), nn.GELU())
        self.video_frame = nn.Sequential(nn.Linear(768, 64), nn.GELU())
        self.video_time = nn.GRU(64, 64, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(258, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        presence: torch.Tensor,
    ) -> torch.Tensor:
        if audio.ndim != 2 or audio.shape[1] != 384:
            raise ValueError("audio input must be [batch,384]")
        if video.ndim != 4 or video.shape[1:] != (4, 16, 768):
            raise ValueError("video input must be [batch,4,16,768]")
        if presence.shape != (len(audio), 2):
            raise ValueError("presence mask must be [batch,2]")
        batch = len(audio)
        a = self.audio(audio) * presence[:, :1]
        frames = self.video_frame(video.reshape(batch * 4, 16, 768))
        sequence, _ = self.video_time(frames)
        sequence = sequence.reshape(batch, 4, 16, 64)
        query = a[:, None, :]
        weights = torch.softmax((sequence * query[:, :, None, :]).sum(-1) / 8.0, dim=-1)
        visual = (sequence * weights[..., None]).sum(dim=2) * presence[:, None, 1:]
        audio_options = a[:, None, :].expand(-1, 4, -1)
        flags = presence[:, None, :].expand(-1, 4, -1)
        joint = torch.cat(
            (audio_options, visual, (audio_options - visual).abs(), audio_options * visual, flags),
            dim=-1,
        )
        return self.fusion(joint).squeeze(-1)


def load_groups(cohort: dict, ids: list[str]):
    id_position = {asset: index for index, asset in enumerate(ids)}
    all_groups = {}
    for split, key in (("train", "train_quartets"), ("validation", "validation_quartets")):
        groups = []
        seen_codes = set()
        for quartet in cohort[key]:
            speaker = quartet["speaker"]
            codes = quartet["codes"]
            options = [f"{speaker}_{code}" for code in codes]
            if len(options) != 4 or len(set(options)) != 4:
                raise ValueError("each matching choice set must have four unique utterances")
            if set(options) - set(id_position):
                raise ValueError("quartet contains utterances absent from feature cache")
            if seen_codes.intersection(options):
                raise ValueError("selected quartets must have disjoint utterance identities")
            seen_codes.update(options)
            option_ids = quartet["option_ids"]
            if len(option_ids) != 4 or len(set(option_ids)) != 4:
                raise ValueError("quartet option identities must be unique")
            option_index = dict(zip(option_ids, range(4), strict=True))
            queries = quartet["queries"]
            if len(queries) != 4:
                raise ValueError("quartet must contain all four audio query utterances")
            audio_rows, target_rows = [], []
            for query in queries:
                target = query["target_id"]
                if query["audio_code"] not in codes or target not in option_index:
                    raise ValueError("invalid audio-to-video match label")
                expected = quartet["option_ids"][codes.index(query["audio_code"])]
                if target != expected:
                    raise ValueError("target does not identify the same-utterance video")
                audio_rows.append(id_position[f"{speaker}_{query['audio_code']}"])
                target_rows.append(option_index[target])
            if sorted(target_rows) != [0, 1, 2, 3]:
                raise ValueError("each video must be the match for exactly one audio query")
            groups.append(
                {
                    "speaker": speaker,
                    "quartet_id": quartet["quartet_id"],
                    "candidate_rows": [id_position[asset] for asset in options],
                    "option_ids": options,
                    "audio_rows": audio_rows,
                    "targets": target_rows,
                    "candidate_ids": [options[:] for _ in range(4)],
                }
            )
        all_groups[split] = groups
    return all_groups


def make_arrays(groups, audio_bank, ids):
    audio_rows, candidate_rows, targets, speakers, quartet_ids, query_ids, candidate_ids = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for group in groups:
        for query_index, (audio_row, target) in enumerate(
            zip(group["audio_rows"], group["targets"], strict=True)
        ):
            audio_rows.append(audio_row)
            candidate_rows.append(group["candidate_rows"])
            targets.append(target)
            speakers.append(group["speaker"])
            quartet_ids.append(group["quartet_id"])
            query_ids.append(ids[audio_row])
            candidate_ids.append(group["candidate_ids"][query_index])
    audio = np.asarray(audio_bank[audio_rows], dtype=np.float32)
    return {
        "audio": audio,
        "targets": np.asarray(targets, dtype=np.int64),
        "speakers": np.asarray(speakers),
        "quartet_ids": np.asarray(quartet_ids),
        "query_rows": np.asarray(audio_rows, dtype=np.int64),
        "query_ids": np.asarray(query_ids),
        "candidate_rows": np.asarray(candidate_rows, dtype=np.int64),
        "candidate_ids": np.asarray(candidate_ids),
    }


def normalize(audio_train, audio_validation, video_train, video_validation):
    audio_mean = audio_train.mean(0, keepdims=True, dtype=np.float64).astype(np.float32)
    audio_scale = audio_train.std(0, keepdims=True, dtype=np.float64).astype(np.float32)
    video_mean = video_train.mean((0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    video_scale = video_train.std((0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    audio_scale = np.maximum(audio_scale, 1e-5)
    video_scale = np.maximum(video_scale, 1e-5)
    return (
        ((audio_train - audio_mean) / audio_scale).astype(np.float32),
        ((audio_validation - audio_mean) / audio_scale).astype(np.float32),
        ((video_train - video_mean) / video_scale).astype(np.float32),
        ((video_validation - video_mean) / video_scale).astype(np.float32),
        audio_mean,
        audio_scale,
        video_mean,
        video_scale,
    )


def option_orders(group_count: int, query_count: int, seed: int):
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(4) for _ in range(group_count * query_count)])


def reorder_candidates(candidates, targets, orders):
    reordered = np.take_along_axis(candidates, orders, axis=1)
    target_positions = np.argmax(orders == targets[:, None], axis=1)
    return reordered, target_positions


def listwise_metrics(logits, targets):
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    target = targets.cpu().numpy()
    one_hot = np.eye(4, dtype=np.float32)[target]
    prediction = probabilities.argmax(-1)
    confidence = probabilities.max(-1)
    correct = (prediction == target).astype(np.float32)
    ece = 0.0
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        chosen = (confidence >= lower) & (confidence < upper if index < 9 else confidence <= upper)
        if chosen.any():
            ece += chosen.mean() * abs(correct[chosen].mean() - confidence[chosen].mean())
    return {
        "accuracy": float(correct.mean()),
        "nll": float(F.cross_entropy(logits, targets).item()),
        "brier": float(np.square(probabilities - one_hot).sum(-1).mean()),
        "ece_10bin": float(ece),
        "mean_confidence": float(confidence.mean()),
        "count": len(target),
        "probabilities": probabilities,
        "predictions": prediction,
    }


def state_sha(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fit_arm(name, train, validation, output):
    torch.manual_seed(SEED)
    model = AVMatchReadout().cpu()
    mask = {"audio": (1.0, 0.0), "video": (0.0, 1.0), "joint": (1.0, 1.0)}[name]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    best_nll = float("inf")
    best_epoch = 0
    history = []
    best_path, last_path = output / f"{name}-best.pt", output / f"{name}-last.pt"
    tick = time.perf_counter()
    group_count = len(train["audio"]) // 4
    train_audio = torch.from_numpy(train["audio"])
    val_audio = torch.from_numpy(validation["audio"])
    val_presence = torch.tensor(mask).repeat(len(val_audio), 1)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        group_order = torch.randperm(group_count, generator=generator)
        losses = []
        option_rng = np.random.default_rng(SEED * 10_000 + epoch)
        for start in range(0, group_count, QUARTETS_PER_BATCH):
            group_ids = group_order[start : start + QUARTETS_PER_BATCH].numpy()
            query_indices = np.concatenate(
                [np.arange(group * 4, group * 4 + 4) for group in group_ids]
            )
            orders = np.stack([option_rng.permutation(4) for _ in query_indices])
            candidate_rows = train["candidate_rows"][query_indices]
            candidate_rows = np.take_along_axis(candidate_rows, orders, axis=1)
            target_positions = np.argmax(orders == train["targets"][query_indices, None], axis=1)
            candidate_video = torch.from_numpy(
                np.asarray(train["video_bank"][candidate_rows], dtype=np.float32)
            )
            indices = torch.from_numpy(query_indices)
            targets = torch.from_numpy(target_positions.astype(np.int64))
            presence = torch.tensor(mask).repeat(len(indices), 1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_audio[indices], candidate_video, presence)
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite audiovisual matching loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise ValueError("nonfinite audiovisual matching gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        val_candidates = torch.from_numpy(
            np.asarray(
                validation["video_bank"][validation["ordered_candidate_rows"]], dtype=np.float32
            )
        )
        with torch.inference_mode():
            val_logits = model(val_audio, val_candidates, val_presence)
        metric = listwise_metrics(val_logits, torch.from_numpy(validation["ordered_targets"]))
        record = {
            "epoch": epoch,
            "train_macro_nll_rolling": float(np.mean(losses)),
            "validation": {
                key: value
                for key, value in metric.items()
                if key not in {"probabilities", "predictions"}
            },
            "elapsed_seconds": time.perf_counter() - tick,
        }
        history.append(record)
        with (output / f"{name}-history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
        torch.save(model.state_dict(), last_path)
        if metric["nll"] < best_nll:
            best_nll, best_epoch = metric["nll"], epoch
            torch.save(model.state_dict(), best_path)
        print(
            json.dumps(
                {
                    "arm": name,
                    "epoch": epoch,
                    "epochs": EPOCHS,
                    "train_nll_rolling": record["train_macro_nll_rolling"],
                    "validation_nll": metric["nll"],
                    "best_epoch": best_epoch,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True), strict=True)
    with torch.inference_mode():
        best_logits = model(val_audio, val_candidates, val_presence)
    selected = listwise_metrics(best_logits, torch.from_numpy(validation["ordered_targets"]))
    np.savez_compressed(
        output / f"{name}-validation-predictions.npz",
        source_ids=validation["query_ids"],
        candidate_ids=validation["ordered_candidate_ids"],
        targets=validation["ordered_targets"],
        probabilities=selected["probabilities"],
        predictions=selected["predictions"],
    )
    return model, {
        "best_epoch": best_epoch,
        "best_nll": best_nll,
        "selected": {
            key: value
            for key, value in selected.items()
            if key not in {"probabilities", "predictions"}
        },
        "history": history,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "best_state_sha256": sha(best_path),
        "state_sha256": state_sha(model),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.exists() or output.is_relative_to(repo):
        raise ValueError("use a new external output directory")
    audio, video, targets, ids, cohort, _ = validate_inputs()
    if sha(FEATURES / "result.json") != FEATURE_RESULT_SHA:
        raise ValueError("frozen Path A features changed")
    groups = load_groups(cohort, ids)
    train_raw = make_arrays(groups["train"], audio, ids)
    validation_raw = make_arrays(groups["validation"], audio, ids)
    if (
        len(groups["train"]) != 192
        or len(groups["validation"]) != 64
        or len(train_raw["targets"]) != 768
        or len(validation_raw["targets"]) != 256
    ):
        raise ValueError("fixed 192/64 quartet cohort required")
    if set(train_raw["speakers"]) & set(validation_raw["speakers"]):
        raise ValueError("speaker leakage between train and validation")
    if set(train_raw["query_rows"].tolist()) & set(validation_raw["query_rows"].tolist()):
        raise ValueError("utterance identity leakage between train and validation")
    train_video_features = video[train_raw["query_rows"]]
    validation_video_features = video[validation_raw["query_rows"]]
    (
        train_raw["audio"],
        validation_raw["audio"],
        _normalized_train_video,
        _normalized_validation_video,
        a_mean,
        a_scale,
        v_mean,
        v_scale,
    ) = normalize(
        train_raw["audio"],
        validation_raw["audio"],
        train_video_features,
        validation_video_features,
    )
    output.mkdir(parents=True)
    np.savez(
        output / "train-only-normalizer.npz",
        audio_mean=a_mean,
        audio_scale=a_scale,
        video_mean=v_mean,
        video_scale=v_scale,
    )
    # Keep the canonical normalized bank to construct candidate option tensors
    # without duplicating normalization logic or changing option identities.
    video_bank = ((video - v_mean) / v_scale).astype(np.float32)
    train_raw["video_bank"] = video_bank
    validation_raw["video_bank"] = video_bank
    rng = np.random.default_rng(EVAL_SEED)
    validation_raw["eval_orders"] = np.stack(
        [rng.permutation(4) for _ in validation_raw["targets"]]
    )
    validation_raw["ordered_candidate_rows"] = np.take_along_axis(
        validation_raw["candidate_rows"], validation_raw["eval_orders"], axis=1
    )
    validation_raw["ordered_candidate_ids"] = np.take_along_axis(
        validation_raw["candidate_ids"], validation_raw["eval_orders"], axis=1
    )
    validation_raw["ordered_targets"] = np.argmax(
        validation_raw["eval_orders"] == validation_raw["targets"][:, None], axis=1
    ).astype(np.int64)
    config = {
        "protocol": "grid-path-a-synchronized-av-match-v1",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_quartets": QUARTETS_PER_BATCH,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "selection": "minimum validation four-choice NLL",
        "arms": ["audio", "video", "joint"],
        "validation_candidate_order_seed": EVAL_SEED,
        "training_candidate_order_seed_rule": "seed*10000+epoch; independent permutation per query",
        "task": "choose the synchronized same-utterance video among three same-speaker distractors",
        "train_quartets": len(groups["train"]),
        "validation_quartets": len(groups["validation"]),
        "train_queries": len(train_raw["targets"]),
        "validation_queries": len(validation_raw["targets"]),
        "train_speakers": cohort["train_speakers"],
        "validation_speakers": cohort["validation_speakers"],
        "feature_result_sha256": sha(FEATURES / "result.json"),
        "cohort_sha256": sha(COHORT),
        "feature_hashes": ARRAY_HASHES,
        "feature_ids_sha256": sha(FEATURES / "feature-ids.json"),
        "gpu_used": False,
        "sealed_audit_opened": False,
    }
    (output / "effective-config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    results, models = {}, {}
    start = time.perf_counter()
    for arm in ("audio", "video", "joint"):
        model, result = fit_arm(arm, train_raw, validation_raw, output)
        models[arm], results[arm] = model, result
    joint = models["joint"]
    q_audio = torch.from_numpy(validation_raw["audio"])
    cand_rows = validation_raw["ordered_candidate_rows"].copy()
    cand_video = torch.from_numpy(
        np.asarray(validation_raw["video_bank"][cand_rows], dtype=np.float32)
    )
    targets_t = torch.from_numpy(validation_raw["ordered_targets"])
    cases = {}
    controls = {
        "matched": (q_audio, cand_video, (1.0, 1.0)),
        "audio_missing": (torch.zeros_like(q_audio), cand_video, (0.0, 1.0)),
        "video_missing": (q_audio, torch.zeros_like(cand_video), (1.0, 0.0)),
        "both_missing": (torch.zeros_like(q_audio), torch.zeros_like(cand_video), (0.0, 0.0)),
        "audio_shuffled_within_quartet": (
            torch.from_numpy(
                np.roll(validation_raw["audio"].reshape(64, 4, 384), 1, axis=1).reshape(256, 384)
            ),
            cand_video,
            (1.0, 1.0),
        ),
        "video_temporal_offset_4": (q_audio, torch.roll(cand_video, shifts=4, dims=2), (1.0, 1.0)),
    }
    # Replace candidate sets with the next disjoint quartet from the same speaker.
    stale_rng = np.random.default_rng(STALE_SEED)
    group_speakers = [group["speaker"] for group in groups["validation"]]
    by_speaker = {speaker: [] for speaker in cohort["validation_speakers"]}
    for index, speaker in enumerate(group_speakers):
        by_speaker[speaker].append(index)
    stale_group_donor = {}
    for _speaker, group_indices in by_speaker.items():
        shift = int(stale_rng.integers(1, len(group_indices)))
        for position, group_index in enumerate(group_indices):
            stale_group_donor[group_index] = group_indices[(position + shift) % len(group_indices)]
    stale_candidates = []
    for group_index in range(64):
        donor_group = stale_group_donor[group_index]
        donor_rows = validation_raw["candidate_rows"][donor_group * 4]
        stale_candidates.extend([donor_rows] * 4)
    stale_rows = np.take_along_axis(
        np.asarray(stale_candidates), validation_raw["eval_orders"], axis=1
    )
    stale_ids = []
    for group_index in range(64):
        donor_group = stale_group_donor[group_index]
        donor_ids = validation_raw["candidate_ids"][donor_group * 4]
        stale_ids.extend([donor_ids] * 4)
    stale_candidate_ids = np.take_along_axis(
        np.asarray(stale_ids), validation_raw["eval_orders"], axis=1
    )
    if any(
        query_id in candidate_ids
        for query_id, candidate_ids in zip(
            validation_raw["query_ids"], stale_candidate_ids, strict=True
        )
    ):
        raise ValueError("stale-video control accidentally contains a matching source")
    controls["stale_video_other_quartet_same_speaker"] = (
        q_audio,
        torch.from_numpy(np.asarray(validation_raw["video_bank"][stale_rows], dtype=np.float32)),
        (1.0, 1.0),
    )
    control_predictions = {}
    for name, (audio_input, video_input, present) in controls.items():
        with torch.inference_mode():
            logits = joint(audio_input, video_input, torch.tensor(present).repeat(256, 1))
        metric = listwise_metrics(logits, targets_t)
        cases[name] = {
            key: value
            for key, value in metric.items()
            if key not in {"probabilities", "predictions"}
        }
        control_predictions[name] = {
            "probabilities": metric["probabilities"],
            "predictions": metric["predictions"],
        }
    np.savez_compressed(
        output / "joint-validation-controls.npz",
        source_ids=validation_raw["query_ids"],
        candidate_ids=validation_raw["ordered_candidate_ids"],
        stale_video_candidate_ids=stale_candidate_ids,
        targets=validation_raw["ordered_targets"],
        **{
            f"{name}_probabilities": value["probabilities"]
            for name, value in control_predictions.items()
        },
        **{
            f"{name}_predictions": value["predictions"]
            for name, value in control_predictions.items()
        },
    )
    results["joint_controls"] = cases
    results["duration_seconds"] = time.perf_counter() - start
    results["status"] = "PASS_GRID_PATH_A_AV_MATCHING"
    (output / "result.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": results["status"],
                "duration_seconds": results["duration_seconds"],
                "best_epochs": {
                    name: value["best_epoch"] for name, value in results.items() if name in models
                },
                "validation_accuracy": {
                    name: value["selected"]["accuracy"]
                    for name, value in results.items()
                    if name in models
                },
                "joint_controls_accuracy": {
                    name: value["accuracy"] for name, value in cases.items()
                },
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
