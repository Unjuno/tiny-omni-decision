"""Fit fixed CPU readouts over the immutable GRID Path A feature cache."""

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

from scripts.grid_path_a_data import OPTIONS

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
FEATURES = ROOT / "grid-path-a-features-20261028-attempt-01"
COHORT = ROOT / "grid-path-a-cohort-20261028-attempt-01/cohort.json"
FEATURE_RESULT_SHA = "40a06e5b4a3c69b6c965a48b59d4f08a32000f7b25c0a11cd1038f37af2e87e3"
CONFIG_SHA = "739f100939b690ab86e3fa95a54fb7ba1a5a338c0e29a2f5ba624e277e884892"
ARRAY_HASHES = {
    "audio-features.npy": "5195324aa6a81fdc5181856eec749a1dfdf06126f5481eacf859f117e4d999cf",
    "video-features.npy": "4ea332c035e2ae263b271e7f7199496caef4123c3b89c6afc23bbadc13a0eda1",
    "targets.npy": "ec6654fb1ed121c23f45d20c9d45c553edc927574317da8b570b65e4a86b8298",
    "feature-ids.json": "6518ad596bc0abb6e554697ebae86b2a73150e97f86e948ce59a641583380010",
}
EPOCHS = 50
BATCH_SIZE = 64
SEED = 17


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SlotReadout(nn.Module):
    def __init__(self, classes: list[int]):
        super().__init__()
        self.audio = nn.Sequential(nn.Linear(384, 64), nn.GELU())
        self.video_frame = nn.Sequential(nn.Linear(768, 64), nn.GELU())
        self.video_time = nn.GRU(64, 64, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(130, 96), nn.GELU())
        self.slots = nn.ModuleList(nn.Linear(96, count) for count in classes)

    def forward(self, audio: torch.Tensor, video: torch.Tensor, mask: torch.Tensor):
        a = self.audio(audio) * mask[:, :1]
        sequence, _ = self.video_time(self.video_frame(video))
        v = sequence[:, -1] * mask[:, 1:]
        shared = self.fusion(torch.cat((a, v, mask), dim=-1))
        return [head(shared) for head in self.slots]


def macro_loss(logits, targets):
    return torch.stack(
        [F.cross_entropy(value, targets[:, i]) for i, value in enumerate(logits)]
    ).mean()


def expected_ids(cohort: dict) -> list[str]:
    out = []
    for split in ("train", "validation"):
        for quartet in cohort[f"{split}_quartets"]:
            for code in quartet["codes"]:
                out.append(f"{quartet['speaker']}_{code}")
    return out


def validate_inputs():
    result_path = FEATURES / "result.json"
    if sha(result_path) != FEATURE_RESULT_SHA:
        raise ValueError("frozen feature result hash differs")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "PASS_GRID_PATH_A_FROZEN_FEATURES"
        or result.get("effective_config_sha256") != CONFIG_SHA
        or result.get("sealed_audit_opened") is not False
        or result.get("gpu_used") is not False
    ):
        raise ValueError("feature result is not the expected sealed-audit-free CPU run")
    for name, expected in ARRAY_HASHES.items():
        if sha(FEATURES / name) != expected:
            raise ValueError(f"immutable feature input changed: {name}")
    ids = json.loads((FEATURES / "feature-ids.json").read_text(encoding="utf-8"))
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    if cohort.get("status") != "PASS_FROZEN_GROUPED_COHORT" or ids != expected_ids(cohort):
        raise ValueError("feature order differs from frozen speaker-grouped cohort")
    if sha(COHORT) != "61b70af8cb826ded36986aaa53b55ef88f3e5973542d9237df1c951086bab752":
        raise ValueError("frozen cohort hash differs")
    audio = np.load(FEATURES / "audio-features.npy", mmap_mode="r", allow_pickle=False)
    video = np.load(FEATURES / "video-features.npy", mmap_mode="r", allow_pickle=False)
    targets = np.load(FEATURES / "targets.npy", mmap_mode="r", allow_pickle=False)
    if (
        audio.shape != (1024, 384)
        or video.shape != (1024, 16, 768)
        or targets.shape != (1024, 6)
        or audio.dtype != np.float32
        or video.dtype != np.float32
        or targets.dtype != np.int64
    ):
        raise ValueError("feature bank dimensions or dtypes differ")
    if not (np.isfinite(audio).all() and np.isfinite(video).all()):
        raise ValueError("frozen features contain NaN/Inf")
    class_counts = [len(OPTIONS[name]) for name in OPTIONS]
    for slot, count in enumerate(class_counts):
        if targets[:, slot].min() < 0 or targets[:, slot].max() >= count:
            raise ValueError("target outside the complete native slot vocabulary")
    train_speakers = set(cohort["train_speakers"])
    validation_speakers = set(cohort["validation_speakers"])
    if (
        train_speakers & validation_speakers
        or {x.split("_", 1)[0] for x in ids[:768]} != train_speakers
        or {x.split("_", 1)[0] for x in ids[768:]} != validation_speakers
        or set(ids[:768]) & set(ids[768:])
    ):
        raise ValueError("speaker-disjoint split integrity failed")
    return audio, video, targets, ids, cohort, class_counts


def normalize(train: np.ndarray, validation: np.ndarray, temporal: bool = False):
    axes = (0, 1) if temporal else (0,)
    mean = train.mean(axis=axes, keepdims=True, dtype=np.float64).astype(np.float32)
    scale = train.std(axis=axes, keepdims=True, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-5)
    return (
        ((train - mean) / scale).astype(np.float32),
        ((validation - mean) / scale).astype(np.float32),
        mean,
        scale,
    )


def evaluate(model, audio, video, targets, mask, batch=128):
    model.eval()
    gathered = [[] for _ in OPTIONS]
    with torch.inference_mode():
        for start in range(0, len(audio), batch):
            result = model(
                audio[start : start + batch],
                video[start : start + batch],
                mask[start : start + batch],
            )
            for slot, value in enumerate(result):
                gathered[slot].append(value.cpu())
    logits = [torch.cat(value) for value in gathered]
    metrics = {}
    probs = []
    nlls, briers, eces, accs = [], [], [], []
    for slot, value in enumerate(logits):
        target = targets[:, slot]
        p = value.softmax(-1).numpy()
        pred = p.argmax(-1)
        one_hot = np.eye(p.shape[1], dtype=np.float32)[target.numpy()]
        confidence = p.max(-1)
        correct = (pred == target.numpy()).astype(np.float32)
        ece = 0.0
        for lo in np.linspace(0, 1, 11)[:-1]:
            hi = lo + 0.1
            selected = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
            if selected.any():
                ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
        item = {
            "accuracy": float(correct.mean()),
            "nll": float(F.cross_entropy(value, target).item()),
            "brier": float(np.mean(np.square(p - one_hot).sum(-1))),
            "ece_10bin": float(ece),
            "count": len(target),
        }
        metrics[list(OPTIONS)[slot]] = item
        probs.append(p)
        nlls.append(item["nll"])
        briers.append(item["brier"])
        eces.append(item["ece_10bin"])
        accs.append(item["accuracy"])
    return {
        "slot_metrics": metrics,
        "macro_accuracy": float(np.mean(accs)),
        "macro_nll": float(np.mean(nlls)),
        "macro_brier": float(np.mean(briers)),
        "macro_ece": float(np.mean(eces)),
        "probabilities": probs,
        "predictions": [p.argmax(-1) for p in probs],
    }


def train_arm(
    name,
    audio_train,
    video_train,
    target_train,
    audio_val,
    video_val,
    target_val,
    validation_ids,
    classes,
    out,
):
    torch.manual_seed(SEED)
    model = SlotReadout(classes).cpu()
    train_mask = torch.tensor(
        {"audio": [1.0, 0.0], "video": [0.0, 1.0], "joint": [1.0, 1.0]}[name]
    ).repeat(len(audio_train), 1)
    val_mask = torch.tensor(
        {"audio": [1.0, 0.0], "video": [0.0, 1.0], "joint": [1.0, 1.0]}[name]
    ).repeat(len(audio_val), 1)
    train_data = [torch.from_numpy(x) for x in (audio_train, video_train, target_train)]
    val_audio, val_video, val_targets = (
        torch.from_numpy(x) for x in (audio_val, video_val, target_val)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    best_nll = float("inf")
    best_epoch = 0
    history = []
    best_path = out / f"{name}-best.pt"
    last_path = out / f"{name}-last.pt"
    epoch_tick = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_data[0]), generator=generator)
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_data[0][index], train_data[1][index], train_mask[index])
            loss = macro_loss(logits, train_data[2][index])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise ValueError("nonfinite gradient norm")
            optimizer.step()
            losses.append(float(loss.detach()))
        result = evaluate(model, val_audio, val_video, val_targets, val_mask)
        record = {
            "epoch": epoch,
            "train_macro_nll": float(np.mean(losses)),
            "validation": {
                k: v for k, v in result.items() if k not in {"probabilities", "predictions"}
            },
            "seconds_since_start": time.perf_counter() - epoch_tick,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        with (out / f"{name}-history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
        print(
            json.dumps(
                {
                    "arm": name,
                    "epoch": epoch,
                    "epochs": EPOCHS,
                    "train_macro_nll": record["train_macro_nll"],
                    "validation_macro_nll": result["macro_nll"],
                    "best_epoch": best_epoch if result["macro_nll"] >= best_nll else epoch,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        torch.save(model.state_dict(), last_path)
        if result["macro_nll"] < best_nll:
            best_nll, best_epoch = result["macro_nll"], epoch
            torch.save(model.state_dict(), best_path)
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True), strict=True)
    selected = evaluate(model, val_audio, val_video, val_targets, val_mask)
    np.savez_compressed(
        out / f"{name}-validation-predictions.npz",
        source_ids=np.asarray(validation_ids),
        targets=target_val,
        **{f"prob_{i}": p for i, p in enumerate(selected["probabilities"])},
        **{f"prediction_{i}": p for i, p in enumerate(selected["predictions"])},
    )
    return model, {
        "best_epoch": best_epoch,
        "best_macro_nll": best_nll,
        "selected": {
            k: v for k, v in selected.items() if k not in {"probabilities", "predictions"}
        },
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.exists() or output.is_relative_to(repo):
        raise ValueError("use a new external artifact directory")
    audio, video, targets, ids, cohort, classes = validate_inputs()
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_n = 768
    audio_train, audio_val, audio_mean, audio_scale = normalize(audio[:train_n], audio[train_n:])
    video_train, video_val, video_mean, video_scale = normalize(
        video[:train_n], video[train_n:], temporal=True
    )
    np.savez(
        output / "train-only-normalizer.npz",
        audio_mean=audio_mean,
        audio_scale=audio_scale,
        video_mean=video_mean,
        video_scale=video_scale,
    )
    config = {
        "protocol": "grid-path-a-speech-slot-readouts-v1",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "selection": "minimum mean six-slot validation NLL",
        "arms": ["audio", "video", "joint"],
        "primary_control": "joint model with validation video swapped across speakers",
        "device": "cpu",
        "gpu_used": False,
        "train_rows": 768,
        "validation_rows": 256,
        "train_speakers": cohort["train_speakers"],
        "validation_speakers": cohort["validation_speakers"],
        "feature_result_sha256": FEATURE_RESULT_SHA,
        "feature_config_sha256": CONFIG_SHA,
        "cohort_sha256": sha(COHORT),
        "ordered_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "feature_hashes": ARRAY_HASHES,
        "class_counts": classes,
        "sealed_audit_opened": False,
    }
    (output / "effective-config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    results = {}
    models = {}
    start = time.perf_counter()
    for arm in ("audio", "video", "joint"):
        model, result = train_arm(
            arm,
            audio_train,
            video_train,
            targets[:train_n],
            audio_val,
            video_val,
            targets[train_n:],
            ids[train_n:],
            classes,
            output,
        )
        models[arm], results[arm] = model, result
    # Paired mismatch: every validation video is replaced by a different held-out
    # speaker's video at the same within-speaker position.
    val_ids = ids[train_n:]
    val_speakers = cohort["validation_speakers"]
    lookup = {speaker: [] for speaker in val_speakers}
    for index, asset in enumerate(val_ids):
        lookup[asset.split("_", 1)[0]].append(index)
    if any(len(rows) != 32 for rows in lookup.values()):
        raise ValueError("expected balanced validation speaker blocks for mismatch control")
    mismatch = np.empty(256, dtype=np.int64)
    for speaker_index, speaker in enumerate(val_speakers):
        donor = val_speakers[(speaker_index + 1) % len(val_speakers)]
        for within in range(32):
            mismatch[lookup[speaker][within]] = lookup[donor][within]
    if np.any(mismatch == np.arange(256)) or any(
        val_ids[i].split("_", 1)[0] == val_ids[j].split("_", 1)[0] for i, j in enumerate(mismatch)
    ):
        raise ValueError("video mismatch control failed speaker-level derangement")
    joint_mask = torch.tensor([1.0, 1.0]).repeat(256, 1)
    mismatched = evaluate(
        models["joint"],
        torch.from_numpy(audio_val),
        torch.from_numpy(video_val[mismatch]),
        torch.from_numpy(targets[train_n:]),
        joint_mask,
    )
    results["joint_mismatched_video"] = {
        k: v for k, v in mismatched.items() if k not in {"probabilities", "predictions"}
    }
    np.savez_compressed(
        output / "joint-mismatched-video-predictions.npz",
        source_ids=np.asarray(val_ids),
        targets=targets[train_n:],
        donor_ids=np.asarray([val_ids[i] for i in mismatch]),
        **{f"prediction_{i}": p for i, p in enumerate(mismatched["predictions"])},
        **{f"prob_{i}": p for i, p in enumerate(mismatched["probabilities"])},
    )
    results["duration_seconds"] = time.perf_counter() - start
    results["status"] = "PASS_GRID_PATH_A_READOUTS"
    (output / "result.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": results["status"],
                "duration_seconds": results["duration_seconds"],
                "best_epochs": {arm: results[arm]["best_epoch"] for arm in models},
                "macro_accuracy": {
                    arm: results[arm]["selected"]["macro_accuracy"] for arm in models
                },
                "macro_nll": {arm: results[arm]["selected"]["macro_nll"] for arm in models},
                "mismatch_accuracy": mismatched["macro_accuracy"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
