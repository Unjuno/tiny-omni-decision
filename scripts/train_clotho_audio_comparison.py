"""Fixed audio-only/joint readouts gated on the complete frozen development corpus."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.clotho_acquisition_gate import complete_acquisition
from scripts.clotho_audio_controls import VALIDATION_MAPPING_SHA256, source_derangement
from scripts.fetch_clotho_development_media import HASHES
from scripts.starss23_readout_common import binary_metrics
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer


def candidate_inputs(audio, question, condition):
    if condition not in {"audio", "joint"}:
        raise ValueError("unknown comparison condition")
    if audio.shape != question.shape or audio.ndim != 2 or audio.shape[1] != 384:
        raise ValueError("aligned384-dimensional query features required")
    shared = torch.cat(
        (audio, question if condition == "joint" else torch.zeros_like(question)), -1
    )
    return torch.cat(
        (shared[:, None].expand(-1, 2, -1), torch.eye(2).expand(len(shared), -1, -1)), -1
    )


def load_inputs(root):
    media = root / "clotho-aqa-development-media-02"
    records = complete_acquisition(media, root / "clotho-aqa-development-split-01")
    media_sha = hashlib.sha256((media / "result.json").read_bytes()).hexdigest()
    screen = json.loads(
        (root / "clotho-aqa-cross-split-waveform-screen-01/result.json").read_text()
    )
    if (
        screen["status"] != "COMPLETE"
        or screen["flagged_pairs"] != 0
        or screen["media_report_sha256"] != media_sha
        or screen["scope"] != "all cross-split whole-template pairs"
        or screen["review_threshold"] != 0.99
        or screen["pairs"] != 180004
        or screen["expected_pairs"] != 180004
    ):
        raise ValueError("whole-template screen incomplete, flagged or different")
    feature_dir = root / "clotho-aqa-audio-features-01"
    audio_info = json.loads((feature_dir / "result.json").read_text())
    path = feature_dir / "features.npy"
    if (
        audio_info["status"] != "PASS"
        or audio_info["media_report_sha256"] != media_sha
        or hashlib.sha256(path.read_bytes()).hexdigest() != audio_info["feature_sha256"]
    ):
        raise ValueError("audio features differ from acquisition")
    if [r["id"] for r in audio_info["rows"]] != [r["source_id"] for r in records]:
        raise ValueError("audio features/source order differ")
    audio = np.load(path, allow_pickle=False)
    if audio.shape != (856, 384) or not np.isfinite(audio).all():
        raise ValueError("invalid audio features")
    lookup = {row["source_id"]: i for i, row in enumerate(records)}
    text_dir = root / "clotho-aqa-question-features-01"
    text_info = json.loads((text_dir / "result.json").read_text())
    x, y, rows = {}, {}, {}
    for split, expected in HASHES.items():
        data = (root / f"clotho-aqa-development-split-01/{split}.json").read_bytes()
        info = text_info["splits"][split]
        path = text_dir / (split + ".npy")
        if (
            hashlib.sha256(data).hexdigest() != expected
            or info["input_sha256"] != expected
            or hashlib.sha256(path.read_bytes()).hexdigest() != info["feature_sha256"]
        ):
            raise ValueError("question feature/input hash mismatch")
        rows[split] = json.loads(data)
        text = np.load(path, allow_pickle=False)
        if text.shape != (len(rows[split]), 384) or not np.isfinite(text).all():
            raise ValueError("invalid text features")
        x[split] = (audio[[lookup[r["source_id"]] for r in rows[split]]], text)
        y[split] = torch.tensor([r["target"] for r in rows[split]])
    return (
        x,
        y,
        rows,
        dict(audio=audio_info, question=text_info, waveform_screen=screen, split_hashes=HASHES),
    )


def fit(root, condition, raw, y, rows, provenance):
    mapping = source_derangement(rows["validation"]) if condition == "joint" else {}
    mapping_json = json.dumps(mapping, sort_keys=True, indent=2)
    if condition == "joint" and hashlib.sha256(mapping_json.encode()).hexdigest() != (
        VALIDATION_MAPPING_SHA256
    ):
        raise ValueError("frozen validation source control changed")
    output = root / f"clotho-aqa-{condition}-control-01"
    output.mkdir(exist_ok=False)
    protocol = dict(
        condition=condition,
        seed=17,
        epochs=24,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=0.001,
        hidden_size=64,
        selector="earliest minimum raw validation NLL",
        calibration="17 geometric temperatures .25..4, calibration split only",
        normalization="train question-weighted features only",
        inference_control_policy="seed17 source-coherent cross-uploader bijection and zero-mean "
        "audio; selected joint weights/temperature frozen",
        inference_control_module_sha256=hashlib.sha256(
            Path(__file__).with_name("clotho_audio_controls.py").read_bytes()
        ).hexdigest(),
        validation_control_sha256=VALIDATION_MAPPING_SHA256 if condition == "joint" else None,
        provenance=provenance,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    means = [a.mean(0) for a in raw["train"]]
    stds = [np.where(a.std(0) > 1e-6, a.std(0), 1) for a in raw["train"]]
    np.savez(
        output / "normalization.npz",
        audio_mean=means[0],
        audio_std=stds[0],
        question_mean=means[1],
        question_std=stds[1],
    )
    x = {
        s: candidate_inputs(
            *[torch.from_numpy((a - m) / d) for a, m, d in zip(pair, means, stds, strict=True)],
            condition,
        )
        for s, pair in raw.items()
    }
    torch.manual_seed(17)
    model = FrozenFeatureCandidateScorer(770, 64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    generator = torch.Generator().manual_seed(17)
    best, history = float("inf"), []
    started = time.perf_counter()
    for epoch in range(1, 25):
        model.train()
        total = 0.0
        for batch in torch.randperm(len(y["train"]), generator=generator).split(128):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x["train"][batch]), y["train"][batch])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        model.eval()
        with torch.inference_mode():
            p = model(x["validation"]).softmax(-1)[:, 1].numpy()
        metrics = binary_metrics(y["validation"].numpy(), p)
        history.append(dict(epoch=epoch, train_ce=total / len(y["train"]), **metrics))
        if metrics["nll"] < best:
            best, best_epoch = metrics["nll"], epoch
            torch.save(model.state_dict(), output / "best.pt")
    torch.save(model.state_dict(), output / "final.pt")
    model.load_state_dict(torch.load(output / "best.pt", weights_only=True))
    with torch.inference_mode():
        cal = model(x["calibration"])
        grid = np.geomspace(0.25, 4, 17)
        temperature = float(
            grid[
                np.argmin([float(F.cross_entropy(cal / float(t), y["calibration"])) for t in grid])
            ]
        )
        logits = model(x["validation"])
        if not torch.equal(logits, model(x["validation"].flip(1)).flip(1)):
            raise ValueError("option permutation changed predictions")
        probs = (logits / temperature).softmax(-1)[:, 1].numpy()
    np.save(output / "validation_probabilities.npy", probs)
    (output / "validation_ids.json").write_text(
        json.dumps([[r["source_id"], r["question"]] for r in rows["validation"]])
    )
    result = dict(
        condition=condition,
        best_epoch=best_epoch,
        temperature=temperature,
        metrics=binary_metrics(y["validation"].numpy(), probs),
        history=history,
        parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.perf_counter() - started,
        consumed_train_questions=24 * len(y["train"]),
        unique_train_questions=len(y["train"]),
        best_sha256=hashlib.sha256((output / "best.pt").read_bytes()).hexdigest(),
    )
    if condition == "joint":
        diagnostic_started = time.perf_counter()
        first = {}
        for index, row in enumerate(rows["validation"]):
            first.setdefault(row["source_id"], index)
        indices = [first[mapping[r["source_id"]]] for r in rows["validation"]]
        changed = x["validation"].clone()
        changed[:, :, :384] = changed[indices, :, :384]
        zero = x["validation"].clone()
        zero[:, :, :384] = 0
        result["inference_controls"] = {}
        with torch.inference_mode():
            for name, features in (("source_deranged", changed), ("audio_zero_mean", zero)):
                values = (model(features) / temperature).softmax(-1)[:, 1].numpy()
                np.save(output / (name + "_probabilities.npy"), values)
                result["inference_controls"][name] = binary_metrics(y["validation"].numpy(), values)
        (output / "source_derangement.json").write_text(mapping_json)
        result["inference_control_seconds"] = time.perf_counter() - diagnostic_started
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "history"}))


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    raw, y, rows, provenance = load_inputs(root)
    torch.set_num_threads(4)
    for condition in ("audio", "joint"):
        fit(root, condition, raw, y, rows, provenance)


if __name__ == "__main__":
    main()
