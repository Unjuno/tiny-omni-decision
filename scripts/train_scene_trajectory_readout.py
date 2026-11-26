"""Fixed trajectory readout comparison; complete pinned corpus mandatory."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.audit_correspondence_validation import sha, write
from tiny_omni_decision.decision_math import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)
from tiny_omni_decision.scene_trajectory_readout import SceneTrajectoryReadout

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
PINS = {
    "train": "8869bb903591af3a0995eeca0eeae1b50b990c8b08752a6e5d4d549f8a356fd7",
    "valid": "30846aaa87e2292e97ee3187c5f2f986ba3c3d57d46a249b361e5a14c0cc8ec2",
}


def verify_media_binding(row, media):
    expected = media.get(row["video_id"])
    if expected is None or (
        row["split"] != expected["split"] or row["media_sha256"] != expected["sha256"]
    ):
        raise ValueError("feature media binding changed")


def load_trajectory_inputs(video_dir, text_dir):
    hashes, pairs = {}, {}
    for split, pin in PINS.items():
        path = ROOT / f"perception-occlusion-pairs-01/{split}.json"
        if sha(path) != pin:
            raise ValueError("pairs changed")
        pairs[split] = json.loads(path.read_text(encoding="utf-8"))["pairs"]
    video = json.loads((video_dir / "result.json").read_text())
    text = json.loads((text_dir / "result.json").read_text())
    if video["status"] != "PASS_FEATURE_EXTRACTION_NOT_QUALITY":
        raise ValueError("complete video feature extraction required")
    expected_manifests = {
        "train": "469299ad240ff85c47ba44abb8920aae9994b0e3ec15a3fca1373c49b2fa6ee3",
        "valid": "213bd33994fb80d79ee05ee0580af2cc9915dfc61094aed00c847c2c67f1d546",
    }
    if video.get("manifest_hashes") != expected_manifests:
        raise ValueError("media manifest provenance changed")
    media, seen_media_hashes = {}, set()
    for split, pin in expected_manifests.items():
        manifest = ROOT / f"perception-occlusion-{split}-media-01/manifest.json"
        if sha(manifest) != pin:
            raise ValueError("actual media manifest changed")
        hashes[str(manifest)] = pin
        for row in json.loads(manifest.read_text())["clips"]:
            if row["video_id"] in media or row["sha256"] in seen_media_hashes:
                raise ValueError("duplicate media identity/content")
            media[row["video_id"]] = dict(row, split=split)
            seen_media_hashes.add(row["sha256"])
    if text["status"] != "PASS_FROZEN_TEXT_FEATURES":
        raise ValueError("verified text features required")
    if text["pair_hashes"] != PINS:
        raise ValueError("feature pair provenance changed")
    for folder in (video_dir, text_dir):
        hashes[str(folder / "result.json")] = sha(folder / "result.json")
    bank, identities, contents = {}, {}, {"train": set(), "valid": set()}
    video_dim = 512
    for row in video["features"]:
        verify_media_binding(row, media)
        key = row["video_id"]
        if key in bank or sha(Path(row["path"])) != row["sha256"]:
            raise ValueError("duplicate/changed video feature")
        value = np.load(row["path"], allow_pickle=False)
        if value.shape != (256, video_dim) or not np.isfinite(value).all():
            raise ValueError("invalid video features")
        bank[key] = torch.from_numpy(value)
        identities[key] = row["split"]
        contents[row["split"]].add(row["prepared_sha256"])
        hashes[row["path"]] = row["sha256"]
    if contents["train"] & contents["valid"]:
        raise ValueError("prepared window overlap")
    if set(bank) != {p["video_id"] for ps in pairs.values() for p in ps} or len(bank) != 356:
        raise ValueError("incomplete/extra video identities")
    data = {}
    for split, ps in pairs.items():
        row = text["features"][split]
        if sha(Path(row["path"])) != row["sha256"]:
            raise ValueError("changed text feature")
        hashes[row["path"]] = row["sha256"]
        array = np.load(row["path"], allow_pickle=False)
        expected = [
            dict(video_id=p["video_id"], condition=q["condition"], target=q["target"])
            for p in ps
            for q in p["questions"]
        ]
        if expected != row["rows"] or array.shape != (2 * len(ps), 3, 1536):
            raise ValueError("text routing/shape mismatch")
        if not np.isfinite(array).all():
            raise ValueError("nonfinite text")
        for pair in ps:
            if identities[pair["video_id"]] != split:
                raise ValueError("split mismatch")
            if [q["condition"] for q in pair["questions"]] != ["initial", "final"]:
                raise ValueError("condition ordering mismatch")
        data[split] = dict(
            pairs=ps,
            text=torch.from_numpy(array),
            targets=torch.tensor([r["target"] for r in expected]),
        )
    return bank, data, hashes


def load_inputs(video_dir, text_dir):
    bank, data, hashes = load_trajectory_inputs(video_dir, text_dir)
    scene_path = ROOT / "perception-occlusion-videomaev2-features-02/result.json"
    admission = ROOT / "perception-occlusion-scene-trajectory-admission-02.json"
    admitted = json.loads(admission.read_text())
    if admitted["status"] != "PASS_REUSE_WITH_EXPLICIT_PREPROCESSING_DIFFERENCE":
        raise ValueError("scene admission missing")
    for name, pin in admitted["ledger_hashes"].items():
        if sha(ROOT / name / "result.json") != pin:
            raise ValueError("admitted ledger changed")
    scene = json.loads(scene_path.read_text())
    trajectory = json.loads((video_dir / "result.json").read_text())
    rows = {r["video_id"]: r for r in trajectory["features"]}
    seen = set()
    for row in scene["features"]:
        key = row["video_id"]
        if key in seen or key not in bank:
            raise ValueError("scene identity mismatch")
        seen.add(key)
        old = rows[key]
        if (row["split"], row["media_sha256"]) != (old["split"], old["media_sha256"]):
            raise ValueError("scene media mismatch")
        path = Path(row["path"])
        if sha(path) != row["sha256"]:
            raise ValueError("scene feature changed")
        values = np.load(path, allow_pickle=False)
        if values.ndim != 2 or values.shape[1] != 384 or not len(values):
            raise ValueError("invalid scene shape")
        if not np.isfinite(values).all():
            raise ValueError("nonfinite scene")
        context = torch.from_numpy(values.mean(0)).expand(len(bank[key]), -1)
        bank[key] = torch.cat([bank[key], context], -1)
        hashes[str(path)] = row["sha256"]
    if seen != set(bank):
        raise ValueError("incomplete scene features")
    hashes[str(scene_path)] = sha(scene_path)
    hashes[str(admission)] = sha(admission)
    return bank, data, hashes


def batch(bank, data, indices, control="normal"):
    clips, texts, targets = [], [], []
    for index in indices:
        video_index = (index + 1) % len(data["pairs"]) if control == "mismatch" else index
        value = bank[data["pairs"][video_index]["video_id"]]
        if control in ("scene_mismatch", "trajectory_mismatch"):
            other = bank[data["pairs"][(index + 1) % len(data["pairs"])]["video_id"]]
            value = value.clone()
            if control == "scene_mismatch":
                value[:, 512:] = other[:, 512:]
            else:
                value[:, :512] = other[:, :512]
        if control == "zero":
            value = torch.zeros_like(value)
        elif control == "zero_scene":
            value = value.clone()
            value[:, 512:] = 0
        elif control == "zero_trajectory":
            value = value.clone()
            value[:, :512] = 0
        clips.extend([value, value])
        texts.append(data["text"][2 * index : 2 * index + 2])
        targets.append(data["targets"][2 * index : 2 * index + 2])
    width = max(len(x) for x in clips)
    windows = torch.zeros(len(clips), width, clips[0].shape[-1])
    mask = torch.zeros(len(clips), width, dtype=torch.bool)
    for i, clip in enumerate(clips):
        windows[i, : len(clip)] = clip
        mask[i, : len(clip)] = True
    return windows, torch.cat(texts), mask, torch.cat(targets)


def metrics(probabilities, targets):
    p, y = probabilities.tolist(), targets.tolist()
    return dict(
        count=len(y),
        accuracy=float((probabilities.argmax(-1) == targets).mean()),
        nll=negative_log_likelihood(p, y),
        brier=brier_score(p, y),
        ece=expected_calibration_error(p, y),
    )


def summarize(probabilities, targets):
    result = metrics(probabilities, targets)
    result["initial"] = metrics(probabilities[::2], targets[::2])
    result["final"] = metrics(probabilities[1::2], targets[1::2])
    correct = (probabilities.argmax(-1) == targets).reshape(-1, 2)
    result["both_correct"] = float(correct.all(-1).mean())
    changed = targets.reshape(-1, 2)[:, 0] != targets.reshape(-1, 2)[:, 1]
    for name, mask in (("changed", changed), ("unchanged", ~changed)):
        question_mask = np.repeat(mask, 2)
        result[name] = (
            dict(
                metrics(probabilities[question_mask], targets[question_mask]),
                both_correct=float(correct[mask].all(-1).mean()),
            )
            if mask.any()
            else None
        )
    return result


def predict(model, bank, data, control="normal"):
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data["pairs"]), 8):
            v, text, mask, _ = batch(
                bank, data, range(start, min(start + 8, len(data["pairs"]))), control
            )
            logits = model(v, text, mask)
            torch.testing.assert_close(model(v, text.flip(1), mask), logits.flip(1))
            values.append(logits.softmax(-1).numpy())
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite predictions")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank, data, hashes = load_inputs(args.video, args.text)
    video_dim = next(iter(bank.values())).shape[-1]
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    write(
        args.output / "config.json",
        dict(
            seed=17,
            epochs=50,
            learning_rate=0.001,
            weight_decay=0.01,
            batch_video_pairs=8,
            hidden=64,
            video_dim=video_dim,
            selector="earliest minimum validation mean-question NLL",
            input_hashes=hashes,
            pair_hashes=PINS,
            torch=torch.__version__,
            python=platform.python_version(),
            device="cpu",
            script_sha256=sha(Path(__file__)),
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        ),
    )
    results = {}
    for mode in ("attention",):
        folder = args.output / mode
        folder.mkdir()
        torch.manual_seed(17)
        generator = torch.Generator().manual_seed(17)
        model = SceneTrajectoryReadout()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        best, best_epoch, history, updates = float("inf"), None, [], 0
        start = time.perf_counter()
        torch.save(model.state_dict(), folder / "untrained.pt")
        write(
            folder / "untrained-predictions.json",
            {split: predict(model, bank, data[split]).tolist() for split in data},
        )
        for epoch in range(1, 51):
            model.train()
            order = torch.randperm(len(data["train"]["pairs"]), generator=generator).tolist()
            for offset in range(0, len(order), 8):
                v, text, mask, targets = batch(bank, data["train"], order[offset : offset + 8])
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(v, text, mask), targets)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                loss.backward()
                if any(
                    p.grad is not None and not p.grad.isfinite().all() for p in model.parameters()
                ):
                    raise ValueError("nonfinite gradient")
                optimizer.step()
                updates += 1
            measurements = {}
            for split in data:
                p = predict(model, bank, data[split])
                measurements[split] = summarize(p, data[split]["targets"].numpy())
            score = measurements["valid"]["nll"]
            if score < best:
                best, best_epoch = score, epoch
                torch.save(model.state_dict(), folder / "best.pt")
            history.append(
                dict(
                    epoch=epoch,
                    updates=updates,
                    presentations=epoch * len(data["train"]["targets"]),
                    best_epoch=best_epoch,
                    seconds=time.perf_counter() - start,
                    **measurements,
                )
            )
            write(folder / "history.json", history)
            print(mode, epoch, score, "best", best_epoch, flush=True)
        torch.save(model.state_dict(), folder / "final.pt")
        write(
            folder / "final-predictions.json",
            {split: predict(model, bank, data[split]).tolist() for split in data},
        )
        model.load_state_dict(torch.load(folder / "best.pt", weights_only=True))
        predictions, controls = {}, {}
        for control in (
            "normal",
            "zero",
            "mismatch",
            "scene_mismatch",
            "trajectory_mismatch",
            "zero_scene",
            "zero_trajectory",
        ):
            p = predict(model, bank, data["valid"], control)
            predictions[control] = p.tolist()
            controls[control] = summarize(p, data["valid"]["targets"].numpy())
        restored = SceneTrajectoryReadout()
        restored.load_state_dict(torch.load(folder / "best.pt", weights_only=True))
        np.testing.assert_array_equal(predictions["normal"], predict(restored, bank, data["valid"]))
        write(
            folder / "predictions.json",
            dict(
                pairs=data["valid"]["pairs"],
                probabilities=predictions,
                targets=data["valid"]["targets"].tolist(),
            ),
        )
        results[mode] = dict(
            best_epoch=best_epoch,
            controls=controls,
            run_presentations=50 * len(data["train"]["targets"]),
            selected_presentations=best_epoch * len(data["train"]["targets"]),
            unique_train_questions=len(data["train"]["targets"]),
            unique_train_videos=len(data["train"]["pairs"]),
            updates=updates,
            seconds=time.perf_counter() - start,
            parameters=sum(p.numel() for p in model.parameters()),
            effective_parameters=sum(
                p.numel()
                for n, p in model.named_parameters()
                if mode != "mean" or not n.startswith(("key.", "position."))
            ),
            artifacts={
                p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
                for p in folder.iterdir()
                if p.is_file()
            },
            reload_exact=True,
        )
        write(args.output / "result.json", results)


if __name__ == "__main__":
    main()
