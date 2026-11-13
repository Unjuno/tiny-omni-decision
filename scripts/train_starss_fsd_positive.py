"""One fixed positive-coverage experiment; preserve original validation and calibration."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.extract_starss23_corpus_features import read_manifest
from scripts.fetch_starss23_frozen_media import write_json
from scripts.fsd_acquisition_gate import complete_acquisition
from scripts.fsd_positive_queries import positive_queries
from scripts.train_starss23_readouts import (
    candidate_inputs,
    load_feature_matrix,
    metrics,
    predict,
    temperature_from_calibration,
)
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    media = root / "fsd50k-training-128-02"
    acquisition = complete_acquisition(media)
    screen = json.loads((root / "fsd-starss-waveform-screen-128-01/result.json").read_text())
    if (screen["status"] != "COMPLETE" or screen["pairs"] != 9984
            or screen["flagged_pairs"] or screen["exact_resampled_duplicates"]
            or screen["media_report_sha256"] != sha256_file(media / "result.json")):
        raise ValueError("waveform screen missing or requires review")
    feature_root = root / "fsd50k-features-128-01"
    report = json.loads((feature_root / "result.json").read_text())
    queries = positive_queries(acquisition["selection"])
    if (report["status"] != "PASS"
            or report["media_report_sha256"] != sha256_file(media / "result.json")
            or [r["id"] for r in report["rows"]] != [q["clip_id"] for q in queries]
            or sha256_file(feature_root / "features.npy") != report["feature_sha256"]):
        raise ValueError("auxiliary features mismatch")
    manifest = root / "starss23-all-windows-v1"
    rows = read_manifest(manifest)
    old = load_feature_matrix(root / "starss23-audio-corpus-features-v1", "audio", rows,
                              sha256_file(manifest / "result.json"))
    extra = np.load(feature_root / "features.npy", allow_pickle=False)
    if extra.shape != (128, 384) or not np.isfinite(extra).all():
        raise ValueError("invalid auxiliary matrix")
    masks = {s: np.array([r["split"] == s for r in rows])
             for s in ("train", "calibration", "validation")}
    config = yaml.safe_load(Path("configs/pretrained_reuse/path_a_starss23_events.yaml")
                            .read_text(encoding="utf-8-sig"))["readout"]
    mean, std = old[masks["train"]].mean(0), old[masks["train"]].std(0)
    std = np.where(std > 1e-6, std, 1.0)
    values = np.zeros((len(old) + len(extra), 1152), dtype=np.float32)
    values[:, :384] = (np.concatenate([old, extra]) - mean) / std
    original_targets = np.array([[q["target"] for q in r["queries"]] for r in rows])
    targets = np.full((len(values), 12), -100, dtype=np.int64)
    targets[:len(old)] = original_targets
    for q in queries:
        targets[len(old) + q["feature_index"], q["event_index"]] = 1
    train_rows = np.r_[masks["train"], np.ones(128, dtype=bool)]
    eligible = np.flatnonzero(np.repeat(train_rows, 12) & (targets.ravel() != -100))
    x = torch.from_numpy(values)
    y = torch.from_numpy(targets.ravel())
    val_x = x[:len(old)][masks["validation"]]
    val_y = torch.from_numpy(original_targets[masks["validation"]]).long()
    cal_x = x[:len(old)][masks["calibration"]]
    cal_y = torch.from_numpy(original_targets[masks["calibration"]]).long()
    output = root / "starss-fsd-positive-128-run-01"
    output.mkdir(exist_ok=False)
    write_json(output / "protocol.json", {
        "readout": config, "seed": 17, "known_train_queries": len(eligible),
        "auxiliary_queries_per_epoch": 128, "unknown_targets_sampled": False,
        "selection_sha256": sha256_file(media / "selection.json"),
        "feature_sha256": report["feature_sha256"],
        "manifest_sha256": sha256_file(manifest / "result.json"),
        "runner_sha256": sha256_file(Path(__file__)), "torch": torch.__version__,
        "device": "cpu", "blind_audit": False, "epochs_not_compute_matched": True,
    })
    validation_ids = [r["id"] for r in rows if r["split"] == "validation"]
    write_json(output / "validation_ids.json", validation_ids)
    np.savez(output / "normalization.npz", mean=mean, std=std)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    model = FrozenFeatureCandidateScorer(1166, 64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(17)
    history, best_loss, best_epoch, updates = [], float("inf"), None, 0
    started = time.perf_counter()
    for epoch in range(1, 25):
        model.train()
        loss_sum = 0.0
        order = torch.from_numpy(eligible)[torch.randperm(len(eligible), generator=generator)]
        for indices in order.split(config["batch_questions"]):
            logits = model(candidate_inputs(x, indices))
            loss = F.cross_entropy(logits, y[indices])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            optimizer.zero_grad()
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in model.parameters()):
                raise ValueError("nonfinite gradient")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            updates += 1
        logits = predict(model, val_x)
        val_loss = float(F.cross_entropy(logits.reshape(-1, 2), val_y.reshape(-1)))
        history.append({"epoch": epoch, "updates": updates, "train_ce": loss_sum / len(eligible),
                        "validation": metrics(val_y.numpy(), logits.softmax(-1)[..., 1].numpy())})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), output / "best.pt")
        write_json(output / "history.json", history)
    torch.save(model.state_dict(), output / "final.pt")
    model.load_state_dict(torch.load(output / "best.pt", weights_only=True))
    temp, grid_losses = temperature_from_calibration(predict(model, cal_x), cal_y)
    logits = predict(model, val_x)
    reloaded_loss = float(F.cross_entropy(logits.reshape(-1, 2), val_y.reshape(-1)))
    if abs(reloaded_loss - best_loss) > 1e-6:
        raise ValueError("best reload loss mismatch")
    if not torch.allclose(logits, predict(model, val_x, True), atol=1e-6, rtol=1e-5):
        raise ValueError("option permutation mismatch")
    raw = logits.softmax(-1)[..., 1].numpy()
    calibrated = (logits / temp).softmax(-1)[..., 1].numpy()
    np.savez(output / "validation_predictions.npz", targets=val_y.numpy(),
             raw_positive_probabilities=raw, positive_probabilities=calibrated)
    result = {"best_epoch": best_epoch, "temperature": temp, "calibration_losses": grid_losses,
              "validation_raw": metrics(val_y.numpy(), raw),
              "validation_calibrated": metrics(val_y.numpy(), calibrated),
              "updates": updates, "seconds": time.perf_counter() - started,
              "best_reload_nll_verified": True, "option_permutation_verified": True,
              "parameters": sum(p.numel() for p in model.parameters()),
              "best_sha256": sha256_file(output / "best.pt"),
              "final_sha256": sha256_file(output / "final.pt")}
    write_json(output / "result.json", result)
    print(json.dumps(result["validation_calibrated"]["macro"]))


if __name__ == "__main__":
    main()
