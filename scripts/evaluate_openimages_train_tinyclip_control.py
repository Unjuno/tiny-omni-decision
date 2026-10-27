"""CPU TinyCLIP image controls on a fixed train-candidate diagnostic."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from scripts.audit_openimages_train_option_prior import HASHES
from scripts.audit_openimages_train_option_prior import PROTOCOL as TASK_PROTOCOL
from scripts.evaluate_openimages_tinyclip_zero_shot import (
    EXPECTED_MODEL_SHA256,
    PROMPT_TEMPLATE,
    gather_choice_logits,
    probabilities_from_logits,
    quality_metrics,
    score_images,
)
from scripts.train_source_unimodal_cv import digest

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
TASK = ROOT / "openimages-train-option-prior-20261027-attempt-01/private-predictions.jsonl"
TASK_SHA = "df80fe10fe4c41af2192afc9a4f8e0b9a4751f7d7df26fe2d49ebe7914751bba"
RECEIPTS = ROOT / (
    "openimages-in-scope-train-candidates-20261027-attempt-01/private-page-checks.jsonl"
)
RECEIPTS_SHA = "5f2b299dde5646711ab0a785643904aa0dd993f3bcd710cd9b76eadd568db9fc"
MODEL = Path("C:/CodexArtifacts/pretrained-reuse-r1/tinyclip-39m19m-audit-01/model")
MEDIA_ROOT = ROOT / "openimages-in-scope-train-media-20261027-attempt-01/train"
PROTOCOL = Path("docs/pretrained_reuse/R2_OPENIMAGES_TRAIN_TINYCLIP_CONTROL_PROTOCOL_20261027.md")
REPO = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def fixed_derangement(count):
    permutation = list(range(count))
    rng = random.Random(20261026)
    for _ in range(1000):
        rng.shuffle(permutation)
        if all(i != donor for i, donor in enumerate(permutation)):
            return permutation
    raise ValueError("unable to construct fixed derangement")


def prepare(task, media, receipts, names, media_root):
    media_by_id = {row["image_id"]: row for row in media if row["status"] == "saved"}
    receipts_by_id = {row["ImageID"]: row for row in receipts}
    if len(task) != 257 or len({row["image_id"] for row in task}) != len(task):
        raise ValueError("unexpected task identities")
    if set(media_by_id) != {row["image_id"] for row in task}:
        raise ValueError("task and saved media identities differ")
    rows, paths = [], []
    root = media_root.resolve(strict=True)
    for row in task:
        image_id = row["image_id"]
        media_row = media_by_id[image_id]
        receipt = receipts_by_id[image_id]
        if (receipt.get("landing_page_check_status") != "metadata_confirmed_on_landing_page"
                or receipt.get("page_license_and_attribution_match") is not True):
            raise ValueError("page declaration mismatch")
        relative = Path(media_row["local_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe image path")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or digest(path) != media_row["media_sha256"]:
            raise ValueError("media path or hash mismatch")
        options = row["options"]
        if len(options) != 4 or len(set(options)) != 4 or row["target"] not in options:
            raise ValueError("invalid four-choice task")
        rows.append(row | {"options": [names[x] for x in options], "target": names[row["target"]]})
        paths.append(path)
    return rows, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError("new external output directory required")
    for path, expected in HASHES.items():
        if digest(path) != expected:
            raise ValueError("source hash mismatch")
    if digest(TASK) != TASK_SHA or digest(RECEIPTS) != RECEIPTS_SHA:
        raise ValueError("task or page receipts hash mismatch")
    if digest(MODEL / "model.safetensors") != EXPECTED_MODEL_SHA256:
        raise ValueError("model checkpoint hash mismatch")
    scope = json.loads(list(HASHES)[2].read_bytes())["selected_target_classes"]
    names = {item["label_id"]: item["display_name"] for item in scope}
    rows, paths = prepare(load_jsonl(TASK), load_jsonl(list(HASHES)[1]), load_jsonl(RECEIPTS),
                          names, MEDIA_ROOT)
    import torch
    from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForZeroShotImageClassification.from_pretrained(MODEL, local_files_only=True)
    model = model.cpu().eval().requires_grad_(False)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != 83_101_185:
        raise ValueError("unexpected parameter count")
    labels = sorted(names.values())
    label_index = {label: i for i, label in enumerate(labels)}
    text = processor(text=[PROMPT_TEMPLATE.format(label=x.lower()) for x in labels],
                     padding=True, truncation=True, return_tensors="pt")
    started = time.perf_counter()
    print("Running aligned image scoring on CPU", flush=True)
    aligned, aligned_seconds = score_images(model, processor, text, paths, row_count=len(rows),
                                            batch_size=8)
    permutation = fixed_derangement(len(rows))
    print("Running blank image control on CPU", flush=True)
    blank, blank_seconds = score_images(model, processor, text, None, row_count=1,
                                        batch_size=1, blank=True)
    scores = {"aligned": aligned, "deranged": aligned[permutation],
              "blank": np.repeat(blank, len(rows), axis=0)}
    targets = np.asarray([row["options"].index(row["target"]) for row in rows], dtype=np.int64)
    probs = {name: probabilities_from_logits(gather_choice_logits(values, rows, label_index))
             for name, values in scores.items()}
    aligned_predictions = probs["aligned"].argmax(axis=1)
    conditions = {}
    for name, values in scores.items():
        logits = gather_choice_logits(values, rows, label_index)
        predictions = probs[name].argmax(axis=1)
        conditions[name] = {
            "metrics": quality_metrics(logits, targets),
            "top1_agreement_with_aligned": float(np.mean(predictions == aligned_predictions)),
            "correctness_changes_vs_aligned": int(np.count_nonzero(
                (predictions == targets) != (aligned_predictions == targets))),
        }
    prior = np.asarray([row["probabilities"] for row in rows])
    model_files = {file.name: digest(file) for file in MODEL.iterdir() if file.is_file()}
    result = {
        "scope": "train-candidate diagnostic, not blind validation", "count": len(rows),
        "conditions": conditions,
        "prior_option_only_accuracy": float(np.mean(prior.argmax(axis=1) == targets)),
        "parameter_count": parameters, "model_files_sha256": model_files,
        "task_sha256": TASK_SHA, "page_receipts_sha256": RECEIPTS_SHA,
        "task_protocol_sha256": digest(TASK_PROTOCOL), "protocol_sha256": digest(PROTOCOL),
        "script_sha256": digest(Path(__file__)), "aligned_seconds": aligned_seconds,
        "blank_seconds": blank_seconds, "wall_seconds": time.perf_counter() - started,
        "image_derangement_seed": 20261026, "device": "cpu",
    }
    output.mkdir(parents=True)
    payload = [{"image_id": row["image_id"], "target": row["target"],
                "options": row["options"],
                "probabilities": {k: v[i].tolist() for k, v in probs.items()},
                "donor_image_id": rows[permutation[i]]["image_id"]}
               for i, row in enumerate(rows)]
    (output / "private-predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in payload), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
