"""Fixed CPU TinyCLIP diagnostic on 15 same-option train-image quartets."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.acquire_openimages_quartet_media import page_matched_quartets
from scripts.audit_openimages_train_option_prior import HASHES
from scripts.evaluate_openimages_tinyclip_zero_shot import (
    EXPECTED_MODEL_SHA256,
    PROMPT_TEMPLATE,
    gather_choice_logits,
    probabilities_from_logits,
    quality_metrics,
    score_images,
)
from scripts.evaluate_openimages_train_tinyclip_control import (
    MEDIA_ROOT,
    MODEL,
    RECEIPTS,
    RECEIPTS_SHA,
    fixed_derangement,
    load_jsonl,
)
from scripts.train_source_unimodal_cv import digest

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
GROUPS = ROOT / "openimages-balanced-quartets-20261027-attempt-01/private-quartets.json"
GROUPS_SHA = "12c9eca99a43c543c3dfcb4e2b565a7bdd7bf01731dc09d0314f44a2619ab901"
PROTOCOL = Path("docs/pretrained_reuse") / (
    "R2_OPENIMAGES_BALANCED_QUARTET_TINYCLIP_PROTOCOL_20261027.md"
)
REPO = Path(__file__).resolve().parents[1]


def prepare(groups, receipts, media, names, root):
    if len(groups) != 15 or len(page_matched_quartets(groups, receipts)) != 15:
        raise ValueError("the fixed 15 quartets must retain all matching page receipts")
    index = {row["image_id"]: row for row in media if row["status"] == "saved"}
    root = root.resolve(strict=True)
    rows, paths = [], []
    for group_index, group in enumerate(groups):
        for row in group["rows"]:
            media_row = index[row["image_id"]]
            relative = Path(media_row["local_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe media path")
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root) or digest(path) != media_row["media_sha256"]:
                raise ValueError("media hash mismatch")
            rows.append({"image_id": row["image_id"], "group": group_index,
                         "target": names[row["target"]],
                         "options": [names[x] for x in group["options"]]})
            paths.append(path)
    return rows, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError("new external output directory required")
    if any(digest(path) != expected for path, expected in HASHES.items()):
        raise ValueError("pinned candidate inputs changed")
    if digest(GROUPS) != GROUPS_SHA or digest(RECEIPTS) != RECEIPTS_SHA:
        raise ValueError("group or page receipt mismatch")
    if digest(MODEL / "model.safetensors") != EXPECTED_MODEL_SHA256:
        raise ValueError("checkpoint mismatch")
    scope = json.loads(list(HASHES)[2].read_bytes())["selected_target_classes"]
    names = {item["label_id"]: item["display_name"] for item in scope}
    groups = json.loads(GROUPS.read_bytes())
    rows, paths = prepare(groups, load_jsonl(RECEIPTS), load_jsonl(list(HASHES)[1]),
                          names, MEDIA_ROOT)
    import torch
    from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForZeroShotImageClassification.from_pretrained(MODEL, local_files_only=True)
    model = model.cpu().eval().requires_grad_(False)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 83_101_185:
        raise ValueError("unexpected parameter count")
    labels = sorted(names.values())
    indexes = {label: i for i, label in enumerate(labels)}
    text = processor(text=[PROMPT_TEMPLATE.format(label=x.lower()) for x in labels],
                     padding=True, truncation=True, return_tensors="pt")
    start = time.perf_counter()
    aligned, aligned_sec = score_images(model, processor, text, paths, row_count=len(rows),
                                        batch_size=8)
    blank, blank_sec = score_images(model, processor, text, None, row_count=1,
                                    batch_size=1, blank=True)
    permutation = fixed_derangement(len(rows))
    conditions = {"aligned": aligned, "deranged": aligned[permutation],
                  "blank": np.repeat(blank, len(rows), axis=0)}
    targets = np.asarray([row["options"].index(row["target"]) for row in rows], dtype=np.int64)
    logits = {
        key: gather_choice_logits(values, rows, indexes) for key, values in conditions.items()
    }
    probs = {key: probabilities_from_logits(values) for key, values in logits.items()}
    aligned_pred = probs["aligned"].argmax(axis=1)
    metrics = {}
    for key, values in logits.items():
        predicted = probs[key].argmax(axis=1)
        metrics[key] = {"metrics": quality_metrics(values, targets),
                        "agreement_with_aligned": float(np.mean(predicted == aligned_pred)),
                        "correctness_changes": int(np.count_nonzero(
                            (predicted == targets) != (aligned_pred == targets))),
                        "correct_by_quartet": [int(np.sum(predicted[i:i + 4] == targets[i:i + 4]))
                                               for i in range(0, len(rows), 4)]}
    result = {"scope": "exploratory train quartets, not independent evaluation",
              "groups": len(groups), "count": len(rows), "conditions": metrics,
              "option_only_accuracy": 0.25, "parameter_count": parameters,
              "groups_sha256": GROUPS_SHA, "page_receipts_sha256": RECEIPTS_SHA,
              "model_files_sha256": {p.name: digest(p) for p in MODEL.iterdir() if p.is_file()},
              "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
              "aligned_seconds": aligned_sec, "blank_seconds": blank_sec,
              "wall_seconds": time.perf_counter() - start, "device": "cpu"}
    output.mkdir(parents=True)
    payload = [{**row, "probabilities": {key: value[i].tolist() for key, value in probs.items()},
                "donor_image_id": rows[permutation[i]]["image_id"]} for i, row in enumerate(rows)]
    (output / "private-predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in payload), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
