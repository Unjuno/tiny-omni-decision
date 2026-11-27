from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

pytest.importorskip("torch")

from scripts.train_bsd10k_audio_readout import (
    ENCODER_COMMIT,
    ENCODER_PARAMETERS,
    ENCODER_SHA256,
    LABELS,
    run,
    sha256_file,
)


def test_bsd10k_audio_readout_checks_splits_calibrates_and_reloads(tmp_path):
    rows = []
    examples = []
    next_id = 1
    feature_values = {"train": [], "validation": []}
    split_ids = {"train": [], "validation": []}
    for split, count_per_class in (("train", 6), ("validation", 2), ("test", 1)):
        for class_index, class_id in enumerate(LABELS):
            for offset in range(count_per_class):
                sound_id = str(next_id)
                next_id += 1
                uploader = f"{split}-{class_id}-{offset}"
                group_hash = hashlib.sha256(uploader.casefold().encode()).hexdigest()
                rows.append(
                    {
                        "sound_id": sound_id,
                        "class": class_id,
                        "class_idx": str(class_index),
                        "class_top": class_id.split("-")[0],
                        "confidence": "4",
                        "uploader": uploader,
                        "license": "https://creativecommons.org/licenses/by/4.0/",
                        "title": "not used",
                        "tags": "not used",
                        "description": "not used",
                    }
                )
                examples.append(
                    {
                        "sound_id": sound_id,
                        "class": class_id,
                        "license": "CC-BY-4.0",
                        "uploader_group_sha256": group_hash,
                        "split": split,
                    }
                )
                if split in feature_values:
                    vector = np.zeros(384, dtype=np.float32)
                    vector[class_index] = 1.0
                    vector[23 + class_index] = 0.1
                    feature_values[split].append(vector)
                    split_ids[split].append(sound_id)

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    metadata_zip = tmp_path / "metadata.zip"
    with zipfile.ZipFile(metadata_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata/BSD10k_metadata.csv", buffer.getvalue())
    metadata_hash = sha256_file(metadata_zip)
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "tiny-omni-decision/bsd10k-group-split-v1",
                "source_record_id": "19868804",
                "metadata_zip_sha256": metadata_hash,
                "examples": examples,
            }
        ),
        encoding="utf-8",
    )
    manifest_hash = sha256_file(manifest_path)

    feature_dirs = {}
    for split in ("train", "validation"):
        directory = tmp_path / f"{split}-features"
        directory.mkdir()
        feature_path = directory / "features.npy"
        np.save(feature_path, np.stack(feature_values[split]), allow_pickle=False)
        (directory / "sample-index.json").write_text(
            json.dumps({"split": split, "sound_ids": split_ids[split]}),
            encoding="utf-8",
        )
        (directory / "run.json").write_text(
            json.dumps(
                {
                    "status": "PASS_FROZEN_FEATURES",
                    "partial_smoke": False,
                    "split": split,
                    "model_parameters": ENCODER_PARAMETERS,
                    "weights_sha256": ENCODER_SHA256,
                    "source_commit": ENCODER_COMMIT,
                    "split_manifest_sha256": manifest_hash,
                    "feature_sha256": sha256_file(feature_path),
                }
            ),
            encoding="utf-8",
        )
        feature_dirs[split] = directory

    report = run(
        metadata_zip,
        manifest_path,
        feature_dirs["train"],
        feature_dirs["validation"],
        tmp_path / "readout-run",
    )

    assert report["test_split_loaded"] is False
    assert report["selected_epoch"] >= 1
    assert report["train_validation_group_overlap"] == 0
    assert report["save_reload_exact_logits"] is True
    assert report["all_persistent_parameters"] == ENCODER_PARAMETERS + 384 * len(LABELS) + len(
        LABELS
    )
    assert report["validation_macro_metrics_uncalibrated"]["accuracy"] > 0.5
    assert len(
        (tmp_path / "readout-run" / "validation-predictions.jsonl").read_text().splitlines()
    ) == len(split_ids["validation"])
