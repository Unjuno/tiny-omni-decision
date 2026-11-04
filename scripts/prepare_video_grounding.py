"""Prepare independent all-descriptive grounding inputs from existing scenes."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.prepare_clevrer_reasoning_controls import QUESTION_HASHES
from scripts.range_generalization import sha, write
from tiny_omni_decision.clevrer import (
    clevrer_binary_choice_content_key,
    expand_descriptive_questions,
)


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    sample = root / "clevrer-expanded-seed17-256train-200val"
    output = root / "video-grounding-descriptive-inputs-01"
    if output.exists():
        raise FileExistsError(output)
    cache = root / "videoprism-clevrer-expanded-features-v0/video-features.npz"
    report_path = cache.parent / "cache-report.json"
    report = json.loads(report_path.read_text())
    if sha(cache) != report["feature_cache_sha256"]:
        raise ValueError("feature cache hash mismatch")
    with np.load(cache, allow_pickle=False) as payload:
        cached_media = set(payload["media_sha256"].astype(str).tolist())
        if payload["features"].shape != (len(cached_media), 768):
            raise ValueError("unexpected feature shape")
        if not np.isfinite(payload["features"]).all():
            raise ValueError("nonfinite features")
    expanded, hashes = {}, {}
    for split in ("train", "validation"):
        source = root / "clevrer-sample-seed17" / f"{split}-questions.json"
        manifest = sample / f"{split}.jsonl"
        if sha(source) != QUESTION_HASHES[split]:
            raise ValueError("source revision mismatch")
        if sha(manifest) != report["sample_manifest_sha256"][split]:
            raise ValueError("cache manifest mismatch")
        sampled = [json.loads(line) for line in manifest.read_text().splitlines()]
        raw = json.loads(source.read_text())
        rows = expand_descriptive_questions(raw, sampled, split=split)
        scene_ids = {r["scene_index"] for r in sampled}
        expected = sum(
            q["question_type"] == "descriptive"
            for r in raw
            if r["scene_index"] in scene_ids
            for q in r["questions"]
        )
        if len(rows) != expected:
            raise ValueError("unsupported descriptive labels would be silently omitted")
        for row in rows:
            path = sample / row["media_path"]
            if not path.is_file() or row["media_sha256"] not in cached_media:
                raise ValueError("missing media or features")
            row["media_path"] = str(path)
        expanded[split] = rows
        hashes.update({str(p): sha(p) for p in (source, manifest)})
    for key in ("scene_index", "media_sha256", "scene_group_id", "id"):
        if {r[key] for r in expanded["train"]} & {r[key] for r in expanded["validation"]}:
            raise ValueError(f"cross-split overlap: {key}")
    val_content = {clevrer_binary_choice_content_key(r) for r in expanded["validation"]}
    removed = [
        r["id"] for r in expanded["train"] if clevrer_binary_choice_content_key(r) in val_content
    ]
    expanded["train"] = [r for r in expanded["train"] if r["id"] not in set(removed)]
    if not expanded["train"]:
        raise ValueError("no content-disjoint training records")
    output.mkdir()
    for split, rows in expanded.items():
        path = output / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
        )
        hashes[str(path)] = sha(path)
    result = dict(
        status="PREPARED_NOT_TRAINED",
        source_hashes=hashes,
        feature_cache=str(cache),
        feature_cache_sha256=sha(cache),
        removed_training_ids=removed,
        validation_removed=0,
        counts={
            s: dict(
                questions=len(rows),
                scenes=len({r["scene_index"] for r in rows}),
                taxonomy=dict(Counter(r["taxonomy"] for r in rows)),
                temporal=dict(Counter(r["question_type"] for r in rows)),
            )
            for s, rows in expanded.items()
        },
    )
    write(output / "report.json", result)
    write(Path("docs/pretrained_reuse/R2_VIDEO_GROUNDING_INPUTS_20261012.json"), result)
    print(json.dumps(result["counts"], indent=2))
    print("Excluded training questions:", len(removed))


if __name__ == "__main__":
    main()
