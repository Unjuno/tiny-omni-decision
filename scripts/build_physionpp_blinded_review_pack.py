"""Build a label-blinded, train-only review pack for Physion++ video pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

try:
    from scripts.audit_physionpp_train_data_quality import (
        VISUAL_FRAMES,
        _aspect_preserving_tile,
        _relative_media_path,
        audit_train_rows,
        load_train_rows,
        pair_visual_review_rows,
        select_visual_review_rows,
        sha256_file,
    )
except ModuleNotFoundError:
    from audit_physionpp_train_data_quality import (
        VISUAL_FRAMES,
        _aspect_preserving_tile,
        _relative_media_path,
        audit_train_rows,
        load_train_rows,
        pair_visual_review_rows,
        select_visual_review_rows,
        sha256_file,
    )

DEFAULT_SEED = 20261128
TRAIN_MANIFEST_SHA256 = "0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b"
TILE_SIZE = (256, 256)
GRID_COLUMNS = 4
ANNOTATION_COLUMNS = (
    "review_id",
    "left_contact_observed",
    "right_contact_observed",
    "uncertain_or_occluded",
    "evidence_notes",
    "reviewer",
)


def _selected_frame_indices(frame_count: int, count: int) -> set[int]:
    if frame_count < 2 or count < 2 or count > frame_count:
        raise ValueError("frame_count and requested frame count are incompatible")
    return {round(i * (frame_count - 1) / (count - 1)) for i in range(count)}


def _randomized_target_order(rng: random.Random) -> tuple[str, str]:
    left = rng.choice(("No", "Yes"))
    return left, "Yes" if left == "No" else "No"


def build_blinded_pack(
    train_jsonl: Path,
    corpus_root: Path,
    image_dir: Path,
    key_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    frames_per_clip: int = VISUAL_FRAMES,
) -> dict[str, Any]:
    try:
        import av
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError("review pack generation requires PyAV and Pillow") from error

    if image_dir.exists() or key_dir.exists():
        raise FileExistsError("refusing to overwrite review images or private key")
    image_dir.mkdir(parents=True)
    key_dir.mkdir(parents=True)
    rows = load_train_rows(train_jsonl)
    manifest_sha = hashlib.sha256(train_jsonl.read_bytes()).hexdigest()
    if manifest_sha != TRAIN_MANIFEST_SHA256:
        raise ValueError(f"train manifest hash mismatch: {manifest_sha}")
    train_audit = audit_train_rows(rows, corpus_root)
    selected = select_visual_review_rows(rows)
    pairs = pair_visual_review_rows(selected)
    rng = random.Random(seed)
    rng.shuffle(pairs)

    by_scenario: dict[str, list[dict[str, Any]]] = {}
    answer_key: list[dict[str, Any]] = []
    panel_audit: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        review_id = f"review-{index:03d}"
        order = _randomized_target_order(rng)
        side_frames: list[list[Any]] = []
        for target in order:
            row = pair[target]
            media = row["media"][0]
            media_relative = _relative_media_path(media["path"])
            path = corpus_root.joinpath(*media_relative.parts)
            if sha256_file(path) != media["sha256"]:
                raise ValueError(f"selected train media hash mismatch: {row['id']}")
            indices = _selected_frame_indices(media["num_frames"], frames_per_clip)
            frames: list[Any] = []
            decoded = 0
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                source_size = (stream.width, stream.height)
                for frame_index, frame in enumerate(container.decode(stream)):
                    decoded = frame_index + 1
                    if frame_index in indices:
                        frames.append(
                            (
                                frame_index,
                                _aspect_preserving_tile(frame.to_image(), TILE_SIZE),
                            )
                        )
            if decoded != media["num_frames"] or len(frames) != frames_per_clip:
                raise ValueError(f"frame decode mismatch for {row['id']}")
            side_frames.append(frames)
            panel_audit.append(
                {
                    "review_id": review_id,
                    "source_frame_size": list(source_size),
                    "decoded_frames": decoded,
                    "sampled_frame_indices": [item[0] for item in frames],
                }
            )

        answer_key.append(
            {
                "review_id": review_id,
                "pair_id": pair["pair_id"],
                "scenario": pair["scenario"],
                "factors": pair["factors"],
                "left_sample_id": pair[order[0]]["id"],
                "left_target": order[0],
                "right_sample_id": pair[order[1]]["id"],
                "right_target": order[1],
            }
        )
        by_scenario.setdefault(pair["scenario"], []).append(
            {"review_id": review_id, "left": side_frames[0], "right": side_frames[1]}
        )

    for scenario, scenario_pairs in sorted(by_scenario.items()):
        scenario_pairs.sort(key=lambda item: item["review_id"])
        grid_rows = (frames_per_clip + GRID_COLUMNS - 1) // GRID_COLUMNS
        panel_width = GRID_COLUMNS * TILE_SIZE[0]
        panel_height = 22 + grid_rows * TILE_SIZE[1]
        sheet = Image.new(
            "RGB",
            (panel_width * 2, panel_height * len(scenario_pairs)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for row_index, pair in enumerate(scenario_pairs):
            y_base = row_index * panel_height
            draw.text(
                (4, y_base + 4),
                f"{pair['review_id']} | left",
                fill="black",
                font=ImageFont.load_default(),
            )
            draw.text(
                (panel_width + 4, y_base + 4),
                f"{pair['review_id']} | right",
                fill="black",
                font=ImageFont.load_default(),
            )
            for side_index, side in enumerate((pair["left"], pair["right"])):
                for position, (_frame_index, image) in enumerate(side):
                    column, line = position % GRID_COLUMNS, position // GRID_COLUMNS
                    sheet.paste(
                        image,
                        (
                            side_index * panel_width + column * TILE_SIZE[0],
                            y_base + 22 + line * TILE_SIZE[1],
                        ),
                    )
        sheet.save(image_dir / f"{scenario}.png")

    key_path = key_dir / "answer-key.jsonl"
    key_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in answer_key),
        encoding="utf-8",
    )
    with (image_dir / "annotation-template.csv").open(
        "w", encoding="utf-8", newline=""
    ) as sink:
        writer = csv.DictWriter(sink, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        for item in answer_key:
            writer.writerow({"review_id": item["review_id"]})
    image_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(image_dir.glob("*.png"))
    }
    report = {
        "status": "complete_train_only_blinded_review_pack",
        "train_manifest_sha256": manifest_sha,
        "verified_train_media_count": train_audit["train_rows"],
        "seed": seed,
        "pair_count": len(pairs),
        "frames_per_clip": frames_per_clip,
        "tile_size": list(TILE_SIZE),
        "pair_order_and_left_right_randomized": True,
        "labels_and_sample_ids_in_images": False,
        "image_sha256": image_hashes,
        "answer_key_sha256": hashlib.sha256(key_path.read_bytes()).hexdigest(),
        "frame_audit": panel_audit,
        "validation_or_sealed_data_opened": False,
        "gpu_used": False,
    }
    (image_dir / "review-pack-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--frames-per-clip", type=int, default=VISUAL_FRAMES)
    args = parser.parse_args()
    print(
        json.dumps(
            build_blinded_pack(
                args.train_jsonl,
                args.corpus_root,
                args.image_dir,
                args.key_dir,
                seed=args.seed,
                frames_per_clip=args.frames_per_clip,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
