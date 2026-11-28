"""Create readable, label-blind single-pair cards from a frozen review pack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

ANNOTATION_COLUMNS = (
    "review_id",
    "left_contact_observed",
    "right_contact_observed",
    "uncertain_or_occluded",
    "evidence_notes",
    "reviewer",
)
REVIEW_SEED = 20261129
TILE = 256
GRID_COLUMNS = 4
FRAMES_PER_CLIP = 32
PAIR_BAR_HEIGHT = 22


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _card_regions(
    size: tuple[int, int], *, tile: int = TILE, frames_per_clip: int = FRAMES_PER_CLIP
) -> list[tuple[int, int, int, int]]:
    panel_height = (frames_per_clip // GRID_COLUMNS) * tile + PAIR_BAR_HEIGHT
    panel_width = 2 * GRID_COLUMNS * tile
    if frames_per_clip % GRID_COLUMNS or size[0] != panel_width or size[1] % panel_height:
        raise ValueError(f"unexpected contact-sheet dimensions: {size}")
    return [
        (0, top, panel_width, top + panel_height)
        for top in range(0, size[1], panel_height)
    ]


def build_review_cards(
    image_dir: Path, output_dir: Path, *, seed: int = REVIEW_SEED
) -> dict[str, Any]:
    """Split the existing frozen sheets; never read or require the answer key."""
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("review-card creation requires Pillow") from error

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite review cards: {output_dir}")
    report_path = image_dir / "review-pack-report.json"
    template_path = image_dir / "annotation-template.csv"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete_train_only_blinded_review_pack":
        raise ValueError("source is not a completed train-only blinded review pack")
    if report.get("labels_and_sample_ids_in_images") is not False:
        raise ValueError("source report does not certify label-blind images")
    if report.get("pair_count") != 75 or report.get("frames_per_clip") != FRAMES_PER_CLIP:
        raise ValueError("unexpected review-pack pair or frame count")
    if not template_path.is_file():
        raise FileNotFoundError(template_path)

    source_hashes = report.get("image_sha256")
    if not isinstance(source_hashes, dict):
        raise ValueError("source image hashes are missing")
    cards: list[tuple[str, int]] = []
    for name, expected_hash in sorted(source_hashes.items()):
        source_path = image_dir / name
        if not source_path.is_file() or _sha256(source_path) != expected_hash:
            raise ValueError(f"source sheet hash mismatch: {name}")
        with Image.open(source_path) as sheet:
            cards.extend((name, row) for row, _box in enumerate(_card_regions(sheet.size)))

    if len(cards) != report["pair_count"]:
        raise ValueError(f"expected {report['pair_count']} cards, found {len(cards)}")
    rng = random.Random(seed)
    rng.shuffle(cards)
    output_dir.mkdir(parents=True)
    card_dir = output_dir / "cards"
    card_dir.mkdir()
    output_hashes: dict[str, str] = {}
    for index, (source_name, row) in enumerate(cards, start=1):
        name = f"card-{index:03d}.png"
        target = card_dir / name
        with Image.open(image_dir / source_name) as sheet:
            box = _card_regions(sheet.size)[row]
            sheet.crop(box).save(target, format="PNG", optimize=True)
        output_hashes[name] = _sha256(target)

    # Preserve the blank response schema, not any source-to-card mapping.
    with template_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != ANNOTATION_COLUMNS:
            raise ValueError("unexpected annotation template columns")
        rows = list(reader)
    if len(rows) != report["pair_count"] or any(
        any(row.get(key, "") for key in ANNOTATION_COLUMNS[1:]) for row in rows
    ):
        raise ValueError("source response template is not blank and complete")
    with (output_dir / "annotation-template.csv").open(
        "w", encoding="utf-8", newline=""
    ) as sink:
        writer = csv.DictWriter(sink, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    card_width = 2 * GRID_COLUMNS * TILE
    card_height = (FRAMES_PER_CLIP // GRID_COLUMNS) * TILE + PAIR_BAR_HEIGHT
    manifest = {
        "status": "train_only_blind_review_cards",
        "seed": seed,
        "pair_count": len(cards),
        "card_size": [card_width, card_height],
        "cards_sha256": output_hashes,
        "source_image_hashes_verified": len(source_hashes),
        "answer_key_read": False,
        "source_to_card_mapping_saved": False,
    }
    (output_dir / "review-card-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Physion++ train-only blinded review cards\n\n"
        "Open the PNG files in `cards/` one at a time; each shows one randomized, "
        "matched pair and its visible review ID. Record judgments in the supplied "
        "blank `annotation-template.csv`. Use the repository's frozen neutral "
        "review instructions. Do not infer one side from the other. The answer key "
        "was not read or copied. This pack is train-only and already observed; it "
        "is not a blind model-quality benchmark.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=REVIEW_SEED)
    args = parser.parse_args()
    print(json.dumps(build_review_cards(args.image_dir, args.output_dir, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
