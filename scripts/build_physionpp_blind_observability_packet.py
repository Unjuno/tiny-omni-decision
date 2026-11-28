"""Create a label-blinded human review packet from frozen Physion++ pair sheets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import secrets
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

SHEET_WIDTH = 2048
SHEET_TOP = 8
PANEL_WIDTH = 1024
PANEL_HEIGHT = 512
PAIR_STRIDE = 540
CARD_HEADER = 24


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pairs(path: Path) -> list[dict[str, Any]]:
    pairs = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not pairs:
        raise ValueError("pair manifest is empty")
    ids = [pair.get("pair_id") for pair in pairs]
    if any(not isinstance(pair_id, str) or not pair_id for pair_id in ids):
        raise ValueError("every pair needs a non-empty pair_id")
    if len(set(ids)) != len(ids):
        raise ValueError("pair_id values must be unique")
    if any(
        not isinstance(pair.get("scenario"), str)
        or not isinstance(pair.get("no_sample_id"), str)
        or not isinstance(pair.get("yes_sample_id"), str)
        for pair in pairs
    ):
        raise ValueError("pairs require scenario and both source sample IDs")
    return pairs


def _assert_separate_paths(packet_dir: Path, key_dir: Path) -> None:
    packet = packet_dir.resolve()
    key = key_dir.resolve()
    if packet == key or packet in key.parents or key in packet.parents:
        raise ValueError("private answer-key directory must be separate from review packet")
    if packet.exists() or key.exists():
        raise FileExistsError("refusing to overwrite an existing review packet or answer key")


def _crop_panel(sheet: Image.Image, scenario_row: int, side: str) -> Image.Image:
    x = 0 if side == "left" else PANEL_WIDTH
    y = SHEET_TOP + scenario_row * PAIR_STRIDE + CARD_HEADER + 4
    panel = sheet.crop((x, y, x + PANEL_WIDTH, y + PANEL_HEIGHT))
    if panel.size != (PANEL_WIDTH, PANEL_HEIGHT):
        raise ValueError("unexpected contact-sheet panel geometry")
    return panel


def build_packet(
    pair_manifest: Path,
    contact_sheet_dir: Path,
    packet_dir: Path,
    private_key_dir: Path,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build anonymous review cards; keep the label mapping in a separate folder."""
    _assert_separate_paths(packet_dir, private_key_dir)
    pairs = _read_pairs(pair_manifest)
    local_seed = seed if seed is not None else secrets.randbits(128)
    rng = random.Random(local_seed)

    rows_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        rows_by_scenario.setdefault(pair["scenario"], []).append(pair)

    panels: dict[str, tuple[Image.Image, Image.Image]] = {}
    for scenario, scenario_pairs in rows_by_scenario.items():
        sheet_path = contact_sheet_dir / f"{scenario}.png"
        with Image.open(sheet_path) as source:
            sheet = source.convert("RGB")
        expected_height = SHEET_TOP + len(scenario_pairs) * PAIR_STRIDE
        if sheet.width != SHEET_WIDTH or sheet.height < expected_height:
            raise ValueError(f"unexpected contact-sheet dimensions for {scenario}")
        for row_index, pair in enumerate(scenario_pairs):
            panels[pair["pair_id"]] = (
                _crop_panel(sheet, row_index, "left"),
                _crop_panel(sheet, row_index, "right"),
            )

    shuffled = list(pairs)
    rng.shuffle(shuffled)
    packet_dir.mkdir(parents=True)
    private_key_dir.mkdir(parents=True)
    (packet_dir / "cards").mkdir()

    manifest_cards: list[dict[str, str]] = []
    answer_cards: list[dict[str, str]] = []
    response_path = packet_dir / "responses.csv"
    with response_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["review_id", "choice", "confidence_1_to_5", "notes"],
        )
        writer.writeheader()
        for index, pair in enumerate(shuffled, start=1):
            review_id = f"item-{index:03d}-{rng.getrandbits(32):08x}"
            source_left, source_right = panels[pair["pair_id"]]
            if rng.getrandbits(1):
                left_panel, right_panel = source_left, source_right
                left_is_no = True
            else:
                left_panel, right_panel = source_right, source_left
                left_is_no = False

            card = Image.new("RGB", (SHEET_WIDTH, PANEL_HEIGHT + CARD_HEADER), "white")
            draw = ImageDraw.Draw(card)
            draw.text((8, 5), f"{review_id}  |  LEFT", fill="black")
            draw.text((PANEL_WIDTH + 8, 5), "RIGHT", fill="black")
            card.paste(left_panel, (0, CARD_HEADER))
            card.paste(right_panel, (PANEL_WIDTH, CARD_HEADER))
            filename = f"{review_id}.png"
            card_path = packet_dir / "cards" / filename
            card.save(card_path, format="PNG", optimize=True)

            writer.writerow(
                {
                    "review_id": review_id,
                    "choice": "",
                    "confidence_1_to_5": "",
                    "notes": "",
                }
            )
            manifest_cards.append(
                {
                    "review_id": review_id,
                    "file": f"cards/{filename}",
                    "sha256": sha256_file(card_path),
                }
            )
            answer_cards.append(
                {
                    "review_id": review_id,
                    "pair_id": pair["pair_id"],
                    "left_sample_id": (
                        pair["no_sample_id"] if left_is_no else pair["yes_sample_id"]
                    ),
                    "right_sample_id": (
                        pair["yes_sample_id"] if left_is_no else pair["no_sample_id"]
                    ),
                    "correct_contact_side": "right" if left_is_no else "left",
                }
            )

    instructions = (
        "# Blinded Physion++ contact observability review\n\n"
        "For each paired card, decide which clip, if either, visibly shows the red object "
        "contacting the yellow-marked target. Frame tiles run left-to-right, then top-to-bottom; "
        "they sample the clip timeline uniformly. Do not infer the answer from the fact that the "
        "two clips are paired. Use `left`, `right`, `both`, `neither`, or `unclear` in `choice`; "
        "record confidence from 1 (low) to 5 (high), and note ambiguity or missing temporal "
        "evidence. "
        "The response file contains no labels, scenario names, source IDs, or factor metadata.\n"
    )
    (packet_dir / "README.md").write_text(instructions, encoding="utf-8")
    public_manifest = {
        "schema_version": 1,
        "status": "label_blinded_train_only_review_packet",
        "items": len(manifest_cards),
        "frame_tiles_per_clip": 32,
        "card_manifest": manifest_cards,
        "responses_csv_sha256": sha256_file(response_path),
        "source_data_scope": "existing train-only matched-pair contact sheets",
    }
    (packet_dir / "packet-manifest.json").write_text(
        json.dumps(public_manifest, indent=2) + "\n", encoding="utf-8"
    )
    private_key = {
        "schema_version": 1,
        "seed": local_seed,
        "pair_manifest_sha256": sha256_file(pair_manifest),
        "answer_cards": answer_cards,
    }
    (private_key_dir / "answer-key.json").write_text(
        json.dumps(private_key, indent=2) + "\n", encoding="utf-8"
    )
    return public_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--contact-sheet-dir", type=Path, required=True)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--private-key-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_packet(
        args.pair_manifest.resolve(strict=True),
        args.contact_sheet_dir.resolve(strict=True),
        args.packet_dir.resolve(),
        args.private_key_dir.resolve(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
