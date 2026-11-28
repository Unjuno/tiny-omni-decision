from __future__ import annotations

import json

import pytest
from PIL import Image

from scripts.build_physionpp_blind_observability_packet import build_packet


def test_packet_masks_labels_randomizes_sides_and_keeps_private_key_separate(tmp_path) -> None:
    sheet_dir = tmp_path / "sheets"
    sheet_dir.mkdir()
    sheet = Image.new("RGB", (2048, 548), "white")
    for y in range(36, 548):
        for x in range(1024):
            sheet.putpixel((x, y), (255, 0, 0))
            sheet.putpixel((x + 1024, y), (0, 0, 255))
    sheet.save(sheet_dir / "scenario.png")
    pair_manifest = tmp_path / "pairs.jsonl"
    pair_manifest.write_text(
        json.dumps(
            {
                "pair_id": "secret-pair",
                "scenario": "scenario",
                "factors": [["target", "sphere"]],
                "no_sample_id": "secret-no-id",
                "yes_sample_id": "secret-yes-id",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    packet_dir = tmp_path / "packet"
    key_dir = tmp_path / "private-answer-key"
    result = build_packet(pair_manifest, sheet_dir, packet_dir, key_dir, seed=4)

    assert result["items"] == 1
    card_file = next((packet_dir / "cards").glob("*.png"))
    with Image.open(card_file) as card:
        assert card.size == (2048, 536)
        left_pixel = card.getpixel((100, 100))
        right_pixel = card.getpixel((1200, 100))
        assert {left_pixel, right_pixel} == {(255, 0, 0), (0, 0, 255)}
    packet_text = " ".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in packet_dir.rglob("*")
        if path.is_file() and path.suffix != ".png"
    )
    assert "secret-no-id" not in packet_text
    assert "secret-yes-id" not in packet_text
    assert "secret-pair" not in packet_text
    key = json.loads((key_dir / "answer-key.json").read_text(encoding="utf-8"))
    assert key["seed"] == 4
    assert key["answer_cards"][0]["pair_id"] == "secret-pair"
    assert key["answer_cards"][0]["correct_contact_side"] in {"left", "right"}


def test_packet_refuses_answer_key_inside_review_packet(tmp_path) -> None:
    pair_manifest = tmp_path / "pairs.jsonl"
    pair_manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="separate"):
        build_packet(pair_manifest, tmp_path, tmp_path / "packet", tmp_path / "packet" / "key")


def test_packet_refuses_overwriting_existing_output(tmp_path) -> None:
    pair_manifest = tmp_path / "pairs.jsonl"
    pair_manifest.write_text("{}\n", encoding="utf-8")
    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_packet(pair_manifest, tmp_path, packet_dir, tmp_path / "key")
