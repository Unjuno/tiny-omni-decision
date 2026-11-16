from tiny_omni_decision.audio_hard_negatives import TranscriptRecord
from tiny_omni_decision.librispeech_taskadapt import build_choice_examples, split_by_speaker


def _row(sample_id, speaker, text, asset):
    return {
        "id": sample_id,
        "speaker_id": speaker,
        "chapter_id": f"chapter-{speaker}",
        "text": text,
        "audio_sha256": asset,
        "audio_path": f"/audio/{sample_id}.flac",
    }


def _record(row):
    words = tuple(row["text"].split())
    phones = tuple(phone for word in words for phone in (word, "|"))
    return TranscriptRecord(
        sample_id=row["id"],
        speaker_id=row["speaker_id"],
        chapter_id=row["chapter_id"],
        text=row["text"],
        words=words,
        phonemes=phones,
    )


def test_speaker_split_is_deterministic_and_has_no_group_or_content_overlap():
    rows = [
        _row("a", "speaker-a", "THE RED BIRD", "asset-a"),
        _row("a2", "speaker-a", "A GREEN FIELD", "asset-a2"),
        _row("b", "speaker-b", "THE RED BIRD", "asset-b"),
        _row("b2", "speaker-b", "A BLUE FIELD", "asset-b2"),
        _row("c", "speaker-c", "A SMALL HOUSE", "asset-c"),
        _row("c2", "speaker-c", "A QUIET RIVER", "asset-c2"),
        _row("d", "speaker-d", "A BRIGHT STAR", "asset-d"),
        _row("d2", "speaker-d", "A LONG ROAD", "asset-d2"),
        _row("e", "speaker-e", "THE WHITE CLOUD", "asset-e"),
        _row("e2", "speaker-e", "THE DARK TREE", "asset-e2"),
    ]
    first = split_by_speaker(rows, seed=17)
    second = split_by_speaker(rows, seed=17)
    train, validation, report = first
    assert [(row["id"]) for row in train] == [row["id"] for row in second[0]]
    assert [(row["id"]) for row in validation] == [row["id"] for row in second[1]]
    assert report == second[2]
    assert {row["speaker_id"] for row in train}.isdisjoint(row["speaker_id"] for row in validation)
    assert {row["audio_sha256"] for row in train}.isdisjoint(
        row["audio_sha256"] for row in validation
    )
    assert {row["text"] for row in train}.isdisjoint(row["text"] for row in validation)
    assert report["train_rows_removed_for_cross_split_text_duplicate"] == 1


def test_speaker_split_rejects_duplicate_ids_and_single_group():
    row = _row("same", "speaker-a", "A RED BIRD", "asset-a")
    try:
        split_by_speaker([row, dict(row)])
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate sample IDs must be rejected")
    try:
        split_by_speaker([row])
    except ValueError as exc:
        assert "two speaker" in str(exc)
    else:
        raise AssertionError("a single speaker cannot be split")


def test_choice_examples_are_paired_deterministic_and_have_four_unique_options():
    texts = [
        "THE CAT SAT ON MAT",
        "THE CAT RAN ON MAT",
        "THE CAT HID ON MAT",
        "THE CAT MET ON MAT",
        "THE DOG SAT ON MAT",
        "ORANGE CLOUDS CROSS THE OCEAN",
        "BRIGHT WINDOWS FACE THE GARDEN",
        "ANCIENT CITIES REST BESIDE RIVERS",
        "YOUNG MUSICIANS PRACTICE BEFORE DAWN",
        "WINTER EVENINGS FEEL QUIET AND LONG",
        "THE CAT SAT ON MAT",
    ]
    rows = [_row(f"id-{i}", f"speaker-{i}", text, f"asset-{i}") for i, text in enumerate(texts)]
    records = [_record(row) for row in rows]
    first, counts = build_choice_examples(rows, records, seed=17)
    second, _ = build_choice_examples(rows, records, seed=17)
    assert first == second
    assert counts["examples_with_paired_conditions"] > 0
    assert counts["excluded_duplicate_normalized_content_targets"] == 1
    assert counts["max_phoneme_similarity_candidates_scored"] == 256
    record_map = {record.sample_id: record for record in records}
    for example in first:
        assert len(example["options_by_condition"]["random_distractors"]) == 4
        assert len(example["options_by_condition"]["hard_distractors"]) == 4
        for condition_options in example["options_by_condition"].values():
            assert len(set(condition_options)) == 4
            assert condition_options[example["target_index"]] == example["target_text"]
        for condition_ids in example["distractor_ids_by_condition"].values():
            assert len(condition_ids) == 3
            assert example["sample_id"] not in condition_ids
            assert all(
                record_map[sample_id].speaker_id != example["speaker_id"]
                and record_map[sample_id].chapter_id != example["chapter_id"]
                for sample_id in condition_ids
            )
