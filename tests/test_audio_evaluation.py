from __future__ import annotations

from tiny_omni_decision.audio_evaluation import (
    build_hard_negative_example,
    hard_negative_score,
    transcript_similarity,
)


def test_transcript_similarity_tracks_phonetics_lexical_overlap_and_length() -> None:
    exact = transcript_similarity("the quick brown fox", "the quick brown fox")
    near = transcript_similarity("the quick brown fox", "the quick brown box")
    far = transcript_similarity("the quick brown fox", "we shall arrive tomorrow")

    assert exact == {"phonetic_soundex": 1.0, "lexical_jaccard": 1.0, "length_ratio": 1.0}
    assert near["phonetic_soundex"] > far["phonetic_soundex"]
    assert near["lexical_jaccard"] > far["lexical_jaccard"]
    assert hard_negative_score("the quick brown fox", "the quick brown box")[0] > 0


def test_build_hard_negative_example_is_deterministic_and_preserves_target_audio() -> None:
    example = {
        "id": "openslr/LibriSpeech:1-2-0000",
        "modality": "audio",
        "state": "spoken excerpt",
        "question": "Which sentence matches the spoken audio?",
        "options": ["THE QUICK BROWN FOX", "SHE WENT HOME", "WE SAW A SHIP", "THE SKY WAS BLUE"],
        "target": "THE QUICK BROWN FOX",
        "media": [{"kind": "audio", "path": "raw/a.flac", "frame_refs": []}],
        "source": "openslr/LibriSpeech",
        "source_revision": "a" * 40,
        "source_record_id": "1-2-0000",
        "split": "validation",
        "task_group_id": "speaker:1",
        "source_target": "THE QUICK BROWN FOX",
        "provenance": {"license": "CC-BY-4.0", "trust_status": "trusted"},
    }
    bank = [
        {"source_record_id": "2-1-0000", "target": "THE QUICK BROWN BOX"},
        {"source_record_id": "2-1-0001", "target": "THE QUICK BROWN FOG"},
        {"source_record_id": "2-1-0002", "target": "A QUICK BROWN FOX"},
        {"source_record_id": "2-1-0003", "target": "THE SHIP IS FAR AWAY"},
    ]

    first, first_audit = build_hard_negative_example(example, bank)
    second, second_audit = build_hard_negative_example(example, bank)

    assert first == second
    assert first_audit == second_audit
    assert first["id"].endswith(":hard-neg-v1")
    assert first["media"] == example["media"]
    assert first["target"] == example["target"]
    assert len(first["options"]) == 4
    assert len(set(first["options"])) == 4
    assert first["target"] in first["options"]
    assert [item["source_record_id"] for item in first_audit] == [
        "2-1-0001",
        "2-1-0000",
        "2-1-0002",
    ]
