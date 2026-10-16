import copy

import pytest

from scripts.native_language_cache import inventory


def fixture():
    from scripts.native_training_readiness import TEXT

    files = {}
    for split, names in TEXT.items():
        for name in names:
            files[name] = [
                dict(
                    id=name,
                    split=split,
                    state=f"state {name}",
                    group=name,
                    source="original",
                    question="Choose.",
                    options=["a", "b"],
                    target="b",
                    task="choice",
                )
            ]
    raw = inventory(files)
    language = dict(
        rows=raw["rows"],
        queries={
            k: dict(file="queries.npy", offset=i, tokens=4) for i, k in enumerate(raw["pairs"])
        },
        states={},
    )
    for split, names in TEXT.items():
        for i, name in enumerate(names):
            key = raw["rows"][name][0]["state"]
            language["states"][key] = dict(
                file=split + "-states.npy", offset=i * 4, tokens=4, split=split
            )
    images, image_records, lombard, starss, videos, audio_rows, video_rows = (
        {},
        {},
        {},
        {},
        {},
        [],
        [],
    )
    for split in TEXT:
        images[split] = [
            dict(
                id="image-" + split,
                split=split,
                color="red",
                texture="rubber",
                count=2,
                shape="cube",
            )
        ]
        image_records["val" if split == "validation" else split] = [
            dict(
                id="image-" + split,
                split="val" if split == "validation" else split,
                sha256="image-sha-" + split,
            )
        ]
        from scripts.lombard_grid_corpus import decode_code

        lombard[split] = [
            dict(
                source_id="speech-" + split,
                split=split,
                speaker_group=split,
                actual_code="bbab2n" if split == "train" else "bbac3n",
                targets=decode_code("bbab2n" if split == "train" else "bbac3n"),
            )
        ]
        starss[split] = [
            dict(
                id="window-" + split,
                split=split,
                room="room-" + split,
                recording="recording-" + split,
                queries=[
                    dict(
                        event="speech",
                        question="Is speech annotated?",
                        options=["no", "yes"],
                        target=1,
                    )
                ],
            )
        ]
        for source, ident, group in [
            ("lombard", "speech-" + split, split),
            ("starss", "window-" + split, "room-" + split),
        ]:
            audio_rows.append(
                dict(
                    id=source + ":" + ident,
                    source=source,
                    split=split,
                    group=group,
                    media_sha256=source + split,
                    speech_offset=0 if source == "lombard" else 4,
                    speech_tokens=4,
                    event_offset=0 if source == "lombard" else 1,
                )
            )
        videos[split] = [
            dict(
                id="video-q-" + split,
                split=split,
                scene_group_id="scene-" + split,
                media_sha256="video-sha-" + split,
                question_type="temporal_descriptive",
                question="Before or after?",
                options=["before", "after"],
                target="after",
            )
        ]
        video_rows.append(
            dict(
                media_sha256="video-sha-" + split,
                split=split,
                file=split + ".npy",
                shape=[2048, 768],
            )
        )
    return dict(
        text_files=files,
        language=language,
        images=images,
        image_records=image_records,
        lombard=lombard,
        starss=starss,
        audio=dict(rows=audio_rows),
        videos=videos,
        video_cache=dict(rows=video_rows),
    )


def test_complete_deterministic_join_and_targets():
    from scripts.native_common_corpus import build_corpus

    data = fixture()
    result = build_corpus(**data)
    assert result == build_corpus(**copy.deepcopy(data))
    assert {s: len(r) for s, r in result["rows"].items()} == {"train": 15, "validation": 15}
    assert len(result["observations"]) == 14
    for split, rows in result["rows"].items():
        for row in rows:
            assert result["observations"][row["observation"]]["split"] == split
            assert row["queries"] == [
                next(k for k, v in result["pairs"].items() if v == [row["question"], opt])
                for opt in row["options"]
            ]
            if row["modality"] == "image":
                assert len(row["options"]) == 40
            if row["source"] == "lombard" and row["task"] == "letter":
                assert len(row["options"]) == 25
            if row["modality"] == "text":
                assert row["target_index"] == 1


@pytest.mark.parametrize(
    "case",
    [
        "text_target",
        "text_join",
        "image_id",
        "image_split",
        "image_overlap",
        "audio_group",
        "audio_missing",
        "audio_extra",
        "speech_target",
        "video_scene",
        "video_hash",
        "duplicate_question",
        "event_target",
    ],
)
def test_bad_bindings_fail_closed(case):
    from scripts.native_common_corpus import build_corpus

    d = fixture()
    if case == "text_target":
        d["text_files"]["binding_train"][0]["target"] = "c"
    elif case == "text_join":
        d["language"]["rows"]["binding_train"][0]["queries"] = []
    elif case == "image_id":
        d["image_records"]["train"][0]["id"] = "wrong"
    elif case == "image_split":
        d["images"]["train"][0]["split"] = "validation"
    elif case == "image_overlap":
        d["image_records"]["val"][0]["sha256"] = "image-sha-train"
    elif case == "audio_group":
        d["lombard"]["validation"][0]["speaker_group"] = "train"
    elif case == "audio_missing":
        d["audio"]["rows"].pop()
    elif case == "audio_extra":
        d["audio"]["rows"].append(dict(d["audio"]["rows"][0], id="extra"))
    elif case == "speech_target":
        d["lombard"]["train"][0]["targets"]["color"] = "green"
    elif case == "video_scene":
        d["videos"]["validation"][0]["scene_group_id"] = "scene-train"
    elif case == "video_hash":
        d["videos"]["train"][0]["media_sha256"] = "wrong"
    elif case == "duplicate_question":
        d["videos"]["train"].append(d["videos"]["train"][0])
    elif case == "event_target":
        d["starss"]["train"][0]["queries"][0]["target"] = 2
    with pytest.raises(ValueError):
        build_corpus(**d)
