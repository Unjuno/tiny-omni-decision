"""Deterministic supervision joins for complete, already admitted native banks."""

import hashlib
from collections import Counter

from scripts.condition_relevance_training import row_key
from scripts.lombard_grid_corpus import SLOTS, decode_code
from scripts.native_language_cache import inventory, pair_key
from scripts.native_training_readiness import TEXT, state_key
from scripts.semantic_state_diagnostic import FIELDS, QUESTIONS, options, oracle_state

SPEECH_QUESTIONS = {
    "command": "Which command word is spoken?",
    "color": "Which color word is spoken?",
    "preposition": "Which preposition is spoken?",
    "letter": "Which letter is spoken?",
    "number": "Which number is spoken?",
    "adverb": "Which adverb is spoken?",
}


def build_corpus(
    *, text_files, language, images, image_records, lombard, starss, audio, videos, video_cache
):
    """Features never contain labels, task names, option IDs or target indices."""
    for parts in (images, lombard, starss, videos):
        if set(parts) != set(TEXT) or any(not rows for rows in parts.values()):
            raise ValueError("complete nonempty train/validation source parts required")
    raw_text = inventory(text_files)
    if raw_text["rows"] != language["rows"] or set(raw_text["pairs"]) != set(language["queries"]):
        raise ValueError("language cache differs from original questions/candidates")
    if set(raw_text["states"]) != set(language["states"]):
        raise ValueError("complete language states required")
    observations, pairs, seen_questions, owners = {}, {}, set(), {}
    rows = {split: [] for split in TEXT}

    def bind(split, kind, value):
        if not isinstance(value, str) or not value:
            raise ValueError("nonempty underlying identity required")
        if owners.setdefault((kind, value), split) != split:
            raise ValueError("cross-split underlying " + kind)

    def observation(ident, split, source, identities, features):
        for kind, value in identities:
            bind(split, kind, value)
        value = dict(split=split, source=source, features=features)
        for f in features:
            if (
                type(f["offset"]) is not int
                or f["offset"] < 0
                or type(f["tokens"]) is not int
                or f["tokens"] < 1
            ):
                raise ValueError("nonnegative feature offset and positive token count required")
        if ident in observations and observations[ident] != value:
            raise ValueError("conflicting observation identity")
        observations[ident] = value

    def question(
        ident,
        split,
        source,
        modality,
        cohort,
        task,
        obs,
        group,
        query,
        opts,
        target,
        ids=None,
        original_id=None,
    ):
        if (
            ident in seen_questions
            or obs not in observations
            or observations[obs]["split"] != split
            or not isinstance(query, str)
            or not query.strip()
            or not 2 <= len(opts) <= 40
            or len(set(opts)) != len(opts)
            or any(not isinstance(v, str) or not v.strip() for v in opts)
            or target not in opts
        ):
            raise ValueError("unique aligned question and full valid target/candidate set required")
        ids = list(ids) if ids is not None else [f"option-{i}" for i in range(len(opts))]
        if len(ids) != len(opts) or len(set(ids)) != len(ids) or any(not i for i in ids):
            raise ValueError("complete unique option IDs required")
        keys = []
        for opt in opts:
            key = pair_key(query, opt)
            value = [query, opt]
            if key in pairs and pairs[key] != value:
                raise ValueError("candidate pair hash collision")
            pairs[key] = value
            keys.append(key)
        seen_questions.add(ident)
        rows[split].append(
            dict(
                id=ident,
                original_id=original_id or ident,
                split=split,
                source=source,
                modality=modality,
                cohort=cohort,
                task=task,
                observation=obs,
                group=group,
                question=query,
                options=list(opts),
                option_ids=ids,
                target_index=opts.index(target),
                queries=keys,
            )
        )

    for split, names in TEXT.items():
        for name in names:
            for row, mapping in zip(text_files[name], language["rows"][name], strict=True):
                key = mapping["state"]
                meta = language["states"][key]
                if meta["split"] != split or meta["file"] != split + "-states.npy":
                    raise ValueError("text state cache split differs")
                ident = "text:" + key
                observation(
                    ident,
                    split,
                    "text",
                    [("text_group", row["group"]), ("normalized_text", state_key(row["state"]))],
                    [
                        dict(
                            kind="text",
                            bank="language/" + meta["file"],
                            offset=meta["offset"],
                            tokens=meta["tokens"],
                        )
                    ],
                )
                question(
                    "text:" + row["id"],
                    split,
                    "text:" + row["source"],
                    "text",
                    name,
                    row_key(row),
                    ident,
                    row["group"],
                    row["question"],
                    row["options"],
                    row["target"],
                    row.get("option_ids"),
                    row["id"],
                )
    candidates = options()
    for split in TEXT:
        cache_split = "val" if split == "validation" else split
        if [r["id"] for r in images[split]] != [r["id"] for r in image_records[cache_split]]:
            raise ValueError("image label/cache identity order mismatch")
        for i, (row, cache) in enumerate(
            zip(images[split], image_records[cache_split], strict=True)
        ):
            if row["split"] != split or cache["split"] != cache_split:
                raise ValueError("image split differs")
            truth = oracle_state(row)
            ident = "image:" + row["id"]
            observation(
                ident,
                split,
                "clevr4",
                [("image_id", row["id"]), ("media_sha256", cache["sha256"])],
                [
                    dict(
                        kind="image",
                        bank=f"image/{cache_split}-cells.npy",
                        index=i,
                        offset=0,
                        tokens=17,
                    )
                ],
            )
            ordered = sorted(
                candidates,
                key=lambda c: hashlib.sha256(f"17:{row['id']}:{c['id']}".encode()).hexdigest(),
            )
            opts, ids = [c["description"] for c in ordered], [c["id"] for c in ordered]
            for field, query in zip(FIELDS, QUESTIONS, strict=True):
                question(
                    ident + ":" + field,
                    split,
                    "clevr4",
                    "image",
                    "clevr4",
                    field,
                    ident,
                    row["id"],
                    query,
                    opts,
                    f"{field}: {truth[field]}",
                    ids,
                )
    audio_map = {r["id"]: r for r in audio["rows"]}
    if len(audio_map) != len(audio["rows"]):
        raise ValueError("duplicate audio feature IDs")
    used_audio = set()
    for split in TEXT:
        for source, parts in (("lombard", lombard), ("starss", starss)):
            for row in parts[split]:
                ident = source + ":" + row["source_id" if source == "lombard" else "id"]
                cached = audio_map.get(ident)
                group = row["speaker_group" if source == "lombard" else "room"]
                if (
                    cached is None
                    or ident in used_audio
                    or row["split"] != split
                    or cached["split"] != split
                    or cached["source"] != source
                    or cached["group"] != group
                ):
                    raise ValueError("complete source/group/split-matched audio features required")
                used_audio.add(ident)
                observation(
                    ident,
                    split,
                    source,
                    [
                        ("audio_id", ident),
                        (source + "_group", group),
                        ("media_sha256", cached["media_sha256"]),
                    ],
                    [
                        dict(
                            kind="speech",
                            bank=f"audio/{split}-speech.npy",
                            offset=cached["speech_offset"],
                            tokens=cached["speech_tokens"],
                        ),
                        dict(
                            kind="event_audio",
                            bank=f"audio/{split}-event.npy",
                            offset=cached["event_offset"],
                            tokens=1,
                        ),
                    ],
                )
                if source == "lombard":
                    if row["targets"] != decode_code(row["actual_code"]):
                        raise ValueError("speech labels disagree with actual utterance")
                    bind(split, "speech_phrase", row["actual_code"])
                    for slot, table in SLOTS.items():
                        question(
                            ident + ":" + slot,
                            split,
                            source,
                            "audio",
                            source,
                            slot,
                            ident,
                            group,
                            SPEECH_QUESTIONS[slot],
                            list(table.values()),
                            row["targets"][slot],
                        )
                else:
                    bind(split, "starss_recording", row["recording"])
                    for query in row["queries"]:
                        if (
                            query["options"] != ["no", "yes"]
                            or type(query["target"]) is not int
                            or query["target"] not in (0, 1)
                        ):
                            raise ValueError("original event binary target required")
                        question(
                            ident + ":" + query["event"],
                            split,
                            source,
                            "audio",
                            source,
                            query["event"],
                            ident,
                            group,
                            query["question"],
                            query["options"],
                            query["options"][query["target"]],
                        )
    if used_audio != set(audio_map):
        raise ValueError("unused/missing audio features")
    video_map = {r["media_sha256"]: r for r in video_cache["rows"]}
    if len(video_map) != len(video_cache["rows"]):
        raise ValueError("duplicate video feature identities")
    used_video = set()
    for split in TEXT:
        for row in videos[split]:
            key = row["media_sha256"]
            cached = video_map.get(key)
            if (
                cached is None
                or row["split"] != split
                or cached["split"] != split
                or cached["shape"] != [2048, 768]
            ):
                raise ValueError("matching full native video features required")
            used_video.add(key)
            ident = "video:" + key
            observation(
                ident,
                split,
                "clevrer",
                [("video_scene", row["scene_group_id"]), ("media_sha256", key)],
                [dict(kind="video", bank="video/" + cached["file"], offset=0, tokens=2048)],
            )
            question(
                "video:" + row["id"],
                split,
                "clevrer",
                "video",
                "clevrer",
                row["question_type"],
                ident,
                row["scene_group_id"],
                row["question"],
                row["options"],
                row["target"],
                original_id=row["id"],
            )
    if used_video != set(video_map):
        raise ValueError("unused/missing video features")
    counts = {}
    for split, items in rows.items():
        counts[split] = dict(
            questions=len(items),
            observations=len({r["observation"] for r in items}),
            modality=dict(Counter(r["modality"] for r in items)),
            source=dict(Counter(r["source"] for r in items)),
            tasks=dict(Counter(r["source"] + ":" + r["task"] for r in items)),
            event_positive=dict(
                Counter(
                    r["task"] for r in items if r["source"] == "starss" and r["target_index"] == 1
                )
            ),
        )
    return dict(
        rows=rows, observations=observations, pairs=dict(sorted(pairs.items())), counts=counts
    )
