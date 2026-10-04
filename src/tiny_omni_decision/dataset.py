from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import random
import re
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .corpus import media_identity, source_asset_identity
from .schema import DatasetManifest, DecisionExample, LicenseProvenance, MediaRef

KNOWN_PERMISSIVE = {"CC0-1.0", "CC-BY-4.0", "MIT", "Apache-2.0", "BSD-3-Clause"}
CLEVR4_TAXONOMIES = {
    "color": [
        "gray",
        "red",
        "blue",
        "green",
        "brown",
        "purple",
        "cyan",
        "yellow",
        "pink",
        "orange",
    ],
    "texture": [
        "rubber",
        "metal",
        "checkered",
        "emojis",
        "wave",
        "brick",
        "star",
        "circles",
        "zigzag",
        "chessboard",
    ],
    "count": [str(value) for value in range(1, 11)],
    "shape": [
        "cube",
        "sphere",
        "monkey",
        "cone",
        "torus",
        "star",
        "teapot",
        "diamond",
        "gear",
        "cylinder",
    ],
}
SPEECH_COMMAND_WORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
SPEECH_COMMAND_LABELS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
    "tree",
    "wow",
    "backward",
    "forward",
    "follow",
    "learn",
    "visual",
    "_silence_",
]
CLEVRER_TAXONOMIES = {
    "exist": ["no", "yes"],
    "query_color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
    "query_material": ["rubber", "metal"],
    "query_shape": ["cube", "sphere", "cylinder"],
    "count": [str(value) for value in range(0, 6)],
}
CLEVRER_NATIVE_REASONING_TYPES = {"explanatory", "predictive", "counterfactual"}
CLEVRER_TEMPORAL_PROGRAM_TOKENS = {
    "after",
    "before",
    "end",
    "filter_collision",
    "filter_in",
    "filter_moving",
    "filter_out",
    "get_frame",
    "start",
}
CLEVRER_TEMPORAL_QUESTION_RE = re.compile(
    r"\b(first|last|before|after|enter(?:s|ed|ing)?|exit(?:s|ed|ing)?|"
    r"begin(?:s|ning)?|ends?|collision|collide|moving|moves?|move|frame)\b",
    re.IGNORECASE,
)


def _canonical_license(value: str) -> str:
    aliases = {
        "cc0-1.0": "CC0-1.0",
        "cc-by-4.0": "CC-BY-4.0",
        "mit": "MIT",
        "apache-2.0": "Apache-2.0",
        "bsd-3-clause": "BSD-3-Clause",
    }
    return aliases.get(value.strip().lower(), value.strip())


def audit_license(
    license_name: str,
    *,
    commercial_use: bool | None,
    derivative_model_training_allowed: bool | None,
    redistribution_allowed: bool | None,
    media_redistribution_allowed: bool | None,
    has_media: bool,
    trust_status: str,
) -> tuple[str, list[str]]:
    unresolved: list[str] = []
    values = {
        "commercial_use": commercial_use,
        "derivative_model_training_allowed": derivative_model_training_allowed,
        "redistribution_allowed": redistribution_allowed,
    }
    if has_media:
        values["media_redistribution_allowed"] = media_redistribution_allowed
    for field, value in values.items():
        if value is None:
            unresolved.append(field)
        elif value is False:
            return "DENY", unresolved
    canonical = _canonical_license(license_name)
    if not canonical or canonical.upper() in {"UNKNOWN", "NONE STATED", "PER-SOURCE"}:
        unresolved.append("license")
    elif canonical not in KNOWN_PERMISSIVE:
        unresolved.append("license_policy_review")
    if trust_status != "trusted":
        unresolved.append("trust_status")
    if unresolved:
        return "REVIEW", sorted(set(unresolved))
    return "ALLOW", []


def audit_manifest(manifest: DatasetManifest) -> dict[str, Any]:
    decision, unresolved = audit_license(
        manifest.license,
        commercial_use=manifest.commercial_use,
        derivative_model_training_allowed=manifest.derivative_model_training_allowed,
        redistribution_allowed=manifest.redistribution_allowed,
        media_redistribution_allowed=manifest.media_redistribution_allowed,
        has_media=any(item in {"image", "audio", "video"} for item in manifest.modalities),
        trust_status=manifest.trust_status,
    )
    component_decisions = []
    for component in manifest.source_components:
        result, missing = audit_license(
            component.license,
            commercial_use=component.commercial_use,
            derivative_model_training_allowed=component.derivative_model_training_allowed,
            redistribution_allowed=component.redistribution_allowed,
            media_redistribution_allowed=component.media_redistribution_allowed,
            has_media=any(
                item in {"image", "audio", "video"}
                for item in (component.modalities or manifest.modalities)
            ),
            trust_status=component.trust_status,
        )
        component_decisions.append(
            {"component_id": component.component_id, "policy": result, "unresolved": missing}
        )
    if component_decisions and any(item["policy"] == "DENY" for item in component_decisions):
        decision = "DENY"
    elif component_decisions and any(item["policy"] != "ALLOW" for item in component_decisions):
        decision = "REVIEW" if decision != "DENY" else decision
    return {
        "dataset_id": manifest.dataset_id,
        "revision": manifest.revision,
        "immutable_revision": bool(re.fullmatch(r"[0-9a-f]{40}", manifest.revision)),
        "license": manifest.license,
        "commercial_use": manifest.commercial_use,
        "derivative_model_training_allowed": manifest.derivative_model_training_allowed,
        "media_redistribution_allowed": manifest.media_redistribution_allowed,
        "unresolved": unresolved,
        "source_components": component_decisions,
        "project_policy": decision,
        "intended_use": manifest.usage,
    }


def filter_mixed_license_rows(
    rows: Iterable[dict[str, Any]],
    components: dict[str, dict[str, Any]],
    *,
    has_media: bool | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep only rows whose pinned source component has fully affirmative policy facts."""
    counts = {"ALLOW": 0, "REVIEW": 0, "DENY": 0, "unresolved_source": 0}
    accepted = list(iter_mixed_license_rows(rows, components, counts=counts, has_media=has_media))
    return accepted, counts


def iter_mixed_license_rows(
    rows: Iterable[dict[str, Any]],
    components: dict[str, dict[str, Any]],
    *,
    counts: dict[str, int],
    has_media: bool | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream only explicitly ALLOWed rows from a mixed-source dataset."""
    for row in rows:
        source = str(row.get("source", ""))
        component = components.get(source)
        if component is None:
            counts["unresolved_source"] += 1
            counts["REVIEW"] += 1
            continue
        if _canonical_license(str(row.get("license", "UNKNOWN"))) != _canonical_license(
            str(component.get("license", "UNKNOWN"))
        ):
            counts["REVIEW"] += 1
            continue
        row_has_media = has_media if has_media is not None else _row_has_media(row)
        policy, _ = audit_license(
            str(component.get("license", "UNKNOWN")),
            commercial_use=component.get("commercial_use"),
            derivative_model_training_allowed=component.get("derivative_model_training_allowed"),
            redistribution_allowed=component.get("redistribution_allowed"),
            media_redistribution_allowed=component.get("media_redistribution_allowed"),
            has_media=row_has_media,
            trust_status=str(component.get("trust_status", "review")),
        )
        counts[policy] += 1
        if policy == "ALLOW":
            yield row


def _row_has_media(row: dict[str, Any]) -> bool:
    """Detect common embedded/reference fields before applying media rights gates."""
    media_fields = (
        "media",
        "images",
        "image",
        "audio",
        "audio_id",
        "audio_path",
        "video",
        "video_id",
        "video_path",
        "frames",
    )
    return any(row.get(field) not in (None, "", [], {}) for field in media_fields)


def shuffle_options(example: DecisionExample, seed: int) -> DecisionExample:
    indices = list(range(len(example.options)))
    random.Random(f"{seed}:{example.id}").shuffle(indices)
    options = [example.options[index] for index in indices]
    return example.model_copy(update={"options": options, "target": example.target})


def _soft(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _single_target(options: list[str], target: Any) -> str:
    """Convert an explicit single-winner annotation; refuse soft/multilabel targets."""
    target = _soft(target)
    if isinstance(target, int) and not isinstance(target, bool):
        target = str(target)
    if isinstance(target, bool):
        label = "true" if target else "false"
        if label in options:
            return label
        raise ValueError("boolean target needs true/false options")
    if isinstance(target, dict):
        scores = [float(target.get(option, 0.0)) for option in options]
    elif isinstance(target, list):
        scores = [float(value) for value in target]
    elif isinstance(target, str):
        if target in options:
            return target
        if len(target) == 1 and target.upper() in {chr(65 + i) for i in range(len(options))}:
            return options[ord(target.upper()) - 65]
        raise ValueError("answer is not one of the supplied options")
    else:
        raise ValueError("source target is not a categorical answer")
    if len(scores) != len(options) or sum(score == 1.0 for score in scores) != 1:
        raise ValueError("soft or malformed target cannot be normalized as a single-choice label")
    winner = scores.index(1.0)
    if any(value != 0.0 for index, value in enumerate(scores) if index != winner):
        raise ValueError("multi-label target cannot be represented by a single-choice example")
    return options[winner]


def _media_refs(
    raw_media: Any, revision: str, dataset_id: str, license_name: str | None = None
) -> list[MediaRef]:
    raw_media = _soft(raw_media)
    if raw_media is None:
        return []
    if isinstance(raw_media, str):
        raw_media = _soft(raw_media)
    if not isinstance(raw_media, list):
        raw_media = [raw_media]
    refs: list[MediaRef] = []
    for item in raw_media:
        if not isinstance(item, dict):
            continue
        kind = item.get("type") or item.get("kind")
        if kind not in {"image", "audio", "video"}:
            continue
        paths = item.get("frames") if kind == "video" else None
        source_path = item.get("path") or item.get("url") or item.get("src")
        if paths:
            refs.append(
                MediaRef(
                    kind="video",
                    uri=f"hf-dataset://{dataset_id}@{revision}/record/{item.get('id', 'video')}",
                    frame_refs=[f"hf-dataset://{dataset_id}@{revision}/{path}" for path in paths],
                    license=license_name,
                )
            )
        elif source_path:
            if str(source_path).startswith(("http://", "https://")):
                refs.append(MediaRef(kind=kind, uri=str(source_path), license=license_name))
            else:
                refs.append(
                    MediaRef(
                        kind=kind,
                        uri=f"hf-dataset://{dataset_id}@{revision}/{source_path}",
                        license=license_name,
                    )
                )
        elif item.get("bytes") is not None:
            raise ValueError(
                "binary media cannot be embedded; preserve an upstream media reference"
            )
    return refs


def _base_example(
    *,
    row: dict[str, Any],
    dataset_id: str,
    revision: str,
    options: list[str],
    target: Any,
    state: Any,
    question: Any,
    split: str,
    media: list[MediaRef],
    license_name: str,
    commercial_use: bool | None,
    derivative_model_training_allowed: bool | None,
    redistribution_allowed: bool | None,
    media_redistribution_allowed: bool | None,
    attribution: str | None = None,
    source_component: str | None = None,
    source_target: Any = None,
    task_type: str | None = None,
    task_group_id: str | None = None,
    trust_status: str = "review",
) -> DecisionExample:
    source_record_id = (
        row.get("id")
        or row.get("source_record_id")
        or row.get("audio_id")
        or row.get("video_id")
        or row.get("video")
    )
    if not source_record_id:
        source_record_id = hashlib.sha256(
            json.dumps(row, sort_keys=True, default=str, ensure_ascii=False).encode()
        ).hexdigest()
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, sort_keys=True)
    if not isinstance(question, str):
        question = json.dumps(question, ensure_ascii=False, sort_keys=True)
    target_label = _single_target(options, target)
    modality = next(
        (kind for kind in ("video", "audio", "image") if any(item.kind == kind for item in media)),
        "text",
    )
    return DecisionExample(
        id=f"{dataset_id}:{source_record_id}",
        modality=modality,
        state=state,
        question=question,
        options=options,
        target=target_label,
        source=dataset_id,
        source_revision=revision,
        source_record_id=str(source_record_id),
        split=split,
        task_type=task_type,
        task_group_id=task_group_id,
        media=media,
        source_target=target if source_target is None else source_target,
        provenance=LicenseProvenance(
            license=license_name,
            commercial_use=commercial_use,
            derivative_model_training_allowed=derivative_model_training_allowed,
            redistribution_allowed=redistribution_allowed,
            media_redistribution_allowed=media_redistribution_allowed,
            attribution=attribution,
            source_component=source_component,
            trust_status=trust_status,
        ),
    )


def _flatten_typed_synth(row: dict[str, Any], manifest: DatasetManifest) -> list[DecisionExample]:
    questions = _soft(row.get("questions"))
    gold = _soft(row.get("gold"))
    state = _soft(row.get("state")) if row.get("state_is_json") else row.get("state")
    if not isinstance(questions, dict) or not isinstance(gold, dict):
        raise ValueError("typed-decisions-synth row needs question and gold objects")
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, sort_keys=True)
    teacher = _soft(row.get("teacher")) or {}
    result = []
    for question_id, definition in questions.items():
        kind = definition.get("type")
        criteria = definition.get("criteria")
        answer = gold.get(question_id)
        if kind == "noul":
            options = ["true", "false"]
            target = "true" if answer is True else "false" if answer is False else answer
            question = str(definition.get("instructions", question_id))
        elif kind == "score":
            options = list(criteria or [])
            target = (
                options[int(answer)]
                if isinstance(answer, int) and 0 <= answer < len(options)
                else answer
            )
            question = str(definition.get("instructions", question_id))
        elif kind == "choice":
            local_teacher = _soft(teacher.get(question_id, {}))
            label_keys = list((local_teacher.get("probabilities") or {}).keys())
            if isinstance(criteria, dict):
                options = [f"{key}: {criteria[key]}" for key in label_keys]
                target_key = str(answer)
                target = next(
                    (
                        option
                        for key, option in zip(label_keys, options, strict=True)
                        if key == target_key
                    ),
                    answer,
                )
            elif isinstance(criteria, list):
                options = list(criteria)
                target = (
                    options[int(answer)]
                    if isinstance(answer, int) and 0 <= answer < len(options)
                    else answer
                )
            else:
                raise ValueError(f"choice {question_id} has no declared answer options")
            question = str(definition.get("instructions", question_id))
        else:
            raise ValueError(f"unsupported typed-decisions-synth question kind: {kind}")
        if len(options) < 2 or len(options) > 20:
            raise ValueError(
                f"question {question_id} has {len(options)} options; supported range is 2–20"
            )
        result.append(
            _base_example(
                row={"id": f"{row['state_id']}:{question_id}"},
                dataset_id=manifest.dataset_id,
                revision=manifest.revision,
                split=str(row.get("split", manifest.split)),
                options=options,
                target=target,
                state=state,
                question=question,
                media=[],
                license_name="MIT",
                commercial_use=True,
                derivative_model_training_allowed=True,
                redistribution_allowed=True,
                media_redistribution_allowed=None,
                attribution="Muhammed Nazeem; synthetic data generated with DeepSeek V4.1 Flash.",
                source_component="n4ze3m/typed-decisions-synth",
                source_target={"gold": answer, "teacher": teacher.get(question_id)},
                trust_status=manifest.trust_status,
            )
        )
    return result


def _flatten_clevr4(row: dict[str, Any], manifest: DatasetManifest) -> list[DecisionExample]:
    """Turn one pinned Clevr-4 image annotation into four categorical decisions."""
    source_record_id = str(row.get("id", ""))
    if not source_record_id:
        raise ValueError("Clevr-4 annotation needs its original image stem as id")
    archive_sha512 = str(manifest.notes.get("archive_sha512", ""))
    if not re.fullmatch(r"[0-9a-f]{128}", archive_sha512):
        raise ValueError("Clevr-4 manifest must pin the source image archive SHA-512")
    examples = []
    for taxonomy, vocabulary in CLEVR4_TAXONOMIES.items():
        if taxonomy not in row:
            raise ValueError(f"Clevr-4 row is missing {taxonomy!r} label")
        options = [str(value) for value in vocabulary]
        target = str(row[taxonomy])
        media = [
            MediaRef(
                kind="image",
                uri=(
                    f"source-ref://sgvaze/clevr4@sha512-{archive_sha512}/"
                    f"images/{source_record_id}.png"
                ),
                license=manifest.license,
            )
        ]
        example = _base_example(
            row={"id": source_record_id},
            dataset_id=manifest.dataset_id,
            revision=manifest.revision,
            split=str(row.get("split", manifest.split)),
            options=options,
            target=str(target),
            state="Synthetic CLEVR-4 image with annotated object attributes.",
            question=f"What is the image's {taxonomy} class?",
            media=media,
            license_name=manifest.license,
            commercial_use=manifest.commercial_use,
            derivative_model_training_allowed=manifest.derivative_model_training_allowed,
            redistribution_allowed=manifest.redistribution_allowed,
            media_redistribution_allowed=manifest.media_redistribution_allowed,
            attribution=manifest.attribution,
            source_component="sgvaze/clevr4",
            source_target=row[taxonomy],
            trust_status=manifest.trust_status,
        )
        examples.append(example.model_copy(update={"id": f"{example.id}:{taxonomy}"}))
    return examples


def _speech_command_label(row: dict[str, Any]) -> str | None:
    label = row.get("label")
    if row.get("is_unknown") is True:
        return None
    if isinstance(label, int) and not isinstance(label, bool):
        if label < 0 or label >= len(SPEECH_COMMAND_LABELS):
            raise ValueError(f"Speech Commands label index is out of range: {label}")
        label = SPEECH_COMMAND_LABELS[label]
    if not isinstance(label, str):
        raise ValueError("Speech Commands row needs a string or integer label")
    return label if label in SPEECH_COMMAND_WORDS else None


def _adapt_speech_command(row: dict[str, Any], manifest: DatasetManifest) -> DecisionExample:
    target = _speech_command_label(row)
    if target is None:
        raise ValueError("Speech Commands row is not one of the ten command words")
    source_id = str(row.get("id") or row.get("file") or "")
    audio = row.get("audio")
    audio = audio if isinstance(audio, dict) else {}
    audio_path = str(audio.get("path") or row.get("path") or "").replace("\\", "/")
    if not source_id:
        source_id = audio_path
    if not source_id:
        raise ValueError("Speech Commands row needs an audio path or stable id")
    if audio_path.startswith("/") or ".." in audio_path.split("/"):
        raise ValueError("Speech Commands audio path must be relative to the pinned archive")
    archive_path = audio_path or f"records/{urllib.parse.quote(source_id, safe='')}"
    media = [
        MediaRef(
            kind="audio",
            uri=(
                f"source-ref://google/speech_commands@{manifest.revision}/"
                f"v0.02/{manifest.split}/{archive_path}"
            ),
            license=manifest.license,
        )
    ]
    return _base_example(
        row={"id": source_id},
        dataset_id=manifest.dataset_id,
        revision=manifest.revision,
        split=str(row.get("split", manifest.split)),
        options=SPEECH_COMMAND_WORDS,
        target=target,
        state="One-second English spoken-keyword recording.",
        question="Which of the ten spoken English keywords was uttered?",
        media=media,
        license_name=manifest.license,
        commercial_use=manifest.commercial_use,
        derivative_model_training_allowed=manifest.derivative_model_training_allowed,
        redistribution_allowed=manifest.redistribution_allowed,
        media_redistribution_allowed=manifest.media_redistribution_allowed,
        attribution=manifest.attribution,
        source_component="google/speech_commands",
        source_target=target,
        trust_status=manifest.trust_status,
    )


def _clevrer_descriptive_task_type(question: dict[str, Any]) -> str:
    """Classify temporal descriptive questions from program semantics, then text fallback."""
    program = question.get("program")
    if isinstance(program, list) and program:
        temporal = bool(CLEVRER_TEMPORAL_PROGRAM_TOKENS.intersection(map(str, program)))
    else:
        text = str(question.get("question") or "")
        temporal = bool(CLEVRER_TEMPORAL_QUESTION_RE.search(text))
    return "temporal_descriptive" if temporal else "static_descriptive"


def _flatten_clevrer(
    row: dict[str, Any], manifest: DatasetManifest, *, video_native: bool = False
) -> list[DecisionExample]:
    """Normalize legacy descriptive or video-native CLEVRER questions for one video."""
    video_name = str(row.get("video_filename") or "")
    if not video_name or Path(video_name).name != video_name or not video_name.endswith(".mp4"):
        raise ValueError("CLEVRER question row needs a plain video_filename")
    scene_index = row.get("scene_index")
    if not isinstance(scene_index, int):
        raise ValueError("CLEVRER question row needs an integer scene_index")
    expected_split = manifest.split
    if expected_split == "train" and not 0 <= scene_index < 10_000:
        raise ValueError("CLEVRER scene_index is outside the official training range")
    if expected_split == "validation" and not 10_000 <= scene_index < 15_000:
        raise ValueError("CLEVRER scene_index is outside the official validation range")
    questions = row.get("questions")
    if not isinstance(questions, list):
        raise ValueError("CLEVRER video row needs a questions list")
    split_dir = "validation" if expected_split == "validation" else "train"
    media = [
        MediaRef(
            kind="video",
            uri=f"source-ref://CLEVRER/{manifest.revision}/videos/{split_dir}/{video_name}",
            license=manifest.license,
        )
    ]
    result = []
    for question in questions:
        question_type = question.get("question_type")
        question_id = question.get("question_id")
        if not isinstance(question_id, (str, int)) or isinstance(question_id, bool):
            raise ValueError("CLEVRER question needs a stable question_id")
        if question_type == "descriptive":
            taxonomy = question.get("question_subtype")
            options = CLEVRER_TAXONOMIES.get(taxonomy)
            answer = question.get("answer")
            if options is None or answer is None:
                continue
            task_type = _clevrer_descriptive_task_type(question) if video_native else None
            source_id = f"{scene_index}:{question_id}"
            result.append(
                _base_example(
                    row={"id": source_id},
                    dataset_id=manifest.dataset_id,
                    revision=manifest.revision,
                    split=expected_split,
                    options=options,
                    target=str(answer),
                    state="Synthetic CLEVRER video of moving and colliding objects.",
                    question=str(question.get("question") or ""),
                    media=media,
                    license_name=manifest.license,
                    commercial_use=manifest.commercial_use,
                    derivative_model_training_allowed=manifest.derivative_model_training_allowed,
                    redistribution_allowed=manifest.redistribution_allowed,
                    media_redistribution_allowed=manifest.media_redistribution_allowed,
                    attribution=manifest.attribution,
                    source_component="MIT-IBM-CLEVRER",
                    source_target=answer,
                    task_type=task_type,
                    task_group_id=(f"{scene_index}:{question_id}" if video_native else None),
                    trust_status=manifest.trust_status,
                )
            )
            continue
        if not video_native or question_type not in CLEVRER_NATIVE_REASONING_TYPES:
            continue
        choices = question.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"CLEVRER {question_type} question needs a non-empty choices list")
        seen_choice_ids: set[str] = set()
        for choice in choices:
            choice_id = choice.get("choice_id")
            if not isinstance(choice_id, (str, int)) or isinstance(choice_id, bool):
                raise ValueError("CLEVRER choice needs a stable choice_id")
            choice_id_text = str(choice_id)
            if not choice_id_text or choice_id_text in seen_choice_ids:
                raise ValueError("CLEVRER choices need unique choice_id values per question")
            seen_choice_ids.add(choice_id_text)
            candidate = str(choice.get("choice") or "").strip()
            target = choice.get("answer")
            if not candidate or target not in {"wrong", "correct"}:
                raise ValueError("CLEVRER choice needs text and a wrong/correct answer")
            source_id = f"{scene_index}:{question_id}:{choice_id_text}"
            result.append(
                _base_example(
                    row={"id": source_id},
                    dataset_id=manifest.dataset_id,
                    revision=manifest.revision,
                    split=expected_split,
                    options=["wrong", "correct"],
                    target=str(target),
                    state="Synthetic CLEVRER video of moving and colliding objects.",
                    question=(
                        f"{str(question.get('question') or '').strip()}\n"
                        f"Candidate statement: {candidate}"
                    ),
                    media=media,
                    license_name=manifest.license,
                    commercial_use=manifest.commercial_use,
                    derivative_model_training_allowed=manifest.derivative_model_training_allowed,
                    redistribution_allowed=manifest.redistribution_allowed,
                    media_redistribution_allowed=manifest.media_redistribution_allowed,
                    attribution=manifest.attribution,
                    source_component="MIT-IBM-CLEVRER",
                    source_target=target,
                    task_type=question_type,
                    task_group_id=f"{scene_index}:{question_id}",
                    trust_status=manifest.trust_status,
                )
            )
    return result


def _clevrer_row_matches_split(row: dict[str, Any], split: str) -> bool:
    scene_index = row.get("scene_index")
    if not isinstance(scene_index, int):
        raise ValueError("CLEVRER question row needs an integer scene_index")
    if split == "train":
        return 0 <= scene_index < 10_000
    if split == "validation":
        return 10_000 <= scene_index < 15_000
    raise ValueError(f"Unsupported CLEVRER split: {split}")


def adapt_row(
    row: dict[str, Any],
    manifest: DatasetManifest,
    adapter: str,
    components: dict[str, dict[str, Any]] | None = None,
) -> DecisionExample:
    components = components or {}
    dataset_id = manifest.dataset_id
    common = {
        "row": row,
        "dataset_id": dataset_id,
        "revision": manifest.revision,
        "split": str(row.get("split", manifest.split)),
    }
    if adapter in {"open-jev", "openjev"}:
        state = _soft(row.get("state_json", row.get("state")))
        return _base_example(
            **common,
            options=list(row["options"]),
            target=row["target"],
            state=state,
            question=row["question"],
            media=[],
            license_name="CC0-1.0",
            commercial_use=True,
            derivative_model_training_allowed=True,
            redistribution_allowed=True,
            media_redistribution_allowed=None,
            attribution="Open-Jev dataset authors; see source citation.",
            trust_status=manifest.trust_status,
        )
    if adapter in {"onejev", "onejev-data"}:
        source = str(row.get("source", ""))
        component = components.get(source)
        license_name = str(row.get("license", "UNKNOWN"))
        if component and _canonical_license(
            str(component.get("license", "UNKNOWN"))
        ) != _canonical_license(license_name):
            raise ValueError(f"row license does not match pinned source component for {source}")
        if component is None:
            component = {"license": license_name}
        answer = _soft(row.get("target"))
        options = list(answer) if isinstance(answer, dict) else row.get("options")
        if not options:
            raise ValueError("OneJev row has no categorical target keys/options")
        media = _media_refs(row.get("media"), manifest.revision, dataset_id, license_name)
        for image in row.get("images", []) or []:
            if isinstance(image, str):
                media.append(
                    MediaRef(
                        kind="image",
                        uri=f"hf-dataset://{dataset_id}@{manifest.revision}/{image}",
                        license=license_name,
                    )
                )
            elif isinstance(image, dict) and image.get("path"):
                media.append(
                    MediaRef(
                        kind="image",
                        uri=f"hf-dataset://{dataset_id}@{manifest.revision}/{image['path']}",
                        license=license_name,
                    )
                )
            elif isinstance(image, dict) and image.get("bytes") is not None:
                media.append(
                    MediaRef(
                        kind="image",
                        uri=f"hf-dataset://{dataset_id}@{manifest.revision}/record/{row.get('id')}/image/{len(media)}",
                        license=license_name,
                    )
                )
        return _base_example(
            **common,
            options=list(options),
            target=answer,
            state=row.get("state", ""),
            question=row.get("question", ""),
            media=media,
            license_name=license_name,
            commercial_use=component.get("commercial_use"),
            derivative_model_training_allowed=component.get("derivative_model_training_allowed"),
            redistribution_allowed=component.get("redistribution_allowed"),
            media_redistribution_allowed=component.get("media_redistribution_allowed"),
            attribution=component.get("attribution"),
            source_component=source,
            trust_status=component.get("trust_status", manifest.trust_status),
        )
    if adapter in {"mmau", "mmau-test-mini"}:
        options = list(row.get("choices", row.get("options", [])))
        media = [
            MediaRef(
                kind="audio",
                uri=f"hf-dataset://{dataset_id}@{manifest.revision}/{row['audio_id']}",
                license=manifest.license,
            )
        ]
        return _base_example(
            **common,
            options=options,
            target=row.get("answer", row.get("target")),
            state="Audio clip supplied by the pinned source.",
            question=row["question"],
            media=media,
            license_name=manifest.license,
            commercial_use=manifest.commercial_use,
            derivative_model_training_allowed=manifest.derivative_model_training_allowed,
            redistribution_allowed=manifest.redistribution_allowed,
            media_redistribution_allowed=manifest.media_redistribution_allowed,
            attribution=manifest.attribution,
            trust_status=manifest.trust_status,
        )
    if adapter in {"mvbench", "video"}:
        options = row.get("candidates", row.get("options"))
        if not options:
            raise ValueError("MVBench row is missing candidates")
        answer = row.get("answer", row.get("target"))
        media_path = row.get("video", row.get("video_path", row.get("video_id")))
        if not media_path:
            raise ValueError("MVBench row is missing video source identity")
        media = [
            MediaRef(
                kind="video",
                uri=f"source-ref://{dataset_id}@{manifest.revision}/{media_path}",
                license=manifest.license,
            )
        ]
        return _base_example(
            **common,
            options=list(options),
            target=answer,
            state=str(row.get("subtitle", "Video clip supplied by its original source.")),
            question=row["question"],
            media=media,
            license_name=manifest.license,
            commercial_use=manifest.commercial_use,
            derivative_model_training_allowed=manifest.derivative_model_training_allowed,
            redistribution_allowed=manifest.redistribution_allowed,
            media_redistribution_allowed=manifest.media_redistribution_allowed,
            attribution=manifest.attribution,
            trust_status=manifest.trust_status,
        )
    if adapter == "generic":
        media = [MediaRef.model_validate(item) for item in row.get("media", [])]
        return _base_example(
            **common,
            options=list(row["options"]),
            target=row["target"],
            state=row["state"],
            question=row["question"],
            media=media,
            license_name=str(row.get("license", manifest.license)),
            commercial_use=row.get("commercial_use", manifest.commercial_use),
            derivative_model_training_allowed=row.get(
                "derivative_model_training_allowed", manifest.derivative_model_training_allowed
            ),
            redistribution_allowed=row.get(
                "redistribution_allowed", manifest.redistribution_allowed
            ),
            media_redistribution_allowed=row.get(
                "media_redistribution_allowed", manifest.media_redistribution_allowed
            ),
            attribution=row.get("attribution", manifest.attribution),
            source_component=row.get("source_component"),
            trust_status=str(row.get("trust_status", manifest.trust_status)),
        )
    raise ValueError(f"Unsupported dataset adapter: {adapter}")


def load_component_csv(path: str | Path) -> dict[str, dict[str, Any]]:
    def boolean(value: str) -> bool | None:
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
        return None

    result = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["source"]] = {
                **row,
                "rows": int(row["rows"]),
                "commercial_use": boolean(row.get("commercial_use", "")),
                "derivative_model_training_allowed": boolean(
                    row.get("derivative_model_training_allowed", "")
                ),
                "redistribution_allowed": boolean(row.get("redistribution_allowed", "")),
                "media_redistribution_allowed": boolean(
                    row.get("media_redistribution_allowed", "")
                ),
                "trust_status": row.get("trust_status") or "review",
            }
    return result


def iter_local_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    file_path = Path(path)
    if file_path.suffix.lower() == ".jsonl":
        with file_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"line {line_number} must be a JSON object")
                    yield row
    elif file_path.suffix.lower() == ".json":
        rows = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(rows, list):
            yield from rows
        elif isinstance(rows, dict):
            for record_id, value in rows.items():
                if not isinstance(value, dict):
                    raise ValueError(f"JSON record {record_id!r} must be an object")
                yield {"id": record_id, **value}
        else:
            raise ValueError("JSON dataset input must be an array or object of records")
    else:
        raise ValueError("Local dataset input must be .json or .jsonl")


def iter_hub_rows(manifest: DatasetManifest) -> Iterator[dict[str, Any]]:
    source_files = manifest.notes.get("data_files") or (
        [manifest.notes["data_file"]] if manifest.notes.get("data_file") else []
    )
    if source_files:
        if isinstance(source_files, str):
            source_files = [source_files]
        repo_id = urllib.parse.quote(manifest.dataset_id, safe="/")
        for source_file in source_files:
            source_path = str(source_file).replace("\\", "/")
            if source_path.startswith("/") or ".." in source_path.split("/"):
                raise ValueError("manifest data_file must be a relative repository path")
            quoted_path = urllib.parse.quote(source_path, safe="/")
            url = (
                f"https://huggingface.co/datasets/{repo_id}/resolve/"
                f"{manifest.revision}/{quoted_path}"
            )
            with urllib.request.urlopen(url, timeout=60) as response:
                if source_path.endswith(".jsonl.gz"):
                    stream = io.TextIOWrapper(gzip.GzipFile(fileobj=response), encoding="utf-8")
                    for line in stream:
                        if line.strip():
                            yield json.loads(line)
                elif source_path.endswith(".jsonl"):
                    stream = io.TextIOWrapper(response, encoding="utf-8")
                    for line in stream:
                        if line.strip():
                            yield json.loads(line)
                elif source_path.endswith(".json"):
                    data = json.load(io.TextIOWrapper(response, encoding="utf-8"))
                    if isinstance(data, list):
                        yield from data
                    elif isinstance(data, dict):
                        yield data
                    else:
                        raise ValueError(f"Expected JSON object or array at {source_path}")
                else:
                    raise ValueError(f"Unsupported pinned Hub source file: {source_path}")
        return
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "This dataset needs Parquet support. Install the data extra: "
            'python -m pip install "tiny-omni-decision[data]". '
            f"Optional reader initialization failed: {exc}"
        ) from exc
    parquet_files = manifest.notes.get("parquet_files")
    if parquet_files:
        from datasets import Audio

        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        repo_id = urllib.parse.quote(manifest.dataset_id, safe="/")
        urls = []
        for parquet_file in parquet_files:
            source_path = str(parquet_file).replace("\\", "/")
            if source_path.startswith("/") or ".." in source_path.split("/"):
                raise ValueError("manifest parquet_files must be relative repository paths")
            quoted_path = urllib.parse.quote(source_path, safe="/")
            urls.append(
                f"https://huggingface.co/datasets/{repo_id}/resolve/"
                f"{manifest.revision}/{quoted_path}"
            )
        rows = load_dataset(
            "parquet",
            data_files={manifest.split: urls},
            split=manifest.split,
            streaming=True,
        )
        if "audio" in rows.features:
            rows = rows.cast_column("audio", Audio(decode=False))
        for row in rows:
            yield {**row, "split": row.get("split", manifest.split)}
        return
    rows = load_dataset(
        manifest.dataset_id,
        manifest.subset,
        split=manifest.split,
        revision=manifest.revision,
        streaming=True,
    )
    from datasets import Audio

    for column_name, feature in rows.features.items():
        if isinstance(feature, Audio) and feature.decode:
            rows = rows.cast_column(column_name, Audio(decode=False))
    for row in rows:
        yield {**row, "split": row.get("split", manifest.split)}


def normalize_jsonl(
    rows: Iterable[dict[str, Any]],
    manifest: DatasetManifest,
    adapter: str,
    *,
    limit: int | None = None,
    seed: int | None = None,
    components: dict[str, dict[str, Any]] | None = None,
) -> Iterator[DecisionExample]:
    source_rows = 0
    for row in rows:
        if adapter in {"clevr4", "clevr-4"} and row.get("split", manifest.split) != manifest.split:
            continue
        if adapter in {"speech-commands", "speech_commands"}:
            if row.get("split") != manifest.split or _speech_command_label(row) is None:
                continue
        if adapter in {"clevrer", "clevrer-video", "clevrer-video-native"}:
            if not _clevrer_row_matches_split(row, manifest.split):
                continue
        if limit is not None and source_rows >= limit:
            break
        source_rows += 1
        if adapter in {"typed-decisions-synth", "typed_synth"}:
            examples = _flatten_typed_synth(row, manifest)
            for example in examples:
                yield shuffle_options(example, seed) if seed is not None else example
            continue
        if adapter in {"clevr4", "clevr-4"}:
            examples = _flatten_clevr4(row, manifest)
            for example in examples:
                yield shuffle_options(example, seed) if seed is not None else example
            continue
        if adapter in {"speech-commands", "speech_commands"}:
            example = _adapt_speech_command(row, manifest)
            yield shuffle_options(example, seed) if seed is not None else example
            continue
        if adapter in {"clevrer", "clevrer-video", "clevrer-video-native"}:
            examples = _flatten_clevrer(
                row, manifest, video_native=adapter == "clevrer-video-native"
            )
            for example in examples:
                yield shuffle_options(example, seed) if seed is not None else example
            continue
        try:
            example = adapt_row(row, manifest, adapter, components)
        except ValueError as exc:
            if adapter in {"open-jev", "openjev"} and str(exc) in {
                "soft or malformed target cannot be normalized as a single-choice label",
                "multi-label target cannot be represented by a single-choice example",
            }:
                continue
            raise
        yield shuffle_options(example, seed) if seed is not None else example


def deterministic_reservoir_sample(
    rows: Iterable[dict[str, Any]], *, limit: int, seed: int, source_key: str
) -> list[dict[str, Any]]:
    """Select a bounded, deterministic uniform sample while streaming source rows."""
    if limit < 1:
        raise ValueError("reservoir limit must be at least one")
    rng = random.Random(f"{seed}:{source_key}")
    reservoir: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index < limit:
            reservoir.append(row)
            continue
        replacement = rng.randrange(index + 1)
        if replacement < limit:
            reservoir[replacement] = row
    return reservoir


def _normal(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split())


def content_fingerprint(example: DecisionExample) -> str:
    media = sorted(media_identity(item) for item in example.media)
    payload = {
        "state": _normal(example.state),
        "question": _normal(example.question),
        "options": sorted(_normal(item) for item in example.options),
        "media": media,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def check_train_eval_splits(
    training: Iterable[DecisionExample], evaluation: Iterable[DecisionExample]
) -> dict[str, Any]:
    train_ids: set[tuple[str, str]] = set()
    train_hashes: set[str] = set()
    train_assets: set[tuple[str, str]] = set()
    train_media: set[str] = set()
    training_count = 0
    for item in training:
        training_count += 1
        train_ids.add((item.source, item.source_record_id))
        train_hashes.add(content_fingerprint(item))
        train_assets.add((item.source, source_asset_identity(item)))
        train_media.update(media_identity(reference) for reference in item.media)
    evaluation_count = 0
    shared_ids: set[tuple[str, str]] = set()
    shared_hashes: set[str] = set()
    shared_assets: set[tuple[str, str]] = set()
    shared_media: set[str] = set()
    for item in evaluation:
        evaluation_count += 1
        identity = (item.source, item.source_record_id)
        if identity in train_ids:
            shared_ids.add(identity)
        asset_identity = (item.source, source_asset_identity(item))
        if asset_identity in train_assets:
            shared_assets.add(asset_identity)
        fingerprint = content_fingerprint(item)
        if fingerprint in train_hashes:
            shared_hashes.add(fingerprint)
        shared_media.update(
            identity
            for identity in (media_identity(reference) for reference in item.media)
            if identity in train_media
        )
    if shared_ids or shared_hashes or shared_assets or shared_media:
        raise ValueError(
            f"train/eval contamination: {len(shared_ids)} shared source IDs, "
            f"{len(shared_assets)} shared source assets, "
            f"{len(shared_media)} shared media identities, "
            f"{len(shared_hashes)} normalized content fingerprints"
        )
    return {
        "training_records": training_count,
        "evaluation_records": evaluation_count,
        "shared_source_ids": 0,
        "shared_source_assets": 0,
        "shared_media_identities": 0,
        "shared_content_fingerprints": 0,
        "status": "disjoint",
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_statistics(examples: Iterable[DecisionExample]) -> dict[str, Any]:
    sources: Counter[str] = Counter()
    modalities: Counter[str] = Counter()
    options: Counter[str] = Counter()
    media_status: Counter[str] = Counter()
    unique_examples: set[tuple[str, str]] = set()
    unique_assets: set[tuple[str, str, str]] = set()
    assets_by_modality: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    assets_by_source: dict[str, set[str]] = defaultdict(set)
    assets_by_source_modality: dict[str, set[str]] = defaultdict(set)
    count = 0
    for example in examples:
        count += 1
        unique_examples.add((example.source, example.id))
        asset = (example.source, example.modality, source_asset_identity(example))
        unique_assets.add(asset)
        assets_by_modality[example.modality].add(asset)
        assets_by_source[example.source].add(asset[2])
        assets_by_source_modality[f"{example.source}:{example.modality}"].add(asset[2])
        sources[example.source] += 1
        modalities[example.modality] += 1
        options[str(len(example.options))] += 1
        for media in example.media:
            status = "local" if media.path else "source_reference"
            media_status[f"{media.kind}:{status}"] += 1
    return {
        "records": count,
        "unique_examples": len(unique_examples),
        "unique_underlying_assets": len(unique_assets),
        "unique_underlying_assets_by_modality": {
            modality: len(items) for modality, items in sorted(assets_by_modality.items())
        },
        "unique_underlying_assets_by_source": {
            source: len(items) for source, items in sorted(assets_by_source.items())
        },
        "unique_underlying_assets_by_source_modality": {
            key: len(items) for key, items in sorted(assets_by_source_modality.items())
        },
        "by_source": dict(sorted(sources.items())),
        "by_modality": dict(sorted(modalities.items())),
        "by_option_count": dict(sorted(options.items(), key=lambda item: int(item[0]))),
        "media_references": dict(sorted(media_status.items())),
    }
