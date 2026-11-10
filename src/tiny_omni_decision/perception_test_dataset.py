"""Strict adapter for the public Perception Test video-QA train/valid splits.

The official test split is intentionally unsupported. Group identity is the
complete source video ID; callers must keep all questions from that video in
the same split and supply a verified relative media path.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from .schema import DecisionExample, LicenseProvenance, MediaRef

PERCEPTION_TEST_REPOSITORY_REVISION = "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf"
PERCEPTION_TEST_SOURCE = "google-deepmind/perception_test"


def assert_video_id_disjoint(
    train_video_ids: Iterable[str], valid_video_ids: Iterable[str]
) -> None:
    """Fail closed if any underlying video appears in both public splits."""
    train_ids = set(train_video_ids)
    valid_ids = set(valid_video_ids)
    overlap = train_ids & valid_ids
    if overlap:
        sample = sorted(overlap)[:3]
        raise ValueError(f"train/valid video ID overlap ({len(overlap)}): {sample}")


def build_perception_test_video_qa(
    annotations: Mapping[str, Any],
    *,
    split: Literal["train", "valid"],
    video_paths: Mapping[str, str],
) -> list[DecisionExample]:
    """Map source QA rows to typed video decisions without changing choices.

    ``video_paths`` must be produced from a verified local media inventory.
    This adapter does not fetch media and does not accept the held-out test
    split. It preserves option order and maps the publisher's integer
    ``answer_id`` to the corresponding exact option string.
    """
    if split not in {"train", "valid"}:
        raise ValueError("only train and valid splits are supported; test is sealed")

    examples: list[DecisionExample] = []
    for video_id in sorted(annotations):
        video = annotations[video_id]
        if not isinstance(video, Mapping):
            raise ValueError(f"video annotation must be an object: {video_id}")
        metadata = video.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("split") != split:
            raise ValueError(f"split metadata does not match requested split for {video_id}")
        if metadata.get("video_id") != video_id:
            raise ValueError(f"video ID metadata does not match annotation key: {video_id}")
        media_path = video_paths.get(video_id)
        if not isinstance(media_path, str) or not media_path:
            raise ValueError(f"video path is missing for {video_id}")

        questions = video.get("mc_question")
        if not isinstance(questions, list):
            raise ValueError(f"mc_question must be a list for {video_id}")
        if any(not isinstance(question, Mapping) for question in questions):
            raise ValueError(f"question annotation must be an object for {video_id}")
        question_ids: set[int] = set()
        for question in questions:
            question_id = question.get("id")
            if isinstance(question_id, bool) or not isinstance(question_id, int):
                raise ValueError(f"question id must be an integer for {video_id}")
            if question_id in question_ids:
                raise ValueError(f"duplicate question id for {video_id}/{question_id}")
            question_ids.add(question_id)

        for question in sorted(questions, key=lambda row: row["id"]):
            question_id = question.get("id")
            prompt = question.get("question")
            options = question.get("options")
            answer_id = question.get("answer_id")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"question text is missing for {video_id}/{question_id}")
            if (
                not isinstance(options, list)
                or len(options) != 3
                or any(not isinstance(option, str) or not option.strip() for option in options)
            ):
                raise ValueError(
                    f"expected exactly three nonempty options for {video_id}/{question_id}"
                )
            if isinstance(answer_id, bool) or not isinstance(answer_id, int):
                raise ValueError(f"answer_id must be an integer for {video_id}/{question_id}")
            if not 0 <= answer_id < len(options):
                raise ValueError(f"answer_id is out of range for {video_id}/{question_id}")

            # Stable choice labels preserve the publisher's answer index even
            # when two source options have identical text. Apply to every row
            # so the source's occasional duplicate distractors are handled
            # uniformly rather than filtered out.
            labeled_options = [
                f"{chr(ord('A') + index)}: {option.strip()}"
                for index, option in enumerate(options)
            ]
            examples.append(
                DecisionExample(
                    id=f"perception-test/{split}/{video_id}/{question_id}",
                    modality="video",
                    state="Perception Test video QA",
                    question=prompt,
                    options=labeled_options,
                    target=labeled_options[answer_id],
                    media=[MediaRef(kind="video", path=media_path, license="CC-BY-4.0")],
                    source=PERCEPTION_TEST_SOURCE,
                    source_revision=PERCEPTION_TEST_REPOSITORY_REVISION,
                    source_record_id=f"{video_id}/{question_id}",
                    split=split,
                    source_target=answer_id,
                    provenance=LicenseProvenance(
                        license="CC-BY-4.0",
                        commercial_use=True,
                        derivative_model_training_allowed=True,
                        redistribution_allowed=True,
                        media_redistribution_allowed=True,
                        attribution="Perception Test; Pătrăucean et al., NeurIPS 2023",
                        source_component="Perception Test annotations and video",
                        trust_status="trusted",
                    ),
                )
            )
    return examples
