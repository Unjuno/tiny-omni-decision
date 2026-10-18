"""Text-free CLEVRER Decision readout over frozen framewise video features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

PAD = "<pad>"
UNK = "<unk>"
FEATURE_FRAMES = 8
FEATURE_WIDTH = 768
VIDEO_WIDTH = 128


@dataclass(frozen=True)
class VideoVocab:
    program_to_id: dict[str, int]
    taxonomy_to_id: dict[str, int]
    option_to_id: dict[str, int]


@dataclass(frozen=True)
class EncodedVideoRows:
    video: Tensor
    program_ids: Tensor
    taxonomy_ids: Tensor
    option_ids: Tensor
    option_mask: Tensor
    targets: Tensor
    sample_ids: tuple[str, ...]


def validate_video_splits(
    training_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
    *,
    allow_normalized_content_overlap: bool = False,
) -> None:
    """Reject scene, media, sample-ID or normalized question/option overlap."""
    if not training_rows or not validation_rows:
        raise ValueError("training and validation splits must be nonempty")

    def identities(rows: Sequence[Mapping[str, object]], split_name: str) -> tuple[set[str], ...]:
        ids: set[str] = set()
        scenes: set[str] = set()
        media: set[str] = set()
        content: set[tuple[str, tuple[str, ...]]] = set()
        for row in rows:
            sample_id = row.get("id")
            scene = row.get("scene_group_id")
            media_hash = row.get("media_sha256")
            question = row.get("question")
            options = row.get("options")
            if (
                not isinstance(sample_id, str)
                or not sample_id
                or not isinstance(scene, str)
                or not scene
                or not isinstance(media_hash, str)
                or not media_hash
                or not isinstance(question, str)
                or not isinstance(options, list)
                or any(not isinstance(option, str) for option in options)
            ):
                raise ValueError(f"invalid split identity/content in {split_name}")
            if sample_id in ids:
                raise ValueError(f"duplicate sample ID in {split_name}: {sample_id}")
            ids.add(sample_id)
            scenes.add(scene)
            media.add(media_hash)
            normalized_question = " ".join(question.casefold().split())
            normalized_options = tuple(sorted(" ".join(x.casefold().split()) for x in options))
            content.add((normalized_question, normalized_options))
        return ids, scenes, media, content

    train_ids, train_scenes, train_media, train_content = identities(training_rows, "training")
    val_ids, val_scenes, val_media, val_content = identities(validation_rows, "validation")
    if train_ids & val_ids:
        raise ValueError("train/validation sample ID overlap")
    if train_scenes & val_scenes:
        raise ValueError("train/validation scene group overlap")
    if train_media & val_media:
        raise ValueError("train/validation media identity overlap")
    if train_content & val_content and not allow_normalized_content_overlap:
        raise ValueError("normalized question/options overlap across splits")


def _vocabulary(values: set[str]) -> dict[str, int]:
    return {PAD: 0, UNK: 1, **{value: index + 2 for index, value in enumerate(sorted(values))}}


def build_video_vocab(training_rows: Sequence[Mapping[str, object]]) -> VideoVocab:
    """Build deterministic symbolic vocabularies using training rows only."""
    if not training_rows:
        raise ValueError("training rows must be nonempty")
    programs: set[str] = set()
    taxonomies: set[str] = set()
    options: set[str] = set()
    for row in training_rows:
        program = row.get("program")
        taxonomy = row.get("taxonomy")
        candidates = row.get("options")
        if (
            not isinstance(program, list)
            or not program
            or any(not isinstance(token, str) or not token.strip() for token in program)
            or not isinstance(taxonomy, str)
            or not taxonomy.strip()
            or not isinstance(candidates, list)
            or len(candidates) < 2
            or any(not isinstance(value, str) or not value.strip() for value in candidates)
        ):
            raise ValueError("rows need nonempty typed programs, taxonomy and options")
        programs.update(program)
        taxonomies.add(taxonomy)
        options.update(candidates)
    return VideoVocab(_vocabulary(programs), _vocabulary(taxonomies), _vocabulary(options))


def encode_video_rows(
    rows: Sequence[Mapping[str, object]],
    features_by_media: Mapping[str, np.ndarray | Tensor],
    vocab: VideoVocab,
) -> EncodedVideoRows:
    """Encode only video features, typed program, taxonomy, options and target.

    Natural-language question strings, source IDs and scene identifiers are never
    copied into the model tensors; IDs are retained only for prediction receipts.
    """
    if not rows:
        raise ValueError("rows must be nonempty")
    video_rows: list[Tensor] = []
    program_rows: list[list[int]] = []
    taxonomy_rows: list[int] = []
    option_rows: list[list[int]] = []
    target_rows: list[int] = []
    ids: list[str] = []
    max_options = 0
    for row in rows:
        media_hash = row.get("media_sha256")
        if not isinstance(media_hash, str) or media_hash not in features_by_media:
            raise ValueError(f"missing video features for sample {row.get('id')}")
        video = torch.as_tensor(features_by_media[media_hash], dtype=torch.float32)
        if video.shape != (FEATURE_FRAMES, FEATURE_WIDTH) or not torch.isfinite(video).all():
            raise ValueError(f"video features must be finite [{FEATURE_FRAMES},{FEATURE_WIDTH}]")
        program = row.get("program")
        taxonomy = row.get("taxonomy")
        options = row.get("options")
        target = row.get("target")
        sample_id = row.get("id")
        if (
            not isinstance(program, list)
            or not program
            or not isinstance(taxonomy, str)
            or not isinstance(options, list)
            or len(options) < 2
            or not isinstance(target, str)
            or target not in options
            or not isinstance(sample_id, str)
            or not sample_id
        ):
            raise ValueError("invalid typed video Decision row")
        if len(set(options)) != len(options):
            raise ValueError(f"duplicate option class in {sample_id}")
        video_rows.append(video)
        program_rows.append(
            [vocab.program_to_id.get(str(token), vocab.program_to_id[UNK]) for token in program]
        )
        taxonomy_rows.append(vocab.taxonomy_to_id.get(taxonomy, vocab.taxonomy_to_id[UNK]))
        option_rows.append(
            [vocab.option_to_id.get(value, vocab.option_to_id[UNK]) for value in options]
        )
        target_rows.append(options.index(target))
        ids.append(sample_id)
        max_options = max(max_options, len(options))

    max_program = max(map(len, program_rows))
    program_ids = torch.zeros((len(rows), max_program), dtype=torch.long)
    option_ids = torch.zeros((len(rows), max_options), dtype=torch.long)
    option_mask = torch.zeros((len(rows), max_options), dtype=torch.bool)
    for index, values in enumerate(program_rows):
        program_ids[index, : len(values)] = torch.tensor(values, dtype=torch.long)
    for index, values in enumerate(option_rows):
        option_ids[index, : len(values)] = torch.tensor(values, dtype=torch.long)
        option_mask[index, : len(values)] = True
    return EncodedVideoRows(
        video=torch.stack(video_rows),
        program_ids=program_ids,
        taxonomy_ids=torch.tensor(taxonomy_rows, dtype=torch.long),
        option_ids=option_ids,
        option_mask=option_mask,
        targets=torch.tensor(target_rows, dtype=torch.long),
        sample_ids=tuple(ids),
    )


class TextFreeVideoReadout(nn.Module):
    """Score supplied choices from frozen video features and typed criteria."""

    def __init__(self, vocab: VideoVocab, temporal_mode: str) -> None:
        super().__init__()
        if temporal_mode not in {"mean_mlp", "bilstm"}:
            raise ValueError("temporal_mode must be 'mean_mlp' or 'bilstm'")
        self.temporal_mode = temporal_mode
        self.video_projection = nn.Sequential(
            nn.Linear(FEATURE_WIDTH, VIDEO_WIDTH),
            nn.LayerNorm(VIDEO_WIDTH),
            nn.GELU(),
        )
        self.program_embedding = nn.Embedding(len(vocab.program_to_id), 32, padding_idx=0)
        self.taxonomy_embedding = nn.Embedding(len(vocab.taxonomy_to_id), 16, padding_idx=0)
        self.option_embedding = nn.Embedding(len(vocab.option_to_id), 32, padding_idx=0)
        self.candidate_scorer = nn.Sequential(
            nn.Linear(VIDEO_WIDTH + 32 + 16 + 32, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        if temporal_mode == "mean_mlp":
            self.temporal_block = nn.Sequential(
                nn.Linear(VIDEO_WIDTH, 384),
                nn.GELU(),
                nn.Linear(384, VIDEO_WIDTH),
                nn.LayerNorm(VIDEO_WIDTH),
            )
        else:
            self.temporal_block = nn.LSTM(
                input_size=VIDEO_WIDTH,
                hidden_size=64,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.temporal_norm = nn.LayerNorm(VIDEO_WIDTH)

    def forward(
        self,
        video: Tensor,
        program_ids: Tensor,
        taxonomy_ids: Tensor,
        option_ids: Tensor,
        option_mask: Tensor,
    ) -> Tensor:
        if video.ndim != 3 or video.shape[1:] != (FEATURE_FRAMES, FEATURE_WIDTH):
            raise ValueError("video input must have shape [batch,8,768]")
        batch_size = video.shape[0]
        if (
            program_ids.ndim != 2
            or program_ids.shape[0] != batch_size
            or taxonomy_ids.shape != (batch_size,)
            or option_ids.ndim != 2
            or option_ids.shape[0] != batch_size
            or option_mask.shape != option_ids.shape
            or not bool(option_mask.any(dim=1).all())
        ):
            raise ValueError("program, taxonomy and candidate tensors have incompatible shapes")
        if not all(
            torch.isfinite(value).all()
            for value in (video, self.program_embedding.weight, self.candidate_scorer[0].weight)
        ):
            raise ValueError("model inputs and weights must be finite")

        projected = self.video_projection(video)
        if self.temporal_mode == "mean_mlp":
            temporal = self.temporal_block(projected)
        else:
            temporal, _ = self.temporal_block(projected)
            temporal = self.temporal_norm(temporal)
        video_context = temporal.mean(dim=1)

        program_mask = program_ids.ne(0)
        program_features = self.program_embedding(program_ids)
        program_context = (program_features * program_mask.unsqueeze(-1)).sum(dim=1)
        program_context = program_context / program_mask.sum(dim=1, keepdim=True).clamp_min(1)
        taxonomy_context = self.taxonomy_embedding(taxonomy_ids)
        option_context = self.option_embedding(option_ids)
        candidate_count = option_ids.shape[1]
        contexts = torch.cat(
            (
                video_context[:, None, :].expand(-1, candidate_count, -1),
                program_context[:, None, :].expand(-1, candidate_count, -1),
                taxonomy_context[:, None, :].expand(-1, candidate_count, -1),
                option_context,
            ),
            dim=-1,
        )
        logits = self.candidate_scorer(contexts).squeeze(-1)
        return logits.masked_fill(~option_mask, -1e9)
