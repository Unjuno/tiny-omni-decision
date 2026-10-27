"""CPU wiring smoke for the three set-decision arms; no checkpoint is saved."""

import importlib.util
import sys

import torch

from scripts.source_set_decision import decode_set, set_loss
from scripts.source_set_metrics import score_record
from scripts.train_source_unimodal_cv import (
    AUDIO,
    LABELS,
    PINNED,
    PLAN,
    SOURCE,
    SOURCE_RUNNER,
    VIDEO,
    CandidateModel,
    digest,
    fold_assignment,
)


def main():
    for path, expected in PINNED.items():
        if digest(path) != expected:
            raise ValueError(f"pinned input mismatch: {path}")
    sys.path.insert(0, str(SOURCE / "src"))
    spec = importlib.util.spec_from_file_location("pinned_source", SOURCE_RUNNER)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    examples, _ = source.load_examples(PLAN, LABELS, VIDEO, AUDIO)
    groups = fold_assignment([e.video_id for e in examples])
    train = [e for e in examples if groups[e.video_id] != 0]
    valid = [e for e in examples if groups[e.video_id] == 0]
    if {e.video_id for e in train} & {e.video_id for e in valid}:
        raise ValueError("video-group overlap")
    torch.set_num_threads(2)
    for mode in ("audio", "video", "joint"):
        torch.manual_seed(17)
        model = CandidateModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        video, audio, targets, mask, _ = source.collate_examples(train, list(range(8)))
        loss = set_loss(model(video, audio, mask, mode), mask, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError("nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        video, audio, _, mask, batch = source.collate_examples(valid, list(range(8)))
        with torch.no_grad():
            logits = model(video, audio, mask, mode)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite validation logits")
            selected = decode_set(logits, mask)
            conditional = torch.sigmoid(logits[:, :-1])
            none = torch.sigmoid(logits[:, -1])
            for j, example in enumerate(batch):
                ids = list(example.candidate_ids)
                choices = [ids[k] for k in range(len(ids)) if selected[j, k]]
                if selected[j, -1]:
                    choices.append("none_visible")
                score_record(
                    ids,
                    sorted(example.target_candidate_ids),
                    choices,
                    conditional[j, : len(ids)].tolist(),
                    float(none[j]),
                )
        print(
            mode,
            "train_loss",
            round(float(loss), 5),
            "validation_rows",
            len(batch),
            "parameters",
            sum(p.numel() for p in model.parameters()),
        )
    print("CPU wiring smoke only; CUDA initialized:", torch.cuda.is_initialized())


if __name__ == "__main__":
    main()
