"""Bounded matched descriptive grounding probe on immutable frozen features."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from scripts.range_generalization import sha, write
from scripts.train_frozen_text_probe import CandidateScorer, _metrics
from scripts.train_frozen_video_probe import _combine, _predict, _text_features


def metrics(rows, logits, targets):
    result = {"overall": _metrics(logits, targets)}
    for field in ("taxonomy", "question_type"):
        for kind in sorted({r[field] for r in rows}):
            indices = [i for i, r in enumerate(rows) if r[field] == kind]
            result[field + ":" + kind] = _metrics(
                [logits[i] for i in indices], [targets[i] for i in indices]
            )
    values = [v for k, v in result.items() if k.startswith("taxonomy:")]
    result["macro_taxonomy"] = {
        k: sum(v[k] for v in values) / len(values)
        for k in ("accuracy", "nll", "brier", "ece_15_bins")
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    inputs = root / "video-grounding-descriptive-inputs-01"
    report = json.loads((inputs / "report.json").read_text())
    for p, h in report["source_hashes"].items():
        if sha(Path(p)) != h:
            raise ValueError("input hash changed")
    cache = Path(report["feature_cache"])
    if sha(cache) != report["feature_cache_sha256"]:
        raise ValueError("cache changed")
    model_path = root / "path-a-audio-multiquery-runtime-bundle-v2/models/text-minilm-en"
    model_hash = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
    if sha(model_path / "model.safetensors") != model_hash:
        raise ValueError("text model changed")
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    rows = {
        s: [json.loads(line) for line in (inputs / f"{s}.jsonl").read_text().splitlines()]
        for s in ("train", "validation")
    }
    with np.load(cache, allow_pickle=False) as data:
        lookup = dict(zip(data["media_sha256"].astype(str), data["features"], strict=True))
    video = {
        s: torch.from_numpy(np.stack([lookup[r["media_sha256"]] for r in rs]))
        for s, rs in rows.items()
    }
    args.output.mkdir()
    started = time.perf_counter()
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    model.requires_grad_(False)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    text, targets = {}, {}
    for s, rs in rows.items():
        text[s], targets[s] = _text_features(rs, tokenizer, model, device)
    if not all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in before.items()):
        raise ValueError("frozen encoder modified")
    encoder_params = sum(p.numel() for p in model.parameters())
    del model, tokenizer, before
    extraction_seconds = time.perf_counter() - started
    epochs = 2 if args.smoke else 24
    write(
        args.output / "config.json",
        dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            seed=17,
            epochs=epochs,
            lr=0.001,
            weight_decay=0.001,
            batch=8,
            inputs=report,
            encoder_parameters=encoder_params,
            video_parameters=114365184,
            text_model_sha256=model_hash,
            text_revision="4ca70771034acceecb2e72475f72050fcdde4ddc",
            smoke=args.smoke,
            selection="earliest minimum equal taxonomy NLL including initialization",
        ),
    )
    results = {}
    for mode in ("video", "no_video"):
        folder = args.output / mode
        folder.mkdir()
        torch.manual_seed(17)
        features = {
            s: _combine(v if mode == "video" else torch.zeros_like(v), text[s])
            for s, v in video.items()
        }
        head = CandidateScorer(features["train"][0].shape[-1]).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.001)
        generator = torch.Generator().manual_seed(17)
        history = []
        best = float("inf")
        start = time.perf_counter()
        for epoch in range(epochs + 1):
            if epoch:
                head.train()
                order = torch.randperm(len(rows["train"]), generator=generator).tolist()
                for offset in range(0, len(order), 8):
                    indices = order[offset : offset + 8]
                    feats = [features["train"][i].to(device) for i in indices]
                    widths = [len(f) for f in feats]
                    scores = head(torch.cat(feats)).split(widths)
                    loss = torch.stack(
                        [
                            -torch.log_softmax(v, dim=0)[targets["train"][i]]
                            for v, i in zip(scores, indices, strict=True)
                        ]
                    ).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            values = {s: _predict(head, f, device) for s, f in features.items()}
            entry = dict(epoch=epoch, **{s: metrics(rows[s], values[s], targets[s]) for s in rows})
            history.append(entry)
            score = entry["validation"]["macro_taxonomy"]["nll"]
            if score < best:
                best, best_epoch = score, epoch
                best_logits = values["validation"]
                torch.save(
                    {k: v.cpu().clone() for k, v in head.state_dict().items()}, folder / "best.pt"
                )
            write(folder / "history.json", history)
            print(mode, epoch, score, flush=True)
        torch.save(head.state_dict(), folder / "final.pt")
        head.load_state_dict(torch.load(folder / "best.pt", weights_only=True, map_location=device))
        reloaded = _predict(head, features["validation"], device)
        if not all(torch.equal(a, b) for a, b in zip(reloaded, best_logits, strict=True)):
            raise ValueError("best reload mismatch")
        scenes = sorted({r["scene_index"] for r in rows["validation"]})
        scene_media = {r["scene_index"]: r["media_sha256"] for r in rows["validation"]}
        mapping = dict(zip(scenes, scenes[1:] + scenes[:1], strict=True))
        wrong = torch.from_numpy(
            np.stack([lookup[scene_media[mapping[r["scene_index"]]]] for r in rows["validation"]])
        )
        wrong_features = _combine(
            wrong if mode == "video" else torch.zeros_like(wrong), text["validation"]
        )
        shuffled = _predict(head, wrong_features, device)
        for name, pred in (("matched", reloaded), ("shuffled", shuffled)):
            write(
                folder / f"{name}-predictions.json",
                [
                    dict(
                        id=r["id"],
                        scene_index=r["scene_index"],
                        taxonomy=r["taxonomy"],
                        target_index=t,
                        probabilities=torch.softmax(p, dim=0).tolist(),
                    )
                    for r, t, p in zip(rows["validation"], targets["validation"], pred, strict=True)
                ],
            )
        results[mode] = dict(
            best_epoch=best_epoch,
            epochs=epochs,
            reload_exact=True,
            head_parameters=sum(p.numel() for p in head.parameters()),
            seconds=time.perf_counter() - start,
            best_sha256=sha(folder / "best.pt"),
            final_sha256=sha(folder / "final.pt"),
            selected=history[best_epoch],
            shuffled=metrics(rows["validation"], shuffled, targets["validation"]),
        )
    write(
        args.output / "result.json",
        dict(
            status="PASS_EXECUTION_ONLY",
            smoke=args.smoke,
            conditions=results,
            extraction_seconds=extraction_seconds,
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
        ),
    )


if __name__ == "__main__":
    main()
