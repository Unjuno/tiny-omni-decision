"""Fixed Whisper candidate-transcript likelihood; no training or calibration."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer
from transformers.modeling_outputs import BaseModelOutput

from scripts.evaluate_librispeech_audio_hard_negatives import CONDITIONS, _metric_summary
from tiny_omni_decision.audio_runtime import AUDIO_WEIGHTS_SHA256
from tiny_omni_decision.transcript_likelihood import (
    mean_transcript_logprob,
    rotated_speaker_indices,
)

INPUT_HASH = "1f17d86a15faecc3ba3e13cd406dfd0119ef17850d1a7206956a70f4eb801923"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def speaker_interval(rows, predictions):
    speakers = sorted({r["speaker_id"] for r in rows})
    differences, counts = [], []
    for speaker in speakers:
        ids = [i for i, r in enumerate(rows) if r["speaker_id"] == speaker]
        counts.append(len(ids))
        total = 0
        for i in ids:
            a, b = (predictions[c][i] for c in CONDITIONS)
            total += int(b["prediction_index"] == b["target_index"])
            total -= int(a["prediction_index"] == a["target_index"])
        differences.append(total)
    rng = np.random.default_rng(17)
    draws = rng.integers(len(speakers), size=(10000, len(speakers)))
    boot = np.asarray(differences)[draws].sum(1) / np.asarray(counts)[draws].sum(1)
    return dict(
        unit="speaker",
        groups=len(speakers),
        replicates=10000,
        seed=17,
        direction="hard minus random",
        point=sum(differences) / sum(counts),
        percentile_95_interval=np.quantile(boot, [0.025, 0.975]).tolist(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "examples", "source", "baseline", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--audio-control", choices=["matched", "rotated"], default="matched")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(args.examples) != INPUT_HASH:
        raise ValueError("fixed examples changed")
    weights = args.model / "model.safetensors"
    if sha(weights) != AUDIO_WEIGHTS_SHA256:
        raise ValueError("model hash mismatch")
    rows = [json.loads(s) for s in args.examples.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 1938 or len({r["id"] for r in rows}) != 1938:
        raise ValueError("wrong fixed example count")
    audio_rows = rows
    if args.audio_control == "rotated":
        indices = rotated_speaker_indices([r["speaker_id"] for r in rows])
        audio_rows = [rows[i] for i in indices]
    if args.smoke:
        rows = rows[:8]
        audio_rows = audio_rows[:8]
    baseline = {}
    for condition in CONDITIONS:
        path = args.baseline / f"{condition}-predictions.jsonl"
        previous = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
        previous = previous[: len(rows)]
        for row, pred in zip(rows, previous, strict=True):
            if (
                row["id"] != pred["sample_id"]
                or row["options_by_condition"][condition] != pred["options"]
                or row["target_index"] != pred["target_index"]
            ):
                raise ValueError("baseline alignment mismatch")
        baseline[condition] = dict(sha256=sha(path), metrics=_metric_summary(previous))
    args.output.mkdir(parents=True)
    config = dict(
        argv=vars(args).copy(),
        started_utc=datetime.now(UTC).isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        runner_sha256=sha(Path(__file__)),
        input_sha256=INPUT_HASH,
        model_files={x.name: sha(x) for x in sorted(args.model.iterdir()) if x.is_file()},
        scoring="mean transcript/EOS logprob excluding prefix/padding; softmax T1",
        python=platform.python_version(),
        versions={
            n: importlib.metadata.version(n)
            for n in ["torch", "transformers", "numpy", "soundfile"]
        },
    )
    config["argv"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write(args.output / "config.json", config)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    start = time.perf_counter()
    tokenizer = WhisperTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        language="english",
        task="transcribe",
        predict_timestamps=False,
    )
    processor = WhisperFeatureExtractor.from_pretrained(args.model, local_files_only=True)
    model, loading = WhisperForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32, output_loading_info=True
    )
    if loading.get("missing_keys") or loading.get("unexpected_keys"):
        raise ValueError(f"nonexact load: {loading}")
    model.eval().requires_grad_(False).to("cuda")
    prefix = tokenizer.prefix_tokens
    if prefix != [50258, 50259, 50359, 50363]:
        raise ValueError("unexpected language/task prefix")
    parameters = sum(x.numel() for x in model.parameters())
    predictions = {c: [] for c in CONDITIONS}
    checks = []
    torch.cuda.reset_peak_memory_stats()

    def score(encoded, options):
        sequences = [tokenizer(s)["input_ids"] for s in options]
        for ids in sequences:
            if ids[: len(prefix)] != prefix or ids[-1] != tokenizer.eos_token_id:
                raise ValueError("invalid prefix/EOS")
            if len(ids) > model.config.max_target_positions:
                raise ValueError("decoder would truncate")
        lengths = [len(x) for x in sequences]
        ids = torch.full(
            (len(options), max(lengths)), tokenizer.pad_token_id, device="cuda", dtype=torch.long
        )
        mask = torch.zeros_like(ids)
        for i, tokens in enumerate(sequences):
            ids[i, : len(tokens)] = torch.tensor(tokens, device="cuda")
            mask[i, : len(tokens)] = 1
        output = model(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoded.expand(len(options), -1, -1)),
            decoder_input_ids=ids[:, :-1],
            decoder_attention_mask=mask[:, :-1],
            use_cache=False,
        ).logits
        chosen = output.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        values = mean_transcript_logprob(chosen.cpu().numpy(), lengths, prefix_length=len(prefix))
        return values, [n - len(prefix) for n in lengths]

    with torch.inference_mode():
        for i, row in enumerate(rows):
            audio_row = audio_rows[i]
            audio = args.source / audio_row["audio_path"]
            if sha(audio) != audio_row["audio_sha256"]:
                raise ValueError("audio hash mismatch")
            pcm, sr = sf.read(audio, dtype="int16")
            if sr != 16000 or pcm.ndim != 1 or not 0 < len(pcm) <= processor.n_samples:
                raise ValueError("audio would truncate/resample or is invalid")
            features = processor(
                pcm.astype(np.float32) / 32768, sampling_rate=sr, return_tensors="pt"
            ).input_features.to("cuda")
            encoded = model.model.encoder(features).last_hidden_state
            options = list(
                dict.fromkeys(s for c in CONDITIONS for s in row["options_by_condition"][c])
            )
            values, lengths = score(encoded, options)
            if args.smoke:
                reverse, _ = score(encoded, options[::-1])
                single = np.array([score(encoded, [s])[0][0] for s in options])
                np.testing.assert_allclose(values, reverse[::-1], atol=2e-5, rtol=1e-5)
                np.testing.assert_allclose(values, single, atol=2e-5, rtol=1e-5)
                checks.append(
                    dict(
                        id=row["id"],
                        permutation_max_abs=float(np.max(np.abs(values - reverse[::-1]))),
                        batching_max_abs=float(np.max(np.abs(values - single))),
                    )
                )
            for condition in CONDITIONS:
                choices = row["options_by_condition"][condition]
                indices = [options.index(s) for s in choices]
                logits = values[indices]
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                pred = dict(
                    sample_id=row["id"],
                    speaker_id=row["speaker_id"],
                    audio_sha256=audio_row["audio_sha256"],
                    audio_sample_id=audio_row["id"],
                    audio_speaker_id=audio_row["speaker_id"],
                    options=choices,
                    target_index=row["target_index"],
                    prediction_index=int(probs.argmax()),
                    probabilities=probs.tolist(),
                    scores=logits.tolist(),
                    scored_token_counts=[lengths[j] for j in indices],
                )
                predictions[condition].append(pred)
                with (args.output / f"{condition}-predictions.jsonl").open(
                    "a", encoding="utf-8"
                ) as f:
                    f.write(json.dumps(pred) + "\n")
            if (i + 1) % 128 == 0:
                print(
                    json.dumps(dict(completed=i + 1, seconds=time.perf_counter() - start)),
                    flush=True,
                )
    torch.cuda.synchronize()
    if sha(weights) != AUDIO_WEIGHTS_SHA256:
        raise ValueError("weights changed")
    result = dict(
        status="PASS_EXECUTION_ONLY",
        smoke=args.smoke,
        count=len(rows),
        parameters=parameters,
        tied_output_embeddings=(
            model.proj_out.weight.data_ptr() == model.model.decoder.embed_tokens.weight.data_ptr()
        ),
        metrics={c: _metric_summary(predictions[c]) for c in CONDITIONS},
        baseline=baseline,
        hard_minus_random=speaker_interval(rows, predictions),
        smoke_checks=checks,
        seconds=time.perf_counter() - start,
        peak_vram_bytes=torch.cuda.max_memory_allocated(),
        gpu=torch.cuda.get_device_name(),
        cost_usd=0,
        weights_unchanged=True,
        finished_utc=datetime.now(UTC).isoformat(),
        training_performed=False,
    )
    write(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
