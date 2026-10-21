"""One fixed inference-only TRAIN diagnosis; never opens a validation/audit corpus."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.semantic_state_diagnostic import (
    FIELDS,
    QUESTIONS,
    donor_control,
    log_probabilities,
    metrics,
    model_pair,
    options,
    oracle_state,
    predicted_state,
    select_train,
    summarize,
    targets,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = "configs/pretrained_reuse/semantic_state_bridge_diagnostic.json"
CODE = [CONFIG, "scripts/evaluate_semantic_state_bridge.py",
        "scripts/semantic_state_diagnostic.py", "src/tiny_omni_decision/dataset.py",
        "tests/test_semantic_state_diagnostic.py",
        "docs/pretrained_reuse/R2_SEMANTIC_BRIDGE_PROTOCOL_20261015.md"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(pins):
    if not pins:
        raise ValueError("empty pins")
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError(f"source changed: {path}")


def tensor_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replay-reference", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if bool(args.replay_reference) != bool(args.replay_sha256) or (
            args.smoke and args.replay_reference):
        raise ValueError("invalid execution mode")
    args.output.mkdir(parents=True, exist_ok=False)
    config = read(REPO / CONFIG)
    tick = time.perf_counter()
    result = dict(status="RUNNING", started_utc=datetime.now(UTC).isoformat(),
                  command=sys.argv, smoke=args.smoke, model_updates=0, cloud_cost=0,
                  validation_loaded=False, sealed_audit_loaded=False,
                  source_commit=subprocess.check_output(
                      ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())
    write(args.output / "result.json", result)
    try:
        import psutil
        # Before CUDA initialization; smoke/full/replay are separate processes.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        def guard():
            memory = psutil.Process().memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            size = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
            if (time.perf_counter() - tick > config["max_seconds"]
                    or peak > config["max_host_bytes"]
                    or torch.cuda.max_memory_allocated() > config["max_vram_bytes"]
                    or size > config["max_output_bytes"]
                    or shutil.disk_usage(args.output).free < config["min_free_disk_bytes"]):
                raise RuntimeError("time/host/VRAM/output/disk guard; incomplete diagnosis")

        subprocess.run(["git", "-C", str(REPO), "diff", "--exit-code", "HEAD", "--", *CODE],
                       check=True, capture_output=True)
        source_pins = {str(REPO / p): sha(REPO / p) for p in CODE}
        root, base = Path(config["root"]), Path(config["base_directory"])
        pins = {str(root / p): h for p, h in config["inputs"].items()}
        audit_file = base / "result-retry.json"
        pins[str(audit_file)] = config["base_audit_sha256"]
        verify(pins)
        audit = read(audit_file)
        pins.update({str(base / "model" / p): h["sha256"] for p, h in audit["files"].items()})
        verify(pins)
        reference = None
        if args.replay_reference:
            reference_file = args.replay_reference / "result.json"
            verify({str(reference_file): args.replay_sha256})
            reference = read(reference_file)
            if reference["status"] != "PASS_FIXED_TRAIN_DIAGNOSIS" or reference["smoke"]:
                raise ValueError("reference must be a completed full diagnosis")
            verify({str(args.replay_reference / p): h for p, h in reference["artifacts"].items()})
            if reference["input_pins"] != pins or reference["source_pins"] != source_pins:
                raise ValueError("replay source mismatch")
        guard()
        labels = [json.loads(line) for line in (
            root / "clevr4-full-coverage-seed17/train-labels.jsonl").read_text().splitlines()]
        manifest_path = root / "clevr4-full-coverage-seed17/train-image-manifest.jsonl"
        manifests = [json.loads(line) for line in manifest_path.read_text().splitlines()]
        if ([r["id"] for r in labels] != [r["id"] for r in manifests]
                or any(r["split"] != "train" for r in manifests)
                or len({r["sha256"] for r in manifests}) != len(manifests)):
            raise ValueError("TRAIN asset alignment or identity violation")
        selected = select_train(labels, config["smoke_images"] if args.smoke else config["images"])
        lookup = {r["id"]: i for i, r in enumerate(labels)}
        indices = [lookup[r["id"]] for r in selected]
        write(args.output / "selection.json", [dict(index=i, label=r, media=manifests[i])
                                               for i, r in zip(indices, selected, strict=True)])
        selection_hash = sha(args.output / "selection.json")
        torch.set_num_threads(4)
        torch.manual_seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        environment = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                           packages={name: importlib.metadata.version(name) for name in (
                               "transformers", "safetensors", "tokenizers", "psutil")},
                           cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                           gpu=torch.cuda.get_device_name(0),
                           gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
                           nvidia_smi=subprocess.check_output([
                               "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                               "--format=csv,noheader"], text=True).strip())
        cells = np.load(root / "videoprism-spatial-image-cache-01/train-cells.npy",
                        mmap_mode="r", allow_pickle=False)
        if cells.shape != (len(labels), 17, 768) or cells.dtype != np.float32:
            raise ValueError("unexpected spatial cache")
        subset = np.array(cells[indices])
        features = np.concatenate((subset[:, 0], subset[:, 1:].mean(1)), 1)
        head = torch.nn.Linear(1536, 40).double().eval().requires_grad_(False)
        head.load_state_dict(torch.load(root / "shared-affine-regularized-01/0.01/final.pt",
                                       map_location="cpu", weights_only=True), strict=True)
        with torch.inference_mode():
            native = head(torch.from_numpy(features).double()).numpy().reshape(-1, 4, 10)
        previous = np.load(root / "shared-affine-regularized-01/0.01/final-training-logits.npy",
                           mmap_mode="r", allow_pickle=False)[indices]
        difference = float(np.max(np.abs(native - previous)))
        if (difference > config["native_parity_absolute_tolerance"]
                or not np.array_equal(native.argmax(-1), previous.argmax(-1))):
            raise ValueError("native head parity failed")
        np.save(args.output / "native-logits.npy", native, allow_pickle=False)
        truth = np.array([targets(oracle_state(r)) for r in selected])
        np.save(args.output / "targets.npy", truth, allow_pickle=False)
        states = dict(oracle=[oracle_state(r) for r in selected],
                      predicted=[predicted_state(s) for s in native])
        write(args.output / "states.json", states)
        tokenizer = AutoTokenizer.from_pretrained(base / "model", local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            base / "model", local_files_only=True, attn_implementation="sdpa")
        delta = torch.load(root / "text-coverage-training-01/best-delta.pt",
                           weights_only=True, map_location="cpu")
        expected = {name for name, _ in model.named_parameters()
                    if name.startswith(("bert.encoder.layer.", "classifier."))}
        if set(delta) != expected:
            raise ValueError("unexpected delta coverage")
        loaded = model.load_state_dict(delta, strict=False)
        if loaded.unexpected_keys or set(loaded.missing_keys) != set(model.state_dict()) - expected:
            raise ValueError("delta load mismatch")
        model = model.cuda().eval().requires_grad_(False)
        parameters = sum(p.numel() for p in model.parameters())
        if parameters != 22713601 or model.config.num_hidden_layers != 6:
            raise ValueError("unexpected base architecture")
        before = dict(text=tensor_digest(model), image=tensor_digest(head))
        effective = dict(config=config, input_pins=pins, source_pins=source_pins,
                         environment=environment, questions=QUESTIONS, options=options(),
                         selection_sha256=selection_hash, text_parameters=parameters,
                         image_head_parameters=sum(p.numel() for p in head.parameters()),
                         text_dtype=str(next(model.parameters()).dtype),
                         attention=model.config._attn_implementation,
                         image_dtype="float64", trainable_parameters=0,
                         spatial_cache_provenance=read(root / "videoprism-spatial-image-cache-01"
                                                      / "config.json"),
                         text_base_revision=audit["revision"])
        write(args.output / "effective-config.json", effective)
        total_pairs, max_tokens, batches = 0, 0, 0
        batch_size = config["pair_batch_size"]

        def infer(facts, questions):
            nonlocal total_pairs, max_tokens, batches
            pairs = [model_pair(state, question, option["description"])
                     for state in facts for question in questions for option in options()]
            chunks = []
            with torch.inference_mode():
                for offset in range(0, len(pairs), batch_size):
                    guard()
                    batch = pairs[offset:offset + batch_size]
                    token = tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                      padding=True, truncation=False, return_tensors="pt")
                    max_tokens = max(max_tokens, token["input_ids"].shape[1])
                    if max_tokens > 512:
                        raise ValueError("token limit; truncation forbidden")
                    scores = model(**token.to("cuda")).logits[:, 0].cpu().numpy()
                    if not np.isfinite(scores).all():
                        raise FloatingPointError("nonfinite text scores")
                    chunks.append(scores)
                    total_pairs += len(batch)
                    batches += 1
            return np.concatenate(chunks).reshape(len(facts), len(questions), 40)

        arrays = {}
        for name, facts in states.items():
            arrays[name] = infer(facts, QUESTIONS)
            print(json.dumps(dict(completed=name, pairs=total_pairs,
                                  seconds=time.perf_counter() - tick)), flush=True)
        arrays["no_evidence"] = np.repeat(infer([{}], QUESTIONS), len(selected), axis=0)
        arrays["no_question"] = np.repeat(infer(states["oracle"], [""]), 4, axis=1)
        arrays["donor"], donor_targets = donor_control(arrays["oracle"], truth)
        np.save(args.output / "donor-targets.npy", donor_targets, allow_pickle=False)
        summaries = {}
        for name, scores in arrays.items():
            np.save(args.output / f"{name}-scores.npy", scores, allow_pickle=False)
            logp = log_probabilities(scores)
            np.save(args.output / f"{name}-logp.npy", logp, allow_pickle=False)
            np.save(args.output / f"{name}-probabilities.npy", np.exp(logp), allow_pickle=False)
            summaries[name] = summarize(scores, truth)
        summaries["donor_corrected"] = summarize(arrays["donor"], donor_targets)
        summaries["native_ten_option"] = {
            field: metrics(native[:, i], truth[:, i] % 10) for i, field in enumerate(FIELDS)}
        perception = native.argmax(-1) == truth % 10
        oracle_correct = arrays["oracle"].argmax(-1) == truth
        predicted_correct = arrays["predicted"].argmax(-1) == truth
        summaries["error_intersections"] = {
            f"native={a},oracle={b},predicted={c}": int((
                (perception == a) & (oracle_correct == b) & (predicted_correct == c)).sum())
            for a in (False, True) for b in (False, True) for c in (False, True)}
        if (summaries["no_question"]["overall"]["accuracy"] > 0.25
                or summaries["no_question"]["changed_question_pair_both_correct"] != 0):
            raise ValueError("question-free control violation")
        if total_pairs != len(selected) * 360 + 160:
            raise ValueError("inference accounting mismatch")
        write(args.output / "metrics.json", summaries)
        if dict(text=tensor_digest(model), image=tensor_digest(head)) != before:
            raise ValueError("frozen weights changed")
        if any(p.requires_grad or p.grad is not None
               for m in (model, head) for p in m.parameters()):
            raise ValueError("unexpected gradients")
        verify(pins)
        verify(source_pins)
        guard()
        if reference:
            replay_files = [p.name for p in args.output.glob("*.npy")] + [
                "selection.json", "states.json", "metrics.json", "effective-config.json"]
            for name in replay_files:
                if sha(args.output / name) != reference["artifacts"].get(name):
                    raise ValueError(f"fresh replay mismatch: {name}")
            result["fresh_replay_files_exact"] = replay_files
        result.update(
            status="PASS_TRAIN_ONLY_SMOKE" if args.smoke else "PASS_FIXED_TRAIN_DIAGNOSIS",
            finished_utc=datetime.now(UTC).isoformat(), seconds=time.perf_counter() - tick,
            input_pins=pins, source_pins=source_pins, unique_train_images=len(selected),
            unique_train_media=len({manifests[i]["sha256"] for i in indices}),
            unique_questions=4 * len(selected), actual_candidate_pairs=total_pairs,
            logical_condition_questions={name: len(selected) * 4 for name in arrays},
            forward_batches=batches, max_tokens=max_tokens, native_parity_max_abs=difference,
            native_argmax_exact=True, frozen_tensor_digests=before, sources_unchanged=True,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_host_bytes=getattr(psutil.Process().memory_info(), "peak_wset", None),
            free_disk_bytes=shutil.disk_usage(args.output).free,
            replay_reference_sha256=args.replay_sha256,
            artifacts={p.name: sha(p) for p in args.output.iterdir()
                       if p.is_file() and p.name != "result.json"})
        write(args.output / "result.json", result)
        print(json.dumps(dict(status=result["status"], seconds=result["seconds"],
                              pairs=total_pairs)), flush=True)
    except Exception:
        result.update(status="INCOMPLETE", traceback=traceback.format_exc(),
                      seconds=time.perf_counter() - tick,
                      finished_utc=datetime.now(UTC).isoformat())
        write(args.output / "result.json", result)
        raise


if __name__ == "__main__":
    main()
