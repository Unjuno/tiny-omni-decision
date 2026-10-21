"""Simultaneous local worker residency, not a complete Omni decision runtime."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def visual_worker(out):
    import jax
    import numpy as np
    import psutil

    from scripts.extract_frozen_videoprism_features import CHECKPOINT_SHA256, SOURCE_COMMIT
    from scripts.range_generalization import sha, write
    from tiny_omni_decision.shared_visual_encoder import SharedVisualEncoder

    source = Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/source")
    checkpoint = source.parent.parent / "videoprism-base-f16r288.npz"
    assert sha(checkpoint) == CHECKPOINT_SHA256
    assert (
        subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        == SOURCE_COMMIT
    )
    sys.path.insert(0, str(source))
    from videoprism import models as vp

    model = vp.get_model("videoprism_public_v1_base")
    variables = jax.device_put(
        vp.load_pretrained_weights("videoprism_public_v1_base", checkpoint_path=str(checkpoint))
    )
    encoder = SharedVisualEncoder(model, variables, compile_fn=jax.jit)
    results = {}
    for modality, frames in (("image", 1), ("video", 8)):
        start = time.perf_counter()
        values = encoder.encode(np.zeros((1, frames, 288, 288, 3), np.float32), modality=modality)
        assert np.isfinite(values).all()
        results[modality] = dict(shape=list(values.shape), seconds=time.perf_counter() - start)
    leaves = jax.tree_util.tree_leaves(encoder.variables)
    write(
        out / "visual-ready.json",
        dict(
            pid=os.getpid(),
            parameters=sum(v.size for v in leaves),
            tensor_bytes=sum(v.size * v.dtype.itemsize for v in leaves),
            checkpoint_sha256=CHECKPOINT_SHA256,
            source=SOURCE_COMMIT,
            memory=psutil.Process().memory_info()._asdict(),
            forwards=results,
        ),
    )
    # Parent retains both workers until its complete inventory is recorded.
    if sys.stdin.readline().strip() != "release":
        raise RuntimeError("missing parent release")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-worker", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.visual_worker:
        visual_worker(args.output)
        return
    import psutil
    import torch
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        WhisperForConditionalGeneration,
    )

    from scripts.extract_starss23_features import AUDIO_SHA, AUDIO_SOURCE
    from scripts.range_generalization import sha, write
    from tiny_omni_decision.resident_inventory import parameter_inventory

    out = args.output
    out.mkdir(exist_ok=False)
    r1 = Path("C:/CodexArtifacts/pretrained-reuse-r1")
    r2 = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    inventory = json.loads(
        Path("docs/pretrained_reuse/R1_EXISTING_CONFIG_RECONCILIATION_20261012.json").read_text(
            encoding="utf-8"
        )
    )["models"]
    models, inputs, forwards = {}, {}, {}
    torch.set_num_threads(2)
    for name, loader in (
        ("minilm", AutoModel),
        ("crossencoder", AutoModelForSequenceClassification),
        ("whisper", WhisperForConditionalGeneration),
    ):
        path = Path(inventory[name]["path"])
        digest = sha(path / "model.safetensors")
        assert digest == inventory[name]["weights_sha256"]
        inputs[str(path / "model.safetensors")] = digest
        models[name] = loader.from_pretrained(path, local_files_only=True).cpu().eval()
    audio_source = r1 / "efficientat-audit-v1/source"
    audio_weights = audio_source.parent / "dymn04_as.pt"
    assert sha(audio_weights) == AUDIO_SHA
    assert (
        subprocess.check_output(
            ["git", "-C", str(audio_source), "rev-parse", "HEAD"], text=True
        ).strip()
        == AUDIO_SOURCE
    )
    inputs[str(audio_weights)] = AUDIO_SHA
    sys.path.insert(0, str(audio_source))
    from models.dymn.model import get_model

    models["audio_events"] = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0).eval()
    models["audio_events"].load_state_dict(torch.load(audio_weights, weights_only=True))
    # Retain all head tensors conservatively, including any buffers and unused values.
    # This is an upper bound on these checkpoint tensors, not a claim of wired heads.
    head_paths = {
        "image": r2 / "shared-affine-regularized-01/0.01/final.pt",
        "video": r2 / "video-token-readout-01/attention/best.pt",
        "sensor": r2 / "sensor-subject-training-01/static/best.pt",
        "fusion": r2 / "vidimu-gated-fusion-budget-01/best.pt",
        "numeric_delta": r2 / "numeric-deterministic-control-01/best-delta.pt",
        "audio_readout": r2 / "path-a-audio-multiquery-runtime-bundle-v2/readout/best-readout.pt",
    }
    states = {}
    for name, path in head_paths.items():
        inputs[str(path)] = sha(path)
        states[name] = torch.load(path, weights_only=True, map_location="cpu")
        if not all(isinstance(v, torch.Tensor) for v in states[name].values()):
            raise ValueError("head checkpoint is not a flat tensor mapping")
    replacement = states.pop("numeric_delta")
    current = models["crossencoder"].state_dict()
    if not all(k in current and current[k].shape == v.shape for k, v in replacement.items()):
        raise ValueError("replacement keys must match existing model tensors")
    replaced_elements = sum(v.numel() for v in replacement.values())
    loaded = models["crossencoder"].load_state_dict(replacement, strict=False)
    assert not loaded.unexpected_keys
    assert all(torch.equal(models["crossencoder"].state_dict()[k], v)
               for k, v in replacement.items())
    del replacement, current
    with torch.inference_mode():
        for name in ("minilm", "crossencoder"):
            output = models[name](
                input_ids=torch.tensor([[101, 102]]),
                attention_mask=torch.ones(1, 2, dtype=torch.long),
            )
            values = output.last_hidden_state if name == "minilm" else output.logits
            assert torch.isfinite(values).all()
            forwards[name] = list(values.shape)
        output = models["whisper"](
            input_features=torch.zeros(1, 80, 3000), decoder_input_ids=torch.tensor([[50258]])
        ).logits
        assert torch.isfinite(output).all()
        forwards["whisper"] = list(output.shape)
        output, _ = models["audio_events"](torch.zeros(1, 1, 128, 1000))
        assert torch.isfinite(output).all()
        forwards["audio_events"] = list(output.shape)
    env = dict(os.environ, PYTHONPATH=str(Path.cwd() / "src"), JAX_PLATFORMS="cpu")
    python = r1 / "videoprism-base-r1/venv/Scripts/python.exe"
    with (out / "visual-worker.log").open("w", encoding="utf-8") as log:
        worker = subprocess.Popen(
            [
                str(python),
                "-m",
                "scripts.probe_omni_resident_budget",
                "--visual-worker",
                "--output",
                str(out),
            ],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=log,
            text=True,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            started = time.monotonic()
            while not (out / "visual-ready.json").exists():
                if worker.poll() is not None:
                    raise RuntimeError("visual worker failed; inspect log")
                if time.monotonic() - started > 180:
                    raise TimeoutError("visual worker observation deadline")
                time.sleep(0.1)
            visual = json.loads((out / "visual-ready.json").read_text())
            assert worker.poll() is None
            process_tree = [psutil.Process(worker.pid),
                            *psutil.Process(worker.pid).children(recursive=True)]
            if visual["pid"] not in {p.pid for p in process_tree}:
                raise ValueError("reported visual process is not our live worker")
            visual_process = psutil.Process(visual["pid"])
            assert "scripts.probe_omni_resident_budget" in visual_process.cmdline()
            heads = {
                k: dict(
                    elements=sum(t.numel() for t in v.values()),
                    bytes=sum(t.numel() * t.element_size() for t in v.values()),
                )
                for k, v in states.items()
            }
            counted = parameter_inventory(models)
            total = (
                counted["total_parameters"]
                + visual["parameters"]
                + sum(v["elements"] for v in heads.values())
            )
            memory = {
                "torch": psutil.Process().memory_info()._asdict(),
                "visual": visual_process.memory_info()._asdict(),
            }
            if worker.pid != visual["pid"]:
                memory["launcher"] = psutil.Process(worker.pid).memory_info()._asdict()
            assert all(sha(Path(p)) == h for p, h in inputs.items())
            write(
                out / "result.json",
                dict(
                    status="PASS_RESIDENCY_SMOKE",
                    torch_models=counted,
                    checkpoint_head_upper_bounds=heads,
                    combined_parameter_upper_bound=total,
                    remaining_to_200M=200000000 - total,
                    workers_live_at_snapshot=True,
                    replacement_elements_not_added=replaced_elements,
                    replacement_applied_in_memory_only=True,
                    visual_process_pid=visual["pid"],
                    launcher_pid=worker.pid,
                    memory_snapshot=memory,
                    summed_rss=sum(v["rss"] for v in memory.values()),
                    visual=visual,
                    synthetic_forwards=forwards,
                    inputs=inputs,
                    source_commit=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    runner_sha256=sha(Path(__file__)),
                    complete_omni_runtime=False,
                    quality_evaluation=False,
                    sealed_audit=False,
                ),
            )
        finally:
            if worker.poll() is None:
                worker.stdin.write("release\n")
                worker.stdin.flush()
            worker.wait(timeout=30)


if __name__ == "__main__":
    main()
