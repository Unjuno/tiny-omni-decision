"""Replay trusted native feature bindings on eight training requests per modality.

No encoder execution, fitting, calibration or evaluation-set scoring. The
legacy feature containers and readouts are hash-pinned and opened read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from scripts.extract_frozen_videoprism_features import uniform_indices
from scripts.extract_perception_temporal_match_efficientat import sha256_file, state_sha
from tiny_omni_decision.cache import ObservationFeatureCache, ObservationFeatureKey
from tiny_omni_decision.native_feature_binding import NativeFeatureBinding, decide_bound_native
from tiny_omni_decision.observations import Observation, ObservationBatch
from tiny_omni_decision.textfree_audio import TextFreeAudioReadout
from tiny_omni_decision.textfree_image import TextFreeImageReadout
from tiny_omni_decision.textfree_video import TextFreeVideoReadout, build_video_vocab

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
PINNED = {
    "audio-probe-v0-retry3/audio-features.pt": (
        "ace323458e7f443daf63382d2d41d99151d2a5801c671d30b0df68d52338ae64"
    ),
    "speech-commands-sample-seed17-128-64-retry2/train.jsonl": (
        "02a43981e153b0f63bd55192f614fe504114f05f212ca0f4da37e2d91480a9da"
    ),
    "textfree-audio-decision-03/best-readout.pt": (
        "46808a882ab75f8f19a138e4a91d13ceb42cfdc82534c4608c42ee615774d3cb"
    ),
    "textfree-audio-decision-03/run-report.json": (
        "a8452995e3d8d55f5a580177f9fd5251b392642ca7fe98396a70900b2d46ddcc"
    ),
    "clevr4-full-videoprism-features-v0/features.npz": (
        "f36b9cf17efed6a5137eb48ffcaf40ff960015fc07bc194fb532bf5562d88fab"
    ),
    "clevr4-full-videoprism-features-v0/cache-report.json": (
        "f8789fed749c8e19741758d52af2468b74598148dc673174ba65fb86181d5467"
    ),
    "clevr4-full-coverage-seed17/train-image-manifest.jsonl": (
        "2c56815beb9d2e9951cd38686eafdbe2ecdee05e69e74b5acfad24fbefbb9010"
    ),
    "textfree-image-decision-512-01/best-readout.pt": (
        "e62711ca1b8fd68df5037cd135266ed7f556e0ad006359305f10e6665789a47d"
    ),
    "textfree-image-decision-512-01/run-report.json": (
        "f2b357b00bf2ce11cc3c1b6f75b64aa8ea1a9c132908601c1fafc2e98fe80b39"
    ),
    "clevrer-framewise-typed-v0/framewise-features.npz": (
        "8878e2a51c9fb617466ec2e5b6dc174e9e7fb6eee666b6bee89b4071c2f00350"
    ),
    "clevrer-framewise-typed-v0/cache-report.json": (
        "fa8af092624ac10de46d4e9473537784e5717927c7477957c1c4702ad32a28e9"
    ),
    "clevrer-expanded-seed17-256train-200val/train.jsonl": (
        "5cd4e5155b0161e35b10aad2c98e14afba7d46407e635bd1c08d786e1e51765d"
    ),
    "textfree-video-clevrer-typed-fit-01/best-bilstm.pt": (
        "1b4f8a6232e8c08cf1a1a5f1525b47a09d5c45a70a7c459e219b36ad4429128e"
    ),
}


def read_json(name):
    return json.loads((ROOT / name).read_bytes())


def read_rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines()]


def video_times(path):
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        declared_frames = int(stream.frames or 0)
        pts = [float(frame.pts * frame.time_base * 1000) for frame in container.decode(stream)]
    if declared_frames and declared_frames != len(pts):
        raise ValueError("video PTS reconstruction differs from legacy decoder frame-count policy")
    return tuple(pts[i] for i in uniform_indices(len(pts)))


def fixtures():
    audio = torch.load(
        ROOT / "audio-probe-v0-retry3/audio-features.pt", map_location="cpu", weights_only=True
    )
    report = read_json("textfree-audio-decision-03/run-report.json")
    model = TextFreeAudioReadout(384, report["class_ids"]).eval()
    model.load_state_dict(
        torch.load(
            ROOT / "textfree-audio-decision-03/best-readout.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    sample = ROOT / "speech-commands-sample-seed17-128-64-retry2"
    for index, row in enumerate(read_rows(str(sample.relative_to(ROOT) / "train.jsonl"))[:8]):
        if audio["train_ids"][index] != row["id"]:
            raise ValueError("audio training ID order differs from media manifest")
        yield (
            "audio",
            row["id"],
            model,
            audio["train_features"][index].numpy(),
            sample / row["media_path"],
            row["media_sha256"],
            row["source"],
            "openai/whisper-tiny",
            audio["encoder_revision"],
            "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395",
            PINNED["audio-probe-v0-retry3/audio-features.pt"],
            report["class_ids"],
            (0.0,),
            {},
        )

    report = read_json("textfree-image-decision-512-01/run-report.json")
    classes = report["effective_config"]["taxonomies"]
    model = TextFreeImageReadout(768, classes).eval()
    model.load_state_dict(
        torch.load(
            ROOT / "textfree-image-decision-512-01/best-readout.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    rows = read_rows("clevr4-full-coverage-seed17/train-image-manifest.jsonl")[:8]
    with np.load(
        ROOT / "clevr4-full-videoprism-features-v0/features.npz", allow_pickle=False
    ) as data:
        ids, hashes, values = (
            data["train_ids"][:8],
            data["train_media_sha256"][:8],
            data["train_features"][:8],
        )
    encoder = read_json("clevr4-full-videoprism-features-v0/cache-report.json")["encoder"]
    for row, sample_id, media_sha, vector in zip(rows, ids, hashes, values, strict=True):
        if row["id"] != sample_id or row["sha256"] != media_sha:
            raise ValueError("image identity differs between source and feature manifest")
        yield (
            "image",
            str(sample_id),
            model,
            vector,
            ROOT / "clevr4-full-coverage-seed17/train/images" / f"{sample_id}.png",
            str(media_sha),
            report["source_revision"],
            encoder["model_id"],
            encoder["model_revision"],
            encoder["checkpoint_sha256"],
            PINNED["clevr4-full-videoprism-features-v0/features.npz"],
            classes["color"],
            (0.0,),
            {"taxonomy": "color"},
        )

    rows = read_rows("clevrer-expanded-seed17-256train-200val/train.jsonl")
    model = TextFreeVideoReadout(build_video_vocab(rows), temporal_mode="bilstm").eval()
    model.load_state_dict(
        torch.load(
            ROOT / "textfree-video-clevrer-typed-fit-01/best-bilstm.pt",
            map_location="cpu",
            weights_only=True,
        )["state_dict"]
    )
    with np.load(
        ROOT / "clevrer-framewise-typed-v0/framewise-features.npz", allow_pickle=False
    ) as data:
        values = dict(zip(data["media_sha256"].tolist(), data["features"], strict=True))
    encoder = read_json("clevrer-framewise-typed-v0/cache-report.json")
    times = {}
    for row in rows[:8]:
        media = ROOT / "clevrer-expanded-seed17-256train-200val" / row["media_path"]
        if row["media_sha256"] not in times:
            times[row["media_sha256"]] = video_times(media)
        yield (
            "video",
            row["id"],
            model,
            values[row["media_sha256"]],
            media,
            row["media_sha256"],
            row["source"],
            encoder["model_id"],
            encoder["model_revision"],
            encoder["checkpoint_sha256"],
            PINNED["clevrer-framewise-typed-v0/framewise-features.npz"],
            row["options"],
            times[row["media_sha256"]],
            {"taxonomy": row["taxonomy"], "program_tokens": row["program"]},
        )


def run(output):
    if output.exists():
        raise FileExistsError("choose a fresh native binding replay output")
    started = time.perf_counter()
    for relative, expected in PINNED.items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"legacy input hash mismatch: {relative}")
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cache = ObservationFeatureCache(output / "cache")
    records, bindings = [], []
    states = {}
    for (
        modality,
        sample_id,
        model,
        values,
        media,
        media_sha,
        source,
        encoder,
        revision,
        weights,
        legacy_cache_sha,
        classes,
        times,
        kwargs,
    ) in fixtures():
        identity = (modality, id(model))
        states.setdefault(identity, state_sha(model))
        source_id, _, source_revision = source.rpartition("@")
        import_contract = {
            "legacy_cache_sha256": legacy_cache_sha,
            "policy": "read-only train-row import; no re-encoding",
            "script_sha256": sha256_file(Path(__file__)),
            "timestamps_ms": list(times),
        }
        contract_sha = hashlib.sha256(
            json.dumps(import_contract, sort_keys=True).encode()
        ).hexdigest()
        key = ObservationFeatureKey(
            modality=modality,
            source_id=source_id,
            source_revision=source_revision,
            observation_sha256=media_sha,
            encoder_id=encoder,
            encoder_revision=revision,
            encoder_weights_sha256=weights,
            preprocessor_id="legacy-frozen-cache-import",
            preprocessor_revision=legacy_cache_sha,
            preprocessing_sha256=contract_sha,
            feature_name="train_feature_import",
            feature_dtype="float32",
            feature_shape=tuple(values.shape),
            temporal_policy=json.dumps(import_contract, sort_keys=True),
        )
        entry = cache.put(key, values.astype("<f4", copy=False).tobytes())
        binding = NativeFeatureBinding(key, entry.payload_sha256, sample_id, "local-clip", times)
        observation = Observation(
            modality=modality,
            source_id=sample_id,
            payload_ref=str(media),
            payload_sha256=media_sha,
            clock_id="local-clip",
            timestamps_ms=times,
            valid_mask=(True,) * len(times),
        )
        batch = ObservationBatch(
            schema_version="1.0",
            observations=(observation,),
            options=tuple(
                {"option_id": f"class-{name}", "attributes": {"class_id": name}} for name in classes
            ),
        )
        tensor = torch.from_numpy(values.copy())
        if modality == "image":
            direct = model.decide_native(tensor, kwargs["taxonomy"], batch)
        else:
            direct = model.decide_native(tensor, batch, **kwargs)
        tick = time.perf_counter()
        bound = decide_bound_native(model, batch, binding, cache, **kwargs)
        elapsed = time.perf_counter() - tick
        provenance = bound.pop("feature_provenance")
        if direct != bound:
            raise ValueError("bound JSON differs from direct legacy-feature JSON")
        reverse = batch.model_copy(update={"options": tuple(reversed(batch.options))})
        reversed_result = decide_bound_native(model, reverse, binding, cache, **kwargs)
        normal_probs = {o["option_id"]: o["probability"] for o in bound["options"]}
        reverse_probs = {o["option_id"]: o["probability"] for o in reversed_result["options"]}
        delta = max(abs(normal_probs[k] - reverse_probs[k]) for k in normal_probs)
        if delta > 1e-6 or bound["selected_option_id"] != reversed_result["selected_option_id"]:
            raise ValueError("candidate reordering changed native decision")
        if state_sha(model) != states[identity]:
            raise ValueError("readout state changed during inference")
        rejected_controls = []
        for name, control_binding, control_batch in (
            ("wrong_feature_digest", replace(binding, feature_payload_sha256="f" * 64), batch),
            (
                "wrong_clock",
                binding,
                batch.model_copy(
                    update={
                        "observations": (
                            observation.model_copy(update={"clock_id": "wrong-clock"}),
                        )
                    }
                ),
            ),
        ):
            try:
                decide_bound_native(model, control_batch, control_binding, cache, **kwargs)
            except ValueError:
                rejected_controls.append(name)
            else:
                raise ValueError(f"invalid native binding was accepted: {name}")
        bindings.append(asdict(binding))
        records.append(
            {
                "modality": modality,
                "sample_id": sample_id,
                "media_path": str(media),
                "direct_bound_exact_json": True,
                "option_reorder_max_probability_delta": delta,
                "bound_decision_seconds": elapsed,
                "provenance": provenance,
                "decision": bound,
                "rejected_controls": rejected_controls,
            }
        )
        print("PASS", modality, sample_id, flush=True)
    for relative, expected in PINNED.items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"legacy input changed during replay: {relative}")
    for record in records:
        if sha256_file(Path(record["media_path"])) != record["provenance"]["media_sha256"]:
            raise ValueError("selected media changed during replay")
    receipt = {
        "status": "PASS_NATIVE_FEATURE_BINDING_REPLAY",
        "finished_at": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha256_file(Path(__file__)),
        "binding_module_sha256": sha256_file(
            Path(__file__).parents[1] / "src/tiny_omni_decision/native_feature_binding.py"
        ),
        "input_hashes": PINNED,
        "device": "cpu",
        "threads": 4,
        "cuda_initialized": torch.cuda.is_initialized(),
        "wall_seconds": time.perf_counter() - started,
        "training": False,
        "quality_evaluated": False,
        "sealed_audit_loaded": False,
        "encoder_executed": False,
        "legacy_artifacts_modified": False,
        "all_legacy_input_hashes_rechecked_after": True,
        "scope": (
            "legacy extraction trust; current media and cache byte integrity; "
            "no encoder attestation"
        ),
        "by_modality": {
            m: {
                "requests": sum(r["modality"] == m for r in records),
                "median_bound_decision_ms": statistics.median(
                    r["bound_decision_seconds"] * 1000 for r in records if r["modality"] == m
                ),
            }
            for m in ("audio", "image", "video")
        },
        "records": records,
    }
    for name, value in (("result.json", receipt), ("bindings.json", bindings)):
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    print(receipt["status"], receipt["by_modality"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
