"""Frozen train-only question/state interventions; no fitting or selection."""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from scripts.audit_correspondence_validation import sha
from scripts.cover_point_metrics import state_hash
from scripts.extract_spatial_context_corpus import load_plans, require_bitwise
from scripts.report_spatial_trajectory_readout import write_publication
from scripts.spatial_readout_inputs import SPATIAL_RESULT_PIN, load_spatial_bank
from scripts.train_role_supervised_qa import load, qa_summary
from scripts.train_spatial_trajectory_readout import (
    paired_batch,
    process_peak_bytes,
    require_limits,
)
from scripts.train_trajectory_readout import ROOT
from tiny_omni_decision.candidate_condition_controls import question_control
from tiny_omni_decision.spatial_trajectory_readout import SpatialTrajectoryReadout

RUN_PIN = "cdb3b521519af36d3394fcd6e5a620f1dd0312da193223d71371075704ab019b"


def verify_original_train(path, expected_hash, values):
    if sha(path) != expected_hash:
        raise ValueError("original epoch archive hash changed")
    with np.load(path, allow_pickle=False) as original:
        for suffix, replay in zip(("", "_cover", "_logp"), values, strict=True):
            require_bitwise(original["train" + suffix], replay)


def predict_control(model, bank, spatial, data, control, limits):
    qa, cover, logp, raw = [], [], [], []
    route = [p["video_id"] for p in data["pairs"]]
    if control == "mismatched_state":
        bank = {v: bank[route[(i + 1) % len(route)]] for i, v in enumerate(route)}
        spatial = {v: spatial[route[(i + 1) % len(route)]] for i, v in enumerate(route)}
    with torch.inference_mode():
        for offset in range(0, len(route), 8):
            require_limits(*limits)
            points, candidates, context, _ = paired_batch(
                bank,
                spatial,
                data,
                range(offset, min(offset + 8, len(route))),
                "temporal_appearance",
                "cuda",
            )
            mode = "normal" if control == "mismatched_state" else control
            candidates = question_control(candidates.reshape(-1, 2, 3, 1536), mode).reshape(
                -1, 6, 1536
            )
            state = model.encode_state(points, context)
            logits = model.score_candidates(state, candidates).reshape(-1, 3)
            qa.append(logits.softmax(-1).cpu().numpy())
            cover.append(state[1].sigmoid().cpu().numpy())
            logp.append(logits.log_softmax(-1).cpu().numpy())
            raw.append(logits.cpu().numpy())
            require_limits(*limits)
    return np.concatenate(qa), np.concatenate(cover), np.concatenate(logp), np.concatenate(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    start = time.perf_counter()
    report = dict(
        status="PREFLIGHT",
        training=False,
        diagnostic_split="train",
        sealed_audit_opened=False,
        cloud_cost=0,
        checkpoints={},
    )
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
            raise MemoryError("GPU must be free for isolated diagnosis")
        run_dir = ROOT / "spatial-readout-full-01"
        if sha(run_dir / "result.json") != RUN_PIN:
            raise ValueError("completed reference run changed")
        run = json.loads((run_dir / "result.json").read_text())
        config = json.loads((run_dir / "config.json").read_text())
        if sha(run_dir / "config.json") != run["config_sha256"]:
            raise ValueError("reference config changed")
        for path, pin in config["code_hashes"].items():
            if sha(Path(path)) != pin:
                raise ValueError("reference executable changed")
        _, plans = load_plans()
        spatial, _ = load_spatial_bank(
            ROOT / "spatial-context-corpus-01/result.json", plans, SPATIAL_RESULT_PIN
        )
        bank, splits, _, _ = load()
        data = splits["train"]
        del splits
        if len(data["pairs"]) != 99 or len(data["targets"]) != 198:
            raise ValueError("train diagnosis denominator changed")
        text = data["text"].reshape(99, 2, 3, 1536)
        question_control(text, "normal")
        q = text[..., :384].numpy()
        u = q[0, :, 0]
        if len(np.unique(q.reshape(-1, 384), axis=0)) != 2 or not np.array_equal(
            q, np.broadcast_to(q[:1, :, :1], q.shape)
        ):
            raise ValueError("exact two declared question vectors required")
        limits = (start, dict(max_seconds=600, peak_cap_bytes=4 * 1024**3))
        report.update(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            runner_sha256=sha(Path(__file__)),
            helper_sha256=sha(Path("src/tiny_omni_decision/candidate_condition_controls.py")),
            reference_result_sha256=RUN_PIN,
            reference_config_sha256=run["config_sha256"],
            train_text_sha256=config["input_hashes"][
                str(ROOT / "perception-occlusion-text-features-01/train.npy")
            ],
            question_feature_geometry=dict(
                distinct_vectors=2,
                norms=np.linalg.norm(u, axis=-1).tolist(),
                cosine=float(np.dot(*u) / np.prod(np.linalg.norm(u, axis=-1))),
                l2_difference=float(np.linalg.norm(u[0] - u[1])),
                factorization_bitwise=True,
                paired_option_maximum_difference=float(
                    (text[:, 0, :, 384:768] - text[:, 1, :, 384:768]).abs().max()
                ),
                paired_option_rounding_videos=int(
                    ((text[:, 0, :, 384:768] - text[:, 1, :, 384:768]).abs().amax((1, 2)) > 0).sum()
                ),
            ),
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            precision="FP32",
            tf32=False,
            observed_development_evaluated=False,
            common_loader_validates_existing_development_cache=True,
        )
        diagnostic_config = dict(
            command=sys.argv,
            source_commit=report["source_commit"],
            runner_sha256=report["runner_sha256"],
            helper_sha256=report["helper_sha256"],
            protocol_sha256=sha(
                Path("docs/pretrained_reuse/R2_CONDITION_BINDING_PROTOCOL_20261014.md")
            ),
            reference_result_sha256=RUN_PIN,
            reference_config_sha256=run["config_sha256"],
            reference_input_hashes=config["input_hashes"],
            controls=["normal", "neutral_question", "swapped_question", "mismatched_state"],
            diagnostic_split="train",
            videos=99,
            questions=198,
            options=3,
            checkpoints={
                k: dict(epoch=e, sha256=run["arms"]["temporal_appearance"]["hashes"][k + ".pt"])
                for k, e in (("best", 6), ("final", 50))
            },
            limits=limits[1],
            batch_videos=8,
            hidden=64,
            latent_count=16,
            heads=4,
            frozen=True,
            training=False,
            checkpoint_selection=False,
            precision="FP32",
            tf32=False,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            python=platform.python_version(),
            gpu=report["gpu"],
            swapped_equivariance_tolerance=dict(rtol=1e-5, atol=1e-6),
            paired_option_tolerance=dict(rtol=1e-5, atol=1e-6),
            paired_option_values_preserved=True,
            original_epoch_replay="shape/dtype/bytes exact",
            sealed_audit_opened=False,
        )
        write_publication(args.output / "config.json", diagnostic_config)
        report["effective_config_sha256"] = sha(args.output / "config.json")
        np.savez_compressed(
            args.output / "train-identities-targets.npz",
            identities=np.array([p["video_id"] for p in data["pairs"]]),
            targets=data["targets"].numpy(),
        )
        folder = run_dir / "temporal_appearance"
        metadata = run["arms"]["temporal_appearance"]
        for kind, epoch in (("best", 6), ("final", 50)):
            path = folder / f"{kind}.pt"
            if sha(path) != metadata["hashes"][path.name]:
                raise ValueError("checkpoint changed")
            model = SpatialTrajectoryReadout().cuda().eval().requires_grad_(False)
            model.load_state_dict(torch.load(path, weights_only=True), strict=True)
            before = state_hash(model)
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            probabilities, metrics, saved_arrays = {}, {}, {}
            for control in ("normal", "neutral_question", "swapped_question", "mismatched_state"):
                p, c, logp, logits = predict_control(model, bank, spatial, data, control, limits)
                if not all(np.isfinite(v).all() for v in (p, c, logp, logits)):
                    raise ValueError("nonfinite diagnostic output")
                probabilities[control] = p
                saved_arrays[control], saved_arrays[control + "_cover"] = p, c
                saved_arrays[control + "_logp"] = logp
                saved_arrays[control + "_logits"] = logits
                measurements = qa_summary(p, data["targets"].numpy(), logp)
                pred = p.argmax(-1).reshape(-1, 2)
                y = data["targets"].numpy().reshape(-1, 2)
                changed = y[:, 0] != y[:, 1]
                measurements.update(
                    different_condition_predictions=int((pred[:, 0] != pred[:, 1]).sum()),
                    changed_target_pairs=int(changed.sum()),
                    changed_target_both_correct=int((pred[changed] == y[changed]).all(1).sum()),
                    argmax_counts=np.bincount(pred.ravel(), minlength=3).tolist(),
                )
                if control != "normal":
                    measurements["answer_changes_from_normal"] = int(
                        (pred.ravel() != probabilities["normal"].argmax(-1)).sum()
                    )
                    measurements["maximum_probability_change"] = float(
                        np.max(np.abs(p - probabilities["normal"]))
                    )
                    difference = logits.astype(np.float64) - saved_arrays["normal_logits"]
                    measurements["maximum_logit_change"] = float(np.abs(difference).max())
                    measurements["maximum_centered_logit_change"] = float(
                        np.abs(difference - difference.mean(-1, keepdims=True)).max()
                    )
                metrics[control] = measurements
            np.savez_compressed(args.output / f"{kind}-predictions.npz", **saved_arrays)
            report["pending_checkpoint_checks"] = dict(kind=kind, epoch=epoch, metrics=metrics)
            write_publication(args.output / "progress.json", report)
            original_path = folder / f"epoch-{epoch:02d}-predictions.npz"
            verify_original_train(
                original_path,
                metadata["hashes"][original_path.name],
                tuple(saved_arrays["normal" + s] for s in ("", "_cover", "_logp")),
            )
            swapped_expected = probabilities["normal"].reshape(-1, 2, 3)[:, ::-1].reshape(-1, 3)
            np.testing.assert_allclose(
                probabilities["swapped_question"], swapped_expected, rtol=1e-5, atol=1e-6
            )
            np.testing.assert_allclose(
                saved_arrays["swapped_question_logits"],
                saved_arrays["normal_logits"].reshape(-1, 2, 3)[:, ::-1].reshape(-1, 3),
                rtol=1e-5,
                atol=1e-6,
            )
            neutral = probabilities["neutral_question"].reshape(-1, 2, 3)
            np.testing.assert_allclose(neutral[:, 0], neutral[:, 1], rtol=1e-5, atol=1e-6)
            assert metrics["neutral_question"]["different_condition_predictions"] == 0
            assert metrics["neutral_question"]["changed_target_both_correct"] == 0
            after = state_hash(model)
            if before != after or any(
                p.requires_grad or p.grad is not None for p in model.parameters()
            ):
                raise ValueError("frozen readout changed")
            if not torch.equal(cpu_rng, torch.get_rng_state()) or not all(
                torch.equal(a, b)
                for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)
            ):
                raise ValueError("inference changed RNG")
            report["checkpoints"][kind] = dict(
                epoch=epoch,
                weights_sha256=sha(path),
                state_before=before,
                state_after=after,
                rng_unchanged=True,
                original_epoch_train_replay_bitwise=True,
                original_epoch_archive_sha256=sha(original_path),
                swapped_question_equivariant=True,
                swapped_question_logits_equivariant=True,
                metrics=metrics,
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=0,
                prediction_sha256=sha(args.output / f"{kind}-predictions.npz"),
            )
            del model
            torch.cuda.empty_cache()
        report.pop("pending_checkpoint_checks", None)
        for path, pin in config["input_hashes"].items():
            require_limits(*limits)
            if sha(Path(path)) != pin:
                raise ValueError("protected input changed")
        require_limits(*limits)
        report.update(
            status="PASS_FIXED_TRAIN_ONLY_CONDITION_STATE_DIAGNOSIS_NOT_QUALITY",
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_process_working_set_bytes=process_peak_bytes(),
            identities_targets_sha256=sha(args.output / "train-identities-targets.npz"),
        )
    except Exception:
        report.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        report["seconds"] = time.perf_counter() - start
        write_publication(args.output / "result.json", report)


if __name__ == "__main__":
    main()
