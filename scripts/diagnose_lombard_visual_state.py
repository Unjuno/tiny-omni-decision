"""Fixed selected models, train-only constants, CPU inference; no training/selection."""

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from scripts.lombard_readout_core import fit_normalizer
from scripts.lombard_readout_data import read_json, sha
from scripts.lombard_temporal_full import verify_snapshot
from scripts.lombard_temporal_model import TemporalSlotHead
from scripts.lombard_visual_state import (
    latent_controls,
    latent_states,
    readout_from_latent,
    summary,
)
from scripts.report_lombard_grid_readouts import paired_bootstrap
from scripts.robomnist_audio_cue import recheck
from scripts.train_lombard_temporal import (
    ROOT,
    bound_sources,
    guard,
    predict,
    source_scope,
    state_sha,
    store_prediction,
    tensors,
)

TRAINING = ROOT / "lombard-temporal-full-01"
EVALUATION = ROOT / "lombard-temporal-evaluation-01"
FIT_SHA = "09ef033c461313a84d34a7db5b6a754c57619871ba1ea9e7f9bdb746e830ec0e"
AUDIT_SHA = "c97d975df27f4931be0b256e5a0ed90f476b2ba7dbbc7534085b7667f3694fdc"
ARMS = ["joint_broadcast", "joint_ordered", "video_ordered"]
EXTRA = [
    "scripts/diagnose_lombard_visual_state.py",
    "scripts/lombard_visual_state.py",
    "tests/test_lombard_visual_state.py",
    "docs/pretrained_reuse/R2_LOMBARD_VISUAL_STATE_PROTOCOL_20261015.md",
]


def bounded(output, tick):
    state = guard(output, tick)
    if state["seconds"] > 600 or state["output_bytes"] + 2**22 > 64 * 2**20:
        raise RuntimeError("fixed visual-state diagnostic safety limit exceeded")
    return state


def answers(logits):
    return np.stack([z.argmax(1) for z in np.split(logits, [4, 8, 12, 37, 47], 1)], 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-reference", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if bool(args.replay_reference) != bool(args.replay_sha256):
        raise ValueError("replay reference and SHA must both be supplied")
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    result = dict(
        status="RUNNING",
        arms={},
        model_updates=0,
        gradients_computed=False,
        gpu_used=False,
        encoders_loaded=False,
        sealed_audit_opened=False,
        cloud_cost=0,
        python=sys.version,
        python_executable=sys.executable,
        resource_bounds=dict(
            seconds=600, peak_cpu_bytes=3 * 2**30, output_bytes=64 * 2**20, disk_reserve_bytes=2**30
        ),
        started_utc=datetime.now(UTC).isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        command=subprocess.list2cmdline(
            [sys.executable, "-m", "scripts.diagnose_lombard_visual_state", *sys.argv[1:]]
        ),
        scope="Fixed selected model diagnosis on observed development; not blind audit",
    )
    try:
        bounded(args.output, tick)
        fit = read_json(TRAINING / "result.json", FIT_SHA)
        audit = read_json(EVALUATION / "result.json", AUDIT_SHA)
        if (
            fit["status"] != "PASS_FIVE_ARM_TRAINING"
            or audit["status"] != "PASS_FRESH_REPLAY_CONTROLS_AND_GROUPED_COMPARISON"
        ):
            raise ValueError("verified fixed comparison required")
        pins = dict(fit["validated_hashes"])
        if bound_sources() != pins:
            raise ValueError("frozen source/runtime differs")
        pins.update(fit["full_execution_hashes"])
        pins.update({str(Path(p).resolve()): sha(Path(p)) for p in EXTRA})
        pins[str((TRAINING / "result.json").resolve())] = FIT_SHA
        pins[str((EVALUATION / "result.json").resolve())] = AUDIT_SHA
        pins[str((TRAINING / "effective-config.json").resolve())] = fit["effective_config_sha256"]
        for arm in ARMS:
            folder = TRAINING / arm
            pins.update(
                {
                    str((folder / k).resolve()): h
                    for k, h in fit["arms"][arm]["artifact_hashes"].items()
                }
            )
            pins[str((folder / "result.json").resolve())] = sha(folder / "result.json")
            if (
                read_json(folder / "result.json", pins[str((folder / "result.json").resolve())])
                != fit["arms"][arm]
            ):
                raise ValueError("fixed arm/controller report differs")
        recheck(pins)
        reference = None
        if args.replay_reference:
            reference = read_json(args.replay_reference / "result.json", args.replay_sha256)
            if (
                reference["status"] != "PASS_FIXED_VISUAL_STATE_DIAGNOSIS"
                or reference["validated_hashes"] != pins
            ):
                raise ValueError("unchanged complete primary diagnostic required")
            recheck(
                {
                    str((args.replay_reference / k).resolve()): v
                    for k, v in reference["artifact_hashes"].items()
                }
            )
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        raw, framed, targets, rows, provenance = source_scope(False)
        if provenance != fit["provenance"]:
            raise ValueError("source/cache identity authority differs")
        result.update(
            validated_hashes=pins,
            provenance=provenance,
            device="cpu",
            dtype="float32",
            torch=torch.__version__,
            numpy=np.__version__,
            cpu_threads=4,
            interop_threads=1,
            deterministic=True,
            training_constant_source_id=rows["train"][0]["source_id"],
        )
        for arm in ARMS:
            bounded(args.output, tick)
            trained = fit["arms"][arm]
            normalizer = fit_normalizer(*raw["train"])
            with np.load(TRAINING / arm / "normalizer.npz", allow_pickle=False) as saved:
                if any(not np.array_equal(saved[k], v) for k, v in normalizer.items()):
                    raise ValueError("original training-only normalization differs")
            values = tensors(raw, framed, normalizer, arm)
            model = TemporalSlotHead().eval()
            model.load_state_dict(
                torch.load(TRAINING / arm / "best.pt", weights_only=True, map_location="cpu"),
                strict=True,
            )
            before = state_sha(model)
            if sum(p.numel() for p in model.parameters()) != 106355:
                raise ValueError("fixed model dimensions differ")
            a, v, present = values["validation"]
            train_h = latent_states(model, values["train"][1])
            val_h = latent_states(model, v)
            original = predict(model, values["validation"])
            saved = verify_snapshot(
                TRAINING / arm / "best-calibrated.npz",
                rows["validation"],
                targets["validation"],
                trained["best_calibrated"],
                trained["temperatures"],
            )
            if not np.array_equal(original, saved) or not np.array_equal(
                original, readout_from_latent(model, a, val_h, present)
            ):
                raise ValueError("native saved/boundary decomposition parity differs")
            replaced = latent_controls(train_h, present)
            np.savez_compressed(
                args.output / (arm + "-latents.npz"),
                train=train_h.numpy(),
                validation=val_h.numpy(),
                train_mean=replaced["latent_train_mean"][0][0].numpy(),
                train_first=train_h[0].numpy(),
                train_source_ids=np.array([r["source_id"] for r in rows["train"]]),
                validation_source_ids=np.array([r["source_id"] for r in rows["validation"]]),
            )
            output = {}
            input_cases = dict(
                matched=(a, v, present),
                input_zero=(a, torch.zeros_like(v), present),
                input_train_first=(a, values["train"][1][:1].repeat(len(a), 1, 1), present),
                reverse=(a, v.flip(1), present),
            )
            for name, given in input_cases.items():
                bounded(args.output, tick)
                output[name] = predict(model, given)
            if not np.array_equal(output["matched"], original):
                raise ValueError("repeated matched-control inference differs")
            for name, (h, mask) in replaced.items():
                output[name] = readout_from_latent(model, a, h, mask)
            controls = {}
            for name, logits in output.items():
                metrics = store_prediction(
                    args.output,
                    arm + "-" + name,
                    logits,
                    targets["validation"],
                    rows["validation"],
                    trained["temperatures"],
                )
                controls[name] = dict(
                    metrics=metrics,
                    slot_answer_changes=int((answers(logits) != answers(original)).sum()),
                    max_abs_raw_logit_change=float(abs(logits - original).max()),
                )
            weight = model.readout[0].weight.detach().numpy().astype(np.float64)
            report = dict(
                best_epoch=trained["best_epoch"],
                checkpoint_sha256=trained["best_sha256"],
                temperatures=trained["temperatures"],
                controls=controls,
                train_latent_summary=summary(train_h.numpy()),
                validation_latent_summary=summary(val_h.numpy()),
                audio_first_affine_summary=summary(
                    a.numpy().astype(np.float64) @ weight[:, :384].T
                ),
                visual_first_affine_summary=summary(
                    val_h.numpy().astype(np.float64) @ weight[:, 384:448].T
                ),
                validation_visual_input_std_mean=float(v.std(dim=0, unbiased=False).mean()),
                unchanged_tensor_sha256=before,
                native_decomposition_bitwise_exact=True,
            )
            if arm.startswith("joint"):
                report["paired_train_mean_minus_matched"] = paired_bootstrap(
                    original,
                    output["latent_train_mean"],
                    rows["validation"],
                    trained["temperatures"],
                    trained["temperatures"],
                    replicates=10000,
                    seed=17,
                )
            if state_sha(model) != before or any(p.grad is not None for p in model.parameters()):
                raise ValueError("diagnostic changed weights or computed gradients")
            result["arms"][arm] = report
            if reference:
                if reference["arms"][arm] != report:
                    raise ValueError("fresh diagnostic metrics/stats/intervals differ")
                for name in output:
                    for key in [
                        "logits",
                        "probabilities",
                        "targets",
                        "source_ids",
                        "speaker_groups",
                    ]:
                        with np.load(
                            args.output / f"{arm}-{name}.npz", allow_pickle=False
                        ) as current:
                            with np.load(
                                args.replay_reference / f"{arm}-{name}.npz", allow_pickle=False
                            ) as prior:
                                if not np.array_equal(current[key], prior[key]):
                                    raise ValueError("fresh control payload differs")
                with np.load(args.output / f"{arm}-latents.npz", allow_pickle=False) as current:
                    with np.load(
                        args.replay_reference / f"{arm}-latents.npz", allow_pickle=False
                    ) as prior:
                        if any(not np.array_equal(current[k], prior[k]) for k in current.files):
                            raise ValueError("fresh full training/validation latent arrays differ")
            del model, values, train_h, val_h, input_cases, replaced, output
            print("diagnosed", arm, flush=True)
        recheck(pins)
        if source_scope(False)[4] != provenance or torch.cuda.is_initialized():
            raise ValueError("source authority changed or CUDA initialized")
        if reference:
            if sha(args.replay_reference / "result.json") != args.replay_sha256:
                raise ValueError("primary diagnostic receipt changed")
            recheck(
                {
                    str((args.replay_reference / k).resolve()): v
                    for k, v in reference["artifact_hashes"].items()
                }
            )
        result.update(
            status="PASS_FIXED_VISUAL_STATE_DIAGNOSIS",
            fresh_process_replay=reference is not None,
            source_and_weights_unchanged=True,
            artifact_hashes={p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
        )
    except Exception:
        result.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        try:
            result.update(**bounded(args.output, tick))
        except Exception as error:
            result.update(status="FAIL_INCOMPLETE", resource_error=repr(error))
        result["ended_utc"] = datetime.now(UTC).isoformat()
        (args.output / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    if result["status"] != "PASS_FIXED_VISUAL_STATE_DIAGNOSIS":
        raise RuntimeError("diagnostic failed finalization")
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
