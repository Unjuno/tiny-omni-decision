"""Known-train, scene-held-out auxiliary mask correspondence; not Omni QA."""

import argparse
import hashlib
import inspect
import json
import math
import platform
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.train_spatial_trajectory_readout import process_peak_bytes
from tiny_omni_decision.decision_math import expected_calibration_error
from tiny_omni_decision.mask_pair_association import (
    association_windows,
    grouped_pair_split,
    labelled_mask_pairs,
    pair_features,
    paired_scene_interval,
    training_standardizer,
)
from tiny_omni_decision.mask_pair_readout import MaskPairReadout
from tiny_omni_decision.mask_trajectory_features import mask_query_points
from tiny_omni_decision.spatial_relations import seed_box_roles

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
PINS = dict(
    zip(
        (
            "native-mask-feature-corpus-eight-01/result.json",
            "native-mask-feature-corpus-independent-audit-01.json",
            "native-role-opportunity-eight-01.json",
            "native-role-opportunity-independent-audit-01.json",
        ),
        (
            "521602ff3fb26f4f4b1732c7b8567da89a0ccdcab2fb506f0a0fa7e7e38c3d90",
            "013bb382ae9679af1ec76e18f8a8906fb8d686eb9ce3dc14944802b21feba9d8",
            "823e0557d798cf5f0d6118a1eb127877f4811591410bc58d8bc83c77935bd052",
            "226013a7234bd5171a19c4574ec286de27709e32b14944bfa492441de9788ab2",
        ),
        strict=True,
    )
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


def verify(pins):
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError("protected inputs/code/config changed")


def limits(start, output):
    if (
        time.perf_counter() - start > 600
        or process_peak_bytes() > 4 * 2**30
        or shutil.disk_usage(output).free < 8 * 2**30
    ):
        raise RuntimeError("predeclared CPU time/memory/disk cap exceeded")


def require_smoke_binding(directory, config_sha, corpus_sha):
    result_path = directory / "result.json"
    config_path = directory / "effective-config.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result["status"] != "PASS_MASK_PAIR_SMOKE_NOT_QUALITY"
        or not result["all_final_reload_exact"]
        or result["config_sha256"] != config_sha
        or sha(config_path) != config_sha
        or result["pair_corpus_sha256"] != corpus_sha
        or result["steps_per_fit"] != 8
        or result["completed_fits"] != 2
        or {r["arm"] for r in result["final"]} != {"geometry", "appearance"}
    ):
        raise ValueError("complete same-config separate-process smoke required")
    pins = {str(result_path): sha(result_path), str(config_path): config_sha}
    for row in result["final"]:
        if not row["reload_exact"] or row["parameters"] != (
            449 if row["arm"] == "geometry" else 98753
        ):
            raise ValueError("unexpected smoke model/reload")
        for prefix in ("weights", "predictions"):
            pins[row[prefix + "_path"]] = row[prefix + "_sha256"]
    verify(pins)
    return pins


def prepare(output, start):
    parents = {}
    for name, pin in PINS.items():
        if sha(ROOT / name) != pin:
            raise ValueError("completed parent changed: " + name)
        parents[name] = json.loads((ROOT / name).read_text(encoding="utf-8"))
    features = parents["native-mask-feature-corpus-eight-01/result.json"]
    census = parents["native-role-opportunity-eight-01.json"]
    if (
        features["status"] != "PASS_EIGHT_CLIP_MASK_FEATURE_INPUT_CORPUS_NOT_QUALITY"
        or parents["native-mask-feature-corpus-independent-audit-01.json"]["arrays_compared"] != 72
        or parents["native-role-opportunity-independent-audit-01.json"]["source_cases_checked"]
        != 1024
    ):
        raise ValueError("complete input and source-geometry admission required")
    cfg_path = ROOT / "native-mask-feature-corpus-eight-01/effective-config.json"
    if sha(cfg_path) != features["config_sha256"]:
        raise ValueError("frozen feature config changed")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    pins = dict(cfg["source_hashes"])
    pins.update({str(ROOT / p): h for p, h in PINS.items()})
    for row in features["rows"]:
        pins[row["path"]] = row["sha256"]
    for p in [
        cfg_path,
        Path(__file__).resolve(),
        Path("src/tiny_omni_decision/mask_pair_association.py").resolve(),
        Path("src/tiny_omni_decision/mask_pair_readout.py").resolve(),
        Path("src/tiny_omni_decision/mask_trajectory_features.py").resolve(),
        Path("src/tiny_omni_decision/spatial_relations.py").resolve(),
        Path("docs/pretrained_reuse/R2_MASK_PAIR_ASSOCIATION_PROTOCOL_20261014.md").resolve(),
        Path(inspect.getfile(process_peak_bytes)),
        Path(inspect.getfile(expected_calibration_error)),
    ]:
        pins[str(p)] = sha(p)
    verify(pins)
    native = json.loads(
        (ROOT / "native-dynamic-corpus-eight-01/result.json").read_text(encoding="utf-8")
    )
    collections = {
        k: []
        for k in (
            "features",
            "control_features",
            "targets",
            "strata",
            "video_indices",
            "pair_local_indices",
            "source_pair_ids",
        )
    }
    ledgers = []
    for video_index, (plan, row) in enumerate(zip(cfg["plans"], features["rows"], strict=True)):
        video = plan["video_id"]
        if row["video_id"] != video:
            raise ValueError("source identity order changed")
        with np.load(row["path"], allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        labels = np.full(row["proposals"], -1, np.int64)
        query_map, destination_map = np.arange(len(labels)), np.arange(len(labels))
        frame_records = [r for r in native["rows"] if r["video_id"] == video]
        for i, record in enumerate(frame_records):
            begin, end = map(int, arrays["frame_offsets"][i : i + 2])
            ids = arrays["proposal_ids"][begin:end]
            if end == begin:
                continue
            with np.load(record["path"], allow_pickle=False) as bank:
                masks = (
                    np.unpackbits(bank["masks_packed"][ids], axis=1, bitorder="big")
                    .reshape(len(ids), 256, 256)
                    .astype(bool)
                )
            xy, nonempty = mask_query_points(masks)
            boxes = arrays["boxes"][begin:end]
            valid = nonempty & (boxes[:, :2] < boxes[:, 2:]).all(1)
            cases = [
                c
                for c in census["cases"]
                if c["video_id"] == video
                and c["frame_index"] == record["frame_index"]
                and c["source_proxy"]["box"] is not None
            ]
            if valid.any():
                labels[begin:end][valid] = seed_box_roles(
                    boxes[valid],
                    xy[valid] / 256,
                    np.array([c["source_track_id"] for c in cases], np.int64),
                    np.array([c["source_proxy"]["box"] for c in cases], np.float64).reshape(-1, 4),
                )
            query_map[begin:end] = np.roll(np.arange(begin, end), -1)
            destination_map[begin:end] = np.roll(np.arange(begin, end), -7)
        pairs, y, strata, ledger = labelled_mask_pairs(
            labels, arrays["frame_offsets"], association_windows(32)
        )
        if len(pairs) == 0 or len(np.unique(y)) != 2:
            raise ValueError("every retained scene needs both pair classes; no scene dropping")
        appearance = np.concatenate((arrays["mean"], arrays["std"]), 1)
        x = pair_features(
            appearance, arrays["boxes"], arrays["frame_indices"], pairs, plan["frames"]
        )
        q, d = (
            appearance[query_map[pairs[:, 0]]].astype(np.float64),
            appearance[destination_map[pairs[:, 1]]].astype(np.float64),
        )
        control = x.copy()
        control[:, :1536] = np.concatenate((np.abs(q - d), q * d), 1).astype(np.float32)
        if not np.isfinite(control).all():
            raise ValueError("nonfinite mismatched appearance control")
        values = dict(
            features=x,
            control_features=control,
            targets=y,
            strata=strata,
            video_indices=np.full(len(y), video_index, np.int64),
            pair_local_indices=pairs,
            source_pair_ids=labels[pairs],
        )
        for k, a in values.items():
            collections[k].append(a)
        unique_ids = np.unique(labels[labels >= 0]).tolist()
        ledger.update(
            video_id=video,
            unique_source_tracks_with_proxy_rows=len(unique_ids),
            unique_proxy_rows=int((labels >= 0).sum()),
            unknown_proxy_rows=int((labels == -1).sum()),
            ambiguous_proxy_rows=int((labels == -2).sum()),
            source_track_ids=unique_ids,
            positive_pairs=int(y.sum()),
            negative_pairs=int((y == 0).sum()),
        )
        for k in range(3):
            ledger["strata"][str(k)].update(
                count=int((strata == k).sum()), same=int(y[strata == k].sum())
            )
        if any(ledger["strata"][str(k)]["count"] == 0 for k in range(3)):
            raise ValueError("all predefined temporal strata required in every scene")
        ledgers.append(ledger)
        print(
            video,
            len(y),
            "LABEL_CONDITIONED_PAIRS",
            ledger["unknown_or_ambiguous_pairs"],
            "UNSCORED",
            flush=True,
        )
        limits(start, output)
    arrays = {k: np.concatenate(v) for k, v in collections.items()}
    dataset = output / "pairs.npz"
    np.savez_compressed(dataset, **arrays)
    with np.load(dataset, allow_pickle=False) as saved:
        for k, a in arrays.items():
            if saved[k].dtype != a.dtype or saved[k].tobytes() != a.tobytes():
                raise ValueError("pair corpus readback differs")
    for fold in range(8):
        train, test = grouped_pair_split(arrays["video_indices"], fold)
        if set(arrays["video_indices"][train]) & set(arrays["video_indices"][test]):
            raise ValueError("scene overlap")
    config = dict(
        source_hashes=pins,
        seed=17,
        steps=512,
        batch=128,
        hidden=64,
        learning_rate=5e-4,
        weight_decay=1e-4,
        gradient_clip=1.0,
        normalizer_floor=1e-6,
        threads=2,
        interop=1,
        checkpoint_policy="fixed final step; no held-out selection",
        eval_steps=[0, 128, 256, 384, 512],
        options=["different", "same"],
        training_sources="8 known train videos; label-conditioned proxy task",
        geometry_features=5,
        appearance_features=1536,
        protocol="R2_MASK_PAIR_ASSOCIATION_PROTOCOL_20261014",
        corpus_sha256=sha(dataset),
        cpu_body_cap=600,
        working_set_cap=4 * 2**30,
        cloud_cost=0,
    )
    write(output / "effective-config.json", config)
    verify(pins)
    limits(start, output)
    write(
        output / "result.json",
        dict(
            status="PASS_PAIR_CORPUS_PREPARATION_NOT_LEARNING",
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            config_sha256=sha(output / "effective-config.json"),
            path=str(dataset),
            sha256=sha(dataset),
            file_bytes=dataset.stat().st_size,
            pairs=len(arrays["targets"]),
            ledgers=ledgers,
            source_assets=8,
            unique_source_track_instances=sum(
                x["unique_source_tracks_with_proxy_rows"] for x in ledgers
            ),
            annotations_supplied_only_as_supervision=True,
            sealed_audit_opened=False,
            seconds=time.perf_counter() - start,
            training=False,
            gpu_or_pretrained_model_loaded=False,
        ),
    )


def metrics(logp, targets):
    p = np.exp(logp.astype(np.float64))
    pred = p.argmax(1)
    recalls = [
        float((pred[targets == i] == i).mean()) if (targets == i).any() else None for i in (0, 1)
    ]
    return dict(
        count=len(targets),
        accuracy=float((pred == targets).mean()),
        nll=float(-logp[np.arange(len(targets)), targets].astype(np.float64).mean()),
        brier=float(((p - np.eye(2)[targets]) ** 2).sum(1).mean()),
        ece=expected_calibration_error(p.tolist(), targets.tolist(), 15),
        class_recalls=recalls,
        balanced_accuracy=sum(recalls) / 2 if None not in recalls else None,
        same_prevalence=float(targets.mean()),
    )


def predict(model, features):
    model.eval()
    values = []
    with torch.no_grad():
        for begin in range(0, len(features), 512):
            values.append(
                model(torch.from_numpy(features[begin : begin + 512])).log_softmax(1).numpy()
            )
    return np.concatenate(values)


def measure(logp, targets, strata):
    return dict(
        all=metrics(logp, targets),
        strata={
            str(k): metrics(logp[strata == k], targets[strata == k])
            if (strata == k).any()
            else None
            for k in range(3)
        },
    )


def fit(args, start):
    if args.corpus is None:
        raise ValueError("prepared pair corpus required")
    prepared_path = args.corpus / "result.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    config_path = args.corpus / "effective-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        prepared["status"] != "PASS_PAIR_CORPUS_PREPARATION_NOT_LEARNING"
        or sha(config_path) != prepared["config_sha256"]
        or sha(prepared["path"]) != prepared["sha256"]
    ):
        raise ValueError("unaltered prepared corpus/config required")
    verify(config["source_hashes"])
    protected = {
        str(prepared_path): sha(prepared_path),
        str(config_path): sha(config_path),
        prepared["path"]: prepared["sha256"],
    }
    if not args.smoke:
        if args.smoke_reference is None:
            raise ValueError("main requires completed separate-process smoke")
        protected.update(
            require_smoke_binding(
                args.smoke_reference, prepared["config_sha256"], prepared["sha256"]
            )
        )
    (args.output / "effective-config.json").write_bytes(config_path.read_bytes())
    protected[str(args.output / "effective-config.json")] = prepared["config_sha256"]
    with np.load(prepared["path"], allow_pickle=False) as saved:
        x, control, y, strata, groups = [
            saved[k] for k in ("features", "control_features", "targets", "strata", "video_indices")
        ]
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    steps = 8 if args.smoke else 512
    folds = range(1) if args.smoke else range(8)
    final, history = [], []
    oof = {
        arm: np.full((len(y), 2), np.nan, np.float32)
        for arm in ("geometry", "appearance", "appearance_mismatch")
    }
    for fold in folds:
        train, test = grouped_pair_split(groups, fold)
        train_ids, test_ids = np.flatnonzero(train), np.flatnonzero(test)
        rng = np.random.default_rng(17)
        order = np.concatenate(
            [rng.permutation(train_ids) for _ in range(math.ceil(steps * 128 / len(train_ids)))]
        )[: steps * 128]
        fold_dir = args.output / ("fold-" + str(fold))
        fold_dir.mkdir()
        np.save(fold_dir / "training-order.npy", order)
        prior = np.array([1 - y[train].mean(), y[train].mean()], np.float64)
        baseline = measure(
            np.broadcast_to(np.log(prior), (len(test_ids), 2)), y[test], strata[test]
        )
        for arm in ("geometry", "appearance"):
            selected = x[:, -5:] if arm == "geometry" else x
            mean, scale = training_standardizer(selected, train)
            normal = (selected - mean) / scale
            if not np.isfinite(normal).all():
                raise ValueError("nonfinite normalized features")
            torch.manual_seed(17)
            model = MaskPairReadout(normal.shape[1], hidden=64)
            expected_params = 449 if arm == "geometry" else 98753
            if sum(p.numel() for p in model.parameters()) != expected_params:
                raise ValueError("unexpected readout capacity")
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
            norm = None
            for step in range(steps + 1):
                if step in (0, 128, 256, 384, steps):
                    train_logp = predict(model, normal[train])
                    heldout_logp = predict(model, normal[test])
                    consumed = order[: step * 128]
                    record = dict(
                        fold=fold,
                        arm=arm,
                        step=step,
                        train=measure(train_logp, y[train], strata[train]),
                        heldout=measure(heldout_logp, y[test], strata[test]),
                        consumed_pairs=len(consumed),
                        unique_consumed_pairs=len(np.unique(consumed)),
                        repeated_pairs=len(consumed) - len(np.unique(consumed)),
                        last_gradient_norm=norm,
                        seconds=time.perf_counter() - start,
                        heldout_used_for_selection=False,
                    )
                    history.append(record)
                    with (args.output / "history.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                    print(fold, arm, step, record["heldout"]["all"]["nll"], flush=True)
                    limits(start, args.output)
                if step == steps:
                    break
                model.train()
                batch = order[step * 128 : (step + 1) * 128]
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(
                    model(torch.from_numpy(normal[batch])), torch.from_numpy(y[batch])
                )
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                if not math.isfinite(norm) or any(
                    p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()
                ):
                    raise ValueError("invalid gradient")
                optimizer.step()
                if step % 32 == 0:
                    limits(start, args.output)
            dest = fold_dir / (arm + ".npz")
            weights = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
            np.savez_compressed(dest, mean=mean, scale=scale, **weights)
            replay = MaskPairReadout(normal.shape[1], hidden=64)
            with np.load(dest, allow_pickle=False) as saved:
                replay.load_state_dict(
                    {k: torch.from_numpy(saved[k].copy()) for k in weights}, strict=True
                )
                replay_normal = (selected - saved["mean"]) / saved["scale"]
            reread = predict(replay, replay_normal[test])
            if not np.array_equal(reread, heldout_logp):
                raise ValueError("final reload differs")
            oof[arm][test] = heldout_logp
            mismatch = None
            if arm == "appearance":
                mismatch = predict(replay, (control[test] - mean) / scale)
                oof["appearance_mismatch"][test] = mismatch
            pred_path = fold_dir / (arm + "-predictions.npz")
            np.savez_compressed(
                pred_path,
                pair_ids=test_ids,
                targets=y[test],
                strata=strata[test],
                logp=heldout_logp,
                probabilities=np.exp(heldout_logp),
                mismatch_logp=mismatch if mismatch is not None else np.empty((0, 2), np.float32),
            )
            final.append(
                dict(
                    fold=fold,
                    arm=arm,
                    parameters=expected_params,
                    weights_path=str(dest),
                    weights_sha256=sha(dest),
                    weights_bytes=dest.stat().st_size,
                    predictions_path=str(pred_path),
                    predictions_sha256=sha(pred_path),
                    metrics=history[-1],
                    prior=baseline,
                    mismatch=measure(mismatch, y[test], strata[test])
                    if mismatch is not None
                    else None,
                    sample_order_sha256=sha(fold_dir / "training-order.npy"),
                    reload_exact=True,
                )
            )
            del model, replay, optimizer, normal, replay_normal
    if not args.smoke and any(not np.isfinite(p).all() for p in oof.values()):
        raise ValueError("incomplete held-out fold coverage")
    verify(config["source_hashes"])
    verify(protected)
    limits(start, args.output)
    comparison = None
    if not args.smoke:
        comparison = {}
        for metric in ("accuracy", "nll", "brier"):
            delta = np.array(
                [
                    next(r for r in final if r["fold"] == fold and r["arm"] == "appearance")[
                        "metrics"
                    ]["heldout"]["all"][metric]
                    - next(r for r in final if r["fold"] == fold and r["arm"] == "geometry")[
                        "metrics"
                    ]["heldout"]["all"][metric]
                    for fold in range(8)
                ],
                np.float64,
            )
            comparison[metric] = paired_scene_interval(delta)
        oof_path = args.output / "oof-predictions.npz"
        np.savez_compressed(oof_path, targets=y, strata=strata, video_indices=groups, **oof)
    verify(config["source_hashes"])
    verify(protected)
    limits(start, args.output)
    write(
        args.output / "result.json",
        dict(
            status="PASS_MASK_PAIR_SMOKE_NOT_QUALITY"
            if args.smoke
            else "PASS_KNOWN_TRAIN_SCENE_HELDOUT_ASSOCIATION_DIAGNOSTIC_NOT_OMNI_QUALITY",
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            config_sha256=prepared["config_sha256"],
            pair_corpus_sha256=prepared["sha256"],
            all_final_reload_exact=True,
            smoke=args.smoke,
            steps_per_fit=steps,
            completed_fits=len(final),
            cpu_only=True,
            pretrained_models_loaded=False,
            gpu_used=False,
            source_hashes_unchanged=True,
            sealed_audit_opened=False,
            original_qa_evaluated=False,
            heldout_selected_checkpoints=False,
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            peak_working_set_bytes=process_peak_bytes(),
            seconds=time.perf_counter() - start,
            cloud_cost=0,
            final=final,
            paired_scene_comparison=comparison,
            oof_predictions_path=str(oof_path) if not args.smoke else None,
            oof_predictions_sha256=sha(oof_path) if not args.smoke else None,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-reference", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    start = time.perf_counter()
    try:
        limits(start, args.output)
        if args.prepare_only:
            prepare(args.output, start)
        else:
            fit(args, start)
    except Exception:
        write(
            args.output / "result.json",
            dict(
                status="FAIL_INCOMPLETE",
                error=traceback.format_exc(),
                seconds=time.perf_counter() - start,
            ),
        )
        raise


if __name__ == "__main__":
    main()
