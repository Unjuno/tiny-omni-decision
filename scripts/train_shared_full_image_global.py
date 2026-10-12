"""One fixed whole-image global arm; same-budget cropped global comparison."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.train_dinov2_image_probe import HASHES, TEXT, sha, write
from scripts.train_frozen_videoprism_clevr4_probe import (
    _option_embeddings,
    _prediction_rows,
    _quality_metrics,
    _taxonomy_metrics,
)
from tiny_omni_decision.bilinear_readout import BilinearCandidateScorer
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES
from tiny_omni_decision.dinov2_features import require_image_splits
from tiny_omni_decision.videoprism_spatial_features import readout_features


def tensor_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "text-model", "output", "feature-cache"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    report = json.loads((a.feature_cache / "result.json").read_text())
    if report["status"] != "PASS":
        raise ValueError("incomplete extraction")
    cache_config = json.loads((a.feature_cache / "config.json").read_text())
    if (cache_config["batch"] != 8 or cache_config["single_frame"] is not True
            or cache_config.get("preprocessing") != "full image bilinear288x288"):
        raise ValueError("extraction contract changed")
    cache_hashes = {f"{key}-cells.npy": value for key, value in report["feature_hashes"].items()}
    cache_hashes["result.json"] = sha(a.feature_cache / "result.json")
    cache_hashes.update({"config.json": report["config_sha256"],
                         "records.json": report["records_sha256"]})
    for name, expected in cache_hashes.items():
        if sha(a.feature_cache / name) != expected:
            raise ValueError(f"cache changed: {name}")
    if sha(a.text_model / "model.safetensors") != TEXT:
        raise ValueError("text model changed")
    directories = {
        "train": a.root / "clevr4-full-coverage-seed17",
        "val": a.root / "clevr4-sample-seed17-1024-512",
    }
    rows = {}
    manifests = {}
    for split, directory in directories.items():
        for kind in ["labels", "image-manifest"]:
            path = directory / f"{split}-{kind}.jsonl"
            if sha(path) != HASHES[path.name]:
                raise ValueError("data hash mismatch")
            data = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
            (rows if kind == "labels" else manifests)[split] = data
        if [r["id"] for r in rows[split]] != [r["id"] for r in manifests[split]]:
            raise ValueError("ID order mismatch")
    require_image_splits(manifests["train"], manifests["val"])
    cached_records = json.loads((a.feature_cache / "records.json").read_text())
    if cached_records != manifests:
        raise ValueError("cache manifest/order mismatch")
    if (len(rows["train"]), len(rows["val"])) != (8424, 512):
        raise ValueError("fixed counts changed")
    if a.smoke:
        rows = {k: v[:8] for k, v in rows.items()}
        manifests = {k: v[:8] for k, v in manifests.items()}
    a.output.mkdir(parents=True)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    start = time.perf_counter()
    config = dict(
        argv={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        runner_sha256=sha(Path(__file__)),
        inputs=HASHES,
        started_at=datetime.now(UTC).isoformat(),
        python=platform.python_version(),
        packages={name: importlib.metadata.version(name)
                  for name in ("torch", "numpy", "transformers")},
        cuda=torch.version.cuda,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        seed=17,
        epochs=1 if a.smoke else 24,
        cache_hashes=cache_hashes,
        normalization="cells /4; spatial mean; frozen single-frame full image bilinear288x288",
        optimizer="AdamW lr1e-3 wd1e-4",
        batch_questions=64,
        selector="earliest min validation NLL",
        readout="bilinear",
        torch=torch.__version__,
        representation_helper_sha256=sha(Path("src/tiny_omni_decision/videoprism_spatial_features.py")),
        extractor_config=cache_config,
        text_model_files={x.name: sha(x) for x in a.text_model.iterdir() if x.is_file()},
    )
    write(a.output / "config.json", config)
    cells = {}
    torch.cuda.reset_peak_memory_stats()
    for split in directories:
        values = np.load(a.feature_cache / f"{split}-cells.npy", allow_pickle=False)
        expected_count = 8424 if split == "train" else 512
        if values.shape != (expected_count, 17, 768) or not np.isfinite(values).all():
            raise ValueError("cache shape or values changed")
        cells[split] = values[:len(rows[split])]
    cache_load_seconds = time.perf_counter() - start
    options, text_encoder = _option_embeddings(device, a.text_model)
    option_tensor = torch.stack(
        [torch.stack([options[f"{t}|{o}"] for o in opts]) for t, opts in CLEVR4_TAXONOMIES.items()]
    ).to(device)
    np.save(a.output / "option-embeddings.npy", option_tensor.cpu().numpy())
    config["option_embeddings_sha256"] = sha(a.output / "option-embeddings.npy")
    write(a.output / "config.json", config)
    del text_encoder, options
    torch.cuda.empty_cache()
    targets = {
        s: [opts.index(str(r[t])) for r in data for t, opts in CLEVR4_TAXONOMIES.items()]
        for s, data in rows.items()
    }
    types = {s: list(CLEVR4_TAXONOMIES) * len(data) for s, data in rows.items()}
    results = {}
    initialization = None
    order_reference = None
    for condition in ["global"]:
        out = a.output / condition
        out.mkdir()
        features = {
            s: torch.from_numpy(readout_features(x, condition)).to(device) for s, x in cells.items()
        }
        torch.manual_seed(17)
        scorer = BilinearCandidateScorer(13056, 384, 128).to(device)
        trainable_parameters = sum(p.numel() for p in scorer.parameters())
        assert trainable_parameters == 1720576
        initial = tensor_hash(scorer)
        reference = json.loads(
            (a.root / "shared-spatial-image-readout-01/global/result.json").read_text())
        if initial != reference["initial_tensor_hash"]:
            raise ValueError("cropped reference initialization mismatch")
        if initialization is not None and initialization != initial:
            raise ValueError("initialization mismatch")
        initialization = initial
        optimizer = torch.optim.AdamW(scorer.parameters(), lr=0.001, weight_decay=0.0001)
        rng = random.Random(17)
        order_hash = hashlib.sha256()
        history = []
        best = float("inf")
        trained_start = time.perf_counter()

        def logits_for(ids, split, rotate=False, _features=features, _scorer=scorer):
            ids = torch.tensor(ids, device=device)
            image_ids = ids // 4
            if rotate:
                image_ids = (image_ids + 1) % len(rows[split])
            visual = _features[split][image_ids, None, :].expand(-1, 10, -1)
            text = option_tensor[ids % 4]
            return _scorer(torch.cat([visual, text], -1)).reshape(len(ids), 10)

        def predict(split, rotate=False, _scorer=scorer, _logits_for=logits_for):
            _scorer.eval()
            output = []
            with torch.inference_mode():
                for offset in range(0, len(targets[split]), 128):
                    ids = list(range(offset, min(offset + 128, len(targets[split]))))
                    output.extend(_logits_for(ids, split, rotate).cpu().unbind())
            return output

        for epoch in range(1, config["epochs"] + 1):
            order = list(range(len(targets["train"])))
            rng.shuffle(order)
            order_hash.update(np.asarray(order, dtype="<i8").tobytes())
            losses = []
            norms = []
            scorer.train()
            for offset in range(0, len(order), 64):
                ids = order[offset : offset + 64]
                logits = logits_for(ids, "train")
                label = torch.tensor([targets["train"][i] for i in ids], device=device)
                loss = F.cross_entropy(logits, label)
                if not torch.isfinite(loss):
                    raise FloatingPointError("loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.linalg.vector_norm(
                    torch.stack([torch.linalg.vector_norm(p.grad) for p in scorer.parameters()])
                )
                if not torch.isfinite(norm):
                    raise FloatingPointError("gradient")
                optimizer.step()
                losses.append(float(loss.detach()))
                norms.append(float(norm))
            evaluation_start = time.perf_counter()
            validation = predict("val")
            training = predict("train")
            quality = _quality_metrics(validation, targets["val"])
            record = dict(
                epoch=epoch,
                elapsed_seconds=time.perf_counter() - trained_start,
                peak_vram_bytes=torch.cuda.max_memory_allocated(),
                evaluation_seconds=time.perf_counter() - evaluation_start,
                rolling_ce=float(np.mean(losses)),
                gradient_norm_mean=float(np.mean(norms)),
                train=_quality_metrics(training, targets["train"]),
                validation=quality,
                train_taxonomies=_taxonomy_metrics(training, targets["train"], types["train"]),
                val_taxonomies=_taxonomy_metrics(validation, targets["val"], types["val"]),
            )
            history.append(record)
            write(out / "history.json", history)
            if quality["nll"] < best:
                best = quality["nll"]
                best_epoch = epoch
                best_logits = [x.clone() for x in validation]
                torch.save(scorer.state_dict(), out / "best.pt")
            print(condition, epoch, quality["accuracy"], quality["nll"], flush=True)
        torch.save(scorer.state_dict(), out / "final.pt")
        scorer.load_state_dict(torch.load(out / "best.pt", weights_only=True, map_location=device))
        selected = predict("val")
        rotated = predict("val", True)
        assert all(torch.equal(x, y) for x, y in zip(selected, best_logits, strict=True))
        for name, values in [("matched", selected), ("rotated", rotated)]:
            write(
                out / f"{name}-predictions.json",
                _prediction_rows(rows["val"], values, targets["val"]),
            )
        order_digest = order_hash.hexdigest()
        if not a.smoke and order_digest != reference["order_sha256"]:
            raise ValueError("cropped reference question order mismatch")
        if order_reference is not None and order_reference != order_digest:
            raise ValueError("sample order mismatch")
        order_reference = order_digest
        results[condition] = dict(
            best_epoch=best_epoch,
            matched=_quality_metrics(selected, targets["val"]),
            rotated=_quality_metrics(rotated, targets["val"]),
            matched_taxonomies=_taxonomy_metrics(selected, targets["val"], types["val"]),
            rotated_taxonomies=_taxonomy_metrics(rotated, targets["val"], types["val"]),
            initial_tensor_hash=initial,
            order_sha256=order_digest,
            best_sha256=sha(out / "best.pt"),
            final_sha256=sha(out / "final.pt"),
            seconds=time.perf_counter() - trained_start,
            reload_exact=True,
            unique_train_images=len(rows["train"]),
            unique_questions=len(targets["train"]),
            consumed=len(targets["train"]) * config["epochs"],
            selected_consumed=len(targets["train"]) * best_epoch,
        )
        write(out / "result.json", results[condition])
        del features, scorer, optimizer, predict, logits_for
        torch.cuda.empty_cache()
    if sha(a.text_model / "model.safetensors") != TEXT:
        raise ValueError("frozen text weights changed")
    for name, expected in cache_hashes.items():
        if sha(a.feature_cache / name) != expected:
            raise ValueError("cache changed during training")
    write(
        a.output / "result.json",
        dict(
            status="PASS_EXECUTION_ONLY",
            finished_at=datetime.now(UTC).isoformat(),
            smoke=a.smoke,
            conditions=results,
            cache_hashes=cache_hashes,
            cache_load_seconds=cache_load_seconds,
            total_seconds=time.perf_counter() - start,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            total_parameters=114365184 + 17389824 + trainable_parameters,
            trainable_parameters=trainable_parameters,
            frozen_weights_unchanged=True,
            gpu=torch.cuda.get_device_name(),
            cost_usd=0,
        ),
    )


if __name__ == "__main__":
    main()
