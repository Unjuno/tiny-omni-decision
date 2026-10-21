"""Read-only audit of the actual frozen common corpus and completed long fit."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import struct
import subprocess
import sys
import time
import traceback
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.native_pipeline_audit import (
    annotation_counts,
    crop_geometry,
    independent_feature_slice,
    label_coverage,
    semantic_text_target,
    validate_associations,
)

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
REPO = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle]
        return json.load(handle)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def real_gradient_check(store, rows, checkpoint):
    import torch

    from scripts.native_fit import option_order
    from tiny_omni_decision.native_candidate import NativeCandidateReadout

    torch.manual_seed(17)
    model = NativeCandidateReadout()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    other = copy.deepcopy(model)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    losses, maximum = [], 0.0
    for i, row in enumerate(rows):
        blocks, q = store.get(row)
        perm = option_order(row, i)
        model.train()
        scores = model(blocks, q[perm])
        loss = -scores.log_softmax(0)[perm.index(row["target_index"])]
        (loss / len(rows)).backward()
        other.eval()
        reference = other(blocks, q)
        torch.testing.assert_close(scores, reference[perm], rtol=2e-4, atol=2e-6)
        maximum = max(maximum, float((scores - reference[perm]).abs().max().detach()))
        losses.append(-reference.log_softmax(0)[row["target_index"]])
    torch.stack(losses).mean().backward()
    gradient_max = 0.0
    for a, b in zip(model.parameters(), other.parameters(), strict=True):
        if a.grad is None:
            if b.grad is not None:
                raise ValueError("gradient reach differs")
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6)
            gradient_max = max(gradient_max, float((a.grad - b.grad).abs().max()))
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, state[k], rtol=0, atol=0)
    return dict(rows=[r["id"] for r in rows], logit_max_abs=maximum,
                gradient_max_abs=gradient_max, rtol=2e-4, atol=2e-6,
                optimizer_updates=0, train_eval_and_permutation_equivalent=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    pins = {}
    report = dict(status="RUNNING", command=sys.argv,
                  source_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  source_worktree_status=subprocess.check_output(
                      ["git", "status", "--porcelain"], cwd=REPO, text=True),
                  started_utc=datetime.now(UTC).isoformat(), optimizer_updates=0,
                  gpu_used=False, calibration_payloads_loaded=False,
                  sealed_audit_loaded=False, cloud_cost=0)

    def pin(path, expected=None):
        path = Path(path)
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError("hash mismatch: " + str(path))
        if str(path) in pins and actual != pins[str(path)]:
            raise ValueError("input changed: " + str(path))
        pins[str(path)] = actual
        return path

    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("explicit CUDA_VISIBLE_DEVICES='' required for CPU-only audit")
        import torch

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        report["environment"] = dict(python=sys.version, versions={
            p: importlib.metadata.version(p) for p in ("torch", "numpy", "av")})
        for name in ("native_pipeline_audit.py", "audit_native_training_pipeline.py",
                     "native_fit.py", "native_fit_runtime.py", "native_candidate_inputs.py"):
            pin(REPO / "scripts" / name)
        pin(REPO / "src/tiny_omni_decision/native_candidate.py")
        config = read(pin(REPO / "configs/pretrained_reuse/native_common_corpus_v1.json"))
        for path, digest in config["input_pins"].items():
            pin(path, digest)
        corpus = ROOT / "native-common-corpus-02"
        receipt = read(pin(corpus / "result.json",
            "47a6d2f3495810bac58f57dec1a962f8b490fa409c75671b46bb2999fde87d49"))
        for name, digest in receipt["artifacts"].items():
            pin(corpus / name, digest)
        # Historical independent audit is evidence about annotations, not an oracle.
        prior = ROOT / "native-common-label-audit-02"
        report["previous_label_audit"] = dict(result_sha256=sha(pin(prior / "result.json")),
            script_sha256=sha(pin(prior / "audit.py")), scope="source-file agreement only")
        observations, pairs, banks, mapping = [read(corpus / name) for name in (
            "observations.json", "pairs.json", "banks.json", "query-map.json")]
        rows = {s: read(corpus / (s + ".jsonl")) for s in ("train", "validation")}
        combined = rows["train"] + rows["validation"]
        report["associations"] = validate_associations(combined, observations, pairs)
        if report["associations"]["conflicting_identical_inputs"]:
            raise ValueError("identical public inputs have conflicting targets")
        print("rows/query/target associations", len(combined), flush=True)

        text_checks, unknown, mismatches = Counter(), Counter(), []
        for path in config["files"]["text"].values():
            for row in read(path):
                target = semantic_text_target(row)
                if target is None:
                    unknown[row["split"] + ":" + row["source"] + ":" + row["task"]] += 1
                else:
                    text_checks[row["split"] + ":" + row["source"]] += 1
                    if target != row["target"]:
                        mismatches.append(dict(id=row["id"], actual=row["target"], expected=target))
        report["structured_text_labels"] = dict(checked=dict(text_checks),
            mismatches=mismatches, unverified_legacy=dict(unknown),
            scope="label recomputation; generator-rule/question wording reviewed separately")
        if mismatches:
            raise ValueError("structured text labels disagree with independent solution")

        annotation_path = pin(ROOT / "starss23-source-audit-01/metadata_dev.zip",
            "2fa848545a520b4a9d490459de7f079aa49ae16daf0affc859c016f5146981d6")
        starss_report = dict(checked_windows=0, checked_queries=0, source_members=[],
                            brief_positive_frames=Counter(), disagreement=[])
        events = ["speech", "clapping", "telephone", "laughter", "domestic sounds",
                  "footsteps", "door", "music", "instrument", "water tap", "bell", "knock"]
        with zipfile.ZipFile(annotation_path) as archive:
            names = archive.namelist()  # Inventory only; read exact admitted members below.
            texts = {}
            for split, path in config["files"]["starss"].items():
                for row in read(path):
                    recording = row["recording"]
                    if recording not in texts:
                        matches = [n for n in names if n.endswith("/" + recording + ".csv")]
                        if len(matches) != 1 or "dev-train" not in matches[0]:
                            raise ValueError("one admitted official dev-train annotation required")
                        texts[recording] = archive.read(matches[0]).decode("utf-8")
                        starss_report["source_members"].append(matches[0])
                    counts = annotation_counts(texts[recording], row["start_seconds"],
                                                row["end_seconds"])
                    if [q["event"] for q in row["queries"]] != events:
                        raise ValueError("event vocabulary order differs")
                    for q, count in zip(row["queries"], counts, strict=True):
                        if q["target"] != int(count > 0) or q["active_annotation_frames"] != count:
                            starss_report["disagreement"].append(row["id"] + ":" + q["event"])
                        if count:
                            starss_report["brief_positive_frames"][split + ":" + str(count)] += 1
                    starss_report["checked_windows"] += 1
                    starss_report["checked_queries"] += 12
        report["independent_starss_annotations"] = starss_report
        if starss_report["disagreement"]:
            raise ValueError("STARSS interval label mismatch")
        print("independent labels", sum(text_checks.values()), starss_report["checked_queries"],
              flush=True)

        fit = ROOT / "native-long-budget-fit-01"
        run = read(pin(fit / "result.json",
            "645bedbe73ed508fad68c4f636a41df6c8630cd996641c44ce2a87d31550250b"))
        for name, digest in run["artifacts"].items():
            pin(fit / name, digest)
        sampling = read(fit / "sampling.json")
        order = sampling["order"]
        used_indices = [i for b in order for i in b]
        unique = [rows["train"][i] for i in sorted(set(used_indices))]
        actual = [rows["train"][i] for i in used_indices]
        history = read(fit / "training-history.jsonl")
        for step, (record, batch) in enumerate(zip(history, order, strict=True), 1):
            if (record["step"] != step or record["sample_ids"] != [rows["train"][i]["id"]
                                                                     for i in batch]):
                raise ValueError("training history differs from consumed plan")
        coverage = {"train_available": label_coverage(rows["train"]),
                    "train_consumed_unique": label_coverage(unique),
                    "train_presentations": label_coverage(actual),
                    "validation": label_coverage(rows["validation"])}
        write(args.output / "label-coverage.json", coverage)
        report["exposure"] = dict(updates=len(order), presentations=len(actual),
            unique_questions=len(unique), by_source=dict(Counter(r["source"] for r in unique)),
            train_available=len(rows["train"]), label_coverage_file="label-coverage.json")
        report["starss_support"] = {
            event: {s: {k: coverage[s]["starss:" + event][k] for k in
                         ("rows", "unique_rows", "positive", "negative", "always_no_accuracy")}
                    for s in coverage} for event in events}
        train_tasks, val_tasks = set(coverage["train_available"]), set(coverage["validation"])
        report["task_support"] = dict(common=sorted(train_tasks & val_tasks),
            train_only=sorted(train_tasks - val_tasks),
            validation_only=sorted(val_tasks - train_tasks),
            note="Literal task IDs include held-out wording; absence is not necessarily absence "
                 "of the underlying semantic operation. Do not drop held-out tasks.")

        query_root = ROOT / "native-common-queries-cache-01"
        query_receipt = read(pin(query_root / "result.json",
            "7f5ffd7362e84770b864a189d50bc29f43be3738f54a07f6dbdc21f1ef7d909e"))
        for name, digest in query_receipt["artifacts"].items():
            pin(query_root / name, digest)
        banks["new/queries.npy"] = dict(path=str(query_root / "queries.npy"),
            sha256=query_receipt["artifacts"]["queries.npy"], shape=[2622, 384])
        bank_stats = {}
        for name, entry in banks.items():
            path = pin(entry["path"], entry["sha256"])
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != entry["shape"] or array.dtype != np.float32:
                raise ValueError("bank shape/dtype mismatch")
            for start in range(0, len(array), 4096):
                if not np.isfinite(array[start:start + 4096]).all():
                    raise ValueError("nonfinite feature bank")
            bank_stats[name] = dict(shape=list(array.shape), bytes=path.stat().st_size)
        write(args.output / "bank-integrity.json", bank_stats)
        from scripts.native_fit_runtime import FeatureStore

        store = FeatureStore(observations, banks, mapping, device="cpu")
        duplicate_vectors, vector_keys, occupied = [], {}, set()
        for key, item in mapping.items():
            address = (item["bank"], item["offset"])
            if address in occupied:
                raise ValueError("two pairs share query offset")
            occupied.add(address)
            vector = store.array(item["bank"])[item["offset"]]
            digest = hashlib.sha256(vector.tobytes()).hexdigest()
            if digest in vector_keys:
                duplicate_vectors.append([vector_keys[digest], key])
            vector_keys[digest] = key
            np.testing.assert_array_equal(vector, store.queries[store.query_indices[key]].numpy())
        representatives = {r["observation"]: r for r in combined}
        if set(representatives) != set(observations):
            raise ValueError("orphan/missing observation")
        token_counts = Counter()
        for i, (ident, row) in enumerate(representatives.items()):
            blocks, q = store.get(row)
            for block, spec in zip(blocks, observations[ident]["features"], strict=True):
                expected = independent_feature_slice(store.array(spec["bank"]), spec)
                np.testing.assert_array_equal(block.values.numpy(), expected)
                if not block.valid.all() or not np.isfinite(block.positions.numpy()).all():
                    raise ValueError("invalid mask/position")
                token_counts[block.kind] += len(expected)
            expected_query = np.stack([store.array(mapping[k]["bank"])[mapping[k]["offset"]]
                                       for k in row["queries"]])
            np.testing.assert_array_equal(q.numpy(), expected_query)
            if (i + 1) % 4096 == 0:
                print("runtime transport", i + 1, "/", len(representatives), flush=True)
        report["transport"] = dict(banks=len(banks), observations=len(representatives),
            query_vectors=len(mapping), tokens=dict(token_counts), runtime_max_abs=0,
            duplicate_query_vectors=duplicate_vectors, query_addresses_unique=True)
        first_batch = [rows["train"][i] for i in order[0]]
        report["actual_gradient_path"] = real_gradient_check(store, first_batch, fit / "initial.pt")

        image_root = ROOT / "videoprism-spatial-image-cache-01"
        image_receipt = read(image_root / "result.json")
        image_records = read(pin(image_root / "records.json", image_receipt["records_sha256"]))
        image_geometry = Counter()
        for split, records in image_records.items():
            path = Path(config["files"]["image"]["train" if split == "train" else "validation"])
            for row in records:
                media_path = path.parent / split / "images" / (row["id"] + ".png")
                pin(media_path, row["sha256"])
                with media_path.open("rb") as handle:
                    header = handle.read(24)
                if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
                    raise ValueError("original PNG geometry required")
                width, height = struct.unpack(">II", header[16:24])
                image_geometry[(split, width, height)] += 1
        report["image_input"] = [dict(split=s, assets=n, crop=crop_geometry(w, h))
                                 for (s, w, h), n in sorted(image_geometry.items())]

        import av

        from scripts.extract_frozen_videoprism_features import uniform_indices

        geometry, video_rows, media = Counter(), {}, {}
        for path in config["files"]["video"].values():
            for row in read(path):
                video_rows[row["id"]] = row
                media[row["media_sha256"]] = (row, Path(path).parent / row["media_path"])
        details = []
        for key, (row, path) in media.items():
            pin(path, key)
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                width, height, count = stream.width, stream.height, stream.frames
                rate = float(stream.average_rate) if stream.average_rate else None
                if not count or not rate:
                    raise ValueError("unknown frame timeline; independent decoding required")
                indices = uniform_indices(count)
                crop = crop_geometry(width, height)
                geometry[str([width, height, count, rate])] += 1
                details.append(dict(scene=row["scene_group_id"], split=row["split"],
                    sha256=key, path=str(path), indices=indices, crop=crop,
                    retained_frames=len(set(indices)) / count,
                    maximum_gap_seconds=max(np.diff(indices)).item() / rate))
        write(args.output / "video-geometry.json", details)
        report["video_input"] = dict(assets=len(details), geometry=dict(geometry),
            crop_retained_fractions=sorted({r["crop"]["retained_fraction"] for r in details}),
            sampled_frame_fractions=sorted({r["retained_frames"] for r in details}),
            maximum_gap_seconds=max(r["maximum_gap_seconds"] for r in details),
            programs=dict(Counter(op for row in video_rows.values() for op in set(row["program"]))),
            limitation="Geometry establishes discarded pixels/times, not which answers need them.")
        report["status"] = "PASS_AUDIT_EXECUTION_WITH_DATA_LIMITATIONS"
    except BaseException as exc:
        report.update(status="INCOMPLETE", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        changed = [p for p, digest in pins.items() if sha(p) != digest]
        report.update(validated_hashes=pins, changed_inputs=changed,
                      seconds=time.perf_counter() - tick,
                      finished_utc=datetime.now(UTC).isoformat(),
                      artifacts={p.name: sha(p) for p in args.output.iterdir()
                                 if p.name != "result.json"})
        if changed:
            report["status"] = "INCOMPLETE_CHANGED_INPUTS"
        write(args.output / "result.json", report)
    print(report["status"], report["seconds"], flush=True)


if __name__ == "__main__":
    main()
