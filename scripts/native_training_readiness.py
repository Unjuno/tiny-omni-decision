"""Read-only source/cache admission evidence for the common native readout fit."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.native_video_cache import sha

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
REPO = Path(__file__).resolve().parents[1]
TEXT = {
    "train": ["binding_train", "numeric_train", "legacy_train"],
    "validation": ["validation", "old_validation", "legacy_validation"],
}
FIELDS = ("color", "texture", "count", "shape")


def state_key(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = " ".join(unicodedata.normalize("NFKC", value).split())
    if isinstance(value, dict) and set(value) == set(FIELDS):
        value = {k: str(v) for k, v in value.items()}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def overlaps(states):
    common = set(states["train"]) & set(states["validation"])
    pairs = Counter(",".join(sorted(states["train"][key])) + " -> "
                    + ",".join(sorted(states["validation"][key])) for key in common)
    return dict(shared_attribute_or_state_signatures=len(common),
                source_pairs=dict(sorted(pairs.items())),
                does_not_by_itself_establish_media_leakage=True)


def token_lengths(tokenizer, texts, pairs=None):
    if not texts:
        raise ValueError("nonempty inventory required")
    sizes = []
    for offset in range(0, len(texts), 512):
        kwargs = dict(add_special_tokens=True, truncation=False)
        if pairs is not None:
            kwargs["text_pair"] = pairs[offset:offset + 512]
        encoded = tokenizer(texts[offset:offset + 512], **kwargs)["input_ids"]
        sizes.extend(len(ids) for ids in encoded)
    return dict(count=len(sizes), max_tokens=max(sizes), over_512=sum(n > 512 for n in sizes),
                over_8192=sum(n > 8192 for n in sizes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO):
        raise ValueError("clean committed source required")
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    pins = {}

    def pin(path):
        pins[str(path)] = sha(path)
        return path

    def read(path):
        return json.loads(pin(path).read_text(encoding="utf-8"))

    def lines(path):
        return [json.loads(line) for line in pin(path).read_text(encoding="utf-8").splitlines()]

    pin(Path(__file__))
    pin(REPO / "scripts/native_video_cache.py")
    model = Path("C:/CodexArtifacts/pretrained-reuse-r1/deberta-xsmall-zeroshot-audit-01/model")
    for name in ("config.json", "added_tokens.json", "special_tokens_map.json", "spm.model",
                 "tokenizer.json", "tokenizer_config.json"):
        pin(model / name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    states = {split: defaultdict(set) for split in TEXT}
    text_states = {split: set() for split in TEXT}
    pairs, counts = set(), {}
    for split, names in TEXT.items():
        for name in names:
            rows = lines(ROOT / "condition-relevance-corpus-01" / (name + ".jsonl"))
            unique = sorted({r["state"] for r in rows})
            counts[name] = dict(questions=len(rows), states=len(unique),
                                token_lengths=token_lengths(tokenizer, unique))
            for row in rows:
                key = state_key(row["state"])
                text_states[split].add(key)
                states[split][key].add("text")
                pairs.update((row["question"], option) for option in row["options"])
    pair_order = sorted(pairs)
    pair_lengths = token_lengths(tokenizer, [x[0] for x in pair_order],
                                 [x[1] for x in pair_order])
    image_records = read(ROOT / "videoprism-spatial-image-cache-01/records.json")
    for split, file in [("train", "clevr4-full-coverage-seed17/train-labels.jsonl"),
                        ("validation", "clevr4-sample-seed17-1024-512/val-labels.jsonl")]:
        rows = lines(ROOT / file)
        cache_split = "val" if split == "validation" else "train"
        if [r["id"] for r in rows] != [r["id"] for r in image_records[cache_split]]:
            raise ValueError("image labels/cache identity order differs")
        for row in rows:
            states[split][state_key({k: row[k] for k in FIELDS})].add("image")
        counts["image_" + split] = dict(assets=len(rows),
            attribute_signatures=len({state_key({k: r[k] for k in FIELDS}) for r in rows}))
    physical = {s: {r["sha256"] for r in rows} for s, rows in image_records.items()}
    image_overlap = len(physical["train"] & physical["val"])
    image_cache = ROOT / "videoprism-spatial-image-cache-01"
    image_report = read(image_cache / "result.json")
    for split in ("train", "val"):
        if sha(pin(image_cache / (split + "-cells.npy"))) != image_report["feature_hashes"][split]:
            raise ValueError("image cache hash mismatch")
    native = ROOT / "native-conditional-candidate-smoke-02"
    pin(native / "image-tokens.npy")
    image_delta = float(np.max(np.abs(np.load(image_cache / "train-cells.npy", mmap_mode="r")[0]
                                     - np.load(native / "image-tokens.npy"))))
    speech = ROOT / "lombard-grid-whisper-features-01"
    speech_report = read(speech / "result.json")
    bank = pin(speech / "tokens.npy")
    if pins[str(bank)] != speech_report["feature_hashes"]["tokens.npy"]:
        raise ValueError("speech cache hash mismatch")
    row = next(r for r in speech_report["rows"] if r["source_id"] == "s11_l_bbab2n")
    pin(native / "speech-tokens.npy")
    speech_delta = float(np.max(np.abs(np.load(bank, mmap_mode="r")
        [row["offset"]:row["offset"] + row["tokens"]]
        - np.load(native / "speech-tokens.npy"))))
    for split in ("train", "validation"):
        rows = read(ROOT / "lombard-grid-layout-02-media-gated" / (split + ".json"))
        counts["speech_" + split] = dict(utterances=len(rows),
            speaker_groups=len({r["speaker_group"] for r in rows}),
            phrases=len({r["actual_code"] for r in rows}))
        windows = lines(ROOT / "starss23-all-windows-v1" / (split + ".jsonl"))
        events = sorted({q["event"] for r in windows for q in r["queries"]})
        positive = Counter(q["event"] for r in windows for q in r["queries"] if q["target"] == 1)
        counts["events_" + split] = dict(windows=len(windows),
            recordings=len({r["recording"] for r in windows}),
            rooms=len({r["room"] for r in windows}),
            positive_events={name: positive[name] for name in events})
    video = read(ROOT / "native-training-video-cache-01/result.json")
    if (video["status"] != "PASS_COMPLETE_NATIVE_VIDEO_TOKEN_CACHE"
            or video["counts"] != {"train": 256, "validation": 200} or video["completed"] != 456):
        raise ValueError("complete456-scene cache required")
    for row in video["rows"]:
        path = ROOT / "native-training-video-cache-01" / row["file"]
        if sha(pin(path)) != row["file_sha256"]:
            raise ValueError("persisted video cache changed")
    result = dict(status="PASS_READINESS_AUDIT_TRAINING_NOT_ADMITTED", source_commit=
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        counts=counts, query_candidate_lengths=pair_lengths, signature_overlap=overlaps(states),
        text_normalized_state_overlap=len(text_states["train"] & text_states["validation"]),
        image_media_hash_overlap=image_overlap, image_native_max_abs=image_delta,
        speech_native_max_abs=speech_delta, fresh_parity_scope="one existing TRAIN input per kind",
        video_cache=dict(counts=video["counts"], completed=456,
                         total_token_file_bytes=video["total_token_file_bytes"]),
        missing_before_fit=["lossless handling of long text states and language token/query cache",
            "unified speech/event feature profile; current speech cache is not exact eager parity",
            "complementary audio features: speech for event clips, event features for speech clips",
            "joint task contract and eight-frame native paired-video coverage",
            "common sampler/budget/selector/retention and source-held-out evaluation contract"],
        calibration_labels_or_feature_slices_loaded=False,
        shared_speech_cache_including_reserved_bytes_hashed_only=True,
        sealed_audit_loaded=False, model_loaded=False, tokenizer_loaded=True, optimizer_updates=0,
        native_quality_evaluated=False, corpus_modified=False,
        seconds=time.perf_counter() - started,
        pins=pins, environment=dict(python=sys.version, numpy=np.__version__,
            transformers=importlib.metadata.version("transformers"),
            tokenizer=type(tokenizer).__name__, command=sys.argv))
    for path, expected in pins.items():
        if sha(Path(path)) != expected:
            raise ValueError("input changed during read-only audit")
    with (args.output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ["status", "seconds", "signature_overlap",
                                           "speech_native_max_abs"]}), flush=True)


if __name__ == "__main__":
    main()
