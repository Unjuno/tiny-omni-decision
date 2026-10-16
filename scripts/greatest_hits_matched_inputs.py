"""Independent complete-bank admission; no encoder/model inference or test-set access."""

import json
from pathlib import Path

import numpy as np

from scripts.greatest_hits_audio_admission import load_revalidated_audio
from scripts.greatest_hits_matched import digest
from scripts.greatest_hits_transport import bind_caches, validate_positions
from scripts.native_video_cache import sha

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
PREFLIGHT = ROOT / "greatest-hits-matched-preflight-01"
COMMON = ROOT / "native-common-corpus-02"
QUERIES = ROOT / "native-common-queries-cache-01"
TASKS = ROOT / "greatest-hits-development-tasks-01"
AUDIO = ROOT / "greatest-hits-audio-query-cache-01"
VIDEO = ROOT / "greatest-hits-video-cache-01"
VIDEO_SMOKE = ROOT / "greatest-hits-video-cache-smoke-01"
AUDIO_PROOF = ROOT / "greatest-hits-audio-cache-revalidation-01/result.json"
AUDIO_PROOF_SHA = "4b273e75298e640b79cbe5de0e234f1d1b6f54e45965af692cba86471b95f4cb"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Pins:
    def __init__(self):
        self.hashes, self.stamps = {}, {}

    @staticmethod
    def stamp(path):
        s = Path(path).stat()
        return s.st_size, s.st_mtime_ns

    def add(self, path, expected=None):
        path = str(Path(path).resolve())
        before = self.stamp(path)
        actual = sha(Path(path))
        if (
            (expected is not None and actual != expected)
            or self.stamp(path) != before
            or path in self.hashes
            and self.hashes[path] != actual
        ):
            raise ValueError("changed pinned input: " + path)
        self.hashes[path], self.stamps[path] = actual, before
        return actual

    def check(self):
        for path, stamp in self.stamps.items():
            if self.stamp(path) != stamp:
                raise ValueError("source/input stamp changed: " + path)

    def verify(self):
        self.check()
        for path, expected in self.hashes.items():
            if sha(Path(path)) != expected:
                raise ValueError("source/input hash changed: " + path)
        self.check()


def require_video(receipt):
    if (
        receipt["status"] != "PASS_COMPLETE_VIDEO_CACHE"
        or receipt["phase"] != "full"
        or receipt["frozen_before"] != receipt["frozen_after"]
        or not receipt["frozen_before"]
        or receipt["optimizer_updates"] != 0
        or receipt["matched_reference_events"] != 21
        or any(
            receipt[p + "_" + k] != n
            for p in ("expected", "completed")
            for k, n in (("events", 11151), ("parents", 409), ("questions", 32582))
        )
    ):
        raise ValueError("complete immutable full video cache required; producer stays untouched")


def receipt(pins, folder, expected=None):
    pins.add(folder / "result.json", expected)
    value = read(folder / "result.json")
    for name, expected_hash in value["artifacts"].items():
        if name != Path(name).name:
            raise ValueError("local artifact basename required")
        pins.add(folder / name, expected_hash)
    return value


def metadata():
    data = {}
    for kind, folder in (("joint", TASKS), ("common", COMMON)):
        for split in ("train", "validation"):
            data[kind, split] = [
                json.loads(line)
                for line in (folder / (split + ".jsonl")).read_text(encoding="utf-8").splitlines()
            ]
    return data


def common_banks():
    banks = read(COMMON / "banks.json")
    banks["new/queries.npy"] = dict(
        path=str(QUERIES / "queries.npy"),
        sha256=read(QUERIES / "result.json")["artifacts"]["queries.npy"],
        shape=[2622, 384],
    )
    return banks


def audit(pins, guard=lambda: None):
    # Check the terminal receipt before touching a live producer's feature banks.
    require_video(read(VIDEO / "result.json"))
    preflight = receipt(
        pins, PREFLIGHT, "d0dc5e585b00a45cfdc17773b42cd40a0e4b9b08015127af014ab8ce5e7b5ac1"
    )
    common = receipt(
        pins, COMMON, "47a6d2f3495810bac58f57dec1a962f8b490fa409c75671b46bb2999fde87d49"
    )
    tasks = receipt(pins, TASKS, "7ef21536a84904df946f96cd774df9694b99027b1f36d6c8221b8b164fe7a831")
    query = receipt(
        pins, QUERIES, "7f5ffd7362e84770b864a189d50bc29f43be3738f54a07f6dbdc21f1ef7d909e"
    )
    video = receipt(pins, VIDEO)
    smoke = receipt(
        pins, VIDEO_SMOKE, "d0ee01083f70a98be68b3a71c0a9abd68ac9b4b82759cc089e0232ff32bd38db"
    )
    if (
        preflight["status"] != "PASS_MATCHED_AV_PREFLIGHT"
        or common["status"] != "PASS_COMPLETE_NATIVE_COMMON_CORPUS"
        or tasks["status"] != "PASS_FIXED_DEVELOPMENT_TASKS"
        or query["status"] != "PASS_NATIVE_COMMON_QUERIES_CACHE"
        or video["source_commit"] != "857edcf6f243fb23cd2d34d812036e8b43205dfb"
    ):
        raise ValueError("original pinned passing prerequisites required")
    pins.add(AUDIO_PROOF, AUDIO_PROOF_SHA)
    pins.add(
        AUDIO_PROOF.parent / "artifact-hashes.json", read(AUDIO_PROOF)["artifact_ledger_sha256"]
    )
    audio = load_revalidated_audio(AUDIO, AUDIO_PROOF, AUDIO_PROOF_SHA)
    receipt(pins, AUDIO, audio["audio_admission"]["producer_result_sha256"])
    data = metadata()
    if [len(data[k, s]) for k in ("joint", "common") for s in ("train", "validation")] != [
        23707,
        8875,
        100580,
        24346,
    ]:
        raise ValueError("complete original corpora required")
    ids = {}
    for split in ("train", "validation"):
        rows = [r for k in ("joint", "common") for r in data[k, split]]
        ids[split] = {r["id"] for r in rows}
        if len(ids[split]) != len(rows) or any(r["split"] != split for r in rows):
            raise ValueError("unique correctly assigned split identities required")
    if ids["train"] & ids["validation"]:
        raise ValueError("cross-split question overlap")
    validation_ids = {k: [r["id"] for r in data[k, "validation"]] for k in ("joint", "common")}
    if validation_ids != read(PREFLIGHT / "validation-ids.json"):
        raise ValueError("fixed validation order differs")
    binding = bind_caches(
        {s: data["joint", s] for s in ("train", "validation")},
        audio,
        video,
        read(AUDIO / "plan.json"),
        read(VIDEO / "plan.json"),
        phase="full",
    )
    observations = read(COMMON / "observations.json")
    used = {}
    groups = {}
    for split in ("train", "validation"):
        rows = data["common", split]
        used[split] = {r["observation"] for r in rows}
        groups[split] = {(r["source"], r["group"]) for r in rows}
        if any(observations[i]["split"] != split for i in used[split]):
            raise ValueError("common observation split mismatch")
    if used["train"] & used["validation"] or groups["train"] & groups["validation"]:
        raise ValueError("common observation/group overlap")
    for bank in common_banks().values():
        pins.add(bank["path"], bank["sha256"])
        arr = np.load(bank["path"], mmap_mode="r", allow_pickle=False)
        if list(arr.shape) != bank["shape"] or arr.dtype != np.float32:
            raise ValueError("common bank shape/dtype differs")
        flat = arr.reshape(-1)
        for start in range(0, flat.size, 2**20):
            if not np.isfinite(flat[start : start + 2**20]).all():
                raise ValueError("nonfinite common feature bank")
        guard()
    smoke_events = {e["event_id"]: e for p in smoke["parents"] for e in p["events"]}
    positions = {
        k: np.load(AUDIO / (k + "-positions.npy"), allow_pickle=False)
        for k in ("speech", "event_audio")
    }
    count, matches = 0, 0
    for parent in video["parents"]:
        name, n = parent["parent"], len(parent["events"])
        arrays = {
            k: np.load(VIDEO / (name + "-" + k + ".npy"), mmap_mode="r", allow_pickle=False)
            for k in ("video", "positions", "pts")
        }
        if (
            arrays["video"].shape != (n, 2048, 768)
            or arrays["video"].dtype != np.float32
            or arrays["positions"].shape != (n, 2048, 3)
            or arrays["pts"].shape != (n, 8)
        ):
            raise ValueError("full video feature/geometry shape differs")
        for i, event in enumerate(parent["events"]):
            if not np.isfinite(arrays["video"][i]).all():
                raise ValueError("nonfinite video features")
            validate_positions(
                dict(positions, video=arrays["positions"][i]), arrays["pts"][i], event["timing"]
            )
            np.testing.assert_array_equal(arrays["pts"][i], event["selected_pts"])
            if event["event_id"] in smoke_events:
                index = smoke_events[event["event_id"]]["bank_index"]
                for key in arrays:
                    reference = np.load(
                        VIDEO_SMOKE / (name + "-" + key + ".npy"), mmap_mode="r", allow_pickle=False
                    )
                    np.testing.assert_array_equal(arrays[key][i], reference[index])
                matches += 1
            count += 1
        guard()
    if count != 11151 or matches != 21:
        raise ValueError("incomplete independent video validation")
    pins.verify()
    return dict(
        binding=binding,
        verified_video_events=count,
        video_smoke_matches=matches,
        audio_admission=audio["audio_admission"],
        input_pins=dict(pins.hashes),
        input_digest=digest(pins.hashes),
        validation_ids=validation_ids,
    )
