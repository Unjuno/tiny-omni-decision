"""Publish small, hash-bound evidence from the fixed Greatest Hits physical probe."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    repo = Path(__file__).resolve().parents[1]

    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def sha(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    probe = root / "greatest-hits-train-probe-01"
    decode = root / "greatest-hits-train-decode-01"
    acquisition, inspection = load(probe / "result.json"), load(decode / "result.json")
    assert acquisition["status"] == "PASS_FIXED_TRAIN_MEMBER_ACQUISITION"
    assert inspection["status"] == "PASS_DECODE_AND_RAW_LABEL_ACCOUNTING"
    assert sha(probe / "result.json") == inspection["probe_receipt_sha256"]
    for name, item in acquisition["fetched"].items():
        assert sha(probe / name) == item["sha256"]
    members = root / "greatest-hits-directory-01/members.json"
    assert sha(members) == acquisition["directory_sha256"]
    groups = defaultdict(set)
    for item in load(members):
        name = item["name"].split("/")[-1]
        if "_" in name:
            ident, suffix = name.split("_", 1)
            groups[ident].add(suffix)
    expected = {"denoised.mp4", "denoised.wav", "denoised_thumb.mp4", "mic.mp4",
                "mic.wav", "times.txt"}
    assert len(groups) == 977 and all(s == expected for s in groups.values())
    lists = [(probe / (s + ".txt")).read_text().splitlines() for s in ("train", "test")]
    prefixes = [{"-".join(i.split("-")[:6]) for i in ids} for ids in lists]
    shared = prefixes[0] & prefixes[1]
    rows = [r for item in inspection["annotations"].values() for r in item["rows"]]
    result = dict(
        format="bounded_train_source_probe_v1", dataset_id="Greatest-Hits",
        policy="REVIEW", license_policy="ALLOW", training_admission="NOT_ADMITTED",
        license="CC-BY-4.0", source_url="https://andrewowens.com/vis/",
        attribution="Owens, Isola, McDermott, Torralba, Adelson, Freeman; "
                    "Visually Indicated Sounds (CVPR 2016)",
        citation="https://arxiv.org/abs/1512.08512",
        scope="fixed official TRAIN physical probe only; no training corpus or quality benchmark",
        source_version=dict(url=acquisition["url"], bytes=acquisition["archive_size"],
                            etag=acquisition["etag"], full_archive_sha256=None,
                            directory_sha256=sha(members)),
        parent_inventory=dict(parents=len(groups), suffixes=sorted(expected),
                              train=len(lists[0]), test=len(lists[1]), exact_id_overlap=0),
        grouping=dict(timestamp_prefix=inspection["timestamp_prefix"],
                      date_prefix=inspection["date_prefix"],
                      shared_timestamp_prefix_train_children=sum(
                          "-".join(i.split("-")[:6]) in shared for i in lists[0]),
                      shared_timestamp_prefix_test_children=sum(
                          "-".join(i.split("-")[:6]) in shared for i in lists[1]),
                      scene_actor_session_identity="UNKNOWN"),
        selected=acquisition["selected"], selection=acquisition["selection"],
        annotations=dict(rows=len(rows), complete=sum(None not in r["fields"] for r in rows),
                         missing_fields_histogram=dict(Counter(
                             sum(x is None for x in r["fields"]) for r in rows)),
                         raw_label_field_counts=inspection["raw_label_field_counts"]),
        decoded_streams=inspection["streams"],
        physical_av_alignment=inspection["physical_av_alignment"],
        first_parent_annotation_times_within_both_streams=inspection[
            "first_parent_annotation_times_within_both_streams"],
        semantic_joint_necessity="UNMEASURED", material_action_reaction_mapping="INFERRED from "
            "released values and primary paper; no header or impact-location column in these files",
        test_id_metadata_read=True, test_media_or_labels_opened=False, sealed_audit_opened=False,
        model_loaded=False, optimizer_updates=0, cloud_cost=0,
        execution=dict(acquisition_source=acquisition["source_commit"],
                       decode_source=inspection["source_commit"],
                       acquisition_command=acquisition["command"],
                       decode_command=inspection["command"],
                       member_transfer_bytes=acquisition["requested_bytes"],
                       acquisition_seconds=acquisition["seconds"],
                       decode_seconds=inspection["seconds"], python=inspection["python"],
                       pyav=inspection["pyav"], numpy=inspection["numpy"],
                       ffmpeg_libraries=inspection["ffmpeg_libraries"]),
        fetched_members=acquisition["fetched"], artifacts={})
    evidence = [probe / "result.json", decode / "result.json", members,
                root / "greatest-hits-directory-01/result.json",
                root / "joint-source-admission-20261017-01/preflight.json"]
    evidence += sorted((root / "joint-source-admission-20261017-01").glob("*.bin"))
    result["artifacts"] = {str(p): sha(p) for p in evidence}
    target = repo / "manifests/pretrained_reuse/greatest-hits-probe-20261017.json"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(path=str(target), sha256=sha(target))))


if __name__ == "__main__":
    main()
