"""Audit pinned public source metadata and one ZIP directory, without media reads."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

BASE = Path("C:/CodexArtifacts/pretrained-reuse-r2/robomnist-source-audit-01")
FOLDER = r"(?P<index>\d{2})_Ar(?P<robot>[12])_V(?P<velocity>high|med|low)_Ac(?P<action>[0-9])"
MEMBER = re.compile(
    FOLDER
    + r"/Rp(?P<repetition>\d{2})_Ac(?P<file_action>[0-9])_V(?P<file_velocity>high|med|low)"
    + r"_Ar(?P<file_robot>[12])_UNC(?P<uncertainty>\d+)_Rx(?P<receiver>all|[123])"
    + r"_(?P<kind>cam|mic|csi|pos)\.(?P<extension>mp4|wav|json)"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_class(index, robot, velocity, action):
    expected = {"high": 0, "med": 20, "low": 40}[velocity] + (int(robot) - 1) * 10 + int(action)
    if int(index) != expected:
        raise ValueError("folder class index/labels disagree")


def group_members(names, *, expected_folder=None):
    groups = {}
    seen = set()
    for name in names:
        if name.startswith("__MACOSX/") or name.endswith("/"):
            continue
        if name in seen:
            raise ValueError("duplicate real ZIP member")
        seen.add(name)
        match = MEMBER.fullmatch(name)
        if not match:
            raise ValueError("unknown or unsafe source member")
        if expected_folder is not None and name.split("/")[0] != expected_folder:
            raise ValueError("member folder does not match the inspected source archive")
        m = match.groupdict()
        require_class(m["index"], m["robot"], m["velocity"], m["action"])
        if any(m[k] != m[f"file_{k}"] for k in ["robot", "velocity", "action"]):
            raise ValueError("folder/file source labels disagree")
        kind = m["kind"]
        extension = {"cam": "mp4", "mic": "wav", "csi": "json", "pos": "json"}[kind]
        if m["extension"] != extension or (kind in ["csi", "pos"]) != (m["receiver"] == "all"):
            raise ValueError("receiver/type/extension mismatch")
        gid = (
            f"Ar{m['robot']}_V{m['velocity']}_Ac{m['action']}"
            f"_Rp{m['repetition']}_UNC{m['uncertainty']}"
        )
        g = groups.setdefault(gid, {})
        key = (kind, m["receiver"])
        if key in g:
            raise ValueError("multiple members for same physical group and stream")
        g[key] = name
    expected = {(kind, str(rx)) for kind in ["cam", "mic"] for rx in [1, 2, 3]} | {
        ("csi", "all"),
        ("pos", "all"),
    }
    if not groups or any(set(g) != expected for g in groups.values()):
        raise ValueError("all eight synchronized source members required per repetition")
    return [
        dict(
            group_id=gid,
            input_members=sorted(name for key, name in groups[gid].items() if key[0] != "pos"),
            supervision_only_member=groups[gid][("pos", "all")],
        )
        for gid in sorted(groups)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new source report required")
    receipt = json.loads((BASE / "source-receipt.json").read_bytes())
    current = json.loads((BASE / "article.json").read_bytes())
    version = json.loads((BASE / "article-version-1.json").read_bytes())
    directory = json.loads((BASE / "first-archive-directory.json").read_bytes())
    if (
        sha(BASE / "article.json") != receipt["article_sha256"]
        or sha(BASE / "upstream-README.md") != receipt["readme_sha256"]
        or sha(BASE / "article-version-1.json") != directory["version_metadata_sha256"]
        or version["version"] != 1
        or version["id"] != 28179383
        or version["files"] != current["files"]
        or version["license"]["url"] != "https://creativecommons.org/licenses/by/4.0/"
        or directory["media_members_opened"]
        or directory["full_archive_md5_verified"]
        or directory["transferred_bytes"] > 524288
        or directory["archive"] not in version["files"]
        or directory["archive"]["name"] != "00_Ar1_Vhigh_Ac0.zip"
    ):
        raise ValueError("pinned release/directory-only source receipt required")
    classes = []
    for f in version["files"]:
        match = re.fullmatch(FOLDER + r"\.zip", f["name"])
        if not match or f["size"] <= 0:
            raise ValueError("unknown released archive")
        m = match.groupdict()
        require_class(m["index"], m["robot"], m["velocity"], m["action"])
        classes.append(int(m["index"]))
    if sorted(classes) != list(range(60)):
        raise ValueError("complete60 source archive classes required")
    names = [m["name"] for m in directory["members"] if not m["directory"]]
    groups = group_members(names, expected_folder=directory["archive"]["name"].removesuffix(".zip"))
    real = [name for name in names if not name.startswith("__MACOSX/")]
    result = dict(
        status="PASS_SOURCE_METADATA_FEASIBILITY_ONLY",
        source="RoboMNIST",
        doi=version["doi"],
        license=version["license"],
        released_media_rights="ALLOW: official dataset CC BY4.0; retain attribution",
        training_admission="NOT_ADMITTED: split/content/sync and complementarity gates open",
        upstream_commit=receipt["upstream_commit"],
        released_archives=len(version["files"]),
        released_bytes=sum(f["size"] for f in version["files"]),
        missing_computed_md5=[f["name"] for f in version["files"] if not f["computed_md5"]],
        inspected_archive=directory["archive"]["name"],
        inspected_classes=1,
        inspected_real_members=len(real),
        inspected_repetition_groups=len(groups),
        actual_member_extensions=dict(Counter(Path(n).suffix for n in real)),
        macos_resource_forks_excluded=len(names) - len(real),
        transferred_directory_bytes=directory["transferred_bytes"],
        repetition_grouping="same class/repetition/uncertainty, all cameras/mics/CSI together",
        position_member_policy="supervision only; never runtime input",
        prospective_decisions=["which robot moved", "which digit trajectory", "which speed level"],
        limits=[
            "Single controlled robot lab; no general Omni or cross-household quality claim",
            "Only one archive directory inspected; other59 group completeness UNKNOWN",
            "Payload timing/schema/content/session grouping and OOD splits unverified",
            "No joint advantage assumed; require singleton, question-only, shuffle controls",
        ],
        training=False,
        model_inference=False,
        media_members_opened=False,
        quality_measured=False,
        sealed_audit_opened=False,
        validated_hashes={
            str(BASE / name): sha(BASE / name)
            for name in [
                "article.json",
                "article-version-1.json",
                "source-receipt.json",
                "upstream-README.md",
                "first-archive-directory.json",
            ]
        },
        auditor_sha256=sha(__file__),
    )
    with args.output.open("x", encoding="utf8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    with args.output.with_suffix(".groups.json").open("x", encoding="utf8") as f:
        json.dump(groups, f, indent=2)
        f.write("\n")
    print(result["status"], "real_members", len(real), "groups", len(groups))


if __name__ == "__main__":
    main()
