"""Development-only uploader split; official AQA test remains unopened."""

import hashlib
import html
import unicodedata


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", html.unescape(value)).casefold().split())


def split_questions(rows, seed=17):
    """Assign by uploader hash before inspecting labels; reject ambiguous provenance.

    Generic question text may recur for different audio. Content identity here
    is source ID plus normalized question, not question text alone. Waveform
    identity must additionally be checked after acquiring media.
    """
    result = {name: [] for name in ("train", "calibration", "validation")}
    owners, filenames, seen = {}, {}, set()
    for row in sorted(rows, key=lambda r: (r["source_id"], r["question"])):
        source, uploader = row["source_id"], normalized(row["uploader"])
        filename = normalized(row["file_name"])
        if not source.isdecimal() or not uploader or not filename:
            raise ValueError("missing source identity")
        if source in owners and owners[source] != uploader:
            raise ValueError("source has conflicting uploaders")
        if filename in filenames and filenames[filename] != source:
            raise ValueError("filename maps to multiple sources")
        owners[source], filenames[filename] = uploader, source
        identity = (source, normalized(row["question"]))
        if not identity[1] or identity in seen:
            raise ValueError("duplicate or empty normalized question identity")
        seen.add(identity)
        votes = row["votes"]
        if len(votes) != 3 or not set(votes) <= {"yes", "no"}:
            raise ValueError("three explicit binary votes required")
        if row["target"] != int(votes.count("yes") >= 2):
            raise ValueError("target disagrees with majority")
        bucket = int(hashlib.sha256(f"{seed}:{uploader}".encode()).hexdigest(), 16) % 100
        split = "train" if bucket < 70 else "calibration" if bucket < 85 else "validation"
        result[split].append(dict(row))
    return result
