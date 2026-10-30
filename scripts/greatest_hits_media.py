"""Fixed media selection and exhaustive window/content integrity contracts."""

from scripts.greatest_hits_source import MEMBER_LIMIT


def planned_media(plan, entries):
    ids = sum(plan.values(), [])
    if len(set(ids)) != len(ids) or set(plan) != {"train", "validation", "quarantine"}:
        raise ValueError("complete disjoint development partition required")
    result = []
    for split in ("train", "validation"):
        for ident in sorted(plan[split]):
            name = "vis-data-256/" + ident + "_denoised.mp4"
            entry = entries[name]
            if (entry["name"] != name or not 0 < entry["bytes"] <= MEMBER_LIMIT
                    or not 0 < entry["compressed"] <= MEMBER_LIMIT):
                raise ValueError("exact bounded media entry required")
            result.append(dict(parent=ident, split=split, entry=entry))
    if sum(30 + 512 + 4096 + r["entry"]["compressed"] for r in result) > 2**30:
        raise ValueError("aggregate1GiB bound exceeded")
    return result


def window_integrity(rows, audio_span, video_span):
    bad_onsets, bad_windows = [], []
    for index, row in enumerate(rows):
        t = row["seconds"]
        if not all(start <= t < end for start, end in [audio_span, video_span]):
            bad_onsets.append(index)
        if not all(start <= t - 0.25 and t + 0.25 <= end
                   for start, end in [audio_span, video_span]):
            bad_windows.append(index)
    return dict(rows=len(rows), valid_windows=len(rows) - len(bad_windows),
                out_of_bounds_onsets=bad_onsets, out_of_bounds_windows=bad_windows)


def reject_cross_split_content(rows):
    owners = {}
    for row in rows:
        for field in ("audio_sha256", "video_sha256"):
            key = (field, row[field])
            owner = owners.setdefault(key, row["split"])
            if owner != row["split"]:
                raise ValueError(f"cross-split decoded {field} overlap")
