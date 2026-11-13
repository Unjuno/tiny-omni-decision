"""Train-only proposal adapter; media and final task admission remain external."""

import math


def visibility_records(video_id, row, cut_frame):
    metadata = row["metadata"]
    if metadata["split"] != "train" or metadata["video_id"] != video_id:
        raise ValueError("train-only matching video identity required")
    fps = metadata["frame_rate"]
    rate = metadata["audio_sample_rate"]
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(rate) or rate <= 0:
        raise ValueError("positive finite rates required")
    if type(cut_frame) is not int or not -1 <= cut_frame < metadata["num_frames"]:
        raise ValueError("explicit valid upstream cut required")
    # Conservative half-open boundary: the cut frame itself is not consumed.
    video_end = (metadata["num_frames"] if cut_frame == -1 else cut_frame) / fps * 1e6
    audio_end = metadata["audio_samples"] / rate * 1e6
    end_limit = min(video_end, audio_end)
    sounds = row["sound_localisation"]
    if len({s["id"] for s in sounds}) != len(sounds):
        raise ValueError("duplicate annotation IDs")
    for sound in sounds:
        start, end = sound["timestamps"]
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or type(sound["is_visible"]) is not int
            or sound["is_visible"] not in (0, 1)
        ):
            raise ValueError("invalid sound interval or target")
    for sound in sounds:
        start, end = sound["timestamps"]
        reasons = []
        if end > end_limit:
            reasons.append("outside_conservative_media_or_cut_boundary")
        conflicting = [
            other["id"]
            for other in sounds
            if other["label"] == sound["label"]
            and other["is_visible"] != sound["is_visible"]
            and max(start, other["timestamps"][0]) < min(end, other["timestamps"][1])
        ]
        if conflicting:
            reasons.append("overlapping_same_class_visibility_conflict")
        yield dict(
            record_id=f"{video_id}:sound:{sound['id']}",
            group_id=video_id,
            split="train",
            status="held" if reasons else "candidate",
            hold_reasons=reasons,
            request=dict(
                task="sound_class_source_visibility",
                video_id=video_id,
                interval_us=[start, end],
                sound_class=sound["label"],
                options=[
                    dict(option_id="not_visible", label="not visible"),
                    dict(option_id="visible", label="visible"),
                ],
            ),
            supervision=dict(target_option_id="visible" if sound["is_visible"] else "not_visible"),
            provenance=dict(
                annotation_id=sound["id"],
                upstream_cut_frame=cut_frame,
                conservative_end_us=end_limit,
                conflicting_annotation_ids=conflicting,
            ),
        )
