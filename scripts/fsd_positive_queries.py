"""Convert fixed acquisition records to explicit positive-only auxiliary queries."""

from scripts.fetch_fsd50k_training_smoke import ALLOW
from scripts.probe_starss23_windows import EVENTS

EVENT_MAP = {
    "Bell": "bell", "Clapping": "clapping", "Telephone": "telephone",
    "Walk_and_footsteps": "footsteps", "Musical_instrument": "instrument",
    "Knock": "knock", "Door": "door", "Water_tap_and_faucet": "water tap",
}


def positive_queries(selection):
    """No unknown event becomes a negative; one declared query per whole clip.

    This function validates selected labels, not acoustic equivalence between
    datasets. Caller must verify official-train membership and media provenance.
    """
    result, seen, uploaders = [], set(), set()
    for index, row in enumerate(selection):
        clip_id, uploader = row["id"], row["uploader"]
        if not isinstance(clip_id, str) or not clip_id.isdecimal() or not uploader:
            raise ValueError("valid clip and uploader identities required")
        if clip_id in seen or uploader in uploaders:
            raise ValueError("selection requires distinct clips and uploaders")
        event = row["event"]
        if row["license"] not in ALLOW:
            raise ValueError("unapproved clip license")
        if event not in EVENT_MAP or event not in row["known_positive_labels"].split(","):
            raise ValueError("query event must have an explicit positive label")
        seen.add(clip_id)
        uploaders.add(uploader)
        result.append({"clip_id": clip_id, "feature_index": index,
                       "event_index": EVENTS.index(EVENT_MAP[event]),
                       "options": ["no", "yes"], "target": 1,
                       "scope": "whole_clip", "unknown_events": "omitted"})
    if not result:
        raise ValueError("nonempty positive selection required")
    return result
