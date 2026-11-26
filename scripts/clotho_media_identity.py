"""Explicit provenance-reviewed archive alias, never generic fuzzy matching."""

import html

ALIAS_SHA256 = "5fb420a29ce8fe73fcd035f2b8366d20f32e095deac57f9328d82f2e848c9a0a"


def media_basename(row):
    name = html.unescape(row["file_name"])
    if name == "souffle_me_tallique.wav":
        if row["source_id"] != "16854" or row["uploader"] != "deegnot":
            raise ValueError("alias provenance mismatch")
        return "souffle_me.tallique.wav"
    return name
