"""Replace only complete visual bank metadata without changing observation indices."""

import re
from pathlib import Path


def bind_visual_banks(original, replacement, output_root):
    expected = {key for key in original if key.startswith(("image/", "video/"))}
    if not expected or set(replacement) != expected:
        raise ValueError("all and only original Image/Video bank keys required")
    root = Path(output_root).resolve()
    paths = set()
    changed = False
    for key, bank in replacement.items():
        prior = original[key]
        path = Path(bank["path"]).resolve()
        if (
            set(bank) != {"path", "sha256", "shape"}
            or bank["shape"] != prior["shape"]
            or not re.fullmatch(r"[0-9a-f]{64}", bank["sha256"])
            or not path.is_relative_to(root)
            or path == Path(prior["path"]).resolve()
            or path in paths
        ):
            raise ValueError("new independent visual arrays with unchanged shapes required")
        paths.add(path)
        changed |= bank["sha256"] != prior["sha256"]
    if not changed:
        raise ValueError("visual preprocessing intervention must change features")
    return {**original, **replacement}
