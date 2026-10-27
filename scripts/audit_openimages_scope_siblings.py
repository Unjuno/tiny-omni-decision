"""Check frozen Open Images targets have mutually valid sibling classes."""

import json
from pathlib import Path

from scripts.audit_openimages_target_scope_shortcut import TARGET_SHA, TARGETS
from scripts.train_source_unimodal_cv import digest
from tiny_omni_decision.openimages_options import build_sibling_groups

HIERARCHY = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/openimages-freshval-20261026-attempt-01/source/hierarchy.json"
)
HIERARCHY_SHA = "a28f06e83c8378c4285d92d768ceda7568a48cac3c02994b6c346df804e7d612"


def inspect(scope, groups):
    if not scope or any(not isinstance(k, str) for k in scope):
        raise ValueError("invalid target scope")
    return {label: len(set(groups.get(label, ())) & (scope - {label})) for label in sorted(scope)}


def main():
    if digest(TARGETS) != TARGET_SHA or digest(HIERARCHY) != HIERARCHY_SHA:
        raise ValueError("pinned scope or hierarchy mismatch")
    targets = json.loads(TARGETS.read_bytes())["selected_target_classes"]
    names = {r["label_id"]: r["display_name"] for r in targets}
    groups = build_sibling_groups(json.loads(HIERARCHY.read_bytes()))
    counts = inspect(set(names), groups)
    print(json.dumps({names[label]: count for label, count in counts.items()}, indent=2))


if __name__ == "__main__":
    main()
