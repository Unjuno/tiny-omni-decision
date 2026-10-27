"""Group connected observation observers and photo photographers before splitting."""

import hashlib


def assign_components(rows):
    """Split fully identified connected components; retain unknowns separately.

    Both keys must be produced by the same identity scheme from verified source
    IDs. This does not verify those IDs, media bytes, rights, or source sessions.
    Adding rows can merge components, so freeze the entire input before using it.
    """
    parent = {}

    def root(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            key = parent[key]
        return key

    valid, quarantine = [], []
    for row in rows:
        obs, photo = row.get("observer_key"), row.get("photographer_key")
        if not all(isinstance(x, str) and x.strip() for x in (obs, photo)):
            quarantine.append(dict(row, quarantine_reason="missing_identity"))
            continue
        a, b = root(obs), root(photo)
        if a != b:
            lo, hi = sorted((a, b))
            parent[hi] = lo
        valid.append(row)
    members = {}
    for key in parent:
        members.setdefault(root(key), []).append(key)
    group_keys = {r: hashlib.sha256(
        ("inat-connected-v1:" + "\n".join(sorted(keys))).encode()
    ).hexdigest() for r, keys in members.items()}
    assigned = []
    for row in valid:
        key = group_keys[root(row["observer_key"])]
        split = "train" if int(key[:16], 16) % 100 < 80 else "validation"
        assigned.append(dict(row, component_key=key, split=split))
    return assigned, quarantine
