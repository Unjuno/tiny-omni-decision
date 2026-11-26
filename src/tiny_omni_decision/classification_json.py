"""Serialize fixed-taxonomy probabilities with user-supplied option IDs."""

import math


def classification_json(probabilities, class_ids, options):
    """Complete class set only; no renormalization or generative JSON decoder."""
    classes = list(class_ids)
    p = [float(value) for value in probabilities]
    if (len(classes) < 2 or len(classes) != len(p) or len(set(classes)) != len(classes)
            or any(not isinstance(c, str) or not c.strip() for c in classes)):
        raise ValueError("unique named classes and aligned probabilities required")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) or abs(sum(p) - 1) > 1e-6:
        raise ValueError("finite normalized probabilities required")
    if any(set(o) != {"option_id", "class_id"} for o in options):
        raise ValueError("unsupported option fields")
    ids = [o["option_id"] for o in options]
    supplied = [o["class_id"] for o in options]
    if (len(options) != len(classes) or set(supplied) != set(classes)
            or len(set(ids)) != len(ids)
            or any(not isinstance(i, str) or not i.strip() for i in ids)):
        raise ValueError("all classes and unique nonempty option IDs required")
    scores = dict(zip(classes, p, strict=True))
    selected = max(sorted(classes), key=scores.__getitem__)
    return {"schema_version": "1.0", "status": "selected",
            "selected_option_id": next(o["option_id"] for o in options
                                       if o["class_id"] == selected),
            "options": [{"option_id": o["option_id"], "probability": scores[o["class_id"]]}
                        for o in options]}
