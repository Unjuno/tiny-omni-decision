"""Exact train-declared question family, no target-dependent admission."""

PREFIX = "The person uses multiple similar objects to play an occlusion game. "
QUESTIONS = {
    "initial": PREFIX
    + "Where is the hidden object at the beginning of the game from the person's point of view?",
    "final": PREFIX
    + "Where is the hidden object at the end of the game from the person's point of view?",
}


def build_pairs(data, split):
    if split not in {"train", "valid"}:
        raise ValueError("development splits only")
    accepted, held = [], []
    for identity, row in sorted(data.items()):
        if row["metadata"]["split"] != split:
            raise ValueError("split mismatch")
        selected = {
            k: [q for q in row["mc_question"] if q["question"] == text]
            for k, text in QUESTIONS.items()
        }
        if not any(selected.values()):
            continue
        if any(len(v) != 1 for v in selected.values()):
            held.append(dict(video_id=identity, reason="missing or duplicate exact question"))
            continue
        first, last = selected["initial"][0], selected["final"][0]
        options = sorted(first["options"])
        if len(options) != 3 or len(set(options)) != 3 or sorted(last["options"]) != options:
            held.append(dict(video_id=identity, reason="incompatible option set"))
            continue
        records = []
        for condition, value in (("initial", first), ("final", last)):
            if value["answer_id"] not in range(3):
                raise ValueError("invalid target")
            records.append(
                dict(
                    condition=condition,
                    question=value["question"],
                    upstream_question_id=value["id"],
                    target=options.index(value["options"][value["answer_id"]]),
                )
            )
        accepted.append(dict(video_id=identity, split=split, options=options, questions=records))
    return dict(pairs=accepted, held=held)
