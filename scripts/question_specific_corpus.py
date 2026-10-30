"""Preserve upstream question-specific candidates for a separate corpus audit."""


def convert(source, video_ids, split):
    rows, seen = [], set()
    for video_id in video_ids:
        for question in source[video_id]["mc_question"]:
            key = f"{video_id}:{question['id']}"
            if key in seen:
                raise ValueError("duplicate question identity")
            seen.add(key)
            options = question["options"]
            target = question["answer_id"]
            if len(options) < 2 or len(set(options)) != len(options):
                raise ValueError("invalid candidate list")
            if not isinstance(target, int) or not 0 <= target < len(options):
                raise ValueError("invalid target")
            rows.append(dict(
                id=key, video_id=video_id, split=split,
                upstream_question_id=question["id"], question=question["question"],
                options=list(options), target=target,
                admission="REVIEW_CUT_FRAME_ANSWERABILITY",
            ))
    return rows
