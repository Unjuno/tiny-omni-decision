"""Fixed semantic serialization for the audited occlusion question family."""

PREFIX = "The person uses multiple similar objects to play an occlusion game. "
QUESTIONS = {
    PREFIX + f"Where is the hidden object at the {time} of the game "
    "from the person's point of view?": time
    for time in ("beginning", "end")
}
OPTIONS = frozenset(
    "Under " + phrase + "."
    for phrase in (
        "the closest object to the person",
        "the farthest object from the person",
        "the first object from the left",
        "the middle object",
        "the second closest object to the person",
        "the second object from the left",
        "the third closest object to the person",
        "the third object from the left",
        "the fourth object from the left",
    )
)


def occlusion_statement(question: str, candidate: str) -> str:
    """Preserve time, viewpoint and candidate relation; never inspect labels."""
    if question not in QUESTIONS or candidate not in OPTIONS:
        raise ValueError("unsupported occlusion question/candidate")
    return (
        "Occlusion game with similar objects. "
        f"At the {QUESTIONS[question]}, from the person's viewpoint: "
        f"the hidden object is {candidate[0].lower() + candidate[1:]}"
    )
