"""Target-independent candidate presentation for a small instruction decoder."""

import string


def option_labels(count):
    if not 2 <= count <= 26:
        raise ValueError("require two to twenty-six options")
    return list(string.ascii_uppercase[:count])


def decision_prompt(row):
    labels = option_labels(len(row["options"]))
    options = "\n".join(
        f"{label}: {text}" for label, text in zip(labels, row["options"], strict=True)
    )
    return (
        f"State: {row['state']}\nQuestion: {row['question']}\nOptions:\n{options}\n"
        "Return only the option label."
    )
