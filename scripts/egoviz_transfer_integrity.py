"""Validate source identity isolation for the fixed pilot transfer."""


def validate_isolation(pilot_rows, current_rows):
    for key in ("group_id", "chunk_id", "sample_id"):
        overlap = {row[key] for row in pilot_rows} & {row[key] for row in current_rows}
        if overlap:
            raise ValueError(f"{key} overlaps between pilot and current corpus")
    pilot_train = {row["group_id"] for row in pilot_rows if row["split"] == "train"}
    pilot_dev = {row["group_id"] for row in pilot_rows if row["split"] == "development"}
    if len(pilot_train) != 8 or len(pilot_dev) != 8 or pilot_train & pilot_dev:
        raise ValueError("pilot group split differs from frozen inventory")
    return {"pilot_train_groups": len(pilot_train), "pilot_development_groups": len(pilot_dev),
            "current_groups": len({row["group_id"] for row in current_rows}),
            "group_overlap": 0, "chunk_overlap": 0, "sample_overlap": 0}
