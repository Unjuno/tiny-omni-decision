"""Copy training encounters with reproducible candidate permutations only."""

import copy
import random


def shuffled_options(rows, seed=17):
    rng = random.Random(seed)
    result = []
    for source in rows:
        if len(set(source["options"])) != len(source["options"]):
            raise ValueError("duplicate option")
        if source["target"] not in source["options"]:
            raise ValueError("missing target")
        row = copy.deepcopy(source)
        rng.shuffle(row["options"])
        result.append(row)
    return result
