"""Target-independent cyclic option balancing, with identity-aligned probabilities."""

import math


def rotations(options):
    if len(options) < 2 or len(set(options)) != len(options):
        raise ValueError("need distinct options")
    return [options[i:] + options[:i] for i in range(len(options))]


def align_mean(options, probabilities):
    orders = rotations(options)
    if len(probabilities) != len(orders):
        raise ValueError("one distribution per rotation required")
    result = dict.fromkeys(options, 0.0)
    for order, probs in zip(orders, probabilities, strict=True):
        if (
            len(probs) != len(options)
            or any(not math.isfinite(p) or p < 0 or p > 1 for p in probs)
            or not math.isclose(sum(probs), 1, abs_tol=1e-5)
        ):
            raise ValueError("invalid probabilities")
        for option, probability in zip(order, probs, strict=True):
            result[option] += probability / len(options)
    return [result[option] for option in options]
