"""Fixed optimizer-update budget and exposure accounting."""


def budget_geometry(*, examples: int, batch_size: int, updates: int):
    if examples <= 0 or batch_size <= 0 or updates <= 0:
        raise ValueError("budget inputs must be positive")
    sizes = [min(batch_size, examples - start) for start in range(0, examples, batch_size)]
    full, partial = divmod(updates, len(sizes))
    return {
        "batches_per_pass": len(sizes),
        "full_passes": full,
        "partial_pass_batches": partial,
        "sample_exposures": full * examples + sum(sizes[:partial]),
    }


def evaluation_steps(updates: int):
    if updates <= 0 or updates % 88:
        raise ValueError("updates must be a positive multiple of 88")
    return [1, *range(88, updates + 1, 88)]
