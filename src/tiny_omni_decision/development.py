from __future__ import annotations

import hashlib
from collections import defaultdict

from .corpus import source_asset_identity
from .schema import DecisionExample


def deterministic_asset_holdout(
    examples: list[DecisionExample],
    *,
    modality: str,
    count: int,
    seed: int,
) -> tuple[list[DecisionExample], set[tuple[str, str]]]:
    """Select one example per asset, balanced round-robin across sources.

    The returned source/asset keys identify whole groups that must be removed
    from training, including any additional questions tied to a selected asset.
    """
    if count < 1:
        raise ValueError("asset holdout count must be positive")
    grouped: dict[str, dict[str, list[DecisionExample]]] = defaultdict(lambda: defaultdict(list))
    for example in examples:
        if example.modality != modality:
            continue
        asset = source_asset_identity(example)
        grouped[example.source][asset].append(example)
    if sum(len(assets) for assets in grouped.values()) < count:
        raise ValueError(
            f"{modality} has only {sum(len(assets) for assets in grouped.values())} "
            f"eligible source/asset groups; {count} are required"
        )

    ordered_by_source: dict[str, list[str]] = {}
    for source, assets in grouped.items():
        ordered_by_source[source] = sorted(
            assets,
            key=lambda asset: hashlib.sha256(
                f"{seed}\0{modality}\0{source}\0{asset}".encode()
            ).digest(),
        )
    positions = dict.fromkeys(ordered_by_source, 0)
    selected: list[DecisionExample] = []
    selected_groups: set[tuple[str, str]] = set()
    while len(selected) < count:
        progressed = False
        for source in sorted(ordered_by_source):
            position = positions[source]
            assets = ordered_by_source[source]
            if position >= len(assets):
                continue
            asset = assets[position]
            positions[source] = position + 1
            rows = grouped[source][asset]
            selected_row = min(
                rows,
                key=lambda item: hashlib.sha256(
                    f"{seed}\0{modality}\0sample\0{item.id}".encode()
                ).digest(),
            )
            selected.append(selected_row)
            selected_groups.add((source, asset))
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise RuntimeError("asset-group holdout exhausted its candidate pool")
    return selected, selected_groups
