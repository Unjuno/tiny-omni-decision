"""Small helpers shared by offline ternary precision diagnostics."""

from __future__ import annotations


def component_for_target(name: str) -> str:
    root = name.split(".", maxsplit=1)[0]
    if root in {"audio_tower", "embed_audio"}:
        return "audio_path"
    if root in {"vision_tower", "embed_vision"}:
        return "vision_path"
    if root == "language_model":
        return "shared_decoder"
    raise ValueError(f"unclassified ternary target root: {root}")


def group_decoder_layer_targets(targets: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Split decoder block targets into contiguous early/middle/late thirds."""
    import re

    pattern = re.compile(r"^language_model\.layers\.(\d+)\.")
    by_layer: dict[int, list[str]] = {}
    for name in targets:
        if not name.startswith("language_model.layers."):
            continue
        match = pattern.match(name)
        if match is None:
            raise ValueError(f"invalid decoder layer target path: {name}")
        by_layer.setdefault(int(match.group(1)), []).append(name)
    layer_ids = sorted(by_layer)
    if not layer_ids or layer_ids != list(range(layer_ids[-1] + 1)):
        raise ValueError("decoder layer target IDs must be contiguous from zero")
    if len(layer_ids) % 3:
        raise ValueError("decoder layer count must divide into three equal groups")
    width = len(layer_ids) // 3
    result = {}
    for group_index, group_name in enumerate(("early", "middle", "late")):
        selected_ids = layer_ids[group_index * width : (group_index + 1) * width]
        result[group_name] = tuple(
            name for layer_id in selected_ids for name in sorted(by_layer[layer_id])
        )
    flattened = [name for names in result.values() for name in names]
    if len(flattened) != len(set(flattened)):
        raise ValueError("decoder depth groups overlap")
    return result
