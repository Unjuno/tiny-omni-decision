"""Inference-only scoring core for the frozen AV replacement protocol.

The caller must admit inputs and freeze/bind the selected checkpoint before use.
This module does not select models or provide a production admission CLI.
"""

from collections import defaultdict

import numpy as np
import torch

from scripts.greatest_hits_replacement import build_inventory, inputs, sensitivity
from tiny_omni_decision.native_av_interaction import ConditionalAVReadout


def aggregate(records):
    """Question-weighted diagnostics with explicit correlated recording groups."""
    def summary(items):
        return dict(
            count=len(items),
            prediction_flip_rate=float(np.mean([r["prediction_flip"] for r in items])),
            mean_total_variation=float(np.mean([r["total_variation"] for r in items])),
            mean_js_divergence=float(np.mean([r["js_divergence"] for r in items])),
            mean_confidence_change=float(np.mean([r["confidence_change"] for r in items])),
            mean_absolute_confidence_change=float(np.mean(
                [abs(r["confidence_change"]) for r in items])),
        )

    result = {"overall": summary(records)}
    for field in ("task", "date", "parent"):
        groups = defaultdict(list)
        for record in records:
            groups[record[field]].append(record)
        result["by_" + field] = {key: summary(groups[key]) for key in sorted(groups)}
    return result


def score(model, store, rows, inventory, *, emit, expected_natural=None, guard=lambda: None):
    """Emit all five views and return target-free paired sensitivity summaries.

    Rows may arrive in any order; frozen inventory order governs output. Optional
    natural logits must be keyed by exact question ID in original option order,
    without padding. Self swaps must reproduce logits AND gate values exactly.
    """
    if inventory != build_inventory(rows):
        raise ValueError("complete frozen inventory differs from validation metadata")
    by_id = {r["id"]: r for r in rows}
    for row in rows:
        bound = store.binding["questions"].get(row["id"])
        if bound is None or bound["options"] != row["options"]:
            raise ValueError("store identities/options differ from frozen metadata")
    if expected_natural is not None and set(expected_natural) != set(by_id):
        raise ValueError("natural reference identities differ")
    diagnostic = isinstance(model, ConditionalAVReadout)
    changes = {key: [] for key in ("donor_audio", "donor_video")}
    calls, exact = 0, 0
    model.eval()
    with torch.inference_mode():
        for pair in inventory["pairs"]:
            row = by_id[pair["id"]]
            natural, natural_gates = None, None
            for view in inventory["views"]:
                guard()
                donor = None
                if view == "natural":
                    blocks, q = store.get(pair["id"])
                else:
                    kind, modality = view.split("_")
                    donor = pair["id"] if kind == "self" else pair["donor_id"]
                    blocks, q = inputs(store, pair["id"], donor, modality)
                if diagnostic:
                    values, detail = model(blocks, q, diagnostics=True)
                    gate = detail["gates"]
                    if (not detail["active"] or gate is None
                            or gate.shape != (len(row["options"]), 2)
                            or not torch.isfinite(gate).all()
                            or ((gate < 0) | (gate > 1)).any()):
                        raise ValueError("complete finite active joint gates required")
                    gates = gate.cpu().numpy().copy()
                else:
                    values, gates = model(blocks, q), None
                calls += 1
                if values.shape != (len(row["options"]),) or not torch.isfinite(values).all():
                    raise ValueError("complete finite replacement logits required")
                logits = values.cpu().numpy().copy()
                if view == "natural":
                    natural, natural_gates = logits, gates
                    if expected_natural is not None and not np.array_equal(
                            logits, expected_natural[pair["id"]]):
                        raise ValueError("natural logits differ from selected model reference")
                elif view.startswith("self_"):
                    if not np.array_equal(natural, logits) or not np.array_equal(
                            natural_gates, gates):
                        raise ValueError("self replacement changed logits or gates")
                    exact += 1
                else:
                    measures = sensitivity(natural[None], logits[None])["per_example"]
                    changes[view].append(dict(
                        pair, **{key: value[0] for key, value in measures.items()}))
                emit(dict(pair, view=view, input_donor_id=donor, options=list(row["options"]),
                          logits=logits.tolist(), gates=None if gates is None else gates.tolist()))
    guard()
    return dict(
        questions=len(rows), model_forwards=calls, self_replacements_exact=exact,
        pairs_sha256=inventory["pairs_sha256"], optimizer_updates=0,
        counterfactual_truth_verified=False,
        sensitivity={key: aggregate(items) for key, items in changes.items()},
        per_example=changes,
    )
