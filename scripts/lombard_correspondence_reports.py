"""Exact finite JSON-domain report equality; no numeric rounding or tolerances."""

import json


def same_report(current, saved):
    return json.loads(json.dumps(current, allow_nan=False)) == json.loads(
        json.dumps(saved, allow_nan=False)
    )
