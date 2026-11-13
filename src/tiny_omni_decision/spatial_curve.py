"""Fail-closed prefix comparison for the fixed spatial learning curve."""


def require_prefix(current, reference):
    if len(current) != 8 or len(reference) != 8 or current != reference:
        raise ValueError("eight-epoch prefix differs; do not extend this run")
