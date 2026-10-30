import numpy as np
import pytest


def test_equal_coefficients_reproduce_full_native_reference_exactly():
    from scripts.av_slot_weights import agreement_terms, weighted_scores
    from scripts.lombard_correspondence import agreement_scores

    g = np.random.default_rng(17)
    a = g.normal(size=(8, 51)).astype(np.float32)
    v = g.normal(size=(8, 4, 51)).astype(np.float32)
    t = np.geomspace(0.25, 4, 6)
    np.testing.assert_array_equal(
        weighted_scores(agreement_terms(a, v, t, t), np.ones(6)), agreement_scores(a, v, t, t)
    )


def test_analytic_gradient_matches_finite_differences_and_inputs_unchanged():
    from scripts.av_slot_weights import loss_gradient

    x = np.random.default_rng(17).normal(size=(12, 4, 6))
    before = x.copy()
    y = np.arange(12) % 4
    w = np.ones(6)
    loss, g = loss_gradient(x, y, w)
    actual = []
    for i in range(6):
        delta = np.zeros(6)
        delta[i] = 1e-6
        actual.append(
            (loss_gradient(x, y, w + delta)[0] - loss_gradient(x, y, w - delta)[0]) / 2e-6
        )
    np.testing.assert_allclose(g, actual, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(x, before)
    assert np.isfinite(loss)


def test_simplex_fit_fixed_iterations_monotonic_and_no_validation_argument():
    from scripts.av_slot_weights import fit_coefficients, project_simplex, weighted_scores

    np.testing.assert_allclose(project_simplex(np.array([-1, 0, 1, 2, 3, 4.0])), [0, 0, 0, 1, 2, 3])
    x = np.zeros((8, 4, 6))
    y = np.arange(8) % 4
    x[np.arange(8), y, 0] = 4
    x[np.arange(8), (y + 1) % 4, 1] = 4
    weights, history = fit_coefficients(x, y, steps=32)
    assert len(history) == 33 and history[-1]["iteration"] == 32
    assert weights[0] > weights[1] and (weights >= 0).all()
    assert np.isclose(weights.sum(), 6)
    assert all(a["nll"] >= b["nll"] - 1e-12 for a, b in zip(history, history[1:], strict=False))
    assert (weighted_scores(x, weights).argmax(1) == y).all()
    with pytest.raises(TypeError):
        fit_coefficients(x, y, validation=x)
    with pytest.raises(ValueError):
        weighted_scores(x, np.full(6, np.nan))


def test_paired_bootstrap_groups_and_candidate_gate_are_explicit():
    from scripts.av_slot_weight_metrics import eligible, paired_comparison

    y = np.tile(np.arange(4), 2)
    p = np.eye(4)[y] * 0.8 + 0.05
    b = np.full_like(p, 0.25)
    lp, bp = np.log(p), np.log(b)
    p, b = np.exp(lp), np.exp(bp)
    result = paired_comparison(p, lp, b, bp, y, ["one"] * 4 + ["two"] * 4)
    assert result["groups"] == 2 and result["replicates"] == 10000
    assert result["differences"]["accuracy"]["lower"] == 0.75
    baseline = dict(accuracy=0.8, nll=0.5, brier=0.3, ece=0.02)
    candidate = dict(accuracy=0.81, nll=0.49, brier=0.29, ece=0.024)
    assert eligible(candidate, baseline, [])
    assert not eligible(dict(candidate, ece=0.03), baseline, [])
    assert not eligible(candidate, baseline, [dict(count=100, accuracy_delta=-0.03)])


def test_objective_accounting_includes_initialization_and_every_retry(monkeypatch):
    from scripts import av_slot_weights as core

    x = np.random.default_rng(17).normal(size=(12, 4, 6))
    y = np.arange(12) % 4
    original = core.loss_gradient
    calls = []

    def tracked(*args):
        calls.append(len(args[1]))
        return original(*args)

    monkeypatch.setattr(core, "loss_gradient", tracked)
    _, history = core.fit_coefficients(x, y, steps=8)
    evaluations = 1 + sum(h["backtracks"] + 1 if h["accepted"] else 24 for h in history[1:])
    assert len(calls) == evaluations and sum(calls) == 12 * evaluations
    assert sum(calls) > 8 * 12
