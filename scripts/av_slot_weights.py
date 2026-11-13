"""Six TRAIN-only simplex coefficients over full native posterior agreements."""

import numpy as np

from scripts.lombard_correspondence import BOUNDS, choice_logps, native_logps


def agreement_terms(audio_logits, candidate_video_logits, audio_temperatures, video_temperatures):
    a = native_logps(audio_logits, audio_temperatures)
    given = np.asarray(candidate_video_logits)
    if given.shape != (len(a), 4, 51):
        raise ValueError("four full native video posteriors required")
    v = native_logps(given.reshape(-1, 51), video_temperatures).reshape(len(a), 4, 51)
    terms = np.empty((len(a), 4, 6), np.float64)
    for i, (start, end) in enumerate(zip(BOUNDS[:-1], BOUNDS[1:], strict=True)):
        joint = a[:, start:end][:, None, :] + v[:, :, start:end]
        maximum = joint.max(-1)
        terms[:, :, i] = maximum + np.log(np.exp(joint - maximum[:, :, None]).sum(-1))
    if not np.isfinite(terms).all():
        raise ValueError("finite log-domain native terms required")
    return terms


def weighted_scores(terms, weights):
    x, w = np.asarray(terms, np.float64), np.asarray(weights, np.float64)
    if (
        x.ndim != 3
        or not len(x)
        or x.shape[1:] != (4, 6)
        or not np.isfinite(x).all()
        or w.shape != (6,)
        or not np.isfinite(w).all()
        or (w < 0).any()
    ):
        raise ValueError(
            "finite full four-choice/six-term arrays and nonnegative coefficients required"
        )
    score = np.zeros(x.shape[:2], np.float64)
    for i in range(6):
        score += x[:, :, i] * w[i]
    return score


def loss_gradient(terms, targets, weights):
    x, y = np.asarray(terms), np.asarray(targets)
    if (
        y.shape != (len(x),)
        or not np.issubdtype(y.dtype, np.integer)
        or (y < 0).any()
        or (y >= 4).any()
    ):
        raise ValueError("TRAIN quartet-choice labels required")
    logp = choice_logps(weighted_scores(x, weights))
    loss = float(-logp[np.arange(len(y)), y].mean())
    residual = np.exp(logp)
    residual[np.arange(len(y)), y] -= 1
    gradient = np.einsum("nc,ncd->d", residual, x) / len(x)
    if not np.isfinite(gradient).all():
        raise ValueError("finite analytic gradient required")
    return loss, gradient


def project_simplex(value):
    v = np.asarray(value, np.float64)
    if v.shape != (6,) or not np.isfinite(v).all():
        raise ValueError("finite six-vector required")
    u = np.sort(v)[::-1]
    s = np.cumsum(u) - 6
    idx = np.arange(1, 7)
    rho = np.flatnonzero(u - s / idx > 0)[-1]
    threshold = s[rho] / (rho + 1)
    return np.maximum(v - threshold, 0)


def fit_coefficients(train_terms, train_targets, *, steps=512, monitor=None):
    if not isinstance(steps, int) or not 1 <= steps <= 512:
        raise ValueError("bounded fixed iterations required")
    weights = np.ones(6)
    loss, gradient = loss_gradient(train_terms, train_targets, weights)
    history = [
        dict(
            iteration=0,
            weights=weights.tolist(),
            nll=loss,
            accepted=False,
            changed=False,
            step_size=0.0,
            gradient_norm=float(np.linalg.norm(gradient)),
            backtracks=0,
        )
    ]
    for iteration in range(1, steps + 1):
        if monitor is not None:
            monitor()
        accepted = False
        step = 1.0
        for _backtracks in range(24):
            candidate = project_simplex(weights - step * gradient)
            current, new_gradient = loss_gradient(train_terms, train_targets, candidate)
            if current <= loss + 0.0001 * np.dot(gradient, candidate - weights):
                accepted = True
                break
            step *= 0.5
        changed = accepted and not np.array_equal(weights, candidate)
        if accepted:
            weights, loss, gradient = candidate, current, new_gradient
        if not np.isclose(weights.sum(), 6, rtol=0, atol=1e-12):
            raise ValueError("simplex sum drift")
        history.append(
            dict(
                iteration=iteration,
                weights=weights.tolist(),
                nll=loss,
                accepted=accepted,
                changed=changed,
                step_size=step if accepted else 0.0,
                gradient_norm=float(np.linalg.norm(gradient)),
                backtracks=_backtracks if accepted else 24,
            )
        )
    return weights, history
