import copy
import hashlib

import numpy as np
import pytest


def rows():
    result = []
    for speaker in ["s1", "s2"]:
        for condition in ["p", "l"]:
            for index in range(6):
                result.append(
                    dict(
                        source_id=f"{speaker}-{condition}-{index}",
                        speaker_group=speaker,
                        condition=condition,
                        split="train",
                        targets=dict(
                            command="bin",
                            color="blue",
                            preposition="at",
                            letter=chr(97 + index),
                            number="1",
                            adverb="now",
                        ),
                    )
                )
    return result


def test_catalog_uses_all_anchors_same_speaker_condition_and_balanced_roles():
    from scripts.lombard_correspondence import build_quartets, query_indices

    sources = rows()
    before = copy.deepcopy(sources)
    for policy in ["random", "hard"]:
        catalog, counts = build_quartets(sources, policy)
        assert catalog.shape == (len(sources), 4)
        assert counts["anchors"] == len(sources)
        assert all(i in catalog[i] for i in range(len(sources)))
        for quartet in catalog:
            assert len(set(quartet)) == 4
            assert (
                len({(sources[i]["speaker_group"], sources[i]["condition"]) for i in quartet}) == 1
            )
        audio, videos, targets = query_indices(catalog)
        assert audio.shape == (len(sources) * 4,)
        assert videos.shape == (len(sources) * 4, 4)
        assert np.all(videos[np.arange(len(audio)), targets] == audio)
        np.testing.assert_array_equal(np.bincount(targets), np.full(4, len(sources)))
        np.testing.assert_array_equal(catalog, build_quartets(sources, policy)[0])
    assert sources == before


def test_catalog_fails_for_ambiguous_small_or_mixed_splits():
    from scripts.lombard_correspondence import build_quartets

    with pytest.raises(ValueError):
        build_quartets(rows()[:3], "hard")
    mixed = rows()
    mixed[0]["split"] = "validation"
    with pytest.raises(ValueError):
        build_quartets(mixed, "random")
    with pytest.raises(ValueError):
        build_quartets(rows(), "scores_based")


def test_semantic_agreement_recognizes_query_without_label_inputs():
    from scripts.lombard_correspondence import agreement_scores, choice_probabilities

    logits = np.zeros((4, 51), np.float32)
    logits[:, 12:37] = -10
    logits[np.arange(4), 12 + np.arange(4)] = 10
    actual = choice_probabilities(
        agreement_scores(logits, np.repeat(logits[None], 4, axis=0), np.ones(6), np.ones(6))
    )
    np.testing.assert_array_equal(actual.argmax(1), np.arange(4))
    assert (actual.max(1) > 0.999).all()


def test_json_order_is_exact_and_invalid_probabilities_fail_closed():
    from scripts.lombard_correspondence import choice_json, choice_metrics

    score = np.array([0.0, 1.0, 2.0, 3.0])
    ids = ["custom-x", "custom-y", "custom-z", "custom-w"]
    assert choice_json(score, ids, 2) == choice_json(score[::-1], ids[::-1], 2)
    with pytest.raises(ValueError):
        choice_metrics(np.full((4, 4), np.nan), np.arange(4))
    with pytest.raises(ValueError):
        choice_json(score, ["same"] * 4, 1)


def test_uniform_bootstrap_keeps_whole_speaker_clusters():
    from scripts.lombard_correspondence import bootstrap_minus_uniform

    probs = np.eye(4)[np.tile(np.arange(4), 2)] * 0.8 + 0.05
    before = hashlib.sha256(probs.tobytes()).hexdigest()
    result = bootstrap_minus_uniform(probs, np.tile(np.arange(4), 2), ["s1"] * 4 + ["s2"] * 4)
    assert result["groups"] == 2 and result["replicates"] == 10000
    assert result["differences"]["accuracy"]["point"] == 0.75
    assert result["differences"]["accuracy"]["lower"] == 0.75
    assert hashlib.sha256(probs.tobytes()).hexdigest() == before


def test_parent_posterior_computation_exactly_matches_frozen_reference():
    pytest.importorskip("torch")
    from scripts.lombard_correspondence import native_logps
    from scripts.lombard_readout_core import _log_probabilities

    logits = np.random.default_rng(17).normal(size=(8, 51)).astype(np.float32)
    temperatures = np.geomspace(0.25, 4, 6)
    np.testing.assert_array_equal(
        native_logps(logits, temperatures), _log_probabilities(logits, temperatures)
    )


def test_counterfactual_snapshot_records_donor_ids_and_corrected_targets(tmp_path):
    pytest.importorskip("torch")
    from scripts.evaluate_lombard_correspondence import snapshot

    parent = rows()[:4]
    y = np.arange(4)
    donor = (y + 1) % 4
    scores = np.full((4, 4), -10.0)
    scores[y, donor] = 10
    report = snapshot(
        tmp_path, "swap", scores, 1, donor, np.tile(np.arange(4), (4, 1)), y, parent, donor
    )
    assert report["original_targets"]["accuracy"] == 0
    assert report["corrected_targets"]["accuracy"] == 1
    with np.load(tmp_path / "swap.npz", allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["corrected_targets"], donor)
        assert payload["audio_source_ids"].tolist() == [parent[i]["source_id"] for i in donor]


def test_extremely_wrong_scores_keep_uncapped_finite_nll():
    from scripts.lombard_correspondence import choice_logps, choice_metrics

    logp = choice_logps(np.array([[0.0, -1600.0, -800.0, -900.0]]))
    p = np.exp(logp)
    assert p[0, 1] == 0
    assert choice_metrics(p, np.array([1]), logp)["nll"] == 1600
