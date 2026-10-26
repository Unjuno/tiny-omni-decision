# Android in the Wild source screen

Date: 2026-10-26  
Status: **REVIEW — no data acquired or model fitting admitted**

## Pinned source and scope

Screened the official `google-research/google-research` repository at master
commit `0f585ccae259c32a877edfd48362c18a0010545f`. The official AitW README at
that commit has SHA-256
`6d8964c95db41dada1d5867e50a5ec61f6cb90595965d0e72e010b8e4a12dcb2`. Only
repository README and license metadata were inspected; no GCS objects, TFRecord
payloads, screenshots, annotations, or split files were fetched or opened.

The repository-root `LICENSE` is Apache-2.0 and covers repository code. The
AitW subdirectory has no separate `LICENSE`, and the official dataset README
does not state terms for the dataset payload. The publisher describes records
from Google applications and third-party apps/websites. Public GCS paths do not
establish redistribution, model-training, or derivative-model rights for these
screenshots and associated content. Keep the data and any training on it
REVIEW-only until the dataset rights holder gives explicit terms that cover
these uses.

## Task fit and split evidence

The source describes 715,142 episodes / 5,689,993 observation-action examples
across five sets, with screenshots, language instructions, action types,
coordinates, episode IDs, and step IDs. It publishes episode-wise standard
train/validation/test split descriptions and OOD splits for unseen Android
version, subject, verb, and domain. This is a strong structural match for
screen-conditioned action selection, and the episode IDs are an appropriate
starting point for grouping trajectories.

The released actions are not already finite supplied-option decisions: taps
and swipes use coordinate pairs, while text entry includes an open text string.
A project task would need to define finite screen-grounded action candidates,
retain stable option IDs, and confirm each target is represented among the
candidates. It must keep complete episodes together and audit repeated
instruction templates and app/domain overlap. The source's standard split is
episode-wise, but that alone does not prove prompt- or domain-disjoint
generalization; the named OOD split manifests would need their own pinned audit.

The task also uses natural-language instructions, so it would test
screen-plus-text control rather than the roadmap's direct text-free
screen-only M-0 task. The existing `groundcua` pilot used oracle regions and
lost to a frequency prior; AitW could be a distinct task if rights and a
candidate-action representation are resolved, but it cannot retroactively
validate or replace that pilot.

## Decision

Do not fetch, parse, train on, score, or package AitW. Keep it as a REVIEW-only
screen-task lead. Revisit only after written payload rights are available and
an episode/template/domain-grouped supplied-action task and evaluation split
are specified. No GPU or cloud resources were used.

## Primary sources

- [Official AitW README at pinned source commit](https://github.com/google-research/google-research/blob/0f585ccae259c32a877edfd48362c18a0010545f/android_in_the_wild/README.md)
- [Official repository Apache-2.0 license at pinned source commit](https://github.com/google-research/google-research/blob/0f585ccae259c32a877edfd48362c18a0010545f/LICENSE)
- [AitW paper](https://arxiv.org/abs/2307.10088)
