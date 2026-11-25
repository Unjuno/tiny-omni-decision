# RoboPro prompt split and action-task contract — 2026-11-23

## Decision

The pinned RoboPro TRAIN metadata can support a deterministic, whole-prompt
grouped *development split* for a narrow demonstrator-action retrieval probe.
This does not admit RoboPro for model fitting yet. The exact action-option
contract and sampled-media checks remain open, and no model, feature or action
payload was inspected for this follow-up.

The proposed target is: given an instruction, synchronized video and the
14-dimensional robot state, rank four candidate **recorded next actions** and
return probabilities. One candidate is the action logged at the query time;
the other three are actions logged in different episodes with the same exact
instruction. The target means “which command did the demonstrator record at
this point?” It does not mean that alternatives are unsafe, unsuccessful, or
invalid. This is a behavioral retrieval task, not task-success evaluation,
optimal control, or general robot competence.

## Frozen metadata split

The splitter is implemented in `src/tiny_omni_decision/robopro_split.py` and
invoked by `scripts/build_robopro_prompt_split.py`. It reads only the three
pinned metadata files already acquired in attempt 01. The publisher supplies
16,000 TRAIN episodes and no official validation/test split. Episode 15,999,
which the publisher declares to duplicate episode 12,000, is excluded; the
pair's metadata prompt and length agree, but the payloads were not compared.

With seed 17 and a 20% exact-prompt-group holdout:

| Partition | Episodes | Exact instruction groups | Groups with at least 4 episodes |
|---|---:|---:|---:|
| Train | 12,587 | 1,298 | 1,283 |
| Development | 3,412 | 324 | 319 |
| Excluded duplicate | 1 | — | — |

Train and development have zero episode-ID overlap and zero exact-prompt
overlap. The split uses all but the declared duplicate. Each partition's sorted
episode-ID and prompt hashes, plus hashes and byte sizes of the source metadata,
are stored in
`manifests/pretrained_reuse/robopro-prompt-development-split-20261123.json`.
The 4-choice eligibility count is only a metadata-level upper bound: actual
action vectors may repeat or prove ambiguous, and a later builder must reject
duplicate candidates and log how many queries survive.

## What this split does and does not establish

- It holds out exact instruction strings, so the 324 development instructions
  are absent from training.
- It does not identify the publisher's 80 underlying base tasks or their
  paraphrase/scene families. Semantically equivalent instructions can therefore
  occur on opposite sides.
- It does not establish physical-scene or room generalization. The dataset card
  says scenes are fixed per task variant; scene IDs are absent from the metadata
  reviewed here.
- It does not establish video identity by media hash across all episodes. The
  source schema names video paths by episode and camera, and episode groups are
  disjoint; all media bytes still need existence/hash checks before use.
- It is a project-created development split drawn from publisher TRAIN, not a
  sealed audit or official test set. No final blind audit was created or read.
- It does not validate the “one correct action” assumption. Different commands
  may be equally valid. Report the metric only as agreement with one recorded
  demonstrator action.

## Required next checks before fitting

1. Materialize a bounded, predeclared sample from both partitions and verify
   the task-index mapping, 50-Hz row/frame indices, finite 14-D action/state,
   video PTS alignment, file hashes, and absence of duplicate action options.
2. Freeze one query per independent episode (or another explicit sampling
   policy), action horizon, candidate donor rule, option order seed, and
   ambiguity handling before looking at development predictions.
3. Compare task-text-only, state-only, video-only, text+state, text+video,
   state+video and text+state+video on identical episodes/options. Include
   video/state shuffles and time offsets. A fusion claim requires the joint
   model to beat the strongest single-stream control with episode-clustered
   uncertainty.
4. Keep text optional as an M3 condition. For this English instruction source,
   a compact English text tower (not Japanese BERT-base) is the appropriate
   in-budget comparator. BERT-base remains an offline teacher/reference; adding
   its 111M parameters to VideoPrism-B's 114.4M already exceeds the 200M
   pre-quantization ceiling before sensor or fusion components.
5. Do not run the quality experiment until the selected media sample and
   candidate-action schema pass those checks. An eight-episode CPU-only
   feature smoke has now decoded the selected train/development clips and
   extracted frozen VideoPrism windows at 25%, 50%, and 75% episode progress.
   Each input uses eight uniformly sampled frames ending at its query frame
   (up to two seconds of history), with the matching eight 14-D state rows;
   no future video/state rows enter the input. All 24 causal windows and the
   cache shapes passed. This verifies the preprocessing/runtime path only; it
   is not a fitted model or task-quality result. The RTX 3080 remained unused.

The feature-smoke report and NPZ cache are external artifacts under
`C:\CodexArtifacts\pretrained-reuse-r2-local\robopro-action-smoke-20261123-attempt-01\videoprism-feature-smoke-02`.
The cache is 625,172 bytes (SHA-256
`8912f6dd2c83f80b4e411cf4e71ccd7ea9e77d21724910068651fbb0819cfce2`). The
forward passes took 36.25 s in total on CPU; the complete process took 48.00 s,
and loading the pinned weights took 0.47 s. Cached arrays are `[8,3,8,768]` temporal video features,
`[8,3,8,14]` aligned state, `[8,3,14]` target actions, and `[2,3,4,14]`
candidate actions. The 8 episodes and 2 prompts remain a smoke sample, not an
evaluation set. Action-label ambiguity and scene-held-out generalization are
still unresolved, so status remains `REVIEW_NOT_ADMITTED`.

## Reproduction

```powershell
$env:PYTHONPATH = '<checkout>\src'
python scripts/build_robopro_prompt_split.py `
  --info <attempt-01>\info.json `
  --tasks <attempt-01>\tasks.jsonl `
  --episodes <attempt-01>\episodes.jsonl `
  --output manifests\pretrained_reuse\robopro-prompt-development-split-20261123.json `
  --seed 17 `
  --development-fraction 0.2
```

The recorded metadata SHA-256 values are the same as the source screen:

- `info.json`: `91c7d161a416685b08f9db5b195a6ba02f7047d966fa188e0262f854af386cfe`
- `tasks.jsonl`: `8e114f65fd235eeac8fcd1b32e3c95a859613c6e4af119179366394b976d65bc`
- `episodes.jsonl`: `88492784987d483618b5144f37bac8c3036bdaaa11d2df730f7b26a804b68658`

Focused tests cover deterministic assignment, prompt/episode disjointness,
duplicate exclusion, malformed prompt mapping, causal-window construction,
and numeric candidate separation. No training, quality evaluation, official
evaluation, sealed audit, or GPU work occurred.
