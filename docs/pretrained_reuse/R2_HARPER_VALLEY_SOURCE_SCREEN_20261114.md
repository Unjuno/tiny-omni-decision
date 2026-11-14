# HarperValleyBank source and task screen — 2026-11-14

## Decision

Do not admit HarperValleyBank as evidence for the R2-M multimodal-fusion gate.
It is a possible narrow, speech-intent M1-A dataset, but it is not selected for
the current Path A baseline: its 59 speakers, scripted/contact-center domain,
and label-bearing scenario metadata make it a weak fit for the project's
four-modality, independently grouped quality objective. No data or media was
downloaded, no model was loaded, and no evaluation was run.

## Pinned source and rights

- Repository: <https://github.com/cricketclub/gridspace-stanford-harper-valley>
- Reviewed source revision: `0bd721e877c4a85d8c13ff837e68661ea6200a98`
- Reviewed files: `README.md` and `LICENSE` at that exact revision.
- The repository LICENSE is CC BY 4.0. The authors' paper describes the corpus
  as released under CC BY and says it is available for education, research, or
  commercial development. This is a source-level permission check, not a
  determination of individual speaker consent or any separate rights issue.
- Paper: <https://arxiv.org/abs/2010.13929> (v2, 2021-03-19).

## Task and split risks

The source describes 23.7 hours, 1,446 simulated human-human banking
conversations, 59 speakers, and eight caller-intent classes. The caller is
given a goal/script; the conversations use templated language and a vocabulary
of about 735 words. Caller intent is derived from the task assigned during
collection. This is useful for a bounded speech-intent study, but is not an
open-domain or general audio Decision benchmark.

The repository pairs each conversation's caller audio, agent audio,
transcript, and metadata by conversation ID. Metadata contains speaker IDs,
task type, and the assigned task; those fields can directly reveal the intent
target. Any future M1-A task would have to use the waveform as the model input,
use intent only as the supervised target, and exclude transcripts, task fields,
speaker IDs, filenames, and other answer-bearing metadata from model inputs.

The paper reports a speaker-split test for its supervised spoken-language
understanding experiment, but its representation-transfer experiments use
class-stratified train/test splits. A project-specific split still needs to be
verified at the participant level. Since speakers can appear as either caller
or agent in different calls, conversation-level grouping or caller-only
grouping is insufficient. There are at most 59 speaker identities, below the
roadmap's provisional 200-independent-scene/clip floor for temporal studies;
that floor does not directly govern audio-only tasks, but the small number of
independent speakers limits precision and generalization.

The paper's reported intent accuracy ranges from 34.96% to 66.3% across the
published setups. These numbers are context only: they use different models,
features, and split protocols, so they are not estimates for this project or a
formal ceiling.

## Exposure and acquisition

The current repository/worktree history and the checked R2 artifact locations
show no prior HarperValleyBank use. Exposure outside those inspected locations
is unknown. No source archive, transcript, annotation payload, waveform, or
model output was fetched or opened for this screen.

## Next action

Continue the Path A/R2 work using already admitted, already pinned sources and
their preserved results. Do not use HarperValleyBank to claim multimodal
fusion. Reconsider it only if a specific speech-intent M1-A gap remains after
the existing audio task matrix is reconciled, and only after a metadata-only
speaker-overlap/split feasibility audit establishes independent groups and a
fresh, untouched development set. Do not duplicate an existing Path A fit or
re-evaluate exposed validation merely to produce another score.
