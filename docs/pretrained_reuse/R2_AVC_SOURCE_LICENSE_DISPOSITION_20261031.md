# R2-M Action Verb Corpus source and task disposition — 2026-10-31

## Decision

**REVIEW; not admitted for data access, training, or quality evaluation.** The official publisher and the DOI record disagree on the dataset license, and the available description does not define a task for which both streams are necessary. No data archive, video, audio, annotation payload, or participant-level content was downloaded or opened. No model was trained or evaluated.

## Pinned metadata evidence

- OFAI's official [Action Verb Corpus page](https://www.ofai.at/resources/avc) reports 46 episodes, 12 people, and 390 TAKE, PUT, and PUSH actions, with German speech/audio, video, and hand/arm motion. It labels the dataset CC BY 4.0.
- The official [Zenodo record, DOI 10.5281/zenodo.5140014](https://zenodo.org/records/5140014) reports the same corpus description but labels the data CC BY-NC-SA 4.0. That record lists a 590.7 MB total across files while its page statistics report 36.1 GB; this unexplained size discrepancy is another reason not to plan acquisition from the summary alone.
- The [LREC 2018 paper](https://aclanthology.org/L18-1338/) describes actions performed while participants verbally describe them. The corpus provides utterance transcripts, action/object annotations, hand-object contact, and tracked hand/object positions. The paper notes that some visual-action annotations (including object movement and hand contact) were manually annotated.
- The publisher description and paper identify recordings/actions but do not publish a train/validation/test manifest or a participant-held-out protocol. At most, 46 episodes and 12 people are available as grouping units; the true independent-group support would be smaller after validation and test allocation.

## Task assessment

A plausible diagnostic would compare what a speaker says with the observed manipulation, using audio/text for the utterance and video/motion for what actually happened. The published corpus, however, is described as participants performing and describing the same instructed actions. It does not establish a balanced set of intentionally mismatched or contradictory speech/action pairs. The simpler action-class task (take/put/push) may be solved from the spoken verb or the visible manipulation alone, so it cannot establish that fusion is required. The existing annotations are not evidence that a joint model improves on matched unimodal controls.

## Gate and next step

Do not download or process the corpus under the conflicting terms. Reconsider only after the provider resolves the CC BY 4.0 versus CC BY-NC-SA 4.0 discrepancy in writing and confirms the permitted downstream use, and after a metadata-only audit establishes enough independent episode/person groups plus a task with valid both-required and unimodal-control examples. No license was accepted and no access request was sent.
