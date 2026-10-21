# MAVC100 audiovisual source screen — 2026-10-22

## Decision

**REVIEW ONLY; do not ingest, train on, or count MAVC100 toward product quality.**
The task definition is unusually relevant to evidence-dependent audiovisual
binding, but the primary dataset page says the original videos came from the
Internet, restricts the dataset to research use, and asks people to contact the
authors if a source video is infringing. The dataset's CC BY 4.0 label does not
establish that the dataset authors can grant commercial or redistribution
rights to those underlying videos. No media, labels, or archives were opened or
downloaded for this screen.

## Task fit

The authors describe a 100-minute musical audiovisual corpus with frame-level
labels. Audio labels mark any speech or singing in the soundtrack; audiovisual
labels mark speech or singing attributable to the visible anchor/target speaker.
This creates a meaningful counterfactual: background speech/music can remain in
audio while the target-person activity label depends on the video. It is a
strong task-design lead for testing whether a compact decision system binds
audio to the correct visible person, rather than merely detecting any voice.

The supplied label schema has four classes (silence, speech, singing, others),
but only speech, singing, and silence intervals are explicitly annotated;
"others" must be derived from remaining time. An audio-to-AV comparison should
therefore retain identical temporal windows and evaluate both targets, with
grouped splits by original stream/event/person where available. The source
screen does not establish that those group IDs or a leakage-safe split are
published, or that the resulting task meets the roadmap's provisional
independent-group target.

## Rights and provenance

- The authors' dataset README says the original videos are from the Internet,
  says the corpus is for research purposes only, and notes that the authors
  will remove backed-up videos if the original video is infringing.
- A catalog description labels the dataset CC BY 4.0, but that does not by
  itself resolve rights in the underlying original video streams or authorize
  commercial model training, derivatives, and redistribution.
- The GitHub repository has an MIT code license; that covers repository code,
  not the original videos or annotations.
- The cited paper motivates the task and reports audiovisual fusion benefits;
  it does not settle media rights or independent evaluation design.

Before reconsidering, obtain a clear grant from the relevant rights holders for
commercial model training and evaluation, derivative weights, and any necessary
media retention/redistribution; pin the dataset, annotation, and source-video
revisions; establish original-stream and target-person group IDs; verify
group-disjoint train/validation/audit splits; and document a legally permitted
acquisition path. If those permissions cannot be obtained, keep MAVC100 as a
task-design reference only and locate a rights-cleared substitute.

## Evidence inspected

- Author project page: <https://yuanbo2020.github.io/Audio-Visual-VAD/>
- Author dataset README: <https://github.com/Yuanbo2020/Audio-Visual-VAD/blob/main/Open_dataset_MAVC100/README.md>
- Author repository: <https://github.com/Yuanbo2020/Audio-Visual-VAD>
- Dataset catalog entry (secondary metadata): <https://datasetcatalog.nlm.nih.gov/dataset?q=0002226000>
- Paper DOI: <https://doi.org/10.1109/ICASSP39728.2021.9413418>

The primary author README and project page were inspected as text only. No
archive member, label row, video, audio, or remote media payload was accessed.
No training, feature extraction, model evaluation, or GPU work was performed.

## Roadmap effect

MAVC100 is not an admitted data source and does not close R2's synchronized-AV
or both-required task gate. It is retained as a high-value task-design lead
whose principal unresolved blocker is rights to the original media; group/split
adequacy is also unknown. No existing corpus, checkpoint, or artifact changed.
