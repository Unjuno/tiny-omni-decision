# Public synchronized AV candidate: admission only

Following the user's authorization to choose public data, audited
[official Lombard Grid](https://spandh.dcs.shef.ac.uk/avlombard/).
The publisher declares CC BY4.0 for the corpus, paired frontal/profile video
and speech from54 talkers, nominal5,400 utterances. There is no sensor stream.
This is a candidate for native spoken-command slot decisions with audiovisual
inputs; lipreading does not itself establish general visual/world reasoning.

## Actual files checked

Saved official-pageSHA
`19534c74e4e18bd3c3069984b9c28b1a2de6a5bac9401a48cf95f17b0521f4bc`.
MetadataZIP64,938bytes,SHA
`1de3299baaa6a13d8bcd458a688c94f2288a0ecca8b5a6817a055b393f5a98a6`.
The server publishes no immutable revision/ETag; local SHA snapshots are pins,
not evidence of an immutable upstream release.

Parsed5,340 metadata records,54 members/talkers:5,271 CORRECT/69 WRONG,
2,679 Lombard/2,661 plain. Talker31 has50 records and51 has90 instead of100;
the60-record discrepancy remains unresolved. Four intended codes are irregular;
one speaker/condition/intended-code identity is duplicated. Some actual
transcriptions are free-form errors/silence, so intended filenames cannot
automatically become correct labels. Preserve these discrepancies before choosing
an eligibility rule; do not silently remove difficult speech after scoring.

Provisional seed17 speaker grouping keeps the published sentence-list aliases
6/29 and25/26 together:40train,4calibration,10validation talkers. It still produces
50 intended phrases shared across partitions. **This is not a frozen split**;
actual-transcription and repeated-phrase policies must be resolved before learning.
Speaker-disjoint audio is not automatically sentence-disjoint evaluation.

Acquired four official sample utterance pairs from talkers16/22, assigned to
provisional train only:8files,1,119,667bytes. All hashes and matched metadata
checked. PyAV18.1 decoded finite/nonzero16kHz float WAV and720x480 H264 video,
52–59 frames; decode audit0.68s CPU. Video rates include24 and289/12fps:
timestamps must be used instead of assuming uniform24fps for every file.
MOV contains video only. Common source utterance IDs and similar duration do
not independently prove precise AV alignment; that check remains open.

## Status and next gate

Media/runtime smoke PASS, labeling/split readiness REVIEW. No model quality,
joint benefit,95–98% result, full acquisition or training is claimed. No artificial
noise/degradation will be added just to manufacture a fusion advantage.
Next reconcile metadata and archive identities, actual spoken labels, repeated
sentences and AV offsets. Freeze a protocol before any quality comparison;
if the source cannot support clean partitions, record that constraint.

Local evidence and exact runners:
`C:/CodexArtifacts/pretrained-reuse-r2/lombard-grid-admission-01/`,
`.../lombard-grid-media-smoke-01/`, `.../lombard_grid_*audit.py` and
`.../lombard_grid_media_smoke.py`. Adjacent JSON records URLs/hashes/streams
and unresolved issues. No existing data/run was modified; cloud cost0.
