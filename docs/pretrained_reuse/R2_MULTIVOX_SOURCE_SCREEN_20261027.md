# R2 source screen: MULTIVOX singing groups

**Disposition: REVIEW; no media or annotations acquired; no training or
evaluation admission.** This is the singing-group dataset, not the different
voice-assistant benchmark that shares the name MultiVox.

The authors' [versioned Zenodo record](https://zenodo.org/records/17058101)
describes 154 performances, around three hours, 18 pieces, two recording spaces,
and six- or sixteen-singer configurations. It lists synchronized 360-degree
video, far-field audio and individual singer recordings, plus metadata for
session, singer roles and facing direction. The [project site](https://multi-vox.github.io/)
states that not every visible performer is active in every recording. This
could motivate an audio-plus-video active-performer Decision, but neither
timestamped labels sufficient for that task nor the necessity of both inputs
has been established. Exact audio/video alignment and per-performance identity
have not been independently verified.

The record discourages use for generative-AI training. It does not explain
whether a non-generative Decision classifier is included; do not treat the
statement as either permission or a prohibition for that specific task. An NLM
catalog listing indicates CC BY 4.0, but the extracted Zenodo page shows a blank
license field and the API request did not complete during this screen. The
performances include arranged popular music; rights for recorded compositions
and downstream use have not been checked. These facts require review of the
authors' license and scope before admitting the source. No license is accepted
or inferred from the unrelated project code.

The dataset comprises 154 performances total, fewer than the roadmap's
provisional 200 independent-group screen even before holding out sessions.
Shared singers, songs and spaces can make the independent-group count smaller.
The four released archives total approximately 81.5 GB, versus approximately
1.2 GiB free on the local drive at this screen. No archive, metadata CSV or
annotation was downloaded. A range-based feasibility proposal would still need
an integrity policy, bounded storage and confirmed terms before use.

This source does not close R2 M-2, support the 95–98% product objective, or
justify changing the active model. For contrast, the [Bird-MML record](https://zenodo.org/records/18920487)
states that its image/audio pairs are algorithmically assembled from separate
observations and that the dataset is non-commercial research only; it is not
an approved synchronized natural-AV source for this project.
