# R2 synchronized AV source discovery — 2026-11-10

## Decision

No new source is admitted for a joint audio-video Path A fit. This was a
source-rights and group-count screen only: no media, feature arrays, weights,
or evaluation records were opened, and no model or GPU work was run.

The current R2-M gate requires a rights-cleared source, an auditable split by
independent source groups, a task where both inputs are useful, and matched
unimodal and input-perturbation controls. The provisional R0 evidence floor is
1,000 examples and 200 independent groups. More windows or pairs from the same
group do not increase the group count.

## Official-source candidates screened

| Source | Officially reported properties | Decision |
|---|---|---|
| [UnAV-100](https://unav100.github.io/) | The project page says the dataset is CC BY 4.0 for commercial/research use, while also stating that copyright remains with the original video owners. It distributes YouTube IDs and timestamps rather than source rights for each video. | Do not use: the per-video rights needed for this project are not established. No files or annotations were accessed. |
| [Ego4D](https://ego4d-data.org/docs/) | The official documentation describes synchronized audio/video and 926 camera wearers, but access requires acceptance of its license agreement. | Excluded under the existing project decision; no agreement was accepted and no data was accessed. |
| [PE Video](https://ai.meta.com/datasets/pe-video/) | The official Meta page lists CC BY-NC 4.0. | Exclude: the noncommercial restriction is incompatible with the intended product. |
| [VoxDub](https://voxdub-dataset.github.io/VoxDub/) | The project page lists CC BY-NC-SA 4.0 and says original source-video copyrights remain with their owners. | Exclude: the annotation license is noncommercial and does not grant the underlying media rights. |
| [DMD](https://dmd.vicomtech.org/) | The publisher states CC BY-NC-ND 4.0 and academic-only use. | Exclude: noncommercial and no-derivatives restrictions. |
| [AVC-1.8M](https://www.avc.gg/) | The project page licenses its annotations under CC BY-NC-SA 4.0 and leaves source media under platform terms. | Exclude: no qualifying commercial source-media grant. |
| [Full-body AV stimuli, Zenodo 22195794](https://doi.org/10.5281/zenodo.22195794) | The paper reports 120 released participants and CC BY 4.0; audio and face-obscured video are synchronized. | Do not fit: 120 participants are below the 200-group floor, and the published stimuli do not by themselves establish a task requiring both inputs. No files were downloaded. |

## Commons screen status

The official MediaWiki API returned category metadata for animal and bird
video categories (for example, the category pages report sizes of 22 for
“Videos of animals producing sounds” and 11 for “Videos of birds singing”).
These are category sizes, not counts of distinct eligible audiovisual assets
or independent source groups. A larger metadata-only enumeration did not
produce a recoverable complete result, so it is not used as an admission count.
The prior bounded bird search had fewer than 200 candidate licensed files even
before confirming A/V streams; it cannot pass the source-group floor. No
Commons media was downloaded.

## Next action

Continue targeted discovery for a fresh, source-rights-cleared synchronized AV
task. Before acquiring media, require a pinned source revision, item-level
rights, a defensible group identity, at least 1,000 examples across at least
200 groups, and a frozen task/control design. Do not repeat the already
observed fresh-speaker Speech Commands transfer or use its cohort for model
selection. Keep R2-M open until these conditions and the measured both-input
controls pass.
