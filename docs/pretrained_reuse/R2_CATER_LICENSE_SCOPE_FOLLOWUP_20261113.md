# CATER pre-generated archive license-scope follow-up — 2026-11-13

## Decision

**ALLOW for a local, non-redistributive Path A research experiment, with
attribution and license notices retained.** The pinned official CATER README
identifies the pre-generated sample as the dataset used in the paper and
states that CATER is Apache-2.0 licensed. The official generator records CC
BY 4.0 as its default generated-data license, and CATER reuses CLEVR assets
that the CLEVR publisher identifies as CC BY 4.0. These are permissive grants
compatible with this local experiment. The selected videos will not be
committed, rehosted, or included in a product package.

Primary sources:

- [Pinned CATER README and dataset link](https://github.com/rohitgirdhar/CATER/blob/13a19643f1a2fb24e931df25abd74353e4f2fdcb/README.md)
- [Pinned CATER LICENSE](https://github.com/rohitgirdhar/CATER/blob/13a19643f1a2fb24e931df25abd74353e4f2fdcb/LICENSE)
- [Pinned generator's output-license default](https://github.com/rohitgirdhar/CATER/blob/13a19643f1a2fb24e931df25abd74353e4f2fdcb/generate/render_videos.py)
- [CLEVR publisher license notice](https://cs.stanford.edu/people/jcjohns/clevr/)

The scene archive contains individual JSON files rather than the generator's
combined `info.license` record, so the exact recorded license value of each
pre-generated file is unavailable. The dataset-level Apache statement in the
pinned README directly covers the linked sample; the CLEVR CC BY notice is
retained as an additional source-asset attribution precaution. This is a
project-level license interpretation, not legal advice.

## Scope controls

- Use only IDs from CATER `train_subsetT` and `train_subsetV` after excluding
  every numeric ID present in the non-sealed Clevr-4 train, selector, or
  reporting manifests. The frozen cohort has 1,679 train and 439 selector
  videos; do not read official CATER validation/test lists.
- Do not commit media or model weights. Store the selected media and hashes
  under a new `D:\CodexArtifacts\pretrained-reuse-r2\` attempt directory.
- CATER remains unadmitted for model fitting until the selected media's
  hashes, train/selector identity checks, and cross-source content checks are
  recorded. This decision resolves the rights gate only.
- The task is synthetic temporal video-only 8x8 snitch localization. It is
  M1 video evidence and cannot close the roadmap's both-input-required M2
  fusion gate.

## 2026-11-14 task-description correction

The admitted development candidate uses a distinct fixed derived question:
which object has the greatest annotated net displacement between the endpoints
of one 75-frame interval. It retains every scene object as an option. The
phrase “8x8 snitch localization” above describes the broader CATER source task
and was inaccurate for the current derived readout. This correction does not
change the source license interpretation or upgrade the candidate beyond a
local video-only Path A development task.
