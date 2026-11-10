# AVMIT metadata-only source and split gate — 2026-11-10

## Disposition

AVMIT passes a **preliminary metadata/split feasibility gate** for a bounded
audio-video correspondence experiment. It is **not yet admitted as an R2-M
quality task**: a feature archive and frozen task protocol still need to be
verified, and the model has not been trained or evaluated. The published
960-clip test set remains excluded. Raw Moments in Time media were not
requested or accessed.

The publisher's [Zenodo record](https://zenodo.org/records/8253350) currently
reports `access_right=open` and `license.id=cc-by` for the released AVMIT
record (annotations and precomputed embeddings). This resolves the prior
screen's missing Zenodo license field for those released files. It does not
grant access to or relicense the original Moments in Time clips; this work
uses no raw clips. Attribute the released record and its authors if the
embeddings are used.

## Source and task lead

The source is pinned to AVMIT repository commit
`e3c06ec0e215b1934d141f86008e92e567366d6c`; the data record is DOI
`10.5281/zenodo.8253350`. The official paper describes 57,177 clips with three
human ratings per clip, and introduces supervised audiovisual correspondence
(SAVC). The paper's controlled 960-video set and reported results are public
and exposed; this project will not score or tune against that set.

A prospective task is binary Decision over supplied options:

- `same_action`: paired audio/video features correspond to the same AVMIT
  action category;
- `different_action`: features come from clips with different AVMIT action
  categories.

The candidate uses paired precomputed audio/video features only. It is a
closed-set correspondence probe, not open-world recognition or product
quality. Before fitting, the experiment protocol must freeze balanced
positive/negative source-pair units and both unimodal, mismatched-pair,
missing-input, option-order and temporal controls. No quality or complementarity
claim follows from this metadata audit.

## Metadata-only audit

Only the small published `video_ratings.csv` and `test_set.csv` were downloaded
to the external artifact directory below. Their record MD5 values match the
Zenodo file metadata. The audit used publisher `video_location` and identity
columns; it did not score a model. It excluded all publisher `validation`
records and every ID in the published `test_set.csv` before reading target
fields for eligible groups. The test file was used through its identity
columns only.

| Source pool | Rows | Treatment |
|---|---:|---|
| Publisher training | 55,516 | Candidate source pool |
| Publisher validation | 1,661 | Excluded |
| Published AVMIT test set | 960 | Excluded by source-video/feature identity; no test target was used |
| Training rows overlapping published test IDs | 957 | Excluded |
| Eligible training source clips | 54,559 | Split below |

The three remaining published-test IDs occur in the excluded publisher
validation pool. Seed `17` and the normalized publisher filename determine a
stable SHA-256 bucket: 70% train, 10% calibration, 10% development validation,
10% sealed audit. The audit found zero overlap of source-video IDs or feature
record paths across the four partitions. All 41 AVMIT action classes appear in
train, calibration, and development validation. The sealed manifest contains
only hashed clip IDs and feature-record paths; it contains no target labels or
label summaries.

| Split | Clips/groups | ID-order SHA-256 | Manifest SHA-256 |
|---|---:|---|---|
| Train | 38,231 | `054e5a3bf494a99873d2ec44237d6170b46b46454eef1e64d565d9ff321a0947` | `4f8bdcf722ee94780737fdd8a23656e6ed70d3cd1988c93c495e73aacb0dd1e2` |
| Calibration | 5,284 | `ccd978f6c65864952b569eead94f88b2a7035dbcb9a2a244c3dedabf54fea9a8` | `69a51a3b36474803bfdf03ab586c3059c669533fb2a438dd0b1480b99054808b` |
| Development validation | 5,510 | `9fd55bf86fc983dc1320ae791115ece4819fa4141543308cfae3505ddf457126` | `7acde5c5742921e8a6e1f7a5754207d94b0fa4f449cb6ad70f09dd220e2e1648` |
| Sealed audit | 5,534 | `7e46c6ca7759bb59133746cc1e4f3387da0d19df0bd983c220a9ab0b6de342e5` | `e47b4aa69e742522893eda260b004f99339a2614f56c2410f26e10c81ffc2b98` |

The source's `filename` is the available clip identity. Parent upload identity
and normalized raw-media overlap cannot be checked from the released tables;
they remain unknown. All partitions are development data, not an untouched
external audit. Although the source table contains public annotations, the
audit code never copies, counts, or serializes sealed targets. Future feature
loading must enforce the frozen split manifest and must not load sealed records
until the architecture, training recipe, and selector are frozen.

## Provenance and local results

- Zenodo API record snapshot SHA-256:
  `0307b526c562b45f249eff6daebc2f0605b4363495fd2d54f54e597ea5d1cc2e`.
- `video_ratings.csv`: 6,867,402 bytes; SHA-256
  `0a4c607b99acb248e30d3c3eb29ab074ac668989e02518459530b4de6994ded9`;
  Zenodo MD5 `9817e7adaaa8aa7d2384823b9d618e7e`.
- `test_set.csv`: 121,081 bytes; SHA-256
  `09c67b5d6bdd6b8eca052f2c3b4084ce391947a3c8266f904b12286ea3f3fb01`;
  Zenodo MD5 `b78453da858755f889cadcf9b0300e22`.
- Split audit JSON SHA-256:
  `17caf9b7e0fecafcfbb6174780794e9747d58356c7d022415fe727dea0a98ed2`.
- External output: `D:\CodexArtifacts\pretrained-reuse-r2\avmit-metadata-gate-20261110-attempt-01`.
- No feature archive, raw media, GPU, training, or model evaluation was used.
- Implementation: `scripts/audit_avmit_metadata.py` and
  `src/tiny_omni_decision/avmit_metadata.py`; focused tests cover deterministic
  splits, test-set exclusion, sealed-label isolation, and duplicate rejection.

## Frozen feature-pair manifests

The non-sealed source manifests were converted to disjoint, balanced SAVC pair
groups before any feature archive was opened. Each group contains two positive
same-action pairs and two negative cross-action pairs from two distinct source
clips. Each source audio and source video occurs once under each target within
its group. A source clip is used in at most one pair group per split. Any later
uncertainty interval must resample whole pair groups, not individual rows.

| Split | Pair groups | Decision rows | Positive / negative | Unpaired clips | Pair JSONL SHA-256 |
|---|---:|---:|---:|---:|---|
| Train | 19,115 | 76,460 | 38,230 / 38,230 | 1 | `113bc99f15714112e4d401dce76eaebbfaee8b729d21ac5113f9cb4ca7883ee4` |
| Calibration | 2,642 | 10,568 | 5,284 / 5,284 | 0 | `bf8f7668a7bbea8b51cf2b38429ba1c9b699d358c2467919c592295d1d49cfeb` |
| Development validation | 2,755 | 11,020 | 5,510 / 5,510 | 0 | `bfa1a3bcca67508b4de64bcf4ec02798ee30087328c8399f50201aca414c4698` |

The one train leftover is an odd remaining source after disjoint pairing; it
was not duplicated. The builder refuses the sealed split. These pair manifests
specify the task labels, not a model result. The pair reports and manifests
remain outside Git under the external artifact root recorded above.

## Next gate

Freeze the pair-construction and scoring protocol, then acquire only the
Zenodo-licensed precomputed feature archive (never raw MIT media). Verify its
publisher checksum, record-to-feature coverage, temporal tensor shapes and
finite values. Start with one bounded frozen-feature fit and paired
source-pair bootstrap; require the joint model to beat both unimodal controls
on validation before considering this a useful fusion result. The selected
Path A encoders are not the feature producers in AVMIT, so any result remains a
task/fusion feasibility diagnostic and cannot be called a product model result.

## Primary sources

- [AVMIT Zenodo record and release metadata](https://zenodo.org/records/8253350)
- [AVMIT peer-reviewed paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10984512/)
- [Pinned AVMIT repository](https://github.com/mjoannou/audiovisual-moments-in-time/tree/e3c06ec0e215b1934d141f86008e92e567366d6c)
- [Moments in Time access instructions referenced by AVMIT](http://moments.csail.mit.edu/)
