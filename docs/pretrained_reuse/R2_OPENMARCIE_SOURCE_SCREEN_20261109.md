# R2-M source screen: OpenMarcie — 2026-11-09

## Disposition

**`REVIEW_NOT_ADMITTED`** for Path A fitting or evaluation. OpenMarcie is a promising *video + inertial sensor* action-recognition candidate: the CVPR 2026 paper reports a subject-disjoint 12-class activity task and higher macro-F1 for inertial+vision than either stream alone. This is publisher-reported evidence, not a result from this repository's encoders or Decision head. The dataset's exact data license and downloadable release metadata are not verifiable from the accessible official records, so this screen does not admit the data.

## Pinned source evidence

- Official project code repository: <https://github.com/HymalaiDFKI/OpenMarcie>
- Observed `main` revision: `fd72cf1ff11961520aafebc1f52587fd5bc2e74f` (read-only `git ls-remote`, 2026-11-09).
- Pinned README SHA-256 over retrieved UTF-8 response: `fb4086f54cba13a557b6a725a78b5b773a01790a82457c64dd63b02ae53fe971`.
- Pinned `LICENSE` SHA-256 over retrieved UTF-8 response: `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
- The repository's Apache-2.0 `LICENSE` applies to the code repository. Its README points separately to the Kaggle data release and does not state a data license.
- Kaggle dataset identifier linked by the publisher: <https://www.kaggle.com/datasets/4942046dbb2608f5a0c3e76c9bd53c6b9b2784d738da7db37f7739b89c2c7f95>. An unauthenticated read-only Kaggle metadata request returned HTTP 403; `licenseName`, version, file list, checksums, and byte size therefore remain **UNKNOWN**. No sign-in, agreement acceptance, download, or payload inspection was performed.
- Publisher paper: <https://openaccess.thecvf.com/content/CVPR2026/papers/Bello_OpenMarcie_Dataset_for_Multimodal_Action_Recognition_in_Industrial_Environments_CVPR_2026_paper.pdf>.
- Publisher supplementary material: <https://openaccess.thecvf.com/content/CVPR2026/supplemental/Bello_OpenMarcie_Dataset_for_CVPR_2026_supplemental.pdf>.

## Candidate task and evidence

The paper describes two studies with 12 bicycle-assembly participants and 24 valid 3D-printer-assembly participants, synchronized camera/wearable streams, and a 12-class activity-recognition task. It explicitly describes subject-disjoint splits. It reports these macro-F1 results for inertial (I), vision (V), and I+V:

| Scenario | I | V | I+V |
| --- | ---: | ---: | ---: |
| Bicycle assembly, no null class | 0.834 | 0.757 | 0.882 |
| 3D-printer assembly, no null class | 0.750 | 0.705 | 0.773 |
| Bicycle assembly, with null class | 0.811 | 0.729 | 0.851 |
| 3D-printer assembly, with null class | 0.674 | 0.655 | 0.685 |

These values are copied from the publisher paper's reported benchmark table; they do not establish that each example requires both modalities, that the published group split is available/auditable, or that this project's frozen VideoPrism/MOMENT encoders and typed Decision readout will reproduce the gains. The paper's audio stream is instrumental sound; its narration transcripts are external observer text, not participant speech. Treat audio/text channels as distinct source modalities and do not assume their labels or rights match the video+IMU task.

The paper and supplementary material describe 36 participants across the two scenarios, ethics approvals, consent, and anonymization. These statements are useful provenance, but do not replace the dataset's binding data terms or an audit of the released files and split identities.

## Admission gates still open

1. Obtain an official, dataset-specific license and terms for the Kaggle payload. Do not infer data rights from the code's Apache-2.0 license, a paper license, or third-party listings.
2. Confirm exact release revision, archive size/checksums, file inventory, and whether local extraction/evaluation fits the no-cloud storage budget.
3. Verify released participant/session identifiers and the actual train/validation/test split files; prove participant and recording separation across splits.
4. Audit whether action labels are available at the intended temporal resolution and can be converted to supplied typed options without leaking annotation text, instructions, participant IDs, or future frames.
5. Freeze a project-specific protocol with matched video-only, IMU-only, joint, missing/shuffled/timing-offset controls before any fitting or validation evaluation. Treat the publisher's benchmark numbers only as a candidate rationale, not as our baseline.

Until these gates pass, no Path A fit or quality evaluation is justified on this source. No data, annotations, media, pretrained weights, validation labels, or sealed audit were accessed in this screen; no model, corpus, checkpoint, or user worktree was modified.

## Machine-readable record

See [`R2_OPENMARCIE_SOURCE_SCREEN_20261109.json`](../../manifests/pretrained_reuse/r2-openmarcie-source-screen-20261109.json).
