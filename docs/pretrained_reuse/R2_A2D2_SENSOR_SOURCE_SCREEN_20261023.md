# A2D2 synchronized sensor / video source screen — 2026-10-23

## Decision

A2D2 is a technically relevant camera/LiDAR/vehicle-bus candidate, but it is
**not admitted for a quality-selected Decision experiment**. The official
paper describes synchronized and registered sensors, but the 392,556-frame
sequential portion comes from only three urban drives. Its CC BY-ND 4.0 terms
permit commercial use of the dataset, but the reviewed sources do not resolve
whether distributing a model trained on the data is an adaptation subject to
the NoDerivatives condition. No data or labels were downloaded or opened.

## Source evidence

- The [official A2D2 project page](https://a2d2-dataset.github.io/) and the
  [AWS Open Data Registry entry](https://registry.opendata.aws/aev-a2d2/)
  identify the dataset license as CC BY-ND 4.0 and expose a public S3 dataset.
- The [Audi-authored dataset paper](https://arxiv.org/abs/2004.06320) says the
  cameras, LiDAR and vehicle-bus signals are time synchronized and mutually
  registered. It reports 41,277 non-sequential annotated frames and 392,556
  sequential, unannotated images from three urban sequences. The bus contains
  signals such as acceleration, angular velocity, steering angle, brake,
  throttle and timestamps.
- These facts make A2D2 relevant to temporal sensor/video feature reuse, but
  they do not define a Decision task whose answer is shown to require both
  streams. The sequential portion has only three independent recording groups;
  overlapping windows cannot be treated as independent validation examples.
  The paper also describes the labelled images as non-sequential, so those
  annotations do not by themselves supply a grouped temporal target.
- Public dataset statistics vary across third-party mirrors. No archive or
  individual media byte, immutable release checksum, exact train/validation
  split, or split-level overlap was verified in this screen.

## Gates and disposition

| Gate | Status | Evidence / remaining work |
|---|---|---|
| Camera, LiDAR and bus synchronization | **Supported by source description** | Official paper reports timestamped, registered streams; runtime-level timestamp audit remains unperformed |
| Dataset license declaration | **Verified: CC BY-ND 4.0** | Project page, AWS registry and paper agree |
| Commercial training use | **Dataset-level permission stated** | Audi says commercial use is permitted subject to the license terms |
| Model distribution under ND terms | **UNKNOWN** | No legal interpretation or rights-holder confirmation; do not assume model weights are outside the restriction |
| Temporal train/validation grouping | **Not established** | Only three sequential drive groups are described; no split was frozen |
| Both-input-required Decision task | **Not established** | Must define counterfactual/missing/shuffle controls from labels or independent evidence |
| Data acquisition / inference / fitting | **Not started** | No files, labels, sealed content, model, or GPU were accessed |

Do not download the multi-terabyte archive or begin training from this screen.
Reconsider only if the intended model-distribution use is resolved, a
rights-compatible bounded subset and its checksums are pinned, and an
independently grouped task with defensible joint-input necessity is specified.
Even then, the three sequential drives are insufficient for a broad temporal
sensor-quality claim; any pilot must be labelled low-power and exploratory.

## References

- A2D2 project and license: <https://a2d2-dataset.github.io/>
- AWS Open Data Registry: <https://registry.opendata.aws/aev-a2d2/>
- Audi-authored paper: <https://arxiv.org/abs/2004.06320>
- CC BY-ND 4.0 legal code: <https://creativecommons.org/licenses/by-nd/4.0/legalcode>
