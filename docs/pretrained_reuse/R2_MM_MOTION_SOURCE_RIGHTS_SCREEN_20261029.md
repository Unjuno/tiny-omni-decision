# MM-Motion source and rights screen — 2026-10-29

## Disposition

Keep `REVIEW_NOT_ADMITTED` for product-directed fitting or evaluation. Zenodo
v2 declares the record open and CC BY 4.0, while the official project
repository contains a dataset agreement requiring a signature and limiting
use to non-commercial research. That agreement also prohibits medical
diagnostic activities and redistribution. The scope conflict is unresolved;
do not sign or treat the Zenodo metadata alone as clearing the publisher's
separate terms. No access request, agreement acceptance, or payload download
was made.

The sensor/video task is structurally relevant to R2-M, but the available
metadata does not prove that both streams are necessary for a target decision.
The dataset describes ankle-sprain risk evaluation, which is not a target for
clinical or diagnostic claims in this project. No model, labels, or media were
evaluated.

## Pinned source evidence

- Zenodo v2 record: [10.5281/zenodo.19951742](https://zenodo.org/records/19951742),
  record ID `19951742`, published 2026-05-10. The official API reports
  `access_right=open`, `license.id=cc-by-4.0`, and 22 files. Retrieved API
  response SHA-256:
  `842fc376e108b4fa5a8f5bd1efcb6264e30dffafbfee34255768fb5a71216b63`.
- The 22 record files comprise 20 subject-named ZIP archives and two small
  metadata files. Their summed declared size is 175,129,648,286 bytes
  (175.13 GB decimal; 163.10 GiB). No archive or annotation spreadsheet was
  opened.
- The Zenodo description reports 16 standardized actions, 33 subjects, 933 IMU
  files, 1,866 Kinect video files and 918 plantar-pressure streams. It states
  that raw data are available via separate access, while v2 lists the
  subject archives as open files. This access-description difference is
  unresolved in addition to the agreement conflict.
- Official project repository: `BSU3DVLab/BSU3DVLab.github.io` at commit
  `51a26dcf42e38bf326e4434bc34f2aade94c27ca`. The agreement path is
  `MM-Motion/agreement/MM-Motion Dataset Agreement.docx`, Git blob SHA-1
  `224862bcfc2c01bed9ef74b4c9507190b2d5da2b`, 17,285 bytes, SHA-256
  `0d54f57bb333c830daf34b91e27d7391ea3e8ab83a5a3380ed0a8fa802f8c84c`.

The agreement text says access is solely for non-commercial academic research
and education, prohibits commercial and medical-diagnostic use, bars further
copying/publication/distribution of dataset portions, and requires a signed
agreement. The Zenodo record's CC BY metadata does not resolve those
restrictions. No interpretation is made that would broaden permitted use.

## Task, split, and resource gates

- **Potential task:** motion/action decisions using video, IMU and plantar
  pressure. Labels, task construction and annotation quality were not opened.
- **Both-input necessity:** `UNKNOWN`; no unimodal or perturbation evidence was
  gathered.
- **Groups/splits:** the Zenodo v2 files are subject-named, but no official
  train/validation/audit split manifest was found in the record metadata. The
  record's 20 subject archives versus 33 described subjects needs
  clarification; no independent grouped split was constructed.
- **Storage:** the full archives total 175.13 GB, far above the current local
  free space (0.482 GiB at 05:40 JST) and the 2-GiB experiment reserve. No
  partial-subject archive was selected because the rights, split and task
  gates already fail.
- **Use performed:** metadata and the small upstream agreement document only;
  no media, annotation payload, feature extraction, fitting, validation or
  sealed audit.

## Next action

Do not use this source unless the provider resolves the license/Agreement
conflict in writing and clarifies the subject grouping/splits and permitted
non-diagnostic task. Any future review or access request requires explicit
authorization; this screen submitted none. Continue seeking an already
admitted source for the R2-M both-input-required task rather than treating this
candidate as a fallback.
