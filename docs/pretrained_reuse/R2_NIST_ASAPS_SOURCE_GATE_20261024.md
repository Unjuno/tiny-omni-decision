# NIST ASAPS source gate (2026-10-24)

## Disposition

**REVIEW / not admitted for this product's training or quality evaluation.**
ASAPS is relevant to multimodal decision support: NIST describes parallel
streams for simulated and staged public-safety events. The official page says
participants consented to release audio and video for research purposes only,
and access requires an account, a signed data-use agreement, NIST approval, and
a time-limited download token. The agreement's actual terms, usable data split,
group identities, label semantics, and media inventory were not verified here.
Do not request access or download data on the basis of this metadata screen.

## Evidence and limits

- Official source: <https://asapsdata.nist.gov/>. NIST describes a continuous
  eight-hour collection with multiple parallel streams and staged emergency
  events, intended for public-safety R&D and decision-support research.
- The page expressly limits participant audio/video consent to research use
  and says use must follow the dataset data-use agreement.
- The download workflow requires account creation, reading and signing the
  agreement, NIST approval, and a time-limited token. The agreement was not
  available from the public metadata page without entering the access flow.
- No account was created, request submitted, agreement signed, or dataset
  content accessed. No annotations, media, test data, or model outputs were
  opened. No GPU, training, cloud compute, or external message was used.
- The public description does not establish caller-supplied option semantics,
  group-disjoint train/validation/audit splits, sufficient independent events,
  or commercial model-use rights. These remain UNKNOWN.

## Decision

ASAPS is a useful task-design reference, but it is not an executable or
product-cleared source under the current evidence. Reopen only after the user
has an authorized access path and the agreement can be reviewed, rights and
splits are established, and a both-input-required Decision task is frozen with
unimodal and synchronization controls. This screen does not close R2 M-2 or
qualify the 95–98% product target.
