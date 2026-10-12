# Smaller video-language candidate follow-up

Existing MobileCLIP/MobileViCLIP audit remains controlling: inherited model-weight
research restrictions are unresolved for product use. No duplicate download/run.
InternVideo2 small vision distillation is not automatically a small aligned full
video/text model. X-CLIP already counts196,585,729 configuration parameters.

## All-in-one: source inspected, not admitted

Official source:https://github.com/showlab/all-in-one
Pinned commit:ae0e288a8b2c505d48057dd0f8f4c215706611df.
Fetched recursive tree, README and AllInOne/config.py; local SHA256 ledger under
`C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-source-audit-01/`.
README advertises unified raw-video/text models Ti12M,S33M,B110M and links official
Google Drive weights/hparams/logs. Counts are publisher claims, not measured
resident parameters. Paper benchmarks are not our task quality.

README states MIT, but fixed recursive tree contains no license-named file;
initial LICENSE request returned404. The full license notice and explicit weight
coverage remain unresolved; do not infer commercial weight rights from code label.
No checkpoint downloaded or executed. Preserve this candidate as REVIEW pending
license/provenance and checkpoint access checks. Old Python3.7/Lightning setup also
needs isolated compatibility inspection; do not downgrade the working environment.

The unified video/text attention is relevant to the observed failure of separately
encoded candidate features, but superiority is a hypothesis requiring actual
load, resource measurements and fixed condition/evidence controls. No current
candidate resolves the four-modality95–98% goal. Continue R1 admission before
allocating another full-corpus quality experiment.
