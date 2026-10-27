# Original GRID provisional development split: preparation result

The [predeclared procedure](R2_GRID_DEVELOPMENT_SPLIT_PROTOCOL_20261028.md)
completed on the hash-verified original alignment archive. The fixed speaker
assignment gives 24 train and 8 development-validation speaker labels; the
technical-smoke speaker and the publisher's missing-video speaker are excluded.
The corresponding alignment candidates are 23,992 train and 7,999 validation.
Nine candidate codes overlapped the existing Lombard train, calibration or
observed development-validation manifests and were excluded. All included
six-word alignments match their decoded IDs. Train/validation code and speaker
labels are disjoint; both are disjoint from the 2,624 unique Lombard actual
codes in those three pinned manifests.

The counts are **alignment rows**, not verified media pairs or independent
people. Speaker identity across GRID and Lombard is unknown; matching or
different numeric labels do not establish identity. No media, predictions,
model training or sealed audit was read. This split is development-only and
does not establish generalization or quality.

Output:
`C:/CodexArtifacts/pretrained-reuse-r2/grid-development-split-20261028-attempt-02/`.
`summary.json` SHA-256
`873cc931850a208759012779f3ff59d55ce53ea6a1e00ad2092b3a113fa72681`;
`train.json` SHA-256
`f6f045931aee53e576fca22efda0bc888a983aed685a30edc38b2918ff156817`;
`validation.json` SHA-256
`c1e77098327841fadc283bb14813c38de94e07fd16728d9cc3602bc261961e11`.
The first attempt is preserved; after a formatting-only script correction the
second reproduced both data hashes exactly. Two split tests and Ruff passed.
