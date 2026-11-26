# VisTouch Git payload recheck — 2026-10-26

## Result

A normal shallow Git clone of `https://github.com/liangnjupt/VisTouch.git`,
branch `main`, succeeded at commit
`a569c7e572a78f232e819b652e0ebb718952e476`. It provides source code,
metadata, documentation, and small baseline checkpoints, but **does not provide
the raw synchronized audio, tactile, or video payloads**. The cloned
`dataset/` contains only `.gitkeep`; there are no Git LFS objects and no GitHub
releases. The pinned README says the data are hosted externally on Baidu
Netdisk and must be placed under `dataset/{audio,tactile,video}/`.

This confirms the earlier source report's native-media access gate. The Git
route can retrieve the repository but cannot make the dataset available for a
Path A fit or fresh evaluation. No external archive was accessed in this
recheck, and no source media, checkpoint, or existing artifact was changed.

## Pinned metadata and task limitations

The repository metadata contains 10,498 synchronized two-second derived clips,
312 source samples, and 24 source sessions. The split is grouped by session:

| Split | Sessions | Clips | Force levels |
| --- | ---: | ---: | --- |
| Train | 16 | 7,019 | 3 N and 6 N |
| Test | 8 | 3,479 | 9 N only |

The published classification report exposes results on that test split:
three-input fusion 65.4%, audio-only 51.6%, video-only 51.2%, and tactile-only
19.4%. These are author-reported results, not reproduced here. Because the test
has already been observed and there is no validation split in the metadata, it
must not be reused as a blind audit or for model selection. The 10,498
micro-clips overlap heavily within 24 source sessions and are not independent
examples.

The source declares dataset files CC-BY-4.0 and code MIT. The license is
recorded, but rights alone do not resolve data availability, split design, or
the dataset's known timing/visual preprocessing gates. Preserve the earlier
metadata audit's cautions about linearly approximated video timing, timestamp
burn-in, force-specific split structure, and unresolved sub-second boundary
checks.

## Reproducibility receipt

The independent local receipt is stored outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\vistouch-git-recheck-20261026-attempt-01\result.json`.
Its SHA-256 is
`B24DE255E4B5E43E4A20FBEA6E0E5DBF5546DA6C39A349E36615DABD52B5BA7C`.
It records the source commit, tracked-file/LFS counts, dataset-directory
contents, metadata hashes, split counts, and declared licenses. The cloned
source path is
`C:\CodexArtifacts\pretrained-reuse-r2\vistouch-git-checkout-a569c7e-20261026`.

## Disposition

Keep VisTouch as a potentially useful synchronized vision/audio/tactile
research candidate, but do not fit or score a new model until the publisher's
native archive is available through an authorized route. If it becomes
available, derive model-selection data from train sessions only, keep the
already observed 9 N test split closed, and group by source session and parent
segment rather than by overlapping micro-clip.
