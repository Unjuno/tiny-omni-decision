# MAVC100 audio-visual VAD source gate — 2026-10-26

## Decision

**Do not admit MAVC100 video assets for this project’s training or quality evaluation on the evidence reviewed.** The official project describes the videos as internet-sourced, limits the dataset to research use, and says potentially infringing video backups will be removed on request. I found no per-video license or rights grant that resolves the underlying media rights for the intended use. The annotations are a plausible synchronized AV task, but that does not cure the media-rights gap.

No dataset payload, labels, or media were downloaded or opened for this screen.

## Task fit

The official annotation README distinguishes audio-only labels (speech/singing from any source) from audio-visual labels (speech/singing by the designated anchor person). This could support a narrow decision such as whether the anchor is vocalizing, with aligned audio and video as input. It is task-relevant, but by itself does not establish that both inputs are necessary for each prediction or general multimodal quality.

## Rights and provenance evidence

The [official dataset README](https://github.com/Yuanbo2020/Audio-Visual-VAD/blob/main/Open_dataset_MAVC100/README.md) says the original video data comes from the Internet, offers original URLs and a backup of source video streams, and limits MAVC100 to research purposes. It says annotations may be used freely for research while the authors reserve legal rights, and requests removal if original video is infringing. These terms do not document a reusable license for each underlying video or grant commercial/product use. The repository’s MIT badge applies to the code repository; it is not a license for the original video streams.

The authors’ [repository overview](https://github.com/Yuanbo2020/Audio-Visual-VAD) describes the paper and task, but does not provide a separate per-asset media-rights schedule in the reviewed material.

## Gate result

- Task relevance: **potentially suitable for a narrow AV vocalization probe**.
- Per-video provenance and reuse permission: **not established**.
- Product-aligned training/evaluation admission: **REJECT pending rights evidence**.
- Media or labels accessed: **no**.
- Sealed audit or existing validation opened: **no**.

Reconsider only if the underlying video rights are documented for the intended use or the project obtains separately authorized recordings. Do not infer media permission from the code license or annotation research-use statement.
