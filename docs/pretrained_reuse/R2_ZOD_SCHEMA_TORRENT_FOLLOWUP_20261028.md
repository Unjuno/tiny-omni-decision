# ZOD schema and public torrent follow-up — 2026-10-28

Status: **candidate task is now schema-supported, but ZOD remains unadmitted**.
This is an additive follow-up to
[`R2_MULTIMODAL_SOURCE_ADMISSION_SCREEN_20261021.md`](R2_MULTIMODAL_SOURCE_ADMISSION_SCREEN_20261021.md).
No dataset payload was downloaded, opened, or evaluated. No access request was
sent, and no model was trained or run.

## Evidence collected

- The current official [download page](https://zod.zenseact.com/download/)
  describes both an access-request/CLI route and an Academic Torrents route.
  Because the page also says access must be requested, whether the public torrent
  route waives that request is **unresolved**. We did not treat public
  reachability as access approval.
- Retrieved only the official Frames torrent descriptor linked from the ZOD
  download page (not media payload): 9,096,453 bytes; SHA-256
  `27e4d32daa4169c4b892613aae7198b8b6b6c2e3204b853ebfbefa7fa951f59c`.
  The descriptor advertises 27 files. The candidate mini/info/annotation
  archives are `frames_mini.tar.gz` (924,749,420 bytes), `infos.tar.gz`
  (601,138,843 bytes), and `annotations.tar.gz` (1,453,421,071 bytes),
  totaling 2,979,309,334 compressed bytes. This is a torrent index claim only;
  extraction size, mini coverage, and whether these files suffice for a task are
  unknown.
- Cloned the official MIT-licensed ZOD SDK read-only at commit
  `601a3ef5cfccad9cc545230362077d299acbf898` into
  `C:\CodexArtifacts\pretrained-reuse-r2\zod-sdk-audit`.
  In `zod/anno/object.py`, `ObjectAnnotation.from_dict` constructs `box2d`
  from the object geometry, reads the shared `properties["annotation_uuid"]`,
  and creates `box3d` only when `location_3d` exists. Thus a single annotated
  object record can pair an image-space 2D box with a LiDAR-frame 3D box under
  one UUID. This confirms annotation-level co-reference, not model-level
  sensor complementarity.
- The official [Frames description](https://zod.zenseact.com/frames/) reports
  100k annotated camera-LiDAR pairs. The [Sequences description](https://zod.zenseact.com/sequences/)
  reports 1,473 twenty-second sequences and annotations only on each central
  keyframe. The official [license](https://zod.zenseact.com/license/) states CC BY-SA 4.0, requires its full
  privacy notice for public use/distribution/display, and says ZOD is not
  intended for military use. Product-model derivative/share-alike implications
  have not been legally assessed.

## Candidate task, not yet admitted

A bounded supplied-option task could ask: given a marked 2D object crop and a
set of candidate 3D LiDAR boxes from the synchronized frame, choose which 3D
box refers to that object. The common annotation UUID can provide a label when
both boxes exist. Same-class, similar-range distractors and matched image-only,
LiDAR-only, joint, and input-shuffle controls would be required before claiming
that both inputs help. Using annotated 3D boxes as options would test
cross-sensor association conditional on supplied candidates; it would not test
raw LiDAR detection or general driving decisions.

Before any payload use or fitting, resolve all of these gates:

1. Confirm whether the Academic Torrents route is permitted without the
   publisher's access request; no email or external contact has been made.
2. Inspect the mini archive's actual split and metadata, then prove grouping by
   sequence and any shared parent drive/session so related recordings cannot
   cross train/validation/audit. The SDK split API exposes frame/sequence IDs;
   drive-level independence is not established by the files inspected so far.
3. Count usable common-UUID 2D/3D annotations and independent source groups.
   Do not infer these counts from the 100k headline or torrent archive sizes.
4. Audit pretraining overlap for the actual frozen image and sensor encoders.
5. Freeze a both-required task and unimodal/shuffle controls before any
   validation-guided model selection. Keep this as a development task unless
   the independent-group count supports the roadmap's quality gate.

## Disposition

The 2026-10-21 statement that co-reference was not established is superseded by
the pinned SDK schema evidence above. The broader admission decision is
unchanged: ZOD is a **promising camera-LiDAR association candidate**, but access,
group independence, encoder overlap, and actual both-input benefit remain open.
No quality run is justified until those gates are resolved. No v1/v2 artifact,
existing run, or active process was changed.
