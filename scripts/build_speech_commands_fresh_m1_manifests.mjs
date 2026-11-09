import { createHash } from 'node:crypto'
import { readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const SOURCE = 'google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da'
const LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
const sha = (value) => createHash('sha256').update(value).digest('hex')

function arg(argv, name) {
  const index = argv.indexOf(`--${name}`)
  if (index < 0 || !argv[index + 1]) throw new Error(`missing --${name}`)
  return argv[index + 1]
}

async function main() {
  const mediaDir = path.resolve(arg(process.argv, 'media-dir'))
  const inventoryPath = path.resolve(arg(process.argv, 'inventory'))
  const reportPath = path.resolve(arg(process.argv, 'materialization-report'))
  const inventoryBytes = await readFile(inventoryPath)
  const inventoryRows = inventoryBytes.toString('utf8').trim().split(/\r?\n/)
    .filter(Boolean).map((line) => JSON.parse(line))
  const materialization = JSON.parse(await readFile(reportPath, 'utf8'))
  if (materialization.status !== 'FRESH_M1_TRAIN_AND_DEVELOPMENT_MEDIA_MATERIALIZED'
      || materialization.source !== SOURCE) throw new Error('unexpected frozen media materialization report')
  const splitRows = { train: [], validation: [] }
  const ids = new Set()
  const splits = Object.fromEntries(['train', 'validation'].map((split) => [split, {
    groups: new Set(), paths: new Set(), media: new Set(), classCounts: Array(LABELS.length).fill(0),
  }]))
  for (const row of inventoryRows) {
    const split = row.materialized_split
    if (!splitRows[split] || row.source !== SOURCE || !LABELS.includes(row.target)) {
      throw new Error(`unexpected inventory row identity: ${row.id}`)
    }
    if (ids.has(row.id)) throw new Error(`duplicate inventory sample ID: ${row.id}`)
    ids.add(row.id)
    const mediaPath = path.resolve(mediaDir, row.local_file)
    if (!mediaPath.startsWith(`${mediaDir}${path.sep}`)) throw new Error(`media path escapes output: ${row.id}`)
    const wav = await readFile(mediaPath)
    const wavHash = sha(wav)
    if (wavHash !== row.audio_sha256 || wav.length !== row.wav.bytes) {
      throw new Error(`media inventory hash/size mismatch: ${row.id}`)
    }
    const group = splits[split]
    group.groups.add(row.speaker_group_sha256)
    group.paths.add(row.source_path_sha256)
    group.media.add(wavHash)
    group.classCounts[LABELS.indexOf(row.target)] += 1
    splitRows[split].push({
      id: row.id,
      source: SOURCE,
      source_revision: SOURCE.split('@')[1],
      split,
      target: row.target,
      speaker_group_sha256: row.speaker_group_sha256,
      source_file: row.source_file,
      source_row: row.source_row,
      source_path_sha256: row.source_path_sha256,
      source_record_id: `${row.target}/${row.id}.wav`,
      media_path: row.local_file,
      media_sha256: wavHash,
    })
  }
  if (inventoryRows.length !== 2500 || splitRows.train.length !== 1500 || splitRows.validation.length !== 1000) {
    throw new Error('inventory row counts do not match the frozen plans')
  }
  if (splits.train.groups.size !== 403 || splits.validation.groups.size !== 200
      || [...splits.train.classCounts].some((count) => count !== 150)
      || [...splits.validation.classCounts].some((count) => count !== 100)) {
    throw new Error('materialized examples do not match frozen group/class quotas')
  }
  const intersectionCount = (left, right) => [...left].filter((item) => right.has(item)).length
  const splitOverlap = {
    speakers: intersectionCount(splits.train.groups, splits.validation.groups),
    source_paths: intersectionCount(splits.train.paths, splits.validation.paths),
    audio_hashes: intersectionCount(splits.train.media, splits.validation.media),
  }
  if (Object.values(splitOverlap).some((count) => count !== 0)) {
    throw new Error(`materialized split identity overlap: ${JSON.stringify(splitOverlap)}`)
  }
  for (const split of ['train', 'validation']) {
    const manifestPath = path.join(mediaDir, `${split}.jsonl`)
    const content = splitRows[split].map((row) => JSON.stringify(row)).join('\n') + '\n'
    await writeFile(manifestPath, content, { flag: 'wx' })
  }
  const trainBytes = await readFile(path.join(mediaDir, 'train.jsonl'))
  const validationBytes = await readFile(path.join(mediaDir, 'validation.jsonl'))
  const report = {
    schema_version: 1,
    status: 'FRESH_M1_TRAINING_MANIFESTS_VERIFIED',
    source: SOURCE,
    inventory_sha256: sha(inventoryBytes),
    materialization_report_sha256: sha(await readFile(reportPath)),
    train_manifest_sha256: sha(trainBytes),
    validation_manifest_sha256: sha(validationBytes),
    train_examples: splitRows.train.length,
    validation_examples: splitRows.validation.length,
    train_unique_speakers: splits.train.groups.size,
    validation_unique_speakers: splits.validation.groups.size,
    train_unique_audio_hashes: splits.train.media.size,
    validation_unique_audio_hashes: splits.validation.media.size,
    train_duplicate_audio_example_excess: splitRows.train.length - splits.train.media.size,
    validation_duplicate_audio_example_excess: splitRows.validation.length - splits.validation.media.size,
    train_validation_overlap: splitOverlap,
    train_class_counts: Object.fromEntries(LABELS.map((label, index) => [label, splits.train.classCounts[index]])),
    validation_class_counts: Object.fromEntries(LABELS.map((label, index) => [label, splits.validation.classCounts[index]])),
    media_sha256_verified: inventoryRows.length,
    gpu_used: false,
    model_loaded: false,
    inference_run: false,
    training_run: false,
    sealed_audit_opened: false,
  }
  await writeFile(path.join(mediaDir, 'training-manifest-report.json'), `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' })
  console.log(JSON.stringify(report, null, 2))
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  main().catch((error) => {
    console.error(error.stack || error)
    process.exitCode = 1
  })
}
