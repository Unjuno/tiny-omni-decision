import { createHash } from 'node:crypto'
import { copyFile, mkdir, readFile, readdir, writeFile } from 'node:fs/promises'
import { constants } from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const REVISION = 'a751309c0fd613e8a5d30d77900f30e8b42bc2da'
const DATASET = 'google/speech_commands'
const LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
const ORIGINAL_MANIFEST_SHA256 = '96c6fe86298ef234e3b49546df8e07858049283a1e1334e4982a5bdbad5daaa4'
const PACKAGE_SHA256 = '8005da09189a9e358d94a678b42315bb38a368a9af5ef9823d457cb04ad8ddea'
const ENTRY_SHA256 = '9a61c1600b11d650609ea664c8de6e290046ac5045870cc00bfae956d3adcb86'
const decoder = new TextDecoder()
const sha = (value) => createHash('sha256').update(value).digest('hex')
const text = (value) => typeof value === 'string' ? value : decoder.decode(value)
const parsers = { stringFromBytes: (bytes) => bytes, jsonFromBytes: (bytes) => bytes && JSON.parse(decoder.decode(bytes)) }

function arg(argv, name) {
  const index = argv.indexOf(`--${name}`)
  if (index < 0 || !argv[index + 1]) throw new Error(`missing --${name}`)
  return argv[index + 1]
}

function rangedFetch(stats) {
  return async (url, init = {}) => {
    const headers = new Headers(init.headers)
    if (init.method !== 'HEAD' && !headers.has('range')) throw new Error('refusing non-range request')
    const response = await fetch(url, { ...init, headers, signal: AbortSignal.timeout(90_000) })
    if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`)
    if (headers.has('range')) {
      stats.range_requests += 1
      stats.range_response_bytes += Number(response.headers.get('content-length') || 0)
    }
    return response
  }
}

async function main() {
  const manifestPath = path.resolve(arg(process.argv, 'manifest'))
  const planPath = path.resolve(arg(process.argv, 'plan'))
  const reportPath = path.resolve(arg(process.argv, 'preflight-report'))
  const mediaDir = path.resolve(arg(process.argv, 'media-dir'))
  const outputDir = path.resolve(arg(process.argv, 'output-dir'))
  const modulePath = path.resolve(arg(process.argv, 'hyparquet-entry'))
  await mkdir(outputDir, { recursive: true })
  if ((await readdir(outputDir)).length) throw new Error('corrected output directory must be empty')

  const manifestText = await readFile(manifestPath, 'utf8')
  if (sha(manifestText) !== ORIGINAL_MANIFEST_SHA256) throw new Error('original cohort hash mismatch')
  const sourceReport = JSON.parse(await readFile(reportPath, 'utf8'))
  const planText = await readFile(planPath, 'utf8')
  const plan = JSON.parse(planText)
  if (sourceReport.dataset !== `${DATASET}@${REVISION}` || plan.count !== 30) {
    throw new Error('source or row-group plan differs from frozen preflight')
  }
  const rows = manifestText.trim().split(/\r?\n/).map((line) => JSON.parse(line))
  const oldInventoryText = await readFile(path.join(mediaDir, 'media-inventory.jsonl'), 'utf8')
  const oldInventory = oldInventoryText.trim().split(/\r?\n/).map((line) => JSON.parse(line))
  const inventoryById = new Map(oldInventory.map((row) => [row.id, row]))
  if (rows.length !== 1000 || inventoryById.size !== rows.length) throw new Error('incomplete old cohort/media')
  const pkgPath = path.resolve(path.dirname(modulePath), '..', 'package.json')
  const pkgBytes = await readFile(pkgPath)
  const entryBytes = await readFile(modulePath)
  const pkg = JSON.parse(pkgBytes.toString('utf8'))
  if (pkg.name !== 'hyparquet' || pkg.version !== '1.31.3' || pkg.license !== 'MIT'
      || sha(pkgBytes) !== PACKAGE_SHA256 || sha(entryBytes) !== ENTRY_SHA256) {
    throw new Error('Parquet reader package hash/version/license mismatch')
  }
  const parquet = await import(pathToFileURL(modulePath).href)
  const stats = { range_requests: 0, range_response_bytes: 0 }
  const fetchRange = rangedFetch(stats)
  const byGroup = new Map()
  for (const row of rows) {
    const key = `${row.source_file}:${row.rowgroup_index}`
    if (!byGroup.has(key)) byGroup.set(key, [])
    byGroup.get(key).push(row)
  }
  const planned = new Map(plan.groups.map((row) => [`${row.file}:${row.index}`, row]))
  const shardBuffers = new Map()
  const corrected = []
  const correctedInventory = []
  const labelCounts = Object.fromEntries(LABELS.map((label) => [label, 0]))
  for (const [key, selected] of byGroup) {
    const [file] = key.split(':')
    const group = planned.get(key)
    if (!group || group.selected_rows !== selected.length) throw new Error(`rowgroup plan mismatch: ${key}`)
    let buffer = shardBuffers.get(file)
    if (!buffer) {
      const url = `https://huggingface.co/datasets/${DATASET}/resolve/${REVISION}/v0.02/${file}?download=true`
      buffer = await parquet.asyncBufferFromUrl({ url, fetch: fetchRange })
      shardBuffers.set(file, buffer)
    }
    const metadata = await parquet.parquetMetadataAsync(buffer)
    const selectedByRow = new Map(selected.map((row) => [row.source_row, row]))
    const records = await parquet.parquetReadObjects({
      file: buffer,
      metadata,
      rowStart: group.row_start,
      rowEnd: group.row_end,
      columns: ['label', 'speaker_id', 'file'],
      parsers,
      includeRowIndex: true,
    })
    let matched = 0
    for (const record of records) {
      const sourceRow = Number(record[parquet.rowIndex])
      const old = selectedByRow.get(sourceRow)
      if (!old) continue
      const sourceFile = text(record.file)
      const sourceClass = sourceFile.split('/')[0]
      const numericLabel = Number(record.label)
      if (LABELS[numericLabel] !== sourceClass) throw new Error(`ClassLabel/path mismatch for ${old.id}`)
      if (sha(text(record.speaker_id)) !== old.speaker_group_sha256
          || sha(sourceFile) !== old.source_path_sha256) {
        throw new Error(`identity changed while correcting target: ${old.id}`)
      }
      matched += 1
      const fixed = { ...old, label: numericLabel, target: sourceClass, source_path_class: sourceClass }
      corrected.push(fixed)
      labelCounts[sourceClass] += 1
      const media = inventoryById.get(old.id)
      if (!media || media.audio_sha256 === undefined) throw new Error(`missing old media: ${old.id}`)
      const oldPath = path.join(mediaDir, media.local_file)
      const newPath = path.join(outputDir, media.local_file)
      if (sha(await readFile(oldPath)) !== media.audio_sha256) {
        throw new Error(`old WAV hash mismatch for ${old.id}`)
      }
      await copyFile(oldPath, newPath, constants.COPYFILE_EXCL)
      if (sha(await readFile(newPath)) !== media.audio_sha256) {
        throw new Error(`copied WAV hash mismatch for ${old.id}`)
      }
      correctedInventory.push({ ...media, label: numericLabel, target: sourceClass, source_path_class: sourceClass })
    }
    if (matched !== selected.length) throw new Error(`selected rows missing from ${key}`)
  }
  if (corrected.length !== rows.length || LABELS.some((label) => labelCounts[label] !== 100)) {
    throw new Error(`corrected labels are not balanced: ${JSON.stringify(labelCounts)}`)
  }
  const originalOrder = new Map(rows.map((row, index) => [row.id, index]))
  corrected.sort((a, b) => originalOrder.get(a.id) - originalOrder.get(b.id))
  correctedInventory.sort((a, b) => originalOrder.get(a.id) - originalOrder.get(b.id))
  const correctedManifestText = corrected.map((row) => JSON.stringify(row)).join('\n') + '\n'
  const correctedInventoryText = correctedInventory.map((row) => JSON.stringify(row)).join('\n') + '\n'
  await writeFile(path.join(outputDir, 'cohort-manifest.jsonl'), correctedManifestText, { flag: 'wx' })
  await writeFile(path.join(outputDir, 'media-inventory.jsonl'), correctedInventoryText, { flag: 'wx' })
  const report = {
    schema_version: 1,
    status: 'SAME_FROZEN_RECORDS_CORRECTED_CLASSLABEL_MAPPING',
    source: `${DATASET}@${REVISION}/v0.02 official TRAIN only`,
    source_license: 'CC-BY-4.0',
    old_manifest_sha256: ORIGINAL_MANIFEST_SHA256,
    corrected_manifest_sha256: sha(correctedManifestText),
    corrected_inventory_sha256: sha(correctedInventoryText),
    same_sample_ids_and_source_rows: true,
    examples: corrected.length,
    speaker_groups: new Set(corrected.map((row) => row.speaker_group_sha256)).size,
    class_counts: labelCounts,
    numeric_label_to_name_mapping: Object.fromEntries(LABELS.map((label, index) => [index, label])),
    correction: 'The initial metadata selector used alphabetical class names as numeric IDs. The pinned dataset card and source file path identify the v0.02 numeric class map; targets now use that map. No audio samples, model weights, temperature, or options changed.',
    prior_evaluation_disposition: 'The metrics on the old manifest are invalid because its targets were permuted. Their report and predictions are preserved and must not be interpreted as model quality.',
    materialized_audio_source: 'Copied byte-identically from the previously verified selected-WAV artifact; no new audio payload was requested.',
    selected_wav_sha256_preserved: correctedInventory.every((row) => inventoryById.get(row.id).audio_sha256 === row.audio_sha256),
    metadata_transport: stats,
    audio_column_requested: false,
    inference_run: false,
    training_run: false,
    official_validation_or_test_read: false,
    sealed_audit_opened: false,
  }
  await writeFile(path.join(outputDir, 'label-correction-report.json'), `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' })
  console.log(JSON.stringify(report, null, 2))
}

main().catch((error) => {
  console.error(error.stack || error)
  process.exitCode = 1
})
