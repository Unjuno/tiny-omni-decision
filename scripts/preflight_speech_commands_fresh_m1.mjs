import { createHash } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

export const REVISION = 'a751309c0fd613e8a5d30d77900f30e8b42bc2da'
export const DATASET = 'google/speech_commands'
export const LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
const SOURCE_LABELS = [
  'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
  'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
  'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow',
  'backward', 'forward', 'follow', 'learn', 'visual', '_silence_',
]
const TRAIN_FILES = Array.from({ length: 6 }, (_, index) =>
  `train-${String(index).padStart(5, '0')}-of-00006.parquet`,
)
const utf8 = new TextDecoder()
const sha = (value) => createHash('sha256').update(value).digest('hex')
const parsers = {
  stringFromBytes: (bytes) => bytes && utf8.decode(bytes),
  jsonFromBytes: (bytes) => bytes && JSON.parse(utf8.decode(bytes)),
}

function argsFromCli(argv) {
  const values = new Map()
  for (let index = 2; index < argv.length; index += 2) {
    const key = argv[index]
    const value = argv[index + 1]
    if (!key?.startsWith('--') || value === undefined) throw new Error(`bad argument: ${key}`)
    values.set(key.slice(2), value)
  }
  for (const key of ['historical-dir', 'observed-manifest', 'output-dir', 'hyparquet-entry']) {
    if (!values.has(key)) throw new Error(`missing --${key}`)
  }
  return Object.fromEntries(values)
}

async function readJsonl(file) {
  const text = await readFile(file, 'utf8')
  return text.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line))
}

async function historicalGroups(dir) {
  const groups = new Set()
  const rows = []
  for (const name of ['train.jsonl', 'validation.jsonl', 'cohort-manifest.jsonl']) {
    try {
      const records = await readJsonl(path.join(dir, name))
      for (const row of records) {
        if (row.speaker_group_sha256) groups.add(row.speaker_group_sha256)
        rows.push(row)
      }
    } catch (error) {
      if (error.code !== 'ENOENT') throw error
    }
  }
  const deduplicated = new Map()
  for (const row of rows) {
    const key = row.source_file && Number.isInteger(Number(row.source_row))
      ? `${row.source_file}:${row.source_row}`
      : `id:${row.id ?? sha(JSON.stringify(row))}`
    deduplicated.set(key, row)
  }
  return { groups, rows: [...deduplicated.values()] }
}

export function chooseGroupRows(rows, { seed = 17, devGroupCount = 200,
  devPerClass = 100, trainPerClass = 150, maxTrainRowsPerSpeaker = 5 } = {}) {
  const groups = [...new Set(rows.map((row) => row.speaker_group_sha256))].sort((a, b) =>
    sha(`r2-speechcmd-m1-seed${seed}|${a}`).localeCompare(
      sha(`r2-speechcmd-m1-seed${seed}|${b}`),
    ),
  )
  if (groups.length < devGroupCount + 2) {
    throw new Error(`only ${groups.length} eligible groups; need at least ${devGroupCount + 2}`)
  }
  const devGroups = new Set(groups.slice(0, devGroupCount))
  const byClass = (classId, split) => rows.filter((row) =>
    row.label === classId && (split === 'dev'
      ? devGroups.has(row.speaker_group_sha256)
      : !devGroups.has(row.speaker_group_sha256)),
  )
  const rankRows = (items, key) => [...items].sort((a, b) =>
    sha(`${seed}|${key}|${a.source_file}|${a.source_row}`).localeCompare(
      sha(`${seed}|${key}|${b.source_file}|${b.source_row}`),
    ),
  )
  const selectPerClass = (split, quota) => {
    const selected = []
    for (let classId = 0; classId < LABELS.length; classId += 1) {
      const seenSpeakers = new Set()
      const ranked = byClass(classId, split).sort((a, b) =>
        sha(`${seed}|${split}|${classId}|${a.source_file}|${a.source_row}`).localeCompare(
          sha(`${seed}|${split}|${classId}|${b.source_file}|${b.source_row}`),
        ),
      )
      let count = 0
      for (const row of ranked) {
        if (seenSpeakers.has(row.speaker_group_sha256)) continue
        if (split === 'train' && selected.filter(
          (item) => item.speaker_group_sha256 === row.speaker_group_sha256,
        ).length >= maxTrainRowsPerSpeaker) continue
        selected.push(row)
        seenSpeakers.add(row.speaker_group_sha256)
        count += 1
        if (count === quota) break
      }
      if (count !== quota) {
        throw new Error(`${split} class ${LABELS[classId]} has ${count}/${quota} distinct speakers`)
      }
    }
    return selected.sort((a, b) => a.label - b.label
      || a.source_file.localeCompare(b.source_file) || a.source_row - b.source_row)
  }
  const train = selectPerClass('train', trainPerClass)
  const validation = []
  const devCounts = Array(LABELS.length).fill(0)
  const orderedDevGroups = groups.slice(0, devGroupCount)
  for (const group of orderedDevGroups) {
    const groupRows = rows.filter((row) => row.speaker_group_sha256 === group)
    const availableLabels = [...new Set(groupRows.map((row) => row.label))]
      .filter((label) => devCounts[label] < devPerClass)
      .sort((left, right) => devCounts[left] - devCounts[right]
        || sha(`${seed}|dev-seed|${group}|${left}`).localeCompare(
          sha(`${seed}|dev-seed|${group}|${right}`),
        ),
      )
    if (!availableLabels.length) throw new Error(`development group ${group} has no quota remaining`)
    const label = availableLabels[0]
    const row = rankRows(groupRows.filter((item) => item.label === label), `dev-seed-${group}-${label}`)[0]
    validation.push(row)
    devCounts[label] += 1
  }
  for (let classId = 0; classId < LABELS.length; classId += 1) {
    const seenSpeakers = new Set(validation
      .filter((row) => row.label === classId)
      .map((row) => row.speaker_group_sha256))
    for (const row of rankRows(byClass(classId, 'dev'), `dev-fill-${classId}`)) {
      if (devCounts[classId] === devPerClass) break
      if (seenSpeakers.has(row.speaker_group_sha256)) continue
      validation.push(row)
      seenSpeakers.add(row.speaker_group_sha256)
      devCounts[classId] += 1
    }
    if (devCounts[classId] !== devPerClass) {
      throw new Error(`development class ${LABELS[classId]} has ${devCounts[classId]}/${devPerClass} distinct speakers`)
    }
  }
  validation.sort((a, b) => a.label - b.label
    || a.source_file.localeCompare(b.source_file) || a.source_row - b.source_row)
  const trainGroups = new Set(train.map((row) => row.speaker_group_sha256))
  const validationGroups = new Set(validation.map((row) => row.speaker_group_sha256))
  if (validationGroups.size !== devGroupCount) {
    throw new Error(`development rows cover ${validationGroups.size}/${devGroupCount} reserved groups`)
  }
  if ([...trainGroups].some((group) => validationGroups.has(group))) {
    throw new Error('train/development speaker group overlap')
  }
  const splitGroups = new Set([...trainGroups, ...validationGroups])
  if (trainGroups.size < 100 || Math.ceil(trainGroups.size * 0.2) < 30) {
    throw new Error('training sample does not support minimum fit/calibration group counts')
  }
  const calibrationGroups = new Set([...trainGroups]
    .sort((a, b) => sha(`17:calibration:${a}`).localeCompare(sha(`17:calibration:${b}`)))
    .slice(0, Math.ceil(trainGroups.size * 0.2)))
  const fitRows = train.filter((row) => !calibrationGroups.has(row.speaker_group_sha256))
  const calibrationRows = train.filter((row) => calibrationGroups.has(row.speaker_group_sha256))
  const minClassRows = (items) => Math.min(...LABELS.map((_, label) =>
    items.filter((row) => row.label === label).length,
  ))
  if (minClassRows(fitRows) < 20 || minClassRows(calibrationRows) < 20) {
    throw new Error('speaker-held-out fit/calibration split has fewer than 20 rows in a class')
  }
  return {
    train, validation, trainGroups, validationGroups, splitGroups, devGroups,
    fitRows, calibrationRows, calibrationGroups,
  }
}

function rangedFetch(stats, proxyBase) {
  return async (url, init = {}) => {
    const headers = new Headers(init.headers)
    for (let attempt = 0; attempt < 6; attempt += 1) {
      try {
        const response = proxyBase
          ? await fetch(`${proxyBase}/range`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url, method: init.method ?? 'GET', headers: Object.fromEntries(headers) }),
            signal: AbortSignal.timeout(120000),
          })
          : await fetch(url, { ...init, headers, signal: AbortSignal.timeout(60000) })
        if (response.status === 429 || response.status >= 500) {
          await response.body?.cancel()
          await new Promise((resolve) => setTimeout(resolve, Math.min(1000 * 2 ** attempt, 20000)))
          continue
        }
        const finalHost = new URL(response.url).hostname
        if (finalHost !== 'huggingface.co' && !finalHost.endsWith('.hf.co')) {
          throw new Error(`unexpected redirect host: ${finalHost}`)
        }
        if (headers.has('range') && response.status === 200
            && Number(response.headers.get('content-length') ?? 0) > 8 * 1024 * 1024) {
          throw new Error('upstream ignored Range; refusing full shard download')
        }
        stats.rangeRequests += headers.has('range') ? 1 : 0
        stats.rangeResponseBytes += Number(response.headers.get('content-length') ?? 0)
        const etag = response.headers.get('etag')
        if (etag) stats.etags.add(etag)
        return response
      } catch (error) {
        if (attempt === 5) throw error
        await new Promise((resolve) => setTimeout(resolve, Math.min(1000 * 2 ** attempt, 20000)))
      }
    }
    throw new Error('range request retry budget exhausted')
  }
}

async function scanShard(name, parquet, fetch) {
  const url = `https://huggingface.co/datasets/${DATASET}/resolve/${REVISION}/v0.02/${name}?download=true`
  const file = await parquet.asyncBufferFromUrl({ url, fetch })
  const metadata = await parquet.parquetMetadataAsync(file)
  const entry = metadata.key_value_metadata?.find(({ key }) => key === 'huggingface')
  if (!entry) throw new Error(`missing ClassLabel metadata: ${name}`)
  const sourceLabels = JSON.parse(entry.value).info?.features?.label?.names
  if (JSON.stringify(sourceLabels) !== JSON.stringify(SOURCE_LABELS)
      || JSON.stringify(SOURCE_LABELS.slice(0, LABELS.length)) !== JSON.stringify(LABELS)) {
    throw new Error(`unexpected pinned label mapping in ${name}: ${JSON.stringify(sourceLabels)}`)
  }
  const rows = []
  let rowStart = 0
  for (const group of metadata.row_groups) {
    const labelColumn = group.columns.find(({ meta_data }) => meta_data.path_in_schema[0] === 'label')
    const stats = labelColumn?.meta_data.statistics
    const low = Number(stats?.min_value ?? stats?.min)
    const high = Number(stats?.max_value ?? stats?.max)
    const count = Number(group.num_rows)
    if (!Number.isInteger(low) || !Number.isInteger(high)) {
      throw new Error(`missing label statistics in ${name}:${rowStart}`)
    }
    if (LABELS.some((_, label) => label >= low && label <= high)) {
      const values = await parquet.parquetReadObjects({
        file, metadata, rowStart, rowEnd: rowStart + count,
        columns: ['label', 'speaker_id', 'file'], parsers,
        includeRowIndex: true,
      })
      for (const row of values) {
        const label = Number(row.label)
        const speaker = String(row.speaker_id ?? '').trim()
        const filePath = String(row.file ?? '').trim()
        if (label < 0 || label >= LABELS.length || !speaker || speaker === 'unknown' || !filePath) continue
        const sourceRow = Number(row[parquet.rowIndex])
        rows.push({
          source: `${DATASET}@${REVISION}`,
          split: 'train',
          source_file: name,
          source_row: sourceRow,
          source_path_sha256: sha(filePath),
          speaker_group_sha256: sha(speaker),
          label,
          target: LABELS[label],
          id: `scm1-${sha(`${REVISION}|${name}|${sourceRow}`).slice(0, 20)}`,
        })
      }
    }
    rowStart += count
  }
  return { rows, rowCount: Number(metadata.num_rows), rowGroups: metadata.row_groups.length }
}

async function main() {
  const args = argsFromCli(process.argv)
  const out = path.resolve(args['output-dir'])
  await mkdir(out, { recursive: true })
  if ((await (await import('node:fs/promises')).readdir(out)).length) {
    throw new Error(`output directory must be empty: ${out}`)
  }
  const historicalDir = path.resolve(args['historical-dir'])
  const observedManifest = path.resolve(args['observed-manifest'])
  const historical = await historicalGroups(historicalDir)
  const observed = await readJsonl(observedManifest)
  const excludedGroups = new Set(historical.groups)
  const excludedRows = new Set()
  const excludedPaths = new Set()
  for (const row of [...historical.rows, ...observed]) {
    if (row.speaker_group_sha256) excludedGroups.add(row.speaker_group_sha256)
    if (row.source_file && Number.isInteger(Number(row.source_row))) {
      excludedRows.add(`${row.source_file}:${row.source_row}`)
    }
    if (row.source_path_sha256) excludedPaths.add(row.source_path_sha256)
  }
  const stats = { rangeRequests: 0, rangeResponseBytes: 0, etags: new Set() }
  const fetch = rangedFetch(stats, args['range-proxy'])
  const parquet = await import(pathToFileURL(path.resolve(args['hyparquet-entry'])).href)
  const files = {}
  const candidates = []
  for (const name of TRAIN_FILES) {
    const result = await scanShard(name, parquet, fetch)
    files[name] = { rows: result.rowCount, row_groups: result.rowGroups, target_rows_after_filters: result.rows.length }
    candidates.push(...result.rows)
  }
  const eligible = candidates.filter((row) =>
    !excludedGroups.has(row.speaker_group_sha256)
      && !excludedRows.has(`${row.source_file}:${row.source_row}`)
      && !excludedPaths.has(row.source_path_sha256),
  )
  const byLabel = Object.fromEntries(LABELS.map((label, id) => {
    const rows = eligible.filter((row) => row.label === id)
    return [label, {
      rows: rows.length,
      unique_speakers: new Set(rows.map((row) => row.speaker_group_sha256)).size,
    }]
  }))
  const plan = chooseGroupRows(eligible)
  for (const [name, rows] of [['train', plan.train], ['validation', plan.validation]]) {
    await writeFile(path.join(out, `${name}-plan.jsonl`),
      rows.map((row) => JSON.stringify(row)).join('\n') + '\n', { flag: 'wx' })
  }
  const report = {
    schema_version: 1,
    status: 'METADATA_ONLY_FRESH_M1_PREFLIGHT_COMPLETE',
    source: `${DATASET}@${REVISION}/v0.02 official TRAIN only`,
    usage: 'metadata columns label/speaker_id/file only; audio column was not requested',
    historical_manifest_rows: historical.rows.length,
    historical_unique_speaker_groups: historical.groups.size,
    observed_cohort_rows: observed.length,
    observed_unique_speaker_groups: new Set(observed.map((row) => row.speaker_group_sha256)).size,
    excluded_unique_groups: excludedGroups.size,
    eligible_source_rows: eligible.length,
    eligible_unique_speaker_groups: new Set(eligible.map((row) => row.speaker_group_sha256)).size,
    eligible_by_class: byLabel,
    selected_train_rows: plan.train.length,
    selected_train_unique_speakers: plan.trainGroups.size,
    estimated_fit_rows: plan.fitRows.length,
    estimated_fit_speakers: plan.trainGroups.size - plan.calibrationGroups.size,
    estimated_fit_class_counts: Object.fromEntries(LABELS.map((label) => [label, plan.fitRows.filter((row) => row.target === label).length])),
    estimated_calibration_rows: plan.calibrationRows.length,
    estimated_calibration_speakers: plan.calibrationGroups.size,
    estimated_calibration_class_counts: Object.fromEntries(LABELS.map((label) => [label, plan.calibrationRows.filter((row) => row.target === label).length])),
    selected_train_class_counts: Object.fromEntries(LABELS.map((label) => [label, plan.train.filter((row) => row.target === label).length])),
    selected_development_rows: plan.validation.length,
    selected_development_unique_speakers: plan.validationGroups.size,
    selected_development_class_counts: Object.fromEntries(LABELS.map((label) => [label, plan.validation.filter((row) => row.target === label).length])),
    train_development_group_overlap: 0,
    historical_or_observed_speaker_overlap: 0,
    files,
    transport: { range_requests: stats.rangeRequests, range_response_bytes: stats.rangeResponseBytes, etags: [...stats.etags] },
    safety: {
      audio_column_requested: false,
      validation_or_test_shards_read: false,
      media_downloaded: false,
      model_loaded: false,
      inference_run: false,
      training_run: false,
      sealed_audit_opened: false,
      gpu_used: false,
      paid_compute: false,
    },
    gate: 'Frozen quotas: 150 train and 100 development rows per class; 200 development groups; 20% speaker-held-out calibration split internal to train. Any failure aborts before media access.',
  }
  await writeFile(path.join(out, 'preflight-report.json'), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' })
  console.log(JSON.stringify(report, null, 2))
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  main().catch((error) => {
    console.error(error.stack ?? error)
    process.exitCode = 1
  })
}
