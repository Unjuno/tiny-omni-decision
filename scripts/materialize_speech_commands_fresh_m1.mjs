import { createHash } from 'node:crypto'
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

export const REVISION = 'a751309c0fd613e8a5d30d77900f30e8b42bc2da'
export const DATASET = 'google/speech_commands'
export const EXPECTED_REPORT_SHA256 = '34dff3cdafcd64188a637d336f46f094c2cc334fb29195b4b196bab959dd17ba'
export const EXPECTED_TRAIN_PLAN_SHA256 = '1ab0048dd24b32fcbbaee2b2528c96832b8d386658140b9526d05d6f656b6e5f'
export const EXPECTED_DEV_PLAN_SHA256 = '2e91f2072090062383bb234a07337c07b13d7382e5a1b22e8552df8eea009da2'
export const EXPECTED_PACKAGE_SHA256 = '8005da09189a9e358d94a678b42315bb38a368a9af5ef9823d457cb04ad8ddea'
export const EXPECTED_ENTRY_SHA256 = '9a61c1600b11d650609ea664c8de6e290046ac5045870cc00bfae956d3adcb86'

const LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
const utf8 = new TextDecoder()
const sha = (value) => createHash('sha256').update(value).digest('hex')
const parsers = {
  stringFromBytes: (bytes) => bytes,
  jsonFromBytes: (bytes) => bytes && JSON.parse(utf8.decode(bytes)),
}

const text = (value) => typeof value === 'string' ? value : utf8.decode(value)

function arg(argv, name) {
  const index = argv.indexOf(`--${name}`)
  if (index < 0 || !argv[index + 1]) throw new Error(`missing --${name}`)
  return argv[index + 1]
}

export function pcmWavInfo(bytes) {
  if (bytes.length < 44 || bytes.toString('ascii', 0, 4) !== 'RIFF'
      || bytes.toString('ascii', 8, 12) !== 'WAVE') throw new Error('selected audio is not a WAV')
  const format = bytes.readUInt16LE(20)
  const channels = bytes.readUInt16LE(22)
  const sampleRate = bytes.readUInt32LE(24)
  const bitsPerSample = bytes.readUInt16LE(34)
  if (format !== 1 || channels !== 1 || sampleRate !== 16_000 || bitsPerSample !== 16) {
    throw new Error('expected mono PCM16 16 kHz Speech Commands audio')
  }
  return { channels, sample_rate: sampleRate, bits_per_sample: bitsPerSample, bytes: bytes.length }
}

export function assertNoCrossSplitAudioOverlap(rows) {
  const byHash = new Map()
  for (const row of rows) {
    const prior = byHash.get(row.audio_sha256)
    if (prior && prior !== row.split) throw new Error(`identical audio content crosses splits: ${row.audio_sha256}`)
    byHash.set(row.audio_sha256, row.split)
  }
}

function rangeFetch(stats) {
  return async (url, init = {}) => {
    const headers = new Headers(init.headers)
    const isHead = init.method === 'HEAD'
    if (!isHead && !headers.has('range')) throw new Error('refusing non-range network request')
    const response = await fetch(url, { ...init, headers, signal: AbortSignal.timeout(90_000) })
    if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`)
    const host = new URL(response.url).hostname
    if (host !== 'huggingface.co' && !host.endsWith('.hf.co')) throw new Error(`unexpected redirect host: ${host}`)
    if (headers.has('range')) {
      if (response.status !== 206) throw new Error(`range request returned HTTP ${response.status}`)
      stats.range_requests += 1
      stats.range_response_bytes += Number(response.headers.get('content-length') || 0)
    }
    return response
  }
}

async function loadJsonl(file) {
  return (await readFile(file, 'utf8')).trim().split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line))
}

function placeRowsInRowGroups(rows, metadata) {
  const byGroup = new Map()
  let rowStart = 0
  for (const group of metadata.row_groups) {
    const rowEnd = rowStart + Number(group.num_rows)
    for (const row of rows) {
      if (row.source_row >= rowStart && row.source_row < rowEnd) {
        if (!byGroup.has(rowStart)) byGroup.set(rowStart, { rowEnd, rows: [] })
        byGroup.get(rowStart).rows.push(row)
      }
    }
    rowStart = rowEnd
  }
  const found = [...byGroup.values()].reduce((sum, group) => sum + group.rows.length, 0)
  if (found !== rows.length) throw new Error(`source row outside shard row groups: ${found}/${rows.length}`)
  return byGroup
}

async function main() {
  const trainPlanPath = path.resolve(arg(process.argv, 'train-plan'))
  const devPlanPath = path.resolve(arg(process.argv, 'validation-plan'))
  const preflightPath = path.resolve(arg(process.argv, 'preflight-report'))
  const outputDir = path.resolve(arg(process.argv, 'output-dir'))
  const modulePath = path.resolve(arg(process.argv, 'hyparquet-entry'))
  await mkdir(outputDir, { recursive: true })
  if ((await readdir(outputDir)).length) throw new Error(`output directory must be empty: ${outputDir}`)

  const trainPlanBytes = await readFile(trainPlanPath)
  const devPlanBytes = await readFile(devPlanPath)
  const reportBytes = await readFile(preflightPath)
  const hashes = {
    train: sha(trainPlanBytes),
    validation: sha(devPlanBytes),
    preflight: sha(reportBytes),
  }
  if (hashes.train !== EXPECTED_TRAIN_PLAN_SHA256
      || hashes.validation !== EXPECTED_DEV_PLAN_SHA256
      || hashes.preflight !== EXPECTED_REPORT_SHA256) {
    throw new Error(`frozen plan/report hash mismatch: ${JSON.stringify(hashes)}`)
  }
  const preflight = JSON.parse(reportBytes.toString('utf8'))
  if (preflight.status !== 'METADATA_ONLY_FRESH_M1_PREFLIGHT_COMPLETE'
      || preflight.source !== `${DATASET}@${REVISION}/v0.02 official TRAIN only`
      || preflight.safety.audio_column_requested !== false
      || preflight.safety.media_downloaded !== false
      || preflight.safety.model_loaded !== false
      || preflight.safety.sealed_audit_opened !== false) {
    throw new Error('metadata-only preflight report does not match the frozen source/safety contract')
  }
  const splitRows = {
    train: await loadJsonl(trainPlanPath),
    validation: await loadJsonl(devPlanPath),
  }
  if (splitRows.train.length !== 1500 || splitRows.validation.length !== 1000) {
    throw new Error('frozen preflight plans have unexpected row counts')
  }
  const rowKeys = new Set()
  const identities = Object.fromEntries(['train', 'validation'].map((split) => [split, {
    groups: new Set(), paths: new Set(), classCounts: Array(LABELS.length).fill(0),
  }]))
  const trainPerSpeaker = new Map()
  for (const [split, rows] of Object.entries(splitRows)) {
    for (const row of rows) {
      const key = `${row.source_file}:${row.source_row}`
      if (row.split !== 'train' || !Number.isInteger(row.label)
          || row.label < 0 || row.label >= LABELS.length || row.target !== LABELS[row.label]
          || row.source !== `${DATASET}@${REVISION}`) {
        throw new Error(`unexpected source identity in ${split} plan: ${row.id}`)
      }
      if (rowKeys.has(key)) throw new Error(`duplicate source row across plans: ${key}`)
      rowKeys.add(key)
      const identity = identities[split]
      identity.groups.add(row.speaker_group_sha256)
      identity.paths.add(row.source_path_sha256)
      identity.classCounts[row.label] += 1
      if (split === 'train') {
        const count = (trainPerSpeaker.get(row.speaker_group_sha256) ?? 0) + 1
        if (count > 5) throw new Error(`train speaker cap exceeded: ${row.speaker_group_sha256}`)
        trainPerSpeaker.set(row.speaker_group_sha256, count)
      }
    }
  }
  for (const split of ['train', 'validation']) {
    const expected = split === 'train' ? 150 : 100
    if (identities[split].classCounts.some((count) => count !== expected)) {
      throw new Error(`${split} plan does not have the frozen per-class quota`)
    }
  }
  const intersections = (left, right) => [...left].filter((value) => right.has(value)).length
  if (intersections(identities.train.groups, identities.validation.groups)
      || intersections(identities.train.paths, identities.validation.paths)) {
    throw new Error('train/validation source identity overlap before media retrieval')
  }
  const validationClassSpeakers = LABELS.map((_, label) => new Set(
    splitRows.validation.filter((row) => row.label === label).map((row) => row.speaker_group_sha256),
  ).size)
  if (identities.validation.groups.size !== 200 || validationClassSpeakers.some((count) => count < 100)) {
    throw new Error('development plan group/class support differs from the frozen quota')
  }

  const pkgPath = path.resolve(path.dirname(modulePath), '..', 'package.json')
  const pkgBytes = await readFile(pkgPath)
  const entryBytes = await readFile(modulePath)
  const pkg = JSON.parse(pkgBytes.toString('utf8'))
  if (pkg.name !== 'hyparquet' || pkg.version !== '1.31.3' || pkg.license !== 'MIT'
      || sha(pkgBytes) !== EXPECTED_PACKAGE_SHA256 || sha(entryBytes) !== EXPECTED_ENTRY_SHA256) {
    throw new Error('pinned MIT Parquet reader identity/hash mismatch')
  }
  const parquet = await import(pathToFileURL(modulePath).href)
  const stats = { range_requests: 0, range_response_bytes: 0 }
  const fetch = rangeFetch(stats)
  const shardRows = new Map()
  for (const [split, rows] of Object.entries(splitRows)) {
    for (const row of rows) {
      const key = `${split}|${row.source_file}`
      if (!shardRows.has(key)) shardRows.set(key, [])
      shardRows.get(key).push(row)
    }
  }

  const inventoryRows = []
  const shardBuffers = new Map()
  let rowgroupsRead = 0
  for (const [key, selected] of shardRows) {
    const [split, file] = key.split('|')
    let buffer = shardBuffers.get(file)
    if (!buffer) {
      const url = `https://huggingface.co/datasets/${DATASET}/resolve/${REVISION}/v0.02/${file}?download=true`
      buffer = await parquet.asyncBufferFromUrl({ url, fetch })
      shardBuffers.set(file, buffer)
    }
    const metadata = await parquet.parquetMetadataAsync(buffer)
    const groups = placeRowsInRowGroups(selected, metadata)
    for (const [rowStart, group] of groups) {
      const { rowEnd, rows: groupRows } = group
      rowgroupsRead += 1
      const expectedRows = new Map(groupRows.map((row) => [row.source_row, row]))
      const records = await parquet.parquetReadObjects({
        file: buffer,
        metadata,
        rowStart,
        rowEnd,
        columns: ['label', 'speaker_id', 'file', 'audio'],
        parsers,
        includeRowIndex: true,
      })
      let matched = 0
      for (const record of records) {
        const sourceRow = Number(record[parquet.rowIndex])
        const frozen = expectedRows.get(sourceRow)
        if (!frozen) continue
        if (Number(record.label) !== frozen.label
            || sha(text(record.speaker_id)) !== frozen.speaker_group_sha256
            || sha(text(record.file)) !== frozen.source_path_sha256) {
          throw new Error(`source identity mismatch for ${split}/${frozen.id}`)
        }
        const payload = record.audio?.bytes ?? record.audio?.data
        const wav = payload instanceof Uint8Array ? Buffer.from(payload)
          : typeof payload === 'string' ? Buffer.from(payload, 'latin1') : undefined
        if (!wav) throw new Error(`missing binary audio bytes for ${split}/${frozen.id}`)
        const wavInfo = pcmWavInfo(wav)
        const audioHash = sha(wav)
        const filename = `${frozen.id}.wav`
        const splitDir = path.join(outputDir, split)
        await mkdir(splitDir, { recursive: true })
        await writeFile(path.join(splitDir, filename), wav, { flag: 'wx' })
        inventoryRows.push({
          ...frozen,
          materialized_split: split,
          local_file: `${split}/${filename}`,
          audio_sha256: audioHash,
          wav: wavInfo,
        })
        expectedRows.delete(sourceRow)
        matched += 1
      }
      if (expectedRows.size || matched !== groupRows.length) {
        throw new Error(`selected row count mismatch for ${split}/${file}:${rowStart}; ${matched}/${groupRows.length}`)
      }
      console.log(JSON.stringify({ split, shard: file, rowgroup_start: rowStart, selected: matched }))
    }
  }
  if (inventoryRows.length !== 2500 || new Set(inventoryRows.map((row) => row.id)).size !== 2500) {
    throw new Error(`materialized plan is incomplete or duplicate: ${inventoryRows.length}/2500`)
  }
  assertNoCrossSplitAudioOverlap(inventoryRows)
  const wavHashesBySplit = Object.fromEntries(['train', 'validation'].map((split) => {
    const rows = inventoryRows.filter((row) => row.materialized_split === split)
    return [split, {
      examples: rows.length,
      wav_bytes: rows.reduce((sum, row) => sum + row.wav.bytes, 0),
      unique_wav_hashes: new Set(rows.map((row) => row.audio_sha256)).size,
    }]
  }))
  inventoryRows.sort((a, b) => a.materialized_split.localeCompare(b.materialized_split) || a.id.localeCompare(b.id))
  await writeFile(path.join(outputDir, 'media-inventory.jsonl'),
    inventoryRows.map((row) => JSON.stringify(row)).join('\n') + '\n', { flag: 'wx' })
  const report = {
    schema_version: 1,
    status: 'FRESH_M1_TRAIN_AND_DEVELOPMENT_MEDIA_MATERIALIZED',
    source: `${DATASET}@${REVISION}`,
    source_split: 'official TRAIN shards only',
    source_license: 'CC-BY-4.0',
    source_license_evidence: `https://huggingface.co/datasets/${DATASET}/blob/${REVISION}/README.md`,
    attribution: 'Pete Warden, Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition (2018), https://arxiv.org/abs/1804.03209; Google Speech Commands v0.02, CC BY 4.0.',
    preflight_hashes: hashes,
    parser: {
      name: pkg.name,
      version: pkg.version,
      license: pkg.license,
      package_json_sha256: EXPECTED_PACKAGE_SHA256,
      entrypoint_sha256: EXPECTED_ENTRY_SHA256,
    },
    examples_written: inventoryRows.length,
    unique_speaker_groups: {
      train: new Set(inventoryRows.filter((row) => row.materialized_split === 'train').map((row) => row.speaker_group_sha256)).size,
      validation: new Set(inventoryRows.filter((row) => row.materialized_split === 'validation').map((row) => row.speaker_group_sha256)).size,
    },
    wav_hashes_by_split: wavHashesBySplit,
    rowgroups_read: rowgroupsRead,
    transport: stats,
    cross_split_speaker_overlap: 0,
    cross_split_source_row_overlap: 0,
    cross_split_source_path_overlap: 0,
    cross_split_audio_content_overlap: 0,
    official_validation_or_test_shards_read: false,
    collateral_audio_persisted: false,
    model_loaded: false,
    inference_run: false,
    training_run: false,
    sealed_audit_opened: false,
    paid_compute: false,
  }
  await writeFile(path.join(outputDir, 'materialization-report.json'), `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' })
  console.log(JSON.stringify({ status: report.status, examples: report.examples_written,
    wav_hashes_by_split: report.wav_hashes_by_split, transport: report.transport }, null, 2))
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  main().catch((error) => {
    console.error(error.stack || error)
    process.exitCode = 1
  })
}
