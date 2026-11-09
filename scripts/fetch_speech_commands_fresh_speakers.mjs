import { createHash } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const REVISION = 'a751309c0fd613e8a5d30d77900f30e8b42bc2da'
const DATASET = 'google/speech_commands'
const LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
const EXPECTED_MANIFEST_SHA256 = '96c6fe86298ef234e3b49546df8e07858049283a1e1334e4982a5bdbad5daaa4'
const EXPECTED_PACKAGE_SHA256 = '8005da09189a9e358d94a678b42315bb38a368a9af5ef9823d457cb04ad8ddea'
const EXPECTED_ENTRY_SHA256 = '9a61c1600b11d650609ea664c8de6e290046ac5045870cc00bfae956d3adcb86'
const utf8 = new TextDecoder()
const sha = (value) => createHash('sha256').update(value).digest('hex')
const parsers = {
  stringFromBytes: (bytes) => bytes && utf8.decode(bytes),
  jsonFromBytes: (bytes) => bytes && JSON.parse(utf8.decode(bytes)),
}

function arg(argv, name) {
  const index = argv.indexOf(`--${name}`)
  if (index < 0 || !argv[index + 1]) throw new Error(`missing --${name}`)
  return argv[index + 1]
}

function rangedFetch(stats) {
  return async (url, init = {}) => {
    const headers = new Headers(init.headers)
    if (init.method !== 'HEAD' && !headers.has('range')) {
      throw new Error('refusing non-range network request')
    }
    const response = await fetch(url, { ...init, headers, signal: AbortSignal.timeout(90_000) })
    if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`)
    if (headers.has('range')) {
      stats.rangeRequests += 1
      stats.rangeResponseBytes += Number(response.headers.get('content-length') || 0)
    }
    return response
  }
}

function sha256Bytes(bytes) {
  return createHash('sha256').update(bytes).digest('hex')
}

function pcmWavInfo(bytes) {
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

async function main() {
  const manifestPath = path.resolve(arg(process.argv, 'manifest'))
  const planPath = path.resolve(arg(process.argv, 'plan'))
  const outputDir = path.resolve(arg(process.argv, 'output-dir'))
  const modulePath = path.resolve(arg(process.argv, 'hyparquet-entry'))
  const sourceReportPath = path.resolve(arg(process.argv, 'preflight-report'))
  await mkdir(outputDir, { recursive: true })
  const directory = await import('node:fs/promises')
  if ((await directory.readdir(outputDir)).length) throw new Error('output directory must be empty')

  const manifestText = await readFile(manifestPath, 'utf8')
  const manifestHash = sha(manifestText)
  if (manifestHash !== EXPECTED_MANIFEST_SHA256) throw new Error('frozen cohort manifest hash mismatch')
  const rows = manifestText.trim().split(/\r?\n/).map((line) => JSON.parse(line))
  if (rows.length !== 1000 || new Set(rows.map((row) => row.id)).size !== rows.length) {
    throw new Error('frozen cohort must contain 1,000 unique IDs')
  }
  const sourceReport = JSON.parse(await readFile(sourceReportPath, 'utf8'))
  if (sourceReport.dataset !== `${DATASET}@${REVISION}`
      || sourceReport.safety.media_downloaded !== false
      || sourceReport.cohort_selection.manifest_sha256 !== EXPECTED_MANIFEST_SHA256) {
    throw new Error('metadata preflight report does not match the frozen dataset/cohort')
  }
  const planText = await readFile(planPath, 'utf8')
  const plan = JSON.parse(planText)
  if (plan.selected_rows !== rows.length || plan.count !== 30) {
    throw new Error('unexpected frozen payload plan')
  }
  const pkgPath = path.resolve(path.dirname(modulePath), '..', 'package.json')
  const pkgBytes = await readFile(pkgPath)
  const entryBytes = await readFile(modulePath)
  const pkg = JSON.parse(pkgBytes.toString('utf8'))
  if (pkg.name !== 'hyparquet' || pkg.version !== '1.31.3' || pkg.license !== 'MIT') {
    throw new Error('unexpected pinned Parquet reader package')
  }
  if (sha(pkgBytes) !== EXPECTED_PACKAGE_SHA256 || sha(entryBytes) !== EXPECTED_ENTRY_SHA256) {
    throw new Error('pinned Parquet reader package hash mismatch')
  }
  const parquet = await import(pathToFileURL(modulePath).href)
  const stats = { rangeRequests: 0, rangeResponseBytes: 0 }
  const fetchRange = rangedFetch(stats)
  const byGroup = new Map()
  for (const row of rows) {
    const key = `${row.source_file}:${row.rowgroup_index}`
    if (!byGroup.has(key)) byGroup.set(key, [])
    byGroup.get(key).push(row)
  }
  const planned = new Map(plan.groups.map((group) => [`${group.file}:${group.index}`, group]))
  if (planned.size !== byGroup.size) throw new Error('manifest and row-group plan disagree')

  const outputRows = []
  const shardBuffers = new Map()
  for (const [key, selected] of byGroup) {
    const [file] = key.split(':')
    const group = planned.get(key)
    if (!group || group.selected_rows !== selected.length) throw new Error(`plan mismatch for ${key}`)
    let buffer = shardBuffers.get(file)
    if (!buffer) {
      const url = `https://huggingface.co/datasets/${DATASET}/resolve/${REVISION}/v0.02/${file}?download=true`
      buffer = await parquet.asyncBufferFromUrl({ url, fetch: fetchRange })
      shardBuffers.set(file, buffer)
    }
    const metadata = await parquet.parquetMetadataAsync(buffer)
    const selectedByIndex = new Map(selected.map((row) => [row.source_row, row]))
    const tableRows = await parquet.parquetReadObjects({
      file: buffer,
      metadata,
      rowStart: group.row_start,
      rowEnd: group.row_end,
      columns: ['label', 'speaker_id', 'file', 'audio'],
      parsers,
      includeRowIndex: true,
    })
    let matched = 0
    for (const record of tableRows) {
      const sourceRow = Number(record[parquet.rowIndex])
      const frozen = selectedByIndex.get(sourceRow)
      if (!frozen) continue
      if (Number(record.label) !== frozen.label
          || sha(String(record.speaker_id)) !== frozen.speaker_group_sha256
          || sha(String(record.file)) !== frozen.source_path_sha256) {
        throw new Error(`source identity mismatch for ${frozen.id}`)
      }
      const audio = record.audio
      const payload = audio?.bytes ?? audio?.data
      if (!(payload instanceof Uint8Array)) throw new Error(`missing binary audio bytes for ${frozen.id}`)
      const wav = Buffer.from(payload)
      const wavInfo = pcmWavInfo(wav)
      const audioHash = sha256Bytes(wav)
      const audioPath = path.join(outputDir, `${frozen.id}.wav`)
      await writeFile(audioPath, wav, { flag: 'wx' })
      outputRows.push({ ...frozen, local_file: `${frozen.id}.wav`, audio_sha256: audioHash, wav: wavInfo })
      matched += 1
    }
    if (matched !== selected.length) throw new Error(`selected row count mismatch for ${key}: ${matched}`)
    console.log(JSON.stringify({ rowgroup: key, selected: matched }))
  }

  outputRows.sort((a, b) => a.id.localeCompare(b.id))
  if (outputRows.length !== rows.length || new Set(outputRows.map((row) => row.id)).size !== rows.length) {
    throw new Error('materialized cohort is incomplete or contains duplicate IDs')
  }
  const inventory = outputRows.map((row) => JSON.stringify(row)).join('\n') + '\n'
  await writeFile(path.join(outputDir, 'media-inventory.jsonl'), inventory, { flag: 'wx' })
  const report = {
    schema_version: 1,
    status: 'FRESH_TRAIN_SPEAKER_MEDIA_MATERIALIZED',
    source: `${DATASET}@${REVISION}`,
    source_split: 'official train shards only',
    source_license: 'CC-BY-4.0',
    source_license_evidence: `https://huggingface.co/datasets/${DATASET}/blob/${REVISION}/README.md`,
    attribution: 'Pete Warden, Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition (2018), https://arxiv.org/abs/1804.03209; Google Speech Commands v0.02, CC BY 4.0.',
    cohort_manifest_sha256: manifestHash,
    payload_plan_sha256: sha(planText),
    parser: {
      name: pkg.name,
      version: pkg.version,
      license: pkg.license,
      package_json_sha256: EXPECTED_PACKAGE_SHA256,
      entrypoint_sha256: EXPECTED_ENTRY_SHA256,
      npm_tarball_sha256: '1d448ce058c3817b4748ff43002fb644bcdca1382949a18c52f767e93eb3bd14',
    },
    selected_examples_written: outputRows.length,
    unique_speaker_groups: new Set(outputRows.map((row) => row.speaker_group_sha256)).size,
    selected_wav_bytes: outputRows.reduce((sum, row) => sum + row.wav.bytes, 0),
    rowgroups_read: plan.count,
    rows_in_selected_rowgroups: plan.row_count,
    collateral_rows_read_but_not_written_or_scored: plan.collateral_rows,
    compressed_audio_bytes_requested: plan.audio_compressed_bytes,
    transport: stats,
    official_validation_or_test_shards_read: false,
    collateral_audio_persisted: false,
    model_loaded: false,
    inference_run: false,
    training_run: false,
    sealed_audit_opened: false,
  }
  await writeFile(path.join(outputDir, 'materialization-report.json'), `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' })
  console.log(JSON.stringify({ status: report.status, examples: report.selected_examples_written,
    speakers: report.unique_speaker_groups, selected_wav_bytes: report.selected_wav_bytes,
    transport: report.transport }, null, 2))
}

main().catch((error) => {
  console.error(error.stack || error)
  process.exitCode = 1
})
