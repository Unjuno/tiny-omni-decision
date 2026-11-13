/**
 * Extract a deterministic, speaker-disjoint 10-keyword Speech Commands sample.
 *
 * This utility expects the separately pinned MIT hyparquet 1.31.3 package.
 * It reads only selected row groups over HTTP Range and writes media outside Git.
 */

import { createHash } from 'node:crypto'
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const REVISION = 'a751309c0fd613e8a5d30d77900f30e8b42bc2da'
const DATASET = 'google/speech_commands'
const TRAIN_FILES = Array.from({ length: 6 }, (_, index) =>
  `train-${String(index).padStart(5, '0')}-of-00006.parquet`,
)
const VALIDATION_FILE = 'validation-00000-of-00001.parquet'
const CLASS_IDS = Array.from({ length: 10 }, (_, index) => index)
const UTF8 = new TextDecoder()
export function decodeTextOrWav(bytes) {
  if (!bytes) return bytes
  const isWav =
    bytes.length >= 12 &&
    bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46 &&
    bytes[8] === 0x57 && bytes[9] === 0x41 && bytes[10] === 0x56 && bytes[11] === 0x45
  return isWav ? bytes : UTF8.decode(bytes)
}
const PARQUET_PARSERS = {
  stringFromBytes: (bytes) => bytes && UTF8.decode(bytes),
  jsonFromBytes: (bytes) => bytes && JSON.parse(UTF8.decode(bytes)),
  geometryFromBytes: () => { throw new Error('geometry parser is not expected in Speech Commands') },
  geographyFromBytes: () => { throw new Error('geography parser is not expected in Speech Commands') },
  uuidFromBytes: (bytes) => bytes && Buffer.from(bytes).toString('hex'),
  timestampFromMilliseconds: (value) => new Date(Number(value)),
  timestampFromMicroseconds: (value) => new Date(Number(value / 1000n)),
  timestampFromNanoseconds: (value) => new Date(Number(value / 1000000n)),
  dateFromDays: (value) => new Date(value * 86400000),
}
const EXPECTED_SHARD_SHA256 = {
  'train-00000-of-00006.parquet': '3104193b88158e7f507ecedd249283b1bfacd23bc84fde2d424df86bb9ca6279',
  'train-00001-of-00006.parquet': '571445a0fa82c0f6c8c92c4545baf836bbe9f439326c9bd8e6b6a3dc8de6b7ff',
  'train-00002-of-00006.parquet': 'f49d14a179e15fc861c708c9cb66479ab8fe8b96cdbfe06e2d8638b7faf4c1ef',
  'train-00003-of-00006.parquet': 'e1e8d9e798b45053015632233265d0f6522b51f23eae80d75f96205acb545ff7',
  'train-00004-of-00006.parquet': '18e1c61afe006401155334731fb27a338a50a65933e0fef71a3eb2204b15a3a0',
  'train-00005-of-00006.parquet': '281d138d5dd89150b55b0f3ca8ccf1ccd5e63557d4972b64a787df11984e28cd',
  'validation-00000-of-00001.parquet': '06683ef0761319a415ee516939c64b9b7a20be0b6bb17ce8d2b6e36abe373a46',
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex')
}

function argsFromCli(argv) {
  const values = new Map()
  for (let index = 2; index < argv.length; index += 2) {
    const key = argv[index]
    const value = argv[index + 1]
    if (!key?.startsWith('--') || value === undefined) {
      throw new Error(`expected --name value arguments, found ${key}`)
    }
    values.set(key.slice(2), value)
  }
  const outputDir = values.get('output-dir')
  const moduleEntry = values.get('hyparquet-entry')
  const proxyBase = values.get('range-proxy')
  const seed = Number(values.get('seed') ?? 17)
  const trainPerClass = Number(values.get('train-per-class') ?? 128)
  const validationPerClass = Number(values.get('validation-per-class') ?? 64)
  if (!outputDir || !moduleEntry) {
    throw new Error('--output-dir and --hyparquet-entry are required')
  }
  if (!Number.isInteger(seed) || seed < 0 || !Number.isInteger(trainPerClass) || trainPerClass < 1 ||
      !Number.isInteger(validationPerClass) || validationPerClass < 1) {
    throw new Error('seed and per-class sample counts must be positive integers')
  }
  return {
    outputDir: path.resolve(outputDir),
    moduleEntry: path.resolve(moduleEntry),
    proxyBase,
    seed,
    trainPerClass,
    validationPerClass,
  }
}

function makeLimitedFetch(concurrency = 6, proxyBase) {
  let active = 0
  const waiting = []
  const etags = new Set()
  const rangeBytes = { requested: 0, response: 0 }
  async function acquire() {
    if (active < concurrency) {
      active += 1
      return
    }
    await new Promise((resolve) => waiting.push(resolve))
    active += 1
  }
  function release() {
    active -= 1
    waiting.shift()?.()
  }
  const fetch = async (url, init = {}) => {
    await acquire()
    try {
      const headers = new Headers(init.headers)
      const range = headers.get('range')
      let lastError
      for (let attempt = 0; attempt < 8; attempt += 1) {
        try {
          const response = proxyBase
            ? await globalThis.fetch(`${proxyBase}/range`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                url,
                method: init.method ?? 'GET',
                headers: Object.fromEntries(headers),
              }),
              signal: AbortSignal.timeout(120000),
            })
            : await globalThis.fetch(url, {
              ...init,
              headers,
              signal: AbortSignal.timeout(60000),
            })
          if (response.status === 429 || response.status >= 500) {
            const retryAfter = Number(response.headers.get('retry-after'))
            await response.body?.cancel()
            await new Promise((resolve) => setTimeout(resolve,
              Number.isFinite(retryAfter) && retryAfter > 0
                ? Math.min(retryAfter * 1000, 30000)
                : Math.min(1000 * (2 ** attempt), 30000),
            ))
            continue
          }
          const etag = response.headers.get('etag')
          if (etag) etags.add(etag)
          if (range && response.ok) {
            rangeBytes.requested += 1
            const contentRange = response.headers.get('content-range')
            if (contentRange) {
              const match = contentRange.match(/bytes (\d+)-(\d+)\//)
              if (match) rangeBytes.response += Number(match[2]) - Number(match[1]) + 1
            }
          }
          return response
        } catch (error) {
          lastError = error
          if (attempt === 7) throw error
          await new Promise((resolve) => setTimeout(resolve, Math.min(1000 * (2 ** attempt), 30000)))
        }
      }
      throw lastError ?? new Error('HTTP retry budget exhausted')
    } finally {
      release()
    }
  }
  return { fetch, etags, rangeBytes }
}

function readLabelNames(metadata) {
  const entry = metadata.key_value_metadata?.find(({ key }) => key === 'huggingface')
  if (!entry) throw new Error('pinned Parquet is missing Hugging Face feature metadata')
  const info = JSON.parse(entry.value)
  const names = info.info?.features?.label?.names
  if (!Array.isArray(names) || names.length !== 36) {
    throw new Error('unexpected Speech Commands ClassLabel names in Parquet metadata')
  }
  return names
}

function targetRowGroups(metadata) {
  const groups = []
  let rowStart = 0
  for (const rowGroup of metadata.row_groups) {
    const labelColumn = rowGroup.columns.find(({ meta_data }) => meta_data.path_in_schema[0] === 'label')
    const stats = labelColumn?.meta_data.statistics
    const low = Number(stats?.min_value ?? stats?.min)
    const high = Number(stats?.max_value ?? stats?.max)
    if (!Number.isInteger(low) || !Number.isInteger(high)) {
      throw new Error('label row-group statistics are absent; refusing a broad media read')
    }
    if (CLASS_IDS.some((label) => label >= low && label <= high)) {
      const count = Number(rowGroup.num_rows)
      groups.push({ rowStart, rowEnd: rowStart + count, minLabel: low, maxLabel: high })
    }
    rowStart += Number(rowGroup.num_rows)
  }
  return groups
}

async function loadCandidates(fileName, parquet, fetch) {
  const url = `https://huggingface.co/datasets/${DATASET}/resolve/${REVISION}/v0.02/${fileName}?download=true`
  const file = await parquet.asyncBufferFromUrl({ url, fetch })
  const metadata = await parquet.parquetMetadataAsync(file)
  const labels = readLabelNames(metadata)
  const groups = targetRowGroups(metadata)
  const candidates = []
  for (const group of groups) {
    const rows = await parquet.parquetReadObjects({
      file,
      metadata,
      rowStart: group.rowStart,
      rowEnd: group.rowEnd,
      columns: ['label', 'speaker_id', 'file'],
      parsers: PARQUET_PARSERS,
      includeRowIndex: true,
    })
    for (let localIndex = 0; localIndex < rows.length; localIndex += 1) {
      const row = rows[localIndex]
      const label = Number(row.label)
      if (CLASS_IDS.includes(label)) {
        candidates.push({
          split: fileName.startsWith('train-') ? 'train' : 'validation',
          fileName,
          sourceRow: row[parquet.rowIndex],
          sourceGroupStart: group.rowStart,
          sourceGroupEnd: group.rowEnd,
          label,
          target: labels[label],
          speakerId: String(row.speaker_id),
          sourcePathSha256: sha256(String(row.file)),
        })
      }
    }
  }
  return { candidates, metadata, labels, file }
}

function deterministicOrder(seed, split, label, rows) {
  return [...rows].sort((left, right) => {
    const a = sha256(`${seed}|${split}|${label}|${left.fileName}|${left.sourceRow}`)
    const b = sha256(`${seed}|${split}|${label}|${right.fileName}|${right.sourceRow}`)
    return a.localeCompare(b)
  })
}

function chooseRows(rows, split, countPerClass, seed) {
  const selected = []
  for (const label of CLASS_IDS) {
    const candidates = deterministicOrder(seed, split, label, rows.filter((row) => row.label === label))
    const seenSpeakers = new Set()
    const chosen = []
    for (const candidate of candidates) {
      if (seenSpeakers.has(candidate.speakerId)) continue
      seenSpeakers.add(candidate.speakerId)
      chosen.push(candidate)
      if (chosen.length === countPerClass) break
    }
    if (chosen.length !== countPerClass) {
      throw new Error(`${split} class ${label} has only ${chosen.length}/${countPerClass} unique speakers`)
    }
    selected.push(...chosen)
  }
  return selected.sort((a, b) => a.label - b.label || a.fileName.localeCompare(b.fileName) || a.sourceRow - b.sourceRow)
}

async function writeSplit(split, rows, parquetByName, outputDir, parquet) {
  const mediaDir = path.join(outputDir, 'media', split)
  await mkdir(mediaDir, { recursive: true })
  const byGroup = new Map()
  for (const row of rows) {
    const key = `${row.fileName}|${row.sourceGroupStart}|${row.sourceGroupEnd}`
    if (!byGroup.has(key)) byGroup.set(key, [])
    byGroup.get(key).push(row)
  }
  const records = []
  const groups = [...byGroup.values()]
  for (const [groupIndex, selected] of groups.entries()) {
    if (groupIndex % 10 === 0 || groupIndex === groups.length - 1) {
      console.log(`extract ${split} audio row groups ${groupIndex + 1}/${groups.length}`)
    }
    const source = parquetByName.get(selected[0].fileName)
    const rowsInGroup = await source.parquet.parquetReadObjects({
      file: source.file,
      metadata: source.metadata,
      rowStart: selected[0].sourceGroupStart,
      rowEnd: selected[0].sourceGroupEnd,
      columns: ['label', 'speaker_id', 'file', 'audio'],
      includeRowIndex: true,
      parsers: {
        ...PARQUET_PARSERS,
        stringFromBytes: decodeTextOrWav,
      },
    })
    for (const item of selected) {
      const row = rowsInGroup.find((candidate) => candidate[parquet.rowIndex] === item.sourceRow)
      if (!row) throw new Error(`selected physical row is absent: ${item.fileName}:${item.sourceRow}`)
      if (Number(row.label) !== item.label || String(row.speaker_id) !== item.speakerId) {
        throw new Error(`selected metadata changed while loading audio: ${item.fileName}:${item.sourceRow}`)
      }
      const bytes = Buffer.from(row.audio?.bytes ?? [])
      if (bytes.length < 44 || bytes.toString('ascii', 0, 4) !== 'RIFF' || bytes.toString('ascii', 8, 12) !== 'WAVE') {
        throw new Error(`selected media is not a complete WAV: ${item.fileName}:${item.sourceRow}`)
      }
      const sampleId = `speechcmd-${split}-${sha256(`${REVISION}|${item.fileName}|${item.sourceRow}`).slice(0, 20)}`
      const relativePath = path.join('media', split, `${sampleId}.wav`).replaceAll('\\', '/')
      await writeFile(path.join(outputDir, relativePath), bytes, { flag: 'wx' })
      records.push({
        id: sampleId,
        split,
        source: `${DATASET}@${REVISION}`,
        source_file: item.fileName,
        source_row: item.sourceRow,
        source_path_sha256: item.sourcePathSha256,
        speaker_group_sha256: sha256(item.speakerId),
        media_path: relativePath,
        media_sha256: sha256(bytes),
        media_bytes: bytes.length,
        sample_rate: 16000,
        label: item.label,
        target: item.target,
        question: 'Which spoken English keyword was uttered?',
        options: ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'],
      })
    }
  }
  records.sort((a, b) => a.label - b.label || a.source_file.localeCompare(b.source_file) || a.source_row - b.source_row)
  await writeFile(
    path.join(outputDir, `${split}.jsonl`),
    records.map((row) => JSON.stringify(row)).join('\n') + '\n',
    { flag: 'wx' },
  )
  return records
}

async function main() {
  const args = argsFromCli(process.argv)
  const packageJson = JSON.parse(await readFile(path.join(path.dirname(args.moduleEntry), '..', 'package.json'), 'utf8').catch(async () =>
    readFile(path.join(path.dirname(args.moduleEntry), 'package.json'), 'utf8')))
  if (packageJson.name !== 'hyparquet' || packageJson.version !== '1.31.3' || packageJson.license !== 'MIT') {
    throw new Error('hyparquet must be the externally pinned MIT package at version 1.31.3')
  }
  const parquet = await import(pathToFileURL(args.moduleEntry).href)
  const outputDir = args.outputDir
  await mkdir(outputDir, { recursive: true })
  if ((await readdir(outputDir)).length) {
    throw new Error(`refusing non-empty output directory: ${outputDir}`)
  }
  const network = makeLimitedFetch(4, args.proxyBase)
  const fileNames = [...TRAIN_FILES, VALIDATION_FILE]
  const sourceResults = new Map()
  const trainCandidates = []
  let validationCandidates = []
  for (const fileName of fileNames) {
    const result = await loadCandidates(fileName, parquet, network.fetch)
    console.log(`scanned ${fileName}: ${result.candidates.length} target-keyword rows`)
    const expectedSha = EXPECTED_SHARD_SHA256[fileName]
    const split = fileName.startsWith('train-') ? 'train' : 'validation'
    sourceResults.set(fileName, {
      split,
      pinned_revision: REVISION,
      expected_full_file_sha256: expectedSha,
      full_file_sha256_verified: false,
      file_bytes: result.file.byteLength,
      rows: Number(result.metadata.num_rows),
      row_groups: result.metadata.row_groups.length,
      selected_class_counts_in_candidate_scan: Object.fromEntries(CLASS_IDS.map((label) => [
        result.labels[label], result.candidates.filter((row) => row.label === label).length,
      ])),
      scanned_row_groups: targetRowGroups(result.metadata).length,
    })
    sourceResults.get(fileName).candidates = result.candidates
    sourceResults.get(fileName).parquet = parquet
    sourceResults.get(fileName).metadata = result.metadata
    sourceResults.get(fileName).file = result.file
    if (split === 'train') trainCandidates.push(...result.candidates)
    else validationCandidates = result.candidates
  }
  const train = chooseRows(trainCandidates, 'train', args.trainPerClass, args.seed)
  const validation = chooseRows(validationCandidates, 'validation', args.validationPerClass, args.seed)
  const trainSpeakers = new Set(train.map((row) => sha256(row.speakerId)))
  const validationSpeakers = new Set(validation.map((row) => sha256(row.speakerId)))
  const speakerOverlap = [...trainSpeakers].filter((id) => validationSpeakers.has(id))
  if (speakerOverlap.length) throw new Error(`train/validation speakers overlap: ${speakerOverlap.length}`)
  const trainRecords = await writeSplit('train', train, sourceResults, outputDir, parquet)
  const validationRecords = await writeSplit('validation', validation, sourceResults, outputDir, parquet)
  const trainAssets = new Set(trainRecords.map((row) => row.media_sha256))
  const validationAssets = new Set(validationRecords.map((row) => row.media_sha256))
  if ([...trainAssets].some((hash) => validationAssets.has(hash))) {
    throw new Error('train/validation media bytes overlap')
  }
  const orderHash = (records) => sha256(records.map((row) => row.id).join('\n'))
  const report = {
    schema_version: 1,
    status: 'complete_fixed_sample_not_sealed_audit',
    dataset: DATASET,
    revision: REVISION,
    subset: 'v0.02',
    license: 'CC-BY-4.0 per repository pinned candidate manifest',
    label_names: ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'],
    omitted_labels: 'all labels outside ids 0-9, including unknown and silence',
    sampler: { seed: args.seed, train_per_class: args.trainPerClass, validation_per_class: args.validationPerClass, unique_speakers_within_each_class: true },
    train_count: trainRecords.length,
    validation_count: validationRecords.length,
    train_class_counts: Object.fromEntries(CLASS_IDS.map((label) => [
      trainRecords.find((row) => row.label === label)?.target,
      trainRecords.filter((row) => row.label === label).length,
    ])),
    validation_class_counts: Object.fromEntries(CLASS_IDS.map((label) => [
      validationRecords.find((row) => row.label === label)?.target,
      validationRecords.filter((row) => row.label === label).length,
    ])),
    train_unique_speakers: trainSpeakers.size,
    validation_unique_speakers: validationSpeakers.size,
    train_validation_speaker_overlap: 0,
    train_validation_asset_overlap: 0,
    train_id_order_sha256: orderHash(trainRecords),
    validation_id_order_sha256: orderHash(validationRecords),
    train_jsonl_sha256: sha256(await readFile(path.join(outputDir, 'train.jsonl'))),
    validation_jsonl_sha256: sha256(await readFile(path.join(outputDir, 'validation.jsonl'))),
    source_files: Object.fromEntries([...sourceResults].map(([name, value]) => [name, Object.fromEntries(Object.entries(value).filter(([key]) => !['candidates', 'parquet', 'metadata', 'file'].includes(key)))])),
    transport: { range_requests: network.rangeBytes.requested, content_range_bytes: network.rangeBytes.response, distinct_etags: [...network.etags] },
    sealed_audit_loaded: false,
    note: 'The source shard full-file SHA-256 values are pinned but not verified because only metadata and selected row-group byte ranges were read. Every extracted WAV is SHA-256 hashed. Pseudonymous speaker IDs are only hashed for split checks; no speaker identity inference is performed.',
  }
  await writeFile(path.join(outputDir, 'fetch-report.json'), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' })
  console.log(JSON.stringify(report, null, 2))
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  main().catch((error) => {
    console.error(error.stack ?? error)
    process.exitCode = 1
  })
}
