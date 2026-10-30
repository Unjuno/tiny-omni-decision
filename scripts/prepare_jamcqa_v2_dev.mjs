/** Extract only the pinned JamC-QA-V2 dev split into a stable external JSONL sample. */

import { createHash } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const DATASET_REVISION = 'cfb4b64d289acaf592237179f90dd23d3f2d5bdf'
const EXPECTED_PARQUET_SHA256 = 'acd98fd9a1cf3fd5c59bb321b4480141fae6d124b70779599280171741ed3198'
const EXPECTED_ETAG = '"41114f12a58217bfa577bd2760da162d623a70741ec247269c6296ba02df8113"'
const EXPECTED_ROWS = 52

function sha256(value) {
  return createHash('sha256').update(value).digest('hex')
}

function parseArgs(argv) {
  const values = new Map()
  for (let index = 2; index < argv.length; index += 2) {
    const key = argv[index]
    const value = argv[index + 1]
    if (!key?.startsWith('--') || value === undefined) {
      throw new Error(`expected --name value arguments, found ${key}`)
    }
    values.set(key.slice(2), value)
  }
  const parquet = values.get('parquet')
  const hyparquetEntry = values.get('hyparquet-entry')
  const outputDir = values.get('output-dir')
  if (!parquet || !hyparquetEntry || !outputDir) {
    throw new Error('--parquet, --hyparquet-entry, and --output-dir are required')
  }
  return {
    parquet: path.resolve(parquet),
    hyparquetEntry: path.resolve(hyparquetEntry),
    outputDir: path.resolve(outputDir),
  }
}

export async function normalizeRows(rows) {
  if (rows.length !== EXPECTED_ROWS) {
    throw new Error(`expected ${EXPECTED_ROWS} dev rows, got ${rows.length}`)
  }
  const ids = new Set()
  const categoryCounts = {}
  const normalized = rows.map((row) => {
    const id = String(row.qid)
    if (!id.startsWith('jamcqa-dev-') || ids.has(id)) {
      throw new Error(`invalid or duplicate dev question id: ${id}`)
    }
    ids.add(id)
    const options = [row.choice0, row.choice1, row.choice2, row.choice3].map(String)
    const targetIndex = Number(row.answer_index)
    if (options.some((option) => !option.trim()) || !Number.isInteger(targetIndex) || targetIndex < 0 || targetIndex > 3) {
      throw new Error(`invalid options or target index: ${id}`)
    }
    const category = String(row.category)
    categoryCounts[category] = (categoryCounts[category] ?? 0) + 1
    return {
      id,
      category,
      sub_category: row.sub_category == null ? '' : String(row.sub_category),
      question: String(row.question),
      options,
      target_index: targetIndex,
      target: options[targetIndex],
      difficulty: row.difficulty == null ? null : Number(row.difficulty),
      split: 'dev',
    }
  })
  const expectedCategories = {
    culture: 4,
    custom: 4,
    regional_identity: 4,
    geography: 4,
    history: 4,
    government: 4,
    law: 4,
    healthcare: 4,
    jsdf: 20,
  }
  if (JSON.stringify(categoryCounts) !== JSON.stringify(expectedCategories)) {
    throw new Error(`unexpected category mix: ${JSON.stringify(categoryCounts)}`)
  }
  return { normalized, categoryCounts }
}

async function main() {
  const args = parseArgs(process.argv)
  const parquetBytes = await readFile(args.parquet)
  const parquetSha256 = sha256(parquetBytes)
  if (parquetSha256 !== EXPECTED_PARQUET_SHA256) {
    throw new Error(`JamC-QA-V2 dev Parquet SHA-256 mismatch: ${parquetSha256}`)
  }
  const packageJson = JSON.parse(
    await readFile(path.join(path.dirname(args.hyparquetEntry), '..', 'package.json'), 'utf8'),
  )
  if (packageJson.name !== 'hyparquet' || packageJson.version !== '1.31.3' || packageJson.license !== 'MIT') {
    throw new Error('expected the existing pinned MIT hyparquet 1.31.3 package')
  }
  const { asyncBufferFromFile, parquetReadObjects } = await import(pathToFileURL(args.hyparquetEntry))
  const rows = await parquetReadObjects({ file: await asyncBufferFromFile(args.parquet) })
  const { normalized, categoryCounts } = await normalizeRows(rows)
  await mkdir(args.outputDir, { recursive: false })
  const jsonlPath = path.join(args.outputDir, 'dev.jsonl')
  const jsonl = `${normalized.map((row) => JSON.stringify(row)).join('\n')}\n`
  await writeFile(jsonlPath, jsonl, { flag: 'wx' })
  const report = {
    schema_version: 1,
    dataset_id: 'sbintuitions/JamC-QA-V2',
    revision: DATASET_REVISION,
    split: 'dev',
    license: 'CC-BY-SA-4.0',
    attribution: 'Oka, Shibata, and Yoshida; JamC-QA-V2 (2026)',
    upstream_url: `https://huggingface.co/datasets/sbintuitions/JamC-QA-V2/tree/${DATASET_REVISION}`,
    parquet_path: args.parquet,
    parquet_bytes: parquetBytes.length,
    parquet_sha256: parquetSha256,
    etag: EXPECTED_ETAG,
    row_count: normalized.length,
    category_counts: categoryCounts,
    duplicate_option_rows: normalized.filter((row) => new Set(row.options).size < row.options.length).map((row) => row.id),
    question_id_order_sha256: sha256(normalized.map((row) => row.id).join('\n')),
    normalized_jsonl_bytes: Buffer.byteLength(jsonl),
    normalized_jsonl_sha256: sha256(jsonl),
    hyparquet_version: packageJson.version,
    test_split_loaded: false,
    usage: 'evaluation only; no training or checkpoint selection',
  }
  await writeFile(path.join(args.outputDir, 'fetch-report.json'), `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' })
  console.log(JSON.stringify(report, null, 2))
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error)
    process.exitCode = 1
  })
}
