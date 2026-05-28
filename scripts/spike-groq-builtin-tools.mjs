#!/usr/bin/env node
/**
 * S0 spike — issue #69: Groq built-in tools wire-shape capture.
 *
 * Foundational step: the LLMResponse.metadata.builtInToolResults type, the
 * Groq response parser, and GROQ_RESPONSE_SCHEMA are all locked to what Groq
 * ACTUALLY returns. This script makes two live calls against the raw Groq
 * endpoint (bypassing our catalog router on purpose — we want Groq's bytes,
 * not our adapter's behaviour) and records both shapes:
 *
 *   1. groq/compound        — built-in tools via compound_custom.tools.enabled_tools,
 *                             citations expected at message.executed_tools[].search_results
 *   2. openai/gpt-oss-120b  — OpenAI-compatible tools: [{ type: 'browser_search' }]
 *
 * Outputs (per call):
 *   - scripts/spike-output/<name>.raw.json      raw response (or error body) for design reference
 *   - src/__tests__/fixtures/response-shapes/<name>.json   shape map (extractShape convention)
 *
 * Run:  GROQ_API_KEY=... node scripts/spike-groq-builtin-tools.mjs
 *
 * The shape maps are committed; the *.raw.json files are for inspection
 * (gitignored). The schema-canary test re-derives shapes with the real
 * extractShape(), so any divergence from the inlined copy below surfaces there.
 */

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const RAW_DIR = join(__dirname, 'spike-output');
const SHAPES_DIR = join(ROOT, 'src/__tests__/fixtures/response-shapes');

const API_KEY = process.env.GROQ_API_KEY;
const BASE_URL = process.env.GROQ_BASE_URL || 'https://api.groq.com/openai/v1';

if (!API_KEY) {
  console.error('GROQ_API_KEY is required. Run: GROQ_API_KEY=... node scripts/spike-groq-builtin-tools.mjs');
  process.exit(1);
}

/**
 * Faithful copy of src/utils/schema-canary.ts:extractShape (depth cap 6,
 * arrays represented by first element, null -> 'null'). Inlined so this spike
 * is runnable without a TS loader; the canary test validates against the real one.
 */
function extractShape(obj, prefix = '', depth = 0) {
  const result = {};
  if (depth > 6) return result;
  if (obj === null) { result[prefix || 'value'] = 'null'; return result; }
  if (Array.isArray(obj)) {
    result[prefix || 'value'] = 'array';
    if (obj.length > 0) Object.assign(result, extractShape(obj[0], `${prefix}[0]`, depth + 1));
    return result;
  }
  if (typeof obj === 'object') {
    if (prefix) result[prefix] = 'object';
    for (const [key, value] of Object.entries(obj)) {
      const path = prefix ? `${prefix}.${key}` : key;
      Object.assign(result, extractShape(value, path, depth + 1));
    }
    return result;
  }
  result[prefix || 'value'] = typeof obj;
  return result;
}

// A query that should force a real web search rather than parametric recall.
const RESEARCH_MESSAGES = [
  { role: 'system', content: 'You are a research scout. Find authoritative, current web sources and cite them.' },
  { role: 'user', content: 'Find 3 authoritative sources published in 2026 about EEAT (Experience, Expertise, Authoritativeness, Trust) signals in search ranking. List each source URL.' },
];

const CALLS = [
  {
    name: 'groq-builtin-compound',
    body: {
      model: 'groq/compound',
      messages: RESEARCH_MESSAGES,
      // Compound: built-in tools configured here, NOT via top-level `tools`.
      compound_custom: { tools: { enabled_tools: ['web_search'] } },
    },
  },
  {
    name: 'groq-builtin-gpt-oss',
    body: {
      model: 'openai/gpt-oss-120b',
      messages: RESEARCH_MESSAGES,
      // GPT-OSS: OpenAI-compatible built-in tool descriptor.
      tools: [{ type: 'browser_search' }],
    },
  },
];

function summarize(name, data) {
  const msg = data?.choices?.[0]?.message ?? {};
  const executed = msg.executed_tools ?? msg.tool_calls ?? null;
  console.log(`\n── ${name} ──────────────────────────────────────────`);
  console.log(`model:            ${data?.model ?? '(none)'}`);
  console.log(`finish_reason:    ${data?.choices?.[0]?.finish_reason ?? '(none)'}`);
  console.log(`has executed_tools: ${Array.isArray(msg.executed_tools)} (len ${msg.executed_tools?.length ?? 0})`);
  console.log(`has tool_calls:     ${Array.isArray(msg.tool_calls)} (len ${msg.tool_calls?.length ?? 0})`);
  const sr = msg.executed_tools?.[0]?.search_results;
  if (sr) {
    console.log(`search_results[0] keys: ${JSON.stringify(Object.keys(sr?.results?.[0] ?? sr?.[0] ?? sr ?? {}))}`);
  }
  console.log(`usage keys:       ${JSON.stringify(Object.keys(data?.usage ?? {}))}`);
  if (executed) {
    // Print the executed-tools subtree shape — the part we must parse.
    console.log('executed-tools shape:');
    console.log(JSON.stringify(extractShape(executed, 'executed_tools'), null, 2));
  }
}

async function run() {
  mkdirSync(RAW_DIR, { recursive: true });
  mkdirSync(SHAPES_DIR, { recursive: true });

  for (const call of CALLS) {
    console.log(`\n>>> POST ${BASE_URL}/chat/completions  [${call.body.model}]`);
    let res, text;
    try {
      res = await fetch(`${BASE_URL}/chat/completions`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${API_KEY}`, 'Content-Type': 'application/json' },
        body: JSON.stringify(call.body),
      });
      text = await res.text();
    } catch (err) {
      console.error(`  network error: ${err.message}`);
      continue;
    }

    const rawPath = join(RAW_DIR, `${call.name}.raw.json`);
    writeFileSync(rawPath, text);
    console.log(`  HTTP ${res.status}  -> raw saved: ${rawPath}`);

    if (!res.ok) {
      console.error(`  non-OK response (body saved for the design record):\n  ${text.slice(0, 500)}`);
      continue; // capture the error shape too — informs the gating story
    }

    let data;
    try { data = JSON.parse(text); } catch {
      console.error('  response was not JSON; skipping shape extraction');
      continue;
    }

    const shape = extractShape(data);
    const shapePath = join(SHAPES_DIR, `${call.name}.json`);
    writeFileSync(shapePath, JSON.stringify(shape, null, 2) + '\n');
    console.log(`  shape map saved: ${shapePath}`);

    summarize(call.name, data);
  }

  console.log('\nDone. Review the *.raw.json files, confirm the executed_tools[].search_results shape,');
  console.log('then lock metadata.builtInToolResults from those bytes (S2).');
}

run();
