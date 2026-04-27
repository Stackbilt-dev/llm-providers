/**
 * Schema canary tests (#39 Part 2)
 *
 * Covers extractShape, compareShapes, and runCanaryCheck against
 * golden fixtures for all five providers.
 */

import { describe, it, expect } from 'vitest';
import { extractShape, compareShapes, runCanaryCheck } from '../utils/schema-canary';
import type { ShapeMap } from '../utils/schema-canary';

import anthropicGolden from './fixtures/response-shapes/anthropic.json';
import openaiGolden from './fixtures/response-shapes/openai.json';
import groqGolden from './fixtures/response-shapes/groq.json';
import cerebrasGolden from './fixtures/response-shapes/cerebras.json';
import cloudflareGolden from './fixtures/response-shapes/cloudflare.json';

// ── extractShape ──────────────────────────────────────────────────────────────

describe('extractShape', () => {
  it('returns primitive type at root', () => {
    expect(extractShape('hello', 'x')).toEqual({ x: 'string' });
    expect(extractShape(42, 'n')).toEqual({ n: 'number' });
    expect(extractShape(null, 'v')).toEqual({ v: 'null' });
  });

  it('walks a flat object', () => {
    const shape = extractShape({ a: 1, b: 'x', c: null });
    expect(shape).toMatchObject({ a: 'number', b: 'string', c: 'null' });
  });

  it('descends into nested objects', () => {
    const shape = extractShape({ usage: { input_tokens: 10, output_tokens: 5 } });
    expect(shape['usage']).toBe('object');
    expect(shape['usage.input_tokens']).toBe('number');
    expect(shape['usage.output_tokens']).toBe('number');
  });

  it('represents arrays by their first element shape', () => {
    const shape = extractShape({ choices: [{ index: 0, finish_reason: 'stop' }] });
    expect(shape['choices']).toBe('array');
    expect(shape['choices[0]']).toBe('object');
    expect(shape['choices[0].index']).toBe('number');
    expect(shape['choices[0].finish_reason']).toBe('string');
  });

  it('treats empty arrays without descending', () => {
    const shape = extractShape({ items: [] });
    expect(shape['items']).toBe('array');
    expect(Object.keys(shape).some(k => k.startsWith('items['))).toBe(false);
  });

  it('does not descend past depth 6', () => {
    const deep = { a: { b: { c: { d: { e: { f: { g: 'too deep' } } } } } } };
    const shape = extractShape(deep);
    expect('a.b.c.d.e.f' in shape).toBe(true);
    expect('a.b.c.d.e.f.g' in shape).toBe(false);
  });
});

// ── compareShapes ─────────────────────────────────────────────────────────────

describe('compareShapes', () => {
  it('reports no diff when shapes are identical', () => {
    const golden: ShapeMap = { id: 'string', count: 'number' };
    const diff = compareShapes(golden, golden);
    expect(diff).toEqual({ added: [], removed: [], changed: [] });
  });

  it('reports added paths when live has new fields', () => {
    const golden: ShapeMap = { id: 'string' };
    const live: ShapeMap = { id: 'string', model: 'string' };
    const diff = compareShapes(golden, live);
    expect(diff.added).toContain('model');
    expect(diff.removed).toHaveLength(0);
  });

  it('reports removed paths when live is missing fields', () => {
    const golden: ShapeMap = { id: 'string', usage: 'object' };
    const live: ShapeMap = { id: 'string' };
    const diff = compareShapes(golden, live);
    expect(diff.removed).toContain('usage');
    expect(diff.added).toHaveLength(0);
  });

  it('reports changed paths when a field type differs', () => {
    const golden: ShapeMap = { id: 'string', count: 'number' };
    const live: ShapeMap = { id: 'string', count: 'string' };
    const diff = compareShapes(golden, live);
    expect(diff.changed).toContain('count');
  });
});

// ── runCanaryCheck ────────────────────────────────────────────────────────────

describe('runCanaryCheck', () => {
  it('returns ok when live matches golden', () => {
    const golden: ShapeMap = { id: 'string', value: 'number' };
    const report = runCanaryCheck('test', golden, { id: 'abc', value: 42 });
    expect(report.status).toBe('ok');
    expect(report.diff).toEqual({ added: [], removed: [], changed: [] });
  });

  it('returns drift when a field is removed', () => {
    const golden: ShapeMap = { id: 'string', usage: 'object' };
    const report = runCanaryCheck('test', golden, { id: 'abc' });
    expect(report.status).toBe('drift');
    expect(report.diff.removed).toContain('usage');
  });

  it('includes provider name in report', () => {
    const golden: ShapeMap = { id: 'string' };
    const report = runCanaryCheck('openai', golden, { id: 'x' });
    expect(report.provider).toBe('openai');
  });
});

// ── Golden fixture checks ─────────────────────────────────────────────────────

const canonicalAnthropicResponse = {
  id: 'msg_1',
  type: 'message',
  role: 'assistant',
  content: [{ type: 'text', text: 'hello' }],
  model: 'claude-haiku-4-5-20251001',
  stop_reason: 'end_turn',
  stop_sequence: null,
  usage: { input_tokens: 10, output_tokens: 5 },
};

const canonicalOpenAiCompatResponse = {
  id: 'chatcmpl_1',
  object: 'chat.completion',
  created: 1700000000,
  model: 'gpt-4o-mini',
  choices: [{
    index: 0,
    message: { role: 'assistant', content: 'hello' },
    finish_reason: 'stop',
  }],
  usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
};

describe('Golden fixture parity — Anthropic', () => {
  it('canonical response matches golden fixture exactly', () => {
    const report = runCanaryCheck('anthropic', anthropicGolden as ShapeMap, canonicalAnthropicResponse);
    expect(report.status).toBe('ok');
  });

  it('detects drift when usage.input_tokens is renamed', () => {
    const drifted = { ...canonicalAnthropicResponse, usage: { prompt_tokens: 10, output_tokens: 5 } };
    const report = runCanaryCheck('anthropic', anthropicGolden as ShapeMap, drifted);
    expect(report.status).toBe('drift');
    expect(report.diff.removed).toContain('usage.input_tokens');
    expect(report.diff.added).toContain('usage.prompt_tokens');
  });
});

describe('Golden fixture parity — OpenAI', () => {
  it('canonical response matches golden fixture exactly', () => {
    const report = runCanaryCheck('openai', openaiGolden as ShapeMap, canonicalOpenAiCompatResponse);
    expect(report.status).toBe('ok');
  });

  it('detects drift when choices is removed', () => {
    const { choices: _c, ...drifted } = canonicalOpenAiCompatResponse;
    const report = runCanaryCheck('openai', openaiGolden as ShapeMap, drifted);
    expect(report.status).toBe('drift');
    expect(report.diff.removed).toContain('choices');
  });
});

describe('Golden fixture parity — Groq', () => {
  it('canonical response matches golden fixture exactly', () => {
    const response = { ...canonicalOpenAiCompatResponse, model: 'llama-3.3-70b-versatile' };
    const report = runCanaryCheck('groq', groqGolden as ShapeMap, response);
    expect(report.status).toBe('ok');
  });
});

describe('Golden fixture parity — Cerebras', () => {
  it('canonical response matches golden fixture exactly', () => {
    const response = { ...canonicalOpenAiCompatResponse, model: 'llama-3.1-8b' };
    const report = runCanaryCheck('cerebras', cerebrasGolden as ShapeMap, response);
    expect(report.status).toBe('ok');
  });
});

describe('Golden fixture parity — Cloudflare', () => {
  it('canonical text response matches golden fixture exactly', () => {
    const report = runCanaryCheck('cloudflare', cloudflareGolden as ShapeMap, { response: 'hello' });
    expect(report.status).toBe('ok');
  });

  it('detects drift when response field is renamed', () => {
    const report = runCanaryCheck('cloudflare', cloudflareGolden as ShapeMap, { output_text: 'hello' });
    expect(report.status).toBe('drift');
    expect(report.diff.removed).toContain('response');
    expect(report.diff.added).toContain('output_text');
  });
});
