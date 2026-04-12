/**
 * Schema Drift Detection Tests
 *
 * Covers the response-envelope schema validator, Anthropic provider drift
 * detection at the parser boundary, factory fallback routing on drift,
 * and onSchemaDrift observability hook firing.
 *
 * Slice 1 of #39 — Anthropic is the template provider. Follow-up tests for
 * openai/groq/cerebras/cloudflare land with the respective wire-ups.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { validateSchema, type SchemaField } from '../utils/schema-validator';
import { SchemaDriftError } from '../errors';
import { AnthropicProvider } from '../providers/anthropic';
import { LLMProviderFactory } from '../factory';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { defaultExhaustionRegistry } from '../utils/exhaustion';
import { defaultCostTracker } from '../utils/cost-tracker';
import { defaultLatencyHistogram } from '../utils/latency-histogram';
import type { ObservabilityHooks, SchemaDriftEvent } from '../utils/hooks';

const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

// ── validateSchema primitives ────────────────────────────────────────────

describe('validateSchema', () => {
  it('accepts a valid envelope with all required fields', () => {
    const schema: SchemaField[] = [
      { path: 'id', type: 'string' },
      { path: 'usage.input_tokens', type: 'number' },
    ];
    expect(() => validateSchema('test', { id: 'x', usage: { input_tokens: 5 } }, schema))
      .not.toThrow();
  });

  it('throws SchemaDriftError when a required path is missing', () => {
    const schema: SchemaField[] = [{ path: 'usage.input_tokens', type: 'number' }];
    try {
      validateSchema('test', { usage: {} }, schema);
      expect.fail('should have thrown');
    } catch (err) {
      expect(err).toBeInstanceOf(SchemaDriftError);
      expect((err as SchemaDriftError).path).toBe('usage.input_tokens');
      expect((err as SchemaDriftError).expected).toBe('number');
      expect((err as SchemaDriftError).actual).toBe('undefined');
    }
  });

  it('throws with correct actual type when field has wrong type', () => {
    const schema: SchemaField[] = [{ path: 'content', type: 'array' }];
    try {
      validateSchema('test', { content: 'oops-its-a-string-now' }, schema);
      expect.fail('should have thrown');
    } catch (err) {
      expect(err).toBeInstanceOf(SchemaDriftError);
      expect((err as SchemaDriftError).actual).toBe('string');
    }
  });

  it('distinguishes null from undefined from object', () => {
    const schema: SchemaField[] = [{ path: 'usage', type: 'object' }];

    try {
      validateSchema('test', { usage: null }, schema);
      expect.fail('should have thrown');
    } catch (err) {
      expect((err as SchemaDriftError).actual).toBe('null');
    }

    try {
      validateSchema('test', {}, schema);
      expect.fail('should have thrown');
    } catch (err) {
      expect((err as SchemaDriftError).actual).toBe('undefined');
    }
  });

  it('allows optional fields to be missing', () => {
    const schema: SchemaField[] = [
      { path: 'id', type: 'string' },
      { path: 'stop_sequence', type: 'string', optional: true },
    ];
    expect(() => validateSchema('test', { id: 'x' }, schema)).not.toThrow();
  });

  it('rejects NaN for number fields (Number.isFinite guard)', () => {
    const schema: SchemaField[] = [{ path: 'tokens', type: 'number' }];
    expect(() => validateSchema('test', { tokens: NaN }, schema))
      .toThrow(SchemaDriftError);
  });

  it('rejects arrays when type is "object"', () => {
    const schema: SchemaField[] = [{ path: 'usage', type: 'object' }];
    try {
      validateSchema('test', { usage: [] }, schema);
      expect.fail('should have thrown');
    } catch (err) {
      expect((err as SchemaDriftError).actual).toBe('array');
    }
  });

  it('throws when root is not an object', () => {
    expect(() => validateSchema('test', null, []))
      .toThrow(SchemaDriftError);
    expect(() => validateSchema('test', 'a string', []))
      .toThrow(SchemaDriftError);
  });

  it('supports string-or-null type for nullable content fields', () => {
    const schema: SchemaField[] = [{ path: 'content', type: 'string-or-null' }];
    expect(() => validateSchema('test', { content: null }, schema)).not.toThrow();
    expect(() => validateSchema('test', { content: 'hi' }, schema)).not.toThrow();
    expect(() => validateSchema('test', { content: 42 }, schema))
      .toThrow(SchemaDriftError);
  });

  it('fails fast on first drift (does not keep walking)', () => {
    const schema: SchemaField[] = [
      { path: 'first', type: 'string' },
      { path: 'second', type: 'string' },
    ];
    try {
      validateSchema('test', { first: 42, second: 99 }, schema);
      expect.fail('should have thrown');
    } catch (err) {
      // Should report the first failure, not the second
      expect((err as SchemaDriftError).path).toBe('first');
    }
  });
});

// ── Anthropic provider drift detection ───────────────────────────────────

describe('AnthropicProvider response schema validation', () => {
  let provider: AnthropicProvider;

  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
    provider = new AnthropicProvider({ apiKey: 'test-key', maxRetries: 0 });
  });

  const validAnthropicResponse = {
    id: 'msg_1',
    type: 'message',
    role: 'assistant',
    content: [{ type: 'text', text: 'hello' }],
    model: 'claude-3-haiku-20240307',
    stop_reason: 'end_turn',
    usage: { input_tokens: 10, output_tokens: 5 },
  };

  it('passes through a well-formed Anthropic response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => validAnthropicResponse,
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    const res = await provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model: 'claude-3-haiku-20240307',
    });

    expect(res.content).toBe('hello');
    expect(res.usage.inputTokens).toBe(10);
  });

  it('throws SchemaDriftError when usage.input_tokens is renamed', async () => {
    // Simulated drift: Anthropic silently renames input_tokens → prompt_tokens
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        ...validAnthropicResponse,
        usage: { prompt_tokens: 10, output_tokens: 5 },
      }),
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    await expect(provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model: 'claude-3-haiku-20240307',
    })).rejects.toMatchObject({
      name: 'LLMProviderError',
      code: 'SCHEMA_DRIFT',
      provider: 'anthropic',
      path: 'usage.input_tokens',
    });
  });

  it('throws SchemaDriftError when content field is removed', async () => {
    const { content: _content, ...rest } = validAnthropicResponse;
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => rest,
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    await expect(provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model: 'claude-3-haiku-20240307',
    })).rejects.toMatchObject({ code: 'SCHEMA_DRIFT', path: 'content' });
  });

  it('throws SchemaDriftError when content type changes from array to string', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ ...validAnthropicResponse, content: 'flat text instead of blocks' }),
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    await expect(provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model: 'claude-3-haiku-20240307',
    })).rejects.toMatchObject({
      code: 'SCHEMA_DRIFT',
      path: 'content',
      expected: 'array',
      actual: 'string',
    });
  });

  it('is non-retryable so retries do not burn budget on drift', async () => {
    // maxRetries on the provider is 0, but even with retries enabled the
    // retry manager should skip non-retryable errors. Assert via a single
    // failing response: if SchemaDriftError were retryable, fetch would be
    // called more than once.
    const retryProvider = new AnthropicProvider({ apiKey: 'test-key', maxRetries: 3 });
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ ...validAnthropicResponse, usage: {} }),
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    await expect(retryProvider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model: 'claude-3-haiku-20240307',
    })).rejects.toMatchObject({ code: 'SCHEMA_DRIFT' });

    expect(mockFetch).toHaveBeenCalledTimes(1);
  });
});

// ── Factory fallback on SchemaDriftError ─────────────────────────────────

describe('LLMProviderFactory schema drift fallback', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
    defaultExhaustionRegistry.reset();
    defaultCostTracker.reset();
    defaultLatencyHistogram.reset();
  });

  const validOpenAIResponse = {
    id: 'chatcmpl-1',
    model: 'gpt-4o',
    choices: [{
      index: 0,
      message: { role: 'assistant', content: 'from openai' },
      finish_reason: 'stop',
    }],
    usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  };

  it('falls over to another provider when the primary drifts', async () => {
    // First call = anthropic with missing content field (drift)
    // Second call = openai with valid shape (success)
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'msg_1',
          type: 'message',
          role: 'assistant',
          // content intentionally missing — drift
          model: 'claude-3-haiku-20240307',
          stop_reason: 'end_turn',
          usage: { input_tokens: 10, output_tokens: 5 },
        }),
        headers: new Headers({ 'content-type': 'application/json' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => validOpenAIResponse,
        headers: new Headers({ 'content-type': 'application/json' }),
      });

    const factory = new LLMProviderFactory({
      anthropic: { apiKey: 'test-anthropic', maxRetries: 0 },
      openai: { apiKey: 'test-openai', maxRetries: 0 },
      preferredProvider: 'anthropic',
    });

    const response = await factory.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
    });

    expect(response.provider).toBe('openai');
    expect(response.content).toBe('from openai');
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it('fires onSchemaDrift hook with path, expected, actual', async () => {
    const driftEvents: SchemaDriftEvent[] = [];
    const hooks: ObservabilityHooks = {
      onSchemaDrift: (e) => driftEvents.push(e),
    };

    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'msg_1',
          type: 'message',
          role: 'assistant',
          content: [],
          model: 'claude-3-haiku-20240307',
          stop_reason: 'end_turn',
          usage: { prompt_tokens: 10 }, // wrong field name — drift on usage.input_tokens
        }),
        headers: new Headers({ 'content-type': 'application/json' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => validOpenAIResponse,
        headers: new Headers({ 'content-type': 'application/json' }),
      });

    const factory = new LLMProviderFactory({
      anthropic: { apiKey: 'test-anthropic', maxRetries: 0 },
      openai: { apiKey: 'test-openai', maxRetries: 0 },
      preferredProvider: 'anthropic',
      hooks,
    });

    await factory.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      requestId: 'req-drift-1',
    });

    expect(driftEvents).toHaveLength(1);
    expect(driftEvents[0]).toMatchObject({
      provider: 'anthropic',
      requestId: 'req-drift-1',
      path: 'usage.input_tokens',
      expected: 'number',
      actual: 'undefined',
    });
    expect(driftEvents[0].timestamp).toBeGreaterThan(0);
  });
});
