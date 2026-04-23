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
import { OpenAIProvider } from '../providers/openai';
import { GroqProvider } from '../providers/groq';
import { CerebrasProvider } from '../providers/cerebras';
import type { BaseProvider } from '../providers/base';
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
    model: 'claude-haiku-4-5-20251001',
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
      model: 'claude-haiku-4-5-20251001',
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
      model: 'claude-haiku-4-5-20251001',
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
      model: 'claude-haiku-4-5-20251001',
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
      model: 'claude-haiku-4-5-20251001',
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
      model: 'claude-haiku-4-5-20251001',
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
    model: 'gpt-4o-mini',
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
          model: 'claude-haiku-4-5-20251001',
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
      defaultProvider: 'anthropic',
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
          model: 'claude-haiku-4-5-20251001',
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
      defaultProvider: 'anthropic',
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

  it('propagates SchemaDriftError at end of chain when no fallback is available (M-2)', async () => {
    // Single-provider chain: if anthropic drifts, there is nowhere to fall
    // over to. The factory must surface the SchemaDriftError to the caller
    // without wrapping it as generic ALL_PROVIDERS_FAILED, and the hook must
    // still fire so ops sees the drift.
    //
    // (Multi-provider end-of-chain coverage lands with the follow-up PR
    // that wires schema validation into the other providers — until then,
    // the other providers' parsers throw TypeError on drift, which isn't
    // this test's concern.)
    const driftEvents: SchemaDriftEvent[] = [];
    const hooks: ObservabilityHooks = {
      onSchemaDrift: (e) => driftEvents.push(e),
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: 'msg_1', type: 'message', role: 'assistant',
        model: 'claude-haiku-4-5-20251001',
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 5 },
        // content intentionally missing — drift
      }),
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    const factory = new LLMProviderFactory({
      anthropic: { apiKey: 'test-anthropic', maxRetries: 0 },
      defaultProvider: 'anthropic',
      hooks,
    });

    await expect(factory.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      requestId: 'req-eoc-1',
    })).rejects.toMatchObject({
      code: 'SCHEMA_DRIFT',
      provider: 'anthropic',
      path: 'content',
    });

    expect(driftEvents).toHaveLength(1);
    expect(driftEvents[0]).toMatchObject({
      provider: 'anthropic',
      requestId: 'req-eoc-1',
    });
  });
});

// ── Circuit breaker + drift interaction ──────────────────────────────────

describe('SchemaDriftError and circuit breaker', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
  });

  it('repeated drift trips the circuit breaker open (M-1)', async () => {
    // The default breaker uses a graduated degradation curve with a
    // probabilistic rejection gate at DEGRADED levels. To get deterministic
    // progression all the way to OPEN, stub Math.random so the gate always
    // admits the request through to fn() - where the drift throws and the
    // breaker advances one failure. Without this stub, the test is racy:
    // locally it would pass by luck, CI would fail when the random sequence
    // bounced attempts off the degradation gate without advancing.
    const randomSpy = vi.spyOn(Math, 'random').mockReturnValue(0);

    try {
      const provider = new AnthropicProvider({ apiKey: 'test-key', maxRetries: 0 });

      // Return drifted response every call - missing usage.input_tokens
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          id: 'msg_1', type: 'message', role: 'assistant',
          content: [],
          model: 'claude-haiku-4-5-20251001',
          stop_reason: 'end_turn',
          usage: { output_tokens: 5 },
        }),
        headers: new Headers({ 'content-type': 'application/json' }),
      });

      // Default failureThreshold = 5. 10 attempts is more than enough with
      // the probabilistic gate pinned open.
      for (let i = 0; i < 10; i++) {
        try {
          await provider.generateResponse({
            messages: [{ role: 'user', content: 'hi' }],
            model: 'claude-haiku-4-5-20251001',
          });
        } catch {
          // Expected - drift or circuit-open
        }
      }

      const breaker = defaultCircuitBreakerManager.getBreaker('anthropic');
      expect(breaker.getState().state).toBe('OPEN');
    } finally {
      randomSpy.mockRestore();
    }
  });
});

// ── Security: no value leakage in error surface ──────────────────────────

describe('SchemaDriftError message surface (security)', () => {
  it('error message and hook payload contain only path + type names, never values', async () => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
    const provider = new AnthropicProvider({ apiKey: 'test-key', maxRetries: 0 });

    // Drifted response contains a sensitive-looking value. If it leaks into
    // the error message we have a telemetry PII problem.
    const SECRET_VALUE = 'sk-verysecret-api-key-shouldnotleak';
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: SECRET_VALUE, // intentionally put a "secret" where we expect a string
        // content missing — triggers drift on content path instead of id
        model: 'claude-haiku-4-5-20251001',
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 5 },
      }),
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    try {
      await provider.generateResponse({
        messages: [{ role: 'user', content: SECRET_VALUE }],
        model: 'claude-haiku-4-5-20251001',
      });
      expect.fail('should have thrown');
    } catch (err) {
      const message = (err as Error).message;
      const serialized = JSON.stringify(err, Object.getOwnPropertyNames(err));

      // The path, expected type, actual type should all be present
      expect(message).toContain('content');
      expect(message).toContain('array');
      expect(message).toContain('undefined');

      // The secret value should NEVER be in the error surface
      expect(message).not.toContain(SECRET_VALUE);
      expect(serialized).not.toContain(SECRET_VALUE);
    }
  });
});

// ── H-2: nested content block validation ─────────────────────────────────

describe('Anthropic nested content-block validation (H-2 / #42)', () => {
  let provider: AnthropicProvider;

  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
    provider = new AnthropicProvider({ apiKey: 'test-key', maxRetries: 0 });
  });

  const envelope = (content: unknown[]) => ({
    id: 'msg_1',
    type: 'message',
    role: 'assistant',
    content,
    model: 'claude-haiku-4-5-20251001',
    stop_reason: 'end_turn',
    usage: { input_tokens: 10, output_tokens: 5 },
  });

  it('accepts a valid tool_use block', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => envelope([
        { type: 'tool_use', id: 'toolu_1', name: 'search', input: { q: 'test' } },
      ]),
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    const res = await provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model: 'claude-haiku-4-5-20251001',
    });
    expect(res.toolCalls).toHaveLength(1);
  });

  it('detects tool_use block with missing id', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => envelope([
        { type: 'tool_use', name: 'search', input: { q: 'test' } },
      ]),
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    await expect(provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model: 'claude-haiku-4-5-20251001',
    })).rejects.toMatchObject({
      code: 'SCHEMA_DRIFT',
      path: 'content[0].id',
      expected: 'string',
      actual: 'undefined',
    });
  });

  it('detects tool_use block with wrong-typed input', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => envelope([
        { type: 'tool_use', id: 'toolu_1', name: 'search', input: 'should-be-object' },
      ]),
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    await expect(provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model: 'claude-haiku-4-5-20251001',
    })).rejects.toMatchObject({
      code: 'SCHEMA_DRIFT',
      path: 'content[0].input',
      expected: 'object',
      actual: 'string',
    });
  });

  it('detects text block with missing .text field', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => envelope([
        { type: 'text' }, // .text renamed/removed
      ]),
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    await expect(provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model: 'claude-haiku-4-5-20251001',
    })).rejects.toMatchObject({
      code: 'SCHEMA_DRIFT',
      path: 'content[0].text',
    });
  });

  it('detects element that is not an object', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => envelope(['a bare string, not a block']),
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    await expect(provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model: 'claude-haiku-4-5-20251001',
    })).rejects.toMatchObject({
      code: 'SCHEMA_DRIFT',
      path: 'content[0]',
      expected: 'object',
      actual: 'string',
    });
  });

  it('detects missing discriminator (type field)', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => envelope([
        { text: 'no type field' },
      ]),
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    await expect(provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model: 'claude-haiku-4-5-20251001',
    })).rejects.toMatchObject({
      code: 'SCHEMA_DRIFT',
      path: 'content[0].type',
      expected: 'string',
    });
  });

  it('accepts unknown block types (forward-compat)', async () => {
    // Anthropic adds a new block type 'reasoning' — we should pass without
    // error so the next deploy doesn't break on additive changes.
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => envelope([
        { type: 'text', text: 'hello' },
        { type: 'reasoning', thoughts: 'internal thinking trace', confidence: 0.9 },
      ]),
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    const res = await provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model: 'claude-haiku-4-5-20251001',
    });
    // Text content extracted; unknown block silently ignored by filter
    expect(res.content).toBe('hello');
  });

  it('accepts response without stop_sequence (optional field)', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => envelope([{ type: 'text', text: 'hi' }]),
      // Note: no stop_sequence key — it's optional
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    const res = await provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model: 'claude-haiku-4-5-20251001',
    });
    expect(res.content).toBe('hi');
  });
});

// ── OpenAI-compat provider schema validation ────────────────────────────
//
// OpenAI, Groq, and Cerebras all serve the /chat/completions envelope.
// Driven through describe.each so drift-parity is enforced by construction —
// if one provider's schema diverges, its tests break loudly.

interface OpenAICompatCase {
  name: string;
  factory: () => BaseProvider;
  model: string;
}

const openAiCompatCases: OpenAICompatCase[] = [
  {
    name: 'openai',
    factory: () => new OpenAIProvider({ apiKey: 'test-key', maxRetries: 0 }),
    model: 'gpt-4o-mini',
  },
  {
    name: 'groq',
    factory: () => new GroqProvider({ apiKey: 'test-key', maxRetries: 0 }),
    model: 'llama-3.1-8b-instant',
  },
  {
    name: 'cerebras',
    factory: () => new CerebrasProvider({ apiKey: 'test-key', maxRetries: 0 }),
    model: 'llama-3.1-8b',
  },
];

describe.each(openAiCompatCases)('$name response schema validation', ({ name, factory, model }) => {
  let provider: BaseProvider;

  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
    provider = factory();
  });

  const validResponse = {
    id: 'chatcmpl_1',
    object: 'chat.completion',
    created: 1700000000,
    model,
    choices: [{
      index: 0,
      message: { role: 'assistant', content: 'hello' },
      finish_reason: 'stop',
    }],
    usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  };

  it('passes through a well-formed response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => validResponse,
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    const res = await provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model,
    });

    expect(res.content).toBe('hello');
    expect(res.usage.inputTokens).toBe(10);
  });

  it('throws SchemaDriftError when usage.prompt_tokens is renamed', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        ...validResponse,
        usage: { input_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      }),
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    await expect(provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model,
    })).rejects.toMatchObject({
      code: 'SCHEMA_DRIFT',
      provider: name,
      path: 'usage.prompt_tokens',
    });
  });

  it('throws SchemaDriftError when choices field is removed', async () => {
    const { choices: _choices, ...rest } = validResponse;
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => rest,
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    await expect(provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model,
    })).rejects.toMatchObject({ code: 'SCHEMA_DRIFT', path: 'choices' });
  });

  it('throws SchemaDriftError when choices is empty (routes through drift, not bare throw)', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ ...validResponse, choices: [] }),
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    await expect(provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model,
    })).rejects.toMatchObject({
      code: 'SCHEMA_DRIFT',
      provider: name,
      path: 'choices[0]',
    });
  });

  it('throws SchemaDriftError when tool_call function.arguments is not a string', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        ...validResponse,
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: null,
            tool_calls: [{
              id: 'call_1',
              type: 'function',
              function: { name: 'my_tool', arguments: { already: 'parsed' } },
            }],
          },
          finish_reason: 'tool_calls',
        }],
      }),
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    await expect(provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model,
    })).rejects.toMatchObject({
      code: 'SCHEMA_DRIFT',
      provider: name,
      path: 'choices[0].message.tool_calls[0].function.arguments',
      expected: 'string',
      actual: 'object',
    });
  });

  it('accepts unknown tool_call type without surfacing it as a function call (forward-compat)', async () => {
    // Schema's discriminator skips unknown `type` values (forward-compat for
    // additive upstream changes). The provider's formatResponse must not
    // dereference the function-shaped payload on a skipped variant, or an
    // unknown shape becomes a bare TypeError (bypassing drift/fallback) or
    // gets mis-surfaced as a normal function call. Mock omits `function`
    // entirely to exercise the TypeError path specifically.
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        ...validResponse,
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: 'hi',
            tool_calls: [{
              id: 'call_1',
              type: 'code_interpreter', // hypothetical future tool type
              // intentionally no `function` field — unknown variants may have a
              // different shape upstream
            }],
          },
          finish_reason: 'stop',
        }],
      }),
      headers: new Headers({ 'content-type': 'application/json' }),
    });

    const res = await provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model,
    });
    expect(res.content).toBe('hi');
    // Critical: the unknown variant must be dropped, not mis-surfaced as a
    // function call and not crashed through.
    expect(res.toolCalls).toBeUndefined();
  });
});
