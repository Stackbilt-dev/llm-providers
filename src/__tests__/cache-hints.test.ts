/**
 * Cache hint translation tests (#52)
 *
 * Covers:
 * - Anthropic cache_control breakpoints on system + tools
 * - OpenAI cached token extraction from prompt_tokens_details
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AnthropicProvider } from '../providers/anthropic';
import { OpenAIProvider } from '../providers/openai';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';

const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

// ── helpers ──────────────────────────────────────────────────────────────────

const JSON_HEADERS = new Headers({ 'content-type': 'application/json' });

function anthropicOkResponse(usageOverrides: Record<string, unknown> = {}) {
  return {
    ok: true,
    headers: JSON_HEADERS,
    json: async () => ({
      id: 'msg_1',
      type: 'message',
      role: 'assistant',
      content: [{ type: 'text', text: 'hello' }],
      model: 'claude-haiku-4-5-20251001',
      stop_reason: 'end_turn',
      stop_sequence: null,
      usage: { input_tokens: 10, output_tokens: 5, ...usageOverrides },
    }),
  };
}

function openAiOkResponse(usageOverrides: Record<string, unknown> = {}) {
  return {
    ok: true,
    headers: JSON_HEADERS,
    json: async () => ({
      id: 'chatcmpl_1',
      object: 'chat.completion',
      created: 1700000000,
      model: 'gpt-4o-mini',
      choices: [{ index: 0, message: { role: 'assistant', content: 'hi' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15, ...usageOverrides },
    }),
  };
}

// ── Anthropic cache_control breakpoints ──────────────────────────────────────

describe('AnthropicProvider cache hint translation (#52)', () => {
  let provider: AnthropicProvider;

  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
    provider = new AnthropicProvider({ apiKey: 'test-key', maxRetries: 0 });
  });

  it('does not add cache_control when no cache hint is set', async () => {
    mockFetch.mockResolvedValueOnce(anthropicOkResponse());

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      systemPrompt: 'You are helpful.',
      model: 'claude-haiku-4-5-20251001',
    });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as RequestInit).body as string);
    expect(typeof body.system).toBe('string');
  });

  it('wraps system as content block with cache_control when strategy is provider-prefix', async () => {
    mockFetch.mockResolvedValueOnce(anthropicOkResponse());

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      systemPrompt: 'You are helpful.',
      model: 'claude-haiku-4-5-20251001',
      cache: { strategy: 'provider-prefix' },
    });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as RequestInit).body as string);
    expect(Array.isArray(body.system)).toBe(true);
    expect(body.system[0]).toMatchObject({
      type: 'text',
      text: 'You are helpful.',
      cache_control: { type: 'ephemeral' },
    });
  });

  it('wraps system with cache_control when strategy is both', async () => {
    mockFetch.mockResolvedValueOnce(anthropicOkResponse());

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      systemPrompt: 'System.',
      model: 'claude-haiku-4-5-20251001',
      cache: { strategy: 'both' },
    });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as RequestInit).body as string);
    expect(body.system[0].cache_control).toEqual({ type: 'ephemeral' });
  });

  it('marks the last tool with cache_control when strategy is provider-prefix and cacheablePrefix is auto', async () => {
    mockFetch.mockResolvedValueOnce(anthropicOkResponse());

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      systemPrompt: 'You are helpful.',
      model: 'claude-haiku-4-5-20251001',
      cache: { strategy: 'provider-prefix', cacheablePrefix: 'auto' },
      tools: [
        { type: 'function', function: { name: 'tool_a', description: 'A', parameters: {} } },
        { type: 'function', function: { name: 'tool_b', description: 'B', parameters: {} } },
      ],
    });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as RequestInit).body as string);
    expect(body.tools[0].cache_control).toBeUndefined();
    expect(body.tools[1].cache_control).toEqual({ type: 'ephemeral' });
  });

  it('does not mark tools when cacheablePrefix is system', async () => {
    mockFetch.mockResolvedValueOnce(anthropicOkResponse());

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      systemPrompt: 'You are helpful.',
      model: 'claude-haiku-4-5-20251001',
      cache: { strategy: 'provider-prefix', cacheablePrefix: 'system' },
      tools: [
        { type: 'function', function: { name: 'tool_a', description: 'A', parameters: {} } },
      ],
    });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as RequestInit).body as string);
    expect(body.tools[0].cache_control).toBeUndefined();
    expect(Array.isArray(body.system)).toBe(true);
  });

  it('does not add cache_control when strategy is off', async () => {
    mockFetch.mockResolvedValueOnce(anthropicOkResponse());

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      systemPrompt: 'System.',
      model: 'claude-haiku-4-5-20251001',
      cache: { strategy: 'off' },
    });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as RequestInit).body as string);
    expect(typeof body.system).toBe('string');
  });

  it('does not add cache_control when strategy is response (no prefix caching)', async () => {
    mockFetch.mockResolvedValueOnce(anthropicOkResponse());

    await provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      systemPrompt: 'System.',
      model: 'claude-haiku-4-5-20251001',
      cache: { strategy: 'response' },
    });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as RequestInit).body as string);
    expect(typeof body.system).toBe('string');
  });
});

// ── OpenAI cached token extraction ───────────────────────────────────────────

describe('OpenAIProvider cached token extraction (#52)', () => {
  let provider: OpenAIProvider;

  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
    provider = new OpenAIProvider({ apiKey: 'test-key', maxRetries: 0 });
  });

  it('populates cachedInputTokens when prompt_tokens_details.cached_tokens is present', async () => {
    mockFetch.mockResolvedValueOnce(openAiOkResponse({
      prompt_tokens_details: { cached_tokens: 8 },
    }));

    const resp = await provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model: 'gpt-4o-mini',
    });

    expect(resp.usage.cachedInputTokens).toBe(8);
  });

  it('does not set cachedInputTokens when prompt_tokens_details is absent', async () => {
    mockFetch.mockResolvedValueOnce(openAiOkResponse());

    const resp = await provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model: 'gpt-4o-mini',
    });

    expect(resp.usage.cachedInputTokens).toBeUndefined();
  });

  it('sets cachedInputTokens to 0 when cached_tokens is 0', async () => {
    mockFetch.mockResolvedValueOnce(openAiOkResponse({
      prompt_tokens_details: { cached_tokens: 0 },
    }));

    const resp = await provider.generateResponse({
      messages: [{ role: 'user', content: 'hi' }],
      model: 'gpt-4o-mini',
    });

    expect(resp.usage.cachedInputTokens).toBe(0);
  });
});

// ── Anthropic balance reporting ───────────────────────────────────────────────

describe('AnthropicProvider getProviderBalance (#25)', () => {
  it('returns unavailable with a message directing to the Admin API', async () => {
    const provider = new AnthropicProvider({ apiKey: 'test-key', maxRetries: 0 });
    const balance = await provider.getProviderBalance();

    expect(balance.provider).toBe('anthropic');
    expect(balance.status).toBe('unavailable');
    expect(balance.source).toBe('not_supported');
    expect(balance.message).toMatch(/Admin API/i);
  });
});
