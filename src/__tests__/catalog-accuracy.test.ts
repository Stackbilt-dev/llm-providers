/**
 * Vendor ID and price corrections (#112, phase 1).
 *
 * Expected values are taken from primary vendor documentation, checked 2026-10-15:
 * - Workers AI: developers.cloudflare.com/workers-ai/platform/pricing/ (updated 2026-10-09)
 * - Anthropic: platform.claude.com/docs/en/about-claude/{models/overview,pricing,model-deprecations}
 * - OpenAI: developers.openai.com/api/docs/deprecations
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AnthropicProvider } from '../providers/anthropic';
import { CloudflareProvider } from '../providers/cloudflare';
import { getCatalogEntry, getProviderDefaultModel, getRoutingInfo, MODEL_CATALOG } from '../model-catalog';
import { MODELS } from '../index';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { getStreamUsage } from '../utils/stream-usage';
import type { LLMRequest } from '../types';

const request: LLMRequest = { messages: [{ role: 'user', content: 'hello' }] };
const JSON_HEADERS = new Headers({ 'content-type': 'application/json' });

function anthropicReply(model: string, usage: Record<string, unknown>) {
  return {
    ok: true, status: 200, headers: JSON_HEADERS,
    json: async () => ({
      id: 'msg_1', type: 'message', role: 'assistant', content: [{ type: 'text', text: 'ok' }],
      model, stop_reason: 'end_turn', stop_sequence: null, usage,
    }),
  };
}
function sse(text: string) {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(bytes); controller.close(); } });
}
async function drain(stream: ReadableStream<string>) {
  const reader = stream.getReader();
  while (!(await reader.read()).done) { /* consume */ }
}

beforeEach(() => { defaultCircuitBreakerManager.resetAll(); });
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe('Workers AI prices', () => {
  it.each([
    ['@cf/meta/llama-3.1-8b-instruct', 0.282, 0.827],
    ['@cf/meta/llama-3-8b-instruct', 0.282, 0.827],
    ['@cf/mistral/mistral-7b-instruct-v0.1', 0.110, 0.190],
  ])('%s costs $%s/$%s per million tokens in the adapter and the catalog', async (model, inPerM, outPerM) => {
    const run = vi.fn().mockResolvedValue({ response: 'ok', usage: { prompt_tokens: 1_000_000, completion_tokens: 1_000_000 } });
    const provider = new CloudflareProvider({ ai: { run } as unknown as Ai, maxRetries: 0 });
    const response = await provider.generateResponse({ ...request, model });
    expect(response.usage.cost).toBeCloseTo(inPerM + outPerM, 6);

    const entry = MODEL_CATALOG.find(candidate => candidate.provider === 'cloudflare' && candidate.model === model)!;
    expect(entry.capabilities.inputTokenCost * 1000).toBeCloseTo(inPerM, 6);
    expect(entry.capabilities.outputTokenCost * 1000).toBeCloseTo(outPerM, 6);
  });
});

describe('Anthropic model IDs', () => {
  it('exports and catalogs the real dateless 4.6 IDs', () => {
    expect(MODELS.CLAUDE_OPUS_4_6).toBe('claude-opus-4-6');
    expect(MODELS.CLAUDE_SONNET_4_6).toBe('claude-sonnet-4-6');
    expect(MODELS.CLAUDE_OPUS_4).toBe('claude-opus-4-20250514');
    expect(MODEL_CATALOG.some(entry => entry.model.endsWith('-20250618'))).toBe(false);
  });

  it.each([
    ['claude-opus-4-6-20250618', 'claude-opus-4-6'],
    ['claude-sonnet-4-6-20250618', 'claude-sonnet-4-6'],
    ['claude-opus-4-20250618', 'claude-opus-4-20250514'],
  ])('accepts the previously published %s and sends %s', async (legacy, real) => {
    const fetch = vi.fn().mockResolvedValue(anthropicReply(real, { input_tokens: 1, output_tokens: 1 }));
    vi.stubGlobal('fetch', fetch);
    const provider = new AnthropicProvider({ apiKey: 'k', maxRetries: 0 });
    await provider.generateResponse({ ...request, model: legacy });
    expect(JSON.parse(fetch.mock.calls[0][1].body).model).toBe(real);
    expect(getCatalogEntry(legacy)?.model).toBe(real);
  });

  it('does not route to models retired on the Claude API', () => {
    const retired = ['claude-opus-4-20250514', 'claude-sonnet-4-20250514', 'claude-3-7-sonnet-20250219',
      'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229'];
    for (const model of retired) expect(getCatalogEntry(model)?.lifecycle).toBe('retired');
    expect(retired).not.toContain(getProviderDefaultModel('anthropic', request));
  });

  it('prices Opus 4.6 at $5/$25 per million tokens', () => {
    const entry = getCatalogEntry('claude-opus-4-6')!;
    expect(entry.capabilities.inputTokenCost * 1000).toBeCloseTo(5, 6);
    expect(entry.capabilities.outputTokenCost * 1000).toBeCloseTo(25, 6);
  });
});

describe('Anthropic prompt-cache pricing', () => {
  // Sonnet 4.6: $3 input, $15 output; cache reads 0.1x, 5m writes 1.25x, 1h writes 2x of input.
  const usage = {
    input_tokens: 1_000_000, output_tokens: 1_000_000,
    cache_read_input_tokens: 1_000_000, cache_creation_input_tokens: 1_000_000,
    cache_creation: { ephemeral_5m_input_tokens: 600_000, ephemeral_1h_input_tokens: 400_000 },
  };
  const expected = 3 + 15 + 0.3 + 0.6 * 3.75 + 0.4 * 6;

  it('includes cache reads and writes in non-streaming cost', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(anthropicReply('claude-sonnet-4-6', usage)));
    const provider = new AnthropicProvider({ apiKey: 'k', maxRetries: 0 });
    const response = await provider.generateResponse({ ...request, model: 'claude-sonnet-4-6' });
    expect(response.usage.cost).toBeCloseTo(expected, 6);
  });

  it('prices cache writes at the 5-minute rate when no TTL split is reported', async () => {
    const { cache_creation: _split, ...noSplit } = usage;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(anthropicReply('claude-sonnet-4-6', noSplit)));
    const provider = new AnthropicProvider({ apiKey: 'k', maxRetries: 0 });
    const response = await provider.generateResponse({ ...request, model: 'claude-sonnet-4-6' });
    expect(response.usage.cost).toBeCloseTo(3 + 15 + 0.3 + 3.75, 6);
  });

  it('includes cache reads and writes in streaming cost', async () => {
    const start = { type: 'message_start', message: { usage: { ...usage, output_tokens: 0 } } };
    const body = `data:${JSON.stringify(start)}\n\n`
      + 'data:{"type":"content_block_delta","delta":{"type":"text_delta","text":"ok"}}\n\n'
      + `data:${JSON.stringify({ type: 'message_delta', usage: { output_tokens: 1_000_000 } })}\n\n`
      + 'data:{"type":"message_stop"}\n\n';
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, body: sse(body) }));
    const provider = new AnthropicProvider({ apiKey: 'k', maxRetries: 0 });
    const stream = await provider.streamResponse({ ...request, model: 'claude-sonnet-4-6' });
    await drain(stream);
    expect((await getStreamUsage(stream))?.cost).toBeCloseTo(expected, 6);
  });
});

describe('OpenAI lifecycle notes', () => {
  it.each(['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'])('%s warns about its 2026-10-23 shutdown', model => {
    expect(getRoutingInfo({ ...request, model }, ['openai']).deprecationWarning).toContain('2026-10-23');
  });

  it('no longer claims the gpt-4o alias was retired on 2026-04-03', () => {
    expect(getCatalogEntry('gpt-4o')?.capabilities.description).not.toMatch(/2026-04-03|Retired GPT-4o/);
  });
});
