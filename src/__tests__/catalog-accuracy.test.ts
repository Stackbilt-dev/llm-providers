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
import { getCatalogEntry, getRecommendedModel, getRoutingInfo, MODEL_CATALOG, type ModelRecommendationUseCase } from '../model-catalog';
import { getGatewayRoutePlan } from '../gateway-routing';
import { LLMProviders, MODELS } from '../index';
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

  const useCases: ModelRecommendationUseCase[] = [
    'COST_EFFECTIVE', 'HIGH_PERFORMANCE', 'BALANCED', 'TOOL_CALLING', 'LONG_CONTEXT', 'VISION', 'RESEARCH'];
  const retired = ['claude-opus-4-20250514', 'claude-sonnet-4-20250514', 'claude-3-7-sonnet-20250219',
    'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229'];

  it.each(useCases)('does not recommend a retired Claude model for %s', useCase => {
    for (const model of retired) expect(getCatalogEntry(model)?.lifecycle).toBe('retired');
    expect(retired).not.toContain(getRecommendedModel(useCase, ['anthropic']));
  });

  it('prices and sizes models to match the vendor pages', () => {
    const opus = getCatalogEntry('claude-opus-4-6')!;
    expect(opus.capabilities.inputTokenCost * 1000).toBeCloseTo(5, 6);
    expect(opus.capabilities.outputTokenCost * 1000).toBeCloseTo(25, 6);
    expect(opus.capabilities.maxContextLength).toBe(1_000_000);
    expect(getCatalogEntry('claude-sonnet-4-6')!.capabilities.maxContextLength).toBe(1_000_000);
    const haiku35 = getCatalogEntry('claude-3-5-haiku-20241022')!;
    expect(haiku35.capabilities.inputTokenCost * 1000).toBeCloseTo(0.8, 6);
    expect(haiku35.capabilities.outputTokenCost * 1000).toBeCloseTo(4, 6);
  });
});

describe('previously published Anthropic IDs outside the adapter', () => {
  it.each([
    ['claude-opus-4-6-20250618', 'claude-opus-4-6'],
    ['claude-sonnet-4-6-20250618', 'claude-sonnet-4-6'],
  ])('routing preflight and gateway plans treat %s as %s', (legacy, real) => {
    const info = getRoutingInfo({ ...request, model: legacy }, ['anthropic']);
    expect(info).toMatchObject({ model: real, provider: 'anthropic', modelLifecycle: 'active' });
    expect(info.catalogEntry?.model).toBe(real);
    const plan = getGatewayRoutePlan({ ...request, model: legacy, stream: true }, ['anthropic']);
    expect(plan.executable).toBe(true);
    expect(plan.selectedModel).toBe(real);
  });

  it('the factory checks quota, emits hooks and records usage under the real ID', async () => {
    const generate = vi.spyOn(AnthropicProvider.prototype, 'generateResponse').mockResolvedValue({
      message: 'ok', provider: 'anthropic', model: 'claude-opus-4-6', responseTime: 1,
      usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2, cost: 0.00003 },
    });
    const check = vi.fn().mockResolvedValue({ allowed: true });
    const record = vi.fn(async () => {});
    const onRequestStart = vi.fn();
    const llm = new LLMProviders({ anthropic: { apiKey: 'k' }, stateScope: 'client',
      quotaHook: { check, record }, hooks: { onRequestStart } });

    await llm.generateResponse({ ...request, model: 'claude-opus-4-6-20250618' });

    expect(check).toHaveBeenCalledWith(expect.objectContaining({ model: 'claude-opus-4-6' }));
    expect(onRequestStart).toHaveBeenCalledWith(expect.objectContaining({ model: 'claude-opus-4-6' }));
    expect(generate.mock.calls[0][0].model).toBe('claude-opus-4-6');
    await vi.waitFor(() => expect(record).toHaveBeenCalledWith(expect.objectContaining({ model: 'claude-opus-4-6' })));
    expect(llm.getRoutingInfo({ ...request, model: 'claude-opus-4-6-20250618' }).model).toBe('claude-opus-4-6');
  });
});

describe('Anthropic prompt-cache pricing', () => {
  // Haiku 4.5 (ID unchanged by this fix): $1 input, $5 output; cache reads 0.1x, 5m writes 1.25x, 1h writes 2x.
  const model = 'claude-haiku-4-5-20251001';
  const usage = {
    input_tokens: 1_000_000, output_tokens: 1_000_000,
    cache_read_input_tokens: 1_000_000, cache_creation_input_tokens: 1_000_000,
    cache_creation: { ephemeral_5m_input_tokens: 600_000, ephemeral_1h_input_tokens: 400_000 },
  };
  const expected = 1 + 5 + 0.1 + 0.6 * 1.25 + 0.4 * 2;

  it('includes cache reads and writes in non-streaming cost', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(anthropicReply(model, usage)));
    const provider = new AnthropicProvider({ apiKey: 'k', maxRetries: 0 });
    const response = await provider.generateResponse({ ...request, model });
    expect(response.usage.cost).toBeCloseTo(expected, 6);
  });

  it('prices cache writes at the 5-minute rate when no TTL split is reported', async () => {
    const { cache_creation: _split, ...noSplit } = usage;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(anthropicReply(model, noSplit)));
    const provider = new AnthropicProvider({ apiKey: 'k', maxRetries: 0 });
    const response = await provider.generateResponse({ ...request, model });
    expect(response.usage.cost).toBeCloseTo(1 + 5 + 0.1 + 1.25, 6);
  });

  it('includes cache reads and writes in streaming cost', async () => {
    // message_delta usage is cumulative: its cache counts replace the ones from message_start.
    const start = { type: 'message_start', message: { usage: { input_tokens: 1_000_000, output_tokens: 0, cache_read_input_tokens: 0 } } };
    const body = `data:${JSON.stringify(start)}\n\n`
      + 'data:{"type":"content_block_delta","delta":{"type":"text_delta","text":"ok"}}\n\n'
      + `data:${JSON.stringify({ type: 'message_delta', usage })}\n\n`
      + 'data:{"type":"message_stop"}\n\n';
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, body: sse(body) }));
    const provider = new AnthropicProvider({ apiKey: 'k', maxRetries: 0 });
    const stream = await provider.streamResponse({ ...request, model });
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
