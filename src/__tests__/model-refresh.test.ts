/**
 * Model list refresh (#112, phase 2).
 *
 * Expected values are taken from primary vendor documentation, checked 2026-10-15:
 * - Workers AI: developers.cloudflare.com/workers-ai/models/ (each model page) and
 *   /workers-ai/platform/pricing/ (updated 2026-10-15); deprecations from the
 *   2026-05-08 "planned model deprecations" changelog post and the 2025-09-18 entry.
 * - OpenAI: developers.openai.com/api/docs/{models,pricing,deprecations}
 * - Anthropic: platform.claude.com/docs/en/about-claude/{models/overview,pricing,model-deprecations}
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AnthropicProvider } from '../providers/anthropic';
import { CloudflareProvider } from '../providers/cloudflare';
import { OpenAIProvider, isOpenAIReasoningModel } from '../providers/openai';
import {
  getCatalogEntry, getRoutingInfo, MODEL_RECOMMENDATIONS, rankModels, type ModelRecommendationUseCase,
} from '../model-catalog';
import { MODELS } from '../index';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import type { LLMRequest } from '../types';

const request: LLMRequest = { messages: [{ role: 'user', content: 'hello' }] };
const JSON_HEADERS = new Headers({ 'content-type': 'application/json' });
const USE_CASES: ModelRecommendationUseCase[] = [
  'COST_EFFECTIVE', 'HIGH_PERFORMANCE', 'BALANCED', 'TOOL_CALLING', 'LONG_CONTEXT', 'VISION', 'RESEARCH'];

function jsonReply(body: unknown) {
  return { ok: true, status: 200, headers: JSON_HEADERS, json: async () => body };
}
function openaiReply(model: string) {
  return jsonReply({
    id: 'chatcmpl_1', object: 'chat.completion', created: 1, model,
    choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  });
}
function anthropicReply(model: string, usage: Record<string, unknown> = { input_tokens: 1, output_tokens: 1 }) {
  return jsonReply({
    id: 'msg_1', type: 'message', role: 'assistant', content: [{ type: 'text', text: 'ok' }],
    model, stop_reason: 'end_turn', stop_sequence: null, usage,
  });
}
function sentBody(fetchMock: ReturnType<typeof vi.fn>): Record<string, unknown> {
  return JSON.parse(fetchMock.mock.calls[0][1].body as string);
}

beforeEach(() => { defaultCircuitBreakerManager.resetAll(); });
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe('Cloudflare-first routing', () => {
  it.each(USE_CASES)('ranks a Workers AI model first for %s when every provider is configured', useCase => {
    const ranked = rankModels(useCase, ['openai', 'anthropic', 'cloudflare', 'cerebras', 'groq', 'nvidia']);
    expect(ranked[0]?.provider).toBe('cloudflare');
  });
});

describe('Workers AI catalog', () => {
  const removed = [
    '@cf/meta/llama-3.1-8b-instruct', '@cf/meta/llama-3.1-70b-instruct', '@cf/meta/llama-3-8b-instruct',
    '@cf/meta/llama-2-7b-chat-int8', '@cf/microsoft/phi-2', '@cf/mistral/mistral-7b-instruct-v0.1',
    '@cf/openchat/openchat-3.5-0106', '@cf/tinyllama/tinyllama-1.1b-chat-v1.0', '@cf/qwen/qwen1.5-0.5b-chat',
    '@cf/qwen/qwen1.5-1.8b-chat', '@cf/qwen/qwen1.5-14b-chat-awq', '@cf/qwen/qwen1.5-7b-chat-awq',
  ];

  it.each(removed)('%s is retired and never recommended', model => {
    expect(getCatalogEntry(model)?.lifecycle).toBe('retired');
    expect(Object.values(MODEL_RECOMMENDATIONS).flat()).not.toContain(model);
  });

  it('keeps removed IDs accepted by the adapter for explicit callers', () => {
    const provider = new CloudflareProvider({ ai: { run: vi.fn() } as unknown as Ai });
    for (const model of removed) expect(provider.models).toContain(model);
  });

  it.each([
    ['@cf/meta/llama-3.3-70b-instruct-fp8-fast', 24_000],
    ['@cf/meta/llama-3.2-1b-instruct', 60_000],
    ['@cf/meta/llama-3.2-3b-instruct', 80_000],
    ['@cf/qwen/qwen3-30b-a3b-fp8', 32_768],
    ['@cf/moonshotai/kimi-k2.6', 262_144],
    ['@cf/moonshotai/kimi-k2.7-code', 262_144],
    ['@cf/deepseek-ai/deepseek-v4-flash-0731', 1_048_576],
    ['deepseek/deepseek-v4-pro', 131_072],
  ])('%s has the documented %i-token context window', (model, context) => {
    expect(getCatalogEntry(model)?.capabilities.maxContextLength).toBe(context);
  });

  it('does not claim function calling where the model page does not list it', () => {
    const coder = getCatalogEntry('@cf/qwen/qwen2.5-coder-32b-instruct')!;
    expect(coder.capabilities.supportsTools).toBe(false);
    expect(coder.useCases).not.toContain('TOOL_CALLING');
    const toolRequest: LLMRequest = {
      ...request,
      tools: [{ type: 'function', function: { name: 'lookup', description: 'Lookup', parameters: { type: 'object' } } }],
    };
    const ranked = rankModels('TOOL_CALLING', ['cloudflare'], { request: toolRequest });
    expect(ranked.map(entry => entry.model)).not.toContain(coder.model);
  });

  it.each([
    ['@cf/meta/llama-3.1-8b-instruct-fp8', 0.152, 0.287],
    ['@cf/ibm-granite/granite-4.0-h-micro', 0.017, 0.112],
    ['@cf/aisingapore/gemma-sea-lion-v4-27b-it', 0.351, 0.555],
    ['deepseek/deepseek-v4-pro', 1.74, 3.48],
    ['anthropic/claude-opus-4.8', 5, 25],
  ])('%s costs $%s/$%s per million tokens in the adapter and the catalog', async (model, inPerM, outPerM) => {
    const run = vi.fn().mockResolvedValue({ response: 'ok', usage: { prompt_tokens: 1_000_000, completion_tokens: 1_000_000 } });
    const provider = new CloudflareProvider({ ai: { run } as unknown as Ai, maxRetries: 0 });
    expect(provider.models).toContain(model);
    const entry = getCatalogEntry(model)!;
    expect(entry.provider).toBe('cloudflare');
    expect(entry.capabilities.inputTokenCost * 1000).toBeCloseTo(inPerM, 6);
    expect(entry.capabilities.outputTokenCost * 1000).toBeCloseTo(outPerM, 6);
    if (model.startsWith('@cf/')) {
      const response = await provider.generateResponse({ ...request, model });
      expect(response.usage.cost).toBeCloseTo(inPerM + outPerM, 6);
    }
  });
});

describe('OpenAI models', () => {
  const current = ['gpt-6-astra', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna', 'gpt-5.5'];

  it.each(['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'])('%s is out of routing', model => {
    expect(getCatalogEntry(model)?.lifecycle).toBe('retired');
    expect(Object.values(MODEL_RECOMMENDATIONS).flat()).not.toContain(model);
  });

  it('flags gpt-4o-mini with its 2027-01-14 shutdown', () => {
    const info = getRoutingInfo({ ...request, model: 'gpt-4o-mini' }, ['openai']);
    expect(info.modelLifecycle).toBe('compatibility');
    expect(info.deprecationWarning).toContain('2027-01-14');
  });

  it.each(current)('%s is accepted by the adapter and catalogued as active', model => {
    expect(new OpenAIProvider({ apiKey: 'k' }).models).toContain(model);
    expect(getCatalogEntry(model)).toMatchObject({ provider: 'openai', lifecycle: 'active' });
  });

  it('defaults OpenAI routing to a current model', () => {
    expect(getRoutingInfo(request, ['openai']).model).toBe('gpt-5.6-luna');
  });

  it.each(current)('sends max_completion_tokens and no temperature to %s', async model => {
    expect(isOpenAIReasoningModel(model)).toBe(true);
    const fetchMock = vi.fn().mockResolvedValue(openaiReply(model));
    vi.stubGlobal('fetch', fetchMock);
    await new OpenAIProvider({ apiKey: 'k', maxRetries: 0 })
      .generateResponse({ ...request, model, temperature: 0.2, maxTokens: 500 });
    const body = sentBody(fetchMock);
    expect(body.max_completion_tokens).toBe(500);
    expect(body).not.toHaveProperty('max_tokens');
    expect(body).not.toHaveProperty('temperature');
  });

  it('keeps max_tokens and temperature for non-reasoning models', async () => {
    expect(isOpenAIReasoningModel('gpt-4o-mini')).toBe(false);
    const fetchMock = vi.fn().mockResolvedValue(openaiReply('gpt-4o-mini'));
    vi.stubGlobal('fetch', fetchMock);
    await new OpenAIProvider({ apiKey: 'k', maxRetries: 0 })
      .generateResponse({ ...request, model: 'gpt-4o-mini', temperature: 0.2, maxTokens: 500 });
    const body = sentBody(fetchMock);
    expect(body).toMatchObject({ max_tokens: 500, temperature: 0.2 });
    expect(body).not.toHaveProperty('max_completion_tokens');
  });

  it('sends the same reasoning parameters on the streaming path', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, body: new ReadableStream({ start: c => c.close() }) });
    vi.stubGlobal('fetch', fetchMock);
    const stream = await new OpenAIProvider({ apiKey: 'k', maxRetries: 0 })
      .streamResponse({ ...request, model: 'gpt-5.6-sol', temperature: 0.2, maxTokens: 500 });
    await stream.getReader().read().catch(() => undefined);
    const body = sentBody(fetchMock);
    expect(body.max_completion_tokens).toBe(500);
    expect(body).not.toHaveProperty('temperature');
  });
});

describe('Anthropic models', () => {
  const current: Array<[string, number, number]> = [
    ['claude-fable-5-1', 10, 50],
    ['claude-opus-5-5', 4, 20],
    ['claude-sonnet-5-5', 2, 10],
    ['claude-haiku-5-5', 0.1, 0.5],
  ];

  it.each(current)('%s is accepted, active and priced at $%s/$%s per million tokens', (model, inPerM, outPerM) => {
    expect(new AnthropicProvider({ apiKey: 'k' }).models).toContain(model);
    const entry = getCatalogEntry(model)!;
    expect(entry).toMatchObject({ provider: 'anthropic', lifecycle: 'active' });
    expect(entry.capabilities.maxContextLength).toBe(1_000_000);
    expect(entry.capabilities.inputTokenCost * 1000).toBeCloseTo(inPerM, 6);
    expect(entry.capabilities.outputTokenCost * 1000).toBeCloseTo(outPerM, 6);
  });

  it.each(['claude-opus-4-6', 'claude-sonnet-4-6', 'claude-haiku-4-5-20251001'])('%s is legacy', model => {
    expect(getCatalogEntry(model)?.lifecycle).toBe('compatibility');
  });

  it.each(current.map(([model]) => model))('does not send temperature to %s', async model => {
    const fetchMock = vi.fn().mockResolvedValue(anthropicReply(model));
    vi.stubGlobal('fetch', fetchMock);
    await new AnthropicProvider({ apiKey: 'k', maxRetries: 0 }).generateResponse({ ...request, model, temperature: 0.2 });
    expect(sentBody(fetchMock)).not.toHaveProperty('temperature');
  });

  it('still sends temperature to Claude 4.6', async () => {
    const fetchMock = vi.fn().mockResolvedValue(anthropicReply('claude-sonnet-4-6'));
    vi.stubGlobal('fetch', fetchMock);
    await new AnthropicProvider({ apiKey: 'k', maxRetries: 0 })
      .generateResponse({ ...request, model: 'claude-sonnet-4-6', temperature: 0.2 });
    expect(sentBody(fetchMock).temperature).toBe(0.2);
  });

  it.each([
    // [model, input $/M, cache-read fraction of input]
    ['claude-fable-5-1', 10, 0.025],
    ['claude-opus-5-5', 4, 0.05],
    ['claude-sonnet-5-5', 2, 0.05],
    ['claude-sonnet-4-6', 3, 0.1],
  ] as const)('prices %s cache reads at the documented rate', async (model, inPerM, readFraction) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(anthropicReply(model,
      { input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 1_000_000 })));
    const response = await new AnthropicProvider({ apiKey: 'k', maxRetries: 0 }).generateResponse({ ...request, model });
    expect(response.usage.cost).toBeCloseTo(inPerM * readFraction, 6);
  });

  it('bills Haiku 5.5 prompts over 100K tokens, cache included, at the long-prompt rate', async () => {
    const provider = new AnthropicProvider({ apiKey: 'k', maxRetries: 0 });
    const cost = async (usage: Record<string, unknown>) => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(anthropicReply('claude-haiku-5-5', usage)));
      return (await provider.generateResponse({ ...request, model: 'claude-haiku-5-5' })).usage.cost;
    };
    // 100K input + 1K output at $0.10/$0.50 per million
    expect(await cost({ input_tokens: 100_000, output_tokens: 1_000 })).toBeCloseTo(0.0105, 8);
    // 60K input + 50K cache read: 110K prompt, so $0.50/$2.50 and $0.05 cache reads
    expect(await cost({ input_tokens: 60_000, output_tokens: 1_000, cache_read_input_tokens: 50_000 }))
      .toBeCloseTo(0.03 + 0.0025 + 0.0025, 8);
  });
});

describe('MODELS constants', () => {
  it.each([
    'GPT_6_ASTRA', 'GPT_5_6_SOL', 'GPT_5_6_TERRA', 'GPT_5_6_LUNA', 'GPT_5_5',
    'CLAUDE_FABLE_5_1', 'CLAUDE_OPUS_5_5', 'CLAUDE_SONNET_5_5', 'CLAUDE_HAIKU_5_5',
    'CF_GLM_5_3', 'CF_GLM_5_3_FLASH', 'CF_KIMI_K2_6', 'CF_DEEPSEEK_V4_PRO', 'CF_DEEPSEEK_V4_FLASH',
    'CF_GPT_OSS_120B', 'CF_GPT_OSS_20B', 'CF_GEMMA_4_26B', 'CF_QWEN_3_8_27B', 'CF_LLAMA_3_3_70B_FAST',
    'CF_LLAMA_3_1_8B_FP8', 'CF_GRANITE_4_MICRO',
  ] as const)('%s names an active catalog model', key => {
    expect(getCatalogEntry(MODELS[key])?.lifecycle).toBe('active');
  });
});
