/**
 * #136: Workers AI output limits. Each model's documented default output limit
 * (`defaultMaxTokens`) and preferred parameter (`outputTokensParameter`) are in its
 * capabilities and catalog entry; the opt-in `CloudflareConfig.defaultMaxTokens` is
 * sent when a request sets no `maxTokens`; otherwise the adapter warns once per model
 * when the model's documented default is small.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CloudflareProvider } from '../providers/cloudflare';
import { cloudflareOutputLimits, SMALL_DEFAULT_MAX_TOKENS } from '../providers/cloudflare-output-limits';
import { ConfigurationError, LLMProviders, MODEL_CATALOG } from '../index';
import type { LLMRequest, ModelCapabilities } from '../types';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { defaultExhaustionRegistry } from '../utils/exhaustion';

const LLAMA_FAST = '@cf/meta/llama-3.3-70b-instruct-fp8-fast';
const QWEN3_30B = '@cf/qwen/qwen3-30b-a3b-fp8';
const GLM_5_3 = '@cf/zai-org/glm-5.3';

function chatResult(content = 'ok') {
  return {
    id: 'chatcmpl-test',
    choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
  };
}

function logger() {
  return { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
}

function provider(config: Record<string, unknown> = {}, run = vi.fn().mockResolvedValue(chatResult())) {
  const log = logger();
  const cf = new CloudflareProvider({ ai: { run } as unknown as Ai, maxRetries: 0, logger: log, ...config });
  return { cf, run, log };
}

function capabilitiesOf(cf: CloudflareProvider): Record<string, ModelCapabilities> {
  return (cf as unknown as { getModelCapabilities(): Record<string, ModelCapabilities> }).getModelCapabilities();
}

function request(model: string, extra: Partial<LLMRequest> = {}): LLMRequest {
  return { model, messages: [{ role: 'user', content: 'Summarise.' }], ...extra };
}

const sentMaxTokens = (run: ReturnType<typeof vi.fn>, call = 0) => (run.mock.calls[call][1] as { max_tokens?: number }).max_tokens;

beforeEach(() => {
  defaultCircuitBreakerManager.resetAll();
  defaultExhaustionRegistry.reset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('documented output limits', () => {
  it('records the documented defaults and preferred parameter', () => {
    expect(cloudflareOutputLimits(LLAMA_FAST)).toEqual({ defaultMaxTokens: 256, outputTokensParameter: 'max_tokens' });
    expect(cloudflareOutputLimits('@cf/openai/gpt-oss-120b')).toEqual({ defaultMaxTokens: 256, outputTokensParameter: 'max_tokens' });
    expect(cloudflareOutputLimits(QWEN3_30B)).toEqual({ defaultMaxTokens: 2000, outputTokensParameter: 'max_tokens' });
    expect(cloudflareOutputLimits(GLM_5_3)).toEqual({ outputTokensParameter: 'max_completion_tokens' });
    expect(cloudflareOutputLimits('@cf/nvidia/nemotron-3-120b-a12b')).toEqual({ outputTokensParameter: 'max_tokens' });
    expect(cloudflareOutputLimits('@cf/example/unknown')).toBeUndefined();
  });

  it('returns a copy, so callers cannot change the table', () => {
    const limits = cloudflareOutputLimits(LLAMA_FAST)!;
    limits.defaultMaxTokens = 1;
    expect(cloudflareOutputLimits(LLAMA_FAST)?.defaultMaxTokens).toBe(256);
  });

  it('covers every active Workers AI model in the catalog, and only models the adapter knows', () => {
    const { cf } = provider();
    const known = new Set(cf.getModels());
    const active = MODEL_CATALOG.filter(entry => entry.provider === 'cloudflare' && entry.lifecycle === 'active' && entry.model.startsWith('@cf/'));
    expect(active.length).toBeGreaterThan(20);
    for (const entry of active) {
      expect(cloudflareOutputLimits(entry.model), entry.model).toBeDefined();
      expect(known.has(entry.model), entry.model).toBe(true);
    }
  });

  it('is exposed in the adapter capabilities and the catalog', () => {
    const { cf } = provider();
    const capabilities = capabilitiesOf(cf);
    for (const model of [LLAMA_FAST, QWEN3_30B, GLM_5_3]) {
      const catalog = MODEL_CATALOG.find(entry => entry.provider === 'cloudflare' && entry.model === model)!.capabilities;
      expect(catalog.defaultMaxTokens).toBe(cloudflareOutputLimits(model)?.defaultMaxTokens);
      expect(catalog.outputTokensParameter).toBe(cloudflareOutputLimits(model)?.outputTokensParameter);
      expect(capabilities[model].defaultMaxTokens).toBe(catalog.defaultMaxTokens);
      expect(capabilities[model].outputTokensParameter).toBe(catalog.outputTokensParameter);
    }
    expect('defaultMaxTokens' in capabilities[GLM_5_3]).toBe(false);
  });

  it('leaves models of other providers unchanged', () => {
    for (const entry of MODEL_CATALOG.filter(candidate => candidate.provider !== 'cloudflare')) {
      expect(entry.capabilities.defaultMaxTokens, entry.model).toBeUndefined();
      expect(entry.capabilities.outputTokensParameter, entry.model).toBeUndefined();
    }
  });
});

describe('CloudflareConfig.defaultMaxTokens', () => {
  it.each([0, -1, 1.5, Number.NaN, Number.MAX_SAFE_INTEGER + 1, '4096', null])('rejects %s', value => {
    expect(() => provider({ defaultMaxTokens: value })).toThrow(ConfigurationError);
  });

  it('is not sent when unset: Workers AI applies the model default', async () => {
    const { cf, run } = provider();
    await cf.generateResponse(request(LLAMA_FAST));
    expect(sentMaxTokens(run)).toBeUndefined();
  });

  it('is sent when a request sets no maxTokens, and a request maxTokens wins', async () => {
    const { cf, run } = provider({ defaultMaxTokens: 4096 });
    await cf.generateResponse(request(LLAMA_FAST));
    await cf.generateResponse(request(LLAMA_FAST, { maxTokens: 100 }));
    await cf.generateResponse(request(GLM_5_3));
    expect([sentMaxTokens(run, 0), sentMaxTokens(run, 1), sentMaxTokens(run, 2)]).toEqual([4096, 100, 4096]);
  });

  it('is sent on the Anthropic-format path', async () => {
    const run = vi.fn().mockResolvedValue({ id: 'msg-1', type: 'message', role: 'assistant', content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn', usage: { input_tokens: 3, output_tokens: 1 } });
    const { cf } = provider({ defaultMaxTokens: 4096 }, run);
    await cf.generateResponse(request('anthropic/claude-opus-4.8'));
    expect(sentMaxTokens(run)).toBe(4096);
  });

  it('is sent on the streaming path', async () => {
    const stream = new ReadableStream({ start(controller) { controller.close(); } });
    const run = vi.fn().mockResolvedValue(stream);
    const { cf } = provider({ defaultMaxTokens: 4096 }, run);
    const body = await cf.streamResponse(request(LLAMA_FAST, { stream: true }));
    const reader = body.getReader();
    while (!(await reader.read()).done) { /* drain */ }
    expect(sentMaxTokens(run)).toBe(4096);
  });

  it('replaces the 512 sent on the Llama 3.2 Vision image path', async () => {
    const vision = '@cf/meta/llama-3.2-11b-vision-instruct';
    const image = { images: [{ data: 'aGk=', mimeType: 'image/png' }] };
    const unset = provider(undefined, vi.fn().mockResolvedValue({ response: 'a cat' }));
    await unset.cf.generateResponse(request(vision, image));
    expect(sentMaxTokens(unset.run)).toBe(512);
    const set = provider({ defaultMaxTokens: 2048 }, vi.fn().mockResolvedValue({ response: 'a cat' }));
    await set.cf.generateResponse(request(vision, image));
    expect(sentMaxTokens(set.run)).toBe(2048);
  });

  it('is used for pre-call cost estimates', () => {
    const unset = provider().cf.estimateCost(request(LLAMA_FAST));
    const set = provider({ defaultMaxTokens: 4096 }).cf.estimateCost(request(LLAMA_FAST));
    const explicit = provider({ defaultMaxTokens: 4096 }).cf.estimateCost(request(LLAMA_FAST, { maxTokens: 4096 }));
    expect(set).toBeGreaterThan(unset);
    expect(set).toBe(explicit);
  });

  it('passes through the factory, and an invalid value leaves Workers AI unconfigured', async () => {
    const run = vi.fn().mockResolvedValue(chatResult());
    const llm = new LLMProviders({ cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0, defaultMaxTokens: 3000 }, stateScope: 'client' });
    await llm.generateResponse(request(LLAMA_FAST));
    expect(sentMaxTokens(run)).toBe(3000);
    expect(() => new LLMProviders({ cloudflare: { ai: { run } as unknown as Ai, defaultMaxTokens: 0 }, stateScope: 'client' }))
      .toThrow('No valid providers configured');
  });

  it('is part of the response cache key when set, leaving default keys unchanged', async () => {
    const keys: Record<string, string> = {};
    for (const [label, defaultMaxTokens] of [['unset', undefined], ['set', 4096]] as const) {
      const cache = { get: vi.fn().mockResolvedValue(null), put: vi.fn().mockResolvedValue(undefined) };
      const run = vi.fn().mockResolvedValue(chatResult());
      const llm = new LLMProviders({
        cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0, ...(defaultMaxTokens ? { defaultMaxTokens } : {}) },
        stateScope: 'client',
        responseCache: cache,
      });
      await llm.generateResponse(request(LLAMA_FAST));
      keys[label] = cache.get.mock.calls[0][0];
    }
    const plain = { get: vi.fn().mockResolvedValue(null), put: vi.fn().mockResolvedValue(undefined) };
    const run = vi.fn().mockResolvedValue(chatResult());
    await new LLMProviders({ cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0, defaultMaxTokens: undefined }, stateScope: 'client', responseCache: plain })
      .generateResponse(request(LLAMA_FAST));
    expect(plain.get.mock.calls[0][0]).toBe(keys.unset);
    expect(keys.set).not.toBe(keys.unset);
  });
});

describe('warning about a small model default', () => {
  it('warns once per model when no limit is set and the documented default is small', async () => {
    const { cf, log } = provider();
    await cf.generateResponse(request(LLAMA_FAST));
    await cf.generateResponse(request(LLAMA_FAST));
    expect(log.warn).toHaveBeenCalledTimes(1);
    const message = String(log.warn.mock.calls[0][0]);
    expect(message).toContain(LLAMA_FAST);
    expect(message).toContain('256 tokens');
    expect(message).toContain('defaultMaxTokens');
    expect(message).not.toContain('Summarise');
    await cf.generateResponse(request('@cf/openai/gpt-oss-20b'));
    expect(log.warn).toHaveBeenCalledTimes(2);
  });

  it("doesn't warn when maxTokens or defaultMaxTokens is set, or the default is large or undocumented", async () => {
    const { cf, log } = provider();
    await cf.generateResponse(request(LLAMA_FAST, { maxTokens: 100 }));
    await cf.generateResponse(request(QWEN3_30B));
    await cf.generateResponse(request(GLM_5_3));
    expect(cloudflareOutputLimits(QWEN3_30B)!.defaultMaxTokens!).toBeGreaterThan(SMALL_DEFAULT_MAX_TOKENS);
    const configured = provider({ defaultMaxTokens: 4096 });
    await configured.cf.generateResponse(request(LLAMA_FAST));
    expect(log.warn).not.toHaveBeenCalled();
    expect(configured.log.warn).not.toHaveBeenCalled();
  });

  it('warns on the streaming path too', async () => {
    const stream = new ReadableStream({ start(controller) { controller.close(); } });
    const { cf, log } = provider(undefined, vi.fn().mockResolvedValue(stream));
    const body = await cf.streamResponse(request(LLAMA_FAST, { stream: true }));
    const reader = body.getReader();
    while (!(await reader.read()).done) { /* drain */ }
    expect(log.warn).toHaveBeenCalledTimes(1);
  });
});
