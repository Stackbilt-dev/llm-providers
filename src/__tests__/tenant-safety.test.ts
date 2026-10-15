import { beforeEach, describe, expect, it, vi } from 'vitest';
import { LLMProviderFactory } from '../factory';
import { QuotaDeniedError, QuotaExceededError } from '../index';
import { CloudflareProvider } from '../providers/cloudflare';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { defaultExhaustionRegistry } from '../utils/exhaustion';
import { deriveResponseCacheKey } from '../utils/response-cache';
import type { LLMRequest, LLMResponse } from '../types';

const request: LLMRequest = {
  messages: [{ role: 'user', content: 'Private prompt' }],
  model: '@cf/meta/llama-3.1-8b-instruct', tenantId: 'tenant-a',
};
const response: LLMResponse = {
  message: 'fresh answer', model: request.model!, provider: 'cloudflare', responseTime: 1,
  usage: { inputTokens: 10, outputTokens: 2, totalTokens: 12, cost: 0.001 },
};

function setup(extra: Partial<ConstructorParameters<typeof LLMProviderFactory>[0]> = {}) {
  const entries = new Map<string, string>();
  const cache = {
    get: vi.fn(async (key: string) => entries.get(key) ?? null),
    put: vi.fn(async (key: string, value: string) => { entries.set(key, value); }),
  };
  const factory = new LLMProviderFactory({
    cloudflare: { ai: { run: vi.fn() } as unknown as Ai },
    responseCache: cache, ...extra,
  });
  return { factory, cache };
}

beforeEach(() => {
  vi.restoreAllMocks();
  defaultCircuitBreakerManager.resetAll();
  defaultExhaustionRegistry.reset();
  vi.spyOn(CloudflareProvider.prototype, 'generateResponse').mockResolvedValue(response);
});

describe('response cache isolation', () => {
  it.each([
    { tenantId: 'tenant-b' }, { systemPrompt: 'new system' },
    { tools: [{ type: 'function', function: { name: 'tool', description: 'desc', parameters: { type: 'object', properties: {} } } }] },
    { builtInTools: [{ type: 'web_search' }] }, { images: [{ data: 'abc', mimeType: 'image/png' }] },
    { response_format: { type: 'json_object' } }, { seed: 1 }, { lora: 'adapter' },
    { reasoning: { effort: 'high' } }, { metadata: { useCase: 'code_draft' } },
    { topP: 0.5 }, { frequencyPenalty: 1 }, { prediction: 'prediction' },
    { toolChoice: 'required' }, { temperature: 0 }, { maxTokens: 10 },
    { cache: { key: 'explicit' } }, { gatewayMetadata: { customMetadata: { route: 'alternate' } } },
  ] as Partial<LLMRequest>[])('separates semantic input %j', async change => {
    expect(await deriveResponseCacheKey({ ...request, ...change }, 'app', {}))
      .not.toBe(await deriveResponseCacheKey(request, 'app', {}));
  });

  it('uses stable, opaque keys while excluding tracing identity', async () => {
    const first = await deriveResponseCacheKey({ ...request, metadata: { a: 1, b: 2 } }, 'app', {});
    const second = await deriveResponseCacheKey({
      ...request, requestId: 'another-call', gatewayMetadata: { requestId: 'trace' },
      metadata: { b: 2, a: 1 },
    }, 'app', {});
    expect(first).toBe(second);
    expect(first).toMatch(/^llmp:v2:[a-f0-9]{64}$/);
    expect(first).not.toContain('Private prompt');
  });

  it('scopes explicit keys by tenant, namespace, and routing policy', async () => {
    const input = { ...request, cache: { key: 'shared' } };
    const key = await deriveResponseCacheKey(input, 'app', { provider: 'a' });
    for (const candidate of [
      deriveResponseCacheKey({ ...input, tenantId: 'b' }, 'app', { provider: 'a' }),
      deriveResponseCacheKey(input, 'another-app', { provider: 'a' }),
      deriveResponseCacheKey(input, 'app', { provider: 'b' }),
    ]) expect(await candidate).not.toBe(key);
  });

  it.each(['off', 'provider-prefix'] as const)('bypasses reads and writes for %s', async strategy => {
    const { factory, cache } = setup();
    await factory.generateResponse({ ...request, cache: { strategy } });
    expect(cache.get).not.toHaveBeenCalled();
    expect(cache.put).not.toHaveBeenCalled();
  });

  it.each([undefined, 'response', 'both'] as const)('caches for strategy %s and marks historical usage', async strategy => {
    const { factory } = setup();
    const input = { ...request, cache: { strategy } };
    await factory.generateResponse(input);
    const hit = await factory.generateResponse(input);
    expect(CloudflareProvider.prototype.generateResponse).toHaveBeenCalledTimes(1);
    expect(hit.cache?.factory).toMatchObject({ status: 'hit', usageKind: 'historical', currentChargeUsd: 0 });
  });

  it('does not reuse another tenant or system prompt completion', async () => {
    const { factory } = setup();
    await factory.generateResponse(request);
    await factory.generateResponse({ ...request, tenantId: 'b' });
    await factory.generateResponse({ ...request, systemPrompt: 'new instructions' });
    expect(CloudflareProvider.prototype.generateResponse).toHaveBeenCalledTimes(3);
  });

  it('separates factory namespaces and transport/account options', async () => {
    const { factory, cache } = setup({ responseCacheNamespace: 'app-a' });
    await factory.generateResponse(request);
    for (const extra of [
      { responseCacheNamespace: 'app-b' },
      { cloudflare: { ai: {} as Ai, accountId: 'different' } },
      { cloudflare: { ai: {} as Ai, baseUrl: 'https://another.example' } },
    ]) {
      const other = setup({ responseCacheNamespace: 'app-a', ...extra, responseCache: cache });
      await other.factory.generateResponse(request);
    }
    expect(CloudflareProvider.prototype.generateResponse).toHaveBeenCalledTimes(4);
    expect(new Set(cache.get.mock.calls.map(([key]) => key)).size).toBe(4);
  });

  it.each([null, {}, { ...response, usage: { cost: 0 } }, { ...response, toolCalls: [{}] }])(
    'bypasses corrupt envelopes %j', async value => {
      const { factory, cache } = setup();
      cache.get.mockResolvedValue(JSON.stringify(value));
      expect((await factory.generateResponse(request)).message).toBe('fresh answer');
      expect(CloudflareProvider.prototype.generateResponse).toHaveBeenCalledTimes(1);
    }
  );

  it.each([BigInt(1), NaN, new Date(), () => 1])('bypasses non-JSON metadata %s', async value => {
    const { factory, cache } = setup();
    await factory.generateResponse({ ...request, metadata: { value } });
    expect(cache.get).not.toHaveBeenCalled();
    expect(cache.put).not.toHaveBeenCalled();
  });

  it('bypasses circular metadata safely', async () => {
    const { factory, cache } = setup();
    const metadata: Record<string, unknown> = {};
    metadata.self = metadata;
    await factory.generateResponse({ ...request, metadata });
    expect(cache.get).not.toHaveBeenCalled();
    expect(cache.put).not.toHaveBeenCalled();
  });
});

describe('local quota denial', () => {
  it.each(['response', 'stream', 'image'] as const)('terminates %s without fallback or poisoning tenants', async mode => {
    const quotaHook = {
      check: vi.fn(async ({ tenantId }: { tenantId?: string }) => ({ allowed: tenantId !== 'tenant-a', reason: 'denied' })),
      record: vi.fn(async () => {}),
    };
    const { factory } = setup({
      responseCache: undefined, quotaHook, openai: { apiKey: 'fake' },
      fallbackRules: [{ condition: 'error', fallbackProvider: 'openai' }],
    });
    const operation = mode === 'stream' ? factory.generateResponseStream(request)
      : mode === 'image' ? factory.analyzeImage({
        image: { data: 'aGVsbG8=', mimeType: 'image/png' }, prompt: 'read',
        tenantId: request.tenantId, model: '@cf/meta/llama-3.2-11b-vision-instruct',
      }) : factory.generateResponse(request);
    await expect(operation).rejects.toBeInstanceOf(QuotaDeniedError);
    expect(quotaHook.check).toHaveBeenCalledTimes(1);
    expect(CloudflareProvider.prototype.generateResponse).not.toHaveBeenCalled();
    expect(defaultExhaustionRegistry.getExhaustedProviders()).toEqual([]);
    await expect(factory.generateResponse({ ...request, tenantId: 'tenant-b' })).resolves.toMatchObject(response);
  });

  it('gates cached access with zero estimated new spend', async () => {
    const quotaHook = { check: vi.fn().mockResolvedValue({ allowed: true }), record: vi.fn(async () => {}) };
    const { factory } = setup({ quotaHook });
    await factory.generateResponse(request);
    quotaHook.check.mockResolvedValue({ allowed: false });
    await expect(factory.generateResponse(request)).rejects.toBeInstanceOf(QuotaDeniedError);
    expect(quotaHook.check).toHaveBeenLastCalledWith(expect.objectContaining({ estimatedCost: 0 }));
    expect(CloudflareProvider.prototype.generateResponse).toHaveBeenCalledTimes(1);
    expect(quotaHook.record).toHaveBeenCalledTimes(1);
    expect(defaultExhaustionRegistry.isExhausted('cloudflare')).toBe(false);
  });

  it.each(['closed', 'open'] as const)('honors fail-%s on quota service errors', async quotaFailPolicy => {
    const { factory } = setup({ quotaFailPolicy, quotaHook: {
      check: vi.fn().mockRejectedValue(new Error('service unavailable')), record: vi.fn(async () => {}),
    } });
    const operation = factory.generateResponse(request);
    if (quotaFailPolicy === 'closed') await expect(operation).rejects.toBeInstanceOf(QuotaDeniedError);
    else await expect(operation).resolves.toMatchObject(response);
    expect(defaultExhaustionRegistry.isExhausted('cloudflare')).toBe(false);
  });

  it.each(['onQuotaCheck', 'onQuotaDenied'] as const)('cannot fail open on a throwing %s observer', async hook => {
    const { factory } = setup({ quotaFailPolicy: 'open',
      hooks: { [hook]: () => { throw new Error('observer failed'); } },
      quotaHook: { check: vi.fn().mockResolvedValue({ allowed: false }), record: vi.fn(async () => {}) },
    });
    await expect(factory.generateResponse(request)).rejects.toBeInstanceOf(QuotaDeniedError);
    expect(CloudflareProvider.prototype.generateResponse).not.toHaveBeenCalled();
    expect(defaultExhaustionRegistry.isExhausted('cloudflare')).toBe(false);
  });

  it('preserves provider exhaustion for vendor quota errors', async () => {
    const { factory } = setup({ responseCache: undefined });
    vi.mocked(CloudflareProvider.prototype.generateResponse).mockRejectedValue(new QuotaExceededError('cloudflare'));
    await expect(factory.generateResponse(request)).rejects.toBeInstanceOf(QuotaExceededError);
    expect(defaultExhaustionRegistry.isExhausted('cloudflare')).toBe(true);
  });
});
