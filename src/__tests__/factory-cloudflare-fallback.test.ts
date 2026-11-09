import { beforeEach, describe, expect, it, vi } from 'vitest';
import { LLMProviderFactory } from '../factory';
import { StructuredOutputUnsupportedError } from '../errors';
import { CloudflareProvider } from '../providers/cloudflare';
import type { LLMRequest } from '../types';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { defaultExhaustionRegistry } from '../utils/exhaustion';

const fallback = vi.hoisted(() => ({
  generateResponse: vi.fn(),
  streamResponse: vi.fn(),
}));

vi.mock('../providers/openai', () => ({
  OpenAIProvider: vi.fn().mockImplementation(() => ({
    name: 'openai', models: ['gpt-4o-mini'], supportsStreaming: true,
    validateConfig: () => true,
    estimateCost: () => 0,
    getMetrics: () => ({ requestCount: 0, successCount: 0, errorCount: 0, averageLatency: 0, totalCost: 0, rateLimitHits: 0, lastUsed: 0 }),
    ...fallback,
  })),
}));

describe('factory fallback from the Cloudflare adapter', () => {
  const run = vi.fn();
  const request: LLMRequest = {
    model: '@cf/meta/llama-3.1-8b-instruct-fp8',
    messages: [{ role: 'user', content: 'Return JSON.' }],
    response_format: { type: 'json_schema', json_schema: { name: 'status', schema: { type: 'object' } } },
  };
  let factory: LLMProviderFactory;

  beforeEach(() => {
    vi.clearAllMocks();
    defaultCircuitBreakerManager.resetAll();
    defaultExhaustionRegistry.reset();
    fallback.generateResponse.mockResolvedValue({
      message: '{}', provider: 'openai', model: 'gpt-4o-mini', responseTime: 1,
      usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2, cost: 0 },
    });
    fallback.streamResponse.mockResolvedValue(new ReadableStream<string>({
      start(controller) { controller.enqueue('fallback'); controller.close(); },
    }));
    factory = new LLMProviderFactory({
      cloudflare: { ai: { run } as unknown as Ai },
      openai: { apiKey: 'test' },
      enableCircuitBreaker: false,
    });
  });

  it('falls back for an unsupported schema without invoking Workers AI', async () => {
    const adapter = vi.spyOn(CloudflareProvider.prototype, 'generateResponse');
    const response = await factory.generateResponse(request);
    expect(adapter).toHaveBeenCalledOnce();
    expect(response.provider).toBe('openai');
    expect(fallback.generateResponse).toHaveBeenCalledOnce();
    expect(run).not.toHaveBeenCalled();
  });

  it('falls back for a streaming schema without invoking Workers AI', async () => {
    const adapter = vi.spyOn(CloudflareProvider.prototype, 'streamResponse');
    const stream = await factory.generateResponseStream(request);
    expect(adapter).toHaveBeenCalledOnce();
    expect((await stream.getReader().read()).value).toBe('fallback');
    expect(fallback.streamResponse).toHaveBeenCalledOnce();
    expect(run).not.toHaveBeenCalled();
  });

  it('surfaces the adapter error when fallback is disabled', async () => {
    const adapter = vi.spyOn(CloudflareProvider.prototype, 'generateResponse');
    await expect(factory.generateResponse({ ...request, allowFallback: false }))
      .rejects.toBeInstanceOf(StructuredOutputUnsupportedError);
    expect(adapter).toHaveBeenCalledOnce();
    expect(fallback.generateResponse).not.toHaveBeenCalled();
    expect(run).not.toHaveBeenCalled();
  });
});
