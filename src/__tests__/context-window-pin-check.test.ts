/**
 * #136: opt-in context-window check for pinned models (contextWindowPinCheck: true).
 * A pinned catalog model whose window can't hold the request's input is never
 * called: with allowFallback: false the request throws ContextWindowExceededError
 * before any call; otherwise the pinned provider is skipped as CONTEXT_WINDOW and the
 * request falls back as usual. Off by default.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MODEL_CATALOG } from '../model-catalog';
import { ContextWindowExceededError, LLMProviders, NoRouteError, TokenLimitError } from '../index';
import type { ProviderFactoryConfig } from '../factory';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { defaultExhaustionRegistry } from '../utils/exhaustion';
import type { LLMRequest } from '../types';

const PINNED = '@cf/meta/llama-3.3-70b-instruct-fp8-fast';
const NVIDIA_DEFAULT = 'deepseek-ai/deepseek-v4-flash';

function counted(tokens: number, extra: Partial<LLMRequest> = {}): LLMRequest {
  return { messages: [{ role: 'user', content: 'Summarise the attached document.' }], maxTokens: 10, inputTokenCount: tokens, model: PINNED, ...extra };
}

function chatResult(content: string) {
  return {
    id: 'chatcmpl-test',
    choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
  };
}

function compatibleResponse(init?: RequestInit) {
  const model = JSON.parse(String(init?.body ?? '{}')).model ?? 'unknown';
  return new Response(JSON.stringify({
    id: 'http-1', object: 'chat.completion', created: 1, model,
    choices: [{ index: 0, message: { role: 'assistant', content: 'from http' }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
  }), { status: 200, headers: { 'Content-Type': 'application/json' } });
}

function sentModels(fetchMock: ReturnType<typeof vi.fn>): string[] {
  return fetchMock.mock.calls.map(call => JSON.parse(String(call[1].body)).model);
}

function setup(config: Partial<ProviderFactoryConfig> = {}) {
  const run = vi.fn().mockResolvedValue(chatResult('from cloudflare'));
  const fetchMock = vi.fn().mockImplementation(async (_url: string, init?: RequestInit) => compatibleResponse(init));
  vi.stubGlobal('fetch', fetchMock);
  const llm = new LLMProviders({
    cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 },
    nvidia: { apiKey: 'test', maxRetries: 0 },
    preferredProvider: 'cloudflare',
    stateScope: 'client',
    ...config,
  });
  return { llm, run, fetchMock };
}

type Plan = { chain: string[]; providerModels: Map<string, string>; contextSkips: Set<string> };
function planOf(llm: LLMProviders, request: LLMRequest): Plan {
  return (llm as unknown as { factory: { buildProviderPlan(request: LLMRequest): Plan } }).factory.buildProviderPlan(request);
}

async function caught(promise: Promise<unknown>): Promise<unknown> {
  try { await promise; } catch (error) { return error; }
  throw new Error('expected a rejection');
}

beforeEach(() => {
  defaultCircuitBreakerManager.resetAll();
  defaultExhaustionRegistry.reset();
  vi.spyOn(console, 'log').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('test assumptions', () => {
  it("the pinned model's catalog window is 24,000 tokens, and NVIDIA's default holds 30k", () => {
    const window = (provider: string, model: string) =>
      MODEL_CATALOG.find(entry => entry.provider === provider && entry.model === model)!.capabilities.maxContextLength;
    expect(window('cloudflare', PINNED)).toBe(24_000);
    expect(window('nvidia', NVIDIA_DEFAULT)).toBeGreaterThan(30_000);
  });
});

describe('ContextWindowExceededError', () => {
  it('is a NoRouteError with code NO_ROUTE and structured details, not a TokenLimitError', () => {
    const error = new ContextWindowExceededError({
      modelProvider: 'cloudflare', model: PINNED, contextWindow: 24_000, inputTokens: 30_000, inputTokensSource: 'caller',
    });
    expect(error).toBeInstanceOf(NoRouteError);
    expect(error).not.toBeInstanceOf(TokenLimitError);
    expect(error.name).toBe('ContextWindowExceededError');
    expect(error.code).toBe('NO_ROUTE');
    expect(error.provider).toBe('factory');
    expect(error.retryable).toBe(false);
    expect(error).toMatchObject({ modelProvider: 'cloudflare', model: PINNED, contextWindow: 24_000, inputTokens: 30_000, inputTokensSource: 'caller' });
    expect(error.message).toBe(
      `Pinned model '${PINNED}' (cloudflare) has a 24000-token context window, which can't hold the request's 30000 input tokens, and allowFallback is false`
    );
    const estimated = new ContextWindowExceededError({
      modelProvider: 'cloudflare', model: PINNED, contextWindow: 24_000, inputTokens: 30_000, inputTokensSource: 'estimate',
    });
    expect(estimated.message).toContain("can't hold the request's 30000 input tokens (estimated), and allowFallback is false");
  });
});

describe('factory with contextWindowPinCheck', () => {
  it.each([undefined, false])('is off by default (%s): the pinned model is called', async contextWindowPinCheck => {
    const { llm, run, fetchMock } = setup({ contextWindowPinCheck });
    const response = await llm.generateResponse(counted(30_000, { allowFallback: false }));
    expect(response.provider).toBe('cloudflare');
    expect(run.mock.calls[0][0]).toBe(PINNED);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('control: contextWindowRouting alone never skips a pinned model', async () => {
    const { llm, run } = setup({ contextWindowRouting: true });
    const response = await llm.generateResponse(counted(30_000));
    expect(response.provider).toBe('cloudflare');
    expect(run.mock.calls[0][0]).toBe(PINNED);
  });

  it('with allowFallback: false, throws ContextWindowExceededError before any call', async () => {
    const { llm, run, fetchMock } = setup({ contextWindowPinCheck: true });
    const error = await caught(llm.generateResponse(counted(30_000, { allowFallback: false })));
    expect(error).toBeInstanceOf(ContextWindowExceededError);
    expect(error).toMatchObject({ modelProvider: 'cloudflare', model: PINNED, contextWindow: 24_000, inputTokens: 30_000, inputTokensSource: 'caller' });
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('reports an estimated count, and never echoes the prompt', async () => {
    const { llm, run } = setup({ contextWindowPinCheck: true });
    const secret = 'family-secret ';
    const content = secret.repeat(Math.ceil(100_000 / secret.length));
    const error = await caught(llm.generateResponse({ messages: [{ role: 'user', content }], model: PINNED, allowFallback: false }));
    expect(error).toBeInstanceOf(ContextWindowExceededError);
    const exceeded = error as ContextWindowExceededError;
    expect(exceeded.inputTokensSource).toBe('estimate');
    expect(exceeded.inputTokens).toBeGreaterThan(24_000);
    expect(exceeded.message).toContain('input tokens (estimated)');
    expect(exceeded.message).not.toContain('family-secret');
    expect(run).not.toHaveBeenCalled();
  });

  it('accepts an input of exactly the window, and nothing past it', async () => {
    const { llm, run } = setup({ contextWindowPinCheck: true });
    const response = await llm.generateResponse(counted(24_000, { allowFallback: false }));
    expect(response.model).toBe(PINNED);
    expect(run).toHaveBeenCalledTimes(1);
    await expect(llm.generateResponse(counted(24_001, { allowFallback: false }))).rejects.toBeInstanceOf(ContextWindowExceededError);
    expect(run).toHaveBeenCalledTimes(1);
  });

  it('with fallback allowed, skips the pinned provider and falls back, marking the skip', async () => {
    const { llm, run, fetchMock } = setup({ contextWindowPinCheck: true });
    const response = await llm.generateResponse(counted(30_000));
    expect(response.provider).toBe('nvidia');
    expect(sentModels(fetchMock)).toEqual([NVIDIA_DEFAULT]);
    expect(run).not.toHaveBeenCalled();
    expect(response.fallback?.failedAttempts).toEqual([{ provider: 'cloudflare', model: PINNED, errorCode: 'CONTEXT_WINDOW' }]);
  });

  it('ends without a call when the pinned provider is the only one configured', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('from cloudflare'));
    const llm = new LLMProviders({ cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 }, stateScope: 'client', contextWindowPinCheck: true });
    const error = await caught(llm.generateResponse(counted(30_000)));
    expect((error as Error).message).toContain('cloudflare: CONTEXT_WINDOW');
    expect(run).not.toHaveBeenCalled();
  });

  it('calls a pinned model that fits', async () => {
    const { llm, run } = setup({ contextWindowPinCheck: true });
    const response = await llm.generateResponse(counted(1_000, { allowFallback: false }));
    expect(response.model).toBe(PINNED);
    expect(run).toHaveBeenCalledTimes(1);
  });

  it("doesn't check a pinned model outside the catalog", () => {
    const { llm } = setup({ contextWindowPinCheck: true });
    const plan = planOf(llm, counted(5_000_000, { model: '@cf/example/not-in-catalog' }));
    expect(plan.providerModels.get('cloudflare')).toBe('@cf/example/not-in-catalog');
    expect(plan.contextSkips.size).toBe(0);
  });

  it("doesn't touch unpinned requests", () => {
    const { llm } = setup({ contextWindowPinCheck: true });
    const plan = planOf(llm, counted(30_000, { model: undefined }));
    expect(plan.contextSkips.size).toBe(0);
  });

  it('is reflected by client getRoutingInfo', () => {
    const { llm } = setup({ contextWindowPinCheck: true });
    expect(llm.getRoutingInfo(counted(30_000)).provider).toBe('nvidia');
    expect(() => llm.getRoutingInfo(counted(30_000, { allowFallback: false }))).toThrow(ContextWindowExceededError);
    expect(setup().llm.getRoutingInfo(counted(30_000)).provider).toBe('cloudflare');
  });

  it('applies on the managed streaming path', async () => {
    const { llm, run } = setup({ contextWindowPinCheck: true });
    const streamBody = () => new Response(
      'data: {"id":"s1","choices":[{"index":0,"delta":{"content":"streamed"},"finish_reason":null}]}\n\n'
      + 'data: {"id":"s1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\n'
      + 'data: [DONE]\n\n',
      { status: 200, headers: { 'Content-Type': 'text/event-stream' } },
    );
    const fetchMock = vi.fn().mockImplementation(async () => streamBody());
    vi.stubGlobal('fetch', fetchMock);
    const managed = await llm.generateManagedResponseStream(counted(30_000, { stream: true }));
    const reader = managed.stream.getReader();
    while (!(await reader.read()).done) { /* drain */ }
    const completion = await managed.completion;
    expect(completion.outcome).toBe('success');
    expect(completion.provider).toBe('nvidia');
    expect(run).not.toHaveBeenCalled();

    await expect(llm.generateManagedResponseStream(counted(30_000, { stream: true, allowFallback: false })))
      .rejects.toBeInstanceOf(ContextWindowExceededError);
    expect(run).not.toHaveBeenCalled();
  });

  it('includes the option in the response cache key, leaving default keys unchanged', async () => {
    const keys: Record<string, string> = {};
    for (const [label, contextWindowPinCheck] of [['unset', undefined], ['false', false], ['true', true]] as const) {
      const cache = { get: vi.fn().mockResolvedValue(null), put: vi.fn().mockResolvedValue(undefined) };
      const { llm } = setup({ contextWindowPinCheck, responseCache: cache });
      await llm.generateResponse(counted(10));
      keys[label] = cache.get.mock.calls[0][0];
    }
    expect(keys.unset).toBe(keys.false);
    expect(keys.true).not.toBe(keys.unset);
  });

  it('turning it on with updateConfig takes effect', async () => {
    const { llm, run } = setup();
    llm.updateConfig({ contextWindowPinCheck: true });
    await expect(llm.generateResponse(counted(30_000, { allowFallback: false }))).rejects.toBeInstanceOf(ContextWindowExceededError);
    expect(run).not.toHaveBeenCalled();
  });
});
