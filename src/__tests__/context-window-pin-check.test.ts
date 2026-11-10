/**
 * #136: opt-in context-window check for pinned models (contextWindowPinCheck: true).
 * A pinned catalog model whose window can't hold the request's input is never
 * called: with allowFallback: false the request throws ContextWindowExceededError
 * before any call, and so it does when no other provider is left in the plan;
 * otherwise the pinned provider is skipped as CONTEXT_WINDOW and the request falls
 * back as usual, and a caller's fallback rule revives it only with a different model.
 * Off by default.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MODEL_CATALOG } from '../model-catalog';
import { ContextWindowExceededError, LLMProviders, NoRouteError, TokenLimitError } from '../index';
import type { ProviderFactoryConfig } from '../factory';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { defaultExhaustionRegistry } from '../utils/exhaustion';
import type { LLMRequest, Tool } from '../types';

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

const lookupTool: Tool = {
  type: 'function',
  function: { name: 'lookup', description: 'Look something up', parameters: { type: 'object', properties: {} } },
};

function compatibleResponse(init?: RequestInit, message: Record<string, unknown> = { role: 'assistant', content: 'from http' }, finish = 'stop') {
  const model = JSON.parse(String(init?.body ?? '{}')).model ?? 'unknown';
  return new Response(JSON.stringify({
    id: 'http-1', object: 'chat.completion', created: 1, model,
    choices: [{ index: 0, message, finish_reason: finish }],
    usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
  }), { status: 200, headers: { 'Content-Type': 'application/json' } });
}

function sentModels(fetchMock: ReturnType<typeof vi.fn>): string[] {
  return fetchMock.mock.calls.map(call => JSON.parse(String(call[1].body)).model);
}

function setup(config: Partial<ProviderFactoryConfig> = {}, fetchMock = vi.fn().mockImplementation(async (_url: string, init?: RequestInit) => compatibleResponse(init))) {
  const run = vi.fn().mockResolvedValue(chatResult('from cloudflare'));
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

type Plan = { chain: string[]; providerModels: Map<string, string>; contextSkips: Set<string>; pinContextSkip?: string };
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
      modelProvider: 'cloudflare', model: PINNED, contextWindow: 24_000, inputTokens: 30_000, inputTokensSource: 'caller', reason: 'allowFallback-false',
    });
    expect(error).toBeInstanceOf(NoRouteError);
    expect(error).not.toBeInstanceOf(TokenLimitError);
    expect(error.name).toBe('ContextWindowExceededError');
    expect(error.code).toBe('NO_ROUTE');
    expect(error.provider).toBe('factory');
    expect(error.retryable).toBe(false);
    expect(error).toMatchObject({ modelProvider: 'cloudflare', model: PINNED, contextWindow: 24_000, inputTokens: 30_000, inputTokensSource: 'caller', reason: 'allowFallback-false' });
    expect(error.message).toBe(
      `Pinned model '${PINNED}' (cloudflare) has a 24000-token context window, which can't hold the request's 30000 input tokens, and allowFallback is false`
    );
    const estimated = new ContextWindowExceededError({
      modelProvider: 'cloudflare', model: PINNED, contextWindow: 24_000, inputTokens: 30_000, inputTokensSource: 'estimate', reason: 'allowFallback-false',
    });
    expect(estimated.message).toContain("can't hold the request's 30000 input tokens (estimated), and allowFallback is false");
    const mixed = new ContextWindowExceededError({
      modelProvider: 'cloudflare', model: PINNED, contextWindow: 24_000, inputTokens: 30_000, inputTokensSource: 'mixed', reason: 'no-other-provider',
    });
    expect(mixed.reason).toBe('no-other-provider');
    expect(mixed.message).toContain("can't hold the request's 30000 input tokens (partly estimated), and no other provider can serve the request");
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
    expect(error).toMatchObject({ modelProvider: 'cloudflare', model: PINNED, contextWindow: 24_000, inputTokens: 30_000, inputTokensSource: 'caller', reason: 'allowFallback-false' });
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

  it('throws ContextWindowExceededError when the pinned provider is the only one configured', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('from cloudflare'));
    const llm = new LLMProviders({ cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 }, stateScope: 'client', contextWindowPinCheck: true });
    const error = await caught(llm.generateResponse(counted(30_000)));
    expect(error).toBeInstanceOf(ContextWindowExceededError);
    expect(error).toMatchObject({ reason: 'no-other-provider', inputTokensSource: 'caller', model: PINNED });
    expect((error as Error).message).toContain('and no other provider can serve the request');
    expect(run).not.toHaveBeenCalled();
  });

  it('throws ContextWindowExceededError when reasoning.strict leaves only the pinned provider', async () => {
    const { llm, run, fetchMock } = setup({ contextWindowPinCheck: true });
    const strict = { reasoning: { effort: 'low' as const, strict: true } };
    const pinned = '@cf/zai-org/glm-4.7-flash';
    const window = MODEL_CATALOG.find(entry => entry.provider === 'cloudflare' && entry.model === pinned)!.capabilities.maxContextLength;
    const error = await caught(llm.generateResponse(counted(window + 1, { ...strict, model: pinned })));
    expect(error).toBeInstanceOf(ContextWindowExceededError);
    expect(error).toMatchObject({ reason: 'no-other-provider', model: pinned });
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("a caller's fallback rule without a different model never brings the pinned model back", async () => {
    const failing = vi.fn().mockResolvedValue(new Response('{"error":"boom"}', { status: 500, headers: { 'Content-Type': 'application/json' } }));
    for (const fallbackModel of [undefined, PINNED]) {
      const { llm, run } = setup({
        contextWindowPinCheck: true,
        fallbackRules: [{ condition: 'error', fallbackProvider: 'cloudflare', ...(fallbackModel ? { fallbackModel } : {}) }],
      }, failing);
      await caught(llm.generateResponse(counted(30_000)));
      expect(run).not.toHaveBeenCalled();
    }
  });

  it("a caller's fallback rule with a different model brings the pinned provider back on that model", async () => {
    const failing = vi.fn().mockResolvedValue(new Response('{"error":"boom"}', { status: 500, headers: { 'Content-Type': 'application/json' } }));
    const { llm, run } = setup({
      contextWindowPinCheck: true,
      fallbackRules: [{ condition: 'error', fallbackProvider: 'cloudflare', fallbackModel: '@cf/zai-org/glm-5.3-flash' }],
    }, failing);
    const response = await llm.generateResponse(counted(30_000));
    expect(response.provider).toBe('cloudflare');
    expect(run.mock.calls.map(call => call[0])).toEqual(['@cf/zai-org/glm-5.3-flash']);
  });

  it('throws ContextWindowExceededError when an image request leaves only the pinned provider', async () => {
    // NVIDIA isn't planned for image input here, so only the pin's provider remains.
    const { llm, run } = setup({ contextWindowPinCheck: true });
    const request = counted(30_000, { images: [{ data: 'aGk=', mimeType: 'image/png' }] });
    const error = await caught(llm.generateResponse(request));
    expect(error).toBeInstanceOf(ContextWindowExceededError);
    expect(error).toMatchObject({ reason: 'no-other-provider', model: PINNED });
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

  it('checks each tool-loop turn, reporting a partly estimated count', async () => {
    const toolCall = { role: 'assistant', content: null,
      tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'lookup', arguments: '{}' } }] };
    const fetchMock = vi.fn()
      .mockImplementationOnce(async (_url: string, init?: RequestInit) => compatibleResponse(init, toolCall, 'tool_calls'))
      .mockImplementation(async (_url: string, init?: RequestInit) => compatibleResponse(init));
    vi.stubGlobal('fetch', fetchMock);
    const llm = new LLMProviders({ nvidia: { apiKey: 'test', maxRetries: 0 }, stateScope: 'client', contextWindowPinCheck: true });
    const pinned = MODEL_CATALOG.find(entry => entry.provider === 'nvidia' && entry.lifecycle === 'active' && entry.capabilities.supportsTools)!;
    const window = pinned.capabilities.maxContextLength;
    const execute = vi.fn(async () => 'x'.repeat(8_000));
    const error = await caught(llm.generateResponseWithTools(
      { messages: [{ role: 'user', content: 'Summarise.' }], tools: [lookupTool], maxTokens: 10, model: pinned.model, inputTokenCount: window - 1_000, allowFallback: false },
      { execute },
    ));
    expect(error).toBeInstanceOf(ContextWindowExceededError);
    expect(error).toMatchObject({ inputTokensSource: 'mixed', reason: 'allowFallback-false', model: pinned.model });
    expect((error as ContextWindowExceededError).inputTokens).toBeGreaterThan(window);
    expect((error as Error).message).toContain('(partly estimated)');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(execute).toHaveBeenCalledTimes(1);
  });

  it("reports classify's added system prompt as a partly estimated count", async () => {
    const { llm, run } = setup({ contextWindowPinCheck: true });
    const error = await caught(llm.classify(counted(24_000, { allowFallback: false })));
    expect(error).toBeInstanceOf(ContextWindowExceededError);
    expect(error).toMatchObject({ inputTokensSource: 'mixed' });
    expect(run).not.toHaveBeenCalled();
  });

  it('keeps the partly estimated mark when a corrected model ID is resolved', async () => {
    const llm = new LLMProviders({ anthropic: { apiKey: 'test', maxRetries: 0 }, stateScope: 'client', contextWindowPinCheck: true });
    const window = MODEL_CATALOG.find(entry => entry.provider === 'anthropic' && entry.model === 'claude-opus-4-6')!.capabilities.maxContextLength;
    const error = await caught(llm.classify({
      messages: [{ role: 'user', content: 'Classify.' }], model: 'claude-opus-4-6-20250618', inputTokenCount: window + 1, allowFallback: false,
    }));
    expect(error).toBeInstanceOf(ContextWindowExceededError);
    expect(error).toMatchObject({ model: 'claude-opus-4-6', inputTokensSource: 'mixed' });
  });

  it('keeps response caching for tool-loop turns and classify, whose counts are marked', async () => {
    const cache = { get: vi.fn().mockResolvedValue(null), put: vi.fn().mockResolvedValue(undefined) };
    const toolCall = { role: 'assistant', content: null,
      tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'lookup', arguments: '{}' } }] };
    const fetchMock = vi.fn()
      .mockImplementationOnce(async (_url: string, init?: RequestInit) => compatibleResponse(init, toolCall, 'tool_calls'))
      .mockImplementation(async (_url: string, init?: RequestInit) => compatibleResponse(init, { role: 'assistant', content: '{"label":"a","confidence":1}' }));
    vi.stubGlobal('fetch', fetchMock);
    const llm = new LLMProviders({ nvidia: { apiKey: 'test', maxRetries: 0 }, stateScope: 'client', contextWindowPinCheck: true, responseCache: cache });
    await llm.generateResponseWithTools(
      { messages: [{ role: 'user', content: 'Summarise.' }], tools: [lookupTool], maxTokens: 10, inputTokenCount: 100 },
      { execute: async () => 'x' },
    );
    await llm.classify({ messages: [{ role: 'user', content: 'Classify.' }], inputTokenCount: 100 });
    expect(cache.get).toHaveBeenCalledTimes(3);
    for (const call of cache.get.mock.calls) expect(call[0]).toMatch(/^llmp:v3:[0-9a-f]{64}$/);
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
