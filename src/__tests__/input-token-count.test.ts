/**
 * #136: a caller-supplied input token count (LLMRequest.inputTokenCount) replaces
 * the character estimate for context-window routing, route plans and pre-call cost
 * estimates (it doesn't change model selection) (provider estimateCost, so the quota hook, and
 * CostTracker). It is validated on entry and never sent to a provider.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  estimateRequestInputTokens,
  getProviderDefaultModel,
  getRecommendedModel,
  getRoutingInfo,
  inferUseCaseFromRequest,
  rankModels,
} from '../model-catalog';
import {
  AnthropicProvider,
  CerebrasProvider,
  CloudflareProvider,
  ConfigurationError,
  CostTracker,
  GroqProvider,
  LLMProviders,
  NvidiaProvider,
  OpenAIProvider,
  getGatewayRoutePlan,
  normalizeLLMRequest,
} from '../index';
import type { BaseProvider } from '../providers/base';
import { canonicalToLLMRequest } from '../canonical';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { defaultExhaustionRegistry } from '../utils/exhaustion';
import type { LLMRequest, Tool } from '../types';

const lookupTool: Tool = {
  type: 'function',
  function: { name: 'lookup', description: 'Look something up', parameters: { type: 'object', properties: {} } },
};

function small(extra: Partial<LLMRequest> = {}): LLMRequest {
  return { messages: [{ role: 'user', content: 'hello' }], maxTokens: 10, ...extra };
}

function chatResult(content: string) {
  return {
    id: 'chatcmpl-test',
    choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
  };
}

function groqResponse() {
  return new Response(JSON.stringify({
    id: 'groq-1',
    object: 'chat.completion',
    created: 1,
    model: 'llama-3.3-70b-versatile',
    choices: [{ index: 0, message: { role: 'assistant', content: 'from groq' }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
  }), { status: 200, headers: { 'Content-Type': 'application/json' } });
}

function setup(run: ReturnType<typeof vi.fn>, contextWindowRouting = true) {
  const fetchMock = vi.fn().mockImplementation(async () => groqResponse());
  vi.stubGlobal('fetch', fetchMock);
  const llm = new LLMProviders({
    cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 },
    groq: { apiKey: 'test', maxRetries: 0 },
    stateScope: 'client',
    contextWindowRouting,
  });
  return { llm, fetchMock };
}

const INVALID = [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, 2 ** 53, '100', null];

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

describe('estimate', () => {
  it('uses a valid caller count instead of the character estimate', () => {
    expect(estimateRequestInputTokens(small())).toBe(2);
    expect(estimateRequestInputTokens(small({ inputTokenCount: 1 }))).toBe(1);
    expect(estimateRequestInputTokens(small({ inputTokenCount: 150_000 }))).toBe(150_000);
    expect(getRoutingInfo(small({ inputTokenCount: 150_000 })).estimatedInputTokens).toBe(150_000);
  });

  it.each(INVALID)('ignores %s in catalog helpers, route plans and cost estimates instead of throwing', value => {
    const request = small({ inputTokenCount: value as number });
    expect(estimateRequestInputTokens(request)).toBe(2);
    expect(() => rankModels('BALANCED', ['groq'], { request })).not.toThrow();
    expect(() => getProviderDefaultModel('groq', request)).not.toThrow();
    expect(() => getRecommendedModel('BALANCED', ['groq'], { request })).not.toThrow();
    expect(getRoutingInfo(request).estimatedInputTokens).toBe(2);
    expect(getGatewayRoutePlan(request, ['groq']).estimatedInputTokens).toBe(2);
    const capabilities = { inputTokenCost: 1, outputTokenCost: 0 } as never;
    expect(new CostTracker().calculateRequestCost(request, capabilities)).toBeCloseTo(0.002);
    const groq = new GroqProvider({ apiKey: 'test' });
    expect(groq.estimateCost(request)).toBe(groq.estimateCost(small()));
  });

  it('does not change model selection: use case, rankings and default models', () => {
    const plain = small();
    const counted = small({ inputTokenCount: 900_000 });
    expect(inferUseCaseFromRequest(counted)).toBe(inferUseCaseFromRequest(plain));
    expect(rankModels('BALANCED', ['cloudflare', 'groq'], { request: counted }).map(entry => entry.model))
      .toEqual(rankModels('BALANCED', ['cloudflare', 'groq'], { request: plain }).map(entry => entry.model));
    for (const provider of ['cloudflare', 'groq', 'openai', 'anthropic'] as const) {
      expect(getProviderDefaultModel(provider, counted)).toBe(getProviderDefaultModel(provider, plain));
    }
    expect(getRecommendedModel('BALANCED', ['cloudflare', 'groq'], { request: counted }))
      .toBe(getRecommendedModel('BALANCED', ['cloudflare', 'groq'], { request: plain }));
  });

  it('feeds the gateway route plan, from legacy and canonical requests', () => {
    const plan = getGatewayRoutePlan(small({ inputTokenCount: 300_000 }), ['groq']);
    expect(plan.estimatedInputTokens).toBe(300_000);
    expect(plan.legacyRequest.inputTokenCount).toBe(300_000);
    expect(plan.warnings.some(warning => warning.includes('300000 >'))).toBe(true);

    const canonical = normalizeLLMRequest(small({ inputTokenCount: 300_000 }));
    expect(canonical.inputTokenCount).toBe(300_000);
    expect(getGatewayRoutePlan(canonical, ['groq']).estimatedInputTokens).toBe(300_000);
    expect(canonicalToLLMRequest(canonical).inputTokenCount).toBe(300_000);
  });

  it('leaves canonical requests without a count unchanged in shape', () => {
    expect('inputTokenCount' in normalizeLLMRequest(small())).toBe(false);
    expect('inputTokenCount' in canonicalToLLMRequest(normalizeLLMRequest(small()))).toBe(false);
  });

  it('prices the pre-call cost estimate from the count', () => {
    const tracker = new CostTracker();
    const capabilities = { inputTokenCost: 1, outputTokenCost: 0 } as never;
    expect(tracker.calculateRequestCost(small({ inputTokenCount: 5_000 }), capabilities)).toBe(5);
    expect(tracker.calculateRequestCost(small({ inputTokenCount: 0 }), capabilities)).toBeCloseTo(0.002);
  });
});

const PROVIDERS: Array<[string, (run?: ReturnType<typeof vi.fn>) => BaseProvider]> = [
  ['openai', () => new OpenAIProvider({ apiKey: 'test', maxRetries: 0 })],
  ['anthropic', () => new AnthropicProvider({ apiKey: 'test', maxRetries: 0 })],
  ['cloudflare', run => new CloudflareProvider({ ai: { run: run ?? vi.fn() } as unknown as Ai, maxRetries: 0 })],
  ['cerebras', () => new CerebrasProvider({ apiKey: 'test', maxRetries: 0 })],
  ['groq', () => new GroqProvider({ apiKey: 'test', maxRetries: 0 })],
  ['nvidia', () => new NvidiaProvider({ apiKey: 'test', maxRetries: 0 })],
];

type Factory = { generateResponse(request: LLMRequest): Promise<unknown> };
function factoryOf(llm: LLMProviders): Factory {
  return (llm as unknown as { factory: Factory }).factory;
}

function toolCallResult() {
  return {
    id: 'chatcmpl-1',
    choices: [{ index: 0, finish_reason: 'tool_calls', message: { role: 'assistant', content: null,
      tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'lookup', arguments: '{}' } }] } }],
    usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
  };
}

describe('provider cost estimates', () => {
  const providers = PROVIDERS;

  it.each(providers)('%s estimateCost prices the counted input', (_name, make) => {
    const provider = make();
    // Priced on the input tokens, which some default models (NVIDIA's free tier) price at 0.
    const calculateCost = vi.spyOn(provider as unknown as { calculateCost(i: number, o: number, m: string): number }, 'calculateCost');
    provider.estimateCost(small({ maxTokens: 1 }));
    provider.estimateCost(small({ maxTokens: 1, inputTokenCount: 1_000_000 }));
    provider.estimateCost(small({ maxTokens: 1, inputTokenCount: 0 }));
    expect(calculateCost.mock.calls.map(call => call[0])).toEqual([2, 1_000_000, 2]);
  });

  it('passes the count-based estimate to quotaHook.check', async () => {
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async () => groqResponse()));
    const check = vi.fn().mockResolvedValue({ allowed: true });
    const llm = new LLMProviders({
      groq: { apiKey: 'test', maxRetries: 0 },
      stateScope: 'client',
      quotaHook: { check, record: vi.fn().mockResolvedValue(undefined) },
    });
    await llm.generateResponse(small());
    await llm.generateResponse(small({ inputTokenCount: 500_000 }));
    const [unset, counted] = check.mock.calls.map(call => call[0].estimatedCost);
    expect(unset).toBeGreaterThan(0);
    expect(counted).toBeGreaterThan(unset * 1000);
  });
});

describe('factory', () => {
  it('skips a provider whose model cannot hold the counted input, even when the text is short', async () => {
    const run = vi.fn().mockRejectedValue(new Error('binding exploded'));
    const { llm, fetchMock } = setup(run);
    await expect(llm.generateResponse(small({ inputTokenCount: 150_000 }))).rejects.toThrow('binding exploded');
    expect(run).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('falls back to a smaller window when the count says the input fits, even when the text is long', async () => {
    const run = vi.fn().mockRejectedValue(new Error('binding exploded'));
    const { llm, fetchMock } = setup(run);
    const response = await llm.generateResponse({
      messages: [{ role: 'user', content: 'x'.repeat(600_000) }], maxTokens: 10, inputTokenCount: 1_000,
    });
    expect(response.provider).toBe('groq');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('is not sent to the provider', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('ok'));
    const { llm, fetchMock } = setup(run, false);
    await llm.generateResponse(small({ inputTokenCount: 42 }));
    expect(JSON.stringify(run.mock.calls[0])).not.toContain('42');
    expect(JSON.stringify(run.mock.calls[0])).not.toContain('inputTokenCount');

    const groq = new GroqProvider({ apiKey: 'test', maxRetries: 0 });
    await groq.generateResponse(small({ inputTokenCount: 42 }));
    const body = String(fetchMock.mock.calls[0][1].body);
    expect(body).not.toContain('inputTokenCount');
  });

  it.each(INVALID)('rejects %s before any provider is called', async value => {
    const run = vi.fn().mockResolvedValue(chatResult('ok'));
    const { llm, fetchMock } = setup(run);
    const request = small({ inputTokenCount: value as number });
    await expect(llm.generateResponse(request)).rejects.toBeInstanceOf(ConfigurationError);
    await expect(llm.generateManagedResponseStream(request)).rejects.toBeInstanceOf(ConfigurationError);
    expect(() => llm.getRoutingInfo(request)).toThrow('inputTokenCount must be a positive safe integer');
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('rejects an invalid count before the response cache is read', async () => {
    const cache = { get: vi.fn().mockResolvedValue(null), put: vi.fn() };
    const run = vi.fn().mockResolvedValue(chatResult('ok'));
    vi.stubGlobal('fetch', vi.fn());
    const llm = new LLMProviders({ cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 }, stateScope: 'client', responseCache: cache });
    await expect(llm.generateResponse(small({ inputTokenCount: -5 }))).rejects.toBeInstanceOf(ConfigurationError);
    expect(cache.get).not.toHaveBeenCalled();
  });

  it.each(PROVIDERS)('is rejected by %s called directly, on generateResponse and streamResponse', async (_name, make) => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const run = vi.fn();
    const provider = make(run);
    for (const call of [
      () => provider.generateResponse(small({ inputTokenCount: 0 })),
      () => provider.streamResponse(small({ inputTokenCount: 1.5, stream: true })),
    ]) {
      const error = await call().catch(caught => caught);
      expect(error).toBeInstanceOf(ConfigurationError);
      expect(error.message).toContain('inputTokenCount must be a positive safe integer');
    }
    expect(fetchMock).not.toHaveBeenCalled();
    expect(run).not.toHaveBeenCalled();
  });

  it.each(PROVIDERS.filter(([name]) => name !== 'cloudflare'))('is not in the %s request body', async (_name, make) => {
    const fetchMock = vi.fn().mockImplementation(async () =>
      new Response(JSON.stringify({ error: { message: 'bad request' } }), { status: 400, headers: { 'Content-Type': 'application/json' } }));
    vi.stubGlobal('fetch', fetchMock);
    await make().generateResponse(small({ inputTokenCount: 424_242 })).catch(() => undefined);
    expect(fetchMock).toHaveBeenCalled();
    const body = String(fetchMock.mock.calls[0][1].body);
    expect(body).not.toContain('inputTokenCount');
    expect(body).not.toContain('424242');
  });
});

describe('usage the package has to estimate', () => {
  it('prices a Workers AI response without usage from the count', async () => {
    const run = vi.fn().mockResolvedValue({ response: 'ok' });
    const provider = new CloudflareProvider({ ai: { run } as unknown as Ai, maxRetries: 0 });
    const plain = await provider.generateResponse(small());
    const counted = await provider.generateResponse(small({ inputTokenCount: 900_000 }));
    expect(plain.usage.inputTokens).toBe(2);
    expect(counted.usage.inputTokens).toBe(900_000);
    expect(counted.usage.totalTokens).toBe(900_000 + counted.usage.outputTokens);
    expect(counted.usage.tokenProvenance).toBe('estimated');
    expect(counted.usage.cost).toBeGreaterThan(plain.usage.cost * 1000);
    // The recorded cost now matches what the pre-call estimate priced for the input.
    expect(provider.estimateCost(small({ inputTokenCount: 900_000, maxTokens: 1 })))
      .toBeGreaterThan(counted.usage.cost * 0.5);
  });

  async function completionOf(llm: LLMProviders, request: LLMRequest) {
    const managed = await llm.generateManagedResponseStream(request);
    const reader = managed.stream.getReader();
    while (!(await reader.read()).done) { /* drain */ }
    return managed.completion;
  }

  it('estimates a Workers AI stream without usage from the count', async () => {
    const run = vi.fn().mockImplementation(async () => new ReadableStream({ start(controller) {
      controller.enqueue({ response: 'streamed' }); controller.close();
    } }));
    const { llm } = setup(run, false);
    const counted = await completionOf(llm, small({ inputTokenCount: 7_777 }));
    expect(counted.usage.inputTokens).toBe(7_777);
    expect(counted.usage.totalTokens).toBe(7_777 + counted.usage.outputTokens);
    expect((await completionOf(llm, small())).usage.inputTokens).toBe(2);
  });

  it('reports the count as input tokens when a provider stream carries no usage at all', async () => {
    const { llm } = setup(vi.fn(), false);
    const providers = (llm as unknown as { factory: { providers: Map<string, BaseProvider> } }).factory.providers;
    vi.spyOn(providers.get('cloudflare')!, 'streamResponse').mockImplementation(async () =>
      new ReadableStream<string>({ start(controller) { controller.enqueue('plain'); controller.close(); } }));
    for (const [request, expected] of [[small({ inputTokenCount: 7_777 }), 7_777], [small(), 0]] as const) {
      const completion = await completionOf(llm, request);
      expect(completion.provider).toBe('cloudflare');
      expect(completion.usage.inputTokens).toBe(expected);
      expect(completion.usage.totalTokens).toBe(expected);
      expect(completion.usage.tokenProvenance).toBe('estimated');
    }
  });
});

describe('classify', () => {
  const schema = { type: 'object', properties: { label: { type: 'string' }, confidence: { type: 'number' } } };

  it('adds an estimate of the system prompt it supplies', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('{"label":"a","confidence":0.9}'));
    const { llm } = setup(run, false);
    const sent = vi.spyOn(factoryOf(llm), 'generateResponse');
    const input: LLMRequest = { messages: [{ role: 'user', content: 'text' }], inputTokenCount: 1_000 };
    await llm.classify(input, { schema });
    const request = sent.mock.calls[0][0] as LLMRequest;
    expect(request.systemPrompt).toContain('JSON schema');
    expect(request.inputTokenCount).toBe(1_000 + Math.ceil(request.systemPrompt!.length / 4));
    expect(input.inputTokenCount).toBe(1_000);
  });

  it("keeps the count when the caller's own system prompt is used, and leaves string input alone", async () => {
    const run = vi.fn().mockResolvedValue(chatResult('{"label":"a"}'));
    const { llm } = setup(run, false);
    const sent = vi.spyOn(factoryOf(llm), 'generateResponse');
    await llm.classify({ messages: [{ role: 'user', content: 'text' }], systemPrompt: 'Mine.', inputTokenCount: 1_000 }, { schema });
    await llm.classify('text', { schema });
    expect((sent.mock.calls[0][0] as LLMRequest).inputTokenCount).toBe(1_000);
    expect('inputTokenCount' in (sent.mock.calls[1][0] as LLMRequest)).toBe(false);
  });

  it('adds when classify replaces the caller system prompt, which can only overcount', async () => {
    const run = vi.fn().mockResolvedValue(chatResult('{"label":"a"}'));
    const { llm } = setup(run, false);
    const sent = vi.spyOn(factoryOf(llm), 'generateResponse');
    await llm.classify(
      { messages: [{ role: 'user', content: 'text' }], systemPrompt: 'Mine.', inputTokenCount: 1_000 },
      { systemPrompt: 'x'.repeat(400) },
    );
    expect((sent.mock.calls[0][0] as LLMRequest).inputTokenCount).toBe(1_100);
  });

  it.each(INVALID)('rejects %s before any call', async value => {
    const run = vi.fn().mockResolvedValue(chatResult('{"label":"a"}'));
    const { llm, fetchMock } = setup(run, false);
    await expect(llm.classify({ messages: [{ role: 'user', content: 'text' }], inputTokenCount: value as number }, { schema }))
      .rejects.toBeInstanceOf(ConfigurationError);
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe('tool loop', () => {
  it('adds an estimate of the appended messages to the count on later calls', async () => {
    const run = vi.fn()
      .mockResolvedValueOnce({
        id: 'chatcmpl-1',
        choices: [{ index: 0, finish_reason: 'tool_calls', message: { role: 'assistant', content: null,
          tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'lookup', arguments: '{}' } }] } }],
        usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
      })
      .mockRejectedValueOnce(new Error('binding exploded'));
    const { llm, fetchMock } = setup(run);
    const factory = (llm as unknown as { factory: { generateResponse(request: LLMRequest): Promise<unknown> } }).factory;
    const sent = vi.spyOn(factory, 'generateResponse');
    const request: LLMRequest = {
      messages: [{ role: 'user', content: 'Summarise.' }], tools: [lookupTool], maxTokens: 10, inputTokenCount: 150_000,
    };
    // Both calls keep the count, so the second call's failure does not fall back
    // to Groq's smaller window, even though the appended tool result is short.
    await expect(llm.generateResponseWithTools(request, { execute: async () => 'x'.repeat(400) }))
      .rejects.toThrow('binding exploded');
    expect(run).toHaveBeenCalledTimes(2);
    expect(fetchMock).not.toHaveBeenCalled();
    const [first, second] = sent.mock.calls.map(call => call[0]);
    expect(first.inputTokenCount).toBe(150_000);
    const appended = second.messages.slice(request.messages.length);
    expect(appended).toHaveLength(2);
    expect(second.inputTokenCount).toBe(150_000 + estimateRequestInputTokens({ messages: appended }));
    expect(second.inputTokenCount).toBeGreaterThan(150_000 + 100);
    expect(request.inputTokenCount).toBe(150_000);
  });

  it('estimates later calls as before when no count is set', async () => {
    const run = vi.fn()
      .mockResolvedValueOnce({
        id: 'chatcmpl-1',
        choices: [{ index: 0, finish_reason: 'tool_calls', message: { role: 'assistant', content: null,
          tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'lookup', arguments: '{}' } }] } }],
        usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
      })
      .mockRejectedValueOnce(new Error('binding exploded'));
    const { llm, fetchMock } = setup(run);
    const factory = (llm as unknown as { factory: { generateResponse(request: LLMRequest): Promise<unknown> } }).factory;
    const sent = vi.spyOn(factory, 'generateResponse');
    const response = await llm.generateResponseWithTools(
      { messages: [{ role: 'user', content: 'Summarise.' }], tools: [lookupTool], maxTokens: 10 },
      { execute: async () => 'short' },
    );
    expect(response.provider).toBe('groq');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(sent.mock.calls.every(call => !('inputTokenCount' in call[0]))).toBe(true);
  });

  it('drops the count instead of throwing when the sum would pass the safe-integer range', async () => {
    const run = vi.fn()
      .mockResolvedValueOnce(toolCallResult())
      .mockResolvedValueOnce(chatResult('done'));
    const { llm } = setup(run, false);
    const sent = vi.spyOn(factoryOf(llm), 'generateResponse');
    const response = await llm.generateResponseWithTools(
      { messages: [{ role: 'user', content: 'Summarise.' }], tools: [lookupTool], maxTokens: 10, inputTokenCount: Number.MAX_SAFE_INTEGER - 1 },
      { execute: async () => 'x'.repeat(400) },
    );
    expect(response.message).toBe('done');
    expect((sent.mock.calls[0][0] as LLMRequest).inputTokenCount).toBe(Number.MAX_SAFE_INTEGER - 1);
    expect('inputTokenCount' in (sent.mock.calls[1][0] as LLMRequest)).toBe(false);
  });

  it('keeps the first call off a small window when the count is large', async () => {
    const run = vi.fn().mockRejectedValue(new Error('binding exploded'));
    const { llm, fetchMock } = setup(run);
    await expect(llm.generateResponseWithTools(
      { messages: [{ role: 'user', content: 'Summarise.' }], tools: [lookupTool], maxTokens: 10, inputTokenCount: 150_000 },
      { execute: async () => 'short' },
    )).rejects.toThrow('binding exploded');
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
