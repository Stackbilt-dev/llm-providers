/**
 * #136 / #113: the context window is a routing constraint. When a provider that can be
 * relied on right now has a planned model that holds the estimated input, providers
 * whose planned model can't are skipped at dispatch and recorded as CONTEXT_WINDOW.
 * The chain, planned models and catalog ranking are unchanged.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  estimateRequestInputTokens,
  fitsContextWindow,
  getCatalogEntry,
  getProviderDefaultModel,
  getRecommendedModel,
  inferUseCaseFromRequest,
  rankModels,
} from '../model-catalog';
import { CreditLedger, LLMProviders, NoRouteError, ResilienceContext } from '../index';
import type { ProviderFactoryConfig } from '../factory';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { defaultExhaustionRegistry } from '../utils/exhaustion';
import type { LLMRequest, Tool } from '../types';

/** A single user message of roughly `tokens` estimated input tokens. */
function requestOf(tokens: number, extra: Partial<LLMRequest> = {}): LLMRequest {
  return { messages: [{ role: 'user', content: 'x'.repeat(tokens * 4) }], maxTokens: 10, ...extra };
}

const lookupTool: Tool = {
  type: 'function',
  function: { name: 'lookup', description: 'Look something up', parameters: { type: 'object', properties: {} } },
};

/** A tool-loop turn whose size is all in the tool result, as generateResponseWithTools builds it. */
function toolLoopOf(tokens: number): LLMRequest {
  return {
    tools: [lookupTool],
    maxTokens: 10,
    messages: [
      { role: 'user', content: 'Summarise the document.' },
      { role: 'assistant', content: '', toolCalls: [{ id: 'call-1', type: 'function', function: { name: 'lookup', arguments: '{}' } }] },
      { role: 'user', content: '', toolResults: [{ id: 'call-1', output: 'x'.repeat(tokens * 4) }] },
    ],
  };
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

function setup(cloudflare: 'ok' | 'fail' | 'stream', config: Partial<ProviderFactoryConfig> = {}) {
  const run = cloudflare === 'ok'
    ? vi.fn().mockResolvedValue(chatResult('from cloudflare'))
    : cloudflare === 'stream'
      ? vi.fn().mockImplementation(async () => new ReadableStream({ start(controller) {
          controller.enqueue({ response: 'streamed' }); controller.close();
        } }))
      : vi.fn().mockRejectedValue(new Error('binding exploded'));
  const fetchMock = vi.fn().mockImplementation(async () => groqResponse());
  vi.stubGlobal('fetch', fetchMock);
  const llm = new LLMProviders({
    cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 },
    groq: { apiKey: 'test', maxRetries: 0 },
    stateScope: 'client',
    ...config,
  });
  return { llm, run, fetchMock };
}

type Plan = { chain: string[]; contextSkips: Set<string> };
function planOf(llm: LLMProviders, request: LLMRequest): Plan {
  return (llm as unknown as { factory: { buildProviderPlan(request: LLMRequest): Plan } }).factory.buildProviderPlan(request);
}

async function drain(stream: ReadableStream<string>) {
  const reader = stream.getReader();
  let text = '';
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) return text;
    text += chunk.value;
  }
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

describe('context-window estimate', () => {
  it('counts message and system-prompt characters divided by four', () => {
    expect(estimateRequestInputTokens({
      systemPrompt: 'abcd',
      messages: [{ role: 'user', content: 'x'.repeat(40) }, { role: 'assistant', content: 'y'.repeat(4) }],
    })).toBe(12);
    expect(estimateRequestInputTokens({ messages: [] })).toBe(1);
  });

  it('counts tool-call arguments, tool results and tool errors', () => {
    expect(estimateRequestInputTokens({
      messages: [
        { role: 'assistant', content: '', toolCalls: [{ id: 'a', type: 'function', function: { name: 'f', arguments: 'x'.repeat(40) } }] },
        { role: 'user', content: '', toolResults: [{ id: 'a', output: 'x'.repeat(80), error: 'x'.repeat(8) }] },
      ],
    })).toBe(32);
    expect(estimateRequestInputTokens(toolLoopOf(150_000))).toBeGreaterThan(150_000);
  });

  it('counts the larger of provider-native text and plain text', () => {
    const thinking = { provider: 'anthropic', blocks: [{ type: 'thinking', thinking: 'x'.repeat(400), signature: 's' }, { type: 'text', text: 'x'.repeat(40) }] };
    expect(estimateRequestInputTokens({ messages: [{ role: 'assistant', content: 'x'.repeat(40), providerContent: thinking }] })).toBe(110);
    expect(estimateRequestInputTokens({ messages: [{ role: 'assistant', content: 'x'.repeat(800), providerContent: thinking }] })).toBe(200);
  });

  it('counts non-string content as zero, so catalog helpers do not throw', () => {
    const malformed = { messages: [{ role: 'assistant', content: null }, { role: 'user', content: 'hi' }] } as unknown as LLMRequest;
    expect(estimateRequestInputTokens(malformed)).toBe(1);
    expect(() => rankModels('BALANCED', ['groq'], { request: malformed })).not.toThrow();
    expect(() => getProviderDefaultModel('groq', malformed)).not.toThrow();
    expect(() => getRecommendedModel('BALANCED', ['groq'], { request: malformed })).not.toThrow();
  });

  it('counts provider-native text and thinking plus tool-call arguments, but not encrypted redacted thinking', () => {
    const blocks = [
      { type: 'thinking', thinking: 'x'.repeat(40), signature: 's' },
      { type: 'redacted_thinking', data: 'x'.repeat(40) },
      { type: 'text', text: 'x'.repeat(40) },
      { type: 'tool_use', id: 't', name: 'f', input: { q: 'x'.repeat(31) } },
    ];
    // 40 thinking + 40 text + 40 tool-call arguments; redacted data and the tool_use block are not counted again.
    expect(estimateRequestInputTokens({ messages: [{ role: 'assistant', content: '',
      toolCalls: [{ id: 't', type: 'function', function: { name: 'f', arguments: 'x'.repeat(40) } }],
      providerContent: { provider: 'anthropic', blocks } }] })).toBe(30);
  });

  it('counts malformed containers as zero instead of throwing', () => {
    const malformed = [
      { role: 'assistant', content: 'x'.repeat(40), providerContent: { provider: 'anthropic', blocks: [null, 'text', 7] } },
      { role: 'assistant', content: '', providerContent: { provider: 'anthropic', blocks: {} } },
      { role: 'assistant', content: '', providerContent: 'not an object' },
      { role: 'assistant', content: '', toolCalls: {} },
      { role: 'assistant', content: '', toolCalls: [null, { function: null }, { function: { arguments: 5 } }] },
      { role: 'user', content: '', toolResults: 'nope' },
      { role: 'user', content: '', toolResults: [null, { output: {} }] },
      null,
    ];
    const request = { messages: malformed } as unknown as LLMRequest;
    expect(estimateRequestInputTokens(request)).toBe(10);
    expect(estimateRequestInputTokens({ messages: {} } as unknown as LLMRequest)).toBe(1);
    expect(() => rankModels('BALANCED', ['groq'], { request: { messages: malformed.slice(0, 3) } as unknown as LLMRequest })).not.toThrow();
  });

  it('does not change use-case inference: only message text counts there', () => {
    const thinking = { provider: 'anthropic', blocks: [{ type: 'thinking', thinking: 'x'.repeat(48_000), signature: 's' }] };
    const request = { messages: [{ role: 'assistant' as const, content: 'x'.repeat(10_000), providerContent: thinking }] };
    expect(inferUseCaseFromRequest(request)).toBe('BALANCED');
    expect(inferUseCaseFromRequest({ messages: [{ role: 'user', content: 'x'.repeat(20_000) }] })).toBe('LONG_CONTEXT');
  });

  it('compares the estimate with the window', () => {
    const groq = getCatalogEntry('llama-3.3-70b-versatile')!;
    const window = groq.capabilities.maxContextLength;
    expect(fitsContextWindow(groq, estimateRequestInputTokens(requestOf(window)))).toBe(true);
    expect(fitsContextWindow(groq, estimateRequestInputTokens(requestOf(window + 1)))).toBe(false);
  });
});

describe('catalog ranking is unchanged', () => {
  it('ranks large requests as before, including models too small for them', () => {
    const request = requestOf(150_000);
    const ranked = rankModels('LONG_CONTEXT', ['cloudflare', 'groq'], { request });
    expect(ranked.some(entry => entry.provider === 'groq')).toBe(true);
    expect(getProviderDefaultModel('groq', request)).toBe('llama-3.3-70b-versatile');
    expect(getRecommendedModel('LONG_CONTEXT', ['cloudflare', 'groq'], { request })).toBe(ranked[0].model);
  });
});

describe('factory fallback respects context windows', () => {
  it('falls back to a smaller-window provider when the input fits it (control)', async () => {
    const { llm, fetchMock } = setup('fail');
    const response = await llm.generateResponse(requestOf(1_000));
    expect(response.provider).toBe('groq');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('does not fall back to a provider whose model cannot hold the input', async () => {
    const { llm, run, fetchMock } = setup('fail');
    await expect(llm.generateResponse(requestOf(150_000))).rejects.toThrow('binding exploded');
    expect(run).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('counts tool results, so a grown tool loop does not fall back to a small window', async () => {
    const { llm, run, fetchMock } = setup('fail');
    await expect(llm.generateResponse(toolLoopOf(150_000))).rejects.toThrow('binding exploded');
    expect(run).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('keeps generateResponseWithTools off a small window once the tool result grows', async () => {
    const run = vi.fn()
      .mockResolvedValueOnce({
        id: 'chatcmpl-1',
        choices: [{ index: 0, finish_reason: 'tool_calls', message: { role: 'assistant', content: null,
          tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'lookup', arguments: '{}' } }] } }],
        usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
      })
      .mockRejectedValueOnce(new Error('binding exploded'));
    const fetchMock = vi.fn().mockImplementation(async () => groqResponse());
    vi.stubGlobal('fetch', fetchMock);
    const llm = new LLMProviders({
      cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 },
      groq: { apiKey: 'test', maxRetries: 0 },
      stateScope: 'client',
    });
    await expect(llm.generateResponseWithTools(
      { messages: [{ role: 'user', content: 'Summarise the document.' }], tools: [lookupTool], maxTokens: 10 },
      { execute: async () => 'x'.repeat(600_000) },
    )).rejects.toThrow('binding exploded');
    expect(run).toHaveBeenCalledTimes(2);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('serves a large request from the provider whose model can hold it', async () => {
    const { llm, run, fetchMock } = setup('ok');
    const response = await llm.generateResponse(requestOf(150_000));
    expect(response.provider).toBe('cloudflare');
    expect(run).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('leaves routing unchanged when no configured model can hold the input', async () => {
    const { llm, run, fetchMock } = setup('fail');
    const response = await llm.generateResponse(requestOf(1_100_000));
    expect(run).toHaveBeenCalledTimes(1);
    expect(response.provider).toBe('groq');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('keeps smaller models when the only fitting provider is quota-exhausted', async () => {
    const resilienceContext = new ResilienceContext();
    resilienceContext.exhaustion.markExhausted('cloudflare');
    const { llm, run, fetchMock } = setup('ok', { resilienceContext });
    const response = await llm.generateResponse(requestOf(150_000));
    expect(response.provider).toBe('groq');
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(llm.getRoutingInfo(requestOf(150_000)).provider).toBe('groq');
  });

  it('keeps smaller models when the only fitting provider has an open breaker, with default config', async () => {
    const resilienceContext = new ResilienceContext();
    resilienceContext.circuitBreakers.getBreaker('cloudflare').forceOpen();
    const { llm, run, fetchMock } = setup('ok', { resilienceContext });
    const response = await llm.generateResponse(requestOf(150_000));
    expect(response.provider).toBe('groq');
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('skips nothing when the only fitting provider is unavailable, with three providers', () => {
    const resilienceContext = new ResilienceContext();
    resilienceContext.exhaustion.markExhausted('cloudflare');
    vi.stubGlobal('fetch', vi.fn());
    const llm = new LLMProviders({
      cloudflare: { ai: { run: vi.fn() } as unknown as Ai, maxRetries: 0 },
      groq: { apiKey: 'test', maxRetries: 0 },
      openai: { apiKey: 'test', maxRetries: 0 },
      stateScope: 'client',
      resilienceContext,
    });
    const plan = planOf(llm, requestOf(150_000));
    expect([...plan.chain].sort()).toEqual(['cloudflare', 'groq', 'openai']);
    expect(plan.contextSkips.size).toBe(0);
  });

  it('keeps smaller models when the only fitting provider has an open circuit breaker', async () => {
    const resilienceContext = new ResilienceContext();
    resilienceContext.circuitBreakers.getBreaker('cloudflare').forceOpen();
    const { llm, run, fetchMock } = setup('ok', { resilienceContext, enableCircuitBreaker: true });
    const response = await llm.generateResponse(requestOf(150_000));
    expect(response.provider).toBe('groq');
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('applies to image requests', async () => {
    const { llm, run, fetchMock } = setup('fail');
    await expect(llm.generateResponse(requestOf(150_000, { images: [{ data: 'YQ==', mimeType: 'image/png' }] })))
      .rejects.toThrow('binding exploded');
    expect(run).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('skips a preferred provider that cannot hold the input', async () => {
    const { llm, run, fetchMock } = setup('ok', { preferredProvider: 'groq' });
    expect((await llm.generateResponse(requestOf(1_000))).provider).toBe('groq');
    fetchMock.mockClear();
    const response = await llm.generateResponse(requestOf(150_000));
    expect(response.provider).toBe('cloudflare');
    expect(run).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('with allowFallback: false and no pin, throws NoRouteError when the first provider is skipped', async () => {
    const { llm, run, fetchMock } = setup('ok', { preferredProvider: 'groq' });
    const error = await llm.generateResponse(requestOf(150_000, { allowFallback: false })).catch(caught => caught);
    expect(error).toBeInstanceOf(NoRouteError);
    expect(error.message).toContain('groq: CONTEXT_WINDOW');
    expect(run).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
    await expect(llm.generateManagedResponseStream(requestOf(150_000, { allowFallback: false })))
      .rejects.toBeInstanceOf(NoRouteError);
  });

  it('records a skip in the fallback report, naming the requested provider', async () => {
    const { llm } = setup('ok', { preferredProvider: 'groq' });
    const response = await llm.generateResponse(requestOf(150_000));
    expect(response.provider).toBe('cloudflare');
    expect(response.fallback).toEqual({
      requestedProvider: 'groq',
      requestedModel: 'llama-3.3-70b-versatile',
      failedAttempts: [{ provider: 'groq', model: 'llama-3.3-70b-versatile', errorCode: 'CONTEXT_WINDOW' }],
    });
  });

  it('contextWindowRouting: false routes without the check', async () => {
    const { llm, run, fetchMock } = setup('fail', { contextWindowRouting: false });
    expect(planOf(llm, requestOf(150_000)).contextSkips.size).toBe(0);
    const response = await llm.generateResponse(requestOf(150_000));
    expect(run).toHaveBeenCalledTimes(1);
    expect(response.provider).toBe('groq');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('skips nothing when the fitting provider has a degraded or recovering breaker', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0);
    vi.useFakeTimers();
    try {
      vi.setSystemTime(1_000_000);
      const resilienceContext = new ResilienceContext();
      const breaker = resilienceContext.circuitBreakers.getBreaker('cloudflare');
      const { llm } = setup('ok', { resilienceContext });
      breaker.beginExecution().failure(new Error('x'));
      vi.advanceTimersByTime(10);
      breaker.beginExecution().failure(new Error('x'));
      vi.advanceTimersByTime(10);
      expect(breaker.getState().state).toBe('DEGRADED');
      expect(planOf(llm, requestOf(150_000)).contextSkips.size).toBe(0);
      breaker.beginExecution().success();
      vi.advanceTimersByTime(10);
      expect(breaker.getState().state).toBe('RECOVERING');
      expect(planOf(llm, requestOf(150_000)).contextSkips.size).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it('skips nothing when the fitting provider is ledger-limited, and logs nothing while planning', () => {
    const ledger = new CreditLedger({ budgets: [{ provider: 'cloudflare', monthlyBudget: 100, rateLimits: { rpm: 1 } }] });
    ledger.record('cloudflare', '@cf/zai-org/glm-5.3-flash', 0, 10, 10);
    const logger = { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
    const { llm } = setup('ok', { ledger, logger });
    expect(planOf(llm, requestOf(150_000)).contextSkips.size).toBe(0);
    expect(logger.warn.mock.calls.some(call => String(call[0]).includes('Rate limit'))).toBe(false);
  });

  it('does not create circuit breakers while planning', () => {
    const resilienceContext = new ResilienceContext();
    const { llm } = setup('ok', { resilienceContext });
    expect(planOf(llm, requestOf(150_000)).contextSkips).toEqual(new Set(['groq']));
    expect(Object.keys(resilienceContext.circuitBreakers.getAllStates())).toEqual([]);
  });

  it('applies on the managed streaming path', async () => {
    const { llm, run, fetchMock } = setup('stream', { preferredProvider: 'groq' });
    const managed = await llm.generateManagedResponseStream(requestOf(150_000));
    expect(await drain(managed.stream)).toBe('streamed');
    await expect(managed.completion).resolves.toMatchObject({
      outcome: 'success', provider: 'cloudflare',
      fallback: { requestedProvider: 'groq', failedAttempts: [{ provider: 'groq', errorCode: 'CONTEXT_WINDOW' }] },
    });
    expect(run).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('serves a strict-reasoning request from the provider that enforces it', async () => {
    const { llm, run, fetchMock } = setup('ok');
    const response = await llm.generateResponse(requestOf(150_000, {
      model: '@cf/zai-org/glm-5.3-flash', reasoning: { effort: 'low', strict: true },
    }));
    expect(response.provider).toBe('cloudflare');
    expect(run).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('still sends a pinned model the request; its provider decides', async () => {
    const { llm, run, fetchMock } = setup('ok');
    const response = await llm.generateResponse(requestOf(150_000, { model: 'llama-3.3-70b-versatile' }));
    expect(response.provider).toBe('groq');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(run).not.toHaveBeenCalled();
  });

  it('reports the planned provider in client routing info', () => {
    const { llm } = setup('ok', { preferredProvider: 'groq' });
    expect(llm.getRoutingInfo(requestOf(150_000)).provider).toBe('cloudflare');
    expect(llm.getRoutingInfo(requestOf(1_100_000)).provider).toBe('groq');
  });
});

describe('fallback rules are unchanged', () => {
  it('a rule can still send the request to a provider whose model is too small', async () => {
    const run = vi.fn().mockRejectedValue(new Error('binding exploded'));
    const fetchMock = vi.fn().mockImplementation(async () => groqResponse());
    vi.stubGlobal('fetch', fetchMock);
    const llm = new LLMProviders({
      cloudflare: { ai: { run } as unknown as Ai, maxRetries: 0 },
      groq: { apiKey: 'test', maxRetries: 0 },
      stateScope: 'client',
      fallbackRules: [{ condition: 'error', fallbackProvider: 'groq' }],
    });
    const response = await llm.generateResponse(requestOf(150_000));
    expect(response.provider).toBe('groq');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
