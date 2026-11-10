/**
 * #136: a caller-supplied input token count (LLMRequest.inputTokenCount) replaces
 * the character estimate for context-window routing, route plans, recommendations
 * and pre-call cost estimates. It is validated on entry and never sent to a provider.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  estimateRequestInputTokens,
  getProviderDefaultModel,
  getRecommendedModel,
  getRoutingInfo,
  rankModels,
} from '../model-catalog';
import {
  ConfigurationError,
  CostTracker,
  GroqProvider,
  LLMProviders,
  getGatewayRoutePlan,
  normalizeLLMRequest,
} from '../index';
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

  it.each(INVALID)('ignores %s in catalog helpers instead of throwing', value => {
    const request = small({ inputTokenCount: value as number });
    expect(estimateRequestInputTokens(request)).toBe(2);
    expect(() => rankModels('BALANCED', ['groq'], { request })).not.toThrow();
    expect(() => getProviderDefaultModel('groq', request)).not.toThrow();
    expect(() => getRecommendedModel('BALANCED', ['groq'], { request })).not.toThrow();
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
    expect(() => llm.getRoutingInfo(request)).toThrow('inputTokenCount must be a positive integer');
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

  it('is rejected by a provider called directly', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const groq = new GroqProvider({ apiKey: 'test', maxRetries: 0 });
    const error = await groq.generateResponse(small({ inputTokenCount: 0 })).catch(caught => caught);
    expect(error).toBeInstanceOf(ConfigurationError);
    expect(error.message).toContain('inputTokenCount must be a positive integer');
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe('tool loop', () => {
  it('uses the count for the first call only, since later calls add tool results', async () => {
    const run = vi.fn()
      .mockResolvedValueOnce({
        id: 'chatcmpl-1',
        choices: [{ index: 0, finish_reason: 'tool_calls', message: { role: 'assistant', content: null,
          tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'lookup', arguments: '{}' } }] } }],
        usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
      })
      .mockRejectedValueOnce(new Error('binding exploded'));
    const { llm, fetchMock } = setup(run);
    const request: LLMRequest = {
      messages: [{ role: 'user', content: 'Summarise.' }], tools: [lookupTool], maxTokens: 10, inputTokenCount: 150_000,
    };
    // First call: the count keeps the request on Cloudflare. Second call: the
    // small tool result is estimated, so the failure falls back to Groq.
    const response = await llm.generateResponseWithTools(request, { execute: async () => 'short' });
    expect(response.provider).toBe('groq');
    expect(run).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(request.inputTokenCount).toBe(150_000);
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
