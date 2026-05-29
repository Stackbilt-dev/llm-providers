/**
 * Built-in-tools routing (issue #69, S1).
 *
 * Guards the cross-provider catalog collision: `openai/gpt-oss-120b` is hosted
 * by BOTH Cerebras and Groq, but only Groq runs built-in tools. The router must:
 *   - keep the prior default (Cerebras) for plain requests — no regression, and
 *   - steer a `builtInTools` request to Groq, the capability-matching host.
 * Also verifies the Groq-unique `groq/compound` system string routes to Groq
 * and reaches the adapter with its model preserved (no silent default swap).
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { LLMProviderFactory } from '../factory';
import type { LLMRequest, LLMResponse } from '../types';
import { defaultCostTracker } from '../utils/cost-tracker';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import { defaultExhaustionRegistry } from '../utils/exhaustion';
import { defaultLatencyHistogram } from '../utils/latency-histogram';

function makeMock(name: string, models: string[]) {
  return {
    name,
    models,
    supportsStreaming: true,
    supportsTools: true,
    supportsBatching: false,
    supportsVision: false,
    generateResponse: vi.fn().mockResolvedValue({
      message: `${name} response`,
      usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30, cost: 0.001 },
      model: models[0],
      provider: name,
      responseTime: 100,
    } as LLMResponse),
    streamResponse: vi.fn(),
    getProviderBalance: vi.fn(),
    validateConfig: vi.fn().mockReturnValue(true),
    getModels: vi.fn().mockReturnValue(models),
    estimateCost: vi.fn().mockReturnValue(0.001),
    healthCheck: vi.fn().mockResolvedValue(true),
    getMetrics: vi.fn().mockReturnValue({
      requestCount: 0, successCount: 0, errorCount: 0, averageLatency: 0,
      totalCost: 0, rateLimitHits: 0, lastUsed: Date.now(),
    }),
    resetMetrics: vi.fn(),
  };
}

const mockCerebras = makeMock('cerebras', ['openai/gpt-oss-120b']);
const mockGroq = makeMock('groq', ['llama-3.3-70b-versatile', 'openai/gpt-oss-120b']);

vi.mock('../providers/cerebras', () => ({
  CerebrasProvider: vi.fn().mockImplementation(() => mockCerebras),
}));
vi.mock('../providers/groq', () => ({
  GroqProvider: vi.fn().mockImplementation(() => mockGroq),
}));

describe('built-in-tools routing — gpt-oss-120b collision', () => {
  let factory: LLMProviderFactory;

  beforeEach(() => {
    vi.clearAllMocks();
    defaultCostTracker.reset();
    defaultCircuitBreakerManager.resetAll();
    defaultExhaustionRegistry.reset();
    defaultLatencyHistogram.reset();
    [mockCerebras, mockGroq].forEach(m => {
      m.generateResponse.mockReset().mockResolvedValue({
        message: `${m.name} response`,
        usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30, cost: 0.001 },
        model: m.models[0],
        provider: m.name,
        responseTime: 100,
      } as LLMResponse);
    });

    // Both collided providers configured — the only situation where the
    // first-match catalog lookup would misroute.
    factory = new LLMProviderFactory({
      cerebras: { apiKey: 'test-cerebras' },
      groq: { apiKey: 'test-groq' },
      defaultProvider: 'auto',
      enableCircuitBreaker: false,
      enableRetries: false,
    });
  });

  const base: Omit<LLMRequest, 'model'> = {
    messages: [{ role: 'user', content: 'hi' }],
    maxTokens: 50,
  };

  it('routes a plain pinned gpt-oss-120b to Cerebras (no regression)', async () => {
    await factory.generateResponse({ ...base, model: 'openai/gpt-oss-120b' });
    expect(mockCerebras.generateResponse).toHaveBeenCalled();
    expect(mockGroq.generateResponse).not.toHaveBeenCalled();
  });

  it('routes a built-in-tools gpt-oss-120b request to Groq (the capable host)', async () => {
    await factory.generateResponse({
      ...base,
      model: 'openai/gpt-oss-120b',
      builtInTools: [{ type: 'web_search' }],
    });
    expect(mockGroq.generateResponse).toHaveBeenCalled();
    expect(mockCerebras.generateResponse).not.toHaveBeenCalled();
    // model is preserved through requestForProvider, not swapped for a default
    expect(mockGroq.generateResponse).toHaveBeenCalledWith(
      expect.objectContaining({ model: 'openai/gpt-oss-120b' }),
    );
  });

  it('routes the Groq-unique groq/compound system to Groq with model preserved', async () => {
    await factory.generateResponse({ ...base, model: 'groq/compound' });
    expect(mockGroq.generateResponse).toHaveBeenCalledWith(
      expect.objectContaining({ model: 'groq/compound' }),
    );
    expect(mockCerebras.generateResponse).not.toHaveBeenCalled();
  });
});
