import { describe, expect, it } from 'vitest';
import { CreditLedger } from '../utils/credit-ledger';
import {
  MODEL_CATALOG,
  MODEL_RECOMMENDATIONS,
  getRecommendedModel,
  inferUseCaseFromRequest,
  type ModelRecommendationUseCase,
} from '../model-catalog';
import type { CircuitBreakerState, LLMRequest } from '../types';

function closedState(): CircuitBreakerState {
  return {
    state: 'CLOSED',
    failures: 0,
    consecutiveFailures: 0,
    primaryTrafficPct: 1,
    totalFailures: 0,
    totalSuccesses: 0,
    totalRequests: 0,
  };
}

describe('model catalog', () => {
  it('excludes retired models from public recommendations', () => {
    const recommended = Object.values(MODEL_RECOMMENDATIONS).flat();

    expect(recommended).not.toContain('gpt-4o');
    expect(recommended).toContain('gpt-4o-mini');
    expect(recommended).toContain('claude-haiku-4-5-20251001');
  });

  it('keeps retired models in the catalog for compatibility metadata only', () => {
    const retired = MODEL_CATALOG.find(entry => entry.model === 'gpt-4o');

    expect(retired).toMatchObject({
      model: 'gpt-4o',
      lifecycle: 'retired',
      provider: 'openai',
    });
  });

  it('infers tool-calling requests from request shape', () => {
    const request: LLMRequest = {
      messages: [{ role: 'user', content: 'Call the weather tool.' }],
      tools: [{
        type: 'function',
        function: {
          name: 'get_weather',
          description: 'Weather lookup',
          parameters: { type: 'object' }
        }
      }]
    };

    expect(inferUseCaseFromRequest(request)).toBe('TOOL_CALLING');
  });

  it('prefers active modern baselines over retired OpenAI models', () => {
    const recommended = getRecommendedModel('BALANCED', ['openai']);

    expect(recommended).toBe('gpt-4o-mini');
  });

  it('uses provider health and burn-rate pressure when selecting a model', () => {
    const ledger = new CreditLedger({
      budgets: [
        { provider: 'cloudflare', monthlyBudget: 1 },
        { provider: 'openai', monthlyBudget: 10 },
      ],
    });

    ledger.record('cloudflare', '@cf/google/gemma-4-26b-a4b-it', 0.95, 10_000, 4_000);

    const request: LLMRequest = {
      messages: [{ role: 'user', content: 'Summarize this incident report and call tools if needed.' }],
      tools: [{
        type: 'function',
        function: {
          name: 'lookup_incident',
          description: 'Look up incidents',
          parameters: { type: 'object' }
        }
      }],
      maxTokens: 800
    };

    const recommended = getRecommendedModel(
      'TOOL_CALLING' satisfies ModelRecommendationUseCase,
      ['cloudflare', 'openai', 'anthropic'],
      {
        request,
        ledger,
        providerHealth: {
          cloudflare: {
            healthy: true,
            circuitBreaker: {
              ...closedState(),
              state: 'DEGRADED',
              primaryTrafficPct: 0.2,
              failures: 2,
              consecutiveFailures: 2,
            }
          },
          openai: { healthy: true, circuitBreaker: closedState() },
          anthropic: { healthy: true, circuitBreaker: closedState() },
        }
      }
    );

    expect(recommended).toBe('gpt-4o-mini');
  });
});
