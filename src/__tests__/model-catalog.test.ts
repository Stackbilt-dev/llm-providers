import { describe, expect, it } from 'vitest';
import { CreditLedger } from '../utils/credit-ledger';
import {
  MODEL_CATALOG,
  MODEL_RECOMMENDATIONS,
  getCatalogEntry,
  getProvidersForCatalogModel,
  modelSupportsBuiltInTools,
  getRecommendedModel,
  getRoutingInfo,
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

  it('does not select Workers AI Gemma for BALANCED when Cerebras is configured', () => {
    const withCerebras = getRecommendedModel('BALANCED', ['cerebras', 'cloudflare']);
    expect(withCerebras).not.toBe('@cf/google/gemma-4-26b-a4b-it');
  });

  it('does not select Workers AI Gemma for BALANCED when Groq is configured', () => {
    const withGroq = getRecommendedModel('BALANCED', ['groq', 'cloudflare']);
    expect(withGroq).not.toBe('@cf/google/gemma-4-26b-a4b-it');
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

describe('getRoutingInfo', () => {
  it('infers TOOL_CALLING use case and flags requiresTools for requests with tools', () => {
    const request: Partial<LLMRequest> = {
      messages: [{ role: 'user', content: 'Call the bash tool.' }],
      tools: [{
        type: 'function',
        function: { name: 'bash', description: 'Run bash', parameters: { type: 'object' } }
      }],
    };
    const info = getRoutingInfo(request, ['groq', 'anthropic']);
    expect(info.useCase).toBe('TOOL_CALLING');
    expect(info.requiresTools).toBe(true);
    expect(info.model).toBeTruthy();
    expect(info.provider).toBeTruthy();
  });

  it('returns deprecation warning for a compatibility-lifecycle model', () => {
    const info = getRoutingInfo(
      { messages: [{ role: 'user', content: 'hi' }], model: 'gpt-4' },
      ['openai']
    );
    expect(info.modelLifecycle).toBe('compatibility');
    expect(info.deprecationWarning).toBeTruthy();
  });

  it('returns deprecation warning with date for models that carry a deprecation date in description', () => {
    // Cerebras llama-3.1-8b is marked '(deprecated 2026-05-27)' in description
    const entry = getCatalogEntry('llama-3.1-8b');
    expect(entry).toBeDefined();
    const info = getRoutingInfo(
      { messages: [{ role: 'user', content: 'hi' }], model: 'llama-3.1-8b' },
      ['cerebras']
    );
    expect(info.deprecationWarning).toMatch(/2026-05-27/);
  });

  it('returns no deprecation warning for an active model', () => {
    const info = getRoutingInfo(
      { messages: [{ role: 'user', content: 'hi' }] },
      ['groq']
    );
    expect(info.modelLifecycle).toBe('active');
    expect(info.deprecationWarning).toBeUndefined();
  });

  it('estimates input tokens from message + system prompt length', () => {
    const content = 'x'.repeat(400); // 400 chars → ~100 tokens
    const systemPrompt = 'y'.repeat(400); // another ~100
    const info = getRoutingInfo(
      { messages: [{ role: 'user', content }], systemPrompt },
      ['openai']
    );
    // 800 chars / 4 = 200 estimated tokens
    expect(info.estimatedInputTokens).toBe(200);
  });

  it('respects a pinned model in the request rather than recommending', () => {
    const info = getRoutingInfo(
      { messages: [{ role: 'user', content: 'hi' }], model: 'gpt-4o-mini' },
      ['anthropic', 'openai']
    );
    expect(info.model).toBe('gpt-4o-mini');
    expect(info.provider).toBe('openai');
  });

  it('sets requestsStreaming when stream flag is true', () => {
    const info = getRoutingInfo(
      { messages: [{ role: 'user', content: 'hi' }], stream: true },
      ['groq']
    );
    expect(info.requestsStreaming).toBe(true);
  });

  it('infers VISION use case and flags requiresVision for image requests', () => {
    const info = getRoutingInfo(
      { messages: [{ role: 'user', content: 'Describe this.' }], images: [{ url: 'https://example.com/img.png' }] },
      ['openai', 'anthropic']
    );
    expect(info.useCase).toBe('VISION');
    expect(info.requiresVision).toBe(true);
    expect(info.catalogEntry?.capabilities.supportsVision).toBe(true);
  });

  it('infers LONG_CONTEXT for requests with many tokens', () => {
    const content = 'word '.repeat(5000); // ~25 000 chars → infers LONG_CONTEXT
    const info = getRoutingInfo(
      { messages: [{ role: 'user', content }] },
      ['anthropic']
    );
    expect(info.useCase).toBe('LONG_CONTEXT');
    expect(info.estimatedInputTokens).toBeGreaterThan(5000);
  });

  it('returns modelLifecycle unknown for an uncatalogued model', () => {
    const info = getRoutingInfo(
      { messages: [{ role: 'user', content: 'hi' }], model: 'custom-private-model' },
      ['openai']
    );
    expect(info.modelLifecycle).toBe('unknown');
    expect(info.deprecationWarning).toBeUndefined();
  });
});

describe('getProvidersForCatalogModel', () => {
  it('returns all providers serving a collided model string, in fallback order', () => {
    // openai/gpt-oss-120b is hosted by both Cerebras and Groq.
    const providers = getProvidersForCatalogModel('openai/gpt-oss-120b');
    expect(providers).toEqual(['cerebras', 'groq']);
  });

  it('returns a single provider for an unambiguous model', () => {
    expect(getProvidersForCatalogModel('llama-3.3-70b-versatile')).toEqual(['groq']);
  });

  it('returns an empty array for an unknown model', () => {
    expect(getProvidersForCatalogModel('groq/compound')).toEqual([]);
    expect(getProvidersForCatalogModel('not-a-real-model')).toEqual([]);
  });
});

describe('modelSupportsBuiltInTools', () => {
  it('is true for the Groq-hosted gpt-oss entry', () => {
    expect(modelSupportsBuiltInTools('openai/gpt-oss-120b', 'groq')).toBe(true);
    expect(modelSupportsBuiltInTools('openai/gpt-oss-120b', 'groq', 'web_search')).toBe(true);
  });

  it('is false for the Cerebras-hosted gpt-oss entry — the disambiguator', () => {
    expect(modelSupportsBuiltInTools('openai/gpt-oss-120b', 'cerebras')).toBe(false);
  });

  it('is false for a tool the model does not advertise', () => {
    expect(modelSupportsBuiltInTools('openai/gpt-oss-120b', 'groq', 'wolfram_alpha')).toBe(false);
  });

  it('is false for function-only models and unknown pairs', () => {
    expect(modelSupportsBuiltInTools('llama-3.3-70b-versatile', 'groq')).toBe(false);
    expect(modelSupportsBuiltInTools('not-a-real-model', 'groq')).toBe(false);
  });
});
