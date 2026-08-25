import { afterEach, describe, expect, it, vi } from 'vitest';
import { LLMProviderFactory } from '../factory';
import { defaultCircuitBreakerManager } from '../utils/circuit-breaker';
import type { ImageAnalysisAttemptEvent, ImageAnalysisCompleteEvent } from '../utils/hooks';

describe('analyzeImage physical attempt observability', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    defaultCircuitBreakerManager.resetAll();
  });

  it('observes real provider-internal retries, not only factory fallbacks', async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new Error('fetch transport failed'))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        id: 'response-1',
        object: 'chat.completion',
        created: 1,
        model: 'gpt-4-turbo',
        choices: [{
          index: 0,
          message: { role: 'assistant', content: 'recipe' },
          finish_reason: 'stop',
        }],
        usage: { prompt_tokens: 10, completion_tokens: 4, total_tokens: 14 },
      }), { status: 200, headers: { 'content-type': 'application/json' } }));
    vi.stubGlobal('fetch', fetchMock);

    const attempts: ImageAnalysisAttemptEvent[] = [];
    const completed: ImageAnalysisCompleteEvent[] = [];
    const factory = new LLMProviderFactory({
      openai: { apiKey: 'test-key', maxRetries: 1, retryDelay: 0 },
      defaultProvider: 'openai',
      enableCircuitBreaker: false,
      costOptimization: false,
      hooks: {
        onImageAnalysisAttempt: event => attempts.push(event),
        onImageAnalysisComplete: event => completed.push(event),
      },
    });

    const response = await factory.analyzeImage({
      image: { data: 'AQID', mimeType: 'image/jpeg' },
      prompt: 'extract recipe',
      model: 'gpt-4-turbo',
      requestId: 'retry-capture',
    });

    expect(response.message).toBe('recipe');
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(attempts).toEqual([
      expect.objectContaining({
        attempt: 1,
        providerAttempt: 1,
        outcome: 'error',
        willRetry: true,
        costUsd: null,
        costProvenance: 'unknown',
      }),
      expect.objectContaining({
        attempt: 2,
        providerAttempt: 2,
        outcome: 'success',
        retry: true,
        inputTokens: 10,
        outputTokens: 4,
        totalTokens: 14,
        tokenProvenance: 'provider_reported',
        costProvenance: 'catalog_estimate',
      }),
    ]);
    expect(completed).toEqual([expect.objectContaining({
      outcome: 'success',
      attempts: 2,
      unknownCostAttempts: 1,
      finalProvider: 'openai',
      finalModel: 'gpt-4-turbo',
    })]);
  });
});
