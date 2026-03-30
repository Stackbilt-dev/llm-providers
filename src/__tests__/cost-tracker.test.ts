import { describe, it, expect, beforeEach } from 'vitest';
import { CostTracker } from '../utils/cost-tracker';
import type { LLMResponse } from '../types';

function createResponse(
  provider: string,
  usage: LLMResponse['usage']
): LLMResponse {
  return {
    message: `${provider} response`,
    usage,
    model: 'test-model',
    provider,
    responseTime: 100
  };
}

describe('CostTracker', () => {
  let tracker: CostTracker;

  beforeEach(() => {
    tracker = new CostTracker();
  });

  it('tracks per-provider cost and token totals', () => {
    tracker.trackCost('openai', createResponse('openai', {
      inputTokens: 10,
      outputTokens: 20,
      totalTokens: 30,
      cost: 0.001
    }));
    tracker.trackCost('openai', createResponse('openai', {
      inputTokens: 5,
      outputTokens: 15,
      totalTokens: 20,
      cost: 0.0005
    }));
    tracker.trackCost('anthropic', createResponse('anthropic', {
      inputTokens: 4,
      outputTokens: 6,
      totalTokens: 10,
      cost: 0.002
    }));

    expect(tracker.total()).toBeCloseTo(0.0035);
    expect(tracker.getTotalCost()).toBeCloseTo(0.0035);
    expect(tracker.getProviderCost('openai')).toBeCloseTo(0.0015);

    expect(tracker.breakdown().openai).toMatchObject({
      totalCost: 0.0015,
      requestCount: 2,
      inputTokens: 15,
      outputTokens: 35
    });
    expect(tracker.breakdown().openai.lastRecordedAt).toBeGreaterThan(0);

    expect(tracker.getCostBreakdown().openai).toMatchObject({
      cost: 0.0015,
      totalCost: 0.0015,
      requests: 2,
      requestCount: 2,
      inputTokens: 15,
      outputTokens: 35,
      tokens: { input: 15, output: 35 },
      averageCostPerRequest: 0.00075
    });
  });

  it('drains tracked provider costs and resets counters', () => {
    tracker.trackCost('cloudflare', createResponse('cloudflare', {
      inputTokens: 12,
      outputTokens: 8,
      totalTokens: 20,
      cost: 0.0002
    }));

    const drained = tracker.drain();

    expect(drained.cloudflare).toMatchObject({
      totalCost: 0.0002,
      requestCount: 1,
      inputTokens: 12,
      outputTokens: 8
    });
    expect(drained.cloudflare.lastRecordedAt).toBeGreaterThan(0);
    expect(tracker.total()).toBe(0);
    expect(tracker.getTotalCost()).toBe(0);
    expect(tracker.breakdown()).toEqual({});
    expect(tracker.getCostBreakdown()).toEqual({});
  });
});
