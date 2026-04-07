/**
 * Tests for observability infrastructure:
 * - ObservabilityHooks + composeHooks
 * - ExhaustionRegistry
 * - LatencyHistogram
 * - CostTracker.record() simplified API
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { composeHooks, noopHooks } from '../utils/hooks';
import type { ObservabilityHooks, RequestStartEvent, FallbackEvent } from '../utils/hooks';
import { ExhaustionRegistry } from '../utils/exhaustion';
import { LatencyHistogram } from '../utils/latency-histogram';
import { CostTracker } from '../utils/cost-tracker';

// ── ObservabilityHooks ───────────────────────────────────────────────────

describe('ObservabilityHooks', () => {
  it('noopHooks is an empty object', () => {
    expect(noopHooks).toEqual({});
  });

  it('composeHooks calls all matching handlers', () => {
    const log1: string[] = [];
    const log2: string[] = [];

    const hooks1: ObservabilityHooks = {
      onRequestStart: (e) => log1.push(e.provider),
    };
    const hooks2: ObservabilityHooks = {
      onRequestStart: (e) => log2.push(e.provider),
    };

    const composed = composeHooks(hooks1, hooks2);
    composed.onRequestStart!({
      provider: 'groq',
      model: 'llama-3.3-70b',
      timestamp: Date.now(),
    });

    expect(log1).toEqual(['groq']);
    expect(log2).toEqual(['groq']);
  });

  it('composeHooks skips methods not implemented by any hook', () => {
    const hooks1: ObservabilityHooks = {
      onRequestStart: () => {},
    };
    const hooks2: ObservabilityHooks = {};

    const composed = composeHooks(hooks1, hooks2);

    // onFallback should not be defined since neither hook implements it
    expect(composed.onFallback).toBeUndefined();
    // onRequestStart should be defined
    expect(composed.onRequestStart).toBeDefined();
  });

  it('composeHooks isolates errors between handlers', () => {
    const log: string[] = [];

    const badHook: ObservabilityHooks = {
      onRequestStart: () => { throw new Error('boom'); },
    };
    const goodHook: ObservabilityHooks = {
      onRequestStart: (e) => log.push(e.provider),
    };

    const composed = composeHooks(badHook, goodHook);

    // Should not throw, and second handler should still fire
    expect(() => {
      composed.onRequestStart!({
        provider: 'anthropic',
        model: 'claude-haiku',
        timestamp: Date.now(),
      });
    }).not.toThrow();

    expect(log).toEqual(['anthropic']);
  });

  it('composeHooks with zero implementations returns empty hooks', () => {
    const composed = composeHooks({}, {});
    expect(composed.onRequestStart).toBeUndefined();
    expect(composed.onRequestEnd).toBeUndefined();
  });
});

// ── ExhaustionRegistry ───────────────────────────────────────────────────

describe('ExhaustionRegistry', () => {
  let registry: ExhaustionRegistry;

  beforeEach(() => {
    registry = new ExhaustionRegistry();
  });

  it('marks and checks provider exhaustion', () => {
    expect(registry.isExhausted('groq')).toBe(false);

    registry.markExhausted('groq');
    expect(registry.isExhausted('groq')).toBe(true);
  });

  it('clears exhaustion manually', () => {
    registry.markExhausted('groq');
    registry.clearExhaustion('groq');
    expect(registry.isExhausted('groq')).toBe(false);
  });

  it('auto-clears after resetAfterMs', () => {
    // Mark with 0ms reset — should clear immediately
    registry.markExhausted('groq', 0);
    expect(registry.isExhausted('groq')).toBe(false);
  });

  it('returns exhausted providers list', () => {
    registry.markExhausted('groq');
    registry.markExhausted('cerebras');
    expect(registry.getExhaustedProviders().sort()).toEqual(['cerebras', 'groq']);
  });

  it('getExhaustedProviders prunes expired entries', () => {
    registry.markExhausted('groq', 0);
    expect(registry.getExhaustedProviders()).toEqual([]);
  });

  it('getEntry returns entry data', () => {
    registry.markExhausted('groq', 60_000);
    const entry = registry.getEntry('groq');
    expect(entry).toBeDefined();
    expect(entry!.provider).toBe('groq');
    expect(entry!.resetAt).toBeGreaterThan(entry!.exhaustedAt);
  });

  it('getEntry returns undefined for non-exhausted provider', () => {
    expect(registry.getEntry('groq')).toBeUndefined();
  });

  it('reset clears all entries', () => {
    registry.markExhausted('groq');
    registry.markExhausted('cerebras');
    registry.reset();
    expect(registry.getExhaustedProviders()).toEqual([]);
  });
});

// ── LatencyHistogram ─────────────────────────────────────────────────────

describe('LatencyHistogram', () => {
  let histogram: LatencyHistogram;

  beforeEach(() => {
    histogram = new LatencyHistogram(100);
  });

  it('returns zero summary for unknown provider', () => {
    const summary = histogram.summary('unknown');
    expect(summary.count).toBe(0);
    expect(summary.p50).toBe(0);
    expect(summary.p95).toBe(0);
    expect(summary.p99).toBe(0);
  });

  it('records and computes percentiles', () => {
    // Record 100 values: 1, 2, 3, ..., 100
    for (let i = 1; i <= 100; i++) {
      histogram.record('groq', i);
    }

    const summary = histogram.summary('groq');
    expect(summary.count).toBe(100);
    expect(summary.min).toBe(1);
    expect(summary.max).toBe(100);
    expect(summary.mean).toBe(50.5);
    expect(summary.p50).toBe(50);
    expect(summary.p95).toBe(95);
    expect(summary.p99).toBe(99);
  });

  it('percentile returns correct value for single sample', () => {
    histogram.record('groq', 42);
    expect(histogram.percentile('groq', 50)).toBe(42);
    expect(histogram.percentile('groq', 99)).toBe(42);
  });

  it('evicts oldest samples when buffer is full', () => {
    const small = new LatencyHistogram(5);
    for (let i = 1; i <= 10; i++) {
      small.record('groq', i);
    }

    const summary = small.summary('groq');
    expect(summary.count).toBe(5);
    // Should have 6, 7, 8, 9, 10 (oldest evicted)
    expect(summary.min).toBe(6);
    expect(summary.max).toBe(10);
  });

  it('allSummaries returns all providers', () => {
    histogram.record('groq', 10);
    histogram.record('cerebras', 20);

    const all = histogram.allSummaries();
    expect(Object.keys(all).sort()).toEqual(['cerebras', 'groq']);
    expect(all['groq']!.count).toBe(1);
    expect(all['cerebras']!.count).toBe(1);
  });

  it('reset clears specific provider', () => {
    histogram.record('groq', 10);
    histogram.record('cerebras', 20);
    histogram.reset('groq');

    expect(histogram.summary('groq').count).toBe(0);
    expect(histogram.summary('cerebras').count).toBe(1);
  });

  it('reset with no arg clears all', () => {
    histogram.record('groq', 10);
    histogram.record('cerebras', 20);
    histogram.reset();

    expect(Object.keys(histogram.allSummaries())).toHaveLength(0);
  });
});

// ── CostTracker.record() ─────────────────────────────────────────────────

describe('CostTracker.record()', () => {
  it('records cost without requiring full LLMResponse', () => {
    const tracker = new CostTracker();
    tracker.record('groq', 0.005, 1000, 500);

    expect(tracker.getProviderCost('groq')).toBe(0.005);
    const breakdown = tracker.getCostBreakdown();
    expect(breakdown['groq']!.requests).toBe(1);
    expect(breakdown['groq']!.tokens.input).toBe(1000);
    expect(breakdown['groq']!.tokens.output).toBe(500);
  });

  it('record with defaults (zero tokens)', () => {
    const tracker = new CostTracker();
    tracker.record('cerebras', 0.001);

    expect(tracker.getProviderCost('cerebras')).toBe(0.001);
    const breakdown = tracker.getCostBreakdown();
    expect(breakdown['cerebras']!.tokens.input).toBe(0);
    expect(breakdown['cerebras']!.tokens.output).toBe(0);
  });

  it('trackCost delegates to record internally', () => {
    const tracker = new CostTracker();
    const spy = vi.spyOn(tracker, 'record');

    tracker.trackCost('groq', {
      message: 'hello',
      usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150, cost: 0.01 },
      model: 'llama-3.3-70b',
      provider: 'groq',
      responseTime: 200,
    });

    expect(spy).toHaveBeenCalledWith('groq', 0.01, 100, 50, 'llama-3.3-70b');
  });
});
