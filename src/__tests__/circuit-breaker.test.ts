import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CircuitBreaker } from '../utils/circuit-breaker';
import { CircuitBreakerOpenError } from '../errors';

describe('CircuitBreaker', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-01-01T00:00:00.000Z'));
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it('degrades with failures, opens at the threshold, and recovers one step per success', async () => {
    const randomSpy = vi.spyOn(Math, 'random').mockReturnValue(0);
    const breaker = new CircuitBreaker('openai', {
      failureThreshold: 3,
      degradationCurve: [1, 0.6, 0.2]
    });

    const fail = async () => {
      throw new Error('boom');
    };

    await expect(breaker.execute(fail)).rejects.toThrow('boom');
    await expect(breaker.execute(fail)).rejects.toThrow('boom');

    expect(breaker.getState()).toMatchObject({
      state: 'DEGRADED',
      failures: 2,
      consecutiveFailures: 2,
      primaryTrafficPct: 0.2,
      totalFailures: 2,
      totalRequests: 2
    });

    await expect(breaker.execute(fail)).rejects.toThrow('boom');

    const openState = breaker.getState();
    expect(openState).toMatchObject({
      state: 'OPEN',
      failures: 3,
      consecutiveFailures: 3,
      primaryTrafficPct: 0
    });

    await expect(breaker.execute(async () => 'nope')).rejects.toBeInstanceOf(CircuitBreakerOpenError);

    vi.setSystemTime(new Date('2026-01-01T00:01:00.000Z'));
    await expect(breaker.execute(async () => 'ok')).resolves.toBe('ok');

    const recoveringState = breaker.getState();
    expect(recoveringState).toMatchObject({
      state: 'RECOVERING',
      failures: 1,
      consecutiveFailures: 1,
      primaryTrafficPct: 0.6,
      totalSuccesses: 1,
      totalRequests: 4
    });

    await expect(breaker.execute(async () => 'ok')).resolves.toBe('ok');
    expect(breaker.getState().state).toBe('CLOSED');

    randomSpy.mockRestore();
  });

  it('rejects primary traffic probabilistically while degraded', async () => {
    const randomSpy = vi.spyOn(Math, 'random').mockReturnValue(0);
    const breaker = new CircuitBreaker('openai');

    await expect(
      breaker.execute(async () => {
        throw new Error('boom');
      })
    ).rejects.toThrow('boom');

    const primary = vi.fn().mockResolvedValue('primary');
    randomSpy.mockReturnValue(0.95);

    await expect(breaker.execute(primary)).rejects.toBeInstanceOf(CircuitBreakerOpenError);

    expect(primary).not.toHaveBeenCalled();
    expect(breaker.getState()).toMatchObject({
      state: 'DEGRADED',
      failures: 1,
      totalFailures: 1,
      totalRequests: 1
    });
  });

  it('execWithFallback uses fallback on primary failure and on degraded routing', async () => {
    const randomSpy = vi.spyOn(Math, 'random').mockReturnValue(0);
    const breaker = new CircuitBreaker('openai');
    const fallback = vi.fn().mockResolvedValue('fallback');
    const primaryFailure = vi.fn().mockRejectedValue(new Error('primary down'));

    await expect(breaker.execWithFallback(primaryFailure, fallback)).resolves.toBe('fallback');

    expect(primaryFailure).toHaveBeenCalledTimes(1);
    expect(fallback).toHaveBeenCalledTimes(1);
    expect(breaker.getState()).toMatchObject({
      state: 'DEGRADED',
      failures: 1,
      totalFailures: 1,
      totalRequests: 1
    });

    randomSpy.mockReturnValue(0.95);
    const primarySuccess = vi.fn().mockResolvedValue('primary');

    await expect(breaker.execWithFallback(primarySuccess, fallback)).resolves.toBe('fallback');

    expect(primarySuccess).not.toHaveBeenCalled();
    expect(fallback).toHaveBeenCalledTimes(2);
    expect(breaker.getState()).toMatchObject({
      state: 'DEGRADED',
      failures: 1,
      totalFailures: 1,
      totalRequests: 1
    });
  });

  it('decays failures after idle intervals without double-counting repeated reads', async () => {
    vi.spyOn(Math, 'random').mockReturnValue(0);
    const breaker = new CircuitBreaker('openai', {
      resetTimeout: 1_000
    });

    await expect(
      breaker.execute(async () => {
        throw new Error('boom');
      })
    ).rejects.toThrow('boom');

    await expect(
      breaker.execute(async () => {
        throw new Error('boom');
      })
    ).rejects.toThrow('boom');

    expect(breaker.getState().failures).toBe(2);

    vi.setSystemTime(new Date('2026-01-01T00:00:01.000Z'));
    expect(breaker.getState()).toMatchObject({
      state: 'DEGRADED',
      failures: 1,
      primaryTrafficPct: 0.9
    });
    expect(breaker.getState().failures).toBe(1);

    vi.setSystemTime(new Date('2026-01-01T00:00:02.000Z'));
    expect(breaker.getState()).toMatchObject({
      state: 'CLOSED',
      failures: 0,
      primaryTrafficPct: 1
    });
  });
});
