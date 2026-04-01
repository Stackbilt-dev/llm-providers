/**
 * Graduated Circuit Breaker
 * Routes traffic away from failing providers with confidence-weighted degradation.
 */

import type { CircuitBreakerConfig, CircuitBreakerState } from '../types';
import type { Logger } from './logger';
import { noopLogger } from './logger';
import { CircuitBreakerOpenError } from '../errors';

const DEFAULT_DEGRADATION_CURVE = [1.0, 0.9, 0.7, 0.4, 0.1];

type ResolvedCircuitBreakerConfig = {
  failureThreshold: number;
  resetTimeout: number;
  monitoringPeriod: number;
  minRequests: number;
  degradationCurve: number[];
};

const DEFAULT_CONFIG: ResolvedCircuitBreakerConfig = {
  failureThreshold: DEFAULT_DEGRADATION_CURVE.length,
  resetTimeout: 60_000,
  monitoringPeriod: 300_000,
  minRequests: 3,
  degradationCurve: DEFAULT_DEGRADATION_CURVE
};

function clampTrafficPct(value: number): number {
  if (Number.isNaN(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function normalizeConfig(config: Partial<CircuitBreakerConfig>): ResolvedCircuitBreakerConfig {
  const degradationCurve = (config.degradationCurve?.length
    ? config.degradationCurve
    : DEFAULT_DEGRADATION_CURVE
  ).map(clampTrafficPct);

  return {
    failureThreshold: Math.max(1, config.failureThreshold ?? DEFAULT_CONFIG.failureThreshold),
    resetTimeout: config.resetTimeout ?? DEFAULT_CONFIG.resetTimeout,
    monitoringPeriod: config.monitoringPeriod ?? DEFAULT_CONFIG.monitoringPeriod,
    minRequests: config.minRequests ?? DEFAULT_CONFIG.minRequests,
    degradationCurve
  };
}

export class CircuitBreaker {
  private readonly config: ResolvedCircuitBreakerConfig;
  private readonly logger: Logger;
  private consecutiveFailures = 0;
  private totalFailures = 0;
  private totalSuccesses = 0;
  private totalRequests = 0;
  private lastFailureAt?: number;
  private lastSuccessAt?: number;
  private lastRequestAt?: number;
  private lastDecayAt = 0;
  private windowStart = Date.now();

  constructor(
    readonly name: string,
    config: Partial<CircuitBreakerConfig> = {},
    logger?: Logger
  ) {
    this.config = normalizeConfig(config);
    this.logger = logger ?? noopLogger;
  }

  /**
   * Backwards-compatible execute alias.
   */
  async execute<T>(fn: () => Promise<T>): Promise<T> {
    return this.exec(fn);
  }

  /**
   * Execute the primary operation if the breaker allows traffic through.
   * Rejected requests throw CircuitBreakerOpenError so callers can route elsewhere.
   */
  async exec<T>(fn: () => Promise<T>): Promise<T> {
    const now = Date.now();
    this.syncState(now);
    this.markTraffic(now);

    const primaryPct = this.currentPrimaryTrafficPct();
    if (primaryPct <= 0) {
      throw this.buildOpenError(now);
    }

    if (primaryPct < 1 && Math.random() > primaryPct) {
      this.logger.info(
        `[CircuitBreaker] ${this.name}: degraded (${(primaryPct * 100).toFixed(0)}% primary), rejecting request for fallback`
      );
      throw new CircuitBreakerOpenError(this.name, 0, this.consecutiveFailures);
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  /**
   * Execute with graduated fallback. Degraded traffic is routed to the fallback
   * probabilistically, and primary failures immediately fail over.
   */
  async execWithFallback<T>(
    primary: () => Promise<T>,
    fallback: () => Promise<T>
  ): Promise<T> {
    const now = Date.now();
    this.syncState(now);
    this.markTraffic(now);

    const primaryPct = this.currentPrimaryTrafficPct();
    if (primaryPct <= 0) {
      this.logger.info(
        `[CircuitBreaker] ${this.name}: fully degraded (${this.consecutiveFailures} failures), using fallback`
      );
      return fallback();
    }

    if (primaryPct < 1 && Math.random() > primaryPct) {
      this.logger.info(
        `[CircuitBreaker] ${this.name}: degraded (${(primaryPct * 100).toFixed(0)}% primary), routing to fallback`
      );
      return fallback();
    }

    try {
      const result = await primary();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      this.logger.warn(
        `[CircuitBreaker] ${this.name}: primary failed (${this.consecutiveFailures} consecutive), trying fallback`
      );
      return fallback();
    }
  }

  /**
   * Current percentage of traffic that should be routed to the primary provider.
   */
  primaryTrafficPct(): number {
    this.syncState(Date.now());
    return this.currentPrimaryTrafficPct();
  }

  /**
   * Get current state and breaker statistics.
   */
  getState(): CircuitBreakerState {
    this.syncState(Date.now());
    return this.snapshot();
  }

  /**
   * Alias for stats-oriented call sites.
   */
  getStats(): CircuitBreakerState {
    return this.getState();
  }

  /**
   * Get current configuration.
   */
  getConfig(): CircuitBreakerConfig {
    return {
      ...this.config,
      degradationCurve: [...this.config.degradationCurve]
    };
  }

  /**
   * Manual reset for testing and operator intervention.
   */
  reset(): void {
    this.consecutiveFailures = 0;
    this.totalFailures = 0;
    this.totalSuccesses = 0;
    this.totalRequests = 0;
    this.lastFailureAt = undefined;
    this.lastSuccessAt = undefined;
    this.lastRequestAt = undefined;
    this.lastDecayAt = 0;
    this.windowStart = Date.now();
    this.logger.info(`[CircuitBreaker] ${this.name}: circuit breaker reset`);
  }

  /**
   * Force the breaker into the fully open state.
   */
  forceOpen(): void {
    const now = Date.now();
    this.consecutiveFailures = this.config.failureThreshold;
    this.lastFailureAt = now;
    this.lastRequestAt = now;
    this.lastDecayAt = now;
    this.logger.info(`[CircuitBreaker] ${this.name}: circuit breaker force opened`);
  }

  isOpen(): boolean {
    return this.getState().state === 'OPEN';
  }

  isClosed(): boolean {
    return this.getState().state === 'CLOSED';
  }

  /**
   * Deprecated alias retained for compatibility with the prior state machine.
   */
  isHalfOpen(): boolean {
    return this.isRecovering();
  }

  isDegraded(): boolean {
    return this.getState().state === 'DEGRADED';
  }

  isRecovering(): boolean {
    return this.getState().state === 'RECOVERING';
  }

  getHealth(): {
    state: CircuitBreakerState['state'];
    failures: number;
    consecutiveFailures: number;
    failureThreshold: number;
    primaryTrafficPct: number;
    healthy: boolean;
    lastFailure?: number;
    lastSuccess?: number;
    nextAttempt?: number;
    totalFailures: number;
    totalSuccesses: number;
    totalRequests: number;
  } {
    const state = this.getState();

    return {
      state: state.state,
      failures: state.failures,
      consecutiveFailures: state.consecutiveFailures,
      failureThreshold: this.config.failureThreshold,
      primaryTrafficPct: state.primaryTrafficPct,
      healthy: state.state === 'CLOSED',
      lastFailure: state.lastFailure,
      lastSuccess: state.lastSuccess,
      nextAttempt: state.nextAttempt,
      totalFailures: state.totalFailures,
      totalSuccesses: state.totalSuccesses,
      totalRequests: state.totalRequests
    };
  }

  private snapshot(now: number = Date.now()): CircuitBreakerState {
    return {
      state: this.currentState(),
      failures: this.consecutiveFailures,
      consecutiveFailures: this.consecutiveFailures,
      primaryTrafficPct: this.currentPrimaryTrafficPct(),
      totalFailures: this.totalFailures,
      totalSuccesses: this.totalSuccesses,
      totalRequests: this.totalRequests,
      lastFailure: this.lastFailureAt,
      lastSuccess: this.lastSuccessAt,
      lastRequest: this.lastRequestAt,
      nextAttempt: this.retryAt(now)
    };
  }

  private currentState(): CircuitBreakerState['state'] {
    const primaryPct = this.currentPrimaryTrafficPct();
    if (primaryPct >= 1) return 'CLOSED';
    if (primaryPct <= 0) return 'OPEN';
    if ((this.lastSuccessAt ?? 0) > (this.lastFailureAt ?? 0)) return 'RECOVERING';
    return 'DEGRADED';
  }

  private currentPrimaryTrafficPct(): number {
    if (this.consecutiveFailures >= this.config.failureThreshold) {
      return 0;
    }

    if (this.consecutiveFailures < this.config.degradationCurve.length) {
      return this.config.degradationCurve[this.consecutiveFailures];
    }

    if (this.config.degradationCurve.length === 0) {
      return this.consecutiveFailures === 0 ? 1 : 0;
    }

    return this.config.degradationCurve[this.config.degradationCurve.length - 1];
  }

  private buildOpenError(now: number): CircuitBreakerOpenError {
    return new CircuitBreakerOpenError(
      this.name,
      this.retryAfterSec(now),
      this.consecutiveFailures
    );
  }

  private retryAfterSec(now: number): number {
    if (this.consecutiveFailures === 0 || this.lastDecayAt === 0) {
      return 0;
    }

    return Math.max(
      0,
      Math.ceil((this.config.resetTimeout - (now - this.lastDecayAt)) / 1000)
    );
  }

  private retryAt(now: number): number | undefined {
    if (this.consecutiveFailures === 0 || this.lastDecayAt === 0) {
      return undefined;
    }
    return Math.max(now, this.lastDecayAt + this.config.resetTimeout);
  }

  private syncState(now: number): void {
    this.maybeDecayFailures(now);
    this.maybeRotateWindow(now);
  }

  private markTraffic(now: number): void {
    this.lastRequestAt = now;
    this.lastDecayAt = now;
  }

  private onSuccess(): void {
    this.totalRequests++;
    this.totalSuccesses++;
    this.lastSuccessAt = Date.now();

    if (this.consecutiveFailures > 0) {
      this.consecutiveFailures--;
      this.logger.info(
        `[CircuitBreaker] ${this.name}: success -> recovering (${this.consecutiveFailures} failures, ${(this.currentPrimaryTrafficPct() * 100).toFixed(0)}% primary)`
      );
    }
  }

  private onFailure(): void {
    this.totalRequests++;
    this.totalFailures++;
    this.consecutiveFailures++;
    this.lastFailureAt = Date.now();

    const primaryPct = this.currentPrimaryTrafficPct();
    if (primaryPct <= 0) {
      this.logger.warn(
        `[CircuitBreaker] ${this.name}: OPEN (${this.consecutiveFailures} consecutive failures)`
      );
      return;
    }

    if (primaryPct < 1) {
      this.logger.info(
        `[CircuitBreaker] ${this.name}: degraded -> ${(primaryPct * 100).toFixed(0)}% primary (${this.consecutiveFailures} failures)`
      );
    }
  }

  private maybeDecayFailures(now: number): void {
    if (this.consecutiveFailures === 0 || this.lastDecayAt === 0) {
      return;
    }

    const elapsed = now - this.lastDecayAt;
    if (elapsed < this.config.resetTimeout) {
      return;
    }

    const steps = Math.floor(elapsed / this.config.resetTimeout);
    const before = this.consecutiveFailures;
    this.consecutiveFailures = Math.max(0, this.consecutiveFailures - steps);
    this.lastDecayAt += steps * this.config.resetTimeout;

    if (before !== this.consecutiveFailures) {
      this.logger.info(
        `[CircuitBreaker] ${this.name}: time-decay ${before} -> ${this.consecutiveFailures} failures (${(this.currentPrimaryTrafficPct() * 100).toFixed(0)}% primary)`
      );
    }
  }

  private maybeRotateWindow(now: number): void {
    if (now - this.windowStart >= this.config.monitoringPeriod) {
      this.windowStart = now;
    }
  }
}

/**
 * Circuit breaker manager for multiple providers.
 */
export class CircuitBreakerManager {
  private breakers: Map<string, CircuitBreaker> = new Map();
  private defaultConfig: Partial<CircuitBreakerConfig>;
  private logger: Logger;

  constructor(defaultConfig?: Partial<CircuitBreakerConfig>, logger?: Logger) {
    this.logger = logger ?? noopLogger;
    this.defaultConfig = {
      failureThreshold: defaultConfig?.failureThreshold ?? DEFAULT_CONFIG.failureThreshold,
      resetTimeout: defaultConfig?.resetTimeout ?? DEFAULT_CONFIG.resetTimeout,
      monitoringPeriod: defaultConfig?.monitoringPeriod ?? DEFAULT_CONFIG.monitoringPeriod,
      minRequests: defaultConfig?.minRequests ?? DEFAULT_CONFIG.minRequests,
      degradationCurve: defaultConfig?.degradationCurve
        ? [...defaultConfig.degradationCurve]
        : [...DEFAULT_DEGRADATION_CURVE]
    };
  }

  getBreaker(name: string, config?: Partial<CircuitBreakerConfig>): CircuitBreaker {
    if (!this.breakers.has(name)) {
      const breakerConfig: Partial<CircuitBreakerConfig> = {
        ...this.defaultConfig,
        ...config,
        degradationCurve: config?.degradationCurve
          ? [...config.degradationCurve]
          : [...(this.defaultConfig.degradationCurve ?? DEFAULT_DEGRADATION_CURVE)]
      };
      this.breakers.set(name, new CircuitBreaker(name, breakerConfig, this.logger));
    }

    return this.breakers.get(name)!;
  }

  async execute<T>(
    name: string,
    fn: () => Promise<T>,
    config?: Partial<CircuitBreakerConfig>
  ): Promise<T> {
    return this.getBreaker(name, config).execute(fn);
  }

  async execWithFallback<T>(
    name: string,
    primary: () => Promise<T>,
    fallback: () => Promise<T>,
    config?: Partial<CircuitBreakerConfig>
  ): Promise<T> {
    return this.getBreaker(name, config).execWithFallback(primary, fallback);
  }

  getAllStates(): Record<string, CircuitBreakerState> {
    const states: Record<string, CircuitBreakerState> = {};

    for (const [name, breaker] of this.breakers) {
      states[name] = breaker.getState();
    }

    return states;
  }

  getHealthStatus(): Record<string, ReturnType<CircuitBreaker['getHealth']>> {
    const health: Record<string, ReturnType<CircuitBreaker['getHealth']>> = {};

    for (const [name, breaker] of this.breakers) {
      health[name] = breaker.getHealth();
    }

    return health;
  }

  resetAll(): void {
    for (const breaker of this.breakers.values()) {
      breaker.reset();
    }
  }

  reset(name: string): void {
    this.breakers.get(name)?.reset();
  }
}

/**
 * Default circuit breaker manager instance.
 */
export const defaultCircuitBreakerManager = new CircuitBreakerManager();
