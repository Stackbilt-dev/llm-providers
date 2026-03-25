/**
 * Circuit Breaker
 * Prevents cascading failures by monitoring provider health
 */

import type { CircuitBreakerConfig, CircuitBreakerState } from '../types';
import { CircuitBreakerOpenError } from '../errors';

export class CircuitBreaker {
  private config: CircuitBreakerConfig;
  private state: CircuitBreakerState;
  private name: string;

  constructor(name: string, config: Partial<CircuitBreakerConfig> = {}) {
    this.name = name;
    this.config = {
      failureThreshold: config.failureThreshold ?? 5,
      resetTimeout: config.resetTimeout ?? 60000, // 1 minute
      monitoringPeriod: config.monitoringPeriod ?? 300000 // 5 minutes
    };

    this.state = {
      state: 'closed',
      failures: 0
    };
  }

  /**
   * Execute a function through the circuit breaker
   */
  async execute<T>(fn: () => Promise<T>): Promise<T> {
    this.checkState();

    if (this.state.state === 'open') {
      throw new CircuitBreakerOpenError(this.name);
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
   * Check and update circuit breaker state
   */
  private checkState(): void {
    const now = Date.now();

    // Reset failure count if monitoring period has passed
    if (this.state.lastFailure && 
        now - this.state.lastFailure > this.config.monitoringPeriod) {
      this.state.failures = 0;
    }

    // Check if we should transition from open to half-open
    if (this.state.state === 'open' && 
        this.state.nextAttempt && 
        now >= this.state.nextAttempt) {
      this.state.state = 'half-open';
      console.log(`[CircuitBreaker] ${this.name}: Transitioning to half-open state`);
    }
  }

  /**
   * Handle successful operation
   */
  private onSuccess(): void {
    if (this.state.state === 'half-open') {
      // Success in half-open state means we can close the circuit
      this.state.state = 'closed';
      this.state.failures = 0;
      this.state.lastFailure = undefined;
      this.state.nextAttempt = undefined;
      console.log(`[CircuitBreaker] ${this.name}: Circuit closed after successful recovery`);
    }
  }

  /**
   * Handle failed operation
   */
  private onFailure(): void {
    this.state.failures++;
    this.state.lastFailure = Date.now();

    if (this.state.state === 'half-open') {
      // Failure in half-open state means we go back to open
      this.state.state = 'open';
      this.state.nextAttempt = Date.now() + this.config.resetTimeout;
      console.log(`[CircuitBreaker] ${this.name}: Circuit re-opened after half-open failure`);
    } else if (this.state.failures >= this.config.failureThreshold) {
      // Too many failures in closed state means we open the circuit
      this.state.state = 'open';
      this.state.nextAttempt = Date.now() + this.config.resetTimeout;
      console.log(`[CircuitBreaker] ${this.name}: Circuit opened after ${this.state.failures} failures`);
    }
  }

  /**
   * Get current state
   */
  getState(): CircuitBreakerState {
    this.checkState(); // Update state before returning
    return { ...this.state };
  }

  /**
   * Get current configuration
   */
  getConfig(): CircuitBreakerConfig {
    return { ...this.config };
  }

  /**
   * Reset the circuit breaker
   */
  reset(): void {
    this.state = {
      state: 'closed',
      failures: 0
    };
    console.log(`[CircuitBreaker] ${this.name}: Circuit breaker reset`);
  }

  /**
   * Force open the circuit breaker
   */
  forceOpen(): void {
    this.state.state = 'open';
    this.state.nextAttempt = Date.now() + this.config.resetTimeout;
    console.log(`[CircuitBreaker] ${this.name}: Circuit breaker force opened`);
  }

  /**
   * Check if circuit breaker is open
   */
  isOpen(): boolean {
    this.checkState();
    return this.state.state === 'open';
  }

  /**
   * Check if circuit breaker is closed
   */
  isClosed(): boolean {
    this.checkState();
    return this.state.state === 'closed';
  }

  /**
   * Check if circuit breaker is half-open
   */
  isHalfOpen(): boolean {
    this.checkState();
    return this.state.state === 'half-open';
  }

  /**
   * Get health status
   */
  getHealth(): {
    state: string;
    failures: number;
    failureThreshold: number;
    healthy: boolean;
    nextAttempt?: number;
  } {
    this.checkState();
    
    return {
      state: this.state.state,
      failures: this.state.failures,
      failureThreshold: this.config.failureThreshold,
      healthy: this.state.state === 'closed',
      nextAttempt: this.state.nextAttempt
    };
  }
}

/**
 * Circuit breaker manager for multiple providers
 */
export class CircuitBreakerManager {
  private breakers: Map<string, CircuitBreaker> = new Map();
  private defaultConfig: CircuitBreakerConfig;

  constructor(defaultConfig?: Partial<CircuitBreakerConfig>) {
    this.defaultConfig = {
      failureThreshold: defaultConfig?.failureThreshold ?? 5,
      resetTimeout: defaultConfig?.resetTimeout ?? 60000,
      monitoringPeriod: defaultConfig?.monitoringPeriod ?? 300000
    };
  }

  /**
   * Get or create circuit breaker for a provider
   */
  getBreaker(name: string, config?: Partial<CircuitBreakerConfig>): CircuitBreaker {
    if (!this.breakers.has(name)) {
      const breakerConfig = { ...this.defaultConfig, ...config };
      this.breakers.set(name, new CircuitBreaker(name, breakerConfig));
    }
    return this.breakers.get(name)!;
  }

  /**
   * Execute function through named circuit breaker
   */
  async execute<T>(
    name: string, 
    fn: () => Promise<T>,
    config?: Partial<CircuitBreakerConfig>
  ): Promise<T> {
    const breaker = this.getBreaker(name, config);
    return breaker.execute(fn);
  }

  /**
   * Get all circuit breaker states
   */
  getAllStates(): Record<string, CircuitBreakerState> {
    const states: Record<string, CircuitBreakerState> = {};
    for (const [name, breaker] of this.breakers) {
      states[name] = breaker.getState();
    }
    return states;
  }

  /**
   * Get health status for all circuit breakers
   */
  getHealthStatus(): Record<string, any> {
    const health: Record<string, any> = {};
    for (const [name, breaker] of this.breakers) {
      health[name] = breaker.getHealth();
    }
    return health;
  }

  /**
   * Reset all circuit breakers
   */
  resetAll(): void {
    for (const [name, breaker] of this.breakers) {
      breaker.reset();
    }
  }

  /**
   * Reset specific circuit breaker
   */
  reset(name: string): void {
    const breaker = this.breakers.get(name);
    if (breaker) {
      breaker.reset();
    }
  }
}

/**
 * Default circuit breaker manager instance
 */
export const defaultCircuitBreakerManager = new CircuitBreakerManager();