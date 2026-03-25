/**
 * Retry Utility
 * Exponential backoff retry logic for LLM provider requests
 */

import type { RetryConfig } from '../types';
import { LLMErrorFactory } from '../errors';

export class RetryManager {
  private config: RetryConfig;

  constructor(config: Partial<RetryConfig> = {}) {
    this.config = {
      maxRetries: config.maxRetries ?? 3,
      initialDelay: config.initialDelay ?? 1000,
      maxDelay: config.maxDelay ?? 30000,
      backoffMultiplier: config.backoffMultiplier ?? 2,
      retryableErrors: config.retryableErrors ?? [
        'NETWORK_ERROR',
        'TIMEOUT',
        'SERVER_ERROR',
        'RATE_LIMIT',
        'CIRCUIT_BREAKER_OPEN'
      ]
    };
  }

  /**
   * Execute a function with retry logic
   */
  async execute<T>(
    fn: () => Promise<T>,
    context: string = 'operation'
  ): Promise<T> {
    let lastError: Error;
    let attempt = 0;

    while (attempt <= this.config.maxRetries) {
      try {
        return await fn();
      } catch (error) {
        lastError = error as Error;
        attempt++;

        // Check if we should retry this error
        if (!this.shouldRetry(error as Error, attempt)) {
          throw error;
        }

        // Calculate delay for next attempt
        const delay = this.calculateDelay(attempt, error as Error);
        
        console.warn(
          `[RetryManager] ${context} failed (attempt ${attempt}/${this.config.maxRetries + 1}): ${lastError.message}. Retrying in ${delay}ms...`
        );

        await this.delay(delay);
      }
    }

    throw lastError!;
  }

  /**
   * Check if an error should be retried
   */
  private shouldRetry(error: Error, attempt: number): boolean {
    // Don't retry if we've exceeded max attempts
    if (attempt > this.config.maxRetries) {
      return false;
    }

    // Check if error is retryable
    if (!LLMErrorFactory.isRetryable(error)) {
      return false;
    }

    // Check if error code is in retryable list
    const errorCode = (error as any).code;
    if (errorCode && !this.config.retryableErrors.includes(errorCode)) {
      return false;
    }

    return true;
  }

  /**
   * Calculate delay before next retry attempt
   */
  private calculateDelay(attempt: number, error: Error): number {
    // Use error-specific delay if available
    const errorDelay = LLMErrorFactory.getRetryDelay(error);
    if (errorDelay > 0) {
      return Math.min(errorDelay * attempt, this.config.maxDelay);
    }

    // Standard exponential backoff
    const delay = this.config.initialDelay * Math.pow(this.config.backoffMultiplier, attempt - 1);
    
    // Add jitter to prevent thundering herd
    const jitter = Math.random() * 0.1 * delay;
    
    return Math.min(delay + jitter, this.config.maxDelay);
  }

  /**
   * Promise-based delay
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Update retry configuration
   */
  updateConfig(config: Partial<RetryConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): RetryConfig {
    return { ...this.config };
  }
}

/**
 * Default retry manager instance
 */
export const defaultRetryManager = new RetryManager();

/**
 * Retry decorator for async functions
 */
export function withRetry<T extends any[], R>(
  retryConfig?: Partial<RetryConfig>
) {
  return function (
    target: any,
    propertyKey: string,
    descriptor: TypedPropertyDescriptor<(...args: T) => Promise<R>>
  ) {
    const originalMethod = descriptor.value!;
    const retryManager = new RetryManager(retryConfig);

    descriptor.value = async function (...args: T): Promise<R> {
      return retryManager.execute(
        () => originalMethod.apply(this, args),
        `${target.constructor.name}.${propertyKey}`
      );
    };

    return descriptor;
  };
}

/**
 * Simple retry function for one-off operations
 */
export async function retry<T>(
  fn: () => Promise<T>,
  config?: Partial<RetryConfig>
): Promise<T> {
  const retryManager = new RetryManager(config);
  return retryManager.execute(fn);
}