/**
 * Observability Hooks
 *
 * Structured lifecycle events for LLM provider operations.
 * Callers inject an ObservabilityHooks implementation to receive
 * typed events at every interesting moment in the request lifecycle.
 *
 * This replaces string-based logger calls for telemetry. The Logger
 * interface stays for debug output; hooks are for structured observability.
 */

import type { CircuitBreakerState, ProviderMetrics, TokenUsage } from '../types';

// ── Event types ──────────────────────────────────────────────────────────

export interface RequestStartEvent {
  provider: string;
  model: string;
  requestId?: string;
  tenantId?: string;
  timestamp: number;
}

export interface RequestEndEvent {
  provider: string;
  model: string;
  requestId?: string;
  tenantId?: string;
  durationMs: number;
  usage: TokenUsage;
  finishReason?: string;
  timestamp: number;
}

export interface RequestErrorEvent {
  provider: string;
  model: string;
  requestId?: string;
  tenantId?: string;
  error: Error;
  errorCode?: string;
  attempt: number;
  willRetry: boolean;
  timestamp: number;
}

export interface RetryEvent {
  provider: string;
  requestId?: string;
  attempt: number;
  maxAttempts: number;
  delayMs: number;
  error: Error;
  timestamp: number;
}

export interface FallbackEvent {
  fromProvider: string;
  toProvider: string;
  requestId?: string;
  reason: string;
  errorCode?: string;
  timestamp: number;
}

export interface CircuitStateChangeEvent {
  provider: string;
  fromState: CircuitBreakerState['state'];
  toState: CircuitBreakerState['state'];
  consecutiveFailures: number;
  trafficPct: number;
  timestamp: number;
}

export interface QuotaExhaustedEvent {
  provider: string;
  resetAfterMs: number;
  timestamp: number;
}

export interface BudgetThresholdEvent {
  provider: string;
  tier: 'warning' | 'critical' | 'emergency';
  utilizationPct: number;
  spend: number;
  budget: number;
  timestamp: number;
}

// ── Hooks interface ──────────────────────────────────────────────────────

export interface ObservabilityHooks {
  onRequestStart?(event: RequestStartEvent): void;
  onRequestEnd?(event: RequestEndEvent): void;
  onRequestError?(event: RequestErrorEvent): void;
  onRetry?(event: RetryEvent): void;
  onFallback?(event: FallbackEvent): void;
  onCircuitStateChange?(event: CircuitStateChangeEvent): void;
  onQuotaExhausted?(event: QuotaExhaustedEvent): void;
  onBudgetThreshold?(event: BudgetThresholdEvent): void;
}

/** Silent hooks — default. */
export const noopHooks: ObservabilityHooks = {};

/**
 * Merge multiple hook implementations. Each matching handler is called
 * in order. Errors in one handler don't block others.
 */
export function composeHooks(...implementations: ObservabilityHooks[]): ObservabilityHooks {
  const methods = [
    'onRequestStart', 'onRequestEnd', 'onRequestError',
    'onRetry', 'onFallback', 'onCircuitStateChange',
    'onQuotaExhausted', 'onBudgetThreshold',
  ] as const;

  const composed: ObservabilityHooks = {};

  for (const method of methods) {
    const handlers = implementations
      .map(impl => impl[method])
      .filter((fn): fn is NonNullable<typeof fn> => fn != null);

    if (handlers.length > 0) {
      (composed as Record<string, Function>)[method] = (event: unknown) => {
        for (const handler of handlers) {
          try {
            handler(event as never);
          } catch {
            // Observability must not break the request path.
          }
        }
      };
    }
  }

  return composed;
}
